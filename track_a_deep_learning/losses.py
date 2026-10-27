"""
losses.py - Multi-Objective Neuroengineering Losses for Track A Deep Learning Denoising.

Combines:
  1. Time-Domain Loss (L1 or MSE)
  2. Multi-Resolution STFT Spectral Loss (penalizes spectral band distortion)
  3. Temporal Pearson Correlation Loss (penalizes phase misalignment & peak flattening)
  4. Auditory ERP Window Loss (heavily weights Cz/Fz in the 50-350ms post-stimulus window)
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution Short-Time Fourier Transform (STFT) spectral loss.
    Computes both linear spectral convergence and log-magnitude spectral distance
    across multiple FFT window resolutions to penalize clinical band distortion.
    """
    def __init__(self, fft_sizes: Tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.windows: Dict[int, torch.Tensor] = {}

    def get_window(self, n_fft: int, device: torch.device) -> torch.Tensor:
        key = (n_fft, str(device))
        if key not in self.windows or self.windows[key].device != device:
            self.windows[key] = torch.hann_window(n_fft, device=device)
        return self.windows[key]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target shape: (B, C, T)
        B, C, T = pred.shape
        p_flat = pred.reshape(B * C, T)
        t_flat = target.reshape(B * C, T)
        total_loss = pred.new_zeros(1)

        for n_fft in self.fft_sizes:
            hop = max(1, n_fft // 4)
            win = self.get_window(n_fft, pred.device)
            p_stft = torch.stft(
                p_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, return_complex=True
            ).abs()
            t_stft = torch.stft(
                t_flat, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=win, return_complex=True
            ).abs()

            # Spectral convergence
            sc = torch.norm(p_stft - t_stft, p="fro") / (torch.norm(t_stft, p="fro") + 1e-8)
            # Log magnitude distance
            log_mag = F.l1_loss(torch.log(p_stft + 1e-8), torch.log(t_stft + 1e-8))
            total_loss = total_loss + sc + log_mag

        return total_loss / len(self.fft_sizes)


class PearsonCorrelationLoss(nn.Module):
    """
    Temporal Pearson correlation loss (1 - r) along the time axis.
    Directly maximizes waveform phase alignment and component morphology.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target shape: (B, C, T)
        p_sub = pred - pred.mean(dim=-1, keepdim=True)
        t_sub = target - target.mean(dim=-1, keepdim=True)

        num = (p_sub * t_sub).sum(dim=-1)
        den = torch.sqrt((p_sub ** 2).sum(dim=-1) * (t_sub ** 2).sum(dim=-1)) + self.eps
        r = num / den
        return 1.0 - r.mean()


class ERPAuditoryWindowLoss(nn.Module):
    """
    Penalizes reconstruction error specifically in central auditory electrodes
    (Cz=ch 6, Fz=ch 1) during the post-stimulus interval (50ms - 350ms).
    """
    def __init__(
        self,
        sfreq: float = 250.0,
        tmin: float = -0.2,
        window: Tuple[float, float] = (0.05, 0.35),
        channel_indices: Tuple[int, ...] = (1, 6),
    ):
        super().__init__()
        self.sfreq = sfreq
        self.tmin = tmin
        self.window = window
        self.channel_indices = channel_indices

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s_idx = max(0, int((self.window[0] - self.tmin) * self.sfreq))
        e_idx = min(pred.shape[-1], int((self.window[1] - self.tmin) * self.sfreq))

        if s_idx >= e_idx or s_idx >= pred.shape[-1]:
            return F.l1_loss(pred, target)

        valid_chs = [ch for ch in self.channel_indices if ch < pred.shape[1]]
        if not valid_chs:
            valid_chs = slice(None)

        pred_win = pred[:, valid_chs, s_idx:e_idx]
        targ_win = target[:, valid_chs, s_idx:e_idx]
        return F.l1_loss(pred_win, targ_win)


class CompositeNeuroLoss(nn.Module):
    """
    Composite neuroengineering loss combining time-domain, spectral,
    correlation, and ERP-window objectives:
      L_total = L_time + lambda_spec * L_spec + lambda_corr * L_corr + lambda_erp * L_erp
    """
    def __init__(
        self,
        time_loss: str = "l1",
        lambda_spec: float = 0.5,
        lambda_corr: float = 0.2,
        lambda_erp: float = 0.3,
        sfreq: float = 250.0,
    ):
        super().__init__()
        self.time_loss = time_loss
        self.lambda_spec = lambda_spec
        self.lambda_corr = lambda_corr
        self.lambda_erp = lambda_erp

        self.spec_fn = MultiResolutionSTFTLoss()
        self.corr_fn = PearsonCorrelationLoss()
        self.erp_fn = ERPAuditoryWindowLoss(sfreq=sfreq)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.time_loss == "mse":
            l_time = F.mse_loss(pred, target)
        else:
            l_time = F.l1_loss(pred, target)

        l_spec = self.spec_fn(pred, target) if self.lambda_spec > 0 else pred.new_zeros(1)
        l_corr = self.corr_fn(pred, target) if self.lambda_corr > 0 else pred.new_zeros(1)
        l_erp = self.erp_fn(pred, target) if self.lambda_erp > 0 else pred.new_zeros(1)

        total = l_time + self.lambda_spec * l_spec + self.lambda_corr * l_corr + self.lambda_erp * l_erp
        metrics = {
            "loss": float(total.item()),
            "time": float(l_time.item()),
            "spec": float(l_spec.item()),
            "corr": float(l_corr.item()),
            "erp": float(l_erp.item()),
        }
        return total, metrics
