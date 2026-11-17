"""
models_suite.py - Multi-Architecture Suite for CI-EEG Denoising Comparison.

Architectures:
1. UNet1D_Baseline       : Direct-prediction 1D U-Net (baseline reference).
2. ResAttnUNet1D         : Residual noise-predicting 1D U-Net with GroupNorm + Spatial Channel Attention.
3. STFT_MaskNet          : Complex Time-Frequency 2D U-Net operating on STFT (Real+Imag) with iSTFT residual subtraction.
4. SpatioTemporalConformer: Learnable 1x1 spatial unmixing + Multi-Head Self-Attention + Depthwise Conv residual denoiser.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline 1D U-Net (Direct Prediction)
# ─────────────────────────────────────────────────────────────────────────────

def _double_conv_base(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
        nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
    )


class UNet1D_Baseline(nn.Module):
    def __init__(self, in_channels: int = 19, out_channels: int = 19):
        super().__init__()
        self.max_pool = nn.MaxPool1d(2, 2)
        self.down1 = _double_conv_base(in_channels, 64)
        self.down2 = _double_conv_base(64, 128)
        self.down3 = _double_conv_base(128, 256)
        self.down4 = _double_conv_base(256, 512)
        self.bottleneck = _double_conv_base(512, 1024)

        self.up1 = nn.ConvTranspose1d(1024, 512, 2, 2)
        self.up_conv1 = _double_conv_base(1024, 512)
        self.up2 = nn.ConvTranspose1d(512, 256, 2, 2)
        self.up_conv2 = _double_conv_base(512, 256)
        self.up3 = nn.ConvTranspose1d(256, 128, 2, 2)
        self.up_conv3 = _double_conv_base(256, 128)
        self.up4 = nn.ConvTranspose1d(128, 64, 2, 2)
        self.up_conv4 = _double_conv_base(128, 64)
        self.out = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x2 = self.down2(self.max_pool(x1))
        x3 = self.down3(self.max_pool(x2))
        x4 = self.down4(self.max_pool(x3))
        x5 = self.bottleneck(self.max_pool(x4))

        u1 = self.up1(x5)
        u1 = self.up_conv1(torch.cat([x4[..., :u1.shape[-1]], u1], dim=1))
        u2 = self.up2(u1)
        u2 = self.up_conv2(torch.cat([x3[..., :u2.shape[-1]], u2], dim=1))
        u3 = self.up3(u2)
        u3 = self.up_conv3(torch.cat([x2[..., :u3.shape[-1]], u3], dim=1))
        u4 = self.up4(u3)
        u4 = self.up_conv4(torch.cat([x1[..., :u4.shape[-1]], u4], dim=1))
        out = self.out(u4)
        return out[..., :x.shape[-1]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Residual Attention 1D U-Net (Noise-Residual Subtraction + SE Attention)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelSE1D(nn.Module):
    """Squeeze-and-Excitation spatial/feature channel attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1)
        return x * w


class ResAttnBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, dilation: int = 1):
        super().__init__()
        groups = min(8, out_c)
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=5, padding=2 * dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(groups, out_c)
        self.se = ChannelSE1D(out_c)
        self.skip = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        return F.silu(h + self.skip(x))


class ResAttnUNet1D(nn.Module):
    """
    Predicts CI artifact noise residual n_hat = f(x) and returns clean = x - n_hat.
    """
    def __init__(self, in_channels: int = 19, out_channels: int = 19, base_c: int = 48):
        super().__init__()
        self.spatial_proj = nn.Conv1d(in_channels, base_c, kernel_size=1)
        self.enc1 = ResAttnBlock1D(base_c, base_c)
        self.enc2 = ResAttnBlock1D(base_c, base_c * 2, dilation=2)
        self.enc3 = ResAttnBlock1D(base_c * 2, base_c * 4, dilation=2)
        self.enc4 = ResAttnBlock1D(base_c * 4, base_c * 8, dilation=4)

        self.pool = nn.MaxPool1d(2, 2)
        self.dec3 = ResAttnBlock1D(base_c * 8 + base_c * 4, base_c * 4)
        self.dec2 = ResAttnBlock1D(base_c * 4 + base_c * 2, base_c * 2)
        self.dec1 = ResAttnBlock1D(base_c * 2 + base_c, base_c)
        self.head = nn.Conv1d(base_c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(self.spatial_proj(x))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.enc4(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[-1], mode="linear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        noise_hat = self.head(d1)
        return x - noise_hat[..., :x.shape[-1]]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Complex Time-Frequency STFT Mask/Residual Network
# ─────────────────────────────────────────────────────────────────────────────

class Conv2dBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_c), out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class STFT_MaskNet(nn.Module):
    """
    Operates on multi-channel complex STFT spectrograms (Real + Imag stacked on channel dim).
    Predicts complex artifact residual in time-frequency space and reconstructs via iSTFT.
    """
    def __init__(self, in_channels: int = 19, n_fft: int = 256, hop_length: int = 64, base_c: int = 48):
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        c_in = in_channels * 2  # Real + Imag
        self.enc1 = Conv2dBlock(c_in, base_c)
        self.enc2 = Conv2dBlock(base_c, base_c * 2)
        self.enc3 = Conv2dBlock(base_c * 2, base_c * 4)
        self.pool = nn.MaxPool2d(2, 2)

        self.dec2 = Conv2dBlock(base_c * 4 + base_c * 2, base_c * 2)
        self.dec1 = Conv2dBlock(base_c * 2 + base_c, base_c)
        self.out_conv = nn.Conv2d(base_c, c_in, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
        stft = torch.stft(
            x_flat, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window, return_complex=True
        )  # (B*C, F, Tf)
        _, F_bins, T_frames = stft.shape
        stft_ri = torch.view_as_real(stft)  # (B*C, F, Tf, 2)
        stft_ri = stft_ri.reshape(B, C, F_bins, T_frames, 2).permute(0, 1, 4, 2, 3).reshape(B, 2 * C, F_bins, T_frames)

        e1 = self.enc1(stft_ri)
        e2 = self.enc2(self.pool(e1))
        b = self.enc3(self.pool(e2))

        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        noise_ri = self.out_conv(d1)  # (B, 2*C, F, Tf)
        clean_ri = stft_ri - noise_ri
        clean_ri = clean_ri.reshape(B, C, 2, F_bins, T_frames).permute(0, 1, 3, 4, 2).contiguous()
        clean_complex = torch.view_as_complex(clean_ri.reshape(B * C, F_bins, T_frames, 2))

        clean_time = torch.istft(
            clean_complex, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=self.window, length=T
        )
        return clean_time.reshape(B, C, T)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Spatio-Temporal Conformer (Learnable Spatial Unmixing + Self-Attention)
# ─────────────────────────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int = 96, nhead: int = 4, kernel_size: int = 15):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.conv_mod = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model),
            nn.GroupNorm(min(8, d_model), d_model),
            nn.SiLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h2 = self.norm2(x).transpose(1, 2)  # (B, D, T)
        x = x + self.conv_mod(h2).transpose(1, 2)

        x = x + 0.5 * self.ffn(self.norm3(x))
        return x


class SpatioTemporalConformer(nn.Module):
    """
    Learnable spatial unmixing (1x1 conv) -> downsampled 4x temporal Conformer -> upsampled residual subtraction.
    """
    def __init__(self, in_channels: int = 19, d_model: int = 96, n_layers: int = 3):
        super().__init__()
        self.spatial_unmix = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
            nn.SiLU(inplace=True),
        )
        # 4x temporal stride for fast global attention over 4s window (1024 -> 256 tokens)
        self.downsample = nn.Conv1d(d_model, d_model, kernel_size=7, stride=4, padding=3)
        self.blocks = nn.ModuleList([ConformerBlock(d_model=d_model, nhead=4, kernel_size=15) for _ in range(n_layers)])
        self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=8, stride=4, padding=2)
        self.spatial_remix = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv1d(d_model, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.spatial_unmix(x)
        z_ds = self.downsample(z)  # (B, D, T/4)
        seq = z_ds.transpose(1, 2)  # (B, T/4, D)
        for blk in self.blocks:
            seq = blk(seq)
        z_up = self.upsample(seq.transpose(1, 2))
        z_up = z_up[..., :x.shape[-1]] + z
        noise_hat = self.spatial_remix(z_up)
        return x - noise_hat


def build_model(name: str, in_channels: int = 19) -> nn.Module:
    if name == "UNet1D_Baseline":
        return UNet1D_Baseline(in_channels=in_channels, out_channels=in_channels)
    elif name == "ResAttnUNet1D":
        return ResAttnUNet1D(in_channels=in_channels, out_channels=in_channels)
    elif name == "STFT_MaskNet":
        return STFT_MaskNet(in_channels=in_channels)
    elif name == "SpatioTemporalConformer":
        return SpatioTemporalConformer(in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model: {name}")
