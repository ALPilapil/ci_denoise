"""
model.py - 1D U-Net Architecture for 21-Channel CI EEG Denoising.
"""

import torch
import torch.nn as nn


def double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
        nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
    )


class UNet1D(nn.Module):
    """
    1D U-Net with skip connections for multi-channel temporal EEG denoising.
    Input shape:  (batch_size, n_channels=21, n_times)
    Output shape: (batch_size, n_channels=21, n_times)
    """
    def __init__(self, in_channels: int = 21, out_channels: int = 21):
        super().__init__()

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(in_channels, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.down_conv_1(x)
        x2 = self.max_pool(x1)

        x3 = self.down_conv_2(x2)
        x4 = self.max_pool(x3)

        x5 = self.down_conv_3(x4)
        x6 = self.max_pool(x5)

        x7 = self.down_conv_4(x6)
        x8 = self.max_pool(x7)

        x9 = self.down_conv_5(x8)

        # Decoder with crop & concat skip connections
        u1 = self.up_trans_1(x9)
        u1 = self.up_conv_1(torch.cat([x7[..., :u1.shape[-1]], u1], dim=1))

        u2 = self.up_trans_2(u1)
        u2 = self.up_conv_2(torch.cat([x5[..., :u2.shape[-1]], u2], dim=1))

        u3 = self.up_trans_3(u2)
        u3 = self.up_conv_3(torch.cat([x3[..., :u3.shape[-1]], u3], dim=1))

        u4 = self.up_trans_4(u3)
        u4 = self.up_conv_4(torch.cat([x1[..., :u4.shape[-1]], u4], dim=1))

        out = self.out(u4)

        # Ensure exact match in time dimension
        min_t = min(out.shape[-1], x.shape[-1])
        return out[..., :min_t]
