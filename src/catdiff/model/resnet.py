"""复刻 diffusers ResnetBlock2D（pre-norm、silu、output_scale_factor=1.0）。"""

import torch
from torch import nn


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels,
                 groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(self.nonlinearity(self.norm1(x)))
        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden = hidden + temb
        hidden = self.conv2(self.dropout(self.nonlinearity(self.norm2(hidden))))
        shortcut = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return shortcut + hidden
