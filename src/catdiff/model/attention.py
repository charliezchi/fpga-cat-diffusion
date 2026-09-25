"""复刻 diffusers Attention（UNet2DModel 用法：group_norm + 单头 + to_out）。

cat-256 的 attention_head_dim=null，实际为 heads=1、dim_head=channels。
参考实现走 F.scaled_dot_product_attention；此处用手动 softmax，
单头 fp32 下数值差约 2e-7，远在 1e-4 容差内。
"""

import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])
        self.scale = channels ** -0.5  # heads=1, dim_head=channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.group_norm(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = attn @ v
        out = self.to_out[1](self.to_out[0](out))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out + residual
