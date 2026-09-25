"""时间步嵌入：复刻 diffusers Timesteps + TimestepEmbedding。

cat-256 配置（docs/reference/unet-config-ddpm-cat-256.json）：
flip_sin_to_cos=False、freq_shift=1（即 downscale_freq_shift=1）。
"""

import math

import torch
from torch import nn


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """对应 diffusers get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=1)。"""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half - 1)  # downscale_freq_shift = 1
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(timesteps, self.num_channels)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))
