"""时间步嵌入：复刻 diffusers Timesteps + TimestepEmbedding。

默认参数为 cat-256 配置（docs/reference/unet-config-ddpm-cat-256.json）：
flip_sin_to_cos=False、downscale_freq_shift=1（即 freq_shift=1，恰为 diffusers 默认）。
butterflies-64 用 flip_sin_to_cos=True、downscale_freq_shift=0，经参数传入。
"""

import math

import torch
from torch import nn


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: int = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """对应 diffusers get_timestep_embedding（默认 cat-256：不翻转、shift=1）。"""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    if flip_sin_to_cos:
        return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = False,
                 downscale_freq_shift: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(
            timesteps, self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))
