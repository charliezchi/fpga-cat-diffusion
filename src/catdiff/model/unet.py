"""手写复刻 diffusers UNet2DModel（ddpm-cat-256 配置族）。

state_dict 命名与 diffusers 完全镜像，load_state_dict 严格加载即完成权重映射。
"""

import types

import torch
from torch import nn

from .blocks import DownBlock, MidBlock, UpBlock
from .time_embed import TimestepEmbedding, Timesteps


class UNet2D(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        chs = config["block_out_channels"]
        layers = config["layers_per_block"]
        groups = config["norm_num_groups"]
        time_embed_dim = chs[0] * 4
        # 以下键缺省即 cat-256 配置族（docs/reference/unet-config-ddpm-cat-256.json）；
        # butterflies-64 经 config 显式给出不同值
        freq_shift = config.get("freq_shift", 1)
        flip_sin_to_cos = config.get("flip_sin_to_cos", False)
        attention_head_dim = config.get("attention_head_dim", None)
        downsample_padding = config.get("downsample_padding", 0)
        eps = config.get("norm_eps", 1e-6)

        self.conv_in = nn.Conv2d(config["in_channels"], chs[0], 3, padding=1)
        self.time_proj = Timesteps(chs[0], flip_sin_to_cos=flip_sin_to_cos,
                                   downscale_freq_shift=freq_shift)
        self.time_embedding = TimestepEmbedding(chs[0], time_embed_dim)

        self.down_blocks = nn.ModuleList()
        cin = chs[0]
        for i, block_type in enumerate(config["down_block_types"]):
            cout = chs[i]
            self.down_blocks.append(DownBlock(
                cin, cout, time_embed_dim, layers, groups,
                add_downsample=i < len(chs) - 1,
                with_attention="Attn" in block_type,
                attention_head_dim=attention_head_dim,
                downsample_padding=downsample_padding,
                eps=eps,
            ))
            cin = cout

        self.mid_block = MidBlock(chs[-1], time_embed_dim, groups,
                                  attention_head_dim=attention_head_dim, eps=eps)

        self.up_blocks = nn.ModuleList()
        rchs = list(reversed(chs))
        prev_out = rchs[0]
        for i, block_type in enumerate(config["up_block_types"]):
            cout = rchs[i]
            cin_skip = rchs[min(i + 1, len(rchs) - 1)]
            self.up_blocks.append(UpBlock(
                cin_skip, cout, prev_out, time_embed_dim, layers + 1, groups,
                add_upsample=i < len(chs) - 1,
                with_attention="Attn" in block_type,
                attention_head_dim=attention_head_dim,
                eps=eps,
            ))
            prev_out = cout

        self.conv_norm_out = nn.GroupNorm(groups, chs[0], eps=eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(chs[0], config["out_channels"], 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        timesteps = timestep.reshape(-1).expand(sample.shape[0]).to(sample.device)
        temb = self.time_embedding(self.time_proj(timesteps))

        x = self.conv_in(sample)
        skips = [x]
        for block in self.down_blocks:
            x, block_skips = block(x, temb)
            skips.extend(block_skips)
        x = self.mid_block(x, temb)
        for block in self.up_blocks:
            take = skips[-len(block.resnets):]
            del skips[-len(block.resnets):]
            x = block(x, temb, list(reversed(take)))
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


def load_handwritten_unet(model_id: str, config: dict) -> UNet2D:
    """加载 diffusers 权重到手写模型（slow 路径：需要 HF 缓存）。"""
    from diffusers import UNet2DModel

    ref = UNet2DModel.from_pretrained(model_id)
    model = UNet2D(config)
    model.load_state_dict(ref.state_dict())  # strict：命名漂移在此报错
    model.eval()
    return model


def as_diffusers_output(model: UNet2D, in_channels: int = 3):
    """把手写模型（返回裸 tensor）适配成 baseline.sample_batch 期望的接口。"""

    class _Adapter:
        def __init__(self, m):
            self._m = m
            self.config = types.SimpleNamespace(in_channels=in_channels)

        def __call__(self, x, t):
            return types.SimpleNamespace(sample=self._m(x, t))

        def to(self, device):
            return self

    return _Adapter(model)
