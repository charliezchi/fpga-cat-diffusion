"""UNet 结构件：上下采样与 down/mid/up 块容器，命名与 diffusers 镜像。"""

import torch
import torch.nn.functional as F
from torch import nn

from .attention import AttentionBlock
from .resnet import ResnetBlock2D


class Downsample2D(nn.Module):
    """diffusers Downsample2D(use_conv=True, padding=0)：非对称 pad + 3x3 stride2。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))  # diffusers 内部行为：左0右1，保证 64->32 整除
        return self.conv(x)


class Upsample2D(nn.Module):
    """diffusers Upsample2D(use_conv=True)：nearest x2 + 3x3 pad1。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    """DownBlock2D / AttnDownBlock2D 复刻。返回 (输出, skip 列表)。"""

    def __init__(self, in_channels, out_channels, temb_channels, num_layers,
                 groups, add_downsample, with_attention):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if with_attention else None
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(cin, out_channels, temb_channels, groups))
            if with_attention:
                self.attentions.append(AttentionBlock(out_channels, groups))
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None
        )

    def forward(self, x, temb):
        skips = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class MidBlock(nn.Module):
    """UNetMidBlock2D(add_attention=True)：resnet -> attention -> resnet。"""

    def __init__(self, channels, temb_channels, groups):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, temb_channels, groups),
            ResnetBlock2D(channels, channels, temb_channels, groups),
        ])
        self.attentions = nn.ModuleList([AttentionBlock(channels, groups)])

    def forward(self, x, temb):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x)
        return self.resnets[1](x, temb)


class UpBlock(nn.Module):
    """UpBlock2D / AttnUpBlock2D 复刻（num_layers = layers_per_block + 1）。

    skips 按消费顺序给出（追加序的逆序），由调用方（UNet2D）准备。
    """

    def __init__(self, in_channels, out_channels, prev_output_channel,
                 temb_channels, num_layers, groups, add_upsample, with_attention):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if with_attention else None
        for i in range(num_layers):
            skip_ch = in_channels if i == num_layers - 1 else out_channels
            cin = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(cin + skip_ch, out_channels, temb_channels, groups)
            )
            if with_attention:
                self.attentions.append(AttentionBlock(out_channels, groups))
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(self, x, temb, skips):
        for i, resnet in enumerate(self.resnets):
            x = torch.cat([x, skips[i]], dim=1)
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x
