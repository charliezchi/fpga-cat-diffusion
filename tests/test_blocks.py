import torch
from diffusers.models.downsampling import Downsample2D as RefDownsample
from diffusers.models.upsampling import Upsample2D as RefUpsample

from catdiff.model.blocks import Downsample2D, Upsample2D


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_downsample_matches_diffusers():
    torch.manual_seed(0)
    # name="Conv2d_0"：与真实 UNet2DModel 内部的注册方式一致——state_dict 只含
    # conv.* 键（默认 name="conv" 会额外注册遗留别名 Conv2d_0，整网里不存在）
    ref = RefDownsample(channels=64, use_conv=True, out_channels=64, padding=0,
                        name="Conv2d_0")
    mine = Downsample2D(64)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(1, 64, 64, 64)
    out_mine, out_ref = mine(x), ref(x)
    assert out_mine.shape == (1, 64, 32, 32)  # 非对称 padding 保证整除
    assert _rel(out_mine, out_ref) < 1e-4


def test_upsample_matches_diffusers():
    torch.manual_seed(0)
    ref = RefUpsample(channels=64, use_conv=True, out_channels=64)
    mine = Upsample2D(64)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(1, 64, 16, 16)
    out_mine, out_ref = mine(x), ref(x)
    assert out_mine.shape == (1, 64, 32, 32)
    assert _rel(out_mine, out_ref) < 1e-4
