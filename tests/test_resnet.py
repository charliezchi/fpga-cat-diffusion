import torch
from diffusers.models.resnet import ResnetBlock2D as RefResnet

from catdiff.model.resnet import ResnetBlock2D


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_resnet_same_channels_matches_diffusers():
    torch.manual_seed(0)
    ref = RefResnet(in_channels=32, out_channels=32, temb_channels=128, groups=8, eps=1e-6)
    mine = ResnetBlock2D(32, 32, 128, groups=8)
    mine.load_state_dict(ref.state_dict())
    x, temb = torch.randn(1, 32, 16, 16), torch.randn(1, 128)
    assert _rel(mine(x, temb), ref(x, temb)) < 1e-4


def test_resnet_channel_change_matches_diffusers():
    torch.manual_seed(0)
    ref = RefResnet(in_channels=32, out_channels=64, temb_channels=128, groups=8, eps=1e-6)
    mine = ResnetBlock2D(32, 64, 128, groups=8)
    mine.load_state_dict(ref.state_dict())  # 含 conv_shortcut 命名对齐
    assert mine.conv_shortcut is not None
    x, temb = torch.randn(1, 32, 16, 16), torch.randn(1, 128)
    assert _rel(mine(x, temb), ref(x, temb)) < 1e-4
