import torch

from catdiff.model.ablate import strip_attention
from catdiff.model.unet import UNet2D, as_diffusers_output
from tests.test_unet import TINY_CONFIG


def _make():
    torch.manual_seed(0)
    return UNet2D(TINY_CONFIG).eval()


def test_strip_attention_replaces_all_six():
    model = _make()
    n = strip_attention(model)
    assert n == 6  # tiny 配置：down_blocks.2 x2 + mid x1 + up_blocks.0 x3


def test_strip_keep_mid():
    model = _make()
    n = strip_attention(model, keep_mid=True)
    assert n == 5  # 保留 mid_block.attentions.0


def test_forward_after_strip_runs_and_changes_output():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        out_full = model(x, t)
        strip_attention(model)
        out_stripped = model(x, t)
    assert out_stripped.shape == out_full.shape
    assert not torch.allclose(out_stripped, out_full)


def test_adapter_exposes_sample_interface():
    model = as_diffusers_output(_make())
    assert model.config.in_channels == 3
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        out = model(x, t)
    assert out.sample.shape == (1, 3, 32, 32)
    assert model.to("cpu") is model
