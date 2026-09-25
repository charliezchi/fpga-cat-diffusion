import torch

from catdiff.model.unet import UNet2D
from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant
from tests.test_unet import TINY_CONFIG


def _scales_for(model, value=0.01):
    return {name: {g: value for g in range(5)}
            for name, _ in model.named_modules() if name}


def test_fakequant_changes_but_preserves_shape():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    ctx = FakeQuantContext(_scales_for(model), num_steps=20, num_groups=5)
    n = apply_fake_quant(model, ctx)
    assert n > 0
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    ref_model = UNet2D(TINY_CONFIG).eval()
    ref_model.load_state_dict(model.state_dict())  # 权重本身未被破坏
    with torch.no_grad():
        ctx.set_step(0)
        out_q = model(x, t)
        out_f = ref_model(x, t)
    assert out_q.shape == out_f.shape
    assert not torch.allclose(out_q, out_f)


def test_step_group_switches_scale():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    scales = _scales_for(model)
    key = "conv_in"  # 必须选挂了 hook 的量化点（_DEFAULT_PATTERN 匹配的层）
    scales[key][0], scales[key][4] = 0.001, 1.0
    ctx = FakeQuantContext(scales, num_steps=20, num_groups=5)
    apply_fake_quant(model, ctx)
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        ctx.set_step(0)
        out_g0 = model(x, t).clone()
        ctx.set_step(19)
        out_g4 = model(x, t).clone()
    assert not torch.allclose(out_g0, out_g4)  # 不同步组 scale 生效
