import torch
from torch import nn

from catdiff.model.unet import UNet2D
from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant
from catdiff.quant.weights import dequantize_per_channel, quantize_per_channel
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


def test_bits_selection_prefix_matching():
    # 按模块名首段前缀匹配（与 spike 的 name.split(".")[0] 语义一致）
    ctx = FakeQuantContext({}, num_steps=50, num_groups=10,
                           int16_weight_layers=("conv_in", "conv_out"),
                           int16_act_layers=("conv_out",))
    assert ctx.weight_bits("conv_in") == 16
    assert ctx.weight_bits("conv_out") == 16
    assert ctx.weight_bits("up_blocks.4.resnets.0.conv1") == 8
    assert ctx.weight_bits("mid_block.attentions.0.to_q") == 8
    assert ctx.act_bits("conv_out") == 16
    assert ctx.act_bits("conv_in") == 8
    assert ctx.act_bits("down_blocks.0.resnets.0.conv1") == 8


def test_mixed_precision_weight_bits_applied():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    original = {n: m.weight.detach().clone() for n, m in model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))}
    ctx = FakeQuantContext(_scales_for(model), num_steps=20, num_groups=5,
                           int16_weight_layers=("conv_in",))
    apply_fake_quant(model, ctx)
    for name, w0 in original.items():
        bits = 16 if name.split(".")[0] == "conv_in" else 8
        w_q, s = quantize_per_channel(w0, axis=0, bits=bits)
        expect = dequantize_per_channel(w_q, s, axis=0)
        got = dict(model.named_modules())[name].weight.detach()
        assert torch.equal(got, expect), name
    assert dict(model.named_modules())["conv_in"].weight.abs().max() <= 32767


def test_int16_act_hook_scale_conversion():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    captured = {}
    # 先注册捕获 hook（执行顺序 = 注册顺序），再挂 fake-quant hook
    model.conv_out.register_forward_hook(
        lambda m, i, o: captured.__setitem__("raw", o.detach()))
    s8 = 0.05
    scales = _scales_for(model)
    scales["conv_out"] = {g: s8 for g in range(5)}
    ctx = FakeQuantContext(scales, num_steps=20, num_groups=5,
                           int16_act_layers=("conv_out",))
    apply_fake_quant(model, ctx)
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        out = model(x, t)
    # INT16 激活 scale = 同点 INT8 scale × 127/32767
    s16 = s8 * 127.0 / 32767
    expect = torch.round(captured["raw"] / s16).clamp(-32768, 32767) * s16
    assert torch.allclose(out, expect, atol=1e-8)
