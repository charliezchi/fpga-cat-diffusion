import torch

from catdiff.quant.weights import dequantize_per_channel, quantize_per_channel


def test_conv_weight_roundtrip_per_out_channel():
    torch.manual_seed(0)
    w = torch.randn(64, 32, 3, 3) * torch.logspace(-3, 0, 64).reshape(64, 1, 1, 1)
    w_q, scales = quantize_per_channel(w, axis=0)
    assert w_q.dtype == torch.int8 and scales.shape == (64,)
    assert w_q.abs().max() <= 127
    # 每个输出通道至少一个元素打满量程
    assert (w_q.abs().amax(dim=(1, 2, 3)) == 127).all()
    w_hat = dequantize_per_channel(w_q, scales, axis=0)
    rel = ((w_hat - w).abs().max() / w.abs().max()).item()
    assert rel < 0.02  # 每通道独立缩放，大幅值通道误差有界


def test_linear_weight_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(512, 128)
    w_q, scales = quantize_per_channel(w, axis=0)
    assert scales.shape == (512,)
    w_hat = dequantize_per_channel(w_q, scales, axis=0)
    assert ((w_hat - w).abs().max() / w.abs().max()).item() < 0.02


def test_zero_channel_is_safe():
    w = torch.zeros(4, 8)
    w_q, scales = quantize_per_channel(w, axis=0)
    assert torch.isfinite(scales).all() and w_q.abs().max() == 0


def test_int16_per_channel_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(4, 3, 3, 3) * torch.logspace(-3, 0, 4).reshape(4, 1, 1, 1)
    w_q, scales = quantize_per_channel(w, axis=0, bits=16)
    assert w_q.dtype == torch.int16 and scales.shape == (4,)
    assert w_q.abs().max() <= 32767
    # 每个输出通道至少一个元素打满量程
    assert (w_q.abs().amax(dim=(1, 2, 3)) == 32767).all()
    w_hat = dequantize_per_channel(w_q, scales, axis=0)
    rel = ((w_hat - w).abs().max() / w.abs().max()).item()
    assert rel < 1e-3  # INT16 量化误差远小于 INT8
