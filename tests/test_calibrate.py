import torch
from torch import nn

from catdiff.quant.calibrate import (step_to_group, finalize_scales,
                                     merge_step_stats,
                                     quantize_weights_in_place,
                                     update_percentile_stats)
from catdiff.model.unet import UNet2D
from catdiff.quant.weights import quantize_per_channel, dequantize_per_channel
from tests.test_unet import TINY_CONFIG


def test_step_to_group_mapping():
    # 20 步 5 组：步 0-3→0, 4-7→1, 8-11→2, 12-15→3, 16-19→4
    expect = [0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4
    assert [step_to_group(i, 20, 5) for i in range(20)] == expect


def test_step_to_group_mapping_v3_schedules():
    # 50 步 10 组（主档）：每组 5 步
    expect50 = [g for g in range(10) for _ in range(5)]
    assert [step_to_group(i, 50, 10) for i in range(50)] == expect50
    # 20 步 10 组（预览档）：每组 2 步
    expect20 = [g for g in range(10) for _ in range(2)]
    assert [step_to_group(i, 20, 10) for i in range(20)] == expect20


def test_stats_finalize_uses_percentile():
    stats = {}
    g = torch.Generator().manual_seed(0)
    for _ in range(8):
        update_percentile_stats(stats, "layer0", 0, torch.randn(256, generator=g) * 3.0)
    scales = finalize_scales(stats, percentile=99.9)
    assert "layer0" in scales and 0 in scales["layer0"]
    s = scales["layer0"][0]
    assert s > 0
    # scale = P99.9(|a|)/127，应明显大于 std=3 的 1 倍
    assert s * 127 > 3.0


def test_merge_step_stats_concatenates_group_steps():
    per_step = {"L": {s: [s] * 3 for s in range(6)}}
    merged = merge_step_stats(per_step, num_steps=6, num_groups=3)
    assert merged["L"][0] == [0, 0, 0, 1, 1, 1]
    assert merged["L"][1] == [2, 2, 2, 3, 3, 3]
    assert merged["L"][2] == [4, 4, 4, 5, 5, 5]
    # 缺步容忍：末步缺失时归并不报错
    partial = {"L": {s: [1.0] for s in range(5)}}
    merged = merge_step_stats(partial, num_steps=6, num_groups=3)
    assert sum(merged["L"][g] == [1.0] * 2 for g in range(3)) == 2


def test_quantize_weights_in_place_mixed_bits():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    original = {n: m.weight.detach().clone() for n, m in model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))}
    n = quantize_weights_in_place(model, int16_layers=("conv_in", "conv_out"))
    assert n == len(original)
    for name, w0 in original.items():
        bits = 16 if name.split(".")[0] in ("conv_in", "conv_out") else 8
        w_q, s = quantize_per_channel(w0, axis=0, bits=bits)
        expect = dequantize_per_channel(w_q, s, axis=0)
        got = dict(model.named_modules())[name].weight.detach()
        assert torch.equal(got, expect), name
        assert not got.requires_grad  # 已就位为推理权重
