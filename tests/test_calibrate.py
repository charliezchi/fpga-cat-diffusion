import torch

from catdiff.quant.calibrate import step_to_group, finalize_scales, update_percentile_stats


def test_step_to_group_mapping():
    # 20 步 5 组：步 0-3→0, 4-7→1, 8-11→2, 12-15→3, 16-19→4
    expect = [0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4
    assert [step_to_group(i, 20, 5) for i in range(20)] == expect


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
