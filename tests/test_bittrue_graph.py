"""Task 1 自查：接线图与 v3 导出包/act_scales/trace 捕获面一致（契约 §3 Step 3）。"""

import json
from pathlib import Path

from catdiff.bittrue.graph import KIND_AV, build_graph

CAT_CONFIG = json.loads(Path(
    "docs/reference/unet-config-ddpm-cat-256.json").read_text(encoding="utf-8"))

TINY_CONFIG = dict(
    in_channels=3, out_channels=3,
    down_block_types=("DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    block_out_channels=[32, 64], layers_per_block=1, norm_num_groups=8,
)


def test_tabled_points_match_trace_pattern():
    from catdiff.model.trace import _DEFAULT_PATTERN
    g = build_graph(CAT_CONFIG)
    assert len(g.tabled_points) == 51
    for name in g.tabled_points:
        assert _DEFAULT_PATTERN.match(name), name
    assert "mid_block" in g.tabled_points and "mid_block.resnets.1" in g.tabled_points


def test_internal_point_count_and_uniqueness():
    g = build_graph(CAT_CONFIG)
    # 32 resnet×5 + 18 up-concat + 6 attn×3 + 2 conv_norm_out = 198
    assert len(g.internal_points) == 198
    assert len(set(g.internal_points)) == 198
    assert not (set(g.internal_points) & set(g.tabled_points))


def test_wired_layers_match_layers_json():
    g = build_graph(CAT_CONFIG)
    layers_json = json.loads(Path(
        "artifacts/f4/export-v3/layers.json").read_text(encoding="utf-8"))
    all_names = {l["name"] for l in layers_json["layers"]}
    wired = set(g.all_layers())
    # wired = 全部 conv/linear - time_embedding - time_emb_proj + av_requant 伪层
    excluded = {n for n in all_names
                if n.split(".")[-1] == "time_emb_proj"
                or n.startswith("time_embedding")}
    expect = (all_names - excluded
              | {f"{a}.av_requant" for a in g.attentions})
    assert wired == expect
    assert len(g.all_layers()) == 126


def test_gn_layers_cover_norm_modules():
    g = build_graph(CAT_CONFIG)
    # 32 resnet×2 + 6 attention + conv_norm_out = 71，键 = GN 模块名
    assert len(g.gn_layers) == 71
    assert "conv_norm_out" in g.gn_layers
    assert "down_blocks.4.attentions.0.group_norm" in g.gn_layers
    assert "up_blocks.1.attentions.2.group_norm" in g.gn_layers
    assert g.gn_layers["down_blocks.0.resnets.0.norm1"].out_point == \
        "down_blocks.0.resnets.0.norm1"
    assert g.gn_layers["conv_norm_out"].out_point == "conv_norm_out.gn"


def test_observer_targets_resolve():
    """观察点 hook 目标必须是手写模型的真实模块名。"""
    from catdiff.model.unet import UNet2D
    g = build_graph(CAT_CONFIG)
    model = UNet2D(CAT_CONFIG).eval()
    mods = {n for n, _ in model.named_modules()}
    for module_name, _kind, _point in g.observers:
        assert module_name in mods, module_name


def test_tiny_config_graph_consistent():
    g = build_graph(TINY_CONFIG)
    # conv_in + (d0: r,ds) + (d1: r,a；末块无 ds) + mid 4
    # + (up0: r,a,r,a,us) + (up1: r,r) + conv_out
    assert len(g.tabled_points) == 17
    assert len(g.resnets) == 2 + 2 + 2 * 2  # down 2×1 + mid 2 + up 2×2
    assert set(g.attentions) == {
        "down_blocks.1.attentions.0", "mid_block.attentions.0",
        "up_blocks.0.attentions.0", "up_blocks.0.attentions.1"}
    assert len(g.gn_layers) == 8 * 2 + 4 + 1


def test_skip_pairing_matches_unet_forward():
    """up 路 concat 配对必须与 UNet2D.forward 的 skip 栈一致（通道数核对）。"""
    import torch
    from catdiff.model.trace import forward_with_trace
    from catdiff.model.unet import UNet2D

    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    x = torch.randn(1, 3, 32, 32)
    _out, trace = forward_with_trace(model, x, 0)
    g = build_graph(TINY_CONFIG)
    mods = dict(model.named_modules())
    for r, info in g.resnets.items():
        if info.concat is None:
            continue
        x_src, skip_src = info.concat
        assert x_src in trace and skip_src in trace, (r, x_src, skip_src)
        cin = mods[f"{r}.conv1"].in_channels
        assert trace[x_src].shape[1] + trace[skip_src].shape[1] == cin, r
    # GN 表定点输入点通道要对上（内部输入点不在 trace 面，由 model 测试覆盖）
    for gn, ref in g.gn_layers.items():
        src = ref.in_point
        if src in g.internal_points or src == "@x":
            continue
        assert src in trace, (gn, src)
        assert mods[gn].num_channels == trace[src].shape[1], gn
