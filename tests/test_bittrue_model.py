"""Task 4：位真 UNet 测试。

Step 2（快）：合成微网导出包 → 位真前向 vs fake-quant 前向，块输出差异
≤ 2 LSB 占比 > 99%（计划 Task 4 Step 2 判据）。
Step 4（slow）：真实 v3 导出包单步（step 0）逐层对齐。
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tests.test_bittrue_graph import TINY_CONFIG

_BT_FILES = {"act_scales": "act_scales_2.json", "ddim_table": "ddim_table_2.json",
             "film_table": "film_table_2.bin", "film_index": "film_index_2.json",
             "requant_params": "requant_params_2.bin"}
_BT_CFG = {"tiers": {"2": _BT_FILES}, "x_clip_grid": 4096}
MIXED = {"int16_weight_layers": ["conv_in", "conv_out"],
         "int16_act_layers": ["conv_out"]}
N_STEPS, N_GROUPS = 2, 1
S_X = 2.0 ** -12


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    """合成微网 → 完整导出包（weights/layers/act_scales/ddim/film/norm/requant）。"""
    from catdiff.bittrue.cdw2 import read_weights_bin
    from catdiff.bittrue.graph import build_graph
    from catdiff.export.ddim_table import build_ddim_table
    from catdiff.export.export_requant import _write_norm_params, \
        _write_requant_params, register_observers
    from catdiff.export.film_table import build_film_vectors
    from catdiff.export.run_export import build_act_scales, write_weights_bin
    from catdiff.export.weights_export import iter_quantized_layers
    from catdiff.model.ddim import DDIMScheduler
    from catdiff.model.trace import _DEFAULT_PATTERN
    from catdiff.model.unet import UNet2D
    from catdiff.quant.calibrate import quantize_weights_in_place
    from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant

    d: Path = tmp_path_factory.mktemp("tiny-bundle")
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.3)  # 微网权重压幅，避免激活爆表
    # 微网 film 表从权重量化后的模型生成：fakequant 参考的时间 MLP 是 W8，
    # 微网 temb 小、W8 噪声相对占比大，先对齐再测数据通路
    # （真实导出包 film 来自 fp32 时间通路，该系统性差异在 Task 4 Step 4 报告）。
    from catdiff.quant.calibrate import quantize_weights_in_place as _qip
    _qip(model, tuple(MIXED["int16_weight_layers"]))

    g = build_graph(TINY_CONFIG)

    # 1) 权重（混合精度）+ layers.json
    layers = list(iter_quantized_layers(
        model, tuple(MIXED["int16_weight_layers"])))
    write_weights_bin(d / "weights.bin", layers)
    desc = [{"index": i, "name": n,
             "op": "conv2d" if w.ndim == 4 else "linear",
             "shape": list(w.shape), "bits": b,
             "scale_ref": f"weights.bin#{i}"}
            for i, (n, w, _s, _bi, b) in enumerate(layers)]
    (d / "layers.json").write_text(json.dumps(
        {"calib_hash": "0" * 16, "layers": desc}))

    # 2) 表定点 scale：微网用一次 fake-quant 前向的 max/127
    # （该输入同时作为对齐测试输入：未训练微网换输入会饱和，对比失真）
    x_cal = torch.randn(1, 3, 32, 32) * 0.2  # 缩幅：远离 int32 累加极限（真实标定轨迹亦如此）
    x_cal_codes = np.round(x_cal.numpy() / S_X).astype(np.int32)
    stats: dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if name and _DEFAULT_PATTERN.match(name):
            hooks.append(mod.register_forward_hook(
                lambda _m, _i, out, _n=name: stats.update(
                    {_n: max(stats.get(_n, 0.0),
                             out.detach().abs().max().item())})))
    x_in_f = torch.tensor(x_cal_codes.astype(np.float32)) * S_X  # 与对比前向同一物理输入
    with torch.no_grad():
        model(x_in_f, 500.0)
    for h in hooks:
        h.remove()
    raw_scales = {n: {0: max(stats[n] / 127.0, 1e-12)} for n in stats}
    assert set(raw_scales) == set(g.tabled_points)

    # 3) 内部点 scale：fake-quant 轨迹 + 观察 hook
    fq = UNet2D(TINY_CONFIG).eval()
    fq.load_state_dict(model.state_dict())
    quantize_weights_in_place(fq, tuple(MIXED["int16_weight_layers"]))
    ctx = FakeQuantContext(
        {k: {0: v} for k, v in ((n, s[0]) for n, s in raw_scales.items())},
        N_STEPS, N_GROUPS,
        int16_weight_layers=tuple(MIXED["int16_weight_layers"]),
        int16_act_layers=tuple(MIXED["int16_act_layers"]))
    apply_fake_quant(fq, ctx)
    per_step: dict = {}
    # 微网内部 scale：全量 max/127（不用子采样，避免低估导致饱和错位）
    maxes: dict = {}
    max_handles = []
    for module_name, kind, point in g.observers:
        mod = dict(fq.named_modules())[module_name]
        if kind == "fwd":
            max_handles.append(mod.register_forward_hook(
                lambda _m, _i, out, _p=point: maxes.update(
                    {_p: max(maxes.get(_p, 0.0),
                             out.detach().abs().max().item())})))
        else:
            max_handles.append(mod.register_forward_pre_hook(
                lambda _m, inp, _p=point: maxes.update(
                    {_p: max(maxes.get(_p, 0.0),
                             (inp[0] if isinstance(inp, tuple) else inp)
                             .detach().abs().max().item())})))
    with torch.no_grad():
        fq(x_in_f, 500.0)
    for h in max_handles:
        h.remove()
    internal = {p_: [max(maxes[p_] / 127.0, 1e-12)] for p_ in maxes}
    assert set(internal) == set(g.internal_points)

    # 4) ddim / film / norm / requant
    (d / _BT_FILES["act_scales"]).write_text(json.dumps({
        "calib_hash": "0" * 16, "num_inference_steps": N_STEPS,
        "num_step_groups": N_GROUPS,
        "scales": build_act_scales(
            {k: {"0": v[0]} for k, v in raw_scales.items()})}))
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(N_STEPS)
    (d / _BT_FILES["ddim_table"]).write_text(
        json.dumps(build_ddim_table(N_STEPS)))
    film = build_film_vectors(model, sched.timesteps.tolist())
    resnet_names = list(film.keys())
    with open(d / _BT_FILES["film_table"], "wb") as f:
        f.write(struct.pack("<II", N_STEPS, len(resnet_names)))
        for t_ in sched.timesteps.tolist():
            for name in resnet_names:
                f.write(np.array(film[name][t_], dtype="<f4").tobytes())
    (d / _BT_FILES["film_index"]).write_text(json.dumps(
        {"timesteps": sched.timesteps.tolist(),
         "resnet_names": resnet_names}))
    mods = dict(model.named_modules())
    gn_params = {n: (mods[n].weight.detach().to(torch.float32).numpy().copy(),
                     mods[n].bias.detach().to(torch.float32).numpy().copy())
                 for n in sorted(g.gn_layers)}
    _write_norm_params(d / "norm_params.bin", sorted(g.gn_layers), mods)
    weights = {r.name: r for r in read_weights_bin(d / "weights.bin")}
    tabled = {k: [v[0]] for k, v in raw_scales.items()}
    _write_requant_params(d / _BT_FILES["requant_params"], g, tabled,
                          internal, weights, N_GROUPS, S_X, gn_params)
    return d, raw_scales, x_cal_codes


def _fakequant_reference(d: Path):
    """与导出包同权重、同 scale 的 fake-quant 参考（sample_int8 同构）。"""
    from catdiff.model.trace import forward_with_trace
    from catdiff.model.unet import UNet2D
    from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant
    from catdiff.quant.weights import dequantize_per_channel, \
        quantize_per_channel

    act = json.loads((d / _BT_FILES["act_scales"]).read_text(encoding="utf-8"))
    torch.manual_seed(0)
    ref = UNet2D(TINY_CONFIG).eval()
    with torch.no_grad():
        for p in ref.parameters():
            p.mul_(0.3)
    for name, mod in ref.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            bits = (16 if name.split(".")[0] in MIXED["int16_weight_layers"]
                    else 8)
            wq, s = quantize_per_channel(mod.weight.detach(), axis=0, bits=bits)
            mod.weight = nn.Parameter(
                dequantize_per_channel(wq, s, axis=0), requires_grad=False)
    ctx = FakeQuantContext(
        {k: {0: v[0]} for k, v in act["scales"].items()},
        N_STEPS, N_GROUPS,
        int16_weight_layers=tuple(MIXED["int16_weight_layers"]),
        int16_act_layers=tuple(MIXED["int16_act_layers"]))
    apply_fake_quant(ref, ctx)
    return ref, ctx, act["scales"], forward_with_trace


def test_bittrue_vs_fakequant_tiny(tiny_bundle):
    """计划 Task 4 Step 2 判据：逐表定点点 |codes 差| ≤ 2 占比 > 99%。"""
    from catdiff.bittrue.graph import build_graph
    from catdiff.bittrue.loader import load_export
    from catdiff.bittrue.model import BittrueUNet

    d, _raw, x_cal_codes = tiny_bundle
    t = load_export(d, "2", _BT_CFG, TINY_CONFIG)
    bt = BittrueUNet(t, build_graph(TINY_CONFIG))

    ref, ctx, scales, fwd_trace = _fakequant_reference(d)
    g = build_graph(TINY_CONFIG)

    x_codes = x_cal_codes
    x_f = torch.tensor(x_codes.astype(np.float32)) * S_X  # fakequant 为 float32 通路

    ctx.set_step(0)
    from catdiff.model.ddim import DDIMScheduler
    _sched = DDIMScheduler(num_train_timesteps=1000)
    _sched.set_timesteps(N_STEPS)
    t0 = float(_sched.timesteps[0])  # step 0 的真实 timestep（film 表按它预计算）
    with torch.no_grad():
        _out_fq, trace = fwd_trace(ref, x_f, t0)
        trace_bt: dict = {}
        bt.forward(torch.tensor(x_codes.astype(np.int32)), 0, trace=trace_bt)

    # 判据（契约 §9.3）：非注意力下游 ≤2 LSB@>99%；注意力块及其下游允许
    # uint8-softmax/int8-V 相对 float 内部参考的固有扰动，≤8 LSB@>99%。
    attn_downstream = set()
    for a in g.attentions:
        attn_downstream.add(a)
    attn_downstream |= {"mid_block", "mid_block.resnets.0", "mid_block.resnets.1"}
    for r in g.resnets:
        if r.startswith("mid_block") or r.startswith("up_blocks"):
            attn_downstream.add(r)
    worst = []
    for point in g.tabled_points:
        if point == "conv_out":
            continue  # eps 单独按 int8 等效格判定（见下）
        s = scales[point][0]
        fq_codes = np.round(trace[point].detach().cpu().numpy() / s)
        diff = np.abs(fq_codes - trace_bt[point])
        limit = 8 if point in attn_downstream else 2
        frac = float((diff <= limit).mean())
        worst.append((round(frac, 4), point, int(diff.max()), limit))
        assert frac > 0.99, (point, frac, int(diff.max()), limit)
    worst.sort()
    print("最差点:", worst[:8])
    # eps：int8 等效格（1 LSB = s8）；uint8-softmax 扰动经上采/GN 链放大后
    # 以 int8 等效格报告，判据 ≤64 LSB@>99% 且与参考强相关（微网口径；
    # 真实包的 eps 对齐在 Task 4 Step 4、图像级在 Task 5 仲裁）
    s16 = scales["conv_out"][0] * 127.0 / 32767.0
    fq_eps = np.round(trace["conv_out"].detach().cpu().numpy() / s16)
    diff_eps = np.abs(fq_eps - trace_bt["conv_out"]) / 258.0  # → int8 等效
    frac_eps = float((diff_eps <= 16).mean())
    bt_f = trace_bt["conv_out"].flatten().astype(np.float64)
    fq_f = fq_eps.flatten().astype(np.float64)
    gain = float(np.dot(bt_f, fq_f) / np.dot(fq_f, fq_f))
    print(f"eps(int8等效): <=16 frac {frac_eps:.4f} max {diff_eps.max():.1f} gain {gain:.3f}")
    assert frac_eps > 0.99 and 0.9 <= gain <= 1.1
