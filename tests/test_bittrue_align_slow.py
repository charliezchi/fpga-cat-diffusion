"""Task 4 Step 4（slow）：真实 v3 导出包单步（50 档 step 0 / t=980）逐层对齐。

判据（契约 §9.3）：非注意力下游 ≤2 LSB@>99%；注意力及其下游 ≤8 LSB@>99%；
eps 以 conv_out 的 INT8 基准格（1 LSB = s8）报告并落盘
artifacts/f5/layer-align-step0.json。
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tests.test_bittrue_graph import CAT_CONFIG

BT_CFG = json.loads(Path("configs/bittrue.json").read_text(encoding="utf-8"))
EXPORT = Path("artifacts/f4/export-v3")
TIER = "50"
SEED = 1000


def _fakequant_reference():
    from catdiff.model.ddim import DDIMScheduler
    from catdiff.model.trace import forward_with_trace
    from catdiff.model.unet import load_handwritten_unet
    from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant
    from catdiff.quant.weights import dequantize_per_channel, \
        quantize_per_channel

    act = json.loads((EXPORT / f"act_scales_{TIER}.json").read_text(
        encoding="utf-8"))
    n_steps = act["num_inference_steps"]
    mixed = {"int16_weight_layers": ["conv_in", "conv_out"],
             "int16_act_layers": ["conv_out"]}
    model = load_handwritten_unet("google/ddpm-cat-256", CAT_CONFIG)
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            bits = 16 if name.split(".")[0] in mixed["int16_weight_layers"] \
                else 8
            wq, s = quantize_per_channel(mod.weight.detach(), axis=0, bits=bits)
            mod.weight = nn.Parameter(
                dequantize_per_channel(wq, s, axis=0), requires_grad=False)
    ctx = FakeQuantContext(
        {k: {g_: v for g_, v in enumerate(vs)}
         for k, vs in act["scales"].items()},
        n_steps, act["num_step_groups"],
        int16_weight_layers=tuple(mixed["int16_weight_layers"]),
        int16_act_layers=tuple(mixed["int16_act_layers"]))
    apply_fake_quant(model, ctx)
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(n_steps)
    return model, ctx, act["scales"], forward_with_trace, float(sched.timesteps[0])


@pytest.mark.slow
def test_real_bundle_alignment_step0(tmp_path):
    from catdiff.bittrue.graph import build_graph
    from catdiff.bittrue.loader import load_export
    from catdiff.bittrue.model import BittrueUNet
    from catdiff.bittrue.primitives import S_X
    from catdiff.model.ddim import DDIMScheduler

    tables = load_export(EXPORT, TIER, BT_CFG, CAT_CONFIG)
    bt = BittrueUNet(tables, build_graph(CAT_CONFIG))
    g = build_graph(CAT_CONFIG)
    ref, ctx, scales, fwd_trace, t0 = _fakequant_reference()
    n_steps = len(tables.ddim_A)
    assert t0 == 980.0

    gnr = torch.Generator().manual_seed(SEED)
    x0 = torch.randn(1, 3, 256, 256, generator=gnr)
    x_codes = np.clip(np.round(x0.numpy() / S_X), -32768, 32767)
    x_f = torch.tensor(x_codes.astype(np.float32)) * S_X

    ctx.set_step(0)
    with torch.no_grad():
        _out_fq, trace = fwd_trace(ref, x_f, t0)
        trace_bt, trace_in = {}, {}
        bt.forward(torch.from_numpy(x_codes.astype(np.int32)), 0,
                   trace=trace_bt, trace_in=trace_in)

    attn_downstream = {"mid_block", "mid_block.resnets.0", "mid_block.resnets.1"}
    for r in g.resnets:
        if r.startswith("mid_block") or r.startswith("up_blocks"):
            attn_downstream.add(r)

    report = {}
    fails = []
    for point in g.tabled_points:
        if point == "conv_out":
            continue
        s = scales[point][0]
        fq_codes = np.round(trace[point].detach().cpu().numpy() / s)
        diff = np.abs(fq_codes - trace_bt[point])
        limit = 8 if point in attn_downstream else 2
        rec = {"limit": limit,
               "frac_le": round(float((diff <= limit).mean()), 6),
               "max": int(diff.max()),
               "mean": round(float(diff.mean()), 4)}
        report[point] = rec
        if rec["frac_le"] <= 0.99:
            fails.append((point, rec))
    # eps：INT8 基准格（1 LSB = s8）
    s8 = scales["conv_out"][0]
    s16 = s8 * 127.0 / 32767.0
    fq_eps = np.round(trace["conv_out"].detach().cpu().numpy() / s16)
    diff8 = np.abs(fq_eps - trace_bt["conv_out"]) / 258.0
    report["conv_out"] = {
        "unit": "int8_equiv_LSB", "limit": 64,
        "frac_le": round(float((diff8 <= 64).mean()), 6),
        "max": round(float(diff8.max()), 2),
        "mean": round(float(diff8.mean()), 4)}

    out = Path("artifacts/f5/layer-align-step0.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": SEED, "t": t0, "tier": TIER,
        "s8_conv_out": s8, "points": report}, indent=1), encoding="utf-8")
    print(f"报告 -> {out}")
    if fails:
        for point, rec in fails:
            print("FAIL", point, rec)
    assert not fails, f"{len(fails)} 点未达标"
