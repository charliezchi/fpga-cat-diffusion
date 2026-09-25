"""F5 扩展导出（docs/bittrue-spec.md §7）：norm_params.bin + requant_params_<steps>.bin。

在 v3 导出包（artifacts/f4/export-v3，CDW2 与既有文件不动）上追加：
1. 内部激活量化点标定（契约 §4，协议同 F4 金标：顺序标定 8 seeds、
   每(点,步)子采样 2048、P99.95），并顺带记录 DDIM x 轨迹 max|x|（s_x 论证）；
2. 全部 wired 层的 requant M/N（含 conv_in 的 x 边界、conv_out 的 s16 出点、
   av_requant 的 1/256、GN 输出 requant 的 Gc/Nc/Bq）。

与 run_export.py 的用法对齐（每档一次）：
  uv run python src/catdiff/export/export_requant.py \
      --quant-config configs/quant_int8.json \
      --calib-stats artifacts/f4/calib-stats-ddim50-seq.json \
      --out-dir artifacts/f4/export-v3 --table-suffix 50
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch

from catdiff.bittrue.cdw2 import read_weights_bin
from catdiff.bittrue.graph import KIND_AV, KIND_CONV, KIND_GN, P_INTERNAL, \
    P_TABLED, P_X, build_graph
from catdiff.bittrue.primitives import scales_to_MN
from catdiff.export.run_export import build_act_scales
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import load_handwritten_unet
from catdiff.quant.calibrate import finalize_scales, merge_step_stats, \
    quantize_weights_in_place, update_percentile_stats
from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant

_MAGIC = b"RQP1"
_NORM_MAGIC = b"NRM1"
_X_POINT = "@x"


def _point_index(g) -> dict[str, int]:
    order = list(g.tabled_points) + list(g.internal_points) + [_X_POINT]
    return {name: i for i, name in enumerate(order)}


def _in_scale(point: str, tabled: dict, internal: dict, s_x: float,
              g_idx: int, g) -> float:
    if point == _X_POINT:
        return s_x
    src = internal if point in g.internal_points else tabled
    return src[point][g_idx]


def _out_scale(point: str, tabled: dict, internal: dict, g_idx: int, g) -> float:
    src = internal if point in g.internal_points else tabled
    s8 = src[point][g_idx]
    if point == "conv_out":  # eps INT16：同一物理量程重标到 INT16 网格（契约 §1.3）
        return s8 * 127.0 / 32767.0
    return s8


def _write_norm_params(path: Path, gn_names: list[str], mods: dict) -> int:
    """71 个 GN 的 γ/β（fp32），记录名 = GN 模块名。"""
    with open(path, "wb") as f:
        f.write(_NORM_MAGIC)
        f.write(struct.pack("<I", len(gn_names)))
        for name in gn_names:
            mod = mods[name]
            gamma = mod.weight.detach().to(torch.float32).numpy().copy()
            beta = mod.bias.detach().to(torch.float32).numpy().copy()
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<I", gamma.size))
            f.write(gamma.astype("<f4").tobytes())
            f.write(beta.astype("<f4").tobytes())
    return len(gn_names)


def _write_requant_params(path: Path, g, tabled: dict, internal: dict,
                          weights: dict, num_groups: int, s_x: float,
                          gn_params: dict[str, tuple[np.ndarray, np.ndarray]]) -> int:
    pidx = _point_index(g)
    layer_refs = g.all_layers()                      # KIND_CONV / KIND_AV
    gn_names = sorted(g.gn_layers)                   # KIND_GN
    n_layers = len(layer_refs) + len(gn_names)
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<IIII", 1, num_groups, n_layers,
                            len(g.internal_points)))
        for name in g.internal_points:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(np.array(internal[name], dtype="<f4").tobytes())
        for name, ref in layer_refs.items():
            rec = weights.get(name)  # av_requant 伪层无权重记录
            assert ref.kind == KIND_AV or rec is not None, name
            in_kind = (P_X if ref.in_point == _X_POINT else
                       (P_INTERNAL if ref.in_point in g.internal_points
                        else P_TABLED))
            out_kind = (P_INTERNAL if ref.out_point in g.internal_points
                        else P_TABLED)
            c_count = 1 if ref.kind == KIND_AV else rec.scales.size
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<BBHBBH", ref.kind, in_kind,
                                pidx[ref.in_point], out_kind,
                                pidx[ref.out_point], c_count))
            for gi in range(num_groups):
                s_in = _in_scale(ref.in_point, tabled, internal, s_x, gi, g)
                s_out = _out_scale(ref.out_point, tabled, internal, gi, g)
                f.write(struct.pack("<ff", s_in, s_out))
                if ref.kind == KIND_AV:
                    M, N = scales_to_MN(s_in * ref.pre / s_out)
                    f.write(struct.pack("<iB", M, N))
                    continue
                M_arr = np.empty(c_count, np.int32)
                N_arr = np.empty(c_count, np.uint8)
                for c in range(c_count):
                    ratio = s_in * float(rec.scales[c]) / s_out * ref.pre
                    M_arr[c], N_arr[c] = scales_to_MN(ratio)
                f.write(M_arr.astype("<i4").tobytes())
                f.write(N_arr.astype("u1").tobytes())
        for name in gn_names:
            ref = g.gn_layers[name]
            gamma, beta = gn_params[name]
            c_count = gamma.size
            nb = name.encode()
            in_kind = (P_INTERNAL if ref.in_point in g.internal_points
                       else P_TABLED)
            f.write(struct.pack("<H", len(nb)) + nb)
            f.write(struct.pack("<BBHBBH", KIND_GN, in_kind,
                                pidx[ref.in_point], P_INTERNAL,
                                pidx[ref.out_point], c_count))
            s_xhat = s_x  # x̂ 的 Q3.12 分辨率 = 2^-12（契约 §3）
            for gi in range(num_groups):
                s_in = _in_scale(ref.in_point, tabled, internal, s_x, gi, g)
                s_out = _out_scale(ref.out_point, tabled, internal, gi, g)
                f.write(struct.pack("<ff", s_in, s_out))
                Gc = np.empty(c_count, np.int32)
                Nc = np.empty(c_count, np.uint8)
                Bq = np.empty(c_count, np.int32)
                for c in range(c_count):
                    ratio = s_xhat * float(gamma[c]) / s_out
                    M, N = scales_to_MN(abs(ratio))
                    Gc[c] = -M if ratio < 0 else M
                    Nc[c] = N
                    Bq[c] = math.floor(float(beta[c]) / s_out + 0.5)
                f.write(Gc.astype("<i4").tobytes())
                f.write(Nc.astype("u1").tobytes())
                f.write(Bq.astype("<i4").tobytes())
    return n_layers


def register_observers(model, g, per_step: dict, step_box: list,
                        subsample: int) -> list:
    """按接线图在内部点注册统计 hook（fwd=输出 / pre=输入）。"""
    mods = dict(model.named_modules())
    handles = []

    def reg(module_name: str, kind: str, point: str) -> None:
        mod = mods[module_name]
        if kind == "fwd":
            def hook(_m, _i, out, _p=point):
                if torch.is_tensor(out):
                    update_percentile_stats(per_step, _p, step_box[0], out,
                                            max_samples=subsample)
            handles.append(mod.register_forward_hook(hook))
        else:
            def pre(_m, inp, _p=point):
                t = inp[0] if isinstance(inp, tuple) else inp
                if torch.is_tensor(t):
                    update_percentile_stats(per_step, _p, step_box[0], t,
                                            max_samples=subsample)
            handles.append(mod.register_forward_pre_hook(pre))

    for name, kind, point in g.observers:
        reg(name, kind, point)
    return handles


@torch.no_grad()
def _calibrate_internal(config: dict, unet_cfg: dict, tabled_raw: dict, g,
                        seeds: list[int], size: int, percentile: float,
                        subsample: int) -> dict[str, list[float]]:
    """在 fake-quant 轨迹上采集内部点统计（协议同 F4）+ 记录 DDIM x max|x|。"""
    model = load_handwritten_unet(config["model_id"], unet_cfg)
    n_steps = config["schedule"]["num_inference_steps"]
    n_groups = config["schedule"]["num_step_groups"]
    mixed = config.get("mixed_precision", {})
    int16_layers = tuple(mixed.get("int16_weight_layers", ()))
    quantize_weights_in_place(model, int16_layers)

    raw = {k: {int(gg): v for gg, v in groups.items()}
           for k, groups in tabled_raw.items()}
    ctx = FakeQuantContext(raw, n_steps, n_groups,
                           int16_weight_layers=int16_layers,
                           int16_act_layers=tuple(mixed.get(
                               "int16_act_layers", ())))
    apply_fake_quant(model, ctx)

    handles = register_observers(model, g, per_step := {}, [0], subsample)

    x_max = 0.0
    for seed in seeds:
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(n_steps)
        gen = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 3, size, size, generator=gen)
        for step_idx, t in enumerate(sched.timesteps):
            step_box[0] = step_idx
            x_max = max(x_max, x.abs().max().item())
            ctx.set_step(step_idx)
            noise = model(x, t)
            x = sched.step(noise, t, x).prev_sample
        print(f"seed {seed} 内部点标定完成", flush=True)
    for h in handles:
        h.remove()
    print(f"DDIM x 轨迹 max|x| = {x_max:.4f}（{len(seeds)} seeds × {n_steps} 步）",
          flush=True)
    merged = merge_step_stats(per_step, n_steps, n_groups)
    scales = finalize_scales(merged, percentile)
    return {k: [v for _, v in sorted(gr.items())] for k, gr in scales.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="F5 扩展导出：内部点 scale + M/N + GN 参数")
    p.add_argument("--quant-config", default="configs/quant_int8.json")
    p.add_argument("--calib-stats", default="artifacts/f4/calib-stats-ddim50-seq.json")
    p.add_argument("--bittrue-config", default="configs/bittrue.json")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/f4/export-v3"))
    p.add_argument("--table-suffix", required=True)
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    bt = json.loads(Path(args.bittrue_config).read_text(encoding="utf-8"))
    tabled_raw = json.loads(Path(args.calib_stats).read_text(encoding="utf-8"))
    n_groups = cfg["schedule"]["num_step_groups"]

    g = build_graph(unet_cfg)
    weights = {r.name: r for r in read_weights_bin(args.out_dir / "weights.bin")}

    # GN γ/β：导出器合法使用 PyTorch 侧权重生成导出文件；模拟器只消费导出包
    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    mods = dict(model.named_modules())
    gn_params = {}
    for name in sorted(g.gn_layers):
        mod = mods[name]
        gn_params[name] = (
            mod.weight.detach().to(torch.float32).numpy().copy(),
            mod.bias.detach().to(torch.float32).numpy().copy())
    n_gn = _write_norm_params(args.out_dir / "norm_params.bin",
                              sorted(g.gn_layers), mods)
    print(f"norm_params.bin：{n_gn} 个 GN", flush=True)
    del model

    t0 = time.time()
    internal = _calibrate_internal(
        cfg, unet_cfg, tabled_raw, g, bt["internal_calibration"]["seeds"],
        bt["internal_calibration"]["image_size"],
        bt["internal_calibration"]["percentile"],
        bt["internal_calibration"]["subsample"])
    print(f"内部点标定耗时 {time.time() - t0:.0f}s", flush=True)

    tabled = build_act_scales(tabled_raw)
    out = args.out_dir / f"requant_params_{args.table_suffix}.bin"
    n = _write_requant_params(out, g, tabled, internal, weights, n_groups,
                              bt["s_x"], gn_params)
    print(f"{out.name}：{n} 条层记录、{len(g.internal_points)} 个内部点",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
