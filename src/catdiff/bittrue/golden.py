"""黄金向量包导出（Task 6，docs/bittrue-spec.md §8）。

用法：
  uv run python src/catdiff/bittrue/golden.py --out-dir artifacts/f5/golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from catdiff.bittrue.graph import build_graph
from catdiff.bittrue.loader import load_export
from catdiff.bittrue.model import BittrueUNet
from catdiff.bittrue.primitives import S_X, ddim_update, x_to_pixel


def _save_tensor(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if arr.dtype == np.int64:  # 统一落盘 int8/int16/int32 小端（契约 §8）
        arr = arr.astype(np.int32)
    path.write_bytes(np.ascontiguousarray(arr).tobytes(order="C"))


def _dtype_of(arr: np.ndarray) -> str:
    return {np.dtype(np.int8): "int8", np.dtype(np.int16): "int16",
            np.dtype(np.int32): "int32"}[arr.dtype if arr.dtype != np.int64
                                        else np.dtype(np.int32)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _keep_point(name: str, shape, full: bool) -> bool:
    """体积控制：step0 全存；其余步只存空间 ≤64×64 与首尾边界点。"""
    if full or name in ("conv_in", "conv_out"):
        return True
    hw = 1
    for d in shape[-2:]:
        hw *= int(d)
    return hw <= 64 * 64


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="F5 黄金向量包导出")
    p.add_argument("--export-dir", default="artifacts/f4/export-v3")
    p.add_argument("--bittrue-config", default="configs/bittrue.json")
    p.add_argument("--unet-config",
                   default="docs/reference/unet-config-ddpm-cat-256.json")
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/f5/golden"))
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args(argv)

    bt_cfg = json.loads(Path(args.bittrue_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(args.unet_config).read_text(encoding="utf-8"))
    g = build_graph(unet_cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"format": "bittrue-golden-v1", "s_x": S_X,
                "points": list(g.tabled_points),
                "internal_points": list(g.internal_points),
                "tiers": {}, "seed": args.seed}

    for tier in ("50", "20"):
        tables = load_export(args.export_dir, tier, bt_cfg, unet_cfg)
        model = BittrueUNet(tables, build_graph(unet_cfg))
        n_steps = len(tables.ddim_A)
        steps_sel = bt_cfg["golden_steps"][tier]
        out = args.out_dir / f"e2e/{tier}"
        (out / "eps").mkdir(parents=True, exist_ok=True)

        # ---- 端到端：初始噪声 → 每步 eps → 像素 ----
        gnr = torch.Generator().manual_seed(args.seed)
        x0 = torch.randn(1, 3, 256, 256, generator=gnr)
        x = np.clip(np.round(x0.numpy() / S_X), -32768, 32767).astype(np.int64)
        _save_tensor(out / "x_init.int16.bin", x.astype(np.int16))
        coeffs = []
        eps_paths = []
        t_start = time.time()
        for step_idx in range(n_steps):
            eps = model.forward(torch.from_numpy(x.astype(np.int32)),
                                step_idx).numpy().astype(np.int64)
            ep = out / "eps" / f"step{step_idx:04d}.int16.bin"
            if tier == "50" or step_idx in steps_sel:  # 20 步档只存选中步
                _save_tensor(ep, eps.astype(np.int16))
                eps_paths.append(str(ep.relative_to(args.out_dir)))
            coeffs.append({"step": step_idx, "A_q8_23": tables.ddim_A[step_idx],
                           "B_q8_23": tables.ddim_B[step_idx],
                           "C_q2_30": tables.ddim_C[step_idx],
                           "D_q2_30": tables.ddim_D[step_idx]})
            x = ddim_update(x, eps, tables.ddim_A[step_idx],
                            tables.ddim_B[step_idx], tables.ddim_C[step_idx],
                            tables.ddim_D[step_idx])
        _save_tensor(out / "x_final.int16.bin", x.astype(np.int16))
        pixels = np.vectorize(x_to_pixel)(x[0]).astype(np.uint8)
        img = Image.fromarray(np.transpose(pixels, (1, 2, 0)))
        img.save(out / "pixels.png")
        _save_tensor(out / "pixels.uint8.bin", pixels)
        (out / "ddim_coeffs_q.json").write_text(json.dumps(coeffs, indent=1),
                                                encoding="utf-8")
        print(f"[{tier}档] 端到端 {n_steps} 步完成 "
              f"({time.time() - t_start:.0f}s)", flush=True)

        # ---- 逐层向量 ----
        for step_idx in steps_sel:
            full = (tier == "50" and step_idx == 0)
            sd = args.out_dir / f"layers/{tier}/step{step_idx:04d}"
            sd.mkdir(parents=True, exist_ok=True)
            trace, trace_in = {}, {}
            x_in = torch.from_numpy(
                np.clip(np.round(x0.numpy() / S_X), -32768, 32767)
                .astype(np.int32))
            model.forward(x_in, step_idx, trace=trace, trace_in=trace_in)
            shapes = {}
            for point in g.tabled_points:
                if point not in trace:
                    continue
                if not _keep_point(point, trace[point].shape, full):
                    continue
                _save_tensor(sd / f"{point}.in.bin", trace_in[point])
                _save_tensor(sd / f"{point}.out.bin", trace[point])
                shapes[point] = {
                    "in": {"shape": list(trace_in[point].shape),
                           "dtype": _dtype_of(trace_in[point])},
                    "out": {"shape": list(trace[point].shape),
                            "dtype": _dtype_of(trace[point])}}
            (sd / "shapes.json").write_text(json.dumps(shapes, indent=1),
                                            encoding="utf-8")
            print(f"[{tier}档] step{step_idx} 逐层向量 {len(shapes)} 点"
                  f"（{'全量' if full else '裁剪'}）", flush=True)

        manifest["tiers"][tier] = {
            "golden_steps": steps_sel, "num_steps": n_steps,
            "full_dump_step": "0" if tier == "50" else None}

    # ---- LUT 归档（silu / exp / rsqrt，主档 step0 组）----
    from catdiff.bittrue.primitives import gen_exp_lut, gen_rsqrt_lut, \
        gen_silu_lut
    lut_dir = args.out_dir / "luts"
    lut_dir.mkdir(parents=True, exist_ok=True)
    tables = load_export(args.export_dir, "50", bt_cfg, unet_cfg)
    for (inp, outp), luts in tables.silu_luts.items():
        _save_tensor(lut_dir / f"silu_{inp}__{outp}__g0.int8.bin", luts[0])
    _save_tensor(lut_dir / "exp_4096xu16.bin", gen_exp_lut().astype(np.int16))
    lut0, lut1 = gen_rsqrt_lut()
    _save_tensor(lut_dir / "rsqrt_even.int32.bin", lut0.astype(np.int32))
    _save_tensor(lut_dir / "rsqrt_odd.int32.bin", lut1.astype(np.int32))

    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")

    # ---- checksums ----
    with open(args.out_dir / "checksums.txt", "w", encoding="utf-8") as f:
        for path in sorted(args.out_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.txt":
                f.write(f"{_sha256(path)}  {path.relative_to(args.out_dir)}\n")
    print(f"黄金向量包完成 -> {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
