"""位真 DDIM 采样（Task 5）：seed=100 断点续跑，输出 artifacts/f5/bittrue-ddim{50,20}/。

用法：
  uv run python src/catdiff/bittrue/sample.py --tier 50 \
      --out-dir artifacts/f5/bittrue-ddim50 --num-samples 20
  uv run python src/catdiff/bittrue/sample.py --tier 20 \
      --out-dir artifacts/f5/bittrue-ddim20 --num-samples 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from catdiff.baseline.sampling import make_grid, tensor_to_pil
from catdiff.bittrue.loader import load_export
from catdiff.bittrue.model import BittrueUNet, load_bittrue_unet
from catdiff.bittrue.primitives import S_X, ddim_update, x_to_pixel
from catdiff.bittrue.graph import build_graph


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="位真定点 DDIM 采样")
    p.add_argument("--export-dir", default="artifacts/f4/export-v3")
    p.add_argument("--bittrue-config", default="configs/bittrue.json")
    p.add_argument("--unet-config",
                   default="docs/reference/unet-config-ddpm-cat-256.json")
    p.add_argument("--tier", choices=["50", "20"], default="50")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args(argv)

    bt_cfg = json.loads(Path(args.bittrue_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(args.unet_config).read_text(encoding="utf-8"))
    tables = load_export(args.export_dir, args.tier, bt_cfg, unet_cfg)
    model = BittrueUNet(tables, build_graph(unet_cfg))
    ddim_A, ddim_B = tables.ddim_A, tables.ddim_B
    ddim_C, ddim_D = tables.ddim_C, tables.ddim_D
    n_steps = len(ddim_A)
    size = 256
    x_clip = tables.x_clip_grid

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    x_max_overall = 0.0
    for idx in range(args.num_samples):
        path = args.out_dir / f"seed{args.seed}_{idx:04d}.png"
        if path.exists():
            continue
        g = torch.Generator().manual_seed(args.seed + idx)
        x0 = torch.randn(1, 3, size, size, generator=g)  # 与 F4 批次同种子同噪声
        x = np.clip(np.round(x0.numpy() / S_X), -32768, 32767).astype(np.int64)
        for step_idx in range(n_steps):
            eps = model.forward(torch.from_numpy(x.astype(np.int32)),
                                step_idx).numpy().astype(np.int64)
            x = ddim_update(x, eps, ddim_A[step_idx], ddim_B[step_idx],
                            ddim_C[step_idx], ddim_D[step_idx])
            xm = int(np.abs(x).max())
            x_max_overall = max(x_max_overall, xm)
            if xm > 8 * x_clip:  # s_x=2^-12 的表示域 ±8（契约 §1）
                raise ValueError(f"seed {idx} step {step_idx}: |x|={xm} 超出 "
                                 f"Q3.12 表示域，s_x 论证失效")
        pixels = np.vectorize(x_to_pixel)(x[0]).astype(np.uint8)
        Image.fromarray(np.transpose(pixels, (1, 2, 0))).save(path)
        print(f"[{idx + 1}/{args.num_samples}] {path.name} 完成", flush=True)

    images = [Image.open(args.out_dir / f"seed{args.seed}_{idx:04d}.png")
              for idx in range(args.num_samples)]
    make_grid(images, 4).save(args.out_dir / "grid.png")
    (args.out_dir / "metadata.json").write_text(json.dumps({
        "backend": "bittrue-int-sim", "tier": args.tier, "seed": args.seed,
        "num_samples": args.num_samples,
        "x_max_grid": x_max_overall, "x_clip_grid": x_clip,
        "elapsed_s": round(time.time() - t_start, 1),
    }, indent=2), encoding="utf-8")
    print(f"完成：{args.out_dir}（max|x|={x_max_overall}）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
