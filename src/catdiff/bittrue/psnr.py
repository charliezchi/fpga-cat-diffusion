"""PSNR 对比（Task 5 机器观测）：两个采样目录逐 seed 图像 PSNR。

用法：
  uv run python src/catdiff/bittrue/psnr.py --a artifacts/f5/bittrue-ddim50 \
      --b artifacts/f4/int8-ddim50-mixed-seq --out artifacts/f5/psnr-vs-f4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 99.0 if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="逐 seed 图像 PSNR")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--num", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    rows = []
    missing = []
    for idx in range(args.num):
        name = f"seed{args.seed}_{idx:04d}.png"
        pa, pb = args.a / name, args.b / name
        if not pa.exists() or not pb.exists():
            missing.append(name)
            continue
        ia = np.asarray(Image.open(pa).convert("RGB"))
        ib = np.asarray(Image.open(pb).convert("RGB"))
        rows.append({"image": name, "psnr_db": round(psnr(ia, ib), 2)})
    if missing:
        print(f"跳过缺失 {len(missing)} 张: {missing[:5]}...")
        print(f"已对比 {len(rows)} 张")
    if not rows:
        return 1
    vals = [r["psnr_db"] for r in rows]
    summary = {"dir_a": str(args.a), "dir_b": str(args.b),
               "min": min(vals), "mean": round(float(np.mean(vals)), 2),
               "images": rows}
    text = json.dumps(summary, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
