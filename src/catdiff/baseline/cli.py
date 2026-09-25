"""采样命令行入口。用法见 README。支持断点续跑：已存在的样本 PNG 直接跳过。"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import diffusers
import torch
from PIL import Image

from .sampling import load_ddim, load_unet, make_grid, sample_batch, tensor_to_pil
from catdiff.model.ablate import strip_attention
from catdiff.model.ddim import DDIMScheduler as HwDDIMScheduler
from catdiff.model.unet import as_diffusers_output, load_handwritten_unet


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DDIM(eta=0) fp32 基线采样")
    p.add_argument("--model-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--backend", choices=["diffusers", "handwritten"],
                   default="diffusers")
    p.add_argument("--config", type=Path, default=None,
                   help="handwritten 后端所需的 UNet config json")
    p.add_argument("--strip-attention", choices=["none", "all", "keep-mid"],
                   default="none", help="注意力消融（仅 handwritten 后端）")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.strip_attention != "none" and args.backend != "handwritten":
        p.error("--strip-attention 仅 handwritten 后端支持")
    if args.backend == "handwritten":
        if args.config is None:
            p.error("--backend handwritten 需要 --config")
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        hw_model = load_handwritten_unet(args.model_id, cfg)
        if args.strip_attention != "none":
            n = strip_attention(hw_model, keep_mid=args.strip_attention == "keep-mid")
            print(f"注意力消融：{args.strip_attention}，替换 {n} 处")
        model = as_diffusers_output(hw_model)
        scheduler = HwDDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(args.num_inference_steps)
    else:
        model = load_unet(args.model_id)
        scheduler = load_ddim(args.model_id, args.num_inference_steps)

    t0 = time.time()
    generated = 0
    for idx in range(args.num_samples):
        path = args.out_dir / f"seed{args.seed}_{idx:04d}.png"
        if path.exists():
            continue
        img = sample_batch(
            model, scheduler,
            num_samples=1, image_size=args.image_size, seed=args.seed + idx,
        )
        tensor_to_pil(img[0]).save(path)
        generated += 1
        print(f"[{idx + 1}/{args.num_samples}] {path.name} 完成", flush=True)
    elapsed = time.time() - t0

    images = [
        Image.open(args.out_dir / f"seed{args.seed}_{idx:04d}.png")
        for idx in range(args.num_samples)
    ]
    make_grid(images, args.cols).save(args.out_dir / "grid.png")

    (args.out_dir / "config.json").write_text(
        json.dumps(
            model.config if isinstance(model.config, dict) else vars(model.config),
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "image_size": args.image_size,
                "num_inference_steps": args.num_inference_steps,
                "eta": 0.0,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "generated_this_run": generated,
                "elapsed_s": round(elapsed, 1),
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "python": platform.python_version(),
                "backend": args.backend,
                "strip_attention": args.strip_attention,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
