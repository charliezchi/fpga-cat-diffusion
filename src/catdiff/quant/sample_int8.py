"""INT8 fake-quant 采样：DDIM-20 @256×256，产出画质关卡样本。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from catdiff.baseline.sampling import make_grid, tensor_to_pil
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import load_handwritten_unet
from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="INT8 fake-quant 采样（画质关卡）")
    p.add_argument("--quant-config", default="configs/quant_int8.json")
    p.add_argument("--calib-stats", default="artifacts/f4/calib-stats.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    raw = json.loads(Path(args.calib_stats).read_text(encoding="utf-8"))
    scales = {k: {int(g): v for g, v in groups.items()} for k, groups in raw.items()}

    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    n_steps = cfg["schedule"]["num_inference_steps"]
    ctx = FakeQuantContext(scales, n_steps, cfg["schedule"]["num_step_groups"])
    n = apply_fake_quant(model, ctx)
    print(f"fake-quant 施加 {n} 处")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for idx in range(args.num_samples):
        path = args.out_dir / f"seed{args.seed}_{idx:04d}.png"
        if path.exists():
            continue
        g = torch.Generator().manual_seed(args.seed + idx)
        x = torch.randn(1, 3, cfg["calibration"]["image_size"],
                        cfg["calibration"]["image_size"], generator=g)
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(n_steps)
        for step_idx, t in enumerate(sched.timesteps):
            ctx.set_step(step_idx)
            x = sched.step(model(x, t), t, x).prev_sample
        tensor_to_pil(x[0]).save(path)
        print(f"[{idx + 1}/{args.num_samples}] {path.name} 完成", flush=True)

    images = [Image.open(args.out_dir / f"seed{args.seed}_{idx:04d}.png")
              for idx in range(args.num_samples)]
    make_grid(images, 4).save(args.out_dir / "grid.png")
    (args.out_dir / "metadata.json").write_text(json.dumps({
        "backend": "handwritten-int8-fakequant",
        "quant_config": cfg, "seed": args.seed,
        "num_samples": args.num_samples,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
