"""一键导出：weights.bin / layers.json / ddim_table.json / film_table.bin（契约 §3）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from catdiff.export.ddim_table import build_ddim_table
from catdiff.export.film_table import build_film_vectors
from catdiff.export.weights_export import iter_quantized_layers
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import load_handwritten_unet

_MAGIC = b"CDW1"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PTQ INT8 硬件导出")
    p.add_argument("--quant-config", default="configs/quant_int8.json")
    p.add_argument("--calib-stats", default="artifacts/f4/calib-stats.json")
    p.add_argument("--out-dir", type=Path, default=Path("export"))
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    calib_raw = Path(args.calib_stats).read_bytes()
    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layers = list(iter_quantized_layers(model))
    calib_hash = hashlib.sha256(calib_raw).hexdigest()[:16]

    # weights.bin：magic + 层数 + 逐层 {name_len, name, ndim, shape, scales, payload}
    with open(args.out_dir / "weights.bin", "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", len(layers)))
        for name, w_q, scales, bias in layers:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            w = w_q.numpy()
            f.write(struct.pack("<I", w.ndim))
            f.write(struct.pack(f"<{w.ndim}I", *w.shape))
            has_bias = 1 if bias is not None else 0
            f.write(struct.pack("<B", has_bias))
            f.write(scales.numpy().astype("<f4").tobytes())
            if has_bias:
                f.write(bias.numpy().astype("<f4").tobytes())
            f.write(w.astype("<i1").tobytes())

    # layers.json：算子级描述符（含量化点与 scale 引用）
    desc = []
    for idx, (name, w_q, scales, bias) in enumerate(layers):
        desc.append({
            "index": idx, "name": name,
            "op": "conv2d" if w_q.ndim == 4 else "linear",
            "shape": list(w_q.shape),
            "scale_ref": f"weights.bin#{idx}",
        })
    (args.out_dir / "layers.json").write_text(
        json.dumps({"calib_hash": calib_hash, "layers": desc}, indent=1),
        encoding="utf-8")

    # ddim_table.json
    n_steps = cfg["schedule"]["num_inference_steps"]
    (args.out_dir / "ddim_table.json").write_text(
        json.dumps(build_ddim_table(n_steps), indent=1), encoding="utf-8")

    # film_table.bin：[20][resnet_idx][out_channels] fp32，resnet 顺序 = layers.json 中
    # time_emb_proj 出现顺序
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(n_steps)
    film = build_film_vectors(model, sched.timesteps.tolist())
    resnet_names = list(film.keys())
    with open(args.out_dir / "film_table.bin", "wb") as f:
        f.write(struct.pack("<II", n_steps, len(resnet_names)))
        for t in sched.timesteps.tolist():
            for name in resnet_names:
                f.write(np.array(film[name][t], dtype="<f4").tobytes())
    (args.out_dir / "film_index.json").write_text(
        json.dumps({"timesteps": sched.timesteps.tolist(),
                    "resnet_names": resnet_names}, indent=1), encoding="utf-8")

    print(f"导出完成：{len(layers)} 层，{len(resnet_names)} 个 resnet 的 FiLM 表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
