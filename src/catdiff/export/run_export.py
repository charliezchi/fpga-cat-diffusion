"""一键导出（契约 §3，v3）：weights.bin / layers.json / DDIM 表 / FiLM 表 / 激活 scale 表。

用法（v3 双档）：
  # 主档 DDIM-50：完整导出（权重 + 50 步表）
  run_export.py --quant-config configs/quant_int8.json \
      --calib-stats artifacts/f4/calib-stats-ddim50-seq.json \
      --out-dir artifacts/f4/export-v3 --table-suffix 50
  # 预览档 DDIM-20：仅补导 20 步表（权重与档位无关，--skip-weights）
  run_export.py --quant-config configs/quant_int8_ddim20.json \
      --calib-stats artifacts/f4/calib-stats-ddim20-seq.json \
      --out-dir artifacts/f4/export-v3 --table-suffix 20 --skip-weights

weights.bin 为 v2 格式（magic "CDW2"）：每记录含 bits 字段（8/16），
INT8 层载荷 <i1、INT16 层载荷 <i2（均小端）。与步数无关，双档共用一份。
档位相关文件（DDIM 表 / FiLM 表 / 激活 scale 表）以 --table-suffix 后缀区分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

from catdiff.export.ddim_table import build_ddim_table
from catdiff.export.film_table import build_film_vectors
from catdiff.export.weights_export import iter_quantized_layers
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import load_handwritten_unet

_MAGIC = b"CDW2"

_NP_PAYLOAD = {8: "<i1", 16: "<i2"}


def _table_names(suffix: str | None) -> dict:
    """档位相关文件的命名；无 suffix 时保持不带后缀的旧名。"""
    s = f"_{suffix}" if suffix else ""
    return {
        "ddim": f"ddim_table{s}.json",
        "film": f"film_table{s}.bin",
        "film_index": f"film_index{s}.json",
        "act_scales": f"act_scales{s}.json",
    }


def build_act_scales(raw: dict) -> dict:
    """标定统计 {layer: {group: scale}} → {layer: [按组号升序的 scale 列表]}。"""
    return {name: [v for _, v in sorted(groups.items(), key=lambda kv: int(kv[0]))]
            for name, groups in raw.items()}


def write_weights_bin(path: Path, layers: list) -> None:
    """weights.bin（CDW2）：magic + 层数 + 逐层记录。"""
    with open(path, "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", len(layers)))
        for name, w_q, scales, bias, bits in layers:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            w = w_q.numpy()
            f.write(struct.pack("<I", w.ndim))
            f.write(struct.pack(f"<{w.ndim}I", *w.shape))
            f.write(struct.pack("<BB", bits, 1 if bias is not None else 0))
            f.write(scales.numpy().astype("<f4").tobytes())
            if bias is not None:
                f.write(bias.numpy().astype("<f4").tobytes())
            f.write(w.astype(_NP_PAYLOAD[bits]).tobytes())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PTQ 混合精度硬件导出（v3 契约）")
    p.add_argument("--quant-config", default="configs/quant_int8.json")
    p.add_argument("--calib-stats", default="artifacts/f4/calib-stats.json")
    p.add_argument("--out-dir", type=Path, default=Path("export"))
    p.add_argument("--table-suffix", default=None,
                   help="档位相关文件名后缀（如 50 / 20）；缺省不带后缀")
    p.add_argument("--skip-weights", action="store_true",
                   help="跳过 weights.bin / layers.json（多档位补导表时用）")
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    calib_raw = Path(args.calib_stats).read_bytes()
    calib_stats = json.loads(calib_raw)
    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = _table_names(args.table_suffix)

    int16_layers = tuple(cfg.get("mixed_precision", {}).get("int16_weight_layers", ()))
    layers = list(iter_quantized_layers(model, int16_layers))
    calib_hash = hashlib.sha256(calib_raw).hexdigest()[:16]

    if not args.skip_weights:
        write_weights_bin(args.out_dir / "weights.bin", layers)

        # layers.json：算子级描述符（含量化点、位宽与 scale 引用）
        desc = []
        for idx, (name, w_q, scales, bias, bits) in enumerate(layers):
            desc.append({
                "index": idx, "name": name,
                "op": "conv2d" if w_q.ndim == 4 else "linear",
                "shape": list(w_q.shape),
                "bits": bits,
                "scale_ref": f"weights.bin#{idx}",
            })
        (args.out_dir / "layers.json").write_text(
            json.dumps({"calib_hash": calib_hash, "layers": desc}, indent=1),
            encoding="utf-8")

    # ddim_table[_suffix].json
    n_steps = cfg["schedule"]["num_inference_steps"]
    (args.out_dir / names["ddim"]).write_text(
        json.dumps(build_ddim_table(n_steps), indent=1), encoding="utf-8")

    # film_table[_suffix].bin：[steps][resnet_idx][out_channels] fp32，resnet 顺序 =
    # layers.json 中 time_emb_proj 出现顺序
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(n_steps)
    film = build_film_vectors(model, sched.timesteps.tolist())
    resnet_names = list(film.keys())
    with open(args.out_dir / names["film"], "wb") as f:
        f.write(struct.pack("<II", n_steps, len(resnet_names)))
        for t in sched.timesteps.tolist():
            for name in resnet_names:
                f.write(np.array(film[name][t], dtype="<f4").tobytes())
    (args.out_dir / names["film_index"]).write_text(
        json.dumps({"timesteps": sched.timesteps.tolist(),
                    "resnet_names": resnet_names}, indent=1), encoding="utf-8")

    # act_scales[_suffix].json：激活 scale 表（层 → 步组有序 scale 列表）
    n_groups = cfg["schedule"]["num_step_groups"]
    (args.out_dir / names["act_scales"]).write_text(json.dumps({
        "calib_hash": calib_hash,
        "num_inference_steps": n_steps,
        "num_step_groups": n_groups,
        "scales": build_act_scales(calib_stats),
    }, indent=1), encoding="utf-8")

    n16 = sum(1 for _, _, _, _, bits in layers if bits == 16)
    print(f"导出完成：{len(layers)} 层（INT16 权重 {n16} 层），"
          f"{len(resnet_names)} 个 resnet 的 {n_steps} 步 FiLM 表 -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
