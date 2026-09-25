import json
import struct

import numpy as np
import torch

from catdiff.export.ddim_table import build_ddim_table
from catdiff.export.film_table import build_film_vectors
from catdiff.export.weights_export import iter_quantized_layers
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import UNet2D
from catdiff.quant.weights import quantize_per_channel
from tests.test_unet import TINY_CONFIG


def test_ddim_table_matches_scheduler():
    table = build_ddim_table(num_inference_steps=20)
    assert len(table) == 20
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(20)
    for row, t in zip(table, sched.timesteps.tolist()):
        assert row["t"] == t
    # 首行 t 最大，末行 alpha_prev == 1.0（set_alpha_to_one）
    assert table[0]["t"] > table[-1]["t"]
    assert abs(table[-1]["alpha_prev"] - 1.0) < 1e-6


def test_ddim_table_50_steps():
    table = build_ddim_table(num_inference_steps=50)
    assert len(table) == 50
    assert table[0]["t"] > table[-1]["t"]


def test_iter_quantized_layers_covers_all_param_modules():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    layers = list(iter_quantized_layers(model))
    names = {name for name, _, _, _, _ in layers}
    import torch.nn as nn
    expect = {n for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))}
    assert names == expect  # 含全部注意力的 to_q/to_k/to_v/to_out.0
    for name, w_q, scales, bias, bits in layers:
        assert w_q.dtype == torch.int8 and bits == 8  # 缺省全 INT8
        assert scales.shape[0] == w_q.shape[0]


def test_iter_quantized_layers_mixed_bits():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    layers = dict((name, (w_q, bits)) for name, w_q, _, _, bits in
                  iter_quantized_layers(model, int16_weight_layers=("conv_in", "conv_out")))
    assert layers["conv_in"][1] == 16 and layers["conv_in"][0].dtype == torch.int16
    assert layers["conv_out"][1] == 16 and layers["conv_out"][0].dtype == torch.int16
    for name, (w_q, bits) in layers.items():
        if name not in ("conv_in", "conv_out"):
            assert bits == 8 and w_q.dtype == torch.int8


def test_film_vectors_match_resnet_injection():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    t_value = 250
    vecs = build_film_vectors(model, [t_value])  # {resnet_name: {t: vector}}
    # 任取一个 resnet，独立计算 time_emb_proj(silu(temb)) 比对
    from catdiff.model.time_embed import timestep_embedding
    name, table = next(iter(vecs.items()))
    resnet = dict(model.named_modules())[name]
    temb = model.time_embedding(
        model.time_proj(torch.tensor([float(t_value)])))
    expect = resnet.time_emb_proj(torch.nn.functional.silu(temb))[0]
    got = torch.tensor(table[t_value])
    assert torch.allclose(got, expect, atol=1e-5)


def _run_export(tmp_path, monkeypatch, tiny_handwritten_unet, suffix, extra=()):
    from catdiff.export import run_export as re_mod

    cfg = {
        "act_quant": {"percentile": 99.95},
        "schedule": {"num_inference_steps": int(suffix), "num_step_groups": 10},
        "mixed_precision": {"int16_weight_layers": ["conv_in", "conv_out"],
                            "int16_act_layers": ["conv_out"]},
        "model_id": "tiny", "unet_config": "unused.json",
    }
    cfg_path = tmp_path / f"cfg{suffix}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    unet_cfg_path = tmp_path / f"unet{suffix}.json"
    unet_cfg_path.write_text("{}", encoding="utf-8")
    cfg["unet_config"] = str(unet_cfg_path)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    stats_path = tmp_path / f"stats{suffix}.json"
    stats_path.write_text(json.dumps(
        {"conv_in": {str(g): 0.01 + g * 1e-3 for g in range(10)}}), encoding="utf-8")
    monkeypatch.setattr(re_mod, "load_handwritten_unet",
                        lambda mid, uc: tiny_handwritten_unet)
    out = tmp_path / "bundle"
    rc = re_mod.main(["--quant-config", str(cfg_path), "--calib-stats", str(stats_path),
                      "--out-dir", str(out), "--table-suffix", suffix, *extra])
    assert rc == 0
    return out


def test_run_export_int16_weight_bytes(tmp_path, monkeypatch, tiny_handwritten_unet):
    """CDW2 记录字节格式：conv_in 为首层，bits=16，载荷小端 int16 且与量化器一致。"""
    out = _run_export(tmp_path, monkeypatch, tiny_handwritten_unet, "20")
    raw = (out / "weights.bin").read_bytes()
    assert raw[:4] == b"CDW2"
    (n_layers,) = struct.unpack("<I", raw[4:8])
    model_modules = dict(tiny_handwritten_unet.named_modules())

    off = 8
    checked_int16 = 0
    for _ in range(n_layers):
        (nl,) = struct.unpack("<H", raw[off:off + 2]); off += 2
        name = raw[off:off + nl].decode(); off += nl
        (ndim,) = struct.unpack("<I", raw[off:off + 4]); off += 4
        shape = struct.unpack(f"<{ndim}I", raw[off:off + 4 * ndim]); off += 4 * ndim
        bits, has_bias = struct.unpack("<BB", raw[off:off + 2]); off += 2
        expect_bits = 16 if name.split(".")[0] in ("conv_in", "conv_out") else 8
        assert bits == expect_bits, name
        numel = int(np.prod(shape))
        off += 4 * shape[0]  # scales fp32
        if has_bias:
            off += 4 * shape[0]  # bias fp32
        if bits == 16:
            w_q, _ = quantize_per_channel(
                model_modules[name].weight.detach(), axis=0, bits=16)
            payload = np.frombuffer(raw[off:off + 2 * numel], dtype="<i2")
            assert np.array_equal(payload.reshape(shape), w_q.numpy()), name
            checked_int16 += 1
        off += numel * (bits // 8)  # payload
    assert checked_int16 == 2  # conv_in + conv_out
    assert off == len(raw)  # 记录读完即文件尾

    layers_json = json.loads((out / "layers.json").read_text(encoding="utf-8"))
    bits_by_name = {d["name"]: d["bits"] for d in layers_json["layers"]}
    assert bits_by_name["conv_in"] == 16 and bits_by_name["conv_out"] == 16
    assert sum(1 for b in bits_by_name.values() if b == 8) == n_layers - 2


def test_run_export_dual_schedule_tables(tmp_path, monkeypatch, tiny_handwritten_unet):
    out = _run_export(tmp_path, monkeypatch, tiny_handwritten_unet, "20")
    weights_first = (out / "weights.bin").read_bytes()
    # 第二档：--skip-weights 补导 50 步表，权重文件不得被触碰
    out2 = _run_export(tmp_path, monkeypatch, tiny_handwritten_unet, "50",
                       extra=("--skip-weights",))
    assert out2 == out
    assert (out / "weights.bin").read_bytes() == weights_first

    t20 = json.loads((out / "ddim_table_20.json").read_text(encoding="utf-8"))
    t50 = json.loads((out / "ddim_table_50.json").read_text(encoding="utf-8"))
    assert len(t20) == 20 and len(t50) == 50

    for suffix in ("20", "50"):
        hdr = struct.unpack("<II", (out / f"film_table_{suffix}.bin").read_bytes()[:8])
        assert hdr[0] == int(suffix) and hdr[1] > 0
        idx = json.loads((out / f"film_index_{suffix}.json").read_text(encoding="utf-8"))
        assert len(idx["timesteps"]) == int(suffix)

        scales_doc = json.loads(
            (out / f"act_scales_{suffix}.json").read_text(encoding="utf-8"))
        assert scales_doc["num_inference_steps"] == int(suffix)
        assert scales_doc["num_step_groups"] == 10
        conv_in_scales = scales_doc["scales"]["conv_in"]
        assert len(conv_in_scales) == 10 and all(s > 0 for s in conv_in_scales)