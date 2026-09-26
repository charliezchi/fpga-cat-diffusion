"""导出包解析（docs/bittrue-spec.md §7）：位真模型只从这里获取全部数据。

消费 artifacts/f4/export-v3 的：weights.bin（CDW2）、layers.json、
act_scales_<steps>.json、requant_params_<steps>.bin、norm_params.bin、
ddim_table_<steps>.json、film_table_<steps>.bin + film_index。
不从 PyTorch 侧补任何数据——导出包缺信息即报错（这验证导出包完备性）。
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from catdiff.bittrue.cdw2 import WeightRecord, read_weights_bin
from catdiff.bittrue.graph import KIND_AV, KIND_CONV, KIND_GN, P_X, \
    build_graph, is_fine, silu_lut_specs
from catdiff.bittrue.primitives import EXP_LOG2_DELTA, S_X, scales_to_MN

@dataclass
class LayerParams:
    """一个 wired 层逐组的 requant 常量与权重。"""
    name: str
    kind: int
    w: np.ndarray            # int8/int16 (O, ...) 或 (O, C)
    w_scales: np.ndarray     # f32[C]
    bias: np.ndarray | None
    M: list[np.ndarray]      # 每组 i32[C]
    N: list[np.ndarray]      # 每组 u8[C]
    b32: list[np.ndarray]    # 每组 i64[C]（acc 域偏置）


@dataclass
class GNParams:
    name: str
    in_point: str
    s_in: list[float]
    s_out: list[float]
    Gc: list[np.ndarray]
    Nc: list[np.ndarray]
    Bq: list[np.ndarray]
    eps_q: list[int]


@dataclass
class BittrueTables:
    unet_cfg: dict
    num_groups: int
    weights: dict[str, WeightRecord]
    layers: dict[str, LayerParams]
    gns: dict[str, GNParams]
    tabled_scales: dict[str, list[float]]     # s8 基准
    internal_scales: dict[str, list[float]]
    s16_eps: list[float]                      # 每组 eps 的 s16
    film_q32: dict[str, np.ndarray]           # resnet 名 → [steps][C] int64
    ddim_A: list[int]
    ddim_B: list[int]
    ddim_C: list[int]
    ddim_D: list[int]
    silu_luts: dict[tuple[str, str], list[np.ndarray]]  # (in,out) → 每组 256 项
    attn_Kexp: dict[str, list[int]]
    attn_Qe: int = 32
    attn_scale: float = 1.0
    x_clip_grid: int = 4096
    fine_div: int = 256                     # 内部细网格除数（契约 §4 v1.3：1024）
    film_timesteps: list[int] = field(default_factory=list)
    _g: object = None

    def point_scale(self, point: str, g: int) -> float:
        """点自身网格 scale：内部点（除 qkv）为细网格 = 标定值/fine_div（契约 §4）。"""
        src = (self.internal_scales if point in self.internal_scales
               else self.tabled_scales)
        v = src[point][g]
        return v / self.fine_div if is_fine(self._g, point) else v

    def group(self, step_idx: int, num_steps: int) -> int:
        per = num_steps // self.num_groups
        return min(step_idx // per, self.num_groups - 1)


def _point_scale(point: str, tabled: dict, internal: dict, g_idx: int, g,
                 fine_div: int = 256) -> float:
    """点自身网格 scale：内部点（除 qkv）细网格 = 标定值/fine_div（契约 §4 v1.3）。"""
    src = internal if point in internal else tabled
    v = src[point][g_idx]
    return v / fine_div if is_fine(g, point) else v


def _read_requant_params(path: Path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RQP1":
        raise ValueError(f"requant_params magic 错误: {data[:4]!r}")
    version, num_groups, num_layers, num_internal = struct.unpack_from(
        "<IIII", data, 4)
    assert version == 3, version
    off = 20
    internal: dict[str, list[float]] = {}
    for _ in range(num_internal):
        (nl,) = struct.unpack_from("<H", data, off)
        off += 2
        name = data[off:off + nl].decode()
        off += nl
        vals = np.frombuffer(data, "<f4", num_groups, off).copy()
        off += 4 * num_groups
        internal[name] = vals.tolist()
    layers: list[tuple] = []
    for _ in range(num_layers):
        (nl,) = struct.unpack_from("<H", data, off)
        off += 2
        name = data[off:off + nl].decode()
        off += nl
        kind, in_kind, in_idx, out_kind, out_idx, c_count = \
            struct.unpack_from("<BBHBBH", data, off)
        off += 8
        groups = []
        for _g in range(num_groups):
            s_in, s_out = struct.unpack_from("<ff", data, off)
            off += 8
            if kind == KIND_GN:
                Gc = np.frombuffer(data, "<i4", c_count, off).copy()
                off += 4 * c_count
                Nc = np.frombuffer(data, "u1", c_count, off).copy()
                off += c_count
                Bq = np.frombuffer(data, "<i4", c_count, off).copy()
                off += 4 * c_count
                groups.append((s_in, s_out, Gc, Nc, Bq))
            else:
                M = np.frombuffer(data, "<i4", c_count, off).copy()
                off += 4 * c_count
                N = np.frombuffer(data, "u1", c_count, off).copy()
                off += c_count
                groups.append((s_in, s_out, M, N))
        layers.append((name, kind, in_kind, in_idx, out_kind, out_idx, groups))
    assert off == len(data), f"requant_params 尾部未消费 {len(data) - off} 字节"
    return internal, layers, num_groups


def _read_norm_params(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"NRM1":
        raise ValueError(f"norm_params magic 错误: {data[:4]!r}")
    (count,) = struct.unpack_from("<I", data, 4)
    off = 8
    out = {}
    for _ in range(count):
        (nl,) = struct.unpack_from("<H", data, off)
        off += 2
        name = data[off:off + nl].decode()
        off += nl
        (c,) = struct.unpack_from("<I", data, off)
        off += 4
        gamma = np.frombuffer(data, "<f4", c, off).copy()
        off += 4 * c
        beta = np.frombuffer(data, "<f4", c, off).copy()
        off += 4 * c
        out[name] = (gamma, beta)
    return out


def _read_film_table(path: Path, resnet_cs: list[int]):
    """变长布局：[step][resnet_i][C_i]，resnet 顺序 = film_index.resnet_names。"""
    with open(path, "rb") as f:
        data = f.read()
    steps, n_resnet = struct.unpack_from("<II", data, 0)
    assert n_resnet == len(resnet_cs)
    off = 8
    out: dict[str, np.ndarray] = {}
    # 逐 resnet 收集（外层 step、内层 resnet 交错写入）
    per_resnet: list[list[np.ndarray]] = [[] for _ in range(n_resnet)]
    for _si in range(steps):
        for ri in range(n_resnet):
            c = resnet_cs[ri]
            per_resnet[ri].append(
                np.frombuffer(data, "<f4", c, off).copy())
            off += 4 * c
    assert off == len(data), f"film_table 尾部未消费 {len(data) - off} 字节"
    return steps, per_resnet


def load_export(export_dir: str | Path, tier: str,
                bt_config: dict, unet_cfg: dict) -> BittrueTables:
    d = Path(export_dir)
    files = bt_config["tiers"][tier]
    fine_div = int(bt_config.get("fine_div", 256))
    g = build_graph(unet_cfg)

    weights = {r.name: r for r in read_weights_bin(d / "weights.bin")}
    layers_json = json.loads((d / "layers.json").read_text(encoding="utf-8"))
    json_names = [l["name"] for l in layers_json["layers"]]
    assert json_names == list(weights), "weights.bin 与 layers.json 层序不一致"

    act = json.loads((d / files["act_scales"]).read_text(encoding="utf-8"))
    num_groups = act["num_step_groups"]
    assert act["num_inference_steps"] == int(tier)
    tabled_scales = {k: list(v) for k, v in act["scales"].items()}
    assert set(tabled_scales) == set(g.tabled_points), \
        "act_scales 点清单与接线图不一致"

    internal, rq_layers, num_groups_rq = _read_requant_params(
        d / files["requant_params"])
    assert num_groups_rq == num_groups
    assert set(internal) == set(g.internal_points), "内部点清单与接线图不一致"

    # wired 层
    layers: dict[str, LayerParams] = {}
    for name, kind, _ik, _ii, _ok, _oi, groups in rq_layers:
        if kind == KIND_GN:
            continue  # GN 记录在下方 gns 段处理
        ref = g.all_layers()[name]
        assert ref.kind == kind, (name, ref.kind, kind)
        rec = weights.get(name)  # av_requant 伪层无权重记录
        M_list, N_list, b32_list = [], [], []
        for gi, payload in enumerate(groups):
            s_in, s_out, arrs = payload[0], payload[1], payload[2:]
            if kind == KIND_AV:
                M_list.append(arrs[0])
                N_list.append(arrs[1])
                b32_list.append(np.zeros(1, np.int64))
                continue
            M_arr, N_arr = arrs
            M_list.append(M_arr)
            N_list.append(N_arr)
            if rec.bias is not None:
                b32 = np.array([
                    math.floor(float(b) / (s_in * float(sw)) + 0.5)
                    for b, sw in zip(rec.bias, rec.scales)], dtype=np.int64)
            else:
                b32 = np.zeros(rec.scales.size, np.int64)
            b32_list.append(b32)
        w = rec.payload.reshape(rec.shape) if rec is not None else None
        layers[name] = LayerParams(name, kind, w,
                                   rec.scales if rec is not None
                                   else np.ones(1, np.float32),
                                   rec.bias if rec is not None else None,
                                   M_list, N_list, b32_list)

    # GN（Gc/Nc/Bq + eps_q）
    norm = _read_norm_params(d / "norm_params.bin")
    assert set(norm) == set(g.gn_layers), "norm_params GN 清单与接线图不一致"
    gns: dict[str, GNParams] = {}
    for name, kind, _ik, _ii, _ok, _oi, groups in rq_layers:
        if kind != KIND_GN:
            continue
        ref = g.gn_layers[name]
        gamma, _beta = norm[name]
        s_in_l, s_out_l, Gc_l, Nc_l, Bq_l, eps_l = [], [], [], [], [], []
        for s_in, s_out, Gc, Nc, Bq in groups:
            s_in_l.append(s_in)
            s_out_l.append(s_out)
            Gc_l.append(Gc.astype(np.int64))
            Nc_l.append(Nc)
            Bq_l.append(Bq.astype(np.int64))
            # eps/var 统一 INT8 等价格（契约 §3 v1.3）：细格输入 s_in 为细格值，
            # 粗格 = s_in·fine_div
            s_in_int8 = s_in * fine_div if is_fine(g, ref.in_point) else s_in
            eps_l.append(int(round(1e-6 / (s_in_int8 * s_in_int8) * (1 << 16))))
        gns[name] = GNParams(name, ref.in_point, s_in_l, s_out_l,
                             Gc_l, Nc_l, Bq_l, eps_l)

    # FiLM 表（fp32 → conv1 acc 域 int64；各 resnet 的 C 不同，变长解析）
    fidx = json.loads((d / files["film_index"]).read_text(encoding="utf-8"))
    resnet_names = fidx["resnet_names"]
    steps = fidx["timesteps"]
    resnet_cs = [weights[f"{r}.conv1"].scales.size for r in resnet_names]
    film_steps, per_resnet = _read_film_table(d / files["film_table"],
                                              resnet_cs)
    assert film_steps == len(steps)
    film_q32: dict[str, np.ndarray] = {}
    for ri, rname in enumerate(resnet_names):
        conv1 = layers[f"{rname}.conv1"]
        silu1_point = g.resnets[rname].conv1_in
        c0 = resnet_cs[ri]
        per_step = np.empty((film_steps, c0), np.int64)
        for si in range(film_steps):
            gi = min(si // (len(steps) // num_groups), num_groups - 1)
            s_in = _point_scale(silu1_point, tabled_scales, internal, gi, g,
                                fine_div)
            vec = per_resnet[ri][si]
            per_step[si] = [
                math.floor(float(v) / (s_in * float(sw)) + 0.5)
                for v, sw in zip(vec, conv1.w_scales)]
        film_q32[rname] = per_step

    # DDIM 定点系数（契约 §6）
    ddim = json.loads((d / files["ddim_table"]).read_text(encoding="utf-8"))
    s16 = [tabled_scales["conv_out"][gi] * 127.0 / 32767.0
           for gi in range(num_groups)]
    A_l, B_l, C_l, D_l = [], [], [], []
    for si, row in enumerate(ddim):
        gi = min(si // (len(ddim) // num_groups), num_groups - 1)
        r = s16[gi] / S_X
        a_t, a_p = row["alpha_t"], row["alpha_prev"]
        A_l.append(math.floor(2 ** 23 / math.sqrt(a_t) + 0.5))
        B_l.append(math.floor(2 ** 23 * math.sqrt(1 - a_t)
                              / math.sqrt(a_t) * r + 0.5))
        C_l.append(math.floor(2 ** 30 * math.sqrt(a_p) + 0.5))
        D_l.append(math.floor(2 ** 30 * math.sqrt(1 - a_p) * r + 0.5))

    # SiLU LUT（每 (输入点, 输出点, 组) 一张）
    silu_luts = {}
    for in_p, out_p in silu_lut_specs(g):
        silu_luts[(in_p, out_p)] = [
            _gen_silu_lut_i(in_p, out_p, gi, internal, tabled_scales, g,
                            fine_div)
            for gi in range(num_groups)]

    # 注意力 exp 地址缩放
    dim_head = (unet_cfg["block_out_channels"][-1]
                if unet_cfg.get("attention_head_dim") is None
                else unet_cfg["attention_head_dim"])
    attn_scale = dim_head ** -0.5
    Kexp = {}
    for a in g.attentions:
        qkv = g.attentions[a].qkv
        Kexp[a] = [
            math.floor((tabled_scales[qkv][gi] if qkv in tabled_scales
                        else internal[qkv][gi]) ** 2 * attn_scale
                       * 2.0 ** -EXP_LOG2_DELTA * (1 << 32) + 0.5)
            for gi in range(num_groups)]

    return BittrueTables(
        unet_cfg=unet_cfg, num_groups=num_groups, weights=weights,
        layers=layers, gns=gns, tabled_scales=tabled_scales,
        internal_scales=internal, s16_eps=s16, film_q32=film_q32,
        ddim_A=A_l, ddim_B=B_l, ddim_C=C_l, ddim_D=D_l,
        silu_luts=silu_luts, attn_Kexp=Kexp, attn_Qe=32,
        attn_scale=attn_scale, x_clip_grid=bt_config["x_clip_grid"],
        fine_div=fine_div, film_timesteps=steps, _g=g)


def _gen_silu_lut_i(in_point, out_point, gi, internal, tabled, g,
                    fine_div=256):
    from catdiff.bittrue.primitives import gen_silu_lut
    s_fine = _point_scale(out_point, tabled, internal, gi, g, fine_div)
    # 输入恒为 GN 输出（细网格点）：粗地址格 = 细格 × fine_div（契约 §5.2 v1.3）
    s_coarse = _point_scale(in_point, tabled, internal, gi, g,
                            fine_div) * fine_div
    return gen_silu_lut(s_coarse, s_fine)
