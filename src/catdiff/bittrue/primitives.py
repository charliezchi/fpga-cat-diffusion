"""位真整数原语库（docs/bittrue-spec.md §0-§6 的可执行形式）。

全部运算在整数域（torch/numpy 整数或 float64 容器，容器内值恒为整数，
与 RTL 逐位等价——float64 尾数 53 位 > 本网络最大精确和 ~2^32）。
scale→M/N 的离线换算是唯一允许 float 的地方（单独函数，附 float 对照测试）。
注释标出 RTL 对应物。
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

QMAX = {8: 127, 16: 32767}
QMIN = {8: -128, 16: -32768}

# LUT 参数（configs/bittrue.json 的默认值，契约 §5.2/§5.3/§3）
EXP_ENTRIES = 4096
EXP_LOG2_DELTA = -8          # Δ = 2^-8
EXP_FRAC = 14                # Q1.14
RSQRT_ENTRIES = 2048         # 每奇偶各一份
RSQRT_MANT_BITS = 11         # 地址 = mantissa >> 11
RSQRT_VAR_MIN_Q16 = 256      # var_x 下限 2^-8
RSQRT_OUT_FRAC = 14          # inv_q14
XHAT_FRAC = 12               # x̂ Q3.12
S_X = 2.0 ** -12             # DDIM 状态 x 的定点分辨率


def sat_int(v, bits: int = 8):
    """饱和到对称域 [-qmax-1, qmax]（RTL: 位截断 + 饱和逻辑）。"""
    lo, hi = QMIN[bits], QMAX[bits]
    if isinstance(v, np.ndarray):
        return np.clip(v, lo, hi)
    return max(lo, min(hi, int(v)))


def requant(acc, M, N, bits: int = 8):
    """y = sat((acc·M + 2^(N-1)) >> N)（契约 §2.2）。

    acc/M 可为标量或 ndarray（广播）；acc·M 用 int64（RTL: 乘加器位宽）。
    算术右移实现 round-half-up，负数行为（-1.5→-1 等）有测试钉死。
    """
    acc = np.asarray(acc, dtype=np.int64)
    M = np.asarray(M, dtype=np.int64)
    if N <= 0:
        shifted = acc * M
    else:
        shifted = (acc * M + (1 << (N - 1))) >> N
    return sat_int(shifted, bits)


def _mn(ratio: float) -> tuple[int, int]:
    e = math.floor(math.log2(abs(ratio)))
    N = min(max(31 - e, 0), 62)
    M = math.floor(abs(ratio) * (1 << N) + 0.5)  # round-half-up
    if M >= (1 << 31) and N > 0:
        N -= 1
        M = math.floor(abs(ratio) * (1 << N) + 0.5)
    return max(int(M), 1), int(N)


def scales_to_MN(ratio: float) -> tuple[int, int]:
    """ratio（float，离线）→ (M: int32, N)。规格化使 M ∈ [2^30, 2^31)。

    契约 §2.2：N = clamp(31 - floor(log2(ratio)), 0, 62)；舍入后 M ≥ 2^31
    则降一档；M ≥ 1 强制。ratio ≤ 0 非法（scale 恒正）。
    """
    if ratio <= 0:
        raise ValueError(f"ratio 必须为正: {ratio}")
    return _mn(ratio)


def signed_scales_to_MN(ratio: float) -> tuple[int, int]:
    """GN 输出 requant（契约 §3）：γ 可负，M 带符号、|M| ∈ [2^30, 2^31)。"""
    if ratio == 0:
        raise ValueError("ratio 不可为 0")
    M, N = _mn(ratio)
    return (-M if ratio < 0 else M), N


# ---------------------------------------------------------------- conv/linear

def int_conv2d(x_q: torch.Tensor, w_q: torch.Tensor, M, N,
               bias32: torch.Tensor | None = None, stride: int = 1,
               pad: int | tuple = 0, out_bits: int = 8) -> torch.Tensor:
    """acc = Σ x·w（+b32），逐通道 requant（契约 §2.1/§2.2）。

    累加用 float64 卷积（整数精确）；pad 为整数 0（对称量化零点）。
    x_q/w_q: 整数值 torch 张量；M/N: 逐输出通道 list/ndarray。
    """
    xf = x_q.to(torch.float64)
    if pad:
        xf = F.pad(xf, _pad_tuple(pad))
    acc = F.conv2d(xf, w_q.to(torch.float64), stride=stride)
    if bias32 is not None:
        acc = acc + bias32.to(torch.float64).view(1, -1, 1, 1)
    return _requant_channels(acc, M, N, out_bits)


def int_linear(x_q: torch.Tensor, w_q: torch.Tensor, M, N,
               bias32: torch.Tensor | None = None,
               out_bits: int = 8) -> torch.Tensor:
    xf = x_q.to(torch.float64)
    acc = xf @ w_q.to(torch.float64).T
    if bias32 is not None:
        acc = acc + bias32.to(torch.float64)
    return _requant_channels(acc, M, N, out_bits)


def _pad_tuple(pad):
    if isinstance(pad, int):
        return (pad, pad, pad, pad)
    l, r, t, b = pad  # 契约 §5.4 downsample: (0,1,0,1)
    return (l, r, t, b)


def _requant_channels(acc: torch.Tensor, M, N, out_bits: int) -> torch.Tensor:
    acc64 = acc.to(torch.int64)
    out = torch.empty_like(acc64)
    flat_M = np.asarray(M, dtype=np.int64).reshape(-1)
    flat_N = np.asarray(N, dtype=np.int64).reshape(-1)
    for c in range(acc64.shape[1]):
        out[:, c] = torch.from_numpy(
            requant(acc64[:, c].numpy(), int(flat_M[c]), int(flat_N[c]),
                    out_bits)).to(torch.int64)
    return out


# ---------------------------------------------------------------- LUT 生成

def gen_silu_lut(s_in: float, s_out: float) -> np.ndarray:
    """256 项 INT8（契约 §5.2）：table[a] = sat_i8(rh(silu((a-128)·s_in)/s_out))。"""
    a = np.arange(256, dtype=np.float64)
    z = (a - 128.0) * s_in
    silu = z / (1.0 + np.exp(-z))
    table = np.floor(silu / s_out + 0.5)  # round-half-up
    return sat_int(table.astype(np.int64), 8).astype(np.int8)


def silu_lut(table: np.ndarray, x_q) -> np.ndarray:
    """查表：地址 = code + 128。"""
    x = np.asarray(x_q, dtype=np.int64)
    return table[x + 128].astype(np.int64)


def gen_exp_lut(entries: int = EXP_ENTRIES, log2_delta: float = EXP_LOG2_DELTA,
                frac: int = EXP_FRAC) -> np.ndarray:
    """exp LUT（契约 §5.3）：entry[a] = sat_u16(rh(e^(-a·2^log2_delta)·2^frac))。"""
    a = np.arange(entries, dtype=np.float64)
    vals = np.exp(-a * 2.0 ** log2_delta) * (1 << frac)
    q = np.floor(vals + 0.5)
    return np.clip(q.astype(np.int64), 0, 65535).astype(np.uint16)


def gen_rsqrt_lut(entries: int = RSQRT_ENTRIES,
                  mant_bits: int = RSQRT_MANT_BITS):
    """rsqrt LUT（契约 §3）：两份（偶/奇指数）。

    输入 v = var_q16（Q16，x 格²）；规格化 v = m·2^E，m ∈ [2^22, 2^23)；
    j = m >> 11（11 位地址）。LUT0[k] = rh(2^25/√((k+2^11)·2^11))，
    LUT1[k] = rh(2^25·√2/√((k+2^11)·2^11))。inv_q14 见 rsqrt_q14。
    """
    k = np.arange(entries, dtype=np.float64)
    m_ref = (k + (1 << mant_bits)) * (1 << mant_bits)  # j·2^11
    lut0 = np.floor(2.0 ** 25 / np.sqrt(m_ref) + 0.5)
    lut1 = np.floor(2.0 ** 25 * math.sqrt(2.0) / np.sqrt(m_ref) + 0.5)
    return lut0.astype(np.int64), lut1.astype(np.int64)


def rsqrt_q14(v_q16: int | np.ndarray, lut0: np.ndarray,
              lut1: np.ndarray) -> np.ndarray | int:
    """v = var_q16 → inv_q14（契约 §3 步骤 1-4）。"""
    v = np.asarray(v_q16, dtype=np.int64)
    scalar = v.ndim == 0
    v = np.atleast_1d(v)
    v = np.maximum(v, RSQRT_VAR_MIN_Q16)
    b = np.floor(np.log2(v.astype(np.float64))).astype(np.int64) + 1  # bit_length
    E = b - 23
    m = np.where(E >= 0, v >> np.maximum(E, 0), v << np.clip(-E, 0, 62))
    j = m >> RSQRT_MANT_BITS
    idx = np.clip(j - (1 << (22 - RSQRT_MANT_BITS)), 0,
                  RSQRT_ENTRIES - 1)  # j ∈ [2048,4096) → [0,2048)

    def _shift(vals: np.ndarray, sh: np.ndarray) -> np.ndarray:
        # 右移 sh；sh < 0 表示左移（var 小于 1 时指数为负）
        return np.where(sh >= 0, vals >> np.maximum(sh, 0),
                        vals << np.clip(-sh, 0, 62))

    inv = np.where(E % 2 == 0,
                   _shift(lut0[idx], 3 + E // 2),
                   _shift(lut1[idx], 4 + (E - 1) // 2))
    return inv[0] if scalar else inv


# ---------------------------------------------------------------- GroupNorm

def groupnorm_int(x_q: np.ndarray, num_groups: int, eps_q: int,
                  lut0: np.ndarray, lut1: np.ndarray,
                  Gc: np.ndarray, Nc: np.ndarray, Bq: np.ndarray) -> np.ndarray:
    """GN 整数流（契约 §3）。x_q: int codes (C,H,W)；Gc/Nc/Bq 逐通道。

    y = sat_i8(((x̂·Gc + 2^(Nc-1)) >> Nc) + Bq)，x̂ = sat16((d·inv + 2^9) >> 10)。
    """
    C, H, W = x_q.shape
    G = num_groups
    ch_per_g = C // G
    n = ch_per_g * H * W
    codes = x_q.astype(np.int64).reshape(G, ch_per_g, H * W)
    s1 = codes.sum(axis=(1, 2))                                   # INT32 域
    s2 = (codes * codes).sum(axis=(1, 2))                         # INT64
    mu = (2 * s1 * (1 << 8) + n) // (2 * n)                       # Q8.8，rh
    var_q16 = (s2 << 16) // n - mu * mu
    var_q16 = np.maximum(var_q16, 0) + eps_q
    inv = rsqrt_q14(var_q16, lut0, lut1)                          # Q14

    d = (codes << 8) - mu[:, None, None]                          # Q8.8
    prod = d * inv[:, None, None]
    xhat = np.clip((prod + (1 << 9)) >> 10, -(1 << 15), (1 << 15) - 1)
    # 逐通道 requant（γ 带符号折进 M）
    xhat = xhat.reshape(G, ch_per_g, H, W).reshape(C, H, W)
    Nc64 = Nc.astype(np.int64)
    t = (xhat * Gc[:, None, None]
         + (1 << (Nc64 - 1))[:, None, None]) >> Nc64[:, None, None]
    y = t + Bq[:, None, None]
    return sat_int(y, 8)


# ---------------------------------------------------------------- softmax

def softmax_uint8(scores: np.ndarray, Kexp: int, Qe: int,
                  exp_lut: np.ndarray) -> np.ndarray:
    """契约 §5.3：行内减 max → exp LUT（Δ=2^-8）→ 归一 → UINT8。

    scores: int64 (..., L)；Kexp/Qe 编码 s_qkv²·scale_attn·2^8。
    """
    s = scores.astype(np.int64)
    m = s.max(axis=-1, keepdims=True)
    a = ((m - s) * Kexp + (1 << (Qe - 1))) >> Qe
    a = np.clip(a, 0, EXP_ENTRIES - 1)
    e = exp_lut[a].astype(np.int64)
    total = e.sum(axis=-1, keepdims=True)
    R = (2 * (1 << 32) + total) // (2 * total)          # rh(2^32/sum)，INT32
    p = (e * R + (1 << 23)) >> 24
    return np.minimum(p, 255).astype(np.uint8)


# ---------------------------------------------------------------- DDIM

def ddim_update(x_q: np.ndarray, eps_q: np.ndarray, A: int, B: int,
                C: int, D: int) -> np.ndarray:
    """DDIM 更新单元（契约 §6）。A/B 为 Q8.23，C/D 为 Q2.30（s16/s_x 已折入）。

    t1 = rh(x·A)>>23；t2 = rh(eps·B)>>23；x0 = clip(t1-t2, ±4096)；
    prev = sat16(rh(x0·C + eps·D) >> 30)。
    """
    x = np.asarray(x_q, dtype=np.int64)
    e = np.asarray(eps_q, dtype=np.int64)
    t1 = (x * A + (1 << 22)) >> 23
    t2 = (e * B + (1 << 22)) >> 23
    x0 = np.clip(t1 - t2, -4096, 4096)
    prev = (x0 * C + e * D + (1 << 29)) >> 30
    return sat_int(prev, 16).astype(np.int64)


def x_to_pixel(x_q: int) -> int:
    """[-1,1] → [0,255]（契约 §6）：px = clamp(((x+4096)·255 + 4096) >> 13)。"""
    px = ((int(x_q) + 4096) * 255 + 4096) >> 13
    return max(0, min(255, px))
