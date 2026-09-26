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
    N_arr = np.asarray(N, dtype=np.int64)
    # 逐通道向量沿 axis=1 广播到 (N,C,H,W)/(N,C)
    if M.ndim == 1:
        if acc.ndim == 3 and acc.shape[0] == M.shape[0]:
            M = M.reshape(-1, 1, 1)
            N_arr = N_arr.reshape(-1, 1, 1)
        elif acc.ndim >= 2 and acc.shape[1] == M.shape[0]:
            tail = (1,) * (acc.ndim - 2)
            M = M.reshape(1, -1, *tail)
            N_arr = N_arr.reshape(1, -1, *tail)
    if N_arr.ndim == 0:
        n = int(N_arr)
        if n <= 0:
            shifted = acc * M
        else:
            shifted = (acc * M + (1 << (n - 1))) >> n
    else:  # 逐通道 N（含 0：不移位不舍入）
        off = np.where(N_arr > 0, (1 << np.maximum(N_arr - 1, 0)), 0)
        shifted = np.where(N_arr > 0, (acc * M + off) >> np.maximum(N_arr, 0),
                           acc * M)
    return sat_int(shifted, bits)


def _mn(ratio: float) -> tuple[int, int]:
    e = math.floor(math.log2(abs(ratio)))
    N = min(max(31 - e, 0), 62)
    M = math.floor(abs(ratio) * (1 << N) + 0.5)  # round-half-up
    if M >= (1 << 31) and N > 0:
        N -= 1
        M = math.floor(abs(ratio) * (1 << N) + 0.5)
    return max(int(M), 1), int(N)


def requant48(acc, M, N, bits: int = 16):
    """细格输入层的 requant：acc 可达 2^47（INT48 累加器，DSP 级联）。

    acc·M 会超 int64 → 拆 16 位精确计算（数学上与 (acc·M + 2^(N-1)) >> N
    逐位一致，测试钉死）：acc = hi·2^16 + lo，
    c = (lo·M + 2^(N-1)) >> 16，y = sat((hi·M + c) >> (N-16))。
    """
    acc = np.asarray(acc, dtype=np.int64)
    M_arr = np.asarray(M, dtype=np.int64)
    hi = acc >> 16
    lo = acc - (hi << 16)
    A = hi * M_arr
    B = lo * M_arr + (1 << (N - 1))
    c = B >> 16
    Q = N - 16
    shifted = (A + c) >> Q
    return sat_int(shifted, bits)


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
               pad: int | tuple = 0, out_bits: int = 8,
               acc_bits: int = 32) -> torch.Tensor:
    """acc = Σ x·w（+b32），逐通道 requant（契约 §2.1/§2.2）。

    累加用 float64 卷积（整数精确）；pad 为整数 0（对称量化零点）。
    acc_bits=48 用于细格输入层（契约 §2.1 v1.1，DSP 级联累加器）。
    """
    xf = x_q.to(torch.float64)
    if pad:
        xf = F.pad(xf, _pad_tuple(pad))
    acc = F.conv2d(xf, w_q.to(torch.float64), stride=stride)
    if bias32 is not None:
        acc = acc + bias32.to(torch.float64).view(1, -1, 1, 1)
    lim = (1 << (acc_bits - 1)) - 1
    acc = torch.clamp(acc, -lim, lim)
    if acc_bits <= 32:
        return _requant_channels(acc, M, N, out_bits)
    # 大数 requant（逐通道，拆位精确）
    acc64 = acc.to(torch.int64)
    M_arr = np.asarray(M, dtype=np.int64).reshape(-1)
    N_arr = np.asarray(N, dtype=np.int64).reshape(-1)
    out = torch.empty_like(acc64)
    for c in range(acc64.shape[1]):
        out[:, c] = torch.from_numpy(
            requant48(acc64[:, c].numpy(), int(M_arr[c]), int(N_arr[c]),
                      out_bits)).to(torch.int64)
    return out


def int_linear(x_q: torch.Tensor, w_q: torch.Tensor, M, N,
               bias32: torch.Tensor | None = None,
               out_bits: int = 8, acc_bits: int = 32) -> torch.Tensor:
    xf = x_q.to(torch.float64)
    acc = xf @ w_q.to(torch.float64).T
    if bias32 is not None:
        acc = acc + bias32.to(torch.float64)
    lim = (1 << (acc_bits - 1)) - 1
    acc = torch.clamp(acc, -lim, lim)
    if acc_bits <= 32:
        return _requant_channels(acc, M, N, out_bits)
    acc64 = acc.to(torch.int64)
    M_arr = np.asarray(M, dtype=np.int64).reshape(-1)
    N_arr = np.asarray(N, dtype=np.int64).reshape(-1)
    out = torch.empty_like(acc64)
    for r_ in range(acc64.shape[0]):
        out[r_] = torch.from_numpy(
            requant48(acc64[r_].numpy(), M_arr, N_arr, out_bits)).to(torch.int64)
    return out


def _sat32(acc: torch.Tensor) -> torch.Tensor:
    """INT32 累加器饱和（RTL 溢出保护；模拟器容器为 float64/int64）。"""
    return torch.clamp(acc, -(2**31 - 1), 2**31 - 1)


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
    """SiLU 基表（契约 §5.2 v1.1）：257 项 INT32。

    T[a] = rh(silu((a-128)·s_coarse)/s_fine)，s_coarse = 输入点 INT8 基准格，
    s_fine = 输出点细格（= s_coarse/256）。配合 silu_lut 的线性插值使用。
    """
    a = np.arange(257, dtype=np.float64)
    z = (a - 128.0) * s_in
    silu = z / (1.0 + np.exp(-z))
    return np.floor(silu / s_out + 0.5).astype(np.int64)  # INT32 域


def silu_lut(table: np.ndarray, x_q) -> np.ndarray:
    """查表 + 线性插值（契约 §5.2）：a = code>>8（算术），f = code-(a<<8)。"""
    x = np.asarray(x_q, dtype=np.int64)
    a = x >> 8
    f = x - (a << 8)
    t0 = table[a + 128]
    t1 = table[a + 129]
    return (t0 + ((t1 - t0) * f >> 8)).astype(np.int64)


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
                  Gc: np.ndarray, Nc: np.ndarray, Bq: np.ndarray,
                  out_bits: int = 16, fine_input: bool = False) -> np.ndarray:
    """GN 整数流（契约 §3 v1.1）。

    x_q: codes (C,H,W)（表定点 INT8 网格或内部点细网格）；Gc/Nc/Bq 逐通道。
    var/eps/inv 统一以 **INT8 等价格式** 计量（细格输入除以 256²），
    保证 rsqrt Q14 输出精度与 int8 输入一致；x̂ 移位按输入网格分支
    （fine_input: d6 为细格 Q6，x̂ = (d6·inv + 2^15) >> 16）。
    out_bits：细网格输出 16，表定点输出 8。
    """
    if x_q.ndim == 4:          # (N,C,H,W)：本项目 batch=1，兼容保留
        assert x_q.shape[0] == 1
        return groupnorm_int(x_q[0], num_groups, eps_q, lut0, lut1,
                             Gc, Nc, Bq, out_bits, fine_input)[None]
    C, H, W = x_q.shape
    G = num_groups
    ch_per_g = C // G
    n = ch_per_g * H * W
    codes = x_q.astype(np.int64).reshape(G, ch_per_g, H * W)
    s1 = codes.sum(axis=(1, 2))                                   # INT64
    mu = (2 * s1 * (1 << 6) + n) // (2 * n)                       # Q0.6，自身格
    d = (codes << 6) - mu[:, None, None]                          # Q0.6，自身格
    # var → INT8 等价格² Q16：细格输入除以 256²（即 >>16 再 <<4 → >>12）
    var_q16 = (d * d).sum(axis=(1, 2)) // n
    var_q16 = (var_q16 << 4) if not fine_input else (var_q16 >> 12)
    var_q16 = np.maximum(var_q16, 0) + eps_q
    inv = rsqrt_q14(var_q16, lut0, lut1)                          # Q14，int8 等价格

    if fine_input:
        # (x-μ)_int8等价 = d6/2^14；x̂_q12 = d6·inv_q14 >> 16
        xhat = np.clip((d * inv[:, None, None] + (1 << 15)) >> 16,
                       -(1 << 15), (1 << 15) - 1)
    else:
        xhat = np.clip((d * inv[:, None, None] + (1 << 7)) >> 8,
                       -(1 << 15), (1 << 15) - 1)
    # 逐通道 requant（γ 带符号折进 M）
    xhat = xhat.reshape(G, ch_per_g, H, W).reshape(C, H, W)
    Nc64 = Nc.astype(np.int64)
    t = (xhat * Gc[:, None, None]
         + (1 << (Nc64 - 1))[:, None, None]) >> Nc64[:, None, None]
    y = t + Bq[:, None, None]
    return sat_int(y, out_bits)


# ---------------------------------------------------------------- softmax

def softmax_uint8(scores: np.ndarray, Kexp: int, Qe: int,
                  exp_lut: np.ndarray) -> np.ndarray:
    """契约 §5.3 v1.2：行内减 max → exp LUT（Δ=2^-8）→ 归一 → UINT16。

    p = min(65535, (e·R + 2^23) >> 24)，R = rh(2^40/Σe)（p ≈ 65536·e/Σe）。
    scores: int64 (..., L)；Kexp/Qe 编码 s_qkv²·scale_attn·2^8。
    """
    s = scores.astype(np.int64)
    m = s.max(axis=-1, keepdims=True)
    a = ((m - s) * Kexp + (1 << (Qe - 1))) >> Qe
    a = np.clip(a, 0, EXP_ENTRIES - 1)
    e = exp_lut[a].astype(np.int64)
    total = e.sum(axis=-1, keepdims=True)
    R = (2 * (1 << 40) + total) // (2 * total)          # rh(2^40/sum)
    p = (e * R + (1 << 23)) >> 24
    return np.minimum(p, 65535).astype(np.uint16)


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
