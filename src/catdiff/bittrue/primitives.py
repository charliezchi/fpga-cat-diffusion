"""位真整数原语库（docs/bittrue-spec.md §0-§3 的可执行形式）。

全部运算在整数域（torch/numpy 整数或 float64 容器，容器内值恒为整数），
与 RTL 逐位等价。scale→M/N 的离线换算是唯一允许 float 的地方（单独函数，
附 float 对照测试）。注释标出 RTL 对应物。
"""

from __future__ import annotations

import math

import numpy as np
import torch

QMAX = {8: 127, 16: 32767}
QMIN = {8: -128, 16: -32768}


def round_half_up_div(num: int, den: int) -> int:
    """round_half_up(num/den)，den>0，整数实现（RTL: 加偏置后算术右移/除法）。"""
    return (2 * num + den) // (2 * den)


def sat_int(v: int | np.ndarray, bits: int):
    """饱和到对称域 [-qmax-1, qmax]（RTL: 位截断 + 饱和逻辑）。"""
    lo, hi = QMIN[bits], QMAX[bits]
    if isinstance(v, np.ndarray):
        return np.clip(v, lo, hi)
    return max(lo, min(hi, int(v)))


def requant(acc, M, N, bits: int = 8):
    """y = sat((acc·M + 2^(N-1)) >> N)（契约 §2.2）。

    acc/M 可为标量或 ndarray（广播）；acc·M 用 int64（RTL: 乘加器位宽）。
    M ≥ 0（conv requant）或有符号（GN requant）均可：算术右移实现
    round-half-up，负数行为有测试钉死。
    """
    acc = np.asarray(acc, dtype=np.int64)
    M = np.asarray(M, dtype=np.int64)
    shifted = (acc * M + (1 << (N - 1))) >> N
    return sat_int(shifted, bits)


def scales_to_MN(ratio: float, mantissa_min_exp: int = 30,
                 max_shift: int = 62) -> tuple[int, int]:
    """ratio（float，离线）→ (M: int32, N)。规格化使 M ∈ [2^30, 2^31)。

    契约 §2.2：N = clamp(31 - floor(log2(ratio)), 0, 62)；
    舍入后 M ≥ 2^31 则降一档。ratio ≤ 0 非法（scale 恒正）。
    """
    if ratio <= 0:
        raise ValueError(f"ratio 必须为正: {ratio}")
    e = math.floor(math.log2(ratio))
    N = min(max(mantissa_min_exp - e, 0), max_shift)
    M = math.floor(ratio * (1 << N) + 0.5)  # round-half-up（ratio 恒正）
    if M >= (1 << 31) and N > 0:
        N -= 1
        M = math.floor(ratio * (1 << N) + 0.5)
    M = max(M, 1)
    return int(M), int(N)
