"""权重 per-channel 对称量化（契约 docs/quant-format.md §1）。

INT8 为默认位宽；契约的混合精度层（conv_in/conv_out）以 bits=16 调用。
"""

import torch

_EPS = 1e-12


def quantize_per_channel(w: torch.Tensor, axis: int = 0, bits: int = 8):
    qmax = 2 ** (bits - 1) - 1
    dims = [d for d in range(w.ndim) if d != axis]
    max_abs = w.abs().amax(dim=dims)
    scales = (max_abs / qmax).clamp_min(_EPS)
    shape = [1] * w.ndim
    shape[axis] = -1
    dtype = torch.int8 if bits <= 8 else torch.int16
    w_q = torch.round(w / scales.reshape(shape)).clamp(-qmax - 1, qmax).to(dtype)
    return w_q, scales.to(torch.float32)


def dequantize_per_channel(w_q: torch.Tensor, scales: torch.Tensor, axis: int = 0):
    shape = [1] * w_q.ndim
    shape[axis] = -1
    return w_q.to(torch.float32) * scales.reshape(shape)
