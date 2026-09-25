"""权重 per-channel 对称 INT8 量化（契约 docs/quant-format.md §1）。"""

import torch

_EPS = 1e-12


def quantize_per_channel(w: torch.Tensor, axis: int = 0):
    dims = [d for d in range(w.ndim) if d != axis]
    max_abs = w.abs().amax(dim=dims)
    scales = (max_abs / 127.0).clamp_min(_EPS)
    shape = [1] * w.ndim
    shape[axis] = -1
    w_q = torch.round(w / scales.reshape(shape)).clamp(-128, 127).to(torch.int8)
    return w_q, scales.to(torch.float32)


def dequantize_per_channel(w_q: torch.Tensor, scales: torch.Tensor, axis: int = 0):
    shape = [1] * w_q.ndim
    shape[axis] = -1
    return w_q.to(torch.float32) * scales.reshape(shape)
