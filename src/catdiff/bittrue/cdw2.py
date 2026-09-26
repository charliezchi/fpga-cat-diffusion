"""CDW2 权重包读取（契约 quant-format.md §4.1）。

位真模拟器与 requant 导出器共用的唯一读取实现；只读，不回写。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

_MAGIC = b"CDW2"


@dataclass
class WeightRecord:
    name: str
    shape: tuple[int, ...]
    bits: int
    has_bias: bool
    scales: np.ndarray      # f32[C]
    bias: np.ndarray | None  # f32[C] 或 None
    payload: np.ndarray     # i8 / i16（已按声明位宽读出）


def read_weights_bin(path) -> list[WeightRecord]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != _MAGIC:
        raise ValueError(f"weights.bin magic 错误: {data[:4]!r}")
    (n_layers,) = struct.unpack_from("<I", data, 4)
    off = 8
    records = []
    for _ in range(n_layers):
        (name_len,) = struct.unpack_from("<H", data, off)
        off += 2
        name = data[off:off + name_len].decode("utf-8")
        off += name_len
        (ndim,) = struct.unpack_from("<I", data, off)
        off += 4
        shape = struct.unpack_from(f"<{ndim}I", data, off)
        off += 4 * ndim
        bits, has_bias = struct.unpack_from("<BB", data, off)
        off += 2
        c = shape[0]
        scales = np.frombuffer(data, "<f4", c, off).copy()
        off += 4 * c
        bias = None
        if has_bias:
            bias = np.frombuffer(data, "<f4", c, off).copy()
            off += 4 * c
        n = int(np.prod(shape))
        dtype = np.dtype("<i1") if bits == 8 else np.dtype("<i2")
        payload = np.frombuffer(data, dtype, n, off).copy()
        off += dtype.itemsize * n
        records.append(WeightRecord(name, shape, bits, bool(has_bias),
                                    scales, bias, payload))
    return records
