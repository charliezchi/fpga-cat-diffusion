"""权重量化遍历（契约 §3）：遍历模型全部 Conv2d/Linear，产出按位宽量化的权重。

v3 混合精度：int16_weight_layers 按模块名首段前缀匹配（如 conv_in/conv_out），
匹配层 per-channel INT16，其余 per-channel INT8。
"""

import torch
from torch import nn

from catdiff.quant.weights import quantize_per_channel


def iter_quantized_layers(model: nn.Module, int16_weight_layers: tuple = ()):
    """yield (name, w_q, scales_fp32, bias_fp32|None, bits)，按 named_modules 顺序。"""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            bits = 16 if name.split(".")[0] in int16_weight_layers else 8
            w_q, scales = quantize_per_channel(mod.weight.detach(), axis=0, bits=bits)
            bias = None if mod.bias is None else mod.bias.detach().to(torch.float32)
            yield name, w_q, scales, bias, bits
