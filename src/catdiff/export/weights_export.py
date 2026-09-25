"""权重量化遍历（契约 §3）：遍历模型全部 Conv2d/Linear，产出 INT8 + scale。"""

import torch
from torch import nn

from catdiff.quant.weights import quantize_per_channel


def iter_quantized_layers(model: nn.Module):
    """yield (name, weight_int8, scales_fp32, bias_fp32|None)，按 named_modules 顺序。"""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w_q, scales = quantize_per_channel(mod.weight.detach(), axis=0)
            bias = None if mod.bias is None else mod.bias.detach().to(torch.float32)
            yield name, w_q, scales, bias
