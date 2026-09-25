"""fake-quant 参考模型：float 近似 INT8 行为，用于快速画质验证（契约 §1）。

权重 per-channel Q/DQ 包在 conv/linear 外；激活 per-tensor Q/DQ 按
forward_with_trace 同款层名的量化点以 hook 注入，scale 来自标定统计。
GN/SiLU 内部保持 float（其整数行为由 F5 位真模拟器定义）。
"""

import torch
from torch import nn

from catdiff.model.trace import _DEFAULT_PATTERN
from catdiff.quant.calibrate import step_to_group
from catdiff.quant.weights import dequantize_per_channel, quantize_per_channel


class FakeQuantContext:
    def __init__(self, scales: dict, num_steps: int, num_groups: int):
        self.scales = scales
        self.num_steps = num_steps
        self.num_groups = num_groups
        self.step_idx = 0

    def set_step(self, step_idx: int) -> None:
        self.step_idx = step_idx

    @property
    def group(self) -> int:
        return step_to_group(self.step_idx, self.num_steps, self.num_groups)


def _fake_quant_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.round(x / scale).clamp(-128, 127) * scale


def apply_fake_quant(model: nn.Module, ctx: FakeQuantContext) -> int:
    """返回施加 Q/DQ 的位置数。权重在调用时 Q/DQ（state_dict 不受污染）。"""
    count = 0

    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w = mod.weight
            w_q, scales = quantize_per_channel(w.detach(), axis=0)
            mod.weight = nn.Parameter(dequantize_per_channel(w_q, scales, axis=0),
                                      requires_grad=False)
            count += 1

    def make_hook(name):
        def hook(_mod, _inp, out):
            if not torch.is_tensor(out):
                return
            group_scales = ctx.scales.get(name)
            if group_scales is None:
                return
            scale = group_scales.get(ctx.group) or group_scales[str(ctx.group)]
            return _fake_quant_tensor(out, scale)
        return hook

    for name, mod in model.named_modules():
        if name and _DEFAULT_PATTERN.match(name):
            mod.register_forward_hook(make_hook(name))
            count += 1
    return count
