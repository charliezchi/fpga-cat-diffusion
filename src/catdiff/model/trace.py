"""前向追踪：抓取关键模块的中间输出，用于逐层对齐调试（F2）、消融（F3）与黄金向量（F5）。

对手写模型与 diffusers 参考模型通用（模块命名一致）。
"""

import re

import torch

_DEFAULT_PATTERN = re.compile(
    r"^(conv_in|conv_out|mid_block|"
    r"(?:down_blocks|up_blocks)\.\d+\.(?:resnets|attentions|downsamplers|upsamplers)\.\d+|"
    r"mid_block\.(?:resnets|attentions)\.\d+)$"
)


def forward_with_trace(model, sample, timestep, name_filter=None):
    """返回 (模型输出, {模块名: 输出 tensor})。

    name_filter: callable(name: str) -> bool；默认捕获 conv_in/conv_out/mid_block
    与所有 resnet/attention/上下采样模块（不含其内部子层）。
    """
    if name_filter is None:
        name_filter = lambda name: bool(_DEFAULT_PATTERN.match(name))
    trace = {}
    hooks = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            if torch.is_tensor(out):
                trace[name] = out.detach()
        return hook

    for name, mod in model.named_modules():
        if name and name_filter(name):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    try:
        out = model(sample, timestep)
    finally:
        for h in hooks:
            h.remove()
    if hasattr(out, "sample"):  # diffusers UNet2DOutput
        out = out.sample
    return out, trace
