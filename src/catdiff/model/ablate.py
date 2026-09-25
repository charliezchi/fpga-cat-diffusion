"""注意力消融：权重加载后把手写 UNet 的注意力块替换为恒等，构造去注意力变体。"""

from torch import nn

from .attention import AttentionBlock


def strip_attention(model: nn.Module, keep_mid: bool = False) -> int:
    """将模型中的 AttentionBlock 替换为 nn.Identity，返回替换数量。

    必须在 load_state_dict 之后调用（Identity 无参数）。
    keep_mid=True 时保留 mid_block 的注意力（"仅 mid 注意力"中间档）。
    """
    targets = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if isinstance(child, AttentionBlock):
                if keep_mid and parent_name.startswith("mid_block"):
                    continue
                targets.append((parent, child_name))
    for parent, child_name in targets:
        setattr(parent, child_name, nn.Identity())
    return len(targets)
