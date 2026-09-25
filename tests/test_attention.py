import torch
from diffusers.models.attention_processor import Attention as RefAttention

from catdiff.model.attention import AttentionBlock


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_attention_matches_diffusers():
    torch.manual_seed(0)
    # residual_connection=True 与 diffusers UNet2DModel 内部（AttnDownBlock2D/
    # UNetMidBlock2D）实例化 Attention 的方式一致（默认 False 是裸 Attention 语义）
    ref = RefAttention(
        query_dim=64, heads=1, dim_head=64, norm_num_groups=8,
        eps=1e-6, bias=True, dropout=0.0, scale_qk=True,
        residual_connection=True,
    )
    mine = AttentionBlock(64, groups=8)
    mine.load_state_dict(ref.state_dict())  # group_norm/to_q/to_k/to_v/to_out 命名对齐
    x = torch.randn(1, 64, 8, 8)
    assert _rel(mine(x), ref(x)) < 1e-4
