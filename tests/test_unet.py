import torch
from diffusers import UNet2DModel

from catdiff.model.trace import forward_with_trace
from catdiff.model.unet import UNet2D

TINY_CONFIG = dict(
    in_channels=3,
    out_channels=3,
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    block_out_channels=(32, 64, 128),
    layers_per_block=2,
    norm_num_groups=8,
)


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def _make_pair():
    torch.manual_seed(0)
    # freq_shift=1 / flip_sin_to_cos=False / norm_eps=1e-6：把参考对齐到 cat-256
    # 配置族（docs/reference/unet-config-ddpm-cat-256.json）——UNet2DModel 的默认值
    # 是 freq_shift=0 / flip_sin_to_cos=True / norm_eps=1e-5，与手写模型硬编码不一致
    ref = UNet2DModel(
        sample_size=32, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8, downsample_padding=0, attention_head_dim=None,
        freq_shift=1, flip_sin_to_cos=False, norm_eps=1e-6,
    ).eval()
    ours = UNet2D(TINY_CONFIG).eval()
    ours.load_state_dict(ref.state_dict())  # 严格加载，命名漂移会在此报错
    return ref, ours


def test_state_dict_strict_load():
    _make_pair()  # 不抛异常即通过


def test_forward_and_layers_match_diffusers():
    ref, ours = _make_pair()
    torch.manual_seed(1)
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        out_ref, trace_ref = forward_with_trace(ref, x, t)
        out_ours, trace_ours = forward_with_trace(ours, x, t)
    assert set(trace_ours) == set(trace_ref), "trace 捕获的层名集合必须一致"
    for name in sorted(trace_ours):
        assert _rel(trace_ours[name], trace_ref[name]) < 1e-4, f"层 {name} 不对齐"
    # forward_with_trace 已把 UNet2DOutput 解包为裸 tensor，两侧均为 Tensor
    assert _rel(out_ours, out_ref) < 1e-4


def test_cross_resolution_64():
    ref, ours = _make_pair()
    torch.manual_seed(2)
    x, t = torch.randn(1, 3, 64, 64), torch.tensor(500)
    with torch.no_grad():
        assert _rel(ours(x, t), ref(x, t).sample) < 1e-4
