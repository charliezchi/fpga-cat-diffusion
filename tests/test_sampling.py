import torch
from PIL import Image

from catdiff.baseline.sampling import make_grid, sample_batch, tensor_to_pil


def test_sample_batch_deterministic(tiny_unet, tiny_scheduler):
    out1 = sample_batch(tiny_unet, tiny_scheduler(), num_samples=2, image_size=32, seed=123)
    out2 = sample_batch(tiny_unet, tiny_scheduler(), num_samples=2, image_size=32, seed=123)
    assert torch.equal(out1, out2)


def test_per_sample_seed_invariant(tiny_unet, tiny_scheduler):
    batched = sample_batch(tiny_unet, tiny_scheduler(), num_samples=3, image_size=32, seed=7)
    for idx in range(3):
        single = sample_batch(tiny_unet, tiny_scheduler(), num_samples=1, image_size=32, seed=7 + idx)
        assert torch.equal(batched[idx : idx + 1], single)


def test_cross_resolution_output_shape(tiny_unet, tiny_scheduler):
    # 模型配置 sample_size=32，以 64 输入：输出形状必须跟随输入（跨分辨率能力，spec 5.1）
    out = sample_batch(tiny_unet, tiny_scheduler(), num_samples=1, image_size=64, seed=0)
    assert out.shape == (1, 3, 64, 64)


def test_tensor_to_pil_range():
    img = torch.tensor([[[-1.0] * 4] * 2, [[0.0] * 4] * 2, [[1.0] * 4] * 2])
    im = tensor_to_pil(img)
    assert im.size == (4, 2)  # PIL 为 (W, H)
    assert im.getpixel((0, 0)) == (0, 128, 255)


def test_make_grid_layout():
    ims = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(5)]
    grid = make_grid(ims, cols=2)
    assert grid.size == (16, 24)  # 2 列 3 行
    assert grid.getpixel((0, 16)) == (4, 0, 0)  # 第 5 张在第 3 行第 1 列
