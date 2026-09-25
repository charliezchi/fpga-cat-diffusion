"""真实权重整网对齐（需要 HF 缓存，分钟级耗时）。

运行：uv run pytest -m slow -v
"""

import json
from pathlib import Path

import pytest
import torch

from catdiff.baseline.sampling import sample_batch, tensor_to_pil
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.trace import forward_with_trace
from catdiff.model.unet import UNet2D, as_diffusers_output, load_handwritten_unet

pytestmark = pytest.mark.slow

REF = Path("docs/reference")
CAT_CONFIG = json.loads((REF / "unet-config-ddpm-cat-256.json").read_text())
BFLY_CONFIG = json.loads((REF / "unet-config-butterflies-64.json").read_text())


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_cat_forward_64_and_256():
    from diffusers import UNet2DModel

    ref = UNet2DModel.from_pretrained("google/ddpm-cat-256").eval()
    ours = load_handwritten_unet("google/ddpm-cat-256", CAT_CONFIG)
    for size in (64, 256):
        torch.manual_seed(size)
        x, t = torch.randn(1, 3, size, size), torch.tensor(250)
        with torch.no_grad():
            out_ref = ref(x, t).sample
            out_ours = ours(x, t)
        assert _rel(out_ours, out_ref) < 1e-4, f"cat 前向 @{size} 不对齐"


def test_butterflies_forward_64():
    from diffusers import UNet2DModel

    ref = UNet2DModel.from_pretrained("jfjensen/sd-class-butterflies-64").eval()
    ours = load_handwritten_unet("jfjensen/sd-class-butterflies-64", BFLY_CONFIG)
    torch.manual_seed(0)
    x, t = torch.randn(1, 3, 64, 64), torch.tensor(250)
    with torch.no_grad():
        assert _rel(ours(x, t), ref(x, t).sample) < 1e-4


def test_cat_ddim3_256_pipeline_swap():
    """手写模型 + 手写调度器 与 diffusers 参考在 256×256 下对齐（spec 硬性要求）。"""
    from diffusers import UNet2DModel
    from diffusers import DDIMScheduler as RefDDIM

    ref_model = UNet2DModel.from_pretrained("google/ddpm-cat-256").eval()
    ours = load_handwritten_unet("google/ddpm-cat-256", CAT_CONFIG)
    ref_sched = RefDDIM.from_pretrained("google/ddpm-cat-256")
    ref_sched.set_timesteps(3)
    our_sched = DDIMScheduler(num_train_timesteps=1000)
    our_sched.set_timesteps(3)

    torch.manual_seed(42)
    noise = torch.randn(1, 3, 256, 256)
    x_ref = noise.clone()
    with torch.no_grad():
        for t in ref_sched.timesteps:
            x_ref = ref_sched.step(ref_model(x_ref, t).sample, t, x_ref,
                                   eta=0.0).prev_sample
    x_ours = noise.clone()
    with torch.no_grad():
        for t in our_sched.timesteps:
            x_ours = our_sched.step(ours(x_ours, t), t, x_ours).prev_sample
    assert _rel(x_ours, x_ref) < 1e-4


def test_cat_ddim20_256_final_image():
    """spec §5.2 最终图像判据：逐像素最大绝对差 < 2e-3。对比图存 artifacts/f2/。"""
    from diffusers import UNet2DModel
    from diffusers import DDIMScheduler as RefDDIM

    ref_model = UNet2DModel.from_pretrained("google/ddpm-cat-256").eval()
    ref_sched = RefDDIM.from_pretrained("google/ddpm-cat-256")
    ref_sched.set_timesteps(20)
    ours = load_handwritten_unet("google/ddpm-cat-256", CAT_CONFIG)
    our_sched = DDIMScheduler(num_train_timesteps=1000)
    our_sched.set_timesteps(20)

    img_ref = sample_batch(ref_model, ref_sched, num_samples=1,
                           image_size=256, seed=100)
    img_ours = sample_batch(as_diffusers_output(ours), our_sched,
                            num_samples=1, image_size=256, seed=100)
    max_abs = (img_ours - img_ref).abs().max().item()
    out_dir = Path("artifacts/f2")
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(img_ref[0]).save(out_dir / "ddim20_ref.png")
    tensor_to_pil(img_ours[0]).save(out_dir / "ddim20_handwritten.png")
    assert max_abs < 2e-3
