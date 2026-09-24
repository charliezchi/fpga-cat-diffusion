"""L0 fp32 基线：diffusers 加载现成权重，DDIM(eta=0) 采样。"""

from __future__ import annotations

import torch
from diffusers import DDIMScheduler, UNet2DModel
from PIL import Image


def load_unet(model_id: str) -> UNet2DModel:
    model = UNet2DModel.from_pretrained(model_id)
    model.eval()
    return model


def load_ddim(model_id: str, num_inference_steps: int) -> DDIMScheduler:
    scheduler = DDIMScheduler.from_pretrained(model_id)
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


@torch.no_grad()
def sample_batch(
    model: UNet2DModel,
    scheduler: DDIMScheduler,
    *,
    num_samples: int,
    image_size: int,
    seed: int,
    device: str = "cpu",
) -> torch.Tensor:
    """返回 (N, 3, H, W)，值域 [-1, 1]。第 idx 张初始噪声 seed 恒为 seed + idx。"""
    model.to(device)
    frames = []
    for idx in range(num_samples):
        g = torch.Generator(device=device).manual_seed(seed + idx)
        sample = torch.randn(
            1, model.config.in_channels, image_size, image_size,
            generator=g, device=device,
        )
        for t in scheduler.timesteps:
            noise_pred = model(sample, t).sample
            sample = scheduler.step(noise_pred, t, sample, eta=0.0).prev_sample
        frames.append(sample)
    return torch.cat(frames)


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (
        (img.clamp(-1, 1) + 1)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(arr, "RGB")


def make_grid(images: list[Image.Image], cols: int) -> Image.Image:
    if not images or cols <= 0:
        raise ValueError("images 非空且 cols > 0")
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid
