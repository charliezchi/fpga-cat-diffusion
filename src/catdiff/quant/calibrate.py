"""激活静态标定：按"层 × 步组"统计 P99.9(|activation|)（契约 §1/§2）。"""

import json
from pathlib import Path

import torch

from catdiff.model.trace import forward_with_trace


def step_to_group(step_idx: int, num_steps: int, num_groups: int) -> int:
    """推理步序号 → 步组号（均分，20 步 5 组 = 每组 4 步）。"""
    per_group = num_steps // num_groups
    return min(step_idx // per_group, num_groups - 1)


def update_percentile_stats(stats: dict, name: str, group: int, tensor: torch.Tensor,
                            max_samples: int = 4096) -> None:
    """记录 |activation| 样本（子采样控制内存）。"""
    flat = tensor.detach().abs().flatten()
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:max_samples]
        flat = flat[idx]
    stats.setdefault(name, {}).setdefault(group, []).extend(flat.tolist())


def finalize_scales(stats: dict, percentile: float = 99.9) -> dict:
    """{layer: {group: scale}}，scale = P99.9(|a|) / 127。"""
    out = {}
    for name, groups in stats.items():
        out[name] = {}
        for group, values in groups.items():
            t = torch.tensor(values)
            q = torch.quantile(t, percentile / 100.0).item()
            out[name][int(group)] = max(q / 127.0, 1e-12)
    return out


@torch.no_grad()
def run_calibration(config_path: str = "configs/quant_int8.json",
                    out_path: str = "artifacts/f4/calib-stats.json") -> None:
    """slow：对手写 cat-256 在标定 seed 集上跑 DDIM-20 全程并采集统计。"""
    from catdiff.model.ddim import DDIMScheduler
    from catdiff.model.unet import load_handwritten_unet

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    n_steps = cfg["schedule"]["num_inference_steps"]
    n_groups = cfg["schedule"]["num_step_groups"]
    size = cfg["calibration"]["image_size"]

    stats = {}
    for seed in cfg["calibration"]["seeds"]:
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(n_steps)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 3, size, size, generator=g)
        for step_idx, t in enumerate(sched.timesteps):
            group = step_to_group(step_idx, n_steps, n_groups)
            noise, trace = forward_with_trace(model, x, t)
            for name, tensor in trace.items():
                update_percentile_stats(stats, name, group, tensor)
            x = sched.step(noise, t, x).prev_sample
        print(f"seed {seed} 标定完成", flush=True)

    scales = finalize_scales(stats, cfg["act_quant"]["percentile"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({k: {str(g): v for g, v in groups.items()}
                    for k, groups in scales.items()}, indent=1),
        encoding="utf-8",
    )
