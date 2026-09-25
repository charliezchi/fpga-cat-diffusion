"""激活静态标定：按"层 × 步组"统计 P99.95(|activation|)（契约 §1/§2）。

顺序标定（契约 v3，config `calibration.sequential=true` 时启用）：先把权重按
目标位宽 Q/DQ 就位，再在真实量化轨迹上采集激活统计。
"""

import json
from pathlib import Path

import torch
from torch import nn

from catdiff.model.trace import forward_with_trace
from catdiff.quant.weights import dequantize_per_channel, quantize_per_channel


def step_to_group(step_idx: int, num_steps: int, num_groups: int) -> int:
    """推理步序号 → 步组号（均分，50 步 10 组 = 每组 5 步）。"""
    per_group = num_steps // num_groups
    return min(step_idx // per_group, num_groups - 1)


def quantize_weights_in_place(model: nn.Module, int16_layers: tuple = ()) -> int:
    """权重按目标位宽 Q/DQ 并就地替换 Parameter（顺序标定第一步）。

    int16_layers 按模块名首段前缀匹配（name.split(".")[0]）。返回处理的层数。
    """
    count = 0
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            bits = 16 if name.split(".")[0] in int16_layers else 8
            w_q, scales = quantize_per_channel(mod.weight.detach(), axis=0, bits=bits)
            mod.weight = nn.Parameter(dequantize_per_channel(w_q, scales, axis=0),
                                      requires_grad=False)
            count += 1
    return count


def update_percentile_stats(stats: dict, name: str, group: int, tensor: torch.Tensor,
                            max_samples: int = 4096) -> None:
    """记录 |activation| 样本（子采样控制内存）。"""
    flat = tensor.detach().abs().flatten()
    if flat.numel() > max_samples:
        idx = torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:max_samples]
        flat = flat[idx]
    stats.setdefault(name, {}).setdefault(group, []).extend(flat.tolist())


def merge_step_stats(per_step: dict, num_steps: int, num_groups: int) -> dict:
    """{layer: {step: [samples]}} → {layer: {group: [组内各步样本归并]}}。

    逐步采集一份数据可离线重分成任意步组数（F4 金标协议的后半段）。
    """
    merged = {}
    for name, steps in per_step.items():
        merged[name] = {}
        for g in range(num_groups):
            vals: list = []
            for s in range(num_steps):
                if step_to_group(s, num_steps, num_groups) == g:
                    vals.extend(steps.get(s, ()))
            merged[name][g] = vals
    return merged


def finalize_scales(stats: dict, percentile: float = 99.95) -> dict:
    """{layer: {group: scale}}，scale = P99.95(|a|) / 127（INT8 量程基准）。"""
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
    """slow：对手写 cat-256 在标定 seed 集上跑 DDIM 全程并采集统计。

    采集协议（F4 金标口径）：每(量化点, 推理步)子采样 2048 个 |a| 样本，
    全部 seed 跑完后按步组归并取 P99.95——一份数据可重分成任意步组数。
    config `calibration.sequential=true` 时先按契约把权重 Q/DQ 就位，
    在真实量化轨迹上采集（v3 顺序标定）；否则保持 fp32 轨迹。
    """
    from catdiff.model.ddim import DDIMScheduler
    from catdiff.model.unet import load_handwritten_unet

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    n_steps = cfg["schedule"]["num_inference_steps"]
    n_groups = cfg["schedule"]["num_step_groups"]
    size = cfg["calibration"]["image_size"]

    int16_layers = tuple(cfg.get("mixed_precision", {}).get("int16_weight_layers", ()))
    if cfg.get("calibration", {}).get("sequential", False):
        n = quantize_weights_in_place(model, int16_layers)
        print(f"顺序标定：权重已预量化 {n} 层（INT16: {list(int16_layers) or '无'}）", flush=True)

    per_step: dict = {}
    for seed in cfg["calibration"]["seeds"]:
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(n_steps)
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(1, 3, size, size, generator=g)
        for step_idx, t in enumerate(sched.timesteps):
            noise, trace = forward_with_trace(model, x, t)
            for name, tensor in trace.items():
                update_percentile_stats(per_step, name, step_idx, tensor,
                                        max_samples=2048)
            x = sched.step(noise, t, x).prev_sample
        print(f"seed {seed} 标定完成", flush=True)

    stats = merge_step_stats(per_step, n_steps, n_groups)
    scales = finalize_scales(stats, cfg["act_quant"]["percentile"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps({k: {str(g): v for g, v in groups.items()}
                    for k, groups in scales.items()}, indent=1),
        encoding="utf-8",
    )
