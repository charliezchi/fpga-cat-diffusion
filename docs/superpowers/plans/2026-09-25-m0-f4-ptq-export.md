# M0-F4 实施计划：PTQ INT8 量化与硬件导出

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对手写 cat-256 UNet 做训练后量化（PTQ INT8：权重 per-channel 对称 + 激活按步组静态标定），通过 INT8 画质关卡，并导出硬件就绪的四类 artifact（weights.bin / 层描述符 / DDIM 常数表 / FiLM 表）。

**Architecture:** 先冻结量化与导出格式契约文档（M2 的输入），再实现：权重量化器 → 标定数据采集（按步组统计激活）→ fake-quant 参考模型（float 近似，用于快速画质验证）→ INT8 画质关卡 → 导出器。量化参考基于 F2 手写模型（256×256 主线、含全部注意力，F3 已定论）。

**Tech Stack:** Python 3.11+、uv、torch（CPU）、numpy、pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md`（§5.4 是验收依据；执行前必读）

## Global Constraints

- 仅 CPU；单元测试不得调用 `from_pretrained`；真实权重走 slow 测试或 CLI/脚本步骤（spec §2）
- 标定与画质关卡针对 **DDIM-20 调度（演示默认档）**，256×256（spec §5.1 已签）；步组 = 把 20 步分为 **5 组 × 4 步**（v1.2 注：spec §5.4 原文"DDIM-50 分 4~8 组"写于 DDIM-20 决策前，本计划按 DDIM-20 执行）
- 量化方案参数集中在**一个配置文件** `configs/quant_int8.yaml`（或 json），可复现实验（spec §5.4）
- 所有采样固定 seed、断点续跑（spec §2）
- 分支 `feat/ptq-export`；`uv run pytest` 全绿才可合入（spec §6）
- F3 已定论：注意力全部保留，量化与导出覆盖全部 6 处注意力块

## Review Focus

1. **per-channel 轴错配**：Conv2d 权重 per-channel 沿 dim 0（out_channels），Linear 沿 dim 0（out_features）；转置/展平后轴错会静默劣化画质——由 Task 2 的 round-trip 测试钉死。
2. **步组索引错位**：标定统计与推理查表必须用同一套 step→group 映射（step 0~3→组 0，…，16~19→组 4）；采样循环步序是 t 从大到小，组号按"第几个推理步"而非 t 值——由 Task 4 测试钉死。
3. **FiLM 注入点**：时间偏置加在 `conv1` 输出之后、`norm2` 之前（ResnetBlock2D 内 `hidden + temb`），不是加在 GN 之后——由 Task 6 导出值与手写模型前向对照测试钉死。
4. **fake-quant 与导出的 scale 一致性**：导出文件里的 scale 必须就是 fake-quant 参考实际使用的值（同一配置、同一标定统计文件），不得二次计算——由 Task 6 的一致性测试钉死。
5. **注意力量化覆盖**：6 个注意力块（GN + 4 个 Linear）必须同样被量化与导出，F3 前写的老代码路径可能漏掉——由 Task 6 导出完整性测试钉死（层描述符覆盖全部 450 个权重 key 所属的模块）。

---

## Task 1: 量化与导出格式契约（`docs/quant-format.md`）

**Files:**
- Create: `docs/quant-format.md`
- Create: `configs/quant_int8.json`

**Interfaces:**
- Consumes: spec §5.4、`docs/reference/unet-config-ddpm-cat-256.json`、`docs/reference/scheduler-config-ddpm-cat-256.json`
- Produces: F4 后续任务与 M2 的共同契约（本任务无代码，产出文档 + 配置）

- [ ] **Step 1: 写契约文档**

`docs/quant-format.md`，内容必须包含以下各节（数值为规定值，不得擅自更改）：

```markdown
# INT8 量化与导出格式契约（M0-F4 产出，M2 输入）

## 1. 量化方案
- 权重：per-(输出)channel 对称 INT8。w_q[c] = round(w[c] / s_w[c])，
  s_w[c] = max|w[c]| / 127。Conv2d 沿 dim 0，Linear 沿 dim 0。
- 激活：per-tensor 静态非对称免零点（对称）INT8。a_q = round(a / s_a)，
  s_a = P99.9(|a|) / 127，按"层 × 步组"各存一套 scale。
- 步组：DDIM-20 的 20 个推理步按顺序分 5 组（步 0-3, 4-7, 8-11, 12-15, 16-19），
  组号按推理步序号（非 timestep 值）。
- 量化点：每个"层描述符算子"的输出（conv/linear/attention-out/GN/SiLU/
  residual-add/concat/上采样/下采样/最终 conv_out）。GN 与 SiLU 的整数实现
  定义属于 F5（位真模拟器），本契约只规定其输入输出量化点。
- 累加与重整：INT8×INT8 乘、INT32 累加；requant y = round(acc × s_in × s_w[c] / s_out)，
  截断饱和到 [-128, 127]。F5 以整数/定点复现该式并定义 RTL 精确行为。

## 2. 标定策略
- 标定集：seed 200..203 共 4 个初始噪声 × DDIM-20 全部 20 步，256×256。
- 统计量：每个量化点、每个步组的 P99.9(|activation|)。
- 产物：artifacts/f4/calib-stats.json（不入库）；其内容哈希写入导出文件头。

## 3. 导出文件
- export/weights.bin：自定义二进制，头部 magic "CDW1" + 层数 + 每记录
  {name, op, shape, per-channel scale(fp32), int8 payload}；层序 = 层描述符顺序。
- export/layers.json：有序层描述符数组 {index, name, op, cin, cout, k, stride,
  pad, quant_group_scales_ref, inputs, outputs}。
- export/ddim_table.json：DDIM-20 逐步 {t, alpha_t, alpha_prev, sqrt_alpha_t,
  sqrt_one_minus_alpha_t, sqrt_alpha_prev, sqrt_one_minus_alpha_prev}（fp64 计算 fp32 存储）。
- export/film_table.bin：对 20 个离散 timestep 预计算每个 ResnetBlock2D 的
  time_emb_proj(silu(temb)) 偏置向量，布局 [20][resnet_idx][out_channels] fp32，
  resnet_idx 按层描述符顺序；运行时按推理步查表，硬件无需时间嵌入 MLP。
```

`configs/quant_int8.json`：

```json
{
  "weight_quant": {"type": "per_channel_symmetric_int8", "axis": 0},
  "act_quant": {"type": "per_tensor_symmetric_int8", "percentile": 99.9},
  "schedule": {"num_inference_steps": 20, "num_step_groups": 5},
  "calibration": {"seeds": [200, 201, 202, 203], "image_size": 256},
  "model_id": "google/ddpm-cat-256",
  "unet_config": "docs/reference/unet-config-ddpm-cat-256.json"
}
```

- [ ] **Step 2: 提交**

```bash
git add docs/quant-format.md configs/quant_int8.json
git commit -m "docs: INT8 量化与导出格式契约 + 集中配置（M0-F4-T1）"
```

---

## Task 2: 权重量化器（`quant/weights.py`）

**Files:**
- Create: `src/catdiff/quant/weights.py`
- Test: `tests/test_quant_weights.py`

**Interfaces:**
- Consumes: 无
- Produces: `catdiff.quant.weights.quantize_per_channel(w: torch.Tensor, axis: int = 0) -> (w_q: torch.Tensor(int8), scales: torch.Tensor(fp32))` 与 `dequantize_per_channel(w_q, scales, axis=0) -> torch.Tensor`；round-trip 保证 `w ≈ dequant(quant(w))`

- [ ] **Step 1: 写失败测试**

`tests/test_quant_weights.py`：

```python
import torch

from catdiff.quant.weights import dequantize_per_channel, quantize_per_channel


def test_conv_weight_roundtrip_per_out_channel():
    torch.manual_seed(0)
    w = torch.randn(64, 32, 3, 3) * torch.logspace(-3, 0, 64).reshape(64, 1, 1, 1)
    w_q, scales = quantize_per_channel(w, axis=0)
    assert w_q.dtype == torch.int8 and scales.shape == (64,)
    assert w_q.abs().max() <= 127
    # 每个输出通道至少一个元素打满量程
    assert (w_q.abs().amax(dim=(1, 2, 3)) == 127).all()
    w_hat = dequantize_per_channel(w_q, scales, axis=0)
    rel = ((w_hat - w).abs().max() / w.abs().max()).item()
    assert rel < 0.02  # 每通道独立缩放，大幅值通道误差有界


def test_linear_weight_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(512, 128)
    w_q, scales = quantize_per_channel(w, axis=0)
    assert scales.shape == (512,)
    w_hat = dequantize_per_channel(w_q, scales, axis=0)
    assert ((w_hat - w).abs().max() / w.abs().max()).item() < 0.02


def test_zero_channel_is_safe():
    w = torch.zeros(4, 8)
    w_q, scales = quantize_per_channel(w, axis=0)
    assert torch.isfinite(scales).all() and w_q.abs().max() == 0
```

```bash
uv run pytest tests/test_quant_weights.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现**

`src/catdiff/quant/weights.py` 完整内容：

```python
"""权重 per-channel 对称 INT8 量化（契约 docs/quant-format.md §1）。"""

import torch

_EPS = 1e-12


def quantize_per_channel(w: torch.Tensor, axis: int = 0):
    dims = [d for d in range(w.ndim) if d != axis]
    max_abs = w.abs().amax(dim=dims)
    scales = (max_abs / 127.0).clamp_min(_EPS)
    shape = [1] * w.ndim
    shape[axis] = -1
    w_q = torch.round(w / scales.reshape(shape)).clamp(-128, 127).to(torch.int8)
    return w_q, scales.to(torch.float32)


def dequantize_per_channel(w_q: torch.Tensor, scales: torch.Tensor, axis: int = 0):
    shape = [1] * w_q.ndim
    shape[axis] = -1
    return w_q.to(torch.float32) * scales.reshape(shape)
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_quant_weights.py -v   # 3 个 PASS
git add src/catdiff/quant/weights.py tests/test_quant_weights.py
git commit -m "feat: 权重 per-channel INT8 量化器（M0-F4-T2）"
```

---

## Task 3: 标定数据采集（`quant/calibrate.py`，slow）

**Files:**
- Create: `src/catdiff/quant/calibrate.py`
- Test: `tests/test_calibrate.py`（统计逻辑单测，不触网）
- 产物: `artifacts/f4/calib-stats.json`（slow 步骤生成，不入库）

**Interfaces:**
- Consumes: F2 手写模型、`forward_with_trace`、`configs/quant_int8.json`
- Produces: `catdiff.quant.calibrate.step_to_group(step_idx: int, num_steps: int, num_groups: int) -> int`、`catdiff.quant.calibrate.update_percentile_stats(stats, name, group, tensor) -> None`、`catdiff.quant.calibrate.run_calibration(out_path) -> None`（slow）；stats 结构 `{layer_name: {group_idx: [p99.9 样本值列表]}}`，最终 scale = 各组列表的 P99.9

- [ ] **Step 1: 写失败测试**

`tests/test_calibrate.py`：

```python
import torch

from catdiff.quant.calibrate import step_to_group, finalize_scales, update_percentile_stats


def test_step_to_group_mapping():
    # 20 步 5 组：步 0-3→0, 4-7→1, 8-11→2, 12-15→3, 16-19→4
    expect = [0]*4 + [1]*4 + [2]*4 + [3]*4 + [4]*4
    assert [step_to_group(i, 20, 5) for i in range(20)] == expect


def test_stats_finalize_uses_percentile():
    stats = {}
    g = torch.Generator().manual_seed(0)
    for _ in range(8):
        update_percentile_stats(stats, "layer0", 0, torch.randn(256, generator=g) * 3.0)
    scales = finalize_scales(stats, percentile=99.9)
    assert "layer0" in scales and 0 in scales["layer0"]
    s = scales["layer0"][0]
    assert s > 0
    # scale = P99.9(|a|)/127，应明显大于 std=3 的 1 倍
    assert s * 127 > 3.0
```

```bash
uv run pytest tests/test_calibrate.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现 calibrate.py**

`src/catdiff/quant/calibrate.py` 完整内容：

```python
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
```

- [ ] **Step 3: 跑单测，然后 slow 标定（约 3~4 分钟）**

```bash
uv run pytest tests/test_calibrate.py -v                          # 2 个 PASS
uv run python -c "from catdiff.quant.calibrate import run_calibration; run_calibration()"
```

预期：`artifacts/f4/calib-stats.json` 生成，覆盖全部 trace 层名 × 5 个步组。

- [ ] **Step 4: 提交**

```bash
git add src/catdiff/quant/calibrate.py tests/test_calibrate.py
git commit -m "feat: 按步组激活标定采集（M0-F4-T3）"
```

---

## Task 4: fake-quant 参考模型（`quant/fakequant.py`）

**Files:**
- Create: `src/catdiff/quant/fakequant.py`
- Test: `tests/test_fakequant.py`

**Interfaces:**
- Consumes: Task 2 量化器、Task 3 标定 scale、F2 手写模型
- Produces: `catdiff.quant.fakequant.FakeQuantContext(scales: dict, num_steps: int, num_groups: int)`（持有当前步序，`.set_step(i)`）与 `catdiff.quant.fakequant.apply_fake_quant(model, ctx) -> int`（给 conv/linear 包权重 Q/DQ、在量化点插激活 Q/DQ hook，返回包装数）；采样循环每步先 `ctx.set_step(i)` 再前向

- [ ] **Step 1: 写失败测试**

`tests/test_fakequant.py`：

```python
import torch

from catdiff.model.unet import UNet2D
from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant
from tests.test_unet import TINY_CONFIG


def _scales_for(model, value=0.01):
    return {name: {g: value for g in range(5)}
            for name, _ in model.named_modules() if name}


def test_fakequant_changes_but_preserves_shape():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    ctx = FakeQuantContext(_scales_for(model), num_steps=20, num_groups=5)
    n = apply_fake_quant(model, ctx)
    assert n > 0
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    ref_model = UNet2D(TINY_CONFIG).eval()
    ref_model.load_state_dict(model.state_dict())  # 权重本身未被破坏
    with torch.no_grad():
        ctx.set_step(0)
        out_q = model(x, t)
        out_f = ref_model(x, t)
    assert out_q.shape == out_f.shape
    assert not torch.allclose(out_q, out_f)


def test_step_group_switches_scale():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    scales = _scales_for(model)
    key = "conv_in"  # 必须选挂了 hook 的量化点（_DEFAULT_PATTERN 匹配的层）
    scales[key][0], scales[key][4] = 0.001, 1.0
    ctx = FakeQuantContext(scales, num_steps=20, num_groups=5)
    apply_fake_quant(model, ctx)
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        ctx.set_step(0)
        out_g0 = model(x, t).clone()
        ctx.set_step(19)
        out_g4 = model(x, t).clone()
    assert not torch.allclose(out_g0, out_g4)  # 不同步组 scale 生效
```

```bash
uv run pytest tests/test_fakequant.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现 fakequant.py**

`src/catdiff/quant/fakequant.py` 完整内容：

```python
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
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_fakequant.py -v   # 2 个 PASS
git add src/catdiff/quant/fakequant.py tests/test_fakequant.py
git commit -m "feat: fake-quant 参考模型（权重 per-channel + 激活按步组）（M0-F4-T4）"
```

---

## Task 5: INT8 画质关卡（slow + 人工签字）

**Files:**
- Create: `src/catdiff/quant/sample_int8.py`
- Create: `docs/quality-gates/f4-int8-quality-report.md`（运行后填写）
- 产物：`artifacts/f4/int8-ddim20/`（不入库）

**Interfaces:**
- Consumes: Task 3 标定文件、Task 4 fake-quant、F1 CLI 的 grid/metadata 工具
- Produces: `catdiff.quant.sample_int8.main()`（CLI：`uv run python -m catdiff.quant.sample_int8 --out-dir ... --num-samples 20 --seed 100`）；INT8 验收报告

- [ ] **Step 1: 实现采样入口**

`src/catdiff/quant/sample_int8.py` 完整内容：

```python
"""INT8 fake-quant 采样：DDIM-20 @256×256，产出画质关卡样本。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from catdiff.baseline.sampling import make_grid, tensor_to_pil
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import load_handwritten_unet
from catdiff.quant.fakequant import FakeQuantContext, apply_fake_quant


@torch.no_grad()
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="INT8 fake-quant 采样（画质关卡）")
    p.add_argument("--quant-config", default="configs/quant_int8.json")
    p.add_argument("--calib-stats", default="artifacts/f4/calib-stats.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    raw = json.loads(Path(args.calib_stats).read_text(encoding="utf-8"))
    scales = {k: {int(g): v for g, v in groups.items()} for k, groups in raw.items()}

    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    n_steps = cfg["schedule"]["num_inference_steps"]
    ctx = FakeQuantContext(scales, n_steps, cfg["schedule"]["num_step_groups"])
    n = apply_fake_quant(model, ctx)
    print(f"fake-quant 施加 {n} 处")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for idx in range(args.num_samples):
        path = args.out_dir / f"seed{args.seed}_{idx:04d}.png"
        if path.exists():
            continue
        g = torch.Generator().manual_seed(args.seed + idx)
        x = torch.randn(1, 3, cfg["calibration"]["image_size"],
                        cfg["calibration"]["image_size"], generator=g)
        sched = DDIMScheduler(num_train_timesteps=1000)
        sched.set_timesteps(n_steps)
        for step_idx, t in enumerate(sched.timesteps):
            ctx.set_step(step_idx)
            x = sched.step(model(x, t), t, x).prev_sample
        tensor_to_pil(x[0]).save(path)
        print(f"[{idx + 1}/{args.num_samples}] {path.name} 完成", flush=True)

    images = [Image.open(args.out_dir / f"seed{args.seed}_{idx:04d}.png")
              for idx in range(args.num_samples)]
    make_grid(images, 4).save(args.out_dir / "grid.png")
    (args.out_dir / "metadata.json").write_text(json.dumps({
        "backend": "handwritten-int8-fakequant",
        "quant_config": cfg, "seed": args.seed,
        "num_samples": args.num_samples,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 先跑 2 张冒烟（约 2 分钟），再跑 20 张正式批次（约 15 分钟）**

```bash
uv run python -m catdiff.quant.sample_int8 --out-dir artifacts/f4/int8-smoke --num-samples 2
uv run python -m catdiff.quant.sample_int8 --out-dir artifacts/f4/int8-ddim20 --num-samples 20
```

预期：冒烟 2 张可辨猫（若全糊，先排查 scale 映射/量化点，不得继续）；正式批次 20 张。

- [ ] **Step 3: 写报告并停止（人工签字）**

`docs/quality-gates/f4-int8-quality-report.md` 模板：

```markdown
# F4 INT8 画质关卡报告

| 项 | 值 |
|---|---|
| 日期 | <填写> |
| 模型 | cat-256 手写复刻 + PTQ INT8（权重 per-channel + 激活 5 步组标定） |
| 采样 | 手写 DDIM eta=0, 20 步, seed=100, 20 张 @256×256 |
| fp32 对照 | artifacts/f3/full-ddim20/grid.png |
| INT8 批次 | artifacts/f4/int8-ddim20/grid.png |

## 人工验收

- [ ] INT8 可辨猫占比：__/20
- [ ] 与 fp32 对照的差异：＿＿＿＿（无可见差异 / 略退化可接受 / 明显退化）
- [ ] 结论：
  - [ ] **Go**：INT8 画质可接受 → 进入 F5（位真模拟器按本方案做黄金向量）
  - [ ] **No-go**：触发总提案 §4.4 备选（激活 fp16 + 权重 INT8 混合精度），通知 PM 修订 spec

验收人签字：____  日期：____
```

**停止点**：报告生成后由项目负责人对照两组 grid 签字。No-go 则停止并通知 PM。

- [ ] **Step 4: 提交**

```bash
git add src/catdiff/quant/sample_int8.py docs/quality-gates/f4-int8-quality-report.md
git commit -m "feat: INT8 fake-quant 采样入口 + 画质关卡报告（M0-F4-T5）"
```

---

## Task 6: 硬件导出器（`export/`）

**Files:**
- Create: `src/catdiff/export/ddim_table.py`
- Create: `src/catdiff/export/film_table.py`
- Create: `src/catdiff/export/weights_export.py`
- Create: `src/catdiff/export/run_export.py`
- Test: `tests/test_export.py`
- 产物：`export/` 目录（**入库**，是 M2 输入；见下方说明）

**Interfaces:**
- Consumes: Task 2/3/4、F2 手写模型、契约文档 §3
- Produces: `export/weights.bin`、`export/layers.json`、`export/ddim_table.json`、`export/film_table.bin`；`catdiff.export.run_export.main()` 一键重新生成

**产物入库说明**：`.gitignore` 既有条目 `export/weights/*.bin` 只忽略 `export/weights/` 子目录；本任务产物在 `export/` 根下。weights.bin（约 45MB）超过 git 常规体积——按总提案 §3.3 与 .gitignore 注释"权重与比特流不进 git，按需走 Release"，本任务**只入库** `layers.json`、`ddim_table.json` 与重新生成脚本；`weights.bin`、`film_table.bin` 不入库（追加 `export/*.bin` 到 .gitignore）。

- [ ] **Step 1: 写失败测试**

`tests/test_export.py`：

```python
import json

import numpy as np
import torch

from catdiff.export.ddim_table import build_ddim_table
from catdiff.export.film_table import build_film_vectors
from catdiff.export.weights_export import iter_quantized_layers
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import UNet2D
from tests.test_unet import TINY_CONFIG


def test_ddim_table_matches_scheduler():
    table = build_ddim_table(num_inference_steps=20)
    assert len(table) == 20
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(20)
    for row, t in zip(table, sched.timesteps.tolist()):
        assert row["t"] == t
    # 首行 t 最大，末行 alpha_prev == 1.0（set_alpha_to_one）
    assert table[0]["t"] > table[-1]["t"]
    assert abs(table[-1]["alpha_prev"] - 1.0) < 1e-6


def test_iter_quantized_layers_covers_all_param_modules():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    layers = list(iter_quantized_layers(model))
    names = {name for name, _, _, _ in layers}
    import torch.nn as nn
    expect = {n for n, m in model.named_modules()
              if isinstance(m, (nn.Conv2d, nn.Linear))}
    assert names == expect  # 含全部注意力的 to_q/to_k/to_v/to_out.0
    for name, w_q, scales, bias in layers:
        assert w_q.dtype == torch.int8
        assert scales.shape[0] == w_q.shape[0]


def test_film_vectors_match_resnet_injection():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    t_value = 250
    vecs = build_film_vectors(model, [t_value])  # {resnet_name: {t: vector}}
    # 任取一个 resnet，独立计算 time_emb_proj(silu(temb)) 比对
    from catdiff.model.time_embed import timestep_embedding
    name, table = next(iter(vecs.items()))
    resnet = dict(model.named_modules())[name]
    temb = model.time_embedding(
        model.time_proj(torch.tensor([float(t_value)])))
    expect = resnet.time_emb_proj(torch.nn.functional.silu(temb))[0]
    got = torch.tensor(table[t_value])
    assert torch.allclose(got, expect, atol=1e-5)
```

```bash
uv run pytest tests/test_export.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现导出器**

`src/catdiff/export/ddim_table.py`：

```python
"""DDIM 常数表导出（契约 §3）：硬件零运行时超越函数。"""

from catdiff.model.ddim import DDIMScheduler


def build_ddim_table(num_inference_steps: int = 20,
                     num_train_timesteps: int = 1000) -> list[dict]:
    sched = DDIMScheduler(num_train_timesteps=num_train_timesteps)
    sched.set_timesteps(num_inference_steps)
    rows = []
    for t in sched.timesteps.tolist():
        prev_t = t - sched._step_ratio
        alpha_t = sched.alphas_cumprod[t].item()
        alpha_prev = (sched.alphas_cumprod[prev_t].item() if prev_t >= 0
                      else sched.final_alpha_cumprod.item())
        rows.append({
            "t": t,
            "alpha_t": alpha_t,
            "alpha_prev": alpha_prev,
            "sqrt_alpha_t": alpha_t ** 0.5,
            "sqrt_one_minus_alpha_t": (1 - alpha_t) ** 0.5,
            "sqrt_alpha_prev": alpha_prev ** 0.5,
            "sqrt_one_minus_alpha_prev": (1 - alpha_prev) ** 0.5,
        })
    return rows
```

`src/catdiff/export/film_table.py`：

```python
"""FiLM 表导出（契约 §3）：预计算每个 ResnetBlock2D 的时间偏置向量。"""

import torch

from catdiff.model.resnet import ResnetBlock2D


@torch.no_grad()
def build_film_vectors(model, timesteps: list[int]) -> dict:
    """{resnet_name: {t: [out_channels] 列表}}，resnet 按 named_modules 顺序。"""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, ResnetBlock2D):
            continue
        table = {}
        for t in timesteps:
            temb = model.time_embedding(
                model.time_proj(torch.tensor([float(t)])))
            vec = mod.time_emb_proj(torch.nn.functional.silu(temb))[0]
            table[t] = vec.tolist()
        out[name] = table
    return out
```

`src/catdiff/export/weights_export.py`：

```python
"""权重量化遍历（契约 §3）：遍历模型全部 Conv2d/Linear，产出 INT8 + scale。"""

import torch
from torch import nn

from catdiff.quant.weights import quantize_per_channel


def iter_quantized_layers(model: nn.Module):
    """yield (name, weight_int8, scales_fp32, bias_fp32|None)，按 named_modules 顺序。"""
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            w_q, scales = quantize_per_channel(mod.weight.detach(), axis=0)
            bias = None if mod.bias is None else mod.bias.detach().to(torch.float32)
            yield name, w_q, scales, bias
```

`src/catdiff/export/run_export.py`：

```python
"""一键导出：weights.bin / layers.json / ddim_table.json / film_table.bin（契约 §3）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from catdiff.export.ddim_table import build_ddim_table
from catdiff.export.film_table import build_film_vectors
from catdiff.export.weights_export import iter_quantized_layers
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.unet import load_handwritten_unet

_MAGIC = b"CDW1"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="PTQ INT8 硬件导出")
    p.add_argument("--quant-config", default="configs/quant_int8.json")
    p.add_argument("--calib-stats", default="artifacts/f4/calib-stats.json")
    p.add_argument("--out-dir", type=Path, default=Path("export"))
    args = p.parse_args(argv)

    cfg = json.loads(Path(args.quant_config).read_text(encoding="utf-8"))
    unet_cfg = json.loads(Path(cfg["unet_config"]).read_text(encoding="utf-8"))
    calib_raw = Path(args.calib_stats).read_bytes()
    model = load_handwritten_unet(cfg["model_id"], unet_cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    layers = list(iter_quantized_layers(model))
    calib_hash = hashlib.sha256(calib_raw).hexdigest()[:16]

    # weights.bin：magic + 层数 + 逐层 {name_len, name, ndim, shape, scales, payload}
    with open(args.out_dir / "weights.bin", "wb") as f:
        f.write(_MAGIC)
        f.write(struct.pack("<I", len(layers)))
        for name, w_q, scales, bias in layers:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb)
            w = w_q.numpy()
            f.write(struct.pack("<I", w.ndim))
            f.write(struct.pack(f"<{w.ndim}I", *w.shape))
            has_bias = 1 if bias is not None else 0
            f.write(struct.pack("<B", has_bias))
            f.write(scales.numpy().astype("<f4").tobytes())
            if has_bias:
                f.write(bias.numpy().astype("<f4").tobytes())
            f.write(w.astype("<i1").tobytes())

    # layers.json：算子级描述符（含量化点与 scale 引用）
    desc = []
    for idx, (name, w_q, scales, bias) in enumerate(layers):
        desc.append({
            "index": idx, "name": name,
            "op": "conv2d" if w_q.ndim == 4 else "linear",
            "shape": list(w_q.shape),
            "scale_ref": f"weights.bin#{idx}",
        })
    (args.out_dir / "layers.json").write_text(
        json.dumps({"calib_hash": calib_hash, "layers": desc}, indent=1),
        encoding="utf-8")

    # ddim_table.json
    n_steps = cfg["schedule"]["num_inference_steps"]
    (args.out_dir / "ddim_table.json").write_text(
        json.dumps(build_ddim_table(n_steps), indent=1), encoding="utf-8")

    # film_table.bin：[20][resnet_idx][out_channels] fp32，resnet 顺序 = layers.json 中
    # time_emb_proj 出现顺序
    sched = DDIMScheduler(num_train_timesteps=1000)
    sched.set_timesteps(n_steps)
    film = build_film_vectors(model, sched.timesteps.tolist())
    resnet_names = list(film.keys())
    with open(args.out_dir / "film_table.bin", "wb") as f:
        f.write(struct.pack("<II", n_steps, len(resnet_names)))
        for t in sched.timesteps.tolist():
            for name in resnet_names:
                f.write(np.array(film[name][t], dtype="<f4").tobytes())
    (args.out_dir / "film_index.json").write_text(
        json.dumps({"timesteps": sched.timesteps.tolist(),
                    "resnet_names": resnet_names}, indent=1), encoding="utf-8")

    print(f"导出完成：{len(layers)} 层，{len(resnet_names)} 个 resnet 的 FiLM 表")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`.gitignore` 追加一行：

```
export/*.bin
```

- [ ] **Step 3: 跑单测，然后 slow 导出（约 1 分钟）**

```bash
uv run pytest tests/test_export.py -v   # 3 个 PASS
uv run python -m catdiff.export.run_export
ls -la export/
```

预期：四个产物 + film_index.json 生成；layers.json 层数 = 手写模型的 Conv2d+Linear 总数（cat-256 为 450 个权重 key 所属模块数）。

- [ ] **Step 4: 提交**

```bash
git add src/catdiff/export tests/test_export.py .gitignore export/layers.json export/ddim_table.json export/film_index.json
git commit -m "feat: 硬件导出器（weights.bin/层描述符/DDIM 表/FiLM 表）（M0-F4-T6）"
```

---

## 分发提示词（交给执行 agent 用）

```
你在仓库 C:\git\fpga-cat-diffusion 工作。严格按 docs/superpowers/plans/2026-09-25-m0-f4-ptq-export.md 逐任务执行（Task 1~6 内有完整代码与命令）。硬性要求：
- 单元测试绝不调用 from_pretrained；真实权重只在 slow 步骤/CLI 中使用
- 量化方案以 configs/quant_int8.json 与 docs/quant-format.md 为准，不得擅自改参数（步组 5 组、P99.9、标定 seed 200..203）
- Task 5 是人工签字停止点：冒烟 2 张全糊必须先排查，报告生成后停下等验收
- weights.bin 与 film_table.bin 不入库（.gitignore 已含 export/*.bin）；完成后 uv run pytest 全绿，不合入 main
开始前先读 docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md 的 §5.4 与 docs/quant-format.md（Task 1 产出前读 spec 即可）。偏离计划需在提交信息中说明理由。
```
