# M0-F0/F1 实施计划：项目脚手架 + fp32 基线与画质关卡

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 uv 管理的 Python 工程骨架，并用 diffusers 跑通两个扩散模型在 64×64 下的 DDIM 采样，完成猫模型画质 Go/No-go 关卡。

**Architecture:** `src/catdiff` 包按职责分层（baseline/model/quant/sim/export），本计划只建骨架与 `baseline/` 层。采样逻辑（`sampling.py`）与命令行入口（`cli.py`）分离，单元测试全部使用内存中构造的微型 UNet（不触网），真实模型采样作为独立步骤执行。

**Tech Stack:** Python 3.11+、uv、torch（CPU）、diffusers、numpy、pillow、pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md`（执行前必读，本计划实现其 §5.1 与 §2、§3）

## Global Constraints

- Python `>=3.11`，uv + `pyproject.toml` 管理，依赖锁定提交 `uv.lock`（spec §2）
- **仅 CPU**：代码不得 import/触发任何 CUDA 路径；torch 用 CPU 版（spec §2）
- 所有采样入口接受显式 seed；测试与验收实验固定 seed 可复现（spec §2）
- 采样脚本必须支持 `--num-samples` 与断点续跑（spec §2）
- DDIM 采样 `eta=0.0`（确定性），初始噪声是唯一随机源（总提案 §4.3）
- `artifacts/` 加入 `.gitignore`，采样产物一律不入库（spec §3）
- 每个 feature 一个分支：F0 用 `feat/project-scaffold`，F1 用 `feat/fp32-baseline`；`uv run pytest` 全绿才可合入（spec §6）

## Review Focus

1. **eta 漂移**：若采样未固定 `eta=0.0`，DDIM 会引入额外随机源，违背"初始噪声是唯一随机源"——由 `test_cli_writes_metadata` 中断言 `metadata["eta"] == 0.0` 钉死（F1 Task 2）。
2. **单元测试触网**：任何测试调用 `from_pretrained` 都会下载数百 MB 权重并引入不确定性——所有测试用 `tiny_unet()` 本地构造；真实模型只在人工执行的采样步骤中使用（F1 Task 1 约定）。
3. **跨分辨率假设**：`ddpm-cat-256` 的 `sample_size` 配置是 256，本项目以 64 输入——由 `test_cross_resolution_output_shape` 钉死输出形状等于输入形状（F1 Task 1）。
4. **断点续跑复现性**：续跑时同一 idx 必须得到逐字节相同的图（seed 推导不变式 `global_seed = seed + idx`）——由 `test_per_sample_seed_invariant` 与 `test_cli_resume_idempotent` 钉死（F1 Task 1、2）。
5. **Linux 下 torch 装上 CUDA 版**：PyPI 的 Linux torch 默认带 CUDA（体积数 GB）——F0 Task 1 验证步骤检查 `torch.cuda.is_available()` 行为与安装体积，Windows 无此问题但需确认（F0 Task 1）。
6. **`model.config` 序列化**：导出 config.json 时若含不可序列化字段会在真实模型运行时才暴露——由 `test_cli_writes_metadata` 用微型模型先行覆盖（F1 Task 2）。

---

## Task F0: 项目脚手架（分支 `feat/project-scaffold`）

**Files:**
- Create: `pyproject.toml`
- Create: `src/catdiff/__init__.py`
- Create: `tests/__init__.py`、`tests/test_smoke.py`
- Modify: `.gitignore`（追加条目）
- Modify: `README.md`（追加开发说明一节）

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: 包 `catdiff`（`__version__ = "0.1.0"`）；目录骨架 `src/catdiff/{baseline,model,quant,sim,export}/`（各含空 `__init__.py`）；后续所有任务在该骨架上添加文件

- [ ] **Step 1: 建分支**

```bash
git checkout -b feat/project-scaffold
```

- [ ] **Step 2: 写 pyproject.toml 与 .gitignore**

`pyproject.toml` 完整内容：

```toml
[project]
name = "catdiff"
version = "0.1.0"
description = "FPGA cat diffusion - PC algorithm pipeline (M0)"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "diffusers>=0.30",
    "numpy>=1.26",
    "pillow>=10.0",
    "huggingface-hub>=0.24",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/catdiff"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore` 追加：

```
.venv/
__pycache__/
*.pyc
artifacts/
dist/
```

- [ ] **Step 3: 同步环境并验证 torch 为 CPU 可用**

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

预期：打印版本号；Windows 上 `False`。若在 Linux 上为 `True` 且安装体积数 GB，改为 CPU 索引源重装（`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`），并把该约束记入 README。

- [ ] **Step 4: 写冒烟测试（先失败）**

`tests/__init__.py`：空文件。

`tests/test_smoke.py`：

```python
def test_package_imports():
    import catdiff

    assert catdiff.__version__ == "0.1.0"


def test_torch_cpu_tensor_works():
    import torch

    t = torch.randn(2, 3)
    assert t.shape == (2, 3)
```

```bash
uv run pytest tests/test_smoke.py -v
```

预期：`test_package_imports` FAIL（`ModuleNotFoundError: No module named 'catdiff'`）。

- [ ] **Step 5: 建包骨架使测试通过**

```bash
mkdir -p src/catdiff/baseline src/catdiff/model src/catdiff/quant src/catdiff/sim src/catdiff/export
```

`src/catdiff/__init__.py`：

```python
__version__ = "0.1.0"
```

`src/catdiff/{baseline,model,quant,sim,export}/__init__.py`：均为空文件。

```bash
uv run pytest -v
```

预期：2 个测试全 PASS。

- [ ] **Step 6: README 追加开发说明**

在 `README.md` 末尾追加：

```markdown
## 开发环境（M0 PC 算法管线）

```bash
uv sync                 # 安装全部依赖（含 pytest）
uv run pytest           # 运行测试
uv run python -m catdiff.baseline.cli --help   # 采样入口（F1 起可用）
```

仅 CPU 环境即可；`artifacts/` 为采样产物目录，不入库。
```

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml uv.lock .gitignore README.md src tests
git commit -m "feat: uv 项目脚手架与包骨架（M0-F0）"
```

---

## Task F1: fp32 基线采样与画质关卡（分支 `feat/fp32-baseline`）

**Files:**
- Create: `src/catdiff/baseline/sampling.py`
- Create: `src/catdiff/baseline/cli.py`
- Create: `tests/conftest.py`
- Test: `tests/test_sampling.py`
- Test: `tests/test_cli.py`
- Create: `docs/quality-gates/f1-cat64-report.md`（运行后由人工填写结论）

**Interfaces:**
- Consumes: F0 的包骨架与 pytest 配置
- Produces（后续任务依赖的精确签名）:
  - `catdiff.baseline.sampling.load_unet(model_id: str) -> UNet2DModel`
  - `catdiff.baseline.sampling.load_ddim(model_id: str, num_inference_steps: int) -> DDIMScheduler`（内部已 `set_timesteps`）
  - `catdiff.baseline.sampling.sample_batch(model, scheduler, *, num_samples: int, image_size: int, seed: int, device: str = "cpu") -> torch.Tensor`，返回 `(N, 3, H, W)`、值域 `[-1, 1]`；第 `idx` 张的初始噪声 seed 恒为 `seed + idx`
  - `catdiff.baseline.sampling.tensor_to_pil(img: torch.Tensor) -> PIL.Image.Image`（`(3,H,W)`、`[-1,1]` → uint8 RGB）
  - `catdiff.baseline.sampling.make_grid(images: list[Image.Image], cols: int) -> Image.Image`
  - CLI：`uv run python -m catdiff.baseline.cli --model-id <id> --out-dir <dir> [--num-samples N] [--image-size 64] [--num-inference-steps 50] [--seed 0] [--cols 4]`
  - 产物契约：`out-dir/` 下 `seed{S}_{idx:04d}.png`、`grid.png`、`metadata.json`、`config.json`；`config.json` 将提交至 `docs/reference/` 供 F2 手写 UNet 使用

- [ ] **Step 1: 建分支并写测试辅助**

```bash
git checkout main && git checkout -b feat/fp32-baseline
```

`tests/conftest.py`（微型模型只在内存构造，任何测试不得触网）：

```python
import pytest
from diffusers import DDIMScheduler, UNet2DModel


@pytest.fixture
def tiny_unet():
    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )


@pytest.fixture
def tiny_scheduler():
    def _make(steps: int = 3) -> DDIMScheduler:
        s = DDIMScheduler(num_train_timesteps=1000)
        s.set_timesteps(steps)
        return s

    return _make
```

- [ ] **Step 2: 写 sampling 的失败测试**

`tests/test_sampling.py`：

```python
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
```

```bash
uv run pytest tests/test_sampling.py -v
```

预期：全部 FAIL（`ModuleNotFoundError: catdiff.baseline.sampling` 或 `ImportError`）。

- [ ] **Step 3: 实现 sampling.py**

`src/catdiff/baseline/sampling.py` 完整内容：

```python
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
```

```bash
uv run pytest tests/test_sampling.py -v
```

预期：5 个测试全 PASS（`test_per_sample_seed_invariant` 验证续跑复现不变式）。

- [ ] **Step 4: 写 CLI 的失败测试**

`tests/test_cli.py`：

```python
import json

from catdiff.baseline import cli


def _run(out, monkeypatch, tiny_unet, tiny_scheduler):
    monkeypatch.setattr(cli, "load_unet", lambda model_id: tiny_unet)
    monkeypatch.setattr(cli, "load_ddim", lambda model_id, steps: tiny_scheduler(2))
    rc = cli.main(
        [
            "--model-id", "tiny",
            "--num-samples", "3",
            "--image-size", "32",
            "--num-inference-steps", "2",
            "--seed", "5",
            "--out-dir", str(out),
        ]
    )
    assert rc == 0


def test_cli_resume_idempotent(tmp_path, monkeypatch, tiny_unet, tiny_scheduler):
    out = tmp_path / "run"
    _run(out, monkeypatch, tiny_unet, tiny_scheduler)
    first = {p.name: p.read_bytes() for p in sorted(out.glob("*.png"))}
    assert {"seed5_0000.png", "seed5_0001.png", "seed5_0002.png", "grid.png"} <= set(first)

    _run(out, monkeypatch, tiny_unet, tiny_scheduler)  # 全部命中续跑，不得重写
    second = {p.name: p.read_bytes() for p in sorted(out.glob("*.png"))}
    assert first == second


def test_cli_writes_metadata(tmp_path, monkeypatch, tiny_unet, tiny_scheduler):
    out = tmp_path / "run"
    _run(out, monkeypatch, tiny_unet, tiny_scheduler)
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_id"] == "tiny"
    assert meta["eta"] == 0.0
    assert meta["seed"] == 5
    assert meta["num_inference_steps"] == 2
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["sample_size"] == 32
```

```bash
uv run pytest tests/test_cli.py -v
```

预期：FAIL（`cli` 无 `main`）。

- [ ] **Step 5: 实现 cli.py**

`src/catdiff/baseline/cli.py` 完整内容：

```python
"""采样命令行入口。用法见 README。支持断点续跑：已存在的样本 PNG 直接跳过。"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import diffusers
import torch
from PIL import Image

from .sampling import load_ddim, load_unet, make_grid, sample_batch, tensor_to_pil


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DDIM(eta=0) fp32 基线采样")
    p.add_argument("--model-id", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_unet(args.model_id)
    scheduler = load_ddim(args.model_id, args.num_inference_steps)

    t0 = time.time()
    generated = 0
    for idx in range(args.num_samples):
        path = args.out_dir / f"seed{args.seed}_{idx:04d}.png"
        if path.exists():
            continue
        img = sample_batch(
            model, scheduler,
            num_samples=1, image_size=args.image_size, seed=args.seed + idx,
        )
        tensor_to_pil(img[0]).save(path)
        generated += 1
        print(f"[{idx + 1}/{args.num_samples}] {path.name} 完成", flush=True)
    elapsed = time.time() - t0

    images = [
        Image.open(args.out_dir / f"seed{args.seed}_{idx:04d}.png")
        for idx in range(args.num_samples)
    ]
    make_grid(images, args.cols).save(args.out_dir / "grid.png")

    (args.out_dir / "config.json").write_text(
        json.dumps(dict(model.config), indent=2, default=str), encoding="utf-8"
    )
    (args.out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "image_size": args.image_size,
                "num_inference_steps": args.num_inference_steps,
                "eta": 0.0,
                "seed": args.seed,
                "num_samples": args.num_samples,
                "generated_this_run": generated,
                "elapsed_s": round(elapsed, 1),
                "torch": torch.__version__,
                "diffusers": diffusers.__version__,
                "python": platform.python_version(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
uv run pytest -v
```

预期：全部测试 PASS（F0 的 2 个 + sampling 5 个 + cli 2 个）。

- [ ] **Step 6: 提交代码**

```bash
git add src/catdiff/baseline tests docs
git commit -m "feat: fp32 DDIM 基线采样 CLI（确定性、可续跑）（M0-F1）"
```

- [ ] **Step 7: 真实模型冒烟——蝴蝶模型 4 张**

```bash
uv run python -m catdiff.baseline.cli \
  --model-id jfjensen/sd-class-butterflies-64 \
  --out-dir artifacts/f1/butterflies64-smoke \
  --num-samples 4 --num-inference-steps 50 --seed 0
```

预期：首次运行自动下载权重（HF 缓存）；`grid.png` 中 4 张图呈蝴蝶状纹理（不必完美）。记录耗时。

- [ ] **Step 8: 蝴蝶模型正式 20 张 + 猫模型 20 张（耗时长，建议夜间）**

```bash
uv run python -m catdiff.baseline.cli \
  --model-id jfjensen/sd-class-butterflies-64 \
  --out-dir artifacts/f1/butterflies64 --num-samples 20 --num-inference-steps 50 --seed 100

uv run python -m catdiff.baseline.cli \
  --model-id google/ddpm-cat-256 \
  --out-dir artifacts/f1/cat64 --num-samples 20 --num-inference-steps 50 --seed 100
```

预期：猫模型 CPU 每张约数分钟，20 张约 1~2 小时；中断后重跑同命令即续跑。

- [ ] **Step 9: 冻结模型 config 供 F2 使用**

```bash
mkdir -p docs/reference
cp artifacts/f1/cat64/config.json docs/reference/unet-config-ddpm-cat-256.json
cp artifacts/f1/butterflies64/config.json docs/reference/unet-config-butterflies-64.json
git add docs/reference
git commit -m "docs: 冻结两模型 UNet config（F2 手写复现的输入契约）"
```

- [ ] **Step 10: 填写画质关卡报告，人工验收（Go/No-go）**

创建 `docs/quality-gates/f1-cat64-report.md`，内容模板（如实填写）：

```markdown
# F1 画质关卡报告：ddpm-cat-256 @ 64×64

| 项 | 值 |
|---|---|
| 日期 | <填写> |
| 模型 | google/ddpm-cat-256 |
| 采样 | DDIM eta=0, 50 步, seed=100, 20 张 |
| 样本 | artifacts/f1/cat64/grid.png |
| 单张均时 | <从 metadata.json 与日志估算> |

## 人工验收

- [ ] 20 张中可明确辨认出猫的占比：__/20
- [ ] 结论（三选一）：
  - [ ] **Go**：64×64 画质可接受，按主线推进
  - [ ] **升档**：64×64 不足，改 96×96 或 128×128 重测（回总提案 §4.4）
  - [ ] **暂停**：画质严重不达标，回到方向讨论

验收人签字：____  日期：____
```

**停止点**：此报告由项目负责人人工验收签字后才能合入。结论为"升档"或"暂停"时，先回本会话更新 spec，不得继续 F2。

- [ ] **Step 11: 提交报告**

```bash
git add docs/quality-gates/f1-cat64-report.md
git commit -m "docs: F1 画质关卡报告（Go/No-go 已验收）"
```

---

## 分发提示词（交给执行 agent 用）

**F0（分支 `feat/project-scaffold`）：**

```
你在仓库 C:\git\fpga-cat-diffusion 工作。严格按 docs/superpowers/plans/2026-09-25-m0-f0-f1-scaffold-and-fp32-baseline.md 中的 "Task F0" 逐步执行（步骤内有完整代码与命令，勾选项逐一完成），完成后确保 `uv run pytest` 全绿，不要提前合入 main。开始前先读 docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md 的 §2、§3 了解全局约束。偏离计划的行为（如自作主张加依赖、改结构）需在提交信息中说明理由。
```

**F1（分支 `feat/fp32-baseline`）：**

```
你在仓库 C:\git\fpga-cat-diffusion 工作。严格按 docs/superpowers/plans/2026-09-25-m0-f0-f1-scaffold-and-fp32-baseline.md 中的 "Task F1" 逐步执行（步骤内有完整代码与命令）。注意：单元测试绝不调用 from_pretrained（用 tests/conftest.py 的微型模型）；Step 8 的真实模型采样耗时长，可在后台/夜间跑；Step 10 是人工验收停止点——你生成报告模板后停下，由项目负责人填写结论并签字。完成后 `uv run pytest` 全绿，不要合入 main。开始前先读 docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md。
```
