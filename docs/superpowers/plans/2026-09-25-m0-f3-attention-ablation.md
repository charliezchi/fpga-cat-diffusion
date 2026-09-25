# M0-F3 实施计划：注意力消融实验（256×256 Go/No-go）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 256×256 下对比"完整手写模型"与"去注意力变体"的采样画质，决定 RTL 是否需要实现注意力块（省一大块 M2 工作量）。

**Architecture:** 手写 UNet 的注意力块是独立 `AttentionBlock` 模块，消融 = 权重加载后将其替换为 `nn.Identity`（形状不变、无参数、不破坏 state_dict）。采样复用 F1 的 CLI（扩展 `--backend handwritten`），全部批次可断点续跑。

**Tech Stack:** Python 3.11+、uv、torch（CPU）、pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md`（§5.3 是验收依据；执行前必读）

## Global Constraints

- 仅 CPU；单元测试不得调用 `from_pretrained`；真实权重运行走 CLI 步骤或 slow 测试（spec §2）
- 所有采样固定 seed 可复现、支持断点续跑（spec §2）
- 消融实验在 **256×256** 下进行（spec §5.3，v1.1：任何 64×64 的消融直觉不可沿用）
- 分支 `feat/attention-ablation`；`uv run pytest` 全绿才可合入（spec §6）
- cat-256 注意力块共 6 处：down_blocks.4 ×2、mid_block ×1、up_blocks.1 ×3

## Review Focus

1. **替换时机**：`strip_attention` 必须在 `load_state_dict` 之后调用（Identity 无参数，先替换会导致严格加载缺 key）——由 Task 1 测试钉死加载→替换→前向的完整顺序。
2. **ModuleList 内替换**：注意力块挂在 `attentions.N`（ModuleList 数字键）下，`setattr(parent, "0", Identity())` 才有效——由 Task 1 计数测试钉死（tiny 模型必须替换满 6 处）。
3. **CLI 复现性**：消融批次必须用入库的命令行参数跑出来（`--backend handwritten --strip-attention all`），不允许临时脚本——由 Task 2 测试钉死。
4. **对照公平性**：A/B 两组必须同 seed、同初始噪声、同采样器，唯一变量是注意力——由 Task 3 的固定命令与 metadata 钉死。

---

## Task 1: 适配器与消融工具（`model/ablate.py`）

**Files:**
- Create: `src/catdiff/model/ablate.py`
- Modify: `src/catdiff/model/unet.py`（追加 `as_diffusers_output`）
- Modify: `tests/test_integration_slow.py`（删除本地 `_AsDiffusersOutput`，改为 import）
- Test: `tests/test_ablate.py`

**Interfaces:**
- Consumes: F2 的 `UNet2D`/`AttentionBlock`/`load_handwritten_unet`
- Produces（F4/F5 依赖）:
  - `catdiff.model.unet.as_diffusers_output(model, in_channels: int = 3)`：包装手写模型，暴露 `.config.in_channels` 与 `__call__(x, t) -> SimpleNamespace(sample=...)`、`.to(device)`，可直接传给 `catdiff.baseline.sampling.sample_batch`
  - `catdiff.model.ablate.strip_attention(model, keep_mid: bool = False) -> int`：**在 load_state_dict 之后调用**，将所有（或仅非 mid 的）`AttentionBlock` 替换为 `nn.Identity`，返回替换数量

- [ ] **Step 1: 建分支，写失败测试**

```bash
git checkout main && git checkout -b feat/attention-ablation
```

`tests/test_ablate.py`：

```python
import torch

from catdiff.model.ablate import strip_attention
from catdiff.model.unet import UNet2D, as_diffusers_output
from tests.test_unet import TINY_CONFIG


def _make():
    torch.manual_seed(0)
    return UNet2D(TINY_CONFIG).eval()


def test_strip_attention_replaces_all_six():
    model = _make()
    n = strip_attention(model)
    assert n == 6  # tiny 配置：down_blocks.2 x2 + mid x1 + up_blocks.0 x3


def test_strip_keep_mid():
    model = _make()
    n = strip_attention(model, keep_mid=True)
    assert n == 5  # 保留 mid_block.attentions.0


def test_forward_after_strip_runs_and_changes_output():
    torch.manual_seed(0)
    model = UNet2D(TINY_CONFIG).eval()
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        out_full = model(x, t)
        strip_attention(model)
        out_stripped = model(x, t)
    assert out_stripped.shape == out_full.shape
    assert not torch.allclose(out_stripped, out_full)


def test_adapter_exposes_sample_interface():
    model = as_diffusers_output(_make())
    assert model.config.in_channels == 3
    x, t = torch.randn(1, 3, 32, 32), torch.tensor(10)
    with torch.no_grad():
        out = model(x, t)
    assert out.sample.shape == (1, 3, 32, 32)
    assert model.to("cpu") is model
```

```bash
uv run pytest tests/test_ablate.py -v   # 预期 FAIL（模块不存在）
```

- [ ] **Step 2: 实现**

`src/catdiff/model/ablate.py` 完整内容：

```python
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
```

`src/catdiff/model/unet.py` 末尾追加（顶部 `import types`）：

```python
def as_diffusers_output(model: UNet2D, in_channels: int = 3):
    """把手写模型（返回裸 tensor）适配成 baseline.sample_batch 期望的接口。"""

    class _Adapter:
        def __init__(self, m):
            self._m = m
            self.config = types.SimpleNamespace(in_channels=in_channels)

        def __call__(self, x, t):
            return types.SimpleNamespace(sample=self._m(x, t))

        def to(self, device):
            return self

    return _Adapter(model)
```

`tests/test_integration_slow.py`：删除本地 `_AsDiffusersOutput` 类，`test_cat_ddim20_256_final_image` 中改用：

```python
from catdiff.model.unet import as_diffusers_output
# ...
img_ours = sample_batch(as_diffusers_output(ours), our_sched,
                        num_samples=1, image_size=256, seed=100)
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest -q                       # 默认套件全绿（含 ablate 4 个新测试）
uv run pytest -m slow -q               # 确认 slow 重构后仍全绿
git add src/catdiff/model tests
git commit -m "feat: 注意力消融工具 strip_attention + 采样适配器（M0-F3-T1）"
```

---

## Task 2: CLI 支持手写后端与消融开关

**Files:**
- Modify: `src/catdiff/baseline/cli.py`
- Test: `tests/test_cli.py`（追加）

**Interfaces:**
- Consumes: Task 1 的 `load_handwritten_unet`/`as_diffusers_output`/`strip_attention`、F2 的手写 `DDIMScheduler`
- Produces: CLI 新参数 `--backend {diffusers,handwritten}`（默认 diffusers，行为不变）、`--config <path>`（handwritten 必填）、`--strip-attention {none,all,keep-mid}`（默认 none，仅 handwritten 合法）；`metadata.json` 新增 `backend` 与 `strip_attention` 字段

- [ ] **Step 1: 写失败测试（追加到 `tests/test_cli.py`）**

```python
def test_cli_handwritten_backend(tmp_path, monkeypatch, tiny_unet, tiny_scheduler):
    import json as _json

    cfg = tmp_path / "cfg.json"
    cfg.write_text(_json.dumps({"stub": True}))
    monkeypatch.setattr(cli, "load_handwritten_unet",
                        lambda model_id, config: tiny_unet)
    monkeypatch.setattr(cli, "HwDDIMScheduler",
                        lambda num_train_timesteps=1000: tiny_scheduler(2))
    out = tmp_path / "run_hw"
    rc = cli.main([
        "--model-id", "tiny", "--backend", "handwritten",
        "--config", str(cfg), "--strip-attention", "all",
        "--num-samples", "2", "--image-size", "32",
        "--num-inference-steps", "2", "--seed", "5", "--out-dir", str(out),
    ])
    assert rc == 0
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "handwritten"
    assert meta["strip_attention"] == "all"


def test_cli_strip_attention_requires_handwritten(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        cli.main([
            "--model-id", "tiny", "--strip-attention", "all",
            "--num-samples", "1", "--out-dir", str(tmp_path / "x"),
        ])
```

注：`tiny_unet` 是 diffusers 微型模型，monkeypatch 后 `as_diffusers_output` 包装它时 `.sample` 访问依然成立（diffusers 模型返回 `UNet2DOutput`——不行，裸 tensor 与 `.sample` 不兼容）。**因此 monkeypatch 的 `load_handwritten_unet` 应返回手写模型**：在 conftest 追加 fixture：

```python
@pytest.fixture
def tiny_handwritten_unet():
    from catdiff.model.unet import UNet2D

    return UNet2D(dict(
        in_channels=3, out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8,
    ))
```

测试里改为 `monkeypatch.setattr(cli, "load_handwritten_unet", lambda model_id, config: tiny_handwritten_unet)`，fixture 名同步替换。

```bash
uv run pytest tests/test_cli.py -v   # 预期 FAIL（参数不存在）
```

- [ ] **Step 2: 修改 cli.py**

在 `src/catdiff/baseline/cli.py` 中：

1. 顶部 import 追加：

```python
from catdiff.model.ddim import DDIMScheduler as HwDDIMScheduler
from catdiff.model.ablate import strip_attention
from catdiff.model.unet import as_diffusers_output, load_handwritten_unet
```

2. argparse 追加：

```python
    p.add_argument("--backend", choices=["diffusers", "handwritten"],
                   default="diffusers")
    p.add_argument("--config", type=Path, default=None,
                   help="handwritten 后端所需的 UNet config json")
    p.add_argument("--strip-attention", choices=["none", "all", "keep-mid"],
                   default="none", help="注意力消融（仅 handwritten 后端）")
```

3. `main()` 中模型/调度器加载段替换为：

```python
    if args.strip_attention != "none" and args.backend != "handwritten":
        p.error("--strip-attention 仅 handwritten 后端支持")
    if args.backend == "handwritten":
        if args.config is None:
            p.error("--backend handwritten 需要 --config")
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
        hw_model = load_handwritten_unet(args.model_id, cfg)
        if args.strip_attention != "none":
            n = strip_attention(hw_model, keep_mid=args.strip_attention == "keep-mid")
            print(f"注意力消融：{args.strip_attention}，替换 {n} 处")
        model = as_diffusers_output(hw_model)
        scheduler = HwDDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(args.num_inference_steps)
    else:
        model = load_unet(args.model_id)
        scheduler = load_ddim(args.model_id, args.num_inference_steps)
```

4. `metadata.json` 字典追加两个字段：

```python
                "backend": args.backend,
                "strip_attention": args.strip_attention,
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest -q   # 全绿（含新增 2 个 CLI 测试）
git add src/catdiff/baseline/cli.py tests/conftest.py tests/test_cli.py
git commit -m "feat: CLI 支持手写后端与注意力消融开关（M0-F3-T2）"
```

---

## Task 3: 消融采样批次（slow，夜间可跑）

**Files:**
- 产物：`artifacts/f3/full-ddim20/`、`artifacts/f3/noattn-ddim20/`（条件批次 `midonly-ddim20/`）

**Interfaces:**
- Consumes: Task 2 的 CLI、`docs/reference/unet-config-ddpm-cat-256.json`
- Produces: 三组（或两组）同 seed 对照样本，供 Task 4 验收

- [ ] **Step 1: A 组——完整手写模型（约 13 分钟）**

```bash
uv run python -m catdiff.baseline.cli \
  --model-id google/ddpm-cat-256 --backend handwritten \
  --config docs/reference/unet-config-ddpm-cat-256.json \
  --out-dir artifacts/f3/full-ddim20 \
  --num-samples 20 --image-size 256 --num-inference-steps 20 --seed 100
```

预期：grid 与 `artifacts/f1/cat256-ddim20/grid.png` 画质一致（F2 已证对齐）；如有出入说明手写管线被破坏，停止排查。

- [ ] **Step 2: B 组——去除全部注意力（约 13 分钟）**

```bash
uv run python -m catdiff.baseline.cli \
  --model-id google/ddpm-cat-256 --backend handwritten \
  --config docs/reference/unet-config-ddpm-cat-256.json \
  --strip-attention all \
  --out-dir artifacts/f3/noattn-ddim20 \
  --num-samples 20 --image-size 256 --num-inference-steps 20 --seed 100
```

- [ ] **Step 3: 初判与条件批次**

对比 A/B 两组 grid：
- B 组画质可接受（能认出猫、纹理退化温和）→ 不需要 C 组，直接进 Task 4，结论候选 **Go（RTL 不实现注意力）**；
- B 组明显崩坏 → 跑 C 组中间档（保留 mid 注意力，仅去 down/up 注意力）：

```bash
uv run python -m catdiff.baseline.cli \
  --model-id google/ddpm-cat-256 --backend handwritten \
  --config docs/reference/unet-config-ddpm-cat-256.json \
  --strip-attention keep-mid \
  --out-dir artifacts/f3/midonly-ddim20 \
  --num-samples 20 --image-size 256 --num-inference-steps 20 --seed 100
```

- [ ] **Step 4: 提交产物清单（产物本身不入库）**

```bash
git status --porcelain   # 确认 artifacts/ 未被跟踪
git commit --allow-empty -m "exp: F3 消融采样批次完成（full/noattn[/midonly] 各 20 张 @256 DDIM-20）（M0-F3-T3）"
```

---

## Task 4: 消融报告与人工签字（Go/No-go 停止点）

**Files:**
- Create: `docs/quality-gates/f3-attention-ablation-report.md`

- [ ] **Step 1: 写报告**

`docs/quality-gates/f3-attention-ablation-report.md` 模板（如实填写）：

```markdown
# F3 注意力消融报告（256×256）

| 项 | 值 |
|---|---|
| 日期 | <填写> |
| 模型 | google/ddpm-cat-256 手写复刻（F2 已对齐） |
| 采样 | 手写 DDIM eta=0, 20 步, seed=100, 20 张/组 |

## 对照组

| 组 | 注意力配置 | 路径 | 可辨猫占比 |
|---|---|---|---|
| A | 完整（6 处） | artifacts/f3/full-ddim20/grid.png | __/20 |
| B | 全部去除 | artifacts/f3/noattn-ddim20/grid.png | __/20 |
| C（条件） | 仅保留 mid | artifacts/f3/midonly-ddim20/grid.png | __/20（未跑则删行） |

## 结论（三选一，验收人签字）

- [ ] **Go**：B 组画质可接受 → RTL 不实现注意力（M2 省整块工作量，F4 起按无注意力网络走）
- [ ] **部分 Go**：B 崩坏但 C 可接受 → RTL 仅实现 mid 块注意力（1 处）
- [ ] **No-go**：C 也不可接受 → 注意力全部保留，注意力 RTL 进主线排期

验收人签字：____  日期：____
```

- [ ] **Step 2: 停止点**

报告生成后停下，由项目负责人对照 grid 人工验收签字。**结论影响后续所有 feature 的范围**：Go → F4/F5 按无注意力网络；部分 Go → 仅 mid 注意力；No-go → 全保留并通知 PM 更新 spec（注意力 RTL 从 M4 可选项升为主线）。

- [ ] **Step 3: 提交报告**

```bash
git add docs/quality-gates/f3-attention-ablation-report.md
git commit -m "docs: F3 注意力消融报告（结论已签字）（M0-F3-T4）"
```

---

## 分发提示词（交给执行 agent 用）

```
你在仓库 C:\git\fpga-cat-diffusion 工作。严格按 docs/superpowers/plans/2026-09-25-m0-f3-attention-ablation.md 逐任务执行（Task 1~4 内有完整代码与命令）。硬性要求：
- strip_attention 必须在 load_state_dict 之后调用；单元测试绝不调用 from_pretrained
- Task 3 的采样批次必须用 CLI 跑（可断点续跑），每组约 13 分钟；B 组崩坏才跑 C 组
- Task 4 报告生成后停止，由项目负责人签字，不要自行下结论、不要合入 main
开始前先读 docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md 的 §5.3。偏离计划需在提交信息中说明理由。
```
