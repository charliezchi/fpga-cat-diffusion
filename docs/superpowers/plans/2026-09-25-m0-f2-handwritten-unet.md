# M0-F2 实施计划：手写 UNet 前向 + 手写 DDIM 采样器

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 不调用 diffusers 的 `UNet2DModel.forward` 与 scheduler，全手写复刻 `ddpm-cat-256` 的 UNet 前向与 DDIM(η=0) 采样器，与 diffusers 参考逐层对齐，为定点化（F4）与 RTL（M2）提供唯一规格来源。

**Architecture:** 手写模块的 **state_dict 命名与 diffusers 完全镜像**（已由 PM 在真实权重上核实，见各任务"已核实事实"），因此 `load_state_dict` 严格加载即完成权重映射，无需映射表。单元测试用内存构造的微型 diffusers 模块作参考（不触网）；`slow` 标记的集成测试用 HF 缓存中的真实 cat-256/butterflies 权重做整网对齐。

**Tech Stack:** Python 3.11+、uv、torch（CPU）、diffusers（仅作参考实现与权重来源）、pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md`（§5.2 是本计划的验收依据；执行前必读）

## Global Constraints

- 仅 CPU；不得 import 任何 CUDA 路径（spec §2）
- 单元测试**不得调用 `from_pretrained`**（不触网）；真实权重集成测试统一打 `@pytest.mark.slow`，默认运行排除
- 所有测试固定 seed 可复现（spec §2）
- 对齐容差：逐层/逐模块相对误差 < 1e-4；最终采样图像逐像素最大绝对差 < 2e-3（spec §5.2）
- 手写模型的 forward 返回**裸 tensor**（不是 diffusers 的 `UNet2DOutput`），与 F3+ 的内部使用约定一致
- 分支 `feat/handwritten-unet`；`uv run pytest` 与 `uv run pytest -m slow` 全绿才可合入（spec §6）
- cat-256 配置的两个非默认值必须硬编码正确：`freq_shift=1`（即 diffusers `downscale_freq_shift=1`，恰好是默认值）、`flip_sin_to_cos=False`（**非默认**，默认是 True）——见 `docs/reference/unet-config-ddpm-cat-256.json`

## Review Focus

1. **时间嵌入非默认值**：`flip_sin_to_cos=False` 若按 diffusers 默认 True 实现，整个模型静默出错——由 Task 1 测试钉死（与 diffusers `get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=1)` 逐值对比）。
2. **Downsample 非对称 padding**：cat-256 `downsample_padding=0`，diffusers 内部做 `F.pad(x, (0,1,0,1))` 非对称填充使 64→32 整除；遗漏则整网形状错位——由 Task 4 测试钉死。
3. **注意力 heads=1**：cat-256 `attention_head_dim=null`，实为单头、dim_head=channels；按多头实现则权重形状对不上——由 Task 3 测试 + Task 7 严格 `load_state_dict` 钉死。
4. **skip 连接顺序**：上采样块按"追加序的逆序"消费 skip 栈（diffusers 从 tuple 末尾弹），顺序错则通道对不上或数值错——由 Task 5 逐层 trace 对比钉死。
5. **DDIM 边界步**：`prev_t < 0` 时须用 `final_alpha_cumprod`（`set_alpha_to_one=True` → 1.0）；timesteps 间距公式已验证为 `(arange(N) * (1000//N)).round().flip(0)`——由 Task 6 测试钉死。
6. **state_dict 命名漂移**：手写模块任一属性名与 diffusers 不同，Task 7 的严格 `load_state_dict` 立即报错——这本身是验收的一部分。

---

## Task 1: slow 标记 + 时间嵌入（`model/time_embed.py`）

**Files:**
- Modify: `pyproject.toml`（注册 slow 标记，默认排除）
- Create: `src/catdiff/model/time_embed.py`
- Test: `tests/test_time_embed.py`

**Interfaces:**
- Consumes: F1 工程骨架
- Produces:
  - `catdiff.model.time_embed.timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor`
  - `catdiff.model.time_embed.Timesteps(num_channels: int)`（无参数，`forward(timesteps)`）
  - `catdiff.model.time_embed.TimestepEmbedding(in_channels: int, time_embed_dim: int)`（属性名 `linear_1`/`act`/`linear_2`，与 diffusers 一致）

**已核实事实**（PM 在真实 cat-256 上验证）：`time_embedding` 为 `Linear(128→512) → SiLU → Linear(512→512)`；sinusoid 维度 = `block_out_channels[0]` = 128，`time_embed_dim` = 128×4 = 512；`flip_sin_to_cos=False`、`downscale_freq_shift=1`。

- [ ] **Step 1: 建分支，注册 slow 标记**

```bash
git checkout main && git checkout -b feat/handwritten-unet
```

`pyproject.toml` 的 `[tool.pytest.ini_options]` 改为：

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow'"
markers = [
    "slow: 需要 HF 缓存真实权重、分钟级耗时的集成测试（用 uv run pytest -m slow 运行）",
]
```

- [ ] **Step 2: 写失败测试**

`tests/test_time_embed.py`：

```python
import torch
from diffusers.models.embeddings import TimestepEmbedding as RefTimestepEmbedding
from diffusers.models.embeddings import get_timestep_embedding

from catdiff.model.time_embed import TimestepEmbedding, Timesteps, timestep_embedding

# cat-256 的非默认配置（docs/reference/unet-config-ddpm-cat-256.json）：
# flip_sin_to_cos=False（diffusers 默认 True），downscale_freq_shift=1（恰好是默认值）


def test_sinusoid_matches_diffusers():
    t = torch.tensor([0, 1, 7, 250, 999], dtype=torch.float32)
    ref = get_timestep_embedding(t, 128, flip_sin_to_cos=False, downscale_freq_shift=1)
    mine = timestep_embedding(t, 128)
    assert torch.allclose(mine, ref, atol=1e-6)


def test_timesteps_module_matches_diffusers():
    from diffusers.models.embeddings import Timesteps as RefTimesteps

    t = torch.tensor([3, 981], dtype=torch.float32)
    ref = RefTimesteps(num_channels=128, flip_sin_to_cos=False, downscale_freq_shift=1)
    assert torch.allclose(Timesteps(128)(t), ref(t), atol=1e-6)


def test_timestep_embedding_mlp_matches_diffusers():
    torch.manual_seed(0)
    ref = RefTimestepEmbedding(in_channels=128, time_embed_dim=512)
    mine = TimestepEmbedding(128, 512)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(4, 128)
    assert torch.allclose(mine(x), ref(x), atol=1e-6)
```

```bash
uv run pytest tests/test_time_embed.py -v
```

预期：FAIL（`ModuleNotFoundError: catdiff.model.time_embed`）。

- [ ] **Step 3: 实现 time_embed.py**

`src/catdiff/model/time_embed.py` 完整内容：

```python
"""时间步嵌入：复刻 diffusers Timesteps + TimestepEmbedding。

cat-256 配置（docs/reference/unet-config-ddpm-cat-256.json）：
flip_sin_to_cos=False、freq_shift=1（即 downscale_freq_shift=1）。
"""

import math

import torch
from torch import nn


def timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    """对应 diffusers get_timestep_embedding(flip_sin_to_cos=False, downscale_freq_shift=1)。"""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half - 1)  # downscale_freq_shift = 1
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timestep_embedding(timesteps, self.num_channels)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(x)))
```

- [ ] **Step 4: 跑测试并提交**

```bash
uv run pytest tests/test_time_embed.py -v   # 3 个 PASS
git add pyproject.toml src/catdiff/model tests/test_time_embed.py
git commit -m "feat: 手写时间嵌入（flip_sin_to_cos=False 对齐）（M0-F2-T1）"
```

---

## Task 2: 残差块（`model/resnet.py`）

**Files:**
- Create: `src/catdiff/model/resnet.py`
- Test: `tests/test_resnet.py`

**Interfaces:**
- Consumes: 无（独立模块）
- Produces: `catdiff.model.resnet.ResnetBlock2D(in_channels, out_channels, temb_channels, groups=32, eps=1e-6, dropout=0.0)`，`forward(x, temb) -> Tensor`；属性名 `norm1/conv1/time_emb_proj/norm2/dropout/conv2/nonlinearity/conv_shortcut` 与 diffusers 一致；`in==out` 时 `conv_shortcut is None`

**已核实事实**：diffusers `ResnetBlock2D` 为 pre-norm 结构、`output_scale_factor=1.0`（即直接 `shortcut + hidden`，无 1/√2）、`time_emb_proj` 输入前对 temb 过一次 SiLU。

- [ ] **Step 1: 写失败测试**

`tests/test_resnet.py`：

```python
import torch
from diffusers.models.resnet import ResnetBlock2D as RefResnet

from catdiff.model.resnet import ResnetBlock2D


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_resnet_same_channels_matches_diffusers():
    torch.manual_seed(0)
    ref = RefResnet(in_channels=32, out_channels=32, temb_channels=128, groups=8, eps=1e-6)
    mine = ResnetBlock2D(32, 32, 128, groups=8)
    mine.load_state_dict(ref.state_dict())
    x, temb = torch.randn(1, 32, 16, 16), torch.randn(1, 128)
    assert _rel(mine(x, temb), ref(x, temb)) < 1e-4


def test_resnet_channel_change_matches_diffusers():
    torch.manual_seed(0)
    ref = RefResnet(in_channels=32, out_channels=64, temb_channels=128, groups=8, eps=1e-6)
    mine = ResnetBlock2D(32, 64, 128, groups=8)
    mine.load_state_dict(ref.state_dict())  # 含 conv_shortcut 命名对齐
    assert mine.conv_shortcut is not None
    x, temb = torch.randn(1, 32, 16, 16), torch.randn(1, 128)
    assert _rel(mine(x, temb), ref(x, temb)) < 1e-4
```

```bash
uv run pytest tests/test_resnet.py -v   # 预期 FAIL（模块不存在）
```

- [ ] **Step 2: 实现 resnet.py**

`src/catdiff/model/resnet.py` 完整内容：

```python
"""复刻 diffusers ResnetBlock2D（pre-norm、silu、output_scale_factor=1.0）。"""

import torch
from torch import nn


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, temb_channels,
                 groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(self.nonlinearity(self.norm1(x)))
        temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        hidden = hidden + temb
        hidden = self.conv2(self.dropout(self.nonlinearity(self.norm2(hidden))))
        shortcut = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        return shortcut + hidden
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_resnet.py -v   # 2 个 PASS
git add src/catdiff/model/resnet.py tests/test_resnet.py
git commit -m "feat: 手写 ResnetBlock2D 与 diffusers 对齐（M0-F2-T2）"
```

---

## Task 3: 注意力块（`model/attention.py`）

**Files:**
- Create: `src/catdiff/model/attention.py`
- Test: `tests/test_attention.py`

**Interfaces:**
- Consumes: 无
- Produces: `catdiff.model.attention.AttentionBlock(channels, groups=32, eps=1e-6)`，`forward(x) -> Tensor`；属性名 `group_norm/to_q/to_k/to_v/to_out` 与 diffusers 一致；**单头**（heads=1，dim_head=channels）

**已核实事实**：cat-256 的注意力模块类名为 `Attention`，`heads=1`、`scale = channels**-0.5`、参考实现走 `F.scaled_dot_product_attention`；PM 实测手写 softmax 与 sdpa 的最大绝对差 2.4e-7，远小于容差。

- [ ] **Step 1: 写失败测试**

`tests/test_attention.py`：

```python
import torch
from diffusers.models.attention_processor import Attention as RefAttention

from catdiff.model.attention import AttentionBlock


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_attention_matches_diffusers():
    torch.manual_seed(0)
    ref = RefAttention(
        query_dim=64, heads=1, dim_head=64, norm_num_groups=8,
        eps=1e-6, bias=True, dropout=0.0, scale_qk=True,
    )
    mine = AttentionBlock(64, groups=8)
    mine.load_state_dict(ref.state_dict())  # group_norm/to_q/to_k/to_v/to_out 命名对齐
    x = torch.randn(1, 64, 8, 8)
    assert _rel(mine(x), ref(x)) < 1e-4
```

```bash
uv run pytest tests/test_attention.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现 attention.py**

`src/catdiff/model/attention.py` 完整内容：

```python
"""复刻 diffusers Attention（UNet2DModel 用法：group_norm + 单头 + to_out）。

cat-256 的 attention_head_dim=null，实际为 heads=1、dim_head=channels。
参考实现走 F.scaled_dot_product_attention；此处用手动 softmax，
单头 fp32 下数值差约 2e-7，远在 1e-4 容差内。
"""

import torch
from torch import nn


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.ModuleList([nn.Linear(channels, channels), nn.Dropout(0.0)])
        self.scale = channels ** -0.5  # heads=1, dim_head=channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.group_norm(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w).transpose(1, 2)  # (b, hw, c)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) * self.scale, dim=-1)
        out = attn @ v
        out = self.to_out[1](self.to_out[0](out))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out + residual
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_attention.py -v   # 1 个 PASS
git add src/catdiff/model/attention.py tests/test_attention.py
git commit -m "feat: 手写单头注意力块与 diffusers 对齐（M0-F2-T3）"
```

---

## Task 4: 上下采样与块容器（`model/blocks.py`）

**Files:**
- Create: `src/catdiff/model/blocks.py`
- Test: `tests/test_blocks.py`

**Interfaces:**
- Consumes: `ResnetBlock2D`（Task 2）、`AttentionBlock`（Task 3）
- Produces:
  - `Downsample2D(channels)`（属性 `conv`；forward 内 `F.pad(x, (0,1,0,1))` 非对称填充 + 3×3 stride2 conv）
  - `Upsample2D(channels)`（属性 `conv`；nearest ×2 + 3×3 pad1 conv）
  - `DownBlock(in_channels, out_channels, temb_channels, num_layers, groups, add_downsample, with_attention)`，`forward(x, temb) -> (x_out, skips: list[Tensor])`；属性 `resnets/attentions/downsamplers`（`attentions`/`downsamplers` 不存在时为 `None`）
  - `MidBlock(channels, temb_channels, groups)`，`forward(x, temb)`；resnet → attention → resnet
  - `UpBlock(in_channels, out_channels, prev_output_channel, temb_channels, num_layers, groups, add_upsample, with_attention)`，`forward(x, temb, skips: list[Tensor])`

**已核实事实**：diffusers `Downsample2D(padding=0)` 在 conv 前做 `F.pad(x, (0,1,0,1))`（左 0 右 1 非对称），保证偶数输入整除（64→32）；`Upsample2D(use_conv=True)` 先 nearest ×2 再过 3×3 pad1 conv。

- [ ] **Step 1: 写失败测试**

`tests/test_blocks.py`：

```python
import torch
from diffusers.models.downsampling import Downsample2D as RefDownsample
from diffusers.models.upsampling import Upsample2D as RefUpsample

from catdiff.model.blocks import Downsample2D, Upsample2D


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def test_downsample_matches_diffusers():
    torch.manual_seed(0)
    ref = RefDownsample(channels=64, use_conv=True, out_channels=64, padding=0)
    mine = Downsample2D(64)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(1, 64, 64, 64)
    out_mine, out_ref = mine(x), ref(x)
    assert out_mine.shape == (1, 64, 32, 32)  # 非对称 padding 保证整除
    assert _rel(out_mine, out_ref) < 1e-4


def test_upsample_matches_diffusers():
    torch.manual_seed(0)
    ref = RefUpsample(channels=64, use_conv=True, out_channels=64)
    mine = Upsample2D(64)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(1, 64, 16, 16)
    out_mine, out_ref = mine(x), ref(x)
    assert out_mine.shape == (1, 64, 32, 32)
    assert _rel(out_mine, out_ref) < 1e-4
```

```bash
uv run pytest tests/test_blocks.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现 blocks.py**

`src/catdiff/model/blocks.py` 完整内容：

```python
"""UNet 结构件：上下采样与 down/mid/up 块容器，命名与 diffusers 镜像。"""

import torch
import torch.nn.functional as F
from torch import nn

from .attention import AttentionBlock
from .resnet import ResnetBlock2D


class Downsample2D(nn.Module):
    """diffusers Downsample2D(use_conv=True, padding=0)：非对称 pad + 3x3 stride2。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1))  # diffusers 内部行为：左0右1，保证 64->32 整除
        return self.conv(x)


class Upsample2D(nn.Module):
    """diffusers Upsample2D(use_conv=True)：nearest x2 + 3x3 pad1。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class DownBlock(nn.Module):
    """DownBlock2D / AttnDownBlock2D 复刻。返回 (输出, skip 列表)。"""

    def __init__(self, in_channels, out_channels, temb_channels, num_layers,
                 groups, add_downsample, with_attention):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if with_attention else None
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(cin, out_channels, temb_channels, groups))
            if with_attention:
                self.attentions.append(AttentionBlock(out_channels, groups))
        self.downsamplers = (
            nn.ModuleList([Downsample2D(out_channels)]) if add_downsample else None
        )

    def forward(self, x, temb):
        skips = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x)
            skips.append(x)
        if self.downsamplers is not None:
            x = self.downsamplers[0](x)
            skips.append(x)
        return x, skips


class MidBlock(nn.Module):
    """UNetMidBlock2D(add_attention=True)：resnet -> attention -> resnet。"""

    def __init__(self, channels, temb_channels, groups):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResnetBlock2D(channels, channels, temb_channels, groups),
            ResnetBlock2D(channels, channels, temb_channels, groups),
        ])
        self.attentions = nn.ModuleList([AttentionBlock(channels, groups)])

    def forward(self, x, temb):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x)
        return self.resnets[1](x, temb)


class UpBlock(nn.Module):
    """UpBlock2D / AttnUpBlock2D 复刻（num_layers = layers_per_block + 1）。

    skips 按消费顺序给出（追加序的逆序），由调用方（UNet2D）准备。
    """

    def __init__(self, in_channels, out_channels, prev_output_channel,
                 temb_channels, num_layers, groups, add_upsample, with_attention):
        super().__init__()
        self.resnets = nn.ModuleList()
        self.attentions = nn.ModuleList() if with_attention else None
        for i in range(num_layers):
            skip_ch = in_channels if i == num_layers - 1 else out_channels
            cin = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(cin + skip_ch, out_channels, temb_channels, groups)
            )
            if with_attention:
                self.attentions.append(AttentionBlock(out_channels, groups))
        self.upsamplers = (
            nn.ModuleList([Upsample2D(out_channels)]) if add_upsample else None
        )

    def forward(self, x, temb, skips):
        for i, resnet in enumerate(self.resnets):
            x = torch.cat([x, skips[i]], dim=1)
            x = resnet(x, temb)
            if self.attentions is not None:
                x = self.attentions[i](x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x)
        return x
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_blocks.py -v   # 2 个 PASS
git add src/catdiff/model/blocks.py tests/test_blocks.py
git commit -m "feat: 手写上下采样与块容器（M0-F2-T4）"
```

---

## Task 5: UNet 组装与前向追踪（`model/unet.py`、`model/trace.py`）

**Files:**
- Create: `src/catdiff/model/unet.py`
- Create: `src/catdiff/model/trace.py`
- Test: `tests/test_unet.py`

**Interfaces:**
- Consumes: Task 1/2/3/4 全部模块
- Produces（F3/F4/F5 依赖）:
  - `catdiff.model.unet.UNet2D(config: dict)`，`forward(sample, timestep) -> Tensor`；`load_state_dict` 与 diffusers `UNet2DModel.state_dict()` 严格兼容
  - `catdiff.model.unet.load_handwritten_unet(model_id: str, config: dict) -> UNet2D`（内部调 diffusers 加载权重后 strict 拷贝；仅 slow 路径使用）
  - `catdiff.model.trace.forward_with_trace(model, sample, timestep, name_filter=None) -> (out: Tensor, trace: dict[str, Tensor])`；默认 filter 捕获 `conv_in/conv_out/mid_block` 及所有 `down_blocks.N.(resnets|attentions|downsamplers).M`、`up_blocks.N.(resnets|attentions|upsamplers).M`、`mid_block.(resnets|attentions).N`——对 diffusers 参考模型同样适用（命名一致），用于逐层定位

**已核实事实**：skip 栈以 `conv_in` 输出开头；diffusers 上采样块从 res_samples tuple **末尾**弹出消费（即追加序逆序）；up 块通道推算（已在真实模型 450 个 key 上验证）：`rchs = reversed(block_out_channels)`，块 i 的 `out=rchs[i]`、`in_skip=rchs[min(i+1, len-1)]`，块内第 j 个 resnet 输入 = `(prev_out if j==0 else out) + (in_skip if j==last else out)`。

- [ ] **Step 1: 写失败测试**

`tests/test_unet.py`：

```python
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
    ref = UNet2DModel(
        sample_size=32, in_channels=3, out_channels=3, layers_per_block=2,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        norm_num_groups=8, downsample_padding=0, attention_head_dim=None,
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
    assert _rel(out_ours, out_ref.sample) < 1e-4


def test_cross_resolution_64():
    ref, ours = _make_pair()
    torch.manual_seed(2)
    x, t = torch.randn(1, 3, 64, 64), torch.tensor(500)
    with torch.no_grad():
        assert _rel(ours(x, t), ref(x, t).sample) < 1e-4
```

```bash
uv run pytest tests/test_unet.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现 trace.py 与 unet.py**

`src/catdiff/model/trace.py` 完整内容：

```python
"""前向追踪：抓取关键模块的中间输出，用于逐层对齐调试（F2）、消融（F3）与黄金向量（F5）。

对手写模型与 diffusers 参考模型通用（模块命名一致）。
"""

import re

import torch

_DEFAULT_PATTERN = re.compile(
    r"^(conv_in|conv_out|mid_block|"
    r"(?:down_blocks|up_blocks)\.\d+\.(?:resnets|attentions|downsamplers|upsamplers)\.\d+|"
    r"mid_block\.(?:resnets|attentions)\.\d+)$"
)


def forward_with_trace(model, sample, timestep, name_filter=None):
    """返回 (模型输出, {模块名: 输出 tensor})。

    name_filter: callable(name: str) -> bool；默认捕获 conv_in/conv_out/mid_block
    与所有 resnet/attention/上下采样模块（不含其内部子层）。
    """
    if name_filter is None:
        name_filter = lambda name: bool(_DEFAULT_PATTERN.match(name))
    trace = {}
    hooks = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            if torch.is_tensor(out):
                trace[name] = out.detach()
        return hook

    for name, mod in model.named_modules():
        if name and name_filter(name):
            hooks.append(mod.register_forward_hook(make_hook(name)))
    try:
        out = model(sample, timestep)
    finally:
        for h in hooks:
            h.remove()
    if hasattr(out, "sample"):  # diffusers UNet2DOutput
        out = out.sample
    return out, trace
```

`src/catdiff/model/unet.py` 完整内容：

```python
"""手写复刻 diffusers UNet2DModel（ddpm-cat-256 配置族）。

state_dict 命名与 diffusers 完全镜像，load_state_dict 严格加载即完成权重映射。
"""

import torch
from torch import nn

from .blocks import DownBlock, MidBlock, UpBlock
from .time_embed import TimestepEmbedding, Timesteps


class UNet2D(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        chs = config["block_out_channels"]
        layers = config["layers_per_block"]
        groups = config["norm_num_groups"]
        time_embed_dim = chs[0] * 4

        self.conv_in = nn.Conv2d(config["in_channels"], chs[0], 3, padding=1)
        self.time_proj = Timesteps(chs[0])
        self.time_embedding = TimestepEmbedding(chs[0], time_embed_dim)

        self.down_blocks = nn.ModuleList()
        cin = chs[0]
        for i, block_type in enumerate(config["down_block_types"]):
            cout = chs[i]
            self.down_blocks.append(DownBlock(
                cin, cout, time_embed_dim, layers, groups,
                add_downsample=i < len(chs) - 1,
                with_attention="Attn" in block_type,
            ))
            cin = cout

        self.mid_block = MidBlock(chs[-1], time_embed_dim, groups)

        self.up_blocks = nn.ModuleList()
        rchs = list(reversed(chs))
        prev_out = rchs[0]
        for i, block_type in enumerate(config["up_block_types"]):
            cout = rchs[i]
            cin_skip = rchs[min(i + 1, len(rchs) - 1)]
            self.up_blocks.append(UpBlock(
                cin_skip, cout, prev_out, time_embed_dim, layers + 1, groups,
                add_upsample=i < len(chs) - 1,
                with_attention="Attn" in block_type,
            ))
            prev_out = cout

        self.conv_norm_out = nn.GroupNorm(groups, chs[0], eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(chs[0], config["out_channels"], 3, padding=1)

    def forward(self, sample: torch.Tensor, timestep) -> torch.Tensor:
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        timesteps = timestep.reshape(-1).expand(sample.shape[0]).to(sample.device)
        temb = self.time_embedding(self.time_proj(timesteps))

        x = self.conv_in(sample)
        skips = [x]
        for block in self.down_blocks:
            x, block_skips = block(x, temb)
            skips.extend(block_skips)
        x = self.mid_block(x, temb)
        for block in self.up_blocks:
            take = skips[-len(block.resnets):]
            del skips[-len(block.resnets):]
            x = block(x, temb, list(reversed(take)))
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


def load_handwritten_unet(model_id: str, config: dict) -> UNet2D:
    """加载 diffusers 权重到手写模型（slow 路径：需要 HF 缓存）。"""
    from diffusers import UNet2DModel

    ref = UNet2DModel.from_pretrained(model_id)
    model = UNet2D(config)
    model.load_state_dict(ref.state_dict())  # strict：命名漂移在此报错
    model.eval()
    return model
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_unet.py -v   # 3 个 PASS
git add src/catdiff/model/unet.py src/catdiff/model/trace.py tests/test_unet.py
git commit -m "feat: 手写 UNet2D 组装与前向追踪，逐层对齐 diffusers（M0-F2-T5）"
```

---

## Task 6: 手写 DDIM 调度器（`model/ddim.py`）

**Files:**
- Create: `src/catdiff/model/ddim.py`
- Test: `tests/test_ddim.py`

**Interfaces:**
- Consumes: 无
- Produces: `catdiff.model.ddim.DDIMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02, beta_schedule="linear", set_alpha_to_one=True)`；`.set_timesteps(n)`、`.timesteps`、`.step(model_output, timestep, sample, eta=0.0) -> SimpleNamespace(prev_sample=...)`——与 `catdiff.baseline.sampling.sample_batch` 的 scheduler 用法鸭子类型兼容，可直接替换 diffusers scheduler

**已核实事实**：PM 已验证 timesteps 公式 `(arange(N) * (1000//N)).round().flip(0)` 与 diffusers 0.40 完全一致（N=20、50 均逐元素相等）；η=0 时方差项为零；`clip_sample` 仅在 `use_clipped_model_output=True` 时生效（本项目不用，无需实现）。

- [ ] **Step 1: 写失败测试**

`tests/test_ddim.py`：

```python
import torch
from diffusers import DDIMScheduler as RefDDIM

from catdiff.model.ddim import DDIMScheduler


def test_timesteps_match_diffusers():
    for n in (20, 50):
        ref = RefDDIM(num_train_timesteps=1000)
        ref.set_timesteps(n)
        mine = DDIMScheduler(num_train_timesteps=1000)
        mine.set_timesteps(n)
        assert torch.equal(mine.timesteps, ref.timesteps.cpu())


def test_full_trajectory_matches_diffusers():
    torch.manual_seed(0)
    sample = torch.randn(1, 3, 8, 8)
    ref = RefDDIM(num_train_timesteps=1000)
    ref.set_timesteps(20)
    mine = DDIMScheduler(num_train_timesteps=1000)
    mine.set_timesteps(20)
    x_ref, x_mine = sample.clone(), sample.clone()
    for t in ref.timesteps:  # 覆盖 prev_t<0 的末步（final_alpha_cumprod 路径）
        noise = torch.randn_like(x_ref)
        x_ref = ref.step(noise, t, x_ref, eta=0.0).prev_sample
        x_mine = mine.step(noise, t, x_mine, eta=0.0).prev_sample
        assert torch.allclose(x_mine, x_ref, rtol=1e-5, atol=1e-7), f"t={t} 不对齐"
```

```bash
uv run pytest tests/test_ddim.py -v   # 预期 FAIL
```

- [ ] **Step 2: 实现 ddim.py**

`src/catdiff/model/ddim.py` 完整内容：

```python
"""手写 DDIM(eta=0) 调度器，复刻 diffusers DDIMScheduler 的数值行为。

仅实现本项目所需子集：epsilon 预测、linear beta、eta=0、
use_clipped_model_output=False（clip_sample 不生效）。
"""

import types

import torch


class DDIMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 beta_schedule="linear", set_alpha_to_one=True,
                 prediction_type="epsilon"):
        assert beta_schedule == "linear", "本项目仅使用 linear beta"
        assert prediction_type == "epsilon", "本项目仅使用 epsilon 预测"
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps,
                               dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = None
        self._step_ratio = None

    def set_timesteps(self, num_inference_steps: int):
        self._step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = (
            (torch.arange(0, num_inference_steps) * self._step_ratio)
            .round().flip(0).long()
        )

    def step(self, model_output, timestep, sample, eta: float = 0.0):
        assert eta == 0.0, "本项目仅支持确定性采样 eta=0"
        t = int(timestep)
        prev_t = t - self._step_ratio
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )
        pred_x0 = (sample - (1 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()
        prev_sample = (
            alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * model_output
        )
        return types.SimpleNamespace(prev_sample=prev_sample)
```

- [ ] **Step 3: 跑测试并提交**

```bash
uv run pytest tests/test_ddim.py -v   # 2 个 PASS
git add src/catdiff/model/ddim.py tests/test_ddim.py
git commit -m "feat: 手写 DDIM 调度器与 diffusers 全轨迹对齐（M0-F2-T6）"
```

---

## Task 7: 真实权重集成对齐（slow）

**Files:**
- Test: `tests/test_integration_slow.py`

**Interfaces:**
- Consumes: `load_handwritten_unet`（Task 5）、手写 `DDIMScheduler`（Task 6）、`catdiff.baseline.sampling.sample_batch`（F1）、`docs/reference/unet-config-ddpm-cat-256.json`、`docs/reference/unet-config-butterflies-64.json`、HF 缓存中的真实权重
- Produces: spec §5.2 验收证据（256×256 完整前向对齐 + 256×256 DDIM-3 对齐 + 最终图像 < 2e-3）；对比图写入 `artifacts/f2/`（不入库）

**config 加载注意**：`docs/reference/unet-config-ddpm-cat-256.json` 中含 `sample_size`、`freq_shift` 等冗余键；`UNet2D(config)` 只读取 `in_channels/out_channels/down_block_types/up_block_types/block_out_channels/layers_per_block/norm_num_groups`，直接传入完整 dict 即可（多余键被忽略）。

- [ ] **Step 1: 写集成测试**

`tests/test_integration_slow.py` 完整内容：

```python
"""真实权重整网对齐（需要 HF 缓存，分钟级耗时）。

运行：uv run pytest -m slow -v
"""

import json
import types
from pathlib import Path

import pytest
import torch

from catdiff.baseline.sampling import sample_batch, tensor_to_pil
from catdiff.model.ddim import DDIMScheduler
from catdiff.model.trace import forward_with_trace
from catdiff.model.unet import UNet2D, load_handwritten_unet

pytestmark = pytest.mark.slow

REF = Path("docs/reference")
CAT_CONFIG = json.loads((REF / "unet-config-ddpm-cat-256.json").read_text())
BFLY_CONFIG = json.loads((REF / "unet-config-butterflies-64.json").read_text())


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


class _AsDiffusersOutput:
    """把手写模型（返回裸 tensor）适配成 sample_batch 期望的 .sample 接口。"""

    def __init__(self, model):
        self._m = model
        self.config = types.SimpleNamespace(in_channels=3)

    def __call__(self, x, t):
        return types.SimpleNamespace(sample=self._m(x, t))

    def to(self, device):
        return self


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
    img_ours = sample_batch(_AsDiffusersOutput(ours), our_sched,
                            num_samples=1, image_size=256, seed=100)
    max_abs = (img_ours - img_ref).abs().max().item()
    out_dir = Path("artifacts/f2")
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(img_ref[0]).save(out_dir / "ddim20_ref.png")
    tensor_to_pil(img_ours[0]).save(out_dir / "ddim20_handwritten.png")
    assert max_abs < 2e-3
```

- [ ] **Step 2: 运行集成测试（约 3~5 分钟）**

```bash
uv run pytest -m slow -v
```

预期：4 个 PASS。若 `test_cat_forward_*` 失败，用 `forward_with_trace` 对比 diffusers 参考定位首个发散层（命名一致，逐层 diff）。

- [ ] **Step 3: 全量回归并提交**

```bash
uv run pytest -q        # 默认套件全绿（不含 slow）
uv run pytest -m slow -q  # slow 全绿
git add tests/test_integration_slow.py
git commit -m "test: 真实权重整网对齐（cat-256 前向@64/256、DDIM-3@256、DDIM-20 最终图像<2e-3）（M0-F2-T7）"
```

- [ ] **Step 4: 人工确认（停止点）**

查看 `artifacts/f2/ddim20_ref.png` 与 `artifacts/f2/ddim20_handwritten.png`：人眼应无可见差异。截图附在 F2 分支的审查记录中，由项目负责人确认后合入。

---

## 分发提示词（交给执行 agent 用）

```
你在仓库 C:\git\fpga-cat-diffusion 工作。严格按 docs/superpowers/plans/2026-09-25-m0-f2-handwritten-unet.md 逐任务执行（Task 1~7 内有完整代码与命令；plan 中"已核实事实"是 PM 在真实权重上验证过的，直接采用不要重新发明）。硬性要求：
- 单元测试绝不调用 from_pretrained（Task 1~6）；真实权重只在 Task 7 的 slow 测试中使用
- state_dict 命名必须与 diffusers 镜像（plan 代码已做到），load_state_dict 严格加载失败即视为实现错误，不得用 strict=False 绕过
- 完成后 uv run pytest 与 uv run pytest -m slow 全绿，不要合入 main
开始前先读 docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md 的 §5.2，以及 docs/reference/unet-config-ddpm-cat-256.json。偏离计划需在提交信息中说明理由。
```
