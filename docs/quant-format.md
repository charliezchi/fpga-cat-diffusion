# 量化与导出格式契约 v3（M0-F4 结案形态，M2 输入）

> 状态：F4 画质关卡 v3 已验收签字 Go（2026-09-25）。方案依据与完整诊断链见
> `docs/quality-gates/f4-int8-quality-report.md` 的 v3 节；本文档只固化契约本身。
> 本版取代 v1/v2（纯 INT8 DDIM-20，被 F4 验收否决）。

## 1. 量化方案（v3 混合精度）

### 1.1 位宽清单

| 对象 | 位宽 | 规则 |
|---|---|---|
| 内部权重层（152 个 conv/linear） | INT8 | per-(输出)channel 对称，axis=0 |
| conv_in / conv_out 权重 | INT16 | per-(输出)channel 对称，axis=0 |
| 内部激活量化点（50 处） | INT8 | per-tensor 对称，按"层 × 步组" |
| eps（conv_out 输出激活） | INT16 | per-tensor 对称，按步组 |
| conv_in 输出激活 | INT8 | 内部特征图，不做例外 |

INT16 层的理由（一句话）：端到端权重损伤几乎全部来自首尾两层（3 通道、各 3456
参数，豁免即 39.4 dB 近乎无损），二者 mult18 原生、算力占比可忽略；eps 是 3 通道
边界张量，带宽代价可忽略。诊断链见 F4 报告 v3 节。

### 1.2 权重

`w_q[c] = clamp(round(w[c] / s_w[c]), -qmax-1, qmax)`，`s_w[c] = max|w[c]| / qmax`
（clamp_min 1e-12）。Conv2d 沿 dim 0，Linear 沿 dim 0。
INT8：qmax=127；INT16：qmax=32767。

### 1.3 激活

标定 scale 以 **INT8 量程**为基准：`s8 = P99.95(|a|) / 127`。
- INT8 点：`a_q = clamp(round(a / s8), -128, 127)`；
- INT16 点（conv_out 输出）：`s16 = s8 × 127/32767`，
  `a_q = clamp(round(a / s16), -32768, 32767)`（同一物理量程重标到 INT16 网格）。

### 1.4 步组

双档，均 10 组，组号按推理步序号（非 timestep 值）：
`g = min(step_idx // (steps / groups), groups - 1)`。
- 主档 **DDIM-50**（eta=0）：50 步分 10 组，每组 5 步；
- 预览档 **DDIM-20**（eta=0）：20 步分 10 组，每组 2 步。
两档的表均导出，运行时切换。DDIM-100 已严格复测确认退步，不再尝试（F4 报告）。

### 1.5 量化点与累加

- 量化点：每个"层描述符算子"的输出（conv/linear/attention-out/GN/SiLU/
  residual-add/concat/上采样/下采样/最终 conv_out），共 51 处（与标定统计键一致）；
- 累加与重整：INT8×INT8 乘、INT32 累加；requant `y = round(acc × s_in × s_w[c] / s_out)`，
  截断饱和到 [-qmax-1, qmax]。INT16 层（conv_in/conv_out 权重、eps）的精确整数
  行为（乘法宽度与重整）与 GN/SiLU 整数实现同属 F5（位真模拟器）定义。

## 2. 标定策略（顺序标定）

- 标定集：seed 200..207 共 8 个初始噪声 × DDIM 全程，256×256；
- **顺序标定**：权重先按 §1.2 量化就位，再在**真实量化轨迹**上采集激活统计
  （而非 fp32 轨迹；+0.5 dB，零硬件成本）；
- 采集协议（金标口径）：每（量化点， 推理步）确定性地子采样 2048 个 |a| 样本，
  全部 seed 采完后按 §1.4 步组归并，取归并样本的 P99.95——一份数据可离线重分成
  任意步组数；
- 产物：`artifacts/f4/calib-stats-ddim50-seq.json`（主档）、
  `artifacts/f4/calib-stats-ddim20-seq.json`（预览档）；内容哈希写入对应
  act_scales 导出文件头（`calib_hash`，SHA-256 前 16 位）。

## 3. 集中配置

- `configs/quant_int8.json`：v3 规范配置（主档 DDIM-50）；
- `configs/quant_int8_ddim20.json`：预览档（仅步数 20，其余同 v3）；
- `mixed_precision` 段按**模块名首段前缀**匹配（`name.split(".")[0]`）：
  `{"int16_weight_layers": ["conv_in", "conv_out"], "int16_act_layers": ["conv_out"]}`。

## 4. 导出文件（v3 双档布局）

v3 导出 bundle 位于 `artifacts/f4/export-v3/`（大文件不入库，按需走 Release；
仓库根 `export/` 是 F4-T6 时代的 v1 单档导出，仅作历史参照）：

```
artifacts/f4/export-v3/
├── weights.bin          # CDW2，与步数无关，双档共用一份
├── layers.json          # 有序层描述符（含 bits 字段）
├── ddim_table_50.json   # 主档 DDIM 常数表
├── ddim_table_20.json   # 预览档 DDIM 常数表
├── film_table_50.bin    # [50][resnet_idx][out_channels] fp32
├── film_table_20.bin    # [20][resnet_idx][out_channels] fp32
├── film_index_50.json   # {timesteps, resnet_names}
├── film_index_20.json
├── act_scales_50.json   # 激活 scale 表（主档统计）
└── act_scales_20.json   # 激活 scale 表（预览档统计）
```

### 4.1 weights.bin（magic "CDW2"）

头部：`"CDW2"` + `u32` 层数；每记录（层序 = 层描述符顺序）：

```
u16 name_len; u8[name_len] name(utf-8)
u32 ndim; u32[ndim] shape
u8  bits            # 8 或 16
u8  has_bias
f32[shape[0]] scales（per-channel，小端）
[f32[shape[0]] bias（has_bias=1 时）]
payload             # bits=8: i8[numel]；bits=16: i16[numel]，均小端
```

### 4.2 layers.json

`{"calib_hash": …, "layers": [{index, name, op(conv2d|linear), shape, bits,
scale_ref: "weights.bin#<index>"}]}`。`calib_hash` 为主档（50 步）统计哈希。

### 4.3 ddim_table_<steps>.json

逐步 `{t, alpha_t, alpha_prev, sqrt_alpha_t, sqrt_one_minus_alpha_t,
sqrt_alpha_prev, sqrt_one_minus_alpha_prev}`（fp64 计算 fp32 存储）。

### 4.4 film_table_<steps>.bin / film_index_<steps>.json

对每档全部离散 timestep 预计算每个 ResnetBlock2D 的 time_emb_proj(silu(temb))
偏置向量，布局 `[steps][resnet_idx][out_channels]` fp32（头部 `u32 steps` +
`u32 resnet 数`），resnet_idx 按层描述符顺序；运行时按推理步查表，硬件无需时间
嵌入 MLP。

### 4.5 act_scales_<steps>.json

`{"calib_hash", "num_inference_steps", "num_step_groups", "scales":
{层名: [按组号升序的 scale 列表]}}`。scale 为 §1.3 的 INT8 基准标定值
`s8`；INT16 点（conv_out 输出）运行时换算 `s16 = s8 × 127/32767`。

### 4.6 生成命令

```bash
# 主档 DDIM-50：权重 + 全部 50 步表
uv run python src/catdiff/export/run_export.py \
  --quant-config configs/quant_int8.json \
  --calib-stats artifacts/f4/calib-stats-ddim50-seq.json \
  --out-dir artifacts/f4/export-v3 --table-suffix 50
# 预览档 DDIM-20：补导 20 步表（权重不动）
uv run python src/catdiff/export/run_export.py \
  --quant-config configs/quant_int8_ddim20.json \
  --calib-stats artifacts/f4/calib-stats-ddim20-seq.json \
  --out-dir artifacts/f4/export-v3 --table-suffix 20 --skip-weights
```

## 5. 版本历史

- **v3**（2026-09-25）：混合精度（convio W16 + eps A16）、顺序标定、DDIM-50/20
  双档、P99.95、10 步组。依据：F4 报告 v3 节（验收 Go）。
- v1/v2：纯 W8A8 DDIM-20（P99.9 5 组 / P99.95 10 组），F4 验收拒签，方案废弃。
