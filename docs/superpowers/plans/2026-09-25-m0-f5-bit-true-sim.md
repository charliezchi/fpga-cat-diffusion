# M0-F5 实施计划：位真定点模拟器与黄金向量

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用纯整数/定点运算在 PC 上逐位（bit-true）模拟未来 RTL 数据通路的全部行为，覆盖 v3 混合精度方案（内部 W8A8 + conv_in/out W16 + eps A16，DDIM-50/20 双档），产出：(1) 位真契约文档（M2 RTL 的唯一真理来源）；(2) 位真模拟器 `src/catdiff/bittrue/`；(3) 黄金向量包（逐层 IO + 端到端 IO）；(4) 位真画质复验报告（用户签字后才算 M0 关闭）。

**Architecture:** 先冻结位真契约（每个算子的整数行为：Q 格式、舍入、饱和、LUT 内容生成规则），再实现整数原语库 → 位真 UNet 前向 → 位真 DDIM 采样 → 画质复验 → 黄金向量导出。输入是 F4 已签字的 v3 导出包 `artifacts/f4/export-v3/`（weights.bin CDW2 + layers.json + 双档 DDIM/FiLM/scale 表），**位真模拟器只消费导出包，不直接读 PyTorch 权重**——这样同时验证了导出包本身的完备性。

**Tech Stack:** Python 3.14、uv、torch（CPU，仅用于张量容器/IO，整数运算用 int64 张量模拟）、numpy、pytest。

**Spec:** `docs/superpowers/specs/2026-09-25-m0-pc-algorithm-pipeline-design.md` v1.3（§5.4/§5.5）；量化契约 `docs/quant-format.md`（v3）；F4 签字报告 `docs/quality-gates/f4-int8-quality-report.md`（v3 节）。执行前必读这三份。

## Global Constraints

- 仅 CPU；单元测试用合成小数据；真实导出包走 slow 测试或 CLI 步骤（spec §2）
- **分支 `feat/bittrue-sim`**；`uv run pytest` 全绿才可申请合入（spec §6）
- 位真模拟器与 F4 fake-quant 的关系：fake-quant 是"float 近似整数行为"（GN/SiLU/softmax 内部 float）；位真模拟器把这些也整数化。两者最终图像允许有差异，但必须是"LUT/定点化引入的小扰动"级别——机器标准：同 seed 最终图像 PSNR ≥ 30 dB（预期远高于此），且画质复验由用户签字
- **禁止改动** `src/catdiff/model/` 与 `src/catdiff/quant/`（已签字冻结）；位真代码全部放新包 `src/catdiff/bittrue/`
- 所有采样固定 seed=100、断点续跑
- 数值定义权在契约文档：代码与文档不一致时改代码

## Review Focus

1. **requant 舍入方向**：acc(int32) × m(int32 Q0.31) >> n 的舍入必须是 round-half-up（加 `1<<(n-1)` 再算术右移），负数的 floor 偏移要在测试里钉死——RTL 里最容易写错的地方。
2. **GroupNorm 是唯一的翻车风险点**：均值/方差用 int 累加无争议；rsqrt 的 LUT 深度 + Newton 迭代次数必须带误差预算推导（Task 3），不达标就走降级方案（fp32 IP 单点，IEEE754 本身确定可位真）。不允许"差不多就行"的 GN。
3. **softmax 输出用 uint8 [0,255]**（概率非负，白捡一位精度）：乘 V 矩阵时按无符号×有符号处理，累加偏移要在契约里写清。
4. **residual add 的 scale 对齐**：两个输入激活各有各的 scale，必须先各自 requant 到输出量化点 scale 再饱和相加——不能直接 int8 相加。
5. **FiLM 偏置注入域**：偏置预量化到 **acc 域（int32）** 作为累加器初值（等价于 F2 的 hidden+temb 注入点），不是加在 INT8 输出上——由 Task 4 的对齐测试钉死。
6. **INT16 层路径**：conv_in 输入 x 是 INT16（Q 格式由契约定义），conv_in 权重 INT16 → int32 累加（18×18 DSP 语义）；conv_out 输出 eps 保持 INT16 出给 DDIM 更新单元，**不**经过 INT8 requant。
7. **DDIM 更新单元**：x0 = (x - β_t·eps)/α_t 的定点化、clip [-1,1] 的饱和位置、最终 [-1,1]→[0,255] 像素映射的舍入——全部逐位定义，这是演示画质的最后一公里。

## Task 1: 位真契约文档 `docs/bittrue-spec.md` + 配置

**Files:**
- Create: `docs/bittrue-spec.md`
- Create: `configs/bittrue.json`

**Interfaces:**
- Consumes: `docs/quant-format.md`（v3）、`artifacts/f4/export-v3/`（layers.json 的层清单与 bits 标记、各表格式）
- Produces: F5 全部后续任务与 M2 RTL 的共同契约

- [ ] **Step 1: 写契约**，必须包含以下各节，数值均为规定值：
  1. **数据格式总表**：内部激活 INT8 对称 per-tensor（scale 来自 act_scales 表）；convio 权重 INT16 per-channel；x/eps 边界张量 INT16 对称 per-tensor（scale 单独定义：x 用 [-1,1]→Q1.14 增益域，即 s_x = 1/16384……注意：x 的实际范围由 DDIM 轨迹决定，契约里写"以 act_scales 表中 conv_in 输入对应语义为准"还是定死 Q 格式，二选一并论证——推荐定死：x ∈ [-4, 4) Q3.12 存储，依据是 DDIM 全程 |x| 实测最大值，用 artifacts/f4/calib-samples-seq50.json 里 conv_in 对应统计佐证）；acc INT32；softmax 概率 UINT8。
  2. **conv/linear 整数流**：acc[c] = b_film[bias] + Σ x_q·w_q（int8×int8 / int16×int16 → int32 累加）；requant y_q = sat_int8/int16( (acc·M[c] + 2^(N[c]-1)) >> N[c] )，M = round(s_in·s_w[c]/s_out · 2^31) Q0.31，N[c] 为逐通道右移位数（由 PC 端从 scale 三元组离线计算，随导出包下发——需在 quant-format.md 增补 M/N 表或说明由 M2 现场计算）。
  3. **GN 整数流**：int8 输入 → int32 累加求均值（H·W 维），方差同理（int64 累加平方或减均值后 int32）；rsqrt(var+eps) 用 LUT（地址位数、表项 Q 格式）+ 可选 1 次 Newton；γ/β 融合进输出 requant 的 M/N。eps = 1e-5（与 PyTorch GroupNorm 默认一致，F2 已按此对齐）。
  4. **SiLU LUT**：256 项，地址 = int8 输入值+128，表项 = sat_int8(round(silu((addr-128)·s_in)/s_out))；表由 PC 脚本生成并随黄金向量归档。
  5. **softmax 整数流**：QK^T 累加 int32 → 逐行减 max（int32 域）→ 缩放（×s_qk·√d⁻¹ 折进 exp LUT 输入映射）→ exp LUT（定义地址位宽与 Q 格式）→ int32 累加求和 → 逐元素倒数（LUT 或移位近似，定义之）→ uint8 概率。
  6. **residual add / concat / upsample(nearest) / downsample(非对称 pad 0,1,0,1)** 的整数规则。
  7. **DDIM 更新单元**：系数表 Q 格式（推荐 Q2.30）、x0 预测式、clip、像素映射 round((x+1)·127.5) 的定点实现。
  8. **黄金向量格式**：目录布局、文件命名（`<step>/<layer>.in.bin` / `.out.bin`，int8/int16/int32 小端 + 形状 json）、端到端 IO（初始噪声 int16、最终 eps/像素）。
- [ ] **Step 2: 写 `configs/bittrue.json`**：导出包路径、档位（50/20）、黄金向量步选择（50 档取 step 0/25/49，20 档取 step 0/10/19）、LUT 参数集中配置。
- [ ] **Step 3: 自查**——契约中每个公式都能由导出包现有数据离线算出（若缺 M/N 表，决定是否扩导出器：允许小改 `src/catdiff/export/` 追加 `requant_params_<steps>.bin`，这不属于"冻结的 quant 包"）。

## Task 2: 整数原语库 `src/catdiff/bittrue/primitives.py`

**Files:**
- Create: `src/catdiff/bittrue/__init__.py`、`src/catdiff/bittrue/primitives.py`
- Create: `tests/test_bittrue_primitives.py`

**Interfaces:**
- Produces: `int_conv2d(x_q, w_q, M, N, bias32, stride, pad) -> int32→requant`、`int_linear`、`requant(acc, M, N, bits) -> sat`、`silu_lut(table, x_q)`、`groupnorm_int(x_q, gamma_MN, beta32, rsqrt_lut)`、`softmax_uint8(scores32)`、`ddim_update(x16, eps16, coeffs_q2_30) -> x16`

- [ ] **Step 1: 失败测试先行**——边界用例必须含：requant 负值舍入（acc=-3 类例）、饱和 ±128/±32768、累加器接近 int32 极限的大通道和、softmax 全零行（减 max 后）、GN 单元素通道。
- [ ] **Step 2: 实现**——全部用 torch int64/int32 张量模拟位宽（`torch.clamp` 实现饱和，注释标出 RTL 对应物）；禁止 float 出现在整数路径（scale→M/N 的离线换算除外，单独函数 `scales_to_MN(s_in, s_w, s_out) -> (int32, uint8)` 并附 float 对照测试：|requant 结果 - float 版| ≤ 1 LSB 的比例 > 99.9%）。
- [ ] **Step 3: LUT 生成器**：`gen_silu_lut(s_in, s_out) -> int8[256]`、`gen_exp_lut(...)`、`gen_rsqrt_lut(...)`，生成结果落盘 artifacts/f5/luts/ 并附校验和打印。

## Task 3: GN 定点化误差预算（关口任务）

**Files:**
- Modify: `docs/bittrue-spec.md`（§GN 参数定稿）
- Create: `artifacts/debug/` 下一次性分析脚本（不入库）

- [ ] **Step 1: 用导出包 + 标定统计推导**：全部 51 层 GN 的 var 动态范围 → 选 LUT 地址位宽（如 1024 项）与 Q 格式；蒙特卡洛注入实际标定激活，测 LUT+0 次 / +1 次 Newton 的 GN 输出 max 误差（LSB 单位）。
- [ ] **Step 2: 判据**：GN 输出误差 ≤ 1 LSB 的元素占比 ≥ 99.9%，且无 > 4 LSB → 定稿整数方案；不达标 → 契约改为"GN 的 rsqrt 用智多晶 fp32 IP，位真模拟以 float32 模拟该单点"，其余仍整数。** whichever 选定，写回契约 §3 并在 commit message 记录判据数据。**

## Task 4: 位真 UNet 前向 `src/catdiff/bittrue/model.py`

**Files:**
- Create: `src/catdiff/bittrue/model.py`、`src/catdiff/bittrue/loader.py`（解析 export-v3 包：CDW2 weights.bin、layers.json、表）
- Create: `tests/test_bittrue_model.py`

**Interfaces:**
- Consumes: `artifacts/f4/export-v3/`、`configs/bittrue.json`、`docs/bittrue-spec.md`
- Produces: `BittrueUNet.forward(x_q16, step_idx) -> eps_q16`，逐层 dict 可选返回（黄金向量用）

- [ ] **Step 1: loader**——只从导出包构建模型（结构参数从 layers.json + docs/reference/unet-config-ddpm-cat-256.json），证明导出包信息完备；若发现缺信息，回 Task 1 Step 3 扩导出器，**不得**从 PyTorch 侧补数据。
- [ ] **Step 2: 失败测试**——合成微网（单 resnet + 单注意力的小配置）：位真前向 vs fake-quant 前向，输出差异 ≤ 2 LSB 占比 > 99%。
- [ ] **Step 3: 实现完整前向**——严格按 F2 手写模型的模块结构与数据流（含 6 处注意力、非对称 pad、FFiLM 注入点），算子全部走 Task 2 原语。
- [ ] **Step 4（slow）: 真实包单步对齐**——t=950（50 档 step 0）与 fake-quant 同输入逐层对比：报告每层 max/mean LSB 误差，存 artifacts/f5/layer-align-step0.json；判据：最终 eps 差异 ≤ 2 LSB（int16 域）占比 > 99%。

## Task 5: 位真采样 + 画质复验批次

**Files:**
- Create: `src/catdiff/bittrue/sample.py`
- Create: `docs/quality-gates/f5-bittrue-quality-report.md`

- [ ] **Step 1: 采样入口**——DDIM-50 档 seed=100 断点续跑，输出 artifacts/f5/bittrue-ddim50/。
- [ ] **Step 2: 机器对比**——20 张与 F4 签字批次 artifacts/f4/int8-ddim50-mixed-seq/ 逐张 PSNR（要求 ≥ 30 dB，预期 ≫）+ 与 fp32 基准 artifacts/f1/cat256-ddim50/ 逐张 PSNR 对照表，写进报告。
- [ ] **Step 3: 20 步档冒烟 4 张**（artifacts/f5/bittrue-ddim20/），确认双档表切换链路工作。
- [ ] **Step 4: 报告 + 验收区**——格式沿用 F4 报告（机器观测 + 空白签字区），明确写"签字后 M0 关闭"。

## Task 6: 黄金向量包导出

**Files:**
- Create: `src/catdiff/bittrue/golden.py`
- Create: `artifacts/f5/golden/README.md`（给 M2 的用法说明，入库 docs/ 下一份拷贝）

- [ ] **Step 1: 逐层向量**——按 configs/bittrue.json 的步选择（50 档 3 步 + 20 档 2 步…按契约定），每层导出 in/out bin + shapes.json；总量控制在 < 500 MB（大层只存一步，在 README 写取舍理由）。
- [ ] **Step 2: 端到端向量**——初始噪声（int16 bin）→ 每步 eps（int16 bin，50 档全 50 步）→ 最终像素（png + bin）；附 checksums.txt（SHA-256 全文件）。
- [ ] **Step 3: README**——M2 对齐流程：先单层（conv_path 冒烟用 conv_in 的 step0 向量）→ 逐层 → 全 UNet 单步 → 端到端 50 步。

## 验收（M0 关闭判据）

1. `uv run pytest` 与 `uv run pytest -m slow` 全绿；
2. F5 画质报告用户签字 Go（位真网格 vs F4 签字网格 vs fp32 基准三方对照）；
3. 黄金向量包完整 + checksums + README；
4. docs/bittrue-spec.md 冻结为 v1.0 ——此后 RTL 行为以它为准，改动需走 spec 修订。
