# 位真定点契约 v1.0（M0-F5，M2 RTL 的唯一数值真理来源）

> 状态：v1.0（2026-09-25，F5 定稿）。上游：量化契约 `docs/quant-format.md` v3
> （位宽、双档表、51 个激活量化点）与 F4 签字报告 v3 节。
> 本文档把 v3 契约未定义的"层内整数行为"全部钉死：Q 格式、舍入、饱和、LUT 内容。
> **代码与本文档不一致时，改代码**；RTL 行为以本文档为准，改动走 spec 修订。
>
> 修订记录：v1.0 初版。与实施计划的两处偏差（均有依据）：
> ① GN eps 用 **1e-6**（模型 config `norm_eps=1e-6` 实测，计划文本"1e-5"有误）；
> ② conv_in 输入 x 定死 **s_x = 2^-12（Q3.12 存储，表示域 ±8）**，
>    实测 DDIM 轨迹 max|x| = **5.4684**（8 seeds × 50 步标定运行日志
>    artifacts/f5-export-requant-50.log），裕量 1.46×。

## 0. 总则

- 一切整数字段均为**二进制补码、小端**。位真模拟器用 torch/numpy 整数或 float64
  容器模拟（float64 仅作容器，所有参与运算的值均为整数，与 RTL 整数运算逐位等价）。
- 所有"除 2^k"一律**算术右移**；所有舍入一律 **round-half-up（半分向上取整，
  对负数同样成立）**：`(v + 2^(k-1)) >> k` 等价于 `floor(v/2^k + 0.5)`。
  RTL 最易写错处：负数的 floor 偏移，测试 `tests/test_bittrue_primitives.py` 钉死。
- 所有饱和到对称域 `[-qmax-1, qmax]`（INT8: [-128,127]，INT16: [-32768,32767]）。
- 量化点清单：**表定点** = v3 契约 §1.5 的 51 处（scale 见导出包
  `act_scales_<steps>.json`，按步组取值）；**内部点** = 本文 §4 定义、由扩出的
  `requant_params_<steps>.bin` 下发（§7）。两类的标定口径一致（P99.95/127，
  顺序标定 8 seeds，F4 金标协议）。
- 步组：`g = min(step_idx // (steps/groups), groups-1)`（v3 §1.4）。
  一步之内所有表定点/内部点/卷积 requant 用同一组 g 的 scale。

## 1. 数据格式总表

| 对象 | 格式 | scale 来源 |
|---|---|---|
| 内部激活（50 个表定点 + 内部点） | INT8 对称 per-tensor，[-128,127] | act_scales 表（表定点）/ requant_params（内部点） |
| x（DDIM 状态，conv_in 输入） | INT16，**s_x = 2^-12 定死**（Q3.12，±8） | 契约常量 |
| eps（conv_out 输出激活） | INT16 对称 per-tensor | s16 = act_scales['conv_out'][g] × 127/32767 |
| 权重（内部 152 层） | INT8 per-channel | weights.bin |
| conv_in/conv_out 权重 | INT16 per-channel | weights.bin |
| 累加器 acc | INT32（乘加中间积 INT64） | — |
| GN 平方和 Σx² | INT64（INT32 不够：127²×262144 = 4.2e9） | — |
| softmax 概率 | UINT8 [0,255]（无符号，无偏置补偿） | — |
| DDIM 系数 | INT32 定点（§6） | — |

**x 的 Q 格式论证**：x 初始为 randn(seed)（8 seeds 实测 max|x|≈5），DDIM 全程
x0 预测被 clip 到 [-1,1]、prev_sample = √ᾱ_prev·x0 + √(1-ᾱ_prev)·eps，
上界 ≈ max|x| ≈ 5.5。s_x = 2^-12（±8）留足裕量且为 2 的幂，便于像素映射移位。
若实测轨迹超 ±8，位真采样会饱和报警（采样脚本断言）。

## 2. conv / linear 整数流

### 2.1 累加

```
acc[i] = Σ_j x_q[j] · w_q[i][j]        # INT8×INT8 或 INT16×INT16 → INT32 累加
```
- 逐乘积 int（≤ 127×127 或 32767×127），累加用 INT32（**饱和到 ±(2^31-1)**，
  RTL: DSP 级联累加的溢出饱和保护；真实标定轨迹下 acc ≤ ~1e8 远离 2^31，
  此为防御性定义），累加中间积用 INT64。
- FiLM 偏置（仅 ResnetBlock 的 conv1）：`acc[i] += film_q32[i]`（见 §5.1）。
- 常规卷积偏置（weights.bin 的 fp32 bias）：**融合进 requant M/N 之外的
  b32 项**：`b32[i] = round_half_up(bias[i] / (s_in·s_w[i]))`（acc 域整数，
  离线由 scale 三元组算出），`acc[i] += b32[i]`。

### 2.2 requant（逐通道）

```
y_q[i] = sat_int(bits)( (acc[i]·M[i] + 2^(N[i]-1)) >> N[i] )
```
- `M[i] = round_half_up(ratio_i · 2^N[i])`，`ratio_i = s_in·s_w[i]/s_out·pre`，
  其中 `pre` 为该 requant 的固定域折算（仅注意力 av 点为 1/256，见 §5.3；其余 1）。
- N[i] 规格化：`N[i] = clamp(31 - floor(log2(ratio_i)), 0, 62)`，使
  M ∈ [2^30, 2^31)；若舍入后 M ≥ 2^31 则 N←N-1、M 重算。M ≥ 1 强制。
- M/N 由 PC 离线算出并随导出包下发（§7 requant_params）；位真模拟器**只消费
  导出包中的 M/N**，不从 scale 现算（以此验证下发文件完备性）。
- softsign 注意：M ≥ 0 恒成立（scale 非负）；符号只来自 acc。GN 输出 requant
  的 M 可为负（§3），公式不变（算术右移对负数即 round-half-up，有测试钉死）。
- INT16 输出点（conv_out）：sat 到 [-32768, 32767]，其余同式。
- conv_in：输入 x（INT16 @ s_x）× INT16 权重 → INT32；输出为表定点 INT8。
  conv_out：输入 INT8（conv_norm_out.silu 点）× INT16 权重 → INT32；
  输出 eps INT16，**不经 INT8 requant**。

## 3. GroupNorm 整数流（71 处：32 resnet×2 + 6 attention + conv_norm_out）

输入：INT8 codes @ s_in（本 GN 的输入点 scale），32 通道组，每组 G 通道 × H×W。

```
S1   = Σ codes                          # INT32（≤127×262144 = 3.3e7）
S2   = Σ codes²                         # INT64（≤4.2e9，RTL 用 64-bit 累加）
N    = G·H·W
μ_q  = round_half_up(S1·2^8 / N)        # Q8.8，输入格
d    = (code << 8) - μ_q                # Q8.8
var_q16 = (S2 << 16)//N - μ_q²          # Q16，输入格²；负则截 0
var_q16 += eps_q                        # eps_q = round(1e-6 / s_in² · 2^16)
inv_q14 = rsqrt_lut(var_q16)            # §3.1
x̂_q    = sat_int16( (d·inv_q14 + 2^9) >> 10 )     # Q3.12（s_xhat = 2^-12，±8）
# 逐通道输出 requant（γ 带符号折进 M，β 加在输出格）：
t     = (x̂_q·Gc[c] + 2^(Nc[c]-1)) >> Nc[c]        # INT64 乘积
y     = sat_int8(t + Bq[c])
```
- `Gc[c] = round_half_up(s_xhat·γ[c]/s_out · 2^Nc[c])`（**有符号** INT32，
  Nc 规格化同 §2.2 但按 |ratio|）；`Bq[c] = round_half_up(β[c]/s_out)`（INT32，
  输出格，精确无舍入误差——β/s_out 本就是格点倍数取整）。
- GN 输出点 scale（norm1/norm2/gn 等）为内部点（§4），γ/β 来自
  norm_params.bin（§7）。
- **rsqrt LUT**（参数见 configs/bittrue.json，默认 2048 项 × 11 位地址）：
  1. v = var_q16；若 v < V_MIN=2^8（即 var_x < 2^-8）则 v = 2^8；
  2. 规格化 v = m·2^E，m ∈ [2^22, 2^23)（b = bit_length(v)，E = b-23，
     m = v >> E，b<23 时 m = v << (23-b)、E 相应为负）；
  3. j = m >> 11 ∈ [2048, 4096)，地址 = j - 2048（11 位）；
  4. 偶 E：`inv_q14 = LUT0[j-2048] >> (3 + E/2)`；
     奇 E：`inv_q14 = LUT1[j-2048] >> (4 + (E-1)/2)`；
     `LUT0[k] = round_half_up(2^25 / √((k+2048)·2^11))`，
     `LUT1[k] = round_half_up(2^25·√2 / √((k+2048)·2^11))`（各 2048 项 UINT32，
     合并寻址 (E&1, j) 亦可）。
  5. 可选 1 次牛顿迭代（Task 3 关口数据决定是否启用，默认不启用；
     若启用公式：`x1 = (x0·(3·2^30 - (v·x0²)>>q) / 2` 定点式由 Task 3 定稿回写本节）。
- 误差预算（Task 3 关口）：GN 输出误差 ≤1 LSB 占比 ≥99.9% 且无 >4 LSB；
  不达标则降级"rsqrt 用 fp32 IP（IEEE754，位真模拟以 float32 模拟该单点）"。

## 4. 内部激活量化点（新增，scale 由扩导出器标定下发）

v3 的 51 处表定点只覆盖模块边界；层内整数化的 conv/GN/SiLU/注意力中间值需要
自己的量化点。命名与清单（per 步组 10 组，与表定点同协议标定）：

| 内部点 | 物理含义 | 消费者 |
|---|---|---|
| `<resnet>.norm1` / `.norm2` | GN1/GN2 输出（SiLU 输入） | SiLU LUT |
| `<resnet>.silu1` / `.silu2` | SiLU 输出 | conv1 / conv2 输入 |
| `<resnet>.hidden` | conv1(+FiLM) 输出 | norm2 输入 |
| `<upresnet>.concat` | up 路 concat 后（=norm1 输入） | norm1（两输入先 requant 到此点） |
| `<attn>.gn` | attention GN 输出 | to_q/to_k/to_v 输入 |
| `<attn>.qkv` | to_q/to_k/to_v 输出（三点共用） | QK^T、V 加权 |
| `<attn>.av` | attn@V 输出 | to_out 输入 |
| `conv_norm_out.gn` / `.silu` | 末端 GN / SiLU 输出 | SiLU LUT / conv_out 输入 |

多输入算子（residual add、concat）的 scale 对齐规则：**各输入先 requant 到
输出量化点的 scale（各自 M/N），饱和到 INT8 后相加，和再饱和到 INT8**
（RTL 不做 int8 直加）。**操作数 requant 饱和域为 INT16（±32767）、和再截
INT8**：块出点 scale 按 |和| 标定，支路本身可超 ±127（相互抵消），若按 INT8
逐操作数饱和会产生大误差（微网实测 max 20 LSB，改 INT16 域后消除）。
无权重支路的恒等 requant（residual 的 x 支路、concat
两输入、mid_block 容器双定点）的 M/N = scales_to_MN(s_from/s_to)，由导出包
scale 确定性推导（与 M/N 下发表同一公式，测试钉死），不再单独下发。

## 5. 各算子整数流

### 5.1 ResnetBlock（pre-norm，output_scale_factor=1）

```
h1  = silu_lut1( gn( x_in, norm1 ) )          # x_in: 表定点或 concat 点
acc = conv1(h1) + film_q32 + b32              # film: §0/§7，acc 域累加器初值
hid = requant(acc, → hidden 点)
h2  = silu_lut2( gn( hid, norm2 ) )
main= requant16(conv2(h2), → 表定块出点)       # INT16 饱和（§4 规则）
sc  = (有 conv_shortcut) requant16(conv_shortcut(x_in), → 同点)
      或 requant16(x_in, → 同点)               # 恒等支路也要 scale 对齐
out = sat_int8(sc + main)                     # 相加后截 INT8 ← 表定块出点
```
FiLM 表（film_table_<steps>.bin，fp32）离线量化到 conv1 acc 域：
`film_q32[i] = round_half_up(film[i] / (s_silu1·s_w1[i]))`。FiLM 表替代了
time_embedding MLP 与 time_emb_proj（运行时不消费这两组权重；注意导出包
film 表来自 fp32 时间通路，与 fake-quant 的 W8 时间 MLP 有 <1 LSB 级系统差，
Task 4 报告记录）。

### 5.2 SiLU LUT

256 项 INT8，`table[a] = sat_int8(round_half_up(silu((a-128)·s_in)/s_out))`，
a∈[0,255] 地址 = 输入 code+128。每（LUT 输入点, 输出点, 步组）一张，
由 PC 生成（`gen_silu_lut`）并随黄金向量归档（§7）。silu(0)=0 → a=128 项为 0。

### 5.3 Attention（单头 d=512，scale = 512^-0.5）

```
g   = gn(x_in, group_norm)                    # → <attn>.gn 点
q,k = requant(to_q(g)) requant(to_k(g))       # → <attn>.qkv 点（共用）
v   = requant(to_v(g))                        # → <attn>.qkv 点
s_ij = Σ q_i·k_j                              # INT32
a_ij = sat_u12( ((s_max - s_ij)·Kexp + 2^(Qe-1)) >> Qe )   # 行内减 max
e_ij = exp_lut[a_ij]                          # 4096 项 UINT16，Q1.14
p_ij = min(255, (e_ij·R + 2^23) >> 24)        # R = round_half_up(2^32 / Σ_j e_ij)
acc_i = Σ_j p_ij·v_j                          # UINT8×INT8 → INT32，**无偏置补偿**
av  = requant(acc_i, → <attn>.av 点, pre=1/256)  # 256 折进 ratio（§2.2）
o   = requant16(to_out(av), → 表定 attention 出点)
res = requant16(x_in, → 同点)
out = sat_int8(o + res)
```
- exp LUT：`entry[a] = sat_u16(round_half_up(e^(-a·2^-8)·2^14))`，Δ=2^-8 定死，
  覆盖 arg ≥ -16（更负截到 a=4095，e^-16≈1e-7 归零，无画质影响）。
- `Kexp/Qe` 离线：`Kexp = round_half_up(s_qkv²·scale_attn·2^8·2^Qe)`，
  Qe = 32（Kexp ≤ 2^31 校验）。
- softmax 全零行（减 max 后 arg 全 0）：Σe = N·2^14，R 正常，p = 256/N——
  无退化（测试钉死）。
- UINT8×INT8 无偏移：无符号概率不含 +128 偏置，累加不需补偿项。

### 5.4 residual add / concat / up/down sample

- residual add：§4 规则（两输入各自 requant 到输出点 → sat → 相加 → sat）。
- concat（up 路）：两输入各自 requant 到 `<resnet>.concat` 点 → 拼通道。
- upsample：nearest×2 纯索引复制（INT8 透传，不 requant），随后 conv 3×3
  requant 到 `upsamplers.0` 表定点。
- downsample：非对称 pad (0,1,0,1) 填 **0**（对称量化下 0 ↔ 0.0），conv s2
  requant 到 `downsamplers.0` 表定点。

### 5.5 conv_in / conv_out（INT16 边界）

- conv_in：x INT16 @ s_x × W INT16 → INT32（18×18 DSP 语义）→ requant INT8
  到 `conv_in` 表定点。
- conv_out：INT8 × W INT16 → INT32 → requant **INT16** 到 eps（s16，§1），
  直接出给 DDIM 更新单元。

## 6. DDIM 更新单元（eta=0，clip_sample=true）

系数（每步，按该步组 g 的 s16 计算，离线由 ddim_table + scale 表算出，
黄金向量包附 `ddim_coeffs_q.json`）：

```
A_q = round_half_up(2^23 / √α_t)                       # Q8.23
B_q = round_half_up(2^23 · √(1-α_t)/√α_t · (s16/s_x))  # Q8.23（s16/s_x 折入）
C_q = round_half_up(2^30 · √α_prev)                    # Q2.30
D_q = round_half_up(2^30 · √(1-α_prev) · (s16/s_x))    # Q2.30
```
（Q2.30 表示不了 1/√α_t ≈ 130 @ t=980，故方向项系数用 Q8.23；A·x、B·eps 的
INT64 积远小于 2^63。）

```
t1  = (x_q·A_q + 2^22) >> 23
t2  = (eps_q·B_q + 2^22) >> 23
x0  = clamp(t1 - t2, -4096, 4096)          # clip [-1,1]（x 格 ±4096）
prev= sat_int16((x0·C_q + eps_q·D_q + 2^29) >> 30)   # 一次舍入
```
- 初始噪声：`x_q = clamp(round_half_up(randn / s_x), -32768, 32767)`。
- 像素映射：`px = clamp(((x_q + 4096)·255 + 4096) >> 13, 0, 255)`
  （等价 round((x+1)·127.5)，x ∈ [-1,1] 已由上一步 clip 保证）。
- use_clipped_model_output=False：方向项用原始 eps（与 F2 手写调度器一致）。

## 7. 导出包增补（F5 扩展，CDW2 与既有文件不动）

```
artifacts/f4/export-v3/
├── …（v3 既有 10 个文件不变）
├── norm_params.bin            # 71 个 GN 的 γ/β（fp32）
└── requant_params_50.bin      # 内部点 scale + 全部 wired 层的 M/N（主档）
└── requant_params_20.bin      # 预览档
```
- **norm_params.bin**：magic "NRM1"，u32 层数；每记录
  `u16 name_len, name, u32 C, f32[C] gamma, f32[C] beta`。
- **requant_params_<steps>.bin**：magic "RQP1"，u32 版本=1，u32 num_groups，
  u32 num_layers，u32 num_internal；先 internal 段：每点
  `u16 name_len, name, f32[num_groups]`；后 layers 段（kind: 0=conv/linear
  逐通道 M/N，1=GN 输出 requant 逐通道 Gc/Nc/Bq，2=注意力 av requant 标量
  M/N，ratio 已含 1/256）：每层
  `u16 name_len, name, u8 kind, u8 in_kind, u16 in_idx, u8 out_kind,
   u16 out_idx, u32 C`（in/out_kind: 0=表定点/1=内部点/2=x 边界），随后逐组：
  kind0/2：`f32 s_in, f32 s_out, i32[C] M, u8[C] N`；
  kind1：`f32 s_in, f32 s_out, i32[C] Gc, u8[C] Nc, i32[C] Bq`。
  conv_in 层的 in_kind=2（x 边界，s_in = s_x = 2^-12）。
- 记录数 = 120 wired conv/linear（不含 time_embedding.\* 与 time_emb_proj，
  FiLM 表替代）+ 6 条 `<attn>.av_requant`（kind2）+ 71 条 GN（kind1，记录名 =
  GN 模块名，同 norm_params.bin）= 197。
- 内部点标定与 M/N 计算由 `src/catdiff/export/export_requant.py` 完成
  （顺序标定协议同 F4：8 seeds 200..207、每(点,步)子采样 2048、P99.95）；
  位真模拟器只消费上述文件，不从 PyTorch 侧补任何数据。
- 清单核对：wired 层数 = 154 - 2(time_embedding) - 32(time_emb_proj) = 120，
  加 6 条 av_requant + 71 条 GN 记录 = 197（见下）。

## 8. 黄金向量格式

```
artifacts/f5/golden/
├── manifest.json                 # 档位、步选择、点清单、格式说明
├── checksums.txt                 # 全文件 SHA-256
├── luts/<tier>/…                 # 生成的 SiLU/exp/rsqrt LUT 二进制
├── e2e/<tier>/
│   ├── x_init.int16.bin          # 256×256×3 INT16
│   ├── eps/step####.int16.bin    # 全部步（50 档 50 步全存）
│   ├── ddim_coeffs_q.json        # 每步 A/B/C/D 定点系数
│   ├── x_final.int16.bin
│   └── pixels.bin / pixels.png
└── layers/<tier>/step####/
    ├── shapes.json               # 本步全部张量形状与 dtype
    └── <point>.in.bin / <point>.out.bin    # INT8/16/32 小端裸序列
```
- 步选择：50 档 step 0/25/49；20 档 step 0/10/19。
- 体积控制：50 档 step0 存全部点；其余步只存空间 ≤64×64 的点与
  conv_in/conv_out（大层取舍理由见 README）。
- `<point>` 命名 = 量化点名（表定点原名，内部点 §4 命名）；
  in = 该量化点输入 codes，out = 输出 codes。

## 9. 验收判据（F5 关口汇总）

1. requant/原语单测全绿（负值舍入、饱和、int32 边界、softmax 全零行、GN 单元素）；
2. Task 3 GN 误差预算：≤1 LSB 占比 ≥99.9%，无 >4 LSB（否则 fp32 IP 降级回写 §3）；
3. Task 4 逐层对齐（真实包 step0 实测修订）：位真相对 fake-quant 的差异由
   两部分构成——(a) 实现正确性：conv_in 逐位精确（max ≤1），首个 resnet
   ≤2 LSB 占比 100%；(b) 内部 int8 量化器（SiLU/hidden/av 等）相对
   fake-quant float 内部的固有噪声，随深度随机游走积累（51 点深度处
   mean ≈2-4 LSB，max ≈30-50 LSB；FiLM 系统差实测 ≤0.11 hidden LSB，
   可忽略）。因此**逐点 ≤2 LSB 判据只适用于首块**；深度积累不设逐点门槛，
   由 §9.4 图像级 PSNR（≥30 dB）与"无结构发散"（误差有界、随深度平滑增长）
   仲裁。eps 以 conv_out 的 **INT8 基准格**（1 LSB = s8）报告
   （int16 域 LSB = s8/258，作浮点级参考同时报告）；
4. Task 5 机器标准：位真批次 vs F4 签字批次逐张 PSNR ≥ 30 dB，
   并附 vs fp32 基准对照表；人工签字由项目负责人完成（报告验收区留空）。
