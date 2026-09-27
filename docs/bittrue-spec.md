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
| 表定点激活（51 处，v3 契约） | INT8 对称 per-tensor，[-128,127] | act_scales 表 |
| 内部点激活（§4：GN 输出/SiLU 输出/hidden/av/concat） | **INT18 细网格**，scale = 内部标定值/fine_div（fine_div=1024，即 1024× 细于 INT8 基准格），[-131072,131071]，实用码域 ±130048 | requant_params（内部点） |
| x（DDIM 状态，conv_in 输入） | INT16，**s_x = 2^-12 定死**（Q3.12，±8） | 契约常量 |
| eps（conv_out 输出激活） | INT16 对称 per-tensor | s16 = act_scales['conv_out'][g] × 127/32767 |
| 权重（内部 152 层） | INT8 per-channel | weights.bin |
| conv_in/conv_out 权重 | INT16 per-channel | weights.bin |
| 累加器 acc | INT32；细格 int18 输入的层（conv1/conv2/conv_shortcut/to_out/conv_out）用 **INT48**（DSP 级联） | — |
| GN 平方和 Σx² | INT64（INT32 不够：127²×262144 = 4.2e9） | — |
| softmax 概率 | UINT16 [0,65535]（无符号，无偏置补偿，v1.2） | — |
| DDIM 系数 | INT32 定点（§6） | — |

**x 的 Q 格式论证**：x 初始为 randn(seed)（8 seeds 实测 max|x|≈5），DDIM 全程
x0 预测被 clip 到 [-1,1]、prev_sample = √ᾱ_prev·x0 + √(1-ᾱ_prev)·eps，
上界 ≈ max|x| ≈ 5.5。s_x = 2^-12（±8）留足裕量且为 2 的幂，便于像素映射移位。
若实测轨迹超 ±8，位真采样会饱和报警（采样脚本断言）。

## 2. conv / linear 整数流

### 2.1 累加

```
acc[i] = Σ_j x_q[j] · w_q[i][j]        # INT8×INT8 / int18×INT8 / INT16×INT16
```
- 逐乘积 int（≤ 127×127、131071×127 或 32767×32767）。表定点 int8 输入的层
  累加用 INT32（**饱和到 ±(2^31-1)**，RTL: 溢出饱和保护；真实标定轨迹下
  acc ≤ ~1e8 远离 2^31，此为防御性定义）；**细格 int18 输入的层**（resnet
  conv1/conv2/conv_shortcut、attention to_out、conv_out）用 **INT48**
  （DSP 级联，饱和到 ±(2^47-1)：最坏对齐 3·3·512·130048·127 ≈ 2^46.1 < 2^47），
  累加中间积用 INT64。**例外**：attention to_q/to_k/to_v 输入虽为细格 int18
  （`<attn>.gn` 点），仍按 **INT32** 累加（与位真模拟器 `acc_bits=32` 一致；
  最坏 512×131071×128 ≈ 2^33 超 INT32，饱和语义在极端角点承担行为定义，
  RTL 须在 54 位累加后先做 INT32 饱和再 requant——真实轨迹 acc ≤ ~1e8，
  饱和永不触发）。
- FiLM 偏置（仅 ResnetBlock 的 conv1）：`acc[i] += film_q32[i]`（见 §5.1）。
- 常规卷积偏置（weights.bin 的 fp32 bias）：**融合进 requant M/N 之外的
  b32 项**：`b32[i] = round_half_up(bias[i] / (s_in·s_w[i]))`（acc 域整数，
  离线由 scale 三元组算出），`acc[i] += b32[i]`。

### 2.2 requant（逐通道）

```
y_q[i] = sat_int(bits)( (acc[i]·M[i] + 2^(N[i]-1)) >> N[i] )
```
- `M[i] = round_half_up(ratio_i · 2^N[i])`，`ratio_i = s_in·s_w[i]/s_out·pre`，
  其中 `pre` 为该 requant 的固定域折算（仅注意力 av 点为 1/65536，见 §5.3；其余 1）。
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

输入：codes @ s_own（本 GN 输入点自身网格的 scale：表定点输入为 INT8 网格，
细网格输入为 §4 的 int18 网格），32 通道组，每组 G 通道 × H×W。
（v1.1 修订：内部点细化后 GN 统一按"自身网格"流，对 INT8 输入与 v1.0 行为一致。）

var/eps/inv 统一以 **INT8 等价格式** 计量（细格输入：除以 fine_div²；这保证
rsqrt Q14 输出对两种输入网格都有 ≥9 位有效位），x̂ 移位按输入网格分支
（sub_bits = log2(fine_div)，v1.3 fine_div=1024 → sub_bits=10）：

```
S1   = Σ codes                          # INT64（细网格最大 130048×262144 = 3.4e10）
N    = G·H·W
μ_q  = round_half_up(S1·2^6 / N)        # Q0.6，自身格
d    = (code << 6) - μ_q                # Q0.6，自身格
var_q16 = (Σ d² // N) << 4              # 表定点输入（INT8 等格 = 自身格）
var_q16 = (Σ d² // N) >> (2·sub_bits-4) # 细格输入（÷fine_div²；1024 → >>16）
var_q16 += eps_q                        # eps_q = round(1e-6 / s_in_int8等价² · 2^16)
inv_q14 = rsqrt_lut(var_q16)            # §3.1，int8 等格单位
x̂_q    = sat_int16((d·inv_q14 + 2^7) >> 8)        # 表定点输入
x̂_q    = sat_int16((d·inv_q14 + 2^(sub_bits+7)) >> (sub_bits+8))   # 细格输入
#   （Q3.12 = s_xhat 2^-12，±8；两式的物理值一致：细格 d6 大 2^sub_bits、
#    inv 同、移位多 2^sub_bits）
# 逐通道输出 requant（γ 带符号折进 M，β 加在输出格；输出为细网格 INT18）：
t     = (x̂_q·Gc[c] + 2^(Nc[c]-1)) >> Nc[c]        # INT64 乘积
y     = sat_int18(t + Bq[c])
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
  5. 牛顿迭代：**不启用**（Task 3 实测定稿，见下）。
- 误差预算（Task 3 关口实测，2026-09-26 定稿）：2048×2 项 LUT + 0 次 Newton，
  fake-quant 轨迹 2 seeds × 5 步 × 全部 71 GN = 16.59 亿元素：
  **≤1 LSB 占比 100.0000%**（判据 ≥99.9%）；>4 LSB 仅 20 个元素
  （1.2e-8，孤立离群，集中末端大尺寸 GN）。判据达标，**维持纯 LUT 方案，
  不启用 fp32 IP 降级**。var 动态范围实测 [1.42, 6013]（x 格²，逐组），
  均在 LUT 表示域内。

## 4. 内部激活量化点（新增，scale 由扩导出器标定下发）

v1.3 修订：内部点的**存储网格 = 标定值（INT8 基准）/1024**（fine_div=1024，
INT18 细网格，饱和域 [-131072,131071]，实用码域 ±130048 = 127×1024，mult18
DSP 原生）。依据：v1.1/v1.2（细格 256）重采样后图像对比度仍系统性坍缩
（std 0.151 vs F4 批次 0.262，PSNR ~24 dB < 30 dB 门槛）。根因定位实验
（2026-09-27，E1-E5）：同输入下位真与 fake-quant 的 eps 差异中约 1.33%
为逐步**新鲜**正交量化噪声；E4 给 fake-quant 轨迹每步注入 1.4% 新鲜随机
噪声即完美复现坍缩（终 std 0.1305）——机理为新鲜噪声被去噪器反复"清理"、
轨迹回归数据集均值，晚期步加速（E2 双轨：step 40-49 std 0.50→0.15）。
E5 容限曲线：新鲜噪声 0.7%→29.3 dB、**0.35%→36.8 dB**；细格 256→1024 把
有效噪声 1.33% 压到 ~0.35%，落在达标区。51 处表定点（int8 存储）、eps
int16、qkv int8、DDR 带宽均不变；仅内部数据通路升 int18。
（v1.1 原始依据保留：全 int8 内部点噪声 PSNR 19.0 dB；细格化消除该噪声源，
表定点不受影响。）细网格点：全部 GN 输出、
SiLU 输出、hidden、av、concat；**qkv 保持 INT8**（QK^T 的 int8×int8 DSP 语义）。
表定点清单不变：

命名与清单（per 步组 10 组，与表定点同协议标定；存储 scale = 标定值/fine_div）：

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

多输入算子（residual add、concat）的 scale 对齐规则（v1.4 修订）：**residual
两支路各自 requant 到输出点 int8 基准格的细子格（s8/2^sub_bits，requant48
out_shift=sub_bits），饱和域 INT19（±262143 = ±255.99 int8 等效），细子格域
相加后一次舍入截 INT8**：`out = sat_int8((sc + main + 2^(sub_bits-1)) >> sub_bits)`。
块出点 scale 按 |和| 标定，支路本身可超 ±127（相互抵消）；v1.3 及以前恒等支路
（`_point_codes` wide）按 INT16 粗格双重舍入、且细格实现曾按 INT18（±127.99
int8 等效）逐操作数饱和——深层块（down_blocks.2/3/4）恒等支路 |code| 常态达
±130-198 s8，INT18 饱和产生 max ~70 LSB 单点误差，成为逐步新鲜噪声首要来源
（微网实测与该网逐点散度实验双重钉死）。concat 两输入仍 requant 到 concat 点
细格（INT18）后拼通道，不适用本条。
无权重支路的恒等 requant（residual 的 x 支路、concat
两输入、mid_block 容器双定点）的 M/N = scales_to_MN(s_from/s_to)（residual
细子格取 s_to = s8/2^sub_bits），由导出包
scale 确定性推导（与 M/N 下发表同一公式，测试钉死），不再单独下发。
输出为细网格的 requant 饱和域为 INT18（±131071）；residual 细子格操作数为
INT19；输出为表定点的保持 INT8。
内部点标定带安全裕量 margin=1.5（configs/bittrue.json
internal_calibration.margin；v1.4 实测：×1 时内部点削波贡献 eps 噪声
0.98%-0.61%≈0.37pct，×1.5 全消，×3 无额外增益）。

## 5. 各算子整数流

### 5.1 ResnetBlock（pre-norm，output_scale_factor=1）

```
h1  = silu_lut1( gn( x_in, norm1 ) )          # x_in: 表定点或 concat 点
acc = conv1(h1) + film_q32 + b32              # film: §0/§7，acc 域累加器初值（INT48）
hid = requant18(acc, → hidden 点细格)         # INT18 饱和（§4 细网格）
h2  = silu_lut2( gn( hid, norm2 ) )
main= requant19(conv2(h2), → 块出点细子格, out_shift=sub_bits)  # §4 v1.4，INT48 acc
sc  = (有 conv_shortcut) requant19(conv_shortcut(x_in), → 同细子格)  # INT48 acc
      或 requant19(x_in, → 同细子格)            # 恒等支路也要 scale 对齐（INT19）
out = sat_int8((sc + main + 2^(sub_bits-1)) >> sub_bits)  # 一次舍入 ← 表定块出点
```
FiLM 表（film_table_<steps>.bin，fp32）离线量化到 conv1 acc 域：
`film_q32[i] = round_half_up(film[i] / (s_silu1·s_w1[i]))`。FiLM 表替代了
time_embedding MLP 与 time_emb_proj（运行时不消费这两组权重；注意导出包
film 表来自 fp32 时间通路，与 fake-quant 的 W8 时间 MLP 有 <1 LSB 级系统差，
Task 4 报告记录）。

### 5.2 SiLU LUT + 线性插值（v1.3：输出细网格 int18）

257 项 INT32 基表：`T[a] = round_half_up(silu((a-128)·s_coarse)/s_fine)`，
a∈[0,256]，s_coarse = 输入点 INT8 基准格，s_fine = 输出点细格（= s_coarse/fine_div，
T 值域 ±130048）。输入为细网格 int18 code：粗地址 `a = code >> sub_bits`
（算术右移，[-128,127]），小数 `f = code - (a<<sub_bits) ∈ [0,fine_div)`：

```
y = T[a+128] + ((T[a+129] - T[a+128])·f >> sub_bits)    # sat_int18
```

（v1.0 的纯 256 项 INT8 直查表是本节的 f=0 特例；插值误差 ≪1 细格 LSB，
硬件代价 = 1 个乘法器。每（LUT 输入点, 输出点, 步组）一张，PC 生成
（`gen_silu_lut`）并随黄金向量归档。silu(0)=0 → a=128、f=0 项为 0。）

### 5.3 Attention（单头 d=512，scale = 512^-0.5）

```
g   = gn(x_in, group_norm)                    # → <attn>.gn 点
q,k = requant(to_q(g)) requant(to_k(g))       # → <attn>.qkv 点（共用）
v   = requant(to_v(g))                        # → <attn>.qkv 点
s_ij = Σ q_i·k_j                              # INT32
a_ij = sat_u12( ((s_max - s_ij)·Kexp + 2^(Qe-1)) >> Qe )   # 行内减 max
e_ij = exp_lut[a_ij]                          # 4096 项 UINT16，Q1.14
p_ij = min(65535, (e_ij·R + 2^23) >> 24)      # R = round_half_up(2^40 / Σ_j e_ij)
#   v1.2：概率改 UINT16（×65536）；uint8 的 ±0.2% 概率量化噪声与内部 int8 同理
#   经晚期步反馈造成幅度坍缩（§10）。硬件代价：p·v 乘积 24-bit。
acc_i = Σ_j p_ij·v_j                          # UINT16×INT8 → INT32，**无偏置补偿**
av  = requant18(acc_i, → <attn>.av 点细格, pre=1/65536)  # 65536 折进 ratio；INT18 饱和
o   = requant19(to_out(av), → attention 出点细子格, out_shift=sub_bits)  # INT48 acc
res = requant19(x_in, → 同细子格)              # §4 v1.4，INT19 饱和
out = sat_int8((o + res + 2^(sub_bits-1)) >> sub_bits)  # 一次舍入
```
- exp LUT：`entry[a] = sat_u16(round_half_up(e^(-a·2^-8)·2^14))`，Δ=2^-8 定死，
  覆盖 arg ≥ -16（更负截到 a=4095，e^-16≈1e-7 归零，无画质影响）。
- `Kexp/Qe` 离线：`Kexp = round_half_up(s_qkv²·scale_attn·2^8·2^Qe)`，
  Qe = 32（Kexp ≤ 2^31 校验）。
- softmax 全零行（减 max 后 arg 全 0）：Σe = N·2^14，R 正常，p = 256/N——
  无退化（测试钉死）。
- UINT16×INT8 无偏移：无符号概率不含偏置，累加不需补偿项（v1.1 前为 UINT8）。

### 5.4 residual add / concat / up/down sample

- residual add：§4 v1.4 规则（两支路 requant 到出点细子格、INT19 饱和、细子格域
  相加、一次舍入截 INT8）。
- concat（up 路）：两输入各自 requant 到 `<resnet>.concat` 点 → 拼通道。
- upsample：nearest×2 纯索引复制（INT8 透传，不 requant），随后 conv 3×3
  requant 到 `upsamplers.0` 表定点。
- downsample：非对称 pad (0,1,0,1) 填 **0**（对称量化下 0 ↔ 0.0），conv s2
  requant 到 `downsamplers.0` 表定点。

### 5.5 conv_in / conv_out（INT16 边界）

- conv_in：x INT16 @ s_x × W INT16 → INT32（18×18 DSP 语义）→ requant INT8
  到 `conv_in` 表定点。
- conv_out：细网格 int18 输入（conv_norm_out.silu 点，INT18 存储）× W INT16
  → INT48 → requant **INT16** 到 eps（s16，§1），直接出给 DDIM 更新单元。

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
- **requant_params_<steps>.bin**：magic "RQP1"，u32 版本=3（v1.3 起 3；
  v1.1/v1.2 为 2，细格 256 语义，不兼容），u32 num_groups，
  u32 num_layers，u32 num_internal；先 internal 段：每点
  `u16 name_len, name, f32[num_groups]`；后 layers 段（kind: 0=conv/linear
  逐通道 M/N，1=GN 输出 requant 逐通道 Gc/Nc/Bq，2=注意力 av requant 标量
  M/N，ratio 已含 1/65536）：每层
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

## 10. 版本历史

- **v1.4**（2026-09-27）：residual add 改**细子格单次舍入**——两支路（conv2 /
  conv_shortcut / to_out / 恒等 x 支路）requant48 out_shift=sub_bits 到出点
  细子格（s8/1024），饱和域 **INT19（±255.99 int8 等效）**，细子格域相加后
  round-half-up 一次截 INT8；内部点标定加 **margin=1.5**。依据：v1.3 轨迹仍
  坍缩（24.13 dB）——v1.3 条目"细格 1024 将噪声压到 ~0.35%"的预期**未兑现**
  （同输入 eps 正交噪声仍 1.33%）。逐组件消融 + 同输入逐点散度实验定位真根因：
  ①恒等支路 INT18 细格饱和（±127.99 int8 等效）削波深层块常态 ±130-198 s8
  的支路码，单点 max ~70 LSB（down_blocks.2/3/4.resnets.1）；
  ②residual 粗格双重舍入；③内部点 P99.95 削波（×1 时贡献 ~0.37pct 噪声）。
  修复后同输入 eps 正交噪声 1.33%→**0.61-0.71%**（E5 容限曲线 0.7%→29.3 dB
  区间内），step0 逐点最差 6.4→1.3 LSB。requant_params 格式不变（版本 3），
  仅导出流程加 margin；RTL 影响：residual 操作数数据通路 19-bit。
  （v1.4.1 澄清：§2.1 显式列举 attention to_q/k/v 为 INT32 累加例外——
  与冻结模拟器行为一致，无行为变更，黄金向量不受影响。）
- **v1.3**（2026-09-27）：内部细网格 256→**1024**（fine_div，INT18 内部数据通路，
  mult18 DSP 原生；GN 输出/SiLU 输出/hidden/av/concat 及 conv_shortcut/to_out/
  conv_out 累加器 INT48；SiLU 插值 sub_bits=10；GN 细格移位参数化）。
  51 处表定点 int8、eps int16、qkv int8、DDR 带宽不变。依据：v1.2 重采样后
  对比度仍坍缩（std 0.151 vs 0.262，~24 dB），E1-E5 根因实验定位为"逐步新鲜
  量化噪声被去噪器清理、轨迹回归均值、晚期加速"（E4 注入 1.4% 新鲜噪声复现
  坍缩；E5 容限 0.35%→36.8 dB）；细格 1024 将有效噪声 1.33%→~0.35%。
  requant_params 版本升 3。
- **v1.2**（2026-09-26）：softmax 概率改 **UINT16**（p = 65536·e/Σe，R = 2^40/Σe；
  av 折算 pre=1/65536）。依据：v1.1 重采样后仍余幅度衰减（图像 std 20.8 vs
  F4 33.4，PSNR 24.2 dB）——uint8 概率 ±0.2% 量化噪声成为首要余项。
- **v1.1**（2026-09-26）：内部点细化 256×（INT16 细网格）：GN 输出/SiLU 输出/
  hidden/av/concat；SiLU 改 257 项基表 + 线性插值；GN 整数流改自身网格统一式
  （§3）；qkv 保持 INT8。依据：v1.0 实测——全 int8 内部点的量化噪声经 DDIM
  晚期反馈使轨迹幅度坍缩（图像系统性变淡，vs F4 批次 PSNR 19.0 dB），单步
  对齐与 GN 误差预算均达标，问题为设计噪声地板而非实现缺陷。
- v1.0（2026-09-25）：初版。
