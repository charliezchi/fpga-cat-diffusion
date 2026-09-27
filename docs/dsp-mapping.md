# 智多晶 Seal DSP 适配性论证与映射方案

| 文档信息 | |
|---|---|
| 版本 | v1.1（2026-09-27，PM 复核修订：风险 5/9 关闭、§4.1 折叠条件核验、§10 新增） |
| 状态 | M2 卷积引擎 RTL 的前置决策文档（PM 复核完成，待项目负责人签批） |
| 上游 | 契约 `docs/bittrue-spec.md` v1.4（位宽唯一真理来源）；原语参考实现 `src/catdiff/bittrue/primitives.py`；算力基线 `docs/feasibility-and-plan.md` §3；黄金向量 `artifacts/f5/golden/` |
| 平台输入 | 智多晶 Seal（海豹）系列 DSP 原语，已对照 HqFpga 3.1.1 仿真库逐模块核实（§2） |
| 分支 | feat/dsp-map（纯文档，不改 src/） |

---

## 0. 结论摘要

**Go**——契约 v1.4 的全部整数运算（conv MAC、requant48、GN 全流程、SiLU 插值、softmax、attention、residual 细子格、DDIM 更新、像素映射）都可以在 Seal DSP 原语（xsMULT18/9 + xsPREADD18/9 + xsALU54）上位真实现，无需要求契约修订的位宽障碍；超宽运算（requant48 79-bit 积、GN 全精度 Σ 2^64、softmax/DDIM 31-bit 系数乘）全部有多拍拆分方案，且逐元素总量只占卷积的 ~1%。

**两项必须带回给决策层的重要发现**：

1. **算力口径修正（主发现）**：按冻结导出包 `artifacts/f4/export-v3/layers.json` 的 154 层真实形状核算，每去噪步实际为 **241.4 GMAC**，是可行性报告 §3.2"48 GMAC/步"的 **5.0 倍**（该表把通道数与分辨率配错了档，详见 §5.1）。分辨率映射已用黄金向量 shapes.json 逐点交叉验证（`conv_in/out` 256²、down_blocks.2 64²、down4 attention 16²=256 token，与 feasibility §5.2 一致）。后果：**"DDIM-50 ≈ 47s"基线在本器件上不可达**——即使用满 744 mult18 @200MHz，DDIM-50 ≈ **86s**；DDIM-20 ≈ **35s**，落在 20~60s 验收窗口内（§5.3）。**已定案（2026-09-27）**：演示默认 DDIM-20，DDIM-50 不作承诺指标；FPGA 预留运行时步数寄存器 + 多档系数表常驻 DDR（§5.3、§9.1 决策①）。
2. ~~**黄金包 SiLU LUT 文件是 v1.0 遗留**~~（**已关闭**，2026-09-27 PM 复核）：`luts/silu_*.int8.bin` 内容实为 257×INT32 且与同源生成器逐位一致——仅文件名与 README 停留在 v1.0 口径；已改名 `.int32.bin`、修正生成器与两份 README、重建 checksums（399/399 通过）。**e2e/layers 黄金向量内容与签名未动。**

三项最大风险：① ~~算力口径修正带来的指标/排期影响~~（**已关闭**，本节第 1 条定案）；② requant48 拆分的位真实现（79-bit 中间积 + round-half-up + out_shift，§4.1，已给出严格精确性证明与两段式方案）；③ 200MHz 时序收敛（累加反馈环本身在单 ALU54 内、风险低，压力在 PE 外围逻辑与 CIN 级联链，§8 风险 3）。

---

## 1. 输入与依据

本文档所有位宽数字标注出处，格式：`primitives.py:L行` 或 `spec §节/L行`；层形状与统计来自冻结导出包；DSP 行为来自仿真库源码行号。速查表见附录 B。

| 依据 | 用途 |
|---|---|
| `docs/bittrue-spec.md` v1.4（§1–§6） | 数据格式总表、各算子整数流 |
| `src/catdiff/bittrue/primitives.py` | 每个原语的精确位宽（RTL 唯一对齐基准） |
| `artifacts/f4/export-v3/layers.json`（154 层，calib_hash `ca1541db484cee7f`） | 实际布线层形状 → 算力核算 |
| `artifacts/f5/golden/layers/50/step0000/shapes.json`（38 点） | 分辨率映射交叉验证 |
| `docs/feasibility-and-plan.md` §3 | 板卡资源（744 DSP18×18 / 120K LE / DDR3-1866）、48G 口径、47s 基线、4.4GB/s 带宽供给 |
| `artifacts/f5/golden/README.md`、`docs/quality-gates/f5-golden-readme.md` | M2 对齐流程（黄金向量要对齐什么） |
| `C:\hqfpga\hqv3_xist_3.1.1_FT092126_win64\build\common\sim\verilog\XIST\seal\` | DSP 原语行为核实（§2） |
| 用户实板确认 | Slice 组成 = 1×xsALU54 + 2×xsMULT18 + 2×xsPREADD18；SA5T-200 共 744 mult18 = 372 slice（仿真库不含 slice 级互连，此处按用户提供口径） |

---

## 2. Seal DSP 原语核实（仿真库证据）

仿真库路径：`...\sim\verilog\XIST\seal\`。与 DSP48E1 逐项对照的结论：**功能家族相同（预加器→乘法器→ALU/累加器，带级联链），但结构为三器件分离**——乘法器不在 ALU 内部（DSP48E1 是 25×18 内嵌），MA/MB 以 36 位端口送入 ALU54。

### 2.1 xsMULT18 / xsMULT9（乘法器）

- 18×18 有符号/无符号乘法 → P[35:0]（`xsMULT18.v:20-36` 端口，`xsMULT18.v:739` `p_sig_i = a_sig_m * b_sig_m`）。**SIGNEDA/SIGNEDB 逐输入独立选择**（`xsMULT18.v:39,677-737` 符号扩展），这是本文多处拆分方案的关键能力：一端吃有符号激活、另一端吃无符号部分积。
- 四级可配置流水：REG_INPUTA / REG_INPUTB / REG_PIPELINE / REG_OUTPUT（`xsMULT18.v:56-67`），静态参数化，200MHz 下满流水无障碍。
- 级联口：SRIA/SRIB（18 位级联入，接 PREADD 的 SROA/SROB）、ROA/ROB/SROA/SROB（级联出），SOURCEA/SOURCEB 选择直通或级联（`xsMULT18.v:39,196-232`）。
- xsMULT9：9×9 → P[17:0]（`xsMULT9.v:20-33`），即 xsMULT18 的可拆半片；用户确认 **1×xsMULT18 = 2×xsMULT9**，int8×int8 双密度模式由 xsALU54 的 `MULT9_MODE` 参数（`xsALU54.v:104`）佐证。

### 2.2 xsALU54（54-bit ALU / 累加器）

端口（`xsALU54.v:20-41`）：A[35:0]、B[35:0]、C[53:0]、MA[35:0]、MB[35:0]（乘法器结果入）、CIN[53:0]（DSP 级联链入，DSP48E1 PCIN 的对应物）、OP[10:0] 操作码 → R[53:0]，另有 EQZ/OVER 等标志位。

输入 mux 结构（行号为 `xsALU54.v`）：

| mux | 选项 | 行号 | 对本项目的意义 |
|---|---|---|---|
| a_mux | `r_sig`（自身输出反馈）/ `MA` / `{C[17:0], A}`（18 位桶形移位）/ 0 | 1516-1523 | **累加反馈环**：R = MA + X + acc 单拍完成；`{C,A}` 是 18 位台阶的乘积对齐 |
| b_mux | `MB<<18` / `MB` / `{C[44:27], B}` / 0 | 1525-1532 | `MB<<18` 把第二个乘积按 18 位台阶对齐——**一个 slice 的 2×MULT18 两积可单拍并入同一累加器** |
| c_mux | 0 / CIN>>18 / CIN / C / {C<<18,A} / `r_sig` / 舍入 pattern | 1566-1580 | CIN 级联用于跨 PE 归约树；C 口吃 requant 舍入偏移 2^(N−1) 等常数 |

运算核（`xsALU54.v:1582-1605`）：`4'b0100: R = A+B+C`、`0101: A−B+C`、`0110: A+B−C`、`0111: A−B−C`，加按位与/或/异或。**纯加减/逻辑，无内建乘法**（乘法在 xsMULT18）。

对本项目的三个核心推论：

1. **单 slice 卷积吞吐 = 2 MAC/拍**：a_mux=MA、b_mux=MB、c_mux=r_sig、OP=A+B+C，一拍内完成 P1 + P2 + acc（两个 36 位积 + 54 位累加 ≤ 2^48，装得下 54 位；契约最坏 acc ≈ 2^46.1，spec §2.1/L59-60）。INT32 与 INT48 累加同为单拍反馈环，**无需跨 slice 级联即可完成 48 位累加**。
2. **18 位桶形台阶天然匹配拆分乘法**：任何 36 位以内操作数按 18 位台阶对齐（{C,A} 左移、MB<<18）都是硬件原生路径——本文所有"宽乘拆部分积"方案都建立在这上面。
3. **CIN 级联 = 跨 PE 54+54 位加法**，用于最终归约树（GN 组间求和、多 PE 部分和合并），位真无损（纯整数加法结合）。

### 2.3 xsPREADD18 / xsPREADD9（预加器）

- 18 位加/减器：`po_sig = OPPRE ? A−B : A+B`（`xsPREADD18.v:1183`），带 C[17:0] 第三输入、SOURCEA_MODE（A_SHIFT/A_PARALLEL/C_SHIFT/A_C_DYNAMIC/HIGHSPEED，`xsPREADD18.v:579-655`）、SOURCEB_MODE（SHIFT/PARALLEL/INTERNAL/ZERO，`xsPREADD18.v:661-687`）。
- 输出 PO[17:0] 可经 SROA/SROB 移位路径馈入 xsMULT18 的 SRIA/SRIB——即 DSP48E1 式"预加后乘"。可用性分析见 §6（结论：本项目大概率闲置）。

### 2.4 与 DSP48E1 的差异清单（RTL 设计注意）

| 项 | DSP48E1 | Seal | M2 影响 |
|---|---|---|---|
| 乘法器 | 25×18 内嵌 + 17 位桶形移位 | 独立 xsMULT18（18×18），ALU 另设 18 位桶形 | 拆分粒度从 17 变 18；**25×18 语义不存在**，INT16×INT16 边界层（conv_in）直接 18×18 覆盖（18>16，单乘即可） |
| 累加器 | 48-bit | 54-bit（更宽，GN 分块更从容） | 契约 INT48 全部落在单 DSP 内 |
| 预加器 | 25-bit（A:B） | 18-bit 独立器件 | 见 §6 |
| 级联 | PCOUT 48-bit | CIN 54-bit | 归约树位宽更富余 |
| 双密度 | 无（INT8 需软件打包） | xsMULT9 原生拆分 + MULT9_MODE | int8×int8 层可双密度（§5.4 结论：不值得为主阵列启用） |

---

## 3. 契约 v1.4 算子 → Seal 映射总表

每步调用次数来自 §5.1 的实测统计（依据 layers.json + 黄金向量 shapes 验证）；"每图（DDIM-50）" = 每步 × 50。DSP·拍以 mult18·拍与 ALU54·拍分开计（一个 slice 每拍供给 2 mult18 + 1 ALU54）；折算时间按方案 A 全阵列（744 mult18 / 372 ALU54 @200MHz）取瓶颈资源。

### 3.1 卷积/矩阵类（占每步算力 99.7%）

| 运算 | 输入精确位宽（出处） | Seal 原语映射 | 单实例 DSP·拍 | 每步调用次数 | 全阵列折算 |
|---|---|---|---|---|---|
| conv 3×3/1×1、linear（细格层：conv1/conv2/conv_shortcut/to_out/to_q/k/v） | 激活 INT18 细格 ±[131072,131071]（primitives.py:17，spec §1/L34、§4/L137-139）× 权重 INT8 per-channel（spec §1/L37） | 1 MAC = 1×xsMULT18（A=18b 激活，B=8b 权重符号扩展）→ ALU54 `A+B+C` 双积合累加；**2 MAC/slice/拍**；INT48 acc 单 DSP 反馈环 | 1 mult18·拍/MAC（ALU 摊 0.5） | 241.4G MAC/步（§5.1） | 1.62s/步（81.4s/图） |
| conv_in 边界层 | x INT16 @ s_x=2^-12 Q3.12（primitives.py:29，spec §1/L35、§2.2/L81）× W INT16（spec §1/L38） | 1×xsMULT18（16b×16b 装得下）→ INT32 acc；256²×3 输入仅 1 层 | 1 mult18·拍/MAC | 0.23G MAC/步（128×3×9×65536） | 1.5ms/步（76ms/图） |
| conv_out 边界层 | 激活 INT18 细格 × W INT16 → INT48（spec §5.5/L261-262） | 1×xsMULT18 → INT48 acc | 1 mult18·拍/MAC | 0.23G MAC/步 | 1.5ms/步 |
| QK^T | q,k INT8（<attn>.qkv，spec §4/L150、§5.3/L225） | 1×xsMULT18（int8×int8）→ INT32 acc，复用主阵列；L=256/64 token | 1 mult18·拍/MAC | 169.9M MAC/步（6 处，§5.1） | 1.15ms/步 |
| av 加权 | p UINT16 [0,65535]（spec §1/L41、§5.3/L230）× v INT8 → INT32 无偏置补偿（spec §5.3/L233、L245） | 1×xsMULT18（16b×8b）→ INT32 acc，复用主阵列 | 1 mult18·拍/MAC | 169.9M MAC/步 | 1.15ms/步 |

### 3.2 逐元素类（占每步算力 0.3%，在 pe_array 时分相位或独立辅助通路执行）

| 运算 | 输入精确位宽（出处） | Seal 原语映射 | 单实例 DSP·拍 | 每步调用次数 | 全阵列折算 |
|---|---|---|---|---|---|
| requant48（细格层输出） | acc INT48 ±2^47（primitives.py:81-83，spec §2.1/L58-61）× M ∈ [2^30,2^31) INT32、N ∈ [0,62]（primitives.py:71-78，spec §2.2/L72-75）；residual 支路 out_shift=sub_bits=10（spec §4/L168-169、v1.4 §10/L361-372） | 见 §4.1：acc 拆 3×18、M 拆 18+13 → 6 部分积，2 阶段精确进位；6 mult18·拍 + ~2 ALU·拍，流水 1 元素/3~4 拍/slice | 6 mult18 + 2 ALU | 158.4M 元素/步（§5.1） | 0.95G mult18·拍 → 6.4ms/步 |
| requant32（表定点输出层 + qkv） | acc INT32（spec §2.1/L56-58）× M/N 同上 | 同 §4.1 降 1 维：acc 拆 2×18 → 4 部分积；4 mult18 + 1 ALU | 4 mult18 + 1 ALU | 25.2M 元素/步（含 qkv 2.1M） | 0.10G mult18·拍 → 0.7ms/步 |
| GN 均值 Σ | codes INT8 或 INT18 细格，Σ INT64（spec §3/L96：细格最大 130048×262144=3.4e10=2^35.0） | ALU54 累加（2^35 ≤ 2^53）或 fabric；N=G·H·W 恒为 2 的幂（§4.2）→ μ 除法退化为移位 | 1 ALU/元素 | 165.9M 元素/步 | 0.17G ALU·拍 → 0.9ms/步 |
| GN 方差 Σd² | d² 单项 ≤ 2^46（d=d6 ≤ 2^23，spec §3/L99、§4/L138），全精度 Σ ≤ 2^64 超 xsALU54（spec §1/L40 明确 INT64） | **代数展开 Σd² = 2^12·Σcode² − 2^7·μ·Σcode + N·μ²（精确整数恒等）**：每元素 1 MULT（code², 18×18）+ 1 ALU 累加（全组 Σcode² 最坏 2^52 ≤ 2^53，**单 ALU54 直接累加、无需分块**，见 §4.2）；右端三项为每组一次的组级常量工作 | 1 mult18 + 1 ALU/元素 | 165.9M 元素/步 | 0.17G mult18·拍 → 0.9ms/步 |
| GN x̂ = d·inv | d = code<<6 ≤ 2^23（24b，spec §3/L99+§4 细格）× inv_q14 值域 [2^7, 2^18]（primitives.py:255-277：shift 3+E/2 可为负；RSQRT_VAR_MIN_Q16=256 primitives.py:26 → 下限角点 inv=2^18 超 18 位符号口，见 §4.3） | d 拆 18+6 → 2×xsMULT18 + ALU54 合并（b_mux=MB<<18），出口 >>18（=sub_bits+8，spec §3/L105）+ 饱和 INT16 | 2 mult18 + 1 ALU | 165.9M 元素/步 | 0.33G mult18·拍 → 2.2ms/步 |
| GN 逐通道 γ requant | x̂ INT16 ±32767（primitives.py:315-321 饱和）× Gc 有符号 INT32、|Gc| ∈ [2^30,2^31)（primitives.py:120-125，spec §3/L112-114）；+ Bq INT32 → 饱和 INT18 | Gc 拆 18+13 → 2×xsMULT18 + ALU54（+2^(Nc−1) 走 C 口），出口 >>Nc + 饱和 INT18 | 2 mult18 + 1 ALU | 165.9M 元素/步 | 0.33G mult18·拍 → 2.2ms/步 |
| SiLU LUT 插值 | 输入 INT18 细格 code；257 项 INT32 基表 T[a]（primitives.py:208-217，spec §5.2/L208-214），f ∈ [0,1023]（primitives.py:220-228） | ΔT=T[a+1]−T[a] ≤ ~2^12（= silu'max×1024+舍入，推导见 §4.4，**远小于 18 位口**）×f（10b）→ **1×xsMULT18 单拍**；+T0 与 >>sub_bits 用 fabric/ALU | 1 mult18（+1 ALU 可选） | 165.9M 元素/步（65 个 GN→SiLU 点对） | 0.17G mult18·拍 → 1.1ms/步 |
| residual 细子格加（v1.4） | 两支路 INT19 ±262143（primitives.py:17-18 QMAX[19]，spec §4/L168-169） | 19b+19b 加 + 2^9 偏移 + >>10 + 饱和 INT8——**纯 fabric**（19 位 > PREADD18 的 18 位口；且发生在 requant 出口、DSP 域外） | 0（fabric 加法器） | 60.6M 元素/步（32 resnet 块出点 + 6 attention 出点） | 0 |
| concat（up 路） | 两输入 requant 到 concat 点 INT18（spec §4/L160、§5.4/L251） | 2×requant48（上两行）+ 通道拼接（布线，无运算） | 同 requant48 | 已含在 158.4M 内 | — |
| 恒等 requant（residual x 支路/mid 容器） | M/N 由 scale 确定性推导（spec §4/L176-179） | 同 requant48/32 通路，M=1 或常规 | 同上 | 已含 | — |
| softmax 减 max | s INT32 域但 \|s\| ≤ 512×128² = 2^23（spec §5.3/L227-228）；Kexp ≤ 2^31（Qe=32，spec §5.3/L241-242） | (m−s) 先按层常量预截至 (4096·2^32−2^31)/Kexp（利用 LUT 索引 clip [0,4095] 的单调性，primitives.py:343），积 ≤ 2^44 → 2×xsMULT18（Kexp 拆 18+13）+ ALU54 >>32 对齐 | 2 mult18 + 1 ALU | 270.3K 元素/步（4×65536+2×4096） | 0.5M mult18·拍 → 忽略 |
| softmax exp 查表 | a ∈ [0,4095]（spec §5.3/L239-240：Δ=2^-8 定死，Q1.14） | exp LUT 4096×u16 BRAM 直查 | 0 | 270.3K/步 | 0 |
| softmax 归一 R | R = round_half_up(2^40/Σe)（primitives.py:346）；Σe ∈ [2^14, L·2^14] ⊂ [2^14, 2^22]（行内 max 元素 a=0 → e=2^14 保底，L ≤ 256）→ R ∈ [2^18, 2^26] | 倒数 LUT + Newton 1~2 次迭代（~4-6 mult18·拍/行）；**PC 端对 Σe 全域 [2^14, 2^22] 穷举验证位真后冻结**（约 420 万值，离线可行） | ~5 mult18/行 | 1536 行/步（6 处×L 行） | 0.4M mult18·拍 → 忽略 |
| softmax p·R | e UINT16 × R ∈ [2^18, 2^26]（primitives.py:346-348，spec §5.3/L230-231）→ 积 ≤ 2^42 | R 拆 12+14 → 2×xsMULT18 + ALU54（+2^23 走 C 口）>>24，min 65535 饱和 | 2 mult18 + 1 ALU | 270.3K 元素/步 | 0.5M mult18·拍 → 忽略 |
| DDIM t1/t2 | x INT16 × A/B Q8.23 ≤ 2^30.03（spec §6/L269-276：1/√α_t≈130@t=980） | 系数拆 18+13 → 2×xsMULT18 + ALU54，+2^22 走 C 口，>>23 fabric | 2 mult18 + 1 ALU（各一次） | 196.6K 元素×2/步（256²×3） | 0.8M mult18·拍 → 忽略 |
| DDIM prev | x0 INT16（实际 ≤ ±4096，13b，spec §6/L281）× C/D Q2.30 ≤ 2^30（spec §6/L271-274） | 4 部分积（C、D 各拆 18+13）→ 2 拍 4 mult18 + 1 ALU 合加 +2^29 >>30 一次舍入 | 4 mult18 + 1 ALU | 196.6K 元素/步 | 0.8M mult18·拍 → 忽略 |
| x_to_pixel | x INT16，px = ((x+4096)·255+4096)>>13（primitives.py:369-372，spec §6/L285） | (x+4096)·255 = ((x+4096)<<8)−(x+4096) 移位减法，**0 DSP** | 0 | 196.6K/图（仅末步） | 0 |
| upsample nearest / downsample pad | INT8 透传/填 0（spec §5.4/L253-255） | 索引复制，**0 DSP** | 0 | — | 0 |
| FiLM 注入 | film_q32 INT32，acc 域累加器初值（spec §5.1/L192、L200-201；黄金 README L56-57） | 作为 conv1 的 ALU54 累加器初值经 C 口注入，**0 额外 DSP**（占用 1 拍累加器写） | ~0 | 9984 通道/步 | 忽略 |
| b32 常规偏置 | INT32 acc 域（spec §2.1/L63-65） | 同 FiLM，C 口注入 | ~0 | 每卷积层 cout | 忽略 |

### 3.3 总账（方案 A 全阵列，@200MHz）

| 类别 | mult18·拍/步 | ALU54·拍/步 | 全阵列折算（瓶颈资源） |
|---|---|---|---|
| 卷积/linear/QK^T/av | 242.2G | 121.1G | 1628ms |
| requant48/32 | 1.05G | 0.35G | 7.1ms |
| GN（统计 + 归一） | 0.83G | 0.68G | 9.1ms（ALU 瓶颈相） |
| SiLU | 0.17G | ~0.08G | 1.1ms |
| softmax + DDIM + 其他 | <0.02G | <0.01G | <0.1ms |
| **合计** | **≈244.3G** | **≈122.2G** | **≈1645ms/步** |

> 两种口径互相印证：按瓶颈资源分相合计 ≈ 1.65s/步；按 slice·拍 合计（GN 4 + SiLU 1 + requant48 3 + requant32 2 拍/元素）= 1.36G slice·拍/步 ÷ 372 slice = 3.6M 拍 = 18.2ms 逐元素相 + 卷积 1622ms ≈ 1.64s/步。**逐元素总量 ≈ 卷积的 1.1%**——这就是 §5.2"逐元素时分复用主阵列"结论的定量依据。

---

## 4. 超宽运算拆分方案（逐项）

### 4.1 requant48：48×31 → 79-bit 积（契约最重的逐元素运算）

**约束**：acc INT48（spec §2.1/L58-61）；M ∈ [2^30, 2^31)、N ∈ [0,62]（primitives.py:71-78）；round-half-up `(acc·M + 2^(N−1)) >> N`（spec §2.2/L70）；residual 支路 out_shift=10（spec §4/L168-169）。79 位全宽积既超 xsMULT18（18×18）也超 xsALU54（54 位）。

**拆分方案（18 位台阶，匹配 ALU 原生对齐）**：
- acc48 = A2·2^36 + A1·2^18 + A0（A2 有符号 12b，A1/A0 18b）；M = Mh·2^18 + Ml（Mh 有符号 13b，Ml **无符号** 18b——SIGNEDB 逐输入可选符号，xsMULT18.v:39）。
- 6 个部分积：A_i·Mj 全部 ≤ 18×18，各占 1 mult18·拍；同 slice 2 MULT/拍 → 3 拍产出。
- **两段式精确进位**（复刻 primitives.py:86-106 的 16 位拆解，任意拆位基 b 均成立）：把 acc 分为低段 lo 与高段 hi，
  `c = (lo·M + 2^(N−1)) >> b`（lo·M ≤ 16×31=47 位或 18×31=49 位，单 ALU54 装下），
  `y = (hi·M + c) >> (N−b)`。
  精确性引理：设 W = lo·M + 2^(N−1)，c = W>>b，余 W' = W mod 2^b，则
  `(P + 2^(N−1))>>N = (hi·M + c + W'/2^b) >> (N−b)`，而 `frac((hi·M+c)/2^(N−b)) + W'/(2^b·2^(N−b)) < (2^(N−b)−1+1)/2^(N−b) = 1`，**严格不跨整数界**——即两段式与全宽计算逐位相同。primitives.py 的 16 位拆解与此 18 位拆解是同一恒等的两个实例，模拟器已单测钉死（primitives.py:85-86"逐位一致，测试钉死"）。
- **out_shift 免费折叠**：`(P·2^os + 2^(N−1))>>N = (P + 2^(N−1−os)) >> (N−os)`（os ≤ N−1；os=10、N≥30 成立）——residual 细子格 requant 与普通 requant48 **共用同一条数据通路，仅 (N, off) 常数不同**，无需独立电路。primitives.py:87-99（文档串 L87-89、预截代码 L98-99）的 `2^62` 预截只是 int64 容器防回绕（primitives.py:89 注释自证"无实际语义影响"），RTL 不需要复刻。**条件核验（2026-09-27 PM 复核）**：折叠要求 N ≥ os+1 = 11；冻结导出包实测全部导出层 min N = 35（down_blocks.4.attentions.0.av_requant），恒等支路 min N = 20（down_blocks.0.resnets.1, g0）——全域满足且裕量 ≥9；M2 导出流程加断言 `N ≥ sub_bits+1` 防回归。
- **成本**：6 mult18·拍 + ~2 ALU·拍/元素，流水后每 slice 吞吐 1 元素/3~4 拍。全网络 158.4M 元素/步 → 占全阵列 16ms/步（表 §3.3）。
- **备选**：CIN 级联双 ALU54（54+54=108 位）单拍式部分积归约——省拍数但占两 slice，不作为主案。

**残余风险**：方案有证明，风险收敛为 RTL 实现错误（负数 floor、饱和对称域 [−2^47, 2^47−1] 等）——由黄金向量 `layers/ddim50/step0000/` 逐位对齐兜底（golden README L38-45 冒烟顺序）。

### 4.2 GN 统计：全精度 Σd² 可达 2^64，超 xsALU54 的 54 位

**约束**：Σx² 契约 INT64（spec §1/L40）；d = (code<<6) − μ，细格 code ≤ 2^17 → d6 ≤ 2^23，d² ≤ 2^46；N = G·H·W ≤ 262144（spec §3/L97）；全精度 Σd² ≤ 2^46×2^18 = 2^64。

**方案（两层）**：

1. **代数展开消去大乘**：Σd² = Σ(code·2^6 − μ)² = **2^12·Σcode² − 2^7·μ·Σcode + N·μ²**——纯整数多项式恒等，无舍入，位真无损。每元素只需 code²（1×xsMULT18，18×18）+ Σcode² 累加（1 ALU；Σcode² ≤ 131071²×2^18 = 2^52 ≤ 2^53，单 DSP 装下）+ Σ 累加（1 ALU）；右端三项是**每组一次**的常数工作量（2272 组/步，忽略不计；注意组级合成式中 2^12·Σcode² 中间值可达 2^64，用 fabric 64 位加/移实现，逐元素通路与 DSP 均不涉及）。对比逐元素直接累加 d²（需 24×24 拆 3 部分积），每元素省 ~1.5 mult18·拍。
2. **溢出核算：主案无需分块**。展开后每元素只累加 code²（≤ 2^34）与 code；全组 Σcode² 最坏 = 131071²×262144 ≈ 2^52.0 ≤ 2^53−1（54 位二补码上限），**单 ALU54 直接累加装得下，仅 1 位裕量**；Σ 本身有正负对消（最坏 |Σ| ≤ 2^35.0，spec §3/L96）更安全。若 M2 出于实现原因选择直算 d²（逐元素 (code<<6−μ)²，需 3 部分积），才需要分块：块大小 K 按最坏界取 K×2^46 ≤ 2^52 → **K = 64 项/块**（262144/64 = 4096 块，块间 CIN 级联双 ALU 108 位或 fabric 64 位加法树归约——整数加法结合律保证与全宽累加逐位相同）；实测 var 域（spec §3/L132 [1.42, 6013]×2^16）下 d² 典型 ≤ 2^41，K 可放大 32 倍以上。K 也可按层离线定值：PC 端从黄金向量直接算出每层每组的真实 max|Σ|，位真性不受 K 选择影响（只要不溢出）。
3. **除法全部退化为移位**：N = ch_per_g×H×W，全网络空间尺寸与每组通道数均为 2 的幂 → N = 2^n。μ_q = round_half_up(S1·2^6/N) = `(S1·128 + N) >> (n+1)`（与 primitives.py:306 `(2·s1·64 + n)//(2n)` 逐位一致，算术右移即契约 round-half-up，spec §0/L18-19）；var 的 //N 同理（spec §3/L100-101）。**GN 无需任何除法器**。
- 成本：统计相 1 mult18 + 2 ALU/元素（Σ 与 Σcode² 两个独立累加器是 ALU 瓶颈，可双 slice 分摊或块状交替）；归一相（x̂+γ）见 §4.3。

### 4.3 GN x̂（24×14~18）与 γ requant（16×31）

- **x̂ = (d·inv + 2^(sub_bits+7)) >> (sub_bits+8)**（spec §3/L105，primitives.py:315-321）。d6 24 位（细格）× inv_q14：inv 值域 [2^7, 2^18]（primitives.py:255-277——Q14 是格式不是幅值上限；b=9 的下限角点 E=−14、左移 4 → inv = 2^18，**恰超 18 位有符号口**）。主案：d 拆 dH·2^18+dL（6b+18b）→ 2×xsMULT18 并行 + ALU54 合并（b_mux=MB<<18），出口 18 位右移 + INT16 饱和在 fabric；**角点处理**：inv=2^18 仅在 var_q16 触 2^8 下限（primitives.py:26）时出现，此时 x̂ = d·(2^18+i)>>18 = d + (d·i)>>18——常数项 d 经 c_mux 并入同一 ALU 拍即可，无需第二通路。实测 var_q16 ∈ [1.42, 6013]×2^16（spec §3/L132）远离下限，角点仅为防御完备性。
- **γ requant：t = (x̂·Gc + 2^(Nc−1)) >> Nc，y = sat18(t + Bq)**（spec §3/L109-110，primitives.py:324-328）。x̂ 16b × |Gc| ∈ [2^30,2^31) 31b：Gc 拆 18+13 → 2 MULT + ALU（积 ≤ 2^46，装得下）；>>Nc 是**无舍入纯移位**（舍入偏移已在 +2^(Nc−1)），fabric 常量移位网络（47b→18b 窗口）实现；Bq 加法与 INT18 饱和在 fabric。
- 成本：归一相每元素 (2M+1A) + (2M+1A) = 4 mult18 + 2 ALU·拍。

### 4.4 SiLU 插值（1 乘法方案）

`y = T[a+128] + ((T[a+129]−T[a])·f >> 10)`（spec §5.2/L214，primitives.py:220-228）。关键宽度是 ΔT = T[a+1]−T[a]：

ΔT = [silu(z+δ) − silu(z)] / s_fine ≈ silu'(z)·s_in/s_fine，其中 silu'max ≈ 1.0998（z≈1.28）、s_in/s_fine = fine_div×(s_in8/s_out8 基准格比)。相邻量化点标定 scale 同量级（比值 O(1)），故 **ΔT ≤ ~2^12**（11~12 位有符号），远小于 18 位乘法口——线性区逐点差恒为 1024，过渡区 ≤ ~1127，负区 |ΔT| ≤ ~57。**注意不能用 T 值域 ±130048（spec §5.2/L210）推断 ΔT**——那是表首尾差，不是相邻差。f = code & 1023（10b）。

映射：ΔT 预减（2 个 BRAM 读口 + fabric 减法，或 ALU）→ 1×xsMULT18（ΔT×f ≤ 2^22）→ +T0、>>10、饱和 INT18 在 fabric/ALU。**单实例 1 mult18·拍**。257×32b 表双端口 BRAM（或 2 片单口）每层换载（§7）。

### 4.5 softmax 三处乘（(m−s)·Kexp、e·R 16×26、QK^T/av）

- **(m−s)·Kexp**（primitives.py:341-343，Qe=32 spec §5.3/L241-242）：|s| ≤ 512×128² = 2^23（512 维 int8 点积上界，含 −128），u = m−s ≤ 2^24，Kexp ≤ 2^31 → **原始积可达 2^55，超 54 位 ALU**；但 a = clip((u·Kexp + 2^31)>>32, 0, 4095)（primitives.py:343）的 clip 单调 → 按**层常量** U_max = ⌈(4096·2^32 − 2^31)/Kexp⌉ 预截 u（一个比较器），预截后积 ≤ 2^44 ✓ 装 ALU54；Kexp 拆 18+13 → 2 MULT + 1 ALU >>32 对齐。
- **e·R**（primitives.py:346-348）：e ≤ 2^16；Σe ≥ 2^14（行内 max 元素 a=0 → e=2^14 保底）、Σe ≤ L·2^14 ≤ 2^22（L ≤ 256）→ R = rh(2^40/Σe) ∈ [2^18, 2^26]；e·R ≤ 2^42 ✓ 装 ALU54。R 拆 12+14 → 2 MULT + 1 ALU，+2^23 走 C 口，>>24 后 min(·,65535)。
- **R = rh(2^40/Σe)** 是唯一的真除法：倒数 LUT（Σe 高位索引）+ Newton/Goldschmidt 1~2 迭代，**迭代算法对 Σe ∈ [2^14, 2^22] 全域的位真性在 PC 端穷举验证（约 420 万值）后冻结 RTL**——这是 softmax 进 M2 前的一个独立小任务，失败兜底为逐行微码迭代除法（1536 行/步，时间无压力）。
- QK^T/av：见 §3.1，复用主阵列；双密度结论见 §5.4。

### 4.6 DDIM 更新四乘（16×24/31、13×31、16×31）与像素映射

- t1 = (x·A + 2^22)>>23、t2 = (eps·B + 2^22)>>23（spec §6/L279-281，primitives.py:361-364）：A/B Q8.23 ≤ 2^30.03（spec §6/L275-276），拆 18+13 → 各 2 MULT + 1 ALU。
- prev = (x0·C + eps·D + 2^29)>>30（primitives.py:365）：x0 ≤ ±4096（13b，clip 后，spec §6/L281）×C/D Q2.30 ≤ 2^30 → 4 部分积 2 拍 + 1 ALU 合加（≤ 2^44+2^44+2^29 ✓ 54 位）+ 一次舍入。
- x_to_pixel：((x+4096)·255+4096)>>13，255·v = (v<<8)−v 移位减法，0 DSP（primitives.py:369-372）。
- 总量：196.6K 元素/步 × ~11 DSP·拍 ≈ 2.2M DSP·拍/步 → 全阵列 ~15μs/步，**建议独立小型 ddim_unit**（步边界运行，与阵列解耦，§9）。

---

## 5. 算力口径修正与阵列规模建议

### 5.1 实测算力：241.4 GMAC/步，可行性口径的 5.0 倍

统计方法（脚本与逐层数据见附录 A）：`artifacts/f4/export-v3/layers.json` 全部 154 层，剔除 time_embedding.linear_1/2 与 32 个 time_emb_proj（运行时不消费，FiLM 替代，spec §5.1/L200-204、黄金 README L56-57），按 UNet2DModel 结构赋分辨率，与黄金向量 shapes.json 交叉验证（conv_in/out 256²、down_blocks.2 64²、down3 32²、down4 16²=256 token ✓ feasibility §5.2/L202 同口径）。

| 类别 | MAC/步 | 占比 |
|---|---|---|
| conv3×3 | 229.1G | 94.9% |
| conv1×1（conv_shortcut） | 10.9G | 4.5% |
| attention linear（to_q/k/v/to_out ×6） | 1.4G | 0.6% |
| **合计** | **241.4G** | 100% |
| （参考）QK^T + av | 339.7M | 0.14% |

逐元素量：卷积输出元素 183.6M/步；GN/SiLU 165.9M/步；细格 requant48 158.4M/步；表定点 requant32 25.2M/步；softmax 270.3K 元素/步；DDIM 196.6K 元素/步。

**与可行性 §3.2/L73"≈48 GMAC"的差异来源**：该表（L77-84）把通道档与分辨率档配错——例如其 L0 行按"64×64、128ch、单层 75.5M"推算，而真实 down_blocks.0（256²、128ch）单层 conv1 = 128×128×9×65536 = **9.66G**，仅 down_blocks.0 一个 block（2 resnet + downsampler 共 5 层 conv3×3）就 ≈ 48.3G，**恰好等于原口径的全网总量**；up_blocks.5（256²、128ch、cin 含 skip 达 256）一个 block ≈ 99.7G。"int18 激活/逐元素/辅助运算"对 MAC 口径无影响（v1.3 细格化只改数据通路位宽，不改 MAC 数，spec §10 v1.3/L373-380"51 处表定点、qkv、DDR 带宽均不变，仅内部数据通路升 int18"）。

### 5.2 阵列配置建议

前提能力（§2.2 推论 1）：**每 slice = 2 MAC/拍**（2×xsMULT18 双积 + ALU54 单拍合累加，INT32/INT48 皆单拍反馈环），744 mult18 = 372 slice = **148.8 GMAC/s @200MHz**。

| 方案 | 配置 | 卷积吞吐 | 逐元素代价 | DDIM-20 | DDIM-50 | 调度复杂度 |
|---|---|---|---|---|---|---|
| **A（推荐）** | **372 slice 全配 pe_array，逐元素为阵列的时分相位**（层描述符加 op 模式；逐元素相位期间权重预取照跑） | 148.8 GMAC/s | 3.6M 拍/步 ≈ 18ms（占 1.1%，§3.3） | **≈34s** | **≈86s** | 中：新增"逐元素相位"描述符与相位切换冲刷 |
| B | 256 slice 主阵列（512 MAC）+ 116 slice 独立辅助通路（232 mult18 + 116 ALU54） | 102.4 GMAC/s | ~58ms/步串行（tile 级可重叠至 ~29ms） | ≈48s | ≈120s | 低：两通路独立，无时分 |
| C（可行性基线口径） | 256 MAC（feasibility §3.2/L89） | 51.2 GMAC/s | — | ≈95s | ≈237s | — |

**推荐 A**（**已定案，2026-09-27：方案 A**），理由：① 逐元素只占 1.1%，为它永久牺牲 1/3 阵列不划算；② slice 结构对两类相位完全同构（GN x̂/γ 的 2 MULT+1 ALU、requant48 的 6 部分积、SiLU 的 1 MULT 都天然落在"2 MULT+1 ALU"的 slice 模板里，§4）；③ 相位切换只发生在层边界——与已有的权重重载/流水冲刷同点，不新增调度粒度；④ B 案的 48s 对 20~60s 窗口只剩 20% 裕量，任何降频都出窗。

时序敏感性：A 案 @150MHz → DDIM-20 ≈ 44s（仍在窗内）；@125MHz → ≈53s（贴边）。B 案 @150MHz → 64s（出窗）。**A 案对频率风险的容忍度也显著更好。**

### 5.3 单图耗时估算与 47s 基线对比（方案 A，200MHz，利用率 100%）

| 项 | 每步 | DDIM-50 | DDIM-20 |
|---|---|---|---|
| 卷积/linear/QK^T/av | 1628ms | 81.4s | 32.6s |
| 逐元素相位（requant/GN/SiLU） | ~18ms | 0.9s | 0.4s |
| 调度/冲刷余量（~5%） | ~82ms | 4.1s | 1.7s |
| **合计** | **≈1.73s** | **≈86s** | **≈35s** |

对比 47s 基线（feasibility §3.2/L90-91：48G ÷ 51.2G/s = 0.94s/步 × 50）：**差异的主因是 §5.1 的 5.0× MAC 口径修正**，不是架构损失；方案 A 相对同口径的"256 MAC"基线配置已有 2.7× 提速（4.71→1.73s/步）。补充敏感性：250MHz 收敛则 DDIM-50 ≈ 69s、DDIM-20 ≈ 28s。

**对外口径（已定案，2026-09-27）**：演示默认 **DDIM-20**（F1 关口已定默认，feasibility L91/风险 1b 关闭记录）≈ 34s ✓ 满足"20~60s"验收（§9/L269）；**DDIM-50 ≈ 86s 不作为承诺指标**，仅作预留档位。FPGA 实现预留**运行时步数寄存器/步数参数**：多档系数表（requant/FiLM/DDIM 系数，每档 ~5.8MB，20/30/50 三档 ~17MB）常驻 DDR，切档 = 切表指针，无额外硬件代价。DDIM-30 中间档已生成并评审后放弃（保真达标 min 32.87/mean 37.84 dB，但画质增量不被认可）。

### 5.4 QK^T 用 xsMULT9 双密度是否值得单独阵列——结论：不值得

- 体量：QK^T + av = 339.7M MAC/步 = 每步算力的 **0.14%**；复用主阵列（单密度）仅 +2.3ms/步。
- 双密度的理论上限收益：int8×int8 拆 2×xsMULT9 → 密度 ×2，省 ~1.2ms/步。而代价是：独立阵列需要独立的累加链（xsALU54 的 MULT9_MODE 模式下 a_mux/b_mux 承载 2×MULT9 拼积）、独立的控制与 BRAM 端口，且与主阵列的 int18 模式互斥、切换有冲刷。
- 附带结论：主阵列对 int8×int8 层（conv_shortcut 1×1、up/downsample conv，合计 <5% MAC）也**不启用双密度**——收益 <2.5%，不值一次模式切换复杂度。主阵列固定 MULT18 单密度。

### 5.5 带宽复核（方案 A）

- 权重：**108.1MB**（INT8 有效载荷，PM 复核订正——feasibility §3.3/L100 的"45MB"有误，layers.json 实测 113.6M 参数、运行时剔除 FiLM 替代的时间通路 5.44M；weights.bin 容器 114MB 即原始码流）÷ 1.73s ≈ **63MB/s**，占 DDR3 供给 4.4GB/s（feasibility §3.4/L108-109）的 1.4%。
- 激活：v1.3 内部点为 INT18 细格（spec §4/L137-139），DDR 侧按 ≥2.25B/元素打包——块出点为表定点 INT8 回落。层边界流量 ≈ 184M 元素 ×(读+写) ×~1.3（中间点）×2.25B ≈ 1.1GB/步 ÷ 1.73s ≈ **640MB/s（15%）**。注意点：**层内细格链（conv→requant48→GN→SiLU）应驻留片上（EBR 行缓冲 + 空间分块流水，feasibility §5.2 conv_path 既有设计），仅块边界落 DDR**，否则全落 DDR 会使流量 ×3 以上逼近供给的 50%+。INT18 打包格式（18/24/32b 对齐）是 M2 描述符要定的一个子项。

---

## 6. 预加器 xsPREADD 可用性分析（诚实评估：本项目大概率闲置）

**结论：744 个 xsPREADD18/9 预计全部闲置；不实例化、零面积代价；记录在案，M2 若遇 fabric 时序压力可回看唯一理论可用点。**

逐类排除：

1. **卷积无对称系数结构**：PREADD 的 DSP48E1 经典用途是对称 FIR 折叠（(x[n]+x[n−k])·w）与 2D 对称卷积省一半乘法器。本项目权重是训练所得的非对称浮点直接量化（spec §1/L37），无对称可折。
2. **全部契约加法都不在乘法前**：卷积加法是乘后累加（ALU54 r_sig 反馈）；residual 细子格加是 19b+19b（spec §4/L168-169）——**超过 PREADD18 的 18 位口**，且发生在 requant 出口、DSP 域外，fabric 加法器更合适；softmax 的 m−s 是 32 位域（primitives.py:341）；GN 的 d = (code<<6) − μ 细格下是 24 位域——都装不进 18 位口。
3. **唯一理论可用点（如实记录）**：表定点 INT8 输入的 GN（块入口 norm1 等，spec §3/L87-88"表定点输入为 INT8 网格"）的 d6 = (code<<6) − μ：code 8b<<6 = 14b ✓ 装得下 PREADD18，且 μ 可走 C 口、A_C_DYNAMIC 动态选路（xsPREADD18.v:610-629），PO 直接馈 MULT 的 SRIA 做 x̂ 的乘法——省一个 fabric 加法器。但 v1.3 之后 GN 输入以细格 INT18 为主流（§4/L149-150 全部 GN 输出/SiLU 输出/hidden/av/concat 为细格；表定点输入仅剩块入口），code<<6 = 24b 溢出口，主流路径用不上；int8 入口的那个 fabric 14 位减法器本来就不是瓶颈。**判定：闲置。**
4. **副产品价值**：PREADD 的 SROA/SROB 移位寄存器链与 HIGHSPEED 半周期采样模式（xsPREADD18.v:59,642-649）是数据通路资源，若 M2 想省行缓冲 BRAM 可评估其级联移位能力——属于备选方案记录，非需求。

---

## 7. ROM/RAM 需求粗表

| 表 | 规格（出处） | 容量 | 放置策略 |
|---|---|---|---|
| SiLU LUT | 每（输入点,输出点,步组）一张 257×INT32（spec §5.2/L208-211，primitives.py:208-217）；65 个 GN→SiLU 点对（32 resnet×2 + conv_norm_out；6 个 attention GN 出 to_q/k/v 不经 SiLU）× 10 步组 = 650 张 | 650×1028B ≈ **0.67MB** | DDR 常驻 + 片上双表换载（当前层 2×257×32b ≈ 2KB BRAM），随层描述符 DMA |
| exp LUT | 4096×UINT16，Δ=2^-8、Q1.14，与步组无关（spec §5.3/L239-240，primitives.py:231-237） | **8KB** | BRAM 常驻 |
| rsqrt LUT ×2 | 偶/奇指数各 2048×UINT32（spec §3/L117-127，primitives.py:240-252） | **16KB** | BRAM 常驻 |
| GN γ/β → Gc/Nc/Bq | 71 GN ≈ 28.8K 通道；量化后 9B/通道（Gc i32 + Nc u8 + Bq i32，spec §3/L112-114、§7/L308-309）；norm_params.bin 实测 222KB（fp32 原始） | 量化后 ≈ **0.26MB** | DDR，按层换载（当前层 ≤ 512ch×9B ≈ 4.6KB） |
| requant M/N + b32 | 120 wired 层（spec §7/L311-313）≈ 30K 输出通道 × 9B（M i32 + N u8 + b32 i32）；requant_params_50.bin 实测 4.54MB（含 10 步组全部记录与内部 scale） | 运行时活跃集 ≈ **0.3MB**（全包 4.5MB） | DDR；当前层 M/N/b32 换载或经描述符流式 |
| FiLM 表 | film_q32：Σ conv1 cout = 9984 通道 × 50 步 × i32（film_table_50.bin 实测 1.95MB，**逐步而非按组**——修正 spec §5.1/L200"按步组"口径的直觉） | **1.95MB** | DDR，按步切换（每步一次 2MB 读，带宽可忽略） |
| DDIM 系数 | 每步 A/B/C/D 4×i32（spec §6/L269-274；golden `ddim_coeffs_q.json`） | 50×16B = **0.8KB** | DDR/寄存器堆 |
| 权重 | layers.json 实测 154 层 113.6M 参数（INT8 权重码 113.56MB）− FiLM 替代的时间通路 5.44M = **运行时 108.13MB**；weights.bin 容器 114MB 即原始码流（feasibility §3.3/L100 的"45MB"为口径错误，PM 复核订正） | **108.1MB** | DDR 常驻 |
| 激活 | 最大层 256²×128ch INT18 ≈ 18.9MB（打包后） | 分块驻留 EBR + DDR 分块（feasibility §3.3/L102 8MB/层按 INT8 口径，**INT18 后 ×2.25 需上修至 ~18MB/层**） | DDR 分块 + EBR 行缓冲 |
| 片上合计 | exp 8KB + rsqrt 16KB + SiLU 2KB + 参数双缓冲 ~10KB | **~36KB ≈ 2~3 个 xsBRAM32K** | 占比极小；EBR 总量与帧存归属按 M1 实测（feasibility §7/M1） |

> 修正记录：feasibility §3.3/L104"DDIM 常数表 + FiLM 表 ~2MB"低估——仅 film_table_50.bin 即 1.95MB，加上 requant_params_50.bin 4.5MB，参数侧非权重总量 ≈ **6.7MB**（DDR 容量无压力）。
>
> 修正记录（PM 复核补）：feasibility §3.3/L100"权重 ≈ 45MB"为第二处口径错误（第一处是 48 GMAC 算力）——真实运行时权重 108.13MB。参数合计（权重 108.1 + requant 4.5 + FiLM 1.95 + GN 0.26 + SiLU 0.67 + 其余 LUT/系数 ~0.1）≈ **114MB，全驻 DDR**（2GB 的 5.7%）。

### 7.1 参数存储与上电加载路径（已定案，2026-09-27）

- **片上**：活跃表仅 ~36KB（上表"片上合计"行），参数主体不可能驻片上。**运行时查表全部命中 BRAM**（exp/rsqrt 常驻；SiLU/requant/GN 参数按层 DMA 预取双缓冲，KB 级换载被流水掩盖），DDR 仅作驻留仓库，无运行时查找流量。
- **片外 Flash**：16Mbit SPI Flash = 2MB，只够比特流 + STM32 固件，**放不下 114MB 参数**。
- **定案：参数全驻 DDR3，上电后经以太网灌入**。路径：PC 上位机 → GbE（板载 AR8035 PHY，feasibility §4/L162）→ FPGA（RGMII+MAC+UDP 分块可靠传输）→ DDR，114MB ≈ **5~10 秒**（UART 3Mbaud/6 分钟方案废弃；PCIe 不予考虑——无硬核、FPGA IP+PC 驱动代价相对 GbE 无收益）。STM32F103 角色收窄为**电源管理**。
- **上位机软件**（M3 正式项）：参数加载、动态种子（高斯噪声 393KB/张下发）、步数寄存器切换（多档表常驻 DDR）、生成图像回传（调试期不依赖 HDMI）。控制面"RISC-V 软核 vs 纯 RTL FSM"在 M0.5/M3 评估。

---

## 8. 风险清单

| # | 风险 | 等级 | 说明与对策 |
|---|---|---|---|
| 1 | ~~**算力口径修正的指标影响**~~ **已关闭**（2026-09-27 项目负责人定案） | ~~高~~ | 原风险：241.4G/步下 DDIM-50 ≈ 86s 超出 20~60s 验收窗。定案：**默认档 DDIM-20（≈34s ✓），不追求 DDIM-50；运行时步数寄存器 + 多档系数表常驻 DDR**（§9.1 决策①）。DDIM-50 仅作为预留档位，无指标承诺 |
| 2 | **requant48 位真拆分**：79 位中间积、round-half-up、out_shift 折叠、负数算术右移 | 高 | 方案已给严格精确性证明（§4.1 引理）+ out_shift 常数折叠；风险收敛为 RTL 实现——黄金向量 `layers/ddim50/step0000/` 冒烟顺序（golden README L38-45）+ primitives 单测对齐（spec §9.1/L345） |
| 3 | **200MHz 时序收敛**：累加反馈环在单 ALU54 内（1 拍、风险低），真正压力在 PE 外围（输入 mux、行缓冲 BRAM、requant 移位网络、饱和逻辑）与 CIN 级联链 | 中 | feasibility 风险 3 延续：频率参数化、150MHz 也能保 DDIM-20 = 44s ✓；DSP 满流水配置（REG_INPUT/PIPELINE/OUTPUT 全开）；CIN 链长度在归约树设计时限制 ≤ 8 级 |
| 4 | **GN Σ 全宽 2^64**：超单 ALU54，展开/累加方案若实现疏漏破坏位真 | 中 | §4.2 双保险：代数展开（精确恒等，Σcode² 最坏 2^52 单 ALU 可容、仅 1 位裕量）；若走直算 d² 备选则 K=64 分块 + CIN 级联归约；PC 端按黄金向量真实 max|Σ| 逐层复核裕量 |
| 5 | ~~to_q/k/v 累加位宽契约未显式列举~~ **已关闭**（2026-09-27 PM 复核） | ~~中~~ | 复核：最坏 \|acc\| = 512×131071×128 ≈ **2^33**（本文原写 2^30.9 有误），确实超 INT32；模拟器 `model.py::_attention` 实际走 `_linear_layer` 默认 `acc_bits=32`（INT32 饱和）。处置：契约 §2.1 已显式增补"qkv 为例外，INT32 累加 + 饱和"，与冻结模拟器行为一致，黄金向量无需重生成；RTL 按"54 位累加 → INT32 饱和 → requant"实现（真实轨迹 acc ≤ ~1e8 ≈ 2^26.6，饱和永不触发） |
| 6 | **inv_q14 下限角点**：var 触 2^8 下限时 inv = 2^18 超 18 位有符号乘法口（primitives.py:26,255-277） | 低 | §4.3 角点路径（常数项并 c_mux）；实测 var_q16 ∈ [1.42,6013]×2^16（spec §3/L132）远离下限，仅防御完备 |
| 7 | **面积/布线**：372 slice 全占用，无 DSP 裕量；requant 移位网/GN 归一 fabric 逻辑消耗 120K LE 的 LUT 预算 | 中 | 方案 B（512+116）作为面积退路；LUT 侧在 M2 综合后实测，SiLU/累加等简单逻辑优先 BRAM/流水复用 |
| 8 | **int18 激活 DDR 流量**：×2.25 于 int8 口径；层内链若全落 DDR 带宽占用将逼近 50%+ | 中 | §5.5：层内细格链驻留 EBR 分块流水，仅块边界落 DDR；INT18 DDR 打包格式在 M2 描述符定稿；M1 实测带宽（feasibility 风险 5 一并核实） |
| 9 | ~~黄金包 SiLU LUT 文件 v1.0 遗留~~ **已关闭**（2026-09-27 PM 复核） | ~~低~~ | 复核：`luts/silu_*.int8.bin` **内容实为 257×INT32**（1028B/张），与 `gen_silu_lut` 同源表逐位一致——只是 v1.3 前的文件名/README 未更新。已处置：文件改名 `.int32.bin`、`golden.py` 生成器与两份 README 同步修正、checksums.txt 重建（399/399 校验通过）；e2e/layers 向量内容与签名均未动 |
| 10 | **softmax R 除法的位真验证**：倒数 LUT+Newton 对 Σe 全域的精确性未证 | 低 | §4.5：PC 端 [2^14, 2^22] 穷举验证（约 420 万值）；兜底逐行微码除法，时间无压力（1536 行/步） |

---

## 9. 结论：Go/No-go 与 M2 引擎划分建议

### 9.1 Go/No-go

**Go。** 判据逐条：

1. 契约 v1.4 每一类整数运算都有位真保持的 Seal 映射（§3 总表 + §4 拆分方案），无"必须改契约才能落地"的位宽；
2. 累加位宽全覆盖：INT32/INT48 单 DSP 反馈环、INT64 统计靠展开+分块、54 位 ALU 对 48 位 acc 有 6 位裕量；
3. 算力充分性：主算力（conv）用满 744 mult18 后每步 1.65s，逐元素仅 +1.1%，attention 类可忽略（0.14%）；
4. 存储无瓶颈：片上表 ~36KB，参数合计 ≈ 114MB 全驻 DDR（权重 108.1MB + 非权重 6.7MB），DDR 带宽占用 ≤ 16%（§5.5、§7、§7.1）。

**附带决策项（M2 启动前需项目负责人确认）**：① ~~DDIM-50 指标口径~~ **已定案（2026-09-27）**：演示默认档 **DDIM-20**，不追求更高档；FPGA 实现预留**运行时步数寄存器/步数参数**——多档系数表（requant/FiLM/DDIM 系数，每档 ~5.8MB，20/30/50 三档 ~17MB）常驻 DDR，切档 = 切表指针。DDIM-30 批次已生成并经评审放弃（位真保真达标：bittrue30 vs fq30-final min 32.87/mean 37.84 dB，画质增量用户不认可）；② ~~方案 A/B 选择~~ **已定案（2026-09-27）：方案 A**（372 slice 全配 pe_array，充分利用 DSP 资源；面积退路 B 不再保留为默认，仅在 A 案布线/时序失败时回看）；③ ~~to_q/k/v 累加位宽~~ **已关闭**（风险 5，契约 v1.4.1 增补）；④ ~~黄金包 LUT 重导出~~ **已关闭**（风险 9，改名+checksums 重建）；⑤ ~~上电加载路径~~ **已定案（2026-09-27）：GbE 以太网加载 + 上位机软件**（§7.1，UART 废弃、PCIe 否决）。

### 9.2 M2 引擎划分建议（方案 A 口径）

```
DDR3 ──权重/表──► [预取 DMA]          ┌────────────────────────────┐
                                      │ pe_array（372 slice 全配）    │
DDR3 ──激活分块──► [行缓冲 EBR] ──►   │ 相位0 conv/linear/QK^T/av   │
                                      │ 相位1 requant48/32           │
                                      │ 相位2 GN 统计（Σ/Σcode²）     │
                                      │ 相位3 GN 归一（x̂/γ）+SiLU     │
                                      │ 相位4 residual/concat        │
                                      └──────────┬─────────────────┘
              gn_unit（逻辑模式而非独立硬件）：        │
              · rsqrt LUT 16KB BRAM + E/m 规格化       │
              · 组状态 RAM（μ/var/inv × 32 组）        │
              · Σcode²/Σ 组累加（单 ALU 直容，§4.2）  ▼
softmax_unit（独立小单元，无 DSP）:          块出点表定点 INT8 → DDR
              · exp LUT 8KB、行 max/Σe、R 除法（PC 验证）
              · QK^T/av 留在 pe_array 相位0
ddim_unit（独立小型，步边界运行）:
              · 4~8 mult18 + ALU54 + 系数寄存器，~15μs/步
              · x_to_pixel 在显示通路前端（移位减法）
```

边界约定建议：

- **pe_array**：唯一的多拍 SIMD 引擎，相位由层描述符 op 字段切换；相位切换点 = 层边界（与权重重载同点）；INT32/INT48 acc 模式、SIGNEDA/SIGNEDB、桶形对齐均为描述符静态参数。
- **elementwise（gn_unit 为主）**：不设独立 DSP 阵列（§5.2 方案 A）；GN 统计/归一两相之间的组同步屏障是唯一新增控制点；requant48 与 GN/SiLU 共享同一"2 MULT + 1 ALU + fabric 移位饱和"模板（§4）。
- **softmax_unit**：与 pe_array 解耦（避免 256 token 行处理打断卷积相位），自持 BRAM；QK^T/av 不进本单元（0.14% 复用主阵列，§5.4）。
- **ddim_unit**：步级串行、一次一图，独立成块以避免与逐元素相位争用；与显示通路共享 x_final。
- **对齐流程**：按 golden README L38-50 七步（conv_in 冒烟 → GN 冒烟 → LUT 比对 → 单 resnet/attention → 全网单步 → 端到端 → 双档切换），每步以 `layers/50/step####/` 的 `.in/.out` 逐位对齐；eps 以 INT8 基准格报告（spec §9.3/L354-355）。

---

## 10. 算法级简化评估（PM 复核，2026-09-27）

目标：在不破坏位真冻结基线（M0 已签字）的前提下，评估契约 v1.4 是否还有
"为 SA5T-200 再省一刀"的空间。结论：**不建议任何契约级改动**——逐元素
开销仅占 1.1%（§3.3），所有候选简化的收益都在噪声里，而每一项都触发
"改契约 → 重生成黄金向量 → 重签字"的完整验证链。逐条记录在案：

| 候选 | 理论收益 | 否决理由 |
|---|---|---|
| requant M 从 31-bit 归一降到 16-bit（requant48 部分积 6→3、requant32 4→2） | 省 ~3.5ms/步（0.2%） | M 降位引入每通道 ~2^-15 系统性增益误差（非随机噪声，比照 FiLM 系统差教训需重新评估）；收益不值得重启验证链 |
| GN 方差先截断到 int8 等价再平方（省 Σcode² 的 18×18→9×9） | 省 ~0.5ms/步 | 破坏位真；§4.2 的代数展开已把成本压到 1 mult18/元素，无进一步空间 |
| fine_div 1024→256（内部码 18→15 位） | DDR 打包 2.25→2 B/元素（带宽 15%→13%） | 乘法口仍是 mult18，DSP 零节省；细格是坍缩修复的一部分，不重新打开 |
| Winograd F(2,3) 卷积（MAC ×0.44） | 卷积 229G→~100G/步（理论） | Winograd 变换含 ÷2 因子，整数域不保位真；与"黄金向量逐位对齐"主策略根本冲突，一票否决 |
| conv_in/conv_out INT16→INT8 | 0.2% MAC + 14KB 权重 | F4 混合精度是签过字的画质决策，不动 |
| QK^T/av 双密度 xsMULT9 | 省 1.2ms/步 | §5.4 已否决（0.14% 体量不值独立阵列） |
| DDIM-35 中间档（~60s 贴边） | 指标口径 | 不是契约改动——DDIM 步数本来就是运行时参数（外循环由控制器给定系数表），§5.3 已列为对外口径选项 |

**唯一采纳项**（不涉及契约，属于 M2 实现建议）：逐元素相位与卷积出口的
**流水融合**——conv 排出口水线直接串 requant48→GN 统计→SiLU（同一元素
不再经 EBR 往返），可把 §3.3 的 18ms/步逐元素相位与 640MB/s 激活流量再压
约一半。作为 M2 pe_array 数据通路设计的输入记录在案，不影响本方案 Go 结论。

---

## 附录 A：算力与元素量统计方法

数据源：`artifacts/f4/export-v3/layers.json`（154 层，calib_hash `ca1541db484cee7f`）。分辨率赋值 `down_blocks.b → 256/2^b`、`up_blocks.b → 8×2^b`、mid → 8、conv_in/conv_out → 256；剔除 `time_embedding.*` 与 `time_emb_proj`（FiLM 替代）。交叉验证：黄金向量 `layers/50/step0000/shapes.json` 38 点（conv_in/out 256²、down_blocks.2 64²、down3 32²、down4 16²）；attention token 数（down4/up1 = 256、mid = 64）与 feasibility §5.2/L202 一致；skip 通道数自洽（up5.conv1 cin=256 = 128+128 等）。

结果：MAC/步 = 241.37G（conv3×3 229.06G + conv1×1 10.91G + attention linear 1.41G）；输出元素/步 = 183.6M；GN/SiLU 元素/步 = 165.9M；细格 requant48 元素/步 = 158.4M（含 av/conv_out）；表定点 requant32 = 25.2M/步（含 qkv 2.1M）；QK^T+av = 339.7M MAC/步；softmax = 270.3K 元素/步；FiLM 通道 = 9984/步。

复算脚本要点（评审可重跑）：

```python
# conv: cout*cin*kh*kw*H*W；linear(attention): cout*cin*L；QK^T/av: 2*L*L*512
# 分辨率映射经 golden shapes.json 逐点核对（见上文）
```

## 附录 B：位宽出处速查表

| 数字 | 出处 |
|---|---|
| INT8/16/18/19 饱和域 ±127/32767/131071/262143（对称 [−qmax−1, qmax]） | primitives.py:17-18；spec §0/L21 |
| 细格 INT18 ±131072，实用 ±130048 = 127×1024，fine_div=1024 | spec §4/L137-139 |
| x INT16 @ s_x=2^-12（Q3.12，±8） | primitives.py:29；spec §1/L35 |
| acc INT32 / INT48（细格层，最坏 2^46.1） | spec §1/L39、§2.1/L56-61；primitives.py:130-137 |
| requant 公式 (acc·M + 2^(N−1))>>N，M ∈ [2^30,2^31)、N ∈ [0,62] | spec §2.2/L70-75；primitives.py:40-78 |
| requant48 拆 16 位两段式 + out_shift + 2^62 预截语义 | primitives.py:81-106 |
| residual 细子格 INT19、out_shift=sub_bits=10、一次舍入 | spec §4/L166-175、§10 v1.4/L361-372 |
| GN：Σ INT64（3.4e10）、μ Q0.6、var_q16 移位 (2·sub_bits−4)、eps_q、x̂ >>（sub_bits+8）sat INT16、Gc 有符号 INT32/Bq INT32 | spec §3/L96-114；primitives.py:282-328 |
| rsqrt LUT 2048×2、11 位地址、2^8 下限、inv 移位 3+E/2 | spec §3/L117-127；primitives.py:21-27,240-277 |
| var 实测 [1.42, 6013]（x 格²） | spec §3/L132 |
| SiLU 257×INT32 基表、T 值域 ±130048、f ∈ [0,1023]、sub_bits=10 | spec §5.2/L208-219；primitives.py:208-228 |
| exp LUT 4096×u16、Δ=2^-8、Q1.14、Kexp ≤ 2^31（Qe=32）、p UINT16、R = 2^40/Σe | spec §5.3/L227-245；primitives.py:231-237,333-348 |
| av requant pre=1/65536（折进 ratio） | spec §2.2/L73、§5.3/L234 |
| DDIM A/B Q8.23（1/√α_t≈130@980）、C/D Q2.30、clip ±4096、>>23/>>30、px >>13 | spec §6/L269-287；primitives.py:353-372 |
| 744 mult18 = 372 slice、120K LE、DDR3 4.4GB/s 供给 | feasibility §3.1/L61、§3.4/L108-109；用户实板确认 |
| 48 GMAC/步、0.94s/步、DDIM-20 19s / DDIM-50 47s（口径修正对象） | feasibility §3.2/L87-92；f1-report L33 |
| 层形状 154 层 / calib_hash / 参数文件实测体积 | artifacts/f4/export-v3/{layers.json,*.bin}；§7 表 |
| 分辨率/token 验证点 | artifacts/f5/golden/layers/50/step0000/shapes.json；golden README L29-36 |
| xsMULT18 18×18、逐输入符号、四级流水、级联口 | xsMULT18.v:20-70,196-232,677-739 |
| xsALU54 54 位、a_mux/b_mux/c_mux、OP=A±B±C、r_sig 反馈、CIN、MULT9_MODE | xsALU54.v:20-41,104,1516-1605 |
| xsPREADD18 PO=A±B、SOURCEA/B_MODE、HIGHSPEED | xsPREADD18.v:21-65,579-687,1183 |
| xsMULT9 9×9→P17:0 | xsMULT9.v:20-33 |

---

*本文档由 feat/dsp-map 分支承载，仅文档变更；位宽以 `docs/bittrue-spec.md` v1.4 与 `src/catdiff/bittrue/primitives.py` 为准，两者与本文冲突时以后者为准并回改本文。*
