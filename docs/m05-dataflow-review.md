# M0.5 算法-数据流审查（面向并行实现）

| 文档信息 | |
|---|---|
| 版本 | v1.0（2026-09-27） |
| 状态 | M2（RTL 设计）启动前最后关口；只读分析 + 本文档，不改代码、不改契约 |
| 上游 | 契约 `docs/bittrue-spec.md` v1.4.1（冻结）；`docs/dsp-mapping.md` v1.1；`docs/feasibility-and-plan.md` v0.2；数据流真相源 `src/catdiff/bittrue/{model,graph,primitives}.py`；层形状 `artifacts/f4/export-v3/layers.json`（154 层，calib_hash `ca1541db484cee7f`）；黄金向量 `artifacts/f5/golden/` |
| 量化脚本 | `artifacts/debug/m05_census.py`（只读，复算命令与逐层数据见该脚本与其输出 `m05_report.md`；本文全部数字均可由其重现） |
| 分支 | feat/m05-review（自 main@4c2f508） |
| 平台输入 | 智多晶 SA5T-200：744 mult18 = 372 slice、120K LE、≥4×4Gbit DDR3（供给按 4.4GB/s 有效带宽）、板载 2×AR8035；器件/IP 证据见 §3.1 与 §6.1（取自 HqFpga 3.1.1 安装目录） |

---

## 0. 结论摘要

**Go——M2 可以启动。** 七项审查全部完成，未发现需要改动位真契约（v1.4.1）的问题（§7，目标为零达成）；方案 A（372 slice 全配 pe_array @200MHz）与全部既定决策在数据流层面成立。三项决策所需的前置论证齐备：

1. **GN 两遍结构定案（本审查最重要项）**：GN 的组内全局归约打破了 dsp-mapping §5.5/§10"层内细格链驻留片上、仅块边界落 DDR"的假设——71 处 GN 中 30 处（覆盖 **95.6%** 的 GN 元素）的输入张量超出器件 EBR 容量，**整层驻片上方案 (a) 不可行**；定案采用 **方案 (b)：conv 出口两遍结构**（第一遍 conv→requant48→写 DDR+统计旁路，第二遍读回→GN 归一→SiLU→卷积行缓冲）。修订后的激活流量为 **586~692 MB/步 = 0.34~0.40 GB/s（供给的 7.7%~9.1%）**，远低于 50% 门槛（2.2GB/s）（§2）。
2. **算力口径二次订正（dsp-mapping §5.1 的姊妹修正）**：up/downsampler 卷积须按**输出侧分辨率**计 MAC，dsp-mapping 的 241.4G/步 少算 6.8G——真实 **248.2G/步**（不含 QK^T/av 0.34G），每步卷积 1.67s，**DDIM-20 ≈ 35.5s**（原定案 ≈34s），仍在 20~60s 验收窗内，**不影响方案 A 定案与 DDIM-20 默认档**（§1.4、§8）。
3. **EBR 是绑定资源（新增 M1 关键实测项 + M2 设计约束）**：器件库实测 372×EBR36/744×EBR18（§3.1）；单层最大权重 4.72MB（5 个 cin=1024 的 conv1）使"整层权重双缓冲"不可能（2×4.72MB ≫ EBR），M2 必须采用 **cout 分块预取或浅窗口流式**（最坏权重流速 2.33GB/s，仍在供给内）；行缓冲 + 分块权重 + 表 + 输出级的推荐配置合计 ≈1.1MB，留 ~0.5MB 裕量（§3）。

其余要点：skip 生命周期与峰值驻留 35.3MB（§1.3）；逐算子并行化策略与两处"算法结构不利并行"的处置（GN→两遍、softmax→独立单元）（§4）；查表机制三项复核全部通过（§5）；**ipdepot 存在官方 EthMAC IP v1.3（SEAL/SEALION）+ gmii2rgmii + mdio**，GbE 路径无需软 MAC，控制面建议 RISC-V 软核（ipdepot Tiny_SoC 为官方 RISC-V SoC）（§6）。

---

## 1. 逐层数据流依赖图

### 1.1 结构口径（与真相源对齐）

- 导出包 154 层 = **120 wired conv/linear** + 6 条 av_requant 记录 + 34 个运行时不布线层（time_embedding.linear_1/2 与 32 个 time_emb_proj，FiLM 表替代，spec §5.1）。位真模型 `model.py` 的遍历规则与 `graph.py` 布线图逐块对齐，本审查以二者为真相源。
- 执行序归并为 **52 个执行单元**：conv_in、down×6 块（12 resnet + 2 attention + 5 downsampler）、mid（2 resnet + 1 attention + 容器二次量化）、up×6 块（18 resnet_cat + 3 attention + 5 upsampler）、conv_norm_out、conv_out。单元级执行序与 MAC 见附录 B；154 层逐层生产-消费表见附录 A（含每层输出点网格与消费者）。
- 总 MAC/步 = **248.5G**（conv3×3 235.85G + conv1×1 10.91G + attention linear 1.41G + QK^T/av 0.34G）；卷积输出元素 183.6M/步（与 dsp-mapping §5.1 一致）。其中 3×3 类较 dsp-mapping 的 229.06G 多 6.8G，全部来自 up/downsampler conv 的分辨率口径（须按输出侧计，订正见 §7-2）。

### 1.2 全局归约点（全部标注）

| 类别 | 处数 | 归约域 | 明细 |
|---|---|---|---|
| GroupNorm 组内全局归约 | **71** | 每组 G/32 通道 × H×W 全空间（最大单组 16×65536 = 2^20 元素）；Σ 与 Σcode² 两个累加量 | 32 resnet×2（norm1/norm2）+ 6 attention group_norm + conv_norm_out；输入分三类：表定点 INT8（21 处）、hidden INT18 细格（32 处）、concat INT18 细格（18 处），逐处清单见 `m05_report.md` |
| softmax 行归约 | **6** | 每行 L 个 score 的 max 与 Σe（L = 256×5 处、64×1 处）；scores 总元素 331,776/步 | down_blocks.4.attentions.0/1（16²，256 token）、mid_block.attentions.0（8²，64 token）、up_blocks.1.attentions.0/1/2（16²） |

> 订正：dsp-mapping §3.2/§5.1 的 softmax 元素口径 270,336（"4×65536+2×4096"）与其中自己的 token 标注（down4/up1=256、mid=64，应为 5×65536+1×4096）不符，实为 **331,776**；影响为 0（该类占每步算力 0.14%）。同处 requant32 类元素 25.2M 漏计恒等支路（约 21.6M），实为 **≈50M**；逐元素总量占卷积比例由 1.1% 升至 ~1.3%，全部结论不变。

### 1.3 跨层生命周期（残差/skip 支路与 DDR 驻留）

**块内支路（38 条，全部流式、零 DDR 驻留）**：32 条 resnet 残差支路（20 条 conv_shortcut conv + 12 条恒等 requant——down 路仅 down2.resnets.0/down4.resnets.0 通道变化需 conv_shortcut，mid 2 条为恒等，up 路 18 条全部 conv_shortcut）+ 6 条 attention 残差支路。v1.4 细子格相加（INT19）在 requant 出口完成，两支路分别在 conv2/conv_shortcut 出口与恒等 requant 通路汇合，不落 DDR。

**跨块 skip（18 条，DDR 常驻）**：down 路每块产出 2 条（resnet/attn 输出）+ 5 条 downsampler 输出 + conv_in 输出，共 18 条；up 路 6 块各消费 3 条（栈序）。逐条驻留账：

| # | 点 | ch×res | MB | 产生单元 | 消亡单元 | 驻留时长(s) |
|---|---|---|---|---|---|---|
| 0 | conv_in | 128×256² | 8.39 | 0 (conv_in) | 49 (up5.resnets.2) | **1.46** |
| 1 | down_blocks.0.resnets.0 | 128×256² | 8.39 | 1 | 48 (up5.resnets.1) | 1.12 |
| 2 | down_blocks.0.resnets.1 | 128×256² | 8.39 | 2 | 47 (up5.resnets.0) | 0.78 |
| 3 | down_blocks.0.downsamplers.0 | 128×128² | 2.10 | 3 | 45 (up4.resnets.2) | 0.65 |
| 4 | down_blocks.1.resnets.0 | 128×128² | 2.10 | 4 | 44 (up4.resnets.1) | 0.56 |
| 5 | down_blocks.1.resnets.1 | 128×128² | 2.10 | 5 | 43 (up4.resnets.0) | 0.46 |
| 6 | down_blocks.1.downsamplers.0 | 128×64² | 0.52 | 6 | 41 (up3.resnets.2) | 0.35 |
| 7 | down_blocks.2.resnets.0 | 256×64² | 1.05 | 7 | 40 (up3.resnets.1) | 0.27 |
| 8 | down_blocks.2.resnets.1 | 256×64² | 1.05 | 8 | 39 (up3.resnets.0) | 0.18 |
| 9 | down_blocks.2.downsamplers.0 | 256×32² | 0.26 | 9 | 37 (up2.resnets.2) | 0.15 |
| 10 | down_blocks.3.resnets.0 | 256×32² | 0.26 | 10 | 36 (up2.resnets.1) | 0.13 |
| 11 | down_blocks.3.resnets.1 | 256×32² | 0.26 | 11 | 35 (up2.resnets.0) | 0.10 |
| 12 | down_blocks.3.downsamplers.0 | 256×16² | 0.07 | 12 | 32 (up1.resnets.2) | 0.07 |
| 13 | down_blocks.4.attentions.0 | 512×16² | 0.13 | 14 | 30 (up1.resnets.1) | 0.05 |
| 14 | down_blocks.4.attentions.1 | 512×16² | 0.13 | 16 | 28 (up1.resnets.0) | 0.02 |
| 15 | down_blocks.4.downsamplers.0 | 512×8² | 0.03 | 17 | 26 (up0.resnets.2) | 0.02 |
| 16 | down_blocks.5.resnets.0 | 512×8² | 0.03 | 18 | 25 (up0.resnets.1) | 0.01 |
| 17 | down_blocks.5.resnets.1 | 512×8² | 0.03 | 19 | 24 (up0.resnets.0) | <0.01 |

（单元号见附录 B；驻留时长按单元 MAC @148.8GMAC/s 累计，未计逐元素相位 ~2%。）

- **峰值并发驻留 = 35.3 MB**（18 条全活），窗口 = down_blocks.5.resnets.1 完成 → up_blocks.0.resnets.0 消费首条（≈5ms，mid 块期间）。DDR 容量无压力（参数 114MB + skip 35.3MB + 激活工作区 <200MB ≪ 2GB，即便按 EVB 备用口径 5×1Gb=640MB 也充裕）。
- **conv_in（8.39MB）驻留 1.46s ≈ 全步时长的 85%**，是最长生命周期张量；它同时是 up5.resnets.2 的 concat 源（远端消费），DDR 分配时不得被激活工作区覆写。

### 1.4 逐层数据流要点（详见附录 A）

- 块内链固定为 `norm1→SiLU→conv1(+FiLM/b32)→requant48→norm2→SiLU→conv2→(conv_shortcut|恒等)→细子格相加→INT8 块出`，up 块入口多两级"INT8→细格 requant→concat"。
- 每个块出点恰好 1~3 个消费者（下一块输入、down/up sampler、远端 skip concat），无扇出超过 3 的激活张量；**qkv 是唯一三点消费点**（QK^T、V 加权），审查建议 to_q/k/v 联合单遍计算（一次 GN 归一读、一次 qkv 写，§2.3）。
- mid_block 容器二次量化（`mid_block.resnets.1→mid_block`，spec §4）在出口流式串接两级 requant，零额外 DDR 流量，位真行为不变。

---

## 2. GN 两遍结构定案（本审查最重要项）

### 2.1 问题本质

GroupNorm 的 μ/var 依赖**组内全空间归约**（32 通道 × 全部 H×W），在整组最后一个元素算完之前 inv 未知。因此 dsp-mapping §5.5"层内细格链（conv→requant48→GN→SiLU）驻留片上（EBR 行缓冲 + 空间分块流水），仅块边界落 DDR"与 §10"conv 排出口水线直接串 requant48→GN 统计→SiLU（同一元素不再经 EBR 往返）"的建议**对 GN 归一+SiLU 段不成立**：空间分块流水走到一半时统计尚不完整，归一化必须等全组数据可再次访问。可行途径只有两条：(a) 整组数据驻片上；(b) 第一遍输出落 DDR、统计完成后第二遍读回归一。

### 2.2 方案 (a) 整层驻片上——不可行（量化论证）

以"单处 GN 输入张量驻片上字节数"为判据（INT8 表定点 1B/元素、INT18 细格 2.25B/元素打包），71 处 GN 逐类需求（全表见 `m05_report.md`）：

| 网格 | 分辨率 | ch | 处数 | 单处驻片上 | 判定 vs ~1.67MB EBR |
|---|---|---|---|---|---|
| INT18 | 256 | 256 | 3（up5 concat） | 37.75 MB | ✗（超 22×） |
| INT18 | 256 | 128 | 5（down0/up5 hidden） | 18.87 MB | ✗（超 11×） |
| INT8 | 256 | 128 | 3（down0 norm1×2、conv_norm_out） | 8.39 MB | ✗（超 5×） |
| INT18 | 128 | 384/256/128 | 8 | 4.72~14.16 MB | ✗ |
| INT8 | 128 | 128 | 2 | 2.10 MB | ✗ |
| INT18 | 64 | 512/384/256 | 8 | 2.36~4.72 MB | ✗ |
| INT8 | 64 | 256/128 | 2 | 0.52~1.05 MB | ✓ |
| INT18 | 32 | 768 | 1（up2 concat） | 1.77 MB | ✗（贴边） |
| INT18/INT8 | ≤32 | ≤512 | 39 | 0.03~1.18 MB | ✓ |

**结论：71 处中 30 处不可行，覆盖 95.6% 的 GN 元素；可行边界在 32²（64² 仅个别 INT8 小张量勉强）。** 即使做"小层驻片上 + 大层落 DDR"的混合方案，也只省 ~10MB/步（1.8% 流量），不值两套机制。且器件 EBR 若非 36Kb/块（M1 待实测，§3.1），结论只会更严。**方案 (a) 否决。**

### 2.3 方案 (b) conv 出口两遍 + 统计旁路（定案）

按 GN 输入的三类来源分别定数据流（统计一律**在数据自然流经处旁路累加**，Σ/Σcode² 为 54 位 ALU/宽加法器，Σcode² 最坏 2^52 单累加器可容，dsp-mapping §4.2）：

- **A 类（21 处，输入 = 已在 DDR 的表定点 INT8 块出点：down 路 norm1、6 处 attention GN、mid norm1×2、conv_norm_out）**：统计在**生产者写 DDR 时旁路**（对同码流累加，位真无损）；归一化遍从 DDR 读一遍 INT8 →GN 归一→SiLU→直接流入卷积行缓冲，输出不落 DDR。额外流量 = 1 读 × 32.5M 元素 = **32.5MB/步**。
- **B 类（32 处，输入 = conv1 出口 requant48 的 hidden，INT18 细格）**：统计在 requant48 出口旁路（该码流本就只在此处存在）；hidden 写 DDR（2.25B/元素）→ 归一化遍读回 →SiLU→conv2 行缓冲。额外流量 = 写+读 × 59.9M × 2.25B = **269.4MB/步**。
- **C 类（18 处，up 路 norm1，输入 = concat 细格）**：concat 不物化——统计在两个 INT8 源流（x_in 与 skip，均为已在 DDR 的块出点）写 DDR 时以"到 concat 点的恒等 requant"旁路（该 requant 的 M/N 由导出包 scale 确定性推导，spec §4，静态可知）；归一化遍与 conv_shortcut 遍各自读两路 INT8 流、飞行 requant 到 concat 点（位真：同码同 M/N 必得同码）。额外流量 = 2 × 73.5M × 1B = **147.1MB/步**（若统计不旁路而走独立读，+73.5MB，见上限口径）。

attention 内部（gn 细格写+读、qkv 联合单读、scores/p 行处理、残差读）共 9.8MB/步；down 恒等/shortcut 读 23.4MB；up/down sampler 输入读 15.5MB；x/eps 边界 2.0MB。

### 2.4 带宽账修订（对 dsp-mapping §5.5 的替换）

| 项 | MB/步 |
|---|---|
| 块出点写（INT8 表定点） | 86.2 |
| A 类归一读（统计旁路） | 32.5 |
| B 类 hidden 写+读（INT18 打包 2.25B） | 269.4 |
| C 类 concat 读 ×2（统计旁路） | 147.1 |
| down 恒等/shortcut 读 | 23.4 |
| attention 内部 | 9.8 |
| up/down sampler 输入读 | 15.5 |
| x/eps 边界（INT16） | 2.0 |
| **下限合计（统计全旁路）** | **586** |
| 上限增量（A/C 类统计改独立读，+32.5+73.5） | +106 |
| **上限合计** | **692** |

**平均带宽 = 586~692MB ÷ 1.73s = 0.34~0.40 GB/s = 供给 4.4GB/s 的 7.7%~9.1%，50% 门槛（2.2GB/s）达标，裕量 5.6×。**

瞬时峰值（按执行单元归账，`m05_census.py::unit_bytes`）：除 conv_in 出口写外全部 ≤0.9GB/s——

- **conv_in 出口写 8.39MB/1.52ms = 5.5GB/s 需求 > 供给**：conv_in 成为写带宽受限层，被限流至 8.39MB÷4.4GB/s = 1.9ms（+0.4ms/步，0.02%）。M2 该层描述符按"写受限"标注即可，无需专门加速。
- **8² 相位（down5/mid/up0，~30ms/步）权重流式 2.33GB/s + 激活 <0.1GB/s ≈ 2.4GB/s（55%）**：短时突发、与 256² 激活高峰在时间上不重叠，供给仍余 45%；详见 §3.3（这是权重侧账，不属于本项激活 50% 门槛范畴，如实并列）。

**对既有文档的修订条目**（不直接改，随 M2 卷积引擎设计执行）：

1. dsp-mapping §5.5"激活 ≈640MB/s（15%）"→ 按本节替换：结构上"仅块边界落 DDR"不成立（B 类必须两遍），但流量反而降至 0.34~0.40GB/s（7.7%~9.1%），INT18 打包格式（18b→72b/4 元素）仍在 M2 描述符定稿。
2. dsp-mapping §10"唯一采纳项：流水融合"→ 限定其适用范围：requant48+统计旁路（conv 出口融合 ✓）、A/C 类归一化读→SiLU→行缓冲（流式 ✓）；**GN 归一化不可与 conv 出口融合**（组屏障），B 类 hidden 的 DDR 往返（269.4MB/步）不可消除。
3. dsp-mapping §8 风险 8（int18 激活流量逼近 50%+）→ 关闭：两遍结构下峰值远低于该假设（本节）。

### 2.5 位真性核查

两遍结构不引入任何新算子与新舍入：GN 输入码与单遍方案逐位相同（A 类同码流旁路；B 类 requant48 输出码原样落 DDR/读回；C 类飞行 requant 与模拟器 `_point_codes` 同 M/N 同码）；统计量 Σ/Σcode² 为整数精确累加（加法结合律，分组归约树不改变结果，dsp-mapping §4.2 已证）。**契约零影响。**

---

## 3. 片上缓冲预算（EBR）

### 3.1 器件 EBR 实证（新增，M1 确认项）

HqFpga 3.1.1 器件库（`C:\hqfpga\hqv3_xist_3.1.1_FT092126_win64\build\common\device\xist\seal\SA5T-200.pdev`）block 普查：**EBR18×744、EBR36×372**（2:1 配对，与 744 mult18 = 372 slice 同源核对一致：MULT18×744、PRADD18×744、ALU54×372）。按命名惯例单块 EBR36 = 36Kb，总量 ≈ **13.4Mbit ≈ 1.67MB**；安装目录无 Seal 5000 独立数据手册可复核单块容量，**列 M1 实测项**（用 EBR IP 生成器或综合报告钉死单块深度×宽度与可用块数；若 <36Kb/块，§3.3 预算按比例收紧）。

### 3.2 分项预算（方案 A，卷积引擎视角）

| 缓冲 | 尺寸公式 | 最大情形 | tile_W=128 | tile_W=256 |
|---|---|---|---|---|
| conv 输入行缓冲（3 行 × cin × tile_W） | 细格输入 2.25B/元素 | up5.conv1：cin=256 细格 | 221 KB | 442 KB |
| 权重缓冲 | 见 §3.3 分块策略 | 单层最大 4.72MB（5 处 [512,1024,3,3] conv1） | 分块后 0.39~0.79 MB×2 | 同左 |
| 输出级缓冲（2 行 × cout × tile_W） | hidden 写 2.25B / INT8 1B | conv1 hidden 写 | 74 KB | 147 KB |
| 查找表（常驻+换载） | exp 8KB + rsqrt 16KB + SiLU 双表 2KB + GN/requant 参数双缓冲 ~10KB | — | 0.04 MB | 0.04 MB |
| GN 组状态（32 组 × (μ,var,inv) + 归约树） | — | — | <0.01 MB | <0.01 MB |
| 显示帧存（可选占用） | 2×256×256×16b | — | 0.26 MB（建议走 DDR） | 同左 |
| **推荐配置合计**（tile_W=128、权重分块、帧存走 DDR） | | | **≈1.1 MB（n=12）~1.5 MB（n=8）** | |

### 3.3 tile 尺寸建议与权重分块（新增约束）

- **tile 建议行主流水（tile_W = 全行 256）为默认、tile_W=128 为压力阀**：3×3 行缓冲对水平分块只引入 1 列 halo（每行 3×cin×2.25B，KB 级重读），GN 统计跨块累加不受影响，hidden DDR 布局按 tile 行段存储即可。EBR 若紧张先收 tile_W。
- **权重必须分块预取（新发现）**：单层最大权重 4.72MB（up0.resnets.0/1/2.conv1、up1.resnets.0/1.conv1 共 5 处 cin=1024 层；次大 3.54MB），"整层双缓冲"需 9.44MB ≫ EBR。定案：**cout 分块预取**（每层按 cout 分 n 块，块间流水；缓冲 2×W/n；n≥12 → 2×0.39MB=0.79MB，或 n=8 → 1.18MB），块计算时间 ≥ 块预取时间恒成立（最坏比 53%，§5.2）；备选浅窗口流式（2×64~128KB ping-pong，最坏权重流速 2.33GB/s——8² 层 1/64 B/MAC ÷ 层时，瞬时 2.4GB/s 仍在供给内）。小层（权重 ≤300KB，共 17 个单元）维持整层双缓冲。
- **显示帧存建议走 DDR**（720p60 RGB565 读出 ≈110MB/s，占供给 2.5%），如 EBR 实测富余再回收片上。

### 3.4 M1 实测项（新增/升格）

1. EBR36 单块容量与可用块数（§3.1）——**决定 §3.3 预算成立性，升格为 M2 前必须完成的实测项**；
2. DDR3 实配（4×4Gb@1866 vs 5×1Gb@1066，feasibility 风险 5）与实测有效带宽（≥2.2GB/s 即满足本项目全部余量结论）；
3. 帧存归属（EBR vs DDR）。

---

## 4. 并行结构适配清单

### 4.1 阵列分块拓扑建议（372 slice）

- **数据流（dataflow）维度选择：输出通道 × 像素并行，权重广播、激活行缓冲流式（output-stationary）**。理由：① 契约的逐通道 requant（M[i]/N[i]）与 GN 逐通道 γ/β 都以输出通道为归属，累加器必须驻定在"输出通道×输出像素"上；② INT32/INT48 累加反馈环在单 xsALU54 内闭合（dsp-mapping §2.2 推论 1），PE = (2×mult18 + 1×ALU54) 恰好承载一个输出通道的一个部分和；③ weight-stationary 变体（权重驻 PE、激活流）对本网络无收益——每权重每步恰好用一次（batch=1），两类方案的 DDR 权重流量相同，而 output-stationary 免去激活多播网络。
- **建议拓扑：372 slice = 24 个输出通道 PE × 31 个 cin 并行加（或 12×62 / 8×93，M2 按布线定）**。每 PE 串 9 拍×⌈cin/31⌉ 完成一个输出像素的一个 cout；cin 分块间累加器不清零（ALU 反馈环），块尾 requant。权重由预取缓冲按 cin 块广播，激活由行缓冲按窗供给。
- 1×1（conv_shortcut）、linear（to_q/k/v/to_out）复用同拓扑（窗口=1、像素=L token）；QK^T/av 复用主阵列（0.14%，dsp-mapping §5.4 已定单密度）。

### 4.2 逐算子并行化策略

| 算子 | 并行维度 | 阵列/通路 | 要点 |
|---|---|---|---|
| conv3×3 | cout×24 PE × cin×31 并行，空间行流水 | pe_array 相位 0 | 行缓冲 3 行；downsample 非对称 pad (0,1,0,1) 填 0 走行缓冲填充；upsampler nearest×2 在行缓冲填充级做索引复制（×2 读放大不出 DDR），conv 在 2H² 上跑（MAC 口径已按输出侧订正，§1.4） |
| conv1×1 / linear | 同上，窗口=1 | pe_array 相位 0 | conv_shortcut 消费 concat 细格（up 路）或 INT8（down 路），两版描述符 |
| attention（to_q/k/v） | cout×像素(token) | pe_array 相位 0 | **qkv 联合单遍**（同一 GN 输出读一次，三份累加器并行），qkv int8 落 softmax 单元 BRAM |
| QK^T / av | L×L 行块 | pe_array 相位 0 | av 累加 INT32 无偏置补偿（v1.4.1 例外，54 位累加后 INT32 饱和） |
| softmax | 行串行（L≤256） | softmax_unit（独立，无 DSP） | 减 max→exp LUT（8KB 常驻）→Σe→R 除法→p·R；与阵列解耦（dsp-mapping §9.2 维持） |
| GN 统计 | 随 requant48 出口/写 DDR 旁路 | pe_array 相位 1（或写通路附加累加器） | §2.3 三类旁路；Σcode² 单 ALU54 累加（2^52），组间 CIN 级联归约 |
| GN 归一+SiLU | 逐元素流式 | pe_array 相位 2/3（2 MULT+1 ALU 模板） | x̂ 2 MULT + γ 2 MULT + SiLU 1 MULT；SiLU 257 项 INT32 双表 BRAM 换载 |
| requant48/32 | 逐元素 | pe_array 相位 1 | §4.1 两段式拆分已定案；residual out_shift=10 折叠为常数 |
| residual/concat | 逐元素 fabric | 出口级 | 19b+19b fabric 加法器；concat = 通道拼接布线 + 两级飞行 requant（§2.3 C 类） |
| DDIM 合成 | 逐元素 | ddim_unit（独立小型） | 步边界运行 ~15μs/步；x_to_pixel 移位减法在显示通路前端 |

### 4.3 "算法结构不利于并行"的层与处置

1. **GroupNorm（71 处）**——全局归约打破纯空间流水：处置 = §2 两遍结构 + 统计旁路 + 组屏障（唯一新增控制点，发生在层边界，与权重重载同点）。残余串行代价已计入流量账与 ~18ms/步逐元素相位。
2. **softmax（6 处，行内全局归约 + 真除法 R）**——256 token 行串行会打断卷积相位：处置 = 独立 softmax_unit 自持 BRAM（维持 dsp-mapping §9.2）；R = rh(2^40/Σe) 的位真除法按 §4.5 PC 穷举验证后冻结（M2 前小任务，与 §5.1 复核结论一致）。
3. 其余算子结构均规则（规则窗口、规则步长、无动态形状）；attention 单头 d=512 反而是最规整的 matmul，无专项处置。

---

## 5. LUT/算术与查表机制复核

### 5.1 (a) 查表函数算术化无收益——验证性结论（与 dsp-mapping §10 一致）

| 函数 | 现方案（契约） | 算术化替代 | 每元素 DSP·拍对比 | 结论 |
|---|---|---|---|---|
| SiLU | 257 项 INT32 基表 + 1 次插值乘（§5.2） | 多项式逼近需 ~8-12 mult/元素保 ≤1 细格 LSB | 1 vs ≥8 | **查表优 ~8×**，且插值乘是唯一代价（ΔT≤2^12 远小于 18 位口，dsp-mapping §4.4） |
| exp（softmax） | 4096×u16 BRAM 直查，0 mult（§5.3） | 多项式 ~10 mult/元素 | 0 vs ≥10 | **查表优**；270K 元素/步虽小，但 softmax_unit 无 DSP 预算 |
| rsqrt（GN） | 2048×2 项 LUT + 规格化移位，0 mult（§3） | Newton 需每迭代 2-3 mult | 0 vs ≥2（且契约已定 0 次迭代实测达标） | **查表优**；16KB BRAM 常驻 |
| softmax 倒数 R | PC 穷举验证的倒数 LUT + Newton 1-2 次（仅 1536 行/步） | 纯迭代除法 | ~5 mult/行 vs 更多 | 行数极小，非瓶颈；维持 |

**结论：四类查表函数维持 LUT 方案，算术化在 DSP 预算、精度验证链、时序三方面均无收益。** 与 dsp-mapping §10 结论一致，无契约影响。

### 5.2 (b) 按层 DMA 预取 + 双缓冲换载开销模型——确认成立，附容量条件

- **表换载**（SiLU 双表 2KB + GN Gc/Nc/Bq ≤4.6KB + requant M/N/b32 ≤4.6KB ≈ 11KB/层）：@4.4GB/s ≈ 2.5μs，vs 最小执行单元（mid attention，64 token）0.48ms——**占比 0.5%，无流水停顿**。FiLM 行（9984 通道 ×4B = 40KB）每步一次 ≈9μs，可忽略。
- **权重换载**：双缓冲判据 = 预取时间 ≤ 前层计算时间。全单元扫描（`m05_census.py`）：**最坏预取/计算比 = 53%**（8² 的 512 通道层族：2.36MB ÷ 1.01ms，比率 = 1/64 B/MAC × 148.8G/4.4G ≡ 0.53，该族一致）；其余层 ≤13%。**带宽维度无停顿**；容量维度要求 §3.3 的 cout 分块（否则 2×4.72MB 装不下）——即"按层 DMA 预取+双缓冲"成立，但**必须加"大权重层分块"条件**（对 dsp-mapping §7.1 的补充）。
- **无 DDR 运行时查找**：exp（8KB）/rsqrt（16KB）常驻 BRAM；SiLU/requant/GN 参数按层预取命中 BRAM；激活查表零 DDR 往返（§2.3 归一化遍直接流行缓冲）。DDR 仅作权重/参数/驻留仓库（§7.1 口径维持）。

### 5.3 (c) 逐元素类 DSP 化无收益——确认

逐元素总量 ~1.3% 步算力（§1.2 订正后口径）；residual 细子格加（19b+19b）超 PREADD18 位口且在 DSP 域外（fabric 加法器，dsp-mapping §6 维持）；concat 为布线；恒等 requant 复用 requant 通路。**无任何逐元素运算需要（或值得）新增 DSP 硬件。**

---

## 6. 主机接口架构（GbE 加载与上位机）

### 6.1 ipdepot / 官方 IP 库调研结论（如实记录）

依据 `C:\hqfpga\hqv3_xist_3.1.1_FT092126_win64\build\ipcreator\sup_files\ipdepot\`（HqFpga 3.1.1，2026-09-21 版）：

| IP | 状态 | 证据 |
|---|---|---|
| **EthMAC** | **存在，v1.3（2026-05-21），devices = SEAL SEALION** | `ipdepot/eth_mac/eth_mac/{EthMAC.xml,UG00031.pdf}`；可配 VLAN/Jumbo/流控/DA 过滤（4 组）、MAC 地址配置；配套文档 UG00031 |
| **gmii2rgmii** | 存在，含 seal_100_366 变体 | `ipdepot/gmii2rgmii/`（UG00064） |
| **mdio** | 存在 | `ipdepot/mdio/`（UG00061） |
| Tiny_SoC | 存在，**"RISC-V Based SoC" v1.5**，AXI0/1 + APB0-4 + UART/SPI/I2C/Timer + PLIC/CLINT，片上 RAM 4KB~512KB 可配，固件 BIN 打包 | `ipdepot/Tiny_SoC/`；devices 列 "Seal SL2S-22E SL2-25E…"，**SA5T-200 适配需在 ipcreator 中实测确认** |
| ten_gig_eth_mac/pcie/qsgmii/serdes 等 | 存在但与本项目无关 | — |

**结论：GbE MAC 有官方 IP，无需软 MAC。** 链路定为 `EthMAC（GMII 侧）↔ gmii2rgmii ↔ AR8035（RGMII）`，mdio 供 PHY 管理；UDP-only（无 TCP 栈需求，无 TCP/UDP offload IP，校验和由 RTL 简单累加或软核完成）。UG00031.pdf 为加密 PDF 未能本地解包，**接口时序细节（端口位宽、描述符/握手风格）在 M2 首周以 ipcreator 实例化 EthMAC 实测钉死**（含 1Gbps 是否需 GMII 125MHz 独立时钟域）。软 MAC 评估作废（原预案仅在无 IP 时启用：~8-10K LE，不再需要）。

### 6.2 UDP 分块可靠传输协议要点（无 TCP）

- **分层**：EthMAC 处理 MAC 帧/CRC；FPGA 侧只做 ARP（应答 + 静态缓存对端）、IPv4（校验和）、UDP（校验和可选，链路可靠即免）、应用协议。
- **应用协议"分块 + 序号 + 累积 ACK + 重传"**：
  - 数据方（上位机）把参数/种子按 ≤8KB 分块（对齐 DDR 突发），每块带 `(会话 id, 块序号, 目的地址, 长度, CRC32)`；
  - 接收方（FPGA）按序写 DDR（DMA 描述符），周期性（每 N 块或 T ms）回累积 ACK（最高连续序号 + CRC 错块位图）；
  - 发送方窗口 ~16-64 块（带宽时延积：GbE × 0.2ms LAN ≈ 25KB，窗口 128~512KB 足够打满），超时重传未确认块；
  - 会话尾用 **CRC32 全量对账**（分块 CRC 逐块确认 + 完成寄存器握手；可选整体读回比对）。
- **加载时长核算**：GbE 线速 1Gbps = 125MB/s，UDP 有效负载 ~95%（1478B/帧）≈ 118MB/s，扣除 ACK/重传余量按 80-100MB/s 计：**114MB ≈ 1.1~1.4s 纯传输，端到端（会话建立 + 对账 + 写入节流）5~10 秒**——dsp-mapping §7.1 的口径成立，单口即可，无需双口聚合（2×AR8035 双口仅作实测不足时的 M3 备选）。加载为一次性开销，与运行时 0.40GB/s 激活流量不同相位，无冲突。

### 6.3 上位机软件接口规约草案（M3 正式项的输入）

**寄存器映射（AXI-Lite/APB，控制面）**：

| 偏移 | 寄存器 | 说明 |
|---|---|---|
| 0x00 | ID/VERSION | 识别 |
| 0x04 | CTRL | bit0 start、bit1 abort、bit2 步数档切换使能 |
| 0x08 | STATUS | bit0 busy、bit1 done、bit2 param_ok（CRC 对账通过）、bit4-7 错误码 |
| 0x0C | SEED[31:0] | 随机种子（按键本地模式）或上位机下发模式选择 |
| 0x10 | STEPS | 步数档（20/30/50 → 系数表指针 0/1/2，多档表常驻 DDR，切档=切指针） |
| 0x14 | CMD_Q_BASE / 0x18 CMD_Q_LEN | 命令队列环形缓冲基址/长度（DDR） |
| 0x1C | CMD_Q_TAIL（读回） | 软核消费进度 |
| 0x20 | IMG_OUT_BASE | 回传图像缓冲基址（乒乓） |
| 0x24 | PARAM_LOAD_CTRL/STAT | 加载会话：块大小、已收序号、CRC 状态 |

**命令队列（DDR 环形，软核消费）**：`LOAD_PARAM(addr,len,crc)`、`LOAD_NOISE(addr,len)`（393KB/张，见下）、`SET_SEED(u32)`、`SET_STEPS(enum)`、`START()`、`CAPTURE_IMG()`、`READ_IMG(addr,len)`。**初始噪声两条路径**（M3 定夺，接口都预留）：① **主路径——上位机按位真口径下发 x_init 张量**（256×256×3 INT16 = 393KB/张，走 §6.2 协议，单张 ~4ms）：与黄金向量 `e2e/x_init.int16.bin` 的生成方式（PC randn(seed) 量化到 s_x）逐位同源，位真对齐链最短，且天然支持"动态种子/每张不同"；② 备选——板上 LFSR+Box-Muller（feasibility §4.3，4B 种子）实现完全离线生成，代价是与 PC 侧 randn 分布细节不完全同源（仍是合法随机图，但不再逐位对应黄金链路）。

**图像回传**：末步 x_final（256×256×3 INT16，393KB）或像素（196.6KB）由软核经 UDP 推送（同 §6.2 协议反向）；调试期不依赖 HDMI（HDMI 渐进动画并行存在）。

### 6.4 控制面"RISC-V 软核 vs 纯 RTL FSM"取舍建议

**建议：控制面用 RISC-V 软核（ipdepot Tiny_SoC 优先，PicoRV32 兜底），数据面（EthMAC↔DMA↔DDR）纯 RTL。** 理由：

1. 以太网可靠传输的协议逻辑（重传计时、乱序/位图处理、ARP 状态机、命令队列解析）在 C 里迭代以秒计，RTL FSM 每改一轮重综合（200K 器件小时级）——可行性 §5.3 的既有论证在以太网场景**加权成立**；
2. 协议角落案例（CRC 错块重传、双口聚合会话）用软件处理显著降低 M3 风险；纯 RTL FSM 仅在"加载协议冻结且永不改"前提下才划算，与本项目的调试节奏矛盾；
3. 数据面（UDP 分块 DMA 写 DDR、图像 DMA 读）保持 RTL 描述符引擎，软核只下发描述符——GbE 线速 125MB/s×2 口远低于 DDR 供给，软核性能无瓶颈；
4. STM32F103 维持仅电源管理（已定案），不参与加载协议。

资源代价：Tiny_SoC（含 RAM/外设）预计 ~5-8K LE + 若干 EBR，120K LE 预算内（与 pe_array 的 LUT 占用并列后仍留余量，风险 7 的 LUT 实测在 M2 综合后复核）。

---

## 7. 契约影响清单（目标为零）

**结论：零。位真契约 v1.4.1 无需任何改动。** 审查发现的所有偏差均为文档口径/实现层问题，逐条记录：

| # | 发现 | 性质 | 影响面 | 处置（不改契约） |
|---|---|---|---|---|
| 1 | GN 两遍结构（§2） | 实现层 | M2 卷积引擎/层描述符 | 按 §2.3 定数据流；位真性核查已过（§2.5） |
| 2 | MAC 口径 241.4G→248.2G（up/downsampler conv 须按输出侧分辨率） | dsp-mapping §5.1/§5.3 订正 | 指标口径（DDIM-20 ≈34s→≈35.5s，仍在窗内） | M2 文档沿用 248.2G；dsp-mapping 修订随下一版 |
| 3 | softmax 元素 270.3K→331.8K；requant32 25.2M→≈50M（漏计恒等支路） | dsp-mapping §3.2/§5.1 订正 | 零（0.14%/1.3% 量级类） | 随下一版订正 |
| 4 | dsp-mapping §5.5/§10 的"细格链驻片上/流水融合"适用范围 | dsp-mapping §5.5/§10 订正 | M2 数据通路设计输入 | 按 §2.4 修订条目执行 |
| 5 | EBR 权重双缓冲不可行，须 cout 分块 | dsp-mapping §7/§7.1 补充条件 | M2 预取引擎 | 按 §3.3 执行；EBR 单块容量 M1 实测 |
| 6 | conv_in 写带宽受限（+0.4ms/步） | 新知（微小） | 层描述符标注 | 写受限标注即可 |
| 7 | requant48 out_shift 折叠 N≥sub_bits+1 | 已在 dsp-mapping §4.1 有断言计划 | 导出器 | 维持（M2 导出断言） |

契约文本、LUT 参数、量化点清单、舍入/饱和语义均未触及；黄金向量无需重生成。

---

## 8. 对 M1/M2 排期的影响

- **M1 新增/升格一项关键实测**：EBR36 单块容量与可用块数（§3.4-1）——若实测显著小于 36Kb/块，§3.3 预算收紧（tile_W 收窄、权重分块加大），**不推翻任何架构决策**；DDR 实配与帧存归属维持原计划。M1 工作量不变（1 周），但该项结论是 M2 卷积引擎缓冲定稿的前置。
- **M2 范围净变化**：① GN 两遍调度（统计旁路 + 组屏障 + B 类 hidden 双遍）进层引擎描述符——这是对 feasibility §5.2 conv_path 的结构性补充，估计 **+1 周**（在 M2 4~8 周区间内消化）；② 权重 cout 分块预取 +0.5 周；③ softmax_unit 的 R 除法 PC 穷举验证为 M2 前置小任务（1-2 天）。**合计 M2 上修约 1~1.5 周**，端到端里程碑（按键出猫）不受影响——DDIM-20 指标 35.5s 仍有 40%+ 裕量；频率敏感性同步更新：150MHz 时 ≈47s（原口径 44s），仍在 20~60s 窗内。
- **M3 输入前移**：EthMAC/gmii2rgmii/mdio IP 实例化验证与 Tiny_SoC 器件适配确认建议在 M2 首周并行完成（半天级探针实验），避免 M3 才发现接口问题；§6.2 加载时长按核算为 5~10 秒（单口），M3 早期实测确认，双口聚合仅作备选。

---

## 附录 A：154 层逐层依赖表

由 `artifacts/debug/m05_census.py` 从 layers.json + graph.py 直接生成（消费列为该层输出点的全部消费者）。

### A.1 逐层表（layers.json 序）

出点网格：int8 表定点 = 块出点（DDR 驻留）；int18 细格 = 内部点（§2 两遍结构）；int19 细子格 = residual 操作数（v1.4，出口 collapse 到 int8）；qkv int8；time_embedding/time_emb_proj 为 FiLM 替代不布线。

| idx | 层 | op | 形状 | res | MAC(M) | 出元素(K) | 出点网格 | 消费者 |
|---|---|---|---|---|---|---|---|---|
| 0 | conv_in | conv2d | 128×3×3×3 | 256 | 226.5 | 8388.6 | int8←x:int16 | down_blocks.0.resnets.0<<(shortcut/identity); up_blocks.5.resnets.2.concat<<skip; down_blocks.0.resnets.0.norm1 |
| 1 | time_embedding.linear_1 | linear | 512×128 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 2 | time_embedding.linear_2 | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 3 | down_blocks.0.resnets.0.conv1 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int18 细格 | down_blocks.0.resnets.0.norm2 |
| 4 | down_blocks.0.resnets.0.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 5 | down_blocks.0.resnets.0.conv2 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int19 细子格→collapse int8 | down_blocks.0.resnets.1<<(shortcut/identity); up_blocks.5.resnets.1.concat<<skip; down_blocks.0.resnets.1.norm1 |
| 6 | down_blocks.0.resnets.1.conv1 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int18 细格 | down_blocks.0.resnets.1.norm2 |
| 7 | down_blocks.0.resnets.1.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 8 | down_blocks.0.resnets.1.conv2 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int19 细子格→collapse int8 | down_blocks.0.downsamplers.0.conv; up_blocks.5.resnets.0.concat<<skip |
| 9 | down_blocks.0.downsamplers.0.conv | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int8 表定点 | down_blocks.1.resnets.0<<(shortcut/identity); up_blocks.4.resnets.2.concat<<skip; down_blocks.1.resnets.0.norm1 |
| 10 | down_blocks.1.resnets.0.conv1 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int18 细格 | down_blocks.1.resnets.0.norm2 |
| 11 | down_blocks.1.resnets.0.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 12 | down_blocks.1.resnets.0.conv2 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int19 细子格→collapse int8 | down_blocks.1.resnets.1<<(shortcut/identity); up_blocks.4.resnets.1.concat<<skip; down_blocks.1.resnets.1.norm1 |
| 13 | down_blocks.1.resnets.1.conv1 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int18 细格 | down_blocks.1.resnets.1.norm2 |
| 14 | down_blocks.1.resnets.1.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 15 | down_blocks.1.resnets.1.conv2 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int19 细子格→collapse int8 | down_blocks.1.downsamplers.0.conv; up_blocks.4.resnets.0.concat<<skip |
| 16 | down_blocks.1.downsamplers.0.conv | conv2d | 128×128×3×3 | 64 | 604.0 | 524.3 | int8 表定点 | down_blocks.2.resnets.0.conv_shortcut; down_blocks.2.resnets.0<<(shortcut/identity); up_blocks.3.resnets.2.concat<<skip; down_blocks.2.resnets.0.norm1 |
| 17 | down_blocks.2.resnets.0.conv1 | conv2d | 256×128×3×3 | 64 | 1208.0 | 1048.6 | int18 细格 | down_blocks.2.resnets.0.norm2 |
| 18 | down_blocks.2.resnets.0.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 19 | down_blocks.2.resnets.0.conv2 | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int19 细子格→collapse int8 | down_blocks.2.resnets.1<<(shortcut/identity); up_blocks.3.resnets.1.concat<<skip; down_blocks.2.resnets.1.norm1 |
| 20 | down_blocks.2.resnets.0.conv_shortcut | conv2d | 256×128×1×1 | 64 | 134.2 | 1048.6 | int19 细子格→collapse int8 | down_blocks.2.resnets.1<<(shortcut/identity); up_blocks.3.resnets.1.concat<<skip; down_blocks.2.resnets.1.norm1 |
| 21 | down_blocks.2.resnets.1.conv1 | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int18 细格 | down_blocks.2.resnets.1.norm2 |
| 22 | down_blocks.2.resnets.1.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 23 | down_blocks.2.resnets.1.conv2 | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int19 细子格→collapse int8 | down_blocks.2.downsamplers.0.conv; up_blocks.3.resnets.0.concat<<skip |
| 24 | down_blocks.2.downsamplers.0.conv | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int8 表定点 | down_blocks.3.resnets.0<<(shortcut/identity); up_blocks.2.resnets.2.concat<<skip; down_blocks.3.resnets.0.norm1 |
| 25 | down_blocks.3.resnets.0.conv1 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int18 细格 | down_blocks.3.resnets.0.norm2 |
| 26 | down_blocks.3.resnets.0.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 27 | down_blocks.3.resnets.0.conv2 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int19 细子格→collapse int8 | down_blocks.3.resnets.1<<(shortcut/identity); up_blocks.2.resnets.1.concat<<skip; down_blocks.3.resnets.1.norm1 |
| 28 | down_blocks.3.resnets.1.conv1 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int18 细格 | down_blocks.3.resnets.1.norm2 |
| 29 | down_blocks.3.resnets.1.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 30 | down_blocks.3.resnets.1.conv2 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int19 细子格→collapse int8 | down_blocks.3.downsamplers.0.conv; up_blocks.2.resnets.0.concat<<skip |
| 31 | down_blocks.3.downsamplers.0.conv | conv2d | 256×256×3×3 | 16 | 151.0 | 65.5 | int8 表定点 | down_blocks.4.resnets.0.conv_shortcut; down_blocks.4.resnets.0<<(shortcut/identity); up_blocks.1.resnets.2.concat<<skip; down_blocks.4.resnets.0.norm1 |
| 32 | down_blocks.4.resnets.0.conv1 | conv2d | 512×256×3×3 | 16 | 302.0 | 131.1 | int18 细格 | down_blocks.4.resnets.0.norm2 |
| 33 | down_blocks.4.resnets.0.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 34 | down_blocks.4.resnets.0.conv2 | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int19 细子格→collapse int8 | down_blocks.4.attentions.0<<(residual); down_blocks.4.attentions.0.group_norm |
| 35 | down_blocks.4.resnets.0.conv_shortcut | conv2d | 512×256×1×1 | 16 | 33.6 | 131.1 | int19 细子格→collapse int8 | down_blocks.4.attentions.0<<(residual); down_blocks.4.attentions.0.group_norm |
| 36 | down_blocks.4.resnets.1.conv1 | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int18 细格 | down_blocks.4.resnets.1.norm2 |
| 37 | down_blocks.4.resnets.1.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 38 | down_blocks.4.resnets.1.conv2 | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int19 细子格→collapse int8 | down_blocks.4.attentions.1<<(residual); down_blocks.4.attentions.1.group_norm |
| 39 | down_blocks.4.attentions.0.to_q | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 40 | down_blocks.4.attentions.0.to_k | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 41 | down_blocks.4.attentions.0.to_v | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 42 | down_blocks.4.attentions.0.to_out.0 | linear | 512×512 | 16 | 67.1 | 131.1 | int19 细子格→collapse int8 | down_blocks.4.resnets.1<<(shortcut/identity); up_blocks.1.resnets.1.concat<<skip; down_blocks.4.resnets.1.norm1 |
| 43 | down_blocks.4.attentions.1.to_q | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 44 | down_blocks.4.attentions.1.to_k | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 45 | down_blocks.4.attentions.1.to_v | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 46 | down_blocks.4.attentions.1.to_out.0 | linear | 512×512 | 16 | 67.1 | 131.1 | int19 细子格→collapse int8 | down_blocks.4.downsamplers.0.conv; up_blocks.1.resnets.0.concat<<skip |
| 47 | down_blocks.4.downsamplers.0.conv | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int8 表定点 | down_blocks.5.resnets.0<<(shortcut/identity); up_blocks.0.resnets.2.concat<<skip; down_blocks.5.resnets.0.norm1 |
| 48 | down_blocks.5.resnets.0.conv1 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int18 细格 | down_blocks.5.resnets.0.norm2 |
| 49 | down_blocks.5.resnets.0.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 50 | down_blocks.5.resnets.0.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | down_blocks.5.resnets.1<<(shortcut/identity); up_blocks.0.resnets.1.concat<<skip; down_blocks.5.resnets.1.norm1 |
| 51 | down_blocks.5.resnets.1.conv1 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int18 细格 | down_blocks.5.resnets.1.norm2 |
| 52 | down_blocks.5.resnets.1.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 53 | down_blocks.5.resnets.1.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | mid_block.resnets.0<<(shortcut/identity); up_blocks.0.resnets.0.concat<<skip; mid_block.resnets.0.norm1 |
| 54 | mid_block.resnets.0.conv1 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int18 细格 | mid_block.resnets.0.norm2 |
| 55 | mid_block.resnets.0.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 56 | mid_block.resnets.0.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | mid_block.attentions.0<<(residual); mid_block.attentions.0.group_norm |
| 57 | mid_block.resnets.1.conv1 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int18 细格 | mid_block.resnets.1.norm2 |
| 58 | mid_block.resnets.1.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 59 | mid_block.resnets.1.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.resnets.0.concat<<x; mid_block<<(容器二次量化) |
| 60 | mid_block.attentions.0.to_q | linear | 512×512 | 8 | 16.8 | 32.8 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 61 | mid_block.attentions.0.to_k | linear | 512×512 | 8 | 16.8 | 32.8 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 62 | mid_block.attentions.0.to_v | linear | 512×512 | 8 | 16.8 | 32.8 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 63 | mid_block.attentions.0.to_out.0 | linear | 512×512 | 8 | 16.8 | 32.8 | int19 细子格→collapse int8 | mid_block.resnets.1<<(shortcut/identity); mid_block.resnets.1.norm1 |
| 64 | up_blocks.0.resnets.0.conv1 | conv2d | 512×1024×3×3 | 8 | 302.0 | 32.8 | int18 细格 | up_blocks.0.resnets.0.norm2 |
| 65 | up_blocks.0.resnets.0.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 66 | up_blocks.0.resnets.0.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.resnets.1.concat<<x |
| 67 | up_blocks.0.resnets.0.conv_shortcut | conv2d | 512×1024×1×1 | 8 | 33.6 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.resnets.1.concat<<x |
| 68 | up_blocks.0.resnets.1.conv1 | conv2d | 512×1024×3×3 | 8 | 302.0 | 32.8 | int18 细格 | up_blocks.0.resnets.1.norm2 |
| 69 | up_blocks.0.resnets.1.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 70 | up_blocks.0.resnets.1.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.resnets.2.concat<<x |
| 71 | up_blocks.0.resnets.1.conv_shortcut | conv2d | 512×1024×1×1 | 8 | 33.6 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.resnets.2.concat<<x |
| 72 | up_blocks.0.resnets.2.conv1 | conv2d | 512×1024×3×3 | 8 | 302.0 | 32.8 | int18 细格 | up_blocks.0.resnets.2.norm2 |
| 73 | up_blocks.0.resnets.2.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 74 | up_blocks.0.resnets.2.conv2 | conv2d | 512×512×3×3 | 8 | 151.0 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.upsamplers.0.conv |
| 75 | up_blocks.0.resnets.2.conv_shortcut | conv2d | 512×1024×1×1 | 8 | 33.6 | 32.8 | int19 细子格→collapse int8 | up_blocks.0.upsamplers.0.conv |
| 76 | up_blocks.0.upsamplers.0.conv | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int8 表定点 | up_blocks.1.resnets.0.concat<<x |
| 77 | up_blocks.1.resnets.0.conv1 | conv2d | 512×1024×3×3 | 16 | 1208.0 | 131.1 | int18 细格 | up_blocks.1.resnets.0.norm2 |
| 78 | up_blocks.1.resnets.0.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 79 | up_blocks.1.resnets.0.conv2 | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.attentions.0<<(residual); up_blocks.1.attentions.0.group_norm |
| 80 | up_blocks.1.resnets.0.conv_shortcut | conv2d | 512×1024×1×1 | 16 | 134.2 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.attentions.0<<(residual); up_blocks.1.attentions.0.group_norm |
| 81 | up_blocks.1.resnets.1.conv1 | conv2d | 512×1024×3×3 | 16 | 1208.0 | 131.1 | int18 细格 | up_blocks.1.resnets.1.norm2 |
| 82 | up_blocks.1.resnets.1.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 83 | up_blocks.1.resnets.1.conv2 | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.attentions.1<<(residual); up_blocks.1.attentions.1.group_norm |
| 84 | up_blocks.1.resnets.1.conv_shortcut | conv2d | 512×1024×1×1 | 16 | 134.2 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.attentions.1<<(residual); up_blocks.1.attentions.1.group_norm |
| 85 | up_blocks.1.resnets.2.conv1 | conv2d | 512×768×3×3 | 16 | 906.0 | 131.1 | int18 细格 | up_blocks.1.resnets.2.norm2 |
| 86 | up_blocks.1.resnets.2.time_emb_proj | linear | 512×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 87 | up_blocks.1.resnets.2.conv2 | conv2d | 512×512×3×3 | 16 | 604.0 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.attentions.2<<(residual); up_blocks.1.attentions.2.group_norm |
| 88 | up_blocks.1.resnets.2.conv_shortcut | conv2d | 512×768×1×1 | 16 | 100.7 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.attentions.2<<(residual); up_blocks.1.attentions.2.group_norm |
| 89 | up_blocks.1.attentions.0.to_q | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 90 | up_blocks.1.attentions.0.to_k | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 91 | up_blocks.1.attentions.0.to_v | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 92 | up_blocks.1.attentions.0.to_out.0 | linear | 512×512 | 16 | 67.1 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.resnets.1.concat<<x |
| 93 | up_blocks.1.attentions.1.to_q | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 94 | up_blocks.1.attentions.1.to_k | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 95 | up_blocks.1.attentions.1.to_v | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 96 | up_blocks.1.attentions.1.to_out.0 | linear | 512×512 | 16 | 67.1 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.resnets.2.concat<<x |
| 97 | up_blocks.1.attentions.2.to_q | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 98 | up_blocks.1.attentions.2.to_k | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 99 | up_blocks.1.attentions.2.to_v | linear | 512×512 | 16 | 67.1 | 131.1 | qkv int8 | QK^T / av 加权（softmax 单元） |
| 100 | up_blocks.1.attentions.2.to_out.0 | linear | 512×512 | 16 | 67.1 | 131.1 | int19 细子格→collapse int8 | up_blocks.1.upsamplers.0.conv |
| 101 | up_blocks.1.upsamplers.0.conv | conv2d | 512×512×3×3 | 32 | 2415.9 | 524.3 | int8 表定点 | up_blocks.2.resnets.0.concat<<x |
| 102 | up_blocks.2.resnets.0.conv1 | conv2d | 256×768×3×3 | 32 | 1811.9 | 262.1 | int18 细格 | up_blocks.2.resnets.0.norm2 |
| 103 | up_blocks.2.resnets.0.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 104 | up_blocks.2.resnets.0.conv2 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int19 细子格→collapse int8 | up_blocks.2.resnets.1.concat<<x |
| 105 | up_blocks.2.resnets.0.conv_shortcut | conv2d | 256×768×1×1 | 32 | 201.3 | 262.1 | int19 细子格→collapse int8 | up_blocks.2.resnets.1.concat<<x |
| 106 | up_blocks.2.resnets.1.conv1 | conv2d | 256×512×3×3 | 32 | 1208.0 | 262.1 | int18 细格 | up_blocks.2.resnets.1.norm2 |
| 107 | up_blocks.2.resnets.1.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 108 | up_blocks.2.resnets.1.conv2 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int19 细子格→collapse int8 | up_blocks.2.resnets.2.concat<<x |
| 109 | up_blocks.2.resnets.1.conv_shortcut | conv2d | 256×512×1×1 | 32 | 134.2 | 262.1 | int19 细子格→collapse int8 | up_blocks.2.resnets.2.concat<<x |
| 110 | up_blocks.2.resnets.2.conv1 | conv2d | 256×512×3×3 | 32 | 1208.0 | 262.1 | int18 细格 | up_blocks.2.resnets.2.norm2 |
| 111 | up_blocks.2.resnets.2.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 112 | up_blocks.2.resnets.2.conv2 | conv2d | 256×256×3×3 | 32 | 604.0 | 262.1 | int19 细子格→collapse int8 | up_blocks.2.upsamplers.0.conv |
| 113 | up_blocks.2.resnets.2.conv_shortcut | conv2d | 256×512×1×1 | 32 | 134.2 | 262.1 | int19 细子格→collapse int8 | up_blocks.2.upsamplers.0.conv |
| 114 | up_blocks.2.upsamplers.0.conv | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int8 表定点 | up_blocks.3.resnets.0.concat<<x |
| 115 | up_blocks.3.resnets.0.conv1 | conv2d | 256×512×3×3 | 64 | 4831.8 | 1048.6 | int18 细格 | up_blocks.3.resnets.0.norm2 |
| 116 | up_blocks.3.resnets.0.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 117 | up_blocks.3.resnets.0.conv2 | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int19 细子格→collapse int8 | up_blocks.3.resnets.1.concat<<x |
| 118 | up_blocks.3.resnets.0.conv_shortcut | conv2d | 256×512×1×1 | 64 | 536.9 | 1048.6 | int19 细子格→collapse int8 | up_blocks.3.resnets.1.concat<<x |
| 119 | up_blocks.3.resnets.1.conv1 | conv2d | 256×512×3×3 | 64 | 4831.8 | 1048.6 | int18 细格 | up_blocks.3.resnets.1.norm2 |
| 120 | up_blocks.3.resnets.1.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 121 | up_blocks.3.resnets.1.conv2 | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int19 细子格→collapse int8 | up_blocks.3.resnets.2.concat<<x |
| 122 | up_blocks.3.resnets.1.conv_shortcut | conv2d | 256×512×1×1 | 64 | 536.9 | 1048.6 | int19 细子格→collapse int8 | up_blocks.3.resnets.2.concat<<x |
| 123 | up_blocks.3.resnets.2.conv1 | conv2d | 256×384×3×3 | 64 | 3623.9 | 1048.6 | int18 细格 | up_blocks.3.resnets.2.norm2 |
| 124 | up_blocks.3.resnets.2.time_emb_proj | linear | 256×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 125 | up_blocks.3.resnets.2.conv2 | conv2d | 256×256×3×3 | 64 | 2415.9 | 1048.6 | int19 细子格→collapse int8 | up_blocks.3.upsamplers.0.conv |
| 126 | up_blocks.3.resnets.2.conv_shortcut | conv2d | 256×384×1×1 | 64 | 402.7 | 1048.6 | int19 细子格→collapse int8 | up_blocks.3.upsamplers.0.conv |
| 127 | up_blocks.3.upsamplers.0.conv | conv2d | 256×256×3×3 | 128 | 9663.7 | 4194.3 | int8 表定点 | up_blocks.4.resnets.0.concat<<x |
| 128 | up_blocks.4.resnets.0.conv1 | conv2d | 128×384×3×3 | 128 | 7247.8 | 2097.2 | int18 细格 | up_blocks.4.resnets.0.norm2 |
| 129 | up_blocks.4.resnets.0.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 130 | up_blocks.4.resnets.0.conv2 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int19 细子格→collapse int8 | up_blocks.4.resnets.1.concat<<x |
| 131 | up_blocks.4.resnets.0.conv_shortcut | conv2d | 128×384×1×1 | 128 | 805.3 | 2097.2 | int19 细子格→collapse int8 | up_blocks.4.resnets.1.concat<<x |
| 132 | up_blocks.4.resnets.1.conv1 | conv2d | 128×256×3×3 | 128 | 4831.8 | 2097.2 | int18 细格 | up_blocks.4.resnets.1.norm2 |
| 133 | up_blocks.4.resnets.1.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 134 | up_blocks.4.resnets.1.conv2 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int19 细子格→collapse int8 | up_blocks.4.resnets.2.concat<<x |
| 135 | up_blocks.4.resnets.1.conv_shortcut | conv2d | 128×256×1×1 | 128 | 536.9 | 2097.2 | int19 细子格→collapse int8 | up_blocks.4.resnets.2.concat<<x |
| 136 | up_blocks.4.resnets.2.conv1 | conv2d | 128×256×3×3 | 128 | 4831.8 | 2097.2 | int18 细格 | up_blocks.4.resnets.2.norm2 |
| 137 | up_blocks.4.resnets.2.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 138 | up_blocks.4.resnets.2.conv2 | conv2d | 128×128×3×3 | 128 | 2415.9 | 2097.2 | int19 细子格→collapse int8 | up_blocks.4.upsamplers.0.conv |
| 139 | up_blocks.4.resnets.2.conv_shortcut | conv2d | 128×256×1×1 | 128 | 536.9 | 2097.2 | int19 细子格→collapse int8 | up_blocks.4.upsamplers.0.conv |
| 140 | up_blocks.4.upsamplers.0.conv | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int8 表定点 | up_blocks.5.resnets.0.concat<<x |
| 141 | up_blocks.5.resnets.0.conv1 | conv2d | 128×256×3×3 | 256 | 19327.4 | 8388.6 | int18 细格 | up_blocks.5.resnets.0.norm2 |
| 142 | up_blocks.5.resnets.0.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 143 | up_blocks.5.resnets.0.conv2 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int19 细子格→collapse int8 | up_blocks.5.resnets.1.concat<<x |
| 144 | up_blocks.5.resnets.0.conv_shortcut | conv2d | 128×256×1×1 | 256 | 2147.5 | 8388.6 | int19 细子格→collapse int8 | up_blocks.5.resnets.1.concat<<x |
| 145 | up_blocks.5.resnets.1.conv1 | conv2d | 128×256×3×3 | 256 | 19327.4 | 8388.6 | int18 细格 | up_blocks.5.resnets.1.norm2 |
| 146 | up_blocks.5.resnets.1.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 147 | up_blocks.5.resnets.1.conv2 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int19 细子格→collapse int8 | up_blocks.5.resnets.2.concat<<x |
| 148 | up_blocks.5.resnets.1.conv_shortcut | conv2d | 128×256×1×1 | 256 | 2147.5 | 8388.6 | int19 细子格→collapse int8 | up_blocks.5.resnets.2.concat<<x |
| 149 | up_blocks.5.resnets.2.conv1 | conv2d | 128×256×3×3 | 256 | 19327.4 | 8388.6 | int18 细格 | up_blocks.5.resnets.2.norm2 |
| 150 | up_blocks.5.resnets.2.time_emb_proj | linear | 128×512 | - | 0.0 | 0.0 | FiLM 替代，运行时不布线 | - |
| 151 | up_blocks.5.resnets.2.conv2 | conv2d | 128×128×3×3 | 256 | 9663.7 | 8388.6 | int19 细子格→collapse int8 | conv_norm_out |
| 152 | up_blocks.5.resnets.2.conv_shortcut | conv2d | 128×256×1×1 | 256 | 2147.5 | 8388.6 | int19 细子格→collapse int8 | conv_norm_out |
| 153 | conv_out | conv2d | 3×128×3×3 | 256 | 226.5 | 196.6 | int16(eps) | DDIM 更新单元 |

## 附录 B：52 单元执行序与 MAC

单元 = 执行调度粒度（conv_in、逐 resnet/attention/sampler、mid 容器、conv_norm_out、conv_out）；skip 生命周期（§1.3）的单元号引用此表。

### B.1 单元级表（依赖图）

| idx | kind | name | res | MAC(M) | wired 层 |
|---|---|---|---|---|---|
| 0 | conv_in | conv_in | 256 | 226.5 | conv_in |
| 1 | resnet | down_blocks.0.resnets.0 | 256 | 19327.4 | down_blocks.0.resnets.0.conv1, down_blocks.0.resnets.0.conv2 |
| 2 | resnet | down_blocks.0.resnets.1 | 256 | 19327.4 | down_blocks.0.resnets.1.conv1, down_blocks.0.resnets.1.conv2 |
| 3 | downsample | down_blocks.0.downsamplers.0.conv | 128 | 2415.9 | down_blocks.0.downsamplers.0.conv |
| 4 | resnet | down_blocks.1.resnets.0 | 128 | 4831.8 | down_blocks.1.resnets.0.conv1, down_blocks.1.resnets.0.conv2 |
| 5 | resnet | down_blocks.1.resnets.1 | 128 | 4831.8 | down_blocks.1.resnets.1.conv1, down_blocks.1.resnets.1.conv2 |
| 6 | downsample | down_blocks.1.downsamplers.0.conv | 64 | 604.0 | down_blocks.1.downsamplers.0.conv |
| 7 | resnet | down_blocks.2.resnets.0 | 64 | 3758.1 | down_blocks.2.resnets.0.conv1, down_blocks.2.resnets.0.conv_shortcut, down_blocks.2.resnets.0.conv2 |
| 8 | resnet | down_blocks.2.resnets.1 | 64 | 4831.8 | down_blocks.2.resnets.1.conv1, down_blocks.2.resnets.1.conv2 |
| 9 | downsample | down_blocks.2.downsamplers.0.conv | 32 | 604.0 | down_blocks.2.downsamplers.0.conv |
| 10 | resnet | down_blocks.3.resnets.0 | 32 | 1208.0 | down_blocks.3.resnets.0.conv1, down_blocks.3.resnets.0.conv2 |
| 11 | resnet | down_blocks.3.resnets.1 | 32 | 1208.0 | down_blocks.3.resnets.1.conv1, down_blocks.3.resnets.1.conv2 |
| 12 | downsample | down_blocks.3.downsamplers.0.conv | 16 | 151.0 | down_blocks.3.downsamplers.0.conv |
| 13 | resnet | down_blocks.4.resnets.0 | 16 | 939.5 | down_blocks.4.resnets.0.conv1, down_blocks.4.resnets.0.conv_shortcut, down_blocks.4.resnets.0.conv2 |
| 14 | attention | down_blocks.4.attentions.0 | 16 | 335.5 | down_blocks.4.attentions.0.to_q, down_blocks.4.attentions.0.to_k, down_blocks.4.attentions.0.to_v, down_blocks.4.attentions.0.to_out.0 |
| 15 | resnet | down_blocks.4.resnets.1 | 16 | 1208.0 | down_blocks.4.resnets.1.conv1, down_blocks.4.resnets.1.conv2 |
| 16 | attention | down_blocks.4.attentions.1 | 16 | 335.5 | down_blocks.4.attentions.1.to_q, down_blocks.4.attentions.1.to_k, down_blocks.4.attentions.1.to_v, down_blocks.4.attentions.1.to_out.0 |
| 17 | downsample | down_blocks.4.downsamplers.0.conv | 8 | 151.0 | down_blocks.4.downsamplers.0.conv |
| 18 | resnet | down_blocks.5.resnets.0 | 8 | 302.0 | down_blocks.5.resnets.0.conv1, down_blocks.5.resnets.0.conv2 |
| 19 | resnet | down_blocks.5.resnets.1 | 8 | 302.0 | down_blocks.5.resnets.1.conv1, down_blocks.5.resnets.1.conv2 |
| 20 | resnet | mid_block.resnets.0 | 8 | 302.0 | mid_block.resnets.0.conv1, mid_block.resnets.0.conv2 |
| 21 | attention | mid_block.attentions.0 | 8 | 71.3 | mid_block.attentions.0.to_q, mid_block.attentions.0.to_k, mid_block.attentions.0.to_v, mid_block.attentions.0.to_out.0 |
| 22 | resnet | mid_block.resnets.1 | 8 | 302.0 | mid_block.resnets.1.conv1, mid_block.resnets.1.conv2 |
| 23 | container | mid_block | 8 | 0.0 | — |
| 24 | resnet_cat | up_blocks.0.resnets.0 | 8 | 486.5 | up_blocks.0.resnets.0.conv1, up_blocks.0.resnets.0.conv_shortcut, up_blocks.0.resnets.0.conv2 |
| 25 | resnet_cat | up_blocks.0.resnets.1 | 8 | 486.5 | up_blocks.0.resnets.1.conv1, up_blocks.0.resnets.1.conv_shortcut, up_blocks.0.resnets.1.conv2 |
| 26 | resnet_cat | up_blocks.0.resnets.2 | 8 | 486.5 | up_blocks.0.resnets.2.conv1, up_blocks.0.resnets.2.conv_shortcut, up_blocks.0.resnets.2.conv2 |
| 27 | upsample | up_blocks.0.upsamplers.0.conv | 16 | 604.0 | up_blocks.0.upsamplers.0.conv |
| 28 | resnet_cat | up_blocks.1.resnets.0 | 16 | 1946.2 | up_blocks.1.resnets.0.conv1, up_blocks.1.resnets.0.conv_shortcut, up_blocks.1.resnets.0.conv2 |
| 29 | attention | up_blocks.1.attentions.0 | 16 | 335.5 | up_blocks.1.attentions.0.to_q, up_blocks.1.attentions.0.to_k, up_blocks.1.attentions.0.to_v, up_blocks.1.attentions.0.to_out.0 |
| 30 | resnet_cat | up_blocks.1.resnets.1 | 16 | 1946.2 | up_blocks.1.resnets.1.conv1, up_blocks.1.resnets.1.conv_shortcut, up_blocks.1.resnets.1.conv2 |
| 31 | attention | up_blocks.1.attentions.1 | 16 | 335.5 | up_blocks.1.attentions.1.to_q, up_blocks.1.attentions.1.to_k, up_blocks.1.attentions.1.to_v, up_blocks.1.attentions.1.to_out.0 |
| 32 | resnet_cat | up_blocks.1.resnets.2 | 16 | 1610.6 | up_blocks.1.resnets.2.conv1, up_blocks.1.resnets.2.conv_shortcut, up_blocks.1.resnets.2.conv2 |
| 33 | attention | up_blocks.1.attentions.2 | 16 | 335.5 | up_blocks.1.attentions.2.to_q, up_blocks.1.attentions.2.to_k, up_blocks.1.attentions.2.to_v, up_blocks.1.attentions.2.to_out.0 |
| 34 | upsample | up_blocks.1.upsamplers.0.conv | 32 | 2415.9 | up_blocks.1.upsamplers.0.conv |
| 35 | resnet_cat | up_blocks.2.resnets.0 | 32 | 2617.2 | up_blocks.2.resnets.0.conv1, up_blocks.2.resnets.0.conv_shortcut, up_blocks.2.resnets.0.conv2 |
| 36 | resnet_cat | up_blocks.2.resnets.1 | 32 | 1946.2 | up_blocks.2.resnets.1.conv1, up_blocks.2.resnets.1.conv_shortcut, up_blocks.2.resnets.1.conv2 |
| 37 | resnet_cat | up_blocks.2.resnets.2 | 32 | 1946.2 | up_blocks.2.resnets.2.conv1, up_blocks.2.resnets.2.conv_shortcut, up_blocks.2.resnets.2.conv2 |
| 38 | upsample | up_blocks.2.upsamplers.0.conv | 64 | 2415.9 | up_blocks.2.upsamplers.0.conv |
| 39 | resnet_cat | up_blocks.3.resnets.0 | 64 | 7784.6 | up_blocks.3.resnets.0.conv1, up_blocks.3.resnets.0.conv_shortcut, up_blocks.3.resnets.0.conv2 |
| 40 | resnet_cat | up_blocks.3.resnets.1 | 64 | 7784.6 | up_blocks.3.resnets.1.conv1, up_blocks.3.resnets.1.conv_shortcut, up_blocks.3.resnets.1.conv2 |
| 41 | resnet_cat | up_blocks.3.resnets.2 | 64 | 6442.5 | up_blocks.3.resnets.2.conv1, up_blocks.3.resnets.2.conv_shortcut, up_blocks.3.resnets.2.conv2 |
| 42 | upsample | up_blocks.3.upsamplers.0.conv | 128 | 9663.7 | up_blocks.3.upsamplers.0.conv |
| 43 | resnet_cat | up_blocks.4.resnets.0 | 128 | 10469.0 | up_blocks.4.resnets.0.conv1, up_blocks.4.resnets.0.conv_shortcut, up_blocks.4.resnets.0.conv2 |
| 44 | resnet_cat | up_blocks.4.resnets.1 | 128 | 7784.6 | up_blocks.4.resnets.1.conv1, up_blocks.4.resnets.1.conv_shortcut, up_blocks.4.resnets.1.conv2 |
| 45 | resnet_cat | up_blocks.4.resnets.2 | 128 | 7784.6 | up_blocks.4.resnets.2.conv1, up_blocks.4.resnets.2.conv_shortcut, up_blocks.4.resnets.2.conv2 |
| 46 | upsample | up_blocks.4.upsamplers.0.conv | 256 | 9663.7 | up_blocks.4.upsamplers.0.conv |
| 47 | resnet_cat | up_blocks.5.resnets.0 | 256 | 31138.5 | up_blocks.5.resnets.0.conv1, up_blocks.5.resnets.0.conv_shortcut, up_blocks.5.resnets.0.conv2 |
| 48 | resnet_cat | up_blocks.5.resnets.1 | 256 | 31138.5 | up_blocks.5.resnets.1.conv1, up_blocks.5.resnets.1.conv_shortcut, up_blocks.5.resnets.1.conv2 |
| 49 | resnet_cat | up_blocks.5.resnets.2 | 256 | 31138.5 | up_blocks.5.resnets.2.conv1, up_blocks.5.resnets.2.conv_shortcut, up_blocks.5.resnets.2.conv2 |
| 50 | gn_out | conv_norm_out | 256 | 0.0 | — |
| 51 | conv_out | conv_out | 256 | 226.5 | conv_out |


---

*本文档由 feat/m05-review 分支承载（只读分析 + 本文档）；与 `docs/bittrue-spec.md`/`src/catdiff/bittrue/` 冲突时以后者为准。量化脚本与逐层明细在 `artifacts/debug/`（gitignore），评审可按 §0 脚本路径重跑复核。*
