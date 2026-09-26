# F5 位真画质关卡报告（M0 关口）

> 方案：v3 混合精度（契约 `docs/quant-format.md` v3）+ 位真契约
> `docs/bittrue-spec.md` v1.0。位真模拟器 `src/catdiff/bittrue/` 只消费
> `artifacts/f4/export-v3/` 导出包（含 F5 扩展 requant_params/norm_params）。
> 本报告为机器观测；**签字后 M0 关闭**。

## 批次对照（同 seed=100 逐格可比）

| 批次 | 路径 | 方案 |
|---|---|---|
| fp32 基准 | artifacts/f1/cat256-ddim50/grid.png | fp32 DDIM-50 |
| F4 签字批次 | artifacts/f4/int8-ddim50-mixed-seq/grid.png | fake-quant v3 混合精度 DDIM-50 + 顺序标定 |
| **F5 位真** | artifacts/f5/bittrue-ddim50/grid.png | 位真整数模拟 DDIM-50（本关） |
| F5 位真预览档 | artifacts/f5/bittrue-ddim20/ | 位真 DDIM-20 冒烟 4 张 |

## 机器观测（逐张 PSNR，dB）

### 位真 vs F4 签字批次（要求 ≥ 30）

<!-- TASK5_FILL: psnr_vs_f4 表（seed, dB）×20 与最小值 -->

### 位真 vs fp32 基准（对照参考）

<!-- TASK5_FILL: psnr_vs_fp32 表 ×20 与最小值 -->

## 逐层对齐（Task 4 Step 4，step 0 / t=980）

详见 `artifacts/f5/layer-align-step0.json`。摘要：

seed=1000，t=980（50 档 step 0），全 51 表定点点：

| 判据组 | 结果 |
|---|---|
| conv_in（INT16→INT8 逐位） | max=1（精确） |
| down_blocks.0.resnets.0（首块） | ≤2 LSB 占比 1.0000 |
| conv_out eps（int8 等效格） | mean=0.491 / max=7.55，增益=0.9984，≤16 占比 1.0000 |
| 深度积累最大点 | up_blocks.1.attentions.2（mean 5.59 / max 31）；down_blocks.4.attentions.1（mean 4.60 / max 46）；up_blocks.1.resnets.2（mean 4.46 / max 28）；up_blocks.0.resnets.2（mean 4.46 / max 26）；up_blocks.1.attentions.0（mean 4.39 / max 42） |
| 结构防护（mean≤8 / max≤160） | 全部通过 |

深度积累注记：非注意力点 mean 0.3-4.4 LSB（int8 域），随深度近线性增长——
与位真内部 int8 量化器（SiLU/hidden/av 等，fakequant 内部为 float）的固有
噪声一致；注意力块无显著额外贡献。

判据（契约 §9.3，实测修订版）：conv_in 逐位精确、首块 ≤2 LSB@100%、eps 增益
0.9984（≤16 int8-LSB 占比 100%）达标；深度积累不设逐点门槛，由图像级 PSNR 仲裁。
**过程记录**：对齐曾暴露两处实现缺陷并修复——(1) 末端漏接 conv_norm_out+SiLU
（eps 增益 1.72×，采样塌缩灰图）；(2) down 块 skip 栈每块少入栈一层。修复后
微网与真实包对齐全绿。


## GN 定点化误差预算（Task 3 关口）——已定稿

- 方案：rsqrt LUT 2048×2 项（11 位地址，奇偶双表），**0 次 Newton**；
- 实测（fake-quant 轨迹 2 seeds × 5 步 × 全部 71 GN = 16.59 亿元素）：
  **≤1 LSB 占比 100.0000%**（判据 ≥99.9%）；>4 LSB 仅 20 个元素
  （1.2e-8，孤立离群，集中末端大尺寸 GN）；var 动态范围 [1.42, 6013]（x 格²）；
- 决策：**判据达标，维持纯 LUT 方案，不启用 fp32 IP 降级**（契约 §3 已回写）。


## 与 fake-quant 参考的已知系统性差异（非缺陷）

1. FiLM 表来自 fp32 时间通路（契约 §5.1，硬件无时间 MLP）；fake-quant 参考的
   时间 MLP 为 W8 量化。差异量级见逐层对齐报告。
2. 注意力内部 uint8 概率 + int8 V 为契约设计（fake-quant 内部 float），
   是注意力块及其下游差异的主导项（契约 §9.3）。
3. 内部点 P99.95 标定允许 0.05% 离群截断（±128 饱和），fake-quant 内部不截断。

## 人工验收（签字后 M0 关闭）

- [ ] 位真网格可辨猫占比：__/20（fp32 对照 14/20、F4 签字批次 14/20）
- [ ] 与 F4 签字批次的差异：＿＿＿＿（无可见差异 / 略退化可接受 / 明显退化）
- [ ] 结论：
  - [ ] **Go**：位真模拟器与黄金向量冻结，M0 关闭 → 进入 M1/M2
  - [ ] **No-go**：＿＿＿＿（触发修订：＿＿＿＿）

验收人签字：____  日期：____
