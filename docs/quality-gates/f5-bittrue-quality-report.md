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

<!-- TASK4_FILL: layer-align-step0.json 摘要：最差点、注意力块与下游分布、eps int8 等效格分布 -->

## GN 定点化误差预算（Task 3 关口）

<!-- TASK3_FILL: LUT 深度、var 动态范围、≤1 LSB 占比、>4 LSB 计数、决策（LUT+0 Newton 定稿 / fp32 IP 降级） -->

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
