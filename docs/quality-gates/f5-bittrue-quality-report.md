# F5 位真画质关卡报告（M0 关口）

> 方案：v3 混合精度（契约 `docs/quant-format.md` v3）+ 位真契约
> `docs/bittrue-spec.md` **v1.4**。位真模拟器 `src/catdiff/bittrue/` 只消费
> `artifacts/f4/export-v3/` 导出包（含 F5 扩展 requant_params/norm_params，
> 内部点标定裕量 margin=1.5）。
> 本报告为机器观测；**签字后 M0 关闭**。

## v1.4 坍缩根因与修复（2026-09-27 定论）

v1.1-v1.3 重采样图像对比度系统性坍缩（std ~19.5 vs F4 批次 33.4，PSNR ~24 dB）。
根因实验链（E1-E5 + 逐组件消融 + 同输入逐点散度）：

1. **机理**：同输入下位真 vs fake-quant 的 eps 差异含 ~1.33% 逐步**新鲜**正交
   噪声，被去噪器反复"清理"，轨迹回归数据集均值，晚期步加速（E4 注入 1.4%
   新鲜噪声完美复现坍缩；E5 容限：0.7%→29.3 dB、0.35%→36.8 dB）。
2. **首要来源**：residual 恒等支路（无 conv_shortcut 的 x 支路）细格实现按
   INT18 饱和（±127.99 int8 等效），而深层块（down_blocks.2/3/4）支路码常态
   ±130-198 s8——削波产生 max ~70 LSB 单点误差（同输入逐点散度实验钉死，
   探针 `artifacts/debug/f5_probe.py`）。
3. **次要来源**：residual 粗格双重舍入（≤1 LSB/块）；内部点 P99.95 削波
   （贡献 ~0.37pct 噪声）。
4. **修复（契约 v1.4）**：residual 两支路 requant 到出点细子格（s8/1024）、
   **INT19 饱和（±255.99 int8 等效）**、细子格域相加、一次舍入截 INT8；
   内部点标定裕量 margin=1.5。
5. **疗效**：同输入 eps 正交噪声 1.33%→**0.61-0.71%**；step0 逐点最差
   6.4→1.3 LSB；单张轨迹预览（seed100_0000）：**PSNR 34.15 dB**（v1.3 为
   24.13 dB），std 34.9 vs F4 批次 33.4（坍缩消失）。

## 批次对照（同 seed=100 逐格可比）

| 批次 | 路径 | 方案 |
|---|---|---|
| fp32 基准 | artifacts/f1/cat256-ddim50/grid.png | fp32 DDIM-50 |
| F4 签字批次 | artifacts/f4/int8-ddim50-mixed-seq/grid.png | fake-quant v3 混合精度 DDIM-50 + 顺序标定 |
| **F5 位真** | artifacts/f5/bittrue-ddim50/grid.png | 位真整数模拟 DDIM-50（本关） |
| F5 位真预览档 | artifacts/f5/bittrue-ddim20/ | 位真 DDIM-20 冒烟 4 张 |

## 机器观测（逐张 PSNR，dB）

### 位真 vs F4 签字批次（要求 ≥ 30）

（数据：`artifacts/f5/psnr-vs-f4.json`，2026-09-27 正式批次，seed=100 × 20 张，
4 分片并行采样，全部 ≥ 30 dB 达标）

| 图 | vs F4 签字批次 (dB) |
|---|---|
| seed100_0000.png | 34.15 |
| seed100_0001.png | 36.68 |
| seed100_0002.png | 42.11 |
| seed100_0003.png | 40.90 |
| seed100_0004.png | 43.58 |
| seed100_0005.png | 43.98 |
| seed100_0006.png | 38.45 |
| seed100_0007.png | 46.46 |
| seed100_0008.png | 40.37 |
| seed100_0009.png | 35.86 |
| seed100_0010.png | 37.33 |
| seed100_0011.png | 40.15 |
| seed100_0012.png | 43.77 |
| seed100_0013.png | 39.96 |
| seed100_0014.png | 40.90 |
| seed100_0015.png | 39.46 |
| seed100_0016.png | 40.36 |
| seed100_0017.png | 36.85 |
| seed100_0018.png | 41.88 |
| seed100_0019.png | 37.25 |
| **均值 / 最小** | **40.02 / 34.15** |

### 位真 vs fp32 基准（对照参考）

（数据：`artifacts/f5/psnr-vs-fp32.json`，同批 20 张。对照参考线、无门槛；
位真与 fake-quant 同为整数定点轨迹，相对 fp32 的固有偏差量级一致——
v1.3 坍缩时期该对照均值仅 13.8 dB，v1.4 修复后回升至 19.31 dB）

| 图 | vs fp32 基准 (dB) |
|---|---|
| seed100_0000.png | 21.48 |
| seed100_0001.png | 18.70 |
| seed100_0002.png | 21.28 |
| seed100_0003.png | 15.05 |
| seed100_0004.png | 18.67 |
| seed100_0005.png | 21.16 |
| seed100_0006.png | 20.95 |
| seed100_0007.png | 18.58 |
| seed100_0008.png | 17.45 |
| seed100_0009.png | 19.76 |
| seed100_0010.png | 20.97 |
| seed100_0011.png | 21.05 |
| seed100_0012.png | 18.82 |
| seed100_0013.png | 14.40 |
| seed100_0014.png | 23.40 |
| seed100_0015.png | 21.34 |
| seed100_0016.png | 22.50 |
| seed100_0017.png | 18.90 |
| seed100_0018.png | 16.81 |
| seed100_0019.png | 14.99 |
| **均值 / 最小** | **19.31 / 14.40** |

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
2. 注意力内部 uint16 概率（v1.2）+ int8 V + int8 qkv 为契约设计
   （fake-quant 内部 float）；v1.4 消融实测对残余噪声无显著贡献。
3. 内部点 P99.95 标定 + margin=1.5 后残余削波可忽略（v1.4 实测）。

## 人工验收（签字后 M0 关闭）

- [x] 位真网格可辨猫占比：**14/20**（fp32 对照 14/20、F4 签字批次 14/20——持平）
- [x] 与 F4 签字批次的差异：**无可见差异**
- [x] 结论：
  - [x] **Go**：位真模拟器与黄金向量冻结，M0 关闭 → 进入 M1/M2
  - [ ] **No-go**：＿＿＿＿（触发修订：＿＿＿＿）

验收人签字：zhangchi  日期：2026-09-27
