# F5 黄金向量包使用说明（给 M2）

> 位置：`artifacts/f5/golden/`；数值契约：`docs/bittrue-spec.md` v1.0；
> 生成器：`src/catdiff/bittrue/golden.py`（可重导出）。

## 包结构

```
golden/
├── manifest.json                 # 档位、步选择、量化点清单、格式说明
├── checksums.txt                 # 全文件 SHA-256（先校验再使用）
├── luts/                         # 由契约公式生成的 LUT（样例档：50 档组 0）
│   ├── silu_<输入点>__<输出点>__g0.int8.bin   # 256 项 INT8（契约 §5.2）
│   ├── exp_4096xu16.bin                       # 4096 项 UINT16，Δ=2^-8，Q1.14（§5.3）
│   └── rsqrt_even/odd.int32.bin               # 各 2048 项 UINT32（§3）
├── e2e/<tier>/                   # tier ∈ {50, 20}
│   ├── x_init.int16.bin          # 初始噪声（randn(seed=100) 量化到 s_x=2^-12）
│   ├── eps/step####.int16.bin    # 位真 eps（50 档全 50 步；20 档仅选中步）
│   ├── ddim_coeffs_q.json        # 每步 A/B（Q8.23）、C/D（Q2.30）
│   ├── x_final.int16.bin
│   └── pixels.png / pixels.uint8.bin
└── layers/<tier>/step####/
    ├── shapes.json               # 张量形状 + dtype
    └── <点>.in.bin / <点>.out.bin
```

## 步选择与体积取舍

- 50 档：step 0 / 25 / 49；20 档：step 0 / 10 / 19（`configs/bittrue.json`）。
- **50 档 step0 为全量转储**（51 个表定点全部 in/out）；其余步只存空间 ≤64×64
  的点与 conv_in/conv_out——大张量（128ch×256²=8.4MB/张）只保留一份即可覆盖
  RTL 各算子的位宽/卷积窗校验，重复步的全量转储只增加体积不增加算子覆盖。
- `<点>.in` = 该量化点的输入 codes；`<点>.out` = 输出 codes（均为定点格上的
  整数，小端）。表定点名与 `act_scales_*.json` 一致；内部点命名见契约 §4。

## M2 对齐流程（建议顺序）

1. **单算子冒烟**：`layers/ddim50/step0000/conv_in.*`——INT16×INT16→INT8 路径
   （§2.1/§2.2 requant、负值舍入、饱和）。
2. **GN 冒烟**：`down_blocks.0.resnets.0.norm1`（用其 `.in` 自查 §3 整数流：
   μ/var/rsqrt LUT/逐通道 requant）。
3. **SiLU/exp/rsqrt LUT**：与 `luts/` 二进制逐项比对（生成公式在契约，PC 端
   `src/catdiff/bittrue/primitives.py::gen_*` 为参考实现）。
4. **单 resnet / 单 attention**：`down_blocks.0.resnets.0`、
   `down_blocks.4.attentions.0`（uint8 softmax、av 1/256、residual INT16 域相加）。
5. **全 UNet 单步**：`x_init → eps/step0000`（50 档），对比全部表定点。
6. **端到端 50 步**：逐位对齐 `eps/step####`，最终 `pixels`。
7. 20 步档：验证双档表切换（`ddim_coeffs_q.json` 中的组号跳变）。

## 注意

- eps 对比以 INT16 码为准；灵敏度分析建议同时换算 INT8 基准格（1 LSB = s8）。
- `mid_block` 容器有第二次量化（契约 §4 恒等 requant 规则），勿漏。
- FiLM 偏置已并入 `down_blocks.*.resnets.*` 的 conv1（acc 域注入）；硬件无需
  时间嵌入 MLP。
