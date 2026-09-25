# INT8 量化与导出格式契约（M0-F4 产出，M2 输入）

## 1. 量化方案
- 权重：per-(输出)channel 对称 INT8。w_q[c] = round(w[c] / s_w[c])，
  s_w[c] = max|w[c]| / 127。Conv2d 沿 dim 0，Linear 沿 dim 0。
- 激活：per-tensor 静态非对称免零点（对称）INT8。a_q = round(a / s_a)，
  s_a = P99.9(|a|) / 127，按"层 × 步组"各存一套 scale。
- 步组：DDIM-20 的 20 个推理步按顺序分 5 组（步 0-3, 4-7, 8-11, 12-15, 16-19），
  组号按推理步序号（非 timestep 值）。
- 量化点：每个"层描述符算子"的输出（conv/linear/attention-out/GN/SiLU/
  residual-add/concat/上采样/下采样/最终 conv_out）。GN 与 SiLU 的整数实现
  定义属于 F5（位真模拟器），本契约只规定其输入输出量化点。
- 累加与重整：INT8×INT8 乘、INT32 累加；requant y = round(acc × s_in × s_w[c] / s_out)，
  截断饱和到 [-128, 127]。F5 以整数/定点复现该式并定义 RTL 精确行为。

## 2. 标定策略
- 标定集：seed 200..203 共 4 个初始噪声 × DDIM-20 全部 20 步，256×256。
- 统计量：每个量化点、每个步组的 P99.9(|activation|)。
- 产物：artifacts/f4/calib-stats.json（不入库）；其内容哈希写入导出文件头。

## 3. 导出文件
- export/weights.bin：自定义二进制，头部 magic "CDW1" + 层数 + 每记录
  {name, op, shape, per-channel scale(fp32), int8 payload}；层序 = 层描述符顺序。
- export/layers.json：有序层描述符数组 {index, name, op, cin, cout, k, stride,
  pad, quant_group_scales_ref, inputs, outputs}。
- export/ddim_table.json：DDIM-20 逐步 {t, alpha_t, alpha_prev, sqrt_alpha_t,
  sqrt_one_minus_alpha_t, sqrt_alpha_prev, sqrt_one_minus_alpha_prev}（fp64 计算 fp32 存储）。
- export/film_table.bin：对 20 个离散 timestep 预计算每个 ResnetBlock2D 的
  time_emb_proj(silu(temb)) 偏置向量，布局 [20][resnet_idx][out_channels] fp32，
  resnet_idx 按层描述符顺序；运行时按推理步查表，硬件无需时间嵌入 MLP。
