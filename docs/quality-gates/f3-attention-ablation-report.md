# F3 注意力消融报告（256×256）

| 项 | 值 |
|---|---|
| 日期 | 2026-09-25 |
| 模型 | google/ddpm-cat-256 手写复刻（F2 已对齐） |
| 采样 | 手写 DDIM eta=0, 20 步, seed=100, 20 张/组 |

## 对照组

| 组 | 注意力配置 | 路径 | 可辨猫占比 |
|---|---|---|---|
| A | 完整（6 处） | artifacts/f3/full-ddim20/grid.png | 20/20（执行者初判） |
| B | 全部去除 | artifacts/f3/noattn-ddim20/grid.png | 0/20（执行者初判） |
| C（条件） | 仅保留 mid | artifacts/f3/midonly-ddim20/grid.png | 0/20（执行者初判） |

## 执行记录

- 三组均由入库 CLI 跑出：`--backend handwritten --config docs/reference/unet-config-ddpm-cat-256.json --num-samples 20 --image-size 256 --num-inference-steps 20 --seed 100`，组间唯一变量为 `--strip-attention {none,all,keep-mid}`（metadata.json 已记录 `backend` 与 `strip_attention` 字段）。
- A 组与 F1 基线（artifacts/f1/cat256-ddim20）同 seed 首样本逐像素最大差为 1（PNG uint8 取整噪声），确认手写管线对齐未破坏。
- B 组日志确认「注意力消融：all，替换 6 处」；C 组确认「keep-mid，替换 5 处」。
- B 组崩坏后按计划触发 C 组；C 组输出与 B 组非同一分布（样本间逐像素最大差 8~28），但画质与 B 组相近、同样不可辨猫，与 A 组差异显著（逐像素最大差 86~129）。
- 上表「可辨猫占比」为执行者对 grid 的初步观察，最终认定以验收人对照原图为准。

## 结论（三选一，验收人签字）

- [ ] **Go**：B 组画质可接受 → RTL 不实现注意力（M2 省整块工作量，F4 起按无注意力网络走）
- [ ] **部分 Go**：B 崩坏但 C 可接受 → RTL 仅实现 mid 块注意力（1 处）
- [x] **No-go**：C 也不可接受 → 注意力全部保留，注意力 RTL 进主线排期

验收结论补充（2026-09-25）：B、C 两组均 0/20 可辨猫（PM 复核与执行者初判一致），注意力（down_blocks.4 ×2 + mid ×1 + up_blocks.1 ×3，全部单头 512 通道）为必需组件。注意力 RTL 由 M4 可选项升为主线工作量；F4/F5 按**完整网络（含注意力）**做量化与黄金向量。

验收人签字：zhangchi（项目负责人，经会话问答确认）  日期：2026-09-25
