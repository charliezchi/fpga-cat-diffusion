# fpga-cat-diffusion

> 🚧 **立项提案评审中** —— 本仓库当前只有可行性评估与实施计划文档，尚未开始任何 RTL / 软件实现。

在智多晶（XiST）SA5T-200 FPGA 开发板上，用**纯硬件逻辑推理扩散模型**：按一下按键，屏幕上的纯噪声在几秒内一步步"凝结"成一只随机的小猫，全程不需要 PC 参与，也不训练任何模型。

## 项目亮点

- **纯硬件扩散推理**：INT8 卷积加速阵列 + 确定性 DDIM 采样，生成过程完全在片上完成
- **零训练**：直接量化 HuggingFace 现成权重（[google/ddpm-cat-256](https://huggingface.co/google/ddpm-cat-256)），PC 侧只做导出与训练后量化
- **可看的生成过程**：每个去噪步实时刷新 HDMI 画面，"噪声 → 猫"的渐进动画本身就是演示

## 硬件与工具链要求

| 项 | 要求 |
|---|---|
| FPGA 板卡 | 智多晶 SA5T-200（Seal 5000 系列：120K LUT / 744 DSP18×18 / DDR3 / HDMI TX） |
| 开发工具 | 智多晶 HqFpga（综合 / 布局布线 / 烧写）、ModelSim 或 QuestaSim（仿真） |
| PC 侧 | Python 3 + PyTorch + diffusers（仅用于模型导出与量化，不做训练） |

## 文档

- [《可行性评估与实施计划》（评审稿 v0.1）](docs/feasibility-and-plan.md) —— 完整的算力核算、零训练模型路线、系统架构、里程碑与风险清单

## 里程碑（规划）

| 阶段 | 内容 | 预估周期 |
|---|---|---|
| M0 | PC 端模型管线：画质验收、PTQ INT8 导出、位真模拟器 | 1~2 周 |
| M1 | 板级显示链路 + DDR3 打通 | 1 周 |
| M2 | 卷积引擎 RTL，端到端跑通验证模型（按键出蝴蝶） | 3~6 周 |
| M3 | 完整采样系统（按键出猫，2~5 秒/张） | 2~4 周 |
| M4 | 可选：注意力块、128/256 高分辨率档 | — |

## License

[MIT](LICENSE)

## 开发环境（M0 PC 算法管线）

```bash
uv sync                 # 安装全部依赖（含 pytest）
uv run pytest           # 运行测试
uv run python -m catdiff.baseline.cli --help   # 采样入口（F1 起可用）
```

仅 CPU 环境即可；`artifacts/` 为采样产物目录，不入库。
