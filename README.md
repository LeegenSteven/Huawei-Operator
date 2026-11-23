# Huawei-Operator

华为鲲鹏与昇腾平台的算子开发仓库。`main` 提供分支导航和服务器接入说明。算子的代码、复现命令和实测记录位于下列开发分支。

## 算子分支

三个算子均于 2026-11-23 在单张 Ascend 910B3 上完成验证。Lightning 和 Mamba-2 各通过 16 组精度校验和 10 项自动化测试。稀疏 Flash 注意力覆盖 16 个任务书形状和 1 个自建边界用例，通过 14 项自动化测试，并与 CANN 原生算子比较四种主负载。加速比为各自基线耗时除以本实现耗时，几何平均对每种形状等权。

| 算子与分支 | 主对照 | 补充对照 | 复现与数据 |
| --- | --- | --- | --- |
| Lightning 块内对角注意力前向（块长 256）<br>[codex/lightning-diag-opt][lightning-branch] | 相对 vLLM 原版 32 分块 **3.81×** | 相对原版 64 分块 **1.64×**（14 组 64 维用例） | [代码与复现][lightning-code] · [实测报告][lightning-report] |
| Mamba-2 分块状态前向<br>[codex/mamba2-chunk-state][mamba-branch] | 相对每个用例实测最快的上游配置（9 选 1）**1.51×** | 上游采用与优化版相同的矩阵分块时 **1.31×**（16 组） | [代码与复现][mamba-code] · [实测报告][mamba-report] |
| 稀疏 Flash 注意力前向（token 级 BSND）<br>[codex/sparse-flash-attention][sfa-branch] | 相对 CANN 原生，4 组几何平均 **1.03×**，总耗时比 **0.42×** | 两组公开样例相对 PyTorch FP32 参考 **8.44×** | [代码与复现][sfa-code] · [实测报告][sfa-report] |

Mamba-2 分支包含 Lightning 和 Mamba-2。需要在同一份代码中运行三个算子时，切换到 `codex/sparse-flash-attention`。

## 服务器与环境

连接服务器可查阅 [VPN、SSH 与鲲鹏 CPU 环境](docs/server.md)。在昇腾卡上复现实验时，请使用 [NPU 开发环境说明][npu-env] 中的 CANN 和 Python 3.11 环境。

[lightning-branch]: https://github.com/LeegenSteven/Huawei-Operator/tree/codex/lightning-diag-opt
[lightning-code]: https://github.com/LeegenSteven/Huawei-Operator/tree/codex/lightning-diag-opt/operators/lightning_diag
[lightning-report]: https://github.com/LeegenSteven/Huawei-Operator/blob/2c0aa90c139d79abfe01b6aba6b17bff1d8841bc/operators/lightning_diag/results/REPORT.md
[mamba-branch]: https://github.com/LeegenSteven/Huawei-Operator/tree/codex/mamba2-chunk-state
[mamba-code]: https://github.com/LeegenSteven/Huawei-Operator/tree/codex/mamba2-chunk-state/operators/mamba2_chunk_state
[mamba-report]: https://github.com/LeegenSteven/Huawei-Operator/blob/fc02d38b66ee16f225dd85f940b7d570fea627d2/operators/mamba2_chunk_state/results/REPORT.md
[npu-env]: https://github.com/LeegenSteven/Huawei-Operator/blob/codex/lightning-diag-opt/docs/server.md

[sfa-branch]: https://github.com/LeegenSteven/Huawei-Operator/tree/codex/sparse-flash-attention
[sfa-code]: https://github.com/LeegenSteven/Huawei-Operator/tree/codex/sparse-flash-attention/operators/sparse_flash
[sfa-report]: https://github.com/LeegenSteven/Huawei-Operator/blob/6da498338f4a22edcd3d835d911faa8142e79fff/operators/sparse_flash/results/REPORT.md
