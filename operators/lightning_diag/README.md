# Lightning 块内对角注意力：Ascend 优化与验证

实现 `BLOCK=256` 的块内因果注意力前向。对每个 batch、head 和 token `i`：

\[
O_i=\sum_{j=256\lfloor i/256\rfloor}^{i}(Q_i\cdot K_j)
\exp[-s_h(i-j)]V_j.
\]

输出包含同一 256-token 块内的贡献；完整 Lightning Attention 的跨块历史项由其他算子计算。该算子使用指数衰减和因果掩码，输出没有 Softmax 归一化。

## 使用

在仓库根目录、已具备 CANN/TorchNPU/Triton-Ascend 的服务器上执行：

```bash
source operators/lightning_diag/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0

# 固定 16 组工作负载：精度检查 + 与原版 vLLM 内核比较
python3.11 -m operators.lightning_diag.benchmark \
  --device npu:0 --cases all --variants optimized \
  --output operators/lightning_diag/results/reproduced.json

# 独立语义、边界与真实 NPU 测试
LIGHTNING_RUN_NPU_TESTS=1 python3.11 -m unittest discover \
  -s operators/lightning_diag/tests -v
```

本次服务器的实验副本位于 `/opt/lightning-diag-lab/src`。`env.sh` 在当前 shell 中加载独立 Python、GCC 运行库和 CANN；可通过 `LIGHTNING_PYTHON_ROOT`、`LIGHTNING_GCC_ROOT`、`LIGHTNING_CANN_ROOT` 指定其他安装路径。使用前通过 `npu-smi info` 选择空闲设备。

在 Python 中调用：

```python
from operators.lightning_diag.kernel import run, baseline

out = run(q, k, v, slopes)
reference_kernel_out = baseline(q, k, v, slopes)  # 原版内核，CBLOCK=32
```

## 输入与数值约定

| 项目 | 约定 |
| --- | --- |
| Q / K | `[B, H, N, D]` |
| V / 输出 | `[B, H, N, E]` |
| 衰减率 | `[H]`，FP32、有限且非负 |
| Q/K/V dtype | 相同的 BF16、FP16 或 FP32 |
| D / E | 各自为 `32`、`64` 或 `128`，允许 `D != E` |
| 布局 | 同一设备上的连续张量；可选 `out` 使用独立存储 |
| 序列长度 | 支持短序列、尾块和空序列；B/H 为正 |
| 累加 | QK 结果、衰减、PV 和输出累加保持 FP32，最终写回输入 dtype |

精度由独立 NumPy FP64 公式计算后舍入到输出 dtype，对所有元素执行 `abs(actual-reference) <= atol + rtol*abs(reference)`：

| dtype | atol | rtol |
| --- | ---: | ---: |
| BF16 | 0.03125 | 0.015625 |
| FP16 | 0.01 | 0.002 |
| FP32 | 0.002 | 0.0001 |

计时前检查有限值、全部元素误差、输出 shape/dtype/device、确定性及输入未被修改。失败的版本不计入性能结果。

## 本仓库的优化贡献

vLLM 的原始对角内核完整保存在 `third_party/vllm/`；`baseline_kernel.py` 逐字提取原函数，测试会核对函数内容和 SHA-256。优化内核在 `kernel.py` 中独立实现。

1. **按计算阶段选择精度。** BF16/FP16 输入在 QK 阶段使用原生输入类型并进行 FP32 累加；衰减后的 score 和 PV 仍使用 FP32。独立探针验证了 QK 路径，同时发现“双 BF16 分解 PV”在当前后端明显放大了数值误差，因此最终路径保留 FP32 PV。
2. **调整 V 的加载时机。** 完成 score 和衰减后再加载 V，缩短其与 Q/K/score 的重叠存活区间。这与原生输入 QK 共同使常见 64 维路径能够使用 128 行分块；同尺寸的提前加载负对照触发 UB 溢出。
3. **裁剪无效尾块任务。** 每个 head 发射 `ceil(N / TILE)` 个程序，避免原始固定 256 块布局对短序列和尾块发射整组无效 query 程序。每个输出位置仍有唯一写入者，块间状态互不混合。
4. **根据形状和 dtype 分派。** 分派只读取元数据：短序列、FP32 大维度使用较小分块，常见 BF16/FP16 64 维完整块使用 128 分块，大维度使用经过验证的保守路径。FP32、D/E 不超过 64 且 N 为 256 整倍数时，复用未修改的 vLLM 64 分块内核；报告以 `engine` 字段明确标识该路径。较小分块保留因果工作量交错排序开关。

该组合体现了针对 Ascend 缓冲容量、数据格式和尾块开销的实现改动。数学定义沿用 Lightning Attention；固定分块参数变化单独作为控制组测量。

矩形归并、局部指数分解、提前/后置 V 加载等配置保留为消融入口，测量结果决定默认路径。部分显式实验配置可能超过目标设备的编译资源上限，默认分派使用已验证配置。

## 验证与性能记录

完整结果、环境版本、逐例耗时、消融和原始数据见 [验证报告](results/REPORT.md)。

计时使用 NPU 图回放和设备事件：预热及编译在计时外，每张图包含 16 次真实调用；每个测量点回放 4 次，每组用例执行 7 轮 ABBA/BAAB，分别报告两实现 14 个测量点的中位数。输出缓冲提前分配，两个实现使用同一批输入。

```bash
# 用未改动的原版内核，复现报告中的全部 14 组 64 分块控制组
python3.11 -m operators.lightning_diag.benchmark \
  --device npu:0 --baseline-tile 64 \
  --cases small,batch2,two_blocks,prefill1k,prefill2k,prefill4k,tail300,tail500,batch3,fp32,short16,zero_decay,strong_decay,long8k \
  --variants optimized \
  --output operators/lightning_diag/results/control64_reproduced.json

# 尾块裁剪的独立消融
python3.11 -m operators.lightning_diag.benchmark \
  --device npu:0 --cases small,tail300,short16 --variants optimized,unpruned \
  --output operators/lightning_diag/results/tail_ablation_reproduced.json
```

报告中的加速比属于这些固定工作负载的单算子设备图回放测量。输入传输、FP64 参考、编译和完整模型执行不在该计时范围内。

报告和静态图由 `summarize.py` 从原始 JSON 生成，绘图另需 `matplotlib>=3.8`。在仓库根目录运行：

```bash
python3.11 -m operators.lightning_diag.summarize \
  --primary operators/lightning_diag/results/20261123-release.json \
  --control operators/lightning_diag/results/20261123-release-control64.json \
  --ablation operators/lightning_diag/results/20261123-release-tail-ablation.json \
  --hardware-log operators/lightning_diag/results/20261123-hardware-tests-release.log \
  --output-dir operators/lightning_diag/results
```

## 来源与许可

- [vLLM 固定版本](https://github.com/vllm-project/vllm/blob/8c1557a79c539ffe82d004d2a0c8d7b5e71159ce/vllm/model_executor/layers/lightning_attn.py)：原版 `_fwd_diag_kernel`，Apache-2.0。
- [Lightning Attention 作者实现](https://github.com/OpenNLPLab/lightning-attention/blob/2363947041390c7258e261df90d5cf305b699d95/lightning_attn/ops/triton/lightning_attn2.py)：核对块内/块间计算边界，MIT。

原文件、许可证及来源哈希保存在 [third_party](third_party/sources.json)。
