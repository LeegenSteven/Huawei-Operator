# Mamba-2 分块状态前向算子

本算子计算 Mamba-2 SSD 每个 chunk 对末状态的局部贡献，入口为 `kernel.run(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True)`。服务器上的单卡耗时、两组上游对照、精度和消融见[实测报告](results/REPORT.md)。

## 输入与输出

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `x` | [batch, T, H, P] | 每个 head 的 P 维输入 |
| `B` | [batch, T, G, N] | 同组 head 共享的 N 维状态系数，要求 H 是 G 的整数倍 |
| `dt` | [batch, H, L, C] | 每个 head、每个块内位置的时间步长 |
| `dA_cumsum` | [batch, H, L, C] | 每块内部的衰减前缀和 |
| `seq_idx` | [batch, T] 或 `None` | 可选的 int32 / int64 序列标识 |
| 输出 | [batch, L, H, P, N] | 每块结束时的局部状态矩阵 |

C 为 chunk size，L = ⌈T/C⌉。给定块 c 和 head h，令 g = ⌊h/(H/G)⌋，i 为真实 token 的块内位置，t = cC+i。权重和状态按下式计算，省略 batch 下标。

```text
w[i] = dt[h,c,i] * exp(min(dA_cumsum[h,c,C-1] - dA_cumsum[h,c,i], 0))
S[c,h] = sum_i outer(x[t,h], cast_to_x_dtype(B[t,g] * w[i]))
```

传入序列标识时，只累加与本块最后一个真实 token 标识相同的位置。该标识为负时，整块输出为零。尾块仍从第 C−1 个前缀槽位取衰减终点。后续的跨块状态传递是 Mamba-2 中单独的一步。

浮点输入支持 BF16、FP16、FP32。加权系数在乘积前按 `x.dtype` 舍入，状态以 FP32 累加。输出默认 FP32，`states_in_fp32=False` 时使用 `B.dtype`，调用方传入 `states` 时使用其 dtype。可写输出须与输入使用独立、无自重叠的存储。在对应 NPU 的设备上下文中调用。连续张量直接进入优化内核。带步长的视图使用设备内连续缓冲、上游内核和按视图写回的兼容路径。T=0 返回形状正确的空张量。

## 上游和本仓库的改动

基准取自作者仓库 `state-spaces/mamba` 的 `v2.3.2.post1`。仓库保存了原文件、Apache-2.0 许可证、9 组配置和 SHA-256。测试逐字校验 `_chunk_state_fwd_kernel` 函数。主机端在 Ascend 上显式选择配置，便于把调优、编译与正式计时分开。[固定版本源码](https://github.com/state-spaces/mamba/blob/a14b1dff0454a3bc27d9eb31355dc01e4b2490ec/mamba_ssm/ops/triton/ssd_chunk_state.py) · [来源清单](third_party/source.json)

1. **按 token 行读取 x。** 连续读取每个 token 的特征，再把片上矩阵转置给 Cube 单元。形状、步长关系和分块大小在编译时已知。对于整除的时间与特征维度，内核直接读取完整块。
2. **扩大单次归约和状态列宽。** 常见低精度路径一次处理最多 128 个 token 和 128 个状态列，减少重复读取 x、步长和前缀。FP32 输入采用较小分块以控制片上缓冲占用。
3. **同组相邻 head 复用 B。** 当组内 head 可成对、输出列块不超过 64 且任务足够多时，一次加载 B，分别计算两个 head 各自的权重、舍入和矩阵乘。每个 head 仍有独立输出。实测收益随形状而变，报告给出关闭此策略的对照。
4. **选择尾块的计算方式。** 单 token 序列或 C=1 时可直接计算外积。一般的部分块使用边界掩码。报告另测了为普通多块输入强制增加单 token 尾块分支的开销。

## 复现

在项目根目录运行。服务器上的发布副本位于 `/opt/lightning-diag-lab/mamba2/src`，设备产生的原始记录位于 `/opt/lightning-diag-lab/mamba2/results`。已发布的调优记录按用例保存全部 9 组原版配置的精度和耗时。重新调优后可以把新的 JSON 传给 `--upstream-configs`。

```bash
source operators/lightning_diag/env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
op=operators.mamba2_chunk_state
r=operators/mamba2_chunk_state/results
MAMBA_RUN_NPU_TESTS=1 python3.11 -m unittest discover -s operators/mamba2_chunk_state/tests -v
python3.11 -m "$op.tune_baseline" --output /tmp/mamba2-upstream-configs.json
python3.11 -m "$op.benchmark" --upstream-configs "$r/upstream-configs.json" --output /tmp/mamba2-primary.json
python3.11 -m "$op.benchmark" --control upstream_same_tiles --output /tmp/mamba2-same-tiles.json
```

主对照以独立输入为每个形状挑选 9 组上游候选中位数最快的一组。同分块控制则让上游使用本仓库选出的 M、N、K，并固定 `num_stages=3, num_warps=4`。两组分别与默认入口成对计时，只把通过逐元素验证的结果计入加速比。

独立 NumPy 参考以 FP64 计算权重与矩阵乘，并复现上游的中间舍入。对照内核和优化内核都需通过有限值、逐元素误差、重复写回以及输入完整性检查，同时要求两者逐元素相符。BF16 输入、FP32 输出使用 atol=0.006、rtol=0.003，FP16 使用 0.001、0.001，FP32 使用 0.0005、0.0001。BF16 输出使用 0.008、0.008。所有阈值在测量之前固定。

消融可以通过 `--control single_head`、`--control k32`、`--control masked_loads` 或 `--control rank_one_tail` 运行，并用 `--cases` 指定报告表中的用例。使用已有记录重建报告和图表需要 NumPy 和 Matplotlib，可在仓库根目录执行 `python3 -m operators.mamba2_chunk_state.report`。
