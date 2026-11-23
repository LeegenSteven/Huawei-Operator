# 稀疏 Flash 注意力前向

本算子实现定长 BSND、MLA-absorb 的 token 级稀疏注意力。输入是上游索引算子选出的 token 编号，输出是各查询头对这些 token 的注意力加权和。两组公开用例、任务书扩展形状、CANN 原生对照和逐项消融见[昇腾实测报告](results/REPORT.md)。

## 调用

```python
from operators.sparse_flash.kernel import run

output = run(query, key, value, sparse_indices, scale_value, query_rope, key_rope)
```

| 输入 | 连续张量形状 | 含义 |
| --- | --- | --- |
| `query`、`query_rope` | [B, Q, H, 512]、[B, Q, H, 64] | 查询的内容与旋转位置分量 |
| `key`、`key_rope` | [B, S, 1, 512]、[B, S, 1, 64] | 各查询头共享的键与位置分量 |
| `value` | [B, S, 1, 512] | 支持独立 V，也支持与 K 共享存储 |
| `sparse_indices` | [B, Q, 1, K]，INT32 | 每行最多选择 K 个 token |
| `scale_value` | 有限 Python 标量 | 任务书使用 `1 / sqrt(576)` |
| 输出 | [B, Q, H, 512] | 与 query 的设备和 dtype 一致 |

浮点输入使用同一种 BF16 或 FP16。对每个有效下标 i，打分为 `(q·k[i] + q_rope·k_rope[i]) * scale_value`。softmax 覆盖该查询的全部有效下标，无因果掩码。重复下标按次数参与，负数及越界下标跳过，整行无效时返回零。点积使用 FP32 累加，softmax 和对 V 的归约使用 FP32，最终写回输入 dtype。

## 本仓库的实现

- **单条稀疏 decode。** QK 内核按索引直接读取 K 和 RoPE。负责首组查询头的程序同时把该块的 V 排成连续 token，后续所有头共享这份 V。K 和 V 指向同一张量时直接复用已加载的 K。在 batch 较大的稀疏 decode 中，先集中打包 K、V 和 RoPE，再为更多查询头分配较大的 Cube 分块。
- **密集 prefill 与填充较多的 decode。** 每个查询在设备上统计物理 token 的出现次数 `c[i]`。权重改写为 `c[i] * exp(score[i] - max_valid_score)`，再除以所有权重之和。按物理顺序读取原 KV 可以消去逐查询的 KV 打包，并把计算宽度从槽位数 K 改成物理长度 S。长 prefill 将八个相邻头交给一个向量程序，共读一次频次数组，每个头独立求最大值和归一化分母。每次调用都重算次数，图重放也能读取新索引。
- **短索引。** K 是不超过 128 的二次幂时，单个 Triton 内核完成索引读取、QK、softmax 和 PV。中间权重留在片上。其他路径沿查询分批复用 FP32 权重缓冲；本报告覆盖的形状每批临时张量合计最多 64 MiB，输出单独分配。

调度仅查看形状和 K/V 的存储关系。公开生成器的随机排列、有效下标位置和随机种子均无需预先知道。默认入口与消融入口见[内核源码](kernel.py)。

## 基准口径

公开题目的 `definition.json` 提供了输入生成器和 PyTorch FP32 黄金实现，仓库保留原始文件及 SHA-256。正式计时使用相同输入成对调用，并记录全部原始样本。原生 CANN 的 MLA-absorb 接口要求 K/V 共享底层张量，因此它的四组性能对照让两侧使用同一个 K/V 张量。公开实现的对照则保留独立 V。[任务书](third_party/public_examples/task.md) · [公开用例](third_party/public_examples/workload.jsonl) · [固定版本与哈希](third_party/source.json) · [CANN 开源实现](https://github.com/hicann/ops-transformer/tree/e32b1329d9e4e22bc4b478ada2b70951b08efbf8/attention/sparse_flash_attention)

## 复现

在仓库根目录、已安装 CANN 和 Triton-Ascend 的环境中运行。设备编号按实际分配填写，本次使用物理卡 1，进程内为 `npu:0`。测试会编译内核，然后校验精度、输入完整性、重复执行和图重放。服务器发布目录为 `/opt/lightning-diag-lab/sparse-flash/src`。

```bash
source operators/lightning_diag/env.sh
export ASCEND_RT_VISIBLE_DEVICES=1
op=operators.sparse_flash
r=/tmp/sparse-flash-results
python3.11 -m unittest discover -s operators/sparse_flash/tests -v
python3.11 -m "$op.benchmark" --suite native --output "$r/native.json"
python3.11 -m "$op.benchmark" --suite public --output "$r/public.json"
python3.11 -m "$op.benchmark" --suite ablation --control pack_wide --cases decode8k,decode16k,decode_batch8,odd_q37 --output "$r/ablation-pack.json"
python3.11 -m "$op.benchmark" --suite ablation --control pack_large --cases prefill1k,prefill2k --output "$r/ablation-counts.json"
python3.11 -m "$op.benchmark" --suite ablation --control pack_narrow --cases decode_batch8 --output "$r/ablation-tiles.json"
python3.11 -m "$op.benchmark" --suite ablation --control counts_single --cases prefill1k,prefill2k --output "$r/ablation-softmax.json"
```

使用 `--cases sfa-001,sfa-002` 可只运行两组公开用例。下载仓库后，装有 Matplotlib 的 Python 可直接从已保存的 JSON 重建表格和图表。

```bash
python3 -m operators.sparse_flash.report
```
