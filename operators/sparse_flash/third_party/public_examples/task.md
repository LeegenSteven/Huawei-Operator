# 基础信息

- **技术标签**：算子开发
- **适配硬件**：Atlas A2 推理系列产品
- **开源仓地址**：https://github.com/Wangsong95/ascend-decoding-turbo/tree/main/ops/triton
- **CANN 版本**：HiDevLab 内置镜像对应版本
- **开发语言**：Triton(基于 Triton-Ascend)

## 任务概述

在昇腾 NPU 上基于 triton-ascend 编程语言，实现 **sparse_flash_attention**（SFA）前向算子。功能对齐 `torch_npu.npu_sparse_flash_attention` 的 **定长 BSND + MLA-absorb** 子集：按 `sparse_indices` 从 KV 中收集稀疏 token，做 scaled-dot-product attention。来自 DeepSeek-V3.2 DSA，典型接在 `lightning_indexer` 之后。

本任务**只验收定长 BSND 前向**，只返回 `attention_out`。不要求 PageAttention、TND / 变长、因果 mask、`return_softmax_lse`、`sinks`、量化版、反向。

标杆：[torch_npu.npu_sparse_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention.md)

公式（无 mask，仅按索引稀疏选择）：

$$
\mathrm{Out}
=
\mathrm{softmax}
\left(
\frac{Q\,\tilde{K}^{\top}}{\sqrt{d_k}}
\right)
\tilde{V}
$$

其中 $Q=[Q_{\mathrm{nope}},\,Q_{\mathrm{rope}}]$，$\tilde{K}=[\tilde{K}_{\mathrm{nope}},\,\tilde{K}_{\mathrm{rope}}]$，$\tilde{K}_{\mathrm{nope}}/\tilde{K}_{\mathrm{rope}}/\tilde{V}$ 由 `sparse_indices` 从 `key` / `key_rope` / `value` 按 token 下标收集。$d_k=576$（$512+64$）。

计算顺序：`按 indices gather KV → QKᵀ × scale → softmax → @V`。无因果 mask。`sparse_block_size` 固定为 **1**（token-wise，与 `lightning_indexer` 的 token 下标对齐）。

DeepSeek-V3.2 默认：$N_1=128$，$N_2=1$，$D_{\mathrm{nope}}=512$，$D_{\mathrm{rope}}=64$，$k=2048$，$\mathrm{scale}=1/\sqrt{576}$。

## 核心开发要求及验收标准

### 功能实现要求

1. 与下方黄金及标杆定长 BSND、MLA-absorb 语义对齐：每个 query token 只对 `sparse_indices` 选出的 key/value 做 attention。
2. query / key / value / query_rope / key_rope 必须支持 **bfloat16、float16**。QK、softmax、加权求和在 **float32** 中算，再 round 回输出 dtype。
3. 必须融合实现 gather + flash-attention 通路，禁止用多个独立 PyTorch 算子拼接（如 `gather` + `scaled_dot_product_attention`）作为最终交付。
4. 覆盖 decode（`S1=1`，`S2` 2K～16K）与 prefill（`S1=S2` 1K～2K）。
5. 给定相同输入，输出必须确定；不得包含 NaN 或 +Inf。`-1` 填充下标不得参与 softmax。

参考黄金（验收对照，不允许作为最终交付）：

```python
import torch

def sparse_flash_attention_golden(
    query, key, value, sparse_indices, scale_value, query_rope, key_rope
):
    """
    query:          [B, S1, N1, 512]
    key / value:    [B, S2, 1, 512]
    query_rope:     [B, S1, N1, 64]
    key_rope:       [B, S2, 1, 64]
    sparse_indices: [B, S1, 1, K]  int32，无效填 -1
    return:         attention_out [B, S1, N1, 512]
    """
    q = torch.cat([query.float(), query_rope.float()], dim=-1)  # [B, S1, N1, 576]
    k = torch.cat([key.float(), key_rope.float()], dim=-1)      # [B, S2, 1, 576]
    v = value.float()                                          # [B, S2, 1, 512]
    idx = sparse_indices.squeeze(2)                            # [B, S1, K]
    B, S1, N1, Dqk = q.shape
    _, S2, _, Dv = v.shape
    valid = (idx >= 0) & (idx < S2)
    idx_safe = idx.clamp(0, S2 - 1)
    k_flat = k.squeeze(2)                                      # [B, S2, 576]
    v_flat = v.squeeze(2)                                      # [B, S2, 512]
    k_sel = torch.gather(
        k_flat.unsqueeze(1).expand(-1, S1, -1, -1),
        2,
        idx_safe.unsqueeze(-1).expand(-1, -1, -1, Dqk),
    )
    v_sel = torch.gather(
        v_flat.unsqueeze(1).expand(-1, S1, -1, -1),
        2,
        idx_safe.unsqueeze(-1).expand(-1, -1, -1, Dv),
    )
    scores = torch.einsum("bsnd,bskd->bsnk", q, k_sel) * scale_value
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.einsum("bsnk,bskd->bsnd", attn, v_sel)
    return out.to(dtype=query.dtype)
```

### 参数说明

```python
attention_out = sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, query_rope, key_rope
)
```

| 参数名 | 输入／输出/属性 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| ------ | --------------- | ---- | -------- | -------- | ----------- |
| query | 输入 | MLA nope 部分 $Q_{\mathrm{nope}}$，连续 | BFLOAT16 / FLOAT16 | ND | [B, S1, N1, 512] |
| key | 输入 | MLA nope 部分 $K_{\mathrm{nope}}$，连续。$N_2=1$ | BFLOAT16 / FLOAT16 | ND | [B, S2, 1, 512] |
| value | 输入 | $\tilde{V}$ 源，连续。shape 与 key 一致 | BFLOAT16 / FLOAT16 | ND | [B, S2, 1, 512] |
| sparse_indices | 输入 | 每个 query token 选出的 key 下标，连续。无效填 **-1** | INT32 | ND | [B, S1, 1, K] |
| scale_value | 属性 | QK 缩放。主验收 **$1/\sqrt{576}$** | float | - | - |
| query_rope | 输入 | MLA rope 部分 $Q_{\mathrm{rope}}$，连续 | BFLOAT16 / FLOAT16 | ND | [B, S1, N1, 64] |
| key_rope | 输入 | MLA rope 部分 $K_{\mathrm{rope}}$，连续 | BFLOAT16 / FLOAT16 | ND | [B, S2, 1, 64] |
| attention_out | 输出 | 稀疏注意力结果，与 query 的 BSND 对齐、尾维为 512 | BFLOAT16 / FLOAT16 | ND | [B, S1, N1, 512] |

tiling 参数不作为接口约束。

### 约束说明

- 布局固定 **定长 BSND**。各 batch 的 `S1`、`S2` 相同，有效长度等于对应 S 维。
- 固定 MLA-absorb：$D_{\mathrm{nope}}=512$，$D_{\mathrm{rope}}=64$。沿 D 拼接后再算 QK，$\tilde{V}$ **不拼接** rope，输出尾维仍为 512。
- $N_2=1$。$N_1\in\{1,2,4,8,16,32,64,128\}$，主验收 $N_1=128$。
- query / key / value / query_rope / key_rope 的 dtype 一致，且均须连续。MLA 场景下 key 与 value 可共享同一底层张量。
- `sparse_indices` 为 token 下标（`sparse_block_size=1`），取值 $\in[0,S_2)$ 或 **-1**。每行有效值在前、无效值在后。$K$ 主验收 2048。
- 对 `sparse_indices` 选出的全部**有效** key 打分后做 softmax，不加因果 mask。`-1` 及越界下标不参与。
- 输入不含 NaN。

### 测试标准

构造输入用固定 seed。`sparse_indices` 每行从 $[0,S_2)$ 无放回抽样 $\min(K,S_2)$ 个下标，不足部分填 **-1**；也可直接使用 `lightning_indexer` 的输出。自验证报告完整、可复现。

默认：`N1=128, N2=1, D=512, Dr=64, K=2048, scale_value=1/sqrt(576)`，全部浮点输入为 bfloat16。float16 至少覆盖 decode `S2=8192` 一组。

#### 1. 主验收

| B | S1 | S2 | 对应场景 |
| - | -- | -- | -------- |
| 1 | 1 | 2048 | decode |
| 1 | 1 | 4096 | decode |
| 1 | 1 | 8192 | decode |
| 1 | 1 | 16384 | 长 decode |
| 8 | 1 | 8192 | decode batch=8 |
| 1 | 1024 | 1024 | prefill |
| 1 | 2048 | 2048 | prefill |
| 4 | 1 | 1024 | 小 batch decode |

#### 2. 泛化

| B | S1 | S2 | N1 | K | 说明 |
| - | -- | -- | -- | - | ---- |
| 1 | 1 | 8192 | 8 | 2048 | 较小头数 |
| 1 | 1 | 8192 | 32 | 2048 | |
| 1 | 1 | 8192 | 64 | 2048 | 与 indexer 头数对齐 |
| 1 | 1 | 512 | 128 | 256 | $S_2<K$，其余为 -1 |
| 1 | 1 | 8192 | 128 | 512 | 更小 topk |
| 1 | 37 | 1000 | 16 | 128 | S 不对齐 |
| 5 | 1 | 1024 | 128 | 2048 | |

### 性能要求

相对 CANN `torch_npu.npu_sparse_flash_attention`（同等定长 BSND、MLA-absorb、无 page、无 mask）整体性能需 **大于 0.5 倍**。

性能主验收（warmup 后，`N1=128, D=512, Dr=64, K=2048, scale=1/sqrt(576)`，bf16）：

| B | S1 | S2 | 备注 |
| - | -- | -- | ---- |
| 1 | 1 | 8192 | decode |
| 1 | 1 | 16384 | 长 decode |
| 1 | 2048 | 2048 | prefill |
| 8 | 1 | 8192 | decode batch=8 |

### 精度要求

算子计算精度需满足浮点计算的 L1 级别精度标准：https://gitcode.com/cann/opbase/blob/c260c8ab7a9be4823ac8f8a07c60442de9bf141e/docs/zh/ops_precision_standard/commercial_standard.md

对照为 fp32 黄金（先拼接 rope，再 gather，再 softmax）。允许 flash / online-softmax 实现，但须对黄金通过 L1。`-1` 填充下标不得进入 softmax。

### 文档规范要求

算子设计文档需根据[参考文档](https://cann-op.obs.cn-north-4.myhuaweicloud.com/triton_file/XXX%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E6%96%B9%E6%A1%88.docx)填写，内容完整、格式规范，且必须通过评审。

## 验收规则与流程

### 提交验收申请

联系昇腾小助手，提交以下**三类交付件**进行验收：

1. Triton-Ascend-Kernels 仓 fork 的个人代码仓链接（需包含：算子及其调用封装函数代码、算子 README 文档、泛化测试代码）；
2. 测试相关交付件，包含：本任务测试脚本的测试自验证报告（参考[XXX算子自验证报告](https://cann-op.obs.cn-north-4.myhuaweicloud.com/triton_file/XXX%E7%AE%97%E5%AD%90%E8%87%AA%E9%AA%8C%E8%AF%81%E6%8A%A5%E5%91%8A.xlsx)，含测试用例执行日志 / 截图、整体测试通过截图、性能数据截图）；
3. 华为评审通过的算子设计文档（按模板填写）。

### 验收结果反馈

验收以提交验收申请时的代码为准，72小时内反馈验收结果，如代码更新请重新提交验收申请，验收时间同步刷新。

### PR 申请合入

验收通过后，在 Triton-Ascend-Kernels 开源仓提交 PR 申请，申请将开发完成的算子合入（[https://gitcode.com/Ascend/triton-ascend-kernels/tree/experimental](https://gitcode.com/Ascend/triton-ascend-kernels/tree/experimental)）。

## 参考资料

1. 文档类：[Triton-Ascend文档](https://triton-ascend.readthedocs.io/zh-cn/latest/quick_start.html)
2. 课程类： [课程1](https://www.hiascend.com/developer/courses/detail/1998217338603737090)   [课程2](https://www.hiascend.com/developer/courses/detail/1988449990298521602)   [课程3](https://www.hiascend.com/developer/courses/detail/1993247708929794049)
3. 代码样例：[https://gitcode.com/Ascend/triton-ascend/tree/main/docs/zh/examples](https://gitcode.com/Ascend/triton-ascend/tree/main/docs/zh/examples)
4. 代码样例：[https://gitcode.com/Ascend/triton-ascend-kernels](https://gitcode.com/Ascend/triton-ascend-kernels)
5. 标杆：[torch_npu.npu_sparse_flash_attention](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/customapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_sparse_flash_attention.md)（本任务仅取其定长 BSND、MLA-absorb 子集）
6. 论文：https://arxiv.org/html/2512.02556v1

## 环境获取

1、优先算力池：使用AtomGit平台提供的notebook算力开发资源；使用指南：https://gitcode.com/linjianyong00111/Triton-1/blob/main/README.md

2、备选算力池：

使用[https://hidevlab.huawei.com/online-develop-intro](https://hidevlab.huawei.com/online-develop-intro)中 IDE 环境，开源仓提供 100 小时免费时长，不使用时请及时关闭，用时耗尽前请务必保存相关资料，建议定时提交备份。免费时长耗尽及其他使用问题请联系昇腾小助手。
环境的使用指南参考[https://hidevlab.huawei.com/support/userGuide?currentKey=ide](https://hidevlab.huawei.com/support/userGuide?currentKey=ide)
创建环境时请选择 A2，配置与 triton-ascend 镜像

## 特别注意事项

1. 开发过程需严格遵循 Triton-Ascend编程规范及算子开发相关要求；
2. 所有交付件需提前完成自验证，确认符合验收标准后再提交验收申请；
3. 只返回 **attention_out**，不要 `softmax_max` / `softmax_sum`；
4. 不加因果 mask；`-1` 只表示无效下标，不是因果屏蔽；
5. 固定 MLA-absorb：先把 nope 与 rope 沿 D 拼接再算 QK，不要做成带 `wkv_b` 的在线 absorb；
6. `sparse_indices` 是 token 下标，不要按 `sparse_block_size>1` 的 block 起始下标解释。
