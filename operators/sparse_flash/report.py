"""Build the Markdown tables and native-comparison figure from on-device JSON."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
NATIVE_ORDER = ("decode8k", "decode16k", "decode_batch8", "prefill2k")
LABELS = ("B1 decode 8K", "B1 decode 16K", "B8 decode 8K", "B1 prefill 2K")


def geometric(rows):
    return math.exp(sum(math.log(row["performance"]["speedup"]) for row in rows) / len(rows))


def figure(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.25), layout="constrained")
    ratios = [r["performance"]["speedup"] for r in rows]
    bars = ax.barh(LABELS, ratios, height=0.54, color=["#19746d" if v >= 1 else "#aa6434" for v in ratios])
    ax.invert_yaxis()
    ax.axvline(1, color="#454b50", lw=1, label="CANN parity")
    ax.axvline(0.5, color="#70777b", ls="--", lw=1, label="0.5× reference")
    ax.bar_label(
        bars,
        labels=[f"{v:.2f}×" for v in ratios],
        padding=4,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )
    ax.set_xlim(0, max(2.2, max(ratios) * 1.16))
    ax.set_xlabel("CANN latency / Triton latency")
    ax.set_title("Token-sparse attention on one Ascend 910B3", loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.savefig(path, dpi=155)
    plt.close(fig)


def build(directory):
    names = ("public", "native", "ablation-pack", "ablation-counts", "ablation-tiles", "ablation-softmax")
    data = {n: json.loads((directory / (n + ".json")).read_text()) for n in names}
    env = data["public"]["environment"]
    for n, report in data.items():
        if (
            not report["summary"]["all_passed"]
            or report["environment"]["source_sha256"] != env["source_sha256"]
        ):
            raise ValueError("Failed cases or inconsistent operator sources in " + n)
    for name, digest in env["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Source has changed since measurement: " + name)
    tests = (directory / "hardware-tests.log").read_text().strip()
    if not tests.endswith("OK"):
        raise ValueError("Incomplete hardware test log")
    ntests = int(re.search(r"Ran (\d+) tests", tests).group(1))
    public = data["public"]["cases"]
    by_name = {row["case"]["name"]: row for row in public}
    native = {row["case"]["name"]: row for row in data["native"]["cases"]}
    rows = [native[name] for name in NATIVE_ORDER]
    assert len(public) == 17 and len(native) == 4 and all(r["shared_value"] for r in rows)
    assert all(not r["shared_value"] for r in public)
    expanded = [r for r in public if r["case"]["name"] not in ("mixed_indices", "prefill2k")]
    two = [by_name[name] for name in ("sfa-001", "sfa-002")]
    s = data["native"]["summary"]
    speed = [row["performance"]["speedup"] for row in rows]
    above = sum(v > 0.5 for v in speed)
    relative_metrics = {
        key: max(r["fp64"]["candidate"][key] / r["fp64"]["public_reference"][key] for r in public)
        for key in ("mare", "mere", "rmse")
    }
    summary = dict(
        public_two_geomean=geometric(two),
        task_full_reference_geomean=geometric(expanded),
        whole_reference_shapes=len(expanded),
        native_geomean=s["geomean"],
        native_summed_latency_ratio=s["ratio_of_sums"],
        native_above_half=above,
        hardware_tests=ntests,
        public_accuracy_rows=len(public),
        worst_absolute_error=max(r["accuracy"]["max_abs"] for r in public),
        checked_output_elements=sum(r["accuracy"]["elements"] for r in public),
        sampled_fp64_max_error_ratios=relative_metrics,
    )
    t = [
        "# 稀疏 Flash 注意力前向实测",
        "",
        f"单张 Ascend 910B3 上，两组公开算例相对原版 PyTorch FP32 参考的几何平均加速为 **{geometric(two):.2f}×**。四组 CANN 原生对照中，单条 8K、16K decode 分别达到 **{speed[0]:.2f}×、{speed[1]:.2f}×**，batch=8 和 2K prefill 分别为 **{speed[2]:.2f}×、{speed[3]:.2f}×**。",
        "",
        "## CANN 原生对照",
        "",
        "双方传入相同的 BF16 张量、逐 token 索引和缩放，K/V 共享存储。四组均为 128 个查询头、K=2048。耗时是同轮 Ascend 事件的中位数，加速比为 CANN 耗时除以本实现耗时。",
        "",
        "| 负载 | B × Q × S | CANN µs | 本实现 µs | 相对 CANN |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row, label in zip(rows, LABELS):
        c, p = row["case"], row["performance"]
        t.append(
            f"| {label} | {c['b']} × {c['q']} × {c['s']} | {p['baseline_us']:,.2f} | {p['candidate_us']:,.2f} | {p['speedup']:.3f}× |"
        )
    t += [
        "",
        f"任务书给出整体性能大于 0.5× 的目标，未指定聚合公式。四种形状等权取几何平均为 **{s['geomean']:.3f}×**，其中 {above}/4 形状超过 0.5×。每种形状各运行一次时，两侧耗时分别求和再相除为 **{s['ratio_of_sums']:.3f}×**，2K prefill 占据主要时间。两种口径均列出供评审采用。[任务书](../third_party/public_examples/task.md) · [原始 CANN 计时](native.json)",
        "",
        "![四种负载相对 CANN 的加速比](speedup.png)",
        "",
        "## 公开 FP32 参考与独立 V",
        "",
        f"16 个任务书形状全部通过。可以整段运行原参考的 {len(expanded)} 个形状，几何平均加速 **{geometric(expanded):.2f}×**。2K prefill 的原参考曾在该卡一次申请 54 GiB 并触发 OOM。该行沿 Q 每 32 行调用未经修改的参考函数，组合出完整输出，计时包含各次调用和结果写回。另列 1 个含重复、越界和全无效行的压力用例。[公开源码与版本](../third_party/source.json) · [逐例原始记录](public.json) · [显存诊断](prefill-reference.log)",
        "",
        "| 用例 | B / Q / S / H / K | dtype | 参考 µs | 本实现 µs | 加速 | 最大绝对误差 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in public:
        c, p, a = row["case"], row["performance"], row["accuracy"]
        shape = " / ".join(str(c[n]) for n in ("b", "q", "s", "h", "k"))
        name = c["name"] + ("（参考分片）" if c["name"] == "prefill2k" else "")
        t.append(
            f"| {name} | {shape} | {'FP16' if c['dtype_code'] else 'BF16'} | {p['baseline_us']:,.2f} | {p['candidate_us']:,.2f} | {p['speedup']:.2f}× | {a['max_abs']:.6g} |"
        )
    t += [
        "",
        "## 调度与消融",
        "",
        "同一组独立 V 输入、相同计时协议，只替换表内的执行路径。窄分块为 32 头 × 64 个 token，宽分块为 64 × 128，大分块为 128 × 256。",
        "",
        "| 用例 | 对照 → 默认 | 对照 µs | 默认 µs | 对照 / 默认 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    speedups = {}
    for key in names[2:]:
        for row in data[key]["cases"]:
            c, p = row["case"], row["performance"]
            if row["input_sha256"] != by_name[c["name"]]["input_sha256"]:
                raise ValueError("Ablation inputs have changed")
            speedups[key, c["name"]] = p["speedup"]
            t.append(
                f"| {c['name']} | `{row['baseline']}` → `{row['config']['mode']}` | {p['baseline_us']:,.2f} | {p['candidate_us']:,.2f} | {p['speedup']:.3f}× |"
            )
    t += [
        "",
        "`gather_wide` 在 QK 内读取 K/RoPE 并一次打包 V。`pack_*` 预先打包三份张量。`counts_large` 用每行下标频次访问物理 KV，长 prefill 每 8 个头共读一份频次。`counts_single` 保持相同的物理遍历和 Cube 分块，每次只归一化一个头。`one_tile` 将短索引归约留在片上。两项单条 decode 对照保持相同分块，短索引一行比较两条完整路径。prefill 的打包对照包括遍历方式和归一化两项差异，另用 `counts_single` 隔离跨头归一化的贡献。各项耗时都来自各自独立的成对运行。[打包与短索引](ablation-pack.json) · [物理顺序与出现次数](ablation-counts.json) · [矩阵分块](ablation-tiles.json) · [跨头归一化](ablation-softmax.json)",
        "",
        f"2K prefill 中，物理顺序遍历加跨头归一化比集中打包快 {speedups['ablation-counts', 'prefill2k']:.2f}×。仅切换归一化的头数，1K 和 2K 分别快 {speedups['ablation-softmax', 'prefill1k']:.2f}× 和 {speedups['ablation-softmax', 'prefill2k']:.2f}×。单条 8K 与 16K decode 的两种打包方式耗时接近，融合读取时每次调用所需的显式临时张量由 5.25 MiB 降至 3.00 MiB（按张量形状和 dtype 计算）。",
        "",
        "## 精度与计时方法",
        "",
        f"- 独立 V 的 17 组共检查 **{summary['checked_output_elements']:,}** 个输出元素。公开容差为 atol=rtol=0.05、匹配率至少 99.5%。本次还要求每个元素满足 BF16 的 atol=rtol=0.008 或 FP16 的 0.002，最大绝对误差为 **{summary['worst_absolute_error']:.6g}**。每组复算一次，输出逐位一致，输入 SHA-256 前后一致。",
        f"- 独立 CPU FP64 实现逐行过滤索引并保留重复项，对所选 batch、query、head 的全部 512 个通道做交叉校验。用本实现对真值的误差除以公开参考对同一真值的误差，17 个用例中最大的 MARE、MERE、RMSE 三项比值分别为 {relative_metrics['mare']:.6f}、{relative_metrics['mere']:.6f}、{relative_metrics['rmse']:.6f}。L1 正常值域对应的三项限值为 5、1.5、1.5。这里报告单卡抽样指标，商业 L1 认证还涉及多平台与更大测试规模。[CANN 精度标准](https://gitcode.com/cann/opbase/blob/c260c8ab7a9be4823ac8f8a07c60442de9bf141e/docs/zh/ops_precision_standard/commercial_standard.md)",
        f"- {ntests} 项自动化测试覆盖来源文件、BF16/FP16、1/2/4 头、重复与无效索引、独立及共享 V、负缩放、非零存储偏移、图重放时变更索引等。[测试日志](hardware-tests.log)",
        "- 预热 5 次再计时。decode 及短用例每张图捕获 6 次调用，先重放预热 2 次，每次计时重放 3 次，交替 ABBA/BAAB 共 7 轮。大 prefill 的公开参考使用 eager 事件、单次调用、3 轮；大 prefill 的原生与消融对照使用各捕获 1 次调用的图、每次重放 1 次、5 轮。每轮每侧计时两次。两侧采用同一协议，计时涵盖算子调用的全部设备工作，编译、输入生成、主机校验和传输另计。",
        f"- 测量日期 {env['timestamp'][:10]}，设备 {env['hardware']['name']} 物理卡 {env['visible_devices']}。软件为 {Path(env['cann']).name}、Triton-Ascend {env['versions']['triton-ascend']}、Python {env['python']}、torch {env['versions']['torch']} 和 torch-npu {env['versions']['torch-npu']}。驱动版本、输入及代码 SHA-256 随原始记录保存。耗时覆盖本前向算子。[复现命令](../README.md) · [机器可读摘要](summary.json)",
        "",
    ]
    (directory / "REPORT.md").write_text("\n".join(t))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figure(rows, directory / "speedup.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    build(parser.parse_args().results_dir)
