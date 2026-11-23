"""Build a Markdown report and static figure from retained benchmark JSON."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    data = json.loads(path.read_text())
    if not data.get("all_passed"):
        raise ValueError(f"Unsuccessful benchmark cannot be reported as a result: {path}")
    for case in data["cases"]:
        if not case["baseline_correctness"]["passed"]:
            raise ValueError("Baseline correctness failed")
        for row in case["variants"].values():
            if not row["correctness"]["passed"]:
                raise ValueError("Candidate correctness failed")
    return data


def geomean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def measurements(data, variant="optimized"):
    return [(row, row["variants"][variant]["performance"]) for row in data["cases"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--hardware-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    primary, control, ablation = (load(p) for p in (args.primary, args.control, args.ablation))
    root = Path(__file__).parent
    for name, expected in primary["environment"]["source_sha256"].items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Published source differs from measured source: {name}")
    for other in (control, ablation):
        if other["environment"]["source_sha256"] != primary["environment"]["source_sha256"]:
            raise ValueError("Final experiments used different source versions")
    by_name = {row["case"]["name"]: row for row in primary["cases"]}
    for other in (control, ablation):
        for row in other["cases"]:
            if row["input_sha256"] != by_name[row["case"]["name"]]["input_sha256"]:
                raise ValueError("Comparison inputs differ")
    hardware = args.hardware_log.read_text().replace("\r", "")
    if "\nOK\n" not in hardware or "FAILED" in hardware:
        raise ValueError("Hardware contract tests did not pass")
    main_rows, control_rows = measurements(primary), measurements(control)
    ratios = [p["speedup"] for _, p in main_rows]
    control_ratios = [p["speedup"] for _, p in control_rows]
    custom_control = [p["speedup"] for row, p in control_rows
                      if row["automatic_configuration"]["engine"] == "custom"]
    summary = {
        "primary_cases": len(main_rows), "primary_geomean": geomean(ratios),
        "primary_range": [min(ratios), max(ratios)],
        "control_cases": len(control_rows), "control_geomean": geomean(control_ratios),
        "control_range": [min(control_ratios), max(control_ratios)],
        "custom_control_cases": len(custom_control), "custom_control_geomean": geomean(custom_control),
        "custom_control_range": [min(custom_control), max(custom_control)],
        "source_sha256": primary["environment"]["source_sha256"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Lightning 对角注意力：验证与性能报告", "",
        "测量日期：2026-11-23（北京时间）。设备：单张 Ascend 910B3。", "",
        f"最终默认入口在 {len(main_rows)} 组固定自建工作负载上全部通过精度检查。"
        f"相对原始 vLLM 32 分块内核，几何平均加速 **{summary['primary_geomean']:.2f}×**，"
        f"范围 **{min(ratios):.2f}–{max(ratios):.2f}×**。", "",
        f"对 {len(control_rows)} 组 D=E=64 工作负载，进一步与同一原版内核的 64 分块控制组比较，"
        f"几何平均为 **{summary['control_geomean']:.2f}×**。"
        f"其中 {len(custom_control)} 组自研 BF16 路径的几何平均为 **{summary['custom_control_geomean']:.2f}×**。"
        "FP32 完整块路径复用未修改的 vLLM 64 分块内核，按来源单独标识。", "",
        "## 环境", "", "| 项目 | 实测值 |", "| --- | --- |",
        "| NPU | Ascend 910B3，20 Cube Core / 40 Vector Core |",
        f"| CANN | `{primary['environment']['cann_installation']}` |",
        f"| 驱动 | `{primary['environment']['driver_version']}` |",
        f"| Python | `{primary['environment']['python']}` |",
    ]
    for name, version in primary["environment"]["versions"].items():
        lines.append(f"| {name} | `{version}` |")
    lines += ["", "## 主对照：未修改的 vLLM 内核，CBLOCK=32", "",
              "| 用例 | B,H,N,D,E | dtype | baseline μs | 默认入口 μs | 加速比 |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row, p in main_rows:
        case = row["case"]
        shape = ",".join(str(case[k]) for k in ("batch", "heads", "tokens", "qk_dim", "value_dim"))
        name = case["name"] + ("（原版64路径）" if row["automatic_configuration"]["engine"] != "custom" else "")
        lines.append(f"| {name} | {shape} | {case['dtype']} | {p['baseline_us']:.3f} | {p['candidate_us']:.3f} | {p['speedup']:.3f}× |")
    lines += ["", "## 更强控制组：同一原版内核，CBLOCK=64", "",
              f"该组选取 D=E=64 的 {len(control_rows)} 个用例；主对照中的其余用例仍完整保留在上表。", "",
              "| 用例 | 原版64 μs | 默认入口 μs | 加速比 |",
              "| --- | ---: | ---: | ---: |"]
    for row, p in control_rows:
        lines.append(f"| {row['case']['name']} | {p['baseline_us']:.3f} | {p['candidate_us']:.3f} | {p['speedup']:.3f}× |")
    lines += ["", "FP32 行调用相同的原版 64 分块内核；接近 1 的差异属于独立计时的波动。", "",
              "## 尾块任务裁剪消融", "",
              "两项实现保持同一精度、分块与数据加载策略，只切换 `prune_tail`。", "",
              "| 用例 | 关闭裁剪 μs | 开启裁剪 μs | 关闭/开启 |",
              "| --- | ---: | ---: | ---: |"]
    for row in ablation["cases"]:
        off = row["variants"]["unpruned"]["performance"]["candidate_us"]
        on = row["variants"]["optimized"]["performance"]["candidate_us"]
        lines.append(f"| {row['case']['name']} | {off:.3f} | {on:.3f} | {off/on:.3f}× |")
    lines += ["", "## 精度与实验约束", "",
              "- 4 项独立参考/来源校验和 6 项真实 NPU 接口与边界测试全部通过，包含多个 dtype/shape 子用例。",
              "- 接口与边界测试覆盖短序列、尾块、D≠E、空序列、FP32 抵消、因果隔离、块间隔离及图回放后输入更新。",
              "- 精度参考是独立 NumPy FP64 公式，舍入到输出 dtype 后检查所有元素；阈值在性能实验前固定。",
              "- BF16 使用 atol=0.03125、rtol=0.015625；FP32 使用 atol=0.002、rtol=0.0001；FP16 补充测试使用 atol=0.01、rtol=0.002。",
              "- 计时为 NPU 图回放设备事件，不包含编译、CPU 参考、主机传输或完整模型。两实现使用相同输入及预分配输出。",
              "- 每张图 16 次调用，每个测量点回放 4 次，7 轮 ABBA/BAAB；两侧各有 14 个原始测量点，分别取中位数。",
              "- 主对照、控制组、尾块消融的输入 SHA-256 已逐例核对一致，当前代码与测量记录中的来源哈希一致。", "",
              "## 实现贡献与取舍", "",
              "1. 原生低精度输入 QK + FP32 score/PV/累加，与 V 后置加载共同控制缓冲存活期，使常见路径可使用 128 分块。",
              "2. 仅发射有效 query tile；短序列/尾块裁剪的贡献由上面的单变量消融测量。",
              "3. 基于 shape/dtype 的保守分派覆盖短序列与大维度；FP32 完整块复用强控制组内核。",
              "4. 局部指数分解和更复杂矩形归并作为探索配置保留，默认版本采用实测更合适的路径。", "",
              "早期探索记录表明：原版直接改为 128 分块，以及自研 128 分块提前加载 V 的负对照，会超出本机 192 KiB UB；"
              "双 BF16 PV 原型的 FP32 输出最大绝对误差约为 0.453，相对 L2 误差约为 0.00168。默认路径因此保持 FP32 PV 并后置加载 V。"
              "相关诊断保存在原始记录中，数学公式的来源与许可见上级说明。", "",
              "## 原始记录与复现", "",
              f"- [主对照]({args.primary.name})",
              f"- [64 分块控制组]({args.control.name})",
              f"- [尾块消融]({args.ablation.name})",
              f"- [真实 NPU 接口与边界测试日志]({args.hardware_log.name})",
              "- [数值路径探针](20261123-precision-probe.json)",
              "- [探索阶段缓冲/尾块消融](20261123-ablation-r04.json)",
              "- [汇总数据与来源哈希](summary.json)", "",
              "运行方式见 [算子说明](../README.md)，报告可使用 `python3.11 -m operators.lightning_diag.summarize --help` 重新生成。", "",
              "![逐例加速比](speedup.png)", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, rows, title in [(axes[0], main_rows, f"Original vLLM CBLOCK=32: {summary['primary_geomean']:.2f}x geometric mean"),
                            (axes[1], control_rows, f"Stronger CBLOCK=64 control: {summary['control_geomean']:.2f}x geometric mean")]:
        names = [row["case"]["name"] for row, _ in rows]
        values = [p["speedup"] for _, p in rows]
        colors = ["#167a83" if row["automatic_configuration"]["engine"] == "custom" else "#88949d" for row, _ in rows]
        ax.bar(range(len(names)), values, color=colors, width=0.7)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(names)), names, rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("Speedup (x)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            ax.text(i, value + max(values) * 0.025, f"{value:.2f}", ha="center", fontsize=7)
    fig.suptitle("Lightning diagonal forward on Ascend 910B3", fontsize=14)
    axes[0].legend(handles=[Patch(color="#167a83", label="Custom kernel"),
                            Patch(color="#88949d", label="Reused vLLM FP32 path")],
                   loc="upper left", frameon=False, fontsize=8)
    fig.savefig(args.output_dir / "speedup.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
