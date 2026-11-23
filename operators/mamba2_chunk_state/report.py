"""Rebuild the checked Markdown report and figure from retained NPU measurements."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import geometric_mean, median

from .inputs import CASES

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ABLATIONS = {
    "ablation-reduction": ("k32", "token 归约宽度固定为 32"),
    "ablation-heads": ("single_head", "同组 head 各自加载 B"),
    "ablation-masks": ("masked_loads", "完整块保留边界掩码"),
    "ablation-tail": ("rank_one_tail", "多块输入额外判断单 token 尾块"),
}


def read(name):
    return json.loads((RESULTS / (name + ".json")).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validated(data, control, primary, tuning_hash):
    protocol = data["protocol"]
    check(data["all_passed"] and protocol["control"] == control, "Unexpected run or failed correctness")
    check(protocol["tuning_sha256"] == tuning_hash, "Different upstream selection file")
    check(data["environment"]["source_sha256"] == primary["environment"]["source_sha256"], "Different source snapshots")
    expected = {r["case"]["name"]: r for r in primary["cases"]}
    rows = {}
    for row in data["cases"]:
        name = row["case"]["name"]
        check(name not in rows and row["input_sha256"] == expected[name]["input_sha256"], "Different or duplicate inputs")
        variant = row["variants"]["optimized"]
        check(all(m["passed"] for m in (row["baseline_correctness"], variant["correctness"], variant["vs_control"])),
              "Failed numerical or output check")
        p = variant["performance"]
        for field in ("baseline_us", "candidate_us"):
            raw = [t for sample in p["samples"] for t in sample[field]]
            check(len(raw) == 2 * protocol["samples"] and median(raw) == p[field], "Raw times or median inconsistent")
        check(p["speedup"] == p["baseline_us"] / p["candidate_us"], "Inconsistent speedup")
        rows[name] = p
    return rows


def draw(primary, matched, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(primary)
    y = list(range(len(names)))
    a = [primary[n]["speedup"] for n in names]
    b = [matched[n]["speedup"] for n in names]
    fig, ax = plt.subplots(figsize=(8.6, 6.3), layout="constrained")
    ax.hlines(y, b, a, color="#b9c2c5", linewidth=1.1, zorder=1)
    ax.scatter(a, y, s=38, color="#14767a", marker="o", label=f"Best of 9 upstream configs  ({stats['primary']['geomean']:.2f}×)")
    ax.scatter(b, y, s=32, color="#aa582f", marker="D", label=f"Upstream with matched M/N/K  ({stats['same_tiles']['geomean']:.2f}×)")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0.9, max(a+b)+0.13)
    ax.axvline(1, color="#73787e", linewidth=1, linestyle="--")
    ax.set_xlabel("Speedup (upstream time / optimized time)")
    ax.set_title("Mamba-2 chunk state forward on Ascend 910B3", loc="left", weight="bold", pad=13)
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(RESULTS / "speedup.png", dpi=180)
    plt.close(fig)


def main():
    primary, matched, tune = read("primary"), read("same-tiles"), read("upstream-configs")
    env = primary["environment"]
    for path, expected in env["source_sha256"].items():
        check(digest(ROOT.parent/path) == expected, "Measured source differs: " + path)
    tuning_hash = digest(RESULTS/"upstream-configs.json")
    a = validated(primary, "upstream", primary, tuning_hash)
    b = validated(matched, "upstream_same_tiles", primary, tuning_hash)
    check(set(a) == set(b) == {c.name for c in CASES} == set(tune["cases"]), "Incomplete case list")
    configs = json.loads((ROOT/"third_party/source.json").read_text())["configs"]
    for main_row, same_row in zip(primary["cases"], matched["cases"]):
        name = main_row["case"]["name"]
        check(same_row["case"]["name"] == name, "Mismatched case order")
        survey = tune["cases"][name]
        check([r["config"] for r in survey["configs"]] == configs, "Incomplete upstream configs")
        check(all(r["correctness"]["passed"] and median(r["us"]) == r["median_us"] for r in survey["configs"]),
              "Unverified upstream configuration")
        best = min(survey["configs"], key=lambda r: r["median_us"])
        check(best["config"] == survey["selection"] == main_row["upstream_config"], "Wrong baseline selection")
        plan = main_row["default_configuration"]
        check(same_row["upstream_config"] == [plan["m"], plan["col"], plan["k"], 3, 4], "Tiles not matched")
    tests = (RESULTS/"hardware-tests.log").read_text().splitlines()
    check(tests[-1] == "OK" and any(line.startswith("Ran 10 tests in ") for line in tests), "Hardware tests incomplete")
    abs_rows = {name: validated(read(name), key, primary, tuning_hash) for name, (key, _) in ABLATIONS.items()}
    stats = {}
    for name, rows in (("primary", a), ("same_tiles", b)):
        ratios = [p["speedup"] for p in rows.values()]
        stats[name] = {"cases": len(ratios), "geomean": geometric_mean(ratios), "min": min(ratios), "max": max(ratios)}
    date = datetime.fromisoformat(env["timestamp_utc"]).astimezone(timezone(timedelta(hours=8))).date()
    v = env["versions"]
    lines = [
        "# Mamba-2 分块状态前向验证与性能报告", "",
        f"{date}，单张 {env['properties']['name']}。16 个固定自建用例全部通过精度验证。"
        f"相对上游 9 组选型中的逐例最快配置，几何平均加速 **{stats['primary']['geomean']:.2f}×**"
        f"（{stats['primary']['min']:.2f}–{stats['primary']['max']:.2f}×）。"
        f"让上游使用与优化版相同的 M/N/K 后，16 例的几何平均加速为 **{stats['same_tiles']['geomean']:.2f}×**"
        f"（{stats['same_tiles']['min']:.2f}–{stats['same_tiles']['max']:.2f}×）。", "",
        "![两组上游基准的逐例加速比](speedup.png)", "",
        "## 逐例耗时", "",
        "形状依次为 batch、序列长度、head 数、共享系数组数、head 维度、状态维度、chunk size。"
        "每一对耗时的单位均为微秒，左侧是对应的上游内核，右侧是同轮实测的优化内核。", "",
        "| 用例 | 形状 | 输入→输出 | 上游最快 / 优化 | 加速 | 上游同分块 / 优化 | 加速 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    dtype_name = {"bfloat16": "BF16", "float16": "FP16", "float32": "FP32"}
    errors = {}
    for row in primary["cases"]:
        c = row["case"]
        name = c["name"]
        shape = ",".join(str(c[f]) for f in ("batch", "length", "heads", "groups", "headdim", "dstate", "chunk"))
        dtypes = dtype_name[c["dtype"]]+"→"+dtype_name[row["output_dtype"]]
        p, q = a[name], b[name]
        lines.append(f"| {name} | {shape} | {dtypes} | {p['baseline_us']:.2f} / {p['candidate_us']:.2f} | {p['speedup']:.3f}× | {q['baseline_us']:.2f} / {q['candidate_us']:.2f} | {q['speedup']:.3f}× |")
        errors.setdefault(dtypes, []).append(row["variants"]["optimized"]["correctness"])
    lines += ["", "## 单变量消融", "",
              "同一输入、同一默认实现，逐项切换一个开关。比值大于 1 表示默认路径更快。", "",
              "| 对照设置 | 用例 | 对照 µs | 默认 µs | 对照 / 默认 |", "| --- | --- | ---: | ---: | ---: |"]
    for name, rows in abs_rows.items():
        for case, p in rows.items():
            lines.append(f"| {ABLATIONS[name][1]} | {case} | {p['baseline_us']:.2f} | {p['candidate_us']:.2f} | {p['speedup']:.3f}× |")
    def span(key):
        values = [p["speedup"] for p in abs_rows[key].values()]
        return f"{min(values):.2f}–{max(values):.2f}×"
    lines += ["", f"在 3 个长归约用例中，放宽 token 归约分块贡献了 {span('ablation-reduction')}。"
              f"在 3 个不整齐的输入上，省去额外的单 token 尾块分支，得到 {span('ablation-tail')}。"
              "前者改变每次矩阵乘覆盖的 token 数，后者保持计算公式和矩阵分块一致。", "",
              f"同组 head 复用 B 在两个用例上为 {span('ablation-heads')}，完整块省略边界判断在三个用例上为 {span('ablation-masks')}。"
              "这两项在所测形状上的独立效果接近持平。", "",
              "## 数值与测量口径", "",
              "两侧均逐元素比对独立 NumPy FP64 参考。参考保留加权 B 在点积前的 dtype 舍入，并在最终输出 dtype 上比较。"
              "优化版还与本轮上游输出逐元素比较。每次检验用 NaN 填充输出、重复执行并检查全部写入、确定性与输入哈希。", "",
              "| 输入→输出 | 用例数 | 最大绝对误差 | 最大相对 L2 误差（逐例） |",
              "| --- | ---: | ---: | ---: |"]
    for dtypes, values in errors.items():
        lines.append(f"| {dtypes} | {len(values)} | {max(e['max_abs'] for e in values):.6g} | {max(e['relative_l2'] for e in values):.6g} |")
    lines += ["", "4 项 CPU 参考/来源测试与 6 项真实 NPU 接口测试通过。覆盖不对齐形状、混合类型、空序列、重复标识、"
              "负末标识、独立设置的补齐前缀终点、图回放后的输入更新，以及非连续输出写回时相邻存储的完整性。", "",
              f"实测环境使用 CANN `{Path(env['cann']).name}`、驱动 `{env['driver'].split('=')[-1]}`、Python {env['python']}、"
              f"PyTorch {v['torch']}、TorchNPU {v['torch-npu']}、Triton-Ascend {v['triton-ascend']}。", "",
              "上游配置调优使用单独的随机种子，每例实测 9 组配置，共 144 组，每组取 5 次设备图测量的中位数。"
              "正式验证使用另一批固定输入；两组上游对照和消融逐例核对输入哈希及代码版本。", "",
              "单算子设备计时使用预分配输出。每张图包含 16 次真实调用，每个测量点回放 4 次，执行 7 轮 ABBA/BAAB，"
              "两侧各取 14 个测量点的中位数再计算比值。编译、参考计算、输入传输和完整模型推理在计时范围之外。", "",
              "## 复现材料", "",
              "- [分块状态数学定义与运行命令](../README.md)",
              "- [144 组上游调优记录](upstream-configs.json)",
              "- [上游逐例最快配置的成对计时](primary.json)",
              "- [上游同分块的成对计时](same-tiles.json)",
              "- [硬件与接口测试日志](hardware-tests.log)",
              "- [归约分块消融](ablation-reduction.json) · [共享读取消融](ablation-heads.json) · [边界掩码消融](ablation-masks.json) · [尾块分支消融](ablation-tail.json)",
              "- [机器可读汇总](summary.json)", ""]
    draw(a, b, stats)
    (RESULTS/"summary.json").write_text(json.dumps(stats, indent=2)+"\n")
    (RESULTS/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
