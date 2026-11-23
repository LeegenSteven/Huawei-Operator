"""Measure all nine original Triton configurations on the target NPU."""

import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import statistics

from operators.lightning_diag.benchmark import initialize, prepare_timer, sample_us
from operators.lightning_diag.reference import error_metrics
from .inputs import CASES, make_inputs
from .interface import UPSTREAM_CONFIGS, baseline
from .reference import reference, tolerances


def as_numpy(v):
    if v is None:
        return None
    cpu = v.detach().cpu()
    return (cpu.float() if cpu.is_floating_point() else cpu).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cases", default="all")
    args = parser.parse_args()
    torch, backend = initialize(args.device)
    selected = {c.name for c in CASES} if args.cases == "all" else set(args.cases.split(","))
    if selected - {c.name for c in CASES}:
        parser.error("Unknown case")
    report = {"seed": 31001, "capture_ops": 8, "replays": 3, "samples": 5, "cases": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(CASES):
        if case.name not in selected:
            continue
        inputs = make_inputs(case, report["seed"]+i, args.device)
        out_dtype = "float32" if case.states_in_fp32 else case.dtype
        oracle = reference(*map(as_numpy, inputs), case.dtype, out_dtype)
        out = torch.empty(oracle.shape, dtype=getattr(torch, out_dtype), device=args.device)
        rows = []
        best = None
        for j, config in enumerate(UPSTREAM_CONFIGS):
            row = {"config": config}
            rows.append(row)
            try:
                fn = partial(baseline, *inputs, states=out, config=config)
                out.fill_(float("nan"))
                fn()
                metrics = error_metrics(as_numpy(out), oracle, *tolerances(case.dtype, out_dtype))
                row["correctness"] = metrics
                if metrics["passed"]:
                    replay, ops, graph = prepare_timer(fn, backend, "graph", report["capture_ops"])
                    row["us"] = [sample_us(replay, backend, report["replays"], ops) for _ in range(report["samples"])]
                    row["median_us"] = statistics.median(row["us"])
                    if best is None or row["median_us"] < best["median_us"]:
                        best = row
                    del replay, graph
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc).strip().splitlines()[-1][:600]
            if "median_us" not in row:
                print("UNAVAILABLE", case.name, j, row.get("error", row.get("correctness")), flush=True)
        report["cases"][case.name] = {"case": asdict(case), "selection": best["config"] if best else None, "configs": rows}
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print("TUNED", case.name, best["config"] if best else None, best["median_us"] if best else None, flush=True)
    return int(any(c["selection"] is None for c in report["cases"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
