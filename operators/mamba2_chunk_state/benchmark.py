"""Verify chunk-state kernels against a CPU oracle, then time paired NPU graphs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import traceback

import numpy as np

from operators.lightning_diag.benchmark import initialize, paired_benchmark
from operators.lightning_diag.reference import error_metrics
from .inputs import CASES, make_inputs
from .interface import DEFAULT_CONFIG, UPSTREAM_CONFIGS, baseline
from .kernel import configuration, experiment, run
from .reference import reference, tolerances
from .tune_baseline import as_numpy

VARIANTS = {"single_head": {"pair": False}, "masked_loads": {"full_tiles": False},
            "rank_one_tail": {"rank_one_tail": True}, "k32": {"k": 32}}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(inputs):
    import torch

    return [None if v is None else sha(v.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
            for v in inputs]


def environment(backend, device):
    root = Path(__file__).parent
    files = [root/name for name in ("baseline_kernel.py", "interface.py", "kernel.py", "inputs.py",
                                     "reference.py", "benchmark.py", "tune_baseline.py", "third_party/source.json")]
    files += [root.parent/"lightning_diag"/name for name in ("benchmark.py", "reference.py")]
    prop = backend.get_device_properties(device)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "device": device, "properties": {k: getattr(prop, k) for k in ("name", "cube_core_num", "vector_core_num", "total_memory") if hasattr(prop, k)},
        "visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "cann": str(Path(os.environ.get("LIGHTNING_CANN_ROOT", "/usr/local/Ascend/cann")).resolve()),
        "driver": Path("/usr/local/Ascend/driver/version.info").read_text().splitlines()[0],
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "torch-npu", "triton", "triton-ascend", "numpy")},
        "source_sha256": {str(p.relative_to(root.parent)): sha(p.read_bytes()) for p in files},
    }


def validate(fn, states, inputs, oracle, expected_hashes, atol, rtol):
    states.fill_(float("nan"))
    result = fn()
    values = as_numpy(result).copy()
    metrics = error_metrics(values, oracle, atol, rtol)
    states.fill_(float("nan"))
    second = fn()
    metrics.update(deterministic=np.array_equal(as_numpy(second), values),
                   output_contract=(result is states and second is states and result.device == inputs[1].device
                                    and result.shape == oracle.shape),
                   inputs_unchanged=fingerprint(inputs) == expected_hashes)
    metrics["passed"] = all((metrics["passed"], metrics["deterministic"], metrics["output_contract"], metrics["inputs_unchanged"]))
    return values, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--variants", default="optimized")
    parser.add_argument("--control", choices=("upstream", "upstream_same_tiles", *VARIANTS), default="upstream")
    parser.add_argument("--upstream-configs", type=Path, help="Tuning JSON. If omitted, use upstream 64x64x32.")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--capture-ops", type=int, default=16)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.timing = "graph"
    if min(args.samples, args.capture_ops, args.replays) < 1:
        parser.error("Timing counts must be positive")
    chosen = set(c.name for c in CASES) if args.cases == "all" else set(args.cases.split(","))
    variants = args.variants.split(",")
    if not chosen or chosen-set(c.name for c in CASES) or set(variants)-({"optimized"} | VARIANTS.keys()):
        parser.error("Unknown case or variant")
    tuned = json.loads(args.upstream_configs.read_bytes())["cases"] if args.upstream_configs else None
    torch, backend = initialize(args.device)
    report = {"environment": environment(backend, args.device), "all_passed": False, "cases": [],
              "protocol": {"seed": args.seed, "timing": args.timing, "samples": args.samples, "capture_ops": args.capture_ops,
                           "replays": args.replays, "variants": variants, "control": args.control,
                           "tuning_sha256": sha(args.upstream_configs.read_bytes()) if tuned else None}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    ok = True
    with torch.inference_mode():
        for i, case in enumerate(CASES):
            if case.name not in chosen:
                continue
            inputs = make_inputs(case, args.seed+i, args.device)
            out_dtype = "float32" if case.states_in_fp32 else case.dtype
            oracle = reference(*map(as_numpy, inputs), case.dtype, out_dtype)
            atol, rtol = tolerances(case.dtype, out_dtype)
            config = tuple(tuned[case.name]["selection"]) if tuned else DEFAULT_CONFIG
            if config not in UPSTREAM_CONFIGS:
                raise ValueError("Invalid upstream tuning record")
            digest = fingerprint(inputs)
            plan = configuration(*inputs[:3])
            if args.control == "upstream_same_tiles":
                config = (plan["m"], plan["col"], plan["k"], 3, 4)
            row = {"case": asdict(case), "input_seed": args.seed+i, "input_sha256": digest,
                   "upstream_config": config if args.control.startswith("upstream") else None,
                   "default_configuration": plan,
                   "output_dtype": out_dtype, "variants": {}}
            report["cases"].append(row)
            base_out = torch.empty(oracle.shape, dtype=getattr(torch, out_dtype), device=args.device)
            if args.control in VARIANTS:
                base_fn = partial(experiment, *inputs, states=base_out, **VARIANTS[args.control])
            else:
                base_fn = partial(baseline, *inputs, states=base_out, config=config)
            try:
                base_values, row["baseline_correctness"] = validate(base_fn, base_out, inputs, oracle, digest, atol, rtol)
            except Exception as exc:
                row["baseline_error"] = str(exc)[-1500:]
                print("CONTROL UNAVAILABLE", case.name, type(exc).__name__, flush=True)
                ok = False
                save()
                continue
            if not row["baseline_correctness"]["passed"]:
                print("BASELINE FAILED", case.name, row["baseline_correctness"], flush=True)
                ok = False
                save()
                continue
            for name in variants:
                result = {}
                row["variants"][name] = result
                out = torch.empty_like(base_out)
                fn = partial(run, *inputs, states=out) if name == "optimized" else partial(experiment, *inputs, states=out, **VARIANTS[name])
                try:
                    values, result["correctness"] = validate(fn, out, inputs, oracle, digest, atol, rtol)
                    result["vs_control"] = error_metrics(values, base_values, atol, rtol)
                    if not result["correctness"]["passed"] or not result["vs_control"]["passed"]:
                        raise AssertionError("Elementwise, output or determinism check failed")
                    p = paired_benchmark(base_fn, fn, backend, args)
                    result["performance"] = p
                    print("RESULT", case.name, name, *(round(p[k], 4) for k in ("baseline_us", "candidate_us", "speedup")), flush=True)
                except Exception as exc:
                    ok = False
                    result["error"] = str(exc)[-1500:]
                    print("FAILED", case.name, name, result.get("correctness"), flush=True)
                    traceback.print_exc(limit=6)
                save()
    report["all_passed"] = ok
    save()
    print("ALL_PASSED", ok, flush=True)
    return int(not ok)


if __name__ == "__main__":
    raise SystemExit(main())
