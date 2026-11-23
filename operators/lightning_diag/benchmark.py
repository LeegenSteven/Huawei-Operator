"""Validate against an FP64 oracle and measure paired baseline/candidate runs.

Example (from the repository root, after sourcing env.sh):
ASCEND_RT_VISIBLE_DEVICES=0 python3.11 -m operators.lightning_diag.benchmark \
    --device npu:0 --cases small,batch2 --variants optimized --timing graph \
    --output operators/lightning_diag/results/smoke.json
"""

import argparse
from datetime import datetime, timedelta, timezone
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import traceback

import numpy as np

from .inputs import CASES, make_torch_inputs
from .reference import numpy_reference, error_metrics


# Fixed before accelerator experiments. A failed check excludes that variant's
# timing; the runner never relaxes tolerances or changes the reference.
TOLERANCES = {
    "bfloat16": (0.03125, 0.015625),
    "float16": (0.01, 0.002),
    "float32": (0.002, 0.0001),
}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(device):
    import torch
    if device.startswith("npu"):
        import torch_npu  # noqa: F401 -- registers the NPU backend
        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU unavailable; no CPU timing fallback")
        torch.npu.set_device(device)
        backend = torch.npu
    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU timing fallback")
        torch.cuda.set_device(device)
        backend = torch.cuda
    else:
        raise ValueError("Use an explicit accelerator device, e.g. npu:0")
    torch.set_num_threads(1)
    return torch, backend


def environment(torch, backend, device):
    root = Path(__file__).parent
    versions = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "versions": versions,
        "device": device,
        "properties": {name: getattr(backend.get_device_properties(device), name)
                       for name in ("name", "total_memory", "cube_core_num", "vector_core_num")
                       if hasattr(backend.get_device_properties(device), name)},
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "cann_installation": str(Path(os.environ.get("LIGHTNING_CANN_ROOT", "/usr/local/Ascend/cann")).resolve()),
        "driver_version": (Path("/usr/local/Ascend/driver/version.info").read_text().splitlines()[0]
                           if Path("/usr/local/Ascend/driver/version.info").exists() else None),
        "source_sha256": {name: sha256_file(root / name) for name in
                          ("kernel.py", "baseline_kernel.py", "inputs.py", "reference.py", "benchmark.py")},
        "upstream": json.loads((root / "third_party/sources.json").read_text()),
    }


def variants():
    from .kernel import baseline, optimized, run
    return {
        "vllm_tile64": functools.partial(baseline, tile=64),
        "vllm_tile128": functools.partial(baseline, tile=128),
        "split64": functools.partial(optimized, tile=64, column_tile=64, anchor=False, native_qk=False, interleave=False),
        "anchor64": functools.partial(optimized, tile=64, column_tile=64, anchor=True, native_qk=False, interleave=False),
        "native32": functools.partial(optimized, tile=32, column_tile=32, anchor=False, native_qk=True, interleave=False),
        "native64": functools.partial(optimized, tile=64, column_tile=64, anchor=False, native_qk=True, interleave=False),
        "rect32": functools.partial(optimized, tile=32, column_tile=128, anchor=False, native_qk=False, interleave=False),
        "rect_native32": functools.partial(optimized, tile=32, column_tile=128, anchor=False, native_qk=True, interleave=False),
        "optimized": run,
        "custom": optimized,
        "unpruned": functools.partial(optimized, prune_tail=False),
        "optimized32": functools.partial(optimized, tile=32, column_tile=128, anchor=False, native_qk=True, interleave=True),
        "optimized64": functools.partial(optimized, tile=64, column_tile=128, anchor=False, native_qk=True, interleave=True),
        "anchor_rect32": functools.partial(optimized, tile=32, column_tile=128, anchor=True, native_qk=True, interleave=True),
        "late64": functools.partial(optimized, tile=64, column_tile=64, native_qk=False, interleave=True, late_value=True),
        "early128": functools.partial(optimized, tile=128, column_tile=128, native_qk=False, interleave=True, late_value=False),
        "late128": functools.partial(optimized, tile=128, column_tile=128, native_qk=False, interleave=True, late_value=True),
        "native128": functools.partial(optimized, tile=128, column_tile=128, native_qk=True, interleave=True, late_value=True),
        "native128_early": functools.partial(optimized, tile=128, column_tile=128, native_qk=True, interleave=True, late_value=False),
    }


def prepare_timer(fn, backend, timing, capture_ops):
    for _ in range(5):
        fn()
    backend.synchronize()
    if timing == "event":
        return fn, 1, None
    graph = backend.NPUGraph() if hasattr(backend, "NPUGraph") else backend.CUDAGraph()
    # Captures the real kernel repeatedly with preallocated output. Compilation
    # has already completed. Graph replay removes Python submission gaps.
    with backend.graph(graph):
        for _ in range(capture_ops):
            fn()
    backend.synchronize()
    for _ in range(2):
        graph.replay()
    backend.synchronize()
    return graph.replay, capture_ops, graph


def sample_us(fn, backend, calls, operations):
    start, end = (backend.Event(enable_timing=True) for _ in range(2))
    backend.synchronize()
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000 / (calls * operations))


def paired_benchmark(baseline_fn, candidate_fn, backend, args):
    base_call, base_ops, base_graph = prepare_timer(baseline_fn, backend, args.timing, args.capture_ops)
    cand_call, cand_ops, cand_graph = prepare_timer(candidate_fn, backend, args.timing, args.capture_ops)
    samples = []
    for i in range(args.samples):
        # Alternate ABBA/BAAB: each sample contains equal work and counteracts
        # slow clock or temperature drift. All raw measurements are retained.
        order = ("baseline", "candidate", "candidate", "baseline") if i % 2 == 0 else ("candidate", "baseline", "baseline", "candidate")
        row = {"baseline_us": [], "candidate_us": [], "order": list(order)}
        for name in order:
            fn, ops = (base_call, base_ops) if name == "baseline" else (cand_call, cand_ops)
            row[name + "_us"].append(sample_us(fn, backend, args.replays, ops))
        samples.append(row)
    base = [t for row in samples for t in row["baseline_us"]]
    cand = [t for row in samples for t in row["candidate_us"]]
    base_median, cand_median = statistics.median(base), statistics.median(cand)
    return {
        "timing": args.timing,
        "capture_ops": base_ops,
        "replays_per_measurement": args.replays,
        "samples": samples,
        "baseline_us": base_median,
        "candidate_us": cand_median,
        "speedup": base_median / cand_median,
        "baseline_min_max_us": [min(base), max(base)],
        "candidate_min_max_us": [min(cand), max(cand)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--variants", default="optimized")
    parser.add_argument("--baseline-tile", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--timing", choices=("graph", "event"), default="graph")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--capture-ops", type=int, default=16)
    parser.add_argument("--replays", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.samples, args.capture_ops, args.replays) <= 0:
        parser.error("Timing counts must be positive")
    torch, backend = initialize(args.device)
    from .kernel import baseline, configuration
    implementations = variants()
    selected = args.variants.split(",")
    unknown = set(selected) - implementations.keys()
    if unknown:
        parser.error("Unknown variants: " + ",".join(sorted(unknown)))
    cases = CASES if args.cases == "all" else [c for name in args.cases.split(",") for c in CASES if c.name == name]
    if not cases or (args.cases != "all" and len(cases) != len(args.cases.split(","))):
        parser.error("Unknown or empty case selection")
    report = {"environment": environment(torch, backend, args.device),
              "protocol": vars(args) | {"output": str(args.output)},
              "baseline": f"unmodified pinned vLLM _fwd_diag_kernel, CBLOCK={args.baseline_tile}",
              "oracle": "independent NumPy FP64 formula, rounded to output dtype",
              "tolerances": TOLERANCES, "cases": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    save()
    failed = False
    with torch.inference_mode():
        for case_index, case in enumerate(cases):
            print("CASE", case.name, case.to_dict(), flush=True)
            case_seed = args.seed + CASES.index(case)
            inputs = make_torch_inputs(case, args.device, case_seed)
            cpu_inputs = [x.float().cpu().numpy().copy() for x in inputs]
            hashes_before = [hashlib.sha256(x.tobytes()).hexdigest() for x in cpu_inputs]
            oracle64 = numpy_reference(*cpu_inputs)
            expected = torch.from_numpy(oracle64).to(inputs[0].dtype).float().numpy()
            atol, rtol = TOLERANCES[case.dtype]
            result = {"case": case.to_dict(), "input_seed": case_seed,
                      "input_sha256": hashes_before,
                      "automatic_configuration": configuration(inputs[0], inputs[2]),
                      "variants": {}}
            report["cases"].append(result)
            base_out = torch.empty_like(inputs[2])
            base_fn = functools.partial(baseline, *inputs, out=base_out, tile=args.baseline_tile)
            try:
                base_fn()
                backend.synchronize()
                base_metrics = error_metrics(base_out.float().cpu().numpy(), expected, atol, rtol)
                result["baseline_correctness"] = base_metrics
                if not base_metrics["passed"]:
                    raise AssertionError("Baseline failed the fixed numerical contract")
            except Exception as error:
                result["baseline_error"] = repr(error)
                traceback.print_exc()
                failed = True
                save()
                continue
            for name in selected:
                print("VARIANT", name, flush=True)
                try:
                    out = torch.empty_like(inputs[2])
                    fn = functools.partial(implementations[name], *inputs, out=out)
                    fn()
                    backend.synchronize()
                    first = out.clone()
                    metrics = error_metrics(first.float().cpu().numpy(), expected, atol, rtol)
                    fn()
                    backend.synchronize()
                    metrics["deterministic"] = bool(torch.equal(first, out))
                    metrics["shape_dtype_device_match"] = bool(out.shape == inputs[2].shape and out.dtype == inputs[0].dtype and out.device == inputs[0].device)
                    current = [x.float().cpu().numpy() for x in inputs]
                    metrics["inputs_unchanged"] = hashes_before == [hashlib.sha256(x.tobytes()).hexdigest() for x in current]
                    metrics["passed"] = bool(metrics["passed"] and metrics["deterministic"] and metrics["shape_dtype_device_match"] and metrics["inputs_unchanged"])
                    row = {"correctness": metrics}
                    result["variants"][name] = row
                    if not metrics["passed"]:
                        raise AssertionError("Candidate failed the fixed numerical/semantic contract")
                    if not args.validate_only:
                        row["performance"] = paired_benchmark(base_fn, fn, backend, args)
                        print("RESULT", case.name, name, row["performance"]["baseline_us"],
                              row["performance"]["candidate_us"], row["performance"]["speedup"], flush=True)
                    else:
                        print("CORRECT", case.name, name, metrics, flush=True)
                except Exception as error:
                    result["variants"].setdefault(name, {})["error"] = repr(error)
                    traceback.print_exc()
                    failed = True
                save()
    summary = {}
    for name in selected:
        valid = [c["variants"].get(name, {}).get("performance", {}).get("speedup") for c in report["cases"]]
        valid = [s for s in valid if s is not None]
        if valid:
            summary[name] = {"measured_cases": len(valid),
                             "geomean_speedup": math.exp(sum(math.log(s) for s in valid) / len(valid)),
                             "min_speedup": min(valid), "max_speedup": max(valid)}
    report["summary"] = summary
    report["all_passed"] = not failed
    save()
    print("SUMMARY", json.dumps(summary), "ALL_PASSED", not failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
