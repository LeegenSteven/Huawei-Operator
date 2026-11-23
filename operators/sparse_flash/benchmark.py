"""Check full outputs, then time identical inputs with alternating paired NPU events."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
from types import SimpleNamespace

from operators.lightning_diag.benchmark import initialize, paired_benchmark

from . import baseline
from .inputs import CASES, NATIVE_CASES, make
from .reference import chunked_reference, compare_all, fingerprint, fp64_rows

ROOT = Path(__file__).parent


def native(query, key, value, sparse_indices, scale_value, query_rope, key_rope):
    import torch_npu

    if key.data_ptr() != value.data_ptr():
        raise ValueError("CANN attention_mode=2 requires shared K and V storage")
    return torch_npu.npu_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        query_rope=query_rope,
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        attention_mode=2,
    )[0]


def environment(backend, device):
    def version(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    props = backend.get_device_properties(device)
    driver = Path("/usr/local/Ascend/driver/version.info")
    files = [
        "kernel.py",
        "inputs.py",
        "reference.py",
        "benchmark.py",
        "baseline.py",
        "../lightning_diag/benchmark.py",
    ]
    return dict(
        timestamp=datetime.now(timezone.utc).isoformat(),
        architecture=platform.machine(),
        python=platform.python_version(),
        device=device,
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        hardware={
            name: getattr(props, name)
            for name in ("name", "total_memory", "cube_core_num", "vector_core_num")
            if hasattr(props, name)
        },
        driver=driver.read_text().splitlines()[0] if driver.is_file() else None,
        cann=str(Path(os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")).resolve()),
        versions={name: version(name) for name in ("torch", "torch-npu", "triton", "triton-ascend", "numpy")},
        source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        upstream=json.loads((ROOT / "third_party/source.json").read_text()),
    )


def protocol(case, suite, samples):
    # The large eager reference is memory intensive. Use one full invocation per event.
    if case.q >= 1024 and suite == "public":
        return SimpleNamespace(timing="event", samples=samples or 3, capture_ops=1, replays=1)
    if case.q >= 1024:
        return SimpleNamespace(timing="graph", samples=samples or 5, capture_ops=1, replays=1)
    return SimpleNamespace(timing="graph", samples=samples or 7, capture_ops=6, replays=3)


def run_case(case, args, torch, backend):
    from .kernel import configuration, experiment, run

    shared = args.suite == "native"
    x = make(case, args.device, shared_value=shared)
    before = fingerprint(x)
    candidate = partial(run, **x)
    if args.suite == "native":
        base, label = partial(native, **x), "CANN npu_sparse_flash_attention"
    elif args.suite == "ablation":
        base, label = partial(experiment, **x, mode=args.control), args.control
    elif case.name == "prefill2k":
        base, label = partial(chunked_reference, x, 32), "public FP32, query slices of 32"
    else:
        base, label = partial(baseline.run, **x), "public FP32"
    result = candidate()
    accuracy = compare_all(result, x)
    fp64 = fp64_rows(x, result)
    repeat = torch.equal(result, candidate())
    del result
    out = dict(
        case=asdict(case),
        shared_value=shared,
        input_sha256=before,
        baseline=label,
        config=configuration(x["query"], x["key"], x["sparse_indices"]),
        accuracy=accuracy,
        fp64=fp64,
        deterministic=repeat,
    )
    if args.suite != "public":
        control = base()
        out["control_accuracy"] = compare_all(control, x)
        del control
    good = (
        accuracy["passed"]
        and fp64["passed"]
        and repeat
        and out.get("control_accuracy", {"passed": True})["passed"]
    )
    if good:
        out["performance"] = paired_benchmark(
            base, candidate, backend, protocol(case, args.suite, args.samples)
        )
    out["inputs_unchanged"] = before == fingerprint(x)
    out["passed"] = bool(good and out["inputs_unchanged"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("public", "native", "ablation"), default="public")
    parser.add_argument("--cases", default="all", help="Comma-separated case names, or all")
    parser.add_argument("--control", help="Explicit alternative kernel schedule for an ablation")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--samples", type=int, help="Override the number of ABBA/BAAB quartets")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.device.startswith("npu") or (args.samples is not None and args.samples <= 0):
        parser.error("Select an Ascend NPU and a positive sample count")
    if (args.suite == "ablation") != bool(args.control):
        parser.error("--control is required only for ablation")
    available = {c.name: c for c in CASES if args.suite != "native" or c.name in NATIVE_CASES}
    names = list(available) if args.cases == "all" else args.cases.split(",")
    if not names or set(names) - available.keys() or len(set(names)) != len(names):
        parser.error("Unknown or duplicate case")
    torch, backend = initialize(args.device)
    report = dict(
        environment=environment(backend, args.device), suite=args.suite, control=args.control, cases=[]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)

    with torch.inference_mode():
        for name in names:
            print("START", args.suite, name, flush=True)
            row = run_case(available[name], args, torch, backend)
            report["cases"].append(row)
            save()
            timing = row.get("performance", {})
            print(
                "CASE",
                name,
                "pass",
                row["passed"],
                "baseline_us",
                timing.get("baseline_us"),
                "candidate_us",
                timing.get("candidate_us"),
                "speedup",
                timing.get("speedup"),
                flush=True,
            )
            if not row["passed"]:
                raise RuntimeError("Validation failed for " + name)
            gc.collect()
            backend.empty_cache()
    times = [row["performance"] for row in report["cases"]]
    report["summary"] = dict(
        geomean=math.exp(sum(math.log(p["speedup"]) for p in times) / len(times)),
        ratio_of_sums=sum(p["baseline_us"] for p in times) / sum(p["candidate_us"] for p in times),
        measured_cases=len(times),
        all_passed=all(row["passed"] for row in report["cases"]),
    )
    save()
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
