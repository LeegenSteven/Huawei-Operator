"""Shared validation and Ascend launcher for the original Mamba implementation."""

import json
from pathlib import Path

import torch
import triton

from .baseline_kernel import _chunk_state_fwd_kernel

_SOURCE = Path(__file__).parent / "third_party" / "source.json"
UPSTREAM_CONFIGS = tuple(tuple(row) for row in json.loads(_SOURCE.read_text())["configs"])
DEFAULT_CONFIG = UPSTREAM_CONFIGS[-1]
DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def prepare(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32):
    if x.ndim != 4 or B.ndim != 4 or dt.ndim != 4:
        raise ValueError("Expected x, B and dt to have four axes")
    b, t, h, p = x.shape
    g, n = B.shape[2:]
    chunks, c = dt.shape[2:]
    if min(b, h, p, g, n, c) < 1 or h % g:
        raise ValueError("Dimensions must be positive and groups must divide heads")
    if B.shape[:2] != (b, t) or dt.shape[:2] != (b, h) or chunks != triton.cdiv(t, c):
        raise ValueError("Incompatible batch, sequence, head or chunk count")
    if dA_cumsum.shape != dt.shape or (seq_idx is not None and seq_idx.shape != (b, t)):
        raise ValueError("Incompatible prefix or sequence index shape")
    if any(v.dtype not in DTYPES for v in (x, B, dt, dA_cumsum)):
        raise TypeError("x, B, dt and dA_cumsum must be BF16, FP16 or FP32")
    if seq_idx is not None and seq_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError("seq_idx must have int32 or int64 dtype")
    inputs = (x, B, dt, dA_cumsum) if seq_idx is None else (x, B, dt, dA_cumsum, seq_idx)
    if x.device.type != "npu" or any(v.device != x.device for v in inputs):
        raise ValueError("Inputs must share the same Ascend NPU")
    shape = (b, chunks, h, p, n)
    if states is None:
        return torch.empty(shape, dtype=torch.float32 if states_in_fp32 else B.dtype, device=x.device)
    if states.shape != shape or states.device != x.device or states.dtype not in DTYPES:
        raise ValueError("Invalid destination shape, dtype or device")
    return states


def launch_upstream(B, x, dt, dA_cumsum, seq_idx, states, config):
    # The 910B3 compiler can emit contiguous stores for a strided destination.
    # Pack unusual layouts on-device and copy back into exactly the user's view.
    destination = states
    B, x, dt, dA_cumsum = (v.contiguous() for v in (B, x, dt, dA_cumsum))
    if seq_idx is not None:
        seq_idx = seq_idx.contiguous()
    if not states.is_contiguous():
        states = torch.empty_like(states, memory_format=torch.contiguous_format)
    b, length, h, p = x.shape
    g, n = B.shape[2:]
    nc, c = dt.shape[2:]
    m, col, k, stages, warps = config
    if not length:
        return destination
    grid = (triton.cdiv(p, m)*triton.cdiv(n, col), b*nc, h)
    _chunk_state_fwd_kernel[grid](
        x, B, states, dt, dA_cumsum, seq_idx, p, n, c, b, length, h//g,
        *x.stride(), *B.stride(), *states.stride(),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *(seq_idx.stride() if seq_idx is not None else (0, 0)),
        HAS_SEQ_IDX=seq_idx is not None, BLOCK_SIZE_M=m, BLOCK_SIZE_N=col, BLOCK_SIZE_K=k,
        num_stages=stages, num_warps=warps,
    )
    if states is not destination:
        destination.copy_(states)
    return destination


def baseline(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True, *, config=DEFAULT_CONFIG):
    states = prepare(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32)
    return launch_upstream(B, x, dt, dA_cumsum, seq_idx, states, config)
