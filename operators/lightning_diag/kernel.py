"""Rectangular causal grouping with native-input QK and FP32 PV.

This implementation is developed in this repository. The unmodified vLLM
baseline and its provenance are kept in baseline_kernel.py and third_party/.
"""

import triton
import triton.language as tl


@triton.jit
def _rectangle(
    q, K, V, slope, bh, block_start, row_start, key_start,
    N: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
    ROW_TILE: tl.constexpr, KEY_TILE: tl.constexpr,
    ANCHOR: tl.constexpr, NATIVE_QK: tl.constexpr, LATE_VALUE: tl.constexpr,
):
    rows = tl.arange(0, ROW_TILE)
    cols = tl.arange(0, KEY_TILE)
    features = tl.arange(0, D)
    output_features = tl.arange(0, E)
    key_pos = block_start + key_start + cols
    key = tl.load(K + bh * N * D + features[:, None] + key_pos[None, :] * D,
                  mask=key_pos[None, :] < N, other=0.0)
    if not LATE_VALUE:
        value = tl.load(V + bh * N * E + key_pos[:, None] * E + output_features[None, :],
                        mask=key_pos[:, None] < N, other=0.0).to(tl.float32)
    if not NATIVE_QK:
        key = key.to(tl.float32)
    score = tl.dot(q, key)
    if ANCHOR:
        # Bounded local factors; retained as an ablation, not assumed faster.
        gap = row_start - (key_start + KEY_TILE - 1)
        row_decay = tl.exp(-slope * rows.to(tl.float32))
        column_decay = tl.exp(-slope * (KEY_TILE - 1 - cols).to(tl.float32))
        score = score * row_decay[:, None] * column_decay[None, :] * tl.exp(-slope * gap.to(tl.float32))
    else:
        distance = row_start + rows[:, None] - key_start - cols[None, :]
        score = score * tl.exp(-slope * distance.to(tl.float32))
    if LATE_VALUE:
        # Keep V out of the score/decay live range. This reduces UB pressure
        # when a wider query tile is used on the Ascend backend.
        value = tl.load(V + bh * N * E + key_pos[:, None] * E + output_features[None, :],
                        mask=key_pos[:, None] < N, other=0.0).to(tl.float32)
    # The independent precision probe rejected a two-BF16 PV decomposition on
    # this backend. Keep the score, V and accumulator in FP32 in the PV stage.
    return tl.dot(score, value)


@triton.jit
def _optimized_diag(
    Q, K, V, Slopes, Out,
    H: tl.constexpr, N: tl.constexpr, D: tl.constexpr, E: tl.constexpr,
    BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr, TILE: tl.constexpr,
    COLUMN_TILE: tl.constexpr, ANCHOR: tl.constexpr,
    NATIVE_QK: tl.constexpr, INTERLEAVE: tl.constexpr, LATE_VALUE: tl.constexpr,
    PRUNE_TAIL: tl.constexpr,
):
    task = tl.program_id(0)
    slots: tl.constexpr = BLOCK // TILE
    if PRUNE_TAIL:
        tiles_per_head: tl.constexpr = triton.cdiv(N, TILE)
        last_slots: tl.constexpr = triton.cdiv((N - 1) % BLOCK + 1, TILE)
        bh = task // tiles_per_head
        within_head = task % tiles_per_head
        block_id = within_head // slots
        slot = within_head % slots
        active_slots = tl.where(block_id == NUM_BLOCKS - 1, last_slots, slots)
    else:
        block_head = task // slots
        slot = task % slots
        bh = block_head // NUM_BLOCKS
        block_id = block_head % NUM_BLOCKS
        active_slots = slots
    if INTERLEAVE:
        # Neighbouring tasks have complementary causal workloads:
        # [0, slots-1, 1, slots-2, ...]. No output atomics or communication.
        row_tile = tl.where(slot % 2 == 0, slot // 2, active_slots - 1 - slot // 2)
    else:
        row_tile = slot
    head = bh % H
    block_start = block_id * BLOCK
    rows = tl.arange(0, TILE)
    cols = tl.arange(0, TILE)
    features = tl.arange(0, D)
    output_features = tl.arange(0, E)
    row_start = row_tile * TILE
    query_pos = block_start + row_start + rows
    q = tl.load(Q + bh * N * D + query_pos[:, None] * D + features[None, :],
                mask=query_pos[:, None] < N, other=0.0)
    if not NATIVE_QK:
        q = q.to(tl.float32)
    slope = tl.load(Slopes + head).to(tl.float32)
    accumulator = tl.zeros((TILE, E), tl.float32)

    # Merge complete preceding tiles into wider rectangles. A short remainder
    # keeps the grouping exact: no padded future rectangle is multiplied.
    full_groups = row_start // COLUMN_TILE
    for group in range(full_groups):
        accumulator += _rectangle(
            q, K, V, slope, bh, block_start, row_start, group * COLUMN_TILE,
            N, D, E, TILE, COLUMN_TILE, ANCHOR, NATIVE_QK, LATE_VALUE,
        )
    remainder_start = full_groups * COLUMN_TILE
    for remainder in range((row_start - remainder_start) // TILE):
        accumulator += _rectangle(
            q, K, V, slope, bh, block_start, row_start,
            remainder_start + remainder * TILE,
            N, D, E, TILE, TILE, ANCHOR, NATIVE_QK, LATE_VALUE,
        )

    # The diagonal tile uses relative distances directly, avoiding any
    # large positive exponent and covering arbitrary final-block lengths.
    key_pos = block_start + row_start + cols
    key = tl.load(K + bh * N * D + features[:, None] + key_pos[None, :] * D,
                  mask=key_pos[None, :] < N, other=0.0)
    if not LATE_VALUE:
        value = tl.load(V + bh * N * E + key_pos[:, None] * E + output_features[None, :],
                        mask=key_pos[:, None] < N, other=0.0)
    if not NATIVE_QK:
        key = key.to(tl.float32)
    distance = rows[:, None] - cols[None, :]
    exponent = tl.where(distance >= 0, -slope * distance.to(tl.float32), float("-inf"))
    score = tl.dot(q, key) * tl.exp(exponent)
    if LATE_VALUE:
        value = tl.load(V + bh * N * E + key_pos[:, None] * E + output_features[None, :],
                        mask=key_pos[:, None] < N, other=0.0)
    accumulator += tl.dot(score, value.to(tl.float32))
    tl.store(Out + bh * N * E + query_pos[:, None] * E + output_features[None, :],
             accumulator.to(Out.dtype.element_ty), mask=query_pos[:, None] < N)


def _validate_metadata(q, k, v, slopes, out):
    import torch

    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected q/k BHND and v BHNE with matching B/H/N")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("Supported input dtypes: bfloat16, float16, float32")
    if k.dtype != q.dtype or v.dtype != q.dtype or slopes.dtype != torch.float32:
        raise TypeError("q/k/v must share dtype and slopes must be float32")
    if slopes.shape != (q.shape[1],) or q.shape[-1] not in (32, 64, 128) or v.shape[-1] not in (32, 64, 128):
        raise ValueError("Expected [H] slopes and D/E in {32, 64, 128}")
    if q.shape[0] <= 0 or q.shape[1] <= 0:
        raise ValueError("B and H must be positive")
    if q.device.type not in ("cuda", "npu"):
        raise ValueError("Triton kernels require a CUDA or Ascend NPU device")
    if any(x.device != q.device or not x.is_contiguous() for x in (q, k, v, slopes)):
        raise ValueError("All inputs must be contiguous on the same device")
    if out is not None and (out.shape != v.shape or out.dtype != v.dtype or out.device != v.device
                            or not out.is_contiguous()):
        raise ValueError("out must match v's shape, dtype, device and contiguous layout")


def baseline(q, k, v, slopes, out=None, tile=32):
    """Launch the unchanged vLLM kernel; tile=32 is the pinned baseline."""
    import torch
    from .baseline_kernel import _fwd_diag_kernel

    _validate_metadata(q, k, v, slopes, out)
    if tile not in (32, 64, 128):
        raise ValueError("Baseline tile must be 32, 64 or 128")
    out = torch.empty_like(v) if out is None else out
    b, h, n, d = q.shape
    if n == 0:
        return out
    blocks = triton.cdiv(n, 256)
    _fwd_diag_kernel[(b * h * blocks, 256 // tile)](
        q, k, v, out, slopes, b, h, n, d, v.shape[-1],
        BLOCK=256, NUM_BLOCK=blocks, CBLOCK=tile,
    )
    return out


def configuration(q, v):
    """Choose a measured configuration using shape/dtype metadata only."""
    import torch
    low_precision_input = q.dtype in (torch.bfloat16, torch.float16)
    if q.shape[2] < 64:
        tile = 32
    elif (q.shape[-1] > 64 or v.shape[-1] > 64) and (not low_precision_input or q.shape[2] < 256):
        tile = 32
    elif low_precision_input and q.shape[2] >= 256 and q.shape[-1] <= 64 and v.shape[-1] <= 64:
        tile = 128
    else:
        tile = 64
    engine = "custom"
    if q.dtype == torch.float32 and q.shape[2] >= 256 and q.shape[2] % 256 == 0 and q.shape[-1] <= 64 and v.shape[-1] <= 64:
        engine = "vllm_fp32_64"
    return {"tile": tile, "column_tile": tile, "native_qk": low_precision_input,
            "anchor": False, "interleave": True, "late_value": True,
            "prune_tail": True, "engine": engine}


def optimized(q, k, v, slopes, out=None, tile=None, column_tile=None,
              anchor=False, native_qk=None, interleave=True, late_value=True,
              prune_tail=True):
    """Compute the block-local causal forward with individually ablatable changes.

    Preconditions: finite inputs and finite nonnegative float32 slopes.
    Numerical contracts and adversarial cases live in the validation runner.
    """
    import torch

    _validate_metadata(q, k, v, slopes, out)
    low_precision_input = q.dtype in (torch.bfloat16, torch.float16)
    selected = configuration(q, v)
    if native_qk is None:
        native_qk = selected["native_qk"]
    if tile is None:
        tile = selected["tile"]
    if column_tile is None:
        column_tile = tile
    if tile not in (32, 64, 128):
        raise ValueError("tile must be 32, 64 or 128")
    if column_tile not in (tile, 128) or column_tile % tile:
        raise ValueError("column_tile must equal tile or 128")
    out = torch.empty_like(v) if out is None else out
    b, h, n, d = q.shape
    if n == 0:
        return out
    blocks = triton.cdiv(n, 256)
    tasks_per_head = triton.cdiv(n, tile) if prune_tail else blocks * (256 // tile)
    _optimized_diag[(b * h * tasks_per_head,)](
        q, k, v, slopes, out, h, n, d, v.shape[-1],
        BLOCK=256, NUM_BLOCKS=blocks, TILE=tile,
        COLUMN_TILE=column_tile, ANCHOR=anchor,
        NATIVE_QK=native_qk and low_precision_input,
        INTERLEAVE=interleave, LATE_VALUE=late_value, PRUNE_TAIL=prune_tail,
    )
    return out


def run(q, k, v, slopes, out=None):
    """Production entry: custom low-precision/tail paths and a tuned FP32 path.

    The complete-block FP32 path reuses the unchanged vLLM kernel with a
    measured 64-row tile; this reuse is recorded in configuration/results.
    """
    _validate_metadata(q, k, v, slopes, out)
    if configuration(q, v)["engine"] == "vllm_fp32_64":
        return baseline(q, k, v, slopes, out=out, tile=64)
    return optimized(q, k, v, slopes, out=out)
