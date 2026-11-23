"""Fixed-length token-indexed sparse Flash attention on Ascend."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _multiplicities(I, Counts, R0, S: tl.constexpr, K: tl.constexpr, BIN: tl.constexpr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, WIDTH)
    idx = tl.load(I + (R0 + row) * K + pos, pos < K, other=-1)
    hist = tl.histogram(tl.where((idx >= 0) & (idx < S), idx, BIN), BIN)
    key = tl.arange(0, BIN)
    tl.store(Counts + row * S + key, hist, key < S)


@triton.jit
def _pack(
    I,
    Keys,
    Values,
    Rope,
    KP,
    VP,
    RP,
    R0,
    QLEN: tl.constexpr,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    orig = R0 + row
    idx = tl.load(I + orig * K + slot, slot < K, other=-1)
    valid = (idx >= 0) & (idx < S)
    safe = tl.where(valid, idx, 0) + (orig // QLEN) * S
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    k = tl.load(Keys + safe[:, None] * 512 + d[None, :])
    v = tl.load(Values + safe[:, None] * 512 + d[None, :])
    rope = tl.load(Rope + safe[:, None] * 64 + r[None, :])
    dest = row * K + slot
    tl.store(KP + dest[:, None] * 512 + d[None, :], k, (slot < K)[:, None])
    tl.store(VP + dest[:, None] * 512 + d[None, :], v, (slot < K)[:, None])
    tl.store(RP + dest[:, None] * 64 + r[None, :], rope, (slot < K)[:, None])


@triton.jit
def _score(
    Q,
    QR,
    Keys,
    KR,
    V,
    VP,
    I,
    Score,
    R0,
    Scale,
    QLEN: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    PACK_V: tl.constexpr,
    SHARED: tl.constexpr,
):
    block = tl.program_id(0)
    h = tl.program_id(1) * BM + tl.arange(0, BM)
    row = tl.program_id(2)
    qbase = ((R0 + row) * H + h) * 512
    source = (R0 + row) // QLEN if PHYSICAL else row
    slot = block * BN + tl.arange(0, BN)
    token = source * T + slot
    if PACK_V:
        index = tl.load(I + (R0 + row) * T + slot, slot < T, other=-1)
        token = (R0 + row) // QLEN * S + tl.where((index >= 0) & (index < S), index, 0)
    kbase = token * 512
    d = tl.arange(0, BD)
    accum = tl.full((BM, BN), 0.0, tl.float32)
    for step in range(512 // BD):
        q = tl.load(Q + qbase[:, None] + step * BD + d[None, :], (h < H)[:, None], other=0.0)
        k = tl.load(Keys + kbase[:, None] + step * BD + d[None, :], (slot < T)[:, None], other=0.0)
        if PACK_V:
            if tl.program_id(1) == 0:
                if SHARED:
                    val = k
                else:
                    val = tl.load(V + kbase[:, None] + step * BD + d[None, :], (slot < T)[:, None], other=0.0)
                tl.store(
                    VP + (row * T + slot[:, None]) * 512 + step * BD + d[None, :], val, (slot < T)[:, None]
                )
        accum += tl.dot(q, tl.trans(k))
    r = tl.arange(0, 64)
    qr = tl.load(QR + ((R0 + row) * H + h[:, None]) * 64 + r[None, :], (h < H)[:, None], other=0.0)
    kr = tl.load(KR + token[:, None] * 64 + r[None, :], (slot < T)[:, None], other=0.0)
    accum += tl.dot(qr, tl.trans(kr))
    dest = (row * H + h[:, None]) * T + slot[None, :]
    logits = accum * Scale
    tl.store(Score + dest, logits, (h < H)[:, None] & (slot < T)[None, :])


@triton.jit
def _prob(
    Score,
    I,
    Counts,
    R0,
    H: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    WIDTH: tl.constexpr,
    PHYSICAL: tl.constexpr,
):
    rh = tl.program_id(0)
    row = rh // H
    col = tl.arange(0, WIDTH)
    logits = tl.load(Score + rh * T + col, col < T, other=0.0)
    if PHYSICAL:
        count = tl.load(Counts + row * T + col, col < T, other=0).to(tl.float32)
    else:
        idx = tl.load(I + (R0 + row) * T + col, col < T, other=-1)
        count = ((idx >= 0) & (idx < S)).to(tl.float32)
    valid = count > 0
    top = tl.max(tl.where(valid, logits, -float("inf")), 0)
    top = tl.where(top == -float("inf"), 0.0, top)
    p = tl.exp(tl.where(valid, logits - top, -float("inf"))) * count
    denom = tl.sum(p, 0)
    normalized = tl.where(denom > 0, p / tl.where(denom > 0, denom, 1.0), 0.0)
    tl.store(Score + rh * T + col, normalized, col < T)


@triton.jit
def _grouped_prob(Score, Counts, H: tl.constexpr, S: tl.constexpr, BH: tl.constexpr, WIDTH: tl.constexpr):
    # Read one histogram for BH adjacent heads, retaining distinct maxima and sums.
    row = tl.program_id(1)
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    col = tl.arange(0, WIDTH)
    count = tl.load(Counts + row * S + col, col < S, other=0).to(tl.float32)
    address = (row * H + h[:, None]) * S + col[None, :]
    active = (h < H)[:, None] & (col < S)[None, :]
    logits = tl.load(Score + address, active, other=0.0)
    masked = tl.where(count[None, :] > 0, logits, -float("inf"))
    maximum = tl.max(masked, 1)
    p = tl.exp(masked - tl.where(maximum == -float("inf"), 0.0, maximum)[:, None]) * count[None, :]
    denom = tl.sum(p, 1)
    normalized = tl.where((denom > 0)[:, None], p / tl.where(denom > 0, denom, 1.0)[:, None], 0.0)
    tl.store(Score + address, normalized, active)


@triton.jit
def _product(
    Score,
    V,
    Out,
    R0,
    QLEN: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    PHYSICAL: tl.constexpr,
    BM: tl.constexpr,
    BV: tl.constexpr,
    KT: tl.constexpr,
):
    d = tl.program_id(0) * BV + tl.arange(0, BV)
    h = tl.program_id(1) * BM + tl.arange(0, BM)
    row = tl.program_id(2)
    orig = R0 + row
    source = orig // QLEN if PHYSICAL else row
    t = tl.arange(0, KT)
    acc = tl.full((BM, BV), 0, tl.float32)
    for j in range(triton.cdiv(T, KT)):
        col = j * KT + t
        p = tl.load(
            Score + (row * H + h[:, None]) * T + col[None, :],
            (h < H)[:, None] & (col < T)[None, :],
            other=0.0,
        )
        v = tl.load(V + (source * T + col[:, None]) * 512 + d[None, :], (col < T)[:, None], other=0.0).to(
            tl.float32
        )
        acc += tl.dot(p, v)
    tl.store(Out + (orig * H + h[:, None]) * 512 + d[None, :], acc, (h < H)[:, None])


@triton.jit
def _one_tile(
    Q,
    QR,
    Keys,
    KR,
    Values,
    I,
    Out,
    Scale,
    QLEN: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BV: tl.constexpr,
):
    head = tl.program_id(0) * BM + tl.arange(0, BM)
    row = tl.program_id(1)
    slot = tl.arange(0, BN)
    index = tl.load(I + row * K + slot, slot < K, other=-1)
    valid = (index >= 0) & (index < S)
    token = (row // QLEN) * S + tl.where(valid, index, 0)
    query_head = row * H + head

    d = tl.arange(0, 512)
    q = tl.load(Q + query_head[:, None] * 512 + d[None, :], (head < H)[:, None], other=0.0)
    k = tl.load(Keys + token[:, None] * 512 + d[None, :])
    r = tl.arange(0, 64)
    qr = tl.load(QR + query_head[:, None] * 64 + r[None, :], (head < H)[:, None], other=0.0)
    kr = tl.load(KR + token[:, None] * 64 + r[None, :])
    logits = (tl.dot(q, tl.trans(k)) + tl.dot(qr, tl.trans(kr))) * Scale
    masked = tl.where(valid[None, :], logits, -float("inf"))
    maximum = tl.max(masked, 1)
    weights = tl.exp(masked - tl.where(maximum == -float("inf"), 0.0, maximum)[:, None])
    total = tl.sum(weights, 1)
    weights = tl.where((total > 0)[:, None], weights / tl.where(total > 0, total, 1.0)[:, None], 0.0)
    for stripe in range(512 // BV):
        channel = stripe * BV + tl.arange(0, BV)
        v = tl.load(Values + token[:, None] * 512 + channel[None, :]).to(tl.float32)
        out = tl.dot(weights, v)
        tl.store(Out + query_head[:, None] * 512 + channel[None, :], out, (head < H)[:, None])


def _check(query, key, value, idx, scale, qr, kr):
    if query.ndim != 4 or key.ndim != 4 or idx.ndim != 4:
        raise ValueError("Expected contiguous BSND tensors and BQ1K indices")
    b, q, h, d = query.shape
    s = key.shape[1]
    if min(b, q, h, s, idx.shape[-1]) < 1 or h > 128 or d != 512:
        raise ValueError("Expected positive dimensions, H<=128 and content dimension 512")
    if (
        key.shape != (b, s, 1, 512)
        or value.shape != key.shape
        or qr.shape != (b, q, h, 64)
        or kr.shape != (b, s, 1, 64)
        or idx.shape[:3] != (b, q, 1)
    ):
        raise ValueError("Incompatible BSND, RoPE or index dimensions")
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or any(v.dtype != query.dtype for v in (key, value, qr, kr))
        or idx.dtype != torch.int32
    ):
        raise TypeError("Expected uniform BF16/FP16 inputs and int32 indices")
    if query.device.type != "npu" or any(
        v.device != query.device or not v.is_contiguous() for v in (query, key, value, idx, qr, kr)
    ):
        raise ValueError("Inputs must be contiguous on one NPU")
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale_value must be a finite scalar")


# Each tuple is (heads, score tokens, QK channels, output channels, PV tokens).
TILES = {
    "narrow": (32, 64, 128, 64, 64),
    "wide": (64, 128, 512, 128, 128),
    "large": (128, 256, 512, 256, 128),
}
SCHEDULES = {"gather_wide", "pack_wide", "pack_large", "pack_narrow", "counts_large", "counts_single"}


def configuration(q, k, idx, mode="auto"):
    batch, queries, heads, _ = q.shape
    length, slots = k.shape[1], idx.shape[-1]
    if mode == "auto":
        if slots <= 128 and (slots & (slots - 1)) == 0:
            mode = "one_tile"
        elif length <= 4096 and (length < slots or (queries > 1 and length <= slots)):
            mode = "counts_large"
        elif batch * queries >= 4 and heads >= 64 and slots >= 1024:
            mode = "pack_large"
        else:
            mode = "gather_wide"
    if mode == "one_tile" and slots <= 128:
        return dict(
            mode=mode,
            head_tile=min(32, max(16, triton.next_power_of_2(heads))),
            score_tile=max(16, triton.next_power_of_2(slots)),
            value_tile=128,
            scratch_bytes=0,
        )
    if mode not in SCHEDULES:
        raise ValueError("Unsupported schedule for these inputs: " + mode)
    route, size = mode.split("_")
    if route == "counts" and length > 16384:
        raise ValueError("Physical histogram supports at most 16384 bins")
    width = length if route == "counts" else slots
    if route == "counts":
        gather_bytes = length * 4
    else:
        channels = 512 if route == "gather" else 1088
        gather_bytes = slots * channels * q.element_size()
    row_bytes = heads * width * 4 + gather_bytes
    wave = min(batch * queries, max(1, 64 * 1024**2 // row_bytes))
    bm, bn, bd, bv, kt = TILES["large" if size == "single" else size]
    grouped = route == "counts" and size != "single" and queries >= 4 and heads >= 8 and length <= 2048
    return dict(
        mode=mode,
        wave=wave,
        width=width,
        scratch_bytes=wave * row_bytes,
        head_tile=min(bm, max(16, triton.next_power_of_2(heads))),
        score_tile=bn,
        dot_tile=bd,
        value_tile=bv,
        product_k=kt,
        softmax_heads=8 if grouped else 1,
    )


def _launch(query, key, value, sparse_indices, scale_value, query_rope, key_rope, mode):
    batch, queries, heads, _ = query.shape
    length, slots = key.shape[1], sparse_indices.shape[-1]
    cfg = configuration(query, key, sparse_indices, mode)
    out = torch.empty_like(query)
    bm, bn, bv = cfg["head_tile"], cfg["score_tile"], cfg["value_tile"]
    head_blocks = triton.cdiv(heads, bm)
    if cfg["mode"] == "one_tile":
        _one_tile[(head_blocks, batch * queries)](
            query,
            query_rope,
            key,
            key_rope,
            value,
            sparse_indices,
            out,
            scale_value,
            queries,
            length,
            heads,
            slots,
            bm,
            bn,
            bv,
        )
        return out

    route = cfg["mode"].split("_")[0]
    physical, fused_gather = route == "counts", route == "gather"
    wave, width = cfg["wave"], cfg["width"]
    score = torch.empty((wave, heads, width), device=query.device, dtype=torch.float32)
    counts = (
        torch.empty((wave, length), device=query.device, dtype=torch.int32) if physical else sparse_indices
    )
    keys, values, rope = key, value, key_rope
    if not physical:
        values = query.new_empty((wave, slots, 512))
        if not fused_gather:
            keys = torch.empty_like(values)
            rope = query.new_empty((wave, slots, 64))
    shared = fused_gather and key.data_ptr() == value.data_ptr()
    for first in range(0, batch * queries, wave):
        rows = min(wave, batch * queries - first)
        if physical:
            _multiplicities[(rows,)](
                sparse_indices,
                counts,
                first,
                length,
                slots,
                triton.next_power_of_2(length),
                triton.next_power_of_2(slots),
            )
        elif not fused_gather:
            _pack[(triton.cdiv(slots, 8), rows)](
                sparse_indices, key, value, key_rope, keys, values, rope, first, queries, length, slots, 8
            )
        _score[(triton.cdiv(width, bn), head_blocks, rows)](
            query,
            query_rope,
            keys,
            rope,
            value,
            values,
            sparse_indices,
            score,
            first,
            scale_value,
            queries,
            heads,
            width,
            length,
            physical,
            bm,
            bn,
            cfg["dot_tile"],
            fused_gather,
            shared,
        )
        if cfg["softmax_heads"] > 1:
            group = cfg["softmax_heads"]
            _grouped_prob[(triton.cdiv(heads, group), rows)](
                score, counts, heads, width, group, triton.next_power_of_2(width)
            )
        else:
            _prob[(rows * heads,)](
                score,
                sparse_indices,
                counts,
                first,
                heads,
                width,
                length,
                triton.next_power_of_2(width),
                physical,
            )
        _product[(512 // bv, head_blocks, rows)](
            score, values, out, first, queries, heads, width, physical, bm, bv, cfg["product_k"]
        )
    return out


def experiment(query, key, value, sparse_indices, scale_value, query_rope, key_rope, *, mode):
    """Run an explicit schedule for a paired ablation."""
    _check(query, key, value, sparse_indices, scale_value, query_rope, key_rope)
    with torch.npu.device(query.device):
        return _launch(query, key, value, sparse_indices, scale_value, query_rope, key_rope, mode)


def run(query, key, value, sparse_indices, scale_value, query_rope, key_rope):
    """Fixed-length, noncausal token-sparse attention. Return the BSND output."""
    return experiment(query, key, value, sparse_indices, scale_value, query_rope, key_rope, mode="auto")
