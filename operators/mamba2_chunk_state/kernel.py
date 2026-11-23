"""Mamba-2 chunk-state forward specialized for contiguous Ascend tensors."""

import triton
import triton.language as tl

from .interface import DEFAULT_CONFIG, launch_upstream, prepare


@triton.jit
def _contribution(X, Delta, Prefix, b_values, valid, ts, ms, head, b, c,
                  T: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
                  C: tl.constexpr, NC: tl.constexpr, ALIGNED: tl.constexpr):
    row = ((b * H + head) * NC + c) * C
    endpoint = tl.load(Prefix + row + C - 1).to(tl.float32)
    prefix = tl.load(Prefix + row + ts, ts < C, other=0.0).to(tl.float32)
    dt = tl.load(Delta + row + ts, ts < C, other=0.0).to(tl.float32)
    scale = tl.where(valid, tl.exp(tl.minimum(endpoint - prefix, 0.0)) * dt, 0.0)
    weighted = (b_values * scale[:, None]).to(X.dtype.element_ty)
    base = X + ((b * T + c * C) * H + head) * P
    in_time = (c * C + ts < T) & (ts < C)
    if ALIGNED:
        x = tl.load(base + ts[:, None] * H * P + ms[None, :])
    else:
        x = tl.load(base + ts[:, None] * H * P + ms[None, :],
                    in_time[:, None] & (ms[None, :] < P), other=0.0)
    return tl.dot(tl.trans(x), weighted)


@triton.jit
def _state_tile(Bcoef, X, Delta, Prefix, IDs, Out,
                T: tl.constexpr, H: tl.constexpr, G: tl.constexpr, P: tl.constexpr,
                N: tl.constexpr, C: tl.constexpr, NC: tl.constexpr, HAS_IDS: tl.constexpr,
                M: tl.constexpr, COL: tl.constexpr, K: tl.constexpr, PAIR: tl.constexpr,
                RANK_ONE_TAIL: tl.constexpr, FULL_TILES: tl.constexpr):
    pid = tl.program_id(0)
    ntiles: tl.constexpr = triton.cdiv(N, COL)
    mtiles: tl.constexpr = triton.cdiv(P, M)
    ns = (pid % ntiles) * COL + tl.arange(0, COL)
    ms = ((pid // ntiles) % mtiles) * M + tl.arange(0, M)
    task = pid // (ntiles * mtiles)
    heads_per_task: tl.constexpr = 2 if PAIR else 1
    aligned: tl.constexpr = FULL_TILES and T % C == 0 and C % K == 0 and P % M == 0 and N % COL == 0
    head = (task % (H // heads_per_task)) * heads_per_task
    bc = task // (H // heads_per_task)
    c = bc % NC
    b = bc // NC
    group = head // (H // G)
    bbase = Bcoef + ((b * T + c * C) * G + group) * N
    real = tl.minimum(T - c * C, C)
    if HAS_IDS:
        idbase = IDs + b * T + c * C
        last_id = tl.load(idbase + real - 1)

    a0 = tl.full((M, COL), 0, tl.float32)
    if PAIR:
        a1 = tl.full((M, COL), 0, tl.float32)
    if RANK_ONE_TAIL and (C == 1 or T == 1):
        single = True
    elif RANK_ONE_TAIL and T % C == 1:
        single = c == NC - 1
    else:
        single = False
    if single:
        # One surviving token gives an outer product. The prefix endpoint still
        # comes from slot C-1, even for an independently supplied padded prefix.
        bv = tl.load(bbase + ns, ns < N, other=0.0).to(tl.float32)
        for j in tl.static_range(heads_per_task):
            off = ((b * H + head + j) * NC + c) * C
            end = tl.load(Prefix + off + C - 1).to(tl.float32)
            begin = tl.load(Prefix + off).to(tl.float32)
            weight = tl.exp(tl.minimum(end - begin, 0.0)) * tl.load(Delta + off).to(tl.float32)
            if HAS_IDS:
                weight = tl.where(last_id >= 0, weight, 0.0)
            weighted = (bv * weight).to(X.dtype.element_ty).to(tl.float32)
            xv = tl.load(X + ((b * T + c * C) * H + head + j) * P + ms,
                         ms < P, other=0.0).to(tl.float32)
            if j == 0:
                a0 = xv[:, None] * weighted[None, :]
            else:
                a1 = xv[:, None] * weighted[None, :]
    else:
        # The group-shared B tile is loaded once, before the two head-specific
        # products. Rounding stays on the weighted B operand for both heads.
        for k in range(triton.cdiv(min(T, C), K)):
            ts = k * K + tl.arange(0, K)
            valid = ts < real
            if aligned:
                bv = tl.load(bbase + ts[:, None] * G * N + ns[None, :]).to(tl.float32)
            else:
                bv = tl.load(bbase + ts[:, None] * G * N + ns[None, :],
                             valid[:, None] & (ns[None, :] < N), other=0.0).to(tl.float32)
            if HAS_IDS:
                ids = tl.load(idbase + ts, valid, other=-1)
                valid = valid & (last_id >= 0) & (ids == last_id)
            a0 += _contribution(X, Delta, Prefix, bv, valid, ts, ms, head, b, c, T, H, P, C, NC, aligned)
            if PAIR:
                a1 += _contribution(X, Delta, Prefix, bv, valid, ts, ms, head + 1, b, c, T, H, P, C, NC, aligned)
    out = Out + ((bc * H + head) * P + ms[:, None]) * N + ns[None, :]
    mask = (ms[:, None] < P) & (ns[None, :] < N)
    tl.store(out, a0, mask)
    if PAIR:
        tl.store(out + P * N, a1, mask)


def configuration(B, x, dt):
    b, length, heads, p = x.shape
    groups, n = B.shape[2:]
    chunks, c = dt.shape[2:]
    m = max(16, min(64, triton.next_power_of_2(p)))
    wide = max(x.element_size(), B.element_size()) == 2
    col = max(16, min(128 if wide else 64, triton.next_power_of_2(n)))
    k = max(16, min(128 if wide else 64,
                    triton.next_power_of_2(max(1, min(length, c)))))
    tasks = b * chunks * heads * triton.cdiv(p, m) * triton.cdiv(n, col)
    return dict(m=m, col=col, k=k,
                pair=wide and col <= 64 and (heads // groups) % 2 == 0 and tasks >= 80,
                rank_one_tail=min(length, c) == 1, full_tiles=True)


def _run(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32, overrides):
    states = prepare(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32)
    b, length, h, p = x.shape
    if not length:
        return states
    inputs = (B, x, dt, dA_cumsum, states) if seq_idx is None else (B, x, dt, dA_cumsum, seq_idx, states)
    if any(not v.is_contiguous() for v in inputs):
        return launch_upstream(B, x, dt, dA_cumsum, seq_idx, states, DEFAULT_CONFIG)
    groups, n = B.shape[2:]
    nc, c = dt.shape[2:]
    config = configuration(B, x, dt) | overrides
    pair = config["pair"]
    if pair and (h // groups) % 2:
        raise ValueError("A pair must lie within a coefficient group")
    m, col, k = (config[key] for key in ("m", "col", "k"))
    grid = (b * nc * (h // (2 if pair else 1)) * triton.cdiv(p, m) * triton.cdiv(n, col),)
    _state_tile[grid](B, x, dt, dA_cumsum, seq_idx, states, length, h, groups, p, n, c, nc,
                      seq_idx is not None, m, col, k, pair, config["rank_one_tail"], config["full_tiles"])
    return states


def run(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    return _run(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32, {})


def experiment(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True, **overrides):
    """Explicit variants for one-factor ablations and development measurements."""
    return _run(B, x, dt, dA_cumsum, seq_idx, states, states_in_fp32, overrides)
