"""Independent mathematical references, separate from measured Triton kernels."""

import numpy as np


def numpy_reference(q, k, v, slopes, block=256, accumulation=np.float64):
    """Compute each causal diagonal block with two dense matrix products.

    q/k: [B,H,N,D], v/out: [B,H,N,E], slopes: [H].
    Only keys in the query's own block contribute. No softmax or normalization.
    Float64 accumulation provides an independent numerical oracle.
    """
    q, k, v = (np.asarray(x) for x in (q, k, v))
    slopes = np.asarray(slopes)
    if q.ndim != 4 or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Expected matching BHND q/k and BHNE v")
    if slopes.shape != (q.shape[1],) or block <= 0:
        raise ValueError("Expected one decay slope per head and positive block")
    if not np.isfinite(slopes).all() or (slopes < 0).any():
        raise ValueError("Decay slopes must be finite and nonnegative")
    qa, ka, va = (x.astype(accumulation, copy=False) for x in (q, k, v))
    out = np.empty(v.shape, dtype=accumulation)
    for start in range(0, q.shape[2], block):
        stop = min(start + block, q.shape[2])
        positions = np.arange(stop - start)
        distance = positions[:, None] - positions[None, :]
        weights = np.exp(-slopes.astype(accumulation)[:, None, None]
                         * np.maximum(distance, 0))
        weights *= (distance >= 0)
        score = qa[:, :, start:stop] @ ka[:, :, start:stop].swapaxes(-1, -2)
        out[:, :, start:stop] = (score * weights) @ va[:, :, start:stop]
    return out


def torch_reference(q, k, v, slopes, block=256):
    """Vectorized FP32 semantic reference; never substituted for a Triton kernel."""
    import torch

    out = torch.empty_like(v)
    for start in range(0, q.shape[2], block):
        stop = min(start + block, q.shape[2])
        position = torch.arange(stop - start, device=q.device)
        distance = position[:, None] - position[None, :]
        weight = torch.exp(-slopes.reshape(-1, 1, 1).float()
                           * distance.clamp_min(0).float())
        weight = weight * (distance >= 0)
        scores = q[:, :, start:stop].float() @ k[:, :, start:stop].float().transpose(-1, -2)
        result = (scores * weight) @ v[:, :, start:stop].float()
        out[:, :, start:stop] = result.to(q.dtype)
    return out


def error_metrics(actual, expected, atol, rtol):
    """Report absolute/relative/RMS errors and the complete elementwise verdict."""
    actual, expected = (np.asarray(x, dtype=np.float64) for x in (actual, expected))
    if actual.shape != expected.shape:
        return {"passed": False, "reason": "shape mismatch"}
    finite = np.isfinite(actual).all() and np.isfinite(expected).all()
    diff = np.abs(actual - expected)
    allowed = atol + rtol * np.abs(expected)
    return {
        "passed": bool(finite and np.all(diff <= allowed)),
        "finite": bool(finite),
        "max_abs": float(diff.max(initial=0)),
        "relative_l2": float(np.linalg.norm(diff.ravel()) /
                             max(np.linalg.norm(expected.ravel()), 1e-30)),
        "max_tolerance_ratio": float((diff / np.maximum(allowed, 1e-30)).max(initial=0)),
        "matched_ratio": float(np.mean(diff <= allowed)) if diff.size else 1.0,
        "atol": atol,
        "rtol": rtol,
    }
