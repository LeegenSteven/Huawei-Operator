"""Independent FP64 oracle and full-output comparisons, outside timed regions."""

import hashlib

import torch

from . import baseline

STRICT = {torch.bfloat16: (0.008, 0.008), torch.float16: (0.002, 0.002)}
QUERY_INPUTS = ("query", "query_rope", "sparse_indices")


def query_slice(inputs, first, last):
    return {
        name: (tensor[:, first:last].contiguous() if name in QUERY_INPUTS else tensor)
        for name, tensor in inputs.items()
    }


def reference_chunks(inputs, size=32):
    q = inputs["query"].shape[1]
    for first in range(0, q, size):
        last = min(q, first + size)
        yield first, last, baseline.run(**query_slice(inputs, first, last))


def chunked_reference(inputs, size=32):
    out = torch.empty_like(inputs["query"])
    for first, last, reference in reference_chunks(inputs, size):
        out[:, first:last].copy_(reference)
    return out


def fingerprint(inputs):
    digest = hashlib.sha256()
    for name in sorted(inputs):
        value = inputs[name]
        digest.update(name.encode())
        if torch.is_tensor(value):
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.detach().contiguous().view(torch.uint8).cpu().numpy().data)
        else:
            digest.update(float(value).hex().encode())
    return digest.hexdigest()


def compare_all(actual, inputs):
    query = inputs["query"]
    if (actual.shape, actual.dtype, actual.device) != (query.shape, query.dtype, query.device):
        raise ValueError("Unexpected output shape, dtype or device")
    abs_tol, rel_tol = STRICT[query.dtype]
    total = strict_matches = public_matches = 0
    max_abs = squared = 0.0
    finite = True
    for first, last, golden in reference_chunks(inputs):
        ref = golden.float().cpu()
        obs = actual[:, first:last].float().cpu()
        diff = (obs - ref).abs()
        finite &= bool(torch.isfinite(diff).all())
        total += diff.numel()
        public_matches += int((diff <= 0.05 + 0.05 * ref.abs()).sum())
        strict_matches += int((diff <= abs_tol + rel_tol * ref.abs()).sum())
        max_abs = max(max_abs, float(diff.max()))
        squared += float(torch.sum(diff.square(), dtype=torch.float64))
    return dict(
        elements=total,
        finite=finite,
        max_abs=max_abs,
        rmse=(squared / total) ** 0.5,
        public_matched=public_matches / total,
        strict_matched=strict_matches / total,
        atol=abs_tol,
        rtol=rel_tol,
        passed=finite and strict_matches == total and public_matches >= 0.995 * total,
    )


def fp64_rows(inputs, actual):
    """Independently filter indices, retain multiplicity, and normalize each CPU row."""
    q = inputs["query"]
    batch, nq, heads, _ = q.shape
    selected_b = sorted({0, batch - 1})
    selected_q = sorted({0, min(1, nq - 1), nq // 2, nq - 1})
    selected_h = list(range(heads)) if heads <= 8 else sorted({0, heads // 2, heads - 1})
    obs_rows, ref_rows, true_rows = [], [], []
    for b in selected_b:
        k, r, v = (inputs[n][b, :, 0].cpu().double() for n in ("key", "key_rope", "value"))
        for pos in selected_q:
            ids = inputs["sparse_indices"][b, pos, 0].cpu().long()
            ids = ids[(ids >= 0) & (ids < k.shape[0])]
            query = q[b, pos].cpu().double()[selected_h]
            rope = inputs["query_rope"][b, pos].cpu().double()[selected_h]
            if ids.numel():
                logits = (query @ k[ids].T + rope @ r[ids].T) * inputs["scale_value"]
                z = (logits - logits.amax(dim=-1, keepdim=True)).exp()
                truth = (z / z.sum(dim=-1, keepdim=True)) @ v[ids]
            else:
                truth = torch.zeros_like(query)
            true_rows.append(truth)
            obs_rows.append(actual[b, pos].cpu().double()[selected_h])
            one = {name: (val[b : b + 1] if torch.is_tensor(val) else val) for name, val in inputs.items()}
            ref_rows.append(baseline.run(**query_slice(one, pos, pos + 1))[0, 0].cpu().double()[selected_h])
    truth, observed, reference = map(torch.cat, (true_rows, obs_rows, ref_rows))

    def errors(x):
        diff = (x - truth).abs()
        relative = diff / (truth.abs() + 1e-7)
        return dict(
            mare=float(relative.max()),
            mere=float(relative.mean()),
            rmse=float(diff.square().mean().sqrt()),
            max_abs=float(diff.max()),
        )

    return dict(
        batch=selected_b,
        queries=selected_q,
        heads=selected_h,
        elements=truth.numel(),
        candidate=errors(observed),
        public_reference=errors(reference),
        passed=torch.allclose(observed, truth, atol=STRICT[q.dtype][0], rtol=STRICT[q.dtype][1]),
    )
