"""Independent NumPy expression for grouped chunk states and upstream rounding."""

import numpy as np


def round_operand(a, dtype):
    """Round to nearest even, returning values representable in the given dtype."""
    a = np.asarray(a, dtype=np.float32)
    if dtype == "bfloat16":
        u = a.view(np.uint32)
        rounded = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
        return rounded.view(np.float32)
    return a.astype(dtype).astype(np.float32)


def reference(coeff, x, dt, prefix, ids, dtype, out_dtype="float32"):
    b, length, heads, dim = x.shape
    _, _, groups, state = coeff.shape
    chunks, chunk = dt.shape[2:]
    result = np.empty((b, chunks, heads, dim, state), dtype=np.float64)
    for c in range(chunks):
        start, stop = c*chunk, min((c+1)*chunk, length)
        size = stop-start
        for i in range(b):
            for h in range(heads):
                # The prefix endpoint is the padded slot; labels end at the last real token.
                a = prefix[i, h, c].astype(np.float64)
                w = np.exp(np.minimum(a[-1]-a[:size], 0)) * dt[i, h, c, :size]
                if ids is not None:
                    last = ids[i, stop-1]
                    w *= (last >= 0) & (ids[i, start:stop] == last)
                values = coeff[i, start:stop, h // (heads//groups), :].astype(np.float64)
                weighted = round_operand(values*w[:, None], dtype).astype(np.float64)
                result[i, c, h] = x[i, start:stop, h].astype(np.float64).T @ weighted
    return round_operand(result, out_dtype)


# Frozen before timing. Both implementations must match the independent oracle
# elementwise; the candidate must also match the selected upstream output.
TOLERANCES = {"bfloat16": (0.006, 0.003), "float16": (0.001, 0.001),
              "float32": (0.0005, 0.0001)}


def tolerances(dtype, out_dtype):
    return (0.008, 0.008) if out_dtype == "bfloat16" else TOLERANCES[dtype]
