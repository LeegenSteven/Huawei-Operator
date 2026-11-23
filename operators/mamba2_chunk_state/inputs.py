"""Fixed cases for the forward chunk state operator, generated on the CPU."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    length: int
    heads: int
    groups: int
    headdim: int
    dstate: int
    chunk: int
    dtype: str = "bfloat16"
    labels: str = "none"
    states_in_fp32: bool = True


CASES = (
    Case("one_chunk", 1, 64, 8, 2, 64, 64, 64),
    Case("many_heads", 1, 256, 32, 4, 64, 64, 64),
    Case("prefill1k", 1, 1024, 16, 4, 64, 128, 128),
    Case("prefill4k", 1, 4096, 16, 4, 64, 128, 256),
    Case("batch2", 2, 512, 8, 2, 64, 128, 128),
    Case("chunk32", 1, 256, 8, 1, 32, 64, 32),
    Case("shared_b", 1, 1024, 32, 1, 64, 64, 128),
    Case("per_head_b", 1, 512, 8, 8, 64, 64, 128),
    Case("tail129", 1, 129, 8, 2, 64, 128, 128),
    Case("short13", 1, 13, 8, 1, 32, 64, 128),
    Case("packed", 2, 300, 8, 2, 64, 64, 64, labels="packed"),
    Case("pad_last", 1, 513, 8, 4, 64, 64, 128, labels="padded"),
    Case("odd_dims", 1, 97, 6, 2, 40, 48, 48),
    Case("fp16", 1, 512, 8, 2, 64, 128, 128, dtype="float16"),
    Case("fp32", 1, 256, 8, 4, 64, 64, 64, dtype="float32"),
    Case("lowp_out", 1, 511, 8, 2, 64, 64, 128, states_in_fp32=False),
)


def make_inputs(case, seed, device):
    import torch

    b, t, h, g, p, n, c = (case.batch, case.length, case.heads, case.groups,
                            case.headdim, case.dstate, case.chunk)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, case.dtype)
    coeff = torch.randn((b, t, g, n), generator=generator).to(dtype)
    x = torch.randn((b, t, h, p), generator=generator).to(dtype)
    dt = torch.rand((b, h, (t+c-1)//c, c), generator=generator) * 0.16 + 0.04
    if t % c:
        dt[:, :, -1, t % c:] = 0
    rate = -torch.linspace(0.1, 1.4, h).view(1, h, 1, 1)
    prefix = torch.cumsum(dt * rate, dim=-1)
    ids = None
    if case.labels != "none":
        positions = torch.arange(t)[None, :] + 7 * torch.arange(b)[:, None]
        ids = (positions // max(3, c // 3)).to(torch.int32)
        if case.labels == "padded":
            ids[:, -min(t, c//2):] = -1
    return tuple(v.to(device) if v is not None else None for v in (coeff, x, dt, prefix, ids))
