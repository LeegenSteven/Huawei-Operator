"""Fixed workloads and input generation shared by every implementation."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    tokens: int
    qk_dim: int
    value_dim: int
    dtype: str = "bfloat16"
    slope_mode: str = "normal"

    def to_dict(self):
        return asdict(self)


CASES = [
    Case("small", 1, 4, 256, 64, 64),
    Case("batch2", 2, 8, 256, 64, 64),
    Case("two_blocks", 1, 8, 512, 64, 64),
    Case("prefill1k", 1, 16, 1024, 64, 64),
    Case("prefill2k", 1, 8, 2048, 64, 64),
    Case("prefill4k", 1, 8, 4096, 64, 64),
    Case("dim128", 1, 4, 512, 128, 128),
    Case("different_dims", 1, 8, 512, 64, 128),
    Case("tail300", 1, 8, 300, 64, 64),
    Case("tail500", 1, 4, 500, 64, 64),
    Case("batch3", 3, 8, 256, 64, 64),
    Case("fp32", 1, 8, 1024, 64, 64, "float32"),
    Case("short16", 1, 4, 16, 64, 64),
    Case("zero_decay", 1, 4, 512, 64, 64, slope_mode="zero"),
    Case("strong_decay", 1, 4, 512, 64, 64, slope_mode="strong"),
    Case("long8k", 1, 16, 8192, 64, 64),
]


def make_torch_inputs(case, device, seed=20261120):
    """Generate on the CPU, then transfer identical tensors to the tested device."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, case.dtype)
    shape = (case.batch, case.heads, case.tokens)
    q = torch.randn(*shape, case.qk_dim, generator=generator, dtype=torch.float32).to(dtype)
    k = torch.randn(*shape, case.qk_dim, generator=generator, dtype=torch.float32).to(dtype)
    v = torch.randn(*shape, case.value_dim, generator=generator, dtype=torch.float32).to(dtype)
    slopes = torch.rand(case.heads, generator=generator, dtype=torch.float32) * 0.25 + 0.01
    if case.slope_mode == "zero":
        slopes.zero_()
    elif case.slope_mode == "strong":
        slopes.fill_(3.0)
    return tuple(x.to(device).contiguous() for x in (q, k, v, slopes))
