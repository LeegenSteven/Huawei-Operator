"""Published shapes and distinct adversarial index patterns."""

from dataclasses import dataclass

from .baseline import make_custom_inputs


@dataclass(frozen=True)
class Case:
    name: str
    b: int
    q: int
    s: int
    h: int = 128
    k: int = 2048
    dtype_code: int = 0
    pattern: str = "public"

    def axes(self):
        return dict(B=self.b, S1=self.q, S2=self.s, N1=self.h, K=self.k, dtype_code=self.dtype_code)


CASES = (
    Case("sfa-001", 1, 1, 2048),
    Case("sfa-002", 1, 1, 4096),
    Case("decode8k", 1, 1, 8192),
    Case("decode16k", 1, 1, 16384),
    Case("decode_batch8", 8, 1, 8192),
    Case("prefill1k", 1, 1024, 1024),
    Case("prefill2k", 1, 2048, 2048),
    Case("decode_batch4", 4, 1, 1024),
    Case("heads8", 1, 1, 8192, 8),
    Case("heads32", 1, 1, 8192, 32),
    Case("heads64", 1, 1, 8192, 64),
    Case("k256", 1, 1, 512, 128, 256),
    Case("k512", 1, 1, 8192, 128, 512),
    Case("odd_q37", 1, 37, 1000, 16, 128),
    Case("batch5", 5, 1, 1024),
    Case("fp16_8k", 1, 1, 8192, dtype_code=1),
    Case("mixed_indices", 2, 7, 193, 8, 97, pattern="mixed"),
)
NATIVE_CASES = ("decode8k", "decode16k", "prefill2k", "decode_batch8")


def make(case, device, shared_value=False):
    import torch

    data = make_custom_inputs(case.axes(), device)
    if case.pattern == "mixed":
        # Alternate invalid slots, duplicates, a zero-only row and an entirely
        # masked row. Values are independent from keys, including for repeats.
        ids = data["sparse_indices"].cpu()
        ids[..., 1::3] = ids[..., :1]
        ids[..., 2::5] = -1
        ids[..., 4::11] = case.s + 3
        ids[:, 0] = -1
        ids[:, 1] = 0
        data["sparse_indices"] = ids.to(device)
    if shared_value:
        data["value"] = data["key"]
    return data
