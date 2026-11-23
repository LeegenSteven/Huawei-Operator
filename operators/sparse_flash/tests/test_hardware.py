"""Regressions for the token index and normalization semantics on Ascend."""

import unittest

import torch
import torch_npu  # noqa: F401

from operators.sparse_flash import baseline, kernel
from operators.sparse_flash.inputs import Case, make
from operators.sparse_flash.reference import STRICT, compare_all, fp64_rows


class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.npu.is_available():
            raise RuntimeError("An Ascend NPU is required")
        torch.npu.set_device(0)
        torch.set_num_threads(1)

    def same(self, x, fn=kernel.run):
        out = fn(**x)
        ref = baseline.run(**x)
        a, r = STRICT[x["query"].dtype]
        torch.testing.assert_close(out.cpu(), ref.cpu(), atol=a, rtol=r)
        self.assertEqual(out.dtype, x["query"].dtype)
        self.assertEqual(out.shape, x["query"].shape)
        return out

    def test_small_heads_and_fp16(self):
        for h in (1, 2, 4, 8, 16):
            with self.subTest(heads=h):
                x = make(Case("small", 1, 3, 31, h, 24, 1), "npu:0")
                self.same(x)
                self.assertTrue(fp64_rows(x, kernel.run(**x))["passed"])

    def test_all_invalid_and_repeated_token(self):
        x = make(Case("invalid", 2, 3, 129, 8, 97, pattern="mixed"), "npu:0")
        ids = x["sparse_indices"].cpu()
        ids[:, 0, 0, ::2] = -11
        ids[:, 0, 0, 1::2] = 129
        x["sparse_indices"] = ids.npu()
        out = self.same(x).cpu()
        self.assertTrue(torch.equal(out[:, 0], torch.zeros_like(out[:, 0])))
        self.assertTrue(torch.equal(out[:, 1], x["value"][:, 0].cpu().expand_as(out[:, 1])))

    def test_multiplicity_and_order(self):
        x = make(Case("repeat", 1, 2, 64, 4, 192), "npu:0")
        ids = torch.full((1, 2, 1, 192), -1, dtype=torch.int32)
        ids[:, 0, 0, :4] = torch.tensor([4, 4, 4, 12])
        ids[:, 1, 0, :4] = torch.tensor([7, 12, 12, 12])
        x.update(sparse_indices=ids.npu(), scale_value=0.0)
        out = self.same(x).cpu()
        v = x["value"].float().cpu()
        expected = ((3 * v[0, 4, 0] + v[0, 12, 0]) / 4).to(out.dtype)
        torch.testing.assert_close(out[0, 0, 0], expected, atol=0, rtol=0)
        x["sparse_indices"] = ids.flip(-1).npu()
        self.assertTrue(torch.equal(kernel.run(**x).cpu(), out))

    def test_rope_only_with_negative_scale(self):
        x = make(Case("rope", 1, 3, 223, 4, 128), "npu:0")
        x["query"].zero_()
        x["key"].zero_()
        x["scale_value"] = -0.37
        self.same(x)

    def test_extreme_logits_and_empty_tiles(self):
        x = make(Case("extreme", 1, 2, 513, 4, 256, 1), "npu:0")
        x["query"].mul_(32)
        ids = x["sparse_indices"].cpu()
        ids[:, 0, :, :128] = -1
        ids[:, 1] = 900
        x["sparse_indices"] = ids.npu()
        self.same(x)

    def test_independent_and_shared_value(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                x = make(Case("alias", 1, 1, 521, 32, 256), "npu:0", shared)
                self.same(x)

    def test_large_finite_values_on_short_path(self):
        x = make(Case("large_values", 1, 1, 32, 2, 32), "npu:0")
        x["query"].zero_()
        x["query_rope"].zero_()
        x["value"].fill_(1e38)
        out = self.same(x)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_storage_offsets(self):
        x = make(Case("offset", 1, 3, 256, 8, 128), "npu:0")
        for name, t in list(x.items()):
            if torch.is_tensor(t):
                buf = torch.empty(t.numel() + 16, device=t.device, dtype=t.dtype)
                view = buf[16:].view_as(t)
                view.copy_(t)
                x[name] = view
        self.same(x)

    def test_graph_replay_reads_current_indices(self):
        for case in (
            Case("counts", 1, 4, 64, 8, 256),
            Case("sparse", 1, 1, 519, 16, 256),
            Case("one_tile", 1, 3, 155, 8, 128),
        ):
            with self.subTest(case=case.name):
                x = make(case, "npu:0")
                self.same(x)
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g):
                    out = kernel.run(**x)
                x["sparse_indices"].fill_(2)
                g.replay()
                torch.npu.synchronize()
                target = x["value"][:, 2:3].expand_as(x["query"])
                self.assertTrue(torch.equal(out, target))
                x["sparse_indices"].fill_(case.s)
                g.replay()
                torch.npu.synchronize()
                self.assertEqual(out.count_nonzero().item(), 0)

    def test_shape_dtype_and_layout_rejection(self):
        x = make(Case("bad", 1, 2, 75, 8, 32), "npu:0")
        for name, value, error in (
            ("sparse_indices", x["sparse_indices"].long(), TypeError),
            ("query", x["query"].transpose(1, 2), ValueError),
            ("scale_value", float("inf"), ValueError),
            ("value", x["value"][:, :, :, :64], ValueError),
        ):
            with self.subTest(name=name), self.assertRaises(error):
                kernel.run(**dict(x, **{name: value}))

    def test_equal_density_with_holes_and_repeats(self):
        x = make(Case("prefill", 1, 4, 256, 128, 256), "npu:0")
        ids = x["sparse_indices"].cpu()
        ids[:, 1] = -1
        ids[
            :,
            2,
            :,
            :,
        ] = 33
        ids[:, 3, 0, ::3] = ids[:, 3, 0, :1].clone()
        x["sparse_indices"] = ids.npu()
        self.assertTrue(compare_all(kernel.run(**x), x)["passed"])


if __name__ == "__main__":
    unittest.main()
