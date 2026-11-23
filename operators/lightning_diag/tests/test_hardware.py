"""Real-device contract checks, enabled explicitly with LIGHTNING_RUN_NPU_TESTS=1."""

import os
import unittest

import numpy as np

from operators.lightning_diag.reference import numpy_reference, error_metrics


@unittest.skipUnless(os.environ.get("LIGHTNING_RUN_NPU_TESTS") == "1", "NPU tests require explicit opt-in")
class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch_npu  # noqa: F401
        from operators.lightning_diag.kernel import run
        if not torch.npu.is_available():
            raise RuntimeError("NPU tests requested but no NPU is available")
        torch.npu.set_device(0)
        torch.set_num_threads(1)
        cls.torch = torch
        cls.run_operator = staticmethod(run)

    def make_inputs(self, tokens, qk_dim=64, value_dim=64, dtype="bfloat16"):
        t = self.torch
        generator = t.Generator(device="cpu").manual_seed(20261123 + tokens)
        q = t.randn(1, 2, tokens, qk_dim, generator=generator).to(getattr(t, dtype))
        k = t.randn(1, 2, tokens, qk_dim, generator=generator).to(getattr(t, dtype))
        v = t.randn(1, 2, tokens, value_dim, generator=generator).to(getattr(t, dtype))
        slopes = t.tensor([0.0, 3.0], dtype=t.float32)
        return tuple(x.npu().contiguous() for x in (q, k, v, slopes))

    def assert_oracle(self, inputs):
        t = self.torch
        oracle = numpy_reference(*(x.float().cpu().numpy() for x in inputs))
        expected = t.from_numpy(oracle).to(inputs[0].dtype).float().numpy()
        actual = self.run_operator(*inputs)
        t.npu.synchronize()
        from operators.lightning_diag.benchmark import TOLERANCES
        dtype = str(inputs[0].dtype).split(".")[-1]
        metrics = error_metrics(actual.float().cpu().numpy(), expected, *TOLERANCES[dtype])
        self.assertTrue(metrics["passed"], metrics)
        self.assertEqual(actual.shape, inputs[2].shape)
        self.assertEqual(actual.dtype, inputs[0].dtype)

    def test_dtypes_dimensions_and_tail_boundaries(self):
        for n, d, e, dtype in [(1, 32, 64, "bfloat16"), (33, 64, 128, "float16"),
                               (65, 64, 64, "float32"), (65, 128, 128, "bfloat16"),
                               (128, 64, 64, "bfloat16"), (256, 64, 64, "float16"),
                               (256, 64, 64, "float32"),
                               (257, 128, 128, "float32"), (257, 128, 32, "bfloat16")]:
            with self.subTest(tokens=n, d=d, e=e, dtype=dtype):
                self.assert_oracle(self.make_inputs(n, d, e, dtype))

    def test_future_and_other_blocks_cannot_change_earlier_outputs(self):
        t = self.torch
        q, k, v, slopes = self.make_inputs(512)
        first = self.run_operator(q, k, v, slopes).clone()
        k[:, :, 64:256].mul_(1000)
        v[:, :, 64:256].mul_(1000)
        second = self.run_operator(q, k, v, slopes).clone()
        t.npu.synchronize()
        self.assertTrue(t.equal(first[:, :, :64], second[:, :, :64]))
        self.assertTrue(t.equal(first[:, :, 256:], second[:, :, 256:]))
        k[:, :, :256].neg_()
        v[:, :, :256].add_(777)
        third = self.run_operator(q, k, v, slopes)
        t.npu.synchronize()
        self.assertTrue(t.equal(second[:, :, 256:], third[:, :, 256:]))

    def test_zero_and_empty_inputs(self):
        t = self.torch
        for n in (0, 65):
            inputs = self.make_inputs(n)
            inputs[0].zero_()
            out = self.run_operator(*inputs)
            t.npu.synchronize()
            self.assertEqual(out.numel(), inputs[2].numel())
            self.assertEqual(t.count_nonzero(out).item(), 0)

    def test_graph_replay_reloads_input_buffers(self):
        t = self.torch
        inputs = self.make_inputs(256)
        original_q = inputs[0].clone()
        out = t.empty_like(inputs[2])
        for _ in range(3):
            self.run_operator(*inputs, out=out)
        t.npu.synchronize()
        expected = out.clone()
        graph = t.npu.NPUGraph()
        with t.npu.graph(graph):
            self.run_operator(*inputs, out=out)
        inputs[0].zero_()
        graph.replay()
        t.npu.synchronize()
        self.assertEqual(t.count_nonzero(out).item(), 0)
        inputs[0].copy_(original_q)
        graph.replay()
        t.npu.synchronize()
        self.assertTrue(t.equal(out, expected))

    def test_fp32_exact_cancellation(self):
        t = self.torch
        inputs = list(self.make_inputs(300, dtype="float32"))
        inputs[0].fill_(1)
        inputs[1].fill_(1)
        inputs[1][:, :, 1::2].neg_()
        inputs[2].fill_(1)
        inputs[3].zero_()
        result = self.run_operator(*inputs).float().cpu().numpy()
        expected = numpy_reference(*(x.cpu().numpy() for x in inputs))
        np.testing.assert_array_equal(result, expected)

    def test_invalid_metadata_is_rejected(self):
        t = self.torch
        q, k, v, slopes = self.make_inputs(256)
        with self.assertRaises(TypeError):
            self.run_operator(q, k, v, slopes.to(t.bfloat16))
        with self.assertRaises(ValueError):
            self.run_operator(q, k, v, slopes, out=t.empty(1, device=q.device))
        with self.assertRaises(ValueError):
            self.run_operator(q.cpu(), k.cpu(), v.cpu(), slopes.cpu())


if __name__ == "__main__":
    unittest.main()
