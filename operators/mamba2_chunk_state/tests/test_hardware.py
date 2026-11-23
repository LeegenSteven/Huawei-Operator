import os
import unittest

import numpy as np

from operators.mamba2_chunk_state.inputs import Case, make_inputs
from operators.mamba2_chunk_state.reference import reference, tolerances


@unittest.skipUnless(os.getenv("MAMBA_RUN_NPU_TESTS") == "1", "Set MAMBA_RUN_NPU_TESTS=1 on Ascend")
class HardwareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from operators.lightning_diag.benchmark import initialize
        from operators.mamba2_chunk_state.kernel import run
        cls.torch, cls.backend = initialize("npu:0")
        cls.run_state = staticmethod(run)

    def check_oracle(self, inputs, states=None, states_in_fp32=True):
        from operators.mamba2_chunk_state.tune_baseline import as_numpy
        result = self.run_state(*inputs, states=states, states_in_fp32=states_in_fp32)
        operand_dtype = str(inputs[1].dtype).split(".")[-1]
        out_dtype = str(result.dtype).split(".")[-1]
        expected = reference(*map(as_numpy, inputs), operand_dtype, out_dtype)
        atol, rtol = tolerances(operand_dtype, out_dtype)
        np.testing.assert_allclose(as_numpy(result), expected, rtol=rtol, atol=atol)
        return result

    def test_tail_endpoint_clamp_and_mixed_types(self):
        torch = self.torch
        case = Case("mixed", 1, 33, 4, 2, 24, 48, 32)
        B, x, dt, prefix, _ = make_inputs(case, 51, "npu:0")
        B = B.float()
        prefix[:, :, -1, -1] = prefix[:, :, -1, 0]-1.5
        prefix[:, :, 0, -1] = 3
        inputs = (B, x, dt.half(), prefix.half(), None)
        out = self.check_oracle(inputs, states_in_fp32=False)
        self.assertEqual(out.dtype, torch.float32)  # states_in_fp32=False selects B.dtype

    def test_packed_repeated_ids_and_negative_last(self):
        torch = self.torch
        case = Case("labels", 1, 44, 4, 1, 32, 32, 16, dtype="float32")
        B, x, dt, a, _ = make_inputs(case, 52, "npu:0")
        ids = torch.tensor([[0, 1, 0, 2]*4 + [1, 2, 1, -1]*4 + [1, 3, 5, 1]*3], dtype=torch.int64, device=x.device)
        result = self.check_oracle((B, x, dt, a, ids))
        self.assertEqual(result[:, 1].abs().max().item(), 0)

    def test_unseen_shapes_and_chunk_sizes(self):
        for length, heads, groups, p, n, c, dtype in [
            (1, 2, 1, 7, 19, 64, "float32"), (127, 4, 4, 48, 16, 64, "bfloat16"),
            (257, 6, 3, 96, 40, 128, "float16"), (100, 3, 1, 17, 31, 1, "float32"),
            (521, 4, 2, 16, 32, 257, "bfloat16"),
        ]:
            with self.subTest(length=length, chunk=c, dtype=dtype):
                case = Case("held_out", 1, length, heads, groups, p, n, c, dtype, labels="packed")
                self.check_oracle(make_inputs(case, 101+length, "npu:0"))

    def test_strided_views_and_output(self):
        torch = self.torch
        case = Case("strided", 1, 67, 4, 2, 32, 32, 32, dtype="float16", labels="packed")
        inputs = make_inputs(case, 53, "npu:0")
        def strided(t):
            backing = torch.empty((*t.shape, 2), dtype=t.dtype, device=t.device)
            view = backing[..., 1]
            view.copy_(t)
            return view
        views = tuple(strided(v) for v in inputs)
        storage = torch.full((1, 3, 4, 32, 32, 2), -123., device="npu:0", dtype=torch.float32)
        out = storage[..., 1]
        self.assertIs(self.check_oracle(views, states=out), out)
        self.assertTrue(bool((storage[..., 0] == -123.).all().item()))

    def test_zero_length_and_invalid_shapes(self):
        torch = self.torch
        case = Case("empty", 1, 0, 4, 2, 16, 32, 32)
        B, x, dt, a, ids = make_inputs(case, 54, "npu:0")
        self.assertEqual(tuple(self.run_state(B, x, dt, a).shape), (1, 0, 4, 16, 32))
        with self.assertRaises(ValueError):
            self.run_state(B, x, dt, a, torch.zeros((2, 0), dtype=torch.int32, device=x.device))
        with self.assertRaises(TypeError):
            self.run_state(B, x, dt.to(torch.int32), a)

    def test_graph_replay_observes_new_inputs(self):
        torch = self.torch
        case = Case("graph", 1, 128, 8, 2, 32, 64, 64)
        B, x, dt, a, ids = make_inputs(case, 55, "npu:0")
        original = self.run_state(B, x, dt, a).clone()
        out = torch.empty_like(original)
        self.backend.synchronize()
        graph = self.backend.NPUGraph()
        with self.backend.graph(graph):
            self.run_state(B, x, dt, a, states=out)
        B.zero_()
        graph.replay()
        self.backend.synchronize()
        self.assertGreater(original.abs().max().item(), 0)
        self.assertEqual(out.abs().max().item(), 0)
