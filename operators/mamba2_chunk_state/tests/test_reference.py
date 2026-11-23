import ast
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from operators.mamba2_chunk_state.reference import reference, round_operand


class ReferenceTests(unittest.TestCase):
    def test_group_and_chunk_boundaries(self):
        x = np.array([[[[1], [2]], [[3], [4]], [[5], [6]]]], np.float32)
        coeff = np.array([[[[1], [7]], [[2], [8]], [[3], [9]]]], np.float32)
        dt = np.ones((1, 2, 2, 2), np.float32)
        observed = reference(coeff, x, dt, np.zeros_like(dt), None, "float32")
        np.testing.assert_array_equal(observed.ravel(), [7, 46, 15, 54])

    def test_padded_endpoint_repeated_and_negative_ids(self):
        x = np.ones((1, 4, 1, 1), np.float32)
        coeff = np.array([1, 300, 5, 9], np.float32).reshape(1, 4, 1, 1)
        dt = np.ones((1, 1, 1, 6), np.float32)
        a = np.zeros_like(dt)
        a[..., -1] = -np.log(2)
        ids = np.array([[7, 8, 7, 7]], np.int32)
        np.testing.assert_allclose(reference(coeff, x, dt, a, ids, "float32"), 7.5)
        ids[0, -1] = -1
        np.testing.assert_array_equal(reference(coeff, x, dt, a, ids, "float32"), 0)
        a[..., -1] = 10
        np.testing.assert_array_equal(reference(coeff, x, dt, a, None, "float32"), 315)

    def test_weighted_operand_rounds_before_reduction(self):
        # BF16 ties around 1 round to the even mantissa, on each side.
        values = np.array([1+2**-8, 1+3*2**-8], np.float32)
        np.testing.assert_array_equal(round_operand(values, "bfloat16"), [1, 1+2**-6])
        coeff = np.ones((1, 2, 1, 1), np.float32)
        x = np.array([1, -1], np.float32).reshape(coeff.shape)
        dt = values.reshape(1, 1, 1, 2)
        np.testing.assert_array_equal(reference(coeff, x, dt, np.zeros_like(dt), None, "bfloat16"), -(2**-6))

    def test_pinned_function_and_all_launch_configs(self):
        root = Path(__file__).resolve().parents[1]
        meta = json.loads((root/"third_party/source.json").read_text())
        upstream = (root/"third_party/mamba/ssd_chunk_state.py").read_bytes()
        self.assertEqual(hashlib.sha256(upstream).hexdigest(), meta["source_sha256"])
        self.assertEqual(hashlib.sha256((root/"third_party/mamba/LICENSE").read_bytes()).hexdigest(), meta["license_sha256"])
        def locate(source):
            return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_chunk_state_fwd_kernel")
        source = upstream.decode()
        original = locate(source)
        current = (root/"baseline_kernel.py").read_text()
        expected = ast.get_source_segment(source, original)
        self.assertEqual(ast.get_source_segment(current, locate(current)), expected)
        self.assertEqual(hashlib.sha256(expected.encode()).hexdigest(), meta["function_sha256"])
        configs = next(n for n in ast.walk(original.decorator_list[0])
                       if isinstance(n, ast.List) and isinstance(n.elts[0], ast.Call))
        parsed = []
        for config in configs.elts:
            params = ast.literal_eval(config.args[0])
            launch = {kw.arg: ast.literal_eval(kw.value) for kw in config.keywords}
            parsed.append([params["BLOCK_SIZE_M"], params["BLOCK_SIZE_N"], params["BLOCK_SIZE_K"], launch["num_stages"], launch["num_warps"]])
        self.assertEqual(meta["configs"], parsed)
