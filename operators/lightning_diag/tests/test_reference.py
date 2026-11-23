"""Independent semantic and numerical checks runnable before NPU setup."""

import ast
import hashlib
import unittest
from pathlib import Path

import numpy as np

from operators.lightning_diag.reference import numpy_reference


class ReferenceTests(unittest.TestCase):
    def test_matches_scalar_formula_with_tail_and_different_dims(self):
        rng = np.random.default_rng(120)
        q = rng.normal(size=(2, 3, 11, 5))
        k = rng.normal(size=q.shape)
        v = rng.normal(size=(2, 3, 11, 7))
        slopes = np.array([0.0, 0.17, 3.0])
        expected = np.zeros(v.shape)
        for b in range(2):
            for h in range(3):
                for i in range(11):
                    for j in range(i // 4 * 4, i + 1):
                        expected[b, h, i] += (q[b, h, i] @ k[b, h, j]) * np.exp(-slopes[h] * (i - j)) * v[b, h, j]
        np.testing.assert_allclose(numpy_reference(q, k, v, slopes, block=4), expected, rtol=1e-12, atol=1e-12)

    def test_future_and_cross_block_values_do_not_leak(self):
        q = k = np.ones((1, 1, 9, 2), dtype=np.float32)
        v = np.ones((1, 1, 9, 3), dtype=np.float32)
        original = numpy_reference(q, k, v, np.zeros(1), block=4)
        v[:, :, 4:] = 1e9
        changed = numpy_reference(q, k, v, np.zeros(1), block=4)
        np.testing.assert_array_equal(original[:, :, :4], changed[:, :, :4])
        v[:, :, :4] = -1e9
        changed_again = numpy_reference(q, k, v, np.zeros(1), block=4)
        np.testing.assert_array_equal(changed[:, :, 4:], changed_again[:, :, 4:])

    def test_local_anchor_remains_finite_for_strong_decay(self):
        for tile in (32, 64):
            r = np.arange(tile)[:, None]
            c = np.arange(tile)[None, :]
            for slope in (0.0, 1e-5, 0.05, 0.3, 3.0, 20.0):
                for previous in range(3):
                    start = 3 * tile
                    direct = np.exp(-slope * (start + r - (previous * tile + c)))
                    factored = (np.exp(-slope * r) * np.exp(-slope * (tile - 1 - c))
                                * np.exp(-slope * (start - (previous * tile + tile - 1))))
                    self.assertTrue(np.isfinite(factored).all())
                    np.testing.assert_allclose(factored, direct, rtol=1e-12, atol=1e-300)

    def test_baseline_is_an_unchanged_upstream_function(self):
        root = Path(__file__).resolve().parents[1]
        def extract(path):
            text = path.read_text()
            node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_fwd_diag_kernel")
            start = min(x.lineno for x in node.decorator_list) - 1
            return "".join(text.splitlines(keepends=True)[start:node.end_lineno])
        original = extract(root / "third_party/vllm/lightning_attn.py")
        self.assertEqual(original, extract(root / "baseline_kernel.py"))
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(), "617e461d8e4fe5bfdef81aed0e40ef975e55795bf894fbb58baf436b56442b2a")


if __name__ == "__main__":
    unittest.main()
