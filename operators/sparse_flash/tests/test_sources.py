"""Verify the published reference and workload definitions without accelerator dependencies."""

import ast
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party/public_examples"


class SourceTests(unittest.TestCase):
    def test_pinned_bytes(self):
        source = json.loads((ROOT / "third_party/source.json").read_text())
        for name, expected in source["sha256"].items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((VENDOR / name).read_bytes()).hexdigest(), expected)

    def test_public_reference_is_exact(self):
        definition = json.loads((VENDOR / "definition.json").read_text())
        self.assertEqual(
            (ROOT / "baseline.py").read_text().rstrip("\n"), definition["reference"].rstrip("\n")
        )
        self.assertIn(
            "run", {n.name for n in ast.parse(definition["reference"]).body if isinstance(n, ast.FunctionDef)}
        )

    def test_public_workloads(self):
        entries = [json.loads(line) for line in (VENDOR / "workload.jsonl").read_text().splitlines()]
        self.assertEqual(
            [(r["uuid"], r["axes"]["S2"]) for r in entries], [("sfa-001", 2048), ("sfa-002", 4096)]
        )
        self.assertTrue(
            all(r["axes"]["K"] == 2048 and r["tolerance"]["required_matched_ratio"] == 0.995 for r in entries)
        )


if __name__ == "__main__":
    unittest.main()
