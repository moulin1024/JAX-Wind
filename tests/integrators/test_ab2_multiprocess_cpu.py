from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(
    os.environ.get("WIRELES_RUN_MULTIPROCESS_CPU_TESTS") == "1",
    "set WIRELES_RUN_MULTIPROCESS_CPU_TESTS=1 to enable loopback collectives",
)
class MultiProcessCPUAB2Tests(unittest.TestCase):
    def test_owned_restart_on_one_two_four_processes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runner = root / "tools" / "run_distributed_ab2_cpu.py"
        cases = (
            (1, "float64"),
            (2, "float64"),
            (4, "float64"),
            (2, "float32"),
            (4, "float32"),
        )
        for processes, dtype in cases:
            with self.subTest(processes=processes, dtype=dtype):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "--processes",
                        str(processes),
                        "--dtype",
                        dtype,
                        "--method",
                        "spike",
                        "--steps",
                        "6",
                    ],
                    cwd=root,
                    env=dict(os.environ),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=360,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout)
                self.assertAlmostEqual(result["accepted_clock"][0], 0.12, places=14)
                self.assertEqual(result["accepted_clock"][1], 6)
                self.assertEqual(result["restart_error"], 0.0)
                self.assertTrue(result["ownership_audit_passed"])
                self.assertTrue(result["dtype_preserved"])
                self.assertEqual(
                    result["local_cells_per_process"] * processes,
                    16 * 8 * 8,
                )

    def test_real_dry_vector_field_on_two_processes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runner = root / "tools" / "run_distributed_ab2_cpu.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(runner),
                "--processes",
                "2",
                "--dtype",
                "float32",
                "--method",
                "spike",
                "--steps",
                "4",
                "--vector-field",
                "dry",
            ],
            cwd=root,
            env=dict(os.environ),
            check=False,
            capture_output=True,
            text=True,
            timeout=360,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["vector_field"], "dry")
        self.assertEqual(result["restart_error"], 0.0)
        self.assertTrue(result["ownership_audit_passed"])
        self.assertTrue(result["dtype_preserved"])


if __name__ == "__main__":
    unittest.main()
