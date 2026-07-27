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
class MultiProcessCPUProjectionTests(unittest.TestCase):
    def test_process_and_dtype_projection_gates(self) -> None:
        root = Path(__file__).resolve().parents[2]
        runner = root / "tools" / "run_distributed_projection_cpu.py"
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
                        "--methods",
                        "transpose,spike,spike-adaptive",
                        "--dtype",
                        dtype,
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
                self.assertEqual(result["processes"], processes)
                self.assertEqual(result["dtype"], dtype)
                self.assertTrue(result["ownership_audit_passed"])
                self.assertEqual(
                    result["local_cells_per_process"] * processes,
                    16 * 8 * 8,
                )
                if processes > 1:
                    self.assertFalse(result["rank_holds_entire_domain"])


if __name__ == "__main__":
    unittest.main()
