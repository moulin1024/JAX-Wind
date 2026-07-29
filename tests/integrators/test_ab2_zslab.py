from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ZSlabAB2CommutingTests(unittest.TestCase):
    def test_reference_and_zslab_ab2_commute_and_restart(self) -> None:
        root = Path(__file__).resolve().parents[2]
        dependency = root.parent / "bw1000_benchmark"
        if importlib.util.find_spec("spectral_fd") is None and not dependency.exists():
            self.skipTest("spectral-fd is not installed or available as a sibling checkout")
        worker = Path(__file__).with_name("ab2_worker.py")
        for devices, dtype in ((1, "float64"), (2, "float32"), (4, "float64")):
            with self.subTest(devices=devices, dtype=dtype):
                environment = dict(os.environ)
                paths = [str(root), str(root / "src")]
                if dependency.exists():
                    paths.append(str(dependency))
                if environment.get("PYTHONPATH"):
                    paths.append(environment["PYTHONPATH"])
                environment["PYTHONPATH"] = os.pathsep.join(paths)
                environment["JAX_ENABLE_X64"] = "1"
                existing_flags = environment.get("XLA_FLAGS", "")
                environment["XLA_FLAGS"] = (
                    f"--xla_force_host_platform_device_count={devices} "
                    f"{existing_flags}"
                ).strip()
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(worker),
                        "--devices",
                        str(devices),
                        "--dtype",
                        dtype,
                    ],
                    cwd=root,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=240,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                tolerance = 5.0e-5 if dtype == "float32" else 5.0e-12
                self.assertLess(result["velocity_error"], tolerance, result)
                self.assertLess(result["history_error"], tolerance, result)
                self.assertLess(result["divergence"], tolerance, result)
                self.assertEqual(result["restart_error"], 0.0, result)
                self.assertTrue(result["dtype_preserved"])
                self.assertEqual(result["clock"], [0.2, 4])
                for actual, expected in zip(
                    result["evaluation_times"],
                    (0.0, 0.05, 0.1, 0.15),
                    strict=True,
                ):
                    self.assertAlmostEqual(actual, expected, places=14)


if __name__ == "__main__":
    unittest.main()
