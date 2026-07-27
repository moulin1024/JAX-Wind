from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class ZSlabProjectionCommutingTests(unittest.TestCase):
    def test_reference_and_spectral_fd_projection_commute(self) -> None:
        root = Path(__file__).resolve().parents[2]
        dependency = root.parent / "bw1000_benchmark"
        if importlib.util.find_spec("spectral_fd") is None and not dependency.exists():
            self.skipTest("spectral-fd is not installed or available as a sibling checkout")
        worker = Path(__file__).with_name("projection_worker.py")
        cases = (
            (1, "float64", "transpose"),
            (2, "float32", "transpose"),
            (4, "float64", "transpose"),
            (2, "float64", "spike"),
            (2, "float64", "spike-adaptive"),
        )
        for devices, dtype, method in cases:
            with self.subTest(devices=devices, dtype=dtype, method=method):
                environment = dict(os.environ)
                paths = [str(root / "src")]
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
                        "--method",
                        method,
                    ],
                    cwd=root,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                tolerance = 2.0e-5 if dtype == "float32" else 3.0e-12
                for name in (
                    "pressure_error",
                    "x_error",
                    "y_error",
                    "z_error",
                    "divergence",
                    "pressure_mean",
                    "idempotence",
                ):
                    self.assertLess(result[name], tolerance, (name, result))


if __name__ == "__main__":
    unittest.main()
