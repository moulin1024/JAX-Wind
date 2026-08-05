from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class BoussinesqZSlabCommutingTests(unittest.TestCase):
    def test_scalar_buoyancy_and_rayleigh_terms_commute_with_reference(self) -> None:
        root = Path(__file__).resolve().parents[2]
        worker = Path(__file__).with_name("boussinesq_worker.py")
        for devices, dtype in ((1, "float64"), (2, "float32"), (4, "float64")):
            with self.subTest(devices=devices, dtype=dtype):
                environment = dict(os.environ)
                paths = [str(root), str(root / "src")]
                if environment.get("PYTHONPATH"):
                    paths.append(environment["PYTHONPATH"])
                environment["PYTHONPATH"] = os.pathsep.join(paths)
                environment["JAX_ENABLE_X64"] = "1"
                environment["XLA_FLAGS"] = (
                    f"--xla_force_host_platform_device_count={devices} "
                    f"{environment.get('XLA_FLAGS', '')}"
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
                tolerance = 8.0e-6 if dtype == "float32" else 2.0e-12
                for term in (
                    "scalar_advection",
                    "scalar_sgs",
                    "buoyancy",
                    "rayleigh",
                    "combined_scalar",
                    "lasd_memory",
                    "lasd_fringe_memory",
                    "lasd_momentum_tendency",
                    "lasd_scalar_tendency",
                    "lasd_diagnostics",
                ):
                    self.assertLess(result[term], tolerance, (term, result))
                self.assertTrue(result["dtype_preserved"])


if __name__ == "__main__":
    unittest.main()
