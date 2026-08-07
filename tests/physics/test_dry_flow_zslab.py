from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class DryFlowZSlabCommutingTests(unittest.TestCase):
    def test_individual_terms_and_sum_commute_with_reference(self) -> None:
        root = Path(__file__).resolve().parents[2]
        worker = Path(__file__).with_name("dry_flow_worker.py")
        for devices, dtype in ((1, "float64"), (2, "float32"), (4, "float64")):
            with self.subTest(devices=devices, dtype=dtype):
                environment = dict(os.environ)
                paths = [str(root), str(root / "src")]
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
                tolerance = 8.0e-6 if dtype == "float32" else 2.0e-12
                for term in (
                    "advection",
                    "pressure_gradient",
                    "wall",
                    "wall_filtered",
                    "sgs",
                    "sgs_vertical_flux",
                    "mgm",
                    "mgm_vertical_flux",
                    "coriolis_geostrophic",
                    "combined",
                ):
                    self.assertLess(result[term], tolerance, (term, result))
                self.assertTrue(result["dtype_preserved"])


if __name__ == "__main__":
    unittest.main()
