from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class JaxZSlabCommutingTests(unittest.TestCase):
    def test_one_two_four_device_interpretations_commute(self) -> None:
        root = Path(__file__).resolve().parents[2]
        worker = Path(__file__).with_name("zslab_worker.py")
        for devices, dtype in ((1, "float64"), (2, "float32"), (4, "float64")):
            with self.subTest(devices=devices, dtype=dtype):
                environment = dict(os.environ)
                environment["PYTHONPATH"] = os.pathsep.join(
                    (str(root), str(root / "src"))
                )
                environment["JAX_ENABLE_X64"] = "1"
                existing_flags = environment.get("XLA_FLAGS", "")
                environment["XLA_FLAGS"] = (
                    f"--xla_force_host_platform_device_count={devices} {existing_flags}"
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
                    timeout=120,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                tolerance = 3.0e-6 if dtype == "float32" else 2.0e-13
                self.assertLess(result["gradient_error"], tolerance)
                self.assertLess(result["divergence_error"], tolerance)
                self.assertLess(result["actuator_line_error"], tolerance)
                self.assertEqual(result["lower_halo_error"], 0.0)
                self.assertEqual(result["upper_halo_error"], 0.0)
                self.assertTrue(result["halo_shape_stable"])
                self.assertEqual(
                    result["halo_elements_per_shard"],
                    result["declared_halo_elements_per_shard"],
                )
                plane = 2 * 3 * 4
                expected_communication = [
                    plane * (int(shard > 0) + int(shard < devices - 1))
                    for shard in range(devices)
                ]
                self.assertEqual(
                    result["communicated_elements"],
                    expected_communication,
                )
                self.assertEqual(
                    result["lower_flags"],
                    [True] + [False] * (devices - 1),
                )
                self.assertEqual(
                    result["upper_flags"],
                    [False] * (devices - 1) + [True],
                )
                self.assertTrue(result["extract_identity"])
                self.assertLess(result["resolved_filter_error"], tolerance)


if __name__ == "__main__":
    unittest.main()
