from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
JAX_ROOT = Path(__file__).resolve().parents[1]


def test_adjoint_dimension_chunk_pipeline_on_four_devices() -> None:
    env = dict(os.environ)
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    env["PYTHONPATH"] = str(JAX_ROOT)
    completed = subprocess.run(
        [sys.executable, str(JAX_ROOT / "tests/adjoint_sharded_worker.py")],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["mesh"] == {"adjoint": 2, "z": 2}
    assert result["global_shape"] == [2, 8, 4, 8]
    assert result["local_shapes"] == [[1, 8, 4, 4]] * 4
    assert result["steps"] == [3, 2]
    assert result["divergence_max"] < 2.0e-4
    assert result["state_difference"] < 1.0e-6
    assert result["target_difference"] < 1.0e-6
