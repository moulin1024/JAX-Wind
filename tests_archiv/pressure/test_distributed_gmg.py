from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_two_device_y_slab_halo_and_coarse_replication() -> None:
    root = Path(__file__).resolve().parents[2]
    worker = Path(__file__).with_name("y_slab_worker.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "src")))
    existing = environment.get("XLA_FLAGS", "")
    environment["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count=2 {existing}"
    ).strip()
    completed = subprocess.run(
        [sys.executable, str(worker)],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["devices"] == 2
    assert result["apply_error"] < 3.0e-4
    assert result["preconditioner_symmetry_error"] < 3.0e-5
    assert result["preconditioner_positive"] > 0.0
    assert result["converged"]
    assert result["relative_residual"] < 2.0e-6
    assert result["relative_error"] < 2.0e-5
    assert result["replication_level"] == 1
    assert result["replicated_shape"] == [16, 4, 4]
    assert result["stage_converged"]
    assert result["stage_divergence_norm"] < 3.0e-4
