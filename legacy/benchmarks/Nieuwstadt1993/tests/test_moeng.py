from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
from integration_moeng import run_integration


def test_moeng_quick(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    run_integration(tmp_path / "moeng_quick", quick=True)


@pytest.mark.skipif(
    os.environ.get("WIRELES_JAX_FULL_INTEGRATION") != "1",
    reason="set WIRELES_JAX_FULL_INTEGRATION=1 to run the full Moeng CBL reproduction",
)
def test_moeng_full(tmp_path: Path) -> None:
    pytest.importorskip("jax")
    run_integration(tmp_path / "moeng_full", quick=False)
