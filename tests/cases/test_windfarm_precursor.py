from __future__ import annotations

import json

from applications.windfarm_precursor.__main__ import main


def test_offline_precursor_dry_run_resolves_the_real_checkpoint(capsys) -> None:
    assert main(["--dry-run", "--precursor-steps", "2"]) == 0

    resolved = json.loads(capsys.readouterr().out)
    assert resolved["case"] == "pressure_driven_lasd_64x64x64"
    assert resolved["precursor_steps"] == 2
    assert resolved["main_steps"] == 2
    assert resolved["section"] == "inflow"
    assert resolved["restart"].endswith(
        "outputs/pressure_driven_lasd_64x64x64_gpu/checkpoint_final.npz"
    )
