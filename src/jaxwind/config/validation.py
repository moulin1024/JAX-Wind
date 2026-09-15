"""Configuration checks that never construct full simulation fields."""
from __future__ import annotations

from pathlib import Path
from .document import load_case


def check_case(case):
    case = load_case(case)
    if case.formulation == "boussinesq":
        from .abl import load_fv_abl
        configured = load_fv_abl(case)
        if "inflow" in case.document.get("physics", {}):
            from .synthetic_inflow import reference_profile
            reference_profile(case.document["physics"]["inflow"]["reference_profile"])
        from jaxwind.io.initial_conditions import load_initial_profile
        load_initial_profile(configured.physical)
        grid = configured.physical.physical_grid
        if configured.options.pressure_backend == "fft":
            import numpy as np
            if not all(np.allclose(widths, widths[0], rtol=1.e-13, atol=0.) for widths in (grid.x_widths, grid.y_widths)):
                raise ValueError("FFT atmospheric pressure requires uniform x and y spacing")
        for source in (configured.physical.initial_condition.path, configured.physical.reference_results):
            if not source.is_file():
                raise FileNotFoundError(f"missing case input: {source}")
    elif case.formulation == "low-mach-abl":
        from .low_mach import load_case as load_native
        from .stages import load_workflow
        from jaxwind.io.initial_conditions import load_initial_profile
        native = load_native(case)
        physical = load_workflow(native.source_workflow).case.physical
        if not physical.physical_grid.is_uniform:
            raise ValueError("periodic low-Mach flow requires a uniform mesh")
        if native.source_checkpoint is None:
            load_initial_profile(physical)
        else:
            from jaxwind.io.checkpoint import checkpoint_metadata
            checkpoint_metadata(native.source_checkpoint)
    else:
        from .jet import load_case as load_native
        load_native(case)
    return {"schema_version": 1, "case": case.document["case"]["name"],
            "formulation": case.formulation, "mesh": case.document.get("mesh"),
            "numerics": case.document["numerics"], "time": case.document["time"],
            "output": str(case.output), "fingerprint": case.fingerprint}
