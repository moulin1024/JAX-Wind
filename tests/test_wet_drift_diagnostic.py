"""Physical limits and conservation for the benchmark wet-plate diagnostic."""

import importlib.util
from pathlib import Path

import jax
import numpy as np

from jaxwind.physics.moisture import MoistureConfig, saturation_mixing_ratio

spec = importlib.util.spec_from_file_location(
    "wet_drift", Path(__file__).resolve().parents[1] / "tools/diagnose_wet_drift.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_evaporating_plate_conserves_water_and_total_enthalpy():
    jax.config.update("jax_enable_x64", True)
    c = MoistureConfig(pressure=101325, dry_air_density=1.125)
    ta, qv, ql, tl = 307.0, 0.01, 0.17, 296.0
    a, v, l, w = map(float, module.wet_plate(ta, qv, ql, tl, 3.0, c))
    assert a < ta and v > qv and l < ql
    np.testing.assert_allclose(v + l, qv + ql, atol=1e-14, rtol=0)
    hin = (
        c.dry_air_heat_capacity * ta
        + c.water_vapor_latent_heat * qv
        + ql * 4182 * (tl - 273.15)
    )
    hout = (
        c.dry_air_heat_capacity * a
        + c.water_vapor_latent_heat * v
        + l * 4182 * (w - 273.15)
    )
    np.testing.assert_allclose(hout, hin, atol=1e-8, rtol=0)
    import jax.numpy as jnp

    sat = float(saturation_mixing_ratio(jnp.asarray(ta), c.pressure, c))
    equilibrium = module.wet_plate(ta, sat, ql, ta, 3.0, c)
    np.testing.assert_allclose(equilibrium, [ta, sat, ql, ta], atol=1e-12, rtol=0)


def test_sensible_exchange_converges_to_two_stream_analytic_solution():
    jax.config.update("jax_enable_x64", True)
    c = MoistureConfig(dry_air_density=1.125)
    cp, cpl, ql, u = 1005.0, 4182.0, 0.17, 3.0
    dh = 2 * 0.036 * 0.64 / (0.036 + 0.64)
    re = 1.125 * u * dh / 1.9e-5
    pr = cp * 1.9e-5 / 0.0265
    f = (0.79 * np.log(re) - 1.64) ** -2
    nu = (
        f
        / 8
        * (re - 1000)
        * pr
        / (1 + 12.7 * np.sqrt(f / 8) * (pr ** (2 / 3) - 1))
        * (1 + (dh / 0.195) ** (2 / 3))
    )
    h = nu * 0.0265 / dh
    exponent = h * 4 / dh / (1.125 * u) * 0.195 * (1 / cp + 1 / (ql * cpl))
    equilibrium = (cp * 307 + ql * cpl * 296) / (cp + ql * cpl)
    exact_air = equilibrium + ql * cpl / (cp + ql * cpl) * 11 * np.exp(-exponent)
    errors = []
    for steps in (256, 1024, 4096):
        result = module.wet_plate(
            307.0, 0.01, ql, 296.0, u, c, steps=steps, mass_transfer=False
        )
        errors.append(abs(float(result[0]) - exact_air))
        np.testing.assert_allclose(float(result[1]), 0.01, atol=1e-14)
    assert errors[2] < errors[1] / 3 < errors[0] / 9
    assert errors[2] < 1e-4
