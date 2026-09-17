"""Water-spray equations from the public Fluent 2026 R1 DPM Theory Guide.

Independent implementation, not ANSYS source or certified Fluent equivalence.
Warm, spherical, pure-water drops; no boiling, radiation, breakup or collisions.
The gas inputs are local CFD conditions, not a separate subcell plume state.
See cases/FluentDPMWater/baseline.json for pinned equations and selected options.
"""

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .moisture import saturation_vapor_pressure_water


@dataclass(frozen=True)
class DPMWaterMaterial:
    """Explicit constant-property test material, NOT Fluent database defaults.

    Enthalpy references: hl=cp_l*(T-Tref), hv=Lv_ref+cp_v*(T-Tref).
    Consequently latent heat is hv-hl at the actual drop temperature.
    The same saturation function/property values must be supplied in Fluent
    before a solver-to-solver comparison is meaningful.
    """

    liquid_density: float = 997.0
    liquid_cp: float = 4182.0
    vapor_cp: float = 1859.0
    dry_air_cp: float = 1005.0
    viscosity: float = 1.9e-5
    conductivity: float = 0.0265
    binary_diffusivity: float = 2.5e-5
    dry_air_gas_constant: float = 287.05
    vapor_gas_constant: float = 461.5
    latent_heat_reference: float = 2.50e6
    reference_temperature: float = 273.15
    boiling_temperature: float = 373.15

    def __post_init__(self):
        if any(not math.isfinite(getattr(self, f.name)) or getattr(self, f.name) <= 0
               for f in fields(self)):
            raise ValueError("DPM material properties must be finite and positive")
        if self.boiling_temperature <= self.reference_temperature:
            raise ValueError("boiling temperature must exceed the warm lower bound")


DEFAULT_MATERIAL = DPMWaterMaterial()


class DPMRates(NamedTuple):
    evaporation: jax.Array  # positive kg/s per physical droplet
    convective_heat: jax.Array  # W into the droplet
    liquid_enthalpy_rate: jax.Array  # Q - evaporation*hv(Tp)
    temperature_rate: jax.Array
    reynolds: jax.Array
    nusselt: jax.Array
    sherwood: jax.Array
    valid: jax.Array


def gas_properties(temperature, vapor_mass_fraction, pressure, material):
    """Ideal binary gas density and mixture cp; inputs use MASS fraction Y."""
    y = jnp.asarray(vapor_mass_fraction)
    gas_constant = (1-y)*material.dry_air_gas_constant + y*material.vapor_gas_constant
    density = pressure/(gas_constant*temperature)
    cp = (1-y)*material.dry_air_cp + y*material.vapor_cp
    return density, cp


def liquid_enthalpy(temperature, material):
    return material.liquid_cp*(temperature-material.reference_temperature)


def vapor_enthalpy(temperature, material):
    return material.latent_heat_reference + material.vapor_cp*(temperature-material.reference_temperature)


def droplet_rates(mass, temperature, gas_temperature, vapor_mass_fraction,
                  pressure, slip_speed, material=DEFAULT_MATERIAL, *,
                  vaporization="diffusion-controlled", evaporation_enabled=True):
    """Laws 1/2, equations 12-78 and 12-82..99, without radiation.

    Baseline evaporation onset is Tref: every warm water drop starts in Law 2.
    Setting evaporation_enabled=False supplies inert Law 1 for verification.
    Diffusion-controlled uses molar concentrations at their OWN temperatures;
    it is not a logarithmic mixing-ratio difference. Convection/diffusion uses
    ln(1+Bm) and the documented unity-Lewis option Bt=Bm for the Nu correction.
    No condensation onto droplets. Variable-Lewis and heat-term averaging are
    not selected. Property/domain violations are returned via valid, not clipped
    into a purportedly valid result. The positive mass floor only protects
    evaluations during rejected integration stages.
    """
    if vaporization not in ("diffusion-controlled", "convection-diffusion-controlled"):
        raise ValueError("unsupported DPM vaporization model")
    mass = jnp.asarray(mass)
    rho, cp = gas_properties(gas_temperature, vapor_mass_fraction, pressure, material)
    d = jnp.cbrt(6*jnp.maximum(mass, 1e-300)/(jnp.pi*material.liquid_density))
    re = rho*jnp.maximum(slip_speed, 0)*d/material.viscosity
    pr = cp*material.viscosity/material.conductivity
    sc = material.viscosity/(rho*material.binary_diffusivity)
    nu0, sh = 2+0.6*jnp.sqrt(re)*pr**(1/3), 2+0.6*jnp.sqrt(re)*sc**(1/3)
    ps = saturation_vapor_pressure_water(temperature)
    epsilon = material.dry_air_gas_constant/material.vapor_gas_constant
    # Vapor mole fraction from binary gas mass fraction.
    xgas = vapor_mass_fraction/(epsilon+(1-epsilon)*vapor_mass_fraction)
    xs = ps/pressure
    ys = epsilon*xs/(1-(1-epsilon)*xs)
    bm = jnp.maximum((ys-vapor_mass_fraction)/jnp.maximum(1-ys, 1e-30), 0)
    if vaporization == "diffusion-controlled":
        # Molecular weight/R_universal = 1/R_vapor; units kg/m^3.
        driving = jnp.maximum(ps/(material.vapor_gas_constant*temperature)
                              - xgas*pressure/(material.vapor_gas_constant*gas_temperature), 0)
        mass_rate = jnp.pi*d*material.binary_diffusivity*sh*driving
        nu = nu0
    else:
        mass_rate = jnp.pi*d*rho*material.binary_diffusivity*sh*jnp.log1p(bm)
        correction = jnp.where(bm > 1e-7, jnp.log1p(bm)/jnp.where(bm > 0, bm, 1),
                               1-bm/2+bm*bm/3)
        nu = nu0*correction
    mass_rate = jnp.where(evaporation_enabled, mass_rate, 0)
    # Inert heating uses uncorrected Nu even when the selected Law 2 has blowing.
    nu = jnp.where(evaporation_enabled, nu, nu0)
    heat = jnp.pi*d*material.conductivity*nu*(gas_temperature-temperature)
    latent = vapor_enthalpy(temperature, material)-liquid_enthalpy(temperature, material)
    td = (heat-mass_rate*latent)/(jnp.maximum(mass, 1e-300)*material.liquid_cp)
    hd = heat-mass_rate*vapor_enthalpy(temperature, material)
    inputs = jnp.stack(jnp.broadcast_arrays(mass, temperature, gas_temperature,
                                           vapor_mass_fraction, pressure, slip_speed))
    valid = jnp.all(jnp.isfinite(inputs), axis=0) & (mass > 0) & (pressure > 0)
    valid &= (temperature >= material.reference_temperature) & (temperature < material.boiling_temperature)
    valid &= (gas_temperature >= material.reference_temperature) & (gas_temperature < material.boiling_temperature)
    valid &= (vapor_mass_fraction >= 0) & (vapor_mass_fraction < 1) & (slip_speed >= 0)
    valid &= (ps < pressure) & (re < 50000) & (latent > 0)
    valid &= jnp.isfinite(td) & jnp.isfinite(hd)
    return DPMRates(mass_rate, heat, hd, td, re, nu, sh, valid)


class DPMThermalUpdate(NamedTuple):
    mass: jax.Array
    temperature: jax.Array
    vapor_mass: jax.Array
    gas_enthalpy_gain: jax.Array
    accepted: jax.Array
    attempts: jax.Array
    terminal_mass: jax.Array  # explicitly transferred residual at disappearance


def advance_droplet(mass, temperature, gas_temperature, vapor_mass_fraction,
                    pressure, slip_speed, dt, material=DEFAULT_MATERIAL, *,
                    vaporization="diffusion-controlled", evaporation_enabled=True,
                    rtol=1e-7, max_steps=4096):
    """Adaptive Cash-Karp 5(4) thermal integration in a frozen local gas state.

    State is normalized liquid mass and enthalpy. Per-step gas enthalpy is the
    opposite liquid enthalpy change; the gas must also gain the evaporated mass.
    This is TOTAL species enthalpy, not a temperature sink to apply verbatim.
    The gas EOS/energy solver accounts for vapor sensible/formation enthalpy.
    Rejects out-of-domain stages and rolls back the whole call on failure.
    A disappearance event transfers any residual below 1e-8 of input mass to
    vapor, including its enthalpy, and reports that terminal mass separately.
    This explicit numerical cutoff must be included in convergence audits.
    Cash-Karp coefficients are standard; these error controls are our explicit
    baseline choices, not a claim to recover Fluent's private solver controls.
    Scalar particle inputs; use vmap for arrays.
    """
    if not 0 < rtol < 1 or type(max_steps) is not int or max_steps < 1:
        raise ValueError("positive integration tolerance and step capacity required")
    mass = jnp.asarray(mass)
    cp = material.liquid_cp
    initial = jnp.array([1.0, temperature-material.reference_temperature], dtype=mass.dtype)
    a = ((), (1/5,), (3/40, 9/40), (3/10, -9/10, 6/5),
         (-11/54, 5/2, -70/27, 35/27),
         (1631/55296, 175/512, 575/13824, 44275/110592, 253/4096))
    b5 = (37/378, 0, 250/621, 125/594, 0, 512/1771)
    b4 = (2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4)

    def rhs(y):
        m = mass*y[0]
        t = material.reference_temperature+y[1]/jnp.maximum(y[0], 1e-30)
        rates = droplet_rates(m, t, gas_temperature, vapor_mass_fraction, pressure,
                              slip_speed, material, vaporization=vaporization,
                              evaporation_enabled=evaporation_enabled)
        return jnp.array([-rates.evaporation/mass, rates.liquid_enthalpy_rate/(mass*cp)]), rates.valid

    def body(state):
        time, h, y, attempts = state
        h = jnp.minimum(h, dt-time)
        ks, valid = [], jnp.asarray(True)
        for coefficients in a:
            trial = y + h*sum((w*k for w, k in zip(coefficients, ks)), jnp.zeros_like(y))
            k, ok = rhs(trial)
            ks.append(k)
            valid &= ok & (trial[0] > 1e-10)
        y5 = y+h*sum((w*k for w, k in zip(b5, ks)), jnp.zeros_like(y))
        y4 = y+h*sum((w*k for w, k in zip(b4, ks)), jnp.zeros_like(y))
        scale = jnp.array([1e-12, 1e-9])+rtol*jnp.maximum(abs(y), abs(y5))
        error = jnp.max(abs(y5-y4)/scale)
        _, ok = rhs(y5)
        accepted = valid & ok & (y5[0] > 1e-10) & jnp.isfinite(error) & (error <= 1)
        factor = jnp.where(valid & ok & jnp.isfinite(error),
                           jnp.clip(0.9*jnp.maximum(error, 1e-16)**(-0.2), 0.1, 5), 0.1)
        return (jnp.where(accepted, time+h, time), h*factor,
                jnp.where(accepted, y5, y), attempts+1)

    def condition(state):
        time, h, _y, attempts = state
        return (time < dt) & (_y[0] >= 1e-8) & (attempts < max_steps) & (time+h > time) & (h > 0)

    _, initial_ok = rhs(initial)
    initial_ok &= jnp.isfinite(dt) & (dt >= 0)
    t, _h, y, attempts = jax.lax.while_loop(
        condition, body, (jnp.asarray(0.0, mass.dtype), jnp.where(initial_ok, dt, 0),
                          initial, jnp.asarray(0, jnp.int32)))
    extinct = (y[0] < 1e-8) & (t <= dt)
    ok = initial_ok & ((t >= dt) | extinct)
    new_mass = jnp.where(ok, jnp.where(extinct, 0, mass*y[0]), mass)
    new_temp = jnp.where(ok, material.reference_temperature+y[1]/y[0], temperature)
    enthalpy = mass*liquid_enthalpy(temperature, material)-new_mass*liquid_enthalpy(new_temp, material)
    return DPMThermalUpdate(new_mass, new_temp, mass-new_mass, enthalpy, ok, attempts,
                            jnp.where(ok & extinct, mass*y[0], 0))


def les_dispersion_scales(characteristic_velocity, *, length=None, time=None):
    """2026 R1 equations 12-35/36: k=v^2, epsilon=k/time OR k^(3/2)/l.

    Inputs must come from the selected LES turbulence model. This function
    cannot infer that model's characteristic velocity from AMD viscosity.
    Exactly one positive scale is required; array validity is caller checked.
    """
    if (length is None) == (time is None):
        raise ValueError("specify exactly one LES length or time scale")
    k = jnp.asarray(characteristic_velocity)**2
    return k, k/time if time is not None else k**1.5/length


def sample_drw_eddy(normal_sample, uniform_sample, k, epsilon, relaxation_time,
                    slip_speed, eddy_length, *, random_lifetime=False,
                    integral_time_constant=0.15):
    """Documented isotropic DRW draw; caller retains it until interaction ends.

    Returns (piecewise-constant xyz fluctuation, interaction duration).
    Lifetime=2*TL or -TL*ln(r), TL=C_L*k/epsilon; limited by crossing time.
    No crossing exists when inertial stopping distance <= eddy length.
    Draws are supplied by the caller for deterministic tests and restart.
    k=0 returns zero fluctuation and infinite duration. Nonzero k requires
    epsilon>0; r must lie strictly within (0,1). Eddy length is explicit:
    no undocumented length prescription or LES energy inference is inserted.
    This draw is not an inhomogeneous well-mixed correction.
    """
    if not math.isfinite(integral_time_constant) or integral_time_constant <= 0:
        raise ValueError("positive finite DRW time constant required")
    k = jnp.asarray(k)
    tl = integral_time_constant*k/jnp.where(k > 0, epsilon, 1)
    lifetime = -tl*jnp.log(uniform_sample) if random_lifetime else 2*tl
    stopping = relaxation_time*slip_speed
    ratio = eddy_length/jnp.where(stopping > 0, stopping, 1)
    can_cross = (stopping > eddy_length) & (eddy_length > 0)
    crossing = jnp.where(can_cross,
        -relaxation_time*jnp.log1p(-jnp.where(can_cross, ratio, 0)), jnp.inf)
    fluctuation = jnp.asarray(normal_sample)*jnp.sqrt(2*k/3)
    return fluctuation, jnp.where(k > 0, jnp.minimum(lifetime, crossing), jnp.inf)
