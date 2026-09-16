# Synthetic turbine inflow with a Mann box

The Python API generates homogeneous neutral turbulence using the Mann uniform
shear spectral tensor, then advects the periodic box past the inlet using
Taylor's frozen-turbulence hypothesis. Generation uses NumPy/SciPy on the host;
the plane sampler can be compiled with `jax.jit` and used with
`build_open_atmospheric_step`, including turbine momentum forcing.
Run this example on a compute node:

```python
import jax
from jaxwind import UniformGrid, generate_mann_box, build_mann_inflow
from jaxwind.io.inflow import write_synthetic_inflow

grid = UniformGrid(nx=64, ny=32, nz=32, lx=800., ly=200., lz=200.)
mean_speed = 10.  # m/s
box = generate_mann_box(
    shape=(256, 32, 32),       # nx, ny, nz
    lengths=(2000., 200., 200.),  # metres; independent of LES x extent
    length_scale=29.4,        # Mann L, metres
    gamma=3.9,               # Mann shear/anisotropy parameter
    alpha_epsilon=1.,         # alpha * epsilon**(2/3), m**(4/3)/s**2
    sigma_u=0.1 * mean_speed, # optional: 10% realized box turbulence intensity
    seed=42,
)
inflow = build_mann_inflow(box, grid, mean_speed=mean_speed, scalar=288.)
plane = jax.jit(inflow)(0.5)  # seconds; returns a staggered InflowPlane
# With an existing open-boundary stepper:
# solution = step(solution, dt, inflow(solution.time))

# Alternatively, export an artifact accepted by the existing open-inflow
# workflow's inflow input. Use an empty destination and match its time step.
write_synthetic_inflow(
    'outputs/mann-inflow', grid, inflow, samples=1000, dt=0.1, chunk_size=64,
)
```

`mean_profile` optionally supplies a scalar or `(nz,)` streamwise mean profile;
`scalar` likewise accepts a constant or vertical profile. Mean profiles are
added after interpolation; the advection speed remains `mean_speed`. A shear
profile does not change the box's fitted Mann parameters. Box nodes start at
zero; stored velocities have shape `(nz, ny, nx, 3)` and contain fluctuations
only. All inputs use SI units.

Without `sigma_u`, amplitude follows the discrete spectral tensor. Setting
`sigma_u` multiplies all three components by one factor, retaining the component
ratios and correlations; the requested intensity describes the full box before
interpolation and boundary enforcement, not an individual inlet plane. The
sampler places u at the inlet x face and v/w at the first x cell centre, with
their respective transverse face locations. It enforces zero vertical velocity
at top/bottom; `wall_y=True` additionally provides wall-bounded y faces. Recorded
artifacts currently require periodic y, matching the existing reader contract.

The box repeats after `lx / mean_speed` seconds (200 seconds in this example).
Choose streamwise length for the required nonrepeating duration, transverse
extent to cover the rotor/domain, and resolution for the desired resolved
eddies. The box is periodic in all directions, assumes neutral homogeneous
conditions, and does not model terrain, atmospheric stability, or surface
blocking. Zero and even-grid Nyquist modes are omitted; unresolved modes are
not compensated. The raw field is spectrally divergence-free; interpolation
and wall enforcement do not guarantee discrete FV divergence-free inflow.
The receiving solver still applies its pressure projection. Host generation
stores full spectral factor arrays and needs substantially more memory than
the final velocity box.

The implementation follows equations (20)–(25) and (31) of
[Jakob Mann's *Atmospheric turbulence* notes](https://breeze.colorado.edu/ftp/RSWE/Jakob_Mann.pdf),
including the hypergeometric eddy lifetime, von Kármán spectrum, and rapid
shear distortion. `gamma=0` recovers isotropic von Kármán turbulence. Tests
cover this limit, spectral incompressibility, absolute Fourier covariance,
seed reproducibility, amplitude scaling, JIT sampling, and artifact round trips.
