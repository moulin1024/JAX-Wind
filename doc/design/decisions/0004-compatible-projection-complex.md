# ADR-0004: Compatible projection operator complex

Status: **Accepted**

## Why this is one decision

For an incompressible update, let \(u^*\) be a candidate velocity, \(D\) the
discrete divergence, and \(G\) the discrete pressure gradient. Projection uses

\[
L\phi = \frac{1}{\Delta t}D u^*,\qquad L=DG,
\]

followed by

\[
u^{n+1}=u^*-\Delta t\,G\phi.
\]

Then \(D u^{n+1}=0\) only if the Poisson operator is the composition of the
same \(D\) and \(G\) used by correction and diagnostics. Choosing a Laplacian,
pressure boundary condition, filter, or divergence diagnostic independently
can leave residual divergence, create a checkerboard null mode, or inject
kinetic energy. That is why C is not merely “which Poisson solver should we
use?”

## Compatibility law

Let \(M_u\) and \(M_p\) define the discrete velocity and pressure inner
products. Apart from the stated physical boundary term, divergence and gradient
should satisfy discrete integration by parts:

\[
\langle q,Du\rangle_p=-\langle Gq,u\rangle_u,
\]

equivalently

\[
M_pD=-G^TM_u.
\]

This makes the projection orthogonal in the velocity inner product and makes
\(DG\) negative semidefinite with a known constant-pressure null space.

## First interpretation

Use the smallest hybrid location set compatible with periodic horizontal
spectral differentiation and bounded vertical fluxes:

- pressure, scalars, and horizontal velocity components live at `Cell`;
- vertical normal velocity/volume flux lives at `ZFace`;
- horizontal divergence and gradient use the same Fourier wave-number symbols;
- vertical divergence is a `ZFace -> Cell` flux difference;
- vertical pressure gradient is its weighted negative adjoint,
  `Cell -> ZFace`;
- the pressure operator is constructed as `divergence(gradient(phi))`, not as
  an independently tuned stencil;
- filtering/dealiasing belongs to nonlinear evaluation and is absent from the
  projection complex unless a later decision changes all members together;
- the pressure gauge is zero cell-volume-weighted mean;
- normal pressure-correction boundary data is derived from the prescribed
  normal-velocity condition. It is not an independently chosen pressure
  boundary condition.

This proposal does not encode the legacy convention that `w[k]` stores an
upper face. `ZFace` is semantic; an interpreter chooses its array indexing and
distributed ownership.

## Alternatives

### Fully staggered MAC grid

Pressure/scalars occupy cells and all velocity components occupy their normal
faces. This gives a very clear finite-volume complex and strongly suppresses
odd-even pressure modes. It requires explicit horizontal staggering, spectral
phase handling, and more interpolation in nonlinear terms.

### Fully collocated grid

All variables occupy cells. It is storage-simple, but a centred finite-
difference interpretation needs an additional compatible flux construction to
avoid checkerboard pressure modes. That extra mechanism would become part of
the semantics rather than a backend detail.

The accepted hybrid retains spectral horizontal planes while staggering the
bounded direction where physical wall fluxes require it.

## Required laws

- `gradient(constant) = 0` including physical boundaries.
- Discrete integration by parts holds to roundoff on the reference grid.
- `divergence(project(u)) = 0` to the linear-solver tolerance.
- `project(project(u)) = project(u)` to the projection tolerance.
- Adding a constant to pressure correction does not change velocity.
- Domain-integrated divergence equals net prescribed boundary flux.
- Local and distributed projections commute within stated tolerance.
- No non-constant pressure null mode exists on the supported topology.

## Consequence

The first reference and JAX interpreters MUST implement the hybrid
`Cell + ZFace` complex. FFT, tridiagonal, multigrid, or SPIKE remain later
solver interpretations of \(DG\); they are not part of this semantic choice.
Supporting a fully staggered or collocated complex in the future requires a new
decision and a complete implementation of the same laws rather than a storage
flag inside this interpretation.
