# Analytic rectilinear meshing application

`jaxwind-mesh` generates a versioned physical mesh artifact independently of
JAX, a pressure backend, or a simulation runner. Each Cartesian axis selects
its own cell count, clustering point, clustering mode, and strength.

Generate and inspect the supplied example from any working directory:

```bash
jaxwind-mesh generate /path/to/JAX-Wind/meshing/example.toml \
  --output "$PWD/mesh.json"
jaxwind-mesh inspect "$PWD/mesh.json"
```

The modes have explicit meanings:

- `uniform`: no point and `strength = 0`;
- `single`: the point is the lower or upper domain boundary, and the mesh is
  exponentially clustered toward it;
- `double`: the point is strictly inside the domain, and independent left and
  right exponential maps cluster toward it.

For every mode, `strength = 0` gives the exact uniform grid. Positive strength
increases clustering. Values that collapse distinct faces at floating-point
precision are rejected. A positive-strength double map places the clustering
point exactly on a face.

Which mode suits an axis depends on how the solver closes it. A bounded axis,
such as the wall-normal one in the ABL solver, is the natural home for `single`.
A periodic axis is better served by `double` with an interior point, because that
leaves the first and last cell widths equal and so keeps the spacing continuous
across the seam where the axis wraps.

The JSON output stores physical face coordinates, the complete generating
configuration, spacing-quality summaries, an explicit `z-y-x` storage order,
and the schema identifier `jaxwind.rectilinear-mesh.v1`. Solvers consume the
face coordinates; they do not depend on any mapping formula.
