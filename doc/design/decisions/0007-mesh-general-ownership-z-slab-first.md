# ADR-0007: Mesh-general ownership with a z-slab first interpreter

Status: **Accepted**

## Context

The semantic domain has global coordinates, but global shape metadata must not
be confused with a globally materialized array. The first production solver
needs z decomposition immediately, while later horizontal pencils, concurrent
domains, and accelerator meshes must not force a rewrite of every field type.

A z-slab-only domain model would be small, but would make a backend mechanism
part of field meaning. A completely generic distributed tensor DSL would move
in the opposite direction: it would add runtime symbolic machinery before a
second topology exists. Neither choice satisfies the static, document-first
design.

`Cell` and `ZFace` also have different ownership at slab interfaces. Leaving
face ownership implicit would permit duplicated persistent values or make a
halo plane appear to be prognostic state.

## Decision

### Semantic topology and realized ownership

The domain layer defines small immutable Python values for:

- named logical domain axes;
- named process-mesh axes and their extents;
- a distribution specification mapping each logical axis to either
  `Replicated` or one process-mesh axis;
- a realized owned region containing global coordinate metadata, addressable
  local intervals, physical-boundary flags, and the field location needed to
  validate local extents.

These values contain metadata only. They are not a symbolic tensor language,
do not import JAX, and do not contain collectives. A field retains the static
ownership parameter required by ADR-0001; changing ownership is an explicit
interpretation, not an in-place mutation of the same semantic value.

Global shape is always logical metadata. It never authorizes a process to
allocate, load, or gather the global production payload.

### First supported production topology

The first production interpretation supports one process-mesh axis partitioning
logical `Z`. Logical `X` and `Y` remain replicated inside each z slab. Other
distribution specifications fail during interpreter construction with an
error naming the unsupported mapping.

The first interpretation requires equal cell slabs. Therefore the number of
cell levels must be divisible by the global mesh size. A pressure backend may
declare additional construction-time divisibility requirements for its
transient spectral redistribution; those requirements are backend capabilities,
not semantic field laws.

No process owns a global field. A process owns only its addressable slabs,
although a backend array object may carry global shape and sharding metadata.

### Cell and vertical-face ownership

`Cell` ownership is the contiguous half-open cell interval assigned to a slab.

For the first equal-shape `ZFace` storage interpretation, a stored vertical
face is owned by the cell immediately below it. Each owned cell contributes its
upper face, including the upper physical face of the last cell. The lower
physical face is supplied by the physical-boundary context. Consequently:

- every persistent stored face has exactly one owner;
- a cell slab and its stored upper-face slab have the same z extent;
- an inter-slab face is owned by the lower slab and appears only as transient
  neighborhood context in the upper slab;
- physical boundary values and inter-slab halo values remain different
  constructors;
- the convention is an interpreter mapping of semantic `ZFace`, not a
  universal definition of `ZFace`.

An alternative face-storage mapping may be added later if it implements the
same semantic coordinates and ownership laws.

### Halos, redistribution, and pressure workspaces

Persistent state contains owned payload only. Halo construction returns a
transient context and cannot change the owner of an element.

Slab-to-pencil transforms and the internal y-pencil used by a pressure solver
are transient interpreter workspaces. They preserve quantity, coordinates,
and phase but are not admitted as persistent prognostic ownership. The
production pressure adapter consumes and returns z-owned cell fields; its
internal transpose or SPIKE interface communication does not leak into the
semantic state type.

Collectives are selected from the declared mesh axis inside the compiled SPMD
program. MPI may launch processes but does not provide a second ordered halo or
transpose path around the compiled step.

## Required laws

### Partition laws

- Owned cell intervals are disjoint and cover the logical cell interval
  exactly.
- Persistent stored `ZFace` coordinates are unique and, together with the
  explicit lower physical boundary face, cover the supported vertical faces.
- Every local stored index maps to exactly one logical coordinate and owner.
- Rank or device relabelling preserves the assembled logical field.

### Halo laws

- Extracting the owned value from a halo context returns the original value.
- Neighboring contexts agree on the declared interface coordinate.
- Repeated halo construction does not grow storage or turn a received value
  into persistent state.
- Halo payload size is determined by stencil width, field location, and the
  partitioned mesh axis.

### Interpretation laws

- On bounded tests, each distributed output shard equals the corresponding
  reference slice within the declared tolerance.
- Any supported redistribution followed by its inverse preserves the owned
  values and semantic coordinates.
- Local and z-slab interpretations of projection commute within solver
  tolerance.
- Production execution and checkpointing never require one process to
  materialize the global payload.

## Failure behavior

Invalid mesh sizes, non-divisible equal-slab grids, inconsistent local extents,
duplicate ownership, and unsupported axis mappings are construction-time
errors in the effect shell. This decision introduces no dynamic compiled
failure status and therefore does not pre-empt open decision E.

## Consequences

The semantic ownership vocabulary can describe later meshes without claiming
that their interpreters exist. The first code remains deliberately narrow: one
z mesh axis, equal slabs, `Cell` and the accepted `ZFace` mapping.

Pressure implementations may use transpose, exact SPIKE, or adaptive SPIKE as
backend interpretations of the same owned `Cell -> Cell` operator. Their
communication and temporary layouts are performance choices, not new field
locations.

ADR-0008 selects JAX as the only array backend and requires an independent JAX
tiny-grid reference. The ownership metadata and its law harness remain
array-independent.
