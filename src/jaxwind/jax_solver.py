"""One JAX solver API for one or many processes and devices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple

from jaxwind.domain import (
    AcceptedClock,
    AddressableField,
    Candidate,
    Cell,
    DistributionSpec,
    EqualVerticalPartition,
    EvaluationTime,
    MeshAxis,
    MeshTopology,
    VerticalBoundary,
    VerticalFaceField,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.effects import DistributedCheckpointLayout, JaxRuntime
from jaxwind.integrators import AB2Config, Evaluation, cold_start_boussinesq
from jaxwind._jax.discretization import build_discretization
from jaxwind._jax.pytrees import register_solver_pytrees
from jaxwind.operators import VelocityVector, project
from jaxwind.physics import (
    BoussinesqFields,
    BoussinesqModel,
    BoussinesqVectorField,
    DiagnosticLasdConstants,
    IdentityClosureEvent,
    LagrangianScaleDependentDynamic,
    LagrangianScaleDependentScalarFlux,
    LasdAcceptedStepEvent,
    WindTunnelBoussinesqVectorField,
    WindTunnelModel,
)
from jaxwind.pressure import build_spectral_fd_pressure_adapter
from jaxwind.solver import Advance, build_solver


class JaxDiagnosticFields(NamedTuple):
    """Process-local numerical diagnostics without exposing backend contexts."""

    surface_transfer: Any
    sgs_tke: Any
    momentum_diffusivity: Any
    scalar_diffusivity: Any
    scalar_variance: Any
    scalar_flux_z: Any
    scalar_upper: Any
    momentum_sgs_tke_transfer: Any
    sgs_flux_xz: Any
    sgs_flux_yz: Any


@dataclass(frozen=True, slots=True)
class JaxSolver:
    """A complete Boussinesq solver over an initialized JAX job.

    The domain partition is private construction state.  Applications provide
    global semantic initial values and receive one solver regardless of whether
    JAX has one process/device or many.  Physics sees only its algebra protocol.
    """

    grid: Any
    runtime: JaxRuntime
    model: BoussinesqModel
    integrator: AB2Config
    advance: Advance
    _algebra: Any
    _pressure_solver: Any
    _decomposition: EqualVerticalPartition
    _addressable_partitions: tuple[int, ...]

    @property
    def local_cell_shape(self) -> tuple[int, int, int, int]:
        return (
            self.runtime.local_devices,
            self._decomposition.cells_per_partition,
            self.grid.ny,
            self.grid.nx,
        )

    def _local_values(self, global_values: Any) -> Any:
        expected = (self.grid.nz, self.grid.ny, self.grid.nx)
        actual = tuple(int(value) for value in global_values.shape)
        if actual != expected:
            raise ValueError(
                f"global solver field has shape {actual}; expected {expected}"
            )
        partitioned = global_values.reshape(
            (
                self.runtime.global_devices,
                self._decomposition.cells_per_partition,
                self.grid.ny,
                self.grid.nx,
            )
        )
        first = self._addressable_partitions[0]
        return partitioned[first : first + self.runtime.local_devices]

    def cell_field(
        self,
        quantity: type,
        phase: type,
        global_values: Any,
    ) -> AddressableField:
        """Lower one global semantic cell field into process-local ownership."""

        regions = self._decomposition.regions(Cell)
        return AddressableField(
            quantity,
            Cell,
            tuple(regions[index] for index in self._addressable_partitions),
            phase,
            self._local_values(global_values),
        )

    def candidate_velocity(
        self,
        u: Any,
        v: Any,
        w_upper: Any,
        *,
        lower_boundary: Any,
    ) -> VelocityVector:
        """Lower global staggered velocity values without exposing partitions."""

        cell_regions = self._decomposition.regions(Cell)
        face_regions = self._decomposition.regions(ZFace)
        local_regions = tuple(
            cell_regions[index] for index in self._addressable_partitions
        )
        local_face_regions = tuple(
            face_regions[index] for index in self._addressable_partitions
        )
        return VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                local_regions,
                Candidate,
                self._local_values(u),
            ),
            AddressableField(
                YVelocity,
                Cell,
                local_regions,
                Candidate,
                self._local_values(v),
            ),
            VerticalFaceField(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    local_face_regions,
                    Candidate,
                    self._local_values(w_upper),
                ),
                lower_boundary,
            ),
        )

    def project_initial_velocity(
        self,
        velocity: VelocityVector,
        *,
        normal_boundary: VerticalBoundary = VerticalBoundary(0.0, 0.0),
    ) -> VelocityVector:
        return project(
            velocity,
            dt=self.integrator.dt,
            normal_boundary=normal_boundary,
            algebra=self._algebra,
            pressure_solver=self._pressure_solver,
        ).velocity

    def initialize_fields(self, fields: BoussinesqFields) -> BoussinesqFields:
        """Initialize closure memory selected by the semantic model."""

        if isinstance(self.model.momentum.sgs, LagrangianScaleDependentDynamic):
            return self._algebra.initialize_lasd_closure(fields, self.model)
        return fields

    def cold_start(
        self,
        fields: BoussinesqFields,
        *,
        clock: AcceptedClock = AcceptedClock(0.0, 0),
    ) -> Any:
        return cold_start_boussinesq(fields, clock=clock, config=self.integrator)

    def checkpoint_layout(self, array_factory: Callable) -> DistributedCheckpointLayout:
        """Describe this process's persistent ownership to the effect shell."""

        return DistributedCheckpointLayout(
            self._decomposition,
            self._addressable_partitions,
            array_factory,
        )

    def global_array(self, local_values: Any) -> Any:
        """Gather a field for host diagnostics, never for solver computation."""

        values = self.runtime.global_array(local_values)
        return values.reshape((self.grid.nz, self.grid.ny, self.grid.nx))

    def diagnostic_fields(
        self,
        fields: BoussinesqFields,
        clock: Any,
    ) -> JaxDiagnosticFields:
        """Evaluate backend diagnostics behind one solver-owned boundary."""

        context = self._algebra.boussinesq_context(fields)
        diagnostic = self._algebra.lasd_diagnostic_fields(
            context,
            self.model.momentum.sgs,
            self.model.scalar_sgs,
            self.model.scalar_boundary,
            constants=DiagnosticLasdConstants(horizontal_homogeneous_wall=True),
            wall=self.model.momentum.wall,
        )
        transfer = self._algebra.surface_transfer(fields, self.model, clock)
        tke_transfer = self._algebra.momentum_sgs_tke_transfer(
            context.momentum,
            self.model.momentum.sgs,
            wall=self.model.momentum.wall,
        )
        flux_xz, flux_yz = self._algebra.sgs_vertical_flux(
            context.momentum,
            self.model.momentum.sgs,
        )
        return JaxDiagnosticFields(
            transfer,
            diagnostic.sgs_tke,
            diagnostic.momentum_diffusivity,
            diagnostic.scalar_diffusivity,
            diagnostic.scalar_variance,
            diagnostic.scalar_flux_z,
            context.arrays.theta_upper,
            tke_transfer,
            flux_xz,
            flux_yz,
        )


def build_jax_solver(
    grid: Any,
    *,
    runtime: JaxRuntime,
    model: BoussinesqModel,
    integrator: AB2Config,
    normal_boundary: Callable[[Any, Any], VerticalBoundary],
    pressure_dtype: str,
    pressure_method: str = "transpose",
    pressure_tridiag: str = "thomas",
    pressure_thomas_chunk: int = 1,
    nonlinear_padding_ratio: float = 1.5,
    nonlinear_dealiasing: str = "three_halves",
    wind_tunnel_model: WindTunnelModel | None = None,
    environment: Any = None,
    optimize_frozen_zero_scalar: bool = False,
    reuse_rhs_momentum_context: bool = False,
    lasd_filter_backend: str = "jax",
) -> JaxSolver:
    """Build the same solver for the full initialized JAX process mesh.

    Device and process counts are runtime effects.  A one-process, one-device
    job therefore follows this exact path with a mesh size of one; there is no
    local solver selection or fallback.
    """

    if grid.nz % runtime.global_devices:
        raise ValueError(
            "the vertical cell count must be divisible by the global JAX "
            "device count"
        )
    if not isinstance(reuse_rhs_momentum_context, bool):
        raise TypeError("LASD momentum-context reuse flag must be boolean")
    if (
        isinstance(model.scalar_sgs, LagrangianScaleDependentScalarFlux)
        and not model.scalar_sgs.dynamic_updates_enabled
        and not optimize_frozen_zero_scalar
    ):
        raise ValueError(
            "disabled scalar LASD updates require a frozen zero scalar"
        )
    register_solver_pytrees()
    decomposition = EqualVerticalPartition(
        grid,
        MeshTopology((MeshAxis("z", runtime.global_devices),)),
        DistributionSpec.vertical(),
    )
    addressable = runtime.addressable_partitions
    wall_correction = getattr(
        model.momentum.wall,
        "porte_agel_correction",
        True,
    )
    algebra = build_discretization(
        decomposition,
        addressable_partitions=addressable,
        porte_agel_wall_correction=wall_correction,
        nonlinear_padding_ratio=nonlinear_padding_ratio,
        nonlinear_dealiasing=nonlinear_dealiasing,
        frozen_zero_scalar=optimize_frozen_zero_scalar,
        lasd_filter_backend=lasd_filter_backend,
    )
    pressure_solver = build_spectral_fd_pressure_adapter(
        decomposition,
        addressable_partitions=addressable,
        runtime=runtime,
        dtype=pressure_dtype,
        method=pressure_method,
        tridiag=pressure_tridiag,
        thomas_chunk=pressure_thomas_chunk,
    )
    vector_field = BoussinesqVectorField(algebra, model)
    if wind_tunnel_model is not None:
        if not isinstance(wind_tunnel_model, WindTunnelModel):
            raise TypeError("wind-tunnel model has an unsupported type")
        vector_field = WindTunnelBoussinesqVectorField(
            algebra,
            vector_field,
            wind_tunnel_model,
        )
    fused_update_evaluation = None
    if (
        reuse_rhs_momentum_context
        and wind_tunnel_model is None
        and isinstance(model.momentum.sgs, LagrangianScaleDependentDynamic)
    ):
        import jax

        interval = model.momentum.sgs.update_interval

        def compiled_update(first_update: bool):
            update_step = interval - 1 if first_update else 2 * interval - 1
            update_clock = AcceptedClock(0.0, update_step)

            def update_and_evaluate(fields, execution_time):
                prepared, _, momentum_context = (
                    algebra.prepare_lasd_closure_with_context(
                        fields,
                        model,
                        update_clock,
                        integrator.dt,
                    )
                )
                evaluation = Evaluation(
                    prepared,
                    EvaluationTime(
                        execution_time,
                        update_step,
                        "fused-lasd-update",
                    ),
                    environment,
                )
                evaluated = vector_field.evaluate_prepared(
                    evaluation,
                    momentum_context,
                )
                return prepared, evaluated.tendency

            return jax.jit(update_and_evaluate)

        first_update_evaluation = compiled_update(True)
        regular_update_evaluation = compiled_update(False)

        def fused_update_evaluation(fields, execution_time, first_update):
            operation = (
                first_update_evaluation
                if first_update
                else regular_update_evaluation
            )
            return operation(fields, execution_time)

    closure_event = (
        LasdAcceptedStepEvent(
            algebra,
            model,
            integrator.dt,
            reuse_rhs_momentum_context,
            fused_update_evaluation,
        )
        if isinstance(model.momentum.sgs, LagrangianScaleDependentDynamic)
        else IdentityClosureEvent()
    )
    advance = build_solver(
        config=integrator,
        vector_field=vector_field,
        normal_boundary=normal_boundary,
        algebra=algebra,
        pressure_solver=pressure_solver,
        closure_event=closure_event,
        environment=environment,
    )
    return JaxSolver(
        grid,
        runtime,
        model,
        integrator,
        advance,
        algebra,
        pressure_solver,
        decomposition,
        addressable,
    )


__all__ = ["JaxDiagnosticFields", "JaxSolver", "build_jax_solver"]
