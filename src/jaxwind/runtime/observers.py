"""Host-side diagnostics with explicit, checkpointable accumulation state."""
from __future__ import annotations

import csv
import math
import numpy as np


class Observer:
    def __init__(self, simulation):
        self.simulation = simulation
        self.history = []
        self.frames = []
        self.last_frame = -1
        self.frame_capture = None
        self.inflow_chunks = []
        self.accumulator = None
        self.low_workflow = None
        if simulation.case.formulation == "low-mach-abl":
            from jaxwind.config.low_mach import load_case
            from jaxwind.config.stages import load_workflow
            from jaxwind.simulation.abl import build_models
            self.low_case = load_case(simulation.case)
            self.low_workflow = load_workflow(self.low_case.source_workflow)
            _, momentum, _, _, _ = build_models(self.low_workflow.case, periodic_x=True, evolve_scalar=False)
            self.low_wall = momentum.surface
        self.surface = {"count": 0, "scalar_flux_sum": 0., "obukhov_sum": 0., "surface_scalar_sum": 0.}
        components = simulation.diagnostics
        if components is not None:
            from .abl_diagnostics import ProfileAccumulator, RadialAccumulator
            case, options = components.configured.physical, components.configured.options
            self.accumulator = RadialAccumulator(case) if options.spectrum_diagnostic == "radial" else ProfileAccumulator(simulation.grid.nz, simulation.grid.nx)
            self.level = int(np.argmin(abs(np.asarray(simulation.grid.z_centers) - case.diagnostic_reference.spectrum_heights_m[0])))

    def next_block(self, state, count, metadata):
        doc = self.simulation.case.document
        settings = doc["time"]
        diag = doc.get("diagnostics", {})
        step = int(state.step) - metadata["initial_step"]
        dt = settings["dt_seconds"]
        target = metadata["target_time"]
        if self.accumulator is not None:
            start = diag["sample_start_step"]
            period = diag["sample_every_steps"]
            if self.simulation.adaptive:
                now = float(state.time) - metadata["initial_time"]
                # Use the same float32-aware tolerance as sample(). A time
                # just below an already sampled boundary must not schedule
                # that boundary again (which can yield a zero-step block).
                tolerance = 8 * np.finfo(np.asarray(state.time).dtype).eps * max(1., metadata["target_time"])
                if now < start * dt - tolerance:
                    boundary = start * dt
                else:
                    index = math.floor((now + tolerance - start * dt) / (period * dt)) + 1
                    boundary = start * dt + index * period * dt
                target = min(target, metadata["initial_time"] + boundary, float(state.time) + settings.get("chunk_steps", 100) * dt)
            else:
                boundary = start if step < start else start + ((step - start) // period + 1) * period
                count = min(count, boundary - step)
        frame_count = settings.get("frame_count", diag.get("frame_count", 0))
        if frame_count:
            from .frames import frame_steps
            upcoming = [item for item in frame_steps(settings["steps"], frame_count) if item > step]
            if self.simulation.adaptive and len(self.frames) < frame_count:
                period = (metadata["target_time"] - metadata["initial_time"]) / frame_count
                target = min(target, metadata["initial_time"] + (len(self.frames) + 1) * period)
            elif upcoming and not self.simulation.adaptive:
                count = min(count, upcoming[0] - step)
        checkpoint_every = settings.get("checkpoint_every_steps")
        if checkpoint_every and not self.simulation.adaptive:
            count = min(count, checkpoint_every - step % checkpoint_every)
        return max(1, count), target

    def sample(self, state, metadata):
        import jax
        import jax.numpy as jnp
        from jaxwind import divergence
        row = {"step": int(state.step), "time_hours": float(state.time) / 3600.,
               "maximum_cfl": float(self.simulation.courant(state))}
        if hasattr(state, "last_dt"):
            row["dt_seconds"] = float(state.last_dt)
            row["rejected_steps"] = int(state.rejected_steps)
        if hasattr(state, "continuity_error"):
            row["continuity_residual_kg_m3_s"] = float(jnp.max(jnp.abs(state.continuity_error)))
        else:
            row["maximum_divergence_s"] = float(jnp.max(jnp.abs(divergence(state.velocity, self.simulation.grid))))
        components = self.simulation.diagnostics
        if components is not None:
            row["mean_scalar"] = float(jnp.mean(state.scalar))
            row["maximum_abs_w_m_s"] = float(jnp.max(jnp.abs(state.velocity.z)))
            if components.history_diagnostic is not None:
                row.update({key: float(value) for key, value in jax.device_get(components.history_diagnostic(state.velocity)).items()})
                from jaxwind import friction_velocity
                row["ustar_m_s"] = float(friction_velocity(state.velocity, self.simulation.grid, components.wall))
            elif components.exchange_diagnostic is not None:
                exchange = jax.device_get(components.exchange_diagnostic(state.velocity, state.scalar, state.time))
                row.update(ustar_m_s=float(exchange.friction_velocity), surface_scalar=float(exchange.surface_scalar),
                           surface_scalar_flux=float(exchange.scalar_flux), obukhov_length_m=float(exchange.obukhov_length))
            case = components.configured.physical
            relative = float(state.time) - metadata["initial_time"]
            period = case.sample_every_steps * case.dt_seconds
            start = case.sample_start_step * case.dt_seconds
            tolerance = 8 * np.finfo(np.asarray(state.time).dtype).eps * max(1., metadata["target_time"])
            due = relative >= start - tolerance and abs(relative - start - round((relative - start) / period) * period) <= tolerance
            if due:
                fields, profiles, ustar, exchange = jax.device_get(components.profile_diagnostic(state.velocity, state.pressure, state.scalar, state.time))
                from .abl_diagnostics import RadialAccumulator
                kw = {"grid": self.simulation.grid} if isinstance(self.accumulator, RadialAccumulator) else {"spectrum_level": self.level}
                self.accumulator.sample(fields, profiles, ustar=float(ustar), **kw)
                row["ustar_m_s"] = float(ustar)
                if exchange is not None:
                    self.surface["count"] += 1
                    for key, attribute in (("scalar_flux_sum", "scalar_flux"), ("obukhov_sum", "obukhov_length"), ("surface_scalar_sum", "surface_scalar")):
                        self.surface[key] += float(getattr(exchange, attribute))
        if self.simulation.turbine_diagnostics is not None:
            row.update({key: float(value) for key, value in jax.device_get(self.simulation.turbine_diagnostics(state)).items()})
        self.history.append(row)
        settings = self.simulation.case.document["time"]
        frame_count = settings.get("frame_count", self.simulation.case.document.get("diagnostics", {}).get("frame_count", 0))
        relative_step = int(state.step) - metadata["initial_step"]
        if frame_count and relative_step != self.last_frame:
            from .frames import frame_steps
            if self.simulation.adaptive:
                period = (metadata["target_time"] - metadata["initial_time"]) / frame_count
                due_time = metadata["initial_time"] + (len(self.frames) + 1) * period
                tolerance = 8 * np.finfo(np.asarray(state.time).dtype).eps * max(1., metadata["target_time"])
                due = len(self.frames) < frame_count and float(state.time) >= due_time - tolerance
            else:
                due = relative_step in frame_steps(settings["steps"], frame_count)
            if due:
                self.frames.append(self._frame(state))
                self.last_frame = relative_step

    def _frame(self, state):
        import jax.numpy as jnp
        grid = self.simulation.grid
        velocity = state.velocity
        def centers(values, axis, cells):
            if values.shape[axis] == cells + 1:
                return .5 * (jnp.take(values, jnp.arange(cells), axis=axis) + jnp.take(values, jnp.arange(1, cells + 1), axis=axis))
            return .5 * (values + jnp.roll(values, -1, axis=axis))
        u = centers(velocity.x, 2, grid.nx)
        turbine = self.simulation.case.document.get("physics", {}).get("turbine", {})
        height = turbine.get("hub_height_m", .876 if self.simulation.case.formulation == "low-mach-abl" else .5 * grid.lz)
        center_y = turbine.get("y_m", .5 * grid.ly)
        hub = int(np.argmin(abs(np.asarray(grid.z_centers) - height)))
        center_index = int(np.argmin(abs(np.asarray(grid.y_centers) - center_y)))
        if hasattr(state, "scalar"):
            from .frames import build_frame_capture, capture_frame
            if self.frame_capture is None:
                self.frame_capture = build_frame_capture(grid, y_m=center_y, z_m=height)
            return capture_frame(state, grid, y_m=center_y, z_m=height, capture=self.frame_capture)
        result = {"time_seconds": float(state.time), "step": int(state.step),
                  "u_hub_yx": np.asarray(u[hub]), "u_center_zx": np.asarray(u[:, center_index])}
        for name in ("scalar", "temperature", "density", "nitrogen", "liquid_water", "ice_water"):
            if hasattr(state, name):
                values = getattr(state, name)
                result[name + "_hub_yx"] = np.asarray(values[hub])
                result[name + "_center_zx"] = np.asarray(values[:, center_index])
        if self.low_workflow is not None:
            from jaxwind import friction_velocity
            v = centers(velocity.y, 1, grid.ny)
            w = centers(velocity.z, 0, grid.nz)
            for name, values in (("u", u), ("v", v), ("w", w)):
                result[name + "_profile_history_m_s"] = np.asarray(jnp.mean(values, axis=(1, 2)))
                result[name + "2_profile_history_m2_s2"] = np.asarray(jnp.mean(values * values, axis=(1, 2)))
            result["surface_friction_velocity_history_m_s"] = np.asarray(friction_velocity(velocity, grid, self.low_wall))
        return result

    def snapshot(self):
        return {"history": self.history, "frames": self.frames, "last_frame": self.last_frame,
                "inflow_chunks": self.inflow_chunks,
                "surface": self.surface, "accumulator": self.accumulator.__dict__ if self.accumulator else {}}

    def restore(self, saved):
        self.history, self.frames = saved["history"], saved["frames"]
        self.last_frame, self.surface = saved["last_frame"], saved["surface"]
        self.inflow_chunks = saved.get("inflow_chunks", [])
        if self.accumulator is not None:
            self.accumulator.__dict__.update(saved["accumulator"])

    def write(self, directory, state):
        if self.inflow_chunks:
            from jaxwind.io.recording import write_manifest
            write_manifest(directory / "inflow", self.simulation.grid, self.inflow_chunks)
        if self.history:
            fields = list(dict.fromkeys(key for row in self.history for key in row))
            with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.history)
        if self.accumulator is not None and self.accumulator.count:
            from .abl_diagnostics import write_profiles, write_streamwise_spectra, write_radial_spectra
            configured = self.simulation.diagnostics.configured
            write_profiles(directory / "profiles.csv", configured.physical, self.accumulator)
            kind = configured.options.spectrum_diagnostic
            if kind == "streamwise":
                write_streamwise_spectra(directory / "spectra.csv", configured.physical, self.accumulator)
            elif kind == "radial":
                write_radial_spectra(directory / "radial_spectra.csv", configured.physical, self.accumulator)
        if self.frames:
            grid = self.simulation.grid
            np.savez_compressed(directory / "flow_frames.npz", **{name: np.asarray([frame[name] for frame in self.frames]) for name in self.frames[0]},
                                x_m=np.asarray(grid.x_centers), y_m=np.asarray(grid.y_centers), z_m=np.asarray(grid.z_centers),
                                x_faces_m=np.asarray(grid.x_faces), y_faces_m=np.asarray(grid.y_faces), z_faces_m=np.asarray(grid.z_faces))
            if self.low_workflow is not None and np.count_nonzero(np.asarray(grid.z_centers) <= min(.1 * grid.lz, 2.)) >= 3:
                from .low_mach import _write_mean_profile
                names = ("u_hub_yx", "u_center_zx", "temperature_hub_yx", "temperature_center_zx", "density_hub_yx", "density_center_zx",
                         "u_profile_history_m_s", "v_profile_history_m_s", "w_profile_history_m_s", "u2_profile_history_m2_s2", "v2_profile_history_m2_s2", "w2_profile_history_m2_s2", "surface_friction_velocity_history_m_s")
                fields = tuple(np.asarray([frame[name] for frame in self.frames]) for name in names)
                window = self.low_case.statistics_window_seconds
                fraction = max(0., 1. - window / (self.low_case.steps * self.low_case.dt)) if window is not None else (.8 if self.low_case.restart_formulation == "configured" else 0.)
                _write_mean_profile(directory, self.low_workflow, grid, fields, [frame["time_seconds"] for frame in self.frames], statistics_start_fraction=fraction)

    def consume(self, directory, outputs):
        from jaxwind.io.recording import write_chunk
        self.inflow_chunks.append(write_chunk(directory / "inflow", len(self.inflow_chunks), outputs))

    def summary(self):
        """Scientific metadata retained for reference-analysis consumers."""
        if self.simulation.diagnostics is None:
            return {}
        from jaxwind.config.abl_resolved import resolved
        from .abl_diagnostics import bulk_metrics, reference_comparison
        configured = self.simulation.diagnostics.configured
        case, options = configured.physical, configured.options
        configuration = resolved(configured)
        metrics = {}
        comparison = None
        if self.accumulator.count:
            metrics["surface_friction_velocity_m_s"] = self.accumulator.ustar
            speed = math.hypot(*configuration["geostrophic_velocity_m_s"])
            if speed:
                metrics["surface_friction_velocity_ratio"] = self.accumulator.ustar / speed
            if options.spectrum_diagnostic == "radial":
                metrics.update(bulk_metrics(case, self.accumulator))
                comparison = reference_comparison(case, metrics)
        if self.surface["count"]:
            for name, total in (("surface_scalar_flux", "scalar_flux_sum"), ("surface_scalar", "surface_scalar_sum"), ("obukhov_length_m", "obukhov_sum")):
                value = self.surface[total] / self.surface["count"]
                metrics[name] = value if math.isfinite(value) else None
        reference = case.diagnostic_reference
        return {
            "case": case.name,
            "solver": {"discretization": "finite-volume", "pressure_backend": options.pressure_backend,
                       "time_integration": options.time_integration.upper(), "momentum_closure": "AMD", "scalar_closure": "AMD eddy diffusivity"},
            "physics": {name: configuration[name] for name in ("geostrophic_velocity_m_s", "coriolis_vertical_s", "coriolis_horizontal_s", "scalar_surface_flux", "buoyancy_acceleration_per_scalar")},
            "diagnostic_reference": {"length_m": reference.length_m, "velocity_m_s": reference.velocity_m_s,
                                     "scalar": reference.scalar, "inversion_search_max_height_m": reference.inversion_search_max_height_m,
                                     "spectrum_heights_m": list(reference.spectrum_heights_m)},
            "diagnostic_metrics": metrics,
            "runtime": {"profile_samples": self.accumulator.count,
                        "ustar_m_s": self.accumulator.ustar if self.accumulator.count else None},
            **({"comparison": comparison} if comparison is not None else {}),
        }
