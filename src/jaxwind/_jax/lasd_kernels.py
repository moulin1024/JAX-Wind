"""Mapped-kernel ingredients for the distributed LASD closure."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def build_lasd_kernels(
    *,
    grid,
    axis_name: str,
    partition_count: int,
    exchange_local,
    strain_magnitude_local,
    filter_two_scales_external=None,
):
    def lasd_diagnostics_local(
        scalar,
        momentum,
        momentum_coefficient,
        scalar_coefficient,
        lower_boundary_flux,
        upper_boundary_flux,
        dissipation_coefficient,
        scalar_variance_coefficient,
        wall_gradient_factor,
        horizontal_homogeneous_wall,
        stability_buoyancy_coefficient,
        stability_beta,
        stability_power,
    ):
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        cell_magnitude = strain_magnitude_local(
            momentum.dudx,
            momentum.dudy,
            momentum.dudz_at_cells,
            momentum.dvdx,
            momentum.dvdy,
            momentum.dvdz_at_cells,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        lower_is_physical = lax.axis_index(axis_name) == 0
        diagnostic_dudz = momentum.dudz_at_cells.at[0].set(
            jnp.where(
                lower_is_physical & (wall_gradient_factor > 0.0),
                momentum.u[0] * wall_gradient_factor,
                momentum.dudz_at_cells[0],
            )
        )
        diagnostic_dvdz = momentum.dvdz_at_cells.at[0].set(
            jnp.where(
                lower_is_physical & (wall_gradient_factor > 0.0),
                momentum.v[0] * wall_gradient_factor,
                momentum.dvdz_at_cells[0],
            )
        )
        diagnostic_magnitude = strain_magnitude_local(
            momentum.dudx,
            momentum.dudy,
            diagnostic_dudz,
            momentum.dvdx,
            momentum.dvdy,
            diagnostic_dvdz,
            momentum.dwdx_at_cells,
            momentum.dwdy_at_cells,
            momentum.dwdz,
        )
        face_magnitude = strain_magnitude_local(
            momentum.dudx_upper,
            momentum.dudy_upper,
            momentum.dudz_upper,
            momentum.dvdx_upper,
            momentum.dvdy_upper,
            momentum.dvdz_upper,
            momentum.dwdx_upper,
            momentum.dwdy_upper,
            momentum.dwdz_upper,
        )
        momentum_diffusivity = momentum_coefficient * delta**2 * cell_magnitude
        n2 = jnp.maximum(
            jnp.asarray(stability_buoyancy_coefficient, dtype=scalar.theta.dtype)
            * scalar.dtheta_dz_at_cells,
            0.0,
        )
        richardson = n2 / jnp.maximum(cell_magnitude**2, 1.0e-24)
        stability = (
            1.0
            + jnp.asarray(stability_beta, dtype=scalar.theta.dtype) * richardson
        ) ** (-jnp.asarray(stability_power, dtype=scalar.theta.dtype))
        effective_scalar_coefficient = scalar_coefficient * stability
        scalar_diffusivity = (
            effective_scalar_coefficient * delta**2 * cell_magnitude
        )
        coefficient_halo = exchange_local(effective_scalar_coefficient[None, ...])
        next_coefficient_plane = jnp.where(
            coefficient_halo.upper_is_physical,
            effective_scalar_coefficient[-1],
            coefficient_halo.upper[0],
        )
        next_coefficient = jnp.concatenate(
            (effective_scalar_coefficient[1:], next_coefficient_plane[None]),
            axis=0,
        )
        face_diffusivity = (
            0.5 * (effective_scalar_coefficient + next_coefficient)
            * delta**2
            * face_magnitude
        )
        flux_x = -scalar_diffusivity * scalar.dtheta_dx
        flux_y = -scalar_diffusivity * scalar.dtheta_dy
        flux_z = -face_diffusivity * scalar.dtheta_dz_upper
        flux_z = flux_z.at[-1].set(
            jnp.where(scalar.upper_is_physical, upper_boundary_flux, flux_z[-1])
        )
        flux_halo = exchange_local(flux_z[None, ...])
        lower_flux_plane = jnp.where(
            flux_halo.lower_is_physical,
            jnp.full_like(flux_z[0], lower_boundary_flux),
            flux_halo.lower[0],
        )
        lower_flux = jnp.concatenate((lower_flux_plane[None], flux_z[:-1]), axis=0)

        gradient_halo = exchange_local(scalar.dtheta_dz_upper[None, ...])
        zero_wall_cross_gradient = jnp.zeros_like(momentum.dwdx_upper[0])
        wall_face_magnitude = strain_magnitude_local(
            momentum.dudx[0],
            momentum.dudy[0],
            momentum.u[0] * wall_gradient_factor,
            momentum.dvdx[0],
            momentum.dvdy[0],
            momentum.v[0] * wall_gradient_factor,
            zero_wall_cross_gradient,
            zero_wall_cross_gradient,
            momentum.dwdz[0],
        )
        wall_scalar_diffusivity = (
            effective_scalar_coefficient[0] * delta**2 * wall_face_magnitude
        )
        wall_scalar_diffusivity = jnp.where(
            horizontal_homogeneous_wall,
            jnp.full_like(wall_scalar_diffusivity, jnp.mean(wall_scalar_diffusivity)),
            wall_scalar_diffusivity,
        )
        diagnostic_lower_face_diffusivity = jnp.where(
            lower_is_physical & (wall_gradient_factor > 0.0),
            wall_scalar_diffusivity,
            face_diffusivity[0],
        )
        boundary_gradient = jnp.where(
            diagnostic_lower_face_diffusivity > 0.0,
            -lower_flux_plane / diagnostic_lower_face_diffusivity,
            0.0,
        )
        lower_gradient_plane = jnp.where(
            gradient_halo.lower_is_physical,
            boundary_gradient,
            gradient_halo.lower[0],
        )
        lower_gradient = jnp.concatenate(
            (lower_gradient_plane[None], scalar.dtheta_dz_upper[:-1]),
            axis=0,
        )
        upper_gradient = scalar.dtheta_dz_upper.at[-1].set(
            jnp.where(
                scalar.upper_is_physical & (face_diffusivity[-1] > 0.0),
                -flux_z[-1]
                / jnp.where(
                    face_diffusivity[-1] > 0.0,
                    face_diffusivity[-1],
                    1.0,
                ),
                scalar.dtheta_dz_upper[-1],
            )
        )
        gradient_z = 0.5 * (lower_gradient + upper_gradient)
        flux_z_at_cells = 0.5 * (lower_flux + flux_z)
        shear_production = momentum_diffusivity * diagnostic_magnitude**2
        buoyancy_destruction = (
            scalar_diffusivity
            * stability_buoyancy_coefficient
            * scalar.dtheta_dz_at_cells
        )
        sgs_tke = jnp.maximum(
            (shear_production - buoyancy_destruction)
            * delta
            / dissipation_coefficient,
            0.0,
        ) ** (2.0 / 3.0)
        scalar_dissipation = -(
            flux_x * scalar.dtheta_dx
            + flux_y * scalar.dtheta_dy
            + flux_z_at_cells * gradient_z
        )
        scalar_length = delta * jnp.sqrt(
            jnp.maximum(effective_scalar_coefficient, 0.0)
        )
        sqrt_tke = jnp.sqrt(jnp.maximum(sgs_tke, 0.0))
        valid = sqrt_tke > jnp.finfo(sqrt_tke.dtype).tiny
        scalar_variance_numerator = (
            2.0 * scalar_length * scalar_dissipation / scalar_variance_coefficient
        )
        scalar_variance = jnp.maximum(
            jnp.where(
                valid,
                scalar_variance_numerator / jnp.where(valid, sqrt_tke, 1.0),
                0.0,
            ),
            0.0,
        )
        return (
            momentum_diffusivity,
            scalar_diffusivity,
            flux_x,
            flux_y,
            flux_z,
            sgs_tke,
            scalar_variance_numerator,
            scalar_variance,
        )

    def lasd_accumulate_local(
        u,
        v,
        w_at_cells,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        update_interval,
    ):
        interval = jnp.asarray(update_interval, dtype=u.dtype)
        return (
            trajectory_x + u / interval,
            trajectory_y + v / interval,
            trajectory_z + w_at_cells / interval,
        )

    def lasd_accumulate_velocity_local(
        u,
        v,
        w_upper,
        lower_boundary,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        update_interval,
    ):
        halo = exchange_local(w_upper[None, ...])
        boundary_plane = jnp.broadcast_to(
            jnp.asarray(lower_boundary, dtype=w_upper.dtype),
            w_upper.shape[1:],
        )
        lower_plane = jnp.where(
            halo.lower_is_physical,
            boundary_plane,
            halo.lower[0],
        )
        lower_faces = jnp.concatenate((lower_plane[None], w_upper[:-1]), axis=0)
        w_at_cells = 0.5 * (lower_faces + w_upper)
        return lasd_accumulate_local(
            u,
            v,
            w_at_cells,
            trajectory_x,
            trajectory_y,
            trajectory_z,

            update_interval,
        )
    def lasd_filter_two_scales_components_local(
        values,
        first_filter_width,
        second_filter_width,
    ):
        component_count = values.shape[0]
        if filter_two_scales_external is not None:
            filtered = filter_two_scales_external(
                values,
                first_filter_width,
                second_filter_width,
            )
        else:
            # Keep the component batch leading all the way into cuFFT. Building
            # component-last tensors and moving the axis here forces XLA to
            # materialize a full-volume transpose before every LASD update.
            spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
            x_mode = jnp.arange(grid.nx // 2 + 1)
            y_mode = jnp.abs(jnp.fft.fftfreq(grid.ny) * grid.ny)

            def mask(filter_width):
                cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width) + 0.5)
                cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width) + 0.5)
                return (y_mode[:, None] < cutoff_y) & (
                    x_mode[None, :] < cutoff_x
                )

            filtered = jnp.fft.irfftn(
                jnp.concatenate(
                    (
                        spectrum * mask(first_filter_width)[None, ...],
                        spectrum * mask(second_filter_width)[None, ...],
                    ),
                    axis=0,
                ),
                s=(grid.ny, grid.nx),
                axes=(-2, -1),
            ).astype(values.dtype)
        return (
            filtered[:component_count],
            filtered[component_count:],
        )

    def strain_tensor_local(momentum):
        return jnp.stack(
            (
                momentum.dudx,
                0.5 * (momentum.dudy + momentum.dvdx),
                0.5 * (momentum.dudz_at_cells + momentum.dwdx_at_cells),
                momentum.dvdy,
                0.5 * (momentum.dvdz_at_cells + momentum.dwdy_at_cells),
                momentum.dwdz,
            ),
            axis=0,
        )

    def symmetric_dot_local(left, right):
        return (
            left[0] * right[0]
            + 2.0 * left[1] * right[1]
            + 2.0 * left[2] * right[2]
            + left[3] * right[3]
            + 2.0 * left[4] * right[4]
            + left[5] * right[5]
        )

    def tensor_magnitude_local(tensor):
        return jnp.sqrt(jnp.maximum(2.0 * symmetric_dot_local(tensor, tensor), 0.0))

    def momentum_contractions_from_filtered_local(filtered, ratio):
        velocity_hat = filtered[0:3]
        products_hat = filtered[3:9]
        tensor_hat = filtered[9:15]
        magnitude_tensor_hat = filtered[15:21]
        resolved = jnp.stack(
            (
                products_hat[0] - velocity_hat[0] ** 2,
                products_hat[1] - velocity_hat[0] * velocity_hat[1],
                products_hat[2] - velocity_hat[0] * velocity_hat[2],
                products_hat[3] - velocity_hat[1] ** 2,
                products_hat[4] - velocity_hat[1] * velocity_hat[2],
                products_hat[5] - velocity_hat[2] ** 2,
            ),
            axis=0,
        )
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        model = (
            2.0
            * delta**2
            * (
                magnitude_tensor_hat
                - ratio**2 * tensor_magnitude_local(tensor_hat)[None] * tensor_hat
            )
        )
        return (
            symmetric_dot_local(resolved, model),
            symmetric_dot_local(model, model),
        )

    def contractions_from_filtered_local(filtered, ratio):
        momentum_lm, momentum_mm = momentum_contractions_from_filtered_local(
            filtered,
            ratio,
        )
        velocity_hat = filtered[0:3]
        tensor_hat = filtered[9:15]
        magnitude_tensor_hat = filtered[15:21]
        scalar_hat = filtered[21]
        velocity_scalar_hat = filtered[22:25]
        gradient_hat = filtered[25:28]
        strain_gradient_hat = filtered[28:31]
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        scalar_resolved = velocity_scalar_hat - velocity_hat * scalar_hat[None]
        scalar_model = delta**2 * (
            strain_gradient_hat
            - ratio**2 * tensor_magnitude_local(tensor_hat)[None] * gradient_hat
        )
        return (
            momentum_lm,
            momentum_mm,
            jnp.sum(scalar_resolved * scalar_model, axis=0),
            jnp.sum(scalar_model * scalar_model, axis=0),
        )

    def momentum_contractions_local(
        momentum,
        filter_grid_ratio,
        test_ratio,
    ):
        tensor = strain_tensor_local(momentum)
        magnitude = tensor_magnitude_local(tensor)
        velocity = jnp.stack((momentum.u, momentum.v, momentum.w_at_cells), axis=0)
        products = jnp.stack(
            (
                velocity[0] ** 2,
                velocity[0] * velocity[1],
                velocity[0] * velocity[2],
                velocity[1] ** 2,
                velocity[1] * velocity[2],
                velocity[2] ** 2,
            ),
            axis=0,
        )
        unfiltered = jnp.concatenate(
            (
                velocity,
                products,
                tensor,
                magnitude[None] * tensor,
            ),
            axis=0,
        )
        filtered_2d, filtered_4d = lasd_filter_two_scales_components_local(
            unfiltered,
            filter_grid_ratio * test_ratio,
            filter_grid_ratio * test_ratio**2,
        )
        return (
            *momentum_contractions_from_filtered_local(filtered_2d, test_ratio),
            *momentum_contractions_from_filtered_local(filtered_4d, test_ratio**2),
        )

    def momentum_scalar_contractions_local(
        momentum,
        scalar,
        filter_grid_ratio,
        test_ratio,
    ):
        tensor = strain_tensor_local(momentum)

        magnitude = tensor_magnitude_local(tensor)
        velocity = jnp.stack((momentum.u, momentum.v, momentum.w_at_cells), axis=0)
        products = jnp.stack(
            (
                velocity[0] ** 2,
                velocity[0] * velocity[1],
                velocity[0] * velocity[2],
                velocity[1] ** 2,
                velocity[1] * velocity[2],
                velocity[2] ** 2,
            ),
            axis=0,
        )
        scalar_anomaly = scalar.theta - jnp.mean(
            scalar.theta,
            axis=(-2, -1),
            keepdims=True,
        )
        velocity_scalar = velocity * scalar_anomaly[None]
        gradient = jnp.stack(
            (scalar.dtheta_dx, scalar.dtheta_dy, scalar.dtheta_dz_at_cells),
            axis=0,
        )
        # Filter all unique momentum/scalar products together and reuse their
        # forward transform for both LASD test scales.
        unfiltered = jnp.concatenate(
            (
                velocity,
                products,
                tensor,
                magnitude[None] * tensor,
                scalar_anomaly[None],
                velocity_scalar,
                gradient,
                magnitude[None] * gradient,
            ),
            axis=0,
        )
        filtered_2d, filtered_4d = lasd_filter_two_scales_components_local(
            unfiltered,
            filter_grid_ratio * test_ratio,
            filter_grid_ratio * test_ratio**2,
        )
        return (
            *contractions_from_filtered_local(filtered_2d, test_ratio),
            *contractions_from_filtered_local(filtered_4d, test_ratio**2),
        )


    def safe_divide_local(numerator, denominator):
        valid = jnp.abs(denominator) > 1.0e-30
        return jnp.where(
            valid,
            numerator / jnp.where(valid, denominator, 1.0),
            0.0,
        )
    def beta_local(coefficient_2d, coefficient_4d, test_ratio, scale_dependent):
        exponent = jnp.log(test_ratio) / (jnp.log(test_ratio**2) - jnp.log(test_ratio))
        raw = (
            jnp.maximum(safe_divide_local(coefficient_4d, coefficient_2d), 0.0)
            ** exponent
        )
        beta = jnp.maximum(raw, 1.0 / test_ratio**3)
        return jnp.where(scale_dependent, beta, jnp.ones_like(beta))

    def history_boundary_local(values):
        if values.shape[0] < 2:
            return values
        index = lax.axis_index(axis_name)
        values = values.at[0].set(jnp.where(index == 0, values[1], values[0]))
        return values.at[-1].set(
            jnp.where(index == partition_count - 1, values[-2], values[-1])
        )

    def departure_interpolate_local(
        values,
        lower_plane,
        upper_plane,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        interval_dt,
    ):
        extended = jnp.concatenate(
            (lower_plane[None], values, upper_plane[None]),
            axis=0,
        )
        local_nz = values.shape[0]
        z_index = jnp.arange(local_nz, dtype=trajectory_x.dtype)[:, None, None]
        y_index = jnp.arange(grid.ny, dtype=trajectory_x.dtype)[None, :, None]
        x_index = jnp.arange(grid.nx, dtype=trajectory_x.dtype)[None, None, :]
        xi = jnp.mod(x_index - trajectory_x * interval_dt / grid.dx, grid.nx)
        eta = jnp.mod(y_index - trajectory_y * interval_dt / grid.dy, grid.ny)
        zeta = jnp.clip(
            z_index - trajectory_z * interval_dt / grid.dz,
            -1.0,
            float(local_nz),
        )
        i0 = jnp.floor(xi).astype(jnp.int32)
        j0 = jnp.floor(eta).astype(jnp.int32)
        k0 = jnp.floor(zeta).astype(jnp.int32) + 1
        i1 = (i0 + 1) % grid.nx
        j1 = (j0 + 1) % grid.ny
        k1 = jnp.minimum(k0 + 1, local_nz + 1)
        fx = xi - jnp.floor(xi)
        fy = eta - jnp.floor(eta)
        fz = zeta - jnp.floor(zeta)
        q000 = extended[k0, j0, i0]
        q100 = extended[k0, j0, i1]
        q010 = extended[k0, j1, i0]
        q110 = extended[k0, j1, i1]
        q001 = extended[k1, j0, i0]
        q101 = extended[k1, j0, i1]
        q011 = extended[k1, j1, i0]
        q111 = extended[k1, j1, i1]
        q00 = (1.0 - fx) * q000 + fx * q100
        q10 = (1.0 - fx) * q010 + fx * q110
        q01 = (1.0 - fx) * q001 + fx * q101
        q11 = (1.0 - fx) * q011 + fx * q111
        q0 = (1.0 - fy) * q00 + fy * q10
        q1 = (1.0 - fy) * q01 + fy * q11
        return (1.0 - fz) * q0 + fz * q1

    def lagrangian_average_local(
        current_a,
        current_b,
        old_a,
        old_b,
        lower_a,
        upper_a,
        lower_b,
        upper_b,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        interval_dt,
        timescale_coefficient,
        timescale_a=None,
        timescale_b=None,
    ):
        scale_a = old_a if timescale_a is None else timescale_a
        scale_b = old_b if timescale_b is None else timescale_b
        product = scale_a * scale_b
        valid = (scale_a > 0.0) & (scale_b >= 0.0) & (product > 0.0)
        delta = (grid.dx * grid.dy * grid.dz) ** (1.0 / 3.0)
        timescale = (
            timescale_coefficient
            * delta
            * jnp.where(
                valid,
                product ** (-0.125),
                1.0,
            )
        )
        weight = jnp.where(
            valid,
            (interval_dt / timescale) / (1.0 + interval_dt / timescale),
            0.0,
        )
        departure_a = departure_interpolate_local(
            old_a,
            lower_a,
            upper_a,
            trajectory_x,
            trajectory_y,
            trajectory_z,
            interval_dt,
        )
        departure_b = departure_interpolate_local(
            old_b,
            lower_b,
            upper_b,
            trajectory_x,
            trajectory_y,
            trajectory_z,
            interval_dt,
        )
        return (
            weight * current_a + (1.0 - weight) * departure_a,
            jnp.maximum(weight * current_b + (1.0 - weight) * departure_b, 0.0),
        )

    def lasd_update_local(
        momentum,
        scalar,
        lm_old,
        mm_old,
        qn_old,
        nn_old,
        scalar_lm_old,
        scalar_mm_old,
        scalar_qn_old,
        scalar_nn_old,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        first_update,
        interval_dt,
        filter_grid_ratio,
        test_ratio,
        timescale_coefficient,
        momentum_initial,
        momentum_minimum,
        momentum_maximum,
        momentum_scale_dependent,
        scalar_initial,
        scalar_minimum,
        scalar_maximum,
        scalar_scale_dependent,
    ):
        (
            lm,
            mm,
            scalar_lm,
            scalar_mm,
            qn,
            nn,
            scalar_qn,
            scalar_nn,
        ) = momentum_scalar_contractions_local(
            momentum,
            scalar,
            filter_grid_ratio,
            test_ratio,
        )
        histories = (
            jnp.where(first_update, momentum_initial * mm, lm_old),
            jnp.where(first_update, mm, mm_old),
            jnp.where(first_update, momentum_initial * nn, qn_old),
            jnp.where(first_update, nn, nn_old),
            jnp.where(first_update, scalar_initial * scalar_mm, scalar_lm_old),
            jnp.where(first_update, scalar_mm, scalar_mm_old),
            jnp.where(first_update, scalar_initial * scalar_nn, scalar_qn_old),
            jnp.where(first_update, scalar_nn, scalar_nn_old),
        )
        histories = tuple(history_boundary_local(value) for value in histories)
        history_halo = exchange_local(jnp.stack(histories, axis=0))
        lower = jnp.where(
            history_halo.lower_is_physical,
            jnp.stack([value[0] for value in histories]),
            history_halo.lower,
        )
        upper = jnp.where(
            history_halo.upper_is_physical,
            jnp.stack([value[-1] for value in histories]),
            history_halo.upper,
        )

        def average_pair(
            current_a, current_b, index, timescale_a=None, timescale_b=None
        ):
            return lagrangian_average_local(
                current_a,
                current_b,
                histories[index],
                histories[index + 1],
                lower[index],
                upper[index],
                lower[index + 1],
                upper[index + 1],
                trajectory_x,
                trajectory_y,
                trajectory_z,
                interval_dt,
                timescale_coefficient,
                timescale_a,
                timescale_b,
            )

        lm_avg, mm_avg = average_pair(lm, mm, 0)
        qn_avg, nn_avg = average_pair(qn, nn, 2)
        coefficient_2d = jnp.maximum(safe_divide_local(lm_avg, mm_avg), 0.0)
        coefficient_4d = jnp.maximum(safe_divide_local(qn_avg, nn_avg), 0.0)
        momentum_coefficient = jnp.clip(
            safe_divide_local(
                coefficient_2d,
                beta_local(
                    coefficient_2d,
                    coefficient_4d,
                    test_ratio,
                    momentum_scale_dependent,
                ),
            ),
            momentum_minimum,
            momentum_maximum,
        )
        scalar_lm_avg, scalar_mm_avg = average_pair(
            scalar_lm,
            scalar_mm,
            4,
            lm_avg,
            mm_avg,
        )
        scalar_qn_avg, scalar_nn_avg = average_pair(
            scalar_qn,
            scalar_nn,
            6,
            qn_avg,
            nn_avg,
        )
        scalar_lm_avg = jnp.where(scalar_lm_avg > 0.0, scalar_lm_avg, 1.0e-32)
        scalar_qn_avg = jnp.where(scalar_qn_avg > 0.0, scalar_qn_avg, 1.0e-32)
        scalar_2d = jnp.maximum(
            safe_divide_local(scalar_lm_avg, scalar_mm_avg),
            0.0,
        )
        scalar_4d = jnp.maximum(
            safe_divide_local(scalar_qn_avg, scalar_nn_avg),
            0.0,
        )
        scalar_coefficient = jnp.clip(
            safe_divide_local(
                scalar_2d,
                beta_local(
                    scalar_2d,
                    scalar_4d,
                    test_ratio,
                    scalar_scale_dependent,
                ),
            ),
            scalar_minimum,
            scalar_maximum,
        )
        return (
            momentum_coefficient,
            lm_avg,
            mm_avg,
            qn_avg,
            nn_avg,
            scalar_coefficient,
            scalar_lm_avg,
            scalar_mm_avg,
            scalar_qn_avg,
            scalar_nn_avg,
        )

    def lasd_update_momentum_local(
        momentum,
        lm_old,
        mm_old,
        qn_old,
        nn_old,
        trajectory_x,
        trajectory_y,
        trajectory_z,
        first_update,
        interval_dt,
        filter_grid_ratio,
        test_ratio,
        timescale_coefficient,
        momentum_initial,
        momentum_minimum,
        momentum_maximum,
        momentum_scale_dependent,
    ):
        lm, mm, qn, nn = momentum_contractions_local(
            momentum,
            filter_grid_ratio,
            test_ratio,
        )
        histories = (
            jnp.where(first_update, momentum_initial * mm, lm_old),
            jnp.where(first_update, mm, mm_old),
            jnp.where(first_update, momentum_initial * nn, qn_old),
            jnp.where(first_update, nn, nn_old),
        )
        histories = tuple(history_boundary_local(value) for value in histories)
        history_halo = exchange_local(jnp.stack(histories, axis=0))
        lower = jnp.where(
            history_halo.lower_is_physical,
            jnp.stack([value[0] for value in histories]),
            history_halo.lower,
        )
        upper = jnp.where(
            history_halo.upper_is_physical,
            jnp.stack([value[-1] for value in histories]),
            history_halo.upper,
        )

        def average_pair(current_a, current_b, index):
            return lagrangian_average_local(
                current_a,
                current_b,
                histories[index],
                histories[index + 1],
                lower[index],
                upper[index],
                lower[index + 1],
                upper[index + 1],
                trajectory_x,
                trajectory_y,
                trajectory_z,
                interval_dt,
                timescale_coefficient,
            )

        lm_avg, mm_avg = average_pair(lm, mm, 0)
        qn_avg, nn_avg = average_pair(qn, nn, 2)
        coefficient_2d = jnp.maximum(safe_divide_local(lm_avg, mm_avg), 0.0)
        coefficient_4d = jnp.maximum(safe_divide_local(qn_avg, nn_avg), 0.0)
        momentum_coefficient = jnp.clip(
            safe_divide_local(
                coefficient_2d,
                beta_local(
                    coefficient_2d,
                    coefficient_4d,
                    test_ratio,
                    momentum_scale_dependent,
                ),
            ),
            momentum_minimum,
            momentum_maximum,
        )
        return momentum_coefficient, lm_avg, mm_avg, qn_avg, nn_avg

    return (
        lasd_diagnostics_local,
        lasd_accumulate_local,
        lasd_accumulate_velocity_local,
        lasd_update_local,
        lasd_update_momentum_local,
    )
