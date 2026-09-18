#!/usr/bin/env python3
"""Independent incompressible LES + inertial water spray (NumPy/SciPy/PyAMG).

Case 3 of the local Montazeri spray-tunnel validation. SI units; gas and droplet
 temperatures are Celsius, q is kg vapour / kg dry air. No jaxwind imports.
See README.md for equations, boundary conditions, limitations and run commands.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pyamg
from scipy import sparse
from scipy.sparse.linalg import cg

RHO, MU, CP, K, DV = 1.125, 1.9e-5, 1005.0, 0.0265, 2.5e-5
RHOW, CPL, LV, PRESSURE = 997.0, 4182.0, 2.5e6, 101325.0
TIN, TWB, TLIN, UIN = 39.2, 18.7, 35.2, 3.0
MDOT = 12.5e-3 / 60 * RHOW
GRAVITY = np.array([0., 0., -9.81 * (1 - RHO / RHOW)])


def saturation_pressure(t):
    t = np.asarray(t) + 273.15
    return np.exp(54.842763 - 6763.22/t - 4.210*np.log(t) + .000367*t
                  + np.tanh(.0415*(t-218.8)) *
                  (53.878-1331.22/t-9.44523*np.log(t)+.014025*t))


def saturation_q(t):
    e = np.minimum(saturation_pressure(t), .99 * PRESSURE)
    return (287.05 / 461.5) * e / (PRESSURE-e)


def inlet_q():
    e = saturation_pressure(TWB) - .00066*(1+.00115*TWB)*PRESSURE*(TIN-TWB)
    return float((287.05/461.5)*e/(PRESSURE-e))


def plane(a, axis, index):
    s = [slice(None)] * 3
    s[axis] = index
    return tuple(s)


def pad(a, axis, low=None, high=None):
    """One ghost plane; defaults to even reflection (zero normal derivative)."""
    lo, hi = a[plane(a, axis, slice(0, 1))], a[plane(a, axis, slice(-1, None))]
    return np.concatenate((lo if low is None else low, a,
                           hi if high is None else high), axis=axis)


def average(a, axis):
    return .5 * (a[plane(a, axis, slice(1, None))]
                 + a[plane(a, axis, slice(None, -1))])


def to_faces(a, axis):
    return average(pad(a, axis), axis)


class Grid:
    def __init__(self, shape, length=(1.9, .585, .585), cs=.16, rtol=1e-9):
        self.n = np.asarray(shape, dtype=int)
        self.length = np.asarray(length, dtype=float)
        self.h = self.length / self.n
        self.volume = float(np.prod(self.h))
        self.cs, self.rtol = cs, rtol
        matrices = []
        for axis, (n, h) in enumerate(zip(self.n, self.h)):
            diagonal = np.full(n, 2.)
            diagonal[[0, -1]] = 1.
            if axis == 0:
                diagonal[-1] += 2.  # p=0 at the outlet, half a cell away
            matrices.append(sparse.diags((-np.ones(n-1), diagonal, -np.ones(n-1)),
                                          (-1, 0, 1), format='csr') / h**2)
        ix, iy, iz = [sparse.eye(int(n), format='csr') for n in self.n]
        ax, ay, az = matrices
        self.A = (sparse.kron(ax, sparse.kron(iy, iz))
                  + sparse.kron(ix, sparse.kron(ay, iz))
                  + sparse.kron(ix, sparse.kron(iy, az))).tocsr()
        # Symmetric V cycle: compatible with CG, unlike nonsymmetric ILU.
        self.amg = pyamg.smoothed_aggregation_solver(
            self.A, symmetry='symmetric',
            presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
            postsmoother=('gauss_seidel', {'sweep': 'symmetric'}))
        self.M = self.amg.aspreconditioner(cycle='V')
        self.last_iterations = 0
        self.max_iterations = 0
        self.last_residual = 0.

    def zeros_velocity(self):
        return [np.zeros(tuple(self.n + np.eye(3, dtype=int)[i])) for i in range(3)]

    def enforce(self, velocity, inlet=UIN):
        velocity[0][0] = inlet
        velocity[1][:, 0, :] = velocity[1][:, -1, :] = 0.
        velocity[2][:, :, 0] = velocity[2][:, :, -1] = 0.
        # Outlet u is a free degree of freedom corrected by the projection.

    def divergence(self, velocity):
        return sum(np.diff(v, axis=i) / self.h[i] for i, v in enumerate(velocity))

    def gradient(self, p):
        result = [np.diff(pad(p, i), axis=i)/self.h[i] for i in range(3)]
        result[0][-1] = -2*p[-1]/self.h[0]
        return result

    def project(self, velocity, dt, inlet=UIN):
        self.enforce(velocity, inlet)
        b = -self.divergence(velocity).ravel()/dt
        count = [0]
        def callback(_):
            count[0] += 1
        if np.linalg.norm(b) == 0:
            p = np.zeros(tuple(self.n))
            self.last_residual = 0.
        else:
            flat, info = cg(self.A, b, M=self.M, rtol=self.rtol, atol=0.,
                            maxiter=300, callback=callback)
            self.last_residual = float(np.linalg.norm(self.A @ flat-b)/np.linalg.norm(b))
            if info != 0 or self.last_residual > 5*self.rtol:
                raise RuntimeError(f'Pressure CG failed: info={info}, residual={self.last_residual:g}')
            p = flat.reshape(tuple(self.n))
        self.last_iterations = count[0]
        self.max_iterations = max(self.max_iterations, count[0])
        for v, grad in zip(velocity, self.gradient(p)):
            v -= dt*grad
        self.enforce(velocity, inlet)
        return velocity, p

    def ghosts(self, a, component, axis):
        lo, hi = a[plane(a, axis, slice(0, 1))], a[plane(a, axis, slice(-1, None))]
        if axis == component:
            low = 2*lo-a[plane(a, axis, slice(1, 2))]
            high = 2*hi-a[plane(a, axis, slice(-2, -1))]
            if axis == 0:
                high = a[plane(a, axis, slice(-2, -1))]  # du/dx=0 predictor
        else:
            low = -lo  # no slip side walls; v=w=0 at physical inlet
            high = hi if axis == 0 else -hi
        return pad(a, axis, low, high)

    def strain(self, velocity):
        normal = [np.diff(v, axis=i)/self.h[i] for i, v in enumerate(velocity)]
        shear = {}
        magnitude2 = 2*sum(s*s for s in normal)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            s = (np.diff(self.ghosts(velocity[i], i, j), axis=j)/self.h[j]
                 + np.diff(self.ghosts(velocity[j], j, i), axis=i)/self.h[i])
            shear[i, j] = s
            centered = average(average(s, i), j)
            magnitude2 += centered**2
        nut = (self.cs*self.volume**(1/3))**2 * np.sqrt(magnitude2)
        return normal, shear, nut

    def momentum_rhs(self, velocity):
        normal, shear, nut = self.strain(velocity)
        viscosity = MU/RHO + nut
        centered = [average(v, i) for i, v in enumerate(velocity)]
        rhs = []
        for i, v in enumerate(velocity):
            out = np.zeros_like(v)
            for j in range(3):
                # Interpolate advecting component onto the transported component.
                advector = v if j == i else to_faces(centered[j], i)
                vp = self.ghosts(v, i, j)
                # Transverse advector on a component grid: use physical wall parity.
                ap = self.ghosts(advector, j, j) if j == i else pad(advector, j)
                if j != i:
                    # At a tangential wall the normal advector changes sign.
                    lo = -advector[plane(advector, j, slice(0, 1))]
                    hi = (advector[plane(advector, j, slice(-1, None))] if j == 0
                          else -advector[plane(advector, j, slice(-1, None))])
                    if j == 0:
                        lo = 2*UIN-advector[plane(advector, j, slice(0, 1))]
                    ap = pad(advector, j, lo, hi)
                slp, slm = plane(vp, j, slice(2, None)), plane(vp, j, slice(None, -2))
                # Skew form 1/2 [a_j D_j u_i + D_j(a_j u_i)].
                out -= .25/self.h[j]*(advector*(vp[slp]-vp[slm])
                                                  +(ap*vp)[slp]-(ap*vp)[slm])
            stress = 2*viscosity*normal[i]
            diff = np.diff(pad(stress, i), axis=i)/self.h[i]
            if i == 0:
                diff[-1] = -2*stress[-1]/self.h[i]
            out += diff
            for j in range(3):
                if j != i:
                    a, b = sorted((i, j))
                    stress = to_faces(to_faces(viscosity, a), b)*shear[a, b]
                    out += np.diff(stress, axis=j)/self.h[j]
            rhs.append(out)
        rhs[0][0] = 0.
        rhs[1][:, 0, :] = rhs[1][:, -1, :] = 0.
        rhs[2][:, :, 0] = rhs[2][:, :, -1] = 0.
        return rhs, nut

    def scalar_rhs(self, scalar, velocity, diffusivity, inlet):
        out = np.zeros_like(scalar)
        for i in range(3):
            sf = to_faces(scalar, i)
            grad = np.diff(pad(scalar, i), axis=i)/self.h[i]
            if i == 0:
                sf[0] = inlet
                grad[0] = 2*(scalar[0]-inlet)/self.h[0]
            flux = velocity[i]*sf-to_faces(diffusivity, i)*grad
            out -= np.diff(flux, axis=i)/self.h[i]
        return out

    def carrier_step(self, velocity, temp, q, dt):
        """SSPRK3 with projection at every stage; spatial differences are order 2."""
        original = [v.copy() for v in velocity], temp.copy(), q.copy()
        for old_weight, stage_weight in ((0., 1.), (.75, .25), (1/3, 2/3)):
            acceleration, nut = self.momentum_rhs(velocity)
            t_rhs = self.scalar_rhs(temp, velocity, K/(RHO*CP)+nut/.7, TIN)
            q_rhs = self.scalar_rhs(q, velocity, DV+nut/.7, inlet_q())
            velocity = [old_weight*a+stage_weight*(b+dt*c)
                        for a, b, c in zip(original[0], velocity, acceleration)]
            temp = old_weight*original[1]+stage_weight*(temp+dt*t_rhs)
            q = old_weight*original[2]+stage_weight*(q+dt*q_rhs)
            velocity, pressure = self.project(velocity, stage_weight*dt)
        return velocity, temp, q, pressure

    def stencil(self, positions, component=None):
        offset = np.full(3, .5)
        shape = self.n.copy()
        if component is not None:
            offset[component] = 0.
            shape[component] += 1
        coordinates = positions/self.h-offset
        low = np.floor(coordinates).astype(int)
        fraction = coordinates-low
        for bits in np.ndindex(2, 2, 2):
            index = low+bits
            # Reflect/merge the cloud at the nearest interior unknown. This keeps
            # the kernel normalized; fixed normal-velocity wall nodes get no force.
            for axis in range(3):
                lower, upper = 0, shape[axis]-1
                if component == axis:
                    lower, upper = 1, shape[axis]-2
                index[:, axis] = np.clip(index[:, axis], lower, upper)
            weight = np.prod(np.where(np.array(bits), fraction, 1-fraction), axis=1)
            yield tuple(index.T), weight

    def gather(self, field, positions, component=None):
        return sum(weight*field[index] for index, weight in self.stencil(positions, component))

    def deposit(self, values, positions, component=None):
        shape = self.n.copy()
        if component is not None:
            shape[component] += 1
        out = np.zeros(tuple(shape))
        for index, weight in self.stencil(positions, component):
            np.add.at(out, index, values*weight)
        return out


class Spray:
    """Each computational parcel represents 'number' identical physical droplets."""
    def __init__(self, grid, rate=4000., seed=7):
        self.grid, self.rate = grid, rate
        self.rng = np.random.default_rng(seed)
        self.state = np.empty((0, 8))  # x,y,z,vx,vy,vz,mass,sensible energy per drop
        self.number = np.empty(0)
        self.injected_count = 0
        self.injected_mass = self.escaped_mass = self.evaporated_mass = 0.
        self.energy_residual = self.momentum_residual = 0.
        self.wall_impulse = np.zeros(3)

    def inject(self, end_time):
        count = int(np.floor(end_time*self.rate+1e-10))-self.injected_count
        if count <= 0:
            return
        # Truncated mass-Rosin--Rammler: equal water mass per computational parcel.
        low, high = 1-np.exp(-(np.array([74e-6, 518e-6])/369e-6)**3.67)
        d = 369e-6*(-np.log1p(-self.rng.uniform(low, high, count)))**(1/3.67)
        phi = self.rng.uniform(0, 2*np.pi, count)
        half_angle = np.deg2rad(18.)
        speed = .9*np.sqrt(2*3e5/RHOW)
        state = np.zeros((count, 8))
        state[:, 0] = 1e-12
        state[:, 1] = self.grid.length[1]/2+.002*np.cos(phi)
        state[:, 2] = self.grid.length[2]/2+.002*np.sin(phi)
        state[:, 3] = speed*np.cos(half_angle)
        state[:, 4] = speed*np.sin(half_angle)*np.cos(phi)
        state[:, 5] = speed*np.sin(half_angle)*np.sin(phi)
        state[:, 6] = RHOW*np.pi/6*d**3
        state[:, 7] = state[:, 6]*CPL*TLIN
        self.number = np.concatenate((self.number, MDOT/self.rate/state[:, 6]))
        self.state = np.concatenate((self.state, state))
        self.injected_count += count
        self.injected_mass += count*MDOT/self.rate

    def rates(self, state, velocity, temp, q):
        g = self.grid
        x, v, m, energy = state[:, :3], state[:, 3:6], state[:, 6], state[:, 7]
        diameter = (6*m/(np.pi*RHOW))**(1/3)
        tl = energy/(m*CPL)
        gas = np.column_stack([g.gather(a, x, i) for i, a in enumerate(velocity)])
        tg, qg = g.gather(temp, x), g.gather(q, x)
        slip = gas-v
        reynolds = RHO*diameter*np.linalg.norm(slip, axis=1)/MU
        if np.any(reynolds >= 50000):
            raise RuntimeError('Morsi--Alexander drag exceeded its supported Re range')
        band = np.searchsorted([.1, 1, 10, 100, 1000, 5000, 10000], reynolds, side='right')
        a = np.array([24., 22.73, 29.17, 46.50, 98.33, 148.62, -490.546, -1662.50])[band]
        b = np.array([0., .09, -3.89, -116.67, -2778., -47500., 578700., 5416700.])[band]
        c = np.array([0., 3.69, 1.22, .62, .36, .36, .46, .52])[band]
        cdre = np.where(reynolds < .1, 24., a+b/np.maximum(reynolds, .1)+c*reynolds)
        relaxation = .75*MU*cdre/(RHOW*diameter**2)
        drag = relaxation[:, None]*slip
        nu = 2+.6*np.sqrt(reynolds)*(MU*CP/K)**(1/3)
        sh = 2+.6*np.sqrt(reynolds)*(MU/(RHO*DV))**(1/3)
        conductance = np.pi*diameter*K*nu
        heat = conductance*(tg-tl)
        evaporation = np.maximum(0., np.pi*diameter*RHO*DV*sh
                                 *np.log((1+saturation_q(tl))/(1+qg)))
        ds = np.column_stack((v, drag+GRAVITY, -evaporation, heat-LV*evaporation))
        # Dilute constant-density carrier: retain drag reaction, do not add a
        # vapour momentum source without also adding the corresponding mass term.
        forces = [-g.deposit(self.number*m*drag[:, i], x, i)/(RHO*g.volume)
                  for i in range(3)]
        temperature_rate = -g.deposit(self.number*heat, x)/(RHO*CP*g.volume)
        humidity_rate = g.deposit(self.number*evaporation, x)/(RHO*g.volume)
        stable_dt = min(.2/np.max(relaxation),
                        .1/np.max(conductance/(m*CPL)),
                        .05/np.max(np.maximum(evaporation/m, 1e-30)))
        return ds, forces, temperature_rate, humidity_rate, stable_dt

    def source_step(self, velocity, temp, q, dt):
        """Subcycled explicit-midpoint exchange; equal/opposite heat and drag."""
        remaining = dt
        while remaining > 1e-15 and len(self.state):
            state0 = self.state
            first = self.rates(state0, velocity, temp, q)
            step = min(remaining, first[4])
            # Keep a step below one quarter cell of parcel travel.
            step = min(step, .25/np.max(np.sum(np.abs(state0[:, 3:6])/self.grid.h, axis=1)))
            middle = state0+.5*step*first[0]
            vm = [v+.5*step*a for v, a in zip(velocity, first[1])]
            tm, qm = temp+.5*step*first[2], q+.5*step*first[3]
            second = self.rates(middle, vm, tm, qm)
            new = state0+step*second[0]
            if np.any(new[:, 6] <= 0):
                raise RuntimeError('Nonpositive droplet mass: reduce time step')
            de = np.sum(self.number*(new[:, 7]-state0[:, 7]))
            dm = np.sum(self.number*(state0[:, 6]-new[:, 6]))
            gas_heat = step*np.sum(second[2])*RHO*CP*self.grid.volume
            self.energy_residual = max(self.energy_residual, abs(de+LV*dm+gas_heat))
            drop_drag = step*np.sum((self.number*middle[:, 6])[:, None]
                                   *(second[0][:, 3:6]-GRAVITY), axis=0)
            gas_drag = step*np.array([a.sum() for a in second[1]])*RHO*self.grid.volume
            self.momentum_residual = max(self.momentum_residual, float(np.max(abs(drop_drag+gas_drag))))
            for v, a in zip(velocity, second[1]):
                v += step*a
            temp += step*second[2]
            q += step*second[3]
            self.evaporated_mass += dm
            self.state = new
            # Wet-wall trapping of normal motion, tangential sliding retained.
            # No separate wall film; normal collision impulse goes to the wall.
            for axis in (1, 2):
                hit = (new[:, axis] <= 0) | (new[:, axis] >= self.grid.length[axis])
                self.wall_impulse[axis] += np.sum(self.number[hit]*new[hit, 6]*new[hit, 3+axis])
                new[hit, axis] = np.clip(new[hit, axis], 1e-12, self.grid.length[axis]-1e-12)
                new[hit, 3+axis] = 0.
            escaped = (new[:, 0] >= self.grid.length[0]) | (new[:, 0] < 0)
            # Finish submicron drops with conservative residual conversion, avoiding
            # infinitely many explicit substeps as the d^2 evaporation law dries out.
            dry = (~escaped) & (new[:, 6] < RHOW*np.pi/6*(1e-6)**3)
            if np.any(dry):
                residual_mass = self.number[dry]*new[dry, 6]
                residual_energy = self.number[dry]*new[dry, 7]
                q += self.grid.deposit(residual_mass, new[dry, :3])/(RHO*self.grid.volume)
                temp += self.grid.deposit(residual_energy-LV*residual_mass, new[dry, :3])/(RHO*CP*self.grid.volume)
                self.evaporated_mass += np.sum(residual_mass)
            self.escaped_mass += np.sum(self.number[escaped]*new[escaped, 6])
            keep = ~(escaped | dry)
            self.state, self.number = new[keep], self.number[keep]
            remaining -= step
        velocity, pressure = self.grid.project(velocity, dt)
        return velocity, temp, q, pressure

    def liquid_mass(self):
        return float(np.sum(self.number*self.state[:, 6]))


def self_test():
    rng = np.random.default_rng(12)
    g = Grid((8, 6, 6), length=(2., 1., 1.))
    p = rng.normal(size=tuple(g.n))
    mismatch = np.max(np.abs((g.A @ p.ravel()).reshape(p.shape)+g.divergence(g.gradient(p))))
    assert mismatch < 1e-11, mismatch
    assert sparse.linalg.norm(g.A-g.A.T) == 0
    assert p.ravel() @ (g.A @ p.ravel()) > 0
    v = [rng.normal(size=a.shape) for a in g.zeros_velocity()]
    g.enforce(v, 0.)
    before = np.linalg.norm(g.divergence(v))
    v, _ = g.project(v, .01, inlet=0.)
    reduction = np.linalg.norm(g.divergence(v))/before
    assert reduction < 5e-9, reduction
    errors = []
    for n in (12, 24, 48):
        # Smooth p: zero derivative at x=0, p=0 at x=L, homogeneous side Neumann.
        mesh = Grid((n, 4, 4), length=(2., 1., 1.))
        x = (np.arange(n)+.5)*mesh.h[0]
        exact = np.broadcast_to(np.cos(np.pi*x[:, None, None]/4), tuple(mesh.n))
        b = (np.pi/4)**2*exact
        solution, info = cg(mesh.A, b.ravel(), M=mesh.M, rtol=1e-11)
        assert info == 0
        errors.append(float(np.sqrt(np.mean((solution.reshape(exact.shape)-exact)**2))))
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all(orders > 1.98), orders
    momentum_errors = []
    for n in (16, 32, 64):
        mesh = Grid((n, n//2, 6), length=(2., 1., 1.), cs=0.)
        vel = mesh.zeros_velocity()
        a, b = np.pi/2, np.pi
        xu = np.arange(n+1)*mesh.h[0]
        yu = (np.arange(n//2)+.5)*mesh.h[1]
        xv = (np.arange(n)+.5)*mesh.h[0]
        yv = np.arange(n//2+1)*mesh.h[1]
        vel[0][:] = np.sin(a*xu[:, None, None])*np.cos(b*yu[None, :, None])
        vel[1][:] = -a/b*np.cos(a*xv[:, None, None])*np.sin(b*yv[None, :, None])
        rhs, _ = mesh.momentum_rhs(vel)
        exact_u = -a*np.sin(a*xu[:, None, None])*np.cos(a*xu[:, None, None]) - MU/RHO*(a*a+b*b)*vel[0]
        exact_v = -a*a/b*np.sin(b*yv[None, :, None])*np.cos(b*yv[None, :, None]) - MU/RHO*(a*a+b*b)*vel[1]
        interior = (slice(2, -2),)*3
        momentum_errors.append(float(np.sqrt(sum(np.mean((r-e)[interior]**2)
                                                for r, e in zip(rhs[:2], (exact_u, exact_v))))))
    momentum_orders = np.log2(np.array(momentum_errors[:-1])/momentum_errors[1:])
    assert np.all(momentum_orders > 1.8), momentum_orders
    locations = rng.uniform([0, 0, 0], g.length, (100, 3))
    values = rng.normal(size=100)
    for component in (None, 0, 1, 2):
        deposited = g.deposit(values, locations, component)
        assert abs(deposited.sum()-values.sum()) < 1e-12
        f = rng.normal(size=deposited.shape)
        assert abs(np.sum(f*deposited)-np.dot(values, g.gather(f, locations, component))) < 1e-12
    velocity = g.zeros_velocity()
    # Exact Couette shear away from walls verifies |S| and classical coefficient.
    y = (np.arange(g.n[1])+.5)*g.h[1]
    velocity[0][:] = 2*y[None, :, None]
    _, _, nut = g.strain(velocity)
    assert np.allclose(nut[1:-1, 1:-1, 1:-1], (g.cs*g.volume**(1/3))**2*2)
    velocity = g.zeros_velocity()
    velocity[0].fill(UIN)
    temp, q = np.full(tuple(g.n), TIN), np.full(tuple(g.n), inlet_q())
    spray = Spray(g)
    spray.inject(.002)
    velocity, temp, q, _ = spray.source_step(velocity, temp, q, 1e-4)
    assert spray.energy_residual < 1e-9
    assert spray.momentum_residual < 1e-12
    mass_error = spray.injected_mass-spray.liquid_mass()-spray.escaped_mass-spray.evaporated_mass
    assert abs(mass_error) < 1e-14
    assert np.all(np.isfinite(temp)) and np.min(q) > 0
    # Exercise terminal dryout: residual liquid enthalpy + vapour latent energy
    # must balance the gas sensible increment, including the final conversion.
    tiny = Spray(g)
    tiny.inject(1/tiny.rate)
    tiny.state[:, :3] = g.length/2
    tiny.state[:, 3:6] = [UIN, 0., 0.]
    tiny.state[:, 6] = RHOW*np.pi/6*(.5e-6)**3
    tiny.state[:, 7] = tiny.state[:, 6]*CPL*TLIN
    tiny.number[:] = 1e10
    tiny.injected_mass = tiny.liquid_mass()
    liquid_energy = float(np.sum(tiny.number*tiny.state[:, 7]))
    temp0, q0 = temp.copy(), q.copy()
    velocity, temp, q, _ = tiny.source_step(velocity, temp, q, 1e-6)
    dryout_energy_error = float(abs(RHO*g.volume*(CP*np.sum(temp-temp0)+LV*np.sum(q-q0))-liquid_energy))
    dryout_water_error = float(abs(RHO*g.volume*np.sum(q-q0)-tiny.injected_mass))
    assert not len(tiny.state)
    assert dryout_energy_error < 1e-9, dryout_energy_error
    assert dryout_water_error < 1e-14, dryout_water_error
    result = dict(operator_error=mismatch, projection_divergence_ratio=reduction,
                  poisson_l2_errors=errors, poisson_orders=orders.tolist(),
                  momentum_l2_errors=momentum_errors, momentum_orders=momentum_orders.tolist(),
                  source_energy_error_J=spray.energy_residual,
                  source_drag_error_kg_m_s=spray.momentum_residual,
                  liquid_mass_error_kg=mass_error, dryout_energy_error_J=dryout_energy_error,
                  dryout_water_error_kg=dryout_water_error)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=3, default=(64, 32, 32), metavar=('NX', 'NY', 'NZ'))
    parser.add_argument('--t-end', type=float, default=.1, help='Physical time [s]; default examines the developing jet')
    parser.add_argument('--dt', type=float, default=2e-4, help='Maximum step [s], further limited by CFL/diffusion')
    parser.add_argument('--cs', type=float, default=.16)
    parser.add_argument('--parcel-rate', type=float, default=4000., help='Computational parcels per second')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--no-spray', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('outputs/standalone_water_spray'))
    parser.add_argument('--save-every', type=float, default=.1, help='Field frame interval [s]')
    parser.add_argument('--mean-start', type=float, default=1., help='Start time for time-weighted field averages [s]')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.shape) < 4 or args.t_end <= 0 or args.dt <= 0 or args.cs < 0 or args.parcel_rate <= 0 or args.save_every <= 0:
        parser.error('Require grid dimensions >=4, positive times/rate, and nonnegative Cs')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'history.csv').exists():
        parser.error('Output already contains history.csv; choose a new --output directory')
    started = time.perf_counter()
    grid = Grid(args.shape, cs=args.cs)
    print(f'CPU MAC grid {tuple(args.shape)}; A={grid.A.shape}, nnz={grid.A.nnz}; '
          f'AMG levels={len(grid.amg.levels)}; q_in={inlet_q():.9f}', flush=True)
    velocity = grid.zeros_velocity()
    velocity[0].fill(UIN)
    temp, q = np.full(tuple(grid.n), TIN), np.full(tuple(grid.n), inlet_q())
    pressure = np.zeros(tuple(grid.n))
    spray = Spray(grid, args.parcel_rate, args.seed)
    means = {key: np.zeros(tuple(grid.n)) for key in ('u', 'v', 'w', 'T', 'q')}
    mean_duration = 0.
    t, step, next_save = 0., 0, args.save_every
    config = vars(args).copy()
    config['output'] = str(args.output)
    config.update(length_m=grid.length.tolist(), density_kg_m3=RHO, inlet_T_C=TIN,
                  inlet_wbt_C=TWB, inlet_q=inlet_q(), water_kg_s=0 if args.no_spray else MDOT,
                  pressure_solver='explicit CSR -DG; symmetric PyAMG V cycle + SciPy CG',
                  spatial_order=2, carrier_time_integrator='projected SSPRK3',
                  coupling='Strang split, midpoint parcel/source substeps',
                  inlet='uniform, no synthetic inlet turbulence')
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    def save(name):
        np.savez_compressed(args.output/name, time=t, h=grid.h, length=grid.length,
                            u=velocity[0], v=velocity[1], w=velocity[2], T=temp, q=q,
                            last_pressure_correction_kinematic=pressure, parcels=spray.state,
                            parcel_number=spray.number)
    with (args.output/'history.csv').open('w', newline='') as stream:
        names = ['step', 'time_s', 'dt_s', 'wall_s', 'parcels', 'max_abs_velocity_component_m_s',
                 'outlet_centre_u_m_s', 'div_linf_s_inv', 'cg_iterations', 'cg_relative_residual',
                 'min_T_C', 'max_T_C', 'min_q', 'max_q', 'injected_kg', 'liquid_kg',
                 'escaped_kg', 'evaporated_kg', 'water_balance_kg', 'source_energy_error_J']
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        save('initial.npz')
        while t < args.t_end-1e-14:
            _, _, nut = grid.strain(velocity)
            advection = sum(np.max(abs(v))/h for v, h in zip(velocity, grid.h))
            diffusion = 2*np.max(np.maximum(MU/RHO+nut, K/(RHO*CP)+nut/.7))*np.sum(1/grid.h**2)
            dt = min(args.dt, .3/max(advection, 1e-20), .4/max(diffusion, 1e-20), args.t_end-t)
            if dt < 1e-10:
                raise RuntimeError('Time step collapsed: solution unstable')
            # Births scheduled at the step midpoint. Injection quantization is at
            # most one parcel mass; it is reported, never silently renormalized.
            if not args.no_spray:
                spray.inject(t+.5*dt)
                velocity, temp, q, pressure = spray.source_step(velocity, temp, q, .5*dt)
            velocity, temp, q, pressure = grid.carrier_step(velocity, temp, q, dt)
            if not args.no_spray:
                velocity, temp, q, pressure = spray.source_step(velocity, temp, q, .5*dt)
            t += dt
            step += 1
            if not all(np.isfinite(a).all() for a in (*velocity, temp, q)) or q.min() < 0 or temp.min() < -50 or temp.max() > 100:
                save('failed.npz')
                raise RuntimeError('Nonphysical field: centered scalar transport has no positivity limiter; reduce dt/refine')
            weight = max(0., t-max(t-dt, args.mean_start))
            if weight:
                for key, field in zip(('u', 'v', 'w', 'T', 'q'),
                                       [average(v, i) for i, v in enumerate(velocity)]+[temp, q]):
                    means[key] += weight*field
                mean_duration += weight
            liquid = spray.liquid_mass()
            row = dict(zip(names, [step, t, dt, time.perf_counter()-started, len(spray.state),
                       max(float(np.max(abs(a))) for a in velocity),
                       float(np.mean(velocity[0][-1, grid.n[1]//2-1:grid.n[1]//2+1,
                                                grid.n[2]//2-1:grid.n[2]//2+1])),
                       float(np.max(abs(grid.divergence(velocity)))), grid.last_iterations, grid.last_residual,
                       float(temp.min()), float(temp.max()), float(q.min()), float(q.max()),
                       spray.injected_mass, liquid, spray.escaped_mass, spray.evaporated_mass,
                       spray.injected_mass-liquid-spray.escaped_mass-spray.evaporated_mass,
                       spray.energy_residual]))
            writer.writerow(row)
            if step == 1 or step % 25 == 0 or t >= args.t_end-1e-14:
                print(f't={t:.6f} step={step} parcels={len(spray.state)} '
                      f'u_out={row["outlet_centre_u_m_s"]:.4f} '
                      f'div={row["div_linf_s_inv"]:.2e} CG={grid.last_iterations} '
                      f'wall={row["wall_s"]:.1f}s', flush=True)
                stream.flush()
            if t >= next_save-1e-14:
                save(f'frame_{step:07d}.npz')
                next_save += args.save_every
        save('final.npz')
    if mean_duration:
        np.savez_compressed(args.output/'mean.npz', **{k: v/mean_duration for k, v in means.items()},
                            duration=mean_duration, h=grid.h, length=grid.length)
    summary = dict(row, mean_duration_s=mean_duration, max_cg_iterations=grid.max_iterations,
                   max_drag_exchange_error_kg_m_s=spray.momentum_residual,
                   parcel_wall_impulse_kg_m_s=spray.wall_impulse.tolist())
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Wrote {args.output}/final.npz and history.csv', flush=True)


if __name__ == '__main__':
    main()
