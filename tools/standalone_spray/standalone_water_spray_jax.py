#!/usr/bin/env python3
"""Standalone JAX GPU spray LES: second-order MAC differences, matrix-free GMG-PCG.

No jaxwind, SciPy or PyAMG dependency. All time stepping, particle exchange and
pressure iterations execute on the selected JAX device. Temperatures in Celsius.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as host
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as np
from jax import lax

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
    return (287.05/461.5)*e/(PRESSURE-e)


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



def pressure_gradient(p, h):
    grad = [np.diff(pad(p, i), axis=i)/h[i] for i in range(3)]
    grad[0] = grad[0].at[-1].set(-2*p[-1]/h[0])
    return tuple(grad)


def divergence(velocity, h):
    return sum(np.diff(v, axis=i)/h[i] for i, v in enumerate(velocity))


def poisson(p, h):
    """Exact -D G stencil; Neumann walls/inlet, physical-outlet Dirichlet."""
    return -divergence(pressure_gradient(p, h), h)


def diagonal(shape, h):
    result = np.zeros(shape)
    for axis in range(3):
        d = np.full((shape[axis],), 2./h[axis]**2)
        d = d.at[0].set(1./h[axis]**2)
        d = d.at[-1].set((3. if axis == 0 else 1.)/h[axis]**2)
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        result += d.reshape(reshape)
    return result


def prolong(coarse):
    """Cell-centred trilinear P with even Neumann / odd outlet ghosts."""
    fine = coarse
    for axis in range(3):
        right = fine[plane(fine, axis, slice(-1, None))]
        padded = pad(fine, axis, high=-right if axis == 0 else right)
        left = padded[plane(padded, axis, slice(None, -2))]
        right = padded[plane(padded, axis, slice(2, None))]
        even, odd = .75*fine+.25*left, .75*fine+.25*right
        shape = list(fine.shape)
        shape[axis] *= 2
        fine = np.stack((even, odd), axis=axis+1).reshape(shape)
    return fine


def restrict(fine):
    """R = P.T/8 exactly, including outlet boundary weights."""
    coarse = fine
    for axis in (2, 1, 0):
        even = coarse[plane(coarse, axis, slice(0, None, 2))]
        odd = coarse[plane(coarse, axis, slice(1, None, 2))]
        zeros = np.zeros_like(even[plane(even, axis, slice(0, 1))])
        left = pad(odd, axis, low=zeros)[plane(odd, axis, slice(None, -2))]
        right = pad(even, axis, high=zeros)[plane(even, axis, slice(2, None))]
        coarse = .75*(even+odd)+.25*(left+right)
        sl = plane(coarse, axis, 0)
        coarse = coarse.at[sl].add(.25*even[sl])
        sl = plane(coarse, axis, -1)
        coarse = coarse.at[sl].add((-.25 if axis == 0 else .25)*odd[sl])
        coarse *= .5
    return coarse


class GMG:
    """Linear symmetric V cycle: adjoint transfers and weighted-Jacobi smoothing.

    Coarse operators are rediscretized -DG, not assembled matrices. Fixed
    iteration counts preserve linearity/SPD of the preconditioner for PCG.
    """
    def __init__(self, shape, h, rtol=1e-9):
        self.shapes, self.spacing = [tuple(shape)], [tuple(h)]
        while min(self.shapes[-1]) >= 4 and all(n % 2 == 0 for n in self.shapes[-1]):
            self.shapes.append(tuple(n//2 for n in self.shapes[-1]))
            self.spacing.append(tuple(2*d for d in self.spacing[-1]))
        self.diagonals = [diagonal(n, d) for n, d in zip(self.shapes, self.spacing)]
        self.rtol = rtol
        self.precondition = jax.jit(lambda b: self.cycle(b, 0))
        self.solve = jax.jit(self.pcg)

    def smooth(self, x, b, level, count):
        h, d = self.spacing[level], self.diagonals[level]
        return lax.fori_loop(0, count, lambda _, v: v+(2/3)*(b-poisson(v, h))/d, x)

    def cycle(self, b, level):
        x = np.zeros_like(b)
        if level == len(self.shapes)-1:
            return self.smooth(x, b, level, 100)
        x = self.smooth(x, b, level, 3)
        residual = b-poisson(x, self.spacing[level])
        x += prolong(self.cycle(restrict(residual), level+1))
        return self.smooth(x, b, level, 3)

    def pcg(self, b):
        h = self.spacing[0]
        normb = np.linalg.norm(b)
        tolerance = np.maximum(self.rtol*normb, 1e-11)
        x, r = np.zeros_like(b), b
        z = self.precondition(r)
        rz = np.vdot(r, z)
        initial = (np.int32(0), x, r, z, rz)
        def condition(carry):
            k, _, r, _, _ = carry
            return (k < 200) & (np.linalg.norm(r) > tolerance)
        def body(carry):
            k, x, r, direction, rz = carry
            ad = poisson(direction, h)
            alpha = rz/np.maximum(np.vdot(direction, ad), 1e-300)
            x, r = x+alpha*direction, r-alpha*ad
            z = self.precondition(r)
            next_rz = np.vdot(r, z)
            direction = z+(next_rz/np.maximum(rz, 1e-300))*direction
            return k+1, x, r, direction, next_rz
        k, x, _, _, _ = lax.while_loop(condition, body, initial)
        error = np.linalg.norm(b-poisson(x, h))
        relative = error/np.maximum(normb, 1e-300)
        success = np.isfinite(error) & (error <= 5*tolerance)
        return x, k, relative, success


class Grid:
    def __init__(self, shape, length=(1.9, .585, .585), cs=.16):
        self.n = tuple(shape)
        self.length = host.asarray(length)
        self.h = tuple(self.length/host.asarray(shape))
        self.volume = float(host.prod(self.h))
        self.cs = cs
        self.gmg = GMG(shape, self.h)

    def zeros_velocity(self):
        return tuple(np.zeros(tuple(n+(i == j) for j, n in enumerate(self.n))) for i in range(3))

    def enforce(self, velocity, inlet=UIN):
        u, v, w = velocity
        return (u.at[0].set(inlet), v.at[:, 0, :].set(0.).at[:, -1, :].set(0.),
                w.at[:, :, 0].set(0.).at[:, :, -1].set(0.))

    def project(self, velocity, dt, inlet=UIN):
        velocity = self.enforce(velocity, inlet)
        p, iterations, residual, ok = self.gmg.solve(-divergence(velocity, self.h)/dt)
        velocity = tuple(v-dt*grad for v, grad in zip(velocity, pressure_gradient(p, self.h)))
        return self.enforce(velocity, inlet), p, iterations, residual, ok

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
                diff = diff.at[-1].set(-2*stress[-1]/self.h[i])
            out += diff
            for j in range(3):
                if j != i:
                    a, b = sorted((i, j))
                    stress = to_faces(to_faces(viscosity, a), b)*shear[a, b]
                    out += np.diff(stress, axis=j)/self.h[j]
            rhs.append(out)
        rhs[0] = rhs[0].at[0].set(0.)
        rhs[1] = rhs[1].at[:, 0, :].set(0.).at[:, -1, :].set(0.)
        rhs[2] = rhs[2].at[:, :, 0].set(0.).at[:, :, -1].set(0.)
        return rhs, nut

    def scalar_rhs(self, scalar, velocity, diffusivity, inlet):
        out = np.zeros_like(scalar)
        for i in range(3):
            sf = to_faces(scalar, i)
            grad = np.diff(pad(scalar, i), axis=i)/self.h[i]
            if i == 0:
                sf = sf.at[0].set(inlet)
                grad = grad.at[0].set(2*(scalar[0]-inlet)/self.h[0])
            flux = velocity[i]*sf-to_faces(diffusivity, i)*grad
            out -= np.diff(flux, axis=i)/self.h[i]
        return out

