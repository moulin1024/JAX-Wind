"""Two smooth BVP regions joined at an unknown realizability transition."""

from types import SimpleNamespace

import numpy as np
from constrained_kepsilon import fields, rhs
from scipy.integrate import solve_bvp
from scipy.integrate._bvp import estimate_rms_residuals
from scipy.optimize import brentq
from solve_constrained_kepsilon import assess_solution, physical


def solve(seed, constants, delta, tol, nodes):
    edge = float(np.exp(seed.p[0]))
    # Locate the initial transition from source-only fields, not measured widths.
    def crossing(z):
        state = physical(np.array([z]), seed.sol(np.array([z])), edge)
        return float(fields(np.array([edge*z]), state, constants,
                            trial=True, active_branch=True)["fraction"][0]-1)

    switch = brentq(crossing, 0.99, min(seed.x[-1], 1-delta), xtol=1e-14)
    xi = np.linspace(0, 1, nodes)
    inner_z, outer_z = switch*xi, switch+(1-delta-switch)*xi
    y = np.concatenate([seed.sol(inner_z), seed.sol(np.minimum(outer_z, seed.x[-1]))])
    tail = outer_z > seed.x[-1]
    if np.any(tail):
        ratio = (1-outer_z[tail])/(1-seed.x[-1])
        last = seed.sol(np.array([seed.x[-1]]))[:, 0]
        y[7, tail] = last[1]*ratio
        y[8, tail] = last[2]+np.log(ratio)
        y[10, tail] = last[4]+constants.sigma_e*np.log(ratio)
        y[9, tail] = -edge**2*y[6, tail]*np.exp(y[8, tail])
        y[11, tail] = -edge**2*y[6, tail]*np.exp(y[10, tail])

    def geometry(parameter):
        edge = np.exp(np.clip(parameter[0], -5, 3))
        gap = np.exp(np.clip(parameter[1], np.log(delta*1.001), np.log(0.1)))
        return edge, 1-gap, gap-delta

    def spatial(z, y, edge, constrained=True, active_branch=False):
        state = physical(z, y, edge)
        d = rhs(edge*z, state, constants, trial=True,
                constrained=constrained, active_branch=active_branch)
        return np.array([d[0]/edge, edge*d[1], edge*d[2]/state[2],
                         edge*d[3], edge*d[4]/state[4], edge*d[5]])

    def equation(xi, y, parameter):
        edge, switch, length = geometry(parameter)
        inner = switch*spatial(switch*xi, y[:6], edge, constrained=False)
        outer = length*spatial(switch+length*xi, y[6:], edge, active_branch=True)
        return np.concatenate([inner, outer])

    def boundary(a, b, parameter):
        edge, switch, _ = geometry(parameter)
        seam = physical(np.array([switch]), b[:6, None], edge)
        value = fields(np.array([edge*switch]), seam, constants, trial=True, active_branch=True)
        state = physical(np.array([1-delta]), b[6:, None], edge)
        ii, _, k, qk, e, qe = state[:, 0]
        ii = max(ii, 1e-20)
        n = fields(np.array([edge*(1-delta)]), state, constants,
                   trial=True, active_branch=True)["viscosity"][0]
        return np.r_[a[[0, 3, 5]], a[1]-1, b[:6]-a[6:], value["fraction"][0]-1,
                     n/(ii*delta/(1-delta))-1, qk/(ii*k)+1, qe/(ii*e)+1]

    raw = solve_bvp(equation, boundary, xi, y, p=np.array([np.log(edge), np.log1p(-switch)]),
                    tol=tol/10, bc_tol=1e-10, max_nodes=30000)
    edge, switch, length = geometry(raw.p)
    bound = float(np.max(abs(boundary(raw.y[:, 0], raw.y[:, -1], raw.p))))
    z = np.unique(np.r_[switch*raw.x, switch+length*raw.x])

    def sol(z, nu=0):
        z = np.asarray(z)
        result = np.empty((6,)+z.shape)
        inside = z <= switch
        result[:, inside] = raw.sol(z[inside]/switch, nu)[:6]/switch**nu
        result[:, ~inside] = raw.sol((z[~inside]-switch)/length, nu)[6:]/length**nu
        return result

    # Evaluate the original-z residual using native interval coordinates.
    # Affine Jacobians cancel in the interval-normalized Lobatto integral;
    # avoiding z -> xi inversion prevents cancellation in the narrow outer strip.
    parameter = raw.p[:1]
    residuals = []
    for component, offset, scale in [(slice(0, 6), 0.0, switch), (slice(6, 12), switch, length)]:
        def original(xi, y, parameter, offset=offset, scale=scale):
            return spatial(offset+scale*xi, y, np.exp(parameter[0]))

        def native(xi, nu=0, component=component, scale=scale):
            return raw.sol(xi, nu)[component]/scale**nu

        h = np.diff(raw.x)
        middle = raw.x[:-1]+h/2
        f_middle = original(middle, native(middle), parameter)
        residuals.append(estimate_rms_residuals(original, native, raw.x, h, parameter,
                                               native(middle, 1)-f_middle, f_middle))
    residual = np.concatenate(residuals)
    result = SimpleNamespace(x=z, y=sol(z), p=parameter, sol=sol,
        status=raw.status, message=raw.message, rms_residuals=residual)
    result, row, eta, f = assess_solution(result, constants, delta, tol, nodes,
        bound, "two smooth domains with solved transition", float(np.max(raw.rms_residuals)))
    gap_inactive = bool(np.log(delta*1.001) < raw.p[1] < np.log(0.1))
    row.update(transition_z=float(switch), transition_eta=float(edge*switch),
               transition_parameter_guard_inactive=gap_inactive, coupled_nodes=raw.x.size)
    row["basic_numerical_pass"] = row["basic_numerical_pass"] and gap_inactive
    return result, row, eta, f
