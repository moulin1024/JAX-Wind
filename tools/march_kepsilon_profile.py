"""Source-only k-epsilon boundary-layer march; no held-out data are read.

This reduced model is unqualified until numerical and source-reproduction
gates pass. Positive states alone do not establish a physical solution.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from jet_streamfunction import conductance, diffusion_rate, geometry, positive_step


def march(c3, n=500, step=0.003, xend=300, um=1e-7, psimax=80, kfactor=1, output=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if c3 not in (0, 0.79) or n < 4 or min(step, xend, um, psimax, kfactor) <= 0:
        raise ValueError("invalid source-march controls")
    faces = np.expm1(np.linspace(0, np.log1p(psimax / 0.08), n + 1)) * 0.08
    widths = np.diff(faces)
    psi = (faces[1:] + faces[:-1]) / 2
    rc = np.r_[
        np.linspace(0, 8, 20001),
        np.geomspace(8.01, np.sqrt(2 * psimax / um) * 1.1, 10000),
    ]
    initial_psi = um * rc**2 / 2 + (1 - um) / 2 * (-np.expm1(-(rc**2)))
    r = np.interp(psi, initial_psi, rc)
    u = um + (1 - um) * np.exp(-r * r)
    ka = um**2
    ea = ka**1.5
    k = ka + 0.08 * kfactor * np.exp(-r * r * 0.5)
    e = ea + 0.12 * kfactor**1.5 * np.exp(-r * r * 0.75)

    def geom(u):
        return geometry(faces, u)

    def coeff(u, k, e):
        _, r2 = geom(u)
        nu = 0.09 * k * k / e
        return conductance(faces, u, nu, um, 0.09 * ka * ka / ea), r2, nu

    def diff(v, d, ambient):
        return diffusion_rate(v, d, widths, ambient)

    def update(v, d, ambient, dt, source=0, sink=0):
        return positive_step(v, d, widths, ambient, dt, source, sink)[0]

    x = 0.0
    it = 0
    history = []
    start = time.time()
    nextlog = 1.0
    j0 = float(np.sum(u * widths))
    bad = False
    exported = 0.0
    reason = None
    while x < xend:
        dt = min(step * max(1.0, x), xend - x)
        try:
            d, r2, nu = coeff(u, k, e)
            ux = diff(u, d, um)
            integral = np.cumsum(widths * ux / u**2) - 0.5 * widths * ux / u**2
            vr = -u / r2 * integral
            up = np.gradient(u, psi, edge_order=2)
            ur = np.sqrt(r2) * u * up
            production = nu * ur**2
            chi = (k / e) ** 3 * 0.25 * ur**2 * vr
            epsrate = e / k / u
            unew, transfer = positive_step(u, d, widths, um, dt)
            knew = update(k, d, ka, dt, source=production / u, sink=epsrate)
            enew = update(
                e,
                d / 1.3,
                ea,
                dt,
                source=1.45 * e / k * production / u
                + np.maximum(c3 * chi, 0) * epsrate * e,
                sink=(1.9 + np.maximum(-c3 * chi, 0)) * epsrate,
            )
        except (ValueError, FloatingPointError) as exc:
            bad = True
            reason = str(exc)
            break
        if not all(
            np.all(np.isfinite(v)) and np.all(v > 0) for v in [unew, knew, enew]
        ):
            bad = True
            reason = "nonpositive or nonfinite state"
            break
        budget = float((np.sum(unew * widths) + exported + transfer) / j0 - 1)
        if abs(budget) > 1e-10:
            bad = True
            reason = "momentum budget exceeds 1e-10"
            break
        exported += transfer
        u, k, e = unew, knew, enew
        x += dt
        it += 1
        if x >= nextlog or x == xend:
            _, r2 = geom(u)
            uc = u[0] - psi[0] * (u[1] - u[0]) / (psi[1] - psi[0])
            half = float(np.interp(uc / 2, u[::-1], np.sqrt(r2)[::-1]))
            row = {
                "x": x,
                "uc": uc,
                "half": half,
                "kc": k[0] / uc**2,
                "ec": e[0] * x / uc**3,
                "momentum": float(np.sum(u * widths)),
                "chi_min": float(chi.min()),
                "chi_max": float(chi.max()),
            }
            history.append(row)
            print(json.dumps(dict(c3=c3, **row)), flush=True)
            nextlog = max(nextlog * 1.5, x * 1.01)
    tag = f"source_{c3}_{n}_{step}_{xend}_{um}_{psimax}_{kfactor}"
    np.savez(
        output / ("march_" + tag + ".npz"),
        psi=psi,
        faces=faces,
        u=u,
        k=k,
        e=e,
        r=np.sqrt(geom(u)[1]),
        x=x,
        history=json.dumps(history),
    )
    report = {
        "c3": c3,
        "n": n,
        "step": step,
        "steps": it,
        "elapsed": time.time() - start,
        "failed": bad,
        "failure_reason": reason,
        "final_x": x,
        "momentum_relative_error": float((np.sum(u * widths) + exported) / j0 - 1),
        "tag": tag,
        "ambient_velocity": um,
        "psi_max": psimax,
        "inlet_k_factor": kfactor,
    }
    late = [row for row in history if row["x"] >= 0.4 * xend]
    if len(late) >= 2:
        slope, intercept = np.polyfit(
            [row["x"] for row in late], [row["half"] for row in late], 1
        )
        report.update(
            slope=float(slope),
            virtual_origin=float(-intercept / slope),
            source_relative_error=float(abs(slope / (0.125 if c3 == 0 else 0.086) - 1)),
        )
    report["source_reproduction_pass"] = bool(
        not bad and report.get("source_relative_error", float("inf")) <= 0.02
    )
    report["holdout_accessed"] = False
    print(json.dumps(report), flush=True)
    (output / ("report_" + tag + ".json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c3", type=float, choices=[0, 0.79], required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--step", type=float, default=0.0015)
    p.add_argument("--xend", type=float, default=300)
    p.add_argument("--um", type=float, default=1e-7)
    p.add_argument("--psimax", type=float, default=80)
    p.add_argument("--kfactor", type=float, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = march(a.c3, a.n, a.step, a.xend, a.um, a.psimax, a.kfactor, a.output)
    if report["failed"]:
        raise SystemExit(2)
