#!/usr/bin/env python3
"""Check a moving analytic jet front, not just its momentum integral."""

import argparse
import json
from pathlib import Path

import numpy as np
from jet_streamfunction import conductance, geometry, positive_step


def exact_cell_means(faces, x):
    # A=a=1, nu=1/8, psi_edge=x/2, U=x^-1 (1-psi/psi_edge)^2.
    primitive_tail = np.maximum(1 - faces / (x / 2), 0) ** 3 / 6
    return -np.diff(primitive_tail) / np.diff(faces)


def assess(n, step_factor=0.05, ambient=1e-8):
    faces = np.linspace(0, 2, n + 1)
    widths = np.diff(faces)
    u = exact_cell_means(faces, 1) + ambient
    initial = float(np.sum(widths * u))
    x = 1.0
    exported = 0.0
    while x < 2:
        dx = min(step_factor / n, 2 - x)
        coefficient = conductance(faces, u, 0.125, ambient, 0.125)
        u, transfer = positive_step(u, coefficient, widths, ambient, dx)
        exported += transfer
        x += dx
    target = exact_cell_means(faces, 2) + ambient
    _, radius2 = geometry(faces, u)
    # Compare the radial half-width as well as streamfunction-cell velocities.
    centers = (faces[:-1] + faces[1:]) / 2
    uc = u[0] - centers[0] * (u[1] - u[0]) / (centers[1] - centers[0])
    half = np.interp(uc / 2, u[::-1], np.sqrt(radius2)[::-1])
    exact_half = 2 * np.sqrt(np.sqrt(2) - 1)
    return {
        "cells": n,
        "step_factor": step_factor,
        "ambient_velocity": ambient,
        "maximum_velocity_error_over_exact_Uc": float(np.max(abs(u - target)) / 0.5),
        "relative_half_width_error": float(abs(half / exact_half - 1)),
        "momentum_budget_error": float(np.sum(widths * u) + exported - initial),
        "minimum_velocity": float(u.min()),
        "maximum_positive_velocity_increment": float(max(0, np.diff(u).max())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for n in [128, 256, 512, 1024]:
        row = assess(n)
        rows.append(row)
        print(json.dumps(row), flush=True)
    # Separate marching-step and coflow sensitivity on a fixed grid.
    for factor, ambient in [(0.025, 1e-8), (0.05, 1e-9)]:
        row = assess(512, factor, ambient)
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analytic-jet.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
