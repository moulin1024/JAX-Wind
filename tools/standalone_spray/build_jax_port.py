"""Development helper: retain audited finite differences while porting their arrays."""
from pathlib import Path
p = Path(__file__).parent
s = (p/'standalone_water_spray.py').read_text()
constants = s[s.index('RHO, MU'):s.index('class Grid:')]
constants = constants.replace('return float((287.05/461.5)*e/(PRESSURE-e))', 'return (287.05/461.5)*e/(PRESSURE-e)')
methods = s[s.index('    def ghosts('):s.index('    def carrier_step(')]
methods = methods.replace('diff[-1] = -2*stress[-1]/self.h[i]', 'diff = diff.at[-1].set(-2*stress[-1]/self.h[i])')
methods = methods.replace('rhs[0][0] = 0.', 'rhs[0] = rhs[0].at[0].set(0.)')
methods = methods.replace('rhs[1][:, 0, :] = rhs[1][:, -1, :] = 0.', 'rhs[1] = rhs[1].at[:, 0, :].set(0.).at[:, -1, :].set(0.)')
methods = methods.replace('rhs[2][:, :, 0] = rhs[2][:, :, -1] = 0.', 'rhs[2] = rhs[2].at[:, :, 0].set(0.).at[:, :, -1].set(0.)')
methods = methods.replace('sf[0] = inlet', 'sf = sf.at[0].set(inlet)').replace('grad[0] = 2*(scalar[0]-inlet)/self.h[0]', 'grad = grad.at[0].set(2*(scalar[0]-inlet)/self.h[0])')
header='''#!/usr/bin/env python3
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

'''
(p/'standalone_water_spray_jax.py').write_text(header+constants+'\n')
(p/'jax_fd_methods.txt').write_text(methods)
