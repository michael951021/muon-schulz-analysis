"""Reproduce independent numerical audit; writes reports/audit_metrics.json.

Run with .venv/bin/python scripts/audit_results.py. No solver or dataset needed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import torch

from nsshape.coefficients import TABLE_PATH, load_table
from nsshape.optim.orthogonalize import orthogonalize


def polynomial(coeffs, x):
    a, b, c = coeffs
    return x * (a + x*x * (b + c*x*x))


def critical(coeffs, bound):
    a, b, c = coeffs
    roots = np.roots([5*c, 3*b, a])
    return [float(np.sqrt(r.real)) for r in roots
            if abs(r.imag) < 1e-12 and 0 < r.real < bound*bound]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('reports/audit_metrics.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    table = load_table()
    rows = []
    for name, cs in table.items():
        if cs.coeffs is None:
            continue
        # Analytical single-step extrema, independent of production grid checks.
        candidates = [0., cs.invariance_bound, *critical(cs.coeffs, cs.invariance_bound)]
        maximum = max(polynomial(cs.coeffs, x) for x in candidates)
        a, b, c = cs.coeffs
        us = [0., cs.invariance_bound**2]
        if c > 0 and 0 < -b/(2*c) < cs.invariance_bound**2:
            us.append(-b/(2*c))
        positive_factor = min(a+b*u+c*u*u for u in us)
        x = np.geomspace(cs.sigma_min, 1., 1_000_000)
        for _ in range(cs.n_steps):
            x = polynomial(cs.coeffs, x)
        lo, hi = float(x.min()), float(x.max())
        # Every normalized input of this controlled spectrum is inside the domain.
        torch.manual_seed(7)
        u, _ = torch.linalg.qr(torch.randn(64, 64))
        v, _ = torch.linalg.qr(torch.randn(64, 64))
        s = torch.logspace(np.log10(cs.sigma_min*3), 0, 64)
        s /= s.norm()
        if s.min() < cs.sigma_min:
            s = torch.logspace(np.log10(cs.sigma_min*8), 0, 64)
            s /= s.norm()
        assert s.min() >= cs.sigma_min
        matrix = u @ torch.diag(s) @ v.T
        out = orthogonalize(matrix, cs)
        actual = torch.linalg.svdvals(out.float())
        rows.append(dict(name=name, gamma=cs.gamma, sigma_min=cs.sigma_min,
                         dense_half_width=(hi-lo)/(hi+lo),
                         normalized_min=lo/cs.scale, normalized_max=hi/cs.scale,
                         analytic_invariance_violation=float(maximum-cs.invariance_bound),
                         min_positive_factor=float(positive_factor),
                         controlled_input_min=float(s.min()),
                         bf16_half_width=float((actual.max()-actual.min())/(actual.max()+actual.min()))))
    zero = torch.zeros(8, 8)
    result = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                  table_sha256=hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest(),
                  dense_grid_points=1_000_000, rows=rows,
                  zero_update_norms={name: float(orthogonalize(zero, cs).norm())
                                     for name, cs in table.items()})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
