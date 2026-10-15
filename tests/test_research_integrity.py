"""Regression checks for audit failures affecting scientific comparisons."""
import copy
import json
import numpy as np
import pytest
import torch

from nsshape.coefficients import TABLE_PATH, get, load_table
from nsshape.optim.muon import ParameterizedMuon
from nsshape.optim.orthogonalize import exact_svd_polar, ns_quintic
from nsshape.polynomial import reachable_bound, compose, positivity_violation, band_extent
from nsshape.diagnostics.svd_logger import SVDLogger


def test_partial_polar_has_no_null_space_update():
    matrix = torch.zeros(8, 8)
    assert exact_svd_polar(matrix).count_nonzero() == 0
    matrix[0, 0] = 2
    expected = matrix / 2
    assert torch.allclose(exact_svd_polar(matrix), expected)


def test_exact_svd_optimizer_zero_gradient_is_noop():
    p = torch.randn(8, 8, requires_grad=True)
    initial = p.detach().clone()
    p.grad = torch.zeros_like(p)
    ParameterizedMuon([p], coeff_set='exact_svd').step()
    assert torch.equal(p, initial)


def test_checkpoint_restores_arm_precision_and_next_update():
    p = torch.randn(8, 8, requires_grad=True)
    opt = ParameterizedMuon([p], coeff_set='gamma_0.10', precision='fp32')
    p.grad = torch.randn_like(p)
    opt.step()
    checkpoint = copy.deepcopy(opt.state_dict())
    q = p.detach().clone().requires_grad_()
    resumed = ParameterizedMuon([q], coeff_set='keller')
    resumed.load_state_dict(checkpoint)
    assert resumed.coefficient_set == opt.coefficient_set
    assert resumed.precision == 'fp32' and resumed.step_count == 1
    p.grad = torch.randn_like(p)
    q.grad = p.grad.clone()
    opt.step()
    resumed.step()
    assert torch.equal(p, q)


def test_parameter_identity_survives_missing_gradients():
    p, q = [torch.randn(8, 8, requires_grad=True) for _ in range(2)]
    calls = []
    q.grad = torch.ones_like(q)
    ParameterizedMuon([p, q], hook=lambda step, index, *args: calls.append(index)).step()
    assert calls == [1]


def test_sampler_actually_uses_seed_and_population(tmp_path):
    def selected(seed):
        path = tmp_path / f'{seed}.jsonl'
        with SVDLogger(path, every_k=1, max_params=3, population_size=20, seed=seed) as log:
            for i in range(20):
                log(0, i, torch.eye(4), torch.eye(4))
        return {json.loads(line)['param_index'] for line in path.read_text().splitlines()}
    assert len(selected(7)) == 3
    assert selected(7) != selected(8)
    assert selected(7) != {0, 1, 2}


def test_logger_records_actual_normalized_domain(tmp_path):
    path = tmp_path / 'log.jsonl'
    with SVDLogger(path, every_k=1, coefficient_set=get('gamma_0.01')) as log:
        log(0, 0, torch.eye(4), torch.eye(4))
    record = json.loads(path.read_text().splitlines()[0])
    assert record['normalized_input_min'] == pytest.approx(.5)
    assert record['below_sigma_min'] == 0
    assert record['coefficient_set'] == 'gamma_0.01'


@pytest.mark.parametrize('name', [name for name, cs in load_table().items() if cs.coeffs])
def test_fp32_backend_matches_scalar_map_for_every_arm(name):
    cs = get(name)
    generator = torch.Generator().manual_seed(7)
    u, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator))
    v, _ = torch.linalg.qr(torch.randn(32, 32, generator=generator))
    s = torch.logspace(np.log10(cs.sigma_min*8), 0, 32)
    s /= s.norm()
    assert s.min() >= cs.sigma_min
    matrix = u @ torch.diag(s) @ v.T
    actual = torch.linalg.svdvals(ns_quintic(matrix, cs.coeffs, scale=cs.scale, precision='fp32'))
    expected = np.sort(compose(cs.coeffs, s.numpy()) / cs.scale)[::-1].copy()
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float32), atol=2e-4, rtol=2e-4)


def test_unconverged_bound_is_rejected():
    assert np.isinf(reachable_bound((1.001, 0, 0), max_iter=2))


def test_analytic_positivity_detects_double_zero():
    # P/x=(1-x²)² touches zero without going negative.
    assert positivity_violation((1, -2, 1), 1.1) == pytest.approx(0)


def test_band_certificate_cannot_skip_interior_oscillation():
    extent = band_extent(get('keller').coeffs, .15, n_steps=1, grid=np.array([.3, 1.]))
    assert extent.sigma_min == 1.


@pytest.mark.parametrize('name', [name for name, cs in load_table().items() if cs.coeffs])
def test_continuous_band_from_independent_polynomial_extrema(name):
    from numpy.polynomial import Polynomial
    cs = get(name)
    a, b, c = cs.coeffs
    polynomial = Polynomial([0, a, 0, b, 0, c])
    roots = polynomial.deriv().roots()
    lower, upper = cs.sigma_min, 1.
    for _ in range(cs.n_steps):
        candidates = [lower, upper]+[float(root.real) for root in roots
                      if abs(root.imag) < 1e-10 and lower < root.real < upper]
        values = polynomial(candidates)
        lower, upper = float(values.min()), float(values.max())
    assert lower > 0
    assert (upper-lower)/(upper+lower) <= cs.gamma+1e-10
    assert (upper+lower)/2 == pytest.approx(cs.scale, rel=1e-10)


def test_zero_spectrum_is_strict_json_with_undefined_ratio():
    from nsshape.diagnostics.schema import spectrum_record
    record = spectrum_record(0, 0, 'after', torch.zeros(4, 4))
    assert record.band_ratio is None
    assert 'Infinity' not in record.to_json()


@pytest.mark.parametrize('mutation', ['duplicate', 'version', 'scale'])
def test_loader_rejects_corrupt_tables(tmp_path, mutation):
    payload = json.loads(TABLE_PATH.read_text())
    if mutation == 'duplicate':
        payload['entries'].append(payload['entries'][0])
    elif mutation == 'version':
        payload['version'] = 99
    else:
        payload['entries'][0]['scale'] = 0
    path = tmp_path / 'table.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_table(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_logger(tmp_path):
    with SVDLogger(tmp_path / 'cuda.jsonl', every_k=1) as log:
        log(0, 0, torch.eye(8, device='cuda'), torch.eye(8, device='cuda'))
