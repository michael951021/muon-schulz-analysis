import pytest

torch = pytest.importorskip("torch")

from nsshape.optim.muon import ParameterizedMuon


def _param(shape, seed=0):
    torch.manual_seed(seed)
    p = torch.randn(*shape, requires_grad=True)
    p.grad = torch.randn(*shape)
    return p


def test_step_changes_parameters():
    p = _param((16, 16))
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1).step()
    assert not torch.allclose(p.detach(), before)


def test_step_is_a_noop_without_gradients():
    p = _param((16, 16))
    p.grad = None
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1).step()
    assert torch.allclose(p.detach(), before)


def test_zero_lr_leaves_parameters_unchanged():
    p = _param((16, 16))
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.0).step()
    assert torch.allclose(p.detach(), before)


def test_momentum_buffer_is_created():
    p = _param((16, 16))
    opt = ParameterizedMuon([p], lr=0.1)
    opt.step()
    assert "momentum_buffer" in opt.state[p]


def test_resolves_coefficient_set_by_name():
    opt = ParameterizedMuon([_param((8, 8))], coeff_set="keller")
    assert opt.coefficient_set.coeffs == (3.4445, -4.7750, 2.0315)


def test_rejects_unknown_coefficient_set():
    with pytest.raises(KeyError, match="unknown coefficient set"):
        ParameterizedMuon([_param((8, 8))], coeff_set="nope")


def test_exact_svd_arm_runs_without_special_casing():
    p = _param((16, 16))
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1, coeff_set="exact_svd").step()
    assert not torch.allclose(p.detach(), before)


def test_rejects_non_matrix_parameters():
    p = torch.randn(16, requires_grad=True)
    p.grad = torch.randn(16)
    with pytest.raises(ValueError, match="2D"):
        ParameterizedMuon([p], lr=0.1).step()


def test_hook_is_called_with_before_and_after():
    calls = []
    p = _param((16, 16))
    opt = ParameterizedMuon(
        [p], lr=0.1, hook=lambda s, i, b, a: calls.append((s, i, b.shape, a.shape))
    )
    opt.step()
    opt.step()
    assert [c[0] for c in calls] == [0, 1]
    assert calls[0][2] == (16, 16) and calls[0][3] == (16, 16)


def test_step_counter_advances():
    opt = ParameterizedMuon([_param((8, 8))], lr=0.1)
    opt.step()
    opt.step()
    assert opt.step_count == 2


def test_weight_decay_shrinks_a_zero_gradient_parameter():
    p = _param((16, 16))
    p.grad = torch.zeros(16, 16)
    before = p.detach().clone()
    ParameterizedMuon([p], lr=0.1, weight_decay=0.1).step()
    assert p.detach().abs().sum() < before.abs().sum()


def test_all_coefficient_sets_are_usable():
    """Every table entry must drive a step without special-casing."""
    from nsshape.coefficients import load_table

    for name in load_table():
        p = _param((16, 16))
        before = p.detach().clone()
        ParameterizedMuon([p], lr=0.1, coeff_set=name).step()
        assert not torch.allclose(p.detach(), before), name


def test_training_reduces_a_simple_quadratic_loss():
    """End-to-end sanity: the optimizer must actually optimize."""
    torch.manual_seed(0)
    W = torch.randn(16, 16, requires_grad=True)
    target = torch.randn(16, 16)
    opt = ParameterizedMuon([W], lr=0.05)
    first = last = None
    for i in range(60):
        opt.zero_grad()
        loss = ((W - target) ** 2).mean()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    assert last < first
