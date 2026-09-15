"""Whole-pipeline test: generator -> artifact -> loader -> optimizer -> spectra -> figure.

Every other test file exercises one module in isolation. This one runs the chain
the README tells a researcher to run, against a table generated inside the test,
so that schema drift or bad wiring between two stages fails here even when every
unit test still passes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from torch import nn

from nsshape import coefficients
from nsshape.coefficients import load_table
from nsshape.diagnostics.schema import SpectrumRecord
from nsshape.diagnostics.svd_logger import SVDLogger
from nsshape.optim.muon import ParameterizedMuon
from nsshape.polynomial import compose, positivity_violation, reachable_bound

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: One gamma is enough to prove the artifact contract; sweeping the full grid is
#: the generator's own job and takes minutes.
E2E_GAMMA = 0.30
E2E_SEEDS = (0, 1, 2)

#: Reduced solver budget. The committed table uses 400x40, which buys precision
#: this test does not assert on.
E2E_MAXITER = 80
E2E_POPSIZE = 15

TRAIN_STEPS = 30
LOG_EVERY = 5
N_MATRICES = 3


def _load_script(name: str):
    """Import a file from scripts/, which is not a package."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _build_problem(seed: int = 0):
    """Small regression task whose weights cover tall, wide and near-square."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(24, 48), nn.Tanh(), nn.Linear(48, 16), nn.Tanh(), nn.Linear(16, 3)
    )
    inputs = torch.randn(128, 24)
    teacher = nn.Sequential(nn.Linear(24, 16), nn.Tanh(), nn.Linear(16, 3))
    with torch.no_grad():
        targets = teacher(inputs)
    return model, inputs, targets


def _train(coeff_set: str, log_path: Path | None = None, lr: float = 0.02):
    """Train with the documented split: 2D weights on Muon, everything else AdamW."""
    model, inputs, targets = _build_problem()
    matrices = [p for p in model.parameters() if p.ndim == 2]
    others = [p for p in model.parameters() if p.ndim != 2]

    log = SVDLogger(log_path, every_k=LOG_EVERY) if log_path is not None else None
    muon = ParameterizedMuon(matrices, lr=lr, coeff_set=coeff_set, hook=log)
    adamw = torch.optim.AdamW(others, lr=0.01)

    losses = []
    try:
        for _ in range(TRAIN_STEPS):
            muon.zero_grad()
            adamw.zero_grad()
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            muon.step()
            adamw.step()
            losses.append(loss.item())
    finally:
        if log is not None:
            log.close()

    records = []
    if log_path is not None:
        records = [
            SpectrumRecord.from_json(line)
            for line in log_path.read_text().splitlines()
        ]
    return losses, records


def _render_figure(out_root: Path) -> Path:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    plotter = _load_script("plot_composed_maps")
    out = out_root / "figures"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["plot_composed_maps.py", "--out", str(out)])
        plotter.main()
    return out / "composed_maps.png"


@dataclass
class Pipeline:
    table_path: Path
    table: dict
    losses: dict[str, list[float]]
    records: dict[str, list[SpectrumRecord]]
    figure: Path


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Run every stage once; each test below inspects one stage of the result."""
    tmp = tmp_path_factory.mktemp("e2e")
    table_path = tmp / "coefficients" / "v1.json"

    generator = _load_script("generate_coefficients")
    full_fit = generator.fit_gamma

    losses, records = {}, {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(generator, "GAMMA_GRID", [E2E_GAMMA])
        mp.setattr(generator, "SEEDS", E2E_SEEDS)
        mp.setattr(generator, "TABLE_PATH", table_path)
        mp.setattr(
            generator,
            "fit_gamma",
            lambda gamma, seeds: full_fit(
                gamma, seeds=seeds, maxiter=E2E_MAXITER, popsize=E2E_POPSIZE
            ),
        )
        generator.main()

        # ParameterizedMuon resolves names through the module-level table path,
        # so the training stage has to be pointed at the artifact just written.
        mp.setattr(coefficients, "TABLE_PATH", table_path)
        table = load_table(table_path)
        for arm in ("keller", "exact_svd"):
            losses[arm], records[arm] = _train(arm, tmp / f"{arm}.jsonl")

    # Outside the patch: the figure stage reads the committed table by design.
    return Pipeline(table_path, table, losses, records, _render_figure(tmp))


def test_generator_writes_an_artifact_the_loader_accepts(pipeline):
    """The schema contract between the generator and the loader."""
    payload = json.loads(pipeline.table_path.read_text())
    assert payload["version"] == 1
    assert {e["name"] for e in payload["entries"]} == set(pipeline.table)


def test_generated_table_carries_all_three_entry_kinds(pipeline):
    assert {cs.kind for cs in pipeline.table.values()} == {
        "fitted",
        "baseline",
        "exact_svd",
    }


def test_generated_certificate_survives_independent_recomputation(pipeline):
    """Recompute from raw coefficients, so a solver bug cannot certify itself."""
    cs = pipeline.table[f"gamma_{E2E_GAMMA:.2f}"]
    assert cs.sigma_min < 1.0

    B = reachable_bound(cs.coeffs)
    assert np.isfinite(B)
    assert B == pytest.approx(cs.invariance_bound, rel=1e-6)
    assert positivity_violation(cs.coeffs, B) <= 1e-9

    xs = np.logspace(np.log10(cs.sigma_min), 0.0, 2000)
    image = compose(cs.coeffs, xs, cs.n_steps)
    relhw = (image.max() - image.min()) / (image.max() + image.min())
    assert relhw <= cs.gamma + 1e-3


@pytest.mark.parametrize("arm", ["keller", "exact_svd"])
def test_training_reduces_the_loss(pipeline, arm):
    losses = pipeline.losses[arm]
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0] / 10


def test_both_backends_reach_comparable_loss(pipeline):
    """The exact-SVD arm needs no special-casing anywhere in the training loop."""
    assert pipeline.losses["exact_svd"][-1] == pytest.approx(
        pipeline.losses["keller"][-1], rel=0.5
    )


@pytest.mark.parametrize("arm", ["keller", "exact_svd"])
def test_logger_sampled_every_kth_step(pipeline, arm):
    steps = sorted({r.step for r in pipeline.records[arm]})
    assert steps == list(range(0, TRAIN_STEPS, LOG_EVERY))


@pytest.mark.parametrize("arm", ["keller", "exact_svd"])
def test_every_weight_matrix_was_observed_before_and_after(pipeline, arm):
    records = pipeline.records[arm]
    seen = {(r.step, r.param_index, r.phase) for r in records}
    for step in {r.step for r in records}:
        for index in range(N_MATRICES):
            assert (step, index, "before") in seen
            assert (step, index, "after") in seen


@pytest.mark.parametrize("arm", ["keller", "exact_svd"])
def test_orthogonalization_compresses_every_sampled_spectrum(pipeline, arm):
    """The scientific payload: on real momentum matrices the step narrows the band."""
    by_key = {(r.step, r.param_index, r.phase): r for r in pipeline.records[arm]}
    for step, index, phase in by_key:
        if phase == "before":
            after = by_key[(step, index, "after")]
            assert after.band_ratio < by_key[(step, index, phase)].band_ratio


def test_exact_svd_arm_flattens_the_spectrum(pipeline):
    after = [r for r in pipeline.records["exact_svd"] if r.phase == "after"]
    assert after
    for record in after:
        assert record.band_ratio == pytest.approx(1.0, abs=1e-3)


def test_logged_records_round_trip_through_the_schema(pipeline):
    record = pipeline.records["keller"][0]
    assert SpectrumRecord.from_json(record.to_json()) == record


def test_figure_stage_renders_from_the_committed_table(pipeline):
    assert pipeline.figure.exists()
    assert pipeline.figure.stat().st_size > 10_000


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known defect. The certificate is computed in fp64 but ns_quintic iterates "
        "in bf16, and the solver leaves no room for the difference. Maximizing "
        "sigma_min pushes each fitted polynomial until its peak touches the ceiling "
        "of its own invariant set, so B - 1 lands between 1e-4 and 7e-4 -- while one "
        "bf16 ulp at 1.0 is 7.8e-3, an order of magnitude wider. A singular value "
        "rounded above B escapes, and P(x) > x above B turns that into divergence. "
        "Only gamma_0.01, gamma_0.02 and keller (B = 1.2024) have real headroom; "
        "gamma_0.05 through gamma_0.60 blow up. Fixing this needs a margin "
        "constraint in solve.py (B >= 1 + delta, delta well above one bf16 ulp) or "
        "an fp32 iteration in ns_quintic."
    ),
)
def test_every_table_entry_survives_a_training_run():
    """The claim the CIFAR sweep rests on: every arm in the table can train."""
    for name in load_table():
        losses, _ = _train(name)
        assert np.isfinite(losses).all(), name
        assert losses[-1] < losses[0], name
