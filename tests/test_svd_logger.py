import json

import pytest

torch = pytest.importorskip("torch")

from nsshape.diagnostics.schema import SpectrumRecord, spectrum_record
from nsshape.diagnostics.svd_logger import SVDLogger
from nsshape.optim.muon import ParameterizedMuon


def test_spectrum_record_round_trips():
    r = spectrum_record(3, 1, "before", torch.randn(8, 8))
    assert SpectrumRecord.from_json(r.to_json()) == r


def test_spectrum_record_json_is_one_line():
    assert "\n" not in spectrum_record(0, 0, "after", torch.randn(8, 8)).to_json()


def test_quantiles_are_ordered():
    q = spectrum_record(0, 0, "before", torch.randn(16, 16)).quantiles
    values = [q["p0"], q["p25"], q["p50"], q["p75"], q["p100"]]
    assert values == sorted(values)


def test_band_ratio_is_one_for_an_orthogonal_matrix():
    Q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert spectrum_record(0, 0, "after", Q).band_ratio == pytest.approx(1.0, abs=1e-4)


def test_band_ratio_exceeds_one_for_random_matrix():
    torch.manual_seed(0)
    assert spectrum_record(0, 0, "before", torch.randn(32, 32)).band_ratio > 2.0


def test_histogram_counts_all_singular_values():
    r = spectrum_record(0, 0, "before", torch.randn(16, 24))
    assert sum(r.histogram) == r.n_singular_values == 16


def test_record_uses_fp32_even_for_bf16_input():
    """The whole point of the logger is an explicit fp32 SVD."""
    r = spectrum_record(0, 0, "after", torch.randn(16, 16).bfloat16())
    assert all(isinstance(v, float) for v in r.quantiles.values())


def test_should_sample_respects_every_k(tmp_path):
    log = SVDLogger(tmp_path / "s.jsonl", every_k=10)
    assert log.should_sample(0) and log.should_sample(10)
    assert not log.should_sample(5)
    log.close()


def test_logger_writes_before_and_after_records(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=1) as log:
        log(0, 0, torch.randn(8, 8), torch.randn(8, 8))
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["phase"] for r in records] == ["before", "after"]


def test_logger_skips_unsampled_steps(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=10) as log:
        for step in range(5):
            log(step, 0, torch.randn(8, 8), torch.randn(8, 8))
    assert len(path.read_text().splitlines()) == 2


def test_logger_limits_sampled_parameters(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=1, max_params=2) as log:
        for index in range(10):
            log(0, index, torch.randn(8, 8), torch.randn(8, 8))
    assert len(path.read_text().splitlines()) == 4  # 2 params x 2 phases


def test_logger_resets_the_budget_each_sampled_step(tmp_path):
    path = tmp_path / "s.jsonl"
    with SVDLogger(path, every_k=1, max_params=2) as log:
        for step in range(3):
            for index in range(10):
                log(step, index, torch.randn(8, 8), torch.randn(8, 8))
    assert len(path.read_text().splitlines()) == 12  # 3 steps x 2 params x 2


def test_logger_parameter_choice_is_seed_reproducible(tmp_path):
    def run(name):
        path = tmp_path / name
        with SVDLogger(path, every_k=1, max_params=2, seed=7) as log:
            for index in range(10):
                log(0, index, torch.randn(8, 8), torch.randn(8, 8))
        return [json.loads(l)["param_index"] for l in path.read_text().splitlines()]

    assert run("a.jsonl") == run("b.jsonl")


def test_logger_integrates_with_optimizer_as_a_hook(tmp_path):
    """The logger must plug into ParameterizedMuon with no adapter."""
    path = tmp_path / "s.jsonl"
    torch.manual_seed(0)
    p = torch.randn(16, 16, requires_grad=True)
    p.grad = torch.randn(16, 16)
    with SVDLogger(path, every_k=1) as log:
        ParameterizedMuon([p], lr=0.05, coeff_set="keller", hook=log).step()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    before = next(r for r in records if r["phase"] == "before")
    after = next(r for r in records if r["phase"] == "after")
    assert after["band_ratio"] < before["band_ratio"]


def test_exact_svd_arm_compresses_the_spectrum_further_than_keller(tmp_path):
    """The logger must be able to tell the arms apart -- this is the diagnostic
    that proves to a reviewer what each polynomial actually did."""

    def band_ratio_after(coeff_set, name):
        path = tmp_path / name
        torch.manual_seed(0)
        p = torch.randn(32, 32, requires_grad=True)
        p.grad = torch.randn(32, 32)
        with SVDLogger(path, every_k=1) as log:
            ParameterizedMuon([p], lr=0.05, coeff_set=coeff_set, hook=log).step()
        records = [json.loads(l) for l in path.read_text().splitlines()]
        return next(r for r in records if r["phase"] == "after")["band_ratio"]

    assert band_ratio_after("exact_svd", "svd.jsonl") < band_ratio_after(
        "keller", "keller.jsonl"
    )
