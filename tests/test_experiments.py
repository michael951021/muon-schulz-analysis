"""Checks for matched budgets, data boundaries, and experimental selection."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from nsshape.experiments.models import GPT, CifarCNN, parameter_groups
from nsshape.experiments.data import token_batch

path = Path(__file__).resolve().parents[1]/'scripts/run_study.py'
spec = importlib.util.spec_from_file_location('study_scheduler', path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_matched_budget_includes_slow_arm_and_fixed_cost():
    jobs = [dict(arm=arm, precision='fp32') for arm in ('a', 'b') for _ in range(3)]
    timings = {('a', 'fp32'):dict(step=1, fixed=10), ('b', 'fp32'):dict(step=4, fixed=10)}
    steps, estimated = study.matched_steps(jobs, timings, 1000, 5, 1000, 2, safety=1, reserve=0)
    assert estimated <= 1000
    costs = [timings[(job['arm'], 'fp32')]['step']*(steps+1)+10 for job in jobs]
    assert study.schedule_duration(costs, 2) > 1000


def test_budget_refuses_underpowered_run():
    with pytest.raises(RuntimeError, match='does not fit'):
        study.matched_steps([dict(arm='a', precision='fp32')],
                            {('a', 'fp32'):dict(step=10, fixed=5)}, 20, 100, 1000, 2)


def test_budget_accounts_for_selection_barriers():
    jobs = [dict(arm='a', precision='fp32', phase=phase) for phase in ('tune', 'confirm')]
    timings = {('a', 'fp32'):dict(step=1, fixed=0)}
    steps, estimate = study.matched_steps(jobs, timings, 100, 1, 1000, 2, safety=1, reserve=0)
    assert steps == 50 and estimate == 100


def test_selection_uses_validation_not_test():
    rows = [dict(status='complete', spec=dict(arm='a', precision='fp32', lr=lr),
                 validation=dict(loss=loss), test=dict(loss=100-loss))
            for lr, loss in ((.01, 2), (.01, 4), (.1, 4), (.1, 5))]
    assert study.select(rows)[('a', 'fp32')] == (3, .01)
    rows[0]['status'] = 'budget_exhausted'
    with pytest.raises(ValueError):
        study.select(rows)


def test_transformer_is_causal_and_embeddings_are_excluded_from_muon():
    torch.manual_seed(0)
    model = GPT(vocab=65, width=32, layers=2, heads=4, context=16).eval()
    x = torch.randint(65, (2, 16))
    y = x.clone()
    y[:, 8:] = (y[:, 8:]+1)%65
    assert torch.allclose(model(x)[:, :8], model(y)[:, :8], atol=1e-6)
    muon, adam = parameter_groups(model)
    assert all(p.ndim == 2 for p in muon)
    assert id(model.token.weight) in {id(p) for p in adam}
    assert len({id(p) for p in muon+adam}) == len(list(model.parameters()))


def test_cifar_convolution_parameters_have_explicit_matrix_shape():
    model = CifarCNN(width=8)
    assert model(torch.randn(4, 3, 32, 32)).shape == (4, 10)
    muon, adam = parameter_groups(model)
    assert all(p.ndim == 2 for p in muon)
    assert len(muon) == 4


def test_token_batch_has_correct_next_token_targets_and_reproducible_order():
    data = np.arange(100, dtype=np.uint16)
    x, y = token_batch(data, 4, 8, torch.Generator().manual_seed(7), 'cpu')
    assert torch.equal(x+1, y)
    repeated, _ = token_batch(data, 4, 8, torch.Generator().manual_seed(7), 'cpu')
    assert torch.equal(x, repeated)
    assert x.max() < 99


def test_gpt_parameter_count_for_planned_model():
    # Meta tensors verify architecture size without allocating 500MB of weights.
    with torch.device('meta'):
        model = GPT()
    assert sum(p.numel() for p in model.parameters()) == 123767040


def test_real_training_worker_keeps_test_out_of_tuning(tmp_path):
    import json
    from nsshape.experiments.data import sha256
    from nsshape.experiments.train import run
    root = tmp_path/'tokens'
    root.mkdir()
    for split, offset in (('train', 0), ('validation', 1), ('test', 2)):
        (np.arange(1024, dtype=np.uint16)%16+offset).tofile(root/f'{split}.bin')
    manifest = dict(dataset='synthetic-test-fixture', tokenizer='characters', vocab_size=18,
                    files={split:dict(sha256=sha256(root/f'{split}.bin'))
                           for split in ('train', 'validation', 'test')})
    (root/'manifest.json').write_text(json.dumps(manifest))
    spec = dict(task='language', arm='keller', precision='fp32', phase='tune', lr=.01,
                seed=0, steps=4, data=str(root), batch=2,
                model=dict(width=16, layers=1, heads=2, context=16),
                eval_tokens=128, eval_every=4, log_every=2, threads=1)
    result = run(spec, tmp_path/'tune', 'cpu')
    assert result['status'] == 'complete' and result['test'] is None
    confirmed = run(dict(spec, phase='confirm'), tmp_path/'confirm', 'cpu')
    assert confirmed['test']['count'] == 128
    assert (tmp_path/'confirm/checkpoint.pt').exists()


def test_cifar_worker_executes_training_and_validation(tmp_path, monkeypatch):
    from nsshape.experiments import train as worker
    generator = torch.Generator().manual_seed(0)
    fixture = (torch.randn(12, 3, 32, 32, generator=generator), torch.arange(12)%10)
    monkeypatch.setattr(worker, 'cifar', lambda root: (fixture, fixture, fixture, {'dataset':'synthetic-test-fixture'}))
    result = worker.run(dict(task='cifar', arm='gamma_0.10', precision='fp32', phase='tune',
                             lr=.01, seed=0, steps=3, data='unused', width=8, batch=4,
                             eval_every=3, log_every=2, threads=1), tmp_path/'cnn', 'cpu')
    assert result['completed_steps'] == 3
    assert result['validation']['count'] == 12
    assert result['test'] is None


def test_full_gpu_protocol_selects_then_confirms_without_gpu(tmp_path, monkeypatch):
    import json
    config = json.loads((path.parents[1]/'configs/dual_3090.json').read_text())
    config['hours'] = .01
    config['time_safety_factor'] = 1
    config['checkpoint_reserve_seconds'] = 0
    config['cifar'].update(minimum_steps=2, maximum_steps=16)
    config['language'].update(minimum_tune_steps=2, minimum_confirm_steps=4, maximum_steps=20)
    calls = []
    class FakeScheduler:
        def __init__(self, out, devices, deadline, source_hash, cpu):
            self.deadline = deadline
        def run(self, specs, timings=None):
            calls.append(specs)
            rows = []
            for job in specs:
                loss = 2 if job['arm']=='gamma_0.10' else 3
                loss += abs(job['lr']-.02)
                rows.append(dict(status='complete', spec=job, completed_steps=job['steps'],
                                 validation=dict(loss=loss), seconds=.1, seconds_per_step=.01,
                                 worker_wall_seconds=.01*job['steps']+.02, final_evaluation_seconds=.01))
            return rows
    monkeypatch.setattr(study, 'Scheduler', FakeScheduler)
    study.study(config, tmp_path/'protocol', cpu=True)
    result = json.loads((tmp_path/'protocol/study.json').read_text())
    assert result['status'] == 'complete'
    assert result['selections']['fitted_winner'] == 'gamma_0.10'
    assert len(result['cifar']['tuning']) == 126
    assert len(result['cifar']['confirmation']) == 15
    assert len(result['language']['confirmation']) == 15
    for task in ('cifar', 'language'):
        rows = result[task]['confirmation']
        assert len({row['spec']['steps'] for row in rows}) == 1
        assert {row['spec']['seed'] for row in rows} == {10, 11, 12}


def test_report_rejects_mismatched_work_and_partial_results():
    report_spec = importlib.util.spec_from_file_location('report_study', path.parent/'report_study.py')
    reporter = importlib.util.module_from_spec(report_spec)
    report_spec.loader.exec_module(reporter)
    rows = [dict(status='complete', completed_steps=steps,
                 spec=dict(arm='keller', precision='fp32', lr=.02, seed=seed, steps=steps, batch=8),
                 test=dict(loss=2, accuracy=.5), seconds=1, diagnostic_seconds=0)
            for seed, steps in ((10, 20), (11, 21))]
    with pytest.raises(ValueError, match='mismatched'):
        reporter.summarize(rows)
    rows[1]['status'] = 'budget_exhausted'
    with pytest.raises(ValueError, match='complete'):
        reporter.summarize(rows)


def test_selection_skips_learning_rates_that_diverged_on_any_seed():
    rows = [dict(status='complete', spec=dict(arm='a', precision='bf16', lr=lr, seed=seed),
                 validation=dict(loss=loss))
            for lr, seed, loss in ((.005, 1, 1), (.02, 0, 3), (.02, 1, 3))]
    diverged = [dict(status='nonfinite_grad', spec=dict(arm='a', precision='bf16', lr=.005, seed=0))]
    assert study.select(rows)[('a', 'bf16')] == (1, .005)
    assert study.select(rows, diverged)[('a', 'bf16')] == (3, .02)
    everything = [dict(status='nonfinite_loss', spec=dict(arm='a', precision='bf16', lr=.02, seed=0))]
    assert ('a', 'bf16') not in study.select(rows, diverged+everything)


def test_scheduler_records_tuning_divergence_from_real_worker_exit(tmp_path, monkeypatch):
    """The worker writes result.json and exits 2 on divergence; tuning must record it,
    while a divergence outside tuning still stops the study."""
    import json, subprocess, sys, time
    fake = tmp_path/'fake_worker.py'
    fake.write_text(
        'import json, sys\n'
        'spec = json.loads(open(sys.argv[sys.argv.index("--spec")+1]).read())\n'
        'out = sys.argv[sys.argv.index("--out")+1]\n'
        'diverge = spec["lr"] > .05\n'
        'result = dict(spec=spec, status="nonfinite_grad" if diverge else "complete",\n'
        '              completed_steps=3 if diverge else spec["steps"], validation=dict(loss=None if diverge else 1.))\n'
        'open(out+"/result.json", "w").write(json.dumps(result))\n'
        'sys.exit(2 if diverge else 0)\n')
    real = subprocess.Popen
    monkeypatch.setattr(study.subprocess, 'Popen',
                        lambda command, **kw: real([sys.executable, str(fake)]+command[3:], **kw))
    def spec(phase, lr):
        return dict(task='cifar', arm='a', precision='bf16', phase=phase, lr=lr, seed=0, steps=10)
    scheduler = study.Scheduler(tmp_path/'out', ['cpu0'], time.monotonic()+60, 'hash', cpu=True)
    rows = scheduler.run([spec('tune', .02), spec('tune', .08)])
    assert [row['spec']['lr'] for row in rows] == [.02]
    assert [row['status'] for row in scheduler.diverged] == ['nonfinite_grad']
    assert study.select(rows, scheduler.diverged)[('a', 'bf16')] == (1., .02)
    with pytest.raises(RuntimeError):
        study.Scheduler(tmp_path/'confirm', ['cpu0'], time.monotonic()+60, 'hash', cpu=True).run([spec('confirm', .08)])
