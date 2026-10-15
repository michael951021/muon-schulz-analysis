"""Calibrate and schedule independent experiments across two CUDA devices.

No DDP: one isolated experiment per GPU. Matched work, fresh confirmation seeds,
hard global deadline, resumable completed jobs, and strict provenance checks.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

from nsshape.coefficients import TABLE_PATH, load_table
from nsshape.experiments.data import sha256
from nsshape.experiments.train import atomic_json


ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    paths = sorted((ROOT/'src').rglob('*.py')) + sorted((ROOT/'scripts').glob('*.py')) + [TABLE_PATH]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def schedule_duration(costs, workers=2):
    """Conservative longest-processing-time greedy makespan estimate."""
    loads = [0.]*workers
    for cost in sorted(costs, reverse=True):
        index = min(range(workers), key=loads.__getitem__)
        loads[index] += cost
    return max(loads)


def matched_steps(jobs, timings, seconds, minimum, maximum, workers, safety=1.5, reserve=20):
    def duration(steps):
        phases = defaultdict(list)
        for job in jobs:
            cost = safety*(timings[(job['arm'], job['precision'])]['step']*steps+
                           timings[(job['arm'], job['precision'])]['fixed'])+reserve
            phases[job.get('phase', 'all')].append(cost)
        # Selection barriers prevent tuning and confirmation from overlapping.
        return sum(schedule_duration(costs, workers) for costs in phases.values())
    if duration(minimum) > seconds:
        raise RuntimeError(f'minimum scientifically usable budget does not fit: '
                           f'{minimum} steps require {duration(minimum)/3600:.2f} h; '
                           f'available {seconds/3600:.2f} h. Increase hours or explicitly revise protocol.')
    left, right = minimum, maximum
    while left < right:
        mid = (left+right+1)//2
        if duration(mid) <= seconds:
            left = mid
        else:
            right = mid-1
    return left, duration(left)


DIVERGED = ('nonfinite_loss', 'nonfinite_grad')


def select(rows, diverged=()):
    """Best mean validation LR per arm. A learning rate that diverged on any tuning
    seed is ineligible; an arm whose every learning rate diverged is absent."""
    excluded = {(row['spec']['arm'], row['spec']['precision'], row['spec']['lr']) for row in diverged}
    groups = defaultdict(list)
    for row in rows:
        if row['status'] != 'complete':
            raise ValueError('incomplete run cannot select hyperparameters')
        spec = row['spec']
        if (spec['arm'], spec['precision'], spec['lr']) in excluded:
            continue
        groups[(spec['arm'], spec['precision'], spec['lr'])].append(row['validation']['loss'])
    by_arm = defaultdict(list)
    for (arm, precision, lr), values in groups.items():
        by_arm[(arm, precision)].append((sum(values)/len(values), lr))
    return {key: min(values) for key, values in by_arm.items()}


class Scheduler:
    def __init__(self, out, devices, deadline, source_hash, cpu=False):
        self.out, self.devices, self.deadline = out, devices, deadline
        self.source_hash, self.cpu = source_hash, cpu
        # Tuning runs that diverged: a result about that (arm, precision, lr), kept
        # out of selection and reported, not a failed study.
        self.diverged = []

    def job_path(self, spec):
        encoded = json.dumps(spec, sort_keys=True).encode()
        return self.out/'jobs'/hashlib.sha256(encoded).hexdigest()[:20]

    def run(self, specs, timings=None):
        pending = list(specs)
        if timings is not None:
            pending.sort(key=lambda job: timings[(job['arm'], job['precision'])]['step']*job['steps']+
                         timings[(job['arm'], job['precision'])]['fixed'], reverse=True)
        active, completed = {}, []
        try:
            while pending or active:
                if time.monotonic() >= self.deadline:
                    raise RuntimeError('global compute deadline reached; partial jobs are not scientific results')
                for device in self.devices:
                    if device in active or not pending:
                        continue
                    spec = dict(pending.pop(0), source_sha256=self.source_hash)
                    directory = self.job_path(spec)
                    directory.mkdir(parents=True, exist_ok=True)
                    result_path = directory/'result.json'
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        if result['status'] == 'complete' and result['spec'] == spec:
                            completed.append(result)
                            print(f"reuse {spec['task']} {spec['phase']} {spec['arm']} {spec['precision']} seed={spec['seed']}", flush=True)
                            continue
                        if result['status'] in DIVERGED and spec['phase'] == 'tune' and result['spec'] == spec:
                            self.diverged.append(result)
                            print(f"reuse diverged {spec['task']} {spec['arm']} {spec['precision']} lr={spec['lr']} seed={spec['seed']}", flush=True)
                            continue
                    atomic_json(directory/'spec.json', spec)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='2')
                    log = (directory/'worker.log').open('w')
                    command = [sys.executable, '-m', 'nsshape.experiments.train',
                               '--spec', str(directory/'spec.json'), '--out', str(directory),
                               '--device', 'cpu' if self.cpu else 'cuda:0']
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                    active[device] = (process, log, directory, spec, time.monotonic())
                    print(f"start GPU {device}: {spec['task']} {spec['phase']} {spec['arm']} {spec['precision']} lr={spec['lr']} seed={spec['seed']} steps={spec['steps']}", flush=True)
                for device, item in list(active.items()):
                    process, log, directory, spec, start = item
                    if process.poll() is None:
                        continue
                    log.close()
                    del active[device]
                    # The worker exits 2 after writing result.json for any non-complete status.
                    written = process.returncode == 2 and (directory/'result.json').exists()
                    if process.returncode != 0 and not written:
                        raise RuntimeError(f'worker failed ({process.returncode}); inspect {directory}/worker.log')
                    result = json.loads((directory/'result.json').read_text())
                    if result['status'] in DIVERGED and spec['phase'] == 'tune':
                        result['worker_wall_seconds'] = time.monotonic()-start
                        atomic_json(directory/'result.json', result)
                        self.diverged.append(result)
                        print(f"diverged GPU {device}: {result['status']} at step {result['completed_steps']}/{spec['steps']} "
                              f"({spec['arm']} {spec['precision']} lr={spec['lr']} seed={spec['seed']})", flush=True)
                        continue
                    if result['status'] != 'complete' or result['completed_steps'] != spec['steps']:
                        raise RuntimeError(f'incomplete job: {directory}')
                    result['worker_wall_seconds'] = time.monotonic()-start
                    atomic_json(directory/'result.json', result)
                    completed.append(result)
                    print(f"done GPU {device}: val loss={result['validation']['loss']:.4f} wall={result['worker_wall_seconds']:.1f}s", flush=True)
                if pending or active:
                    time.sleep(.2)
        finally:
            for process, log, _, _, _ in active.values():
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
        return completed


def study(config, out, cpu=False):
    started = time.monotonic()
    seconds = config['hours']*3600
    out.mkdir(parents=True, exist_ok=True)
    source_hash = fingerprint()
    manifest = dict(config=config, source_sha256=source_hash,
                    table_sha256=sha256(TABLE_PATH),
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     numpy=np.__version__, cuda=torch.version.cuda))
    if (out/'manifest.json').exists():
        if json.loads((out/'manifest.json').read_text()) != manifest:
            raise RuntimeError('output provenance changed; choose a new output directory')
    atomic_json(out/'manifest.json', manifest)
    consumed = (json.loads((out/'budget_progress.json').read_text())['consumed_seconds']
                if (out/'budget_progress.json').exists() else 0.)
    if consumed >= seconds:
        raise RuntimeError('cumulative compute budget is exhausted')
    scheduler = Scheduler(out, config['devices'], started+seconds-consumed, source_hash, cpu)
    table = load_table()
    arms = [(name, precision) for precision in ('fp32', 'bf16') for name in table
            if precision == 'fp32' or name != 'exact_svd']
    calibrations, budgets = {}, {}
    planned = json.loads((out/'stage_plan.json').read_text()) if (out/'stage_plan.json').exists() else {}

    def base(task, arm, precision, phase, lr=.02, seed=0, steps=8):
        fields = {key: value for key, value in config[task].items()
                  if not key.startswith('minimum') and key != 'maximum_steps'}
        return dict(fields, task=task, arm=arm, precision=precision, phase=phase,
                    lr=lr, seed=seed, steps=steps, data=str((ROOT/config['data'][task]).resolve()),
                    weight_decay=.01, max_seconds=max(1, seconds))

    def calibrate(task, choices):
        rows = []
        for arm, precision in choices:
            rows += scheduler.run([base(task, arm, precision, 'calibration', seed=seed,
                                        steps=config['calibration_steps'])
                                   for seed in range(len(config['devices']))])
        timings = {}
        for row in rows:
            key = row['spec']['arm'], row['spec']['precision']
            # Includes startup/dataset hashing and final evaluation. For normal
            # jobs account for periodic evaluation and diagnostics conservatively.
            wall = row.get('worker_wall_seconds', row['seconds']+row['final_evaluation_seconds']+10)
            steps = row['spec']['steps']
            step_cost = row['seconds_per_step']
            evaluation = row['final_evaluation_seconds']
            extra_eval = evaluation/config[task].get('eval_every', 200)
            previous = timings.get(key, dict(step=0, fixed=0))
            timings[key] = dict(step=max(previous['step'], step_cost+extra_eval),
                                fixed=max(previous['fixed'], max(0, wall-steps*step_cost)+evaluation))
        calibrations[task] = rows
        return timings

    timings = calibrate('cifar', arms)
    jobs = [base('cifar', arm, precision, 'tune', lr, seed)
            for arm, precision in arms for lr in config['learning_rates'] for seed in config['tune_seeds']]
    cifar_seconds = seconds*config['fractions']['cifar']
    # Reserve confirmation before running the grid. Exact SVD can dominate this
    # stage, so a fixed 80/20 split would fail when its share of jobs increases.
    planning_timings = dict(timings)
    planning_followup = [('worst_fit', 'fp32'), ('keller', 'fp32'), ('exact_svd', 'fp32'),
                         ('worst_fit', 'bf16'), ('keller', 'bf16')]
    for precision in ('fp32', 'bf16'):
        fitted_timings = [value for (arm, mode), value in timings.items()
                          if mode == precision and table[arm].kind == 'fitted']
        planning_timings[('worst_fit', precision)] = dict(
            step=max(value['step'] for value in fitted_timings),
            fixed=max(value['fixed'] for value in fitted_timings))
    worst_confirmation = [base('cifar', arm, precision, 'confirm', seed=seed)
                          for arm, precision in planning_followup for seed in config['confirm_seeds']]
    steps, estimated = matched_steps(jobs+worst_confirmation, planning_timings, cifar_seconds,
                                    config['cifar']['minimum_steps'], config['cifar']['maximum_steps'],
                                    len(config['devices']), config['time_safety_factor'],
                                    config['checkpoint_reserve_seconds'])
    if 'cifar' in planned:
        steps, estimated = planned['cifar']['steps'], planned['cifar']['estimated_seconds']
    else:
        planned['cifar'] = dict(steps=steps, estimated_seconds=estimated)
        atomic_json(out/'stage_plan.json', planned)
    for job in jobs:
        job['steps'] = steps
    tuning = scheduler.run(jobs, timings)
    cifar_diverged = list(getattr(scheduler, 'diverged', []))
    selected = select(tuning, cifar_diverged)
    fitted = [(value[0], arm) for (arm, precision), value in selected.items()
              if precision == 'fp32' and table[arm].kind == 'fitted' and (arm, 'bf16') in selected]
    _, winner = min(fitted)
    followup = [(winner, 'fp32'), ('keller', 'fp32'), ('exact_svd', 'fp32'),
                (winner, 'bf16'), ('keller', 'bf16')]
    missing = [key for key in followup if key not in selected]
    if missing:
        raise RuntimeError(f'every tuning learning rate diverged for {missing}; revise protocol')
    confirmation = [base('cifar', arm, precision, 'confirm', selected[(arm, precision)][1], seed, steps)
                    for arm, precision in followup for seed in config['confirm_seeds']]
    estimated_confirm = schedule_duration([
        config['time_safety_factor']*(timings[(job['arm'], job['precision'])]['step']*steps+
                                     timings[(job['arm'], job['precision'])]['fixed'])+
        config['checkpoint_reserve_seconds'] for job in confirmation], len(config['devices']))
    estimated_tune = schedule_duration([
        config['time_safety_factor']*(timings[(job['arm'], job['precision'])]['step']*steps+
                                     timings[(job['arm'], job['precision'])]['fixed'])+
        config['checkpoint_reserve_seconds'] for job in jobs], len(config['devices']))
    if estimated_tune+estimated_confirm > cifar_seconds:
        raise RuntimeError('CIFAR confirmation exceeds reserved budget; revise protocol before running further')
    cifar_confirm = scheduler.run(confirmation, timings)
    budgets['cifar'] = dict(steps=steps, estimated_seconds=estimated_tune+estimated_confirm)
    selections = dict(fitted_winner=winner,
                      cifar_lrs={f'{arm}/{precision}': lr for (arm, precision), (_, lr) in selected.items()})
    atomic_json(out/'cifar_stage.json', dict(tuning=tuning, diverged=cifar_diverged,
                                             confirmation=cifar_confirm, selections=selections))

    timings = calibrate('language', followup)
    jobs = [base('language', arm, precision, 'tune', lr, seed)
            for arm, precision in followup for lr in config['learning_rates']
            for seed in config['language_tune_seeds']]
    tune_seconds = seconds*config['fractions']['language_tune']
    tune_steps, estimated = matched_steps(jobs, timings, tune_seconds,
                                         config['language']['minimum_tune_steps'], config['language']['maximum_steps'],
                                         len(config['devices']), config['time_safety_factor'],
                                         config['checkpoint_reserve_seconds'])
    if 'language_tune' in planned:
        tune_steps, estimated = planned['language_tune']['steps'], planned['language_tune']['estimated_seconds']
    else:
        planned['language_tune'] = dict(steps=tune_steps, estimated_seconds=estimated)
        atomic_json(out/'stage_plan.json', planned)
    for job in jobs:
        job['steps'] = tune_steps
    language_tune = scheduler.run(jobs, timings)
    language_diverged = list(getattr(scheduler, 'diverged', []))[len(cifar_diverged):]
    chosen = select(language_tune, language_diverged)
    missing = [key for key in followup if key not in chosen]
    if missing:
        raise RuntimeError(f'every language tuning learning rate diverged for {missing}; revise protocol')
    jobs = [base('language', arm, precision, 'confirm', chosen[(arm, precision)][1], seed)
            for arm, precision in followup for seed in config['language_confirm_seeds']]
    # Spend unused capped CIFAR/tuning allocation on confirmation, preserving
    # the global reserve. A budget ceiling need not waste available evidence.
    confirm_seconds = scheduler.deadline-time.monotonic()-seconds*config['fractions']['reserve']
    if 'language_confirm' in planned:
        confirm_steps, estimated_confirm = planned['language_confirm']['steps'], planned['language_confirm']['estimated_seconds']
    else:
        confirm_steps, estimated_confirm = matched_steps(jobs, timings, confirm_seconds,
                                                        config['language']['minimum_confirm_steps'],
                                                        config['language']['maximum_steps'], len(config['devices']),
                                                        config['time_safety_factor'], config['checkpoint_reserve_seconds'])
        planned['language_confirm'] = dict(steps=confirm_steps, estimated_seconds=estimated_confirm)
        atomic_json(out/'stage_plan.json', planned)
    for job in jobs:
        job['steps'] = confirm_steps
    language_confirm = scheduler.run(jobs, timings)
    budgets['language'] = dict(tune_steps=tune_steps, confirm_steps=confirm_steps,
                              tokens_per_confirm_run=confirm_steps*config['language']['batch']*config['language']['model']['context'],
                              estimated_seconds=estimated+estimated_confirm)
    selections['language_lrs'] = {f'{arm}/{precision}': lr for (arm, precision), (_, lr) in chosen.items()}
    result = dict(status='complete', manifest=manifest, calibrations=calibrations,
                  budgets=budgets, selections=selections,
                  cifar=dict(tuning=tuning, diverged=cifar_diverged, confirmation=cifar_confirm),
                  language=dict(tuning=language_tune, diverged=language_diverged, confirmation=language_confirm),
                  total_wall_seconds=consumed+time.monotonic()-started)
    atomic_json(out/'study.json', result)
    print(f'Complete: {out}/study.json', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dual_3090.json')
    parser.add_argument('--out', type=Path, default=ROOT/'runs/dual3090')
    parser.add_argument('--hours', type=float)
    parser.add_argument('--cpu', action='store_true', help='smoke protocol only, not full GPU study')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.hours is not None:
        config['hours'] = args.hours
    if not args.cpu and not 6 <= config['hours'] <= 10:
        parser.error('GPU study hours must be between 6 and 10')
    if len(config['devices']) != 2:
        parser.error('this protocol requires two independent GPU workers')
    started = time.monotonic()
    progress = args.out/'budget_progress.json'
    consumed = json.loads(progress.read_text())['consumed_seconds'] if progress.exists() else 0.
    try:
        study(config, args.out.resolve(), args.cpu)
    except Exception as exc:
        atomic_json(args.out/'failure.json', dict(status='incomplete', error=str(exc)))
        raise
    finally:
        atomic_json(progress, dict(consumed_seconds=consumed+time.monotonic()-started,
                                   maximum_seconds=config['hours']*3600))


if __name__ == '__main__':
    main()
