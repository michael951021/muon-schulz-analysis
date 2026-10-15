"""Compact, publishable summary of a finished study directory.

The run directory holds checkpoints and per-job spectra (tens of GB). This
keeps what the write-up needs: the tuning grid with diverged cells, the
confirmation curves, realized update spectra, and cost.

    python scripts/summarize_run.py runs/dual3090-r3 reports/dual3090/summary.json
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def job_dirs(run):
    for directory in sorted((run/'jobs').iterdir()):
        result = directory/'result.json'
        if result.exists():
            yield directory, json.loads(result.read_text())


def spectra(directory):
    """Median quantiles of the orthogonalized update over every sampled step and matrix."""
    path = directory/'spectra.jsonl'
    if not path.exists():
        return None
    after = defaultdict(list)
    below = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record['phase'] != 'after':
            continue
        for key, value in record['quantiles'].items():
            after[key].append(value)
        if record.get('below_sigma_min') is not None:
            below.append(record['below_sigma_min']/min(record['shape']))
    if not after:
        return None
    result = {key: statistics.median(values) for key, values in after.items()}
    result['records'] = len(after['p50'])
    result['below_sigma_min_fraction'] = statistics.mean(below) if below else None
    return result


def main(run, out):
    study = json.loads((run/'study.json').read_text())
    tuning = defaultdict(lambda: dict(losses=[], diverged=[]))
    for task in ('cifar', 'language'):
        for row in study[task]['tuning']:
            spec = row['spec']
            tuning[(task, spec['arm'], spec['precision'], spec['lr'])]['losses'].append(row['validation']['loss'])
        for row in study[task].get('diverged', []):
            spec = row['spec']
            tuning[(task, spec['arm'], spec['precision'], spec['lr'])]['diverged'].append(
                dict(seed=spec['seed'], status=row['status'], step=row['completed_steps'], steps=spec['steps']))
    grid = [dict(task=task, arm=arm, precision=precision, lr=lr,
                 mean_validation_loss=statistics.mean(cell['losses']) if cell['losses'] else None,
                 seeds=len(cell['losses']), diverged=cell['diverged'])
            for (task, arm, precision, lr), cell in sorted(tuning.items())]

    confirm = []
    for directory, result in job_dirs(run):
        spec = result['spec']
        if spec['phase'] != 'confirm' or result['status'] != 'complete':
            continue
        confirm.append(dict(task=spec['task'], arm=spec['arm'], precision=spec['precision'], seed=spec['seed'],
                            lr=spec['lr'], steps=result['completed_steps'], test=result['test'],
                            curve=[dict(step=p['step'], validation_loss=p['validation']['loss']) for p in result['curve']],
                            seconds=result['seconds'], peak_gpu_gb=result['peak_gpu_bytes']/2**30,
                            tokens=result['tokens_seen'], spectra=spectra(directory)))
    summary = dict(source=str(run), selections=study['selections'], budgets=study['budgets'],
                   total_wall_hours=study['total_wall_seconds']/3600, tuning=grid,
                   confirmation=sorted(confirm, key=lambda r: (r['task'], r['arm'], r['precision'], r['seed'])))
    out.write_text(json.dumps(summary, indent=1, allow_nan=False)+'\n')
    print(f'{out}: {len(grid)} tuning cells, {len(confirm)} confirmation runs')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
