"""Summarize completed confirmation runs, pairing seeds against Keller."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
from nsshape.experiments.train import atomic_json


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['status'] != 'complete' or row['test'] is None:
            raise ValueError('report requires complete confirmation runs with held-out tests')
        spec = row['spec']
        if row['completed_steps'] != spec['steps']:
            raise ValueError('partial work cannot be included')
        groups[(spec['arm'], spec['precision'])].append(row)
    work = {(row['spec']['steps'], row['spec']['batch'], row['spec'].get('model', {}).get('context', 0)) for row in rows}
    if len(work) != 1:
        raise ValueError('comparison has mismatched training work')
    summaries = []
    for (arm, precision), group in sorted(groups.items()):
        seeds = [row['spec']['seed'] for row in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError('duplicate seed within an arm')
        losses = [row['test']['loss'] for row in group]
        baseline = groups.get(('keller', precision))
        paired = None
        if baseline is not None:
            reference = {row['spec']['seed']: row['test']['loss'] for row in baseline}
            if set(reference) != set(seeds):
                raise ValueError('paired comparisons require identical confirmation seeds')
            differences = np.array([row['test']['loss']-reference[row['spec']['seed']] for row in group])
            mean = float(differences.mean())
            half = float(t.ppf(.975, len(differences)-1)*differences.std(ddof=1)/np.sqrt(len(differences))) if len(differences) > 1 else None
            paired = dict(mean=mean, ci95=[mean-half, mean+half] if half is not None else None,
                          n=len(differences), negative_favors_arm=True)
        summaries.append(dict(arm=arm, precision=precision, seeds=seeds,
                              mean_test_loss=float(np.mean(losses)),
                              sd_test_loss=float(np.std(losses, ddof=1)) if len(losses)>1 else None,
                              mean_test_accuracy=float(np.mean([row['test']['accuracy'] for row in group])),
                              mean_seconds=float(np.mean([row['seconds'] for row in group])),
                              lr=group[0]['spec']['lr'], steps=group[0]['spec']['steps'],
                              mean_diagnostic_seconds=float(np.mean([row['diagnostic_seconds'] for row in group])),
                              tokens_per_run=group[0].get('tokens_seen'), paired_vs_keller=paired))
        exact = groups.get(('exact_svd', 'fp32'))
        if exact is not None:
            reference = {row['spec']['seed']: row['test']['loss'] for row in exact}
            if set(reference) != set(seeds):
                raise ValueError('exact-SVD comparison requires identical seeds')
            differences = np.array([row['test']['loss']-reference[row['spec']['seed']] for row in group])
            mean = float(differences.mean())
            half = float(t.ppf(.975, len(differences)-1)*differences.std(ddof=1)/np.sqrt(len(differences))) if len(differences)>1 else None
            summaries[-1]['paired_vs_exact_svd'] = dict(mean=mean, ci95=[mean-half, mean+half] if half is not None else None)
    return summaries


def report(input_path, out):
    payload = json.loads(input_path.read_text())
    if payload['status'] != 'complete':
        raise ValueError('study is incomplete')
    out.mkdir(parents=True, exist_ok=True)
    smoke = 'scope' in payload
    sections = {'language': payload} if smoke else {name: payload[name] for name in ('cifar', 'language')}
    metrics = {name: summarize(section['confirmation']) for name, section in sections.items()}
    title = 'Local training smoke results' if smoke else 'Dual RTX 3090 ablation results'
    lines = [f'# {title}', '',
             ('Scope: CPU tinyshakespeare, two-layer 64-wide character transformer. '
              '**This is a pipeline check, not the planned CIFAR/FineWeb confirmation.**' if smoke else
              'Scope: budget-limited CIFAR-10 CNN sweep and approximately 124M-parameter GPT on FineWeb.'), '',
             'Learning rates were selected on validation loss in tuning runs. '
             'Reported test metrics come only from fresh confirmation seeds, at the final checkpoint. '
             'Lower loss is better. Paired differences use the same seed against Keller at the same precision.', '']
    for task, rows in metrics.items():
        lines += [f'## {task.capitalize()}', '',
                  '| Arm | Precision | LR | Steps | Test loss mean ± SD | Accuracy | Seconds/run | Loss difference vs Keller (95% CI) |',
                  '|---|---|---:|---:|---:|---:|---:|---|']
        for row in rows:
            paired = row['paired_vs_keller']
            difference = '—'
            if paired and paired['ci95']:
                lo, hi = paired['ci95']
                difference = f"{paired['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            sd = row['sd_test_loss']
            spread = f'{sd:.4f}' if sd is not None else 'n/a'
            lines.append(f"| {row['arm']} | {row['precision']} | {row['lr']} | {row['steps']} | "
                         f"{row['mean_test_loss']:.4f} ± {spread} | {row['mean_test_accuracy']:.2%} | "
                         f"{row['mean_seconds']:.1f} | {difference} |")
        lines += ['', f"Confirmation seeds: {sorted(rows[0]['seeds'])}.", '']
        lines += ['| Arm | Precision | Paired test-loss difference vs exact SVD (95% CI) |',
                  '|---|---|---|']
        for row in rows:
            paired = row.get('paired_vs_exact_svd')
            if paired and paired['ci95']:
                lo, hi = paired['ci95']
                lines.append(f"| {row['arm']} | {row['precision']} | {paired['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
        lines += ['']
    lines += ['## Interpretation', '',
              'Intervals are Student-t intervals over paired seed differences, not uncertainty over '
              'individual examples. Three confirmation seeds give low statistical power. '
              'The intervals are descriptive and are not corrected for multiple comparisons. '
              'Selection among fitted arms occurred before these fresh seeds, but interpretation '
              'still applies only to the selected family and this training regime.', '',
              'Exact SVD is a partial polar factor with numerical null-space directions removed; '
              'its cost is measured, rather than described as compute-matched to the quintics. '
              'All polynomial arms use five NS iterations. Runtime includes periodic validation '
              'and diagnostics; diagnostic time is available separately in metrics.json.', '']
    if smoke:
        lines += ['No conclusion about H1, CIFAR accuracy, or FineWeb language modeling follows '
                  'from these smoke results. The two-GPU protocol remains to be run on the target machine.', '']
    else:
        fp32 = next(row for row in metrics['language'] if row['arm'].startswith('gamma_') and row['precision']=='fp32')
        comparison = fp32['paired_vs_keller']
        if comparison['ci95'][1] < 0:
            conclusion = 'The selected fp32 fitted arm has lower held-out language loss than Keller in this budget-limited run.'
        elif comparison['ci95'][0] > 0:
            conclusion = 'Keller has lower held-out language loss than the selected fp32 fitted arm in this budget-limited run.'
        else:
            conclusion = 'The paired interval does not resolve a language-loss difference between the selected fp32 fit and Keller.'
        lines += [conclusion+' This is not proof of universal optimality. Compare the exact-SVD '
                  'anchor and realized bf16 spectra before attributing differences to shape alone.', '',
                  'The approximately 124M model uses context 256 and the calibrated token budget shown '
                  'in study.json. It is a shortened confirmation, not a 2–5B-token reproduction '
                  'of modded-nanogpt or the airbench speedrun.', '']
        exact_comparison = fp32['paired_vs_exact_svd']
        if exact_comparison['ci95'][1] < 0:
            lines += ['The fitted fp32 arm also has a negative paired interval against exact SVD. '
                      'This is descriptive evidence consistent with H1 in this training regime; '
                      'it remains subject to limited seeds and multiple comparisons.', '']
        else:
            lines += ['The fitted fp32 arm does not show a wholly negative paired interval against '
                      'exact SVD. This experiment therefore does not resolve H1 in favor of '
                      'deliberate distortion over the exact-polar anchor.', '']
    lines += ['## Provenance', '', f'Input: `{input_path}`.', '',
              'The raw result file records exact coefficient values and their hash, source fingerprint, '
              'dataset hashes and splits, versions, measured memory, curves, and completed work. '
              'Per-job spectra and checkpoints remain in the run directory. This report should be '
              'read alongside those artifacts; aggregate loss does not certify the realized spectral band.', '']
    (out/'results.md').write_text('\n'.join(lines))
    atomic_json(out/'metrics.json', metrics)
    # Preserve the complete compact records in a tracked report directory.
    atomic_json(out/'raw_results.json', payload)
    print(f'Wrote {out}/results.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report(args.input, args.out)


if __name__ == '__main__':
    main()
