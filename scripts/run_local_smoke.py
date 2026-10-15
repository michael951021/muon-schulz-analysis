"""Real small-language training check; explicitly not the GPU confirmation."""
import argparse
import json
from pathlib import Path
import time

from run_study import Scheduler, fingerprint, select
from nsshape.experiments.train import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('data/datasets/shakespeare'))
    parser.add_argument('--out', type=Path, default=Path('runs/local_smoke'))
    args = parser.parse_args()
    source = fingerprint()
    scheduler = Scheduler(args.out.resolve(), ['cpu0', 'cpu1'], time.monotonic()+1800, source, cpu=True)
    arms = [('gamma_0.10', 'fp32'), ('keller', 'fp32'), ('exact_svd', 'fp32'),
            ('gamma_0.10', 'bf16'), ('keller', 'bf16')]
    def spec(arm, precision, phase, lr, seed, steps):
        return dict(task='language', arm=arm, precision=precision, phase=phase,
                    lr=lr, seed=seed, steps=steps, data=str(args.data.resolve()),
                    batch=8, model=dict(width=64, layers=2, heads=4, context=64),
                    adam_lr=.001, weight_decay=.01, eval_tokens=8192, eval_every=150,
                    log_every=150, max_seconds=300, threads=1)
    tuning = scheduler.run([spec(arm, precision, 'tune', lr, 0, 300)
                            for arm, precision in arms for lr in (.005, .02)])
    chosen = select(tuning)
    confirmation = scheduler.run([spec(arm, precision, 'confirm', chosen[(arm, precision)][1], seed, 600)
                                  for arm, precision in arms for seed in (10, 11, 12)])
    result = dict(status='complete', scope='CPU tinyshakespeare smoke; not CIFAR/FineWeb confirmation',
                  source_sha256=source, tuning=tuning, confirmation=confirmation,
                  selections={f'{arm}/{precision}': lr for (arm, precision), (_, lr) in chosen.items()})
    atomic_json(args.out/'smoke.json', result)
    print(f'Wrote {args.out}/smoke.json')


if __name__ == '__main__':
    main()
