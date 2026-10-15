"""One isolated training job. The scheduler supplies an immutable JSON spec."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from nsshape.coefficients import TABLE_PATH, get
from nsshape.diagnostics.svd_logger import SVDLogger
from nsshape.experiments.data import cifar, language, token_batch, sha256
from nsshape.experiments.models import CifarCNN, GPT, parameter_groups
from nsshape.optim.muon import ParameterizedMuon


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(spec, out, device):
    seed = spec['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(spec.get('threads', 2))
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    task = spec['task']
    if task == 'cifar':
        train, validation, test, manifest = cifar(spec['data'])
        model = CifarCNN(spec.get('width', 64))
    elif task == 'language':
        train, validation, test, manifest = language(spec['data'])
        model = GPT(vocab=manifest['vocab_size'], **spec['model'])
    else:
        raise ValueError(f'unsupported task {task}')
    model.to(device)
    muon_params, adam_params = parameter_groups(model)
    cs = get(spec['arm'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    logger = SVDLogger(out/'spectra.jsonl', every_k=spec.get('log_every', 100),
                       max_params=2, population_size=len(muon_params), seed=seed,
                       coefficient_set=cs, precision=spec['precision'])
    hook_seconds = 0.
    def hook(*args):
        nonlocal hook_seconds
        if logger.should_sample(args[0]):
            start = time.perf_counter()
            logger(*args)
            hook_seconds += time.perf_counter()-start
    opt = ParameterizedMuon(muon_params, lr=spec['lr'], coeff_set=spec['arm'],
                            precision=spec['precision'], weight_decay=spec.get('weight_decay', .01),
                            hook=None if spec['phase'] == 'calibration' else hook)
    adam = torch.optim.AdamW(adam_params, lr=spec.get('adam_lr', .001),
                            weight_decay=spec.get('weight_decay', .01)) if adam_params else None
    generator = torch.Generator().manual_seed(seed+10000)
    batch = spec['batch']
    context = spec.get('model', {}).get('context', 0)
    steps = spec['steps']
    if task == 'cifar':
        order = torch.randperm(len(train[1]), generator=generator)
        position = 0

    def autocast():
        return torch.autocast('cuda', dtype=torch.bfloat16) if device.startswith('cuda') else nullcontext()

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        total_loss, correct, count = 0., 0, 0
        if task == 'cifar':
            iterator = ((split[0][i:i+batch].to(device), split[1][i:i+batch].to(device))
                        for i in range(0, len(split[1]), batch))
        else:
            # Fixed, non-overlapping targets; same evaluation tokens for all seeds.
            limit = min(len(split)-1, spec.get('eval_tokens', 65536))
            starts = range(0, limit-context+1, context)
            positions = list(starts)
            def iterator_fn():
                for offset in range(0, len(positions), batch):
                    selected = positions[offset:offset+batch]
                    xs = np.stack([split[i:i+context] for i in selected]).astype(np.int64)
                    ys = np.stack([split[i+1:i+context+1] for i in selected]).astype(np.int64)
                    yield torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)
            iterator = iterator_fn()
        for x, y in iterator:
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='sum')
            total_loss += float(loss)
            correct += int((logits.argmax(-1) == y).sum())
            count += y.numel()
        model.train()
        if count == 0:
            raise ValueError('no evaluation examples/tokens')
        return dict(loss=total_loss/count, accuracy=correct/count, count=count)

    start = time.perf_counter()
    curve, durations = [], []
    status = 'complete'
    completed = 0
    try:
        for step in range(steps):
            if time.perf_counter()-start > spec.get('max_seconds', math.inf):
                status = 'budget_exhausted'
                break
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            before = time.perf_counter()
            if task == 'cifar':
                if position >= len(order):
                    order = torch.randperm(len(order), generator=generator)
                    position = 0
                indices = order[position:position+batch]
                position += batch
                x, y = train[0][indices].to(device), train[1][indices].to(device)
                # Paired augmentation RNG, independent of optimizer behavior.
                flips = torch.rand(len(x), generator=generator) < .5
                x[flips.to(device)] = x[flips.to(device)].flip(-1)
            else:
                x, y = token_batch(train, batch, context, generator, device)
            opt.zero_grad(set_to_none=True)
            if adam:
                adam.zero_grad(set_to_none=True)
            warmup = min(spec.get('warmup', 50), max(1, steps//10))
            fraction = .5*(1+math.cos(math.pi*max(0, step-warmup)/max(1, steps-warmup)))
            schedule = min(1., (step+1)/warmup)*max(.1, fraction)
            for group in opt.param_groups:
                group['lr'] = spec['lr']*schedule
            if adam:
                for group in adam.param_groups:
                    group['lr'] = spec.get('adam_lr', .001)*schedule
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            if not torch.isfinite(loss):
                status = 'nonfinite_loss'
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            if adam:
                adam.step()
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            durations.append(time.perf_counter()-before)
            completed += 1
            if (step+1) % spec.get('eval_every', 200) == 0 or step+1 == steps:
                curve.append(dict(step=step+1, train_loss=float(loss.detach()), validation=evaluate(validation)))
        training_seconds = time.perf_counter()-start
        eval_start = time.perf_counter()
        val = evaluate(validation)
        test_result = evaluate(test) if spec['phase'] == 'confirm' and status == 'complete' else None
        evaluation_seconds = time.perf_counter()-eval_start
        result = dict(spec=spec, status=status, completed_steps=completed, validation=val,
                      test=test_result, curve=curve, seconds=training_seconds,
                      final_evaluation_seconds=evaluation_seconds, diagnostic_seconds=hook_seconds,
                      seconds_per_step=float(np.quantile(durations[2:] or durations, .8)),
                      parameters=sum(p.numel() for p in model.parameters()),
                      dataset=manifest, coefficient_set=asdict(cs),
                      coefficient_table_sha256=sha256(TABLE_PATH),
                      versions=dict(torch=torch.__version__, numpy=np.__version__),
                      device=(torch.cuda.get_device_name() if device.startswith('cuda') else device),
                      cuda_version=torch.version.cuda,
                      artifact_directory=str(out),
                      peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                      tokens_seen=completed*batch*context if task == 'language' else None,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        if spec['phase'] == 'confirm' and status == 'complete':
            torch.save(dict(model=model.state_dict(), muon=opt.state_dict(),
                            adam=adam.state_dict() if adam else None, spec=spec), out/'checkpoint.pt')
        atomic_json(out/'result.json', result)
        return result
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    result = run(json.loads(args.spec.read_text()), args.out, args.device)
    print(json.dumps(json_safe(result), allow_nan=False))
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
