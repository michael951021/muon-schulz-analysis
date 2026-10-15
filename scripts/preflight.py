"""Fail early on the target machine; no GPU claims from CPU-only checks."""
import argparse
import json
import torch
from nsshape.coefficients import load_table
from nsshape.optim.orthogonalize import orthogonalize
from nsshape.diagnostics.schema import spectrum_record
from nsshape.experiments.models import GPT, parameter_groups
from nsshape.optim.muon import ParameterizedMuon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-other-gpus', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit('need two visible CUDA GPUs; no study was started')
    results = []
    for index in (0, 1):
        torch.cuda.set_device(index)
        properties = torch.cuda.get_device_properties(index)
        if not args.allow_other_gpus and '3090' not in properties.name:
            raise SystemExit(f'GPU {index} is {properties.name}; explicitly revise hardware protocol')
        if properties.total_memory < 23*1024**3 or not torch.cuda.is_bf16_supported():
            raise SystemExit(f'GPU {index} lacks required 24GB class memory or bf16 support')
        table = load_table()
        matrix = torch.randn(32, 64, device=f'cuda:{index}')
        for cs in table.values():
            for precision in ('fp32', 'bf16'):
                output = orthogonalize(matrix, cs, precision)
                if not torch.isfinite(output).all():
                    raise RuntimeError(f'nonfinite backend output: {cs.name}/{precision}')
                spectrum_record(0, 0, 'after', output).to_json()
        model = GPT().to(f'cuda:{index}')
        muon, adam = parameter_groups(model)
        optimizer = ParameterizedMuon(muon, coeff_set='exact_svd', precision='fp32')
        adam_optimizer = torch.optim.AdamW(adam, lr=.0003)
        tokens = torch.randint(50257, (8, 256), device=f'cuda:{index}')
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 50257), tokens.reshape(-1))
        loss.backward()
        optimizer.step()
        adam_optimizer.step()
        torch.cuda.synchronize()
        results.append(dict(index=index, name=properties.name,
                            total_memory=properties.total_memory,
                            peak_memory=torch.cuda.max_memory_allocated(),
                            torch=torch.__version__, cuda=torch.version.cuda))
        del model, optimizer, adam_optimizer, logits, loss, tokens, muon, adam
        torch.cuda.empty_cache()
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
