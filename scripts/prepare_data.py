"""Prepare checksum-pinned CIFAR or document-disjoint GPT-2 token splits."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import urllib.request

import numpy as np
from nsshape.experiments.data import sha256
from nsshape.experiments.train import atomic_json


def prepare_shakespeare(source, out):
    text = source.read_text()
    vocabulary = sorted(set(text))
    lookup = {char: i for i, char in enumerate(vocabulary)}
    encoded = np.array([lookup[c] for c in text], dtype=np.uint16)
    left, right = int(len(encoded)*.85), int(len(encoded)*.925)
    for name, array in zip(('train', 'validation', 'test'),
                           (encoded[:left], encoded[left:right], encoded[right:])):
        array.tofile(out/f'{name}.bin')
    return dict(dataset='tinyshakespeare', source_sha256=sha256(source),
                tokenizer='characters', vocab_size=len(vocabulary), vocabulary=vocabulary,
                split='contiguous 85%/7.5%/7.5%; local smoke only')


def prepare_fineweb(args, out):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import tiktoken
    repository = 'HuggingFaceFW/fineweb'
    revision = args.revision or HfApi().dataset_info(repository).sha
    encoder = tiktoken.get_encoding('gpt2')
    dataset = load_dataset(repository, name='sample-10BT', split='train',
                           revision=revision, streaming=True)
    limits = dict(train=args.train_tokens, validation=args.eval_tokens, test=args.eval_tokens)
    counts = dict.fromkeys(limits, 0)
    seen = set()
    digests = {name: hashlib.sha256() for name in limits}
    paths = {name: (out/f'{name}.bin').open('wb') for name in limits}
    try:
        for record in dataset:
            text = record['text']
            digest = hashlib.sha256(text.encode()).digest()
            if digest in seen:
                continue
            seen.add(digest)
            bucket = int.from_bytes(digest[:4], 'big') % 100
            split = 'test' if bucket < 2 else 'validation' if bucket < 4 else 'train'
            if counts[split] >= limits[split]:
                continue
            tokens = encoder.encode_ordinary(text)+[encoder.eot_token]
            tokens = tokens[:limits[split]-counts[split]]
            array = np.array(tokens, dtype=np.uint16)
            paths[split].write(array.tobytes())
            digests[split].update(digest)
            before = counts[split]
            counts[split] += len(array)
            if before//1_000_000 != counts[split]//1_000_000:
                print(counts, flush=True)
            if all(counts[name] >= limits[name] for name in limits):
                break
        if not all(counts[name] >= limits[name] for name in limits):
            raise RuntimeError(f'dataset exhausted before token limits: {counts}')
    finally:
        for handle in paths.values():
            handle.close()
    return dict(dataset=repository, configuration='sample-10BT', revision=revision,
                tokenizer='gpt2', vocab_size=50257, tiktoken=importlib.metadata.version('tiktoken'),
                split='SHA256(document) % 100: test 0-1, validation 2-3, train 4-99',
                document_digests={name: d.hexdigest() for name, d in digests.items()},
                token_limits=limits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['cifar', 'fineweb', 'shakespeare'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--text', type=Path)
    parser.add_argument('--revision')
    parser.add_argument('--train-tokens', type=int, default=100_000_000)
    parser.add_argument('--eval-tokens', type=int, default=1_000_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == 'cifar':
        path = args.out/'cifar-10-python.tar.gz'
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != 'c58f30108f718f92721af3b95e74349a':
            temporary = path.with_suffix('.download')
            urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', temporary)
            if hashlib.md5(temporary.read_bytes()).hexdigest() != 'c58f30108f718f92721af3b95e74349a':
                raise ValueError('CIFAR download checksum mismatch')
            temporary.replace(path)
        print(f'CIFAR ready: {path}')
        return
    if (args.out/'manifest.json').exists():
        raise SystemExit('output already has a manifest; choose a new directory to preserve dataset provenance')
    if args.source == 'shakespeare':
        if args.text is None:
            parser.error('--text is required for shakespeare')
        manifest = prepare_shakespeare(args.text, args.out)
    else:
        manifest = prepare_fineweb(args, args.out)
    manifest['files'] = {split: dict(sha256=sha256(args.out/f'{split}.bin'),
                                    tokens=(args.out/f'{split}.bin').stat().st_size//2)
                         for split in ('train', 'validation', 'test')}
    atomic_json(args.out/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
