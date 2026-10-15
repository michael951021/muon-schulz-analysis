"""Dataset manifests and fixed train/validation/test splits."""
from pathlib import Path
import hashlib
import json
import pickle
import tarfile
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cifar(root):
    archive = Path(root) / 'cifar-10-python.tar.gz'
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if md5 != 'c58f30108f718f92721af3b95e74349a':
        raise ValueError('CIFAR archive checksum mismatch; download the official complete archive')
    with tarfile.open(archive, 'r:gz') as handle:
        def read(name):
            # Verified official pickle; no arbitrary archive extraction.
            member = handle.extractfile(f'cifar-10-batches-py/{name}')
            record = pickle.load(member, encoding='bytes')
            return record[b'data'], np.array(record[b'labels'])
        batches = [read(f'data_batch_{i}') for i in range(1, 6)]
        x = np.concatenate([a for a, _ in batches])
        y = np.concatenate([b for _, b in batches])
        tx, ty = read('test_batch')
    # Identical fixed random split across every arm and initialization seed.
    indices = np.random.default_rng(1729).permutation(len(y))
    def tensors(a, b):
        images = torch.from_numpy(a.copy()).reshape(-1, 3, 32, 32).float()/255
        mean = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
        std = torch.tensor([.247, .2435, .2616])[None, :, None, None]
        return (images-mean)/std, torch.from_numpy(b.copy()).long()
    return (tensors(x[indices[5000:]], y[indices[5000:]]),
            tensors(x[indices[:5000]], y[indices[:5000]]), tensors(tx, ty),
            dict(dataset='CIFAR-10', sha256=sha256(archive), split_seed=1729,
                 train=45000, validation=5000, test=10000))


def language(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['tokenizer'] not in {'gpt2', 'characters'}:
        raise ValueError('unsupported tokenizer')
    arrays = []
    for split in ('train', 'validation', 'test'):
        path = root/f'{split}.bin'
        if sha256(path) != manifest['files'][split]['sha256']:
            raise ValueError(f'{split} checksum mismatch')
        arrays.append(np.memmap(path, dtype=np.uint16, mode='r'))
    return (*arrays, manifest)


def token_batch(data, batch, context, generator, device):
    if len(data) <= context:
        raise ValueError('token split is shorter than context')
    offsets = torch.randint(len(data)-context, (batch,), generator=generator)
    x = torch.stack([torch.from_numpy(np.array(data[int(i):int(i)+context], dtype=np.int64)) for i in offsets])
    y = torch.stack([torch.from_numpy(np.array(data[int(i)+1:int(i)+context+1], dtype=np.int64)) for i in offsets])
    return x.to(device), y.to(device)
