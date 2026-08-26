"""Sampling hook that logs singular-value spectra before and after the step.

Attaches to ParameterizedMuon as a plain callable, so the same logger works for
cifar10-airbench and modded-nanogpt without an adapter.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from nsshape.diagnostics.schema import spectrum_record


class SVDLogger:
    """Log fp32 spectra of a sampled subset of momentum matrices every k steps.

    Args:
        path: JSONL output file.
        every_k: sample on steps where step % every_k == 0.
        max_params: at most this many distinct parameters per sampled step.
        n_bins: histogram resolution.
        seed: controls which parameters get sampled, for reproducibility.
    """

    def __init__(
        self,
        path: str | Path,
        every_k: int = 50,
        max_params: int = 4,
        n_bins: int = 32,
        seed: int = 0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every_k = every_k
        self.max_params = max_params
        self.n_bins = n_bins
        self._rng = random.Random(seed)
        self._handle = self.path.open("w")
        self._sampled_this_step = 0
        self._current_step: int | None = None

    def should_sample(self, step: int) -> bool:
        return step % self.every_k == 0

    def __call__(
        self,
        step: int,
        param_index: int,
        M_before: torch.Tensor,
        M_after: torch.Tensor,
    ) -> None:
        if not self.should_sample(step):
            return
        if step != self._current_step:
            self._current_step = step
            self._sampled_this_step = 0
        if self._sampled_this_step >= self.max_params:
            return
        self._sampled_this_step += 1

        for phase, M in (("before", M_before), ("after", M_after)):
            record = spectrum_record(step, param_index, phase, M, self.n_bins)
            self._handle.write(record.to_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> SVDLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
