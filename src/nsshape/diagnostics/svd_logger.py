"""Sampling hook that logs singular-value spectra before and after the step.

Attaches to ParameterizedMuon as a plain callable, so the same logger works for
cifar10-airbench and modded-nanogpt without an adapter.
"""

from __future__ import annotations

import random
from dataclasses import replace
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
        seed: controls uniform sampling when population_size is provided.
        population_size: stable parameter population size. If omitted, log the
            first max_params calls for compatibility; this policy is not random.
    """

    def __init__(
        self,
        path: str | Path,
        every_k: int = 50,
        max_params: int = 4,
        n_bins: int = 32,
        seed: int = 0,
        population_size: int | None = None,
        coefficient_set=None,
        precision: str = "bf16",
    ):
        if every_k < 1 or max_params < 1 or n_bins < 1:
            raise ValueError("logging intervals, budget, and bins must be positive")
        if population_size is not None and population_size < 1:
            raise ValueError("population_size must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every_k = every_k
        self.max_params = max_params
        self.n_bins = n_bins
        self._rng = random.Random(seed)
        self.population_size = population_size
        self.coefficient_set = coefficient_set
        self.precision = precision
        self._selected = None
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
            self._selected = (set(self._rng.sample(range(self.population_size),
                                min(self.max_params, self.population_size)))
                              if self.population_size is not None else None)
        if self._selected is not None and param_index not in self._selected:
            return
        if self._sampled_this_step >= self.max_params:
            return
        self._sampled_this_step += 1

        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float32
        normalized = M_before.detach().to(dtype)
        norm = normalized.norm()
        normalized = normalized / (norm + 1e-7)
        s = torch.linalg.svdvals(normalized.float()).cpu()
        cs = self.coefficient_set

        for phase, M in (("before", M_before), ("after", M_after)):
            record = spectrum_record(step, param_index, phase, M, self.n_bins)
            record = replace(record,
                             coefficient_set=cs.name if cs else None,
                             precision=self.precision, input_norm=float(norm),
                             normalized_input_min=float(s.min()),
                             normalized_input_max=float(s.max()),
                             below_sigma_min=int((s < cs.sigma_min).sum()) if cs else None)
            self._handle.write(record.to_json() + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> SVDLogger:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
