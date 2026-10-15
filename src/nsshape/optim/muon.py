"""Muon, parameterized by a named coefficient set.

The optimizer knows only a coefficient-set name and an optional hook. It has no
idea what model it is training, which is what lets the same code drive both
cifar10-airbench and modded-nanogpt.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from nsshape.coefficients import get
from nsshape.optim.orthogonalize import orthogonalize


class ParameterizedMuon(torch.optim.Optimizer):
    """Muon with a swappable orthogonalization backend.

    Args:
        params: 2D parameters only. Embeddings, biases and norms belong in a
            separate AdamW group, as in standard Muon.
        lr: learning rate.
        momentum: heavy-ball coefficient on the gradient buffer.
        nesterov: use the Nesterov-style lookahead buffer.
        weight_decay: decoupled weight decay.
        coeff_set: entry name in the committed coefficient table.
        hook: optional callable(step, param_index, M_before, M_after), where
            M_before is the momentum matrix and M_after the orthogonalized
            update. Used by the SVD spectral logger.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        coeff_set: str = "keller",
        hook: Callable[[int, int, torch.Tensor, torch.Tensor], None] | None = None,
        precision: str = "bf16",
    ):
        if lr < 0 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError("invalid learning rate, momentum, or weight decay")
        if precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        self.coefficient_set = get(coeff_set)
        self.precision = precision
        self.hook = hook
        self.step_count = 0
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def state_dict(self):
        result = super().state_dict()
        from dataclasses import asdict
        result["nsshape"] = dict(coefficient_set=asdict(self.coefficient_set),
                                 step_count=self.step_count, precision=self.precision)
        return result

    def load_state_dict(self, state_dict):
        from nsshape.coefficients import CoefficientSet
        data = dict(state_dict)
        metadata = data.pop("nsshape", None)
        if metadata is None:
            raise ValueError("checkpoint lacks nsshape coefficient/precision metadata")
        cs = dict(metadata["coefficient_set"])
        if cs["coeffs"] is not None:
            cs["coeffs"] = tuple(cs["coeffs"])
        self.coefficient_set = CoefficientSet(**cs)
        self.precision = metadata["precision"]
        self.step_count = metadata["step_count"]
        return super().load_state_dict(data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        index = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_index = index
                index += 1
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"ParameterizedMuon requires 2D parameters, got {p.ndim}D; "
                        "put other parameters in a separate AdamW group"
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(p.grad)
                M = (
                    p.grad.add(buf, alpha=group["momentum"])
                    if group["nesterov"]
                    else buf
                )

                update = orthogonalize(M, self.coefficient_set, self.precision)

                if self.hook is not None:
                    self.hook(self.step_count, param_index, M, update)

                if group["weight_decay"] != 0.0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Standard Muon shape correction: scale by sqrt of the aspect
                # ratio so wide and tall matrices take comparable-sized steps.
                shape_scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(update, alpha=-group["lr"] * shape_scale)

        self.step_count += 1
        return loss
