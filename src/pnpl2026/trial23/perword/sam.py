"""Sharpness-aware minimisation (Foret et al., 2021) around any base optimiser.

One step, given Lightning's closure (forward, zero_grad, backward):

    1. the closure: the loss and its gradient g at the weights w
    2. move to w + e, e = rho g / ||g|| (one norm over every parameter)
    3. the closure again: the gradient at w + e replaces g
    4. back to w exactly; the base optimiser steps with that gradient

So each step costs two forward-backward passes, and descends on the worst
loss within radius rho rather than the loss at w: flatter minima.

⚠️ Lightning hands `optimizer.step` a closure only without a GradScaler:
precision "32" or "bf16-mixed", never "16-mixed" (`config.py` refuses it).

The base optimiser shares this object's `param_groups`, so a learning-rate
scheduler built on SAM sets the rates the base steps with.
"""

from __future__ import annotations

import torch

#keeps the step finite when every gradient is zero
NORM_FLOOR = 1e-12


class SAM(torch.optim.Optimizer):
    """Sharpness-aware minimisation around a base optimiser such as AdamW.

    ATTRS:
        base_optimiser (torch.optim.Optimizer): takes the actual step
        param_groups (list): shared with the base optimiser
    """

    def __init__(self, params, base_optimiser_class, rho=0.05, **base_kwargs):
        """
        ARGS:
            params (iterable): the trainable parameters
            base_optimiser_class (type): e.g. torch.optim.AdamW
            rho (float): the neighbourhood's radius, > 0
            **base_kwargs: to the base optimiser -- lr, weight_decay...
        """
        if rho <= 0:
            raise ValueError(f"rho must be > 0, got {rho}")
        super().__init__(params, dict(rho=rho, **base_kwargs))
        self.base_optimiser = base_optimiser_class(self.param_groups, **base_kwargs)

        #one set of groups: the scheduler's rates reach the base's step
        self.param_groups = self.base_optimiser.param_groups
        self.defaults.update(self.base_optimiser.defaults)

    @torch.no_grad()
    def gradient_norm(self):
        """The L2 norm of every parameter's gradient taken together.

        RETURNS:
            scalar tensor
        """
        norms = [parameter.grad.norm(p=2) for group in self.param_groups for parameter in group["params"]
                 if parameter.grad is not None]
        if not norms:
            raise RuntimeError("SAM found no gradient to perturb along; did the closure call backward?")
        #dim=0 stacks one norm per parameter into the global one
        return torch.stack(norms, dim=0).norm(p=2)

    @torch.no_grad()
    def perturb(self):
        """Move every parameter to w + e, keeping w to return to exactly."""
        norm = self.gradient_norm()
        for group in self.param_groups:
            scale = group["rho"] / (norm + NORM_FLOOR)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                #a copy of w, since (w + e) - e need not round back to w
                self.state[parameter]["sam_origin"] = parameter.detach().clone()
                parameter.add_(parameter.grad * scale.to(parameter))

    @torch.no_grad()
    def restore(self):
        """Put every perturbed parameter back to w, bit for bit."""
        for group in self.param_groups:
            for parameter in group["params"]:
                origin = self.state[parameter].pop("sam_origin", None)
                if origin is not None:
                    parameter.copy_(origin)

    @torch.no_grad()
    def step(self, closure=None):
        """One SAM step: gradient at w, gradient at w + e, the base step at w.

        ARGS:
            closure (callable): forward, zero_grad and backward on one batch

        RETURNS:
            the closure's loss at w
        """
        if closure is None:
            raise RuntimeError("SAM needs a closure: it evaluates every batch twice.")
        closure = torch.enable_grad()(closure)

        #the gradient at w decides the direction of the perturbation
        loss = closure()
        self.perturb()

        #the gradient at w + e is the one the base optimiser follows
        closure()
        self.restore()
        self.base_optimiser.step()
        return loss
