from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, 
        lr=1e-3, weight_decay=0.01,
        betas=(0.9, 0.999), eps=1e-8
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr, "beta_1": betas[0] , "beta_2": betas[1],
            "weight_decay_rate": weight_decay, "eps": eps
        }
        super().__init__(params, defaults)


    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay_rate = group["weight_decay_rate"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # Get the gradient of loss wrt to p
                grad = p.grad.data
                
                # Apply weight decay to parameters
                p.data -= lr * weight_decay_rate * p.data
                
                # Update the first moment estimate that is used for the direction of the grads
                moment_1 = state.get("moment_1", 0.0)
                moment_1 = (beta_1 * moment_1) + ((1 - beta_1) * grad)

                # Update the second moment estimate that is used for the magnitude of the grads
                moment_2 = state.get("moment_2", 0.0)
                moment_2 = (beta_2 * moment_2) + ((1 - beta_2) * (grad ** 2))

                # Compute adjusted learning rate for iteration t
                t = state.get("t", 1)
                lr_step_t = lr * ((1 - beta_2 ** t) ** 0.5) / (1 - beta_1 ** t)
                
                # Apply moment-adjusted weight updates
                p.data -= lr_step_t * moment_1 / ((moment_2 ** 0.5) + eps)

                # Update the states of each parameter
                state["t"] = t + 1
                state["moment_1"] = moment_1
                state["moment_2"] = moment_2
        return loss

def cosine_lr_scheduler(it, lr_max, lr_min, warmup_it, cosine_cycle_it):
    """
    parameters:
        it: iteration number to get the learning rate for
        lr_max: the maximum learning rate
        lr_min: the minimum/final learning rate
        warmup_it: the number of iterations to linearly warm-up the learning rate.
        cosine_cycle_iters: iteration number to check to continue cosine or not
    """
    if it < warmup_it:
        return (it / warmup_it) * lr_max
    elif warmup_it <= it <= cosine_cycle_it:
        cos_anneal = (
            (1 + math.cos(((it - warmup_it) * math.pi) / (cosine_cycle_it - warmup_it))) / 2
        )
        return lr_min + cos_anneal * (lr_max - lr_min)
    return lr_min

def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter],
        max_l2_norm: float
) -> None:
    params = list(parameters)
    g = 0.0
    for p in params:
        if p.grad is None:
            continue
        g += torch.linalg.vector_norm(p.grad.data) ** 2
    g = g ** 0.5
    if g >= max_l2_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data *= max_l2_norm / (g + 1e-6)
