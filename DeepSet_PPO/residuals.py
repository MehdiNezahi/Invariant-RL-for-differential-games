"""HJB residual R_i, Hamiltonian mismatch P_i, and the Nash-gap bound

    Nash_Gap(pi_hat) <= (2/delta) sum_i ||R_i||_inf + (1/delta) sum_i ||P_i||_inf

R_i(x) = delta V_i(x) - F_i(x, q) - grad_x V_i(x) . f(x, q),   f = -q,  q = pi_hat(x)
P_i(x) = max_{q_i'} H_i(x, q_i', q_-i) - H_i(x, q_i, q_-i)  >= 0

`value_fn(x) -> (B, N)` can be the DeepSet critic or Salo's quadratic value,
so the same code is checked on the closed form (R = P = 0 in the interior).
"""
from __future__ import annotations

import torch
from torch.func import jvp


def hjb_residual(game, value_fn, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact dynamics f = -q (no finite difference); grad V . f is a JVP, so
    it includes the cross terms dV_i/dx_j * (-q_j). Differentiable w.r.t. the
    parameters of value_fn."""
    V, dV_f = jvp(value_fn, (x,), (-q,))
    return game.delta * V - game.reward_density(x, q) - dV_f


def diag_gradient(value_fn, x: torch.Tensor) -> torch.Tensor:
    """dV_i/dx_i for a generic value_fn (N backward passes). The DeepSet
    critic has a one-pass version, deepset.own_gradient."""
    x = x.detach().requires_grad_(True)
    V = value_fn(x)
    cols = [
        torch.autograd.grad(V[..., i].sum(), x, retain_graph=True)[0][..., i]
        for i in range(x.shape[-1])
    ]
    return torch.stack(cols, dim=-1)


def hamiltonian_gap(game, x, q, dVi_dxi):
    """Returns (P, q_star): P_i = H_i(q_i*) - H_i(q_i), q_-i held fixed."""
    q_others = q.sum(-1, keepdim=True) - q
    q_star = game.hamiltonian_argmax(x, q_others, dVi_dxi)
    P = game.hamiltonian_own(x, q_star, q_others, dVi_dxi) - game.hamiltonian_own(x, q, q_others, dVi_dxi)
    return P, q_star


def nash_gap_bound(R: torch.Tensor, P: torch.Tensor, delta: float) -> dict:
    """Empirical bound on a batch of states (sup norm -> max over the batch).
    R, P: (B, N)."""
    R_sup = R.abs().amax(0)    # (N,)
    P_sup = P.amax(0)
    return {
        "R_sup": R_sup.sum().item(),
        "P_sup": P_sup.sum().item(),
        "R_mean": R.abs().mean().item(),
        "P_mean": P.mean().item(),
        "bound": (2.0 / delta) * R_sup.sum().item() + (1.0 / delta) * P_sup.sum().item(),
    }
