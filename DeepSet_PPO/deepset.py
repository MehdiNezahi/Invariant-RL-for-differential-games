"""Permutation-equivariant DeepSet actor and critic, shared by all players.

For a joint state x of shape (B, N), player i's input is
    h_i = [ x_i / s ,  sum_{j != i} phi(x_j / s) ]
and the heads rho_actor / rho_critic map h_i to player i's output. Hence
    f(P x) = P f(x)  for every permutation P,
and the same weights work for any N (sum pooling keeps the scale of
z_i = sum_{j != i} x_j that Salo's feedback law depends on).

`x_own` lets the caller feed the own-input path separately from the pooled
path: with x_own = x.requires_grad_() and x detached, autograd of
sum_i V_i w.r.t. x_own gives exactly the diagonal dV_i/dx_i (the pooled path
excludes j = i, so x_i only enters V_i through x_own).

SiLU activations: the HJB residual differentiates V w.r.t. x, so V must be
smooth (ReLU would give piecewise-constant gradients).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(inp: int, hidden: int, out: int, layers: int = 2) -> nn.Sequential:
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    mods.append(nn.Linear(d, out))
    return nn.Sequential(*mods)


class DeepSetEncoder(nn.Module):
    def __init__(self, hidden: int = 64, x_scale: float = 10.0):
        super().__init__()
        self.phi = mlp(1, hidden, hidden)
        self.x_scale = x_scale
        self.out_dim = 1 + hidden

    def forward(self, x: torch.Tensor, x_own: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N) -> (B, N, 1 + hidden)."""
        x_own = x if x_own is None else x_own
        e = self.phi((x / self.x_scale).unsqueeze(-1))       # (B, N, H)
        pooled_others = e.sum(-2, keepdim=True) - e          # sum_{j != i}
        return torch.cat([(x_own / self.x_scale).unsqueeze(-1), pooled_others], dim=-1)


class DeepSetActor(nn.Module):
    """Beta policy on u in (0, 1); the extraction is q = u * upper(x)."""

    def __init__(self, hidden: int = 64, x_scale: float = 10.0):
        super().__init__()
        self.enc = DeepSetEncoder(hidden, x_scale)
        self.rho = mlp(self.enc.out_dim, hidden, 2)

    def forward(self, x: torch.Tensor, x_own: torch.Tensor | None = None):
        a, b = self.rho(self.enc(x, x_own)).unbind(-1)
        # 1 + softplus keeps the Beta unimodal (alpha, beta >= 1)
        return torch.distributions.Beta(1.0 + F.softplus(a), 1.0 + F.softplus(b))

    def mean_action(self, x: torch.Tensor, game) -> torch.Tensor:
        """Deterministic policy pi_hat(x) = E[u] * upper(x), shape (B, N)."""
        return self(x).mean * game.action_upper(x)


class DeepSetCritic(nn.Module):
    def __init__(self, hidden: int = 64, x_scale: float = 10.0):
        super().__init__()
        self.enc = DeepSetEncoder(hidden, x_scale)
        self.rho = mlp(self.enc.out_dim, hidden, 1)

    def forward(self, x: torch.Tensor, x_own: torch.Tensor | None = None) -> torch.Tensor:
        return self.rho(self.enc(x, x_own)).squeeze(-1)


def own_gradient(critic: nn.Module, x: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """dV_i/dx_i for every player, shape (B, N), in one backward pass."""
    x_own = x.detach().clone().requires_grad_(True)
    V = critic(x.detach(), x_own=x_own)
    return torch.autograd.grad(V.sum(), x_own, create_graph=create_graph)[0]
