"""Salo & Tahvonen (2001) symmetric common-pool oligopoly, vectorized in torch.

State x of shape (..., N): one stock per player. Control q of shape (..., N).
    dx_i/dt = -q_i
    F_i(x, q) = (p - unit_cost_i) q_i = (c x_i - sum_j q_j) q_i
with p = pbar - sum_j q_j and unit_cost_i = pbar - c x_i (pbar cancels out, it
is kept only for parity with salo_tahvonen_feedback.ipynb).

Feasibility: a player cannot extract more than its stock within one step,
q_i <= x_i / dt, on top of the action bound q_i <= q_max.

Discounting uses gamma = exp(-delta dt) so the discrete returns match Salo's
e^{-delta t} weighting and the continuous HJB  delta V = F + grad V . f
holds with the same delta.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch


@dataclass(frozen=True)
class SaloGame:
    n_players: int = 3
    c: float = 0.5
    delta: float = 0.05
    pbar: float = 10.0
    dt: float = 0.1
    q_max: float = 2.0

    @property
    def gamma(self) -> float:
        return math.exp(-self.delta * self.dt)

    def with_n(self, n_players: int) -> "SaloGame":
        return replace(self, n_players=n_players)

    # --- dynamics / reward -------------------------------------------------
    def action_upper(self, x: torch.Tensor) -> torch.Tensor:
        """State-dependent upper bound on q_i: min(q_max, x_i / dt)."""
        return torch.clamp(x / self.dt, max=self.q_max)

    def reward_density(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """F_i(x, q), shape (..., N). Not yet multiplied by dt."""
        return (self.c * x - q.sum(-1, keepdim=True)) * q

    def step(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x - q * self.dt, min=0.0)

    # --- Hamiltonian -------------------------------------------------------
    def hamiltonian_own(self, x, q_own, q_others_sum, dVi_dxi):
        """Part of H_i that depends on q_i (the terms in q_j, j != i, cancel in
        any comparison over q_i):  F_i(x; q_i, q_-i) - dV_i/dx_i * q_i."""
        return (self.c * x - q_own - q_others_sum) * q_own - dVi_dxi * q_own

    def hamiltonian_argmax(self, x, q_others_sum, dVi_dxi):
        """argmax_{q_i in [0, upper(x_i)]} H_i. H_i is concave quadratic in q_i
        (d2H/dq_i^2 = -2), so the argmax is the clipped FOC:
            q_i* = (c x_i - sum_{j!=i} q_j - dV_i/dx_i) / 2
        """
        q = 0.5 * (self.c * x - q_others_sum - dVi_dxi)
        return torch.minimum(torch.clamp(q, min=0.0), self.action_upper(x))


# ---------------------------------------------------------------------------
# Closed-form reference (Salo & Tahvonen 2001, Prop. 3 + Appendix A)
# ---------------------------------------------------------------------------
def salo_coefficients(n: int, c: float, delta: float):
    """Symmetric Nash feedback law q_i = mu1 z_i + mu2 x_i, z_i = sum_{j!=i} x_j.
    Same formula as salo_tahvonen_feedback.ipynb."""
    gamma = delta / c
    a0 = 1.0 / np.sqrt((3 * n + 1) * (n - 1))
    a1 = a0 * np.sqrt(delta * (2 * c + delta))
    a2 = 0.5 * np.sqrt(delta * (2 * c + delta))
    p0 = -1 + gamma + 3 * n + gamma * n
    p1 = np.sqrt(16 + 4 * gamma + gamma**2
                 + (24 + 16 * gamma + 2 * gamma**2) * n
                 + (6 * gamma + gamma**2) * n**2)
    p2 = (44 + 30 * gamma + 6 * gamma**2 + gamma**3
          + (180 + 102 * gamma + 30 * gamma**2 + 3 * gamma**3) * n
          + (108 * gamma + 33 * gamma**2 + 3 * gamma**3) * n**2
          + (9 * gamma**2 + gamma**3) * n**3)
    s = (a0 / 3.0) * (p0 + 2 * p1 * np.cos((1 / 3) * np.arccos(p2 / p1**3)))
    mu1 = -a1 / np.sqrt(1 + s**2)
    mu2 = (a2 * s + 0.5 * a1 * (n - 1)) / np.sqrt(1 + s**2) - 0.5 * delta
    return float(mu1), float(mu2)


def salo_feedback_matrix(n: int, mu1: float, mu2: float) -> np.ndarray:
    """K such that q = K x (unclipped): K = mu2 I + mu1 (11^T - I)."""
    return mu2 * np.eye(n) + mu1 * (np.ones((n, n)) - np.eye(n))


def salo_value_matrices(n: int, c: float, delta: float) -> np.ndarray:
    """Equilibrium values V_i(x) = x^T M_i x, valid in the interior region
    where every q_i = (K x)_i > 0 (the linear-quadratic regime of the paper).

    Under q = K x the state follows xdot = A x with A = -K, and
        F_i = x^T B_i x,  B_i = c e_i k_i^T - (K^T 1) k_i^T   (k_i^T = row i of K),
    so M_i solves the Lyapunov equation  A^T M + M A - delta M = -sym(B_i).
    Returns an array of shape (n, n, n), M[i] = M_i.
    """
    mu1, mu2 = salo_coefficients(n, c, delta)
    K = salo_feedback_matrix(n, mu1, mu2)
    A = -K - 0.5 * delta * np.eye(n)       # A^T M + M A - delta M = A'^T M + M A'
    I = np.eye(n)
    # column-major vec: vec(A'^T M) = (I kron A'^T) vec M, vec(M A') = (A'^T kron I) vec M
    L = np.kron(I, A.T) + np.kron(A.T, I)
    ones = np.ones(n)
    M = np.zeros((n, n, n))
    for i in range(n):
        e_i = I[i]
        B = c * np.outer(e_i, K[i]) - np.outer(K.T @ ones, K[i])
        R = 0.5 * (B + B.T)
        vecM = np.linalg.solve(L, -R.flatten(order="F"))
        M[i] = vecM.reshape(n, n, order="F")
        M[i] = 0.5 * (M[i] + M[i].T)
    return M


class SaloReference:
    """Torch callables for Salo's equilibrium policy and value, used both as a
    ground truth for the DeepSet networks and to sanity-check the residual code
    (R_i and P_i must vanish on it)."""

    def __init__(self, game: SaloGame):
        self.game = game
        self.mu1, self.mu2 = salo_coefficients(game.n_players, game.c, game.delta)
        self.M = torch.as_tensor(
            salo_value_matrices(game.n_players, game.c, game.delta), dtype=torch.float32
        )

    def policy(self, x: torch.Tensor) -> torch.Tensor:
        z = x.sum(-1, keepdim=True) - x
        q = self.mu1 * z + self.mu2 * x
        return torch.minimum(torch.clamp(q, min=0.0), self.game.action_upper(x))

    def interior_mask(self, x: torch.Tensor) -> torch.Tensor:
        """States where every player's unclipped Salo extraction is > 0, i.e.
        where the quadratic value function is exact."""
        z = x.sum(-1, keepdim=True) - x
        return ((self.mu1 * z + self.mu2 * x) > 0).all(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...j,ijk,...k->...i", x, self.M.to(x), x)
