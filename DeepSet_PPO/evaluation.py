"""Evaluation of the shared DeepSet profile against Salo & Tahvonen.

- rollout(): deterministic joint rollout from a given x0, discounted J_i
- fit_linear_feedback(): least-squares (mu1, mu2) of q_hat = mu1 z + mu2 x
- evaluate(): J vs Salo at x0, policy / value errors on interior states,
  residuals R_i, P_i and the Nash-gap bound
- make_nash_gap_policy() / salo_gamespec(): adapters to the Nash_Gap package
  (empirical gap by best-response training)
"""
from __future__ import annotations

import numpy as np
import torch

from .deepset import own_gradient
from .residuals import hamiltonian_gap, hjb_residual, nash_gap_bound
from .salo_game import SaloReference


@torch.no_grad()
def rollout(game, policy_fn, x0, steps: int = 2000):
    """policy_fn(x (B, N)) -> q (B, N). Returns J (N,), X (steps+1, N), Q (steps, N).
    Left Riemann sum with weights e^{-delta t}, as in salo_tahvonen_feedback.ipynb."""
    x = torch.as_tensor(np.asarray(x0, dtype=np.float32)).reshape(1, -1)
    J = torch.zeros(x.shape[1])
    X, Q = [x[0].clone()], []
    disc = 1.0
    for _ in range(steps):
        q = policy_fn(x)
        J += disc * game.reward_density(x, q)[0] * game.dt
        x = game.step(x, q)
        disc *= game.gamma
        X.append(x[0].clone())
        Q.append(q[0].clone())
    return J.numpy(), torch.stack(X).numpy(), torch.stack(Q).numpy()


def deepset_policy_fn(actor, game):
    return lambda x: actor.mean_action(x, game)


@torch.no_grad()
def fit_linear_feedback(q: torch.Tensor, x: torch.Tensor, min_q: float = 1e-3):
    """Least squares q_i = mu1 z_i + mu2 x_i on samples with q_i > min_q."""
    z = x.sum(-1, keepdim=True) - x
    m = q > min_q
    A = torch.stack([z[m], x[m]], dim=-1).double()
    sol = torch.linalg.lstsq(A, q[m].double().unsqueeze(-1)).solution.squeeze(-1)
    return float(sol[0]), float(sol[1])


def sample_states(game, n_samples: int, low: float, high: float, generator=None):
    return low + (high - low) * torch.rand(n_samples, game.n_players, generator=generator)


def evaluate(actor, critic, game, x0=None, steps: int = 2000, n_samples: int = 4096,
             low: float = 0.5, high: float = 10.0, seed: int = 123) -> dict:
    ref = SaloReference(game)
    gen = torch.Generator().manual_seed(seed)
    out = {"mu1_salo": ref.mu1, "mu2_salo": ref.mu2}

    if x0 is not None:
        J_rl, _, _ = rollout(game, deepset_policy_fn(actor, game), x0, steps)
        J_salo, _, _ = rollout(game, ref.policy, x0, steps)
        out.update(J_rl=J_rl.tolist(), J_salo=J_salo.tolist(),
                   J_rel_err=float(np.abs(J_rl - J_salo).sum() / np.abs(J_salo).sum()))

    x = sample_states(game, n_samples, low, high, gen)
    with torch.no_grad():
        q_hat = actor.mean_action(x, game)
        V_hat = critic(x)
    out["mu1_fit"], out["mu2_fit"] = fit_linear_feedback(q_hat, x)

    inside = ref.interior_mask(x)
    out["interior_frac"] = inside.float().mean().item()
    if inside.any():
        xi = x[inside]
        out["policy_rmse"] = (q_hat[inside] - ref.policy(xi)).pow(2).mean().sqrt().item()
        V_ref = ref.value(xi)
        out["value_rel_err"] = ((V_hat[inside] - V_ref).abs().mean() / V_ref.abs().mean()).item()

    # residuals of the learned pair (V_hat, pi_hat) on all sampled states
    R = hjb_residual(game, critic, x, q_hat).detach()
    P, _ = hamiltonian_gap(game, x, q_hat, own_gradient(critic, x).detach())
    out.update(nash_gap_bound(R, P.detach(), game.delta))
    return out


def make_eval_fn(x0, steps: int = 2000, n_samples: int = 4096, low: float = 0.5, high: float = 10.0):
    """Eval hook for SharedPPOTrainer.train: evaluated on the base game (its N)."""
    def eval_fn(trainer):
        return evaluate(trainer.actor, trainer.critic, trainer.game, x0, steps, n_samples, low, high)
    return eval_fn


# ---------------------------------------------------------------------------
# Adapters to the Nash_Gap package (empirical gap by best-response training)
# ---------------------------------------------------------------------------
def make_nash_gap_policy(actor, game):
    """Policy with the Nash_Gap contract policy(own, others, deterministic, rng).
    Player order inside the joint state is irrelevant (equivariance)."""
    def policy(own_state, other_states, deterministic=True, rng=None):
        x = np.concatenate([np.asarray(own_state, float).reshape(-1),
                            np.asarray(other_states, float).reshape(-1)])
        xt = torch.as_tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = actor(xt)
            u = dist.mean if deterministic else dist.sample()
            q = u * game.action_upper(xt)
        return q[0, :1].numpy()
    return policy


def salo_gamespec(game, max_steps: int = 2000):
    """numpy GameSpec with the same dynamics / feasibility as SaloGame."""
    from Nash_Gap import GameSpec

    def feasible(x, q):
        return np.minimum(q.reshape(-1), x.reshape(-1) / game.dt)

    def dynamics(x, q):
        x = x.reshape(-1)
        return np.maximum(x - feasible(x, q) * game.dt, 0.0).reshape(-1, 1)

    def reward_density(x, q):
        x = x.reshape(-1)
        q = feasible(x, q)
        return (game.c * x - q.sum()) * q

    return GameSpec(n_players=game.n_players, dt=game.dt, discount_rate=game.delta,
                    dynamics=dynamics, reward_density=reward_density,
                    action_low=0.0, action_high=game.q_max, max_steps=max_steps)
