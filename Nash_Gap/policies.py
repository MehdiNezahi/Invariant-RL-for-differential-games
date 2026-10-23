"""Unified policy contract + adapters.

Nash_Gap only ever calls a policy as:

    action = policy(own_state, other_states, deterministic=..., rng=...)

`own_state`   : this player's own state, shape (state_dim,)
`other_states`: the other N-1 players' states, shape (n_players-1, state_dim)
`deterministic`: if True, return the policy's mean/greedy action instead of sampling
`rng`         : numpy Generator, for adapters that need randomness themselves

Both Salo's closed-form feedback law and a MARL DeepSet actor are wrapped
into this same signature so `Nash_Gap.compute_nash_gap` never needs to know
which one it's dealing with.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

Policy = Callable[..., np.ndarray]


def make_linear_feedback_policy(mu1: float, mu2: float, action_low: float = 0.0, action_high: float = np.inf) -> Policy:
    """Adapter for Salo & Tahvonen's symmetric closed-form feedback law:
    q_i = mu1 * z_i + mu2 * x_i, with z_i = sum_{j != i} x_j.
    """

    def policy(own_state, other_states, deterministic: bool = True, rng: Optional[np.random.Generator] = None):
        own = np.asarray(own_state, dtype=float).reshape(-1)[0]
        z = np.asarray(other_states, dtype=float).sum()
        q = mu1 * z + mu2 * own
        return np.clip(np.array([q]), action_low, action_high)

    return policy


def make_torch_policy(
    actor,
    obs_transform: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    action_low: float = 0.0,
    action_high: float = 1.0,
) -> Policy:
    """Adapter for a torch policy network (e.g. the DeepSet actor) whose
    `forward(obs) -> torch.distributions.Distribution` on some observation
    built from (own_state, other_states).

    `obs_transform(own_state, other_states) -> np.ndarray` lets you plug in
    whatever input layout the DeepSet actor expects (e.g. own state
    concatenated with the permutation-invariant aggregate of the others).
    Defaults to a flat concatenation if not provided.
    """
    import torch

    def policy(own_state, other_states, deterministic: bool = True, rng: Optional[np.random.Generator] = None):
        own = np.asarray(own_state, dtype=float).reshape(-1)
        others = np.asarray(other_states, dtype=float)
        if obs_transform is not None:
            obs = obs_transform(own, others)
        else:
            obs = np.concatenate([own, others.reshape(-1)])

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = actor(obs_t)
            u = dist.mean if deterministic else dist.sample()
        u = np.clip(u.reshape(-1).cpu().numpy(), 0.0, 1.0)
        return action_low + (action_high - action_low) * u

    return policy
