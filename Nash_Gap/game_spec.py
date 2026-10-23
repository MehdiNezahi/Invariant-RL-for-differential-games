"""Generic description of an N-player differential game.

A `GameSpec` only needs the dynamics and the instantaneous reward density of
the underlying game (common-pool depletion, or anything else). It knows
nothing about Salo's closed-form feedback law or about any RL policy network
-- those are plugged in from the outside as `Policy` callables (see
`policies.py`), which is what keeps this module reusable both for
Salo-Tahvonen and for the MARL/DeepSet training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

DynamicsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
# dynamics(x, q) -> x_next   with x, q of shape (n_players, state_dim) / (n_players, action_dim)

RewardDensityFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
# reward_density(x, q) -> (n_players,) instantaneous reward flux (NOT yet multiplied by dt)

TerminationFn = Callable[[np.ndarray], bool]
# is_terminal(x) -> True if the episode should stop before max_steps (e.g. full depletion)


@dataclass
class GameSpec:
    n_players: int
    dt: float
    discount_rate: float
    dynamics: DynamicsFn
    reward_density: RewardDensityFn
    state_dim: int = 1
    action_dim: int = 1
    action_low: float = 0.0
    action_high: float = 1.0
    max_steps: int = 200
    is_terminal: Optional[TerminationFn] = None

    def __post_init__(self):
        if self.action_dim != 1:
            raise NotImplementedError(
                "Nash_Gap currently assumes a scalar control per player "
                "(action_dim=1), matching Salo/Petroleum. Extend BetaActor "
                "for vector actions if that's ever needed."
            )

    @property
    def gamma(self) -> float:
        """Per-step discount factor consistent with a continuous rate `discount_rate`."""
        return 1.0 / (1.0 + self.discount_rate * self.dt)

    def check_terminal(self, x: np.ndarray) -> bool:
        return bool(self.is_terminal(x)) if self.is_terminal is not None else False


def _as_state(x, n_players: int, state_dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x.reshape(n_players, state_dim)


def simulate_joint_policy(
    game: GameSpec,
    policies: Sequence,
    x0,
    deterministic: bool = True,
    rng: Optional[np.random.Generator] = None,
):
    """Roll the full policy profile forward (every player plays its own fixed
    policy) and return each player's discounted reward, plus the state/action
    trajectories.

    `policies[i]` must follow the unified contract from `policies.py`:
    `policy(own_state, other_states, deterministic, rng) -> action`.
    """
    n, sd = game.n_players, game.state_dim
    x = _as_state(x0, n, sd)
    rng = rng if rng is not None else np.random.default_rng()

    J = np.zeros(n)
    gamma_k = 1.0
    states = [x.copy()]
    actions = []

    for _ in range(game.max_steps):
        q = np.zeros((n, game.action_dim))
        for i in range(n):
            own = x[i]
            others = np.delete(x, i, axis=0)
            a = policies[i](own, others, deterministic=deterministic, rng=rng)
            q[i] = np.clip(a, game.action_low, game.action_high)

        r_density = np.asarray(game.reward_density(x, q), dtype=float)
        J += gamma_k * r_density * game.dt

        x = game.dynamics(x, q)
        actions.append(q.copy())
        states.append(x.copy())
        gamma_k *= game.gamma

        if game.check_terminal(x):
            break

    return J, np.array(states), np.array(actions)


def discounted_reward_from_trajectory(game: GameSpec, states, actions) -> np.ndarray:
    """Integrate the discounted reward of every player from an ALREADY
    COMPUTED trajectory (e.g. Salo's own `X`, `Q` arrays, or a logged MARL
    rollout) instead of re-simulating it.

    states:  (T+1, n_players[, state_dim])
    actions: (T,   n_players[, action_dim])
    """
    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if states.ndim == 2:
        states = states[..., None]
    if actions.ndim == 2:
        actions = actions[..., None]

    T = actions.shape[0]
    J = np.zeros(game.n_players)
    gamma_k = 1.0
    for k in range(T):
        r_density = np.asarray(game.reward_density(states[k], actions[k]), dtype=float)
        J += gamma_k * r_density * game.dt
        gamma_k *= game.gamma
    return J
