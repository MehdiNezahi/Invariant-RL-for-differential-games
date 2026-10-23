"""Single-agent environment used to train one player's best response.

Wraps the full N-player joint dynamics of a `GameSpec`, but only exposes
control of player `player_idx` to the caller: at every step, the other
players' actions are computed from their fixed policies against the current
joint state, so the *actual* game coupling (e.g. price depending on
everyone's extraction) is preserved -- this is not a naive single-agent
wrapper that ignores the externality.

Exposes the same (obs, reward, terminated, truncated, info) step contract as
gymnasium.Env so the HJB-PPO training loop from Petroleum_HJBPPO.ipynb can be
reused almost verbatim.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .game_spec import GameSpec


class BestResponseEnv:
    def __init__(
        self,
        game: GameSpec,
        policies: Sequence,
        player_idx: int,
        x0,
        deterministic_opponents: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        self.game = game
        self.policies = policies
        self.i = player_idx
        self.x0 = np.asarray(x0, dtype=float).reshape(game.n_players, game.state_dim)
        self.deterministic_opponents = deterministic_opponents
        self.rng = rng if rng is not None else np.random.default_rng()
        self.obs_dim = game.n_players * game.state_dim

        self.x = None
        self.step_count = 0
        self.reset()

    def reset(self):
        self.x = self.x0.copy()
        self.step_count = 0
        return self._obs()

    def _obs(self) -> np.ndarray:
        return self.x.reshape(-1).astype(np.float32)

    def step(self, action_i):
        game = self.game
        q = np.zeros((game.n_players, game.action_dim))

        for j in range(game.n_players):
            if j == self.i:
                q[j] = np.clip(action_i, game.action_low, game.action_high)
            else:
                own = self.x[j]
                others = np.delete(self.x, j, axis=0)
                a = self.policies[j](
                    own, others, deterministic=self.deterministic_opponents, rng=self.rng
                )
                q[j] = np.clip(a, game.action_low, game.action_high)

        r_density = np.asarray(game.reward_density(self.x, q), dtype=float)
        reward_i = float(r_density[self.i]) * game.dt

        self.x = game.dynamics(self.x, q)
        self.step_count += 1

        terminated = game.check_terminal(self.x)
        truncated = self.step_count >= game.max_steps
        return self._obs(), reward_i, terminated, truncated, {}
