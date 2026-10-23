"""Nash gap of a feedback policy profile, estimated via best-response training.

For a policy profile `policies` (one callable per player, see
`policies.py`) on a game `game`, and for every player i:
  1. freeze policies[j], j != i
  2. train a best response for player i by HJB-regularized PPO
     (BestResponseEnv + HJBPPOAgent)
  3. compare its discounted return to the return player i actually gets
     under the full profile (J_policy[i])

gap_i = J_best_response_i - J_policy_i  (>= 0 at a true Nash equilibrium, up
to optimization error -- a positive gap means player i can do strictly
better by deviating, i.e. `policies` is not an equilibrium).

The profile-level summary reported is the mean gap over players,
nash_gap = (1/n) sum_i gap_i, so it stays comparable across games with
different numbers of players.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .br_env import BestResponseEnv
from .game_spec import GameSpec, simulate_joint_policy
from .ppo_hjb import HJBPPOAgent, PPOConfig, evaluate_deterministic, train_hjb_ppo


@dataclass
class PlayerGapResult:
    player: int
    J_policy: float
    J_best_response: float
    gap: float
    reward_history: list = field(default_factory=list)


@dataclass
class NashGapResult:
    per_player: List[PlayerGapResult]
    nash_gap: float  # mean_i gap_i


def compute_nash_gap(
    game: GameSpec,
    policies: Sequence,
    x0,
    ppo_config: Optional[PPOConfig] = None,
    J_policy: Optional[np.ndarray] = None,
    verbose: bool = False,
    log_every: Optional[int] = None,
) -> NashGapResult:
    """
    `J_policy`, if given, is the (n_players,) array of discounted returns
    already achieved by `policies` (e.g. computed once from Salo's own `X`,
    `Q` trajectories via `discounted_reward_from_trajectory`) -- pass it to
    skip re-simulating the joint rollout.

    `log_every`, with `verbose=True`, controls how often (in episodes)
    best-response training progress is printed, and prints the live gap
    against `J_policy[i]` alongside it.
    """
    ppo_config = ppo_config or PPOConfig()
    n = game.n_players

    if J_policy is None:
        J_policy, _, _ = simulate_joint_policy(game, policies, x0, deterministic=True)
    J_policy = np.asarray(J_policy, dtype=float)

    results = []
    for i in range(n):
        if verbose:
            print(f"[Nash_Gap] training best response for player {i}...")

        env = BestResponseEnv(game, policies, player_idx=i, x0=x0)
        agent = HJBPPOAgent(
            obs_dim=env.obs_dim,
            config=ppo_config,
            action_low=game.action_low,
            action_high=game.action_high,
        )
        reward_history = train_hjb_ppo(
            env, agent, game, ppo_config,
            verbose=verbose, log_every=log_every, baseline=float(J_policy[i]),
        )
        J_br = evaluate_deterministic(env, agent, game)

        gap_i = J_br - float(J_policy[i])
        results.append(
            PlayerGapResult(
                player=i,
                J_policy=float(J_policy[i]),
                J_best_response=J_br,
                gap=gap_i,
                reward_history=reward_history,
            )
        )
        if verbose:
            print(f"[Nash_Gap] player {i}: J_policy={J_policy[i]:.4f}  J_BR={J_br:.4f}  gap={gap_i:.4f}")

    gaps = np.array([r.gap for r in results])
    return NashGapResult(per_player=results, nash_gap=float(gaps.mean()))
