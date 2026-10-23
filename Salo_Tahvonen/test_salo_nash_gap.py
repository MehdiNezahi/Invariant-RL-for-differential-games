"""Standalone test: compute the Nash gap of Salo & Tahvonen's theoretical
feedback equilibrium, using the Nash_Gap module.

Since (mu1, mu2) IS the closed-form Nash equilibrium of the symmetric
oligopoly, the expected result is nash_gap ~= 0 (up to PPO optimization
error) -- a strictly positive gap would mean the best-response training
found a profitable deviation, which should not happen at a true equilibrium.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Nash_Gap/ lives in the parent dir

from Nash_Gap import GameSpec, PPOConfig, compute_nash_gap, make_linear_feedback_policy

# --- Salo & Tahvonen (2001) symmetric oligopoly, same params as salo_tahvonen_feedback.ipynb ---
N_PLAYERS = 3
COST_SLOPE = 0.5      # c
DISCOUNT_RATE = 0.05  # delta
CHOKE_PRICE = 10.0    # pbar
X0 = np.array([8.0, 5.0, 4.0])
DT = 0.1

# --- Best-response training budget, both expressed directly in time steps. ---
MAX_STEPS = 2000       # rollout horizon per episode
PPO_EPISODES = 1000
LOG_EVERY = 10         # print progress every LOG_EVERY episodes


def theoretical_solution(n, c, delta):
    """Closed-form Nash feedback coefficients (Prop. 3), copied from
    salo_tahvonen_feedback.ipynb."""
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
    return mu1, mu2


def dynamics(x, q):
    return np.maximum(x - q * DT, 0.0)


def reward_density(x, q):
    x = x.reshape(-1)
    q = q.reshape(-1)
    price = CHOKE_PRICE - q.sum()
    unit_cost = CHOKE_PRICE - COST_SLOPE * x
    return (price - unit_cost) * q


def main():
    mu1, mu2 = theoretical_solution(N_PLAYERS, COST_SLOPE, DISCOUNT_RATE)
    print(f"Theoretical equilibrium: mu1={mu1:.6f}  mu2={mu2:.6f}")

    policies = [
        make_linear_feedback_policy(mu1, mu2, action_low=0.0, action_high=CHOKE_PRICE)
        for _ in range(N_PLAYERS)
    ]

    game = GameSpec(
        n_players=N_PLAYERS,
        dt=DT,
        discount_rate=DISCOUNT_RATE,
        dynamics=dynamics,
        reward_density=reward_density,
        action_low=0.0,
        action_high=CHOKE_PRICE,
        max_steps=MAX_STEPS,
    )

    ppo_config = PPOConfig(episodes=PPO_EPISODES)

    result = compute_nash_gap(
        game, policies, X0, ppo_config=ppo_config, verbose=True, log_every=LOG_EVERY,
    )

    print("\n=== Nash gap of Salo & Tahvonen's theoretical equilibrium ===")
    for r in result.per_player:
        print(f"player {r.player}: J_policy={r.J_policy:.4f}  J_best_response={r.J_best_response:.4f}  gap={r.gap:.4f}")
    print(f"\nnash_gap (mean over players) = {result.nash_gap:.4f}")


if __name__ == "__main__":
    main()
