"""HJB-regularized PPO, generalized from Petroleum_HJBPPO.ipynb.

Same algorithm (Beta-policy PPO with a critic trained on both Monte-Carlo
returns and an HJB residual), but:
  - obs_dim is arbitrary (was hardcoded to 1) -- needed since a best-response
    agent observes the full joint state (n_players * state_dim), not just its
    own stock.
  - action bounds are configurable (was hardcoded to [0, 1]).
  - hyperparameters are passed explicitly via `PPOConfig` instead of module
    globals, since Nash_Gap trains many independent best responses (one per
    player) that must not share mutable global state.

Assumes a scalar control per player (action_dim=1), matching both
Salo-Tahvonen and the Petroleum toy game -- see GameSpec.__post_init__.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad

from .game_spec import GameSpec


@dataclass
class PPOConfig:
    rollout_len: int = 256
    episodes: int = 5000
    clip_eps: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 1024
    max_grad_norm: float = 0.5
    lambda_hjb: float = 0.1
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5


class BetaActor(nn.Module):
    def __init__(self, obs_dim: int, action_low: float = 0.0, action_high: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.softplus = nn.Softplus()
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, x):
        alpha, beta = self.net(x).unbind(-1)
        alpha = self.softplus(alpha) + 1e-4
        beta = self.softplus(beta) + 1e-4
        return torch.distributions.Beta(alpha, beta)

    def to_action(self, u):
        return self.action_low + (self.action_high - self.action_low) * u


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.obs = torch.zeros(size + 1, obs_dim)
        self.actions = torch.zeros(size)
        self.logprobs = torch.zeros(size)
        self.rewards = torch.zeros(size)
        self.dones = torch.zeros(size)
        self.values = torch.zeros(size + 1)
        self.ptr = 0
        self.max_size = size

    def add(self, obs, action, logp, reward, done, value):
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32)
        self.actions[i] = action
        self.logprobs[i] = logp
        self.rewards[i] = reward
        self.dones[i] = done
        self.values[i] = value
        self.ptr += 1

    def bootstrap(self, last_value, last_obs):
        self.values[self.ptr] = last_value
        self.obs[self.ptr] = torch.as_tensor(last_obs, dtype=torch.float32)

    def reset(self):
        self.ptr = 0


def compute_hjb_batch_loss(states, next_states, rewards, critic, gamma, dt):
    """HJB residual: V(x) log(gamma) + R_density(x) + grad_x V(x) . f(x) ~= 0,
    with f(x) approximated by the observed finite difference (x_{t+1}-x_t)/dt.
    Generalized to a vector state via a dot product over the last dim (was a
    scalar product when obs_dim was hardcoded to 1).
    """
    R_density = rewards.detach() / dt

    states = states.detach().requires_grad_(True)
    next_states = next_states.detach()

    v_pred = critic(states)
    grads = grad(v_pred.sum(), states, retain_graph=True)[0]
    f_approx = (next_states - states) / dt
    directional = (grads * f_approx).sum(dim=-1)

    # log(gamma)/dt ~= -r: the discount term must be a rate, like R_density and
    # grad V . f (without /dt the game's discount was effectively r*dt).
    resid = v_pred * torch.log(torch.tensor(gamma)) / dt + R_density + directional
    return (resid ** 2).mean()


def ppo_clip_loss(dist, actions, old_logp, advantages, clip_eps):
    new_logp = dist.log_prob(actions)
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    actor_loss = -torch.min(unclipped, clipped).mean()
    entropy = dist.entropy().mean()
    return actor_loss, entropy


class HJBPPOAgent:
    def __init__(self, obs_dim: int, config: PPOConfig, action_low: float = 0.0, action_high: float = 1.0):
        self.actor = BetaActor(obs_dim, action_low, action_high)
        self.critic = Critic(obs_dim)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.buffer = RolloutBuffer(config.rollout_len, obs_dim)

    def select_action(self, obs, noise: bool = True):
        """Returns (raw_u, scaled_action, logp, value).
        raw_u is the Beta-space sample (needed for PPO's log-prob ratio),
        scaled_action is what should actually be passed to env.step.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        dist = self.actor(obs_t)
        value = self.critic(obs_t).item()
        u = dist.sample() if noise else dist.mean
        logp = dist.log_prob(u).item()
        scaled_action = self.actor.to_action(u).item()
        return u.item(), scaled_action, logp, value

    def store_transition(self, obs, raw_action, logp, reward, done, value):
        self.buffer.add(obs, raw_action, logp, reward, done, value)

    def update(self, last_obs, game: GameSpec, config: PPOConfig):
        with torch.no_grad():
            last_value = self.critic(torch.as_tensor(last_obs, dtype=torch.float32)).item()
        self.buffer.bootstrap(last_value, last_obs)

        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        returns = torch.zeros_like(rewards)
        acc = last_value
        for t in reversed(range(len(rewards))):
            acc = rewards[t] + game.gamma * acc * (1 - dones[t])
            returns[t] = acc

        advantages = returns - values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = self.buffer.obs[:-1]
        actions = self.buffer.actions
        old_logp = self.buffer.logprobs
        rollout_len = config.rollout_len

        for _ in range(config.update_epochs):
            idx = torch.randperm(rollout_len)
            for start in range(0, rollout_len, config.minibatch_size):
                mb = idx[start:start + config.minibatch_size]

                dist = self.actor(obs[mb])
                a_loss, ent = ppo_clip_loss(dist, actions[mb], old_logp[mb], advantages[mb], config.clip_eps)

                v_pred = self.critic(obs[mb])
                mse_v = (returns[mb] - v_pred).pow(2).mean()

                nb_states = obs[mb]
                nb_next = self.buffer.obs[mb + 1]
                nb_rewards = self.buffer.rewards[mb]
                hjb_loss = compute_hjb_batch_loss(nb_states, nb_next, nb_rewards, self.critic, game.gamma, game.dt)

                critic_loss = 0.5 * mse_v + config.lambda_hjb * hjb_loss
                total_loss = a_loss + config.value_coef * critic_loss - config.entropy_coef * ent

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), config.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

        self.buffer.reset()

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


def train_hjb_ppo(
    env,
    agent: HJBPPOAgent,
    game: GameSpec,
    config: PPOConfig,
    verbose: bool = False,
    log_every: int = None,
    baseline: float = None,
):
    """Runs the same episode/update loop as the 'Train the agent' cell in
    Petroleum_HJBPPO.ipynb, against any env exposing reset()/step() with the
    (obs, reward, terminated, truncated, info) contract (e.g. BestResponseEnv).

    `log_every` controls how often (in episodes) progress is printed when
    `verbose=True` -- defaults to ~20 prints over the whole run if not given.
    `baseline`, if given (e.g. J_policy[i] from Nash_Gap), is printed
    alongside the running episode return as `gap = episode_return - baseline`,
    so you can watch the best response close in on (or overtake) the policy's
    own reward as training progresses.
    """
    reward_history = []
    obs = env.reset()
    episode_discounted = 0.0
    step_count = 0
    episode_idx = 0

    while episode_idx < config.episodes:
        raw_u, scaled_action, logp, value = agent.select_action(obs, noise=True)
        next_obs, reward, terminated, truncated, _ = env.step(scaled_action)
        done = terminated or truncated

        agent.store_transition(obs, raw_u, logp, reward, done, value)
        obs = next_obs

        episode_discounted += (game.gamma ** step_count) * reward
        step_count += 1

        if agent.buffer.ptr >= config.rollout_len:
            agent.update(obs, game, config)

        if done:
            episode_idx += 1
            reward_history.append(episode_discounted)
            every = log_every if log_every is not None else max(1, config.episodes // 20)
            if verbose and episode_idx % every == 0:
                msg = f"  episode {episode_idx:>5} | discounted reward {episode_discounted:.4f}"
                if baseline is not None:
                    msg += f" | J_policy={baseline:.4f} | gap={episode_discounted - baseline:.4f}"
                print(msg)
            episode_discounted = 0.0
            step_count = 0
            obs = env.reset()

    return reward_history


def evaluate_deterministic(env, agent: HJBPPOAgent, game: GameSpec) -> float:
    """Mirrors the 'Noiseless Reward Simulation' cell: rolls out the
    deterministic (mean) policy and returns its discounted reward."""
    obs = env.reset()
    done = False
    discounted = 0.0
    step_count = 0
    while not done:
        _, scaled_action, _, _ = agent.select_action(obs, noise=False)
        obs, reward, terminated, truncated, _ = env.step(scaled_action)
        done = terminated or truncated
        discounted += (game.gamma ** step_count) * reward
        step_count += 1
    return discounted
