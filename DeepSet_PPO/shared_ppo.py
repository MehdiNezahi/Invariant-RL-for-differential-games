"""One shared DeepSet actor/critic trained by PPO on the N-player game.

Every simulated step of E parallel games yields E x N player-samples for the
same networks. Each sample uses its own reward F_i and its own log-prob, so
the PPO gradient is the partial gradient of J_i w.r.t. player i's control
(the standard PPO estimator, nothing special to add).

Losses
  actor : PPO-clip  + lambda_P * (1/delta) mean (q_hat_i - q_i*)^2  - ent * H[pi]
          with q_i* = argmax H_i built from the critic -- DETACHED target
  critic: MSE(V, GAE returns) + lambda_R * mean((R_i / delta)^2)

R_i and P_i are scaled by 1/delta as in the Nash-gap bound, so both extra
terms are in value units. R_i uses the deterministic profile pi_hat (Beta
mean), which is the profile the bound and the evaluation are about.

Episodes never terminate (depletion is absorbing with zero reward): the end of
an episode is always a truncation, bootstrapped with V(x_{t+1}).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .deepset import DeepSetActor, DeepSetCritic, own_gradient
from .residuals import hjb_residual


@dataclass
class SharedPPOConfig:
    # environment sampling
    n_players_choices: tuple = (3,)   # N drawn per iteration (DeepSet: any N)
    n_envs: int = 64
    rollout_len: int = 256
    episode_len: int = 2000           # t = 200 -> e^{-delta t} ~ 4.5e-5
    x0_low: float = 0.5
    x0_high: float = 10.0
    # optimisation
    iterations: int = 500
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 2048        # in joint states (each holds N player-samples)
    max_grad_norm: float = 0.5
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 1e-3
    # residual terms
    lambda_hjb: float = 1.0
    lambda_mismatch: float = 1.0
    mismatch_warmup: int = 50         # iterations with lambda_P = 0 (critic not yet meaningful)
    # networks
    hidden: int = 64
    x_scale: float = 10.0
    seed: int = 0


class SharedPPOTrainer:
    def __init__(self, game, config: SharedPPOConfig, actor=None, critic=None):
        self.game = game
        self.cfg = config
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.actor = actor or DeepSetActor(config.hidden, config.x_scale)
        self.critic = critic or DeepSetCritic(config.hidden, config.x_scale)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.x = None
        self.ep_steps = None
        self.iteration = 0
        self.history = []

    # ------------------------------------------------------------------ envs
    def _sample_x0(self, n_envs: int, n: int) -> torch.Tensor:
        c = self.cfg
        return c.x0_low + (c.x0_high - c.x0_low) * torch.rand(n_envs, n)

    def _reset_all(self, n: int):
        self.x = self._sample_x0(self.cfg.n_envs, n)
        # staggered episode clocks so resets are spread over time
        self.ep_steps = torch.randint(0, self.cfg.episode_len, (self.cfg.n_envs,))

    # --------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self, game) -> dict:
        c, n = self.cfg, game.n_players
        if self.x is None or self.x.shape[1] != n:
            self._reset_all(n)

        T, E = c.rollout_len, c.n_envs
        buf = {k: torch.zeros(T, E, n) for k in ("x", "u", "logp", "r", "v", "v_next")}
        buf["trunc"] = torch.zeros(T, E)

        x = self.x
        for t in range(T):
            dist = self.actor(x)
            u = dist.sample().clamp(1e-6, 1 - 1e-6)
            q = u * game.action_upper(x)
            r = game.reward_density(x, q) * game.dt
            x_next = game.step(x, q)

            buf["x"][t], buf["u"][t], buf["logp"][t] = x, u, dist.log_prob(u)
            buf["r"][t], buf["v"][t] = r, self.critic(x)
            buf["v_next"][t] = self.critic(x_next)      # before any reset: truncation bootstrap

            self.ep_steps += 1
            trunc = self.ep_steps >= c.episode_len
            buf["trunc"][t] = trunc.float()
            if trunc.any():
                x_next = x_next.clone()
                x_next[trunc] = self._sample_x0(int(trunc.sum()), n)
                self.ep_steps[trunc] = 0
            x = x_next
        self.x = x

        # GAE; trunc cuts the recursion but the bootstrap V(x_{t+1}) is kept
        adv = torch.zeros(T, E, n)
        gae = torch.zeros(E, n)
        lam_gamma = c.gae_lambda * game.gamma
        for t in reversed(range(T)):
            delta_t = buf["r"][t] + game.gamma * buf["v_next"][t] - buf["v"][t]
            gae = delta_t + lam_gamma * (1 - buf["trunc"][t]).unsqueeze(-1) * gae
            adv[t] = gae
        buf["adv"] = adv
        buf["ret"] = adv + buf["v"]
        return buf

    # ---------------------------------------------------------------- update
    def update(self, game, buf: dict) -> dict:
        c = self.cfg
        n = game.n_players
        flat = {k: buf[k].reshape(-1, n) for k in ("x", "u", "logp", "adv", "ret")}
        adv = flat["adv"]
        flat["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = flat["x"].shape[0]
        lam_P = c.lambda_mismatch if self.iteration >= c.mismatch_warmup else 0.0

        stats = {k: [] for k in ("actor_loss", "entropy", "mismatch", "value_mse", "hjb")}
        for _ in range(c.update_epochs):
            perm = torch.randperm(B)
            for start in range(0, B, c.minibatch_size):
                mb = perm[start:start + c.minibatch_size]
                xb = flat["x"][mb]

                # ---- actor: PPO on each player's own log-prob / advantage
                dist = self.actor(xb)
                ratio = torch.exp(dist.log_prob(flat["u"][mb]) - flat["logp"][mb])
                a = flat["adv"][mb]
                ppo = -torch.min(ratio * a, ratio.clamp(1 - c.clip_eps, 1 + c.clip_eps) * a).mean()
                ent = dist.entropy().mean()

                q_hat = dist.mean * game.action_upper(xb)
                if lam_P > 0:
                    dVi = own_gradient(self.critic, xb).detach()
                    q_others = (q_hat.sum(-1, keepdim=True) - q_hat).detach()
                    q_star = game.hamiltonian_argmax(xb, q_others, dVi).detach()
                    mismatch = ((q_hat - q_star) ** 2).mean() / game.delta
                else:
                    mismatch = torch.zeros(())

                actor_loss = ppo + lam_P * mismatch - c.entropy_coef * ent
                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
                self.actor_opt.step()

                # ---- critic: returns + HJB residual on the deterministic profile
                v = self.critic(xb)
                value_mse = ((flat["ret"][mb] - v) ** 2).mean()
                q_bar = q_hat.detach()
                R = hjb_residual(game, self.critic, xb, q_bar)
                hjb = ((R / game.delta) ** 2).mean()

                critic_loss = value_mse + c.lambda_hjb * hjb
                self.critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                self.critic_opt.step()

                stats["actor_loss"].append(ppo.item())
                stats["entropy"].append(ent.item())
                stats["mismatch"].append(mismatch.item())
                stats["value_mse"].append(value_mse.item())
                stats["hjb"].append(hjb.item())
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------ loop
    def train(self, iterations: int | None = None, eval_fn=None, eval_every: int = 25, log_every: int = 10):
        """`eval_fn(trainer) -> dict` (e.g. evaluation.make_eval_fn) is called
        every `eval_every` iterations; its entries are merged into history."""
        iterations = iterations or self.cfg.iterations
        t0 = time.time()
        for _ in range(iterations):
            n = int(self.rng.choice(self.cfg.n_players_choices))
            game = self.game.with_n(n)
            buf = self.collect(game)
            stats = self.update(game, buf)
            stats.update(iteration=self.iteration, n_players=n,
                         reward_rate=buf["r"].mean().item() / game.dt)

            if eval_fn is not None and (self.iteration % eval_every == 0 or self.iteration == iterations - 1):
                stats.update(eval_fn(self))
            self.history.append(stats)

            if self.iteration % log_every == 0:
                msg = (f"it {self.iteration:>4} N={n} | ppo {stats['actor_loss']:+.3f} "
                       f"ent {stats['entropy']:+.2f} | vMSE {stats['value_mse']:.3e} "
                       f"hjb {stats['hjb']:.3e} mis {stats['mismatch']:.3e} | {time.time() - t0:.0f}s")
                if "J_rl" in stats:
                    msg += ("\n        J_rl " + np.array2string(np.asarray(stats["J_rl"]), precision=3)
                            + " J_salo " + np.array2string(np.asarray(stats["J_salo"]), precision=3)
                            + f" | mu_fit ({stats['mu1_fit']:+.4f}, {stats['mu2_fit']:+.4f})"
                            + f" | bound {stats['bound']:.3f}")
                print(msg)
            self.iteration += 1
        return self.history

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "config": asdict(self.cfg), "game": asdict(self.game),
                    "history": self.history}, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.history = ckpt.get("history", [])
        self.iteration = len(self.history)
