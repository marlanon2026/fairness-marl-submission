# utils/fen_reward_handler.py

import os
import numpy as np
from collections import deque

class FenRewardHandler:
    """
    Pure FEN reward shaping:
      - Per-step workload balance only (no skill-task alignment shaping).
      - Optional 'delta mode' (reward changes in fairness, not level).
      - Team fairness bonus is distributed equally across agents.
    Env vars:
      FAIRNESS_METRIC=inv_cv|gini
      FAIR_REW=0.8
      ZEROSHIFT=1.0
      CLIP_FAIR=1.0
      T_WINDOW=25
      LAMBDA_FAIRNESS=30
      EPS_BAR=1e-6
      FEN_DELTA=1 (use delta of fairness)
      FEN_DEBUG=0/1 (prints)
    """

    def __init__(self, n_agents: int):
        self.n_agents = n_agents

        # Core knobs
        self.metric = os.getenv("FAIRNESS_METRIC", "inv_cv")
        self.fair_rew = float(os.getenv("FAIR_REW", "0.8"))
        self.zeroshift = float(os.getenv("ZEROSHIFT", "1.0"))
        self.clip_fair = float(os.getenv("CLIP_FAIR", "1.0"))
        self.t_window = int(os.getenv("T_WINDOW", "25"))
        self.lmbda = float(os.getenv("LAMBDA_FAIRNESS", "30"))

        # Numerical guards
        self.eps = 1e-8
        self.u_bar_floor = float(os.getenv("EPS_BAR", "1e-6"))

        # Delta mode + debug
        self.use_delta = bool(int(os.getenv("FEN_DELTA", "1")))  # default ON
        self.prev_fair_core = 0.0
        self.debug = bool(int(os.getenv("FEN_DEBUG", "0")))
        self._t = 0  # local step counter for logging

        # Sliding per-agent utilities for L1/L3
        self.util_window = [deque(maxlen=self.t_window) for _ in range(self.n_agents)]

        # Optional efficiency normalization with EMA (kept from your version)
        self.eff_norm_ema = None
        self.ema_beta = 0.99

        # We keep the API but **do not** use L2 in FEN
        self.use_episode_L2 = False
        self.episode_skill_mismatch = []

        if self.debug:
            print(f"[FEN] init | metric={self.metric} | lambda={self.lmbda:.3f} | "
                  f"fair_rew={self.fair_rew:.3f} | t_window={self.t_window} | "
                  f"clip={self.clip_fair:.3f} | delta={int(self.use_delta)}")

    # ---------- internals ----------

    def _log(self, msg):
        if self.debug:
            print(msg)

    def _update_eff_norm(self, eff_scalar):
        if eff_scalar is None:
            return 1.0
        x = abs(float(eff_scalar))
        if self.eff_norm_ema is None:
            self.eff_norm_ema = x + 1.0
        else:
            self.eff_norm_ema = self.ema_beta * self.eff_norm_ema + (1 - self.ema_beta) * (x + 1.0)
        self._log(f"[FEN] eff_norm_ema={self.eff_norm_ema:.6f}")
        return max(self.eff_norm_ema, 1.0)

    def _inv_cv(self, arr):
        mean = np.mean(arr)
        std = np.std(arr)
        mean = max(mean, self.u_bar_floor)
        fairness_param = mean / (std + self.eps)
        self._log(f"[FEN] inv_cv={fairness_param:.6f} (mean={mean:.6f}, std={std:.6f})")
        return fairness_param

    def _gini(self, arr):
        x = np.sort(np.maximum(arr, 0.0))
        n = len(x)
        if n == 0:
            return 0.0
        mean = np.mean(x)
        mean = max(mean, self.u_bar_floor)
        cum = np.cumsum(x)
        gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n if cum[-1] > 0 else 0.0
        fairness_param = 1.0 - np.clip(gini, 0.0, 1.0)
        self._log(f"[FEN] gini_fair={fairness_param:.6f}")
        return fairness_param

    def _fairness_param(self, arr):
        if self.metric == "gini":
            return self._gini(arr)
        return self._inv_cv(arr)

    # ---------- episode API ----------

    def reset_episode(self):
        for dq in self.util_window:
            dq.clear()
        self.episode_skill_mismatch.clear()
        # Reset delta baseline each episode
        self.prev_fair_core = 0.0
        self._t = 0

    def record_skill_mismatch(self, l2_scalar):
        # Kept for interface compatibility; unused in FEN
        if l2_scalar is not None:
            self.episode_skill_mismatch.append(float(l2_scalar))

    # ---------- main shaping ----------

    def per_step_reward(self, base_eff_rewards, per_agent_util_increments):
        """
        base_eff_rewards: scalar or per-agent array (efficiency signal)
        per_agent_util_increments: list length n_agents (utilities accrued this step)
        returns: list of shaped rewards per agent to ADD to env reward
        """
        self._t += 1

        # 1) Efficiency normalization (vectorized)
        eff_norm = self._update_eff_norm(
            np.mean(base_eff_rewards) if np.ndim(base_eff_rewards) else base_eff_rewards
        )
        if np.ndim(base_eff_rewards):
            r_eff_norm = np.asarray(base_eff_rewards, dtype=float) / eff_norm
        else:
            r_eff_norm = np.full(self.n_agents, float(base_eff_rewards) / eff_norm, dtype=float)

        # 2) Update sliding window utilities
        for i in range(self.n_agents):
            self.util_window[i].append(float(per_agent_util_increments[i]))

        # 3) Fairness signal
        recent = np.array([np.sum(dq) if len(dq) else 0.0 for dq in self.util_window], dtype=float)
        fairness_param = self._fairness_param(recent)

        # Core fairness (bounded by tanh, then clipped)
        fair_core = self.fair_rew * np.tanh(fairness_param - self.zeroshift)
        fair_core = float(np.clip(fair_core, -self.clip_fair, self.clip_fair))

        # Delta mode (encourage becoming fair)
        if self.use_delta:
            fair_signal = fair_core - self.prev_fair_core
            self.prev_fair_core = fair_core
        else:
            fair_signal = fair_core

        # Team bonus scaled by lambda
        fair_team = self.lmbda * fair_signal

        # Distribute team fairness across agents (avoids n_agents multiplication)
        per_agent_fair = fair_team / self.n_agents

        # 4) Combine with efficiency term
        total = r_eff_norm + per_agent_fair

        # Debug line similar to your run logs
        self._log(
            f"[FEN] t={self._t:3d} | inc={list(map(float, per_agent_util_increments))} "
            f"| base={np.mean(r_eff_norm):+0.6f} | fair_core={fair_core:+0.6f} "
            f"| fair_term={fair_team:+0.6f} | shaped_mean={np.mean(total):+0.6f}"
        )

        return total.tolist()

    def end_of_episode_bonus(self):
        """
        FEN baseline: NO skill-task alignment shaping at episode end.
        Metrics may be recorded elsewhere for analysis, but return 0.0 here.
        """
        return 0.0
