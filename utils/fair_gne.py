"""
Fair-GNE: Fairness-Constrained MARL via Dual Ascent
max J(π)  s.t.  JFI(workloads) ≥ τ
"""

import os
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any

class FairGNE:
    """
    Lagrangian dual ascent for fairness constraint:
      L(π, λ) = J(π) - λ (τ - JFI)
    """

    def __init__(self, tau: float = 0.85, dual_lr: float = 0.01, lambda_max: float = 10.0):
        self.tau = float(tau)
        self.dual_lr = float(dual_lr)
        self.lambda_max = float(lambda_max)
        self.lambda_fair = 0.0

        # Runtime knobs (synced/overridden by wrapper if set)
        self.full_lagrangian = os.getenv("FAIR_GNE_FULL_LAGRANGIAN", "false").lower() in ("1", "true", "yes")
        self.update_freq = int(os.getenv("FAIR_GNE_UPDATE_FREQ", "10"))

        # Logging (optional)
        self.metrics = defaultdict(list)
        self.episode_count = 0

    @staticmethod
    def compute_jfi(workloads: List[int]) -> float:
        """Jain's Fairness Index: (Σw_i)^2 / (N * Σw_i^2). Treat zero-sum as perfectly fair."""
        w = np.asarray(workloads, dtype=float)
        n = len(w)
        if n == 0:
            return 1.0
        s = w.sum()
        if s <= 0:
            return 1.0
        sq = (w * w).sum()
        if sq <= 0:
            return 1.0
        return (s * s) / (n * sq)

    def constraint_violation(self, jfi: float) -> float:
        """g(jfi) = τ − JFI. Positive when constraint is violated."""
        return float(self.tau - float(jfi))

    def update_dual(self, g_fair: float) -> float:
        """Projected dual ascent: λ ← [λ + η·g]^+ within [0, λ_max]."""
        lam = self.lambda_fair + self.dual_lr * float(g_fair)
        lam = max(0.0, min(self.lambda_max, lam))
        self.lambda_fair = lam
        return self.lambda_fair

    def process_episode(self, workloads: List[int], base_reward: float) -> Tuple[float, Dict[str, Any]]:
        """
        Call at episode end.
        Returns: (shaped_reward, metrics_dict)
        Note: Only applies episode-level shaping if full_lagrangian=True.
        """
        jfi = self.compute_jfi(workloads)
        g = self.constraint_violation(jfi)

        # Update λ with raw g (can be negative), projection keeps λ ≥ 0
        self.update_dual(g)

        # Episode shaping: only when explicitly requested
        penalty = self.lambda_fair * g
        shaped_reward = base_reward if not self.full_lagrangian else (base_reward - penalty)

        self.episode_count += 1
        metrics = {
            "episode": self.episode_count,
            "jfi": float(jfi),
            "g_fair": float(g),
            "lambda": float(self.lambda_fair),
            "penalty": float(penalty),
            "shaped_reward": float(shaped_reward),
        }

        # Optional history
        for k, v in metrics.items():
            self.metrics[k].append(v)

        return shaped_reward, metrics
