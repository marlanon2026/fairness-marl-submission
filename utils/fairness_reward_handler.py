from utils.hosp_reward_handler import HospRewardHandler
from utils.fairness_metrics_cal import compute_L1, compute_L2, compute_L3
from utils.adaptive_lambda_scheduler import AdaptiveLambdaScheduler
from utils.reward_normalizer import AdaptiveRewardNormalizer
import numpy as np
import random

class FairnessRewardHandler(HospRewardHandler):
    """
    Extends HospRewardHandler with fairness-aware reward shaping.
    """
    def __init__(self, state, config):
        super().__init__(state)
        
        # Extract fairness parameters from config
        self.use_fairness = config.get('use_fairness', False)
        self.lambda_fairness = config.get('lambda_fairness', 50.0)
        self.fairness_alpha = config.get('fairness_alpha', 0.7)  # Weight between L1 and L2
        
        # Initialize adaptive components
        self.scheduler = AdaptiveLambdaScheduler(
            initial_lambda=config.get('initial_lambda', 0.5),
            target_lambda=self.lambda_fairness,
            warmup_episodes=config.get('warmup_episodes', 2000),
            schedule_type=config.get('schedule_type', 'cosine')
        )
        
        self.normalizer = AdaptiveRewardNormalizer(
            window_size=config.get('normalizer_window', 1000),
            min_samples=config.get('normalizer_min_samples', 100)
        )
        
        # Track episode count for scheduler
        self.episode_count = 0
        
        # Track metrics for current episode
        self.episode_metrics = {
            'agent_task_counts': [],
            'task_skill_pairs': [],
            'all_agents_skills': {}
        }
        
    def compute_fairness_reward(self, base_reward, fairness_info=None):
        """
        Compute the fairness-adjusted reward.
        
        Args:
            base_reward: Original reward from parent class
            fairness_info: Dict containing fairness metrics
            
        Returns:
            tuple: (final_reward, debug_info)
        """
        if not self.use_fairness or fairness_info is None:
            return base_reward, {'fairness_applied': False}
        
        # Extract fairness metrics
        agent_task_counts = fairness_info.get('agent_task_counts', [])
        task_skill_pairs = fairness_info.get('task_skill_pairs', [])
        all_agents_skills = fairness_info.get('all_agents_skills', {})
        
        # Compute disparity measures
        L1 = compute_L1(agent_task_counts)
        L2 = compute_L2(task_skill_pairs, all_agents_skills)
        L3 = compute_L3(L1, L2, alpha=self.fairness_alpha)
        
        # Get current lambda from scheduler
        current_lambda = self.scheduler.get_lambda()
        
        # Normalize the reward
        final_reward = self.normalizer.normalize_reward(
            base_reward, L3, current_lambda
        )
        
        # Debug information
        debug_info = {
            'fairness_applied': True,
            'base_reward': base_reward,
            'L1': L1,
            'L2': L2,
            'L3': L3,
            'current_lambda': current_lambda,
            'fairness_penalty': L3 * current_lambda,
            'final_reward': final_reward,
            'scheduler_progress': self.scheduler.get_progress(),
            'normalizer_stats': self.normalizer.get_stats()
        }

        # === ADD THESE DEBUG PRINTS ===
        # Print every 100 episodes or every 1000 steps
        if self.episode_count % 100 == 0: # or random.random() < 0.001:
            print(f"\n[FAIRNESS DEBUG] Episode {self.episode_count}")
            print(f"  Lambda: {current_lambda:.2f} (target: {self.lambda_fairness})")
            print(f"  Progress: {self.scheduler.get_progress():.1f}%")
            print(f"  Base reward: {base_reward:.4f}")
            print(f"  L1: {L1:.3f}, L2: {L2:.3f}, L3: {L3:.3f}")
            print(f"  Fairness penalty: {L3 * current_lambda:.4f}")
            print(f"  Final reward: {final_reward:.4f}")
        
        # Normalizer stats
        stats = self.normalizer.get_stats()
        if stats:
            pass
            # print(f"  Normalizer - Avg base: {stats['avg_base']:.3f}, "
            #       f"Avg penalty: {stats['avg_penalty']:.3f}, "
            #       f"Avg final: {stats['avg_final']:.3f}")
        
        return final_reward, debug_info
    
    def heuristic_reward(self, obs, state, fairness_info=None):
        """
        Override to add fairness consideration.
        """
        # Get base reward from parent class
        base_reward = super().heuristic_reward(obs, state)
        
        # Apply fairness adjustment
        final_reward, debug_info = self.compute_fairness_reward(base_reward, fairness_info)
        
        # Store debug info for logging
        self.last_debug_info = debug_info
        
        return final_reward
    
    def episode_end(self):
        """Call this at the end of each episode to update scheduler."""
        self.episode_count += 1
        self.scheduler.step()
        
        # Reset episode metrics
        self.episode_metrics = {
            'agent_task_counts': [],
            'task_skill_pairs': [],
            'all_agents_skills': {}
        }
        
    # def reset(self):
    #     """Reset the handler state."""
    #     super().reset()


    def reset(self):
            """Reset the handler state."""
            # FIXED: Call parent's reset method using proper syntax
            super(FairnessRewardHandler, self).reset()
        # Don't reset scheduler or normalizer - they maintain across episodes
        
    def get_current_lambda(self):
        """Get the current lambda value."""
        return self.scheduler.get_lambda()
    
    def get_stats(self):
        """Get current statistics for logging."""
        return {
            'episode': self.episode_count,
            'lambda': self.scheduler.get_lambda(),
            'scheduler_progress': self.scheduler.get_progress(),
            'normalizer_stats': self.normalizer.get_stats(),
            'last_debug': getattr(self, 'last_debug_info', {})
        }