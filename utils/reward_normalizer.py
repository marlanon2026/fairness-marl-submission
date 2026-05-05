import numpy as np
from collections import deque

class AdaptiveRewardNormalizer:
    """
    Maintains running statistics to normalize rewards and prevent 
    fairness penalties from dominating the learning signal.
    """
    def __init__(self, window_size=1000, min_samples=100):
        self.window_size = window_size
        self.min_samples = min_samples
        self.base_rewards = deque(maxlen=window_size)
        self.fairness_penalties = deque(maxlen=window_size)
        self.final_rewards = deque(maxlen=window_size)
        
    def normalize_reward(self, base_reward, fairness_penalty, lambda_fairness):
        """
        Normalize rewards to prevent fairness from dominating.
        
        Args:
            base_reward: Original task reward
            fairness_penalty: L3 disparity measure (0-1)
            lambda_fairness: Current lambda value
            
        Returns:
            Normalized final reward
        """
        # Store statistics
        self.base_rewards.append(base_reward)
        self.fairness_penalties.append(fairness_penalty)
        
        # Method 1: Relative scaling based on running statistics
        if len(self.base_rewards) >= self.min_samples:
            avg_base = np.mean(self.base_rewards)
            std_base = np.std(self.base_rewards) + 1e-8
            avg_penalty = np.mean(self.fairness_penalties)
            
            # Scale fairness penalty to be proportional to base reward magnitude
            if avg_penalty > 0 and avg_base > 0:
                # Fairness penalty should be at most 50% of average base reward
                scale_factor = min(1.0, (0.5 * avg_base) / (lambda_fairness * avg_penalty))
                scaled_penalty = lambda_fairness * fairness_penalty * scale_factor
            else:
                scaled_penalty = lambda_fairness * fairness_penalty * 0.1
        else:
            # During initial phase, use conservative scaling
            scaled_penalty = lambda_fairness * fairness_penalty * 0.1
        
        # Method 2: Soft clipping with tanh to prevent extreme penalties
        max_penalty = max(abs(base_reward) * 0.5, 0.1)  # At least 0.1
        if scaled_penalty > max_penalty:
            scaled_penalty = max_penalty * np.tanh(scaled_penalty / max_penalty)
        
        # Compute final reward
        final_reward = base_reward - scaled_penalty
        
        # Method 3: Ensure minimum reward for valid actions
        if base_reward > 0:  # Valid action that makes progress
            final_reward = max(final_reward, 0.01)  # Small positive reward
        
        # Store final reward for statistics
        self.final_rewards.append(final_reward)
        
        return final_reward
        
    def get_stats(self):
        """Get current reward statistics for monitoring."""
        if len(self.final_rewards) > 0:
            return {
                'avg_base': np.mean(self.base_rewards) if self.base_rewards else 0,
                'std_base': np.std(self.base_rewards) if self.base_rewards else 0,
                'avg_penalty': np.mean(self.fairness_penalties) if self.fairness_penalties else 0,
                'avg_final': np.mean(self.final_rewards) if self.final_rewards else 0,
                'std_final': np.std(self.final_rewards) if self.final_rewards else 0,
                'num_samples': len(self.final_rewards)
            }
        return {
            'avg_base': 0, 'std_base': 0, 'avg_penalty': 0,
            'avg_final': 0, 'std_final': 0, 'num_samples': 0
        }
    
    def reset(self):
        """Reset all statistics."""
        self.base_rewards.clear()
        self.fairness_penalties.clear()
        self.final_rewards.clear()