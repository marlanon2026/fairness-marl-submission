import math
import numpy as np

class AdaptiveLambdaScheduler:
    """
    Gradually increases fairness constraint strength to allow initial task learning.
    """
    def __init__(self, 
                 initial_lambda=0.1,
                 target_lambda=50.0,
                 warmup_episodes=1000,
                 schedule_type='cosine'):
        self.initial_lambda = initial_lambda
        self.target_lambda = target_lambda
        self.warmup_episodes = warmup_episodes
        self.schedule_type = schedule_type
        self.current_episode = 0
        
    def get_lambda(self, episode=None):
        """Get current lambda value based on training progress."""
        if episode is not None:
            self.current_episode = episode
            
        if self.current_episode >= self.warmup_episodes:
            return self.target_lambda
            
        progress = self.current_episode / self.warmup_episodes
        
        if self.schedule_type == 'linear':
            return self.initial_lambda + (self.target_lambda - self.initial_lambda) * progress
        elif self.schedule_type == 'exponential':
            # Exponential growth - slower at start, faster at end
            return self.initial_lambda * (self.target_lambda / self.initial_lambda) ** progress
        elif self.schedule_type == 'cosine':
            # Cosine annealing - smooth transition
            cos_progress = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
            return self.initial_lambda + (self.target_lambda - self.initial_lambda) * (1 - cos_progress)
        elif self.schedule_type == 'step':
            # Step increases at 25%, 50%, 75% of warmup
            if progress < 0.25:
                return self.initial_lambda
            elif progress < 0.5:
                return self.initial_lambda + 0.25 * (self.target_lambda - self.initial_lambda)
            elif progress < 0.75:
                return self.initial_lambda + 0.5 * (self.target_lambda - self.initial_lambda)
            else:
                return self.initial_lambda + 0.75 * (self.target_lambda - self.initial_lambda)
    
    def step(self):
        """Increment episode counter."""
        self.current_episode += 1
        return self.get_lambda()
        
    def reset(self):
        """Reset the scheduler."""
        self.current_episode = 0
        
    def get_progress(self):
        """Get training progress as percentage."""
        return min(100.0, (self.current_episode / self.warmup_episodes) * 100)