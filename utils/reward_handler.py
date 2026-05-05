class RewardHandler:
    def __init__(self):
        self.reward = None
        self.prev_reward = None
        self.obs = None
        self.state = None

    def heuristic_reward(self, obs, state):
        raise NotImplementedError
