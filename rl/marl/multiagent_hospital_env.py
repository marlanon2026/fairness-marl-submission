import numpy as np
from rl.marl.marl_wrapper import MARLWrapper
from rl.marl.multiagentenv import MultiAgentEnv
from utils.robotouille_utils import get_valid_moves
import utils.pddlgym_utils as pddlgym_utils

import gym
import torch as th
import os


class MAHospital_robotouille(MultiAgentEnv):
    def __init__(
        self,
        env,
        json,
        renderer,
    ):

        observation_mode = os.environ.get('OBSERVATION_MODE', 'LARGE')
        #print(f"DEBUG: Using observation_mode from environment variable: {observation_mode}")

                
        # Get reward type from environment variable (NEW)
        reward_type = os.environ.get('REWARD_TYPE', 'normal')
        
        # Get lambda fairness from environment variable or config (NEW)
        lambda_fairness = float(os.environ.get('LAMBDA_FAIRNESS', 
        json.get('lambda_fairness', 0.1)))
        json['lambda_fairness'] = lambda_fairness

        
        fairness_alpha = float(os.environ.get('FAIRNESS_ALPHA', 
            json.get('fairness_alpha', 0.7)))
        json['fairness_alpha'] = fairness_alpha
        
        print(f"[MAHospital] Observation mode: {observation_mode}")
        print(f"[MAHospital] Reward type: {reward_type}")
        print(f"[MAHospital] Lambda fairness: {lambda_fairness}")
        print(f"[MAHospital] Fairness alpha: {fairness_alpha}")
        if reward_type == 'fairness':
            print(f"[MAHospital] Lambda fairness: {lambda_fairness}")
        
        # Update json config with reward settings (NEW)
        json['reward_config'] = {
            'reward_type': reward_type,
            'lambda_fairness': lambda_fairness
        }
    
        self.pddl_env = env
        self.json = json
        self.renderer = renderer
        self.n_agents = env.num_players
        self.env = MARLWrapper(
            self.pddl_env, 
            self.json, 
            self.renderer, 
            self.n_agents, 
            observation_mode=observation_mode
        )

        self.action_space = [
            gym.spaces.Discrete(self.env.env.action_space.n)
            for _ in range(self.n_agents)
        ]

        self.observation_space = [
            gym.spaces.MultiBinary(
                self.env.env.observation_space.n,
            )
            for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.obs = None
        self.episode_limit = 50

    def step(self, _actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(_actions):
            actions = _actions.cpu().numpy()
        else:
            actions = _actions
        self.time_step += 1
        obs, rewards, done, infos = self.env.step(actions.tolist())

        self.obs = obs

        if self.time_step >= self.episode_limit:
            done = True
        # print("sum(rewards): ", sum(rewards))
        return sum(rewards), done, {}

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self.obs.reshape(self.n_agents, -1)

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs[agent_id].reshape(-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        obs_size = np.array(self.env.observation_space.shape)
        return int(obs_size.prod())

    def get_global_state(self):
        return self.obs.flatten()

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.obs, _ = self.env.reset()
        return self.get_obs(), self.get_global_state()

    def render(self):
        self.pddl_env.render(mode="human")

    def save_episode(self, filename):
        self.pddl_env.save_episode(filename)

    def get_episode_actions(self):
        return self.pddl_env.get_episode_actions()

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass

    def get_stats(self):
        return {}
    
    def get_episode_goal_reached(self):
        """Returns whether the goal was reached in the current episode."""
        if hasattr(self.env, 'get_episode_goal_reached'):
            # Forward to MARLWrapper
            result = self.env.get_episode_goal_reached()
            # print(f"MAHospital_robotouille: get_episode_goal_reached returning {result}")
            return result
        
        print("Warning: env does not have get_episode_goal_reached method")
        return 0