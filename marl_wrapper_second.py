from typing import List, Optional, Union
import gym
import pddlgym
from rl.marl.hosp_marl_env import HospitalMARLEnv
from rl.marl.marl_env import MARLEnv
from utils.robotouille_utils import get_valid_moves
import utils.pddlgym_utils as pddlgym_utils
import utils.robotouille_wrapper as robotouille_wrapper
import torch as th


class MARLWrapper(robotouille_wrapper.RobotouilleWrapper):
    """
    This class is a wrapper around the Robotouille environment to make it compatible with stable-baselines3. It simplifies the environment for the RL agent by converting the state and action space to a format that is easier for the RL agent to learn.
    """

    def __init__(self, env, json, renderer, n_agents):
        print(f"Number of PyTorch threads in MARL wrapper: {th.get_num_threads()}")
        self.env = env  # gym environment
        self.pddl_env = (
            env  # robotouille wrapper environment, not pddl environment just yet
        )
        self.json = json
        self.n_agents = n_agents
        self.max_steps = 50
        self.episode_reward = 0
        self.renderer = renderer
        self._wrap_env()
        
    def _wrap_env(self):
        """
        Wrap the environment to make it compatible with epymarl.
        """
        expanded_truths, expanded_states = pddlgym_utils.expand_state(
            self.pddl_env.prev_step[0].literals, self.pddl_env.prev_step[0].objects
        )

        valid_actions = get_valid_moves(  # Potential bug: the valid actions for the a state are the valid actions at the end of previous step - error prone

            self.pddl_env, self.pddl_env.prev_step[0], self.renderer
        )
        all_actions = list(
            self.pddl_env.action_space.all_ground_literals(
                self.pddl_env.prev_step[0], valid_only=False
            )
        )

        # if the environment is a RobotouilleWrapper, we need to change it to MARLEnv. Otherwise, just step the MARLEnv
        # TODO: How to incorporate other information about the state from robotouille wrapper?
        # What is the required format for HospitalMARLEnv?
        if not isinstance(self.env, HospitalMARLEnv):
            self.env = HospitalMARLEnv(
                self.n_agents,
                expanded_truths,
                expanded_states,
                valid_actions,
                all_actions,
                self.json,
            )
        else:
            self.env.step(expanded_truths, valid_actions)

        self.observation_space = self.env.observation_space

    def step(self, actions=None, interactive=False, debug=False):
        """
        Take a step in the environment.

        Returns:
            state (list): The state of the environment after the step.
            reward (float): The reward obtained from the step.
            done (bool): Whether the episode is done.
            truncated (bool): Whether the episode was truncated.
            info (dict): A dictionary containing information about the environment.
        """

        rewards = []
        done = False
        info = {}
        for i in range(len(actions)):
            action = self.env.unwrap_move(i, actions[i])
            if debug:
                print(f"Agent {i} taking action: {action}")
            # if moving, check if action is valid
            if action == "invalid":
                # obs, reward, done, info = self.pddl_env.prev_step
                # # print(f"Invalid action for action {actions[i]}")
                # obs, _, _, _ = self.pddl_env._change_selected_player(obs)
                # self.pddl_env.taken_actions.append("noop")
                # reward = 0 # TODO: no reward punishment yet for invalid action
                # self.pddl_env.prev_step = (obs, reward, done, info)
                # rewards.append(reward)
                # self.pddl_env.move_counter += 1
                # if self.pddl_env._current_selected_player(obs) == "robot1":
                #     self.pddl_env.timesteps += 1
                # info["timesteps"] = self.pddl_env.timesteps
                obs, reward, done, info = self.pddl_env.step("noop", interactive)
                #reward -= 0.005
                rewards.append(reward)
            else:
                action = str(action)
                obs, reward, done, info = self.pddl_env.step(action, interactive)
                # Reward .05 for correct action. .05 * 3 agents * 100 timesteps + max 35 reward = 50- cooking setup
                # Reward .01 for correct action. .01 * 4 agents * 100 timesteps + max 50 after normalizing by dividing with timesteps in robotouille wrapper
                #  reward = 50 - For hospital setup
                # - cooking setup
                # Scale between 0 to 1, # TODO: lets not make this hardcoded
                # /194 for givemedicineequal, /217 for givemedicinespec #already add 4 from the top
                # /91 for giverescuebreaths, /99 for giverescuebreathsspec#already add 4 from the top

                #reward = (reward + 0.01)

                self.pddl_env.prev_step = (obs, reward, done, info)

                rewards.append(reward)

            # NOTE: We need to do this because when we filter for vaild moves during each step,
            # we need to have player grid locations maintained in the renderer
            # TODO: Maybe have this inside robotouille_wrapper.py?
            self.pddl_env.renderer.canvas.update_all_player_pos(obs.literals)

            self._wrap_env()

        self.episode_reward += sum(rewards)

        return (
            self.env.state,  # HospitalMARLEnv state
            rewards,
            done,
            info,
        )
    # In MARLWrapper class
    def get_episode_goal_reached(self):
        """Returns 1 if a goal was reached in the current episode, 0 otherwise."""
        # Check if pddl_env has get_episode_goal_reached method
        if hasattr(self.pddl_env, 'get_episode_goal_reached'):
            result = self.pddl_env.get_episode_goal_reached()
            # print(f"MARLWrapper: get_episode_goal_reached returning {result}")
            return result
        
        # # Alternative: try to access goal_announced directly
        # elif hasattr(self.pddl_env, 'goal_announced'):
        #     result = 1 if self.pddl_env.goal_announced else 0
        #     print(f"MARLWrapper: using goal_announced, returning {result}")
        #     return result
        
        # Fallback only if neither option works
        # print("MARLWrapper: Warning - no goal information found")
        return 0  # Default to failure when we can't determine goal state

    def reset(self, seed=42, options=None):
        """
        Reset the environment to its initial state.

        Returns:
            state (list): The initial state of the environment.
            info (dict): A dictionary containing information about the environment.
        """
        obs, info = self.pddl_env.reset()
        self.episode_reward = 0
        self._wrap_env()
        # print("self.env.state", self.env.state)
        return self.env.state, info

    def render(self, *args, **kwargs):
        self.pddl_env.render(mode="rgb_array")
