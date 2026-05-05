from typing import List, Optional, Union
import gym
import pddlgym
from rl.marl.hosp_marl_env import HospitalMARLEnv
from rl.marl.marl_env import MARLEnv
from utils.robotouille_utils import get_valid_moves
import utils.pddlgym_utils as pddlgym_utils
import utils.robotouille_wrapper as robotouille_wrapper
import torch as th
import gzip
import os

class MARLWrapper(robotouille_wrapper.RobotouilleWrapper):
    """
    This class is a wrapper around the Robotouille environment to make it compatible with stable-baselines3. It simplifies the environment for the RL agent by converting the state and action space to a format that is easier for the RL agent to learn.
    """

    def __init__(self, env, json, renderer, n_agents, observation_mode=None):
        # print(f"DEBUG: ===== MARLWRAPPER INIT DEBUG =====")
        # print(f"DEBUG: MARLWrapper received observation_mode parameter: {observation_mode}")
        # print(f"DEBUG: MARLWrapper observation_mode type: {type(observation_mode)}")
        # print(f"DEBUG: MARLWrapper json keys: {list(json.keys()) if isinstance(json, dict) else 'Not a dict'}")
        # if isinstance(json, dict):
        #     print(f"DEBUG: MARLWrapper json['env_args']: {json.get('env_args', 'NOT_FOUND')}")
        #     print(f"DEBUG: MARLWrapper json['config']: {json.get('config', 'NOT_FOUND')}")
        # print(f"DEBUG: ===== END MARLWRAPPER INIT DEBUG =====")
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
        self.observation_mode = observation_mode

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
                observation_mode=self.observation_mode
            )
        else:
            self.env.step(expanded_truths, valid_actions)

        # Update the observation space reference after wrapping/stepping
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
                # print(f"Agent {i} taking action: {action}")
                pass
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
                reward -= 0.0005
                rewards.append(reward)
            else:
                action = str(action)
                obs, reward, done, info = self.pddl_env.step(action, interactive)
                # print("[DEBUG] PDDL State (Raw):")
                # print(obs)
                # if hasattr(obs, "literals"):
                #     for literal in obs.literals:
                #         print("  -", literal)

                #print(f"[Action Log] Agent {i}: {action}", flush=True)
                # Reward .05 for correct action. .05 * 3 agents * 100 timesteps + max 35 reward = 50- cooking setup
                # Reward .01 for correct action. .01 * 4 agents * 100 timesteps + max 50 after normalizing by dividing with timesteps in robotouille wrapper
                #  reward = 50 - For hospital setup
                # - cooking setup
                # Scale between 0 to 1, # TODO: lets not make this hardcoded
                # /194 for givemedicineequal, /217 for givemedicinespec #already add 4 from the top
                # /91 for giverescuebreaths, /99 for giverescuebreathsspec#already add 4 from the top

                #reward = (reward + 0.01)
                # Print the state names to see what was included

                # Get the goal type by checking the PDDL predicates
                is_rescue_breath_goal = False
                for literal in self.pddl_env.prev_step[0].literals:
                    if "isrescuebreathed" in str(literal.predicate.name) and "goal" in str(literal):
                        is_rescue_breath_goal = True
                        break

                # Apply appropriate normalization based on goal type
                if is_rescue_breath_goal:
                    # For rescue breaths goal
                    reward = reward #/ 150  # Normalize to 0-1 range
                else:
                    # For give medicine or other goals
                    reward = reward #/ 300  # Normalize to 0-1 range

                #print("reward in Marl_wrapper:", reward)

                self.pddl_env.prev_step = (obs, reward, done, info)

                rewards.append(reward)

            # NOTE: We need to do this because when we filter for vaild moves during each step,
            # we need to have player grid locations maintained in the renderer
            # TODO: Maybe have this inside robotouille_wrapper.py?
            self.pddl_env.renderer.canvas.update_all_player_pos(obs.literals)

            self._wrap_env()

        self.episode_reward += sum(rewards)

        # print("="*60, flush=True)
        # print("[DEBUG] MARLWrapper state contents:", self.env.state, flush=True)
        # print("[DEBUG] MARLWrapper observation space:", self.observation_space, flush=True)
        # print("[DEBUG] MARLWrapper action space:", self.env.action_space, flush=True)
        #print("="*60, flush=True)

        if hasattr(self.pddl_env.reward_handler, 'get_stats') and self.pddl_env.timesteps % 10 == 0:
            stats = self.pddl_env.reward_handler.get_stats()
            if debug:
                print(f"Step {self.pddl_env.timesteps}: Lambda={stats['lambda']:.2f}, "
                    f"Progress={stats['scheduler_progress']:.1f}%")

        return (
            self.env.state,  # HospitalMARLEnv state
            rewards,
            done,
            info,
        )
    

    def save_current_episode_log(self):
        """Force save the current episode log"""
        if self._log_test_mode and self._action_log and hasattr(self, '_log_dir'):
            filename = os.path.join(self._log_dir, f"episode_{self._episode_counter}.json.gz")
            ep_data = {
                "episode": self._episode_counter,
                "actions": self._action_log,
                "timesteps": self.timesteps
            }
            with gzip.open(filename, 'wt') as f:
                json.dump(ep_data, f)
            print(f"[WRAPPER] Manually saved {len(self._action_log)} actions to {filename}")


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
        print("MARLWrapper: Warning - no goal information found")
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
        #print("="*60, flush=True)
        #print("[DEBUG] MARLWrapper initial state:", self.env.state, flush=True)
        #print("[DEBUG] MARLWrapper observation space:", self.observation_space, flush=True)
        #print("[DEBUG] MARLWrapper action space:", self.env.action_space, flush=True)
        #print("="*60, flush=True)


        return self.env.state, info

    def render(self, *args, **kwargs):
        self.pddl_env.render(mode="rgb_array")