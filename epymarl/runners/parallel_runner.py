from copy import deepcopy
import os
import sys
import time  # ADD THIS IMPORT
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import cloudpickle
import pickle

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.global_episode_counter = 0
        
        # Add per-environment tracking
        self.env_episode_fairness = {i: [] for i in range(self.batch_size)}  # Track per env
        self.env_episode_counters = {i: 0 for i in range(self.batch_size)}

        # Add timestep-based tracking
        self.fairness_by_timestep = {
            "train": {"timesteps": [], "L1": [], "L2": [], "L3": []},
            "test": {"timesteps": [], "L1": [], "L2": [], "L3": []}
        }
        
        # Add episode-level detailed tracking
        self.detailed_episode_fairness = []  # List of dicts with full episode info

        # Extract tracking directory from env_args (like episode_runner)
        self.tracking_dir = self.args.env_args.get("tracking_dir", None)
        if self.tracking_dir:
            pass
            # print(f"[ParallelRunner] Will use tracking directory: {self.tracking_dir}")

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(
            *[Pipe() for _ in range(self.batch_size)]
        )
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            # env_args[i]["seed"] += i
            env_args[i]["seed"] = args.env_args["seed"] + (i * 1000)  # Ensures deterministic offsets
            # Ensure each worker gets the tracking directory
            if self.tracking_dir:
                env_args[i]["tracking_dir"] = self.tracking_dir

        self.ps = []
        for i, (env_arg, worker_conn) in enumerate(zip(env_args, self.worker_conns)):
            p = Process(
                target=env_worker,
                args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg)), env_args[i]),
            )
            self.ps.append(p)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0
        self.t_env = 0

        self.best_episode_reward = 0
        self.best_actions = []
        self.best_test_episode_reward = 0
        self.best_test_actions = []

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.train_goals = []
        self.test_goals = []
        self.train_episode_count = 0
        self.test_episode_count = 0
        
        # Add tracking for cumulative goals across all training
        self.total_train_goals_reached = 0
        self.total_train_episodes = 0
        self.total_test_goals_reached = 0
        self.total_test_episodes = 0

        # Add episode ID counters (like episode_runner)
        self.train_episode_id = 0
        self.test_episode_id = 0

        # Cumulative milestone tracking for training - MEDICAL TASKS
        self.total_train_chest_compressed = 0
        self.total_train_rescue_breaths = 0
        self.total_train_shock = 0
        self.total_train_medicine_administered = 0

        # Cumulative milestone tracking for testing - MEDICAL TASKS
        self.total_test_chest_compressed = 0
        self.total_test_rescue_breaths = 0
        self.total_test_shock = 0
        self.total_test_medicine_administered = 0

        # NEW: Cumulative milestone tracking for training - MOVEITEM TASKS
        self.total_train_first_item_moved = 0
        self.total_train_second_item_moved = 0
        self.total_train_all_items_moved = 0
        self.total_train_correct_stacking = 0

        # NEW: Cumulative milestone tracking for testing - MOVEITEM TASKS
        self.total_test_first_item_moved = 0
        self.total_test_second_item_moved = 0
        self.total_test_all_items_moved = 0
        self.total_test_correct_stacking = 0

        # ADD THESE NEW VARIABLES FOR FAIRNESS TRACKING
        self.train_fairness_L1 = []
        self.train_fairness_L2 = []
        self.train_fairness_L3 = []
        self.test_fairness_L1 = []
        self.test_fairness_L2 = []
        self.test_fairness_L3 = []

        self.accumulated_train_L1 = 0.0
        self.accumulated_train_L2 = 0.0
        self.accumulated_train_L3 = 0.0
        self.accumulated_test_L1 = 0.0
        self.accumulated_test_L2 = 0.0
        self.accumulated_test_L3 = 0.0
        self.fairness_episode_count_train = 0
        self.fairness_episode_count_test = 0

        # Initialize with a reasonable default, will be updated dynamically
        self.num_agents = 3  # Default, will be queried from environment
        self.accumulated_train_agent_tasks = {}
        self.accumulated_test_agent_tasks = {}
        self.accumulated_train_agent_workload = {}
        self.accumulated_test_agent_workload = {}

        # ADD PER-AGENT FAIRNESS TRACKING
        self.agent_workload_history = {}
        self.agent_skill_alignment_history = {}
        
        # Initialize recent_L3_values for variance tracking
        self.recent_L3_values = []

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        self.parent_conns[0].send(("save_replay", None))

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        # Reset the envs with test_mode
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", {"test_mode": test_mode}))

        pre_transition_data = {"state": [], "avail_actions": [], "obs": []}
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def _get_next_episode_id(self, test_mode):
        """Get the next episode ID for logging purposes"""
        if test_mode:
            self.test_episode_id += 1
            return self.test_episode_id
        else:
            self.train_episode_id += 1
            return self.train_episode_id

    def _update_cumulative_milestones(self, milestone_data, test_mode):
        """Update cumulative milestone counters based on milestone data"""
        if test_mode:
            # Medical milestones
            if milestone_data.get("chest_compressed", 0) == 1:
                self.total_test_chest_compressed += 1
            if milestone_data.get("rescue_breaths", 0) == 1:
                self.total_test_rescue_breaths += 1
            if milestone_data.get("shock", 0) == 1:
                self.total_test_shock += 1
            if milestone_data.get("medicine_administered", 0) == 1:
                self.total_test_medicine_administered += 1
            
            # NEW: Moveitem milestones
            if milestone_data.get("first_item_moved", 0) == 1:
                self.total_test_first_item_moved += 1
            if milestone_data.get("second_item_moved", 0) == 1:
                self.total_test_second_item_moved += 1
            if milestone_data.get("all_items_moved", 0) == 1:
                self.total_test_all_items_moved += 1
            if milestone_data.get("correct_stacking", 0) == 1:
                self.total_test_correct_stacking += 1
        else:
            # Medical milestones
            if milestone_data.get("chest_compressed", 0) == 1:
                self.total_train_chest_compressed += 1
            if milestone_data.get("rescue_breaths", 0) == 1:
                self.total_train_rescue_breaths += 1
            if milestone_data.get("shock", 0) == 1:
                self.total_train_shock += 1
            if milestone_data.get("medicine_administered", 0) == 1:
                self.total_train_medicine_administered += 1
            
            # NEW: Moveitem milestones
            if milestone_data.get("first_item_moved", 0) == 1:
                self.total_train_first_item_moved += 1
            if milestone_data.get("second_item_moved", 0) == 1:
                self.total_train_second_item_moved += 1
            if milestone_data.get("all_items_moved", 0) == 1:
                self.total_train_all_items_moved += 1
            if milestone_data.get("correct_stacking", 0) == 1:
                self.total_train_correct_stacking += 1

    def _log_cumulative_milestones(self, test_mode, log_prefix):
        """Log cumulative milestone statistics for both medical and moveitem tasks"""
        if test_mode:
            # Test milestone statistics
            total_episodes = max(1, self.total_test_episodes)
            
            # Medical milestones
            self.logger.log_stat(log_prefix + "total_chest_compressed", self.total_test_chest_compressed, self.t_env)
            self.logger.log_stat(log_prefix + "total_rescue_breaths", self.total_test_rescue_breaths, self.t_env)
            self.logger.log_stat(log_prefix + "total_shock", self.total_test_shock, self.t_env)
            self.logger.log_stat(log_prefix + "total_medicine_administered", self.total_test_medicine_administered, self.t_env)
            
            # NEW: Moveitem milestones
            self.logger.log_stat(log_prefix + "total_first_item_moved", self.total_test_first_item_moved, self.t_env)
            self.logger.log_stat(log_prefix + "total_second_item_moved", self.total_test_second_item_moved, self.t_env)
            self.logger.log_stat(log_prefix + "total_all_items_moved", self.total_test_all_items_moved, self.t_env)
            self.logger.log_stat(log_prefix + "total_correct_stacking", self.total_test_correct_stacking, self.t_env)
            
            # Test milestone success rates - Medical
            self.logger.log_stat(log_prefix + "cumulative_chest_compressed_rate", 
                                self.total_test_chest_compressed / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_rescue_breaths_rate", 
                                self.total_test_rescue_breaths / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_shock_rate", 
                                self.total_test_shock / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_medicine_administered_rate", 
                                self.total_test_medicine_administered / total_episodes, self.t_env)
            
            # NEW: Test milestone success rates - Moveitem
            self.logger.log_stat(log_prefix + "cumulative_first_item_moved_rate", 
                                self.total_test_first_item_moved / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_second_item_moved_rate", 
                                self.total_test_second_item_moved / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_all_items_moved_rate", 
                                self.total_test_all_items_moved / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_correct_stacking_rate", 
                                self.total_test_correct_stacking / total_episodes, self.t_env)
        else:
            # Training milestone statistics
            total_episodes = max(1, self.total_train_episodes)
            
            # Medical milestones
            self.logger.log_stat(log_prefix + "total_chest_compressed", self.total_train_chest_compressed, self.t_env)
            self.logger.log_stat(log_prefix + "total_rescue_breaths", self.total_train_rescue_breaths, self.t_env)
            self.logger.log_stat(log_prefix + "total_shock", self.total_train_shock, self.t_env)
            self.logger.log_stat(log_prefix + "total_medicine_administered", self.total_train_medicine_administered, self.t_env)
            
            # NEW: Moveitem milestones
            self.logger.log_stat(log_prefix + "total_first_item_moved", self.total_train_first_item_moved, self.t_env)
            self.logger.log_stat(log_prefix + "total_second_item_moved", self.total_train_second_item_moved, self.t_env)
            self.logger.log_stat(log_prefix + "total_all_items_moved", self.total_train_all_items_moved, self.t_env)
            self.logger.log_stat(log_prefix + "total_correct_stacking", self.total_train_correct_stacking, self.t_env)
            
            # Training milestone success rates - Medical
            self.logger.log_stat(log_prefix + "cumulative_chest_compressed_rate", 
                                self.total_train_chest_compressed / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_rescue_breaths_rate", 
                                self.total_train_rescue_breaths / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_shock_rate", 
                                self.total_train_shock / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_medicine_administered_rate", 
                                self.total_train_medicine_administered / total_episodes, self.t_env)
            
            # NEW: Training milestone success rates - Moveitem
            self.logger.log_stat(log_prefix + "cumulative_first_item_moved_rate", 
                                self.total_train_first_item_moved / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_second_item_moved_rate", 
                                self.total_train_second_item_moved / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_all_items_moved_rate", 
                                self.total_train_all_items_moved / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_correct_stacking_rate", 
                                self.total_train_correct_stacking / total_episodes, self.t_env)

    def _log_fairness_metrics(self, fairness_metrics, agent_contributions, prefix):
        """Log fairness metrics to WandB/logger"""
        if fairness_metrics:
            # CRITICAL: Log with exact names expected by CSV
            self.logger.log_stat(prefix + "L1", 
                                fairness_metrics.get("L1", 0.0), self.t_env)
            self.logger.log_stat(prefix + "L2", 
                                fairness_metrics.get("L2", 0.0), self.t_env)
            self.logger.log_stat(prefix + "L3", 
                                fairness_metrics.get("L3", 0.0), self.t_env)
            
            # Agent percentages from normalized_contributions
            if "normalized_contributions" in fairness_metrics:
                for i, contrib in enumerate(fairness_metrics["normalized_contributions"]):
                    self.logger.log_stat(f"{prefix}Agent {i} %", contrib * 100, self.t_env)
        
        # Additional detailed metrics if agent_contributions provided
        if agent_contributions:
            # Log per-agent metrics for paper visualizations
            for agent_id, contrib in agent_contributions.items():
                agent_prefix = f"{prefix}agent_{agent_id}/"
                
                # Workload metrics
                self.logger.log_stat(agent_prefix + "L1_workload_balance", 
                                    contrib.get("L1_workload_balance", 0.0), self.t_env)
                self.logger.log_stat(agent_prefix + "L2_skill_alignment", 
                                    contrib.get("L2_skill_alignment", 0.0), self.t_env)
                self.logger.log_stat(agent_prefix + "total_tasks", 
                                    contrib.get("total_tasks", 0), self.t_env)
                self.logger.log_stat(agent_prefix + "total_skill_contributed", 
                                    contrib.get("total_skill_contributed", 0.0), self.t_env)
                
                # For visualization: percentage of workload
                if "workload_percentage" in contrib:
                    self.logger.log_stat(agent_prefix + "workload_percentage", 
                                        contrib["workload_percentage"], self.t_env)
        
        # Log fairness "violations" for analysis (useful for ICLR paper)
        if fairness_metrics:
            # High L1 indicates workload imbalance
            if fairness_metrics.get("L1", 0) > 0.3:  # Threshold for "unfair"
                self.logger.log_stat(prefix + "fairness/high_workload_imbalance", 1.0, self.t_env)
            else:
                self.logger.log_stat(prefix + "fairness/high_workload_imbalance", 0.0, self.t_env)
            
            # High L2 indicates poor skill utilization
            if fairness_metrics.get("L2", 0) > 0.3:  # Threshold for "poor skill use"
                self.logger.log_stat(prefix + "fairness/poor_skill_utilization", 1.0, self.t_env)
            else:
                self.logger.log_stat(prefix + "fairness/poor_skill_utilization", 0.0, self.t_env)

    def run(self, test_mode=False):
        self.reset(test_mode=test_mode)

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        episode_goals = [0 for _ in range(self.batch_size)]  # Track goals for each env
        episode_fairness_metrics = [None for _ in range(self.batch_size)]  # Track fairness for each env
        episode_agent_contributions = [None for _ in range(self.batch_size)]  # Track agent contributions
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [
            b_idx for b_idx, termed in enumerate(terminated) if not termed
        ]
        final_env_infos = (
            []
        )  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                bs=envs_not_terminated,
                test_mode=test_mode,
            )
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {"actions": actions.unsqueeze(1)}
            self.batch.update(
                actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False
            )

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[
                        idx
                    ]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [
                b_idx for b_idx, termed in enumerate(terminated) if not termed
            ]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {"reward": [], "terminated": []}
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {"state": [], "avail_actions": [], "obs": []}

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get(
                        "episode_limit", False
                    ):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Check if goal has been reached when terminated
                    if data["terminated"]:
                        # Get goal status from the environment
                        parent_conn.send(("get_episode_goal_reached", None))
                        goal_reached = parent_conn.recv()
                        episode_goals[idx] = goal_reached

                        # Get milestone status when episode terminates
                        parent_conn.send(("get_milestone_status", None))
                        milestone_data = parent_conn.recv()
                        self._update_cumulative_milestones(milestone_data, test_mode)

                        # Get fairness metrics when episode terminates
                        parent_conn.send(("get_fairness_metrics", None))
                        fairness_data = parent_conn.recv()
                        if fairness_data:
                            episode_fairness_metrics[idx] = fairness_data.get("metrics", None)
                            episode_agent_contributions[idx] = fairness_data.get("contributions", None)
                            
                            # Accumulate fairness metrics
                            if episode_fairness_metrics[idx]:
                                if test_mode:
                                    self.accumulated_test_L1 += episode_fairness_metrics[idx].get("L1", 0.0)
                                    self.accumulated_test_L2 += episode_fairness_metrics[idx].get("L2", 0.0)
                                    self.accumulated_test_L3 += episode_fairness_metrics[idx].get("L3", 0.0)
                                    self.fairness_episode_count_test += 1
                                else:
                                    self.accumulated_train_L1 += episode_fairness_metrics[idx].get("L1", 0.0)
                                    self.accumulated_train_L2 += episode_fairness_metrics[idx].get("L2", 0.0)
                                    self.accumulated_train_L3 += episode_fairness_metrics[idx].get("L3", 0.0)
                                    self.fairness_episode_count_train += 1
                            
                            # Accumulate agent data
                            if episode_agent_contributions[idx]:
                                for agent_id, contrib in episode_agent_contributions[idx].items():
                                    # Initialize dictionaries if needed
                                    if agent_id not in self.accumulated_test_agent_tasks:
                                        self.accumulated_test_agent_tasks[agent_id] = 0
                                        self.accumulated_test_agent_workload[agent_id] = 0.0
                                    if agent_id not in self.accumulated_train_agent_tasks:
                                        self.accumulated_train_agent_tasks[agent_id] = 0
                                        self.accumulated_train_agent_workload[agent_id] = 0.0
                                    
                                    if test_mode:
                                        self.accumulated_test_agent_tasks[agent_id] += contrib["action_count"]
                                        self.accumulated_test_agent_workload[agent_id] += contrib["L1_workload_balance"]
                                    else:
                                        self.accumulated_train_agent_tasks[agent_id] += contrib["action_count"]
                                        self.accumulated_train_agent_workload[agent_id] += contrib["L1_workload_balance"]

                        # Just update episode ID counter (no individual episode logging)
                        current_episode_id = self.global_episode_counter
                        self.global_episode_counter += 1

                        # Save JSON log with proper test mode
                        parent_conn.send(("save_current_episode_log_with_id", {"episode_id": current_episode_id, "test_mode": test_mode}))  
                        _ = parent_conn.recv()                        

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(
                post_transition_data,
                bs=envs_not_terminated,
                ts=self.t,
                mark_filled=False,
            )

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(
                pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True
            )

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # Store goal results after all episodes are done
        cur_goals = self.test_goals if test_mode else self.train_goals
        cur_goals.extend(episode_goals) 
        
        # After all episodes, increment episode counters
        if test_mode:
            self.test_episode_count += self.batch_size
            self.total_test_episodes += self.batch_size
            self.total_test_goals_reached += sum(episode_goals)
        else:
            self.train_episode_count += self.batch_size
            self.total_train_episodes += self.batch_size
            self.total_train_goals_reached += sum(episode_goals)
               
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update(
            {
                k: sum(d.get(k, 0) for d in infos)
                for k in set.union(*[set(d) for d in infos])
            }
        )
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        self.episode_returns = episode_returns

        n_test_runs = (
            max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        )
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, cur_goals, log_prefix)
            
            # LOG AVERAGED FAIRNESS METRICS FOR TEST EPISODES
            if self.fairness_episode_count_test > 0:
                avg_test_fairness = {
                    "L1": self.accumulated_test_L1 / self.fairness_episode_count_test,
                    "L2": self.accumulated_test_L2 / self.fairness_episode_count_test,
                    "L3": self.accumulated_test_L3 / self.fairness_episode_count_test
                }

                # CREATE AVERAGED AGENT CONTRIBUTIONS
                avg_agent_contributions = {}
                # Get number of agents from first environment
                self.parent_conns[0].send(("get_num_agents", None))
                num_agents = self.parent_conns[0].recv()
                
                for agent_id in range(num_agents):
                    avg_agent_contributions[agent_id] = {
                        "L1_workload_balance": self.accumulated_test_agent_workload.get(agent_id, 0.0) / self.fairness_episode_count_test,
                        "action_count": self.accumulated_test_agent_tasks.get(agent_id, 0) / self.fairness_episode_count_test,
                        "workload_percentage": (self.accumulated_test_agent_workload.get(agent_id, 0.0) / self.fairness_episode_count_test) * 100
                    }
                self._log_fairness_metrics(avg_test_fairness, avg_agent_contributions, log_prefix)
                
                # RESET ACCUMULATORS
                self.accumulated_test_L1 = 0.0
                self.accumulated_test_L2 = 0.0
                self.accumulated_test_L3 = 0.0
                self.fairness_episode_count_test = 0
                # RESET AGENT ACCUMULATORS TOO!
                for agent_id in list(self.accumulated_test_agent_tasks.keys()):
                    self.accumulated_test_agent_tasks[agent_id] = 0
                    self.accumulated_test_agent_workload[agent_id] = 0.0
            
            # Log the maximum possible goals for testing
            self.logger.log_stat(log_prefix + "max_possible_goals", self.test_episode_count, self.t_env)
            # Log cumulative goal success rate for testing
            self.logger.log_stat(log_prefix + "total_goals_reached", self.total_test_goals_reached, self.t_env)
            self.logger.log_stat(log_prefix + "total_episodes", self.total_test_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_goal_success_rate", 
            self.total_test_goals_reached / max(1, self.total_test_episodes), self.t_env)

            # Log cumulative milestone statistics for testing
            self._log_cumulative_milestones(test_mode, log_prefix)
           
            self.test_episode_count = 0
        
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, cur_goals, log_prefix)
            
            # LOG AVERAGED FAIRNESS METRICS FOR TRAINING EPISODES
            if self.fairness_episode_count_train > 0:
                avg_train_fairness = {
                    "L1": self.accumulated_train_L1 / self.fairness_episode_count_train,
                    "L2": self.accumulated_train_L2 / self.fairness_episode_count_train,
                    "L3": self.accumulated_train_L3 / self.fairness_episode_count_train
                }
                
                # Get number of agents from first environment
                self.parent_conns[0].send(("get_num_agents", None))
                num_agents = self.parent_conns[0].recv()
                
                avg_agent_contributions = {}
                for agent_id in range(num_agents):
                    avg_agent_contributions[agent_id] = {
                        "L1_workload_balance": self.accumulated_train_agent_workload.get(agent_id, 0.0) / self.fairness_episode_count_train,
                        "action_count": self.accumulated_train_agent_tasks.get(agent_id, 0) / self.fairness_episode_count_train,
                        "workload_percentage": (self.accumulated_train_agent_workload.get(agent_id, 0.0) / self.fairness_episode_count_train) * 100
                    }
                
                self._log_fairness_metrics(avg_train_fairness, avg_agent_contributions, log_prefix)            
                    
                # RESET ACCUMULATORS
                self.accumulated_train_L1 = 0.0
                self.accumulated_train_L2 = 0.0
                self.accumulated_train_L3 = 0.0
                self.fairness_episode_count_train = 0    
                # RESET AGENT ACCUMULATORS TOO!
                for agent_id in list(self.accumulated_train_agent_tasks.keys()):
                    self.accumulated_train_agent_tasks[agent_id] = 0
                    self.accumulated_train_agent_workload[agent_id] = 0.0
            
            # Log the maximum possible goals for training
            self.logger.log_stat(log_prefix + "max_possible_goals", self.train_episode_count, self.t_env)
            # Log cumulative goal success rate for training
            self.logger.log_stat(log_prefix + "total_goals_reached", self.total_train_goals_reached, self.t_env)
            self.logger.log_stat(log_prefix + "total_episodes", self.total_train_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_goal_success_rate", 
                                self.total_train_goals_reached / max(1, self.total_train_episodes), self.t_env)
            
            # Log cumulative milestone statistics for training
            self._log_cumulative_milestones(test_mode, log_prefix)
            
            # Reset the train_episode_count for the next interval
            self.train_episode_count = 0
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env
                )
            self.log_train_stats_t = self.t_env

        return self.batch

    def update_best_actions(self):
        for idx, parent_conn in enumerate(self.parent_conns):
            if self.episode_returns[idx] > self.best_episode_reward:
                parent_conn.send(("get_episode_actions", None))
                episode_actions = parent_conn.recv()
                self.best_episode_reward = self.episode_returns[idx]
                self.best_actions = episode_actions

    def _log(self, returns, stats, goals, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        self.logger.log_stat(
            prefix + "best_episode_reward", self.best_episode_reward, self.t_env
        )

        # Log goal statistics
        if goals:
            self.logger.log_stat(prefix + "goals_reached", sum(goals), self.t_env)
            self.logger.log_stat(prefix + "goal_success_rate", np.mean(goals), self.t_env)
        
        returns.clear()
        goals.clear()  

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(
                    prefix + k + "_mean", v / stats["n_episodes"], self.t_env
                )
        stats.clear()

    def update_best_test_actions(self):
        for idx, parent_conn in enumerate(self.parent_conns):
            if self.episode_returns[idx] > getattr(self, 'best_test_episode_reward', 0):
                parent_conn.send(("get_episode_actions", None))
                episode_actions = parent_conn.recv()
                self.best_test_episode_reward = self.episode_returns[idx]
                self.best_test_actions = episode_actions

    def save_best_test_actions(self, save_path):
        with open(os.path.join(save_path, "test_best_actions.txt"), "w") as f:
            for action in getattr(self, 'best_test_actions', []):
                f.write(f"{action}\n")

    def save_best_actions(self, save_path):
        with open(os.path.join(save_path, "best_actions.txt"), "w") as f:
            for action in self.best_actions:
                f.write(f"{action}\n")

    def _track_episode_fairness(self, env_idx, episode_data, test_mode=False):
        """Track fairness metrics for a completed episode"""
        
        # Create detailed episode record
        episode_record = {
            "env_id": env_idx,
            "episode_id": self.env_episode_counters[env_idx],
            "timestep": self.t_env,
            "test_mode": test_mode,
            "episode_length": episode_data.get("length", 0),
            "return": episode_data.get("return", 0),
            "goal_reached": episode_data.get("goal_reached", 0),
            "fairness_L1": episode_data.get("L1", 0),
            "fairness_L2": episode_data.get("L2", 0),
            "fairness_L3": episode_data.get("L3", 0),
            "agent_workloads": episode_data.get("agent_workloads", {}),
            "agent_skills_used": episode_data.get("agent_skills_used", {}),
            "timestamp": time.time()  # Real-world timestamp
        }
        
        # Store detailed record
        self.detailed_episode_fairness.append(episode_record)
        
        # Update per-environment tracking
        self.env_episode_fairness[env_idx].append(episode_record)
        self.env_episode_counters[env_idx] += 1
        
        # Update timestep-based tracking
        mode = "test" if test_mode else "train"
        self.fairness_by_timestep[mode]["timesteps"].append(self.t_env)
        self.fairness_by_timestep[mode]["L1"].append(episode_data.get("L1", 0))
        self.fairness_by_timestep[mode]["L2"].append(episode_data.get("L2", 0))
        self.fairness_by_timestep[mode]["L3"].append(episode_data.get("L3", 0))

    def save_fairness_analysis(self, save_path):
        """Save detailed fairness analysis to files"""
        import json
        try:
            import pandas as pd
            has_pandas = True
        except ImportError:
            has_pandas = False
            print("[WARNING] pandas not available. Saving JSON only.")
        
        # Save detailed episode records as JSON
        with open(os.path.join(save_path, "detailed_fairness_episodes.json"), "w") as f:
            json.dump(self.detailed_episode_fairness, f, indent=2)
        
        # Create DataFrame for easier analysis if pandas is available
        if has_pandas and self.detailed_episode_fairness:
            df = pd.DataFrame(self.detailed_episode_fairness)
            
            # Save as CSV for analysis
            df.to_csv(os.path.join(save_path, "fairness_episodes.csv"), index=False)
            
            # Create summary statistics by environment
            env_summary = df.groupby('env_id').agg({
                'fairness_L1': ['mean', 'std', 'min', 'max'],
                'fairness_L2': ['mean', 'std', 'min', 'max'],
                'fairness_L3': ['mean', 'std', 'min', 'max'],
                'return': 'mean',
                'goal_reached': 'mean',
                'episode_length': 'mean'
            })
            env_summary.to_csv(os.path.join(save_path, "fairness_by_environment.csv"))
            
            # Create rolling window analysis
            window_size = 10
            rolling_fairness = df.groupby('test_mode').apply(
                lambda x: x.sort_values('timestep')[['fairness_L1', 'fairness_L2', 'fairness_L3']]
                .rolling(window=window_size, min_periods=1)
                .mean()
            )
            rolling_fairness.to_csv(os.path.join(save_path, "fairness_rolling_average.csv"))
            
        # Save timestep-based tracking
        for mode in ["train", "test"]:
            if self.fairness_by_timestep[mode]["timesteps"]:
                if has_pandas:
                    timestep_df = pd.DataFrame(self.fairness_by_timestep[mode])
                    timestep_df.to_csv(os.path.join(save_path, f"fairness_by_timestep_{mode}.csv"), index=False)
                else:
                    # Save as JSON if pandas not available
                    with open(os.path.join(save_path, f"fairness_by_timestep_{mode}.json"), "w") as f:
                        json.dump(self.fairness_by_timestep[mode], f, indent=2)

    def log_fairness_summary(self, log_prefix=""):
        """Log comprehensive fairness summary"""
        if not self.detailed_episode_fairness:
            return
        
        try:
            import pandas as pd
            has_pandas = True
        except ImportError:
            has_pandas = False
            return  # Skip if pandas not available
        
        df = pd.DataFrame(self.detailed_episode_fairness)
        
        # Filter by mode
        is_test = log_prefix.startswith("test_")
        mode_df = df[df['test_mode'] == is_test]
        
        if len(mode_df) == 0:
            return
        
        # Log overall statistics
        self.logger.log_stat(f"{log_prefix}fairness/episodes_analyzed", len(mode_df), self.t_env)
        
        # Log percentile statistics for better understanding of distribution
        for metric in ['fairness_L1', 'fairness_L2', 'fairness_L3']:
            self.logger.log_stat(f"{log_prefix}{metric}_p25", mode_df[metric].quantile(0.25), self.t_env)
            self.logger.log_stat(f"{log_prefix}{metric}_p50", mode_df[metric].quantile(0.50), self.t_env)
            self.logger.log_stat(f"{log_prefix}{metric}_p75", mode_df[metric].quantile(0.75), self.t_env)
            self.logger.log_stat(f"{log_prefix}{metric}_p90", mode_df[metric].quantile(0.90), self.t_env)
        
        # Log per-environment variance to detect if some envs are consistently unfair
        env_variance = mode_df.groupby('env_id')['fairness_L3'].var().mean()
        self.logger.log_stat(f"{log_prefix}fairness/env_variance", env_variance, self.t_env)
        
        # Log correlation between fairness and performance
        if 'return' in mode_df.columns:
            corr_L3_return = mode_df['fairness_L3'].corr(mode_df['return'])
            self.logger.log_stat(f"{log_prefix}fairness/L3_return_correlation", corr_L3_return, self.t_env)


# MOVE env_worker OUTSIDE THE CLASS
def env_worker(remote, env_fn, full_env_args=None):
    # Make environment - KEEP THE WORKING APPROACH
    env = deepcopy(env_REGISTRY["hosp_env"])  # ✅ This works!
    #env = env_fn(**env_arg) 
    
    # Set tracking directory if provided
    if full_env_args and "tracking_dir" in full_env_args:
        tracking_dir = full_env_args["tracking_dir"]
        if tracking_dir and hasattr(env, 'pddl_env'):
            env.pddl_env._log_dir = tracking_dir
            # print(f"[Worker] Set tracking directory to: {tracking_dir}")
            pass
    
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send(
                {
                    # Data for the next timestep needed to pick an action
                    "state": state,
                    "avail_actions": avail_actions,
                    "obs": obs,
                    # Rest of the data for the current timestep
                    "reward": reward,
                    "terminated": terminated,
                    "info": env_info,
                }
            )
        elif cmd == "reset":
            reset_data = data if data else {}
            test_mode = reset_data.get("test_mode", False)
            
            # Reset environment
            env.reset()
            
            # Set test mode and episode counter in wrapper
            if hasattr(env, 'pddl_env'):
                env.pddl_env._log_test_mode = test_mode
                # The episode counter will be set when save_current_episode_log_with_id is called
                
            remote.send(
                {
                    "state": env.get_state(),
                    "avail_actions": env.get_avail_actions(),
                    "obs": env.get_obs(),
                }
            )
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "get_episode_goal_reached":
            # Check if the environment has the method to get goal status
            if hasattr(env, 'get_episode_goal_reached'):
                goal_reached = env.get_episode_goal_reached()
                remote.send(goal_reached)
            else:
                # Fallback: If the method doesn't exist, assume goal not reached
                remote.send(0)
        elif cmd == "get_milestone_status":
            # Get milestone status from reward handler
            if hasattr(env, 'reward_handler') and hasattr(env.reward_handler, 'get_milestone_status'):
                milestone_data = env.reward_handler.get_milestone_status()
                remote.send(milestone_data)
            elif hasattr(env, '_env') and hasattr(env._env, 'reward_handler') and hasattr(env._env.reward_handler, 'get_milestone_status'):
                # Try accessing through _env wrapper
                milestone_data = env._env.reward_handler.get_milestone_status()
                remote.send(milestone_data)
            elif hasattr(env, 'pddl_env') and hasattr(env.pddl_env, 'reward_handler') and hasattr(env.pddl_env.reward_handler, 'get_milestone_status'):
                # Try accessing through pddl_env wrapper (LIKELY LOCATION)
                milestone_data = env.pddl_env.reward_handler.get_milestone_status()
                remote.send(milestone_data)
            elif hasattr(env, 'env') and hasattr(env.env, 'reward_handler') and hasattr(env.env.reward_handler, 'get_milestone_status'):
                # Try accessing through env wrapper
                milestone_data = env.env.reward_handler.get_milestone_status()
                remote.send(milestone_data)
            else:
                # Fallback: Return empty dict if milestone tracking not available
                remote.send({})
        elif cmd == "get_fairness_metrics":
            # Try wrapper first
            if hasattr(env, 'pddl_env') and hasattr(env.pddl_env, 'get_last_fairness_metrics'):
                wrapper_metrics = env.pddl_env.get_last_fairness_metrics()
                if wrapper_metrics:
                    remote.send({"metrics": wrapper_metrics, "contributions": {}})
                    continue
            
            # Then fallback to computing metrics
            if hasattr(env, 'pddl_env'):
                try:
                    from utils.fairness_metrics_cal import compute_L1, compute_L2, compute_L3
                    
                    # GET NUMBER OF PLAYERS
                    num_players = env.pddl_env.num_players
                    
                    # INITIALIZE TRACKING VARIABLES
                    agent_counts = {i: 0 for i in range(num_players)}
                    critical_types = {"compresschest", "giverescuebreaths", "giveshock", "givemedicine", "stack", "stackunder", "moveitem"}
                    treatment_tasks = {"compresschest", "giverescuebreaths", "giveshock", "givemedicine"}
                    
                    # COLLECT AGENT SKILLS FROM CONFIG
                    all_agents_skills = {}
                    config = env.pddl_env.config
                    
                    for agent_id in range(num_players):
                        player_key = f"robot{agent_id+1}"
                        agent_skills = {}
                        if "player_info" in config and player_key in config["player_info"]:
                            skill_info = config["player_info"][player_key]
                            # Include both medical and moveitem skills
                            for task in treatment_tasks:
                                agent_skills[task] = skill_info.get(task, 1.0)
                            # Add moveitem skill
                            agent_skills["moveitem"] = skill_info.get("moveitem", 1.0)
                        else:
                            for task in treatment_tasks:
                                agent_skills[task] = 1.0
                            agent_skills["moveitem"] = 1.0
                        all_agents_skills[agent_id] = agent_skills
                    
                    # COLLECT TASK-SKILL PAIRS FOR L2
                    task_skill_pairs = []
                    
                    if hasattr(env.pddl_env, '_action_log'):
                        # COUNT CRITICAL ACTIONS
                        for entry in env.pddl_env._action_log:
                            action_str = entry.get("action", "")
                            agent_id = entry["agent_id"]
                            
                            # CHECK CRITICAL ACTIONS FOR L1
                            for crit in critical_types:
                                if action_str.startswith(crit) or (crit == "stackunder" and "under" in action_str.lower()):
                                    agent_counts[agent_id] += 1
                                    break
                            
                            # COLLECT TREATMENT TASKS FOR L2
                            for task in treatment_tasks:
                                if action_str.startswith(task):
                                    skill = all_agents_skills[agent_id].get(task, 1.0)
                                    task_skill_pairs.append((task, agent_id, skill))
                                    break
                            
                            # Also check for moveitem tasks
                            if action_str.startswith("moveitem"):
                                skill = all_agents_skills[agent_id].get("moveitem", 1.0)
                                task_skill_pairs.append(("moveitem", agent_id, skill))
                        
                        # CONVERT TO LISTS
                        agent_counts_list = [agent_counts[i] for i in range(num_players)]
                        
                        # COMPUTE FAIRNESS METRICS HERE!!!
                        L1 = compute_L1(agent_counts_list) if sum(agent_counts_list) > 0 else 0.0
                        L2 = compute_L2(task_skill_pairs, all_agents_skills) if task_skill_pairs else 0.0
                        L3 = compute_L3(L1, L2, alpha=0.5)
                        
                        # NOW CREATE THE DICTIONARY AFTER COMPUTING L1, L2, L3
                        episode_fairness_metrics = {
                            "L1": L1,
                            "L2": L2,
                            "L3": L3
                        }
                        
                        # COMPUTE AGENT CONTRIBUTIONS
                        total_actions = sum(agent_counts_list)
                        episode_agent_contributions = None
                        if total_actions > 0:
                            episode_agent_contributions = {
                                i: {
                                    "L1_workload_balance": agent_counts[i] / total_actions,
                                    "workload_percentage": (agent_counts[i] / total_actions) * 100,
                                    "action_count": agent_counts[i]
                                } for i in range(num_players)
                            }
                        
                        remote.send({
                            "metrics": episode_fairness_metrics,
                            "contributions": episode_agent_contributions
                        })
                    else:
                        remote.send({})
                except Exception as e:
                    print(f"[WARNING] Could not compute fairness metrics in worker: {e}")
                    remote.send({})
            else:
                remote.send({})
        elif cmd == "get_num_agents":
            # Get number of agents from environment
            if hasattr(env, 'pddl_env') and hasattr(env.pddl_env, 'num_players'):
                remote.send(env.pddl_env.num_players)
            else:
                # Default to 3 agents if not available
                remote.send(3)
        elif cmd == "save_current_episode_log_with_id":
            episode_data = data
            episode_id = episode_data["episode_id"] if isinstance(episode_data, dict) else episode_data
            test_mode = episode_data.get("test_mode", False) if isinstance(episode_data, dict) else False
            
            if hasattr(env, "pddl_env") and hasattr(env.pddl_env, "save_current_episode_log"):
                env.pddl_env._episode_counter = episode_id
                env.pddl_env._log_test_mode = test_mode
                env.pddl_env.save_current_episode_log()
            remote.send(True)
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        elif cmd == "get_episode_actions":
            remote.send(env.get_episode_actions())
        elif cmd == "save_current_episode_log":
            if hasattr(env, "pddl_env") and hasattr(env.pddl_env, "save_current_episode_log"):
                env.pddl_env.save_current_episode_log()
            remote.send(True)  # Acknowledge
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)