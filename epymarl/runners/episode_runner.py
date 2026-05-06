# Fixed episode_runner.py with full task-skill alignment metrics support

import os
import sys
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # Apply seed with same deterministic approach as in parallel_runner
        if "seed" in self.args.env_args:
            seed = self.args.env_args.get("seed", 123)
            if seed is None:
                seed = 123
                print("[WARNING] No seed provided. Using default seed: 123")
            self.args.env_args["seed"] = int(seed)
            print(f"[INFO] Setting environment seed: {self.args.env_args['seed']}")

        # Create environment ONCE
        self.env = env_REGISTRY["hosp_env"](**self.args.env_args) if callable(env_REGISTRY["hosp_env"]) else env_REGISTRY["hosp_env"]

        # Extract tracking directory from env_args
        self.tracking_dir = self.args.env_args.get("tracking_dir", None)

        # Ensure the wrapper gets the tracking directory
        if self.tracking_dir and hasattr(self.env, 'pddl_env'):
            self.env.pddl_env._log_dir = self.tracking_dir

        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.best_episode_reward = 0
        self.best_actions = []

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.train_goals = []
        self.test_goals = []
        self.train_episode_count = 0
        self.test_episode_count = 0
        self.best_test_episode_reward = 0
        self.best_test_actions = []

        # Add episode ID counters
        self.train_episode_id = 0
        self.test_episode_id = 0

        # Initialize episode counter for logging
        self._episode_counter = 0

        # Add tracking for cumulative goals across all training/testing
        self.total_train_goals_reached = 0
        self.total_train_episodes = 0
        self.total_test_goals_reached = 0
        self.total_test_episodes = 0

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

        # Cumulative correct stacking tracking
        self.total_train_correct_stacking = 0
        self.total_test_correct_stacking = 0

        # FAIRNESS TRACKING
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
        self.accumulated_train_agent_tasks = {i: 0 for i in range(self.num_agents)}
        self.accumulated_test_agent_tasks = {i: 0 for i in range(self.num_agents)}
        self.accumulated_train_agent_workload = {i: 0.0 for i in range(self.num_agents)}
        self.accumulated_test_agent_workload = {i: 0.0 for i in range(self.num_agents)}

        # PER-AGENT FAIRNESS HISTORY
        self.agent_workload_history = {i: [] for i in range(self.num_agents)}
        self.agent_skill_alignment_history = {i: [] for i in range(self.num_agents)}

        # FEN-specific tracking
        self.fen_utilities_history = []
        self.fen_avg_utilities_history = []
        self.current_reward_type = None
        self.fen_step_counts = []

        # FEN agent action tracking
        self.fen_agent_action_counts = {}
        self.fen_total_actions_count = 0

        # Additional metrics
        self.fairness_history = {'L1': [], 'L2': [], 'L3': []}
        self.alignment_history = []
        self.train_contribution_variances = []
        self.test_contribution_variances = []
        self.train_specialist_utilizations = []
        self.test_specialist_utilizations = []
        self.train_overall_alignments = []
        self.test_overall_alignments = []
        self.train_milestone_entropies = []
        self.test_milestone_entropies = []
        self.train_episode_lengths = []
        self.test_episode_lengths = []
        self.train_workload_ranges = []
        self.test_workload_ranges = []

        # Debug tracking for missing metrics
        self.debug_missing_metrics = set()

        # Log the first run
        self.log_train_stats_t = -1000000

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

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self, test_mode=False):
        self.batch = self.new_batch()

        # Reset environment with test_mode flag if supported
        if hasattr(self.env, 'reset'):
            try:
                obs = self.env.reset(test_mode=test_mode)
            except TypeError:
                obs = self.env.reset()
        else:
            obs = self.env.reset()

        # Ensure wrapper knows about test mode and episode ID
        if hasattr(self.env, 'pddl_env'):
            self.env.pddl_env._log_test_mode = test_mode
            if test_mode:
                self.env.pddl_env._episode_counter = self.test_episode_id
            else:
                self.env.pddl_env._episode_counter = self.train_episode_id

            # Ensure log directory is still set
            if self.tracking_dir and self.env.pddl_env._log_dir != self.tracking_dir:
                self.env.pddl_env._log_dir = self.tracking_dir

        self.t = 0
        self.episode_goal_achieved = False

        # For logging in test episodes
        self._log_test_mode = test_mode
        if self._log_test_mode:
            self._episode_counter += 1

    def _update_cumulative_milestones(self, milestone_data, test_mode):
        """Update cumulative milestone counters based on milestone data"""
        if test_mode:
            if milestone_data.get("chest_compressed", 0) == 1:
                self.total_test_chest_compressed += 1
            if milestone_data.get("rescue_breaths", 0) == 1:
                self.total_test_rescue_breaths += 1
            if milestone_data.get("shock", 0) == 1:
                self.total_test_shock += 1
            if milestone_data.get("medicine_administered", 0) == 1:
                self.total_test_medicine_administered += 1
            if milestone_data.get("correct_stacking", 0) == 1:
                self.total_test_correct_stacking += 1
        else:
            if milestone_data.get("chest_compressed", 0) == 1:
                self.total_train_chest_compressed += 1
            if milestone_data.get("rescue_breaths", 0) == 1:
                self.total_train_rescue_breaths += 1
            if milestone_data.get("shock", 0) == 1:
                self.total_train_shock += 1
            if milestone_data.get("medicine_administered", 0) == 1:
                self.total_train_medicine_administered += 1
            if milestone_data.get("correct_stacking", 0) == 1:
                self.total_train_correct_stacking += 1

    def _extract_skill_alignment_metrics(self, info, test_mode):
        """Extract and compute skill alignment metrics with fallback calculations"""
        # Debug logging
        if info and isinstance(info, dict):
            available_keys = set(info.keys())
            expected_keys = {'fairness_metrics', 'skill_task_alignment', 'normalized_contributions', 
                           'agent_action_counts', 'agent_skills', 'agent_medical_events'}
            missing = expected_keys - available_keys
            if missing and missing != self.debug_missing_metrics:
                self.debug_missing_metrics = missing
                # print(f"[DEBUG] Info keys available: {sorted(available_keys)}")
                # print(f"[DEBUG] Missing expected keys: {sorted(missing)}")
        
        # Try to extract fairness metrics
        if 'fairness_metrics' in info:
            fairness_data = info['fairness_metrics']
            self.fairness_history['L1'].append(fairness_data.get('L1_workload_imbalance', 0))
            self.fairness_history['L2'].append(fairness_data.get('L2_skill_misalignment', 0))
            self.fairness_history['L3'].append(fairness_data.get('L3_composite', 0))
            
            if 'workload_range' in fairness_data:
                workload_ranges = self.test_workload_ranges if test_mode else self.train_workload_ranges
                workload_ranges.append(fairness_data['workload_range'])
        
        # Extract or compute contribution variance
        if 'normalized_contributions' in info:
            contribs = info['normalized_contributions']
            if isinstance(contribs, (list, np.ndarray)) and len(contribs) > 0:
                contrib_variance = float(np.var(contribs))
                contribution_variances = self.test_contribution_variances if test_mode else self.train_contribution_variances
                contribution_variances.append(contrib_variance)
        elif 'agent_action_counts' in info:
            # Fallback: compute from action counts
            action_counts = info['agent_action_counts']
            if isinstance(action_counts, dict):
                counts = list(action_counts.values())
                total = sum(counts)
                if total > 0:
                    normalized = [c/total for c in counts]
                    contrib_variance = float(np.var(normalized))
                    contribution_variances = self.test_contribution_variances if test_mode else self.train_contribution_variances
                    contribution_variances.append(contrib_variance)
        
        # Extract skill-task alignment metrics
        skill_alignment_found = False
        
        # Try primary location
        if 'skill_task_alignment' in info:
            alignment_data = info['skill_task_alignment']
            skill_alignment_found = True
            
            # Overall alignment score
            if 'overall_alignment_score' in alignment_data:
                score = float(alignment_data['overall_alignment_score'])
                overall_alignments = self.test_overall_alignments if test_mode else self.train_overall_alignments
                overall_alignments.append(score)
                self.alignment_history.append(score)
            
            # Per-agent alignment for specialist utilization
            per_agent = None
            if 'per_agent_alignment' in alignment_data and isinstance(alignment_data['per_agent_alignment'], dict):
                per_agent = alignment_data['per_agent_alignment']
            elif 'alignment_scores' in alignment_data and isinstance(alignment_data['alignment_scores'], dict):
                per_agent = alignment_data['alignment_scores']
            
            if per_agent:
                specialist_rates = [float(score) for score in per_agent.values() if score is not None]
                if specialist_rates:
                    avg_specialist_utilization = float(np.mean(specialist_rates))
                    specialist_utilizations = self.test_specialist_utilizations if test_mode else self.train_specialist_utilizations
                    specialist_utilizations.append(avg_specialist_utilization)
        
        # Fallback: Try alternative locations
        if not skill_alignment_found:
            # Check if metrics are at top level
            if 'overall_alignment_score' in info:
                score = float(info['overall_alignment_score'])
                overall_alignments = self.test_overall_alignments if test_mode else self.train_overall_alignments
                overall_alignments.append(score)
                self.alignment_history.append(score)
                skill_alignment_found = True
            
            # Check for L2_skill_misalignment as a proxy
            if 'L2_skill_misalignment' in info:
                # Convert misalignment to alignment (1 - misalignment)
                misalignment = float(info['L2_skill_misalignment'])
                alignment = max(0.0, 1.0 - misalignment)
                overall_alignments = self.test_overall_alignments if test_mode else self.train_overall_alignments
                overall_alignments.append(alignment)
                skill_alignment_found = True
        
        # Compute specialist utilization from agent skills if available
        if not skill_alignment_found and 'agent_skills' in info and 'agent_action_counts' in info:
            agent_skills = info['agent_skills']
            action_counts = info['agent_action_counts']
            if isinstance(agent_skills, dict) and isinstance(action_counts, dict):
                # Simple specialist utilization: agents with high skills should have proportional actions
                utilizations = []
                for agent_id in agent_skills:
                    if agent_id in action_counts:
                        skill = float(agent_skills[agent_id])
                        actions = float(action_counts[agent_id])
                        total_actions = sum(action_counts.values())
                        if total_actions > 0 and skill > 0:
                            expected_ratio = skill / sum(agent_skills.values())
                            actual_ratio = actions / total_actions
                            utilization = min(1.0, actual_ratio / expected_ratio) if expected_ratio > 0 else 0
                            utilizations.append(utilization)
                
                if utilizations:
                    avg_utilization = float(np.mean(utilizations))
                    specialist_utilizations = self.test_specialist_utilizations if test_mode else self.train_specialist_utilizations
                    specialist_utilizations.append(avg_utilization)
                    
                    # Also compute a simple alignment score
                    alignment_score = avg_utilization  # Simple proxy
                    overall_alignments = self.test_overall_alignments if test_mode else self.train_overall_alignments
                    overall_alignments.append(alignment_score)
        
        # Medical milestone distribution entropy (existing code works)
        if 'agent_medical_events' in info:
            medical_events = info['agent_medical_events']
            agent_milestone_counts = {}
            for event, agent_id in medical_events.items():
                if agent_id is not None:
                    agent_milestone_counts[agent_id] = agent_milestone_counts.get(agent_id, 0) + 1
            
            if agent_milestone_counts:
                n_agents = self.num_agents
                counts = [0] * n_agents
                for agent_id, count in agent_milestone_counts.items():
                    if 0 <= agent_id < n_agents:
                        counts[agent_id] = count
                total = sum(counts)
                if total > 0:
                    probs = [c / total for c in counts]
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                else:
                    entropy = 0.0
                milestone_entropies = self.test_milestone_entropies if test_mode else self.train_milestone_entropies
                milestone_entropies.append(float(entropy))

    def run(self, test_mode=False):
        self.reset(test_mode=test_mode)
        episode_goal_reached = 0

        terminated = False
        episode_fairness_metrics = None
        episode_agent_contributions = None
        episode_return = 0

        # FEN tracking for this episode
        episode_fen_utilities = []
        episode_fen_avg_utilities = []
        episode_fen_step_count = 0
        episode_fen_agent_actions = {}
        episode_fen_total_actions = 0

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)
            action_outputs = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
            )
            if isinstance(action_outputs, tuple):
                actions, selected_roles, role_avail_actions = action_outputs
            else:
                actions = action_outputs
                selected_roles = None
                role_avail_actions = None

            reward, terminated, env_info = self.env.step(actions[0])
            # FEN-specific info
            if env_info:
                if self.current_reward_type is None and "reward_type" in env_info:
                    self.current_reward_type = env_info["reward_type"]
                    if self.current_reward_type == "fen":
                        print(f"[INFO] Using reward type: {self.current_reward_type}")

                if "fen_utilities" in env_info:
                    episode_fen_utilities.append(env_info["fen_utilities"].copy())

                if "fen_avg_utility" in env_info:
                    episode_fen_avg_utilities.append(env_info["fen_avg_utility"])

                if "fen_step_count" in env_info:
                    episode_fen_step_count = env_info["fen_step_count"]

                if "agent_id" in env_info and self.current_reward_type == "fen":
                    agent_id = env_info["agent_id"]
                    episode_fen_agent_actions[agent_id] = episode_fen_agent_actions.get(agent_id, 0) + 1
                    episode_fen_total_actions += 1

            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if selected_roles is not None:
                post_transition_data["roles"] = selected_roles.unsqueeze(-1)
                post_transition_data["role_avail_actions"] = role_avail_actions

            self.batch.update(post_transition_data, ts=self.t)

            # Check for goal achievement after each step
            if hasattr(self.env, 'get_episode_goal_reached'):
                if self.env.get_episode_goal_reached():
                    self.episode_goal_achieved = True

            self.t += 1

        if hasattr(self.env, 'pddl_env') and hasattr(self.env.pddl_env, 'get_episode_actions') and (self.train_episode_id % 100 == 0 or test_mode):
            episode_actions = self.env.pddl_env.get_episode_actions()
            if episode_actions:
                log_prefix = "test_" if test_mode else "train_"
                episode_id = self.test_episode_id if test_mode else self.train_episode_id
                # print(f"[{log_prefix}Episode {episode_id}] Actions: {episode_actions}")  # silenced for cleaner output
        # Record episode length
        episode_lengths = self.test_episode_lengths if test_mode else self.train_episode_lengths
        episode_lengths.append(self.t)

        # On termination, pull fairness & alignment info and accumulate
        if terminated and hasattr(self.env, 'pddl_env'):
            try:
                # Prefer wrapper-provided fairness metrics
                if hasattr(self.env.pddl_env, 'get_last_fairness_metrics'):
                    wrapper_metrics = self.env.pddl_env.get_last_fairness_metrics()
                    if wrapper_metrics:
                        episode_fairness_metrics = wrapper_metrics

                        # Extract normalized contributions for agent workload
                        if "normalized_contributions" in wrapper_metrics:
                            episode_agent_contributions = {}
                            num_agents = len(wrapper_metrics["normalized_contributions"])

                            # Ensure num_agents seen by runner is correct
                            self.num_agents = num_agents
                            for i in range(num_agents):
                                if i not in self.accumulated_train_agent_tasks:
                                    self.accumulated_train_agent_tasks[i] = 0
                                    self.accumulated_train_agent_workload[i] = 0.0
                                if i not in self.accumulated_test_agent_tasks:
                                    self.accumulated_test_agent_tasks[i] = 0
                                    self.accumulated_test_agent_workload[i] = 0.0

                            for i, contrib in enumerate(wrapper_metrics["normalized_contributions"]):
                                episode_agent_contributions[i] = {
                                    "workload_percentage": contrib * 100,
                                    "L1_workload_balance": float(contrib),
                                    "action_count": wrapper_metrics.get("agent_action_counts", {}).get(i, 0)
                                }
                        else:
                            episode_agent_contributions = None
                    else:
                        episode_fairness_metrics = None
                        episode_agent_contributions = None

                # Extract additional metrics from wrapper/env info
                info = None
                try:
                    # Try multiple methods to get info
                    if hasattr(self.env, 'get_latest_info'):
                        info = self.env.get_latest_info()
                    elif hasattr(self.env, 'pddl_env') and hasattr(self.env.pddl_env, 'get_latest_info'):
                        info = self.env.pddl_env.get_latest_info()
                    elif hasattr(self.env, 'get_info'):
                        info = self.env.get_info()
                    elif hasattr(self.env, '_get_info'):
                        info = self.env._get_info()
                except Exception as e:
                    print(f"[WARNING] Could not get info from environment: {e}")
                    info = None

                # Extract skill alignment metrics with fallback calculations
                if info and isinstance(info, dict):
                    self._extract_skill_alignment_metrics(info, test_mode)

                    # Enrich per-agent contributions with L2 if available
                    if episode_agent_contributions and 'skill_task_alignment' in info:
                        alignment_data = info['skill_task_alignment']
                        per_agent = alignment_data.get('per_agent_alignment', alignment_data.get('alignment_scores', {}))
                        if per_agent:
                            for aid, score in per_agent.items():
                                if aid in episode_agent_contributions and score is not None:
                                    episode_agent_contributions[aid]['L2_skill_alignment'] = float(score)

                # Accumulate fairness metrics
                if episode_fairness_metrics:
                    if test_mode:
                        self.accumulated_test_L1 += float(episode_fairness_metrics.get("L1", 0.0))
                        self.accumulated_test_L2 += float(episode_fairness_metrics.get("L2", 0.0))
                        self.accumulated_test_L3 += float(episode_fairness_metrics.get("L3", 0.0))
                        self.fairness_episode_count_test += 1
                    else:
                        self.accumulated_train_L1 += float(episode_fairness_metrics.get("L1", 0.0))
                        self.accumulated_train_L2 += float(episode_fairness_metrics.get("L2", 0.0))
                        self.accumulated_train_L3 += float(episode_fairness_metrics.get("L3", 0.0))
                        self.fairness_episode_count_train += 1

                # Accumulate per-agent data
                if episode_agent_contributions:
                    for agent_id, contrib in episode_agent_contributions.items():
                        if test_mode:
                            self.accumulated_test_agent_tasks[agent_id] += int(contrib.get("action_count", 0))
                            self.accumulated_test_agent_workload[agent_id] += float(contrib.get("L1_workload_balance", 0.0))
                        else:
                            self.accumulated_train_agent_tasks[agent_id] += int(contrib.get("action_count", 0))
                            self.accumulated_train_agent_workload[agent_id] += float(contrib.get("L1_workload_balance", 0.0))

            except Exception as e:
                print(f"[WARNING] Error extracting metrics: {e}")
                import traceback
                traceback.print_exc()

        # UPDATE EPISODE COUNTERS
        if test_mode:
            self.test_episode_count += 1
            self.total_test_episodes += 1
            self.test_episode_id += 1
        else:
            self.train_episode_count += 1
            self.total_train_episodes += 1
            self.train_episode_id += 1

        if terminated and test_mode and hasattr(self.env, 'pddl_env') and hasattr(self.env.pddl_env, 'save_current_episode_log'):
            self.env.pddl_env.save_current_episode_log()

        # Assign goal value AFTER episode ends
        episode_goal_reached = 0
        log_prefix = "test_" if test_mode else ""

        if hasattr(self.env, 'get_episode_goal_reached') and terminated:
            current_goal_status = self.env.get_episode_goal_reached()
            episode_goal_reached = 1 if (self.episode_goal_achieved or current_goal_status) else 0

            # Add milestone logging HERE after episode completion
            milestone_data = None

            if hasattr(self.env, 'pddl_env') and hasattr(self.env.pddl_env, 'reward_handler'):
                if hasattr(self.env.pddl_env.reward_handler, 'get_milestone_status'):
                    milestone_data = self.env.pddl_env.reward_handler.get_milestone_status()
            elif hasattr(self.env, 'reward_handler') and hasattr(self.env.reward_handler, 'get_milestone_status'):
                milestone_data = self.env.reward_handler.get_milestone_status()
            elif hasattr(self.env, '_env') and hasattr(self.env._env, 'reward_handler'):
                if hasattr(self.env._env.reward_handler, 'get_milestone_status'):
                    milestone_data = self.env._env.reward_handler.get_milestone_status()
            elif hasattr(self.env, 'env') and hasattr(self.env.env, 'reward_handler'):
                if hasattr(self.env.env.reward_handler, 'get_milestone_status'):
                    milestone_data = self.env.env.reward_handler.get_milestone_status()

            if milestone_data:
                self._update_cumulative_milestones(milestone_data, test_mode)

        # Update cumulative counters AFTER computing goal status
        if test_mode:
            self.total_test_goals_reached += episode_goal_reached
        else:
            self.total_train_goals_reached += episode_goal_reached

        # Store goal result after episode ends
        cur_goals = self.test_goals if test_mode else self.train_goals
        cur_goals.append(episode_goal_reached)

        # Store FEN data for this episode
        if episode_fen_utilities:
            self.fen_utilities_history.extend(episode_fen_utilities)
        if episode_fen_avg_utilities:
            self.fen_avg_utilities_history.extend(episode_fen_avg_utilities)
        if episode_fen_step_count > 0:
            self.fen_step_counts.append(episode_fen_step_count)

        # Store agent action data for percentage calculations
        for agent_id, count in episode_fen_agent_actions.items():
            self.fen_agent_action_counts[agent_id] = self.fen_agent_action_counts.get(agent_id, 0) + count
        self.fen_total_actions_count += episode_fen_total_actions

        # Handle final batch update
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.episode_return = episode_return
        if test_mode and self.args.render:
            pass
        self.update_best_test_actions()

        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        action_outputs = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode
        )
        if isinstance(action_outputs, tuple):
            actions = action_outputs[0]
        else:
            actions = action_outputs
        self.batch.update({"actions": actions}, ts=self.t)

        # Update statistics
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        # Note: env_info refers to last step's info; safe union update
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # Logging
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, cur_goals, log_prefix)

            # Averaged fairness metrics for test episodes
            if self.fairness_episode_count_test > 0:
                avg_test_fairness = {
                    "L1": self.accumulated_test_L1 / self.fairness_episode_count_test,
                    "L2": self.accumulated_test_L2 / self.fairness_episode_count_test,
                    "L3": self.accumulated_test_L3 / self.fairness_episode_count_test
                }

                # Averaged agent contributions
                avg_agent_contributions = {}
                num_agents = len(self.accumulated_test_agent_tasks)
                for agent_id in range(num_agents):
                    avg_agent_contributions[agent_id] = {
                        "L1_workload_balance": self.accumulated_test_agent_workload[agent_id] / self.fairness_episode_count_test,
                        "action_count": self.accumulated_test_agent_tasks[agent_id] / self.fairness_episode_count_test,
                        "workload_percentage": (self.accumulated_test_agent_workload[agent_id] / self.fairness_episode_count_test) * 100
                    }

                if episode_fairness_metrics and "normalized_contributions" in episode_fairness_metrics:
                    avg_test_fairness["normalized_contributions"] = episode_fairness_metrics["normalized_contributions"]

                self._log_fairness_metrics(avg_test_fairness, avg_agent_contributions, log_prefix)

                # Reset test accumulators
                self.accumulated_test_L1 = 0.0
                self.accumulated_test_L2 = 0.0
                self.accumulated_test_L3 = 0.0
                self.fairness_episode_count_test = 0
                for agent_id in range(num_agents):
                    self.accumulated_test_agent_tasks[agent_id] = 0
                    self.accumulated_test_agent_workload[agent_id] = 0.0

            # Log additional metrics
            self._log_additional_metrics(log_prefix)

            # Summary goal stats
            self.logger.log_stat(log_prefix + "max_possible_goals", self.args.test_nepisode, self.t_env)
            self.logger.log_stat(log_prefix + "total_goals_reached", self.total_test_goals_reached, self.t_env)
            self.logger.log_stat(log_prefix + "total_episodes", self.total_test_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_goal_success_rate",
                                 self.total_test_goals_reached / max(1, self.total_test_episodes), self.t_env)

            # Cumulative milestone statistics (test)
            self._log_cumulative_milestones(test_mode, log_prefix)

            # FEN metrics (test)
            self._log_fen_metrics(log_prefix)

        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self.test_episode_count = 0
            self._log(cur_returns, cur_stats, cur_goals, log_prefix)

            # Averaged fairness metrics for training episodes
            if self.fairness_episode_count_train > 0:
                avg_train_fairness = {
                    "L1": self.accumulated_train_L1 / self.fairness_episode_count_train,
                    "L2": self.accumulated_train_L2 / self.fairness_episode_count_train,
                    "L3": self.accumulated_train_L3 / self.fairness_episode_count_train
                }
                avg_agent_contributions = {}
                num_agents = len(self.accumulated_train_agent_tasks)
                for agent_id in range(num_agents):
                    avg_agent_contributions[agent_id] = {
                        "L1_workload_balance": self.accumulated_train_agent_workload[agent_id] / self.fairness_episode_count_train,
                        "action_count": self.accumulated_train_agent_tasks[agent_id] / self.fairness_episode_count_train,
                        "workload_percentage": (self.accumulated_train_agent_workload[agent_id] / self.fairness_episode_count_train) * 100
                    }

                if episode_fairness_metrics and "normalized_contributions" in episode_fairness_metrics:
                    avg_train_fairness["normalized_contributions"] = episode_fairness_metrics["normalized_contributions"]

                self._log_fairness_metrics(avg_train_fairness, avg_agent_contributions, log_prefix)

                # Reset train accumulators
                self.accumulated_train_L1 = 0.0
                self.accumulated_train_L2 = 0.0
                self.accumulated_train_L3 = 0.0
                self.fairness_episode_count_train = 0
                for agent_id in range(num_agents):
                    self.accumulated_train_agent_tasks[agent_id] = 0
                    self.accumulated_train_agent_workload[agent_id] = 0.0

            # Log additional metrics
            self._log_additional_metrics(log_prefix)

            # Cumulative goal stats (train)
            self.logger.log_stat(log_prefix + "max_possible_goals", self.total_train_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "total_goals_reached", self.total_train_goals_reached, self.t_env)
            self.logger.log_stat(log_prefix + "total_episodes", self.total_train_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_goal_success_rate",
                                 self.total_train_goals_reached / max(1, self.total_train_episodes), self.t_env)

            # Cumulative milestones (train)
            self._log_cumulative_milestones(test_mode, log_prefix)

            # FEN metrics (train)
            self._log_fen_metrics(log_prefix)

            # Reset for next interval
            self.train_episode_count = 0
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log_additional_metrics(self, prefix):
        """Log additional metrics from document"""
        is_test = prefix == "test_"

        # Choose lists by mode
        contribution_variances = self.test_contribution_variances if is_test else self.train_contribution_variances
        specialist_utilizations = self.test_specialist_utilizations if is_test else self.train_specialist_utilizations
        overall_alignments = self.test_overall_alignments if is_test else self.train_overall_alignments
        milestone_entropies = self.test_milestone_entropies if is_test else self.train_milestone_entropies
        episode_lengths = self.test_episode_lengths if is_test else self.train_episode_lengths
        workload_ranges = self.test_workload_ranges if is_test else self.train_workload_ranges

        # Contribution variance (log singular + plural for dashboard compatibility)
        if contribution_variances:
            mean_var = float(np.mean(contribution_variances))
            self.logger.log_stat(f"{prefix}contribution_variance", mean_var, self.t_env)
            self.logger.log_stat(f"{prefix}contribution_variances", mean_var, self.t_env)
            print(f"[DEBUG] Logged {prefix}contribution_variance: {mean_var:.4f} (from {len(contribution_variances)} episodes)")
            contribution_variances.clear()
        else:
            print(f"[DEBUG] No contribution variances to log for {prefix}")

        if specialist_utilizations:
            mean_util = float(np.mean(specialist_utilizations))
            self.logger.log_stat(f"{prefix}specialist_utilization_rate", mean_util, self.t_env)
            print(f"[DEBUG] Logged {prefix}specialist_utilization_rate: {mean_util:.4f} (from {len(specialist_utilizations)} episodes)")
            specialist_utilizations.clear()
        else:
            print(f"[DEBUG] No specialist utilizations to log for {prefix}")

        if overall_alignments:
            mean_align = float(np.mean(overall_alignments))
            self.logger.log_stat(f"{prefix}overall_alignment_score", mean_align, self.t_env)
            print(f"[DEBUG] Logged {prefix}overall_alignment_score: {mean_align:.4f} (from {len(overall_alignments)} episodes)")
            overall_alignments.clear()
        else:
            print(f"[DEBUG] No overall alignment scores to log for {prefix}")

        if milestone_entropies:
            mean_entropy = float(np.mean(milestone_entropies))
            self.logger.log_stat(f"{prefix}milestone_distribution_entropy", mean_entropy, self.t_env)
            print(f"[DEBUG] Logged {prefix}milestone_distribution_entropy: {mean_entropy:.4f} (from {len(milestone_entropies)} episodes)")
            milestone_entropies.clear()

        if episode_lengths:
            self.logger.log_stat(f"{prefix}episode_length", float(np.mean(episode_lengths)), self.t_env)
            episode_lengths.clear()

        if workload_ranges:
            self.logger.log_stat(f"{prefix}workload_range", float(np.mean(workload_ranges)), self.t_env)
            workload_ranges.clear()

        # Moving averages for training fairness
        if not is_test:
            for metric in ['L1', 'L2', 'L3']:
                if len(self.fairness_history[metric]) >= 100:
                    ma_100 = float(np.mean(self.fairness_history[metric][-100:]))
                    self.logger.log_stat(f"{prefix}{metric}_ma100", ma_100, self.t_env)
                if len(self.fairness_history[metric]) >= 50:
                    ma_50 = float(np.mean(self.fairness_history[metric][-50:]))
                    self.logger.log_stat(f"{prefix}{metric}_ma50", ma_50, self.t_env)

    def update_best_actions(self):
        if self.episode_return > self.best_episode_reward:
            self.best_episode_reward = self.episode_return
            self.best_actions = self.env.pddl_env.get_episode_actions()

    def save_best_actions(self, save_path):
        with open(os.path.join(save_path, "best_actions.txt"), "w") as f:
            for action in self.best_actions:
                f.write(f"{action}\n")

    def update_best_test_actions(self):
        if self.episode_return > self.best_test_episode_reward:
            self.best_test_episode_reward = self.episode_return
            self.best_test_actions = self.env.pddl_env.get_episode_actions()

    def save_best_test_actions(self, save_path):
        with open(os.path.join(save_path, "test_best_actions.txt"), "w") as f:
            for action in self.best_test_actions:
                f.write(f"{action}\n")

    def _log(self, returns, stats, goals, prefix):
        self.logger.log_stat(prefix + "return_mean", float(np.mean(returns)), self.t_env)
        self.logger.log_stat(prefix + "return_std", float(np.std(returns)), self.t_env)
        self.logger.log_stat(prefix + "best_episode_reward", self.best_episode_reward, self.t_env)

        if goals:
            self.logger.log_stat(prefix + "goals_reached", int(sum(goals)), self.t_env)
            self.logger.log_stat(prefix + "goal_success_rate", float(np.mean(goals)), self.t_env)

        returns.clear()
        goals.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", float(v) / stats["n_episodes"], self.t_env)
        stats.clear()

    def _log_cumulative_milestones(self, test_mode, log_prefix):
        """Log cumulative milestone statistics for both medical and moveitem tasks"""
        if test_mode:
            total_episodes = max(1, self.total_test_episodes)

            # Medical milestones (test)
            self.logger.log_stat(log_prefix + "total_chest_compressed", self.total_test_chest_compressed, self.t_env)
            self.logger.log_stat(log_prefix + "total_rescue_breaths", self.total_test_rescue_breaths, self.t_env)
            self.logger.log_stat(log_prefix + "total_shock", self.total_test_shock, self.t_env)
            self.logger.log_stat(log_prefix + "total_medicine_administered", self.total_test_medicine_administered, self.t_env)
            self.logger.log_stat(log_prefix + "total_correct_stacking", self.total_test_correct_stacking, self.t_env)

            self.logger.log_stat(log_prefix + "cumulative_chest_compressed_rate",
                                 self.total_test_chest_compressed / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_rescue_breaths_rate",
                                 self.total_test_rescue_breaths / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_shock_rate",
                                 self.total_test_shock / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_medicine_administered_rate",
                                 self.total_test_medicine_administered / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_correct_stacking_rate",
                                 self.total_test_correct_stacking / total_episodes, self.t_env)
        else:
            total_episodes = max(1, self.total_train_episodes)

            # Medical milestones (train)
            self.logger.log_stat(log_prefix + "total_chest_compressed", self.total_train_chest_compressed, self.t_env)
            self.logger.log_stat(log_prefix + "total_rescue_breaths", self.total_train_rescue_breaths, self.t_env)
            self.logger.log_stat(log_prefix + "total_shock", self.total_train_shock, self.t_env)
            self.logger.log_stat(log_prefix + "total_medicine_administered", self.total_train_medicine_administered, self.t_env)
            self.logger.log_stat(log_prefix + "total_correct_stacking", self.total_train_correct_stacking, self.t_env)

            self.logger.log_stat(log_prefix + "cumulative_chest_compressed_rate",
                                 self.total_train_chest_compressed / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_rescue_breaths_rate",
                                 self.total_train_rescue_breaths / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_shock_rate",
                                 self.total_train_shock / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_medicine_administered_rate",
                                 self.total_train_medicine_administered / total_episodes, self.t_env)
            self.logger.log_stat(log_prefix + "cumulative_correct_stacking_rate",
                                 self.total_train_correct_stacking / total_episodes, self.t_env)

    def _log_fairness_metrics(self, fairness_metrics, agent_contributions, prefix):
        """Log fairness metrics to WandB/logger"""
        if fairness_metrics:
            self.logger.log_stat(prefix + "L1", float(fairness_metrics.get("L1", 0.0)), self.t_env)
            self.logger.log_stat(prefix + "L2", float(fairness_metrics.get("L2", 0.0)), self.t_env)
            self.logger.log_stat(prefix + "L3", float(fairness_metrics.get("L3", 0.0)), self.t_env)

        # Additional detailed metrics if agent_contributions provided
        if agent_contributions:
            for agent_id, contrib in agent_contributions.items():
                agent_prefix = f"{prefix}agent_{agent_id}/"
                self.logger.log_stat(agent_prefix + "L1_workload_balance", float(contrib.get("L1_workload_balance", 0.0)), self.t_env)
                self.logger.log_stat(agent_prefix + "L2_skill_alignment", float(contrib.get("L2_skill_alignment", 0.0)), self.t_env)
                self.logger.log_stat(agent_prefix + "total_tasks", int(contrib.get("total_tasks", 0)), self.t_env)
                self.logger.log_stat(agent_prefix + "total_skill_contributed", float(contrib.get("total_skill_contributed", 0.0)), self.t_env)
                if "workload_percentage" in contrib:
                    self.logger.log_stat(agent_prefix + "workload_percentage", float(contrib["workload_percentage"]), self.t_env)

        # Fairness "violations" for analysis
        if fairness_metrics:
            self.logger.log_stat(prefix + "fairness/high_workload_imbalance",
                                 1.0 if fairness_metrics.get("L1", 0) > 0.3 else 0.0, self.t_env)
            self.logger.log_stat(prefix + "fairness/poor_skill_utilization",
                                 1.0 if fairness_metrics.get("L2", 0) > 0.3 else 0.0, self.t_env)

        # Guard: fairness_metrics might be None
        if fairness_metrics and "normalized_contributions" in fairness_metrics:
            for i, contrib in enumerate(fairness_metrics["normalized_contributions"]):
                self.logger.log_stat(f"{prefix}Agent {i} %", float(contrib) * 100.0, self.t_env)
                self.logger.log_stat(f"{prefix}agent_{i}_normalized_contribution", float(contrib), self.t_env)
                self.logger.log_stat(f"{prefix}agent_{i}_task_percentage", float(contrib) * 100.0, self.t_env)

    def _log_fen_metrics(self, prefix):
        """Log FEN-specific metrics when available"""
        if self.current_reward_type != "fen":
            return

        # Mark that FEN is active
        self.logger.log_stat(prefix + "reward_type", 1, self.t_env)

        # FEN utilities over steps
        if self.fen_utilities_history:
            all_utilities = np.array(self.fen_utilities_history)
            fairness_index = 0.0  # default for print below

            if all_utilities.size > 0:
                mean_utilities = np.mean(all_utilities, axis=0)
                total_utility = float(np.sum(mean_utilities))

                for agent_id, mean_util in enumerate(mean_utilities):
                    self.logger.log_stat(f"{prefix}fen_agent_{agent_id}_utility_mean", float(mean_util), self.t_env)
                    if total_utility > 0.0:
                        utility_percentage = (float(mean_util) / total_utility) * 100.0
                        self.logger.log_stat(f"{prefix}Agent {agent_id} %", utility_percentage, self.t_env)
                    else:
                        equal_percentage = 100.0 / max(1, len(mean_utilities))
                        self.logger.log_stat(f"{prefix}Agent {agent_id} %", equal_percentage, self.t_env)

                utility_variance = float(np.var(mean_utilities))
                self.logger.log_stat(prefix + "fen_utility_variance", utility_variance, self.t_env)

                if total_utility > 0.0:
                    percentages = [(float(util) / total_utility) * 100.0 for util in mean_utilities]
                    percentage_variance = float(np.var(percentages))
                    self.logger.log_stat(prefix + "fen_percentage_variance", percentage_variance, self.t_env)

                    expected_percentage = 100.0 / max(1, len(mean_utilities))
                    fairness_deviations = [abs(p - expected_percentage) for p in percentages]
                    avg_deviation = float(np.mean(fairness_deviations))
                    fairness_index = max(0.0, 1.0 - (avg_deviation / expected_percentage))
                    self.logger.log_stat(prefix + "fen_fairness_index", fairness_index, self.t_env)

                    print(f"[FEN] Fairness Index: {fairness_index:.3f}")

                print(f"[FEN] Summary - Variance: {utility_variance:.4f}, Fairness: {fairness_index:.3f}")

            # Clear history after logging
            self.fen_utilities_history.clear()

        # FEN agent action percentages
        if self.fen_agent_action_counts and self.fen_total_actions_count > 0:
            num_agents = len(self.fen_agent_action_counts)
            for agent_id in sorted(self.fen_agent_action_counts.keys()):
                action_count = self.fen_agent_action_counts[agent_id]
                action_percentage = (action_count / self.fen_total_actions_count) * 100.0
                self.logger.log_stat(f"{prefix}fen_agent_{agent_id}_action_percent", float(action_percentage), self.t_env)

            expected_percentage = 100.0 / max(1, num_agents)
            action_percentages = [(self.fen_agent_action_counts.get(i, 0) / self.fen_total_actions_count) * 100.0
                                  for i in range(num_agents)]
            action_variance = float(np.var(action_percentages))
            self.logger.log_stat(prefix + "fen_action_percentage_variance", action_variance, self.t_env)

            action_deviations = [abs(p - expected_percentage) for p in action_percentages]
            avg_action_deviation = float(np.mean(action_deviations))
            action_fairness_index = max(0.0, 1.0 - (avg_action_deviation / expected_percentage))
            self.logger.log_stat(prefix + "fen_action_fairness_index", action_fairness_index, self.t_env)
            print(f"[FEN] Action Fairness Index: {action_fairness_index:.3f}")

            # Clear action counts after logging
            self.fen_agent_action_counts.clear()
            self.fen_total_actions_count = 0

        # FEN average utilities
        if self.fen_avg_utilities_history:
            avg_utility = float(np.mean(self.fen_avg_utilities_history))
            self.logger.log_stat(prefix + "fen_avg_utility_mean", avg_utility, self.t_env)
            self.fen_avg_utilities_history.clear()

        # FEN step counts
        if self.fen_step_counts:
            avg_step_count = float(np.mean(self.fen_step_counts))
            self.logger.log_stat(prefix + "fen_step_count_mean", avg_step_count, self.t_env)
            self.fen_step_counts.clear()