import json
import os
import gzip
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict

import gym
import pddlgym

from utils.fair_gne import FairGNE
from utils.hosp_reward_handler import HospRewardHandler
from utils.robotouille_reward_handler import RobotouilleRewardHandler
from utils.goal_focused_reward_handler import GoalFocusedRewardHandler
from utils.fen_reward_handler import FenRewardHandler

import utils.robotouille_utils as robotouille_utils
import utils.pddlgym_utils as pddlgym_utils
from environments.env_generator.object_enums import Item


class RobotouilleWrapper(gym.Wrapper):
    """
    Wrapper around the PDDLGym environment that adds non-PDDL state, reward shaping,
    per-agent logging, and episode JSON logging.
    """

    def __init__(self, env, config, renderer):
        super(RobotouilleWrapper, self).__init__(env)
        self.env = env
        self.config = config
        self.renderer = renderer
        self.use_flexible_rewards = False

        self.prev_step = None
        self.timesteps = 0
        self.state = {}
        self.num_players = config["num_players"]
        self.move_counter = 0
        self.taken_actions = []
        self.reward_handler = HospRewardHandler(self.state)
        self.episode_reward = 0.0
        self.goal_announced = False
        self.goals_reached = 0
        self.max_steps = 50
        self.pddl_goal_achieved = False
        self.agent_skills = self._extract_agent_skills()
        print(f"[DEBUG] Extracted agent skills: {self.agent_skills}")

        self._log_dir = None
        self._log_test_mode = False
        self._episode_counter = 0

        self.agent_action_counts = defaultdict(int)
        # O(1) per-step workloads cache (excludes noop/move)
        self._workload_cache = [0] * self.num_players

        self.agent_medical_progress = {
            "compresschest": defaultdict(int),
            "giverescuebreaths": defaultdict(int),
            "giveshock": defaultdict(int),
            "givemedicine": defaultdict(int),
        }

        self.agent_medical_events = {
            "chest_compressed_by": None,
            "rescue_breaths_by": None,
            "shock_by": None,
            "medicine_by": None,
        }

        self._last_actor_id = None
        self._action_log = []
        self._last_episode_metrics = None  

        use_fairness_env = os.getenv("USE_FAIRNESS", str(
            self.config.get("reward_config", {}).get("use_fairness", False)
        )).lower() in ("1", "true", "yes")

        reward_type_env = str(os.getenv(
            "REWARD_TYPE",
            self.config.get("reward_config", {}).get("reward_type", "")
        )).lower()

        self._fen_enabled = use_fairness_env or (reward_type_env == "fen")
        self._fairskill_enabled = use_fairness_env

        self._skill_alpha = float(os.getenv(
            "FAIRNESS_ALPHA",
            str(self.config.get("reward_config", {}).get("alpha", 0.0))
        ))

        self._fen = FenRewardHandler(self.num_players) if self._fen_enabled else None

        print(f"[WRAPPER] Mode: FEN={'on' if self._fen_enabled else 'off'} | "
              f"FairSkill={'on' if self._fairskill_enabled else 'off'} | "
              f"alpha={self._skill_alpha:.3f}")

        # -------- Fair-GNE flags from env (fix: no tabs, correct indent) --------
        self._use_fair_gne = os.getenv("USE_FAIR_GNE", "false").lower() in ("1", "true", "yes")
        self._fair_gne_step_shaping = os.getenv("FAIR_GNE_STEP_SHAPING", "true").lower() in ("1", "true", "yes")
        self._fair_gne_update_freq = int(os.getenv("FAIR_GNE_UPDATE_FREQ", "10"))
        self._fair_gne_full_lagrangian = os.getenv("FAIR_GNE_FULL_LAGRANGIAN", "false").lower() in ("1", "true", "yes")

        if self._use_fair_gne:
            gne_tau = float(os.getenv("FAIR_GNE_TAU", "0.85"))
            gne_dual_lr = float(os.getenv("FAIR_GNE_DUAL_LR", "0.01"))
            gne_lambda_max = float(os.getenv("FAIR_GNE_LAMBDA_MAX", "10.0"))

            self._fair_gne = FairGNE(
                tau=gne_tau,
                dual_lr=gne_dual_lr,
                lambda_max=gne_lambda_max
            )

            # Keep python-side object consistent with environment flags
            self._fair_gne.full_lagrangian = self._fair_gne_full_lagrangian
            self._fair_gne.update_freq = self._fair_gne_update_freq
            # ===== ADD THIS LINE =====
            self._last_fair_gne_metrics = {}  # Initialize storage for Fair-GNE metrics
            # =========================
            print(f"[Fair-GNE] Enabled | tau={gne_tau} | dual_lr={gne_dual_lr} | lambda_max={gne_lambda_max}")

            if self._fen_enabled or self._fairskill_enabled:
                print("[WARNING] Fair-GNE enabled, disabling FEN and FairSkillMARL modes")
                self._fen_enabled = False
                self._fairskill_enabled = False
        else:
            self._fair_gne = None

    def _interactive_starter_prints(self, expanded_truths):
        print("\n" * 10)
        if self.timesteps % 10 == 0:
            print(f"You have made {self.timesteps} steps.")
        robotouille_utils.print_states(self.prev_step[0])
        print("\n")
        robotouille_utils.print_actions(self.env, self.prev_step[0], self.renderer)
        print(f"True Predicates: {expanded_truths.sum()}")

    def _count_players(self, obs):
        num_players = 0
        for literal in obs.literals:
            if "isrobot" in literal.predicate.name:
                num_players += 1
        return num_players

    def _current_selected_player(self, obs):
        for literal in obs.literals:
            if "selected" == literal.predicate.name:
                return literal.variables[0].name

    def _change_selected_player(self, obs):
        current_player = self._current_selected_player(obs)
        current_player_index = int(current_player[5:])
        next_player = current_player_index % self.num_players + 1
        next_player = f"robot{next_player}"
        action = f"select({current_player}:player,{next_player}:player)"
        try:
            action = robotouille_utils.create_action(
                self.env, obs, action, self.renderer
            )
        except Exception:
            print("Error in changing player")
            return self.prev_step
        return self.env.step(action)

    def _check_cooked(self, obs):
        for literal in obs.literals:
            if "iscooked" == literal.predicate.name:
                return True
        return False

    def _extract_agent_skills(self):
        skills = {}
        player_info = self.config.get("player_info", {})
        if player_info:
            for i in range(self.num_players):
                agent_key = f"robot{i+1}"
                if agent_key in player_info:
                    skills[i] = player_info[agent_key]

        if not skills:
            for i in range(self.num_players):
                skills[i] = {
                    "compresschest": 1.0,
                    "giverescuebreaths": 1.0,
                    "giveshock": 1.0,
                    "givemedicine": 1.0,
                    "move": 1.0,
                    "moveitem": 1.0,
                    "pick-up": 1.0,
                    "place": 1.0,
                    "stack": 1.0,
                    "stackunder": 1.0
                }
        return skills

    def _compute_total_workloads(self):
        """
        Compute workload counts excluding move and noop actions.
        Only counts meaningful task-related actions to get proper fairness metrics.
        """
        workloads = [0] * self.num_players
        for entry in self._action_log:
            agent_id = entry.get("agent_id")
            action = entry.get("action", "")
            if action in ["noop"]:
                continue
            if 0 <= agent_id < self.num_players:
                workloads[agent_id] += 1
        return workloads

    def _requirement_for(self, action_key, item_name_base):
        if action_key == "compresschest":
            d = self.config.get("num_compressions", {})
        elif action_key == "giverescuebreaths":
            d = self.config.get("num_breaths", {})
        elif action_key == "giveshock":
            d = self.config.get("num_shocks", {})
        elif action_key == "givemedicine":
            d = self.config.get("num_medicine_doses", {})
        else:
            return 1

        if isinstance(d, dict):
            if item_name_base in d:
                return d[item_name_base]
            if "patient" in d:
                return d["patient"]
            return d.get("default", 1)
        return 1

    def _handle_action(self, action):
        if action == "noop":
            return self.prev_step

        action_name = action.predicate.name
        items = [var.name for var in action.variables if var.var_type == "item"]

        current_player = self._current_selected_player(self.prev_step[0])
        self._last_actor_id = int(current_player.replace("robot", "")) - 1

        self.agent_action_counts[str(self._last_actor_id)] += 1

        self._action_log.append({
            "agent_id": self._last_actor_id,
            "action": action_name,
            "timestep": self.timesteps,
            "full_action": str(action)
        })
        # Keep O(1) per-step workloads (exclude noop/move)
        if self._last_actor_id is not None and action_name not in ["noop", "move"]:
            if 0 <= self._last_actor_id < self.num_players:
                self._workload_cache[self._last_actor_id] += 1

        simple_map = {
            "compresschest_simple": ("compresschest", "chest_compressed_by"),
            "giverescuebreaths_simple": ("giverescuebreaths", "rescue_breaths_by"),
            "giveshock_simple": ("giveshock", "shock_by"),
            "givemedicine_simple": ("givemedicine", "medicine_by"),
        }
        if action_name in simple_map:
            action_key, event_key = simple_map[action_name]
            if self.agent_medical_events[event_key] is None:
                self.agent_medical_events[event_key] = self._last_actor_id

            if items:
                base, _ = robotouille_utils.trim_item_ID(items[0])
                req = self._requirement_for(action_key, base)
            else:
                req = self._requirement_for(action_key, "patient")
            self.agent_medical_progress[action_key][self._last_actor_id] += int(req)

        if action_name == "cut":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"cut": 1}
            elif item_status.get("cut") is None:
                item_status["cut"] = 1
            else:
                item_status["cut"] += 1
                if item_status["cut"] == 3:
                    item_status["picked-up"] = False
            return self.prev_step

        elif action_name == "compresschest":
            player = next(filter(lambda te: te.var_type == "player", action.variables))
            player_status = self.state.get(player.name)
            energy_cfg = self.config["energy_levels"]
            if player_status is None:
                self.state[player.name] = {"energy": int(energy_cfg["max"])}
                player_status = self.state[player.name]
            elif player_status.get("energy") is None:
                player_status["energy"] = int(energy_cfg["max"])

            if energy_cfg["compresschest_cost"] <= player_status["energy"]:
                item = next(filter(lambda te: te.var_type == "item", action.variables))
                item_status = self.state.get(item.name)
                if item_status is None:
                    self.state[item.name] = {"compresschest": 1}
                    item_status = self.state[item.name]
                elif item_status.get("compresschest") is None:
                    item_status["compresschest"] = 1
                else:
                    item_status["compresschest"] += 1
                    if item_status["compresschest"] == 3:
                        item_status["picked-up"] = False

                self.agent_medical_progress["compresschest"][self._last_actor_id] += 1
                item_status["compresschest_by"] = self._last_actor_id

                player_status["energy"] -= int(
                    energy_cfg["compresschest_cost"] + energy_cfg["recharge_rate"]
                )
            return self.prev_step

        elif action_name == "giverescuebreaths":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"giverescuebreaths": 1}
                item_status = self.state[item.name]
            elif item_status.get("giverescuebreaths") is None:
                item_status["giverescuebreaths"] = 1
            else:
                item_status["giverescuebreaths"] += 1

            self.agent_medical_progress["giverescuebreaths"][self._last_actor_id] += 1
            item_status["giverescuebreaths_by"] = self._last_actor_id
            return self.prev_step

        elif action_name == "giveshock":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"giveshock": 1}
                item_status = self.state[item.name]
            elif item_status.get("giveshock") is None:
                item_status["giveshock"] = 1
            else:
                item_status["giveshock"] += 1

            self.agent_medical_progress["giveshock"][self._last_actor_id] += 1
            item_status["giveshock_by"] = self._last_actor_id
            return self.prev_step

        elif action_name == "givemedicine":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"givemedicine": 1}
                item_status = self.state[item.name]
            elif item_status.get("givemedicine") is None:
                item_status["givemedicine"] = 1
            else:
                item_status["givemedicine"] += 1

            self.agent_medical_progress["givemedicine"][self._last_actor_id] += 1
            item_status["givemedicine_by"] = self._last_actor_id
            return self.prev_step

        elif action_name == "cook":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"cook": {"cook_time": -1, "cooking": True}}
            elif item_status.get("cook") is None:
                item_status["cook"] = {"cook_time": -1, "cooking": True}
            else:
                item_status["cook"]["cooking"] = True
            return self.prev_step

        elif action_name == "fry" or action_name == "fry_cut_item":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"fry": {"fry_time": -1, "frying": True}}
            elif item_status.get("fry") is None:
                item_status["fry"] = {"fry_time": -1, "frying": True}
            else:
                item_status["fry"]["frying"] = True
            return self.prev_step

        elif action_name in ["stack", "stackunder"]:
            for item_name in items:
                item_status = self.state.get(item_name, {})
                item_status["stacked"] = True
                self.state[item_name] = item_status
            return self.prev_step

        elif action_name == "pick-up":
            item = next(filter(lambda te: te.var_type == "item", action.variables))
            item_status = self.state.get(item.name)
            if item_status is not None and item_status.get("cook") is not None:
                item_status["cook"]["cooking"] = False
            if item_status is not None and item_status.get("fry") is not None:
                item_status["fry"]["frying"] = False

            item_status = self.state.get("patty1")
            cooked = self._check_cooked(self.prev_step[0])

            if item_status is None:
                self.state[item.name] = {"picked-up": False}
                item_status = self.state.get(item.name)
            elif item_status.get("picked-up") is None:
                item_status["picked-up"] = False

            if cooked:
                item_status["picked-up"] = True
            return self.prev_step

        return self.env.step(action)

    def _is_end_of_timestep(self):
        return self.move_counter % self.num_players == self.num_players - 1

    def _state_update(self):
        state_updates = []
        state_removals = []

        for item, status_dict in self.state.items():
            for status, s in status_dict.items():
                if status == "cut":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_cuts = self.config["num_cuts"]
                    max_num_cuts = num_cuts.get(item_name, num_cuts.get("default", 3))
                    if s >= max_num_cuts:
                        literal = pddlgym_utils.str_to_literal(f"iscut({item}:item)")
                        state_updates.append(literal)

                elif status == "cook":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    cook_time = self.config["cook_time"]
                    max_cook_time = cook_time.get(item_name, cook_time.get("default", 1))
                    if s["cooking"]:
                        s["cook_time"] += 1
                        if s["cook_time"] == max_cook_time:
                            status_dict["picked-up"] = False
                    if s["cook_time"] >= max_cook_time:
                        literal = pddlgym_utils.str_to_literal(f"iscooked({item}:item)")
                        state_updates.append(literal)

                elif status == "fry":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    fry_time = self.config["fry_time"]
                    max_fry_time = fry_time.get(item_name, fry_time.get("default", 1))
                    if s["frying"]:
                        s["fry_time"] += 1
                    if s["fry_time"] >= max_fry_time:
                        literal = pddlgym_utils.str_to_literal(f"isfried({item}:item)")
                        state_updates.append(literal)

                elif status == "compresschest":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_compressions = self.config["num_compressions"]
                    max_num = num_compressions.get(item_name, num_compressions.get("default", 3))
                    if s >= max_num:
                        literal = pddlgym_utils.str_to_literal(
                            f"ischestcompressed({item}:item)"
                        )
                        state_updates.append(literal)

                elif status == "giverescuebreaths":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_breaths = self.config["num_breaths"]
                    max_num = num_breaths.get(item_name, num_breaths.get("default", 2))
                    if s >= max_num:
                        literal = pddlgym_utils.str_to_literal(
                            f"isrescuebreathed({item}:item)"
                        )
                        state_updates.append(literal)

                elif status == "giveshock":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_shocks = self.config["num_shocks"]
                    max_num = num_shocks.get(item_name, num_shocks.get("default", 2))
                    if s >= max_num:
                        literal = pddlgym_utils.str_to_literal(
                            f"isshocked({item}:item)"
                        )
                        state_updates.append(literal)

                elif status == "givemedicine":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_meds = self.config["num_medicine_doses"]
                    max_num = num_meds.get(item_name, num_meds.get("default", 1))
                    if s >= max_num:
                        literal = pddlgym_utils.str_to_literal(
                            f"istreated({item}:item)"
                        )
                        state_updates.append(literal)

                elif status == "energy":
                    if self._is_end_of_timestep():
                        energy_cfg = self.config["energy_levels"]
                        status_dict["energy"] = min(
                            s + energy_cfg["recharge_rate"], energy_cfg["max"]
                        )
                        literal = pddlgym_utils.str_to_literal(
                            f"istired({item}:player)"
                        )
                        if status_dict["energy"] < energy_cfg["compresschest_cost"]:
                            state_updates.append(literal)
                        else:
                            state_removals.append(literal)

        env_state = self.env.get_state()
        new_literals = env_state.literals.difference(set(state_removals))
        new_literals = new_literals.union(state_updates)
        new_env_state = pddlgym.structs.State(
            new_literals, env_state.objects, env_state.goal
        )
        self.env.set_state(new_env_state)

        goal_reached = pddlgym.inference.check_goal(new_env_state, env_state.goal)
        if goal_reached:
            self.pddl_goal_achieved = True
        return new_env_state, goal_reached

    def get_latest_info(self):
        return self.prev_step[3] if self.prev_step else None

    def test_step(self, action):
        obs, reward, done, info = self._handle_action(action)
        _, reward, done, info = self.prev_step
        self.prev_step = (obs, reward, done, info)
        return obs, reward, done, info


    def step(self, action=None, interactive=False):
        expanded_truths = self.prev_step[3]["expanded_truths"]
        expanded_states = self.prev_step[3]["expanded_states"]
        prev_expanded_states = expanded_states

        self.taken_actions.append(action)

        if interactive:
            self._interactive_starter_prints(expanded_truths)
            action = robotouille_utils.create_action_repl(
                self.env, self.prev_step[0], self.renderer
            )
        else:
            action = robotouille_utils.create_action(
                self.env, self.prev_step[0], action, self.renderer
            )

        prev_heuristic = self.reward_handler.heuristic_reward(
            self.prev_step[0], self.state, env_timestep=self.timesteps, bump_counter=False
        )

        obs, reward, done, info = self._handle_action(action)
        obs, reward, _, info = self._change_selected_player(obs)
        obs, goal_reached = self._state_update()

        done = goal_reached or (self.timesteps >= self.max_steps)

        toggle_array = pddlgym_utils.create_toggle_array(
            expanded_truths, expanded_states, obs.literals
        )

        def _now_true(pred):
            return any(l.predicate.name == pred and "patient" in l.variables[0].name
                    for l in obs.literals)

        for changed, lit in zip(toggle_array, prev_expanded_states):
            if not changed:
                continue
            if not getattr(lit, "predicate", None) or not getattr(lit, "variables", None):
                continue
            pred = lit.predicate.name
            first_arg = getattr(lit.variables[0], "name", "")
            if "patient" not in first_arg:
                continue

            if pred == "ischestcompressed" and _now_true("ischestcompressed") and self.agent_medical_events["chest_compressed_by"] is None:
                self.agent_medical_events["chest_compressed_by"] = self._last_actor_id
            elif pred == "isrescuebreathed" and _now_true("isrescuebreathed") and self.agent_medical_events["rescue_breaths_by"] is None:
                self.agent_medical_events["rescue_breaths_by"] = self._last_actor_id
            elif pred == "isshocked" and _now_true("isshocked") and self.agent_medical_events["shock_by"] is None:
                self.agent_medical_events["shock_by"] = self._last_actor_id
            elif pred == "istreated" and _now_true("istreated") and self.agent_medical_events["medicine_by"] is None:
                self.agent_medical_events["medicine_by"] = self._last_actor_id

        expanded_truths, expanded_states = pddlgym_utils.expand_state(
            obs.literals, obs.objects
        )

        if self._current_selected_player(obs) == "robot1":
            self.timesteps += 1
        self.move_counter += 1

        info = {
            "timesteps": self.timesteps,
            "expanded_truths": expanded_truths,
            "expanded_states": expanded_states,
            "toggle_array": toggle_array,
            "state": self.state,
            "pddl_goal_achieved": self.pddl_goal_achieved,
            "agent_action_counts": dict(self.agent_action_counts),
            "agent_medical_events": dict(self.agent_medical_events),
            "agent_medical_counts": {
                k: {str(aid): int(cnt) for aid, cnt in d.items()}
                for k, d in self.agent_medical_progress.items()
            },
            "agent_id": self._last_actor_id,
        }

        curr_heuristic = self.reward_handler.heuristic_reward(
            obs, self.state, env_timestep=self.timesteps, bump_counter=True
        )
        reward = curr_heuristic - prev_heuristic

        if self._fen is not None:
            inc = [0.0] * self.num_players
            if self._last_actor_id is not None:
                inc[self._last_actor_id] = 1.0
            fen_vec = self._fen.per_step_reward(reward, inc)
            reward = float(np.mean(fen_vec))

        if self._fairskill_enabled and self._skill_alpha > 0.0:
            l2 = self._compute_L2_alignment_step()
            reward += (-self._skill_alpha) * l2
            if os.getenv("FAIRSKILL_DEBUG", "0") in ("1", "true", "yes"):
                print(f"[FAIRSKILL] t={self.timesteps:4d} | L2={l2:.6f} | alpha={self._skill_alpha:.3f} | shaped={reward:+.6f}")

        # ======== Fair-GNE per-step shaping (needed for value-based methods like QMIX) ========
        if self._use_fair_gne and self._fair_gne_step_shaping and (self._fair_gne is not None):
            workloads_now = self._workload_cache  # Current workloads excluding noop/move
            if sum(workloads_now) > 0:
                jfi_now = self._fair_gne.compute_jfi(workloads_now)
                g_fair_now = self._fair_gne.constraint_violation(jfi_now)
                penalty_now = self._fair_gne.lambda_fair * g_fair_now
                
                # DIAGNOSTIC PRINT
                if self.timesteps % 10 == 0:  # Print every 10 steps
                    print(f"[Step {self.timesteps}] Workloads: {workloads_now}, "
                        f"JFI: {jfi_now:.3f}, λ: {self._fair_gne.lambda_fair:.2f}, "
                        f"penalty: {penalty_now:+.3f}, reward_before: {reward:+.3f}, ", end="")
                
                # shaped reward flows into the learner
                reward = reward - penalty_now
                
                # DIAGNOSTIC PRINT CONTINUED
                if self.timesteps % 10 == 0:
                    print(f"reward_after: {reward:+.3f}")

                # Update λ periodically or at episode termination
                if (self.timesteps % max(1, self._fair_gne.update_freq) == 0) or done:
                    self._fair_gne.update_dual(g_fair_now)
        # ========================================================================



        # ========================================================================

        # ===== COMPUTE FAIRNESS METRICS FOR ALL EPISODES (TRAIN + TEST) =====
        if done:
            normalized_contribs, task_breakdown = self.calculate_robust_normalized_contributions(self._action_log)
            
            try:
                from utils.fairness_metrics_cal import compute_L1, compute_L2, compute_L3
            except Exception:
                compute_L1 = lambda x: 0.0
                compute_L2 = lambda x, y: 0.0
                compute_L3 = lambda x, y, z: 0.0
            
            raw_action_counts = [self.agent_action_counts.get(str(i + 1), 0) for i in range(self.num_players)]
            L1_raw = compute_L1(raw_action_counts) if sum(raw_action_counts) > 0 else 0.0
            
            task_skill_pairs = []
            all_agents_skills = {i: self.config.get("player_info", {}).get(f"robot{i+1}", {}) for i in range(self.num_players)}
            tracked_actions = set()
            for v in all_agents_skills.values():
                tracked_actions.update(v.keys())
            
            for entry in self._action_log:
                action = entry["action"]
                agent_id = entry["agent_id"]
                if action == "noop":
                    continue
                normalized_action = action[:-7] if action.endswith("_simple") else action
                if normalized_action in tracked_actions:
                    skill = all_agents_skills.get(agent_id, {}).get(normalized_action, 1.0)
                    task_skill_pairs.append((normalized_action, agent_id, skill))
            
            L2 = compute_L2(task_skill_pairs, all_agents_skills) if task_skill_pairs else 0.0
            L3 = compute_L3(L1_raw, L2, alpha=0.5)
            
            workloads = self._compute_total_workloads()
            if sum(workloads) > 0:
                sum_workloads = sum(workloads)
                sum_squared = sum(w**2 for w in workloads)
                jfi = (sum_workloads ** 2) / (self.num_players * sum_squared) if sum_squared > 0 else 1.0
            else:
                jfi = 1.0


            base_metrics = {
                "L1": L1_raw,
                "L2": L2,
                "L3": L3,
                "jfi": jfi,  # Regular JFI (always present)
                "normalized_contributions": normalized_contribs,
                "agent_action_counts": {i: self.agent_action_counts.get(str(i + 1), 0) 
                                    for i in range(self.num_players)},
                "workload_counts": workloads
            }

            # Add Fair-GNE metrics ALWAYS (with actual values if enabled, placeholders if not)
            if self._use_fair_gne and hasattr(self, '_last_fair_gne_metrics') and self._last_fair_gne_metrics:
                # Fair-GNE ENABLED - use actual tracked metrics
                base_metrics.update({
                    "fair_gne_enabled": True,
                    "fair_gne_jfi": float(self._last_fair_gne_metrics.get('jfi', jfi)),
                    "fair_gne_lambda": float(self._last_fair_gne_metrics.get('lambda', 0.0)),
                    "fair_gne_g_fair": float(self._last_fair_gne_metrics.get('g_fair', 0.0)),
                    "fair_gne_penalty": float(self._last_fair_gne_metrics.get('penalty', 0.0)),
                    "fair_gne_shaped_reward": float(self._last_fair_gne_metrics.get('shaped_reward', self.episode_reward)),
                    "fair_gne_constraint_satisfied": bool(self._last_fair_gne_metrics.get('jfi', jfi) >= self._fair_gne.tau),
                    "fair_gne_tau": float(self._fair_gne.tau),
                })
            else:
                # Fair-GNE DISABLED - log JFI with neutral values for Fair-GNE metrics
                base_metrics.update({
                    "fair_gne_enabled": False,
                    "fair_gne_jfi": float(jfi),  # Use always-computed JFI
                    "fair_gne_lambda": 0.0,
                    "fair_gne_g_fair": 0.0,
                    "fair_gne_penalty": 0.0,
                    "fair_gne_shaped_reward": float(self.episode_reward),
                    "fair_gne_constraint_satisfied": None,
                    "fair_gne_tau": 0.85,  # Default reference value
                })
            
            self._last_episode_metrics = base_metrics                            
            
            # self._last_episode_metrics = {
            #     "L1": L1_raw,
            #     "L2": L2,
            #     "L3": L3,
            #     "jfi": jfi,
            #     "normalized_contributions": normalized_contribs,
            #     "agent_action_counts": {i: self.agent_action_counts.get(str(i + 1), 0) 
            #                            for i in range(self.num_players)},
            #     "workload_counts": workloads
            # }
            
            print(f"[TEST FAIRNESS] JFI: {jfi:.3f}, L1: {L1_raw:.3f}, L2: {L2:.3f}, L3: {L3:.3f}")
        # ============================================================

        # ===== EPISODE-END FAIR-GNE PROCESSING =====        

        # ===== EPISODE-END FAIR-GNE PROCESSING =====
        if done and self._use_fair_gne and (self._fair_gne is not None):
            workloads = self._compute_total_workloads()
            
            # ==================== COMPREHENSIVE EPISODE DIAGNOSTIC ====================
            print(f"\n{'='*80}")
            print(f"FAIR-GNE EPISODE END - Episode {self._episode_counter}")
            print(f"{'='*80}")
            print(f"Mode: {'TEST' if self._log_test_mode else 'TRAIN'}")
            print(f"Timesteps: {self.timesteps} / {self.max_steps}")
            
            # ACTION LOG ANALYSIS
            print(f"\n--- ACTION LOG BREAKDOWN ---")
            print(f"Total action log entries: {len(self._action_log)}")
            
            # Per-agent action breakdown
            for agent_id in range(self.num_players):
                agent_actions = [e for e in self._action_log if e.get('agent_id') == agent_id]
                print(f"\n  Agent {agent_id} ({len(agent_actions)} total actions):")
                
                agent_action_types = {}
                for e in agent_actions:
                    action = e.get('action', 'unknown')
                    agent_action_types[action] = agent_action_types.get(action, 0) + 1
                
                for action, count in sorted(agent_action_types.items()):
                    excluded = "❌ EXCLUDED" if action in ["noop", "move"] else "✓ COUNTED"
                    print(f"    {action:20s}: {count:3d}  {excluded}")
            
            # WORKLOAD CALCULATIONS
            print(f"\n--- WORKLOAD CALCULATIONS ---")
            
            # Method 1: _compute_total_workloads (excludes noop/move)
            workloads_method1 = self._compute_total_workloads()
            print(f"Method 1 (_compute_total_workloads): {workloads_method1}")
            
            # Method 2: _workload_cache
            workloads_method2 = self._workload_cache.copy()
            print(f"Method 2 (_workload_cache):          {workloads_method2}")
            
            # Method 3: Manual count from action_log
            workloads_method3 = [0] * self.num_players
            for entry in self._action_log:
                action = entry.get('action', '')
                agent_id = entry.get('agent_id', -1)
                if action not in ['noop', 'move'] and 0 <= agent_id < self.num_players:
                    workloads_method3[agent_id] += 1
            print(f"Method 3 (manual from log):          {workloads_method3}")
            
            # Verify consistency
            if workloads_method1 != workloads_method2:
                print(f"⚠️  WARNING: Method 1 ≠ Method 2")
            if workloads_method1 != workloads_method3:
                print(f"⚠️  WARNING: Method 1 ≠ Method 3")
            
            # JFI CALCULATION
            print(f"\n--- JFI CALCULATION ---")
            print(f"Workloads: {workloads}")
            print(f"Sum: {sum(workloads)}")
            print(f"Sum of squares: {sum(w**2 for w in workloads)}")
            
            if sum(workloads) > 0:
                w_sum = sum(workloads)
                w_sum_sq = sum(w**2 for w in workloads)
                n = len(workloads)
                manual_jfi = (w_sum ** 2) / (n * w_sum_sq) if w_sum_sq > 0 else 1.0
                print(f"\nManual JFI = ({w_sum}²) / ({n} × {w_sum_sq}) = {manual_jfi:.4f}")
            else:
                manual_jfi = 1.0
                print(f"\n⚠️  All workloads are ZERO - defaulting to JFI = 1.0")

            # # ===== CACHE METRICS FOR EPISODE_RUNNER (FOR WANDB LOGGING) =====
            # self._last_episode_metrics = {
            #     "L1": L1_raw,
            #     "L2": L2,
            #     "L3": L3,
            #     "jfi": jfi,
            #     "normalized_contributions": normalized_contribs,
            #     "agent_action_counts": {i: self.agent_action_counts.get(str(i + 1), 0) 
            #                         for i in range(self.num_players)},
            #     "workload_counts": workloads
            # }
            # # ================================================================

            
            # WORKLOAD DIAGNOSTIC
            total_meaningful_actions = sum(workloads)
            total_steps = self.timesteps
            meaningful_action_rate = total_meaningful_actions / max(1, total_steps)
            
            print(f"\n--- WORKLOAD DIAGNOSTIC ---")
            print(f"Total timesteps: {total_steps}")
            print(f"Total meaningful actions: {total_meaningful_actions}")
            print(f"Meaningful action rate: {meaningful_action_rate:.1%}")
            print(f"Average actions per agent: {total_meaningful_actions / self.num_players:.2f}")
            
            if meaningful_action_rate < 0.1:
                print(f"\n🔴 PROBLEM: Meaningful action rate < 10%!")
                print(f"   Agents are mostly taking noop/move actions")
            
            if any(w == 0 for w in workloads):
                zero_agents = [i for i, w in enumerate(workloads) if w == 0]
                print(f"\n⚠️  WARNING: Agents {zero_agents} have ZERO workload!")
            # ==========================================================================
            
            # Episode-level shaping is safe for actor-critic (IPPO/MAPPO),
            # and harmless for QMIX since we already shaped per-step.
            shaped_reward, fair_gne_metrics = self._fair_gne.process_episode(
                workloads, self.episode_reward
            )
            
            # ==================== COMPARE RESULTS ====================
            print(f"\n--- FAIR-GNE RESULTS ---")
            print(f"Fair-GNE JFI: {fair_gne_metrics['jfi']:.4f}")
            print(f"Manual JFI:   {manual_jfi:.4f}")
            
            jfi_diff = abs(fair_gne_metrics['jfi'] - manual_jfi)
            if jfi_diff < 0.001:
                print(f"✓ JFI calculations match (diff: {jfi_diff:.6f})")
            else:
                print(f"🔴 JFI MISMATCH (diff: {jfi_diff:.6f})")
            
            print(f"\nConstraint: JFI ({fair_gne_metrics['jfi']:.4f}) "
                f"{'≥' if fair_gne_metrics['jfi'] >= self._fair_gne.tau else '<'} "
                f"tau ({self._fair_gne.tau})")
            print(f"g_fair: {fair_gne_metrics['g_fair']:+.4f}")
            print(f"Lambda: {fair_gne_metrics['lambda']:.4f}")
            
            if fair_gne_metrics['lambda'] >= 9.9:
                print(f"⚠️  Lambda SATURATED - constraint persistently violated")
            
            print(f"\n--- REWARD ANALYSIS ---")
            print(f"Episode reward (before Fair-GNE): {self.episode_reward:+.2f}")
            print(f"Episode reward (after Fair-GNE):  {shaped_reward:+.2f}")
            print(f"Total penalty applied: {self.episode_reward - shaped_reward:+.2f}")
            
            if sum(workloads) > 0:
                penalty_per_action = (self.episode_reward - shaped_reward) / sum(workloads)
                print(f"Penalty per task action: {penalty_per_action:+.2f}")
                
                if abs(penalty_per_action) > 5.0:
                    print(f"⚠️  Penalty ({penalty_per_action:+.2f}) may be too strong!")
            
            print(f"{'='*80}\n")
            # =========================================================
            
            self.episode_reward = shaped_reward
            self._last_fair_gne_metrics = fair_gne_metrics

            # Surface diagnostics
            info['fair_gne_metrics'] = {
                **fair_gne_metrics,
                'tau': self._fair_gne.tau,
                'full_lagrangian': self._fair_gne.full_lagrangian,
                'step_shaping': bool(self._fair_gne_step_shaping),
            }

            if self._episode_counter % 10 == 0:
                print(f"[Fair-GNE] Ep {self._episode_counter:4d} | "
                    f"W: {workloads} | "
                    f"JFI: {fair_gne_metrics['jfi']:.3f} | "
                    f"λ: {fair_gne_metrics['lambda']:.3f} | "
                    f"g: {fair_gne_metrics['g_fair']:+.3f}")
        # ==========================================

        # Finalize step
        self.prev_step = (obs, reward, done, info)
        self.episode_reward += reward

        if self.pddl_goal_achieved and not self.goal_announced:
            self.goal_announced = True
            self.goals_reached += 1

        if hasattr(self.renderer, "canvas"):
            self.renderer.canvas.update_all_player_pos(obs.literals)

        return obs, reward, done, info        



    def _compute_L2_alignment_step(self):
        """Returns L2 (skill-task misalignment) for current episode prefix, or 0.0 if not available."""
        try:
            from utils.fairness_metrics_cal import compute_L2
        except Exception:
            return 0.0

        # Build task-skill pairs from the action log so far
        task_skill_pairs = []
        all_agents_skills = {i: self.agent_skills.get(i, {}) for i in range(self.num_players)}

        # Consider any action that appears in skill tables
        skill_actions = set()
        for d in all_agents_skills.values():
            skill_actions.update(d.keys())

        for e in self._action_log:
            action = self._normalize_action_name(e["action"])
            if action == "noop" or action not in skill_actions:
                continue
            aid = e["agent_id"]
            skill = float(all_agents_skills.get(aid, {}).get(action, 1.0))
            task_skill_pairs.append((action, aid, skill))

        if not task_skill_pairs:
            return 0.0

        try:
            return float(max(0.0, compute_L2(task_skill_pairs, all_agents_skills)))
        except Exception:
            return 0.0

    # -------------------- Episode logging --------------------

    def save_episode(self, filename):
        with open(filename, "w") as f:
            for action in self.taken_actions:
                f.write(str(action) + "\n")

    def get_episode_actions(self):
        return self.taken_actions

    def get_episode_goal_reached(self):
        """Returns 1 if the PDDL goal was achieved in this episode."""
        return 1 if self.pddl_goal_achieved else 0

    def _normalize_action_name(self, name: str) -> str:
        """Map *_simple variants back to the base task names so counts are consistent."""
        if not isinstance(name, str):
            name = str(name)
        if name.endswith("_simple"):
            base = name[:-7]  # strip "_simple"
            if base in {"compresschest", "giverescuebreaths", "giveshock", "givemedicine"}:
                return base
        return name

    def _get_config_weight(self, config_key, default=1):
        """Helper to get weight from config with fallback to default"""
        config_section = self.config.get(config_key, {})
        if isinstance(config_section, dict):
            # Try patient first, then default
            return config_section.get("patient", config_section.get("default", default))
        return default

    def calculate_robust_normalized_contributions(self, action_log):
        """Calculate normalized agent contributions that sum to ~1.0 regardless of task distribution."""
        agent_task_counts = {
            i: {"setup": 0, "compress": 0, "breathe": 0, "shock": 0, "medicine": 0,
                "moveitem": 0, "move": 0, "pickup": 0, "place": 0}
            for i in range(self.num_players)
        }

        # Fill in counts
        for entry in action_log:
            agent_id = entry["agent_id"]
            action = entry["action"]

            if action == "noop":
                continue

            normalized_action = action[:-7] if action.endswith("_simple") else action

            if any(prefix in action for prefix in ["stackunder", "stack"]):
                agent_task_counts[agent_id]["setup"] += 1
            elif "place" in normalized_action:
                agent_task_counts[agent_id]["place"] += 1
            elif normalized_action == "pick-up" or "pickup" in normalized_action:
                agent_task_counts[agent_id]["pickup"] += 1
            elif normalized_action == "compresschest":
                agent_task_counts[agent_id]["compress"] += 1
            elif normalized_action == "giverescuebreaths":
                agent_task_counts[agent_id]["breathe"] += 1
            elif normalized_action == "giveshock":
                agent_task_counts[agent_id]["shock"] += 1
            elif normalized_action == "givemedicine":
                agent_task_counts[agent_id]["medicine"] += 1
            elif "moveitem" in normalized_action:
                agent_task_counts[agent_id]["moveitem"] += 1
            elif normalized_action == "move":
                agent_task_counts[agent_id]["move"] += 1

        # Weights
        compress_weight = self._get_config_weight("num_compressions", default=3)
        breathe_weight = self._get_config_weight("num_breaths", default=2)
        shock_weight = self._get_config_weight("num_shocks", default=2)
        medicine_weight = self._get_config_weight("num_medicine_doses", default=1)

        setup_weight = 2
        moveitem_weight = 2
        move_weight = 2
        pickup_weight = 2
        place_weight = 2

        agent_weights = []
        for agent_id in range(self.num_players):
            weight = (
                agent_task_counts[agent_id]["setup"] / setup_weight +
                agent_task_counts[agent_id]["compress"] / compress_weight +
                agent_task_counts[agent_id]["breathe"] / breathe_weight +
                agent_task_counts[agent_id]["shock"] / shock_weight +
                agent_task_counts[agent_id]["medicine"] / medicine_weight +
                agent_task_counts[agent_id]["moveitem"] / moveitem_weight +
                agent_task_counts[agent_id]["move"] / move_weight +
                agent_task_counts[agent_id]["pickup"] / pickup_weight +
                agent_task_counts[agent_id]["place"] / place_weight
            )
            agent_weights.append(weight)

        total_weight = sum(agent_weights)
        if total_weight > 0:
            normalized_contributions = [w / total_weight for w in agent_weights]
        else:
            normalized_contributions = [1.0 / self.num_players] * self.num_players

        return normalized_contributions, agent_task_counts

    def _calculate_skill_task_alignment_metrics(self):
        """Compute alignment/frequency metrics; logging only, no shaping happens here."""
        all_tracked_actions = ["compresschest", "giverescuebreaths", "giveshock", "givemedicine",
                               "move", "moveitem", "pick-up", "place", "stack", "stackunder"]

        agent_task_counts = {
            i: {task: 0 for task in all_tracked_actions}
            for i in range(self.num_players)
        }
        agent_task_timesteps = {
            i: {task: None for task in all_tracked_actions}
            for i in range(self.num_players)
        }

        for entry in self._action_log:
            action = self._normalize_action_name(entry["action"])
            agent_id = entry["agent_id"]
            timestep = entry["timestep"]
            if action in agent_task_counts[agent_id]:
                agent_task_counts[agent_id][action] += 1
                if agent_task_timesteps[agent_id][action] is None:
                    agent_task_timesteps[agent_id][action] = timestep

        # Skills by agent
        agent_skills = {}
        for i in range(self.num_players):
            robot_name = f"robot{i + 1}"
            agent_skills[i] = self.config.get("player_info", {}).get(robot_name, {})

        # Frequencies
        task_frequencies = {}
        for task in all_tracked_actions:
            total = sum(agent_task_counts[i][task] for i in range(self.num_players))
            if total > 0:
                task_frequencies[task] = {
                    f"agent_{i}": agent_task_counts[i][task] / total
                    for i in range(self.num_players)
                }
                task_frequencies[task]["total_count"] = total
            else:
                task_frequencies[task] = {f"agent_{i}": 0 for i in range(self.num_players)}
                task_frequencies[task]["total_count"] = 0

        # Best-skilled agent per task
        best_agents = {}
        for task in all_tracked_actions:
            best_skill = 0
            best_agent = None
            for i in range(self.num_players):
                skill = agent_skills[i].get(task, 1.0)
                if skill > best_skill:
                    best_skill = skill
                    best_agent = i
            best_agents[task] = {"agent_id": best_agent, "skill": best_skill}

        alignment_scores = {}
        specialist_performance = {}
        for task in all_tracked_actions:
            if best_agents[task]["agent_id"] is not None:
                best_agent_id = best_agents[task]["agent_id"]
                total_count = task_frequencies[task]["total_count"]
                if total_count > 0:
                    specialist_freq = task_frequencies[task][f"agent_{best_agent_id}"]
                    alignment_scores[task] = specialist_freq
                    specialist_first_time = agent_task_timesteps[best_agent_id][task]
                    all_first_times = [
                        agent_task_timesteps[i][task]
                        for i in range(self.num_players)
                        if agent_task_timesteps[i][task] is not None
                    ]
                    specialist_performance[task] = {
                        "specialist_agent_id": best_agent_id,
                        "specialist_skill": best_agents[task]["skill"],
                        "specialist_frequency": specialist_freq,
                        "specialist_count": agent_task_counts[best_agent_id][task],
                        "specialist_first_action_timestep": specialist_first_time,
                        "earliest_action_timestep": min(all_first_times) if all_first_times else None,
                        "response_delay": (specialist_first_time - min(all_first_times))
                        if all_first_times and specialist_first_time is not None else None
                    }
                else:
                    alignment_scores[task] = None
                    specialist_performance[task] = {
                        "specialist_agent_id": best_agent_id,
                        "specialist_skill": best_agents[task]["skill"],
                        "specialist_frequency": 0,
                        "specialist_count": 0,
                        "specialist_first_action_timestep": None,
                        "earliest_action_timestep": None,
                        "response_delay": None
                    }
            else:
                alignment_scores[task] = None
                specialist_performance[task] = {
                    "specialist_agent_id": None,
                    "specialist_skill": 1.0,
                    "specialist_frequency": 0,
                    "specialist_count": 0,
                    "specialist_first_action_timestep": None,
                    "earliest_action_timestep": None,
                    "response_delay": None
                }

        valid_scores = [s for s in alignment_scores.values() if s is not None]
        overall_alignment = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        specialists = {}
        for task in all_tracked_actions:
            best_agent_id = None
            best_skill = 1
            for agent_id in range(self.num_players):
                robot_name = f"robot{agent_id + 1}"
                skill = self.config.get("player_info", {}).get(robot_name, {}).get(task, 1)
                if skill > best_skill:
                    best_skill = skill
                    best_agent_id = agent_id
            if best_agent_id is not None and best_skill > 1:
                specialists[task] = {"agent_id": best_agent_id, "skill": best_skill}
            else:
                specialists[task] = None

        specific_alignment = {}
        medical_tasks = [
            ("compresschest", "chest_compressions"),
            ("giverescuebreaths", "rescue_breaths"),
            ("giveshock", "shock_delivery"),
            ("givemedicine", "medicine_administration")
        ]
        for task_key, display_name in medical_tasks:
            if specialists.get(task_key):
                specialist_id = specialists[task_key]["agent_id"]
                specific_alignment[display_name] = {
                    **{f"agent_{i}_count": agent_task_counts[i][task_key] for i in range(self.num_players)},
                    "specialist_agent_id": specialist_id,
                    "specialist_skill": specialists[task_key]["skill"],
                    "specialist_frequency": task_frequencies[task_key].get(f"agent_{specialist_id}", 0),
                    "alignment_interpretation": f"Higher agent_{specialist_id} frequency = better alignment"
                }
            else:
                specific_alignment[display_name] = {
                    **{f"agent_{i}_count": agent_task_counts[i][task_key] for i in range(self.num_players)},
                    "specialist_agent_id": None,
                    "specialist_skill": None,
                    "specialist_frequency": 0,
                    "alignment_interpretation": "No specialist - all agents have skill <= 1"
                }

        auxiliary_tasks = [
            ("move", "movement"),
            ("moveitem", "item_transportation"),
            ("pick-up", "item_pickup"),
            ("place", "item_placement"),
            ("stack", "item_stacking"),
            ("stackunder", "stackunder_operations")
        ]
        for task_key, display_name in auxiliary_tasks:
            if specialists.get(task_key):
                specialist_id = specialists[task_key]["agent_id"]
                specific_alignment[display_name] = {
                    **{f"agent_{i}_count": agent_task_counts[i][task_key] for i in range(self.num_players)},
                    "specialist_agent_id": specialist_id,
                    "specialist_skill": specialists[task_key]["skill"],
                    "specialist_frequency": task_frequencies[task_key].get(f"agent_{specialist_id}", 0),
                    "alignment_interpretation": f"Higher agent_{specialist_id} frequency = better alignment"
                }
            else:
                specific_alignment[display_name] = {
                    **{f"agent_{i}_count": agent_task_counts[i][task_key] for i in range(self.num_players)},
                    "specialist_agent_id": None,
                    "specialist_skill": None,
                    "specialist_frequency": 0,
                    "alignment_interpretation": "No specialist - all agents have skill <= 1"
                }

        return {
            "task_frequencies": task_frequencies,
            "alignment_scores": alignment_scores,
            "overall_alignment_score": overall_alignment,
            "specialist_performance": specialist_performance,
            "specific_task_alignment": specific_alignment,
            "agent_task_counts": {str(k): v for k, v in agent_task_counts.items()},
            "best_agents_for_tasks": best_agents,
            "interpretation": {
                "overall_alignment_score": "0=random assignment, 1=perfect specialist usage",
                "alignment_scores": "Per-task: fraction done by the best-skilled agent",
                "response_delay": "Timesteps between first action and specialist's first action"
            }
        }

    def save_current_episode_log(self):
        """Save episode log with action history, per-agent counts, and milestones."""
        if not (hasattr(self, "_log_test_mode") and self._log_test_mode and hasattr(self, "_log_dir")):
            return  # only log when test-mode logging is enabled

        os.makedirs(self._log_dir, exist_ok=True)
        filename_gz = os.path.join(self._log_dir, f"episode_{self._episode_counter}.json.gz")

        # Helper to get requirements from config
        def _get_requirement(action_key: str, item_key: str = "patient") -> int:
            mapping = {
                "compresschest": "num_compressions",
                "giverescuebreaths": "num_breaths",
                "giveshock": "num_shocks",
                "givemedicine": "num_medicine_doses"
            }
            cfg_key = mapping.get(action_key)
            if not cfg_key:
                return 1
            bucket = self.config.get(cfg_key, {})
            if isinstance(bucket, dict):
                return int(bucket.get(item_key, bucket.get("default", 1)))
            return 1

        # Normalized contributions
        normalized_contribs, task_breakdown = self.calculate_robust_normalized_contributions(self._action_log)

        # Totals from medical progress
        total_actions_per_agent = {}
        for action_type, agent_counts in self.agent_medical_progress.items():
            for agent_id, count in agent_counts.items():
                total_actions_per_agent[agent_id] = total_actions_per_agent.get(agent_id, 0) + count

        milestones = {}
        if hasattr(self.reward_handler, "get_milestone_status"):
            try:
                milestones = self.reward_handler.get_milestone_status() or {}
            except Exception:
                pass

        total_compressions = sum(self.agent_medical_progress.get("compresschest", {}).values())
        total_breaths = sum(self.agent_medical_progress.get("giverescuebreaths", {}).values())
        total_shocks = sum(self.agent_medical_progress.get("giveshock", {}).values())
        total_meds = sum(self.agent_medical_progress.get("givemedicine", {}).values())

        need_compressions = _get_requirement("compresschest", "patient")
        need_breaths = _get_requirement("giverescuebreaths", "patient")
        need_shocks = _get_requirement("giveshock", "patient")
        need_meds = _get_requirement("givemedicine", "patient")

        milestones["chest_compressed"] = 1 if total_compressions >= need_compressions else 0
        milestones["rescue_breaths"] = 1 if total_breaths >= need_breaths else 0
        milestones["shock"] = 1 if total_shocks >= need_shocks else 0
        milestones["medicine_administered"] = 1 if total_meds >= need_meds else 0

        final_predicates = []
        try:
            if self.prev_step and self.prev_step[0]:
                final_predicates = [str(lit) for lit in self.prev_step[0].literals]
        except Exception:
            pass

        action_strings = []
        for action in self.taken_actions:
            try:
                action_strings.append(str(action))
            except:
                action_strings.append(repr(action))

        # Agent contributions summary
        agent_contributions = {}
        for agent_id in range(self.num_players):
            agent_skills = self.agent_skills.get(agent_id, {})
            skill_weighted_actions = {
                "compresschest": self.agent_medical_progress["compresschest"].get(agent_id, 0) * agent_skills.get("compresschest", 1),
                "giverescuebreaths": self.agent_medical_progress["giverescuebreaths"].get(agent_id, 0) * agent_skills.get("giverescuebreaths", 1),
                "giveshock": self.agent_medical_progress["giveshock"].get(agent_id, 0) * agent_skills.get("giveshock", 1),
                "givemedicine": self.agent_medical_progress["givemedicine"].get(agent_id, 0) * agent_skills.get("givemedicine", 1)
            }
            agent_contributions[agent_id] = {
                "total_actions": self.agent_action_counts.get(str(agent_id + 1), 0),
                "normalized_contribution": normalized_contribs[agent_id],
                "medical_actions": {
                    "compresschest": self.agent_medical_progress["compresschest"].get(agent_id, 0),
                    "giverescuebreaths": self.agent_medical_progress["giverescuebreaths"].get(agent_id, 0),
                    "giveshock": self.agent_medical_progress["giveshock"].get(agent_id, 0),
                    "givemedicine": self.agent_medical_progress["givemedicine"].get(agent_id, 0)
                },
                "task_breakdown": task_breakdown[agent_id],
                "agent_skills": agent_skills,
                "skill_weighted_actions": skill_weighted_actions,
                "total_skill_contributed": sum(skill_weighted_actions.values())
            }

        # Simple fairness metric on normalized contributions
        import statistics
        fairness_score = 1.0 - statistics.stdev(normalized_contribs) if len(normalized_contribs) > 1 else 1.0

        # Detailed fairness metrics
        try:
            from utils.fairness_metrics_cal import compute_L1, compute_L2, compute_L3
        except Exception:
            compute_L1 = lambda x: 0.0
            compute_L2 = lambda x, y: 0.0
            compute_L3 = lambda x, y, z: 0.0

        raw_action_counts = [self.agent_action_counts.get(str(i + 1), 0) for i in range(self.num_players)]
        L1_raw = compute_L1(raw_action_counts) if sum(raw_action_counts) > 0 else 0.0

        task_skill_pairs = []
        all_agents_skills = {i: self.config.get("player_info", {}).get(f"robot{i+1}", {}) for i in range(self.num_players)}
        tracked_actions = set()
        for v in all_agents_skills.values():
            tracked_actions.update(v.keys())

        for entry in self._action_log:
            action = entry["action"]
            agent_id = entry["agent_id"]
            if action == "noop":
                continue
            normalized_action = action[:-7] if action.endswith("_simple") else action
            if normalized_action in tracked_actions:
                skill = all_agents_skills.get(agent_id, {}).get(normalized_action, 1.0)
                task_skill_pairs.append((normalized_action, agent_id, skill))

        L2 = compute_L2(task_skill_pairs, all_agents_skills) if task_skill_pairs else 0.0
        L3 = compute_L3(L1_raw, L2, alpha=0.5)

        # ===== COMPUTE JFI (JAIN'S FAIRNESS INDEX) =====
        workloads = self._compute_total_workloads()
        if sum(workloads) > 0:
            sum_workloads = sum(workloads)
            sum_squared = sum(w**2 for w in workloads)
            jfi = (sum_workloads ** 2) / (self.num_players * sum_squared) if sum_squared > 0 else 1.0
        else:
            jfi = 1.0

        # ===== CACHE METRICS FOR EPISODE_RUNNER (FOR WANDB LOGGING) =====
        base_metrics = {
            "L1": L1_raw,
            "L2": L2,
            "L3": L3,
            "jfi": jfi,
            "normalized_contributions": normalized_contribs,
            "agent_action_counts": {i: self.agent_action_counts.get(str(i + 1), 0) 
                                   for i in range(self.num_players)},
            "workload_counts": workloads
        }
        
        # ===== ALWAYS ADD FAIR-GNE METRICS (EVEN IF DISABLED) =====
        if self._use_fair_gne and hasattr(self, '_last_fair_gne_metrics') and self._last_fair_gne_metrics:
            # Fair-GNE ENABLED
            base_metrics.update({
                "fair_gne_enabled": True,
                "fair_gne_jfi": float(self._last_fair_gne_metrics.get('jfi', jfi)),
                "fair_gne_lambda": float(self._last_fair_gne_metrics.get('lambda', 0.0)),
                "fair_gne_g_fair": float(self._last_fair_gne_metrics.get('g_fair', 0.0)),
                "fair_gne_penalty": float(self._last_fair_gne_metrics.get('penalty', 0.0)),
                "fair_gne_shaped_reward": float(self._last_fair_gne_metrics.get('shaped_reward', self.episode_reward)),
                "fair_gne_constraint_satisfied": bool(self._last_fair_gne_metrics.get('jfi', jfi) >= self._fair_gne.tau),
                "fair_gne_tau": float(self._fair_gne.tau),
            })
        else:
            # Fair-GNE DISABLED
            base_metrics.update({
                "fair_gne_enabled": False,
                "fair_gne_jfi": float(jfi),
                "fair_gne_lambda": 0.0,
                "fair_gne_g_fair": 0.0,
                "fair_gne_penalty": 0.0,
                "fair_gne_shaped_reward": float(self.episode_reward),
                "fair_gne_constraint_satisfied": None,
                "fair_gne_tau": 0.85,
            })
        
        self._last_episode_metrics = base_metrics

        # ================================================
        
        # # ===== CACHE METRICS FOR EPISODE_RUNNER (FOR WANDB LOGGING) =====
        # self._last_episode_metrics = {
        #     "L1": L1_raw,
        #     "L2": L2,
        #     "L3": L3,
        #     "jfi": jfi,
        #     "normalized_contributions": normalized_contribs,
        #     "agent_action_counts": {i: self.agent_action_counts.get(str(i + 1), 0) 
        #                            for i in range(self.num_players)},
        #     "workload_counts": workloads
        # }
        # # ================================================================
        
        # ===== ADD FAIR-GNE METRICS (NEW) =====
        fair_gne_data = {}
        print("Fair-GNE:", self._use_fair_gne, "has_metrics:", hasattr(self, "_last_fair_gne_metrics"))
        if self._use_fair_gne and hasattr(self, '_last_fair_gne_metrics'):
            fair_gne_data = {
                "enabled": True,
                "jfi": float(self._last_fair_gne_metrics.get('jfi', 0.0)),
                "tau": float(self._fair_gne.tau),
                "g_fair": float(self._last_fair_gne_metrics.get('g_fair', 0.0)),
                "lambda": float(self._last_fair_gne_metrics.get('lambda', 0.0)),
                "penalty": float(self._last_fair_gne_metrics.get('penalty', 0.0)),
                "constraint_satisfied": bool(self._last_fair_gne_metrics.get('jfi', 0.0) >= self._fair_gne.tau),
                "lambda_saturated": bool(self._last_fair_gne_metrics.get('lambda', 0.0) >= 0.99 * self._fair_gne.lambda_max),
                "workloads": self._compute_total_workloads(),
                "hyperparameters": {
                    "dual_lr": float(self._fair_gne.dual_lr),
                    "lambda_max": float(self._fair_gne.lambda_max),
                    "update_freq": int(self._fair_gne.update_freq),
                    "step_shaping": bool(self._fair_gne_step_shaping),
                    "full_lagrangian": bool(self._fair_gne.full_lagrangian)
                }
            }
        else:
            fair_gne_data = {
                "enabled": False,
                "jfi": float(jfi), 
            }

        episode_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "episode_id": self._episode_counter,
            "test_mode": self._log_test_mode,
            "goal_achieved": bool(self.pddl_goal_achieved),
            "total_timesteps": int(self.timesteps),
            "episode_reward": float(self.episode_reward),

            "actions": action_strings,
            "final_state_predicates": final_predicates,

            "agent_action_counts": dict(self.agent_action_counts),
            "agent_contributions": agent_contributions,
            "agent_medical_events": dict(self.agent_medical_events),

            "normalized_contributions": normalized_contribs,
            "fairness_metrics": {
                "simple_fairness_score": fairness_score,
                "L1_workload_imbalance": L1_raw,
                "L2_skill_misalignment": L2,
                "L3_composite": L3,
		        "JFI_eps_data": jfi,
                "workload_range": max(normalized_contribs) - min(normalized_contribs),
                
                # Fair-GNE metrics (NEW)
                "fair_gne": fair_gne_data,
                "interpretation": {
                    "L1": "0=perfect balance, 1=maximum imbalance (Gini index)",
                    "L2": "0=optimal skill usage, 1=worst skill usage",
                    "L3": "Composite score (0.5*L1 + 0.5*L2)",
                    "workload_range": "Max - Min contribution (0=perfect, 1=one agent did everything)",
                    "fair_gne_jfi": "Jain's Fairness Index (0=unfair, 1=perfect fairness)",
                    "fair_gne_g_fair": "Constraint violation (positive=violated, negative=satisfied)",
                    "fair_gne_lambda": "Lagrange multiplier (penalty strength)"
                }
            },

            "skill_task_alignment": self._calculate_skill_task_alignment_metrics(),

            "task_breakdown_by_agent": task_breakdown,
            "workload_counts": workloads,
            "workload_jfi": float(jfi),
            "skill_analysis": {
                "task_skill_pairs": [(t, a, float(s)) for t, a, s in task_skill_pairs],
                "total_skill_used": sum(s for _, _, s in task_skill_pairs),
                "agents_skill_levels": {str(k): v for k, v in all_agents_skills.items()}
            },

            "milestones": milestones,

            "task_totals": {
                "compresschest": total_compressions,
                "giverescuebreaths": total_breaths,
                "giveshock": total_shocks,
                "givemedicine": total_meds
            },

            "task_requirements": {
                "compresschest": need_compressions,
                "giverescuebreaths": need_breaths,
                "giveshock": need_shocks,
                "givemedicine": need_meds
            },

            "action_type_summary": {
                "medical_actions": sum(1 for e in self._action_log if e["action"] in [
                    "compresschest", "giverescuebreaths", "giveshock", "givemedicine",
                    "compresschest_simple", "giverescuebreaths_simple", "giveshock_simple", "givemedicine_simple"
                ]),
                "move_actions": sum(1 for e in self._action_log if e["action"] == "move"),
                "moveitem_actions": sum(1 for e in self._action_log if "moveitem" in e["action"]),
                "pickup_actions": sum(1 for e in self._action_log if e["action"] == "pick-up" or "pickup" in e["action"]),
                "place_actions": sum(1 for e in self._action_log if "place" in e["action"]),
                "stack_stackunder_actions": sum(1 for e in self._action_log if any(prefix in e["action"] for prefix in ["stackunder", "stack"])),
                "noop_actions": sum(1 for e in self._action_log if e["action"] == "noop"),
                "other_actions": sum(1 for e in self._action_log if e["action"] not in [
                    "noop", "move", "pick-up", "compresschest", "giverescuebreaths", "giveshock", "givemedicine"
                ] and not any(x in e["action"] for x in ["moveitem", "stackunder", "stack", "place", "pickup", "_simple"]))
            }
        }

        # Save to file
        try:
            with gzip.open(filename_gz, "wt", encoding="utf-8") as f:
                json.dump(episode_data, f, indent=2)
        except Exception:
            filename_json = filename_gz.replace('.json.gz', '.json')
            try:
                with open(filename_json, "w") as f:
                    json.dump(episode_data, f, indent=2)
            except Exception as e2:
                print(f"[WRAPPER] Failed to save episode log: {e2}")


    def get_last_fairness_metrics(self):
        """
        Return cached fairness metrics computed at episode end.
        Used by episode_runner to log metrics to WandB.
        """
        result = self._last_episode_metrics if hasattr(self, '_last_episode_metrics') else None
        
        # ===== DEBUG LOGGING =====
        print(f"\n[get_last_fairness_metrics] Called")
        print(f"  hasattr: {hasattr(self, '_last_episode_metrics')}")
        print(f"  result is None: {result is None}")
        if result:
            print(f"  Keys: {list(result.keys())}")
            print(f"  JFI: {result.get('jfi', 'MISSING')}")
            print(f"  L1: {result.get('L1', 'MISSING')}")
            print(f"  fair_gne_enabled: {result.get('fair_gne_enabled', 'MISSING')}")
        else:
            print(f"  ❌ RETURNING NONE!")
            if hasattr(self, '_last_episode_metrics'):
                print(f"  _last_episode_metrics value: {self._last_episode_metrics}")
        # =========================
        
        return result
        
    def reset(self):
        obs, _ = self.env.reset()
        if hasattr(self.renderer, "canvas"):
            self.renderer.canvas.reset_player_positions()

        expanded_truths, expanded_states = pddlgym_utils.expand_state(
            obs.literals, obs.objects
        )
        info = {
            "timesteps": self.timesteps,
            "expanded_truths": expanded_truths,
            "expanded_states": expanded_states,
            "toggle_array": None,
            "state": {},
        }
        self.prev_step = (obs, 0.0, False, info)

        # Episode counters
        self.timesteps = 0
        self.move_counter = 0
        self.state = {}
        self.taken_actions = []
        self.num_players = self._count_players(obs)
        self.episode_reward = 0.0
        self.goal_announced = False
        self.pddl_goal_achieved = False

        # Clear per-agent analytics
        self.agent_action_counts.clear()
        for d in self.agent_medical_progress.values():
            d.clear()
        self.agent_medical_events = {
            "chest_compressed_by": None,
            "rescue_breaths_by": None,
            "shock_by": None,
            "medicine_by": None,
        }
        self._last_actor_id = None
        self._action_log = []

        self._workload_cache = [0] * self.num_players  # Reset workload cache
        self._last_episode_metrics = None  # Clear cache for next episode
        

        # Reset heuristic handler if supported
        if hasattr(self.reward_handler, 'reset'):
            self.reward_handler.reset()

        # Reset FEN episode windows if enabled
        if self._fen:
            self._fen.reset_episode()

        return obs, info
