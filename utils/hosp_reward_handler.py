from utils.reward_handler import RewardHandler
import pddlgym.inference
from pddlgym.structs import Literal, LiteralConjunction, LiteralDisjunction


class HospRewardHandler(RewardHandler):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ===== TOGGLE THIS TO SWITCH BETWEEN MODES =====
        self.use_flexible_rewards = True  # Set to True for flexible mode
        
        self.correct_order = ["cpr_board", "patient", "pump", "aed", "syringe"]
        self.max_possible_reward = 700

        # Goal tracking flags
        self.goal_reached = False

        # Milestone tracking
        self.milestones_reached = {
            "chest_compressed": False,
            "rescue_breaths": False,
            "shock": False,
            "medicine_administered": False,
            "correct_stacking": False
        }

        # Timestamp tracking for milestones (in timesteps)
        self.milestone_timestamps = {
            "chest_compressed": None,
            "rescue_breaths": None,
            "shock": None,
            "medicine_administered": None,
            "correct_stacking": None
        }

        # Track item movement for moveitem tasks
        self.items_moved = set()
        self.item_pickup_rewards = {
            "cpr_board": False,
            "pump": False,
            "syringe": False
        }

        # Counter for tracking progress
        self.step_counter = 0
        self.previous_obs = None
        self.previous_held_items = set()
        
        # For flexible mode - track what's been rewarded
        self.flexible_rewarded_items = set()
        self.flexible_rewarded_placements = set()
        
        # Track player locations for movement rewards
        self.previous_player_locations = {}

    # -------------------- Small helpers --------------------

    def _find_stacking_index(self, item1, item2):
        for i in range(len(self.correct_order) - 1):
            if self.correct_order[i] in item1 and self.correct_order[i + 1] in item2:
                return i
        return -1

    def _check_item_on_station(self, obs, item, station):
        for literal in obs.literals:
            if (
                "on" == literal.predicate.name
                and item in literal.variables[0].name
                and station in literal.variables[1].name
            ):
                return True
        return False

    def _check_item_on_item(self, obs, top_item, bottom_item):
        for literal in obs.literals:
            if (
                "atop" == literal.predicate.name
                and top_item in literal.variables[0].name
                and bottom_item in literal.variables[1].name
            ):
                return True
        return False

    def _check_item_held(self, obs, item):
        for literal in obs.literals:
            if "has" == literal.predicate.name and item in literal.variables[1].name:
                return True
        return False

    def _check_predicate(self, obs, predicate, item):
        for literal in obs.literals:
            if (
                predicate == literal.predicate.name
                and item in literal.variables[0].name
            ):
                return True
        return False

    def _check_player_at_station(self, obs, station):
        """Check if any player is at the specified station"""
        for literal in obs.literals:
            if (
                "loc" == literal.predicate.name
                and station in literal.variables[1].name
            ):
                return True
        return False

    def _check_action_progress(self, state, action, item="patient1"):
        item_status = state.get(item, {})
        return item_status.get(action, 0)

    def _calculate_action_reward(self, progress, max_progress):
        return min(20, 20 * progress / max_progress)

    # -------------------- New helper methods for flexible mode --------------------
    
    def _get_player_locations(self, obs):
        """Get the current location of all players"""
        locations = {}
        for literal in obs.literals:
            if literal.predicate.name == "loc" and len(literal.variables) >= 2:
                player = literal.variables[0].name
                location = literal.variables[1].name
                locations[player] = location
        return locations

    def _check_player_holding_item(self, obs, player):
        """Check what item a specific player is holding"""
        for literal in obs.literals:
            if (literal.predicate.name == "has" and 
                len(literal.variables) >= 2 and
                player in literal.variables[0].name):
                return literal.variables[1].name
        return None

    def _get_player_skill(self, player_name, skill_type):
        """Get a player's skill level for a specific action"""
        if 'player_info' in self.config:
            player_info = self.config['player_info'].get(player_name, {})
            return player_info.get(skill_type, 1)
        return 1

    def _check_player_tired(self, obs, player):
        """Check if a specific player is tired"""
        for literal in obs.literals:
            if literal.predicate.name == "istired" and player in literal.variables[0].name:
                return True
        return False

    # -------------------- Reset & milestone export --------------------

    def reset(self):
        """Reset the goal reached flags between episodes"""
        self.goal_reached = False

        # Reset milestone tracking
        for key in self.milestones_reached:
            self.milestones_reached[key] = False
            self.milestone_timestamps[key] = None

        self.step_counter = 0
        self.previous_obs = None
        self.previous_held_items = set()
        self.items_moved = set()

        # Reset item pickup rewards
        for key in getattr(self, "item_pickup_rewards", {}):
            self.item_pickup_rewards[key] = False
            
        # Reset flexible mode tracking
        self.flexible_rewarded_items.clear()
        self.flexible_rewarded_placements.clear()
        self.previous_player_locations = {}

    def get_milestone_status(self):
        """Returns the milestone status dictionary for logging"""
        status = {}
        for k, v in self.milestones_reached.items():
            status[k] = 1 if v else 0
        for k, v in self.milestone_timestamps.items():
            status[f"{k}_timestep"] = v
        return status

    # -------------------- Internal goal parsing helpers --------------------

    def _is_moveitem_task(self, obs):
        """Determine whether the current goal is a moveitem task or a medical one."""
        def extract_all_literals(goal_node):
            if isinstance(goal_node, Literal):
                return [goal_node]
            elif isinstance(goal_node, (LiteralConjunction, LiteralDisjunction)):
                literals = []
                for g in goal_node.literals:
                    literals.extend(extract_all_literals(g))
                return literals
            else:
                raise ValueError(f"Unsupported goal type: {type(goal_node)}")

        try:
            goal_literals = extract_all_literals(obs.goal)
        except Exception as e:
            print(f"[ERROR] Failed to extract literals from goal: {e}")
            return False

        medical_predicates = {"ischestcompressed", "isrescuebreathed", "isshocked", "istreated"}
        for goal in goal_literals:
            if goal.predicate.name in medical_predicates:
                return False

        return True

    def _update_medical_milestones(self, obs, env_timestep=None):
        """Update medical milestones based on predicates"""
        mapping = [
            ("ischestcompressed", "chest_compressed"),
            ("isrescuebreathed", "rescue_breaths"),
            ("isshocked", "shock"),
            ("istreated", "medicine_administered"),
        ]
        for pred, key in mapping:
            if (not self.milestones_reached[key]
                and self._check_predicate(obs, pred, "patient1")):
                self.milestones_reached[key] = True
                self.milestone_timestamps[key] = (
                    env_timestep if env_timestep is not None else self.step_counter
                )

    # -------------------- Main reward function --------------------

    def heuristic_reward(self, obs, state, env_timestep=None, bump_counter=True):
        """Main reward function that switches based on mode"""
        if bump_counter:
            self.step_counter += 1

        # Always update medical milestones first
        self._update_medical_milestones(obs, env_timestep)

        # Moveitem tasks use same logic in both modes
        if self._is_moveitem_task(obs):
            return self._calculate_moveitem_rewards(obs, state)

        # Use different reward logic based on mode
        if self.use_flexible_rewards:
            return self._flexible_heuristic_reward(obs, state)
        else:
            return self._ordered_heuristic_reward(obs, state)

    def _flexible_heuristic_reward(self, obs, state):
        """Flexible reward logic optimized for *_simple actions"""
        # Goal check
        if pddlgym.inference.check_goal(obs, obs.goal):
            self.goal_reached = True
            if self._check_predicate(obs, "istreated", "patient1"):
                return 700
            elif self._check_predicate(obs, "isshocked", "patient1"):
                return 500
            elif self._check_predicate(obs, "isrescuebreathed", "patient1"):
                return 300  # Higher reward for simple path completion
            else:
                return 500

        score = 0
        
        # Get current game state info
        chest_compressed = self._check_predicate(obs, "ischestcompressed", "patient1")
        breaths_given = self._check_predicate(obs, "isrescuebreathed", "patient1")
        
        # PRIORITY 1: Reward positioning for *_simple actions
        current_locations = self._get_player_locations(obs)
        for player, location in current_locations.items():
            if "patient_bed_station" in location or "patient_legs" in location:
                held_item = self._check_player_holding_item(obs, player)
                
                # KEY CHANGE: Reward being at patient WITHOUT items (for simple actions)
                if not held_item:
                    is_tired = self._check_player_tired(obs, player)
                    
                    if not is_tired:
                        # Check player's skills to determine best reward
                        compress_skill = self._get_player_skill(player, "compresschest")
                        breathe_skill = self._get_player_skill(player, "giverescuebreaths")
                        
                        # Phase 1: Chest compressions needed
                        if not chest_compressed:
                            if compress_skill >= 2:
                                score += 30  # High reward for compression specialist in position
                            else:
                                score += 15  # Moderate reward for any capable agent
                        
                        # Phase 2: Rescue breaths needed
                        elif not breaths_given:
                            if breathe_skill >= 2:
                                score += 35  # High reward for breathing specialist in position
                            else:
                                score += 20  # Moderate reward for any capable agent
                    else:
                        # Tired agent at patient - small penalty to encourage rotation
                        score += 0  # No reward for tired agents at patient
        
        # PRIORITY 2: Reward movement toward patient when appropriate
        prev_locations = getattr(self, 'previous_player_locations', {})
        for player, current_loc in current_locations.items():
            prev_loc = prev_locations.get(player)
            if prev_loc and prev_loc != current_loc:
                held_item = self._check_player_holding_item(obs, player)
                
                # Reward moving to patient area without items
                if not held_item and ("patient_bed" in current_loc or "patient_legs" in current_loc):
                    is_tired = self._check_player_tired(obs, player)
                    
                    if not is_tired:
                        compress_skill = self._get_player_skill(player, "compresschest")
                        breathe_skill = self._get_player_skill(player, "giverescuebreaths")
                        
                        if not chest_compressed and compress_skill >= 1:
                            score += 20  # Good move toward compression
                        elif chest_compressed and not breaths_given and breathe_skill >= 1:
                            score += 25  # Good move toward breathing
        
        # PRIORITY 3: Energy management rewards
        for player in current_locations:
            is_tired = self._check_player_tired(obs, player)
            
            # Reward non-tired, skilled agents being available
            if not is_tired:
                compress_skill = self._get_player_skill(player, "compresschest")
                breathe_skill = self._get_player_skill(player, "giverescuebreaths")
                
                if not chest_compressed and compress_skill >= 2:
                    score += 5  # Bonus for having compression specialist ready
                elif chest_compressed and not breaths_given and breathe_skill >= 2:
                    score += 5  # Bonus for having breathing specialist ready
        
        # PRIORITY 4: Strong rewards for simple action progress
        compress_progress = self._check_action_progress(state, "compresschest")
        breathe_progress = self._check_action_progress(state, "giverescuebreaths")
        
        # Reward ANY compression progress highly
        if compress_progress > 0:
            score += 40 * compress_progress
        
        # Reward ANY breathing progress highly
        if breathe_progress > 0:
            score += 50 * breathe_progress
        
        # PRIORITY 5: Milestone bonuses for simple path
        if self.milestones_reached["chest_compressed"]:
            score += 80  # Big bonus for compression milestone
        if self.milestones_reached["rescue_breaths"]:
            score += 100  # Big bonus for breathing milestone
        
        # REDUCED PRIORITY: Traditional item-based path
        # Still reward it but much less than simple path
        for item in ["cpr_board", "pump", "aed", "syringe"]:
            if self._check_item_held(obs, item) and item not in self.flexible_rewarded_items:
                score += 5  # Reduced from 15
                self.flexible_rewarded_items.add(item)
        
        # Placement rewards (reduced)
        placements = [
            ("cpr_board", "patient_bed_station", "on_station", 10),  # Reduced from 30
            ("pump", "patient", "on_item", 15),  # Reduced from 35
            ("aed", "pump", "on_item", 15),  # Reduced from 35
            ("syringe", "aed", "on_item", 10),  # Reduced from 30
        ]
        
        for item, location, placement_type, reward in placements:
            key = f"{item}_{placement_type}_{location}"
            if key not in self.flexible_rewarded_placements:
                if placement_type == "on_item":
                    if self._check_item_on_item(obs, item, location):
                        score += reward
                        self.flexible_rewarded_placements.add(key)
                else:
                    if self._check_item_on_station(obs, item, location):
                        score += reward
                        self.flexible_rewarded_placements.add(key)
        
        # Update location tracking
        self.previous_player_locations = current_locations
        
        return score

    def _ordered_heuristic_reward(self, obs, state):
        """Original sequential reward logic"""
        # Goal check
        if pddlgym.inference.check_goal(obs, obs.goal):
            self.goal_reached = True
            if self._check_predicate(obs, "istreated", "patient1"):
                return 700
            elif self._check_predicate(obs, "isshocked", "patient1"):
                return 500
            elif (self._check_predicate(obs, "isrescuebreathed", "patient1")
                  and not self._check_predicate(obs, "istreated", "patient1")):
                return 150
            else:
                return 500

        score = 0

        # Phase 1: Basic setup and chest compressions
        correct_stacking = [False] * (len(self.correct_order) - 1)
        for literal in obs.literals:
            if literal.predicate.name == "atop":
                index = self._find_stacking_index(
                    literal.variables[1].name, literal.variables[0].name
                )
                if index != -1:
                    correct_stacking[index] = True

        for i, stacked in enumerate(correct_stacking):
            if stacked:
                score += 8 * (i + 1)
            else:
                break

        # CPR board placement
        cpr_board_on_station = self._check_item_on_station(obs, "cpr_board", "patient_bed_station")
        cpr_board_held = self._check_item_held(obs, "cpr_board")
        if cpr_board_on_station:
            score += 8
        elif cpr_board_held:
            score += 4

        # Chest compressions shaping
        chest_compression_progress = self._check_action_progress(state, "compresschest")
        score += self._calculate_action_reward(chest_compression_progress, 3)
        if self._check_predicate(obs, "ischestcompressed", "patient1"):
            score += 12

        # Rescue breaths shaping
        pump_on_patient = self._check_item_on_item(obs, "pump", "patient")
        pump_held = self._check_item_held(obs, "pump")
        if pump_on_patient:
            score += 10
        elif pump_held:
            score += 5

        rescue_breath_progress = self._check_action_progress(state, "giverescuebreaths")
        score += self._calculate_action_reward(rescue_breath_progress, 2)
        if self._check_predicate(obs, "isrescuebreathed", "patient1"):
            score += 15

        # Shock shaping
        aed_on_pump = self._check_item_on_item(obs, "aed", "pump")
        aed_held = self._check_item_held(obs, "aed")
        if aed_on_pump:
            score += 35
        elif aed_held:
            score += 18

        if self._check_predicate(obs, "isrescuebreathed", "patient1"):
            if aed_held:
                score += 45
            if aed_on_pump:
                score += 50
                if self._check_player_at_station(obs, "patient_bed_station"):
                    score += 43

        max_shocks = 2
        if hasattr(self, 'config') and isinstance(self.config, dict):
            if 'num_shocks' in self.config and 'patient' in self.config['num_shocks']:
                max_shocks = self.config['num_shocks']['patient']

        shock_progress = self._check_action_progress(state, "giveshock")
        if shock_progress <= max_shocks:
            score += self._calculate_action_reward(shock_progress, max_shocks) * 2
        else:
            excess_shocks = shock_progress - max_shocks
            score -= 15 * excess_shocks

        if self._check_predicate(obs, "isshocked", "patient1"):
            score += 70
            medicine_progress = self._check_action_progress(state, "givemedicine")
            if medicine_progress > 0:
                score += 20

        # Medicine shaping
        is_medicine_phase = self._check_predicate(obs, "isshocked", "patient1")
        if is_medicine_phase:
            if (self._check_item_on_item(obs, "aed", "pump") and
                self._check_item_on_item(obs, "pump", "patient")):
                score += 16

            if self._check_item_held(obs, "syringe"):
                score += 26
                if self._check_player_at_station(obs, "patient_bed_station"):
                    score += 19
                    if self._check_item_on_item(obs, "syringe", "aed"):
                        score += 39
                        medicine_progress = self._check_action_progress(state, "givemedicine")
                        if medicine_progress > 0:
                            score += 32
            else:
                if (self._check_item_on_station(obs, "syringe", "hospital_cart_left") or
                    self._check_item_on_station(obs, "syringe", "hospital_cart_right") or
                    self._check_item_on_station(obs, "syringe", "hospital_cart")):
                    score += 15

            medicine_progress = self._check_action_progress(state, "givemedicine")
            if medicine_progress > 0:
                score += 26 * medicine_progress
                if self._check_predicate(obs, "istreated", "patient1"):
                    score += 65
        else:
            if self._check_item_held(obs, "syringe"):
                score += 15
            medicine_progress = self._check_action_progress(state, "givemedicine")
            score += self._calculate_action_reward(medicine_progress, 2)

        return score

    def _calculate_moveitem_rewards(self, obs, state):
        """Calculate rewards for moveitem tasks"""
        score = 0

        # Check goal completion first
        if pddlgym.inference.check_goal(obs, obs.goal):
            self.goal_reached = True
            return 500

        # Parse the specific stacking goal
        target_stacking = []
        target_placement = {}

        def extract_all_literals(goal_obj):
            literals = []
            if isinstance(goal_obj, Literal):
                literals.append(goal_obj)
            elif isinstance(goal_obj, (LiteralConjunction, LiteralDisjunction)):
                if hasattr(goal_obj, 'literals'):
                    for lit in goal_obj.literals:
                        literals.extend(extract_all_literals(lit))
            return literals

        goal_literals = extract_all_literals(obs.goal)

        for goal in goal_literals:
            if hasattr(goal, 'predicate') and hasattr(goal.predicate, 'name'):
                if goal.predicate.name == "atop":
                    if len(goal.variables) >= 2:
                        target_stacking.append((goal.variables[0].name, goal.variables[1].name))
                elif goal.predicate.name == "on":
                    if len(goal.variables) >= 2:
                        target_placement[goal.variables[0].name] = goal.variables[1].name

        # Track current item locations
        items_held = {}
        items_on_stations = {}
        current_stacking = []

        for literal in obs.literals:
            if literal.predicate.name == "has":
                player = literal.variables[0].name
                item = literal.variables[1].name
                items_held[player] = item
            elif literal.predicate.name == "on":
                item = literal.variables[0].name
                station = literal.variables[1].name
                items_on_stations[item] = station
            elif literal.predicate.name == "atop":
                top = literal.variables[0].name
                bottom = literal.variables[1].name
                current_stacking.append((top, bottom))

        # Track items moved
        currently_held_items = set(items_held.values())
        newly_picked_items = currently_held_items - (self.previous_held_items or set())

        if newly_picked_items:
            self.items_moved.update(newly_picked_items)

        self.previous_held_items = currently_held_items

        # Correct stacking achieved
        for target_top, target_bottom in target_stacking:
            if (target_top, target_bottom) in current_stacking:
                score += 200
                if not self.milestones_reached.get("correct_stacking", False):
                    self.milestones_reached["correct_stacking"] = True
                    self.milestone_timestamps["correct_stacking"] = self.step_counter
            elif (target_bottom, target_top) in current_stacking:
                score -= 50

        # Other moveitem rewards (simplified)
        for item in items_held.values():
            score += 20  # Holding relevant items
        
        return score