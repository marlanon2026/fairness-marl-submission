# utils/goal_focused_reward_handler.py

from utils.reward_handler import RewardHandler
import pddlgym.inference



class GoalFocusedRewardHandler(RewardHandler):
    """
    A unified reward handler that works for both Hospital and Robotouille environments,
    focusing primarily on goal completion with minimal intermediate rewards.
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.max_possible_reward = 1.0
        self.previous_progress = 0
        self.goal_reached = False
        self.environment_type = None  # Will be detected on first call
        
        
        # Hospital environment-specific settings
        self.hospital_correct_order = ["cpr_board", "patient", "pump", "aed", "syringe"]
        
        # Robotouille environment-specific settings
        self.robotouille_correct_order = ["topbun", "lettuce", "patty", "bottombun"]
    
    def reset(self):
        """Reset the reward handler's internal state."""
        self.previous_progress = 0
        self.goal_reached = False
    
    #--------------------------------
    # Common Helper Methods
    #--------------------------------
    
    def _check_predicate(self, obs, predicate, item):
        """Check if a predicate is true for an item."""
        for literal in obs.literals:
            if (
                predicate == literal.predicate.name
                and item in literal.variables[0].name
            ):
                return True
        return False
    
    def _check_item_held(self, obs, item):
        """Check if an item is being held."""
        for literal in obs.literals:
            if "has" == literal.predicate.name and item in literal.variables[1].name:
                return True
        return False
    
    def _check_action_progress(self, state, action, item):
        """Check the progress of an action on an item."""
        item_status = state.get(item, {})
        return item_status.get(action, 0)
    
    #--------------------------------
    # Hospital Environment Methods
    #--------------------------------
    
    def _hosp_find_stacking_index(self, item1, item2):
        """Find the index of stacking in the hospital correct order."""
        for i in range(len(self.hospital_correct_order) - 1):
            if (self.hospital_correct_order[i] in item1 and 
                self.hospital_correct_order[i + 1] in item2):
                return i
        return -1

    def _hosp_check_item_on_station(self, obs, item, station):
        """Check if an item is on a station in hospital environment."""
        for literal in obs.literals:
            if (
                "on" == literal.predicate.name
                and item in literal.variables[0].name
                and station in literal.variables[1].name
            ):
                return True
        return False

    def _hosp_check_item_on_item(self, obs, top_item, bottom_item):
        """Check if an item is on top of another item in hospital environment."""
        for literal in obs.literals:
            if (
                "atop" == literal.predicate.name
                and top_item in literal.variables[0].name
                and bottom_item in literal.variables[1].name
            ):
                return True
        return False
    
    def _hosp_calculate_progress(self, obs, state):
        """Calculate progress toward the goal in hospital environment."""
        # Check if patient is treated (final goal)
        # if self._check_predicate(obs, "istreated", "patient1"):
        #     self.goal_reached = True
        #     return 1.0
            # Check if the actual PDDL goal is satisfied
        if pddlgym.inference.check_goal(obs, obs.goal):
            self.goal_reached = True
            return 300
        
        # Track progress on important subgoals
        progress_metrics = {
            "stacking": 0,
            "cpr_board": 0,
            "compressions": 0,
            "rescue_breaths": 0,
            "shock": 0,
            "medicine": 0
        }
        
        # Check stacking progress
        stacking_count = 0
        for literal in obs.literals:
            if literal.predicate.name == "atop":
                index = self._hosp_find_stacking_index(
                    literal.variables[1].name, literal.variables[0].name
                )
                if index != -1:
                    stacking_count += 1
        
        progress_metrics["stacking"] = stacking_count / (len(self.hospital_correct_order) - 1)
        
        # CPR board placement
        if self._hosp_check_item_on_station(obs, "cpr_board", "patient_bed_station"):
            progress_metrics["cpr_board"] = 1.0
        elif self._check_item_held(obs, "cpr_board"):
            progress_metrics["cpr_board"] = 0.5
        
        # Chest compressions
        if self._check_predicate(obs, "ischestcompressed", "patient1"):
            progress_metrics["compressions"] = 1.0
        else:
            compression_progress = self._check_action_progress(state, "compresschest", "patient1")
            progress_metrics["compressions"] = min(compression_progress / 3, 0.9)
        
        # Rescue breaths
        if self._check_predicate(obs, "isrescuebreathed", "patient1"):
            progress_metrics["rescue_breaths"] = 1.0
        else:
            breath_progress = self._check_action_progress(state, "giverescuebreaths", "patient1")
            progress_metrics["rescue_breaths"] = min(breath_progress / 2, 0.9)
        
        # Shock
        if self._check_predicate(obs, "isshocked", "patient1"):
            progress_metrics["shock"] = 1.0
        else:
            shock_progress = self._check_action_progress(state, "giveshock", "patient1")
            progress_metrics["shock"] = min(shock_progress, 0.9)
        
        # Medicine
        medicine_progress = self._check_action_progress(state, "givemedicine", "patient1")
        progress_metrics["medicine"] = min(medicine_progress, 0.9)
        
        # Calculate weighted average of progress metrics
        weights = {
            "stacking": 0.15,
            "cpr_board": 0.05,
            "compressions": 0.2,
            "rescue_breaths": 0.2,
            "shock": 0.2,
            "medicine": 0.2
        }
        
        total_progress = sum(metric * weights[key] for key, metric in progress_metrics.items())
        return min(1.0, total_progress)
    
    #--------------------------------
    # Robotouille Environment Methods
    #--------------------------------
    
    def _robotouille_find_stacking_index(self, goal):
        """Find the index of stacking in the Robotouille correct order."""
        for i in range(len(self.robotouille_correct_order) - 1):
            if (
                self.robotouille_correct_order[i] in goal.variables[0].name
                and self.robotouille_correct_order[i + 1] in goal.variables[1].name
            ):
                return i
        return -1
    
    def _robotouille_check_patty_stove(self, obs):
        """Check if the patty is on the stove."""
        for literal in obs.literals:
            if (
                "on" == literal.predicate.name
                and "patty1" in literal.variables[0].name
                and "stove" in literal.variables[1].name
            ):
                return True
        return False
    
    def _robotouille_check_cooking_start(self, state):
        """Check if cooking has started."""
        for item, status_dict in state.items():
            for status, state_value in status_dict.items():
                if status == "cook" and state_value.get("cooking", False):
                    return True
        return False
    
    def _robotouille_check_lettuce_board(self, obs):
        """Check if lettuce is on the board."""
        for literal in obs.literals:
            if (
                "on" == literal.predicate.name
                and "lettuce1" in literal.variables[0].name
                and "board" in literal.variables[1].name
            ):
                return True
        return False
    
    def _robotouille_calculate_progress(self, obs, state):
        """Calculate progress toward the goal in Robotouille environment."""
        # Define the correct burger stacking order
        correct_order = self.robotouille_correct_order
        
        # Track goal components
        stacking_progress = 0
        cooked = False
        cut = False
        
        # Check if goals are met
        total_goal_conditions = 0
        satisfied_goal_conditions = 0
        
        for clause in obs.goal.literals:
            for goal in clause.literals:
                total_goal_conditions += 1
                for literal in obs.literals:
                    if goal == literal:
                        satisfied_goal_conditions += 1
                        
                        # Track specific goal components
                        if goal.predicate.name == "atop":
                            index = self._robotouille_find_stacking_index(goal)
                            if index >= 0:
                                stacking_progress += 1
                        elif goal.predicate.name == "iscooked":
                            cooked = True
                        elif goal.predicate.name == "iscut":
                            cut = True
        
        # Full goal is achieved if all conditions are met
        if total_goal_conditions > 0 and satisfied_goal_conditions == total_goal_conditions:
            self.goal_reached = True
            return 1.0
            
        # Progress metrics
        progress_metrics = {
            "stacking": 0,
            "cooking": 0,
            "cutting": 0
        }
        
        # Calculate stacking progress
        if stacking_progress > 0:
            progress_metrics["stacking"] = stacking_progress / 3  # 3 stacking relationships
        
        # Calculate cooking progress
        if cooked:
            progress_metrics["cooking"] = 1.0
        else:
            # Check cook progress in state
            for item, status_dict in state.items():
                if "patty" in item and "cook" in status_dict:
                    cook_time = status_dict["cook"].get("cook_time", -1)
                    if cook_time >= 0:
                        # Assuming max cook time is 3
                        progress_metrics["cooking"] = min(cook_time / 3, 0.9)
                        break
            
            # Check for patty on stove or cooking started
            if progress_metrics["cooking"] == 0:
                if self._robotouille_check_patty_stove(obs):
                    progress_metrics["cooking"] = 0.3
                elif self._robotouille_check_cooking_start(state):
                    progress_metrics["cooking"] = 0.2
                elif self._check_item_held(obs, "patty1"):
                    progress_metrics["cooking"] = 0.1
        
        # Calculate cutting progress
        if cut:
            progress_metrics["cutting"] = 1.0
        else:
            # Check cut progress in state
            for item, status_dict in state.items():
                if "lettuce" in item and "cut" in status_dict:
                    cut_value = status_dict["cut"]
                    # Assuming max cuts is 3
                    progress_metrics["cutting"] = min(cut_value / 3, 0.9)
                    break
            
            # Check for lettuce on board
            if progress_metrics["cutting"] == 0:
                if self._robotouille_check_lettuce_board(obs):
                    progress_metrics["cutting"] = 0.3
                elif self._check_item_held(obs, "lettuce1"):
                    progress_metrics["cutting"] = 0.1
        
        # Calculate weighted average of progress metrics
        weights = {
            "stacking": 0.4,
            "cooking": 0.3,
            "cutting": 0.3
        }
        
        total_progress = sum(metric * weights[key] for key, metric in progress_metrics.items())
        return min(1.0, total_progress)
    
    #--------------------------------
    # Main Reward Method
    #--------------------------------
    
    def heuristic_reward(self, obs, state):
        """
        Calculate reward based on goal progress with stronger shaping rewards.
        """
        self.obs = obs
        self.state = state
        
        # Detect environment type if not already set
        if self.environment_type is None:
            # Code to detect environment type - leave this unchanged
            is_hospital = False
            is_robotouille = False
            
            for literal in obs.literals:
                if "patient" in str(literal) or "cpr_board" in str(literal):
                    is_hospital = True
                    break
                if "patty" in str(literal) or "lettuce" in str(literal) or "bun" in str(literal):
                    is_robotouille = True
                    break
            
            if is_hospital:
                self.environment_type = "hospital"
            elif is_robotouille:
                self.environment_type = "robotouille"
            else:
                # Default to robotouille if unable to determine
                self.environment_type = "robotouille"
        
        # Calculate progress based on environment type
        if self.environment_type == "hospital":
            current_progress = self._hosp_calculate_progress(obs, state)
        else:  # robotouille
            current_progress = self._robotouille_calculate_progress(obs, state)
        
        # Full reward for goal completion
        if self.goal_reached:
            return 1.0
        
        # Calculate progress delta
        progress_delta = current_progress - self.previous_progress
        self.previous_progress = current_progress
        
        # Enhanced reward shaping (all kept between 0 and 1)
        if progress_delta > 0:
            # Significant reward for making progress
            progress_reward = 0.2 + (progress_delta * 0.6)
            return min(progress_reward, 0.8)
        elif progress_delta == 0:
            # Reward for maintaining progress
            return 0.01 + (current_progress * 0.2)
        else:
            # Small reward even when losing progress
            return -0.001