import gym
import pddlgym
from utils.hosp_reward_handler import HospRewardHandler
from utils.robotouille_reward_handler import RobotouilleRewardHandler
from utils.goal_focused_reward_handler import GoalFocusedRewardHandler
import utils.robotouille_utils as robotouille_utils
import utils.pddlgym_utils as pddlgym_utils
from environments.env_generator.object_enums import Item


class RobotouilleWrapper(gym.Wrapper):
    """
    This wrapper wraps around the Robotouille environment from PDDLGym.

    This wrapper is necessary because while the PDDL language is powerful, it can be
    cumbersome to implement data-driven state such as cutting X times or cooking something
    for Y timesteps. This does not mean it is impossible but rather than littering the
    observation space with a bunch of predicates to represent time and number of cuts, we
    offload this to the wrapper's metadata.
    """

    def __init__(self, env, config, renderer):
        """
        Initialize the Robotouille wrapper.

        Args:
            env (PDDLGym Environment): The environment to wrap.
            config (dict): A configuration JSON with custom values
        """
        super(RobotouilleWrapper, self).__init__(env)
        # The PDDLGym environment.
        self.env = env
        # The previous step of the environment.
        # This is useful for the interactive mode and for cases where nothing changes (e.g. noop)
        self.prev_step = None
        # The number of timesteps that have passed. For one timestep, all players have taken an action.
        # The number of timesteps that have passed. For one timestep, all players have taken an action.
        self.timesteps = 0
        # The state of the environment (for non-PDDL states like cut and cook)
        self.state = {}
        # The configuration for this environment.
        # This is used to specify things such as cooking times and cutting amounts
        self.config = config
        self.num_players = config["num_players"]
        # The amount of individual steps that have been taken (num_players steps per timestep)
        self.move_counter = 0
        self.taken_actions = []
        self.renderer = renderer
        #self.reward_handler = HospRewardHandler(self.state)
        self.reward_handler = GoalFocusedRewardHandler()
        self.episode_reward = 0
        self.goal_announced = False
        self.goals_reached = 0
        self.max_steps = 50


    def _interactive_starter_prints(self, expanded_truths):
        """
        This function prints the initial state of the environment and the valid actions.

        Args:
            expanded_truths (np.array): Array of 0s and 1s where 1 indicates the literal is true
        """
        print("\n" * 10)
        if self.timesteps % 10 == 0:
            print(f"You have made {self.timesteps} steps.")
        robotouille_utils.print_states(self.prev_step[0])
        print("\n")
        robotouille_utils.print_actions(self.env, self.prev_step[0], self.renderer)
        print(f"True Predicates: {expanded_truths.sum()}")

    def _handle_action(self, action):
        """
        This function handles the action taken by the environment.

        Args: action (str): The action to take.
        """
        if action == "noop":
            return self.prev_step
        action_name = action.predicate.name
        items = [
            var.name for var in action.variables if var.var_type == "item"
        ]  # newly added
        current_player = self._current_selected_player(self.prev_step[0])
        try:
            value = self.config["player_info"][current_player][action_name]
        except KeyError:
            value = 1
        if action_name == "cut":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"cut": value}
            elif item_status.get("cut") is None:
                item_status["cut"] = value
            else:
                item_status["cut"] += value

                if item_status["cut"] == 3:
                    item_status["picked-up"] = False
            return self.prev_step
        # add a similar logic for action compresschest
        elif action_name == "compresschest":
            player = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "player",
                    action.variables,
                )
            )
            player_status = self.state.get(player.name)
            # TODO: figure out self.state structure for players
            energy_config = self.config["energy_levels"]
            if player_status is None:
                self.state[player.name] = {"energy": energy_config["max"]}
                player_status = self.state[player.name]
            elif player_status.get("energy") is None:
                player_status["energy"] = energy_config["max"]

            # check to see if player has enough energy to compress patient chest
            if energy_config["compresschest_cost"] <= player_status["energy"]:
                item = next(
                    filter(
                        lambda typed_entity: typed_entity.var_type == "item",
                        action.variables,
                    )
                )
                item_status = self.state.get(item.name)
                if item_status is None:
                    self.state[item.name] = {"compresschest": value}
                elif item_status.get("compresschest") is None:
                    item_status["compresschest"] = value
                else:
                    item_status["compresschest"] += value
                    if item_status["compresschest"] == 3:
                        item_status["picked-up"] = False

                player_status["energy"] -= (
                    # need to subtract recharge rate to offset recharging elsewhere
                    energy_config["compresschest_cost"]
                    + energy_config["recharge_rate"]
                )
            return self.prev_step
        # add a similar logic for action giverescuebreaths
        elif action_name == "giverescuebreaths":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"giverescuebreaths": value}
            elif item_status.get("giverescuebreaths") is None:
                item_status["giverescuebreaths"] = value
            else:
                item_status["giverescuebreaths"] += value

            return self.prev_step

        # add a similar logic for action giveshock
        elif action_name == "giveshock":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"giveshock": value}
            elif item_status.get("giveshock") is None:
                item_status["giveshock"] = value
            else:
                item_status["giveshock"] += value

            return self.prev_step

        # add a similar logic for action givemedicine
        elif action_name == "givemedicine":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"givemedicine": value}
            elif item_status.get("givemedicine") is None:
                item_status["givemedicine"] = value
            else:
                item_status["givemedicine"] += value

            return self.prev_step

        # add a similar logic for action gatherfood
        elif action_name == "cook":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"cook": {"cook_time": -1, "cooking": True}}
            elif item_status.get("cook") is None:
                item_status["cook"] = {"cook_time": -1, "cooking": True}
            else:
                item_status["cook"]["cooking"] = True
            return self.prev_step
        elif action_name == "fry" or action_name == "fry_cut_item":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is None:
                self.state[item.name] = {"fry": {"fry_time": -1, "frying": True}}
            elif item_status.get("fry") is None:
                item_status["fry"] = {"fry_time": -1, "frying": True}
            else:
                item_status["fry"]["frying"] = True
            return self.prev_step
        elif action_name in ["stack", "stackunder"]:
            for item in items:
                item_status = self.state.get(item, {})
                item_status["stacked"] = True
                self.state[item] = item_status

        elif action_name == "pick-up":
            item = next(
                filter(
                    lambda typed_entity: typed_entity.var_type == "item",
                    action.variables,
                )
            )
            item_status = self.state.get(item.name)
            if item_status is not None and item_status.get("cook") is not None:
                item_status["cook"]["cooking"] = False
            if item_status is not None and item_status.get("fry") is not None:
                item_status["fry"]["frying"] = False

            # Spaghetti code to handle the fact that the patty is not stacked after cooking
            item_status = self.state.get("patty1")
            cooked = self._check_cooked(self.prev_step[0])

            # initialize state["patty1"]["picked-up"] if it doesn't exist
            if item_status is None:
                self.state[item.name] = {"picked-up": False}
                item_status = self.state.get(item.name)
            elif item_status.get("picked-up") is None:
                item_status["picked-up"] = False

            if cooked:
                item_status["picked-up"] = True
        # TODO: Probably stop cooking if something is stacked on top of meat
        return self.env.step(action)
    
    def is_start_of_timestep(self):
        """This function returns true if it is the first players' timestep"""
        return self.move_counter % self.num_players == 0

    def _is_end_of_timestep(self):
        """This function returns true if it is the last players' timestep"""
        return self.move_counter % self.num_players == self.num_players - 1

    def _state_update(self):
        """
        This function updates the custom non-PDDL state of the environment.

        This function is called after every step in the environment. It can either update
        the custom state (e.g. incrementing the cook time of a cooking item) or directly
        modify the PDDL state (e.g. adding the iscut predicate to an item that has been
        fully cut).

        Returns:
            new_env_state (PDDLGym State): The new state of the environment.
        """
        state_updates = []
        state_removals = []
        for item, status_dict in self.state.items():
            for status, state in status_dict.items():
                if status == "cut":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    # TODO: Might need to trim item name
                    num_cuts = self.config["num_cuts"]
                    max_num_cuts = num_cuts.get(item_name, num_cuts["default"])
                    if state >= max_num_cuts:
                        literal = pddlgym_utils.str_to_literal(f"iscut({item}:item)")
                        state_updates.append(literal)
                elif status == "cook":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    cook_time = self.config["cook_time"]
                    max_cook_time = cook_time.get(item_name, cook_time["default"])
                    if state["cooking"]:
                        state["cook_time"] += 1
                        if state["cook_time"] == max_cook_time:
                            status_dict["picked-up"] = False
                    if state["cook_time"] >= max_cook_time:
                        literal = pddlgym_utils.str_to_literal(f"iscooked({item}:item)")
                        state_updates.append(literal)
                elif status == "fry":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    fry_time = self.config["fry_time"]
                    max_fry_time = fry_time.get(item_name, fry_time["default"])
                    if state["frying"]:
                        state["fry_time"] += 1
                    if state["fry_time"] >= max_fry_time:
                        literal = pddlgym_utils.str_to_literal(f"isfried({item}:item)")
                        state_updates.append(literal)
                # add similar logic for compresschest and giverescuebreaths
                elif status == "compresschest":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_compressions = self.config["num_compressions"]
                    max_num_compressions = num_compressions.get(
                        item_name, num_compressions["default"]
                    )
                    if state >= max_num_compressions:
                        literal = pddlgym_utils.str_to_literal(
                            f"ischestcompressed({item}:item)"
                        )
                        # print literal
                        state_updates.append(literal)
                elif status == "giverescuebreaths":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_breaths = self.config["num_breaths"]
                    max_num_breaths = num_breaths.get(item_name, num_breaths["default"])
                    if state >= max_num_breaths:
                        literal = pddlgym_utils.str_to_literal(
                            f"isrescuebreathed({item}:item)"
                        )
                        state_updates.append(literal)
                    # if item_name == "patient":
                    #     literal = pddlgym_utils.str_to_literal(f"ischestcompressable({item}:item)")
                    #     state_updates.append(literal)
                # write similar logic for giveshock
                elif status == "giveshock":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_shocks = self.config["num_shocks"]
                    max_num_shocks = num_shocks.get(item_name, num_shocks["default"])
                    if state >= max_num_shocks:
                        literal = pddlgym_utils.str_to_literal(
                            f"isshocked({item}:item)"
                        )
                        state_updates.append(literal)
                elif status == "givemedicine":
                    item_name, _ = robotouille_utils.trim_item_ID(item)
                    num_medicine_doses = self.config["num_medicine_doses"]
                    max_num_medicine_doses = num_medicine_doses.get(
                        item_name, num_medicine_doses["default"]
                    )
                    if state >= max_num_medicine_doses:
                        literal = pddlgym_utils.str_to_literal(
                            f"istreated({item}:item)"
                        )
                        state_updates.append(literal)

                # for now, players are also in the state dict, so the item here is actually a player
                elif status == "energy":
                    if self._is_end_of_timestep():
                        # recharge players' energy at end of timestep
                        # print(f"energy level for player {item} before recharge: {state}")
                        status_dict["energy"] = min(
                            state + self.config["energy_levels"]["recharge_rate"],
                            self.config["energy_levels"]["max"],
                        )
                        # update pddl literal to be tired or not tired for player
                        literal = pddlgym_utils.str_to_literal(
                            f"istired({item}:player)"
                        )
                        if (
                            status_dict["energy"]
                            < self.config["energy_levels"]["compresschest_cost"]
                        ):
                            # print(f"Making player {item} with energy {status_dict['energy']} tired")
                            state_updates.append(literal)
                        else:
                            # remove istired state from pddl env
                            # print(f"Making player {item} with energy {status_dict['energy']} no longer tired")
                            state_removals.append(literal)

        env_state = self.env.get_state()
        new_literals = env_state.literals.difference(set(state_removals))
        new_literals = new_literals.union(state_updates)
        new_env_state = pddlgym.structs.State(
            new_literals, env_state.objects, env_state.goal
        )
        self.env.set_state(new_env_state)
        goal_reached = pddlgym.inference.check_goal(new_env_state, env_state.goal)
        # print(f"Goal check result: {goal_reached}")
        return new_env_state, goal_reached

    def _count_players(self, obs):
        """
        This function counts the number of players in the environment.

        Args:
            obs (PDDLGym State): The current state of the environment.

        Returns:
            num_players (int): The number of players in the environment.
        """
        num_players = 0
        for literal in obs.literals:
            if "isrobot" in literal.predicate.name:
                num_players += 1
        return num_players

    def _current_selected_player(self, obs):
        """
        This function returns the current selected player in the environment.

        Returns:
            selected_player (str): The name of the selected player.
        """
        for literal in obs.literals:
            if "selected" == literal.predicate.name:
                return literal.variables[0].name

    def _change_selected_player(self, obs):
        """
        This function changes the player in the environment.

        Returns:
            new_env_state (PDDLGym State): The new state of the environment.
        """

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

    def get_latest_info(self):
        """
        Get the latest info dictionary from the environment.

        Returns:
            dict: The latest info dictionary.
        """
        return self.prev_step[3] if self.prev_step else None

    def test_step(self, action):
        obs, reward, done, info = self._handle_action(action)
        _, reward, done, info = self.prev_step
        self.prev_step = (obs, reward, done, info)
        return obs, reward, done, info

    def step(self, action=None, interactive=False):
        """
        This function steps the environment forward.

        Most of the output of this function comes from PDDLGym. The observation is a frozenset of
        PDDLGym literals (predicates), objects, and the goal. The reward is 1 if the goal is met and
        0 otherwise. The done flag is True if the goal is met and False otherwise.

        The info metadata is where the wrapper adds the interesting things. The info metadata consists of
        the following:
            - timesteps (int): The number of timesteps that have passed. Currently every action takes
                1 timestep.
            - expanded_truths (np.array): Array of 0s and 1s where 1 indicates the literal is true. PDDLGym
                only provides us with the predicates that are true, but we also need to know which predicates
                are false. This array includes the true and false predicates as a 1D array of 0s and 1s.
            - expanded_states (np.array): Array of literals corresponding to the expanded truths. This is a 1D
                array of the same shape as the expanded truths array. This array's indices map a literal to its
                corresponding truth value in the expanded truths array.
            - toggle_array (np.array): Array of 0s and 1s where 1 indicates the literal changed from time step t
                to t+1. This array is similar to the expanded truths array and it is useful for quickly determining
                how many predicates changed.
            - state (dict): The custom non-PDDL state of the environment. See the state_update function for more
                information.

        Args:
            action (str): The action to take. If None, then it is assumed that interactive is True.
            interactive (bool): Whether or not to use interactive mode.

        Returns:
            obs (PDDLGym State): The new state of the environment.
            reward (float): The reward for the action.
            done (bool): Whether or not the episode is done.
            info (dict): A dictionary of metadata about the step.
        """
        # print("action: ", action)
        expanded_truths = self.prev_step[3]["expanded_truths"]
        expanded_states = self.prev_step[3]["expanded_states"]

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
            self.prev_step[0], self.state
        )
        obs, reward, done, info = self._handle_action(action)
        obs, reward, _, info = self._change_selected_player(obs)
        obs, done = self._state_update()
        # print(f"After state update - done: {done}")
        #breakpoint()
        # print(
        #     f"ENERGY LEVELS: {[(player, pdict['energy']) for player, pdict in self.state.items() if 'energy' in pdict]}"
        # )
        toggle_array = pddlgym_utils.create_toggle_array(
            expanded_truths, expanded_states, obs.literals
        )
        if interactive:
            print(f"Predicates Changed: {toggle_array.sum()}")

        expanded_truths, expanded_states = pddlgym_utils.expand_state(
            obs.literals, obs.objects
        )

        if self._current_selected_player(obs) == "robot1":
            self.timesteps += 1

        self.move_counter += 1
        # print(f"NEXT MOVE NUMBER: {self.move_counter}")

        info = {
            "timesteps": self.timesteps,
            "expanded_truths": expanded_truths,
            "expanded_states": expanded_states,
            "toggle_array": toggle_array,
            "state": self.state,
        }
        #curr_heuristic = self.reward_handler.heuristic_reward(obs, self.state)
        reward = self.reward_handler.heuristic_reward(obs, self.state) #curr_heuristic - prev_heuristic
        #reward *= 0.999**self.timesteps
        # reward /= self.timesteps + 1
            
        # # Apply smaller penalty for timeout (scaled to match new reward scale)
        # if self.timesteps >= self.max_steps:
        #     done = True
        #     # if not self.goal_announced:
        #     #     reward -= 0.1 

        self.prev_step = (obs, reward, done, info)
        self.episode_reward += reward
        if done and not self.goal_announced:
            print(f"Goal Reached in RobotouilleWrapper at timestep {self.timesteps}!")
            self.goal_announced = True
            # print("Goal Announced: ", self.goal_announced)
            self.goals_reached += 1
            # print("episode_reward", self.episode_reward)
        # print("reward: ", reward)
        return obs, reward, done, info

    def save_episode(self, filename):
        with open(filename, "w") as f:
            for action in self.taken_actions:
                f.write(action + "\n")

    def get_episode_actions(self):
        return self.taken_actions
    

    def get_episode_goal_reached(self):
        """Returns 1 if a goal was reached in the current episode, 0 otherwise."""
        result = 1 if self.goal_announced else 0
        # print(f"get_episode_goal_reached called, returning {result} (goal_announced={self.goal_announced})")
        return result
            


    def reset(self):
        """
        This function resets the environment.

        Returns:
            obs (PDDLGym State): The initial state of the environment.
            info (dict): A dictionary of metadata about the step.
        """
        obs, _ = self.env.reset()
        # reset the player grid locations in the renderer
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
        self.prev_step = (obs, 0, False, info)
        self.timesteps = 0
        self.move_counter = 0
        self.state = {}
        self.taken_actions = []
        self.num_players = self._count_players(obs)
        self.episode_reward = 0
        self.goal_announced = False
        # Reset the reward handler
        if hasattr(self.reward_handler, 'reset'):
            self.reward_handler.reset()
        return obs, info
