from utils.reward_handler import RewardHandler



class RobotouilleRewardHandler(RewardHandler):
    def __init__(self):
        super().__init__()

    def _find_stacking_index(self, goal, correct_order):
        for i in range(len(correct_order) - 1):
            if (
                correct_order[i] in goal.variables[0]
                and correct_order[i + 1] in goal.variables[1]
            ):
                return i

    def _check_patty_stove(self, obs):
        for literal in obs.literals:
            if (
                "on" == literal.predicate.name
                and "patty1" in literal.variables[0].name
                and "stove" in literal.variables[1].name
            ):
                return True
        return False

    def _check_cooking_start(self):
        for item, status_dict in self.state.items():
            for status, state in status_dict.items():
                if status == "cook" and state["cooking"]:
                    return True
        return False

    def _check_patty_held(self, obs):
        for literal in obs.literals:
            if (
                "has" == literal.predicate.name
                and "patty1" in literal.variables[1].name
            ):
                return True
        return False

    def _check_lettuce_board(self, obs):
        for literal in obs.literals:
            if (
                "on" == literal.predicate.name
                and "lettuce1" in literal.variables[0].name
                and "board" in literal.variables[1].name
            ):
                return True
        return False

    def _check_lettuce_held(self, obs):
        for literal in obs.literals:
            if (
                "has" == literal.predicate.name
                and "lettuce1" in literal.variables[1].name
            ):
                return True
        return False

    def _not_stacking(self, obs):
        if not self.state["patty1"]["picked-up"]:
            return False

        table = None
        item_below = None

        for literal in obs.literals:
            if "at" == literal.predicate.name and "patty1" in literal.variables[0].name:
                table = literal.variables[1].name
            if (
                "atop" == literal.predicate.name
                and "patty1" in literal.variables[0].name
            ):
                item_below = literal.variables[1].name

        if table is None:
            return False

        if item_below is None or "bottombun" not in item_below:
            return True

        return False

    def _at_bottombun(self, obs):
        bottombun_station = None
        for literal in obs.literals:
            if (
                "at" == literal.predicate.name
                and "bottombun" in literal.variables[0].name
            ):
                bottombun_station = literal.variables[1].name
        if bottombun_station is None:
            return False
        for literal in obs.literals:
            if (
                "loc" == literal.predicate.name
                and bottombun_station in literal.variables[1].name
            ):
                return True
        return False

    def _at_stove(self, obs):
        for literal in obs.literals:
            if "loc" == literal.predicate.name and "stove" in literal.variables[1].name:
                return True

        return False

    def heuristic_reward(self, obs, state):
        """
        This function is a heuristic function that is used to generate a plan.

        Args:
            obs (PDDLGym State): The current state of the environment.

        Returns: The measure of "goodness" of the state.
        """

        # Example goal: [AND[iscut(lettuce1:item), atop(topbun1:item,lettuce1:item), iscooked(patty1:item), atop(lettuce1:item,patty1:item), atop(patty1:item,bottombun1:item)]]
        self.obs = obs
        self.state = state
        correct_order = ["topbun", "lettuce", "patty", "bottombun"]
        correct_stacking = [False] * 3
        cooked = False
        cut = False

        # See if goals (stacking, cooking, cutting) are met
        score = 0
        for clause in obs.goal.literals:
            for goal in clause.literals:
                for literal in obs.literals:
                    if goal == literal:
                        if goal.predicate == "atop":
                            index = self._find_stacking_index(goal, correct_order)
                            correct_stacking[index] = True
                        if goal.predicate == "iscooked":
                            cooked = True
                        if goal.predicate == "iscut":
                            cut = True

        # Check if the stacking is correct (Correct order + cooked burger + cut lettuce)
        for i in range(len(correct_stacking)):
            if not correct_stacking[2 - i]:
                break
            if i == 0 and not cooked:
                break
            elif i == 1 and (not cut or not cooked):
                break
            elif i == 2 and not cut:
                break
            score += 30

        # Give reward for cooked. If not, give reward for uncooked patty on stove, cooking started, and uncooked patty held
        if cooked:
            score += 35
            score += 10 if self._check_patty_held(obs) else 0
            if self._check_patty_held(obs) and self._at_bottombun(obs):
                score += 10
            elif self._check_patty_held(obs) and not (
                self._at_bottombun(obs) or self._at_stove(obs)
            ):
                score -= 10

            # score -= 10 if self._not_stacking(obs) else 0
        else:
            score += 5 if self._check_patty_held(obs) else 0
            score += 15 if self._check_patty_stove(obs) else 0
            score += 10 if self._check_cooking_start() else 0

        # Give reward for cut. If not, give reward for uncut lettuce on board and uncut lettuce held
        if cut:
            score += 45
        else:
            score += 15 if self._check_lettuce_board(obs) else 0
            score += 5 if self._check_lettuce_held(obs) else 0

        # Give reward for each cut in the lettuce
        item_status = self.state.get("lettuce1")
        if item_status is not None and item_status.get("cut") is not None:
            score += 10 * item_status["cut"] if item_status["cut"] < 3 else 0

        return score
