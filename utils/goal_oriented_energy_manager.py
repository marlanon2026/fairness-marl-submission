class GoalOrientedEnergyManager:
    """
    Manages energy depletion based on goal-directed actions.
    Only depletes energy for actions that contribute to the next logical subgoal.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Define the treatment progression for patients
        self.treatment_progression = [
            'compresschest',
            'giverescuebreaths', 
            'giveshock',
            'givemedicine'
        ]
        
        # Define what items/stations are needed for each treatment step
        self.treatment_requirements = {
            'compresschest': {
                'required_items': ['cpr_board'],
                'required_stations': ['patient_bed_station'],
                'setup_actions': ['stackunder']  # placing cpr_board under patient
            },
            'giverescuebreaths': {
                'required_items': ['pump'],
                'required_stations': ['patient_bed_station'],
                'setup_actions': ['pick-up', 'stack']  # getting pump and placing on patient
            },
            'giveshock': {
                'required_items': ['aed'],
                'required_stations': ['patient_bed_station'], 
                'setup_actions': ['pick-up', 'stack']  # getting aed and placing on pump
            },
            'givemedicine': {
                'required_items': ['syringe'],
                'required_stations': ['patient_bed_station'],
                'setup_actions': ['pick-up', 'stack']  # getting syringe and placing on aed
            }
        }
        
        # Energy costs for different action types
        self.energy_costs = {
            'treatment_action': config["energy_levels"].get("treatment_cost", 20),
            'goal_directed_setup': config["energy_levels"].get("setup_cost", 8),
            'goal_directed_movement': config["energy_levels"].get("movement_cost", 3),
            'non_goal_action': config["energy_levels"].get("non_goal_penalty", 1)
        }

    def get_patient_treatment_state(self, env_state, patient_name):
        """Determine what treatment stage the patient is currently at"""
        patient_state = {
            'ischestcompressed': False,
            'isrescuebreathed': False, 
            'isshocked': False,
            'istreated': False
        }
        
        # Check current patient state from PDDL literals
        for literal in env_state.literals:
            predicate_name = literal.predicate.name
            if predicate_name in patient_state:
                # Check if this literal applies to our patient
                for var in literal.variables:
                    if var.name == patient_name:
                        patient_state[predicate_name] = True
                        break
        
        return patient_state

    def get_next_required_treatment(self, patient_state):
        """Determine the next treatment step needed for the patient"""
        if not patient_state['ischestcompressed']:
            return 'compresschest'
        elif not patient_state['isrescuebreathed']:
            return 'giverescuebreaths'
        elif not patient_state['isshocked']:
            return 'giveshock'
        elif not patient_state['istreated']:
            return 'givemedicine'
        else:
            return None  # Patient fully treated

    def is_action_goal_directed(self, action, env_state, wrapper_state, current_player):
        """
        Determine if an action contributes to the next logical treatment subgoal.
        """
        if action == "noop":
            return False, 0
            
        action_name = action.predicate.name
        
        # Find patients in the environment
        patients = self._find_patients(env_state)
        
        for patient_name in patients:
            patient_state = self.get_patient_treatment_state(env_state, patient_name)
            next_treatment = self.get_next_required_treatment(patient_state)
            
            if next_treatment is None:
                continue  # Patient fully treated
            
            # Check if this is the actual treatment action
            if action_name == next_treatment:
                return True, self.energy_costs['treatment_action']
            
            # Check if this is a setup action for the next treatment
            goal_directed, cost = self._is_setup_action_goal_directed(
                action, action_name, next_treatment, env_state, current_player
            )
            if goal_directed:
                return True, cost
                
        # If we get here, action doesn't contribute to any patient's next treatment step
        return False, self.energy_costs['non_goal_action']

    def _is_setup_action_goal_directed(self, action, action_name, next_treatment, env_state, current_player):
        """Check if a setup action (move, pick-up, stack, etc.) contributes to the next treatment"""
        treatment_req = self.treatment_requirements.get(next_treatment, {})
        required_items = treatment_req.get('required_items', [])
        required_stations = treatment_req.get('required_stations', [])
        setup_actions = treatment_req.get('setup_actions', [])
        
        if action_name not in setup_actions and action_name != 'move':
            return False, 0
            
        # Check movement toward required stations
        if action_name == 'move':
            return self._is_movement_goal_directed(action, required_stations, required_items, env_state)
            
        # Check item manipulation actions
        elif action_name in ['pick-up', 'stack', 'stackunder', 'place']:
            return self._is_item_action_goal_directed(action, required_items, next_treatment)
            
        return False, 0

    def _is_movement_goal_directed(self, action, required_stations, required_items, env_state):
        """Check if movement is toward a station needed for the next treatment"""
        # Get destination station from move action
        destination_station = None
        for var in action.variables:
            if var.var_type == "station":
                destination_station = var.name
                break
                
        if not destination_station:
            return False, 0
            
        # Check if moving to a required station type
        for station_type in required_stations:
            if self._station_has_type(destination_station, station_type, env_state):
                return True, self.energy_costs['goal_directed_movement']
                
        # Check if moving to get a required item
        for item_type in required_items:
            if self._station_has_required_item(destination_station, item_type, env_state):
                return True, self.energy_costs['goal_directed_movement']
                
        return False, 0

    def _is_item_action_goal_directed(self, action, required_items, next_treatment):
        """Check if item manipulation involves items needed for next treatment"""
        action_items = []
        for var in action.variables:
            if var.var_type == "item":
                action_items.append(var.name)
                
        # Check if any item in the action is required for next treatment
        for item_name in action_items:
            for required_type in required_items:
                if required_type.lower() in item_name.lower():
                    return True, self.energy_costs['goal_directed_setup']
                    
        return False, 0

    def _find_patients(self, env_state):
        """Find all patients in the environment"""
        patients = []
        for literal in env_state.literals:
            if literal.predicate.name == 'ispatient':
                for var in literal.variables:
                    if var.var_type == "item":
                        patients.append(var.name)
        return patients

    def _station_has_type(self, station_name, station_type, env_state):
        """Check if a station has a specific type"""
        predicate_name = f"is{station_type}"
        for literal in env_state.literals:
            if literal.predicate.name == predicate_name:
                for var in literal.variables:
                    if var.name == station_name:
                        return True
        return False

    def _station_has_required_item(self, station_name, item_type, env_state):
        """Check if a station contains an item of the required type"""
        # First find items at the station
        items_at_station = []
        for literal in env_state.literals:
            if literal.predicate.name in ['at', 'on']:
                if len(literal.variables) >= 2:
                    item_var, station_var = literal.variables[0], literal.variables[1]
                    if station_var.name == station_name:
                        items_at_station.append(item_var.name)
        
        # Check if any item at station matches required type
        for item_name in items_at_station:
            if item_type.lower() in item_name.lower():
                return True
        return False