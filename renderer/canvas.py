import os
import pygame
import numpy as np
import utils.robotouille_utils as robotouille_utils
from copy import deepcopy


class RobotouilleCanvas:
    """
    This class is responsible for drawing the game state on a pygame surface. Some of
    the rendered information isn't necessarily provided by the game state (e.g. the
    location of the stations) so it is necessary to provide a layout upon initialization.
    """

    # The directory containing the assets
    ASSETS_DIRECTORY = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "assets"
    )

    # The offset to draw food and stations relative to the center of the grid square
    STATION_FOOD_OFFSET = 0.25
    # This offset helps make it look like the player is standing more on the center of the tile
    PLAYER_OFFSET = 0.2

    def __init__(self, layout, players, window_size=np.array([512, 512]), show_selected_player=True):
        """
        Initializes the canvas.

        Args:
            layout (List[List[Optional[str]]]): 2D array of station names (or None)
            window_size (np.array): (width, height) of the window
            show_selected_player (bool): Whether to show if a certain player is selected
        """
        # print("Initializing RobotouilleCanvas...")
        # The layout of the game
        self.layout = layout
        # print(f"layout: {self.layout}")
        # The player's position and direction
        players_pos = [
            (players[i]["x"], len(layout) - players[i]["y"] - 1)
            for i in range(len(players))
        ]
        # print(f"players_pos: {players_pos}")

        self.players_pose = [ # NOTE: This maintains the canvas state but is also relied on for the RL state
            {
                "name": players[i]["name"],
                "position": players_pos[i],
                "direction": tuple(players[i]["direction"]),
            }
            for i in range(len(players_pos))
        ]
        # print(f"players_pose: {self.players_pose}")

        self._init_players_pose = deepcopy(self.players_pose) # need to maintain this for when environment is reset

        grid_dimensions = np.array([len(layout[0]), len(layout)])
        # print(f"grid_dimensions: {grid_dimensions}")
        # The scaling factor for a grid square
        self.pix_square_size = window_size / grid_dimensions
        # print(f"pix_square_size: {self.pix_square_size}")
        # A dictionary which maps image names to loaded images
        self.asset_directory = {}
        # print(f"asset_directory: {self.asset_directory}")
        # print(f"Station positions: {self._get_station_locations(layout)}")
        self.show_selected_player = show_selected_player


    def __deepcopy__(self, memo):
        """
        This function is called by the deepcopy function in the copy module.

        This function carries over references to objects that are not deepcopyable (PyGame surfaces)
        """
        #import pdb; pdb.set_trace(), should stop to show that this function is being indexed
        new_canvas = RobotouilleCanvas(self.layout, [], np.array([1,1]))
        new_canvas.players_pose = deepcopy(self.players_pose, memo)
        new_canvas.pix_square_size = self.pix_square_size # Constant
        new_canvas.asset_directory = self.asset_directory # References to PyGame surfaces
        memo[id(self)] = new_canvas
        return new_canvas
    
    def reset_player_positions(self):
        """Resets the player positions to their initial positions, used when resetting the environment"""
        self.players_pose = deepcopy(self._init_players_pose)

    def _get_station_position(self, station_name):
        """
        Gets the position of a station.

        Args:
            station_name (str): Name of the station

        Returns:
            position (np.array): (x, y) position of the station
        """
        # print(f"Getting position for station: {station_name}")
        for i, row in enumerate(self.layout):
            for j, col in enumerate(row):
                if col == station_name:
                    position = np.array([j, i], dtype=float)
                    # print(f"Found position: {position}")
                    return position

    def _draw_image(self, surface, image_name, position, scale, flip=False):
        """
        Draws an image on the canvas.

        Args:
            surface (pygame.Surface): Surface to draw on
            image_name (str): Name of the image
            position (np.array): (x, y) position of the image
            scale (np.array): (width, height) to scale the image by
        """
        # print(
        #     f"Drawing image: {image_name} at position: {position} with scale: {scale}"
        # )
        if image_name not in self.asset_directory:
            self.asset_directory[image_name] = pygame.image.load(
                os.path.join(RobotouilleCanvas.ASSETS_DIRECTORY, image_name)
            )
        image = self.asset_directory[image_name]
        image = pygame.transform.smoothscale(image, scale)
        if flip:
            image = pygame.transform.flip(
                image, True, False
            )  # flip if player facing left with aed/syringe
        surface.blit(image, position)

    def _draw_food_image(self, surface, food_name, obs, position):
        """
        Helper to draw a food image on the canvas.

        Args:
            surface (pygame.Surface): Surface to draw on
            food_name (str): Name of the food
            obs (List[Literal]): Game state predicates
            position (np.array): (x, y) position of the food (with pix_square_size factor accounted for)
        """
        # print(f"Drawing food image: {food_name} at position: {position}")
        food_image_name = food_name

        # # Check if the item is a patient and on top of a CPR board
        # is_patient_on_cpr_board = False
        # for literal in obs:
        #     if literal.predicate == "atop" and literal.variables[0].name == food_name and "cpr_board" in literal.variables[1].name:
        #         is_patient_on_cpr_board = True
        #         break

        # Check if cut or cooked or fried
        for literal in obs:
            if literal.predicate == "iscut" and literal.variables[0] == food_image_name:
                food_image_name = "cut" + food_image_name
            if (
                literal.predicate == "iscooked"
                and literal.variables[0] == food_image_name
            ):
                food_image_name = "cooked" + food_image_name
            if literal.predicate == "isfried":
                if literal.variables[0] == food_image_name:
                    food_image_name = "fried" + food_image_name
                elif literal.variables[0] == food_image_name[3:]:
                    food_image_name = "fried" + food_image_name[3:]

            # if literal.predicate == "ischestcompressed":
            #     print(f"Found chestcompressed: {literal.variables[0]}")
            #     if literal.variables[0] == food_image_name:
            #         food_image_name = "chestcompressed" + food_image_name
            #     elif literal.variables[0] == food_image_name[3:]:
            #         food_image_name = "chestcompressed" + food_image_name[3:]
            # if literal.predicate == "isrescuebreathed":
            #     if literal.variables[0] == food_image_name:
            #         print(f"Found rescuebreathed: {literal.variables[0]}")
            #         food_image_name = "rescuebreathed" + food_image_name
            #     elif literal.variables[0] == food_image_name[3:]:
            #         food_image_name = "rescuebreathed" + food_image_name[3:]
            # #add similar logic to the "isshocked" predicate
            # if literal.predicate == "isshocked":
            #     if literal.variables[0] == food_image_name:
            #         food_image_name = "shocked" + food_image_name
            #     elif literal.variables[0] == food_image_name[3:]:
            #         food_image_name = "shocked" + food_image_name[3:]
            # #add similar logic to the "istreated" predicate
            # if literal.predicate == "istreated":
            #     if literal.variables[0] == food_image_name:
            #         food_image_name = "treated" + food_image_name
            #     elif literal.variables[0] == food_image_name[3:]:
            #         food_image_name = "treated" + food_image_name[3:]

        # Remove and store ID
        # NOTE: Remove and store ID used to be done right before calling draw image,
        # there might be unforeseen side effects of this change still
        food_image_name, food_id = robotouille_utils.trim_item_ID(food_image_name)

        patient_states = [
            "istreated",
            "isshocked",
            "isrescuebreathed",
            "ischestcompressed",
        ]
        if "patient" in food_image_name:
            # get patient state
            # TODO: fix shock asset
            for state in patient_states:
                found_state = False
                for literal in obs:
                    if literal.predicate == state:
                        found_state = True
                        # print(f"Found {state[2:]}: {literal.variables[0]}")
                        if literal.variables[0] == food_image_name:
                            food_image_name = f"{state[2:]}" + food_image_name
                        elif food_image_name in literal.variables[0]:
                            food_image_name = f"{state[2:]}" + food_image_name
                if found_state:
                    break

        flip_img = False

        for literal in obs:
            if "aed" in food_image_name:
                # aed visualization code
                if literal.predicate == "has" and "aed" in literal.variables[1]:
                    # case when aed is held by player
                    food_image_name = "aed_on_hcw"
                    # check to see if aed will need to be flipped if player is facing left
                    flip_img = self._get_player_direction(literal) == (-1, 0)
                if (
                    literal.predicate == "at"
                    and "aed" in literal.variables[0]
                    and "patient" in literal.variables[1]
                ):
                    # case when aed is on patient
                    food_image_name = "aed_white"
            if "pump" in food_image_name:
                # pump visualization code
                if literal.predicate == "has" and "pump" in literal.variables[1]:
                    # case when pump is held by player
                    food_image_name = "pump_on_hcw"
                if (
                    literal.predicate == "at"
                    and "pump" in literal.variables[0]
                    and "patient" in literal.variables[1]
                ):
                    # case when pump is on patient
                    food_image_name = "pump_on_patient"
            if "syringe" in food_image_name:
                # syringe visualization code
                if literal.predicate == "has" and "syringe" in literal.variables[1]:
                    # case when syringe is held by player
                    food_image_name = "syringe_on_hcw"
                    # check to see if syringe will need to be flipped if player is facing left
                    flip_img = self._get_player_direction(literal) == (-1, 0)
                if (
                    literal.predicate == "at"
                    and "syringe" in literal.variables[0]
                    and "patient" in literal.variables[1]
                ):
                    # case when syringe is on patient
                    food_image_name = "syringe_on_patient"
            if "cpr_board" in food_image_name:
                # TODO: put offset here and check if it's on cart (or on patient too)
                if literal.predicate == "at" and "cpr_board" in literal.variables[0]:
                    if "cart" in literal.variables[1]:
                        position += self.pix_square_size * np.array(
                            [0, -0.2], dtype=float
                        )
                    if "patient" in literal.variables[1]:
                        # TODO
                        position += self.pix_square_size * np.array(
                            [0, 0.4], dtype=float
                        )

        # print(f"Final food image name: {food_image_name}")

        # Case when we don't want to shrink asset
        if (
            "patient" in food_image_name
            or "syringe" in food_image_name
            or "aed" in food_image_name
        ):
            patient_names = [f"{state[2:]}" + "patient" for state in patient_states]
            if food_image_name in patient_names:
                # temporary fix until assets are fixed - draw patient and then draw component on top for given state
                self._draw_image(
                    surface, f"patient.png", position, self.pix_square_size
                )
                if food_image_name == "treatedpatient":
                    component_name = "treatedpatient"  # we dont really have a special image for this but doesn't matter since it's goal state
                if food_image_name == "shockedpatient":
                    component_name = "lightning_bolts_on_aed"
                if food_image_name == "rescuebreathedpatient":
                    component_name = "rescuebreathe"
                if food_image_name == "chestcompressedpatient":
                    component_name = "apply_pressure"
                self._draw_image(
                    surface, f"{component_name}.png", position, self.pix_square_size
                )
            else:
                self._draw_image(
                    surface,
                    f"{food_image_name}.png",
                    position,
                    self.pix_square_size,
                    flip=flip_img,
                )
        # # special case for cpr_board since asset is too low
        # elif "cpr_board" in food_image_name:
        #     cpr_offset = self.pix_square_size * np.array([0, -0.2], dtype=float)
        #     self._draw_image(
        #         surface,
        #         f"{food_image_name}.png",
        #         position + self.pix_square_size * 0.125 + cpr_offset,
        #         self.pix_square_size * 0.75,
        #     )

        else:
            self._draw_image(
                surface,
                f"{food_image_name}.png",
                position + self.pix_square_size * 0.125,
                self.pix_square_size * 0.75,
                flip=flip_img,
            )

    def _draw_floor(self, surface):
        """
        Draw the floor on the canvas.

        Note the game state is not necessary to draw the floor as this rendering information
        is provided by the layout.

        Args:
            surface (pygame.Surface): Surface to draw on
        """
        # print("Drawing floor...")
        clamped_pix_square_size = np.ceil(
            self.pix_square_size
        )  # Necessary to avoid 1 pixel gaps from decimals
        for row in range(len(self.layout)):
            for col in range(len(self.layout[0])):
                self._draw_image(
                    surface,
                    "floorhospital.png",
                    np.array([col, row]) * clamped_pix_square_size,
                    clamped_pix_square_size,
                )

    def _draw_stations(self, surface):
        """
        Draws the stations on the canvas.

        Note the game state is not necessary to draw the floor as this rendering information
        is provided by the layout.

        Args:
            surface (pygame.Surface): Surface to draw on
        """
        # print("Drawing stations...")
        for i, row in enumerate(self.layout):
            for j, col in enumerate(row):
                if col is not None:
                    while col[-1].isdigit():
                        col = col[:-1]
                    self._draw_image(
                        surface,
                        f"{col}.png",
                        np.array([j, i]) * self.pix_square_size,
                        self.pix_square_size,
                    )

    def _get_player_index(self, literal):
        player_index = int(literal.variables[0].name[5:]) - 1
        # print(f"Player index: {player_index}")
        return player_index

    def _get_player_name_and_index(self, literal):
        return robotouille_utils.trim_item_ID(literal.variables[0].name)

    def _get_player_direction(self, literal):
        player_index = self._get_player_index(literal)
        return self.players_pose[player_index]["direction"]

    def _get_station_locations(self, layout):
        """
        Gets the locations of all stations in the layout.

        Args:
            layout (List[List[Optional[str]]]): 2D array of station names (or None)

        Returns:
            station_locations (List[tuple]): List of (x, y) positions of stations
        """
        station_locations = []
        for i, row in enumerate(layout):
            for j, col in enumerate(row):
                if col is not None:
                    station_locations.append((j, i))
        # print(f"Station locations: {station_locations}")
        return station_locations

    def _get_player_positions(self, player_index):
        """Returns positions of all players except the player at player_index"""
        player_positions = []
        for i in range(len(self.players_pose)):
            if i != player_index:
                player_positions.append(self.players_pose[i]["position"])
        # print(f"Player positions excluding player {player_index}: {player_positions}")
        return player_positions

    def _move_player_to_station(
        self, player_index, player_position, station_position, layout, test=False, verbose=False
    ):
        """
        Moves the player from their current position to a position adjacent to a station using BFS.

        BFS is used to determine the final state. As an additional constraint, the player cannot
        move through a station or out of bounds.

        Args:
            player_position (tuple): (x, y) position of the player
            station_position (tuple): (x, y) position of the station
            layout (List[List[Optional[str]]]): 2D array of station names (or None)

        Returns:
            new_player_position (tuple): (x, y) position of the player after moving
            new_player_direction (tuple): unit vector of the player's direction after moving

        Raises:
            ValueError: If the player cannot reach the station
        """
        # print(
        #     f"Moving player {player_index} from {player_position} to {station_position}"
        # )
        width, height = len(layout[0]), len(layout)
        # print(f"Layout dimensions: width={width}, height={height}")
        obstacle_locations = self._get_station_locations(layout)
        player_positions = self._get_player_positions(player_index)
        curr_prev = (
            player_position,
            player_position,
        )  # current position, previous position
        queue = [curr_prev]
        visited = set()

        while queue:
            curr_prev = queue.pop(0)
            curr_position = curr_prev[0]
            prev_position = curr_prev[1]
            if curr_position == station_position:
                # Reached target station
                new_position = prev_position, (
                    curr_position[0] - prev_position[0],
                    prev_position[1] - curr_position[1],
                )
                # print(f"New position: {new_position}")
                return new_position
            if (
                curr_position[0] < 0
                or curr_position[0] >= width
                or curr_position[1] < 0
                or curr_position[1] >= height
            ):
                # Out of bounds
                continue

            if (curr_position in obstacle_locations) or (
                curr_position in player_positions
            ):
                # Cannot move through station
                continue
            if curr_position in visited:
                # Already visited
                continue
            visited.add(curr_position)
            for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                next_position = (
                    curr_position[0] + direction[0],
                    curr_position[1] + direction[1],
                )
                if next_position != prev_position:
                    queue.append((next_position, curr_position))

        if verbose:
            print(f"Player poses: {self.players_pose}")
            print(f"Tried to move player {player_index} from {player_position} to {station_position}")
            # print all station positions
            print(f"Station positions: {self._get_station_locations(layout)}")

        assert False, "Player could not be moved to station"

    def _check_selected(self, obs, player_index):
        for literal in obs:
            if literal.predicate == "selected":
                selected_player_index = self._get_player_index(literal)
                # print(
                #     f"Player {player_index} selected: {selected_player_index == player_index}"
                # )
                return selected_player_index == player_index
        return False

    def _get_player_image_name(self, player_name, player_id, direction, selected):
        """
        Returns the image name of the player given their direction.

        Args:
            direction (tuple): Unit vector of the player's direction

        Returns:
            image_name (str): Image name of the player

        Raises:
            AssertionError: If the direction is invalid
        """

        selected_string = "_selected" if selected and self.show_selected_player else ""
        nurse_str = "nurse"
        nurse_id_number = int(player_id) - 1

        if direction == (0, 1):
            if player_name == "robot" and player_id == "1":
                return "robot_back" + selected_string + ".png"
            else:
                return f"{nurse_str}_{nurse_id_number}_back" + selected_string + ".png"
        elif direction == (0, -1):
            if player_name == "robot" and player_id == "1":
                return "robot_front" + selected_string + ".png"
            else:
                return f"{nurse_str}_{nurse_id_number}_front" + selected_string + ".png"
        elif direction == (1, 0):
            if player_name == "robot" and player_id == "1":
                return "robot_right" + selected_string + ".png"
            else:
                return f"{nurse_str}_{nurse_id_number}_right" + selected_string + ".png"
        elif direction == (-1, 0):
            if player_name == "robot" and player_id == "1":
                return "robot_left" + selected_string + ".png"
            else:
                return f"{nurse_str}_{nurse_id_number}_left" + selected_string + ".png"
        # print(f"Invalid direction: {direction}")
        assert False, "Invalid player name or direction"

    def test_new_positions(self, obs):
        for literal in obs.literals:
            if literal.predicate == "loc":
                player_station = literal.variables[1].name
                station_pos = self._get_station_position(player_station)
                player_index = self._get_player_index(literal)
                player_pos = self.players_pose[player_index]["position"]
                player_pos, player_direction = self._move_player_to_station(
                    player_index, player_pos, tuple(station_pos), self.layout, test=True, verbose=False
                )
                # print(
                #     f"Tested new position for player {player_index}: {player_pos}, direction: {player_direction}"
                # )
    
    def test_move_action(self, action):
        assert "move" == action.predicate.name
        target_station = action.variables[2].name
        station_pos = self._get_station_position(target_station)
        player_index = self._get_player_index(action)
        player_pos = self.players_pose[player_index]["position"]
        player_pos, player_direction = self._move_player_to_station(
            player_index, player_pos, tuple(station_pos), self.layout, test=True
        )

    def _update_player_pos(self, player_index, initial_pos, station_position):
        """"includes both moving and state update"""
        player_pos, player_direction = self._move_player_to_station(
                    player_index, initial_pos, station_position, self.layout, verbose=True
                )
        self.players_pose[player_index]["position"] = player_pos # NOTE: This updates the canvas state but is also relied on for the RL state
        self.players_pose[player_index]["direction"] = player_direction # NOTE: This updates the canvas state but is also relied on for the RL state

        return player_pos, player_direction

    def update_all_player_pos(self, obs):
        """Workaround meant for training only. Updates the player positions in the class state to match the obs.
        We need to do this because when we filter for vaild moves during each step, we need to have player grid locations maintained here
        """
        for literal in obs:
            if literal.predicate.name == "loc":
                player_station = literal.variables[1].name
                station_pos = self._get_station_position(player_station)
                player_index = self._get_player_index(literal)
                initial_pos = self.players_pose[player_index]["position"]

                player_pos, player_direction = self._update_player_pos(
                    player_index, initial_pos, tuple(station_pos)
                )

    def _draw_player(self, surface, obs, train=False):
        """
        Draws the player on the canvas with detailed debugging, and updates the player's saved position in the class state.
        """
        # print("\n--- Start of _draw_player ---")
        for literal in obs:
            if literal.predicate.name == "loc":
                player_station = literal.variables[1].name
                station_pos = self._get_station_position(player_station)
                player_index = self._get_player_index(literal)
                initial_pos = self.players_pose[player_index]["position"]

                # print(f"Player {player_index}:")
                # print(f"  Initial position: {initial_pos}")
                # print(f"  Station: {player_station}")
                # print(f"  Station position: {station_pos}")

                player_pos, player_direction = self._update_player_pos(
                    player_index, initial_pos, tuple(station_pos)
                )

                # print(f"  After moving player {player_index} : {player_pos}")

                # Check if the player is on a CPR stool
                cpr_stool_offset = 0.2 if player_station == "cpr_stool" else 0.0

                selected = self._check_selected(obs, player_index)
                player_name, player_id = self._get_player_name_and_index(literal)
                player_image_name = self._get_player_image_name(
                    player_name, player_id, player_direction, selected
                )
                # Calculate the drawing position in pixels
                # player_pos[0] * self.pix_square_size[0]: Convert x-coordinate from grid to pixels
                # (player_pos[1] - 0.2) * self.pix_square_size[1]: Convert y-coordinate from grid to pixels,
                # subtracting PLAYER_OFFSET of 0.2 (20% of a cell) to move the player up slightly
                # This ensures the player is drawn above the object they're standing on
                # (so it looks like they're standing on the center of the tile)
                draw_pos = (
                    player_pos[0] * self.pix_square_size[0],
                    (player_pos[1] - self.PLAYER_OFFSET - cpr_stool_offset) * self.pix_square_size[1],
                )

                # print(f"  Final draw position (pixels): {draw_pos}")
                # print(f"  Image: {player_image_name}")
                self._draw_image(
                    surface,
                    player_image_name,
                    draw_pos,
                    self.pix_square_size,
                )

        # print("--- End of _draw_player ---\n")

    def _draw_food(self, surface, obs):
        """
        This helper draws food on the canvas.

        Since food can be stacked, the stack information must first be determined with the on and atop predicates.
        Any food with an on predicate is the bottom of a stack and is drawn first. The foods with atop predicates
        are then drawn in the correct order afterward.

        Args:
            surface (pygame.Surface): Surface to draw on
            obs (List[Literal]): Game state predicates
        """
        at_patient_set = set()
        for literal in obs:
            # keep track of items on patient to prevent them from getting stack offset
            if (
                literal.predicate == "at"
                and "patient_bed_station" in literal.variables[1].name
            ):
                item = literal.variables[0].name
                at_patient_set.add(item)
        # print("items at patient station: ", at_patient_set)

        stack_list = []  # In the form (x, y) such that x is stacked on y
        stack_number = {}  # Stores the food item and current stack number
        for literal in obs:
            if literal.predicate == "on":
                food = literal.variables[0].name
                stack_number[food] = 1
                food_station = literal.variables[1].name
                pos = self._get_station_position(food_station)
                if "patient" not in food and "cart" not in food_station:
                    pos[
                        1
                    ] -= (
                        RobotouilleCanvas.STATION_FOOD_OFFSET
                    )  # place the food slightly above the station if it's not a cart or patient

                # print(f"Drawing food: {food} at {pos}")
                self._draw_food_image(surface, food, obs, pos * self.pix_square_size)
            if literal.predicate == "atop":
                stack = (literal.variables[0].name, literal.variables[1].name)
                stack_list.append(stack)

            # adding items onto players
            if literal.predicate == "has":
                # player_name = literal.variables[0].name
                item = literal.variables[1].name
                player_index = self._get_player_index(literal)
                player_pos = self.players_pose[player_index]["position"]
                # make sure to adjust item by same amount that player is offset by
                non_syringe_offset = 0.0 if "syringe" in item else 0.05
                pos = np.array(
                    [
                        player_pos[0],
                        player_pos[1] - self.PLAYER_OFFSET + non_syringe_offset,
                    ],
                    dtype=float,
                )
                self._draw_food_image(surface, item, obs, pos * self.pix_square_size)

        # Add stacked items
        while len(stack_list) > 0:
            i = 0
            while i < len(stack_list):
                food_above, food_below = stack_list[i]
                if food_below in stack_number:
                    stack_list.remove(stack_list[i])
                    stack_number[food_above] = stack_number[food_below] + 1
                    # Get location of station
                    for literal in obs:
                        if (
                            literal.predicate == "at"
                            and literal.variables[0].name == food_below
                        ):
                            station_pos = self._get_station_position(
                                literal.variables[1].name
                            )
                            break
                    # cheese_offset = (
                    #     -0.05 if "cheese" in food_above or "onion" in food_above else 0
                    # )
                    # on_patient_offset = (
                    #     -(self.STATION_FOOD_OFFSET + .1) if "paitent" in food_above or "patient" in food_below else 0
                    # )
                    # station_pos[1] -= (
                    #     self.STATION_FOOD_OFFSET
                    #     + 0.1 * (stack_number[food_above] - 1)
                    #     + on_patient_offset
                    # )
                    station_pos[1] -= (
                        # removed the STATION_FOOD_OFFSET here since it was only for robotouiile kitchen table
                        (0.1 * (stack_number[food_above] - 1))
                        if food_above not in at_patient_set
                        else 0
                    )
                    # print(f"Drawing stacked food: {food_above} at {station_pos}")
                    self._draw_food_image(
                        surface, food_above, obs, station_pos * self.pix_square_size
                    )
                else:
                    i += 1

    def _debug_print_layout(self):
        print("\n--- Layout Debug ---")
        for y in range(len(self.layout)):
            row = []
            for x in range(len(self.layout[0])):
                cell = self.layout[y][x]
                if cell is None:
                    row.append(".")
                else:
                    row.append(cell[:2])  # First two characters of station name
            print(" ".join(row))

        print("\nPlayer Positions:")
        for i, player in enumerate(self.players_pose):
            print(f"Player {i}: {player['position']}")
        print("--- End Layout Debug ---\n")

    def draw_to_surface(self, surface, obs):
        """
        Draws the game state to the surface.

        Args:
            surface (pygame.Surface): Surface to draw on
            obs (List[Literal]): Game state predicates
        """
        # print("Drawing to surface...")
        # self._debug_print_layout()
        self._draw_floor(surface)
        self._draw_stations(surface)
        self._draw_player(surface, obs)
        self._draw_food(surface, obs)

    # def _draw_medical_equipment(self, surface, obs):
    #     """
    #     Draws the medical equipment on the canvas.

    #     Args:
    #         surface (pygame.Surface): Surface to draw on
    #         obs (List[Literal]): Game state predicates
    #     """
    #     # for literal in obs:
    #     #     print("literal",literal)
    #     #     breakpoint()
    #     # print("obs", obs)
    #     for literal in obs:
    #         if literal.predicate == "atop":
    #             item_name = literal.variables[0].name
    #             station_name = literal.variables[1].name
    #             station_pos = self._get_station_position(station_name)

    #             if item_name == "aed":
    #                 self._draw_image(
    #                     surface,
    #                     "aed.png",
    #                     station_pos * self.pix_square_size,
    #                     self.pix_square_size * 0.5,
    #                 )
    #             elif item_name == "cpr_kit":
    #                 self._draw_image(
    #                     surface,
    #                     "cpr_kit.png",
    #                     station_pos * self.pix_square_size,
    #                     self.pix_square_size * 0.5,
    #                 )
    #             elif item_name == "ventilator":
    #                 self._draw_image(
    #                     surface,
    #                     "ventilator.png",
    #                     station_pos * self.pix_square_size,
    #                     self.pix_square_size * 0.5,
    #                 )
    #             elif item_name == "medicine":
    #                 self._draw_image(
    #                     surface,
    #                     "medicine.png",
    #                     station_pos * self.pix_square_size,
    #                     self.pix_square_size * 0.5,
    #                 )
