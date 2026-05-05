import utils.pddlgym_interface as pddlgym_interface
from renderer.renderer import RobotouilleRenderer
from utils.robotouille_wrapper import RobotouilleWrapper
from environments.env_generator import builder
from environments.env_generator import procedural_generator


def _parse_renderer_layout(environment_json):
    """
    Parses and returns renderer layout from the environment json.

    Args:
        environment_json (dict): The environment json.

    Returns:
        layout (list): A 2D list of station names representing where stations are in the environment.
    """
    # Update environment names with unique versions
    width, height = environment_json["width"], environment_json["height"]
    layout = [[None for _ in range(width)] for _ in range(height)]
    _, updated_environment_json = builder.build_objects(environment_json)
    stations = updated_environment_json["stations"]
    for station in stations:
        x, y = (
            station["x"],
            height - station["y"] - 1,
        )  # Origin is in the bottom left corner
        layout[y][x] = station["name"]
    return layout


def _procedurally_generate(environment_json, seed, noisy_randomization):
    """
    Attempts to procedurally generated environment until success and returns the environment.

    Args:
        environment_json (dict): The environment json.
        seed (int): The seed to be used for randomization.
        noisy_randomization (bool): Whether or not to use noisy randomization.

    Returns:
        env (RobotouilleWrapper): The Robotouille environment.
    """
    generated_environment_json = None
    while generated_environment_json is None:
        try:
            generated_environment_json = procedural_generator.randomize_environment(
                environment_json, seed, noisy_randomization
            )
            print(f"Successfully created environment with seed {seed}.")
        except Exception as e:
            print(f"Encountered error when creating environment with seed {seed}.")
            print(e)
            seed += 1
    return generated_environment_json

def add_player_info(environment_json):
    environment_json["config"]["num_players"] = len(environment_json["players"])
    environment_json["config"]["player_info"] = {}
    
    expected_skills = [
        "compresschest", "giverescuebreaths", "giveshock", "givemedicine",
        "move", "moveitem", "pick-up", "place", "stack", "stackunder"
    ]
    
    for i, player in enumerate(environment_json["players"]):
        player_name = player['name']
        player_skills = {}
        for skill in expected_skills:
            player_skills[skill] = player.get("skill_info", {}).get(skill, 1)
        environment_json["config"]["player_info"][player_name] = player_skills

    #     # After processing
    # for player, skills in environment_json["config"]["player_info"].items():
    #     print(f"[CONFIG] {player}: {skills}")

        
    
    return environment_json["config"]


def create_robotouille_env(problem_filename, seed=None, noisy_randomization=False, show_selected_player=True):
    """
    Creates and returns an Robotouille environment.

    Note this function is used to load pre-created environments. These can be located
    in the environments/env_generator/examples folder.

    Args:
        problem_filename (str): The name of the problem file (without extension).
        seed (int): The seed to be used for randomization or None for the pre-created environment.
        noisy_randomization (bool): Whether or not to use noisy randomization.

    Returns:
        env (RobotouilleWrapper): The Robotouille environment.
        environment_json (dict): The environment json.
        renderer (RobotouilleRenderer): The Robotouille renderer.
    """
    env_name = "robotouille"
    is_test_env = False
    json_filename = f"{problem_filename}.json"
    environment_json = builder.load_environment(json_filename)
    if seed is not None:
        environment_json = _procedurally_generate(
            environment_json, int(seed), noisy_randomization
        )
    layout = _parse_renderer_layout(environment_json)
    renderer = RobotouilleRenderer(layout=layout, players=environment_json["players"], show_selected_player=show_selected_player)
    # breakpoint()
    render_fn = renderer.render
    problem_string, environment_json = builder.build_problem(
        environment_json
    )  # IDs objects in environment
    builder.write_problem_file(problem_string, f"{problem_filename}.pddl")
    pddl_env = pddlgym_interface.create_pddl_env(
        env_name, is_test_env, render_fn, f"{problem_filename}.pddl"
    )
    environment_json["config"] = add_player_info(environment_json)
    # builder.delete_problem_file(f"{problem_filename}.pddl")
    return (
        RobotouilleWrapper(pddl_env, environment_json["config"], renderer),
        environment_json,
        renderer,
    )
