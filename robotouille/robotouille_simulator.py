from enum import Enum
import subprocess
import time
import numpy as np
import pygame
from stable_baselines3 import PPO
from rl.rl_wrapper import RLWrapper
from utils.robotouille_input import create_action_from_control
from robotouille.robotouille_env import create_robotouille_env

# from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_vec_env


class Mode(Enum):
    PLAY = 1
    TRAIN = 2
    LOAD = 3


class type(Enum):
    SINGLE = 1
    MULTI = 2


file = "runs/custom_ppo.zip"


def simulator(
    environment_name: str,
    seed: int = 42,
    noisy_randomization: bool = False,
    mode=Mode.TRAIN,
    type=type.MULTI,
    alg: str = "qmix",
    note: str = "",
    tags: list = [],
):
    # Load or train agent
    if (mode == mode.TRAIN or mode == mode.LOAD) and type == type.SINGLE:
        single_rl_simulator(environment_name, seed, noisy_randomization, mode)
        return

    if (mode == mode.TRAIN or mode == mode.LOAD) and type == type.MULTI:
        if mode == mode.TRAIN:
            multi_rl_simulator(
                environment_name, seed, noisy_randomization, alg, note, tags
            )
        elif mode == mode.LOAD:
            load_multi_simulator(environment_name, seed, noisy_randomization)
        return

    # Your code for robotouille goes here
    env, json, renderer = create_robotouille_env(
        environment_name, seed, noisy_randomization
    )
    obs, info = env.reset()
    env.render(mode="human")
    done = False
    truncated = False
    interactive = False  # Set to True to interact with the environment through terminal REPL (ignores input)

    # Simulate the environment
    while not done and not truncated:

        # Construct action from input
        pygame_events = pygame.event.get()
        # Mouse clicks for movement and pick/place stack/unstack
        mousedown_events = list(
            filter(lambda e: e.type == pygame.MOUSEBUTTONDOWN, pygame_events)
        )
        # Keyboard events ('e' button) for cut/cook ('space' button) for noop
        keydown_events = list(filter(lambda e: e.type == pygame.KEYDOWN, pygame_events))
        action = create_action_from_control(
            env, obs, mousedown_events + keydown_events, renderer
        )

        if not interactive and action is None:
            # Retry for keyboard input
            continue
        obs, reward, done, info = env.step(action=action, interactive=interactive)

        env.render(mode="human")
    env.render(close=True)


def single_rl_simulator(
    environment_name: str, seed: int, noisy_randomization: bool, mode
):
    config = {
        "num_cuts": {"lettuce": 3, "default": 3},
        "cook_time": {"patty": 3, "default": 3},
        "num_compressions": {"patient": 3, "default": 3},
        "num_breaths": {"patient": 3, "default": 3},
        "num_shocks": {"patient": 1, "default": 1},
        "num_medicine_doses": {"patient": 1, "default": 1},
    }

    env, json, renderer = create_robotouille_env(
        environment_name, seed, noisy_randomization
    )
    obs, info = env.reset()
    env.render(mode="human")
    done = False
    truncated = False
    interactive = False

    rl_env = RLWrapper(env, config, renderer)
    rl_env.render(mode="human")
    obs, info = rl_env.reset()

    if mode == mode.LOAD:
        model = PPO.load(file, env=rl_env)

    else:
        model = PPO("MlpPolicy", rl_env, verbose=1, n_steps=1024, ent_coef=0.01)

        model.learn(
            total_timesteps=100000, reset_num_timesteps=False, progress_bar=True
        )
        model.save(file)

    obs, info = rl_env.reset()

    while not done and not truncated:
        pygame_events = pygame.event.get()
        keydown_events = list(filter(lambda e: e.type == pygame.KEYDOWN, pygame_events))

        if len(keydown_events) == 0:
            continue

        if keydown_events[0].key == pygame.K_SPACE:
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = rl_env.step(action=action, debug=True)
            env.render(mode="human")
    env.render(close=True)


def multi_rl_simulator(
    environment_name: str, seed: int, noisy_randomization: bool, alg, note, tags
):
    tags = " ".join(tags)
    arguments = [
        "python",
        "epymarl/main.py",
        f"--config={alg}",
        "--env-config=gymma",
        "with",
        "env_args.time_limit=50",
        f"--env-name={environment_name}",
        'checkpoint_path="results/models/vdn_seed836503631_None_2025-01-12 00:12:45.119702/4000098"',
        "evaluate=True",
        "render=True"
        f"--note={note}",
        f"--tags={tags}",
    ]

    epymarl = subprocess.run(arguments)


def load_multi_simulator(environment_name, seed, noisy_randomization):
    env, json, renderer = create_robotouille_env(
        environment_name, seed, noisy_randomization, show_selected_player=False
    )
    obs, info = env.reset()
    env.render(mode="human")
    done = False
    truncated = False
    interactive = False

    with open(
        "results/models/mappo_seed457845154_None_2025-01-05 14:20:01.563234/8000942",
        "r",
    ) as f:
        for line in f:
            action = line.strip()

            if not interactive and action is None:
                # Retry for keyboard input
                continue
            # print("taking action: ", action)
            obs, reward, done, info = env.step(action=action, interactive=interactive)
            # only render when all agents have moved once
            if env.is_start_of_timestep():
                time.sleep(0.7)
                env.render(mode="human")

    env.render(close=True)
