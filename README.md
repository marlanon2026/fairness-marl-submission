# 🏥 MARL Benchmark
<a name="readme-top"></a>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#repository-outline">Repository Outline</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#play-mode-interactive">Play Mode (Interactive)</a></li>
        <li><a href="#existing-environments">Existing Environments</a></li>
        <li><a href="#creating-your-own-environment">Creating Your Own Environment</a></li>
      </ul>
    </li>
    <li>
      <a href="#training-and-reproducing-paper-experiments">Training and Reproducing Paper Experiments</a>
      <ul>
        <li><a href="#algorithm-selection">Algorithm Selection</a></li>
        <li><a href="#fairness-configuration">Fairness Configuration</a></li>
        <li><a href="#quick-sanity-check">Quick Sanity Check</a></li>
        <li><a href="#full-training-run">Full Training Run</a></li>
        <li><a href="#reproducing-all-paper-conditions">Reproducing All Paper Conditions</a></li>
      </ul>
    </li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

<p align="middle">
  <img src="README_assets/MARL_video_simulator.gif" alt="A team of healthcare workers performing CPR, rescue breaths, and giving medication to a patient" width="250" height="250"/>
</p>

In future healthcare systems, robots will collaborate with medical professionals in real-time critical settings. To prepare them, we need to simulate complex sequences of collaborative medical actions such as administering medications, performing CPR, and assisting in triage decisions. Agents can be taught to break down complex tasks by composing simpler subtasks.

This benchmark transforms the original cooking-based Robotouille simulator into a hospital ER simulator, enabling reinforcement learning agents to train on tasks like `givemedicine`, `rescuebreaths`, and more. These tasks stress-test both coordination and specialization across heterogeneous agents, and provide an environment for evaluating fairness-aware multi-agent reinforcement learning.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Setup

1. Create and activate a virtual environment:
   ```sh
   python3 -m venv <venv-name>
   source <venv-name>/bin/activate
   ```
2. Install the benchmark and its dependencies:
   ```sh
   pip install -e .
   ```
3. Run the simulator:
   ```sh
   python main.py
   ```
   Or import it from your own code:
   ```python
   from robotouille import simulator
   simulator("original")
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Outline

- `robotouille/` — Core environment logic and simulator
- `environments/env_generator/examples/` — JSON definitions for initializing environments
- `environments/robotouille/` — PDDL files corresponding to the JSON definitions
- `epymarl/` — EPyMARL integration for MARL training
- `utils/` — Environment wrappers, RL inputs, reward handlers, and fairness components

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Play Mode (Interactive)

To play an environment interactively with keyboard and mouse:

```sh
python main.py --environment_name multiagent_rescuebreaths --mode PLAY
```

Controls:
- Click to move the agent to stations and pick up or place objects. Stack and unstack are also click-based.
- `e` performs treatment tasks at stations or on patients.
- `space` waits in place or switches the active agent (e.g., while a station is occupied).

### Existing Environments

Environment definitions live as JSON files in `environments/env_generator/examples/`. To run a specific environment:

```sh
python main.py --environment_name multiagent_test
```

To procedurally generate variants from a base JSON:

```sh
python main.py --environment_name multiagent_test --seed 42
python main.py --environment_name multiagent_test --seed 42 --noisy_randomization
```

See `environments/env_generator/README.md` for details on procedural generation.

To switch between tasks during training, change `--env-name` to any of the supported environments, such as:
- `multiagent_givemedicine_specforced_coop`
- `multiagent_rescuebreaths_specsimplemoreskills`
- `multiagent_rescuebreaths_specskilled_energy`

### Creating Your Own Environment

Add a JSON file under `environments/env_generator/examples/` and a corresponding PDDL file in `environments/robotouille/`. The transition logic lives in `robotouille.pddl` under `environments/`. The current implementation has limited support for non-Markovian actions (cut, cook) and for rendering new objects/actions; we plan to extend this.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training and Reproducing Paper Experiments

Training uses the integrated [EPyMARL](https://github.com/uoe-agents/epymarl) framework. The fairness penalty is configured via environment variables; the algorithm and environment are selected via standard EPyMARL flags.

### Algorithm Selection

Algorithm choice is controlled via the `--config` flag, which corresponds to a YAML file in `epymarl/config/algs/`. Common options:

- `qmix.yaml` — Value-decomposition for cooperative settings (used in the paper)
- `mappo.yaml` — Multi-Agent Proximal Policy Optimization
- `vdn.yaml` — Simpler value decomposition
- `coma.yaml` — Counterfactual Multi-Agent Policy Gradients

Each YAML defines hyperparameters and structural components. To extend or add algorithms, modify `epymarl/learners/` and `epymarl/controllers/` and link them via the YAML config.

### Fairness Configuration

The paper's fixed-penalty conditions are configured through environment variables read by the reward handlers in `utils/`:

| Variable | Values used in paper | Description |
|---|---|---|
| `USE_FAIRNESS` | `True` | Enable workload-based fairness reward shaping |
| `LAMBDA_FAIRNESS` | `0`, `10`, `30`, `50` | Fixed-penalty weight (`0` = no-fairness baseline) |
| `FAIRNESS_ALPHA` | `0`, `1.0` | Workload/skill trade-off (composite objective `L3 = αL1 + (1−α)L2`) |
| `INITIAL_LAMBDA` | `0` | Starting λ for adaptive schedules |
| `WARMUP_EPISODES` | `0` | Warmup before λ updates |
| `SCHEDULE_TYPE` | `constant` | λ schedule type for fixed-penalty runs |
| `OBSERVATION_MODE` | `LARGE` | Observation tensor configuration |

The FEN baseline uses a separate set of variables (see *Reproducing All Paper Conditions* below).

### Quick Sanity Check

A short single-seed run that verifies the environment loads, the fairness handler attaches, and training proceeds. Runtime is roughly 10–20 minutes on a single GPU.

```sh
USE_FAIRNESS=True LAMBDA_FAIRNESS=10 FAIRNESS_ALPHA=0 \
INITIAL_LAMBDA=0 WARMUP_EPISODES=0 SCHEDULE_TYPE=constant \
OBSERVATION_MODE=LARGE \
python epymarl/main.py \
    --config=qmix \
    --env-config=gymma \
    with env_args.time_limit=0 \
    env_args.observation_mode=LARGE \
    t_max=200000 \
    test_interval=2000 \
    test_nepisode=50 \
    seed=123 \
    --env-name=multiagent_rescuebreaths_specsimplemoreskills
```

### Full Training Run

A single full-length run matching the paper (40M timesteps, fixed-λ workload fairness):

```sh
USE_FAIRNESS=True LAMBDA_FAIRNESS=10 FAIRNESS_ALPHA=0 \
INITIAL_LAMBDA=0 WARMUP_EPISODES=0 SCHEDULE_TYPE=constant \
OBSERVATION_MODE=LARGE \
python epymarl/main.py \
    --config=qmix \
    --env-config=gymma \
    with env_args.time_limit=0 \
    env_args.observation_mode=LARGE \
    t_max=40000000 \
    test_interval=2000 \
    save_model_interval=10000000 \
    test_nepisode=200 \
    seed=123 \
    --env-name=multiagent_rescuebreaths_specsimplemoreskills \
    use_cuda=True \
    buffer_cpu_only=False \
    checkpoint_path="checkpoints/qmix_l10_a0_s123"
```

### Reproducing All Paper Conditions

The paper reports five seeds (`12345, 23456, 34567, 45678, 56789`).

**No-fairness baseline and fixed-λ conditions** use the *Full Training Run* command above, varying `LAMBDA_FAIRNESS`:

| Condition | LAMBDA_FAIRNESS | FAIRNESS_ALPHA |
|---|---|---|
| No fairness baseline | `0` | `0` |
| Fixed λ=10 | `10` | `0` |
| Fixed λ=30 | `30` | `0` |
| Fixed λ=50 | `50` | `0` |

**FEN baseline** uses a different reward mechanism, environment, and hyperparameter set:

```sh
USE_FAIRNESS=false USE_FAIR_GNE=false \
REWARD_TYPE=fen FEN_WEIGHT=1.0 FEN_C=1.0 FEN_EPSILON=1e-6 \
OBSERVATION_MODE=LARGE \
python epymarl/main.py \
    --config=qmix \
    --env-config=gymma \
    with env_args.time_limit=50 \
    env_args.observation_mode=LARGE \
    batch_size=64 \
    batch_size_run=1 \
    buffer_cpu_only=True \
    buffer_size=2500 \
    lr=0.0005 \
    optim_alpha=0.99 \
    optim_eps=0.00001 \
    grad_norm_clip=10 \
    t_max=40000000 \
    test_interval=50000 \
    save_model_interval=1000000 \
    test_nepisode=10 \
    seed=123 \
    --env-name=multiagent_rescuebreaths_specsimplefen \
    use_cuda=True \
    checkpoint_path="checkpoints/qmix_fen_s123"
```

Run each of the four fixed-λ conditions plus FEN with all five seeds: 25 runs total (4 × 5 + 5).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Built With

- [EPyMARL](https://github.com/uoe-agents/epymarl) — Multi-agent RL framework
- [PDDLGym](https://github.com/tomsilver/pddlgym) — PDDL-based environment interface
- [PyGame](https://www.pygame.org/) — Rendering

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

See `LICENSE`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
