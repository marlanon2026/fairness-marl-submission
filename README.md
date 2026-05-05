# 🏥 Hosp_Robotouille: Hospital Task Simulator
<a name="readme-top"></a> 

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setup">Setup</a></li>
      </ul>
    </li>
    <li><a href="#repository-outline-with-major-directories">Repository Outline with Major Directories</a></li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#use-existing-environments">Use Existing Environments</a></li>
        <li><a href="#create-your-own-environment">Create your own Environment!</a></li>
        <li><a href="#changing-environments">Changing Environments</a></li>
      </ul>
    </li>
    <li><a href="#running-marl">Running MARL</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

In future healthcare systems, robots will collaborate with medical professionals in real-time critical settings. To prepare them, we need to simulate complex sequences of collaborative medical actions such as administering medications, performing CPR, or assisting in triage decisions. We can teach robots to break down complex tasks by showing them how to perform easier tasks, subtasks, and then combine those subtasks to perform harder tasks. 

**Hosp_Robotouille** transforms the original cooking-based Robotouille simulator into a hospital ER simulator, enabling reinforcement learning agents to train on tasks like `givemedicine`, `rescuebreaths`, and more. These tasks stress test both coordination and specialization across agents.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

It is super easy to get started by trying out an existing environment or creating your own environment!

### Setup

1. Create and activate your virtual environment
   ```sh
   python3 -m venv <venv-name>
   source <venv-name>/bin/activate
   ```
2. Install Hosp_Robotouille and its dependencies
   ```sh
   pip install -e .
   ```
3. Run Robotouille!

   ```sh
   python main.py
   ```

   or import the simulator to any code by adding

   ```python
   from robotouille import simulator

   simulator("original")
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Repository Outline with Major Directories
- `robotouille/` - Core environment logic and simulator
- `environments/env_generator/examples` - JSON definitions to init envs
- `environments/robotouille/` - PDDL corresponding to JSON definitions
- `epymarl/` - Epymarl integration for MARL training
- `utils/` - Environment wrappers, RL inputs and utils, reward functions

## Usage

### Use Existing Environments

To play an existing environment, you can choose from the JSON files under `environments/env_generator/examples/`. For example, to play the `multiagent_test` environment, simply run

```sh
python main.py --environment_name multiagent_test
```

You can interact with the environment with keyboard and mouse, using the following keys:

- Click to move the robot to stations and pick up or place down objects. You can also stack and unstack objects by clicking.
- 'e' can be used to perform treatment tasks at stations or on patients.
- 'space' can be used to stay in place/ change agent selected (e.g. you are waiting for a station to be freed )

If you would like to procedurally generate an environment based off a JSON file, run the following commands

```sh
python main.py --environment_name multiagent_test --seed 42
python main.py --environment_name multiagent_test --seed 42 --noisy_randomization
```
Refer to the `README.md` under `environments/env_generator` for details on procedural generation.


### Changing Environments
To explore or benchmark different tasks, change --env-name to any of the supported names, such as:
- multiagent_givemedicine_specforced_coop
- multiagent_rescuebreaths_specskilled_energy

These names are linked to environment definitions in 'robotouille/env_generator'.

### Create your own Environment!

Add JSON files under environments/env_generator/examples/. Also, add a corresponding PDDL file in environments/robotouille/.
Contact us if you'd like to customize PDDL mechanics or actions. If you would like to modify the transitions of the environment entirely, refer to `robotouille.pddl` under `environments`. We currently have limited support for customization through the PDDL for non-Markovian actions (cut / cook) and for rendering new objects / actions but plan to add more support in the future. Please contact us for more details if interested.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running MARL: 

Hosp_Robotouille supports integration with Epymarl for multi-agent reinforcement learning.

### Modes
Set mode in robotouille_simulator.py:

```python
self.mode = Mode.PLAY   # interact with keyboard on renderer
self.mode = Mode.TRAIN  # RL training
self.mode = Mode.LOAD   # load and test agent
```

### Selecting Training Algorithms 
Supports a variety of MARL algorithms. The choice of algorithm is controlled via the --config flag when launching training scripts and corresponds to YAML configuration files found in: 
`epymarl/config/algs/` 

Each YAML file defines hyperparameters and structural components for a particular algorithm. Some commonly used configs include:

+ mappo.yaml – Multi-Agent Proximal Policy Optimization (recommended for stability and scalability)
+ coma.yaml – Counterfactual Multi-Agent Policy Gradients (best for discrete action spaces with dense rewards)
+ qmix.yaml – Value-decomposition for cooperative settings
+ vdn.yaml – Simpler value decomposition (used for baselines or low-complexity tasks)

Example:
```sh
python epymarl/main.py --config=mappo ...
```
This command uses the `epymarl/config/algs/mappo.yaml` file to initialize the training process with MAPPO.
You can modify or extend algorithm configs by:
- Adjusting learning rates, batch sizes, entropy regularization

- Switching network architectures (e.g. recurrent vs feedforward)

- Changing value mixing strategies in cooperative agents

Advanced users can also implement custom algorithms by extending Epymarl/ `learners/` and `controllers/` directories and linking their logic in the YAML config.

### Training via Epymarl

Following is an example of the terminal command to test an algorithm in an environment: 
```sh
python epymarl/main.py \
    --config=mappo \
    --env-config=gymma with env_args.time_limit=50 \
    --env-name=multiagent_givemedicine_specforced_coop \
    --note="100nquad_unscaledrewardforced_coop_givemedicine_mappo" \
    --tags="100n forced_coop givemedicine mappo name=multiagent_rescuebreaths_specskilled_energy"
```

Explanation of Flags:
--config: Selects the RL algorithm config from epymarl/config/algs/ (e.g., mappo.yaml).


--env-config=gymma: Specifies the wrapper config used by Epymarl (must be gymma for Gym environments).


with env_args.time_limit=50: Override default environment arguments from env_args. Sets episode length to 50.


--env-name: Matches the registered name in the Gym registry (e.g., multiagent_givemedicine_specforced_coop). This name should correspond to a file or spec in robotouille/env_generator or the gymma wrapper.


--note: A descriptive string saved with experiment logs/checkpoints.


--tags: Useful for grouping experiment metadata, like number of agents (100n), reward structure (unscaledrewardforced), task name, or algorithm.


You can launch these manually through the above command or by running `slurm scripts`. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Running RL:

Running RL:
We have integrated robust reinforcement learning (RL) capabilities into Robotouille, allowing users to explore and train RL agents within our diverse cooking environments. This section guides you through the process of setting up and running RL algorithms with Robotouille.

Setup for RL Training:

1. Initialize the Environment: First, ensure that you have set up Robotouille as per the instructions in the Setup section. In robotouille/robotouille_simulator, you can change the mode to be mode.PLAY, mode.TRAIN, or mode.LOAD. When using RL (mode.TRAIN or mode.LOAD), the model gets saved/loaded in from the file field. We do not support changing the run configuration through terminal at this point.

2. Select an RL Algorithm: Robotouille supports various RL algorithms. You can choose from standard options like Proximal Policy Optimization (PPO) and Advantage Actor Critic (A2C), among others. The choice of algorithm can significantly affect how the agent learns and performs tasks in the cooking environment. You can either load a pretrained PPO model or train one from scratch per simulation.

3. Configure the Algorithm: Modify the algorithm's parameters to suit your specific requirements. This could include setting the learning rate, the number of training episodes, or the reward structure. We provide a default configuration, but encourage experimentation for optimal results. A good metric might be 500,000 timesteps. You can change how the agent learns in robotouille/robotouille_simulator by changing between PPO/A2C and changing the run parameters of n_steps and total_timesteps. You can change the episode length (how many steps a robot can take in an environment before truncating) by changing self.max_steps in utils/rl_wrapper.


Running the Training:
To initiate the training process, follow these steps:

1. Launch the Training Script: Run the training script with the chosen environment and RL algorithm. You can use a command like:

2. Monitor the Training: Training an RL agent can be time-consuming. Monitor the agent's progress through the logs or visualizations provided. Watching the agent learn and adapt over time can offer valuable insights into the effectiveness of your chosen RL setup.

3. Evaluate the Agent: After the training is complete, evaluate the agent's performance. We provide tools to test the agent in the environment, allowing you to assess its ability to perform cooking tasks with efficiency and accuracy.

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#setup">Setup</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#use-existing-environments">Use Existing Environments</a></li>
        <li><a href="#create-your-own-environment">Create your own Environment!</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

<p align="middle">
  <img src="README_assets/MARL_video_simulator.gif" alt="A team of Healthcare workers performing give medicine on a patient starting from giving CPR , then rescue breathes , then giving air, before finally giving mediine ]" width="250" height="250"/>
</p>
