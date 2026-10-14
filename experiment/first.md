# Experiment 01 — Reinforcement Learning for Indian Road Traffic Negotiation

Project type: Applied RL · Simulator development · Autonomous driving research

## 1. Experiment objective

Build a simulated Indian road intersection where an autonomous scooter learns to navigate dense, mixed traffic by predicting vehicle movement, identifying safe gaps, and deciding when to move, brake, or wait.

The first experiment will use a Python-based 2D simulator. Once the agent demonstrates reliable behavior, the same environment can be extended and connected to a 3D Roblox game.

The goal is not simply to train an agent that reaches its destination. It is to investigate whether reinforcement learning can produce a policy that negotiates dense traffic safely and efficiently despite uncertain behavior from other road users.

### Research questions

1. Can an RL agent learn to select safe gaps in dense traffic?
2. Can it learn when waiting is better than attempting a crossing?
3. Does training against diverse driver behaviors improve generalization?
4. Can the learned policy outperform a rule-based driving baseline?
5. Does the policy continue to work when traffic density, driver behavior, and road configurations change?

### Experimental hypothesis

A PPO-trained policy, using structured observations of nearby vehicles and relative motion, will achieve a higher destination-completion rate than a naive rule-based policy in the training environment. Whether it also outperforms a conservative gap-selection baseline under unseen traffic conditions will be tested experimentally.

## 2. Overall experiment architecture

Traffic world

Intersection, vehicles, roads, destinations, traffic signals

Observation generator

Ego state, nearby vehicles, relative speed, gaps, signal state

PPO agent

Selects throttle, braking, steering or waiting actions

Reward and evaluation

Progress, collisions, near misses, travel time, generalization

The simulator is the environment. The agent receives observations and chooses actions. The environment advances the vehicles, calculates the reward, and returns the next observation. PPO learns from these repeated interactions.

Use these tools:

| Tool | Purpose |
| --- | --- |
| Python | Main simulator and experiment logic |
| NumPy | Vehicle states, geometry, numerical calculations |
| Pygame | 2D visualization and debugging |
| Gymnasium | Standard RL environment interface |
| Stable-Baselines3 | PPO implementation |
| PyTorch | Neural-network training backend used by the RL library |
| Matplotlib | Learning curves and evaluation plots |
| Pandas | Store and analyze experiment results |
| pytest | Test the environment and its safety logic |

Use a fixed simulation timestep initially, such as 0.1 seconds, so that each action has a consistent physical meaning.

## 3. Project folder structure

Create a project named `indian_traffic_rl`.

Use the following structure from the beginning. Each file has one clear responsibility, which makes the simulator easier to debug and the experiment easier to reproduce.

text
```

indian_traffic_rl/
│
├── README.md
├── requirements.txt
├── config.py
│
├── traffic_env/
│   ├── __init__.py
│   ├── environment.py
│   ├── road.py
│   ├── vehicle.py
│   ├── traffic.py
│   ├── observations.py
│   ├── actions.py
│   ├── rewards.py
│   └── scenarios.py
│
├── agents/
│   ├── __init__.py
│   ├── rule_based.py
│   └── train_ppo.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   └── plots.py
│
├── visualization/
│   ├── render.py
│   └── play.py
│
├── tests/
│   ├── test_environment.py
│   ├── test_observations.py
│   └── test_safety.py
│
├── experiments/
│   └── experiment_01.yaml
│
├── models/
├── results/
└── logs/
```

### What each file is responsible for

| File | What you implement in it |
| --- | --- |
| `config.py` | Global parameters: timestep, road size, vehicle count, speed limits |
| `environment.py` | Gymnasium `reset()` and `step()`; coordinates the simulation |
| `road.py` | Road geometry, intersection boundaries, lanes, routes and destinations |
| `vehicle.py` | Vehicle position, orientation, velocity, acceleration and collision geometry |
| `traffic.py` | Spawns NPCs and advances their driving behavior |
| `observations.py` | Converts the world state into the agent's numerical input |
| `actions.py` | Maps the agent's actions into steering, throttle and braking |
| `rewards.py` | Computes progress, safety, smoothness and completion rewards |
| `scenarios.py` | Generates traffic layouts, densities and driver personalities |
| `rule_based.py` | Implements non-learning driving baselines |
| `train_ppo.py` | Trains and saves the PPO policy |
| `evaluate.py` | Tests saved models on fixed evaluation scenarios |
| `metrics.py` | Calculates success rate, collision rate, travel time and other metrics |
| `plots.py` | Produces charts from recorded experiment data |
| `render.py` | Draws the road, vehicles, traffic signals and debug overlays |
| `play.py` | Runs the simulation visually, using a selected policy |
| `experiment_01.yaml` | Stores the exact settings for a reproducible run |
| `tests/` | Checks environment behavior, observation validity and safety calculations |

`models/`, `results/`, and `logs/` are output directories. You generally do not write code inside them.

## 4. Phase A — Set up the Python environment

### Step A1. Create the environment

Install Python 3.11 or another compatible version, then create a virtual environment.

Run these commands from your terminal:

bash
```
mkdir indian_traffic_rl
cd indian_traffic_rl

python -m venv .venv
```

Activate it on Windows:

bash
```
.venv\Scripts\activate
```

On Linux or macOS:

bash
```
source .venv/bin/activate
```

### Step A2. Create `requirements.txt`

Put the initial dependencies in this file:

text
```
numpy
pygame
gymnasium
stable-baselines3
torch
matplotlib
pandas
pyyaml
pytest
```

Install them:

bash
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For reproducibility, save exact working package versions once the setup succeeds:

bash
```
pip freeze > requirements-lock.txt
```

If you use a GPU, verify PyTorch's hardware support separately. A GPU is not necessary for this small initial experiment.

### Step A3. Create `README.md`

Record the project goal, installation instructions, training command, evaluation command, and the meaning of each metric. This becomes your experiment notebook in repository form.

Deliverable for Phase A: a working Python environment that can import Gymnasium, Stable-Baselines3, NumPy, and Pygame.

## 5. Phase B — Build the simplest possible traffic simulator

Do not start with a complete Indian city. Build one four-way intersection.

Crossroads Of Two Roads With Pedestrian Paths, Curbs And Boards, Road Markings, Grass, Bushes And Trees. Top View Vector Flat Illustration. Royalty Free SVG, Cliparts, Vectors, and Stock Illustration. Image 125368856.

### Step B1. Implement `road.py`

Create a two-dimensional coordinate system:

- The intersection is centered at $(0,0)$.
- The horizontal road runs along the X-axis.
- The vertical road runs along the Y-axis.
- Each road has a configurable width.
- Vehicles have a destination and a desired route through the intersection.

For the first version, model straight-through routes only. Add left and right turns after the basic system works.

Represent each vehicle with a position, heading, speed and dimensions. You do not need detailed vehicle physics or suspension yet.

### Step B2. Implement `vehicle.py`

Every vehicle should have a state similar to:

$$
x_i = (p_x,p_y,v,\theta,a)
$$

where:

- $p_x,p_y$: position.
- $v$: speed.
- $\theta$: heading angle.
- $a$: longitudinal acceleration.

Add vehicle dimensions for collision checking.

Use a bicycle-model approximation later if you need realistic steering. For the first experiment, a simple kinematic model with bounded acceleration and steering is enough.

### Step B3. Implement `traffic.py`

Create background traffic consisting of cars, scooters, autos and buses.

Each NPC needs:

- A route and destination.
- A preferred speed.
- A reaction delay.
- A gap-acceptance threshold.
- A yielding tendency.
- A driver personality.

Start with a rule-based NPC policy. For example, an NPC can slow down when the predicted distance to a vehicle ahead falls below a safety threshold.

The NPC should not teleport to its next position. Every vehicle must advance through the same simulation clock.

### Step B4. Implement `scenarios.py`

Create several predefined traffic situations:

1. Low-density intersection.
2. Medium-density intersection.
3. Dense mixed traffic.
4. An approaching vehicle that blocks the obvious gap.
5. A vehicle that unexpectedly slows down.
6. A cautious driver and an assertive driver approaching the same conflict point.

The purpose is to create different decision-making situations, not merely different vehicle counts.

### Step B5. Implement `visualization/render.py`

Draw:

- Road surfaces and intersection boundaries.
- Vehicle rectangles or sprites.
- The ego vehicle in a distinct color.
- Vehicle velocity vectors.
- Predicted paths and collision zones.
- The destination and route.

Add optional debug overlays for each NPC's intended action and gap-acceptance threshold. These make it much easier to understand why an agent fails.

Deliverable for Phase B: a visually working intersection where NPC vehicles move and interact without an RL agent.

## 6. Phase C — Define the RL environment

This is the most important engineering phase. Before training anything, implement a correct Gymnasium environment.

Create `traffic_env/environment.py`.

The environment must implement:

- `__init__()`: configure the road, vehicles and observation/action spaces.
- `reset()`: create a new scenario and return the initial observation.
- `step(action)`: apply an action, advance the simulation, compute reward, and return the next state.
- `render()`: show the simulation if visualization is enabled.
- `close()`: release visualization resources.

The standard Gymnasium interface returns:

python
```
observation, reward, terminated, truncated, info
```

Here, `terminated` means a natural terminal event such as reaching the destination or a terminal collision. `truncated` means the episode ended because of an external limit, such as a time limit.

### Step C1. Design the observation space

Create `traffic_env/observations.py`.

Use a fixed-length numerical vector. A practical initial observation might include:

| Feature | Representation |
| --- | --- |
| Ego speed | Normalized scalar |
| Distance to destination | Normalized scalar |
| Lateral position error | Normalized scalar |
| Heading error | Normalized scalar |
| Traffic-light state | Encoded value |
| Nearest 8 vehicles | Fixed-size feature slots |
| Each nearby vehicle | Relative X/Y, relative velocity X/Y, heading difference, dimensions or type |
| Closest predicted conflict | Time-to-collision or a safe sentinel value |
| Available gaps | Relative gap distances and closing speeds |

For example, eight neighboring vehicles with six features each, plus ten ego and intersection features, produce a 58-dimensional observation vector.

The exact number is not important. Consistency is.

Normalize continuous features to reasonably bounded ranges. Use a consistent convention for missing vehicles, such as zero-valued features plus an explicit presence mask. This avoids confusing a missing vehicle with a real vehicle at zero relative distance.

Do not include future ground-truth positions of other vehicles in the agent's observation. That would create information leakage.

### Step C2. Design the action space

Create `traffic_env/actions.py`.

For the first experiment, use continuous control:

$$
a_t = [u_{\text{throttle}},u_{\text{steer}}]
$$

where each action is bounded between $-1$ and $1$.

Map the throttle value to acceleration or braking, and map steering to a bounded steering angle or steering rate. Apply physical limits inside the simulator.

This is a natural fit for PPO with a continuous action space.

A discrete action space such as accelerate, brake, wait and steer can also work, but I recommend selecting one action design and holding it fixed throughout the initial comparison.

Honk behavior: keep honking disabled or cosmetic in Experiment 01. Add it as a meaningful action in a later experiment, when you can measure whether it helps drivers coordinate or merely creates noise.

### Step C3. Choose the simulation timestep

Start with a fixed timestep of 0.1 seconds, equivalent to 10 simulation updates per second.

At each environment step:

1. Receive the ego agent's action.
2. Convert it into vehicle controls.
3. Advance all NPC policies and vehicle dynamics.
4. Check vehicle collisions and safety violations.
5. Update the destination progress.
6. Generate the next observation.
7. Calculate reward and termination conditions.

Keep the timestep consistent during training and evaluation.

## 7. Phase D — Design the reward function

Create `traffic_env/rewards.py`.

The agent must learn to reach the destination while avoiding collisions, rather than discovering that sitting still or rushing forward is an easy way to maximize reward.

Use an interpretable reward function:

$$
r_t =
w_p r_{\text{progress}}
-w_c c_{\text{collision}}
-w_n c_{\text{near-miss}}
-w_j c_{\text{jerk}}
-w_v c_{\text{rule}}
$$

Add a terminal completion bonus and an appropriate time penalty.

### Reward components

| Component | Desired behavior |
| --- | --- |
| Progress reward | Move closer to the destination |
| Completion bonus | Finish the route |
| Collision penalty | Avoid collisions |
| Near-miss penalty | Maintain a reasonable safety margin |
| Smoothness penalty | Avoid unnecessary harsh acceleration and steering |
| Rule penalty | Respect red signals and defined safety constraints |
| Time penalty | Avoid waiting forever |

The coefficients should be chosen through experiments rather than assumed to be optimal.

### Important design rule

A near miss should not be defined solely by distance. Two vehicles may be close but moving apart safely. A better measure considers relative velocity, predicted closest approach, and the time remaining before a potential conflict.

Likewise, do not penalize all waiting. Waiting for a genuinely safe gap is correct behavior. Penalize unnecessary delay through the time cost, but ensure the agent receives sufficient progress and completion rewards to avoid learning a stationary policy.

Record each reward component separately in `info`. This helps you identify reward hacking and understand what the policy is optimizing.

## 8. Phase E — Create the rule-based baseline

Create `agents/rule_based.py`.

Before training PPO, implement a baseline that accepts a gap only when it exceeds a predefined safety threshold.

Its logic might be:

1. Identify vehicles that could intersect the ego vehicle's path.
2. Estimate the time until each vehicle reaches the conflict zone.
3. Estimate the time needed for the ego vehicle to cross.
4. Account for a safety margin and uncertainty.
5. Proceed only when the predicted crossing is sufficiently safe.
6. Otherwise, slow down or wait.

This baseline is not intended to be perfect. It establishes a meaningful comparison.

Also include a conservative variant that requires larger gaps. Together, these baselines help distinguish whether the RL agent has learned useful negotiation behavior or simply moves more aggressively.

Test both baselines under the same scenario seeds, traffic configurations and evaluation rules that you will later use for PPO.

Deliverable for Phase E: two working baseline policies and a report of their success rates, collisions, and travel times.

## 9. Phase F — Validate the environment before training

Create the files in `tests/`.

Do not begin a long training run until the simulator passes basic checks.

### `tests/test_environment.py`

Test that:

- `reset()` returns an observation with the correct shape.
- Every observation contains finite numerical values.
- Every action within the defined action space can be processed.
- `step()` returns all five expected values.
- An episode eventually terminates or truncates.
- Resetting with the same seed reproduces the same initial scenario.

### `tests/test_observations.py`

Test that:

- Missing vehicles are represented consistently.
- Vehicle ordering is deterministic, for example by distance to the ego vehicle.
- Relative positions and velocities have the correct signs.
- Observation values remain within expected bounds.
- The observation never uses hidden future state.

### `tests/test_safety.py`

Test that:

- Vehicle collision geometry works at different orientations.
- Stationary vehicles do not spontaneously move.
- Braking produces the expected reduction in speed.
- Vehicles respect their maximum acceleration and speed.
- A red-light crossing is detected when the test scenario requires a stop.
- The simulator does not allow vehicles to teleport through obstacles.

Run:

bash
```
pytest -q
```

Then run a short visual simulation and manually inspect several traffic scenarios.

Success criterion: the simulator advances correctly, collision checks work, and the baseline can complete at least some routes without relying on broken physics.

## 10. Phase G — Train the first PPO agent

Create `agents/train_ppo.py`.

Use Stable-Baselines3's PPO implementation rather than implementing PPO from scratch for this first experiment. This lets you focus on the environment, reward design, and evaluation. You can implement PPO yourself later as a separate algorithmic exercise.

### Initial training configuration

Put the experiment settings in `experiments/experiment_01.yaml`.

yaml
```

experiment_name: indian_traffic_exp01

seed: 42

environment:
  timestep: 0.1
  episode_seconds: 60
  traffic_vehicles_min: 10
  traffic_vehicles_max: 30
  max_episode_steps: 600

observation:
  nearest_vehicles: 8
  normalize: true

action:
  type: continuous
  steering_and_throttle: true

training:
  algorithm: PPO
  total_timesteps: 500000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  policy: MlpPolicy

evaluation:
  episodes: 100
  deterministic: true
  seed_start: 10000

output:
  model_dir: models
  results_dir: results
  logs_dir: logs
```

These are starting hyperparameters, not a claim that this is the optimal configuration. The initial 500,000 training steps are a budget for a first serious run; a smoke test with a much smaller budget should come first.

If the observation vector is 58 dimensions, use that dimension when defining the environment's observation space. Do not hardcode 58 into the training script if it can be obtained from the environment.

### What `train_ppo.py` should do

1. Load the experiment configuration.
2. Create the Gymnasium environment.
3. Wrap it with any required monitoring or normalization wrappers.
4. Initialize PPO with the selected hyperparameters.
5. Train the agent.
6. Save the policy checkpoint.
7. Save the observation-normalization statistics, if normalization is used.
8. Record training rewards and episode outcomes.
9. Save the final configuration and random seed alongside the model.

The training entry point should be runnable with a command like:

bash
```
python -m agents.train_ppo
```

The code should resolve the configuration path relative to the project root rather than relying on the terminal's current directory.

### Recommended training schedule

## 01

Smoke test — 5,000 steps

Verify that training runs, the reward is finite, checkpoints save correctly, and the agent can be loaded again.

## 02

Initial training — 100,000 steps

Check whether the agent is learning progress-oriented behavior and whether collision frequency is changing.

## 03

Full baseline run — 500,000 steps

Train using the frozen initial configuration. Save periodic checkpoints so you can compare earlier and later policies.

## 04

Repeatability — 3–5 independent seeds

Repeat the training experiment with different random seeds to estimate sensitivity to initialization and sampling.

The final budget may need to increase if the policy has not converged. Conversely, if the simulator is broken or the agent receives poor observations, additional training steps will not fix the underlying issue.

### What files should training produce?

Your training run should generate artifacts such as:

text
```
models/
    exp01_seed42_final.zip
    exp01_seed42_best.zip

results/
    exp01_config.yaml
    exp01_training_metrics.csv
    exp01_evaluation.csv

logs/
    exp01_seed42/
        progress.csv
        events.out.tfevents...
```

Exact filenames depend on how you implement checkpointing and logging.

Save any observation-normalization statistics as separate artifacts if you use a normalization wrapper. The same statistics must be applied during evaluation and deployment.

## 11. Phase H — Evaluate the agent scientifically

Create `evaluation/metrics.py`, `evaluation/evaluate.py`, and `evaluation/plots.py`.

A rising reward curve alone does not establish that the agent has learned useful driving behavior. You need a separate evaluation procedure.

### Primary metrics

| Metric | Definition |
| --- | --- |
| Destination completion rate | Percentage of episodes reaching the destination |
| Collision rate | Percentage of episodes with at least one collision |
| Collision frequency | Collisions per simulated kilometre |
| Travel time | Time taken to complete the route |
| Near-miss rate | Frequency of predefined unsafe proximity events |
| Waiting time | Time spent stopped or moving below a threshold |
| Progress | Distance toward the destination over the episode |
| Rule violations | Number of defined traffic-rule violations |

Define the collision and near-miss rules before comparing policies. Otherwise, it becomes too easy to change the metrics to favor one agent.

### `evaluation/evaluate.py`

The evaluation script should:

1. Load a saved policy and its normalization statistics.
2. Load the fixed evaluation scenario set.
3. Run the requested number of episodes.
4. Record all metrics per episode.
5. Save the raw results to CSV.
6. Print aggregate statistics and confidence intervals.
7. Save representative failure episodes for visual inspection.

Use 100 evaluation episodes per policy as an initial target, spread across predefined seeds and traffic conditions. More episodes may be needed to estimate rare collisions reliably.

Use a separate seed range from training. Do not select evaluation scenarios based on whether the agent performs well on them.

### `evaluation/plots.py`

Generate these plots:

- Episode reward versus training steps.
- Destination-completion rate versus training steps.
- Collision rate versus training steps.
- Travel time versus training steps.
- PPO versus rule-based baselines.
- Performance versus traffic density.
- Performance versus driver assertiveness.
- Performance versus unseen road configurations.

Plot uncertainty across training seeds where possible, not just the best run.

## 12. Phase I — Run the main experiment

Now conduct the experiment as a controlled comparison.

### Experiment groups

### Group A — Conservative baseline

Control

The agent proceeds only when its predefined gap threshold is satisfied. It prioritizes safety, even at the cost of delay.

### Group B — Assertive rule-based baseline

The agent uses a smaller gap threshold and more aggressive progress decisions, while retaining explicit collision avoidance.

### Group C — PPO policy

Learned

PPO learns from the reward signal and observations. It does not receive a hardcoded gap-acceptance decision, though safety constraints can still be applied independently.

### Experimental controls

Keep the following identical between groups:

- Road geometry and destination.
- Vehicle dimensions and movement model.
- Traffic-generation distributions.
- Evaluation scenario seeds.
- Episode time limits.
- Collision detection.
- Reward-independent evaluation metrics.

The baseline policies and PPO must face comparable traffic conditions. A fair experiment should not give PPO easier NPCs or more favorable starting positions.

### Test conditions

Use at least four evaluation categories:

1. Nominal traffic: similar density and driver characteristics to training.
2. High density: more vehicles and fewer safe gaps.
3. Behavior shift: more assertive, more cautious, or slower-reacting NPCs.
4. Scenario shift: different vehicle spawn positions, routes, or intersection configurations.

Keep a fixed held-out evaluation set for the main comparison. Use a separate development set if you need to tune hyperparameters.

### Main hypothesis test

The primary comparison should be destination-completion rate. Secondary outcomes should include collision frequency, near misses, and travel time.

Report differences between policies with confidence intervals. For a more formal comparison, use paired evaluation scenarios where each policy encounters the same scenario seeds, and analyze episode-level outcomes accordingly.

Do not declare PPO superior simply because its average reward is higher. It must also meet your independently defined safety criteria.

## 13. Phase J — The best follow-up experiment: learning under uncertain driver behavior

Once Experiment 01 works, this is the experiment I would prioritize. It gets closer to the defining feature of your game: drivers infer each other's intentions from movement rather than following perfectly predictable rules.

### Experiment 02 — Robust gap negotiation under behavioral uncertainty

Research question: Does training against a diverse population of drivers help an RL agent negotiate dense traffic more safely than training against one fixed traffic policy?

#### Independent variables

- Traffic density: low, medium and high.
- Driver behavior: cautious, ordinary, assertive and inconsistent.
- Reaction time: short, medium and long.
- Gap acceptance: conservative versus aggressive.
- Observation quality: full structured local observations versus noisy or delayed observations.

Vary these systematically rather than changing all factors simultaneously in your first analysis.

#### Experimental groups

- A — Fixed NPC training: train PPO against one driver configuration.
- B — Randomized NPC training: randomize driver characteristics each episode.
- C — Randomized NPC training with observation noise: also introduce realistic measurement uncertainty and delays.

Train each group with the same PPO implementation and a comparable training budget. Evaluate all three on the same held-out scenarios, including combinations of driver characteristics that were not seen during training.

#### Procedure

1. Freeze the simulator's vehicle dynamics and collision logic.
2. Define distributions for NPC reaction times, gap acceptance and yielding behavior.
3. Train Group A with fixed NPC characteristics.
4. Train Group B with randomized NPC characteristics.
5. Train Group C with randomized characteristics and imperfect observations.
6. Evaluate each policy on identical unseen scenarios.
7. Compare completion rate, collisions, near misses, travel time and performance degradation.
8. Inspect failure episodes to determine whether the agent misjudged a gap, reacted too late, or encountered behavior outside its training distribution.

#### Expected outcome

The hypothesis is that randomized training will improve robustness to driver variation, though it may initially learn more slowly or produce more conservative behavior. This must be verified rather than assumed.

The key result is not simply a higher training reward. It is whether the learned policy maintains acceptable safety and completion performance on traffic scenarios it has never encountered before.

### Why this is a stronger experiment

It tests whether the policy learns a general strategy for negotiating traffic or merely memorizes the behavior of one particular set of NPCs.

That is the difference between a demonstration that works in a controlled simulator and an agent that may eventually be useful in a dynamic game.

## 14. Phase K — Add explicit prediction and uncertainty

After the robust-policy experiment, introduce an explicit prediction module.

Create two additional files:

text
```
traffic_env/
    predictor.py
    uncertainty.py
```

- `predictor.py` estimates the future positions of nearby vehicles over a short horizon using their observed positions, headings and velocities.
- `uncertainty.py` represents uncertainty in those estimates, such as a range of plausible future positions or time-to-conflict values.

For the initial version, use a constant-velocity predictor as a baseline:

$$
\hat p_i(t+\tau)=p_i(t)+v_i(t)\tau
$$

This is only an approximation. Drivers can brake, accelerate or change direction, so the prediction should not be treated as ground truth.

### Compare three observation designs

- Experiment 03A: ego state and current relative positions and velocities.
- Experiment 03B: the same observations plus predicted time-to-collision and closest-approach estimates.
- Experiment 03C: predicted conflict estimates plus uncertainty indicators.

Train the policies using the same scenario distributions and evaluation protocol.

The question is whether these additional features improve performance, especially when driver behavior is inconsistent or observations are delayed.

If they do not, investigate whether the features are redundant, noisy or poorly normalized.

## 15. Phase L — Introduce horn signaling

Only after the driving policy works without honking should you make horn behavior part of the experiment.

Create or extend:

text
```
traffic_env/actions.py
traffic_env/traffic.py
traffic_env/rewards.py
evaluation/metrics.py
```

Model a horn as an observable signal that may affect the behavior of nearby NPCs. For example, some drivers may yield slightly, some may maintain their trajectory, and others may ignore the signal.

Do not assume that a horn always produces the same reaction. The response should depend on the NPC's modeled behavior, distance and context.

Compare:

- A: no horn action.
- B: horn action available, with no explicit communication reward.
- C: horn action available, with a carefully designed coordination objective.

Measure whether signaling improves completion time or safety, and whether it creates excessive noise or causes other vehicles to behave less predictably.

This gives you a testable version of the idea that horns act as a social communication channel rather than merely a sound effect.

## 16. Phase M — Move the learned behavior into Roblox

Once you have a reliable Python policy, connect it to Roblox Studio.

The cleanest design is to keep the trained policy and its input preprocessing consistent between the Python simulator and Roblox.

Roblox client

3D road, player controls, vehicle state and visual effects

Inference service

Receives observations and returns model actions

Trained policy

Frozen model, preprocessing parameters and action mapping

For an initial prototype, Roblox can send structured observations to a local Python service during development and receive the policy's action in return. Keep communication asynchronous and use a bounded action timeout so a slow response cannot leave a vehicle executing unsafe stale commands.

For a public multiplayer game, you will need to address server authority, latency, service availability and abuse prevention. A local Python bridge is a development tool, not a production deployment architecture.

Create a separate Roblox project directory for the Luau scripts and assets; keep it outside the Python package. The interface between the systems should be a documented observation/action schema, not a collection of loosely coordinated scripts.

Before connecting the model, replay recorded scenarios in Roblox and compare the model's observations and actions against the Python simulator. Differences in coordinate conventions, speed units, vehicle geometry and action scaling can invalidate an otherwise well-trained policy.

## 17. Experiment tracking and reproducibility

Create a record for every training run. Do not rely on memory or filenames alone.

Each run should store:

- Experiment ID and hypothesis.
- Git commit or code version.
- Configuration file and random seed.
- Training steps and elapsed time.
- Model checkpoint and normalization statistics.
- Training reward curves.
- Evaluation scenario IDs and results.
- Known failures and observations.

A useful record format in `results/experiment_registry.csv` is:

text
```
run_id,experiment,seed,policy,train_steps,success_rate,collision_rate,mean_time
exp01_a,exp01,42,rule_conservative,0,,
exp01_b,exp01,42,rule_assertive,0,,
exp01_c,exp01,42,ppo,500000,,
```

The blank fields are intentional placeholders for actual evaluation results. Do not fill them with estimates.

Use a tool such as TensorBoard for training curves if desired. Save raw episode-level results as well, because aggregate plots alone can hide failure patterns.

## 18. Final experiment checklist

## Project completion tracker

### 0/20

Environment

[ ]Create the four-way intersection and vehicle dynamics

[ ]Implement rule-based NPC traffic

[ ]Create predefined traffic scenarios

[ ]Render and inspect the simulation

RL interface

[ ]Implement Gymnasium reset() and step()

[ ]Define fixed-length observations and bounded actions

[ ]Implement and log reward components

[ ]Pass environment and safety tests

Baseline and training

[ ]Implement conservative and assertive baselines

[ ]Run a 5,000-step PPO smoke test

[ ]Train the initial PPO model

[ ]Save checkpoints, configuration and logs

Evaluation

[ ]Use held-out scenario seeds

[ ]Measure completion, collisions and travel time

[ ]Compare all policies on identical scenarios

[ ]Repeat training with independent seeds

Research extension

[ ]Test generalization to varied NPC behaviors

[ ]Add motion prediction and uncertainty features

[ ]Evaluate the effect of horn signaling

[ ]Replay the trained policy in Roblox

 Copy checklist

## 19. What counts as a successful first experiment?

Experiment 01 is complete when:

1. The simulator runs reproducibly.
2. PPO can train and its saved policy can be reloaded.
3. The learned policy can complete at least some traffic scenarios.
4. Its performance is compared against both rule-based baselines.
5. Collision and near-miss behavior is evaluated independently of reward.
6. Results can be reproduced from the saved configuration, model and seeds.

The initial experiment does not need to beat every baseline. A policy that fails safely, exposes a weakness in the observations, or reveals a reward-design problem can still produce a useful result.

My recommended order: finish the simulator, validate the Gymnasium interface, establish the rule-based baselines, train PPO, and evaluate it. Then investigate behavioral randomization, explicit prediction, and horn signaling one at a time.

That sequence gives you a manageable beginner implementation while preserving a path toward a genuinely interesting RL research project.
