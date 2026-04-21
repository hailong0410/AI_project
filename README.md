# AI Final Project - PPO Reproduction and Multi-Agent Adaptation

## Project Overview

This project is for the Artificial Intelligence final project. Our team chose **Track 2: PPO**.

The goal of this project is to reproduce Proximal Policy Optimization (PPO) on a standard MuJoCo benchmark and then adapt PPO to a multi-agent coordination environment.

The project has two main phases:

1. **Reproduction Phase:** Train PPO on `Hopper-v5`.
2. **Adaptation Phase:** Apply PPO to `MPE2 Simple Spread` and compare a baseline method with an adapted method using a centralized critic.

---

## Project Structure

```text
Project_AI/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
├── notes/
├── report/
├── results/
│   ├── logs/
│   └── plots/
└── src/
    ├── models.py
    ├── buffer.py
    ├── ppo_agent.py
    ├── ma_ppo_agent.py
    ├── utils.py
    ├── train_single.py
    ├── run_reproduction_seeds.py
    ├── summarize_reproduction.py
    ├── plot_reproduction_comparison.py
    ├── train_simple_spread_baseline.py
    ├── run_simple_spread_baseline_seeds.py
    ├── train_simple_spread_adapted.py
    ├── run_simple_spread_adapted_seeds.py
    ├── compare_simple_spread_results.py
    └── write_experiment_summary.py
```

---

## Folder Description

- `src/`: contains all source code for training, testing, and evaluation.
- `results/logs/`: contains saved JSON logs and experiment summaries.
- `results/plots/`: contains saved training plots and comparison plots.
- `report/`: contains report drafts or final report files.
- `configs/`: contains experiment configuration notes or parameter settings.
- `notes/`: contains project notes and planning documents.

---

## Environments

### Hopper-v5

Used for PPO reproduction.

```text
Observation dimension: 11
Action dimension: 3
Action space: continuous
```

### MPE2 Simple Spread

Used for multi-agent adaptation.

```text
Number of agents: 3
Local observation dimension: 18
Action dimension: 5
Joint observation dimension: 54
```

Simple Spread is a multi-agent coordination task. The agents need to spread out, cover landmarks, and avoid collisions.

---

## Methods

### PPO Reproduction

We implemented a single-agent PPO algorithm and trained it on `Hopper-v5`.

The implementation includes:

- Actor network
- Critic network
- Rollout buffer
- Advantage and return computation
- PPO clipped objective
- Reward logging and plotting

### Simple Spread Baseline

The baseline method is a naive shared-policy PPO.

- All agents share the same policy.
- Each agent only uses its own local observation.
- The critic also uses local observation.
- There is no centralized critic.

### Adapted Method

The adapted method uses a shared actor with a centralized critic.

- The actor uses local observations.
- The critic uses the joint observation of all three agents.
- Local observation dimension = 18
- Joint observation dimension = 18 x 3 = 54

The purpose of the centralized critic is to provide global multi-agent information for value estimation.

---

## Random Seeds

The experiments use three random seeds:

```text
42, 123, 999
```

Using multiple seeds helps reduce the effect of randomness in reinforcement learning.

---

## Current Results

Simple Spread results across three seeds:

```text
Baseline mean: -109.30
Adapted mean: -121.72
```

Since rewards are negative, values closer to zero are better. In the current setting, the baseline performs better than the adapted method.

The centralized critic is still a meaningful adaptation, but it may require longer training, better hyperparameter tuning, or improved action and reward handling.

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Move into the source folder:

```bash
cd src
```

Run PPO reproduction:

```bash
python run_reproduction_seeds.py
```

Run Simple Spread baseline:

```bash
python run_simple_spread_baseline_seeds.py
```

Run adapted Simple Spread method:

```bash
python run_simple_spread_adapted_seeds.py
```

Compare baseline and adapted results:

```bash
python compare_simple_spread_results.py
```

Generate experiment summary:

```bash
python write_experiment_summary.py
```

---

## Outputs

Experiment outputs are saved in:

```text
results/logs/
results/plots/
```

The project generates reward logs, experiment summaries, and training/comparison plots.

---

## Main Requirements

```text
torch
numpy
matplotlib
gymnasium
mujoco
imageio
imageio-ffmpeg
pillow
pygame
pettingzoo==1.25.0
mpe2
```

---

## Status

The main code pipeline is complete.

Completed work:

- PPO reproduction on `Hopper-v5`
- Three-seed reproduction experiment
- Simple Spread environment setup
- Naive shared-policy PPO baseline
- Adapted PPO with centralized critic
- Three-seed baseline and adapted experiments
- Baseline vs adapted comparison
- Experiment summary generation
