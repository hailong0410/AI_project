README.md AND .gitignore GUIDE FOR AI FINAL PROJECT
====================================================

This file contains:
1. Complete README.md content
2. Complete requirements.txt content
3. Complete .gitignore content
4. Short instructions for creating these files

----------------------------------------------------
1. README.md CONTENT
----------------------------------------------------

Create this file at:
Project_AI/README.md

Copy the content below into README.md:

# AI Final Project - PPO Reproduction and Multi-Agent Adaptation

## Project Overview

This project is for the Artificial Intelligence final project.
Our team chose **Track 2: PPO**.

The goal of this project is to reproduce Proximal Policy Optimization (PPO) on a standard MuJoCo benchmark and then adapt PPO to a multi-agent coordination environment.

The project has two main phases:

1. **Reproduction Phase**
   Reproduce PPO on `Hopper-v5`.

2. **Adaptation Phase**
   Apply PPO to `MPE2 Simple Spread`, a multi-agent environment, and compare a simple baseline method with an adapted method using a centralized critic.

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

### Reproduction Environment

The reproduction phase uses:

```text
Hopper-v5
```

Environment information:

```text
Observation dimension: 11
Action dimension: 3
Action space: continuous
```

### Adaptation Environment

The adaptation phase uses:

```text
MPE2 Simple Spread
```

Environment information:

```text
Number of agents: 3
Local observation dimension: 18
Action dimension: 5
Action space: continuous
```

Simple Spread is a multi-agent coordination task. The agents need to spread out, cover landmarks, and avoid collisions.

---

## Methods

## 1. PPO Reproduction

For the reproduction phase, we implemented a single-agent PPO algorithm and trained it on `Hopper-v5`.

The PPO implementation includes:

- Actor network
- Critic network
- Rollout buffer
- Advantage and return computation
- PPO clipped objective
- Value loss
- Entropy tracking
- Reward logging and plotting

Main files:

```text
models.py
buffer.py
ppo_agent.py
train_single.py
run_reproduction_seeds.py
```

---

## 2. Simple Spread Baseline

For the adaptation environment, we first implemented a simple baseline method.

The baseline method is a **naive shared-policy PPO**.

In this baseline:

- All agents share the same policy.
- Each agent only uses its own local observation.
- The critic also uses local observation.
- There is no centralized critic.

Main files:

```text
train_simple_spread_baseline.py
run_simple_spread_baseline_seeds.py
```

---

## 3. Adapted Method: Shared Actor + Centralized Critic

The adapted method uses:

```text
Shared actor + centralized critic
```

In this method:

- The actor uses each agent's local observation.
- The critic uses the joint observation of all three agents.
- Local observation dimension = 18.
- Joint observation dimension = 18 x 3 = 54.

The purpose of the centralized critic is to give the value function access to global multi-agent information. This is intended to help with coordination and credit assignment in the multi-agent environment.

Main files:

```text
ma_ppo_agent.py
train_simple_spread_adapted.py
run_simple_spread_adapted_seeds.py
```

---

## Random Seeds

We used three random seeds:

```text
42, 123, 999
```

These seeds were used because reinforcement learning results can vary due to random initialization, stochastic action sampling, and environment randomness.

The exact seed numbers are not special. They are simply common and easy to track. The important point is that the same seeds were used across experiments to make comparisons more consistent.

---

## Current Results

### Simple Spread Baseline vs Adapted Method

Current results across three seeds:

```text
Baseline mean: -109.30
Adapted mean: -121.72
```

Since Simple Spread rewards are negative, values closer to zero are better.
Under the current training setup, the baseline performs better than the adapted method.

This does not mean the adaptation is invalid. The centralized critic is still a meaningful algorithmic change because it gives the critic access to joint multi-agent information. However, the current result suggests that the adapted method may require longer training, more hyperparameter tuning, or improved action and reward handling.

---

## How to Run the Code

Before running the code, activate the virtual environment.

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, run files directly using:

```powershell
.\.venv\Scripts\python.exe <filename.py>
```

---

## Install Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

If using the virtual environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Run Reproduction Experiments

Move into the `src` folder:

```bash
cd src
```

Run PPO reproduction on `Hopper-v5`:

```bash
python run_reproduction_seeds.py
```

Summarize reproduction results:

```bash
python summarize_reproduction.py
```

Plot reproduction comparison:

```bash
python plot_reproduction_comparison.py
```

---

## Run Simple Spread Baseline

```bash
python run_simple_spread_baseline_seeds.py
```

---

## Run Adapted Simple Spread Method

```bash
python run_simple_spread_adapted_seeds.py
```

---

## Compare Baseline and Adapted Results

```bash
python compare_simple_spread_results.py
```

---

## Generate Experiment Summary

```bash
python write_experiment_summary.py
```

This generates:

```text
results/logs/experiment_summary.txt
```

---

## Output Files

Experiment outputs are saved in:

```text
results/logs/
results/plots/
```

The logs include:

- reward histories
- experiment summaries
- baseline vs adapted comparison
- final experiment summary

The plots include:

- Hopper reward curves
- Hopper seed comparison
- Simple Spread baseline curves
- Simple Spread adapted curves
- baseline vs adapted comparison plots

---

## Main Requirements

The project uses the following main packages:

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
pettingzoo
mpe2
```

---

## Notes

The main coding pipeline is complete.

Completed work:

- PPO reproduction on `Hopper-v5`
- Three-seed reproduction experiment
- Simple Spread environment setup
- Naive shared-policy PPO baseline
- Adapted PPO with centralized critic
- Three-seed baseline and adapted experiments
- Baseline vs adapted comparison
- Experiment summary generation

Next steps:

- Write the final report
- Prepare presentation slides
- Discuss limitations and possible improvements

----------------------------------------------------
2. requirements.txt CONTENT
----------------------------------------------------

Create this file at:
Project_AI/requirements.txt

Copy the content below into requirements.txt:

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

----------------------------------------------------
3. .gitignore CONTENT
----------------------------------------------------

Create this file at:
Project_AI/.gitignore

Copy the content below into .gitignore:

# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/
env/

# VS Code settings
.vscode/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# System files
.DS_Store
Thumbs.db

# Logs
*.log

# Model checkpoints
*.pt
*.pth
*.ckpt

# Large temporary files
*.tmp
*.temp

# Python build files
build/
dist/
*.egg-info/

# Environment files
.env

# Optional: ignore large videos or recordings
*.mp4
*.avi
*.mov



#   A I _ p r o j e c t  
 