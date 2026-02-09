# Executor Attack

## Overview
This module implements the bottom-level meta-command executor for attack. It converts meta-commands into control actions in the Harfang-based air combat environment.

## Key Components
- `algorithms/`: PPO and MAPPO implementations.
- `envs/HarfangSim/`: Air combat environment, tasks, and utilities.
- `runner/`: Training/evaluation runners.
- `scripts/train/`: Training entry points.
- `config.py`: Hyperparameter definitions.

## Quick Start
```bash
python Executor_attack/scripts/train/train_harfang.py
```

## Dependencies
- Python 3
- PyTorch
- NumPy
- Gym
- Matplotlib (optional, for plotting)
- Harfang3D Dog-Fight Sandbox (simulation backend)

## Weights
Pretrained weights are under `weights/Executor_attack/`.

## Notes
Simulation backend is based on the Harfang3D Dog-Fight Sandbox.
