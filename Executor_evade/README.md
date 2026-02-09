# Executor Evade

## Overview
Meta-command executor for evasion. It converts meta-commands into control actions in the Harfang-based air combat environment.

## Key Components
- `algorithms/`: PPO and MAPPO implementations.
- `envs/HarfangSim/`: Air combat environment and utilities.
- `runner/`: Training/evaluation runners.
- `scripts/train/`: Training entry points.
- `config.py`: Hyperparameter definitions.

## Quick Start
```bash
python Executor_evade/scripts/train/train_harfang.py
```

## Dependencies
- Python 3
- PyTorch
- NumPy
- Gym
- Matplotlib (optional, for plotting)
- Harfang3D Dog-Fight Sandbox (simulation backend)

## Weights
Pretrained weights are under `weights/Executor_evade/`.

## Notes
- The default script calls `runner.render2()` for visualization; switch to `runner.run()` for training.
- Simulation backend is based on the Harfang3D Dog-Fight Sandbox.
