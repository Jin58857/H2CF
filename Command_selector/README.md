# Command Selector

## Overview
This module implements the mid-level meta-command selector. It selects high-level commands from environment observations and human preference inputs.

## Key Components
- `algorithms/`: PPO and MAPPO implementations.
- `envs/HighEnvSim/`: High-level command environments and human control socket.
- `runner/`: Training/evaluation runners.
- `scripts/train/`: Training entry points.
- `config.py` and `low_config.py`: Hyperparameter definitions.

## Quick Start
```bash
python Command_selector/scripts/train/train_harfang.py
```

## Dependencies
- Python 3
- PyTorch
- NumPy
- Gym
- Matplotlib (optional, for plotting)
- Harfang3D Dog-Fight Sandbox (simulation backend)

## Weights
Pretrained weights are under `weights/Command_selector/`.

## Notes
Simulation backend is based on the Harfang3D Dog-Fight Sandbox.
