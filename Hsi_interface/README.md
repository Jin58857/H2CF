# HSI Interface

## Overview
This module provides the top-level human-swarm interaction GUI. It captures human intent/preferences and exchanges data with the backend over TCP.

## Key Components
- `design_v16.py`: Main GUI entry point.
- `design_v16_write.py`: GUI variant.
- `design_v15_same.py`: GUI variant.
- `models/`: 3D aircraft models.
- `icons/`: UI assets.

## Quick Start
```bash
python Hsi_interface/design_v16.py
```

## Dependencies
- Python 3
- PyQt6
- PyOpenGL
- pyqtgraph
- trimesh
- Pillow
- NumPy

## Weights
Not applicable.

## Notes
- Default backend host/port: `127.0.0.1:9999`.
- The backend is expected to stream line-delimited JSON messages.
