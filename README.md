# Piecewise Developable Surfaces

Python/Polyscope port of the original Coin3D `MyDemo` mesh viewer.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python mydemo_polyscope.py /path/to/mesh.off
```

If no mesh path is supplied, the script uses the original `MyDemo/0.off` path
on the development machine.
