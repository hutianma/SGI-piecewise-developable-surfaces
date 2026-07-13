# Piecewise Developable Surfaces

Python/Polyscope port of the original Coin3D `MyDemo` mesh viewer. It supports
OFF and OBJ triangle meshes.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python mydemo_polyscope.py /path/to/mesh.off
python mydemo_polyscope.py /path/to/mesh.obj
python mydemo_polyscope.py /path/to/mesh.obj --seed 100 --epsilon 0.01
python mydemo_polyscope.py /path/to/mesh.obj --all-patches --epsilon 0.01
```

If no mesh path is supplied, the script uses the original `MyDemo/0.off` path
on the development machine.

The viewer displays both integrated Gaussian curvature (angle defect) and
pointwise Gaussian curvature (angle defect divided by vertex dual area). The
integrated quantity is enabled by default because it is scale-independent and
matches the Geometry Collective demo. The `coolwarm` color map shows zero as
white, positive curvature as red, and negative curvature as blue.

Area-weighted vertex normals are shown as short blue arrows at every vertex in
the `Vertex normals` vector quantity, similar to the Geometry Collective
discrete-curvatures-and-normals demo. This layer can be toggled from the
Polyscope sidebar.

## Flat-patch flood fill

Pass `--seed` to grow the connected component of vertices whose absolute angle
defect is at most `--epsilon`. The resulting patch is yellow, the seed is
magenta, and rejected/outside vertices are gray. A curved seed produces an
empty patch. The default epsilon is 0.01 radians (about 0.57 degrees).

Use `--all-patches` to traverse all near-zero-curvature vertices and assign a
different color to every connected component. Components smaller than 10
vertices are hidden by default; change this with `--min-patch-size`. The terminal
reports both the total number of connected flat patches and the number displayed.
