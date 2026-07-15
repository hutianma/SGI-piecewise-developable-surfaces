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
python mydemo_polyscope.py /path/to/mesh.obj --seed-face 100 --epsilon 0.01
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

Pass `--seed` to run the original vertex diagnostic: it grows through vertices
whose absolute pointwise Gaussian curvature (`angle defect / vertex area`) is
at most `--epsilon`. The resulting patch is
yellow, the seed is magenta, and rejected/outside vertices are gray. The default
epsilon is 0.01 in inverse model-length squared. The previous angle-defect
stopping condition remains commented out in the source for comparison.

Use `--seed-face` to grow a triangle patch. A candidate face is accepted only
when every vertex it newly moves into the patch interior has finite pointwise
Gaussian curvature with absolute value at most `--epsilon`. High-curvature
vertices may remain on patch boundaries or seams, but not in patch interiors.

Use `--all-patches` to greedily partition all triangles into edge-connected
developable face patches. The next seed is the flattest unassigned face.
Components smaller than 10 faces are hidden by default; change this with
`--min-patch-size`. The terminal reports total and displayed patch counts.
