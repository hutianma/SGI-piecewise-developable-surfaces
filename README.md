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
python mydemo_polyscope.py /path/to/mesh.obj --seed 100 --epsilon 0.01 --leakage-report leakage.csv
python mydemo_polyscope.py /path/to/mesh.obj --seed-face 100 --epsilon 0.01
python mydemo_polyscope.py /path/to/mesh.obj --all-patches --epsilon 0.01
python mydemo_polyscope.py /path/to/mesh.obj --seed 100 --epsilon-sweep 0.005 0.01 0.02 0.05 0.1 --epsilon-sweep-report sweep.csv
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
whose absolute integrated Gaussian curvature (angle defect) is at most
`--epsilon`. The resulting patch is
yellow, the seed is magenta, and rejected/outside vertices are gray. The default
epsilon is 0.01 radians. Angle defect is invariant to uniform model scaling,
but its per-vertex value depends on mesh sampling density.

To inspect a gray vertex, enable `Gray/outside vertices (enable to inspect
IDs)` under Point Clouds and click a point. Its `Original vertex index` and
angle defect are attached as point quantities. The terminal also separates
vertices rejected by the angle-defect threshold from near-zero vertices that
belong to a different connected component.

Use `--gray-report gray_vertices.csv` to save every gray vertex ID, its angle
defect, pointwise Gaussian curvature, and rejection reason. The terminal also
prints the complete gray vertex ID list.

Use `--epsilon-sweep` to compare several angle-defect thresholds without
modifying the mesh or opening the viewer. The report includes global near-zero
coverage, the seed-connected patch size, gray counts split by curvature sign,
and connected-component counts.

## Normal-displacement experiment

Run the first bounded normal-displacement experiment without overwriting the
input mesh:

```bash
python optimize_angle_defect.py test-models/data/fandisk.obj \
  --epsilon 0.1 \
  --output-dir results/fandisk/fandisk_comparison_package/methods/normal_joint_one_ring_shape_constrained_eps_0.1
```

The optimizer fixes the initial vertices with `|angle defect| > 1`, jointly
moves moderate-defect vertices and their one-rings along frozen normals, and
accepts only globally energy-decreasing, flip-free updates. It writes a new
OBJ plus iteration, singular-vertex, gray-vertex, and displacement CSV files.
By default, a weak normalized zero-curvature term with weight `0.001` helps move
vertices away from the classification threshold; change it with
`--zero-curvature-weight` or set it to zero to reproduce the threshold-only
baseline.
Use `--movable-rings 2` to test a wider coordinated normal-displacement
region while retaining one additional ring for curvature monitoring.

## Cone/sphere ray-search experiment

The ray-search experiment uses the cone-of-rays construction as a direction
generator, not as a claim that Shape Diameter Function is being computed. For
each moderate gray vertex it samples inward/outward cone directions, brackets
angle-defect sign changes, and applies binary search when a root is bracketed:

```bash
python optimize_angle_defect_rays.py test-models/data/fandisk.obj \
  --direction-mode cone \
  --rays 30 \
  --search-fraction 0.1 \
  --epsilon 0.1 \
  --output-dir results/fandisk/fandisk_comparison_package/methods/double_cone_30_rays_geometry_safety_constrained_eps_0.1
```

Use the following command for full Fibonacci-sphere direction sampling:

```bash
python optimize_angle_defect_rays.py test-models/data/fandisk.obj \
  --direction-mode sphere \
  --rays 30 \
  --search-fraction 0.1 \
  --epsilon 0.1 \
  --output-dir results/fandisk/fandisk_comparison_package/methods/virtual_sphere_30_rays_geometry_safety_constrained_eps_0.1
```

For a vertex patch grown with `--seed`, the viewer also marks candidate leakage
edges in red. A candidate is an edge accepted by the zero-curvature BFS whose
two incident faces differ by at least `--leakage-angle` degrees (20 by default).
Use `--leakage-report` to save their edge/vertex indices, dihedral angles, and
endpoint angle defects (in radians) as CSV. Connected candidate edges are grouped into seam
components. Red lines show the seams, blue points show the first vertices
reached when the BFS tree directly crosses a seam, and cyan points show open
seam endpoints around which the vertex BFS can leak without crossing a red
edge. These are diagnostics:
they are not automatically
removed from the patch because Gaussian curvature alone cannot distinguish two
adjacent developable pieces.

Use `--seed-face` to grow a triangle patch. A candidate face is accepted only
when every vertex it newly moves into the patch interior has finite angle
defect with absolute value at most `--epsilon`. High-curvature
vertices may remain on patch boundaries or seams, but not in patch interiors.

Use `--all-patches` to greedily partition all triangles into edge-connected
developable face patches. The next seed is the flattest unassigned face.
Components smaller than 10 faces are hidden by default; change this with
`--min-patch-size`. The terminal reports total and displayed patch counts.
