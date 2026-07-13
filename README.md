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
