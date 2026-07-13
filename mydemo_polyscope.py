from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polyscope as ps
import trimesh


@dataclass
class Vertex:
    index: int
    coordinates: tuple[float, float, float]
    vertex_indices: list[int] = field(default_factory=list)
    triangle_indices: list[int] = field(default_factory=list)
    edge_indices: list[int] = field(default_factory=list)


@dataclass
class Edge:
    index: int
    vertex_1: int
    vertex_2: int


@dataclass
class Triangle:
    index: int
    vertex_1: int
    vertex_2: int
    vertex_3: int


class Mesh:
    def __init__(self) -> None:
        self.vertices: list[Vertex] = []
        self.edges: list[Edge] = []
        self.triangles: list[Triangle] = []

    @classmethod
    def load(cls, filename: str | Path) -> Mesh:
        filename = Path(filename).expanduser()
        suffix = filename.suffix.lower()
        if suffix == ".off":
            return cls.load_off(filename)
        if suffix == ".obj":
            return cls.load_obj(filename)
        raise ValueError(
            f"Unsupported mesh format '{suffix}'. Expected an .off or .obj file."
        )

    @classmethod
    def load_off(cls, filename: str | Path) -> Mesh:
        mesh = cls()
        tokens = cls._off_tokens(Path(filename))

        try:
            if next(tokens) != "OFF":
                raise ValueError("Not an OFF file")

            vertex_count = int(next(tokens))
            face_count = int(next(tokens))
            next(tokens)  # Edge count is optional metadata and is not needed here.

            for _ in range(vertex_count):
                mesh.add_vertex(
                    float(next(tokens)),
                    float(next(tokens)),
                    float(next(tokens)),
                )

            for face_index in range(face_count):
                face_size = int(next(tokens))
                indices = [int(next(tokens)) for _ in range(face_size)]
                if face_size != 3:
                    raise ValueError(
                        f"Face {face_index} has {face_size} vertices; "
                        "this demo expects a triangulated OFF mesh"
                    )
                mesh.add_triangle(*indices)
        except StopIteration as error:
            raise ValueError(f"Incomplete OFF file: {filename}") from error

        return mesh

    @classmethod
    def load_obj(cls, filename: str | Path) -> Mesh:
        loaded = trimesh.load_mesh(
            Path(filename),
            process=True,
            maintain_order=True,
        )

        # An OBJ may contain several named objects. Combine them into one mesh,
        # matching the single-mesh data structure used by the original demo.
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError(f"OBJ file contains no mesh geometry: {filename}")
            loaded = loaded.to_mesh()

        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Could not read a triangle mesh from: {filename}")

        mesh = cls()
        for x, y, z in np.asarray(loaded.vertices):
            mesh.add_vertex(float(x), float(y), float(z))
        for face in np.asarray(loaded.faces):
            if len(face) != 3:
                raise ValueError("OBJ contains a face which could not be triangulated")
            mesh.add_triangle(int(face[0]), int(face[1]), int(face[2]))
        return mesh

    @staticmethod
    def _off_tokens(filename: Path):
        with filename.open("r", encoding="utf-8") as off_file:
            for line in off_file:
                content = line.split("#", 1)[0]
                yield from content.split()

    def add_vertex(self, x: float, y: float, z: float) -> None:
        self.vertices.append(Vertex(len(self.vertices), (x, y, z)))

    def add_triangle(self, v1: int, v2: int, v3: int) -> None:
        vertex_count = len(self.vertices)
        for index in (v1, v2, v3):
            if not 0 <= index < vertex_count:
                raise ValueError(f"Vertex index {index} is out of range")

        triangle_index = len(self.triangles)
        self.triangles.append(Triangle(triangle_index, v1, v2, v3))

        for vertex_index in (v1, v2, v3):
            self.vertices[vertex_index].triangle_indices.append(triangle_index)

        self._connect_vertices(v1, v2)
        self._connect_vertices(v1, v3)
        self._connect_vertices(v2, v3)

    def _connect_vertices(self, v1: int, v2: int) -> None:
        if v2 in self.vertices[v1].vertex_indices:
            return

        self.vertices[v1].vertex_indices.append(v2)
        self.vertices[v2].vertex_indices.append(v1)

        edge_index = len(self.edges)
        self.edges.append(Edge(edge_index, v1, v2))
        self.vertices[v1].edge_indices.append(edge_index)
        self.vertices[v2].edge_indices.append(edge_index)

    def vertex_array(self) -> np.ndarray:
        return np.asarray([vertex.coordinates for vertex in self.vertices])

    def triangle_array(self) -> np.ndarray:
        return np.asarray(
            [
                (triangle.vertex_1, triangle.vertex_2, triangle.vertex_3)
                for triangle in self.triangles
            ],
            dtype=np.int32,
        )

    def gaussian_curvature(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute vertex Gaussian curvature using angle defect / barycentric area."""
        positions = self.vertex_array()
        faces = self.triangle_array()
        vertex_count = len(positions)
        angle_sums = np.zeros(vertex_count)
        dual_areas = np.zeros(vertex_count)
        edge_face_counts: dict[tuple[int, int], int] = {}

        for face in faces:
            i, j, k = (int(face[0]), int(face[1]), int(face[2]))
            p_i, p_j, p_k = positions[[i, j, k]]

            twice_area = np.linalg.norm(np.cross(p_j - p_i, p_k - p_i))
            if twice_area <= np.finfo(float).eps:
                continue

            face_area = 0.5 * twice_area
            dual_areas[[i, j, k]] += face_area / 3.0

            corners = ((i, p_j - p_i, p_k - p_i),
                       (j, p_k - p_j, p_i - p_j),
                       (k, p_i - p_k, p_j - p_k))
            for vertex_index, u, v in corners:
                # atan2 is more stable than arccos near angles 0 and pi.
                angle_sums[vertex_index] += np.arctan2(
                    np.linalg.norm(np.cross(u, v)), np.dot(u, v)
                )

            for a, b in ((i, j), (j, k), (k, i)):
                edge = (min(a, b), max(a, b))
                edge_face_counts[edge] = edge_face_counts.get(edge, 0) + 1

        boundary = np.zeros(vertex_count, dtype=bool)
        for (i, j), face_count in edge_face_counts.items():
            if face_count == 1:
                boundary[[i, j]] = True

        active = dual_areas > np.finfo(float).eps
        target_angles = np.where(boundary, np.pi, 2.0 * np.pi)
        angle_defects = target_angles - angle_sums
        angle_defects[~active] = 0.0
        curvature = np.divide(
            angle_defects,
            dual_areas,
            out=np.zeros_like(angle_defects),
            where=dual_areas > np.finfo(float).eps,
        )
        return curvature, angle_defects, boundary

    def vertex_normals(self) -> np.ndarray:
        """Compute area-weighted unit normals at mesh vertices."""
        positions = self.vertex_array()
        normals = np.zeros_like(positions, dtype=float)

        for i, j, k in self.triangle_array():
            i, j, k = int(i), int(j), int(k)
            face_normal = np.cross(
                positions[j] - positions[i], positions[k] - positions[i]
            )
            # The cross product has magnitude 2A, so summing it gives an
            # area-weighted average of the incident face normals.
            normals[[i, j, k]] += face_normal

        lengths = np.linalg.norm(normals, axis=1)
        nonzero = lengths > np.finfo(float).eps
        normals[nonzero] /= lengths[nonzero, np.newaxis]
        return normals


def show_mesh(mesh: Mesh) -> None:
    curvature, _, boundary = mesh.gaussian_curvature()
    normals = mesh.vertex_normals()
    finite_abs_curvature = np.abs(curvature[np.isfinite(curvature)])
    color_limit = (
        float(np.percentile(finite_abs_curvature, 98))
        if finite_abs_curvature.size
        else 1.0
    )
    if color_limit <= np.finfo(float).eps:
        color_limit = 1.0

    ps.init()
    surface = ps.register_surface_mesh(
        "0.off",
        mesh.vertex_array(),
        mesh.triangle_array(),
        smooth_shade=True,
    )
    surface.set_color((0.0, 1.0, 0.0))
    surface.add_scalar_quantity(
        "Gaussian curvature K",
        curvature,
        defined_on="vertices",
        datatype="symmetric",
        cmap="coolwarm",
        vminmax=(-color_limit, color_limit),
        enabled=True,
        onscreen_colorbar_enabled=True,
    )
    surface.add_vector_quantity(
        "Vertex normals",
        normals,
        defined_on="vertices",
        vectortype="standard",
        color=(0.0, 0.0, 1.0),
        length=0.008,
        radius=0.00012,
        enabled=True,
    )

    interior = curvature[~boundary]
    if interior.size:
        print(
            "Interior Gaussian curvature: "
            f"min={interior.min():.6g}, max={interior.max():.6g}, "
            f"mean |K|={np.mean(np.abs(interior)):.6g}"
        )
    print(
        f"Boundary vertices: {np.count_nonzero(boundary)}; "
        f"color range: [{-color_limit:.6g}, {color_limit:.6g}]"
    )
    ps.show()


def main() -> None:
    original_model = Path("/Users/huyufan/Downloads/MyDemo/0.off")
    parser = argparse.ArgumentParser(
        description="Python/Polyscope port of MyDemo (supports OFF and OBJ)"
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=original_model,
        help=f"OFF or OBJ mesh to display (default: {original_model})",
    )
    args = parser.parse_args()

    mesh = Mesh.load(args.mesh)
    print(
        f"Loaded {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles, and {len(mesh.edges)} edges"
    )

    vertex_index = 4
    if vertex_index < len(mesh.vertices):
        vertex = mesh.vertices[vertex_index]
        print(f"Vertex {vertex_index} 1-ring neighbors: {vertex.vertex_indices}")

        neighbors_from_edges = []
        for edge_index in vertex.edge_indices:
            edge = mesh.edges[edge_index]
            neighbors_from_edges.append(
                edge.vertex_2 if edge.vertex_1 == vertex_index else edge.vertex_1
            )
        print(f"Vertex {vertex_index} neighbors from edges: {neighbors_from_edges}")

    show_mesh(mesh)


if __name__ == "__main__":
    main()
