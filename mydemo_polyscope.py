from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polyscope as ps


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


def show_mesh(mesh: Mesh) -> None:
    ps.init()
    surface = ps.register_surface_mesh(
        "0.off",
        mesh.vertex_array(),
        mesh.triangle_array(),
        smooth_shade=True,
    )
    surface.set_color((0.0, 1.0, 0.0))
    ps.show()


def main() -> None:
    original_model = Path("/Users/huyufan/Downloads/MyDemo/0.off")
    parser = argparse.ArgumentParser(description="Python/Polyscope port of MyDemo")
    parser.add_argument(
        "mesh",
        nargs="?",
        type=Path,
        default=original_model,
        help=f"OFF mesh to display (default: {original_model})",
    )
    args = parser.parse_args()

    mesh = Mesh.load_off(args.mesh)
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
