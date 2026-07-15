from __future__ import annotations

import argparse
import colorsys
from collections import deque
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
    triangle_indices: list[int] = field(default_factory=list)


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
        self._edge_lookup: dict[tuple[int, int], int] = {}

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

        self._connect_vertices(v1, v2, triangle_index)
        self._connect_vertices(v1, v3, triangle_index)
        self._connect_vertices(v2, v3, triangle_index)

    def _connect_vertices(self, v1: int, v2: int, triangle_index: int) -> None:
        key = (min(v1, v2), max(v1, v2))
        existing_edge = self._edge_lookup.get(key)
        if existing_edge is not None:
            self.edges[existing_edge].triangle_indices.append(triangle_index)
            return

        self.vertices[v1].vertex_indices.append(v2)
        self.vertices[v2].vertex_indices.append(v1)

        edge_index = len(self.edges)
        self.edges.append(Edge(edge_index, v1, v2, [triangle_index]))
        self._edge_lookup[key] = edge_index
        self.vertices[v1].edge_indices.append(edge_index)
        self.vertices[v2].edge_indices.append(edge_index)

    def face_neighbors(self) -> list[set[int]]:
        """Return edge-adjacent triangle indices for every triangle."""
        neighbors = [set() for _ in self.triangles]
        for edge in self.edges:
            for face_index in edge.triangle_indices:
                neighbors[face_index].update(
                    other
                    for other in edge.triangle_indices
                    if other != face_index
                )
        return neighbors

    def boundary_vertices(self) -> set[int]:
        boundary: set[int] = set()
        for edge in self.edges:
            if len(edge.triangle_indices) == 1:
                boundary.update((edge.vertex_1, edge.vertex_2))
        return boundary

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


def grow_flat_patch(
    mesh: Mesh,
    pointwise_curvature: np.ndarray,
    seed: int,
    epsilon: float,
) -> set[int]:
    """Flood-fill the zero-curvature component containing ``seed``."""
    if not 0 <= seed < len(mesh.vertices):
        raise ValueError(
            f"Seed {seed} is out of range; expected 0..{len(mesh.vertices) - 1}"
        )
    if epsilon < 0:
        raise ValueError("Epsilon must be nonnegative")
    # Old stopping condition (integrated curvature / angle defect):
    # if not np.isfinite(angle_defects[seed]) or abs(angle_defects[seed]) > epsilon:
    if (
        not np.isfinite(pointwise_curvature[seed])
        or abs(pointwise_curvature[seed]) > epsilon
    ):
        return set()

    patch = {seed}
    visited = {seed}
    queue = deque([seed])

    while queue:
        current = queue.popleft()
        for neighbor in mesh.vertices[current].vertex_indices:
            if neighbor in visited:
                continue
            visited.add(neighbor)

            # Old stopping condition:
            # if (np.isfinite(angle_defects[neighbor])
            #         and abs(angle_defects[neighbor]) <= epsilon):
            if (
                np.isfinite(pointwise_curvature[neighbor])
                and abs(pointwise_curvature[neighbor]) <= epsilon
            ):
                patch.add(neighbor)
                queue.append(neighbor)

    return patch


def scale_normalized_curvature(mesh: Mesh, curvature: np.ndarray) -> np.ndarray:
    """Make pointwise Gaussian curvature invariant to uniform model scaling."""
    positions = mesh.vertex_array()
    diagonal = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    if diagonal <= np.finfo(float).eps:
        return np.full_like(curvature, np.nan)
    return curvature * diagonal * diagonal


def find_all_flat_patches(
    mesh: Mesh,
    pointwise_curvature: np.ndarray,
    epsilon: float,
    min_size: int = 1,
) -> list[set[int]]:
    """Return every connected component of near-zero-curvature vertices."""
    if epsilon < 0:
        raise ValueError("Epsilon must be nonnegative")
    if min_size < 1:
        raise ValueError("Minimum patch size must be at least 1")

    assigned: set[int] = set()
    patches: list[set[int]] = []
    for vertex_index in range(len(mesh.vertices)):
        if (
            vertex_index in assigned
            # Old conditions:
            # or not np.isfinite(angle_defects[vertex_index])
            # or abs(angle_defects[vertex_index]) > epsilon
            or not np.isfinite(pointwise_curvature[vertex_index])
            or abs(pointwise_curvature[vertex_index]) > epsilon
        ):
            continue

        patch = grow_flat_patch(mesh, pointwise_curvature, vertex_index, epsilon)
        assigned.update(patch)
        if len(patch) >= min_size:
            patches.append(patch)

    patches.sort(key=len, reverse=True)
    return patches


def grow_developable_face_patch(
    mesh: Mesh,
    pointwise_curvature: np.ndarray,
    seed_face: int,
    epsilon: float,
    blocked_faces: set[int] | None = None,
) -> set[int]:
    """Grow faces without enclosing a high-curvature interior vertex."""
    if not 0 <= seed_face < len(mesh.triangles):
        raise ValueError(
            f"Seed face {seed_face} is out of range; "
            f"expected 0..{len(mesh.triangles) - 1}"
        )
    if epsilon < 0:
        raise ValueError("Epsilon must be nonnegative")

    blocked = blocked_faces or set()
    if seed_face in blocked:
        return set()

    face_neighbors = mesh.face_neighbors()
    mesh_boundary_vertices = mesh.boundary_vertices()
    patch_faces = {seed_face}
    interior_vertices: set[int] = set()
    rejected_faces: set[int] = set()
    queue = deque(face_neighbors[seed_face] - blocked)
    queued_faces = set(queue)

    while queue:
        candidate = queue.popleft()
        queued_faces.discard(candidate)
        if (
            candidate in patch_faces
            or candidate in rejected_faces
            or candidate in blocked
        ):
            continue

        trial_patch = patch_faces | {candidate}
        triangle = mesh.triangles[candidate]
        candidate_vertices = (
            triangle.vertex_1,
            triangle.vertex_2,
            triangle.vertex_3,
        )
        newly_created_interior = {
            vertex_index
            for vertex_index in candidate_vertices
            if vertex_index not in interior_vertices
            and vertex_index not in mesh_boundary_vertices
            and set(mesh.vertices[vertex_index].triangle_indices).issubset(
                trial_patch
            )
        }

        # Old stopping condition used integrated curvature (angle defect):
        # acceptable = all(
        #     np.isfinite(angle_defects[vertex_index])
        #     and abs(angle_defects[vertex_index]) <= epsilon
        #     for vertex_index in newly_created_interior
        # )
        acceptable = all(
            np.isfinite(pointwise_curvature[vertex_index])
            and abs(pointwise_curvature[vertex_index]) <= epsilon
            for vertex_index in newly_created_interior
        )
        if not acceptable:
            rejected_faces.add(candidate)
            continue

        patch_faces.add(candidate)
        interior_vertices.update(newly_created_interior)
        for neighbor_face in face_neighbors[candidate]:
            if (
                neighbor_face not in patch_faces
                and neighbor_face not in rejected_faces
                and neighbor_face not in blocked
                and neighbor_face not in queued_faces
            ):
                queue.append(neighbor_face)
                queued_faces.add(neighbor_face)

    return patch_faces


def find_all_developable_face_patches(
    mesh: Mesh,
    pointwise_curvature: np.ndarray,
    epsilon: float,
) -> list[set[int]]:
    """Greedily partition all triangles into developable face patches."""
    if epsilon < 0:
        raise ValueError("Epsilon must be nonnegative")

    assigned_faces: set[int] = set()
    patches: list[set[int]] = []
    all_faces = set(range(len(mesh.triangles)))

    while assigned_faces != all_faces:
        unassigned = all_faces - assigned_faces

        def seed_score(face_index: int) -> float:
            triangle = mesh.triangles[face_index]
            return max(
                # Old seed score: abs(angle_defects[vertex_index])
                abs(pointwise_curvature[vertex_index])
                for vertex_index in (
                    triangle.vertex_1,
                    triangle.vertex_2,
                    triangle.vertex_3,
                )
            )

        seed_face = min(unassigned, key=lambda face: (seed_score(face), face))
        patch = grow_developable_face_patch(
            mesh,
            pointwise_curvature,
            seed_face,
            epsilon,
            blocked_faces=assigned_faces,
        )
        if not patch:
            patch = {seed_face}
        patches.append(patch)
        assigned_faces.update(patch)

    patches.sort(key=len, reverse=True)
    return patches


def show_mesh(
    mesh: Mesh,
    seed: int | None,
    seed_face: int | None,
    epsilon: float,
    show_all_patches: bool,
    min_patch_size: int,
) -> None:
    curvature, angle_defects, boundary = mesh.gaussian_curvature()
    normalized_curvature = scale_normalized_curvature(mesh, curvature)
    normals = mesh.vertex_normals()
    finite_abs_curvature = np.abs(curvature[np.isfinite(curvature)])
    color_limit = (
        float(np.percentile(finite_abs_curvature, 98))
        if finite_abs_curvature.size
        else 1.0
    )
    if color_limit <= np.finfo(float).eps:
        color_limit = 1.0

    finite_abs_defect = np.abs(angle_defects[np.isfinite(angle_defects)])
    defect_color_limit = (
        float(np.percentile(finite_abs_defect, 98))
        if finite_abs_defect.size
        else 1.0
    )
    if defect_color_limit <= np.finfo(float).eps:
        defect_color_limit = 1.0

    ps.init()
    surface = ps.register_surface_mesh(
        "0.off",
        mesh.vertex_array(),
        mesh.triangle_array(),
        smooth_shade=True,
    )
    surface.set_color((0.0, 1.0, 0.0))
    surface.add_scalar_quantity(
        "Pointwise Gaussian curvature K",
        curvature,
        defined_on="vertices",
        datatype="symmetric",
        cmap="coolwarm",
        vminmax=(-color_limit, color_limit),
        enabled=False,
        onscreen_colorbar_enabled=True,
    )
    finite_normalized = np.abs(
        normalized_curvature[np.isfinite(normalized_curvature)]
    )
    normalized_limit = (
        float(np.percentile(finite_normalized, 98))
        if finite_normalized.size
        else 1.0
    )
    if normalized_limit <= np.finfo(float).eps:
        normalized_limit = 1.0
    surface.add_scalar_quantity(
        "Scale-normalized Gaussian curvature K L^2",
        normalized_curvature,
        defined_on="vertices",
        datatype="symmetric",
        cmap="coolwarm",
        vminmax=(-normalized_limit, normalized_limit),
        enabled=False,
        onscreen_colorbar_enabled=True,
    )
    surface.add_scalar_quantity(
        "Integrated Gaussian curvature (angle defect)",
        angle_defects,
        defined_on="vertices",
        datatype="symmetric",
        cmap="coolwarm",
        vminmax=(-defect_color_limit, defect_color_limit),
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

    if seed is not None:
        # Old call: grow_flat_patch(mesh, angle_defects, seed, epsilon)
        patch = grow_flat_patch(mesh, curvature, seed, epsilon)
        patch_colors = np.full((len(mesh.vertices), 3), (0.65, 0.65, 0.65))
        if patch:
            patch_colors[list(patch)] = (1.0, 0.75, 0.0)
        patch_colors[seed] = (1.0, 0.0, 1.0)
        surface.add_color_quantity(
            "Flat patch (yellow), seed (magenta)",
            patch_colors,
            defined_on="vertices",
            enabled=True,
        )

        if patch:
            print(
                f"Flat patch from seed {seed}: {len(patch)} vertices "
                f"(|pointwise K| <= {epsilon:g})"
            )
        else:
            print(
                f"Seed {seed} was rejected: |pointwise K|="
                f"{abs(curvature[seed]):.6g}, epsilon={epsilon:g}"
            )

    if seed_face is not None:
        # Old call used angle_defects instead of curvature.
        face_patch = grow_developable_face_patch(mesh, curvature, seed_face, epsilon)
        face_colors = np.full((len(mesh.triangles), 3), (0.65, 0.65, 0.65))
        face_colors[list(face_patch)] = (1.0, 0.75, 0.0)
        face_colors[seed_face] = (1.0, 0.0, 1.0)
        surface.add_color_quantity(
            "Developable face patch (yellow), seed face (magenta)",
            face_colors,
            defined_on="faces",
            enabled=True,
        )
        print(
            f"Developable patch from face {seed_face}: "
            f"{len(face_patch)} faces (interior |pointwise K| <= {epsilon:g})"
        )

    if show_all_patches:
        # Old call used angle_defects instead of curvature.
        all_patches = find_all_developable_face_patches(mesh, curvature, epsilon)
        patches = [
            patch for patch in all_patches if len(patch) >= min_patch_size
        ]
        patch_colors = np.full((len(mesh.triangles), 3), (0.65, 0.65, 0.65))
        for patch_index, patch in enumerate(patches):
            hue = (patch_index * 0.61803398875) % 1.0
            color = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
            patch_colors[list(patch)] = color

        surface.add_color_quantity(
            "All developable face patches",
            patch_colors,
            defined_on="faces",
            enabled=True,
        )
        largest_sizes = [len(patch) for patch in patches[:10]]
        print(f"Total developable face patches: {len(all_patches)}")
        print(
            f"Displayed patches (faces >= {min_patch_size}): {len(patches)} "
            f"(epsilon={epsilon:g})"
        )
        print(f"Largest patch sizes: {largest_sizes}")

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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="vertex index from which to grow a connected flat patch",
    )
    parser.add_argument(
        "--seed-face",
        type=int,
        default=None,
        help="triangle index from which to grow one developable face patch",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help=(
            "maximum absolute pointwise Gaussian curvature "
            "(angle defect / vertex area; default: 0.01)"
        ),
    )
    parser.add_argument(
        "--all-patches",
        action="store_true",
        help="partition and color all triangles as developable face patches",
    )
    parser.add_argument(
        "--min-patch-size",
        type=int,
        default=10,
        help="hide face patches smaller than this many triangles (default: 10)",
    )
    args = parser.parse_args()
    selected_modes = sum(
        (args.seed is not None, args.seed_face is not None, args.all_patches)
    )
    if selected_modes > 1:
        parser.error("use only one of --seed, --seed-face, or --all-patches")

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

    show_mesh(
        mesh,
        args.seed,
        args.seed_face,
        args.epsilon,
        args.all_patches,
        args.min_patch_size,
    )


if __name__ == "__main__":
    main()
