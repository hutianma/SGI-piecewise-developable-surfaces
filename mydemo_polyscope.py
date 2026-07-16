from __future__ import annotations

import argparse
import colorsys
import csv
from collections import deque
from dataclasses import dataclass, field, replace
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


@dataclass(frozen=True)
class LeakageCandidate:
    """A sharp mesh edge which a near-zero-curvature BFS can cross."""

    edge_index: int
    vertex_1: int
    vertex_2: int
    dihedral_degrees: float
    angle_defect_1: float
    angle_defect_2: float
    gaussian_curvature_1: float
    gaussian_curvature_2: float
    component: int = -1
    is_entry: bool = False
    entry_vertex: int | None = None
    discovery_order: int | None = None
    component_edge_count: int = 0
    component_endpoints: tuple[int, ...] = ()


@dataclass(frozen=True)
class FlatPatchTrace:
    """A flat-patch result together with its deterministic BFS tree."""

    patch: set[int]
    parent: dict[int, int | None]
    discovery_order: dict[int, int]


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

    def face_normals(self) -> np.ndarray:
        """Return consistently sized unit normals for all triangles."""
        positions = self.vertex_array()
        normals = np.zeros((len(self.triangles), 3), dtype=float)
        for face_index, (i, j, k) in enumerate(self.triangle_array()):
            normal = np.cross(positions[j] - positions[i], positions[k] - positions[i])
            length = np.linalg.norm(normal)
            if length > np.finfo(float).eps:
                normals[face_index] = normal / length
        return normals


def grow_flat_patch(
    mesh: Mesh,
    angle_defects: np.ndarray,
    seed: int,
    epsilon: float,
) -> set[int]:
    """Flood-fill the zero-curvature component containing ``seed``."""
    return grow_flat_patch_with_trace(
        mesh, angle_defects, seed, epsilon
    ).patch


def grow_flat_patch_with_trace(
    mesh: Mesh,
    angle_defects: np.ndarray,
    seed: int,
    epsilon: float,
) -> FlatPatchTrace:
    """Flood-fill a flat patch and retain how the BFS first reached each vertex."""
    if not 0 <= seed < len(mesh.vertices):
        raise ValueError(
            f"Seed {seed} is out of range; expected 0..{len(mesh.vertices) - 1}"
        )
    if epsilon < 0:
        raise ValueError("Epsilon must be nonnegative")
    if (
        not np.isfinite(angle_defects[seed])
        or abs(angle_defects[seed]) > epsilon
    ):
        return FlatPatchTrace(set(), {}, {})

    patch = {seed}
    visited = {seed}
    queue = deque([seed])
    parent: dict[int, int | None] = {seed: None}
    discovery_order = {seed: 0}

    while queue:
        current = queue.popleft()
        for neighbor in mesh.vertices[current].vertex_indices:
            if neighbor in visited:
                continue
            visited.add(neighbor)

            if (
                np.isfinite(angle_defects[neighbor])
                and abs(angle_defects[neighbor]) <= epsilon
            ):
                patch.add(neighbor)
                parent[neighbor] = current
                discovery_order[neighbor] = len(discovery_order)
                queue.append(neighbor)

    return FlatPatchTrace(patch, parent, discovery_order)


def find_leakage_candidates(
    mesh: Mesh,
    angle_defects: np.ndarray,
    gaussian_curvature: np.ndarray,
    patch: set[int],
    min_dihedral_degrees: float,
) -> list[LeakageCandidate]:
    """Find sharp interior edges crossed by a zero-curvature vertex patch.

    Gaussian curvature alone cannot distinguish adjacent developable pieces.
    An edge is therefore reported (not automatically blocked) when both of its
    endpoints were accepted by the BFS and its two incident faces form a fold
    at least ``min_dihedral_degrees`` wide.
    """
    if not 0.0 <= min_dihedral_degrees <= 180.0:
        raise ValueError("Leakage angle must be between 0 and 180 degrees")

    face_normals = mesh.face_normals()
    candidates: list[LeakageCandidate] = []
    for edge in mesh.edges:
        if (
            edge.vertex_1 not in patch
            or edge.vertex_2 not in patch
            or len(edge.triangle_indices) != 2
        ):
            continue
        normal_1, normal_2 = face_normals[edge.triangle_indices]
        if not np.any(normal_1) or not np.any(normal_2):
            continue
        cosine = float(np.clip(np.dot(normal_1, normal_2), -1.0, 1.0))
        dihedral_degrees = float(np.degrees(np.arccos(cosine)))
        if dihedral_degrees + 1e-12 < min_dihedral_degrees:
            continue
        candidates.append(
            LeakageCandidate(
                edge.index,
                edge.vertex_1,
                edge.vertex_2,
                dihedral_degrees,
                float(angle_defects[edge.vertex_1]),
                float(angle_defects[edge.vertex_2]),
                float(gaussian_curvature[edge.vertex_1]),
                float(gaussian_curvature[edge.vertex_2]),
            )
        )

    candidates.sort(
        key=lambda candidate: (-candidate.dihedral_degrees, candidate.edge_index)
    )
    return candidates


def annotate_leakage_entries(
    candidates: list[LeakageCandidate],
    trace: FlatPatchTrace,
) -> list[LeakageCandidate]:
    """Group sharp edges into seams and mark the first BFS crossing of each seam."""
    if not candidates:
        return []

    candidates_by_vertex: dict[int, set[int]] = {}
    for candidate_index, candidate in enumerate(candidates):
        for vertex in (candidate.vertex_1, candidate.vertex_2):
            candidates_by_vertex.setdefault(vertex, set()).add(candidate_index)

    component_by_candidate: dict[int, int] = {}
    component = 0
    for start in range(len(candidates)):
        if start in component_by_candidate:
            continue
        component_by_candidate[start] = component
        queue = deque([start])
        while queue:
            current = queue.popleft()
            edge = candidates[current]
            for vertex in (edge.vertex_1, edge.vertex_2):
                for neighbor in candidates_by_vertex[vertex]:
                    if neighbor not in component_by_candidate:
                        component_by_candidate[neighbor] = component
                        queue.append(neighbor)
        component += 1

    first_crossing: dict[int, tuple[int, int, int]] = {}
    for index, candidate in enumerate(candidates):
        entry_vertex = None
        if trace.parent.get(candidate.vertex_1) == candidate.vertex_2:
            entry_vertex = candidate.vertex_1
        elif trace.parent.get(candidate.vertex_2) == candidate.vertex_1:
            entry_vertex = candidate.vertex_2
        if entry_vertex is None:
            continue
        seam = component_by_candidate[index]
        order = trace.discovery_order[entry_vertex]
        crossing = (order, index, entry_vertex)
        if seam not in first_crossing or crossing < first_crossing[seam]:
            first_crossing[seam] = crossing

    indices_by_component: dict[int, list[int]] = {}
    for index, seam in component_by_candidate.items():
        indices_by_component.setdefault(seam, []).append(index)
    endpoints_by_component: dict[int, tuple[int, ...]] = {}
    for seam, indices in indices_by_component.items():
        degree: dict[int, int] = {}
        for index in indices:
            edge = candidates[index]
            for vertex in (edge.vertex_1, edge.vertex_2):
                degree[vertex] = degree.get(vertex, 0) + 1
        endpoints_by_component[seam] = tuple(
            sorted(vertex for vertex, count in degree.items() if count == 1)
        )

    annotated = []
    for index, candidate in enumerate(candidates):
        seam = component_by_candidate[index]
        crossing = first_crossing.get(seam)
        is_entry = crossing is not None and crossing[1] == index
        annotated.append(
            replace(
                candidate,
                component=seam,
                is_entry=is_entry,
                entry_vertex=crossing[2] if is_entry else None,
                discovery_order=crossing[0] if is_entry else None,
                component_edge_count=len(indices_by_component[seam]),
                component_endpoints=endpoints_by_component[seam],
            )
        )
    annotated.sort(
        key=lambda candidate: (
            candidate.component,
            not candidate.is_entry,
            -candidate.dihedral_degrees,
            candidate.edge_index,
        )
    )
    return annotated


def write_leakage_report(
    filename: str | Path,
    candidates: list[LeakageCandidate],
) -> None:
    """Write leakage candidates in a format suitable for batch comparisons."""
    filename = Path(filename)
    with filename.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            (
                "component",
                "is_entry",
                "entry_vertex",
                "discovery_order",
                "component_edge_count",
                "component_endpoints",
                "edge",
                "vertex_1",
                "vertex_2",
                "dihedral_degrees",
                "angle_defect_1_radians",
                "angle_defect_2_radians",
                "gaussian_curvature_1",
                "gaussian_curvature_2",
            )
        )
        for candidate in candidates:
            writer.writerow(
                (
                    candidate.component,
                    candidate.is_entry,
                    candidate.entry_vertex if candidate.entry_vertex is not None else "",
                    candidate.discovery_order
                    if candidate.discovery_order is not None
                    else "",
                    candidate.component_edge_count,
                    ";".join(str(vertex) for vertex in candidate.component_endpoints),
                    candidate.edge_index,
                    candidate.vertex_1,
                    candidate.vertex_2,
                    f"{candidate.dihedral_degrees:.12g}",
                    f"{candidate.angle_defect_1:.12g}",
                    f"{candidate.angle_defect_2:.12g}",
                    f"{candidate.gaussian_curvature_1:.12g}",
                    f"{candidate.gaussian_curvature_2:.12g}",
                )
            )


def run_epsilon_sweep(
    mesh: Mesh,
    angle_defects: np.ndarray,
    epsilons: list[float],
    seed: int | None,
    report: Path | None,
) -> None:
    """Report near-zero angle-defect coverage without changing the mesh."""
    if not epsilons:
        raise ValueError("Epsilon sweep requires at least one value")
    if any(epsilon < 0 for epsilon in epsilons):
        raise ValueError("Epsilon sweep values must be nonnegative")

    rows = []
    vertex_count = len(mesh.vertices)
    print(
        "epsilon | near-zero | seed patch | gray | +outside | -outside | "
        "components | largest"
    )
    for epsilon in sorted(set(epsilons)):
        near_zero = np.isfinite(angle_defects) & (np.abs(angle_defects) <= epsilon)
        patches = find_all_flat_patches(mesh, angle_defects, epsilon)
        seed_patch_size = ""
        if seed is not None:
            seed_patch_size = len(
                grow_flat_patch(mesh, angle_defects, seed, epsilon)
            )
        row = {
            "epsilon_radians": epsilon,
            "near_zero_vertices": int(np.count_nonzero(near_zero)),
            "near_zero_percent": 100.0 * np.count_nonzero(near_zero) / vertex_count,
            "seed": "" if seed is None else seed,
            "seed_patch_vertices": seed_patch_size,
            "gray_vertices": int(np.count_nonzero(~near_zero)),
            "positive_outside": int(np.count_nonzero(angle_defects > epsilon)),
            "negative_outside": int(np.count_nonzero(angle_defects < -epsilon)),
            "near_zero_components": len(patches),
            "largest_component": len(patches[0]) if patches else 0,
            "max_abs_angle_defect": float(np.max(np.abs(angle_defects))),
            "total_abs_angle_defect": float(np.sum(np.abs(angle_defects))),
        }
        rows.append(row)
        print(
            f"{epsilon:7g} | {row['near_zero_vertices']:9d} | "
            f"{str(seed_patch_size):10s} | {row['gray_vertices']:4d} | "
            f"{row['positive_outside']:8d} | {row['negative_outside']:8d} | "
            f"{row['near_zero_components']:10d} | {row['largest_component']:7d}"
        )

    if report is not None:
        with report.open("w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote epsilon sweep report: {report}")


def scale_normalized_curvature(mesh: Mesh, curvature: np.ndarray) -> np.ndarray:
    """Make pointwise Gaussian curvature invariant to uniform model scaling."""
    positions = mesh.vertex_array()
    diagonal = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    if diagonal <= np.finfo(float).eps:
        return np.full_like(curvature, np.nan)
    return curvature * diagonal * diagonal


def find_all_flat_patches(
    mesh: Mesh,
    angle_defects: np.ndarray,
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
            or not np.isfinite(angle_defects[vertex_index])
            or abs(angle_defects[vertex_index]) > epsilon
        ):
            continue

        patch = grow_flat_patch(mesh, angle_defects, vertex_index, epsilon)
        assigned.update(patch)
        if len(patch) >= min_size:
            patches.append(patch)

    patches.sort(key=len, reverse=True)
    return patches


def grow_developable_face_patch(
    mesh: Mesh,
    angle_defects: np.ndarray,
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

        acceptable = all(
            np.isfinite(angle_defects[vertex_index])
            and abs(angle_defects[vertex_index]) <= epsilon
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
    angle_defects: np.ndarray,
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
                abs(angle_defects[vertex_index])
                for vertex_index in (
                    triangle.vertex_1,
                    triangle.vertex_2,
                    triangle.vertex_3,
                )
            )

        seed_face = min(unassigned, key=lambda face: (seed_score(face), face))
        patch = grow_developable_face_patch(
            mesh,
            angle_defects,
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
    leakage_angle_degrees: float,
    leakage_report: Path | None,
    gray_report: Path | None,
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
        trace = grow_flat_patch_with_trace(mesh, angle_defects, seed, epsilon)
        patch = trace.patch
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

        outside_vertices = sorted(set(range(len(mesh.vertices))) - patch)
        if outside_vertices:
            outside_cloud = ps.register_point_cloud(
                "Gray/outside vertices (enable to inspect IDs)",
                mesh.vertex_array()[outside_vertices],
                enabled=False,
            )
            outside_cloud.set_color((0.35, 0.35, 0.35))
            outside_cloud.set_radius(0.005, relative=True)
            outside_cloud.add_scalar_quantity(
                "Original vertex index",
                np.asarray(outside_vertices, dtype=float),
                enabled=False,
            )
            outside_cloud.add_scalar_quantity(
                "Angle defect (radians)",
                angle_defects[outside_vertices],
                datatype="symmetric",
                enabled=False,
            )

        high_defect_outside = [
            vertex
            for vertex in outside_vertices
            if not np.isfinite(angle_defects[vertex])
            or abs(angle_defects[vertex]) > epsilon
        ]
        disconnected_zero_defect = sorted(
            set(outside_vertices) - set(high_defect_outside)
        )
        print(
            f"Gray/outside vertices: {len(outside_vertices)} "
            f"({len(high_defect_outside)} rejected by angle defect, "
            f"{len(disconnected_zero_defect)} near-zero but disconnected)"
        )
        print(f"Gray vertex IDs: {outside_vertices}")
        if gray_report is not None:
            with gray_report.open("w", newline="", encoding="utf-8") as report_file:
                writer = csv.writer(report_file)
                writer.writerow(
                    (
                        "vertex",
                        "angle_defect_radians",
                        "gaussian_curvature",
                        "reason",
                    )
                )
                high_defect_set = set(high_defect_outside)
                for vertex in outside_vertices:
                    writer.writerow(
                        (
                            vertex,
                            f"{angle_defects[vertex]:.12g}",
                            f"{curvature[vertex]:.12g}",
                            "angle_defect" if vertex in high_defect_set else "disconnected",
                        )
                    )
            print(f"Wrote gray vertex report: {gray_report}")

        leakage_candidates = annotate_leakage_entries(
            find_leakage_candidates(
                mesh,
                angle_defects,
                curvature,
                patch,
                leakage_angle_degrees,
            ),
            trace,
        )
        if leakage_candidates:
            leakage_vertex_indices = sorted(
                {
                    vertex
                    for candidate in leakage_candidates
                    for vertex in (candidate.vertex_1, candidate.vertex_2)
                }
            )
            leakage_vertex_map = {
                vertex: local_index
                for local_index, vertex in enumerate(leakage_vertex_indices)
            }
            leakage_edges = np.asarray(
                [
                    (
                        leakage_vertex_map[candidate.vertex_1],
                        leakage_vertex_map[candidate.vertex_2],
                    )
                    for candidate in leakage_candidates
                ],
                dtype=np.int32,
            )
            leakage_network = ps.register_curve_network(
                "Leakage candidates",
                mesh.vertex_array()[leakage_vertex_indices],
                leakage_edges,
                enabled=True,
            )
            leakage_network.set_color((1.0, 0.0, 0.0))
            leakage_network.set_radius(0.004, relative=True)

            entry_vertices = [
                candidate.entry_vertex
                for candidate in leakage_candidates
                if candidate.is_entry and candidate.entry_vertex is not None
            ]
            if entry_vertices:
                entry_points = ps.register_point_cloud(
                    "Leakage entry vertices",
                    mesh.vertex_array()[entry_vertices],
                    enabled=True,
                )
                entry_points.set_color((0.1, 0.3, 1.0))
                entry_points.set_radius(0.012, relative=True)

            seam_endpoints = sorted(
                {
                    vertex
                    for candidate in leakage_candidates
                    for vertex in candidate.component_endpoints
                }
            )
            if seam_endpoints:
                endpoint_points = ps.register_point_cloud(
                    "Open seam endpoints (possible leakage-around points)",
                    mesh.vertex_array()[seam_endpoints],
                    enabled=True,
                )
                endpoint_points.set_color((0.0, 1.0, 1.0))
                endpoint_points.set_radius(0.009, relative=True)

        component_count = len(
            {candidate.component for candidate in leakage_candidates}
        )
        entries = [
            candidate for candidate in leakage_candidates if candidate.is_entry
        ]
        print(
            f"Leakage candidates (accepted edges with dihedral >= "
            f"{leakage_angle_degrees:g} degrees): {len(leakage_candidates)}"
        )
        print(
            f"Leakage seam components: {component_count}; "
            f"first BFS entry vertices found: {len(entries)}"
        )
        component_representatives = {
            candidate.component: candidate
            for candidate in leakage_candidates
        }
        for component in sorted(component_representatives):
            representative = component_representatives[component]
            print(
                f"  component {component}: "
                f"{representative.component_edge_count} edges, "
                f"open endpoints={list(representative.component_endpoints)}"
            )
        for candidate in entries:
            print(
                f"  component {candidate.component}, entry vertex "
                f"{candidate.entry_vertex} at BFS order "
                f"{candidate.discovery_order}, edge {candidate.edge_index}: "
                f"v{candidate.vertex_1}--v{candidate.vertex_2}, "
                f"dihedral={candidate.dihedral_degrees:.6g} deg, "
                f"angle defect=({candidate.angle_defect_1:.6g}, "
                f"{candidate.angle_defect_2:.6g}) rad"
            )
        if leakage_report is not None:
            write_leakage_report(leakage_report, leakage_candidates)
            print(f"Wrote leakage report: {leakage_report}")

        if patch:
            print(
                f"Flat patch from seed {seed}: {len(patch)} vertices "
                f"(|angle defect| <= {epsilon:g} radians)"
            )
        else:
            print(
                f"Seed {seed} was rejected: |angle defect|="
                f"{abs(angle_defects[seed]):.6g}, epsilon={epsilon:g}"
            )

    if seed_face is not None:
        face_patch = grow_developable_face_patch(
            mesh, angle_defects, seed_face, epsilon
        )
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
            f"{len(face_patch)} faces (interior |angle defect| <= {epsilon:g})"
        )

    if show_all_patches:
        all_patches = find_all_developable_face_patches(
            mesh, angle_defects, epsilon
        )
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
            "maximum absolute integrated Gaussian curvature "
            "(angle defect, in radians; default: 0.01)"
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
    parser.add_argument(
        "--leakage-angle",
        type=float,
        default=20.0,
        help=(
            "report BFS-accepted edges whose face-normal dihedral is at least "
            "this many degrees (default: 20)"
        ),
    )
    parser.add_argument(
        "--leakage-report",
        type=Path,
        default=None,
        help="write leakage candidates for --seed to a CSV file",
    )
    parser.add_argument(
        "--gray-report",
        type=Path,
        default=None,
        help="write gray/outside vertex IDs and curvature values to a CSV file",
    )
    parser.add_argument(
        "--epsilon-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="EPSILON",
        help="analyze several angle-defect thresholds and exit without opening the viewer",
    )
    parser.add_argument(
        "--epsilon-sweep-report",
        type=Path,
        default=None,
        help="write --epsilon-sweep measurements to a CSV file",
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

    if args.epsilon_sweep is not None:
        _, angle_defects, _ = mesh.gaussian_curvature()
        run_epsilon_sweep(
            mesh,
            angle_defects,
            args.epsilon_sweep,
            args.seed,
            args.epsilon_sweep_report,
        )
        return

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
        args.leakage_angle,
        args.leakage_report,
        args.gray_report,
    )


if __name__ == "__main__":
    main()
