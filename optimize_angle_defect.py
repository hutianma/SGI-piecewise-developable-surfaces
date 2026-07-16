from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import trimesh

from mydemo_polyscope import Mesh


def angle_defects(
    faces: np.ndarray, positions: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    """Vectorized interior/boundary angle defects for fixed mesh topology."""
    defects = np.where(boundary, np.pi, 2.0 * np.pi).astype(float)
    for corner, other_1, other_2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = positions[faces[:, other_1]] - positions[faces[:, corner]]
        v = positions[faces[:, other_2]] - positions[faces[:, corner]]
        angles = np.arctan2(
            np.linalg.norm(np.cross(u, v), axis=1),
            np.einsum("ij,ij->i", u, v),
        )
        np.add.at(defects, faces[:, corner], -angles)
    return defects


def vertex_normals(faces: np.ndarray, positions: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions)
    face_vectors = np.cross(
        positions[faces[:, 1]] - positions[faces[:, 0]],
        positions[faces[:, 2]] - positions[faces[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_vectors)
    lengths = np.linalg.norm(normals, axis=1)
    active = lengths > np.finfo(float).eps
    normals[active] /= lengths[active, None]
    return normals


def local_average_edge_lengths(
    vertex_count: int, edges: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    sums = np.zeros(vertex_count)
    counts = np.zeros(vertex_count, dtype=int)
    for corner in range(2):
        np.add.at(sums, edges[:, corner], lengths)
        np.add.at(counts, edges[:, corner], 1)
    return np.divide(sums, counts, out=np.ones(vertex_count), where=counts > 0)


def expand_vertices(mesh: Mesh, vertices: set[int]) -> set[int]:
    expanded = set(vertices)
    for vertex in vertices:
        expanded.update(mesh.vertices[vertex].vertex_indices)
    return expanded


def expand_vertex_rings(mesh: Mesh, vertices: set[int], rings: int) -> set[int]:
    expanded = set(vertices)
    for _ in range(rings):
        expanded = expand_vertices(mesh, expanded)
    return expanded


def triangle_metrics(
    faces: np.ndarray,
    old_positions: np.ndarray,
    new_positions: np.ndarray,
) -> tuple[bool, float, float]:
    old_cross = np.cross(
        old_positions[faces[:, 1]] - old_positions[faces[:, 0]],
        old_positions[faces[:, 2]] - old_positions[faces[:, 0]],
    )
    new_cross = np.cross(
        new_positions[faces[:, 1]] - new_positions[faces[:, 0]],
        new_positions[faces[:, 2]] - new_positions[faces[:, 0]],
    )
    old_twice_area = np.linalg.norm(old_cross, axis=1)
    new_twice_area = np.linalg.norm(new_cross, axis=1)
    valid = old_twice_area > np.finfo(float).eps
    orientation_ok = bool(
        np.all(np.einsum("ij,ij->i", old_cross[valid], new_cross[valid]) > 0)
    )
    area_ratio = float(np.min(new_twice_area[valid] / old_twice_area[valid]))

    a2 = np.sum((new_positions[faces[:, 1]] - new_positions[faces[:, 0]]) ** 2, axis=1)
    b2 = np.sum((new_positions[faces[:, 2]] - new_positions[faces[:, 1]]) ** 2, axis=1)
    c2 = np.sum((new_positions[faces[:, 0]] - new_positions[faces[:, 2]]) ** 2, axis=1)
    quality = np.divide(
        2.0 * np.sqrt(3.0) * new_twice_area,
        a2 + b2 + c2,
        out=np.zeros_like(new_twice_area),
        where=(a2 + b2 + c2) > np.finfo(float).eps,
    )
    return orientation_ok, area_ratio, float(np.min(quality))


def global_energy(defects: np.ndarray, singular: np.ndarray, epsilon: float) -> float:
    active = ~singular
    excess = np.maximum(np.abs(defects[active]) - epsilon, 0.0)
    return float(np.sum(excess * excess))


def write_vertex_report(
    filename: Path,
    vertices: np.ndarray,
    positions: np.ndarray,
    defects: np.ndarray,
) -> None:
    with filename.open("w", newline="", encoding="utf-8") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(("vertex", "x", "y", "z", "angle_defect_radians"))
        for vertex in vertices:
            writer.writerow((vertex, *positions[vertex], defects[vertex]))


def optimize(args: argparse.Namespace) -> None:
    mesh = Mesh.load(args.mesh)
    faces = mesh.triangle_array()
    edges = np.asarray([(edge.vertex_1, edge.vertex_2) for edge in mesh.edges])
    original = mesh.vertex_array().copy()
    positions = original.copy()
    boundary_set = mesh.boundary_vertices()
    boundary = np.zeros(len(positions), dtype=bool)
    if boundary_set:
        boundary[list(boundary_set)] = True

    initial_defects = angle_defects(faces, positions, boundary)
    initial_lengths = local_average_edge_lengths(len(positions), edges, positions)
    singular = (~boundary) & (np.abs(initial_defects) > args.singular_threshold)
    target_gauss_bonnet = float(np.sum(initial_defects))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vertex_report(
        args.output_dir / "singular_vertices.csv",
        np.flatnonzero(singular),
        original,
        initial_defects,
    )
    initial_gray = np.flatnonzero((~singular) & (np.abs(initial_defects) > args.epsilon))
    write_vertex_report(
        args.output_dir / "initial_gray_vertices.csv",
        initial_gray,
        original,
        initial_defects,
    )

    rows: list[dict[str, object]] = []
    stagnant = 0
    previous_gray = -1
    print(
        f"Loaded {len(positions)} vertices; boundary={np.count_nonzero(boundary)}, "
        f"singular={np.count_nonzero(singular)}, "
        f"sum(angle defect)={target_gauss_bonnet:.12g}"
    )

    for iteration in range(args.max_iterations):
        defects = angle_defects(faces, positions, boundary)
        target = (
            (~singular)
            & (~boundary)
            & (np.abs(defects) > args.epsilon)
            & (np.abs(defects) <= args.singular_threshold)
        )
        gray = (~singular) & (np.abs(defects) > args.epsilon)
        gray_count = int(np.count_nonzero(gray))
        if not np.any(target):
            print(f"Iteration {iteration}: no moderate-defect targets remain")
            break

        target_set = set(np.flatnonzero(target).tolist())
        movable_set = expand_vertex_rings(
            mesh, target_set, args.movable_rings
        ) - set(np.flatnonzero(singular))
        movable_set -= boundary_set
        monitored_set = expand_vertices(mesh, movable_set)
        movable = np.asarray(sorted(movable_set), dtype=int)
        monitored = np.asarray(sorted(monitored_set), dtype=int)
        monitored_non_singular = monitored[~singular[monitored]]
        monitored_singular = monitored[singular[monitored]]
        movable_lookup = np.full(len(positions), -1, dtype=int)
        movable_lookup[movable] = np.arange(len(movable))

        normals = vertex_normals(faces, positions)
        local_lengths = local_average_edge_lengths(len(positions), edges, positions)
        step_bounds = args.step_fraction * local_lengths[movable]
        remaining = np.maximum(
            args.total_fraction * initial_lengths[movable]
            - np.linalg.norm(positions[movable] - original[movable], axis=1),
            0.0,
        )
        bounds_magnitude = np.minimum(step_bounds, remaining)
        bounds = [(-bound, bound) for bound in bounds_magnitude]

        affected_edge_mask = np.isin(edges[:, 0], movable) | np.isin(edges[:, 1], movable)
        affected_edges = edges[affected_edge_mask]
        old_edge_lengths = np.linalg.norm(
            positions[affected_edges[:, 0]] - positions[affected_edges[:, 1]], axis=1
        )
        smooth_scale = 0.5 * (
            local_lengths[affected_edges[:, 0]] + local_lengths[affected_edges[:, 1]]
        )
        singular_denominator = np.maximum(
            np.abs(initial_defects[monitored_singular]), args.epsilon
        )

        def candidate_positions(displacements: np.ndarray) -> np.ndarray:
            candidate = positions.copy()
            candidate[movable] += displacements[:, None] * normals[movable]
            return candidate

        def objective(displacements: np.ndarray) -> float:
            candidate = candidate_positions(displacements)
            candidate_defects = angle_defects(faces, candidate, boundary)
            excess = np.maximum(
                np.abs(candidate_defects[monitored_non_singular]) - args.epsilon,
                0.0,
            )
            e_k = float(np.mean((excess / args.epsilon) ** 2))
            e_0 = float(
                np.mean(
                    (candidate_defects[monitored_non_singular] / args.epsilon) ** 2
                )
            )
            if len(monitored_singular):
                e_s = float(
                    np.mean(
                        (
                            (candidate_defects[monitored_singular]
                             - initial_defects[monitored_singular])
                            / singular_denominator
                        )
                        ** 2
                    )
                )
            else:
                e_s = 0.0
            e_p = float(np.mean((displacements / local_lengths[movable]) ** 2))

            displacement_vectors = np.zeros_like(positions)
            displacement_vectors[movable] = displacements[:, None] * normals[movable]
            smooth_difference = (
                displacement_vectors[affected_edges[:, 0]]
                - displacement_vectors[affected_edges[:, 1]]
            )
            e_l = float(
                np.mean(np.sum(smooth_difference * smooth_difference, axis=1) / smooth_scale**2)
            )
            new_edge_lengths = np.linalg.norm(
                candidate[affected_edges[:, 0]] - candidate[affected_edges[:, 1]], axis=1
            )
            e_e = float(np.mean(((new_edge_lengths - old_edge_lengths) / old_edge_lengths) ** 2))
            return (
                e_k
                + args.zero_curvature_weight * e_0
                + 0.1 * e_s
                + 0.01 * e_p
                + 0.05 * e_l
                + 0.1 * e_e
            )

        old_global = global_energy(defects, singular, args.epsilon)
        result = minimize(
            objective,
            np.zeros(len(movable)),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": args.inner_iterations, "ftol": 1e-10, "maxls": 20},
        )

        accepted = False
        accepted_scale = 0.0
        accepted_positions = positions
        accepted_defects = defects
        accepted_edge_distortion = 0.0
        accepted_area_ratio = 1.0
        accepted_quality = 1.0
        for exponent in range(11):
            scale = 0.5**exponent
            trial_displacements = scale * result.x
            trial = candidate_positions(trial_displacements)
            trial_defects = angle_defects(faces, trial, boundary)
            new_global = global_energy(trial_defects, singular, args.epsilon)
            orientation_ok, area_ratio, quality = triangle_metrics(faces, positions, trial)
            new_lengths = np.linalg.norm(
                trial[affected_edges[:, 0]] - trial[affected_edges[:, 1]], axis=1
            )
            edge_distortion = float(
                np.max(np.abs(new_lengths - old_edge_lengths) / old_edge_lengths)
            )
            cumulative_ok = bool(
                np.all(
                    np.linalg.norm(trial[movable] - original[movable], axis=1)
                    <= args.total_fraction * initial_lengths[movable] + 1e-12
                )
            )
            if (
                new_global < old_global - 1e-14
                and orientation_ok
                and area_ratio > args.min_area_ratio
                and edge_distortion < args.max_edge_distortion
                and cumulative_ok
                and np.all(np.isfinite(trial_defects))
            ):
                accepted = True
                accepted_scale = scale
                accepted_positions = trial
                accepted_defects = trial_defects
                accepted_edge_distortion = edge_distortion
                accepted_area_ratio = area_ratio
                accepted_quality = quality
                break

        new_gray_count = int(
            np.count_nonzero((~singular) & (np.abs(accepted_defects) > args.epsilon))
        )
        final_gray_mask = (~singular) & (np.abs(accepted_defects) > args.epsilon)
        local_non_singular = monitored[~singular[monitored]]
        singular_change = accepted_defects[singular] - initial_defects[singular]
        max_step = float(
            np.max(np.linalg.norm(accepted_positions - positions, axis=1))
        )
        positions = accepted_positions
        gauss_sum = float(np.sum(accepted_defects))
        row = {
            "iteration": iteration,
            "epsilon": args.epsilon,
            "target_count": int(np.count_nonzero(target)),
            "gray_before": gray_count,
            "gray_after": new_gray_count,
            "positive_gray_after": int(
                np.count_nonzero(final_gray_mask & (accepted_defects > 0))
            ),
            "negative_gray_after": int(
                np.count_nonzero(final_gray_mask & (accepted_defects < 0))
            ),
            "movable_count": len(movable),
            "monitored_count": len(monitored),
            "global_energy_before": old_global,
            "global_energy_after": global_energy(accepted_defects, singular, args.epsilon),
            "local_zero_energy": float(
                np.sum(accepted_defects[local_non_singular] ** 2)
            ),
            "global_zero_energy": float(
                np.sum(accepted_defects[~singular] ** 2)
            ),
            "singular_preservation_energy": float(np.sum(singular_change**2)),
            "max_abs_defect_outside_singular": float(
                np.max(np.abs(accepted_defects[~singular]))
            ),
            "max_step": max_step,
            "max_cumulative_displacement": float(
                np.max(np.linalg.norm(positions - original, axis=1))
            ),
            "max_relative_edge_distortion": accepted_edge_distortion,
            "min_area_ratio": accepted_area_ratio,
            "min_triangle_quality": accepted_quality,
            "gauss_bonnet_sum": gauss_sum,
            "gauss_bonnet_residual": gauss_sum - target_gauss_bonnet,
            "accepted": accepted,
            "backtracking_scale": accepted_scale,
            "optimizer_success": result.success,
            "optimizer_message": result.message,
        }
        rows.append(row)
        print(
            f"Iteration {iteration}: gray {gray_count}->{new_gray_count}, "
            f"energy {old_global:.6g}->{row['global_energy_after']:.6g}, "
            f"move={max_step:.3g}, accepted={accepted}"
        )

        if not accepted:
            print("No safe energy-decreasing update found")
            break
        if new_gray_count == previous_gray:
            stagnant += 1
        else:
            stagnant = 0
        previous_gray = new_gray_count
        if stagnant >= 5 or max_step < 1e-6 * np.linalg.norm(np.ptp(positions, axis=0)):
            print("Convergence criterion reached")
            break

    final_defects = angle_defects(faces, positions, boundary)
    stats_path = args.output_dir / "optimization_iterations.csv"
    if rows:
        with stats_path.open("w", newline="", encoding="utf-8") as report_file:
            writer = csv.DictWriter(report_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    final_gray = np.flatnonzero((~singular) & (np.abs(final_defects) > args.epsilon))
    write_vertex_report(
        args.output_dir / "final_gray_vertices.csv", final_gray, positions, final_defects
    )
    displacement = positions - original
    with (args.output_dir / "vertex_displacements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        writer = csv.writer(report_file)
        writer.writerow(("vertex", "dx", "dy", "dz", "magnitude"))
        for vertex, vector in enumerate(displacement):
            writer.writerow((vertex, *vector, np.linalg.norm(vector)))

    output_mesh = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
    output_name = f"{args.mesh.stem}_optimized_epsilon_{args.epsilon:g}.obj"
    output_mesh.export(args.output_dir / output_name)
    print(
        f"Finished: moderate gray={len(final_gray)}, "
        f"total gray including singular={len(final_gray) + np.count_nonzero(singular)}"
    )
    print(f"Outputs: {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normal-displacement optimization of angle defect"
    )
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/normal_optimization"))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--singular-threshold", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--inner-iterations", type=int, default=20)
    parser.add_argument("--step-fraction", type=float, default=0.02)
    parser.add_argument("--total-fraction", type=float, default=0.1)
    parser.add_argument("--min-area-ratio", type=float, default=0.2)
    parser.add_argument("--max-edge-distortion", type=float, default=0.1)
    parser.add_argument(
        "--movable-rings",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="number of target-neighborhood rings included as movable vertices",
    )
    parser.add_argument(
        "--zero-curvature-weight",
        type=float,
        default=0.001,
        help="weight of the normalized local sum of squared angle defects",
    )
    optimize(parser.parse_args())


# The original normal-displacement optimizer is retained as an experimental
# baseline, but its default command-line entry point is disabled while the
# ray-search update proposed in the next experiment is being developed.
#
# if __name__ == "__main__":
#     main()
