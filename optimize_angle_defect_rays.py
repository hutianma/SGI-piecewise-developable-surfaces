from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh

from mydemo_polyscope import Mesh
from optimize_angle_defect import (
    angle_defects,
    local_average_edge_lengths,
    triangle_metrics,
    vertex_normals,
    write_vertex_report,
)


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct an orthonormal tangent basis around a unit normal."""
    reference = np.array((1.0, 0.0, 0.0))
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.array((0.0, 1.0, 0.0))
    tangent_1 = np.cross(normal, reference)
    tangent_1 /= np.linalg.norm(tangent_1)
    tangent_2 = np.cross(normal, tangent_1)
    return tangent_1, tangent_2


def double_cone_directions(
    normal: np.ndarray, ray_count: int, half_angle_degrees: float
) -> np.ndarray:
    """Sample antipodal caps around the outward and inward normals."""
    if ray_count < 2:
        raise ValueError("Double-cone search requires at least two rays")
    half_count = max(1, ray_count // 2)
    tangent_1, tangent_2 = tangent_basis(normal)
    cosine_min = np.cos(np.radians(half_angle_degrees))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    outward = []
    for sample in range(half_count):
        fraction = (sample + 0.5) / half_count
        cosine = 1.0 - fraction * (1.0 - cosine_min)
        sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
        azimuth = sample * golden_angle
        direction = (
            cosine * normal
            + sine * np.cos(azimuth) * tangent_1
            + sine * np.sin(azimuth) * tangent_2
        )
        outward.append(direction / np.linalg.norm(direction))
    directions = np.asarray(outward + [-direction for direction in outward])
    return directions[:ray_count]


def sphere_directions(ray_count: int) -> np.ndarray:
    """Uniform deterministic Fibonacci sampling of the unit sphere."""
    if ray_count < 2:
        raise ValueError("Sphere search requires at least two rays")
    indices = np.arange(ray_count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / ray_count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = indices * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def threshold_energy(
    defects: np.ndarray, singular: np.ndarray, epsilon: float
) -> float:
    excess = np.maximum(np.abs(defects[~singular]) - epsilon, 0.0)
    return float(np.sum(excess * excess))


def singular_energy(
    defects: np.ndarray,
    initial_defects: np.ndarray,
    singular: np.ndarray,
    epsilon: float,
) -> float:
    if not np.any(singular):
        return 0.0
    denominator = np.maximum(np.abs(initial_defects[singular]), epsilon)
    residual = (defects[singular] - initial_defects[singular]) / denominator
    return float(np.mean(residual * residual))


def merit(
    defects: np.ndarray,
    initial_defects: np.ndarray,
    singular: np.ndarray,
    epsilon: float,
    singular_weight: float,
) -> float:
    return threshold_energy(defects, singular, epsilon) + singular_weight * singular_energy(
        defects, initial_defects, singular, epsilon
    )


def vertex_trial(
    positions: np.ndarray,
    vertex: int,
    direction: np.ndarray,
    distance: float,
) -> np.ndarray:
    trial = positions.copy()
    trial[vertex] += distance * direction
    return trial


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
    gauss_bonnet_target = float(np.sum(initial_defects))
    cumulative_limit = args.total_fraction * initial_lengths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_vertex_report(
        args.output_dir / "singular_vertices.csv",
        np.flatnonzero(singular),
        original,
        initial_defects,
    )
    write_vertex_report(
        args.output_dir / "initial_gray_vertices.csv",
        np.flatnonzero((~singular) & (np.abs(initial_defects) > args.epsilon)),
        original,
        initial_defects,
    )

    statistics: list[dict[str, object]] = []
    print(
        f"Loaded {len(positions)} vertices; singular={np.count_nonzero(singular)}, "
        f"sum(angle defect)={gauss_bonnet_target:.12g}"
    )
    print(
        f"Direction mode={args.direction_mode}, rays={args.rays}, "
        f"coarse samples={args.coarse_samples}, binary steps={args.binary_steps}"
    )

    for outer_iteration in range(args.max_iterations):
        defects = angle_defects(faces, positions, boundary)
        gray = (~singular) & (~boundary) & (np.abs(defects) > args.epsilon)
        targets = np.flatnonzero(gray & (np.abs(defects) <= args.singular_threshold))
        targets = targets[np.argsort(-np.abs(defects[targets]))]
        if not len(targets):
            print(f"Iteration {outer_iteration}: no moderate gray targets remain")
            break

        energy_before = threshold_energy(defects, singular, args.epsilon)
        gray_before = int(np.count_nonzero(gray))
        accepted_updates = 0
        sign_brackets = 0
        tested_candidates = 0
        max_step = 0.0

        for vertex in targets:
            defects = angle_defects(faces, positions, boundary)
            if abs(defects[vertex]) <= args.epsilon or singular[vertex]:
                continue
            normals = vertex_normals(faces, positions)
            normal = normals[vertex]
            if np.linalg.norm(normal) <= np.finfo(float).eps:
                continue
            if args.direction_mode == "cone":
                directions = double_cone_directions(
                    normal, args.rays, args.cone_half_angle
                )
            else:
                directions = sphere_directions(args.rays)

            current_lengths = local_average_edge_lengths(len(positions), edges, positions)
            search_radius = args.search_fraction * current_lengths[vertex]
            remaining = max(
                0.0,
                cumulative_limit[vertex]
                - np.linalg.norm(positions[vertex] - original[vertex]),
            )
            search_limit = min(search_radius, remaining)
            if search_limit <= np.finfo(float).eps:
                continue

            current_merit = merit(
                defects,
                initial_defects,
                singular,
                args.epsilon,
                args.singular_weight,
            )
            best_merit = current_merit
            best_positions = None
            best_defects = None
            best_distance = 0.0

            for direction in directions:
                samples: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []
                for distance in np.linspace(
                    0.0, search_limit, args.coarse_samples + 1
                ):
                    trial = vertex_trial(positions, vertex, direction, float(distance))
                    trial_defects = angle_defects(faces, trial, boundary)
                    trial_merit = merit(
                        trial_defects,
                        initial_defects,
                        singular,
                        args.epsilon,
                        args.singular_weight,
                    )
                    samples.append(
                        (
                            float(distance),
                            float(trial_defects[vertex]),
                            trial_merit,
                            trial,
                            trial_defects,
                        )
                    )
                    tested_candidates += 1
                    if trial_merit < best_merit:
                        best_merit = trial_merit
                        best_positions = trial
                        best_defects = trial_defects
                        best_distance = float(distance)

                for left, right in zip(samples[:-1], samples[1:]):
                    if left[1] == 0.0 or right[1] == 0.0 or left[1] * right[1] > 0.0:
                        continue
                    sign_brackets += 1
                    low_distance, low_defect = left[0], left[1]
                    high_distance, high_defect = right[0], right[1]
                    for _ in range(args.binary_steps):
                        mid_distance = 0.5 * (low_distance + high_distance)
                        trial = vertex_trial(
                            positions, vertex, direction, mid_distance
                        )
                        trial_defects = angle_defects(faces, trial, boundary)
                        trial_merit = merit(
                            trial_defects,
                            initial_defects,
                            singular,
                            args.epsilon,
                            args.singular_weight,
                        )
                        tested_candidates += 1
                        if trial_merit < best_merit:
                            best_merit = trial_merit
                            best_positions = trial
                            best_defects = trial_defects
                            best_distance = mid_distance
                        mid_defect = float(trial_defects[vertex])
                        if abs(mid_defect) <= args.root_tolerance:
                            break
                        if low_defect * mid_defect <= 0.0:
                            high_distance, high_defect = mid_distance, mid_defect
                        else:
                            low_distance, low_defect = mid_distance, mid_defect

            if best_positions is None or best_defects is None:
                continue
            orientation_ok, area_ratio, _ = triangle_metrics(
                faces, positions, best_positions
            )
            old_edge_lengths = np.linalg.norm(
                positions[edges[:, 0]] - positions[edges[:, 1]], axis=1
            )
            new_edge_lengths = np.linalg.norm(
                best_positions[edges[:, 0]] - best_positions[edges[:, 1]], axis=1
            )
            edge_distortion = float(
                np.max(np.abs(new_edge_lengths - old_edge_lengths) / old_edge_lengths)
            )
            cumulative_ok = bool(
                np.linalg.norm(best_positions[vertex] - original[vertex])
                <= cumulative_limit[vertex] + 1e-12
            )
            if (
                best_merit < current_merit - args.minimum_improvement
                and threshold_energy(best_defects, singular, args.epsilon)
                < threshold_energy(defects, singular, args.epsilon) - args.minimum_improvement
                and orientation_ok
                and area_ratio > args.min_area_ratio
                and edge_distortion < args.max_edge_distortion
                and cumulative_ok
            ):
                positions = best_positions
                accepted_updates += 1
                max_step = max(max_step, best_distance)

        final_defects = angle_defects(faces, positions, boundary)
        final_gray = (~singular) & (np.abs(final_defects) > args.epsilon)
        energy_after = threshold_energy(final_defects, singular, args.epsilon)
        gauss_sum = float(np.sum(final_defects))
        row = {
            "iteration": outer_iteration,
            "direction_mode": args.direction_mode,
            "ray_count": args.rays,
            "gray_before": gray_before,
            "gray_after": int(np.count_nonzero(final_gray)),
            "accepted_vertex_updates": accepted_updates,
            "sign_change_brackets": sign_brackets,
            "tested_candidates": tested_candidates,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "max_step": max_step,
            "max_cumulative_displacement": float(
                np.max(np.linalg.norm(positions - original, axis=1))
            ),
            "gauss_bonnet_sum": gauss_sum,
            "gauss_bonnet_residual": gauss_sum - gauss_bonnet_target,
        }
        statistics.append(row)
        print(
            f"Iteration {outer_iteration}: gray {gray_before}->{row['gray_after']}, "
            f"energy {energy_before:.6g}->{energy_after:.6g}, "
            f"accepted vertices={accepted_updates}, sign brackets={sign_brackets}"
        )
        if accepted_updates == 0 or energy_before - energy_after < args.minimum_improvement:
            print("Ray search converged: no meaningful safe improvement")
            break

    final_defects = angle_defects(faces, positions, boundary)
    final_gray_vertices = np.flatnonzero(
        (~singular) & (np.abs(final_defects) > args.epsilon)
    )
    write_vertex_report(
        args.output_dir / "final_gray_vertices.csv",
        final_gray_vertices,
        positions,
        final_defects,
    )
    with (args.output_dir / "optimization_iterations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        if statistics:
            writer = csv.DictWriter(report_file, fieldnames=list(statistics[0]))
            writer.writeheader()
            writer.writerows(statistics)
    with (args.output_dir / "vertex_displacements.csv").open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        writer = csv.writer(report_file)
        writer.writerow(("vertex", "dx", "dy", "dz", "magnitude"))
        for vertex, displacement in enumerate(positions - original):
            writer.writerow((vertex, *displacement, np.linalg.norm(displacement)))

    output_name = f"{args.mesh.stem}_ray_optimized_{args.direction_mode}.obj"
    trimesh.Trimesh(vertices=positions, faces=faces, process=False).export(
        args.output_dir / output_name
    )
    print(
        f"Finished: moderate gray={len(final_gray_vertices)}, "
        f"total gray including singular="
        f"{len(final_gray_vertices) + np.count_nonzero(singular)}"
    )
    print(f"Outputs: {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Angle-defect vertex updates by ray sampling and binary search"
    )
    parser.add_argument("mesh", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/fandisk_ray_optimization")
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--singular-threshold", type=float, default=1.0)
    parser.add_argument("--direction-mode", choices=("cone", "sphere"), default="cone")
    parser.add_argument("--rays", type=int, default=30)
    parser.add_argument("--cone-half-angle", type=float, default=60.0)
    parser.add_argument("--coarse-samples", type=int, default=5)
    parser.add_argument("--binary-steps", type=int, default=12)
    parser.add_argument("--root-tolerance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--search-fraction",
        type=float,
        default=0.1,
        help="maximum ray-search distance as a fraction of local edge length",
    )
    parser.add_argument("--total-fraction", type=float, default=0.1)
    parser.add_argument("--singular-weight", type=float, default=0.1)
    parser.add_argument("--min-area-ratio", type=float, default=0.2)
    parser.add_argument("--max-edge-distortion", type=float, default=0.1)
    parser.add_argument("--minimum-improvement", type=float, default=1e-12)
    optimize(parser.parse_args())


if __name__ == "__main__":
    main()
