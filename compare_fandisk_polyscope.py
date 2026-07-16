from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps

from mydemo_polyscope import Mesh, grow_flat_patch


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "results/fandisk/fandisk_comparison_package"

DEFAULT_MODELS = (
    (
        "Original — no optimization",
        PACKAGE / "original_mesh/fandisk_original_no_optimization.obj",
    ),
    (
        "Normal — joint one-ring + shape constraints",
        PACKAGE
        / "methods/normal_joint_one_ring_shape_constrained_eps_0.1"
        / "fandisk_normal_joint_one_ring_shape_constrained.obj",
    ),
    (
        "Double cone — 30 rays + geometry safety",
        PACKAGE
        / "methods/double_cone_30_rays_geometry_safety_constrained_eps_0.1"
        / "fandisk_double_cone_30_rays_geometry_safety_constrained.obj",
    ),
    (
        "Virtual sphere — 30 rays + geometry safety",
        PACKAGE
        / "methods/virtual_sphere_30_rays_geometry_safety_constrained_eps_0.1"
        / "fandisk_virtual_sphere_30_rays_geometry_safety_constrained.obj",
    ),
)


def register_result(
    name: str,
    mesh: Mesh,
    epsilon: float,
    seed: int,
    offset: np.ndarray,
) -> tuple[int, int]:
    positions = mesh.vertex_array() + offset
    _, angle_defects, _ = mesh.gaussian_curvature()
    patch = grow_flat_patch(mesh, angle_defects, seed, epsilon)
    outside = np.asarray(sorted(set(range(len(mesh.vertices))) - patch), dtype=int)

    colors = np.full((len(mesh.vertices), 3), (0.62, 0.62, 0.62))
    if patch:
        colors[list(patch)] = (1.0, 0.75, 0.0)
    colors[seed] = (1.0, 0.0, 1.0)

    surface = ps.register_surface_mesh(
        name,
        positions,
        mesh.triangle_array(),
        smooth_shade=True,
    )
    surface.add_color_quantity(
        "Yellow BFS patch; gray outside; magenta seed",
        colors,
        defined_on="vertices",
        enabled=True,
    )
    surface.add_scalar_quantity(
        "Angle defect (radians)",
        angle_defects,
        defined_on="vertices",
        datatype="symmetric",
        cmap="coolwarm",
        enabled=False,
    )

    if outside.size:
        cloud = ps.register_point_cloud(
            f"{name} — gray/outside vertices",
            positions[outside],
            enabled=True,
        )
        cloud.set_color((0.22, 0.22, 0.22))
        cloud.set_radius(0.006, relative=True)
        cloud.add_scalar_quantity(
            "Original vertex index",
            outside.astype(float),
            enabled=False,
        )
        cloud.add_scalar_quantity(
            "Angle defect (radians)",
            angle_defects[outside],
            datatype="symmetric",
            enabled=False,
        )

    return len(patch), len(outside)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare original and optimized FanDisk yellow/gray regions"
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layout",
        choices=("row", "overlay"),
        default="row",
        help="place models side by side or at identical coordinates",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.25,
        help="row spacing measured in original-model bounding-box widths",
    )
    args = parser.parse_args()

    loaded = [(name, Mesh.load(path)) for name, path in DEFAULT_MODELS]
    if not 0 <= args.seed < len(loaded[0][1].vertices):
        parser.error(f"seed must be in [0, {len(loaded[0][1].vertices) - 1}]")

    reference_positions = loaded[0][1].vertex_array()
    width = float(np.ptp(reference_positions[:, 0]))

    ps.init()
    for index, (name, mesh) in enumerate(loaded):
        if len(mesh.vertices) != len(loaded[0][1].vertices):
            raise ValueError(f"{name} does not share the reference vertex indexing")
        offset = np.zeros(3)
        if args.layout == "row":
            offset[0] = index * args.spacing * width
        yellow, gray = register_result(name, mesh, args.epsilon, args.seed, offset)
        print(f"{name}: yellow={yellow}, gray/outside={gray}")

    ps.show()


if __name__ == "__main__":
    main()
