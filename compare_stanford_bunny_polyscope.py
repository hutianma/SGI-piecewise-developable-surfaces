from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polyscope as ps

from compare_fandisk_polyscope import register_result
from mydemo_polyscope import Mesh


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "results/stanford_bunny/stanford_bunny_comparison_package"

MODELS = (
    ("Original — no optimization", ROOT / "test-models/data/stanford-bunny.obj"),
    (
        "Normal — local objective + joint one-ring constraints",
        PACKAGE
        / "methods/normal_joint_one_ring_local_objective_shape_constrained_eps_0.1"
        / "stanford-bunny_optimized_epsilon_0.1.obj",
    ),
    (
        "Double cone — 30 rays + geometry safety",
        PACKAGE
        / "methods/double_cone_30_rays_geometry_safety_constrained_eps_0.1"
        / "stanford-bunny_ray_optimized_cone.obj",
    ),
    (
        "Virtual sphere — 30 rays + geometry safety",
        PACKAGE
        / "methods/virtual_sphere_30_rays_geometry_safety_constrained_eps_0.1"
        / "stanford-bunny_ray_optimized_sphere.obj",
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Stanford Bunny original and three optimizer results"
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=("row", "overlay"), default="row")
    parser.add_argument("--spacing", type=float, default=1.25)
    args = parser.parse_args()

    loaded = [(name, Mesh.load(path)) for name, path in MODELS]
    reference_count = len(loaded[0][1].vertices)
    if not 0 <= args.seed < reference_count:
        parser.error(f"seed must be in [0, {reference_count - 1}]")

    reference_positions = loaded[0][1].vertex_array()
    width = float(np.ptp(reference_positions[:, 0]))

    ps.init()
    for index, (name, mesh) in enumerate(loaded):
        if len(mesh.vertices) != reference_count:
            raise ValueError(f"{name} does not share the original vertex indexing")
        offset = np.zeros(3)
        if args.layout == "row":
            offset[0] = index * args.spacing * width
        yellow, gray = register_result(name, mesh, args.epsilon, args.seed, offset)
        print(f"{name}: yellow={yellow}, gray/outside={gray}")

    ps.show()


if __name__ == "__main__":
    main()
