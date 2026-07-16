# FanDisk comparison package

This self-contained package contains the undeformed FanDisk reference and the
three retained optimization results evaluated at angle-defect threshold
`epsilon = 0.1` radians.

## Contents

- `original_mesh/`: original FanDisk OBJ with no optimization.
- `methods/normal_joint_one_ring_shape_constrained_eps_0.1/`: joint one-ring
  optimization restricted to vertex-normal displacement.
- `methods/double_cone_30_rays_geometry_safety_constrained_eps_0.1/`: sequential
  search over 30 inward/outward cone rays.
- `methods/virtual_sphere_30_rays_geometry_safety_constrained_eps_0.1/`:
  sequential search over 30 Fibonacci-sphere rays.
- `comparison/fandisk_optimizer_comparison.csv`: common quantitative metrics.
- `comparison/fandisk_retained_methods_manifest.csv`: exact movement model,
  parameters, penalty weights, and safety constraints for each method.
- `comparison/fandisk_optimizer_comparison_NOTES.md`: notation and definitions.

The leakage and epsilon-sweep files remain outside this package because they
are diagnostic analyses rather than optimization-method outputs.
