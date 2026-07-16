# Stanford Bunny global-shape comparison

These metrics compare each optimized surface with the original surface as a
whole. All methods preserve topology and vertex correspondence.

- `area_weighted_rms_surface_displacement`: exact RMS displacement of the
  piecewise-linear surface, integrated using original triangle areas.
- `normalized_area_weighted_rms_surface_displacement`: the preceding value
  divided by the original bounding-box diagonal. This is the primary global
  shape-difference metric.
- `sampled_*_corresponding_surface_displacement`: mean, 95th percentile, and
  maximum displacement of 100,000 deterministic area-uniform samples. Each
  original sample is compared with the point having the same face and
  barycentric coordinates on the optimized surface.
- `normalized_sampled_*`: the corresponding sampled distance divided by the
  original bounding-box diagonal.
- `relative_total_surface_area_change`: signed total-area change divided by the
  original total area; positive means expansion and negative means contraction.
- `relative_bbox_[x|y|z]_change`: signed change of each bounding-box dimension.
- `surface_centroid_shift`: distance between area-weighted surface centroids.
- `normalized_surface_centroid_shift`: centroid shift divided by the original
  bounding-box diagonal.

The sampled 95th-percentile displacement is zero for the cone and sphere up to
floating-point tolerance because fewer than 5% of the surface samples lie on
changed faces. This does not mean that those methods make no local changes.

An independently sampled point-cloud Hausdorff estimate was intentionally not
reported: at this displacement scale, sampling spacing dominated the measured
distance. A robust exact surface Hausdorff computation would require a separate
triangle-distance implementation.
