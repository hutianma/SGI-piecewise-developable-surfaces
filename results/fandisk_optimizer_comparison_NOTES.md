# FanDisk optimizer-comparison notation

This file defines the notation and every column in
`fandisk_optimizer_comparison.csv`.

## Geometry

- \(V\): mesh vertex set.
- \(E\): mesh edge set.
- \(F\): mesh triangle set.
- \(x_i^0\in\mathbb R^3\): original position of vertex \(i\).
- \(x_i'\in\mathbb R^3\): optimized position of vertex \(i\).
- \(A_f^0,A_f'\): original and optimized areas of triangle \(f\).
- \(l_{ij}^0=\|x_i^0-x_j^0\|\): original length of edge \((i,j)\).
- \(l_{ij}'=\|x_i'-x_j'\|\): optimized edge length.

## Curvature

For an interior vertex,

\[
\Omega_i=2\pi-\sum_{f\ni i}\theta_{if}.
\]

For a boundary vertex,

\[
\Omega_i^\partial=\pi-\sum_{f\ni i}\theta_{if}.
\]

- \(\Omega_i\): integrated discrete Gaussian curvature (angle defect), in
  radians.
- \(K_i=\Omega_i/A_i\): pointwise Gaussian-curvature estimate. It is recorded
  in some diagnostic CSV files but is not used by the current BFS.
- \(\epsilon=0.1\): yellow/gray angle-defect threshold in this comparison.
- Yellow non-singular vertex: \(|\Omega_i|\le\epsilon\).
- Moderate gray vertex: \(|\Omega_i|>\epsilon\) and \(i\notin S_0\).
- \(S_0=\{i:|\Omega_i^0|>1\}\): fixed initial singular-candidate set. FanDisk
  contains 22 such vertices.
- Total gray count: moderate gray count plus the 22 fixed singular candidates.

FanDisk is closed and genus zero, so discrete Gauss--Bonnet requires

\[
\sum_i\Omega_i\approx4\pi.
\]

The optimizer therefore cannot make every angle defect exactly zero while
preserving the same closed topology.

## Curvature energies

Threshold energy:

\[
E_{\mathrm{threshold}}
=
\sum_{i\notin S_0}
\left[\max(|\Omega_i|-\epsilon,0)\right]^2.
\]

It measures the severity of threshold violations. Vertices already satisfying
\(|\Omega_i|\le\epsilon\) contribute zero.

Global zero energy:

\[
E_{\mathrm{zero}}
=
\sum_{i\notin S_0}\Omega_i^2.
\]

It measures distance from exact zero angle defect over the entire non-singular
mesh. Already-yellow vertices with nonzero angle defect still contribute.

Both energies exclude \(S_0\) and have units of radians squared. A lower value
is better, but neither energy alone equals the number of gray vertices.

## Curvature-transfer counts

- `original_gray_became_yellow`: number of initially moderate-gray vertices
  that satisfy \(|\Omega_i'|\le\epsilon\) after optimization.
- `original_yellow_became_gray`: number of initially yellow non-singular
  vertices that satisfy \(|\Omega_i'|>\epsilon\) after optimization.

For example, the normal optimizer changes six original gray vertices to yellow
but changes five original yellow vertices to gray, giving a net improvement of
only one gray vertex.

## Shape-distortion measurements

Maximum displacement:

\[
D_{\max}=\max_i\|x_i'-x_i^0\|.
\]

Mean moved displacement averages the same norm only over vertices with nonzero
movement.

Maximum accumulated edge distortion:

\[
E_{\mathrm{edge,max}}
=
\max_{(i,j)\in E}
\frac{|l_{ij}'-l_{ij}^0|}{l_{ij}^0}.
\]

Minimum area ratio:

\[
A_{\min\mathrm{-ratio}}
=
\min_{f\in F}\frac{A_f'}{A_f^0}.
\]

Triangle quality for a triangle with side lengths \(a,b,c\) and area \(A\):

\[
q_f=\frac{4\sqrt{3}A}{a^2+b^2+c^2}.
\]

An equilateral triangle has \(q_f=1\), while a degenerate triangle approaches
zero. `min_triangle_quality` is the minimum over all triangles.

## Method names

- `original`: undeformed FanDisk.
- `normal_one_ring`: joint bounded optimization in which the moderate gray
  vertices and their one-ring neighbors move along fixed per-iteration normals.
- `double_cone`: sequential ray search using directions sampled in outward and
  inward cones around the vertex normal. Binary search is used when sampled
  positions bracket an angle-defect sign change.
- `virtual_sphere`: the same sequential search using Fibonacci-sphere directions
  rather than two normal-centered cones.

The cone directions borrow the ray-sampling geometry of the Shape Diameter
Function paper. This experiment does not yet compute the complete Shape
Diameter Function or use SDF values as thickness constraints.

## Reading the comparison

The primary yellow-area measurements are `moderate_gray`, `total_gray`, and the
two curvature-transfer columns. The two curvature energies describe severity,
while displacement, edge distortion, area ratio, and triangle quality describe
the geometric cost.

The current comparison is preliminary because the normal method and ray methods
use different update schedules and iteration counts. A controlled benchmark
should use the same initialization, \(\epsilon\), singular set, cumulative
displacement budget, stopping rule, and computational budget.
