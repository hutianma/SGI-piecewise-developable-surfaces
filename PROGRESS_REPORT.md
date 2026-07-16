# Progress Report: Angle-Defect Patch Growth and Vertex Optimization

## Project objective

The current objective is to maximize the portion of the input mesh that is
classified as approximately developable. In the visualization, accepted
vertices are shown in yellow and rejected vertices are shown in gray.

The current experiment uses discrete integrated Gaussian curvature (angle
defect) rather than pointwise Gaussian curvature as the BFS growth criterion.
The longer-term goal is to determine whether vertices with moderate nonzero
angle defect can be moved so that larger regions become approximately
zero-curvature, while preserving the original geometry and mesh quality.

## Current curvature measurement

For an interior vertex \(v_i\), the angle defect is

\[
\Omega_i=2\pi-\sum_{f\ni v_i}\theta_{if}.
\]

For a boundary vertex, it is

\[
\Omega_i^{\partial}=\pi-\sum_{f\ni v_i}\theta_{if}.
\]

The pointwise Gaussian-curvature estimate is also computed for diagnostic
purposes:

\[
K_i=\frac{\Omega_i}{A_i},
\]

where \(A_i\) is one third of the total area of the incident triangles.
However, the current BFS uses \(\Omega_i\), not \(K_i\).

The BFS accepts a neighboring vertex when

\[
|\Omega_i|\leq\epsilon.
\]

Angle defect is invariant under uniform scaling of the model, although its
per-vertex magnitude depends on mesh sampling density.

## Gauss--Bonnet constraint

The discrete Gauss--Bonnet relation is

\[
\sum_{i\in V}\Omega_i=2\pi\chi(M).
\]

The currently loaded FanDisk mesh is closed, has no boundary vertices, and has
genus-zero topology. Therefore,

\[
\sum_i\Omega_i\approx4\pi.
\]

This is an important limitation: all angle defects cannot be made zero without
changing topology, introducing boundaries, or allowing curvature to remain at
a set of corners or singular vertices. A more appropriate objective is to make
large patch interiors approximately zero-curvature while concentrating the
unavoidable curvature at a small, geometrically meaningful singular set.

## Epsilon sweep on FanDisk

The initial FanDisk model contains 6,475 vertices and 12,946 triangles. Using
seed vertex 248 gives the following angle-defect classifications.

| Epsilon | Near-zero vertices | Coverage | Gray vertices |
|---:|---:|---:|---:|
| 0.001 | 5,714 | 88.25% | 761 |
| 0.005 | 5,952 | 91.92% | 523 |
| 0.01 | 6,066 | 93.68% | 409 |
| 0.02 | 6,071 | 93.76% | 404 |
| 0.05 | 6,385 | 98.61% | 90 |
| 0.10 | 6,427 | 99.26% | 48 |
| 0.20 | 6,438 | 99.43% | 37 |
| 0.50 | 6,453 | 99.66% | 22 |
| 1.60 | 6,475 | 100% | 0 |

Increasing epsilon changes the classification but does not change the
geometry. In particular, epsilon 1.6 makes every vertex yellow only because it
exceeds the maximum absolute angle defect. It does not mean the mesh has become
developable. For the first deformation experiment, epsilon 0.1 was selected.

## Leakage diagnostics

The implementation also reports sharp edges whose endpoints are accepted by
the angle-defect BFS:

\[
|\Omega_i|,|\Omega_j|\leq\epsilon,
\qquad
\theta_{\mathrm{dihedral}}(i,j)\geq20^\circ.
\]

These edges are only diagnostic candidates. A large dihedral angle is an
extrinsic sharpness measurement and is not itself a developability criterion.
A folded sheet can have zero intrinsic Gaussian curvature while having a large
dihedral angle. Therefore, leakage candidates are not currently used to stop
the BFS or partition the yellow region.

## Normal-displacement optimization model

At epsilon 0.1, the initial fixed singular candidate set is

\[
S_0=\{i:|\Omega_i^0|>1\}.
\]

FanDisk has 22 such vertices. They remain fixed during the first experiment.
The moderate-defect target set at iteration \(k\) is

\[
G^{(k)}=
\{i\notin S_0:0.1<|\Omega_i^{(k)}|\leq1\}.
\]

The one-ring experiment uses

\[
M^{(k)}=
\left(G^{(k)}\cup\operatorname{oneRing}(G^{(k)})\right)\setminus S_0
\]

as the movable set. Each movable vertex is constrained to move along its
current normal:

\[
x_i'=x_i+d_i n_i^{(k)}.
\]

Normals and active sets are frozen during each inner solve and recomputed after
an accepted outer update.

The main normalized threshold energy is

\[
\widehat E_K=
\frac{1}{|R^{(k)}\setminus S_0|}
\sum_{i\in R^{(k)}\setminus S_0}
\left[
\frac{\max(|\Omega_i'|-\epsilon,0)}{\epsilon}
\right]^2.
\]

The complete objective also contains:

- singular-curvature preservation;
- positional displacement regularization;
- displacement-vector smoothness;
- relative edge-length preservation;
- an optional weak zero-curvature energy.

The per-iteration displacement bound is

\[
|d_i|\leq0.02l_i,
\]

and the cumulative displacement bound is

\[
\|x_i^{(k)}-x_i^0\|\leq0.1l_i^0.
\]

Candidate updates are accepted only when the global thresholded curvature
energy decreases and all geometric safety conditions are satisfied. The
checks include triangle orientation, area ratio, relative edge distortion,
cumulative displacement, and finite angle defects. Backtracking is used when a
full update is unsafe.

## Experimental results

### One-ring normal displacement

The threshold-only one-ring experiment produced:

| Measurement | Initial | Final |
|---|---:|---:|
| Moderate gray vertices | 26 | 25 |
| Total gray vertices, including 22 singular candidates | 48 | 47 |
| Global thresholded energy | 1.19454 | 0.16444 |

The thresholded energy decreased by approximately 86.2%, no triangle flipped,
and the angle-defect sum remained approximately \(4\pi\). However, the gray
count decreased by only one vertex.

Several original moderate defects were substantially reduced, for example from
approximately 0.39 to the range 0.13--0.26. At the same time, several vertices
whose initial defects were approximately zero became slightly larger than 0.1.
This demonstrates curvature redistribution: normal displacement reduces the
largest local defects but can transfer curvature to neighboring vertices.

Seven of the final 25 moderate gray vertices reached at least 95% of their
cumulative displacement bounds. The median cumulative-bound usage among the
remaining gray vertices was approximately 79%.

### Weak zero-curvature term

An additional normalized zero-curvature term was tested:

\[
\lambda_0
\frac{1}{|R\setminus S_0|}
\sum_{i\in R\setminus S_0}
\left(\frac{\Omega_i}{\epsilon}\right)^2.
\]

| \(\lambda_0\) | Moderate gray | Total gray | Threshold energy | Global zero energy |
|---:|---:|---:|---:|---:|
| 0 | 25 | 47 | 0.164440 | 2.165985 |
| 0.001 | 25 | 47 | 0.164432 | 2.160925 |
| 0.01 | 28 | 50 | 0.164455 | 2.129148 |

A larger zero-curvature weight reduced the total squared curvature but
increased the number of vertices above the classification threshold. The term
is therefore optional and does not currently improve the primary yellow-area
objective.

### Two-ring normal displacement

The movable region was expanded to two rings while all other parameters were
kept unchanged.

| Measurement | One ring | Two rings |
|---|---:|---:|
| Moderate gray vertices | 25 | 28 |
| Total gray vertices | 47 | 50 |
| Threshold energy | 0.164440 | 0.165895 |
| Global zero energy | 2.165985 | 2.181596 |

The wider region preserved singular curvature more strongly, but distributed
curvature to more ordinary vertices. It did not improve the main objective.

## Current interpretation

Normal-only displacement is not completely ineffective: it substantially
reduces the magnitude of moderate angle defects while preserving topology and
mesh validity. However, under the current bounds it has limited ability to
reduce the number of gray vertices. The primary difficulty appears to be
curvature redistribution and insufficient movement directions, rather than an
active region that is too small.

The experiments do not yet demonstrate that normal displacement can produce a
better piecewise-developable partition. They demonstrate only that local
angle-defect energy can be reduced safely under controlled deformation.

## Proposed next steps

1. Keep the one-ring active region; the two-ring experiment was less effective.
2. Use the threshold-only objective, or retain only a very small optional
   zero-curvature weight.
3. Test a small tangential displacement in addition to normal displacement:

   \[
   x_i'=x_i+d_i^n n_i+d_i^{t_1}t_{i1}+d_i^{t_2}t_{i2}.
   \]

   The tangential bound should be substantially smaller than the normal bound,
   for example 0.5% of local edge length versus 2% in the normal direction.
4. If the additional movement directions improve the gray count without
   excessive distortion, introduce Shape Diameter Function information as an
   adaptive displacement bound or thickness-preservation regularizer.
5. If epsilon 0.1 stabilizes successfully, use the continuation schedule

   \[
   0.1\rightarrow0.05\rightarrow0.02\rightarrow0.01.
   \]

## Concerns and questions

1. **Global impossibility of zero curvature everywhere.** Because FanDisk is a
   closed genus-zero mesh, Gauss--Bonnet requires total angle defect \(4\pi\).
   We need a principled definition of which corners or singular vertices are
   allowed to retain this curvature.

2. **Dependence on mesh sampling.** Angle defect is scale-invariant but depends
   on triangulation density. A fixed epsilon may not be comparable across the
   common 3D test models.

3. **Objective mismatch.** Reducing squared curvature energy does not
   necessarily reduce the number of gray vertices. Curvature may be spread
   among multiple vertices just above the threshold.

4. **Curvature transfer.** Moving one vertex changes the angle defects of its
   neighbors. A local improvement can create new gray vertices elsewhere.

5. **Normal-only limitation.** Some remaining vertices are near their
   displacement bounds, while others appear insensitive to further normal
   motion. Additional tangential freedom may be required.

6. **Meaning of the singular set.** The threshold \(|\Omega|>1\) is currently
   heuristic. These vertices should be inspected to determine whether they are
   true geometric corners, discretization artifacts, or optimization
   singularities.

7. **Definition of a developable patch.** Connected zero Gaussian curvature
   alone may merge visually different developable regions. Conversely, sharp
   dihedral seams are extrinsic features rather than intrinsic developability
   criteria. The desired patch-coherence definition needs to be clarified.

## Questions for discussion

- Should the singular set be selected from known geometric features rather
  than an angle-defect threshold?
- Is maximizing the number of vertices below a fixed angle-defect tolerance the
  intended objective, or should the method optimize a continuous curvature
  norm?
- Should patch boundaries be prescribed before optimization so that curvature
  can be concentrated there?
- Should the next experiment add small tangential displacement, or should Shape
  Diameter Function constraints be introduced first?
- Which curvature measurement and epsilon should be used when comparing meshes
  with different sampling densities?
