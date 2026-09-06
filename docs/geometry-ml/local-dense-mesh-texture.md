# Steps 14–17: Local Dense Reconstruction, Mesh, and Photo Texture

## Outcome

Steps 14–17 are complete for the frozen **local** Step 10 sparse component. The integrated report records `steps14_17_success=true` and `blender_manual_cleanup_started=false`.

This is a partial 3D reconstruction from one 73-view capture arc. It is not a global 288-image or complete 360-degree model. The 266-image Step 13 learned model remains evidence only because it missed the frozen 274-image global gate; it was never used as dense input.

## Frozen source and backend

The only sparse source was `reconstruction/sparse/best`:

| Measurement | Value |
|---|---:|
| Registered images | 73 |
| Sparse points | 6,099 |
| Observations | 21,351 |
| Mean reprojection error | 1.237305 px |
| Camera | One `SIMPLE_RADIAL` camera |
| Ordered-name SHA-256 | `5548006c3368bbd71cff6563b19b789bb9b397faf7e265f078e54dc2dc704c00` |

The installed pyCOLMAP 4.2.0 wheel reports `has_cuda=false`, so dense stereo used the official COLMAP 4.2.0 Windows CUDA CLI at `C:\Tools\COLMAP-4.2.0\bin\colmap.exe`. The live command probes verified `image_undistorter`, `patch_match_stereo`, `stereo_fusion`, both meshers, `mesh_simplifier`, and `mesh_texturer`. The same capability record verifies Blender 5.2.0 LTS at `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe` for automated preview validation.

## Step 14: dense workspace preparation

COLMAP undistortion created a workspace containing exactly the 73 registered Step 10 images. Every undistorted image reopened at 1202 × 1600. The sparse points, registered-image names, calibrated poses, and pyCOLMAP-derived `PINHOLE` intrinsics were rechecked against the source model. Raw and selected-image integrity checks passed before and after preparation.

The real sparse-source preview is `reconstruction/local_dense/previews/step14_01_local_sparse_source.png`.

## Step 15: CUDA PatchMatch and fusion

PatchMatch used one preferred configuration:

```text
max_image_size = 1600
geom_consistency = true
filter = true
gpu_index = 0
cache_size = 1 GiB
num_threads = 2
```

The CUDA run completed all 73 geometric depth and normal maps in 1,811.67 seconds. No lower-resolution PatchMatch fallback ran.

The first geometric fusion process used a 1 GiB cache and was interrupted after more than five hours of pathological cache thrashing: it had consumed 19,254 CPU seconds and about 1,792.8 GiB of cumulative reads without producing `fused.ply`. The controlled correction reused the already-completed depth maps and changed only `StereoFusion.cache_size` from 1 GiB to 4 GiB. It completed in 101.83 seconds. This was an operational fusion-cache correction, not a second PatchMatch experiment.

The fused cloud contains:

| Measurement | Value |
|---|---:|
| Points | 391,899 |
| Dense/sparse ratio | 64.2563× |
| Finite XYZ | 391,899 / 391,899 |
| Colors and normals | Present |
| Nonzero finite normals | 100% |
| Points inside the Step 10 sparse box | 99.6428% |
| Points inside a one-span expanded box | 99.9980% |
| Points outside that expanded box | 8 |

The full coordinate bounds include a few long-range points. The catastrophic-bounds gate therefore measures the 0.1–99.9 percentile core and requires at least 99% of points inside the one-span expanded Step 10 box. The full cloud and outliers remain visible in `reconstruction/local_dense/previews/step15_01_dense_cloud.png`; nothing was cropped to improve the presentation.

## Step 16: mesh selection, component cleanup, and simplification

Poisson was the primary mesher because all fused normals were finite and nonzero and the vessel surface is smooth. It completed in 58.36 seconds and produced 1,088,150 vertices and 2,040,189 faces. Its plausible vessel-related surface was fragmented into 5,476 connected components; the largest component held 36.00% of faces.

The one allowed Delaunay alternative completed in 45.26 seconds with 69,153 faces and a nominal 89.50% dominant component. Visual inspection rejected it because large unsupported triangular sheets span empty space and form the forbidden giant-shell failure. This shows why face count or component dominance alone cannot select the mesh.

The Poisson surface retained the recognizable vessel silhouette and plausible bounds, so cleanup used one deterministic rule: keep only connected components containing at least 0.5% of the Poisson faces. The threshold was 10,201 faces. It kept 18 components and 1,971,613 faces, removed 5,458 small components, and retained 96.6387% of all Poisson faces. No manual deletion, sculpting, hole filling, remeshing, or invented geometry occurred.

The component-filtered mesh was still larger than needed for texture mapping and automated Blender validation. One COLMAP QEM simplification therefore targeted 500,000 faces using the measured ratio 0.253599. It completed in 26.91 seconds and produced the accepted final mesh:

| Measurement | Value |
|---|---:|
| Vertices | 283,341 |
| Faces | 499,999 |
| File size | 9,900,265 bytes |
| Connected components | 21 |
| Largest component share | 36.4871% |
| Finite XYZ and spatial rank | 100%; rank 3 |
| SHA-256 | `deaded890a09b81ed8840bbad2ed465a1650fb628346574c8099b15bb5c4878c` |

The primary and QEM previews preserve the vessel silhouette and major surface ribbons without an artificial enclosing shell. Because both meshes exceed the deterministic 180,000-face preview limit, Step 16 records that these figures use up to 100,000 sampled vertices over the full coordinate bounds rather than rendered triangles. They support silhouette, bounds, fragmentation, and giant-sheet review; the exact final triangle surface receives the stronger import/render check in Step 17. The final mesh is `reconstruction/local_dense/mesh/final_mesh.ply`; the unsimplified Poisson and component-filtered outputs remain as evidence.

## Step 17: COLMAP photo texture and Blender validation

COLMAP `mesh_texturer` completed one primary attempt in 48.80 seconds. No texturing retry ran. Its isolated output is `reconstruction/local_dense/texture/attempt_1/`.

Validation reopened both generated files and compared the textured mesh to `final_mesh.ply`:

| Measurement | Value |
|---|---:|
| Textured vertices/faces | 283,341 / 499,999 |
| Coordinates preserved | Yes |
| Face connectivity and order preserved | Yes |
| Meaningful UV face coverage | 73.0045% |
| Required UV coverage | At least 50% |
| Finite UV atlas range | Within normalized [0, 1] range |
| Atlas | 4096 × 1902 PNG |
| Textured mesh SHA-256 | `84390f29775c37624dbb1df7e66730fa9440f17557cd0372b01df9bc20094f22` |
| Atlas SHA-256 | `434878f4b5347ada92f05321bfc78ffc91e795869677dcaf8ab72173ad183156` |

The headless Blender validator imported the exact triangular mesh and UVs, loaded the COLMAP atlas, created one material, and rendered three calibrated views. It did not save a `.blend` file or perform manual cleanup. The final triptych is `reconstruction/local_dense/previews/step17_01_textured_preview.png`.

The triptych shows recognizable image-derived dark brass, gold highlights, and view-dependent surface color. It also honestly shows the limitations: disconnected geometry, missing areas outside the local capture support, dark regions, texture seams, reflective-highlight inconsistency, and unassigned faces. No procedural material was substituted for the photo texture.

## Publication and storage boundary

The public repository contains the source code, tests, reports, previews, and compact accepted final outputs needed to inspect the measured result. Heavy restart state under `reconstruction/local_dense/work/`, `.codegraph/`, and the private CNN checkpoint remain local-only. Oversized or redundant intermediate mesh files also remain local evidence; in particular, `component_filtered_mesh.ply` is about 111 MiB and is not suitable for ordinary GitHub publication. Their measured counts, decisions, hashes, and comparison previews remain recorded in the published Step 16 reports. This storage boundary does not change the accepted final mesh or textured result.

## Gate summary

| Stage | Hard gate | Result |
|---|---|---|
| 14 | Exact Step 10 source, 73-view workspace, CUDA dense capability, protected inputs unchanged | Passed |
| 15 | Reopenable finite colored cloud, more than 6,099 points, plausible robust bounds, coherent visual structure | Passed |
| 16 | Reopenable finite 3D mesh, supported dominant component, plausible bounds, no giant shell in the disclosed sampled-vertex preview | Passed |
| 17 | Exact topology preserved, readable photo atlas, normalized UVs with substantial coverage, exact Blender import/render, accepted visual | Passed |

The integrated source of truth is `reconstruction/local_dense/reports/steps14_17_summary.json`.

## Reproduce or inspect

The stages are restartable and preserve their measured attempt ledgers:

```powershell
python -B run_local_reconstruction.py --stage prepare
python -B run_local_reconstruction.py --stage stereo
python -B run_local_reconstruction.py --stage mesh
python -B run_local_reconstruction.py --stage texture
python -B run_local_reconstruction.py --stage finalize
```

`--stage all` uses the same tested gate-owning sequence. Existing accepted artifacts are reopened and hash-checked instead of rerunning successful COLMAP work. A new visual decision requires both `--visual-status` and a substantive `--visual-note`; the report binds that decision to the current artifact and preview hashes. On Windows, interruption terminates the complete launched process tree, and restart checks include surviving descendants. Unrecorded pre-existing final-mesh and texture-attempt outputs are rejected rather than overwritten.

Manual Blender cleanup and final presentation work require a separate phase. They were not started here.

## Verification

- 49 focused Steps 14-17 tests passed, including a real Windows descendant-process timeout check.
- 106 Step 10-13 regression tests passed.
- The complete project suite passed: 247 tests.
- All four implementation modules and all four focused test files compiled successfully.
- The real `--stage all` path completed with acceptance true and reused the accepted artifacts without starting another PatchMatch, mesher, simplifier, or texturer attempt.
- Final reopening verified the Step 10 model at 73 images / 6,099 points / 1.237305 px; the 391,899-point dense cloud; the 283,341-vertex / 499,999-face final mesh; exact textured topology; the 4096 × 1902 atlas; and the three-view Blender report.
- Integrity rechecked all 297 raw images, all 288 selected images, the exact 73 local image hashes, and 160 protected Step 10-13 files with zero mismatches.
- `.codegraph/` and the private segmentation checkpoint remained present. No `.blend`, manual/sculpt output, or intermediate per-camera PNG remained.
- Markdown links, whitespace, and Git diff checks passed after documentation and cleanup.

The stage reports retain the exact subprocess arguments, versions, runtime, hashes, metrics, acceptance checks, and visual-review notes used for this result.
