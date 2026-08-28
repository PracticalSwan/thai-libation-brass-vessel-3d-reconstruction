# Active Context

Updated: 2026-08-28

## Current focus

Preprocessing and Step 6 geometry detection/analysis are complete and verified.
The next possible implementation scope is the still-provisional combined Steps
7+8 SAM 2.1 segmentation and feature-mask analysis plan, but it must be revised
against the real Step 6 interfaces before execution. pyCOLMAP/reconstruction is
not currently authorized or planned for execution.

## Verified preprocessing state

- Raw source: `IMG20260826122949/`, 297 immutable JPEG files at 3072 x 4080.
- Final decisions: 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT`.
- Rejected images: indices 289-297 only, because they are the separate hand-held/flipped sequence with object movement and hand occlusion.
- Selected set: all 288 `ACCEPT` + `WARN` images.
- Selected variant: PREPROCESSED, using a geometry-preserving 15% LAB-luminance CLAHE blend.
- Matching evidence: 2,483 PREPROCESSED versus 2,376 RAW fundamental-matrix RANSAC inliers over ten representative neighboring pairs; PREPROCESSED was non-worse on 9 of 10. Matching used decoded quality-95 JPEG bytes identical to the final export encoding.
- Final input: `preprocessing/pycolmap_input/images/`, 288 readable files at unchanged 3072 x 4080 geometry.
- Integrity: no duplicate selected-output hashes and zero raw SHA-256/size mismatches across all 297 originals.
- Visual review: all ten before/after previews and all four sheets containing every WARN/REJECT case were inspected.

## Completed Step 6

- `analysis_common.py` verifies deterministic manifest access plus selected-file
  readability, geometry, byte size, and SHA-256 before output creation.
- `geometry_detection.py` provides scaled SIFT extraction, BF-L2 ratio-test
  matching, Fundamental Matrix/RANSAC, epilines, and Sampson residuals.
- `shape_geometry.py` provides grayscale/Canny evidence, deterministic
  classical contour selection, bounding box, centroid, PCA principal axis, and
  a residual-gated optional ellipse.
- `run_geometry_analysis.py` reproduced 11 intentional artifacts; six are
  presentation PNGs and five are machine-readable JSON/CSV reports.
- `show_geometry_visuals.py` reuses the real analysis for `matches`,
  `epipolar`, `shape`, or `all` popup modes; the live popup path and
  no-display path both completed in bounded smoke tests.
- Primary pair 165-166: 4,653 / 4,643 keypoints, 478 candidate matches, 300
  RANSAC inliers, 0.628 inlier ratio, and 0.1431 median Sampson error in squared
  analysis-pixel units.
- Supporting pair 255-256: 1,233 / 2,289 keypoints, 57 candidates, and 18
  inliers, preserving the honestly weaker view.
- Shape 165: contour/centroid/box/PCA retained; ellipse omitted as
  `ellipse_unavailable` because its normalized fit residuals failed.
- Shape 255: contour/centroid/box/PCA plus ellipse retained as `ok`.
- All six presentation figures were visually inspected after the final real
  run. The input verifier passed for 288/288 selected images, and the final raw
  check passed for 297/297 unchanged photographs.
- Results: `docs/geometry-ml/geometry-results.md`.

## Planned Steps 7 + 8

- Pretrained Meta SAM 2.1 using `sam2.1_hiera_small` first.
- Isolated ML runtime; no changes to the verified preprocessing environment.
- Segment only ten representative selected images in the current phase: 15, 45, 75, 105, 135, 165, 195, 225, 255, 280.
- Create binary-mask QA and presentation overlays.
- Reuse Step 6 SIFT extraction to count features inside versus outside the vessel masks.
- Produce measured segmentation and feature-mask presentation figures without claiming reconstruction improvement.
- Plan: `docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`.

## Shared planning sources

- Design: `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`.
- Plan index: `docs/superpowers/plans/2026-08-27-geometry-ml-integration.md`.
- Step 6 scripts/reports/figures now exist. No SAM masks/model weights,
  pyCOLMAP outputs, reconstruction, meshing, texturing, or Blender outputs
  exist.

## Next action

Before any ML implementation, re-read the completed Step 6 results and revise
the provisional Steps 7+8 plan around the stable selected-record and SIFT-scale
interfaces. Do not install PyTorch, download SAM checkpoints, generate masks,
or begin pyCOLMAP/reconstruction without separate authorization.
