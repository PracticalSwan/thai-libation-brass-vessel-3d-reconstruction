# Active Context

Updated: 2026-09-01

## Current focus

Preprocessing and Step 6 geometry detection/analysis are complete and verified.
The next planned implementation scope is the revised combined Steps 7+8 custom
CNN segmentation and feature-mask analysis workflow. The architecture is now
planned in detail but remains unimplemented and requires separate execution
authorization. pyCOLMAP/reconstruction is not currently authorized or planned
for execution.

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

- Train a compact project-defined binary segmentation CNN from random initialization; no pretrained backbone, SAM checkpoint, transfer learning, or external segmentation API is part of the baseline.
- Start with 36 manually annotated selected images using a leakage-controlled 24 train / 6 validation / 6 held-out test split based on separated capture positions/view groups.
- Preferred model is a small U-Net-like `SmallSegCNN` with three encoder stages, a compact bottleneck, skip-connected decoder stages, and one output-logit channel.
- Preferred input tensor size is 384 x 288 `(H x W)` to preserve the source portrait aspect ratio in memory.
- Train with BCE-with-logits + Dice loss, Adam, seed 4213, validation-based early stopping, and no test-set tuning.
- Evaluate unedited held-out predictions with Dice, IoU, foreground precision, recall, and visible failure analysis.
- Reuse Step 6 SIFT extraction to count features inside versus outside CNN-predicted vessel masks on the held-out test images.
- Produce real training, segmentation, feature-mask, and summary figures without claiming reconstruction improvement.
- Plan: `docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`.

## Shared planning sources

- Design: `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`.
- Plan index: `docs/superpowers/plans/2026-08-27-geometry-ml-integration.md`.
- Step 6 scripts/reports/figures now exist. No CNN labels, CNN model weights,
  ML predictions, pyCOLMAP outputs, reconstruction, meshing, texturing, or
  Blender outputs exist.

## Next action

The Steps 7+8 architecture revision is complete. Before implementation, re-read
the completed Step 6 results plus the revised CNN design/plan, then verify the
current PyTorch environment and exact labeled-image selection. Do not create
manual masks, modify dependencies, implement/train the CNN, generate ML
predictions, or begin pyCOLMAP/reconstruction without separate authorization.
