# Progress

Updated: 2026-09-01

## Completed and verified

- Public repository created and published at `PracticalSwan/thai-libation-brass-vessel-3d-reconstruction`.
- All 297 raw photographs and the original audit evidence published while preserving raw immutability.
- Full read-only audit and ten contact-sheet review completed over the 297 raw captures.
- Final three-script preprocessing architecture implemented:
  - `quality_check.py` — standardized metrics, dataset-relative thresholds, and decisions;
  - `preprocess_images.py` — deterministic 15% LAB-luminance CLAHE blend with unchanged geometry;
  - `run_preprocessing.py` — raw verification, orchestration, reports, previews, SIFT comparison, and selected-set export.
- Twenty-one focused tests pass, including synthetic sharp/blur behavior, clipping metrics, unreadable inputs, warning semantics, deterministic geometry-preserving preprocessing, raw-manifest mismatch detection, preflight count/path safety, degenerate geometric-fit handling, exact JPEG matching parity, SIFT geometric verification, and a complete miniature pipeline.
- Full real run completed over all 297 photographs.
- Final decisions: 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT`; all 288 non-rejected images retained.
- Rejections restricted to images 289-297, the visually confirmed hand-held/flipped and hand-occluded sequence.
- Ten neighboring-pair RAW/PREPROCESSED comparisons completed with SIFT, BF-L2 matching, 0.75 ratio test, and fundamental-matrix RANSAC.
- PREPROCESSED selected with 2,483 verified inliers versus 2,376 for RAW and non-worse results on 9 of 10 pairs after comparing decoded quality-95 JPEG bytes identical to the final export encoding.
- Final 288-image pyCOLMAP-ready set created in `preprocessing/pycolmap_input/images/`.
- Independent integrity audit reopened all 288 outputs at 3072 x 4080, matched every selection-manifest hash, found no output duplicates, and re-hashed all 297 originals with zero mismatches.
- All four WARN/REJECT decision sheets and all ten before/after previews visually inspected.
- Course-presentation walkthrough artifacts were delivered and later removed from the workspace after user approval; processing evidence remains under `preprocessing/` and `docs/preprocessing/`.
- Geometry/ML planning is split into two implementation plans: completed Step 6 geometry detection/analysis and planned Steps 7+8 custom CNN segmentation + feature-mask analysis.
- Step 6 covers SIFT/RANSAC matches, epipolar geometry, and classical 2D vessel-shape geometry with explicit presentation figures.
- On 2026-09-01, the Steps 7+8 architecture was revised from pretrained SAM inference to a small binary segmentation CNN trained from random initialization.
- The revised ML plan starts with 36 manual masks using a sequence-aware 24 train / 6 validation / 6 held-out test split, with at most 12 later training-only labels if validation shows a real coverage gap.
- The planned CNN is a compact U-Net-like `SmallSegCNN`, preferred input size 384 x 288 `(H x W)`, BCE-with-logits + Dice loss, Adam, seed 4213, validation-based early stopping, and no test-set tuning.
- Held-out predictions will be evaluated with Dice, IoU, foreground precision/recall and visible failure analysis before Step 8 reuses Step 6 SIFT to count features inside versus outside CNN-predicted vessel masks.
- The combined plan index still stops before pyCOLMAP/reconstruction; no CNN labels, CNN model weights, ML predictions, or reconstruction outputs were created by this planning revision.
- Step 6 remains independent of the future CNN and exposes only the selected-record and SIFT-scale interfaces required downstream.
- Step 6 implemented in `analysis_common.py`, `geometry_detection.py`,
  `shape_geometry.py`, `run_geometry_analysis.py`, and
  `show_geometry_visuals.py` with 31 focused unit/integration tests.
- The real Step 6 orchestrator verified all 288 selected inputs, then generated
  11 intentional artifacts: six presentation figures and five JSON/CSV reports.
- Real pair 165-166 produced 478 Lowe-ratio candidates and 300 Fundamental
  Matrix RANSAC inliers (0.628 ratio); real supporting pair 255-256 produced 57
  candidates and 18 inliers (0.316 ratio).
- Primary epipolar evidence reported median/p90 Sampson errors of 0.1431/0.7594
  analysis pixels squared and explicitly excluded pose or reconstruction claims.
- Classical shape analysis retained contour, centroid, bounding box, and PCA
  axis for both representative images. It rejected the misleading global
  side-view ellipse for image 165 and retained the valid top-down/detail ellipse
  for image 255.
- All six final presentation figures were visually inspected. A clipped
  epipolar header and the weak image-165 ellipse were found during review,
  corrected, regenerated, and inspected again.
- The real popup visualizer and no-display smoke path both completed. Final
  integrity checks confirmed all 297 raw photographs unchanged and all 288
  selected images still matching the manifest.

## Next phase

- The CNN-based Steps 7+8 plan revision is complete. Implementation still requires separate authorization.
- The first implementation action, when authorized, is to verify the current PyTorch environment and select/freeze the 36-image manual-label set before creating masks or model code.
- Stop after Step 8. No pyCOLMAP, sparse reconstruction, dense reconstruction, meshing, texturing, or Blender work is currently planned for execution.
