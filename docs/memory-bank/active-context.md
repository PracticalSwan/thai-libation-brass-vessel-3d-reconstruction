# Active Context

Updated: 2026-08-27

## Current focus

Preprocessing is complete and verified. The next implementation scope is now split into exactly two plans: Step 6 geometry detection/analysis, followed by combined Steps 7+8 SAM 2.1 segmentation and feature-mask analysis. The project intentionally stops after Step 8; pyCOLMAP/reconstruction is not currently planned for execution.

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

## Planned Step 6

- SIFT keypoints and candidate matches.
- Fundamental Matrix estimation and RANSAC inlier visualization.
- Epipolar-line visualization and Sampson/geometric residuals.
- Canny edges, contours, ellipse fitting where valid, centroid/bounding box, and principal/symmetry axis.
- Primary pair: 165-166; supporting low-feature pair: 255-256.
- Primary shape image: 165; supporting top-down/detail image: 255.
- Plan: `docs/superpowers/plans/2026-08-27-step-6-geometry-detection-analysis.md`.

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
- No geometry-extension scripts, SAM masks/model weights, or pyCOLMAP outputs exist yet.

## Next action

When implementation is explicitly authorized, execute the Step 6 plan first. After Step 6 is verified, execute the combined Steps 7+8 plan. Stop after Step 8 and report the geometry/ML evidence before planning any reconstruction work.
