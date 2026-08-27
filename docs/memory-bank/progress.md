# Progress

Updated: 2026-08-27

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
- Geometry/ML planning is now split into two implementation plans: Step 6 geometry detection/analysis and combined Steps 7+8 SAM 2.1 segmentation + feature-mask analysis.
- Step 6 plans SIFT/RANSAC matches, epipolar geometry, and classical 2D vessel-shape geometry with explicit presentation figures.
- Steps 7+8 plan SAM 2.1 segmentation on ten representative selected images and SIFT feature counts inside versus outside the vessel masks; full 288-image masking is deliberately deferred.
- The previous combined plan is now an index only. pyCOLMAP/reconstruction tasks were removed from the current implementation scope. This remains documentation only; no geometry scripts, SAM masks, model weights, or reconstruction outputs were created.

## Next phase

- When authorized, implement Step 6 first and verify its geometry figures/results.
- Then implement the combined Steps 7+8 plan and verify its segmentation/feature-mask figures/results.
- Stop after Step 8. No pyCOLMAP, sparse reconstruction, dense reconstruction, meshing, texturing, or Blender work is currently planned for execution.
