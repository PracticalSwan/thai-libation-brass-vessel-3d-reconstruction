# Changelog

All notable verified project milestones are recorded here.

## Unreleased

### Added
- Public repository and shared contributor/AI-agent workflow.
- Real 297-image capture audit with readability, EXIF, quality metrics, hashes, duplicate checks, and contact-sheet review.
- Immutable raw-data policy and explicit cleanup/removal policy.
- Public versioning of the reviewed 297-image raw capture set and image-processing audit evidence, per user authorization.
- Final `quality_check.py`, `preprocess_images.py`, and `run_preprocessing.py` workflow with 21 focused tests.
- Dataset-relative `ACCEPT` / `WARN` / `REJECT` decisions: 207 / 81 / 9, retaining 288 images.
- Ten neighboring-pair RAW vs PREPROCESSED SIFT comparisons with fundamental-matrix RANSAC verification.
- PREPROCESSED reconstruction input selection from 2,483 verified inliers versus 2,376 for RAW, non-worse on 9 of 10 pairs, using the exact exported quality-95 JPEG encoding.
- Deterministic 288-image `preprocessing/pycolmap_input/images/` set and selection manifest.
- Ten before/after previews, four complete WARN/REJECT sheets, and SIFT inlier chart.
- Final raw integrity and output checks: 297/297 original hashes unchanged; 288/288 selected outputs readable at 3072 x 4080; no duplicate output hashes.
- Initial Geometry/ML planning split the work into Step 6 geometry detection/analysis and a combined Steps 7+8 segmentation + feature-mask phase, while keeping pyCOLMAP/reconstruction outside the current scope.
- Revised the Steps 7+8 architecture on 2026-09-01 from pretrained SAM inference to a small binary segmentation CNN trained from random initialization: initial 36 manual masks, sequence-aware 24/6/6 train/validation/test split, validation-based model selection, held-out Dice/IoU evaluation, and Step 8 reuse of Step 6 SIFT on CNN-predicted masks.
- Step 6 selected-input verifier and shared scale contract in `analysis_common.py`,
  including manifest-order lookup plus readability, dimensions, size, and
  SHA-256 checks for all 288 inputs before output creation.
- Step 6 SIFT/Fundamental Matrix implementation with BF-L2 ratio-test matches,
  strict RANSAC-mask handling, epipolar lines, and Sampson residual reports.
- Step 6 classical shape implementation with grayscale/Canny evidence,
  deterministic brass-color contour selection with Canny fallback, bounding
  box, centroid, PCA principal axis, and residual-gated optional ellipse fit.
- Deterministic `run_geometry_analysis.py` orchestration and
  `show_geometry_visuals.py` popup visualizer, backed by 31 focused Step 6
  tests and real-input smoke checks.
- Six visually inspected real Step 6 presentation figures and five
  machine-readable report files under `analysis/`, including complete source,
  configuration, runtime, measurement, and scope-exclusion provenance.
- Measured Step 6 evidence: pair 165-166 produced 478 candidates / 300 RANSAC
  inliers; pair 255-256 produced 57 / 18; primary-pair median Sampson error was
  0.1431 analysis pixels squared.
- Final Step 6 integrity proof: 288/288 selected images matched their manifest,
  and all 297 raw originals remained unchanged with zero size or SHA-256
  mismatches.

### Fixed
- Fail preprocessing before creating generated outputs when the configured expected raw-image count disagrees with the verified baseline.
- Treat a degenerate OpenCV fundamental-matrix fit as zero geometric inliers instead of crashing the matching experiment.
- Correct stale phase/test-count documentation and use the cache-free pytest command in the reproduction steps.

### Removed
- Course-presentation DOCX/PDF walkthrough artifacts after delivery, while retaining all measured preprocessing reports, contact sheets, previews, and reconstruction-input evidence.

### Next
- The CNN-based Steps 7+8 planning revision is complete. Before implementation, verify the current PyTorch runtime and freeze the exact 36-image annotation/split manifest; do not create labels, code, dependencies, or training outputs without separate authorization.
- Continue to stop before pyCOLMAP and reconstruction unless that later phase is explicitly authorized.
