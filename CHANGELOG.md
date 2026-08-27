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
- Geometry/ML planning split into two bounded implementation plans: Step 6 geometry detection/analysis and combined Steps 7+8 SAM 2.1 segmentation + feature-mask analysis. The ML phase is limited to ten representative selected images, and pyCOLMAP/reconstruction tasks are excluded from the current implementation scope. Planning only; no new analysis outputs were produced.

### Fixed
- Fail preprocessing before creating generated outputs when the configured expected raw-image count disagrees with the verified baseline.
- Treat a degenerate OpenCV fundamental-matrix fit as zero geometric inliers instead of crashing the matching experiment.
- Correct stale phase/test-count documentation and use the cache-free pytest command in the reproduction steps.

### Removed
- Course-presentation DOCX/PDF walkthrough artifacts after delivery, while retaining all measured preprocessing reports, contact sheets, previews, and reconstruction-input evidence.

### Next
- When explicitly authorized, implement the Step 6 geometry plan first.
- After Step 6 verification, implement the combined Steps 7+8 SAM 2.1 segmentation and feature-mask analysis plan, then stop before pyCOLMAP.
