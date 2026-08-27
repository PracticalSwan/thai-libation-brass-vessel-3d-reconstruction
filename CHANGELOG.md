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
- Ten before/after previews, four complete WARN/REJECT sheets, SIFT inlier chart, and professor-facing progress walkthrough.
- Final raw integrity and output checks: 297/297 original hashes unchanged; 288/288 selected outputs readable at 3072 x 4080; no duplicate output hashes.

### Fixed
- Fail preprocessing before creating generated outputs when the configured expected raw-image count disagrees with the verified baseline.
- Treat a degenerate OpenCV fundamental-matrix fit as zero geometric inliers instead of crashing the matching experiment.
- Correct stale phase/test-count documentation and use the cache-free pytest command in the reproduction steps.

### Next
- Begin pyCOLMAP feature extraction/matching and reconstruction in a later phase. No pyCOLMAP stage has run yet.
