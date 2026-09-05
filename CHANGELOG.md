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
- Frozen Steps 7+8 segmentation dataset with 36 reviewed source-size masks and a sequence-aware 24 train / 6 validation / 6 held-out test split. The label manifest SHA-256 is `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Project-defined `SmallSegCNN` binary segmenter trained from random initialization with 487,297 trainable parameters and no pretrained weights/backbone.
- Measured baseline training on CUDA: 49 epochs, best epoch 39, validation Dice 0.9681 / IoU 0.9383, 332.1 s runtime on the NVIDIA GeForce RTX 5050 Laptop GPU.
- Frozen six-image held-out evaluation with mean Dice 0.9525 and mean IoU 0.9107; the image-72 yellow-wall `background_false_positive` remains visible and unedited in the evidence.
- Step 8 feature-mask analysis reusing `geometry_detection.extract_sift`: 28,673 held-out SIFT keypoints, 27,431 inside CNN-predicted vessel masks and 1,242 outside.
- Six ML presentation figures plus machine-readable training, held-out, summary, and masked-feature reports generated from real outputs and visually reviewed.
- Final Steps 7+8 verification: 13 ML-focused tests and 66 complete project tests passed; changed Python compiled; all 297 raw photographs and 288 selected images remained unchanged.
- Step 9 reconstruction-readiness design and four implementation plans covering full-sequence CNN inference, masked-vs-unmasked geometry benchmarking, full-sequence connectivity/subset analysis, and camera/EXIF readiness.
- Step 9A full-sequence inference: 288 frozen source-size CNN predictions plus 288 deterministic connected-component cleanup masks; cleanup changed 30 predictions while preserving connected failure cases such as the known index-72 yellow-wall false positive.
- Step 9B frozen 20-pair x 3-mode geometry benchmark: unmasked SIFT produced 3,146 RANSAC inliers versus 2,841 for both masked modes. Each masked mode retained only 90.31% of unmasked inliers and failed the fixed 95% qualification floor, so unmasked SIFT is the measured readiness baseline.
- Step 9C full-sequence connectivity audit: 287 adjacent edges, 273 strong and 14 weak; 14 local skip bridges tested, zero strong. Conservative subset remains all 288 selected images with zero exclusions.
- Step 9D raw-EXIF audit: one complete camera signature across all 288 selected filenames (OPPO Reno12 F, 3072 x 4080, orientation 1, 3.98 mm focal length, 26 mm 35-mm equivalent, digital zoom 1.0), supporting one shared camera/intrinsics group as the later SfM starting recommendation.
- Step 9 machine-readable evidence, 288-image inclusion manifest, and four visually inspected presentation figures under `analysis/` and `preprocessing/reconstruction_input_v1/`.
- Final Step 9 verification: 26 focused tests and 92 complete project tests passed; changed Step 9 Python compiled; 297/297 raw files remained hash-identical, 288/288 selected images matched their manifest, and all 576 generated Step 9 masks re-opened as hash-matching source-size binary PNGs.
- Step 10 sparse-SfM design, implementation plan, `sparse_reconstruction.py`, `run_sparse_reconstruction.py`, focused tests, and `pycolmap>=4.2,<5` dependency.
- Real pyCOLMAP 4.2.0 CPU sparse run over all 288 PREPROCESSED inputs using native unmasked SIFT, one shared `SIMPLE_RADIAL` camera, internal `max_image_size=1200`, sequential matching, and incremental mapping.
- Step 10 baseline overlap 20: 1,255,153 SIFT features, 1,500 non-empty matched pairs, 902 verified pairs, seven sparse models, 216-image union coverage; largest component 73 images / 6,099 points / 1.2373 px mean reprojection error.
- Single controlled overlap-40 retry: seven sparse models, 223-image union coverage; largest component remained 73 images / 5,769 points, so the frozen ranking retained the baseline component.
- Selected sparse COLMAP model plus PLY export under `reconstruction/sparse/best/`, machine-readable Step 10 reports, and two visually inspected sparse/registration figures.
- Step 10 records `acceptance_met=false`: the local sparse reconstruction is plausible, but the full sequence remains fragmented, so dense reconstruction was not started and no success claim was made for a global 288-image model.
- Final Step 10 verification: 11 focused tests and 103 complete project tests passed; changed Step 10 Python compiled; the selected sparse model re-opened with matching summary metrics; 297/297 raw files and 288/288 selected files remained verified; transient COLMAP databases/caches were removed while preserving the sparse models and evidence.
- Step 11 sparse-component-bridging design, implementation plan, shared Step 10 pyCOLMAP option/runtime helpers, `sparse_bridging.py`, `run_sparse_bridging.py`, and focused tests.
- Deterministic non-local diagnosis of exactly 2,340 pairs: 780 around each fixed boundary 73-74, 145-146, and 203-204. The first two boundaries produced zero geometrically verified candidates; 203-204 produced 68 qualified candidates and 8 selected bridges.
- Fail-closed targeted gate: targeted mapping was skipped because every boundary required at least one selected qualified bridge.
- Exactly one CPU exhaustive fallback with block size 50: 14,900 non-empty match rows, 3,020 geometrically verified rows, eight sparse models, and 224-image union coverage.
- Selected Step 11 sparse model plus PLY under `reconstruction/bridging/best/`, seven machine-readable reports, and three visually inspected candidate/sparse/registration figures.
- Step 11 records `bridge_success=false`: the strongest single model remains 73/288 images with 3,443 points and 1.1989 px mean reprojection error, so disconnected-model union coverage is not misreported as a global reconstruction and dense work remains blocked.
- Final Step 11 review verification: 32 focused tests and 141 complete project tests passed; changed sparse modules compiled; both Step 10 and Step 11 models re-opened with matching metrics; 297/297 raw files and 288/288 selected files remained verified.
- Post-implementation review hardened feature-cache and interrupted exhaustive-resume identity validation using exact image/camera/keypoint/descriptor layout, and fixed the zero-inlier bridge-candidate figure so its scale cannot display impossible negative inlier values. These fixes do not change the measured Step 11 reconstruction result.

### Fixed
- Fail preprocessing before creating generated outputs when the configured expected raw-image count disagrees with the verified baseline.
- Treat a degenerate OpenCV fundamental-matrix fit as zero geometric inliers instead of crashing the matching experiment.
- Correct stale phase/test-count documentation and use the cache-free pytest command in the reproduction steps.

### Removed
- Course-presentation DOCX/PDF walkthrough artifacts after delivery, while retaining all measured preprocessing reports, contact sheets, previews, and reconstruction-input evidence.

### Next
- Steps 6-11 are complete and verified. The bounded Step 11 recovery path did not create a healthy global sparse model; do not start dense MVS from the selected 73-image component or the 224-image disconnected union.
- Any next phase requires a separately authorized decision among recapture, a materially different sparse strategy, or accepting a local-only deliverable. Step 11 does not authorize any of them.
- Keep the CNN masks as analysis evidence rather than assumed reconstruction inputs; Step 9 measured lower correspondence coverage for both masked modes, and Steps 10-11 used unmasked native pyCOLMAP SIFT.
