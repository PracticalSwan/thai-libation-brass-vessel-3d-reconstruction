# Active Context

Updated: 2026-09-06

## Current focus

Preprocessing and Steps 6-13 are complete and verified. Step 13 used the separately authorized external ALIKED-N16Rot + LightGlue runtime on CUDA, recovered all three exact Step 11 boundaries, and produced a strongest single sparse model with 266/288 images, 29,713 points, and 1.374824 px mean reprojection error. The frozen global acceptance gate required at least 274 images, so `step13_success=false` and sparse recovery is closed. The 266-image model is retained as evidence, while the frozen fallback ranking selects Step 10 `reconstruction/sparse/best` (73 images / 6,099 points) as the downstream sparse source. Dense reconstruction has not started; the next phase must explicitly design/authorize local-only dense reconstruction from that selected local model.

## Verified preprocessing state

- Raw source: `IMG20260826122949/`, 297 immutable JPEG files at 3072 x 4080.
- Final decisions: 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT`.
- Rejected images: indices 289-297 only, the separate hand-held/flipped sequence.
- Selected set: all 288 `ACCEPT` + `WARN` images.
- Selected variant: PREPROCESSED, using a geometry-preserving 15% LAB-luminance CLAHE blend.
- Matching evidence: 2,483 PREPROCESSED versus 2,376 RAW fundamental-matrix RANSAC inliers over ten representative neighboring pairs; PREPROCESSED was non-worse on 9/10.
- Final selected input: `preprocessing/pycolmap_input/images/`.
- Fresh 2026-09-05 integrity check: 297/297 raw files unchanged and 288/288 selected files verified against `selection_manifest.csv`.

## Completed Step 6

- `analysis_common.py` provides deterministic selected-manifest access and source integrity verification.
- `geometry_detection.py` provides scaled SIFT, BF-L2 ratio matching, Fundamental Matrix/RANSAC, epilines, Sampson residuals, and explicit original/analysis scale metadata.
- `shape_geometry.py` provides grayscale/Canny evidence, classical contour selection, box/centroid/PCA axis, and residual-gated optional ellipse fitting.
- Primary pair 165-166: 4,653 / 4,643 keypoints, 478 candidates, 300 RANSAC inliers, 0.628 inlier ratio, median Sampson error 0.1431 px².
- Supporting pair 255-256: 57 candidates / 18 inliers.
- Six Step 6 presentation figures were generated and visually verified.
- Measured results: `docs/geometry-ml/geometry-results.md`.

## Completed Steps 7 + 8

### Frozen segmentation dataset

- 36 reviewed source-size binary masks under `ml_dataset/masks/`.
- Sequence-aware split: 24 train / 6 validation / 6 held-out test.
- Test indices: 72, 142, 165, 200, 255, 288.
- Label manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Annotation method: `opencv_assisted_visually_reviewed_bounded_correction`.
- No CNN predictions were used as labels; no optional training-only expansion was needed.
- Dataset record: `docs/geometry-ml/cnn-dataset.md`.

### SmallSegCNN training

- Project-defined compact U-Net-like CNN trained from random initialization with no pretrained weights/backbone.
- Actual trainable parameters: 487,297.
- Input: 384 x 288 `(H x W)`; BCE-with-logits + Dice loss; Adam lr 1e-3; batch 8; seed 4213; threshold 0.5.
- Training environment: Python 3.14.2, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 5050 Laptop GPU.
- 49 epochs completed; best validation epoch 39; runtime 332.112 s.
- Best validation Dice 0.968066; best validation IoU 0.938252.
- Final checkpoint remains local at `analysis/ml/checkpoints/best_small_seg_cnn.pt` by default.

### Frozen held-out evaluation

- Mean Dice 0.952521; median Dice 0.963377.
- Mean IoU 0.910745; median IoU 0.929347.
- Mean precision 0.930884; mean recall 0.976529.
- Index 72 is the retained weak case: `background_false_positive` caused by the yellow classroom wall/background.
- Index 200 has `minor_boundary_error`; the other four are recorded as `ok`.
- Test predictions were not manually repaired and the model was not tuned after held-out inspection.

### Step 8 feature-mask analysis

- Primary analysis reuses `geometry_detection.extract_sift`; there is no duplicate SIFT pipeline.
- Primary masks are CNN-predicted held-out masks, not ground truth.
- Across six tests: 28,673 SIFT keypoints; 27,431 inside predicted vessel masks; 1,242 outside.
- Mean per-image vessel feature fraction: 0.952693.
- These are descriptive counts only and do not prove reconstruction improvement.
- Measured results: `docs/geometry-ml/ml-results.md`.

## Completed Step 9 — reconstruction readiness

### 9A full-sequence inference

- Reused the frozen `SmallSegCNN` checkpoint without retraining or changing the 0.5 threshold.
- Generated 288 unedited full-sequence predictions under `analysis/ml/full_predictions/`.
- Generated 288 deterministic connected-component cleanup masks under `analysis/ml/reconstruction_masks/`.
- Cleanup changed 30 predictions; mean foreground fraction changed from 0.275260 to 0.274487.
- The cleanup is intentionally conservative and does not remove false-positive regions that remain connected to the predicted vessel; the index-72 yellow-wall limitation remains visible.

### 9B masked versus unmasked geometry benchmark

- Frozen benchmark: 20 representative pairs x 3 feature modes using the existing Step 6 SIFT/Fundamental-Matrix stack.
- `unmasked`: 5,344 candidates, 3,146 RANSAC inliers, median inlier ratio 0.506391, median Sampson error 0.133932, median grid coverage 0.625.
- `raw_cnn`: 4,602 candidates, 2,841 inliers, median ratio 0.501026, median Sampson error 0.121050, median grid coverage 0.500.
- `reconstruction_mask`: identical aggregate result to `raw_cnn` on the frozen pairs.
- Both masked modes retained only 90.31% of unmasked inliers and failed the fixed 95% qualification floor.
- Frozen Step 9 recommendation: **unmasked Step 6 SIFT** for later reconstruction preparation.

### 9C full-sequence connectivity

- Evaluated all 287 adjacent selected-image transitions with the frozen unmasked feature mode.
- Strong adjacent edges: 273; weak adjacent edges: 14.
- Tested 14 local skip bridges around weak transitions; strong skip bridges: 0.
- Conservative subset decision: include 288/288 images; excluded count: 0.
- `preprocessing/reconstruction_input_v1/manifest.csv` records all include decisions and references the existing selected JPEGs instead of duplicating them.

### 9D camera readiness

- Audited raw EXIF for all 288 selected filenames.
- One complete camera signature across all 288: OPPO Reno12 F, 3072 x 4080, orientation 1, focal length 3.98 mm, 35-mm equivalent 26 mm, digital zoom 1.0, no missing recorded camera-readiness fields.
- Starting recommendation for later SfM: one shared camera/intrinsics group, to be validated by actual reconstruction behavior.
- No calibration, undistortion, image resampling, or reconstruction was performed.

## Completed Step 10 — sparse SfM

- Added `pycolmap>=4.2,<5`; measured runtime used pyCOLMAP 4.2.0 on Windows.
- Windows pyCOLMAP wheel exposed CPU-only SIFT, so the final internal sparse-feature limit is 1200 pixels; the 3072 x 4080 source JPEGs remain unchanged.
- Camera mode: one shared `SIMPLE_RADIAL` camera initialized from the Step 9 26 mm 35-mm-equivalent evidence at `f=3069.0507 px`, center `(1536, 2040)`, `k=0`.
- Baseline sequential overlap 20: 1,255,153 SIFT features, 1,500 non-empty matched pairs, 902 verified pairs, 7 sparse models, 216-image union coverage. Largest model: 73/288 images, 6,099 points, 21,351 observations, mean track 3.5007, mean reprojection error 1.2373 px.
- One controlled overlap-40 retry: 7 sparse models, 223-image union coverage. Largest model again 73 images with 5,769 points, so the frozen ranking selected the baseline component.
- Selected output: `reconstruction/sparse/best/` plus `points3D.ply`.
- Visual review found a coherent local camera arc and plausible point cloud, but the fixed >=274-image global acceptance target was not met; `step10_summary.json` records `acceptance_met=false`.
- Large component boundaries are consistent with earlier Step 9 weak transitions at 73-74, 145-146, and 203-204; this is evidence of fragmentation, not proof of a single cause.
- Measured narrative: `docs/geometry-ml/sparse-reconstruction.md`.

## Completed Step 11 — sparse component bridging

- Added shared public Step 10 pyCOLMAP option/extraction/mapping helpers, `sparse_bridging.py`, `run_sparse_bridging.py`, and focused orchestration/contract tests.
- Diagnosed exactly 2,340 deterministic non-local candidate pairs: 780 around each fixed boundary 73-74, 145-146, and 203-204.
- Boundaries 73-74 and 145-146 produced zero geometrically verified candidates. Boundary 203-204 produced 68 qualified candidates and 8 selected bridges.
- Targeted mapping was skipped by the frozen fail-closed gate because every boundary required at least one selected qualified bridge.
- The one authorized CPU exhaustive fallback used block size 50 and produced 14,900 non-empty raw-match rows, 3,020 geometrically verified rows, eight sparse models, and 224-image union coverage.
- The selected single model registers 73/288 images with 3,443 points, 12,914 observations, mean track length 3.7508, 1.1989 px mean reprojection error, and one shared `SIMPLE_RADIAL` camera.
- Visual review found a smooth local camera arc and plausible local point structure, but incomplete coverage and outliers; the registration figure confirms only indices 1-73 are registered.
- `step11_summary.json` records `bridge_success=false` and `dense_reconstruction_started=false`. The disconnected 224-image union is diagnostic evidence, not a global model.
- Measured narrative: `docs/geometry-ml/sparse-component-bridging.md`.

## Completed Step 12 — learned sparse recovery capability boundary

- Implemented `learned_sparse_recovery.py`, `run_learned_sparse_recovery.py`, and focused domain/runner tests using only ALIKED-N16Rot + ALIKED-LightGlue and the conditional LoMa-B + LoMa-L fallback.
- The deterministic boundary preserves explicit CPU learned matcher options for both sequential and imported matching, exact Step 11 candidate identity, strict per-image feature-cache layout fingerprints, strongest-single-model acceptance, bounded cleanup, and no learned exhaustive or dense API.
- Python 3.14.2 / pyCOLMAP 4.2.0 exposed the required learned enums and valid CPU option objects with `max_image_size=1600`; `pycolmap.has_cuda=false`.
- The real ALIKED smoke on `IMG20260826122949.jpg` and `IMG20260826122953.jpg` failed before extraction because the installed wheel lacks ONNX support. The exact exception is preserved in `step12_capability.json` and `step12_aliked_attempt.json`.
- The frozen gate stopped 288-image ALIKED extraction and all learned diagnostics/mapping. LoMa stayed `not_run` because it cannot be used as a silent bypass for a missing native learned-runtime prerequisite.
- Final reports record no selected frontend/model, `metric_acceptance_met=false`, `visual_plausibility_status=failed`, `learned_recovery_success=false`, and `dense_reconstruction_started=false`.
- Only the strongest-single-model SIFT comparison figure was applicable; it was visually inspected and contains no fabricated learned result. Measured narrative: `docs/geometry-ml/learned-sparse-recovery.md`.

## Completed Step 13 — external learned global recovery

- Added one official CVG LightGlue dependency pinned to commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`; no existing Torch, torchvision, pyCOLMAP, NumPy, or CUDA package was upgraded/downgraded for Step 13.
- Real CUDA capability smoke succeeded with 4,096 ALIKED-N16Rot features per smoke image and 2,952 raw LightGlue matches.
- Reused the exact Step 11 2,340-pair candidate identities and unchanged bridge thresholds. Learned matching produced 778 qualified candidates at 73-74, 418 at 145-146, and 745 at 203-204, with 8 selected bridges per boundary.
- Diagnostic accounting records 363,318 imported LightGlue matches before verification and 363,171 COLMAP match rows after verification; 147 were removed by geometric verification.
- One full mapping schedule contained 5,550 sequential overlap-20 pairs plus 24 selected learned bridges = 5,574 unique pairs. It produced 5,269,937 raw LightGlue matches; the verified mapping DB contained 288 images, 1,129,555 keypoints, 5,531 non-empty match rows, and 5,264 verified pairs.
- pyCOLMAP produced three sparse models. The strongest registered 266/288 images with 29,713 points, 106,480 observations, mean track length 3.5836, one `SIMPLE_RADIAL` camera, and 1.374824 px mean reprojection error.
- The strongest model crosses the prior 73-74, 145-146, and 203-204 breaks but leaves selected indices 267-288 unregistered. It missed the frozen >=274 global gate by 8 images, so `metric_acceptance_met=false`, `visual_plausibility_status=failed`, and `step13_success=false`.
- The 266-image model is preserved under `reconstruction/external_learned_recovery/best/` as evidence. Finalization re-ranked the existing local candidates and selected Step 10 `reconstruction/sparse/best` as the downstream source because Steps 10 and 11 both register 73 images and Step 10 has 6,099 points versus 3,443.
- Four real Step 13 figures were opened and reviewed. The sparse view exposes camera/geometry outliers rather than hiding them, the registration view shows the 267-288 tail gap, and the model comparison shows 266 below the 274 gate.
- Step 13 is the final sparse-recovery experiment. No retry, second learned frontend, parameter sweep, learned exhaustive matching, dense MVS, meshing, texturing, or Blender work was run.
- Measured narrative: `docs/geometry-ml/external-learned-global-recovery.md`.

## Verification and evidence

- Step 11-focused suite after review: **32 passed**.
- Fresh complete project suite after Step 11 review: **141 passed**.
- Step 12 maintenance boundary: **20 domain tests passed**, **14 runner tests passed**, and **49 Step 10/11 regression tests passed** after the fixes.
- The runtime-specific LoMa `not_run` reason regression first raised the Step 12 runner suite to **15 passing tests**; final review added a blocked-attempt frontend-identity regression, bringing it to **16 passing tests**.
- Final complete project verification after Step 12 review: **177 tests passed**; all four Step 12 source/test files compiled successfully.
- Final Step 13 verification: **21 focused tests passed** and **198 complete project tests passed**. The four Step 13 source/test files compiled successfully and their generated compile-cache residue was removed.
- Final Step 13 integrity verification found 297/297 raw images unchanged and 288/288 selected inputs matching the frozen manifest; all five protected Step 10/11 report hashes matched their pre-Step-13 values.
- The retained Step 13 evidence model reopened at 266 registered images, 29,713 points, one `SIMPLE_RADIAL` camera, and 1.374823762049934 px mean reprojection error. The selected downstream Step 10 model reopened at 73 images / 6,099 points / 1.2373052447638215 px.
- Step 13 cleanup removed its feature cache, diagnostic/mapping databases, pair lists, mapper workspace, and `work/` directory while preserving reports, the evidence model/PLY, and four figures.
- Step 10/11 selected models reopened with their recorded metrics, and five protected Step 10/11 report hashes exactly matched the pre-Step-12 snapshots.
- Final integrity verification again found 297/297 raw images unchanged and 288/288 selected images matching the frozen manifest.
- Step 12 cleanup left no learned transient database, `work/` directory, learned `best/` model, PLY, or compile cache.
- Syntax compilation succeeded for `learned_sparse_recovery.py`, `run_learned_sparse_recovery.py`, and both Step 12 test files.
- `sparse_reconstruction.py`, `sparse_bridging.py`, `run_sparse_reconstruction.py`, and `run_sparse_bridging.py` compile successfully with `python -B -m py_compile`.
- Fresh protected-source verification: 297/297 raw unchanged with zero mismatches; 288/288 selected images verified against `selection_manifest.csv`.
- The selected Step 10 sparse model re-opened with pyCOLMAP 4.2.0 after finalization and matched `step10_summary.json`: 73 registered images, 6,099 points, one camera, 1.2373052447638215 px mean reprojection error.
- The selected Step 11 model re-opened and matched `step11_summary.json`: 73 registered images, 3,443 points, one `SIMPLE_RADIAL` camera, 1.1988826674412258 px mean reprojection error.
- Post-implementation review hardened feature-cache/exhaustive-resume identity validation and corrected the zero-inlier candidate-figure scale; neither change alters the measured Step 11 SfM result.
- The three Step 10 JSON report hashes exactly matched their pre-Step-11 snapshots.
- Both final Step 10 figures were visually inspected and explicitly label the 73-image output as the selected component rather than a global 288-image reconstruction.
- Transient ~195 MB COLMAP databases from baseline/retry and task-created Python caches were removed after model/report export; the sparse component models, selected model, PLY, reports, and figures were preserved.
- Dense/MVS/mesh/texture/Blender work was not started.
- Step 9 measured evidence remains preserved under `analysis/`; Step 10 and Step 11 evidence is under `reconstruction/` and documented in the two sparse-result reports.

## Local tooling

- CodeGraph 1.6.0 is installed and this repository is initialized at `.codegraph/`; the maintenance review confirmed the project index is available and current.
- Use CodeGraph when dependency, call-path, architecture, or change-impact analysis materially helps. Preserve `.codegraph/`; direct inspection is preferred for trivial edits.

## Next action

Step 13 is complete and closes sparse-recovery experimentation. The external learned run improved the strongest single model to 266/288 images but did not meet the frozen >=274 global acceptance gate. The selected downstream sparse source is Step 10 `reconstruction/sparse/best` (73 images / 6,099 points); the Step 13 266-image model remains evidence only. The next task is to create and approve a local-only dense-reconstruction design/acceptance plan for that selected Step 10 component. Only after that new phase is authorized should the project proceed to undistortion/dense preparation, dense stereo/fusion, meshing, texturing, Blender cleanup, final model validation, and final coursework/report/presentation packaging.
