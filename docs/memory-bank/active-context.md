# Active Context

Updated: 2026-09-05

## Current focus

Preprocessing, Steps 6-9 analysis, and Step 10 pyCOLMAP sparse SfM are complete and verified. Step 10 produced valid local sparse reconstructions but the 288-image sequence fragmented into seven models; the selected component registers 73 images. The current project boundary is a deliberate stop before dense reconstruction. Do not start dense MVS, meshing, texturing, or Blender until a separately authorized sparse-component recovery/acceptance phase resolves this limitation.

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

## Verification and evidence

- Step 10-focused suite: **11 passed**.
- Fresh complete project suite after Step 10: **103 passed**.
- `sparse_reconstruction.py` and `run_sparse_reconstruction.py` compile successfully with `python -B -m py_compile`.
- Fresh protected-source verification: 297/297 raw unchanged with zero mismatches; 288/288 selected images verified against `selection_manifest.csv`.
- The selected sparse model re-opened with pyCOLMAP 4.2.0 after finalization and matched `step10_summary.json`: 73 registered images, 6,099 points, one camera, 1.2373052447638215 px mean reprojection error.
- Both final Step 10 figures were visually inspected and explicitly label the 73-image output as the selected component rather than a global 288-image reconstruction.
- Transient ~195 MB COLMAP databases from baseline/retry and task-created Python caches were removed after model/report export; the sparse component models, selected model, PLY, reports, and figures were preserved.
- Dense/MVS/mesh/texture/Blender work was not started.
- Step 9 measured evidence remains preserved under `analysis/`; Step 10 evidence is under `reconstruction/` and documented in `docs/geometry-ml/sparse-reconstruction.md`.

## Next action

Step 10 execution is complete, but its healthy-single-model acceptance gate failed because the sequence remains fragmented. Do not start dense reconstruction yet. The next recommended separately authorized phase is a narrow sparse-component bridging investigation around the measured component boundaries, using the preserved pyCOLMAP sparse models and Step 9 connectivity evidence rather than adding broad parameter sweeps or hiding the fragmentation.
