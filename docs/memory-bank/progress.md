# Progress

Updated: 2026-09-05

## Completed and verified

### Preprocessing

- Published and audited the 297-image real capture set while preserving raw-file immutability.
- Final quality decisions are 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT`; all 288 non-rejected images remain in the selected set.
- The final PREPROCESSED variant uses the geometry-preserving 15% LAB-luminance CLAHE blend.
- Ten neighboring-pair RAW/PREPROCESSED comparisons produced 2,483 verified PREPROCESSED Fundamental-Matrix RANSAC inliers versus 2,376 RAW; PREPROCESSED was non-worse on 9/10 pairs.
- `preprocessing/pycolmap_input/images/` contains the deterministic 288-image next-stage input set.
- Final integrity verification found 0 raw mismatches across all 297 photographs and verified all 288 selected outputs against `selection_manifest.csv`.

### Step 6 — classical geometry

- Implemented `analysis_common.py`, `geometry_detection.py`, `shape_geometry.py`, `run_geometry_analysis.py`, and `show_geometry_visuals.py`.
- Pair 165-166: 4,653 / 4,643 SIFT keypoints, 478 candidates, 300 RANSAC inliers, 0.628 inlier ratio, median Sampson error 0.1431 px².
- Supporting pair 255-256: 57 candidates, 18 RANSAC inliers.
- Classical shape analysis retained contour/centroid/bounding-box/PCA evidence for images 165 and 255, rejected the weak global ellipse for 165, and retained the valid ellipse for 255.
- Six Step 6 presentation figures and machine-readable reports were generated and visually verified.

### Steps 7 + 8 — custom CNN segmentation and SIFT feature-mask analysis

- Frozen 36 reviewed source-resolution vessel masks under `ml_dataset/masks/`.
- Sequence-aware split: 24 train / 6 validation / 6 held-out test.
- Held-out indices: 72, 142, 165, 200, 255, 288.
- Label manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Annotation method: `opencv_assisted_visually_reviewed_bounded_correction`; CNN predictions were never used as labels.
- Implemented `segmentation_data.py`, `cnn_segmentation.py`, `train_cnn_segmentation.py`, `ml_feature_analysis.py`, and `run_ml_analysis.py` plus focused tests.
- `SmallSegCNN` is a project-defined compact U-Net-like network trained from random initialization with no pretrained weights/backbone and **487,297** trainable parameters.
- Fixed training configuration: 384 x 288 `(H x W)`, BCE-with-logits + soft Dice loss, Adam lr 1e-3, batch 8, seed 4213, threshold 0.5, validation-only checkpoint selection.
- Measured training environment: Python 3.14.2, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, CUDA 13.0, NVIDIA GeForce RTX 5050 Laptop GPU.
- Training completed 49 epochs in 332.112 s. Best epoch 39 reached validation Dice 0.968066 and IoU 0.938252. No training-only label expansion was needed.
- Frozen held-out aggregate: mean Dice 0.952521, median Dice 0.963377, mean IoU 0.910745, median IoU 0.929347, mean precision 0.930884, mean recall 0.976529.
- The held-out image-72 yellow-wall `background_false_positive` remains visible and unedited; index 200 retains a smaller `minor_boundary_error`. The other four test predictions are recorded as `ok`.
- Step 8 reuses `geometry_detection.extract_sift` and CNN-predicted masks: 28,673 total held-out SIFT keypoints, 27,431 inside predicted vessel masks, and 1,242 outside. Mean per-image vessel-feature fraction is 0.952693.
- Six ML presentation figures were generated from real outputs and visually inspected. The feature-mask figures retain the weak image-72 limitation instead of implying that high vessel-feature fractions prove reconstruction improvement.

### Step 9 — reconstruction readiness

- Implemented `reconstruction_masks.py`, `reconstruction_matching.py`, `camera_readiness.py`, and `run_reconstruction_readiness.py` with bounded stages `masks`, `benchmark`, `connectivity`, `camera`, and `summary`.
- Full-sequence inference produced 288 source-size raw CNN predictions and 288 deterministic cleanup masks. Cleanup changed 30 predictions; mean foreground fraction changed from 0.275260 to 0.274487.
- Frozen 20-pair x 3-mode geometry benchmark chose **unmasked** SIFT: 3,146 RANSAC inliers versus 2,841 for both masked modes. Both masked modes retained 90.31% of unmasked inliers and failed the fixed 95% qualification floor.
- Full 287-edge adjacent audit found 273 strong and 14 weak transitions. Fourteen local skip bridges were tested and none was strong, so the conservative recommended subset remains 288/288 images with zero exclusions.
- Camera/EXIF audit found one complete signature across all 288 selected filenames: OPPO Reno12 F, 3072 x 4080, orientation 1, 3.98 mm focal length, 26 mm 35-mm equivalent, digital zoom 1.0. The measured starting recommendation is one shared camera/intrinsics group.
- Four Step 9 presentation figures and machine-readable CSV/JSON evidence were generated and visually inspected. A zero-range camera-metadata plot defect found during review was fixed.
- `docs/geometry-ml/reconstruction-readiness.md` records the measured Step 9 method, results, limitations, and no-reconstruction boundary.

### Step 10 — sparse Structure from Motion

- Added `pycolmap>=4.2,<5`, `sparse_reconstruction.py`, `run_sparse_reconstruction.py`, and focused Step 10 tests.
- Final sparse-SIFT runtime used pyCOLMAP 4.2.0, CPU feature extraction, `max_image_size=1200`, native unmasked SIFT, one shared `SIMPLE_RADIAL` camera, and sequential matching.
- Baseline overlap 20: 1,255,153 SIFT features, 1,500 non-empty matched pairs, 902 verified pairs, 7 sparse models, 216 distinct images represented across those models. Largest model: 73/288 images, 6,099 points, 21,351 observations, mean track length 3.5007, mean reprojection error 1.2373 px.
- The single planned overlap-40 retry also produced 7 models; union coverage increased to 223 images but the largest model remained 73 images with 5,769 points.
- The frozen ranking rule selected the baseline 73-image component; it is exported under `reconstruction/sparse/best/` with `points3D.ply`.
- Both Step 10 figures were visually inspected. The selected local component has a coherent camera arc and plausible point cloud, but the full sequence remains fragmented, so `acceptance_met=false` and dense reconstruction was not started.
- `docs/geometry-ml/sparse-reconstruction.md` records the measured implementation, retry decision, fragmentation evidence, outputs, and boundary.

### Step 11 — sparse component bridging

- Refactored the Step 10 image-reader, feature-extraction, and incremental-mapping configuration into shared public helpers so Step 11 uses the same `SIMPLE_RADIAL`, CPU SIFT, and mapper contract.
- Implemented deterministic bridge candidate generation, pair-list safety, SQLite match summaries, qualification/selection, the fail-closed targeted gate, resumable feature/exhaustive databases, and durable CLI stages.
- Matched exactly 2,340 non-local diagnostic candidates: 780 around each of boundaries 73-74, 145-146, and 203-204.
- Boundaries 73-74 and 145-146 had zero geometrically verified candidates. Boundary 203-204 had 68 qualified candidates and 8 selected bridges, so the targeted mapper was skipped because all three boundaries did not qualify.
- Ran exactly one CPU exhaustive fallback with block size 50. Its database contains 14,900 non-empty match rows and 3,020 verified pair rows.
- Exhaustive mapping produced eight models with 224 distinct images across their disconnected union. The strongest single model registers 73/288 images with 3,443 points, 12,914 observations, mean track length 3.7508, and 1.1989 px mean reprojection error.
- The Step 11 candidate, sparse-model, and registration figures were visually inspected. They show the two empty bridge boundaries, the eight selected 203-204 bridges, a plausible local camera arc/point structure, and registration limited to indices 1-73.
- `reconstruction/bridging/best/`, `points3D.ply`, all component models, seven reports, and three figures preserve the measured result. `bridge_success=false`; dense reconstruction remains blocked.
- `docs/geometry-ml/sparse-component-bridging.md` records the method, interruption/resume provenance, measurements, visual review, acceptance decision, artifacts, and boundary.

## Verification

- Fresh Step 11-focused suite after review: **32 passed**.
- Fresh complete project suite after Step 11 review: **141 passed**.
- Changed sparse Python modules completed `python -B -m py_compile` successfully.
- Fresh final source-integrity verification: 297/297 raw unchanged with zero mismatches; 288/288 selected images verified against `selection_manifest.csv`.
- The Step 10 selected sparse model re-opened with pyCOLMAP 4.2.0 and exactly matched the summary metrics: 73 registered images, 6,099 sparse points, one camera, 1.2373052447638215 px mean reprojection error.
- The Step 11 selected model re-opened and exactly matched its summary metrics: 73 registered images, 3,443 points, one `SIMPLE_RADIAL` camera, 1.1988826674412258 px mean reprojection error.
- Review fixes now reject stale feature/resume databases whose per-image names, camera IDs, keypoint rows, or descriptor rows differ despite matching aggregate counts; the zero-inlier candidate figure also uses a nonnegative scale.
- Frozen Step 10 report hashes remained exactly unchanged after Step 11.
- Both final Step 10 figures were visually inspected; they explicitly identify the 73-image result as the selected component.
- Transient baseline/retry COLMAP databases and task-created caches were removed after model/report export; intentional sparse models, PLY, reports, figures, source code, spec, and plan remain.
- Final model checkpoint remains preserved locally at `analysis/ml/checkpoints/best_small_seg_cnn.pt` and is not intended for repository publication.
- No dense MVS, mesh, texture, or Blender artifact was created by Step 11.

## Local tooling

- Installed CodeGraph 1.6.0 and connected it only to Codex CLI and Claude Code.
- Initialized the repository graph under `.codegraph/` and verified an up-to-date index of 42 Python files, 994 nodes, and 2,649 edges.

## Next phase

Steps 6-11 are complete. Step 11 executed the authorized bounded recovery path but did not create a healthy global model: the selected component contains 73/288 images, and the 224-image exhaustive union is split across eight coordinate frames. Do not start dense reconstruction, meshing, texturing, or Blender from this state.

Any next phase requires an explicit decision among recapture, a materially different sparse strategy, or accepting a local-only deliverable. Keep CNN masks as analysis evidence; Step 9 already showed they reduce correspondence coverage, and Steps 10-11 did not use them for pyCOLMAP features.
