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

## Verification

- Fresh Step 9-focused suite: **26 passed**.
- Fresh complete project suite: **92 passed**.
- Changed Step 9 Python modules completed `python -B -m py_compile` successfully.
- Fresh final source-integrity verification: 297/297 raw unchanged with 0 mismatches; 288/288 selected images verified against `selection_manifest.csv`.
- Final mask-manifest verification re-opened and hash-checked all 288 raw predictions and 288 cleanup masks as source-size binary PNGs.
- Final model checkpoint is preserved locally at `analysis/ml/checkpoints/best_small_seg_cnn.pt` and is not intended for repository publication by default.
- Step 9 did not run pyCOLMAP/COLMAP or create camera poses, triangulated points, sparse/dense reconstructions, meshes, textures, or Blender outputs.

## Next phase

Steps 6-9 are complete. The repository deliberately stops before pyCOLMAP/reconstruction. Do not start camera-pose estimation, triangulation, sparse/dense reconstruction, meshing, texturing, or Blender until that later phase is explicitly authorized.

If reconstruction is authorized later, start from the unchanged 288-image PREPROCESSED set, use unmasked Step 6 SIFT as the current evidence-backed matching baseline, begin from one shared camera/intrinsics group, and validate those choices with actual SfM results. The Step 9 benchmark specifically found that the CNN-mask variants reduce correspondence coverage, so they should remain analysis evidence unless a later reconstruction experiment demonstrates a benefit.
