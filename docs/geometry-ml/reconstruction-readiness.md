# Step 9 Reconstruction Readiness

Updated: 2026-09-05

## Status

Step 9 is complete and verified. It prepared and measured reconstruction-readiness evidence but deliberately stopped before pyCOLMAP, camera-pose estimation, triangulation, sparse/dense reconstruction, meshing, texturing, or Blender.

## Inputs and provenance

- Raw source: `IMG20260826122949/`, 297 immutable 3072 x 4080 JPEGs.
- Verified selected source: `preprocessing/pycolmap_input/images/`, 288 PREPROCESSED JPEGs.
- Selection manifest SHA-256: `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`.
- Frozen segmentation manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Frozen local checkpoint SHA-256: `de2009bfcd00307fa64176cd41b66f5c9e5ac97ab1e8e08226de74df928f526b`.
- Model: project-defined `SmallSegCNN`, 487,297 trainable parameters, random initialization, no pretrained weights, fixed threshold 0.5.
- Existing Step 6 geometry path reused: `geometry_detection.extract_sift`, BF-L2 ratio matching, Fundamental Matrix RANSAC, Sampson residuals, and explicit original/analysis scale metadata.

## 9A — Full-sequence CNN inference

The frozen checkpoint was run over all 288 verified selected images without retraining or threshold changes.

Outputs:

- `analysis/ml/full_predictions/`: 288 unedited source-size CNN predictions.
- `analysis/ml/reconstruction_masks/`: 288 deterministic cleanup masks.
- `analysis/reports/reconstruction_mask_manifest.csv`: dimensions, hashes, paths, and foreground fractions for every mask pair.
- `analysis/reports/step9_masks.json`: stage summary.
- `analysis/previews/presentation/step9_01_reconstruction_masks.png`: visually reviewed samples across the capture sequence.

The cleanup operates only on predicted connected components. It anchors the largest component intersecting the central ROI, retains only sufficiently large nearby secondary components, removes detached fragments, and does not fill holes or erode the vessel silhouette. It changed 30 of 288 predictions. Mean foreground fraction changed from 0.275260 to 0.274487.

This cleanup is intentionally conservative. It does not remove a false-positive region that is connected to the predicted vessel. The known held-out index-72 yellow-wall false positive therefore remains visible rather than being manually repaired or hidden. The CNN masks are analysis evidence, not assumed reconstruction masks.

## 9B — Masked versus unmasked geometric benchmark

Twenty frozen representative pairs were evaluated in three modes using the same Step 6 SIFT detections and geometry stack:

| Mode | Total candidates | Total RANSAC inliers | Median inlier ratio | Median Sampson error | Median 4x4 coverage | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `unmasked` | 5,344 | **3,146** | **0.506391** | 0.133932 | **0.625** | baseline |
| `raw_cnn` | 4,602 | 2,841 | 0.501026 | **0.121050** | 0.500 | no |
| `reconstruction_mask` | 4,602 | 2,841 | 0.501026 | **0.121050** | 0.500 | no |

The predeclared qualification rule required a masked mode to retain at least 95% of unmasked total inliers, not reduce the median inlier ratio, and keep median Sampson error within 110% of the unmasked value.

Both masked modes retained only 2,841 / 3,146 = **90.31%** of unmasked inliers and also had a slightly lower median inlier ratio. Their lower Sampson error therefore did not compensate for the lost correspondence coverage.

**Measured decision: use `unmasked` Step 6 SIFT features for reconstruction preparation.**

This is a useful negative result: the CNN segmentation performs well as a segmentation model, but filtering SIFT through its masks removes too much geometrically useful correspondence evidence for this capture sequence.

## 9C — Full-sequence connectivity and subset decision

Using the frozen `unmasked` decision, all 287 adjacent selected-image edges were measured. An edge is strong only when Fundamental-Matrix status is `ok`, RANSAC inliers are at least 15, and inlier ratio is at least 0.15.

Measured result:

- adjacent edges: **287**
- strong adjacent edges: **273**
- weak adjacent edges: **14**
- skip bridges tested around weak transitions: **14**
- strong skip bridges: **0**
- recommended included images: **288 / 288**
- safely excluded images under the conservative rule: **0**

Weak adjacent pairs are:

```text
73-74, 145-146, 203-204, 246-247, 258-259, 266-267, 267-268,
271-272, 272-273, 273-274, 276-277, 279-280, 283-284, 285-286
```

No weak middle image had a strong predecessor-to-successor skip bridge, so the evidence does not justify automatically dropping any frame. The recommended reconstruction subset therefore remains all 288 selected images. This is deliberately conservative because eliminating weak-but-needed frames could reduce coverage further.

`preprocessing/reconstruction_input_v1/manifest.csv` records the include decision for every selected image and references the existing selected JPEGs rather than duplicating them. Because Step 9B chose `unmasked`, no mask path is assigned as a required reconstruction input.

## 9D — Camera and EXIF readiness

Camera metadata was read from the immutable raw JPEG corresponding to each selected filename. All 288 selected frames have one complete, consistent camera signature:

- Make: OPPO
- Model: OPPO Reno12 F
- LensModel: `OPPO Reno12 F back camera 26mm f/1.8`
- image geometry: 3072 x 4080
- orientation: 1
- focal length: 3.98 mm
- 35 mm equivalent focal length: 26 mm
- digital zoom ratio: 1.0
- missing values across the recorded camera-readiness fields: 0

**Measured recommendation: one shared camera/intrinsics group is justified by the available EXIF evidence.**

Step 9 did not calibrate, undistort, resize, crop, rotate, or otherwise modify the source images.

## Verification

Final verification after Step 9 implementation:

- Step 9 focused suite: **26 passed**.
- Complete project suite: **92 passed**.
- `reconstruction_masks.py`, `reconstruction_matching.py`, `camera_readiness.py`, and `run_reconstruction_readiness.py` compiled successfully with `python -B -m py_compile`.
- Raw baseline verification: **297 / 297 unchanged**, 0 missing, 0 unexpected, 0 size mismatches, 0 SHA-256 mismatches.
- Selected-image verification: **288 / 288** matched `preprocessing/reports/selection_manifest.csv`.
- Reconstruction-mask manifest verification re-opened and hash-checked all **288 raw predictions + 288 reconstruction masks**; all were source-size binary PNGs with matching recorded hashes and foreground fractions.
- All four Step 9 presentation figures were visually inspected. A misleading zero-range camera-metadata plot was found during review and fixed; the final figure now explicitly shows zero missing values.
- The most strongly changed cleanup cases were also visually inspected to confirm that detached background components were removed without reshaping the vessel.
- No pyCOLMAP/COLMAP, pose-estimation, triangulation, sparse/dense reconstruction, mesh, texture, or Blender artifact was created by Step 9.

## Machine-readable evidence

- `analysis/reports/reconstruction_mask_manifest.csv`
- `analysis/reports/step9_masks.json`
- `analysis/reports/step9_match_benchmark.csv`
- `analysis/reports/step9_match_benchmark.json`
- `analysis/reports/step9_connectivity.csv`
- `analysis/reports/step9_connectivity.json`
- `analysis/reports/step9_camera_readiness.csv`
- `analysis/reports/step9_camera_readiness.json`
- `analysis/reports/step9_summary.json`
- `analysis/previews/presentation/step9_01_reconstruction_masks.png`
- `analysis/previews/presentation/step9_02_match_benchmark.png`
- `analysis/previews/presentation/step9_03_connectivity.png`
- `analysis/previews/presentation/step9_04_camera_readiness.png`
- `preprocessing/reconstruction_input_v1/manifest.csv`

## Boundary after Step 9

Step 9 establishes readiness evidence; it does **not** prove reconstruction quality. The next phase, if separately authorized, is pyCOLMAP/SfM. That later phase should begin from the unchanged 288 PREPROCESSED images, use the Step 9 recommendation of unmasked Step 6 SIFT evidence as the current matching baseline, and treat the single-camera EXIF grouping as a starting configuration to be validated by actual reconstruction results.
