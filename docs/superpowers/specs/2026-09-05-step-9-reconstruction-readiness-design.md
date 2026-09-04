# Step 9 Reconstruction Readiness Design

Updated: 2026-09-05

## Status

Approved for implementation by the user. Step 9 is a pre-reconstruction phase only. It must stop before pyCOLMAP, camera-pose estimation, triangulation, sparse/dense reconstruction, meshing, texturing, or Blender.

## Goal

Prepare the verified 288-image PREPROCESSED sequence for later reconstruction by producing full-sequence CNN masks, measuring whether masking helps or harms geometric matching, auditing sequence connectivity and a conservative reconstruction subset, and determining the camera/intrinsics grouping from raw EXIF evidence.

## Immutable inputs and provenance

- Raw photographs: `IMG20260826122949/`, 297 immutable JPEGs.
- Verified selected images: `preprocessing/pycolmap_input/images/`, 288 PREPROCESSED JPEGs.
- Selection manifest: `preprocessing/reports/selection_manifest.csv`.
- Frozen segmentation manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`.
- Frozen local checkpoint: `analysis/ml/checkpoints/best_small_seg_cnn.pt`.
- `SmallSegCNN`: 487,297 trainable parameters, random initialization, no pretrained weights, threshold 0.5.
- Existing Step 6 SIFT contract: `geometry_detection.extract_sift`, `match_sift`, `estimate_fundamental_geometry`, and explicit analysis/original scale metadata.

The raw and selected JPEG bytes must not be modified. All Step 9 outputs are derived artifacts outside those directories.

## Step 9A — Full-sequence CNN inference and reconstruction masks

Run the frozen checkpoint on all 288 verified selected images. Save the unedited source-size binary CNN predictions under:

```text
analysis/ml/full_predictions/
```

Derive a separate reconstruction-mask set under:

```text
analysis/ml/reconstruction_masks/
```

The reconstruction-mask cleanup is deterministic and fixed before geometric benchmarking:

1. operate on the 384 x 288 model-space binary prediction;
2. find connected foreground components;
3. choose the anchor component from components intersecting the central ROI (`x=25%-75%`, `y=20%-85%`) using largest area; if none intersects, use the largest component;
4. retain the anchor plus nearby secondary components whose area is at least 2% of the anchor and whose bounding-box gap is no more than 3% of the model-space image diagonal;
5. remove other detached components;
6. do not fill holes/openings and do not erode the vessel silhouette;
7. resize the cleaned binary mask to source geometry with nearest-neighbor interpolation only.

This is a downstream reconstruction-mask transform, not a repaired CNN test prediction. Existing held-out predictions and Step 7/8 metrics remain untouched.

Outputs:

```text
analysis/reports/reconstruction_mask_manifest.csv
analysis/previews/presentation/step9_01_reconstruction_masks.png
```

The contact sheet samples all capture phases and must visibly include held-out index 72 so the known yellow-wall failure remains auditable.

## Step 9B — Masked versus unmasked geometric benchmark

Use exactly the existing Step 6 SIFT settings (`SiftConfig()` defaults) and the same selected PREPROCESSED JPEGs. Do not redetect on edited photographs.

Benchmark three feature modes:

- `unmasked`: all Step 6 SIFT features;
- `raw_cnn`: Step 6 SIFT features filtered by the unedited full-sequence CNN prediction;
- `reconstruction_mask`: Step 6 SIFT features filtered by the deterministic reconstruction mask.

Representative pairs are frozen as the ten established neighboring pairs plus ten skip-one counterparts:

```text
15-16, 45-46, 75-76, 105-106, 135-136,
165-166, 195-196, 225-226, 255-256, 280-281,
15-17, 45-47, 75-77, 105-107, 135-137,
165-167, 195-197, 225-227, 255-257, 280-282
```

For each pair/mode report:

- feature counts in both images;
- Lowe-ratio candidate matches;
- Fundamental-Matrix RANSAC inliers;
- inlier ratio;
- median and p90 Sampson error for inliers;
- 4x4 grid inlier coverage averaged over both images.

A masked mode qualifies for the later connectivity audit only if, versus `unmasked` across the 20-pair benchmark:

- total RANSAC inliers are at least 95% of unmasked total;
- median inlier ratio is not lower than unmasked median;
- median inlier Sampson error is no more than 110% of unmasked median.

Among qualified masked modes choose the one with highest median inlier ratio, breaking ties by total inliers. If no masked mode qualifies, choose `unmasked`.

Outputs:

```text
analysis/reports/step9_match_benchmark.csv
analysis/reports/step9_match_benchmark.json
analysis/previews/presentation/step9_02_match_benchmark.png
```

## Step 9C — Full-sequence connectivity and reconstruction subset

Use the mode chosen by Step 9B. Extract Step 6 SIFT features once per selected image and evaluate all 287 adjacent sequence edges.

An adjacent edge is `strong` when:

- Fundamental Matrix status is `ok`;
- RANSAC inliers >= 15;
- inlier ratio >= 0.15.

For each weak adjacent edge `(i, i+1)`, evaluate one skip bridge `(i-1, i+1)` when both endpoints exist. A weak image may be excluded from the recommended reconstruction subset only when:

- neither incident adjacent edge is strong; and
- its immediate predecessor and successor have a strong skip bridge.

Otherwise retain the image as `keep_weak_bridge_needed` to preserve sequence coverage. This rule is intentionally conservative: redundancy alone is not a reason to remove a frame.

Outputs:

```text
analysis/reports/step9_connectivity.csv
analysis/reports/step9_connectivity.json
preprocessing/reconstruction_input_v1/manifest.csv
preprocessing/reconstruction_input_v1/README.md
analysis/previews/presentation/step9_03_connectivity.png
```

The reconstruction-input manifest references the existing selected JPEGs; do not duplicate hundreds of source images into another tracked directory.

## Step 9D — Camera and EXIF readiness

Read EXIF from the immutable raw JPEG corresponding to each of the 288 selected filenames. Read both top-level TIFF tags and the nested EXIF IFD.

Record:

- image index and filename;
- width/height and orientation;
- Make, Model, LensModel;
- FocalLength;
- FocalLengthIn35mmFilm;
- DigitalZoomRatio;
- DateTimeOriginal.

Build a camera signature from dimensions, orientation, make/model/lens, focal length, 35 mm equivalent focal length, and digital zoom.

Recommend one shared camera/intrinsics group only if all 288 selected frames have one consistent camera signature, orientation, and geometry. If signatures differ, recommend separate groups by signature. Missing metadata must be reported explicitly rather than guessed.

Outputs:

```text
analysis/reports/step9_camera_readiness.csv
analysis/reports/step9_camera_readiness.json
analysis/previews/presentation/step9_04_camera_readiness.png
```

No calibration, undistortion, or image resampling is performed in Step 9.

## Final Step 9 summary

`run_reconstruction_readiness.py` provides bounded stages `masks`, `benchmark`, `connectivity`, `camera`, and `summary`; it never invokes pyCOLMAP.

Final report:

```text
analysis/reports/step9_summary.json
docs/geometry-ml/reconstruction-readiness.md
```

The summary records:

- exact source/checkpoint provenance;
- full-mask counts and cleanup behavior;
- benchmark winner and why;
- full-sequence strong/weak edge counts;
- included/excluded reconstruction-subset counts and reasons;
- camera-signature grouping recommendation;
- explicit stop boundary before reconstruction.

## Verification

Before completion:

1. focused Step 9 tests pass;
2. full project test suite passes;
3. changed Python compiles;
4. all 288 raw predictions and 288 reconstruction masks are readable source-size binary PNGs;
5. all 297 raw JPEGs remain hash-identical to the baseline manifest;
6. all 288 selected PREPROCESSED JPEGs remain verified against `selection_manifest.csv`;
7. all four Step 9 presentation figures are visually inspected;
8. temporary/cache/failed outputs are removed;
9. no pyCOLMAP/reconstruction artifacts exist from Step 9;
10. intended Git diff excludes the local checkpoint, DOCX/PDF report files, secrets, and unrelated work.
