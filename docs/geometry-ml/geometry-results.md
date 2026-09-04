# Step 6 Geometry Detection and Analysis Results

Updated: 2026-08-28

## Outcome

Step 6 is implemented and verified on the real selected-image set. The workflow
exposes three classical computer-vision techniques for coursework explanation:

1. SIFT keypoint detection and neighboring-frame descriptor matching;
2. Fundamental Matrix estimation with RANSAC inlier filtering and epipolar-line
   visualization;
3. grayscale/Canny edge evidence plus classical vessel contour, centroid,
   bounding-box, PCA-axis, and optional ellipse measurements.

These are 2D and two-view projective measurements. No camera pose,
triangulation, sparse or dense reconstruction, CNN segmentation/training,
meshing, texturing, or Blender work was performed.

## Verified inputs

The orchestrator reads the deterministic row order in
`preprocessing/reports/selection_manifest.csv`. Before creating or replacing
Step 6 outputs it verifies every selected file's presence, readability,
dimensions, byte size, and SHA-256 digest.

- selected records verified: **288 / 288**;
- selection-manifest SHA-256:
  `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`;
- selected-image dimensions: **3072 x 4080**;
- raw photographs rechecked after Step 6: **297 / 297 unchanged**, with no
  missing, unexpected, size-mismatched, or hash-mismatched files.

The analysis copy is scaled without cropping or warping to a maximum width of
1200 pixels, producing 1200 x 1594 analysis images. Reports preserve the
separate x/y scale factors back to the 3072 x 4080 originals.

## Technique 1: SIFT matching and RANSAC

The fixed configuration is SIFT `nfeatures=8000`, BF-L2 two-nearest-neighbor
matching, Lowe ratio `0.75`, Fundamental Matrix RANSAC threshold `1.5` analysis
pixels, confidence `0.99`, and RNG seed `4213`.

| Pair | Role | Keypoints | Ratio-test candidates | RANSAC inliers | Inlier ratio |
|---|---|---:|---:|---:|---:|
| 165-166 | primary neighboring view | 4,653 / 4,643 | 478 | 300 | 0.628 |
| 255-256 | supporting lower-feature view | 1,233 / 2,289 | 57 | 18 | 0.316 |

The supporting pair is intentionally retained because it shows that feature and
inlier counts depend on view content. It is not substituted with a stronger
pair to make the result look better.

## Technique 2: Fundamental Matrix and epipolar geometry

The primary pair uses the exact RANSAC inlier mask from Fundamental Matrix
estimation. Ten spatially distributed inliers are drawn with corresponding
points and same-color epipolar lines. Sampson error is reported in squared
analysis-pixel units:

- RANSAC inliers: **300**;
- median Sampson error: **0.1431 analysis pixels squared**;
- 90th-percentile Sampson error: **0.7594 analysis pixels squared**;
- maximum Sampson error: **1.0906 analysis pixels squared**.

The lines demonstrate the two-view relation `x2^T F x1 = 0`. They do not recover
camera pose or prove that a 3D reconstruction has succeeded.

## Technique 3: classical 2D vessel geometry

Every shape panel includes the original analysis image, grayscale image, Canny
edges, the selected classical contour, its bounding box and centroid, and the
PCA principal axis. Because broad background edges can join the vessel in these
brass photographs, contour selection uses a deterministic HSV gold-saturation
mask when it contains enough pixels, with Canny morphology as the neutral-image
fallback. This remains classical thresholding; no ML segmentation is used.

| Index | View | Contour area fraction | PCA axis | Ellipse result |
|---:|---|---:|---:|---|
| 165 | primary side view | 0.2035 | 88.99 degrees | omitted: normalized median residual 0.1284 and p90 residual 0.4381 exceed the fit limits |
| 255 | supporting top-down/detail view | 0.1436 | 82.50 degrees | retained: axes 558.7 x 639.8 analysis pixels; median residual 0.0258 and p90 residual 0.1070 |

The side-view contour is useful for centroid, box, and principal-axis evidence,
but a single ellipse does not describe its long, non-elliptical silhouette.
The workflow therefore reports `ellipse_unavailable` instead of drawing a
misleading fit. The top-down/detail contour supports an ellipse and is reported
as `ok`.

## Presentation figures

All six figures below were generated from the real analysis and visually
inspected for legibility, point/line correspondence, honest labels, and the
absence of reconstruction claims:

- `analysis/previews/presentation/geometry_01_matches_165_166.png`;
- `analysis/previews/presentation/geometry_01_matches_255_256.png`;
- `analysis/previews/presentation/geometry_02_epipolar_165_166.png`;
- `analysis/previews/presentation/geometry_03_shape_165.png`;
- `analysis/previews/presentation/geometry_03_shape_255.png`;
- `analysis/previews/presentation/geometry_04_summary.png`.

The matching figures show up to 60 spatially distributed matches so the
presentation remains readable. Machine-readable reports retain all candidate
and inlier counts.

## Reproduce the analysis

From the repository root in the verified preprocessing environment:

```powershell
python -B run_geometry_analysis.py
```

Open all live OpenCV popup views and close them with `q` or `Esc`:

```powershell
python -B show_geometry_visuals.py --mode all
```

Individual popup modes are `matches`, `epipolar`, and `shape`. A bounded
non-GUI verification path is also available:

```powershell
python -B show_geometry_visuals.py --mode all --no-display
```

## Evidence files

- `analysis/reports/input_verification.json` — complete selected-set verification;
- `analysis/reports/geometry_summary.json` — source provenance, fixed
  configuration, runtime, measurements, scope exclusions, and artifact list;
- `analysis/geometry/pair_metrics.csv` — pair-level keypoint/match/inlier metrics;
- `analysis/geometry/epipolar_metrics.json` — Fundamental Matrix and Sampson
  residual evidence;
- `analysis/geometry/shape_metrics.csv` — contour, box, centroid, PCA, and
  ellipse-fit measurements.

## Limits and next boundary

- The classical color threshold is dataset-specific evidence, not a general
  brass-object segmentation model.
- The primary side-view silhouette is not globally elliptical, so its ellipse
  is deliberately omitted.
- Fundamental Matrix inliers and low residuals support two-view consistency;
  they do not guarantee successful multi-view reconstruction.
- Steps 7+8 are now implemented and verified using a small from-scratch CNN segmentation workflow that reuses this finished Step 6 SIFT/scale interface. Their measured results are documented in `cnn-dataset.md` and `ml-results.md`.
- pyCOLMAP and all reconstruction stages remain separately authorized future work. No reconstruction-improvement claim is inferred from the CNN/SIFT feature-mask analysis.
