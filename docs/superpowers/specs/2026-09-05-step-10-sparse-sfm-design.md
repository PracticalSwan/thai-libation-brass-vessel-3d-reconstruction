# Step 10 Sparse SfM Design

Updated: 2026-09-05

## Status

Completed and verified on 2026-09-05. Step 10 used pyCOLMAP 4.2.0 to extract native SIFT, run sequential matching, estimate sparse camera poses/points, and execute the single allowed overlap-40 retry. Both attempts produced seven disconnected sparse components; the largest component registered 73/288 images, so the fixed healthy-single-model acceptance gate was not met. The frozen ranking selected the baseline 73-image / 6,099-point component, and the measured limitation is documented in `docs/geometry-ml/sparse-reconstruction.md`.

Step 10 stopped before image undistortion for dense MVS, patch-match stereo, stereo fusion, dense point clouds, meshing, texturing, or Blender.

## Frozen inputs from Step 9

- Raw source remains immutable: `IMG20260826122949/`, 297 JPEGs.
- Reconstruction image source remains immutable: `preprocessing/pycolmap_input/images/`, 288 PREPROCESSED JPEGs.
- Step 9 recommends retaining all 288 selected images.
- Step 9 recommends unmasked features; no CNN mask is supplied to pyCOLMAP feature extraction.
- Step 9 measured one camera signature across all 288 selected filenames: OPPO Reno12 F, 3072 x 4080, orientation 1, focal length 3.98 mm, 35 mm-equivalent focal length 26 mm, digital zoom 1.0.
- Step 9 weak-transition list remains diagnostic evidence only; frames are not deleted automatically.

## Dependency choice

Use the current stable `pycolmap` wheel supported by the project Python environment. At implementation time, PyPI dry-run evidence shows `pycolmap 4.2.0` is available for Python 3.14 on Windows. Add a bounded requirement `pycolmap>=4.2,<5` to `requirements.txt` after successful installation/import verification.

Do not add Open3D, trimesh, PyMeshLab, learned local features, vocabulary-tree downloads, or unrelated reconstruction dependencies for Step 10.

## Camera initialization

Use one shared camera for all 288 images with `pycolmap.CameraMode.SINGLE`.

Baseline camera model: `SIMPLE_RADIAL`.

Because the PREPROCESSED JPEGs may not retain the original EXIF block, initialize focal length explicitly from the measured 35 mm-equivalent focal length using diagonal field-of-view equivalence:

```text
f_px = (f_35mm / diagonal_35mm) * diagonal_pixels

where:
diagonal_35mm = sqrt(36^2 + 24^2) = 43.2666153 mm
diagonal_pixels = sqrt(3072^2 + 4080^2)
f_35mm = 26 mm
```

Use principal point at image center and zero initial radial distortion:

```text
SIMPLE_RADIAL params = f_px, cx=1536, cy=2040, k=0
```

This is an initialization prior, not a calibration claim. The incremental mapper may refine focal length and radial distortion while keeping one shared camera.

## Baseline feature extraction

Use pyCOLMAP native SIFT on the unchanged PREPROCESSED images.

- no CNN masks;
- no image resizing on disk;
- feature extraction may use COLMAP's internal maximum-image-size limit;
- target `max_image_size=1200` after the real Windows pyCOLMAP 4.2 wheel reported no CUDA support and the initial CPU-only 3200-pixel extraction processed only 55/288 images before the execution window ended; this is an internal sparse-SIFT limit only and does not resize the source JPEGs;
- target maximum features: 8192 per image;
- use CPU-compatible pyCOLMAP defaults available on Windows.

Record feature counts from the COLMAP database before mapping.

## Baseline matching

Use sequential matching because the capture is ordered.

Baseline pairing options:

```text
overlap = 20
quadratic_overlap = true
loop_detection = false
```

Do not use exhaustive matching in the baseline. Do not download a vocabulary tree.

Record matched-pair and geometrically verified-pair counts from the COLMAP database.

## Incremental mapping

Run `pycolmap.incremental_mapping` into a dedicated baseline output directory.

Use standard incremental SfM with:

- minimum 15 inlier matches for usable pairs;
- multiple-model output enabled so disconnected components remain visible instead of being hidden;
- minimum model size 10;
- one shared camera from feature extraction;
- focal length and extra radial parameter refinement enabled;
- principal-point refinement disabled;
- deterministic/random-seed setting when the API exposes it.

Do not force a hand-selected initial image pair unless the automatic mapper fails to initialize.

## Baseline acceptance and controlled retry

Summarize every returned sparse model. Rank models by registered-image count, then sparse-point count.

Baseline is accepted without retry when the largest model:

- registers at least 274 of 288 images (>=95.1%);
- contains at least 1,000 sparse 3D points;
- has finite mean reprojection error;
- uses one shared camera;
- does not show obviously impossible camera-pose geometry in the generated preview.

A controlled retry is allowed only when the baseline fails the registration threshold or fragments into multiple meaningful components.

Retry strategy is intentionally narrow:

1. reuse the same camera model and camera initialization;
2. increase sequential overlap from 20 to 40;
3. keep `quadratic_overlap=true` and `loop_detection=false`;
4. rerun matching/mapping in a fresh retry workspace;
5. select the better attempt by largest-model registered-image count, then sparse-point count, then lower mean reprojection error.

Do not add exhaustive matching, camera-model sweeps, learned features, or repeated parameter search in Step 10 unless the retry cannot initialize at all. If both attempts fail to produce a usable model, stop and document the blocker rather than over-tuning.

## Outputs

Persistent Step 10 outputs:

```text
reconstruction/sparse/baseline/              baseline sparse model(s)
reconstruction/sparse/retry_overlap40/       only if retry is required
reconstruction/sparse/best/                  copy of the selected sparse model
reconstruction/reports/step10_attempts.csv
reconstruction/reports/step10_summary.json
reconstruction/reports/step10_registered_images.csv
reconstruction/previews/step10_01_sparse_model.png
reconstruction/previews/step10_02_registration.png
docs/geometry-ml/sparse-reconstruction.md
```

The transient COLMAP database is an implementation artifact. Keep it only if small enough to be useful and safe for the repository; otherwise remove it after metrics/model export because Step 10 is reproducible from the immutable image set and script.

The selected sparse binary model files (`cameras.bin`, `images.bin`, `points3D.bin`, plus rigs/frames when emitted) are expected to be small enough to commit. Export a PLY point cloud for inspection if pyCOLMAP supports it directly without adding dependencies.

## Diagnostics

For the selected model record:

- attempt name and matching overlap;
- number of input images;
- number and sizes of returned models;
- registered image count and percentage;
- unregistered image filenames/indices;
- sparse point count;
- observation count;
- mean track length;
- mean observations per registered image when available;
- mean reprojection error;
- shared camera model and final parameters;
- focal-length change from initialization;
- camera centers for preview rendering.

The sparse-model preview must plot real 3D points and camera centers from the selected reconstruction. The registration preview must show registered/unregistered sequence positions and highlight the Step 9 weak-transition regions without implying causation.

## Implementation structure

Create a focused `sparse_reconstruction.py` module for pure configuration, camera initialization, pyCOLMAP runtime wrappers, database/model metrics, attempt ranking, and report serialization.

Create `run_sparse_reconstruction.py` as the bounded Step 10 orchestration entry point. It verifies the 288-image input manifest before running, creates clean attempt directories, performs baseline and optional retry, selects the best model, writes reports/figures, and never invokes dense reconstruction APIs.

## Verification

Before completion:

1. focused Step 10 tests pass;
2. complete project tests pass;
3. changed Python compiles;
4. pyCOLMAP import/version is recorded;
5. all 288 selected input images still verify against `selection_manifest.csv`;
6. all 297 raw images still verify against `raw_manifest_before.json`;
7. selected sparse model reopens with pyCOLMAP and metrics match the written summary;
8. sparse and registration figures are visually inspected;
9. no dense/MVS/mesh/texture/Blender artifacts exist from Step 10;
10. task-created caches, failed partial outputs, and transient databases not needed for the final deliverable are removed;
11. DOCX/PDF reports and the local CNN checkpoint remain untouched/unpublished;
12. intended changes are committed to `main`, pushed without force, and local `HEAD`, fetched `origin/main`, and direct remote `main` are verified equal.
