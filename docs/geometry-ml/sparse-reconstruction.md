# Step 10 Sparse Reconstruction

Updated: 2026-09-05

## Status

Step 10 sparse Structure-from-Motion (SfM) is implemented, executed, reviewed, and verified. The real pyCOLMAP run produced valid local sparse reconstructions and camera trajectories, but it did **not** meet the predefined healthy-single-model acceptance target because the 288-image sequence fragmented into multiple sparse components.

This is a measured reconstruction limitation, not a hidden failure. The approved Step 10 design allowed one controlled retry only; that retry did not merge the sequence into a dominant model, so Step 10 stops here rather than over-tuning.

No dense reconstruction, image undistortion for MVS, patch-match stereo, stereo fusion, meshing, texturing, or Blender work was performed.

## Inputs

- Immutable raw source: `IMG20260826122949/`, 297 JPEGs.
- Verified reconstruction source: `preprocessing/pycolmap_input/images/`, 288 PREPROCESSED JPEGs.
- Selection manifest SHA-256: `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`.
- Step 9 decision: use all 288 selected images, do not apply CNN masks to reconstruction features, and begin from one shared camera/intrinsics group.

## Runtime environment

- Python: 3.14.x project runtime.
- pyCOLMAP: **4.2.0**.
- Windows pyCOLMAP wheel: CPU SIFT path; CUDA support was not available in the installed wheel.
- Added dependency: `pycolmap>=4.2,<5`.

An initial CPU-only SIFT attempt at internal `max_image_size=3200` processed only 55 of 288 images before the execution window ended. Because no mapping result existed at that point, this was treated as a runtime feasibility correction rather than reconstruction-result tuning. The final sparse-SIFT limit was reduced to **1200 pixels internally**. The 3072 x 4080 source JPEGs were never resized or modified on disk.

## Camera initialization

All 288 selected filenames share the Step 9 OPPO Reno12 F EXIF signature. Step 10 therefore used `pycolmap.CameraMode.SINGLE` with one shared `SIMPLE_RADIAL` camera.

Initial focal length used the measured 26 mm 35-mm-equivalent focal length:

```text
f_px = (26 / sqrt(36^2 + 24^2)) * sqrt(3072^2 + 4080^2)
     = 3069.0506675517754 px
```

Initial camera parameters:

```text
[f, cx, cy, k] = [3069.0506675517754, 1536.0, 2040.0, 0.0]
```

The incremental mapper was allowed to refine focal length and radial distortion while principal point refinement remained disabled.

## Feature extraction and matching

Final sparse-feature configuration:

- native pyCOLMAP SIFT;
- no CNN mask;
- CPU feature extraction;
- internal `max_image_size=1200`;
- maximum 8,192 SIFT features per image;
- one shared `SIMPLE_RADIAL` camera;
- sequential pairing;
- quadratic overlap enabled;
- loop detection disabled;
- minimum 15 matches for the incremental pipeline.

### Baseline — sequential overlap 20

Database evidence:

- images: **288**;
- SIFT features: **1,255,153**;
- non-empty matched pairs: **1,500**;
- non-empty geometrically verified pairs: **902**;
- runtime: **346.224 s**.

pyCOLMAP returned **7 sparse models**. Their registered-image counts were:

```text
2, 14, 72, 47, 73, 31, 13
```

Across all baseline models, **216 distinct input images** appeared in at least one sparse component. The largest individual model registered **73/288 images (25.35%)**.

Baseline largest model (`reconstruction/sparse/baseline/4`):

- registered images: **73**;
- sparse points: **6,099**;
- observations: **21,351**;
- mean track length: **3.5007**;
- mean observations per registered image: **292.4795**;
- mean reprojection error: **1.2373 px**;
- camera count: **1**;
- final camera model: `SIMPLE_RADIAL`;
- final camera parameters: `[3542.7206959261907, 1536.0, 2040.0, -0.01641661056124677]`.

The refined focal length is about 15.4% above the EXIF-derived initialization. This value is an SfM optimization result for this component, not a physical calibration measurement.

## Controlled retry — sequential overlap 40

The baseline failed the fixed registration target and was strongly fragmented, so the one approved retry increased only sequential overlap from 20 to 40. Camera model, image set, SIFT limit, masks, and mapper policy stayed unchanged.

Retry database evidence:

- images: **288**;
- SIFT features: **1,255,153**;
- non-empty matched pairs: **1,501**;
- non-empty geometrically verified pairs: **898**;
- runtime: **337.119 s**.

The retry again returned **7 sparse models** with registered-image counts:

```text
7, 16, 72, 73, 31, 47, 13
```

Across all retry models, **223 distinct images** appeared in at least one sparse component. The largest individual model again registered **73/288 images**.

Retry largest model (`reconstruction/sparse/retry_overlap40/3`):

- registered images: **73**;
- sparse points: **5,769**;
- observations: **20,372**;
- mean track length: **3.5313**;
- mean reprojection error: **1.2163 px**;
- final shared camera parameters: `[3504.289718184827, 1536.0, 2040.0, -0.006269725216613968]`.

The retry did not improve the dominant-component registration count. Under the frozen ranking rule—registered images first, then sparse points, then lower reprojection error—the **baseline** is selected because both dominant models register 73 images but the baseline contains more sparse points: 6,099 versus 5,769.

## Fragmentation diagnosis

The baseline model membership is strongly sequence-local rather than random:

- selected component: indices **1-73**;
- another large component: indices **74-145**;
- another component: indices **146-176**;
- an overlapping later component: indices **157-203**;
- smaller later components occur beyond index 203.

The retry similarly reconstructs large local ranges but does not merge them into one coordinate frame.

This matters because Step 9 had already flagged weak adjacent transitions at **73-74**, **145-146**, and **203-204**. The sparse-model boundaries are therefore consistent with the earlier connectivity warnings. This is an observation, not proof that those individual edges alone caused the fragmentation.

The retry's seven components cover 223 distinct images in total, leaving 65 images outside every returned retry model. Missing retry-union ranges are:

```text
204-217, 231-239, 247-288
```

Increasing sequential overlap alone is therefore insufficient to create a single global reconstruction.

## Selected sparse output

The frozen ranking rule selected the baseline 73-image component. It is copied to:

```text
reconstruction/sparse/best/
```

The selected model includes COLMAP binary sparse-model files plus:

```text
reconstruction/sparse/best/points3D.ply
```

Machine-readable evidence:

- `reconstruction/reports/step10_baseline.json`
- `reconstruction/reports/step10_retry_overlap40.json`
- `reconstruction/reports/step10_attempts.csv`
- `reconstruction/reports/step10_registered_images.csv`
- `reconstruction/reports/step10_summary.json`

Visual evidence:

- `reconstruction/previews/step10_01_sparse_model.png`
- `reconstruction/previews/step10_02_registration.png`

## Visual review

The selected-component sparse preview was visually inspected. Its registered camera centers form a coherent curved trajectory around the reconstructed point cloud rather than obviously teleporting or scattering randomly. The point cloud contains structured brass/object-colored geometry and background structure, so it is a plausible local SfM reconstruction.

The registration figure was also inspected. It correctly shows that the selected component covers sequence indices 1-73 and that later images are outside this selected coordinate frame. The figure labels this explicitly as **selected-component registration** so it is not confused with union coverage across all disconnected sparse models.

## Acceptance decision

The predefined healthy-single-model acceptance criteria required at least 274/288 registered images, at least 1,000 sparse points, finite reprojection error, one shared camera, and plausible camera geometry.

The selected model satisfies the point, reprojection, camera-count, and visual-plausibility criteria, but registers only **73/288 images**. `step10_summary.json` therefore records:

```text
acceptance_met = false
```

Step 10 is considered **execution-complete but globally fragmented**. The correct outcome is to preserve the measured sparse components and stop, not to claim full-sequence reconstruction success.

## Final verification

- Step 10-focused tests: **11 passed**.
- Fresh complete project suite: **103 passed**.
- `sparse_reconstruction.py` and `run_sparse_reconstruction.py` completed `python -B -m py_compile` successfully.
- Raw baseline re-verification: **297/297 unchanged**, zero missing/unexpected/size/hash mismatches.
- Selected-input re-verification: **288/288** matched `preprocessing/reports/selection_manifest.csv` with manifest SHA-256 `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`.
- `reconstruction/sparse/best/` re-opened with pyCOLMAP 4.2.0 and matched `step10_summary.json`: 73 registered images, 6,099 sparse points, one camera, and 1.2373052447638215 px mean reprojection error.
- Both final Step 10 figures were visually inspected.
- No dense-like reconstruction artifacts were present under `reconstruction/`.
- The transient baseline/retry COLMAP databases (about 195 MB each) and task-created caches were removed after their metrics and sparse models were safely exported.
- The local CNN checkpoint remained untouched and outside the publication set.

## What Step 10 did not do

Step 10 deliberately did not:

- switch to exhaustive matching after seeing the result;
- sweep camera models;
- add learned feature matching;
- hand-repair or delete difficult frames;
- use CNN reconstruction masks;
- start dense MVS from a fragmented sparse model;
- merge components with guessed transforms;
- create a mesh or texture;
- invoke Blender.

Those would be new reconstruction decisions and require a separate evidence-backed plan.

## Next technical decision

Dense reconstruction should **not** start from the current 73-image selected component if the project objective remains a full-vessel model from the complete capture sequence.

The next recommended phase is a narrowly scoped sparse-component bridging investigation: identify cross-component image pairs around the measured boundaries, test targeted cross-sequence matching or a different global matching strategy, and verify whether the large local models can be merged into one consistent SfM reconstruction. That is outside Step 10 and should be planned separately rather than silently added as another retry.
