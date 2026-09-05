# Step 11 Sparse Component Bridging Design

Updated: 2026-09-05

## Status

Completed, reviewed, and verified on 2026-09-05. Step 11 executed the bounded recovery design against all 288 verified PREPROCESSED images. The non-local diagnosis found no geometrically verified bridge at boundaries 73-74 or 145-146, so targeted mapping was skipped by the frozen gate. The single exhaustive fallback produced eight disconnected models with 224-image union coverage; its strongest single model still registered only 73/288 images. The fixed global acceptance gate therefore failed and `bridge_success=false` remains the measured outcome.

Post-implementation review also hardened feature-cache/exhaustive-resume identity checks to compare exact image names, camera IDs, and per-image keypoint/descriptor layout, and corrected the zero-inlier candidate-figure scale so impossible negative inlier values are no longer displayed.

Step 11 stops before dense MVS, image undistortion for dense reconstruction, PatchMatch stereo, stereo fusion, meshing, texturing, or Blender.

## Step 10 evidence that controls Step 11

Step 11 must begin from the measured Step 10 state rather than inventing a new reconstruction pipeline.

- Input images: 288 immutable PREPROCESSED JPEGs under `preprocessing/pycolmap_input/images/`.
- Selection manifest SHA-256: `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`.
- pyCOLMAP: 4.2.0.
- Feature type: native unmasked pyCOLMAP SIFT.
- Sparse feature scale: internal `max_image_size=1200`; source JPEGs remain 3072 x 4080.
- Shared camera: one `SIMPLE_RADIAL` camera initialized from the measured 26 mm 35-mm-equivalent EXIF evidence.
- Step 10 baseline overlap 20: 7 sparse components, 216-image union coverage; largest component 73/288 images with 6,099 points and 1.2373 px mean reprojection error.
- Step 10 overlap-40 retry: 7 sparse components, 223-image union coverage; largest component still 73/288 images with 5,769 points.
- The overlap-40 retry did not solve fragmentation, so Step 11 must add genuinely new cross-component pairing evidence rather than merely increasing sequential overlap again.
- Major observed sequence boundaries are consistent with Step 9 weak transitions at 73-74, 145-146, and 203-204.

## Objective

The primary objective is to produce one plausible sparse model registering at least 274 of the 288 selected images while retaining the Step 10 camera and feature assumptions unless direct Step 11 evidence requires a separate future phase.

Step 11 should answer two questions:

1. Can explicitly matched non-local cross-boundary image pairs connect the large local Step 10 components?
2. If targeted bridges are insufficient, can one bounded exhaustive SIFT matching fallback produce a healthy global model without changing camera model, preprocessing, feature type, or image subset?

## Non-goals

Step 11 does not:

- retrain or modify the CNN;
- use CNN masks for SfM features;
- alter, remove, rename, resize, crop, rotate, recompress, or undistort selected images;
- change the Step 10 `SIMPLE_RADIAL` camera model;
- sweep camera models or focal priors;
- introduce learned local features or learned matchers;
- use a vocabulary tree or external retrieval model;
- repeatedly search matcher thresholds;
- delete difficult frames to force a result;
- merge sparse models with guessed transforms;
- start dense reconstruction.

If the targeted attempt and the one exhaustive fallback both fail, Step 11 stops and reports the capture/matching limitation.

## Architecture

Step 11 reuses the Step 10 pyCOLMAP configuration and model-metric code. It adds a focused bridging layer rather than creating a second reconstruction stack.

### Reusable Step 10 seam

`SparseRunConfig` remains the source of truth for:

```text
expected_images = 288
camera_model = SIMPLE_RADIAL
max_image_size = 1200
max_num_features = 8192
minimum_matches = 15
min_model_size = 10
random_seed = 4213
```

A small Step 10 refactor may expose public helpers for:

- image-reader options;
- feature-extraction options;
- incremental-mapping options;
- feature extraction into a specified database;
- sparse mapping from an already matched database.

`run_sparse_attempt` must continue to use those same helpers so Step 10 behavior does not fork.

### Step 11 module boundary

Create `sparse_bridging.py` for:

- fixed boundary-search configuration;
- deterministic non-local candidate-pair generation;
- imported pair-list writing;
- COLMAP database pair-metric extraction;
- candidate qualification and diversity-aware top-pair selection;
- targeted and exhaustive attempt wrappers;
- Step 11 acceptance and ranking.

Create `run_sparse_bridging.py` for bounded stage orchestration, reports, figures, resumability, cleanup decisions, and final selection.

## 11A — Non-local bridge diagnosis

### Critical boundaries

Freeze the three primary evidence-backed boundaries:

```text
73-74
145-146
203-204
```

These are not assumed causes. They are search anchors because they align with the Step 10 component breaks and earlier Step 9 weak transitions.

### Candidate windows

For each boundary `(left, right)`:

- left window: up to 40 images ending at `left`;
- right window: up to 40 images starting at `right`;
- generate the cross-product of those windows;
- exclude any pair whose absolute sequence-index gap is 40 or less.

The gap exclusion is important. Step 10 already executed sequential overlap 40. Step 11 diagnostics must therefore test non-local pairs that Step 10's sequential retry did not intentionally cover as ordinary local neighbors.

For full 40 x 40 windows, this yields 780 non-local candidate pairs per boundary and at most 2,340 candidate pairs before deduplication.

### Feature cache

Extract pyCOLMAP SIFT once into a transient features-only database using the exact Step 10 camera and feature settings.

Expected transient structure:

```text
reconstruction/bridging/work/features.db
reconstruction/bridging/work/features_complete.json
```

The completion marker is written only after:

- the database contains exactly 288 images;
- every image has a keypoint row;
- total feature count is non-zero;
- selected-input manifest verification succeeded.

An interrupted features database without a valid completion marker is partial and may be rebuilt.

### Diagnostic matching

Copy the features-only database to:

```text
reconstruction/bridging/work/diagnostic.db
```

Write all candidate pairs to:

```text
reconstruction/bridging/work/diagnostic_pairs.txt
```

Use `pycolmap.match_image_pairs` with `ImportedPairingOptions(match_list_path=...)` and CPU feature matching.

Do not run incremental mapping in the diagnostic stage.

### Pair metrics

For every candidate pair record:

- boundary id;
- left/right selected index;
- left/right filename;
- sequence gap;
- raw match count from `matches.rows`;
- geometrically verified inlier count from `two_view_geometries.rows`;
- inlier ratio = verified inliers / raw matches when raw matches > 0;
- qualification status.

Use `pycolmap.image_pair_to_pair_id` to address pair rows; do not reimplement COLMAP pair-id arithmetic.

### Bridge qualification

A candidate is a qualified non-local bridge when:

```text
verified_inliers >= 15
inlier_ratio >= 0.15
```

This intentionally reuses the Step 9 strong-edge geometry floor instead of inventing a new post-hoc threshold.

For each boundary, retain at most 8 bridge pairs ranked by:

1. verified inliers descending;
2. inlier ratio descending;
3. sequence gap ascending;
4. filename pair lexicographically for deterministic ties.

To avoid concentrating all bridges through one frame, each endpoint filename may appear in at most two selected bridge pairs per boundary. The greedy selector walks the ranked list and skips candidates that would exceed this endpoint-reuse limit.

A targeted mapping attempt is worthwhile only when every critical boundary has at least one qualified selected bridge pair. If any boundary has zero qualified bridges, skip targeted mapping and proceed directly to the one exhaustive fallback after writing the diagnostic evidence.

## 11B — Targeted bridging reconstruction

Create a fresh copy of the features-only database:

```text
reconstruction/bridging/work/targeted.db
```

Then:

1. run standard sequential matching with overlap 20, quadratic overlap enabled, loop detection disabled;
2. write only the selected qualified bridge pairs to `targeted_bridge_pairs.txt`;
3. run `pycolmap.match_image_pairs` on those selected bridges;
4. run incremental mapping using the same Step 10 `IncrementalPipelineOptions`;
5. preserve all returned sparse components under `reconstruction/bridging/targeted/`;
6. summarize every component with the existing `ModelMetrics` contract.

No matcher threshold, camera model, focal prior, image subset, SIFT scale, or mapper option changes are allowed in this targeted attempt.

## 11C — One exhaustive fallback

Run the exhaustive fallback only when:

- targeted mapping was skipped because at least one boundary had no qualified bridge, or
- targeted mapping completed but its best model did not satisfy the Step 11 acceptance gate.

Create another fresh copy of the features-only database:

```text
reconstruction/bridging/work/exhaustive.db
```

Use `pycolmap.match_exhaustive` with:

```text
ExhaustivePairingOptions.block_size = 50
FeatureMatchingOptions.use_gpu = false
```

Then run incremental mapping with the same camera and mapper options.

This is the only global matching fallback in Step 11. Do not follow it with camera-model sweeps, matcher-ratio sweeps, learned matchers, or another image-selection experiment.

Because exhaustive matching over 288 images is potentially long-running on the CPU-only Windows wheel, execution must distinguish a tool-wrapper timeout from a real process failure. Before restarting an apparently timed-out stage, check whether the process is still active and whether its database/model outputs are progressing.

## Step 11 acceptance gate

A Step 11 model is accepted for progression toward dense reconstruction only when all of the following are true:

```text
registered_images >= 274
sparse_points >= 1000
camera_count == 1
mean_reprojection_error is finite
camera_model == SIMPLE_RADIAL
```

Additionally, visual review must show a coherent camera trajectory and plausible sparse geometry without obvious large pose teleportation or unrelated coordinate clusters.

The 274/288 threshold is the same >=95% registration target used by Step 10.

Record diagnostics but do not automatically reject solely because of:

- a small number of unregistered frames;
- reflective-brass outliers;
- moderate focal refinement;
- background points in the sparse cloud.

## Attempt selection

When both targeted and exhaustive attempts exist:

1. prefer an attempt whose best model passes the acceptance gate;
2. if both pass, rank by registered images, then sparse points, then lower mean reprojection error;
3. if neither passes, use the same ranking only to identify the strongest diagnostic result and record `bridge_success=false`.

Do not overwrite Step 10 outputs. Step 11 selected output belongs under:

```text
reconstruction/bridging/best/
```

The Step 10 `reconstruction/sparse/best/` directory remains preserved as baseline evidence.

## Persistent outputs

Expected Step 11 deliverables:

```text
reconstruction/bridging/targeted/                  only when targeted mapping runs
reconstruction/bridging/exhaustive/                only when fallback runs
reconstruction/bridging/best/                      selected Step 11 sparse model
reconstruction/bridging/reports/step11_candidates.csv
reconstruction/bridging/reports/step11_boundary_summary.json
reconstruction/bridging/reports/step11_attempts.csv
reconstruction/bridging/reports/step11_registered_images.csv
reconstruction/bridging/reports/step11_summary.json
reconstruction/bridging/previews/step11_01_bridge_candidates.png
reconstruction/bridging/previews/step11_02_sparse_model.png
reconstruction/bridging/previews/step11_03_registration.png
docs/geometry-ml/sparse-component-bridging.md
```

Export `points3D.ply` for the selected Step 11 model when one exists.

Transient databases and pair-list work files under `reconstruction/bridging/work/` are not publication artifacts. Remove them after final metrics/model export and verification unless execution is interrupted and they are required to resume safely.

## Figures

### Bridge-candidate figure

Show the three critical boundaries separately. Plot candidate sequence-index pairs with marker size or vertical value representing verified inliers. Clearly distinguish qualified selected bridges from rejected candidates. The figure must be generated from the real diagnostic CSV.

### Sparse-model figure

Reuse the Step 10 visual convention: real sparse points plus registered camera centers from the selected Step 11 model. Title must state the attempt and registered-image count.

### Registration figure

Show all 288 selected indices as registered/unregistered for the selected Step 11 model. Mark the three critical boundaries. Do not confuse union coverage across disconnected components with registration in the selected coordinate frame.

## Reports

`step11_summary.json` must include:

- pyCOLMAP version;
- selection-manifest SHA;
- Step 10 baseline metrics used for comparison;
- critical boundaries;
- candidate-pair count per boundary;
- qualified/selected bridge count per boundary;
- targeted-attempt metrics or explicit skipped reason;
- exhaustive-attempt metrics or explicit skipped reason;
- selected Step 11 attempt;
- registered/unregistered images and indices;
- selected model camera parameters;
- `bridge_success` boolean;
- `dense_reconstruction_started: false`;
- next boundary statement.

## Verification

Before Step 11 can be declared complete:

1. focused Step 11 tests pass;
2. full project test suite passes;
3. changed Python compiles;
4. all 297 raw images still match `raw_manifest_before.json`;
5. all 288 selected images still match `selection_manifest.csv`;
6. Step 10 sparse evidence still exists and reopens; Step 11 must not replace it;
7. selected Step 11 model, if any, reopens through pyCOLMAP and matches the JSON metrics;
8. all final Step 11 figures are visually inspected;
9. no dense/MVS/mesh/texture/Blender artifacts were created;
10. transient databases/caches are cleaned after evidence is safely exported;
11. DOCX/PDF reports and the local CNN checkpoint remain untouched/unpublished;
12. Git staging excludes the local checkpoint and unrelated work;
13. commit/push happens only when explicitly authorized in the implementation task.

## Failure outcome

Step 11 may legitimately complete with `bridge_success=false`.

If both the targeted and exhaustive attempts fail the acceptance gate, preserve their measured sparse models and evidence, document the exact remaining component structure, and stop. The next decision would then be whether the capture sequence itself needs additional photographs or whether a separately approved advanced matcher/component-registration strategy is justified. Do not silently turn Step 11 into an unbounded reconstruction search.
