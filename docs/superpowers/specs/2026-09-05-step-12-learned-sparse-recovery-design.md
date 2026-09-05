# Step 12 Learned Sparse Recovery Design

**Status:** Implemented; execution complete at approved capability blocker
**Written:** 2026-09-06
**Canonical project:** `C:\Assumption University\CSX4213\Project`
**Baseline commit:** `ff57a67f6989432ca88d6559ad40507037f7dede`
**Implementation status:** COMPLETE AT FAILURE BOUNDARY — the real ALIKED smoke stopped with `RuntimeError: ALIKED feature extraction requires ONNX support.`; no full learned extraction/matching/mapping or LoMa inference ran

## 1. Purpose

Step 12 is a bounded sparse Structure-from-Motion recovery phase that changes the local feature/matcher family after the classical SIFT path was exhausted in Steps 10 and 11.

The phase tests whether native learned frontends in the already-installed pyCOLMAP 4.2 runtime can recover the three known sequence breaks and produce one globally useful sparse reconstruction from the existing immutable 288-image input.

Step 12 does **not** start dense reconstruction and does **not** introduce recapture, external learned-matching frameworks, a benchmark zoo, or an open-ended parameter search.

## 2. Why Step 12 exists

Step 10 showed that SIFT can reconstruct strong local components but not the full capture:

- 288 selected images
- best single sparse component: 73 registered images
- 6,099 points
- one `SIMPLE_RADIAL` camera
- 1.2373 px mean reprojection error
- 216-image disconnected union coverage

Step 11 then directly attacked the three dominant sequence boundaries:

- `73-74`
- `145-146`
- `203-204`

The frozen 40-by-40 cross-boundary search produced 2,340 candidate pairs. SIFT found zero geometrically verified pairs across the first two boundaries, while `203-204` was recoverable. One bounded exhaustive SIFT fallback still produced disconnected models, with only 73 images in the strongest single model.

This establishes a clean failure boundary for classical matching. The next controlled variable is the feature/matcher family.

## 3. Runtime evidence and architectural choice

The live project runtime was inspected before this design was written:

```text
Python      3.14.2
pyCOLMAP    4.2.0
has_cuda    False
```

The installed pyCOLMAP exposes the following relevant native enum members:

```text
FeatureExtractorType.ALIKED_N16ROT
FeatureMatcherType.ALIKED_LIGHTGLUE
FeatureExtractorType.LOMA_B
FeatureMatcherType.LOMA_L
```

The local option objects validate successfully for all four types. Both learned extractors report:

```text
requires_rgb()      == True
requires_opengl()   == False
eff_max_image_size()== 1600 when max_image_size <= 0
```

The Python bindings expose a generic `FeatureExtractionOptions` object rather than separate ALIKED/LoMa option blocks. The implementation will therefore freeze the learned image-size limit explicitly at `1600`, use the native model defaults for learned-specific internals, and avoid inventing unsupported tuning controls.

The native path is preferred because it preserves the existing COLMAP database, camera setup, geometric verification, and mapper architecture.

## 4. Frontend budget

Exactly two learned frontend families are allowed in Step 12.

### Primary

```text
ALIKED_N16ROT + ALIKED_LIGHTGLUE
```

### Single fallback

```text
LOMA_B + LOMA_L
```

LoMa runs only when ALIKED either:

1. fails the three-boundary diagnostic gate, or
2. maps successfully but the strongest single model fails the global Step 12 acceptance gate.

If ALIKED passes global acceptance, Step 12 stops immediately and LoMa is recorded as not run.

Step 12 must not add or try:

- ALIKED-N32
- other LoMa matcher variants
- LoFTR
- hloc
- external LightGlue
- SuperPoint/SuperGlue
- exhaustive learned matching
- broad parameter sweeps

A future phase may consider those only after a separate architectural decision.

## 5. Immutable inputs and protected evidence

The reconstruction input remains:

```text
preprocessing/pycolmap_input/images/
288 PREPROCESSED JPEGs
```

The selection manifest remains authoritative:

```text
preprocessing/reports/selection_manifest.csv
SHA-256: 79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91
```

Protected and read-only during Step 12:

```text
IMG20260826122949/
preprocessing/pycolmap_input/images/
reconstruction/sparse/
reconstruction/bridging/
```

The local CNN checkpoint must remain unpublished:

```text
analysis/ml/checkpoints/best_small_seg_cnn.pt
```

Step 12 must not modify DOCX/PDF report artifacts.

## 6. Frozen camera and mapper configuration

Learned recovery changes only the feature/matcher frontend.

The camera and mapper path remains the Step 10/11 path:

```text
CameraMode.SINGLE
Camera model: SIMPLE_RADIAL
Initial focal estimate: 3069.0506675517754 px
Principal point: image center
Initial radial distortion: 0.0
```

Reuse `SparseRunConfig`, `build_image_reader_options`, `build_incremental_pipeline_options`, `map_sparse_database`, and existing model-summary helpers from `sparse_reconstruction.py`.

The targeted sequence overlap remains:

```text
20
quadratic_overlap = True
loop_detection = False
```

The implementation must pass the selected learned `FeatureMatchingOptions` explicitly into **both**:

```python
pycolmap.match_sequential(...)
pycolmap.match_image_pairs(...)
```

Omitting `matching_options` would fall back to the default SIFT matcher and invalidate the experiment.

All learned extraction/matching is CPU-bound in the current wheel:

```text
pycolmap.Device.cpu
use_gpu = False
```

No CUDA rebuild is part of Step 12.

## 7. Capability gate

Capability verification is mandatory before full learned extraction.

### 7.1 Up-front runtime snapshot

Record in `step12_capability.json`:

- Python version
- pyCOLMAP version
- `pycolmap.has_cuda`
- complete `FeatureExtractorType` member names
- complete `FeatureMatcherType` member names
- ALIKED extraction option snapshot
- ALIKED-LightGlue matching option snapshot
- LoMa-B extraction option snapshot
- LoMa-L matching option snapshot
- effective learned maximum image size
- RGB/OpenGL requirements
- option `check()` results
- selected CPU device policy

### 7.2 Primary inference smoke test

Before extracting all 288 images, run an ALIKED/LightGlue in-memory smoke test on one known local-overlap pair from the verified selected set.

Use `FeatureExtractor.create(..., Device.cpu)`, `extract_from_uint8_array`, `FeatureMatcher.create(..., Device.cpu)`, and `match(...)` so that model creation and inference are tested without creating a persistent reconstruction database.

The smoke passes only if:

- both images load as RGB arrays,
- both extractions return non-empty keypoints/descriptors,
- descriptor/keypoint counts agree per image,
- the matcher returns a valid `N x 2` match array,
- `N > 0`,
- no native model/runtime exception occurs.

The smoke is a capability test only. Its match count is not part of the Step 12 scientific result.

### 7.3 LoMa smoke timing

Record LoMa enum/options availability up front, but do not initialize/download/run LoMa model inference unless the ALIKED fallback gate is reached.

Immediately before LoMa diagnostics, run the same one-pair in-memory LoMa-B/LoMa-L smoke test.

This avoids unnecessary model/cache work when ALIKED succeeds while still preventing a full 288-image LoMa extraction from starting on an unusable runtime.

### 7.4 Capability blocker

If a required native learned frontend cannot be created or run because ONNX/model/runtime support is unavailable:

```text
learned_recovery_success = false
dense_reconstruction_started = false
```

Write the capability failure with the exact exception class/message and stop that path.

Do not automatically:

- rebuild COLMAP/pyCOLMAP,
- replace the current pyCOLMAP wheel,
- install hloc,
- clone external LightGlue,
- add LoFTR,
- install a different ONNX stack.

Those are separate architectural decisions.

Native model caches that pyCOLMAP creates outside the repository may be used, but they must never be copied into or committed from the project.

## 8. Learned feature database contract

Each frontend receives an independent feature database and completion marker under its Step 12 work area.

Cache reuse must be at least as strict as the hardened Step 11 cache validation. A learned cache is reusable only when all of the following agree:

- selection manifest SHA-256,
- frontend identity,
- extractor enum name,
- matcher enum name used by the path,
- pyCOLMAP version,
- CPU device policy,
- explicit `max_image_size=1600`,
- exact image count,
- exact image-name set,
- camera IDs,
- per-image keypoint row presence,
- per-image descriptor row presence,
- per-image keypoint/descriptor count agreement,
- total feature count greater than zero.

The marker is written atomically only after successful extraction and validation.

Do not reuse an ALIKED database for LoMa or vice versa.

## 9. Frozen boundary experiment

Step 12 reuses the Step 11 candidate generator and configuration exactly:

```text
boundaries              = 73-74, 145-146, 203-204
left window             = 40 images
right window            = 40 images
minimum sequence gap    = 41
candidates/boundary     = 780
total candidates        = 2,340
minimum verified inliers= 15
minimum inlier ratio    = 0.15
max selected/boundary   = 8
max endpoint reuse      = 2
```

The generated Step 12 candidate keys must be compared against the existing Step 11 candidate CSV before matching. The set/order must be identical. A mismatch is an integrity error, not a reason to silently generate a new search space.

For each frontend:

1. copy its validated feature cache to a diagnostic database;
2. write the exact 2,340 imported pair list;
3. match those pairs once with the frontend's explicit learned matcher;
4. let COLMAP perform the same geometric verification path;
5. measure raw match rows and verified two-view geometry rows from the database;
6. compute `verified_inliers / raw_matches`;
7. apply the frozen Step 11 qualification rule;
8. select at most eight deterministic bridges per boundary with the Step 11 endpoint-reuse rule.

Mapping is allowed only when every critical boundary has at least one selected qualified bridge.

## 10. Targeted learned mapping

When a frontend passes the boundary gate:

1. clone its validated feature cache into a fresh targeted database;
2. run learned sequential matching with overlap 20;
3. run learned imported-pair matching for only the selected learned bridges;
4. run the existing incremental mapper with the frozen Step 10/11 camera and mapper settings;
5. summarize every sparse model;
6. identify the strongest single model by:
   - registered images,
   - sparse points,
   - lower finite reprojection error.

Do not report disconnected model-union coverage as a registered-image result.

There is no learned exhaustive fallback in Step 12.

## 11. Global acceptance gate

The strongest single model for a frontend passes the automatic metric gate only when all of the following hold:

```text
registered_images >= 274
sparse_points >= 1000
camera_count == 1
camera_model == SIMPLE_RADIAL
mean_reprojection_error is finite
```

This retains the >=95% registration objective for the 288-image selected sequence.

Final Step 12 acceptance also requires manual/visual plausibility of:

- camera trajectory,
- sparse vessel geometry,
- absence of obvious catastrophic geometry collapse.

The machine-readable summary should therefore separate:

```text
metric_acceptance_met
visual_plausibility_status
learned_recovery_success
```

`learned_recovery_success` may become true only after both metric acceptance and the recorded visual plausibility review are positive.

No dense API may be invoked even when Step 12 succeeds.

## 12. ALIKED-to-LoMa fallback state machine

The orchestration is intentionally small:

```text
capability snapshot
  -> ALIKED smoke
  -> ALIKED extract
  -> ALIKED 2,340-pair diagnostics
     -> boundary gate fails ---------------------> LoMa smoke
     -> boundary gate passes
          -> ALIKED targeted map
             -> global acceptance passes --------> finalize and stop
             -> global acceptance fails ---------> LoMa smoke

LoMa smoke
  -> LoMa extract
  -> LoMa 2,340-pair diagnostics
     -> boundary gate fails ----------------------> finalize failure
     -> boundary gate passes
          -> LoMa targeted map
             -> global acceptance passes --------> finalize success
             -> global acceptance fails ---------> finalize failure
```

No third frontend is permitted.

## 13. Output contract

Step 12 writes only under:

```text
reconstruction/learned_recovery/
```

Proposed structure:

```text
reconstruction/learned_recovery/
  aliked/
  loma/
  best/
  reports/
    step12_capability.json
    step12_aliked_candidates.csv
    step12_aliked_boundary_summary.json
    step12_aliked_attempt.json
    step12_loma_candidates.csv
    step12_loma_boundary_summary.json
    step12_loma_attempt.json
    step12_attempts.csv
    step12_registered_images.csv
    step12_summary.json
  previews/
    step12_01_boundary_comparison.png
    step12_02_sparse_model.png
    step12_03_registration.png
    step12_04_frontend_comparison.png
  work/
```

Files for a frontend that never ran must not contain fabricated measurements. Its attempt status may be recorded as `not_run`; pair-level CSVs should exist only after real diagnostics run.

`work/` contains transient databases, pair lists, and completion markers. Cleanup must use an explicit allowlist and never traverse/delete arbitrary files.

The selected `best/` directory contains only the copied sparse model and derived visualization export from the strongest Step 12 model that actually ran. It must not overwrite Step 10 or Step 11 evidence.

## 14. Report semantics

### `step12_capability.json`

Runtime/options/smoke evidence and capability blockers.

### `step12_<frontend>_candidates.csv`

Exactly one row per measured frozen candidate, including:

- boundary
- indices and filenames
- sequence gap
- raw matches
- verified inliers
- inlier ratio
- qualified
- selected
- frontend

### `step12_<frontend>_boundary_summary.json`

Counts and extrema by the same three boundaries, plus the targeted gate result.

### `step12_<frontend>_attempt.json`

Status (`completed`, `skipped`, `blocked`, or `not_run`), reason, frontend identity, and targeted mapping metrics when mapping ran.

### `step12_attempts.csv`

One compact row for ALIKED and one for LoMa, keeping `not_run`/`skipped` state explicit. Only strongest single-model metrics populate model columns.

### `step12_registered_images.csv`

One row for each selected sequence index `1..288`, with filename and registered/unregistered status for the final selected Step 12 model.

### `step12_summary.json`

Must include:

- input/manifest provenance,
- runtime/frontends,
- exact frozen boundary configuration,
- ALIKED result/status,
- LoMa result/status,
- selected frontend/attempt if any,
- strongest single-model metrics,
- registered and unregistered filenames/indices,
- metric acceptance,
- visual plausibility review status,
- `learned_recovery_success`,
- `dense_reconstruction_started: false`,
- explicit next boundary.

## 15. Figures

All figures must use real measured project data.

### 15.1 Boundary comparison

`step12_01_boundary_comparison.png` compares verified inliers over the exact same 2,340 candidate keys for:

- Step 11 SIFT,
- ALIKED-LightGlue,
- LoMa-L only if LoMa diagnostics actually ran.

The Step 11 baseline must remain visually explicit:

```text
73-74   zero verified SIFT bridges
145-146 zero verified SIFT bridges
203-204 recoverable with SIFT
```

### 15.2 Sparse model

`step12_02_sparse_model.png` plots real 3D sparse points and registered camera centers from the final selected Step 12 model.

### 15.3 Registration

`step12_03_registration.png` plots selected indices `1..288` as registered/unregistered for the final selected model.

### 15.4 Frontend comparison

`step12_04_frontend_comparison.png` compares strongest **single-model** results only:

- Step 10 SIFT baseline,
- Step 11 exhaustive SIFT,
- Step 12 ALIKED if mapped,
- Step 12 LoMa if mapped.

Disconnected union coverage is never substituted for registered-image count.

## 16. Tests and verification boundaries

Deterministic logic is developed test-first.

New tests cover at minimum:

- frontend enum/config mapping,
- explicit learned matcher propagation to sequential and imported matching,
- CPU-only policy,
- capability snapshot serialization,
- smoke-result validation without requiring real model inference in unit tests,
- strict learned feature-cache identity validation,
- exact reuse of all 2,340 Step 11 candidate keys,
- frozen thresholds/selection/gate,
- ALIKED-first orchestration,
- LoMa fallback only when required,
- no LoMa execution after ALIKED acceptance,
- no mapping when a boundary gate fails,
- no exhaustive learned matching,
- single-model acceptance semantics,
- report/summary state for completed/skipped/blocked/not-run paths,
- transient cleanup allowlist,
- `dense_reconstruction_started == false`.

After unit tests pass, implementation execution should proceed through the staged real-data workflow. The actual learned smoke tests and 288-image run are implementation/runtime verification, not planning-time verification.

## 17. Failure boundary and next decision

If neither ALIKED-LightGlue nor LoMa-L yields an accepted, visually plausible >=274-image single sparse model:

```text
learned_recovery_success = false
dense_reconstruction_started = false
```

Step 12 stops there.

The next project decision must be explicit and separate:

1. accept a local-only sparse reconstruction, or
2. authorize a more experimental phase such as LoFTR/advanced global learned matching/component alignment.

Recapture is not a normal next step because the user has stated that recapture is not possible.

## 18. Non-goals

Step 12 does not:

- modify source JPEGs,
- modify the selected manifest,
- use CNN masks for reconstruction,
- rerun SIFT exhaustive matching,
- start dense reconstruction,
- rebuild pyCOLMAP,
- add external learned frameworks,
- try multiple matcher variants per family,
- perform camera-model sweeps,
- perform overlap/threshold sweeps,
- align disconnected sparse components manually,
- publish local model caches/checkpoints,
- modify DOCX/PDF reports,
- commit or push unless the later implementation request explicitly authorizes those external actions.
