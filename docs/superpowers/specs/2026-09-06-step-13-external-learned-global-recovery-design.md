# Step 13 External Learned Global Recovery Design

**Status:** Implemented and verified on 2026-09-06; global acceptance failed at 266/288, local fallback selected
**Canonical project:** `C:\Assumption University\CSX4213\Project`
**Baseline commit:** `27424a0774aae705575e67fc08452457f5ca2b1e`
**Decision:** Run one bounded external ALIKED + LightGlue recovery experiment. If it succeeds, use its accepted global sparse model. If it fails, stop sparse experimentation and continue with the best existing local sparse reconstruction.

## 1. Purpose

Step 13 is the final bounded sparse-recovery experiment for this project. Step 12 proved that the installed pyCOLMAP 4.2.0 wheel exposes native ALIKED/LightGlue enums but cannot execute ALIKED because the wheel lacks ONNX support. Step 13 changes only the learned inference runtime while preserving the existing image set, camera model, geometric verification, and pyCOLMAP mapping path.

The question Step 13 answers is deliberately narrow:

> Can one external ALIKED + LightGlue frontend establish geometrically valid bridges across the known sequence breaks and produce one accepted, visually plausible sparse reconstruction?

Step 13 does not start dense reconstruction.

## 2. Existing evidence

The immutable selected input remains 288 PREPROCESSED JPEGs under:

```text
preprocessing/pycolmap_input/images/
```

Selection manifest:

```text
preprocessing/reports/selection_manifest.csv
SHA-256: 79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91
```

Existing sparse evidence:

| Phase | Strongest single model | Sparse points | Mean reprojection error | Result |
| --- | ---: | ---: | ---: | --- |
| Step 10 SIFT | 73/288 | 6,099 | 1.2373 px | local only |
| Step 11 exhaustive SIFT | 73/288 | 3,443 | 1.1989 px | local only |

Step 11 also found 224 images only as a union across eight disconnected models. That union must never be reported as one registered model.

The frozen critical boundaries are:

```text
73-74
145-146
203-204
```

The exact Step 11 diagnostic search space is:

```text
window size                         40 left + 40 right
minimum sequence gap                41
candidate pairs per boundary        780
total candidate pairs               2,340
minimum verified inliers            15
minimum verified/raw inlier ratio   0.15
max selected bridges per boundary   8
max endpoint reuse                  2
```

SIFT found zero geometrically verified candidates at 73-74 and 145-146. Boundary 203-204 was recoverable.

## 3. Runtime decision

### 3.1 Selected architecture

Use the official CVG LightGlue package directly, pinned to the inspected upstream commit:

```text
repository: https://github.com/cvg/LightGlue.git
commit:     eb42fee2d71449efb0aa5c10549752b5d75384d8
frontend:   ALIKED + LightGlue
```

The project runtime already provides:

```text
Python       3.14.2
PyTorch      2.13.0+cu130
Torchvision  0.28.0+cu130
CUDA         available through PyTorch
GPU          NVIDIA GeForce RTX 5050 Laptop GPU
pyCOLMAP     4.2.0
```

The learned frontend runs in PyTorch. pyCOLMAP remains responsible for:

- importing the 288 images and the frozen shared camera model;
- storing imported keypoints and raw matches;
- two-view geometric verification through `pycolmap.verify_matches`;
- incremental mapping through the existing `map_sparse_database` helper;
- sparse model inspection and export.

### 3.2 Rejected alternatives

Do not use these inside Step 13:

1. **Rebuild/replace pyCOLMAP with ONNX-enabled COLMAP.** This is a larger toolchain change and is unnecessary when the current pyCOLMAP database/mapping APIs can consume externally computed keypoints and matches.
2. **Install the full hloc framework.** hloc demonstrates the same external-feature/imported-match architecture, but Step 13 needs only one frontend and a small subset of that integration. Pulling in the complete toolbox would add avoidable dependencies and abstraction.
3. **Try multiple learned frontends or parameter sweeps.** No SuperPoint/SuperGlue, LoFTR, LoMa, ALIKED variants, matcher zoo, or broad threshold search is allowed in this phase.

## 4. Protected boundaries

Read-only throughout Step 13:

```text
IMG20260826122949/
preprocessing/pycolmap_input/images/
preprocessing/reports/selection_manifest.csv
reconstruction/sparse/
reconstruction/bridging/
reconstruction/learned_recovery/
```

Also preserve:

```text
analysis/ml/checkpoints/best_small_seg_cnn.pt
.codegraph/
```

The CNN checkpoint remains private and must not be staged or published. DOCX/PDF artifacts must not be modified.

Step 13 writes only under its own source/docs paths and:

```text
reconstruction/external_learned_recovery/
```

## 5. Dependency contract

The project shall add exactly one learned-runtime dependency, pinned to the inspected commit:

```text
lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8
```

Do not upgrade or downgrade existing PyTorch, torchvision, pyCOLMAP, NumPy, OpenCV, Pillow, or CUDA packages merely to run Step 13.

If the pinned LightGlue package cannot import and run on the current Python/PyTorch environment without changing those established packages, Step 13 shall record a capability blocker and stop rather than initiating dependency churn.

## 6. Learned frontend configuration

Use one explicit ALIKED + LightGlue configuration. The implementation must record the effective values returned by the installed API.

Initial fixed intent:

```text
extractor               ALIKED
matcher                 LightGlue(features="aliked")
maximum image side      1600 px
maximum ALIKED keypoints 4096
inference device         CUDA when torch.cuda.is_available(), otherwise CPU
```

The original JPEGs remain unchanged. Any resize is in-memory only for learned inference. Returned ALIKED keypoints must be converted back to the original image coordinate system before they are written to the COLMAP database.

COLMAP keypoint coordinates use the pixel-center convention expected by the database import path. The adapter must apply the same coordinate convention consistently and must be regression-tested with synthetic scale cases.

## 7. Capability gate

Before any 288-image learned extraction:

1. verify all 288 selected images and the frozen manifest hash;
2. record Python, Torch, CUDA, GPU, pyCOLMAP, and LightGlue provenance;
3. load selected images 1 and 2 read-only;
4. run ALIKED extraction on both;
5. run LightGlue matching between them;
6. verify non-empty keypoints and matches;
7. verify all match indices are in bounds;
8. verify keypoints convert back to finite original-image coordinates.

Capability succeeds only if the actual external frontend performs real inference. Static import alone is insufficient.

If capability fails:

```text
step13_success = false
dense_reconstruction_started = false
selected_sparse_source = existing local sparse reconstruction
```

and no full extraction/matching/mapping runs.

## 8. Feature cache contract

ALIKED features are extracted once for the 288 selected images and stored as task-local reusable data under Step 13 `work/`.

The cache marker shall include at minimum:

- selection manifest SHA-256;
- exact ordered filenames;
- source image dimensions;
- frontend identity;
- pinned LightGlue source commit;
- Torch version;
- device policy;
- max image side;
- max keypoints;
- per-image keypoint count;
- deterministic fingerprint of the ordered feature layout.

A stale/partial cache must be rejected and rebuilt. Cache files are transient and are removed after final durable evidence is written.

No learned model weight cache downloaded into user/global cache locations is copied into the repository.

## 9. Exact boundary diagnostic

Step 13 reuses the Step 11 candidate generator and verifies exact identity/order against:

```text
reconstruction/bridging/reports/step11_candidates.csv
```

Exactly 2,340 candidate pairs are matched once with external LightGlue.

For the diagnostic database:

1. import the 288 image records with `CameraMode.SINGLE` and the existing `SIMPLE_RADIAL` camera initialization;
2. write external ALIKED keypoints for all required images;
3. write the raw LightGlue match indices for each candidate pair;
4. write the exact candidate pair list;
5. run `pycolmap.verify_matches` once over those imported matches;
6. summarize raw matches and verified two-view rows with the existing Step 11 helpers;
7. apply the unchanged Step 11 qualification and bridge-selection rules.

Mapping is allowed only when every critical boundary has at least one selected qualified bridge.

This diagnostic is the key early stop. If either 73-74 or 145-146 still has no selected qualified bridge, Step 13 stops before a full learned map.

## 10. Full learned mapping pair schedule

Only after the diagnostic gate passes, build one fresh mapping database using:

1. deterministic sequential pairs where `1 <= right_index - left_index <= 20`;
2. the selected qualified cross-boundary bridge pairs;
3. duplicate pairs removed deterministically.

For 288 images and overlap 20, the sequential schedule contains exactly 5,550 unique pairs before bridge de-duplication.

The implementation shall:

- reuse the validated ALIKED feature cache;
- run LightGlue only for the required mapping pairs;
- import raw matches;
- geometrically verify all imported pairs with pyCOLMAP;
- run the existing frozen incremental mapper and camera configuration;
- summarize every sparse model;
- rank strongest single models only.

There is no learned exhaustive matcher and no second frontend fallback.

## 11. Global acceptance gate

The strongest Step 13 single sparse model passes the automatic metric gate only when all conditions hold:

```text
registered_images >= 274
sparse_points >= 1000
camera_count == 1
camera_model == SIMPLE_RADIAL
mean_reprojection_error is finite
```

Final success also requires visual plausibility of:

- camera trajectory;
- sparse vessel/object geometry;
- absence of obvious collapse, explosion, or multiple unrelated coordinate frames.

Machine-readable status shall separate:

```text
metric_acceptance_met
visual_plausibility_status
step13_success
```

`step13_success` becomes true only when both metric and visual gates pass.

## 12. Local fallback selection

If Step 13 fails, downstream reconstruction must use one existing local model rather than run more sparse recovery experiments.

Choose between the Step 10 and Step 11 strongest local models using the existing strongest-single-model ranking:

1. more registered images;
2. more sparse points;
3. lower finite reprojection error.

With the currently measured values this selects the Step 10 model because both register 73 images and Step 10 has 6,099 points versus 3,443 for Step 11. This choice must be reverified from the actual model/report at finalization instead of hard-coding the outcome.

The Step 13 summary shall record:

```text
selected_sparse_source
selected_sparse_reason
```

If Step 13 succeeds, the source is Step 13 `best/`. If it fails, the source is the verified existing local model selected by the rule above.

## 13. Output contract

```text
reconstruction/external_learned_recovery/
  reports/
    step13_capability.json
    step13_candidates.csv                 # only after real diagnostic matching
    step13_boundary_summary.json          # only after real diagnostic matching
    step13_attempt.json
    step13_summary.json
    step13_registered_images.csv          # only when a Step 13 sparse model exists
  previews/
    step13_01_boundary_comparison.png      # only after diagnostics
    step13_02_sparse_model.png             # only when mapped
    step13_03_registration.png             # only when mapped
    step13_04_model_comparison.png
  best/                                    # only when Step 13 maps
  work/                                    # transient
```

No report may contain fabricated pair/model measurements. Files for stages that did not run are omitted or represented only by explicit status fields in the durable attempt/summary.

## 14. Summary contract

`step13_summary.json` must include:

- selected-input provenance and manifest hash;
- external frontend/pinned-source provenance;
- runtime/device details;
- exact boundary configuration;
- capability result;
- diagnostic boundary result if run;
- mapping result if run;
- strongest Step 13 single-model metrics if one exists;
- metric and visual acceptance states;
- `step13_success`;
- `dense_reconstruction_started: false`;
- existing local fallback comparison;
- `selected_sparse_source` and reason;
- explicit next boundary.

## 15. Figures

All figures use real project evidence only.

### 15.1 Boundary comparison

Compare Step 11 SIFT versus Step 13 ALIKED+LightGlue over the exact same 2,340 candidate keys. Keep zero-inlier SIFT boundaries visibly at zero and all axes nonnegative.

### 15.2 Sparse model

If mapping runs, render real Step 13 sparse points and registered camera centers.

### 15.3 Registration

If mapping runs, show binary registered/unregistered state for selected indices 1..288 and mark all three critical boundaries.

### 15.4 Model comparison

Compare strongest single models only:

- Step 10 SIFT;
- Step 11 exhaustive SIFT;
- Step 13 external ALIKED+LightGlue if mapped.

Never substitute disconnected union coverage for registered-image count.

## 16. Verification requirements

Deterministic code is developed test-first. Tests cover at minimum:

- pinned frontend/dependency identity;
- image resize and original-coordinate keypoint restoration;
- match-array shape/index validation;
- deterministic 5,550 sequential pair schedule;
- selected bridge merge/de-duplication;
- strict feature-cache provenance;
- COLMAP image/keypoint/match import contract;
- exact 2,340 Step 11 candidate identity reuse;
- frozen qualification/selection rules;
- diagnostic gate preventing mapping when any boundary fails;
- one mapping attempt only after gate success;
- strongest-single-model acceptance;
- local fallback ranking;
- truthful report states;
- cleanup allowlist;
- no dense API invocation.

After unit tests:

1. run real capability smoke;
2. run real 2,340-pair diagnostics;
3. map only if diagnostic gate passes;
4. finalize reports/figures;
5. inspect every generated figure;
6. run the full relevant and complete project test suites;
7. reverify raw/selected image integrity;
8. re-open Step 10/11 and Step 13 models as applicable;
9. confirm protected Step 10/11 evidence was not altered;
10. inspect final Git diff and staged file set.

## 17. Stop conditions

Step 13 immediately stops sparse recovery when any of these is true:

- external LightGlue cannot import/run without disruptive package changes;
- capability smoke fails;
- any critical boundary fails the diagnostic gate;
- mapping runs but strongest single model fails metric acceptance;
- metric acceptance passes but visual plausibility fails.

No second learned frontend or additional tuning phase is added automatically.

## 18. Final decision

### Success

If Step 13 produces a visually plausible accepted global model:

```text
step13_success = true
selected_sparse_source = reconstruction/external_learned_recovery/best
dense_reconstruction_started = false
```

The next separately executed project phase may begin dense reconstruction from that global sparse model.

### Failure

If Step 13 does not produce an accepted global model:

```text
step13_success = false
selected_sparse_source = verified existing local model
dense_reconstruction_started = false
```

Sparse recovery experimentation ends. The next project phase uses that local component for a best-effort local dense reconstruction, mesh, texture, and Blender deliverable, with the incomplete-view limitation documented explicitly.
