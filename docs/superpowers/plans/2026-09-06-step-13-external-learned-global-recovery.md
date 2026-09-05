# Step 13 External Learned Global Recovery Implementation Plan

> Execute this plan sequentially. Step 13 is the final bounded sparse-recovery experiment. Do not add a second learned frontend, do not broaden into parameter sweeps, and do not start dense reconstruction.

**Goal:** Test one pinned external ALIKED + LightGlue frontend against the exact failed Step 11 boundaries and, only if those boundaries recover, attempt one full learned sparse reconstruction. If the accepted global-model gate fails, select the verified best existing local sparse model and close sparse recovery.

**Spec:** `docs/superpowers/specs/2026-09-06-step-13-external-learned-global-recovery-design.md`

**Baseline:** `27424a0774aae705575e67fc08452457f5ca2b1e` on `main`.

**Execution outcome:** Completed on 2026-09-06. The external learned diagnostic recovered all three fixed boundaries and the one full mapping attempt reached 266/288 registered images with 29,713 points, but the frozen >=274-image acceptance gate failed. Per this plan, no retry/sweep was run; Step 10 `reconstruction/sparse/best` is the selected local downstream source and sparse recovery is closed. Measured evidence is documented in `docs/geometry-ml/external-learned-global-recovery.md`.

## Frozen decisions

```text
External package repository        https://github.com/cvg/LightGlue.git
Pinned commit                      eb42fee2d71449efb0aa5c10549752b5d75384d8
Frontend                           ALIKED + LightGlue(features="aliked")
Max inference image side           1600
Max ALIKED keypoints               4096
Preferred device                   CUDA when available, CPU otherwise
Selected images                    288 PREPROCESSED JPEGs
Selection manifest SHA-256         79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91
Camera mode                        SINGLE
Camera model                       SIMPLE_RADIAL
Initial focal                      3069.0506675517754 px
Diagnostic candidates              exact Step 11 2,340 pairs
Boundaries                         73-74, 145-146, 203-204
Min verified inliers               15
Min inlier ratio                   0.15
Max selected/boundary              8
Max endpoint reuse                 2
Mapping sequential overlap         20
Sequential pair count              5,550
Global min registered              274
Global min sparse points           1,000
Required camera count/model        1 / SIMPLE_RADIAL
Dense reconstruction               forbidden in Step 13
```

## Task 1 — Freeze live state and protected evidence

1. Confirm Git root, branch, HEAD, `origin/main`, and status.
2. Confirm only expected local-only `.codegraph/` and `analysis/ml/checkpoints/` are untracked before Step 13 work.
3. Reverify 288 selected inputs and manifest SHA-256.
4. Reverify 297 raw images against the published raw manifest.
5. Record hashes for protected Step 10/11 reports used to detect accidental modification.
6. Reopen Step 10 and Step 11 selected sparse models and record their strongest-single-model metrics.
7. Do not write under raw, selected, Step 10, Step 11, or Step 12 evidence roots.

**Acceptance:** protected evidence is internally consistent before implementation changes.

## Task 2 — Validate the external dependency decision before installation

1. Confirm the official LightGlue repository HEAD and pin the inspected commit.
2. Dry-run installation of the exact commit.
3. Inspect declared package dependencies.
4. Confirm current Torch/Torchvision/PyCOLMAP versions remain sufficient.
5. Add exactly one pinned direct dependency to `requirements.txt` only if no established package upgrades/downgrades are required.
6. Install the pinned package without upgrading unrelated dependencies.
7. Confirm import and expose the actual installed package/source provenance.

**Stop condition:** if installation requires disruptive changes to established project packages, write a capability-blocked Step 13 result and stop learned execution.

## Task 3 — Define deterministic Step 13 domain contracts test-first

**Create:**

```text
external_learned_recovery.py
tests/test_external_learned_recovery.py
```

Write failing tests first for:

1. frozen config values and pinned commit;
2. runtime device resolution (`cuda` preferred when available);
3. image loading/resizing without mutating source files;
4. restoration of resized ALIKED keypoints to original pixel coordinates;
5. finite keypoint validation;
6. match array shape `(N, 2)`, nonnegative indices, and feature-count bounds;
7. deterministic sequential pair generation:
   - 288 images;
   - offsets 1..20;
   - exactly 5,550 unique ordered pairs;
8. merging selected cross-boundary bridges without duplicates;
9. local fallback ranking: registered images, points, lower finite error;
10. global metric acceptance conditions.

Implement only enough production logic to make each test pass.

## Task 4 — Build a narrow LightGlue adapter

Keep external-library code behind a small interface so the rest of the project remains testable without GPU/model downloads.

Required operations:

```python
load_frontend(...)
extract_image_features(...)
match_feature_pair(...)
```

The adapter shall:

- instantiate `ALIKED(max_num_keypoints=4096)`;
- instantiate `LightGlue(features="aliked")`;
- move both models to one resolved Torch device;
- use eval/inference mode;
- resize in memory to max side 1600;
- return original-coordinate float32 keypoints;
- return uint32 COLMAP-compatible match index pairs;
- never write model weights into the repository.

Use injected/fake extractors/matchers in unit tests. Real inference is reserved for the capability stage.

## Task 5 — Implement strict external feature cache

Use Step 13-local transient files under:

```text
reconstruction/external_learned_recovery/work/features/
```

A practical cache format may use one compressed NumPy file per image plus one JSON marker/index. Do not introduce a new serialization dependency.

Marker identity must cover:

- ordered filename list;
- source dimensions;
- selection manifest SHA-256;
- pinned upstream commit;
- frontend/config;
- Torch version;
- device policy;
- per-image keypoint count;
- deterministic layout fingerprint.

Tests must prove a changed filename, dimension, keypoint count/layout, manifest hash, config, or pinned commit invalidates cache reuse.

## Task 6 — Implement COLMAP external-feature database import test-first

Create helpers that:

1. create a fresh DB with `pycolmap.import_images` using the existing `build_image_reader_options(SparseRunConfig())`;
2. load image IDs by filename;
3. write ALIKED keypoints with `Database.write_keypoints`;
4. write imported raw matches with `Database.write_matches`;
5. write an exact pair-list text file;
6. run `pycolmap.verify_matches`;
7. summarize database/match results with existing project helpers.

Use a tiny real temporary pyCOLMAP database test where practical. Keep unit fixtures small and local.

Key invariants:

- exactly one shared `SIMPLE_RADIAL` camera;
- 288 image names on real runs;
- matches reference valid keypoint rows;
- no descriptors are required for mapping after imported matches are geometrically verified;
- no default native feature matcher is called.

## Task 7 — Freeze Step 11 diagnostic identity

Reuse rather than duplicate:

```text
BridgeSearchConfig
generate_candidate_pairs
verify_step11_candidate_identity
select_bridge_pairs
boundary_bridge_summary
targeted_gate
summarize_bridge_pairs
```

Tests must retain:

```text
2,340 total candidates
780 / boundary
minimum gap 41
exact Step 11 identity and order
same qualification/selection thresholds
```

No Step 13-specific threshold tuning is allowed.

## Task 8 — Implement staged runner test-first

**Create:**

```text
run_external_learned_recovery.py
tests/test_run_external_learned_recovery.py
```

Expose exactly these stages:

```text
capability
diagnose
map
finalize
all
```

Runner tests shall prove:

1. `capability` verifies inputs and runs only a two-image real-inference adapter call;
2. blocked capability prevents all full extraction/matching/mapping;
3. `diagnose` uses exactly the 2,340 frozen candidate pairs;
4. diagnostic gate failure prevents `map`;
5. `map` uses exactly one deterministic learned mapping attempt;
6. no second frontend/fallback exists;
7. no exhaustive learned matching exists;
8. finalization selects Step 13 only after metric + visual acceptance;
9. failure finalization chooses the best existing local model by frozen ranking;
10. `dense_reconstruction_started` is always false.

## Task 9 — Implement real capability report

Write:

```text
reconstruction/external_learned_recovery/reports/step13_capability.json
```

Record:

- Python version;
- Torch/Torchvision versions;
- CUDA availability and CUDA version;
- GPU name when available;
- pyCOLMAP version;
- LightGlue package/source identity;
- pinned commit;
- fixed frontend parameters;
- selection manifest hash and image count;
- smoke image identities;
- feature counts and raw match count;
- exact exception class/message if blocked.

Run real ALIKED + LightGlue inference on selected indices 1 and 2.

**Continue only if:** both feature sets and raw matches are non-empty and structurally valid.

## Task 10 — Run full 288-image ALIKED extraction

Only after capability passes:

1. extract/cache all 288 images;
2. validate every cache record;
3. verify all keypoints are finite and inside a tolerance of image bounds;
4. record extraction counts/runtime;
5. preserve no source-image writes.

Do not match mapping pairs yet.

## Task 11 — Execute exact 2,340-pair learned diagnostic

1. regenerate Step 11 candidates;
2. verify exact identity/order;
3. match each pair with LightGlue using cached ALIKED features;
4. import keypoints/raw matches into a diagnostic COLMAP DB;
5. run pyCOLMAP geometric verification;
6. write `step13_candidates.csv`;
7. select bridges using frozen Step 11 logic;
8. write `step13_boundary_summary.json`;
9. render boundary-comparison figure.

**Gate:** all three boundaries require at least one selected qualified bridge.

If the gate fails, write a skipped mapping attempt and move directly to finalization/local fallback.

## Task 12 — Execute one full learned mapping attempt only if diagnostic gate passes

Build deterministic pair schedule:

```text
5,550 sequential pairs (offset <= 20)
+ selected qualified bridge pairs
- duplicates
```

Then:

1. run LightGlue for the required mapping pairs;
2. build a fresh COLMAP DB from ALIKED keypoints and raw matches;
3. geometrically verify imported pairs;
4. run existing incremental mapping;
5. summarize all produced models;
6. choose strongest single model;
7. apply the fixed >=274 / >=1000 / camera / finite-error metric gate;
8. write `step13_attempt.json`.

No retry, second matcher, or parameter sweep is permitted.

## Task 13 — Finalize and choose the downstream sparse source

Finalization shall always reverify selected inputs and reopen candidate source models.

### If Step 13 maps

- copy only its selected strongest model to Step 13 `best/`;
- reopen and verify copied metrics;
- export PLY;
- write 288-row registered-image report;
- render sparse, registration, and model-comparison figures.

### Local fallback

Reopen Step 10 and Step 11 best local models and rank them by:

```text
registered images desc
sparse points desc
finite reprojection error asc
```

Record the winner even if Step 13 succeeds, for auditability.

### Visual gate

If Step 13 meets metric acceptance, inspect the actual sparse/camera figures and record `passed` or `failed` through a runner flag, not by hand-editing JSON.

Derive:

```text
step13_success = metric_acceptance_met and visual_status == "passed"
```

Final `selected_sparse_source`:

- Step 13 `best/` if success;
- verified local fallback if failure.

Always record:

```text
dense_reconstruction_started = false
```

## Task 14 — Safe cleanup

Remove only exact Step 13 transient files/directories after durable reports are written:

- cached `.npz` features;
- diagnostic/mapping temporary DBs;
- temporary pair lists;
- temporary marker files;
- task-created Python caches.

Do not delete unknown files, Step 10/11/12 evidence, selected/raw images, `.codegraph/`, or the CNN checkpoint.

## Task 15 — Measured Step 13 documentation

Create:

```text
docs/geometry-ml/external-learned-global-recovery.md
```

Update:

```text
README.md
AGENTS.md
CHANGELOG.md
LESSONS.md                    only for durable process lessons
docs/memory-bank/active-context.md
docs/memory-bank/progress.md
```

Document only real runtime evidence. Explicitly distinguish:

- dependency/capability evidence;
- boundary diagnostic measurements;
- mapping measurements if run;
- skipped stages;
- final global success/failure;
- selected downstream sparse source;
- dense-not-started boundary.

## Task 16 — Review and verification

### 16.1 Spec-compliance review

Check implementation against every frozen Step 13 constraint:

- one frontend only;
- exact pinned dependency;
- exact 2,340 diagnostic pairs;
- unchanged thresholds;
- no mapping before diagnostic gate;
- one map attempt max;
- strongest single model only;
- local fallback deterministic;
- no dense API.

### 16.2 Code-quality review

Inspect:

- error handling and blocked states;
- cache provenance;
- coordinate restoration correctness;
- match index validation;
- SQLite/pyCOLMAP connection lifecycle on Windows;
- deterministic ordering;
- cleanup safety;
- report truthfulness;
- avoidance of duplicated Step 10/11 logic.

Fix actionable defects and add narrow regressions before publication.

### 16.3 Automated verification

Run focused Step 13 tests, Step 10-12 regressions, then full project tests.

Compile touched Python without retaining cache residue.

### 16.4 Evidence verification

Reverify:

- 297 raw images unchanged;
- 288 selected images unchanged and manifest hash exact;
- protected Step 10/11 report hashes unchanged;
- Step 10/11 source sparse models reopen with expected metrics;
- Step 13 model/report agreement when mapped;
- no dense outputs;
- no transient Step 13 work remains after cleanup.

Visually inspect every generated Step 13 figure.

## Task 17 — Git publication

Before staging:

1. fetch `origin/main` and confirm no unexpected divergence;
2. run whitespace/document-link/preflight checks;
3. inspect full intended diff;
4. exclude `.codegraph/`, private checkpoint, model/download caches, temporary DBs, and unrelated work;
5. stage only intended Step 13 source/tests/docs/evidence plus the pinned requirement change;
6. commit with a scoped conventional message;
7. push `main` normally, never force;
8. fetch and verify `HEAD == origin/main == remote main`;
9. report any intentionally local untracked paths.

## Final acceptance checklist

Step 13 is complete when all applicable items are resolved truthfully:

- [ ] Detailed design and plan written before implementation.
- [ ] Protected input/evidence baseline verified.
- [ ] Exact LightGlue commit pinned.
- [ ] No established dependency upgraded/downgraded unnecessarily.
- [ ] TDD coverage added for deterministic logic.
- [ ] Real external ALIKED + LightGlue smoke executed.
- [ ] Exact 2,340 candidate identity retained if diagnostics ran.
- [ ] Mapping ran only if all three boundaries qualified.
- [ ] At most one learned mapping attempt ran.
- [ ] No second frontend, exhaustive learned matcher, or parameter sweep ran.
- [ ] Strongest single-model gate used.
- [ ] Visual acceptance recorded when applicable.
- [ ] Deterministic local fallback selected on failure.
- [ ] `dense_reconstruction_started == false`.
- [ ] Raw/selected images unchanged.
- [ ] Step 10/11/12 evidence unchanged.
- [ ] Focused/regression/full tests passed or any failure is explicitly separated.
- [ ] Touched Python compiles.
- [ ] Generated figures inspected.
- [ ] Transient Step 13 work cleaned.
- [ ] Docs match measured outcome.
- [ ] Intended diff reviewed and preflighted.
- [ ] Private/local-only files excluded from Git.
- [ ] Commit and push verified on `origin/main`.

## Final branch after Step 13

### If Step 13 succeeds

Use its accepted global sparse model. Remaining reconstruction work is dense MVS, meshing, texturing, Blender cleanup, final validation, and coursework deliverables.

### If Step 13 fails

Sparse recovery ends. Use the deterministically selected existing local sparse model and continue with a best-effort local dense reconstruction, meshing, texturing, Blender cleanup, final validation, and coursework deliverables. The final report must state that the 3D result covers only the locally reconstructed capture arc rather than the complete 288-image sequence.
