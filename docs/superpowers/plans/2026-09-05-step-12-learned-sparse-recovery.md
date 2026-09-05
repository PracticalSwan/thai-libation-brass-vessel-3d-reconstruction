# Step 12 Learned Sparse Recovery Implementation Plan

> **For the implementation agent:** Execute this plan sequentially. Recover live state before editing. Use test-driven development for deterministic logic. Do not broaden Step 12 beyond the two native learned frontends defined here.

**Goal:** Determine whether native pyCOLMAP learned features/matchers can recover the three known sparse sequence breaks and produce one visually plausible sparse model registering at least 274 of the immutable 288 selected images, without starting dense reconstruction.

**Architecture:** Add one learned-recovery domain module and one staged runner. Reuse the Step 10 camera/mapper/model helpers and Step 11 candidate/qualification/selection logic. Run ALIKED-N16Rot + LightGlue first, and run exactly one LoMa-B + LoMa-L fallback only if ALIKED fails the boundary or global acceptance gate. Keep all Step 12 artifacts under `reconstruction/learned_recovery/` and preserve Step 10/11 evidence unchanged.

**Tech Stack:** Python 3.14, pyCOLMAP 4.2.x (live runtime 4.2.0), SQLite COLMAP database, NumPy, Pillow, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-step-12-learned-sparse-recovery-design.md`

**Implementation state (2026-09-06):** Complete at the approved native capability failure boundary. The reviewed domain/runner implementation passed 20/20 domain and 14/14 runner tests before execution. The real ALIKED smoke then stopped with `RuntimeError: ALIKED feature extraction requires ONNX support.` Runtime/final-review reporting regressions brought the runner suite to 16 tests and the complete project suite to 177 tests. Full learned extraction/matching/mapping and all LoMa inference were intentionally not run.

**Final boundary:** Step 12 is closed at its specified capability blocker. Do not rerun later learned stages, rebuild/replace pyCOLMAP, introduce an external learned stack, or start dense reconstruction without a separate explicit architecture decision.

## Global Constraints

- Canonical project: `C:\Assumption University\CSX4213\Project`.
- Read `AGENTS.md`, `CLAUDE.md`, `LESSONS.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md` before editing.
- Expected implementation baseline: `ff57a67f6989432ca88d6559ad40507037f7dede` on `main`, synchronized with `origin/main` unless newer user-authorized work exists.
- Preserve unrelated user work and expected local paths `.codegraph/` and `analysis/ml/checkpoints/`.
- Never publish `analysis/ml/checkpoints/best_small_seg_cnn.pt` or learned model caches.
- Protected/read-only inputs:
  - `IMG20260826122949/`
  - `preprocessing/pycolmap_input/images/`
  - `preprocessing/reports/selection_manifest.csv`
- Preserve all Step 10/11 evidence under `reconstruction/sparse/` and `reconstruction/bridging/`.
- Do not modify DOCX/PDF reports.
- Do not recapture or recommend recapture as the normal execution path.
- Do not run dense reconstruction or call dense APIs.
- Do not add hloc, external LightGlue, LoFTR, a new COLMAP build, or a new ONNX stack.
- Do not perform exhaustive learned matching.
- Do not create a learned matcher benchmark zoo or parameter sweep.
- Do not change `requirements.txt` unless implementation proves the existing `pycolmap>=4.2,<5` contract is factually insufficient; if that happens, stop and report rather than silently changing architecture.
- No commit or push unless the implementation request explicitly authorizes it.

## Frozen Experimental Constants

```text
input images                         288 PREPROCESSED JPEGs
selection manifest SHA-256           79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91
camera mode                          SINGLE
camera model                         SIMPLE_RADIAL
initial focal                        3069.0506675517754 px
learned max image size               1600
learned device                       CPU
primary extractor                    ALIKED_N16ROT
primary matcher                      ALIKED_LIGHTGLUE
fallback extractor                   LOMA_B
fallback matcher                     LOMA_L
critical boundaries                  73-74, 145-146, 203-204
window size                          40 left + 40 right
minimum sequence gap                 41
candidates/boundary                  780
total candidates                     2340
minimum verified inliers             15
minimum inlier ratio                 0.15
max selected bridges/boundary        8
max endpoint reuse                   2
targeted sequential overlap          20
minimum registered images            274
minimum sparse points                1000
required camera count                1
required camera model                SIMPLE_RADIAL
```

---

## Task 1: Recover and freeze the live implementation baseline

**Files:**
- Read: `AGENTS.md`
- Read: `CLAUDE.md`
- Read: `LESSONS.md`
- Read: `docs/memory-bank/active-context.md`
- Read: `docs/memory-bank/progress.md`
- Read: `docs/geometry-ml/sparse-reconstruction.md`
- Read: `docs/geometry-ml/sparse-component-bridging.md`
- Read: `docs/superpowers/specs/2026-09-05-step-12-learned-sparse-recovery-design.md`
- Read: `sparse_reconstruction.py`
- Read: `sparse_bridging.py`
- Read: `run_sparse_bridging.py`
- Read: `tests/test_sparse_bridging.py`
- Read: `tests/test_run_sparse_bridging.py`
- Read: `reconstruction/reports/step10_summary.json`
- Read: `reconstruction/bridging/reports/step11_summary.json`
- Read: `reconstruction/bridging/reports/step11_candidates.csv`
- Read: `reconstruction/bridging/reports/step11_boundary_summary.json`

### Step 1.1: Verify Git/worktree state before edits

Run:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git rev-parse origin/main
git status --short --branch
```

Expected baseline when no newer authorized work exists:

```text
branch      main
HEAD        ff57a67f6989432ca88d6559ad40507037f7dede
origin/main ff57a67f6989432ca88d6559ad40507037f7dede
```

Expected intentional untracked paths:

```text
.codegraph/
analysis/ml/checkpoints/
```

If live state is newer, inspect the newer diff/history and adapt the plan to the authoritative live state. Do not reset or discard user work.

### Step 1.2: Reverify immutable selected input

Use the existing selected-image verifier rather than writing a new integrity script. Confirm:

```text
288 selected images
manifest hash = 79408d59...f37a91
```

No write may target the selected image directory.

### Step 1.3: Confirm no Step 12 implementation already exists

Check for:

```text
learned_sparse_recovery.py
run_learned_sparse_recovery.py
tests/test_learned_sparse_recovery.py
tests/test_run_learned_sparse_recovery.py
reconstruction/learned_recovery/
```

If any exist, inspect them before editing and continue from actual state instead of recreating work.

---

## Task 2: Define learned frontend contracts test-first

**Files:**
- Create: `learned_sparse_recovery.py`
- Create: `tests/test_learned_sparse_recovery.py`
- Reuse: `sparse_reconstruction.py`
- Reuse: `sparse_bridging.py`

### Step 2.1: Write failing tests for frontend identities

Add tests requiring exactly these two specs:

```python
ALIKED_FRONTEND = LearnedFrontendSpec(
    name="aliked",
    extractor_type=pycolmap.FeatureExtractorType.ALIKED_N16ROT,
    matcher_type=pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE,
    max_image_size=1600,
)

LOMA_FRONTEND = LearnedFrontendSpec(
    name="loma",
    extractor_type=pycolmap.FeatureExtractorType.LOMA_B,
    matcher_type=pycolmap.FeatureMatcherType.LOMA_L,
    max_image_size=1600,
)
```

Test that only `aliked` and `loma` are accepted frontend names and that no third frontend constant exists in the Step 12 registry.

Run:

```bash
python -m pytest tests/test_learned_sparse_recovery.py -q
```

Expected: FAIL because the module/types do not exist yet.

### Step 2.2: Implement the minimum frontend dataclasses/config

In `learned_sparse_recovery.py`, add:

```python
@dataclass(frozen=True)
class LearnedFrontendSpec:
    name: str
    extractor_type: pycolmap.FeatureExtractorType
    matcher_type: pycolmap.FeatureMatcherType
    max_image_size: int = 1600

@dataclass(frozen=True)
class LearnedRecoveryConfig:
    expected_images: int = 288
    minimum_registered_images: int = 274
    minimum_sparse_points: int = 1000
    targeted_overlap: int = 20
    visual_status_pending: str = "pending"
```

Do not duplicate camera or mapper settings into this config. Those remain `SparseRunConfig` responsibilities.

### Step 2.3: Write failing tests for extraction option construction

Required behavior:

```python
build_learned_extraction_options(ALIKED_FRONTEND)
```

must produce:

```text
type           ALIKED_N16ROT
max_image_size 1600
use_gpu        False
check()        True
```

and LoMa must analogously produce `LOMA_B`.

Do not assert or alter `options.sift.max_num_features` for learned extraction. It is not the learned model control.

### Step 2.4: Implement extraction options

Implement:

```python
def build_learned_extraction_options(
    frontend: LearnedFrontendSpec,
) -> pycolmap.FeatureExtractionOptions:
    ...
```

Set only the learned type, explicit max image size, and CPU policy needed by this phase.

### Step 2.5: Write failing tests for matching option construction

Require:

```python
build_learned_matching_options(ALIKED_FRONTEND).type
    == FeatureMatcherType.ALIKED_LIGHTGLUE

build_learned_matching_options(LOMA_FRONTEND).type
    == FeatureMatcherType.LOMA_L
```

Both must set `use_gpu=False` and pass `check()`.

### Step 2.6: Implement matching options

Implement:

```python
def build_learned_matching_options(
    frontend: LearnedFrontendSpec,
) -> pycolmap.FeatureMatchingOptions:
    ...
```

### Step 2.7: Verify Task 2

Run:

```bash
python -m pytest tests/test_learned_sparse_recovery.py -q
```

Expected: PASS for the new config/option tests.

Do not run model inference yet.

---

## Task 3: Implement the capability report and in-memory smoke gate test-first

**Files:**
- Modify: `learned_sparse_recovery.py`
- Modify: `tests/test_learned_sparse_recovery.py`
- Create later through runner: `reconstruction/learned_recovery/reports/step12_capability.json`

### Step 3.1: Write failing tests for the static runtime snapshot

Implement a pure snapshot builder that records at minimum:

```text
python_version
pycolmap_version
has_cuda
extractor_members
matcher_members
device_policy
frontends.aliked.extractor
frontends.aliked.matcher
frontends.aliked.max_image_size
frontends.aliked.effective_max_image_size
frontends.aliked.requires_rgb
frontends.aliked.requires_opengl
frontends.aliked.extraction_options_valid
frontends.aliked.matching_options_valid
frontends.loma.*
```

The test must assert that the required four enum names are present.

### Step 3.2: Implement `build_capability_snapshot()`

Keep it side-effect free: option construction/introspection only. No model creation and no network/model download.

### Step 3.3: Define smoke result data structure

Add:

```python
@dataclass(frozen=True)
class FrontendSmokeResult:
    frontend: str
    status: str
    first_features: int
    second_features: int
    raw_matches: int
    exception_type: str = ""
    exception_message: str = ""
```

Allowed statuses:

```text
passed
blocked
not_run
```

### Step 3.4: Write tests for smoke validation using fakes

Unit tests must not require learned model inference. Inject or monkeypatch extractor/matcher factories and assert:

- RGB `uint8` arrays are passed to extraction;
- non-empty keypoints/descriptors are required;
- per-image keypoint/descriptor counts must agree;
- returned match array must have shape `(N, 2)`;
- `N > 0` is required for `passed`;
- native exceptions become a structured `blocked` result without hiding exception class/message.

### Step 3.5: Implement `smoke_frontend(...)`

Suggested signature:

```python
def smoke_frontend(
    frontend: LearnedFrontendSpec,
    first_image: np.ndarray,
    second_image: np.ndarray,
    *,
    extractor_factory: Callable[..., object] = pycolmap.FeatureExtractor.create,
    matcher_factory: Callable[..., object] = pycolmap.FeatureMatcher.create,
) -> FrontendSmokeResult:
    ...
```

Create both native objects with `pycolmap.Device.cpu`.

Do not perform geometric verification in the smoke. Its only purpose is native learned model construction/extraction/matching capability.

### Step 3.6: Choose a deterministic known-overlap smoke pair

Do not invent new image selection logic. Use two adjacent selected images from the already verified sequence that Step 10 successfully registered together. Prefer the first two selected images because they are known to belong to the 73-image baseline component:

```text
selected index 1: IMG20260826122949.jpg
selected index 2: IMG20260826122953.jpg
```

The runner must resolve these through the verified manifest records, not by hard-coded absolute paths.

### Step 3.7: Verify Task 3 unit tests

Run:

```bash
python -m pytest tests/test_learned_sparse_recovery.py -q
```

Expected: PASS without downloading/running real learned models.

---

## Task 4: Execute the real ALIKED capability gate before any 288-image learned extraction

**Files:**
- Create: `run_learned_sparse_recovery.py`
- Create: `tests/test_run_learned_sparse_recovery.py`
- Runtime output: `reconstruction/learned_recovery/reports/step12_capability.json`

### Step 4.1: Write runner-stage tests first

Define runner paths as module constants, matching the project pattern:

```text
LEARNED_ROOT
ALIKED_DIR
LOMA_DIR
BEST_DIR
REPORTS_DIR
PREVIEWS_DIR
WORK_DIR
CAPABILITY_JSON
...
```

Add a `capability` stage and test that it:

1. verifies selected inputs first;
2. builds the static snapshot;
3. loads the two smoke images read-only as RGB;
4. runs only the ALIKED smoke initially;
5. records LoMa smoke as `not_run`;
6. writes JSON only under Step 12 output root;
7. does not create any sparse model directory;
8. does not call any dense API.

### Step 4.2: Implement capability stage

Use Pillow or the project's existing safe image-loading pattern to produce RGB `uint8` NumPy arrays.

Write JSON atomically using the same `*.tmp -> replace` pattern used for durable completion markers.

### Step 4.3: Run focused runner tests

```bash
python -m pytest tests/test_run_learned_sparse_recovery.py -q
```

Expected: PASS with fakes.

### Step 4.4: Run real capability stage

Only after focused tests pass:

```bash
python run_learned_sparse_recovery.py --stage capability
```

Inspect:

```text
reconstruction/learned_recovery/reports/step12_capability.json
```

Required real ALIKED result before continuing:

```text
status == passed
first_features > 0
second_features > 0
raw_matches > 0
```

If ALIKED native model creation/extraction/matching fails:

- record `blocked` with exact exception evidence;
- do not start full extraction;
- do not install or rebuild anything;
- stop Step 12 and report the capability blocker.

Do **not** run LoMa simply to bypass a missing general native/ONNX runtime if the evidence indicates the native learned subsystem itself is unavailable. If the ALIKED failure is clearly model-family-specific while the runtime remains otherwise functional, report it before broadening execution; do not silently redefine the primary path.

---

## Task 5: Implement strict per-frontend learned feature caches test-first

**Files:**
- Modify: `learned_sparse_recovery.py`
- Modify: `tests/test_learned_sparse_recovery.py`
- Reuse: `analysis_common.py`
- Reuse: `sparse_reconstruction.py`

### Step 5.1: Define cache marker contract

Marker must include:

```text
frontend
extractor_type
matcher_type
max_image_size
image_count
feature_count
camera_model
pycolmap_version
device
selection_manifest_sha256
```

The matcher is included in marker provenance to prevent accidental cross-path reuse even though matching has not yet populated the feature-only DB.

### Step 5.2: Write SQLite identity tests

Create small temporary COLMAP-like fixture DBs and assert the validator rejects:

- missing image;
- extra image;
- wrong image name;
- wrong camera ID layout;
- missing keypoint row;
- missing descriptor row;
- mismatched keypoint/descriptor row counts;
- zero total features;
- marker frontend mismatch;
- marker extractor mismatch;
- marker matcher mismatch;
- marker manifest hash mismatch;
- marker max image size mismatch;
- marker pyCOLMAP mismatch.

Also test that SQLite connections close so Windows cleanup can remove/rebuild the DB.

### Step 5.3: Implement generic learned DB layout validation locally

Do not alter Step 11's hardened cache behavior unless a minimal, regression-tested extraction of a generic helper is clearly simpler.

Preferred low-risk approach: implement Step 12's learned cache validator in `learned_sparse_recovery.py` using the same SQL identity fields as Step 11:

```sql
images.image_id
images.name
images.camera_id
keypoints.rows
keypoints.cols
descriptors.rows
descriptors.cols
```

This avoids changing published Step 11 behavior merely for code deduplication.

### Step 5.4: Write failing tests for extraction invocation

Require `extract_learned_features(...)` to call:

```python
pycolmap.extract_features(
    database_path=...,
    image_path=...,
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=build_image_reader_options(SparseRunConfig()),
    extraction_options=build_learned_extraction_options(frontend),
    device=pycolmap.Device.cpu,
)
```

Verify it never writes into the selected image directory and produces exactly 288 image records before marking completion.

### Step 5.5: Implement `extract_learned_features(...)`

Reuse `build_image_reader_options(SparseRunConfig())` for the frozen camera setup.

### Step 5.6: Implement `prepare_learned_feature_cache(...)`

Suggested signature:

```python
def prepare_learned_feature_cache(
    image_dir: Path,
    selection_manifest: Path,
    work_dir: Path,
    frontend: LearnedFrontendSpec,
    sparse_config: SparseRunConfig = SparseRunConfig(),
) -> Path:
    ...
```

Use frontend-specific work subdirectories:

```text
work/aliked/features.db
work/aliked/features_complete.json
work/loma/features.db
work/loma/features_complete.json
```

Marker write is atomic and happens only after exact DB validation.

### Step 5.7: Verify Task 5

```bash
python -m pytest tests/test_learned_sparse_recovery.py -q
```

Expected: PASS.

Do not run full extraction as part of the unit test command.

---

## Task 6: Freeze Step 11 candidate identity for learned diagnostics

**Files:**
- Modify: `learned_sparse_recovery.py`
- Modify: `tests/test_learned_sparse_recovery.py`
- Reuse: `sparse_bridging.py`
- Read: `reconstruction/bridging/reports/step11_candidates.csv`

### Step 6.1: Reuse Step 11 generation and selection logic

Import, do not duplicate:

```python
BridgeSearchConfig
generate_candidate_pairs
write_pair_list
summarize_bridge_pairs
select_bridge_pairs
boundary_bridge_summary
targeted_gate
```

Use a default `BridgeSearchConfig()` unchanged.

### Step 6.2: Write tests for exact 2,340-pair identity

Add a pure helper:

```python
def candidate_identity(candidate: BridgeCandidate) -> tuple[...]:
    ...
```

and a verifier that compares generated Step 12 candidates against Step 11 CSV keys/order.

Test:

```text
2340 total
780 for 73-74
780 for 145-146
780 for 203-204
minimum gap 41
identical keys/order to Step 11 source rows
```

A single changed/missing/extra key must raise an integrity error before learned matching starts.

### Step 6.3: Keep qualification frozen

Do not create Step 12 thresholds. Use the Step 11 `BridgeSearchConfig` values directly:

```text
verified_inliers >= 15
inlier_ratio >= 0.15
```

Selection remains max 8 per boundary and max 2 endpoint reuse.

### Step 6.4: Verify Task 6

```bash
python -m pytest tests/test_learned_sparse_recovery.py tests/test_sparse_bridging.py -q
```

Expected: PASS, proving Step 12 did not alter Step 11 candidate semantics.

---

## Task 7: Implement learned boundary diagnostics with explicit matcher propagation

**Files:**
- Modify: `learned_sparse_recovery.py`
- Modify: `tests/test_learned_sparse_recovery.py`

### Step 7.1: Write the critical failing test

Test `run_learned_diagnostics(...)` with monkeypatched `pycolmap.match_image_pairs` and assert:

```text
matching_options.type == frontend.matcher_type
matching_options.use_gpu is False
device == pycolmap.Device.cpu
pairing_options.match_list_path == exact frontend diagnostic list
```

This test is mandatory because the pyCOLMAP default matcher is SIFT.

### Step 7.2: Implement learned diagnostics

Suggested signature:

```python
def run_learned_diagnostics(
    features_database: Path,
    workspace: Path,
    candidates: Sequence[BridgeCandidate],
    frontend: LearnedFrontendSpec,
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> tuple[BridgePairMetrics, ...]:
    ...
```

Workflow:

1. copy the verified feature DB to `diagnostic.db`;
2. write exact candidate pair list;
3. call `match_image_pairs` once with explicit learned matching options;
4. read raw/verified row counts with existing `summarize_bridge_pairs`;
5. return measured `BridgePairMetrics`.

Do not use `match_exhaustive` anywhere in this module.

### Step 7.3: Add source-level guard test against learned exhaustive matching

A simple targeted test can monkeypatch `pycolmap.match_exhaustive` to raise if called during any learned stage. The orchestration tests later should preserve this invariant.

### Step 7.4: Verify Task 7

```bash
python -m pytest tests/test_learned_sparse_recovery.py tests/test_sparse_bridging.py -q
```

Expected: PASS.

---

## Task 8: Implement targeted learned mapping with explicit learned matcher for BOTH match phases

**Files:**
- Modify: `learned_sparse_recovery.py`
- Modify: `tests/test_learned_sparse_recovery.py`
- Reuse: `sparse_reconstruction.py`
- Reuse: `sparse_bridging.py`

### Step 8.1: Write failing call-order tests

For a gate-passing selected bridge set, require this exact order:

```text
1. learned sequential matching overlap 20
2. learned imported matching for selected bridges
3. sparse incremental mapping
```

The test must inspect both matching calls and assert:

```text
matching_options.type == frontend.matcher_type
matching_options.use_gpu is False
device == pycolmap.Device.cpu
```

For sequential matching additionally assert:

```text
overlap == 20
quadratic_overlap is True
loop_detection is False
```

For imported matching assert the pair list contains only the selected qualified bridges.

### Step 8.2: Implement `run_learned_targeted_attempt(...)`

Suggested signature:

```python
def run_learned_targeted_attempt(
    image_dir: Path,
    features_database: Path,
    output_dir: Path,
    selected_bridges: Sequence[BridgePairMetrics],
    frontend: LearnedFrontendSpec,
    sparse_config: SparseRunConfig = SparseRunConfig(),
    bridge_config: BridgeSearchConfig = BridgeSearchConfig(),
) -> AttemptMetrics:
    ...
```

Use frontend-specific work DB names to prevent collision:

```text
work/aliked/targeted.db
work/aliked/targeted_bridge_pairs.txt
work/loma/targeted.db
work/loma/targeted_bridge_pairs.txt
```

Call existing `map_sparse_database(...)`; do not fork mapper logic.

Attempt names must identify the frontend, e.g.:

```text
aliked_targeted
loma_targeted
```

### Step 8.3: Reuse acceptance semantics

Add a Step 12-named wrapper only if useful for clarity:

```python
def learned_model_metric_accepted(model: ModelMetrics) -> bool:
    return bridge_model_accepted(model, BridgeSearchConfig())
```

This preserves:

```text
>=274 registered
>=1000 points
one SIMPLE_RADIAL camera
finite reprojection error
```

Visual plausibility remains a later finalization condition and is not invented inside this pure metric helper.

### Step 8.4: Verify Task 8

```bash
python -m pytest tests/test_learned_sparse_recovery.py -q
```

Expected: PASS.

---

## Task 9: Build staged ALIKED-first runner logic test-first

**Files:**
- Modify: `run_learned_sparse_recovery.py`
- Modify: `tests/test_run_learned_sparse_recovery.py`

### Step 9.1: Define runner stages

Use these explicit stages:

```text
capability
aliked-diagnose
aliked-map
loma-diagnose
loma-map
finalize
all
```

This keeps expensive phases resumable and makes the fallback boundary inspectable.

### Step 9.2: Implement ALIKED diagnostic stage tests

Test that `aliked-diagnose`:

1. requires passed ALIKED capability evidence;
2. verifies selected images/manifest;
3. prepares only the ALIKED feature cache;
4. regenerates the Step 11 candidates;
5. validates exact candidate identity against Step 11 CSV;
6. runs ALIKED learned diagnostics;
7. selects bridges with Step 11 rules;
8. writes ALIKED candidate CSV and boundary summary;
9. performs no mapping.

Report names:

```text
reports/step12_aliked_candidates.csv
reports/step12_aliked_boundary_summary.json
```

### Step 9.3: Implement ALIKED mapping stage tests

Test two paths.

**Boundary gate failure:**

```text
status = skipped
reason identifies missing boundary
attempt = null
mapping function not called
```

Write:

```text
reports/step12_aliked_attempt.json
```

**Boundary gate pass:** verify ALIKED feature cache, selected bridges, learned targeted attempt, and durable attempt report.

### Step 9.4: Implement ALIKED stages

Keep report state explicit and reloadable. Do not infer completion solely from directory existence.

### Step 9.5: Verify Task 9

```bash
python -m pytest tests/test_run_learned_sparse_recovery.py -q
```

Expected: PASS.

---

## Task 10: Add the single LoMa fallback gate test-first

**Files:**
- Modify: `run_learned_sparse_recovery.py`
- Modify: `tests/test_run_learned_sparse_recovery.py`
- Modify: `learned_sparse_recovery.py` only if a reusable helper is required

### Step 10.1: Define `loma_required(...)` as pure logic

LoMa is required exactly when ALIKED:

- diagnostic boundary gate failed, or
- mapping completed but strongest model failed metric acceptance.

LoMa is **not** required when ALIKED strongest model passes metric acceptance.

Write table-driven tests for all statuses.

### Step 10.2: Implement lazy LoMa smoke

`loma-diagnose` must first inspect the capability report:

- static LoMa option evidence must be present;
- if LoMa smoke is `not_run`, execute the one-pair LoMa smoke and update capability JSON atomically;
- if the smoke is `blocked`, write a blocked LoMa attempt status and do not extract 288 LoMa features;
- if passed, proceed.

### Step 10.3: Mirror diagnostic/mapping stages without duplicating orchestration logic

Prefer parameterized internal helpers such as:

```python
_diagnose_frontend(frontend, paths)
_map_frontend(frontend, paths)
```

Public stage dispatch remains explicit.

Do not make a generic loop over every enum or model variant. Only the two frozen specs are allowed.

### Step 10.4: Write orchestration tests proving bounded fallback

Required tests:

1. ALIKED accepted -> LoMa smoke/extraction/diagnostics/mapping never called.
2. ALIKED boundary gate fails -> LoMa diagnose is called exactly once.
3. ALIKED maps but fails acceptance -> LoMa diagnose/map path is allowed exactly once.
4. LoMa boundary gate fails -> LoMa mapping is skipped and workflow finalizes failure.
5. LoMa maps but fails acceptance -> final failure.
6. LoMa maps and passes -> final selected frontend is LoMa.
7. `match_exhaustive` is never called in any path.

### Step 10.5: Verify Task 10

```bash
python -m pytest tests/test_run_learned_sparse_recovery.py tests/test_learned_sparse_recovery.py -q
```

Expected: PASS.

---

## Task 11: Implement final selection, reports, and truthful status semantics

**Files:**
- Modify: `run_learned_sparse_recovery.py`
- Modify: `tests/test_run_learned_sparse_recovery.py`
- Runtime outputs under: `reconstruction/learned_recovery/reports/`

### Step 11.1: Implement stage-report loader/serializer

Use portable project-relative model paths, JSON-safe finite handling, and explicit statuses:

```text
completed
skipped
blocked
not_run
```

Do not fabricate `AttemptMetrics` for non-completed paths.

### Step 11.2: Build `step12_attempts.csv`

Always write one logical row for ALIKED and one for LoMa.

Populate model metrics only when that frontend actually mapped.

Columns should include at minimum:

```text
frontend
status
reason
boundary_gate_allowed
model_count
registered_images
registration_fraction
sparse_points
observations
mean_track_length
mean_reprojection_error
camera_count
camera_model
runtime_seconds
feature_count
matched_pair_count
verified_pair_count
metric_acceptance_met
```

Do not use model-union registration as `registered_images`.

### Step 11.3: Define attempt ranking when both mapped

Ranking is:

1. metric-accepted model over nonaccepted model;
2. more registered images;
3. more sparse points;
4. lower finite reprojection error.

Normally ALIKED acceptance stops before LoMa, so both mapped only when ALIKED failed metric acceptance and LoMa subsequently mapped.

### Step 11.4: Copy only selected sparse model to `best/`

Use existing `copy_sparse_model(...)`.

Reopen and resummarize the copied `best/` model and assert it matches the report before finalization.

Export PLY only from real selected `best/` sparse model.

### Step 11.5: Build `step12_registered_images.csv`

Exactly 288 rows ordered by selected sequence index, with:

```text
selected_index
filename
registered
selected_frontend
```

Only the final selected single model determines `registered`.

### Step 11.6: Build `step12_summary.json`

Required fields include:

```text
pycolmap_version
selection_manifest_sha256
input_image_count
learned_feature_max_image_size
critical_boundaries
candidate_count
qualification_thresholds
aliked_result
loma_result
selected_frontend
selected_attempt
best_model
registered_images
registered_indices
unregistered_images
unregistered_indices
metric_acceptance_met
visual_plausibility_status
learned_recovery_success
dense_reconstruction_started
next_boundary
```

At the end of automated mapping, initialize:

```text
visual_plausibility_status = pending
learned_recovery_success = false
```

if metric acceptance passes but figures have not yet been reviewed.

Only the later visual review step may change visual status to `passed` or `failed` and derive final `learned_recovery_success`.

`dense_reconstruction_started` is always `false` in Step 12.

### Step 11.7: Test no-model failure finalization

If neither frontend maps, finalization must still produce truthful capability/attempt/summary evidence but must not create fake `best/`, registered CSV, PLY, or sparse figure.

### Step 11.8: Verify Task 11

```bash
python -m pytest tests/test_run_learned_sparse_recovery.py -q
```

Expected: PASS.

---

## Task 12: Implement all four real-data figures

**Files:**
- Modify: `run_learned_sparse_recovery.py`
- Modify: `tests/test_run_learned_sparse_recovery.py`
- Runtime outputs: `reconstruction/learned_recovery/previews/`

### Step 12.1: Boundary comparison integrity first

Load:

```text
reconstruction/bridging/reports/step11_candidates.csv
step12_aliked_candidates.csv if diagnostics ran
step12_loma_candidates.csv if diagnostics ran
```

Before plotting, assert all present learned candidate tables have the exact same 2,340 pair keys in the same deterministic order as Step 11.

### Step 12.2: Render `step12_01_boundary_comparison.png`

Use verified inliers for the same candidates, grouped by boundary and frontend.

The figure must make the Step 11 zero-SIFT result for `73-74` and `145-146` visible rather than hiding it through autoscaling. All verified-inlier axes/bounds must remain nonnegative.

Do not imply LoMa results if LoMa did not run.

### Step 12.3: Render `step12_02_sparse_model.png`

Reuse the Step 11 approach to load real `points3D` and camera projection centers from the selected `best/` model.

The title must report selected frontend, registered count, sparse points, and reprojection error.

### Step 12.4: Render `step12_03_registration.png`

Plot binary registered/unregistered state over indices `1..288` and mark the three critical boundaries.

### Step 12.5: Render `step12_04_frontend_comparison.png`

Load authoritative strongest single-model evidence only:

```text
Step 10 SIFT baseline: reconstruction/reports/step10_summary.json
Step 11 exhaustive SIFT: reconstruction/bridging/reports/step11_exhaustive.json
Step 12 ALIKED: if mapped
Step 12 LoMa: if mapped
```

Compare at minimum:

```text
registered images
sparse points
mean reprojection error
```

Do not plot Step 10/11 disconnected union coverage as registration.

### Step 12.6: Add figure-generation tests

Use small synthetic report fixtures, not fake project measurements, to test:

- candidate key mismatch rejects plotting;
- zero-inlier SIFT boundaries use nonnegative scale;
- not-run LoMa is omitted;
- only strongest model fields feed frontend registration chart;
- no-model workflow omits model-only figures cleanly.

### Step 12.7: Verify Task 12

```bash
python -m pytest tests/test_run_learned_sparse_recovery.py -q
```

Expected: PASS.

---

## Task 13: Implement safe transient cleanup

**Files:**
- Modify: `run_learned_sparse_recovery.py`
- Modify: `tests/test_run_learned_sparse_recovery.py`

### Step 13.1: Enumerate exact transient filenames

Do not recursively delete `work/`.

Allow only known files such as:

```text
work/aliked/features.db
work/aliked/features_complete.json
work/aliked/diagnostic.db
work/aliked/diagnostic_pairs.txt
work/aliked/targeted.db
work/aliked/targeted_bridge_pairs.txt
work/loma/features.db
work/loma/features_complete.json
work/loma/diagnostic.db
work/loma/diagnostic_pairs.txt
work/loma/targeted.db
work/loma/targeted_bridge_pairs.txt
```

Only remove files explicitly named by the cleanup function after final evidence is durable.

### Step 13.2: Test cleanup boundaries

Test that:

- enumerated regular files are removed;
- unexpected files remain untouched;
- symlinks are rejected;
- path escapes are rejected;
- nonempty directories are not forcibly removed.

### Step 13.3: Verify Task 13

```bash
python -m pytest tests/test_run_learned_sparse_recovery.py -q
```

Expected: PASS.

---

## Task 14: Run deterministic test/compile verification before expensive real execution

**Files:** none beyond implemented source/tests.

### Step 14.1: Run Step 12 focused tests

```bash
python -m pytest tests/test_learned_sparse_recovery.py tests/test_run_learned_sparse_recovery.py -q
```

Expected: all pass.

### Step 14.2: Run Step 10/11 regression tests

```bash
python -m pytest \
  tests/test_sparse_reconstruction.py \
  tests/test_run_sparse_reconstruction.py \
  tests/test_sparse_bridging.py \
  tests/test_run_sparse_bridging.py \
  -q
```

Expected: all pass.

### Step 14.3: Compile touched Python modules

```bash
python -m py_compile \
  learned_sparse_recovery.py \
  run_learned_sparse_recovery.py \
  tests/test_learned_sparse_recovery.py \
  tests/test_run_learned_sparse_recovery.py
```

Expected: exit 0.

Do not claim the real learned pipeline works yet. At this point only deterministic code paths are verified.

---

## Task 15: Execute ALIKED primary path on real project data

**Files/outputs:**
- `reconstruction/learned_recovery/work/aliked/`
- `reconstruction/learned_recovery/aliked/`
- ALIKED reports under `reconstruction/learned_recovery/reports/`

### Step 15.1: Recheck capability evidence

If Task 4 already produced a passed ALIKED smoke and the capability report still matches current pyCOLMAP/version/config, reuse it. Otherwise rerun:

```bash
python run_learned_sparse_recovery.py --stage capability
```

### Step 15.2: Run ALIKED diagnostics

```bash
python run_learned_sparse_recovery.py --stage aliked-diagnose
```

This is the first full learned extraction over all 288 images.

After completion verify:

- cache marker matches ALIKED config;
- exact 288 image names/camera rows/features validated;
- 2,340 candidate CSV rows;
- 780 rows per boundary;
- candidate keys exactly match Step 11;
- boundary summary truthfully reports verified/qualified/selected counts.

### Step 15.3: Apply ALIKED boundary gate

If any boundary has zero selected qualified bridges:

- write ALIKED mapping status `skipped`;
- do not map ALIKED;
- proceed to Task 16 LoMa fallback.

If all three pass, continue.

### Step 15.4: Run ALIKED targeted map

```bash
python run_learned_sparse_recovery.py --stage aliked-map
```

Verify the real match database used ALIKED-LightGlue for both sequential and imported matching through report/config evidence and tests; do not rely on function defaults.

Inspect strongest single model metrics.

### Step 15.5: Apply metric gate

If the strongest ALIKED single model satisfies:

```text
registered >= 274
points >= 1000
one SIMPLE_RADIAL camera
finite reprojection error
```

skip LoMa and continue to finalization/visual review.

Otherwise proceed to Task 16.

---

## Task 16: Execute the single LoMa fallback only if required

**Files/outputs:**
- `reconstruction/learned_recovery/work/loma/`
- `reconstruction/learned_recovery/loma/`
- LoMa reports under `reconstruction/learned_recovery/reports/`

### Step 16.1: Confirm fallback is justified

The runner must be able to state one of exactly these reasons:

```text
ALIKED boundary gate failed
ALIKED targeted model failed global metric acceptance
```

If ALIKED passed metric acceptance, do not run LoMa.

### Step 16.2: Run LoMa lazy smoke and diagnostics

```bash
python run_learned_sparse_recovery.py --stage loma-diagnose
```

The stage must first run/update the LoMa smoke if it was previously `not_run`.

If LoMa smoke is blocked, stop LoMa immediately and preserve blocker evidence.

If passed, perform exactly the same 2,340-pair diagnostic experiment.

### Step 16.3: Apply LoMa boundary gate

If any boundary lacks a selected qualified bridge:

- write LoMa map status `skipped`;
- do not map;
- proceed to failure finalization.

### Step 16.4: Run LoMa targeted map only if allowed

```bash
python run_learned_sparse_recovery.py --stage loma-map
```

Use overlap 20 + selected bridges with `LOMA_L` matching options explicitly supplied to both matching APIs.

### Step 16.5: Apply metric gate

No further matcher/front-end fallback is allowed regardless of result.

---

## Task 17: Finalize reports and visual evidence

**Files/outputs:**
- `reconstruction/learned_recovery/best/`
- `reconstruction/learned_recovery/reports/step12_attempts.csv`
- `reconstruction/learned_recovery/reports/step12_registered_images.csv` when a model exists
- `reconstruction/learned_recovery/reports/step12_summary.json`
- `reconstruction/learned_recovery/previews/*.png`

### Step 17.1: Run automated finalization

```bash
python run_learned_sparse_recovery.py --stage finalize
```

The command must:

1. load only durable stage reports;
2. select the strongest valid mapped attempt;
3. reopen and verify its source model;
4. copy sparse model to `best/`;
5. reopen and verify copied model;
6. write attempts/registration reports;
7. generate available figures;
8. write summary with visual status `pending` when metric acceptance is met but not yet reviewed;
9. keep `dense_reconstruction_started=false`.

### Step 17.2: Inspect actual figures

Use an image viewer/tool to inspect:

```text
step12_01_boundary_comparison.png
step12_02_sparse_model.png (if model exists)
step12_03_registration.png (if model exists)
step12_04_frontend_comparison.png
```

Visual review criteria:

- camera centers form a coherent capture trajectory rather than an obvious collapsed/exploded configuration;
- sparse points plausibly occupy one vessel/object scene rather than several unrelated clouds;
- registration plot corresponds to reported 1..288 indices;
- boundary comparison does not misstate zero-inlier SIFT boundaries;
- frontend comparison uses strongest single models only.

### Step 17.3: Record visual review outcome deterministically

Implement a small finalization input/flag rather than hand-editing JSON. For example:

```bash
python run_learned_sparse_recovery.py --stage finalize --visual-status passed
```

Allowed values:

```text
pending
passed
failed
```

The runner must reject `passed` if metric acceptance is false.

Derive:

```text
learned_recovery_success = metric_acceptance_met and visual_status == "passed"
```

### Step 17.4: Stop at Step 12 boundary

Even on success:

```text
dense_reconstruction_started = false
```

Do not call dense reconstruction.

---

## Task 18: Write measured Step 12 documentation after execution

**Files:**
- Create: `docs/geometry-ml/learned-sparse-recovery.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/memory-bank/active-context.md`
- Modify: `docs/memory-bank/progress.md`

### Step 18.1: Create measured result document

`docs/geometry-ml/learned-sparse-recovery.md` must report only real measurements from the generated Step 12 artifacts.

Include:

- rationale after Step 11 SIFT failure;
- capability evidence;
- ALIKED boundary counts and mapping result;
- LoMa result only if fallback actually ran;
- strongest single-model metrics;
- acceptance status;
- figure references;
- explicit dense-not-started statement;
- next decision boundary if failure.

Do not claim LoMa was tested if it was not run.

### Step 18.2: Update project overview/status docs

Update README/AGENTS/memory bank with the actual Step 12 outcome and next boundary.

If Step 12 fails, next choices are:

```text
accept local-only reconstruction
or separately authorize experimental learned global matching/component alignment
```

Do not add recapture as the normal recommendation.

### Step 18.3: Update changelog

Record implementation and measured result concisely. Do not claim dense reconstruction.

---

## Task 19: Full verification and defect-focused review

### Step 19.1: Run full project tests

```bash
python -m pytest -q
```

Expected: all relevant project tests pass.

### Step 19.2: Compile all touched Python

```bash
python -m py_compile \
  learned_sparse_recovery.py \
  run_learned_sparse_recovery.py \
  tests/test_learned_sparse_recovery.py \
  tests/test_run_learned_sparse_recovery.py
```

Expected: exit 0.

### Step 19.3: Reverify protected selected data

Use existing verifier and confirm again:

```text
288/288 selected images unchanged
selection manifest SHA-256 unchanged
```

Where the project has a raw-image integrity verifier, also confirm the 297 raw images remain unchanged.

### Step 19.4: Reopen baseline and Step 12 model evidence

Reopen:

```text
reconstruction/sparse/best/
reconstruction/bridging/best/
reconstruction/learned_recovery/best/ (if model exists)
```

Confirm Step 10/11 baseline files were not changed and Step 12 summary agrees with the reopened Step 12 model.

### Step 19.5: Search for forbidden dense execution

Inspect Step 12 source and outputs for dense API usage/artifacts. There must be none.

### Step 19.6: Inspect final Git diff/status

Run:

```bash
git status --short --branch
git diff --check
git diff --stat
git diff -- learned_sparse_recovery.py run_learned_sparse_recovery.py tests docs README.md AGENTS.md CHANGELOG.md
```

Confirm:

- only intended Step 12 code/docs/evidence changed;
- `.codegraph/` remains local;
- CNN checkpoint remains local/untracked;
- no model cache or transient DB is staged/tracked;
- no selected/raw JPEG changed;
- no DOCX/PDF changed.

### Step 19.7: Do not overstate verification

Report separately:

- unit/regression test results;
- compile result;
- real capability smoke result;
- real ALIKED/LoMa execution status;
- actual strongest single-model metrics;
- visual review result;
- protected-data verification;
- skipped/not-run frontend state;
- residual limitation if global acceptance failed.

### Step 19.8: Git publication boundary

Do not commit or push unless the user's implementation instruction explicitly authorizes those actions.

If authorized, inspect status and intended diff first, stage only intended Step 12 files/evidence, explicitly exclude `.codegraph/`, local checkpoints, native model caches, transient databases, and unrelated changes, then commit/push and verify remote state.

---

## Required Implementation File Set

New source/tests/docs:

```text
learned_sparse_recovery.py
run_learned_sparse_recovery.py
tests/test_learned_sparse_recovery.py
tests/test_run_learned_sparse_recovery.py
docs/geometry-ml/learned-sparse-recovery.md
```

Already-created planning artifacts:

```text
docs/superpowers/specs/2026-09-05-step-12-learned-sparse-recovery-design.md
docs/superpowers/plans/2026-09-05-step-12-learned-sparse-recovery.md
```

Expected measured output root:

```text
reconstruction/learned_recovery/
```

Existing implementation modules should be reused rather than modified unless a proven defect/compatibility need requires a minimal regression-tested change:

```text
analysis_common.py
sparse_reconstruction.py
sparse_bridging.py
```

## Acceptance Checklist

Step 12 implementation is complete only when all applicable items are true. This checklist is retained as the original execution gate: downstream learned-extraction/matching/mapping items remain unchecked because the approved ALIKED capability blocker stopped execution before those stages, not because they are unfinished obligations within the closed Step 12 scope. The header and measured report record the terminal blocker and completed verification.

- [ ] Live state/instructions recovered before edits.
- [ ] Protected 288-image input verified unchanged.
- [ ] Native pyCOLMAP learned types/options recorded.
- [ ] ALIKED one-pair native inference smoke passed, or capability blocker was truthfully recorded and execution stopped.
- [ ] ALIKED full feature cache used strict identity/provenance validation.
- [ ] Exactly the Step 11 2,340 candidate keys were reused.
- [ ] ALIKED-LightGlue was explicitly supplied to both imported and sequential matching APIs.
- [ ] Frozen qualification/selection rules were used.
- [ ] Mapping ran only after all three boundary gates passed.
- [ ] LoMa ran only if the ALIKED fallback condition was met.
- [ ] LoMa native smoke ran before LoMa full extraction.
- [ ] LoMa-L was explicitly supplied to both matching APIs if LoMa mapped.
- [ ] No exhaustive learned matching ran.
- [ ] No third matcher/frontend ran.
- [ ] Strongest single-model metrics, not union coverage, drive acceptance.
- [ ] `registered_images >= 274` for success.
- [ ] `sparse_points >= 1000` for success.
- [ ] Exactly one `SIMPLE_RADIAL` camera for success.
- [ ] Reprojection error finite for success.
- [ ] Camera trajectory and sparse geometry visually reviewed for success.
- [ ] `learned_recovery_success` reflects both metric and visual gates.
- [ ] `dense_reconstruction_started == false` always.
- [ ] Step 10/11 evidence was not overwritten.
- [ ] No raw/selected JPEG was modified.
- [ ] No CNN/native learned model weight cache was published.
- [ ] Focused Step 12 tests pass.
- [ ] Step 10/11 regression tests pass.
- [ ] Full project tests pass or any unrelated pre-existing failure is clearly separated.
- [ ] Touched Python compiles.
- [ ] Generated reports/figures were reopened/inspected where applicable.
- [ ] Final diff contains only intended Step 12 work.
- [ ] No commit/push occurred unless explicitly authorized.

## Final Failure Boundary

If both allowed learned frontend paths fail the global sparse acceptance goal, stop Step 12 with:

```text
learned_recovery_success = false
dense_reconstruction_started = false
```

Do not add another matcher automatically.

The next separately authorized decision is either:

```text
accept local-only sparse reconstruction
```

or:

```text
experimental global learned matching / component alignment phase
```

Possible future tools such as LoFTR are out of scope for this plan.
