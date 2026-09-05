# Step 11 Sparse Component Bridging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attempt to connect the disconnected Step 10 pyCOLMAP sparse components into one coherent sparse reconstruction by adding measured non-local cross-boundary matches, with one exhaustive SIFT fallback only if targeted bridging is insufficient.

**Architecture:** Reuse the Step 10 camera, feature, mapper, metrics, and integrity contracts. Add a focused `sparse_bridging.py` module for non-local pair generation, pair-metric analysis, targeted/exhaustive matching, and attempt selection, plus `run_sparse_bridging.py` for bounded orchestration and evidence generation. Preserve Step 10 outputs unchanged and stop before dense reconstruction regardless of Step 11 outcome.

**Tech Stack:** Python 3.14, pyCOLMAP 4.2.x, SQLite, NumPy, matplotlib, pytest, existing `analysis_common.py`, `run_preprocessing.py`, and `sparse_reconstruction.py` helpers.

**Spec:** `docs/superpowers/specs/2026-09-05-step-11-sparse-component-bridging-design.md`

**Execution result (2026-09-05):** Completed on all 288 verified inputs. Diagnosis evaluated exactly 2,340 non-local pairs; boundaries 73-74 and 145-146 had zero geometrically verified candidates, while 203-204 produced qualified bridges. The targeted mapper was therefore skipped by design. The single exhaustive fallback produced eight disconnected models with 224-image union coverage, but its strongest single model remained 73/288 images with 3,443 points, so `bridge_success=false` and dense reconstruction remains blocked. Post-implementation review hardened cache/resume identity validation and corrected the zero-inlier candidate-plot scale without changing the measured SfM result.

## Global Constraints

- Canonical Git root: `C:\Assumption University\CSX4213\Project`.
- Expected starting commit before Step 11 implementation: `ef8574e396de3fa94eac72df24e95c2f21946654` unless newer authorized work has landed; always inspect live `HEAD` first.
- Preserve unrelated local work. The local `analysis/ml/checkpoints/best_small_seg_cnn.pt` remains untracked/unpublished.
- Never modify `IMG20260826122949/` or `preprocessing/pycolmap_input/images/`.
- Reverify the selected manifest before every real Step 11 execution stage that consumes the 288 images.
- Preserve Step 10 outputs under `reconstruction/sparse/`; Step 11 writes only under `reconstruction/bridging/` plus Step 11 source/tests/docs.
- Keep `SparseRunConfig`: one shared `SIMPLE_RADIAL` camera, SIFT `max_image_size=1200`, `max_num_features=8192`, minimum matches 15, minimum model size 10, seed 4213.
- Use native unmasked pyCOLMAP SIFT only.
- Primary boundaries: `(73, 74)`, `(145, 146)`, `(203, 204)`.
- Targeted diagnostic window: 40 frames on each side; exclude candidate pairs with sequence gap <= 40.
- Candidate qualification: verified inliers >= 15 and inlier ratio >= 0.15.
- Select at most 8 bridge pairs per boundary; any endpoint filename may appear at most twice per boundary.
- Targeted mapping runs only if every primary boundary has at least one selected qualified bridge.
- If targeted mapping fails the fixed acceptance gate, allow exactly one exhaustive SIFT fallback.
- Step 11 acceptance: best model registers >=274/288 images, has >=1000 sparse points, one `SIMPLE_RADIAL` camera, finite mean reprojection error, and visually plausible camera/point geometry.
- Do not add camera-model sweeps, learned features/matchers, CNN masks, vocabulary trees, image deletion, guessed model transforms, dense MVS, mesh, texture, or Blender.
- No commit or push unless the implementation task explicitly authorizes it.

---

### Task 1: Expose reusable Step 10 pyCOLMAP runtime helpers without changing Step 10 behavior

**Files:**
- Modify: `sparse_reconstruction.py`
- Modify: `tests/test_sparse_reconstruction.py`

**Interfaces:**
- Produces: `build_image_reader_options(config: SparseRunConfig) -> pycolmap.ImageReaderOptions`
- Produces: `build_feature_extraction_options(config: SparseRunConfig) -> pycolmap.FeatureExtractionOptions`
- Produces: `build_incremental_pipeline_options(config: SparseRunConfig) -> pycolmap.IncrementalPipelineOptions`
- Produces: `extract_sparse_features(image_dir: Path, database_path: Path, config: SparseRunConfig) -> DatabaseMetrics`
- Produces: `map_sparse_database(database_path: Path, image_dir: Path, output_dir: Path, config: SparseRunConfig) -> tuple[ModelMetrics, ...]`
- Consumes: existing `SparseRunConfig`, `DatabaseMetrics`, `ModelMetrics`, `summarize_database`, `summarize_reconstruction`

- [ ] **Step 1: Add focused tests for the public option builders**

Add tests that assert the public helpers preserve the exact Step 10 configuration:

```python
def test_build_feature_extraction_options_preserves_step10_settings():
    options = build_feature_extraction_options(SparseRunConfig())
    assert options.max_image_size == 1200
    assert options.sift.max_num_features == 8192
    assert options.use_gpu is False


def test_build_incremental_pipeline_options_preserves_step10_settings():
    options = build_incremental_pipeline_options(SparseRunConfig())
    assert options.min_num_matches == 15
    assert options.multiple_models is True
    assert options.min_model_size == 10
    assert options.random_seed == 4213
    assert options.ba_refine_focal_length is True
    assert options.ba_refine_principal_point is False
    assert options.ba_refine_extra_params is True
```

Also assert `build_image_reader_options` returns `SIMPLE_RADIAL` and the expected comma-separated four-parameter camera initialization derived from 26 mm 35-mm equivalent.

- [ ] **Step 2: Run the focused test file and confirm the new tests fail before implementation**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_reconstruction.py
```

Expected: failures because the new public helpers do not yet exist.

- [ ] **Step 3: Refactor the existing private option construction into the public helpers**

Implement:

```python
def build_image_reader_options(config: SparseRunConfig) -> pycolmap.ImageReaderOptions:
    params = simple_radial_camera_params(
        config.image_width,
        config.image_height,
        config.focal_35mm,
    )
    return pycolmap.ImageReaderOptions(
        camera_model=config.camera_model,
        camera_params=",".join(f"{value:.12g}" for value in params),
    )


def build_feature_extraction_options(
    config: SparseRunConfig,
) -> pycolmap.FeatureExtractionOptions:
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = config.max_image_size
    options.sift.max_num_features = config.max_num_features
    options.use_gpu = False
    return options


def build_incremental_pipeline_options(
    config: SparseRunConfig,
) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = config.minimum_matches
    options.multiple_models = True
    options.min_model_size = config.min_model_size
    options.random_seed = config.random_seed
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = True
    options.mapper.random_seed = config.random_seed
    options.triangulation.random_seed = config.random_seed
    return options
```

Make the existing `run_sparse_attempt` call these helpers instead of maintaining duplicate option construction.

- [ ] **Step 4: Add reusable feature-extraction and mapping wrappers**

`extract_sparse_features` must:

1. require an existing image directory;
2. require an empty/non-existent destination database path;
3. call `pycolmap.extract_features` with `CameraMode.SINGLE`, the public reader/options helpers, and `Device.cpu`;
4. return `summarize_database(database_path)`;
5. fail if the resulting database does not contain exactly `config.expected_images` images.

`map_sparse_database` must:

1. require an existing database and image directory;
2. require an empty/non-existent mapping output directory;
3. call `pycolmap.incremental_mapping` with the public pipeline helper;
4. fail when no reconstruction is returned;
5. write/reopen every model under numeric subdirectories;
6. return all `ModelMetrics`, sorted by numeric model id.

Refactor `run_sparse_attempt` to use both wrappers. Its current external behavior and reports must remain unchanged.

- [ ] **Step 5: Run Step 10-focused regression tests**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_reconstruction.py tests/test_run_sparse_reconstruction.py
```

Require PASS before starting Step 11 code.

---

### Task 2: Implement deterministic non-local bridge candidate generation and selection

**Files:**
- Create: `sparse_bridging.py`
- Create: `tests/test_sparse_bridging.py`

**Interfaces:**
- Produces: `BridgeSearchConfig`
- Produces: `BridgeCandidate`
- Produces: `BridgePairMetrics`
- Produces: `generate_candidate_pairs(records: Sequence[SelectedImageRecord], config: BridgeSearchConfig) -> tuple[BridgeCandidate, ...]`
- Produces: `write_pair_list(path: Path, pairs: Sequence[BridgeCandidate | BridgePairMetrics]) -> None`
- Produces: `qualified_bridge(pair: BridgePairMetrics, config: BridgeSearchConfig) -> bool`
- Produces: `select_bridge_pairs(metrics: Sequence[BridgePairMetrics], config: BridgeSearchConfig) -> tuple[BridgePairMetrics, ...]`
- Consumes: `analysis_common.SelectedImageRecord`

- [ ] **Step 1: Write candidate-generation tests first**

Freeze the configuration:

```python
@dataclass(frozen=True)
class BridgeSearchConfig:
    boundaries: tuple[tuple[int, int], ...] = ((73, 74), (145, 146), (203, 204))
    window_size: int = 40
    minimum_sequence_gap: int = 41
    minimum_verified_inliers: int = 15
    minimum_inlier_ratio: float = 0.15
    max_pairs_per_boundary: int = 8
    max_endpoint_reuse: int = 2
    targeted_sequential_overlap: int = 20
    exhaustive_block_size: int = 50
    minimum_registered_images: int = 274
    minimum_sparse_points: int = 1000
```

Tests must assert:

```python
def test_candidate_generation_excludes_pairs_already_inside_overlap40(records):
    pairs = generate_candidate_pairs(records, BridgeSearchConfig())
    assert pairs
    assert all(abs(pair.right_index - pair.left_index) >= 41 for pair in pairs)


def test_candidate_generation_is_bounded_and_deterministic(records):
    pairs = generate_candidate_pairs(records, BridgeSearchConfig())
    assert len(pairs) <= 2340
    assert pairs == tuple(sorted(pairs, key=lambda pair: (
        pair.boundary_left,
        pair.left_index,
        pair.right_index,
        pair.left_filename,
        pair.right_filename,
    )))
```

Also test that every candidate crosses exactly one configured boundary and no filename refers outside the selected manifest.

- [ ] **Step 2: Run the new test file red**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py
```

Expected: import failure because `sparse_bridging.py` does not exist.

- [ ] **Step 3: Implement the data contracts and candidate generator**

Use explicit frozen dataclasses:

```python
@dataclass(frozen=True)
class BridgeCandidate:
    boundary_left: int
    boundary_right: int
    left_index: int
    right_index: int
    left_filename: str
    right_filename: str

    @property
    def sequence_gap(self) -> int:
        return self.right_index - self.left_index


@dataclass(frozen=True)
class BridgePairMetrics:
    candidate: BridgeCandidate
    raw_matches: int
    verified_inliers: int
    inlier_ratio: float
    qualified: bool
```

For each boundary, generate the left/right window cross-product, then retain only pairs whose positive sequence gap is at least 41. Deduplicate by filename pair and return deterministic ordering.

- [ ] **Step 4: Implement exact imported pair-list writing**

Write one pair per line:

```text
IMG20260826122949.jpg IMG20260826122953.jpg
```

Requirements:

- no blank lines;
- no duplicate lines;
- newline at EOF;
- filenames only, no absolute paths;
- reject filenames containing whitespace because imported-pair parsing would be ambiguous.

Add tests for deterministic content and duplicate rejection.

- [ ] **Step 5: Implement bridge qualification and diversity-aware selection**

Qualification is fixed:

```python
pair.verified_inliers >= 15 and pair.inlier_ratio >= 0.15
```

Selection per boundary:

1. filter qualified pairs;
2. sort by `(-verified_inliers, -inlier_ratio, sequence_gap, left_filename, right_filename)`;
3. greedily accept while neither endpoint would exceed `max_endpoint_reuse=2`;
4. stop at `max_pairs_per_boundary=8`.

Add a test where the highest-scoring three candidates share one endpoint and verify that the third is skipped once endpoint reuse reaches two.

- [ ] **Step 6: Run the bridge-unit tests green**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py
```

---

### Task 3: Add COLMAP pair-metric extraction and diagnostic matching

**Files:**
- Modify: `sparse_bridging.py`
- Modify: `tests/test_sparse_bridging.py`

**Interfaces:**
- Produces: `summarize_bridge_pairs(database_path: Path, candidates: Sequence[BridgeCandidate], config: BridgeSearchConfig) -> tuple[BridgePairMetrics, ...]`
- Produces: `run_bridge_diagnostics(features_database: Path, workspace: Path, candidates: Sequence[BridgeCandidate], config: BridgeSearchConfig) -> tuple[BridgePairMetrics, ...]`
- Consumes: `pycolmap.image_pair_to_pair_id`, `pycolmap.match_image_pairs`

- [ ] **Step 1: Write a tiny SQLite pair-metric fixture test**

Create a temporary COLMAP-like database containing:

```sql
CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
```

Insert two images, raw matches = 100, verified inliers = 25. Compute the pair id with:

```python
pair_id = pycolmap.image_pair_to_pair_id(1, 2)
```

Assert `summarize_bridge_pairs` returns:

```text
raw_matches = 100
verified_inliers = 25
inlier_ratio = 0.25
qualified = true
```

Also cover missing match rows and zero raw matches; those must produce zero counts, ratio 0.0, and `qualified=false`.

- [ ] **Step 2: Run the new tests red**

Run the same focused test file and verify failures are limited to the new functions.

- [ ] **Step 3: Implement read-only pair-metric summarization**

Map image names to image ids through the `images` table. For every candidate:

1. obtain both image ids;
2. use `pycolmap.image_pair_to_pair_id`;
3. query `matches.rows` and `two_view_geometries.rows`;
4. compute ratio only when raw matches > 0;
5. run `qualified_bridge` using the frozen config.

Do not mutate the database while summarizing.

- [ ] **Step 4: Implement the diagnostic database workflow**

`run_bridge_diagnostics` must:

1. require a valid features-only source database;
2. create `workspace/diagnostic.db` by `shutil.copy2`;
3. create `workspace/diagnostic_pairs.txt` through `write_pair_list`;
4. set `ImportedPairingOptions.match_list_path` to the pair file;
5. use `FeatureMatchingOptions()` with `use_gpu=False`;
6. call `pycolmap.match_image_pairs` exactly once;
7. return `summarize_bridge_pairs(diagnostic_database, candidates, config)`.

It must not run incremental mapping.

- [ ] **Step 5: Add boundary summary helper**

Add:

```python
def boundary_bridge_summary(
    metrics: Sequence[BridgePairMetrics],
    selected: Sequence[BridgePairMetrics],
    config: BridgeSearchConfig,
) -> tuple[dict[str, object], ...]:
```

Each boundary row must include candidate count, nonzero-match count, qualified count, selected count, maximum verified inliers, and maximum inlier ratio.

- [ ] **Step 6: Run focused tests green**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py
```

---

### Task 4: Implement shared feature-cache creation and targeted bridging attempt

**Files:**
- Modify: `sparse_bridging.py`
- Create: `tests/test_run_sparse_bridging.py`
- Create: `run_sparse_bridging.py`

**Interfaces:**
- Produces: `TargetedGateResult(allowed: bool, reason: str)`
- Produces: `targeted_gate(selected_bridges: Sequence[BridgePairMetrics], config: BridgeSearchConfig) -> TargetedGateResult`
- Produces: `prepare_feature_cache(image_dir: Path, selection_manifest: Path, work_dir: Path, sparse_config: SparseRunConfig) -> Path`
- Produces: `run_targeted_attempt(image_dir: Path, features_database: Path, output_dir: Path, selected_bridges: Sequence[BridgePairMetrics], sparse_config: SparseRunConfig, bridge_config: BridgeSearchConfig) -> AttemptMetrics`
- Produces CLI stage: `python -B run_sparse_bridging.py --stage diagnose`
- Produces CLI stage: `python -B run_sparse_bridging.py --stage targeted`

- [ ] **Step 1: Write feature-cache completion tests**

Use an injected feature-extractor callable in orchestration tests so unit tests do not run real SIFT.

Verify:

- `features_complete.json` is written only after successful extraction and database validation;
- an existing `features.db` without a valid completion marker is treated as partial and rebuilt;
- a valid completion marker records image count, feature count, pyCOLMAP version, selection-manifest SHA, camera model, max image size, and max features.

- [ ] **Step 2: Implement feature cache creation**

Flow:

```text
verify_selected_images(image_dir, selection_manifest, expected_count=288)
create/rebuild reconstruction/bridging/work/features.db
extract_sparse_features(image_dir, features_database, sparse_config)
validate image_count == 288 and feature_count > 0
write features_complete.json atomically
return features.db
```

Do not copy or modify selected JPEGs.

- [ ] **Step 3: Write targeted-attempt orchestration tests with fake runners**

Test that targeted mapping is skipped when any boundary has zero selected bridges:

```python
def test_targeted_stage_skips_when_a_boundary_has_no_bridge():
    first = BridgePairMetrics(
        candidate=BridgeCandidate(73, 74, 70, 111, "left-a.jpg", "right-a.jpg"),
        raw_matches=100,
        verified_inliers=30,
        inlier_ratio=0.30,
        qualified=True,
    )
    second = BridgePairMetrics(
        candidate=BridgeCandidate(145, 146, 140, 181, "left-b.jpg", "right-b.jpg"),
        raw_matches=100,
        verified_inliers=28,
        inlier_ratio=0.28,
        qualified=True,
    )
    result = targeted_gate((first, second), BridgeSearchConfig())
    assert result.allowed is False
    assert "203-204" in result.reason
```

Test that one or more selected bridges for all three boundaries enables the attempt.

- [ ] **Step 4: Implement targeted mapping**

Create a fresh copy of `features.db` as `targeted.db`.

Then run:

```python
pycolmap.match_sequential(
    database_path=targeted_db,
    pairing_options=pycolmap.SequentialPairingOptions(
        overlap=20,
        quadratic_overlap=True,
        loop_detection=False,
    ),
    device=pycolmap.Device.cpu,
)
```

Write only selected qualified bridge pairs to `targeted_bridge_pairs.txt`, then call `pycolmap.match_image_pairs` with imported pairing options.

Call `map_sparse_database` into `reconstruction/bridging/targeted/`.

Build `AttemptMetrics` using:

- `summarize_database(targeted_db)`;
- every returned model;
- best model ranked by registered images, sparse points, lower finite reprojection error;
- attempt name `targeted_bridges`;
- `overlap=20`.

- [ ] **Step 5: Implement the `diagnose` and `targeted` CLI stages**

`diagnose` must:

1. verify Step 10 summary/report files exist;
2. verify the 288-image selection manifest;
3. prepare/reuse the feature cache;
4. generate candidate pairs;
5. run diagnostics;
6. select bridge pairs;
7. write candidate CSV and boundary-summary JSON;
8. not map cameras.

`targeted` must require completed diagnostic reports and either run the allowed targeted attempt or write an explicit skipped result.

- [ ] **Step 6: Run focused tests**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py tests/test_run_sparse_bridging.py
```

---

### Task 5: Implement the fixed Step 11 acceptance gate, exhaustive fallback, and attempt ranking

**Files:**
- Modify: `sparse_bridging.py`
- Modify: `run_sparse_bridging.py`
- Modify: `tests/test_sparse_bridging.py`
- Modify: `tests/test_run_sparse_bridging.py`

**Interfaces:**
- Produces: `bridge_model_accepted(model: ModelMetrics, bridge_config: BridgeSearchConfig) -> bool`
- Produces: `choose_bridge_attempt(attempts: Sequence[AttemptMetrics], bridge_config: BridgeSearchConfig) -> AttemptMetrics`
- Produces: `run_exhaustive_attempt(image_dir: Path, features_database: Path, output_dir: Path, sparse_config: SparseRunConfig, bridge_config: BridgeSearchConfig) -> AttemptMetrics`
- Produces CLI stage: `python -B run_sparse_bridging.py --stage exhaustive`

- [ ] **Step 1: Write acceptance tests**

Assert an accepted model needs all fixed conditions:

```python
accepted = model_metrics(
    registered_images=274,
    sparse_points=1000,
    camera_count=1,
    camera_model="SIMPLE_RADIAL",
    mean_reprojection_error=1.0,
)
assert bridge_model_accepted(accepted, BridgeSearchConfig())
assert not bridge_model_accepted(replace(accepted, registered_images=273), BridgeSearchConfig())
assert not bridge_model_accepted(replace(accepted, sparse_points=999), BridgeSearchConfig())
assert not bridge_model_accepted(replace(accepted, camera_count=2), BridgeSearchConfig())
assert not bridge_model_accepted(replace(accepted, camera_model="OPENCV"), BridgeSearchConfig())
assert not bridge_model_accepted(
    replace(accepted, mean_reprojection_error=float("nan")),
    BridgeSearchConfig(),
)
```

- [ ] **Step 2: Write attempt-ranking tests**

Test these priorities exactly:

1. accepted attempt beats non-accepted attempt;
2. registered-image count;
3. sparse-point count;
4. lower finite reprojection error.

- [ ] **Step 3: Implement exhaustive matching fallback**

Create `exhaustive.db` from the same features-only cache.

Use:

```python
matching_options = pycolmap.FeatureMatchingOptions()
matching_options.use_gpu = False
pairing_options = pycolmap.ExhaustivePairingOptions()
pairing_options.block_size = 50
pycolmap.match_exhaustive(
    database_path=exhaustive_db,
    matching_options=matching_options,
    pairing_options=pairing_options,
    device=pycolmap.Device.cpu,
)
```

Then run `map_sparse_database(exhaustive_db, image_dir, output_dir, sparse_config)`, rank the returned models by registered images, sparse points, and lower finite reprojection error, and return an `AttemptMetrics` with `name="exhaustive"`, `workspace=output_dir`, `overlap=0`, the measured `summarize_database(exhaustive_db)` result, all models, the ranked best model, measured runtime, and `pycolmap.__version__`.

- [ ] **Step 4: Implement the fallback gate**

The `exhaustive` CLI stage runs only when:

- targeted was skipped, or
- targeted completed and `bridge_model_accepted(targeted.best_model, bridge_config)` is false.

If targeted already passes acceptance, exhaustive must be skipped with a report explaining that the fallback was unnecessary.

- [ ] **Step 5: Add long-running process guidance to CLI output**

Before starting exhaustive matching, print a single concise line with:

```text
Step 11 exhaustive fallback: 288 images, CPU SIFT matching, block_size=50; this may outlive an interactive tool timeout.
```

Do not add internal watchdog threads. External Codex execution should check the process before restarting after a wrapper timeout.

- [ ] **Step 6: Run focused tests green**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py tests/test_run_sparse_bridging.py
```

---

### Task 6: Implement finalization, reports, figures, and bounded cleanup

**Files:**
- Modify: `run_sparse_bridging.py`
- Modify: `tests/test_run_sparse_bridging.py`

**Interfaces:**
- Produces CLI stage: `python -B run_sparse_bridging.py --stage finalize`
- Produces CLI stage: `python -B run_sparse_bridging.py --stage all`
- Produces: `reconstruction/bridging/reports/step11_candidates.csv`
- Produces: `reconstruction/bridging/reports/step11_boundary_summary.json`
- Produces: `reconstruction/bridging/reports/step11_attempts.csv`
- Produces: `reconstruction/bridging/reports/step11_registered_images.csv`
- Produces: `reconstruction/bridging/reports/step11_summary.json`
- Produces: `reconstruction/bridging/previews/step11_01_bridge_candidates.png`
- Produces: `reconstruction/bridging/previews/step11_02_sparse_model.png`
- Produces: `reconstruction/bridging/previews/step11_03_registration.png`
- Produces: `reconstruction/bridging/best/`

- [ ] **Step 1: Write final-summary contract tests**

The summary must include exactly the evidence required by the spec and explicitly contain:

```python
assert summary["bridge_success"] in (True, False)
assert summary["dense_reconstruction_started"] is False
assert summary["step10_baseline"]["registered_images"] == 73
assert summary["critical_boundaries"] == [[73, 74], [145, 146], [203, 204]]
```

When no Step 11 attempt passes acceptance, `bridge_success` must remain false even if one attempt is better than Step 10.

- [ ] **Step 2: Implement selected-model copy/export without touching Step 10**

Choose among Step 11 attempts with `choose_bridge_attempt`.

Copy the selected Step 11 model to:

```text
reconstruction/bridging/best/
```

Use existing `copy_sparse_model`. Reopen the copied model through pyCOLMAP, then export:

```text
reconstruction/bridging/best/points3D.ply
```

Never modify `reconstruction/sparse/best/`.

- [ ] **Step 3: Implement candidate figure**

Generate the figure from `step11_candidates.csv` only. Use three vertically stacked panels or one clearly separated axis group, one per boundary. Show candidate inlier count and visually distinguish selected bridge pairs.

Do not encode unmeasured values or manually annotate a bridge as successful before mapping evidence exists.

- [ ] **Step 4: Implement sparse-model and registration figures**

Reuse Step 10's real pyCOLMAP reconstruction access pattern:

- 3D sparse points;
- registered camera centers;
- title includes selected attempt and `registered/288` count.

Registration figure covers all selected indices 1-288 and marks boundaries 73-74, 145-146, 203-204.

- [ ] **Step 5: Implement `finalize` and `all` stages**

`all` sequence:

```text
diagnose
if all boundaries have selected bridges: targeted
if targeted skipped or targeted best fails acceptance: exhaustive
finalize
```

`finalize` must be resumable from completed JSON reports and existing sparse models without re-running matching.

- [ ] **Step 6: Implement bounded cleanup helper**

After final reports/model verification, remove only:

```text
reconstruction/bridging/work/features.db
reconstruction/bridging/work/features_complete.json
reconstruction/bridging/work/diagnostic.db
reconstruction/bridging/work/diagnostic_pairs.txt
reconstruction/bridging/work/targeted.db
reconstruction/bridging/work/targeted_bridge_pairs.txt
reconstruction/bridging/work/exhaustive.db
```

Remove only files that exist and are regular files inside `reconstruction/bridging/work/`. Then remove the empty `work/` directory. Do not remove targeted/exhaustive sparse models or reports.

- [ ] **Step 7: Run focused tests green**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py tests/test_run_sparse_bridging.py
```

---

### Task 7: Execute Step 11 on the real 288-image dataset

**Files generated:**
- `reconstruction/bridging/targeted/` when targeted gate allows it
- `reconstruction/bridging/exhaustive/` when fallback is required
- `reconstruction/bridging/best/`
- `reconstruction/bridging/reports/*`
- `reconstruction/bridging/previews/*`

- [ ] **Step 1: Establish a fresh pre-execution gate**

Before running Step 11:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_reconstruction.py tests/test_run_sparse_reconstruction.py tests/test_sparse_bridging.py tests/test_run_sparse_bridging.py
```

Then verify:

```text
297 raw images unchanged
288 selected images match selection_manifest.csv
Step 10 summary exists and reports 73 registered images / 6099 points for the selected baseline
reconstruction/sparse/best opens through pyCOLMAP
```

- [ ] **Step 2: Run diagnostics**

```powershell
python -B run_sparse_bridging.py --stage diagnose
```

Inspect `step11_boundary_summary.json` before deciding whether targeted mapping is permitted. Do not change thresholds after seeing the result.

- [ ] **Step 3: Run targeted mapping only when the frozen gate permits it**

```powershell
python -B run_sparse_bridging.py --stage targeted
```

If the stage reports it was skipped because a critical boundary has zero qualified bridge pairs, accept the skip and continue to exhaustive fallback. Do not lower the 15-inlier / 0.15-ratio gate.

- [ ] **Step 4: Measure the targeted result before fallback**

Record:

```text
model count
best registered images
sparse points
mean track length
mean reprojection error
camera count/model/params
unregistered image ranges
```

If the best targeted model passes the fixed acceptance gate, do not run exhaustive matching.

- [ ] **Step 5: Run the one exhaustive fallback only when required**

```powershell
python -B run_sparse_bridging.py --stage exhaustive
```

Because this is CPU-only and may exceed an interactive timeout, if the tool call times out:

1. check whether the exact `run_sparse_bridging.py --stage exhaustive` process is still alive;
2. inspect whether `exhaustive.db` or sparse model outputs are growing;
3. wait/poll the existing process when it is active;
4. never start a duplicate exhaustive job merely because the wrapper timed out.

- [ ] **Step 6: Finalize**

```powershell
python -B run_sparse_bridging.py --stage finalize
```

Read `step11_summary.json` before documenting success or failure.

- [ ] **Step 7: Visually inspect all generated Step 11 figures**

Confirm:

- candidate figure corresponds to the CSV;
- selected bridge markers are real measured candidates;
- sparse camera centers are coherent rather than scattered/teleported;
- registration figure corresponds to the selected model, not union coverage;
- titles state the actual attempt and registration count.

If visual geometry is implausible despite a high registration count, `bridge_success` must not be treated as sufficient for dense progression until the defect is diagnosed.

---

### Task 8: Document measured Step 11 outcome and update project state

**Files:**
- Create: `docs/geometry-ml/sparse-component-bridging.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/memory-bank/active-context.md`
- Modify: `docs/memory-bank/progress.md`
- Modify: `preprocessing/pycolmap_input/README.md`

- [ ] **Step 1: Read real Step 11 evidence before editing documentation**

Use the final JSON/CSV and reopened pyCOLMAP model. Never guess candidate counts, model counts, registered-image totals, points, errors, or success state.

- [ ] **Step 2: Write the measured Step 11 report**

The report must include:

- the Step 10 fragmentation baseline;
- why Step 11 excluded <=40-gap pairs from targeted diagnostics;
- candidate count and qualified/selected bridge pairs per boundary;
- whether targeted mapping ran or was skipped;
- targeted metrics when run;
- whether exhaustive fallback ran;
- exhaustive metrics when run;
- selected Step 11 attempt;
- final camera parameters;
- visual review;
- `bridge_success` and exact acceptance reason;
- explicit dense-reconstruction boundary.

- [ ] **Step 3: Update durable project status accurately**

If `bridge_success=true`, project state becomes:

```text
Step 11 sparse component bridging complete and accepted; dense reconstruction is the next separately authorized phase.
```

If `bridge_success=false`, project state becomes:

```text
Step 11 bridging attempts complete but global sparse reconstruction remains below acceptance; dense reconstruction remains blocked.
```

Do not phrase a failed acceptance as a successful full reconstruction.

---

### Task 9: Final verification, review, cleanup, and optional publication

- [ ] **Step 1: Run focused Step 11 tests**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_bridging.py tests/test_run_sparse_bridging.py
```

- [ ] **Step 2: Run the full project suite**

```powershell
python -B -m pytest -p no:cacheprovider -q
```

Report the actual observed test count only.

- [ ] **Step 3: Compile changed Python**

```powershell
python -B -m py_compile sparse_reconstruction.py sparse_bridging.py run_sparse_reconstruction.py run_sparse_bridging.py
```

- [ ] **Step 4: Reopen and cross-check the selected Step 11 sparse model**

When `reconstruction/bridging/best/` exists:

- reopen it with pyCOLMAP;
- assert registered-image count matches `step11_summary.json`;
- assert sparse-point count matches;
- assert camera count/model match;
- assert mean reprojection error matches within floating-point tolerance.

- [ ] **Step 5: Reverify protected data and Step 10 preservation**

Require:

```text
297/297 raw unchanged
288/288 selected verified
reconstruction/sparse/best still exists and opens
Step 10 summary unchanged
```

- [ ] **Step 6: Verify the dense boundary**

Search the Step 11 implementation and output tree for dense/MVS/mesh/texture calls/artifacts. No Step 11 code may invoke:

```text
undistort_images
patch_match_stereo
stereo_fusion
poisson_meshing
```

No dense/mesh/texture/Blender output may exist under `reconstruction/bridging/`.

- [ ] **Step 7: Remove task-created caches after all verification commands finish**

Remove only task-created `__pycache__/`, `.pytest_cache/`, temporary review images, and transient `reconstruction/bridging/work/` files defined by Task 6. Preserve reports, figures, targeted/exhaustive models, best model, source, tests, spec, and plan.

- [ ] **Step 8: Review Git state and intended diff**

Confirm:

- no DOCX/PDF modifications;
- no raw or selected JPEG changes;
- local CNN checkpoint is not staged;
- no transient database is staged;
- no secret-bearing file is staged;
- no unrelated contributor work is included.

- [ ] **Step 9: Commit and push only when the user explicitly authorized publication in the implementation task**

If authorized, preferred commit message:

```text
feat(sfm): bridge sparse reconstruction components
```

Push `main` without force and verify local `HEAD`, fetched `origin/main`, and `git ls-remote origin refs/heads/main` are equal.

If publication was not explicitly authorized, leave the completed implementation uncommitted and report the exact Git state instead.
