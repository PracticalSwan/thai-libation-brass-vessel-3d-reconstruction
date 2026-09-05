# Step 10 Sparse SfM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible pyCOLMAP sparse Structure-from-Motion pipeline over the verified 288-image PREPROCESSED set, diagnose the resulting camera registration and sparse geometry, perform at most one evidence-backed overlap retry, and publish the selected sparse model plus measured reports.

**Architecture:** Keep reconstruction logic in `sparse_reconstruction.py` and orchestration in `run_sparse_reconstruction.py`. The runtime uses one shared `SIMPLE_RADIAL` camera initialized from Step 9 EXIF evidence, pyCOLMAP native SIFT, sequential matching, and incremental mapping. A baseline with overlap 20 is primary; only if its largest model fails the fixed registration acceptance gate may a fresh overlap-40 retry run.

**Tech Stack:** Python 3.14, pyCOLMAP 4.2.x, NumPy, matplotlib, SQLite, pytest, existing project integrity verifiers.

**Spec:** `docs/superpowers/specs/2026-09-05-step-10-sparse-sfm-design.md`

**Execution result (2026-09-05):** Implemented and run on all 288 verified inputs. Both the overlap-20 baseline and single overlap-40 retry produced seven disconnected sparse models; the largest model registered 73 images. The frozen ranking selected the baseline 73-image / 6,099-point component, `acceptance_met=false`, and Step 10 stopped before dense reconstruction as designed. Measured results are in `docs/geometry-ml/sparse-reconstruction.md`.

## Global Constraints

- Never modify `IMG20260826122949/` or `preprocessing/pycolmap_input/images/`.
- Use all 288 selected PREPROCESSED images unless the mapper itself leaves images unregistered.
- Use unmasked pyCOLMAP native SIFT; do not feed CNN masks into feature extraction.
- Use one shared `SIMPLE_RADIAL` camera initialized from the measured 26 mm 35-mm-equivalent camera evidence.
- Baseline sequential overlap = 20; controlled retry overlap = 40 only if baseline fails the acceptance gate.
- No exhaustive matching, vocabulary-tree download, learned features, camera-model sweep, dense MVS, mesh, texture, or Blender in Step 10.
- Keep the local CNN checkpoint untouched and unpublished.

---

### Task 1: Add pyCOLMAP dependency and pure camera/configuration helpers

**Files:**
- Modify: `requirements.txt`
- Create: `sparse_reconstruction.py`
- Create: `tests/test_sparse_reconstruction.py`

**Interfaces:**
- Produces: `SparseRunConfig`
- Produces: `focal_pixels_from_35mm_equivalent(width: int, height: int, focal_35mm: float) -> float`
- Produces: `simple_radial_camera_params(width: int, height: int, focal_35mm: float) -> tuple[float, float, float, float]`
- Produces: `should_retry(metrics: ModelMetrics, total_images: int) -> bool`
- Produces: `choose_best_attempt(attempts: Sequence[AttemptMetrics]) -> AttemptMetrics`

- [ ] **Step 1: Write failing pure-function tests**

```python
def test_focal_pixels_uses_full_frame_diagonal_equivalence():
    focal = focal_pixels_from_35mm_equivalent(3072, 4080, 26.0)
    assert focal == pytest.approx(3070.0, rel=0.01)


def test_simple_radial_initialization_uses_center_and_zero_distortion():
    f, cx, cy, k = simple_radial_camera_params(3072, 4080, 26.0)
    assert cx == 1536.0
    assert cy == 2040.0
    assert k == 0.0
    assert f > 3000


def test_retry_only_when_registration_gate_fails():
    accepted = ModelMetrics(
        model_path=Path("accepted"),
        registered_images=274,
        total_images=288,
        sparse_points=2000,
        observations=6000,
        mean_track_length=3.0,
        mean_reprojection_error=0.8,
        camera_count=1,
        camera_model="SIMPLE_RADIAL",
        camera_params=(3070.0, 1536.0, 2040.0, 0.0),
    )
    incomplete = replace(accepted, registered_images=273)
    assert not should_retry(accepted, 288)
    assert should_retry(incomplete, 288)
```

Also test attempt ranking with two explicit `AttemptMetrics` instances: registration count first, then sparse-point count, then lower reprojection error.

- [ ] **Step 2: Run red test**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_reconstruction.py
```

Expected: import failure because `sparse_reconstruction.py` does not yet exist.

- [ ] **Step 3: Install and verify pyCOLMAP, then add bounded requirement**

```powershell
python -m pip install "pycolmap>=4.2,<5"
python -B -c "import pycolmap; print(pycolmap.__version__)"
```

Require an importable 4.x version. Add exactly `pycolmap>=4.2,<5` to `requirements.txt`; do not add Open3D or other reconstruction packages.

- [ ] **Step 4: Implement the minimal pure helpers and dataclasses**

`SparseRunConfig` constants:

```text
camera_model = SIMPLE_RADIAL
focal_35mm = 26.0
max_image_size = 1200
max_num_features = 8192
baseline_overlap = 20
retry_overlap = 40
minimum_registered_images = 274
minimum_sparse_points = 1000
```

- [ ] **Step 5: Run green test**

Run the focused test file and require PASS.

---

### Task 2: Add pyCOLMAP runtime wrappers and database/model metrics

**Files:**
- Modify: `sparse_reconstruction.py`
- Modify: `tests/test_sparse_reconstruction.py`

**Interfaces:**
- Produces: `run_sparse_attempt(image_dir: Path, workspace: Path, *, overlap: int, config: SparseRunConfig) -> AttemptMetrics`
- Produces: `summarize_database(database_path: Path) -> DatabaseMetrics`
- Produces: `summarize_reconstruction(model_path: Path, total_images: int) -> ModelMetrics`
- Produces: `registered_image_names(model_path: Path) -> tuple[str, ...]`

- [ ] **Step 1: Add failing contract tests using tiny temporary SQLite fixtures and fake model metrics**

Tests must verify:

```text
workspace cannot overlap raw/selected input directory
sequential overlap must be positive
input image count must equal expected total
attempt metric selection is deterministic
```

Database summary test creates a minimal SQLite file with these schemas and rows:

```sql
CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE keypoints(image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
CREATE TABLE matches(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER, cols INTEGER, data BLOB);
```

Insert two image rows, keypoint rows totaling 30, one non-empty match row, and one non-empty two-view row; assert exactly those counts.

- [ ] **Step 2: Run tests red**

Confirm failures are due to missing runtime/database helpers.

- [ ] **Step 3: Implement database metrics with `sqlite3`**

Count:

```text
images
SUM(keypoints.rows)
rows in matches where rows > 0
rows in two_view_geometries where rows > 0
```

Do not mutate the database during summarization.

- [ ] **Step 4: Implement pyCOLMAP attempt wrapper**

Flow:

```python
reader = pycolmap.ImageReaderOptions(
    camera_model="SIMPLE_RADIAL",
    camera_params=",".join(f"{value:.12g}" for value in camera_params),
)
feature_options = pycolmap.FeatureExtractionOptions()
feature_options.sift.max_num_features = config.max_num_features
feature_options.max_image_size = config.max_image_size
pycolmap.extract_features(
    database_path=db_path,
    image_path=image_dir,
    camera_mode=pycolmap.CameraMode.SINGLE,
    reader_options=reader,
    extraction_options=feature_options,
    device=pycolmap.Device.cpu,
)
pycolmap.match_sequential(
    database_path=db_path,
    pairing_options=pycolmap.SequentialPairingOptions(
        overlap=overlap,
        quadratic_overlap=True,
        loop_detection=False,
    ),
    device=pycolmap.Device.cpu,
)
models = pycolmap.incremental_mapping(
    database_path=db_path,
    image_path=image_dir,
    output_path=sparse_dir,
    options=pipeline_options,
)
```

If the installed 4.2 API exposes `max_image_size` or SIFT feature-count fields at a different nested path, inspect `FeatureExtractionOptions().todict()` once and use the exact 4.2 field names without changing the frozen values.

Set pipeline `multiple_models=True`, `min_model_size=10`, `min_num_matches=15`, `ba_refine_focal_length=True`, `ba_refine_principal_point=False`, `ba_refine_extra_params=True`, and `random_seed=4213` when supported.

- [ ] **Step 5: Run focused tests green**

No real 288-image reconstruction in the test suite.

---

### Task 3: Add Step 10 orchestration, reports, and figures

**Files:**
- Create: `run_sparse_reconstruction.py`
- Create: `tests/test_run_sparse_reconstruction.py`

**Interfaces:**
- CLI: `python -B run_sparse_reconstruction.py --stage {all,baseline,retry,finalize}`
- Produces: `execute_attempt_sequence(image_dir: Path, reconstruction_root: Path, *, runner: Callable[..., AttemptMetrics], config: SparseRunConfig) -> tuple[AttemptMetrics, ...]`
- Produces: `reconstruction/reports/step10_attempts.csv`
- Produces: `reconstruction/reports/step10_summary.json`
- Produces: `reconstruction/reports/step10_registered_images.csv`
- Produces: `reconstruction/previews/step10_01_sparse_model.png`
- Produces: `reconstruction/previews/step10_02_registration.png`
- Produces: `reconstruction/sparse/best/`

- [ ] **Step 1: Write failing orchestration tests**

Test pure/isolated orchestration behaviors:

```python
def test_attempt_sequence_skips_retry_when_baseline_passes():
    calls = []
    def runner(*, overlap, **kwargs):
        calls.append(overlap)
        return accepted_attempt(overlap)
    attempts = execute_attempt_sequence(
        image_dir=Path("images"),
        reconstruction_root=Path("reconstruction"),
        runner=runner,
        config=SparseRunConfig(),
    )
    assert calls == [20]
    assert len(attempts) == 1


def test_attempt_sequence_runs_overlap40_only_after_failed_baseline():
    calls = []
    def runner(*, overlap, **kwargs):
        calls.append(overlap)
        return incomplete_attempt(overlap) if overlap == 20 else accepted_attempt(overlap)
    attempts = execute_attempt_sequence(
        image_dir=Path("images"),
        reconstruction_root=Path("reconstruction"),
        runner=runner,
        config=SparseRunConfig(),
    )
    assert calls == [20, 40]
```

Also verify `copy_best_model` copies only sparse-model files and generated summary sets `dense_reconstruction_started` to `False`.

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement bounded orchestration and resumable stages**

Workflow:

```text
verify 288 selected inputs
record pyCOLMAP version
baseline stage: create/reuse completed overlap-20 attempt
retry stage: permitted only when baseline summary fails should_retry gate
finalize stage: select best model, copy sparse model to reconstruction/sparse/best/, export PLY when supported, write CSV/JSON evidence, render figures
all stage: baseline -> optional retry -> finalize
```

A completed attempt is resumable only when its attempt JSON exists and the referenced sparse model reopens successfully with pyCOLMAP; otherwise the attempt directory is treated as partial and rebuilt.

The script must not call `undistort_images`, `patch_match_stereo`, `stereo_fusion`, `poisson_meshing`, or any dense/mesh API.

- [ ] **Step 4: Run focused tests green**

---

### Task 4: Execute real baseline and controlled retry only if required

**Files generated:**
- `reconstruction/sparse/baseline/`
- `reconstruction/sparse/retry_overlap40/` only if triggered
- `reconstruction/sparse/best/`
- `reconstruction/reports/*`
- `reconstruction/previews/*`

- [ ] **Step 1: Run the real baseline stage**

```powershell
python -B run_sparse_reconstruction.py --stage baseline
```

Use the full 288-image verified input. The stage writes progress-complete evidence only after feature extraction, matching, and incremental mapping return successfully.

- [ ] **Step 2: Inspect baseline attempt JSON**

Check:

```text
registered image count / 288
number of sparse models
sparse points
mean track length
mean reprojection error
final shared camera parameters
unregistered image list
```

- [ ] **Step 3: Apply the fixed retry rule only if baseline requires it**

If `should_retry` is false, do not run a retry. If true:

```powershell
python -B run_sparse_reconstruction.py --stage retry
```

The retry must use overlap 40 and otherwise the same configuration.

- [ ] **Step 4: Finalize selected model and figures**

```powershell
python -B run_sparse_reconstruction.py --stage finalize
```

- [ ] **Step 5: Visually inspect both Step 10 figures**

Sparse preview must show plausible camera centers around the object point cloud without obviously impossible scattering/teleportation. Registration plot must match the reported registered/unregistered list.

If the sparse preview is obviously invalid despite high registration, diagnose the root cause before changing configuration. Any correction remains limited to the approved camera initialization and one overlap-40 retry unless the pipeline cannot initialize at all.

---

### Task 5: Document measured results and update project state

**Files:**
- Create: `docs/geometry-ml/sparse-reconstruction.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/memory-bank/active-context.md`
- Modify: `docs/memory-bank/progress.md`
- Modify: `preprocessing/pycolmap_input/README.md`

- [ ] **Step 1: Read actual Step 10 JSON/CSV/model metrics before editing docs**

Do not guess registration, point, camera, or reprojection numbers.

- [ ] **Step 2: Write measured sparse-reconstruction report**

Include:

```text
pyCOLMAP version
camera initialization derivation
feature/matching configuration
baseline and retry decision
registered/unregistered images
sparse-point/track/reprojection metrics
final camera parameters
visual-review observations
limitations
explicit stop before dense reconstruction
```

- [ ] **Step 3: Update status docs**

Advance project state to “Step 10 sparse SfM complete; next phase is dense reconstruction only if separately authorized.” Do not claim dense/mesh/texture completion.

---

### Task 6: Final verification, cleanup, review, commit, and push

- [ ] **Step 1: Run focused Step 10 tests**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_sparse_reconstruction.py tests/test_run_sparse_reconstruction.py
```

- [ ] **Step 2: Run complete project tests**

```powershell
python -B -m pytest -p no:cacheprovider -q
```

- [ ] **Step 3: Compile changed Python**

```powershell
python -B -m py_compile sparse_reconstruction.py run_sparse_reconstruction.py
```

- [ ] **Step 4: Reopen selected sparse model with pyCOLMAP**

Assert registered-image count, sparse-point count, mean reprojection error, and camera count match `step10_summary.json`.

- [ ] **Step 5: Reverify protected sources**

Require:

```text
297/297 raw unchanged
288/288 selected verified
```

- [ ] **Step 6: Inspect final artifacts and cleanup**

Remove `__pycache__/`, `.pytest_cache/`, temporary review images, failed partial attempt directories, and transient `database.db` files if they are not needed as a final deliverable. Do not remove selected sparse binary model, PLY, reports, figures, plans/spec, or source code.

- [ ] **Step 7: Two-stage self-review**

First verify spec compliance; then review code quality/correctness. Fix only material Step 10 defects and rerun affected verification.

- [ ] **Step 8: Inspect Git state**

Confirm no DOCX/PDF modifications, no raw/selected JPEG changes, no local CNN checkpoint staged, no secrets, no dense/MVS/mesh/texture/Blender artifacts, and no unrelated work.

- [ ] **Step 9: Commit implementation**

Preferred message:

```text
feat(sfm): complete sparse reconstruction
```

- [ ] **Step 10: Push `main` and verify remote equality**

No force push. Fetch `origin/main` and verify local `HEAD`, fetched `origin/main`, and `git ls-remote origin refs/heads/main` are identical.
