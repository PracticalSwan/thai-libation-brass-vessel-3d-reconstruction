# Final CV Evidence and Measurement Implementation Plan

> **For agentic workers:** Execute this checklist task-by-task using any relevant skills/plugins. GLM-only subagents may be used for independent image/CV reviews; verify their findings in the parent agent. If an installed plan-execution workflow is available, use it; otherwise follow this file directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the V2 computer-vision evidence package, canonical camera/reference set, component measurements, ornament references, and multi-view parametric fit that will drive Blender geometry.

**Architecture:** Reuse the existing reviewed masks/CNN predictions, Step 6 SIFT/RANSAC utilities, and Step 13 ALIKED/LightGlue + pyCOLMAP camera solution. V2 computes a deterministic set of source views, CV diagnostics, landmarks, component profiles, and per-view residuals. No Blender geometry is created until this plan passes.

**Tech Stack:** Python, OpenCV, NumPy, pyCOLMAP 4.2, existing PyTorch/CNN outputs, existing ALIKED/LightGlue code, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

**Verified completion (2026-09-08):** Plan 1 is accepted and frozen for Blender consumption. Final candidate metrics: median silhouette IoU `0.901495`, minimum reliable-view IoU `0.856102`, median landmark error `0.011306` object height, p95 landmark error `0.038080`; exact candidate-bound bowl/globe/neck/lid/finial visual vetoes all pass. The reference geometry-mask/landmark review is separately hash-bound. Start Plan 2 immediately unless downstream Blender evidence proves this accepted contract wrong.

## Global Constraints

- Read source photos and protected reconstruction evidence only; never modify them.
- Reuse the existing CNN and outputs. Do not retrain unless a new explicit task authorizes it.
- Reuse Step 6 SIFT utilities rather than creating a second unrelated SIFT implementation.
- Reuse Step 13 camera evidence; do not rerun global mapping.
- Step 13 sparse points are not final surface geometry.
- Canonical geometry views should be Step 13-registered whenever possible.
- Unregistered detail views 267/268/278/288 are detail/ornament references, not primary camera-fit views.
- Output only under `reconstruction/reference_assisted_v2/` and new V2 source/tests/docs.
- Computer vision here is functional, not ceremonial: SIFT/RANSAC, CNN masks, learned matching where needed, and Step 13 cameras must materially constrain the V2 model or its validation.
- Use substantial compute for evidence-directed camera/profile fitting and diagnostic rendering when it improves same-object fidelity; do not stop merely because a minimum threshold is first crossed.
- If an intended CV method fails, diagnose and use a stronger evidence-preserving method rather than weakening the final requirement. Do not fabricate measurements or model output.

---

## File structure

**Create:**

```text
final_reference_evidence.py
final_cv_model_fit.py
final_model_io.py
tests/test_final_reference_evidence.py
tests/test_final_cv_model_fit.py
tests/test_final_model_io.py
```

**Consume read-only:**

```text
ml_dataset/manifest.csv
ml_dataset/masks/*.png
analysis/ml/full_predictions/*.png
analysis/ml/reconstruction_masks/*.png
analysis/ml/checkpoints/best_small_seg_cnn.pt
geometry_detection.py
external_learned_recovery.py
reconstruction/external_learned_recovery/best/
reconstruction/external_learned_recovery/reports/step13_registered_images.csv
reconstruction/external_learned_recovery/reports/step13_summary.json
reconstruction/reports/step9_summary.json
```

**Produce:**

```text
reconstruction/reference_assisted_v2/evidence/reference_boards/*.png
reconstruction/reference_assisted_v2/evidence/aligned_views/*.png
reconstruction/reference_assisted_v2/evidence/annotations/landmarks.json
reconstruction/reference_assisted_v2/evidence/component_masks/*.png
reconstruction/reference_assisted_v2/evidence/cv_matches/*.png
reconstruction/reference_assisted_v2/evidence/ornament_crops/*.png
reconstruction/reference_assisted_v2/reports/final_reference_evidence.json
reconstruction/reference_assisted_v2/reports/final_cv_fit.json
reconstruction/reference_assisted_v2/reports/final_profiles.json
```

---

### Task 1: Add V2 safe-path and artifact ownership contracts

**Files:**
- Create: `final_model_io.py`
- Create: `tests/test_final_model_io.py`

**Interfaces:**
- Produces: `V2_ROOT`, `ensure_under_v2_root()`, `sha256_file()`, `write_json_atomic()`, `write_csv_atomic()`, `owned_work_path()`.
- Consumed by all later V2 tasks.

- [ ] **Step 1: Write failing tests for V2 boundary enforcement**

```python
from pathlib import Path
import pytest
from final_model_io import ensure_under_v2_root


def test_v2_boundary_accepts_owned_output(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    path = root / "reports" / "x.json"
    assert ensure_under_v2_root(path, root) == path.resolve()


def test_v2_boundary_rejects_parent_escape(tmp_path: Path):
    root = tmp_path / "reconstruction" / "reference_assisted_v2"
    with pytest.raises(ValueError, match="outside V2 root"):
        ensure_under_v2_root(tmp_path / "IMG20260826122949" / "x.jpg", root)
```

- [ ] **Step 2: Run tests and verify failure because functions do not yet exist**

Run:

```text
python -B -m pytest tests/test_final_model_io.py -q
```

Expected: collection/import failure for missing V2 IO module.

- [ ] **Step 3: Implement exact safe IO primitives**

Required signatures:

```python
V2_RELATIVE_ROOT = Path("reconstruction/reference_assisted_v2")


def ensure_under_v2_root(path: Path, v2_root: Path) -> Path: ...
def sha256_file(path: Path) -> str: ...
def write_json_atomic(path: Path, payload: object, v2_root: Path) -> None: ...
def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]], v2_root: Path) -> None: ...
def owned_work_path(path: Path, v2_root: Path) -> bool: ...
```

Implementation rules:

- resolve both path and root;
- require `path == root` or root in `path.parents`;
- JSON uses sorted deterministic keys only when semantic order is irrelevant;
- atomic write uses same-directory temporary file and `replace()`;
- never create/modify outside V2 root from these helpers.

- [ ] **Step 4: Re-run focused tests**

Expected: all pass.

---

### Task 2: Load reviewed masks, CNN predictions, and Step 13 registration into one reference catalog

**Files:**
- Create: `final_reference_evidence.py`
- Create: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces: `FinalReferenceView`, `load_reviewed_reference_views()`, `load_step13_registration()`.
- Consumes: `sha256_file()`.

- [ ] **Step 1: Write tests for reviewed-manifest parsing**

```python
from pathlib import Path
from final_reference_evidence import load_reviewed_reference_views


def test_reference_catalog_preserves_view_categories(project_root: Path):
    views = load_reviewed_reference_views(project_root)
    assert len(views) == 36
    categories = {v.view_category for v in views}
    assert categories == {
        "normal_side",
        "low_angle_pedestal",
        "elevated_oblique",
        "top_down_rim",
        "oblique_detail",
    }
```

Use the existing project-root fixture pattern if one exists. Otherwise create a tiny temporary manifest fixture for pure unit coverage and reserve the 36-count assertion for a real-input integration test marked appropriately.

- [ ] **Step 2: Define the exact dataclass**

```python
@dataclass(frozen=True)
class FinalReferenceView:
    selected_index: int
    filename: str
    split: str
    view_category: str
    quality_condition: str
    source_sha256: str
    reviewed_mask_path: Path
    reviewed_mask_sha256: str
    cnn_prediction_path: Path
    reconstruction_mask_path: Path
    step13_registered: bool
    step13_image_id: int | None
```

- [ ] **Step 3: Implement catalog loading with fail-closed provenance**

For every reviewed row:

1. resolve immutable raw photo;
2. hash and compare to manifest `source_sha256`;
3. resolve reviewed mask;
4. hash and compare to `mask_sha256`;
5. resolve matching full CNN prediction by selected-index prefix;
6. resolve matching reconstruction mask;
7. load `step13_registered_images.csv`;
8. mark registration truthfully.

Do not generate replacement masks when a required reviewed mask is missing.

- [ ] **Step 4: Add real-input smoke assertion**

Expected facts:

```text
reviewed views = 36
normal_side = 8
low_angle_pedestal = 8
elevated_oblique = 8
top_down_rim = 8
oblique_detail = 4
267/268/278/288 step13_registered = false
```

- [ ] **Step 5: Run focused tests**

```text
python -B -m pytest tests/test_final_reference_evidence.py -q
```

---

### Task 3: Create the V1 rejection baseline and source reference boards

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces: `build_reference_boards()`, `build_v1_rejection_board()`.

- [ ] **Step 1: Write a deterministic image-grid helper test**

The helper must keep aspect ratio, label source filename/index, and avoid modifying originals.

Signature:

```python
def make_labeled_contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    *,
    cell_size: tuple[int, int] = (512, 512),
    columns: int = 4,
) -> Path: ...
```

- [ ] **Step 2: Implement category boards**

Create boards for:

```text
normal_side
low_angle_pedestal
elevated_oblique
top_down_rim
oblique_detail
neck_lid_detail
globe_ornament_detail
bowl_pedestal_detail
```

The first five come directly from manifest categories. The last three are selected from source photos by explicit filename/index list recorded in `final_reference_evidence.json`.

- [ ] **Step 3: Create V1 rejection evidence**

Use the existing V1 render(s), at minimum:

```text
reconstruction/reference_assisted/previews/reference_front.png
reconstruction/reference_assisted/previews/reference_quarter.png
```

Place next to representative originals such as indices 3/72/148/165. Add text labels only. Do not manipulate the original imagery beyond resize/crop-to-fit.

The report must record V1 mismatch categories:

```text
wrong globe/body profile
wrong bowl/pedestal profile
wrong neck proportions
unsupported generic decoration
unsupported chain unless later evidence found
material/value mismatch
insufficient single-aggregate validation
```

- [ ] **Step 4: Open the generated boards for visual inspection**

Reject any board with wrong source labels, unreadable crops, or mismatched orientation.

---

### Task 4: Establish the canonical 16-view geometry set from CV/SfM evidence

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces: `select_canonical_geometry_views()`.

- [ ] **Step 1: Write selection-policy tests**

Selection must require four registered views from each category:

```text
normal_side
low_angle_pedestal
elevated_oblique
top_down_rim
```

and reject unregistered views for the primary set.

Starting preferred indices:

```python
PREFERRED_GEOMETRY_INDICES = {
    "normal_side": (3, 19, 50, 72),
    "low_angle_pedestal": (90, 114, 128, 142),
    "elevated_oblique": (148, 165, 188, 200),
    "top_down_rim": (206, 221, 243, 255),
}
```

- [ ] **Step 2: Implement deterministic replacement policy**

When a preferred view fails readability, hash, reviewed-mask, or Step 13 registration:

1. choose another registered view from same category;
2. prefer `quality_condition == "normal"`;
3. then prefer validation/test split diversity only as a tie-breaker, not as a ML-evaluation claim;
4. record replacement reason.

Do not substitute across categories.

- [ ] **Step 3: Persist the exact canonical set**

Write into `final_reference_evidence.json`:

```json
{
  "canonical_geometry_views": [
    {
      "selected_index": 3,
      "filename": "IMG...jpg",
      "view_category": "normal_side",
      "step13_registered": true,
      "selection_reason": "preferred_registered_normal_quality"
    }
  ]
}
```

---

### Task 5: Reuse Step 6 SIFT/RANSAC for local correspondence diagnostics

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`
- Read-only dependency: `geometry_detection.py`

**Interfaces:**
- Produces: `build_sift_pair_diagnostic()`.

- [ ] **Step 1: Write a test using synthetic translated features/images or existing helper fakes**

The new code must call/import the existing Step 6 interfaces rather than duplicating SIFT logic:

```python
from geometry_detection import (
    SiftConfig,
    extract_sift,
    match_sift,
    estimate_fundamental_geometry,
)
```

- [ ] **Step 2: Implement pair diagnostic**

Signature:

```python
def build_sift_pair_diagnostic(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    *,
    config: SiftConfig = SiftConfig(maximum_width=1200),
) -> dict: ...
```

Return/report:

```text
candidate_matches
ransac_inliers
inlier_ratio
fundamental_matrix_status
median_sampson_error when available
analysis/original scale metadata
```

- [ ] **Step 3: Run diagnostics on representative adjacent or nearby pairs**

At minimum one pair per geometry category, chosen among nearby frames where overlap is expected. Examples may include:

```text
3 ↔ 10 or closer neighboring source frames around index 3
90 ↔ nearby low-angle frame
148 ↔ nearby elevated frame
206 ↔ nearby top-down frame
```

Do not force the exact pair if capture spacing is too wide; choose a nearby registered neighbor and document it.

- [ ] **Step 4: Save real match visualizations**

Output under:

```text
reconstruction/reference_assisted_v2/evidence/cv_matches/sift_*.png
```

Use real keypoints/inliers, not drawn placeholders.

---

### Task 6: Add ALIKED/LightGlue fallback only for difficult detail alignment

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`
- Read-only dependency: `external_learned_recovery.py`

**Interfaces:**
- Produces: `align_detail_pair()` with SIFT-first policy.

- [ ] **Step 1: Define a bounded alignment decision contract**

```python
@dataclass(frozen=True)
class PairAlignmentDecision:
    method: str              # "sift" or "aliked_lightglue"
    match_count: int
    inlier_count: int
    homography: tuple[float, ...] | None
    reason: str
```

- [ ] **Step 2: Implement SIFT-first acceptance rule**

Accept SIFT for local patch alignment if all are true:

```text
candidate matches >= 30
RANSAC inliers >= 15
inlier ratio >= 0.35
homography/geometry finite
```

These are V2 patch-alignment thresholds, not changes to Step 6 historical metrics.

- [ ] **Step 3: If SIFT fails, reuse Step 13 learned frontend code**

Do not edit the frozen Step 13 config. Instantiate/use the same pinned ALIKED-N16Rot + LightGlue stack through existing helpers where practical.

Use only for selected local pairs, especially unregistered detail views 267/268/278/288 against nearby registered views.

- [ ] **Step 4: Save learned-match diagnostic**

Write raw and geometrically filtered correspondence counts and a real match image under `evidence/cv_matches/`.

No new mapping/reconstruction database is created.

---

### Task 7: Define and record object/component landmarks

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces: `Landmark`, `validate_landmark_set()`, `landmarks.json`.

- [ ] **Step 1: Implement normalized landmark schema**

Coordinates are source-image pixels plus normalized copy; never ambiguous resized pixels.

Required landmark names where visible:

```text
axis_top
axis_bottom
finial_top
finial_bottom
lid_max_left
lid_max_right
lid_lower_left
lid_lower_right
neck_top_left
neck_top_right
neck_base_left
neck_base_right
globe_max_left
globe_max_right
globe_top_transition
globe_bottom_transition
bowl_rim_left
bowl_rim_right
bowl_bottom_transition_left
bowl_bottom_transition_right
pedestal_waist_left
pedestal_waist_right
foot_left
foot_right
```

For top-down views additionally:

```text
bowl_rim_center
bowl_rim_major_axis_a
bowl_rim_major_axis_b
globe_center
```

- [ ] **Step 2: Create a deterministic annotation-seed procedure**

Use reviewed/full CNN mask and contour geometry to propose automatic seeds for axis, extrema, and major transitions. For example:

- axis from median row centers / PCA;
- left/right extrema from mask contour;
- component transition seeds from radius-vs-height derivative peaks;
- rim ellipse seed from Canny/contour/ellipse fitting on top-down references.

The implementation may visually correct landmark seeds because this is a reference annotation stage, but every manual correction must be labeled `annotation_method="cv_seeded_visually_reviewed"`, not CNN output.

- [ ] **Step 3: Save per-view landmark overlay images**

Durable output:

```text
reconstruction/reference_assisted_v2/evidence/annotations/<index>_landmarks.png
```

- [ ] **Step 4: Validate landmark consistency**

Fail if:

- required landmarks lie outside image bounds;
- left/right ordering reverses;
- vertical component ordering is impossible;
- axis endpoints are degenerate;
- declared high-confidence landmark is missing in a view where it should be visible.

---

### Task 8: Build component masks/regions without retraining the CNN

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces: component masks for `vessel`, `bowl_pedestal`, and optional `lid_finial` where separable.

- [ ] **Step 1: Define component masks as reviewed V2 annotations, not new model predictions**

Use the existing whole-object CNN/reviewed mask as the outer constraint.

Split using landmark transition heights and contour connectivity. The primary need is geometry evaluation, not a new segmentation benchmark.

- [ ] **Step 2: Implement mask split helper**

```python
def split_component_masks(
    whole_mask: np.ndarray,
    landmarks: Sequence[Landmark],
) -> dict[str, np.ndarray]: ...
```

For normal/elevated views, at minimum produce:

```text
main_vessel
receiving_bowl_pedestal
```

When lid/finial boundaries are confident, optionally produce:

```text
lid_finial
neck
globe
```

- [ ] **Step 3: Write component-mask overlays and hashes**

Save under `evidence/component_masks/` and record annotation method.

---

### Task 9: Load Step 13 cameras as V2 camera references

**Files:**
- Create: `final_cv_model_fit.py`
- Create: `tests/test_final_cv_model_fit.py`

**Interfaces:**
- Produces: `CameraReference`, `load_step13_cameras()`.

- [ ] **Step 1: Write tests against a tiny pyCOLMAP reconstruction fixture or mocked camera/image objects**

Exact dataclass:

```python
@dataclass(frozen=True)
class CameraReference:
    selected_index: int
    filename: str
    image_id: int
    camera_model: str
    camera_params: tuple[float, ...]
    cam_from_world_rotation_xyzw: tuple[float, float, float, float]
    cam_from_world_translation: tuple[float, float, float]
    image_size: tuple[int, int]
```

- [ ] **Step 2: Implement real model loading**

Open:

```text
reconstruction/external_learned_recovery/best/
```

Map image name to selected index using the selected manifest.

Assert for current expected model:

```text
camera_count = 1
camera_model = SIMPLE_RADIAL
registered images = 266
```

Do not hard-fail merely because future verified state contains a different registered count; fail only if it contradicts authoritative Step 13 report/model consistency.

- [ ] **Step 3: Save canonical camera evidence**

For each selected canonical geometry view, persist intrinsics/extrinsics in V2 report.

---

### Task 10: Fit a common rotational model coordinate system to the registered cameras/masks

**Files:**
- Modify: `final_cv_model_fit.py`
- Test: `tests/test_final_cv_model_fit.py`

**Interfaces:**
- Produces: `ModelAlignment`, `fit_model_alignment()`.

- [ ] **Step 1: Define alignment variables**

```python
@dataclass(frozen=True)
class ModelAlignment:
    scale: float
    rotation_matrix: tuple[tuple[float, float, float], ...]
    translation: tuple[float, float, float]
```

The model uses local Z as the vessel axis and normalized overall set height = `1.0`.

- [ ] **Step 2: Implement a silhouette/axis-based initial alignment**

Use canonical view axes and Step 13 cameras to find a similarity transform that makes projected model Z-axis align with observed image axes.

Do not use Step 13 sparse points to define detailed surface shape. Sparse point centroid/bounds may be used only as a coarse initialization if documented.

- [ ] **Step 3: Validate projection primitives**

Add unit tests for projecting 3D points through SIMPLE_RADIAL camera parameters using pyCOLMAP APIs where possible; avoid reimplementing distortion math if pyCOLMAP exposes a trustworthy projection path.

---

### Task 11: Fit independent component profiles from multi-view evidence

**Files:**
- Modify: `final_cv_model_fit.py`
- Test: `tests/test_final_cv_model_fit.py`

**Interfaces:**
- Produces: `FittedComponentProfile`, `FitResult`, `fit_vessel_model()`.

- [ ] **Step 1: Define profile control sections**

Do not use V1 literal radius arrays.

Use independent normalized control profiles:

```text
pedestal/foot
bowl outer wall + rim
bowl inner wall
vessel globe + lower seat
shoulder steps
neck outer profile
visible neck inner/opening profile
lid stepped cone
finial stem/ball envelope
```

Each profile uses monotonic Z control samples with radius bounds derived from measured landmark extrema.

- [ ] **Step 2: Write synthetic projection-fit tests**

Generate a known radial profile, project into synthetic cameras, fit from generated silhouettes, and assert radius/control recovery within tolerance.

This tests optimization math without depending on the real artifact.

- [ ] **Step 3: Implement objective function**

For each canonical view, combine:

```text
silhouette residual
landmark reprojection residual
axis residual
component-width residual
```

Suggested normalized objective weights:

```python
WEIGHTS = {
    "silhouette": 1.0,
    "landmark": 1.5,
    "axis": 0.5,
    "component_width": 1.0,
}
```

Weights may be adjusted once from measured residual scale, but do not perform blind sweeps.

- [ ] **Step 4: Optimize in deterministic bounded stages**

Stage order:

1. global similarity alignment;
2. bowl/pedestal profile;
3. globe/shoulder profile;
4. neck profile;
5. lid/finial profile;
6. joint refinement of all profile radii within tight bounds.

Use SciPy only if already installed. Otherwise implement deterministic coordinate refinement.

- [ ] **Step 5: Preserve per-view residuals**

Write for all 16 canonical views:

```text
silhouette IoU
landmark median error fraction
landmark max error fraction
component mask IoUs when available
camera source
fit inclusion/exclusion reason
```

---

### Task 12: Gate the CV fit and write final profile contract for Blender

**Files:**
- Modify: `final_cv_model_fit.py`
- Test: `tests/test_final_cv_model_fit.py`

**Interfaces:**
- Produces: `final_cv_fit.json`, `final_profiles.json`.

- [ ] **Step 1: Implement hard metric gate**

Pass only if:

```text
median whole-object silhouette IoU >= 0.90
minimum reliable-view silhouette IoU >= 0.84
median landmark error fraction <= 0.020
95th percentile landmark error fraction <= 0.040
```

- [ ] **Step 2: Add manual/visual veto field to report contract**

```json
{
  "metrics_passed": true,
  "visual_component_review": {
    "bowl": "pass",
    "globe": "pass",
    "neck": "pass",
    "lid": "pass",
    "finial": "pass"
  },
  "accepted": true
}
```

The execution agent must open the source/fit overlays and set the visual review based on evidence. A visible mismatch sets `accepted=false` even if numeric metrics pass.

- [ ] **Step 3: Write Blender-consumable `final_profiles.json`**

Required fields:

```json
{
  "coordinate_contract": {
    "axis": "+Z",
    "normalized_set_height": 1.0,
    "scale_status": "relative_no_physical_measurement"
  },
  "profiles": {
    "pedestal": [[0.0, 0.0]],
    "bowl_outer": [],
    "bowl_inner": [],
    "globe": [],
    "shoulder": [],
    "neck_outer": [],
    "neck_inner": [],
    "lid": [],
    "finial": []
  },
  "source_views": [],
  "fit_metrics": {}
}
```

Actual arrays come from the fit, not placeholders.

- [ ] **Step 4: Generate CV-fit diagnostic panels**

For each canonical view, save:

```text
source.png
mask.png
fit_silhouette.png
overlay.png
difference.png
landmarks.png
```

under `diagnostics/10_cv_reference_fit/<index>/`.

- [ ] **Step 5: Run verification for this plan**

Run:

```text
python -B -m pytest tests/test_final_model_io.py tests/test_final_reference_evidence.py tests/test_final_cv_model_fit.py -q
python -B -m py_compile final_model_io.py final_reference_evidence.py final_cv_model_fit.py
```

Then run the real analysis/CV-fit stages through `run_final_model.py` once that orchestrator exists in the next plan, or temporarily through narrowly scoped module entry points if needed.

**Plan completion gate:** Do not start Blender base modeling until `final_cv_fit.json` is accepted and the canonical diagnostic panels have been visually reviewed.
