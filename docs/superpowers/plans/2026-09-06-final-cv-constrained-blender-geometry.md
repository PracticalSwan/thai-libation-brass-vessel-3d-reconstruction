# Final CV-Constrained Blender Geometry Implementation Plan

> **For agentic workers:** Execute this checklist task-by-task using any relevant Blender/CV skills/plugins. GLM-only subagents may be used for independent source-vs-model reviews; the parent agent must verify them. If an installed plan-execution workflow is available, use it; otherwise follow this file directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new V2 Blender base model from the accepted CV-fitted profiles and Step 13 camera evidence, then prove the unornamented geometry matches the photographed object before any decorative/detail work.

**Architecture:** Blender is a consumer of `final_profiles.json`, canonical camera parameters, and source-image diagnostics produced by Plan 1. Primary shapes remain non-destructive and profile-driven until multi-view geometry acceptance. Camera/reference matching follows the installed Blender Director reference-image workflow.

**Tech Stack:** Blender 5.2 Python plus available Blender integration/plugins, V2 JSON reports, pyCOLMAP-derived camera transforms, Blender curves/Screw or deterministic lathe meshes, pytest only for non-Blender orchestration/contracts.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

## Global Constraints

- Do not import V1 geometry as a shape source.
- Do not import/repair the Step 17 Poisson mesh as V2 geometry.
- Do not add ornament, chain, surface noise, or beauty materials before base geometry passes.
- Preserve parametric source curves/profiles until base geometry is accepted.
- Use normalized scale; no physical metric claim.
- Use Blender MCP or an equivalent available Blender inspection/render path after every major shape/camera phase.
- Load and follow `blender-director`, `prop-artist`, `blender-modeler`, `realistic-style`, `camera-cinematography`, and `qa-review` guidance before relevant operations.
- The accepted metric thresholds are floors, not a signal to stop. Continue evidence-directed profile/camera correction while meaningful visible mismatch remains.
- Use substantial Blender/CV compute when it improves reference matching. Do not trade away visual identity merely to save execution time.

---

## Files

**Create:**

```text
build_final_model_blender.py
run_final_model.py
tests/test_run_final_model.py
```

**Consume:**

```text
reconstruction/reference_assisted_v2/reports/final_profiles.json
reconstruction/reference_assisted_v2/reports/final_reference_evidence.json
reconstruction/reference_assisted_v2/reports/final_cv_fit.json
reconstruction/reference_assisted_v2/evidence/**
```

**Produce:**

```text
reconstruction/reference_assisted_v2/work/base_geometry.blend
reconstruction/reference_assisted_v2/reports/base_geometry_report.json
reconstruction/reference_assisted_v2/diagnostics/20_base_geometry/**
```

---

### Task 1: Add the restartable V2 orchestrator skeleton

**Files:**
- Create: `run_final_model.py`
- Create: `tests/test_run_final_model.py`

**Interfaces:**
- Consumes Plan 1 modules.
- Produces stages `analyze`, `cv-fit`, `base`, `geometry-validate`, later extended by later plans.

- [ ] **Step 1: Write failing parser/stage-order tests**

```python
from run_final_model import STAGE_ORDER


def test_final_stage_order_starts_with_cv_before_blender():
    assert STAGE_ORDER[:4] == (
        "analyze",
        "cv-fit",
        "base",
        "geometry-validate",
    )
```

Also test that `base` refuses to run if `final_cv_fit.json` is absent or `accepted != true`.

- [ ] **Step 2: Implement stage dispatch**

Required public entry points:

```python
STAGE_ORDER = (
    "analyze",
    "cv-fit",
    "base",
    "geometry-validate",
    "ornament",
    "cleanup",
    "uv-bake",
    "lookdev",
    "final-validate",
    "export",
)


def run_stage(stage: str) -> dict: ...
def run_all_through(stage: str | None = None) -> dict: ...
```

At this plan, unimplemented later stages must fail explicitly with `NotImplementedError` rather than silently succeed.

- [ ] **Step 3: Add Blender process runner**

Use current verified Blender 5.2 executable discovery pattern. Invoke Blender with:

```text
--background
--python-exit-code 2
--python build_final_model_blender.py
-- <stage> <v2_root>
```

Use `stdin=DEVNULL`, durable log under V2 `work/logs/`, and fail on non-zero exit.

- [ ] **Step 4: Run parser/orchestrator tests**

---

### Task 2: Create Blender scene hierarchy and immutable rollback baseline

**Files:**
- Create: `build_final_model_blender.py`

**Interfaces:**
- Produces `setup_v2_scene()` and the base `.blend` rollback structure.

- [ ] **Step 1: Define exact collection names**

```python
COLLECTIONS = (
    "COL_FINAL_V2",
    "COL_REFERENCE",
    "COL_BLOCKOUT",
    "COL_HIGH",
    "COL_LOW",
    "COL_ORNAMENT_HIGH",
    "COL_DIAGNOSTICS",
    "COL_CAMERAS",
    "COL_LIGHTS_NEUTRAL",
    "COL_LIGHTS_BEAUTY",
    "COL_EXPORT",
)
```

- [ ] **Step 2: Implement scene setup with normalized unit contract**

Set:

```text
unit_system = METRIC
scale_length = 1.0
```

but record on `COL_FINAL_V2`:

```text
scale_status = relative_no_physical_measurement
normalized_set_height = 1.0
```

Do not tell documentation that one Blender unit equals one real meter for this asset.

- [ ] **Step 3: Create root empty and custom provenance fields**

Object:

```text
ROOT_ThaiLibationV2
```

Custom properties:

```text
method = cv_constrained_reference_reconstruction
source_profiles = reconstruction/reference_assisted_v2/reports/final_profiles.json
manual_visual_finish = true_after_cv_geometry_gate
```

- [ ] **Step 4: Save an empty/setup rollback blend**

```text
reconstruction/reference_assisted_v2/work/00_scene_setup.blend
```

Do not overwrite this rollback file later.

---

### Task 3: Load canonical source images and create camera/reference objects

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `CAM_REF_<index>` cameras and `REF_<index>` image reference planes/empties.

- [ ] **Step 1: Implement camera conversion helper**

Signature:

```python
def create_camera_from_cv_reference(
    scene: bpy.types.Scene,
    record: dict,
    model_alignment: dict,
) -> bpy.types.Object: ...
```

Use Step 13 camera intrinsics/extrinsics from V2 report. Preserve SIMPLE_RADIAL evidence in custom properties even if Blender camera approximates distortion-free perspective for viewport review.

If lens distortion must be handled for exact overlays, prefer creating undistorted derived reference images/cameras under V2 evidence rather than pretending Blender's standard perspective includes SIMPLE_RADIAL distortion.

- [ ] **Step 2: Create derived undistorted review images if required**

The source image remains untouched. Derived undistorted images go under:

```text
reconstruction/reference_assisted_v2/evidence/aligned_views/
```

Record exact source image and transformation parameters.

- [ ] **Step 3: Add reference images to `COL_REFERENCE`**

Each object/empty includes:

```text
selected_index
source_filename
view_category
step13_registered
source_sha256
```

Set opacity only for Blender display. Never bake opacity/change into originals.

- [ ] **Step 4: Capture a baseline reference scene screenshot**

Use the available Blender screenshot/inspection interface (Blender MCP when available) with no V2 geometry yet. Save/copy the durable capture as:

```text
diagnostics/20_base_geometry/00_reference_scene.png
```

Verify camera/source orientation before modeling.

---

### Task 4: Build the receiving pedestal and bowl from CV-fitted profiles

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `CRV_PedestalProfile`, `CRV_BowlOuterProfile`, `CRV_BowlInnerProfile`, `SM_Pedestal`, `SM_Bowl`.

- [ ] **Step 1: Read exact profiles from `final_profiles.json`**

Fail if profile is absent, non-monotonic in Z where required, contains negative radii, or fit report is not accepted.

- [ ] **Step 2: Create editable profile curves**

Use one curve object per profile. Points are direct CV-fit values. Do not eyeball an alternate shape at creation time.

- [ ] **Step 3: Revolve pedestal outer profile**

Preferred Blender strategy:

1. profile curve/mesh in XZ plane;
2. Screw modifier around global Z;
3. 128-192 steps for hero smoothness;
4. merge axis only where profile intentionally reaches axis;
5. keep modifier live.

- [ ] **Step 4: Build bowl wall as a physical shell**

Use outer and inner CV profiles to create actual wall/rim thickness. The inside must be visible from top-down source cameras.

Do **not** use a black cavity disk like V1.

- [ ] **Step 5: Model the rolled/raised rim explicitly from profile evidence**

Use bevel/profile rings, not a detached oversized torus unless the photo truly shows a separate toroidal bead.

- [ ] **Step 6: Capture diagnostics**

Save:

```text
20_base_geometry/10_bowl_pedestal_clay.png
20_base_geometry/11_bowl_pedestal_wire.png
20_base_geometry/12_bowl_pedestal_low_angle_overlay.png
20_base_geometry/13_bowl_pedestal_top_overlay.png
```

Use a neutral clay material. No shiny brass yet.

- [ ] **Step 7: Fix only measured mismatches**

If bowl rim/waist/foot mismatch appears:

1. diagnose which CV profile/control point is wrong;
2. correct the fit/profile source or apply a documented bounded control-point correction tied to landmarks;
3. re-render overlay;
4. do not sculpt the bowl silhouette into shape.

---

### Task 5: Build the main globe and shoulder steps

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `CRV_GlobeProfile`, `CRV_ShoulderProfile`, `SM_VesselBody`.

- [ ] **Step 1: Build a nearly spherical fitted globe**

Use the actual V2 globe profile. The source photos show a rounded/flattened globe with a strong horizontal band/seam, not V1's stretched pear envelope.

- [ ] **Step 2: Build lower vessel seat where visible inside receiving bowl**

Top-down/elevated views show how the vessel sits inside the bowl. Include the visible base/stem geometry only where source evidence supports it.

- [ ] **Step 3: Add the horizontal globe seam/band as construction geometry**

The seam is integrated around the globe; use a shallow profile ridge/groove or high-poly detail, not a floating decoration object.

- [ ] **Step 4: Build the concentric shoulder steps**

Use explicit shallow revolution rings/steps based on photographs 148/165/267/268-class views. Maintain continuity into the neck base.

- [ ] **Step 5: Capture source-matched clay overlays**

At minimum:

```text
20_base_geometry/20_globe_front_overlay.png
20_base_geometry/21_globe_elevated_overlay.png
20_base_geometry/22_globe_top_overlay.png
20_base_geometry/23_globe_wire.png
```

- [ ] **Step 6: Correct profile, not ornament/material, when shape is wrong**

Do not let later relief hide a wrong globe.

---

### Task 6: Build the long tapered neck and visible interior/opening

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `CRV_NeckOuterProfile`, `CRV_NeckInnerProfile`, `SM_VesselNeck`.

- [ ] **Step 1: Revolve the fitted neck profile**

Source expectation:

- long and slender;
- slight taper;
- small rounded lower collar;
- decorated upper collar/band area;
- no V1 bulbous or generic cylinder proportions.

- [ ] **Step 2: Build wall thickness/opening**

If the lid geometry closes over an opening, model the lip/seat and inner wall sufficiently for top/close views. Do not create impossible zero-thickness surfaces.

- [ ] **Step 3: Keep upper decorative collar as macro geometry only**

At this phase create only measured band/ring depths. Ornament pattern is deferred.

- [ ] **Step 4: Capture neck diagnostics**

```text
20_base_geometry/30_neck_side_overlay.png
20_base_geometry/31_neck_elevated_overlay.png
20_base_geometry/32_neck_detail_clay.png
20_base_geometry/33_neck_wire.png
```

- [ ] **Step 5: Compare neck length-to-globe diameter explicitly**

Record ratio in `base_geometry_report.json` and compare to CV-fit report. Fail if Blender geometry drifts outside `0.5%` normalized difference from fitted control dimensions before manual sculpting is allowed.

---

### Task 7: Build the stepped conical lid and ball finial exactly from source construction

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `SM_Lid`, `SM_Finial`.

- [ ] **Step 1: Build lid base/seat**

Match the lower lip visible above the neck. Keep as a coherent removable part if source construction suggests separation.

- [ ] **Step 2: Build stepped cone tiers**

The photos show multiple thin progressively smaller rings/tier steps. Use one revolved profile with explicit tier geometry where possible rather than stacking unrelated torus objects.

- [ ] **Step 3: Build finial stem and ball**

Use a UV sphere or revolved sphere-like profile for the ball with a short neck/stem. Match ball size to lid width from landmarks.

- [ ] **Step 4: Capture close diagnostic views**

```text
20_base_geometry/40_lid_finial_front.png
20_base_geometry/41_lid_finial_side.png
20_base_geometry/42_lid_finial_wire.png
20_base_geometry/43_lid_source_overlay.png
```

- [ ] **Step 5: Reject generic pagoda-like exaggeration**

If tier spacing/height looks stylized compared with the source, return to fitted profile and correct it before proceeding.

---

### Task 8: Assemble parts and establish actual relative placement

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces one coherent V2 assembly under `ROOT_ThaiLibationV2`.

- [ ] **Step 1: Set origins logically**

```text
SM_Pedestal / SM_Bowl    origin on global center axis
SM_VesselBody            origin on vessel axis
SM_VesselNeck            aligned with body axis
SM_Lid                   origin on lid rotation/seat axis
SM_Finial                aligned to lid axis
```

- [ ] **Step 2: Place vessel into bowl according to multi-view top/elevated evidence**

Use CV fitted Z positions and top-down centering. Do not place by visual convenience alone.

- [ ] **Step 3: Check collision/intersection intentionally**

The vessel may physically sit/contact inside the bowl. Distinguish intentional contact/interpenetration from accidental geometry overlap.

- [ ] **Step 4: Save pre-validation rollback**

```text
reconstruction/reference_assisted_v2/work/20_base_geometry_pre_validation.blend
```

---

### Task 9: Implement Blender render masks and camera-matched geometry diagnostics

**Files:**
- Modify: `build_final_model_blender.py`
- Create: `final_model_validation.py` if absent; otherwise modify it
- Create: `tests/test_final_model_validation.py` if absent; otherwise modify it
- Modify: `run_final_model.py`

**Interfaces:**
- Produces `geometry-validate` stage artifacts.

- [ ] **Step 1: Render neutral clay source-camera views**

For all 16 canonical cameras render:

```text
clay.png
silhouette.png
wire_overlay.png
```

Use transparent silhouette pass for exact mask comparison.

- [ ] **Step 2: Generate source/render panels in regular Python**

For each view write:

```text
source.png
render_clay.png
overlay.png
difference.png
landmarks.png
```

under:

```text
diagnostics/20_base_geometry/<index>/
```

- [ ] **Step 3: Compute metrics against reviewed masks and component masks**

Use Plan 1 validation functions. Do not compare against a single aggregate silhouette.

- [ ] **Step 4: Open at least one panel from each category plus worst metric view**

The implementation agent must inspect real images. Numeric-only acceptance is forbidden.

---

### Task 10: Run bounded geometry correction loop

**Files:**
- Modify: `build_final_model_blender.py`
- Modify if justified: `final_cv_model_fit.py`

**Interfaces:**
- Produces final accepted base geometry.

- [ ] **Step 1: Sort failures by category**

Priority:

```text
1. camera/alignment
2. global silhouette
3. bowl/pedestal profile
4. globe/shoulder
5. neck
6. lid/finial
7. only then small construction steps
```

- [ ] **Step 2: Diagnose the cause before editing**

For each failed view, decide whether the mismatch comes from:

```text
camera transform
lens/undistortion mismatch
CV profile/control fit
Blender profile import
assembly placement
```

Do not fix camera errors by deforming the model.

- [ ] **Step 3: Apply one major variable change per iteration**

Follow installed look/reference workflow discipline. Re-render after each correction batch.

Maximum three broad correction iterations per component before stopping to inspect the underlying CV measurement logic. This prevents endless eyeballing.

- [ ] **Step 4: Persist before/after images for each major fix**

Example report entry:

```json
{
  "defect": "neck too wide in canonical side views",
  "diagnosis": "imported radius used whole-neck max instead of fitted taper samples",
  "before": ".../before.png",
  "fix": "restore per-z radius samples from final_profiles.json",
  "after": ".../after.png",
  "metric_before": 0.81,
  "metric_after": 0.93
}
```

---

### Task 11: Apply the hard geometry acceptance gate

**Files:**
- Modify: `final_model_validation.py`
- Modify: `run_final_model.py`

- [ ] **Step 1: Enforce metric thresholds**

```text
median whole-object silhouette IoU >= 0.90
minimum reliable-view silhouette IoU >= 0.84
median landmark error fraction <= 0.020
95th-percentile landmark error fraction <= 0.040
```

- [ ] **Step 2: Add component visual verdicts**

Required fields:

```text
bowl_pedestal = pass/fail
globe_shoulder = pass/fail
neck = pass/fail
lid_finial = pass/fail
overall_identity = pass/fail
```

Use screenshot comparison. A visible fail vetoes numeric metrics.

- [ ] **Step 3: Save accepted base blend**

Only after pass:

```text
reconstruction/reference_assisted_v2/work/30_BASE_GEOMETRY_ACCEPTED.blend
```

This becomes the immutable rollback source for ornament work.

- [ ] **Step 4: Write `base_geometry_report.json`**

Include object list, profiles consumed, dimensions in normalized units, camera set, metrics, diagnostic paths, and `ornament_started=false` at this checkpoint.

- [ ] **Step 5: Verify this plan**

Run focused orchestrator/validation tests and compile new Python modules. Reopen accepted blend in background Blender and confirm required objects/collections exist.

**Plan completion gate:** No ornament, sculpted wear, final UVs, or beauty materials begin until the accepted base blend and geometry report exist.
