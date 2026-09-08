# Final High-Fidelity Thai Libation Vessel Design

**Date:** 2026-09-06

**Status:** V2 Plan 1 is implemented and accepted. The current candidate passes Gate B with median silhouette IoU 0.901495, minimum reliable-view IoU 0.856102, median landmark error 0.011306 object height, p95 landmark error 0.038080, and the exact candidate-bound bowl/globe/neck/lid/finial visual review passes. Steps 1-17 and rejected V1 remain protected. The next execution window starts Blender immediately and continues through construction, ornament/detail, cleanup/sculpt/retopology, UV/bakes, texture/material/lookdev, and final Blender validation. Export/re-import/publication are explicitly deferred until the user inspects the complete Blender model and approves export.

## Goal

Produce a complete, polished, working Blender model that is recognizably the same physical Thai libation brass vessel set shown in the project's photographs **and remains meaningfully rooted in the CSX4213 computer-vision workflow**.

The final asset is not allowed to become an unrelated hand-modeled art exercise. Computer vision must determine or constrain the camera geometry, object masks, component proportions, silhouette/profile measurements, view correspondence, ornament references, photo projection, and final validation. Blender is the reconstruction, cleanup, baking, look-development, and delivery environment that consumes those CV-derived measurements.

## Execution and quality policy

- The design is harness-agnostic. The executor should use its own native tools plus any relevant installed skills/plugins that materially improve the work rather than depending on a particular orchestration wrapper.
- Subagents are optional, but if they are used they must be GLM variants only. Non-GLM subagents are not an allowed fallback. The parent agent remains responsible for verifying their claims from repository/runtime evidence.
- Quality and same-object fidelity outrank speed. Use substantial available CPU, GPU, VRAM, RAM, storage, Blender render time, CV optimization, source-image inspection, and GLM review when they can materially improve the result.
- Avoid blind brute-force sweeps. Spend compute on evidence-directed fitting, rendering, validation, texture projection, baking, and defect correction.
- `NO-SHIP` is an intermediate diagnosis, not an acceptable final delivery. When a gate fails, return to the responsible stage, diagnose the root cause, change the method when necessary, and continue. Stop only for a genuine external blocker that cannot be overcome with the available evidence/tools/resources; never fabricate a pass.
- Numerical thresholds in this design are minimum gates, not quality targets. Continue improving beyond them when the source evidence and available compute support a materially better same-object match.

## Why V2 is required

The current `reconstruction/reference_assisted/` V1 asset is technically clean but visually rejected by the user. The main causes are visible in the implementation:

- `reference_assisted_model.py` uses eight whole-object masks but converts them into only one aggregate width/height measurement;
- `model_profiles()` then scales a hand-authored template rather than recovering the bowl, globe, neck, lid, and finial profiles independently;
- `build_reference_model_blender.py` adds generic spheres/diamonds as decoration and an inferred chain;
- the current acceptance criterion is a single normalized aggregate silhouette IoU with a low `0.70` threshold;
- no independent camera-matched per-view validation is required;
- no component-level landmark reprojection is required;
- the ornament does not come from actual motif crops;
- the material is mainly a procedural dark-gold shader rather than a photograph-derived multi-view appearance reconstruction.

V1 remains preserved as a failed presentation attempt. V2 is a separate reconstruction.

## Source-photo observations driving V2

Planning inspection of source-photo previews derived from the immutable originals confirms that the photographed object has a distinctive construction that V1 does not reproduce well.

Representative originals inspected:

```text
IMG20260826122956.jpg
IMG20260826123820.jpg
IMG20260826124024.jpg
IMG20260826124908.jpg
IMG20260826124927.jpg
IMG20260826125013.jpg
IMG20260826125527.jpg
IMG20260826130051.jpg
IMG20260826130313.jpg
IMG20260826130322.jpg
IMG20260826130427.jpg
```

The physical set contains:

### Receiving bowl / pedestal

- broad circular receiving bowl;
- strong rolled/raised rim;
- smooth flared bowl wall;
- repeated flame/lotus-like ornament around the lower outer bowl;
- narrow waist/transition below the bowl;
- flared pedestal with multiple raised rings;
- broad circular bottom foot.

### Main water vessel

- near-spherical or slightly flattened globe rather than V1's generic pear shape;
- horizontal seam/band around the globe;
- shallow concentric shoulder steps leading into the neck;
- long slender neck, slightly tapered rather than a plain cylinder;
- decorated neck band/collar near the top;
- stepped conical lid made from multiple narrow horizontal tiers;
- ball finial on a short stem;
- fine cross-hatched/pebbled field texture;
- repeated Thai/floral/lotus/scroll engraving and relief integrated into the surface;
- localized darker recesses, seams, wear, and reflective variation.

A large hanging chain is **not supported by the inspected reference photographs**. V2 therefore has no chain by default. A chain can be introduced only if implementation finds clear evidence in at least two independent source photographs or another authoritative project artifact that establishes its location and attachment.

Likewise, V1's floating generic oval and diamond studs are rejected. V2 ornament must be traceable to real motif crops from the source images.

## Computer-vision requirement

This phase is part of the Computer Vision coursework. At least these existing project results must materially contribute to V2:

1. **Step 6 SIFT + RANSAC + epipolar geometry**
   - reuse `geometry_detection.extract_sift`, `match_sift`, and `estimate_fundamental_geometry` for local photo correspondence, reference-pair quality, patch alignment, and/or diagnostic geometry;
   - preserve explicit analysis/original pixel coordinate mappings.

2. **Steps 7-9 CNN segmentation**
   - reuse the trained `SmallSegCNN` and existing full-sequence predictions as a primary object-mask source;
   - use the 36 reviewed masks as higher-confidence validation/reference masks;
   - weak CNN cases remain visible and cannot be silently relabeled as CNN output.

3. **Step 10 / Step 13 camera geometry**
   - use the 266-image ALIKED-N16Rot + LightGlue model under `reconstruction/external_learned_recovery/best/` as the preferred camera-pose/intrinsics reference for V2 views that are registered;
   - do not use its sparse points as final surface geometry;
   - use its one shared `SIMPLE_RADIAL` camera starting parameters when applicable;
   - retain Step 10 local model only as secondary comparison evidence.

4. **Step 13 learned feature matching**
   - reuse the pinned ALIKED + LightGlue stack when SIFT cannot align reflective-detail image patches reliably;
   - feature correspondences may support ornament patch rectification, view-to-view registration, or camera refinement;
   - do not rerun global sparse recovery.

5. **New V2 multi-view model fitting**
   - fit a parametric rotational model against multiple CNN/reviewed silhouettes and camera poses;
   - optimize the global model alignment plus component profile parameters against image evidence;
   - save per-view silhouette and landmark residuals.

6. **New V2 multi-view texture projection**
   - project selected source photographs onto the V2 surface using known/refined cameras;
   - estimate bounded overlap-based photometric normalization from trustworthy corresponding regions so exposure/white-balance differences do not become texture seams; reject clipped/specular samples from that solve and disclose that the result is relative harmonization, not color-calibrated reflectance;
   - weight contributions by view angle, visibility, mask confidence, clipping/saturation, and consistency;
   - use robust median/weighted fusion to reduce transient polished-brass specular highlights;
   - derive object-specific base-color/roughness/detail masks from the photos where defensible.

The final report must be able to explain the pipeline as:

```text
smartphone photographs
→ conservative preprocessing
→ SIFT/RANSAC geometry
→ CNN vessel segmentation
→ ALIKED/LightGlue + SfM camera recovery
→ CV-selected reference views
→ multi-view silhouette/profile fitting
→ Blender parametric reconstruction
→ distortion-consistent registered-view coverage audit
→ CV-aligned ornament/image patch extraction
→ Blender high/low detail + UV/bakes
→ multi-view photo texture projection
→ reference-camera + non-canonical render validation
→ final Blender/GLB presentation asset
```

If the final asset can be reproduced without any of the CV evidence above, the implementation has failed this design.

## CV continuity and full-surface evidence coverage

Plan 1's accepted 16-view fit is the primary macro-geometry constraint, but V2 must not overfit only those cameras. Downstream Blender validation therefore adds a secondary **registered-view coverage audit** over Step 13-registered, non-canonical views using the existing Step 9/CNN masks (reviewed masks where available). This audit is a generalization/cross-check workload, not a second optimization set: render low-cost binary silhouettes, compute per-view overlap diagnostics, inspect the worst views, and classify each failure as `mask_failure`, `camera_failure`, or `model_mismatch`. A real structural `model_mismatch` routes back to the responsible geometry stage; noisy CNN masks are documented rather than used to deform the model.

All camera-metric paths must be distortion-consistent. Raw source photographs use the frozen `SIMPLE_RADIAL` camera model through pyCOLMAP/exact verified distortion math. Blender pinhole cameras may be used for viewport/reference display only when paired with explicitly generated undistorted images and the corresponding derived camera. Never compare a pinhole render directly to a raw distorted source and call the result an exact CV metric.

The Blender reconstruction must also maintain a **surface-evidence coverage manifest**. For each major component and meaningful azimuth/elevation region, record whether final geometry/appearance is:

```text
direct_multi_view          supported by registered source views
reviewed_single_or_detail  supported by a reviewed/detail source but not multi-view camera coverage
symmetry_repetition        inferred from established rotational/manufactured repetition
hidden_generic_fill        physically plausible completion with no direct source support
```

This manifest constrains backside completion, ornament replication, texture filling, and final reporting. It does not forbid completing unseen surfaces; it makes the completion method explicit and prevents inferred regions from being presented as directly reconstructed.

As an independent classical-CV check, reuse Step 6 Canny/contour/PCA/ellipse evidence where source edges are reliable to compare projected circular rims, feet, shoulder/lid rings, and the vessel axis. These measurements are diagnostic cross-checks only; they do not override stronger reviewed masks, landmarks, or accepted multi-view profile evidence.

Do not add a new NeRF/3DGS/neural reconstruction branch, retrain the frozen CNN, or reopen global sparse/dense recovery merely to finish the deadline model. The approved V2 architecture already contains the necessary CV stages; additional methods are justified only if a concrete downstream blocker proves the current evidence insufficient.

## Protected evidence boundary

Read-only protected evidence:

```text
IMG20260826122949/
preprocessing/pycolmap_input/images/
reconstruction/sparse/best/
reconstruction/bridging/
reconstruction/learned_recovery/
reconstruction/external_learned_recovery/
reconstruction/local_dense/
```

The Step 13 266-image model remains measured evidence and camera/reference data only. Sparse recovery is not reopened.

Current V1 source/artifacts also remain preserved until V2 passes:

```text
reference_assisted_model.py
build_reference_model_blender.py
run_reference_assisted_model.py
reconstruction/reference_assisted/
```

## V2 directory contract

Create V2 separately:

```text
reconstruction/reference_assisted_v2/
├── evidence/
│   ├── reference_boards/
│   ├── aligned_views/
│   ├── annotations/
│   ├── component_masks/
│   ├── cv_matches/
│   └── ornament_crops/
├── diagnostics/
│   ├── 00_v1_rejection/
│   ├── 10_cv_reference_fit/
│   ├── 20_base_geometry/
│   ├── 30_ornament/
│   ├── 40_topology_cleanup/
│   ├── 50_uv_bakes/
│   ├── 60_material_lookdev/
│   ├── 70_final_photo_match/
│   └── 80_export_reimport/
├── textures/
├── previews/
├── reports/
├── work/
└── final/
```

`work/` is transient/local-only. Durable reports, diagnostics, textures, previews, master `.blend`, and final `.glb` may be published if reasonable in size.

## Planned V2 source modules

Create focused modules:

```text
final_reference_evidence.py
final_cv_model_fit.py
final_model_validation.py
final_model_io.py
build_final_model_blender.py
run_final_model.py
```

### `final_reference_evidence.py`

Responsibilities:

- load the 36 reviewed masks and full CNN prediction set;
- verify source/mask hashes;
- read Step 13 registration/camera data;
- select canonical reference views with coverage and quality diversity;
- create contact/reference boards;
- store explicit landmark annotations and confidence;
- define component image regions/masks where needed;
- create SIFT/ALIKED match diagnostics for selected view pairs;
- create ornament crop manifests.

Primary interfaces:

```python
@dataclass(frozen=True)
class FinalReferenceView:
    selected_index: int
    filename: str
    view_category: str
    quality_condition: str
    source_sha256: str
    reviewed_mask_path: Path | None
    cnn_mask_path: Path
    step13_registered: bool
    step13_image_id: int | None

@dataclass(frozen=True)
class Landmark:
    view_index: int
    component: str
    name: str
    x: float
    y: float
    confidence: str

@dataclass(frozen=True)
class OrnamentPatch:
    family: str
    view_index: int
    bbox_xyxy: tuple[int, int, int, int]
    source_path: Path
    support_count: int


def build_final_reference_evidence(project_root: Path, output_root: Path) -> dict: ...
def build_reference_boards(project_root: Path, references: Sequence[FinalReferenceView], output_root: Path) -> dict[str, Path]: ...
def build_cv_pair_diagnostics(...) -> dict: ...
```

### `final_cv_model_fit.py`

Responsibilities:

- load camera intrinsics/extrinsics from Step 13 for registered views;
- choose registered canonical views preferentially;
- estimate/refine a common vessel axis and global similarity transform;
- define normalized parametric profiles for bowl, globe/shoulder, neck, lid, finial;
- render/project parametric silhouettes into each camera;
- optimize profile/control parameters against reviewed/CNN masks plus landmarks;
- write uncertainty/residual reports;
- output a deterministic `final_profiles.json` consumed by Blender.

Primary interfaces:

```python
@dataclass(frozen=True)
class CameraReference:
    selected_index: int
    camera_model: str
    params: tuple[float, ...]
    rotation: tuple[float, float, float, float]
    translation: tuple[float, float, float]

@dataclass(frozen=True)
class FittedComponentProfile:
    component: str
    points_rz: tuple[tuple[float, float], ...]
    source_views: tuple[int, ...]
    median_residual: float

@dataclass(frozen=True)
class FitResult:
    profiles: dict[str, FittedComponentProfile]
    world_from_model: tuple[tuple[float, ...], ...]
    per_view_iou: dict[int, float]
    per_view_landmark_error: dict[int, float]


def load_step13_cameras(project_root: Path) -> dict[int, CameraReference]: ...
def fit_vessel_model(project_root: Path, evidence_report: dict, output_root: Path) -> FitResult: ...
```

Use `scipy.optimize.least_squares` only if SciPy is already available in the environment. Otherwise use deterministic bounded coordinate/grid refinement with NumPy/OpenCV; do not add a dependency merely for convenience.

### `final_model_validation.py`

Responsibilities:

- compute whole-object and component silhouette IoU;
- compute landmark reprojection error normalized by object height;
- generate source/render/overlay/difference/landmark panels;
- validate Blender topology/UV/material reports;
- compare master vs re-imported GLB renders;
- aggregate the final QA gate.

### `final_model_io.py`

Responsibilities:

- safe JSON/CSV/image writes;
- SHA-256 manifests;
- V2 attempt ownership;
- final-promotion guard;
- cleanup allowlist for `work/` only.

### `build_final_model_blender.py`

Responsibilities:

- create reference collections and cameras from CV evidence;
- build measured rotational base geometry from `final_profiles.json`;
- create evidence-supported high-poly ornament;
- create clean presentation topology;
- perform bounded cleanup/sculpt/retopo steps from the Blender plans;
- UV unwrap and bake;
- build object-specific brass materials/textures;
- set neutral/beauty lookdev scenes;
- create diagnostic and final renders;
- save master and export-ready scene.

### `run_final_model.py`

Stages:

```text
analyze
cv-fit
base
geometry-validate
ornament
cleanup
uv-bake
lookdev
final-validate
export
all
```

Each stage is restartable and fail-closed. A failed geometry gate prevents ornament; a failed cleanup/UV gate prevents final texturing; a failed final visual gate prevents export promotion.

## Blender skill routing

Execution must load installed Codex-local Blender skills from `C:\Users\LOQ\.codex\skills` and follow them. The planning-selected route is:

```text
blender-director
→ reference-image-match workflow
→ prop-artist
→ blender-modeler
→ realistic-style
→ sculpting where evidence supports surface relief/wear
→ retopology
→ uv-workflow
→ texture-workflow
→ materials
→ lookdev
→ camera-cinematography
→ lighting
→ rendering
→ asset-optimization
→ export-pipeline
→ qa-review
```

Use hard-surface only for manufactured ring/rim/stepped geometry where useful. Do not introduce sci-fi panel language.

The Blender skills are fixed to the Codex-local skill path. Do not move or sync them to parent skill paths.

## Canonical reference-view policy

The 36 reviewed masks provide five view categories:

```text
normal_side          3, 10, 19, 28, 39, 50, 62, 72
low_angle_pedestal   76, 82, 90, 98, 106, 114, 128, 142
elevated_oblique     148, 151, 154, 165, 177, 180, 188, 200
top_down_rim         206, 209, 212, 221, 230, 233, 243, 255
oblique_detail       267, 268, 278, 288
```

Step 13 registration exists for the reviewed references through index 255, while 267/268/278/288 are not registered in the Step 13 global model. Therefore:

- use registered references as the primary geometry/camera fitting set;
- use 267/268/278/288 as unregistered high-resolution ornament/detail references;
- if an unregistered detail view must be geometrically aligned, match it to a registered neighbor using SIFT first and ALIKED/LightGlue only when SIFT quality is inadequate;
- do not rerun global mapping.

Minimum canonical geometry set should include at least 16 registered views with coverage such as:

```text
normal_side:        3, 19, 50, 72
low_angle:          90, 114, 128, 142
elevated_oblique:   148, 165, 188, 200
top_down:           206, 221, 243, 255
```

The implementation may replace a view when image quality or camera evidence makes it unsuitable, but replacement must stay in the same category and be documented.

## Geometry architecture

V2 macro geometry is a measured parametric reconstruction, not a sculpt-first model.

Blender collections:

```text
COL_FINAL_V2
├── COL_REFERENCE
├── COL_BLOCKOUT
├── COL_HIGH
├── COL_LOW
├── COL_ORNAMENT_HIGH
├── COL_DIAGNOSTICS
├── COL_CAMERAS
├── COL_LIGHTS_NEUTRAL
├── COL_LIGHTS_BEAUTY
└── COL_EXPORT
```

Logical components:

```text
SM_VesselBody
SM_VesselNeck
SM_Lid
SM_Finial
SM_Bowl
SM_Pedestal
```

The editable master keeps physical parts meaningful. Export may join only where appropriate.

Main rotational forms use a curve/Screw or equivalent revolution workflow based on **CV-fitted profiles**, not V1 arrays. Interior bowl and visible vessel opening/lid-seat geometry must be physically plausible.

## Ornament architecture

Source photographs show actual integrated engraving/relief. Use a documented ornament inventory before modeling.

Representation by scale:

- rolled rims / concentric rings: explicit geometry;
- large repeated Thai/floral/lotus relief: high-poly geometry or displacement source, then bake as appropriate;
- fine line engraving/scrollwork: curve/high source or height/normal texture;
- dense cross-hatch/pebble field: tileable or projected normal/roughness/height detail;
- local discoloration/wear: texture masks.

No generic floating decorative objects are allowed on the final low/presentation mesh unless they correspond to separate physical hardware.

## Material and texture architecture

Use a small physically plausible set:

```text
MAT_Brass_Main
MAT_Brass_Interior        # only if multi-view analysis proves stable difference
MAT_Brass_RecessDark
```

`MAT_Brass_Main` metallic = `1.0`. Roughness varies spatially. Source-photo color is estimated from multiple images while rejecting clipped highlights and deep shadows, but is not claimed colorimetrically calibrated because no color chart exists.

The final texture set is CV/photo driven:

```text
T_ThaiLibation_BaseColor.png
T_ThaiLibation_Roughness.png
T_ThaiLibation_Metallic.png
T_ThaiLibation_Normal.png
T_ThaiLibation_AO.png
T_ThaiLibation_Height.png      # optional if retained
```

Minimum final resolution is 4096 square. 8192 is allowed for the master if source detail and VRAM justify it. External CC0 metal micro-texture may supplement generic micro-normal/roughness only; it cannot replace object-specific ornament.

## Scale contract

No defensible physical measurement exists. During fitting:

```text
normalized overall set height = 1.0
```

Do not claim centimeters/millimeters. The master must record:

```text
scale_status = "relative_no_physical_measurement"
```

If a real measurement becomes available later, apply one documented uniform global scale.

## Required diagnostic image trail

Every major correction must have image-backed evidence:

```text
00_v1_rejection       V1 vs photographed object
10_cv_reference_fit   source views, masks, SIFT/learned correspondences, camera fit
20_base_geometry      clay/wireframe source overlays
30_ornament           source crop vs recreated/baked motif
40_topology_cleanup   solid, wireframe, face orientation, non-manifold, intersections
50_uv_bakes           checker, UV layout, normal/height/AO bake previews
60_material_lookdev   neutral and beauty material comparisons
70_final_photo_match  source/render/overlay/difference panels
80_export_reimport    master vs fresh GLB import
```

Each durable defect record includes:

1. before image;
2. diagnosis;
3. exact fix;
4. after image;
5. metric/check result.

## Acceptance gates

### Gate A — CV evidence gate

Pass only if:

- source hashes match protected manifests;
- CNN predictions are reused and distinguished from reviewed masks;
- at least one SIFT/RANSAC correspondence diagnostic is generated for each required local alignment task;
- Step 13 camera/intrinsic evidence is used for registered canonical views;
- the final CV evidence report names exactly how Step 6, CNN, and Step 13 contribute to V2.

### Gate B — base geometry gate

Use at least 16 canonical registered views covering four geometry categories.

Required metrics:

```text
median whole-object silhouette IoU       >= 0.90
minimum reliable-view silhouette IoU     >= 0.84
median landmark error / object height    <= 0.020
95th-percentile landmark error            <= 0.040
```

Component-level visual inspection has veto power. If bowl, globe, neck, lid, or finial is visibly wrong, the gate fails even if aggregate metrics pass.

### Gate C — ornament gate

- no unsupported V1 chain;
- no unsupported generic oval/diamond studs;
- every hero motif family has at least one source crop and preferably two-view support;
- repeated count/spacing derives from source evidence where observable;
- no visible floating ornament;
- detail reads at both hero close-up and normal presentation distance.

### Gate D — topology/cleanup gate

Final logical assemblies:

- no accidental loose geometry;
- no unintended non-manifold edges;
- no inverted normals;
- no duplicate faces/vertices after measured cleanup threshold;
- no visible self-intersections;
- no shading artifacts hiding topology defects;
- intentional separate physical parts documented.

### Gate E — UV/bake gate

- final production UVs use manual/logical seam strategy;
- no unintended unique-detail overlap;
- consistent visible texel density;
- pack efficiency target >= 75% unless a documented UDIM layout is better;
- 4K minimum final texture set;
- correct color-space assignments;
- normal map orientation verified;
- no visible bake seams in hero views.

### Gate F — lookdev gate

Under neutral diagnostic light:

- bright polished brass reads similarly to the photos;
- large areas do not collapse to dark brown/black;
- engraved recesses remain readable;
- large highlights do not clip away form;
- direct-photo lighting is not baked into base color.

Beauty lighting may improve presentation but cannot hide shape or texture defects.

### Gate G — final visual identity gate

Use installed Blender `visual-match-checklist` and `qa-review`.

Required:

- camera: `Match` or `Close`;
- silhouette/proportions: `Match` or `Close` with no `Miss` on canonical views;
- depth/construction: `Match` or `Close`;
- hero ornament: no `Miss`;
- materials: `Match` or `Close`;
- QA verdict: `SHIP`.

### Gate H — export/reimport gate

Final files:

```text
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.blend
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.glb
```

Fresh GLB import must preserve intended objects, material/textures, bounds, orientation, normals, and rendered appearance. A re-imported comparison render must pass the export plan's image-difference tolerance.

## Coursework truthfulness

Final documentation must distinguish:

```text
reconstruction/local_dense/              measured photogrammetry output
reconstruction/reference_assisted_v2/    CV-constrained + Blender-completed presentation reconstruction
```

The V2 model may contain inferred unseen ornament repetition, Blender-modeled surface completion, sculpted evidence-supported irregularity, and baked texture synthesis. These are explicitly disclosed. The model is not described as a direct complete COLMAP dense reconstruction.
