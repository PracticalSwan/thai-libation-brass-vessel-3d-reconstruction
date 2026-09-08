# Final Multi-View Texture, Material, Lookdev, and Rendering Implementation Plan

> **For agentic workers:** Execute this checklist task-by-task using any relevant CV/Blender/material/rendering skills/plugins. GLM-only subagents may independently review texture projection or source-vs-render appearance; verify their conclusions in the parent agent. If an installed plan-execution workflow is available, use it; otherwise follow this file directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build object-specific brass textures and materials from the project's photographs using the recovered cameras/CNN masks, then create neutral and beauty lookdev/render setups that match the photographed vessel without baking view-dependent lighting into albedo.

**Architecture:** Multi-view texture projection is a computer-vision stage, not a generic Blender shader replacement. Registered source photos are projected to the accepted V2 UV surface using Step 13 cameras; contributions are visibility/mask/view-angle/exposure weighted and robustly fused to suppress transient polished-metal highlights. Blender then combines these object-specific maps with baked ornament normals/AO and physically plausible brass shading for neutral and presentation rendering.

**Tech Stack:** Python/OpenCV/NumPy, pyCOLMAP camera evidence, CNN/reviewed masks, Blender 5.2, `materials`, `texture-workflow`, `realistic-style`, `lookdev`, `lighting`, `camera-cinematography`, `rendering`, `qa-review`.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

## Global Constraints

- Start only from `work/60_UV_BAKE_ACCEPTED.blend`.
- Use project photographs as the primary appearance source.
- Do not bake large photographed specular highlights into Base Color.
- Do not use V1 procedural material as the final appearance.
- External textures are optional secondary micro-detail only and require explicit license/provenance.
- Neutral diagnostic lighting must remain available even after beauty lighting is built.
- Beauty lighting may not hide shape/topology/texture defects.
- Final material must read as the bright polished gold/brass object in the photographs, not dark brown/black metal.
- Use substantial compute for depth/visibility maps, multi-view projection, robust fusion, high-resolution bakes/textures, and meaningful render iterations when it improves source fidelity.
- Material acceptance is visual as well as technical. If the final surface still reads as generic procedural brass rather than the photographed object, return to projection/fusion/detail reconstruction and continue.

---

## Files

**Create/Modify:**

```text
final_texture_projection.py
final_model_validation.py
build_final_model_blender.py
run_final_model.py
tests/test_final_texture_projection.py
tests/test_final_model_validation.py
```

**Produce:**

```text
reconstruction/reference_assisted_v2/textures/projected/**
reconstruction/reference_assisted_v2/textures/final/**
reconstruction/reference_assisted_v2/reports/texture_projection_report.json
reconstruction/reference_assisted_v2/reports/lookdev_report.json
reconstruction/reference_assisted_v2/diagnostics/60_material_lookdev/**
reconstruction/reference_assisted_v2/diagnostics/70_final_photo_match/**
reconstruction/reference_assisted_v2/previews/**
reconstruction/reference_assisted_v2/work/70_LOOKDEV_ACCEPTED.blend
```

---

### Task 1: Add deterministic multi-view texture projection domain contracts

**Files:**
- Create: `final_texture_projection.py`
- Create: `tests/test_final_texture_projection.py`

**Interfaces:**
- Produces `ProjectionView`, `ProjectionSample`, `ProjectionConfig`, weighting/fusion helpers.

- [ ] **Step 1: Write unit tests for clipping and weighting**

```python
from final_texture_projection import view_weight


def test_view_weight_rejects_saturated_highlight():
    assert view_weight(
        cos_view_angle=0.9,
        mask_confidence=1.0,
        luminance=0.995,
        saturation=0.05,
        visible=True,
    ) == 0.0


def test_view_weight_prefers_front_facing_visible_sample():
    front = view_weight(0.95, 1.0, 0.55, 0.7, True)
    grazing = view_weight(0.25, 1.0, 0.55, 0.7, True)
    assert front > grazing > 0
```

- [ ] **Step 2: Define exact dataclasses**

```python
@dataclass(frozen=True)
class ProjectionView:
    selected_index: int
    image_path: Path
    whole_mask_path: Path
    camera: CameraReference
    quality_condition: str
    source_sha256: str

@dataclass(frozen=True)
class ProjectionConfig:
    texture_size: int = 4096
    minimum_cos_view_angle: float = 0.20
    highlight_value_cutoff: float = 0.97
    deep_shadow_value_cutoff: float = 0.08
    minimum_mask_confidence: float = 0.5
    minimum_samples_per_texel: int = 2
```

- [ ] **Step 3: Implement weighting function**

Use multiplicative factors:

```text
visibility
whole-object/component mask confidence
view-angle cosine
quality-condition penalty
highlight/shadow rejection
optional reprojection confidence
```

Do not use camera distance alone as a quality proxy.

---

### Task 2: Export final low mesh, UV coordinates, normals, and camera-ready geometry for projection

**Files:**
- Modify: `build_final_model_blender.py`
- Modify: `final_texture_projection.py`

**Interfaces:**
- Produces deterministic mesh projection package.

- [ ] **Step 1: Export machine-readable geometry from accepted UV blend**

Prefer a compact NPZ/PLY + JSON package under V2 `work/` containing:

```text
world-space vertex positions
triangle indices
per-loop/per-corner UVs
vertex/loop normals
material/component IDs
object transforms already resolved
```

- [ ] **Step 2: Validate UV-to-triangle correspondence**

Write a unit test for barycentric UV interpolation on a synthetic triangle.

- [ ] **Step 3: Record exact accepted blend/hash source**

Projection report must point to `60_UV_BAKE_ACCEPTED.blend` hash.

---

### Task 3: Select the registered multi-view texture source set using CV evidence

**Files:**
- Modify: `final_texture_projection.py`

**Interfaces:**
- Produces `select_projection_views()`.

- [ ] **Step 1: Start from Step 13-registered selected images only**

The 266-image model provides camera poses. Do not project unregistered detail photos as if their cameras were known.

- [ ] **Step 2: Use the existing CNN/reconstruction masks**

For each registered image, require a matching Step 9 reconstruction mask or full prediction. Reviewed masks take precedence for the 36 reviewed indices when available.

- [ ] **Step 3: Downrank known weak/poor views rather than deleting capture coverage blindly**

Examples:

```text
known yellow-wall false-positive index 72: use reviewed mask, not raw CNN prediction
bright_clipping / high_brightness: stronger highlight rejection penalty
low_sharpness_low_features: lower fine-detail contribution
```

- [ ] **Step 4: Select a bounded high-quality projection set**

Do not automatically use all 266 for every texel. Build a coverage set that spans azimuth/elevation while prioritizing normal-quality and front-facing views.

Suggested target:

```text
24-64 projection views
```

The exact count should be determined from camera coverage and image quality, not tuned for an arbitrary number.

- [ ] **Step 5: Save projection-view manifest**

Include filename/index, camera id, quality condition, mask source, view category when reviewed, and inclusion reason.

---

### Task 4: Project source images to UV texels with visibility checks

**Files:**
- Modify: `final_texture_projection.py`
- Test: `tests/test_final_texture_projection.py`

- [ ] **Step 1: Implement triangle/texel rasterization or use an existing project dependency**

Do not add a large rendering dependency if NumPy/OpenCV plus a deterministic triangle rasterizer is sufficient.

For each final triangle:

1. rasterize its UV footprint;
2. compute corresponding 3D surface point/normal by barycentric interpolation;
3. project point into each candidate camera;
4. evaluate mask containment;
5. run visibility/occlusion check;
6. sample source color;
7. compute contribution weight.

- [ ] **Step 2: Implement visibility**

Preferred approaches, in order:

1. render depth map from each Step 13 camera using Blender/headless V2 geometry and compare projected depth;
2. CPU BVH ray test if a depth pipeline is simpler;
3. never omit visibility entirely.

Store depth maps under transient `work/projection_depth/` unless needed as durable diagnostics.

- [ ] **Step 3: Respect SIMPLE_RADIAL camera model**

Project through pyCOLMAP camera APIs or exact verified distortion math. Do not assume a pinhole camera when sampling raw distorted source photos.

If projection uses undistorted derived source images, record and use the corresponding undistorted camera model consistently.

- [ ] **Step 4: Write coverage diagnostics**

Produce:

```text
60_material_lookdev/00_projection_sample_count.png
60_material_lookdev/01_projection_weight_sum.png
60_material_lookdev/02_projection_best_view_index.png
```

---

### Task 5: Robustly fuse multi-view brass color while suppressing transient reflections

**Files:**
- Modify: `final_texture_projection.py`
- Test: `tests/test_final_texture_projection.py`

- [ ] **Step 1: Implement weighted robust fusion**

For each texel with accepted samples:

1. reject invalid/highlight/deep-shadow samples using HSV/value and local consistency;
2. compute weighted median per channel or weighted trimmed mean;
3. optionally compare chromaticity rather than raw luminance for polished metal;
4. retain sample-count/confidence map.

- [ ] **Step 2: Do not over-neutralize real dark engraving**

A dark value repeated at the same surface location across several geometrically consistent views is likely object detail/recess and should remain. A bright streak that moves with viewpoint is likely specular and should be suppressed from base color.

- [ ] **Step 3: Write raw and fused base-color maps**

```text
textures/projected/T_ThaiLibation_BaseColor_Raw.png
textures/projected/T_ThaiLibation_BaseColor_Fused.png
textures/projected/T_ThaiLibation_ProjectionConfidence.png
```

- [ ] **Step 4: Save hero-region source/fusion comparison panels**

At least:

```text
globe hero motif
neck motif
bowl flame band
lid band
```

---

### Task 6: Derive object-specific roughness/wear masks from multi-view evidence

**Files:**
- Modify: `final_texture_projection.py`

- [ ] **Step 1: Treat roughness as inferred, not directly measured**

No polarization/light-stage data exists. Roughness is estimated from repeatable spatial variation and photographed highlight spread.

- [ ] **Step 2: Build relative roughness cues**

Use:

```text
engraved/cross-hatch/recess masks from ornament plan
multi-view local highlight variance
curvature/AO bake
stable dark/recess regions
```

- [ ] **Step 3: Map to physically plausible polished-brass range**

Suggested final range:

```text
polished raised/ring areas: roughness ~0.14-0.25
engraved/textured fields:   roughness ~0.24-0.42
recess/darker worn areas:   roughness ~0.30-0.48
```

These are starting bounds. Adjust from neutral source/render comparison, not arbitrary artistic taste.

- [ ] **Step 4: Metallic map**

Default metal surface is `1.0`. Only deviate if source evidence indicates non-metal inserts/coatings.

---

### Task 7: Fill unsupported texture texels conservatively

**Files:**
- Modify: `final_texture_projection.py`

- [ ] **Step 1: Identify low-confidence/zero-sample UV texels**

Use projection confidence map.

- [ ] **Step 2: Fill in order of evidence strength**

1. symmetry/repetition from same component when motif repetition is established;
2. nearest same-material neighborhood with edge-aware inpainting for small gaps;
3. generic brass base only for hidden/unsupported regions.

Do not synthesize a new hero motif using unrelated online imagery.

- [ ] **Step 3: Save inferred-region mask**

```text
textures/final/T_ThaiLibation_InferredRegionMask.png
```

Final report must disclose percentage of final texture that is direct multi-view projection vs inferred/fill.

---

### Task 8: Assemble final texture set with baked ornament detail

**Files:**
- Modify: `final_texture_projection.py`
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Combine projected appearance with Plan 4 bakes**

Final maps:

```text
T_ThaiLibation_BaseColor.png
T_ThaiLibation_Roughness.png
T_ThaiLibation_Metallic.png
T_ThaiLibation_Normal.png
T_ThaiLibation_AO.png
T_ThaiLibation_Height.png  # only if retained
```

- [ ] **Step 2: Keep color-space correct**

```text
BaseColor: sRGB
Normal/Roughness/Metallic/AO/Height: Non-Color/linear
```

- [ ] **Step 3: Do not multiply AO permanently into Base Color**

AO may influence viewport material separately.

- [ ] **Step 4: Hash every final texture and store dimensions**

---

### Task 9: Build final Blender brass materials from the texture set

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Load `materials`, `realistic-style`, and `lookdev` skills**

- [ ] **Step 2: Create `MAT_Brass_Main`**

Principled contract:

```text
Base Color     <- final BaseColor
Metallic       <- final Metallic, default 1.0
Roughness      <- final Roughness
Normal         <- final Normal via Normal Map node
AO             <- separate controlled multiply/mask if used, not baked into BC
```

Do not recreate V1's large-scale procedural color noise over the photo-derived base color.

- [ ] **Step 3: Add `MAT_Brass_Interior` only if justified**

Compare bowl interior across multiple views. If warm/copper appearance moves with reflections, keep one material. If stable surface difference exists, create interior variation and document evidence.

- [ ] **Step 4: Add recess darkening through texture masks, not black geometry disks**

Any `MAT_Brass_RecessDark` use must correspond to real recessed/engraved regions or cavity interiors.

---

### Task 10: Build neutral diagnostic lighting and material review scene

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Create `COL_LIGHTS_NEUTRAL`**

Use broad neutral area lights, e.g.:

```text
LGT_Neutral_Key
LGT_Neutral_Fill
LGT_Neutral_Rim
```

Color near 5000-6500K equivalent, soft enough to show overall curvature but directional enough to read metallic response.

- [ ] **Step 2: Use neutral grey background/floor**

Avoid dark cinematic environment for diagnostics.

- [ ] **Step 3: Render fixed neutral views**

```text
60_material_lookdev/10_neutral_front.png
60_material_lookdev/11_neutral_quarter.png
60_material_lookdev/12_neutral_side.png
60_material_lookdev/13_neutral_top.png
60_material_lookdev/14_neutral_globe_close.png
60_material_lookdev/15_neutral_neck_close.png
```

- [ ] **Step 4: Diagnose material defects**

Common fixes:

| Symptom | Diagnose first | Fix |
|---|---|---|
| object too dark/brown | BC too dark or roughness/light rig wrong | inspect BC histogram, neutral light, then roughness |
| white blown globe | specular/lighting exposure | reduce light/exposure; do not darken BC |
| engraving disappears | normal strength/roughness too weak | correct normal scale and roughness contrast |
| fake painted reflections | highlight contamination in BC | improve multi-view fusion/reprojection |
| noisy sparkling detail | normal/height too strong | lower micro normal/height strength |

- [ ] **Step 5: Iterate lookdev with one major variable per pass**

Follow installed `lookdev` workflow. Keep a gap list and save screenshots for each pass.

---

### Task 11: Create reference-camera material comparison renders

**Files:**
- Modify: `build_final_model_blender.py`
- Modify: `final_model_validation.py`

- [ ] **Step 1: Render from selected source cameras**

At least 8 representative registered views:

```text
2 normal_side
2 low_angle
2 elevated_oblique
2 top_down
```

- [ ] **Step 2: Use neutral/light-matched comparison mode**

Do not expect exact illumination from the classroom scene, but match gross exposure/key direction enough to judge surface family.

- [ ] **Step 3: Create photo/render side-by-side and overlay panels**

Save under `70_final_photo_match/`.

- [ ] **Step 4: Record `materials` category rating using visual-match checklist**

Required before beauty pass: `Match` or `Close`.

---

### Task 12: Build beauty presentation lighting without hiding defects

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Create separate `COL_LIGHTS_BEAUTY`**

Use product-style motivated lights:

```text
LGT_Beauty_Key_Large
LGT_Beauty_Fill
LGT_Beauty_Rim
LGT_Beauty_TopStrip
```

- [ ] **Step 2: Optional HDRI**

If using external HDRI:

- use CC0/permissive source;
- store provenance/license/hash;
- use HDRI for reflections, with a clean background if desired;
- do not make final asset depend on unavailable proprietary asset.

- [ ] **Step 3: Set AgX and exposure deliberately**

Use AgX. Adjust exposure so gold highlights remain detailed and engraved regions remain visible.

- [ ] **Step 4: Create clean floor/contact shadow**

No dramatic black pedestal sink or excessive bloom.

---

### Task 13: Build final cameras and presentation renders

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Create named presentation cameras**

```text
CAM_Final_Front
CAM_Final_Quarter_Left
CAM_Final_Quarter_Right
CAM_Final_Side
CAM_Final_Elevated
CAM_Final_Top
CAM_Final_Low_Pedestal
CAM_Final_Globe_Detail
CAM_Final_Neck_Lid_Detail
CAM_Final_Bowl_Detail
CAM_Final_Turntable
```

- [ ] **Step 2: Match hero lens to product/object scale**

Use 50-100mm equivalent range as starting point for low-distortion hero renders, then frame intentionally. Do not blindly use 58mm from V1.

- [ ] **Step 3: Render final stills**

At minimum:

```text
previews/final_front.png
previews/final_quarter.png
previews/final_side.png
previews/final_elevated.png
previews/final_top.png
previews/final_globe_detail.png
previews/final_neck_detail.png
previews/final_bowl_detail.png
```

Hero render target: 4K long dimension or higher if practical.

- [ ] **Step 4: Create a complete 360-degree turntable**

Use a root/turntable empty rotating 0-360° at constant object position. The turntable must expose all sides and cannot hide incomplete texturing/ornament.

Preview can be Eevee; final may use Cycles if render cost is reasonable.

---

### Task 14: Apply final lookdev/material gate

**Files:**
- Modify: `final_model_validation.py`
- Modify: `run_final_model.py`

- [ ] **Step 1: Enforce texture integrity**

Check files, dimensions, hashes, color spaces, UV references, missing images, and inferred-region percentage.

- [ ] **Step 2: Visual material gate**

Pass only if:

```text
bright polished brass family matches photos
large regions not dark brown/black
engraving readable
no obvious baked moving highlights
no severe UV/projection seams
no obvious unsupported texture synthesis on hero region
```

- [ ] **Step 3: Re-run geometry silhouette gate under final material**

Materials/normal/displacement must not change silhouette enough to fail Plan 2 thresholds.

- [ ] **Step 4: Save accepted lookdev blend**

```text
work/70_LOOKDEV_ACCEPTED.blend
```

- [ ] **Step 5: Write `texture_projection_report.json` and `lookdev_report.json`**

Reports must identify:

```text
source views/cameras
CNN/reviewed masks used
visibility method
projection coverage
fusion method
highlight rejection
projected vs inferred texture percentage
baked map integration
material parameters
lighting rigs
visual-match ratings
```

**Plan completion gate:** Final export/publication begins only after `70_LOOKDEV_ACCEPTED.blend` and accepted lookdev report exist.
