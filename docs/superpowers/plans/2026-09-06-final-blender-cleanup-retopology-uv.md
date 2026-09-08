# Final Blender Cleanup, Sculpting, Retopology, UV, and Bake Implementation Plan

> **For agentic workers:** Execute this checklist task-by-task using any relevant Blender skills/plugins. GLM-only subagents may perform independent topology/render QA, but the parent agent must verify findings in Blender/runtime evidence. If an installed plan-execution workflow is available, use it; otherwise follow this file directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the accepted CV-fitted + ornamented V2 high/detail scene into a clean production presentation mesh with evidence-backed corrections, controlled sculpting, deliberate topology, production UVs, and validated bakes.

**Architecture:** This plan is deliberately diagnostic-first. Every destructive Blender operation requires a rollback file, a screenshot showing the problem, an exact diagnosis, a narrowly scoped fix, and an after screenshot. Large proportions remain locked to the CV geometry gate; sculpting is allowed only for source-supported manufacturing irregularity, relief transitions, dents/wear, and local surface continuity.

**Tech Stack:** Blender 5.2 Python plus available Blender integration/plugins, `blender-modeler`, `sculpting`, `retopology`, `uv-workflow`, `texture-workflow`, `asset-optimization`, `qa-review`.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

## Global Constraints

- Start only from `work/40_ORNAMENT_ACCEPTED.blend`.
- Do not change macro proportions to make topology easier.
- Do not use voxel remesh, global Decimate, Merge by Distance, Boolean union, or normal recalculation blindly.
- Every destructive action requires a saved rollback state first.
- Preserve high-poly ornament/bake sources separately from final presentation geometry.
- Final logical pieces remain physically meaningful: bowl/pedestal, vessel body/neck, lid/finial, and any evidence-supported hardware.
- Intentional physical separation is not a topology defect.
- Use Blender MCP or an equivalent available Blender inspection path throughout; do not merely describe what would be clicked.
- Quality outranks speed: use sufficient subdivision, baking resolution, render diagnostics, and cleanup iterations where they materially improve the model, while avoiding blind destructive operations.
- Do not ship a cleaner version of a visibly wrong component. If cleanup exposes a macro mismatch, return to the CV/base geometry gate rather than sculpting around it.

---

## Files / outputs

**Modify:**

```text
build_final_model_blender.py
final_model_validation.py
run_final_model.py
```

**Produce:**

```text
reconstruction/reference_assisted_v2/work/50_CLEAN_TOPOLOGY_ACCEPTED.blend
reconstruction/reference_assisted_v2/work/60_UV_BAKE_ACCEPTED.blend
reconstruction/reference_assisted_v2/reports/cleanup_report.json
reconstruction/reference_assisted_v2/reports/uv_bake_report.json
reconstruction/reference_assisted_v2/diagnostics/40_topology_cleanup/**
reconstruction/reference_assisted_v2/diagnostics/50_uv_bakes/**
reconstruction/reference_assisted_v2/textures/baked/**
```

---

### Task 1: Freeze pre-clean rollback and collect scene statistics

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Open only the accepted ornament blend**

Expected source:

```text
reconstruction/reference_assisted_v2/work/40_ORNAMENT_ACCEPTED.blend
```

Fail if missing or ornament report is not accepted.

- [ ] **Step 2: Save immutable pre-clean copy**

```text
reconstruction/reference_assisted_v2/work/41_PRE_CLEAN_ROLLBACK.blend
```

Never overwrite this file during the plan.

- [ ] **Step 3: Record exact scene topology stats**

For every mesh object capture:

```text
vertex count
edge count
polygon count
triangle count after evaluated modifiers
loose vertices/edges
non-manifold edge count
boundary edge count
material slots
modifier stack
transform state
bounding box
```

Write initial section of `cleanup_report.json`.

- [ ] **Step 4: Query object/collection hierarchy through the available Blender inspection interface (Blender MCP when available)**

Confirm no V1 objects or Step 17 photogrammetry objects are accidentally mixed into `COL_FINAL_V2`.

---

### Task 2: Capture required pre-clean diagnostic images

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `capture_cleanup_diagnostics(label: str)`.

- [ ] **Step 1: Neutral solid/matcap diagnostic render**

Use a neutral grey material and broad neutral lighting so shape defects are obvious.

Capture:

```text
40_topology_cleanup/00_preclean_front_solid.png
40_topology_cleanup/01_preclean_quarter_solid.png
40_topology_cleanup/02_preclean_side_solid.png
40_topology_cleanup/03_preclean_top.png
```

Look for:

```text
silhouette wobble
pinching
lumpy profile
surface discontinuity
visible ornament intersections
unintended sharp creases
```

- [ ] **Step 2: Wireframe diagnostic**

Capture front/quarter/close-up wire views with final/high source clearly distinguished.

Look for:

```text
uneven radial topology
excessive pole concentration
long thin faces
n-gons in high-curvature regions
unnecessary density
modifier-generated overlap
```

- [ ] **Step 3: Face Orientation diagnostic**

Temporarily use face-orientation viewport/shader or a deterministic front/back material diagnostic.

Capture:

```text
40_topology_cleanup/10_face_orientation_pre.png
```

Record every object with inward-facing exterior regions.

- [ ] **Step 4: Non-manifold diagnostic**

Use `bmesh` to identify edges where `len(edge.link_faces) != 2` on pieces expected to be closed.

Create report and optional highlighted copy/screenshot:

```text
40_topology_cleanup/11_nonmanifold_pre.png
```

Do not classify intentional open helper/high-source objects as final mesh failures; exclude them by collection role.

- [ ] **Step 5: Self/inter-object intersection diagnostic**

Check likely collision zones:

```text
bowl rim vs vessel globe
vessel lower seat vs bowl interior
neck base vs shoulder
lid vs neck seat
ornament high source vs host
pedestal bowl transition
```

Use BVH overlap or evaluated-mesh intersection checks where practical. Capture X-ray/wire screenshots of detected problem regions.

- [ ] **Step 6: Store defect list before modifying anything**

Each defect record:

```json
{
  "id": "TOP-001",
  "severity": "blocker|major|minor|note",
  "object": "SM_Bowl",
  "diagnostic": "...png",
  "symptom": "...",
  "suspected_cause": "...",
  "planned_fix": "..."
}
```

---

### Task 3: Normalize transforms and modifier dependencies safely

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Identify modifiers that depend on object scale**

Before applying transforms, inspect Screw, Solidify, Bevel, Shrinkwrap, Array/Geometry Nodes, and displacement stacks.

- [ ] **Step 2: Apply scale only when equivalent geometry is verified**

If an object has non-unit scale:

1. duplicate to hidden rollback collection;
2. apply scale on working copy;
3. compare evaluated bounding box and source-camera silhouette;
4. keep only if difference is within numerical tolerance.

- [ ] **Step 3: Do not flatten profile/Screw source until low topology is locked**

Keep editable profile curves and high source archived in `COL_HIGH`/`COL_ORNAMENT_HIGH`.

---

### Task 4: Fix normals and duplicate/loose geometry based on measured defects

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Correct known flipped normals object-by-object**

Use `bmesh.ops.recalc_face_normals` only on the selected connected exterior shell when appropriate. Preserve inner wall normals that intentionally face the cavity.

After each object correction, recapture face orientation.

- [ ] **Step 2: Remove loose geometry only when not part of intended ornament source**

Use connected-component analysis. Do not delete a separate lid/finial merely because it is not connected to vessel body.

- [ ] **Step 3: Measure Merge-by-Distance candidates before merging**

Compute minimum/near-duplicate vertex distances. Choose the smallest threshold that joins only accidental duplicates.

Starting diagnostic threshold may be:

```text
1e-6 to 1e-5 normalized units
```

Do **not** reuse Blender-modeler skill's generic `0.0001m` blindly because this project is normalized, not real meters.

- [ ] **Step 4: Verify silhouette after merge**

Any whole-object silhouette IoU decrease greater than `0.002` from the accepted geometry baseline requires reverting the merge and using a narrower/local fix.

---

### Task 5: Repair local surface continuity without changing the validated macro profile

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Prefer source profile correction for rotational surfaces**

If a ripple spans the circumference, fix the curve/profile or radial topology rather than sculpting around the entire object.

- [ ] **Step 2: Repair small local mesh defects manually/scriptedly**

For isolated pinches or bad poles:

```text
slide/reposition vertices along surface tangent
redistribute edge loops
replace local n-gon with controlled quad/tri fan
add/remove support loop where curvature requires it
```

- [ ] **Step 3: Preserve shoulder/lid/pedestal ring sharpness**

Use support loops or bevel widths consistent with source photos. Do not smooth construction rings into the host surface.

- [ ] **Step 4: Capture before/after diagnostics for each major repair**

At least one screenshot and report entry per major affected component.

---

### Task 6: Perform restrained evidence-supported sculpting

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces high-detail sculpt source, not untraceable macro changes.

- [ ] **Step 1: Load `sculpting` skill and create separate sculpt duplicate**

Name examples:

```text
HP_SCULPT_VesselBody
HP_SCULPT_Bowl
```

Keep accepted base/low object unchanged until bake/transfer decision.

- [ ] **Step 2: Use Multires for controlled detail**

Default to Multires, not Dyntopo, because macro topology/shape is already validated.

- [ ] **Step 3: Sculpt only photographed characteristics**

Allowed:

```text
subtle small dents actually visible in multiple photos
slight manufacturing waviness that appears consistently
softened relief edges
engraved recess depth refinement
localized wear/scratch forms visible in source close-ups
```

Forbidden:

```text
making globe bigger/smaller by eye
changing bowl silhouette
random dents for realism
adding missing decorative ideas not seen in photos
```

- [ ] **Step 4: Compare sculpt against source before accepting**

Use source close-up and neutral render side-by-side. If the sculpt is merely aesthetically pleasing but unsupported, remove it.

- [ ] **Step 5: Bake/transfer sculpt detail rather than exporting extreme multires**

High sculpt remains bake source.

---

### Task 7: Decide whether manual retopology is actually needed per component

**Files:**
- Modify: `build_final_model_blender.py`
- Modify: `cleanup_report.json`

- [ ] **Step 1: Classify each component**

```text
KEEP_PARAMETRIC_TOPOLOGY
LOCAL_RETOPO
FULL_RETOPO
```

Criteria:

- rotational Screw mesh with even loops and clean shading: keep;
- local Boolean/ornament integration damage: local retopo;
- globally irregular destructive high mesh: full retopo only if necessary.

- [ ] **Step 2: Do not force quad-only topology for static closed prop**

This is a static hero prop, not a deforming character. Quads are preferred for editability/subdivision, but clean triangles in hidden/non-deforming regions are acceptable.

- [ ] **Step 3: Record the decision and evidence per component**

---

### Task 8: Retopologize only components classified for it

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Create `COL_LOW` working mesh**

Names:

```text
SM_VesselBody_LOW
SM_VesselNeck_LOW
SM_Lid_LOW
SM_Finial_LOW
SM_Bowl_LOW
SM_Pedestal_LOW
```

- [ ] **Step 2: For rotational parts, regenerate clean topology from accepted profile where possible**

This is preferred over generic auto-remesh. Use same fitted profile with appropriate radial segment count and support loops.

- [ ] **Step 3: For non-rotational/local relief surfaces, use Shrinkwrap-guided retopo**

1. high source in hidden render collection;
2. low mesh with even topology;
3. Shrinkwrap to high source;
4. preserve validated macro silhouette;
5. add loops only where relief silhouette requires geometry.

- [ ] **Step 4: Avoid one-click Quad Remesh as final hero topology**

If used as a starting point, manually inspect and repair poles, ring continuity, and boundaries before acceptance.

- [ ] **Step 5: Compare source-camera silhouette to accepted base**

Retopo must not reduce the accepted geometry metrics below gate. The target is effectively identical macro silhouette.

---

### Task 9: Integrate macro ornament appropriately into final presentation mesh

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Classify high ornament**

For each family choose:

```text
retain_as_geometry
bake_to_normal_height
bake_to_normal_only
material_mask_only
```

- [ ] **Step 2: Geometry only where relief changes silhouette or casts meaningful close-up shadows**

Raised rings/rims remain geometry. Most shallow engraved motifs should bake rather than remain hundreds of separate floating objects.

- [ ] **Step 3: If Boolean union is necessary, do it on a duplicate**

After Boolean:

1. inspect manifold state;
2. inspect shading;
3. inspect wireframe;
4. compare silhouette;
5. retopo local damage if needed.

- [ ] **Step 4: No orphan ornament objects in `COL_EXPORT`**

Every exported ornament object must correspond to intentional physical geometry.

---

### Task 10: Run the clean-topology acceptance diagnostics

**Files:**
- Modify: `final_model_validation.py`
- Modify: `run_final_model.py`

- [ ] **Step 1: Capture after-clean diagnostic set matching Task 2 views**

Save under:

```text
40_topology_cleanup/after/
```

- [ ] **Step 2: Validate**

Final low/export candidate must have:

```text
0 unintended non-manifold edges
0 loose final vertices/edges/faces
0 inverted exterior faces
0 known accidental self-intersections
0 default Cube/Sphere/Torus-style names
all transforms documented/applied as required
```

- [ ] **Step 3: Re-run source-camera geometry metrics**

Do not accept cleanup if geometry falls below Plan 2 gate.

- [ ] **Step 4: Save accepted topology file**

```text
work/50_CLEAN_TOPOLOGY_ACCEPTED.blend
```

---

### Task 11: Plan and mark production UV seams component-by-component

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces `UV_Main` on final low meshes.

- [ ] **Step 1: Load `uv-workflow` skill**

Use manual seam + unwrap as production default. Smart UV is allowed only as temporary diagnostic, never final.

- [ ] **Step 2: Place seams on low-visibility/construction locations**

Recommended seam logic:

```text
Vessel globe: one rear meridian + natural horizontal construction seam as needed
Neck: one rear meridian + collar boundaries
Lid: rear meridian + underside/lip boundaries
Finial: rear meridian / hidden underside
Bowl: rear meridian + underside/rim construction edges
Pedestal: rear meridian + underside foot boundary
```

The actual seam locations should be chosen relative to canonical hero orientation and ornament placement so they do not cut through the photographed hero motif when avoidable.

- [ ] **Step 3: Unwrap strips with Follow Active Quads/manual unwrap where suitable**

Cylindrical/revolved surfaces should produce stable low-distortion strips.

- [ ] **Step 4: Assign unique UV space to unique ornamented regions**

Do not mirror/overlap regions whose left/right ornament or wear differs visibly.

---

### Task 12: Set texel density and pack UVs

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Choose one 4K atlas or documented UDIM/8K strategy**

Default:

```text
one 4096x4096 hero atlas
```

Move to 8192 or UDIM only if 4K source-close-up bake inspection proves insufficient and VRAM/runtime support it.

- [ ] **Step 2: Normalize visible texel density**

Give hero globe/neck/bowl visible surfaces priority. Hidden underside can use lower density only when documented.

- [ ] **Step 3: Pack with minimum safe bake margin**

At 4K use at least 16 px effective bake padding. Increase if mip/export seams remain visible.

- [ ] **Step 4: Measure pack efficiency**

Target `>=75%` occupied 0-1 area unless documented separation/UDIM makes this metric inappropriate.

- [ ] **Step 5: Create checker diagnostics**

Render front/quarter/top and ornament close-ups using a UV checker.

Save:

```text
50_uv_bakes/00_uv_layout.png
50_uv_bakes/01_checker_front.png
50_uv_bakes/02_checker_quarter.png
50_uv_bakes/03_checker_closeup.png
```

Reject visible stretching or inconsistent checks on hero surfaces.

---

### Task 13: Bake high-detail geometry into final texture-space maps

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces baked normal/AO/height/curvature sources.

- [ ] **Step 1: Load `texture-workflow` skill**

- [ ] **Step 2: Build explicit high/low bake pairs**

For every component store custom properties or report mapping:

```text
low object
high source objects
cage/ray distance
texture target
```

- [ ] **Step 3: Determine cage distance from normalized geometry, not generic meter values**

Measure maximum relief depth and use a small safety margin. Start with approximately `1.25x` measured maximum relief depth, then inspect bake.

- [ ] **Step 4: Bake maps**

Required:

```text
Normal (OpenGL convention for GLB)
AO
Curvature or equivalent wear mask source
```

Optional only if useful:

```text
Height/Displacement
Position
```

Do not bake AO into base color.

- [ ] **Step 5: Inspect tangent normal orientation**

Test a recognizable raised/engraved motif under neutral light. If relief appears inverted, correct normal convention/settings before proceeding.

- [ ] **Step 6: Save bake diagnostics**

```text
50_uv_bakes/10_normal_preview.png
50_uv_bakes/11_ao_preview.png
50_uv_bakes/12_curvature_preview.png
50_uv_bakes/13_bake_closeup_source_vs_low.png
```

- [ ] **Step 7: Fix bake artifacts systematically**

Common diagnosis/fix mapping:

| Symptom | Diagnosis | Fix |
|---|---|---|
| black/empty patches | rays miss high source | adjust cage locally/ray distance |
| doubled relief | overlapping high objects | isolate bake set / clean source |
| seam line | insufficient margin/tangent split | increase margin, align hard edges/seams |
| wavy normal | low topology too coarse | add supporting low geometry or correct cage |
| inverted emboss | normal convention/ray direction | correct bake orientation |

Do not hide bake errors with roughness/color.

---

### Task 14: Apply UV/bake hard gate and save accepted checkpoint

**Files:**
- Modify: `final_model_validation.py`
- Modify: `run_final_model.py`

- [ ] **Step 1: Validate UV contract**

Required:

```text
all final visible meshes have UV_Main
no unintended unique-detail overlap
pack efficiency >=75% or documented exception
checker has no major stretching
seams avoid primary hero motif where possible
```

- [ ] **Step 2: Validate baked maps**

Required files non-empty/readable, correct resolution/color-space contract, normal orientation verified, no major hero-view seam/bake artifact.

- [ ] **Step 3: Write `uv_bake_report.json`**

Include texture resolution, UV density metrics, overlap results, high/low pairs, cage settings, bake hashes, diagnostic screenshots, and acceptance.

- [ ] **Step 4: Save accepted UV/bake blend**

```text
work/60_UV_BAKE_ACCEPTED.blend
```

- [ ] **Step 5: Reopen the accepted blend in background Blender**

Verify UV layers and baked image references survive reopening.

**Plan completion gate:** Material/photo projection lookdev begins only from `60_UV_BAKE_ACCEPTED.blend`.
