# Final Ornament and Photo-Detail Reconstruction Implementation Plan

> **For agentic workers:** Execute this checklist task-by-task using any relevant CV/Blender skills/plugins. GLM-only subagents may independently inspect motif evidence or rendered detail; verify their conclusions in the parent agent. If an installed plan-execution workflow is available, use it; otherwise follow this file directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconstruct the photographed object's actual engraved/embossed motif families and micro-surface detail from source images, using CV correspondence/rectification plus Blender high-detail/bake workflows, without reintroducing generic V1 decoration.

**Architecture:** Ornament is treated as image evidence first and Blender geometry second. Source patches are selected, matched across views using Step 6 SIFT/RANSAC or the existing ALIKED/LightGlue fallback, rectified onto approximate local tangent/cylindrical coordinates, consolidated into motif masks/height sources, then represented as high-poly relief or normal/height detail according to scale. All hero ornament remains source-traceable.

**Tech Stack:** OpenCV, NumPy, existing SIFT/ALIKED/LightGlue utilities, Blender 5.2 Python plus available Blender integration/plugins, curves/Geometry Nodes or high-poly mesh, bake pipeline.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

## Global Constraints

- Requires accepted base geometry from Plan 2.
- V1 generic oval/diamond ornament is forbidden unless an exact source motif proves it.
- No chain by default. Add one only with two-view source evidence establishing it.
- Ornament on unseen backsides may be repeated only when the visible manufactured pattern clearly repeats.
- Never claim inferred repetition as directly photographed.
- Large form remains unchanged in this plan; ornament cannot compensate for bad silhouette.
- Consume Plan 2 `surface_evidence_coverage.json`. Directly observed sectors, detail-only sectors, symmetry/repetition-inferred sectors, and unsupported hidden sectors must remain distinguishable in ornament provenance.
- Do not populate an unseen backside with ornament merely because a front-side motif exists. Full-ring repetition is allowed only when manufactured repetition/count/phase is established from the multi-view evidence; otherwise keep unsupported sectors conservative and disclose them.
- Use Blender `sculpting`, `texture-workflow`, `materials`, and `qa-review` skills as applicable.
- Spend compute on real multi-view correspondence, rectification, motif consolidation, high-detail generation, and close-up render comparison when it improves fidelity. Do not replace difficult source reconstruction with generic decoration.
- A technically clean ornament pass that does not reproduce the photographed motif family is a failed gate and must be revised.

---

## Files

**Modify/Create:**

```text
final_reference_evidence.py
final_model_validation.py
build_final_model_blender.py
run_final_model.py
tests/test_final_reference_evidence.py
tests/test_final_model_validation.py
```

**Produce:**

```text
reconstruction/reference_assisted_v2/evidence/ornament_crops/**
reconstruction/reference_assisted_v2/evidence/cv_matches/ornament_*.png
reconstruction/reference_assisted_v2/reports/ornament_manifest.json
reconstruction/reference_assisted_v2/reports/ornament_build_report.json
reconstruction/reference_assisted_v2/diagnostics/30_ornament/**
reconstruction/reference_assisted_v2/work/40_ORNAMENT_ACCEPTED.blend
```

---

### Task 1: Create an explicit photographed ornament inventory

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces `OrnamentFamily`, `build_ornament_inventory()`.

- [ ] **Step 1: Define exact ornament-family schema**

```python
@dataclass(frozen=True)
class OrnamentFamily:
    family_id: str
    host_component: str
    representation: str
    primary_view_indices: tuple[int, ...]
    repeat_mode: str
    repeat_count: int | None
    confidence: str
    notes: str
```

Allowed representation values:

```text
explicit_geometry
highpoly_bake
normal_height
roughness_only
color_roughness
```

- [ ] **Step 2: Start inventory from actual source-visible families**

At minimum inspect and classify:

```text
ORB_GLOBE_CROSSHATCH       dense cross-hatched/pebbled field on globe
ORB_GLOBE_HERO_MOTIF       large triangular/floral/Thai hero motif seen in close oblique photos
ORB_GLOBE_SCROLL_BAND       scrolling/floral motifs around globe
ORB_GLOBE_LOWER_BAND        horizontal lower textural/seam band
ORB_SHOULDER_RINGS           concentric shoulder steps/rings
ORB_NECK_FIELD               fine neck background texture
ORB_NECK_LOTUS               lotus/flame-like vertical motifs on neck
ORB_NECK_UPPER_BAND          decorated upper neck collar/band
ORB_LID_TIERS                stepped lid rings, macro geometry already present
ORB_LID_DECOR_BAND           ornament immediately below stepped lid where visible
ORB_BOWL_FLAME_BAND          repeated flame/lotus-like band around bowl wall
ORB_PEDESTAL_RINGS           pedestal construction rings
```

Do not assume all are independent motifs; merge families after visual inspection if they are the same repeated pattern at different locations.

- [ ] **Step 3: Use strongest reference views**

Prioritize high-detail source images including reviewed oblique-detail indices:

```text
267 IMG20260826130313.jpg
268 IMG20260826130322.jpg
278 IMG20260826130357.jpg
288 IMG20260826130427.jpg
```

and elevated/top-down sources such as 148/154/165/206/255 where they show ornament cleanly.

- [ ] **Step 4: Write `ornament_manifest.json` with source provenance**

For each family store source filename/index, bounding box, image hash, whether registered in Step 13, and whether the family is directly observed or inferred repetition.

---

### Task 2: Extract high-resolution ornament crops without altering source images

**Files:**
- Modify: `final_reference_evidence.py`

**Interfaces:**
- Produces `extract_ornament_crop()` and durable crops.

- [ ] **Step 1: Add bounding-box validation tests**

```python
def test_ornament_bbox_must_stay_inside_source(): ...
```

- [ ] **Step 2: Extract lossless PNG crops from immutable JPEG originals**

Signature:

```python
def extract_ornament_crop(
    source_path: Path,
    bbox_xyxy: tuple[int, int, int, int],
    output_path: Path,
) -> dict: ...
```

Record:

```text
source_sha256
crop_sha256
source_bbox
source_dimensions
```

- [ ] **Step 3: Create crop contact sheets by family**

Each family sheet should show at least two occurrences/views when available. If only one clear source exists, record `support_count=1` and lower confidence.

- [ ] **Step 4: Visually inspect crop correctness**

Reject crops dominated by specular blowout, motion blur, or background when a better source exists.

---

### Task 3: Align repeated ornament patches using the existing CV stack

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Consumes `align_detail_pair()` from Plan 1.
- Produces family-specific rectified patch stacks.

- [ ] **Step 1: Match each secondary crop to a designated anchor crop**

Use:

```text
SIFT + RANSAC first
ALIKED + LightGlue only when SIFT fails V2 alignment thresholds
```

Do not use manual point placement as the first alignment method.

- [ ] **Step 2: Estimate local homography only for approximately planar/small tangent patches**

For large curved globe patches, subdivide into smaller overlapping regions or use cylindrical parameterization from the accepted base geometry/camera projection rather than forcing one global homography.

- [ ] **Step 3: Write match diagnostics**

For every anchor/secondary pair save:

```text
raw_match_view.png
inlier_match_view.png
rectified_overlay.png
```

and report match/inlier counts.

- [ ] **Step 4: Reject bad photometric alignment explicitly**

If geometric inliers are good but polished reflections dominate one patch, keep it for geometry support but downweight/exclude it from texture/height consolidation.

---

### Task 4: Convert source ornament into clean motif masks/height references

**Files:**
- Modify: `final_reference_evidence.py`
- Test: `tests/test_final_reference_evidence.py`

**Interfaces:**
- Produces `build_motif_reference()`.

- [ ] **Step 1: Create grayscale/contrast-normalized analysis copies**

Use local contrast enhancement only on derived crops. Do not claim enhanced copies are original texture.

- [ ] **Step 2: Derive line/relief candidates using classical CV**

For engraved/raised motif extraction, combine as appropriate:

```text
gradient magnitude
Canny edges
adaptive/local threshold
morphological closing/opening
multi-view median of rectified crops
```

Keep parameters family-specific in the manifest/report rather than one global threshold.

- [ ] **Step 3: Build consensus mask from multiple aligned views**

When two or more aligned patches exist:

- normalize intensities;
- compute robust median/trimmed mean;
- preserve structures repeated across views;
- suppress view-specific bright specular streaks;
- save confidence mask showing multi-view support.

- [ ] **Step 4: Preserve authentic asymmetry**

Do not smooth away clear non-symmetric motif edges merely to make a perfect vector shape.

- [ ] **Step 5: Save source/analysis/consensus triplets**

For each hero family:

```text
source_anchor.png
consensus.png
mask_or_height.png
```

---

### Task 5: Determine repeat count and placement from the photographs

**Files:**
- Modify: `final_reference_evidence.py`

**Interfaces:**
- Produces placement entries in `ornament_manifest.json`.

- [ ] **Step 1: Measure visible angular spacing using camera-aware geometry**

For registered views, project the accepted rotational host surface and estimate motif center azimuths from image positions. Cross-reference `surface_evidence_coverage.json` so every measured center is tagged as directly observed, and every later replicated sector is tagged as inferred rather than silently promoted to observed evidence.

- [ ] **Step 2: Infer full repeat count only from consistent evidence**

Example rule:

```text
visible adjacent center spacing must be observed in >=2 views
estimated full count must agree within ±1 between views
```

If not, do not force a full exact count; use only photographed motifs on visible presentation sectors and disclose incomplete hidden ornament.

- [ ] **Step 3: Store phase/orientation**

A repeated ring needs both count and phase angle. Align the first replicated motif with a real source-visible motif, not an arbitrary world axis.

---

### Task 6: Build explicit macro ornament as high-detail Blender source

**Files:**
- Modify: `build_final_model_blender.py`

**Interfaces:**
- Produces high-poly ornament under `COL_ORNAMENT_HIGH`.

- [ ] **Step 1: Load accepted base blend, never rebuild base from V1**

Start from:

```text
work/30_BASE_GEOMETRY_ACCEPTED.blend
```

Save first ornament rollback:

```text
work/31_before_ornament.blend
```

- [ ] **Step 2: Recreate only evidence-supported large motif families**

Recommended methods:

```text
curves projected/shrinkwrapped to host surface
curve bevel converted to mesh for raised/engraved line sources
controlled displacement from family height masks
Geometry Nodes radial repetition using measured count/phase
```

- [ ] **Step 3: Keep high ornament separate from clean low host**

High source naming:

```text
HP_ORN_GlobeHeroMotif
HP_ORN_GlobeScrollBand
HP_ORN_NeckLotus
HP_ORN_BowlFlameBand
HP_ORN_LidDecorBand
```

- [ ] **Step 4: Do not apply destructive booleans to the accepted base without backup**

For shallow detail, prefer bake source over boolean union. Use geometry union only where the physical relief changes silhouette or produces visible true depth.

- [ ] **Step 5: Capture source-vs-highpoly screenshots**

For every hero family create:

```text
source_crop.png
highpoly_clay.png
overlay_or_side_by_side.png
wire.png
```

under `diagnostics/30_ornament/<family>/`.

---

### Task 7: Build cross-hatch and fine engraving as texture/bake sources, not floating geometry

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Convert consensus field textures to height/normal source**

Cross-hatch/pebble detail should tile or project cleanly at the host surface scale. Use the photographed field frequency, not arbitrary procedural noise.

- [ ] **Step 2: Establish UV-independent high source if final UVs are not yet locked**

Use object/local cylindrical coordinates or shrinkwrapped displacement source for now. Final bake happens in Plan 4 after UV lock.

- [ ] **Step 3: Verify density against source close-ups**

Render neutral macro views at approximately the source crop framing. Detail should neither disappear nor become oversized diamond scales.

---

### Task 8: Diagnose the chain question explicitly

**Files:**
- Modify: `final_reference_evidence.py`
- Modify: `ornament_manifest.json`

- [ ] **Step 1: Search the full source image set for visible chain-like hardware**

Use visual inspection plus optional classical line/loop candidate review. Do not use V1 as evidence.

- [ ] **Step 2: Require two-view confirmation before modeling**

If no two independent images show a chain and attachment location:

```json
{
  "chain": {
    "supported": false,
    "decision": "omit_from_v2"
  }
}
```

If supported, add a dedicated sub-entry with exact image references and follow physical link geometry. No generic decorative dangling chain.

---

### Task 9: Run ornament visual-match review before topology/bakes

**Files:**
- Modify: `final_model_validation.py`
- Modify: `run_final_model.py`

- [ ] **Step 1: Render neutral ornament checks from at least six cameras**

Required coverage:

```text
globe hero motif
opposite/adjacent globe scroll region
neck lower motif
neck upper band
bowl flame band
lid band/tier close view
```

- [ ] **Step 2: Create source-vs-render panels**

Use real source crops with labels and a neutral clay/highpoly render. Do not use beauty material for this gate.

- [ ] **Step 3: Apply hard rejection rules**

Fail if any of these remain:

```text
floating decorative studs
unsupported chain
wrong motif family
wrong repeat scale/count obvious from source
ornament intersects host visibly
hero motif absent where photographed
cross-hatch scale obviously wrong
```

- [ ] **Step 4: Save accepted ornament blend**

Only after pass:

```text
reconstruction/reference_assisted_v2/work/40_ORNAMENT_ACCEPTED.blend
```

- [ ] **Step 5: Write `ornament_build_report.json`**

Include family sources, CV alignment method, support count, representation, inferred repetition, screenshot paths, and acceptance verdict.

**Plan completion gate:** Cleanup/retopology/UV work starts only from `40_ORNAMENT_ACCEPTED.blend`.
