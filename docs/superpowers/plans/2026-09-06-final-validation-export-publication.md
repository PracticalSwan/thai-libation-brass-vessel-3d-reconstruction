# Final Validation, Export, Documentation, and Publication Implementation Plan

> **For agentic workers:** Execute this checklist task-by-task using any relevant QA/export/documentation skills/plugins. GLM-only subagents may independently review final renders, source matching, code, tests, or publication readiness; the parent agent must verify every accepted conclusion. If an installed plan-execution workflow is available, use it; otherwise follow this file directly. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the completed V2 model matches the photographed object and preserves the CV workflow, export a working `.blend` and `.glb`, verify a fresh GLB re-import/render, update authoritative documentation, and publish only the intended project artifacts.

**Architecture:** Final acceptance combines machine checks, CV/reference-camera metrics, screenshot-backed Blender QA, fresh exported-asset re-import, and repository verification. A technically valid file that visually fails the real photographs receives `NO-SHIP`. Only accepted artifacts are promoted to `reconstruction/reference_assisted_v2/final/`.

**Tech Stack:** Python/OpenCV/NumPy, Blender 5.2 Python/background plus available integration/plugins, pytest, Git/GitHub, `asset-optimization`, `export-pipeline`, `qa-review`, `documentation-verification`, `avoid-ai-writing` for prose cleanup.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

## Global Constraints

> **Current user-approval boundary:** Execute Tasks 1-4 to produce and verify the complete Blender model, then STOP for user inspection. Tasks 5 onward (GLB export, clean re-import, promotion/publication) are forbidden until explicit user approval is recorded in `reconstruction/reference_assisted_v2/reports/user_export_approval.json` with `approved=true`. The runtime `export` stage enforces this gate.

- Start only from `work/70_LOOKDEV_ACCEPTED.blend`.
- Do not change the model merely to make export easier without rerunning all affected visual/geometry gates.
- Final QA is screenshot-backed and source-photo-backed.
- V2 is described as CV-constrained + Blender-completed, not direct complete photogrammetry.
- Direct Steps 1-17 evidence remains untouched.
- No commit/push until final diff, artifact sizes, secrets/private paths, and tests are inspected.
- Never force-push.
- Final `NO-SHIP` is not an acceptable stopping state. A failed QA category routes back to the responsible earlier plan for correction and revalidation. Stop only if a genuine external blocker remains after reasonable evidence-directed alternatives are exhausted; never fabricate `SHIP`.
- Use enough compute/rendering/verification to make the final decision credible. Do not weaken final QA merely to finish faster.

---

## Files / outputs

**Create/Modify:**

```text
final_model_validation.py
run_final_model.py
build_final_model_blender.py
tests/test_final_model_validation.py
tests/test_run_final_model.py
README.md
AGENTS.md
CHANGELOG.md
LESSONS.md
docs/memory-bank/active-context.md
docs/memory-bank/progress.md
docs/geometry-ml/reference-assisted-presentation-model.md
docs/geometry-ml/final-high-fidelity-presentation-model.md
```

**Final outputs:**

```text
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.blend
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.glb
reconstruction/reference_assisted_v2/final/textures/**
reconstruction/reference_assisted_v2/reports/final_validation_report.json
reconstruction/reference_assisted_v2/reports/final_asset_manifest.json
reconstruction/reference_assisted_v2/reports/final_qa_report.md
reconstruction/reference_assisted_v2/diagnostics/70_final_photo_match/**
reconstruction/reference_assisted_v2/diagnostics/80_export_reimport/**
```

---

### Task 1: Complete final validation module contracts

**Files:**
- Create/Modify: `final_model_validation.py`
- Create/Modify: `tests/test_final_model_validation.py`

**Interfaces:**
- Produces `FinalValidationResult`, `aggregate_final_validation()`, `compare_render_images()`.

- [ ] **Step 1: Define exact result structure**

```python
@dataclass(frozen=True)
class FinalValidationResult:
    geometry_passed: bool
    ornament_passed: bool
    topology_passed: bool
    uv_bake_passed: bool
    texture_passed: bool
    visual_identity_passed: bool
    export_reimport_passed: bool
    qa_verdict: str
    accepted: bool
```

- [ ] **Step 2: Write aggregation tests**

```python
def test_final_acceptance_requires_every_gate():
    result = aggregate_final_validation(
        geometry=True,
        ornament=True,
        topology=True,
        uv_bake=True,
        texture=True,
        visual_identity=True,
        export_reimport=False,
        qa_verdict="SHIP",
    )
    assert result.accepted is False
```

Also test that `qa_verdict="SHIP WITH NOTES"` does not satisfy final canonical acceptance.

- [ ] **Step 3: Add image comparison helper for master/export renders**

Signature:

```python
def compare_render_images(reference_path: Path, candidate_path: Path) -> dict: ...
```

Normalize exact dimensions/color space, then report at least:

```text
mean absolute RGB difference
95th percentile absolute difference
silhouette IoU
alpha/foreground coverage difference if transparent
```

Do not use this metric to judge artistic photo fidelity; it is specifically for master-vs-reimport export equivalence.

---

### Task 2: Run a final full-source visual identity review

**Files:**
- Modify: `final_model_validation.py`
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Select final reference review set**

Use at least:

```text
4 normal/side references
4 low-angle pedestal references
4 elevated-oblique references
4 top-down references
4 ornament/detail references
```

The 16 geometry canonical views can satisfy the first four groups; use 267/268/278/288 plus other close crops for detail.

- [ ] **Step 2: Render final model from registered source cameras**

Use final material but neutral/reference-comparison lighting mode. Create the same view alignment used for geometry validation.

- [ ] **Step 3: Create final comparison panels**

For geometry views:

```text
SOURCE | FINAL RENDER | SILHOUETTE OVERLAY | DIFFERENCE/LANDMARKS
```

For detail views:

```text
SOURCE CROP | FINAL DETAIL RENDER | ORNAMENT REFERENCE MASK/ANNOTATION
```

- [ ] **Step 4: Recompute final geometry metrics**

Still require:

```text
median silhouette IoU >= 0.90
minimum reliable-view IoU >= 0.84
median landmark error <= 0.020 object height
95th percentile landmark error <= 0.040
```

- [ ] **Step 5: Apply visual identity veto**

Explicitly rate:

```text
bowl/pedestal profile
globe proportions
shoulder construction
neck proportions
neck ornament
lid tier profile
finial
globe hero ornament
bowl ornament
surface finish
overall same-object identity
```

Any `Miss` on bowl/globe/neck/lid/finial or overall identity = fail.

---

### Task 3: Run full Blender asset QA using installed skills

**Files:**
- Modify: `final_model_validation.py`
- Produce: `final_qa_report.md`

- [ ] **Step 1: Load `asset-optimization` and `qa-review` skills**

Use their checklists, but adapt game-only requirements that are irrelevant to a coursework presentation prop rather than inventing collision/LOD requirements.

- [ ] **Step 2: Capture required final QA screenshots**

At minimum:

```text
hero 3/4 beauty
orthographic/profile clay
wireframe overview
face orientation
UV checker/layout
neutral material render
source-vs-final comparison
```

- [ ] **Step 3: Run final naming/organization audit**

Fail on:

```text
Cube.001 / Sphere.### / Torus.### style final names
orphan test objects
V1 geometry mixed into export
unused material slots
missing images
unintended hidden export geometry
```

- [ ] **Step 4: Run topology audit on final export collection**

Require zero unintended non-manifold/loose/inverted/intersection defects according to accepted cleanup report.

- [ ] **Step 5: Issue QA verdict**

Use exact:

```text
SHIP
SHIP WITH NOTES
NO-SHIP
```

Canonical promotion requires `SHIP`.

---

### Task 4: Save canonical final Blender master without losing source/editability data

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Duplicate accepted lookdev blend to final working copy**

Do not rename/move the only accepted rollback.

- [ ] **Step 2: Keep useful editable sources in the `.blend` but isolate them**

The final `.blend` may retain hidden:

```text
COL_REFERENCE
COL_HIGH
COL_ORNAMENT_HIGH
profile curves
CV cameras
neutral lights
```

They should be excluded from `COL_EXPORT` and clearly named.

- [ ] **Step 3: Set startup scene for user usability**

When opening final `.blend`:

- final model visible;
- beauty or neutral view usable;
- test/debug geometry hidden;
- final model selected logically;
- cameras/lights organized;
- no V1 object visible.

- [ ] **Step 4: Save canonical master**

```text
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.blend
```

Compute SHA-256 only after all final saves are complete.

---

### Task 5: Export canonical GLB through `COL_EXPORT`

**HARD PRECONDITION:** Do not begin this task until the user has inspected the complete Blender model and explicitly approved export. Without `reports/user_export_approval.json` containing `approved=true`, stop and wait for user fixes/approval.

**Files:**
- Modify: `build_final_model_blender.py`

- [ ] **Step 1: Load `export-pipeline` skill**

- [ ] **Step 2: Prepare only intended export objects**

Final export collection should include cleaned presentation meshes and material dependencies only. It should not include:

```text
reference images
CV cameras
lights
high-poly bake sources
profile helper curves
diagnostic geometry
rollback duplicates
```

- [ ] **Step 3: Apply export transforms on a duplicate/export representation when needed**

Do not destructively flatten master editability simply for GLB.

- [ ] **Step 4: Export GLB**

Target:

```text
reconstruction/reference_assisted_v2/final/Thai_Libation_Vessel_FINAL.glb
```

Use appropriate GLB settings:

```text
selected/export collection only
materials included
apply evaluated modifiers as intended
+Y-up GLTF convention handled by exporter
normal/tangent data exported
no animations unless turntable is intentionally included (default: do not include turntable animation in asset GLB)
```

- [ ] **Step 5: Record export settings in report**

---

### Task 6: Freshly re-import the GLB into a clean Blender process

**Files:**
- Modify: `build_final_model_blender.py` or add a bounded validation entry point in same script.

- [ ] **Step 1: Open factory-empty Blender background process**

Do not validate by importing into the already-populated master scene.

- [ ] **Step 2: Import `Thai_Libation_Vessel_FINAL.glb`**

Record:

```text
object count
mesh count
vertex count
polygon/triangle count
material count
image/texture count
bounds
orientation
```

- [ ] **Step 3: Check missing external files**

GLB should be self-contained where expected. Any external texture references must resolve and be intentionally packaged.

- [ ] **Step 4: Render neutral comparison image from re-imported GLB**

Use the same fixed neutral camera/lighting numeric setup as the master export-equivalence render.

Save:

```text
diagnostics/80_export_reimport/master_neutral.png
diagnostics/80_export_reimport/reimport_neutral.png
diagnostics/80_export_reimport/difference.png
```

- [ ] **Step 5: Compare master/reimport renders**

Expected tolerance should allow minor raster/export tangent differences but not material/geometry loss. Starting acceptance:

```text
silhouette IoU >= 0.995
mean absolute RGB difference <= 0.03 on foreground-normalized 0-1 RGB
no missing hero ornament/materials by visual inspection
```

If tangent/renderer differences make RGB threshold noisy but visual result is equivalent, document the measured reason and use a defensible adjusted threshold once. Do not simply disable the comparison.

---

### Task 7: Promote final artifacts only after all gates pass

**Files:**
- Modify: `final_model_io.py`
- Modify: `run_final_model.py`

- [ ] **Step 1: Implement final-promotion guard**

Promotion requires all accepted reports:

```text
final_cv_fit.json
base_geometry_report.json
ornament_build_report.json
cleanup_report.json
uv_bake_report.json
texture_projection_report.json
lookdev_report.json
final_validation_report.json
```

- [ ] **Step 2: Build `final_asset_manifest.json`**

Record:

```text
master blend path/size/hash
GLB path/size/hash
texture paths/sizes/hashes
preview/render paths
CV evidence report hashes
source/protected boundary statement
scale status
```

- [ ] **Step 3: Remove only V2 transient work that is no longer required**

Use exact allowlist inside:

```text
reconstruction/reference_assisted_v2/work/
```

Retain accepted rollback files if they are useful and not too large; otherwise document local-only retention before removal. Never delete V1 or Steps 1-17 evidence.

---

### Task 8: Run complete code/test/runtime verification

**Files:**
- All V2 Python/test files.

- [ ] **Step 1: Compile changed/new regular Python modules**

```text
python -B -m py_compile final_model_io.py final_reference_evidence.py final_cv_model_fit.py final_texture_projection.py final_model_validation.py run_final_model.py
```

Blender-only script syntax should also be checked through Blender's Python execution path.

- [ ] **Step 2: Run all focused V2 tests**

```text
python -B -m pytest tests/test_final_model_io.py tests/test_final_reference_evidence.py tests/test_final_cv_model_fit.py tests/test_final_texture_projection.py tests/test_final_model_validation.py tests/test_run_final_model.py -q
```

- [ ] **Step 3: Run the complete project test suite**

```text
python -B -m pytest -q
```

Do not claim all tests pass unless this full relevant suite completes.

- [ ] **Step 4: Run the real integrated final pipeline**

```text
python -B run_final_model.py --stage all
```

It must reuse accepted stages where appropriate and end with final acceptance/export success.

- [ ] **Step 5: Reverify protected evidence**

At minimum verify:

```text
297 raw originals unchanged against authoritative hashes
288 selected inputs unchanged
Step 10-13 protected reports/models unchanged where their hashes were previously frozen
reconstruction/local_dense accepted evidence unchanged
```

---

### Task 9: Write the final measured project report

**Files:**
- Create: `docs/geometry-ml/final-high-fidelity-presentation-model.md`
- Modify: `docs/geometry-ml/reference-assisted-presentation-model.md`

- [ ] **Step 1: Use `avoid-ai-writing` for prose cleanup after facts are written**

Do not let style cleanup change measurements or claims.

- [ ] **Step 2: Document the full CV-to-Blender chain**

Required sections:

```text
why V1 failed visually
source reference categories
CNN mask reuse
SIFT/RANSAC use
ALIKED/LightGlue/Step 13 camera use
multi-view profile fitting
Blender measured geometry workflow
ornament CV extraction/alignment
cleanup/sculpt/retopo decisions
UV/bake workflow
multi-view texture projection
material/lookdev
final photo-match metrics
export reimport verification
limitations and inference disclosure
```

- [ ] **Step 3: Clearly distinguish reconstructed vs inferred content**

Examples:

```text
camera geometry: measured Step 13 SfM evidence
whole-object masks: CNN + reviewed masks
profile geometry: CV-fitted, Blender revolved
unseen rotational surface: symmetry-completed
unseen repeated ornament: inference only where supported
surface cleanup/sculpt: Blender/manual, source-constrained
base color: multi-view projected/fused where coverage exists
unsupported texture gaps: disclosed fill/inference
```

---

### Task 10: Update authoritative project docs and instructions

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`
- Modify: `LESSONS.md`
- Modify: `docs/memory-bank/active-context.md`
- Modify: `docs/memory-bank/progress.md`

- [ ] **Step 1: Update status from V1 pre-clean prototype to V2 final asset**

Do not erase historical V1 outcome; mark it rejected/superseded for presentation.

- [ ] **Step 2: Update pipeline diagram**

Include V2 chain:

```text
CNN/SIFT/ALIKED-LightGlue/SfM cameras
→ CV multi-view fit
→ Blender V2 geometry
→ ornament CV alignment
→ UV/bake
→ photo projection
→ final QA/export
```

- [ ] **Step 3: Update next action**

After final V2 is complete, next action should be final coursework demo/report packaging, not more reconstruction experimentation unless a new requirement arises.

- [ ] **Step 4: Run documentation verification**

Check internal links, named files, metrics, stale V1 wording, and final artifact paths.

---

### Task 11: Perform final defect-focused diff/repository review

**Files:**
- Entire intended change set.

- [ ] **Step 1: Inspect Git status and complete diff**

Current pre-execution worktree already contains uncommitted V1 source/docs/artifacts. Treat them as user-owned/current-task state, not disposable junk.

Determine which of those are intended to publish together with V2 and which remain local-only.

- [ ] **Step 2: Exclude local/private/heavy paths**

Always exclude unless authoritative project instructions change:

```text
.codegraph/
analysis/ml/checkpoints/
reconstruction/local_dense/work/
reconstruction/reference_assisted_v2/work/   # unless a small accepted rollback is deliberately published; default local-only
```

Also exclude oversized/redundant intermediate meshes and temporary reference-preview planning files such as `.ai-bridge/plan-reference-previews/`.

- [ ] **Step 3: Check staged sizes and GitHub limits**

No staged file may exceed GitHub's normal 100 MiB per-file limit. Avoid redundant binary copies.

- [ ] **Step 4: Check for secrets/private state**

No checkpoints, tokens, credentials, or unrelated local metadata.

---

### Task 12: Commit, push, and verify remote synchronization

- [ ] **Step 1: Fetch origin and verify no upstream divergence**

If remote has new commits, inspect/reconcile normally before publication. Never force-push.

- [ ] **Step 2: Stage only intended files**

Use explicit paths or a reviewed staged set. Do not blindly `git add .`.

- [ ] **Step 3: Inspect staged diff/stat and binary set**

- [ ] **Step 4: Commit with an accurate subject**

Suggested subject only if it matches actual final diff:

```text
feat(reconstruction): complete CV-constrained final vessel model
```

- [ ] **Step 5: Push normally to `origin/main`**

- [ ] **Step 6: Verify all three revisions**

```text
local HEAD
origin/main after fetch
live remote main
```

Require exact equality.

- [ ] **Step 7: Verify post-push working tree**

Only intentional local-only paths may remain.

---

### Task 13: Final completion report to the user

The execution agent's report must include measured evidence, not generic claims.

Required sections:

```text
1. CV workflow contribution
   - CNN
   - SIFT/RANSAC
   - ALIKED/LightGlue + Step 13 camera geometry
   - multi-view fit
   - multi-view texture projection

2. Final geometry
   - component organization
   - normalized dimensions/proportions
   - topology counts/status

3. Ornament
   - real source motif families
   - inferred hidden repetitions
   - chain decision/evidence

4. Blender cleanup
   - defects found
   - exact fixes
   - before/after diagnostic paths

5. Textures/materials
   - projection coverage
   - inferred region percentage
   - texture resolutions
   - material setup

6. Fidelity
   - silhouette metrics
   - landmark metrics
   - visual-match ratings
   - QA verdict

7. Exports
   - .blend path/hash
   - .glb path/hash
   - GLB re-import result

8. Verification
   - focused tests
   - full project tests
   - integrated pipeline
   - protected evidence integrity

9. Git
   - commit hash
   - push verification

10. Remaining limitations
   - only real limitations, including relative scale if still unmeasured
```

**Final completion gate:** The task is not complete if the model merely opens and renders. It must receive `SHIP`, pass the source-photo identity gate, preserve the CV evidence chain, pass export re-import, and be published/verified as requested.
