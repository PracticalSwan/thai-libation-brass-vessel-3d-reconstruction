# Final High-Fidelity Thai Libation Vessel Implementation Plan

> **For agentic workers:** Execute this plan and its linked subplans task-by-task. Use any relevant installed skills/plugins. Subagents are optional but must be GLM variants only; never fall back to non-GLM agents. If an installed plan-execution workflow is available, use it; otherwise follow these files directly. Current deadline mode is one continuous implementation through a complete user-review-ready Blender model: finish/freeze Plan 1, continue immediately through Blender Plans 2-5 and pre-export final validation, then STOP before export so the user can inspect and request fixes. Export/publication resumes only after explicit user approval.

**Goal:** Replace the visually rejected V1 presentation asset with a complete V2 model that matches the user's photographed Thai libation brass vessel and is demonstrably derived from the project's computer-vision pipeline before Blender finishing.

**Architecture:** V2 is a separate CV-constrained presentation reconstruction. Existing CNN masks, SIFT/RANSAC geometry, Step 13 ALIKED/LightGlue camera recovery, and multi-view source photographs feed a deterministic evidence/model-fit stage; Blender consumes those fitted profiles/cameras to create, clean, texture, validate, and export the final asset. V1 and Steps 1-17 remain preserved as evidence.

**Tech Stack:** Python 3.x, OpenCV, NumPy, pyCOLMAP 4.2, existing PyTorch `SmallSegCNN`, existing ALIKED/LightGlue stack, Blender 5.2 Python plus available Blender integration/plugins, pytest.

**Spec:** `docs/superpowers/specs/2026-09-06-final-high-fidelity-vessel-design.md`

## Global Constraints

- Current baseline: Steps 1-17 are complete/verified at their approved boundaries and V1 remains the visually rejected prototype. V2 Plan 1 is accepted and frozen for downstream use: median silhouette IoU 0.901495, minimum reliable-view IoU 0.856102, median landmark error 0.011306, p95 landmark error 0.038080, with the candidate-bound bowl/globe/neck/lid/finial visual review accepted. Continue from Plan 2 unless later Blender evidence proves the accepted CV contract wrong.
- The current worktree contains existing V1 code/assets/docs plus this plan set. Treat that state as contributor-owned current work; do not discard it while starting V2.
- `IMG20260826122949/` and `preprocessing/pycolmap_input/images/` are immutable.
- Do not reopen sparse recovery, retrain the existing segmentation CNN, or rerun dense photogrammetry merely to improve presentation.
- Step 13 camera poses/intrinsics may constrain V2; Step 13 sparse points are not final surface truth.
- V1 `reconstruction/reference_assisted/` is preserved as a rejected prototype until V2 passes all final gates.
- V2 must materially reuse CNN segmentation, SIFT/RANSAC, and Step 13 ALIKED/LightGlue/SfM evidence.
- No unsupported V1 chain or generic oval/diamond ornament may be copied into V2.
- No physical centimeters/millimeters may be claimed without a real measurement.
- Blender work must use relevant installed Blender skills/plugins. Use Blender MCP or an equivalent available Blender automation/inspection interface where useful; do not make the implementation depend on one harness-specific wrapper.
- Use any relevant skills/plugins automatically when they improve fidelity, verification, or implementation quality. If subagents are used, use GLM variants only and verify their conclusions in the parent agent.
- Quality outranks speed. Use as much available compute/resource budget as is materially useful for CV fitting, rendering, baking, projection, optimization, visual comparison, and QA. Avoid only undirected waste or blind sweeps.
- Failure of an intermediate technique is not permission to accept a weak final asset. Diagnose, adapt, and continue until the final gates pass or a genuine external blocker is proven.
- Deadline execution: once a gate is accepted, move immediately to the next Blender stage; do not spend another 5-hour window re-optimizing an accepted upstream stage unless a downstream defect proves it wrong.
- Current pre-export delivery requires a fully constructed, detailed, cleaned, UV/baked, textured/materialed and screenshot-verified Blender model ready for user inspection. Stop after `final-validate`; GLB export/re-import, final promotion/publication and final-completion Git publication are deferred until explicit user approval.
- Preserve CV continuity after Plan 1: exact metrics use the Step 13 `SIMPLE_RADIAL` camera model or explicitly matched undistorted derived views, never an unqualified Blender pinhole approximation against raw photos.
- During Plan 2, run a low-cost registered-view coverage audit outside the 16 canonical fit views using Step 13 cameras and existing masks. Treat it as a cross-check rather than a second optimization set; inspect the worst views and route genuine model mismatch back to geometry instead of deforming the model around bad masks.
- Maintain `surface_evidence_coverage.json` from Plan 2 onward, classifying major component regions as direct multi-view, reviewed/detail-only, symmetry/repetition inferred, or hidden generic fill. Plan 3 ornament, Plan 5 texture filling, and Plan 6 final QA must consume this provenance.
- Use Step 6 classical edge/contour/PCA/ellipse measurements as independent diagnostics for circular rims/rings and axis consistency where reliable; do not add new NeRF/3DGS/retraining/sparse-recovery branches merely to finish the deadline model.

---

## Plan sequence

The execution order is fixed because each plan has a hard gate consumed by the next.

### Plan 1 — CV evidence and multi-view fit

Read and execute:

`docs/superpowers/plans/2026-09-06-final-cv-evidence-and-measurement.md`

Produces:

```text
reconstruction/reference_assisted_v2/evidence/**
reconstruction/reference_assisted_v2/reports/final_reference_evidence.json
reconstruction/reference_assisted_v2/reports/final_cv_fit.json
reconstruction/reference_assisted_v2/reports/final_profiles.json
```

Hard gate: CV evidence complete, canonical cameras established, profiles fitted, geometry target metrics defined.

### Plan 2 — CV-constrained Blender base geometry

Read and execute:

`docs/superpowers/plans/2026-09-06-final-cv-constrained-blender-geometry.md`

Produces a measured, unornamented V2 base mesh, distortion-consistent canonical camera diagnostics, a non-canonical registered-view coverage audit, and `surface_evidence_coverage.json` describing direct versus inferred surface support.

Hard gate:

```text
median silhouette IoU >= 0.90
minimum reliable-view IoU >= 0.84
median landmark error <= 2.0% object height
95th percentile landmark error <= 4.0% object height
```

A visible component mismatch vetoes the numeric pass. The non-canonical registered-view coverage audit adds a generalization veto: repeated usable-view `model_mismatch` also blocks Plan 2 even if the 16 canonical metrics pass. These are minimum floors, not quality targets; continue evidence-directed correction when it materially improves the same-object match.

### Plan 3 — Ornament and photo-detail reconstruction

Read and execute:

`docs/superpowers/plans/2026-09-06-final-ornament-and-photo-detail.md`

Produces source-traceable ornament high detail and photo-derived surface-detail sources.

Hard gate: hero motif families match photographed motifs; no generic floating decoration remains.

### Plan 4 — Blender cleanup, sculpting, retopology, UV and bake

Read and execute:

`docs/superpowers/plans/2026-09-06-final-blender-cleanup-retopology-uv.md`

Produces the clean final presentation topology, production UVs, and baked detail maps.

Hard gate: topology/normal/intersection/UV/bake checks pass with diagnostic images.

### Plan 5 — Multi-view texture projection, materials, lookdev and rendering

Read and execute:

`docs/superpowers/plans/2026-09-06-final-material-lookdev-rendering.md`

Produces overlap-normalized and robustly fused object-specific brass textures/materials, neutral diagnostic lookdev, beauty scene, reference comparisons, and final renders. Photometric normalization is bounded and evidence-based so exposure/white-balance differences are reduced without treating moving brass highlights as albedo.

Hard gate: material/lighting/reference-match review passes; the model looks like the photographed object rather than a generic vessel.

### Plan 6 — Final validation, export, documentation and publication

Read and execute:

`docs/superpowers/plans/2026-09-06-final-validation-export-publication.md`

Produces canonical `.blend`, `.glb`, final reports, fresh re-import evidence, updated docs, commit, push, and verified remote synchronization.

Hard gate: `qa-review` verdict `SHIP`, GLB re-import passes, full relevant tests pass, Git state is synchronized.

---

## Cross-plan state machine

`run_final_model.py` must eventually expose this stage order:

```text
analyze
  ↓
cv-fit
  ↓
base
  ↓
geometry-validate
  ↓
ornament
  ↓
cleanup
  ↓
uv-bake
  ↓
lookdev
  ↓
final-validate
  ↓
export
```

Rules:

- a failed stage does not fabricate downstream artifacts;
- restart reads reports/hashes and reuses accepted upstream work;
- stage-owned work paths are explicit;
- promotion to `final/` happens only after final validation;
- `work/` may be deleted only through an exact allowlist after successful promotion;
- `NO-SHIP` routes execution back to the responsible earlier stage for diagnosis/correction rather than terminating as the final project state.

## Required Blender skill sequence

Before relevant phases, load the actual installed skills rather than assuming their content:

```text
blender-director
prop-artist
blender-modeler
realistic-style
sculpting
retopology
uv-workflow
texture-workflow
materials
lookdev
camera-cinematography
lighting
rendering
asset-optimization
export-pipeline
qa-review
```

Also use the Blender Director reference-photo workflow and visual-match checklist bundled with the installed Blender skills.

Use additional relevant skills/plugins when they improve the result. GLM-only subagents may independently review source selection, CV fits, Blender geometry, ornament, textures, renders, tests, or code, but the parent agent must integrate and verify all findings.

## Final completion definition

The entire plan set is complete only when all are true:

- the final model is recognizably the same photographed object from multiple views;
- CV evidence is materially used in geometry, camera alignment, texture/detail reconstruction, and validation;
- no protected Steps 1-17 evidence was rewritten;
- the V1 rejection remains documented;
- all logical final parts are clean and intentional;
- final textures/materials reproduce the photographed brass and ornament credibly;
- final source-vs-render panels pass the visual gate;
- `Thai_Libation_Vessel_FINAL.blend` and `.glb` are verified by fresh opening/re-import;
- relevant test suite passes;
- documentation truthfully separates direct photogrammetry from CV-constrained Blender completion;
- intended files are committed and pushed normally to `origin/main`;
- local `HEAD == origin/main == remote main` after fetch/push verification.

`NO-SHIP`, an attractive generic vessel, or a technically clean model that does not look like the user's photographed object is not completion. Iterate back to the failing plan and continue unless a genuine external blocker prevents further progress. Never fabricate success to satisfy the completion target.
