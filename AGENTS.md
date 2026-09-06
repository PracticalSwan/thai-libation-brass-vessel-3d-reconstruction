# Project Agent Instructions

## Scope and objective

This repository is the CSX4213 Computer Vision project for reconstructing a Thai brass libation vessel from smartphone photographs.

Current phase: preprocessing and Steps 6-17 are complete and verified. Step 13 ran the one approved external ALIKED-N16Rot + LightGlue recovery experiment on CUDA and produced a strongest single model registering 266/288 images with 29,713 points, but it failed the frozen >=274 global gate. Sparse recovery is closed and the 266-image model remains evidence only. Steps 14-17 used only the frozen Step 10 `reconstruction/sparse/best` fallback (73 images / 6,099 points): CUDA dense fusion produced 391,899 points, the accepted Poisson path was component-filtered and simplified to 499,999 faces, and COLMAP generated a topology-preserving photo texture with 73.00% meaningful UV coverage. Headless Blender validated the asset. `steps14_17_success=true` and `blender_manual_cleanup_started=false`; any manual cleanup or final presentation work requires a separate phase.

## Core rules

- Do not over-engineer, over-complicate, or over-test. Prefer the smallest coherent solution and verification proportional to risk.
- Preserve the student's existing structure and course explainability. Code must remain understandable enough to explain and reproduce for coursework assessment.
- Treat `IMG20260826122949/` as immutable raw data. Never overwrite, rename, resize, crop, rotate, recompress, normalize, or delete any original image.
- The user has authorized publishing the raw photographs and image-processing evidence to the public repository. Raw-file immutability still applies after publication.
- Preserve photogrammetric geometry. Preprocessing may change photometry mildly but must not crop, warp, perspective-correct, rotate, synthesize detail, remove reflections with AI, or otherwise move image features.
- Use most usable photographs. Reject only frames that meaningfully harm SfM; preserve angular coverage and overlap.
- Reflective brass highlights are expected. Do not reject images solely because they contain specular reflections.
- Use pyCOLMAP for the reconstruction stage rather than hiding the workflow behind the COLMAP GUI.
- Do not deploy anything unless deployment is explicitly required. This project is primarily local/offline coursework.

## Session startup

1. Read this file and `CLAUDE.md`.
2. Read `LESSONS.md` if present, then `docs/memory-bank/active-context.md` and `docs/memory-bank/progress.md` if present.
3. Inspect Git status and the files relevant to the requested task before editing.
4. Treat unrelated local changes as contributor-owned work.

## CodeGraph

- `.codegraph/` is installed project state. Use CodeGraph when it materially helps with dependencies, call paths, architecture, or change impact; use direct inspection for trivial edits.
- Preserve `.codegraph/` unless an explicit CodeGraph maintenance task requires changing or removing it.

## Skills, plugins, and subagents

- Use relevant installed skills and plugins automatically when they materially improve the task; do not add ceremony to simple work.
- Subagents may be used automatically when they materially reduce uncertainty or parallelize independent work.
- If using subagents, use only available `*-glm` / GLM-variant agents backed by GLM-5.3. Do not invoke non-GLM variants for this project.
- Prefer the GLM-5.3 1M-context variants for broad repository or dataset reasoning so the large context is used effectively rather than spawning many narrow agents.
- Good fits include `python-pro-glm`, `data-scientist-glm`, `machine-learning-engineer-glm`, `test-automator-glm`, and `code-reviewer-glm` when their scopes match.
- If the GLM request limit is reached, stop using subagents and continue with the parent agent only. Do not fall back to other subagent models.
- The parent agent must verify subagent claims from files, diffs, tests, or runtime evidence before accepting them.

## Preprocessing requirements

- Audit all raw images for readability, dimensions, EXIF, blur, exposure, contrast, clipping, duplicates, and useful local features.
- Base thresholds on the real dataset plus visual inspection, not old demonstration thresholds.
- Decisions should distinguish `ACCEPT`, `WARN`, and `REJECT`; warnings do not automatically become rejects.
- Compare representative RAW vs PREPROCESSED neighboring-frame SIFT matching before choosing the final reconstruction input variant.
- Produce explicit reports and a deterministic selected-image set for later pyCOLMAP use.
- Stop before pyCOLMAP unless the user explicitly continues to reconstruction after preprocessing is verified complete.

## Geometry and ML extension

The shared geometry/ML design is `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`. Steps 6-17 are implemented and verified to their approved boundaries:

- Step 6: `docs/superpowers/plans/2026-08-27-step-6-geometry-detection-analysis.md`.
- Steps 7+8: `docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`.
- Step 9: `docs/superpowers/plans/2026-09-05-step-9-reconstruction-readiness.md`; measured results are in `docs/geometry-ml/reconstruction-readiness.md`.
- Step 10: `docs/superpowers/plans/2026-09-05-step-10-sparse-sfm.md`; measured results are in `docs/geometry-ml/sparse-reconstruction.md`.
- Step 11: `docs/superpowers/plans/2026-09-05-step-11-sparse-component-bridging.md`; measured results are in `docs/geometry-ml/sparse-component-bridging.md`.
- Step 12: design `docs/superpowers/specs/2026-09-05-step-12-learned-sparse-recovery-design.md`; plan `docs/superpowers/plans/2026-09-05-step-12-learned-sparse-recovery.md`; measured capability boundary `docs/geometry-ml/learned-sparse-recovery.md`. The native learned path stopped at the missing-ONNX capability gate.
- Step 13: design `docs/superpowers/specs/2026-09-06-step-13-external-learned-global-recovery-design.md`; plan `docs/superpowers/plans/2026-09-06-step-13-external-learned-global-recovery.md`; measured result `docs/geometry-ml/external-learned-global-recovery.md`. The one external learned map reached 266/288 but failed the frozen >=274 gate, so Step 10 is the selected local fallback.
- Steps 14-17: design `docs/superpowers/specs/2026-09-06-steps-14-17-local-dense-mesh-texture-design.md`; plan `docs/superpowers/plans/2026-09-06-steps-14-17-local-dense-mesh-texture.md`; measured result `docs/geometry-ml/local-dense-mesh-texture.md`. The local CUDA dense, mesh, photo-texture, and headless Blender validation gates all passed.

- Step 6 exposes verified selected-image access, reusable SIFT keypoints/descriptors and scale metadata, Fundamental Matrix/RANSAC, epipolar geometry, and classical 2D vessel geometry.
- Steps 7+8 use a small project-defined binary segmentation CNN trained from random initialization; no pretrained backbone, SAM checkpoint, transfer learning, or external segmentation API is part of the verified baseline.
- The frozen labeled set contains 36 reviewed selected images: 24 train, 6 validation, 6 held-out test, split by separated capture positions/view groups rather than a random neighboring-frame shuffle.
- Model selection used training/validation evidence only. The held-out test split was evaluated after the model and 0.5 threshold were frozen.
- Step 8 reuses Step 6 SIFT extraction to measure features inside versus outside CNN-predicted vessel masks; it does not claim reconstruction improvement.
- Measured Steps 7+8 results are documented in `docs/geometry-ml/cnn-dataset.md` and `docs/geometry-ml/ml-results.md`.
- Step 9 ran the frozen CNN across all 288 selected images, benchmarked unmasked versus two masked SIFT modes, audited all 287 adjacent transitions, and audited raw EXIF for every selected filename.
- Step 9B measured `unmasked` SIFT as the reconstruction-readiness baseline: the masked modes retained only 90.31% of unmasked RANSAC inliers and failed the fixed 95% qualification floor.
- Step 9C conservatively keeps all 288 selected images because none of the 14 weak adjacent transitions had a strong local skip bridge that justified removing the middle frame.
- Step 9D measured one complete camera signature across all 288 selected frames, supporting one shared camera/intrinsics group as the starting recommendation for later validation.
- Step 10 used pyCOLMAP 4.2.0 with native unmasked SIFT, one shared `SIMPLE_RADIAL` camera, sequential matching, and incremental mapping. The selected baseline component registers 73/288 images with 6,099 points and 1.2373 px mean reprojection error.
- The single overlap-40 retry did not improve the largest-component registration count; Step 10 therefore records `acceptance_met=false` and preserves the disconnected sparse components instead of claiming global reconstruction success.
- Step 11 evaluated exactly 2,340 deterministic non-local pairs around boundaries 73-74, 145-146, and 203-204. The first two boundaries had zero geometrically verified candidates; only 203-204 produced qualified bridges, so targeted mapping was skipped by design.
- The one authorized CPU exhaustive fallback produced eight models with 224-image union coverage, but its strongest single model still registers 73/288 images with 3,443 points and 1.1989 px mean reprojection error. Step 11 records `bridge_success=false`.
- Weak CNN predictions must remain visible and documented; they must not be manually repaired and reported as model output.
- Course-presentation figures must come from real generated project outputs. Do not fabricate geometry, segmentation, training metrics, camera poses, point clouds, or reconstruction results.
- Step 12 stopped at the approved native capability gate because the installed pyCOLMAP wheel lacks the ONNX support required by ALIKED extraction. Step 13 was the separately authorized architecture change and is complete: one pinned external ALIKED + LightGlue frontend, one diagnostic, one mapping attempt, no retry/sweep/exhaustive fallback, and no dense API. Do not reopen sparse-recovery experimentation or change the >=274 acceptance gate after the result.
- Steps 14-17 are complete at the approved local boundary. Preserve the exact Step 10 source and all dense/mesh/texture attempt ledgers. Do not rerun PatchMatch, add mesh candidates, alter the deterministic component rule, replace the photo texture, or start manual Blender cleanup as an optimization sweep. The accepted final mesh is `reconstruction/local_dense/mesh/final_mesh.ply`; the accepted textured asset is under `reconstruction/local_dense/texture/attempt_1/`.

## Verification

For changed Python code, run the narrow relevant checks and then the real path:

- `python -m py_compile` for changed preprocessing scripts.
- Relevant tests only; do not inflate the suite without a real regression risk.
- Run the full preprocessing pipeline on the real capture set before claiming readiness.
- Confirm every selected derived output is readable and report/output counts agree.
- Re-hash the raw dataset and prove originals are unchanged.
- Visually inspect representative outputs and any rejected/warned outliers.
- Before commit/push, inspect `git status`, the intended diff, and staged files; exclude secrets, temporary/test junk, and unrelated work. Raw images and intentional image-processing evidence are allowed repository content.

## Removal and cleanup policy

Cleanup is required before task completion, but it must be bounded and evidence-based.

- Remove test residue and task-created temporary artifacts once they are no longer needed: `__pycache__/`, `.pytest_cache/`, temporary probes, scratch scripts, transient contact sheets/previews that are not intentional deliverables, debug logs, temporary exports, partial failed outputs, and stale generated files superseded by the verified pipeline.
- Remove obsolete demo-only files or references when the real workflow replaces them, provided they are inside this project and are not protected/user-authored material still needed for history or assessment.
- Do not retain duplicate generated outputs, abandoned experiments, temporary comparison folders, or tool-created residue merely because they are harmless.
- Never delete or modify `IMG20260826122949/` or its original image files. Preserve raw manifests/evidence needed to prove immutability.
- Never delete contributor work, source code, reports, documentation, Git history, or assets unless the target is clearly obsolete/task-created or the user explicitly authorizes removal.
- Before broad cleanup, inspect the exact removal set. Prefer exact paths over wildcards. After cleanup, verify required outputs still exist and Git status contains no unintended deletions.

## Git and collaboration

- Repository: `PracticalSwan/thai-libation-brass-vessel-3d-reconstruction`.
- Default branch: `main`.
- Raw photographs and intentional image-processing evidence are tracked in Git by user direction. Only secrets belong in `.gitignore`; cleanup policy handles disposable residue instead of hiding it indefinitely.
- `IMG20260826122949.zip` is a redundant local archive, not part of the requested raw-image/processing publication set. Leave it untracked unless the user separately requests an archive-publication method.
- Keep commits scoped and descriptive. Do not rewrite history or force-push without explicit authorization.
- Update `CHANGELOG.md` and `docs/memory-bank/` after meaningful verified milestones, not after trivial edits.

## Completion standard

A phase is complete only when the requested behavior is implemented, directly verified, temporary residue is cleaned, protected raw data is unchanged, documentation/memory reflects the real state, and Git/publication state is verified when publication was in scope.
