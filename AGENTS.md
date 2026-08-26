# Project Agent Instructions

## Scope and objective

This repository is the CSX4213 Computer Vision project for reconstructing a Thai brass libation vessel from smartphone photographs.

Current phase: finish image QA/preprocessing and prepare a photogrammetry-ready image set. Do not start pyCOLMAP until the preprocessing completion criteria are satisfied.

## Core rules

- Do not over-engineer, over-complicate, or over-test. Prefer the smallest coherent solution and verification proportional to risk.
- Preserve the student's existing structure and course explainability. Code must remain understandable enough to demonstrate to the professor.
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

## Subagents

- Use subagents only when they materially reduce uncertainty or parallelize independent work.
- If using Codex subagents, use only available `*-glm` / GLM-variant agents backed by GLM-5.3. Do not invoke non-GLM variants for this project.
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
