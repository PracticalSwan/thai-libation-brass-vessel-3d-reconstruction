# Claude Code Instructions

@AGENTS.md

`AGENTS.md` is the canonical project instruction source and must be read before modifying this repository.

## Current execution target

The project has a verified end-to-end V4 reconstruction and a completed **V7 scan-preserving final release**. Continue only from the newest verified repository state. The selected sparse/dense/Poisson artifacts, V139 authoring master, canonical Blender/GLB assets, final report, and user-approved professor submission package are the protected finished state.

Do not resume reconstruction diagnostics, candidate generation, Blender repair, material authoring, or export work unless the user explicitly reopens that scope. Finished-project maintenance should be limited to evidence-backed cleanup, documentation synchronization, integrity verification, and publication.

Use `docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`, `docs/superpowers/plans/2026-09-14-v4-full-repair.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md` as the detailed technical history. Public-facing project documentation is `README.md` and `docs/PROJECT_REPORT.md`.

## Mandatory local rules

- This must remain a full computer-vision reconstruction pipeline. Accepted geometry and appearance must be algorithmically derived from the project-captured images/reconstruction evidence; no external/reference-assisted modeling, manual sculpting/vertex editing, symmetry/lathe/primitive replacement, hand-built anatomy, manual reference matching, or hand-authored appearance fallback is allowed. Blender is limited to scripted/reproducible non-creative cleanup, LOD/UV/baking/material construction from measured project data, and export/verification.
- Preserve the canonical V4 output while V7 refinement is being evaluated. Promotion requires completed evidence, reproducible lineage, final asset verification, and exact hashes.
- Do not over-engineer, over-complicate, or over-test. Prefer the smallest coherent solution and verification proportional to risk.
- After a host restart, inspect persisted COLMAP/CUDA logs, workspaces, reports, and hashes before rerunning any interrupted dense job. Never infer completion from a previously launched process.
- Preserve `CSX4213_Project_V4_Images/` as the only raw source set. Never delete, modify, rename, overwrite, crop, resize, rotate, recompress, or otherwise transform its 688 original photographs.
- Treat `capture_v4/derived/mvs_images/`, `capture_v4/derived/masks/`, and `capture_v4/derived/feature_masks/` as the only current processed-image sets. They are intentional Git LFS content; pre-V4 image data/checkpoints have been removed and must not be restored.
- Do not deploy unless deployment is explicitly required.
- Use relevant installed skills and plugins automatically when they materially improve the task.
- `.codegraph/` is installed project state. Use CodeGraph for dependency/call-path/change-impact questions when useful, and preserve it unless explicit maintenance requires otherwise.
- Do not use Codex CLI.
- Subagents may be used automatically when useful, but only available GLM-5.3 `*-glm` variants are allowed. Use the 1M context effectively instead of spawning unnecessary agents; if GLM request limits are reached, stop using subagents entirely rather than switching models.
- The parent agent must verify subagent work from files, diffs, tests, or runtime evidence before accepting it.

## Documentation and publication rules

- Keep `README.md` and `docs/PROJECT_REPORT.md` focused on the project, dataset, computer vision methods, outputs, validation, and reproducibility. Do not include implementation-assistant, agent, orchestration, or internal workflow references in those two public-facing documents.
- Historical plans/specifications may retain detailed engineering history, but current-status docs must clearly distinguish the verified canonical baseline from in-progress refinement.
- Before a public Git milestone, exclude `.ai-bridge/`, `.codegraph/`, private checkpoints, caches, transient runtime logs, scratch workspaces, and incomplete process outputs.

## Removal and cleanup policy

Before publication, inspect and remove only task-created residue, temporary probes, scratch scripts/files, debug logs, failed or partial outputs, stale generated outputs, duplicate outputs, abandoned experiment folders, `__pycache__/`, `.pytest_cache/`, and other temporary artifacts that no longer support a verified deliverable. Do not delete a live reconstruction workspace or output while its owning process is still running.

Do not remove contributor-owned work, valid source code, required reports, documentation, Git history, or assessment assets merely to make the tree cleaner. Inspect cleanup targets first, use exact-path bounded cleanup, and verify Git status afterward so required files remain and no unintended deletions occurred.

## V7 Blender policy override — 2026-09-16

For the V7 authoring pass, the user explicitly permits localized merge/cleanup/sculpt/smooth operations on the reconstructed scan mesh and same-object donor geometry. This does **not** authorize reference-built replacement anatomy, profile/lathe reconstruction, primitives, or generic decoration. The V113/V114 rebuilt-looking direction is rejected. The completed V7 final is `SM_V4_V139_SCAN_PRESERVING_FINAL`, with versioned master `Thai_Libation_Vessel_V4_V7_SCAN_PRESERVING_FINAL_V139.blend`. It remains V94-derived, preserves the real 72-edge bottom opening, uses guarded CleanHigh/same-scan evidence, and finishes with bounded full-surface fairing plus existing-vertex terminal repair. It uses `MAT_V4_V115_BrassStatsOnly`. V139 has been promoted to the canonical V4 `.blend` and GLB and verified by a fresh Blender 5.2 GLB import. Treat those canonical files as the completed release unless a later user request explicitly reopens geometry or export work.
