# Claude Code Instructions

@AGENTS.md

`AGENTS.md` is the canonical project instruction source and must be read before modifying this repository.

## Current execution target

The project has a verified end-to-end V4 reconstruction baseline and is now in **V7 scan-preserving refinement and final documentation/publication maintenance**. Continue only from the newest verified repository/runtime state. The canonical V4 sparse, dense, Poisson, Blender, and GLB artifacts remain protected baselines unless a completed, independently verified V7 result is demonstrably stronger and explicitly promoted.

A bounded dense reconstruction diagnostic may be running from the V7 repair workspace. Never interrupt, overwrite, delete, or publish a live workspace merely because it appears incomplete. Inspect the process, logs, reports, and output hashes after completion before deciding whether any result is promotable.

Use `docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`, `docs/superpowers/plans/2026-09-14-v4-full-repair.md`, `docs/memory-bank/active-context.md`, and `docs/memory-bank/progress.md` as the detailed technical history. Public-facing project documentation is `README.md` and `docs/PROJECT_REPORT.md`.

## Mandatory local rules

- This must remain a full computer-vision reconstruction pipeline. Accepted geometry and appearance must be algorithmically derived from the project-captured images/reconstruction evidence; no external/reference-assisted modeling, manual sculpting/vertex editing, symmetry/lathe/primitive replacement, hand-built anatomy, manual reference matching, or hand-authored appearance fallback is allowed. Blender is limited to scripted/reproducible non-creative cleanup, LOD/UV/baking/material construction from measured project data, and export/verification.
- Preserve the canonical V4 output while V7 refinement is being evaluated. Promotion requires completed evidence, reproducible lineage, final asset verification, and exact hashes.
- Do not over-engineer, over-complicate, or over-test. Prefer the smallest coherent solution and verification proportional to risk.
- After a host restart, inspect persisted COLMAP/CUDA logs, workspaces, reports, and hashes before rerunning any interrupted dense job. Never infer completion from a previously launched process.
- Preserve `CSX4213_Project_V4_Images/` and `IMG20260826122949/` as immutable source data. Never delete, modify, rename, overwrite, crop, resize, rotate, recompress, or otherwise transform their original photographs.
- Never publish private checkpoints from `analysis/ml/checkpoints/`.
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
