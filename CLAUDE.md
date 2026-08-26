# Claude Code Instructions

@AGENTS.md

`AGENTS.md` is the canonical project instruction source and must be read before modifying this repository.

## Mandatory local rules

- Do not over-engineer, over-complicate, or over-test. Prefer the smallest coherent solution and verification proportional to risk.
- Preserve `IMG20260826122949/` as immutable raw source data. Never delete, modify, rename, overwrite, crop, resize, rotate, recompress, or otherwise transform its original photographs.
- Do not deploy unless deployment is explicitly required.
- For subagents, use only available GLM-5.3 `*-glm` variants. Use the 1M context effectively instead of spawning unnecessary agents; if GLM request limits are reached, stop using subagents entirely rather than switching models.
- The parent agent must verify subagent work from files, diffs, tests, or runtime evidence before accepting it.

## Removal and cleanup policy

Before task completion, inspect and remove task-created test residue, temporary probes, scratch scripts/files, debug logs, failed or partial outputs, stale generated outputs, duplicate outputs, abandoned experiment folders, `__pycache__/`, `.pytest_cache/`, and other temporary artifacts that no longer support a verified deliverable. Remove obsolete demo material when it is genuinely superseded and no longer needed.

Do not remove contributor-owned work, valid source code, required reports, documentation, Git history, or assessment assets merely to make the tree cleaner. Never delete, modify, overwrite, rename, crop, resize, rotate, recompress, or otherwise transform the original files in `IMG20260826122949/`. Inspect cleanup targets first, use exact-path bounded cleanup, and verify Git status afterward so required files remain and no unintended deletions occurred.
