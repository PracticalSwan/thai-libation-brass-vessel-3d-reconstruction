# Real Preprocessing and Public Repository Implementation Plan

> **For agentic workers:** Execute task-by-task with tests first and verification checkpoints.

**Goal:** Produce and verify the real 297-image preprocessing/QA pipeline, then publish the source/documentation as a public collaborative GitHub repository.

**Architecture:** `quality_check.py` owns image metrics and decisions, `preprocess_images.py` owns conservative photometric normalization, and `run_preprocessing.py` orchestrates reports, outputs, previews, hashes, and matching checks. Repository-facing policy/docs remain separate from generated data.

**Tech Stack:** Python 3.14, OpenCV 4.13, NumPy, Pillow, pytest, Git/GitHub CLI, later pyCOLMAP.

**Spec:** `docs/superpowers/specs/2026-08-27-real-preprocessing-and-public-repo-design.md`

## Global Constraints

- Never modify `IMG20260826122949/` in place.
- Retain most usable views and preserve capture coverage.
- No geometric transformations in preprocessing.
- Thresholds must come from the real dataset, not the old demo.
- Stop before pyCOLMAP reconstruction.
- Do not publish the raw capture set or ZIP by default.
- Do not over-engineer, over-complicate, or over-test; keep verification proportional to risk.

---

### Task 1: Real capture audit

**Files:** Create `preprocessing/reports/dataset_audit.csv`, `preprocessing/reports/raw_manifest.json`, and temporary contact sheets.

- [ ] Compute hashes, readability, dimensions, EXIF orientation/camera metadata, luminance, contrast, clipping, blur, color statistics, and SIFT counts for all 297 files.
- [ ] Compute exact/near-duplicate candidates and neighboring-frame similarity.
- [ ] Generate contact sheets and visually review the full sequence plus metric outliers.
- [ ] Record dataset-relative thresholds and any manually justified rejects.

### Task 2: Quality metric behavior (TDD)

**Files:** Create `tests/test_quality_check.py`; create `quality_check.py` only after tests fail.

- [ ] Test that a synthetic sharp/textured image scores sharper than a Gaussian-blurred version.
- [ ] Test dark/bright clipping percentages on controlled images.
- [ ] Test that unreadable paths are reported rather than silently skipped.
- [ ] Test decision logic so warnings do not automatically become rejects.
- [ ] Run tests RED, implement minimum metric/decision code, then run GREEN.

### Task 3: Conservative preprocessing (TDD)

**Files:** Create `tests/test_preprocess_images.py`; replace stale `preprocess_images.py` after tests fail.

- [ ] Test that output dimensions are identical to input dimensions.
- [ ] Test that mild normalization changes photometry without cropping/rotation/warping.
- [ ] Test deterministic output and valid uint8 BGR data.
- [ ] Run RED, implement dataset-calibrated mild LAB/white-balance/CLAHE operations, then run GREEN.

### Task 4: Orchestrator, reports, previews, and feature-match comparison

**Files:** Create `run_preprocessing.py`, `tests/test_run_preprocessing.py`, reports/previews/output directories.

- [ ] Test manifest/report count consistency and immutable raw hashes.
- [ ] Test accepted-output naming and readability.
- [ ] Implement one-command orchestration over the real source folder.
- [ ] Generate representative Original | Preprocessed preview composites.
- [ ] Compare representative neighboring RAW/PREPROCESSED SIFT+ratio-test matches and document the chosen reconstruction input variant.

### Task 5: Full preprocessing verification

- [ ] Run `python -m py_compile quality_check.py preprocess_images.py run_preprocessing.py`.
- [ ] Run `pytest -q`.
- [ ] Run the full pipeline on all 297 photographs.
- [ ] Verify input/report/output counts and zero raw hash changes.
- [ ] Open contact sheets and representative outputs for visual inspection.
- [ ] Remove only task-created temporary residue.

### Task 6: GitHub-facing project structure

**Files:** Create/update `README.md`, `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `.gitignore`, `AGENTS.md`, `CLAUDE.md`, `CHANGELOG.md`, `docs/memory-bank/*`.

- [ ] Rewrite the stale web-demo README around the real project and future pyCOLMAP/Blender pipeline.
- [ ] Add MIT license with Sithu Win San, Eaint Myat Thu, and Gulizara Benjapalaporn as 2026 copyright holders.
- [ ] Add security/reporting guidance appropriate for a student CV project.
- [ ] Add contribution workflow and project-local AI-agent constraints.
- [ ] Ensure raw data, ZIPs, generated reconstruction outputs, caches, and secrets are ignored.
- [ ] Update Serena and fallback memory after verified work.

### Task 7: Initialize and publish repository

- [ ] Verify Git/GitHub authentication and repository name availability.
- [ ] Initialize Git in `Project`, set default branch `main`, inspect intended status/diff, and stage only intended files.
- [ ] Create the public repository `thai-libation-brass-vessel-3d-reconstruction` with the documented description.
- [ ] Commit and push the verified project files.
- [ ] Verify repository metadata and files from GitHub after push.
- [x] Collaborator invitations are owner-managed and out of scope for agents; do not invite collaborators.
- [ ] Do not start pyCOLMAP yet.

