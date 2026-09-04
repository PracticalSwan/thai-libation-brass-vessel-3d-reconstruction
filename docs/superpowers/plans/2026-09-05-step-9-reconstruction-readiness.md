# Step 9 Reconstruction Readiness Implementation Plan Index

> **For agentic workers:** This file is an index and final completion plan. Execute the four Step 9 sub-plans in order, then complete the final verification/documentation tasks below.

Updated: 2026-09-05

**Goal:** Complete all pre-reconstruction readiness work while preserving the verified source set and stopping before pyCOLMAP.

**Architecture:** Step 9 is split into four independently verifiable units: full-sequence masks, masked/unmasked geometric benchmarking, full-sequence connectivity/subset selection, and camera/EXIF readiness. A final summary stage combines their measured outputs without running reconstruction.

**Tech Stack:** Python, PyTorch, OpenCV, NumPy, Pillow, matplotlib, CSV/JSON, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-09-05-step-9-reconstruction-readiness-design.md`

## Execution order

1. `docs/superpowers/plans/2026-09-05-step-9a-full-mask-inference.md`
2. `docs/superpowers/plans/2026-09-05-step-9b-masked-unmasked-benchmark.md`
3. `docs/superpowers/plans/2026-09-05-step-9c-connectivity-and-subset.md`
4. `docs/superpowers/plans/2026-09-05-step-9d-camera-readiness.md`
5. Final summary, documentation, verification, cleanup, commit, and push.

## Hard boundary

Step 9 must not invoke pyCOLMAP/COLMAP, estimate camera poses, triangulate points, create a sparse/dense model, mesh, texture, or use Blender. It may only prepare and evaluate inputs/evidence for a later explicitly authorized reconstruction phase.

---

### Task 1: Final Step 9 summary stage

**Files:**
- Modify: `run_reconstruction_readiness.py`
- Create/modify: `tests/test_run_reconstruction_readiness.py`
- Generate: `analysis/reports/step9_summary.json`

**Interfaces:**
- `run_reconstruction_readiness.py --stage summary`
- `run_reconstruction_readiness.py --stage all` runs `masks -> benchmark -> connectivity -> camera -> summary` in that order only.

- [ ] **Step 1: Write failing summary-contract tests**

Tests must reject summary generation when any required stage report is absent or malformed, and must assert the summary contains:

```text
source/checkpoint provenance
mask count
chosen feature mode
benchmark aggregates
adjacent strong/weak counts
subset included/excluded counts
camera signature count/recommendation
reconstruction_started = false
```

- [ ] **Step 2: Run red tests**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_run_reconstruction_readiness.py
```

- [ ] **Step 3: Implement minimal summary/all orchestration**

The `all` stage may call only Step 9 stage functions. No pyCOLMAP imports or subprocess calls are permitted.

- [ ] **Step 4: Run focused tests green**

- [ ] **Step 5: Generate the real final summary**

```powershell
python -B run_reconstruction_readiness.py --stage summary
```

---

### Task 2: Write measured Step 9 documentation and update project state

**Files:**
- Create: `docs/geometry-ml/reconstruction-readiness.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/memory-bank/active-context.md`
- Modify: `docs/memory-bank/progress.md`
- Modify: `preprocessing/pycolmap_input/README.md`

- [ ] **Step 1: Read measured JSON/CSV outputs before editing documentation**

Do not write guessed result numbers.

- [ ] **Step 2: Document measured results**

`reconstruction-readiness.md` must include:

```text
288-mask inference provenance
cleanup rule and limitations
20-pair three-mode benchmark table/decision
full 287-edge connectivity result and skip-bridge evidence
recommended reconstruction-subset count and exact exclusion reasons
camera signature/EXIF evidence and intrinsics-group recommendation
source-integrity verification
explicit no-reconstruction boundary
```

- [ ] **Step 3: Update status docs only where stale**

Advance the project status to “Step 9 reconstruction readiness complete; next phase is separately authorized pyCOLMAP/SfM.” Do not claim the 3D reconstruction is improved or complete.

---

### Task 3: Final verification and visual review

**Files:**
- No production edits unless verification exposes a real defect.

- [ ] **Step 1: Run focused Step 9 tests**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_reconstruction_masks.py tests/test_reconstruction_matching.py tests/test_reconstruction_connectivity.py tests/test_camera_readiness.py tests/test_run_reconstruction_readiness.py
```

- [ ] **Step 2: Run full project tests**

```powershell
python -B -m pytest -p no:cacheprovider -q
```

- [ ] **Step 3: Compile changed Python**

```powershell
python -B -m py_compile reconstruction_masks.py reconstruction_matching.py camera_readiness.py run_reconstruction_readiness.py
```

- [ ] **Step 4: Verify derived mask contract**

Check exactly 288 raw full predictions and 288 reconstruction masks; each must be readable, 3072 x 4080, and contain only 0/255.

- [ ] **Step 5: Re-hash protected sources**

Use existing project verifiers to prove:

```text
297/297 raw unchanged, 0 mismatches
288/288 selected verified against selection_manifest.csv
```

- [ ] **Step 6: Visually inspect all Step 9 figures**

```text
step9_01_reconstruction_masks.png
step9_02_match_benchmark.png
step9_03_connectivity.png
step9_04_camera_readiness.png
```

Fix clipping, unreadable labels, misleading status language, or visible mask-cleanup defects; rerun affected stage and verification after any fix.

---

### Task 4: Bounded cleanup, Git review, commit, push, and remote verification

- [ ] **Step 1: Remove only task-created residue**

Remove `__pycache__/`, `.pytest_cache/`, temporary review exports, failed partial outputs, and superseded Step 9 scratch artifacts. Preserve final reports, masks, plans/spec, figures, manifests, and the local checkpoint.

- [ ] **Step 2: Inspect Git state and intended diff**

Confirm:

```text
no DOCX/PDF report modifications
no raw or selected JPEG modifications
no local checkpoint staged
no secrets/temp files staged
no reconstruction/pyCOLMAP artifacts
```

- [ ] **Step 3: Stage only intended Step 9 files**

- [ ] **Step 4: Commit with a scoped message**

Preferred message:

```text
feat(readiness): complete pre-reconstruction analysis
```

- [ ] **Step 5: Push `main` to `origin`**

No force push.

- [ ] **Step 6: Fetch and verify SHA equality**

Verify local `HEAD`, `origin/main`, and `git ls-remote origin refs/heads/main` are identical.

- [ ] **Step 7: Final status**

The only permitted intentional untracked content is the local model checkpoint directory if it remains excluded from publication by project policy.
