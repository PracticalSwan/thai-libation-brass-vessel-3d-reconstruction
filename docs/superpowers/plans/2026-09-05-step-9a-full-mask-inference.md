# Step 9A Full-Sequence CNN Mask Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen SmallSegCNN over all 288 verified selected images, preserve unedited predictions, derive conservative reconstruction masks, and publish a complete mask manifest plus visual evidence.

**Architecture:** Add a focused `reconstruction_masks.py` module for checkpoint/source verification, full-sequence inference, deterministic connected-component cleanup, and mask-manifest generation. `run_reconstruction_readiness.py` will invoke this stage but will not train or modify the model.

**Tech Stack:** Python 3.14, PyTorch, OpenCV, NumPy, CSV, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-step-9-reconstruction-readiness-design.md`

## Global Constraints

- Never modify `IMG20260826122949/` or `preprocessing/pycolmap_input/images/`.
- Load only the frozen local `best_small_seg_cnn.pt`; no retraining or threshold tuning.
- Preserve raw CNN predictions separately from reconstruction masks.
- Reconstruction cleanup is deterministic and fixed before geometric benchmarking.
- No hole filling, erosion, warping, pyCOLMAP, or reconstruction.

---

### Task 1: Define and test deterministic reconstruction-mask cleanup

**Files:**
- Create: `tests/test_reconstruction_masks.py`
- Create: `reconstruction_masks.py`

**Interfaces:**
- Produces: `cleanup_reconstruction_mask(mask: np.ndarray) -> np.ndarray`
- Produces: `restore_binary_mask(mask: np.ndarray, source_size: tuple[int, int]) -> np.ndarray`

- [ ] **Step 1: Write failing cleanup tests**

```python
def test_cleanup_keeps_central_vessel_and_removes_detached_border_blob():
    mask = np.zeros((100, 80), np.uint8)
    mask[20:85, 25:55] = 255
    mask[5:25, 0:10] = 255
    cleaned = cleanup_reconstruction_mask(mask)
    assert cleaned[50, 40] == 255
    assert cleaned[10, 5] == 0


def test_cleanup_preserves_hole_inside_anchor_component():
    mask = np.zeros((100, 80), np.uint8)
    mask[20:85, 20:60] = 255
    mask[40:55, 30:50] = 0
    cleaned = cleanup_reconstruction_mask(mask)
    assert cleaned[45, 40] == 0
```

- [ ] **Step 2: Run red test**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_reconstruction_masks.py
```

Expected: import/function failure because `reconstruction_masks.py` does not yet exist.

- [ ] **Step 3: Implement minimal cleanup**

Implementation requirements:

```text
central ROI = x 25%-75%, y 20%-85%
anchor = largest component intersecting ROI, else largest foreground component
secondary minimum area = max(8 pixels, 2% of anchor area)
secondary maximum bbox gap = 3% of model-space diagonal
keep anchor + qualifying nearby secondaries
preserve zeros/holes inside retained components
return uint8 0/255 only
```

- [ ] **Step 4: Run green test**

Run the same pytest command and require PASS.

---

### Task 2: Verify frozen checkpoint and infer all selected records

**Files:**
- Modify: `reconstruction_masks.py`
- Modify: `tests/test_reconstruction_masks.py`

**Interfaces:**
- Produces: `verify_reconstruction_checkpoint(checkpoint: dict, manifest_sha256: str) -> None`
- Produces: `infer_selected_masks(...) -> list[MaskRecord]`
- `MaskRecord` contains selected index, filename, raw prediction path/hash, reconstruction mask path/hash, source width/height, and foreground fractions.

- [ ] **Step 1: Add failing provenance/output-safety tests**

Tests must reject:

```text
wrong model name
pretrained_weights != False
random_initialization != True
segmentation manifest hash mismatch
threshold != 0.5
output directory overlapping raw or selected source directories
```

- [ ] **Step 2: Verify the tests fail for the missing behavior**

Run only `tests/test_reconstruction_masks.py` and confirm expected failures.

- [ ] **Step 3: Implement checkpoint verification and source-size inference**

Requirements:

```text
load selected records through existing verified manifest APIs
resize only in memory to checkpoint input geometry
normalize exactly as Steps 7+8
save raw model prediction to analysis/ml/full_predictions/
cleanup in model space, then nearest-neighbor restore to source size
save reconstruction mask to analysis/ml/reconstruction_masks/
write no files inside raw/selected directories
```

- [ ] **Step 4: Run focused tests**

Require all Step 9A tests to pass.

---

### Task 3: Generate real 288-image masks, manifest, and visual QA sheet

**Files:**
- Create/modify: `run_reconstruction_readiness.py`
- Generate: `analysis/reports/reconstruction_mask_manifest.csv`
- Generate: `analysis/previews/presentation/step9_01_reconstruction_masks.png`

**Interfaces:**
- `run_reconstruction_readiness.py --stage masks`

- [ ] **Step 1: Add a stage-level test**

The test uses a tiny synthetic selected dataset/checkpoint fixture and verifies that the stage writes one raw prediction and one reconstruction mask per input record without touching source files.

- [ ] **Step 2: Run the stage test red**

Expected: failure because the stage/orchestrator does not yet exist.

- [ ] **Step 3: Implement the bounded `masks` stage and contact sheet**

Contact-sheet sample must include index 72 and deterministic samples spanning the full sequence. Each tile shows original, raw CNN overlay, reconstruction-mask overlay, and raw/cleaned foreground fractions.

- [ ] **Step 4: Run focused tests green**

- [ ] **Step 5: Run the real masks stage**

```powershell
python -B run_reconstruction_readiness.py --stage masks
```

Acceptance:

```text
288 raw predictions
288 reconstruction masks
all PNGs source-size and binary 0/255
manifest has 288 rows
held-out Step 7/8 prediction directory remains unchanged
```

- [ ] **Step 6: Visually inspect `step9_01_reconstruction_masks.png`**

Check normal side, low-angle, elevated, top-down, detail, and known index-72 failure. If cleanup introduces a structural defect, fix only the deterministic cleanup rule, rerun 9A, and document the change before Step 9B begins.
