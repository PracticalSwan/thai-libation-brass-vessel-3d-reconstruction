# Step 9D Camera and EXIF Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit camera metadata for all 288 selected filenames from the immutable raw JPEGs and produce an evidence-backed recommendation for later pyCOLMAP camera/intrinsics grouping without calibrating or modifying images.

**Architecture:** Add `camera_readiness.py` to read top-level TIFF EXIF plus the nested EXIF IFD, normalize focal/zoom values, build camera signatures, and summarize whether one shared camera group is justified. The Step 9 orchestrator writes detailed CSV/JSON reports and one presentation summary.

**Tech Stack:** Python, Pillow, CSV/JSON, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-step-9-reconstruction-readiness-design.md`

## Global Constraints

- Read EXIF only from immutable raw JPEGs corresponding by filename to the selected set.
- Do not write EXIF, undistort, resample, calibrate, crop, or recompress any photograph.
- Missing values remain missing; never guess focal length, lens, or zoom.
- Recommend a shared camera group only from consistent measured metadata.
- Do not run pyCOLMAP.

---

### Task 1: Read nested EXIF fields safely

**Files:**
- Create: `camera_readiness.py`
- Create: `tests/test_camera_readiness.py`

**Interfaces:**
- Produces: `CameraRecord`
- Produces: `read_camera_record(selected_index: int, raw_path: Path) -> CameraRecord`

- [ ] **Step 1: Write failing EXIF tests**

Use a small temporary JPEG with synthetic EXIF where practical and a missing-EXIF JPEG. Verify normalized fields are returned without mutation and missing fields become `None`.

Required fields:

```text
selected_index
filename
width
height
orientation
make
model
lens_model
focal_length_mm
focal_length_35mm
digital_zoom_ratio
datetime_original
```

- [ ] **Step 2: Run red test**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_camera_readiness.py
```

Expected: import/function failure.

- [ ] **Step 3: Implement EXIF parsing**

Read `Image.getexif()` plus `get_ifd(0x8769)` for nested EXIF. Convert Pillow rational values to finite floats when possible; preserve strings exactly after trimming.

- [ ] **Step 4: Run green test**

---

### Task 2: Build camera signatures and grouping recommendation

**Files:**
- Modify: `camera_readiness.py`
- Modify: `tests/test_camera_readiness.py`

**Interfaces:**
- Produces: `camera_signature(record: CameraRecord) -> tuple[object, ...]`
- Produces: `summarize_camera_records(records: Sequence[CameraRecord]) -> dict[str, object]`

- [ ] **Step 1: Write failing grouping tests**

Cases:

```text
same dimensions/orientation/make/model/lens/focal/35mm/zoom across all records -> one shared camera group recommended
focal or digital zoom differs -> separate camera groups recommended
missing focal metadata -> report missing count and do not invent a value
mixed orientation or geometry -> separate groups recommended
```

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement exact signature/grouping rule**

Signature fields:

```text
(width, height, orientation, make, model, lens_model,
 focal_length_mm, focal_length_35mm, digital_zoom_ratio)
```

Summary includes unique signature count, missing-field counts, per-signature frame counts, and `camera_group_recommendation`.

- [ ] **Step 4: Run green tests**

---

### Task 3: Run real 288-image camera audit and generate evidence

**Files:**
- Modify: `run_reconstruction_readiness.py`
- Generate: `analysis/reports/step9_camera_readiness.csv`
- Generate: `analysis/reports/step9_camera_readiness.json`
- Generate: `analysis/previews/presentation/step9_04_camera_readiness.png`

**Interfaces:**
- `run_reconstruction_readiness.py --stage camera`

- [ ] **Step 1: Add stage-contract test**

Verify that every selected filename must exist in raw, and that record count must equal selected-manifest count.

- [ ] **Step 2: Run red test**

- [ ] **Step 3: Implement camera stage**

Read all 288 selected filenames from raw, write one CSV row per selected image, then write aggregate JSON and a simple figure showing camera-group count, focal/zoom consistency, orientation/geometry consistency, and the recommendation.

- [ ] **Step 4: Run focused tests green**

- [ ] **Step 5: Run real camera stage**

```powershell
python -B run_reconstruction_readiness.py --stage camera
```

Acceptance:

```text
288 camera rows
raw filename mapping complete
aggregate recommendation derived only from measured signatures
no source image modification
```

- [ ] **Step 6: Visually inspect `step9_04_camera_readiness.png`**

Ensure missing metadata, if any, is visible rather than silently omitted.
