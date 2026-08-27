# Geometry Detection and ML Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three explicit geometry-analysis demonstrations and SAM 2.1 vessel segmentation, create clear course-presentation figures from real project images, and later compare unmasked versus ML-mask-assisted pyCOLMAP reconstruction without modifying the verified 288-image input set.

**Architecture:** Keep geometry, 2D shape analysis, ML segmentation, and orchestration as separate modules. Geometry/ML analysis writes only to a new `analysis/` tree. SAM 2 masks are derived annotations consumed by pyCOLMAP through `ImageReader.mask_path`; they never replace JPEG inputs. The first implementation phase stops before pyCOLMAP. A later separately authorized phase performs the baseline-versus-mask reconstruction experiment.

**Tech Stack:** Existing Python/OpenCV/NumPy/Pillow stack for geometry; Meta SAM 2.1 in an isolated compatible PyTorch environment for ML segmentation; later pyCOLMAP/COLMAP for reconstruction comparison.

**Spec:** `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

## Global Constraints

- `IMG20260826122949/` remains immutable.
- `preprocessing/pycolmap_input/images/` remains the verified 288-image reconstruction-input set and must not be overwritten.
- No crop, resize, rotate, warp, perspective correction, generative edit, or artificial detail insertion.
- Geometry modules are measurements/visualizations, not preprocessing transforms.
- Start ML with `sam2.1_hiera_small`; move to a larger checkpoint only if representative-mask evidence justifies it.
- Keep SAM 2/PyTorch dependencies isolated from the verified preprocessing environment until compatibility is proven.
- Never commit model checkpoints, caches, or downloaded weights.
- A failed ML mask must fall back to a full-white mask for that image rather than removing the image.
- Do not claim ML improves reconstruction until a controlled pyCOLMAP comparison is run.
- Phase A below may be implemented without pyCOLMAP. Phase B requires a separate explicit reconstruction authorization.
- Do not over-engineer or add additional ML models unless a measured failure requires them.

---

## Phase A — Geometry and ML analysis only

### Task 1: Protect the verified preprocessing boundary

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/memory-bank/active-context.md`
- Create later during implementation: `analysis/reports/analysis_manifest.json`

**Interfaces:**
- Consumes: `preprocessing/reports/selection_manifest.csv`, `preprocessing/pycolmap_input/images/`
- Produces: a verified list of exactly 288 selected filenames and hashes for all downstream analysis tasks

- [ ] **Step 1: Write a failing boundary test**

Create `tests/test_run_geometry_ml_analysis.py` with a test that loads `selection_manifest.csv`, verifies 288 selected rows, and rejects an analysis input directory whose filenames or hashes differ from the manifest.

- [ ] **Step 2: Run the boundary test and confirm RED**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_run_geometry_ml_analysis.py
```

Expected: failure because the new analysis orchestrator does not yet exist.

- [ ] **Step 3: Implement only the analysis-input verifier**

Create `run_geometry_ml_analysis.py` with one focused verifier that returns the ordered selected-image records from `selection_manifest.csv` and raises before creating generated outputs when filename/hash/dimension checks fail.

- [ ] **Step 4: Run the boundary test and confirm GREEN**

- [ ] **Step 5: Record an analysis manifest**

When the full Phase A run eventually succeeds, write `analysis/reports/analysis_manifest.json` containing selected input count, source manifest hash, module versions, and generated artifact list.

---

### Task 2: Geometry 1 — keypoints, candidate matches, and RANSAC inliers

**Files:**
- Create: `geometry_detection.py`
- Create: `tests/test_geometry_detection.py`
- Later generated: `analysis/geometry/pair_metrics.csv`
- Later generated: `analysis/previews/presentation/geometry_01_matches_165_166.png`

**Interfaces:**
- `extract_sift(image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]`
- `match_sift(desc_a: np.ndarray, desc_b: np.ndarray, ratio: float = 0.75) -> list[cv2.DMatch]`
- `estimate_fundamental_geometry(kp_a, kp_b, matches, ransac_threshold: float = 1.5, confidence: float = 0.99) -> GeometryMatchResult`
- `GeometryMatchResult` contains candidate matches, inlier matches, `F`, point arrays, and scalar metrics.

- [ ] **Step 1: Write synthetic correspondence tests**

Tests must cover:

- textured synthetic images produce keypoints;
- ratio-test output contains only valid match indices;
- fewer than eight usable matches returns a structured `insufficient_geometry` result rather than crashing;
- synthetic correspondences projected from non-coplanar 3D points into two camera views produce a finite 3x3 Fundamental Matrix and a non-empty RANSAC inlier set;
- a degenerate pair is handled without an uncaught OpenCV exception.

- [ ] **Step 2: Run the tests and confirm RED**

- [ ] **Step 3: Implement the minimum geometry module**

Reuse the verified preprocessing experiment settings where they already exist: SIFT, BF-L2, ratio `0.75`, Fundamental Matrix RANSAC `1.5 px`, confidence `0.99`, fixed OpenCV RNG seed.

Do not duplicate preprocessing decision logic.

- [ ] **Step 4: Add deterministic match-visualization selection**

For display only, select a spatially distributed subset of at most 60 candidate matches and at most 60 inliers. Full match/inlier counts remain in the report.

- [ ] **Step 5: Run on real pair 165-166 and low-feature pair 255-256**

Generate metrics for both pairs. Do not hard-code expected counts from the old RAW/PREPROCESSED experiment; record fresh results from the exact selected JPEG inputs.

- [ ] **Step 6: Generate the primary presentation figure**

`geometry_01_matches_165_166.png` must show:

- candidate SIFT matches;
- RANSAC-verified matches;
- pair names;
- keypoints per image;
- candidate matches;
- inliers;
- inlier ratio.

- [ ] **Step 7: Visually inspect the figure**

Reject a figure with unreadable labels, excessive match-line clutter, stretched images, or mismatched pair labels.

---

### Task 3: Geometry 2 — epipolar lines and residuals

**Files:**
- Modify: `geometry_detection.py`
- Modify: `tests/test_geometry_detection.py`
- Later generated: `analysis/previews/presentation/geometry_02_epipolar_165_166.png`

**Interfaces:**
- `compute_epilines(F, points, which_image: int) -> np.ndarray`
- `sampson_errors(F, points_a, points_b) -> np.ndarray`
- `select_spatial_inliers(points, max_points: int = 10) -> np.ndarray`

- [ ] **Step 1: Write geometry-consistency tests**

Tests must verify:

- epipolar line arrays are finite and correspond one-to-one with selected points;
- the point-to-epipolar-line residual for synthetic inliers is lower than deliberately corrupted correspondences;
- Sampson errors are finite for valid geometry;
- display selection is deterministic and capped at 10 points.

- [ ] **Step 2: Run tests and confirm RED**

- [ ] **Step 3: Implement epipolar calculations from the same `F` as Task 2**

Do not estimate a second unrelated Fundamental Matrix only for drawing.

- [ ] **Step 4: Generate `geometry_02_epipolar_165_166.png`**

Use 8-10 spatially distributed verified inliers. Give each point/line pair a matching display color. Keep the figure readable and show the median Sampson/geometric residual.

- [ ] **Step 5: Visually verify point/line correspondence**

Check that each selected matching point lies close to its corresponding epipolar line in the opposite image.

---

### Task 4: Geometry 3 — edges, contour, ellipse, and principal axis

**Files:**
- Create: `shape_geometry.py`
- Create: `tests/test_shape_geometry.py`
- Later generated: `analysis/geometry/shape_metrics.csv`
- Later generated: `analysis/previews/presentation/geometry_03_shape_165.png`

**Interfaces:**
- `detect_edges(image: np.ndarray) -> np.ndarray`
- `find_shape_candidates(edges: np.ndarray, mask: np.ndarray | None = None) -> list[np.ndarray]`
- `measure_contour_geometry(contour: np.ndarray) -> ShapeGeometryResult`
- `ShapeGeometryResult` contains area, bounding box, centroid, PCA/principal-axis angle, and optional fitted ellipse.

- [ ] **Step 1: Write controlled-shape tests**

Use synthetic ellipse/vessel-like silhouettes to verify:

- edge output is binary and same width/height as input;
- contour extraction finds the dominant synthetic shape;
- ellipse fitting returns sensible center/axes for a known ellipse;
- principal-axis angle follows a rotated synthetic shape within a reasonable numeric tolerance;
- a contour with too few points returns `ellipse=None` instead of failing.

- [ ] **Step 2: Run tests and confirm RED**

- [ ] **Step 3: Implement the classical geometry path**

Use grayscale, Canny edges, contours, contour moments/PCA, and `cv2.fitEllipse` only when valid. Keep thresholds/configuration explicit and report them.

- [ ] **Step 4: Add optional mask-assisted contour isolation**

Accept a same-size binary mask but keep the classical unmasked edge image available for presentation. Label mask-assisted measurements explicitly in reports.

- [ ] **Step 5: Generate the four-panel presentation figure**

`geometry_03_shape_165.png`:

1. selected image;
2. Canny edges;
3. vessel contour overlay;
4. bounding box, centroid, ellipse candidate(s), and principal/symmetry axis.

- [ ] **Step 6: Inspect a second top-down/detail view**

Run the same analysis on index 255 and document where ellipse/axis interpretation differs because viewpoint changes.

---

### Task 5: ML environment preflight and dependency isolation

**Files:**
- Create only after compatibility is verified: `requirements-ml.txt` or `environment-ml.yml`
- Modify: `README.md`
- Modify: `docs/memory-bank/active-context.md`

**Interfaces:**
- Produces: a documented local SAM 2.1 runtime that does not modify the existing preprocessing environment

- [ ] **Step 1: Re-check current official SAM 2 installation requirements**

Use the official Meta repository at implementation time. Do not rely on versions copied from this plan if the upstream requirements changed.

- [ ] **Step 2: Prefer an isolated WSL/Conda environment on Windows**

The current SAM 2 documentation recommends WSL on Windows. Do not upgrade/downgrade the verified preprocessing environment just to make PyTorch work.

- [ ] **Step 3: Install only the smallest required inference stack**

Start with the official `sam2.1_hiera_small` checkpoint. Do not download base+/large as part of the first pass.

- [ ] **Step 4: Verify one-image inference before processing the dataset**

Success criteria:

- model loads;
- one selected image can be segmented from a point/box prompt;
- mask is returned at the image's original geometry or can be mapped back exactly;
- no preprocessing file changes occur.

- [ ] **Step 5: Record exact model/runtime provenance**

Document model name, checkpoint identifier, upstream repository commit/release if practical, Python/PyTorch/CUDA environment, and inference device. Do not commit checkpoint bytes.

---

### Task 6: SAM 2.1 sequence segmentation and mask QA

**Files:**
- Create: `ml_segmentation.py`
- Create: `tests/test_ml_segmentation.py`
- Later generated: `analysis/ml/masks/*.jpg.png`
- Later generated: `analysis/reports/ml_mask_manifest.csv`
- Later generated: `analysis/previews/presentation/ml_01_segmentation_165.png`
- Later generated: `analysis/previews/presentation/ml_02_mask_contact_sheet.png`

**Interfaces:**
- `normalize_mask(mask, width: int, height: int) -> np.ndarray`
- `write_colmap_mask(image_name: str, mask: np.ndarray, output_dir: Path) -> Path`
- `mask_quality_metrics(mask: np.ndarray) -> MaskQualityResult`
- `full_white_fallback(width: int, height: int) -> np.ndarray`

SAM-specific predictor construction should remain behind a small adapter so mask-format/unit tests do not require loading the model.

- [ ] **Step 1: Write mask-format tests without loading SAM 2**

Verify:

- output masks contain only `0` and `255`;
- output dimensions exactly match the source image;
- `IMG....jpg` maps to `IMG....jpg.png`;
- empty/all-black masks are rejected by QA;
- fallback masks are all white and same size;
- source JPEG bytes remain unchanged.

- [ ] **Step 2: Implement the model-independent mask utilities**

- [ ] **Step 3: Add the SAM 2.1 adapter**

Use promptable sequence/video propagation for the ordered selected images. Seed the vessel with a point or box prompt on an anchor frame and add correction prompts only when visual QA shows drift.

- [ ] **Step 4: Generate masks for all 288 selected images**

Never remove an image because segmentation is weak. Use a full-white fallback for any unresolved mask and record `fallback=true` in `ml_mask_manifest.csv`.

- [ ] **Step 5: Generate presentation segmentation evidence**

`ml_01_segmentation_165.png` must show original, binary mask, and overlay.

`ml_02_mask_contact_sheet.png` must cover representative middle, low, elevated, top-down, detail, and WARN views, plus every fallback/correction case.

- [ ] **Step 6: Visual QA the representative set and all fallback/correction cases**

A valid vessel mask must include the rim, neck, bowl, pedestal, and silhouette boundary without large background regions.

---

### Task 7: Show how ML changes feature locations

**Files:**
- Modify: `geometry_detection.py`
- Modify: `tests/test_geometry_detection.py`
- Later generated: `analysis/previews/presentation/ml_03_masked_features_165.png`
- Later generated: `analysis/reports/masked_feature_counts.csv`

**Interfaces:**
- `filter_keypoints_by_mask(keypoints, descriptors, mask) -> tuple[list[cv2.KeyPoint], np.ndarray | None]`

- [ ] **Step 1: Write mask/keypoint filtering tests**

Synthetic keypoints in white mask regions must survive; keypoints in black regions must be removed. Descriptor rows must remain aligned with surviving keypoints.

- [ ] **Step 2: Implement filtering and count reporting**

- [ ] **Step 3: Generate `ml_03_masked_features_165.png`**

Show:

1. normal SIFT keypoints;
2. SAM mask overlay;
3. keypoints retained inside the mask;
4. counts: total, inside, suppressed background.

This is the clearest immediate explanation of why ML segmentation is useful before any reconstruction claim exists.

---

### Task 8: Phase A orchestration, presentation summary, and verification

**Files:**
- Modify: `run_geometry_ml_analysis.py`
- Modify: `tests/test_run_geometry_ml_analysis.py`
- Later generated: `analysis/reports/geometry_ml_summary.json`
- Later generated: `analysis/previews/presentation/geometry_ml_pipeline_summary.png`
- Modify: `README.md`
- Create/modify: `docs/geometry-ml/geometry-ml-results.md`
- Modify: `CHANGELOG.md`
- Modify: `docs/memory-bank/active-context.md`
- Modify: `docs/memory-bank/progress.md`

- [ ] **Step 1: Add integration tests for planned artifact naming/counts**

The orchestrator must refuse to claim completion unless the required presentation figures and machine-readable reports exist and refer to the same selected-image manifest.

- [ ] **Step 2: Run focused tests**

Use cache-free pytest and do not expand the suite beyond the real module risks.

- [ ] **Step 3: Run the real Phase A analysis**

Generate geometry figures, masks, mask QA, feature-location comparison, and summary reports without calling pyCOLMAP.

- [ ] **Step 4: Generate the one-page summary figure**

`geometry_ml_pipeline_summary.png` should contain actual thumbnails from:

- Geometry 1 match verification;
- Geometry 2 epipolar lines;
- Geometry 3 vessel shape geometry;
- SAM 2.1 segmentation;
- masked/unmasked feature-location comparison.

Clearly label `pyCOLMAP comparison: NEXT PHASE` if reconstruction has not yet run.

- [ ] **Step 5: Verify no input mutation**

Re-hash all 297 raw photographs and all 288 selected PREPROCESSED images against their existing manifests.

- [ ] **Step 6: Update documentation using measured results only**

Do not copy planned numeric results into the results document. Replace planning language only after actual evidence exists.

- [ ] **Step 7: Cleanup**

Remove model caches from the repository workspace, scratch prompts, temporary render frames, `__pycache__`, pytest cache, and abandoned mask experiments. Keep intentional masks/reports/previews if publication is still authorized.

- [ ] **Step 8: Commit Phase A separately**

Suggested commit scope:

```text
feat(analysis): add geometry and vessel segmentation evidence
```

**Hard stop:** Do not start pyCOLMAP unless the user separately authorizes Phase B.

---

## Phase B — Later pyCOLMAP baseline vs ML-mask comparison

### Task 9: Verify pyCOLMAP environment and camera configuration

**Files:**
- Create later: `run_pycolmap_reconstruction.py`
- Create later: `tests/test_pycolmap_config.py`
- Create later: `reconstruction/config/*.json`

- [ ] Verify installed pyCOLMAP/COLMAP version and the exact API for feature extraction, matching, masks, and mapping.
- [ ] Confirm whether all 288 images can share intrinsics based on EXIF/image-size evidence and the chosen camera model.
- [ ] Record the final configuration before running either experiment.

---

### Task 10: Run controlled baseline A — no masks

**Input:** `preprocessing/pycolmap_input/images/`

- [ ] Extract features without `mask_path`.
- [ ] Match using the selected documented strategy.
- [ ] Run sparse incremental mapping.
- [ ] Record registered images, feature/match counts, 3D point count, track statistics, and reprojection error.
- [ ] Save camera/point-cloud visualization evidence.

---

### Task 11: Run controlled experiment B — SAM 2 masks

**Input images:** same 288 images.

**Masks:** `analysis/ml/masks/` via `ImageReader.mask_path`.

- [ ] Keep camera/matching configuration equivalent to baseline A except for the mask path.
- [ ] Extract features, match, and run sparse mapping.
- [ ] Record the same metrics as baseline A.
- [ ] Save comparable camera/point-cloud visualization evidence.

---

### Task 12: Compare and choose reconstruction path

**Files:**
- Create later: `reconstruction/reports/baseline_vs_ml_masks.json`
- Create later: `reconstruction/previews/baseline_vs_ml_masks.png`
- Modify later: `README.md`
- Modify later: `docs/geometry-ml/geometry-ml-results.md`

- [ ] Compare both runs in one table.
- [ ] Reject the mask-assisted variant if registered-image count falls by more than 5% versus baseline unless a documented benefit justifies the trade-off.
- [ ] Inspect background clutter and vessel coverage visually.
- [ ] Choose the reconstruction path from evidence rather than model novelty.
- [ ] Generate one course-presentation comparison image with identical viewpoints/scales where practical.
- [ ] Only after this decision proceed to dense reconstruction, meshing, texturing, and Blender in later plans.

## Plan self-review

- Geometry 1, 2, and 3 are independently testable and produce explicit presentation artifacts.
- SAM 2.1 is the only required new ML model; no unnecessary second model is introduced.
- ML masks integrate through an officially supported COLMAP mask interface rather than modifying image geometry.
- The current verified preprocessing dataset remains the source of truth.
- Phase A cannot accidentally claim pyCOLMAP results because reconstruction is a separate hard-gated phase.
- Every planned visual has a concrete filename and stated contents.
- No model checkpoint or generated result is claimed to exist in this planning commit.
