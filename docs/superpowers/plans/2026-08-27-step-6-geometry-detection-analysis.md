# Step 6 Geometry Detection and Analysis Implementation Plan

Updated: 2026-08-28

**Status:** Implemented and locally verified on 2026-08-28. The final
commit/push gate follows this documentation snapshot and is verified from Git,
not predicted inside the plan. The checkboxes below preserve the original
observable execution gates rather than serving as the live completion record;
measured results are in `docs/geometry-ml/geometry-results.md`.

**Goal:** Implement the three planned geometry demonstrations—SIFT/Fundamental-Matrix/RANSAC matching, epipolar geometry, and classical 2D vessel-shape geometry—and generate clear presentation figures from the verified selected images.

**Architecture:** Add a small shared analysis-input verifier plus two focused geometry modules. `geometry_detection.py` owns two-view feature geometry and epipolar calculations; `shape_geometry.py` owns single-image Canny/contour/ellipse/principal-axis measurements; `run_geometry_analysis.py` verifies the 288-image source set, runs representative analyses, writes reports, and creates presentation figures. This plan is completely independent of SAM 2 and stops before any machine-learning or pyCOLMAP work.

**Tech Stack:** Python 3.14, OpenCV 4.13, NumPy, Pillow, pytest. Reuse the verified SIFT experiment settings from `run_preprocessing.py`: maximum analysis width 1200, up to 8000 SIFT features, BF-L2 matching, Lowe ratio 0.75, Fundamental Matrix RANSAC threshold 1.5 px, confidence 0.99, OpenCV RNG seed 4213.

**Spec:** `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

## Implemented result

- all 288 selected images verified before output generation;
- pair 165-166 measured 478 candidates and 300 RANSAC inliers;
- pair 255-256 measured 57 candidates and 18 RANSAC inliers;
- primary median Sampson error measured 0.1431 analysis pixels squared;
- shape 165 retained contour/box/centroid/PCA but rejected its weak global
  ellipse; shape 255 retained its valid ellipse;
- six presentation figures and five machine-readable reports generated;
- all six figures visually inspected, and both live-popup and no-display
  visualizer paths completed bounded smoke checks;
- 31 focused Step 6 tests and the final 52-test repository suite passed;
- 297 raw and 288 selected images reverified unchanged.

## Observable acceptance criteria

- full verification of the 288-row selected manifest succeeds before final
  Step 6 output paths are created or replaced;
- the selected-image boundary reports missing, extra, unreadable, dimension,
  size, and SHA-256 failures distinctly and keeps deterministic one-based order;
- SIFT results expose original/analysis dimensions and explicit coordinate scale
  metadata suitable for later mask/keypoint alignment;
- Fundamental-Matrix success contains one finite 3x3 matrix and its exact RANSAC
  inlier mask; insufficient/degenerate cases are structured failures;
- epipolar lines and Sampson errors consume that same matrix/inlier set;
- shape results contain explainable candidates, Canny edges, selected contour
  when defensible, box, centroid, PCA axis, and ellipse only when valid;
- machine-readable reports retain complete measured data while presentation
  figures draw deterministic readable subsets;
- the manual Python launcher reuses production analysis/rendering code, displays
  real project visuals by default, and has a bounded no-display smoke path;
- every final figure is visually inspected, both source sets reverify with zero
  mismatches, focused tests and compilation pass freshly, residue is cleaned,
  documentation matches measured results, and the intentional commit is pushed
  and verified on GitHub.

## Global Constraints

- `IMG20260826122949/` is immutable.
- `preprocessing/pycolmap_input/images/` remains the verified 288-image PREPROCESSED source set and must not be modified.
- Geometry work is analysis/visualization only: no crop, permanent resize, rotate, warp, perspective correction, reflection removal, synthesis, or source-image overwrite.
- Temporary in-memory downscaling to maximum width 1200 is allowed only for SIFT analysis and must preserve aspect ratio.
- Step 6 must not load SAM 2, create ML masks, install PyTorch, or invoke pyCOLMAP.
- Presentation figures must use real project images and real measured geometry; do not fabricate matches, lines, contours, or metrics.
- Preferred two-view presentation pair: selected-image indices 165-166.
- Preferred supporting/low-feature pair: selected-image indices 255-256.
- Preferred single-image shape examples: selected-image indices 165 and 255.
- These indices are defaults. Replace a default with a measured, visually better
  representative only when the default is insufficient/misleading; record the
  actual index, metrics, substitution reason, and resulting artifact name.
- Reports must state when a pair or shape measurement is insufficient rather than forcing a result.
- Use red-green TDD for production behavior. Documentation/generated figures do
  not receive source-text tests; their real outputs and consuming paths are
  checked instead.
- Do not make intermediate commits. After the full completion gate passes, stage
  only intentional Step 6 files, preserve the user's `.gitignore` change, make
  one scoped commit, push `main`, and verify the remote commit.

## Public coordinate and configuration conventions

- Public image sizes are `(width, height)`; OpenCV array shapes remain
  `(height, width[, channels])`.
- SIFT keypoints, correspondences, `F`, epilines, and Sampson errors use analysis
  pixels. Shape measurements also state their coordinate space explicitly.
- `scale_x_to_original = original_width / analysis_width` and
  `scale_y_to_original = original_height / analysis_height`. Both values are
  exposed even though aspect-ratio preservation makes them nearly equal; callers
  must not infer the scale from one rounded dimension.
- `SiftConfig`: maximum width `1200`, `nfeatures=8000`, BF-L2 `k=2`, Lowe ratio
  `0.75`, minimum correspondences `8`, FM-RANSAC threshold `1.5` analysis pixels,
  confidence `0.99`, RNG seed `4213`.
- `ShapeConfig` keeps analysis width, Canny thresholds, optional small morphology,
  minimum contour/ellipse criteria, and simple candidate-score weights together.
  Tune these only from representative real images and serialize final values.
- Full measured match/inlier arrays stay in memory/results. Display sampling is
  deterministic and affects drawing only.

---

### Task 1: Shared verified-analysis input boundary

**Files:**
- Create: `analysis_common.py`
- Create: `tests/test_analysis_common.py`
- Later generated: `analysis/reports/input_verification.json`

**Interfaces:**
- `SelectedImageRecord(index, filename, variant, width, height, size_bytes, sha256, decision, reasons)`
- `VerifiedSelectedSet(records, manifest_sha256, images_dir)`
- `load_selected_manifest(manifest_path: Path) -> tuple[SelectedImageRecord, ...]`
- `verify_selected_images(images_dir: Path, manifest_path: Path, expected_count: int | None = None) -> VerifiedSelectedSet`
- `verify_selected_record(images_dir: Path, record: SelectedImageRecord, verify_hash: bool = True) -> Path`
- `path_for_index(records: list[SelectedImageRecord], images_dir: Path, one_based_index: int) -> Path`

- [ ] **Step 1: Write failing manifest-boundary tests**

Create `tests/test_analysis_common.py` with controlled temporary files. The tests must prove that valid filename/hash/dimension records pass and that a hash mismatch, missing file, extra selected file, unreadable image, or wrong dimension raises before analysis begins.

```python
def test_load_verified_selected_images_rejects_hash_mismatch(tmp_path):
    images_dir, manifest = build_fixture(tmp_path, corrupt_hash=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_verified_selected_images(images_dir, manifest)
```

Also test one-based index lookup so index `165` means the 165th ordered record, not a filename substring.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_analysis_common.py
```

Expected: import/function failures because `analysis_common.py` does not exist.

- [ ] **Step 3: Implement the minimum verifier**

Read `preprocessing/reports/selection_manifest.csv`, preserve row order as the
canonical selected index, validate safe unique filenames and required fields,
reopen each image, compare dimensions and size, and SHA-256 every file. The
generic verifier accepts fixture counts; the real orchestrator explicitly
requires 288. Do not create final `analysis/` outputs until this succeeds.

- [ ] **Step 4: Run the focused test and verify GREEN**

- [ ] **Step 5: Verify the real selected set**

Run a read-only verifier against:

```text
preprocessing/pycolmap_input/images/
preprocessing/reports/selection_manifest.csv
```

Expected real result: 288 verified records with no mismatch. Write `analysis/reports/input_verification.json` only when the final Step 6 orchestrator runs.

- [ ] **Step 6: Record the boundary as ready for the next red-green increment**

Do not commit yet; the final publication gate owns the single Step 6 commit.

---

### Task 2: Two-view SIFT matching and Fundamental Matrix/RANSAC geometry

**Files:**
- Create: `geometry_detection.py`
- Create: `tests/test_geometry_detection.py`
- Later generated: `analysis/geometry/pair_metrics.csv`
- Later generated: `analysis/previews/presentation/geometry_01_matches_165_166.png`
- Later generated: `analysis/previews/presentation/geometry_01_matches_255_256.png`

**Interfaces:**
- `ImageScale(original_size, analysis_size, scale_x_to_original, scale_y_to_original)` with explicit point-conversion helpers
- `SiftFeatures(analysis_image, keypoints, descriptors, scale, status)`
- `prepare_analysis_image(image: np.ndarray, maximum_width: int = 1200) -> tuple[np.ndarray, ImageScale]`
- `extract_sift(image: np.ndarray, config: SiftConfig = SiftConfig()) -> SiftFeatures`
- `match_sift(first: SiftFeatures, second: SiftFeatures, ratio_threshold: float = 0.75) -> list[cv2.DMatch]`
- `estimate_fundamental_geometry(first: SiftFeatures, second: SiftFeatures, matches: list[cv2.DMatch], ransac_threshold: float = 1.5, confidence: float = 0.99, rng_seed: int = 4213) -> GeometryMatchResult`
- `GeometryMatchResult` stores the finite `F` when available, candidate matches,
  exact inlier mask/matches, candidate point arrays, keypoint counts, counts,
  ratio, configuration, and `ok`/`descriptors_unavailable`/
  `insufficient_geometry`/`degenerate_fundamental_matrix` status.

- [ ] **Step 1: Write failing SIFT/matching tests**

Tests must cover:

```python
def test_extract_sift_preserves_analysis_scale_mapping():
    image = synthetic_textured_image(width=1600, height=1200)
    features = extract_sift(image, maximum_width=1200)
    assert features.analysis_image.shape[1] == 1200
    assert features.scale_to_original == pytest.approx(1600 / 1200)
```

Also verify:

- textured synthetic images produce keypoints/descriptors;
- blank images return a structured empty result;
- every ratio-test match has valid query/train indices;
- fewer than eight matches returns `status="insufficient_geometry"`;
- degenerate `cv2.findFundamentalMat` output returns a structured failure, not an exception;
- synthetic correspondences projected from non-coplanar 3D points through two cameras can produce a finite 3x3 `F` and RANSAC inliers.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_geometry_detection.py
```

- [ ] **Step 3: Implement analysis-scale SIFT extraction**

Mirror the already verified preprocessing experiment settings from `run_preprocessing.py:253-321` without changing that preprocessing code. The analysis-scale resize exists only in memory and must preserve aspect ratio.

- [ ] **Step 4: Implement BF-L2 + Lowe 0.75 matching**

Use `cv2.BFMatcher(cv2.NORM_L2).knnMatch(..., k=2)` and keep a match only when both neighbors exist and `m.distance < 0.75 * n.distance`.

- [ ] **Step 5: Implement Fundamental Matrix/RANSAC verification**

Set `cv2.setRNGSeed(4213)` before `cv2.findFundamentalMat`. Use `cv2.FM_RANSAC`, threshold `1.5`, confidence `0.99`. Validate that `F` is finite and 3x3. Keep the exact inlier mask for later visualization/epipolar analysis.

- [ ] **Step 6: Run tests and verify GREEN**

- [ ] **Step 7: Add deterministic presentation-match sampling**

Create `select_display_matches(...)` that returns at most 60 spatially distributed matches. Sampling affects drawing only; reports must retain full counts.

- [ ] **Step 8: Generate real pair metrics for 165-166 and 255-256**

Use the exact selected JPEG files resolved through `analysis_common.path_for_index`. Record fresh selected-JPEG measurements; do not copy the old RAW-vs-PREPROCESSED numbers into this report.

- [ ] **Step 9: Generate two-row match figures**

Each figure must show:

```text
Row 1: Candidate SIFT matches after Lowe ratio test
Row 2: Fundamental-Matrix RANSAC inliers only
```

Include filenames, keypoints per image, candidate matches, inliers, and inlier ratio in a compact information panel.

- [ ] **Step 10: Visually inspect both figures**

Reject output if labels overlap, images are stretched, match lines are unreadably dense, or the low-feature example is presented as a failure when it simply has fewer correspondences.

- [ ] **Step 11: Record two-view geometry as ready for integration**

Do not commit yet.

---

### Task 3: Epipolar geometry and residual visualization

**Files:**
- Modify: `geometry_detection.py`
- Modify: `tests/test_geometry_detection.py`
- Later generated: `analysis/previews/presentation/geometry_02_epipolar_165_166.png`
- Later generated: `analysis/geometry/epipolar_metrics.json`

**Interfaces:**
- `compute_epilines(F: np.ndarray, points: np.ndarray, which_image: int) -> np.ndarray`
- `sampson_errors(F: np.ndarray, points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray`
- `select_spatial_inliers(points_a: np.ndarray, points_b: np.ndarray, inlier_mask: np.ndarray, max_points: int = 10) -> np.ndarray`
- `clip_epiline_to_image(line: np.ndarray, image_size: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None`

- [ ] **Step 1: Write failing epipolar tests**

Use controlled correspondence data and verify:

```python
def test_corrupted_correspondences_have_larger_sampson_error():
    clean = sampson_errors(F, points_a, points_b)
    corrupted = sampson_errors(F, points_a, points_b + np.array([40.0, 0.0]))
    assert np.median(clean) < np.median(corrupted)
```

Also verify finite line coefficients, one line per selected point, deterministic selection capped at 10 correspondences, and safe handling when `F` is unavailable.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement epipolar calculations from Task 2's exact `F`**

Do not estimate a second Fundamental Matrix for the drawing. Use only Task 2 RANSAC inliers.

- [ ] **Step 4: Implement clipped line drawing**

Convert each epipolar line `ax + by + c = 0` into two valid intersections with the image border. Skip numerically invalid lines rather than drawing outside the image.

- [ ] **Step 5: Generate `geometry_02_epipolar_165_166.png`**

Select 8-10 spatially distributed inliers. Draw each point and corresponding epipolar line with a consistent per-correspondence display color. Report median Sampson error and the number of displayed versus total inliers.

- [ ] **Step 6: Visually verify the figure**

Each matching point should lie close to its corresponding line in the opposite view. Confirm the figure does not imply camera-pose recovery or a completed 3D reconstruction.

- [ ] **Step 7: Record epipolar analysis as ready for integration**

Do not commit yet.

---

### Task 4: Classical 2D vessel-shape geometry

**Files:**
- Create: `shape_geometry.py`
- Create: `tests/test_shape_geometry.py`
- Later generated: `analysis/geometry/shape_metrics.csv`
- Later generated: `analysis/previews/presentation/geometry_03_shape_165.png`
- Later generated: `analysis/previews/presentation/geometry_03_shape_255.png`

**Interfaces:**
- `ShapeConfig(...)` centralizes deterministic analysis/edge/cleanup/scoring values
- `detect_edges(image: np.ndarray, config: ShapeConfig = ShapeConfig()) -> EdgeResult`
- `find_contour_candidates(edges: np.ndarray) -> list[np.ndarray]`
- `score_contour_candidates(candidates, image_shape, config) -> tuple[ContourCandidate, ...]`
- `select_vessel_contour(candidates, image_shape, config) -> ContourCandidate | None`
- `measure_contour_geometry(contour: np.ndarray) -> ShapeGeometryResult`
- `ShapeAnalysisResult` contains the analysis image/scale, binary edge map,
  candidate diagnostics, optional selected contour, contour area, bounding box,
  centroid, principal-axis endpoints/angle, optional ellipse center/axes/angle,
  confidence notes, and structured status.

- [ ] **Step 1: Write failing controlled-shape tests**

Use synthetic filled ellipses and rotated vessel-like silhouettes. Verify:

```python
def test_known_ellipse_returns_reasonable_axes():
    image = synthetic_ellipse(width=800, height=600, axes=(180, 90), angle=25)
    result = analyze_shape(image)
    assert result.ellipse is not None
    assert min(result.ellipse.axes) == pytest.approx(180, rel=0.15)
```

Also verify edge output is binary, output dimensions match input, too-small contours return `ellipse=None`, and PCA/principal-axis angle follows a rotated synthetic shape within a documented tolerance.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement deterministic grayscale + Canny edge detection**

Keep Canny thresholds explicit in one configuration object. Do not modify source image pixels on disk.

- [ ] **Step 4: Implement contour candidate extraction**

Use `cv2.findContours`. Filter only obviously unusable tiny contours. Keep candidate metrics so the selection is explainable.

- [ ] **Step 5: Implement vessel-contour scoring**

Score candidates using a simple explainable combination of area, centrality, and plausible extent. Do not introduce a learned classifier. If no contour is credible, report `status="no_reliable_contour"` instead of forcing a silhouette.

- [ ] **Step 6: Implement centroid, PCA/principal axis, bounding box, and optional ellipse**

Use contour moments for centroid, PCA or equivalent second-moment analysis for the principal axis, `cv2.boundingRect` for the box, and `cv2.fitEllipse` only when at least five contour points exist.

- [ ] **Step 7: Run tests and verify GREEN**

- [ ] **Step 8: Generate four-panel figures for indices 165 and 255**

Each figure must show:

1. selected image;
2. Canny edge map;
3. selected contour overlay;
4. bounding box, centroid, optional ellipse, and principal-axis overlay.

- [ ] **Step 9: Visually inspect both viewpoints**

Document if the top-down/detail view produces different or weaker ellipse/axis interpretation. Do not conceal a weak measurement.

- [ ] **Step 10: Record classical shape geometry as ready for integration**

Do not commit yet.

---

### Task 5: Step 6 orchestrator, reports, and presentation summary

**Files:**
- Create: `run_geometry_analysis.py`
- Create: `tests/test_run_geometry_analysis.py`
- Create: `show_geometry_visuals.py`
- Create or extend: a focused no-display visualizer smoke test
- Later generated: `analysis/reports/geometry_summary.json`
- Later generated: `analysis/previews/presentation/geometry_04_summary.png`
- Create after real run: `docs/geometry-ml/geometry-results.md`
- Modify after real run: `README.md`
- Modify after real run: `CHANGELOG.md`
- Modify after real run: `docs/memory-bank/active-context.md`
- Modify after real run: `docs/memory-bank/progress.md`

**Interfaces:**
- `run_geometry_analysis(images_dir: Path, selection_manifest: Path, output_root: Path) -> GeometryAnalysisSummary`
- `render/save` helpers are reusable by both the orchestrator and visualizer; the
  visualizer must not duplicate SIFT, `F`, epipolar, or shape algorithms.
- `show_geometry_visuals.py --mode {matches,epipolar,shape,all}` displays real
  labeled visuals by default; `--no-display` exercises analysis/rendering and
  exits without blocking. The documented close key is `q` or `Esc`.

- [ ] **Step 1: Write failing orchestration tests**

Use a miniature fixture and mock the expensive visual modules. Verify the orchestrator:

- verifies inputs before creating output directories;
- records configuration and source-manifest hash;
- requires the planned presentation-artifact names before claiming completion;
- never writes inside raw or selected-image directories.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_run_geometry_analysis.py
```

- [ ] **Step 3: Implement the minimum orchestrator**

Real run order:

```text
verify 288 selected inputs
→ two-view matches/RANSAC for 165-166 and 255-256
→ epipolar analysis for 165-166
→ shape analysis for 165 and 255
→ reports
→ presentation figures
→ final summary
```

The orchestrator validates that `output_root` does not overlap raw or selected
sources, requires exactly 288 verified records in the real entrypoint, resolves
preferred defaults through manifest records, and records actual substitutions.
Write JSON/CSV atomically enough that a failed run cannot be mistaken for a
complete summary; the completion flag is written last.

- [ ] **Step 4: Implement and smoke-test the professor popup visualizer**

Normal execution recomputes/reuses the real analysis functions for requested
modes, prints concise metrics, opens display-scaled copies, blocks for manual
viewing, and always destroys windows cleanly. `--no-display` must not call a GUI
wait and must complete in bounded time.

- [ ] **Step 5: Generate `geometry_04_summary.png`**

Create one presentation-ready summary image containing real thumbnails from:

- candidate/RANSAC matches;
- epipolar geometry;
- shape-geometry overlay.

Label the three techniques explicitly. Do not include ML or pyCOLMAP content in this Step 6 summary.

- [ ] **Step 6: Run all Step 6 focused tests**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_analysis_common.py tests/test_geometry_detection.py tests/test_shape_geometry.py tests/test_run_geometry_analysis.py
```

Expected: all Step 6 tests pass.

- [ ] **Step 7: Run the real Step 6 analysis**

Run the orchestrator on the verified selected set. Record actual metrics only.

- [ ] **Step 8: Verify source immutability**

Re-hash all 297 raw photographs and all 288 selected images against their existing manifests. Expected: zero mismatches.

- [ ] **Step 9: Visually inspect every Step 6 presentation figure**

Required figures:

```text
geometry_01_matches_165_166.png
geometry_01_matches_255_256.png
geometry_02_epipolar_165_166.png
geometry_03_shape_165.png
geometry_03_shape_255.png
geometry_04_summary.png
```

- [ ] **Step 10: Write measured Step 6 results documentation**

Create `docs/geometry-ml/geometry-results.md` from actual generated metrics and figures. Keep planned versus measured claims separate.

- [ ] **Step 11: Cleanup and final verification**

Remove only task-created caches, scratch renders, temporary probes, and failed partial figures. Preserve intentional reports and presentation evidence.

- [ ] **Step 12: Run the fresh final gate, commit, push, and verify GitHub**

Before committing, run changed-file compilation, the complete focused Step 6
test command, real selected/raw integrity checks, required-artifact validation,
`git diff --check`, and inspect the exact status/diff. Stage the intentional
Step 6 source, tests, documentation, reports, and final figures only; do not
stage the user's unrelated `.gitignore` modification. Then commit and push:

```text
feat(geometry): add presentation-ready geometry analysis
```

Verify local `HEAD`, `origin/main`, and the pushed commit SHA agree and the
remaining worktree change is only the preserved user-owned change, if still
present.

## Step 6 completion gate

Step 6 is complete only when:

- all three geometry demonstrations are implemented and measured on real selected images;
- the required presentation figures are visually readable;
- reports contain fresh selected-JPEG measurements rather than copied preprocessing numbers;
- the 297 raw files and 288 selected images still match their manifests;
- the manual popup entrypoint and bounded no-display path both use production
  rendering/analysis functions;
- final tests/compilation/diff/integrity checks are fresh and successful;
- the intentional Step 6 commit is present on `origin/main` while unrelated
  local work remains unstaged;
- no ML/SAM or pyCOLMAP work was started.

## Plan self-review

- Geometry 1 is covered by Task 2.
- Geometry 2 is covered by Task 3 and reuses Task 2's exact Fundamental Matrix/inliers.
- Geometry 3 is covered by Task 4 and does not depend on ML.
- Input protection and reproducibility are covered by Tasks 1 and 5.
- Every planned Step 6 presentation artifact has an explicit filename and acceptance check.
- No task invokes SAM 2, PyTorch, pyCOLMAP, sparse reconstruction, dense reconstruction, meshing, texturing, or Blender.
- Every task contains concrete actions, interfaces, verification commands, and acceptance checks.
