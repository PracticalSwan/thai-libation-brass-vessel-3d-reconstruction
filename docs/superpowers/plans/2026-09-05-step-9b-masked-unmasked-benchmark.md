# Step 9B Masked-vs-Unmasked Geometry Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether full-image SIFT, raw CNN-mask SIFT, or cleaned reconstruction-mask SIFT gives the strongest geometrically verified correspondence evidence on a frozen 20-pair benchmark.

**Architecture:** Add `reconstruction_matching.py` to filter existing Step 6 `SiftFeatures` by source-size masks, run the existing SIFT/BF-L2/Fundamental-Matrix pipeline unchanged, measure Sampson/grid-coverage evidence, aggregate three modes, and apply the fixed qualification rule from the design.

**Tech Stack:** Python, OpenCV, NumPy, CSV/JSON, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-step-9-reconstruction-readiness-design.md`

## Global Constraints

- Reuse `geometry_detection.extract_sift`, `match_sift`, `estimate_fundamental_geometry`, and `sampson_errors`.
- Do not alter selected JPEGs or redetect features on modified images.
- Filter keypoints/descriptors using source-size binary masks and existing Step 6 scale metadata.
- Use exactly the 20 frozen benchmark pairs in the spec.
- Do not tune the decision rule after seeing results.

---

### Task 1: Filter Step 6 SIFT features by a source-size mask

**Files:**
- Create: `reconstruction_matching.py`
- Create: `tests/test_reconstruction_matching.py`

**Interfaces:**
- Produces: `filter_sift_features(features: SiftFeatures, mask: np.ndarray) -> SiftFeatures`

- [ ] **Step 1: Write failing coordinate/filter tests**

```python
def test_filter_sift_features_keeps_only_keypoints_inside_mask():
    # Build SiftFeatures with two known analysis-space keypoints and explicit scale.
    # Mark only the first corresponding original pixel foreground.
    filtered = filter_sift_features(features, mask)
    assert len(filtered.keypoints) == 1
    assert filtered.descriptors.shape == (1, 128)
```

Also test source-size mismatch and non-binary masks are rejected.

- [ ] **Step 2: Run red test**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_reconstruction_matching.py
```

Expected: import/function failure.

- [ ] **Step 3: Implement minimal feature filtering**

Map each keypoint using `scale_x_to_original` / `scale_y_to_original`; retain descriptor rows in exactly the same order as retained keypoints. Keep the original analysis image and scale metadata.

- [ ] **Step 4: Run green test**

Require PASS.

---

### Task 2: Measure one pair/mode with the existing Step 6 geometry stack

**Files:**
- Modify: `reconstruction_matching.py`
- Modify: `tests/test_reconstruction_matching.py`

**Interfaces:**
- Produces: `PairGeometryMetrics`
- Produces: `measure_pair(first: SiftFeatures, second: SiftFeatures, *, pair: tuple[int, int], mode: str, config: SiftConfig) -> PairGeometryMetrics`
- Produces: `grid_coverage(points: np.ndarray, image_size: tuple[int, int], grid_size: int = 4) -> float`

- [ ] **Step 1: Add failing metric tests**

Verify:

```text
4x4 grid coverage = occupied cells / 16
insufficient geometry returns zero inliers and null/NaN residual summaries safely
valid synthetic correspondences report candidate/inlier counts from the actual GeometryMatchResult
```

- [ ] **Step 2: Run tests red**

- [ ] **Step 3: Implement pair measurement**

Use:

```python
matches = match_sift(first, second, config.ratio_threshold)
geometry = estimate_fundamental_geometry(
    first,
    second,
    matches,
    ransac_threshold=config.ransac_threshold,
    confidence=config.confidence,
    rng_seed=config.rng_seed,
    minimum_correspondences=config.minimum_correspondences,
)
```

For valid geometry, compute Sampson residuals for inlier point rows and average 4x4 coverage over both images.

- [ ] **Step 4: Run tests green**

---

### Task 3: Freeze benchmark pairs and choose the evidence-backed feature mode

**Files:**
- Modify: `reconstruction_matching.py`
- Modify: `tests/test_reconstruction_matching.py`

**Interfaces:**
- Constant: `BENCHMARK_PAIRS`
- Produces: `summarize_benchmark(rows: Sequence[PairGeometryMetrics]) -> dict[str, object]`
- Produces: `choose_feature_mode(summary: dict[str, object]) -> str`

- [ ] **Step 1: Write failing decision-rule tests**

Cases:

```text
masked mode with 96% total inliers, better median inlier ratio, and <=110% Sampson error qualifies
masked mode with 94% total inliers does not qualify
masked mode with worse median inlier ratio does not qualify
no qualified masks -> unmasked
raw_cnn and reconstruction_mask both qualify -> highest median ratio, then total inliers
```

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement exact frozen pair list and qualification rule**

Do not add data-dependent thresholds beyond the spec.

- [ ] **Step 4: Run green tests**

---

### Task 4: Run the real 20-pair, three-mode benchmark and generate evidence

**Files:**
- Modify: `run_reconstruction_readiness.py`
- Generate: `analysis/reports/step9_match_benchmark.csv`
- Generate: `analysis/reports/step9_match_benchmark.json`
- Generate: `analysis/previews/presentation/step9_02_match_benchmark.png`

**Interfaces:**
- `run_reconstruction_readiness.py --stage benchmark`

- [ ] **Step 1: Add orchestration test**

Test that benchmark refuses to start if the 288-row mask manifest or any required mask is missing/mismatched.

- [ ] **Step 2: Run red test**

- [ ] **Step 3: Implement benchmark stage**

Cache extracted `SiftFeatures` in memory by selected index for the 20-pair run. For each pair, evaluate `unmasked`, `raw_cnn`, and `reconstruction_mask` using the same cached unmasked features.

- [ ] **Step 4: Run focused tests green**

- [ ] **Step 5: Run real benchmark**

```powershell
python -B run_reconstruction_readiness.py --stage benchmark
```

Acceptance:

```text
60 CSV rows = 20 pairs x 3 modes
JSON contains aggregate evidence for all three modes
chosen_mode follows the predeclared rule exactly
figure shows total inliers, median inlier ratio, median Sampson error, and qualification status
```

- [ ] **Step 6: Visually inspect `step9_02_match_benchmark.png`**

Confirm labels are readable and the selected mode is not presented as superior when its qualification condition is false.
