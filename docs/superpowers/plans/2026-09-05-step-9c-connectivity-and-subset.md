# Step 9C Full-Sequence Connectivity and Reconstruction Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit all adjacent selected-image transitions using the Step 9B-selected feature mode, identify weak transitions, test local skip bridges, and produce a conservative reconstruction-subset manifest without duplicating the 288 JPEGs.

**Architecture:** Extend `reconstruction_matching.py` with full-sequence connectivity metrics and a conservative subset-selection rule. `run_reconstruction_readiness.py --stage connectivity` reads the Step 9B decision, computes adjacent edges, evaluates only the skip bridges needed for weak transitions, writes connectivity/subset reports, and creates one sequence-level visual.

**Tech Stack:** Python, OpenCV, NumPy, CSV/JSON, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-05-step-9-reconstruction-readiness-design.md`

## Global Constraints

- Use only the feature mode chosen by the frozen Step 9B benchmark rule.
- Evaluate every one of the 287 adjacent selected-image pairs.
- Strong edge: status `ok`, inliers >= 15, inlier ratio >= 0.15.
- Test a skip bridge only around a weak adjacent transition.
- Exclude a weak middle image only when neither incident adjacent edge is strong and predecessor/successor have a strong skip bridge.
- Do not reject images merely for redundancy or specular reflections.
- Do not duplicate selected JPEGs into another tracked folder.

---

### Task 1: Define connectivity-edge classification and bridge selection

**Files:**
- Modify: `reconstruction_matching.py`
- Create: `tests/test_reconstruction_connectivity.py`

**Interfaces:**
- Produces: `is_strong_edge(metrics: PairGeometryMetrics) -> bool`
- Produces: `adjacent_pairs(indices: Sequence[int]) -> tuple[tuple[int, int], ...]`
- Produces: `bridge_pairs_for_weak_edges(indices: Sequence[int], edge_rows: Sequence[PairGeometryMetrics]) -> tuple[tuple[int, int], ...]`

- [ ] **Step 1: Write failing threshold/bridge tests**

```python
def test_strong_edge_requires_status_inliers_and_ratio():
    assert is_strong_edge(metric(status="ok", inliers=15, ratio=0.15))
    assert not is_strong_edge(metric(status="ok", inliers=14, ratio=0.50))
    assert not is_strong_edge(metric(status="ok", inliers=30, ratio=0.14))
    assert not is_strong_edge(metric(status="insufficient_geometry", inliers=30, ratio=0.50))
```

Also verify a weak edge between positions `i` and `i+1` requests only `(i-1, i+1)` when both endpoints exist and avoids duplicate bridge pairs.

- [ ] **Step 2: Run red tests**

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_reconstruction_connectivity.py
```

- [ ] **Step 3: Implement the exact fixed strong-edge rule and bridge generator**

- [ ] **Step 4: Run green tests**

---

### Task 2: Define conservative reconstruction-subset selection

**Files:**
- Modify: `reconstruction_matching.py`
- Modify: `tests/test_reconstruction_connectivity.py`

**Interfaces:**
- Produces: `SubsetDecision(selected_index: int, filename: str, include: bool, reason: str)`
- Produces: `choose_reconstruction_subset(records, adjacent_rows, bridge_rows) -> tuple[SubsetDecision, ...]`

- [ ] **Step 1: Write failing subset-rule tests**

Cases:

```text
image with a strong left or right adjacent edge -> include / keep_connected
image with neither adjacent edge strong but a strong predecessor-successor bridge -> exclude / exclude_weak_bridged
image with neither adjacent edge strong and no strong bridge -> include / keep_weak_bridge_needed
first/last image -> include unless its only adjacent relationship is explicitly strong/weak; never exclude an endpoint automatically
```

- [ ] **Step 2: Run red tests**

- [ ] **Step 3: Implement minimal subset decision logic**

Do not add quality-score or redundancy heuristics to this rule.

- [ ] **Step 4: Run green tests**

---

### Task 3: Run full-sequence connectivity and generate subset evidence

**Files:**
- Modify: `run_reconstruction_readiness.py`
- Generate: `analysis/reports/step9_connectivity.csv`
- Generate: `analysis/reports/step9_connectivity.json`
- Generate: `preprocessing/reconstruction_input_v1/manifest.csv`
- Generate: `preprocessing/reconstruction_input_v1/README.md`
- Generate: `analysis/previews/presentation/step9_03_connectivity.png`

**Interfaces:**
- `run_reconstruction_readiness.py --stage connectivity`

- [ ] **Step 1: Add stage-contract test**

Test that connectivity refuses to run when Step 9B JSON is missing, chosen mode is unknown, or required masks for a masked mode are missing.

- [ ] **Step 2: Run red test**

- [ ] **Step 3: Implement full-sequence stage**

Implementation flow:

```text
verify all 288 selected source records
load chosen feature mode from step9_match_benchmark.json
extract Step 6 SIFT once per selected image
apply chosen masks if mode is masked
measure 287 adjacent edges
identify weak adjacent edges
measure unique skip bridges around those weak transitions
choose conservative subset
write edge and subset reports
```

- [ ] **Step 4: Run focused tests green**

- [ ] **Step 5: Run real connectivity stage**

```powershell
python -B run_reconstruction_readiness.py --stage connectivity
```

Acceptance:

```text
287 adjacent rows exist
every weak adjacent transition has an eligible skip bridge result when both endpoints exist
subset manifest contains exactly 288 decisions
included + excluded = 288
README states selected JPEGs are referenced, not duplicated
```

- [ ] **Step 6: Generate and visually inspect `step9_03_connectivity.png`**

The figure must make weak transitions obvious, report strong/weak counts, and state included/excluded subset counts without implying reconstruction success.
