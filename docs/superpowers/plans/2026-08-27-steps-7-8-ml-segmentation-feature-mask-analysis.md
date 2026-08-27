# Steps 7 and 8 ML Segmentation and Feature-Mask Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate pretrained SAM 2.1 vessel segmentation on a representative set of verified project images, then quantify and visualize how vessel masks change the distribution of SIFT features for an easy-to-show machine-learning demonstration.

**Architecture:** Keep ML inference isolated from the verified preprocessing environment. `ml_segmentation.py` owns SAM 2.1 prompt-based segmentation, binary-mask normalization, QA, and overlays. `ml_feature_analysis.py` reuses Step 6's public `geometry_detection.extract_sift` function to count features inside versus outside each vessel mask without changing source images. `run_ml_analysis.py` verifies inputs, runs the ten-image representative experiment, writes reports, and creates presentation figures. This plan ends after feature-mask analysis; it does not invoke pyCOLMAP or perform reconstruction.

**Tech Stack:** Meta SAM 2.1 with the `sam2.1_hiera_small` checkpoint in an isolated compatible PyTorch environment; Python/OpenCV/NumPy/Pillow for mask handling, SIFT feature analysis, reporting, and figures; pytest for model-independent unit tests. Use the Step 6 SIFT settings: maximum analysis width 1200 and up to 8000 SIFT features.

**Spec:** `docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`

## Prerequisite

Step 6 must be implemented first through at least `analysis_common.py` and `geometry_detection.extract_sift`. If those interfaces are absent or their tests fail, stop and complete the Step 6 plan instead of duplicating SIFT/input-verification code here.

## Global Constraints

- `IMG20260826122949/` is immutable.
- `preprocessing/pycolmap_input/images/` remains the verified 288-image PREPROCESSED source set and must not be modified.
- Steps 7+8 use **ten representative selected images only** for the current coursework demonstration: indices `15, 45, 75, 105, 135, 165, 195, 225, 255, 280`.
- Do not generate masks for all 288 images in this phase; that is unnecessary before reconstruction is authorized.
- Start with pretrained `sam2.1_hiera_small`. Do not train/fine-tune a custom model and do not download larger checkpoints unless the ten-image feasibility result demonstrates a real need.
- Keep SAM 2/PyTorch runtime dependencies isolated from the verified preprocessing environment.
- Never commit model checkpoint bytes, package caches, CUDA caches, temporary model downloads, or other runtime residue.
- ML must segment/label pixels only. Do not inpaint, remove reflections, synthesize texture, generate details, alter geometry, or overwrite source JPEGs.
- Every binary mask must have the same width/height as its source image and contain only `0` and `255`.
- A failed segmentation stays a documented failure/correction case; do not reject the source photograph.
- Step 8 measures feature distribution only. Do not claim masks improve 3D reconstruction because reconstruction is out of scope.
- No task in this plan may invoke pyCOLMAP, COLMAP, sparse/dense reconstruction, meshing, texturing, or Blender.

---

### Task 1: ML environment preflight and reproducible boundary

**Files:**
- Create only after compatibility is verified: `requirements-ml.txt` or `environment-ml.yml`
- Create: `docs/geometry-ml/ml-environment.md`
- Modify after verified setup: `README.md`

**Interfaces:**
- Produces a documented ML runtime capable of loading `sam2.1_hiera_small` and segmenting one project image without changing the existing preprocessing environment.

- [ ] **Step 1: Confirm the Step 6 prerequisite**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_analysis_common.py tests/test_geometry_detection.py
```

Expected: pass. If either file/test does not exist because Step 6 has not been implemented, stop this plan.

- [ ] **Step 2: Re-check the official SAM 2 installation instructions**

At implementation time, use the current official Meta SAM 2 repository as the source of truth for supported Python/PyTorch/CUDA and Windows/WSL setup. Do not copy dependency versions from old notes when upstream requirements have changed.

- [ ] **Step 3: Create an isolated ML environment**

Prefer a separate WSL/Conda/venv environment rather than modifying the working preprocessing environment. Record the chosen environment location and activation command in `docs/geometry-ml/ml-environment.md`.

- [ ] **Step 4: Install only the required inference stack**

Install the official SAM 2 package and the `sam2.1_hiera_small` checkpoint only. Keep the checkpoint outside tracked repository paths.

- [ ] **Step 5: Verify one-image inference on selected index 165**

Use a single box prompt around the vessel. Success criteria:

```text
model loads
source image opens at 3072 x 4080
mask inference completes
returned mask maps exactly to 3072 x 4080
source JPEG hash is unchanged
```

Do not proceed to the ten-image run if this bounded feasibility test fails.

- [ ] **Step 6: Record exact runtime provenance**

Document:

- SAM model/checkpoint name;
- upstream package/repository version or commit when practical;
- Python version;
- PyTorch version;
- CUDA/device used, or CPU if applicable;
- checkpoint path policy;
- one-image feasibility result.

- [ ] **Step 7: Commit only the environment description/config**

Suggested commit:

```text
chore(ml): document isolated SAM 2 runtime
```

---

### Task 2: Representative-image prompt configuration

**Files:**
- Create: `analysis/config/ml_prompts.json`
- Create: `tests/test_ml_prompt_config.py`

**Interfaces:**
- `load_prompt_config(path: Path, records: list[SelectedImageRecord]) -> list[SegmentationPrompt]`
- `SegmentationPrompt` contains `index`, `filename`, and normalized box `[x1, y1, x2, y2]` where every coordinate is in `[0.0, 1.0]`.

- [ ] **Step 1: Write failing prompt-schema tests**

Tests must reject missing representative indices, duplicate indices, filenames that disagree with `selection_manifest.csv`, boxes outside `[0,1]`, inverted boxes, and boxes that cover less than a small minimum fraction of the image.

```python
def test_prompt_box_must_be_normalized_and_ordered(tmp_path):
    config = write_prompt_fixture(tmp_path, box=[0.8, 0.2, 0.4, 0.9])
    with pytest.raises(ValueError, match="box"):
        load_prompt_config(config, records)
```

- [ ] **Step 2: Run the test and verify RED**

- [ ] **Step 3: Implement prompt-config validation**

The required index set is exactly:

```python
REPRESENTATIVE_INDICES = (15, 45, 75, 105, 135, 165, 195, 225, 255, 280)
```

Resolve filenames from the verified selection manifest; do not type filenames from memory.

- [ ] **Step 4: Visually inspect the ten selected images**

Open the exact selected JPEG for each required index and record one conservative box around the full vessel. Boxes should include rim, neck, bowl, pedestal, and a small margin, while excluding as much unrelated background as practical.

- [ ] **Step 5: Save the reproducible prompt file**

Write `analysis/config/ml_prompts.json` with exactly these top-level fields:

- `model`: exactly `"sam2.1_hiera_small"`;
- `coordinate_system`: exactly `"normalized_xyxy"`;
- `prompts`: exactly ten objects, one for each required representative index.

Each prompt object must contain:

- `index`: one of the ten required integer indices;
- `filename`: the exact filename resolved from `selection_manifest.csv` for that index;
- `box`: four visually measured normalized floats `[x1, y1, x2, y2]` satisfying `0 <= x1 < x2 <= 1` and `0 <= y1 < y2 <= 1`.

Do not commit synthetic/full-image box values. The committed configuration must contain the actual ten inspected filenames and measured vessel boxes.

- [ ] **Step 6: Run the prompt-schema tests and verify GREEN**

- [ ] **Step 7: Commit the prompt configuration**

Suggested commit:

```text
chore(ml): add representative vessel prompts
```

---

### Task 3: SAM 2.1 segmentation adapter and binary-mask QA

**Files:**
- Create: `ml_segmentation.py`
- Create: `tests/test_ml_segmentation.py`
- Later generated: `analysis/ml/masks/*.png`
- Later generated: `analysis/reports/ml_mask_manifest.csv`

**Interfaces:**
- `normalize_binary_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray`
- `mask_quality_metrics(mask: np.ndarray) -> MaskQualityResult`
- `write_mask(source_filename: str, mask: np.ndarray, output_dir: Path) -> Path`
- `Sam2ImageSegmenter.segment(image: np.ndarray, normalized_box: tuple[float, float, float, float]) -> np.ndarray`

The model adapter must remain separate from mask-format utilities so unit tests can run without loading SAM 2.

- [ ] **Step 1: Write failing model-independent mask tests**

Verify:

```python
def test_normalize_binary_mask_returns_only_zero_and_255():
    source = np.array([[0.1, 0.9], [0.6, 0.2]], dtype=np.float32)
    mask = normalize_binary_mask(source, width=2, height=2)
    assert set(np.unique(mask)) <= {0, 255}
```

Also test:

- output dimensions equal requested source dimensions;
- empty/all-black masks fail QA;
- all-white masks are flagged as suspicious rather than silently accepted;
- mask area fraction and bounding box are reported;
- writing a mask never modifies the source JPEG;
- mask filenames map deterministically from source filenames.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement mask normalization and QA utilities**

Use nearest-neighbor interpolation if a model output must be resized back to the source geometry. Threshold once into `uint8` `{0,255}`. Record foreground area fraction and mask bounding box.

- [ ] **Step 4: Implement the SAM 2.1 image-predictor adapter**

Convert normalized box coordinates to source-image pixel coordinates, run the official image predictor, select the highest-confidence vessel mask returned for the box prompt, and return only the raw mask array to the caller.

- [ ] **Step 5: Run model-independent tests and verify GREEN**

- [ ] **Step 6: Run real segmentation on the ten representative images**

For each index:

```text
verify selected image
→ load normalized box prompt
→ SAM 2.1 inference
→ normalize to binary source-size mask
→ QA metrics
→ write mask
→ record manifest row
```

- [ ] **Step 7: Handle weak masks explicitly**

If the mask clips a major vessel part or includes a large unrelated region, adjust that image's box prompt once and rerun it. Record `corrected=true` and the final prompt in the manifest. If a usable mask still cannot be produced, record `status="failed_segmentation"` and preserve the image unchanged; do not invent a mask.

- [ ] **Step 8: Commit segmentation code and measured representative masks**

Suggested commit:

```text
feat(ml): segment representative vessel images
```

---

### Task 4: Presentation-ready segmentation evidence

**Files:**
- Modify: `ml_segmentation.py`
- Modify: `tests/test_ml_segmentation.py`
- Later generated: `analysis/previews/presentation/ml_01_segmentation_165.png`
- Later generated: `analysis/previews/presentation/ml_02_mask_contact_sheet.png`

**Interfaces:**
- `render_segmentation_triptych(image: np.ndarray, mask: np.ndarray, metrics: MaskQualityResult) -> np.ndarray`
- `render_mask_contact_sheet(items: list[SegmentationPreview]) -> np.ndarray`

- [ ] **Step 1: Write image-layout tests**

Tests should validate output readability structurally: expected panel count, nonzero output dimensions, deterministic title/label strings, and no source-image mutation.

- [ ] **Step 2: Implement `ml_01_segmentation_165.png`**

The figure must show three clearly labeled panels:

```text
Original selected image | Binary vessel mask | Mask overlay
```

Include model name, source index/filename, mask foreground percentage, and whether a correction was required.

- [ ] **Step 3: Implement the ten-image mask contact sheet**

`ml_02_mask_contact_sheet.png` must show all ten representative indices in chronological order with the vessel mask overlaid. Each tile must display the index and segmentation status.

- [ ] **Step 4: Visually inspect both outputs**

Reject figures with unreadable labels, stretched images, overlays that hide the vessel, or masks whose visible errors contradict the recorded QA status.

- [ ] **Step 5: Commit presentation segmentation evidence**

Suggested commit:

```text
feat(ml): add segmentation presentation evidence
```

---

### Task 5: Step 8 SIFT feature-mask analysis

**Files:**
- Create: `ml_feature_analysis.py`
- Create: `tests/test_ml_feature_analysis.py`
- Later generated: `analysis/reports/masked_feature_counts.csv`
- Later generated: `analysis/previews/presentation/ml_03_masked_features_165.png`
- Later generated: `analysis/previews/presentation/ml_04_feature_mask_summary.png`

**Interfaces:**
- Consumes: `geometry_detection.extract_sift` from Step 6 and the ten representative binary masks from Task 3.
- `classify_keypoints_by_mask(features: SiftFeatures, full_resolution_mask: np.ndarray) -> FeatureMaskResult`
- `FeatureMaskResult` contains total keypoints, vessel keypoints, background keypoints, vessel fraction, background-suppression fraction, and aligned keypoint index lists.

- [ ] **Step 1: Write failing keypoint-mask tests**

Create synthetic `SiftFeatures` with known keypoint coordinates and verify inside/outside classification after analysis-scale conversion.

```python
def test_keypoints_are_classified_using_matching_scale():
    features = synthetic_features(points=[(10, 10), (90, 90)], analysis_size=(100, 100))
    mask = binary_mask_with_white_top_left(width=200, height=200)
    result = classify_keypoints_by_mask(features, mask)
    assert result.vessel_count == 1
    assert result.background_count == 1
```

Also test an all-white mask, all-black mask, dimension mismatch, and descriptor/keypoint alignment.

- [ ] **Step 2: Run tests and verify RED**

- [ ] **Step 3: Implement mask-to-SIFT-scale alignment**

Resize the binary mask to `features.analysis_image` using nearest-neighbor interpolation. Never resize or rewrite the source JPEG.

- [ ] **Step 4: Implement feature classification and metrics**

For every keypoint, sample the aligned mask at its rounded/clipped coordinates. Count white-mask keypoints as vessel features and black-mask keypoints as background features.

- [ ] **Step 5: Run tests and verify GREEN**

- [ ] **Step 6: Analyze all ten representative images**

Write `masked_feature_counts.csv` with at least:

```text
index
filename
total_keypoints
vessel_keypoints
background_keypoints
vessel_feature_fraction
background_suppression_fraction
mask_foreground_fraction
segmentation_status
```

- [ ] **Step 7: Generate `ml_03_masked_features_165.png`**

Use four panels:

1. all SIFT keypoints;
2. SAM vessel mask overlay;
3. vessel-inside keypoints only;
4. compact counts/percentages panel.

This figure explains the ML integration without making any reconstruction claim.

- [ ] **Step 8: Generate `ml_04_feature_mask_summary.png`**

Create one simple summary chart/table across all ten representative indices showing total keypoints versus vessel-inside keypoints and the percentage of background keypoints identified by the mask. Keep each chart visually separate and readable.

- [ ] **Step 9: Visually inspect both figures**

Confirm keypoints shown as vessel-inside actually lie on visible vessel regions in the overlay. Investigate large discrepancies rather than hiding them.

- [ ] **Step 10: Commit feature-mask analysis**

Suggested commit:

```text
feat(ml): analyze vessel-masked SIFT features
```

---

### Task 6: Steps 7+8 orchestrator, results, and final demonstration summary

**Files:**
- Create: `run_ml_analysis.py`
- Create: `tests/test_run_ml_analysis.py`
- Later generated: `analysis/reports/ml_summary.json`
- Later generated: `analysis/previews/presentation/ml_05_summary.png`
- Create after real run: `docs/geometry-ml/ml-results.md`
- Modify after real run: `README.md`
- Modify after real run: `CHANGELOG.md`
- Modify after real run: `docs/memory-bank/active-context.md`
- Modify after real run: `docs/memory-bank/progress.md`

**Interfaces:**
- `run_ml_analysis(images_dir: Path, selection_manifest: Path, prompt_config: Path, output_root: Path) -> MlAnalysisSummary`

- [ ] **Step 1: Write failing orchestration tests**

Use model-independent fixtures and a fake segmenter. Verify:

- prerequisite interfaces are checked before output creation;
- exactly the ten representative indices are processed;
- every successful segmentation has a same-size binary mask and feature-mask metrics;
- failed segmentation remains documented and does not delete/exclude the source image;
- no write occurs inside raw or selected-image directories;
- completion does not depend on pyCOLMAP artifacts.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_run_ml_analysis.py
```

- [ ] **Step 3: Implement the minimum orchestrator**

Real run order:

```text
verify Step 6 prerequisite
→ verify 288 selected inputs
→ load ten representative prompts
→ segment ten images
→ QA/correct masks
→ analyze SIFT feature distribution
→ reports
→ presentation figures
→ final ML summary
```

- [ ] **Step 4: Generate `ml_05_summary.png`**

Create a professor-ready one-page summary using real outputs:

```text
SAM 2.1 input + prompt
→ vessel segmentation mask
→ mask overlay
→ SIFT features inside/outside mask
→ measured ten-image feature summary
```

Do not include pyCOLMAP, point clouds, reconstruction metrics, or fabricated future results.

- [ ] **Step 5: Run all model-independent Steps 7+8 tests**

Run in the repository Python environment where possible:

```powershell
python -B -m pytest -p no:cacheprovider -q tests/test_ml_prompt_config.py tests/test_ml_segmentation.py tests/test_ml_feature_analysis.py tests/test_run_ml_analysis.py
```

Model-loading/inference verification belongs to the isolated ML environment and must be reported separately.

- [ ] **Step 6: Run the real ten-image ML analysis**

Run SAM inference only on the ten representative images. Record actual mask/feature metrics and correction/failure cases.

- [ ] **Step 7: Verify source immutability**

Re-hash all 297 raw photographs and all 288 selected images against their existing manifests. Expected: zero mismatches.

- [ ] **Step 8: Visually inspect every required ML presentation figure**

Required:

```text
ml_01_segmentation_165.png
ml_02_mask_contact_sheet.png
ml_03_masked_features_165.png
ml_04_feature_mask_summary.png
ml_05_summary.png
```

- [ ] **Step 9: Write measured ML results documentation**

Create `docs/geometry-ml/ml-results.md` containing model/runtime provenance, prompt method, ten-image segmentation QA, feature-mask measurements, limitations, and the figures above. State explicitly that no reconstruction experiment has been run.

- [ ] **Step 10: Cleanup**

Remove checkpoint copies inside the repository, caches, temporary renders, scratch prompts, failed partial outputs, `__pycache__`, pytest cache, and abandoned mask experiments. Preserve the final prompt config, intentional representative masks, reports, and presentation figures.

- [ ] **Step 11: Commit Steps 7+8 implementation separately**

Suggested commit:

```text
feat(ml): add vessel segmentation and feature-mask analysis
```

## Steps 7+8 completion gate

Steps 7+8 are complete only when:

- `sam2.1_hiera_small` is verified in an isolated runtime;
- all ten representative images have either a verified mask or an explicitly documented segmentation failure;
- segmentation and feature-mask presentation figures are readable and based on real outputs;
- feature counts are measured using Step 6's SIFT implementation;
- raw and selected-image manifests still match;
- no claim is made that masking improves 3D reconstruction;
- no pyCOLMAP/reconstruction work was started.

## Plan self-review

- Step 7 segmentation is covered by Tasks 1-4.
- Step 8 feature-mask analysis is covered by Task 5.
- Task 6 integrates the two steps into a single presentation-ready result without extending into reconstruction.
- The current phase processes only ten representative images, avoiding unnecessary 288-image mask generation.
- Step 6 SIFT/input logic is reused rather than duplicated.
- Every required ML presentation artifact has an explicit filename and acceptance check.
- No task invokes pyCOLMAP, COLMAP, sparse/dense reconstruction, meshing, texturing, or Blender.
- Every task contains concrete actions, interfaces, verification commands, and acceptance checks.
