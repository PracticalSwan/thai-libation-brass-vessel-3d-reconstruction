# CNN Segmentation Dataset

Updated: 2026-09-05

## Frozen dataset

The Step 7 segmentation dataset contains **36 source-resolution binary masks** paired with the verified PREPROCESSED JPEGs in `preprocessing/pycolmap_input/images/`.

- source image geometry: **3072 x 4080**;
- labels: **24 train / 6 validation / 6 held-out test**;
- frozen manifest: `ml_dataset/manifest.csv`;
- frozen manifest SHA-256: `9925bccf367221472e2301d7c360bd7ea4f5f947981d81b5da22f71fe5b02e0f`;
- masks: `ml_dataset/masks/`;
- model input geometry: **384 x 288 (H x W)**, created only in memory;
- source JPEGs were never resized or rewritten on disk.

The split is sequence-aware rather than a random neighboring-frame split. The photographs form a sequential orbit around the same stationary vessel, so near-duplicate adjacent frames were not randomly scattered between train, validation, and test.

## Exact split

### Train — 24

```text
3, 10, 19, 28, 39, 50,
76, 82, 90, 98, 106, 114,
148, 151, 154, 177, 180,
206, 209, 212, 230, 233,
267, 268
```

### Validation — 6

```text
62, 128, 188, 221, 243, 278
```

### Held-out test — 6

```text
72, 142, 165, 200, 255, 288
```

The held-out set includes the Step 6 continuity anchors **165** and **255**. Validation/test membership was frozen before model training. The test split was not used for architecture choice, augmentation choice, threshold tuning, early stopping, or checkpoint selection.

Numerically close boundaries such as 72/76, 142/148, and 200/206 correspond to visible changes in capture height/view phase rather than arbitrary random neighboring-frame splits. The manifest also records the view category and quality condition for each label.

## Mask semantics

```text
255 = visible brass vessel surface
0   = background
```

Background includes the table, classroom, hands or unrelated objects, and any visible opening through which background can be seen.

Masks are source-size PNG files. Training/validation/test resizing uses nearest-neighbor interpolation for masks. CNN predictions were never used as ground truth.

## Annotation process

The final labels were produced with an **OpenCV-assisted, visually reviewed annotation workflow**:

1. A classical OpenCV candidate silhouette was generated from the selected vessel images without using any segmentation CNN.
2. All 36 candidate masks were reviewed on image overlays. The first candidates were rejected as final labels because several views omitted real brass vessel parts, especially the low-angle rear dome/lid and the top-down knob/tiered cone.
3. Broad automatic correction attempts were also rejected when they absorbed yellow classroom surfaces or table/background regions.
4. The final masks retained the conservative reviewed main silhouette and used bounded, view-specific corrections only where source-image inspection supported them: the low-angle dome/lid, the top-down knob/cone, and narrow close-detail boundary corrections.
5. All six final review sheets were inspected before the mask set was frozen and hashed.

Manifest field `annotation_method` records:

```text
opencv_assisted_visually_reviewed_bounded_correction
```

This is not a claim of perfect pixel-level hand tracing. Some narrow rear-side brass slivers remain conservatively excluded in a few oblique/top views. That annotation limitation is retained and documented rather than corrected after seeing held-out model performance.

## Leakage and integrity controls

`segmentation_data.py` verifies:

- selected index and filename agreement with `preprocessing/reports/selection_manifest.csv`;
- no duplicate labeled indices;
- only `train`, `val`, and `test` split names;
- exact initial 24/6/6 split counts;
- source dimensions and SHA-256 provenance;
- readable source-size masks;
- binary mask values limited to `0` and `255`;
- mask SHA-256 values;
- safe project-relative mask paths.

The final source-integrity verification on 2026-09-05 found:

```text
297 / 297 raw photographs unchanged
288 / 288 selected PREPROCESSED images verified
0 raw mismatches
0 selected mismatches
```

No optional training-only label expansion was used because the first fixed baseline exceeded the validation targets.
