# Step 12 Learned Sparse Recovery

## Outcome

Step 12 reached its approved native-capability failure boundary and stopped
without running full learned feature extraction or sparse mapping. The real
ALIKED-N16Rot smoke test reached pyCOLMAP's native extractor, but pyCOLMAP 4.2.0
raised:

```text
RuntimeError: ALIKED feature extraction requires ONNX support.
```

The result is therefore a runtime capability blocker, not a learned-feature
quality result. No ALIKED boundary measurements, LoMa measurements, or learned
sparse model exist. Step 11 remains the latest measured reconstruction result:
73/288 images in the strongest single model, with 224 images represented only
across eight disconnected models.

Step 12 records:

```text
metric_acceptance_met       false
visual_plausibility_status  failed
learned_recovery_success    false
dense_reconstruction_started false
```

## Fixed scope and protected inputs

The capability run used the frozen Step 12 setup:

- 288 manifest-verified PREPROCESSED JPEGs from
  `preprocessing/pycolmap_input/images/`;
- selection-manifest SHA-256
  `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`;
- Python 3.14.2 and pyCOLMAP 4.2.0;
- CPU device policy with `pycolmap.has_cuda == false`;
- learned `max_image_size=1600`;
- primary frontend ALIKED-N16Rot + ALIKED-LightGlue;
- conditional fallback LoMa-B + LoMa-L.

The smoke pair was selected indices 1 and 2:

```text
IMG20260826122949.jpg
IMG20260826122953.jpg
```

The raw and selected photographs remained read-only. Step 10 and Step 11
reports/models were not overwritten. The local CNN checkpoint was not used or
published. No DOCX/PDF artifact was modified.

## Capability evidence

Static option construction succeeded for both approved frontend definitions.
The installed runtime exposed the required enum members, both extraction and
matching option objects passed `check()`, both extractors reported RGB input as
required and OpenGL as unnecessary, and both used the explicit CPU policy.

The real ALIKED smoke then attempted to create and run the native extractor on
the two RGB images. It returned zero features because ONNX support is not
available in the installed pyCOLMAP learned-extraction path. The structured
smoke result preserves the exact exception class and message.

This failure occurred before:

- the 288-image ALIKED feature database;
- the exact 2,340-pair boundary experiment;
- ALIKED sequential or imported-pair matching;
- ALIKED sparse mapping;
- any LoMa model initialization, extraction, matching, or mapping.

LoMa remained `not_run`. The approved design explicitly forbids using LoMa to
bypass a missing native learned-runtime prerequisite and forbids silently
rebuilding pyCOLMAP, replacing the wheel, or installing an external learned
matching stack. Consequently, the blocker was recorded and execution stopped.

## Acceptance decision

| Requirement | Required | Measured | Result |
| --- | ---: | ---: | --- |
| Native ALIKED smoke | non-empty features and matches | blocked before extraction | fail |
| Boundary candidates | exactly 2,340 measured pairs | not run | not measured |
| Registered images in one model | at least 274 | no learned model | fail |
| Sparse points | at least 1,000 | no learned model | fail |
| Camera | one `SIMPLE_RADIAL` | no learned model | not measured |
| Mean reprojection error | finite | no learned model | not measured |
| Visual model plausibility | plausible | no learned model | fail |

This table must not be interpreted as evidence that ALIKED or LoMa performs
poorly on the photographs. The experiment could not reach learned inference in
the installed runtime.

## Visual evidence

Only `step12_04_frontend_comparison.png` was generated because no learned
candidate table or sparse model exists. It was opened and visually inspected.
The figure correctly contains only the two authoritative strongest-single-model
SIFT baselines:

- Step 10 SIFT: 73 registered images, 6,099 points, 1.2373 px mean
  reprojection error;
- Step 11 exhaustive SIFT: 73 registered images, 3,443 points, 1.1989 px mean
  reprojection error.

No learned bar, boundary trace, camera trajectory, point cloud, or registration
plot was fabricated.

## Artifacts

Machine-readable evidence:

- `reconstruction/learned_recovery/reports/step12_capability.json`
- `reconstruction/learned_recovery/reports/step12_aliked_attempt.json`
- `reconstruction/learned_recovery/reports/step12_attempts.csv`
- `reconstruction/learned_recovery/reports/step12_summary.json`

Visual evidence:

- `reconstruction/learned_recovery/previews/step12_04_frontend_comparison.png`

Intentionally absent because the corresponding real stage did not run:

- ALIKED/LoMa candidate CSV and boundary-summary reports;
- LoMa attempt report;
- `best/` learned sparse model and PLY;
- learned registered-image CSV;
- boundary, sparse-model, and registration figures;
- transient learned feature/matching databases.

## Verification

Before the capability run, the refreshed implementation checkpoint passed:

- Step 12 domain tests: **20 passed**;
- Step 12 runner tests: **14 passed**;
- Step 10/11 regression tests: **49 passed**.

The real capability run then exposed one truthful reporting gap: no-model
finalization called LoMa merely “not required,” which could be mistaken for an
ALIKED success. A focused regression was added and the runner now carries the
primary capability blocker into LoMa's `not_run` reason. That change raised the
runner suite to **15 passed**. Final review found a second report-contract gap:
blocked attempt JSON omitted its frontend/extractor/matcher identity. A focused
regression fixed that path without rerunning learned inference, bringing the
runner suite to **16 passed**.

Input verification before runtime found:

- 288/288 selected images matched the selection manifest;
- the selection-manifest SHA-256 matched the frozen value;
- 297/297 raw images matched their size/SHA-256 baseline with zero missing,
  unexpected, size-mismatched, or hash-mismatched files.

Final verification then observed:

- complete project suite after final review: **177 passed**;
- all four Step 12 source/test files compiled successfully;
- the Step 10 model reopened at 73 images, 6,099 points, one `SIMPLE_RADIAL`
  camera, and 1.237305 px mean reprojection error;
- the Step 11 model reopened at 73 images, 3,443 points, one `SIMPLE_RADIAL`
  camera, and 1.198883 px mean reprojection error;
- the five frozen Step 10/11 report hashes matched their pre-Step-12 snapshots;
- the final selected/raw verification again found 288/288 selected images and
  297/297 unchanged raw images;
- the Step 12 output contains no transient database, `work/` directory,
  learned `best/` model, learned PLY, or dense artifact;
- the four `.pyc` files created by compilation and their two empty
  `__pycache__/` directories were removed after verification.

## Boundary after Step 12

The two permitted native learned frontends cannot be evaluated with the current
pyCOLMAP wheel without changing the approved runtime architecture. Dense MVS,
meshing, texturing, and Blender remain blocked.

Any continuation requires a new explicit decision between:

1. accepting a local-only sparse reconstruction; or
2. authorizing a separate experimental learned global-matching/component-
   alignment phase, including its runtime/dependency change.

Step 12 does not authorize either choice and does not authorize a pyCOLMAP
rebuild or external learned stack by itself.
