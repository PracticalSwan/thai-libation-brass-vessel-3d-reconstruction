# Step 11 Sparse Component Bridging

## Outcome

Step 11 is complete, but sparse-component bridging did **not** recover one
global reconstruction. The deterministic bridge diagnosis found qualified
non-local matches only around boundary 203-204. Boundaries 73-74 and 145-146
had no geometrically verified candidate at all, so the targeted mapping gate
correctly skipped the targeted attempt.

The one allowed exhaustive fallback produced eight disconnected sparse models.
Their union contains 224 of the 288 selected images, but the strongest single
model still registers only 73 images. It therefore fails the fixed requirement
of at least 274 registered images in one coherent coordinate frame. Dense
reconstruction, meshing, texturing, and Blender work remain blocked.

## Fixed scope and preserved inputs

The run used the same frozen conditions as Step 10:

- 288 manifest-verified PREPROCESSED JPEGs from
  `preprocessing/pycolmap_input/images/`;
- manifest SHA-256
  `79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91`;
- native unmasked pyCOLMAP SIFT on CPU with `max_image_size=1200`;
- one shared `SIMPLE_RADIAL` camera;
- unchanged Step 10 feature-extraction and incremental-mapping builders;
- pyCOLMAP 4.2.0.

Step 11 did not modify raw images, selected images, Step 10 sparse outputs, or
CNN evidence. It did not use masks, retrain the CNN, change the camera model,
delete images, merge models manually, or invoke a dense-reconstruction API.

## Deterministic bridge diagnosis

The frozen critical boundaries were 73-74, 145-146, and 203-204. For each
boundary, Step 11 paired the 40 selected images on the left with the 40 on the
right, excluding pairs whose sequence gap was 40 or less. This produced exactly
780 candidates per boundary and 2,340 candidates overall.

Every candidate was matched in one diagnostic database. A pair qualified only
when it had at least 15 verified geometric inliers and an inlier ratio of at
least 0.15. Selection was deterministic, limited to eight pairs per boundary,
and limited each image endpoint to two selected uses.

| Boundary | Candidates | Pairs with raw matches | Maximum verified inliers | Maximum inlier ratio | Qualified | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 73-74 | 780 | 195 | 0 | 0.0000 | 0 | 0 |
| 145-146 | 780 | 155 | 0 | 0.0000 | 0 | 0 |
| 203-204 | 780 | 252 | 41 | 0.8235 | 68 | 8 |

The raw-match counts show that the first two windows were not empty searches:
many candidate pairs produced descriptor matches, but geometric verification
rejected every one. Because all three boundaries needed at least one selected
qualified bridge, `targeted_allowed=false`. The recorded gate reason is
`no selected qualified bridge for boundary 73-74`, and no targeted mapper run
was started.

## One exhaustive fallback

The single permitted exhaustive fallback reused the verified 288-image feature
database and ran CPU exhaustive matching with block size 50. A host-process
interruption occurred after part of matching had been durably written. Before
resuming, both databases passed SQLite integrity checks and agreed on all 288
images and 1,255,153 features. The runner then resumed the same
`exhaustive.db`; it did not create or count a second fallback attempt.

The completed database contains:

- 288 images;
- 1,255,153 SIFT features;
- 14,900 non-empty raw-match pair rows;
- 3,020 geometrically verified pair rows.

Incremental mapping still returned eight components. Across those components,
224 distinct selected images appear in at least one model. This union is useful
diagnostic evidence, but it is not one coordinate frame and is not used for the
acceptance decision.

The strongest exhaustive component contains:

- **73/288 registered images**;
- **3,443 sparse points**;
- **12,914 observations**;
- mean track length **3.7508**;
- mean reprojection error **1.1989 px**;
- one `SIMPLE_RADIAL` camera with parameters
  `[4413.5764, 1536.0, 2040.0, 0.1362969]`.

The report's `503.7423 s` runtime measures the successful resumed exhaustive
invocation. It does not include the partial matching work completed before the
host-process interruption, so it must not be presented as total end-to-end
wall time.

## Acceptance decision

| Requirement | Required | Measured | Result |
| --- | ---: | ---: | --- |
| Registered images in one model | at least 274 | 73 | fail |
| Sparse points | at least 1,000 | 3,443 | pass |
| Shared camera | one `SIMPLE_RADIAL` | one `SIMPLE_RADIAL` | pass |
| Mean reprojection error | finite | 1.1989 px | pass |
| Visual geometry/camera layout | plausible | plausible local component only | local-only |

`step11_summary.json` therefore records `bridge_success=false` and
`dense_reconstruction_started=false`.

Step 11 did not improve the main acceptance metric over Step 10: both selected
models register the same first 73 sequence images. The Step 10 model remains
denser, with 6,099 points versus 3,443. The slightly lower Step 11 mean
reprojection error does not compensate for unchanged registration coverage and
must not be interpreted as global recovery.

## Visual inspection

All three generated figures were opened and inspected after finalization:

1. `step11_01_bridge_candidates.png` separates all three fixed boundaries,
   shows zero verified inliers at 73-74 and 145-146, and marks the eight selected
   203-204 bridges with red stars.
2. `step11_02_sparse_model.png` renders the actual selected points and camera
   centers. The cameras form a smooth local arc and the central point structure
   is compatible with a local vessel view, but elongated/outlier structure and
   incomplete coverage make it unsuitable as a global model.
3. `step11_03_registration.png` shows that the selected exhaustive model
   registers indices 1-73 only. It does not substitute the 224-image union for
   the selected-model result.

## Artifacts

Machine-readable evidence:

- `reconstruction/bridging/reports/step11_candidates.csv`
- `reconstruction/bridging/reports/step11_boundary_summary.json`
- `reconstruction/bridging/reports/step11_targeted.json`
- `reconstruction/bridging/reports/step11_exhaustive.json`
- `reconstruction/bridging/reports/step11_attempts.csv`
- `reconstruction/bridging/reports/step11_registered_images.csv`
- `reconstruction/bridging/reports/step11_summary.json`

Selected sparse artifact:

- `reconstruction/bridging/best/` contains the reopened COLMAP binary model;
- `reconstruction/bridging/best/points3D.ply` is the exported point cloud.

Visual evidence:

- `reconstruction/bridging/previews/step11_01_bridge_candidates.png`
- `reconstruction/bridging/previews/step11_02_sparse_model.png`
- `reconstruction/bridging/previews/step11_03_registration.png`

The individual exhaustive components remain under
`reconstruction/bridging/exhaustive/` so the fragmentation evidence is not
hidden.

## Verification

Fresh post-review checks observed:

- Step 11-focused tests: **32 passed**;
- complete project tests: **141 passed**;
- `sparse_reconstruction.py`, `sparse_bridging.py`,
  `run_sparse_reconstruction.py`, and `run_sparse_bridging.py` compiled with
  `python -B -m py_compile`;
- raw source: 297 expected / 297 present, zero missing, unexpected, size, or
  SHA-256 mismatches;
- selected source: 288/288 verified against the frozen manifest;
- all three Step 10 report hashes exactly matched their pre-Step-11 snapshots;
- Step 10 best model reopened at 73 images, 6,099 points, one
  `SIMPLE_RADIAL` camera, and 1.237305 px mean reprojection error;
- Step 11 best model reopened at 73 images, 3,443 points, one
  `SIMPLE_RADIAL` camera, and 1.198883 px mean reprojection error.

Post-implementation review also tightened cache/resume validation: a reusable or
partially resumed feature database must now match the exact image names, camera
IDs, and per-image keypoint/descriptor layout rather than only aggregate counts.
The candidate figure was regenerated with a nonnegative color scale for the two
zero-inlier boundaries. These review fixes changed neither matching evidence nor
any sparse-model measurement.

## Boundary after Step 11

The bounded recovery plan has been exhausted without meeting global sparse
acceptance. Do not start dense MVS, meshing, texturing, or Blender from this
state. Any later work requires a new, explicitly authorized decision about
recapture, a materially different sparse strategy, or accepting a local-only
deliverable. Step 11 itself does not authorize any of those paths.
