# Step 13 External Learned Global Recovery

## Outcome

Step 13 completed the one approved external learned sparse-recovery experiment.
The external **ALIKED-N16Rot + LightGlue** frontend ran successfully on CUDA,
recovered all three previously targeted sequence boundaries, and materially
improved the strongest single sparse reconstruction from 73 registered images
to **266/288**. However, the frozen global acceptance gate required at least
**274/288** registered images. Step 13 therefore records:

```text
metric_acceptance_met        false
visual_plausibility_status   failed
step13_success               false
dense_reconstruction_started false
selected_sparse_source       reconstruction/sparse/best
```

The 266-image Step 13 model is retained as measured evidence under
`reconstruction/external_learned_recovery/best/`, but it is **not** the
selected downstream reconstruction source. Per the predeclared fallback rule,
Step 10's verified 73-image local model is selected because Steps 10 and 11 tie
on registered images and Step 10 contains more sparse points.

Sparse-recovery experimentation is now closed. The next reconstruction phase,
if authorized, should be designed explicitly around the selected local sparse
model rather than changing the Step 13 acceptance threshold after seeing the
result.

## Runtime and dependency provenance

Step 13 intentionally changed only the learned matching runtime. It used the
official CVG LightGlue repository pinned to:

```text
https://github.com/cvg/LightGlue.git
eb42fee2d71449efb0aa5c10549752b5d75384d8
```

Measured runtime:

```text
Python                 3.14.2
PyTorch                2.13.0+cu130
Torchvision            0.28.0+cu130
Torch CUDA             13.0
GPU                     NVIDIA GeForce RTX 5050 Laptop GPU
pyCOLMAP                4.2.0
ALIKED model            aliked-n16rot
Max inference side      1600 px
Max keypoints/image     4096
```

The installed LightGlue package's direct-URL metadata resolves to the exact
pinned commit. pyCOLMAP remained responsible for image/camera database import,
external keypoint/match storage, geometric verification, incremental mapping,
and sparse-model inspection.

The project requirements continue to list `opencv-python`. The current machine
has `opencv-contrib-python 4.13.0.92` providing `cv2`, so `pip check` reports
LightGlue's package-metadata requirement for the separately named
`opencv-python` distribution even though OpenCV 4.13 is installed and the real
Step 13 runtime plus complete test suite execute successfully. No duplicate
OpenCV wheel was installed merely to silence that environment-level metadata
warning.

## Protected inputs

The experiment used the frozen 288 PREPROCESSED JPEGs under:

```text
preprocessing/pycolmap_input/images/
```

Selection-manifest SHA-256:

```text
79408d59b022803e1acc42d4c1e118c209a8120e76e689ea9b60909015f37a91
```

Final integrity verification found all 288 selected images valid and all 297
raw JPEGs unchanged. Five protected Step 10/11 report hashes also remained
exactly unchanged.

## Capability gate

The real capability smoke used selected indices 1 and 2:

```text
IMG20260826122949.jpg
IMG20260826122953.jpg
```

Results:

```text
first ALIKED features   4096
second ALIKED features  4096
LightGlue raw matches   2952
device                  cuda
status                  passed
```

This is the runtime path that Step 12 could not execute through native
pyCOLMAP because its wheel lacked ONNX support.

## Exact 2,340-pair learned boundary diagnostic

Step 13 reused the exact Step 11 candidate identities and unchanged bridge
qualification rules. No threshold tuning was performed after inspecting the
learned results.

| Boundary | Candidates | Qualified | Selected | Maximum verified inliers | Maximum inlier ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 73-74 | 780 | 778 | 8 | 249 | 0.6727 |
| 145-146 | 780 | 418 | 8 | 120 | 0.7037 |
| 203-204 | 780 | 745 | 8 | 1,129 | 0.7083 |

All three critical boundaries passed the frozen gate, so the single full
learned mapping attempt was allowed. This is a substantial difference from
Step 11 SIFT, where boundaries 73-74 and 145-146 had zero verified inliers.

The diagnostic reporting distinguishes learned correspondences before COLMAP
verification from the database match rows after verification:

```text
LightGlue raw matches before verification    363,318
COLMAP raw matches after verification        363,171
removed during geometric verification             147
selected learned bridges                          24
```

## Single full learned mapping attempt

The mapping schedule was frozen before execution:

```text
sequential overlap-20 pairs     5,550
selected learned bridges           24
final unique pair schedule       5,574
LightGlue raw matches          5,269,937
LightGlue matching runtime       596.912 s
mapping runtime                 4,850.738 s
```

The resulting COLMAP database contained:

```text
images                288
external keypoints    1,129,555
non-empty match rows  5,531
verified pair rows    5,264
```

pyCOLMAP produced three retained sparse models. The strongest single model was:

```text
registered images             266 / 288 (92.36%)
sparse points                 29,713
observations                  106,480
mean track length             3.5836
mean reprojection error       1.374824 px
camera count                  1
camera model                  SIMPLE_RADIAL
focal parameter               4969.6110 px
radial parameter              -0.34418
```

The remaining 22 images are the tail of the selected sequence, indices
267-288. The model therefore crossed the earlier 73-74, 145-146, and 203-204
breaks but did not absorb enough of the final capture segment to reach the
274-image acceptance floor.

## Acceptance decision

| Requirement | Required | Measured | Result |
| --- | ---: | ---: | --- |
| Registered images in one model | >=274 | 266 | fail |
| Sparse points | >=1,000 | 29,713 | pass |
| Camera count | 1 | 1 | pass |
| Camera model | `SIMPLE_RADIAL` | `SIMPLE_RADIAL` | pass |
| Mean reprojection error | finite | 1.374824 px | pass |
| Visual acceptance | only considered after metric gate | metric gate failed | fail |

The model missed the registration requirement by **8 images**. The threshold
was not relaxed after observing the result, and no second matcher, mapping
retry, parameter sweep, or learned exhaustive search was run.

## Visual evidence

Four Step 13 figures were generated from real project outputs and opened for
review:

- `step13_01_boundary_comparison.png` shows the exact Step 11-vs-Step 13
  candidate comparison and preserves the zero-inlier SIFT boundaries at zero.
- `step13_02_sparse_model.png` shows the real 266-image ALIKED+LightGlue point
  cloud and camera centers. It also exposes visible camera/geometry outliers
  instead of hiding them.
- `step13_03_registration.png` shows registration through selected index 266
  and the unregistered 267-288 tail.
- `step13_04_model_comparison.png` shows the 266-image model below the fixed
  274-image acceptance line while documenting its much higher point count.

Because the metric gate already failed, the final visual status is recorded as
`failed`; no visual override can convert the experiment into a success.

## Local fallback

Finalization reopened both existing candidate local models and applied the
frozen ranking rule: registered images, then sparse points, then lower finite
reprojection error.

| Local source | Registered | Points | Mean reprojection error |
| --- | ---: | ---: | ---: |
| Step 10 SIFT | 73 | 6,099 | 1.237305 px |
| Step 11 exhaustive SIFT | 73 | 3,443 | 1.198883 px |

Both have 73 registered images, so the point-count tie-break selects Step 10:

```text
reconstruction/sparse/best
```

The 266-image Step 13 model remains evidence of a strong but below-threshold
learned recovery attempt. It is not silently substituted for the declared
fallback.

## Artifacts

Durable evidence:

```text
reconstruction/external_learned_recovery/
  best/
    cameras.bin
    frames.bin
    images.bin
    points3D.bin
    points3D.ply
    rigs.bin
  reports/
    step13_capability.json
    step13_candidates.csv
    step13_boundary_summary.json
    step13_attempt.json
    step13_registered_images.csv
    step13_summary.json
  previews/
    step13_01_boundary_comparison.png
    step13_02_sparse_model.png
    step13_03_registration.png
    step13_04_model_comparison.png
```

The Step 13 `work/` directory, learned feature cache, diagnostic database,
mapping database, pair lists, and mapper component workspace were removed after
final durable evidence was written.

## Verification

Final review and verification established:

- focused Step 13 suite: **21 passed**;
- complete project suite: **198 passed**;
- all four Step 13 source/test files compiled successfully, then generated
  `.pyc`/`__pycache__` residue was removed;
- 297/297 raw JPEGs matched the immutable baseline with zero missing,
  unexpected, size-mismatched, or hash-mismatched files;
- 288/288 selected inputs matched the frozen selection manifest;
- the strongest Step 13 model reopened at 266 images, 29,713 points, one
  `SIMPLE_RADIAL` camera, and 1.374824 px mean reprojection error;
- Step 10 and Step 11 source models reopened with their prior measured metrics;
- the five protected Step 10/11 report hashes remained unchanged;
- Step 13 source contains no learned exhaustive matcher or dense/MVS API;
- transient Step 13 `work/` state is absent;
- all four Step 13 figures were visually inspected.

## Boundary after Step 13

Step 13 was the final bounded sparse-recovery experiment. Its learned frontend
proved that the earlier fragmentation was substantially recoverable, but the
accepted global sparse gate was still not met. Per the approved decision rule,
sparse recovery is closed and the selected downstream source is the verified
Step 10 local sparse model.

A later phase may explicitly authorize and design **local-only dense
reconstruction** from that selected model. Dense MVS, meshing, texturing, and
Blender were not started by Step 13.
