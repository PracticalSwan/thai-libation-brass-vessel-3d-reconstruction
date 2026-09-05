# Final pyCOLMAP Input Set

This directory contains the verified 288-image PREPROCESSED set used by Step 10 sparse SfM and Step 11 sparse-component bridging.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames
- Step 9 subset decision: keep all 288 images
- Step 9/10/11 feature decision: unmasked features
- Step 9/10/11 camera decision: one shared camera/intrinsics group as the starting configuration

The selected JPEGs remain immutable reconstruction inputs. Steps 10 and 11 read them directly and did not resize, crop, rotate, recompress, undistort, or otherwise modify them.

Step 10 used pyCOLMAP 4.2.0 native SIFT with one shared `SIMPLE_RADIAL` camera. Because the installed Windows pyCOLMAP wheel was CPU-only, sparse feature extraction used COLMAP's internal `max_image_size=1200`; this changes only the feature-extraction working scale, not these 3072 x 4080 files.

Measured Step 10 result:

- baseline sequential overlap 20: 7 sparse components, 216-image union coverage; largest component 73 images / 6,099 points / 1.2373 px mean reprojection error;
- controlled overlap-40 retry: 7 sparse components, 223-image union coverage; largest component still 73 images / 5,769 points;
- frozen ranking selected the baseline 73-image component under `../../reconstruction/sparse/best/`;
- the fixed >=274-image healthy-single-model acceptance target was not met, so dense reconstruction has not started.

The large component breaks are consistent with Step 9 weak transitions around 73-74, 145-146, and 203-204. Step 11 investigated those boundaries without deleting any selected image and confirmed that the current input still does not support one accepted global sparse model.

Measured Step 11 result:

- deterministic diagnosis matched 2,340 non-local candidates, 780 around each fixed boundary;
- 73-74 and 145-146 had zero geometrically verified candidates, while 203-204 had 68 qualified candidates and 8 selected bridges;
- targeted mapping was skipped because every boundary needed at least one selected qualified bridge;
- the one CPU exhaustive fallback produced eight sparse models with 224-image union coverage;
- its strongest single model remained 73 images, with 3,443 points and 1.1989 px mean reprojection error;
- the >=274-image acceptance target still failed, so `bridge_success=false` and dense reconstruction remains blocked.

See `../../docs/geometry-ml/sparse-component-bridging.md` for measured Step 11 results, `../../docs/geometry-ml/sparse-reconstruction.md` for Step 10 results, `../../docs/geometry-ml/reconstruction-readiness.md` for Step 9 evidence, and `../../docs/superpowers/plans/2026-09-05-step-11-sparse-component-bridging.md` for the Step 11 implementation plan.
