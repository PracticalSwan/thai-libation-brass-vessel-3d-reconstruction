# Final pyCOLMAP Input Set

This directory contains the verified 288-image PREPROCESSED set used by Step 10 sparse SfM.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames
- Step 9 subset decision: keep all 288 images
- Step 9/10 feature decision: unmasked features
- Step 9/10 camera decision: one shared camera/intrinsics group as the starting configuration

The selected JPEGs remain immutable reconstruction inputs. Step 10 read them directly and did not resize, crop, rotate, recompress, undistort, or otherwise modify them.

Step 10 used pyCOLMAP 4.2.0 native SIFT with one shared `SIMPLE_RADIAL` camera. Because the installed Windows pyCOLMAP wheel was CPU-only, sparse feature extraction used COLMAP's internal `max_image_size=1200`; this changes only the feature-extraction working scale, not these 3072 x 4080 files.

Measured Step 10 result:

- baseline sequential overlap 20: 7 sparse components, 216-image union coverage; largest component 73 images / 6,099 points / 1.2373 px mean reprojection error;
- controlled overlap-40 retry: 7 sparse components, 223-image union coverage; largest component still 73 images / 5,769 points;
- frozen ranking selected the baseline 73-image component under `../../reconstruction/sparse/best/`;
- the fixed >=274-image healthy-single-model acceptance target was not met, so dense reconstruction has not started.

The large component breaks are consistent with Step 9 weak transitions around 73-74, 145-146, and 203-204. The current evidence therefore supports a future targeted sparse-component bridging investigation rather than immediately starting dense MVS or deleting broad groups of images.

See `../../docs/geometry-ml/sparse-reconstruction.md` for measured Step 10 results, `../../docs/geometry-ml/reconstruction-readiness.md` for Step 9 evidence, and `../../docs/superpowers/plans/2026-09-05-step-10-sparse-sfm.md` for the implementation plan.
