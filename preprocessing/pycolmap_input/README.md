# Final pyCOLMAP Input Set

This directory contains the verified 288-image PREPROCESSED set reserved for later reconstruction work.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames
- Current Step 9 subset recommendation: **keep all 288 images**
- Current Step 9 feature recommendation: **unmasked Step 6 SIFT**
- Current Step 9 camera recommendation: **one shared camera/intrinsics group as the starting configuration**

The variant was selected from the representative SIFT matching evidence in `../reports/sift_matching.json`. `WARN` images remain included because a warning is a review signal, not an automatic rejection.

Step 6 geometry analysis, Steps 7+8 custom CNN/feature-mask analysis, and Step 9 reconstruction-readiness analysis read selected images from this directory but never modify them.

Step 9 did run the frozen CNN across all 288 selected images and created separate derived masks under `../../analysis/ml/`. Those masks are **not** the recommended reconstruction matching input: on the frozen 20-pair benchmark, unmasked SIFT produced 3,146 Fundamental-Matrix RANSAC inliers versus 2,841 for both masked modes. The masked modes retained only 90.31% of unmasked inliers and failed the fixed 95% qualification floor.

The full adjacent-sequence audit measured 273 strong and 14 weak transitions over 287 adjacent pairs. None of the 14 tested local skip bridges was strong, so there is no evidence-backed frame that can be removed without risking sequence coverage. `../reconstruction_input_v1/manifest.csv` therefore includes all 288 selected images and references these existing JPEGs instead of duplicating them.

The camera-readiness audit found one consistent raw-EXIF signature across all 288 selected filenames: OPPO Reno12 F, 3072 x 4080, orientation 1, 3.98 mm focal length, 26 mm 35-mm equivalent, and digital zoom 1.0. This supports starting later SfM with one shared camera/intrinsics group, subject to validation by actual reconstruction results.

The current implementation work still stops before pyCOLMAP. No camera poses, triangulation, sparse/dense model, mesh, texture, or Blender output has been created by Step 9.

See `../../docs/geometry-ml/reconstruction-readiness.md` for measured Step 9 results, `../../docs/superpowers/plans/2026-09-05-step-9-reconstruction-readiness.md` for the Step 9 plan index, and `../../docs/geometry-ml/ml-results.md` for the completed Steps 7+8 ML results.
