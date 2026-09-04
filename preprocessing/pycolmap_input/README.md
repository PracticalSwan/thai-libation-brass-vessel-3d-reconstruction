# Final pyCOLMAP Input Set

This directory contains the verified 288-image PREPROCESSED set reserved for later reconstruction work.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames

The variant was selected from the representative SIFT matching evidence in `../reports/sift_matching.json`. `WARN` images remain included because a warning is a review signal, not an automatic rejection.

The current implementation work stops before pyCOLMAP. Completed Step 6 geometry analysis and completed Steps 7+8 custom CNN/feature-mask analysis read selected images from this directory but never modify them. The ML phase used a frozen 36-image reviewed subset for training/validation/test and produced analysis predictions only; it did **not** create or authorize a full 288-image reconstruction-mask pipeline.

See `../../docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md` for the implemented design, `../../docs/superpowers/plans/2026-08-27-geometry-ml-integration.md` for the completed Step 6 / Steps 7+8 plan index, and `../../docs/geometry-ml/ml-results.md` for measured ML results.
