# Final pyCOLMAP Input Set

This directory contains the verified 288-image PREPROCESSED set reserved for later reconstruction work.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames

The variant was selected from the representative SIFT matching evidence in `../reports/sift_matching.json`. `WARN` images remain included because a warning is a review signal, not an automatic rejection.

The current implementation work stops before pyCOLMAP. Completed Step 6 geometry analysis and the planned Steps 7+8 custom CNN/feature-mask workflow may read selected images from this directory, but they must never modify it. The CNN plan references a small manually labeled subset for training/validation/test and does not prepare a full 288-image reconstruction mask set.

See `../../docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md` for the current design and `../../docs/superpowers/plans/2026-08-27-geometry-ml-integration.md` for the Step 6 / Steps 7+8 plan index.
