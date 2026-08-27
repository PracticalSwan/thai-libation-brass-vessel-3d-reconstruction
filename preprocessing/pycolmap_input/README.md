# Final pyCOLMAP Input Set

Use every image in `images/` as the input to the later pyCOLMAP reconstruction experiments.

- Selected variant: **PREPROCESSED**
- Selected images: **288 of 297**
- Selection rule: include `ACCEPT` and `WARN`; exclude only `REJECT`
- Naming: original deterministic capture filenames

The variant was selected from the representative SIFT matching evidence in `../reports/sift_matching.json`. `WARN` images remain included because a warning is a review signal, not an automatic rejection.

This directory remains the canonical 288-image input for both later reconstruction experiments. The planned geometry/ML Phase A will analyze these images and generate SAM 2.1 vessel masks without modifying them. A separately authorized Phase B may then compare an unmasked pyCOLMAP baseline against the same images with per-image masks supplied through `ImageReader.mask_path`.

See `../../docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md` for the design and `../../docs/superpowers/plans/2026-08-27-geometry-ml-integration.md` for the future implementation plan.
