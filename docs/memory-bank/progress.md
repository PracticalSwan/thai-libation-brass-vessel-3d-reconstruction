# Progress

Updated: 2026-08-27

## Completed and verified

- Public repository created: `PracticalSwan/thai-libation-brass-vessel-3d-reconstruction`.
- Full read-only audit completed over 297 raw photographs.
- `dataset_audit.csv`, `dataset_summary.json`, and `raw_manifest_before.json` generated.
- Contact-sheet visual review completed across all 297 images.
- Shared project agent rules, cleanup policy, Git ignore rules, and GitHub-facing documentation prepared.

## Remaining before pyCOLMAP

- Implement focused tests and final `quality_check.py`, `preprocess_images.py`, and `run_preprocessing.py`.
- Calibrate decisions from real metrics plus visual coverage.
- Run RAW vs PREPROCESSED neighboring-frame SIFT match comparison.
- Produce deterministic selected/derived image set and final reports/previews.
- Re-hash originals, verify counts/readability/geometry, visually inspect outputs, and clean temporary residue.
- Commit/push the verified preprocessing completion milestone.

Do not start pyCOLMAP until those readiness checks are satisfied.
