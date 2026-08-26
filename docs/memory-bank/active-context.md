# Active Context

Updated: 2026-08-27

## Current focus

Finish the real-image preprocessing stage and stop when the dataset is fully ready for pyCOLMAP feature extraction/matching.

## Verified dataset state

- Raw source: `IMG20260826122949/`, 297 JPEG files.
- All 297 are readable, 3072x4080, OPPO Reno12 F, EXIF orientation 1.
- No exact SHA-256 duplicates and no adjacent near-duplicate candidates under the conservative dHash probe.
- Ten contact sheets were visually reviewed. The capture has dense middle/low/elevated/top-down/detail coverage.
- Images 289-297 are a separate hand-held/flipped sequence with object movement and hand occlusion; they require exclusion from the standard fixed-object SfM set unless a later justified reconstruction strategy uses them separately.

## Next action

Replace the stale demo preprocessing with the smallest real-data QA/preprocessing pipeline, validate representative RAW vs PREPROCESSED SIFT matching, finalize ACCEPT/WARN/REJECT selection without coverage gaps, run the full pipeline, prove raw hashes unchanged, clean residue, and leave deterministic pyCOLMAP-ready inputs.
