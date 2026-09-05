# Lessons Learned

Read this after `AGENTS.md` when starting substantive work. Keep process lessons here; keep current project state in `docs/memory-bank/`.

## 2026-08-27

- Do not over-engineer, over-complicate, or over-test. The coursework benefits from simple, explainable code and verification proportional to actual risk.
- Numerical image-quality thresholds are not authoritative by themselves. Validate them against the real capture distribution and visual coverage before rejecting frames.
- Polished brass naturally produces moving highlights; reflection alone is not a rejection reason.
- Raw smartphone images are immutable source evidence. Derived data must live outside the raw directory, and cleanup must never touch the originals.
- Contact-sheet review found that the final hand-held/flipped sequence changes object pose/background relation and should not be treated like the fixed-object SfM orbit without explicit justification.
- Choose a reconstruction input variant from the exact exported artifact's geometric correspondence evidence, not visual preference or an in-memory approximation. In the final ten-pair experiment, the mild quality-95 JPEG preprocessing produced more total verified inliers and was non-worse on 9 of 10 pairs.

## 2026-09-06

- Keep one owner for orchestration gates. Step 12's `all` stage duplicated ALIKED capability interpretation that already belonged to the diagnose stage, causing state-machine drift and two failing orchestration tests. Let each stage own its report semantics and let the top-level runner branch only on stage results.
- Cache provenance must identify the exact per-image database layout, not only aggregate feature counts. A learned-feature database can preserve the same total while changing image-level keypoint/descriptor rows; persist a deterministic layout fingerprint before reusing it.
- Learned pyCOLMAP matching must pass the chosen `FeatureMatchingOptions` explicitly to both sequential and imported-pair matching. Omitting either call can silently fall back to SIFT and invalidate the learned-recovery experiment.
- On Windows tests that rebuild SQLite files, explicitly close fixture connections before unlinking or replacing the database; transaction context management alone does not close the connection.
- Unit/orchestration tests validate the Step 12 control contract, not real learned reconstruction quality. Do not report ALIKED/LoMa recovery results until the native runtime stages are actually executed and measured.
