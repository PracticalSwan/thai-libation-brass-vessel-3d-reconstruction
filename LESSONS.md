# Lessons Learned

Read this after `AGENTS.md` when starting substantive work. Keep process lessons here; keep current project state in `docs/memory-bank/`.

## 2026-08-27

- Do not over-engineer, over-complicate, or over-test. The coursework benefits from simple, explainable code and verification proportional to actual risk.
- Numerical image-quality thresholds are not authoritative by themselves. Validate them against the real capture distribution and visual coverage before rejecting frames.
- Polished brass naturally produces moving highlights; reflection alone is not a rejection reason.
- Raw smartphone images are immutable source evidence. Derived data must live outside the raw directory, and cleanup must never touch the originals.
- Contact-sheet review found that the final hand-held/flipped sequence changes object pose/background relation and should not be treated like the fixed-object SfM orbit without explicit justification.
- Choose a reconstruction input variant from the exact exported artifact's geometric correspondence evidence, not visual preference or an in-memory approximation. In the final ten-pair experiment, the mild quality-95 JPEG preprocessing produced more total verified inliers and was non-worse on 9 of 10 pairs.
