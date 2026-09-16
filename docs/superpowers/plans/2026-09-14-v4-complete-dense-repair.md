# Superseded: V4 Complete Dense Repair Implementation Plan

**Status:** Superseded on 2026-09-14. Do not execute the former dense-only task sequence.

Use instead:

`docs/superpowers/plans/2026-09-14-v4-full-repair.md`

with design:

`docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`

The superseding evidence is a confirmed upstream sparse-geometry defect, not merely a dense-tuning issue: several `geo_g10` acquisition-adjacent pairs have strong verified two-view geometry but incompatible final SfM poses and near-zero final shared-track continuity. The historical true3 tiling also allowed duplicate reference rewrites. Therefore repair now begins with strict sparse-integrity diagnostics and repaired/safely replaced sparse geometry, followed by a fresh compatible dense workspace and strict dense/Poisson anatomical continuity gates. As of 2026-09-15, the current Codex/local executor stops at the verified Poisson handoff; ChatGPT + Blender MCP owns scan-preserving Blender finalization, appearance, GLB re-import, final docs, and final publication.

Historical V4 outputs and unrelated work must remain preserved while the full repair runs.