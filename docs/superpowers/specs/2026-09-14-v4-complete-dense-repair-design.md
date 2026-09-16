# Superseded: V4 Complete Dense Repair Design

**Status:** Superseded on 2026-09-14 after direct investigation proved that `geo_g10` contains sparse-pose inconsistencies. Do not execute the former dense-only design.

The current canonical design is:

`docs/superpowers/specs/2026-09-14-v4-full-repair-design.md`

Reason for supersession: the earlier design assumed the 372/372 sparse reconstruction could remain the accepted geometric coordinate system and concentrated repair at dense stereo/fusion. Later evidence showed strong adjacent two-view geometry contradicting final `geo_g10` camera poses, including a roughly 15.9-degree verified two-view relation becoming roughly 103.8 degrees in the final sparse solution with zero shared final 3D tracks. The historical true3 dense run also permitted duplicate reference writes across tiles (567 reference entries for 372 unique views, 158 duplicated references). Therefore current work must begin with sparse-integrity diagnosis/repair or safe sparse replacement, then regenerate compatible dense evidence and freeze the strongest defensible Poisson. As of 2026-09-15, the current Codex/local executor stops at that pre-Blender handoff; ChatGPT + Blender MCP owns Blender and everything after Blender.

Use the canonical implementation plan:

`docs/superpowers/plans/2026-09-14-v4-full-repair.md`

Historical V4 artifacts remain preserved evidence and must not be overwritten while implementing the full repair.