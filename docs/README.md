# Documentation Index

This directory contains the current technical documentation for the Thai Libation Brass Vessel 3D Reconstruction project.

## Current project documentation

- **[Project Report](PROJECT_REPORT.md)** — dataset, computer vision pipeline, reconstruction outputs, asset preparation, validation, and reproducibility.
- **[Repository README](../README.md)** — concise project overview, pipeline summary, important artifacts, and repository map.
- **[Active Context](memory-bank/active-context.md)** — newest verified technical state and continuation context.
- **[Progress Log](memory-bank/progress.md)** — current V4/V7 engineering progress and verification milestones.
- **[V4 Full Repair Design](superpowers/specs/2026-09-14-v4-full-repair-design.md)** — current reconstruction/repair architecture and evidence boundaries.
- **[V4 Full Repair Plan](superpowers/plans/2026-09-14-v4-full-repair.md)** — current implementation and verification plan.
- **[V4 Full Repair Handoff](handoffs/2026-09-14-v4-full-repair-codex.md)** — retained V4 execution handoff.

## Current media record

The repository contains only the active V4 capture lineage:

- `CSX4213_Project_V4_Images/` — 688 immutable raw JPEG photographs.
- `capture_v4/manifests/` — role, sequence, exclusion, and media-audit records.
- `capture_v4/derived/mvs_images/` — 372 current reconstruction images.
- `capture_v4/derived/masks/` — 372 current vessel masks.
- `capture_v4/derived/feature_masks/` — 372 current feature masks.

The raw and derived image trees are versioned with Git LFS. Pre-V4 raw data, preprocessing products, analysis outputs, checkpoints, obsolete pipeline code/tests, and superseded implementation documents have been removed from the active repository.

Large transient COLMAP/CUDA workspaces, caches, runtime logs, model weights, and rejected diagnostic outputs are not required for the completed deliverable. The scan-preserving V7 authoring record is under `reconstruction/v4/blender/best_defensible_v1_trim5_authoring_v7_scan_preserving/`; V139 is the final authoring master, and its verified geometry has been promoted to the canonical `.blend` and GLB.
