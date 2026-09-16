# Documentation Index

This directory contains the technical and project documentation for the Thai Libation Brass Vessel 3D Reconstruction project.

## Current project documentation

- **[Project Report](PROJECT_REPORT.md)** — human-facing overview of the project, dataset, computer vision pipeline, reconstruction outputs, asset preparation, and validation.
- **[Repository README](../README.md)** — concise project overview, pipeline summary, important artifacts, and repository map.
- **[Active Context](memory-bank/active-context.md)** — current technical state and latest verified project context.
- **[Progress Log](memory-bank/progress.md)** — chronological technical progress and major verification milestones.
- **[V4 Full Repair Design](superpowers/specs/2026-09-14-v4-full-repair-design.md)** — detailed V4 reconstruction and repair architecture.
- **[V4 Full Repair Plan](superpowers/plans/2026-09-14-v4-full-repair.md)** — implementation and verification plan for the V4 reconstruction lineage.

## Supporting technical documentation

### Preprocessing

- `preprocessing/preprocessing-results.md` — dataset preprocessing and source-image preparation results.

### Geometry and reconstruction

- `geometry-ml/geometry-results.md` — geometric analysis results.
- `geometry-ml/reconstruction-readiness.md` — reconstruction-readiness analysis.
- `geometry-ml/sparse-reconstruction.md` — sparse reconstruction methodology and evidence.
- `geometry-ml/sparse-component-bridging.md` — sparse connectivity work.
- `geometry-ml/learned-sparse-recovery.md` — learned-feature sparse recovery work.
- `geometry-ml/external-learned-global-recovery.md` — global recovery experiments and evidence.

### Segmentation and feature analysis

- `geometry-ml/cnn-dataset.md` — earlier segmentation dataset documentation.
- `geometry-ml/ml-results.md` — segmentation and feature-mask analysis results.

## Historical plans and specifications

Files under `superpowers/plans/` and `superpowers/specs/` dated before the V4 full-repair documents record earlier stages of the project. They are retained for project history and implementation provenance. The current public summary is the Project Report, while the 2026-09-14 V4 full-repair design and plan remain the detailed technical references for the current reconstruction lineage.

## Documentation scope

The documentation separates three kinds of information:

1. **Public project explanation** — `README.md` and `PROJECT_REPORT.md` describe what the project is, what methods were used, and what was produced.
2. **Current engineering state** — `memory-bank/` records the newest verified technical state and continuation context.
3. **Historical implementation evidence** — older plans, specifications, and geometry/segmentation documents preserve how earlier project stages were implemented and evaluated.

Large temporary reconstruction workspaces, caches, repeated runtime logs, and disposable diagnostic outputs are not documentation deliverables. Selected reports, representative previews, manifests, and reproducible scripts are retained instead.
