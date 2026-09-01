# Thai Libation Brass Vessel 3D Reconstruction

Computer vision coursework project for reconstructing a real Thai brass libation vessel from smartphone photographs.

The explainable project pipeline is:

```text
smartphone capture
-> OpenCV quality analysis and conservative preprocessing
-> classical 2D/two-view geometry analysis
-> planned custom CNN vessel segmentation + SIFT feature-mask analysis
-> pyCOLMAP feature extraction and matching
-> Structure from Motion / sparse reconstruction
-> dense reconstruction where practical
-> meshing and texturing
-> Blender cleanup and final model
```

## Project status

The real-image preprocessing, QA, and Step 6 classical geometry-analysis phases
are complete and verified. The next planned phase is a separately authorized
from-scratch CNN segmentation and SIFT feature-mask analysis workflow. No CNN
training, ML masks, or pyCOLMAP reconstruction has been run.

Measured preprocessing result:

- 297 immutable OPPO Reno12 F JPEG captures at 3072 x 4080;
- 207 `ACCEPT`, 81 `WARN`, and 9 `REJECT` decisions;
- all `ACCEPT` and `WARN` images retained, giving 288 final inputs;
- rejects are exactly images 289-297, the separate hand-held/flipped sequence with object movement and hand occlusion;
- PREPROCESSED selected from ten neighboring-pair SIFT comparisons using the exact exported quality-95 JPEG encoding: 2,483 fundamental-matrix RANSAC inliers versus 2,376 for RAW, with PREPROCESSED non-worse on 9 of 10 pairs;
- all 288 selected outputs reopened successfully at 3072 x 4080 with no duplicate hashes;
- all 297 originals re-hashed against the publication baseline with zero size or SHA-256 mismatches.

Measured Step 6 geometry result:

- all 288 selected inputs were re-verified against the selection manifest before analysis;
- neighboring pair 165-166 produced 478 ratio-test candidates and 300 Fundamental Matrix RANSAC inliers, an inlier ratio of 0.628;
- supporting pair 255-256 produced 57 candidates and 18 inliers, honestly retaining its lower-feature result;
- primary-pair median Sampson error was 0.1431 analysis pixels squared;
- classical contour/PCA measurements were produced for images 165 and 255; the weak global ellipse for the non-elliptical side view was deliberately omitted, while the top-down/detail ellipse passed residual checks;
- six real presentation figures and five machine-readable reports were generated under `analysis/`;
- all 297 raw photographs and all 288 selected images were rechecked unchanged after analysis.

The exact next-stage input directory is:

```text
preprocessing/pycolmap_input/images/
```

Read [`preprocessing/pycolmap_input/README.md`](preprocessing/pycolmap_input/README.md) before reconstruction. The full measured method, tables, visual evidence, verification details, and limitations are in [`docs/preprocessing/preprocessing-results.md`](docs/preprocessing/preprocessing-results.md).

## Geometry + planned machine-learning extension

Step 6 is implemented and verified. Steps 7+8 now have a revised CNN-based
planning architecture but remain unimplemented and require separate execution
authorization. The project boundary still stops before pyCOLMAP.

### Step 6 — Geometry Detection / Analysis

- implemented SIFT keypoints and candidate matches;
- implemented Fundamental Matrix + exact RANSAC inlier handling;
- implemented epipolar lines and Sampson residuals;
- implemented Canny edges, classical contours, ellipse fitting where valid,
  centroid/bounding box, and PCA principal axis;
- generated and visually verified six presentation-ready geometry figures.

### Steps 7 + 8 — From-Scratch CNN + Feature-Mask Analysis

- manually annotate an initial 36-image binary vessel-segmentation dataset;
- split it as 24 train / 6 validation / 6 held-out test using separated capture positions/view groups rather than a random neighboring-frame split;
- train a small U-Net-like `SmallSegCNN` from random initialization with no pretrained weights;
- evaluate held-out predictions with Dice, IoU, foreground precision, recall, and visible failure analysis;
- reuse Step 6 SIFT extraction to measure keypoints inside versus outside CNN-predicted vessel masks;
- generate training, segmentation, feature-mask, and summary figures from real outputs.

```mermaid
flowchart LR
    A[288 verified PREPROCESSED images] --> B[Step 6: SIFT + RANSAC]
    B --> C[Epipolar geometry]
    A --> D[Step 6: Canny + contour + ellipse/axis]
    A --> E[Manual vessel masks + leakage-safe split]
    E --> F[Step 7: SmallSegCNN trained from scratch]
    F --> G[Held-out predictions + Dice/IoU]
    G --> H[Step 8: SIFT inside vs outside predicted mask]
    C --> I[Verified Step 6 geometry evidence]
    D --> I
    H --> J[ML evidence]
    I --> K[STOP before pyCOLMAP]
    J --> K
```

- Design: [`docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md`](docs/superpowers/specs/2026-08-27-geometry-ml-integration-design.md)
- Plan index: [`docs/superpowers/plans/2026-08-27-geometry-ml-integration.md`](docs/superpowers/plans/2026-08-27-geometry-ml-integration.md)
- Step 6 measured results: [`docs/geometry-ml/geometry-results.md`](docs/geometry-ml/geometry-results.md)
- Step 6 implementation plan: [`docs/superpowers/plans/2026-08-27-step-6-geometry-detection-analysis.md`](docs/superpowers/plans/2026-08-27-step-6-geometry-detection-analysis.md)
- Steps 7+8 implementation plan: [`docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md`](docs/superpowers/plans/2026-08-27-steps-7-8-ml-segmentation-feature-mask-analysis.md)

## Why preprocessing is conservative

Polished brass naturally produces moving specular highlights. Reflection alone is not a rejection reason. The selected transform changes only luminance photometry: CLAHE-enhanced LAB luminance is blended at 15% with the original luminance.

The workflow does **not** crop, rotate, resize, warp, perspective-correct, synthesize detail, remove reflections with AI, or otherwise move image features. Final derived images keep the original 3072 x 4080 geometry.

## Reproduce preprocessing

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m py_compile quality_check.py preprocess_images.py run_preprocessing.py
python -B -m pytest -p no:cacheprovider -q
python run_preprocessing.py
```

`run_preprocessing.py` verifies the raw baseline before processing, writes deterministic reports/previews/selected outputs outside the raw folder, and verifies the raw baseline again at the end. It never invokes pyCOLMAP.

The verified local environment used Python 3.14.2, OpenCV 4.13.0, NumPy 2.4.0, Pillow 12.1.1, and pytest 9.1.1.

## Reproduce Step 6 geometry analysis

Run the deterministic analysis and regenerate its reports and presentation
figures:

```powershell
python -B run_geometry_analysis.py
```

Open the live OpenCV popup views; press `q` or `Esc` to close all windows:

```powershell
python -B show_geometry_visuals.py --mode all
```

Modes `matches`, `epipolar`, and `shape` are also available. For a bounded
non-GUI smoke check, add `--no-display`.

## Repository layout

```text
quality_check.py                         quality metrics, calibration, decisions
preprocess_images.py                     geometry-preserving photometric transform
run_preprocessing.py                     reports, previews, SIFT experiment, export
tests/                                   52 focused preprocessing and Step 6 tests
analysis_common.py                       selected-manifest loading and integrity verification
geometry_detection.py                    scaled SIFT, RANSAC, epilines, and residuals
shape_geometry.py                        classical edges, contour, PCA, and optional ellipse
run_geometry_analysis.py                 Step 6 orchestration, reports, and figures
show_geometry_visuals.py                 real popup visualizer for the Step 6 flow
analysis/                                Step 6 machine reports and presentation figures
preprocessing/reports/                   audit and final measured reports
preprocessing/previews/contact_sheets/   full raw-sequence visual audit
preprocessing/previews/final/            before/after, decision, and SIFT figures
preprocessing/pycolmap_input/images/      exact 288-image next-stage input set
IMG20260826122949/                        versioned immutable raw photographs
docs/preprocessing/                       measured method, results, and verification evidence
```

The separate local `IMG20260826122949.zip` is only a redundant archive of the same photographs. It is intentionally untracked and is not part of the publication set.

## Evidence entry points

- `preprocessing/reports/quality_decisions.csv` — one final decision and reason set per raw image.
- `preprocessing/reports/quality_thresholds.json` — thresholds derived from eligible real-capture distributions.
- `preprocessing/reports/sift_matching.csv` and `.json` — pair-level RAW/PREPROCESSED evidence and selection rule.
- `preprocessing/reports/selection_manifest.csv` — dimensions, hashes, and decision provenance for every selected output.
- `preprocessing/reports/raw_verification_after.json` — final raw-data immutability proof.
- `preprocessing/reports/preprocessing_summary.json` — phase-level count and outcome summary.
- `preprocessing/previews/final/` — ten before/after previews, four complete WARN/REJECT sheets, and the SIFT inlier chart.
- `analysis/reports/geometry_summary.json` — Step 6 source, configuration,
  measurements, exclusions, and artifact manifest.
- `analysis/geometry/` — pair, epipolar, and classical shape measurements.
- `analysis/previews/presentation/` — the six visually inspected real Step 6
  figures.

## Collaboration

Project contributors:

- Sithu Win San
- Eaint Myat Thu
- Gulizara Benjapalaporn

Read `CONTRIBUTING.md` before changing the repository and `AGENTS.md` before using an AI coding agent. Raw capture images must remain immutable.

## Course relevance

The project demonstrates image-quality measurement, feature detection and matching, geometric verification, Structure from Motion preparation, and later multi-view 3D reconstruction using a real Thai cultural object.

## License

Code and repository-authored material are released under the [MIT License](LICENSE) by Sithu Win San, Eaint Myat Thu, and Gulizara Benjapalaporn. Raw photographs and third-party assets are not automatically covered by the software license unless explicitly stated.
