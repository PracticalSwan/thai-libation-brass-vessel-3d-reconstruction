# Thai Libation Brass Vessel 3D Reconstruction

Computer vision coursework project for reconstructing a real Thai brass libation vessel from smartphone photographs.

The project demonstrates an explainable image-based 3D reconstruction pipeline:

```text
smartphone capture
-> OpenCV quality analysis and conservative preprocessing
-> pyCOLMAP feature extraction and matching
-> Structure from Motion / sparse reconstruction
-> dense reconstruction where practical
-> meshing and texturing
-> Blender cleanup and final model
```

## Project status

The current phase is **real-image preprocessing and QA**. The final capture set contains 297 portrait JPEG photographs of the same physical vessel, captured with an OPPO Reno12 F at 3072 x 4080.

The authoritative raw dataset is versioned in this public repository at:

```text
IMG20260826122949/
```

The separate local `IMG20260826122949.zip` is only a redundant archive of the same photographs and is not part of the Git publication set.

The preprocessing audit has confirmed that all 297 images are readable, share the same dimensions/camera/orientation, and contain no exact duplicates. Audit reports and contact-sheet evidence are also versioned so the preprocessing decisions remain reproducible. Full preprocessing selection and RAW-vs-preprocessed feature-match validation are still in progress.

## Why preprocessing is conservative

Polished brass is highly reflective. Moving specular highlights can create unstable local features, while the engraved patterns and geometric details provide useful correspondences. The pipeline therefore focuses on quality control and only mild photometric normalization.

It does **not** crop, rotate, warp, perspective-correct, synthesize detail, remove reflections with AI, or otherwise change image geometry.

## Planned Python workflow

The final preprocessing stage will expose simple, professor-explainable scripts for:

- image inventory and readability checks;
- blur, luminance, contrast, clipping, and feature-count measurements;
- `ACCEPT` / `WARN` / `REJECT` decisions based on the real dataset and visual review;
- conservative photometric normalization;
- representative before/after previews;
- RAW vs PREPROCESSED SIFT matching checks;
- deterministic export of the image set used by pyCOLMAP.

## Repository layout

```text
AGENTS.md                 canonical AI-agent rules
CLAUDE.md                 Claude entry point; imports AGENTS.md
preprocessing/            versioned QA reports and derived evidence
docs/                     design, plans, memory and handoff references
tests/                    focused preprocessing tests when implemented
IMG20260826122949/        versioned immutable raw photographs
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The currently verified local environment uses Python 3.14.2, OpenCV 4.13.0, NumPy 2.4.0, and Pillow 12.1.1. Future pyCOLMAP work must verify its installed/current API and Python compatibility rather than assuming these versions are sufficient.

## Collaboration

Project contributors:

- Sithu Win San
- Eaint Myat Thu
- Gulizara Benjapalaporn

Read `CONTRIBUTING.md` before changing the repository and `AGENTS.md` before using an AI coding agent. Raw capture images must remain immutable.

## Course relevance

The project directly demonstrates feature detection and matching, Structure from Motion, multi-view reconstruction, and 3D reconstruction using a real Thai cultural object.

## License

Code and repository-authored material are released under the [MIT License](LICENSE) by Sithu Win San, Eaint Myat Thu, and Gulizara Benjapalaporn. Raw photographs and third-party assets are not automatically covered by the software license unless explicitly stated.
