# Real Preprocessing and Public Repository Design

## Objective

Finish the real-image preprocessing stage for the Thai Libation Brass Vessel 3D reconstruction project, then publish the verified code/documentation as a public GitHub repository.

## Source Data

- `IMG20260826122949/` is the authoritative 297-image raw capture set.
- Raw files are immutable: never overwrite, rename, resize, rotate, crop, recompress, or normalize them in place.
- Do not over-engineer, over-complicate, or over-test; use the smallest coherent workflow and risk-proportional verification.
- The user subsequently authorized publication of the reviewed raw capture photographs and image-processing evidence. The raw images remain immutable even though they are versioned publicly.

## Preprocessing Workflow

1. Inventory all raw images and record dimensions, readability, EXIF orientation/camera metadata where available, hashes, and quality metrics.
2. Measure blur, luminance, contrast, dark/bright clipping, color statistics, and SIFT feature counts on a standardized analysis scale.
3. Detect exact and near duplicates conservatively.
4. Use dataset-relative statistics plus visual inspection to label images `ACCEPT`, `WARN`, or `REJECT`; preserve angular coverage and overlap.
5. Apply only mild photometric normalization to accepted images; never apply geometric warps.
6. Compare representative RAW vs PREPROCESSED neighboring-frame SIFT matching and keep the variant that preserves or improves geometric matching.
7. Produce CSV/JSON reports, representative preview composites, and a photogrammetry-ready derived image set.

## Repository Design

Repository name: `thai-libation-brass-vessel-3d-reconstruction`.

Description: Computer vision project reconstructing a Thai brass libation vessel from smartphone photographs using OpenCV preprocessing/QA, pyCOLMAP Structure from Motion, and Blender.

The repository will contain reproducible source code, tests, documentation, the reviewed raw photographs, intentional image-processing reports/previews/derived datasets, MIT license, security policy, contribution guidance, and shared AI-agent instructions.

## Collaboration

- Sithu Win San, Eaint Myat Thu, and Gulizara Benjapalaporn are project authors/copyright holders.
- The repository owner will invite the two teammates manually; agents must not send collaborator invitations.
- `AGENTS.md` is the canonical project-local instruction file; `CLAUDE.md` imports it.
- Contributors and AI agents must preserve raw image immutability, avoid destructive cleanup, run the documented verification commands, and never commit secrets or disposable task residue.

## Verification

- Hash the raw set before and after processing and confirm no change.
- Compile all preprocessing Python files.
- Run the complete pipeline on all 297 raw photographs.
- Confirm report/output counts and readability.
- Visually inspect contact sheets and representative before/after previews.
- Verify RAW vs PREPROCESSED feature matching before selecting the final reconstruction input set.
- Inspect Git status and intended diff before the first commit/push.
- Verify the public repository metadata and committed files after publication.
- Stop before pyCOLMAP reconstruction; that is the next stage.

