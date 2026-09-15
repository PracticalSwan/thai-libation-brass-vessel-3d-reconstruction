# Thai Libation Brass Vessel 3D Reconstruction

Computer Vision coursework project for reconstructing a real Thai brass libation vessel from photographs.

## Current status — V4 repair, pre-Blender handoff frozen; Blender pending

V1 and V2 were rejected by the professor. V3 was also visually rejected after a full 266-view learned-feature photometric reconstruction was inspected in Blender. Historical evidence remains preserved where required, but the active target is the repaired V4 pipeline.

The final V4 source is immutable `CSX4213_Project_V4_Images/`: **688 JPEGs = 158 uncoated appearance/reference + 107 empty-board/background + 423 coated/marked object-bearing geometry images**. Historical photographs in `IMG20260826122949/` remain preserved as evidence/reference only and are not V4 reconstruction input. The previous V4 Blender/GLB continuation is preserved as diagnostic history, but final V4 completion is being re-established from the strongest defensible repaired sparse -> fresh dense -> Poisson lineage.

## Current verified repair state

- The repaired sparse pipeline has frozen v47 as the best-defensible 372-view source in `best_defensible_sparse_v2.json` (model SHA-256 `b1c4f142…23922e`; strict mask-projection failures remain explicit). The V2 selector also prevents the historical zero-residual failed v50b BA diagnostic from qualifying for downstream CV.
- The fresh v47-derived 2000px CUDA PatchMatch run remains the selected complete dense candidate: 372/372 geometric depth and normal maps, one-reference-one-write provenance, and fused-cloud SHA-256 `4a596f56…c7d73`. Its strict dense/anatomy gate still fails, so it is explicitly best-defensible rather than a strict pass.
- The bounded post-restart 3072px `geo_g12` recovery completed in a separate workspace with 37/37 photometric and geometric maps and fused-cloud SHA-256 `5bce4a2a…c980`. It did not improve the baseline (finial/lid ratio `1.0922269` vs `1.0177846`, unresolved), so it is retained as comparison-only evidence.
- The strongest scan-derived Poisson remains the depth-13/trim-5 shell: 2,158 components, dominant face fraction `0.9925223`, second-largest `0.0007196`, mesh SHA-256 `33fe1f6e…d6941`. Its finial and major-hole anatomy gates remain failed. The versioned pre-Blender handoff is `reconstruction/v4/reports/raw_poisson_best_defensible_v2_handoff.json` and records `handoff_allowed=true` while strict `promotion_allowed=false`.
- Existing Blender/GLB authoring outputs are preserved diagnostic evidence only for the current phase. They are not the active completion target and must not be modified by the pre-Blender executor.

## Current execution ownership

The work is deliberately split at the Poisson handoff. **Codex/local engineering completes everything before Blender**: restart recovery, bounded high-ring dense recovery, dense/Poisson selection, lineage, relevant tests/hashes/docs, and the focused pre-Blender Git milestone. It then stops. **ChatGPT using Blender MCP owns Blender cleanup and everything after Blender**: LOD0/UV/normal-detail/AO, project-derived appearance/material, final `.blend`/GLB, fresh re-import, visual verification, final docs, and final publication.

## V4 target

Produce one complete working 3D asset from the new photographs using one fixed pipeline, with a deliberate execution handoff after Poisson:

```text
new immutable fixed-camera turntable capture
-> FFmpeg/OpenCV ingest + ExifTool metadata + minimal QA
-> Grounding DINO-T + SAM 2.1 full-resolution vessel masks
-> conservative geometry-preserving preprocessing
-> ALIKED-N16Rot + LightGlue feature matching
-> COLMAP geometric verification
-> pyCOLMAP incremental SfM + bundle adjustment
-> image + mask undistortion
-> CUDA COLMAP PatchMatch geometric consistency
-> mask-aware geometric stereo fusion
-> Poisson raw mesh
-> PRE-BLENDER FREEZE + HASHED HANDOFF
   [Codex/local executor stops here]
-> Blender raw-mesh inspection
-> conservative cleanup / production mesh / UV / detail bake
-> brass PBR material from uncoated references
-> editable Blender master + final GLB
-> clean GLB re-import verification
   [ChatGPT + Blender MCP owns these stages]
```

V4 intentionally uses this one route. It does not schedule SIFT-vs-ALIKED, VGGT-vs-COLMAP, dense-method, or meshing bake-offs. A resource-only retry such as reducing PatchMatch from 2000 px to 1600 px is allowed when the same chosen algorithm hits VRAM limits.

## Authoritative V4 capture

The final acquisition is a **white-background fixed-camera turntable/object-rotation capture**. The audited set contains six object-bearing passes spanning lower/horizontal through upper/steep-high coverage, plus rotating empty-board sequences and a separate uncoated appearance/reference set. The geometry/empty sequences report one locked 3072x4080 OPPO Reno12 F rear-lens state at 3.98 mm / 26 mm-equivalent, digital zoom 1, ISO 100, 1/100 s, with manual-exposure/WB metadata.

The dry-shampoo coating reduced but did not eliminate brass reflections. Random black markers provide useful non-periodic texture across most of the bowl, globe, neck, pedestal, and transitions; finial/rim/interior/base-contact regions remain higher risk.

The white cloth background has visible stationary folds/seams, while the wooden board rotates with the vessel and is highly feature-rich. Board leakage is therefore a first-class SfM/MVS failure mode. Grounding DINO-T + SAM 2.1 is the authoritative isolation route; G8/G9 same-setup empty-board tails are used as negative refinement evidence near the base. Simple white-threshold segmentation is not the V4 plan, and no recapture is expected or requested.

## Fast execution policy

For the current V4 repair, continue from the newest verified checkpoint in the audited V4 implementation plan. After a restart, inspect persisted compute evidence before rerunning anything. Skip broad historical regression suites, algorithm comparisons, large reports, and parameter sweeps. Keep only the checks that prevent wasting reconstruction time:

1. V4 media decode correctly;
2. masks preserve the complete vessel;
3. sparse virtual camera rings/points are coherent and cross-elevation registration is connected;
4. the real raw dense cloud/Poisson mesh is visually plausible before cleanup and the pre-Blender handoff is hash-bound;
5. **post-Blender:** the final Blender master opens correctly;
6. **post-Blender:** the exported GLB cleanly re-imports.

## Retained reusable code

The project keeps capture-independent OpenCV QA/geometry utilities, ALIKED/LightGlue feature and database helpers, pyCOLMAP/COLMAP orchestration, PLY/mesh helpers, and segmentation utilities for V4. Legacy V3/Step-specific assumptions must be generalized rather than reused blindly.
