# Lessons Learned

Read this after `AGENTS.md` when starting substantive work. Keep process lessons here; keep current project state in `docs/memory-bank/`.

## Capture and preprocessing

- Do not over-engineer or over-test. This coursework benefits from simple, explainable code and verification proportional to risk.
- Numerical image-quality thresholds are not authoritative by themselves. Validate them against the real capture distribution and visual coverage.
- Polished brass produces moving specular highlights; reflection alone is not a reason to reject a frame, but uncontrolled reflections can destroy dense multi-view consistency.
- Raw smartphone images are immutable source evidence. Derived data belongs outside the raw directory.
- A rotating-object sequence with a static background is valid for V4 only after the stationary background is excluded. The authoritative V4 capture is now a fixed-camera turntable sequence: the vessel rotates, background does not, so segmentation/masking must happen before learned feature extraction and must also constrain dense fusion.
- On the white-background V4 setup, do not assume white-threshold segmentation will separate the vessel because the temporary dry-shampoo coating can be similarly light. Grounding DINO-T + SAM 2.1 is the primary isolation route; a fixed-camera empty-background reference is a refinement cue rather than a competing segmentation pipeline.
- One horizontal orbit cannot observe the top, bowl interior, finial/lid transitions, and lower pedestal transitions well enough. Use multiple complete 360-degree circles at different fixed camera elevations while keeping the turntable axis fixed.
- Temporary matte coating suppresses unstable specular highlights, while random non-periodic black spots on the coating supply local texture. Treat those marks only as correspondence texture and capture separate uncoated material references.
- Choose preprocessing from actual geometric correspondence evidence. Do not copy the old LAB-CLAHE decision blindly to V4.

## Feature matching and sparse reconstruction

- Learned matching must pass the chosen feature/matcher configuration explicitly through every extraction/matching/import stage; silent fallback to SIFT invalidates the experiment.
- Cache provenance should identify per-image feature layout, mask identity, model/config fingerprint, and source image identity rather than only aggregate counts.
- ALIKED-N16Rot + LightGlue recovered far more camera coverage than the old SIFT reconstruction on the original capture, so it remains the V4 matcher.
- Better sparse registration does **not** imply better dense surface reconstruction. V3 reached 266 registered views yet still produced an unacceptable mesh.
- V4 does not schedule a SIFT/learned-method comparison. Pair ALIKED/LightGlue by circular phase: nearby angular views, wider local neighbors, orbit closure, and corresponding phases across elevation rings.
- Preserve lens-model consistency. If V4 uses identical fixed camera/lens/settings across rings, shared intrinsics are preferable to letting every frame drift independently. Do not force a shared camera across images whose real imaging contract differs.
- The reconstructed cameras represent relative object/camera motion. For a turntable sequence, coherent circular virtual-camera rings are expected even though the physical camera was stationary during each capture circle.

## Dense reconstruction and meshing

- COLMAP PatchMatch geometric consistency is the default and only planned dense route after a coherent sparse model.
- Geometric stereo fusion must use vessel masks transformed consistently into the undistorted dense geometry so stationary background cannot be fused into the object cloud.
- COLMAP StereoFusion resolves masks as `<image_name>.png`; when the image name already includes `.jpg`, the mask filename is therefore `.jpg.png`. A PNG payload stored only under the JPEG filename can silently bypass the intended mask contract.
- Cross-view depth consistency must compare quantities expressed in the same camera frame. After backprojecting a reference depth sample and projecting the 3D point into a source view, compare the source depth map to the **reprojected source-camera Z**, never to the original reference-camera Z when the poses differ.
- Numeric continuity thresholds are diagnostics unless the authoritative project plan explicitly makes them acceptance criteria. A threshold such as `mean_consistent_fraction_at_1pct >= 0.50` must not override direct evidence that the real mask-constrained fused cloud is recognizable and free of dominant board/background contamination.
- Dense full-coordinate minima/maxima are sensitive to stereo outliers; inspect the actual cloud/mesh, not only scalar bounds.
- Component ratios are evidence, not a substitute for visual inspection. A large dominant component can still be the wrong shape.
- Poisson can create a technically valid but visually wrong surface from noisy/fragmented dense points. Always show and preserve the raw surface before cleanup.
- If the chosen dense route hits a resource limit, make only the bounded same-method resolution retry. Do not switch to a competing 3D reconstruction stack merely to compare methods.

## Blender and final asset

- Technical mesh validity is not visual identity. The user/professor must be able to recognize the real object.
- Do not sculpt or hole-fill the first reconstruction preview before it is judged.
- After raw geometry passes, Blender cleanup may remove tiny components, smooth/relax conservatively, repair defensible holes, use controlled voxel-remesh/shrinkwrap only where necessary, create a practical production mesh, UV unwrap, and build the final material.
- If the geometry capture uses removable matte spray/marker spots, final brass appearance must come from a separate uncoated reference set rather than from the coated geometry images.
- A working final asset requires both an editable Blender master and a cleanly re-importable exported GLB.

## V3 rejection lesson — 2026-09-10

The V3 ALIKED-N16Rot + LightGlue solution registered 266/288 views and the full photometric run produced 266 depth/normal maps, about 80,015 fused dense points, and a 657,693-face Poisson mesh. Direct Blender inspection was still visually poor. The old capture is therefore abandoned for reconstruction. **V4 must spend effort on acquisition quality first, not on further parameter tuning of the rejected dataset.**
