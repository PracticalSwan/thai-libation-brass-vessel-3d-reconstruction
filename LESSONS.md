# Lessons Learned

Read this after `AGENTS.md` when starting substantive work. Keep process lessons here; keep current project state in `docs/memory-bank/`.

## Capture and preprocessing

- Do not over-engineer or over-test. This coursework benefits from simple, explainable code and verification proportional to risk.
- For deadline-bound reconstruction, research-grade gates are valuable diagnostics but must not create endless candidate churn. Use a bounded causal ladder; once exhausted, freeze the strongest defensible genuine CV artifact, keep failed checks explicitly failed, and finish the downstream pipeline with precise limitations rather than fabricating geometry.
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
- ALIKED-N16Rot + LightGlue is the current V4 matcher and must remain explicitly configured through extraction, matching, and import stages.
- Better sparse registration does **not** imply better dense surface reconstruction. Dense acceptance still requires direct geometry, coverage, contamination, connectivity, and visual checks.
- V4 does not schedule a SIFT/learned-method comparison. Pair ALIKED/LightGlue by circular phase: nearby angular views, wider local neighbors, orbit closure, and corresponding phases across elevation rings.
- Preserve lens-model consistency. If V4 uses identical fixed camera/lens/settings across rings, shared intrinsics are preferable to letting every frame drift independently. Do not force a shared camera across images whose real imaging contract differs.
- The reconstructed cameras represent relative object/camera motion. For a turntable sequence, coherent circular virtual-camera rings are expected even though the physical camera was stationary during each capture circle.
- Full image registration and a good global mean reprojection error do **not** prove pose integrity. Strong acquisition-adjacent two-view geometry must remain compatible with the final SfM relative poses and shared-track continuity; a pair with hundreds of verified inliers but a radically different final relative rotation or zero shared final tracks is a real sparse defect, not a dense-only problem.
- `phase_01` is ordered sequence phase, not measured physical azimuth. Use it as a locality prior only; cross-ring pairing/source selection must be dominated by verified inlier support, camera geometry and accepted co-visibility rather than forced phase-nearest edges.

## Dense reconstruction and meshing

- COLMAP PatchMatch geometric consistency is the default and only planned dense route after a coherent sparse model.
- Geometric stereo fusion must use vessel masks transformed consistently into the undistorted dense geometry so stationary background cannot be fused into the object cloud.
- COLMAP StereoFusion resolves masks as `<image_name>.png`; when the image name already includes `.jpg`, the mask filename is therefore `.jpg.png`. A PNG payload stored only under the JPEG filename can silently bypass the intended mask contract.
- Cross-view depth consistency must compare quantities expressed in the same camera frame. After backprojecting a reference depth sample and projecting the 3D point into a source view, compare the source depth map to the **reprojected source-camera Z**, never to the original reference-camera Z when the poses differ.
- Numeric continuity thresholds are diagnostics unless the authoritative project plan explicitly makes them acceptance criteria. A threshold such as `mean_consistent_fraction_at_1pct >= 0.50` must not override direct evidence that the real mask-constrained fused cloud is recognizable and free of dominant board/background contamination.
- Dense full-coordinate minima/maxima are sensitive to stereo outliers; inspect the actual cloud/mesh, not only scalar bounds.
- After sparse camera geometry changes, historical PatchMatch depth/normal maps are not final evidence unless compatibility is explicitly proven. Re-undistort/regenerate a fresh dense workspace rather than silently reusing maps tied to bad poses.
- A tiled PatchMatch run must assign each reference image exactly once. Source/dependency images may repeat, but duplicate reference writes across tiles make final map provenance ambiguous and can silently overwrite better maps.
- Component ratios are evidence, not a substitute for visual inspection, but gross fragmentation must be a hard rejection. The historical 644-component Poisson with only ~66.6% of faces in its largest component shows that finite/nonzero/recognizable gates alone are too permissive.
- Poisson can create a technically valid but visually wrong surface from noisy/fragmented dense points. Always show and preserve the raw surface before cleanup.
- If the chosen dense route hits a resource limit, make only the bounded same-method resolution retry. Do not switch to a competing 3D reconstruction stack merely to compare methods.
- After a host restart, persisted PatchMatch output counts are not completion proof: classify the prior log/workspace first, rerun only an incomplete versioned candidate, and require both photometric and geometric phases before fusion. When a current sparse report supersedes a historical name but the model/gate bytes are unchanged, preserve old reports and emit a new lineage wrapper that rehashes every current artifact. Component fractions can justify “strongest connected” selection, but they cannot prove semantic detached-component support; carry that limitation to the Blender handoff.

## Blender and final asset

- Technical mesh validity is not visual identity. The user/professor must be able to recognize the real object.
- Do not sculpt or hole-fill the first reconstruction preview before it is judged.
- After raw geometry passes, Blender cleanup may remove tiny components, smooth/relax conservatively, repair defensible holes, use controlled voxel-remesh/shrinkwrap only where necessary, create a practical production mesh, UV unwrap, and build the final material.
- If the geometry capture uses removable matte spray/marker spots, final brass appearance must come from a separate uncoated reference set rather than from the coated geometry images.
- A working final asset requires both an editable Blender master and a cleanly re-importable exported GLB.
- When execution is deliberately split at the Poisson/Blender boundary, the pre-Blender executor must freeze a hash-bound Poisson + evidence handoff and stop. Do not let a pre-Blender recovery task opportunistically mutate historical Blender/GLB artifacts; the designated Blender owner must start from the frozen handoff and independently verify the final scene/export.
- A technically present UV layer is not enough for production bakes. Inspect actual UV face-area utilization and the baked AO/normal images; highly fragmented scan UVs can pass presence checks while wasting almost the entire texture. A deterministic geometry-derived atlas is a valid technical repair when it changes UVs only and preserves mesh geometry/transforms.
- Apparent duplicate geometry in Blender must be separated from export duplication. Overlapping preserved CleanHigh and LOD0 objects can create viewport z-fighting even when a fresh GLB contains exactly one mesh; diagnose scene visibility and fresh-import object counts before deleting source meshes.
