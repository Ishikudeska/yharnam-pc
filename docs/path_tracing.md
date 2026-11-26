# Ray-traced lighting for Bloodborne (path-tracing branch)

Goal: hardware ray-traced global illumination in the port's Vulkan renderer, on top of the
game's own deferred lighting, so fog, transparents, post-processing, tonemapping and the UI stay
the game's. The mod is a modified `bb-probe.exe`; the game files and the launcher stay unchanged.
Target hardware for development: RTX 3060 12 GB, 1080p output, DLSS.

## What can and cannot be traced

The renderer only sees GPU commands: draws, recompiled shaders, buffers and render targets. It
has no scene graph, no material list and no light list. So:

- **Visibility** can be traced against real triangles. The recompiled G-buffer vertex shaders
  already write their clip-space positions to GPU memory for object motion vectors
  (`EmitVertexMotion`, `shader_recompiler/backend/spirv/emit_spirv_special.cpp`). The same
  store, for every G-buffer draw, gives each frame's triangles. The scene constants give the
  inverse view-projection that turns them into world space.
- **Shading at a ray's hit point** cannot run the game's material shaders. Hits are shaded from
  radiance the game already computed: the previous frame's lit scene color when the hit point is
  on screen, and a world-space radiance cache filled from on-screen pixels when it is not. That
  feedback also gives multiple bounces over successive frames.
- **Geometry outside the view** is only in the BLAS if a pass drew it this frame. The G-buffer
  pass covers the view frustum; the shadow passes cover casters toward the sun/moon. Static
  geometry can be kept across frames by draw identity (the keys in `motion_history.h`).

Full path tracing in the strict sense (every light and material evaluated at every bounce) would
need the game's light and material data decoded from its shaders and constant buffers. That is
not in scope for the first version.

## Where it goes in the frame

From `docs/upscaler.md` (1920x1080):

| Pass | Targets | Mod |
|---|---|---|
| shadow | 4096² D32 | later: capture caster geometry |
| G-buffer | RGBA8 ×3, sRGB albedo, B10G11R11, RGBA16F + D32S8 | capture geometry; read normals, albedo, roughness |
| lighting | light volumes into two B10G11R11 | **after this pass: trace, add indirect light** |
| scene color | RGBA16F: combine, transparents, effects | unchanged |
| volumetric fog | compute (reads linear depth R32F), full-screen composite into scene color | **light the fog: shafts and shadows (step 5)** |
| post, tonemap, UI | | unchanged |

Hooks already in the rasterizer (`vk_rasterizer.cpp`): the scene constants
(`CameraMotion::OnConstants`), the G-buffer pass and its depth (`OnGBufferPass`), the scene
color target (`TemporalUpscaler::OnSceneColor`) and the display pass (`OnDisplayPass`). Compute
work is recorded between guest passes the way `CameraMotion::Overlay` does it
(`scheduler.EndRendering()`, `runtime.Transit`, dispatch).

## New code

All under `gpu/shadps4/video_core/renderer_vulkan/rt/`, off unless `BB_RT=1` (later a
`bbport.ini` setting and an in-game menu entry):

- `rt_capture` — the G-buffer vertex shader epilogue stores clip positions into a per-frame
  capture buffer; the CPU records each draw's range, index buffer, topology and instances.
- `rt_scene` — after the G-buffer pass: a compute pass turns clip positions into world positions;
  one BLAS per frame (fast build) and a one-instance TLAS. Later: static draws cached and refit.
- `rt_gi` — after the lighting pass: per pixel (half resolution first), position and normal from
  depth and the G-buffer, one cosine-weighted ray (`VK_KHR_ray_query`), hit shaded from the
  previous frame's scene color or the radiance cache, sky on a miss. Temporal accumulation with
  the port's camera motion vectors, then an à-trous spatial filter. The result is added to the
  diffuse light target.
- `rt_fog` — volumetric lighting for the game's fog (below).
- `rt_debug` — `BB_RT_DEBUG=<view>`: normals, albedo, ray hits, raw and filtered GI, fog
  density and in-scattering.

## What the frame holds (capture d000, Hunter's Dream, 2026-11-25)

Frame analyzer pass numbers from that capture; G-buffer slots in the order of the G-buffer pass.

| Image | Contents |
|---|---|
| G-buffer 0, RGBA8 | world-space normal, `rgb * 2 - 1` (unit length; checked against normals from depth: world space wins on 74% of wall pixels) |
| G-buffer 1, RGBA8 | material masks in r and b (b dark under foliage: probably baked shadowing) |
| G-buffer 2, RGBA8 sRGB | albedo; alpha a coverage/translucency mask |
| G-buffer 3, RGBA8 | specular color (gray) |
| G-buffer 4, B10G11R11 | **the game's ambient light**: bluish sky light with baked occlusion, no albedo; read only by the lighting pass's first draw (#62) |
| G-buffer 5, RGBA16F | ambient reflections (strong on wet stone) |
| G-buffer 6, RGBA16F | only one later G-buffer draw writes it; zero in this capture |
| depth D32S8 | view z = `proj.w / (depth - proj.z)`; stencil 9 world, 8 characters, 0 sky |
| shadow D32 4096² | only characters and small props here: environment shadows look baked |

Lighting: #62 (full screen) puts the ambient, the moon and screen-space reflections into the
two B10G11R11 targets, #63 adds the light volumes (lamps), **#64 multiplies the diffuse target
by the albedo**, #65–#66 (scene depth, stencil) look like skin scattering, #67 combines into the
scene color. Transparents and effects follow (#71–#85).

Fog: compute `a040875f` (#87) reads linear depth, the scene color, a 32×32 RGBA32F table and an
RGBA8 mask and writes an RGBA16F image the size of the scene; a draw with 2160 indices (#88, a grid,
not a full-screen quad) composites it into the scene color. #89 copies the finished scene color
for the next frame's reflections; post-processing starts with compute `9a9cf8a9` (#91).

## Volumetric fog

The game's fog is a per-pixel screen effect (above), not a volume. The mod keeps that fog where
the game puts it and lights it properly, so the moon and lamps cast light shafts through it and
geometry shadows it:

- **Volume:** a view-aligned froxel grid (about 160×90×64, exponential depth slices to ~200 m),
  computed after the lighting pass.
- **Density:** the game's fog density for each froxel, taken from its fog pass (its output volume
  or its parameters, found in the capture), times a multiplier. An optional base density adds
  fog where the game has none.
- **Light:** the moon or sun, with visibility traced against the captured geometry. The game's
  shadow map only holds characters and props (environment shadows are baked), so it cannot give
  shafts from buildings and trees. A Henyey-Greenstein phase function gives the
  forward-scattered shafts. Ambient light comes from the GI radiance cache once it exists. Lamps
  and torches come later, from the lighting pass's light volumes.
- **Stability:** jittered slice offsets and temporal reprojection of the volume, then
  front-to-back integration into in-scattering and transmittance.
- **Composite:** at the game's fog composite, either modulate the game's fog by the computed
  light visibility (keeps its colors, adds shafts and shadows) or replace it with the computed
  fog. Setting `rt_fog=off|shadowed|replace`, plus density and base-density settings.

The fog's light visibility needs the traced geometry, so it comes after the hardware rays.
Geometry outside the view casts no shafts until the static geometry cache exists.

## Steps

Each step ends with something that runs in the game.

1. **Build from source.** MSYS2 CLANG64 (`packaging/windows/README.md`), unmodified
   `out/bb-probe.exe` started with `BB_PROBE`; same behavior as the release.
2. **Frame analysis.** A frame dump (`BB_RT_DUMP_TRIGGER=<file>`): every target of the
   G-buffer and lighting passes, the shadow map, and the scene color before and after each
   full-screen pass over it, as images. Together with the port's frame analyzer
   (`BB_CAPTURE_TRIGGER`, every pass and dispatch with its shaders and images): the normal
   encoding, albedo, roughness, the game's ambient term, which lighting target is diffuse, the
   fog's compute pass, its output and its composite.
3. **Screen-space prototype** (works in game, 2026-11-25). At the lighting pass's first draw,
   `rt_gi_trace.comp` marches two cosine-weighted rays per half-resolution pixel through the
   depth buffer; hits take their light from the previous frame's finished scene color (copied
   at #89), misses leave the sky open; the result is accumulated over frames (bilinear,
   depth-checked reprojection, per-pixel history length). `rt_gi_apply.comp` upsamples and
   filters it (edge-aware, depth and normal) and rewrites G-buffer 4 in place:
   `ambient * mix(1, sky visibility, BB_RT_AO) + bounced light * BB_RT_BOUNCE`. The game's own
   lighting pass then reads it. Native scene size only (no reduced proxies yet).
4. **Hardware rays** (started). With `BB_RT_SCENE=1` (and `BB_OBJECT_MOTION_ALL=1`): the
   device gets acceleration structures and ray queries (`vk_instance.cpp`); every G-buffer
   draw that stores its clip positions for object motion is recorded (`RtScene::AddDraw`,
   indices by device address from `ResolveIndexBuffer`); at the lighting pass
   `rt_gather.comp` expands them into one world-space triangle list, built into a BLAS and a
   one-instance TLAS every frame (`rt/rt_scene.cpp`). A dump adds `rt_validate`: camera rays
   against the TLAS next to the depth buffer (`rt_dump.py` scores match / nearer / missing).
   First check (capture d001, Hunter's Dream): 93.9% of pixels match, 0% missing, 6.1% nearer
   (alpha-tested foliage traced as quads); 2.8M triangles of 133 draws; ~32 FPS against ~60.
   Second version: one BLAS per draw identity (hash of the draw's description and its order
   among identical draws), with its own world-space vertex storage. `rt_gather.comp` compares
   each triangle with the storage and flags moving draws; the CPU reads the flags four frames
   later: an entry is built once, rebuilt while it moves, left alone after 8 frames standing
   still. The TLAS holds this frame's entries plus those drawn in the last 3 frames; with
   `BB_RT_KEEP=1`, static entries drawn in the last 600 frames too (geometry off screen).
   The first try (refittable BLASes, everything kept 10 s) failed in game: 1.3 GB of
   acceleration structures on top of ~6 GB of textures, the GPU ran out of memory and the game
   fell to 5 FPS. Now plain fast builds (~4x smaller), entries go after 300 frames or past
   `BB_RT_MEMORY_MB` (768), none are created while device memory is over 85% of its budget
   (`VK_EXT_memory_budget` through VMA), and alpha-tested draws only count as moving past 25 cm
   (wind). `BB_OBJECT_MOTION_VERTICES` raises object motion's position budget (test mode: 6 Mi). Alpha-tested draws (`Shader::Info::has_discard`) get instance mask
   2: hardware rays include them with probability `BB_RT_FOLIAGE` (0.5). With the scene built,
   the indirect light uses `rt_gi_trace.comp` with `HW_RAYS` (ray queries; a hit takes its light
   from the previous frame where the hit point is visible on screen, else none yet). GPU
   timestamps print the cost of gather, BLAS, TLAS, trace and filter every 300 frames.
   Next: light for hits off screen (radiance cache), lighter capture (positions only, no motion
   vectors).
5. **Volumetric fog.** `rt_fog`: the froxel volume lit by the moon or sun with traced
   visibility, composited at the game's fog pass (#87/#88).
6. **Quality.** Radiance cache for off-screen hits, shadow-pass geometry, static geometry
   cache, better denoising, ray-traced reflections from the RGBA16F/roughness targets, lamps
   and ambient light in the fog.
7. **Settings and package.** `bbport.ini` keys, the launcher's graphics page, the in-game menu,
   a drop-in `bb-probe.exe`.

## Running a development build

    # MSYS2 CLANG64 shell, in the repository
    bash build.sh                                  # out/bb-probe.exe
    cp /c/Bloodborne/bbport-windows/bin/{bbport_dlss,nvngx_dlss,nvngx_dlssg}.dll out/

Then start the game with `BB_PROBE=<repository>\out\bb-probe.exe` and `BB_RT=1` in the
environment: `play-rt.cmd` in the project folder sets them (with the dump triggers and
`BB_TOGGLE_FILE`) and runs `Bloodborne.exe --play`, which uses the launcher's saved settings and
passes its environment on. `rt-toggle.cmd` switches the indirect light on and off while the game
runs; `capture.cmd` takes a dump. (The launcher's own extra-environment field splits on spaces,
so paths with spaces need their 8.3 names there.) `run.py` puts `C:\msys64\clang64\bin` on
`PATH` for a build without `SDL3.dll` beside it.

## Risks

- Cost: capturing positions for every G-buffer draw was measurably slow for object motion
  (`docs/motion_vectors.md`); BLAS builds of a full frame are several milliseconds on an RTX 3060.
- Reduced scene proxies (live resolution with DLSS/FSR presets) put the scene in other images;
  the first versions run at native size (DLAA / Native AA).
- Hits behind the camera miss geometry until the static cache exists.
- The game keeps data in the scene color's alpha; the composite must leave it alone.
- Fog: the game's fog may be computed at low resolution or as a 2D effect rather than a volume;
  then its density per froxel has to be estimated from its parameters, not read.
