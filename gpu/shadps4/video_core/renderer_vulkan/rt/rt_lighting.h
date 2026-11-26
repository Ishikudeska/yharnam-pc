// SPDX-License-Identifier: GPL-2.0-or-later
// bbport path tracing (docs/path_tracing.md): follows the G-buffer and lighting passes of each
// frame and adds indirect light to the game's ambient term before the lighting pass reads it.
//
// BB_RT=1 turns it on. Step 3: screen-space indirect light (rt_gi_trace.comp, rt_gi_apply.comp),
// BB_RT_GI=0 turns it off. Settings: BB_RT_AO (0.5), BB_RT_BOUNCE (1.0), BB_RT_DISTANCE (4.0),
// BB_RT_THICKNESS (0.5), BB_RT_HISTORY (0.9). BB_TOGGLE_FILE bits while the game runs: 1<<60
// indirect light off, 1<<61 bounced light only, 1<<62 sky visibility only.
//
// BB_RT_DUMP_TRIGGER=<file>: creating the file writes one frame to BB_RT_DUMP_DIR (default
// out/rt-dump) as raw files dNNN_<name>_<w>x<h>_<format>.raw (tools/rt/rt_dump.py): the G-buffer
// and shadow map before the indirect light, the ambient term and indirect light after it, the
// lighting targets, the scene constants and the finished scene color.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/rt/rt_scene.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;

class RtLighting {
public:
    static constexpr u32 MaxTargets = 8;
    static constexpr u32 SceneConstantFloats = 864 / 4;

    RtLighting(const Instance& instance, Scheduler& scheduler,
               VideoCore::TextureCache& texture_cache, Runtime& runtime);
    ~RtLighting();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    /// The scene constants bound by a G-buffer draw (864 bytes, see CameraMotion::OnConstants);
    /// the first of a frame is the main camera's.
    void OnConstants(const float* data);

    /// A draw's render targets (PrepareRenderState): the color target of each slot (empty for
    /// unused slots) and the depth target. Runs the work at the lighting pass's start.
    void OnTargets(const std::array<VideoCore::ImageId, MaxTargets>& colors,
                   VideoCore::ImageId depth);

    /// The display pass: the previous frame is finished. `jitter` is the new frame's viewport
    /// shift in pixels (TemporalUpscaler::Jitter).
    void OnFrameStart(std::array<float, 2> jitter);

    /// Step 4: G-buffer geometry goes into acceleration structures (any thread may ask).
    [[nodiscard]] bool CapturesGeometry() const noexcept {
        return scene && scene->Enabled();
    }
    /// A G-buffer draw whose vertex shader stored its clip positions (see RtScene::AddDraw).
    void OnGeometryDraw(const RtScene::DrawInput& draw);

private:
    enum class Phase : u8 {
        Idle,     ///< before this frame's G-buffer pass
        GBuffer,  ///< G-buffer draws seen
        Lighting, ///< the lighting pass started (ambient, sun/moon, light volumes)
        Done,     ///< the lighting is combined with the albedo; scene color draws
    };

    struct Camera {
        std::array<float, 12> view{};     ///< world to view, 3x4 rows
        std::array<float, 12> inv_view{}; ///< view to world, 3x4 rows
        std::array<float, 4> proj{};      ///< x scale, y scale, z scale, z offset
        std::array<float, 2> jitter{};
    };

    /// An image of this module, always in layout General.
    struct OwnImage {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
    };

    struct DumpItem {
        VideoCore::ImageId id;            ///< a texture cache image, or
        vk::Image raw{};                  ///< one of this module's images (layout General)
        u32 width = 0, height = 0;        ///< size and format of `raw`
        vk::Format format{};
        std::string name;
        vk::ImageAspectFlagBits aspect = vk::ImageAspectFlagBits::eColor;
    };

    /// The first draw of the lighting pass: the G-buffer is complete and not read yet.
    void OnLightingStart();
    /// The first draw after the scene color pass: the scene color is finished (with fog).
    void OnSceneComplete();

    void CreateGiPipelines();
    void CreateValidatePipeline();
    void CreateHwTracePipeline();
    /// Adds the GPU times of a timestamp slot's last frame; prints averages now and then.
    void ReadTimestamps(u32 slot);
    bool EnsureGiImages(u32 width, u32 height);
    /// The main camera of this frame (scene constants).
    [[nodiscard]] Camera CurrentCamera() const;
    /// Step 4 check, dumped as rt_validate: camera rays against the acceleration structure.
    void RunValidation(vk::ImageView depth_view, u32 width, u32 height);
    /// Traces and filters the indirect light and writes it into the ambient term.
    bool RunGi();
    /// A view of a texture cache image, destroyed once the GPU is done with this frame.
    vk::ImageView FrameView(const VideoCore::Image& image, vk::Format format,
                            vk::ImageAspectFlags aspect);

    /// Copies the images after the commands recorded so far; files are written once the GPU
    /// is done with them.
    void DumpImages(const std::vector<DumpItem>& items);
    static DumpItem Item(VideoCore::ImageId id, std::string name,
                         vk::ImageAspectFlagBits aspect = vk::ImageAspectFlagBits::eColor) {
        DumpItem item;
        item.id = id;
        item.name = std::move(name);
        item.aspect = aspect;
        return item;
    }

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    bool enabled = false;

    // Frame state.
    Phase phase = Phase::Idle;
    std::array<VideoCore::ImageId, MaxTargets> gbuffer{};
    VideoCore::ImageId depth{};
    std::array<VideoCore::ImageId, 2> lighting{};
    VideoCore::ImageId shadow{};      ///< the largest square depth-only target before the G-buffer
    VideoCore::ImageId scene_color{}; ///< RGBA16F target with the scene depth (phase Done)
    bool scene_complete = false;
    std::array<float, SceneConstantFloats> constants{};
    bool frame_has_constants = false;
    std::array<float, 2> jitter{};

    // Indirect light (step 3).
    bool gi_enabled = false;
    float ao_strength = 0.5f;
    float bounce_strength = 1.0f;
    float max_distance = 4.0f;
    float thickness = 0.5f;
    float history_weight = 0.9f;
    float max_radiance = 8.0f;
    float foliage_opacity = 0.5f; ///< chance that alpha-tested geometry stops a hardware ray
    vk::UniqueDescriptorSetLayout trace_set_layout, apply_set_layout;
    vk::UniquePipelineLayout trace_layout, apply_layout;
    vk::UniquePipeline trace_pipeline, apply_pipeline;
    u32 gi_width = 0, gi_height = 0;
    OwnImage radiance;               ///< the previous frame's finished scene color
    std::array<OwnImage, 2> result;  ///< accumulated indirect light, ping-pong
    std::array<OwnImage, 2> result_z; ///< view z of each result pixel
    u32 result_index = 0;
    bool radiance_valid = false;
    bool history_valid = false;
    bool gi_ran = false;              ///< this frame's ambient term holds the indirect light
    Camera previous;                  ///< the camera of the last frame with indirect light
    u32 gi_frame = 0;

    // Acceleration structures (step 4).
    std::unique_ptr<RtScene> scene;
    bool scene_built = false;          ///< this frame's TLAS holds this frame's geometry
    vk::UniqueDescriptorSetLayout trace_hw_set_layout;
    vk::UniquePipelineLayout trace_hw_layout;
    vk::UniquePipeline trace_hw_pipeline; ///< rt_gi_trace.comp with HW_RAYS
    // GPU timing (RtLighting::ReadTimestamps).
    vk::UniqueQueryPool timestamps;
    double timestamp_period_ms = 0.0;
    std::array<bool, 4> timestamp_written{};
    std::array<double, 5> timing_sum{};
    u32 timing_samples = 0;
    vk::UniqueDescriptorSetLayout validate_set_layout;
    vk::UniquePipelineLayout validate_layout;
    vk::UniquePipeline validate_pipeline;
    OwnImage validate;                 ///< rg32f: hit view z, depth view z (full resolution)

    // Dumps.
    const char* dump_trigger = nullptr;
    std::string dump_dir;
    bool dump_armed = false;
    bool dumping = false;       ///< this frame is being dumped
    u32 dump_index = 0;         ///< number of the dump in progress (file prefix dNNN)
    u64 frame = 0;
    u32 logged_frames = 0;
};

} // namespace Vulkan
