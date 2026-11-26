// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/rt/rt_lighting.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <vk_mem_alloc.h>

#include "bbport_toggles.h"
#include "video_core/host_shaders/rt_gi_apply_comp.h"
#include "video_core/host_shaders/rt_gi_trace_comp.h"
#include "video_core/host_shaders/rt_gi_trace_hw_comp.h"
#include "video_core/host_shaders/rt_validate_comp.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scene_resolution.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {

namespace {

/// BB_TOGGLE_FILE bits (bbport_toggles.h leaves 60-63 free).
constexpr u64 ToggleGiOff = 1ull << 60;
constexpr u64 ToggleBounceOnly = 1ull << 61;
constexpr u64 ToggleVisibilityOnly = 1ull << 62;

/// rt_gi_trace.comp push constants (240 bytes: needs maxPushConstantsSize >= 240).
struct TracePush {
    std::array<float, 12> view_rot; ///< world -> view, 3x4 rows
    std::array<float, 12> reproject;
    std::array<float, 4> proj;
    std::array<float, 4> prev_proj;
    std::array<float, 2> size;
    std::array<float, 2> jitter;
    std::array<float, 2> prev_jitter;
    float max_distance;
    float thickness;
    u32 frame;
    u32 flags;
    float history_weight;
    float max_radiance;
    std::array<float, 12> inv_view; ///< view -> world, 3x4 rows (hardware rays)
    float foliage_opacity;
    float pad[3];
};
static_assert(sizeof(TracePush) == 240);

/// GPU timestamps of a frame's work (RtLighting::RunGi): scene start, gathered, BLAS done,
/// TLAS done (= trace start), traced, filtered.
constexpr u32 TimestampsPerSlot = 8;
constexpr u32 TimestampSlots = 4;

/// rt_gi_apply.comp push constants.
struct ApplyPush {
    std::array<float, 4> proj;
    std::array<float, 2> size;
    float ao_strength;
    float bounce_strength;
    u32 mode;
};
static_assert(sizeof(ApplyPush) == 36);

/// rt_validate.comp push constants.
struct ValidatePush {
    std::array<float, 12> inv_view;
    std::array<float, 4> proj;
    std::array<float, 2> size;
    std::array<float, 2> jitter;
};
static_assert(sizeof(ValidatePush) == 80);

/// a * b for 3x4 affine matrices (rows [R | t]).
std::array<float, 12> Multiply(const std::array<float, 12>& a, const std::array<float, 12>& b) {
    std::array<float, 12> out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            float v = c == 3 ? a[r * 4 + 3] : 0.0f;
            for (int k = 0; k < 3; ++k) {
                v += a[r * 4 + k] * b[k * 4 + c];
            }
            out[r * 4 + c] = v;
        }
    }
    return out;
}

/// Bytes per pixel of one aspect of `format` as vkCmdCopyImageToBuffer writes it, or 0.
u32 CopyBytesPerPixel(vk::Format format, vk::ImageAspectFlagBits aspect) {
    if (aspect == vk::ImageAspectFlagBits::eStencil) {
        return 1;
    }
    switch (format) {
    case vk::Format::eR8Unorm:
    case vk::Format::eR8Uint:
        return 1;
    case vk::Format::eR8G8Unorm:
    case vk::Format::eR16Sfloat:
    case vk::Format::eR16Unorm:
    case vk::Format::eR16Uint:
    case vk::Format::eD16Unorm:
        return 2;
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eR8G8B8A8Snorm:
    case vk::Format::eR8G8B8A8Uint:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eE5B9G9R9UfloatPack32:
    case vk::Format::eR16G16Sfloat:
    case vk::Format::eR16G16Unorm:
    case vk::Format::eR32Sfloat:
    case vk::Format::eR32Uint:
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint:
        return 4;
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eR32G32Sfloat:
        return 8;
    case vk::Format::eR32G32B32A32Sfloat:
        return 16;
    default:
        return 0;
    }
}

bool HasStencil(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD16UnormS8Uint;
}

bool IsDepth32(vk::Format format) {
    return format == vk::Format::eD32Sfloat || format == vk::Format::eD32SfloatS8Uint;
}

unsigned long long Ull(u64 value) {
    return static_cast<unsigned long long>(value);
}

float EnvFloat(const char* name, float fallback, float lo, float hi) {
    const char* value = std::getenv(name);
    if (!value || !value[0]) {
        return fallback;
    }
    const float parsed = std::strtof(value, nullptr);
    return parsed >= lo && parsed <= hi ? parsed : fallback;
}

void RtBarrier(vk::CommandBuffer cmdbuf, vk::PipelineStageFlags2 src_stage,
                   vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
                   vk::AccessFlags2 dst_access) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
    };
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

/// Logs a reason the indirect light did not run, once per reason.
void SkipOnce(bool& logged, const char* reason) {
    if (!logged) {
        logged = true;
        std::printf("RT: indirect light skipped: %s\n", reason);
    }
}

} // namespace

RtLighting::RtLighting(const Instance& instance_, Scheduler& scheduler_,
                       VideoCore::TextureCache& texture_cache_, Runtime& runtime_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_},
      runtime{runtime_} {
    const char* env = std::getenv("BB_RT");
    enabled = env && env[0] == '1';
    if (!enabled) {
        return;
    }
    dump_trigger = std::getenv("BB_RT_DUMP_TRIGGER");
    if (dump_trigger && !dump_trigger[0]) {
        dump_trigger = nullptr;
    }
    const char* dir = std::getenv("BB_RT_DUMP_DIR");
    dump_dir = dir && dir[0] ? dir : "out/rt-dump";
    // Dumps of earlier runs stay: numbering continues after the highest dNNN_ in the folder.
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(dump_dir, error)) {
        unsigned number = 0;
        if (std::sscanf(entry.path().filename().string().c_str(), "d%3u_", &number) == 1) {
            dump_index = std::max(dump_index, number + 1);
        }
    }

    const char* gi = std::getenv("BB_RT_GI");
    gi_enabled = !(gi && gi[0] == '0');
    ao_strength = EnvFloat("BB_RT_AO", ao_strength, 0.0f, 1.0f);
    bounce_strength = EnvFloat("BB_RT_BOUNCE", bounce_strength, 0.0f, 8.0f);
    max_distance = EnvFloat("BB_RT_DISTANCE", max_distance, 0.1f, 50.0f);
    thickness = EnvFloat("BB_RT_THICKNESS", thickness, 0.01f, 10.0f);
    history_weight = EnvFloat("BB_RT_HISTORY", history_weight, 0.0f, 0.99f);
    foliage_opacity = EnvFloat("BB_RT_FOLIAGE", foliage_opacity, 0.0f, 1.0f);
    const auto limits = instance.GetPhysicalDevice().getProperties().limits;
    if (limits.timestampComputeAndGraphics && limits.timestampPeriod > 0.0f) {
        timestamp_period_ms = double(limits.timestampPeriod) * 1e-6;
        timestamps = Check(instance.GetDevice().createQueryPoolUnique({
            .queryType = vk::QueryType::eTimestamp,
            .queryCount = TimestampsPerSlot * TimestampSlots,
        }));
    }
    const u32 push_limit = limits.maxPushConstantsSize;
    if (gi_enabled && push_limit < sizeof(TracePush)) {
        std::printf("RT: indirect light off: %u bytes of push constants, needs %zu\n", push_limit,
                    sizeof(TracePush));
        gi_enabled = false;
    }
    if (gi_enabled && !instance.IsFormatSupported(vk::Format::eB10G11R11UfloatPack32,
                                                  vk::FormatFeatureFlagBits2::eStorageImage)) {
        std::printf("RT: indirect light off: no storage image support for B10G11R11\n");
        gi_enabled = false;
    }
    if (gi_enabled) {
        CreateGiPipelines();
    }
    // Step 4: acceleration structures of the G-buffer geometry (the device has ray queries
    // with BB_RT=1 where the GPU supports them).
    if (instance.IsRayQuerySupported()) {
        scene = std::make_unique<RtScene>(instance, scheduler);
        if (scene->Enabled()) {
            CreateValidatePipeline();
            if (gi_enabled) {
                CreateHwTracePipeline();
            }
            const char* all_motion = std::getenv("BB_OBJECT_MOTION_ALL");
            if (!(all_motion && all_motion[0] == '1')) {
                std::printf("RT: BB_OBJECT_MOTION_ALL is not 1: only animated draws reach the "
                            "acceleration structure\n");
            }
        }
    }
    std::printf("RT: on (docs/path_tracing.md); indirect light %s (ao %.2f, bounce %.2f, "
                "distance %.1f, thickness %.2f, history %.2f)%s%s\n",
                gi_enabled ? "on" : "off", ao_strength, bounce_strength, max_distance, thickness,
                history_weight, dump_trigger ? ", dump trigger " : "",
                dump_trigger ? dump_trigger : "");
}

RtLighting::~RtLighting() = default;

void RtLighting::CreateGiPipelines() {
    const auto device = instance.GetDevice();
    const auto make_layout = [&](std::initializer_list<vk::DescriptorType> types) {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        u32 index = 0;
        for (const auto type : types) {
            bindings.push_back({.binding = index++,
                                .descriptorType = type,
                                .descriptorCount = 1,
                                .stageFlags = vk::ShaderStageFlagBits::eCompute});
        }
        return Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
    };
    using T = vk::DescriptorType;
    trace_set_layout = make_layout({T::eSampledImage, T::eSampledImage, T::eSampledImage,
                                    T::eSampledImage, T::eSampledImage, T::eStorageImage,
                                    T::eStorageImage});
    apply_set_layout = make_layout({T::eSampledImage, T::eSampledImage, T::eSampledImage,
                                    T::eSampledImage, T::eStorageImage});

    const auto make_pipeline = [&](vk::DescriptorSetLayout set_layout, u32 push_size,
                                   std::span<const u32> code, vk::UniquePipelineLayout& layout,
                                   vk::UniquePipeline& pipeline) {
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = push_size};
        layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(code, device);
        pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = module,
                              .pName = "main"},
                    .layout = *layout,
                }));
        device.destroyShaderModule(module);
    };
    make_pipeline(*trace_set_layout, sizeof(TracePush), RT_GI_TRACE_COMP, trace_layout,
                  trace_pipeline);
    make_pipeline(*apply_set_layout, sizeof(ApplyPush), RT_GI_APPLY_COMP, apply_layout,
                  apply_pipeline);
}

void RtLighting::CreateValidatePipeline() {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    validate_set_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(ValidatePush)};
    validate_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*validate_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(RT_VALIDATE_COMP, device);
    validate_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *validate_layout,
            }));
    device.destroyShaderModule(module);
}

void RtLighting::CreateHwTracePipeline() {
    // rt_gi_trace.comp with HW_RAYS: the screen-space bindings and the TLAS at binding 7.
    const auto device = instance.GetDevice();
    std::array<vk::DescriptorSetLayoutBinding, 8> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i] = {
            .binding = i,
            .descriptorType = i == 7   ? vk::DescriptorType::eAccelerationStructureKHR
                              : i >= 5 ? vk::DescriptorType::eStorageImage
                                       : vk::DescriptorType::eSampledImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        };
    }
    trace_hw_set_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(TracePush)};
    trace_hw_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*trace_hw_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(RT_GI_TRACE_HW_COMP, device);
    trace_hw_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *trace_hw_layout,
            }));
    device.destroyShaderModule(module);
}

void RtLighting::ReadTimestamps(u32 slot) {
    // The slot's last frame: scene start, gathered, BLAS done, trace start, traced, filtered.
    std::array<u64, 12> data{};
    const auto result = instance.GetDevice().getQueryPoolResults(
        *timestamps, slot * TimestampsPerSlot, 6, sizeof(data), data.data(), 2 * sizeof(u64),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
    if (result != vk::Result::eSuccess && result != vk::Result::eNotReady) {
        return;
    }
    const auto available = [&](u32 i) { return data[i * 2 + 1] != 0; };
    const auto ms = [&](u32 a, u32 b) {
        return data[b * 2] >= data[a * 2] ? double(data[b * 2] - data[a * 2]) * timestamp_period_ms
                                          : 0.0;
    };
    if (!available(3) || !available(4) || !available(5)) {
        return;
    }
    const bool scene_timed = available(0) && available(1) && available(2);
    timing_sum[0] += scene_timed ? ms(0, 1) : 0.0;
    timing_sum[1] += scene_timed ? ms(1, 2) : 0.0;
    timing_sum[2] += scene_timed ? ms(2, 3) : 0.0;
    timing_sum[3] += ms(3, 4);
    timing_sum[4] += ms(4, 5);
    if (++timing_samples == 300) {
        const double n = timing_samples;
        const double total = (timing_sum[0] + timing_sum[1] + timing_sum[2] + timing_sum[3] +
                              timing_sum[4]) / n;
        std::printf("RT: GPU time (ms, average of %u frames): gather %.2f, BLAS %.2f, TLAS "
                    "%.2f, trace %.2f, filter %.2f; total %.2f\n",
                    timing_samples, timing_sum[0] / n, timing_sum[1] / n, timing_sum[2] / n,
                    timing_sum[3] / n, timing_sum[4] / n, total);
        if (scene && scene->Enabled()) {
            const auto& s = scene->LastStats();
            std::printf("RT: scene now: %u draws (%u triangles): %u built, %u rebuilt, "
                        "%u unchanged; %u kept off screen; %u entries, %.0f MiB\n",
                        s.drawn, s.triangles, s.built, s.rebuilt, s.skipped, s.kept, s.entries,
                        double(s.bytes) / 1048576.0);
        }
        timing_sum = {};
        timing_samples = 0;
    }
}

bool RtLighting::EnsureGiImages(u32 width, u32 height) {
    if (width == gi_width && height == gi_height && radiance.image) {
        return true;
    }
    const auto device = instance.GetDevice();
    // The old images may still be in use by the GPU: destroyed after the work recorded so far.
    const auto retire = [&](OwnImage& own) {
        if (own.image) {
            scheduler.DeferOperation([image = std::move(own.image),
                                      view = std::move(own.view)]() mutable {
                view.reset();
                image.Destroy();
            });
        }
    };
    retire(radiance);
    retire(validate);
    for (u32 i = 0; i < 2; ++i) {
        retire(result[i]);
        retire(result_z[i]);
    }
    radiance_valid = history_valid = false;
    gi_width = gi_height = 0;

    // The indirect light is traced at half resolution (rt_gi_trace.comp).
    const u32 half_width = (width + 1) / 2, half_height = (height + 1) / 2;
    std::vector<vk::Image> created;
    const auto create = [&](OwnImage& own, vk::Format format, vk::ImageUsageFlags usage,
                            bool half) {
        own.image = VideoCore::UniqueImage(device, instance.GetAllocator());
        if (!own.image.TryCreate(vk::ImageCreateInfo{
                .imageType = vk::ImageType::e2D,
                .format = format,
                .extent = {half ? half_width : width, half ? half_height : height, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1,
                .tiling = vk::ImageTiling::eOptimal,
                .usage = usage | vk::ImageUsageFlagBits::eSampled |
                         vk::ImageUsageFlagBits::eTransferSrc,
                .initialLayout = vk::ImageLayout::eUndefined,
            })) {
            return false;
        }
        own.view = Check(device.createImageViewUnique({
            .image = vk::Image(own.image),
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
        created.push_back(vk::Image(own.image));
        return true;
    };
    bool ok = create(radiance, vk::Format::eR16G16B16A16Sfloat,
                     vk::ImageUsageFlagBits::eTransferDst, false);
    for (u32 i = 0; i < 2 && ok; ++i) {
        ok = create(result[i], vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage,
                    true) &&
             create(result_z[i], vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eStorage, true);
    }
    if (ok && scene && scene->Enabled()) {
        ok = create(validate, vk::Format::eR32G32Sfloat, vk::ImageUsageFlagBits::eStorage, false);
    }
    if (!ok) {
        std::printf("RT: no GPU memory for the indirect light images (%ux%u)\n", width, height);
        return false;
    }
    // All of this module's images stay in layout General.
    std::vector<vk::ImageMemoryBarrier2> barriers;
    for (const auto image : created) {
        barriers.push_back({
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .srcAccessMask = vk::AccessFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                            vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                             vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
    }
    scheduler.CommandBuffer().pipelineBarrier2({
        .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    });
    gi_width = width;
    gi_height = height;
    std::printf("RT: indirect light images %ux%u\n", width, height);
    return true;
}

vk::ImageView RtLighting::FrameView(const VideoCore::Image& image, vk::Format format,
                                    vk::ImageAspectFlags aspect) {
    const auto device = instance.GetDevice();
    const auto view = Check(device.createImageView({
        .image = image.GetImage(),
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {aspect, 0, 1, 0, 1},
    }));
    scheduler.DeferOperation([device, view] { device.destroyImageView(view); });
    return view;
}

void RtLighting::OnConstants(const float* data) {
    // The signature CameraMotion::OnConstants checks: far plane 3000, 1/far, render size.
    if (!enabled || frame_has_constants || data[0] != 3000.0f || data[4] < 64.0f ||
        data[5] < 64.0f) {
        return;
    }
    std::memcpy(constants.data(), data, sizeof(constants));
    frame_has_constants = true;
}

void RtLighting::OnTargets(const std::array<VideoCore::ImageId, MaxTargets>& colors,
                           VideoCore::ImageId depth_target) {
    if (!enabled) {
        return;
    }
    u32 count = 0;
    for (const auto id : colors) {
        count += id ? 1u : 0u;
    }
    // Shadow maps: square D32 depth-only passes before the G-buffer (4096x4096 for the sun
    // and moon cascades); the largest one is kept.
    if (count == 0 && depth_target && phase == Phase::Idle) {
        const auto& image = texture_cache.GetImage(depth_target);
        const u32 size = image.info.size.width;
        if (IsDepth32(image.info.pixel_format) && size == image.info.size.height && size >= 1024 &&
            (!shadow || size > texture_cache.GetImage(shadow).info.size.width)) {
            shadow = depth_target;
        }
        return;
    }
    // The G-buffer pass: 5+ color targets and a depth target (as Rasterizer::gbuffer_draw).
    if (count >= 5 && depth_target) {
        if (phase != Phase::GBuffer && logged_frames < 3) {
            std::printf("RT: frame %llu: G-buffer pass\n", Ull(frame));
        }
        gbuffer = colors;
        depth = depth_target;
        phase = Phase::GBuffer;
        return;
    }
    if (phase == Phase::Idle || !depth) {
        return;
    }
    const auto& scene_depth = texture_cache.GetImage(depth);
    const auto scene_size = [&](const VideoCore::Image& image) {
        return image.info.size.width == scene_depth.info.size.width &&
               image.info.size.height == scene_depth.info.size.height;
    };
    const auto scene_hdr = [&](VideoCore::ImageId id) {
        const auto& image = texture_cache.GetImage(id);
        return image.info.pixel_format == vk::Format::eR16G16B16A16Sfloat && scene_size(image);
    };
    // Scene color draws: the RGBA16F target drawn with the scene depth (the last one is the
    // scene color the fog is composited into). The first draw after them with another target
    // and without the scene depth starts the post-processing (Bloodborne: a copy of the
    // finished scene color, kept for the next frame's reflections).
    if (phase == Phase::Done) {
        if (depth_target == depth && colors[0]) {
            if (scene_hdr(colors[0])) {
                scene_color = colors[0];
            }
        } else if (!scene_complete && scene_color && colors[0] && colors[0] != scene_color &&
                   depth_target != depth) {
            scene_complete = true;
            OnSceneComplete();
        }
        return;
    }
    // The lighting pass: the ambient and sun/moon pass, then light volumes, into two scene-size
    // B10G11R11 targets.
    if (count == 2) {
        std::array<VideoCore::ImageId, 2> found{};
        u32 n = 0;
        for (const auto id : colors) {
            if (id) {
                found[n++] = id;
            }
        }
        const auto& a = texture_cache.GetImage(found[0]);
        const auto& b = texture_cache.GetImage(found[1]);
        if (a.info.pixel_format == vk::Format::eB10G11R11UfloatPack32 &&
            b.info.pixel_format == vk::Format::eB10G11R11UfloatPack32 && scene_size(a) &&
            scene_size(b)) {
            lighting = found;
            if (phase != Phase::Lighting) {
                phase = Phase::Lighting;
                if (logged_frames < 3) {
                    std::printf("RT: frame %llu: lighting pass\n", Ull(frame));
                }
                OnLightingStart();
            }
            return;
        }
    }
    // After the lighting pass, its diffuse target is multiplied by the albedo into an RGBA16F
    // target; the first RGBA16F target with the scene depth follows that.
    if (phase == Phase::Lighting && depth_target == depth && colors[0] && scene_hdr(colors[0])) {
        phase = Phase::Done;
        scene_color = colors[0];
        if (logged_frames < 3) {
            std::printf("RT: frame %llu: scene color pass\n", Ull(frame));
            ++logged_frames;
        }
        if (dumping) {
            std::vector<DumpItem> items;
            for (u32 i = 0; i < lighting.size(); ++i) {
                if (lighting[i]) {
                    items.push_back(Item(lighting[i], "light" + std::to_string(i)));
                }
            }
            DumpImages(items);
        }
    }
}

void RtLighting::OnFrameStart(std::array<float, 2> new_jitter) {
    if (!enabled) {
        return;
    }
    if (phase != Phase::Idle && phase != Phase::Done && logged_frames < 3) {
        std::printf("RT: frame %llu ended before the scene color pass (phase %u)\n", Ull(frame),
                    static_cast<unsigned>(phase));
        ++logged_frames;
    }
    if (dumping) {
        std::printf("RT: dump d%03u of frame %llu done%s\n", dump_index, Ull(frame),
                    scene_complete ? "" : " (the finished scene color was not found)");
        dumping = false;
        ++dump_index;
    }
    // A frame without indirect light (menus, loading, toggled off) breaks the history.
    if (!gi_ran) {
        history_valid = false;
    }
    gi_ran = false;
    scene_built = false;
    if (scene) {
        scene->OnFrameStart();
    }
    ++frame;
    phase = Phase::Idle;
    gbuffer = {};
    lighting = {};
    depth = {};
    shadow = {};
    scene_color = {};
    scene_complete = false;
    frame_has_constants = false;
    jitter = new_jitter;
    // A file system call every 30 frames, not every frame.
    if (dump_trigger && !dump_armed && frame % 30 == 0 && std::remove(dump_trigger) == 0) {
        dump_armed = true;
        std::printf("RT: dump armed\n");
    }
}

void RtLighting::OnLightingStart() {
    if (dump_armed) {
        dump_armed = false;
        dumping = true;
        std::error_code error;
        std::filesystem::create_directories(dump_dir, error);
        char path[512];
        std::snprintf(path, sizeof(path), "%s/d%03u_constants_864.bin", dump_dir.c_str(),
                      dump_index);
        if (FILE* f = std::fopen(path, "wb")) {
            std::fwrite(constants.data(), 1, sizeof(constants), f);
            std::fclose(f);
        }
        std::vector<DumpItem> items;
        for (u32 slot = 0; slot < MaxTargets; ++slot) {
            if (gbuffer[slot]) {
                items.push_back(Item(gbuffer[slot], "gbuffer" + std::to_string(slot)));
            }
        }
        items.push_back(Item(depth, "depth", vk::ImageAspectFlagBits::eDepth));
        if (HasStencil(texture_cache.GetImage(depth).info.pixel_format)) {
            items.push_back(Item(depth, "stencil", vk::ImageAspectFlagBits::eStencil));
        }
        if (shadow) {
            items.push_back(Item(shadow, "shadow", vk::ImageAspectFlagBits::eDepth));
        }
        DumpImages(items);
    }
    const bool ran = RunGi();
    if (dumping && ran) {
        DumpItem gi;
        gi.raw = vk::Image(result[result_index ^ 1].image); // RunGi flipped the index
        gi.width = (gi_width + 1) / 2;
        gi.height = (gi_height + 1) / 2;
        gi.format = vk::Format::eR16G16B16A16Sfloat;
        gi.name = "gi";
        std::vector<DumpItem> items{Item(gbuffer[4], "gbuffer4_with_gi"), gi};
        if (scene_built && validate.image) {
            DumpItem check;
            check.raw = vk::Image(validate.image);
            check.width = gi_width;
            check.height = gi_height;
            check.format = vk::Format::eR32G32Sfloat;
            check.name = "rt_validate";
            items.push_back(check);
        }
        DumpImages(items);
    }
}

bool RtLighting::RunGi() {
    static bool logged_constants = false, logged_targets = false, logged_reduced = false,
                logged_first = false;
    if (!gi_enabled || BbToggle::Disabled(ToggleGiOff)) {
        return false;
    }
    if (!frame_has_constants) {
        SkipOnce(logged_constants, "no scene constants this frame");
        return false;
    }
    if (runtime.scene_targets && runtime.scene_targets->Reduced()) {
        SkipOnce(logged_reduced, "reduced scene targets (live resolution); not supported yet");
        return false;
    }
    if (!gbuffer[0] || !gbuffer[4]) {
        SkipOnce(logged_targets, "G-buffer slots 0 and 4 not found");
        return false;
    }
    auto& depth_image = texture_cache.GetImage(depth);
    auto& normal_image = texture_cache.GetImage(gbuffer[0]);
    auto& ambient_image = texture_cache.GetImage(gbuffer[4]);
    const u32 width = depth_image.info.size.width, height = depth_image.info.size.height;
    const auto same_size = [&](const VideoCore::Image& image) {
        return image.info.size.width == width && image.info.size.height == height;
    };
    if (!IsDepth32(depth_image.info.pixel_format) ||
        normal_image.info.pixel_format != vk::Format::eR8G8B8A8Unorm ||
        ambient_image.info.pixel_format != vk::Format::eB10G11R11UfloatPack32 ||
        !same_size(normal_image) || !same_size(ambient_image) ||
        !(ambient_image.usage_flags & vk::ImageUsageFlagBits::eStorage)) {
        if (!logged_targets) {
            std::printf("RT: indirect light skipped: unexpected G-buffer (depth %s %ux%u, "
                        "normals %s %ux%u, ambient %s %ux%u); logged once\n",
                        vk::to_string(depth_image.info.pixel_format).c_str(), width, height,
                        vk::to_string(normal_image.info.pixel_format).c_str(),
                        normal_image.info.size.width, normal_image.info.size.height,
                        vk::to_string(ambient_image.info.pixel_format).c_str(),
                        ambient_image.info.size.width, ambient_image.info.size.height);
            logged_targets = true;
        }
        return false;
    }

    scheduler.EndRendering();
    if (!EnsureGiImages(width, height)) {
        gi_enabled = false;
        return false;
    }
    const auto depth_view =
        FrameView(depth_image, depth_image.info.pixel_format, vk::ImageAspectFlagBits::eDepth);
    const auto normal_view =
        FrameView(normal_image, vk::Format::eR8G8B8A8Unorm, vk::ImageAspectFlagBits::eColor);
    const auto ambient_view = FrameView(ambient_image, vk::Format::eB10G11R11UfloatPack32,
                                        vk::ImageAspectFlagBits::eColor);
    runtime.Transit(&depth_image, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&normal_image, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&ambient_image, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    runtime.FlushBarriers();
    const Camera current = CurrentCamera();
    // GPU timing: this slot's last frame is long done; its queries are read, then reset.
    const u32 timestamp_slot = gi_frame % TimestampSlots;
    const u32 timestamp_base = timestamp_slot * TimestampsPerSlot;
    if (timestamps) {
        if (timestamp_written[timestamp_slot]) {
            ReadTimestamps(timestamp_slot);
        }
        scheduler.CommandBuffer().resetQueryPool(*timestamps, timestamp_base, TimestampsPerSlot);
        timestamp_written[timestamp_slot] = true;
    }
    // Step 4: this frame's G-buffer geometry as acceleration structures.
    scene_built = scene && scene->Build(current.inv_view, current.proj,
                                        Shader::MotionVectors::positions_address,
                                        timestamps ? *timestamps : vk::QueryPool{},
                                        timestamp_base);
    if (scene_built && dumping) {
        RunValidation(depth_view, width, height);
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    if (timestamps && !scene_built) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *timestamps,
                               timestamp_base + 3);
    }
    // This module's images: last frame's trace, filter and radiance copy are done.
    RtBarrier(cmdbuf,
                  vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eTransfer,
                  vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eTransferWrite,
                  vk::PipelineStageFlagBits2::eComputeShader,
                  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    const bool has_previous = previous.proj[0] != 0.0f;

    const u32 cur = result_index, prev = result_index ^ 1;
    TracePush trace{
        .view_rot = current.view,
        .reproject = Multiply(has_previous ? previous.view : current.view, current.inv_view),
        .proj = current.proj,
        .prev_proj = has_previous ? previous.proj : current.proj,
        .size = {float(width), float(height)},
        .jitter = current.jitter,
        .prev_jitter = has_previous ? previous.jitter : current.jitter,
        .max_distance = max_distance,
        .thickness = thickness,
        .frame = gi_frame++,
        .flags = (radiance_valid && has_previous ? 1u : 0u) |
                 (history_valid && has_previous ? 2u : 0u),
        .history_weight = history_weight,
        .max_radiance = max_radiance,
        .inv_view = current.inv_view,
        .foliage_opacity = foliage_opacity,
    };
    const auto sampled = [](vk::ImageView view) {
        return vk::DescriptorImageInfo{.imageView = view, .imageLayout = vk::ImageLayout::eGeneral};
    };
    const std::array<vk::DescriptorImageInfo, 7> trace_images = {
        sampled(depth_view),         sampled(normal_view),        sampled(*radiance.view),
        sampled(*result[prev].view), sampled(*result_z[prev].view), sampled(*result[cur].view),
        sampled(*result_z[cur].view),
    };
    std::array<vk::WriteDescriptorSet, 7> trace_writes{};
    for (u32 i = 0; i < trace_writes.size(); ++i) {
        trace_writes[i] = {
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = i >= 5 ? vk::DescriptorType::eStorageImage
                                     : vk::DescriptorType::eSampledImage,
            .pImageInfo = &trace_images[i],
        };
    }
    // Traced at half resolution, applied at full resolution. With this frame's acceleration
    // structure the rays go through the geometry (HW_RAYS), else through the depth buffer.
    const u32 half_width = (width + 1) / 2, half_height = (height + 1) / 2;
    const u32 groups_x = (width + 7) / 8, groups_y = (height + 7) / 8;
    const bool hw_rays = scene_built && trace_hw_pipeline;
    if (hw_rays) {
        const vk::AccelerationStructureKHR tlas = scene->Tlas();
        const vk::WriteDescriptorSetAccelerationStructureKHR tlas_write{
            .accelerationStructureCount = 1,
            .pAccelerationStructures = &tlas,
        };
        std::array<vk::WriteDescriptorSet, 8> hw_writes{};
        std::copy(trace_writes.begin(), trace_writes.end(), hw_writes.begin());
        hw_writes[7] = {
            .pNext = &tlas_write,
            .dstBinding = 7,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eAccelerationStructureKHR,
        };
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *trace_hw_pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *trace_hw_layout, 0,
                                    hw_writes);
        cmdbuf.pushConstants(*trace_hw_layout, vk::ShaderStageFlagBits::eCompute, 0,
                             sizeof(trace), &trace);
    } else {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *trace_pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *trace_layout, 0,
                                    trace_writes);
        cmdbuf.pushConstants(*trace_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(trace),
                             &trace);
    }
    cmdbuf.dispatch((half_width + 7) / 8, (half_height + 7) / 8, 1);
    if (timestamps) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *timestamps,
                               timestamp_base + 4);
    }
    RtBarrier(cmdbuf, vk::PipelineStageFlagBits2::eComputeShader,
                  vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eComputeShader,
                  vk::AccessFlagBits2::eShaderRead);

    const ApplyPush apply{
        .proj = current.proj,
        .size = {float(width), float(height)},
        .ao_strength = ao_strength,
        .bounce_strength = bounce_strength,
        .mode = BbToggle::Disabled(ToggleBounceOnly)       ? 1u
                : BbToggle::Disabled(ToggleVisibilityOnly) ? 2u
                                                           : 0u,
    };
    const std::array<vk::DescriptorImageInfo, 5> apply_images = {
        sampled(depth_view), sampled(normal_view), sampled(*result[cur].view),
        sampled(*result_z[cur].view), sampled(ambient_view)};
    std::array<vk::WriteDescriptorSet, 5> apply_writes{};
    for (u32 i = 0; i < apply_writes.size(); ++i) {
        apply_writes[i] = {
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = i == 4 ? vk::DescriptorType::eStorageImage
                                     : vk::DescriptorType::eSampledImage,
            .pImageInfo = &apply_images[i],
        };
    }
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *apply_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *apply_layout, 0, apply_writes);
    cmdbuf.pushConstants(*apply_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(apply),
                         &apply);
    cmdbuf.dispatch(groups_x, groups_y, 1);
    if (timestamps) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, *timestamps,
                               timestamp_base + 5);
    }
    // The ambient image's next use (the lighting pass) transitions from the state Transit
    // recorded above: compute shader read/write.

    previous = current;
    history_valid = true;
    result_index ^= 1;
    gi_ran = true;
    static bool logged_hw = false;
    if (!logged_first || (hw_rays && !logged_hw)) {
        logged_first = true;
        logged_hw = logged_hw || hw_rays;
        std::printf("RT: indirect light running at %ux%u (traced at %ux%u, %s)\n", width, height,
                    half_width, half_height,
                    hw_rays ? "hardware rays through the acceleration structure"
                            : "screen-space rays through the depth buffer");
    }
    return true;
}

RtLighting::Camera RtLighting::CurrentCamera() const {
    Camera camera;
    std::memcpy(camera.view.data(), constants.data() + 8, sizeof(camera.view));
    std::memcpy(camera.inv_view.data(), constants.data() + 180, sizeof(camera.inv_view));
    camera.proj = {constants[52], constants[57], constants[62], constants[63]};
    camera.jitter = jitter;
    return camera;
}

void RtLighting::RunValidation(vk::ImageView depth_view, u32 width, u32 height) {
    if (!validate.image) {
        return;
    }
    const Camera camera = CurrentCamera();
    const ValidatePush push{
        .inv_view = camera.inv_view,
        .proj = camera.proj,
        .size = {float(width), float(height)},
        .jitter = camera.jitter,
    };
    const vk::AccelerationStructureKHR tlas = scene->Tlas();
    const vk::WriteDescriptorSetAccelerationStructureKHR tlas_write{
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &tlas,
    };
    const vk::DescriptorImageInfo depth_info{.imageView = depth_view,
                                             .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo result_info{.imageView = *validate.view,
                                              .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 3> writes = {{
        {.pNext = &tlas_write,
         .dstBinding = 0,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eAccelerationStructureKHR},
        {.dstBinding = 1,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .pImageInfo = &depth_info},
        {.dstBinding = 2,
         .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .pImageInfo = &result_info},
    }};
    const auto cmdbuf = scheduler.CommandBuffer();
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *validate_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *validate_layout, 0, writes);
    cmdbuf.pushConstants(*validate_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                         &push);
    cmdbuf.dispatch((width + 7) / 8, (height + 7) / 8, 1);
    const auto& stats = scene->LastStats();
    std::printf("RT: validation rays against %u draws (%u triangles) and %u kept off screen\n",
                stats.drawn, stats.triangles, stats.kept);
}

void RtLighting::OnGeometryDraw(const RtScene::DrawInput& draw) {
    if (scene && phase == Phase::GBuffer) {
        scene->AddDraw(draw);
    }
}

void RtLighting::OnSceneComplete() {
    if (dumping) {
        DumpImages({Item(scene_color, "scene_final")});
    }
    if (!gi_ran || !radiance.image) {
        return;
    }
    auto& color = texture_cache.GetImage(scene_color);
    if (color.info.size.width != gi_width || color.info.size.height != gi_height) {
        return;
    }
    // The finished scene color lights the next frame's ray hits.
    scheduler.EndRendering();
    runtime.Transit(&color, vk::ImageLayout::eTransferSrcOptimal,
                    vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
    runtime.FlushBarriers();
    const auto cmdbuf = scheduler.CommandBuffer();
    RtBarrier(cmdbuf, vk::PipelineStageFlagBits2::eComputeShader,
                  vk::AccessFlagBits2::eShaderRead, vk::PipelineStageFlagBits2::eTransfer,
                  vk::AccessFlagBits2::eTransferWrite);
    const vk::ImageCopy region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .extent = {gi_width, gi_height, 1},
    };
    cmdbuf.copyImage(color.GetImage(), vk::ImageLayout::eTransferSrcOptimal,
                     vk::Image(radiance.image), vk::ImageLayout::eGeneral, region);
    radiance_valid = true;
}

void RtLighting::DumpImages(const std::vector<DumpItem>& items) {
    // Native images: Transit resolves a reduced proxy into its native image first.
    scheduler.EndRendering();
    for (const auto& item : items) {
        if (!item.raw) {
            runtime.Transit(&texture_cache.GetImage(item.id), vk::ImageLayout::eGeneral,
                            vk::PipelineStageFlagBits2::eTransfer,
                            vk::AccessFlagBits2::eTransferRead);
        }
    }
    runtime.FlushBarriers();
    const auto cmdbuf = scheduler.CommandBuffer();
    // This module's images were written by compute shaders.
    RtBarrier(cmdbuf, vk::PipelineStageFlagBits2::eComputeShader,
                  vk::AccessFlagBits2::eShaderWrite, vk::PipelineStageFlagBits2::eTransfer,
                  vk::AccessFlagBits2::eTransferRead);
    const auto allocator = instance.GetAllocator();
    char path[512];
    for (const auto& item : items) {
        vk::Image source = item.raw;
        u32 width = item.width, height = item.height;
        vk::Format format = item.format;
        if (!source) {
            const auto& image = texture_cache.GetImage(item.id);
            source = image.GetImage();
            width = image.info.size.width;
            height = image.info.size.height;
            format = image.info.pixel_format;
        }
        const u32 bpp = CopyBytesPerPixel(format, item.aspect);
        if (bpp == 0) {
            std::printf("RT: dump skips %s (%s)\n", item.name.c_str(),
                        vk::to_string(format).c_str());
            continue;
        }
        const VkDeviceSize size = VkDeviceSize(width) * height * bpp;
        const VkBufferCreateInfo buffer_ci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        const VmaAllocationCreateInfo alloc_ci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        };
        VkBuffer buffer{};
        VmaAllocation allocation{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &buffer, &allocation, &info) !=
            VK_SUCCESS) {
            std::printf("RT: dump has no host memory for %s\n", item.name.c_str());
            continue;
        }
        const vk::BufferImageCopy region{
            .imageSubresource = {item.aspect, 0, 0, 1},
            .imageExtent = {width, height, 1},
        };
        cmdbuf.copyImageToBuffer(source, vk::ImageLayout::eGeneral, buffer, region);
        const std::string format_name = item.aspect == vk::ImageAspectFlagBits::eStencil ? "S8Uint"
                                        : item.aspect == vk::ImageAspectFlagBits::eDepth
                                            ? "D32Sfloat"
                                            : vk::to_string(format);
        std::snprintf(path, sizeof(path), "%s/d%03u_%s_%ux%u_%s.raw", dump_dir.c_str(),
                      dump_index, item.name.c_str(), width, height, format_name.c_str());
        scheduler.DeferPriorityOperation([allocator, buffer, allocation, info, size,
                                          file = std::string{path}] {
            vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
            if (FILE* f = std::fopen(file.c_str(), "wb")) {
                std::fwrite(info.pMappedData, 1, size, f);
                std::fclose(f);
            }
            vmaDestroyBuffer(allocator, buffer, allocation);
        });
    }
    RtBarrier(cmdbuf, vk::PipelineStageFlagBits2::eTransfer,
                  vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eHost,
                  vk::AccessFlagBits2::eHostRead);
    std::printf("RT: dump d%03u: %zu images of frame %llu\n", dump_index, items.size(),
                Ull(frame));
}

} // namespace Vulkan
