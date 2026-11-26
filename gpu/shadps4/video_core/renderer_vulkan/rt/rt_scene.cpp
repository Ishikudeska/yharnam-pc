// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/rt/rt_scene.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "video_core/host_shaders/rt_gather_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {

/// rt_gather.comp push constants.
struct GatherPush {
    std::array<float, 12> inv_view;
    std::array<float, 4> proj;
    u64 positions;
    u64 draw_list;
    u64 changed_flags;
    u32 draw_count;
    u32 triangle_count;
};
static_assert(sizeof(GatherPush) == 96);

// No eAllowUpdate: refittable BLASes were ~4x larger. Moving draws are rebuilt instead.
constexpr vk::BuildAccelerationStructureFlagsKHR BlasFlags =
    vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;

void Barrier(vk::CommandBuffer cmdbuf, vk::PipelineStageFlags2 src_stage,
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

vk::AccelerationStructureGeometryKHR TriangleGeometry(vk::DeviceAddress vertices, u32 triangles) {
    return {
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry = {.triangles = {.vertexFormat = vk::Format::eR32G32B32Sfloat,
                                   .vertexData = {.deviceAddress = vertices},
                                   .vertexStride = 12,
                                   .maxVertex = triangles * 3 - 1,
                                   .indexType = vk::IndexType::eNoneKHR}},
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };
}

u64 Mix(u64 a, u64 b) {
    u64 h = a ^ (b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2));
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    return h;
}

} // namespace

RtScene::RtScene(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    if (!instance.IsRayQuerySupported()) {
        return;
    }
    if (const char* value = std::getenv("BB_RT_MEMORY_MB"); value && value[0]) {
        const unsigned long parsed = std::strtoul(value, nullptr, 10);
        if (parsed >= 128 && parsed <= 16384) {
            memory_budget = u64(parsed) << 20;
        }
    }
    const char* keep = std::getenv("BB_RT_KEEP");
    keep_offscreen = keep && keep[0] == '1';
    keep_frames = keep_offscreen ? 600 : GraceFrames;
    evict_frames = keep_offscreen ? 1800 : 300;
    frame_draws.reserve(MaxDraws);
    enabled = CreateResources();
    std::printf("RT: acceleration structures %s (one BLAS per draw, up to %llu MiB; geometry "
                "off screen %s)\n",
                enabled ? "on" : "off: no GPU memory",
                static_cast<unsigned long long>(memory_budget >> 20),
                keep_offscreen ? "kept 10 s" : "dropped after 3 frames");
}

RtScene::~RtScene() {
    for (u32 i = 0; i < entries.size(); ++i) {
        if (entries[i].used) {
            entries[i].blas.reset();
            DestroyBuffer(entries[i].storage);
            DestroyBuffer(entries[i].blas_buffer);
        }
    }
    tlas.reset();
    DestroyBuffer(scratch);
    DestroyBuffer(tlas_buffer);
    DestroyBuffer(draw_device);
    for (u32 i = 0; i < Slots; ++i) {
        DestroyBuffer(draw_buffers[i]);
        DestroyBuffer(flag_buffers[i]);
        DestroyBuffer(instance_buffers[i]);
    }
}

bool RtScene::CreateBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool host,
                           VkDeviceSize alignment) {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    const VmaAllocationCreateInfo alloc_ci{
        .flags = host ? VMA_ALLOCATION_CREATE_MAPPED_BIT |
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                      : VmaAllocationCreateFlags{},
        .usage = host ? VMA_MEMORY_USAGE_AUTO_PREFER_HOST : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VmaAllocationInfo info{};
    const VkResult result =
        alignment ? vmaCreateBufferWithAlignment(instance.GetAllocator(), &buffer_ci, &alloc_ci,
                                                 alignment, &buffer.buffer, &buffer.allocation,
                                                 &info)
                  : vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer.buffer,
                                    &buffer.allocation, &info);
    if (result != VK_SUCCESS) {
        buffer = {};
        return false;
    }
    buffer.mapped = info.pMappedData;
    buffer.address = instance.GetDevice().getBufferAddress({.buffer = buffer.buffer});
    return true;
}

void RtScene::DestroyBuffer(Buffer& buffer) {
    if (buffer.buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), buffer.buffer, buffer.allocation);
    }
    buffer = {};
}

bool RtScene::CreateResources() {
    const auto device = instance.GetDevice();
    const auto properties = instance.GetPhysicalDevice()
                                .getProperties2<vk::PhysicalDeviceProperties2,
                                                vk::PhysicalDeviceAccelerationStructurePropertiesKHR>();
    scratch_alignment = properties.get<vk::PhysicalDeviceAccelerationStructurePropertiesKHR>()
                            .minAccelerationStructureScratchOffsetAlignment;

    const vk::AccelerationStructureGeometryKHR tlas_geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = {.instances = {.arrayOfPointers = false}},
    };
    const vk::AccelerationStructureBuildGeometryInfoKHR tlas_info{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &tlas_geometry,
    };
    const u32 max_instances = MaxInstances;
    const auto tlas_sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tlas_info, max_instances);
    // The TLAS build takes the last part of the scratch buffer, after the BLAS batches.
    bool ok = CreateBuffer(scratch, ScratchCapacity + AlignScratch(tlas_sizes.buildScratchSize),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false, scratch_alignment) &&
              CreateBuffer(tlas_buffer, tlas_sizes.accelerationStructureSize,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false);
    // The gather's threads search the draw list: in device memory, copied from the host slot.
    ok = ok && CreateBuffer(draw_device, VkDeviceSize(MaxDraws) * sizeof(GatherDraw),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            false);
    for (u32 i = 0; i < Slots && ok; ++i) {
        ok = CreateBuffer(draw_buffers[i], VkDeviceSize(MaxDraws) * sizeof(GatherDraw),
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true) &&
             CreateBuffer(flag_buffers[i], VkDeviceSize(MaxDraws) * 2 * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, true) &&
             CreateBuffer(instance_buffers[i],
                          VkDeviceSize(MaxInstances) * sizeof(VkAccelerationStructureInstanceKHR),
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                          true);
        if (ok) {
            std::memset(flag_buffers[i].mapped, 0, MaxDraws * 2 * sizeof(u32));
            vmaFlushAllocation(instance.GetAllocator(), flag_buffers[i].allocation, 0,
                               VK_WHOLE_SIZE);
        }
    }
    if (!ok) {
        return false;
    }
    tlas = Check(device.createAccelerationStructureKHRUnique({
        .buffer = tlas_buffer.buffer,
        .size = tlas_sizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
    }));

    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(GatherPush)};
    gather_layout = Check(device.createPipelineLayoutUnique({
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(RT_GATHER_COMP, device);
    gather_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *gather_layout,
            }));
    device.destroyShaderModule(module);
    return true;
}

void RtScene::OnFrameStart() {
    frame_draws.clear();
    ordinals.clear();
    ++frame;
}

void RtScene::AddDraw(const DrawInput& draw) {
    if (!enabled) {
        return;
    }
    const u32 per_instance = draw.topology == Topology::List
                                 ? draw.count / 3
                                 : (draw.count >= 3 ? draw.count - 2 : 0);
    const u64 total = u64(per_instance) * draw.instances;
    if (total == 0) {
        return;
    }
    if (frame_draws.size() >= MaxDraws || total > 16'000'000) {
        if (!overflow_logged) {
            overflow_logged = true;
            std::printf("RT: a frame has more than %u draws, or a draw with too many triangles; "
                        "the rest are left out\n", MaxDraws);
        }
        return;
    }
    // Identical draws of a frame (the same mesh at several places) are told apart by order.
    const u32 ordinal = ordinals[draw.key]++;
    frame_draws.push_back({Mix(draw.key, ordinal), draw, static_cast<u32>(total)});
}

u32 RtScene::FindOrCreate(u64 key, u32 triangles, bool alpha_tested) {
    if (const auto it = entry_of.find(key); it != entry_of.end()) {
        Entry& entry = entries[it->second];
        if (entry.triangles == triangles) {
            return it->second;
        }
        Retire(it->second); // another triangle count under the same identity: start over
    }
    const auto device = instance.GetDevice();
    const auto geometry = TriangleGeometry(0, triangles);
    const vk::AccelerationStructureBuildGeometryInfoKHR info{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = BlasFlags,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1,
        .pGeometries = &geometry,
    };
    const auto sizes = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, info, triangles);
    const u64 bytes = u64(triangles) * 36 + sizes.accelerationStructureSize;
    if (entry_bytes + bytes > memory_budget || DeviceMemoryTight(bytes)) {
        Evict();
        if (entry_bytes + bytes > memory_budget || DeviceMemoryTight(bytes)) {
            if (!memory_logged) {
                memory_logged = true;
                std::printf("RT: no memory for more acceleration structures (%llu of %llu MiB, "
                            "BB_RT_MEMORY_MB; device %llu of %llu MiB); new draws are left out\n",
                            static_cast<unsigned long long>(entry_bytes >> 20),
                            static_cast<unsigned long long>(memory_budget >> 20),
                            static_cast<unsigned long long>(stats.device_used >> 20),
                            static_cast<unsigned long long>(stats.device_budget >> 20));
            }
            return ~0u;
        }
    }
    u32 index;
    if (!free_entries.empty()) {
        index = free_entries.back();
        free_entries.pop_back();
    } else {
        index = static_cast<u32>(entries.size());
        entries.emplace_back();
    }
    Entry& entry = entries[index];
    entry = Entry{};
    if (!CreateBuffer(entry.storage, VkDeviceSize(triangles) * 36,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                      false) ||
        !CreateBuffer(entry.blas_buffer, sizes.accelerationStructureSize,
                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, false)) {
        DestroyBuffer(entry.storage);
        DestroyBuffer(entry.blas_buffer);
        free_entries.push_back(index);
        return ~0u;
    }
    entry.blas = Check(device.createAccelerationStructureKHRUnique({
        .buffer = entry.blas_buffer.buffer,
        .size = sizes.accelerationStructureSize,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
    }));
    entry.blas_address =
        device.getAccelerationStructureAddressKHR({.accelerationStructure = *entry.blas});
    entry.used = true;
    entry.key = key;
    entry.triangles = triangles;
    entry.alpha_tested = alpha_tested;
    entry.build_scratch = sizes.buildScratchSize;
    entry.update_scratch = sizes.updateScratchSize;
    entry.bytes = bytes;
    entry.last_changed = frame; // unknown motion: refit until the flags say it stands still
    entry_bytes += bytes;
    entry_of[key] = index;
    return index;
}

void RtScene::Retire(u32 index) {
    Entry& entry = entries[index];
    if (!entry.used) {
        return;
    }
    entry_of.erase(entry.key);
    entry_bytes -= entry.bytes;
    // The GPU may still use it: destroyed after the work recorded so far.
    scheduler.DeferOperation([allocator = instance.GetAllocator(), blas = std::move(entry.blas),
                              storage = entry.storage, blas_buffer = entry.blas_buffer]() mutable {
        blas.reset();
        if (storage.buffer) {
            vmaDestroyBuffer(allocator, storage.buffer, storage.allocation);
        }
        if (blas_buffer.buffer) {
            vmaDestroyBuffer(allocator, blas_buffer.buffer, blas_buffer.allocation);
        }
    });
    entry = Entry{};
    free_entries.push_back(index);
}

void RtScene::Evict() {
    // Not drawn for a long time, then the ones seen longest ago while over the budget; never
    // the ones of this frame.
    std::vector<u32> candidates;
    for (u32 i = 0; i < entries.size(); ++i) {
        const Entry& entry = entries[i];
        if (!entry.used || entry.last_seen == frame) {
            continue;
        }
        if (frame - entry.last_seen > evict_frames) {
            Retire(i);
        } else {
            candidates.push_back(i);
        }
    }
    if (entry_bytes <= memory_budget * 9 / 10 && !DeviceMemoryTight(0)) {
        return;
    }
    std::sort(candidates.begin(), candidates.end(), [&](u32 a, u32 b) {
        return entries[a].last_seen < entries[b].last_seen;
    });
    // Down to 80% of the budget, or half the cache when device memory is tight.
    const u64 target = DeviceMemoryTight(0) ? entry_bytes / 2 : memory_budget * 8 / 10;
    for (const u32 i : candidates) {
        if (entry_bytes <= target) {
            break;
        }
        Retire(i);
    }
}

bool RtScene::DeviceMemoryTight(u64 extra) {
    const VkPhysicalDeviceMemoryProperties* properties = nullptr;
    vmaGetMemoryProperties(instance.GetAllocator(), &properties);
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(instance.GetAllocator(), budgets.data());
    bool tight = false;
    stats.device_used = stats.device_budget = 0;
    for (u32 heap = 0; heap < properties->memoryHeapCount; ++heap) {
        if (!(properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
            continue;
        }
        const VmaBudget& budget = budgets[heap];
        if (budget.budget > stats.device_budget) {
            stats.device_used = budget.usage;
            stats.device_budget = budget.budget;
        }
        tight = tight || budget.usage + extra > budget.budget / 100 * 85;
    }
    return tight;
}

bool RtScene::Build(const std::array<float, 12>& inv_view, const std::array<float, 4>& proj,
                    u64 positions, vk::QueryPool pool, u32 timestamps) {
    if (!enabled || frame_draws.empty() || positions == 0) {
        return false;
    }
    // This frame's host buffers, which the GPU is done with; their flags say which draws moved
    // in the frame that used them last.
    slot = (slot + 1) % Slots;
    if (slot_ticks[slot]) {
        scheduler.Wait(slot_ticks[slot]);
    }
    {
        auto* flags = static_cast<u32*>(flag_buffers[slot].mapped);
        vmaInvalidateAllocation(instance.GetAllocator(), flag_buffers[slot].allocation, 0,
                                VK_WHOLE_SIZE);
        const auto& owners = slot_entries[slot];
        for (u32 i = 0; i < owners.size(); ++i) {
            if (flags[i * 2] && owners[i] < entries.size() && entries[owners[i]].used) {
                Entry& entry = entries[owners[i]];
                entry.last_changed = std::max(entry.last_changed, slot_frames[slot]);
                // A triangle became valid or invalid, which a refit cannot express. (A first
                // build compares against empty storage: not a reason to rebuild.)
                if (flags[i * 2 + 1] && entry.built && entry.built_frame < slot_frames[slot]) {
                    entry.built = false;
                }
            }
        }
        std::memset(flags, 0, owners.size() * 2 * sizeof(u32));
        vmaFlushAllocation(instance.GetAllocator(), flag_buffers[slot].allocation, 0,
                           VK_WHOLE_SIZE);
    }
    slot_frames[slot] = frame;
    slot_ticks[slot] = scheduler.CurrentTick();
    auto& owners = slot_entries[slot];
    owners.clear();

    // Entries of this frame's draws, and what each needs.
    enum class Work : u8 { Skip, Rebuild, Build };
    struct Job {
        u32 entry;
        Work work;
    };
    std::vector<Job> jobs;
    jobs.reserve(frame_draws.size());
    auto* records = static_cast<GatherDraw*>(draw_buffers[slot].mapped);
    stats = {};
    u32 triangles = 0;
    for (const auto& draw : frame_draws) {
        const u32 index = FindOrCreate(draw.key, draw.triangles, draw.input.alpha_tested);
        if (index == ~0u) {
            continue;
        }
        Entry& entry = entries[index];
        entry.last_seen = frame;
        const Work work = !entry.built                            ? Work::Build
                          : frame - entry.last_changed < StaticFrames ? Work::Rebuild
                                                                     : Work::Skip;
        const auto& in = draw.input;
        records[owners.size()] = {
            .index_address = in.index_address,
            .storage = entry.storage.address,
            .index_size = in.index_size,
            .topology = static_cast<u32>(in.topology),
            .triangles = draw.triangles / std::max(in.instances, 1u),
            .instances = in.instances,
            .first_triangle = triangles,
            .store_base = in.store_base,
            .vertices = in.vertices,
            .first_vertex = in.first_vertex,
            .base_vertex = in.base_vertex,
            .restart = in.index_size == 2 ? 0xFFFFu : 0xFFFFFFFFu,
            .flags = (in.restart_enabled ? 1u : 0u) | (work != Work::Skip ? 2u : 0u) |
                     (in.alpha_tested ? 4u : 0u),
            .draw_index = static_cast<u32>(owners.size()),
        };
        owners.push_back(index);
        triangles += draw.triangles;
        jobs.push_back({index, work});
        stats.built += work == Work::Build;
        stats.rebuilt += work == Work::Rebuild;
        stats.skipped += work == Work::Skip;
    }
    if (owners.empty()) {
        return false;
    }
    vmaFlushAllocation(instance.GetAllocator(), draw_buffers[slot].allocation, 0,
                       owners.size() * sizeof(GatherDraw));

    const auto cmdbuf = scheduler.CommandBuffer();
    // The vertex shaders' position stores, index uploads and the previous frame's traces.
    Barrier(cmdbuf, vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    Barrier(cmdbuf, vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR | vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eComputeShader |
                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
    if (pool) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, timestamps + 0);
    }
    // The draw list into device memory (the last frame's gather is done reading it).
    Barrier(cmdbuf, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    const vk::BufferCopy draw_copy{.size = owners.size() * sizeof(GatherDraw)};
    cmdbuf.copyBuffer(draw_buffers[slot].buffer, draw_device.buffer, draw_copy);
    Barrier(cmdbuf, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead);

    // Gather: compare with and update the entries' storage, flag moving draws.
    const GatherPush push{
        .inv_view = inv_view,
        .proj = proj,
        .positions = positions,
        .draw_list = draw_device.address,
        .changed_flags = flag_buffers[slot].address,
        .draw_count = static_cast<u32>(owners.size()),
        .triangle_count = triangles,
    };
    const u32 groups = (triangles + 63) / 64;
    const u32 groups_x = std::min(groups, 65535u);
    const u32 groups_y = (groups + groups_x - 1) / groups_x;
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *gather_pipeline);
    cmdbuf.pushConstants(*gather_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
    cmdbuf.dispatch(groups_x, groups_y, 1);
    Barrier(cmdbuf, vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eShaderRead);
    if (pool) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eComputeShader, pool, timestamps + 1);
    }

    // BLAS builds (new or moving entries), in batches that fit the scratch buffer.
    std::vector<vk::AccelerationStructureGeometryKHR> geometries;
    std::vector<vk::AccelerationStructureBuildGeometryInfoKHR> infos;
    std::vector<vk::AccelerationStructureBuildRangeInfoKHR> ranges;
    std::vector<const vk::AccelerationStructureBuildRangeInfoKHR*> range_pointers;
    geometries.reserve(jobs.size());
    infos.reserve(jobs.size());
    ranges.reserve(jobs.size());
    range_pointers.reserve(jobs.size());
    VkDeviceSize scratch_offset = 0;
    const auto flush = [&] {
        if (infos.empty()) {
            return;
        }
        cmdbuf.buildAccelerationStructuresKHR(static_cast<u32>(infos.size()), infos.data(),
                                              range_pointers.data());
        Barrier(cmdbuf, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
                vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
                vk::AccessFlagBits2::eAccelerationStructureReadKHR |
                    vk::AccessFlagBits2::eAccelerationStructureWriteKHR);
        geometries.clear();
        infos.clear();
        ranges.clear();
        range_pointers.clear();
        scratch_offset = 0;
    };
    for (const auto& job : jobs) {
        if (job.work == Work::Skip) {
            continue;
        }
        Entry& entry = entries[job.entry];
        const VkDeviceSize need = AlignScratch(entry.build_scratch);
        if (need > ScratchCapacity) {
            continue;
        }
        if (scratch_offset + need > ScratchCapacity) {
            flush();
        }
        geometries.push_back(TriangleGeometry(entry.storage.address, entry.triangles));
        infos.push_back({
            .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
            .flags = BlasFlags,
            .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
            .dstAccelerationStructure = *entry.blas,
            .geometryCount = 1,
            .pGeometries = &geometries.back(),
            .scratchData = {.deviceAddress = scratch.address + scratch_offset},
        });
        ranges.push_back({.primitiveCount = entry.triangles});
        range_pointers.push_back(&ranges.back());
        scratch_offset += need;
        entry.built_frame = frame;
        entry.built = true;
    }
    flush();
    if (pool) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, pool,
                               timestamps + 2);
    }

    // TLAS: this frame's entries, and static entries seen recently (geometry off screen).
    auto* instances = static_cast<VkAccelerationStructureInstanceKHR*>(instance_buffers[slot].mapped);
    u32 count = 0;
    const auto add_instance = [&](u32 index) {
        const Entry& entry = entries[index];
        VkAccelerationStructureInstanceKHR& record = instances[count++];
        record = {};
        record.transform.matrix[0][0] = 1.0f;
        record.transform.matrix[1][1] = 1.0f;
        record.transform.matrix[2][2] = 1.0f;
        record.instanceCustomIndex = index & 0xFFFFFF;
        record.mask = entry.alpha_tested ? 0x02 : 0x01;
        record.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        record.accelerationStructureReference = entry.blas_address;
    };
    for (const auto& job : jobs) {
        if (count < MaxInstances && entries[job.entry].built) {
            add_instance(job.entry);
        }
    }
    stats.drawn = count;
    // Entries not drawn this frame: for a few frames always (draws object motion skipped,
    // culling flicker), for 10 s with BB_RT_KEEP=1 if they stood still.
    for (u32 i = 0; i < entries.size() && count < MaxInstances; ++i) {
        const Entry& entry = entries[i];
        if (!entry.used || !entry.built || entry.last_seen == frame ||
            frame - entry.last_seen > keep_frames) {
            continue;
        }
        if (frame - entry.last_seen > GraceFrames &&
            frame - entry.last_changed < KeepStaticFrames) {
            continue;
        }
        add_instance(i);
        ++stats.kept;
    }
    vmaFlushAllocation(instance.GetAllocator(), instance_buffers[slot].allocation, 0,
                       VkDeviceSize(count) * sizeof(VkAccelerationStructureInstanceKHR));
    const vk::AccelerationStructureGeometryKHR tlas_geometry{
        .geometryType = vk::GeometryTypeKHR::eInstances,
        .geometry = {.instances = {.arrayOfPointers = false,
                                   .data = {.deviceAddress = instance_buffers[slot].address}}},
    };
    const vk::AccelerationStructureBuildGeometryInfoKHR tlas_info{
        .type = vk::AccelerationStructureTypeKHR::eTopLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .dstAccelerationStructure = *tlas,
        .geometryCount = 1,
        .pGeometries = &tlas_geometry,
        .scratchData = {.deviceAddress = scratch.address + ScratchCapacity},
    };
    const vk::AccelerationStructureBuildRangeInfoKHR tlas_range{.primitiveCount = count};
    const vk::AccelerationStructureBuildRangeInfoKHR* tlas_ranges = &tlas_range;
    cmdbuf.buildAccelerationStructuresKHR(1, &tlas_info, &tlas_ranges);
    Barrier(cmdbuf, vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR,
            vk::AccessFlagBits2::eAccelerationStructureWriteKHR,
            vk::PipelineStageFlagBits2::eComputeShader,
            vk::AccessFlagBits2::eAccelerationStructureReadKHR);
    if (pool) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAccelerationStructureBuildKHR, pool,
                               timestamps + 3);
    }

    stats.triangles = triangles;
    stats.entries = static_cast<u32>(entry_of.size());
    stats.bytes = entry_bytes;
    if (frame % 60 == 0) {
        Evict();
    }
    if (builds++ % 600 == 0) {
        DeviceMemoryTight(0);
        std::printf("RT: scene: %u draws (%u triangles): %u built, %u rebuilt, %u unchanged; "
                    "%u kept off screen; %u entries, %.0f MiB; device memory %llu of %llu MiB\n",
                    stats.drawn, triangles, stats.built, stats.rebuilt, stats.skipped, stats.kept,
                    stats.entries, double(entry_bytes) / 1048576.0,
                    static_cast<unsigned long long>(stats.device_used >> 20),
                    static_cast<unsigned long long>(stats.device_budget >> 20));
    }
    // The TLAS (possibly empty) and the timestamps are recorded either way.
    return true;
}

} // namespace Vulkan
