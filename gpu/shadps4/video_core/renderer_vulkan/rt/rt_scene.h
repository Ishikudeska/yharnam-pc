// SPDX-License-Identifier: GPL-2.0-or-later
// bbport path tracing, step 4 (docs/path_tracing.md): the scene as acceleration structures.
//
// G-buffer draws whose vertex shaders stored their clip positions (object motion) are recorded
// with AddDraw. Each draw identity (its description and its place among identical draws of the
// frame) has an entry: its own world-space vertex storage and bottom-level structure (BLAS).
// Build gathers the frame's triangles in world space (rt_gather.comp) and compares them with
// each entry's storage: an entry is built once, rebuilt while it moves (the GPU flags moving
// draws, the CPU reads the flags a few frames later), and left alone while it stands still.
// BLASes are plain fast builds: refittable ones took ~4x the memory (2026-11-25: 1.3 GB in the
// Hunter's Dream, the GPU ran out of memory and the game fell to 5 FPS).
// The top-level structure (TLAS) holds the entries drawn this frame or in the last 3 frames
// (draws object motion skipped, culling flicker); BB_RT_KEEP=1 keeps static entries for 10
// seconds, so rays also find geometry behind the camera (overlapping copies slow the rays).
// Alpha-tested draws (leaves, grass) get instance mask 2, the rest mask 1.
//
// Needs Instance::IsRayQuerySupported() (BB_RT=1 and BB_RT_SCENE=1) and object motion on every
// G-buffer draw (BB_OBJECT_MOTION_ALL=1). BB_RT_MEMORY_MB (default 768) bounds the entries'
// memory, and no entry is created while device memory is over 85% of its budget.

#pragma once

#include <array>
#include <unordered_map>
#include <vector>

#include "common/types.h"
// vk_common.h configures the Vulkan headers (beta extensions, no prototypes): before VMA's.
#include "video_core/renderer_vulkan/vk_common.h"
#include <vk_mem_alloc.h>

namespace Vulkan {

class Instance;
class Scheduler;

class RtScene {
public:
    enum class Topology : u32 { List = 0, Strip = 1 };

    /// A G-buffer draw that stored its clip positions this frame.
    struct DrawInput {
        u64 key{};           ///< identity of the draw (shader, vertex streams, indices, range)
        u64 index_address{}; ///< index data on the GPU, 0 for draws without indices
        u32 index_size{};    ///< 2 or 4
        Topology topology{};
        u32 count{};         ///< index (or vertex) count
        u32 instances{};
        u32 store_base{};    ///< first stored clip position
        u32 vertices{};      ///< stored positions per instance
        u32 first_vertex{};  ///< vertex index of the first stored position
        s32 base_vertex{};
        bool restart_enabled{};
        bool alpha_tested{}; ///< the pixel shader discards (cut-out leaves, grass)
    };

    struct Stats {
        u32 drawn = 0, kept = 0, built = 0, rebuilt = 0, skipped = 0, entries = 0;
        u32 triangles = 0;
        u64 bytes = 0;
        u64 device_used = 0, device_budget = 0; ///< device-local heap, whole process
    };

    RtScene(const Instance& instance, Scheduler& scheduler);
    ~RtScene();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    void OnFrameStart();
    void AddDraw(const DrawInput& draw);

    /// After the G-buffer pass (no rendering active): gathers this frame's triangles with the
    /// camera (`inv_view` 3x4 rows, `proj` x/y scale, z scale/offset), builds or refits the
    /// entries that need it and builds the TLAS. `positions` is the device address of the
    /// stored clip positions. Writes timestamps `timestamps` + 0..3 (start, gathered, BLAS
    /// done, TLAS done) when `pool` is set. False when there is nothing to trace.
    bool Build(const std::array<float, 12>& inv_view, const std::array<float, 4>& proj,
               u64 positions, vk::QueryPool pool, u32 timestamps);

    [[nodiscard]] vk::AccelerationStructureKHR Tlas() const noexcept {
        return *tlas;
    }
    [[nodiscard]] const Stats& LastStats() const noexcept {
        return stats;
    }

private:
    /// One draw as rt_gather.comp reads it.
    struct GatherDraw {
        u64 index_address{};
        u64 storage{};
        u32 index_size{};
        u32 topology{};
        u32 triangles{};     ///< per instance
        u32 instances{};
        u32 first_triangle{};
        u32 store_base{};
        u32 vertices{};
        u32 first_vertex{};
        s32 base_vertex{};
        u32 restart{};
        u32 flags{};         ///< 1: primitive restart, 2: write the storage
        u32 draw_index{};
    };
    static_assert(sizeof(GatherDraw) == 64);

    struct Buffer {
        VkBuffer buffer{};
        VmaAllocation allocation{};
        vk::DeviceAddress address{};
        void* mapped{};
    };

    struct Entry {
        bool used = false;
        u64 key = 0;
        u32 triangles = 0;   ///< all instances
        bool alpha_tested = false;
        bool built = false;
        Buffer storage, blas_buffer;
        vk::UniqueAccelerationStructureKHR blas;
        vk::DeviceAddress blas_address = 0;
        VkDeviceSize build_scratch = 0, update_scratch = 0;
        u64 bytes = 0;
        u64 last_seen = 0;    ///< frame it was last drawn
        u64 last_changed = 0; ///< frame of the last known movement
        u64 built_frame = 0;  ///< frame of the last full build
    };

    struct FrameDraw {
        u64 key;
        DrawInput input;
        u32 triangles; ///< all instances
    };

    static constexpr u32 Slots = 4;
    static constexpr u32 MaxDraws = 32768;
    static constexpr u32 MaxInstances = 16384;
    static constexpr VkDeviceSize ScratchCapacity = 128ull << 20;
    static constexpr u64 StaticFrames = 8;     ///< unmoved this long: no more rebuilds
    static constexpr u64 KeepStaticFrames = 60; ///< unmoved this long: kept when off screen
    static constexpr u64 GraceFrames = 3;      ///< kept this long after last drawn (always)

    bool CreateResources();
    bool CreateBuffer(Buffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage, bool host,
                      VkDeviceSize alignment = 0);
    void DestroyBuffer(Buffer& buffer);
    /// The entry of `key`, created with GPU memory for `triangles` triangles; ~0u on failure.
    u32 FindOrCreate(u64 key, u32 triangles, bool alpha_tested);
    void Retire(u32 index);
    void Evict();
    /// Device-local memory (the whole process, VK_EXT_memory_budget through VMA) would pass
    /// 85% of its budget with `extra` more bytes. Fills the stats' device fields.
    bool DeviceMemoryTight(u64 extra);
    VkDeviceSize AlignScratch(VkDeviceSize size) const {
        return (size + scratch_alignment - 1) / scratch_alignment * scratch_alignment;
    }

    const Instance& instance;
    Scheduler& scheduler;
    bool enabled = false;
    bool keep_offscreen = false;
    u64 keep_frames = GraceFrames;  ///< in the TLAS this long after last drawn
    u64 evict_frames = 300;         ///< destroyed this long after last drawn
    u64 memory_budget = 768ull << 20;
    VkDeviceSize scratch_alignment = 256;
    u64 frame = 0;

    std::vector<FrameDraw> frame_draws;
    std::unordered_map<u64, u32> ordinals; ///< identical draws of this frame so far
    std::vector<Entry> entries;
    std::vector<u32> free_entries;
    std::unordered_map<u64, u32> entry_of;
    u64 entry_bytes = 0;
    bool overflow_logged = false, memory_logged = false;

    Buffer scratch, tlas_buffer, draw_device;
    vk::UniqueAccelerationStructureKHR tlas;
    std::array<Buffer, Slots> draw_buffers{}, flag_buffers{}, instance_buffers{};
    std::array<u64, Slots> slot_ticks{};
    std::array<u64, Slots> slot_frames{};
    std::array<std::vector<u32>, Slots> slot_entries; ///< entry of each flag of the slot
    u32 slot = 0;
    vk::UniquePipelineLayout gather_layout;
    vk::UniquePipeline gather_pipeline;
    Stats stats;
    u64 builds = 0;
};

} // namespace Vulkan
