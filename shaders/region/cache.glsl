#include "../common/contract.glsl"

// The Radiance cache's per-face state (ADR 0019): one slab per resident
// Region, one entry per exposed voxel face. The resolve pass is the only
// state writer (cache_store); traced-ray hits are the only readers
// (cache_fetch) and feed the accumulator half (cache_deposit). A false from
// cache_fetch is ADR 0019's uncovered-region gate: the caller uses the
// direct BSDF estimate. Reads land only at traced-ray hit positions — never
// stamped into primary-surface radiance.

VKO_DECLARE_STORAGE_BUFFER(cache_table, CacheTable{
    uint64_t[REGION_TABLE_ENTRIES] bdas;
    uint[REGION_TABLE_ENTRIES] entry_counts;
    uint[REGION_TABLE_ENTRIES] bitmap_prefix;
})

VKO_DECLARE_STORAGE_BUFFER(cache_state, CacheState{
    uint64_t stats_bda;
    uint frame_index;
    uint event_frames;
    uint stats_enabled;
})

#define cache_table vko_buffer(cache_table, cache_table_buffer_id)
#define cache_state vko_buffer(cache_state, cache_state_buffer_id)

layout(buffer_reference, std430) buffer CacheStats {
    uint lookups[REGION_TABLE_ENTRIES];
    uint fallbacks[REGION_TABLE_ENTRIES];
    uint touched[REGION_TABLE_ENTRIES];
    uint deposits[REGION_TABLE_ENTRIES];
    uint landed[REGION_TABLE_ENTRIES];
};

layout(buffer_reference, std430) buffer CacheSlab {
    uint words[];
};

uint cache_voxel_slot(ivec3 cell) {
    return uint(cell.x & 7) + VOXEL_STRIDE_Y * uint(cell.y & 7) + VOXEL_STRIDE_Z * uint(cell.z & 7);
}

uint cache_slot(uint mc_block, uint voxel_idx, uint face) {
    return mc_block * CACHE_ENTRIES_PER_MC + voxel_idx * 6u + face;
}

uint cache_entry_base(uint mc_block, uint voxel_idx, uint face) {
    return cache_slot(mc_block, voxel_idx, face) * CACHE_ENTRY_STRIDE;
}

// The uncovered-region gate (ADR 0019): a face is covered when its Region
// owns a slab (hard budget cap in the allocator; bda 0 falls back), the
// entry exists (nonzero stamp), and its age is inside CACHE_STALE_T.
bool cache_fresh(uint region_id, uint mc_block, uint voxel_idx, uint face) {
    uint64_t bda = cache_table.bdas[region_id];

    if (bda == 0ul) {
        return false;
    }

    uint stamp = CacheSlab(bda).words[cache_entry_base(mc_block, voxel_idx, face) + 2u];
    return stamp != 0u && cache_state.frame_index - stamp < CACHE_STALE_T;
}

bool cache_fetch(uint region_id, uint mc_block, uint voxel_idx, uint face, out vec3 irradiance) {
    irradiance = vec3(0.0);

    if (!cache_fresh(region_id, mc_block, voxel_idx, face)) {
        if (cache_state.stats_enabled != 0u) {
            atomicAdd(CacheStats(cache_state.stats_bda).fallbacks[region_id], 1u);
        }

        return false;
    }

    if (cache_state.stats_enabled != 0u) {
        CacheStats stats = CacheStats(cache_state.stats_bda);
        atomicAdd(stats.lookups[region_id], 1u);
    }

    uint base = cache_entry_base(mc_block, voxel_idx, face);
    vec2 rg = unpackHalf2x16(CacheSlab(cache_table.bdas[region_id]).words[base]);
    vec2 bz = unpackHalf2x16(CacheSlab(cache_table.bdas[region_id]).words[base + 1u]);
    irradiance = pow(vec3(rg.x, rg.y, bz.x), vec3(CACHE_IRRADIANCE_GAMMA));

    return true;
}

// The accumulate half of 02's accumulate→resolve: a deposit adds its
// fixed-point radiance to the entry's accumulator and marks the entry
// touched for the resolve's bitmap scan. `count` tags the deposit that
// opens its chain, so the resolve's mean divides by chains, not by the
// partial deposits a multi-hop chain scatters along its path.
void cache_deposit(uint region_id, uint mc_block, uint voxel_idx, uint face, bool count, vec3 radiance) {
    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).deposits[region_id], 1u);
    }

    uint64_t bda = cache_table.bdas[region_id];

    if (bda == 0ul) {
        return;
    }
    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).landed[region_id], 1u);
    }

    uint slot = cache_slot(mc_block, voxel_idx, face);
    uint base = slot * CACHE_ENTRY_STRIDE + CACHE_ACC_OFFSET;
    CacheSlab slab = CacheSlab(bda);
    vec3 ticks = clamp(radiance, vec3(0.0), vec3(CACHE_ACC_TICK_CAP / CACHE_ACC_SCALE)) * CACHE_ACC_SCALE;

    atomicAdd(slab.words[base + 0u], uint(ticks.r));
    atomicAdd(slab.words[base + 1u], uint(ticks.g));
    atomicAdd(slab.words[base + 2u], uint(ticks.b));

    if (count) {
        atomicAdd(slab.words[base + 3u], 1u);
    }

    uint bitmap_base = cache_table.entry_counts[region_id] * CACHE_ENTRY_STRIDE;
    atomicOr(slab.words[bitmap_base + (slot >> 5u)], 1u << (slot & 31u));
}

// Resolve-only store (02): gamma-encoded, the stamp renews coverage. Plain
// writes — the resolve is the entry's single writer per frame.
void cache_store(uint region_id, uint mc_block, uint voxel_idx, uint face, vec3 irradiance) {
    uint64_t bda = cache_table.bdas[region_id];

    if (bda == 0ul) {
        return;
    }

    vec3 encoded = pow(max(irradiance, vec3(0.0)), vec3(1.0 / CACHE_IRRADIANCE_GAMMA));
    uint base = cache_entry_base(mc_block, voxel_idx, face);
    CacheSlab slab = CacheSlab(bda);
    slab.words[base] = packHalf2x16(encoded.rg);
    slab.words[base + 1u] = packHalf2x16(vec2(encoded.b, 0.0));
    slab.words[base + 2u] = cache_state.frame_index;
}

// The resolve pass notes each blended entry once (unique touched faces).
void cache_note_touched(uint region_id) {
    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).touched[region_id], 1u);
    }
}

// The transient gate (02): the Scene uniform-compare's reduction event
// lowers the resolve's old-weight for CACHE_EVENT_FRAMES frames.
float cache_old_weight(float base) {
    return base * (cache_state.event_frames > 0u ? CACHE_EVENT_REDUCTION : 1.0);
}
