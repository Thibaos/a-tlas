#include "../common/contract.glsl"

#extension GL_EXT_shader_atomic_int64 : require

// The Radiance cache's fixed sparse table (06): SHaRC's three buffers — 64-bit
// keys, 16 B accumulators, 16 B resolved records — one entry per exposed
// voxel face, keyed by the DDA's exact hit. The resolve pass is the only
// state writer (cache_store); traced-ray hits are the only readers
// (cache_fetch) and feed the accumulator half (cache_deposit). A false from
// cache_fetch is ADR 0019's uncovered gate: the caller uses the direct BSDF
// estimate. Reads land only at traced-ray hit positions — never stamped
// into primary-surface radiance.

VKO_DECLARE_STORAGE_BUFFER(cache_state, CacheState{
    uint64_t stats_bda;
    uint64_t keys_bda;
    uint64_t accum_bda;
    uint64_t resolved_bda;
    uint64_t dirty_bda;
    uint frame_index;
    uint event_frames;
    uint stats_enabled;
})

#define cache_state vko_buffer(cache_state, cache_state_buffer_id)

layout(buffer_reference, std430) buffer CacheStats {
    uint lookups[REGION_TABLE_ENTRIES];
    uint fallbacks[REGION_TABLE_ENTRIES];
    uint touched[REGION_TABLE_ENTRIES];
    uint deposits[REGION_TABLE_ENTRIES];
    uint landed[REGION_TABLE_ENTRIES];
    uint live[REGION_TABLE_ENTRIES];
    uint young[REGION_TABLE_ENTRIES];
};

layout(buffer_reference, std430) buffer CacheKeys {
    uint64_t keys[];
};

layout(buffer_reference, std430) buffer CacheAccum {
    uvec4 ticks[];
};

layout(buffer_reference, std430) buffer CacheDirty {
    uint words[CACHE_DIRTY_WORDS];
};
layout(buffer_reference, std430) buffer CacheResolved {
    uvec4 recs[];
};

// SHaRC's hash grid: the key's bucket is CACHE_BUCKET_SIZE contiguous slots
// from a Jenkins-mixed base slot, never wrapping the table's end. Keys are
// tagged with bit 63 because key 0 (origin Region, first block, -X face) is
// a real face; 0 stays free as the empty marker.
#define CACHE_BUCKET_SIZE 16u
#define CACHE_PROBE_EMPTY_LIMIT 2u
#define CACHE_KEY_TAG (1UL << 63)
uint cache_voxel_slot(ivec3 cell) {
    return uint(cell.x & 7) + VOXEL_STRIDE_Y * uint(cell.y & 7) + VOXEL_STRIDE_Z * uint(cell.z & 7);
}

uint64_t cache_key(uint region_id, uint mc_block, uint voxel_idx, uint face) {
    uint64_t key = uint64_t(region_id) | (uint64_t(mc_block) << 12)
        | (uint64_t(voxel_idx) << 27) | (uint64_t(face) << 36);

    return key | CACHE_KEY_TAG;
}

// http://burtleburtle.net/bob/hash/integer.html (HashGridCommon.h)
uint cache_jenkins32(uint a) {
    a = (a + 0x7ed55d16u) + (a << 12);
    a = (a ^ 0xc761c23cu) ^ (a >> 19);
    a = (a + 0x165667b1u) + (a << 5);
    a = (a + 0xfd7046c5u) + (a << 3);
    a = (a ^ 0xb55a4f09u) ^ (a >> 16);

    return a;
}

uint cache_base_slot(uint64_t key) {
    uint hash = cache_jenkins32(uint(key)) ^ cache_jenkins32(uint(key >> 32));

    return hash % (CACHE_TABLE_ENTRIES - CACHE_BUCKET_SIZE + 1u);
}

// SHaRC's Find: a read-only scan of the key's bucket, stopping after the
// empty-slot limit. No atomics.
bool cache_find(uint64_t key, uint base, out uint slot) {
    CacheKeys keys = CacheKeys(cache_state.keys_bda);
    uint empties = 0u;

    for (uint i = 0u; i < CACHE_BUCKET_SIZE; ++i) {
        uint64_t stored = keys.keys[base + i];

        if (stored == 0ul) {
            if (empties > CACHE_PROBE_EMPTY_LIMIT) {
                break;
            }

            ++empties;
        } else if (stored == key) {
            slot = base + i;

            return true;
        }
    }

    return false;
}

// The uncovered-region gate (ADR 0019): a face is covered when its entry
// exists in the table, has been resolved (nonzero stamp), and its age is
// inside CACHE_STALE_T. `immature` reports a young chain history — the
// caller renders such faces as uncovered (full-rate retrace, traced
// radiance) until the resolve has sampled enough chains (CACHE_MATURE_T),
// so coverage engages only at a converged mean.
bool cache_fetch(uint region_id, uint mc_block, uint voxel_idx, uint face,
        out vec3 irradiance, out bool immature) {
    irradiance = vec3(0.0);
    immature = false;

    uint64_t key = cache_key(region_id, mc_block, voxel_idx, face);
    uint slot;

    if (!cache_find(key, cache_base_slot(key), slot)) {
        if (cache_state.stats_enabled != 0u) {
            atomicAdd(CacheStats(cache_state.stats_bda).fallbacks[region_id], 1u);
        }

        return false;
    }

    uvec4 rec = CacheResolved(cache_state.resolved_bda).recs[slot];
    immature = rec.w < CACHE_MATURE_T;

    if (rec.z == 0u || cache_state.frame_index - rec.z >= CACHE_STALE_T) {
        if (cache_state.stats_enabled != 0u) {
            atomicAdd(CacheStats(cache_state.stats_bda).fallbacks[region_id], 1u);
        }

        return false;
    }

    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).lookups[region_id], 1u);
    }

    vec2 rg = unpackHalf2x16(rec.x);
    vec2 bz = unpackHalf2x16(rec.y);
    irradiance = pow(vec3(rg.x, rg.y, bz.x), vec3(CACHE_IRRADIANCE_GAMMA));

    return true;
}

// The accumulate half of 02's accumulate→resolve: a deposit adds its
// fixed-point radiance to the entry's accumulator. `count` tags the deposit
// that opens its chain, so the resolve's mean divides by chains, not by the
// partial deposits a multi-hop chain scatters along its path. SHaRC's
// Insert: claim-or-find across the bucket, one CAS per slot — a full bucket
// drops the deposit (capacity pressure reads as churn, 06).
void cache_deposit(uint region_id, uint mc_block, uint voxel_idx, uint face, bool count, vec3 radiance) {
    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).deposits[region_id], 1u);
    }

    uint64_t key = cache_key(region_id, mc_block, voxel_idx, face);
    uint base = cache_base_slot(key);
    CacheKeys keys = CacheKeys(cache_state.keys_bda);
    uint slot = 0u;
    bool landed = false;

    for (uint i = 0u; i < CACHE_BUCKET_SIZE; ++i) {
        uint64_t prev = atomicCompSwap(keys.keys[base + i], uint64_t(0), key);

        if (prev == 0ul || prev == key) {
            slot = base + i;
            landed = true;

            break;
        }
    }

    if (!landed) {
        return;
    }
    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).landed[region_id], 1u);
    }

    vec3 ticks = clamp(radiance, vec3(0.0), vec3(CACHE_ACC_TICK_CAP / CACHE_ACC_SCALE)) * CACHE_ACC_SCALE;
    CacheAccum accum = CacheAccum(cache_state.accum_bda);

    if (ticks.r != 0.0) {
        atomicAdd(accum.ticks[slot].x, uint(ticks.r));
    }
    if (ticks.g != 0.0) {
        atomicAdd(accum.ticks[slot].y, uint(ticks.g));
    }
    if (ticks.b != 0.0) {
        atomicAdd(accum.ticks[slot].z, uint(ticks.b));
    }
    if (count) {
        atomicAdd(accum.ticks[slot].w, 1u);
    }
}

// Resolve-only store (02): gamma-encoded, the stamp renews coverage and the
// chain history feeds the next blend's step size. Plain writes — the
// resolve is the entry's single writer per frame.
void cache_store(uint slot, vec3 irradiance, uint history) {
    vec3 encoded = pow(max(irradiance, vec3(0.0)), vec3(1.0 / CACHE_IRRADIANCE_GAMMA));
    CacheResolved resolved = CacheResolved(cache_state.resolved_bda);

    resolved.recs[slot].x = packHalf2x16(encoded.rg);
    resolved.recs[slot].y = packHalf2x16(vec2(encoded.b, 0.0));
    resolved.recs[slot].z = cache_state.frame_index;
    resolved.recs[slot].w = history;
}

// The resolve pass notes each blended entry once (unique touched faces).
void cache_note_touched(uint region_id) {
    if (cache_state.stats_enabled != 0u) {
        atomicAdd(CacheStats(cache_state.stats_bda).touched[region_id], 1u);
    }
}
