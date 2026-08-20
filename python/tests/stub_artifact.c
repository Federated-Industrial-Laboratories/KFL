/* stub_artifact.c: a hand-written artifact implementing the thirteen
 * k26rl_ symbols just far enough to mint cases no compiled artifact
 * produces: an ABI major above the package's, a missing symbol, a
 * status value outside the package's known set, and a known status
 * whose decode string differs from the usual registry text (so a
 * status table cached at package build time fails the decode gate
 * even for known values).
 *
 * Build variants, selected by compile-time defines:
 *   STUB_ABI_MAJOR2   k26rl_abi_version reports major 2
 *   STUB_ABI_MINOR    k26rl_abi_version reports that minor, while the
 *                     stub still exports only the version 1.0
 *                     surface: an artifact claiming symbols a later
 *                     minor added and carrying none of them
 *   STUB_OMIT_SYMBOL  k26rl_env_reset_seeded is not exported
 *   STUB_SPEC_SIZING_NEGATIVE
 *                     the spec sizing call (capacity 0) returns the
 *                     negated seed-reuse status; the fill call is
 *                     unchanged
 *   STUB_NULL_DECODE  k26rl_status_str returns NULL for the minted
 *                     status 300
 *   STUB_PAIR_SCALE   defines the global double k26rl_stub_pair_scale
 *                     at the given value and scales every observation
 *                     by it; two builds differing only in this value
 *                     export the same symbol name, so a loader that
 *                     lets one artifact's globals into the process
 *                     scope changes the other's observations
 *   (default)         functional stub: create, spec, and the getters
 *                     work; step returns the unknown status 300;
 *                     reset_seeded returns the seed-reuse status,
 *                     whose decode text here is altered
 *
 * Build: cc -shared -fPIC -O2 -I<...>/libk26rl/include stub_artifact.c
 */
#include <stdlib.h>
#include <string.h>

#include "k26rl_env.h"

#define STUB_OBS_TOTAL 3u
#define STUB_ACT_TOTAL 2u
#define STUB_UNKNOWN_STATUS ((K26RlStatus)300)

#ifndef STUB_ABI_MINOR
#define STUB_ABI_MINOR 0
#endif

struct K26RlEnv {
    uint32_t n_envs;
};

#ifdef STUB_PAIR_SCALE
/* Deliberately a global with default visibility, and deliberately the
 * same name in every pair build: compiled without an export map,
 * nothing confines it, so if the loader lets a previously loaded
 * artifact's globals into the process scope, this artifact's reads of
 * the symbol resolve against the other definition and its
 * observations change scale. The isolation gate pins the isolated
 * values. */
double k26rl_stub_pair_scale = STUB_PAIR_SCALE;
#endif

uint32_t k26rl_abi_version(void)
{
#ifdef STUB_ABI_MAJOR2
    return 0x00020000u;
#else
    /* Version 1.0 by default: the surface's first version, carrying
     * none of the symbols later minors added. STUB_ABI_MINOR claims a
     * higher minor without exporting what that minor carries, which
     * is the artifact a consumer must not believe. */
    return 0x00010000u | (uint32_t)STUB_ABI_MINOR;
#endif
}

K26RlStatus k26rl_env_create(uint64_t seed, uint32_t n_envs,
                             K26RlEnv **out_env)
{
    K26RlEnv *env;
    (void)seed;
    if (!out_env || n_envs == 0) return K26RL_E_NULL;
    env = (K26RlEnv *)malloc(sizeof *env);
    if (!env) return K26RL_E_INTERNAL;
    env->n_envs = n_envs;
    *out_env = env;
    return K26RL_OK;
}

K26RlStatus k26rl_env_output(K26RlEnv *env, const char *path)
{
    (void)env;
    (void)path;
    return K26RL_OK;
}

K26RlStatus k26rl_env_reset(K26RlEnv *env)
{
    (void)env;
    return K26RL_OK;
}

#ifndef STUB_OMIT_SYMBOL
K26RlStatus k26rl_env_reset_seeded(K26RlEnv *env, uint64_t seed)
{
    (void)env;
    (void)seed;
    /* Always the seed-reuse refusal, so the caller's decode of a
     * known value can be checked against this stub's altered text. */
    return K26RL_E_SEED_REUSE;
}
#endif

K26RlStatus k26rl_env_step(K26RlEnv *env, const double *actions)
{
    (void)env;
    (void)actions;
    /* A status value outside the version 1 registry: the caller must
     * decode it through this artifact, never through its own table. */
    return STUB_UNKNOWN_STATUS;
}

K26RlStatus k26rl_env_obs(const K26RlEnv *env, double *out)
{
    uint32_t i;
    if (!env || !out) return K26RL_E_NULL;
    for (i = 0; i < env->n_envs * STUB_OBS_TOTAL; i++) {
#ifdef STUB_PAIR_SCALE
        out[i] = 0.25 * (double)(i + 1u) * k26rl_stub_pair_scale;
#else
        out[i] = 0.25 * (double)(i + 1u);
#endif
    }
    return K26RL_OK;
}

K26RlStatus k26rl_env_reward(const K26RlEnv *env, double *out)
{
    uint32_t i;
    if (!env || !out) return K26RL_E_NULL;
    for (i = 0; i < env->n_envs; i++) out[i] = 0.0;
    return K26RL_OK;
}

K26RlStatus k26rl_env_flags(const K26RlEnv *env, uint32_t *out)
{
    uint32_t i;
    if (!env || !out) return K26RL_E_NULL;
    for (i = 0; i < env->n_envs; i++) out[i] = 0u;
    return K26RL_OK;
}

K26RlStatus k26rl_env_fault_codes(const K26RlEnv *env, uint16_t *out)
{
    uint32_t i;
    if (!env || !out) return K26RL_E_NULL;
    for (i = 0; i < env->n_envs; i++) out[i] = 0u;
    return K26RL_OK;
}

/* ---- spec blob ------------------------------------------------------ */

static void put_u16_(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}

static void put_u32_(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static void put_u64_(uint8_t *p, uint64_t v)
{
    put_u32_(p, (uint32_t)(v & 0xffffffffu));
    put_u32_(p + 4, (uint32_t)(v >> 32));
}

static uint64_t f64_bits_(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

static uint32_t tlv_(uint8_t **p, uint16_t tag, uint32_t len,
                     const uint8_t *val)
{
    if (*p) {
        put_u16_(*p, tag);
        put_u32_(*p + 2, len);
        if (len) memcpy(*p + 6, val, len);
        *p += 6 + len;
    }
    return 6 + len;
}

static uint32_t spec_write_(uint8_t *buf, const K26RlEnv *env)
{
    uint8_t *p = buf;
    uint8_t v[20];
    uint32_t total = 0;
    uint32_t i;

    put_u32_(v, k26rl_abi_version());
    total += tlv_(&p, K26RL_TAG_ABI_VERSION, 4, v);
    put_u32_(v, 0x01020304u);
    total += tlv_(&p, K26RL_TAG_ENDIAN_PROBE, 4, v);
    put_u32_(v, 1u);
    total += tlv_(&p, K26RL_TAG_AGENT_COUNT, 4, v);
    put_u32_(v, env->n_envs);
    total += tlv_(&p, K26RL_TAG_N_ENVS, 4, v);
    put_u64_(v, f64_bits_(0.5));
    total += tlv_(&p, K26RL_TAG_CONTROL_DT, 8, v);
    put_u32_(v, 5u);
    total += tlv_(&p, K26RL_TAG_HORIZON, 4, v);
    put_u32_(v, STUB_OBS_TOTAL);
    total += tlv_(&p, K26RL_TAG_OBS_TOTAL, 4, v);
    put_u32_(v, STUB_ACT_TOTAL);
    total += tlv_(&p, K26RL_TAG_ACT_TOTAL, 4, v);
    put_u32_(v, 0u);
    put_u32_(v + 4, 0u);
    put_u32_(v + 8, STUB_OBS_TOTAL);
    total += tlv_(&p, K26RL_TAG_AGENT_OBS_SLICE, 12, v);
    put_u32_(v + 8, STUB_ACT_TOTAL);
    total += tlv_(&p, K26RL_TAG_AGENT_ACT_SLICE, 12, v);
    put_u32_(v, 0u);
    put_u64_(v + 4, f64_bits_(-1.0));
    put_u64_(v + 12, f64_bits_(1.0));
    total += tlv_(&p, K26RL_TAG_ACT_BOUNDS, 20, v);
    put_u32_(v, 1u);
    put_u64_(v + 4, f64_bits_(0.0));
    put_u64_(v + 12, f64_bits_(2.0));
    total += tlv_(&p, K26RL_TAG_ACT_BOUNDS, 20, v);
    put_u32_(v, 0u);
    put_u16_(v + 4, K26RL_ACT_KIND_BOX);
    total += tlv_(&p, K26RL_TAG_ACT_KIND, 6, v);
    put_u32_(v, 1u);
    put_u16_(v + 4, K26RL_ACT_KIND_DISCRETE);
    put_u32_(v + 6, 3u);
    total += tlv_(&p, K26RL_TAG_ACT_KIND, 10, v);
    for (i = 0; i < STUB_OBS_TOTAL; i++) {
        put_u32_(v, i);
        put_u16_(v + 4, K26RL_OBS_KIND_VECTOR);
        total += tlv_(&p, K26RL_TAG_OBS_CHANNEL_KIND, 6, v);
    }
    put_u32_(v, 1u);
    total += tlv_(&p, K26RL_TAG_EPISODE_FLAGS, 4, v);
    return total;
}

int32_t k26rl_env_spec(const K26RlEnv *env, uint8_t *out,
                       uint32_t capacity)
{
    uint32_t need;
    if (!env) return -(int32_t)K26RL_E_NULL;
#ifdef STUB_SPEC_SIZING_NEGATIVE
    /* Only the sizing call fails, so a caller that drops the sign
     * instead of decoding it proceeds to a fill call with a nonsense
     * capacity and is caught on the size disagreement, not here. */
    if (capacity == 0) return -(int32_t)K26RL_E_SEED_REUSE;
#endif
    need = spec_write_(NULL, env);
    if (out && capacity >= need) (void)spec_write_(out, env);
    return (int32_t)need;
}

const char *k26rl_status_str(K26RlStatus status)
{
    switch ((int)status) {
    case K26RL_OK:           return "stub ok";
    case K26RL_E_SEED_REUSE: return "stub altered seed reuse text";
    case (int)STUB_UNKNOWN_STATUS:
#ifdef STUB_NULL_DECODE
        /* A decoder with no string for this value: the caller must
         * render the bare number and invent no name. */
        return NULL;
#else
        return "stub minted status 300";
#endif
    default:                 return "stub other status";
    }
}

void k26rl_env_destroy(K26RlEnv *env)
{
    free(env);
}
