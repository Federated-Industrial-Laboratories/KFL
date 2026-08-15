/* rl_gate_util.h - shared plumbing for the Grammar 3.2 environment
 * gate binaries (test_rl_determinism, test_rl_fault, test_rl_resim).
 *
 * Everything here follows the test_rl_emit pattern: compile a .kfl
 * source through the built kflc with the stack's include and archive
 * lists, dlopen the produced shared object, resolve the frozen
 * surface, and walk the env_spec TLV blob. Gates that need the stack
 * archives call rl_libs_present_() first and skip with exit code 77
 * when a sibling archive is missing.
 */
#ifndef RL_GATE_UTIL_H
#define RL_GATE_UTIL_H

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "k26rl_env.h"

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static const char *const RL_INCLUDE_DIRS_[] = {
    "../libk26astro_rt/include",   "../libk26astro_body/include",
    "../libk26astro_core/include", "../libk26astro_grav/include",
    "../libk26astro_conics/include", "../libk26astro_ephem/include",
    "../libk26astro_vehicle/include", "../libk26astro_att/include",
    "../libk26astro_atmos/include",
    "../libk26astro_geomag/include",
    "../libk26astro_coll/include",
    "../libk26astro_prox/include",
    "../libk26tick/include",       "../libk26compute/include",
    "../libk26m3d/include",        "../libk26rl/include",
    "../libk26rng/include",        NULL
};

static const char *const RL_LINK_LIBS_[] = {
    "../libk26rl/libk26rl.a",
    "../libk26rng/libk26rng.a",
    "../libk26astro_rt/libk26astro_rt.a",
    "../libk26astro_att/libk26astro_att.a",
    "../libk26astro_coll/libk26astro_coll.a",
    "../libk26astro_prox/libk26astro_prox.a",
    "../libk26astro_geomag/libk26astro_geomag.a",
    "../libk26astro_vehicle/libk26astro_vehicle.a",
    "../libk26astro_atmos/libk26astro_atmos.a",
    "../libk26astro_grav/libk26astro_grav.a",
    "../libk26astro_conics/libk26astro_conics.a",
    "../libk26astro_body/libk26astro_body.a",
    "../libk26astro_ephem/libk26astro_ephem.a",
    "../libk26astro_core/libk26astro_core.a",
    "../libk26compute/libk26compute.a",
    "../libk26tick/libk26tick.a",
    "../libk26m3d/libk26m3d.a",
    NULL
};

static inline int rl_file_exists_(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

/* 1 when every sibling stack archive is built; callers skip (77)
 * otherwise. */
static inline int rl_libs_present_(const char *gate_name)
{
    for (int i = 0; RL_LINK_LIBS_[i]; i++) {
        if (!rl_file_exists_(RL_LINK_LIBS_[i])) {
            fprintf(stderr, "%s: skip: %s not built\n", gate_name,
                    RL_LINK_LIBS_[i]);
            return 0;
        }
    }
    return 1;
}

static inline void rl_run_or_die_(const char *cmd)
{
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "command failed (rc=%d): %s\n", rc, cmd);
        exit(1);
    }
}

static inline void rl_write_file_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    ASSERT(f != NULL);
    fputs(content, f);
    fclose(f);
}

/* Compile a .kfl source through the built kflc with the stack's
 * include and archive lists; the kflc log lands in <work_dir>/kflc.log
 * and is shown on failure. */
static inline void rl_compile_(const char *kfl_path, const char *out_path,
                               const char *work_dir)
{
    char cflags[4096];
    int n = snprintf(cflags, sizeof cflags,
        "-O2 -g -std=c++11 -Wno-format-truncation "
        "-ffp-contract=off -fexcess-precision=standard");
    for (int i = 0; RL_INCLUDE_DIRS_[i]; i++) {
        n += snprintf(cflags + n, sizeof cflags - (size_t)n, " -I%s",
                      RL_INCLUDE_DIRS_[i]);
    }
    ASSERT((size_t)n < sizeof cflags);

    char ldlibs[4096];
    n = 0;
    for (int i = 0; RL_LINK_LIBS_[i]; i++) {
        n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n, "%s%s",
                      i ? " " : "", RL_LINK_LIBS_[i]);
    }
    n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n,
                  " -lgfortran -lm");
    ASSERT((size_t)n < sizeof ldlibs);

    char cmd[16384];
    n = snprintf(cmd, sizeof cmd,
        "KFLC_CFLAGS=\"%s\" KFLC_LDLIBS=\"%s\" ./bin/kflc %s -o %s "
        "> %s/kflc.log 2>&1",
        cflags, ldlibs, kfl_path, out_path, work_dir);
    ASSERT((size_t)n < sizeof cmd);
    int rc = system(cmd);
    if (rc != 0) {
        char show[512];
        snprintf(show, sizeof show, "cat %s/kflc.log", work_dir);
        (void)!system(show);
        fprintf(stderr, "kflc failed (rc=%d) for %s\n", rc, kfl_path);
        exit(1);
    }
}

/* Resolved frozen surface, filled by dlsym. */
typedef struct {
    uint32_t    (*abi_version)(void);
    K26RlStatus (*create)(uint64_t, uint32_t, K26RlEnv **);
    K26RlStatus (*output)(K26RlEnv *, const char *);
    K26RlStatus (*tap)(K26RlEnv *, const char *);
    K26RlStatus (*reset)(K26RlEnv *);
    K26RlStatus (*reset_seeded)(K26RlEnv *, uint64_t);
    K26RlStatus (*step)(K26RlEnv *, const double *);
    K26RlStatus (*obs)(const K26RlEnv *, double *);
    K26RlStatus (*reward)(const K26RlEnv *, double *);
    K26RlStatus (*flags)(const K26RlEnv *, uint32_t *);
    K26RlStatus (*fault_codes)(const K26RlEnv *, uint16_t *);
    int32_t     (*spec)(const K26RlEnv *, uint8_t *, uint32_t);
    int32_t     (*bodies)(const K26RlEnv *, uint32_t, double *,
                          uint32_t);
    int32_t     (*attitudes)(const K26RlEnv *, double *, uint32_t);
    const char *(*status_str)(K26RlStatus);
    void        (*destroy)(K26RlEnv *);
} RlSurface;

static inline void rl_resolve_surface_(void *so, RlSurface *s)
{
    /* dlsym returns void *; the ISO-C-clean conversion goes through
     * memcpy of the pointer value. */
#define RL_RESOLVE_(field, name) do { \
        void *p_ = dlsym(so, name); \
        ASSERT(p_ != NULL); \
        memcpy(&s->field, &p_, sizeof p_); \
    } while (0)
    RL_RESOLVE_(abi_version,  "k26rl_abi_version");
    RL_RESOLVE_(create,       "k26rl_env_create");
    RL_RESOLVE_(output,       "k26rl_env_output");
    RL_RESOLVE_(tap,          "k26rl_env_tap");
    RL_RESOLVE_(reset,        "k26rl_env_reset");
    RL_RESOLVE_(reset_seeded, "k26rl_env_reset_seeded");
    RL_RESOLVE_(step,         "k26rl_env_step");
    RL_RESOLVE_(obs,          "k26rl_env_obs");
    RL_RESOLVE_(reward,       "k26rl_env_reward");
    RL_RESOLVE_(flags,        "k26rl_env_flags");
    RL_RESOLVE_(fault_codes,  "k26rl_env_fault_codes");
    RL_RESOLVE_(spec,         "k26rl_env_spec");
    RL_RESOLVE_(bodies,       "k26rl_env_bodies");
    RL_RESOLVE_(attitudes,    "k26rl_env_attitudes");
    RL_RESOLVE_(status_str,   "k26rl_status_str");
    RL_RESOLVE_(destroy,      "k26rl_env_destroy");
#undef RL_RESOLVE_
}

/* dlopen with a required-open assertion and the same flags every
 * consumer of the artifact uses. */
static inline void *rl_dlopen_(const char *path)
{
    void *so = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!so) fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
    ASSERT(so != NULL);
    return so;
}

/* ---- Spec TLV walk -------------------------------------------------- */

typedef struct {
    uint32_t abi_version;
    uint32_t endian_probe;
    uint32_t agent_count;
    uint32_t n_envs;
    uint64_t control_dt_bits;
    uint32_t horizon;
    uint32_t obs_total;
    uint32_t act_total;
    uint32_t episode_flags;
    int      saw_bounds;
    int      saw_kind;
    int      saw_names;
    uint16_t modes[64];        /* per channel, K26RL_TAG_OBS_CHANNEL_MODE */
    int      n_modes;
    char     body_names[16][64];
    int      n_body_names;
} RlSpecView;

static inline uint32_t rl_get_u32_(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint16_t rl_get_u16_(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint64_t rl_get_u64_(const uint8_t *p)
{
    return (uint64_t)rl_get_u32_(p) | ((uint64_t)rl_get_u32_(p + 4) << 32);
}

static inline double rl_get_f64_(const uint8_t *p)
{
    uint64_t bits = rl_get_u64_(p);
    double v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

static inline void rl_parse_spec_(const uint8_t *blob, uint32_t len,
                                  RlSpecView *v)
{
    memset(v, 0, sizeof *v);
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint16_t tag = rl_get_u16_(blob + off);
        uint32_t l   = rl_get_u32_(blob + off + 2);
        const uint8_t *val = blob + off + 6;
        ASSERT(off + 6 + l <= len);
        switch (tag) {
        case K26RL_TAG_ABI_VERSION:  v->abi_version  = rl_get_u32_(val); break;
        case K26RL_TAG_ENDIAN_PROBE: v->endian_probe = rl_get_u32_(val); break;
        case K26RL_TAG_AGENT_COUNT:  v->agent_count  = rl_get_u32_(val); break;
        case K26RL_TAG_N_ENVS:       v->n_envs       = rl_get_u32_(val); break;
        case K26RL_TAG_CONTROL_DT:   v->control_dt_bits = rl_get_u64_(val); break;
        case K26RL_TAG_HORIZON:      v->horizon      = rl_get_u32_(val); break;
        case K26RL_TAG_OBS_TOTAL:    v->obs_total    = rl_get_u32_(val); break;
        case K26RL_TAG_ACT_TOTAL:    v->act_total    = rl_get_u32_(val); break;
        case K26RL_TAG_ACT_BOUNDS:   v->saw_bounds   = 1; break;
        case K26RL_TAG_ACT_KIND:     v->saw_kind     = 1; break;
        case K26RL_TAG_OBS_CHANNEL_NAME: v->saw_names = 1; break;
        case K26RL_TAG_EPISODE_FLAGS: v->episode_flags = rl_get_u32_(val); break;
        case K26RL_TAG_OBS_CHANNEL_MODE: {
            uint32_t ch = rl_get_u32_(val);
            if (ch < 64) {
                v->modes[ch] = rl_get_u16_(val + 4);
                if ((int)ch + 1 > v->n_modes) v->n_modes = (int)ch + 1;
            }
            break;
        }
        case K26RL_TAG_BODY_NAME: {
            uint32_t bi = rl_get_u32_(val);
            if (bi < 16 && l >= 4) {
                uint32_t nl = l - 4;
                if (nl > 63) nl = 63;
                memcpy(v->body_names[bi], val + 4, nl);
                v->body_names[bi][nl] = '\0';
                if ((int)bi + 1 > v->n_body_names) v->n_body_names = (int)bi + 1;
            }
            break;
        }
        default: break;   /* unknown tags skipped by length */
        }
        off += 6 + l;
    }
    ASSERT(off == len);
}

/* Whole-file bitwise comparison; returns 1 when equal. */
static inline int rl_files_equal_(const char *a, const char *b)
{
    FILE *fa = fopen(a, "rb");
    FILE *fb = fopen(b, "rb");
    ASSERT(fa != NULL && fb != NULL);
    int equal = 1;
    for (;;) {
        unsigned char ba[4096], bb[4096];
        size_t na = fread(ba, 1, sizeof ba, fa);
        size_t nb = fread(bb, 1, sizeof bb, fb);
        if (na != nb || memcmp(ba, bb, na) != 0) { equal = 0; break; }
        if (na == 0) break;
    }
    fclose(fa);
    fclose(fb);
    return equal;
}

#endif /* RL_GATE_UTIL_H */
