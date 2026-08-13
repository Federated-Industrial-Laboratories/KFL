/* test_rl_emit.c: the Grammar 3.2 dual-mode artifact, smoke level.
 *
 * Gates:
 *   1. `kflc <rl program> -o` produces the batch executable and the
 *      companion shared object; the shared object's dynamic exports
 *      are exactly the k26rl_env.h symbol set.
 *   2. A small batch run with --envs/--episodes/--seed/--out writes
 *      an episode file that the libk26rl reader opens (the open is a
 *      sequential CRC pass over every frame), reports a clean close,
 *      and indexes at least K completed episodes per environment.
 *   3. Serve path: dlopen the shared object, resolve every frozen
 *      symbol, create/step/reset/destroy against the header, parse
 *      the env_spec TLV blob, and check the refusal statuses (seed
 *      reuse, output timing, output exists).
 *   4. Bit identity: batch and serve on the same seed and the same
 *      (default) action stream produce bitwise-identical observation,
 *      reward, and flag streams, checked against the recorded episode.
 *   5. Fault path: a program whose reward divides by zero at a known
 *      step faults exactly there with the documented per-environment
 *      carriers (flag bit 2, fault code, pre-step observations, zero
 *      reward), auto-resets on the next step, and faults again in the
 *      next episode (a fault-reset-fault sequence).
 *   6. Batch completion under diverging draws: two seed/configuration
 *      pairs whose episode resets draw the craft close enough to the
 *      central body that the trajectory blows up must still complete
 *      within a generous walltime bound, with every environment
 *      recording its requested episode count. Pins the fix for the
 *      sector fold in k26astro_pos_normalise, which used to walk a
 *      diverged offset back one sector per loop iteration and wedge
 *      the batch executable inside a single step. This gate pins
 *      termination and episode accounting ONLY: the recorded
 *      episodes carry large finite observations with no fault code,
 *      because the divergence detector keys on finiteness, and
 *      whether large-but-finite magnitudes should fault is an open
 *      policy question outside this gate's claim.
 *
 * Pattern: run ./bin/kflc via system() with the stack's include and
 * archive paths, then drive the artifacts directly. Requires the
 * sibling libraries to be built (make at the repository root).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "k26rl_env.h"
#include "k26rl_episode.h"

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#define WORK_DIR "/tmp/kflc_rl_emit_test"

static const char *const INCLUDE_DIRS[] = {
    "../libk26astro_rt/include",   "../libk26astro_body/include",
    "../libk26astro_core/include", "../libk26astro_grav/include",
    "../libk26astro_conics/include", "../libk26astro_ephem/include",
    "../libk26astro_vehicle/include", "../libk26astro_atmos/include",
    "../libk26tick/include",       "../libk26compute/include",
    "../libk26m3d/include",        "../libk26rl/include",
    "../libk26rng/include",        NULL
};

static const char *const LINK_LIBS[] = {
    "../libk26rl/libk26rl.a",
    "../libk26rng/libk26rng.a",
    "../libk26astro_rt/libk26astro_rt.a",
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

/* The frozen export set, sorted, one name per line, as the nm sweep
 * below produces it. */
static const char *const FROZEN_EXPORTS =
    "k26rl_abi_version\n"
    "k26rl_env_create\n"
    "k26rl_env_destroy\n"
    "k26rl_env_fault_codes\n"
    "k26rl_env_flags\n"
    "k26rl_env_obs\n"
    "k26rl_env_output\n"
    "k26rl_env_reset\n"
    "k26rl_env_reset_seeded\n"
    "k26rl_env_reward\n"
    "k26rl_env_spec\n"
    "k26rl_env_step\n"
    "k26rl_status_str\n";

/* Resolved frozen surface, filled by dlsym. */
typedef struct {
    uint32_t    (*abi_version)(void);
    K26RlStatus (*create)(uint64_t, uint32_t, K26RlEnv **);
    K26RlStatus (*output)(K26RlEnv *, const char *);
    K26RlStatus (*reset)(K26RlEnv *);
    K26RlStatus (*reset_seeded)(K26RlEnv *, uint64_t);
    K26RlStatus (*step)(K26RlEnv *, const double *);
    K26RlStatus (*obs)(const K26RlEnv *, double *);
    K26RlStatus (*reward)(const K26RlEnv *, double *);
    K26RlStatus (*flags)(const K26RlEnv *, uint32_t *);
    K26RlStatus (*fault_codes)(const K26RlEnv *, uint16_t *);
    int32_t     (*spec)(const K26RlEnv *, uint8_t *, uint32_t);
    const char *(*status_str)(K26RlStatus);
    void        (*destroy)(K26RlEnv *);
} RlSurface;

static void resolve_surface_(void *so, RlSurface *s)
{
    /* dlsym returns void *; the ISO-C-clean conversion goes through
     * memcpy of the pointer value. */
#define RESOLVE(field, name) do { \
        void *p_ = dlsym(so, name); \
        ASSERT(p_ != NULL); \
        memcpy(&s->field, &p_, sizeof p_); \
    } while (0)
    RESOLVE(abi_version,  "k26rl_abi_version");
    RESOLVE(create,       "k26rl_env_create");
    RESOLVE(output,       "k26rl_env_output");
    RESOLVE(reset,        "k26rl_env_reset");
    RESOLVE(reset_seeded, "k26rl_env_reset_seeded");
    RESOLVE(step,         "k26rl_env_step");
    RESOLVE(obs,          "k26rl_env_obs");
    RESOLVE(reward,       "k26rl_env_reward");
    RESOLVE(flags,        "k26rl_env_flags");
    RESOLVE(fault_codes,  "k26rl_env_fault_codes");
    RESOLVE(spec,         "k26rl_env_spec");
    RESOLVE(status_str,   "k26rl_status_str");
    RESOLVE(destroy,      "k26rl_env_destroy");
#undef RESOLVE
}

static void run_or_die_(const char *cmd)
{
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr, "command failed (rc=%d): %s\n", rc, cmd);
        exit(1);
    }
}

/* Compile a .kfl source through the built kflc with the stack's
 * include and archive lists. */
static void compile_rl_(const char *kfl_path, const char *out_path)
{
    char cflags[4096];
    int n = snprintf(cflags, sizeof cflags,
        "-O2 -g -std=c++11 -Wno-format-truncation "
        "-ffp-contract=off -fexcess-precision=standard");
    for (int i = 0; INCLUDE_DIRS[i]; i++) {
        n += snprintf(cflags + n, sizeof cflags - (size_t)n, " -I%s",
                      INCLUDE_DIRS[i]);
    }
    ASSERT((size_t)n < sizeof cflags);

    char ldlibs[4096];
    n = 0;
    for (int i = 0; LINK_LIBS[i]; i++) {
        n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n, "%s%s",
                      i ? " " : "", LINK_LIBS[i]);
    }
    n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n,
                  " -lgfortran -lm");
    ASSERT((size_t)n < sizeof ldlibs);

    char cmd[16384];
    n = snprintf(cmd, sizeof cmd,
        "KFLC_CFLAGS=\"%s\" KFLC_LDLIBS=\"%s\" ./bin/kflc %s -o %s "
        "> %s/kflc.log 2>&1",
        cflags, ldlibs, kfl_path, out_path, WORK_DIR);
    ASSERT((size_t)n < sizeof cmd);
    int rc = system(cmd);
    if (rc != 0) {
        char show[256];
        snprintf(show, sizeof show, "cat %s/kflc.log", WORK_DIR);
        (void)!system(show);
        fprintf(stderr, "kflc failed (rc=%d) for %s\n", rc, kfl_path);
        exit(1);
    }
}

static void write_file_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    ASSERT(f != NULL);
    fputs(content, f);
    fclose(f);
}

static int file_exists_(const char *path)
{
    struct stat st;
    return stat(path, &st) == 0;
}

/* Open a batch episode file and require a clean close and at least
 * `need` completed episodes for each of `n_envs` environments. */
static void check_batch_counts_(const char *path, uint32_t n_envs,
                                uint32_t need)
{
    K26RlEpisodeReader *rd = NULL;
    ASSERT(k26rl_episode_reader_open(path, &rd) == K26RL_OK);
    K26RlEpisodeInfo info;
    ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
    ASSERT(info.clean_close == 1);
    ASSERT(info.n_envs == n_envs);
    uint32_t counts[16] = { 0 };
    ASSERT(n_envs <= 16);
    for (uint32_t k = 0; k < info.episode_count; k++) {
        uint32_t ord = 0, env = 0, ep = 0;
        ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &ep) == K26RL_OK);
        ASSERT(env < n_envs);
        counts[env]++;
    }
    for (uint32_t e = 0; e < n_envs; e++) ASSERT(counts[e] >= need);
    k26rl_episode_reader_close(rd);
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
} SpecView;

static uint32_t get_u32_(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t get_u16_(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint64_t get_u64_(const uint8_t *p)
{
    return (uint64_t)get_u32_(p) | ((uint64_t)get_u32_(p + 4) << 32);
}

static void parse_spec_(const uint8_t *blob, uint32_t len, SpecView *v)
{
    memset(v, 0, sizeof *v);
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint16_t tag = get_u16_(blob + off);
        uint32_t l   = get_u32_(blob + off + 2);
        const uint8_t *val = blob + off + 6;
        ASSERT(off + 6 + l <= len);
        switch (tag) {
        case K26RL_TAG_ABI_VERSION:  v->abi_version  = get_u32_(val); break;
        case K26RL_TAG_ENDIAN_PROBE: v->endian_probe = get_u32_(val); break;
        case K26RL_TAG_AGENT_COUNT:  v->agent_count  = get_u32_(val); break;
        case K26RL_TAG_N_ENVS:       v->n_envs       = get_u32_(val); break;
        case K26RL_TAG_CONTROL_DT:   v->control_dt_bits = get_u64_(val); break;
        case K26RL_TAG_HORIZON:      v->horizon      = get_u32_(val); break;
        case K26RL_TAG_OBS_TOTAL:    v->obs_total    = get_u32_(val); break;
        case K26RL_TAG_ACT_TOTAL:    v->act_total    = get_u32_(val); break;
        case K26RL_TAG_ACT_BOUNDS:   v->saw_bounds   = 1; break;
        case K26RL_TAG_ACT_KIND:     v->saw_kind     = 1; break;
        case K26RL_TAG_OBS_CHANNEL_NAME: v->saw_names = 1; break;
        case K26RL_TAG_EPISODE_FLAGS: v->episode_flags = get_u32_(val); break;
        default: break;   /* unknown tags skipped by length */
        }
        off += 6 + l;
    }
    ASSERT(off == len);
}

/* ---- The fault fixture ----------------------------------------------- *
 *
 * reward 1.0 / (3.0 - episode.steps): finite at transitions 1 and 2,
 * division by zero (infinite, non-finite) at transition 3, so every
 * episode faults exactly there. */
static const char *const FAULT_KFL =
    "form RL_FAULT\n"
    "fn world fault_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 10\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0 / (3.0 - episode.steps)\n"
    "    end\n"
    "end\n"
    "end\n";

static int n_pass = 0;

int main(void)
{
    /* This gate drives compiled artifacts against the full library
     * stack; without the sibling archives built there is nothing to
     * link, so skip (exit 77, the tree's skip convention). */
    for (int i = 0; LINK_LIBS[i]; i++) {
        if (!file_exists_(LINK_LIBS[i])) {
            fprintf(stderr, "test_rl_emit: skip: %s not built\n",
                    LINK_LIBS[i]);
            return 77;
        }
    }

    run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    /* Gate 1: both artifacts from one invocation; exact export set. */
    compile_rl_("integration_tests/rl_pointing.kfl",
                WORK_DIR "/rl_pointing");
    ASSERT(file_exists_(WORK_DIR "/rl_pointing"));
    ASSERT(file_exists_(WORK_DIR "/rl_pointing.rlenv.so"));
    run_or_die_("nm -D --defined-only " WORK_DIR "/rl_pointing.rlenv.so"
                " | awk '{ print $NF }' | sort > " WORK_DIR "/exports");
    {
        FILE *f = fopen(WORK_DIR "/exports", "rb");
        ASSERT(f != NULL);
        char buf[4096];
        size_t got = fread(buf, 1, sizeof buf - 1, f);
        buf[got] = '\0';
        fclose(f);
        if (strcmp(buf, FROZEN_EXPORTS) != 0) {
            fprintf(stderr, "export set mismatch; got:\n%s", buf);
            ASSERT(0);
        }
    }
    n_pass++;
    printf("gate 1: dual artifact + exact export set: OK\n");

    /* Gate 2: batch run, then the reader's sequential CRC pass. */
    run_or_die_(WORK_DIR "/rl_pointing --envs 2 --episodes 2 --seed 42"
                " --out " WORK_DIR "/batch.k26epi > /dev/null");
    K26RlEpisodeReader *rd = NULL;
    ASSERT(k26rl_episode_reader_open(WORK_DIR "/batch.k26epi", &rd)
           == K26RL_OK);
    K26RlEpisodeInfo info;
    ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
    ASSERT(info.clean_close == 1);
    ASSERT(info.n_envs == 2);
    ASSERT(info.governing_seed == 42);
    ASSERT(info.rekey_ordinal == 0);
    ASSERT(info.obs_total == 4);
    ASSERT(info.act_total == 2);
    ASSERT(info.agent_count == 1);
    {
        uint32_t per_env[2] = { 0, 0 };
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord = 0, env = 0, ep = 0;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &ep)
                   == K26RL_OK);
            ASSERT(ord == 0);
            ASSERT(env < 2);
            per_env[env]++;
        }
        ASSERT(per_env[0] >= 2 && per_env[1] >= 2);
    }
    n_pass++;
    printf("gate 2: batch episode file reads back, %u episodes: OK\n",
           info.episode_count);

    /* Gate 3 + 4: serve path against the frozen header, and the
     * batch/serve bit-identity check over episode 0 of each
     * environment. */
    void *so = dlopen(WORK_DIR "/rl_pointing.rlenv.so",
                      RTLD_NOW | RTLD_LOCAL);
    if (!so) fprintf(stderr, "dlopen: %s\n", dlerror());
    ASSERT(so != NULL);
    RlSurface s;
    resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    K26RlEnv *env = NULL;
    ASSERT(s.create(42, 2, &env) == K26RL_OK);
    ASSERT(env != NULL);

    int32_t spec_len = s.spec(env, NULL, 0);
    ASSERT(spec_len > 0);
    uint8_t *blob = malloc((size_t)spec_len);
    ASSERT(blob != NULL);
    ASSERT(s.spec(env, blob, (uint32_t)spec_len) == spec_len);
    SpecView v;
    parse_spec_(blob, (uint32_t)spec_len, &v);
    ASSERT(v.abi_version == K26RL_ABI_VERSION);
    ASSERT(v.endian_probe == 0x01020304u);
    ASSERT(v.agent_count == 1);
    ASSERT(v.n_envs == 2);
    ASSERT(v.obs_total == 4);
    ASSERT(v.act_total == 2);
    ASSERT(v.horizon == 1000);
    {
        double dt;
        uint64_t bits = v.control_dt_bits;
        memcpy(&dt, &bits, sizeof dt);
        ASSERT(dt == 0.1);
    }
    ASSERT(v.saw_bounds && v.saw_kind && v.saw_names);
    ASSERT(v.episode_flags & 1u);   /* auto-reset on */
    /* The recorded file embeds the same blob verbatim. */
    {
        const uint8_t *fblob = NULL;
        uint32_t flen = 0;
        ASSERT(k26rl_episode_reader_spec(rd, &fblob, &flen) == K26RL_OK);
        ASSERT(flen == (uint32_t)spec_len);
        ASSERT(memcmp(fblob, blob, flen) == 0);
    }
    free(blob);
    n_pass++;
    printf("gate 3: dlopen + create + spec round trip: OK\n");

    /* Refusal statuses. */
    ASSERT(s.reset_seeded(env, 42) == K26RL_E_SEED_REUSE);
    ASSERT(s.output(env, WORK_DIR "/batch.k26epi")
           == K26RL_E_OUTPUT_EXISTS);

    /* Bit identity: drive the same default action stream (all
     * channels at their declared defaults: 0.0, 0.0) and compare
     * against the recorded episode streams. */
    {
        K26RlEpisodeData ep0, ep1;
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep0) == K26RL_OK);
        ASSERT(k26rl_episode_read(rd, 0, 1, 0, &ep1) == K26RL_OK);
        ASSERT(ep0.end_reason == K26RL_END_TERMINATED);
        ASSERT(ep0.step_count == 901);   /* terminated when steps > 900 */
        ASSERT(ep1.step_count == 901);

        double actions[2 * 2] = { 0.0, 0.0, 0.0, 0.0 };
        double obs[2 * 4];
        double rew[2];
        uint32_t fl[2];

        /* Initial observations equal the episode-start frames'. */
        ASSERT(s.obs(env, obs) == K26RL_OK);
        ASSERT(memcmp(obs, ep0.initial_obs, 4 * sizeof(double)) == 0);
        ASSERT(memcmp(obs + 4, ep1.initial_obs, 4 * sizeof(double)) == 0);

        for (uint32_t t = 0; t < 901; t++) {
            ASSERT(s.step(env, actions) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.reward(env, rew) == K26RL_OK);
            ASSERT(s.flags(env, fl) == K26RL_OK);
            ASSERT(memcmp(obs, ep0.obs + (size_t)t * 4,
                          4 * sizeof(double)) == 0);
            ASSERT(memcmp(obs + 4, ep1.obs + (size_t)t * 4,
                          4 * sizeof(double)) == 0);
            ASSERT(memcmp(&rew[0], &ep0.rewards[t], sizeof(double)) == 0);
            ASSERT(memcmp(&rew[1], &ep1.rewards[t], sizeof(double)) == 0);
            ASSERT(fl[0] == ep0.flags[t]);
            ASSERT(fl[1] == ep1.flags[t]);
            ASSERT(memcmp(&ep0.applied_dt[t], &(double){0.1},
                          sizeof(double)) == 0);
        }
        /* Output timing: a step has happened, no boundary. */
        ASSERT(s.output(env, WORK_DIR "/other.k26epi")
               == K26RL_E_OUTPUT_TIMING);
        k26rl_episode_free(&ep0);
        k26rl_episode_free(&ep1);
    }
    s.destroy(env);
    env = NULL;
    n_pass++;
    printf("gate 4: batch/serve bit identity over 901 steps"
           " x 2 environments: OK\n");
    k26rl_episode_reader_close(rd);
    rd = NULL;

    /* Gate 5: the fault path, twice through a reset (a
     * fault-reset-fault sequence). */
    write_file_(WORK_DIR "/rl_fault.kfl", FAULT_KFL);
    compile_rl_(WORK_DIR "/rl_fault.kfl", WORK_DIR "/rl_fault");
    ASSERT(file_exists_(WORK_DIR "/rl_fault.rlenv.so"));
    void *so2 = dlopen(WORK_DIR "/rl_fault.rlenv.so",
                       RTLD_NOW | RTLD_LOCAL);
    ASSERT(so2 != NULL);
    RlSurface f;
    resolve_surface_(so2, &f);
    K26RlEnv *fe = NULL;
    ASSERT(f.create(7, 1, &fe) == K26RL_OK);
    {
        double act[1] = { 0.0 };
        double obs_pre[4], obs_now[4];
        double rew;
        uint32_t fl;
        uint16_t fc;

        ASSERT(f.flags(fe, &fl) == K26RL_OK);
        ASSERT(fl == 0);

        for (int episode = 0; episode < 2; episode++) {
            /* Two clean transitions. */
            for (int t = 0; t < 2; t++) {
                ASSERT(f.step(fe, act) == K26RL_OK);
                ASSERT(f.flags(fe, &fl) == K26RL_OK);
                ASSERT(f.fault_codes(fe, &fc) == K26RL_OK);
                ASSERT(fl == 0);
                ASSERT(fc == 0);
                ASSERT(f.reward(fe, &rew) == K26RL_OK);
                ASSERT(isfinite(rew) && rew != 0.0);
            }
            ASSERT(f.obs(fe, obs_pre) == K26RL_OK);

            /* Transition 3: division by zero in the reward. The call
             * succeeds; the fault is per environment; no transition
             * completes, so the observations are the pre-step ones
             * and the reward is zero. */
            ASSERT(f.step(fe, act) == K26RL_OK);
            ASSERT(f.flags(fe, &fl) == K26RL_OK);
            ASSERT(fl == K26RL_FLAG_FAULT);
            ASSERT(f.fault_codes(fe, &fc) == K26RL_OK);
            ASSERT(fc == (uint16_t)K26RL_E_ENV_INTERNAL);
            ASSERT(f.obs(fe, obs_now) == K26RL_OK);
            ASSERT(memcmp(obs_now, obs_pre, sizeof obs_pre) == 0);
            ASSERT(f.reward(fe, &rew) == K26RL_OK);
            ASSERT(rew == 0.0);

            /* Auto-reset boundary: only the reset bit, fault cleared,
             * zero reward. */
            ASSERT(f.step(fe, act) == K26RL_OK);
            ASSERT(f.flags(fe, &fl) == K26RL_OK);
            ASSERT(fl == K26RL_FLAG_RESET_BOUNDARY);
            ASSERT(f.fault_codes(fe, &fc) == K26RL_OK);
            ASSERT(fc == 0);
            ASSERT(f.reward(fe, &rew) == K26RL_OK);
            ASSERT(rew == 0.0);
        }

        /* Explicit reset still works after a fault cycle, and the
         * decoder answers for the fault reason. */
        ASSERT(f.reset(fe) == K26RL_OK);
        ASSERT(strlen(f.status_str(K26RL_E_ENV_INTERNAL)) > 0);
    }
    f.destroy(fe);
    n_pass++;
    printf("gate 5: fault-reset-fault sequence with pre-step"
           " observations held: OK\n");

    dlclose(so2);
    dlclose(so);

    /* Gate 6: batch completion under diverging draws. Both
     * configurations used to wedge inside a single step: the episode
     * reset can draw the craft essentially at the central body's
     * centre, the near-singular dynamics throw a coordinate out by
     * around 1e23 m, and the sector fold walked it back one 2^36 m
     * sector per iteration (and never finished at all once an offset
     * went non-finite). The 120 s bound is generous walltime for
     * runs that take about a second when healthy, not a performance
     * assertion. */
    run_or_die_("timeout 120 " WORK_DIR "/rl_pointing"
                " --envs 2 --episodes 3 --seed 123456789"
                " --out " WORK_DIR "/diverge1.k26epi > /dev/null");
    check_batch_counts_(WORK_DIR "/diverge1.k26epi", 2, 3);
    run_or_die_("timeout 120 " WORK_DIR "/rl_pointing"
                " --envs 4 --episodes 6 --seed 20260813"
                " --out " WORK_DIR "/diverge2.k26epi > /dev/null");
    check_batch_counts_(WORK_DIR "/diverge2.k26epi", 4, 6);
    n_pass++;
    printf("gate 6: diverging-draw batch runs complete with full"
           " episode counts: OK\n");

    printf("test_rl_emit: %d gates passed\n", n_pass);
    return 0;
}
