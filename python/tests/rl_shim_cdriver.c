/* rl_shim_cdriver.c: the C reference driver for the package's
 * cross-driver gates. Drives a compiled artifact's shared object
 * with the same scripted action stream the Python gates compute,
 * literal for literal, so the two drivers feed bit-identical
 * doubles.
 *
 * Modes:
 *   rl_shim_cdriver emit <so> <seed> <n_envs> <T> <episode_out>
 *       enable episode output before the first step, drive T steps,
 *       destroy; the artifact writes the episode file.
 *   rl_shim_cdriver dump <so> <seed> <n_envs> <T> <dump_out>
 *       drive T steps with no output and append the raw getter
 *       results per step to dump_out: n*obs_total doubles, n
 *       rewards, n uint32 flag words.
 *
 * The scripted stream leaves the declared action bounds and arity
 * on purpose; the artifact takes values as given.
 *
 * Build: cc -O2 -I<...>/libk26rl/include rl_shim_cdriver.c -ldl -lm
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k26rl_env.h"

#define DIE(...) do { fprintf(stderr, __VA_ARGS__); \
    fputc('\n', stderr); exit(1); } while (0)

typedef struct {
    uint32_t    (*abi_version)(void);
    K26RlStatus (*create)(uint64_t, uint32_t, K26RlEnv **);
    K26RlStatus (*output)(K26RlEnv *, const char *);
    K26RlStatus (*step)(K26RlEnv *, const double *);
    K26RlStatus (*obs)(const K26RlEnv *, double *);
    K26RlStatus (*reward)(const K26RlEnv *, double *);
    K26RlStatus (*flags)(const K26RlEnv *, uint32_t *);
    int32_t     (*spec)(const K26RlEnv *, uint8_t *, uint32_t);
    void        (*destroy)(K26RlEnv *);
} Surface;

static void *sym_(void *so, const char *name)
{
    void *p = dlsym(so, name);
    if (!p) DIE("missing symbol %s: %s", name, dlerror());
    return p;
}

/* The scripted stream: keep these expressions identical to the
 * Python side's act_thrust and act_gear. */
static double act_thrust_(uint32_t t, uint32_t e)
{
    return ((double)((t * 7u + e * 3u) % 13u)) / 13.0 * 4.0 - 2.0;
}

static double act_gear_(uint32_t t, uint32_t e)
{
    return (double)((t + e) % 5u);
}

static uint32_t spec_u32_(const uint8_t *blob, uint32_t len,
                          uint16_t want)
{
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint16_t tag = (uint16_t)(blob[off] | (blob[off + 1] << 8));
        uint32_t l = (uint32_t)blob[off + 2]
            | ((uint32_t)blob[off + 3] << 8)
            | ((uint32_t)blob[off + 4] << 16)
            | ((uint32_t)blob[off + 5] << 24);
        if (tag == want && l == 4) {
            const uint8_t *v = blob + off + 6;
            return (uint32_t)v[0] | ((uint32_t)v[1] << 8)
                | ((uint32_t)v[2] << 16) | ((uint32_t)v[3] << 24);
        }
        off += 6 + l;
    }
    DIE("spec tag 0x%04x not found", want);
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        DIE("usage: %s emit|dump <so> <seed> <n_envs> <T> <out>",
            argv[0]);
    }
    const int emit = strcmp(argv[1], "emit") == 0;
    if (!emit && strcmp(argv[1], "dump") != 0) {
        DIE("mode must be emit or dump, not %s", argv[1]);
    }
    const char *so_path = argv[2];
    uint64_t seed = strtoull(argv[3], NULL, 10);
    uint32_t n_envs = (uint32_t)strtoul(argv[4], NULL, 10);
    uint32_t T = (uint32_t)strtoul(argv[5], NULL, 10);
    const char *out_path = argv[6];

    void *so = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
    if (!so) DIE("dlopen %s: %s", so_path, dlerror());

    Surface s;
    /* dlsym returns void *; ISO-C-clean conversion via memcpy. */
#define RESOLVE(field, name) do { void *p_ = sym_(so, name); \
    memcpy(&s.field, &p_, sizeof p_); } while (0)
    RESOLVE(abi_version, "k26rl_abi_version");
    RESOLVE(create, "k26rl_env_create");
    RESOLVE(output, "k26rl_env_output");
    RESOLVE(step, "k26rl_env_step");
    RESOLVE(obs, "k26rl_env_obs");
    RESOLVE(reward, "k26rl_env_reward");
    RESOLVE(flags, "k26rl_env_flags");
    RESOLVE(spec, "k26rl_env_spec");
    RESOLVE(destroy, "k26rl_env_destroy");
#undef RESOLVE
    if ((s.abi_version() >> 16) != 1u) DIE("ABI major is not 1");

    K26RlEnv *env = NULL;
    if (s.create(seed, n_envs, &env) != K26RL_OK) DIE("create failed");

    int32_t spec_len = s.spec(env, NULL, 0);
    if (spec_len <= 0) DIE("spec sizing failed (%d)", spec_len);
    uint8_t *blob = malloc((size_t)spec_len);
    if (!blob) DIE("out of memory");
    if (s.spec(env, blob, (uint32_t)spec_len) != spec_len) {
        DIE("spec fetch failed");
    }
    uint32_t obs_total = spec_u32_(blob, (uint32_t)spec_len,
                                   K26RL_TAG_OBS_TOTAL);
    uint32_t act_total = spec_u32_(blob, (uint32_t)spec_len,
                                   K26RL_TAG_ACT_TOTAL);
    free(blob);
    if (act_total != 2u) DIE("fixture act_total is %u, expected 2",
                             act_total);

    FILE *dump = NULL;
    if (emit) {
        if (s.output(env, out_path) != K26RL_OK) {
            DIE("output enable failed for %s", out_path);
        }
    } else {
        dump = fopen(out_path, "wb");
        if (!dump) DIE("cannot open %s", out_path);
    }

    double *act = malloc(sizeof(double) * n_envs * act_total);
    double *ob = malloc(sizeof(double) * n_envs * obs_total);
    double *rew = malloc(sizeof(double) * n_envs);
    uint32_t *fl = malloc(sizeof(uint32_t) * n_envs);
    if (!act || !ob || !rew || !fl) DIE("out of memory");

    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t e = 0; e < n_envs; e++) {
            act[e * act_total + 0] = act_thrust_(t, e);
            act[e * act_total + 1] = act_gear_(t, e);
        }
        if (s.step(env, act) != K26RL_OK) DIE("step %u failed", t);
        if (dump) {
            if (s.obs(env, ob) != K26RL_OK) DIE("obs read failed");
            if (s.reward(env, rew) != K26RL_OK) DIE("reward read failed");
            if (s.flags(env, fl) != K26RL_OK) DIE("flags read failed");
            if (fwrite(ob, sizeof(double), n_envs * obs_total, dump)
                    != n_envs * obs_total ||
                fwrite(rew, sizeof(double), n_envs, dump) != n_envs ||
                fwrite(fl, sizeof(uint32_t), n_envs, dump) != n_envs) {
                DIE("dump write failed");
            }
        }
    }

    free(act);
    free(ob);
    free(rew);
    free(fl);
    if (dump && fclose(dump) != 0) DIE("dump close failed");
    s.destroy(env);
    dlclose(so);
    return 0;
}
