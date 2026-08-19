/* k26rl_rtdrive - run a trained policy over a compiled world against
 * the wall clock, at a declared rate, and report what happened.
 *
 * The stepping driver answers what a policy does. This one answers
 * whether it would run on the vehicle's computer at rate and hold: the
 * simulation advances against a monotonic clock, each control period
 * asks the policy for an action, and an action that has not arrived by
 * the period's end leaves the previous command in force and counts a
 * miss. The loop itself is libk26rl's; this program is the part that
 * opens an artifact, loads a policy, checks the two against each other
 * and prints the result.
 *
 * It asks the operating system for nothing: no pinned core, no raised
 * priority, no locked pages, no particular kernel. What it reports is
 * therefore what an ordinary scheduler gave it, and the machine's load
 * average is printed beside the numbers because a latency tail
 * measured on a busy machine belongs partly to the scheduler.
 *
 * Usage:
 *   k26rl_rtdrive --artifact PATH.rlenv.so --policy PATH.k26pol
 *                 [--rate HZ] [--periods N] [--seed N]
 *                 [--tolerance FRACTION] [--episode PATH]
 *                 [--actions PATH]
 *
 * The rate defaults to the reciprocal of the world's own declared
 * control interval, which is the rate at which one period of wall
 * clock is one period of simulated time. Naming another rate runs the
 * world faster or slower than itself, and the reported real-time
 * factor says which.
 *
 * --episode writes the episode record the stepping driver writes, so a
 * wall-clock run can be drawn and re-simulated like any other run.
 * --actions writes the action stream actually applied, held actions
 * included, as raw binary64 in period order: replaying that stream
 * through the stepping driver reproduces the run's physics bit for
 * bit, which is what makes a wall-clock result reproducible.
 *
 * Exit codes: 0 the run completed, 1 a usage, loading or run error.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k26rl_env.h"
#include "k26rl_policy.h"
#include "k26rl_realtime.h"

#define DEFAULT_PERIODS   4000u
#define DEFAULT_SEED      1u
#define DEFAULT_TOLERANCE 0.02

/* The artifact's exports, resolved by name. A consumer of this surface
 * probes symbols rather than linking them, which is what lets one
 * program drive any world the compiler emitted. */
typedef struct {
    uint32_t    (*abi_version)(void);
    K26RlStatus (*create)(uint64_t, uint32_t, K26RlEnv **);
    K26RlStatus (*output)(K26RlEnv *, const char *);
    K26RlStatus (*step)(K26RlEnv *, const double *);
    K26RlStatus (*obs)(const K26RlEnv *, double *);
    int32_t     (*spec)(const K26RlEnv *, uint8_t *, uint32_t);
    const char *(*status_str)(K26RlStatus);
    void        (*destroy)(K26RlEnv *);
} Surface;

static int resolve_(void *so, Surface *s)
{
    int ok = 1;

#define RESOLVE_(field, name) do { \
        void *p_ = dlsym(so, name); \
        if (!p_) { \
            fprintf(stderr, "k26rl_rtdrive: %s: %s\n", name, dlerror()); \
            ok = 0; \
        } else { \
            memcpy(&s->field, &p_, sizeof p_); \
        } \
    } while (0)
    RESOLVE_(abi_version, "k26rl_abi_version");
    RESOLVE_(create,      "k26rl_env_create");
    RESOLVE_(output,      "k26rl_env_output");
    RESOLVE_(step,        "k26rl_env_step");
    RESOLVE_(obs,         "k26rl_env_obs");
    RESOLVE_(spec,        "k26rl_env_spec");
    RESOLVE_(status_str,  "k26rl_status_str");
    RESOLVE_(destroy,     "k26rl_env_destroy");
#undef RESOLVE_
    return ok;
}

/* The spec fields this program needs, read out of the TLV blob by
 * length so an unknown tag is skipped rather than misread. */
typedef struct {
    uint32_t obs_total, act_total, agent_count, horizon;
    double   control_dt;
    int      saw_control_dt;
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

static int parse_spec_(const uint8_t *blob, uint32_t len, SpecView *v)
{
    uint32_t off = 0;

    memset(v, 0, sizeof *v);
    while (off + 6u <= len) {
        uint16_t tag = get_u16_(blob + off);
        uint32_t l = get_u32_(blob + off + 2);
        const uint8_t *val = blob + off + 6;

        if ((uint64_t)off + 6u + l > len)
            return 0;
        switch (tag) {
        case K26RL_TAG_OBS_TOTAL:   v->obs_total = get_u32_(val); break;
        case K26RL_TAG_ACT_TOTAL:   v->act_total = get_u32_(val); break;
        case K26RL_TAG_AGENT_COUNT: v->agent_count = get_u32_(val); break;
        case K26RL_TAG_HORIZON:     v->horizon = get_u32_(val); break;
        case K26RL_TAG_CONTROL_DT: {
            uint64_t bits = (uint64_t)get_u32_(val) |
                            ((uint64_t)get_u32_(val + 4) << 32);

            memcpy(&v->control_dt, &bits, sizeof v->control_dt);
            v->saw_control_dt = 1;
            break;
        }
        default: break;
        }
        off += 6u + l;
    }
    return off == len;
}

/* What the driver calls. The policy writes its own action slice into
 * the whole environment's action vector, leaving every other channel
 * as the caller had it, so a world of several agents can be driven by
 * one policy per agent with the same loop. */
typedef struct {
    const Surface     *s;
    K26RlEnv          *env;
    const K26RlPolicy *policy;
    K26RlStatus        last_status;
} Context;

static int decide_(void *context, const double *obs, double *action)
{
    Context *c = (Context *)context;

    return k26rl_policy_act_env(c->policy, obs, action) == K26RL_POLICY_OK
           ? 0 : 1;
}

static int advance_(void *context, const double *action, double *obs)
{
    Context *c = (Context *)context;

    c->last_status = c->s->step(c->env, action);
    if (c->last_status != K26RL_OK)
        return 1;
    c->last_status = c->s->obs(c->env, obs);
    return c->last_status == K26RL_OK ? 0 : 1;
}

static void load_average_(char *out, size_t cap)
{
    FILE *f = fopen("/proc/loadavg", "r");
    double a = 0.0, b = 0.0, c = 0.0;

    if (f && fscanf(f, "%lf %lf %lf", &a, &b, &c) == 3)
        snprintf(out, cap, "%.2f %.2f %.2f", a, b, c);
    else
        snprintf(out, cap, "unavailable");
    if (f)
        fclose(f);
}

static void usage_(void)
{
    fprintf(stderr,
        "usage: k26rl_rtdrive --artifact PATH.rlenv.so "
        "--policy PATH.k26pol\n"
        "                     [--rate HZ] [--periods N] [--seed N]\n"
        "                     [--tolerance FRACTION] [--episode PATH]\n"
        "                     [--actions PATH]\n");
}

int main(int argc, char **argv)
{
    const char *artifact = NULL, *policy_path = NULL;
    const char *episode_path = NULL, *actions_path = NULL;
    double rate_hz = 0.0, tolerance = DEFAULT_TOLERANCE;
    unsigned long long periods = DEFAULT_PERIODS, seed = DEFAULT_SEED;
    void *so;
    Surface s;
    SpecView v;
    Context context;
    K26RlEnv *env = NULL;
    K26RlPolicy *policy = NULL;
    K26RlPolicyStatus pstatus;
    K26RlRtConfig config;
    K26RlRtHooks hooks;
    K26RlRtBuffers buffers;
    K26RlRtReport report;
    K26RlRtStatus rc;
    uint8_t *spec;
    int32_t spec_len;
    char detail[256], load[64];
    int i;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has_value = (i + 1 < argc);

#define TAKE_(name, target) \
        if (strcmp(a, name) == 0) { \
            if (!has_value) { usage_(); return 1; } \
            target = argv[++i]; \
            continue; \
        }
        TAKE_("--artifact", artifact)
        TAKE_("--policy", policy_path)
        TAKE_("--episode", episode_path)
        TAKE_("--actions", actions_path)
#undef TAKE_
        if (strcmp(a, "--rate") == 0 && has_value) {
            rate_hz = strtod(argv[++i], NULL);
            continue;
        }
        if (strcmp(a, "--periods") == 0 && has_value) {
            periods = strtoull(argv[++i], NULL, 10);
            continue;
        }
        if (strcmp(a, "--seed") == 0 && has_value) {
            seed = strtoull(argv[++i], NULL, 10);
            continue;
        }
        if (strcmp(a, "--tolerance") == 0 && has_value) {
            tolerance = strtod(argv[++i], NULL);
            continue;
        }
        usage_();
        return 1;
    }
    if (!artifact || !policy_path) {
        usage_();
        return 1;
    }

    so = dlopen(artifact, RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        fprintf(stderr, "k26rl_rtdrive: %s\n", dlerror());
        return 1;
    }
    if (!resolve_(so, &s))
        return 1;

    /* One world, because there is one clock and one deadline. A handle
     * of many environments is how the stepping driver goes faster, and
     * going faster is not what this mode is for. */
    if (s.create((uint64_t)seed, 1u, &env) != K26RL_OK) {
        fprintf(stderr, "k26rl_rtdrive: the world refused to start\n");
        return 1;
    }
    spec_len = s.spec(env, NULL, 0);
    if (spec_len <= 0) {
        fprintf(stderr, "k26rl_rtdrive: the world published no spec\n");
        return 1;
    }
    spec = (uint8_t *)malloc((size_t)spec_len);
    if (!spec || s.spec(env, spec, (uint32_t)spec_len) != spec_len ||
        !parse_spec_(spec, (uint32_t)spec_len, &v)) {
        fprintf(stderr, "k26rl_rtdrive: the spec could not be read\n");
        return 1;
    }
    if (!v.saw_control_dt || !(v.control_dt > 0.0)) {
        fprintf(stderr, "k26rl_rtdrive: the world declares no control "
                        "interval\n");
        return 1;
    }

    pstatus = k26rl_policy_open(policy_path, &policy);
    if (pstatus != K26RL_POLICY_OK) {
        fprintf(stderr, "k26rl_rtdrive: %s: %s\n", policy_path,
                k26rl_policy_status_str(pstatus));
        return 1;
    }
    detail[0] = '\0';
    pstatus = k26rl_policy_check_spec(policy, spec, (uint32_t)spec_len,
                                      detail, sizeof detail);
    if (pstatus != K26RL_POLICY_OK) {
        fprintf(stderr, "k26rl_rtdrive: the policy does not fit this "
                        "world: %s: %s\n",
                k26rl_policy_status_str(pstatus), detail);
        return 1;
    }

    /* Real time in the ordinary sense: one period of wall clock for one
     * period of simulated time, unless the caller asked for another
     * rate on purpose. */
    if (!(rate_hz > 0.0))
        rate_hz = 1.0 / v.control_dt;

    if (episode_path && s.output(env, episode_path) != K26RL_OK) {
        fprintf(stderr, "k26rl_rtdrive: %s could not be opened for the "
                        "episode record\n", episode_path);
        return 1;
    }

    memset(&buffers, 0, sizeof buffers);
    buffers.obs = (double *)calloc(v.obs_total, sizeof(double));
    buffers.candidate = (double *)calloc(v.act_total, sizeof(double));
    buffers.applied = (double *)calloc(v.act_total, sizeof(double));
    buffers.action_log = (double *)calloc((size_t)periods * v.act_total,
                                          sizeof(double));
    buffers.latency_us = (double *)calloc((size_t)periods, sizeof(double));
    buffers.lateness_us = (double *)calloc((size_t)periods, sizeof(double));
    buffers.scratch = (double *)calloc((size_t)periods, sizeof(double));
    buffers.missed = (uint8_t *)calloc((size_t)periods, 1u);
    if (!buffers.obs || !buffers.candidate || !buffers.applied ||
        !buffers.action_log || !buffers.latency_us ||
        !buffers.lateness_us || !buffers.scratch || !buffers.missed) {
        fprintf(stderr, "k26rl_rtdrive: out of memory for %llu periods\n",
                periods);
        return 1;
    }
    if (s.obs(env, buffers.obs) != K26RL_OK) {
        fprintf(stderr, "k26rl_rtdrive: the world published no initial "
                        "observation\n");
        return 1;
    }

    context.s = &s;
    context.env = env;
    context.policy = policy;
    context.last_status = K26RL_OK;
    hooks.decide = decide_;
    hooks.advance = advance_;
    hooks.context = &context;

    memset(&config, 0, sizeof config);
    config.rate_hz = rate_hz;
    config.control_dt = v.control_dt;
    config.periods = periods;
    config.obs_width = v.obs_total;
    config.act_width = v.act_total;
    config.rate_tolerance = tolerance;

    /* One evaluation before the first period. The first calls after a
     * load are the dear ones, because they page the weights in, and a
     * caller with a deadline pays that outside a period rather than
     * inside one. */
    (void)k26rl_policy_act_env(policy, buffers.obs, buffers.candidate);
    memset(buffers.candidate, 0, sizeof(double) * v.act_total);

    load_average_(load, sizeof load);
    rc = k26rl_rt_run(&config, &hooks, &buffers, &report);
    if (rc != K26RL_RT_OK) {
        fprintf(stderr, "k26rl_rtdrive: the run stopped after %llu "
                        "periods: %s",
                (unsigned long long)report.periods_run,
                k26rl_rt_status_str(rc));
        if (rc == K26RL_RT_E_ADVANCE) {
            fprintf(stderr, ": %s",
                    s.status_str(context.last_status));
        }
        fprintf(stderr, "\n");
    }

    printf("world             %s\n", artifact);
    printf("policy            %s\n", policy_path);
    printf("rate              %.6g Hz, period %llu ns\n", rate_hz,
           (unsigned long long)report.period_ns);
    printf("control interval  %.9g simulated s per period\n", v.control_dt);
    printf("periods run       %llu\n",
           (unsigned long long)report.periods_run);
    printf("periods missed    %llu (%.4f per cent)\n",
           (unsigned long long)report.periods_missed,
           report.periods_run
           ? 100.0 * (double)report.periods_missed /
             (double)report.periods_run : 0.0);
    printf("worst overrun     %.3f us\n", report.worst_overrun_us);
    printf("worst start lag   %.3f us\n", report.worst_start_lateness_us);
    printf("wall clock        %.6f s\n", report.wall_clock_s);
    printf("nominal duration  %.6f s\n", report.nominal_s);
    printf("simulated         %.6f s\n", report.simulated_s);
    printf("real-time factor  %.6f\n", report.real_time_factor);
    printf("rate held         %s\n", report.rate_held ? "yes" : "no");
    printf("decision latency  n %llu  min %.3f  p50 %.3f  p90 %.3f  "
           "p99 %.3f  p99.9 %.3f  max %.3f  (mean %.3f) us\n",
           (unsigned long long)report.decision.count,
           report.decision.min_us, report.decision.p50_us,
           report.decision.p90_us, report.decision.p99_us,
           report.decision.p999_us, report.decision.max_us,
           report.decision.mean_us);
    printf("period start lag  n %llu  min %.3f  p50 %.3f  p90 %.3f  "
           "p99 %.3f  p99.9 %.3f  max %.3f  (mean %.3f) us\n",
           (unsigned long long)report.start_lateness.count,
           report.start_lateness.min_us, report.start_lateness.p50_us,
           report.start_lateness.p90_us, report.start_lateness.p99_us,
           report.start_lateness.p999_us, report.start_lateness.max_us,
           report.start_lateness.mean_us);
    printf("load average      %s at the start of the run\n", load);
    printf("scheduling        no pinned core, no raised priority, no "
           "locked pages, no real-time kernel required\n");

    if (actions_path) {
        FILE *f = fopen(actions_path, "wb");
        size_t n = (size_t)report.periods_run * v.act_total;

        if (!f || fwrite(buffers.action_log, sizeof(double), n, f) != n ||
            fclose(f) != 0) {
            fprintf(stderr, "k26rl_rtdrive: %s could not be written\n",
                    actions_path);
            return 1;
        }
        printf("applied actions   %s, %llu periods of %u binary64 "
               "channels\n", actions_path,
               (unsigned long long)report.periods_run, v.act_total);
    }

    s.destroy(env);
    if (episode_path)
        printf("episode record    %s\n", episode_path);

    k26rl_policy_close(policy);
    free(buffers.obs);
    free(buffers.candidate);
    free(buffers.applied);
    free(buffers.action_log);
    free(buffers.latency_us);
    free(buffers.lateness_us);
    free(buffers.scratch);
    free(buffers.missed);
    free(spec);
    dlclose(so);
    return rc == K26RL_RT_OK ? 0 : 1;
}
