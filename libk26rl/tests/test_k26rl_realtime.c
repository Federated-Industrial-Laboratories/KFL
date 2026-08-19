/* test_k26rl_realtime.c: the wall-clock driver's own surface, with no
 * world under it.
 *
 * The gate that drives a compiled world holds the properties that need
 * one: the replay property, what a miss does to real physics, and the
 * latency of a real policy. What it cannot reach are the paths a
 * healthy run never takes, and a status a nothing can return is a
 * status no gate can fail on. This file supplies stub hooks instead of
 * a world, so every refusal, every failure path and the arithmetic of
 * the schedule are exercised directly and quickly.
 *
 * Arms:
 *   1. Refusals. Every null argument, every unusable configured value,
 *      each reaching its own code rather than a general one.
 *   2. The schedule. The period is the declared rate rounded once to
 *      the nanosecond, and the report carries the rounded value, so a
 *      rate that is not a whole number of nanoseconds is measured
 *      against the schedule that ran.
 *   3. A run that meets every deadline: the action stream is what the
 *      decide hook produced, the observation is threaded from each
 *      advance into the next decision, and nothing is recorded as
 *      missed.
 *   4. A run that misses every deadline, from a decide hook made
 *      slower than the period: every period is counted, every one
 *      holds the register it was given, and the run reports that it
 *      did not hold rate.
 *   5. A hook that fails. The run stops, names which hook failed, and
 *      reports the periods that completed rather than going blank.
 *   6. The summary, against a distribution whose quantiles are known
 *      by construction, and its refusals.
 *   7. The status registry: every code decodes to its own string, and a
 *      value the registry does not carry decodes rather than failing.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k26rl_realtime.h"

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int g_arms;

#define PERIODS   24u
#define OBS_WIDTH 3u
#define ACT_WIDTH 2u

static double  g_obs[OBS_WIDTH];
static double  g_candidate[ACT_WIDTH];
static double  g_applied[ACT_WIDTH];
static double  g_action_log[PERIODS * ACT_WIDTH];
static double  g_latency[PERIODS];
static double  g_lateness[PERIODS];
static double  g_scratch[PERIODS];
static uint8_t g_missed[PERIODS];

static const double PRELOAD[ACT_WIDTH] = { 0.25, -0.75 };

/* A stub world. The observation it publishes is a function of the
 * period alone, and the action the decision returns is a function of
 * the observation, so what should have been applied on any period can
 * be recomputed here and compared. */
typedef struct {
    uint64_t period;
    uint64_t decide_calls, advance_calls;
    uint64_t fail_decide_at, fail_advance_at;   /* PERIODS means never */
    double   stall_us;
} Stub;

static double stub_obs_(uint64_t period, uint32_t channel)
{
    return (double)period * 0.5 + (double)channel;
}

static double stub_action_(const double *obs, uint32_t channel)
{
    return obs[channel] * 0.125 + (double)channel;
}

static void busy_spin_us_(double us)
{
    struct timespec a, b;

    clock_gettime(CLOCK_MONOTONIC, &a);
    for (;;) {
        double elapsed;

        clock_gettime(CLOCK_MONOTONIC, &b);
        elapsed = (double)(b.tv_sec - a.tv_sec) * 1e6 +
                  (double)(b.tv_nsec - a.tv_nsec) * 1e-3;
        if (elapsed >= us)
            return;
    }
}

static int stub_decide_(void *context, const double *obs, double *action)
{
    Stub *s = (Stub *)context;
    uint32_t i;

    s->decide_calls++;
    if (s->period == s->fail_decide_at)
        return 1;
    if (s->stall_us > 0.0)
        busy_spin_us_(s->stall_us);
    for (i = 0; i < ACT_WIDTH; i++)
        action[i] = stub_action_(obs, i);
    return 0;
}

static int stub_advance_(void *context, const double *action, double *obs)
{
    Stub *s = (Stub *)context;
    uint32_t i;

    (void)action;
    s->advance_calls++;
    if (s->period == s->fail_advance_at)
        return 1;
    s->period++;
    for (i = 0; i < OBS_WIDTH; i++)
        obs[i] = stub_obs_(s->period, i);
    return 0;
}

static void fill_(K26RlRtConfig *config, K26RlRtHooks *hooks,
                  K26RlRtBuffers *buffers, Stub *stub, double rate_hz,
                  uint64_t periods)
{
    uint32_t i;

    memset(stub, 0, sizeof *stub);
    stub->fail_decide_at = periods;
    stub->fail_advance_at = periods;

    memset(config, 0, sizeof *config);
    config->rate_hz = rate_hz;
    config->control_dt = 1.0 / rate_hz;
    config->periods = periods;
    config->obs_width = OBS_WIDTH;
    config->act_width = ACT_WIDTH;
    config->rate_tolerance = 0.05;

    hooks->decide = stub_decide_;
    hooks->advance = stub_advance_;
    hooks->context = stub;

    for (i = 0; i < OBS_WIDTH; i++)
        g_obs[i] = stub_obs_(0u, i);
    memcpy(g_applied, PRELOAD, sizeof g_applied);
    memcpy(g_candidate, PRELOAD, sizeof g_candidate);
    memset(g_action_log, 0, sizeof g_action_log);
    memset(g_latency, 0, sizeof g_latency);
    memset(g_lateness, 0, sizeof g_lateness);
    memset(g_missed, 0, sizeof g_missed);

    buffers->obs = g_obs;
    buffers->candidate = g_candidate;
    buffers->applied = g_applied;
    buffers->action_log = g_action_log;
    buffers->latency_us = g_latency;
    buffers->lateness_us = g_lateness;
    buffers->scratch = g_scratch;
    buffers->missed = g_missed;
}

/* ---- arm 1: refusals ------------------------------------------------ */

static void arm_refusals_(void)
{
    K26RlRtConfig config;
    K26RlRtHooks hooks;
    K26RlRtBuffers buffers, spoiled;
    K26RlRtReport report;
    Stub stub;

    fill_(&config, &hooks, &buffers, &stub, 1000.0, 4u);

    ASSERT(k26rl_rt_run(NULL, &hooks, &buffers, &report) ==
           K26RL_RT_E_NULL);
    ASSERT(k26rl_rt_run(&config, NULL, &buffers, &report) ==
           K26RL_RT_E_NULL);
    ASSERT(k26rl_rt_run(&config, &hooks, NULL, &report) ==
           K26RL_RT_E_NULL);
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, NULL) ==
           K26RL_RT_E_NULL);
    {
        K26RlRtHooks half = hooks;

        half.decide = NULL;
        ASSERT(k26rl_rt_run(&config, &half, &buffers, &report) ==
               K26RL_RT_E_NULL);
        half = hooks;
        half.advance = NULL;
        ASSERT(k26rl_rt_run(&config, &half, &buffers, &report) ==
               K26RL_RT_E_NULL);
    }

    /* Every buffer in turn, named one at a time rather than walked as
     * an array: they are separate members and arithmetic across them
     * would be arithmetic the language does not define. */
#define WITHOUT_(member) do { \
        spoiled = buffers; \
        spoiled.member = NULL; \
        ASSERT(k26rl_rt_run(&config, &hooks, &spoiled, &report) == \
               K26RL_RT_E_NULL); \
    } while (0)
    WITHOUT_(obs);
    WITHOUT_(candidate);
    WITHOUT_(applied);
    WITHOUT_(action_log);
    WITHOUT_(latency_us);
    WITHOUT_(lateness_us);
    WITHOUT_(scratch);
    WITHOUT_(missed);
#undef WITHOUT_

    {
        K26RlRtConfig bad;
        double zero = 0.0;

        bad = config; bad.rate_hz = 0.0;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.rate_hz = -400.0;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.rate_hz = zero / zero;         /* not a number */
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.rate_hz = 1.0e10;              /* below a ns */
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.periods = 0u;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.act_width = 0u;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.obs_width = 0u;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
        bad = config; bad.rate_tolerance = -0.1;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &report) ==
               K26RL_RT_E_CONFIG);
    }

    /* Nothing above ran a period, which is the point of a refusal. */
    ASSERT(stub.decide_calls == 0u && stub.advance_calls == 0u);

    /* And a refusal leaves a report a caller can print rather than one
     * holding whatever was on the stack. The check is made against a
     * report deliberately filled with a pattern first, so a run that
     * left it alone would be caught. */
    {
        K26RlRtConfig bad = config;
        K26RlRtReport dirty;
        unsigned char zeroed[sizeof dirty];

        memset(&dirty, 0x5A, sizeof dirty);
        memset(zeroed, 0, sizeof zeroed);
        bad.periods = 0u;
        ASSERT(k26rl_rt_run(&bad, &hooks, &buffers, &dirty) ==
               K26RL_RT_E_CONFIG);
        ASSERT(memcmp(&dirty, zeroed, sizeof dirty) == 0);
    }
    printf("  refusals: every null argument and every unusable "
           "configured value, none of them reaching a period\n");
    g_arms++;
}

/* ---- arm 2 and arm 3: the schedule, and a run that holds it --------- */

static void arm_schedule_(void)
{
    K26RlRtConfig config;
    K26RlRtHooks hooks;
    K26RlRtBuffers buffers;
    K26RlRtReport report;
    Stub stub;
    uint64_t p;
    uint32_t i;
    double expected[OBS_WIDTH];

    /* A rate that divides a second exactly, and one that does not. */
    fill_(&config, &hooks, &buffers, &stub, 400.0, 2u);
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, &report) ==
           K26RL_RT_OK);
    ASSERT(report.period_ns == 2500000u);

    fill_(&config, &hooks, &buffers, &stub, 333.0, 2u);
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, &report) ==
           K26RL_RT_OK);
    ASSERT(report.period_ns == 3003003u);
    printf("  the schedule: 400 Hz is a 2500000 ns period and 333 Hz a "
           "3003003 ns one, and the report carries the rounded value\n");
    g_arms++;

    /* A run that meets every deadline. Two thousand hertz leaves half a
     * millisecond for a decision that does almost nothing. */
    fill_(&config, &hooks, &buffers, &stub, 2000.0, PERIODS);
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, &report) ==
           K26RL_RT_OK);
    ASSERT(report.periods_run == PERIODS);
    ASSERT(report.periods_missed == 0u);
    ASSERT(report.worst_overrun_us == 0.0);
    ASSERT(report.rate_held == 1);
    ASSERT(stub.decide_calls == PERIODS && stub.advance_calls == PERIODS);
    ASSERT(report.decision.count == PERIODS);

    /* The applied stream is what the decisions produced, and each
     * decision saw the observation the advance before it wrote. */
    for (p = 0; p < PERIODS; p++) {
        for (i = 0; i < OBS_WIDTH; i++)
            expected[i] = stub_obs_(p, i);
        for (i = 0; i < ACT_WIDTH; i++) {
            ASSERT(g_action_log[p * ACT_WIDTH + i] ==
                   stub_action_(expected, i));
        }
        ASSERT(g_missed[p] == 0u);
    }
    /* The register the run was handed is gone, because every period
     * wrote over it. */
    ASSERT(memcmp(g_action_log, PRELOAD, sizeof PRELOAD) != 0);
    printf("  a run that met every deadline applied the action every "
           "decision produced, over the observation the advance before "
           "it wrote\n");
    g_arms++;
}

/* ---- arm 4: a run that misses every deadline ------------------------ */

static void arm_all_missed_(void)
{
    K26RlRtConfig config;
    K26RlRtHooks hooks;
    K26RlRtBuffers buffers;
    K26RlRtReport report;
    Stub stub;
    uint64_t p;

    /* Two hundred hertz is a five millisecond period; the decision is
     * made to take eight. */
    fill_(&config, &hooks, &buffers, &stub, 200.0, 8u);
    stub.stall_us = 8000.0;
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, &report) ==
           K26RL_RT_OK);

    ASSERT(report.periods_run == 8u);
    ASSERT(report.periods_missed == 8u);
    ASSERT(report.worst_overrun_us > 2000.0);
    ASSERT(report.rate_held == 0);
    ASSERT(report.wall_clock_s > report.nominal_s);
    ASSERT(report.real_time_factor < 1.0);
    ASSERT(report.decision.p50_us > 7000.0);
    ASSERT(report.start_lateness.max_us > 1000.0);

    /* Every period held the register it was handed, because no decision
     * ever arrived in time to write to it. */
    for (p = 0; p < 8u; p++) {
        ASSERT(g_missed[p] == 1u);
        ASSERT(memcmp(g_action_log + p * ACT_WIDTH, PRELOAD,
                      sizeof PRELOAD) == 0);
    }
    printf("  a decision made slower than its period misses every one, "
           "holds the register it was handed, and the run says it did "
           "not hold rate\n");
    g_arms++;
}

/* ---- arm 5: a hook that fails --------------------------------------- */

static void arm_hook_failure_(void)
{
    K26RlRtConfig config;
    K26RlRtHooks hooks;
    K26RlRtBuffers buffers;
    K26RlRtReport report;
    Stub stub;

    fill_(&config, &hooks, &buffers, &stub, 2000.0, PERIODS);
    stub.fail_decide_at = 5u;
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, &report) ==
           K26RL_RT_E_DECIDE);
    ASSERT(report.periods_run == 5u);
    ASSERT(report.decision.count == 5u);
    ASSERT(stub.advance_calls == 5u);

    fill_(&config, &hooks, &buffers, &stub, 2000.0, PERIODS);
    stub.fail_advance_at = 7u;
    ASSERT(k26rl_rt_run(&config, &hooks, &buffers, &report) ==
           K26RL_RT_E_ADVANCE);
    ASSERT(report.periods_run == 7u);
    ASSERT(report.decision.count == 7u);
    ASSERT(stub.decide_calls == 8u);
    printf("  a failing hook stops the run, names which one failed, and "
           "reports the periods that completed\n");
    g_arms++;
}

/* ---- arm 6: the summary --------------------------------------------- */

static void arm_summary_(void)
{
    enum { N = 1000 };
    static double samples[N], scratch[N];
    K26RlRtLatency d;
    int i;

    for (i = 0; i < N - 1; i++)
        samples[i] = 4.0;
    samples[N - 1] = 900.0;

    ASSERT(k26rl_rt_summarise(samples, (uint64_t)N, scratch, &d) ==
           K26RL_RT_OK);
    ASSERT(d.count == (uint64_t)N);
    ASSERT(d.min_us == 4.0 && d.p50_us == 4.0 && d.p90_us == 4.0);
    ASSERT(d.p99_us == 4.0 && d.p999_us == 4.0);
    ASSERT(d.max_us == 900.0);
    ASSERT(d.mean_us == ((double)(N - 1) * 4.0 + 900.0) / (double)N);

    /* The samples are left in the order they were measured: a caller
     * that wants them in time order still has them. */
    ASSERT(samples[N - 1] == 900.0 && samples[0] == 4.0);

    ASSERT(k26rl_rt_summarise(NULL, (uint64_t)N, scratch, &d) ==
           K26RL_RT_E_NULL);
    ASSERT(k26rl_rt_summarise(samples, (uint64_t)N, NULL, &d) ==
           K26RL_RT_E_NULL);
    ASSERT(k26rl_rt_summarise(samples, (uint64_t)N, scratch, NULL) ==
           K26RL_RT_E_NULL);
    ASSERT(k26rl_rt_summarise(samples, 0u, scratch, &d) ==
           K26RL_RT_E_SAMPLES);

    /* One sample is a distribution too, and every quantile of it is
     * that sample. */
    ASSERT(k26rl_rt_summarise(samples, 1u, scratch, &d) == K26RL_RT_OK);
    ASSERT(d.min_us == 4.0 && d.max_us == 4.0 && d.mean_us == 4.0);
    printf("  the summary of a known distribution, and its refusals\n");
    g_arms++;
}

/* ---- arm 7: the status registry ------------------------------------- */

static void arm_status_(void)
{
    static const K26RlRtStatus CODES[] = {
        K26RL_RT_OK, K26RL_RT_E_NULL, K26RL_RT_E_CONFIG, K26RL_RT_E_CLOCK,
        K26RL_RT_E_DECIDE, K26RL_RT_E_ADVANCE, K26RL_RT_E_SAMPLES
    };
    size_t n = sizeof CODES / sizeof CODES[0];
    size_t i, j;

    for (i = 0; i < n; i++) {
        ASSERT(k26rl_rt_status_str(CODES[i]) != NULL);
        ASSERT(k26rl_rt_status_str(CODES[i])[0] != '\0');
        for (j = i + 1; j < n; j++) {
            ASSERT(strcmp(k26rl_rt_status_str(CODES[i]),
                          k26rl_rt_status_str(CODES[j])) != 0);
        }
    }
    ASSERT(strcmp(k26rl_rt_status_str((K26RlRtStatus)999),
                  "unknown status") == 0);
    printf("  every status decodes to its own string, and a value the "
           "registry does not carry decodes rather than failing\n");
    g_arms++;
}

int main(void)
{
    arm_refusals_();
    arm_schedule_();
    arm_all_missed_();
    arm_hook_failure_();
    arm_summary_();
    arm_status_();
    printf("test_k26rl_realtime: %d arms passed\n", g_arms);
    return 0;
}
