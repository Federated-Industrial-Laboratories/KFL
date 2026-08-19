/* realtime.c - the wall-clock driver declared in k26rl_realtime.h.
 *
 * The header states the loop, the miss rule, the replay property and
 * what the driver asks the operating system for; this file implements
 * exactly those and adds nothing of its own.
 *
 * Two disciplines govern it.
 *
 * The schedule is kept in integer nanoseconds, never in seconds as a
 * real. A run of a hundred thousand periods accumulates every rounding
 * a floating-point schedule would make, and the quantity being
 * measured is the difference between when a period was due and when it
 * happened, which is exactly the size the rounding would reach. The
 * period is the declared rate rounded once to the nanosecond and every
 * deadline after that is an exact integer multiple of it, so the
 * schedule a run is held against is a schedule that can be written
 * down.
 *
 * Nothing in the period loop allocates, opens a file or takes a lock.
 * That is not a performance preference: a decision path that reaches
 * an allocator has a latency tail belonging to the allocator, and this
 * file exists to measure a latency tail. Every buffer is the caller's
 * and every write into one is a copy of a fixed size the configuration
 * fixed before the first period.
 */
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "k26rl_realtime.h"

#define NS_PER_S  ((int64_t)1000000000)
#define NS_PER_US ((double)1000.0)

const char *k26rl_rt_status_str(K26RlRtStatus status)
{
    switch (status) {
    case K26RL_RT_OK:        return "ok";
    case K26RL_RT_E_NULL:    return "null pointer argument";
    case K26RL_RT_E_CONFIG:  return "a configured value is unusable";
    case K26RL_RT_E_CLOCK:   return "the monotonic clock refused";
    case K26RL_RT_E_DECIDE:  return "the decide hook reported failure";
    case K26RL_RT_E_ADVANCE: return "the advance hook reported failure";
    case K26RL_RT_E_SAMPLES: return "no samples to summarise";
    }
    return "unknown status";
}

/* CLOCK_MONOTONIC in whole nanoseconds. The clock is monotonic and
 * unaffected by anyone setting the system time, which a driver
 * measuring deadlines needs and CLOCK_REALTIME does not give. */
static int now_ns_(int64_t *out)
{
    struct timespec t;

    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
        return -1;
    *out = (int64_t)t.tv_sec * NS_PER_S + (int64_t)t.tv_nsec;
    return 0;
}

/* Sleep until an absolute time on the same clock the deadlines are
 * on, so a sleep cannot drift against them the way a relative sleep
 * would. A time already past returns at once, which is how a run that
 * has fallen behind keeps going without catching up by skipping. The
 * only failure worth retrying is an interruption. */
static void sleep_until_ns_(int64_t when)
{
    struct timespec t;

    t.tv_sec  = (time_t)(when / NS_PER_S);
    t.tv_nsec = (long)(when % NS_PER_S);
    for (;;) {
        int rc = clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL);

        if (rc != EINTR)
            return;
    }
}

static int compare_double_(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* The sample at a quantile of an already sorted array, by nearest
 * rank. A quantile deeper than one part in n resolves to the largest
 * sample; the header says so and the count travels with the summary
 * so a reader can see it. */
static double quantile_(const double *sorted, uint64_t n, double q)
{
    double pos = q * (double)(n - 1u) + 0.5;
    int64_t i = (int64_t)pos;

    if (i < 0)
        i = 0;
    if ((uint64_t)i >= n)
        i = (int64_t)n - 1;
    return sorted[i];
}

K26RlRtStatus k26rl_rt_summarise(const double *samples, uint64_t count,
                                 double *scratch, K26RlRtLatency *out)
{
    double sum = 0.0;
    uint64_t i;

    if (!samples || !scratch || !out)
        return K26RL_RT_E_NULL;
    if (count == 0u)
        return K26RL_RT_E_SAMPLES;

    memcpy(scratch, samples, sizeof(double) * (size_t)count);
    qsort(scratch, (size_t)count, sizeof(double), compare_double_);
    for (i = 0; i < count; i++)
        sum += samples[i];

    out->count   = count;
    out->min_us  = scratch[0];
    out->p50_us  = quantile_(scratch, count, 0.50);
    out->p90_us  = quantile_(scratch, count, 0.90);
    out->p99_us  = quantile_(scratch, count, 0.99);
    out->p999_us = quantile_(scratch, count, 0.999);
    out->max_us  = scratch[count - 1u];
    out->mean_us = sum / (double)count;
    return K26RL_RT_OK;
}

/* Fill in everything the report can carry once the loop has stopped,
 * however it stopped. A run cut short by a failing hook still reports
 * the periods it completed: the numbers are what they are, and a
 * report that went blank on an error would throw away the evidence of
 * what the run was doing when it failed. */
static void finish_report_(K26RlRtReport *out, const K26RlRtConfig *config,
                           const K26RlRtBuffers *buffers,
                           int64_t started_ns, int64_t ended_ns)
{
    out->wall_clock_s = (double)(ended_ns - started_ns) / (double)NS_PER_S;
    out->nominal_s = (double)(int64_t)out->periods_run *
                     (double)(int64_t)out->period_ns / (double)NS_PER_S;
    out->simulated_s = (double)(int64_t)out->periods_run * config->control_dt;
    out->real_time_factor = (out->wall_clock_s > 0.0)
                            ? out->simulated_s / out->wall_clock_s : 0.0;
    out->rate_held = (out->periods_run > 0u) &&
                     (out->wall_clock_s <=
                      out->nominal_s * (1.0 + config->rate_tolerance));
    if (out->periods_run > 0u) {
        /* One scratch buffer serves both summaries: each sorts a copy
         * and is finished with it before the next begins, so a second
         * buffer would buy nothing. */
        (void)k26rl_rt_summarise(buffers->latency_us, out->periods_run,
                                 buffers->scratch, &out->decision);
        (void)k26rl_rt_summarise(buffers->lateness_us, out->periods_run,
                                 buffers->scratch, &out->start_lateness);
    }
}

K26RlRtStatus k26rl_rt_run(const K26RlRtConfig *config,
                           const K26RlRtHooks *hooks,
                           const K26RlRtBuffers *buffers,
                           K26RlRtReport *out)
{
    size_t act_bytes;
    int64_t period_ns, started_ns, ended_ns = 0;
    uint64_t p;

    if (!config || !hooks || !buffers || !out)
        return K26RL_RT_E_NULL;

    /* Cleared before anything can refuse, not after the last check
     * passes. A caller prints a report, and a refused run that left one
     * untouched would have it printing whatever was on its stack.
     * Zeroed, a refusal reads as a run of no periods, which is what it
     * was. */
    memset(out, 0, sizeof *out);

    if (!hooks->decide || !hooks->advance)
        return K26RL_RT_E_NULL;
    if (!buffers->obs || !buffers->candidate || !buffers->applied ||
        !buffers->action_log || !buffers->latency_us ||
        !buffers->lateness_us || !buffers->scratch || !buffers->missed)
        return K26RL_RT_E_NULL;

    /* The period is rounded once, here, and the report carries it, so
     * a rate that is not a whole number of nanoseconds is measured
     * against the schedule it actually ran rather than the one it
     * asked for. */
    if (!(config->rate_hz > 0.0) || !(config->rate_hz <= 1.0e9))
        return K26RL_RT_E_CONFIG;
    period_ns = (int64_t)((double)NS_PER_S / config->rate_hz + 0.5);
    if (period_ns < 1)
        return K26RL_RT_E_CONFIG;
    if (config->periods == 0u || config->act_width == 0u ||
        config->obs_width == 0u)
        return K26RL_RT_E_CONFIG;
    if (!(config->rate_tolerance >= 0.0))
        return K26RL_RT_E_CONFIG;

    out->period_ns = (uint64_t)period_ns;
    act_bytes = sizeof(double) * (size_t)config->act_width;
    memset(buffers->missed, 0, (size_t)config->periods);

    if (now_ns_(&started_ns) != 0)
        return K26RL_RT_E_CLOCK;

    for (p = 0; p < config->periods; p++) {
        int64_t nominal_start = started_ns + (int64_t)p * period_ns;
        int64_t deadline = nominal_start + period_ns;
        int64_t begun, answered;
        double lateness_us;

        if (now_ns_(&begun) != 0)
            return K26RL_RT_E_CLOCK;

        /* A period that begins after the schedule said it would is
         * late, and the lateness is reported apart from any overrun so
         * that a slow decision and an unrecovered slip are not read as
         * one thing. */
        lateness_us = (double)(begun - nominal_start) / NS_PER_US;
        buffers->lateness_us[p] = lateness_us;
        if (lateness_us > out->worst_start_lateness_us)
            out->worst_start_lateness_us = lateness_us;

        if (hooks->decide(hooks->context, buffers->obs,
                          buffers->candidate) != 0) {
            if (now_ns_(&ended_ns) != 0)
                ended_ns = started_ns;
            finish_report_(out, config, buffers, started_ns, ended_ns);
            return K26RL_RT_E_DECIDE;
        }
        if (now_ns_(&answered) != 0)
            return K26RL_RT_E_CLOCK;

        buffers->latency_us[p] = (double)(answered - begun) / NS_PER_US;

        /* The deadline is the period's own end in absolute time, not a
         * budget counted from when the decision happened to start. An
         * action that arrives after it is discarded and the command
         * register keeps what it held. */
        if (answered <= deadline) {
            memcpy(buffers->applied, buffers->candidate, act_bytes);
        } else {
            double overrun_us = (double)(answered - deadline) / NS_PER_US;

            buffers->missed[p] = 1u;
            out->periods_missed++;
            if (overrun_us > out->worst_overrun_us)
                out->worst_overrun_us = overrun_us;
        }

        memcpy(buffers->action_log + (size_t)p * config->act_width,
               buffers->applied, act_bytes);

        /* The period is counted after its advance returns, not before,
         * so the simulated duration a report carries is a duration the
         * world computed. A run cut short by a failing advance reports
         * the periods that completed; the miss counters may include the
         * period that failed, whose action was applied and whose world
         * did not move. */
        if (hooks->advance(hooks->context, buffers->applied,
                           buffers->obs) != 0) {
            if (now_ns_(&ended_ns) != 0)
                ended_ns = started_ns;
            finish_report_(out, config, buffers, started_ns, ended_ns);
            return K26RL_RT_E_ADVANCE;
        }
        out->periods_run = p + 1u;

        /* The wait belongs to the run: a period that finished early
         * still occupies its whole period, which is what makes the
         * wall-clock duration comparable with the simulated one. */
        sleep_until_ns_(started_ns + (int64_t)(p + 1u) * period_ns);
    }

    if (now_ns_(&ended_ns) != 0)
        return K26RL_RT_E_CLOCK;
    finish_report_(out, config, buffers, started_ns, ended_ns);
    return K26RL_RT_OK;
}
