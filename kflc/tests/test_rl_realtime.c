/* test_rl_realtime.c: the wall-clock driver over a compiled world.
 *
 * The stepping driver gives a controller all the time it wants. This
 * one runs a fixed control period against a monotonic clock, so a
 * decision that arrives late is late, and what the driver does then is
 * the whole of what this gate holds.
 *
 * Arms:
 *   1. Replay, on a run that missed nothing. A real world is driven at
 *      rate by a real policy fitted to that world's own declared
 *      slices, recording an episode file as the stepping driver does.
 *      The action stream the run applied is then replayed through the
 *      stepping driver into a second episode file, and the two files
 *      must be identical byte for byte. Compared as whole files rather
 *      than as a summary, because a summary can agree while the bytes
 *      behind it do not. A red control perturbs one action of the
 *      replayed stream and requires the comparison to fail, so a pass
 *      here is a comparison that ran rather than one that could not
 *      tell two files apart. The same run's wall clock is held against
 *      its simulated duration: a run at the world's own control rate
 *      that reports holding rate must have taken as long as the world
 *      it simulated.
 *   2. Replay, on a run that missed. The same world and policy with
 *      stalls injected past the deadline, so the recorded stream now
 *      contains held actions. It must replay to identical bytes, and
 *      the decisive red control replaces each held action with the late
 *      action the policy actually produced on that period and requires
 *      the bytes to differ. That is what says the held actions are in
 *      the stream and that the physics answered them.
 *   3. What a miss does. The same run's stream is checked against the
 *      hold rule: a missed period carries the previous period's action
 *      bit for bit, an unmissed period carries the action the policy
 *      produced, and no miss substitutes the declared default. The
 *      check is a function of its own, and it is shown rejecting a
 *      stream whose misses carry the default, one whose misses carry
 *      the late action, and one where an unmissed period carries
 *      something else, so a pass is a live check. The miss flags are
 *      held against the timings that produced them by the same method.
 *   4. A policy that cannot fit, which is not a stall but a network. A
 *      policy is built wide enough that one evaluation measurably
 *      exceeds the period, and it is checked against the world's spec
 *      like any other, so what drives this arm is a policy the world
 *      would accept and the clock will not. Every period must miss,
 *      the command register must keep what it held, and the run must
 *      report that it did not hold rate, with its wall clock past its
 *      simulated duration. A policy that never misses cannot fail this
 *      arm, which is why this one is built to miss.
 *   5. The distribution, and the worst. A long run at rate with a
 *      single slow period injected among fast ones: the reported
 *      maximum must be that period, the median must not move, and the
 *      mean must sit so far below the maximum that a summary quoting
 *      the mean alone would have hidden it. The summary is then held
 *      against a distribution constructed here whose every quantile is
 *      known, and two defective summaries, one reporting the mean as
 *      the maximum and one dropping its deepest samples, are required
 *      to be rejected by the same assertions.
 *   6. The stepping driver, unchanged. The wall-clock driver is in the
 *      archive the artifact links, so it is in the same binary as the
 *      stepping path. A stepped run in this process, made after the
 *      wall-clock driver has run here, must produce bytes identical to
 *      stepped runs in two fresh processes that never touched it. A red
 *      control at a different seed must differ.
 *
 * Timing numbers are reported with the machine's load average beside
 * them. A distribution measured on a loaded machine has a tail
 * belonging to the scheduler rather than to the code, and the tail is
 * the point, so the reader is given the number needed to weigh it. For
 * the same reason no absolute maximum is gated: a single preempted
 * call measures the scheduler on a machine this gate does not own. The
 * median is gated, the tail is gated as a multiple of the median, and
 * the maximum is printed as evidence.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <time.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"
#include "k26rl_policy.h"
#include "k26rl_realtime.h"

#define WORK_DIR "/tmp/kflc_rl_rt_test"

/* The world's control rate, and the rate every arm drives it at, so
 * one period of wall clock is one control period of simulated time and
 * the two durations in a report are directly comparable. */
#define RT_RATE_HZ    400.0
#define RT_CONTROL_DT 0.0025

/* The tolerance a run is allowed against its own schedule before it
 * stops claiming to have held rate. Sleeping to an absolute time
 * always overshoots a little, so a run's wall clock is always a little
 * past its nominal duration; this is far above that overshoot and far
 * below anything a missed deadline costs. */
#define RT_RATE_TOLERANCE 0.02

/* Arms run, so the closing line carries a count: a gate that stopped
 * running half of what it holds would otherwise print the line it
 * prints when it runs all of it. */
static int g_arms;

/* ---- the fixture ----------------------------------------------------
 *
 * Two action channels that both reach the dynamics and a reward that
 * reads one of them directly, so any change to any applied action
 * changes the recorded bytes. The horizon is short against the run
 * lengths below, so every run crosses episode boundaries and what
 * replays includes the boundary resets. */
static const char *const RT_KFL =
    "form RL_RT\n"
    "fn world rt_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=120.0 vel_y=normal(7350.0,10.0) vel_z=0.0\n"
    "    episode\n"
    "        control_dt 0.0025\n"
    "        horizon 150\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 10.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    action yaw box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "        craft.vel_z = craft.vel_z + yaw * 0.01\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range * 1.0e-6 + push + yaw * 0.5\n"
    "    end\n"
    "end\n"
    "end\n";

/* The fixture's declared action defaults: the command register's
 * contents before anything writes to it in ordinary use, and the value
 * a driver that substituted a default instead of holding would put on
 * a missed period. */
static const double RT_DEFAULTS[2] = { 0.0, 0.0 };

/* A distinctive command register the arms preload where the point is
 * to tell a hold from a substitution. It is not the declared default
 * and not anything the policy produces. */
static const double RT_PRELOAD[2] = { 0.6875, -0.4375 };

/* ---- machine load, reported beside every timing number -------------- */

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

/* ---- this file's own policy encoder ---------------------------------
 *
 * The gate writes the bytes itself rather than calling anything the
 * library ships, so a layout the reader gets wrong cannot be cancelled
 * by a writer that gets it wrong the same way. */

static unsigned char *g_pol;
static size_t g_at, g_cap;

static void emit_(const void *p, size_t n)
{
    if (g_at + n > g_cap) {
        g_cap = (g_at + n) * 2u + 65536u;
        g_pol = (unsigned char *)realloc(g_pol, g_cap);
        ASSERT(g_pol != NULL);
    }
    memcpy(g_pol + g_at, p, n);
    g_at += n;
}

static void emit_u16_(uint16_t v)
{
    unsigned char b[2];

    b[0] = (unsigned char)(v & 0xFFu);
    b[1] = (unsigned char)((v >> 8) & 0xFFu);
    emit_(b, sizeof b);
}

static void emit_u32_(uint32_t v)
{
    unsigned char b[4];
    int i;

    for (i = 0; i < 4; i++)
        b[i] = (unsigned char)((v >> (8 * i)) & 0xFFu);
    emit_(b, sizeof b);
}

static void emit_f64_(double v)
{
    uint64_t bits;
    unsigned char b[8];
    int i;

    memcpy(&bits, &v, sizeof bits);
    for (i = 0; i < 8; i++)
        b[i] = (unsigned char)((bits >> (8 * i)) & 0xFFu);
    emit_(b, sizeof b);
}

/* A weight from its position and a salt, in [-1, 1], scaled by the
 * layer's fan-in so a wide hidden layer does not saturate every unit.
 * Generated rather than tabulated because these arms measure timing
 * and the hold rule, not what a particular set of weights returns. */
static double weight_at_(uint64_t k, uint32_t fan_in)
{
    double raw = (double)(int64_t)((k * 2654435761u + 12345u) % 2003u) /
                 1001.0 - 1.0;

    return raw * 1.5 / sqrt((double)fan_in);
}

/* What the world declares about the agent's slices, taken from the
 * spec so nothing here hard-codes a channel count. */
typedef struct {
    uint32_t obs_total, act_total;
    uint32_t obs_offset, obs_width;
    uint32_t act_offset, act_width;
    uint32_t agent_count;
} Slices;

/* Build a policy over one agent's declared slices with `depth` hidden
 * layers of `hidden` units, standardised by the statistics the caller
 * measured from the world itself and clamped inside the world's own
 * action bounds. The caller frees the bytes. */
static unsigned char *build_policy_(const Slices *sl, uint32_t hidden,
                                    uint32_t depth, const double *mean,
                                    const double *variance, uint64_t salt,
                                    size_t *out_len, uint64_t *out_params)
{
    static const char PROVENANCE[] = "a policy fitted to a timed world";
    static const unsigned char ZEROS[K26RL_SHA256_BYTES] = { 0 };
    K26RlSha256 sha;
    unsigned char digest[K26RL_SHA256_BYTES];
    uint64_t params = 0, counter = salt;
    uint32_t layers = depth + 1u;
    uint32_t i, o, k, in_width;
    unsigned char *bytes;

    g_at = 0;
    emit_(K26RL_POLICY_MAGIC, 8);
    emit_u32_(K26RL_POLICY_FORMAT_VERSION);
    emit_u32_(K26RL_POLICY_HEADER_BYTES);
    emit_(ZEROS, K26RL_SHA256_BYTES);
    emit_u32_(K26RL_POLICY_FLAG_STANDARDISE | K26RL_POLICY_FLAG_CLAMP);
    emit_u32_(sl->agent_count);
    emit_u32_(0u);                      /* agent index */
    emit_u32_(sl->obs_total);
    emit_u32_(sl->act_total);
    emit_u32_(sl->obs_offset);
    emit_u32_(sl->obs_width);
    emit_u32_(sl->act_offset);
    emit_u32_(sl->act_width);
    emit_u32_(layers);
    emit_u32_((uint32_t)(sizeof PROVENANCE - 1u));
    emit_u32_(0u);                      /* reserved */
    emit_(PROVENANCE, sizeof PROVENANCE - 1u);

    in_width = sl->obs_width;
    for (i = 0; i < layers; i++) {
        uint32_t out_width = (i + 1u == layers) ? sl->act_width : hidden;
        uint16_t activation = (i + 1u == layers) ? K26RL_POLICY_ACT_IDENTITY
                                                 : K26RL_POLICY_ACT_TANH;

        emit_u32_(in_width);
        emit_u32_(out_width);
        emit_u16_(activation);
        emit_u16_(0u);
        for (o = 0; o < out_width; o++)
            for (k = 0; k < in_width; k++)
                emit_f64_(weight_at_(counter++, in_width));
        for (o = 0; o < out_width; o++)
            emit_f64_(weight_at_(counter++, in_width));
        params += (uint64_t)in_width * out_width + out_width;
        in_width = out_width;
    }

    for (i = 0; i < sl->obs_width; i++)
        emit_f64_(mean[i]);
    for (i = 0; i < sl->obs_width; i++)
        emit_f64_(variance[i]);
    emit_f64_(1e-8);                    /* epsilon */
    emit_f64_(5.0);                     /* clip */
    for (i = 0; i < sl->act_width; i++)
        emit_f64_(-0.9);                /* lower, inside the world's -1 */
    for (i = 0; i < sl->act_width; i++)
        emit_f64_(0.9);                 /* upper, inside the world's +1 */

    k26rl_sha256_init(&sha);
    k26rl_sha256_update(&sha, g_pol, (uint64_t)K26RL_POLICY_DIGEST_OFFSET);
    k26rl_sha256_update(&sha, ZEROS, (uint64_t)K26RL_SHA256_BYTES);
    k26rl_sha256_update(&sha,
                        g_pol + K26RL_POLICY_DIGEST_OFFSET +
                        K26RL_SHA256_BYTES,
                        (uint64_t)g_at - K26RL_POLICY_DIGEST_OFFSET -
                        K26RL_SHA256_BYTES);
    k26rl_sha256_final(&sha, digest);
    memcpy(g_pol + K26RL_POLICY_DIGEST_OFFSET, digest, K26RL_SHA256_BYTES);

    bytes = (unsigned char *)malloc(g_at);
    ASSERT(bytes != NULL);
    memcpy(bytes, g_pol, g_at);
    *out_len = g_at;
    if (out_params)
        *out_params = params;
    return bytes;
}

/* ---- the world under the driver -------------------------------------
 *
 * The hooks the driver calls. Everything the arms vary lives here: an
 * injected stall, and the log of what the policy produced on each
 * period, which is what the late actions of a missed period are. */
typedef struct {
    const RlSurface *s;
    K26RlEnv        *env;
    const K26RlPolicy *policy;
    uint32_t obs_total, act_total;
    uint64_t period;
    /* Stall the decision on every period whose index is congruent to
     * `stall_phase` modulo `stall_every`, or on the single period
     * `stall_once` when `stall_every` is zero. */
    uint64_t stall_every, stall_phase, stall_once;
    double   stall_us;
    double  *candidates;        /* periods * act_total, what decide made */
    uint64_t decide_calls, advance_calls;
} RtWorld;

/* Occupy the processor rather than yield it, so an injected stall is a
 * slow decision and not a sleeping one: a sleep would hand the period
 * back to the scheduler, which is not what a model that is thinking
 * too hard does. */
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

static int rt_decide_(void *context, const double *obs, double *action)
{
    RtWorld *w = (RtWorld *)context;
    int stall = 0;

    if (w->stall_every > 0u)
        stall = (w->period % w->stall_every) == w->stall_phase;
    else if (w->stall_us > 0.0)
        stall = (w->period == w->stall_once);
    if (stall)
        busy_spin_us_(w->stall_us);

    if (k26rl_policy_act_env(w->policy, obs, action) != K26RL_POLICY_OK)
        return 1;
    if (w->candidates) {
        memcpy(w->candidates + w->period * w->act_total, action,
               sizeof(double) * w->act_total);
    }
    w->decide_calls++;
    return 0;
}

static int rt_advance_(void *context, const double *action, double *obs)
{
    RtWorld *w = (RtWorld *)context;

    if (w->s->step(w->env, action) != K26RL_OK)
        return 1;
    if (w->s->obs(w->env, obs) != K26RL_OK)
        return 1;
    w->period++;
    w->advance_calls++;
    return 0;
}

/* ---- buffers a run needs -------------------------------------------- */

typedef struct {
    double  *obs, *candidate, *applied, *action_log;
    double  *latency_us, *lateness_us, *scratch;
    uint8_t *missed;
    double  *candidates;
    uint64_t periods;
    uint32_t act_total;
} RunBuffers;

static void run_buffers_alloc_(RunBuffers *b, uint64_t periods,
                               uint32_t obs_total, uint32_t act_total)
{
    memset(b, 0, sizeof *b);
    b->periods = periods;
    b->act_total = act_total;
    b->obs        = (double *)calloc(obs_total, sizeof(double));
    b->candidate  = (double *)calloc(act_total, sizeof(double));
    b->applied    = (double *)calloc(act_total, sizeof(double));
    b->action_log = (double *)calloc((size_t)periods * act_total,
                                     sizeof(double));
    b->candidates = (double *)calloc((size_t)periods * act_total,
                                     sizeof(double));
    b->latency_us  = (double *)calloc((size_t)periods, sizeof(double));
    b->lateness_us = (double *)calloc((size_t)periods, sizeof(double));
    b->scratch     = (double *)calloc((size_t)periods, sizeof(double));
    b->missed      = (uint8_t *)calloc((size_t)periods, 1);
    ASSERT(b->obs && b->candidate && b->applied && b->action_log &&
           b->candidates && b->latency_us && b->lateness_us &&
           b->scratch && b->missed);
}

static void run_buffers_free_(RunBuffers *b)
{
    free(b->obs);
    free(b->candidate);
    free(b->applied);
    free(b->action_log);
    free(b->candidates);
    free(b->latency_us);
    free(b->lateness_us);
    free(b->scratch);
    free(b->missed);
}

/* Drive one wall-clock run over a fresh handle, recording an episode
 * file exactly as the stepping driver would. */
static void realtime_run_(const RlSurface *s, const char *episode_path,
                          uint64_t seed, const K26RlPolicy *policy,
                          const Slices *sl, RunBuffers *b,
                          const double *preload, double rate_hz,
                          uint64_t stall_every, uint64_t stall_phase,
                          uint64_t stall_once, double stall_us,
                          K26RlRtReport *report)
{
    K26RlRtConfig config;
    K26RlRtHooks hooks;
    K26RlRtBuffers buffers;
    RtWorld world;
    K26RlEnv *env = NULL;
    K26RlRtStatus rc;

    ASSERT(s->create(seed, 1, &env) == K26RL_OK);
    if (episode_path)
        ASSERT(s->output(env, episode_path) == K26RL_OK);
    ASSERT(s->obs(env, b->obs) == K26RL_OK);

    memset(&world, 0, sizeof world);
    world.s = s;
    world.env = env;
    world.policy = policy;
    world.obs_total = sl->obs_total;
    world.act_total = sl->act_total;
    world.stall_every = stall_every;
    world.stall_phase = stall_phase;
    world.stall_once = stall_once;
    world.stall_us = stall_us;
    world.candidates = b->candidates;

    memcpy(b->applied, preload, sizeof(double) * sl->act_total);
    memcpy(b->candidate, preload, sizeof(double) * sl->act_total);

    memset(&config, 0, sizeof config);
    config.rate_hz = rate_hz;
    config.control_dt = RT_CONTROL_DT;
    config.periods = b->periods;
    config.obs_width = sl->obs_total;
    config.act_width = sl->act_total;
    config.rate_tolerance = RT_RATE_TOLERANCE;

    hooks.decide = rt_decide_;
    hooks.advance = rt_advance_;
    hooks.context = &world;

    buffers.obs = b->obs;
    buffers.candidate = b->candidate;
    buffers.applied = b->applied;
    buffers.action_log = b->action_log;
    buffers.latency_us = b->latency_us;
    buffers.lateness_us = b->lateness_us;
    buffers.scratch = b->scratch;
    buffers.missed = b->missed;

    rc = k26rl_rt_run(&config, &hooks, &buffers, report);
    if (rc != K26RL_RT_OK) {
        fprintf(stderr, "FAIL wall-clock run: %s\n", k26rl_rt_status_str(rc));
        exit(1);
    }
    ASSERT(report->periods_run == b->periods);
    ASSERT(world.decide_calls == b->periods);
    ASSERT(world.advance_calls == b->periods);
    s->destroy(env);
}

/* Replay an action stream through the stepping driver, recording an
 * episode file the same way. This is the stepping path exactly as any
 * other consumer drives it: nothing here knows a clock exists. */
static void replay_stream_(const RlSurface *s, const char *episode_path,
                           uint64_t seed, const double *stream,
                           uint64_t periods, uint32_t act_total)
{
    K26RlEnv *env = NULL;
    uint64_t p;

    ASSERT(s->create(seed, 1, &env) == K26RL_OK);
    ASSERT(s->output(env, episode_path) == K26RL_OK);
    for (p = 0; p < periods; p++)
        ASSERT(s->step(env, stream + (size_t)p * act_total) == K26RL_OK);
    s->destroy(env);
}

/* ---- the hold rule, as a check that can fail ------------------------
 *
 * Returns NULL when the applied stream obeys the rule, and the defect
 * otherwise. It is a function so that the arms can require it to reject
 * each defect it names: a check only ever run against correct data is
 * a check nobody has seen work. */
static const char *hold_defect_(const double *applied,
                                const double *candidates,
                                const uint8_t *missed,
                                const double *initial,
                                const double *defaults,
                                uint64_t periods, uint32_t act_total)
{
    size_t width = sizeof(double) * act_total;
    uint64_t p;

    for (p = 0; p < periods; p++) {
        const double *here = applied + (size_t)p * act_total;
        const double *before = (p == 0u) ? initial
                                         : applied + (size_t)(p - 1u) *
                                           act_total;

        if (missed[p]) {
            if (memcmp(here, before, width) != 0) {
                /* The substitution is named apart from the general
                 * failure to hold, because it is the alternative the
                 * design declined and a reader of a failure should not
                 * have to work out which one happened. */
                if (memcmp(here, defaults, width) == 0 &&
                    memcmp(before, defaults, width) != 0)
                    return "a missed period substituted the declared "
                           "default";
                return "a missed period did not hold the action in force";
            }
        } else {
            if (memcmp(here, candidates + (size_t)p * act_total,
                       width) != 0)
                return "a period that met its deadline did not apply the "
                       "action the policy produced";
        }
    }
    return NULL;
}

/* The miss flags against the timings that produced them. A period
 * whose decision finished after its own end must be flagged, and one
 * that finished before it must not be. Periods landing inside the
 * guard band either side of the deadline are counted and reported
 * rather than judged, since the driver compares whole nanoseconds and
 * this compares the microsecond figures derived from them. */
#define FLAG_GUARD_US 0.002

static const char *flag_defect_(const double *latency_us,
                               const double *lateness_us,
                               const uint8_t *missed, uint64_t periods,
                               double period_us, uint64_t *out_borderline)
{
    uint64_t p, borderline = 0;
    const char *defect = NULL;

    for (p = 0; p < periods; p++) {
        double used = lateness_us[p] + latency_us[p];

        if (used > period_us + FLAG_GUARD_US) {
            if (!missed[p] && !defect)
                defect = "a period whose decision finished after its own "
                         "end was not recorded as a miss";
        } else if (used < period_us - FLAG_GUARD_US) {
            if (missed[p] && !defect)
                defect = "a period whose decision finished inside its own "
                         "end was recorded as a miss";
        } else {
            borderline++;
        }
    }
    if (out_borderline)
        *out_borderline = borderline;
    return defect;
}

/* What a summary of a known distribution must say. The distribution
 * is `n` samples, all but the last at 10 microseconds and the last at
 * 4000, so every quantile below the very top sits at the flat value
 * and the maximum is the one sample that matters. Returns NULL when
 * the summary says so and the defect otherwise, and the arm requires
 * it to reject a summary that reports the mean as the maximum and one
 * that dropped its deepest samples. */
#define KNOWN_FLAT_US  10.0
#define KNOWN_SPIKE_US 4000.0

static const char *summary_defect_(const K26RlRtLatency *d, uint64_t n)
{
    double expected_mean = ((double)(n - 1u) * KNOWN_FLAT_US +
                            KNOWN_SPIKE_US) / (double)n;

    if (d->count != n)
        return "the summary counted a different number of samples";
    if (d->min_us != KNOWN_FLAT_US || d->p50_us != KNOWN_FLAT_US ||
        d->p90_us != KNOWN_FLAT_US || d->p99_us != KNOWN_FLAT_US ||
        d->p999_us != KNOWN_FLAT_US)
        return "a quantile of a flat distribution is not the flat value";
    if (d->max_us != KNOWN_SPIKE_US)
        return "the maximum is not the largest sample";
    if (fabs(d->mean_us - expected_mean) > 1e-9)
        return "the mean is not the mean";
    return NULL;
}

/* ---- the stepping drive, for the arm that holds it unchanged -------- */

static double stepped_action_(uint64_t t, uint32_t ch)
{
    if (ch == 0u)
        return ((double)((t * 7u + 3u) % 13u)) / 13.0 * 2.0 - 1.0;
    return ((double)((t * 5u + 1u) % 11u)) / 11.0 - 0.5;
}

static void stepped_drive_(const char *so_path, uint64_t seed,
                           uint64_t steps, const char *out_path)
{
    void *so = rl_dlopen_(so_path);
    RlSurface s;
    K26RlEnv *env = NULL;
    uint8_t *spec;
    int32_t spec_len;
    RlSpecView v;
    double *act;
    uint64_t t;

    rl_resolve_surface_(so, &s);
    ASSERT(s.create(seed, 1, &env) == K26RL_OK);
    spec_len = s.spec(env, NULL, 0);
    ASSERT(spec_len > 0);
    spec = (uint8_t *)malloc((size_t)spec_len);
    ASSERT(spec != NULL);
    ASSERT(s.spec(env, spec, (uint32_t)spec_len) == spec_len);
    rl_parse_spec_(spec, (uint32_t)spec_len, &v);
    act = (double *)calloc(v.act_total, sizeof(double));
    ASSERT(act != NULL);

    ASSERT(s.output(env, out_path) == K26RL_OK);
    for (t = 0; t < steps; t++) {
        uint32_t ch;

        for (ch = 0; ch < v.act_total; ch++)
            act[ch] = stepped_action_(t, ch);
        ASSERT(s.step(env, act) == K26RL_OK);
    }
    s.destroy(env);
    free(act);
    free(spec);
    dlclose(so);
}

/* ---- reporting ------------------------------------------------------ */

static void report_run_(const char *tag, const K26RlRtReport *r,
                        const char *load)
{
    printf("  %s\n", tag);
    printf("    periods %llu, missed %llu (%.3f per cent), "
           "worst overrun %.1f us, worst start lateness %.1f us\n",
           (unsigned long long)r->periods_run,
           (unsigned long long)r->periods_missed,
           100.0 * (double)r->periods_missed / (double)r->periods_run,
           r->worst_overrun_us, r->worst_start_lateness_us);
    printf("    wall clock %.4f s, nominal %.4f s, simulated %.4f s, "
           "real-time factor %.4f, rate held %s\n",
           r->wall_clock_s, r->nominal_s, r->simulated_s,
           r->real_time_factor, r->rate_held ? "yes" : "no");
    printf("    decision latency n %llu  min %.3f  p50 %.3f  p90 %.3f  "
           "p99 %.3f  p99.9 %.3f  max %.3f  (mean %.3f) us\n",
           (unsigned long long)r->decision.count, r->decision.min_us,
           r->decision.p50_us, r->decision.p90_us, r->decision.p99_us,
           r->decision.p999_us, r->decision.max_us, r->decision.mean_us);
    printf("    period start lateness  min %.3f  p50 %.3f  p90 %.3f  "
           "p99 %.3f  p99.9 %.3f  max %.3f  (mean %.3f) us\n",
           r->start_lateness.min_us, r->start_lateness.p50_us,
           r->start_lateness.p90_us, r->start_lateness.p99_us,
           r->start_lateness.p999_us, r->start_lateness.max_us,
           r->start_lateness.mean_us);
    printf("    period %llu ns; load average %s; the driver asked the "
           "operating system for no core, no priority and no locked "
           "pages\n",
           (unsigned long long)r->period_ns, load);
}

/* ---- probe: the statistics a policy standardises by ------------------
 *
 * Taken from the world itself rather than invented, so the network's
 * input is of order one and its output moves as the world moves, which
 * is what makes a held action distinguishable from a fresh one. */
static void probe_statistics_(const RlSurface *s, uint64_t seed,
                              const Slices *sl, uint32_t steps,
                              double *mean, double *variance)
{
    K26RlEnv *env = NULL;
    double *obs, *sum, *sumsq, *act;
    uint32_t i, t;

    obs = (double *)calloc(sl->obs_total, sizeof(double));
    sum = (double *)calloc(sl->obs_width, sizeof(double));
    sumsq = (double *)calloc(sl->obs_width, sizeof(double));
    act = (double *)calloc(sl->act_total, sizeof(double));
    ASSERT(obs && sum && sumsq && act);

    ASSERT(s->create(seed, 1, &env) == K26RL_OK);
    for (t = 0; t < steps; t++) {
        ASSERT(s->obs(env, obs) == K26RL_OK);
        for (i = 0; i < sl->obs_width; i++) {
            double x = obs[sl->obs_offset + i];

            sum[i] += x;
            sumsq[i] += x * x;
        }
        ASSERT(s->step(env, act) == K26RL_OK);
    }
    s->destroy(env);

    for (i = 0; i < sl->obs_width; i++) {
        double m = sum[i] / (double)steps;
        double var = sumsq[i] / (double)steps - m * m;

        /* A channel that never moved has no scale of its own; the
         * floor keeps the standardisation defined rather than letting
         * one constant channel divide by nothing. */
        if (!(var > 1e-12))
            var = 1e-12;
        mean[i] = m;
        variance[i] = var;
    }
    free(obs);
    free(sum);
    free(sumsq);
    free(act);
}

/* ---- arm 1 and arm 2: the replay property --------------------------- */

/* Run, replay, compare as whole files, and require a perturbed stream
 * to be caught. `stall_every` zero drives a run that misses nothing. */
static void arm_replay_(const RlSurface *s, const K26RlPolicy *policy,
                        const Slices *sl, uint64_t periods, uint64_t seed,
                        uint64_t stall_every, uint64_t stall_phase,
                        double stall_us, const char *stem,
                        const char *label, const char *load, RunBuffers *b,
                        K26RlRtReport *report, char *out_live,
                        size_t live_cap)
{
    char live[512], replayed[512], red[512];
    double *altered;
    size_t stream_doubles = (size_t)periods * sl->act_total;

    snprintf(live, sizeof live, "%s/%s_live.k26epi", WORK_DIR, stem);
    snprintf(replayed, sizeof replayed, "%s/%s_replay.k26epi", WORK_DIR,
             stem);
    snprintf(red, sizeof red, "%s/%s_red.k26epi", WORK_DIR, stem);
    if (out_live)
        snprintf(out_live, live_cap, "%s", live);

    realtime_run_(s, live, seed, policy, sl, b, RT_DEFAULTS, RT_RATE_HZ,
                  stall_every, stall_phase, 0u, stall_us, report);
    report_run_(label, report, load);

    replay_stream_(s, replayed, seed, b->action_log, periods, sl->act_total);
    ASSERT(rl_files_equal_(live, replayed));
    g_arms++;

    /* The red control: one action moved by the smallest amount a
     * double can carry. The files must stop agreeing, which is what
     * says the comparison above compared something. */
    altered = (double *)malloc(sizeof(double) * stream_doubles);
    ASSERT(altered != NULL);
    memcpy(altered, b->action_log, sizeof(double) * stream_doubles);
    altered[stream_doubles / 2u] = nextafter(altered[stream_doubles / 2u],
                                             1.0e9);
    ASSERT(memcmp(altered, b->action_log,
                  sizeof(double) * stream_doubles) != 0);
    replay_stream_(s, red, seed, altered, periods, sl->act_total);
    if (rl_files_equal_(live, red)) {
        fprintf(stderr, "FAIL %s: a stream altered in one action replayed "
                        "to identical bytes, so the comparison measures "
                        "nothing\n", label);
        exit(1);
    }
    free(altered);
    g_arms++;
}

int main(int argc, char **argv)
{
    void *so;
    RlSurface s;
    RlSpecView v;
    Slices sl;
    K26RlEnv *env = NULL;
    uint8_t *spec;
    int32_t spec_len;
    double *mean, *variance;
    unsigned char *fast_bytes;
    size_t fast_len;
    uint64_t fast_params;
    K26RlPolicy *fast = NULL;
    char detail[256], load[64];
    char so_path[512];

    /* Subprocess mode for the arm that holds the stepping driver
     * unchanged. It runs before anything else so a subprocess never
     * recompiles the fixture. */
    if (argc == 6 && strcmp(argv[1], "--stepped") == 0) {
        stepped_drive_(argv[2], strtoull(argv[3], NULL, 10),
                       strtoull(argv[4], NULL, 10), argv[5]);
        return 0;
    }

    if (!rl_libs_present_("test_rl_realtime"))
        return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    load_average_(load, sizeof load);
    printf("test_rl_realtime: load average at start %s\n", load);

    rl_write_file_(WORK_DIR "/rt.kfl", RT_KFL);
    rl_compile_(WORK_DIR "/rt.kfl", WORK_DIR "/rt", WORK_DIR);
    snprintf(so_path, sizeof so_path, "%s/rt.rlenv.so", WORK_DIR);
    ASSERT(rl_file_exists_(so_path));
    so = rl_dlopen_(so_path);
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    ASSERT(s.create(1, 1, &env) == K26RL_OK);
    spec_len = s.spec(env, NULL, 0);
    ASSERT(spec_len > 0);
    spec = (uint8_t *)malloc((size_t)spec_len);
    ASSERT(spec != NULL);
    ASSERT(s.spec(env, spec, (uint32_t)spec_len) == spec_len);
    rl_parse_spec_(spec, (uint32_t)spec_len, &v);
    s.destroy(env);

    memset(&sl, 0, sizeof sl);
    sl.agent_count = v.agent_count;
    sl.obs_total = v.obs_total;
    sl.act_total = v.act_total;
    ASSERT(v.n_obs_slices == 1 && v.n_act_slices == 1);
    sl.obs_offset = v.obs_slice[0][0];
    sl.obs_width  = v.obs_slice[0][1];
    sl.act_offset = v.act_slice[0][0];
    sl.act_width  = v.act_slice[0][1];
    /* The arms preload and compare whole action vectors, so the one
     * agent must own all of them; a fixture that changed would say so
     * here rather than by comparing the wrong bytes. */
    ASSERT(sl.act_offset == 0u && sl.act_width == sl.act_total);
    ASSERT(sl.act_total == 2u);
    {
        double dt_bits;

        memcpy(&dt_bits, &v.control_dt_bits, sizeof dt_bits);
        ASSERT(dt_bits == RT_CONTROL_DT);
    }
    printf("  world: %u observation channels, %u action channels, "
           "control dt %g s, driven at %g Hz\n",
           sl.obs_total, sl.act_total, RT_CONTROL_DT, RT_RATE_HZ);

    mean = (double *)calloc(sl.obs_width, sizeof(double));
    variance = (double *)calloc(sl.obs_width, sizeof(double));
    ASSERT(mean && variance);
    probe_statistics_(&s, 11u, &sl, 400u, mean, variance);

    fast_bytes = build_policy_(&sl, 64u, 2u, mean, variance, 7u,
                               &fast_len, &fast_params);
    ASSERT(k26rl_policy_parse(fast_bytes, (uint64_t)fast_len, &fast) ==
           K26RL_POLICY_OK);
    if (k26rl_policy_check_spec(fast, spec, (uint32_t)spec_len, detail,
                                sizeof detail) != K26RL_POLICY_OK) {
        fprintf(stderr, "FAIL the policy does not fit the world: %s\n",
                detail);
        exit(1);
    }
    printf("  policy: %llu parameters, checked against the world's own "
           "declared slices\n", (unsigned long long)fast_params);

    /* ---- arm 1: replay on a run that missed nothing ---------------- */
    {
        RunBuffers b;
        K26RlRtReport r;
        const char *defect;
        uint64_t borderline = 0;

        run_buffers_alloc_(&b, 600u, sl.obs_total, sl.act_total);
        arm_replay_(&s, fast, &sl, 600u, 42u, 0u, 0u, 0.0, "clean",
                    "clean run", load, &b, &r, NULL, 0);
        printf("    the recorded stream replayed through the stepping "
               "driver to identical bytes, and one action moved by an "
               "ulp did not\n");

        /* A run at the world's own rate that reports holding it must
         * have taken as long as the world it simulated. */
        ASSERT(r.rate_held == 1);
        ASSERT(fabs(r.wall_clock_s - r.simulated_s) <=
               0.02 * r.simulated_s);
        ASSERT(fabs(r.real_time_factor - 1.0) <= 0.02);
        g_arms++;

        /* A quiet machine should miss nothing here; a busy one may
         * miss a little and the property above still holds. A run that
         * missed a large share of its periods is not the clean run
         * this arm is about, and says so. */
        ASSERT(r.periods_missed * 100u <= r.periods_run);
        defect = hold_defect_(b.action_log, b.candidates, b.missed,
                              RT_DEFAULTS, RT_DEFAULTS, r.periods_run,
                              sl.act_total);
        ASSERT(defect == NULL);
        defect = flag_defect_(b.latency_us, b.lateness_us, b.missed,
                              r.periods_run,
                              (double)r.period_ns / 1000.0, &borderline);
        ASSERT(defect == NULL);
        printf("    miss flags agree with the timings that produced "
               "them, %llu periods inside the guard band\n",
               (unsigned long long)borderline);
        g_arms++;
        run_buffers_free_(&b);
    }

    /* ---- arm 2 and arm 3: a run that missed, and what a miss does --- */
    {
        RunBuffers b;
        K26RlRtReport r;
        const char *defect;
        uint64_t p, borderline = 0, differing = 0;
        size_t width = sizeof(double) * sl.act_total;
        char red_path[512], live_path[512];
        double *late_stream;

        run_buffers_alloc_(&b, 600u, sl.obs_total, sl.act_total);
        /* Every eighth period is made to think for a period and a
         * third, which is past its own end and inside the next one, so
         * a miss is followed by a period that meets its deadline. */
        arm_replay_(&s, fast, &sl, 600u, 43u, 8u, 3u, 3200.0, "stalled",
                    "run with stalls injected past the deadline",
                    load, &b, &r, live_path, sizeof live_path);
        ASSERT(r.periods_missed >= 60u);
        printf("    the recorded stream, held actions included, "
               "replayed to identical bytes\n");
        g_arms++;

        /* The decisive red control: replace each held action with the
         * late action the policy actually produced on that period. If
         * the held actions were not what the run applied, or if they
         * did not reach the physics, these bytes would still agree. */
        for (p = 0; p < r.periods_run; p++) {
            if (b.missed[p] &&
                memcmp(b.action_log + (size_t)p * sl.act_total,
                       b.candidates + (size_t)p * sl.act_total,
                       width) != 0) {
                differing++;
            }
        }
        ASSERT(differing == r.periods_missed);
        printf("    on every missed period the held action differs from "
               "the late one the policy produced\n");

        late_stream = (double *)malloc(sizeof(double) * (size_t)600u *
                                       sl.act_total);
        ASSERT(late_stream != NULL);
        memcpy(late_stream, b.action_log,
               sizeof(double) * (size_t)600u * sl.act_total);
        for (p = 0; p < r.periods_run; p++) {
            if (b.missed[p]) {
                memcpy(late_stream + (size_t)p * sl.act_total,
                       b.candidates + (size_t)p * sl.act_total, width);
            }
        }
        snprintf(red_path, sizeof red_path, "%s/late_actions.k26epi",
                 WORK_DIR);
        replay_stream_(&s, red_path, 43u, late_stream, 600u, sl.act_total);
        if (rl_files_equal_(live_path, red_path)) {
            fprintf(stderr, "FAIL a stream with the late actions in place "
                            "of the held ones replayed to identical "
                            "bytes\n");
            exit(1);
        }
        free(late_stream);
        g_arms++;

        /* The hold rule itself, and the check shown rejecting each
         * defect it names. */
        defect = hold_defect_(b.action_log, b.candidates, b.missed,
                              RT_DEFAULTS, RT_DEFAULTS, r.periods_run,
                              sl.act_total);
        if (defect) {
            fprintf(stderr, "FAIL the applied stream: %s\n", defect);
            exit(1);
        }
        printf("    every missed period held the action in force and "
               "every other applied what the policy produced\n");
        g_arms++;

        {
            double *spoiled = (double *)malloc(sizeof(double) *
                                               (size_t)600u * sl.act_total);
            uint64_t first_miss = r.periods_run;

            ASSERT(spoiled != NULL);
            for (p = 0; p < r.periods_run; p++) {
                if (b.missed[p] && p > 0u) {
                    first_miss = p;
                    break;
                }
            }
            ASSERT(first_miss < r.periods_run);

            /* A miss that substituted the declared default. */
            memcpy(spoiled, b.action_log,
                   sizeof(double) * (size_t)600u * sl.act_total);
            memcpy(spoiled + (size_t)first_miss * sl.act_total,
                   RT_DEFAULTS, width);
            ASSERT(hold_defect_(spoiled, b.candidates, b.missed,
                                RT_DEFAULTS, RT_DEFAULTS, r.periods_run,
                                sl.act_total) != NULL);

            /* A miss that applied the late action after all. */
            memcpy(spoiled, b.action_log,
                   sizeof(double) * (size_t)600u * sl.act_total);
            memcpy(spoiled + (size_t)first_miss * sl.act_total,
                   b.candidates + (size_t)first_miss * sl.act_total, width);
            ASSERT(hold_defect_(spoiled, b.candidates, b.missed,
                                RT_DEFAULTS, RT_DEFAULTS, r.periods_run,
                                sl.act_total) != NULL);

            /* A period that met its deadline and applied something
             * else. */
            memcpy(spoiled, b.action_log,
                   sizeof(double) * (size_t)600u * sl.act_total);
            ASSERT(!b.missed[first_miss + 1u]);
            spoiled[(size_t)(first_miss + 1u) * sl.act_total] += 1.0;
            ASSERT(hold_defect_(spoiled, b.candidates, b.missed,
                                RT_DEFAULTS, RT_DEFAULTS, r.periods_run,
                                sl.act_total) != NULL);
            free(spoiled);
            printf("    the hold check rejects a substituted default, a "
                   "late action applied, and a wrong action on a period "
                   "that met its deadline\n");
            g_arms++;
        }

        /* The miss flags against the timings, and that check shown
         * rejecting a miss gone unrecorded and one recorded that did
         * not happen. */
        defect = flag_defect_(b.latency_us, b.lateness_us, b.missed,
                              r.periods_run,
                              (double)r.period_ns / 1000.0, &borderline);
        if (defect) {
            fprintf(stderr, "FAIL the miss flags: %s\n", defect);
            exit(1);
        }
        {
            uint8_t *flags = (uint8_t *)malloc((size_t)600u);
            uint64_t first_miss = 0;

            ASSERT(flags != NULL);
            for (p = 0; p < r.periods_run; p++) {
                if (b.missed[p]) { first_miss = p; break; }
            }
            memcpy(flags, b.missed, (size_t)600u);
            flags[first_miss] = 0u;
            ASSERT(flag_defect_(b.latency_us, b.lateness_us, flags,
                                r.periods_run,
                                (double)r.period_ns / 1000.0,
                                NULL) != NULL);
            memcpy(flags, b.missed, (size_t)600u);
            flags[first_miss + 1u] = 1u;
            ASSERT(flag_defect_(b.latency_us, b.lateness_us, flags,
                                r.periods_run,
                                (double)r.period_ns / 1000.0,
                                NULL) != NULL);
            free(flags);
            printf("    the flag check rejects a miss gone unrecorded and "
                   "one recorded where nothing was late, %llu periods "
                   "inside the guard band\n",
                   (unsigned long long)borderline);
            g_arms++;
        }
        run_buffers_free_(&b);
    }

    /* ---- arm 4: a policy the clock will not take ------------------- */
    {
        unsigned char *slow_bytes;
        size_t slow_len;
        uint64_t slow_params = 0;
        K26RlPolicy *slow = NULL;
        uint32_t hidden = 2048u;
        double one_call_us = 0.0;
        double period_us = 1e6 / RT_RATE_HZ;
        RunBuffers b;
        K26RlRtReport r;
        const char *defect;
        uint64_t p;
        double *scrap;

        scrap = (double *)calloc(sl.act_total, sizeof(double));
        ASSERT(scrap != NULL);
        for (;;) {
            struct timespec t0, t1;
            double *probe_obs = (double *)calloc(sl.obs_total,
                                                 sizeof(double));
            int i;

            ASSERT(probe_obs != NULL);
            slow_bytes = build_policy_(&sl, hidden, 2u, mean, variance,
                                       9973u, &slow_len, &slow_params);
            ASSERT(k26rl_policy_parse(slow_bytes, (uint64_t)slow_len,
                                      &slow) == K26RL_POLICY_OK);
            ASSERT(k26rl_policy_check_spec(slow, spec, (uint32_t)spec_len,
                                           detail, sizeof detail) ==
                   K26RL_POLICY_OK);
            for (i = 0; i < 3; i++)
                (void)k26rl_policy_act_env(slow, probe_obs, scrap);
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (i = 0; i < 3; i++)
                (void)k26rl_policy_act_env(slow, probe_obs, scrap);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            one_call_us = ((double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                           (double)(t1.tv_nsec - t0.tv_nsec) * 1e-3) / 3.0;
            free(probe_obs);
            printf("  a policy of %llu parameters evaluates in %.0f us "
                   "against a %.0f us period\n",
                   (unsigned long long)slow_params, one_call_us, period_us);
            if (one_call_us > 1.5 * period_us || hidden >= 4096u)
                break;
            k26rl_policy_close(slow);
            slow = NULL;
            free(slow_bytes);
            hidden *= 2u;
        }
        free(scrap);
        if (!(one_call_us > 1.5 * period_us)) {
            fprintf(stderr, "FAIL no policy inside the size cap is slow "
                            "enough to miss this period on this machine; "
                            "the arm would prove nothing\n");
            exit(1);
        }

        run_buffers_alloc_(&b, 150u, sl.obs_total, sl.act_total);
        realtime_run_(&s, NULL, 44u, slow, &sl, &b, RT_PRELOAD,
                      RT_RATE_HZ, 0u, 0u, 0u, 0.0, &r);
        report_run_("a policy too large for the period", &r, load);

        /* Every period misses, and every one holds what the command
         * register held. That register is not the declared default, so
         * a driver that substituted the default would be caught here
         * whatever else it did. */
        ASSERT(r.periods_missed == r.periods_run);
        for (p = 0; p < r.periods_run; p++) {
            ASSERT(memcmp(b.action_log + (size_t)p * sl.act_total,
                          RT_PRELOAD,
                          sizeof(double) * sl.act_total) == 0);
            ASSERT(memcmp(b.action_log + (size_t)p * sl.act_total,
                          RT_DEFAULTS,
                          sizeof(double) * sl.act_total) != 0);
        }
        defect = hold_defect_(b.action_log, b.candidates, b.missed,
                              RT_PRELOAD, RT_DEFAULTS, r.periods_run,
                              sl.act_total);
        ASSERT(defect == NULL);
        printf("    every period missed, and every one held the command "
               "register rather than the declared default\n");
        g_arms++;

        /* And the run says it fell behind rather than reporting a rate
         * it did not hold. */
        ASSERT(r.rate_held == 0);
        ASSERT(r.wall_clock_s > r.simulated_s);
        ASSERT(r.real_time_factor < 1.0);
        ASSERT(r.worst_start_lateness_us > period_us);
        printf("    the run reports that it did not hold rate, with its "
               "wall clock %.3f s past its simulated %.3f s\n",
               r.wall_clock_s, r.simulated_s);
        g_arms++;

        run_buffers_free_(&b);
        k26rl_policy_close(slow);
        free(slow_bytes);
    }

    /* ---- arm 5: the distribution, and the worst -------------------- */
    {
        enum { N = 2000, SPIKE = 1000 };
        RunBuffers b;
        K26RlRtReport r;
        double raw_max = 0.0;
        uint64_t p, above = 0;

        run_buffers_alloc_(&b, (uint64_t)N, sl.obs_total, sl.act_total);
        realtime_run_(&s, NULL, 45u, fast, &sl, &b, RT_DEFAULTS,
                      RT_RATE_HZ, 0u, 0u, (uint64_t)SPIKE, 4000.0, &r);
        report_run_("one slow period among fast ones", &r, load);

        for (p = 0; p < r.periods_run; p++) {
            if (b.latency_us[p] > raw_max)
                raw_max = b.latency_us[p];
            if (b.latency_us[p] > 2000.0)
                above++;
        }
        /* The reported maximum is the worst sample and not a summary of
         * the others, and the slow period is the one that produced it. */
        ASSERT(r.decision.max_us == raw_max);
        ASSERT(b.latency_us[SPIKE] == raw_max);
        ASSERT(raw_max > 3800.0 && raw_max < 6000.0);
        if (above != 1u) {
            fprintf(stderr, "FAIL %llu decisions took more than two "
                            "milliseconds where one was injected; on a "
                            "contended machine the extra ones are the "
                            "scheduler and the arm should be re-run "
                            "quiet\n", (unsigned long long)above);
            exit(1);
        }
        g_arms++;

        /* The median did not move, and a summary quoting the mean alone
         * would have understated the worst by two orders of magnitude. */
        ASSERT(r.decision.p50_us < 0.05 * (double)r.period_ns / 1000.0);
        ASSERT(r.decision.p99_us <= 25.0 * r.decision.p50_us);
        ASSERT(r.decision.p999_us <= 40.0 * r.decision.p50_us);
        ASSERT(r.decision.max_us > 50.0 * r.decision.mean_us);
        printf("    the maximum is %.0f times the mean, so a summary "
               "quoting the mean alone would have hidden the slow "
               "period\n", r.decision.max_us / r.decision.mean_us);
        g_arms++;

        /* The worst overrun is the slow period's own, recomputed here
         * from that period's own two timings rather than allowed a
         * band: how far past its end an action arrived is how late the
         * period began plus how long the decision took, less the
         * period. A driver reporting the last overrun, or an average,
         * or the wrong period's, disagrees with this arithmetic. */
        {
            double expected = b.lateness_us[SPIKE] + b.latency_us[SPIKE] -
                              (double)r.period_ns / 1000.0;

            ASSERT(fabs(r.worst_overrun_us - expected) < 0.01);
        }
        ASSERT(r.periods_missed >= 1u);
        ASSERT(r.periods_missed <= 1u + r.periods_run / 100u);
        g_arms++;
        run_buffers_free_(&b);
    }

    /* ---- arm 5b: the summary against a distribution it cannot argue
     * with, and two defective summaries it must reject -------------- */
    {
        enum { N = 1000 };
        double *samples = (double *)malloc(sizeof(double) * N);
        double *scratch = (double *)malloc(sizeof(double) * N);
        K26RlRtLatency d, broken;
        int i;

        ASSERT(samples && scratch);
        for (i = 0; i < N - 1; i++)
            samples[i] = 10.0;
        samples[N - 1] = 4000.0;

        ASSERT(k26rl_rt_summarise(samples, (uint64_t)N, scratch, &d) ==
               K26RL_RT_OK);
        ASSERT(summary_defect_(&d, (uint64_t)N) == NULL);
        printf("  a known distribution of %d samples, %d flat and one "
               "slow: p50 %.2f, p99 %.2f, p99.9 %.2f, max %.2f, mean "
               "%.4f us\n", N, N - 1, d.p50_us, d.p99_us, d.p999_us,
               d.max_us, d.mean_us);
        printf("    even the 99.9th percentile sits at the flat value, "
               "which is why the maximum is reported beside it\n");
        g_arms++;

        /* A summary reporting the mean as the maximum. */
        broken = d;
        broken.max_us = broken.mean_us;
        ASSERT(summary_defect_(&broken, (uint64_t)N) != NULL);

        /* A summary that dropped its deepest samples: the same
         * summariser over the array with its last entry cut off. */
        ASSERT(k26rl_rt_summarise(samples, (uint64_t)N - 1u, scratch,
                                  &broken) == K26RL_RT_OK);
        ASSERT(summary_defect_(&broken, (uint64_t)N) != NULL);
        printf("    the check rejects a summary reporting the mean as the "
               "maximum, and one that dropped its deepest samples\n");
        g_arms++;

        ASSERT(k26rl_rt_summarise(NULL, 1u, scratch, &d) ==
               K26RL_RT_E_NULL);
        ASSERT(k26rl_rt_summarise(samples, 0u, scratch, &d) ==
               K26RL_RT_E_SAMPLES);
        g_arms++;
        free(samples);
        free(scratch);
    }

    /* ---- arm 5c: whose cost is a decision's cost -------------------
     *
     * A decision measured inside the loop costs visibly more than the
     * same decision measured back to back, and the difference has to
     * be attributed before it can be reported. Three measurements
     * settle it: the same policy called back to back, called once per
     * control period with nothing but a sleep between calls, and
     * called by the driver. If the driver charged its own work to the
     * decision, only the third would be dear; if the machine is
     * cheaper when it is busy, the second and third agree and the
     * first is the odd one. */
    {
        enum { BACK_TO_BACK = 20000, DUTY_CYCLED = 400 };
        double *samples, *scratch, *probe_obs, *scrap;
        K26RlRtLatency tight, duty;
        RunBuffers b;
        K26RlRtReport r;
        int i;

        samples = (double *)malloc(sizeof(double) * BACK_TO_BACK);
        scratch = (double *)malloc(sizeof(double) * BACK_TO_BACK);
        probe_obs = (double *)calloc(sl.obs_total, sizeof(double));
        scrap = (double *)calloc(sl.act_total, sizeof(double));
        ASSERT(samples && scratch && probe_obs && scrap);
        {
            K26RlEnv *probe = NULL;

            ASSERT(s.create(5u, 1u, &probe) == K26RL_OK);
            ASSERT(s.obs(probe, probe_obs) == K26RL_OK);
            s.destroy(probe);
        }
        for (i = 0; i < 2000; i++)
            (void)k26rl_policy_act_env(fast, probe_obs, scrap);
        for (i = 0; i < BACK_TO_BACK; i++) {
            struct timespec t0, t1;

            clock_gettime(CLOCK_MONOTONIC, &t0);
            (void)k26rl_policy_act_env(fast, probe_obs, scrap);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            samples[i] = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                         (double)(t1.tv_nsec - t0.tv_nsec) * 1e-3;
        }
        ASSERT(k26rl_rt_summarise(samples, (uint64_t)BACK_TO_BACK, scratch,
                                  &tight) == K26RL_RT_OK);

        for (i = 0; i < DUTY_CYCLED; i++) {
            struct timespec t0, t1, wake;

            clock_gettime(CLOCK_MONOTONIC, &t0);
            (void)k26rl_policy_act_env(fast, probe_obs, scrap);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            samples[i] = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                         (double)(t1.tv_nsec - t0.tv_nsec) * 1e-3;
            wake = t1;
            wake.tv_nsec += (long)(1e9 / RT_RATE_HZ);
            if (wake.tv_nsec >= 1000000000L) {
                wake.tv_nsec -= 1000000000L;
                wake.tv_sec += 1;
            }
            while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &wake,
                                   NULL) == EINTR)
                ;
        }
        ASSERT(k26rl_rt_summarise(samples, (uint64_t)DUTY_CYCLED, scratch,
                                  &duty) == K26RL_RT_OK);

        run_buffers_alloc_(&b, (uint64_t)DUTY_CYCLED, sl.obs_total,
                           sl.act_total);
        realtime_run_(&s, NULL, 46u, fast, &sl, &b, RT_DEFAULTS,
                      RT_RATE_HZ, 0u, 0u, 0u, 0.0, &r);

        printf("  the same decision, measured three ways at load %s\n",
               load);
        printf("    back to back, %d calls        p50 %8.3f  p99 %8.3f  "
               "max %8.3f us\n", BACK_TO_BACK, tight.p50_us, tight.p99_us,
               tight.max_us);
        printf("    once per period, no driver    p50 %8.3f  p99 %8.3f  "
               "max %8.3f us\n", duty.p50_us, duty.p99_us, duty.max_us);
        printf("    once per period, in the loop  p50 %8.3f  p99 %8.3f  "
               "max %8.3f us\n", r.decision.p50_us, r.decision.p99_us,
               r.decision.max_us);
        printf("    a decision costs %.2f times as much inside a "
               "duty-cycled loop as it does back to back, and the driver "
               "accounts for %.2f of that\n",
               duty.p50_us / tight.p50_us,
               r.decision.p50_us / duty.p50_us);

        /* The driver must not be charging its own work to the decision:
         * the loop's median and the bare duty cycle's must agree. A
         * driver that did per-period work inside the measured window
         * would separate them and fail here. */
        ASSERT(r.decision.p50_us < 1.5 * duty.p50_us);
        /* And both must sit well inside the period, which is the
         * requirement the whole mode exists to check. */
        ASSERT(r.decision.p50_us < 0.05 * (double)r.period_ns / 1000.0);
        g_arms++;

        run_buffers_free_(&b);
        free(samples);
        free(scratch);
        free(probe_obs);
        free(scrap);
    }

    /* ---- arm 6: the stepping driver, with this one in the binary --- */
    {
        char cmd[1024];
        int n;

        fflush(stdout);
        stepped_drive_(so_path, 909u, 400u,
                       WORK_DIR "/stepped_here.k26epi");
        n = snprintf(cmd, sizeof cmd,
                     "%s --stepped %s 909 400 %s/stepped_p1.k26epi",
                     argv[0], so_path, WORK_DIR);
        ASSERT(n > 0 && (size_t)n < sizeof cmd);
        rl_run_or_die_(cmd);
        n = snprintf(cmd, sizeof cmd,
                     "%s --stepped %s 909 400 %s/stepped_p2.k26epi",
                     argv[0], so_path, WORK_DIR);
        ASSERT(n > 0 && (size_t)n < sizeof cmd);
        rl_run_or_die_(cmd);
        ASSERT(rl_files_equal_(WORK_DIR "/stepped_p1.k26epi",
                               WORK_DIR "/stepped_p2.k26epi"));
        ASSERT(rl_files_equal_(WORK_DIR "/stepped_here.k26epi",
                               WORK_DIR "/stepped_p1.k26epi"));
        printf("  the stepping driver is bit-identical across two fresh "
               "processes and this one, which has run the wall-clock "
               "driver five times\n");
        g_arms++;

        n = snprintf(cmd, sizeof cmd,
                     "%s --stepped %s 910 400 %s/stepped_alt.k26epi",
                     argv[0], so_path, WORK_DIR);
        ASSERT(n > 0 && (size_t)n < sizeof cmd);
        rl_run_or_die_(cmd);
        if (rl_files_equal_(WORK_DIR "/stepped_here.k26epi",
                            WORK_DIR "/stepped_alt.k26epi")) {
            fprintf(stderr, "FAIL two different seeds recorded identical "
                            "bytes, so the comparison measures nothing\n");
            exit(1);
        }
        g_arms++;
    }

    /* ---- arm 7: the program a person runs -------------------------- */
    {
        char cmd[2048];
        char policy_file[512], episode[512], actions[512], replay[512];
        FILE *f;
        double *stream;
        size_t want;
        int n;

        snprintf(policy_file, sizeof policy_file,
                 "%s/fitted" K26RL_POLICY_SUFFIX, WORK_DIR);
        f = fopen(policy_file, "wb");
        ASSERT(f != NULL);
        ASSERT(fwrite(fast_bytes, 1, fast_len, f) == fast_len);
        ASSERT(fclose(f) == 0);

        snprintf(episode, sizeof episode, "%s/tool.k26epi", WORK_DIR);
        snprintf(actions, sizeof actions, "%s/tool_actions.bin", WORK_DIR);
        snprintf(replay, sizeof replay, "%s/tool_replay.k26epi", WORK_DIR);
        n = snprintf(cmd, sizeof cmd,
                     "../tools/k26rl_rtdrive/k26rl_rtdrive --artifact %s "
                     "--policy %s --periods 300 --seed 77 --episode %s "
                     "--actions %s > %s/tool.log 2>&1",
                     so_path, policy_file, episode, actions, WORK_DIR);
        ASSERT(n > 0 && (size_t)n < sizeof cmd);
        fflush(stdout);
        rl_run_or_die_(cmd);
        rl_run_or_die_("sed 's/^/    /' " WORK_DIR "/tool.log");

        want = (size_t)300u * sl.act_total;
        stream = (double *)malloc(sizeof(double) * want);
        ASSERT(stream != NULL);
        f = fopen(actions, "rb");
        ASSERT(f != NULL);
        ASSERT(fread(stream, sizeof(double), want, f) == want);
        ASSERT(fgetc(f) == EOF);
        fclose(f);

        replay_stream_(&s, replay, 77u, stream, 300u, sl.act_total);
        ASSERT(rl_files_equal_(episode, replay));
        printf("    the program's own recorded actions replayed through "
               "the stepping driver to bytes identical to the episode it "
               "wrote\n");
        free(stream);
        g_arms++;
    }

    k26rl_policy_close(fast);
    free(fast_bytes);
    free(mean);
    free(variance);
    free(spec);
    free(g_pol);
    dlclose(so);
    load_average_(load, sizeof load);
    printf("test_rl_realtime: %d arms passed; load average at end %s\n",
           g_arms, load);
    return 0;
}
