/* test_rl_policy_latency.c: what one decision costs, and the
 * properties a caller working to a deadline depends on.
 *
 * A controller that runs against a wall clock has a fixed budget per
 * decision: a loop at several hundred hertz leaves a few milliseconds,
 * and the environment step is a small fraction of that, so what is
 * left is the policy's own evaluation. This gate measures that
 * evaluation and holds it against a budget, and it measures it as a
 * distribution because a policy that is usually fast and occasionally
 * slow is the shape that misses deadlines, and a mean hides it.
 *
 * Arms:
 *   1. The distribution. A policy of the shape a trained controller
 *      has is evaluated many times, every call timed on its own, and
 *      the minimum, median, 90th, 99th and 99.9th percentiles and the
 *      maximum are reported. The median is held under a ceiling well
 *      inside the budget, and the 99th and 99.9th percentiles are each
 *      held to a multiple of the median, which is the tail property
 *      that matters: the evaluator must not have a heavy tail of its
 *      own making, whether it takes its slow path on one call in a
 *      hundred or on one in a thousand. The maximum is reported rather
 *      than gated, because on a machine that is not reserved a single
 *      preempted call measures the scheduler and not this code; a
 *      driver accounts for that itself, and the gate says so rather
 *      than pretending to measure it.
 *   2. That the timing measures the evaluation at all. A policy an
 *      order of magnitude larger in parameters must come out several
 *      times slower through the same harness. A harness that timed
 *      nothing, or timed its own loop overhead, would pass every
 *      ceiling in this file and fail this arm.
 *   3. That the summary reports the worst and that a mean would not.
 *      One deliberately slow sample is placed among the fast ones:
 *      the reported maximum must be that sample, the median must not
 *      move, and the mean must move by so little that a summary
 *      quoting the mean alone would have hidden it.
 *   4. No allocation on the decision path. Every allocator entry the
 *      evaluator could reach is counted across a long run of calls and
 *      must be zero, because a decision that allocates has a latency
 *      tail belonging to the allocator rather than to the policy. A
 *      control allocation inside the counted region proves the
 *      counter is live, so a zero here is a measurement and not a
 *      counter that never ran.
 *   5. Determinism. The same weights over the same observation give
 *      the same bits on every call; two independently loaded copies of
 *      one file agree bit for bit; and the whole-vector entry point
 *      agrees bit for bit with the slice entry point, which is the
 *      pair a driver chooses between.
 *   6. The first decision, taken through the entry point that reads a
 *      file. A caller that starts cold gets the same action as one
 *      that has been running, bit for bit, and the cost of the first
 *      calls is reported beside the steady state. Those first costs
 *      are reported and not gated: this process is warm by the time
 *      the arm runs, so they understate what a fresh process pays to
 *      page its working set in, and the sound advice for a caller with
 *      a deadline is to evaluate once before its first period.
 *
 * Nothing here needs a compiled world, so every arm always runs.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "k26rl_digest.h"
#include "k26rl_policy.h"

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

/* Arms run, so the closing line carries a count: a gate that stopped
 * running half of what it holds would otherwise print the line it
 * prints when it runs all of it. */
static int g_arms;

/* The budget a decision has at the tight end of a several-hundred
 * hertz loop, in microseconds, and the share of it this gate insists
 * the median evaluation stay inside. The ceiling is a twentieth of
 * the whole rather than the whole, so the gate fails on a regression
 * that is still nominally inside the budget, and it is far enough
 * above what the evaluator costs that ordinary machine variation
 * cannot reach it. */
#define DECISION_BUDGET_US   2000.0
#define MEDIAN_CEILING_US    100.0

/* The tail is held at two depths, each as a multiple of the median
 * rather than as an absolute figure, because on a machine this gate
 * does not own the absolute tail belongs partly to the scheduler while
 * the shape of the distribution belongs to the evaluator. The first
 * catches a slow path taken on one call in a hundred, the second one
 * taken on one call in a thousand. Both ceilings sit an order of
 * magnitude above what the evaluator measures and well below what a
 * single slow path costs, and both were set by measuring an evaluator
 * with a slow path deliberately put into it rather than by choosing a
 * round number.
 *
 * A slow path rarer than one call in a thousand is not separable from
 * scheduling here and the gate does not pretend to hold it. The run's
 * own maximum is printed for that reason: it is evidence a reader can
 * weigh rather than a threshold this code can defend. */
#define TAIL_RATIO_CEILING       25.0    /* 99th percentile */
#define DEEP_TAIL_RATIO_CEILING  40.0    /* 99.9th percentile */

/* Sample counts. The first gives a 99.9th percentile with samples to
 * spare; the second is smaller because each of its calls costs more. */
#define SAMPLES_SMALL        50000
#define SAMPLES_LARGE        5000

/* ---- allocator counters --------------------------------------------
 *
 * Linked with --wrap, so every allocator call in this binary and in
 * the library archive it links passes through here. Calls made inside
 * the C library are not counted, and are not on the decision path:
 * the evaluator's own contract is that it performs no I/O. */
/* Volatile because the compiler treats the standard allocator entry
 * points as unable to touch a variable of this file's, so it would
 * otherwise read a counter it had cached across the call the wrapper
 * increments, and the arm would compare a stale number with itself. */
static volatile long g_malloc, g_calloc, g_realloc, g_free;

extern void *__real_malloc(size_t size);
extern void *__real_calloc(size_t count, size_t size);
extern void *__real_realloc(void *ptr, size_t size);
extern void  __real_free(void *ptr);

void *__wrap_malloc(size_t size);
void *__wrap_calloc(size_t count, size_t size);
void *__wrap_realloc(void *ptr, size_t size);
void  __wrap_free(void *ptr);

void *__wrap_malloc(size_t size)
{
    g_malloc++;
    return __real_malloc(size);
}

void *__wrap_calloc(size_t count, size_t size)
{
    g_calloc++;
    return __real_calloc(count, size);
}

void *__wrap_realloc(void *ptr, size_t size)
{
    g_realloc++;
    return __real_realloc(ptr, size);
}

void __wrap_free(void *ptr)
{
    g_free++;
    __real_free(ptr);
}

/* ---- this file's own encoder ----------------------------------------
 *
 * The gate writes the bytes itself rather than calling anything the
 * library ships, so a layout the reader gets wrong cannot be cancelled
 * by a writer that gets it wrong the same way. The weights are
 * generated from a counter rather than read from anywhere, because
 * this gate measures how long an evaluation takes and not what it
 * returns, and a policy whose numbers are reproducible from a line of
 * arithmetic keeps the file self-contained. */

static unsigned char *g_buf;
static size_t g_at, g_cap;

static void emit_(const void *p, size_t n)
{
    if (g_at + n > g_cap) {
        g_cap = (g_at + n) * 2u + 4096u;
        g_buf = (unsigned char *)realloc(g_buf, g_cap);
        ASSERT(g_buf != NULL);
    }
    memcpy(g_buf + g_at, p, n);
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

    b[0] = (unsigned char)(v & 0xFFu);
    b[1] = (unsigned char)((v >> 8) & 0xFFu);
    b[2] = (unsigned char)((v >> 16) & 0xFFu);
    b[3] = (unsigned char)((v >> 24) & 0xFFu);
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

/* A weight from its position alone, small and of both signs, so that
 * the hidden layers neither saturate every unit nor sit at zero. */
static double weight_at_(uint64_t k)
{
    return (double)(int64_t)(k % 401u) * 0.001 - 0.2;
}

/* Build a policy of `depth` hidden layers of `hidden` units between an
 * observation of `obs` channels and an action of `act` channels, with
 * standardisation and a clamp, and return its bytes. The caller frees
 * them. */
static unsigned char *build_policy_(uint32_t obs, uint32_t act,
                                    uint32_t hidden, uint32_t depth,
                                    size_t *out_len,
                                    unsigned long long *out_params)
{
    static const char PROVENANCE[] = "a policy shaped for a timing gate";
    static const unsigned char ZEROS[K26RL_SHA256_BYTES] = { 0 };
    K26RlSha256 sha;
    unsigned char digest[K26RL_SHA256_BYTES];
    unsigned long long params = 0;
    uint32_t layers = depth + 1u;
    uint32_t i, o, k, in_width;
    uint64_t counter = 0;
    unsigned char *bytes;

    g_at = 0;
    emit_(K26RL_POLICY_MAGIC, 8);
    emit_u32_(K26RL_POLICY_FORMAT_VERSION);
    emit_u32_(K26RL_POLICY_HEADER_BYTES);
    emit_(ZEROS, K26RL_SHA256_BYTES);
    emit_u32_(K26RL_POLICY_FLAG_STANDARDISE | K26RL_POLICY_FLAG_CLAMP);
    emit_u32_(1u);              /* agent count */
    emit_u32_(0u);              /* agent index */
    emit_u32_(obs);             /* observation total */
    emit_u32_(act);             /* action total */
    emit_u32_(0u);              /* observation slice offset */
    emit_u32_(obs);             /* observation slice width */
    emit_u32_(0u);              /* action slice offset */
    emit_u32_(act);             /* action slice width */
    emit_u32_(layers);
    emit_u32_((uint32_t)(sizeof PROVENANCE - 1u));
    emit_u32_(0u);              /* reserved */
    emit_(PROVENANCE, sizeof PROVENANCE - 1u);

    in_width = obs;
    for (i = 0; i < layers; i++) {
        uint32_t out_width = (i + 1u == layers) ? act : hidden;
        uint16_t activation = (i + 1u == layers) ? K26RL_POLICY_ACT_IDENTITY
                                                 : K26RL_POLICY_ACT_TANH;

        emit_u32_(in_width);
        emit_u32_(out_width);
        emit_u16_(activation);
        emit_u16_(0u);
        for (o = 0; o < out_width; o++)
            for (k = 0; k < in_width; k++)
                emit_f64_(weight_at_(counter++));
        for (o = 0; o < out_width; o++)
            emit_f64_(weight_at_(counter++));
        params += (unsigned long long)in_width * out_width + out_width;
        in_width = out_width;
    }

    for (i = 0; i < obs; i++)
        emit_f64_(0.25);                        /* means */
    for (i = 0; i < obs; i++)
        emit_f64_(1.5);                         /* variances */
    emit_f64_(1e-8);                            /* epsilon */
    emit_f64_(10.0);                            /* clip */
    for (i = 0; i < act; i++)
        emit_f64_(-1.0);                        /* lower bounds */
    for (i = 0; i < act; i++)
        emit_f64_(1.0);                         /* upper bounds */

    k26rl_sha256_init(&sha);
    k26rl_sha256_update(&sha, g_buf, (uint64_t)K26RL_POLICY_DIGEST_OFFSET);
    k26rl_sha256_update(&sha, ZEROS, (uint64_t)K26RL_SHA256_BYTES);
    k26rl_sha256_update(&sha,
                        g_buf + K26RL_POLICY_DIGEST_OFFSET
                              + K26RL_SHA256_BYTES,
                        (uint64_t)g_at - K26RL_POLICY_DIGEST_OFFSET
                                       - K26RL_SHA256_BYTES);
    k26rl_sha256_final(&sha, digest);
    memcpy(g_buf + K26RL_POLICY_DIGEST_OFFSET, digest, K26RL_SHA256_BYTES);

    bytes = (unsigned char *)malloc(g_at);
    ASSERT(bytes != NULL);
    memcpy(bytes, g_buf, g_at);
    *out_len = g_at;
    if (out_params)
        *out_params = params;
    return bytes;
}

/* ---- the distribution ----------------------------------------------- */

typedef struct {
    long   count;
    double min, p50, p90, p99, p999, max, mean;
} Distribution;

static int compare_double_(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;

    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

/* The sample at a quantile of an already sorted array, by nearest
 * rank. A distribution reported from fewer samples than a quantile
 * can resolve would be a number with nothing behind it, so the caller
 * is required to bring enough. */
static double quantile_(const double *sorted, long n, double q)
{
    long i = (long)(q * (double)(n - 1) + 0.5);

    if (i < 0)
        i = 0;
    if (i >= n)
        i = n - 1;
    return sorted[i];
}

/* Summarise raw samples. The array is copied before sorting so the
 * caller keeps the order it measured in. */
static void summarise_(const double *samples, long n, Distribution *out)
{
    double *sorted;
    double sum = 0.0;
    long i;

    ASSERT(n >= 1000);          /* enough for the 99.9th percentile */
    sorted = (double *)malloc(sizeof(double) * (size_t)n);
    ASSERT(sorted != NULL);
    memcpy(sorted, samples, sizeof(double) * (size_t)n);
    qsort(sorted, (size_t)n, sizeof(double), compare_double_);
    for (i = 0; i < n; i++)
        sum += samples[i];
    out->count = n;
    out->min  = sorted[0];
    out->p50  = quantile_(sorted, n, 0.50);
    out->p90  = quantile_(sorted, n, 0.90);
    out->p99  = quantile_(sorted, n, 0.99);
    out->p999 = quantile_(sorted, n, 0.999);
    out->max  = sorted[n - 1];
    out->mean = sum / (double)n;
    free(sorted);
}

static void report_(const char *tag, const Distribution *d)
{
    printf("  %-28s n %6ld  min %8.3f  p50 %8.3f  p90 %8.3f  "
           "p99 %8.3f  p99.9 %8.3f  max %9.3f  (mean %8.3f) us\n",
           tag, d->count, d->min, d->p50, d->p90, d->p99, d->p999,
           d->max, d->mean);
}

static double now_us_(void)
{
    struct timespec t;

    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec * 1e-3;
}

/* Time `n` evaluations of one policy, one call per sample. The
 * observation and the output buffer are the caller's, so nothing in
 * the timed region touches the allocator. */
static void time_calls_(const K26RlPolicy *policy, const double *obs,
                        double *act, double *samples, long n)
{
    long i;

    for (i = 0; i < n; i++) {
        double t0 = now_us_();

        ASSERT(k26rl_policy_act(policy, obs, act) == K26RL_POLICY_OK);
        samples[i] = now_us_() - t0;
    }
}

/* An observation that exercises the standardisation and reaches every
 * channel, generated rather than tabulated for the same reason the
 * weights are. */
static void fill_observation_(double *obs, uint32_t width)
{
    uint32_t i;

    for (i = 0; i < width; i++)
        obs[i] = (double)(int32_t)(i % 17u) * 0.125 - 1.0;
}

/* ---- one measured policy -------------------------------------------- */

typedef struct {
    K26RlPolicy *policy;
    double      *obs;
    double      *act;
    double      *samples;
    long         sample_count;
    unsigned long long params;
    unsigned char *bytes;
    size_t         length;
    Distribution   distribution;
} Measured;

static void measure_(Measured *m, uint32_t obs_width, uint32_t act_width,
                     uint32_t hidden, uint32_t depth, long samples,
                     const char *tag)
{
    K26RlPolicyInfo info;
    long i;

    memset(m, 0, sizeof *m);
    m->bytes = build_policy_(obs_width, act_width, hidden, depth,
                             &m->length, &m->params);
    ASSERT(k26rl_policy_parse(m->bytes, (uint64_t)m->length, &m->policy) ==
           K26RL_POLICY_OK);
    ASSERT(k26rl_policy_info(m->policy, &info) == K26RL_POLICY_OK);
    ASSERT(info.obs_width == obs_width && info.act_width == act_width);

    m->obs = (double *)malloc(sizeof(double) * obs_width);
    m->act = (double *)malloc(sizeof(double) * act_width);
    m->samples = (double *)malloc(sizeof(double) * (size_t)samples);
    ASSERT(m->obs && m->act && m->samples);
    fill_observation_(m->obs, obs_width);
    m->sample_count = samples;

    /* Warm before measuring, so the distribution is the steady state a
     * driver runs in; the cost of starting cold is arm 6's business
     * and is measured there rather than smeared through this one. */
    for (i = 0; i < 2000; i++)
        ASSERT(k26rl_policy_act(m->policy, m->obs, m->act) ==
               K26RL_POLICY_OK);

    time_calls_(m->policy, m->obs, m->act, m->samples, samples);
    summarise_(m->samples, samples, &m->distribution);
    report_(tag, &m->distribution);
}

static void release_(Measured *m)
{
    k26rl_policy_close(m->policy);
    free(m->obs);
    free(m->act);
    free(m->samples);
    free(m->bytes);
}

/* ---- arm 1: the distribution, against the budget --------------------- */

static void arm_distribution_(const Measured *m)
{
    const Distribution *d = &m->distribution;

    printf("  budget %.0f us per decision; median ceiling %.0f us; "
           "tail ceilings %.0f and %.0f times the median at the 99th "
           "and 99.9th\n",
           DECISION_BUDGET_US, MEDIAN_CEILING_US, TAIL_RATIO_CEILING,
           DEEP_TAIL_RATIO_CEILING);
    printf("  median is %.2f per cent of the budget, "
           "99.9th percentile %.2f per cent\n",
           100.0 * d->p50 / DECISION_BUDGET_US,
           100.0 * d->p999 / DECISION_BUDGET_US);

    ASSERT(d->min > 0.0);
    ASSERT(d->p50 <= MEDIAN_CEILING_US);
    ASSERT(d->p99 <= TAIL_RATIO_CEILING * d->p50);
    ASSERT(d->p999 <= DEEP_TAIL_RATIO_CEILING * d->p50);
    /* The quantiles must be ordered, or the summary is not a summary
     * of these samples. */
    ASSERT(d->min <= d->p50 && d->p50 <= d->p90 && d->p90 <= d->p99 &&
           d->p99 <= d->p999 && d->p999 <= d->max);
    g_arms++;
}

/* ---- arm 2: the timing measures the evaluation ----------------------- */

static void arm_scale_(const Measured *small, const Measured *large)
{
    double ratio = large->distribution.p50 / small->distribution.p50;

    printf("  %llu parameters against %llu, median %.2f times slower\n",
           large->params, small->params, ratio);
    ASSERT(large->params > small->params * 10ull);
    ASSERT(ratio >= 5.0);
    g_arms++;
}

/* ---- arm 3: the summary reports the worst ---------------------------- */

static void arm_tail_is_reported_(const Measured *m)
{
    Distribution plain, spiked;
    double *with_spike;
    double spike;
    long n = m->sample_count;

    summarise_(m->samples, n, &plain);
    with_spike = (double *)malloc(sizeof(double) * (size_t)n);
    ASSERT(with_spike != NULL);
    memcpy(with_spike, m->samples, sizeof(double) * (size_t)n);

    /* One period that overran the budget outright, dropped among calls
     * that all met it. This is the run a driver must be able to see. */
    spike = DECISION_BUDGET_US * 1.5;
    with_spike[n / 3] = spike;
    summarise_(with_spike, n, &spiked);

    printf("  one slow sample of %.1f us among %ld: max %.3f -> %.3f, "
           "median %.3f -> %.3f, mean %.3f -> %.3f\n",
           spike, n, plain.max, spiked.max, plain.p50, spiked.p50,
           plain.mean, spiked.mean);

    /* The worst is the worst. */
    ASSERT(spiked.max == spike);
    /* The median does not notice, which is why a median alone is not a
     * report either. */
    ASSERT(spiked.p50 == plain.p50);
    /* And the mean moves by so little that a summary quoting it alone
     * would have hidden a missed deadline. */
    ASSERT(spiked.mean < plain.mean + 0.01 * spike);
    ASSERT(spiked.mean > plain.mean);
    free(with_spike);
    g_arms++;
}

/* ---- arm 4: no allocation on the decision path ----------------------- */

/* Kept out of the compiler's reach so the control allocation below is
 * really made. */
static volatile size_t g_control_size = 64;
static void *volatile g_control_sink;

static void arm_no_allocation_(const Measured *m)
{
    long before_malloc, before_calloc, before_realloc, before_free;
    long calls = 20000;
    long i;
    void *control;

    before_malloc = g_malloc;
    before_calloc = g_calloc;
    before_realloc = g_realloc;
    before_free = g_free;
    for (i = 0; i < calls; i++)
        ASSERT(k26rl_policy_act(m->policy, m->obs, m->act) ==
               K26RL_POLICY_OK);
    printf("  %ld evaluations: malloc %ld, calloc %ld, realloc %ld, "
           "free %ld\n", calls, g_malloc - before_malloc,
           g_calloc - before_calloc, g_realloc - before_realloc,
           g_free - before_free);
    ASSERT(g_malloc == before_malloc);
    ASSERT(g_calloc == before_calloc);
    ASSERT(g_realloc == before_realloc);
    ASSERT(g_free == before_free);
    g_arms++;

    /* The counter is live: an allocation made here, in the same region
     * and through the same wrapper, registers. Without this the zero
     * above would also be what a counter that never ran would print.
     * The size and the pointer are volatile so the compiler cannot
     * decide the pair is unobservable and remove it, which would make
     * this arm agree with a counter that never ran. */
    before_malloc = g_malloc;
    before_free = g_free;
    control = malloc((size_t)g_control_size);
    ASSERT(control != NULL);
    g_control_sink = control;
    free(control);
    g_control_sink = NULL;
    ASSERT(g_malloc == before_malloc + 1);
    ASSERT(g_free == before_free + 1);
    g_arms++;

    /* And the entry point that loads a policy does allocate, so the
     * wrapper is reaching inside the library and not only inside this
     * file. Which of the allocator entry points the loader reaches for
     * is its own business, so the arm counts all of them together. */
    {
        K26RlPolicy *fresh = NULL;
        long before_any = g_malloc + g_calloc + g_realloc;

        ASSERT(k26rl_policy_parse(m->bytes, (uint64_t)m->length, &fresh) ==
               K26RL_POLICY_OK);
        ASSERT(g_malloc + g_calloc + g_realloc > before_any);
        k26rl_policy_close(fresh);
    }
    g_arms++;
}

/* ---- arm 5: determinism --------------------------------------------- */

static void arm_determinism_(const Measured *m)
{
    K26RlPolicy *second = NULL;
    K26RlPolicyInfo info;
    double *first_result, *again, *from_env, *env_obs, *env_act;
    uint32_t width, act_width;
    long i;

    ASSERT(k26rl_policy_info(m->policy, &info) == K26RL_POLICY_OK);
    width = info.obs_width;
    act_width = info.act_width;

    first_result = (double *)malloc(sizeof(double) * act_width);
    again = (double *)malloc(sizeof(double) * act_width);
    from_env = (double *)malloc(sizeof(double) * act_width);
    env_obs = (double *)malloc(sizeof(double) * info.obs_total);
    env_act = (double *)malloc(sizeof(double) * info.act_total);
    ASSERT(first_result && again && from_env && env_obs && env_act);

    ASSERT(k26rl_policy_act(m->policy, m->obs, first_result) ==
           K26RL_POLICY_OK);

    /* Every call, not merely the second: an evaluator that carried
     * state from one call into the next would drift rather than
     * disagree at once. */
    for (i = 0; i < 100000; i++) {
        ASSERT(k26rl_policy_act(m->policy, m->obs, again) ==
               K26RL_POLICY_OK);
        ASSERT(memcmp(again, first_result,
                      sizeof(double) * act_width) == 0);
    }
    g_arms++;

    /* A second load of the same bytes is a second policy in this
     * process, which is how a driver runs more than one agent. It must
     * agree bit for bit rather than nearly. */
    ASSERT(k26rl_policy_parse(m->bytes, (uint64_t)m->length, &second) ==
           K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(second, m->obs, again) == K26RL_POLICY_OK);
    ASSERT(memcmp(again, first_result, sizeof(double) * act_width) == 0);
    /* Interleaving the two leaves each one's answer where it was. */
    ASSERT(k26rl_policy_act(m->policy, m->obs, again) == K26RL_POLICY_OK);
    ASSERT(memcmp(again, first_result, sizeof(double) * act_width) == 0);
    k26rl_policy_close(second);
    g_arms++;

    /* The whole-vector entry point is the one a driver holding an
     * environment's buffers reaches for. It must be the same
     * arithmetic and not a second one. */
    memcpy(env_obs, m->obs, sizeof(double) * width);
    ASSERT(k26rl_policy_act_env(m->policy, env_obs, env_act) ==
           K26RL_POLICY_OK);
    memcpy(from_env, env_act, sizeof(double) * act_width);
    ASSERT(memcmp(from_env, first_result,
                  sizeof(double) * act_width) == 0);
    g_arms++;

    free(first_result);
    free(again);
    free(from_env);
    free(env_obs);
    free(env_act);
}

/* ---- arm 6: the first decision --------------------------------------- */

/* Where this arm writes the policy it then loads from disk, so that it
 * goes through the entry point a driver uses rather than through the
 * one the rest of this file uses. */
#define WORK_DIR "/tmp/kflc_rl_policy_latency_test"

static void arm_first_decision_(void)
{
    unsigned char *bytes;
    size_t length;
    K26RlPolicy *policy = NULL;
    K26RlPolicyInfo info;
    double *obs, *act, *first_result;
    double early[8];
    char path[512];
    FILE *f;
    long i;

    bytes = build_policy_(13u, 15u, 64u, 2u, &length, NULL);
    /* An existing directory is as good as a fresh one here, so the
     * only failure worth refusing on is one that leaves it absent. */
    if (mkdir(WORK_DIR, 0700) != 0)
        ASSERT(errno == EEXIST);
    snprintf(path, sizeof path, "%s/first%s", WORK_DIR,
             K26RL_POLICY_SUFFIX);
    f = fopen(path, "wb");
    ASSERT(f != NULL);
    ASSERT(fwrite(bytes, 1, length, f) == length);
    ASSERT(fclose(f) == 0);
    ASSERT(k26rl_policy_open(path, &policy) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_info(policy, &info) == K26RL_POLICY_OK);
    obs = (double *)malloc(sizeof(double) * info.obs_width);
    act = (double *)malloc(sizeof(double) * info.act_width);
    first_result = (double *)malloc(sizeof(double) * info.act_width);
    ASSERT(obs && act && first_result);
    fill_observation_(obs, info.obs_width);

    for (i = 0; i < (long)(sizeof early / sizeof early[0]); i++) {
        double t0 = now_us_();

        ASSERT(k26rl_policy_act(policy, obs, act) == K26RL_POLICY_OK);
        early[i] = now_us_() - t0;
        if (i == 0)
            memcpy(first_result, act, sizeof(double) * info.act_width);
    }
    /* Reported and not gated. This process has already evaluated
     * millions of times, so its heap and its instruction cache are
     * warm and these numbers are the warm ones; a driver taking its
     * very first decision in a fresh process pays a page-in cost on
     * top, which is why the advice is to evaluate once before the
     * first period rather than to trust a number measured here. */
    printf("  first calls after loading from a file, us:");
    for (i = 0; i < (long)(sizeof early / sizeof early[0]); i++)
        printf(" %.2f", early[i]);
    printf("\n");

    /* Warm, then check that the cold answer was already the right
     * one. The first decision of a run is the one a driver cannot
     * take again. */
    for (i = 0; i < 20000; i++)
        ASSERT(k26rl_policy_act(policy, obs, act) == K26RL_POLICY_OK);
    ASSERT(memcmp(act, first_result, sizeof(double) * info.act_width) == 0);
    g_arms++;

    /* Even warm, the first call is inside the budget, so a driver that
     * did not warm up is late at worst and not broken. */
    ASSERT(early[0] < DECISION_BUDGET_US);
    g_arms++;

    k26rl_policy_close(policy);
    free(obs);
    free(act);
    free(first_result);
    free(bytes);
}

int main(void)
{
    Measured small, large;

    printf("test_rl_policy_latency: what one decision costs\n");

    measure_(&small, 13u, 15u, 64u, 2u, SAMPLES_SMALL,
             "13 in, 64 x 2, 15 out");
    arm_distribution_(&small);

    measure_(&large, 13u, 15u, 256u, 2u, SAMPLES_LARGE,
             "13 in, 256 x 2, 15 out");
    arm_scale_(&small, &large);

    arm_tail_is_reported_(&small);
    arm_no_allocation_(&small);
    arm_determinism_(&small);
    arm_first_decision_();

    release_(&small);
    release_(&large);
    free(g_buf);
    printf("test_rl_policy_latency: OK (%d arms)\n", g_arms);
    return 0;
}
