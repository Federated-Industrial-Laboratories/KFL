/* test_att_substep.c - the attitude advance's sub-step rule is a pure
 * function of the state it is integrating.
 *
 * WHY THIS GATE EXISTS. The advance subdivides its own interval by
 * the rate it is given, so how much arithmetic it performs is decided
 * by the state. That is admissible only while the decision is a pure
 * function of that state: the same state must give the same count on
 * every run, in every process and on every host, or two runs of one
 * artifact stop agreeing bit for bit and every claim built on that
 * agreement goes with them. A rule that consulted a clock, counted
 * its own calls, carried an error estimate between calls, read
 * anything outside its arguments, or spent a budget of work would
 * each break it, and none of them would announce itself: the run
 * would simply produce a different trajectory the next time, or on
 * the next machine, which is exactly the class of defect that is
 * cheapest to prevent and most expensive to find.
 *
 * WHAT IT MEASURES.
 *   1. Repetition. One state evaluated many times gives one answer.
 *   2. History. A state's answer does not change because other
 *      states were evaluated between two asks.
 *   3. Order. A battery evaluated forwards, backwards and through a
 *      fixed permutation gives each state the same answer.
 *   4. Elapsed time. The battery re-evaluated after the wall clock
 *      has crossed at least two second boundaries gives the same
 *      answers, so a rule reading a clock at any granularity down to
 *      the second is caught.
 *   5. Process. The same battery, and the same driven advance,
 *      digested in child processes that differ from this one in
 *      process identity, environment block and stack layout, gives
 *      the same digest in every one of them.
 *   6. The advance, not only the count. Every arm above is applied to
 *      the quaternion and rate bits a driven tumble leaves behind,
 *      because a pure count consumed impurely would still be a defect.
 *   8. The defect itself, as a conserved quantity: a torque-free
 *      body turning a radian in a declared interval keeps the
 *      magnitude of its world-frame angular momentum to one part in
 *      a million, which without the subdivision it holds to two
 *      parts in a hundred.
 *   7. The rule is the rule it claims to be: below the declared angle
 *      the count is one and the advance is bit for bit the single
 *      step it was before it subdivided anything, and above it the
 *      chosen count holds the angle turned in one sub-interval inside
 *      that bound.
 *
 * WHY IT CAN FAIL. The rule is the only symbol in its translation
 * unit, so a red control that defines the same symbol is linked in
 * its place and the library's copy is never extracted from the
 * archive. Seven red controls are built from this same source and the
 * wiring requires each to be rejected by the named arm, run on its
 * own, rather than by whichever arm happens to run first:
 *
 *     1  reads a clock                     rejected by  elapsed
 *     2  counts its own calls              rejected by  repetition
 *     3  carries an error between calls    rejected by  history
 *     4  reads an input outside the state  rejected by  process
 *     5  spends a budget of work           rejected by  order
 *     6  never subdivides                  rejected by  bound
 *     6  never subdivides                  rejected by  tumble
 *     7  subdivides ordinary flight        rejected by  unchanged
 *
 * so every arm of this gate is shown failing on a defect it names.
 * A gate that only shows two runs agreeing proves nothing about a
 * rule nobody made impure.
 *
 * Wire: see kflc/Makefile.
 */
#define _GNU_SOURCE
#include "k26astro_att/att.h"

#include "k26astro_body/attitude.h"
#include "k26astro_body/body.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#define REJECT(arm, ...) do { \
    fprintf(stderr, "FAIL: %s\n  ", (arm)); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    exit(1); } while (0)

/* ---- the red controls -------------------------------------------- *
 *
 * Each of these defines the library's own rule symbol, so the linker
 * satisfies the advance's reference from here and never extracts the
 * library's translation unit. Each is impure in exactly one of the
 * ways the header forbids, and each must make this gate fail.
 */
#ifdef ATT_SUBSTEP_INJECT

static int inject_base_(K26V3 w, double dt)
{
    if (!(dt > 0.0) || !isfinite(dt)) return 1;
    double theta = sqrt(w.x * w.x + w.y * w.y + w.z * w.z) * dt;
    if (!isfinite(theta)) return 1;
    if (theta <= K26ASTRO_ATT_SUBSTEP_ANGLE) return 1;
    double want = ceil(theta / K26ASTRO_ATT_SUBSTEP_ANGLE);
    if (!(want < (double)K26ASTRO_ATT_SUBSTEP_MAX)) {
        return K26ASTRO_ATT_SUBSTEP_MAX;
    }
    return (int)want;
}

int k26astro_att_substep_count(K26V3 omega_body, double dt)
{
    int m = inject_base_(omega_body, dt);
#if ATT_SUBSTEP_INJECT == 1
    /* Reads a clock. */
    if ((long)time(NULL) % 2 == 0) m += 1;
#elif ATT_SUBSTEP_INJECT == 2
    /* Counts its own calls. */
    static unsigned long calls = 0;
    if (++calls % 7u == 0u) m += 1;
#elif ATT_SUBSTEP_INJECT == 3
    /* Carries an error estimate that depends on what arrived before.
     * The increment is bounded so that one state that is not a number
     * cannot poison the accumulator and hide the impurity, which is
     * how an unbounded version of this control passed its own arm. */
    static double carried = 0.0;
    carried += 1.0 / (1.0 + (double)m);
    if (carried > 60.0) m += 1;
#elif ATT_SUBSTEP_INJECT == 4
    /* Reads an input outside the state being integrated. It spends
     * one extra sub-interval, which is finer than the bound asks for
     * and so passes every arm that reads the arithmetic; only running
     * it in another process shows what it is. */
    if (m > 1 && m < K26ASTRO_ATT_SUBSTEP_MAX &&
        (long)getpid() % 2 == 0) {
        m += 1;
    }
#elif ATT_SUBSTEP_INJECT == 5
    /* Spends a budget of work. */
    static long budget = 4000;
    if (budget <= 0) return 1;
    budget -= m;
#elif ATT_SUBSTEP_INJECT == 6
    /* Pure, and not the rule: it never subdivides, which is the
     * behaviour that was there before this rule existed. */
    m = 1;
#elif ATT_SUBSTEP_INJECT == 7
    /* Pure, and not the rule: it subdivides everything, including the
     * flight that was owed its trajectory unchanged. */
    m = (m > 2) ? m : 2;
#else
#error "ATT_SUBSTEP_INJECT must name one of the seven red controls"
#endif
    if (m < 1) m = 1;
    if (m > K26ASTRO_ATT_SUBSTEP_MAX) m = K26ASTRO_ATT_SUBSTEP_MAX;
    return m;
}
#endif /* ATT_SUBSTEP_INJECT */

/* ---- a digest ----------------------------------------------------- */

/* FNV-1a over raw bytes. A digest rather than a comparison because
 * the process arm has to carry a whole battery's answers back across
 * a pipe, and one number that changes when any answer changes is what
 * fits. */
static double v3len_(K26V3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static uint64_t fnv_(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ULL;
    }
    return h;
}

#define FNV_SEED 1469598103934665603ULL

/* ---- the battery -------------------------------------------------- *
 *
 * States chosen to cover every branch of the rule and both sides of
 * its bound: rates far below it, rates astride it, rates deep past
 * it, rates that reach the ceiling, and rates and intervals that are
 * not numbers at all. The spread is generated from a fixed generator
 * written here rather than drawn from a library, so the battery is
 * the same battery in every process without carrying a table.
 */
#define N_BATTERY 512

typedef struct {
    K26V3  omega;
    double dt;
} Probe;

static Probe battery_[N_BATTERY];

static double lcg_(uint64_t *s)
{
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((*s >> 11) & 0x1FFFFFFFFFFFFFULL) / 9007199254740992.0;
}

static void build_battery_(void)
{
    static const Probe fixed[] = {
        { { 0.0, 0.0, 0.0 },       0.1 },
        { { 0.0, 0.0, 0.0 },       0.0 },
        { { 1.0, 0.0, 0.0 },      -0.1 },
        { { 1.0, 0.0, 0.0 },       0.0 },
        { { 0.5, 0.0, 0.0 },       0.1 },   /* exactly at the bound */
        { { 0.5000001, 0.0, 0.0 }, 0.1 },   /* one hair past it */
        { { 0.4999999, 0.0, 0.0 }, 0.1 },   /* one hair inside it */
        { { 70.0, 0.0, 0.0 },      0.1 },
        { { 0.0, -120.0, 33.0 },   0.1 },
        { { 1.0e6, 0.0, 0.0 },     1.0 },   /* at the ceiling */
        { { 1.0e300, 0.0, 0.0 },   1.0 },   /* overflows to the ceiling */
        { { 0.0, 0.0, 0.0 },       INFINITY },
    };
    size_t n_fixed = sizeof fixed / sizeof fixed[0];
    ASSERT(n_fixed < N_BATTERY);
    for (size_t i = 0; i < n_fixed; i++) battery_[i] = fixed[i];

    /* A rate that is not a number, written without a literal so the
     * gate does not depend on how a compiler folds one. */
    double zero = 0.0;
    battery_[n_fixed].omega = k26m3d_v3(zero / zero, 0.0, 0.0);
    battery_[n_fixed].dt = 0.1;
    n_fixed++;

    uint64_t s = 0x5DEECE66DULL;
    for (size_t i = n_fixed; i < N_BATTERY; i++) {
        /* Rates spanning nine decades, so the count spans one to the
         * ceiling and every branch between. */
        double scale = pow(10.0, lcg_(&s) * 9.0 - 4.0);
        battery_[i].omega = k26m3d_v3((lcg_(&s) - 0.5) * 2.0 * scale,
                                      (lcg_(&s) - 0.5) * 2.0 * scale,
                                      (lcg_(&s) - 0.5) * 2.0 * scale);
        battery_[i].dt = 0.001 + lcg_(&s) * 0.999;
    }
}

static void evaluate_all_(int *out)
{
    for (int i = 0; i < N_BATTERY; i++) {
        out[i] = k26astro_att_substep_count(battery_[i].omega,
                                            battery_[i].dt);
    }
}

/* ---- a driven tumble --------------------------------------------- *
 *
 * The count is only half of it: a pure count consumed impurely is
 * still a defect, so the arms below digest what the advance actually
 * leaves in the body. The vehicle is spun up by a constant body
 * torque through rates that cross the bound, so most of the run is
 * subdivided and the digest is a digest of the subdivided path.
 */
static uint64_t tumble_digest_(void)
{
    K26AstroBody body;
    K26AstroVehicle *v = k26astro_vehicle_new();
    ASSERT(v != NULL);
    k26astro_vehicle_set_dry_mass(v, 10000.0);
    k26astro_vehicle_set_inertia_diag(v, 17000.0, 42000.0, 42000.0);
    k26astro_body_init(&body);
    k26astro_vehicle_bind_body(v, &body);

    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    ASSERT(a != NULL);
    a->omega_body = k26m3d_v3(0.02, 0.11, -0.07);
    body.omega    = a->omega_body;
    body.attitude = a->q;

    uint64_t h = FNV_SEED;
    K26V3 torque = k26m3d_v3(2000.0, 12000.0, -9000.0);
    for (int i = 0; i < 400; i++) {
        K26AstroAttStatus st = k26astro_att_step(v, torque, 0.1);
        ASSERT(st == K26ASTRO_ATT_OK);
        h = fnv_(h, &body.attitude, sizeof body.attitude);
        h = fnv_(h, &body.omega, sizeof body.omega);
    }
    k26astro_vehicle_destroy(v);
    return h;
}

static uint64_t battery_digest_(void)
{
    int counts[N_BATTERY];
    evaluate_all_(counts);
    return fnv_(FNV_SEED, counts, sizeof counts);
}

/* The one number a child process reports: the battery and the driven
 * tumble together, so a difference in either is a difference here. */
static uint64_t whole_digest_(void)
{
    uint64_t h = battery_digest_();
    uint64_t t = tumble_digest_();
    return fnv_(h, &t, sizeof t);
}

/* ---- arms ---------------------------------------------------------- */

static int n_pass_ = 0;

static void arm_repetition_(void)
{
    int first[N_BATTERY];
    evaluate_all_(first);
    for (int rep = 0; rep < 64; rep++) {
        int again[N_BATTERY];
        evaluate_all_(again);
        for (int i = 0; i < N_BATTERY; i++) {
            if (again[i] != first[i]) {
                REJECT("repetition",
                       "state %d gave %d and then %d: the count is not a "
                       "function of the state alone", i, first[i], again[i]);
            }
        }
    }
    printf("  repetition: %d states asked %d times each, one answer "
           "each: OK\n", N_BATTERY, 65);
    n_pass_++;
}

static void arm_history_(void)
{
    /* One probe, asked before and after a long run of other states.
     * A rule accumulating anything across calls answers differently
     * the second time. */
    for (int i = 0; i < N_BATTERY; i++) {
        int before = k26astro_att_substep_count(battery_[i].omega,
                                                battery_[i].dt);
        for (int j = 0; j < N_BATTERY; j++) {
            (void)k26astro_att_substep_count(battery_[j].omega,
                                             battery_[j].dt);
        }
        int after = k26astro_att_substep_count(battery_[i].omega,
                                               battery_[i].dt);
        if (before != after) {
            REJECT("history",
                   "state %d gave %d, then %d after %d other states were "
                   "asked: the rule carries something between calls",
                   i, before, after, N_BATTERY);
        }
    }
    printf("  history: every state's answer survives %d other states "
           "asked between the two asks: OK\n", N_BATTERY);
    n_pass_++;
}

static void arm_order_(void)
{
    int forward[N_BATTERY], backward[N_BATTERY], shuffled[N_BATTERY];
    for (int i = 0; i < N_BATTERY; i++) {
        forward[i] = k26astro_att_substep_count(battery_[i].omega,
                                                battery_[i].dt);
    }
    for (int i = N_BATTERY - 1; i >= 0; i--) {
        backward[i] = k26astro_att_substep_count(battery_[i].omega,
                                                 battery_[i].dt);
    }
    /* A fixed permutation, which is a third arrival order and not a
     * reversal of the first. */
    for (int k = 0; k < N_BATTERY; k++) {
        int i = (int)(((long)k * 137L + 41L) % N_BATTERY);
        shuffled[i] = k26astro_att_substep_count(battery_[i].omega,
                                                 battery_[i].dt);
    }
    for (int i = 0; i < N_BATTERY; i++) {
        if (forward[i] != backward[i] || forward[i] != shuffled[i]) {
            REJECT("order",
                   "state %d gave %d asked forwards, %d backwards and %d "
                   "permuted: the answer depends on what arrived when",
                   i, forward[i], backward[i], shuffled[i]);
        }
    }
    printf("  order: three arrival orders over %d states, one answer "
           "each: OK\n", N_BATTERY);
    n_pass_++;
}

static void arm_elapsed_(void)
{
    /* The digest is taken again and again across a span longer than
     * two whole seconds, so a rule reading a clock at any granularity
     * from a second down to the sampling interval has read a
     * different clock between two of these samples. Sampling rather
     * than comparing two instants is the point: two instants two
     * seconds apart share the parity of the second, and a rule
     * keyed on that parity would pass a two-instant comparison while
     * giving a different answer in between.
     */
    uint64_t first = whole_digest_();
    time_t t0 = time(NULL);
    int samples = 1;
    for (;;) {
        struct timespec ts = { 0, 50L * 1000L * 1000L };
        (void)nanosleep(&ts, NULL);
        uint64_t now = whole_digest_();
        samples++;
        if (now != first) {
            REJECT("elapsed time",
                   "the digest was %016llx and is %016llx after %d "
                   "samples: something in the decision reads a clock",
                   (unsigned long long)first, (unsigned long long)now,
                   samples);
        }
        if (time(NULL) >= t0 + 3) break;
    }
    printf("  elapsed time: digest %016llx over %d samples spanning "
           "three second boundaries: OK\n",
           (unsigned long long)first, samples);
    n_pass_++;
}

/* Run this same binary in a child, in a different environment, and
 * read back the digest it prints. */
static uint64_t child_digest_(const char *self, char *const envp[])
{
    int fd[2];
    ASSERT(pipe(fd) == 0);
    pid_t pid = fork();
    ASSERT(pid >= 0);
    if (pid == 0) {
        (void)close(fd[0]);
        (void)dup2(fd[1], STDOUT_FILENO);
        (void)close(fd[1]);
        char *const argv[] = { (char *)self, (char *)"--digest", NULL };
        execve(self, argv, envp);
        _exit(127);
    }
    (void)close(fd[1]);
    char buf[64];
    size_t got = 0;
    ssize_t n;
    while (got + 1 < sizeof buf &&
           (n = read(fd[0], buf + got, sizeof buf - 1 - got)) > 0) {
        got += (size_t)n;
    }
    buf[got] = '\0';
    (void)close(fd[0]);
    int status = 0;
    ASSERT(waitpid(pid, &status, 0) == pid);
    ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    unsigned long long value = 0;
    ASSERT(sscanf(buf, "%llx", &value) == 1);
    return (uint64_t)value;
}

static void arm_process_(const char *self)
{
    /* Four children, whose process identities are consecutive and so
     * cannot all share the parity of this process, in environments
     * that differ from this one and from each other, and whose stacks
     * are laid out afresh. A rule reading any of those three answers
     * differently in at least one of them. */
    char *env_a[] = { (char *)"PATH=/usr/bin:/bin", NULL };
    char *env_b[] = { (char *)"PATH=/usr/bin:/bin",
                      (char *)"K26_GATE_PADDING=0123456789", NULL };
    char *env_c[] = { NULL };
    char *env_d[] = { (char *)"PATH=/bin",
                      (char *)"K26_GATE_PADDING=abc",
                      (char *)"K26_GATE_SECOND=xyz", NULL };
    char *const *envs[4];
    envs[0] = env_a; envs[1] = env_b; envs[2] = env_c; envs[3] = env_d;

    uint64_t mine = whole_digest_();
    for (int i = 0; i < 4; i++) {
        uint64_t theirs = child_digest_(self, envs[i]);
        if (theirs != mine) {
            REJECT("process",
                   "this process digests %016llx and child %d digests "
                   "%016llx: the decision reads something outside the "
                   "state, which differs between processes",
                   (unsigned long long)mine, i,
                   (unsigned long long)theirs);
        }
    }
    printf("  process: digest %016llx in this process and in four "
           "children differing in identity, environment and layout: OK\n",
           (unsigned long long)mine);
    n_pass_++;
}

static void arm_bound_(void)
{
    /* The rule is the rule it claims to be. Below the declared angle
     * the count is one; above it the count holds the angle turned in
     * one sub-interval inside the bound, unless the ceiling is what
     * decided the count, which the ceiling's own arm covers. */
    int at_one = 0, subdivided = 0, at_ceiling = 0;
    for (int i = 0; i < N_BATTERY; i++) {
        K26V3 w = battery_[i].omega;
        double dt = battery_[i].dt;
        int m = k26astro_att_substep_count(w, dt);
        ASSERT(m >= 1 && m <= K26ASTRO_ATT_SUBSTEP_MAX);
        double theta = sqrt(w.x * w.x + w.y * w.y + w.z * w.z) * dt;
        if (!isfinite(theta) || !(dt > 0.0)) {
            if (m != 1) {
                REJECT("bound",
                       "state %d is not a state a subdivision can rescue "
                       "and the rule spent %d sub-intervals on it", i, m);
            }
            at_one++;
            continue;
        }
        if (theta <= K26ASTRO_ATT_SUBSTEP_ANGLE) {
            if (m != 1) {
                REJECT("bound",
                       "state %d turns %.17g rad, inside the bound of "
                       "%.17g, "
                       "and the rule spent %d sub-intervals on it",
                       i, theta, (double)K26ASTRO_ATT_SUBSTEP_ANGLE, m);
            }
            at_one++;
            continue;
        }
        if (m == K26ASTRO_ATT_SUBSTEP_MAX) { at_ceiling++; continue; }
        double per = theta / (double)m;
        if (!(per <= K26ASTRO_ATT_SUBSTEP_ANGLE)) {
            REJECT("bound",
                   "state %d turns %.17g rad in one sub-interval of %d, "
                   "past the bound of %.17g",
                   i, per, m, (double)K26ASTRO_ATT_SUBSTEP_ANGLE);
        }
        subdivided++;
    }
    /* A count that is arithmetically right over a battery that never
     * leaves one branch measures nothing, so the branches are counted
     * and required. */
    ASSERT(at_one > 0 && subdivided > 0 && at_ceiling > 0);
    printf("  bound: %d states inside the angle take one sub-interval, "
           "%d subdivide within it, %d reach the ceiling: OK\n",
           at_one, subdivided, at_ceiling);
    n_pass_++;
}

static void arm_unchanged_below_(void)
{
    /* What an existing program is owed: below the bound the advance
     * is the single step it always was, bit for bit. The reference is
     * the body library's own step applied once over the whole
     * interval, which is what the advance did before it subdivided
     * anything. */
    K26AstroBody body;
    K26AstroVehicle *v = k26astro_vehicle_new();
    ASSERT(v != NULL);
    k26astro_vehicle_set_dry_mass(v, 10000.0);
    k26astro_vehicle_set_inertia_diag(v, 17000.0, 42000.0, 42000.0);
    k26astro_body_init(&body);
    k26astro_vehicle_bind_body(v, &body);
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    ASSERT(a != NULL);

    K26AstroAttitudeStateExt ref;
    k26astro_attitude_init_ext(&ref, a->inertia);

    K26V3 w0 = k26m3d_v3(0.03, -0.12, 0.09);   /* 0.0153 rad at dt 0.1 */
    a->omega_body = w0;
    body.omega    = w0;
    body.attitude = a->q;
    ref.omega_body = w0;
    ref.q          = a->q;

    K26V3 torque = k26m3d_v3(3.0, -7.0, 2.0);
    for (int i = 0; i < 200; i++) {
        int m = k26astro_att_substep_count(body.omega, 0.1);
        if (m != 1) {
            REJECT("unchanged below the bound",
                   "step %d turns %.17g rad, inside the bound of %.17g, "
                   "and the rule spent %d sub-intervals on it: ordinary "
                   "flight is paying for a provision it does not need",
                   i, v3len_(body.omega) * 0.1,
                   (double)K26ASTRO_ATT_SUBSTEP_ANGLE, m);
        }
        ASSERT(k26astro_att_step(v, torque, 0.1) == K26ASTRO_ATT_OK);
        /* The reference normalises first because the advance does,
         * and the claim is about the advance as it stood and not
         * about the body library's step in isolation. */
        ref.q = k26m3d_quat_norm(ref.q);
        k26astro_attitude_step_torque_ext(&ref, torque, 0.1);
        if (memcmp(&body.attitude, &ref.q, sizeof ref.q) != 0 ||
            memcmp(&body.omega, &ref.omega_body,
                   sizeof ref.omega_body) != 0) {
            REJECT("unchanged below the bound",
                   "step %d differs from the single step it was: a "
                   "program whose craft stays inside the bound has had "
                   "its trajectory moved", i);
        }
    }
    k26astro_attitude_destroy_ext(&ref);
    k26astro_vehicle_destroy(v);
    printf("  unchanged below the bound: 200 advances bit for bit equal "
           "to the single step they were: OK\n");
    n_pass_++;
}

/* The defect this rule was built for, reduced to a property with a
 * closed form behind it. A torque-free body conserves the magnitude
 * of its angular momentum in the world frame exactly, whatever it is
 * doing in its own frame, so anything that happens to it here is the
 * integrator's. At a rate whose declared interval turns the body a
 * whole radian at a time, the step is asked to follow a curve it has
 * only sampled a small part of, and the error it makes is the error
 * it then integrates.
 *
 * What the bound below is worth stating carefully. When this arm was
 * written the step underneath was first order, and at this rate it
 * did not merely lose accuracy without the subdivision: it left the
 * number line by the hundred and sixtieth interval, and the arm's
 * bound of five was chosen to sit between that and the 2.8 the rule
 * delivered. The step is higher order now, and on this fixture it
 * survives the horizon whether it is subdivided or not, so a bound of
 * five would pass on a rule that never subdivided anything and this
 * arm would have stopped measuring what it names.
 *
 * The bound is therefore set where the two now separate, which is
 * still four decades apart. Measured on this exact fixture, ten
 * radians a second at a tenth of a second over three hundred
 * intervals: 6.7283e-09 with the rule and 2.5049e-02 without it, a
 * factor of three and a half million. The bound of one part in a
 * million leaves the rule a hundred and fifty times its measured
 * worst and rejects the control that never subdivides by four
 * decades, which is what a red control is for.
 */
static void arm_tumble_(void)
{
    K26AstroBody body;
    K26AstroVehicle *v = k26astro_vehicle_new();
    ASSERT(v != NULL);
    k26astro_vehicle_set_dry_mass(v, 10000.0);
    k26astro_vehicle_set_inertia_diag(v, 17000.0, 42000.0, 42000.0);
    k26astro_body_init(&body);
    k26astro_vehicle_bind_body(v, &body);
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    ASSERT(a != NULL);

    K26V3 w0 = k26m3d_v3(9.5, 3.0, 1.0);       /* 10.01 rad/s */
    a->omega_body = w0;
    body.omega    = w0;
    body.attitude = a->q;

    const double dt = 0.1;                     /* one radian a time */
    const int    n  = 300;                     /* thirty seconds */
    int m0 = k26astro_att_substep_count(w0, dt);
    K26V3 h0;
    ASSERT(k26astro_att_momentum_world(v, &h0) == K26ASTRO_ATT_OK);
    double h0_mag = v3len_(h0);

    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        K26AstroAttStatus st = k26astro_att_step(v, k26m3d_v3(0, 0, 0), dt);
        if (st != K26ASTRO_ATT_OK) {
            REJECT("tumble",
                   "the advance reported %s at interval %d of %d: a "
                   "rotation the declared interval cannot resolve is "
                   "still not being resolved",
                   k26astro_att_status_str(st), i, n);
        }
        K26V3 h;
        ASSERT(k26astro_att_momentum_world(v, &h) == K26ASTRO_ATT_OK);
        double drift = fabs(v3len_(h) - h0_mag) / h0_mag;
        if (drift > worst) worst = drift;
    }
    /* Measured at 6.7283e-09 with this rule in place, against
     * 2.5049e-02 without it. */
    if (!(worst < 1.0e-6)) {
        REJECT("tumble",
               "angular momentum drifted by %.4e over %d intervals at "
               "%.2f rad/s, past the stated one part in a million",
               worst, n, v3len_(w0));
    }
    k26astro_vehicle_destroy(v);
    printf("  tumble: %.2f rad/s at dt %.2f is %d sub-intervals, and "
           "angular momentum holds to %.4e over %d intervals, where "
           "one sub-interval holds it to 2.5e-02: OK\n",
           v3len_(w0), dt, m0, worst, n);
    n_pass_++;
}

typedef struct {
    const char *name;
    void (*run)(void);
} Arm;

static const char *self_ = NULL;

static void arm_process_wrapper_(void) { arm_process_(self_); }

int main(int argc, char **argv)
{
    static const Arm arms[] = {
        { "repetition", arm_repetition_ },
        { "history",    arm_history_ },
        { "order",      arm_order_ },
        { "bound",      arm_bound_ },
        { "unchanged",  arm_unchanged_below_ },
        { "elapsed",    arm_elapsed_ },
        { "process",    arm_process_wrapper_ },
        { "tumble",     arm_tumble_ },
    };
    const int n_arms = (int)(sizeof arms / sizeof arms[0]);

    build_battery_();
    self_ = argv[0];

    if (argc > 1 && strcmp(argv[1], "--digest") == 0) {
        printf("%016llx\n", (unsigned long long)whole_digest_());
        return 0;
    }

    /* One arm at a time, so the wiring can require a named class of
     * impurity to be caught by the arm that names it rather than by
     * whichever arm happens to run first. */
    if (argc > 1 && strncmp(argv[1], "--arm=", 6) == 0) {
        const char *want = argv[1] + 6;
        for (int i = 0; i < n_arms; i++) {
            if (strcmp(arms[i].name, want) == 0) {
                printf("test_att_substep: arm %s\n", want);
                arms[i].run();
                return 0;
            }
        }
        fprintf(stderr, "test_att_substep: no arm named %s\n", want);
        return 2;
    }

    printf("test_att_substep: the sub-step rule is a pure function of "
           "the state\n");
    for (int i = 0; i < n_arms; i++) arms[i].run();
    printf("test_att_substep: %d arms passed\n", n_pass_);
    return 0;
}
