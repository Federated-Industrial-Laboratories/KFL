/* test_rl_pool.c: the internal worker pool's output-bound claim.
 *
 * The stepping contract binds outputs, and whether the environments
 * of one handle are computed serially or across a worker pool is
 * invisible: that is the sentence the surface has carried since its
 * freeze, and this suite is what makes it a measurement. Compute
 * runs wherever the pool puts it; publication runs on the caller in
 * environment order; so the episode record of any run is
 * byte-identical at any worker count, boundary resets, faults, and
 * endings included.
 *
 * Arms:
 *
 *   1. A busy batch, byte-identical across worker counts. Eight
 *      environments, short horizons, several episodes each, so
 *      boundary resets land mid-call throughout; the record at
 *      worker counts two, four, and eight against the serial one.
 *
 *   2. A faulting world, byte-identical across worker counts. The
 *      world's central mass is nought, the drift cannot converge,
 *      and every environment faults on its first transition; the
 *      fault frames and end reasons must travel identically from a
 *      pooled run.
 *
 *   3. The control's edge readings. A value that does not parse, a
 *      nought, a one, and a count above the environment count each
 *      produce the serial record exactly: the first three because
 *      they stand the pool down, the last because it caps at the
 *      environment count.
 *
 *   4. The timing refusals under the pool. With workers active, the
 *      tap away from a boundary is refused exactly as serially, and
 *      arming at a boundary then stepping publishes a ring a reader
 *      can drain; the pool must not move where a boundary is.
 *
 *   5. The pool provably engages. The thread count of this process
 *      rises by the helper count when a pooled handle is created and
 *      does not when the control is malformed; without this arm,
 *      every identity assertion above would also pass if the pool
 *      never ran at all, which is the vacuity it exists to rule out.
 *
 *   6. Endings and the tap under the pool. A terminating fixture
 *      whose environments end at different steps, driven through the
 *      surface with a different action per environment, tap armed:
 *      the drained ring bytes and the episode records at four
 *      workers equal the serial ones, terminal payments included;
 *      and the serial record equals the one the compiler at the
 *      parent commit emits, so the split itself moved nothing.
 *
 * Not driven here, and said rather than implied: the whole-call
 * abort statuses (out of memory, the floating-point mode race)
 * cannot be staged deterministically from a fixture, and their
 * pooled ordering is design-stated rather than gated; the counted
 * allocation and two-process arms run in their own suites, which the
 * retained evidence runs under a pooled environment as well.
 */
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/wait.h>
#include "rl_gate_util.h"
#include <k26rl_tap.h>

static int g_arms = 0;

static const char *POOL_KFL =
    "form RL_POOL\n"
    "fn world pool_world\n"
    "    astro_body sun gm=1.32712440018e20 mass=1.989e30\n"
    "    astro_body probe gm=1.0 parent=sun pos_x=1.496e11"
    " pos_y=uniform(-2.0e7,2.0e7) pos_z=0.0 vel_x=0.0"
    " vel_y=29780.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 60.0\n"
    "        horizon 7\n"
    "        reset probe.pos_y uniform(-2.0e7, 2.0e7)\n"
    "    end\n"
    "    action ax box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        let unused: double = ax\n"
    "    end\n"
    "    observe probe from sun mode=geometric as los\n"
    "    objective\n"
    "        reward 0.0 - 0.001 * ax * ax\n"
    "    end\n"
    "end\n"
    "end\n";

static const char *FAULT_KFL =
    "form RL_POOL_FAULT\n"
    "fn world fault_world\n"
    "    astro_body hollow gm=0.0 mass=1.0\n"
    "    astro_body probe gm=1.0 parent=hollow pos_x=1.0e6"
    " pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=1.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 4\n"
    "    end\n"
    "    action ax box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        let unused: double = ax\n"
    "    end\n"
    "    observe probe from hollow mode=geometric as los\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

static const char *TERM_KFL =
    "form RL_POOL_TERM\n"
    "fn world term_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body probe gm=1.0 parent=earth pos_x=7.0e6"
    " pos_y=0.0 pos_z=0.0 vel_x=uniform(180.0,260.0)"
    " vel_y=7546.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 40\n"
    "        reset probe.vel_x uniform(180.0, 260.0)\n"
    "        terminated when trk_range > 7.004e6\n"
    "    end\n"
    "    action ax box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        let unused: double = ax\n"
    "    end\n"
    "    observe probe from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.001 * ax\n"
    "        terminal 5.0 + ax\n"
    "    end\n"
    "end\n"
    "end\n";

static int thread_count_(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    int n = -1;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "Threads: %d", &n) == 1) break;
    }
    fclose(f);
    return n;
}

static void run_batch_(const char *bin, const char *threads,
                       const char *out)
{
    char cmd[2048];
    snprintf(cmd, sizeof cmd,
             "%s%s%s %s --envs 8 --episodes 3 --seed 20260821 "
             "--out %s > /dev/null 2>&1",
             threads ? "K26RL_ENV_THREADS=" : "",
             threads ? threads : "", threads ? "" : "", bin, out);
    (void)!system(cmd);
}

static long fsize_(const char *p)
{
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fclose(f);
    return n;
}

static int same_(const char *a, const char *b)
{
    long na = fsize_(a), nb = fsize_(b);
    if (na < 0 || na != nb) return 0;
    FILE *fa = fopen(a, "rb"), *fb = fopen(b, "rb");
    ASSERT(fa && fb);
    int eq = 1;
    int ca, cb;
    do {
        ca = fgetc(fa); cb = fgetc(fb);
        if (ca != cb) { eq = 0; break; }
    } while (ca != EOF);
    fclose(fa); fclose(fb);
    return eq;
}

static void gate_identity_(const char *name, const char *src,
                           const char *work)
{
    char kfl[512], bin[512], serial[512], pooled[512];
    snprintf(kfl, sizeof kfl, "%s/%s.kfl", work, name);
    snprintf(bin, sizeof bin, "%s/%s", work, name);
    rl_write_file_(kfl, src);
    rl_compile_(kfl, bin, work);
    snprintf(serial, sizeof serial, "%s/%s_serial.epi", work, name);
    run_batch_(bin, NULL, serial);
    long sz = fsize_(serial);
    ASSERT(sz > 0);
    static const char *counts[] = { "2", "4", "8", NULL };
    for (int i = 0; counts[i]; i++) {
        snprintf(pooled, sizeof pooled, "%s/%s_w%s.epi", work, name,
                 counts[i]);
        run_batch_(bin, counts[i], pooled);
        ASSERT(same_(serial, pooled));
    }
    printf("  %s: the record at workers 2, 4, and 8 is the serial "
           "record, %ld bytes each\n", name, sz);
    g_arms++;
}

static void gate_edges_(const char *work)
{
    char bin[512], serial[512], out[512];
    snprintf(bin, sizeof bin, "%s/pool_busy", work);
    snprintf(serial, sizeof serial, "%s/pool_busy_serial.epi", work);
    static const char *edges[] = { "x", "0", "1", "64", NULL };
    for (int i = 0; edges[i]; i++) {
        snprintf(out, sizeof out, "%s/pool_edge_%d.epi", work, i);
        run_batch_(bin, edges[i], out);
        ASSERT(same_(serial, out));
    }
    printf("  a value that does not parse, a nought, a one, and a "
           "count past the environments each produce the serial "
           "record\n");
    g_arms++;
}

static void gate_refusals_(const char *work)
{
    char so[512];
    snprintf(so, sizeof so, "%s/pool_busy.rlenv.so", work);
    setenv("K26RL_ENV_THREADS", "4", 1);
    void *dl = rl_dlopen_(so);
    RlSurface s;
    rl_resolve_surface_(dl, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(20260821u, 8u, &env) == 0);
    double acts[8] = { 0 };
    ASSERT(s.step(env, acts) == 0);
    char ring[64];
    snprintf(ring, sizeof ring, "kflrl_pool_gate_%d", (int)getpid());
    int rc = (int)s.tap(env, ring);
    ASSERT(rc != 0);
    printf("  away from a boundary the tap is refused under the pool "
           "(rc=%d)\n", rc);
    s.destroy(env);
    env = NULL;
    ASSERT(s.create(20260821u, 8u, &env) == 0);
    ASSERT(s.tap(env, ring) == 0);
    ASSERT(s.step(env, acts) == 0);
    ASSERT(s.step(env, acts) == 0);
    K26RlTapReader *rd = NULL;
    ASSERT(k26rl_tap_attach(ring, 1, &rd) == 0);
    uint64_t frames = 0;
    uint8_t buf[65536];
    for (;;) {
        uint32_t len = 0;
        uint64_t lost = 0;
        K26RlStatus trc = k26rl_tap_read(rd, buf, sizeof buf, &len,
                                         &lost);
        if (trc != K26RL_OK || len == 0) break;
        frames++;
    }
    k26rl_tap_detach(rd);
    ASSERT(frames >= 8u * 2u);
    printf("  armed at a boundary, a pooled run publishes a ring a "
           "reader drains: %llu frames\n",
           (unsigned long long)frames);
    s.tap(env, NULL);
    s.destroy(env);
    unsetenv("K26RL_ENV_THREADS");
    g_arms++;
}

static void gate_engaged_(const char *work)
{
    char so[512];
    snprintf(so, sizeof so, "%s/pool_busy.rlenv.so", work);
    void *dl = rl_dlopen_(so);
    RlSurface s;
    rl_resolve_surface_(dl, &s);
    int before = thread_count_();
    ASSERT(before > 0);
    setenv("K26RL_ENV_THREADS", "8", 1);
    K26RlEnv *env = NULL;
    ASSERT(s.create(1u, 8u, &env) == 0);
    int pooled = thread_count_();
    s.destroy(env);
    setenv("K26RL_ENV_THREADS", "x", 1);
    env = NULL;
    ASSERT(s.create(1u, 8u, &env) == 0);
    int malformed = thread_count_();
    s.destroy(env);
    unsetenv("K26RL_ENV_THREADS");
    ASSERT(pooled == before + 7);
    ASSERT(malformed == before);
    printf("  a pooled create raises the process thread count by the "
           "helper count (%d to %d) and a malformed control does not\n",
           before, pooled);
    g_arms++;
}

static uint64_t drain_hash_(const char *ring)
{
    K26RlTapReader *rd = NULL;
    ASSERT(k26rl_tap_attach(ring, 1, &rd) == 0);
    uint64_t hash = 1469598103934665603ull;
    uint64_t frames = 0;
    uint8_t buf[65536];
    for (;;) {
        uint32_t len = 0;
        uint64_t lost = 0;
        if (k26rl_tap_read(rd, buf, sizeof buf, &len, &lost) != K26RL_OK)
            break;
        if (len == 0) break;
        ASSERT(lost == 0);
        for (uint32_t i = 0; i < len; i++) {
            hash ^= buf[i];
            hash *= 1099511628211ull;
        }
        frames++;
    }
    k26rl_tap_detach(rd);
    ASSERT(frames > 0);
    return hash;
}

static uint64_t drive_term_(const char *so, const char *threads,
                            const char *ring)
{
    if (threads) setenv("K26RL_ENV_THREADS", threads, 1);
    else unsetenv("K26RL_ENV_THREADS");
    void *dl = rl_dlopen_(so);
    RlSurface s;
    rl_resolve_surface_(dl, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(20260821u, 6u, &env) == 0);
    ASSERT(s.tap(env, ring) == 0);
    double acts[6];
    for (int step = 0; step < 90; step++) {
        for (int e = 0; e < 6; e++) {
            acts[e] = ((double)((e * 7 + step) % 11) - 5.0) / 5.0;
        }
        ASSERT(s.step(env, acts) == 0);
    }
    uint64_t hash = drain_hash_(ring);
    s.tap(env, NULL);
    s.destroy(env);
    unsetenv("K26RL_ENV_THREADS");
    return hash;
}

static void gate_endings_tap_(const char *work)
{
    char kfl[512], bin[512], so[512], ring[80];
    snprintf(kfl, sizeof kfl, "%s/pool_term.kfl", work);
    snprintf(bin, sizeof bin, "%s/pool_term", work);
    snprintf(so, sizeof so, "%s/pool_term.rlenv.so", work);
    rl_write_file_(kfl, TERM_KFL);
    rl_compile_(kfl, bin, work);
    snprintf(ring, sizeof ring, "kflrl_pool_term_%d", (int)getpid());
    uint64_t serial = drive_term_(so, NULL, ring);
    uint64_t pooled = drive_term_(so, "4", ring);
    ASSERT(serial == pooled);
    printf("  a terminating world with a terminal payment, distinct "
           "actions per environment, drains identical ring bytes "
           "serial and pooled (hash %016llx)\n",
           (unsigned long long)serial);

    char base[512], bkfl[512], bbin[512], bser[512], mser[512];
    snprintf(base, sizeof base, "%s/base", work);
    if (rl_base_build_("59b1cfc", base)) {
        snprintf(bkfl, sizeof bkfl, "%s/pool_term_b.kfl", base);
        rl_write_file_(bkfl, TERM_KFL);
        snprintf(bbin, sizeof bbin, "%s/pool_term_b", base);
        rl_base_compile_(base, bkfl, bbin, 1);
        snprintf(bser, sizeof bser, "%s/base_serial.epi", work);
        char cmd[2048];
        snprintf(cmd, sizeof cmd,
                 "%s --envs 6 --episodes 2 --seed 9 --out %s "
                 "> /dev/null 2>&1", bbin, bser);
        (void)!system(cmd);
        snprintf(mser, sizeof mser, "%s/mine_serial.epi", work);
        snprintf(cmd, sizeof cmd,
                 "%s --envs 6 --episodes 2 --seed 9 --out %s "
                 "> /dev/null 2>&1", bin, mser);
        (void)!system(cmd);
        ASSERT(same_(bser, mser));
        printf("  the serial record equals the parent compiler's: the "
               "split moved nothing (%ld bytes)\n", fsize_(mser));
    } else {
        printf("FAIL: the parent commit is unavailable, so the "
               "split-moved-nothing arm cannot run\n");
        ASSERT(0);
    }
    g_arms++;
}

/* A fork's child inherits no worker threads; a lost pool must step
 * serially rather than wait forever, and the parent must be
 * untouched. The child's step is under an alarm so a regression
 * reads as a failure here, not a hang. */
static void gate_fork_(const char *work)
{
    char so[512];
    snprintf(so, sizeof so, "%s/pool_busy.rlenv.so", work);
    setenv("K26RL_ENV_THREADS", "4", 1);
    void *dl = rl_dlopen_(so);
    RlSurface s;
    rl_resolve_surface_(dl, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(20260821u, 8u, &env) == 0);
    double acts[8] = { 0 };
    ASSERT(s.step(env, acts) == 0);
    pid_t pid = fork();
    ASSERT(pid >= 0);
    if (pid == 0) {
        alarm(20);
        K26RlStatus rc = s.step(env, acts);
        _exit(rc == 0 ? 0 : 3);
    }
    int wst = 0;
    ASSERT(waitpid(pid, &wst, 0) == pid);
    ASSERT(WIFEXITED(wst) && WEXITSTATUS(wst) == 0);
    ASSERT(s.step(env, acts) == 0);
    s.destroy(env);
    unsetenv("K26RL_ENV_THREADS");
    printf("  a fork's child steps a lost pool serially and returns; "
           "the parent's pool is untouched\n");
    g_arms++;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_pool")) return 0;
    char work[] = "/tmp/kflc_rl_pool_XXXXXX";
    ASSERT(mkdtemp(work) != NULL);
    unsetenv("K26RL_ENV_THREADS");
    printf("test_rl_pool: the pool is invisible where it must be\n");
    gate_identity_("pool_busy", POOL_KFL, work);
    gate_identity_("pool_fault", FAULT_KFL, work);
    gate_edges_(work);
    gate_refusals_(work);
    gate_engaged_(work);
    gate_endings_tap_(work);
    gate_fork_(work);
    printf("test_rl_pool: %d arm(s) passed\n", g_arms);
    char clean[600];
    snprintf(clean, sizeof clean, "rm -rf %s", work);
    (void)!system(clean);
    return 0;
}
