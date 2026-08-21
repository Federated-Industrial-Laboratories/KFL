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
 * Not driven here, and said rather than implied: the whole-call
 * abort statuses (out of memory, the floating-point mode race)
 * cannot be staged deterministically from a fixture, and their
 * pooled ordering is design-stated rather than gated; the counted
 * allocation and two-process arms run in their own suites, which the
 * retained evidence runs under a pooled environment as well.
 */
#define _GNU_SOURCE
#include <unistd.h>
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
    printf("test_rl_pool: %d arm(s) passed\n", g_arms);
    return 0;
}
