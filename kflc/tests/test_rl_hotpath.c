/* test_rl_hotpath.c: the stepping hot loop's no-allocation, no-I/O
 * contract (k26rl_env.h: "The stepping hot loop performs no
 * allocation and no I/O: the per-step path touches preallocated
 * memory only").
 *
 * Mechanism: the gate compiles a small interposer shared object and
 * re-executes itself under LD_PRELOAD of it, so malloc, calloc,
 * realloc, free, write, and fwrite from every module in the child
 * process (the dlopened .rlenv.so included) are counted while a
 * per-region arming flag is set. The counters are proven live by a
 * positive control: a deliberate malloc + free + write inside an
 * armed window must read exactly 1/1/1 (and a deliberate fwrite
 * exactly 1) before the zero assertions mean anything.
 *
 * Gates (serve surface, dlopen):
 *   1. Positive control, as above.
 *   2. Output disabled: create, reset, then 60 steps of 2
 *      environments with the counters armed across the steps only.
 *      The window crosses two horizon-24 truncation boundaries per
 *      environment, so boundary auto-resets are inside the pinned
 *      region. All six counters must read zero.
 *   3. Output enabled: same drive with an episode file attached.
 *      Allocation counters must still read zero; write activity is
 *      bounded, not zero: the writer emits one fwrite per finished
 *      frame, and within this window frames leave only at episode
 *      boundaries (steps_per_chunk is 1024, so no chunk fills
 *      mid-episode; each boundary flushes at most a partial chunk
 *      frame, an end record, and the next start record: 3 frames
 *      per boundary per environment). 2 environments x 2 boundaries
 *      x 3 frames gives 12; the bound asserts (0, 12] and at least
 *      one write-family call.
 *   4. Telemetry tap enabled, no consumer: the same drive again. The
 *      acceptance is equality with gate 2's counts rather than a
 *      bound of its own, because publication into a shared mapping is
 *      a store and not a system call. Both counts must therefore be
 *      exactly what the untapped drive read.
 *   5. Tap enabled with a consumer attached, reading continuously in
 *      its own process for the whole window: the same equality. This
 *      is where an implementation that published over a pipe, or that
 *      remapped or allocated per frame, or that waited on a consumer,
 *      would show up.
 *
 * Batch path: the batch executable shares the stepping machinery
 * with the serve surface by construction, and test_rl_determinism
 * gate 6 pins whole-file batch/serve equivalence; the batch binary
 * offers no hook for arming the counters around its stepping region
 * alone, so this gate scopes the interposition check to the serve
 * surface.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

#include "rl_gate_util.h"
#include "k26rl_tap.h"
#include <time.h>

#define WORK_DIR "/tmp/kflc_rl_hotpath_test"

/* Same two-body geometry as the determinism gate: the craft sits
 * inside the MERCURIUS transition window around earth, so every
 * step runs the paper-faithful split (WH far pass plus IAS15 near
 * pass), the deepest stepping path. Horizon 24 with no termination
 * predicate: episodes truncate on a fixed cadence inside the armed
 * window. The block writes body state, one velocity key and one
 * position key, so the state accessors and the position fold are
 * measured too. */
static const char *const HP_KFL =
    "form RL_HOTPATH\n"
    "fn world hp_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0,10.0) vel_z=0.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 24\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 10.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    action gear discrete 3 default 1\n"
    "    on_step\n"
    "        let scale: double = 1.0 + push * 0.001\n"
    /* One velocity key and one position key, so the position
     * write's sector-fold renormalisation sits inside the measured
     * window rather than beside it. */
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "        craft.pos_z = craft.pos_z + push\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range + push\n"
    "    end\n"
    "end\n"
    "end\n";

/* The interposer, compiled into WORK_DIR/interpose.so at gate time.
 * Allocation calls forward to the glibc-internal entry points;
 * write forwards to the kernel directly; fwrite forwards through a
 * constructor-resolved RTLD_NEXT pointer so no lookup happens while
 * armed. Counter slots: 0 malloc, 1 calloc, 2 realloc, 3 free,
 * 4 write, 5 fwrite. */
static const char *const HP_INTERPOSE_C =
    "#define _GNU_SOURCE\n"
    "#include <dlfcn.h>\n"
    "#include <stddef.h>\n"
    "#include <stdio.h>\n"
    "#include <sys/syscall.h>\n"
    "#include <unistd.h>\n"
    "extern void *__libc_malloc(size_t);\n"
    "extern void *__libc_calloc(size_t, size_t);\n"
    "extern void *__libc_realloc(void *, size_t);\n"
    "extern void __libc_free(void *);\n"
    "volatile unsigned long k26hp_counts[6];\n"
    "volatile int k26hp_armed;\n"
    "static size_t (*real_fwrite_)(const void *, size_t, size_t, FILE *);\n"
    "__attribute__((constructor)) static void k26hp_init_(void)\n"
    "{\n"
    "    real_fwrite_ = (size_t (*)(const void *, size_t, size_t, FILE *))\n"
    "        dlsym(RTLD_NEXT, \"fwrite\");\n"
    "}\n"
    "void *malloc(size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[0]++; return __libc_malloc(n); }\n"
    "void *calloc(size_t m, size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[1]++; return __libc_calloc(m, n); }\n"
    "void *realloc(void *p, size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[2]++; return __libc_realloc(p, n); }\n"
    "void free(void *p)\n"
    "{ if (k26hp_armed) k26hp_counts[3]++; __libc_free(p); }\n"
    "ssize_t write(int fd, const void *buf, size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[4]++;\n"
    "  return syscall(SYS_write, fd, buf, n); }\n"
    "size_t fwrite(const void *p, size_t sz, size_t n, FILE *f)\n"
    "{ if (k26hp_armed) k26hp_counts[5]++;\n"
    "  return real_fwrite_(p, sz, n, f); }\n";

enum { HP_ENVS = 2, HP_STEPS = 60, HP_ACT = 2 };

/* Write-family bound for the output-enabled window; derivation in
 * the header comment. */
#define HP_WRITE_BOUND 12ul

/* ---- Child: runs under LD_PRELOAD of the interposer ------------- */

static volatile unsigned long *counts_;
static volatile int *armed_;

static void counters_clear_(void)
{
    for (int i = 0; i < 6; i++) counts_[i] = 0;
}

static unsigned long alloc_total_(void)
{
    return counts_[0] + counts_[1] + counts_[2] + counts_[3];
}

static unsigned long write_total_(void)
{
    return counts_[4] + counts_[5];
}

static unsigned long untapped_writes;

static int child_main_(void)
{
    counts_ = (volatile unsigned long *)dlsym(RTLD_DEFAULT, "k26hp_counts");
    armed_  = (volatile int *)dlsym(RTLD_DEFAULT, "k26hp_armed");
    ASSERT(counts_ != NULL && armed_ != NULL);

    /* Gate 1: positive control. Prime stdio buffers and the /dev/null
     * descriptors before arming so only the deliberate calls count. */
    int devnull = open("/dev/null", O_WRONLY);
    ASSERT(devnull >= 0);
    FILE *fnull = fopen("/dev/null", "w");
    ASSERT(fnull != NULL);
    ASSERT(fwrite("p", 1, 1, fnull) == 1);
    printf("test_rl_hotpath: child up\n");
    fflush(stdout);

    counters_clear_();
    *armed_ = 1;
    void *volatile ctrl = malloc(32);
    free((void *)ctrl);
    ASSERT(write(devnull, "x", 1) == 1);
    ASSERT(fwrite("y", 1, 1, fnull) == 1);
    *armed_ = 0;
    ASSERT(counts_[0] == 1);   /* malloc */
    ASSERT(counts_[3] == 1);   /* free */
    ASSERT(counts_[4] == 1);   /* write */
    ASSERT(counts_[5] == 1);   /* fwrite */
    printf("gate 1: positive control malloc/free/write/fwrite"
           " = 1/1/1/1: OK\n");

    void *so = rl_dlopen_(WORK_DIR "/hp.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    static double act[HP_ENVS * HP_ACT];
    for (int e = 0; e < HP_ENVS; e++) {
        act[e * HP_ACT + 0] = 0.25;
        act[e * HP_ACT + 1] = 1.0;
    }

    /* Gate 2: output disabled. The armed window covers the steps
     * alone, first step after reset included (nothing may be
     * lazy-initialised on the step path). */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 2: %d steps x %d envs, output disabled:"
               " alloc-family %lu (malloc %lu calloc %lu realloc %lu"
               " free %lu), write-family %lu\n",
               HP_STEPS, HP_ENVS, a, counts_[0], counts_[1],
               counts_[2], counts_[3], w);
        ASSERT(a == 0);
        ASSERT(w == 0);
        /* Kept as the reference the tap arms below must equal. */
        untapped_writes = w;
        s.destroy(env);
    }
    printf("gate 2: zero allocations, zero writes: OK\n");

    /* Gate 3: output enabled. Steps still allocate nothing; the
     * writer's flushes stay within the frame bound derived in the
     * header comment. */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.output(env, WORK_DIR "/hp_out.k26epi") == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 3: %d steps x %d envs, output enabled:"
               " alloc-family %lu, write-family %lu (bound %lu)\n",
               HP_STEPS, HP_ENVS, a, w, HP_WRITE_BOUND);
        ASSERT(a == 0);
        ASSERT(w > 0);
        ASSERT(w <= HP_WRITE_BOUND);
        s.destroy(env);
    }
    printf("gate 3: zero allocations, bounded writes: OK\n");

    /* Gates 4 and 5: the telemetry tap costs the step path no
     * allocation and no system call, whether or not anyone is
     * listening. The acceptance is not a bound of its own: it is
     * equality with the untapped drive gate 2 just measured in this
     * same process, over the same fixture and the same window. An
     * implementation that published over a pipe, or remapped per
     * frame, would move the write-family count and fail here. */
    char tap_name[64];
    snprintf(tap_name, sizeof tap_name, "hotpath.%ld", (long)getpid());
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.tap(env, tap_name) == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 4: %d steps x %d envs, tap enabled, no consumer:"
               " alloc-family %lu, write-family %lu"
               " (untapped drive was %lu)\n",
               HP_STEPS, HP_ENVS, a, w, untapped_writes);
        ASSERT(a == 0);
        ASSERT(w == untapped_writes);
        s.destroy(env);
    }
    printf("gate 4: tap enabled costs no allocation and no syscall: OK\n");

    {
        K26RlEnv *env = NULL;
        pid_t consumer;
        int ready[2];
        char ready_byte = 0;

        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.tap(env, tap_name) == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        /* A real consumer, in its own process, reading continuously
         * for as long as the producer runs. Its own allocations are
         * its own process's and cannot reach these counters, which is
         * the point: a consumer is invisible to the producer.
         *
         * The parent waits for the consumer to report itself attached
         * before it arms and steps. Without that handshake the child
         * races the whole measured window and can lose it on a busy
         * machine, and a gate that claims to measure the cost with a
         * consumer attached must know that one is. */
        ASSERT(pipe(ready) == 0);
        consumer = fork();
        ASSERT(consumer >= 0);
        if (consumer == 0) {
            K26RlTapReader *r = NULL;
            uint32_t slot = 0;
            close(ready[0]);
            /* Attach with a wall-clock bound rather than a spin count,
             * so a slow start waits instead of giving up. */
            for (int tries = 0; tries < 10000 && !r; tries++) {
                struct timespec ts;
                if (k26rl_tap_attach(tap_name, 1, &r) == K26RL_OK)
                    break;
                ts.tv_sec = 0;
                ts.tv_nsec = 1000000;   /* 1 ms; 10 s in total */
                nanosleep(&ts, NULL);
            }
            if (!r)
                _exit(2);
            if (k26rl_tap_reader_info(r, &slot, NULL) != K26RL_OK)
                _exit(2);
            uint8_t *buf = (uint8_t *)malloc(slot);
            if (!buf)
                _exit(2);
            if (write(ready[1], "a", 1) != 1)
                _exit(2);
            close(ready[1]);
            for (;;) {
                uint32_t len = 0;
                if (k26rl_tap_read(r, buf, slot, &len, NULL) != K26RL_OK)
                    _exit(2);
                if (!len && k26rl_tap_reader_closed(r))
                    break;
            }
            _exit(0);
        }
        close(ready[1]);
        /* One byte, and only after the consumer holds a mapping. */
        ASSERT(read(ready[0], &ready_byte, 1) == 1);
        ASSERT(ready_byte == 'a');
        close(ready[0]);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 5: %d steps x %d envs, tap enabled, consumer"
               " attached: alloc-family %lu, write-family %lu"
               " (untapped drive was %lu)\n",
               HP_STEPS, HP_ENVS, a, w, untapped_writes);
        ASSERT(a == 0);
        ASSERT(w == untapped_writes);
        s.destroy(env);

        int cst = 0;
        ASSERT(waitpid(consumer, &cst, 0) == consumer);
        ASSERT(WIFEXITED(cst) && WEXITSTATUS(cst) == 0);
    }
    printf("gate 5: an attached consumer changes neither count: OK\n");

    dlclose(so);
    fclose(fnull);
    close(devnull);
    printf("test_rl_hotpath: 5 gates passed\n");
    return 0;
}

/* ---- Parent: compile fixture + interposer, re-exec under preload */

int main(void)
{
    if (getenv("K26RL_HOTPATH_CHILD")) return child_main_();

    if (!rl_libs_present_("test_rl_hotpath")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/hp.kfl", HP_KFL);
    rl_compile_(WORK_DIR "/hp.kfl", WORK_DIR "/hp", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/hp.rlenv.so"));

    rl_write_file_(WORK_DIR "/interpose.c", HP_INTERPOSE_C);
    rl_run_or_die_("cc -O2 -fPIC -shared -o " WORK_DIR "/interpose.so "
                   WORK_DIR "/interpose.c -ldl");

    char self[512];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    ASSERT(n > 0);
    self[n] = '\0';

    char cmd[1024];
    int m = snprintf(cmd, sizeof cmd,
                     "env LD_PRELOAD=" WORK_DIR "/interpose.so"
                     " K26RL_HOTPATH_CHILD=1 %s", self);
    ASSERT((size_t)m < sizeof cmd);
    int rc = system(cmd);
    ASSERT(rc != -1);
    ASSERT(WIFEXITED(rc));
    return WEXITSTATUS(rc);
}
