/* test_rl_31_pin.c: Grammar 3.1 pins beside the 3.2 reinforcement
 * learning work.
 *
 * Gates:
 *   1. Step-lowering pin: the 3.1 `step` statement still lowers to
 *      the clamped stepping entry `k26astro_world_step`, and the
 *      exact-dt entry appears nowhere in 3.1 emission. The exact-dt
 *      path belongs to the reinforcement learning episode machinery
 *      alone, entered through `k26rl_env_step`; that split was a
 *      deliberate decision when the exact-dt entry landed, and this
 *      gate makes rewiring it a conscious act.
 *   2. Constructs-removed pin: a reinforcement learning program with
 *      every 3.2 construct removed (episode, action, on_step,
 *      objective, the observe `as` clause, distribution values) is a
 *      plain 3.1 program: it checks clean and emits the ordinary
 *      batch program with no trace of the environment surface. The
 *      six scalar state keys stay valid with constant values, being
 *      an additive 3.1 extension.
 * The baseline byte-identity comparison against the pre-RL base
 * commit lives in test_rl_31_baseline, whose skip (history absent)
 * is the harness's own 77 rather than a silent pass; the pins here
 * carry no external dependency and always run.
 *
 * Pattern: run ./bin/kflc via system() on fixtures, assert on the
 * captured output (test_rl_grammar's harness style).
 */
#define _GNU_SOURCE
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#define WORK_DIR "/tmp/kflc_rl_31_pin_test"

#define BASE_COMMIT "53a4452"

static const char *const STEP31_KFL =
    "form STEP31\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    step 0.5\n"
    "    step 0.25\n"
    "end\n"
    "end\n";

/* A minimal reinforcement learning program, for the contrast half of
 * the step-lowering pin. */
static const char *const RL_MIN_KFL =
    "form RL_MIN\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 4\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "end\n"
    "end\n";

/* The round-trip fixture's world with every 3.2 construct removed;
 * the scalar state keys stay, with constant values. */
static const char *const RL_STRIPPED_KFL =
    "form RL_STRIPPED\n"
    "fn world pointing_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7350.0 vel_z=0.0\n"
    "    observe craft from earth mode=astrometric\n"
    "end\n"
    "end\n";

static void write_file_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    ASSERT(f != NULL);
    fputs(content, f);
    fclose(f);
}

static int run_(const char *cmd)
{
    int rc = system(cmd);
    if (rc < 0) return -1;
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* Whole file into a NUL-terminated heap buffer; caller frees. */
static char *slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    long sz = ftell(f);
    ASSERT(sz >= 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char *buf = malloc((size_t)sz + 1);
    ASSERT(buf != NULL);
    if (sz > 0) ASSERT(fread(buf, 1, (size_t)sz, f) == (size_t)sz);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static int count_(const char *hay, const char *needle)
{
    int n = 0;
    size_t nl = strlen(needle);
    for (const char *p = hay; (p = strstr(p, needle)) != NULL; p += nl) {
        n++;
    }
    return n;
}


/* Emit a fixture through the given kflc binary; returns the exit
 * code, with stdout and stderr captured to files. */
static int emit_(const char *kflc, const char *kfl, const char *out,
                 const char *err)
{
    char cmd[1024];
    int n = snprintf(cmd, sizeof cmd, "%s --emit %s > %s 2> %s",
                     kflc, kfl, out, err);
    ASSERT((size_t)n < sizeof cmd);
    return run_(cmd);
}

int main(void)
{
    ASSERT(run_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR) == 0);

    /* Gate 1: the step-lowering pin, both halves. */
    write_file_(WORK_DIR "/step31.kfl", STEP31_KFL);
    ASSERT(emit_("./bin/kflc", WORK_DIR "/step31.kfl",
                 WORK_DIR "/step31.cc", WORK_DIR "/step31.err") == 0);
    {
        char *cc = slurp_(WORK_DIR "/step31.cc");
        ASSERT(strstr(cc, "(void)k26astro_world_step(world, (0.5));")
               != NULL);
        ASSERT(strstr(cc, "(void)k26astro_world_step(world, (0.25));")
               != NULL);
        ASSERT(count_(cc, "k26astro_world_step(") == 2);
        ASSERT(strstr(cc, "k26astro_world_step_exact") == NULL);
        ASSERT(strstr(cc, "k26tick_advance_exact") == NULL);
        ASSERT(strstr(cc, "k26rl_") == NULL);
        free(cc);
    }
    write_file_(WORK_DIR "/rl_min.kfl", RL_MIN_KFL);
    ASSERT(emit_("./bin/kflc", WORK_DIR "/rl_min.kfl",
                 WORK_DIR "/rl_min.cc", WORK_DIR "/rl_min.err") == 0);
    {
        char *cc = slurp_(WORK_DIR "/rl_min.cc");
        ASSERT(strstr(cc, "k26astro_world_step_exact(") != NULL);
        /* The clamped entry never appears in the environment; the
         * substring below cannot match the exact-dt entry, whose name
         * continues with an underscore. */
        ASSERT(count_(cc, "k26astro_world_step(") == 0);
        ASSERT(strstr(cc, "k26rl_env_step") != NULL);
        free(cc);
    }
    printf("gate 1: 3.1 step lowers to the clamped entry; the"
           " exact-dt entry stays with the episode machinery: OK\n");

    /* Gate 2: constructs removed, a plain 3.1 program remains. */
    write_file_(WORK_DIR "/stripped.kfl", RL_STRIPPED_KFL);
    ASSERT(run_("./bin/kflc --check " WORK_DIR "/stripped.kfl"
                " 2> /dev/null") == 0);
    ASSERT(emit_("./bin/kflc", WORK_DIR "/stripped.kfl",
                 WORK_DIR "/stripped.cc", WORK_DIR "/stripped.err")
           == 0);
    {
        char *cc = slurp_(WORK_DIR "/stripped.cc");
        ASSERT(strstr(cc, "int main") != NULL);
        ASSERT(strstr(cc, "k26rl_") == NULL);
        ASSERT(strstr(cc, "k26astro_world_step_exact") == NULL);
        free(cc);
    }
    printf("gate 2: constructs removed leaves a plain 3.1 batch"
           " program: OK\n");

    /* Gate 3: the emitted reset path restores the create-time
     * baseline whole. The four restorations below are load-bearing
     * for re-simulation, but today's fixtures cannot all witness
     * them behaviourally (a two-body geometric observable reads
     * neither the epoch nor the integrator predictor carry), so
     * their presence is pinned textually, the step-lowering pin's
     * technique: deleting any of them is a red gate, not a silent
     * pass. */
    {
        char *cc = slurp_(WORK_DIR "/rl_min.cc");
        ASSERT(strstr(cc, "memcpy(b0, h->baseline") != NULL);
        ASSERT(strstr(cc, "g->t = h->baseline_t[e];") != NULL);
        ASSERT(strstr(cc, "g->dt_last = 0.0;") != NULL);
        ASSERT(strstr(cc, "g->ias15_dt_last = 0.0;") != NULL);
        ASSERT(strstr(cc, "k26astro_grav_ias15_reset(g);") != NULL);
        ASSERT(strstr(cc, "k26astro_world_set_seed(w, h->seed);")
               != NULL);
        free(cc);
    }
    /* Same technique for the create-time draw handoff: the world
     * build hands its attribute-draw values out through _kfl_dr0 and
     * the record step reads them back instead of re-drawing, so each
     * (seed, coordinate) pair is consumed once per create. A re-draw
     * would yield the identical value, so no behavioural gate can
     * see this; the pin can. */
    {
        static const char *const draws_kfl =
            "form RL_PIN_DRAWS\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft gm=1.0 parent=earth"
            " pos_x=uniform(6.6e6, 7.0e6) vel_y=7350.0\n"
            "    episode\n"
            "        control_dt 0.1\n"
            "        horizon 4\n"
            "    end\n"
            "    observe craft from earth mode=geometric as trk\n"
            "end\n"
            "end\n";
        write_file_(WORK_DIR "/pin_draws.kfl", draws_kfl);
        ASSERT(emit_("./bin/kflc", WORK_DIR "/pin_draws.kfl",
                     WORK_DIR "/pin_draws.cc",
                     WORK_DIR "/pin_draws.err") == 0);
        char *cc = slurp_(WORK_DIR "/pin_draws.cc");
        ASSERT(strstr(cc, "if (_kfl_dr0) _kfl_dr0[0] = _kfl_v;")
               != NULL);
        ASSERT(strstr(cc, "_kfl_dr0 ? _kfl_dr0[0] : (") != NULL);
        free(cc);
    }
    printf("gate 3: the emitted reset restores the baseline whole"
           " (epoch, step sizes, predictor carry, noise stream), and"
           " create consumes each draw coordinate once: OK\n");

    printf("test_rl_31_pin: 3 gates passed\n");
    return 0;
}
