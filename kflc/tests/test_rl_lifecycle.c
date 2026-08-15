/* test_rl_lifecycle.c: environment handles over the life of one
 * process, and the buffer sizes the frozen getters require.
 *
 * The rest of the drive gates take one handle each, or take their
 * handles one at a time in fresh child processes. Nothing measured
 * what happens when several handles, and several environments inside
 * one handle, are built and torn down in one process over a program
 * that binds more than one vehicle. That is the shape in which a
 * vehicle bound to a body pointer taken before the body set was
 * complete showed up: `k26astro_world_body_at` returns a pointer into
 * the world's body array, stable across everything except a further
 * `k26astro_world_add_body`, which reallocates it
 * (k26astro_rt/world.h). A vehicle bound too early therefore read and
 * wrote a freed block for the rest of its life, which reached the
 * caller as an integrator divergence from the third world onwards and,
 * at other heap layouts, as heap corruption.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   A fixture with one assembly-bound body declared last has no body
 *   added after its bind, so it cannot show the defect at all. This
 *   one declares two assembly-bound bodies and a plain body after
 *   both, so every vehicle in it has a body added after its bind.
 *
 *   A fixture whose attitude never advances would not exercise the
 *   writer that reaches the stale pointer. The gate therefore asserts
 *   that both vehicles' orientations move across the drive, before it
 *   asserts anything about faults or identity.
 *
 *   Handles that are created and destroyed without stepping show
 *   nothing: create alone succeeded throughout. Every handle here is
 *   stepped, and the stream is compared step by step.
 *
 *   A stream identity arm over a fixture whose observations never
 *   change would pass on any two runs at all. The liveness assertion
 *   above is what rules that out, and it runs first.
 *
 *   A sizing arm at one environment would pass an implementation that
 *   writes a fixed single element. The two unsized getters are
 *   measured at one environment and at eight, with guard elements on
 *   both sides of the required span and a sentinel inside it, so a
 *   short write and a long write are both visible.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_lifecycle_test"

static int n_pass = 0;

/* Wall-clock guard: an artifact can livelock, and a gate that hangs
 * stops the suite from reporting at all. The handler writes with
 * write(2) and ends with _exit, neither of which needs stdio to be
 * reentrant. */
static const char *rl_stage_name_ = "startup";

static void rl_deadline_fired_(int sig)
{
    (void)sig;
    const char *a = "test_rl_lifecycle: DEADLINE EXCEEDED at stage: ";
    (void)!write(2, a, strlen(a));
    (void)!write(2, rl_stage_name_, strlen(rl_stage_name_));
    (void)!write(2, "\n", 1);
    _exit(1);
}

static void rl_stage_(const char *name, unsigned secs)
{
    rl_stage_name_ = name;
    alarm(secs);
}

static void rl_stage_done_(void)
{
    alarm(0);
    rl_stage_name_ = "between stages";
}

/* The fixture. Two bodies bind an assembly, so two vehicles are
 * created per environment, and a plain body follows both of them, so
 * neither vehicle is the last thing added to its world. The assembly
 * carries a collider, which is the configuration the defect was found
 * in; the two are forty metres apart with no closing speed, so the
 * broadphase runs every sub-advance and no contact occurs, and the
 * arms below measure the handle lifecycle rather than the collision
 * resolution. Both bodies carry an initial rate about every axis, so
 * the attitude advance writes through both vehicles on every
 * sub-advance. */
static const char *const LC_ASM =
    "assembly lifecycle_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    "end\n";

static const char *const LC_KFL =
    "form RL_LIFECYCLE\n"
    "fn world lc_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " ephem_naif_id=399\n"
    "    astro_body alpha assembly=\"lifecycle.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0"
    " quat_w=1.0 omega_x=0.01 omega_y=0.02 omega_z=0.03\n"
    "    astro_body beta assembly=\"lifecycle.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_z=-40.0 vel_y=7546.0"
    " quat_w=1.0 omega_x=-0.02 omega_y=0.01 omega_z=0.02\n"
    /* The plain body after both assemblies: it is what makes the
     * second vehicle's bind precede an add as well as the first's. */
    "    astro_body probe mass=1.0 parent=earth"
    " pos_x=7.2e6 vel_y=7440.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 200\n"
    "        substeps 8\n"
    "    end\n"
    "    action idle box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        alpha.omega_x = alpha.omega_x + idle * 0.0\n"
    "    end\n"
    "    observe attitude of alpha as aatt\n"
    "    observe attitude of beta as batt\n"
    "    observe alpha from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* Observation layout: seven attitude channels for alpha, seven for
 * beta, then five tracking channels. */
#define LC_OBS      19
#define LC_A_QUAT_W  0
#define LC_B_QUAT_W  7

enum { LC_HANDLES = 4, LC_STEPS = 60, LC_WIDE = 8 };

/* Drive one handle for LC_STEPS steps, writing the observation of
 * environment `e_of` into `stream` (LC_STEPS * LC_OBS doubles) and
 * failing on any per-environment fault. */
static void lc_drive_(const RlSurface *s, K26RlEnv *env, uint32_t n_envs,
                      uint32_t e_of, double *stream)
{
    double  *act = (double *)calloc(n_envs, sizeof(double));
    double  *obs = (double *)calloc((size_t)n_envs * LC_OBS, sizeof(double));
    uint32_t *fl = (uint32_t *)calloc(n_envs, sizeof(uint32_t));
    uint16_t *fc = (uint16_t *)calloc(n_envs, sizeof(uint16_t));
    ASSERT(act && obs && fl && fc);

    for (int t = 0; t < LC_STEPS; t++) {
        ASSERT(s->step(env, act) == K26RL_OK);
        ASSERT(s->obs(env, obs) == K26RL_OK);
        ASSERT(s->flags(env, fl) == K26RL_OK);
        ASSERT(s->fault_codes(env, fc) == K26RL_OK);
        for (uint32_t e = 0; e < n_envs; e++) {
            if ((fl[e] & 4u) || fc[e] != 0u) {
                fprintf(stderr,
                        "FAIL %s:%d: env %u faulted at step %d,"
                        " flags 0x%08x code %u\n",
                        __FILE__, __LINE__, e, t + 1, fl[e],
                        (unsigned)fc[e]);
                exit(1);
            }
        }
        memcpy(stream + (size_t)t * LC_OBS,
               obs + (size_t)e_of * LC_OBS, LC_OBS * sizeof(double));
    }
    free(act); free(obs); free(fl); free(fc);
}

int main(void)
{
    /* Line buffered, so an arm's output reaches the log as it is
     * produced rather than at exit. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGALRM, rl_deadline_fired_);

    if (!rl_libs_present_("test_rl_lifecycle")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/lifecycle.k26asm", LC_ASM);
    rl_write_file_(WORK_DIR "/lc.kfl", LC_KFL);

    rl_stage_("compiling the lifecycle artifact", 900u);
    rl_compile_(WORK_DIR "/lc.kfl", WORK_DIR "/lc", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/lc.rlenv.so"));
    rl_stage_done_();

    void *so = rl_dlopen_(WORK_DIR "/lc.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    double *first = (double *)calloc(LC_STEPS * LC_OBS, sizeof(double));
    double *again = (double *)calloc(LC_STEPS * LC_OBS, sizeof(double));
    ASSERT(first && again);

    /* ---- 1. The fixture moves what the defect corrupted ---------- */
    rl_stage_("driving the first handle", 300u);
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
        lc_drive_(&s, env, 1u, 0u, first);
        s.destroy(env);

        double da = fabs(first[(LC_STEPS - 1) * LC_OBS + LC_A_QUAT_W]
                         - first[LC_A_QUAT_W]);
        double db = fabs(first[(LC_STEPS - 1) * LC_OBS + LC_B_QUAT_W]
                         - first[LC_B_QUAT_W]);
        printf("gate 1: %d steps, alpha quat_w moved %.6g,"
               " beta quat_w moved %.6g\n", LC_STEPS, da, db);
        ASSERT(da > 1e-9);
        ASSERT(db > 1e-9);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 1: both vehicles' attitudes advance over the drive: OK\n");

    /* ---- 2. Handles after the first are the same handle ---------- */
    rl_stage_("driving the later handles", 600u);
    {
        for (int h = 2; h <= LC_HANDLES; h++) {
            K26RlEnv *env = NULL;
            ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
            lc_drive_(&s, env, 1u, 0u, again);
            s.destroy(env);
            int same = memcmp(first, again,
                              (size_t)LC_STEPS * LC_OBS * sizeof(double)) == 0;
            printf("gate 2: handle %d of %d: fault-free, stream identical"
                   " to handle 1: %s\n", h, LC_HANDLES, same ? "yes" : "NO");
            ASSERT(same);
        }
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 2: %d handles in one process, no fault and no drift: OK\n",
           LC_HANDLES);

    /* ---- 3. Worlds after the second inside one handle ------------ *
     *
     * The same count of worlds, reached the other way. The defect
     * appeared at the third world of a process however the worlds were
     * distributed between handles, so a gate that only took handles
     * one environment at a time would miss the same defect arriving
     * inside a single wide handle. */
    rl_stage_("driving one wide handle", 600u);
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(11u, (uint32_t)LC_WIDE, &env) == K26RL_OK);
        /* The last environment of the handle: the furthest from the
         * first world, which is the one the defect left alone. */
        lc_drive_(&s, env, (uint32_t)LC_WIDE, (uint32_t)LC_WIDE - 1u, again);
        s.destroy(env);
        int same = memcmp(first, again,
                          (size_t)LC_STEPS * LC_OBS * sizeof(double)) == 0;
        printf("gate 3: %d environments in one handle: fault-free,"
               " environment %d's stream identical to the single"
               " environment's: %s\n", LC_WIDE, LC_WIDE - 1,
               same ? "yes" : "NO");
        ASSERT(same);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 3: a wide handle's later worlds are the first world: OK\n");

    /* ---- 4. What the two unsized getters write ------------------- *
     *
     * `k26rl_env_flags` and `k26rl_env_fault_codes` take a pointer
     * with no capacity beside it, and the contract is that the caller
     * provides one element per environment. This arm pins the
     * artifact's half of that contract: exactly n_envs elements
     * written, none before the buffer and none after it. */
    rl_stage_("measuring the getter spans", 300u);
    {
        static const uint32_t widths[] = { 1u, (uint32_t)LC_WIDE };
        const uint32_t guard32 = 0xDEADBEEFu;
        const uint16_t guard16 = 0xBEEFu;

        for (int w = 0; w < 2; w++) {
            uint32_t n = widths[w];
            K26RlEnv *env = NULL;
            ASSERT(s.create(21u + n, n, &env) == K26RL_OK);

            /* Four guard elements each side of the required span. */
            uint32_t *fl = (uint32_t *)malloc((n + 8) * sizeof(uint32_t));
            uint16_t *fc = (uint16_t *)malloc((n + 8) * sizeof(uint16_t));
            ASSERT(fl && fc);
            for (uint32_t i = 0; i < n + 8; i++) {
                fl[i] = guard32;
                fc[i] = guard16;
            }
            ASSERT(s.flags(env, fl + 4) == K26RL_OK);
            ASSERT(s.fault_codes(env, fc + 4) == K26RL_OK);

            for (uint32_t i = 0; i < 4; i++) {
                ASSERT(fl[i] == guard32);
                ASSERT(fl[4 + n + i] == guard32);
                ASSERT(fc[i] == guard16);
                ASSERT(fc[4 + n + i] == guard16);
            }
            for (uint32_t i = 0; i < n; i++) {
                ASSERT(fl[4 + i] != guard32);
                ASSERT(fc[4 + i] != guard16);
                ASSERT(fc[4 + i] == 0u);
            }
            printf("gate 4: n_envs %u: flags and fault codes wrote"
                   " exactly %u element(s), guards intact\n", n, n);

            ASSERT(s.flags(env, NULL) == K26RL_E_NULL);
            ASSERT(s.fault_codes(env, NULL) == K26RL_E_NULL);
            free(fl);
            free(fc);
            s.destroy(env);
        }
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 4: the unsized getters write one element per"
           " environment and refuse a null buffer: OK\n");

    free(first);
    free(again);
    dlclose(so);
    printf("test_rl_lifecycle: %d gates passed\n", n_pass);
    return 0;
}
