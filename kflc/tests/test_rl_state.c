/* test_rl_state.c: gates for body state in on_step and for the
 * range-rate observation component.
 *
 * Gates:
 *   1. Write before advance. Two drives of one fixture at one seed
 *      whose action streams first differ at step t produce observation
 *      streams that first differ at step t itself, not at t + 1. The
 *      step index of first divergence is the assertion, so a write
 *      applied one step late fails here even though it would keep
 *      every determinism gate green.
 *   2. The same ordering, analytically, on a single-step fixture: the
 *      observer sits at the world origin and the target's unwritten
 *      position components are zero, so the reported range is the
 *      magnitude of the written component alone. A write of pos_x to a
 *      stated radius makes that same step's post-advance range agree
 *      with the written radius to a bound computed from control_dt,
 *      the initial velocity, and the gravitational acceleration across
 *      one control period, and disagree with the pre-write radius by
 *      more than that bound.
 *   3. Position writes mean metres from the world origin. The written
 *      radius crosses two sector boundaries of the runtime's position
 *      fold, so a fold applied wrongly but consistently, which every
 *      other gate here would accept because the streams stay
 *      reproducible, lands the range a whole sector edge away.
 *   4. Statement order inside the block: a later write to a key
 *      overrides an earlier one, and a read sees the writes before it.
 *      Both are driven against a fixture written the direct way, and
 *      the streams must match bitwise. The block does not run on a
 *      boundary-reset step, pinned on a fixture with no reset draws:
 *      every episode's initial observation must be the same bytes.
 *   5. The range-rate component against a finite difference of the
 *      range channel, on two fixtures with separately justified
 *      tolerances, plus the zero-separation rule.
 *   6. The two validation environments: repeated runs at one seed are
 *      byte-identical, distinct seeds differ, and each environment
 *      reaches the ending its source claims.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_state_test"

#define MU_EARTH 3.986004418e14

/* Fixture 1: one box action applied to the craft's velocity along x,
 * so an action at step t moves the state the step t advance
 * integrates. No reset draws, so the only difference between two
 * drives is the action stream. */
static const char *const ORDER_KFL =
    "form RL_ORDER\n"
    "fn world order_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 30.0\n"
    "        horizon 40\n"
    "    end\n"
    "    action push box -50.0 50.0 default 0.0\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0 - trk_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* Fixture 2: the analytic single-step fixture. The observer is at the
 * origin and the craft's unwritten position components are zero, so
 * the range is the written component's magnitude. */
static const char *const SNAP_KFL =
    "form RL_SNAP\n"
    "fn world snap_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=100.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 8\n"
    "    end\n"
    "    action place box 0.0 2.0e11 default 0.0\n"
    "    on_step\n"
    "        craft.pos_x = place\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* Fixture 3a: two writes to one key, the later winning, and a read of
 * a key written earlier in the same body. */
static const char *const ORDER2_KFL =
    "form RL_ORDER2\n"
    "fn world order2_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 30.0\n"
    "        horizon 20\n"
    "    end\n"
    "    action push box -50.0 50.0 default 3.0\n"
    "    on_step\n"
    "        craft.vel_x = push\n"
    "        craft.vel_x = push * 2.0\n"
    "        craft.vel_z = craft.vel_x * 0.5\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0 - trk_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* Fixture 3b: the same effect written directly, with no shadowed
 * write and no read of an earlier one. */
static const char *const ORDER2_REF_KFL =
    "form RL_ORDER2REF\n"
    "fn world order2ref_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 30.0\n"
    "        horizon 20\n"
    "    end\n"
    "    action push box -50.0 50.0 default 3.0\n"
    "    on_step\n"
    "        craft.vel_x = push * 2.0\n"
    "        craft.vel_z = push\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0 - trk_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* Fixture 4: no reset draws and a write every step, so the boundary
 * step is visible: every episode must open on the same observation. */
static const char *const BOUNDARY_KFL =
    "form RL_BOUNDARY\n"
    "fn world boundary_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 30.0\n"
    "        horizon 6\n"
    "    end\n"
    "    action push box -50.0 50.0 default 25.0\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0 - trk_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* Fixture 5: an eccentric orbit sampled every 30 s, so the range
 * swings and its rate is worth measuring. Emitted twice, once per
 * observation mode, by substituting MODE. */
static const char *const RATE_KFL_FMT =
    "form RL_RATE%s\n"
    "fn world rate_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7700.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 30.0\n"
    "        horizon 100\n"
    "    end\n"
    "    observe craft from earth mode=%s as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* Fixture 6: two bodies declared at the same point, so the initial
 * observation exercises the zero-separation rule with no advance. */
static const char *const COINCIDENT_KFL =
    "form RL_COINCIDENT\n"
    "fn world coincident_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=0.0 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=0.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 4\n"
    "    end\n"
    "    action nudge box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.pos_x = nudge * 0.0\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* Drive one session, recording every environment's observation stream.
 * act_of returns the action for (step, environment). */
typedef double (*ActFn)(uint32_t t, uint32_t e);

static void drive_(const char *so_path, uint64_t seed, uint32_t n_envs,
                   uint32_t T, ActFn act_of, double *obs_out,
                   uint32_t *obs_total_out)
{
    void *so = rl_dlopen_(so_path);
    RlSurface s;
    rl_resolve_surface_(so, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(seed, n_envs, &env) == K26RL_OK);

    int32_t len = s.spec(env, NULL, 0);
    ASSERT(len > 0);
    uint8_t *blob = malloc((size_t)len);
    ASSERT(blob != NULL);
    ASSERT(s.spec(env, blob, (uint32_t)len) == len);
    RlSpecView v;
    rl_parse_spec_(blob, (uint32_t)len, &v);
    free(blob);
    const uint32_t obs_total = v.obs_total;
    if (obs_total_out) *obs_total_out = obs_total;

    double *act = malloc(sizeof(double) * n_envs * v.act_total);
    ASSERT(act != NULL);
    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t e = 0; e < n_envs; e++) {
            for (uint32_t c = 0; c < v.act_total; c++) {
                act[e * v.act_total + c] = act_of(t, e);
            }
        }
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, obs_out + (size_t)t * n_envs * obs_total)
               == K26RL_OK);
    }
    free(act);
    s.destroy(env);
    dlclose(so);
}

/* Two action streams that agree until step 7 and differ from there. */
#define DIVERGE_AT 7u
static double act_a_(uint32_t t, uint32_t e) { (void)e; (void)t; return 5.0; }
static double act_b_(uint32_t t, uint32_t e)
{
    (void)e;
    return t < DIVERGE_AT ? 5.0 : 25.0;
}
static double act_zero_(uint32_t t, uint32_t e) { (void)t; (void)e; return 0.0; }

static double g_place = 0.0;
static double act_place_(uint32_t t, uint32_t e)
{
    (void)t; (void)e;
    return g_place;
}

/* Range of the observation vector's channel, environment 0. */
static double range_at_(const double *obs, uint32_t t, uint32_t n_envs,
                        uint32_t obs_total)
{
    return obs[(size_t)t * n_envs * obs_total + 3];
}

static double rate_at_(const double *obs, uint32_t t, uint32_t n_envs,
                       uint32_t obs_total)
{
    return obs[(size_t)t * n_envs * obs_total + 4];
}

static void write_and_compile_(const char *name, const char *src)
{
    char path[256], out[256];
    snprintf(path, sizeof path, "%s/%s.kfl", WORK_DIR, name);
    snprintf(out, sizeof out, "%s/%s", WORK_DIR, name);
    rl_write_file_(path, src);
    rl_compile_(path, out, WORK_DIR);
}

int main(void)
{
    if (!rl_libs_present_("test_rl_state")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    /* ---- Gate 1: write before advance ------------------------------ */
    write_and_compile_("order", ORDER_KFL);
    {
        enum { T = 20, N = 1 };
        static double oa[T * 5], ob[T * 5];
        uint32_t obs_total = 0;
        drive_(WORK_DIR "/order.rlenv.so", 5, N, T, act_a_, oa, &obs_total);
        ASSERT(obs_total == 5);
        drive_(WORK_DIR "/order.rlenv.so", 5, N, T, act_b_, ob, NULL);

        uint32_t first = T;
        for (uint32_t t = 0; t < T; t++) {
            if (memcmp(oa + (size_t)t * 5, ob + (size_t)t * 5,
                       5 * sizeof(double)) != 0) { first = t; break; }
        }
        if (first != DIVERGE_AT) {
            fprintf(stderr, "first divergence at step %u, expected %u\n",
                    first, DIVERGE_AT);
        }
        ASSERT(first == DIVERGE_AT);
    }
    printf("gate 1: the step whose action changes is the step whose"
           " observation changes: OK\n");

    /* ---- Gate 2: the analytic clause ------------------------------- */
    write_and_compile_("snap", SNAP_KFL);
    {
        enum { T = 1, N = 1 };
        const double R = 8.0e6;          /* written radius */
        const double dt = 1.0;           /* the fixture's control_dt */
        const double v0 = 100.0;         /* the fixture's initial speed */
        const double g = MU_EARTH / (R * R);
        const double bound = v0 * dt + 0.5 * g * dt * dt;
        static double pre[T * 5], post[T * 5];

        g_place = 0.0;
        drive_(WORK_DIR "/snap.rlenv.so", 3, N, T, act_place_, pre, NULL);
        const double r_pre = range_at_(pre, 0, N, 5);

        g_place = R;
        drive_(WORK_DIR "/snap.rlenv.so", 3, N, T, act_place_, post, NULL);
        const double r_post = range_at_(post, 0, N, 5);

        printf("gate 2: written %.1f m, reported %.6f m, bound %.6f m,"
               " unwritten %.1f m\n", R, r_post, bound, r_pre);
        ASSERT(fabs(r_post - R) <= bound);
        ASSERT(fabs(r_post - r_pre) > bound);
    }
    printf("gate 2: the write lands on the state this step integrates:"
           " OK\n");

    /* ---- Gate 3: metres from the world origin ---------------------- */
    {
        enum { T = 1, N = 1 };
        /* Two sector edges out (the fold's edge is 2^36 m), so a fold
         * applied wrongly lands a whole edge away from the written
         * value. */
        const double R = 1.5e11;
        const double dt = 1.0, v0 = 100.0;
        const double g = MU_EARTH / (R * R);
        const double bound = v0 * dt + 0.5 * g * dt * dt;
        static double post[T * 5];

        g_place = R;
        drive_(WORK_DIR "/snap.rlenv.so", 3, N, T, act_place_, post, NULL);
        const double r_post = range_at_(post, 0, N, 5);
        printf("gate 3: written %.1f m across %.1f sector edges,"
               " reported %.3f m, bound %.3f m\n",
               R, R / 68719476736.0, r_post, bound);
        ASSERT(fabs(r_post - R) <= bound);
    }
    printf("gate 3: a position key means metres from the world origin,"
           " sector fold included: OK\n");

    /* ---- Gate 4: statement order and the boundary step -------------- */
    write_and_compile_("order2", ORDER2_KFL);
    write_and_compile_("order2ref", ORDER2_REF_KFL);
    {
        enum { T = 15, N = 1 };
        static double o1[T * 5], o2[T * 5];
        drive_(WORK_DIR "/order2.rlenv.so", 9, N, T, act_zero_, o1, NULL);
        drive_(WORK_DIR "/order2ref.rlenv.so", 9, N, T, act_zero_, o2,
               NULL);
        ASSERT(memcmp(o1, o2, sizeof o1) == 0);
    }
    write_and_compile_("boundary", BOUNDARY_KFL);
    {
        rl_run_or_die_(WORK_DIR "/boundary --envs 1 --episodes 3"
                       " --seed 4 --out " WORK_DIR "/boundary.k26epi"
                       " > /dev/null");
        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/boundary.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeInfo info;
        ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
        ASSERT(info.obs_total == 5);
        K26RlEpisodeData e0, e1, e2;
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &e0) == K26RL_OK);
        ASSERT(k26rl_episode_read(rd, 0, 0, 1, &e1) == K26RL_OK);
        ASSERT(k26rl_episode_read(rd, 0, 0, 2, &e2) == K26RL_OK);
        /* No reset draws, so the only thing that could move an
         * episode's opening observation is the block running on the
         * boundary step. */
        ASSERT(memcmp(e0.initial_obs, e1.initial_obs,
                      5 * sizeof(double)) == 0);
        ASSERT(memcmp(e0.initial_obs, e2.initial_obs,
                      5 * sizeof(double)) == 0);
        /* The write does move the episode, so the pin above is not
         * vacuous. */
        ASSERT(memcmp(e0.initial_obs, e0.obs, 5 * sizeof(double)) != 0);
        k26rl_episode_free(&e0);
        k26rl_episode_free(&e1);
        k26rl_episode_free(&e2);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 4: later write wins, a read sees earlier writes, and"
           " the block does not run on a boundary reset: OK\n");

    /* ---- Gate 5: the range-rate component --------------------------- */
    {
        enum { T = 60, N = 1 };
        const double dt = 30.0;
        char src[4096];
        static double og[T * 5], oa[T * 5];

        snprintf(src, sizeof src, RATE_KFL_FMT, "GEO", "geometric");
        write_and_compile_("rate_geo", src);
        snprintf(src, sizeof src, RATE_KFL_FMT, "AST", "astrometric");
        write_and_compile_("rate_ast", src);
        drive_(WORK_DIR "/rate_geo.rlenv.so", 2, N, T, act_zero_, og,
               NULL);
        drive_(WORK_DIR "/rate_ast.rlenv.so", 2, N, T, act_zero_, oa,
               NULL);

        /* Central-difference truncation is (dt^2 / 6) |r'''|. The
         * range oscillates at the orbital rate w = sqrt(mu / a^3)
         * with amplitude A, so |r'''| <= A w^3. Both are read off the
         * recorded series rather than assumed, and the bound carries a
         * factor of four for the higher harmonics of an eccentric
         * orbit. */
        double rmin = range_at_(og, 0, N, 5), rmax = rmin;
        for (uint32_t t = 1; t < T; t++) {
            double r = range_at_(og, t, N, 5);
            if (r < rmin) rmin = r;
            if (r > rmax) rmax = r;
        }
        const double amp = 0.5 * (rmax - rmin);
        const double mid = 0.5 * (rmax + rmin);
        const double w = sqrt(MU_EARTH / (mid * mid * mid));
        const double trunc_tol = 4.0 * (dt * dt / 6.0) * amp * w * w * w;
        /* The astrometric channel corrects the target position for
         * light time, which moves the range by about |rdot| * tau with
         * tau = range / c, so its rate differs from the geometric one
         * by about that correction's own rate. The bound adds it. */
        const double c_light = 2.99792458e8;
        const double rdot_max = amp * w;
        const double lt_tol = 4.0 * (rdot_max * (mid / c_light)) * w;
        const double astro_tol = trunc_tol + lt_tol;

        double worst_g = 0.0, worst_a = 0.0;
        for (uint32_t t = 1; t + 1 < T; t++) {
            double fd_g = (range_at_(og, t + 1, N, 5)
                           - range_at_(og, t - 1, N, 5)) / (2.0 * dt);
            double fd_a = (range_at_(oa, t + 1, N, 5)
                           - range_at_(oa, t - 1, N, 5)) / (2.0 * dt);
            double dg = fabs(fd_g - rate_at_(og, t, N, 5));
            double da = fabs(fd_a - rate_at_(oa, t, N, 5));
            if (dg > worst_g) worst_g = dg;
            if (da > worst_a) worst_a = da;
        }
        printf("gate 5: geometric worst %.6g m/s against tolerance"
               " %.6g; astrometric worst %.6g against %.6g\n",
               worst_g, trunc_tol, worst_a, astro_tol);
        ASSERT(worst_g <= trunc_tol);
        ASSERT(worst_a <= astro_tol);

        /* Zero separation on the reset path: the bodies are declared
         * coincident, so the episode's opening observation exercises
         * the rule with no advance behind it. */
        write_and_compile_("coincident", COINCIDENT_KFL);
        {
            void *so = rl_dlopen_(WORK_DIR "/coincident.rlenv.so");
            RlSurface s;
            rl_resolve_surface_(so, &s);
            K26RlEnv *env = NULL;
            ASSERT(s.create(1, 1, &env) == K26RL_OK);
            double obs[5];
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(obs[3] == 0.0);
            ASSERT(obs[4] == 0.0);
            /* A write that keeps the pair coincident leaves the
             * channel finite, whatever the advance then reports. */
            double act[1] = { 1.0 };
            (void)s.step(env, act);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(isfinite(obs[4]));
            s.destroy(env);
            dlclose(so);
        }
    }
    printf("gate 5: the range rate tracks the range's own derivative,"
           " and zero separation reports 0.0: OK\n");

    /* ---- Gate 6: the validation environments ------------------------ */
    {
        struct { const char *name; const char *src; int terminates; }
        envs[2] = {
            { "orbit_transfer", "../kflc/integration_tests/"
              "orbit_transfer.kfl", 0 },
            { "stationkeeping", "../kflc/integration_tests/"
              "stationkeeping.kfl", 1 },
        };
        for (int i = 0; i < 2; i++) {
            char src[256], out[256], cmd[1024];
            snprintf(src, sizeof src, "integration_tests/%s.kfl",
                     envs[i].name);
            snprintf(out, sizeof out, "%s/%s", WORK_DIR, envs[i].name);
            rl_compile_(src, out, WORK_DIR);

            snprintf(cmd, sizeof cmd,
                     "%s --envs 2 --episodes 2 --seed 31 --out %s/%s_a"
                     ".k26epi > /dev/null", out, WORK_DIR, envs[i].name);
            rl_run_or_die_(cmd);
            snprintf(cmd, sizeof cmd,
                     "%s --envs 2 --episodes 2 --seed 31 --out %s/%s_b"
                     ".k26epi > /dev/null", out, WORK_DIR, envs[i].name);
            rl_run_or_die_(cmd);
            snprintf(cmd, sizeof cmd,
                     "%s --envs 2 --episodes 2 --seed 32 --out %s/%s_c"
                     ".k26epi > /dev/null", out, WORK_DIR, envs[i].name);
            rl_run_or_die_(cmd);

            char pa[256], pb[256], pc[256];
            snprintf(pa, sizeof pa, "%s/%s_a.k26epi", WORK_DIR,
                     envs[i].name);
            snprintf(pb, sizeof pb, "%s/%s_b.k26epi", WORK_DIR,
                     envs[i].name);
            snprintf(pc, sizeof pc, "%s/%s_c.k26epi", WORK_DIR,
                     envs[i].name);
            ASSERT(rl_files_equal_(pa, pb));
            ASSERT(!rl_files_equal_(pa, pc));

            K26RlEpisodeReader *rd = NULL;
            ASSERT(k26rl_episode_reader_open(pa, &rd) == K26RL_OK);
            K26RlEpisodeInfo info;
            ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
            ASSERT(info.obs_total == 5);
            K26RlEpisodeData ep;
            ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep) == K26RL_OK);
            /* At the declared default action each environment reaches
             * the ending its source claims: the transfer runs out its
             * horizon, the station box is left. */
            if (envs[i].terminates) {
                ASSERT(ep.end_reason == K26RL_END_TERMINATED);
            } else {
                ASSERT(ep.end_reason == K26RL_END_TRUNCATED);
            }
            printf("gate 6: %s: same seed byte-identical, distinct seed"
                   " differs, first episode %s after %u steps\n",
                   envs[i].name,
                   envs[i].terminates ? "terminated" : "truncated",
                   ep.step_count);
            k26rl_episode_free(&ep);
            k26rl_episode_reader_close(rd);
        }
    }
    printf("gate 6: both validation environments determinate and"
           " reaching their stated endings: OK\n");

    printf("test_rl_state: 6 gates passed\n");
    return 0;
}
