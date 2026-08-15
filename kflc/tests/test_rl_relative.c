/* test_rl_relative.c: the relative observe form through the stepping
 * surface.
 *
 * The proximity library's own gates measure the frame and the
 * linearised propagator against closed forms and against an integrated
 * trajectory. This one measures what only an artifact can show: that
 * the six channels are published under the names and in the order the
 * grammar promises, that they carry the relative state of the two
 * bodies the program named rather than some other pair's, that they
 * are resolved onto the chief's own axes, and that they are the state
 * seen from the rotating frame rather than an inertial difference
 * dressed in new axes.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   Comparing the published channels against the same library that
 *   computed them would compare a thing with itself. The expected
 *   values are rebuilt here from the body-state getter, with a frame
 *   constructed in this file from the axis definitions in words, and
 *   nothing in that path calls the proximity library.
 *
 *   A fixture whose two craft have different inertial velocities lets
 *   an implementation that published the inertial difference look
 *   almost right, because the frame's own rotation is then a small
 *   part of what is measured. The fixture here gives both craft the
 *   same inertial velocity, so the entire published velocity is the
 *   frame-rotation term: an implementation that omitted it publishes
 *   exactly zero, and the arm asserts a value that is not zero and
 *   equals n times the along-track offset in closed form.
 *
 *   A channel-name arm alone would not catch a form that published its
 *   six values into five slots, so the observe declared after the
 *   relative one is checked to start where a six-wide form leaves it,
 *   and its own values are checked to be that body's line of sight.
 *
 *   An arm over a fixture at rest would confirm nothing about a
 *   changing state, so the comparison runs over a drive of sixty
 *   steps and is asserted at every one of them.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_relative_test"

static int n_pass = 0;

static const char *rl_stage_name_ = "startup";

static void rl_deadline_fired_(int sig)
{
    (void)sig;
    const char *a = "test_rl_relative: DEADLINE EXCEEDED at stage: ";
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

static void rl_stage_done_(void) { alarm(0); }

/* The fixture. Both craft carry the same inertial velocity and differ
 * only in their along-track position, which is the configuration that
 * separates the rotating-frame rate from the inertial difference: the
 * inertial difference is exactly zero and the true relative velocity
 * is entirely the frame's own rotation. The masses are small enough
 * that their mutual attraction is nothing on this timescale, so the
 * only dynamics are the central body's. */
#define REL_KFL \
    "form RL_RELATIVE\n" \
    "fn world rel_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body chief mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    astro_body deputy mass=1.0 parent=earth" \
    " pos_x=7.0e6 pos_y=30.0 pos_z=12.0 vel_y=7546.049108166324\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 400\n" \
    "        substeps 1\n" \
    "    end\n" \
    "    action idle box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        deputy.vel_x = deputy.vel_x + idle * 0.0\n" \
    "    end\n" \
    "    observe relative deputy from chief as rel\n" \
    "    observe deputy from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward rel_r_y\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* Six relative channels then five line-of-sight ones. */
#define REL_OBS   11
#define REL_BASE   0
#define TRK_BASE   6
#define N_BODIES   3
#define B_EARTH    0
#define B_CHIEF    1
#define B_DEPUTY   2

#define MU_EARTH 3.986004418e14

static double absd_(double x) { return x < 0.0 ? -x : x; }

typedef struct { double x, y, z; } V3;

static V3 v3_(double x, double y, double z) { V3 v = { x, y, z }; return v; }
static V3 sub_(V3 a, V3 b) { return v3_(a.x - b.x, a.y - b.y, a.z - b.z); }
static double dot_(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3 cross_(V3 a, V3 b)
{
    return v3_(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static V3 scale_(V3 a, double s) { return v3_(a.x*s, a.y*s, a.z*s); }
static double len_(V3 a) { return sqrt(dot_(a, a)); }

/* The expected six channels, rebuilt here from the body states with a
 * frame written from the axis definitions in words: the first axis
 * points from the central body to the chief, the third along the
 * orbital angular momentum, the second completes the right-handed
 * set. Nothing here calls the library under test. */
static void expect_relative_(const double *bodies, double out[6])
{
    V3 rc = v3_(bodies[B_CHIEF * 6 + 0], bodies[B_CHIEF * 6 + 1],
                bodies[B_CHIEF * 6 + 2]);
    V3 rd = v3_(bodies[B_DEPUTY * 6 + 0], bodies[B_DEPUTY * 6 + 1],
                bodies[B_DEPUTY * 6 + 2]);
    V3 vc = v3_(bodies[B_CHIEF * 6 + 3], bodies[B_CHIEF * 6 + 4],
                bodies[B_CHIEF * 6 + 5]);
    V3 vd = v3_(bodies[B_DEPUTY * 6 + 3], bodies[B_DEPUTY * 6 + 4],
                bodies[B_DEPUTY * 6 + 5]);
    V3 ve = v3_(bodies[B_EARTH * 6 + 3], bodies[B_EARTH * 6 + 4],
                bodies[B_EARTH * 6 + 5]);

    /* The getter's positions are already relative to the reference
     * body, which this gate asks for as the central body. */
    V3 vrel_c = sub_(vc, ve);
    V3 h  = cross_(rc, vrel_c);
    V3 e1 = scale_(rc, 1.0 / len_(rc));
    V3 e3 = scale_(h, 1.0 / len_(h));
    V3 e2 = cross_(e3, e1);
    V3 om = scale_(h, 1.0 / dot_(rc, rc));

    V3 rho = sub_(rd, rc);
    V3 dv  = sub_(vd, vc);
    V3 vrot = sub_(dv, cross_(om, rho));

    out[0] = dot_(rho, e1);
    out[1] = dot_(rho, e2);
    out[2] = dot_(rho, e3);
    out[3] = dot_(vrot, e1);
    out[4] = dot_(vrot, e2);
    out[5] = dot_(vrot, e3);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGALRM, rl_deadline_fired_);

    if (!rl_libs_present_("test_rl_relative")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/rel.kfl", REL_KFL);

    rl_stage_("compiling the relative artifact", 900u);
    rl_compile_(WORK_DIR "/rel.kfl", WORK_DIR "/rel", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/rel.rlenv.so"));
    rl_stage_done_();

    void *so = rl_dlopen_(WORK_DIR "/rel.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    K26RlEnv *env = NULL;
    ASSERT(s.create(19u, 1u, &env) == K26RL_OK);

    /* ---- 1. The published names, and what follows them ----------- */
    rl_stage_("reading the spec", 60u);
    {
        int32_t need = s.spec(env, NULL, 0);
        ASSERT(need > 0);
        uint8_t *blob = (uint8_t *)malloc((size_t)need);
        ASSERT(blob != NULL);
        ASSERT(s.spec(env, blob, (uint32_t)need) == need);
        RlSpecView v;
        rl_parse_spec_(blob, (uint32_t)need, &v);

        printf("gate 1: obs_total %u (six relative channels then five"
               " line-of-sight ones)\n", v.obs_total);
        ASSERT(v.obs_total == REL_OBS);

        static const char *const want[] = {
            "rel_r_x", "rel_r_y", "rel_r_z", "rel_v_x", "rel_v_y",
            "rel_v_z", "trk_dir_x"
        };
        /* The names, walked from the tag stream a consumer walks. */
        uint32_t off = 0;
        int found = 0;
        while (off + 6 <= (uint32_t)need) {
            uint16_t tag = rl_get_u16_(blob + off);
            uint32_t l   = rl_get_u32_(blob + off + 2);
            if (tag == K26RL_TAG_OBS_CHANNEL_NAME && l >= 4) {
                uint32_t ch = rl_get_u32_(blob + off + 6);
                if (ch < 7) {
                    char nm[64];
                    uint32_t nl = l - 4;
                    if (nl > 63) nl = 63;
                    memcpy(nm, blob + off + 10, nl);
                    nm[nl] = '\0';
                    printf("  channel %u is `%s`, expected `%s`\n",
                           ch, nm, want[ch]);
                    ASSERT(strcmp(nm, want[ch]) == 0);
                    found++;
                }
            }
            off += 6 + l;
        }
        ASSERT(found == 7);

        /* Every relative channel is geometric: the form resolves a
         * state the integrator produced onto a set of axes and applies
         * no correction, so a mode asserting one would be false. */
        ASSERT(v.n_modes >= REL_OBS);
        for (int c = 0; c < 6; c++) {
            ASSERT(v.modes[c] == K26RL_OBS_MODE_GEOMETRIC);
        }
        printf("  all six relative channels publish the geometric"
               " mode\n");
        free(blob);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 1: six channels at their names, and the observe after"
           " them is not displaced: OK\n");

    /* ---- 2. The frame-rotation term, in closed form -------------- */
    rl_stage_("reading the initial state", 60u);
    {
        double obs[REL_OBS];
        double bodies[N_BODIES * 6];
        ASSERT(s.obs(env, obs) == K26RL_OK);
        ASSERT(s.bodies(env, B_EARTH, bodies, N_BODIES * 6) == N_BODIES * 6);

        /* The frame's rotation rate, from the chief's own state: the
         * angular momentum per unit radius squared. The Keplerian mean
         * motion of a circular orbit at this radius is printed beside
         * it, because the two agree only when the declared velocity is
         * exactly circular and this fixture's is not, to eight digits.
         * The rate that governs the frame is the one the state gives,
         * and it is what the closed form below uses. */
        double rc[3] = { bodies[B_CHIEF * 6 + 0], bodies[B_CHIEF * 6 + 1],
                         bodies[B_CHIEF * 6 + 2] };
        double vc[3] = { bodies[B_CHIEF * 6 + 3], bodies[B_CHIEF * 6 + 4],
                         bodies[B_CHIEF * 6 + 5] };
        double hz = rc[0] * vc[1] - rc[1] * vc[0];
        double r2 = rc[0]*rc[0] + rc[1]*rc[1] + rc[2]*rc[2];
        double n  = hz / r2;
        double r0 = sqrt(r2);
        printf("gate 2: at the episode's first state\n");
        printf("  the chief's frame turns at %.12e rad/s; a circular"
               " orbit at %.1f m would turn at %.12e\n",
               n, r0, sqrt(MU_EARTH / (r0 * r0 * r0)));
        printf("  r = (%+.6f, %+.6f, %+.6f) m\n",
               obs[REL_BASE + 0], obs[REL_BASE + 1], obs[REL_BASE + 2]);
        printf("  v = (%+.9f, %+.9f, %+.9f) m/s\n",
               obs[REL_BASE + 3], obs[REL_BASE + 4], obs[REL_BASE + 5]);

        /* The declared offsets, resolved onto the chief's axes. At the
         * first state the chief sits on the radial axis with the
         * declared velocity, so along-track is the declared y and
         * cross-track the declared z. */
        ASSERT(absd_(obs[REL_BASE + 0]) < 1e-3);
        ASSERT(absd_(obs[REL_BASE + 1] - 30.0) < 1e-3);
        ASSERT(absd_(obs[REL_BASE + 2] - 12.0) < 1e-6);

        /* The whole of the velocity is the frame's rotation, because
         * the two craft were given the same inertial velocity. An
         * implementation publishing the inertial difference reads zero
         * here; one publishing the rotating-frame rate reads n times
         * the along-track offset on the radial axis. */
        double want_vx = n * obs[REL_BASE + 1];
        printf("  so v_x must be %+.9f m/s and not zero\n", want_vx);
        ASSERT(absd_(want_vx) > 0.03);
        ASSERT(absd_(obs[REL_BASE + 3] - want_vx) < 1e-9);
        ASSERT(absd_(obs[REL_BASE + 4]) < 1e-9);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 2: the published velocity is the rotating frame's,"
           " not the inertial difference: OK\n");

    /* ---- 3. Every step, against a frame built here --------------- */
    rl_stage_("driving the artifact", 300u);
    {
        int32_t need = s.bodies(env, B_EARTH, NULL, 0);
        printf("gate 3: the body getter needs %d doubles"
               " (%d bodies by six)\n", need, N_BODIES);
        ASSERT(need == N_BODIES * 6);
        double bodies[N_BODIES * 6];
        double obs[REL_OBS];
        double act[1] = { 0.0 };
        double worst = 0.0, worst_v = 0.0;
        double moved = 0.0;

        double first_r[3] = { 0, 0, 0 };
        for (int t = 0; t <= 60; t++) {
            if (t > 0) ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.bodies(env, B_EARTH, bodies, (uint32_t)need) == need);

            double want[6];
            expect_relative_(bodies, want);
            for (int k = 0; k < 6; k++) {
                double e = absd_(obs[REL_BASE + k] - want[k]);
                if (k < 3 && e > worst)   worst = e;
                if (k >= 3 && e > worst_v) worst_v = e;
            }
            if (t == 0) {
                for (int k = 0; k < 3; k++) first_r[k] = obs[REL_BASE + k];
            } else {
                for (int k = 0; k < 3; k++) {
                    double d = absd_(obs[REL_BASE + k] - first_r[k]);
                    if (d > moved) moved = d;
                }
            }
        }
        printf("  worst position disagreement over 60 steps %.3e m,"
               " worst velocity %.3e m/s\n", worst, worst_v);
        printf("  the relative position moved %.4f m over the drive,"
               " so the comparison is not of a constant\n", moved);
        ASSERT(moved > 0.5);
        ASSERT(worst < 1e-6);
        ASSERT(worst_v < 1e-9);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 3: the channels are the relative state at every step"
           " of a drive: OK\n");

    /* ---- 4. The channels after the relative ones are that body's -- */
    {
        double obs[REL_OBS];
        double bodies[N_BODIES * 6];
        ASSERT(s.obs(env, obs) == K26RL_OK);
        ASSERT(s.bodies(env, B_EARTH, bodies, N_BODIES * 6) == N_BODIES * 6);
        V3 rd = v3_(bodies[B_DEPUTY * 6 + 0], bodies[B_DEPUTY * 6 + 1],
                    bodies[B_DEPUTY * 6 + 2]);
        double range = len_(rd);
        printf("gate 4: the line-of-sight range at channel %d reads"
               " %.6f m, the deputy's own distance is %.6f m\n",
               TRK_BASE + 3, obs[TRK_BASE + 3], range);
        ASSERT(absd_(obs[TRK_BASE + 3] - range) < 1e-3);
        /* And the direction is a unit vector, so the slot really is
         * the line-of-sight family and not six relative values spilled
         * one place further on. */
        double dl = sqrt(obs[TRK_BASE + 0] * obs[TRK_BASE + 0] +
                         obs[TRK_BASE + 1] * obs[TRK_BASE + 1] +
                         obs[TRK_BASE + 2] * obs[TRK_BASE + 2]);
        printf("  the direction at channels %d to %d has length"
               " %.12f\n", TRK_BASE, TRK_BASE + 2, dl);
        ASSERT(absd_(dl - 1.0) < 1e-9);
        n_pass++;
    }
    printf("gate 4: a six-wide form leaves the next observe at the"
           " right offset: OK\n");

    s.destroy(env);
    dlclose(so);
    printf("test_rl_relative: %d gates passed\n", n_pass);
    return 0;
}
