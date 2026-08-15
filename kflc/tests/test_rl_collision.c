/* test_rl_collision.c: the collision pass through the stepping
 * surface.
 *
 * The library's own gates measure the kernels and the resolutions.
 * This one measures what only an artifact can show: that the pass
 * runs between the sub-advances of a real step, that a contact ends
 * an episode through the program's own predicate rather than through
 * a fault, that the contact channels carry what the design says they
 * carry, that the applied duration is untouched, and that two runs
 * agree bitwise over a fixture whose episodes end in contact.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   A fixture whose bodies never meet would pass for a build with no
 *   collision pass at all, so the first arm asserts a contact
 *   happens, at a predicted step, and a companion fixture with the
 *   target moved aside asserts none does. Both run the same program
 *   text with one number changed.
 *
 *   A contact-channel arm that only read `_hit` could not tell a
 *   latch that fires from one that fires with the wrong numbers
 *   beside it, so the fraction is checked against the analytic
 *   crossing time and the speed against the closing speed the
 *   fixture declares.
 *
 *   A termination arm that only checked the episode ended would pass
 *   for an episode that truncated on the horizon, so it asserts the
 *   terminated flag specifically, that the fault code is zero, and
 *   that the step it ended on is earlier than the horizon.
 *
 *   A determinism arm over a fixture with no contact would measure
 *   the determinism of everything except the thing under test, so the
 *   fixture used is the contacting one and the arm additionally
 *   asserts that the runs it compares did contain contacts.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"
#include <math.h>

#define WORK_DIR "/tmp/kflc_rl_collision_test"

static int n_pass = 0;

/* ---- Wall-clock guard --------------------------------------------- *
 *
 * Every arm below drives a compiled artifact, and an artifact can
 * livelock: this item found a configuration in which two collidable
 * bodies reach coincident centres inside one sub-advance and hand the
 * adaptive integrator a singular field it spins on rather than
 * faulting. A gate that hangs is worse than a gate that fails, because
 * it stops the whole suite from ever reporting; so each stretch of
 * work that can reach an artifact is given a deadline, and passing it
 * ends the gate loudly with the stage named.
 *
 * The handler writes with write(2) rather than printf, because it runs
 * from a signal and stdio is not safe there, and ends with _exit for
 * the same reason. */
static const char *rl_stage_name_ = "startup";

static void rl_deadline_fired_(int sig)
{
    (void)sig;
    const char *a = "test_rl_collision: DEADLINE EXCEEDED at stage: ";
    (void)!write(2, a, strlen(a));
    (void)!write(2, rl_stage_name_, strlen(rl_stage_name_));
    (void)!write(2, "\n", 1);
    _exit(1);
}

/* Begin a stage with `secs` seconds of wall clock allowed. */
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

/* Run an arm in a child process and require it to succeed.
 *
 * The arms below need a live handle, and a program carrying two
 * bodies that each bind an assembly faults on the third environment
 * handle created in one process, whatever the seeds. That defect was
 * measured with the collision pass compiled out of the same program,
 * so it is neither the sweep nor the resolution, and it has its own
 * item; what it means here is that an arm's correctness must not
 * depend on how many arms ran before it. A child process per arm
 * gives each one the first handle in its own process, which is both
 * a way around the defect and better isolation than counting on an
 * ordering. */
#define IN_CHILD(...) do {                                            \
    fflush(stdout);                                                   \
    pid_t _p = fork();                                                \
    ASSERT(_p >= 0);                                                  \
    if (_p == 0) { __VA_ARGS__; fflush(stdout); _exit(0); }            \
    int _st = 0;                                                      \
    ASSERT(waitpid(_p, &_st, 0) == _p);                               \
    if (!(WIFEXITED(_st) && WEXITSTATUS(_st) == 0)) {                 \
        fprintf(stderr, "FAIL %s:%d: arm failed in its child\n",       \
                __FILE__, __LINE__);                                  \
        exit(1);                                                      \
    }                                                                 \
} while (0)

/* The name published for one channel index, or 0 when the spec
 * carries none. Walks the same tag stream a consumer walks. */
static int spec_name_at_(const uint8_t *blob, uint32_t len, int want,
                         char *out, size_t cap)
{
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint16_t tag = rl_get_u16_(blob + off);
        uint32_t l   = rl_get_u32_(blob + off + 2);
        if (off + 6 + l > len) break;
        if (tag == K26RL_TAG_OBS_CHANNEL_NAME && l > 4 &&
            (int)rl_get_u32_(blob + off + 6) == want) {
            uint32_t nl = l - 4;
            if (nl >= cap) nl = (uint32_t)cap - 1;
            memcpy(out, blob + off + 10, nl);
            out[nl] = '\0';
            return 1;
        }
        off += 6 + l;
    }
    return 0;
}

/* A target plate and an approaching craft. The craft starts 60 m out
 * and closes at a declared rate; the plate is a thin box across its
 * path, half a metre thick along the approach and twelve metres
 * across it. Contact is when the craft's own collider reaches the
 * plate's face, and the control period and closing rate are chosen so
 * that the crossing lands inside a step rather than on its boundary.
 *
 * The approach covers six metres per control period against a target
 * a tenth of a metre thick, and one and a half metres per
 * sub-advance against the same tenth, so a test made at the ends of
 * either would step over the plate entirely: fifteen times its
 * thickness within one sub-advance and sixty times within one control
 * period. That is the tunnelling case at the artifact level, and it
 * is why the fixture is shaped this way.
 *
 * The chaser's collider is three metres across the approach, which is
 * not decoration. The pass runs after the sub-advance, so the world
 * has already advanced past the contact by the time the sweep reports
 * it and the resolution puts the pair back; a pair that travels from
 * first touch to coincident centres inside one sub-advance therefore
 * hands the integrator a configuration it cannot integrate, because
 * an assembly sets a body's gravitational parameter from its mass and
 * two coincident masses are a singular field. That is a real
 * precondition of the design's ordering and is reported rather than
 * hidden: the contact separation must exceed what the pair closes in
 * one sub-advance. Here it is 3.05 m against 1.5 m, so the fixture
 * sits inside the supported regime and measures the collision system
 * rather than the integrator's behaviour at a singularity.
 *
 * Both bodies orbit, so the motion the sweep sees is the integrator's
 * and not a straight line the fixture arranged: the approach is a
 * real relative velocity in a real world, which is what makes the
 * curvature margin's job a real one. */
static const char *const COLL_ASM_TARGET =
    "assembly coll_target\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component plate\n"
    "        mass 5000.0\n"
    "        at 0 0 0\n"
    "        collider box 6.0 6.0 0.05\n"
    "    end\n"
    "end\n";

static const char *const COLL_ASM_CHASER =
    "assembly coll_chaser\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider sphere 0 0 0 3.0\n"
    "    end\n"
    "end\n";

/* OFFSET is the chaser's lateral displacement: zero for the arm that
 * must contact, wide for the arm that must not. CLOSE is its closing
 * speed towards the target in metres per second. */
#define COLL_KFL(OFFSET, CLOSE) \
    "form RL_COLL\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body target assembly=\"coll_target.k26asm\" parent=earth" \
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n" \
    "    astro_body chaser assembly=\"coll_chaser.k26asm\" parent=earth" \
    " pos_x=7.0e6 pos_y=" OFFSET " pos_z=-60.0" \
    " vel_y=7546.0 vel_z=" CLOSE " quat_w=1.0\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 40\n" \
    "        substeps 4\n" \
    "        terminated when hit_hit > 0.5\n" \
    "    end\n" \
    "    action idle box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        chaser.omega_x = idle * 0.0\n" \
    "    end\n" \
    "    observe contact of chaser as hit\n" \
    "    observe chaser from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward hit_hit\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* A fixture that survives its own contact. Every arm above ends the
 * episode on the contacting step, which leaves everything the
 * resolution does to the world unexamined: the velocity merge that
 * removes the pair's relative motion is never read, because nothing
 * reads anything after it. This one never terminates, so the steps
 * after the contact are ordinary steps and can be measured.
 *
 * RESOLUTION is the episode's `contact` line, OFFSET the chaser's
 * lateral displacement. An offset chaser meets the plate away from
 * its centre, so the lever arm from the plate's centre of mass to the
 * contact point is not parallel to the normal and an impulse there
 * must spin the plate. A centred fixture cannot see that, and cannot
 * see an inverse inertia left at zero. */
#define COLL_LIVE_KFL(OFFSET, RESOLUTION) \
    "form RL_COLLIVE\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body target assembly=\"coll_target.k26asm\" parent=earth" \
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n" \
    "    astro_body chaser assembly=\"coll_chaser.k26asm\" parent=earth" \
    " pos_x=7.0e6 pos_y=" OFFSET " pos_z=-60.0" \
    " vel_y=7546.0 vel_z=6.0 quat_w=1.0\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 40\n" \
    "        substeps 4\n" \
    RESOLUTION \
    "        terminated when hit_hit > 1.5\n" \
    "    end\n" \
    "    action idle box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        chaser.omega_x = idle * 0.0\n" \
    "    end\n" \
    "    observe contact of chaser as hit\n" \
    "    observe attitude of target as tatt\n" \
    "    observe chaser from earth mode=geometric as trk\n" \
    "    observe target from chaser mode=geometric as sep\n" \
    "    objective\n" \
    "        reward hit_hit\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* The observation width of the fixture above, and the indices this
 * gate reads. Three contact channels, then seven attitude channels,
 * then five tracking channels, then five more for the line of sight
 * from the chaser to the target. That last group is what makes the
 * resolution readable from the recorded stream rather than from a
 * live handle: its range rate is the pair's closing rate, which is
 * exactly what arrest removes and what a bounce reverses. */
#define LIVE_OBS       20
#define LIVE_HIT        0
#define LIVE_SPEED      2
#define LIVE_TARGET_WY  8      /* tatt_omega_y */
#define LIVE_SEP_RANGE 18
#define LIVE_SEP_RATE  19

int main(void)
{
    /* Line buffered, so an arm's output reaches the log as it is
     * produced. Fully buffered output made an earlier failure look
     * like a hang with no output at all, when the gate was in fact
     * still compiling an artifact: the distinction matters and cost
     * time to establish. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGALRM, rl_deadline_fired_);

    if (!rl_libs_present_("test_rl_collision")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/coll_target.k26asm", COLL_ASM_TARGET);
    rl_write_file_(WORK_DIR "/coll_chaser.k26asm", COLL_ASM_CHASER);

    /* The analytic crossing. The chaser starts 60 m below the target
     * and closes at 6 m/s, so their separation along the approach is
     * 60 - 6 t seconds after the episode begins. Contact is when the
     * sphere's surface reaches the plate's face, a centre separation
     * of 0.05 + 3.0 = 3.05 m, which is at t = 56.95 / 6 = 9.4917 s.
     * With a control period of one second that is inside step 10,
     * counting from one, at a fraction of 0.4917 through it. */
    const double want_time = 56.95 / 6.0;
    const int    want_step = 10;
    const double want_frac = want_time - 9.0;
    const double want_sep  = 3.05;

    /* ---- 1. A contact happens, and only when the paths cross ----- */
    {
        rl_stage_("compiling the contacting artifact", 900u);
        rl_write_file_(WORK_DIR "/hit.kfl", COLL_KFL("0.0", "6.0"));
        rl_compile_(WORK_DIR "/hit.kfl", WORK_DIR "/hit", WORK_DIR);
        rl_stage_("driving the contacting artifact", 120u);
        void *so = rl_dlopen_(WORK_DIR "/hit.rlenv.so");
        RlSurface s;
        rl_resolve_surface_(so, &s);

        K26RlEnv *env = NULL;
        ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
        double act[1] = { 0.0 };
        int step_hit = -1;
        double frac = -1.0, speed = -1.0;
        uint32_t fl = 0;
        for (int t = 1; t <= 20 && step_hit < 0; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
            double obs[8];
            ASSERT(s.obs(env, obs) == K26RL_OK);
            if (obs[0] > 0.5) { step_hit = t; frac = obs[1]; speed = obs[2]; }
        }
        ASSERT(s.flags(env, &fl) == K26RL_OK);
        printf("  contact on step %d, predicted %d\n", step_hit, want_step);
        printf("  fraction %.9f, predicted %.9f\n", frac, want_frac);
        printf("  closing speed %.9f, declared %.9f\n", speed, 6.0);
        ASSERT(step_hit == want_step);
        /* The fraction is of the control period and lies inside it,
         * which is the property that distinguishes a swept report
         * from one made at a step boundary. */
        ASSERT(frac > 0.0 && frac < 1.0);
        ASSERT(fabs(frac - want_frac) < 2.0e-2);
        /* The speed is the closing rate the fixture declares. The
         * two bodies also share an orbital velocity of 7546 m/s, so
         * a speed reported along the wrong direction, or as a total
         * rather than a normal component, would be three orders
         * larger and this bound would catch it. */
        ASSERT(fabs(speed - 6.0) < 1.0e-2);
        printf("  a contact is reported at the predicted step, at a "
               "fraction inside the period, with the declared closing "
               "speed: OK\n");
        n_pass++;

        /* It is a termination, not a fault: the program's own
         * predicate ended the episode, flag bit 0 is set, and the
         * fault registry is untouched. */
        printf("  flags %#x\n", fl);
        ASSERT((fl & K26RL_FLAG_TERMINATED) != 0);
        ASSERT((fl & K26RL_FLAG_TRUNCATED) == 0);
        uint16_t codes[1] = { 0xFFFFu };
        ASSERT(s.fault_codes(env, codes) == K26RL_OK);
        printf("  fault code %u, horizon 40, ended on step %d\n",
               codes[0], step_hit);
        ASSERT(codes[0] == 0);
        ASSERT(step_hit < 40);
        printf("  a contact ends the episode through the declared "
               "predicate and adds no fault: OK\n");
        n_pass++;

        /* Arrest, at the artifact level. The pair is placed at the
         * configuration the sweep itself computed, so after the
         * contacting transition their separation along the approach
         * is the contact separation and not whatever the integrator
         * carried them to. Without it the two would be well past each
         * other by the end of the step: they close six metres per
         * period and the contact happens five sixths of the way
         * through one, so an unarrested pair ends the step about a
         * metre beyond the touching configuration rather than at it.
         *
         * This is read from the body getter rather than from the
         * contact channels, so it measures the state the episode
         * recorded and not the report that described it. */
        /* Three bodies in declaration order, six doubles each:
         * earth, then the target, then the chaser. */
        double bod[3 * 6];
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, bod, 18) == 18);
        double sep = bod[2 * 6 + 2] - bod[1 * 6 + 2];
        printf("  separation along the approach after the contacting "
               "transition %.9f m, contact separation %.9f m\n",
               sep, -want_sep);
        /* The chaser is the second body and approaches from below, so
         * the separation is minus the contact separation. */
        ASSERT(fabs(sep + want_sep) < 1.0e-6);
        printf("  arrest leaves the pair at the sweep's own impact "
               "configuration: OK\n");
        n_pass++;
        s.destroy(env);

        /* The companion: the same program with the chaser displaced
         * laterally past the plate's own extent, so the paths do not
         * cross. Without this the arm above would pass for a build
         * that reported contact unconditionally. */
        rl_stage_("compiling the non-contacting artifact", 900u);
        rl_write_file_(WORK_DIR "/miss.kfl", COLL_KFL("40.0", "6.0"));
        rl_compile_(WORK_DIR "/miss.kfl", WORK_DIR "/miss", WORK_DIR);
        rl_stage_("driving the non-contacting artifact", 120u);
        void *so2 = rl_dlopen_(WORK_DIR "/miss.rlenv.so");
        RlSurface s2;
        rl_resolve_surface_(so2, &s2);
        K26RlEnv *m = NULL;
        ASSERT(s2.create(11u, 1u, &m) == K26RL_OK);
        int any = 0;
        double worst = 0.0;
        for (int t = 0; t < 20; t++) {
            ASSERT(s2.step(m, act) == K26RL_OK);
            double obs[8];
            ASSERT(s2.obs(m, obs) == K26RL_OK);
            if (obs[0] > 0.5) any = 1;
            if (obs[1] != 0.0 || obs[2] != 0.0) worst = 1.0;
        }
        printf("  the displaced chaser reports contact: %s\n",
               any ? "yes" : "no");
        ASSERT(any == 0);
        /* And on a step with no contact all three channels read
         * exactly zero, not merely small. */
        ASSERT(worst == 0.0);
        printf("  a path that does not cross reports no contact and "
               "three channels of exact zero: OK\n");
        n_pass++;
        s2.destroy(m);
    }

    /* ---- 2. The applied duration is untouched -------------------- */
    {
        /* A contact arrests the pair at the impact configuration
         * rather than shortening the step, so the transition still
         * advanced exactly one control period. The episode file
         * records the applied duration, and that is what is read
         * back here rather than recomputed. */
        rl_stage_("recording one episode", 180u);
        rl_run_or_die_(WORK_DIR "/hit --envs 1 --episodes 1 --seed 11"
                       " --out " WORK_DIR "/hit.k26epi > /dev/null");
        K26RlEpisodeReader *r = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/hit.k26epi", &r) ==
               K26RL_OK);
        uint32_t ord = 0, env = 0, ep = 0;
        ASSERT(k26rl_episode_reader_at(r, 0, &ord, &env, &ep) == K26RL_OK);
        K26RlEpisodeData d;
        memset(&d, 0, sizeof d);
        ASSERT(k26rl_episode_read(r, ord, env, ep, &d) == K26RL_OK);
        ASSERT(d.step_count > 0);
        int exact = 1;
        for (uint32_t i = 0; i < d.step_count; i++) {
            if (d.applied_dt[i] != 1.0) exact = 0;
        }
        printf("  %u transition(s), applied dt %s the declared 1.0\n",
               d.step_count, exact ? "exactly" : "NOT exactly");
        ASSERT(exact);
        /* The episode really did end in a contact rather than on the
         * horizon, or the arm measures a step the pass never touched. */
        printf("  ended after %u transitions, horizon 40\n", d.step_count);
        ASSERT(d.step_count < 40);
        k26rl_episode_reader_close(r);
        printf("  a contacting transition still applies exactly the "
               "declared control period: OK\n");
        n_pass++;
    }

    /* ---- 3. Determinism over a contacting fixture ---------------- */
    {
        /* Measured through the recorded episode files rather than
         * through live handles, which is both the stronger check and
         * the one this gate can make: determinism is a property of
         * the recorded stream, and the record is what a consumer
         * replays from.
         *
         * It is also a way around a defect this item found and did
         * not cause: a program carrying two assembly-bound bodies
         * faults with an integrator divergence on the first step of
         * the THIRD environment handle created in one process,
         * whatever the seeds, and does so identically when the
         * collision pass is compiled out of it, so it is neither the
         * sweep nor the resolution. Two separate processes have one
         * handle each and are unaffected. The finding is reported
         * rather than worked around silently; this arm simply does
         * not need a third handle to make its measurement. */
        rl_stage_("recording the determinism runs", 300u);
        rl_run_or_die_(WORK_DIR "/hit --envs 3 --episodes 2 --seed 77"
                       " --out " WORK_DIR "/a.k26epi > /dev/null");
        rl_run_or_die_(WORK_DIR "/hit --envs 3 --episodes 2 --seed 77"
                       " --out " WORK_DIR "/b.k26epi > /dev/null");
        printf("  two runs over a contacting fixture: %s\n",
               rl_files_equal_(WORK_DIR "/a.k26epi", WORK_DIR "/b.k26epi")
               ? "byte identical" : "DIFFER");
        ASSERT(rl_files_equal_(WORK_DIR "/a.k26epi", WORK_DIR "/b.k26epi"));

        /* Environment 0 beside three neighbours, against environment
         * 0 run alone. The neighbours are driven by the same action
         * stream here because the program's action does nothing, so
         * what this pins is that the presence and count of
         * neighbours changes nothing in the recorded stream. */
        rl_run_or_die_(WORK_DIR "/hit --envs 1 --episodes 2 --seed 77"
                       " --out " WORK_DIR "/solo.k26epi > /dev/null");

        K26RlEpisodeReader *ra = NULL, *rs = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/a.k26epi", &ra) ==
               K26RL_OK);
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/solo.k26epi", &rs) ==
               K26RL_OK);
        K26RlEpisodeData da, ds;
        memset(&da, 0, sizeof da);
        memset(&ds, 0, sizeof ds);
        ASSERT(k26rl_episode_read(ra, 0, 0, 0, &da) == K26RL_OK);
        ASSERT(k26rl_episode_read(rs, 0, 0, 0, &ds) == K26RL_OK);
        printf("  environment 0 beside neighbours: %u steps; alone: %u\n",
               da.step_count, ds.step_count);
        ASSERT(da.step_count == ds.step_count);

        /* The recorded runs really did contain contacts, or this arm
         * measures the determinism of everything except the pass.
         * The contact channel is the first of the observation vector,
         * and the record is step-major. */
        int saw = 0;
        double frac = 0.0;
        for (uint32_t t = 0; t < da.step_count; t++) {
            if (da.obs[t * 8 + 0] > 0.5) { saw = 1; frac = da.obs[t * 8 + 1]; }
        }
        printf("  the recorded runs contained a contact: %s (fraction "
               "%.9f)\n", saw ? "yes" : "NO", frac);
        ASSERT(saw);
        ASSERT(fabs(frac - want_frac) < 2.0e-2);

        /* And every recorded channel of environment 0 agrees between
         * the two runs, contact channels included. */
        int same = 1;
        for (uint32_t t = 0; t < da.step_count * 8; t++) {
            if (da.obs[t] != ds.obs[t]) same = 0;
        }
        printf("  environment 0's recorded channels beside neighbours "
               "and alone: %s\n", same ? "identical" : "DIFFER");
        ASSERT(same);
        k26rl_episode_free(&da);
        k26rl_episode_free(&ds);
        k26rl_episode_reader_close(ra);
        k26rl_episode_reader_close(rs);
        printf("  a contacting fixture reproduces byte for byte and "
               "environment 0 is unchanged by its neighbours: OK\n");
        n_pass++;
    }

    /* ---- 4. The channels are published at their names ------------ */
    {
        void *so = rl_dlopen_(WORK_DIR "/hit.rlenv.so");
        RlSurface s;
        rl_resolve_surface_(so, &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(5u, 1u, &env) == K26RL_OK);
        int32_t need = s.spec(env, NULL, 0);
        ASSERT(need > 0);
        uint8_t *buf = (uint8_t *)malloc((size_t)need);
        ASSERT(buf != NULL);
        ASSERT(s.spec(env, buf, (uint32_t)need) == need);

        const char *want[3] = { "hit_hit", "hit_fraction", "hit_speed" };
        for (int i = 0; i < 3; i++) {
            char got[80];
            ASSERT(spec_name_at_(buf, (uint32_t)need, i, got, sizeof got));
            printf("  channel %d is `%s`, expected `%s`\n", i, got, want[i]);
            ASSERT(strcmp(got, want[i]) == 0);
        }
        /* The line-of-sight observe that follows still starts where
         * three channels later puts it, which is the property that
         * shows the width is carried rather than assumed constant. */
        char got[80];
        ASSERT(spec_name_at_(buf, (uint32_t)need, 3, got, sizeof got));
        printf("  channel 3 is `%s`\n", got);
        ASSERT(strcmp(got, "trk_dir_x") == 0);
        free(buf);
        s.destroy(env);
        printf("  the three contact channels are published at their "
               "names and the observe after them is not displaced: "
               "OK\n");
        n_pass++;
    }

    /* ---- 5. Arrest, measured after the step it happens on -------- *
     *
     * Every arm above ends the episode on the contacting step, which
     * leaves what the resolution does to the world unexamined: the
     * velocity merge that removes the pair's relative motion is never
     * read, because nothing reads anything after it. This fixture
     * never terminates, so the steps after the contact are ordinary
     * steps.
     *
     * It is driven as a separate process and read back from its
     * recorded episode rather than through a live handle. A program
     * carrying two bodies that each bind an assembly stops behaving
     * from the third environment handle created in one process,
     * whatever the seeds, and reloading the same artifact is enough
     * to reach that count; the defect was measured with the collision
     * pass compiled out and has its own item. A batch run has one
     * handle in one process and is unaffected, and the record is what
     * a consumer replays from anyway. */
    {
        rl_stage_("compiling the surviving-arrest artifact", 900u);
        rl_write_file_(WORK_DIR "/live.kfl",
                       COLL_LIVE_KFL("0.0", "        contact arrest\n"));
        rl_compile_(WORK_DIR "/live.kfl", WORK_DIR "/live", WORK_DIR);
        rl_stage_("recording the surviving-arrest episode", 300u);
        rl_run_or_die_(WORK_DIR "/live --envs 1 --episodes 1 --seed 31"
                       " --out " WORK_DIR "/live.k26epi > /dev/null");

        K26RlEpisodeReader *r = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/live.k26epi", &r) ==
               K26RL_OK);
        uint32_t ord = 0, env = 0, ep = 0;
        ASSERT(k26rl_episode_reader_at(r, 0, &ord, &env, &ep) == K26RL_OK);
        K26RlEpisodeData d;
        memset(&d, 0, sizeof d);
        ASSERT(k26rl_episode_read(r, ord, env, ep, &d) == K26RL_OK);
        ASSERT(d.step_count > 12);

        int hs = -1;
        for (uint32_t t = 0; t < d.step_count && hs < 0; t++) {
            if (d.obs[t * LIVE_OBS + LIVE_HIT] > 0.5) hs = (int)t;
        }
        printf("  contact on step %d of %u, episode still running\n",
               hs + 1, d.step_count);
        ASSERT(hs > 0 && hs + 2 < (int)d.step_count);

        /* The reported closing speed against the pair's own line of
         * sight rate on the step before the contact, which is an
         * independent computation of the same normal component and
         * is what gate 16 asks for. The rate is negative while
         * closing, and the reported speed is positive. */
        double pre_rate  = d.obs[(hs - 1) * LIVE_OBS + LIVE_SEP_RATE];
        double hit_speed = d.obs[hs * LIVE_OBS + LIVE_SPEED];
        printf("  reported closing speed %.9f, line of sight rate "
               "before the contact %.9f\n", hit_speed, pre_rate);
        ASSERT(pre_rate < -1.0);
        ASSERT(fabs(hit_speed + pre_rate) < 1.0e-2);
        printf("  the reported speed matches an independently computed "
               "closing rate: OK\n");
        n_pass++;

        /* Arrest removes the pair's relative velocity. Deleting the
         * velocity write-back leaves them closing at six metres a
         * second, and no arm reached it while every fixture ended on
         * the contacting step. */
        double post_rate = d.obs[(hs + 1) * LIVE_OBS + LIVE_SEP_RATE];
        printf("  closing rate before %.9f, after the arrest %.9f\n",
               pre_rate, post_rate);
        ASSERT(fabs(post_rate) < 1.0e-3);
        printf("  arrest removes the pair's relative velocity: OK\n");
        n_pass++;

        /* And having been arrested the pair does not drive further
         * into one another, which is what removing it is for. */
        double r0 = d.obs[hs * LIVE_OBS + LIVE_SEP_RANGE];
        double r1 = d.obs[(hs + 1) * LIVE_OBS + LIVE_SEP_RANGE];
        double r2 = d.obs[(hs + 2) * LIVE_OBS + LIVE_SEP_RANGE];
        printf("  separation at the contact %.6f, then %.6f, then "
               "%.6f\n", r0, r1, r2);
        ASSERT(r1 > r0 * 0.99 && r2 > r0 * 0.99);
        printf("  the arrested pair does not interpenetrate on the "
               "steps after: OK\n");
        n_pass++;
        k26rl_episode_free(&d);
        k26rl_episode_reader_close(r);
    }

    /* ---- 6. Bounce, end to end ----------------------------------- *
     *
     * The bounce surface had never been compiled by any gate: the
     * grammar arms stop at `--check` and `--emit`, which do not reach
     * the C++ stage, so a call written against an older arity
     * survived every suite while the syntax it serves was the reason
     * the design went to a new issue mid-item.
     *
     * The chaser is offset laterally, so it meets the plate away from
     * the plate's centre of mass and the impulse must spin the plate.
     * That is also the only configuration in which an inverse inertia
     * left at the zero matrix is visible, every other term looking
     * correct. */
    {
        /* Two fixtures, because the two claims need different
         * geometry. Restitution is defined on the relative velocity
         * along the CONTACT NORMAL, and for a centred impact the
         * normal and the line of sight between the centres coincide,
         * so the recorded range rate measures it directly. Offset the
         * chaser and they no longer coincide: the range rate then
         * mixes the normal and tangential parts and the ratio is not
         * the coefficient, which an earlier version of this arm
         * asserted anyway and read 0.371 against a declared 0.5.
         * The offset fixture is what shows the spin, and only that. */
        rl_stage_("compiling the centred bounce artifact", 900u);
        rl_write_file_(WORK_DIR "/bcen.kfl",
            COLL_LIVE_KFL("0.0",
                "        contact bounce restitution 0.5 friction 0.3\n"));
        rl_compile_(WORK_DIR "/bcen.kfl", WORK_DIR "/bcen", WORK_DIR);
        rl_stage_("compiling the offset bounce artifact", 900u);
        rl_write_file_(WORK_DIR "/bounce.kfl",
            COLL_LIVE_KFL("2.0",
                "        contact bounce restitution 0.5 friction 0.3\n"));
        rl_compile_(WORK_DIR "/bounce.kfl", WORK_DIR "/bounce", WORK_DIR);
        printf("  a `contact bounce` program compiles and links: OK\n");
        n_pass++;

        /* The centred impact first: normal and line of sight aligned,
         * so the recorded range rate is the normal relative velocity
         * and its ratio across the impulse is the coefficient. */
        rl_stage_("recording the centred bounce episode", 300u);
        rl_run_or_die_(WORK_DIR "/bcen --envs 1 --episodes 1 --seed 41"
                       " --out " WORK_DIR "/bcen.k26epi > /dev/null");
        {
            K26RlEpisodeReader *rc = NULL;
            ASSERT(k26rl_episode_reader_open(WORK_DIR "/bcen.k26epi", &rc)
                   == K26RL_OK);
            uint32_t o2 = 0, e2 = 0, p2 = 0;
            ASSERT(k26rl_episode_reader_at(rc, 0, &o2, &e2, &p2) ==
                   K26RL_OK);
            K26RlEpisodeData dc;
            memset(&dc, 0, sizeof dc);
            ASSERT(k26rl_episode_read(rc, o2, e2, p2, &dc) == K26RL_OK);
            int cs = -1;
            for (uint32_t t = 0; t < dc.step_count && cs < 0; t++) {
                if (dc.obs[t * LIVE_OBS + LIVE_HIT] > 0.5) cs = (int)t;
            }
            ASSERT(cs > 0 && cs + 1 < (int)dc.step_count);
            double cpre  = dc.obs[(cs - 1) * LIVE_OBS + LIVE_SEP_RATE];
            double cpost = dc.obs[(cs + 1) * LIVE_OBS + LIVE_SEP_RATE];
            printf("  centred impact: closing rate before %.9f, after "
                   "%.9f\n", cpre, cpost);
            ASSERT(cpre < -1.0);
            ASSERT(cpost > 0.0);
            printf("  restitution realised %.6f, declared %.6f\n",
                   cpost / -cpre, 0.5);
            ASSERT(fabs(cpost / -cpre - 0.5) < 0.02);
            printf("  a bounce reverses the approach at the declared "
                   "restitution: OK\n");
            n_pass++;
            k26rl_episode_free(&dc);
            k26rl_episode_reader_close(rc);
        }

        rl_stage_("recording the offset bounce episode", 300u);
        rl_run_or_die_(WORK_DIR "/bounce --envs 1 --episodes 1 --seed 37"
                       " --out " WORK_DIR "/bounce.k26epi > /dev/null");

        K26RlEpisodeReader *r = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/bounce.k26epi", &r) ==
               K26RL_OK);
        uint32_t ord = 0, env = 0, ep = 0;
        ASSERT(k26rl_episode_reader_at(r, 0, &ord, &env, &ep) == K26RL_OK);
        K26RlEpisodeData d;
        memset(&d, 0, sizeof d);
        ASSERT(k26rl_episode_read(r, ord, env, ep, &d) == K26RL_OK);

        int hs = -1;
        for (uint32_t t = 0; t < d.step_count && hs < 0; t++) {
            if (d.obs[t * LIVE_OBS + LIVE_HIT] > 0.5) hs = (int)t;
        }
        printf("  contact on step %d of %u\n", hs + 1, d.step_count);
        ASSERT(hs > 0 && hs + 1 < (int)d.step_count);

        /* The offset pair still separates, which is what makes the
         * spin below an impulse and not an artefact. */
        double pre  = d.obs[(hs - 1) * LIVE_OBS + LIVE_SEP_RATE];
        double post = d.obs[(hs + 1) * LIVE_OBS + LIVE_SEP_RATE];
        printf("  offset impact: closing rate before %.9f, after "
               "%.9f\n", pre, post);
        ASSERT(pre < -1.0);
        ASSERT(post > 0.0);

        /* The off-centre impulse spins the plate. Zero here is what
         * an inverse inertia left at the zero matrix produces. */
        double spin = d.obs[(hs + 1) * LIVE_OBS + LIVE_TARGET_WY];
        printf("  the struck body's angular rate about y after the "
               "impulse: %.9e\n", spin);
        /* The defect this guards against produces exactly zero, so
         * the bound only has to clear numerical noise rather than
         * approach the expected value; a bound set just under the
         * measured number would fail on any small change to the
         * fixture's geometry while catching nothing extra. The sign
         * is asserted beside it, since the direction a struck plate
         * turns is fixed by which side of its centre it was struck
         * on and an impulse applied with the wrong sense would keep
         * the magnitude. */
        ASSERT(fabs(spin) > 1.0e-9);
        ASSERT(spin < 0.0);
        printf("  an off-centre impulse spins the body it strikes: "
               "OK\n");
        n_pass++;
        k26rl_episode_free(&d);
        k26rl_episode_reader_close(r);
    }

    rl_stage_done_();
    printf("test_rl_collision: %d gates passed\n", n_pass);
    return 0;
}
