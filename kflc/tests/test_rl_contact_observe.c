/* test_rl_contact_observe.c: what the rest of the world reads on the
 * transition a contact lands.
 *
 * A contact is resolved inside a sub-advance: the pair is placed at
 * the configuration the sweep computed, at a moment part way through
 * that sub-advance, while every other body stands at its end. The
 * observation is taken once, after the whole transition. So the
 * question this gate asks is whether the pair rejoins the rest of the
 * world in time before anything reads it: a relative observe of a
 * body that was not struck is taken from the chief's own frame, and a
 * chief left behind at the contact instant carries every one of those
 * readings with it.
 *
 * The fixture is a plate and an approaching craft, as the collision
 * gate's is, with two beacons the craft never touches and a relative
 * observe of each. The episode does not end on the contact, so the
 * steps around it are ordinary steps.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   A fixture that never contacts would pass whatever the step did,
 *   so the first arm asserts a contact happens, inside a period
 *   rather than on its boundary, at the declared closing speed.
 *
 *   A continuity arm with no scale on it would pass for any tolerance
 *   at all, so the tolerance is held against two measurements the run
 *   itself makes: the same residual on the steps around the contact,
 *   which is the noise floor, and the displacement the chief would
 *   carry if it were left at the contact instant, which is the defect
 *   this gate exists for. The second is computed from the published
 *   contact fraction and the chief's own speed, and the arm reports
 *   both, so a tolerance that could not tell them apart is visible
 *   rather than implied.
 *
 *   An arm that only measured the contacting step could not tell a
 *   step that reads correctly from a run whose readings are all
 *   equally wrong, so the residual is measured on every step of the
 *   run and the largest one away from the contact is printed beside
 *   the contacting one.
 *
 *   Relative observes are read from the chief's own frame, so a chief
 *   that stopped moving altogether would carry them all equally and
 *   they would read smoothly again a step later. The chief's own
 *   travel is therefore measured beside them, from the body getter.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <signal.h>
#include <unistd.h>

#include "rl_gate_util.h"
#include <math.h>

#define WORK_DIR "/tmp/kflc_rl_contact_observe_test"

static int n_pass = 0;

/* ---- Wall-clock guard ----------------------------------------------
 *
 * Every arm drives a compiled artifact, and an artifact can livelock;
 * a gate that hangs stops the suite from reporting at all. Each
 * stretch of work that reaches an artifact carries a deadline, and
 * passing it ends the gate loudly with the stage named. */
static const char *rl_stage_name_ = "(none)";
static void rl_deadline_fired_(int sig)
{
    (void)sig;
    (void)!write(2, "deadline passed during: ", 24);
    (void)!write(2, rl_stage_name_, strlen(rl_stage_name_));
    (void)!write(2, "\n", 1);
    _exit(1);
}
static void rl_stage_(const char *name, unsigned secs)
{ rl_stage_name_ = name; alarm(secs); }
static void rl_stage_done_(void) { alarm(0); }

/* The plate and the craft, as the collision gate declares them: a
 * thin box across the approach and a sphere coming at it. */
static const char *const CO_ASM_TARGET =
    "assembly co_target\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component plate\n"
    "        mass 5000.0\n"
    "        at 0 0 0\n"
    "        collider box 6.0 6.0 0.05\n"
    "    end\n"
    "end\n";

static const char *const CO_ASM_CHASER =
    "assembly co_chaser\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider sphere 0 0 0 3.0\n"
    "    end\n"
    "end\n";

/* The craft starts 60 m below the plate and closes at 5.5 m/s, so the
 * sphere's surface reaches the plate's face at a centre separation of
 * 3.05 m, which is 56.95 / 5.5 = 10.3545 s after the episode begins:
 * inside step 11 of a one-second period, and 41.4 sub-advances in, so
 * the contact lands early in its own sub-advance rather than near its
 * end. That is deliberate. What is left unadvanced when a contact is
 * resolved is the rest of its sub-advance, so a contact landing at
 * the very end of one would leave almost nothing behind and the arms
 * below would have almost nothing to measure.
 *
 * The contact separation, 3.05 m, exceeds the 1.375 m the pair closes
 * in one sub-advance, which the collision system requires: a pair
 * travelling from first touch to coincident centres inside one
 * sub-advance hands the integrator a singular field.
 *
 * The two beacons are ordinary bodies with no assembly, so they carry
 * no collider and take no part in the collision pass. They sit a few
 * kilometres from the craft, ahead of it and behind it, and each is
 * observed relative to the craft. Two rather than one because a
 * single one could not show that what moves is the chief rather than
 * the body it is observing. */
#define CO_KFL \
    "form RL_CONOBS\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body target assembly=\"co_target.k26asm\" parent=earth" \
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n" \
    "    astro_body chaser assembly=\"co_chaser.k26asm\" parent=earth" \
    " pos_x=7.0e6 pos_z=-60.0 vel_y=7546.0 vel_z=5.5 quat_w=1.0\n" \
    "    astro_body beacon_lead mass=1.0 parent=earth" \
    " pos_x=7.0e6 pos_y=4000.0 vel_y=7546.0\n" \
    "    astro_body beacon_trail mass=1.0 parent=earth" \
    " pos_x=7.0e6 pos_y=-2500.0 vel_y=7546.0\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 30\n" \
    "        substeps 4\n" \
    "        terminated when hit_hit > 1.5\n" \
    "    end\n" \
    "    action idle box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        chaser.omega_x = idle * 0.0\n" \
    "    end\n" \
    "    observe contact of chaser as hit\n" \
    "    observe relative beacon_lead from chaser as lead\n" \
    "    observe relative beacon_trail from chaser as trail\n" \
    "    observe relative target from chaser as tgt\n" \
    "    objective\n" \
    "        reward hit_hit\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* Three contact channels, then three relative observes of six each.
 * The along-track axis is the one the chief's own motion runs along,
 * so a chief left behind in time moves every reading on it and none
 * of the others by much; it is the component these arms follow. */
#define CO_OBS        21
#define CO_HIT         0
#define CO_FRACTION    1
#define CO_SPEED       2
#define CO_LEAD_RY     4      /* lead_r_y  */
#define CO_TRAIL_RY   10      /* trail_r_y */
#define CO_TGT_RY     16      /* tgt_r_y   */

#define CO_STEPS      25
#define CO_CONTROL_DT  1.0
#define CO_SUBSTEPS    4
#define CO_CLOSE       5.5

/* The continuity bound, in metres. The residual a correct step leaves
 * is the difference between the chord the sweep resolves a contact on
 * and the arc the integrator flies, which is bounded by the curvature
 * over one sub-advance: an eighth of the local gravity times the
 * square of the sub-advance, about 0.06 m here. The bound is an order
 * above that and three orders below the displacement the defect
 * produces, and the arms print all three. */
#define CO_TOL_M       1.0

/* The residual of one series at index t against the straight line its
 * two previous readings lie on. On a smooth trajectory this is the
 * curvature over a step; a reading displaced by a chief that stopped
 * short carries that displacement in full. */
static double residual_(const double *v, int t)
{
    return v[t] - (2.0 * v[t - 1] - v[t - 2]);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGALRM, rl_deadline_fired_);

    if (!rl_libs_present_("test_rl_contact_observe")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/co_target.k26asm", CO_ASM_TARGET);
    rl_write_file_(WORK_DIR "/co_chaser.k26asm", CO_ASM_CHASER);
    rl_write_file_(WORK_DIR "/co.kfl", CO_KFL);

    rl_stage_("compiling the contacting artifact", 900u);
    rl_compile_(WORK_DIR "/co.kfl", WORK_DIR "/co", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/co.rlenv.so"));
    rl_stage_done_();

    void *so = rl_dlopen_(WORK_DIR "/co.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    double lead[CO_STEPS + 1], trail[CO_STEPS + 1], tgt[CO_STEPS + 1];
    double chief[CO_STEPS + 1];
    int    hit_step = -1;
    double frac = -1.0, speed = -1.0, chief_speed = 0.0;

    rl_stage_("driving the contacting artifact", 300u);
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(19u, 1u, &env) == K26RL_OK);
        double obs[CO_OBS], act[1] = { 0.0 };

        ASSERT(s.obs(env, obs) == K26RL_OK);
        /* Five bodies in declaration order, six doubles each: earth,
         * the target, the chief, then the two beacons. The chief's
         * own state is read from the body getter beside the
         * observations, because what a chief carries into a relative
         * observe is where it is, and a run in which every reading
         * moved together would leave the observes smooth. */
        double bod[5 * 6];
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, bod, 30) == 30);
        lead[0]  = obs[CO_LEAD_RY];
        trail[0] = obs[CO_TRAIL_RY];
        tgt[0]   = obs[CO_TGT_RY];
        chief[0] = bod[2 * 6 + 1];
        for (int t = 1; t <= CO_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, bod, 30) == 30);
            lead[t]  = obs[CO_LEAD_RY];
            trail[t] = obs[CO_TRAIL_RY];
            tgt[t]   = obs[CO_TGT_RY];
            chief[t] = bod[2 * 6 + 1];
            if (obs[CO_HIT] > 0.5 && hit_step < 0) {
                hit_step = t;
                frac  = obs[CO_FRACTION];
                speed = obs[CO_SPEED];
            }
        }
        /* The chief's own speed: it is what turns a duration left
         * unadvanced into a displacement. */
        chief_speed = sqrt(bod[2 * 6 + 3] * bod[2 * 6 + 3] +
                           bod[2 * 6 + 4] * bod[2 * 6 + 4] +
                           bod[2 * 6 + 5] * bod[2 * 6 + 5]);
        s.destroy(env);
    }
    rl_stage_done_();

    /* ---- 1. A contact happens, inside a period ------------------- */
    printf("  contact on step %d, fraction %.9f, speed %.9f\n",
           hit_step, frac, speed);
    ASSERT(hit_step >= 3);
    ASSERT(hit_step + 1 < CO_STEPS);
    ASSERT(frac > 0.0 && frac < 1.0);
    /* The closing speed the fixture declares. Both bodies share an
     * orbital velocity of 7546 m/s, so a speed reported as a total
     * rather than as the normal component would be three orders
     * larger and this bound would catch it. */
    ASSERT(fabs(speed - CO_CLOSE) < 1.0e-2);
    printf("  the fixture reaches a contact, inside a control period, "
           "at the declared closing speed: OK\n");
    n_pass++;

    /* ---- 2. What a contact leaves unadvanced -------------------- */
    /* The sub-advance the contact lands in, and how much of it the
     * resolution leaves behind: the published fraction is of the
     * whole period, so its position within its own sub-advance is
     * what remains, and the chief carries that duration at its own
     * speed. This is the displacement the defect this gate exists
     * for puts into every relative observe at once. */
    double sub_dt  = CO_CONTROL_DT / (double)CO_SUBSTEPS;
    double within  = frac * (double)CO_SUBSTEPS;
    double tau     = within - floor(within);
    double unadvanced = (1.0 - tau) * sub_dt;
    double lag = chief_speed * unadvanced;
    printf("  chief speed %.6f m/s, %.9f s of the sub-advance left "
           "unadvanced, %.6f m of travel\n",
           chief_speed, unadvanced, lag);
    ASSERT(chief_speed > 1000.0);
    ASSERT(unadvanced > 0.0 && unadvanced < sub_dt);
    /* An arm cannot show a displacement it could not have seen. The
     * bound below is a thousandth of this one, so a chief left at the
     * contact instant fails it by three orders. */
    ASSERT(lag > 100.0 * CO_TOL_M);
    printf("  the displacement a chief left at the contact instant "
           "would carry is %.1f times the bound these arms hold: OK\n",
           lag / CO_TOL_M);
    n_pass++;

    /* ---- 3. The unstruck observes are continuous ---------------- */
    double worst_away = 0.0;
    int    worst_at = -1;
    for (int t = 2; t <= CO_STEPS; t++) {
        if (t == hit_step) continue;
        double ra = fabs(residual_(lead, t));
        double rb = fabs(residual_(trail, t));
        double r = ra > rb ? ra : rb;
        if (r > worst_away) { worst_away = r; worst_at = t; }
    }
    double hit_lead  = residual_(lead, hit_step);
    double hit_trail = residual_(trail, hit_step);
    double hit_tgt   = residual_(tgt, hit_step);
    printf("  along-track residual on the contacting step: lead "
           "%+.6f m, trail %+.6f m, struck body %+.6f m\n",
           hit_lead, hit_trail, hit_tgt);
    printf("  worst residual away from it %.6f m, on step %d\n",
           worst_away, worst_at);
    /* The floor the run itself establishes. A run whose readings were
     * all equally wrong would raise this and the arm would say so
     * rather than passing on a tolerance that had swallowed the
     * defect. */
    ASSERT(worst_away < CO_TOL_M);
    ASSERT(fabs(hit_lead) < CO_TOL_M);
    ASSERT(fabs(hit_trail) < CO_TOL_M);
    printf("  a contact leaves the relative observes of bodies it did "
           "not strike where they were: OK\n");
    n_pass++;

    /* ---- 4. The chief advances through the transition ----------- */
    /* The observes above are read from the chief's own frame, so a
     * chief that stopped short carries every one of them with it; a
     * chief that stopped altogether carries them all equally and
     * leaves them smooth again a step later. This arm reads the
     * chief's own along-track travel instead, and holds the
     * contacting transition's against the ordinary ones on either
     * side of it. */
    double chief_hit  = residual_(chief, hit_step);
    double chief_next = residual_(chief, hit_step + 1);
    printf("  the chief's own along-track travel: %.6f m on the "
           "transition before the contact, %.6f m on it, %.6f m "
           "after\n", chief[hit_step - 1] - chief[hit_step - 2],
           chief[hit_step] - chief[hit_step - 1],
           chief[hit_step + 1] - chief[hit_step]);
    ASSERT(chief[hit_step] - chief[hit_step - 1] > 0.0);
    ASSERT(fabs(chief_hit) < CO_TOL_M);
    ASSERT(fabs(chief_next) < CO_TOL_M);
    ASSERT(fabs(chief_hit + lag) >= CO_TOL_M);
    printf("  a contact costs the craft it strikes none of its own "
           "travel, on that transition or the next: OK\n");
    n_pass++;

    /* ---- 5. The same arm against the defect it names ------------- */
    /* The readings a chief left at the contact instant would produce
     * are the ones above with its lost travel in them. The arm must
     * reject those: one that did not would pass whatever the step
     * did, which is how this defect survived a suite that already
     * gated the contact channels, the applied duration, and the
     * resolution itself. */
    ASSERT(fabs(hit_lead + lag) >= CO_TOL_M);
    ASSERT(fabs(hit_trail + lag) >= CO_TOL_M);
    printf("  the same arm applied to the readings a chief left "
           "behind would produce, %+.6f m and %+.6f m, fails: OK\n",
           hit_lead + lag, hit_trail + lag);
    n_pass++;

    printf("test_rl_contact_observe: %d arms passed\n", n_pass);
    return 0;
}
