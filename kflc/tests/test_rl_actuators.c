/* test_rl_actuators.c - actuators through the stepping surface.
 *
 * Acceptance:
 *   1. A wheel command reaches the physics. Commanding a wheel turns
 *      the body the other way, at the rate the momentum exchange
 *      predicts, and the reading channels report the momentum and
 *      rate the command produced.
 *   2. A thruster command reaches translation as well as rotation.
 *      The same declaration accelerates the body and torques it, and
 *      an unfired thruster does neither.
 *   3. Actuator state is episode state: a wheel spun up in one
 *      episode starts the next at zero, so episode k+1 is not a
 *      function of episode k.
 *   4. Determinism: two runs of one artifact at one seed and one
 *      action stream agree bitwise with actuators commanded, and an
 *      environment's actuators are unmoved by its neighbours'.
 *   5. The Fortran split. A program declaring a magnetorquer links
 *      the field model's Fortran runtime; one that does not, does
 *      not, even when the archive and the runtime are both on the
 *      link line. The dependency follows the declaration.
 *   6. The actuator getter reports what the step drove: sizing and
 *      refusals by the surface convention, zeros before any step,
 *      descriptors in declaration order with the body they bind,
 *      commands clamped as the library clamps them, a non-finite
 *      command reported as the zero the library makes of it, and
 *      two environments' records at their own slots.
 *   7. The thruster's applied figure is the step mean: on the step
 *      its tank runs dry the getter reports the force whose impulse
 *      the dynamics imparted, agreeing with the velocity the body
 *      actually gained against a coasting control, and the step
 *      after, tank empty, reports zero.
 *
 * On vacuity: each arm that asserts a quantity is unchanged is paired
 * with one that asserts the same quantity moves when commanded, so a
 * program whose actuators were wired to nothing would fail rather
 * than pass. The split arm compares two artifacts built from the same
 * link line, so it cannot pass by the caller having trimmed it.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_actuators_test"

/* A wheel about z and a thruster offset in y, so a command to either
 * is visible in a different component and neither can be mistaken for
 * the other. */
static const char *const ACT_ASM =
    "assembly act_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    "    wheel yaw\n"
    "        axis 0.0 0.0 1.0\n"
    "        spin_inertia 0.05\n"
    "        max_momentum 15.0\n"
    "        max_torque 0.20\n"
    "    end\n"
    "    thruster rcs_py\n"
    "        at 1.05 0.92 0.0\n"
    "        dir 0.0 -1.0 0.0\n"
    "        thrust 400.0\n"
    "    end\n"
    "end\n";

static const char *const ACT_KFL =
    "form RL_ACT\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"act.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"
    " omega_x=0.0 omega_y=0.0 omega_z=0.0\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 6\n"
    "        substeps 4\n"
    "    end\n"
    "    action spin box -1.0 1.0 default 0.0\n"
    "    action push box 0.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.yaw.torque = spin * 0.2\n"
    "        craft.rcs_py.throttle = push\n"
    "    end\n"
    "    observe attitude of craft as att\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward att_omega_z + craft_reads\n"
    "    end\n"
    "end\n"
    "end\n";

/* The same program with the wheel's readings in the reward, so the
 * reading path is exercised through a real artifact rather than only
 * at the checker. */

/* A tank the first step drains: one thruster whose full-throttle
 * demand over a control period is over three times the propellant
 * carried, so the scale is 1.0 on the first sub-interval, fractional
 * on the second and 0.0 on the rest, and the step mean is none of
 * the four. The thruster pushes through the centre of mass, so the
 * velocity change is the impulse over the mass and nothing turns. */
/* The driving twin of the fixture above: the same assembly and the
 * same action shaping, with a reward that exists, so the getter
 * gates below run on an artifact that compiles. */
static const char *const DRV_KFL =
    "form RL_DRV\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"act.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0 omega_z=0.0\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 6\n"
    "        substeps 4\n"
    "    end\n"
    "    action spin box -1.0 1.0 default 0.0\n"
    "    action push box 0.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.yaw.torque = spin * 0.2\n"
    "        craft.rcs_py.throttle = push\n"
    "    end\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward att_omega_z\n"
    "    end\n"
    "end\n"
    "end\n";

static const char *const TANK_ASM =
    "assembly tank_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0.0 0.0 0.0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    "    component fuel\n"
    "        mass 1.2\n"
    "        at 0.0 0.0 0.0\n"
    "        collider box 0.1 0.1 0.1\n"
    "        propellant\n"
    "    end\n"
    "    thruster main\n"
    "        at 1.0 0.0 0.0\n"
    "        dir -1.0 0.0 0.0\n"
    "        thrust 4000.0\n"
    "        isp_s 100.0\n"
    "    end\n"
    "end\n";

static const char *const TANK_KFL =
    "form RL_TANK\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"tank.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0 omega_z=0.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 6\n"
    "        substeps 4\n"
    "    end\n"
    "    action burn box 0.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.main.throttle = burn\n"
    "    end\n"
    "    observe attitude of craft as att\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range\n"
    "    end\n"
    "end\n"
    "end\n";
static const char *const ACT_KFL_READ =
    "form RL_ACTR\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"act.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0"
    " quat_w=1.0 omega_z=0.0\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 6\n"
    "    end\n"
    "    action spin box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.yaw.torque = spin * 0.2\n"
    "        let h: double = craft.yaw.momentum\n"
    "        let r: double = craft.yaw.rate\n"
    /* The wheel's own momentum, scaled into a body-state key so the
     * getter publishes it. Without this the reset arm below would be
     * blind: a wheel that carried its momentum into the next episode
     * exerts no torque on a body that is at rest, so the leak is
     * invisible in the body rate and has to be read from the wheel.
     * The scale is small enough that the write does not disturb the
     * rates the other arms measure. */
    "        craft.omega_x = h * 1.0e-6 + r * 0.0\n"
    "    end\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward att_omega_z\n"
    "    end\n"
    "end\n"
    "end\n";

/* A magnetorquer, for the link audit and to prove the field chain
 * compiles and runs. */
static const char *const MAG_ASM =
    "assembly mag_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    "    magnetorquer m_y\n"
    "        axis 0.0 1.0 0.0\n"
    "        max_dipole 30.0\n"
    "    end\n"
    "end\n";

static const char *const MAG_KFL =
    "form RL_MAG\n"
    "fn world w\n"
    /* The NAIF id is what names the parent's rotation model, and the
     * rotation model is what makes a longitude mean anything: without
     * it the field chain has no frame and returns zero. */
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " ephem_naif_id=399\n"
    "    astro_body craft assembly=\"mag.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0"
    " quat_w=1.0 omega_z=0.0\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 6\n"
    "    end\n"
    "    action mag box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.m_y.dipole = mag * 30.0\n"
    "    end\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward att_omega_z\n"
    "    end\n"
    "end\n"
    "end\n";

static int n_pass = 0;

/* The needed-library list of an artifact. */
/* Defined symbols from the field model in an artifact. This is the
 * structural observable: archive extraction is what does or does not
 * pull the field model's objects into the link, and its outcome is
 * the presence of that code. The needed-library list below is a
 * consequence of it under this toolchain, but only under this one:
 * positioned before the runtime, -Wl,--no-as-needed records a
 * DT_NEEDED entry for a library whose code was never extracted, so a
 * gate resting on that alone would report a failure that is a link
 * option and not a drift, and could not tell the two apart. Both are
 * asserted, and this is the one that cannot move. */
static int geomag_symbols_(const char *path)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "nm -C %s 2>/dev/null | grep -c k26astro_geomag_ > "
             WORK_DIR "/syms.txt", path);
    (void)!system(cmd);
    FILE *f = fopen(WORK_DIR "/syms.txt", "rb");
    if (!f) return -1;
    int n = 0;
    if (fscanf(f, "%d", &n) != 1) n = -1;
    fclose(f);
    return n;
}

static int needs_fortran_(const char *path)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "readelf -d %s 2>/dev/null | grep -c 'libgfortran' > "
             WORK_DIR "/needed.txt", path);
    (void)!system(cmd);
    FILE *f = fopen(WORK_DIR "/needed.txt", "rb");
    if (!f) return -1;
    int n = 0;
    if (fscanf(f, "%d", &n) != 1) n = -1;
    fclose(f);
    return n;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_actuators")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/act.k26asm", ACT_ASM);
    rl_write_file_(WORK_DIR "/mag.k26asm", MAG_ASM);
    rl_write_file_(WORK_DIR "/act.kfl", ACT_KFL_READ);
    rl_compile_(WORK_DIR "/act.kfl", WORK_DIR "/act", WORK_DIR);

    void *so = rl_dlopen_(WORK_DIR "/act.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);

    /* ---- 1. A wheel command reaches the physics ------------------ */
    {
        K26RlEnv *quiet = NULL, *driven = NULL;
        ASSERT(s.create(5u, 1u, &quiet) == K26RL_OK);
        ASSERT(s.create(5u, 1u, &driven) == K26RL_OK);
        double none[1] = { 0.0 }, full[1] = { 1.0 };
        for (int i = 0; i < 5; i++) {
            ASSERT(s.step(quiet, none) == K26RL_OK);
            ASSERT(s.step(driven, full) == K26RL_OK);
        }
        double aq[14], ad[14];
        ASSERT(s.attitudes(quiet, aq, 14) == 14);
        ASSERT(s.attitudes(driven, ad, 14) == 14);
        printf("  uncommanded omega_z %.12e, commanded %.12e\n",
               aq[13], ad[13]);
        /* The wheel takes momentum one way, so the body turns the
         * other: a positive torque on the wheel gives the body a
         * negative rate. */
        ASSERT(ad[13] < 0.0);
        /* The uncommanded body is not perfectly still: the
         * gravity-gradient torque acts on it, as the attitude gates
         * measure. What matters here is the separation of scales, so
         * the assertion is a ratio rather than an absolute, and it
         * cannot drift as that torque is refined. */
        printf("  commanded rate is %.0f times the uncommanded one\n",
               fabs(ad[13] / aq[13]));
        ASSERT(fabs(ad[13]) > 1.0e4 * fabs(aq[13]));

        /* The rate the momentum exchange predicts: the wheel absorbs
         * torque times time, and the body carries the opposite
         * momentum over its own moment about that axis. The craft is
         * a uniform box of 1000 kg with half extents 1.0, 0.5, 0.5,
         * so its moment about z is m(hx^2 + hy^2)/3. */
        double izz  = 1000.0 * (1.0 * 1.0 + 0.5 * 0.5) / 3.0;
        double hw   = 0.2 * (0.5 * 5.0);        /* torque times time */
        double want = -hw / izz;
        double err  = fabs(ad[13] - want) / fabs(want);
        printf("  body rate %.12e against the momentum-exchange "
               "prediction %.12e, relative error %.3e\n",
               ad[13], want, err);
        ASSERT(err < 1e-3);
        printf("  a wheel command turns the body the other way at the "
               "predicted rate: OK\n");
        n_pass++;
        s.destroy(quiet);
        s.destroy(driven);
    }

    /* ---- 3. Actuator state is episode state ---------------------- */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(9u, 1u, &env) == K26RL_OK);
        double full[1] = { 1.0 };
        /* Spin the wheel up through a whole episode, then over the
         * horizon into the next one. */
        for (int i = 0; i < 6; i++) ASSERT(s.step(env, full) == K26RL_OK);
        double a_end[14];
        ASSERT(s.attitudes(env, a_end, 14) == 14);
        uint32_t fl = 0;
        ASSERT(s.flags(env, &fl) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_TRUNCATED) != 0);
        ASSERT(fabs(a_end[13]) > 1e-9);      /* it really did spin up */

        double none[1] = { 0.0 };
        ASSERT(s.step(env, none) == K26RL_OK);   /* the boundary reset */
        ASSERT(s.flags(env, &fl) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_RESET_BOUNDARY) != 0);
        double a_new[14];
        ASSERT(s.attitudes(env, a_new, 14) == 14);
        printf("  end of episode 0 omega_z %.12e, start of episode 1 "
               "%.12e\n", a_end[13], a_new[13]);
        /* Exactly zero: the reset clears the actuator block, and the
         * episode's initial attitude is the declared one. */
        ASSERT(a_new[13] == 0.0);
        /* And the wheel itself starts empty: one more step at zero
         * command leaves the body still, which it would not if the
         * wheel had carried its momentum across. */
        ASSERT(s.step(env, none) == K26RL_OK);
        ASSERT(s.attitudes(env, a_new, 14) == 14);
        printf("  after a step at zero command in the new episode: "
               "%.12e\n", a_new[13]);
        ASSERT(fabs(a_new[13]) < 1.0e-3 * fabs(a_end[13]));
        /* The wheel itself, read through the channel the program
         * mirrors into omega_x. This is the assertion that fails if
         * the reset stops clearing the actuator block. */
        printf("  wheel reading at the end of episode 0 %.12e, after "
               "the boundary %.12e\n", a_end[11], a_new[11]);
        ASSERT(fabs(a_end[11]) > 1e-12);
        ASSERT(fabs(a_new[11]) < 1.0e-2 * fabs(a_end[11]));
        printf("  a wheel spun up in one episode starts the next "
               "empty: OK\n");
        n_pass++;
        s.destroy(env);
    }

    /* ---- 4. Determinism and independence -------------------------- */
    {
        K26RlEnv *a1 = NULL, *many = NULL;
        ASSERT(s.create(17u, 1u, &a1) == K26RL_OK);
        ASSERT(s.create(17u, 4u, &many) == K26RL_OK);
        double one[1] = { 0.6 };
        double four[4] = { 0.6, -1.0, 0.2, 0.9 };
        for (int i = 0; i < 4; i++) {
            ASSERT(s.step(a1, one) == K26RL_OK);
            ASSERT(s.step(many, four) == K26RL_OK);
        }
        double v1[14], vm[4 * 14];
        ASSERT(s.attitudes(a1, v1, 14) == 14);
        ASSERT(s.attitudes(many, vm, 4 * 14) == 4 * 14);
        ASSERT(memcmp(vm, v1, sizeof v1) == 0);
        /* The neighbours must genuinely differ, or the comparison
         * above would hold for a program that ignored its actions. */
        ASSERT(memcmp(vm + 14, v1, sizeof v1) != 0);
        printf("  environment 0 matches a solo run bitwise while its "
               "neighbours differ: OK\n");
        n_pass++;
        s.destroy(a1);
        s.destroy(many);
    }

    /* ---- 2. A thruster reaches translation and rotation ---------- */
    {
        rl_write_file_(WORK_DIR "/thr.kfl", ACT_KFL);
        /* The reward names a channel that does not exist, so this
         * fixture is the one the refusal arms use; the driving
         * fixture is written below. */
        rl_write_file_(WORK_DIR "/thr.kfl",
            "form RL_THR\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft assembly=\"act.k26asm\" parent=earth"
            " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0 omega_z=0.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 6\n"
            "        substeps 4\n"
            "    end\n"
            "    action push box 0.0 1.0 default 0.0\n"
            "    on_step\n"
            "        craft.rcs_py.throttle = push\n"
            "    end\n"
            "    observe attitude of craft as att\n"
            "    observe craft from earth mode=geometric as trk\n"
            "    objective\n"
            "        reward trk_range\n"
            "    end\n"
            "end\n"
            "end\n");
        rl_compile_(WORK_DIR "/thr.kfl", WORK_DIR "/thr", WORK_DIR);
        void *so2 = rl_dlopen_(WORK_DIR "/thr.rlenv.so");
        RlSurface s2;
        rl_resolve_surface_(so2, &s2);

        K26RlEnv *quiet = NULL, *firing = NULL;
        ASSERT(s2.create(23u, 1u, &quiet) == K26RL_OK);
        ASSERT(s2.create(23u, 1u, &firing) == K26RL_OK);
        double none[1] = { 0.0 }, full[1] = { 1.0 };
        for (int i = 0; i < 5; i++) {
            ASSERT(s2.step(quiet, none) == K26RL_OK);
            ASSERT(s2.step(firing, full) == K26RL_OK);
        }
        double oq[16], of[16], aq[14], af[14];
        ASSERT(s2.obs(quiet, oq) == K26RL_OK);
        ASSERT(s2.obs(firing, of) == K26RL_OK);
        ASSERT(s2.attitudes(quiet, aq, 14) == 14);
        ASSERT(s2.attitudes(firing, af, 14) == 14);

        /* Translation, asserted on the body states rather than on
         * the observation vector. The vector carries the attitude
         * channels beside the tracking ones, so a scan of the whole
         * of it would be satisfied by the rotation alone and would
         * say nothing about whether the thrust reached the
         * integrator. The craft's position is a translation and
         * nothing else. */
        double bq[2 * 6], bf[2 * 6];
        ASSERT(s2.bodies(quiet, K26RL_BODY_REF_ORIGIN, bq, 12) == 12);
        ASSERT(s2.bodies(firing, K26RL_BODY_REF_ORIGIN, bf, 12) == 12);
        double dpos = 0.0, dvel = 0.0;
        for (int i = 0; i < 3; i++) {
            dpos += fabs(bf[6 + i] - bq[6 + i]);
            dvel += fabs(bf[9 + i] - bq[9 + i]);
        }
        printf("  firing moves the craft by %.6e m and %.6e m/s\n",
               dpos, dvel);
        ASSERT(dpos > 0.0);
        ASSERT(dvel > 0.0);
        /* And the observation stream carries it: a consumer that
         * never calls the body getter still sees the difference. */
        int moved = 0;
        for (int i = 0; i < 12; i++) if (oq[i] != of[i]) moved = 1;
        printf("  firing changes the observation stream: %s\n",
               moved ? "yes" : "NO");
        ASSERT(moved);
        /* Rotation: the same declaration torqued it, because the
         * thruster is offset from the centre of mass. */
        printf("  quiet omega_z %.12e, firing %.12e\n", aq[13], af[13]);
        ASSERT(fabs(af[13]) > 1.0e3 * fabs(aq[13]));
        printf("  one thruster declaration accelerates the body and "
               "torques it: OK\n");
        n_pass++;
        s2.destroy(quiet);
        s2.destroy(firing);
    }

    /* ---- 5. The Fortran split ------------------------------------ */
    {
        rl_write_file_(WORK_DIR "/mag.kfl", MAG_KFL);
        rl_compile_(WORK_DIR "/mag.kfl", WORK_DIR "/mag", WORK_DIR);
        int mag_needs = needs_fortran_(WORK_DIR "/mag");
        int act_needs = needs_fortran_(WORK_DIR "/act");
        printf("  magnetorquer program needs libgfortran: %d\n", mag_needs);
        printf("  wheel and thruster program needs it: %d\n", act_needs);
        /* Both were built from the same link line, which carries the
         * field model's archive and the Fortran runtime either way,
         * so the difference is what the program declares and not what
         * the caller passed. */
        ASSERT(mag_needs == 1);
        ASSERT(act_needs == 0);

        int mag_syms = geomag_symbols_(WORK_DIR "/mag");
        int act_syms = geomag_symbols_(WORK_DIR "/act");
        printf("  field model symbols: magnetorquer program %d, wheel "
               "and thruster program %d\n", mag_syms, act_syms);
        /* The count on the left is asserted as non-zero rather than
         * as its present value, so that the field model gaining or
         * losing an entry point is not read here as a split that
         * failed. The count on the right is the whole of the claim
         * and is exact. */
        ASSERT(mag_syms > 0);
        ASSERT(act_syms == 0);
        printf("  the Fortran runtime follows the declaration, not the "
               "link line, and the field model's code is in one "
               "artifact and not the other: OK\n");
        n_pass++;

        /* And the field chain runs: a commanded magnetorquer moves the
         * body, an uncommanded one does not. */
        void *so3 = rl_dlopen_(WORK_DIR "/mag.rlenv.so");
        RlSurface s3;
        rl_resolve_surface_(so3, &s3);
        K26RlEnv *quiet = NULL, *driven = NULL;
        ASSERT(s3.create(29u, 1u, &quiet) == K26RL_OK);
        ASSERT(s3.create(29u, 1u, &driven) == K26RL_OK);
        double none[1] = { 0.0 }, full[1] = { 1.0 };
        for (int i = 0; i < 5; i++) {
            ASSERT(s3.step(quiet, none) == K26RL_OK);
            ASSERT(s3.step(driven, full) == K26RL_OK);
        }
        double aq[14], ad[14];
        ASSERT(s3.attitudes(quiet, aq, 14) == 14);
        ASSERT(s3.attitudes(driven, ad, 14) == 14);
        double moved = fabs(ad[11] - aq[11]) + fabs(ad[12] - aq[12]) +
                       fabs(ad[13] - aq[13]);
        printf("  commanded dipole moves the body rate by %.3e rad/s\n",
               moved);
        ASSERT(moved > 0.0);
        printf("  the field chain produces a torque a command can "
               "steer: OK\n");
        n_pass++;
        s3.destroy(quiet);
        s3.destroy(driven);
    }


    /* ---- 6. The getter reports what the step drove --------------- */
    {
        rl_write_file_(WORK_DIR "/drv.kfl", DRV_KFL);
        rl_compile_(WORK_DIR "/drv.kfl", WORK_DIR "/drv", WORK_DIR);
        void *so6 = rl_dlopen_(WORK_DIR "/drv.rlenv.so");
        RlSurface s6;
        rl_resolve_surface_(so6, &s6);

        K26RlEnv *env = NULL;
        ASSERT(s6.actuators != NULL);
        ASSERT(s6.create(31u, 2u, &env) == K26RL_OK);

        /* Sizing and refusals, the surface's own convention. Two
         * environments, a wheel and a thruster each: 40 doubles. */
        int32_t need = s6.actuators(env, NULL, 0);
        ASSERT(need == 2 * 2 * 10);
        double drv[40];
        for (int i = 0; i < 40; i++) drv[i] = -777.0;
        ASSERT(s6.actuators(env, drv, (uint32_t)need - 1) == need);
        for (int i = 0; i < 40; i++) ASSERT(drv[i] == -777.0);
        ASSERT(s6.actuators(env, NULL, (uint32_t)need) ==
               -(int32_t)K26RL_E_NULL);

        /* Before any step: descriptors filled, nothing applied. The
         * order is fixed, wheels then thrusters, env-major, and the
         * body index is the craft's declaration slot. */
        ASSERT(s6.actuators(env, drv, (uint32_t)need) == need);
        for (int e = 0; e < 2; e++) {
            const double *w = drv + e * 20;
            const double *t = drv + e * 20 + 10;
            ASSERT(w[0] == 1.0 && w[1] == 0.0);
            ASSERT(w[2] == 0.0 && w[3] == 0.0 && w[4] == 0.0);
            ASSERT(w[5] == 0.0 && w[6] == 0.0 && w[7] == 1.0);
            ASSERT(w[8] == 0.0 && w[9] == 0.20);
            ASSERT(t[0] == 1.0 && t[1] == 2.0);
            ASSERT(t[2] == 1.05 && t[3] == 0.92 && t[4] == 0.0);
            ASSERT(t[5] == 0.0 && t[6] == -1.0 && t[7] == 0.0);
            ASSERT(t[8] == 0.0 && t[9] == 400.0);
        }
        printf("  sizing, refusals, zeros before any step, and the "
               "descriptors in order: OK\n");
        n_pass++;

        /* Each environment's records at its own slot, and the wheel's
         * clamp: env 0 commands five times the wheel's limit and a
         * 0.7 throttle, env 1 the mirror image at quarter throttle.
         * The program scales the wheel action by 0.2, so an action of
         * 5.0 commands 1.0 N m against a 0.20 N m limit and the
         * getter must report the limit the library holds it to. */
        double act6[4] = { 5.0, 0.7, -5.0, 0.25 };
        ASSERT(s6.step(env, act6) == K26RL_OK);
        ASSERT(s6.actuators(env, drv, (uint32_t)need) == need);
        ASSERT(drv[8] == 0.20);            /* env 0 wheel, clamped   */
        ASSERT(drv[18] == 0.7 * 400.0);    /* env 0 thruster, 280 N  */
        ASSERT(drv[28] == -0.20);          /* env 1 wheel, clamped   */
        ASSERT(drv[38] == 0.25 * 400.0);   /* env 1 thruster, 100 N  */
        printf("  clamped commands and per-environment slots: 0.20, "
               "280, -0.20, 100: OK\n");
        n_pass++;

        /* A non-finite command reports the zero the library makes of
         * it, on every kind. The library's own clamp returns 0.0 for
         * a non-finite value, so a getter reporting the limit for an
         * infinite command, or a NaN for a NaN, would be reporting a
         * torque the body never felt. */
        double bad[4] = { INFINITY, NAN, NAN, -INFINITY };
        ASSERT(s6.step(env, bad) == K26RL_OK);
        ASSERT(s6.actuators(env, drv, (uint32_t)need) == need);
        ASSERT(drv[8] == 0.0);
        ASSERT(drv[18] == 0.0);
        ASSERT(drv[28] == 0.0);
        ASSERT(drv[38] == 0.0);
        printf("  non-finite commands report zero on every kind: "
               "OK\n");
        n_pass++;

        s6.destroy(env);
    }

    /* ---- 7. The applied figure is the step mean ------------------- */
    {
        rl_write_file_(WORK_DIR "/tank.k26asm", TANK_ASM);
        rl_write_file_(WORK_DIR "/tank.kfl", TANK_KFL);
        rl_compile_(WORK_DIR "/tank.kfl", WORK_DIR "/tank", WORK_DIR);
        void *so7 = rl_dlopen_(WORK_DIR "/tank.rlenv.so");
        RlSurface s7;
        rl_resolve_surface_(so7, &s7);

        /* The expectation, from the declared numbers alone: at full
         * throttle the mass flow is thrust over exhaust speed, each
         * of the four sub-intervals demands flow times a quarter
         * second, the 1.2 kg tank pays the first in full and part of
         * the second, and the mean of the four fractions is the
         * fraction of the step the tank paid for. */
        const double g0 = 9.80665;
        const double flow = 4000.0 / (100.0 * g0);
        const double demand = flow * 0.25;
        const double s1 = (1.2 - demand) / demand;
        const double mean = (1.0 + s1) * 0.25;
        const double expected = mean * 4000.0;

        ASSERT(s7.actuators != NULL);
        K26RlEnv *powered = NULL, *coast = NULL;
        ASSERT(s7.create(7u, 1u, &powered) == K26RL_OK);
        ASSERT(s7.create(7u, 1u, &coast) == K26RL_OK);
        double full7[1] = { 1.0 }, none7[1] = { 0.0 };
        ASSERT(s7.step(powered, full7) == K26RL_OK);
        ASSERT(s7.step(coast, none7) == K26RL_OK);

        double rec[10];
        ASSERT(s7.actuators(powered, rec, 10) == 10);
        printf("  tank runs dry inside the step: getter %.6f N, "
               "declared arithmetic %.6f N\n", rec[8], expected);
        ASSERT(fabs(rec[8] - expected) <= 1.0e-9 * expected);

        /* And the dynamics agree: the velocity the burn added over
         * the coasting control is the reported force times the step
         * over the craft's mass. The tank is 1.2 kg of 1001.2, so
         * half a per cent bounds the mass the burn moved through.
         * A getter reporting the last sub-interval (0 N) or the
         * first (4000 N) fails this by three orders or by 3.4x. */
        double bp[12], bc[12];
        ASSERT(s7.bodies(powered, K26RL_BODY_REF_ORIGIN, bp, 12) == 12);
        ASSERT(s7.bodies(coast, K26RL_BODY_REF_ORIGIN, bc, 12) == 12);
        double dx = bp[9] - bc[9], dy = bp[10] - bc[10],
               dz = bp[11] - bc[11];
        double dv = sqrt(dx * dx + dy * dy + dz * dz);
        double dv_pred = expected * 1.0 / 1000.6;
        printf("  burn added %.5f m/s against %.5f predicted from the "
               "reported force\n", dv, dv_pred);
        ASSERT(fabs(dv - dv_pred) <= 0.005 * dv_pred);

        /* The step after: the tank is empty, the throttle is still
         * open, and zero is what an empty tank imparts. */
        ASSERT(s7.step(powered, full7) == K26RL_OK);
        ASSERT(s7.actuators(powered, rec, 10) == 10);
        ASSERT(rec[8] == 0.0);
        printf("  the dry step reports the impulse it imparted and "
               "the empty one reports zero: OK\n");
        n_pass++;

        s7.destroy(powered);
        s7.destroy(coast);
    }

    printf("test_rl_actuators: %d gates passed\n", n_pass);
    return 0;
}
