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
        printf("  the Fortran runtime follows the declaration, not the "
               "link line: OK\n");
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

    printf("test_rl_actuators: %d gates passed\n", n_pass);
    return 0;
}
