/* test_rl_attitude.c - attitude through the stepping surface.
 *
 * Acceptance:
 *   1. Geometry. A capacity-0 call returns exactly n_envs times the
 *      body count times seven and writes nothing; a call one element
 *      short writes nothing and returns the same requirement; a call
 *      at exactly that capacity fills it. The layout is env-major and
 *      per body in declaration order, the same order the body getter
 *      and the body-name tags use.
 *   2. The advance runs. A body given an angular rate rotates: its
 *      quaternion moves, its rate is unchanged with no torque acting,
 *      and the orientation after a number of steps matches the
 *      analytic rotation about the axis to a stated bound.
 *   3. The subdivision is published and it changes the physics. A
 *      program declaring `substeps 8` publishes 8 in its spec, and a
 *      program differing only in that declaration produces a
 *      different trajectory, which is why the value is part of the
 *      program's identity rather than a tuning knob.
 *   4. Determinism and independence. Two runs of one artifact at one
 *      seed and one action stream give bitwise identical attitude
 *      streams, and environment 0's attitude is bitwise unchanged
 *      when its neighbours are driven differently.
 *   5. The faulted-step contract, stated as it is: after an induced
 *      fault the attitude getter reports the state the fault left,
 *      while the observation getters report the last honestly
 *      computed values, and the two differ.
 *   6. Attitude observation channels carry the body's orientation and
 *      rate under their published names, and agree with the getter.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_attitude_test"

/* One spinning body. The craft is given an angular rate about z at
 * declaration and never torqued, so its attitude is a pure rotation
 * about that axis and the analytic answer is available. */
static const char *const ATT_KFL =
    "form RL_ATT\n"
    "fn world att_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"att.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"
    " omega_x=0.0 omega_y=0.0 omega_z=0.4\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 16\n"
    "        substeps 8\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward att_omega_z\n"
    "    end\n"
    "end\n"
    "end\n";

/* The same program with one subdivision instead of eight. */
static const char *const ATT_KFL_SUB1 =
    "form RL_ATT1\n"
    "fn world att_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"att.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"
    " omega_x=0.0 omega_y=0.0 omega_z=0.4\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 16\n"
    "        substeps 1\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward att_omega_z\n"
    "    end\n"
    "end\n"
    "end\n";

/* A body whose written quaternion has zero norm. The advance cannot
 * produce a finite orientation from it, which ends the episode as a
 * fault through the mapping that already exists. */
static const char *const ATT_KFL_FAULT =
    "form RL_ATTF\n"
    "fn world att_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"att.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0 omega_z=0.2\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 16\n"
    "    end\n"
    "    action zap box 0.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.omega_x = craft.omega_x + zap * 1.0e308 * 1.0e308\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* A body whose quaternion is written to zero norm. There is no
 * orientation with that norm, so the advance cannot produce one and
 * reports divergence; the episode ends as a fault through the mapping
 * that already exists, and no new fault reason appears. */
static const char *const ATT_KFL_ZERO =
    "form RL_ATTZ\n"
    "fn world att_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"att.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0 omega_z=0.2\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 16\n"
    "    end\n"
    "    action zap box 0.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.quat_w = 1.0 - zap\n"
    "        craft.quat_x = 0.0\n"
    "        craft.quat_y = 0.0\n"
    "        craft.quat_z = 0.0\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* A calibration assembly: a uniform box, so the inertia tensor is
 * diagonal and a rate about a principal axis stays there. */
static const char *const ATT_ASM =
    "assembly att_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    "end\n";

/* The same program at a declared subdivision the test substitutes,
 * so convergence can be measured through the declaration rather than
 * through a raw interval. The craft is asymmetric and tumbling, so
 * the splitting error is large enough to have a rate. */
static const char *const ATT_KFL_CONV_FMT =
    "form RL_ATTC%u\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"conv.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0"
    " omega_x=0.9 omega_y=0.9 omega_z=0.12\n"
    "    episode\n"
    "        control_dt 0.02\n"
    "        horizon 200\n"
    "        substeps %u\n"
    "    end\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* An asymmetric assembly: three unequal moments, so the body-frame
 * rate genuinely moves within an interval and the splitting has
 * something to lose. */
static const char *const CONV_ASM =
    "assembly conv_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.6 0.9 0.4\n"
    "    end\n"
    "end\n";

/* A zero-norm quaternion written under a subdivision greater than
 * one. The first sub-advance completes its translation and then the
 * attitude step fails, so the world has advanced part of a control
 * period when the fault is recorded, which the record states as an
 * applied dt of zero. That is the case the suspicion is about. */
static const char *const ATT_KFL_PARTIAL =
    "form RL_ATTP\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"att.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"
    " omega_x=0.0 omega_y=0.0 omega_z=0.2\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 3\n"
    "        substeps 4\n"
    "    end\n"
    "    action zap box 0.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.quat_w = 1.0 - zap\n"
    "        craft.quat_x = 0.0\n"
    "        craft.quat_y = 0.0\n"
    "        craft.quat_z = 0.0\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

static int n_pass = 0;

/* Find the channel index published under `want`, or -1. */
static int spec_channel_(const uint8_t *blob, uint32_t len, const char *want)
{
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint16_t tag = rl_get_u16_(blob + off);
        uint32_t l   = rl_get_u32_(blob + off + 2);
        if (off + 6 + l > len) break;
        if (tag == K26RL_TAG_OBS_CHANNEL_NAME && l > 4) {
            char nm[128];
            uint32_t nl = l - 4;
            if (nl >= sizeof nm) nl = sizeof nm - 1;
            memcpy(nm, blob + off + 10, nl);
            nm[nl] = '\0';
            if (strcmp(nm, want) == 0) {
                return (int)rl_get_u32_(blob + off + 6);
            }
        }
        off += 6 + l;
    }
    return -1;
}

/* Read the substeps tag out of a spec blob. */
static uint32_t spec_substeps_(const uint8_t *blob, uint32_t len)
{
    uint32_t off = 0;
    while (off + 6 <= len) {
        uint16_t tag = rl_get_u16_(blob + off);
        uint32_t l   = rl_get_u32_(blob + off + 2);
        if (off + 6 + l > len) break;
        if (tag == K26RL_TAG_SUBSTEPS && l == 4) {
            return rl_get_u32_(blob + off + 6);
        }
        off += 6 + l;
    }
    return 0;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_attitude")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/att.k26asm", ATT_ASM);
    rl_write_file_(WORK_DIR "/att.kfl", ATT_KFL);
    rl_write_file_(WORK_DIR "/att1.kfl", ATT_KFL_SUB1);
    rl_write_file_(WORK_DIR "/attf.kfl", ATT_KFL_FAULT);
    rl_compile_(WORK_DIR "/att.kfl", WORK_DIR "/att", WORK_DIR);
    rl_compile_(WORK_DIR "/att1.kfl", WORK_DIR "/att1", WORK_DIR);
    rl_compile_(WORK_DIR "/attf.kfl", WORK_DIR "/attf", WORK_DIR);

    void *so = rl_dlopen_(WORK_DIR "/att.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == 0x00010005u);

    /* ---- 1. Geometry -------------------------------------------- */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(11u, 4u, &env) == K26RL_OK);
        int32_t need = s.attitudes(env, NULL, 0);
        /* Two bodies, four environments, seven doubles each. */
        ASSERT(need == 4 * 2 * 7);
        double *buf = (double *)malloc((size_t)need * sizeof(double));
        ASSERT(buf != NULL);
        for (int32_t i = 0; i < need; i++) buf[i] = -12345.0;
        ASSERT(s.attitudes(env, buf, (uint32_t)need - 1) == need);
        for (int32_t i = 0; i < need; i++) ASSERT(buf[i] == -12345.0);
        ASSERT(s.attitudes(env, buf, (uint32_t)need) == need);
        /* Body 0 is earth, which carries no assembly and is never
         * advanced: the identity and a zero rate. Body 1 is the
         * craft, declared with a rate about z. */
        ASSERT(buf[0] == 1.0 && buf[1] == 0.0 && buf[2] == 0.0 &&
               buf[3] == 0.0);
        ASSERT(buf[4] == 0.0 && buf[5] == 0.0 && buf[6] == 0.0);
        ASSERT(buf[7] == 1.0);                 /* craft quat w */
        ASSERT(buf[13] == 0.4);                /* craft omega z */
        printf("  geometry: %d elements, env-major, seven per body, "
               "short call writes nothing: OK\n", need);
        n_pass++;
        free(buf);
        s.destroy(env);
    }

    /* ---- 2. The advance runs ------------------------------------- */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
        double act[1] = { 0.0 };
        const int steps = 8;
        for (int i = 0; i < steps; i++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        double a[2 * 7];
        ASSERT(s.attitudes(env, a, 14) == 14);
        /* Analytic: a rotation of omega * t about z, with the
         * quaternion's scalar part cos(theta/2). About a fixed
         * principal axis the integrator itself is exact, so the whole
         * of the deviation below is physics rather than method: the
         * gravity-gradient torque acts on this craft and a pure
         * rotation is not quite what happens. The bound is stated at
         * the scale that torque produces here, and the arm asserts
         * the deviation is both small and not zero, so neither a
         * missing torque nor a wrong one passes. */
        double t     = 0.5 * steps;
        double theta = 0.4 * t;
        double want_w = cos(theta / 2.0);
        double want_z = sin(theta / 2.0);
        double got_w = a[7], got_z = a[10];
        printf("  after %d steps: quat w %.12f (analytic %.12f), "
               "z %.12f (analytic %.12f)\n", steps, got_w, want_w,
               got_z, want_z);
        double qdev = fabs(got_w - want_w) + fabs(got_z - want_z);
        printf("  deviation from a pure rotation: %.3e\n", qdev);
        ASSERT(qdev < 1e-4);
        ASSERT(qdev > 0.0);
        ASSERT(fabs(a[8]) < 1e-15 && fabs(a[9]) < 1e-15);
        /* The rate is no longer exactly conserved, because a torque
         * now acts: the gravity gradient. The bound below is computed
         * rather than guessed, and the first version of this comment
         * was a decade out, which is why it is written down.
         *
         * The craft is a uniform box of 1000 kg and half extents 1.0
         * by 0.5 by 0.5 m, so its moments are 166.7 and 416.7 kg m^2
         * and their difference is 250. At seven thousand kilometres
         * from a body of mu 3.986e14, three mu over r cubed is
         * 3.486e-6 per second squared, so the torque is at most half
         * of that times the difference, 4.36e-4 N m, and the angular
         * acceleration at most 1.05e-6 rad/s^2. Over the four seconds
         * this arm runs that is 4.18e-6 rad/s, or 1.05e-5 of the
         * declared rate; measured, 6.55e-6. The bound is set at five
         * times the prediction, and the arm also asserts the drift is
         * not zero, so a torque that vanished would fail here as
         * surely as one that was too large. */
        double drift = fabs(a[13] - 0.4) / 0.4;
        printf("  rate after %d steps: %.15f, relative drift from the "
               "declared 0.4 is %.3e\n", steps, a[13], drift);
        ASSERT(drift < 5.0e-5);
        ASSERT(a[13] != 0.4);
        printf("  a declared rate rotates the body, and the "
               "gravity-gradient torque perturbs it within the "
               "predicted %.2e: OK\n", 1.046e-5);
        n_pass++;

        /* 6. The attitude observation channels carry the same values
         * under their published names. */
        uint8_t blob[4096];
        int32_t blen = s.spec(env, blob, sizeof blob);
        ASSERT(blen > 0);
        int base = spec_channel_(blob, (uint32_t)blen, "att_quat_w");
        ASSERT(base >= 0);
        double obs[64];
        ASSERT(s.obs(env, obs) == K26RL_OK);
        for (int k = 0; k < 7; k++) {
            ASSERT(obs[base + k] == a[7 + k]);
        }
        printf("  the attitude channels publish under their names and "
               "equal the getter bitwise: OK\n");
        n_pass++;

        /* Their published mode is the geometric one, because that is
         * what is true: an attitude observe has no observer, applies
         * no light-time correction and no aberration, so the
         * astrometric value would claim a correction that never
         * happens. The line-of-sight observe beside it keeps the mode
         * its own declaration asked for, so this is not a blanket
         * change. */
        uint32_t off = 0;
        int att_modes = 0, los_geometric = 0;
        while (off + 6 <= (uint32_t)blen) {
            uint16_t tag = rl_get_u16_(blob + off);
            uint32_t l   = rl_get_u32_(blob + off + 2);
            if (off + 6 + l > (uint32_t)blen) break;
            if (tag == K26RL_TAG_OBS_CHANNEL_MODE && l == 6) {
                uint32_t ch = rl_get_u32_(blob + off + 6);
                uint16_t md = rl_get_u16_(blob + off + 10);
                if ((int)ch >= base && (int)ch < base + 7) {
                    ASSERT(md == K26RL_OBS_MODE_GEOMETRIC);
                    att_modes++;
                } else if (md == K26RL_OBS_MODE_GEOMETRIC) {
                    los_geometric++;
                }
            }
            off += 6 + l;
        }
        ASSERT(att_modes == 7);
        ASSERT(los_geometric == 5);   /* the observe declared geometric */
        printf("  all seven attitude channels publish the geometric "
               "mode, which is the one that is true: OK\n");
        n_pass++;
        s.destroy(env);
    }

    /* ---- 3. The subdivision is published and it matters ---------- */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(3u, 1u, &env) == K26RL_OK);
        uint8_t blob[4096];
        int32_t blen = s.spec(env, blob, sizeof blob);
        ASSERT(blen > 0);
        ASSERT(spec_substeps_(blob, (uint32_t)blen) == 8u);
        s.destroy(env);

        void *so1 = rl_dlopen_(WORK_DIR "/att1.rlenv.so");
        RlSurface s1;
        rl_resolve_surface_(so1, &s1);
        K26RlEnv *e8 = NULL, *e1 = NULL;
        ASSERT(s.create(5u, 1u, &e8) == K26RL_OK);
        ASSERT(s1.create(5u, 1u, &e1) == K26RL_OK);
        uint8_t b1[4096];
        int32_t l1 = s1.spec(e1, b1, sizeof b1);
        ASSERT(l1 > 0);
        ASSERT(spec_substeps_(b1, (uint32_t)l1) == 1u);
        double act[1] = { 0.5 };
        for (int i = 0; i < 6; i++) {
            ASSERT(s.step(e8, act) == K26RL_OK);
            ASSERT(s1.step(e1, act) == K26RL_OK);
        }
        double o8[64], o1[64];
        ASSERT(s.obs(e8, o8) == K26RL_OK);
        ASSERT(s1.obs(e1, o1) == K26RL_OK);
        int differs = 0;
        for (int k = 0; k < 12; k++) if (o8[k] != o1[k]) differs = 1;
        printf("  substeps 8 and substeps 1 publish 8 and 1, and their "
               "observation streams %s\n",
               differs ? "differ" : "AGREE (they should not)");
        ASSERT(differs);
        printf("  the subdivision is published and changes the physics, "
               "which is why it is part of the program: OK\n");
        n_pass++;
        s.destroy(e8);
        s1.destroy(e1);
    }

    /* ---- D7: what the declared subdivision buys ------------------ */
    {
        /* This arm used to measure the attitude step's convergence
         * rate through the declaration, and it could, because the
         * step was first order and its truncation error was the
         * largest thing moving the body's angular momentum on this
         * fixture. It measured 1.780e-02, 8.844e-03 and 4.407e-03 at
         * one, two and four declared subdivisions, halving as the
         * declaration doubled.
         *
         * The step is not first order any more, and that measurement
         * cannot be taken on this fixture at any declaration. The
         * craft here orbits, so a gravity-gradient torque acts on it,
         * and that torque genuinely changes the body-frame angular
         * momentum: it is physics and not error. The step's own
         * truncation is now four decades below it, and the advance
         * caps the angle turned in one sub-interval at its own bound
         * whatever the program declares, so there is no declaration
         * at which the truncation could climb back above the physics.
         * A gate asserting a ratio here would be asserting a ratio of
         * one quantity that has converged to another.
         *
         * So the two things that are still true are what is asserted,
         * and they are asserted separately because they fail on
         * different defects.
         *
         * First: the conserved quantity has stopped depending on the
         * declaration, which is what convergence looks like from
         * outside. Every declaration agrees to within a small factor
         * and every one of them is far below what the first-order
         * step left behind. A step that lost its order fails this on
         * both counts at once, by four decades on the bound and by a
         * factor of two on the agreement.
         *
         * Second: the declaration still buys something, and what it
         * buys is first order, because what is first order is no
         * longer the attitude step but the splitting around it.
         * Translation advances, then attitude advances over the same
         * sub-interval with the torque held at its start. Halving the
         * declared sub-interval halves that splitting error, and the
         * distance from a finely subdivided run measures it. This is
         * the arm that fails if the sub-advance ordering is ever
         * changed to something that does not converge, and it fails
         * if the attitude step stops being the accurate half of the
         * pair, since then the ratio moves off two.
         *
         * The declarations swept are two, four, eight and sixteen,
         * against a run subdivided two hundred and fifty six ways.
         * One is left out on purpose: at one the whole control period
         * of twenty milliseconds is a single sub-advance and the
         * splitting is outside the range where it has a rate at all,
         * measured at a ratio of 4.14 from one to two against 2.43,
         * 2.10 and 2.08 across the four kept. Asserting a first-order
         * window on a step that is not yet in its asymptotic range is
         * how the first version of this arm came to report 7e-2, 2.5
         * and 0.76 and call it a rate.
         *
         * Measured, on this fixture: drift 7.340e-07, 7.387e-07,
         * 7.411e-07 and 7.423e-07, agreeing to one part in ninety;
         * distance from the reference 1.473e-08, 6.071e-09, 2.887e-09
         * and 1.387e-09. */
        rl_write_file_(WORK_DIR "/conv.k26asm", CONV_ASM);
        const uint32_t subs[4] = { 2, 4, 8, 16 };
        const uint32_t sub_ref = 256;
        double drift[4];
        double att[4][14];
        double att_ref[14];

        for (int i = 0; i < 5; i++) {
            uint32_t nsub = (i < 4) ? subs[i] : sub_ref;
            char src[2048], kfl[256], bin[256];
            snprintf(src, sizeof src, ATT_KFL_CONV_FMT, nsub, nsub);
            snprintf(kfl, sizeof kfl, WORK_DIR "/conv%u.kfl", nsub);
            snprintf(bin, sizeof bin, WORK_DIR "/conv%u", nsub);
            rl_write_file_(kfl, src);
            rl_compile_(kfl, bin, WORK_DIR);
            char so[300];
            snprintf(so, sizeof so, "%s.rlenv.so", bin);
            void *h = rl_dlopen_(so);
            RlSurface sc;
            rl_resolve_surface_(h, &sc);
            K26RlEnv *env = NULL;
            ASSERT(sc.create(31u, 1u, &env) == K26RL_OK);
            double a0[14];
            ASSERT(sc.attitudes(env, a0, 14) == 14);
            /* The momentum magnitude in the body frame is what the
             * getter exposes. The inertia here is the assembly's
             * derived tensor for a uniform box. */
            const double Ixx = 1000.0 * (0.9 * 0.9 + 0.4 * 0.4) / 3.0;
            const double Iyy = 1000.0 * (1.6 * 1.6 + 0.4 * 0.4) / 3.0;
            const double Izz = 1000.0 * (1.6 * 1.6 + 0.9 * 0.9) / 3.0;
            double hx0 = Ixx * a0[11], hy0 = Iyy * a0[12], hz0 = Izz * a0[13];
            double h0 = sqrt(hx0 * hx0 + hy0 * hy0 + hz0 * hz0);
            double act[1] = { 0.0 };
            for (int k = 0; k < 200; k++) {
                if (sc.step(env, act) != K26RL_OK) break;
            }
            double a1[14];
            ASSERT(sc.attitudes(env, a1, 14) == 14);
            double hx1 = Ixx * a1[11], hy1 = Iyy * a1[12], hz1 = Izz * a1[13];
            double h1 = sqrt(hx1 * hx1 + hy1 * hy1 + hz1 * hz1);
            if (i < 4) {
                drift[i] = fabs(h1 - h0) / h0;
                memcpy(att[i], a1, sizeof a1);
            } else {
                memcpy(att_ref, a1, sizeof a1);
            }
            sc.destroy(env);
        }

        printf("  momentum drift by declared substeps: 2 -> %.3e, "
               "4 -> %.3e, 8 -> %.3e, 16 -> %.3e\n",
               drift[0], drift[1], drift[2], drift[3]);
        double worst_ratio = 1.0;
        for (int i = 0; i < 4; i++) {
            ASSERT(drift[i] > 0.0);
            for (int j = 0; j < 4; j++) {
                double r = drift[i] / drift[j];
                if (r > worst_ratio) worst_ratio = r;
            }
        }
        printf("  the widest disagreement between any two of them is a "
               "factor of %.4f\n", worst_ratio);
        /* The first-order step this replaced left 1.780e-02 here and
         * halved it with each doubling of the declaration, so either
         * assertion alone rejects a return to it. */
        for (int i = 0; i < 4; i++) ASSERT(drift[i] < 1.0e-5);
        ASSERT(worst_ratio < 1.05);
        printf("  the conserved quantity no longer depends on the "
               "declared subdivision, which is what a converged "
               "integration looks like from outside: OK\n");
        n_pass++;

        /* The distance from a run subdivided 256 ways, over the
         * craft's whole attitude state: four quaternion components
         * and three body rates. */
        double gap[4];
        for (int i = 0; i < 4; i++) {
            double acc = 0.0;
            for (int c = 7; c < 14; c++) {
                double d = att[i][c] - att_ref[c];
                acc += d * d;
            }
            gap[i] = sqrt(acc);
        }
        printf("  distance from a 256-way subdivision: %.3e, %.3e, "
               "%.3e, %.3e\n", gap[0], gap[1], gap[2], gap[3]);
        double g1 = gap[0] / gap[1], g2 = gap[1] / gap[2],
               g3 = gap[2] / gap[3];
        printf("  ratios as the declaration doubles: %.2f, %.2f, %.2f\n",
               g1, g2, g3);
        ASSERT(gap[0] > gap[1] && gap[1] > gap[2] && gap[2] > gap[3]);
        ASSERT(g1 > 1.6 && g1 < 2.6);
        ASSERT(g2 > 1.6 && g2 < 2.6);
        ASSERT(g3 > 1.6 && g3 < 2.6);
        printf("  the declared subdivision still buys a first-order "
               "improvement, and what is first order is the splitting "
               "and not the attitude step: OK\n");
        n_pass++;
    }

    /* ---- 4. Determinism and vector independence ------------------ */
    {
        K26RlEnv *a1 = NULL, *a2 = NULL;
        ASSERT(s.create(21u, 1u, &a1) == K26RL_OK);
        ASSERT(s.create(21u, 1u, &a2) == K26RL_OK);
        double act[1] = { 0.3 };
        double v1[14], v2[14];
        for (int i = 0; i < 5; i++) {
            ASSERT(s.step(a1, act) == K26RL_OK);
            ASSERT(s.step(a2, act) == K26RL_OK);
        }
        ASSERT(s.attitudes(a1, v1, 14) == 14);
        ASSERT(s.attitudes(a2, v2, 14) == 14);
        ASSERT(memcmp(v1, v2, sizeof v1) == 0);
        s.destroy(a2);

        /* Environment 0 beside neighbours driven differently. */
        K26RlEnv *many = NULL;
        ASSERT(s.create(21u, 4u, &many) == K26RL_OK);
        double acts[4] = { 0.3, -1.0, 0.9, 0.05 };
        for (int i = 0; i < 5; i++) {
            ASSERT(s.step(many, acts) == K26RL_OK);
        }
        double vm[4 * 2 * 7];
        ASSERT(s.attitudes(many, vm, 4 * 2 * 7) == 4 * 2 * 7);
        ASSERT(memcmp(vm, v1, sizeof v1) == 0);
        printf("  two runs agree bitwise, and environment 0 is unmoved by "
               "its neighbours: OK\n");
        n_pass++;
        s.destroy(a1);
        s.destroy(many);
    }

    /* ---- 5. The faulted-step contract ---------------------------- */
    {
        void *sof = rl_dlopen_(WORK_DIR "/attf.rlenv.so");
        RlSurface sf;
        rl_resolve_surface_(sof, &sf);
        K26RlEnv *env = NULL;
        ASSERT(sf.create(7u, 1u, &env) == K26RL_OK);
        double quiet[1] = { 0.0 };
        ASSERT(sf.step(env, quiet) == K26RL_OK);
        double before_obs[64], before_att[14];
        ASSERT(sf.obs(env, before_obs) == K26RL_OK);
        ASSERT(sf.attitudes(env, before_att, 14) == 14);

        double zap[1] = { 1.0 };
        ASSERT(sf.step(env, zap) == K26RL_OK);
        uint32_t fl = 0;
        uint16_t fc = 0;
        ASSERT(sf.flags(env, &fl) == K26RL_OK);
        ASSERT(sf.fault_codes(env, &fc) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_FAULT) != 0);
        ASSERT(fc == (uint16_t)K26RL_E_DIVERGED);

        double after_obs[64], after_att[14];
        ASSERT(sf.obs(env, after_obs) == K26RL_OK);
        ASSERT(sf.attitudes(env, after_att, 14) == 14);
        /* The observation getters hold the last honestly computed
         * values: the fault path leaves their cache alone. */
        ASSERT(memcmp(after_obs, before_obs, 12 * sizeof(double)) == 0);
        /* The attitude getter reads the live world, so it reports what
         * the fault left, and that is not what the observation
         * getters hold. */
        int att_moved = memcmp(after_att, before_att, sizeof after_att) != 0;
        printf("  after the fault: observation getters unchanged, "
               "attitude getter %s\n",
               att_moved ? "reports the state the fault left" : "UNCHANGED");
        ASSERT(att_moved);
        printf("  the faulted-step contract holds as stated, and the two "
               "getters differ: OK\n");
        n_pass++;
        sf.destroy(env);
    }

    /* ---- The zero-norm quaternion ------------------------------- */
    {
        rl_write_file_(WORK_DIR "/attz.kfl", ATT_KFL_ZERO);
        rl_compile_(WORK_DIR "/attz.kfl", WORK_DIR "/attz", WORK_DIR);
        void *soz = rl_dlopen_(WORK_DIR "/attz.rlenv.so");
        RlSurface sz;
        rl_resolve_surface_(soz, &sz);
        K26RlEnv *env = NULL;
        ASSERT(sz.create(9u, 1u, &env) == K26RL_OK);
        /* A unit quaternion first: the step is ordinary. */
        double keep[1] = { 0.0 };
        ASSERT(sz.step(env, keep) == K26RL_OK);
        uint32_t fl = 0;
        ASSERT(sz.flags(env, &fl) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_FAULT) == 0);
        /* Now write every component to zero. */
        double zero[1] = { 1.0 };
        ASSERT(sz.step(env, zero) == K26RL_OK);
        uint16_t fc = 0;
        ASSERT(sz.flags(env, &fl) == K26RL_OK);
        ASSERT(sz.fault_codes(env, &fc) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_FAULT) != 0);
        ASSERT(fc == (uint16_t)K26RL_E_DIVERGED);
        printf("  a quaternion written to zero norm ends the episode as a "
               "fault carrying the existing reason: OK\n");
        n_pass++;
        sz.destroy(env);
    }

    /* ---- The advanced time sums to the control period exactly ---- */
    {
        /* A transition is `substeps` sub-advances of control_dt over
         * substeps, except the last, which takes the remainder. The
         * claim is that the simulated time a transition advances is
         * control_dt exactly, whatever that division rounded to, and
         * the record is where it can be read: the format stores the
         * applied dt of every step. 0.5 divided by 8 is exact, so the
         * fixture that matters is the one whose division is not: this
         * program's control period is 0.1 with 3 subdivisions, and a
         * third of that is not representable. */
        rl_write_file_(WORK_DIR "/sum.kfl",
            "form RL_ATTSUM\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft assembly=\"att.k26asm\" parent=earth"
            " pos_x=7.0e6 vel_y=7546.0 omega_z=0.3\n"
            "    episode\n"
            "        control_dt 0.1\n"
            "        horizon 5\n"
            "        substeps 3\n"
            "    end\n"
            "    observe craft from earth mode=geometric as trk\n"
            "    objective\n"
            "        reward 0.0\n"
            "    end\n"
            "end\n"
            "end\n");
        rl_compile_(WORK_DIR "/sum.kfl", WORK_DIR "/sum", WORK_DIR);
        rl_run_or_die_(WORK_DIR "/sum --envs 1 --episodes 1 --seed 4 "
                       "--out " WORK_DIR "/sum.k26epi > /dev/null");

        K26RlEpisodeReader *r = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/sum.k26epi", &r) ==
               K26RL_OK);
        uint32_t ord = 0, env = 0, ep = 0;
        ASSERT(k26rl_episode_reader_at(r, 0, &ord, &env, &ep) == K26RL_OK);
        K26RlEpisodeData d;
        memset(&d, 0, sizeof d);
        ASSERT(k26rl_episode_read(r, ord, env, ep, &d) == K26RL_OK);
        ASSERT(d.step_count > 0);
        /* Not approximately: the recorded value is the declared one,
         * bit for bit, on every transition. */
        int exact = 1;
        for (uint32_t i = 0; i < d.step_count; i++) {
            if (d.applied_dt[i] != 0.1) exact = 0;
        }
        printf("  %u transitions at control_dt 0.1 over 3 subdivisions: "
               "applied dt %s\n", d.step_count,
               exact ? "exactly 0.1 on every one" : "NOT exact");
        ASSERT(exact);
        printf("  the advanced time sums to the control period exactly "
               "under a division that does not: OK\n");
        n_pass++;
        k26rl_episode_free(&d);
        k26rl_episode_reader_close(r);

        /* The record above says the applied dt is the declared one.
         * What makes that true rather than a claim is the arithmetic
         * the step loop uses, which is reproduced here over a sweep
         * of periods and subdivisions: every one sums to its period
         * exactly. The naive form, every sub-advance taking the
         * quotient, is run beside it and is shown not to, which is
         * what the remainder is for. */
        static const double PERIODS[] = { 0.1, 0.2, 1.0 / 3.0, 0.5,
                                          0.7, 1.0, 2.5 };
        static const uint32_t SUBS[] = { 1, 2, 3, 5, 7, 8, 10, 16, 100 };
        int naive_failures = 0, exact_failures = 0;
        for (unsigned pi = 0; pi < sizeof PERIODS / sizeof PERIODS[0]; pi++) {
            for (unsigned si = 0; si < sizeof SUBS / sizeof SUBS[0]; si++) {
                double period = PERIODS[pi];
                uint32_t n = SUBS[si];
                double q = period / (double)n;
                double advanced = 0.0;
                for (uint32_t k = 0; k < n; k++) {
                    double step_dt = (k + 1u == n) ? (period - advanced) : q;
                    advanced += step_dt;
                }
                if (advanced != period) exact_failures++;
                double naive = 0.0;
                for (uint32_t k = 0; k < n; k++) naive += q;
                if (naive != period) naive_failures++;
            }
        }
        printf("  over %u period and subdivision pairs: the remainder "
               "form is exact %u time(s) out of %u, the quotient form "
               "%u\n",
               (unsigned)(sizeof PERIODS / sizeof PERIODS[0] *
                          sizeof SUBS / sizeof SUBS[0]),
               (unsigned)(sizeof PERIODS / sizeof PERIODS[0] *
                          sizeof SUBS / sizeof SUBS[0]) -
                   (unsigned)exact_failures,
               (unsigned)(sizeof PERIODS / sizeof PERIODS[0] *
                          sizeof SUBS / sizeof SUBS[0]),
               (unsigned)(sizeof PERIODS / sizeof PERIODS[0] *
                          sizeof SUBS / sizeof SUBS[0]) -
                   (unsigned)naive_failures);
        ASSERT(exact_failures == 0);
        /* If the naive form were exact everywhere, the arm above
         * would be testing nothing. */
        ASSERT(naive_failures > 0);
        printf("  the remainder form is exact on every pair and the "
               "quotient form is not, which is why it is there: OK\n");
        n_pass++;
    }

    /* ---- A fault after a partial advance ------------------------ */
    {
        /* When the attitude step fails at a sub-advance, the world
         * has already advanced the translation of that sub-advance
         * and of any before it, while the fault record states an
         * applied dt of zero. The record is true about the
         * transition, which did not complete, and it is made true
         * about the world by the boundary reset, which restores the
         * episode's baseline whole. This arm is what shows the second
         * half: after such a fault the next episode begins at the
         * declared state exactly, not at a state carrying a fraction
         * of a control period of motion. */
        rl_write_file_(WORK_DIR "/attp.kfl", ATT_KFL_PARTIAL);
        rl_compile_(WORK_DIR "/attp.kfl", WORK_DIR "/attp", WORK_DIR);
        void *sop = rl_dlopen_(WORK_DIR "/attp.rlenv.so");
        RlSurface sp2;
        rl_resolve_surface_(sop, &sp2);
        K26RlEnv *env = NULL;
        ASSERT(sp2.create(13u, 1u, &env) == K26RL_OK);

        double quiet[1] = { 0.0 };
        ASSERT(sp2.step(env, quiet) == K26RL_OK);
        uint32_t fl = 0;
        ASSERT(sp2.flags(env, &fl) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_FAULT) == 0);

        double zap[1] = { 1.0 };
        ASSERT(sp2.step(env, zap) == K26RL_OK);
        uint16_t fc = 0;
        ASSERT(sp2.flags(env, &fl) == K26RL_OK);
        ASSERT(sp2.fault_codes(env, &fc) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_FAULT) != 0);
        ASSERT(fc == (uint16_t)K26RL_E_DIVERGED);

        /* The boundary reset. The action slice is ignored on it. */
        ASSERT(sp2.step(env, quiet) == K26RL_OK);
        ASSERT(sp2.flags(env, &fl) == K26RL_OK);
        ASSERT((fl & K26RL_FLAG_RESET_BOUNDARY) != 0);
        double a[14];
        ASSERT(sp2.attitudes(env, a, 14) == 14);
        printf("  after a fault at a sub-advance, the next episode "
               "begins at quat (%.17g, %.17g, %.17g, %.17g) rate z %.17g\n",
               a[7], a[8], a[9], a[10], a[13]);
        /* Exactly the declared state: nothing of the partial advance
         * survives into the episode the record says begins fresh. */
        ASSERT(a[7] == 1.0 && a[8] == 0.0 && a[9] == 0.0 && a[10] == 0.0);
        ASSERT(a[11] == 0.0 && a[12] == 0.0 && a[13] == 0.2);
        printf("  a partial advance leaves nothing behind, so the "
               "record's zero applied dt is true of the world too: OK\n");
        n_pass++;
        sp2.destroy(env);
    }

    printf("test_rl_attitude: %d gates passed\n", n_pass);
    return 0;
}
