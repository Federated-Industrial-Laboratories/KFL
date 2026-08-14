/* test_att_actuators.c - reaction wheels, magnetorquers, thrusters.
 *
 * Acceptance:
 *   1. A wheel commanded at its torque limit reaches its momentum
 *      limit at the analytic time and delivers no further torque in
 *      that direction afterwards, while friction can still slow it.
 *   2. Total angular momentum, the body's own plus the wheels', is
 *      conserved under wheel commands alone. This is the property the
 *      momentum-augmented equation exists to produce: the unaugmented
 *      one does not have it, which the arm demonstrates by computing
 *      what that equation would have given on the same inputs.
 *   3. Friction: the viscous term spins a wheel down at the analytic
 *      exponential rate, the Coulomb term at the analytic linear one,
 *      and the Coulomb term is exactly zero below the dead rate.
 *   4. A magnetorquer's torque is exactly zero when its dipole is
 *      parallel to the field, and equals the analytic cross product
 *      otherwise; the command is clamped to the declared dipole.
 *   5. A thruster at an offset produces the analytic force and the
 *      analytic torque about the centre of mass, and one firing
 *      through the centre of mass produces no torque however hard.
 *
 * On vacuity: every arm here is checked against a closed form rather
 * than against the implementation's own output, and each arm that
 * asserts something is zero also asserts a neighbouring quantity is
 * not, so an actuator wired to do nothing at all would fail rather
 * than pass. The conservation arm carries its own counter-example.
 *
 * Wire: see libk26astro_att/Makefile.
 */
#include "k26astro_att/att.h"

#include "k26astro_body/body.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

static double relerr_(double got, double want)
{
    double d = got - want;
    if (d < 0.0) d = -d;
    double w = want < 0.0 ? -want : want;
    return w > 0.0 ? d / w : d;
}

static double v3len_(K26V3 v)
{
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static K26AstroVehicle *make_vehicle_(double ixx, double iyy, double izz,
                                      K26AstroBody *body)
{
    K26AstroVehicle *v = k26astro_vehicle_new();
    ASSERT(v != NULL);
    k26astro_vehicle_set_dry_mass(v, 1000.0);
    k26astro_vehicle_set_inertia_diag(v, ixx, iyy, izz);
    k26astro_body_init(body);
    k26astro_vehicle_bind_body(v, body);
    return v;
}

static void set_body_rate_(K26AstroVehicle *v, double x, double y, double z)
{
    K26AstroBody *b = k26astro_vehicle_body(v);
    ASSERT(b != NULL);
    b->omega = k26m3d_v3(x, y, z);
}

int main(void)
{
    const K26V3 ZERO = { 0.0, 0.0, 0.0 };

    printf("reaction wheel: torque limit and saturation:\n");
    {
        /* No friction here, so the momentum is the integral of the
         * commanded torque and the saturation time is exact. */
        K26AstroAttWheel w;
        memset(&w, 0, sizeof w);
        w.axis         = k26m3d_v3(0.0, 0.0, 1.0);
        w.spin_inertia = 0.05;
        w.max_momentum = 2.0;
        w.max_torque   = 0.1;
        w.command      = 0.4;          /* over the limit; clamps to 0.1 */
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels = &w;
        act.n_wheels = 1;

        const double dt = 0.001;
        const double t_sat = 2.0 / 0.1;      /* h_max over the clamp */
        int steps = 0;
        double t = 0.0;
        K26V3 tq, stored;
        while (t < t_sat * 1.5) {
            ASSERT(k26astro_att_wheels_step(&act, dt, &tq, &stored) ==
                   K26ASTRO_ATT_OK);
            t += dt;
            steps++;
            if (w.momentum >= w.max_momentum && steps > 0) break;
        }
        printf("  saturates at t = %.6f s, analytic %.6f\n", t, t_sat);
        ASSERT(relerr_(t, t_sat) < 2.0 * dt / t_sat + 1e-12);
        /* Before saturation the reaction is the clamped command, not
         * the command as written. */
        ASSERT(w.momentum == w.max_momentum);

        /* Past the limit, no further torque in that direction. */
        ASSERT(k26astro_att_wheels_step(&act, dt, &tq, &stored) ==
               K26ASTRO_ATT_OK);
        printf("  reaction torque once saturated: (%.3e, %.3e, %.3e)\n",
               tq.x, tq.y, tq.z);
        ASSERT(tq.z == 0.0);
        ASSERT(w.momentum == w.max_momentum);
        ASSERT(relerr_(v3len_(stored), 2.0) < 1e-15);

        /* The command reversed, the wheel comes off the limit at the
         * clamped rate, so saturation clamped the state and not the
         * command. */
        w.command = -0.4;
        ASSERT(k26astro_att_wheels_step(&act, dt, &tq, &stored) ==
               K26ASTRO_ATT_OK);
        printf("  reversing the command gives torque z %.6e (clamped "
               "command %.3f times dt)\n", tq.z, 0.1);
        ASSERT(relerr_(tq.z, 0.1) < 1e-12);
        ASSERT(w.momentum < w.max_momentum);
        /* Friction still acts at the limit, which is what the state
         * clamp buys over clamping the command: a saturated wheel
         * that is no longer commanded slows down. */
        w.command = 0.0;
        w.viscous = 0.01;
        w.momentum = w.max_momentum;
        double at_limit = w.momentum;
        ASSERT(k26astro_att_wheels_step(&act, dt, &tq, &stored) ==
               K26ASTRO_ATT_OK);
        printf("  saturated and uncommanded, friction takes it from "
               "%.6f to %.6f\n", at_limit, w.momentum);
        ASSERT(w.momentum < at_limit);
        ASSERT(tq.z > 0.0);          /* the body feels the slowing */
        w.viscous = 0.0;
        printf("  friction still acts at the limit, which is what "
               "clamping the state rather than the command buys: OK\n");
        n_pass++;

        printf("  a wheel reaches its limit at the analytic time and "
               "delivers nothing further into it: OK\n");
        n_pass++;
    }

    printf("reaction wheel: total angular momentum:\n");
    {
        /* A wheel spun up against a free body. The body's own
         * momentum plus the wheel's is a constant of the motion: the
         * wheel takes momentum from the body and gives it back, and
         * nothing else acts. */
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(120.0, 300.0, 380.0, &body);
        K26AstroAttWheel w;
        memset(&w, 0, sizeof w);
        w.axis         = k26m3d_v3(0.0, 0.0, 1.0);
        w.spin_inertia = 0.05;
        w.max_momentum = 40.0;
        w.max_torque   = 0.5;
        w.command      = 0.3;
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels = &w;
        act.n_wheels = 1;
        set_body_rate_(v, 0.02, -0.01, 0.05);

        K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
        a->q = k26m3d_quat_identity();
        body.attitude = a->q;

        /* The conserved quantity is measured in the world frame,
         * because the body frame turns. */
        K26V3 h0;
        ASSERT(k26astro_att_momentum_world(v, &h0) == K26ASTRO_ATT_OK);
        K26V3 hw0 = { 0.0, 0.0, w.momentum };
        K26V3 hw0w = k26m3d_quat_rotate_v3(a->q, hw0);
        K26V3 tot0 = { h0.x + hw0w.x, h0.y + hw0w.y, h0.z + hw0w.z };

        const double dt = 0.002;
        const int steps = 5000;
        for (int i = 0; i < steps; i++) {
            ASSERT(k26astro_att_step_actuated(v, &act, ZERO, ZERO, dt) ==
                   K26ASTRO_ATT_OK);
        }
        K26V3 h1;
        ASSERT(k26astro_att_momentum_world(v, &h1) == K26ASTRO_ATT_OK);
        K26V3 hw1 = { 0.0, 0.0, w.momentum };
        K26V3 hw1w = k26m3d_quat_rotate_v3(a->q, hw1);
        K26V3 tot1 = { h1.x + hw1w.x, h1.y + hw1w.y, h1.z + hw1w.z };
        K26V3 d = { tot1.x - tot0.x, tot1.y - tot0.y, tot1.z - tot0.z };
        double drift = v3len_(d) / v3len_(tot0);
        printf("  wheel momentum ran from 0 to %.3f N m s over %.1f s\n",
               w.momentum, steps * dt);
        printf("  total angular momentum drift: %.3e\n", drift);
        /* The wheel must actually have done something, or conserving
         * the total would be trivial. */
        ASSERT(w.momentum > 1.0);
        ASSERT(drift < 5e-3);
        printf("  the body plus wheel momentum is conserved while the "
               "wheel spins up: OK\n");
        n_pass++;

        /* The counter-example, computed here on the same inputs: with
         * the gyroscopic term carrying only the body's own momentum,
         * as the unaugmented equation does, the total is not
         * conserved. Without this the arm above would not show that
         * the augmentation is what produces the property. */
        double bad = 0.0;
        {
            K26V3 w_body = k26m3d_v3(0.02, -0.01, 0.05);
            double I[3] = { 120.0, 300.0, 380.0 };
            double hw = 0.0;
            K26V3 tot_first = { 0.0, 0.0, 0.0 }, tot_last = { 0.0, 0.0, 0.0 };
            for (int i = 0; i < steps; i++) {
                double u = 0.3;
                double hdot = u;
                hw += hdot * dt;
                if (hw > 40.0) { hdot = 0.0; hw = 40.0; }
                K26V3 Iw = { I[0] * w_body.x, I[1] * w_body.y,
                             I[2] * w_body.z };
                /* The omission: the stored momentum is left out of
                 * the gyroscopic term. */
                K26V3 gyro = k26m3d_v3_cross(w_body, Iw);
                K26V3 tau  = { -0.0, -0.0, -hdot };
                K26V3 rhs  = { tau.x - gyro.x, tau.y - gyro.y,
                               tau.z - gyro.z };
                w_body.x += rhs.x / I[0] * dt;
                w_body.y += rhs.y / I[1] * dt;
                w_body.z += rhs.z / I[2] * dt;
                K26V3 tot = { I[0] * w_body.x, I[1] * w_body.y,
                              I[2] * w_body.z + hw };
                if (i == 0) tot_first = tot;
                tot_last = tot;
            }
            K26V3 dd = { tot_last.x - tot_first.x, tot_last.y - tot_first.y,
                         tot_last.z - tot_first.z };
            bad = v3len_(dd) / v3len_(tot_first);
        }
        printf("  the same run without the stored momentum in the "
               "gyroscopic term drifts %.3e, %.0f times as far\n",
               bad, bad / drift);
        ASSERT(bad > 20.0 * drift);
        printf("  the augmented terms are what produce the "
               "conservation: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }

    printf("reaction wheel: friction:\n");
    {
        /* Viscous alone: the momentum decays exponentially with time
         * constant J over the viscous coefficient. */
        K26AstroAttWheel w;
        memset(&w, 0, sizeof w);
        w.axis         = k26m3d_v3(1.0, 0.0, 0.0);
        w.spin_inertia = 0.05;
        w.max_momentum = 10.0;
        w.max_torque   = 1.0;
        w.viscous      = 0.002;
        w.momentum     = 1.0;
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels = &w;
        act.n_wheels = 1;

        const double dt = 0.0005, T = 2.0;
        for (int i = 0; i < (int)(T / dt); i++) {
            ASSERT(k26astro_att_wheels_step(&act, dt, NULL, NULL) ==
                   K26ASTRO_ATT_OK);
        }
        double tau_c = w.spin_inertia / w.viscous;
        double want  = 1.0 * exp(-T / tau_c);
        printf("  viscous decay over %.1f s: %.9f, analytic %.9f\n",
               T, w.momentum, want);
        ASSERT(relerr_(w.momentum, want) < 2e-3);
        printf("  the viscous term decays the wheel at the analytic "
               "rate: OK\n");
        n_pass++;
    }
    {
        /* Coulomb alone above the dead rate: a constant torque, so
         * the momentum falls linearly. */
        K26AstroAttWheel w;
        memset(&w, 0, sizeof w);
        w.axis         = k26m3d_v3(1.0, 0.0, 0.0);
        w.spin_inertia = 0.05;
        w.max_momentum = 10.0;
        w.max_torque   = 1.0;
        w.coulomb      = 0.001;
        w.dead_rate    = 0.5;
        w.momentum     = 1.0;                    /* rate 20 rad/s */
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels = &w;
        act.n_wheels = 1;

        const double dt = 0.001, T = 1.0;
        for (int i = 0; i < (int)(T / dt); i++) {
            ASSERT(k26astro_att_wheels_step(&act, dt, NULL, NULL) ==
                   K26ASTRO_ATT_OK);
        }
        double want = 1.0 - 0.001 * T;
        printf("  Coulomb decay over %.1f s: %.12f, analytic %.12f\n",
               w.momentum, want, want);
        ASSERT(relerr_(w.momentum, want) < 1e-12);
        printf("  the Coulomb term decays the wheel linearly at the "
               "analytic rate: OK\n");
        n_pass++;

        /* Below the dead rate it is exactly zero, not merely small:
         * the momentum stops changing altogether. */
        w.momentum = 0.5 * w.dead_rate * w.spin_inertia;   /* half of it */
        double before = w.momentum;
        for (int i = 0; i < 1000; i++) {
            ASSERT(k26astro_att_wheels_step(&act, dt, NULL, NULL) ==
                   K26ASTRO_ATT_OK);
        }
        printf("  below the dead rate: momentum %.17g, unchanged from "
               "%.17g\n", w.momentum, before);
        ASSERT(w.momentum == before);
        /* And it is the dead rate that does it, not a zero
         * coefficient: just above the threshold it moves again. */
        w.momentum = 1.5 * w.dead_rate * w.spin_inertia;
        double before2 = w.momentum;
        ASSERT(k26astro_att_wheels_step(&act, dt, NULL, NULL) ==
               K26ASTRO_ATT_OK);
        ASSERT(w.momentum < before2);
        printf("  the Coulomb term is exactly zero below the dead rate "
               "and acts above it: OK\n");
        n_pass++;
    }

    printf("magnetorquer:\n");
    {
        K26AstroAttTorquer q;
        memset(&q, 0, sizeof q);
        q.axis       = k26m3d_v3(0.0, 1.0, 0.0);
        q.max_dipole = 30.0;
        q.command    = 25.0;
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.torquers = &q;
        act.n_torquers = 1;

        /* Parallel to the field: exactly zero, and the same dipole
         * against a perpendicular field is not, so the zero is the
         * cross product vanishing and not the actuator being inert. */
        K26V3 out;
        ASSERT(k26astro_att_torquers_torque(&act, k26m3d_v3(0.0, 4.0e-5, 0.0),
                                            &out) == K26ASTRO_ATT_OK);
        printf("  dipole parallel to the field: (%.3e, %.3e, %.3e)\n",
               out.x, out.y, out.z);
        ASSERT(out.x == 0.0 && out.y == 0.0 && out.z == 0.0);

        K26V3 field = { 3.0e-5, 0.0, 2.0e-5 };
        ASSERT(k26astro_att_torquers_torque(&act, field, &out) ==
               K26ASTRO_ATT_OK);
        /* m cross B with m = (0, 25, 0). */
        K26V3 m = { 0.0, 25.0, 0.0 };
        K26V3 want = k26m3d_v3_cross(m, field);
        printf("  perpendicular component: (%.6e, %.6e, %.6e), analytic "
               "(%.6e, %.6e, %.6e)\n", out.x, out.y, out.z,
               want.x, want.y, want.z);
        ASSERT(out.x == want.x && out.y == want.y && out.z == want.z);
        ASSERT(v3len_(out) > 0.0);
        printf("  the torque is the dipole crossed with the field, and "
               "exactly zero along it: OK\n");
        n_pass++;

        /* The command is clamped to the declared dipole. */
        q.command = 100.0;
        ASSERT(k26astro_att_torquers_torque(&act, field, &out) ==
               K26ASTRO_ATT_OK);
        K26V3 mmax = { 0.0, 30.0, 0.0 };
        K26V3 wmax = k26m3d_v3_cross(mmax, field);
        ASSERT(out.x == wmax.x && out.y == wmax.y && out.z == wmax.z);
        printf("  a command over the declared dipole clamps to it: OK\n");
        n_pass++;
    }

    /* ---- More than one unit, off the coordinate axes ------------- *
     *
     * Every arm above declares a single unit on a single coordinate
     * axis, and that shape cannot see four distinct mistakes: a loop
     * that steps only its first element, an axis whose components are
     * transposed, a sum that keeps only its first term, and a sign
     * flipped on one component. Each of those is invisible when there
     * is one unit and two of its three axis components are zero. The
     * configuration below has neither property, and it is also the
     * only configuration in which momentum management exists at all,
     * since one wheel on one axis has nothing to dump into.
     */
    printf("more than one unit, off the coordinate axes:\n");
    {
        K26AstroAttWheel w[2];
        memset(w, 0, sizeof w);
        w[0].axis         = k26m3d_v3(0.6, 0.8, 0.0);
        w[0].spin_inertia = 0.05;
        w[0].max_momentum = 15.0;
        w[0].max_torque   = 0.5;
        w[0].command      = 0.3;
        w[1].axis         = k26m3d_v3(0.0, 0.6, 0.8);
        w[1].spin_inertia = 0.04;
        w[1].max_momentum = 12.0;
        w[1].max_torque   = 0.5;
        w[1].command      = -0.2;

        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels   = w;
        act.n_wheels = 2;

        const double dt = 0.5;
        K26V3 torque, stored;
        ASSERT(k26astro_att_wheels_step(&act, dt, &torque, &stored) ==
               K26ASTRO_ATT_OK);

        /* No friction and no saturation here, so each wheel takes
         * exactly its commanded torque for the interval and the
         * analytic values are the sums over both units. Summed in the
         * implementation's own order, so the comparison is exact
         * rather than to a tolerance that would hide a small error. */
        double h0 = 0.3 * dt, h1 = -0.2 * dt;
        K26V3 want_h = { 0.0, 0.0, 0.0 };
        want_h.x += h0 * w[0].axis.x; want_h.y += h0 * w[0].axis.y;
        want_h.z += h0 * w[0].axis.z;
        want_h.x += h1 * w[1].axis.x; want_h.y += h1 * w[1].axis.y;
        want_h.z += h1 * w[1].axis.z;
        K26V3 want_t = { 0.0, 0.0, 0.0 };
        want_t.x -= 0.3 * w[0].axis.x; want_t.y -= 0.3 * w[0].axis.y;
        want_t.z -= 0.3 * w[0].axis.z;
        want_t.x -= -0.2 * w[1].axis.x; want_t.y -= -0.2 * w[1].axis.y;
        want_t.z -= -0.2 * w[1].axis.z;

        printf("  wheel momenta %.12f and %.12f\n",
               w[0].momentum, w[1].momentum);
        printf("  stored (%.12f, %.12f, %.12f), analytic "
               "(%.12f, %.12f, %.12f)\n", stored.x, stored.y, stored.z,
               want_h.x, want_h.y, want_h.z);
        printf("  reaction (%.12f, %.12f, %.12f), analytic "
               "(%.12f, %.12f, %.12f)\n", torque.x, torque.y, torque.z,
               want_t.x, want_t.y, want_t.z);

        /* Both wheels moved, so a loop that stepped only the first
         * would leave the second at zero. */
        ASSERT(w[0].momentum == h0);
        ASSERT(w[1].momentum == h1);
        ASSERT(w[1].momentum != 0.0);
        /* Every component of both vectors, so a transposition or a
         * sign flip on one axis has nowhere to hide. */
        ASSERT(stored.x == want_h.x);
        ASSERT(stored.y == want_h.y);
        ASSERT(stored.z == want_h.z);
        ASSERT(torque.x == want_t.x);
        ASSERT(torque.y == want_t.y);
        ASSERT(torque.z == want_t.z);
        /* And no component is zero, so an assertion of equality on it
         * is not an assertion that nothing happened. */
        ASSERT(stored.x != 0.0 && stored.y != 0.0 && stored.z != 0.0);
        ASSERT(torque.x != 0.0 && torque.y != 0.0 && torque.z != 0.0);
        /* The two axes are genuinely different, so a run that used
         * one wheel's axis for both would differ. */
        ASSERT(w[0].axis.x != w[1].axis.x);
        printf("  two wheels on two off-axis directions give the "
               "analytic stored momentum and reaction in every "
               "component: OK\n");
        n_pass++;
    }
    {
        K26AstroAttTorquer q[2];
        memset(q, 0, sizeof q);
        q[0].axis       = k26m3d_v3(0.8, 0.0, 0.6);
        q[0].max_dipole = 30.0;
        q[0].command    = 20.0;
        q[1].axis       = k26m3d_v3(0.0, 1.0, 0.0);
        q[1].max_dipole = 25.0;
        q[1].command    = -15.0;

        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.torquers   = q;
        act.n_torquers = 2;

        K26V3 field = { 3.0e-5, 1.0e-5, 2.0e-5 };
        K26V3 out;
        ASSERT(k26astro_att_torquers_torque(&act, field, &out) ==
               K26ASTRO_ATT_OK);

        /* The dipoles sum and the sum crosses the field once, which
         * is the implementation's own order. */
        K26V3 m = { 0.0, 0.0, 0.0 };
        m.x += 20.0 * q[0].axis.x; m.y += 20.0 * q[0].axis.y;
        m.z += 20.0 * q[0].axis.z;
        m.x += -15.0 * q[1].axis.x; m.y += -15.0 * q[1].axis.y;
        m.z += -15.0 * q[1].axis.z;
        K26V3 want = k26m3d_v3_cross(m, field);
        printf("  two dipoles: (%.9e, %.9e, %.9e), analytic "
               "(%.9e, %.9e, %.9e)\n", out.x, out.y, out.z,
               want.x, want.y, want.z);
        ASSERT(out.x == want.x && out.y == want.y && out.z == want.z);
        ASSERT(out.x != 0.0 && out.y != 0.0 && out.z != 0.0);

        /* The second unit contributes: the same call with it removed
         * gives a different answer, so a sum that kept only the first
         * term would be a different number and not a rounding of this
         * one. */
        act.n_torquers = 1;
        K26V3 first_only;
        ASSERT(k26astro_att_torquers_torque(&act, field, &first_only) ==
               K26ASTRO_ATT_OK);
        printf("  first unit alone: (%.9e, %.9e, %.9e)\n",
               first_only.x, first_only.y, first_only.z);
        ASSERT(first_only.x != out.x || first_only.y != out.y ||
               first_only.z != out.z);
        printf("  two magnetorquers sum their dipoles before the cross "
               "product, and the second one counts: OK\n");
        n_pass++;
    }

    /* ---- The guards the actuated entry owns ---------------------- *
     *
     * This is the entry an emitted artifact takes. A guard the
     * unactuated entry carries and this one does not protects a path
     * the product does not use, so the same condition is asserted
     * here on the same vehicle rather than assumed to carry over. */
    printf("the actuated entry's own guards:\n");
    {
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(120.0, 300.0, 380.0, &body);
        K26AstroAttWheel w;
        memset(&w, 0, sizeof w);
        w.axis         = k26m3d_v3(0.0, 0.0, 1.0);
        w.spin_inertia = 0.05;
        w.max_momentum = 15.0;
        w.max_torque   = 0.2;
        w.command      = 0.1;
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels = &w;
        act.n_wheels = 1;

        /* Sound tensor first, so the refusal below is the tensor and
         * not the fixture. */
        ASSERT(k26astro_att_step_actuated(v, &act, ZERO, ZERO, 0.1) ==
               K26ASTRO_ATT_OK);

        K26M3 zero;
        memset(&zero, 0, sizeof zero);
        k26astro_vehicle_set_inertia_full(v, zero);
        K26AstroAttStatus st =
            k26astro_att_step_actuated(v, &act, ZERO, ZERO, 0.1);
        printf("  singular tensor through the actuated entry: %d "
               "(%s)\n", (int)st, k26astro_att_status_str(st));
        ASSERT(st == K26ASTRO_ATT_E_SINGULAR);
        /* The unactuated entry on the same vehicle agrees, which is
         * the point: one library, one answer. */
        ASSERT(k26astro_att_step(v, ZERO, 0.1) ==
               K26ASTRO_ATT_E_SINGULAR);
        printf("  both entries report a singular tensor rather than "
               "stepping through it: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }

    printf("thruster:\n");
    {
        K26AstroAttThruster th[2];
        memset(th, 0, sizeof th);
        /* Offset from the centre of mass, firing along -y. */
        th[0].at         = k26m3d_v3(1.05, 0.92, 0.0);
        th[0].dir        = k26m3d_v3(0.0, -1.0, 0.0);
        th[0].max_thrust = 400.0;
        th[0].command    = 0.5;
        /* Through the centre of mass, firing along +x. */
        th[1].at         = k26m3d_v3(0.25, 0.0, 0.0);
        th[1].dir        = k26m3d_v3(1.0, 0.0, 0.0);
        th[1].max_thrust = 400.0;
        th[1].command    = 0.0;

        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.thrusters = th;
        act.n_thrusters = 2;
        act.com = k26m3d_v3(0.25, 0.0, 0.0);

        K26V3 f, t;
        ASSERT(k26astro_att_thrusters_wrench(&act, &f, &t) ==
               K26ASTRO_ATT_OK);
        K26V3 fw = { 0.0, -200.0, 0.0 };
        K26V3 r  = { 1.05 - 0.25, 0.92, 0.0 };
        K26V3 tw = k26m3d_v3_cross(r, fw);
        printf("  force (%.3f, %.3f, %.3f), analytic (%.3f, %.3f, %.3f)\n",
               f.x, f.y, f.z, fw.x, fw.y, fw.z);
        printf("  torque (%.3f, %.3f, %.3f), analytic (%.3f, %.3f, %.3f)\n",
               t.x, t.y, t.z, tw.x, tw.y, tw.z);
        ASSERT(f.x == fw.x && f.y == fw.y && f.z == fw.z);
        ASSERT(t.x == tw.x && t.y == tw.y && t.z == tw.z);
        ASSERT(v3len_(t) > 0.0);
        printf("  a thruster at an offset gives the analytic force and "
               "torque about the centre of mass: OK\n");
        n_pass++;

        /* The one through the centre of mass at full throttle: force
         * but no torque, and the offset one silent, so the zero is
         * the moment arm vanishing rather than the thruster. */
        th[0].command = 0.0;
        th[1].command = 1.0;
        ASSERT(k26astro_att_thrusters_wrench(&act, &f, &t) ==
               K26ASTRO_ATT_OK);
        printf("  through the centre of mass at full throttle: force "
               "(%.1f, %.1f, %.1f), torque (%.3e, %.3e, %.3e)\n",
               f.x, f.y, f.z, t.x, t.y, t.z);
        ASSERT(f.x == 400.0 && f.y == 0.0 && f.z == 0.0);
        ASSERT(t.x == 0.0 && t.y == 0.0 && t.z == 0.0);
        printf("  a thruster through the centre of mass produces force "
               "and no torque: OK\n");
        n_pass++;

        /* The throttle is clamped to the closed unit interval at both
         * ends. */
        th[1].command = 2.5;
        ASSERT(k26astro_att_thrusters_wrench(&act, &f, NULL) ==
               K26ASTRO_ATT_OK);
        ASSERT(f.x == 400.0);
        th[1].command = -3.0;
        ASSERT(k26astro_att_thrusters_wrench(&act, &f, NULL) ==
               K26ASTRO_ATT_OK);
        ASSERT(f.x == 0.0 && f.y == 0.0 && f.z == 0.0);
        printf("  the throttle clamps to the closed unit interval at "
               "both ends: OK\n");
        n_pass++;
    }

    printf("geodetic conversion:\n");
    {
        /* Known points first. A point on the equator at zero
         * longitude sits at the equatorial radius; one on the polar
         * axis sits at the polar radius, which the flattening fixes
         * and which is where a wrong flattening shows up. */
        double lat, lon, alt;
        ASSERT(k26astro_att_geodetic(k26m3d_v3(6378137.0, 0.0, 0.0),
                                     &lat, &lon, &alt) == K26ASTRO_ATT_OK);
        printf("  equator, zero longitude: lat %.9f lon %.9f alt %.6f\n",
               lat, lon, alt);
        ASSERT(fabs(lat) < 1e-12 && fabs(lon) < 1e-12);
        ASSERT(fabs(alt) < 1e-6);

        double b = 6378137.0 * (1.0 - 1.0 / 298.257223563);
        ASSERT(k26astro_att_geodetic(k26m3d_v3(0.0, 0.0, b),
                                     &lat, &lon, &alt) == K26ASTRO_ATT_OK);
        printf("  north pole: lat %.9f alt %.6f (polar radius %.3f)\n",
               lat, alt, b);
        ASSERT(fabs(lat - M_PI / 2.0) < 1e-12);
        ASSERT(fabs(alt) < 1e-6);

        /* The round trip, at a place where latitude and longitude are
         * both away from an axis and the height is a real orbit's.
         * Geodetic latitude differs from the geocentric one by up to
         * about a fifth of a degree, so a conversion that confused
         * them would fail this by kilometres. */
        const double la = 0.9, lo = -1.3, al = 4.2e5;
        double sp = sin(la), cp = cos(la);
        double e2 = (1.0 / 298.257223563) * (2.0 - 1.0 / 298.257223563);
        double N  = 6378137.0 / sqrt(1.0 - e2 * sp * sp);
        K26V3 ecef = {
            (N + al) * cp * cos(lo),
            (N + al) * cp * sin(lo),
            (N * (1.0 - e2) + al) * sp
        };
        ASSERT(k26astro_att_geodetic(ecef, &lat, &lon, &alt) ==
               K26ASTRO_ATT_OK);
        printf("  round trip at 420 km: lat error %.3e rad, lon %.3e, "
               "alt %.6f m\n", fabs(lat - la), fabs(lon - lo),
               fabs(alt - al));
        ASSERT(fabs(lat - la) < 1e-12);
        ASSERT(fabs(lon - lo) < 1e-12);
        ASSERT(fabs(alt - al) < 1e-6);
        /* The geocentric latitude at this point differs from the
         * geodetic one by more than a milliradian, so the round trip
         * above is not passing by treating them as the same. */
        double geoc = atan2(ecef.z, sqrt(ecef.x * ecef.x +
                                         ecef.y * ecef.y));
        printf("  geodetic minus geocentric latitude here: %.6f rad\n",
               la - geoc);
        ASSERT(fabs(la - geoc) > 1e-3);
        printf("  the conversion reproduces known points and round "
               "trips to under a micrometre: OK\n");
        n_pass++;
    }
    {
        /* The local-frame rotation, at a point where every component
         * is distinguishable: up must point away from the centre,
         * east must be perpendicular to the axis, and north must
         * complete the set. */
        const double la = 0.9, lo = -1.3;
        K26V3 up    = k26astro_att_enu_to_ecef(k26m3d_v3(0, 0, 1), la, lo);
        K26V3 east  = k26astro_att_enu_to_ecef(k26m3d_v3(1, 0, 0), la, lo);
        K26V3 north = k26astro_att_enu_to_ecef(k26m3d_v3(0, 1, 0), la, lo);
        printf("  up (%.6f, %.6f, %.6f), east (%.6f, %.6f, %.6f)\n",
               up.x, up.y, up.z, east.x, east.y, east.z);
        /* Up has the sign of the latitude in z and points outward. */
        ASSERT(up.z > 0.0);
        ASSERT(relerr_(up.z, sin(la)) < 1e-15);
        /* East has no z component anywhere. */
        ASSERT(fabs(east.z) < 1e-15);
        /* The three are orthonormal. */
        ASSERT(fabs(k26m3d_v3_dot(up, east)) < 1e-15);
        ASSERT(fabs(k26m3d_v3_dot(up, north)) < 1e-15);
        ASSERT(fabs(k26m3d_v3_dot(east, north)) < 1e-15);
        ASSERT(relerr_(v3len_(up), 1.0) < 1e-15);
        ASSERT(relerr_(v3len_(north), 1.0) < 1e-15);
        printf("  the local frame is orthonormal and up points "
               "outward: OK\n");
        n_pass++;
    }

    printf("the actuated step against the unactuated one:\n");
    {
        /* With no actuators commanded the two paths must agree
         * bitwise, so adding the actuator machinery changes nothing
         * for a body that does not use it. */
        K26AstroBody b1, b2;
        K26AstroVehicle *v1 = make_vehicle_(120.0, 300.0, 380.0, &b1);
        K26AstroVehicle *v2 = make_vehicle_(120.0, 300.0, 380.0, &b2);
        set_body_rate_(v1, 0.3, 0.2, 0.1);
        set_body_rate_(v2, 0.3, 0.2, 0.1);
        K26AstroAttWheel w;
        memset(&w, 0, sizeof w);
        w.axis = k26m3d_v3(0.0, 0.0, 1.0);
        w.spin_inertia = 0.05;
        w.max_momentum = 10.0;
        w.max_torque = 1.0;
        K26AstroAttActuators act;
        memset(&act, 0, sizeof act);
        act.wheels = &w;
        act.n_wheels = 1;

        for (int i = 0; i < 200; i++) {
            ASSERT(k26astro_att_step_actuated(v1, &act, ZERO, ZERO, 0.01) ==
                   K26ASTRO_ATT_OK);
            ASSERT(k26astro_att_step(v2, ZERO, 0.01) == K26ASTRO_ATT_OK);
        }
        printf("  quiescent actuators: quat (%.17g) against (%.17g)\n",
               b1.attitude.z, b2.attitude.z);
        ASSERT(b1.attitude.w == b2.attitude.w);
        ASSERT(b1.attitude.x == b2.attitude.x);
        ASSERT(b1.attitude.y == b2.attitude.y);
        ASSERT(b1.attitude.z == b2.attitude.z);
        ASSERT(b1.omega.x == b2.omega.x);
        printf("  an uncommanded actuator set leaves the advance "
               "bitwise unchanged: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v1);
        k26astro_vehicle_destroy(v2);
    }

    printf("test_att_actuators: %d check(s) passed\n", n_pass);
    return 0;
}
