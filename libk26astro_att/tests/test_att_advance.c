/* test_att_advance.c - the attitude advance against closed forms.
 *
 * Acceptance:
 *   1. Torque-free rotation of a symmetric body about a principal
 *      axis. The axis holds and the rate holds: an angular velocity
 *      along a principal axis is a fixed point of Euler's equation,
 *      so any drift here is the integrator's and not the physics.
 *      The orientation reaches the analytic angle after a stated
 *      number of turns.
 *   2. Torque-free rotation of an asymmetric body. The world-frame
 *      angular momentum is conserved: no torque acts, so its
 *      magnitude and direction are constants of the motion, while the
 *      body-frame rate is not. This is the arm that would catch a
 *      missing cross-coupling term, since without it the body-frame
 *      rate would be constant and the world-frame momentum would
 *      swing.
 *   3. A constant body-frame torque about a principal axis reproduces
 *      the analytic angular acceleration.
 *   4. Order. Every bound above is reported with the step count it
 *      holds at, and the error falls at the rate the advance's own
 *      scheme gives. The rate advances by a fourth-order Runge-Kutta
 *      step and the orientation by an exponential map of a rotation
 *      vector built from that step's stages, which is third order,
 *      so the world-frame momentum measured below, being a vector and
 *      therefore carrying the orientation error as well as the rate
 *      error, falls by eight when the step halves. An error that did
 *      not fall at that rate would be a defect wearing a tolerance's
 *      clothes, and a ratio of two would say the first-order step
 *      this advance was built on had come back.
 *   5. Contract. A null vehicle, a negative interval, a singular
 *      inertia tensor, and a non-finite torque are each reported;
 *      a diverged step leaves the last finite state in place; and a
 *      successful step writes through to the bound body.
 *
 * Wire: see libk26astro_att/Makefile.
 */
#include "k26astro_att/att.h"

#include "k26astro_body/body.h"
#include "k26astro_body/attitude.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

static K26AstroVehicle *make_vehicle_(double ixx, double iyy, double izz,
                                      K26AstroBody *body)
{
    K26AstroVehicle *v = k26astro_vehicle_new();
    ASSERT(v != NULL);
    k26astro_vehicle_set_dry_mass(v, 1000.0);
    k26astro_vehicle_set_inertia_diag(v, ixx, iyy, izz);
    if (body) {
        k26astro_body_init(body);
        k26astro_vehicle_bind_body(v, body);
    }
    return v;
}

/* The bound body is the state, so a rate is set there; with no body
 * bound the vehicle's own state is set instead. */
static void set_omega_(K26AstroVehicle *v, double x, double y, double z)
{
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    ASSERT(a != NULL);
    a->omega_body.x = x;
    a->omega_body.y = y;
    a->omega_body.z = z;
    K26AstroBody *b = k26astro_vehicle_body(v);
    if (b) {
        b->omega    = a->omega_body;
        b->attitude = a->q;
    }
}

/* Relative error, for comparisons against a closed form. */
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

/* Rotate a body through `turns` full turns about its z principal axis
 * in `steps` steps, and return the angle error against the analytic
 * result, in radians. */
static double spin_axis_error_(int steps, double turns, double *rate_err)
{
    const double izz = 400.0;
    K26AstroBody body;
    K26AstroVehicle *v = make_vehicle_(300.0, 300.0, izz, &body);
    const double w     = 0.35;                 /* rad/s about z */
    const double total = turns * 2.0 * M_PI / w;
    const double dt    = total / (double)steps;
    set_omega_(v, 0.0, 0.0, w);
    for (int i = 0; i < steps; i++) {
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), dt) ==
               K26ASTRO_ATT_OK);
    }
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    /* The rate about the spin axis is a fixed point: it must not have
     * moved, and no rate about the other two axes may have appeared. */
    *rate_err = fabs(a->omega_body.z - w)
              + fabs(a->omega_body.x) + fabs(a->omega_body.y);
    /* After a whole number of turns the orientation is the identity,
     * up to the quaternion's double cover. The angle of the residual
     * rotation is the error, measured from the vector part rather
     * than from the scalar part: near the identity the scalar part is
     * within rounding of one and an arc cosine of it cannot resolve a
     * small residual at all, while the vector part carries it
     * directly. */
    double vec = sqrt(a->q.x * a->q.x + a->q.y * a->q.y
                      + a->q.z * a->q.z);
    if (vec > 1.0) vec = 1.0;
    double ang = 2.0 * asin(vec);
    k26astro_vehicle_destroy(v);
    return ang;
}

/* Torque-free asymmetric body: the world-frame angular momentum is a
 * constant of the motion. Returns its relative drift over the run. */
static double momentum_drift_(int steps, double seconds, double *body_rate_swing)
{
    K26AstroBody body;
    K26AstroVehicle *v = make_vehicle_(120.0, 300.0, 380.0, &body);
    set_omega_(v, 0.9, 0.9, 0.12);
    K26V3 h0;
    ASSERT(k26astro_att_momentum_world(v, &h0) == K26ASTRO_ATT_OK);
    double dt = seconds / (double)steps;
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    double wx_min = a->omega_body.x, wx_max = a->omega_body.x;
    for (int i = 0; i < steps; i++) {
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), dt) ==
               K26ASTRO_ATT_OK);
        if (a->omega_body.x < wx_min) wx_min = a->omega_body.x;
        if (a->omega_body.x > wx_max) wx_max = a->omega_body.x;
    }
    K26V3 h1;
    ASSERT(k26astro_att_momentum_world(v, &h1) == K26ASTRO_ATT_OK);
    K26V3 d = { h1.x - h0.x, h1.y - h0.y, h1.z - h0.z };
    double drift = v3len_(d) / v3len_(h0);
    *body_rate_swing = (wx_max - wx_min) / fabs(wx_max);
    k26astro_vehicle_destroy(v);
    return drift;
}

int main(void)
{
    printf("torque-free rotation about a principal axis:\n");
    {
        double rate_err_a = 0.0, rate_err_b = 0.0;
        double ang_a = spin_axis_error_(2000, 4.0, &rate_err_a);
        double ang_b = spin_axis_error_(4000, 4.0, &rate_err_b);
        printf("  4 turns: angle error %.3e at 2000 steps, %.3e at 4000\n",
               ang_a, ang_b);
        printf("  rate error %.3e and %.3e\n", rate_err_a, rate_err_b);
        /* The rate about a principal axis is exactly conserved by the
         * update: with the angular velocity along a principal axis
         * the cross term is identically zero, so nothing perturbs it.
         *
         * The orientation is exact too, and that is worth saying
         * rather than tolerating. About a fixed axis the exponential
         * map composes exactly: each step is a rotation of omega
         * times dt about z, and the product of rotations about one
         * axis is the rotation by the sum of the angles. There is no
         * truncation error here to measure and none to watch fall
         * with the step, which is why this arm asserts rounding
         * rather than convergence, and why the convergence arm is the
         * asymmetric one below, where the rate genuinely moves within
         * an interval. Measured here, repeatably across runs but not
         * necessarily across hosts or library versions, since these
         * are accumulated rounding and the orientation update calls
         * the platform's sine and cosine: 3.140e-16 rad after four
         * turns in 2000 steps and 1.284e-16 in 4000. There is no
         * ratio here and none is printed, because there is no
         * truncation to converge: the numbers are rounding, and they
         * do not order themselves by step count. That the exactness
         * survives is the point of the arm. A rate held constant by
         * the physics makes every stage of the Runge-Kutta step
         * agree, so the rotation vector reduces to omega times the
         * step and the exponential map composes the turns exactly,
         * which is what it did before the order was raised and is a
         * property a scheme carrying the orientation through
         * quaternion components would have lost. */
        ASSERT(rate_err_a < 1e-15);
        ASSERT(rate_err_b < 1e-15);
        ASSERT(ang_a < 1e-12);
        ASSERT(ang_b < 1e-12);
        printf("  the spin axis and rate hold exactly, and the "
               "orientation error stays at rounding: OK\n");
        n_pass++;
    }

    printf("torque-free rotation of an asymmetric body:\n");
    {
        double swing_a = 0.0, swing_b = 0.0;
        double drift_a = momentum_drift_(4000, 40.0, &swing_a);
        double drift_b = momentum_drift_(8000, 40.0, &swing_b);
        printf("  momentum drift %.3e at 4000 steps, %.3e at 8000 "
               "(ratio %.2f)\n", drift_a, drift_b, drift_a / drift_b);
        printf("  body-frame rate swing %.3f of its peak, so the "
               "cross-coupling is live\n", swing_a);
        /* The bound is stated at the step count it holds at. This is
         * a hard tumble at nearly a radian and a half a second
         * sampled every ten milliseconds, chosen because it makes the
         * truncation error large enough to measure a convergence rate
         * in. Measured: 1.005e-08 at 4000 steps, 1.254e-09 at 8000,
         * 1.567e-10 at 16000, ratios 8.010 and 8.005.
         *
         * Both numbers gate, and each names a different defect. The
         * bound catches an order that has been lost: the first-order
         * step this advance was built on drifts 7.8e-02 on this
         * fixture, seven decades past the bound below. The ratio
         * catches a bound that has been loosened to accommodate one:
         * a window around eight admits neither the two a first-order
         * orientation gives nor the sixteen that would mean the
         * orientation update had changed to something whose error
         * this gate no longer describes. */
        ASSERT(drift_a < 5e-8);
        double ratio = drift_a / drift_b;
        ASSERT(ratio > 7.0 && ratio < 9.0);
        /* The body-frame rate must genuinely move, or the arm above
         * would be conserving the momentum of a body that is not
         * nutating and would prove nothing about the cross term. */
        ASSERT(swing_a > 0.2);
        printf("  world-frame angular momentum holds to eight decimal "
               "places and its error falls by eight when the step "
               "halves, while the body-frame rate nutates: OK\n");
        n_pass++;
    }

    printf("the same integrator in the regime this capability works in:\n");
    {
        /* The arm above measures a rate; this one measures whether
         * the method is good enough where it is actually used. A
         * vehicle on a docking approach turns slowly, and the step is
         * a control period divided by its declared subdivision.
         * Measured across regimes, drift over the run:
         *
         *   tumble   0.9 rad/s, 40 s, dt 0.01    1.005e-08
         *   tumble   0.9 rad/s, 40 s, dt 0.001   1.004e-11
         *   detumble 0.1 rad/s, 100 s, dt 0.05   2.078e-09
         *   docking  0.01 rad/s, 300 s, dt 0.05  1.716e-12
         *   docking  0.01 rad/s, 300 s, dt 0.20  1.103e-10
         *
         * The same five regimes on the first-order step this advance
         * was built on read 7.8e-2, 7.4e-3, 1.1e-2, 4.1e-4 and
         * 1.7e-3, so the coarsest step here is now better than the
         * finest step there by seven decades. The subdivision is
         * still what keeps a fast tumble representable; what it no
         * longer has to do is buy accuracy the step could not
         * provide. */
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(120.0, 300.0, 380.0, &body);
        set_omega_(v, 0.01, 0.01, 0.002);
        K26V3 h0, h1;
        ASSERT(k26astro_att_momentum_world(v, &h0) == K26ASTRO_ATT_OK);
        for (int i = 0; i < 1500; i++) {     /* 300 s at dt 0.2 */
            ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), 0.2) ==
                   K26ASTRO_ATT_OK);
        }
        ASSERT(k26astro_att_momentum_world(v, &h1) == K26ASTRO_ATT_OK);
        K26V3 d = { h1.x - h0.x, h1.y - h0.y, h1.z - h0.z };
        double drift = v3len_(d) / v3len_(h0);
        printf("  0.01 rad/s over 300 s at a 0.2 s step: momentum drift "
               "%.3e\n", drift);
        ASSERT(drift < 1e-9);
        printf("  the advance holds angular momentum to nine decimal "
               "places on an approach: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }

    printf("constant torque about a principal axis:\n");
    {
        const double izz = 250.0, tau = 5.0, seconds = 10.0;
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(250.0, 250.0, izz, &body);
        const int steps = 20000;
        double dt = seconds / (double)steps;
        for (int i = 0; i < steps; i++) {
            ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, tau), dt) ==
                   K26ASTRO_ATT_OK);
        }
        K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
        double want = tau / izz * seconds;      /* omega = alpha t */
        double err  = fabs(a->omega_body.z - want) / want;
        printf("  omega_z %.12f against the analytic %.12f, relative "
               "error %.3e\n", a->omega_body.z, want, err);
        /* The angular-velocity update is exact for a constant torque
         * about a principal axis: the cross term vanishes and the
         * increment is the same every step, so only rounding
         * separates the sum from the closed form. */
        ASSERT(err < 1e-12);
        ASSERT(fabs(a->omega_body.x) < 1e-15);
        ASSERT(fabs(a->omega_body.y) < 1e-15);
        printf("  a constant principal-axis torque reproduces the "
               "analytic angular acceleration: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }

    printf("the bound body is the state:\n");
    {
        /* The vehicle's own attitude state is left untouched and only
         * the body is written, which is what every writer in the
         * compiler does: a declaration, a reset draw and a step-time
         * assignment all write the body. The advance must therefore
         * load from it. Without that load the vehicle would integrate
         * its own stale copy, the body would be overwritten with the
         * result, and a declared rate would silently vanish; that is
         * the defect this arm exists to catch, and the library's own
         * suite could not see it before. */
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(200.0, 200.0, 200.0, &body);
        K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
        ASSERT(a->omega_body.x == 0.0 && a->omega_body.y == 0.0 &&
               a->omega_body.z == 0.0);
        /* Body only. The working copy still reads zero. */
        body.omega = k26m3d_v3(0.0, 0.0, 0.25);
        ASSERT(a->omega_body.z == 0.0);

        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), 2.0) ==
               K26ASTRO_ATT_OK);
        /* The rate the body carried is the rate that was integrated,
         * and it survives the step. */
        ASSERT(body.omega.z == 0.25);
        ASSERT(a->omega_body.z == 0.25);
        /* Two seconds at a quarter radian per second is half a
         * radian, and the quaternion's vector part is the sine of
         * half of that about z. */
        double want = sin(0.25);
        printf("  body-only write: quat z %.12f, analytic %.12f\n",
               body.attitude.z, want);
        ASSERT(fabs(body.attitude.z - want) < 1e-12);
        ASSERT(fabs(body.attitude.w - cos(0.25)) < 1e-12);
        printf("  a write to the body alone is what the advance "
               "integrates: OK\n");
        n_pass++;

        /* A step of zero duration resynchronises the working copy
         * with the body rather than leaving a stale one behind. */
        body.omega = k26m3d_v3(0.0, 0.0, -0.75);
        ASSERT(a->omega_body.z == 0.25);
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), 0.0) ==
               K26ASTRO_ATT_OK);
        ASSERT(a->omega_body.z == -0.75);
        printf("  a zero-length step resynchronises the working copy: "
               "OK\n");
        n_pass++;

        /* And the momentum query reports the body's state, not a copy
         * that predates the last write to it. */
        body.omega = k26m3d_v3(0.0, 0.0, 0.5);
        K26V3 h;
        ASSERT(k26astro_att_momentum_world(v, &h) == K26ASTRO_ATT_OK);
        printf("  momentum from a body-only write: |h| %.6f "
               "(analytic %.6f)\n", v3len_(h), 200.0 * 0.5);
        ASSERT(relerr_(v3len_(h), 200.0 * 0.5) < 1e-12);
        printf("  the momentum query reads the body too: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }

    printf("gravity-gradient torque:\n");
    {
        /* Against the body library's own expression. That entry sees
         * only a diagonal inertia, so the two must agree exactly when
         * the tensor is diagonal; the full-tensor form exists because
         * a vehicle built from an assembly is not generally diagonal,
         * and the second arm shows the two parting company when it is
         * not, which is the term that would otherwise be dropped. */
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(120.0, 300.0, 380.0, &body);
        /* An orientation that is not the identity, so the rotation
         * into the body frame is doing work. */
        K26V3 axis = { 0.0, 0.0, 1.0 };
        K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
        a->q = k26m3d_quat_from_axis_angle(axis, 0.6);
        body.attitude = a->q;

        const double mu = 3.986004418e14;
        K26V3 r_world = { 6.9e6, 1.1e6, -0.4e6 };
        K26V3 got;
        ASSERT(k26astro_att_gravity_gradient(v, r_world, mu, &got) ==
               K26ASTRO_ATT_OK);

        /* The reference: rotate into the body frame here, then call
         * the body library with the diagonal it can see. */
        K26V3 r_body = k26m3d_quat_rotate_v3(k26m3d_quat_conj(a->q),
                                             r_world);
        double r = v3len_(r_body);
        K26V3 diag = { 120.0, 300.0, 380.0 };
        K26V3 want = k26astro_torque_gravity_gradient(r_body, r, mu, diag);
        printf("  diagonal tensor: got (%.6e, %.6e, %.6e)\n",
               got.x, got.y, got.z);
        printf("  body library:    got (%.6e, %.6e, %.6e)\n",
               want.x, want.y, want.z);
        ASSERT(relerr_(got.x, want.x) < 1e-14);
        ASSERT(relerr_(got.y, want.y) < 1e-14);
        ASSERT(relerr_(got.z, want.z) < 1e-14);
        ASSERT(v3len_(want) > 0.0);
        printf("  the full-tensor form agrees with the body library's "
               "diagonal one to rounding: OK\n");
        n_pass++;

        /* With products of inertia the two part company, which is the
         * whole reason this entry exists. */
        K26M3 full;
        memset(&full, 0, sizeof full);
        full.m[0][0] = 120.0; full.m[1][1] = 300.0; full.m[2][2] = 380.0;
        full.m[0][1] = full.m[1][0] = -40.0;
        full.m[1][2] = full.m[2][1] =  25.0;
        k26astro_vehicle_set_inertia_full(v, full);
        K26V3 got2;
        ASSERT(k26astro_att_gravity_gradient(v, r_world, mu, &got2) ==
               K26ASTRO_ATT_OK);
        double sep = v3len_(k26m3d_v3_sub(got2, want)) / v3len_(want);
        printf("  with products of inertia the torque differs from the "
               "diagonal-only answer by %.1f per cent\n", sep * 100.0);
        ASSERT(sep > 0.01);
        printf("  the products of inertia change the torque, so dropping "
               "them would not be free: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }
    {
        /* The analytic case: a body whose principal axes are aligned
         * with the separation feels no gravity-gradient torque, since
         * the inertia tensor applied to the separation is parallel to
         * it and their cross product vanishes. Any torque here would
         * be a sign or frame error. */
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(120.0, 300.0, 380.0, &body);
        K26V3 r_world = { 7.0e6, 0.0, 0.0 };
        K26V3 got;
        ASSERT(k26astro_att_gravity_gradient(v, r_world, 3.986004418e14,
                                             &got) == K26ASTRO_ATT_OK);
        printf("  aligned with a principal axis: |torque| %.3e\n",
               v3len_(got));
        ASSERT(v3len_(got) < 1e-12);

        /* Rotated by an angle about z, the classic case. With the
         * separation along world x, the body-frame separation is the
         * unit vector (cos t, -sin t, 0), so the inertia tensor
         * applied to it is (Ixx cos t, -Iyy sin t, 0) and their cross
         * product has only a z component, equal to
         * (Ixx - Iyy) sin t cos t. The torque is therefore three mu
         * over two r cubed times (Ixx - Iyy) sin 2t, which at
         * forty-five degrees is three mu over two r cubed times
         * (Ixx - Iyy). Ixx is the smaller moment here, so the torque
         * is negative: the sign is part of the assertion, and the
         * first version of this arm had it backwards while the
         * magnitude matched, which is exactly the error an assertion
         * on magnitude alone would have let through. */
        K26V3 zaxis = { 0.0, 0.0, 1.0 };
        K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
        a->q = k26m3d_quat_from_axis_angle(zaxis, M_PI / 4.0);
        body.attitude = a->q;
        ASSERT(k26astro_att_gravity_gradient(v, r_world, 3.986004418e14,
                                             &got) == K26ASTRO_ATT_OK);
        double r  = 7.0e6;
        double mu = 3.986004418e14;
        double want_z = 1.5 * mu / (r * r * r) * (120.0 - 300.0);
        printf("  at forty-five degrees: torque z %.9e, analytic %.9e\n",
               got.z, want_z);
        ASSERT(fabs(got.x) < 1e-12 && fabs(got.y) < 1e-12);
        ASSERT(relerr_(got.z, want_z) < 1e-12);
        printf("  the closed form for a principal-axis body at "
               "forty-five degrees is reproduced: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }

    printf("contract:\n");
    {
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(100.0, 200.0, 300.0, &body);
        set_omega_(v, 0.1, 0.0, 0.0);

        ASSERT(k26astro_att_step(NULL, k26m3d_v3(0, 0, 0), 1.0) ==
               K26ASTRO_ATT_E_NULL);
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), -1.0) ==
               K26ASTRO_ATT_E_BAD_DT);
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), (double)NAN) ==
               K26ASTRO_ATT_E_BAD_DT);
        ASSERT(k26astro_att_step(v, k26m3d_v3((double)NAN, 0, 0), 1.0) ==
               K26ASTRO_ATT_E_DIVERGED);
        /* Zero interval is accepted and moves nothing. */
        K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
        K26Quat before = a->q;
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), 0.0) ==
               K26ASTRO_ATT_OK);
        ASSERT(a->q.w == before.w && a->q.x == before.x);
        printf("  null, negative and non-finite intervals, a non-finite "
               "torque, and a zero interval: OK\n");
        n_pass++;

        /* A step writes through to the bound body. */
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 0), 1.0) ==
               K26ASTRO_ATT_OK);
        ASSERT(body.attitude.w == a->q.w && body.attitude.x == a->q.x &&
               body.attitude.y == a->q.y && body.attitude.z == a->q.z);
        ASSERT(body.omega.x == a->omega_body.x);
        printf("  a step writes the orientation and rate through to the "
               "bound body: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }
    {
        /* A singular inertia tensor is reported rather than silently
         * doing nothing, which is what a zeroed inverse would give. */
        K26AstroBody body;
        K26AstroVehicle *v = make_vehicle_(100.0, 200.0, 300.0, &body);
        K26M3 zero;
        memset(&zero, 0, sizeof zero);
        k26astro_vehicle_set_inertia_full(v, zero);
        ASSERT(k26astro_att_step(v, k26m3d_v3(0, 0, 1.0), 1.0) ==
               K26ASTRO_ATT_E_SINGULAR);
        printf("  a singular inertia tensor is reported, not stepped "
               "through: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v);
    }
    {
        /* The set entry walks in the order given and skips holes. */
        K26AstroBody b1, b2;
        K26AstroVehicle *v1 = make_vehicle_(10.0, 10.0, 10.0, &b1);
        K26AstroVehicle *v2 = make_vehicle_(10.0, 10.0, 10.0, &b2);
        set_omega_(v1, 0.0, 0.0, 0.5);
        set_omega_(v2, 0.0, 0.0, 0.5);
        K26AstroVehicle *set[3] = { v1, NULL, v2 };
        ASSERT(k26astro_att_step_all(set, 3, NULL, 0.25) ==
               K26ASTRO_ATT_OK);
        ASSERT(b1.attitude.z != 0.0);
        ASSERT(b2.attitude.z == b1.attitude.z);
        printf("  the set entry advances every vehicle it is given and "
               "skips holes: OK\n");
        n_pass++;
        k26astro_vehicle_destroy(v1);
        k26astro_vehicle_destroy(v2);
    }

    printf("test_att_advance: %d check(s) passed\n", n_pass);
    return 0;
}
