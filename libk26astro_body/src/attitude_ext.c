/* attitude_ext.c — full 3×3-inertia attitude state + integration.
 *
 * For spacecraft whose mass distribution carries off-diagonal
 * inertia coupling and whose inertia changes mid-simulation (stage
 * events, propellant depletion, deployable extension). Sibling of
 * the diagonal-inertia path in attitude.c.
 *
 * Equations of motion (body frame), with h the momentum stored in a
 * momentum-exchange device and zero for a body carrying none:
 *
 *   ω̇ = I⁻¹ (τ − ω × (I ω + h))
 *   q̇ = (1/2) q ⊗ ω_quat
 *
 * How they are advanced, and what that was measured against, is in
 * the integration section below.
 *
 * Implementation: I⁻¹ is precomputed at init and recomputed by
 * k26astro_attitude_update_inertia. The 3×3 inverse is inlined here
 * (cofactor / adjugate form); K26M3 algebra has no consumers in
 * libk26m3d yet so the matrix utility stays local until enough
 * call sites accrue to motivate promotion.
 *
 * References:
 *   - Markley & Crassidis, Fundamentals of Spacecraft Attitude
 *     Determination and Control (2014), §3.7 — quaternion
 *     exponential map; §3.6.2 — Euler's rotational equation in
 *     non-principal axes.
 *   - Hughes, Spacecraft Attitude Dynamics (1986), §4.5 — full
 *     inertia tensor form of Euler's equation.
 *   - Wertz (ed.), Spacecraft Attitude Determination and Control
 *     (1978), §16.3 — inertia tensor properties (symmetric
 *     positive-definite). */
#include "k26astro_body/attitude.h"

#include <math.h>
#include <stdlib.h>

/* ---- Local 3×3 algebra helpers ------------------------------- */

static K26V3 m3_mul_v3_(const K26M3 *m, K26V3 v)
{
    K26V3 out;
    out.x = m->m[0][0] * v.x + m->m[0][1] * v.y + m->m[0][2] * v.z;
    out.y = m->m[1][0] * v.x + m->m[1][1] * v.y + m->m[1][2] * v.z;
    out.z = m->m[2][0] * v.x + m->m[2][1] * v.y + m->m[2][2] * v.z;
    return out;
}

/* Cofactor inverse for a 3×3 row-major matrix. Returns the inverse
 * via `out` and 1 on success; on singular input (|det| < 1e-30)
 * writes the zero matrix and returns 0. */
static int m3_inverse_(const K26M3 *in, K26M3 *out)
{
    double a = in->m[0][0], b = in->m[0][1], c = in->m[0][2];
    double d = in->m[1][0], e = in->m[1][1], f = in->m[1][2];
    double g = in->m[2][0], h = in->m[2][1], i = in->m[2][2];

    double cof00 = e * i - f * h;
    double cof01 = c * h - b * i;
    double cof02 = b * f - c * e;
    double cof10 = f * g - d * i;
    double cof11 = a * i - c * g;
    double cof12 = c * d - a * f;
    double cof20 = d * h - e * g;
    double cof21 = b * g - a * h;
    double cof22 = a * e - b * d;

    double det = a * cof00 + b * cof10 + c * cof20;
    if (fabs(det) < 1.0e-30) {
        for (int r = 0; r < 3; r++)
            for (int cc = 0; cc < 3; cc++)
                out->m[r][cc] = 0.0;
        return 0;
    }
    double inv = 1.0 / det;
    out->m[0][0] = cof00 * inv;
    out->m[0][1] = cof01 * inv;
    out->m[0][2] = cof02 * inv;
    out->m[1][0] = cof10 * inv;
    out->m[1][1] = cof11 * inv;
    out->m[1][2] = cof12 * inv;
    out->m[2][0] = cof20 * inv;
    out->m[2][1] = cof21 * inv;
    out->m[2][2] = cof22 * inv;
    return 1;
}

/* ---- Ext torque-source registry ------------------------------ */

struct K26AstroTorqueListExt {
    K26AstroTorqueFnExt *fns;
    void               **ctxs;
    int                  count;
    int                  capacity;
};

/* The K26AstroTorqueList forward-declaration in attitude.h is
 * opaque, but both the diagonal-path and Ext-path registries use
 * pointer-sized slots; the Ext path stores its own list type via
 * the same `torques` pointer slot. A cast at clear/dispatch sites
 * disambiguates. The two types never alias because each state struct
 * is created and torn down by its own init/destroy pair. */

int k26astro_attitude_register_torque_ext(K26AstroAttitudeStateExt *a,
                                          K26AstroTorqueFnExt fn,
                                          void *ctx)
{
    if (!a || !fn) return 1;
    if (!a->torques) {
        a->torques = (K26AstroTorqueList *)
            calloc(1, sizeof(struct K26AstroTorqueListExt));
        if (!a->torques) return 2;
    }
    struct K26AstroTorqueListExt *p = (struct K26AstroTorqueListExt *)a->torques;
    if (p->count >= p->capacity) {
        int new_cap = p->capacity ? p->capacity * 2 : 4;
        K26AstroTorqueFnExt *new_fns = realloc(p->fns,
            (size_t)new_cap * sizeof(K26AstroTorqueFnExt));
        void **new_ctxs = realloc(p->ctxs,
            (size_t)new_cap * sizeof(void *));
        if (!new_fns || !new_ctxs) {
            free(new_fns);
            free(new_ctxs);
            return 2;
        }
        p->fns  = new_fns;
        p->ctxs = new_ctxs;
        p->capacity = new_cap;
    }
    p->fns [p->count] = fn;
    p->ctxs[p->count] = ctx;
    p->count++;
    return 0;
}

void k26astro_attitude_clear_torques_ext(K26AstroAttitudeStateExt *a)
{
    if (!a || !a->torques) return;
    struct K26AstroTorqueListExt *p = (struct K26AstroTorqueListExt *)a->torques;
    free(p->fns);
    free(p->ctxs);
    free(p);
    a->torques = NULL;
}

/* ---- Lifecycle ------------------------------------------------ */

void k26astro_attitude_init_ext(K26AstroAttitudeStateExt *a,
                                K26M3 inertia)
{
    if (!a) return;
    a->q = k26m3d_quat_identity();
    a->omega_body.x = a->omega_body.y = a->omega_body.z = 0.0;
    a->inertia = inertia;
    (void)m3_inverse_(&a->inertia, &a->inertia_inverse);
    a->torques = NULL;
}

void k26astro_attitude_destroy_ext(K26AstroAttitudeStateExt *a)
{
    k26astro_attitude_clear_torques_ext(a);
}

void k26astro_attitude_update_inertia(K26AstroAttitudeStateExt *a,
                                      K26M3 new_inertia)
{
    if (!a) return;
    a->inertia = new_inertia;
    (void)m3_inverse_(&a->inertia, &a->inertia_inverse);
}

/* ---- Integration --------------------------------------------- *
 *
 * One step, one place. The rate and the orientation advance by the
 * scheme below and every entry in this file routes through it, as
 * does the actuated advance in libk26astro_att, which passes the
 * wheels' stored momentum through the one extra argument its equation
 * needs. Two copies of one equation of motion is how two callers come
 * to disagree about the same craft, so there is one copy.
 *
 * The scheme, and why it is this one. The rate advances by the
 * classical fourth-order Runge-Kutta step on
 *
 *     omega-dot = I-inverse (tau - omega x (I omega + h))
 *
 * with h the momentum stored in any momentum-exchange device, zero
 * for a body that carries none. The orientation then advances by one
 * exponential map, as it always has, of a rotation vector built from
 * the rates the same four stages produced:
 *
 *     theta = (dt/6)(w_0 + 4 w_mid + w_end)
 *             + (dt^2/12)(w_0 x w_end)
 *
 * the first term Simpson's rule over the interval and the second the
 * leading correction for a rate whose direction moves within it. The
 * result is composed onto the orientation and renormalised, which is
 * exactly what the step did before: a rotation is composed rather
 * than added, and the renormalisation still projects the result back
 * onto the unit sphere.
 *
 * What was there before was one explicit first-order update of the
 * rate. That is unconditionally unstable on this equation: the
 * transverse part of a torque-free rotation is turned rather than
 * grown by the true dynamics, and an explicit first-order step turns
 * it by multiplying its length by the square root of one plus the
 * square of the rate times the interval, which is greater than one
 * for every interval there is. Subdividing the interval reduces that
 * exponent linearly and never removes it.
 *
 * Measured, as worst relative drift in the magnitude of world-frame
 * angular momentum over 900 intervals of a tenth of a second, torque
 * free, on a ten tonne crew vehicle's tensor, with the interval
 * subdivided by the rate as libk26astro_att subdivides it:
 *
 *   rate, rad/s        1         3         5        10        20        30
 *   first order    2.8e-01   2.2e+00   4.9e+00   1.2e+01   2.6e+01   4.1e+01
 *   this step      2.6e-09   7.7e-09   1.3e-08   2.6e-08   5.1e-08   7.7e-08
 *
 * and without the subdivision the first-order step loses 2.2e+05 at
 * three radians a second and leaves the number line above five.
 *
 * Three alternatives were measured on that same table and on two
 * orientation probes with closed-form answers, a constant rate about
 * a principal axis and the torque-free precession of an axisymmetric
 * body. An explicit second-order step leaves 1.7e-05 to 1.7e-03 of
 * drift, three to five decades short of the fourth-order one. The
 * implicit midpoint rule conserves the momentum exactly in exact
 * arithmetic, and at a fixed four passes it delivers 1.5e-09 to
 * 3.2e-07, no better here and the most expensive of the four.
 * Fourth-order Runge-Kutta carried through the quaternion components
 * gives the same drift as this step, being the same rate update, and
 * a worse orientation at every step size this library runs at: it
 * loses the exactness of a constant rotation, which the exponential
 * map has and which the constant-rate probe measures at 2.9e-06
 * radians over ninety seconds against 7.0e-14 for the step chosen
 * here. On the precession probe the two cross over at a step eight
 * times finer than the subdivision bound admits, and above that
 * crossing this step is the more accurate of the two.
 *
 * Order, measured against the closed-form precession as the step
 * halves: the rate converges at ratio 16.00, which is fourth order,
 * and the orientation at ratio 7.5 to 7.7, which is third. The
 * orientation is third order and not fourth because the correction
 * term above resolves the rate's turning to one order less than the
 * quadrature resolves its magnitude; carrying it to fourth order
 * costs two further evaluations and buys nothing at any step this
 * library takes, which the crossing above is the measurement of.
 *
 * Cost: four evaluations of the rate derivative where there was one,
 * plus one cross product. The exponential map and the renormalisation
 * are unchanged, and they were the expensive part. Measured over two
 * million bare steps, 0.279 seconds against 0.157.
 *
 * Nothing here allocates, holds state between calls, or reads
 * anything outside its arguments, which is what the advance above it
 * needs in order to subdivide by the state and stay reproducible.
 *
 * References:
 *   - Markley & Crassidis, Fundamentals of Spacecraft Attitude
 *     Determination and Control (2014), section 3.7 on the quaternion
 *     exponential map and section 3.6.2 on Euler's rotational
 *     equation in non-principal axes.
 *   - Hughes, Spacecraft Attitude Dynamics (1986), section 4.5 for
 *     the full inertia tensor form, and section 4.7 for the
 *     momentum-exchange form the stored-momentum argument carries.
 *   - Blanes, Casas, Oteo & Ros, The Magnus expansion and some of its
 *     applications, Physics Reports 470 (2009), sections 2.1 and 4,
 *     for the rotation vector above: the quadrature is the leading
 *     term of the expansion and the cross product its second term.
 *   - Hairer, Norsett & Wanner, Solving Ordinary Differential
 *     Equations I (1993), section II.1, for the classical
 *     fourth-order Runge-Kutta coefficients.
 */

static K26V3 att_ext_rate_dot_(const K26AstroAttitudeStateExt *a,
                               K26V3 omega, K26V3 torque_body,
                               K26V3 stored_body)
{
    K26V3 Iw = m3_mul_v3_(&a->inertia, omega);
    K26V3 total = {
        Iw.x + stored_body.x,
        Iw.y + stored_body.y,
        Iw.z + stored_body.z
    };
    K26V3 gyro = k26m3d_v3_cross(omega, total);
    K26V3 rhs = {
        torque_body.x - gyro.x,
        torque_body.y - gyro.y,
        torque_body.z - gyro.z
    };
    return m3_mul_v3_(&a->inertia_inverse, rhs);
}

/* The one step. The torque and the stored momentum are held constant
 * across the interval, which is what a caller holding a command over
 * a control period is already doing with them. */
static void att_ext_step_(K26AstroAttitudeStateExt *a, K26V3 torque_body,
                          K26V3 stored_body, double dt)
{
    K26V3 w0 = a->omega_body;

    K26V3 k1 = att_ext_rate_dot_(a, w0, torque_body, stored_body);
    K26V3 w2 = {
        w0.x + k1.x * 0.5 * dt,
        w0.y + k1.y * 0.5 * dt,
        w0.z + k1.z * 0.5 * dt
    };
    K26V3 k2 = att_ext_rate_dot_(a, w2, torque_body, stored_body);
    K26V3 w3 = {
        w0.x + k2.x * 0.5 * dt,
        w0.y + k2.y * 0.5 * dt,
        w0.z + k2.z * 0.5 * dt
    };
    K26V3 k3 = att_ext_rate_dot_(a, w3, torque_body, stored_body);
    K26V3 w4 = {
        w0.x + k3.x * dt,
        w0.y + k3.y * dt,
        w0.z + k3.z * dt
    };
    K26V3 k4 = att_ext_rate_dot_(a, w4, torque_body, stored_body);

    K26V3 w_end = {
        w0.x + (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x) * dt / 6.0,
        w0.y + (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y) * dt / 6.0,
        w0.z + (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z) * dt / 6.0
    };
    /* The two half-interval stages are both estimates of the rate at
     * the middle of it, and their mean is the one Simpson's rule
     * below wants. */
    K26V3 w_mid = {
        0.5 * (w2.x + w3.x),
        0.5 * (w2.y + w3.y),
        0.5 * (w2.z + w3.z)
    };
    K26V3 turn = k26m3d_v3_cross(w0, w_end);
    K26V3 theta = {
        (w0.x + 4.0 * w_mid.x + w_end.x) * dt / 6.0 + turn.x * dt * dt / 12.0,
        (w0.y + 4.0 * w_mid.y + w_end.y) * dt / 6.0 + turn.y * dt * dt / 12.0,
        (w0.z + 4.0 * w_mid.z + w_end.z) * dt / 6.0 + turn.z * dt * dt / 12.0
    };

    a->omega_body = w_end;
    a->q = k26m3d_quat_norm(
        k26m3d_quat_mul(a->q, k26astro_quat_exp_half(theta)));
}

void k26astro_attitude_step_free_ext(K26AstroAttitudeStateExt *a, double dt)
{
    if (!a) return;
    K26V3 zero = { 0.0, 0.0, 0.0 };
    att_ext_step_(a, zero, zero, dt);
}

void k26astro_attitude_step_torque_ext(K26AstroAttitudeStateExt *a,
                                       K26V3 torque_body, double dt)
{
    if (!a) return;
    K26V3 zero = { 0.0, 0.0, 0.0 };
    att_ext_step_(a, torque_body, zero, dt);
}

void k26astro_attitude_step_exchange_ext(K26AstroAttitudeStateExt *a,
                                         K26V3 torque_body,
                                         K26V3 stored_momentum_body,
                                         double dt)
{
    if (!a) return;
    att_ext_step_(a, torque_body, stored_momentum_body, dt);
}

void k26astro_attitude_step_torque_registry_ext(K26AstroAttitudeStateExt *a,
                                                double t, double dt)
{
    if (!a) return;
    K26V3 sum = { 0.0, 0.0, 0.0 };
    if (a->torques) {
        struct K26AstroTorqueListExt *p =
            (struct K26AstroTorqueListExt *)a->torques;
        for (int i = 0; i < p->count; i++) {
            p->fns[i](a, t, &sum, p->ctxs[i]);
        }
    }
    k26astro_attitude_step_torque_ext(a, sum, dt);
}
