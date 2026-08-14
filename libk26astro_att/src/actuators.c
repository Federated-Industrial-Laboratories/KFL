/* actuators.c - reaction wheels, magnetorquers and thrusters.
 *
 * The three kinds share one shape: a description the assembly fixed,
 * a command a control loop writes once per control period, and, for
 * the wheels alone, a state that persists between intervals. Nothing
 * here allocates, and every array is the caller's, because this runs
 * inside the stepping path.
 *
 * The wheel model is the one the design fixes: the commanded torque
 * is clamped to the wheel's limit, friction opposes motion as a
 * viscous term plus a constant Coulomb term with a dead rate, and
 * saturation clamps the state rather than the command, so a saturated
 * wheel delivers no further torque in the direction that would
 * increase its momentum while friction can still slow it. The
 * alternative, clamping the command, would let a saturated wheel go
 * on accepting torque it cannot store.
 */
#include "k26astro_att/att.h"

#include "k26astro_body/body.h"

#include <math.h>

static int act_finite3_(K26V3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

static double act_clamp_(double v, double lo, double hi)
{
    if (!isfinite(v)) return 0.0;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

K26AstroAttStatus k26astro_att_wheels_step(K26AstroAttActuators *act,
                                           double dt, K26V3 *torque,
                                           K26V3 *stored)
{
    K26V3 t = { 0.0, 0.0, 0.0 };
    K26V3 h = { 0.0, 0.0, 0.0 };
    if (torque) *torque = t;
    if (stored) *stored = h;
    if (!act) return K26ASTRO_ATT_E_NULL;
    if (!isfinite(dt) || dt < 0.0) return K26ASTRO_ATT_E_BAD_DT;
    if (act->n_wheels > 0 && !act->wheels) return K26ASTRO_ATT_E_NULL;

    for (int i = 0; i < act->n_wheels; i++) {
        K26AstroAttWheel *w = &act->wheels[i];
        double u = act_clamp_(w->command, -w->max_torque, w->max_torque);

        /* Friction opposes the wheel's own motion. Below the dead
         * rate the Coulomb term is exactly zero rather than small,
         * which is what stops a wheel at rest from chattering across
         * the sign change every interval. */
        double rate = (w->spin_inertia > 0.0)
                    ? w->momentum / w->spin_inertia : 0.0;
        double fric = w->viscous * rate;
        if (fabs(rate) > w->dead_rate) {
            fric += (rate > 0.0 ? w->coulomb : -w->coulomb);
        }
        double hdot = u - fric;

        /* Saturation clamps the state, and the clamp below is the
         * whole of the mechanism. An earlier version also zeroed the
         * rate of change when the wheel was already at its limit and
         * the command pushed further; that branch was removed because
         * no case distinguishes it from the clamp, which was
         * established by mutation rather than by reading: deleting it
         * changed no gate. What matters is the property, and the
         * clamp has it. A wheel at its limit takes no further
         * momentum in the direction it is saturated in, so the
         * realised change is zero and the body feels no torque;
         * friction acts the other way, is not clamped, and can still
         * slow the wheel off the limit. */
        double next = w->momentum + hdot * dt;
        if (w->max_momentum > 0.0) {
            if (next >  w->max_momentum) next =  w->max_momentum;
            if (next < -w->max_momentum) next = -w->max_momentum;
        }
        /* The realised rate of change is what the body reacts to, so
         * a step clipped by the limit above reacts by the clipped
         * amount and not by what was asked for. */
        double realised = (dt > 0.0) ? (next - w->momentum) / dt : 0.0;
        w->momentum = next;

        t.x -= realised * w->axis.x;
        t.y -= realised * w->axis.y;
        t.z -= realised * w->axis.z;
        h.x += next * w->axis.x;
        h.y += next * w->axis.y;
        h.z += next * w->axis.z;
    }
    if (!act_finite3_(t) || !act_finite3_(h)) {
        return K26ASTRO_ATT_E_DIVERGED;
    }
    if (torque) *torque = t;
    if (stored) *stored = h;
    return K26ASTRO_ATT_OK;
}

K26AstroAttStatus k26astro_att_torquers_torque(
    const K26AstroAttActuators *act, K26V3 b_body, K26V3 *out)
{
    if (!out) return K26ASTRO_ATT_E_NULL;
    out->x = out->y = out->z = 0.0;
    if (!act) return K26ASTRO_ATT_E_NULL;
    if (act->n_torquers > 0 && !act->torquers) return K26ASTRO_ATT_E_NULL;
    if (!act_finite3_(b_body)) return K26ASTRO_ATT_E_DIVERGED;

    K26V3 m = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < act->n_torquers; i++) {
        const K26AstroAttTorquer *q = &act->torquers[i];
        double d = act_clamp_(q->command, -q->max_dipole, q->max_dipole);
        m.x += d * q->axis.x;
        m.y += d * q->axis.y;
        m.z += d * q->axis.z;
    }
    *out = k26m3d_v3_cross(m, b_body);
    if (!act_finite3_(*out)) {
        out->x = out->y = out->z = 0.0;
        return K26ASTRO_ATT_E_DIVERGED;
    }
    return K26ASTRO_ATT_OK;
}

K26AstroAttStatus k26astro_att_thrusters_wrench(
    const K26AstroAttActuators *act, K26V3 *force, K26V3 *out)
{
    K26V3 f = { 0.0, 0.0, 0.0 };
    K26V3 t = { 0.0, 0.0, 0.0 };
    if (force) *force = f;
    if (out) *out = t;
    if (!act) return K26ASTRO_ATT_E_NULL;
    if (act->n_thrusters > 0 && !act->thrusters) return K26ASTRO_ATT_E_NULL;

    for (int i = 0; i < act->n_thrusters; i++) {
        const K26AstroAttThruster *th = &act->thrusters[i];
        double u = act_clamp_(th->command, 0.0, 1.0);
        if (u == 0.0) continue;
        K26V3 fi = {
            u * th->max_thrust * th->dir.x,
            u * th->max_thrust * th->dir.y,
            u * th->max_thrust * th->dir.z
        };
        /* The torque is about the centre of mass, so a thruster
         * through it produces none however hard it fires. */
        K26V3 r = {
            th->at.x - act->com.x,
            th->at.y - act->com.y,
            th->at.z - act->com.z
        };
        K26V3 ti = k26m3d_v3_cross(r, fi);
        f.x += fi.x; f.y += fi.y; f.z += fi.z;
        t.x += ti.x; t.y += ti.y; t.z += ti.z;
    }
    if (!act_finite3_(f) || !act_finite3_(t)) {
        return K26ASTRO_ATT_E_DIVERGED;
    }
    if (force) *force = f;
    if (out) *out = t;
    return K26ASTRO_ATT_OK;
}

/* The momentum-exchange form of Euler's rotational equation:
 *
 *   I omega-dot = tau - omega x (I omega + h) - h-dot
 *
 * with h the stored wheel momentum in the body frame. The two terms
 * the wheels add are the whole difference between a spacecraft with
 * reaction wheels and one without: without them a wheel would change
 * the body's momentum out of nothing, and the conservation the gates
 * check would fail.
 *
 * The orientation then advances by the same exponential map the
 * unactuated step uses, so the two paths differ in the angular
 * velocity update alone.
 */
K26AstroAttStatus k26astro_att_step_actuated(K26AstroVehicle *v,
                                             K26AstroAttActuators *act,
                                             K26V3 extra, K26V3 b_body,
                                             double dt)
{
    if (!v) return K26ASTRO_ATT_E_NULL;
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    if (!a) return K26ASTRO_ATT_E_NULL;
    if (!isfinite(dt) || dt < 0.0) return K26ASTRO_ATT_E_BAD_DT;
    if (!act) return k26astro_att_step(v, extra, dt);

    /* The bound body is the state, exactly as for the unactuated
     * step; see that function for why the load happens here. */
    K26AstroBody *b = k26astro_vehicle_body(v);
    if (b) {
        a->q          = b->attitude;
        a->omega_body = b->omega;
    }
    if (dt == 0.0) return K26ASTRO_ATT_OK;
    if (!act_finite3_(extra)) return K26ASTRO_ATT_E_DIVERGED;

    double n2 = a->q.w * a->q.w + a->q.x * a->q.x
              + a->q.y * a->q.y + a->q.z * a->q.z;
    if (!isfinite(n2) || n2 == 0.0) return K26ASTRO_ATT_E_DIVERGED;
    a->q = k26m3d_quat_norm(a->q);

    K26Quat q0 = a->q;
    K26V3   w0 = a->omega_body;

    K26V3 wheel_torque, stored, mag_torque, thr_torque;
    K26AstroAttStatus s = k26astro_att_wheels_step(act, dt, &wheel_torque,
                                                   &stored);
    if (s != K26ASTRO_ATT_OK) return s;
    s = k26astro_att_torquers_torque(act, b_body, &mag_torque);
    if (s != K26ASTRO_ATT_OK) return s;
    s = k26astro_att_thrusters_wrench(act, NULL, &thr_torque);
    if (s != K26ASTRO_ATT_OK) return s;

    /* The applied torque: everything external, plus the actuators
     * that act on the body directly, plus the wheels' reaction, which
     * is the negated rate of change of their stored momentum. */
    K26V3 tau = {
        extra.x + mag_torque.x + thr_torque.x + wheel_torque.x,
        extra.y + mag_torque.y + thr_torque.y + wheel_torque.y,
        extra.z + mag_torque.z + thr_torque.z + wheel_torque.z
    };

    /* The gyroscopic term carries the stored momentum with the
     * body's own, which is the other half of what a wheel does. */
    K26V3 Iw;
    Iw.x = a->inertia.m[0][0] * w0.x + a->inertia.m[0][1] * w0.y
         + a->inertia.m[0][2] * w0.z;
    Iw.y = a->inertia.m[1][0] * w0.x + a->inertia.m[1][1] * w0.y
         + a->inertia.m[1][2] * w0.z;
    Iw.z = a->inertia.m[2][0] * w0.x + a->inertia.m[2][1] * w0.y
         + a->inertia.m[2][2] * w0.z;
    K26V3 total = { Iw.x + stored.x, Iw.y + stored.y, Iw.z + stored.z };
    K26V3 gyro  = k26m3d_v3_cross(w0, total);
    K26V3 rhs   = { tau.x - gyro.x, tau.y - gyro.y, tau.z - gyro.z };

    K26V3 wdot;
    wdot.x = a->inertia_inverse.m[0][0] * rhs.x
           + a->inertia_inverse.m[0][1] * rhs.y
           + a->inertia_inverse.m[0][2] * rhs.z;
    wdot.y = a->inertia_inverse.m[1][0] * rhs.x
           + a->inertia_inverse.m[1][1] * rhs.y
           + a->inertia_inverse.m[1][2] * rhs.z;
    wdot.z = a->inertia_inverse.m[2][0] * rhs.x
           + a->inertia_inverse.m[2][1] * rhs.y
           + a->inertia_inverse.m[2][2] * rhs.z;

    a->omega_body.x = w0.x + wdot.x * dt;
    a->omega_body.y = w0.y + wdot.y * dt;
    a->omega_body.z = w0.z + wdot.z * dt;

    K26V3 theta = {
        a->omega_body.x * dt, a->omega_body.y * dt, a->omega_body.z * dt
    };
    a->q = k26m3d_quat_norm(
        k26m3d_quat_mul(a->q, k26astro_quat_exp_half(theta)));

    if (!isfinite(a->q.w) || !isfinite(a->q.x) || !isfinite(a->q.y) ||
        !isfinite(a->q.z) || !act_finite3_(a->omega_body)) {
        a->q          = q0;
        a->omega_body = w0;
        return K26ASTRO_ATT_E_DIVERGED;
    }
    if (b) {
        b->attitude = a->q;
        b->omega    = a->omega_body;
    }
    return K26ASTRO_ATT_OK;
}
