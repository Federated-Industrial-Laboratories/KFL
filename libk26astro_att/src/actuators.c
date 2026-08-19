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

#include "att_internal.h"

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
 * The equation is the body library's and so is the step that advances
 * it: this entry evaluates the wheels, the magnetorquers and the
 * thrusters over the sub-interval, sums what they apply, and hands
 * the torque and the stored momentum to the same function the
 * unactuated advance calls with a stored momentum of zero. The two
 * paths therefore have one integrator between them, at one order, and
 * an accuracy claim proved of one is a claim about both.
 *
 * What the wheels do inside a sub-interval is unchanged and stays
 * first order deliberately. Their momentum is clamped at saturation
 * and their Coulomb friction has a dead rate, so the wheel state is
 * not a smooth function of itself and a higher-order rule through
 * either discontinuity would resolve a curve that is not there. What
 * the body feels from them, the realised reaction torque and the
 * stored momentum, is held across the sub-interval exactly as the
 * commands are, which is what a zero-order hold over a control period
 * already means.
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
    /* The same guard the unactuated entry carries, on the same terms.
     * This is the entry an emitted artifact takes, so a guard present
     * only on the other one protects a path the product does not
     * use: a singular tensor would step here and silently do nothing
     * rather than be reported. */
    if (att_inverse_is_zero_(&a->inertia_inverse)) {
        return K26ASTRO_ATT_E_SINGULAR;
    }
    if (!act_finite3_(extra)) return K26ASTRO_ATT_E_DIVERGED;

    double n2 = a->q.w * a->q.w + a->q.x * a->q.x
              + a->q.y * a->q.y + a->q.z * a->q.z;
    if (!isfinite(n2) || n2 == 0.0) return K26ASTRO_ATT_E_DIVERGED;
    a->q = k26m3d_quat_norm(a->q);

    K26Quat q0 = a->q;
    K26V3   w0 = a->omega_body;

    /* The interval is subdivided by the rate the advance is entered
     * with, so a craft a policy has set tumbling resolves its own
     * rotation and a craft in ordinary flight pays nothing for the
     * provision: at a count of one what follows is the single step it
     * always was, operation for operation. The commands are held
     * across the sub-intervals, which is what they are held across
     * the control period for, so the zero-order hold is unchanged.
     * The last sub-interval takes the remainder rather than the
     * quotient, so the durations sum to dt exactly however the
     * division rounded. */
    int m = k26astro_att_substep_count(w0, dt);
    double advanced = 0.0;
    K26AstroAttStatus s = K26ASTRO_ATT_OK;

    for (int k = 0; k < m; k++) {
        double sub_dt = (k + 1 == m) ? (dt - advanced) : (dt / (double)m);

        K26V3 wheel_torque, stored, mag_torque, thr_torque;
        s = k26astro_att_wheels_step(act, sub_dt, &wheel_torque, &stored);
        if (s != K26ASTRO_ATT_OK) break;
        s = k26astro_att_torquers_torque(act, b_body, &mag_torque);
        if (s != K26ASTRO_ATT_OK) break;
        s = k26astro_att_thrusters_wrench(act, NULL, &thr_torque);
        if (s != K26ASTRO_ATT_OK) break;

        /* The applied torque: everything external, plus the actuators
         * that act on the body directly, plus the wheels' reaction,
         * which is the negated rate of change of their stored
         * momentum. */
        K26V3 tau = {
            extra.x + mag_torque.x + thr_torque.x + wheel_torque.x,
            extra.y + mag_torque.y + thr_torque.y + wheel_torque.y,
            extra.z + mag_torque.z + thr_torque.z + wheel_torque.z
        };

        /* The body library's step, with the wheels' stored momentum
         * passed through so the gyroscopic term carries it alongside
         * the body's own, which is the other half of what a wheel
         * does. This entry used to write out that equation and
         * advance it here. It does not any more: a second copy of one
         * equation of motion is how two callers come to disagree
         * about the same craft, and raising the order of one copy
         * while leaving the other is exactly that failure. The
         * unactuated advance reaches the same function with a zero
         * stored momentum. */
        k26astro_attitude_step_exchange_ext(a, tau, stored, sub_dt);

        advanced += sub_dt;
    }

    /* A sub-interval that failed leaves the state as the advance was
     * entered, on the same terms a single step does: the transition
     * did not happen, so nothing it would have moved has moved. The
     * wheel momenta a completed sub-interval already changed are the
     * caller's working copy, which a caller discards on a failed
     * advance for exactly this reason. */
    if (s != K26ASTRO_ATT_OK) {
        a->q          = q0;
        a->omega_body = w0;
        return s;
    }
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

/* ---- Geodetic conversion ------------------------------------------ *
 *
 * Bowring's method: an auxiliary parametric latitude gives a first
 * estimate that is already accurate to well under a millimetre for
 * heights of interest, and the loop below refines it until it stops
 * moving. The alternative, a closed-form quartic solution, is exact
 * but longer and needs care near the poles; this converges in two or
 * three passes everywhere and its terminating condition is that the
 * latitude has stopped changing at all.
 */
#define ATT_WGS84_A  6378137.0
#define ATT_WGS84_F  (1.0 / 298.257223563)

K26AstroAttStatus k26astro_att_geodetic(K26V3 ecef, double *lat,
                                        double *lon, double *alt)
{
    if (!lat || !lon || !alt) return K26ASTRO_ATT_E_NULL;
    const double a  = ATT_WGS84_A;
    const double f  = ATT_WGS84_F;
    const double b  = a * (1.0 - f);
    const double e2 = f * (2.0 - f);
    const double ep2 = e2 / (1.0 - e2);

    double p = sqrt(ecef.x * ecef.x + ecef.y * ecef.y);
    *lon = atan2(ecef.y, ecef.x);
    if (p == 0.0) {
        /* On the axis: the latitude is a pole and the height is the
         * distance from the polar radius. */
        *lat = (ecef.z >= 0.0) ? (M_PI / 2.0) : -(M_PI / 2.0);
        *alt = fabs(ecef.z) - b;
        return K26ASTRO_ATT_OK;
    }
    double theta = atan2(ecef.z * a, p * b);
    double st = sin(theta), ct = cos(theta);
    double phi = atan2(ecef.z + ep2 * b * st * st * st,
                       p - e2 * a * ct * ct * ct);
    for (int i = 0; i < 8; i++) {
        double sp = sin(phi);
        double N  = a / sqrt(1.0 - e2 * sp * sp);
        double h  = p / cos(phi) - N;
        double next = atan2(ecef.z, p * (1.0 - e2 * N / (N + h)));
        if (next == phi) break;
        phi = next;
    }
    double sp = sin(phi);
    double N  = a / sqrt(1.0 - e2 * sp * sp);
    *lat = phi;
    *alt = p / cos(phi) - N;
    return K26ASTRO_ATT_OK;
}

K26V3 k26astro_att_enu_to_ecef(K26V3 enu, double lat, double lon)
{
    double sl = sin(lat), cl = cos(lat);
    double so = sin(lon), co = cos(lon);
    K26V3 out;
    out.x = -so * enu.x - sl * co * enu.y + cl * co * enu.z;
    out.y =  co * enu.x - sl * so * enu.y + cl * so * enu.z;
    out.z =              cl      * enu.y + sl      * enu.z;
    return out;
}
