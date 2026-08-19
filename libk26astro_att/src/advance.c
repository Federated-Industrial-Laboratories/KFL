/* advance.c - attitude on the exact stepping path.
 *
 * The equations are libk26astro_body's; this file sequences them and
 * writes the result back to the body, so that a consumer reading a
 * body sees an orientation that has actually been integrated.
 *
 * Two properties are kept deliberately:
 *
 * The advance is driven, not scheduled. It happens when the caller
 * says, over the interval the caller names, so a subdivision of the
 * control period decides the accuracy and no rate setting anywhere
 * else can change the physics.
 *
 * A non-finite result is reported rather than published. The body
 * keeps the last finite orientation and rate, and the caller ends the
 * episode. Writing a value that is not a number into a body would put
 * it into an observation, a record, and a trained policy.
 */
#include "k26astro_att/att.h"

#include "k26astro_body/body.h"

#include <math.h>

#include "att_internal.h"

static int att_finite_v3_(K26V3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

const char *k26astro_att_status_str(K26AstroAttStatus s)
{
    switch (s) {
    case K26ASTRO_ATT_OK:         return "ok";
    case K26ASTRO_ATT_E_NULL:     return "null vehicle or missing attitude state";
    case K26ASTRO_ATT_E_BAD_DT:   return "interval is negative or not finite";
    case K26ASTRO_ATT_E_SINGULAR: return "inertia tensor has no inverse";
    case K26ASTRO_ATT_E_DIVERGED: return "the step produced a non-finite state";
    }
    return "unknown status";
}

static int att_finite_quat_(K26Quat q)
{
    return isfinite(q.w) && isfinite(q.x) && isfinite(q.y) && isfinite(q.z);
}


/* Load the bound body's orientation and rate into the vehicle's
 * attitude state. The body is the state; this state is the
 * integrator's working copy and the inertia it works with. Every
 * entry that reads or advances attitude calls this first, so none of
 * them can report a stale copy of something a declaration, a reset
 * draw or a step-time write has since changed. */
static void att_load_from_body_(K26AstroVehicle *v,
                                K26AstroAttitudeStateExt *a)
{
    K26AstroBody *b = k26astro_vehicle_body(v);
    if (!b) return;
    a->q          = b->attitude;
    a->omega_body = b->omega;
}

K26AstroAttStatus k26astro_att_step(K26AstroVehicle *v, K26V3 torque,
                                    double dt)
{
    if (!v) return K26ASTRO_ATT_E_NULL;
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    if (!a) return K26ASTRO_ATT_E_NULL;
    if (!isfinite(dt) || dt < 0.0) return K26ASTRO_ATT_E_BAD_DT;
    /* The load happens before the interval is examined, so a step of
     * zero duration resynchronises the working copy with the body
     * rather than leaving a caller holding a stale one. */
    att_load_from_body_(v, a);
    if (dt == 0.0) return K26ASTRO_ATT_OK;
    if (att_inverse_is_zero_(&a->inertia_inverse)) {
        return K26ASTRO_ATT_E_SINGULAR;
    }
    if (!att_finite_v3_(torque)) return K26ASTRO_ATT_E_DIVERGED;

    K26Quat q0 = a->q;
    K26V3   w0 = a->omega_body;

    /* A quaternion is written component by component, so it arrives
     * here in whatever state the writer left it. Normalising at the
     * point of use is what makes that safe; a zero-norm quaternion
     * cannot be normalised and is reported as divergence below rather
     * than propagated. */
    double n2 = a->q.w * a->q.w + a->q.x * a->q.x
              + a->q.y * a->q.y + a->q.z * a->q.z;
    if (!isfinite(n2) || n2 == 0.0) return K26ASTRO_ATT_E_DIVERGED;
    a->q = k26m3d_quat_norm(a->q);

    /* The interval is subdivided by the rate it is entered with, so
     * a body turning fast pays for the resolution it needs and a body
     * in ordinary flight pays nothing: at a count of one this is the
     * single step it always was, operation for operation. The last
     * sub-interval takes the remainder rather than the quotient, so
     * the durations sum to dt exactly however the division rounded,
     * which is the same rule the caller applies to its own
     * subdivision of a control period. */
    int m = k26astro_att_substep_count(w0, dt);
    double advanced = 0.0;
    for (int k = 0; k < m; k++) {
        double h = (k + 1 == m) ? (dt - advanced) : (dt / (double)m);
        k26astro_attitude_step_torque_ext(a, torque, h);
        advanced += h;
    }

    if (!att_finite_quat_(a->q) || !att_finite_v3_(a->omega_body)) {
        /* Leave the state as it was, so the last honest orientation
         * is what anything reading it gets. */
        a->q          = q0;
        a->omega_body = w0;
        return K26ASTRO_ATT_E_DIVERGED;
    }

    K26AstroBody *b = k26astro_vehicle_body(v);
    if (b) {
        b->attitude = a->q;
        b->omega    = a->omega_body;
    }
    return K26ASTRO_ATT_OK;
}

K26AstroAttStatus k26astro_att_gravity_gradient(const K26AstroVehicle *v,
                                                K26V3 r_world, double mu,
                                                K26V3 *out)
{
    if (!v || !out) return K26ASTRO_ATT_E_NULL;
    out->x = out->y = out->z = 0.0;
    K26AstroAttitudeStateExt *a =
        k26astro_vehicle_attitude_ext((K26AstroVehicle *)v);
    if (!a) return K26ASTRO_ATT_E_NULL;
    att_load_from_body_((K26AstroVehicle *)v, a);

    double r2 = r_world.x * r_world.x + r_world.y * r_world.y
              + r_world.z * r_world.z;
    if (!(r2 > 0.0) || !(mu > 0.0) || !isfinite(r2) || !isfinite(mu)) {
        /* No separation or no attractor is no torque, not an error:
         * a body at the origin of its own attraction is a
         * configuration, not a failure. */
        return K26ASTRO_ATT_OK;
    }
    /* The expression is three mu over r cubed, times r-hat crossed
     * with I r-hat. Written with the unnormalised separation it is
     * three mu over r to the fifth, times r crossed with I r, which
     * spends one square root instead of three divisions and keeps the
     * two forms algebraically identical. */
    K26V3 r_body = k26m3d_quat_rotate_v3(k26m3d_quat_conj(a->q), r_world);
    K26V3 Ir;
    Ir.x = a->inertia.m[0][0] * r_body.x + a->inertia.m[0][1] * r_body.y
         + a->inertia.m[0][2] * r_body.z;
    Ir.y = a->inertia.m[1][0] * r_body.x + a->inertia.m[1][1] * r_body.y
         + a->inertia.m[1][2] * r_body.z;
    Ir.z = a->inertia.m[2][0] * r_body.x + a->inertia.m[2][1] * r_body.y
         + a->inertia.m[2][2] * r_body.z;
    K26V3 cross = k26m3d_v3_cross(r_body, Ir);
    double r  = sqrt(r2);
    double k  = 3.0 * mu / (r2 * r2 * r);
    out->x = k * cross.x;
    out->y = k * cross.y;
    out->z = k * cross.z;
    if (!att_finite_v3_(*out)) {
        out->x = out->y = out->z = 0.0;
        return K26ASTRO_ATT_E_DIVERGED;
    }
    return K26ASTRO_ATT_OK;
}

K26AstroAttStatus k26astro_att_step_all(K26AstroVehicle *const *v, int n,
                                        const K26V3 *torques, double dt)
{
    if (!v && n > 0) return K26ASTRO_ATT_E_NULL;
    K26AstroAttStatus first = K26ASTRO_ATT_OK;
    K26V3 zero = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < n; i++) {
        if (!v[i]) continue;
        K26AstroAttStatus s =
            k26astro_att_step(v[i], torques ? torques[i] : zero, dt);
        if (s != K26ASTRO_ATT_OK && first == K26ASTRO_ATT_OK) first = s;
    }
    return first;
}

K26AstroAttStatus k26astro_att_momentum_world(const K26AstroVehicle *v,
                                              K26V3 *out)
{
    if (!v || !out) return K26ASTRO_ATT_E_NULL;
    K26AstroAttitudeStateExt *a =
        k26astro_vehicle_attitude_ext((K26AstroVehicle *)v);
    if (!a) return K26ASTRO_ATT_E_NULL;
    att_load_from_body_((K26AstroVehicle *)v, a);
    K26V3 h_body;
    h_body.x = a->inertia.m[0][0] * a->omega_body.x
             + a->inertia.m[0][1] * a->omega_body.y
             + a->inertia.m[0][2] * a->omega_body.z;
    h_body.y = a->inertia.m[1][0] * a->omega_body.x
             + a->inertia.m[1][1] * a->omega_body.y
             + a->inertia.m[1][2] * a->omega_body.z;
    h_body.z = a->inertia.m[2][0] * a->omega_body.x
             + a->inertia.m[2][1] * a->omega_body.y
             + a->inertia.m[2][2] * a->omega_body.z;
    *out = k26m3d_quat_rotate_v3(a->q, h_body);
    return K26ASTRO_ATT_OK;
}
