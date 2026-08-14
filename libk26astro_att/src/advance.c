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

static int att_finite_v3_(K26V3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

/* The inertia inverse is zeroed by the body library when the tensor it
 * was given is singular, which is exactly the state in which a torque
 * step would silently do nothing. Detecting it here turns that into a
 * status. */
static int att_inverse_is_zero_(const K26M3 *inv)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (inv->m[i][j] != 0.0) return 0;
        }
    }
    return 1;
}

K26AstroAttStatus k26astro_att_step(K26AstroVehicle *v, K26V3 torque,
                                    double dt)
{
    if (!v) return K26ASTRO_ATT_E_NULL;
    K26AstroAttitudeStateExt *a = k26astro_vehicle_attitude_ext(v);
    if (!a) return K26ASTRO_ATT_E_NULL;
    if (!isfinite(dt) || dt < 0.0) return K26ASTRO_ATT_E_BAD_DT;
    if (dt == 0.0) return K26ASTRO_ATT_OK;
    if (att_inverse_is_zero_(&a->inertia_inverse)) {
        return K26ASTRO_ATT_E_SINGULAR;
    }
    if (!att_finite_v3_(torque)) return K26ASTRO_ATT_E_DIVERGED;

    K26Quat q0 = a->q;
    K26V3   w0 = a->omega_body;

    k26astro_attitude_step_torque_ext(a, torque, dt);

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

K26AstroAttStatus k26astro_att_step_all(K26AstroVehicle *const *v, int n,
                                        double dt)
{
    if (!v && n > 0) return K26ASTRO_ATT_E_NULL;
    K26AstroAttStatus first = K26ASTRO_ATT_OK;
    K26V3 zero = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < n; i++) {
        if (!v[i]) continue;
        K26AstroAttStatus s = k26astro_att_step(v[i], zero, dt);
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
