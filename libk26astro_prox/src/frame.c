/* frame.c: the chief's local-vertical local-horizontal basis, and the
 * target's state resolved onto it.
 *
 * Both are algebra on states the integrator already produced, so both
 * are exact for any orbit and any perturbation. Nothing here
 * linearises anything; the linearised model lives in cw.c and is a
 * tool the gates and the initial-condition construction use.
 *
 * The frame's angular velocity, derived rather than asserted. The
 * first axis is e1 = r / |r|. Differentiating,
 *
 *     d(e1)/dt = v/|r| - r (dr/dt)/|r|^2 = (v - e1 (e1 . v)) / |r|,
 *
 * which is the transverse part of the velocity over the radius. If the
 * basis turns with angular velocity w then d(e1)/dt = w x e1, and a
 * component of w along e1 moves e1 not at all while a component along
 * e2 would tilt e1 out of the plane. Since e3 is by construction
 * perpendicular to v, the transverse velocity has no e3 part, so the
 * tilt rate is zero and w = (|v_t|/|r|) e3 = (r x v) / |r|^2. The
 * remaining freedom, rotation about e1, is fixed by how e3 moves, and
 * is zero exactly when the angular momentum direction holds still.
 * Taking it as zero is the standard convention; the header states what
 * that neglects.
 */
#include <math.h>

#include "k26astro_prox/prox.h"

const char *k26astro_prox_status_str(K26AstroProxStatus s)
{
    switch (s) {
    case K26ASTRO_PROX_OK:           return "ok";
    case K26ASTRO_PROX_E_NULL:       return "null argument";
    case K26ASTRO_PROX_E_DEGENERATE: return "no frame at this state";
    case K26ASTRO_PROX_E_RANGE:      return "argument out of range";
    }
    return "unknown status";
}

/* Finite in the sense every check here needs: a number that is
 * neither a NaN nor an infinity. Written out rather than calling
 * isfinite so the file's arithmetic claim covers every line of it. */
static int prox_finite_(double x)
{
    return (x == x) && (x - x == 0.0);
}

static int prox_finite_v3_(K26V3 v)
{
    return prox_finite_(v.x) && prox_finite_(v.y) && prox_finite_(v.z);
}

K26AstroProxStatus k26astro_prox_frame(const K26AstroPos *central_pos,
                                       K26V3 central_vel,
                                       const K26AstroPos *chief_pos,
                                       K26V3 chief_vel,
                                       K26AstroProxFrame *out)
{
    if (!central_pos || !chief_pos || !out) return K26ASTRO_PROX_E_NULL;
    if (!prox_finite_v3_(central_vel) || !prox_finite_v3_(chief_vel)) {
        return K26ASTRO_PROX_E_DEGENERATE;
    }

    /* The exact sector-aware difference. The pair may sit anywhere in
     * the system and the separation is the small quantity, which is
     * the case a flattened coordinate difference destroys. */
    K26V3 r = k26astro_pos_sub(chief_pos, central_pos);
    K26V3 v = k26m3d_v3_sub(chief_vel, central_vel);
    if (!prox_finite_v3_(r)) return K26ASTRO_PROX_E_DEGENERATE;

    double r2 = k26m3d_v3_dot(r, r);
    if (!(r2 > 0.0) || !prox_finite_(r2)) return K26ASTRO_PROX_E_DEGENERATE;
    double rmag = sqrt(r2);

    K26V3 h = k26m3d_v3_cross(r, v);
    double h2 = k26m3d_v3_dot(h, h);
    if (!(h2 > 0.0) || !prox_finite_(h2)) return K26ASTRO_PROX_E_DEGENERATE;

    K26V3 e1 = k26m3d_v3_scale(r, 1.0 / rmag);
    K26V3 e3 = k26m3d_v3_scale(h, 1.0 / sqrt(h2));
    K26V3 e2 = k26m3d_v3_cross(e3, e1);

    out->e1     = e1;
    out->e2     = e2;
    out->e3     = e3;
    out->omega  = k26m3d_v3_scale(h, 1.0 / r2);
    out->radius = rmag;
    return K26ASTRO_PROX_OK;
}

K26AstroProxStatus k26astro_prox_relative(const K26AstroProxFrame *f,
                                          const K26AstroPos *chief_pos,
                                          K26V3 chief_vel,
                                          const K26AstroPos *target_pos,
                                          K26V3 target_vel,
                                          K26AstroProxRel *out)
{
    if (!f || !chief_pos || !target_pos || !out) {
        return K26ASTRO_PROX_E_NULL;
    }
    if (!prox_finite_v3_(chief_vel) || !prox_finite_v3_(target_vel)) {
        return K26ASTRO_PROX_E_DEGENERATE;
    }

    K26V3 rho = k26astro_pos_sub(target_pos, chief_pos);
    if (!prox_finite_v3_(rho)) return K26ASTRO_PROX_E_DEGENERATE;

    /* The rate an observer riding the frame sees. Subtracting the
     * frame's own rotation is what makes a station-keeping pair read
     * zero rather than reading the difference of two orbital
     * velocities, and it is the quantity the linearised propagator
     * transports. */
    K26V3 dv  = k26m3d_v3_sub(target_vel, chief_vel);
    K26V3 rot = k26m3d_v3_cross(f->omega, rho);
    K26V3 vel = k26m3d_v3_sub(dv, rot);

    out->r = k26m3d_v3(k26m3d_v3_dot(rho, f->e1),
                       k26m3d_v3_dot(rho, f->e2),
                       k26m3d_v3_dot(rho, f->e3));
    out->v = k26m3d_v3(k26m3d_v3_dot(vel, f->e1),
                       k26m3d_v3_dot(vel, f->e2),
                       k26m3d_v3_dot(vel, f->e3));
    return K26ASTRO_PROX_OK;
}
