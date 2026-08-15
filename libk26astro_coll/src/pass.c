/* pass.c: the broadphase, the selection rule, and the two
 * resolutions.
 *
 * The pass runs between the sub-advances of an exact step, which is
 * the whole point of it: a body that crosses a target between one
 * control period and the next is not detected by looking at the two
 * endpoints, and shortening the step to catch it would make the
 * applied duration a function of the geometry. Sweeping inside the
 * sub-advance detects the crossing and leaves the applied duration
 * exactly what was declared.
 *
 * Nothing here allocates, and the loops are over fixed counts in
 * declaration order.
 */

#include "k26astro_coll/coll.h"

#include <math.h>
#include <string.h>

/* The centre of mass in the frame the pair shares, at the start of
 * the interval. */
static K26V3 pass_com_(const K26AstroCollBody *b)
{
    K26V3 o = k26m3d_quat_rotate_v3(b->orientation, b->com_offset);
    return k26m3d_v3(b->pos0.x + o.x, b->pos0.y + o.y, b->pos0.z + o.z);
}

/* One primitive rotated out of its body frame into the frame the pair
 * shares. The centre is left relative to the body's own position, so
 * the caller's offset carries the separation. */
static K26AstroCollShape pass_world_shape_(const K26AstroCollShape *s,
                                           K26Quat q)
{
    K26AstroCollShape w = *s;
    w.centre = k26m3d_quat_rotate_v3(q, s->centre);
    for (int i = 0; i < 3; i++) w.axis[i] = k26m3d_quat_rotate_v3(q, s->axis[i]);
    return w;
}

double k26astro_coll_curvature_margin(K26V3 dv, double dt)
{
    return 0.5 * sqrt(k26m3d_v3_dot(dv, dv)) * dt;
}

K26AstroCollStatus k26astro_coll_pass(const K26AstroCollBody *bodies,
                                      int n_bodies, double dt,
                                      K26AstroCollContact *out)
{
    if (!bodies || !out) return K26ASTRO_COLL_E_NULL;
    if (!isfinite(dt) || dt < 0.0) return K26ASTRO_COLL_E_BAD_DT;
    memset(out, 0, sizeof *out);
    out->body_a = -1;
    out->body_b = -1;
    if (n_bodies < 2 || dt == 0.0) return K26ASTRO_COLL_OK;

    for (int ia = 0; ia < n_bodies; ia++) {
        const K26AstroCollBody *A = &bodies[ia];
        if (A->n_shapes <= 0) continue;
        K26V3 da = { A->pos1.x - A->pos0.x, A->pos1.y - A->pos0.y,
                     A->pos1.z - A->pos0.z };

        for (int ib = ia + 1; ib < n_bodies; ib++) {
            const K26AstroCollBody *B = &bodies[ib];
            if (B->n_shapes <= 0) continue;

            K26V3 db = { B->pos1.x - B->pos0.x, B->pos1.y - B->pos0.y,
                         B->pos1.z - B->pos0.z };
            K26V3 disp = { db.x - da.x, db.y - da.y, db.z - da.z };
            K26V3 off  = { B->pos0.x - A->pos0.x, B->pos0.y - A->pos0.y,
                           B->pos0.z - A->pos0.z };

            /* The curvature margin. The linear sweep is a chord of the
             * true path; over one interval the two differ by at most
             * half the relative acceleration times the interval
             * squared, and the relative acceleration is read from the
             * velocities the integrator itself produced rather than
             * assumed. Inflating the test volume by it makes a
             * contact the integrated trajectory makes impossible for
             * the linear test to miss. A pair that clears the
             * broadphase and finds nothing in the narrowphase is the
             * margin doing its work, not a defect. */
            K26V3 dv = { (B->vel1.x - B->vel0.x) - (A->vel1.x - A->vel0.x),
                         (B->vel1.y - B->vel0.y) - (A->vel1.y - A->vel0.y),
                         (B->vel1.z - B->vel0.z) - (A->vel1.z - A->vel0.z) };
            double margin = k26astro_coll_curvature_margin(dv, dt);

            K26AstroCollHit broad;
            if (!k26astro_coll_sweep_sphere_sphere(A->bound_radius + margin,
                                                   B->bound_radius + margin,
                                                   off, disp, &broad)) {
                continue;
            }

            for (int pa = 0; pa < A->n_shapes; pa++) {
                K26AstroCollShape sa =
                    pass_world_shape_(&A->shapes[pa], A->orientation);
                for (int pb = 0; pb < B->n_shapes; pb++) {
                    K26AstroCollShape sb =
                        pass_world_shape_(&B->shapes[pb], B->orientation);

                    /* The offset the kernels want is between the two
                     * primitives' centres, so the bodies' separation
                     * and the primitives' placements fold into one
                     * vector here and the kernels stay free of both. */
                    K26V3 poff = { off.x + sb.centre.x - sa.centre.x,
                                   off.y + sb.centre.y - sa.centre.y,
                                   off.z + sb.centre.z - sa.centre.z };
                    K26AstroCollHit h;
                    if (!k26astro_coll_sweep_pair(&sa, &sb, poff, disp, &h)) {
                        continue;
                    }
                    /* Least impact time wins; a tie is broken by the
                     * first body index, then the second, then the
                     * first primitive index, then the second. That is
                     * a total order over the candidates, so the
                     * reported contact never depends on the order the
                     * loops happen to run in. */
                    int better;
                    if (!out->hit) {
                        better = 1;
                    } else if (h.time < out->time) {
                        better = 1;
                    } else if (h.time > out->time) {
                        better = 0;
                    } else if (ia != out->body_a) {
                        better = ia < out->body_a;
                    } else if (ib != out->body_b) {
                        better = ib < out->body_b;
                    } else if (pa != out->shape_a) {
                        better = pa < out->shape_a;
                    } else {
                        better = pb < out->shape_b;
                    }
                    if (!better) continue;

                    out->hit     = 1;
                    out->time    = h.time;
                    out->normal  = h.normal;
                    out->body_a  = ia;
                    out->body_b  = ib;
                    out->shape_a = pa;
                    out->shape_b = pb;
                    /* The contact point the kernel reports is
                     * relative to the first primitive's centre, so it
                     * is carried back to the shared frame here. */
                    K26V3 ca = { A->pos0.x + sa.centre.x,
                                 A->pos0.y + sa.centre.y,
                                 A->pos0.z + sa.centre.z };
                    out->point = k26m3d_v3(ca.x + h.point.x + h.time * da.x,
                                           ca.y + h.point.y + h.time * da.y,
                                           ca.z + h.point.z + h.time * da.z);
                }
            }
        }
    }

    if (out->hit) {
        /* The closing speed along the normal, taken from the
         * velocities at the start of the interval and not from an
         * interpolation between its ends.
         *
         * The end is not usable and the reason is worth stating,
         * because the arithmetic looks more careful the other way. A
         * caller advances the whole interval before this pass looks
         * at it, so by the end the two bodies have already passed
         * through the contact and, for a pair whose paths cross,
         * through each other: the velocity there carries whatever a
         * near-coincident encounter did to them, which is a state the
         * resolution is about to discard. It was measured at roughly
         * nine tenths of the true closing speed on a fixture whose
         * answer was known, which is the kind of error that reads as
         * plausible.
         *
         * The start is entirely on the approach, and over an interval
         * short enough for the linear sweep to be the honest model of
         * the motion it is the honest estimate of the speed at the
         * impact configuration under that same model. Positive means
         * approaching, which is the sense a task reads. */
        const K26AstroCollBody *A = &bodies[out->body_a];
        const K26AstroCollBody *B = &bodies[out->body_b];
        K26V3 rel = { B->vel0.x - A->vel0.x, B->vel0.y - A->vel0.y,
                      B->vel0.z - A->vel0.z };
        out->speed = -k26m3d_v3_dot(rel, out->normal);
    }
    return K26ASTRO_COLL_OK;
}

/* The position at the impact configuration: the same linear
 * interpolation the sweep itself used, so the reported contact and
 * the recorded state agree exactly and not merely closely. */
static K26V3 pass_at_(const K26AstroCollBody *b, double t)
{
    return k26m3d_v3(b->pos0.x + t * (b->pos1.x - b->pos0.x),
                     b->pos0.y + t * (b->pos1.y - b->pos0.y),
                     b->pos0.z + t * (b->pos1.z - b->pos0.z));
}

static K26V3 pass_vel_at_(const K26AstroCollBody *b, double t)
{
    return k26m3d_v3(b->vel0.x + t * (b->vel1.x - b->vel0.x),
                     b->vel0.y + t * (b->vel1.y - b->vel0.y),
                     b->vel0.z + t * (b->vel1.z - b->vel0.z));
}

K26AstroCollStatus k26astro_coll_arrest(const K26AstroCollBody *a,
                                        const K26AstroCollBody *b,
                                        double time,
                                        K26V3 *pos_a, K26V3 *vel_a,
                                        K26V3 *pos_b, K26V3 *vel_b)
{
    if (!a || !b || !pos_a || !vel_a || !pos_b || !vel_b) {
        return K26ASTRO_COLL_E_NULL;
    }
    if (!isfinite(time) || time < 0.0 || time > 1.0) {
        return K26ASTRO_COLL_E_BAD_DT;
    }
    *pos_a = pass_at_(a, time);
    *pos_b = pass_at_(b, time);

    K26V3 va = pass_vel_at_(a, time);
    K26V3 vb = pass_vel_at_(b, time);

    /* The relative translational velocity is removed by a merge that
     * conserves linear momentum: the pair leaves at the velocity of
     * their common centre of mass. A body of non-positive mass is
     * immovable and keeps its own velocity, which the other body then
     * takes. */
    double ma = a->mass > 0.0 ? a->mass : 0.0;
    double mb = b->mass > 0.0 ? b->mass : 0.0;
    if (ma == 0.0 && mb == 0.0) {
        *vel_a = va; *vel_b = vb;
        return K26ASTRO_COLL_OK;
    }
    if (ma == 0.0) { *vel_a = va; *vel_b = va; return K26ASTRO_COLL_OK; }
    if (mb == 0.0) { *vel_a = vb; *vel_b = vb; return K26ASTRO_COLL_OK; }

    double inv = 1.0 / (ma + mb);
    K26V3 v = k26m3d_v3((ma * va.x + mb * vb.x) * inv,
                        (ma * va.y + mb * vb.y) * inv,
                        (ma * va.z + mb * vb.z) * inv);
    *vel_a = v;
    *vel_b = v;
    return K26ASTRO_COLL_OK;
}

/* The inverse inertia in the shared frame: the body-frame tensor
 * conjugated by the orientation. Written out rather than assembled
 * from a matrix product so that the operation count is visible and
 * the arithmetic stays inside the five. */
static K26M3 pass_inv_inertia_world_(const K26AstroCollBody *b)
{
    K26M3 R, out;
    K26V3 e0 = k26m3d_quat_rotate_v3(b->orientation, k26m3d_v3(1, 0, 0));
    K26V3 e1 = k26m3d_quat_rotate_v3(b->orientation, k26m3d_v3(0, 1, 0));
    K26V3 e2 = k26m3d_quat_rotate_v3(b->orientation, k26m3d_v3(0, 0, 1));
    R.m[0][0] = e0.x; R.m[0][1] = e1.x; R.m[0][2] = e2.x;
    R.m[1][0] = e0.y; R.m[1][1] = e1.y; R.m[1][2] = e2.y;
    R.m[2][0] = e0.z; R.m[2][1] = e1.z; R.m[2][2] = e2.z;

    K26M3 tmp;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += R.m[i][k] * b->inv_inertia.m[k][j];
            tmp.m[i][j] = s;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += tmp.m[i][k] * R.m[j][k];
            out.m[i][j] = s;
        }
    }
    return out;
}

static K26V3 pass_m3_mul_(K26M3 m, K26V3 v)
{
    return k26m3d_v3(m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
                     m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
                     m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z);
}

K26AstroCollStatus k26astro_coll_bounce(const K26AstroCollBody *a,
                                        const K26AstroCollBody *b,
                                        const K26AstroCollContact *hit,
                                        double restitution,
                                        double friction,
                                        K26V3 *pos_a, K26V3 *vel_a,
                                        K26V3 *omega_a,
                                        K26V3 *pos_b, K26V3 *vel_b,
                                        K26V3 *omega_b)
{
    if (!a || !b || !hit || !pos_a || !vel_a || !omega_a || !pos_b ||
        !vel_b || !omega_b) {
        return K26ASTRO_COLL_E_NULL;
    }
    if (!isfinite(restitution) || restitution < 0.0 || restitution > 1.0) {
        return K26ASTRO_COLL_E_BAD_DT;
    }
    if (!isfinite(friction) || friction < 0.0) {
        return K26ASTRO_COLL_E_BAD_DT;
    }
    double t = hit->time;
    *pos_a = pass_at_(a, t);
    *pos_b = pass_at_(b, t);
    *omega_a = a->omega;
    *omega_b = b->omega;

    K26V3 va = pass_vel_at_(a, t);
    K26V3 vb = pass_vel_at_(b, t);
    *vel_a = va;
    *vel_b = vb;

    double ma = a->mass > 0.0 ? a->mass : 0.0;
    double mb = b->mass > 0.0 ? b->mass : 0.0;
    if (ma == 0.0 && mb == 0.0) return K26ASTRO_COLL_OK;

    /* The lever arms from each centre of mass to the contact point,
     * at the impact configuration. */
    K26V3 ca = pass_com_(a), cb = pass_com_(b);
    ca = k26m3d_v3(ca.x + t * (a->pos1.x - a->pos0.x),
                   ca.y + t * (a->pos1.y - a->pos0.y),
                   ca.z + t * (a->pos1.z - a->pos0.z));
    cb = k26m3d_v3(cb.x + t * (b->pos1.x - b->pos0.x),
                   cb.y + t * (b->pos1.y - b->pos0.y),
                   cb.z + t * (b->pos1.z - b->pos0.z));
    K26V3 ra = { hit->point.x - ca.x, hit->point.y - ca.y,
                 hit->point.z - ca.z };
    K26V3 rb = { hit->point.x - cb.x, hit->point.y - cb.y,
                 hit->point.z - cb.z };

    K26M3 ia = pass_inv_inertia_world_(a);
    K26M3 ib = pass_inv_inertia_world_(b);

    /* The velocities of the material points in contact, which is
     * where the angular state enters: a spinning body's surface moves
     * even when its centre does not. */
    K26V3 wa = k26m3d_quat_rotate_v3(a->orientation, a->omega);
    K26V3 wb = k26m3d_quat_rotate_v3(b->orientation, b->omega);
    K26V3 pa = k26m3d_v3_cross(wa, ra);
    K26V3 pb = k26m3d_v3_cross(wb, rb);
    K26V3 rel = { (vb.x + pb.x) - (va.x + pa.x),
                  (vb.y + pb.y) - (va.y + pa.y),
                  (vb.z + pb.z) - (va.z + pa.z) };
    double vn = k26m3d_v3_dot(rel, hit->normal);
    /* Already separating: a conservative kernel may report a contact
     * for a pair that is moving apart, and answering it with an
     * impulse would create energy from a margin. */
    if (vn >= 0.0) return K26ASTRO_COLL_OK;

    /* The effective mass at the contact point. Omitting the two
     * angular terms is the recorded defect in the prior art this
     * capability does not carry: without them an off-centre impulse
     * is too large, because the geometry's resistance to rotating
     * about the contact is left out of the denominator. */
    double inv_m = (ma > 0.0 ? 1.0 / ma : 0.0) + (mb > 0.0 ? 1.0 / mb : 0.0);
    K26V3 angA = k26m3d_v3_cross(pass_m3_mul_(ia, k26m3d_v3_cross(ra, hit->normal)), ra);
    K26V3 angB = k26m3d_v3_cross(pass_m3_mul_(ib, k26m3d_v3_cross(rb, hit->normal)), rb);
    double denom = inv_m;
    if (ma > 0.0) denom += k26m3d_v3_dot(angA, hit->normal);
    if (mb > 0.0) denom += k26m3d_v3_dot(angB, hit->normal);
    if (denom <= 0.0) return K26ASTRO_COLL_OK;

    double j = -(1.0 + restitution) * vn / denom;
    K26V3 imp = k26m3d_v3(hit->normal.x * j, hit->normal.y * j,
                          hit->normal.z * j);

    /* The tangential impulse. The sliding at the contact point is
     * what is left of the relative velocity once the normal part is
     * removed; the friction impulse opposes it, and is whatever
     * would stop it outright or the coefficient times the normal
     * impulse, whichever is smaller. A pair with no sliding gets
     * none, which is why the direction is only formed when there is
     * one to form it from. */
    K26V3 timp = { 0.0, 0.0, 0.0 };
    if (friction > 0.0) {
        K26V3 vt = { rel.x - hit->normal.x * vn,
                     rel.y - hit->normal.y * vn,
                     rel.z - hit->normal.z * vn };
        double vt2 = k26m3d_v3_dot(vt, vt);
        if (vt2 > 0.0) {
            double inv = 1.0 / sqrt(vt2);
            K26V3 tdir = k26m3d_v3(vt.x * inv, vt.y * inv, vt.z * inv);
            double td = inv_m;
            K26V3 ta = k26m3d_v3_cross(
                pass_m3_mul_(ia, k26m3d_v3_cross(ra, tdir)), ra);
            K26V3 tb = k26m3d_v3_cross(
                pass_m3_mul_(ib, k26m3d_v3_cross(rb, tdir)), rb);
            if (ma > 0.0) td += k26m3d_v3_dot(ta, tdir);
            if (mb > 0.0) td += k26m3d_v3_dot(tb, tdir);
            if (td > 0.0) {
                double jt = sqrt(vt2) / td;
                double cap = friction * j;
                if (jt > cap) jt = cap;
                /* Against the sliding, not along it: the second
                 * body receives the impulse as written, and its
                 * tangential motion relative to the first is what is
                 * being opposed. */
                timp = k26m3d_v3(-tdir.x * jt, -tdir.y * jt, -tdir.z * jt);
            }
        }
    }
    imp = k26m3d_v3(imp.x + timp.x, imp.y + timp.y, imp.z + timp.z);

    if (ma > 0.0) {
        *vel_a = k26m3d_v3(va.x - imp.x / ma, va.y - imp.y / ma,
                           va.z - imp.z / ma);
        K26V3 dwa = pass_m3_mul_(ia, k26m3d_v3_cross(ra, imp));
        K26V3 body = k26m3d_quat_rotate_v3(k26m3d_quat_conj(a->orientation),
                                           dwa);
        *omega_a = k26m3d_v3(a->omega.x - body.x, a->omega.y - body.y,
                             a->omega.z - body.z);
    }
    if (mb > 0.0) {
        *vel_b = k26m3d_v3(vb.x + imp.x / mb, vb.y + imp.y / mb,
                           vb.z + imp.z / mb);
        K26V3 dwb = pass_m3_mul_(ib, k26m3d_v3_cross(rb, imp));
        K26V3 body = k26m3d_quat_rotate_v3(k26m3d_quat_conj(b->orientation),
                                           dwb);
        *omega_b = k26m3d_v3(b->omega.x + body.x, b->omega.y + body.y,
                             b->omega.z + body.z);
    }
    return K26ASTRO_COLL_OK;
}
