/* kernels.c: the swept narrowphase.
 *
 * Every kernel answers one question: over an interval in which the
 * second primitive moves in a straight line relative to the first and
 * neither rotates, what is the least fraction of the interval at
 * which the two touch? Holding the orientations is what makes the
 * question exactly solvable rather than a search, and the sub-advance
 * is short enough for the held orientation to be the honest model of
 * one; the curvature the linear model omits is covered by the
 * broadphase margin in pass.c and not by a tolerance here.
 *
 * Arithmetic is restricted to the five correctly rounded IEEE-754
 * operations, so a contact time is reproducible across platforms.
 * Where a magnitude is wanted it is written as a comparison and a
 * negation rather than as a call to fabs, and where a smaller of two
 * values is wanted it is written as a comparison rather than as a
 * call to fmin: both are exact, and neither pulls in a library
 * function whose rounding this file would then have to argue about.
 */

#include "k26astro_coll/coll.h"

#include <math.h>
#include <string.h>

const char *k26astro_coll_status_str(K26AstroCollStatus s)
{
    switch (s) {
    case K26ASTRO_COLL_OK:       return "ok";
    case K26ASTRO_COLL_E_NULL:   return "null argument";
    case K26ASTRO_COLL_E_BAD_DT: return "interval is not finite or is negative";
    case K26ASTRO_COLL_E_LIMIT:  return "more colliders than the fixed set holds";
    }
    return "unknown status";
}

static double coll_abs_(double v)
{
    return v < 0.0 ? -v : v;
}

static void coll_clear_(K26AstroCollHit *h)
{
    memset(h, 0, sizeof *h);
}

/* The least root in [0, 1] of  a t^2 + 2 b t + c = 0,  for a >= 0.
 *
 * Returns 1 and writes the root when one exists. A c of zero or less
 * means the pair is already touching at the start of the interval and
 * the root is zero; that is a contact and not an error, because a
 * reset is allowed to place two bodies overlapping.
 *
 * The root is taken in the form that does not cancel. For an
 * approaching pair b is negative, so -b + sqrt(disc) is a sum of two
 * positive quantities and c divided by it is the smaller root to full
 * precision, where the textbook form would subtract two nearly equal
 * numbers exactly when the pair grazes. */
static int coll_first_root_(double a, double b, double c, double *out_t)
{
    if (c <= 0.0) { *out_t = 0.0; return 1; }
    if (a <= 0.0) return 0;              /* no relative motion, and apart */
    if (b >= 0.0) return 0;              /* separating from the start */
    double disc = b * b - a * c;
    if (disc < 0.0) return 0;
    double q = -b + sqrt(disc);
    if (q <= 0.0) return 0;
    double t = c / q;
    if (t < 0.0 || t > 1.0) return 0;
    *out_t = t;
    return 1;
}

int k26astro_coll_sweep_sphere_sphere(double ra, double rb,
                                      K26V3 offset, K26V3 disp,
                                      K26AstroCollHit *out)
{
    if (!out) return 0;
    coll_clear_(out);
    double R = ra + rb;
    double a = k26m3d_v3_dot(disp, disp);
    double b = k26m3d_v3_dot(offset, disp);
    double c = k26m3d_v3_dot(offset, offset) - R * R;
    double t;
    if (!coll_first_root_(a, b, c, &t)) return 0;

    K26V3 sep = { offset.x + t * disp.x,
                  offset.y + t * disp.y,
                  offset.z + t * disp.z };
    double len2 = k26m3d_v3_dot(sep, sep);
    K26V3 n;
    if (len2 > 0.0) {
        double inv = 1.0 / sqrt(len2);
        n = k26m3d_v3(sep.x * inv, sep.y * inv, sep.z * inv);
    } else {
        /* Concentric: no direction is distinguished by the geometry,
         * so one is chosen by rule rather than by whatever the
         * arithmetic happens to produce. */
        n = k26m3d_v3(1.0, 0.0, 0.0);
    }
    out->hit    = 1;
    out->time   = t;
    out->normal = n;
    out->point  = k26m3d_v3(n.x * ra, n.y * ra, n.z * ra);
    return 1;
}

/* The moving point starts at `offset` from the segment's frame and
 * travels `disp` over the interval. Three regions, each exactly
 * solvable, and the answer is the least valid time among them: the
 * two end caps, where the closest feature is an endpoint, and the
 * side, where it is the interior of the segment. Which region holds
 * is decided at the candidate time itself rather than at the start,
 * because the closest feature changes as the point travels. */
int k26astro_coll_sweep_point_segment(K26V3 a, K26V3 b, double radius,
                                      K26V3 offset, K26V3 disp,
                                      K26AstroCollHit *out)
{
    if (!out) return 0;
    coll_clear_(out);

    K26V3 e  = { b.x - a.x, b.y - a.y, b.z - a.z };
    double L2 = k26m3d_v3_dot(e, e);
    K26V3 wa = { offset.x - a.x, offset.y - a.y, offset.z - a.z };
    K26V3 wb = { offset.x - b.x, offset.y - b.y, offset.z - b.z };

    double best = 2.0;
    int    found = 0;
    K26V3  closest = a;

    /* Cap at the first endpoint, valid where the projection has not
     * yet reached the segment. */
    {
        double t;
        if (coll_first_root_(k26m3d_v3_dot(disp, disp),
                             k26m3d_v3_dot(wa, disp),
                             k26m3d_v3_dot(wa, wa) - radius * radius, &t)) {
            double s = L2 > 0.0
                     ? (k26m3d_v3_dot(wa, e) + t * k26m3d_v3_dot(disp, e)) / L2
                     : 0.0;
            if (s <= 0.0 || L2 == 0.0) {
                best = t; found = 1; closest = a;
            }
        }
    }
    /* Cap at the second endpoint. */
    if (L2 > 0.0) {
        double t;
        if (coll_first_root_(k26m3d_v3_dot(disp, disp),
                             k26m3d_v3_dot(wb, disp),
                             k26m3d_v3_dot(wb, wb) - radius * radius, &t)) {
            double s = (k26m3d_v3_dot(wa, e) + t * k26m3d_v3_dot(disp, e)) / L2;
            if (s >= 1.0 && (!found || t < best)) {
                best = t; found = 1; closest = b;
            }
        }
    }
    /* The side. The squared distance to the infinite line is the
     * squared distance to the endpoint less the squared projection
     * onto the direction, and both are quadratics in the interval
     * fraction, so their difference is one too. */
    if (L2 > 0.0) {
        double u0 = k26m3d_v3_dot(wa, e);
        double ud = k26m3d_v3_dot(disp, e);
        double qa = k26m3d_v3_dot(disp, disp) - ud * ud / L2;
        double qb = k26m3d_v3_dot(wa, disp)   - u0 * ud / L2;
        double qc = k26m3d_v3_dot(wa, wa) - u0 * u0 / L2 - radius * radius;
        double t;
        if (coll_first_root_(qa, qb, qc, &t)) {
            double s = (u0 + t * ud) / L2;
            if (s >= 0.0 && s <= 1.0 && (!found || t < best)) {
                best = t; found = 1;
                closest = k26m3d_v3(a.x + s * e.x, a.y + s * e.y,
                                    a.z + s * e.z);
            }
        }
    }
    if (!found) return 0;

    K26V3 p = { offset.x + best * disp.x,
                offset.y + best * disp.y,
                offset.z + best * disp.z };
    K26V3 sep = { p.x - closest.x, p.y - closest.y, p.z - closest.z };
    double len2 = k26m3d_v3_dot(sep, sep);
    K26V3 n;
    if (len2 > 0.0) {
        double inv = 1.0 / sqrt(len2);
        n = k26m3d_v3(sep.x * inv, sep.y * inv, sep.z * inv);
    } else {
        n = k26m3d_v3(1.0, 0.0, 0.0);
    }
    out->hit    = 1;
    out->time   = best;
    out->normal = n;
    out->point  = closest;
    return 1;
}

/* Two capsules. The closest approach of two segments is attained
 * either between their interiors, where the common normal is
 * perpendicular to both directions, or at an endpoint of one against
 * the other. Both cases are solved in closed form and the least valid
 * time among them is the answer, so this is exact rather than a
 * search. The interior case is skipped when the two directions are
 * near parallel, which costs nothing: for parallel segments the
 * closest approach is attained at an endpoint of one of them, so an
 * endpoint case already covers it. */
int k26astro_coll_sweep_capsule_capsule(K26V3 a0, K26V3 a1, double ra,
                                        K26V3 b0, K26V3 b1, double rb,
                                        K26V3 offset, K26V3 disp,
                                        K26AstroCollHit *out)
{
    if (!out) return 0;
    coll_clear_(out);
    double R = ra + rb;

    double best = 2.0;
    int    found = 0;
    K26AstroCollHit cand;

    /* The second capsule's endpoints against the first's segment. The
     * moving point is the endpoint, offset into the first's frame. */
    K26V3 bpt[2] = { b0, b1 };
    for (int i = 0; i < 2; i++) {
        K26V3 off = { offset.x + bpt[i].x, offset.y + bpt[i].y,
                      offset.z + bpt[i].z };
        if (k26astro_coll_sweep_point_segment(a0, a1, R, off, disp, &cand)) {
            if (!found || cand.time < best) {
                best = cand.time; found = 1;
                *out = cand;
                out->point = k26m3d_v3(cand.point.x + cand.normal.x * ra,
                                       cand.point.y + cand.normal.y * ra,
                                       cand.point.z + cand.normal.z * ra);
            }
        }
    }
    /* The first capsule's endpoints against the second's segment. Here
     * the roles reverse: the first's endpoint moves backwards relative
     * to the second, so the displacement is negated and the reported
     * normal is negated back to the shared convention. */
    K26V3 apt[2] = { a0, a1 };
    K26V3 nb0 = { b0.x + offset.x, b0.y + offset.y, b0.z + offset.z };
    K26V3 nb1 = { b1.x + offset.x, b1.y + offset.y, b1.z + offset.z };
    K26V3 back = { -disp.x, -disp.y, -disp.z };
    for (int i = 0; i < 2; i++) {
        if (k26astro_coll_sweep_point_segment(nb0, nb1, R, apt[i], back,
                                              &cand)) {
            if (!found || cand.time < best) {
                best = cand.time; found = 1;
                out->hit    = 1;
                out->time   = cand.time;
                out->normal = k26m3d_v3(-cand.normal.x, -cand.normal.y,
                                        -cand.normal.z);
                out->point  = k26m3d_v3(cand.point.x + cand.normal.x * rb,
                                        cand.point.y + cand.normal.y * rb,
                                        cand.point.z + cand.normal.z * rb);
            }
        }
    }
    /* The interiors. Along the common normal the separation is an
     * affine function of the interval fraction, so the crossing time
     * is one division. The parameters along both segments are then
     * checked to lie inside their ranges; when they do not, an
     * endpoint case above is the true closest approach and has
     * already been taken. */
    K26V3 ea = { a1.x - a0.x, a1.y - a0.y, a1.z - a0.z };
    K26V3 eb = { b1.x - b0.x, b1.y - b0.y, b1.z - b0.z };
    K26V3 cr = k26m3d_v3_cross(ea, eb);
    double cr2 = k26m3d_v3_dot(cr, cr);
    double la2 = k26m3d_v3_dot(ea, ea), lb2 = k26m3d_v3_dot(eb, eb);
    if (cr2 > K26ASTRO_COLL_PARALLEL_EPS2 * la2 * lb2 && la2 > 0.0 &&
        lb2 > 0.0) {
        double inv = 1.0 / sqrt(cr2);
        K26V3 n = k26m3d_v3(cr.x * inv, cr.y * inv, cr.z * inv);
        K26V3 w0 = { offset.x + b0.x - a0.x, offset.y + b0.y - a0.y,
                     offset.z + b0.z - a0.z };
        double d0 = k26m3d_v3_dot(w0, n);
        double dv = k26m3d_v3_dot(disp, n);
        /* The crossing of either face of the slab of half-width R. */
        for (int sign = 0; sign < 2; sign++) {
            double target = sign == 0 ? R : -R;
            double t;
            if (dv == 0.0) {
                if (coll_abs_(d0) > R) continue;
                t = 0.0;
            } else {
                t = (target - d0) / dv;
                if (t < 0.0 || t > 1.0) continue;
            }
            if (found && t >= best) continue;
            /* Where on each segment the common normal meets, at that
             * time. The two-by-two system is solved directly. */
            K26V3 w = { w0.x + t * disp.x, w0.y + t * disp.y,
                        w0.z + t * disp.z };
            double eaeb = k26m3d_v3_dot(ea, eb);
            double det  = la2 * lb2 - eaeb * eaeb;
            if (det == 0.0) continue;
            double wea = k26m3d_v3_dot(w, ea), web = k26m3d_v3_dot(w, eb);
            double s = ( lb2 * wea - eaeb * web) / det;
            double l = ( eaeb * wea - la2 * web) / det;
            if (s < 0.0 || s > 1.0 || l < 0.0 || l > 1.0) continue;
            K26V3 pa = { a0.x + s * ea.x, a0.y + s * ea.y, a0.z + s * ea.z };
            double sgn = d0 + t * dv >= 0.0 ? 1.0 : -1.0;
            best = t; found = 1;
            out->hit    = 1;
            out->time   = t;
            out->normal = k26m3d_v3(n.x * sgn, n.y * sgn, n.z * sgn);
            out->point  = k26m3d_v3(pa.x + out->normal.x * ra,
                                    pa.y + out->normal.y * ra,
                                    pa.z + out->normal.z * ra);
        }
    }
    return found;
}

/* The half-width of a primitive's projection onto an axis, the axis
 * not normalised. Exact for every kind: a box's is the sum of its
 * half-extents times the magnitudes of its axes' projections, a
 * sphere's is its radius times the axis length, and a capsule's is
 * both. */
static double coll_extent_(const K26AstroCollShape *s, K26V3 n, double nlen)
{
    switch (s->kind) {
    case K26ASTRO_COLL_SPHERE:
        return s->half[0] * nlen;
    case K26ASTRO_COLL_CAPSULE:
        return s->half[0] * nlen +
               s->half[2] * coll_abs_(k26m3d_v3_dot(s->axis[2], n));
    case K26ASTRO_COLL_BOX:
    default:
        return s->half[0] * coll_abs_(k26m3d_v3_dot(s->axis[0], n)) +
               s->half[1] * coll_abs_(k26m3d_v3_dot(s->axis[1], n)) +
               s->half[2] * coll_abs_(k26m3d_v3_dot(s->axis[2], n));
    }
}

/* One candidate axis folded into the running entry and exit times.
 * Returns 0 when this axis separates the pair over the whole
 * interval, which ends the test immediately. */
static int coll_sat_axis_(const K26AstroCollShape *a,
                          const K26AstroCollShape *b,
                          K26V3 n, K26V3 offset, K26V3 disp,
                          double *t_enter, double *t_exit, K26V3 *n_enter)
{
    double n2 = k26m3d_v3_dot(n, n);
    if (n2 <= 0.0) return 1;             /* degenerate axis, no information */
    double nlen = sqrt(n2);
    double E    = coll_extent_(a, n, nlen) + coll_extent_(b, n, nlen);
    /* The projections are compared in units of the axis length, so
     * the extents above carry that length and no division is needed
     * to make the comparison meaningful. */
    double s0 = k26m3d_v3_dot(offset, n);
    double v  = k26m3d_v3_dot(disp, n);

    if (v == 0.0) {
        return coll_abs_(s0) <= E;
    }
    double t1 = (-E - s0) / v;
    double t2 = ( E - s0) / v;
    double lo = t1 < t2 ? t1 : t2;
    double hi = t1 < t2 ? t2 : t1;
    if (lo > *t_enter) {
        *t_enter = lo;
        /* The normal points from the first primitive to the second,
         * which is the sign of the separation at the moment of
         * entry. */
        double at = s0 + lo * v;
        double inv = 1.0 / nlen;
        double sgn = at >= 0.0 ? inv : -inv;
        *n_enter = k26m3d_v3(n.x * sgn, n.y * sgn, n.z * sgn);
    }
    if (hi < *t_exit) *t_exit = hi;
    return *t_enter <= *t_exit;
}

int k26astro_coll_sweep_sat(const K26AstroCollShape *a,
                            const K26AstroCollShape *b,
                            K26V3 offset, K26V3 disp,
                            K26AstroCollHit *out)
{
    if (!a || !b || !out) return 0;
    coll_clear_(out);

    double t_enter = 0.0, t_exit = 1.0;
    K26V3  n_enter = k26m3d_v3(1.0, 0.0, 0.0);

    /* The face normals of whichever primitives have them, then a
     * capsule's own axis, then the cross products, all in a fixed
     * order so that the axis selected on a tie never depends on
     * iteration accident. */
    K26V3 axes_a[3]; int na = 0;
    K26V3 axes_b[3]; int nb = 0;
    if (a->kind == K26ASTRO_COLL_BOX) {
        for (int i = 0; i < 3; i++) axes_a[na++] = a->axis[i];
    } else if (a->kind == K26ASTRO_COLL_CAPSULE) {
        axes_a[na++] = a->axis[2];
    }
    if (b->kind == K26ASTRO_COLL_BOX) {
        for (int i = 0; i < 3; i++) axes_b[nb++] = b->axis[i];
    } else if (b->kind == K26ASTRO_COLL_CAPSULE) {
        axes_b[nb++] = b->axis[2];
    }

    for (int i = 0; i < na; i++) {
        if (!coll_sat_axis_(a, b, axes_a[i], offset, disp,
                            &t_enter, &t_exit, &n_enter)) return 0;
    }
    for (int j = 0; j < nb; j++) {
        if (!coll_sat_axis_(a, b, axes_b[j], offset, disp,
                            &t_enter, &t_exit, &n_enter)) return 0;
    }
    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            K26V3 cr = k26m3d_v3_cross(axes_a[i], axes_b[j]);
            if (k26m3d_v3_dot(cr, cr) <= K26ASTRO_COLL_PARALLEL_EPS2) {
                continue;      /* near parallel: no information here */
            }
            if (!coll_sat_axis_(a, b, cr, offset, disp,
                                &t_enter, &t_exit, &n_enter)) return 0;
        }
    }
    /* A pair with no axis at all is two spheres, which this kernel is
     * never asked for; the dispatcher sends those to the exact one. */
    if (na == 0 && nb == 0) return 0;

    if (t_enter > 1.0 || t_exit < 0.0) return 0;
    double t = t_enter < 0.0 ? 0.0 : t_enter;

    out->hit    = 1;
    out->time   = t;
    out->normal = n_enter;
    /* The contact point. Along the entry normal it is the first
     * primitive's own surface; across the normal it is where the
     * second primitive is, which is what puts a contact on a face
     * where the second primitive actually arrived rather than at the
     * face's centre. Losing the lateral position would make every
     * impact on a flat face read as central, and the angular part of
     * a bounce would then be zero for a case that plainly spins. */
    K26V3 sep = { offset.x + t * disp.x, offset.y + t * disp.y,
                  offset.z + t * disp.z };
    double along = k26m3d_v3_dot(sep, n_enter);
    double ext   = coll_extent_(a, n_enter, 1.0);
    out->point = k26m3d_v3(sep.x - n_enter.x * along + n_enter.x * ext,
                           sep.y - n_enter.y * along + n_enter.y * ext,
                           sep.z - n_enter.z * along + n_enter.z * ext);
    return 1;
}

int k26astro_coll_sweep_pair(const K26AstroCollShape *a,
                             const K26AstroCollShape *b,
                             K26V3 offset, K26V3 disp,
                             K26AstroCollHit *out)
{
    if (!a || !b || !out) return 0;

    if (a->kind == K26ASTRO_COLL_BOX || b->kind == K26ASTRO_COLL_BOX) {
        return k26astro_coll_sweep_sat(a, b, offset, disp, out);
    }
    if (a->kind == K26ASTRO_COLL_SPHERE && b->kind == K26ASTRO_COLL_SPHERE) {
        return k26astro_coll_sweep_sphere_sphere(a->half[0], b->half[0],
                                                 offset, disp, out);
    }
    /* A sphere is the capsule whose two endpoints coincide, so one
     * kernel serves the remaining three combinations and there is no
     * separate sphere-against-capsule path to keep agreeing with this
     * one. */
    /* Both primitives are expressed relative to the first one's
     * centre, which is the origin the caller's offset is measured
     * from. */
    K26V3 a0 = k26m3d_v3(0.0, 0.0, 0.0), a1 = a0;
    if (a->kind == K26ASTRO_COLL_CAPSULE) {
        a0 = k26m3d_v3(-a->axis[2].x * a->half[2],
                       -a->axis[2].y * a->half[2],
                       -a->axis[2].z * a->half[2]);
        a1 = k26m3d_v3( a->axis[2].x * a->half[2],
                        a->axis[2].y * a->half[2],
                        a->axis[2].z * a->half[2]);
    }
    K26V3 b0 = k26m3d_v3(0.0, 0.0, 0.0), b1 = b0;
    if (b->kind == K26ASTRO_COLL_CAPSULE) {
        b0 = k26m3d_v3(-b->axis[2].x * b->half[2], -b->axis[2].y * b->half[2],
                       -b->axis[2].z * b->half[2]);
        b1 = k26m3d_v3( b->axis[2].x * b->half[2],  b->axis[2].y * b->half[2],
                        b->axis[2].z * b->half[2]);
    }
    return k26astro_coll_sweep_capsule_capsule(a0, a1, a->half[0],
                                               b0, b1, b->half[0],
                                               offset, disp, out);
}
