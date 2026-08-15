/* port.c: docking port geometry and the capture envelope test.
 *
 * A capture envelope is a statement about the instant of first
 * contact: how fast the two interfaces were closing, how far they
 * were from coaxial, and how fast that misalignment was changing.
 * This file resolves those quantities from the state the sweep
 * reports and compares them against declared limits.
 *
 * Two properties of the model are worth stating where the code is.
 *
 * The state is taken at the impact configuration, not at the end of
 * the sub-advance. By the end of the sub-advance the resolution has
 * already removed the pair's relative velocity, so a closing rate
 * read there is zero for every approach and the test would refuse
 * every one of them.
 *
 * The angles are Euler angles and their extraction calls the inverse
 * trigonometric functions. No rearrangement removes them, since an
 * angle is not an algebraic function of a rotation matrix, so this
 * file carries the per-binary determinism claim rather than the
 * cross-platform one the swept kernels carry. The header says so
 * beside the declarations.
 */

#include "k26astro_coll/coll.h"

#include <math.h>
#include <string.h>

static double port_abs_(double v)
{
    return v < 0.0 ? -v : v;
}

static K26V3 port_axpy_(K26V3 a, K26V3 b, double s)
{
    return k26m3d_v3(a.x + s * b.x, a.y + s * b.y, a.z + s * b.z);
}

/* The component of v perpendicular to a unit direction n. */
static K26V3 port_perp_(K26V3 v, K26V3 n)
{
    return port_axpy_(v, n, -k26m3d_v3_dot(v, n));
}

int k26astro_coll_port_basis(K26V3 axis, K26V3 roll_ref, K26V3 at,
                             K26AstroCollPort *out)
{
    if (!out) return 1;
    memset(out, 0, sizeof *out);
    out->at = at;

    double al = sqrt(k26m3d_v3_dot(axis, axis));
    K26V3  x;
    if (al > 0.0) {
        x = k26m3d_v3(axis.x / al, axis.y / al, axis.z / al);
    } else {
        x = k26m3d_v3(1.0, 0.0, 0.0);
    }

    int    fallback = 0;
    K26V3  y = port_perp_(roll_ref, x);
    double yl = sqrt(k26m3d_v3_dot(y, y));
    if (!(yl > 0.0)) {
        /* The roll reference names no direction in the plane. Take
         * the world axis least aligned with the port axis, which is
         * one fixed choice for one input rather than whichever axis a
         * loop happened to reach first. */
        double ax = port_abs_(x.x), ay = port_abs_(x.y), az = port_abs_(x.z);
        K26V3  pick = k26m3d_v3(1.0, 0.0, 0.0);
        if (ay <= ax && ay <= az)      pick = k26m3d_v3(0.0, 1.0, 0.0);
        else if (az <= ax && az <= ay) pick = k26m3d_v3(0.0, 0.0, 1.0);
        y  = port_perp_(pick, x);
        yl = sqrt(k26m3d_v3_dot(y, y));
        fallback = 1;
    }
    y = k26m3d_v3(y.x / yl, y.y / yl, y.z / yl);

    out->axis[0] = x;
    out->axis[1] = y;
    out->axis[2] = k26m3d_v3_cross(x, y);
    return fallback;
}

/* A port's origin and axes in the frame the pair shares, at the given
 * interval fraction. */
static void port_world_(const K26AstroCollBody *b, const K26AstroCollPort *p,
                        double time, K26V3 *origin, K26V3 axis[3])
{
    K26V3 pos = k26m3d_v3(b->pos0.x + (b->pos1.x - b->pos0.x) * time,
                          b->pos0.y + (b->pos1.y - b->pos0.y) * time,
                          b->pos0.z + (b->pos1.z - b->pos0.z) * time);
    K26V3 at  = k26m3d_quat_rotate_v3(b->orientation, p->at);
    *origin   = k26m3d_v3(pos.x + at.x, pos.y + at.y, pos.z + at.z);
    for (int i = 0; i < 3; i++) {
        axis[i] = k26m3d_quat_rotate_v3(b->orientation, p->axis[i]);
    }
}

/* The body's own velocity at the given fraction. It is the velocity
 * of the centre of mass, which is the convention the resolution reads
 * it in, and it is interpolated to the impact fraction so that the
 * rates this file reports and the configuration the arrest writes
 * describe the same instant. */
static K26V3 port_vel_at_(const K26AstroCollBody *b, double time)
{
    return k26m3d_v3(b->vel0.x + (b->vel1.x - b->vel0.x) * time,
                     b->vel0.y + (b->vel1.y - b->vel0.y) * time,
                     b->vel0.z + (b->vel1.z - b->vel0.z) * time);
}

/* Velocity of a point rigidly attached to a body, given the point's
 * offset from the body's centre of mass in the shared frame. */
static K26V3 port_point_vel_(const K26AstroCollBody *b, K26V3 arm, double time)
{
    K26V3 v = port_vel_at_(b, time);
    K26V3 w = k26m3d_quat_rotate_v3(b->orientation, b->omega);
    K26V3 c = k26m3d_v3_cross(w, arm);
    return k26m3d_v3(v.x + c.x, v.y + c.y, v.z + c.z);
}

/* The centre of mass in the shared frame at the given fraction. */
static K26V3 port_com_(const K26AstroCollBody *b, double time)
{
    K26V3 pos = k26m3d_v3(b->pos0.x + (b->pos1.x - b->pos0.x) * time,
                          b->pos0.y + (b->pos1.y - b->pos0.y) * time,
                          b->pos0.z + (b->pos1.z - b->pos0.z) * time);
    K26V3 o   = k26m3d_quat_rotate_v3(b->orientation, b->com_offset);
    return k26m3d_v3(pos.x + o.x, pos.y + o.y, pos.z + o.z);
}

K26AstroCollStatus k26astro_coll_port_state(const K26AstroCollBody *active,
                                            const K26AstroCollPort *ap,
                                            const K26AstroCollBody *passive,
                                            const K26AstroCollPort *pp,
                                            double time,
                                            K26AstroCollPortState *out)
{
    if (!active || !ap || !passive || !pp || !out) return K26ASTRO_COLL_E_NULL;
    if (!isfinite(time) || time < 0.0 || time > 1.0) {
        return K26ASTRO_COLL_E_BAD_DT;
    }
    memset(out, 0, sizeof *out);

    K26V3 apos, aax[3], ppos, pax[3];
    port_world_(active, ap, time, &apos, aax);
    port_world_(passive, pp, time, &ppos, pax);

    /* The mated configuration is the passive port turned to face the
     * way the active one arrives: its outward normal reversed, its
     * roll reference kept. Every residual below is the departure of
     * the active port frame from that frame. */
    K26V3 dax[3];
    dax[0] = k26m3d_v3_neg(pax[0]);
    dax[1] = pax[1];
    dax[2] = k26m3d_v3_cross(dax[0], dax[1]);

    K26V3 delta = k26m3d_v3_sub(apos, ppos);
    out->axial   = k26m3d_v3_dot(delta, pax[0]);
    K26V3 lat    = port_perp_(delta, pax[0]);
    out->lateral = sqrt(k26m3d_v3_dot(lat, lat));

    /* The active port basis expressed on the mated frame's axes:
     * m[i][j] is the i-th mated axis against the j-th active axis. */
    double m[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) m[i][j] = k26m3d_v3_dot(dax[i], aax[j]);
    }
    double s = m[2][0];
    if (s > 1.0)  s = 1.0;
    if (s < -1.0) s = -1.0;
    double pitch = -asin(s);
    double yaw, roll;
    double c2 = m[0][0] * m[0][0] + m[1][0] * m[1][0];
    if (c2 > 0.0) {
        yaw  = atan2(m[1][0], m[0][0]);
        roll = atan2(m[2][1], m[2][2]);
    } else {
        /* The active axis lies along the mated frame's third axis, so
         * yaw and roll turn about the same line and only their sum is
         * defined. The whole of it is reported as roll, which keeps
         * the vector sum of pitch and yaw finite and truthful; the
         * configuration is a quarter turn from mated and fails every
         * envelope this test is written for. */
        yaw  = 0.0;
        roll = atan2(-m[0][1], m[1][1]);
    }
    out->pitchyaw = sqrt(pitch * pitch + yaw * yaw);
    out->roll     = port_abs_(roll);

    K26V3 avel = port_point_vel_(active,
                                 k26m3d_v3_sub(apos, port_com_(active, time)),
                                 time);
    K26V3 pvel = port_point_vel_(passive,
                                 k26m3d_v3_sub(ppos, port_com_(passive, time)),
                                 time);
    K26V3 vrel = k26m3d_v3_sub(avel, pvel);
    out->v_axial = -k26m3d_v3_dot(vrel, pax[0]);
    K26V3 vlat   = port_perp_(vrel, pax[0]);
    out->v_lateral = sqrt(k26m3d_v3_dot(vlat, vlat));

    K26V3 wa   = k26m3d_quat_rotate_v3(active->orientation, active->omega);
    K26V3 wp   = k26m3d_quat_rotate_v3(passive->orientation, passive->omega);
    K26V3 wrel = k26m3d_v3_sub(wa, wp);
    out->v_roll = port_abs_(k26m3d_v3_dot(wrel, pax[0]));
    K26V3 wlat  = port_perp_(wrel, pax[0]);
    out->v_pitchyaw = sqrt(k26m3d_v3_dot(wlat, wlat));

    /* The rate the envelope's note adds, in the note's own terms:
     * what the combination of the lateral rate at the interface and
     * the pitch or yaw rate produces at the active vehicle's centre
     * of mass, which is the relative velocity carried along the arm
     * from the interface to that centre.
     *
     * It is not the difference of the two bodies' own velocities.
     * Those are the velocities of two points that are not in the same
     * place, so their difference carries the passive body's own
     * rotation across the separation, which is a rate of the
     * configuration and not a rate at the interface. On a pair
     * holding station on one orbit about a rotating structure that
     * term alone can exceed the limit while nothing at the interface
     * is moving at all. */
    K26V3 arm  = k26m3d_v3_sub(port_com_(active, time), apos);
    K26V3 cvel = k26m3d_v3_add(vrel, k26m3d_v3_cross(wrel, arm));
    K26V3 clat = port_perp_(cvel, pax[0]);
    out->v_lateral_cg = sqrt(k26m3d_v3_dot(clat, clat));

    return K26ASTRO_COLL_OK;
}

int k26astro_coll_port_captured(const K26AstroCollPortState *s,
                                const K26AstroCollEnvelope *e)
{
    if (!s || !e) return 0;
    if (!(s->v_axial >= e->axial_rate_min)) return 0;
    if (!(s->v_axial <= e->axial_rate_max)) return 0;
    if (!(s->v_lateral <= e->lateral_rate)) return 0;
    if (!(s->v_pitchyaw <= e->pitchyaw_rate)) return 0;
    if (!(s->v_roll <= e->roll_rate)) return 0;
    if (!(s->lateral <= e->lateral)) return 0;
    if (!(s->pitchyaw <= e->pitchyaw)) return 0;
    if (!(s->roll <= e->roll)) return 0;
    if (!(s->v_lateral_cg <= e->lateral_rate)) return 0;
    return 1;
}
