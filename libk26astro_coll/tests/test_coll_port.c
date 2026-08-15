/* test_coll_port.c: docking port geometry and the capture envelope.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   A residual arm that compared the library's answer against the
 *   same formula written again would measure nothing but a copy. The
 *   alignment residuals are therefore checked against values the
 *   fixture CONSTRUCTS rather than computes: the active port frame is
 *   built from a chosen yaw, pitch and roll and placed at a chosen
 *   axial and lateral offset, so the expected numbers are the inputs
 *   and the library has to recover them.
 *
 *   A fixture with both bodies at the origin, unrotated and at rest,
 *   would let a residual resolved on the wrong axis, or taken in the
 *   wrong frame, agree with one resolved correctly. Both bodies carry
 *   a general orientation, a translation, a linear velocity, an
 *   angular velocity and an offset centre of mass, and the passive
 *   port sits away from its body's origin, so no two axes and no two
 *   frames coincide anywhere in the fixture.
 *
 *   An envelope arm that tested a state well inside and one well
 *   outside would pass for any limit within a wide band. Each
 *   condition is tested exactly at its limit and one unit of least
 *   precision beyond it, so the arm fails for any limit that is not
 *   the declared one.
 *
 *   The envelope's derived condition on the rate at the centres of
 *   mass would be untested by any state in which it equals the
 *   lateral rate at the ports. Its own arm holds the port rate inside
 *   the limit while the centre-of-mass rate exceeds it, so deleting
 *   the condition changes the verdict.
 */
#include "k26astro_coll/coll.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

static void near_(const char *what, double got, double want, double tol)
{
    double e = got - want;
    if (e < 0.0) e = -e;
    printf("  %-40s %+.15f  want %+.15f  err %.3e\n", what, got, want, e);
    ASSERT(e <= tol);
}

/* ---- fixture arithmetic, written here and not borrowed ----------- */

static K26V3 v3_(double x, double y, double z) { return k26m3d_v3(x, y, z); }

static K26V3 add_(K26V3 a, K26V3 b) { return v3_(a.x + b.x, a.y + b.y, a.z + b.z); }
static K26V3 sub_(K26V3 a, K26V3 b) { return v3_(a.x - b.x, a.y - b.y, a.z - b.z); }
static K26V3 mul_(K26V3 a, double s) { return v3_(a.x * s, a.y * s, a.z * s); }
static double dot_(K26V3 a, K26V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static K26V3 cross_(K26V3 a, K26V3 b)
{
    return v3_(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x);
}
static double len_(K26V3 a) { return sqrt(dot_(a, a)); }
static K26V3 perp_(K26V3 v, K26V3 n) { return sub_(v, mul_(n, dot_(v, n))); }

/* The yaw-pitch-roll rotation the docking standard's coordinate
 * transition names: yaw about the third axis, then pitch about the
 * second, then roll about the first. Written out so the fixture's
 * construction owes nothing to the library's extraction. */
static void euler_matrix_(double yaw, double pitch, double roll, double m[3][3])
{
    double cy = cos(yaw),   sy = sin(yaw);
    double cp = cos(pitch), sp = sin(pitch);
    double cr = cos(roll),  sr = sin(roll);
    m[0][0] = cp * cy;
    m[0][1] = cy * sp * sr - sy * cr;
    m[0][2] = cy * sp * cr + sy * sr;
    m[1][0] = cp * sy;
    m[1][1] = sy * sp * sr + cy * cr;
    m[1][2] = sy * sp * cr - cy * sr;
    m[2][0] = -sp;
    m[2][1] = cp * sr;
    m[2][2] = cp * cr;
}

/* A quaternion whose rotation carries the identity axes onto the
 * three given columns. Shepperd's branch on the largest denominator,
 * so no near-zero divide. */
static K26Quat quat_from_columns_(K26V3 c0, K26V3 c1, K26V3 c2)
{
    double m[3][3];
    m[0][0] = c0.x; m[0][1] = c1.x; m[0][2] = c2.x;
    m[1][0] = c0.y; m[1][1] = c1.y; m[1][2] = c2.y;
    m[2][0] = c0.z; m[2][1] = c1.z; m[2][2] = c2.z;
    double tr = m[0][0] + m[1][1] + m[2][2];
    K26Quat q;
    if (tr > 0.0) {
        double s = sqrt(tr + 1.0) * 2.0;
        q.w = 0.25 * s;
        q.x = (m[2][1] - m[1][2]) / s;
        q.y = (m[0][2] - m[2][0]) / s;
        q.z = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        double s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q.w = (m[2][1] - m[1][2]) / s;
        q.x = 0.25 * s;
        q.y = (m[0][1] + m[1][0]) / s;
        q.z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        double s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q.w = (m[0][2] - m[2][0]) / s;
        q.x = (m[0][1] + m[1][0]) / s;
        q.y = 0.25 * s;
        q.z = (m[1][2] + m[2][1]) / s;
    } else {
        double s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q.w = (m[1][0] - m[0][1]) / s;
        q.x = (m[0][2] + m[2][0]) / s;
        q.y = (m[1][2] + m[2][1]) / s;
        q.z = 0.25 * s;
    }
    return q;
}

static K26V3 rot_(K26Quat q, K26V3 v) { return k26m3d_quat_rotate_v3(q, v); }

static K26V3 lerp_(K26V3 a, K26V3 b, double t)
{
    return v3_(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
               a.z + (b.z - a.z) * t);
}

/* ---- the fixture ------------------------------------------------- */

/* Chosen residuals. Nothing round, nothing symmetric, every angle
 * distinct in magnitude so a swapped pair cannot pass. */
#define YAW_    0.031000000000000
#define PITCH_ (-0.017000000000000)
#define ROLL_   0.043000000000000
#define AXIAL_  0.012500000000000
#define TIME_   0.370000000000000

int main(void)
{
    /* The passive body: rotated, translated, moving, spinning, with
     * its centre of mass off its own origin and its port well away
     * from both. */
    K26AstroCollBody P;
    memset(&P, 0, sizeof P);
    P.pos0 = v3_(13.0, -7.0, 5.0);
    P.pos1 = v3_(13.0 + 0.31 * 0.05, -7.0 - 0.17 * 0.05, 5.0 + 0.23 * 0.05);
    P.vel0 = v3_(0.31, -0.17, 0.23);
    P.vel1 = v3_(0.312, -0.169, 0.2315);
    P.omega = v3_(0.0040, -0.0021, 0.0063);
    P.com_offset = v3_(0.40, -0.20, 0.10);
    P.mass = 3.5e5;
    {
        K26V3 ax = v3_(2.0, -1.0, 3.0);
        ax = mul_(ax, 1.0 / len_(ax));
        P.orientation = k26m3d_quat_from_axis_angle(ax, 0.37);
    }

    K26AstroCollPort pp;
    ASSERT(k26astro_coll_port_basis(v3_(1.0, 0.0, 0.0), v3_(0.0, 1.0, 0.0),
                                    v3_(20.0, 0.0, 2.0), &pp) == 0);

    /* ---- 1. the basis ------------------------------------------- */
    {
        printf("basis: orthonormal, right handed, and total\n");
        near_("x.x", pp.axis[0].x, 1.0, 0.0);
        for (int i = 0; i < 3; i++) {
            near_("unit", len_(pp.axis[i]), 1.0, 1e-15);
        }
        near_("x.y", dot_(pp.axis[0], pp.axis[1]), 0.0, 1e-15);
        near_("x.z", dot_(pp.axis[0], pp.axis[2]), 0.0, 1e-15);
        near_("y.z", dot_(pp.axis[1], pp.axis[2]), 0.0, 1e-15);
        K26V3 c = cross_(pp.axis[0], pp.axis[1]);
        near_("right handed", len_(sub_(c, pp.axis[2])), 0.0, 1e-15);

        /* A non-unit axis is normalised rather than refused. */
        K26AstroCollPort q;
        ASSERT(k26astro_coll_port_basis(v3_(0.0, 0.0, 7.0), v3_(3.0, 0.0, 0.0),
                                        v3_(0.0, 0.0, 0.0), &q) == 0);
        near_("normalised axis z", q.axis[0].z, 1.0, 0.0);
        near_("roll reference x", q.axis[1].x, 1.0, 1e-15);

        /* A roll reference along the axis names no direction in the
         * plane; the fallback still returns an orthonormal basis and
         * says it was used. */
        K26AstroCollPort r;
        ASSERT(k26astro_coll_port_basis(v3_(1.0, 0.0, 0.0), v3_(2.0, 0.0, 0.0),
                                        v3_(0.0, 0.0, 0.0), &r) == 1);
        near_("fallback unit y", len_(r.axis[1]), 1.0, 1e-15);
        near_("fallback y perpendicular", dot_(r.axis[0], r.axis[1]), 0.0, 1e-15);
        n_pass++;
    }

    /* The passive port frame in the shared frame, at TIME_. */
    K26V3 ppos, pax[3];
    {
        K26V3 pos = lerp_(P.pos0, P.pos1, TIME_);
        ppos = add_(pos, rot_(P.orientation, pp.at));
        for (int i = 0; i < 3; i++) pax[i] = rot_(P.orientation, pp.axis[i]);
    }
    /* The mated frame: the passive port turned to face the arrival. */
    K26V3 dax[3];
    dax[0] = mul_(pax[0], -1.0);
    dax[1] = pax[1];
    dax[2] = cross_(dax[0], dax[1]);

    /* The active body, constructed to sit at the chosen residuals. */
    double m[3][3];
    euler_matrix_(YAW_, PITCH_, ROLL_, m);
    K26V3 aax[3];
    for (int j = 0; j < 3; j++) {
        aax[j] = add_(add_(mul_(dax[0], m[0][j]), mul_(dax[1], m[1][j])),
                      mul_(dax[2], m[2][j]));
    }
    K26Quat qa = quat_from_columns_(aax[0], aax[1], aax[2]);

    /* The lateral offset, built in the plane from the two axes that
     * span it so the fixture's lateral distance is exactly known. */
    K26V3 lat_off = add_(mul_(pax[1], 0.0210), mul_(pax[2], -0.0280));
    double lateral_want = sqrt(0.0210 * 0.0210 + 0.0280 * 0.0280);

    K26AstroCollBody A;
    memset(&A, 0, sizeof A);
    A.vel0 = v3_(0.2810, -0.1730, 0.2380);
    A.vel1 = v3_(0.2814, -0.1727, 0.2384);
    A.omega = v3_(0.00110, 0.00230, -0.00070);
    A.com_offset = v3_(0.050, 0.020, -0.030);
    A.mass = 1.0e4;
    A.orientation = qa;

    K26AstroCollPort ap;
    ASSERT(k26astro_coll_port_basis(v3_(1.0, 0.0, 0.0), v3_(0.0, 1.0, 0.0),
                                    v3_(3.5, 0.0, 0.0), &ap) == 0);

    /* Place the body so its port lands at the chosen offset. */
    K26V3 apos_want = add_(add_(ppos, mul_(pax[0], AXIAL_)), lat_off);
    {
        K26V3 arm = rot_(qa, ap.at);
        K26V3 at_time = sub_(apos_want, arm);
        /* pos0 and pos1 straddle the sample so the interpolation is
         * exercised rather than bypassed. */
        K26V3 drift = mul_(A.vel0, 0.05);
        A.pos0 = sub_(at_time, mul_(drift, TIME_));
        A.pos1 = add_(A.pos0, drift);
    }

    K26AstroCollPortState s;
    ASSERT(k26astro_coll_port_state(&A, &ap, &P, &pp, TIME_, &s)
           == K26ASTRO_COLL_OK);

    /* ---- 2. the residuals, against the fixture's own inputs ------ */
    {
        printf("residuals: the library recovers what the fixture built\n");
        near_("axial", s.axial, AXIAL_, 1e-12);
        near_("lateral", s.lateral, lateral_want, 1e-12);
        near_("pitch/yaw vector sum",
              s.pitchyaw, sqrt(YAW_ * YAW_ + PITCH_ * PITCH_), 1e-12);
        near_("roll", s.roll, ROLL_, 1e-12);
        n_pass++;
    }

    /* ---- 3. the rates ------------------------------------------- */
    {
        printf("rates: resolved on the passive axis, at the same instant\n");
        K26V3 acom = add_(lerp_(A.pos0, A.pos1, TIME_), rot_(qa, A.com_offset));
        K26V3 pcom = add_(lerp_(P.pos0, P.pos1, TIME_),
                          rot_(P.orientation, P.com_offset));
        K26V3 wa = rot_(qa, A.omega);
        K26V3 wp = rot_(P.orientation, P.omega);
        K26V3 va = add_(lerp_(A.vel0, A.vel1, TIME_),
                        cross_(wa, sub_(apos_want, acom)));
        K26V3 vp = add_(lerp_(P.vel0, P.vel1, TIME_),
                        cross_(wp, sub_(ppos, pcom)));
        K26V3 vrel = sub_(va, vp);
        K26V3 wrel = sub_(wa, wp);
        near_("closing rate", s.v_axial, -dot_(vrel, pax[0]), 1e-14);
        near_("lateral rate", s.v_lateral, len_(perp_(vrel, pax[0])), 1e-14);
        near_("pitch/yaw rate", s.v_pitchyaw, len_(perp_(wrel, pax[0])), 1e-16);
        near_("roll rate", s.v_roll, fabs(dot_(wrel, pax[0])), 1e-16);
        K26V3 arm  = sub_(acom, apos_want);
        K26V3 cvel = add_(vrel, cross_(wrel, arm));
        near_("lateral rate at the centre of mass",
              s.v_lateral_cg, len_(perp_(cvel, pax[0])), 1e-16);
        /* The two lateral rates differ on this fixture, so an arm
         * that read one for the other would fail; and neither equals
         * the difference of the two bodies' own velocities, which is
         * the quantity this rate is not. */
        K26V3 own = sub_(lerp_(A.vel0, A.vel1, TIME_),
                         lerp_(P.vel0, P.vel1, TIME_));
        double own_lat = len_(perp_(own, pax[0]));
        printf("  difference of the two bodies' own velocities %.12e\n",
               own_lat);
        ASSERT(fabs(s.v_lateral_cg - own_lat) > 1e-6);
        printf("  port lateral %.12e, centre of mass lateral %.12e\n",
               s.v_lateral, s.v_lateral_cg);
        ASSERT(fabs(s.v_lateral - s.v_lateral_cg) > 1e-6);
        n_pass++;
    }

    /* ---- 4. signs ------------------------------------------------ */
    {
        printf("signs: apart is positive axial, approaching is positive rate\n");
        K26AstroCollBody A2 = A, P2 = P;
        K26AstroCollPort ap2 = ap, pp2 = pp;
        memset(&A2, 0, sizeof A2);
        memset(&P2, 0, sizeof P2);
        A2.orientation = k26m3d_quat_from_axis_angle(v3_(0.0, 1.0, 0.0),
                                                     3.14159265358979323846);
        P2.orientation = k26m3d_quat_identity();
        A2.mass = 1.0e4;
        P2.mass = 3.5e5;
        ASSERT(k26astro_coll_port_basis(v3_(1.0, 0.0, 0.0), v3_(0.0, 1.0, 0.0),
                                        v3_(1.0, 0.0, 0.0), &ap2) == 0);
        ASSERT(k26astro_coll_port_basis(v3_(1.0, 0.0, 0.0), v3_(0.0, 1.0, 0.0),
                                        v3_(2.0, 0.0, 0.0), &pp2) == 0);
        /* Passive port at x = 2; the active body is turned about, so
         * its port at body x = 1 sticks out at world x = pos - 1. */
        A2.pos0 = v3_(2.0 + 1.0 + 0.4, 0.0, 0.0);
        A2.pos1 = A2.pos0;
        A2.vel0 = v3_(-0.07, 0.0, 0.0);
        A2.vel1 = A2.vel0;

        K26AstroCollPortState t;
        ASSERT(k26astro_coll_port_state(&A2, &ap2, &P2, &pp2, 0.0, &t)
               == K26ASTRO_COLL_OK);
        near_("axial while apart", t.axial, 0.4, 1e-14);
        near_("lateral while coaxial", t.lateral, 0.0, 1e-15);
        near_("pitch/yaw while mated in attitude", t.pitchyaw, 0.0, 1e-8);
        near_("roll while mated in attitude", t.roll, 0.0, 1e-8);
        near_("closing rate while approaching", t.v_axial, 0.07, 1e-15);
        near_("lateral rate while purely axial", t.v_lateral, 0.0, 1e-15);
        n_pass++;
    }

    /* ---- 5. the envelope, limit by limit ------------------------- */
    {
        printf("envelope: at each limit accepted, one bit beyond refused\n");
        K26AstroCollEnvelope e;
        e.axial_rate_min = 0.05;
        e.axial_rate_max = 0.10;
        e.lateral_rate   = 0.04;
        e.pitchyaw_rate  = 0.0034906585039886593;   /* 0.20 deg/s */
        e.roll_rate      = 0.0034906585039886593;
        e.lateral        = 0.10;
        e.pitchyaw       = 0.06981317007977318;     /* 4.0 deg */
        e.roll           = 0.06981317007977318;

        /* A state comfortably inside every condition, which each arm
         * then pushes on one axis at a time. */
        K26AstroCollPortState base;
        memset(&base, 0, sizeof base);
        base.v_axial = 0.07;
        base.v_lateral = 0.01;
        base.v_pitchyaw = 0.001;
        base.v_roll = 0.001;
        base.lateral = 0.03;
        base.pitchyaw = 0.02;
        base.roll = 0.02;
        base.v_lateral_cg = 0.01;
        ASSERT(k26astro_coll_port_captured(&base, &e) == 1);
        printf("  the base state captures: OK\n");

        struct { const char *name; size_t off; double lim; int upper; } arm[] = {
            { "closing rate lower bound",
              offsetof(K26AstroCollPortState, v_axial), 0.05, 0 },
            { "closing rate upper bound",
              offsetof(K26AstroCollPortState, v_axial), 0.10, 1 },
            { "lateral rate",
              offsetof(K26AstroCollPortState, v_lateral), 0.04, 1 },
            { "pitch/yaw rate",
              offsetof(K26AstroCollPortState, v_pitchyaw),
              0.0034906585039886593, 1 },
            { "roll rate",
              offsetof(K26AstroCollPortState, v_roll),
              0.0034906585039886593, 1 },
            { "lateral misalignment",
              offsetof(K26AstroCollPortState, lateral), 0.10, 1 },
            { "pitch/yaw misalignment",
              offsetof(K26AstroCollPortState, pitchyaw),
              0.06981317007977318, 1 },
            { "roll misalignment",
              offsetof(K26AstroCollPortState, roll),
              0.06981317007977318, 1 },
            { "lateral rate at the centre of mass",
              offsetof(K26AstroCollPortState, v_lateral_cg), 0.04, 1 },
        };
        for (size_t i = 0; i < sizeof arm / sizeof arm[0]; i++) {
            K26AstroCollPortState at = base, beyond = base;
            double *pa = (double *)((char *)&at + arm[i].off);
            double *pb = (double *)((char *)&beyond + arm[i].off);
            *pa = arm[i].lim;
            *pb = arm[i].upper ? nextafter(arm[i].lim, 2.0)
                               : nextafter(arm[i].lim, -2.0);
            int ok_at = k26astro_coll_port_captured(&at, &e);
            int ok_beyond = k26astro_coll_port_captured(&beyond, &e);
            printf("  %-38s at %.17g -> %d, %.17g -> %d\n",
                   arm[i].name, *pa, ok_at, *pb, ok_beyond);
            ASSERT(ok_at == 1);
            ASSERT(ok_beyond == 0);
        }
        n_pass++;

        /* ---- 6. the derived condition earns its place ------------ */
        printf("the centre-of-mass rate decides a case the port rate does not\n");
        K26AstroCollPortState swung = base;
        swung.v_lateral = 0.02;              /* inside the limit */
        swung.v_pitchyaw = 0.0030;           /* inside the limit */
        swung.v_lateral_cg = 0.055;          /* outside it */
        ASSERT(k26astro_coll_port_captured(&swung, &e) == 0);
        swung.v_lateral_cg = 0.02;
        ASSERT(k26astro_coll_port_captured(&swung, &e) == 1);
        printf("  a contact inside every port limit is refused on the "
               "rate at the centres of mass: OK\n");
        n_pass++;

        /* Every condition is simultaneous: any one failure refuses. */
        printf("the conditions apply simultaneously\n");
        for (size_t i = 0; i < sizeof arm / sizeof arm[0]; i++) {
            K26AstroCollPortState bad = base;
            double *p = (double *)((char *)&bad + arm[i].off);
            *p = arm[i].upper ? arm[i].lim * 2.0 + 1.0 : 0.0;
            ASSERT(k26astro_coll_port_captured(&bad, &e) == 0);
        }
        printf("  each condition alone refuses a state inside all others: OK\n");
        n_pass++;
    }

    /* ---- 7. refusals -------------------------------------------- */
    {
        printf("refusals\n");
        K26AstroCollPortState t;
        ASSERT(k26astro_coll_port_state(NULL, &ap, &P, &pp, 0.0, &t)
               == K26ASTRO_COLL_E_NULL);
        ASSERT(k26astro_coll_port_state(&A, &ap, &P, &pp, 1.5, &t)
               == K26ASTRO_COLL_E_BAD_DT);
        ASSERT(k26astro_coll_port_state(&A, &ap, &P, &pp, -0.0001, &t)
               == K26ASTRO_COLL_E_BAD_DT);
        ASSERT(k26astro_coll_port_captured(NULL, NULL) == 0);
        printf("  null arguments and an out-of-interval fraction: OK\n");
        n_pass++;
    }

    printf("test_coll_port: %d check(s) passed\n", n_pass);
    return 0;
}
