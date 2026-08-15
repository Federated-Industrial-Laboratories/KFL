/* test_prox_frame.c: the chief's local-vertical local-horizontal
 * frame and the target's state resolved onto it.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   An orthonormality arm proves nothing about which way the axes
 *   point: a frame with the radial and along-track axes swapped is
 *   just as orthonormal. Every axis is therefore also compared
 *   against a frame written out by hand in this file from the same
 *   numbers, so the identity of each axis is pinned as well as the
 *   set.
 *
 *   A relative-velocity arm over a fixture whose two craft have the
 *   same inertial velocity cannot tell the rotating-frame rate from
 *   the inertial difference, because the two coincide there. The
 *   station-keeping arm uses a pair on one circular orbit separated in
 *   phase, where the two are equal and opposite: the arm prints the
 *   inertial difference, requires it to be well away from zero, and
 *   requires the published rate to be zero, so an implementation that
 *   omitted the frame's rotation fails by the whole of that
 *   difference. At thirty metres of separation on this orbit the
 *   difference is 0.0323 m/s, and the arm's floor is 0.03.
 *
 *   An arm run near the coordinate origin cannot see whether the
 *   separation was taken through the exact sector-aware subtraction or
 *   through flattened coordinates, since both agree there. Distance
 *   alone is not enough either, and the reason is arithmetic rather
 *   than magnitude: a sector boundary is a power of two, the spacing
 *   of binary64 coordinates there is a power of two, and a separation
 *   of one or two metres is a whole number of spacings, so a flattened
 *   difference reproduces it exactly however far out the pair sits.
 *   The first version of the far-field arm did exactly that, fourteen
 *   sectors out with a one-metre separation, and its control
 *   reproduced the answer to every digit: the arm measured nothing.
 *   The shipped arm puts the pair eighty-seven sectors out, at about
 *   Pluto's distance, which is the case the position type's own header
 *   argues from, and separates it by a third, two sevenths and a
 *   eleventh of a metre, none of which can land on the grid. It then
 *   requires the flattened control to be visibly wrong before it
 *   credits the exact path: the control misses by 3.3e-4 m and the
 *   exact path holds the value to 3.1e-10 m.
 */
#include "k26astro_prox/prox.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

static void near_(const char *what, double got, double want, double tol)
{
    double e = got - want;
    if (e < 0.0) e = -e;
    printf("  %-46s %+.12e  want %+.12e  err %.3e\n", what, got, want, e);
    ASSERT(e <= tol);
}

#define MU_EARTH 3.986004418e14

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* ---- 1. Orthonormal, and each axis the one the design names --- */
    {
        printf("gate 1: the frame's axes\n");
        /* A chief on a circular orbit in the x-y plane, at the point
         * where it sits on the +x axis moving towards +y. The frame
         * is then exactly (+x, +y, +z) and can be written down. */
        double r0 = 7.0e6;
        double v0 = sqrt(MU_EARTH / r0);
        K26AstroPos centre = k26astro_pos_from_m(0.0, 0.0, 0.0);
        K26AstroPos chief  = k26astro_pos_from_m(r0, 0.0, 0.0);
        K26AstroProxFrame f;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0),
                                   &chief, k26m3d_v3(0.0, v0, 0.0),
                                   &f) == K26ASTRO_PROX_OK);

        near_("e1 . e1", k26m3d_v3_dot(f.e1, f.e1), 1.0, 1e-15);
        near_("e2 . e2", k26m3d_v3_dot(f.e2, f.e2), 1.0, 1e-15);
        near_("e3 . e3", k26m3d_v3_dot(f.e3, f.e3), 1.0, 1e-15);
        near_("e1 . e2", k26m3d_v3_dot(f.e1, f.e2), 0.0, 1e-15);
        near_("e1 . e3", k26m3d_v3_dot(f.e1, f.e3), 0.0, 1e-15);
        near_("e2 . e3", k26m3d_v3_dot(f.e2, f.e3), 0.0, 1e-15);

        /* Right-handed: e1 cross e2 is e3, not its negative. */
        K26V3 cr = k26m3d_v3_cross(f.e1, f.e2);
        near_("(e1 x e2) . e3", k26m3d_v3_dot(cr, f.e3), 1.0, 1e-15);

        /* Identity, against the frame written out by hand. */
        near_("e1.x (radial, outward)",   f.e1.x, 1.0, 1e-15);
        near_("e1.y",                     f.e1.y, 0.0, 1e-15);
        near_("e2.y (along the motion)",  f.e2.y, 1.0, 1e-15);
        near_("e2.x",                     f.e2.x, 0.0, 1e-15);
        near_("e3.z (angular momentum)",  f.e3.z, 1.0, 1e-15);
        near_("radius",                   f.radius, r0, 1e-9);
        near_("|omega| (mean motion)",
              k26m3d_v3_len(f.omega), sqrt(MU_EARTH / (r0 * r0 * r0)),
              1e-18);
        n_pass++;
        printf("gate 1: orthonormal, right-handed, each axis the named"
               " one: OK\n");
    }

    /* ---- 2. The axes follow the chief round the orbit ------------- */
    {
        printf("gate 2: the frame at a quarter turn\n");
        /* The same orbit a quarter turn on: the chief sits on +y
         * moving towards -x, so the radial axis is +y and the
         * direction of motion is -x. An implementation that built the
         * frame from the world axes rather than from the state would
         * pass gate 1 and fail here. */
        double r0 = 7.0e6;
        double v0 = sqrt(MU_EARTH / r0);
        K26AstroPos centre = k26astro_pos_from_m(0.0, 0.0, 0.0);
        K26AstroPos chief  = k26astro_pos_from_m(0.0, r0, 0.0);
        K26AstroProxFrame f;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0),
                                   &chief, k26m3d_v3(-v0, 0.0, 0.0),
                                   &f) == K26ASTRO_PROX_OK);
        near_("e1.y (radial)",            f.e1.y, 1.0, 1e-15);
        near_("e2.x (along the motion)",  f.e2.x, -1.0, 1e-15);
        near_("e3.z (angular momentum)",  f.e3.z, 1.0, 1e-15);
        n_pass++;
        printf("gate 2: the frame is the chief's, not the world's: OK\n");
    }

    /* ---- 3. A station-keeping pair reads zero relative velocity --- */
    {
        printf("gate 3: two craft holding station on one orbit\n");
        /* Both on the same circular orbit, separated by a small angle,
         * so the relative motion is zero by construction while the
         * inertial velocities differ in direction. */
        double r0 = 7.0e6;
        double v0 = sqrt(MU_EARTH / r0);
        double dth = 30.0 / r0;             /* about 30 m along-track */

        K26AstroPos centre = k26astro_pos_from_m(0.0, 0.0, 0.0);
        K26AstroPos chief  = k26astro_pos_from_m(r0, 0.0, 0.0);
        K26V3       vchief = k26m3d_v3(0.0, v0, 0.0);
        K26AstroPos target = k26astro_pos_from_m(r0 * cos(dth),
                                                 r0 * sin(dth), 0.0);
        K26V3       vtarget = k26m3d_v3(-v0 * sin(dth), v0 * cos(dth), 0.0);

        K26AstroProxFrame f;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0),
                                   &chief, vchief, &f) == K26ASTRO_PROX_OK);
        K26AstroProxRel rel;
        ASSERT(k26astro_prox_relative(&f, &chief, vchief, &target, vtarget,
                                      &rel) == K26ASTRO_PROX_OK);

        /* The inertial difference this arm is telling apart from the
         * rotating-frame rate. It is not small. */
        double inertial = k26m3d_v3_len(k26m3d_v3_sub(vtarget, vchief));
        printf("  inertial velocity difference                   "
               "%.6f m/s\n", inertial);
        ASSERT(inertial > 0.03);

        /* Along-track, and at this separation the chord's radial sag
         * is r0 (1 - cos dth), about 6.4e-5 m. */
        near_("relative r_x (radial)", rel.r.x, r0 * (cos(dth) - 1.0), 1e-9);
        near_("relative r_y (along-track)", rel.r.y, r0 * sin(dth), 1e-8);
        near_("relative r_z (cross-track)", rel.r.z, 0.0, 1e-9);
        near_("relative v_x", rel.v.x, 0.0, 1e-12);
        near_("relative v_y", rel.v.y, 0.0, 1e-12);
        near_("relative v_z", rel.v.z, 0.0, 1e-12);
        n_pass++;
        printf("gate 3: the frame's own rotation is removed, so a"
               " station-keeping pair reads zero: OK\n");
    }

    /* ---- 4. A radial and a cross-track offset, signed ------------- */
    {
        printf("gate 4: the sign and identity of each component\n");
        double r0 = 7.0e6;
        double v0 = sqrt(MU_EARTH / r0);
        K26AstroPos centre = k26astro_pos_from_m(0.0, 0.0, 0.0);
        K26AstroPos chief  = k26astro_pos_from_m(r0, 0.0, 0.0);
        K26V3       vchief = k26m3d_v3(0.0, v0, 0.0);
        /* 40 m further out, 25 m ahead, 10 m north of the plane. */
        K26AstroPos target = k26astro_pos_from_m(r0 + 40.0, 25.0, 10.0);
        K26AstroProxFrame f;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0),
                                   &chief, vchief, &f) == K26ASTRO_PROX_OK);
        K26AstroProxRel rel;
        ASSERT(k26astro_prox_relative(&f, &chief, vchief, &target, vchief,
                                      &rel) == K26ASTRO_PROX_OK);
        near_("r_x, further out is positive",  rel.r.x, 40.0, 1e-9);
        near_("r_y, ahead is positive",        rel.r.y, 25.0, 1e-9);
        near_("r_z, out of plane is positive", rel.r.z, 10.0, 1e-9);
        /* Equal inertial velocities, so the rotating-frame rate is
         * exactly minus omega cross rho. */
        double n = sqrt(MU_EARTH / (r0 * r0 * r0));
        near_("v_x = +n r_y", rel.v.x, n * 25.0, 1e-12);
        near_("v_y = -n r_x", rel.v.y, -n * 40.0, 1e-12);
        near_("v_z",          rel.v.z, 0.0, 1e-15);
        n_pass++;
        printf("gate 4: components carry the named directions and"
               " signs: OK\n");
    }

    /* ---- 5. Far from the origin, where flattening loses it -------- */
    {
        printf("gate 5: the same pair at Pluto's distance\n");
        /* One sector edge is 2^36 m; eighty-seven of them is 5.98e12
         * m, about Pluto's distance, which is the case the position
         * type's own header argues from. A binary64 metre-from-origin
         * coordinate there is spaced about 0.98 mm apart, so a
         * difference of two of them cannot carry a separation finer
         * than that. The sector-aware subtraction carries it exactly.
         *
         * The separation components are deliberately not powers of
         * two. A first version of this arm used one and two metres and
         * the flattened control reproduced them exactly, because at a
         * sector boundary the coordinate spacing is itself a power of
         * two and a separation of one metre is a whole number of
         * spacings: the arm was measuring nothing. Thirds, sevenths
         * and elevenths of a metre cannot land on the grid. */
        double edge = K26ASTRO_SECTOR_EDGE_M;
        double r0   = 7.0e6;
        double v0   = sqrt(MU_EARTH / r0);
        double dx   = 1.0 / 3.0, dy = 2.0 / 7.0, dz = 1.0 / 11.0;

        K26AstroPos centre = k26astro_pos_zero();
        centre.sx = 87; centre.lx = 0.0;
        K26AstroPos chief = centre;
        chief.lx += r0;
        K26AstroPos target = chief;
        target.lx += dx;
        target.ly += dy;
        target.lz += dz;
        k26astro_pos_normalise(&centre);
        k26astro_pos_normalise(&chief);
        k26astro_pos_normalise(&target);

        double dist = 87.0 * edge;
        printf("  centre at sector %ld, %.4e m from the origin,"
               " coordinate spacing there %.3e m\n",
               (long)centre.sx, dist, nextafter(dist, 2.0 * dist) - dist);

        K26AstroProxFrame f;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0),
                                   &chief, k26m3d_v3(0.0, v0, 0.0),
                                   &f) == K26ASTRO_PROX_OK);
        K26AstroProxRel rel;
        ASSERT(k26astro_prox_relative(&f, &chief, k26m3d_v3(0.0, v0, 0.0),
                                      &target, k26m3d_v3(0.0, v0, 0.0),
                                      &rel) == K26ASTRO_PROX_OK);

        /* The control: the same difference taken through flattened
         * coordinates, which is the mistake this arm exists to rule
         * out. It must be visibly wrong, or the arm proves nothing. */
        K26V3 fa = k26astro_pos_to_m_approx(&target);
        K26V3 fb = k26astro_pos_to_m_approx(&chief);
        double flat_x = fa.x - fb.x;
        double flat_e = flat_x - dx;
        if (flat_e < 0.0) flat_e = -flat_e;
        printf("  flattened control, radial component            "
               "%+.9f m  want %+.9f  err %.3e\n", flat_x, dx, flat_e);
        ASSERT(flat_e > 1.0e-5);

        near_("r_x through the sector-aware subtraction", rel.r.x, dx,
              1e-9);
        near_("r_y through the sector-aware subtraction", rel.r.y, dy,
              1e-9);
        near_("r_z through the sector-aware subtraction", rel.r.z, dz,
              1e-9);
        n_pass++;
        printf("gate 5: the separation survives where a flattened"
               " difference loses it: OK\n");
    }

    /* ---- 6. What has no frame is refused ------------------------- */
    {
        printf("gate 6: degenerate states\n");
        K26AstroPos centre = k26astro_pos_from_m(0.0, 0.0, 0.0);
        K26AstroPos same   = k26astro_pos_from_m(0.0, 0.0, 0.0);
        K26AstroPos out    = k26astro_pos_from_m(7.0e6, 0.0, 0.0);
        K26AstroProxFrame f;

        ASSERT(k26astro_prox_frame(NULL, k26m3d_v3(0, 0, 0), &out,
                                   k26m3d_v3(0, 7500, 0), &f)
               == K26ASTRO_PROX_E_NULL);
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0), &out,
                                   k26m3d_v3(0, 7500, 0), NULL)
               == K26ASTRO_PROX_E_NULL);
        /* Coincident centres: no radial direction. */
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0), &same,
                                   k26m3d_v3(0, 7500, 0), &f)
               == K26ASTRO_PROX_E_DEGENERATE);
        /* Purely radial motion: no direction of motion in the sense
         * the frame means, and no angular momentum to orient by. */
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0), &out,
                                   k26m3d_v3(500.0, 0.0, 0.0), &f)
               == K26ASTRO_PROX_E_DEGENERATE);
        printf("  null, coincident centres, and purely radial motion"
               " each refused\n");

        /* A broken input and a state that names no frame are different
         * conditions and carry different statuses, because a caller
         * can act on the difference. */
        double nan_ = 0.0 / 0.0, inf_ = 1.0 / 0.0;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0), &out,
                                   k26m3d_v3(0.0, nan_, 0.0), &f)
               == K26ASTRO_PROX_E_RANGE);
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(inf_, 0, 0), &out,
                                   k26m3d_v3(0, 7500, 0), &f)
               == K26ASTRO_PROX_E_RANGE);
        K26AstroProxFrame good;
        ASSERT(k26astro_prox_frame(&centre, k26m3d_v3(0, 0, 0), &out,
                                   k26m3d_v3(0, 7500, 0), &good)
               == K26ASTRO_PROX_OK);
        K26AstroProxRel rel;
        ASSERT(k26astro_prox_relative(&good, &out, k26m3d_v3(0, nan_, 0),
                                      &centre, k26m3d_v3(0, 0, 0), &rel)
               == K26ASTRO_PROX_E_RANGE);
        ASSERT(k26astro_prox_relative(&good, &out, k26m3d_v3(0, 7500, 0),
                                      &centre, k26m3d_v3(inf_, 0, 0), &rel)
               == K26ASTRO_PROX_E_RANGE);
        printf("  a velocity that is not a finite number is refused as"
               " out of range, not as a degenerate geometry\n");
        n_pass++;
        printf("gate 6: a state with no frame is refused, not"
               " approximated, and a broken input is told apart from"
               " one: OK\n");
    }

    printf("test_prox_frame: %d gates passed\n", n_pass);
    return 0;
}
