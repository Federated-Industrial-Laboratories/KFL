/* test_coll_pass.c: the broadphase, the tunnelling case the sweep
 * exists for, the selection rule, and the two resolutions.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   The tunnelling arm would be worthless if the fixture were one a
 *   test of the endpoints could also solve, so the same fixture is
 *   run through an endpoint overlap test written here in the gate and
 *   that test is asserted to MISS it. That control is what makes the
 *   arm a measurement of the sweep rather than of contact detection
 *   in general.
 *
 *   The selection arm would be worthless if the two candidate
 *   contacts differed in impact time, since then any rule that
 *   preferred the earlier one would pass. The fixture is built so the
 *   two times are equal to the last bit, which is the only case in
 *   which the tie-break is the thing under test.
 *
 *   The arrest arm would be worthless if it compared the recorded
 *   position against a fresh interpolation, since a wrong
 *   interpolation would then match itself. It compares bitwise
 *   against the same arithmetic the sweep used, and separately
 *   asserts the reported contact time is strictly inside the
 *   interval, so the configuration is not simply an endpoint.
 *
 *   The bounce arm would be worthless if it asserted only that
 *   momentum is conserved, since equal and opposite impulses conserve
 *   it whatever their size. It also asserts the analytic outgoing
 *   velocities of a head-on impact at a declared restitution, which
 *   pins the size.
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
    printf("  %-42s %.15f  want %.15f  err %.3e\n", what, got, want, e);
    ASSERT(e <= tol);
}

static K26AstroCollShape sphere_(double r)
{
    K26AstroCollShape s;
    memset(&s, 0, sizeof s);
    s.kind = K26ASTRO_COLL_SPHERE;
    s.half[0] = r;
    s.axis[0] = k26m3d_v3(1, 0, 0);
    s.axis[1] = k26m3d_v3(0, 1, 0);
    s.axis[2] = k26m3d_v3(0, 0, 1);
    return s;
}

static K26AstroCollShape box_at_(K26V3 c, double hx, double hy, double hz)
{
    K26AstroCollShape s;
    memset(&s, 0, sizeof s);
    s.kind = K26ASTRO_COLL_BOX;
    s.centre = c;
    s.half[0] = hx; s.half[1] = hy; s.half[2] = hz;
    s.axis[0] = k26m3d_v3(1, 0, 0);
    s.axis[1] = k26m3d_v3(0, 1, 0);
    s.axis[2] = k26m3d_v3(0, 0, 1);
    return s;
}

static K26AstroCollBody body_(K26V3 p0, K26V3 p1, const K26AstroCollShape *sh,
                              int n, double bound, double mass)
{
    K26AstroCollBody b;
    memset(&b, 0, sizeof b);
    b.pos0 = p0; b.pos1 = p1;
    b.orientation = k26m3d_v3_len2(k26m3d_v3(0, 0, 0)) == 0.0
                  ? (K26Quat){ 0.0, 0.0, 0.0, 1.0 } : (K26Quat){ 0, 0, 0, 1 };
    b.shapes = sh; b.n_shapes = n;
    b.bound_radius = bound;
    b.mass = mass;
    /* A unit inverse inertia keeps the angular arithmetic legible; the
     * arms that care set their own. */
    for (int i = 0; i < 3; i++) b.inv_inertia.m[i][i] = 1.0;
    return b;
}

/* The test the sweep exists to replace: does the pair overlap at
 * either end of the interval? Written here in the gate, not called
 * from the library, precisely so that the tunnelling arm can show it
 * failing on a case the sweep answers. */
static int endpoint_overlap_(const K26AstroCollBody *a,
                             const K26AstroCollBody *b)
{
    for (int end = 0; end < 2; end++) {
        K26V3 pa = end == 0 ? a->pos0 : a->pos1;
        K26V3 pb = end == 0 ? b->pos0 : b->pos1;
        K26V3 off = { pb.x - pa.x, pb.y - pa.y, pb.z - pa.z };
        for (int i = 0; i < a->n_shapes; i++) {
            for (int j = 0; j < b->n_shapes; j++) {
                K26V3 poff = { off.x + b->shapes[j].centre.x
                                     - a->shapes[i].centre.x,
                               off.y + b->shapes[j].centre.y
                                     - a->shapes[i].centre.y,
                               off.z + b->shapes[j].centre.z
                                     - a->shapes[i].centre.z };
                K26AstroCollHit h;
                if (k26astro_coll_sweep_pair(&a->shapes[i], &b->shapes[j],
                                             poff, k26m3d_v3(0, 0, 0), &h)) {
                    return 1;
                }
            }
        }
    }
    return 0;
}

int main(void)
{
    printf("tunnelling:\n");
    {
        /* A plate 0.2 thick across the path, and a projectile of
         * radius 0.05 that travels 6 in one interval: thirty times
         * the plate's own thickness. Contact is when the projectile's
         * surface reaches the plate's face, a centre separation of
         * 0.1 + 0.05 = 0.15. The projectile starts at x = -3 and
         * travels +6, so its centre is -3 + 6 t and reaches -0.15 at
         * t = 2.85 / 6 = 0.475. */
        K26AstroCollShape plate = box_at_(k26m3d_v3(0, 0, 0), 0.1, 5.0, 5.0);
        K26AstroCollShape shot  = sphere_(0.05);

        K26AstroCollBody bodies[2];
        bodies[0] = body_(k26m3d_v3(0, 0, 0), k26m3d_v3(0, 0, 0),
                          &plate, 1, 7.071774883294858, 0.0);
        bodies[1] = body_(k26m3d_v3(-3.0, 0, 0), k26m3d_v3(3.0, 0, 0),
                          &shot, 1, 0.05, 1.0);
        bodies[1].vel0 = k26m3d_v3(6.0, 0, 0);
        bodies[1].vel1 = k26m3d_v3(6.0, 0, 0);

        /* The control first, so the arm cannot be read as proving
         * something a cheaper test also proves. */
        int endpoints = endpoint_overlap_(&bodies[0], &bodies[1]);
        printf("  an endpoint overlap test finds it: %s\n",
               endpoints ? "yes" : "no");
        ASSERT(endpoints == 0);

        K26AstroCollContact c;
        ASSERT(k26astro_coll_pass(bodies, 2, 1.0, &c) == K26ASTRO_COLL_OK);
        printf("  the sweep finds it: %s\n", c.hit ? "yes" : "no");
        ASSERT(c.hit);
        near_("impact fraction", c.time, 2.85 / 6.0, 1e-15);
        ASSERT(c.body_a == 0 && c.body_b == 1);
        printf("  displacement %.1f against a plate %.1f thick, a factor "
               "of %.0f\n", 6.0, 0.2, 6.0 / 0.2);
        printf("  a body crossing a thin target inside one interval is "
               "detected, and the endpoint test that would replace the "
               "sweep misses it: OK\n");
        n_pass++;

        /* The reported closing speed is the relative speed along the
         * normal, which here is the whole of the projectile's speed
         * because the approach is along the face normal. */
        near_("closing speed", c.speed, 6.0, 1e-12);
        n_pass++;
    }

    printf("the selection rule:\n");
    {
        /* Two primitives on the moving body placed symmetrically, so
         * the two contacts happen at the same instant to the last
         * bit. Only a tie-break decides which is reported, which is
         * the whole point of the fixture: with different times any
         * rule that took the earlier one would pass. */
        K26AstroCollShape plate = box_at_(k26m3d_v3(0, 0, 0), 0.5, 5.0, 5.0);
        K26AstroCollShape pair[2];
        pair[0] = sphere_(0.25); pair[0].centre = k26m3d_v3(0.0,  1.0, 0.0);
        pair[1] = sphere_(0.25); pair[1].centre = k26m3d_v3(0.0, -1.0, 0.0);

        K26AstroCollBody bodies[2];
        bodies[0] = body_(k26m3d_v3(0, 0, 0), k26m3d_v3(0, 0, 0),
                          &plate, 1, 7.09, 0.0);
        bodies[1] = body_(k26m3d_v3(-4.0, 0, 0), k26m3d_v3(4.0, 0, 0),
                          pair, 2, 1.25, 1.0);
        bodies[1].vel0 = k26m3d_v3(8.0, 0, 0);
        bodies[1].vel1 = k26m3d_v3(8.0, 0, 0);

        /* Each sphere reaches the face at the same fraction, checked
         * directly so the tie is a measured fact and not an
         * assumption. */
        K26AstroCollHit h0, h1;
        ASSERT(k26astro_coll_sweep_pair(&plate, &pair[0],
                   k26m3d_v3(-4.0 + 0.0, 1.0, 0.0), k26m3d_v3(8.0, 0, 0), &h0));
        ASSERT(k26astro_coll_sweep_pair(&plate, &pair[1],
                   k26m3d_v3(-4.0 + 0.0, -1.0, 0.0), k26m3d_v3(8.0, 0, 0), &h1));
        printf("  the two candidates are at %.17g and %.17g\n",
               h0.time, h1.time);
        ASSERT(h0.time == h1.time);

        K26AstroCollContact c;
        ASSERT(k26astro_coll_pass(bodies, 2, 1.0, &c) == K26ASTRO_COLL_OK);
        ASSERT(c.hit);
        printf("  the pass reports primitive %d of body %d\n",
               c.shape_b, c.body_b);
        ASSERT(c.shape_b == 0);
        /* And repeating it gives the same answer, which is the
         * property the rule exists for. */
        for (int k = 0; k < 8; k++) {
            K26AstroCollContact again;
            ASSERT(k26astro_coll_pass(bodies, 2, 1.0, &again) ==
                   K26ASTRO_COLL_OK);
            ASSERT(memcmp(&again, &c, sizeof c) == 0);
        }
        printf("  a simultaneous pair is broken by the declared order "
               "and reports the same primitive every run: OK\n");
        n_pass++;
    }

    printf("arrest:\n");
    {
        K26AstroCollShape sa = sphere_(1.0), sb = sphere_(2.0);
        K26AstroCollBody bodies[2];
        bodies[0] = body_(k26m3d_v3(0, 0, 0), k26m3d_v3(0, 0, 0),
                          &sa, 1, 1.0, 1.0);
        bodies[1] = body_(k26m3d_v3(10.0, 0, 0), k26m3d_v3(-4.0, 0, 0),
                          &sb, 1, 2.0, 3.0);
        bodies[1].vel0 = k26m3d_v3(-14.0, 0, 0);
        bodies[1].vel1 = k26m3d_v3(-14.0, 0, 0);

        K26AstroCollContact c;
        ASSERT(k26astro_coll_pass(bodies, 2, 1.0, &c) == K26ASTRO_COLL_OK);
        ASSERT(c.hit);
        near_("impact fraction", c.time, 0.5, 0.0);
        /* Strictly inside the interval, so the configuration below is
         * not simply one of the two endpoints. */
        ASSERT(c.time > 0.0 && c.time < 1.0);

        K26V3 pa, va, pb, vb;
        ASSERT(k26astro_coll_arrest(&bodies[0], &bodies[1], c.time,
                                    &pa, &va, &pb, &vb) == K26ASTRO_COLL_OK);
        /* Bitwise against the same interpolation the sweep used. */
        K26V3 want_b = k26m3d_v3(
            bodies[1].pos0.x + c.time * (bodies[1].pos1.x - bodies[1].pos0.x),
            bodies[1].pos0.y + c.time * (bodies[1].pos1.y - bodies[1].pos0.y),
            bodies[1].pos0.z + c.time * (bodies[1].pos1.z - bodies[1].pos0.z));
        printf("  arrested at x %.17g, the sweep's own interpolation "
               "%.17g\n", pb.x, want_b.x);
        ASSERT(pb.x == want_b.x && pb.y == want_b.y && pb.z == want_b.z);
        /* And the pair really is touching there: centres 3 apart for
         * radii 1 and 2. */
        near_("separation at the arrest configuration", pb.x - pa.x, 3.0,
              1e-12);
        /* The merge leaves one velocity, and it is the common centre
         * of mass velocity: (1*0 + 3*(-14)) / 4 = -10.5. */
        near_("merged velocity", va.x, -10.5, 1e-15);
        ASSERT(vb.x == va.x);
        printf("  arrest places the pair at the sweep's own impact "
               "configuration and removes the relative velocity: OK\n");
        n_pass++;
    }

    printf("bounce:\n");
    {
        /* Head on along the line of centres, so the lever arms are
         * parallel to the normal and the angular terms vanish: the
         * outgoing velocities are the textbook one-dimensional
         * result. Masses 1 and 3, approach speed 4, restitution 0.5:
         * the impulse is (1 + e) u / (1/ma + 1/mb) = 6 / (4/3) = 4.5,
         * so the first leaves at 4 - 4.5 = -0.5 and the second at
         * 4.5 / 3 = 1.5. Momentum before is 4 and after is
         * -0.5 + 4.5 = 4. */
        K26AstroCollShape sa = sphere_(1.0), sb = sphere_(1.0);
        K26AstroCollBody a = body_(k26m3d_v3(-4.0, 0, 0), k26m3d_v3(0.0, 0, 0),
                                   &sa, 1, 1.0, 1.0);
        K26AstroCollBody b = body_(k26m3d_v3(0.0, 0, 0), k26m3d_v3(0.0, 0, 0),
                                   &sb, 1, 1.0, 3.0);
        a.vel0 = a.vel1 = k26m3d_v3(4.0, 0, 0);

        K26AstroCollBody bodies[2] = { a, b };
        K26AstroCollContact c;
        ASSERT(k26astro_coll_pass(bodies, 2, 1.0, &c) == K26ASTRO_COLL_OK);
        ASSERT(c.hit);

        K26V3 pa, va, wa, pb, vb, wb;
        ASSERT(k26astro_coll_bounce(&a, &b, &c, 0.5, &pa, &va, &wa,
                                    &pb, &vb, &wb) == K26ASTRO_COLL_OK);
        near_("first body outgoing velocity",  va.x, -0.5, 1e-12);
        near_("second body outgoing velocity", vb.x,  1.5, 1e-12);
        near_("linear momentum after", 1.0 * va.x + 3.0 * vb.x,
              1.0 * 4.0 + 3.0 * 0.0, 1e-12);
        /* The separation speed over the approach speed is the
         * restitution, which is what the coefficient means. */
        near_("restitution realised", (vb.x - va.x) / 4.0, 0.5, 1e-12);
        printf("  a head-on impact reproduces the analytic outgoing "
               "velocities and conserves linear momentum: OK\n");
        n_pass++;
    }

    printf("bounce, off centre:\n");
    {
        /* A genuinely off-centre contact, which two spheres cannot
         * produce: their contact point is always on the line of
         * centres, so the lever arm is parallel to the normal and the
         * angular term is identically zero. An earlier version of
         * this arm used two spheres and passed on rounding noise,
         * which is an arm asserting a quantity that cannot move under
         * the defect it names. A flat face fixes it: a sphere meets a
         * box's face away from its centre, so the lever arm and the
         * normal are not parallel and the first body must spin.
         *
         * The box has half extent 1 and sits at the origin; the
         * sphere has radius 0.5 and travels along -x at y = 0.7.
         * Contact is at a centre separation of 1.5 along x, so from
         * x = 3 travelling -2.4 the fraction is 1.5 / 2.4 = 0.625,
         * and the contact point is (1, 0.7, 0). */
        K26AstroCollShape bx = box_at_(k26m3d_v3(0, 0, 0), 1.0, 1.0, 1.0);
        K26AstroCollShape sp = sphere_(0.5);
        K26AstroCollBody a = body_(k26m3d_v3(0, 0, 0), k26m3d_v3(0, 0, 0),
                                   &bx, 1, 1.7320508075688772, 2.0);
        K26AstroCollBody b = body_(k26m3d_v3(3.0, 0.7, 0.0),
                                   k26m3d_v3(0.6, 0.7, 0.0),
                                   &sp, 1, 0.5, 1.0);
        b.vel0 = b.vel1 = k26m3d_v3(-2.4, 0, 0);
        for (int i = 0; i < 3; i++) { a.inv_inertia.m[i][i] = 0.5;
                                      b.inv_inertia.m[i][i] = 2.0; }

        K26AstroCollBody bodies[2] = { a, b };
        K26AstroCollContact c;
        ASSERT(k26astro_coll_pass(bodies, 2, 1.0, &c) == K26ASTRO_COLL_OK);
        ASSERT(c.hit);
        near_("impact fraction", c.time, 1.5 / 2.4, 1e-15);
        printf("  contact point (%.9f, %.9f, %.9f), normal "
               "(%.6f, %.6f, %.6f)\n", c.point.x, c.point.y, c.point.z,
               c.normal.x, c.normal.y, c.normal.z);
        near_("contact point x", c.point.x, 1.0, 1e-12);
        near_("contact point y", c.point.y, 0.7, 1e-12);
        /* The lever arm and the normal are not parallel, which is the
         * precondition for the whole arm: their cross product is the
         * lever the angular term acts on, and it is asserted at its
         * analytic size rather than merely non-zero, so the arm
         * cannot pass on rounding noise the way its predecessor did. */
        K26V3 ra0 = { c.point.x, c.point.y, c.point.z };
        K26V3 lever = k26m3d_v3_cross(ra0, c.normal);
        near_("lever arm z", lever.z, -0.7, 1e-12);

        K26V3 pa, va, wa, pb, vb, wb;
        ASSERT(k26astro_coll_bounce(&a, &b, &c, 0.4, &pa, &va, &wa,
                                    &pb, &vb, &wb) == K26ASTRO_COLL_OK);
        printf("  first body angular velocity after "
               "(%.9f, %.9f, %.9f)\n", wa.x, wa.y, wa.z);

        /* The analytic outcome, derived from the geometry rather than
         * read back from the kernel. The lever is 0.7 and the first
         * body's inverse inertia 0.5, so the angular contribution to
         * the effective mass is 0.5 * 0.7 * 0.7 = 0.245; the second
         * body's lever is zero because the contact sits on its own
         * line of centres. The denominator is therefore
         * 1/2 + 1/1 + 0.245, and the impulse is (1 + e) times the
         * approach speed divided by it.
         *
         * These three assertions are what make the arm measure the
         * effective-mass term rather than merely notice a spin: an
         * implementation that dropped the angular part of the
         * denominator conserves both momenta exactly and still spins
         * the body, just by the wrong amount. */
        double denom = 1.0 / 2.0 + 1.0 / 1.0 + 0.5 * 0.7 * 0.7;
        double j     = (1.0 + 0.4) * 2.4 / denom;
        near_("first body velocity after",  va.x, -j / 2.0,      1e-14);
        near_("second body velocity after", vb.x, -2.4 + j,      1e-14);
        near_("first body spin after",      wa.z,  0.5 * 0.7 * j, 1e-14);
        /* And the same numbers computed with the angular part left
         * out differ by a wide margin, so the assertions above are
         * not satisfied by both forms. */
        double j_flat = (1.0 + 0.4) * 2.4 / (1.0 / 2.0 + 1.0 / 1.0);
        printf("  impulse with the effective mass %.12f, without it "
               "%.12f, a ratio of %.6f\n", j, j_flat, j_flat / j);
        ASSERT(j_flat > j * 1.1);

        /* Angular momentum about the world origin, before and after.
         * The two impulses act at one common point, so it is
         * conserved whatever their size; the arm below pins the
         * size. */
        double Ia = 1.0 / 0.5, Ib = 1.0 / 2.0;
        K26V3 la0 = k26m3d_v3_cross(pa, a.vel0);
        K26V3 lb0 = k26m3d_v3_cross(pb, b.vel0);
        K26V3 la1 = k26m3d_v3_cross(pa, va);
        K26V3 lb1 = k26m3d_v3_cross(pb, vb);
        double before = 2.0 * la0.z + 1.0 * lb0.z + Ia * a.omega.z
                      + Ib * b.omega.z;
        double after  = 2.0 * la1.z + 1.0 * lb1.z + Ia * wa.z + Ib * wb.z;
        near_("angular momentum z", after, before, 1e-12);
        near_("linear momentum x", 2.0 * va.x + 1.0 * vb.x,
              2.0 * a.vel0.x + 1.0 * b.vel0.x, 1e-12);
        printf("  an off-centre impulse spins the body and conserves "
               "both momenta: OK\n");
        n_pass++;

        /* The effective mass term is what makes that spin the right
         * size. Recomputed here without it, the way the prior art
         * this capability does not carry computes it, the denominator
         * is smaller, so the impulse would be larger. The two numbers
         * must differ by a real margin, or the term is not being
         * exercised by this fixture at all. */
        double inv_m = 1.0 / 2.0 + 1.0 / 1.0;
        K26V3 rb0 = { c.point.x - pb.x, c.point.y - pb.y,
                      c.point.z - pb.z };
        K26V3 xa = k26m3d_v3_cross(ra0, c.normal);
        K26V3 xb = k26m3d_v3_cross(rb0, c.normal);
        double ang = k26m3d_v3_dot(k26m3d_v3_cross(
                        k26m3d_v3_scale(xa, 0.5), ra0), c.normal)
                   + k26m3d_v3_dot(k26m3d_v3_cross(
                        k26m3d_v3_scale(xb, 2.0), rb0), c.normal);
        printf("  denominator without the angular term %.9f, with it "
               "%.9f\n", inv_m, inv_m + ang);
        /* The lever is 0.7 and the inverse inertia 0.5, so the first
         * body contributes 0.5 * 0.7 * 0.7 = 0.245; the second's
         * lever is zero because the contact is on its own line of
         * centres. */
        near_("the angular contribution", ang, 0.245, 1e-12);
        ASSERT(inv_m + ang > inv_m * 1.1);
        printf("  the effective mass at the contact point is materially "
               "larger than the linear one, so omitting it would "
               "over-impulse: OK\n");
        n_pass++;
    }

    printf("the second primitive's own axes:\n");
    {
        /* A fixture whose separating axis belongs to the second
         * primitive alone. An earlier version of the rotated-bar arm
         * in the kernel gate did not have this property: its
         * separating axis was one the first box also carried, so an
         * implementation that never looked at the second box's axes
         * passed it. Here the first box is a cube, so none of its
         * three axes separates the pair at any point of the approach,
         * and the second box's own normal is the only axis that does.
         *
         * The second box's axes come from a rotation about x by
         * forty-five degrees followed by one about z by thirty, so
         * that no axis of the pair is parallel to any other and no
         * cross product of the two sets reproduces the separating
         * one. A cross product of the first box's axis with any axis
         * of the second is perpendicular to that axis, so it can
         * never be the separating axis itself; the arm therefore
         * measures the direct term and not the cross terms. */
        const double c45 = 0.70710678118654752440;
        const double c30 = 0.86602540378443864676, s30 = 0.5;
        K26AstroCollShape cube = box_at_(k26m3d_v3(0, 0, 0), 1.0, 1.0, 1.0);
        K26AstroCollShape plate = box_at_(k26m3d_v3(0, 0, 0), 2.0, 2.0, 0.1);
        /* Rows of R = Rz(30) Rx(45), read out as the plate's axes. */
        plate.axis[0] = k26m3d_v3(c30, s30, 0.0);
        plate.axis[1] = k26m3d_v3(-s30 * c45, c30 * c45, c45);
        plate.axis[2] = k26m3d_v3( s30 * c45, -c30 * c45, c45);
        /* Orthonormal, asserted rather than assumed: a fixture whose
         * axes were not a frame would make every number below
         * meaningless. */
        for (int i = 0; i < 3; i++) {
            near_("axis length", k26m3d_v3_len(plate.axis[i]), 1.0, 1e-15);
        }
        near_("axis 0 against 1", k26m3d_v3_dot(plate.axis[0], plate.axis[1]),
              0.0, 1e-15);
        near_("axis 0 against 2", k26m3d_v3_dot(plate.axis[0], plate.axis[2]),
              0.0, 1e-15);
        near_("axis 1 against 2", k26m3d_v3_dot(plate.axis[1], plate.axis[2]),
              0.0, 1e-15);

        /* The plate approaches along its own normal from four units
         * out. Contact is when the separation along that normal
         * equals the two extents along it, which is computed here
         * from the declared geometry and not read from the kernel. */
        K26V3 n = plate.axis[2];
        double ext_cube = 1.0 * (n.x < 0 ? -n.x : n.x)
                        + 1.0 * (n.y < 0 ? -n.y : n.y)
                        + 1.0 * (n.z < 0 ? -n.z : n.z);
        double ext_plate = 0.1;
        double d0 = 4.0, travel = 4.0;
        double want_t = (d0 - (ext_cube + ext_plate)) / travel;
        printf("  cube extent along the plate's normal %.15f, contact at "
               "%.15f\n", ext_cube, want_t);

        K26V3 off  = k26m3d_v3(n.x * d0, n.y * d0, n.z * d0);
        K26V3 disp = k26m3d_v3(-n.x * travel, -n.y * travel, -n.z * travel);

        /* None of the cube's own axes separates the pair at that
         * moment, which is what leaves the plate's normal as the only
         * candidate that can produce the answer. */
        K26V3 cube_axes[3] = { k26m3d_v3(1, 0, 0), k26m3d_v3(0, 1, 0),
                               k26m3d_v3(0, 0, 1) };
        for (int i = 0; i < 3; i++) {
            K26V3 ax = cube_axes[i];
            double sep = k26m3d_v3_dot(off, ax) + want_t *
                         k26m3d_v3_dot(disp, ax);
            double ea = 1.0;
            double eb = 2.0 * (k26m3d_v3_dot(plate.axis[0], ax) < 0
                               ? -k26m3d_v3_dot(plate.axis[0], ax)
                               :  k26m3d_v3_dot(plate.axis[0], ax))
                      + 2.0 * (k26m3d_v3_dot(plate.axis[1], ax) < 0
                               ? -k26m3d_v3_dot(plate.axis[1], ax)
                               :  k26m3d_v3_dot(plate.axis[1], ax))
                      + 0.1 * (k26m3d_v3_dot(plate.axis[2], ax) < 0
                               ? -k26m3d_v3_dot(plate.axis[2], ax)
                               :  k26m3d_v3_dot(plate.axis[2], ax));
            double gap = (sep < 0 ? -sep : sep) - (ea + eb);
            printf("  cube axis %d: overlap margin %.9f\n", i, -gap);
            ASSERT(gap < 0.0);
        }

        K26AstroCollHit h;
        ASSERT(k26astro_coll_sweep_pair(&cube, &plate, off, disp, &h));
        near_("impact fraction", h.time, want_t, 1e-14);
        printf("  a pair separated only by the second primitive's own "
               "face normal is answered at the analytic time: OK\n");
        n_pass++;

        /* Stopped just short: the contact needs 0.5567 of the
         * declared travel, so 0.55 of it must not reach. */
        ASSERT(want_t > 0.55);
        ASSERT(!k26astro_coll_sweep_pair(&cube, &plate, off,
                   k26m3d_v3(disp.x * 0.55, disp.y * 0.55, disp.z * 0.55),
                   &h));
        printf("  the same approach stopped short reports none: OK\n");
        n_pass++;
    }

    printf("test_coll_pass: %d check(s) passed\n", n_pass);
    return 0;
}
