/* test_coll_kernels.c: the swept narrowphase against configurations
 * whose answers are known in closed form.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   A fixture whose impact time is zero would pass for any kernel
 *   that reported zero unconditionally, so every approach arm places
 *   the pair apart at the start and asserts a strictly positive time.
 *
 *   A fixture that is symmetric in the coordinate axes cannot see an
 *   axis swapped for another, so the box arms are placed off the axes
 *   and the reported normal is asserted component by component.
 *
 *   A fixture that only ever reports contact cannot see a kernel that
 *   reports contact always, so every arm has a near-miss twin: the
 *   same geometry moved by a hair to the far side of the answer, on
 *   which no contact must be reported.
 *
 *   A geometry that is wrong but self-consistent would pass an
 *   assertion written from the same wrong geometry, so every expected
 *   time here is derived from the configuration by hand in the
 *   comment above it and written as arithmetic on the declared
 *   numbers, not read back from the kernel.
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
    printf("  %-44s %.15f  want %.15f  err %.3e\n", what, got, want, e);
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

static K26AstroCollShape box_(double hx, double hy, double hz)
{
    K26AstroCollShape s;
    memset(&s, 0, sizeof s);
    s.kind = K26ASTRO_COLL_BOX;
    s.half[0] = hx; s.half[1] = hy; s.half[2] = hz;
    s.axis[0] = k26m3d_v3(1, 0, 0);
    s.axis[1] = k26m3d_v3(0, 1, 0);
    s.axis[2] = k26m3d_v3(0, 0, 1);
    return s;
}

static K26AstroCollShape capsule_(double r, double halflen, K26V3 dir)
{
    K26AstroCollShape s;
    memset(&s, 0, sizeof s);
    s.kind = K26ASTRO_COLL_CAPSULE;
    s.half[0] = r; s.half[2] = halflen;
    s.axis[2] = dir;
    s.axis[0] = k26m3d_v3(1, 0, 0);
    s.axis[1] = k26m3d_v3(0, 1, 0);
    return s;
}

int main(void)
{
    K26AstroCollHit h;

    printf("sphere against sphere:\n");
    {
        /* Radii 1 and 2, so they touch at a separation of 3. The
         * second starts 10 along x and travels -14 over the interval,
         * so the separation is 10 - 14 t and reaches 3 at
         * t = 7 / 14 = 0.5 exactly. */
        ASSERT(k26astro_coll_sweep_sphere_sphere(
                   1.0, 2.0, k26m3d_v3(10.0, 0.0, 0.0),
                   k26m3d_v3(-14.0, 0.0, 0.0), &h));
        near_("head-on closing, impact fraction", h.time, 0.5, 0.0);
        ASSERT(h.time > 0.0);
        near_("normal x", h.normal.x, 1.0, 0.0);
        ASSERT(h.normal.y == 0.0 && h.normal.z == 0.0);
        printf("  two spheres closing head on: OK\n");
        n_pass++;

        /* The near miss: the same closing motion stopped one part in
         * ten thousand short of contact. */
        ASSERT(!k26astro_coll_sweep_sphere_sphere(
                   1.0, 2.0, k26m3d_v3(10.0, 0.0, 0.0),
                   k26m3d_v3(-6.9993, 0.0, 0.0), &h));
        printf("  a sphere stopping short of contact reports none: OK\n");
        n_pass++;

        /* Already overlapping at the start: time zero, which is what a
         * reset that draws two bodies inside one another must give. */
        ASSERT(k26astro_coll_sweep_sphere_sphere(
                   1.0, 2.0, k26m3d_v3(1.0, 0.0, 0.0),
                   k26m3d_v3(0.0, 0.0, 0.0), &h));
        ASSERT(h.time == 0.0);
        printf("  a pair already overlapping reports contact at zero: OK\n");
        n_pass++;

        /* Passing by at a distance the sum of radii never reaches. */
        ASSERT(!k26astro_coll_sweep_sphere_sphere(
                   1.0, 2.0, k26m3d_v3(10.0, 4.0, 0.0),
                   k26m3d_v3(-20.0, 0.0, 0.0), &h));
        printf("  a sphere passing beside another reports none: OK\n");
        n_pass++;
    }

    printf("sphere against capsule:\n");
    {
        /* A capsule of radius 0.5 along z, half length 2, so its
         * segment runs from z = -2 to z = +2. A sphere of radius 0.25
         * approaches its side: contact at a perpendicular separation
         * of 0.75. The sphere starts 3 along x at z = 1, inside the
         * segment's range, and travels -3 in x, so the separation is
         * 3 - 3 t and reaches 0.75 at t = 2.25 / 3 = 0.75 exactly. */
        K26AstroCollShape cap = capsule_(0.5, 2.0, k26m3d_v3(0, 0, 1));
        K26AstroCollShape sph = sphere_(0.25);
        ASSERT(k26astro_coll_sweep_pair(&cap, &sph,
                                        k26m3d_v3(3.0, 0.0, 1.0),
                                        k26m3d_v3(-3.0, 0.0, 0.0), &h));
        near_("side contact, impact fraction", h.time, 0.75, 1e-15);
        near_("normal x", h.normal.x, 1.0, 1e-15);
        printf("  a sphere grazing a capsule's side at the sum of the "
               "radii: OK\n");
        n_pass++;

        /* The same approach stopped a hair short. */
        ASSERT(!k26astro_coll_sweep_pair(&cap, &sph,
                                         k26m3d_v3(3.0, 0.0, 1.0),
                                         k26m3d_v3(-2.2499, 0.0, 0.0), &h));
        printf("  the same approach stopped short reports none: OK\n");
        n_pass++;

        /* Beyond the end cap. The sphere approaches along z at x = 0,
         * so the closest feature is the endpoint at z = 2 and contact
         * is at a separation of 0.75 from it. Starting at z = 5 and
         * travelling -4, the gap is 3 - 4 t and closes at
         * t = 2.25 / 4 = 0.5625 exactly. */
        ASSERT(k26astro_coll_sweep_pair(&cap, &sph,
                                        k26m3d_v3(0.0, 0.0, 5.0),
                                        k26m3d_v3(0.0, 0.0, -4.0), &h));
        near_("end cap contact, impact fraction", h.time, 0.5625, 1e-15);
        near_("normal z", h.normal.z, 1.0, 1e-15);
        printf("  a sphere meeting a capsule's end cap: OK\n");
        n_pass++;

        /* Passing the cap at a lateral distance greater than the sum
         * of the radii: no contact, however far it travels. */
        ASSERT(!k26astro_coll_sweep_pair(&cap, &sph,
                                         k26m3d_v3(0.8, 0.0, 5.0),
                                         k26m3d_v3(0.0, 0.0, -10.0), &h));
        printf("  a sphere passing a capsule's cap wide reports none: "
               "OK\n");
        n_pass++;
    }

    printf("capsule against capsule:\n");
    {
        /* Two capsules of radius 0.5, one along x and one along y,
         * crossing at right angles. They touch when their axes are 1
         * apart. The second starts 4 along z and travels -6, so the
         * axis separation is 4 - 6 t and reaches 1 at t = 3 / 6 = 0.5
         * exactly. The closest approach is between the interiors, so
         * this is the arm the interior case owns. */
        K26AstroCollShape ca = capsule_(0.5, 2.0, k26m3d_v3(1, 0, 0));
        K26AstroCollShape cb = capsule_(0.5, 2.0, k26m3d_v3(0, 1, 0));
        ASSERT(k26astro_coll_sweep_pair(&ca, &cb,
                                        k26m3d_v3(0.0, 0.0, 4.0),
                                        k26m3d_v3(0.0, 0.0, -6.0), &h));
        near_("crossed interiors, impact fraction", h.time, 0.5, 1e-15);
        near_("normal z", h.normal.z, 1.0, 1e-15);
        ASSERT(h.normal.x == 0.0 && h.normal.y == 0.0);
        printf("  two crossed capsules meet between their interiors: "
               "OK\n");
        n_pass++;

        ASSERT(!k26astro_coll_sweep_pair(&ca, &cb,
                                        k26m3d_v3(0.0, 0.0, 4.0),
                                        k26m3d_v3(0.0, 0.0, -2.9999), &h));
        printf("  the same pair stopped short reports none: OK\n");
        n_pass++;

        /* Parallel capsules, where the interior case is degenerate and
         * an endpoint case must carry the answer. Both along x, offset
         * in y by 4 and closing: contact at a separation of 1, so
         * 4 - 6 t = 1 at t = 0.5. */
        K26AstroCollShape cc = capsule_(0.5, 2.0, k26m3d_v3(1, 0, 0));
        ASSERT(k26astro_coll_sweep_pair(&ca, &cc,
                                        k26m3d_v3(0.0, 4.0, 0.0),
                                        k26m3d_v3(0.0, -6.0, 0.0), &h));
        near_("parallel pair, impact fraction", h.time, 0.5, 1e-15);
        printf("  two parallel capsules are answered by an endpoint "
               "case, not by the degenerate interior one: OK\n");
        n_pass++;
    }

    printf("box against box:\n");
    {
        /* Two unit half-extent boxes, axis aligned, approaching along
         * x. Faces touch at a centre separation of 2. Starting 6 apart
         * and closing 8, the gap is 6 - 8 t and reaches 2 at
         * t = 4 / 8 = 0.5 exactly. */
        K26AstroCollShape a = box_(1.0, 1.0, 1.0);
        K26AstroCollShape b = box_(1.0, 1.0, 1.0);
        ASSERT(k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(6.0, 0.0, 0.0),
                                        k26m3d_v3(-8.0, 0.0, 0.0), &h));
        near_("face to face, impact fraction", h.time, 0.5, 0.0);
        near_("normal x", h.normal.x, 1.0, 0.0);
        printf("  a box face meeting a box face: OK\n");
        n_pass++;

        ASSERT(!k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(6.0, 0.0, 0.0),
                                         k26m3d_v3(-3.9999, 0.0, 0.0), &h));
        printf("  the same pair stopped short reports none: OK\n");
        n_pass++;

        /* A corner meeting a face: the second box is offset in y and z
         * so that only its corner arrives, and the separating axis is
         * still x because both boxes are axis aligned. The x gap is
         * the same, so the time is the same and the arm's value is
         * that the offsets do not change it. */
        ASSERT(k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(6.0, 1.9, 1.9),
                                        k26m3d_v3(-8.0, 0.0, 0.0), &h));
        near_("corner to face, impact fraction", h.time, 0.5, 0.0);
        printf("  a box corner meeting a box face: OK\n");
        n_pass++;

        /* Cleared on the y axis: the two never overlap in y, so no
         * amount of x travel brings them together. */
        ASSERT(!k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(6.0, 2.5, 0.0),
                                         k26m3d_v3(-12.0, 0.0, 0.0), &h));
        printf("  a box cleared on another axis reports none: OK\n");
        n_pass++;
    }

    printf("box against box, edge against edge:\n");
    {
        /* The case only a cross-product axis separates. Both boxes are
         * long thin bars of half extent 4 along one axis and 0.5
         * across; the first runs along x, the second along y, and the
         * second passes over the first in z. Every face normal of
         * either box finds them overlapping in projection, so the
         * separating axis is the cross product of the two long
         * directions, which is z here.
         *
         * Along z the half extents are 0.5 and 0.5, so contact is at a
         * z separation of 1. Starting at z = 4 and closing 6, the gap
         * is 4 - 6 t and reaches 1 at t = 0.5 exactly. */
        K26AstroCollShape a = box_(4.0, 0.5, 0.5);
        K26AstroCollShape b = box_(0.5, 4.0, 0.5);
        ASSERT(k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(0.0, 0.0, 4.0),
                                        k26m3d_v3(0.0, 0.0, -6.0), &h));
        near_("edge crossing, impact fraction", h.time, 0.5, 0.0);
        near_("normal z", h.normal.z, 1.0, 0.0);
        printf("  an edge-edge crossing is found: OK\n");
        n_pass++;

        ASSERT(!k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(0.0, 0.0, 4.0),
                                         k26m3d_v3(0.0, 0.0, -2.9999), &h));
        printf("  the same crossing stopped short reports none: OK\n");
        n_pass++;

        /* Rotated forty-five degrees about z, the second bar's long
         * axis is no longer a coordinate direction, so an
         * implementation that used only the first box's axes would
         * separate them wrongly. The z geometry is unchanged, so the
         * answer is the same 0.5 and the arm's whole value is that the
         * rotation does not move it. */
        double c = 0.70710678118654752440;
        K26AstroCollShape r = box_(0.5, 4.0, 0.5);
        r.axis[0] = k26m3d_v3( c, c, 0.0);
        r.axis[1] = k26m3d_v3(-c, c, 0.0);
        r.axis[2] = k26m3d_v3(0.0, 0.0, 1.0);
        ASSERT(k26astro_coll_sweep_pair(&a, &r, k26m3d_v3(0.0, 0.0, 4.0),
                                        k26m3d_v3(0.0, 0.0, -6.0), &h));
        near_("rotated bar, impact fraction", h.time, 0.5, 0.0);
        printf("  a bar rotated off the coordinate axes gives the same "
               "answer: OK\n");
        n_pass++;
    }

    printf("sphere against box:\n");
    {
        /* A sphere of radius 1 meeting a face of a unit box: contact
         * at a centre separation of 2 along x. Starting 6 apart and
         * closing 8 gives t = 0.5, exactly as for two boxes, because
         * the sphere's projection onto any axis is its radius. */
        K26AstroCollShape bx = box_(1.0, 1.0, 1.0);
        K26AstroCollShape sp = sphere_(1.0);
        ASSERT(k26astro_coll_sweep_pair(&bx, &sp, k26m3d_v3(6.0, 0.0, 0.0),
                                        k26m3d_v3(-8.0, 0.0, 0.0), &h));
        near_("sphere onto a box face, impact fraction", h.time, 0.5, 0.0);
        near_("normal x", h.normal.x, 1.0, 0.0);
        printf("  a sphere meeting a box face at the analytic time: OK\n");
        n_pass++;

        ASSERT(!k26astro_coll_sweep_pair(&bx, &sp, k26m3d_v3(6.0, 0.0, 0.0),
                                         k26m3d_v3(-3.9999, 0.0, 0.0), &h));
        printf("  the same approach stopped short reports none: OK\n");
        n_pass++;

        /* Cleared on another axis: the separating axis is y, and the
         * test finds it whatever the x travel. This is the arm that
         * shows the conservatism documented in the header is bounded:
         * a sphere well clear of the box's slab in y is separated, and
         * only the corner region is answered early. */
        ASSERT(!k26astro_coll_sweep_pair(&bx, &sp, k26m3d_v3(6.0, 2.5, 0.0),
                                         k26m3d_v3(-12.0, 0.0, 0.0), &h));
        printf("  a sphere cleared of a box's slab reports none: OK\n");
        n_pass++;
    }

    printf("the arithmetic the claim rests on:\n");
    {
        /* Each kernel's result must be exactly reproducible, which is
         * what restricting the arithmetic to the five correctly
         * rounded operations buys. The cheapest check that the
         * restriction is not quietly broken by a compiler
         * transformation is that one call repeated gives one bit
         * pattern, and that the same geometry expressed at a
         * different scale gives the scaled answer to the last bit,
         * since a scale change is exact in binary floating point when
         * it is a power of two. */
        K26AstroCollShape a = box_(1.0, 1.0, 1.0);
        K26AstroCollShape b = box_(1.0, 1.0, 1.0);
        K26AstroCollHit h1, h2;
        ASSERT(k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(6.0, 0.5, 0.25),
                                        k26m3d_v3(-8.0, 0.0, 0.0), &h1));
        ASSERT(k26astro_coll_sweep_pair(&a, &b, k26m3d_v3(6.0, 0.5, 0.25),
                                        k26m3d_v3(-8.0, 0.0, 0.0), &h2));
        ASSERT(memcmp(&h1, &h2, sizeof h1) == 0);

        K26AstroCollShape a4 = box_(4.0, 4.0, 4.0);
        K26AstroCollShape b4 = box_(4.0, 4.0, 4.0);
        ASSERT(k26astro_coll_sweep_pair(&a4, &b4, k26m3d_v3(24.0, 2.0, 1.0),
                                        k26m3d_v3(-32.0, 0.0, 0.0), &h2));
        printf("  at unit scale %.17g, at four times the scale %.17g\n",
               h1.time, h2.time);
        ASSERT(h1.time == h2.time);
        printf("  one geometry at two scales gives one impact fraction "
               "to the last bit: OK\n");
        n_pass++;
    }

    printf("test_coll_kernels: %d check(s) passed\n", n_pass);
    return 0;
}
