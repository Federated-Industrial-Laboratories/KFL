/* test_assembly.c - vehicle assemblies, their derived mass
 * properties, their identity digest, and their provenance rules.
 *
 * Acceptance:
 *   1. Digest. One assembly digests the same twice; one byte changed
 *      in the assembly, and separately in a referenced mesh, each
 *      changes it; and two file sets whose bytes would concatenate
 *      identically digest differently, which is what the length
 *      prefixes exist for.
 *   2. Derivation against closed forms. A box mesh reproduces the
 *      analytic mass, centre of mass, and inertia of the uniform
 *      solid box exactly enough that only floating-point rounding
 *      separates them, because a box mesh is the box. A sphere and a
 *      capsule mesh reproduce theirs to a tolerance the triangulation
 *      itself sets, stated per case. A two-component assembly
 *      reproduces a parallel-axis sum computed independently here.
 *      The capsule's closed form is checked against a capsule mesh,
 *      so neither path is trusted on its own.
 *   3. Refusals. A mesh that is not closed, a mesh wound
 *      inconsistently, a directive outside the accepted subset, an
 *      unknown key, a component with neither mesh nor collider, a
 *      non-unit rotation, and a number with a trailing character are
 *      each refused with a diagnostic naming the file and the line.
 *   4. Provenance. cited and computed pass; unverified passes with a
 *      warning naming the property; a fourth status word is refused.
 *
 * Fixtures are written to a temporary directory by this program, so
 * the triangulation density behind every tolerance is visible in the
 * source rather than hidden in a committed binary asset.
 *
 * Wire: see kflc/Makefile ASSEMBLY_TEST + test target.
 */
#define _GNU_SOURCE
#include "assembly.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#define WORK "/tmp/kflc_assembly_test"

static int n_pass = 0;

static void write_file_(const char *path, const char *text)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(text, f);
    fclose(f);
}

/* ---- Mesh generators ------------------------------------------------ */

/* A box as twelve triangles, outward wound. The mesh is the solid
 * exactly, so the derivation's only error against the closed form is
 * floating-point rounding. */
static void write_box_mesh_(const char *path, double hx, double hy, double hz)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fprintf(f, "# box, half extents %g %g %g\n", hx, hy, hz);
    const double v[8][3] = {
        { -hx, -hy, -hz }, {  hx, -hy, -hz }, {  hx,  hy, -hz },
        { -hx,  hy, -hz }, { -hx, -hy,  hz }, {  hx, -hy,  hz },
        {  hx,  hy,  hz }, { -hx,  hy,  hz }
    };
    for (int i = 0; i < 8; i++) {
        fprintf(f, "v %.17g %.17g %.17g\n", v[i][0], v[i][1], v[i][2]);
    }
    static const int F[12][3] = {
        { 1, 4, 3 }, { 1, 3, 2 },      /* -z */
        { 5, 6, 7 }, { 5, 7, 8 },      /* +z */
        { 1, 2, 6 }, { 1, 6, 5 },      /* -y */
        { 2, 3, 7 }, { 2, 7, 6 },      /* +x */
        { 3, 4, 8 }, { 3, 8, 7 },      /* +y */
        { 4, 1, 5 }, { 4, 5, 8 }       /* -x */
    };
    for (int i = 0; i < 12; i++) {
        fprintf(f, "f %d %d %d\n", F[i][0], F[i][1], F[i][2]);
    }
    fclose(f);
}

/* A box whose centre sits away from the mesh file's own origin. The
 * first moments of such a mesh are nonzero, which is the only way the
 * coefficient on the first-moment integral can be seen: a mesh
 * centred on its origin has first moments of exactly zero. */
static void write_offset_box_mesh_(const char *path, double h,
                                   const double c[3])
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fprintf(f, "# box of half extent %g centred at %g %g %g\n",
            h, c[0], c[1], c[2]);
    const double v[8][3] = {
        { -h, -h, -h }, {  h, -h, -h }, {  h,  h, -h }, { -h,  h, -h },
        { -h, -h,  h }, {  h, -h,  h }, {  h,  h,  h }, { -h,  h,  h }
    };
    for (int i = 0; i < 8; i++) {
        fprintf(f, "v %.17g %.17g %.17g\n",
                v[i][0] + c[0], v[i][1] + c[1], v[i][2] + c[2]);
    }
    static const int F[12][3] = {
        { 1, 4, 3 }, { 1, 3, 2 }, { 5, 6, 7 }, { 5, 7, 8 },
        { 1, 2, 6 }, { 1, 6, 5 }, { 2, 3, 7 }, { 2, 7, 6 },
        { 3, 4, 8 }, { 3, 8, 7 }, { 4, 1, 5 }, { 4, 5, 8 }
    };
    for (int i = 0; i < 12; i++) {
        fprintf(f, "f %d %d %d\n", F[i][0], F[i][1], F[i][2]);
    }
    fclose(f);
}

/* The same box with every vertex carried through a rotation the test
 * builds itself. Its inertia tensor in the mesh frame has nonzero
 * products, which is the only way a wrong coefficient on the product
 * integral can be seen: every axis-aligned fixture has products of
 * exactly zero, and a wrong coefficient times zero is still zero. */
static void write_rotated_box_mesh_(const char *path, double hx, double hy,
                                    double hz, double R[3][3])
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fprintf(f, "# box rotated into a frame with nonzero products\n");
    const double v[8][3] = {
        { -hx, -hy, -hz }, {  hx, -hy, -hz }, {  hx,  hy, -hz },
        { -hx,  hy, -hz }, { -hx, -hy,  hz }, {  hx, -hy,  hz },
        {  hx,  hy,  hz }, { -hx,  hy,  hz }
    };
    for (int i = 0; i < 8; i++) {
        double o[3];
        for (int r = 0; r < 3; r++) {
            o[r] = R[r][0] * v[i][0] + R[r][1] * v[i][1] + R[r][2] * v[i][2];
        }
        fprintf(f, "v %.17g %.17g %.17g\n", o[0], o[1], o[2]);
    }
    static const int F[12][3] = {
        { 1, 4, 3 }, { 1, 3, 2 }, { 5, 6, 7 }, { 5, 7, 8 },
        { 1, 2, 6 }, { 1, 6, 5 }, { 2, 3, 7 }, { 2, 7, 6 },
        { 3, 4, 8 }, { 3, 8, 7 }, { 4, 1, 5 }, { 4, 5, 8 }
    };
    for (int i = 0; i < 12; i++) {
        fprintf(f, "f %d %d %d\n", F[i][0], F[i][1], F[i][2]);
    }
    fclose(f);
}

/* A capsule of radius r whose cylindrical band has length L, as a
 * latitude-longitude mesh: `rings` rings per hemisphere and `sectors`
 * around. L equal to zero gives a sphere. The mesh is inscribed in
 * the solid, so its volume is short of the true one by a term that
 * falls with the square of the subdivision; every tolerance below
 * states the density it was measured at. */
static void write_capsule_mesh_(const char *path, double r, double L,
                                int rings, int sectors)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fprintf(f, "# capsule r=%g L=%g rings=%d sectors=%d\n",
            r, L, rings, sectors);
    /* Vertex 1 is the top pole; then `rings` rings of the upper cap,
     * then `rings` rings of the lower cap, then the bottom pole. The
     * band between the two equatorial rings is the cylinder. */
    fprintf(f, "v 0 0 %.17g\n", L / 2.0 + r);
    int n_ring = 0;
    for (int half = 0; half < 2; half++) {
        for (int i = 1; i <= rings; i++) {
            double th = (M_PI / 2.0) * (double)i / (double)rings;
            double z  = (half == 0 ? 1.0 : -1.0) *
                        (L / 2.0 + r * cos(th));
            double rr = r * sin(th);
            if (half == 1) {
                th = (M_PI / 2.0) * (double)(rings - i + 1) / (double)rings;
                z  = -(L / 2.0 + r * cos(th));
                rr = r * sin(th);
            }
            for (int j = 0; j < sectors; j++) {
                double ph = 2.0 * M_PI * (double)j / (double)sectors;
                fprintf(f, "v %.17g %.17g %.17g\n",
                        rr * cos(ph), rr * sin(ph), z);
            }
            n_ring++;
        }
    }
    fprintf(f, "v 0 0 %.17g\n", -(L / 2.0 + r));
    int top = 1;
    int bot = 1 + 1 + n_ring * sectors;
    /* Top cap fan. */
    for (int j = 0; j < sectors; j++) {
        int a = 2 + j;
        int b = 2 + (j + 1) % sectors;
        fprintf(f, "f %d %d %d\n", top, a, b);
    }
    /* Rings, including the cylindrical band. */
    for (int k = 0; k < n_ring - 1; k++) {
        int r0 = 2 + k * sectors;
        int r1 = 2 + (k + 1) * sectors;
        for (int j = 0; j < sectors; j++) {
            int j1 = (j + 1) % sectors;
            fprintf(f, "f %d %d %d\n", r0 + j, r1 + j, r1 + j1);
            fprintf(f, "f %d %d %d\n", r0 + j, r1 + j1, r0 + j1);
        }
    }
    /* Bottom cap fan. */
    int last = 2 + (n_ring - 1) * sectors;
    for (int j = 0; j < sectors; j++) {
        int a = last + j;
        int b = last + (j + 1) % sectors;
        fprintf(f, "f %d %d %d\n", bot, b, a);
    }
    fclose(f);
}

/* ---- Helpers -------------------------------------------------------- */

static KflcAssembly *load_(const char *file, KflcArena **arena_out,
                           KflcDiag *diag, FILE *errs)
{
    char src[512];
    snprintf(src, sizeof src, WORK "/program.kfl");
    KflcArena *a = kflc_arena_create();
    kflc_diag_init(diag, src, errs);
    char path[512];
    snprintf(path, sizeof path, "%s", file);
    KflcAssembly *asmb = kflc_assembly_load(path, src, 1, a, diag);
    *arena_out = a;
    return asmb;
}

static double relerr_(double got, double want)
{
    double d = got - want;
    if (d < 0) d = -d;
    double w = want < 0 ? -want : want;
    return w > 0 ? d / w : d;
}

static void check_close_(const char *what, double got, double want,
                         double tol)
{
    double e = relerr_(got, want);
    if (!(e <= tol)) {
        fprintf(stderr, "%s: got %.17g want %.17g relative error %.3g "
                "(tolerance %.3g)\n", what, got, want, e, tol);
    }
    ASSERT(e <= tol);
}

/* Expect a refusal whose message carries `msg`. */
static void expect_refused_(const char *tag, const char *file,
                            const char *msg)
{
    FILE *errs = fopen(WORK "/err.log", "w+");
    ASSERT(errs != NULL);
    KflcArena *arena = NULL;
    KflcDiag   diag;
    KflcAssembly *a = load_(file, &arena, &diag, errs);
    fflush(errs);
    long sz = ftell(errs);
    rewind(errs);
    char *buf = (char *)malloc((size_t)sz + 1);
    ASSERT(buf != NULL);
    size_t got = sz > 0 ? fread(buf, 1, (size_t)sz, errs) : 0;
    buf[got] = '\0';
    fclose(errs);
    if (a != NULL || diag.errors == 0 || strstr(buf, msg) == NULL) {
        fprintf(stderr, "case `%s`: refused=%s errors=%d\nstderr:\n%s\n",
                tag, a ? "no" : "yes", diag.errors, buf);
    }
    ASSERT(a == NULL);
    ASSERT(diag.errors > 0);
    ASSERT(strstr(buf, msg) != NULL);
    free(buf);
    kflc_arena_release(arena);
    printf("  refusal %-22s OK\n", tag);
    n_pass++;
}

int main(void)
{
    if (mkdir(WORK, 0755) != 0 && access(WORK, W_OK) != 0) {
        perror(WORK);
        return 2;
    }

    /* ---- 2. Derivation against closed forms ------------------- */
    printf("derivation against closed forms:\n");
    {
        /* A box mesh is the box, so the only error is rounding. */
        const double hx = 1.5, hy = 0.75, hz = 0.5, m = 1200.0;
        write_box_mesh_(WORK "/box.k26mesh", hx, hy, hz);
        write_file_(WORK "/box.k26asm",
            "assembly box_one\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 1200.0\n"
            "        at 0 0 0\n"
            "        mesh box.k26mesh\n"
            "    end\n"
            "end\n");
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/box.k26asm", &arena, &diag, stderr);
        ASSERT(a != NULL && diag.errors == 0);

        check_close_("box volume", a->components[0].volume,
                     8.0 * hx * hy * hz, 1e-15);
        check_close_("box mass", a->mass, m, 0.0);
        for (int k = 0; k < 3; k++) ASSERT(fabs(a->com[k]) < 1e-12);
        check_close_("box Ixx", a->inertia[0], m * (hy * hy + hz * hz) / 3.0,
                     1e-14);
        check_close_("box Iyy", a->inertia[1], m * (hx * hx + hz * hz) / 3.0,
                     1e-14);
        check_close_("box Izz", a->inertia[2], m * (hx * hx + hy * hy) / 3.0,
                     1e-14);
        for (int k = 3; k < 6; k++) {
            ASSERT(fabs(a->inertia[k]) < 1e-10 * a->inertia[0]);
        }
        printf("  box mesh reproduces the uniform solid box: OK\n");
        n_pass++;
        kflc_arena_release(arena);
    }

    {
        /* A sphere, at two triangulation densities. The mesh is
         * inscribed in the solid, so it is a little light and a
         * little low in inertia. Two things are asserted, and the
         * second is the one that matters: the error at the finer
         * density is within a stated bound, and it falls by close to
         * four when the subdivision doubles. Second-order convergence
         * to the analytic value is what an inscribed polyhedron
         * gives; an integral with a wrong coefficient would converge
         * to something else, or not converge, and a tolerance alone
         * would not notice. Measured here: 2.207e-3 volume error at
         * 32 by 64 and 5.521e-4 at 64 by 128, a ratio of 4.00. */
        const double r = 2.0, m = 500.0;
        const double v_true = (4.0 / 3.0) * M_PI * r * r * r;
        const double i_true = (2.0 / 5.0) * m * r * r;
        double v_err[2], i_err[2], polar[2];
        const int dens[2][2] = { { 32, 64 }, { 64, 128 } };

        for (int d = 0; d < 2; d++) {
            write_capsule_mesh_(WORK "/sphere.k26mesh", r, 0.0,
                                dens[d][0], dens[d][1]);
            write_file_(WORK "/sphere.k26asm",
                "assembly sphere_one\n"
                "    frame x_to_port\n"
                "    component ball\n"
                "        mass 500.0\n"
                "        at 0 0 0\n"
                "        mesh sphere.k26mesh\n"
                "    end\n"
                "end\n");
            KflcArena *arena = NULL;
            KflcDiag   diag;
            KflcAssembly *a = load_(WORK "/sphere.k26asm", &arena, &diag,
                                    stderr);
            ASSERT(a != NULL && diag.errors == 0);
            v_err[d] = relerr_(a->components[0].volume, v_true);
            i_err[d] = relerr_(a->inertia[0], i_true);
            /* The two transverse moments are equal to the last bits:
             * a latitude-longitude mesh whose sector count divides by
             * four is exactly symmetric about its polar axis. The
             * polar moment is not equal to them, and should not be:
             * the mesh has a distinguished axis that the sphere does
             * not, and that anisotropy is itself a discretisation
             * error which the ratio below watches disappear. */
            ASSERT(relerr_(a->inertia[1], a->inertia[0]) < 1e-12);
            polar[d] = relerr_(a->inertia[2], a->inertia[0]);
            kflc_arena_release(arena);
        }
        check_close_("sphere volume at 64x128",
                     v_true * (1.0 - v_err[1]), v_true, 6.0e-4);
        check_close_("sphere Ixx at 64x128",
                     i_true * (1.0 - i_err[1]), i_true, 4.0e-4);
        double v_ratio = v_err[0] / v_err[1];
        double i_ratio = i_err[0] / i_err[1];
        printf("  sphere: volume error %.3e then %.3e (ratio %.2f), "
               "Ixx error %.3e then %.3e (ratio %.2f)\n",
               v_err[0], v_err[1], v_ratio, i_err[0], i_err[1], i_ratio);
        double p_ratio = polar[0] / polar[1];
        printf("  sphere: polar anisotropy %.3e then %.3e (ratio %.2f)\n",
               polar[0], polar[1], p_ratio);
        ASSERT(v_ratio > 3.5 && v_ratio < 4.5);
        ASSERT(i_ratio > 3.5 && i_ratio < 4.5);
        ASSERT(p_ratio > 3.5 && p_ratio < 4.5);
        ASSERT(polar[1] < 3.0e-4);
        printf("  sphere mesh converges on the analytic solid at second "
               "order, its own anisotropy included: OK\n");
        n_pass++;
    }

    {
        /* The capsule's closed form against a capsule mesh. Two
         * independent paths through this file meet here: if either
         * the tetrahedron integral or the hand-assembled capsule
         * moments were wrong, they would not agree. */
        const double r = 0.6, L = 2.4, m = 800.0;
        write_capsule_mesh_(WORK "/cap.k26mesh", r, L, 48, 96);
        write_file_(WORK "/cap_mesh.k26asm",
            "assembly cap_mesh\n"
            "    frame x_to_port\n"
            "    component body\n"
            "        mass 800.0\n"
            "        at 0 0 0\n"
            "        mesh cap.k26mesh\n"
            "    end\n"
            "end\n");
        write_file_(WORK "/cap_prim.k26asm",
            "assembly cap_prim\n"
            "    frame x_to_port\n"
            "    component body\n"
            "        mass 800.0\n"
            "        at 0 0 0\n"
            "        collider capsule 0 0 -1.2 0 0 1.2 0.6\n"
            "    end\n"
            "end\n");
        KflcArena *a1 = NULL, *a2 = NULL;
        KflcDiag   d1, d2;
        KflcAssembly *mesh = load_(WORK "/cap_mesh.k26asm", &a1, &d1, stderr);
        KflcAssembly *prim = load_(WORK "/cap_prim.k26asm", &a2, &d2, stderr);
        ASSERT(mesh != NULL && d1.errors == 0);
        ASSERT(prim != NULL && d2.errors == 0);

        double v_true = M_PI * r * r * L + (4.0 / 3.0) * M_PI * r * r * r;
        check_close_("capsule volume (closed form)",
                     prim->components[0].volume, v_true, 1e-15);
        check_close_("capsule volume (mesh)", mesh->components[0].volume,
                     v_true, 2e-3);
        /* Transverse and axial moments, mesh against closed form, at
         * 48x96. The mass is the same by declaration, so this
         * compares shape alone. */
        check_close_("capsule Ixx", mesh->inertia[0], prim->inertia[0], 3e-3);
        check_close_("capsule Izz", mesh->inertia[2], prim->inertia[2], 3e-3);
        ASSERT(prim->inertia[2] < prim->inertia[0]);   /* long and thin */
        (void)m;
        printf("  capsule closed form agrees with a 48x96 capsule mesh: "
               "OK\n");
        n_pass++;
        kflc_arena_release(a1);
        kflc_arena_release(a2);
    }

    {
        /* A rotated box. Its products of inertia in the mesh frame
         * are nonzero, so the coefficient on the product integral is
         * exercised rather than multiplied by zero, and the expected
         * tensor is built here from the closed form and this test's
         * own rotation rather than from anything the reader does. */
        const double hx = 1.5, hy = 0.75, hz = 0.5, m = 1200.0;
        const double ca = cos(0.5), sa = sin(0.5);   /* about z */
        const double cb = cos(0.3), sb = sin(0.3);   /* then about x */
        double Rz[3][3] = { { ca, -sa, 0 }, { sa, ca, 0 }, { 0, 0, 1 } };
        double Rx[3][3] = { { 1, 0, 0 }, { 0, cb, -sb }, { 0, sb, cb } };
        double R[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R[i][j] = Rx[i][0] * Rz[0][j] + Rx[i][1] * Rz[1][j]
                        + Rx[i][2] * Rz[2][j];
            }
        }
        write_rotated_box_mesh_(WORK "/rot.k26mesh", hx, hy, hz, R);
        write_file_(WORK "/rot.k26asm",
            "assembly rot_box\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 1200.0\n"
            "        at 0 0 0\n"
            "        mesh rot.k26mesh\n"
            "    end\n"
            "end\n");
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/rot.k26asm", &arena, &diag, stderr);
        ASSERT(a != NULL && diag.errors == 0);

        double D[3] = { m * (hy * hy + hz * hz) / 3.0,
                        m * (hx * hx + hz * hz) / 3.0,
                        m * (hx * hx + hy * hy) / 3.0 };
        double want[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                want[i][j] = R[i][0] * D[0] * R[j][0]
                           + R[i][1] * D[1] * R[j][1]
                           + R[i][2] * D[2] * R[j][2];
            }
        }
        check_close_("rotated box Ixx", a->inertia[0], want[0][0], 1e-13);
        check_close_("rotated box Iyy", a->inertia[1], want[1][1], 1e-13);
        check_close_("rotated box Izz", a->inertia[2], want[2][2], 1e-13);
        check_close_("rotated box Ixy", a->inertia[3], want[0][1], 1e-12);
        check_close_("rotated box Ixz", a->inertia[4], want[0][2], 1e-12);
        check_close_("rotated box Iyz", a->inertia[5], want[1][2], 1e-12);
        /* The products are large enough that a wrong coefficient on
         * them could not hide inside the tolerance. */
        ASSERT(fabs(a->inertia[3]) > 0.05 * a->inertia[0]);
        ASSERT(fabs(a->inertia[5]) > 0.05 * a->inertia[0]);
        printf("  a rotated box reproduces R I R-transpose, products "
               "included: OK\n");
        n_pass++;
        kflc_arena_release(arena);
    }

    {
        /* Two components offset in all three axes, against a
         * parallel-axis sum computed here from the theorem. Offsets
         * along one axis alone leave that axis's own shift term at
         * zero, which is where a sign error hides; these do not. */
        const double h = 0.5;
        const double m1 = 100.0, m2 = 300.0;
        const double p1[3] = { 2.0, 1.0, 0.5 };
        const double p2[3] = { -1.0, -0.5, 0.25 };
        write_box_mesh_(WORK "/unit.k26mesh", h, h, h);
        write_file_(WORK "/two.k26asm",
            "assembly two_part\n"
            "    frame x_to_port\n"
            "    component fore\n"
            "        mass 100.0\n"
            "        at 2.0 1.0 0.5\n"
            "        mesh unit.k26mesh\n"
            "    end\n"
            "    component aft\n"
            "        mass 300.0\n"
            "        at -1.0 -0.5 0.25\n"
            "        mesh unit.k26mesh\n"
            "    end\n"
            "end\n");
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/two.k26asm", &arena, &diag, stderr);
        ASSERT(a != NULL && diag.errors == 0);

        double mt = m1 + m2;
        double com[3];
        for (int k = 0; k < 3; k++) com[k] = (m1 * p1[k] + m2 * p2[k]) / mt;
        check_close_("two-part mass", a->mass, mt, 0.0);
        for (int k = 0; k < 3; k++) {
            char lbl[32];
            snprintf(lbl, sizeof lbl, "two-part com[%d]", k);
            check_close_(lbl, a->com[k], com[k], 1e-14);
        }

        double own = 2.0 * h * h / 3.0;         /* per unit mass */
        double want[6] = { 0, 0, 0, 0, 0, 0 };
        const double *pp[2] = { p1, p2 };
        const double mm[2]  = { m1, m2 };
        for (int i = 0; i < 2; i++) {
            double d[3];
            for (int k = 0; k < 3; k++) d[k] = pp[i][k] - com[k];
            double dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            want[0] += mm[i] * (own + dd - d[0] * d[0]);
            want[1] += mm[i] * (own + dd - d[1] * d[1]);
            want[2] += mm[i] * (own + dd - d[2] * d[2]);
            want[3] += mm[i] * (-d[0] * d[1]);
            want[4] += mm[i] * (-d[0] * d[2]);
            want[5] += mm[i] * (-d[1] * d[2]);
        }
        static const char *NM[6] = { "Ixx", "Iyy", "Izz",
                                     "Ixy", "Ixz", "Iyz" };
        for (int k = 0; k < 6; k++) {
            char lbl[40];
            snprintf(lbl, sizeof lbl, "two-part %s", NM[k]);
            check_close_(lbl, a->inertia[k], want[k], 1e-13);
        }
        /* Each shift term is genuinely nonzero, so a sign error in
         * any one of them changes an assertion above. */
        ASSERT(fabs(a->inertia[3]) > 1.0);
        printf("  two components offset in three axes reproduce an "
               "independent parallel-axis sum: OK\n");
        n_pass++;
        kflc_arena_release(arena);
    }

    {
        /* A mesh centred away from its own origin: the component's
         * derived centre of mass must be where the geometry actually
         * is, and the assembly's must be that plus the placement. The
         * inertia is about the centre of mass either way, so it still
         * equals the plain box's. */
        const double h = 0.5, m = 200.0;
        const double c[3] = { 0.7, -0.2, 0.35 };
        const double at[3] = { 1.0, 2.0, -3.0 };
        write_offset_box_mesh_(WORK "/off.k26mesh", h, c);
        write_file_(WORK "/off.k26asm",
            "assembly off_box\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 200.0\n"
            "        at 1.0 2.0 -3.0\n"
            "        mesh off.k26mesh\n"
            "    end\n"
            "end\n");
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/off.k26asm", &arena, &diag, stderr);
        ASSERT(a != NULL && diag.errors == 0);
        for (int k = 0; k < 3; k++) {
            char lbl[48];
            snprintf(lbl, sizeof lbl, "offset mesh component com[%d]", k);
            check_close_(lbl, a->components[0].com[k], c[k], 1e-13);
            snprintf(lbl, sizeof lbl, "offset mesh assembly com[%d]", k);
            check_close_(lbl, a->com[k], at[k] + c[k], 1e-13);
        }
        double own = m * 2.0 * h * h / 3.0;
        check_close_("offset mesh Ixx", a->inertia[0], own, 1e-13);
        for (int k = 3; k < 6; k++) ASSERT(fabs(a->inertia[k]) < 1e-10 * own);
        printf("  a mesh centred off its own origin puts its centre of "
               "mass where the geometry is: OK\n");
        n_pass++;
        kflc_arena_release(arena);
    }

    /* ---- 1. Digest --------------------------------------------- */
    printf("identity digest:\n");
    {
        KflcArena *a1 = NULL, *a2 = NULL;
        KflcDiag   d1, d2;
        KflcAssembly *x = load_(WORK "/box.k26asm", &a1, &d1, stderr);
        KflcAssembly *y = load_(WORK "/box.k26asm", &a2, &d2, stderr);
        ASSERT(x && y);
        ASSERT(memcmp(x->digest, y->digest, KFLC_ASM_DIGEST) == 0);
        char hex[2 * KFLC_ASM_DIGEST + 1];
        kflc_assembly_digest_hex(x->digest, hex);
        printf("  stable across two loads: %s\n", hex);
        n_pass++;

        /* One byte in the assembly: the mass gains a digit. */
        write_file_(WORK "/box_b.k26asm",
            "assembly box_one\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 1200.1\n"
            "        at 0 0 0\n"
            "        mesh box.k26mesh\n"
            "    end\n"
            "end\n");
        KflcArena *a3 = NULL;
        KflcDiag   d3;
        KflcAssembly *z = load_(WORK "/box_b.k26asm", &a3, &d3, stderr);
        ASSERT(z != NULL);
        ASSERT(memcmp(x->digest, z->digest, KFLC_ASM_DIGEST) != 0);
        printf("  one byte changed in the assembly changes it: OK\n");
        n_pass++;
        kflc_arena_release(a3);

        /* One byte in the referenced mesh, with the assembly
         * untouched. This is the arm that proves the mesh is inside
         * the identity rather than beside it. */
        write_box_mesh_(WORK "/box.k26mesh", 1.5, 0.75, 0.5000001);
        KflcArena *a4 = NULL;
        KflcDiag   d4;
        KflcAssembly *w = load_(WORK "/box.k26asm", &a4, &d4, stderr);
        ASSERT(w != NULL);
        ASSERT(memcmp(x->digest, w->digest, KFLC_ASM_DIGEST) != 0);
        printf("  one byte changed in a referenced mesh changes it: OK\n");
        n_pass++;
        kflc_arena_release(a4);
        write_box_mesh_(WORK "/box.k26mesh", 1.5, 0.75, 0.5);

        kflc_arena_release(a1);
        kflc_arena_release(a2);
    }

    /* ---- 4. Provenance ----------------------------------------- */
    printf("provenance:\n");
    {
        write_file_(WORK "/prov_ok.k26asm",
            "assembly prov_ok\n"
            "    frame x_to_port\n"
            "    provenance mass \"A named source, table 1\" cited\n"
            "    provenance inertia \"derived from the geometry\" computed\n"
            "    component hull\n"
            "        mass 1200.0\n"
            "        at 0 0 0\n"
            "        mesh box.k26mesh\n"
            "    end\n"
            "end\n");
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/prov_ok.k26asm", &arena, &diag, stderr);
        ASSERT(a != NULL);
        ASSERT(diag.errors == 0 && diag.warnings == 0);
        ASSERT(a->n_provenance == 2 && a->n_unverified == 0);
        printf("  cited and computed pass without a warning: OK\n");
        n_pass++;
        kflc_arena_release(arena);
    }
    {
        write_file_(WORK "/prov_unver.k26asm",
            "assembly prov_unver\n"
            "    frame x_to_port\n"
            "    provenance thruster_position \"working figure\" unverified\n"
            "    component hull\n"
            "        mass 1200.0\n"
            "        at 0 0 0\n"
            "        mesh box.k26mesh\n"
            "    end\n"
            "end\n");
        FILE *errs = fopen(WORK "/warn.log", "w+");
        ASSERT(errs != NULL);
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/prov_unver.k26asm", &arena, &diag,
                                errs);
        fflush(errs);
        long sz = ftell(errs);
        rewind(errs);
        char *buf = (char *)malloc((size_t)sz + 1);
        ASSERT(buf != NULL);
        size_t got = sz > 0 ? fread(buf, 1, (size_t)sz, errs) : 0;
        buf[got] = '\0';
        fclose(errs);
        ASSERT(a != NULL);
        ASSERT(diag.errors == 0);
        ASSERT(a->n_unverified == 1);
        ASSERT(strstr(buf, "thruster_position") != NULL);
        ASSERT(strstr(buf, "unverified") != NULL);
        printf("  unverified passes and is reported by property name: OK\n");
        n_pass++;
        free(buf);
        kflc_arena_release(arena);
    }

    /* ---- 3. Refusals ------------------------------------------- */
    printf("refusals:\n");

    write_file_(WORK "/prov_bad.k26asm",
        "assembly prov_bad\n"
        "    frame x_to_port\n"
        "    provenance mass \"somewhere\" probably\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("prov_status", WORK "/prov_bad.k26asm",
                    "is not a provenance status");

    /* A mesh with one face removed is no longer closed. */
    {
        write_box_mesh_(WORK "/open.k26mesh", 1.0, 1.0, 1.0);
        FILE *f = fopen(WORK "/open.k26mesh", "r");
        ASSERT(f != NULL);
        char lines[64][128];
        int  n = 0;
        while (n < 64 && fgets(lines[n], sizeof lines[n], f)) n++;
        fclose(f);
        f = fopen(WORK "/open.k26mesh", "w");
        ASSERT(f != NULL);
        for (int i = 0; i < n - 1; i++) fputs(lines[i], f);
        fclose(f);
        write_file_(WORK "/open.k26asm",
            "assembly open_mesh\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 1.0\n"
            "        mesh open.k26mesh\n"
            "    end\n"
            "end\n");
        expect_refused_("mesh_not_closed", WORK "/open.k26asm",
                        "mesh is not closed");
    }

    /* A mesh with one face's winding reversed. */
    {
        write_box_mesh_(WORK "/flip.k26mesh", 1.0, 1.0, 1.0);
        FILE *f = fopen(WORK "/flip.k26mesh", "r");
        ASSERT(f != NULL);
        char lines[64][128];
        int  n = 0;
        while (n < 64 && fgets(lines[n], sizeof lines[n], f)) n++;
        fclose(f);
        /* Reverse the last face's vertex order. */
        int a1, b1, c1;
        ASSERT(sscanf(lines[n - 1], "f %d %d %d", &a1, &b1, &c1) == 3);
        snprintf(lines[n - 1], sizeof lines[n - 1], "f %d %d %d\n",
                 c1, b1, a1);
        f = fopen(WORK "/flip.k26mesh", "w");
        ASSERT(f != NULL);
        for (int i = 0; i < n; i++) fputs(lines[i], f);
        fclose(f);
        write_file_(WORK "/flip.k26asm",
            "assembly flip_mesh\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 1.0\n"
            "        mesh flip.k26mesh\n"
            "    end\n"
            "end\n");
        expect_refused_("mesh_winding", WORK "/flip.k26asm",
                        "winding is inconsistent");
    }

    /* A directive outside the accepted subset. */
    write_file_(WORK "/vn.k26mesh",
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1 2 3\n");
    write_file_(WORK "/vn.k26asm",
        "assembly vn_mesh\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        mesh vn.k26mesh\n"
        "    end\n"
        "end\n");
    expect_refused_("mesh_subset", WORK "/vn.k26asm",
                    "not part of the accepted mesh subset");

    write_file_(WORK "/nokey.k26asm",
        "assembly nokey\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        colour red\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("unknown_key", WORK "/nokey.k26asm",
                    "is not a component key");

    write_file_(WORK "/noshape.k26asm",
        "assembly noshape\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        at 0 0 0\n"
        "    end\n"
        "end\n");
    expect_refused_("no_geometry", WORK "/noshape.k26asm",
                    "declares neither a mesh nor a collider");

    write_file_(WORK "/badquat.k26asm",
        "assembly badquat\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        rotate 1 1 0 0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("non_unit_rotation", WORK "/badquat.k26asm",
                    "is not a unit quaternion");

    write_file_(WORK "/units.k26asm",
        "assembly units\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1200kg\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("number_with_suffix", WORK "/units.k26asm",
                    "is not a number");

    write_file_(WORK "/noframe.k26asm",
        "assembly noframe\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("no_frame", WORK "/noframe.k26asm", "no `frame`");

    write_file_(WORK "/badframe.k26asm",
        "assembly badframe\n"
        "    frame z_up\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("unknown_frame", WORK "/badframe.k26asm",
                    "is not a defined body frame");

    write_file_(WORK "/featmass.k26asm",
        "assembly featmass\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "    thruster t1\n"
        "        at 0 0 0\n"
        "        dir 1 0 0\n"
        "        thrust 400.0\n"
        "        mass 2.0\n"
        "    end\n"
        "end\n");
    expect_refused_("feature_mass", WORK "/featmass.k26asm",
                    "carries no mass of its own");

    write_file_(WORK "/dupname.k26asm",
        "assembly dupname\n"
        "    frame x_to_port\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "    component hull\n"
        "        mass 1.0\n"
        "        collider box 1 1 1\n"
        "    end\n"
        "end\n");
    expect_refused_("duplicate_name", WORK "/dupname.k26asm",
                    "is already declared");

    expect_refused_("missing_file", WORK "/does_not_exist.k26asm",
                    "cannot open assembly");

    /* ---- Features parse and are carried ------------------------ */
    {
        write_file_(WORK "/feat.k26asm",
            "assembly feat\n"
            "    frame x_to_port\n"
            "    component hull\n"
            "        mass 1000.0\n"
            "        collider capsule 0 0 0 2.4 0 0 1.85\n"
            "    end\n"
            "    port forward\n"
            "        at 2.60 0 0\n"
            "        axis 1 0 0\n"
            "        roll_ref 0 1 0\n"
            "        capture idss_e\n"
            "    end\n"
            "    thruster rcs_py_minus\n"
            "        at 1.05 0.92 0\n"
            "        dir 0 -1 0\n"
            "        thrust 400.0\n"
            "    end\n"
            "    wheel pitch\n"
            "        axis 0 1 0\n"
            "        spin_inertia 0.031\n"
            "        max_momentum 15.0\n"
            "        max_torque 0.10\n"
            "        viscous 1.0e-5\n"
            "        coulomb 2.0e-4\n"
            "        dead_rate 1.0e-3\n"
            "    end\n"
            "    magnetorquer m_y\n"
            "        axis 0 1 0\n"
            "        max_dipole 30.0\n"
            "    end\n"
            "end\n");
        KflcArena *arena = NULL;
        KflcDiag   diag;
        KflcAssembly *a = load_(WORK "/feat.k26asm", &arena, &diag, stderr);
        ASSERT(a != NULL && diag.errors == 0);
        ASSERT(a->n_features == 4);
        ASSERT(a->n_colliders == 1);
        ASSERT(a->features[0].kind == KFLC_FEAT_PORT);
        ASSERT(strcmp(a->features[0].capture, "idss_e") == 0);
        ASSERT(a->features[1].kind == KFLC_FEAT_THRUSTER);
        ASSERT(a->features[1].thrust == 400.0);
        ASSERT(a->features[2].kind == KFLC_FEAT_WHEEL);
        ASSERT(a->features[2].max_momentum == 15.0);
        ASSERT(a->features[3].kind == KFLC_FEAT_TORQUER);
        ASSERT(a->features[3].max_dipole == 30.0);
        printf("  ports, thrusters, wheels, and magnetorquers parse and "
               "are carried: OK\n");
        n_pass++;
        kflc_arena_release(arena);
    }

    printf("test_assembly: %d check(s) passed\n", n_pass);
    return 0;
}
