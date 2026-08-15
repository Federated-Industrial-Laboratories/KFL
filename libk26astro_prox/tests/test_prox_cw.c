/* test_prox_cw.c: the linearised propagator, its structure, and the
 * range over which it is worth anything.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   Checking the matrix against the closed form it was written from
 *   would check a transcription against itself. Every structural arm
 *   here is built from the equations of motion instead: the matrix is
 *   compared against the exponential of the system matrix, computed in
 *   this file by scaling and squaring and sharing no line with the
 *   library, and the propagated state is differentiated numerically
 *   and required to satisfy the equations themselves. A wrong entry
 *   fails both.
 *
 *   Comparing the propagator against an integrated trajectory proves
 *   nothing if the integration error is the size of the disagreement
 *   being measured. The deputy is integrated at two step sizes and the
 *   difference between them is reported and required to be two orders
 *   below the disagreement, so what the arm measures is the
 *   linearisation and not the integrator. The chief is not integrated
 *   at all: its orbit is circular by construction and is advanced by
 *   rotating it, so the frame the comparison is made in carries no
 *   error either.
 *
 *   An agreement arm at one separation would pass for any model that
 *   happened to be close there. The disagreement is measured at four
 *   separations and required to fall as the square of the separation,
 *   which is the order of the term the linearisation drops.
 *
 *   What the bound is, and what it is not. The bound below is a
 *   ceiling derived from a worst-case coefficient and a heuristic
 *   double integration, and it carries roughly sevenfold slack. Both
 *   it and the disagreement grow as the square of the separation, so
 *   their ratio is nearly constant and in fact falls slowly with
 *   separation: 0.137 at a hundred metres and 0.133 at two hundred
 *   kilometres. The disagreement therefore does not leave this bound
 *   at any separation at which two-body motion is still the question,
 *   and no arm here claims that it does. What the range arm
 *   establishes is the growth: the disagreement as a fraction of the
 *   separation is itself proportional to the separation, so a model
 *   good to three millimetres in a hundred is wrong by eleven
 *   kilometres in two hundred, and that is the statement a reader can
 *   act on.
 *
 *   A hold-point arm asserting only that the craft returns after one
 *   orbit would pass for the trivial answer of zero velocity at a zero
 *   hold point. The same fixture is run with the drift-removing term
 *   omitted, and that control must fail to return, by kilometres.
 */
#include "k26astro_prox/prox.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

#define MU_EARTH 3.986004418e14

static double absd_(double x) { return x < 0.0 ? -x : x; }

/* ---- The system matrix, written from the equations of motion ------ *
 *
 * x'' = 3 n^2 x + 2 n y',   y'' = -2 n x',   z'' = -n^2 z,
 * in the state order r_x, r_y, r_z, v_x, v_y, v_z. This is the only
 * place in the gate that the dynamics are stated, and everything
 * structural below is derived from it rather than from the library. */
static void sys_matrix_(double n, double *a)
{
    for (int i = 0; i < 36; i++) a[i] = 0.0;
    a[0 * 6 + 3] = 1.0;
    a[1 * 6 + 4] = 1.0;
    a[2 * 6 + 5] = 1.0;
    a[3 * 6 + 0] = 3.0 * n * n;
    a[3 * 6 + 4] = 2.0 * n;
    a[4 * 6 + 3] = -2.0 * n;
    a[5 * 6 + 2] = -n * n;
}

static void mat_mul_(const double *a, const double *b, double *out)
{
    double tmp[36];
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            double acc = 0.0;
            for (int k = 0; k < 6; k++) acc += a[i * 6 + k] * b[k * 6 + j];
            tmp[i * 6 + j] = acc;
        }
    }
    memcpy(out, tmp, sizeof tmp);
}

/* exp(M) by scaling and squaring with a truncated Taylor series. */
static void mat_exp_(const double *m, double *out)
{
    enum { SQUARINGS = 20, TERMS = 40 };
    double s[36], term[36], acc[36];
    for (int i = 0; i < 36; i++) s[i] = m[i] / (double)(1L << SQUARINGS);
    for (int i = 0; i < 36; i++) {
        acc[i]  = (i % 7 == 0) ? 1.0 : 0.0;
        term[i] = (i % 7 == 0) ? 1.0 : 0.0;
    }
    for (int k = 1; k < TERMS; k++) {
        mat_mul_(term, s, term);
        for (int i = 0; i < 36; i++) term[i] /= (double)k;
        for (int i = 0; i < 36; i++) acc[i] += term[i];
    }
    for (int q = 0; q < SQUARINGS; q++) mat_mul_(acc, acc, acc);
    memcpy(out, acc, 36 * sizeof(double));
}

/* Determinant by Gaussian elimination with partial pivoting. */
static double det6_(const double *m)
{
    double a[36];
    memcpy(a, m, sizeof a);
    double det = 1.0;
    for (int c = 0; c < 6; c++) {
        int piv = c;
        for (int r = c + 1; r < 6; r++) {
            if (absd_(a[r * 6 + c]) > absd_(a[piv * 6 + c])) piv = r;
        }
        if (absd_(a[piv * 6 + c]) == 0.0) return 0.0;
        if (piv != c) {
            for (int k = 0; k < 6; k++) {
                double t = a[c * 6 + k];
                a[c * 6 + k] = a[piv * 6 + k];
                a[piv * 6 + k] = t;
            }
            det = -det;
        }
        det *= a[c * 6 + c];
        for (int r = c + 1; r < 6; r++) {
            double f = a[r * 6 + c] / a[c * 6 + c];
            for (int k = c; k < 6; k++) a[r * 6 + k] -= f * a[c * 6 + k];
        }
    }
    return det;
}

/* ---- Two-body integration, for the comparison arm ---------------- */

typedef struct { double r[3], v[3]; } State;

static void accel_(const double r[3], double a[3])
{
    double r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    double rm = sqrt(r2);
    double k  = -MU_EARTH / (r2 * rm);
    a[0] = k * r[0]; a[1] = k * r[1]; a[2] = k * r[2];
}

static void deriv_(const State *s, State *d)
{
    for (int i = 0; i < 3; i++) d->r[i] = s->v[i];
    accel_(s->r, d->v);
}

static void rk4_(State *s, double h, long steps)
{
    for (long n = 0; n < steps; n++) {
        State k1, k2, k3, k4, t;
        deriv_(s, &k1);
        for (int i = 0; i < 3; i++) {
            t.r[i] = s->r[i] + 0.5 * h * k1.r[i];
            t.v[i] = s->v[i] + 0.5 * h * k1.v[i];
        }
        deriv_(&t, &k2);
        for (int i = 0; i < 3; i++) {
            t.r[i] = s->r[i] + 0.5 * h * k2.r[i];
            t.v[i] = s->v[i] + 0.5 * h * k2.v[i];
        }
        deriv_(&t, &k3);
        for (int i = 0; i < 3; i++) {
            t.r[i] = s->r[i] + h * k3.r[i];
            t.v[i] = s->v[i] + h * k3.v[i];
        }
        deriv_(&t, &k4);
        for (int i = 0; i < 3; i++) {
            s->r[i] += (h / 6.0) * (k1.r[i] + 2*k2.r[i] + 2*k3.r[i] + k4.r[i]);
            s->v[i] += (h / 6.0) * (k1.v[i] + 2*k2.v[i] + 2*k3.v[i] + k4.v[i]);
        }
    }
}

/* The relative position of an integrated deputy, resolved onto the
 * chief's frame at the same instant. The chief's circular orbit is
 * advanced by rotating it, so no integration error enters the frame. */
static K26V3 integrated_rel_(double r0, double rel0[6], double t,
                             double h, double *drift_out)
{
    double n  = sqrt(MU_EARTH / (r0 * r0 * r0));
    double v0 = sqrt(MU_EARTH / r0);

    /* At t = 0 the chief sits on +x moving towards +y, so its frame is
     * the world frame and the relative state maps straight through.
     * The deputy's inertial velocity carries the frame's rotation
     * back: v_inertial = v_chief + v_rel + omega x rho. */
    State dep;
    dep.r[0] = r0 + rel0[0];
    dep.r[1] = rel0[1];
    dep.r[2] = rel0[2];
    dep.v[0] = rel0[3] - n * rel0[1];
    dep.v[1] = v0 + rel0[4] + n * rel0[0];
    dep.v[2] = rel0[5];

    State coarse = dep, fine = dep;
    long steps = (long)(t / h + 0.5);
    rk4_(&coarse, t / (double)steps, steps);
    rk4_(&fine, t / (double)(2 * steps), 2 * steps);
    if (drift_out) {
        double d = 0.0;
        for (int i = 0; i < 3; i++) {
            double e = coarse.r[i] - fine.r[i];
            d += e * e;
        }
        *drift_out = sqrt(d);
    }

    /* The chief, rotated. */
    double th = n * t, c = cos(th), s = sin(th);
    K26V3 e1 = k26m3d_v3(c, s, 0.0);
    K26V3 e2 = k26m3d_v3(-s, c, 0.0);
    K26V3 e3 = k26m3d_v3(0.0, 0.0, 1.0);
    K26V3 rho = k26m3d_v3(fine.r[0] - r0 * c,
                          fine.r[1] - r0 * s,
                          fine.r[2]);
    return k26m3d_v3(k26m3d_v3_dot(rho, e1),
                     k26m3d_v3_dot(rho, e2),
                     k26m3d_v3_dot(rho, e3));
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    double n = sqrt(MU_EARTH / (7.0e6 * 7.0e6 * 7.0e6));

    /* ---- 1. The identity at a zero interval ---------------------- */
    {
        double phi[36];
        ASSERT(k26astro_prox_cw_stm(n, 0.0, phi) == K26ASTRO_PROX_OK);
        double worst = 0.0;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                double want = (i == j) ? 1.0 : 0.0;
                double e = absd_(phi[i * 6 + j] - want);
                if (e > worst) worst = e;
            }
        }
        printf("gate 1: worst element of Phi(0) - I: %.3e\n", worst);
        ASSERT(worst == 0.0);
        n_pass++;
        printf("gate 1: a zero interval transports nothing: OK\n");
    }

    /* ---- 2. Unit determinant, at every interval ------------------ */
    {
        /* The system matrix has zero trace, so the determinant is
         * exp(0) = 1 for every interval. A wrong entry almost always
         * breaks it. */
        static const double ts[] = { 1.0, 60.0, 900.0, 2900.0, -1700.0 };
        double worst = 0.0;
        for (int k = 0; k < 5; k++) {
            double phi[36];
            ASSERT(k26astro_prox_cw_stm(n, ts[k], phi) == K26ASTRO_PROX_OK);
            double d = det6_(phi);
            printf("  det Phi(%8.1f s) = %.15f\n", ts[k], d);
            if (absd_(d - 1.0) > worst) worst = absd_(d - 1.0);
        }
        ASSERT(worst < 1e-11);
        n_pass++;
        printf("gate 2: unit determinant at every interval: OK\n");
    }

    /* ---- 3. It composes as a semigroup --------------------------- */
    {
        double p1[36], p2[36], psum[36], prod[36];
        double t1 = 613.0, t2 = 1979.0;
        ASSERT(k26astro_prox_cw_stm(n, t1, p1) == K26ASTRO_PROX_OK);
        ASSERT(k26astro_prox_cw_stm(n, t2, p2) == K26ASTRO_PROX_OK);
        ASSERT(k26astro_prox_cw_stm(n, t1 + t2, psum) == K26ASTRO_PROX_OK);
        mat_mul_(p2, p1, prod);
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < 36; i++) {
            if (absd_(psum[i]) > scale) scale = absd_(psum[i]);
            double e = absd_(psum[i] - prod[i]);
            if (e > worst) worst = e;
        }
        printf("gate 3: worst |Phi(t1+t2) - Phi(t2)Phi(t1)| / scale:"
               " %.3e\n", worst / scale);
        ASSERT(worst / scale < 1e-12);
        n_pass++;
        printf("gate 3: the matrix composes over consecutive"
               " intervals: OK\n");
    }

    /* ---- 4. It is the exponential of the system matrix ----------- */
    {
        /* The independent construction: exp(A t) from the equations of
         * motion, sharing no line with the library. */
        static const double ts[] = { 1.0, 37.0, 250.0, 1000.0, 2900.0 };
        double worst = 0.0;
        for (int k = 0; k < 5; k++) {
            double a[36], at[36], e[36], phi[36];
            sys_matrix_(n, a);
            for (int i = 0; i < 36; i++) at[i] = a[i] * ts[k];
            mat_exp_(at, e);
            ASSERT(k26astro_prox_cw_stm(n, ts[k], phi) == K26ASTRO_PROX_OK);
            double scale = 0.0, err = 0.0;
            for (int i = 0; i < 36; i++) {
                if (absd_(e[i]) > scale) scale = absd_(e[i]);
                double d = absd_(e[i] - phi[i]);
                if (d > err) err = d;
            }
            printf("  t = %7.1f s: worst |exp(At) - Phi| / scale ="
                   " %.3e\n", ts[k], err / scale);
            if (err / scale > worst) worst = err / scale;
        }
        ASSERT(worst < 1e-9);
        n_pass++;
        printf("gate 4: the matrix is the exponential of the system"
               " the equations give: OK\n");
    }

    /* ---- 5. The propagated state satisfies the equations --------- */
    {
        /* Differentiate the propagated state numerically and require
         * A x. This catches an entry that is wrong in a way the
         * determinant happens to tolerate, and it names the equations
         * rather than the solution. */
        K26AstroProxRel s0;
        s0.r = k26m3d_v3(120.0, -45.0, 33.0);
        s0.v = k26m3d_v3(0.11, -0.26, 0.07);
        double a[36];
        sys_matrix_(n, a);
        double worst = 0.0;
        for (double t = 0.0; t <= 2400.0; t += 300.0) {
            double dt = 0.05;
            K26AstroProxRel lo, hi, mid;
            ASSERT(k26astro_prox_cw_propagate(n, t - dt, &s0, &lo)
                   == K26ASTRO_PROX_OK);
            ASSERT(k26astro_prox_cw_propagate(n, t + dt, &s0, &hi)
                   == K26ASTRO_PROX_OK);
            ASSERT(k26astro_prox_cw_propagate(n, t, &s0, &mid)
                   == K26ASTRO_PROX_OK);
            double x[6] = { mid.r.x, mid.r.y, mid.r.z,
                            mid.v.x, mid.v.y, mid.v.z };
            double num[6] = {
                (hi.r.x - lo.r.x) / (2 * dt), (hi.r.y - lo.r.y) / (2 * dt),
                (hi.r.z - lo.r.z) / (2 * dt), (hi.v.x - lo.v.x) / (2 * dt),
                (hi.v.y - lo.v.y) / (2 * dt), (hi.v.z - lo.v.z) / (2 * dt)
            };
            for (int i = 0; i < 6; i++) {
                double want = 0.0;
                for (int j = 0; j < 6; j++) want += a[i * 6 + j] * x[j];
                double e = absd_(num[i] - want);
                double sc = absd_(want) > 1e-6 ? absd_(want) : 1e-6;
                if (e / sc > worst) worst = e / sc;
            }
        }
        printf("gate 5: worst relative residual of x' - A x over the"
               " orbit: %.3e\n", worst);
        ASSERT(worst < 1e-6);
        n_pass++;
        printf("gate 5: the propagated state obeys the equations of"
               " motion: OK\n");
    }

    /* ---- 6. The hold point, and the control that it is not free -- */
    {
        double period = 2.0 * M_PI / n;
        K26V3 hold = k26m3d_v3(150.0, -400.0, 0.0);
        K26V3 v;
        ASSERT(k26astro_prox_cw_hold(n, hold, &v) == K26ASTRO_PROX_OK);
        printf("gate 6: hold point (%.1f, %.1f, %.1f) m gives"
               " v = (%.6f, %.6f, %.6f) m/s\n",
               hold.x, hold.y, hold.z, v.x, v.y, v.z);

        K26AstroProxRel s = { hold, v }, out;
        ASSERT(k26astro_prox_cw_propagate(n, period, &s, &out)
               == K26ASTRO_PROX_OK);
        double back = sqrt((out.r.x - hold.x) * (out.r.x - hold.x) +
                           (out.r.y - hold.y) * (out.r.y - hold.y) +
                           (out.r.z - hold.z) * (out.r.z - hold.z));
        printf("  after one orbit the bounded state is %.6e m from"
               " where it began\n", back);
        ASSERT(back < 1e-6);

        /* The control: the same hold point with the drift-removing
         * term dropped. Without it the arm above would pass for any
         * implementation that returned any velocity at all. */
        K26AstroProxRel bad = { hold, k26m3d_v3(0.0, 0.0, 0.0) }, bout;
        ASSERT(k26astro_prox_cw_propagate(n, period, &bad, &bout)
               == K26ASTRO_PROX_OK);
        double away = absd_(bout.r.y - hold.y);
        printf("  the same point with a zero velocity drifts %.1f m"
               " along-track in the same orbit\n", away);
        ASSERT(away > 1000.0);

        /* A hold point on the along-track axis alone is a fixed point:
         * zero velocity, and it does not move at all. */
        K26V3 vbar = k26m3d_v3(0.0, -250.0, 0.0), vv;
        ASSERT(k26astro_prox_cw_hold(n, vbar, &vv) == K26ASTRO_PROX_OK);
        ASSERT(vv.x == 0.0 && vv.y == 0.0 && vv.z == 0.0);
        K26AstroProxRel sv = { vbar, vv }, svo;
        ASSERT(k26astro_prox_cw_propagate(n, 0.37 * period, &sv, &svo)
               == K26ASTRO_PROX_OK);
        printf("  an along-track hold point sits still: moved %.3e m\n",
               absd_(svo.r.y - vbar.y) + absd_(svo.r.x) + absd_(svo.r.z));
        ASSERT(absd_(svo.r.y - vbar.y) < 1e-9);
        ASSERT(absd_(svo.r.x) < 1e-9);
        n_pass++;
        printf("gate 6: the hold point removes the secular drift, and"
               " dropping it does not: OK\n");
    }

    /* ---- 7. Against an integrated trajectory, and its range ------ */
    {
        double r0 = 7.0e6;
        /* The bound, from the term the linearisation drops. The exact
         * differential gravity expanded in rho/r0 has a second-order
         * term bounded by 12 mu rho^2 / r0^4 = 12 n^2 rho^2 / r0;
         * accumulated over an interval T it displaces the relative
         * position by at most half that times T squared, so
         *
         *     |error| <= 6 n^2 T^2 rho^2 / r0,
         *
         * which over a quarter orbit, where n T = pi/2, is
         * 14.8 rho^2 / r0. As a fraction of the separation itself
         * that is 14.8 rho / r0. */
        printf("gate 7: against two-body motion at r0 = %.1f km, at"
               " two intervals so that no term of the solution can"
               " vanish at both\n", r0 / 1000.0);

        /* Two intervals, and the reason is a coincidence found while
         * red-checking this file. At exactly a quarter orbit the
         * cosine is zero, so any entry whose only appearance is a
         * coefficient on cos(nt) contributes nothing there and a
         * mutation of that coefficient leaves every number in this arm
         * unchanged. The structural arms above catch such a mutation
         * through the determinant, the matrix exponential and the
         * residual of the equations, so the gate set was never blind
         * to it, but an arm that cannot move under a defect is worth
         * no more than the fixture it runs on. The second interval is
         * not a quarter, a half or a whole orbit, so no trigonometric
         * term of the solution vanishes at it. */
        static const double fracs[] = { 0.25, 0.37 };
        static const double seps[] = { 400.0, 200.0, 100.0, 50.0 };
        for (int j = 0; j < 2; j++) {
        double interval = fracs[j] * 2.0 * M_PI / n;
        double icoeff = 6.0 * (fracs[j] * 2.0 * M_PI) * (fracs[j] * 2.0 * M_PI);
        double prev_err = 0.0;
        printf("  at %.2f of an orbit, bound coefficient %.4f\n",
               fracs[j], icoeff);
        for (int k = 0; k < 4; k++) {
            double rho = seps[k];
            K26V3 hold = k26m3d_v3(rho, 0.0, 0.0), hv;
            ASSERT(k26astro_prox_cw_hold(n, hold, &hv) == K26ASTRO_PROX_OK);
            double rel0[6] = { hold.x, hold.y, hold.z, hv.x, hv.y, hv.z };

            K26AstroProxRel s0 = { hold, hv }, cw;
            ASSERT(k26astro_prox_cw_propagate(n, interval, &s0, &cw)
                   == K26ASTRO_PROX_OK);

            double drift = 0.0;
            K26V3 tru = integrated_rel_(r0, rel0, interval, 0.05, &drift);
            double err = sqrt((cw.r.x - tru.x) * (cw.r.x - tru.x) +
                              (cw.r.y - tru.y) * (cw.r.y - tru.y) +
                              (cw.r.z - tru.z) * (cw.r.z - tru.z));
            double bound = icoeff * rho * rho / r0;

            printf("    rho = %6.1f m: disagreement %.6e m, bound %.6e m,"
                   " integrator's own %.3e m\n", rho, err, bound, drift);
            /* The integration must not be what is being measured. */
            ASSERT(drift * 100.0 < err);
            ASSERT(err < bound);
            if (k > 0) {
                double ratio = prev_err / err;
                printf("      halving the separation divided the"
                       " disagreement by %.3f\n", ratio);
                ASSERT(ratio > 3.0 && ratio < 5.0);
            }
            prev_err = err;
        }
        }
        n_pass++;
        printf("gate 7: the disagreement is second order in the"
               " separation and inside the stated bound: OK\n");
    }

    /* ---- 8. How far the model reaches --------------------------- *
     *
     * The quantity compared here is the disagreement as a fraction of
     * the separation, at a hundred metres and at two hundred
     * kilometres. Since the disagreement is quadratic in the
     * separation, that fraction is linear in it, and the arm requires
     * it to grow in proportion. That is what "the model has a range"
     * means in a number a reader can act on.
     *
     * The bound is printed at both separations, with the ratio the
     * disagreement bears to it, and it is deliberately not the arm's
     * acceptance: it grows quadratically too, so the ratio is nearly
     * constant and the disagreement stays inside it everywhere
     * two-body motion is still the question. A bound that cannot be
     * exceeded cannot show a limit, and asserting that it was would be
     * a claim the numbers on this line contradict. */
    {
        double r0 = 7.0e6;
        double quarter = (M_PI / 2.0) / n;
        double coeff = 6.0 * (M_PI / 2.0) * (M_PI / 2.0);
        static const double seps[] = { 100.0, 2.0e5 };
        double frac[2] = { 0.0, 0.0 };

        for (int k = 0; k < 2; k++) {
            double rho = seps[k];
            K26V3 hold = k26m3d_v3(rho, 0.0, 0.0), hv;
            ASSERT(k26astro_prox_cw_hold(n, hold, &hv) == K26ASTRO_PROX_OK);
            double rel0[6] = { hold.x, hold.y, hold.z, hv.x, hv.y, hv.z };
            K26AstroProxRel s0 = { hold, hv }, cw;
            ASSERT(k26astro_prox_cw_propagate(n, quarter, &s0, &cw)
                   == K26ASTRO_PROX_OK);
            double drift = 0.0;
            K26V3 tru = integrated_rel_(r0, rel0, quarter, 0.05, &drift);
            double err = sqrt((cw.r.x - tru.x) * (cw.r.x - tru.x) +
                              (cw.r.y - tru.y) * (cw.r.y - tru.y) +
                              (cw.r.z - tru.z) * (cw.r.z - tru.z));
            double bound = coeff * rho * rho / r0;
            frac[k] = err / rho;
            printf("gate 8: rho = %9.1f m (%.4f%% of the orbit radius):"
                   " disagreement %.4e m, %.4e of the separation;"
                   " bound %.4e m, disagreement is %.3f of it\n",
                   rho, 100.0 * rho / r0, err, frac[k], bound,
                   err / bound);
            ASSERT(drift * 100.0 < err);
            /* Inside the bound at both, which is the fact this arm
             * reports rather than the one an earlier version of it
             * claimed. */
            ASSERT(err < bound);
        }

        double grew = frac[1] / frac[0];
        double sep_ratio = seps[1] / seps[0];
        printf("  the separation grew by a factor of %.0f and the"
               " fractional disagreement by %.0f\n", sep_ratio, grew);
        printf("  at %.0f m the model predicts to %.4e of the"
               " separation; at %.0f km to %.4e of it\n",
               seps[0], frac[0], seps[1] / 1000.0, frac[1]);
        /* Linear growth of the fraction, to a wide tolerance, since
         * what is asserted is the order and not a coefficient. */
        ASSERT(grew > 0.5 * sep_ratio && grew < 2.0 * sep_ratio);
        /* And the far case is quantitatively useless, which is the
         * plain reading of a range: several per cent of the separation
         * rather than parts per hundred thousand. */
        ASSERT(frac[1] > 0.01);
        ASSERT(frac[0] < 1.0e-4);
        n_pass++;
        printf("gate 8: the disagreement is a fixed fraction of the"
               " separation per unit separation, so the model's reach"
               " is measured and not assumed: OK\n");
    }

    /* ---- 9. Refusals -------------------------------------------- */
    {
        double phi[36];
        K26AstroProxRel s = { { 1, 2, 3 }, { 0, 0, 0 } }, o;
        K26V3 v;
        ASSERT(k26astro_prox_cw_stm(n, 1.0, NULL) == K26ASTRO_PROX_E_NULL);
        ASSERT(k26astro_prox_cw_stm(0.0, 1.0, phi) == K26ASTRO_PROX_E_RANGE);
        ASSERT(k26astro_prox_cw_stm(-n, 1.0, phi) == K26ASTRO_PROX_E_RANGE);
        ASSERT(k26astro_prox_cw_propagate(n, 1.0, &s, NULL)
               == K26ASTRO_PROX_E_NULL);
        ASSERT(k26astro_prox_cw_propagate(0.0, 1.0, &s, &o)
               == K26ASTRO_PROX_E_RANGE);
        ASSERT(k26astro_prox_cw_hold(0.0, k26m3d_v3(1, 0, 0), &v)
               == K26ASTRO_PROX_E_RANGE);
        ASSERT(k26astro_prox_cw_hold(n, k26m3d_v3(1, 0, 0), NULL)
               == K26ASTRO_PROX_E_NULL);
        ASSERT(k26astro_prox_mean_motion(-1.0, 7.0e6) == 0.0);
        ASSERT(k26astro_prox_mean_motion(MU_EARTH, 0.0) == 0.0);
        /* The mean motion is the one the frame reports. */
        double mm = k26astro_prox_mean_motion(MU_EARTH, 7.0e6);
        printf("gate 9: mean motion at 7000 km: %.12e rad/s\n", mm);
        ASSERT(absd_(mm - n) < 1e-18);
        n_pass++;
        printf("gate 9: arguments outside the model's terms are"
               " refused: OK\n");
    }

    /* An aliased propagate must not read what it has written. */
    {
        K26AstroProxRel s = { { 10.0, 20.0, 30.0 }, { 0.1, 0.2, 0.3 } };
        K26AstroProxRel t = s;
        ASSERT(k26astro_prox_cw_propagate(n, 400.0, &s, &s)
               == K26ASTRO_PROX_OK);
        K26AstroProxRel u;
        ASSERT(k26astro_prox_cw_propagate(n, 400.0, &t, &u)
               == K26ASTRO_PROX_OK);
        ASSERT(s.r.x == u.r.x && s.r.y == u.r.y && s.r.z == u.r.z);
        ASSERT(s.v.x == u.v.x && s.v.y == u.v.y && s.v.z == u.v.z);
        printf("gate 10: propagating in place gives the same answer as"
               " propagating into a second state: OK\n");
        n_pass++;
    }

    printf("test_prox_cw: %d gates passed\n", n_pass);
    return 0;
}
