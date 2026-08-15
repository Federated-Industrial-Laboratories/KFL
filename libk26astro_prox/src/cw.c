/* cw.c: the linearised relative motion about a circular reference
 * orbit, and its state transition matrix.
 *
 * The equations, derived rather than transcribed.
 * ----------------------------------------------
 * Let the chief travel a circular orbit of radius R about a body of
 * gravitational parameter mu, so its frame turns at the constant rate
 * n = sqrt(mu / R^3) about e3. For a deputy at r = R e1 + rho, write
 * rho = (x, y, z) in the chief's axes. In a frame turning at
 * w = n e3 the deputy's relative acceleration obeys
 *
 *     rho'' + 2 w x rho' + w x (w x rho) = dg,
 *
 * the chief's own free fall having cancelled the leading terms, where
 * dg is the difference between the gravity at the deputy and at the
 * chief. Expanding dg to first order in |rho|/R,
 *
 *     dg = -(mu/R^3) [ rho - 3 (e1 . rho) e1 ] = n^2 (2x, -y, -z).
 *
 * With w = (0, 0, n): 2 w x rho' = (-2n y', 2n x', 0) and
 * w x (w x rho) = (-n^2 x, -n^2 y, 0). Componentwise,
 *
 *     x'' - 2n y' - n^2 x =  2 n^2 x     ->  x'' = 3 n^2 x + 2 n y'
 *     y'' + 2n x' - n^2 y = -n^2 y       ->  y'' = -2 n x'
 *     z''                 = -n^2 z       ->  z'' = -n^2 z
 *
 * which are the Clohessy-Wiltshire equations, x radial, y along-track,
 * z along the angular momentum.
 *
 * The solution, integrated from those equations.
 * ----------------------------------------------
 * The second equation integrates once to y' = y0' - 2n (x - x0).
 * Substituting into the first,
 *
 *     x'' + n^2 x = 2 n y0' + 4 n^2 x0,
 *
 * a harmonic oscillator with a constant forcing, whose solution under
 * the initial conditions is
 *
 *     x(t)  = (4 - 3c) x0 + (s/n) x0' + (2(1-c)/n) y0'
 *     x'(t) = 3 n s x0 + c x0' + 2 s y0'
 *
 * with c = cos(nt), s = sin(nt). Putting x(t) back into the once
 * integrated second equation gives y'(t), and integrating that from y0
 * gives y(t):
 *
 *     y'(t) = -6 n (1-c) x0 - 2 s x0' + (4c - 3) y0'
 *     y(t)  = y0 + 6 (s - nt) x0 - (2(1-c)/n) x0' + ((4s - 3nt)/n) y0'
 *
 * The third equation is a plain oscillator: z(t) = c z0 + (s/n) z0',
 * z'(t) = -n s z0 + c z0'.
 *
 * Those six expressions are the rows written below. The system matrix
 * has zero trace, so the matrix has unit determinant at every
 * interval; the coefficients are constant, so the matrix composes as a
 * semigroup. Both, the identity at zero interval, and agreement with
 * the matrix exponential of the system matrix are what
 * tests/test_prox_cw.c checks, and they are what the implementation
 * rests on rather than on a transcription being faithful.
 */
#include <math.h>

#include "k26astro_prox/prox.h"

static int cw_finite_(double x)
{
    return (x == x) && (x - x == 0.0);
}

double k26astro_prox_mean_motion(double mu, double radius)
{
    if (!cw_finite_(mu) || !cw_finite_(radius)) return 0.0;
    if (!(mu > 0.0) || !(radius > 0.0)) return 0.0;
    return sqrt(mu / (radius * radius * radius));
}

/* Row-major 6 by 6; state order r_x, r_y, r_z, v_x, v_y, v_z. */
#define PHI(row, col) phi[(row) * 6 + (col)]

K26AstroProxStatus k26astro_prox_cw_stm(double n, double t, double *phi)
{
    if (!phi) return K26ASTRO_PROX_E_NULL;
    if (!cw_finite_(n) || !cw_finite_(t) || !(n > 0.0)) {
        return K26ASTRO_PROX_E_RANGE;
    }

    double tau = n * t;
    double s   = sin(tau);
    double c   = cos(tau);

    for (int i = 0; i < 36; i++) phi[i] = 0.0;

    PHI(0, 0) = 4.0 - 3.0 * c;
    PHI(0, 3) = s / n;
    PHI(0, 4) = 2.0 * (1.0 - c) / n;

    PHI(1, 0) = 6.0 * (s - tau);
    PHI(1, 1) = 1.0;
    PHI(1, 3) = -2.0 * (1.0 - c) / n;
    PHI(1, 4) = (4.0 * s - 3.0 * tau) / n;

    PHI(2, 2) = c;
    PHI(2, 5) = s / n;

    PHI(3, 0) = 3.0 * n * s;
    PHI(3, 3) = c;
    PHI(3, 4) = 2.0 * s;

    PHI(4, 0) = -6.0 * n * (1.0 - c);
    PHI(4, 3) = -2.0 * s;
    PHI(4, 4) = 4.0 * c - 3.0;

    PHI(5, 2) = -n * s;
    PHI(5, 5) = c;

    return K26ASTRO_PROX_OK;
}

K26AstroProxStatus k26astro_prox_cw_propagate(double n, double t,
                                              const K26AstroProxRel *in,
                                              K26AstroProxRel *out)
{
    if (!in || !out) return K26ASTRO_PROX_E_NULL;

    double phi[36];
    K26AstroProxStatus st = k26astro_prox_cw_stm(n, t, phi);
    if (st != K26ASTRO_PROX_OK) return st;

    const double x[6] = { in->r.x, in->r.y, in->r.z,
                          in->v.x, in->v.y, in->v.z };
    double y[6];
    for (int i = 0; i < 6; i++) {
        double acc = 0.0;
        for (int j = 0; j < 6; j++) acc += PHI(i, j) * x[j];
        y[i] = acc;
    }
    /* Written after the sum, so `out` may alias `in`. */
    out->r = k26m3d_v3(y[0], y[1], y[2]);
    out->v = k26m3d_v3(y[3], y[4], y[5]);
    return K26ASTRO_PROX_OK;
}

K26AstroProxStatus k26astro_prox_cw_hold(double n, K26V3 hold, K26V3 *out_v)
{
    if (!out_v) return K26ASTRO_PROX_E_NULL;
    if (!cw_finite_(n) || !(n > 0.0)) return K26ASTRO_PROX_E_RANGE;
    if (!cw_finite_(hold.x) || !cw_finite_(hold.y) || !cw_finite_(hold.z)) {
        return K26ASTRO_PROX_E_RANGE;
    }

    /* The along-track solution above carries the secular term
     * -(6 n x0 + 3 y0') t. Setting y0' = -2 n x0 removes it, which is
     * the whole of the bounded-motion condition; nothing else in the
     * solution grows without bound. No sine or cosine is reached, so a
     * reset placed through this function is reproducible across
     * platforms as well as across runs. */
    *out_v = k26m3d_v3(0.0, -2.0 * n * hold.x, 0.0);
    return K26ASTRO_PROX_OK;
}

#undef PHI
