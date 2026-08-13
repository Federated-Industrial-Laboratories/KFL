/* dist.c - distribution primitives over the Philox core.
 *
 * Everything here is a fixed code path over IEEE basic operations
 * (add, subtract, multiply, divide, square root): no libm call, so a
 * draw is bit-identical across platforms and libm versions. Each
 * primitive consumes exactly one counter tick.
 *
 * The normal quantile is Wichura's AS 241 PPND16 (Applied Statistics
 * 37 no. 3, 1988), transcribed from the published Fortran; the
 * published hash sums of the coefficient mantissas are re-checked in
 * tests/test_k26rng_quantile.c, and the absolute error bound (at or
 * below 1e-9) is demonstrated there against externally computed
 * references. */
#include "k26rng.h"
#include "k26rng_internal.h"

#include <math.h>    /* sqrt only: an IEEE basic operation, exact per
                      * the standard, unlike the transcendentals this
                      * file deliberately avoids. */
#include <string.h>

double k26rng_uniform01(K26RngKey key, K26RngCoords c)
{
    uint64_t u = k26rng_u64(key, c);
    return (double)(u >> 11) * 0x1.0p-53;
}

double k26rng_uniform(K26RngKey key, K26RngCoords c, double a, double b)
{
    return a + k26rng_uniform01(key, c) * (b - a);
}

uint64_t k26rng_bounded(K26RngKey key, K26RngCoords c, uint64_t n)
{
    if (n == 0) return 0;
    return (uint64_t)(((unsigned __int128)k26rng_u64(key, c) * n) >> 64);
}

/* ln(x) for finite x > 0, basic operations only.
 *
 * x = m * 2^e with m in [0.5, 1) by bit manipulation (exact), m
 * shifted into [sqrt(1/2), sqrt(2)) so the series argument
 * s = (m-1)/(m+1) satisfies |s| <= 0.1716. Then
 * ln m = 2 atanh(s) = 2s (1 + s^2/3 + s^4/5 + ...), truncated at the
 * s^16 term, whose next term is below 2^-80 of the sum; and
 * ln x = ln m + e ln 2 with ln 2 as a compile-time constant. The
 * path is fixed, so the result is bit-stable even where the final
 * rounding is not correctly rounded as a whole. */
double k26rng_internal_ln(double x)
{
    static const double LN2 = 0x1.62e42fefa39efp-1;
    static const double SQRT_HALF = 0x1.6a09e667f3bcdp-1;
    uint64_t bits;
    double m, s, s2, series;
    int e;

    memcpy(&bits, &x, sizeof bits);
    e = (int)((bits >> 52) & 0x7FF) - 1022;
    bits = (bits & 0x000FFFFFFFFFFFFFull) | 0x3FE0000000000000ull;
    memcpy(&m, &bits, sizeof m);
    if (m < SQRT_HALF) {
        m = m * 2.0;
        e = e - 1;
    }

    s = (m - 1.0) / (m + 1.0);
    s2 = s * s;
    series = 1.0 / 17.0;
    series = series * s2 + 1.0 / 15.0;
    series = series * s2 + 1.0 / 13.0;
    series = series * s2 + 1.0 / 11.0;
    series = series * s2 + 1.0 / 9.0;
    series = series * s2 + 1.0 / 7.0;
    series = series * s2 + 1.0 / 5.0;
    series = series * s2 + 1.0 / 3.0;
    series = series * s2 + 1.0;

    return 2.0 * s * series + (double)e * LN2;
}

/* AS 241 PPND16 coefficients, transcribed from the published source.
 * The paper ships hash sums of the coefficient mantissas to check
 * transcription; test_k26rng_quantile.c recomputes them. */
static const double A[8] = {
    3.3871328727963666080e0,  1.3314166789178437745e2,
    1.9715909503065514427e3,  1.3731693765509461125e4,
    4.5921953931549871457e4,  6.7265770927008700853e4,
    3.3430575583588128105e4,  2.5090809287301226727e3
};
static const double B[8] = {
    1.0,                      4.2313330701600911252e1,
    6.8718700749205790830e2,  5.3941960214247511077e3,
    2.1213794301586595867e4,  3.9307895800092710610e4,
    2.8729085735721942674e4,  5.2264952788528545610e3
};
static const double C[8] = {
    1.42343711074968357734e0, 4.63033784615654529590e0,
    5.76949722146069140550e0, 3.64784832476320460504e0,
    1.27045825245236838258e0, 2.41780725177450611770e-1,
    2.27238449892691845833e-2, 7.74545014278341407640e-4
};
static const double D[8] = {
    1.0,                      2.05319162663775882187e0,
    1.67638483018380384940e0, 6.89767334985100004550e-1,
    1.48103976427480074590e-1, 1.51986665636164571966e-2,
    5.47593808499534494600e-4, 1.05075007164441684324e-9
};
static const double E[8] = {
    6.65790464350110377720e0, 5.46378491116411436990e0,
    1.78482653991729133580e0, 2.96560571828504891230e-1,
    2.65321895265761230930e-2, 1.24266094738807843860e-3,
    2.71155556874348757815e-5, 2.01033439929228813265e-7
};
static const double F[8] = {
    1.0,                      5.99832206555887937690e-1,
    1.36929880922735805310e-1, 1.48753612908506148525e-2,
    7.86869131145613259100e-4, 1.84631831751005468180e-5,
    1.42151175831644588870e-7, 2.04426310338993978564e-15
};

static double ratio_(const double *num, const double *den, double r)
{
    double p = ((((((num[7] * r + num[6]) * r + num[5]) * r + num[4]) * r
                  + num[3]) * r + num[2]) * r + num[1]) * r + num[0];
    double q = ((((((den[7] * r + den[6]) * r + den[5]) * r + den[4]) * r
                  + den[3]) * r + den[2]) * r + den[1]) * r + den[0];
    return p / q;
}

double k26rng_internal_quantile(double p)
{
    double q = p - 0.5;
    double r;

    if (q < 0.0 ? -q <= 0.425 : q <= 0.425) {
        r = 0.180625 - q * q;
        return q * ratio_(A, B, r);
    }
    r = q < 0.0 ? p : 1.0 - p;
    r = sqrt(-k26rng_internal_ln(r));
    if (r <= 5.0) {
        r = r - 1.6;
        r = ratio_(C, D, r);
    } else {
        r = r - 5.0;
        r = ratio_(E, F, r);
    }
    return q < 0.0 ? -r : r;
}

double k26rng_normal(K26RngKey key, K26RngCoords c)
{
    /* The quantile is finite only on (0, 1) while the uniform grid
     * includes zero, so the low grid bit is set before scaling: every
     * draw lands on an odd multiple of 2^-53 in (0, 1), with no
     * data-dependent branch to keep untestably rare. The grid
     * coarsening is 2^-53 scale, far below the demonstrated 1e-9
     * quantile accuracy. Documented in the header. */
    uint64_t u = k26rng_u64(key, c);
    double p = (double)((u >> 11) | 1u) * 0x1.0p-53;
    return k26rng_internal_quantile(p);
}
