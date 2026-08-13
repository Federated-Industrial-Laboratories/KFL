/* test_k26rng_quantile.c - quantile transcription and accuracy gates.
 *
 * Acceptance: the AS 241 coefficient transcription reproduces the
 * paper's published mantissa hash sums; the in-house logarithm and
 * the PPND16 quantile agree with externally computed high-precision
 * references to the stated bounds (quantile absolute error at or
 * below 1e-9). References were computed with mpmath 1.3.0 at 50
 * significant digits and emitted as hex doubles; the generating
 * command is recorded with the work. Hash sums are from the published
 * AS 241 source (Wichura, Applied Statistics 37 no. 3, 1988), which
 * ships them for exactly this transcription check.
 *
 * This gate demonstrates accuracy; bit-stability across builds is
 * separately pinned in test_k26rng_dist.c, and the two are not
 * conflated. */
#include "k26rng.h"
#include "k26rng_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

/* The coefficients again, with each one's power-of-ten exponent, so
 * the hash sums check the test's own copy against the paper and the
 * library's copy against the test through the accuracy sweep. */
typedef struct { double v; int exp10; } CoefRow;

static const CoefRow AB[] = {
    { 3.3871328727963666080e0, 0 },  { 1.3314166789178437745e2, 2 },
    { 1.9715909503065514427e3, 3 },  { 1.3731693765509461125e4, 4 },
    { 4.5921953931549871457e4, 4 },  { 6.7265770927008700853e4, 4 },
    { 3.3430575583588128105e4, 4 },  { 2.5090809287301226727e3, 3 },
    { 4.2313330701600911252e1, 1 },  { 6.8718700749205790830e2, 2 },
    { 5.3941960214247511077e3, 3 },  { 2.1213794301586595867e4, 4 },
    { 3.9307895800092710610e4, 4 },  { 2.8729085735721942674e4, 4 },
    { 5.2264952788528545610e3, 3 },
};
static const CoefRow CD[] = {
    { 1.42343711074968357734e0, 0 },  { 4.63033784615654529590e0, 0 },
    { 5.76949722146069140550e0, 0 },  { 3.64784832476320460504e0, 0 },
    { 1.27045825245236838258e0, 0 },  { 2.41780725177450611770e-1, -1 },
    { 2.27238449892691845833e-2, -2 }, { 7.74545014278341407640e-4, -4 },
    { 2.05319162663775882187e0, 0 },  { 1.67638483018380384940e0, 0 },
    { 6.89767334985100004550e-1, -1 }, { 1.48103976427480074590e-1, -1 },
    { 1.51986665636164571966e-2, -2 }, { 5.47593808499534494600e-4, -4 },
    { 1.05075007164441684324e-9, -9 },
};
static const CoefRow EF[] = {
    { 6.65790464350110377720e0, 0 },  { 5.46378491116411436990e0, 0 },
    { 1.78482653991729133580e0, 0 },  { 2.96560571828504891230e-1, -1 },
    { 2.65321895265761230930e-2, -2 }, { 1.24266094738807843860e-3, -3 },
    { 2.71155556874348757815e-5, -5 }, { 2.01033439929228813265e-7, -7 },
    { 5.99832206555887937690e-1, -1 }, { 1.36929880922735805310e-1, -1 },
    { 1.48753612908506148525e-2, -2 }, { 7.86869131145613259100e-4, -4 },
    { 1.84631831751005468180e-5, -5 }, { 1.42151175831644588870e-7, -7 },
    { 2.04426310338993978564e-15, -15 },
};

static double pow10i_(int e)
{
    double r = 1.0;
    int n = e < 0 ? -e : e;
    for (int i = 0; i < n; i++) r *= 10.0;
    return e < 0 ? 1.0 / r : r;
}

static double mantissa_sum_(const CoefRow *t, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += t[i].v / pow10i_(t[i].exp10);
    return s;
}

/* mpmath 1.3.0, 50 digits: z = sqrt(2) * erfinv(2p - 1) at the exact
 * double p, rounded once to double, emitted as hex. */
static const struct { double p; double ref; } REFS[] = {
    { 0x1.0000000000000p-53, -0x1.06b48528cea52p+3 },
    { 0x1.cd2b297d889bcp-54, -0x1.071b4c29e9b5ap+3 },
    { 0x1.19799812dea11p-40, -0x1.c234fba57a32ap+2 },
    { 0x1.12e0be826d695p-30, -0x1.7fdc11f44b5a7p+2 },
    { 0x1.0c6f7a0b5ed8dp-20, -0x1.30381a9799f7ep+2 },
    { 0x1.a36e2eb1c432dp-14, -0x1.dc08bb712893bp+1 },
    { 0x1.0624dd2f1a9fcp-10, -0x1.8b8cbb7204471p+1 },
    { 0x1.47ae147ae147bp-7, -0x1.29c5c4630ff0fp+1 },
    { 0x1.999999999999ap-5, -0x1.a515209676abdp+0 },
    { 0x1.2f1a9fbe76c8bp-4, -0x1.72567aa9caedap+0 },
    { 0x1.3333333333333p-4, -0x1.7085226d3e524p+0 },
    { 0x1.374bc6a7ef9dbp-4, -0x1.6eb87f9160bcbp+0 },
    { 0x1.999999999999ap-4, -0x1.4813c36e26d32p+0 },
    { 0x1.999999999999ap-3, -0x1.aee8fa73a1333p-1 },
    { 0x1.0000000000000p-2, -0x1.5956b87528a49p-1 },
    { 0x1.5555555555555p-2, -0x1.b91093bfdfa2bp-2 },
    { 0x1.999999999999ap-2, -0x1.036d6c4a04b59p-2 },
    { 0x1.0000000000000p-1, 0x0.0p+0 },
    { 0x1.3333333333333p-1, 0x1.036d6c4a04b59p-2 },
    { 0x1.5555555555555p-1, 0x1.b91093bfdfa28p-2 },
    { 0x1.8000000000000p-1, 0x1.5956b87528a49p-1 },
    { 0x1.999999999999ap-1, 0x1.aee8fa73a1335p-1 },
    { 0x1.ccccccccccccdp-1, 0x1.4813c36e26d33p+0 },
    { 0x1.d916872b020c5p-1, 0x1.6eb87f9160bccp+0 },
    { 0x1.d99999999999ap-1, 0x1.7085226d3e525p+0 },
    { 0x1.da1cac083126fp-1, 0x1.72567aa9caedbp+0 },
    { 0x1.e666666666666p-1, 0x1.a515209676abbp+0 },
    { 0x1.fae147ae147aep-1, 0x1.29c5c4630ff0ep+1 },
    { 0x1.ff7ced916872bp-1, 0x1.8b8cbb7204470p+1 },
    { 0x1.fff2e48e8a71ep-1, 0x1.dc08bb712897ap+1 },
    { 0x1.ffffde7210be9p-1, 0x1.30381a97985efp+2 },
    { 0x1.fffffff768fa1p-1, 0x1.7fdc11f93a210p+2 },
    { 0x1.fffffffffdcd1p-1, 0x1.c2350895b2ea4p+2 },
    { 0x1.fffffffffffffp-1, 0x1.06b48528cea52p+3 },
};

/* mpmath 1.3.0, 50 digits: ln(x) at the exact double x. */
static const struct { double x; double ref; } LN_REFS[] = {
    { 0x1.0000000000000p-53, -0x1.25e4f7b2737fap+5 },
    { 0x1.19799812dea11p-40, -0x1.ba18a998fffa0p+4 },
    { 0x1.12e0be826d695p-30, -0x1.4b927f32bffb8p+4 },
    { 0x1.a36e2eb1c432dp-14, -0x1.26bb1bbb55515p+3 },
    { 0x1.999999999999ap-4, -0x1.26bb1bbb55515p+1 },
    { 0x1.ae147ae147ae1p-2, -0x1.bc2908cf1b3f9p-1 },
    { 0x1.0000000000000p-1, -0x1.62e42fefa39efp-1 },
    { 0x1.6a09e65dc27dfp-1, -0x1.62e4300c77ae8p-2 },
    { 0x1.fffffca501acbp-1, -0x1.ad7f2b1049b9fp-24 },
};

int main(void)
{
    /* ---- Transcription: the paper's mantissa hash sums. ------------ */
    {
        double ab = mantissa_sum_(AB, 15);
        double cd = mantissa_sum_(CD, 15);
        double ef = mantissa_sum_(EF, 15);
        printf("hash sums: AB %.17g CD %.17g EF %.17g\n", ab, cd, ef);
        ASSERT(fabs(ab - 55.8831928806149014439) < 1e-12);
        ASSERT(fabs(cd - 49.3320650330161028904) < 1e-12);
        ASSERT(fabs(ef - 47.5258331754928967163) < 1e-12);
    }

    /* ---- In-house logarithm against external references. ---------- */
    {
        double worst = 0.0;
        for (unsigned i = 0; i < sizeof LN_REFS / sizeof LN_REFS[0]; i++) {
            double got = k26rng_internal_ln(LN_REFS[i].x);
            double rel = fabs(got - LN_REFS[i].ref) /
                         (fabs(LN_REFS[i].ref) > 1.0 ? fabs(LN_REFS[i].ref)
                                                     : 1.0);
            if (rel > worst) worst = rel;
        }
        printf("ln worst error (relative above 1, absolute below): %.3g\n",
               worst);
        ASSERT(worst < 1e-14);
    }

    /* ---- Quantile absolute error at or below 1e-9. ----------------- */
    {
        double worst = 0.0;
        for (unsigned i = 0; i < sizeof REFS / sizeof REFS[0]; i++) {
            double got = k26rng_internal_quantile(REFS[i].p);
            double err = fabs(got - REFS[i].ref);
            if (err > worst) worst = err;
        }
        printf("quantile worst absolute error: %.3g over %zu points\n",
               worst, sizeof REFS / sizeof REFS[0]);
        ASSERT(worst <= 1e-9);
    }

    /* ---- The draw path lands on the quantile of its own grid
     * probability: normal == quantile((u64 >> 11 | 1) * 2^-53). ----- */
    {
        K26RngKey key = k26rng_key(0x4B464C0000000004ull);
        for (uint32_t d = 0; d < 256; d++) {
            K26RngCoords c = { K26RNG_STREAM_SENSOR_NOISE, 3, 1, 0, d };
            uint64_t u = k26rng_u64(key, c);
            double p = (double)((u >> 11) | 1u) * 0x1.0p-53;
            ASSERT(k26rng_normal(key, c) == k26rng_internal_quantile(p));
        }
    }

    printf("test_k26rng_quantile: all assertions passed "
           "(transcription sums, ln, quantile accuracy, draw path)\n");
    return 0;
}
