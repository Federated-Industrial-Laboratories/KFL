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

/* The hash sums check the arrays the library computes with, reached
 * through the internal seam; only each coefficient's power-of-ten
 * exponent lives here, read from the published source. B[0], D[0],
 * F[0] are the rational forms' fixed 1.0 and are not summed, matching
 * the paper's fifteen-coefficient sums. */
static const int EXP_A[8] = { 0, 2, 3, 4, 4, 4, 4, 3 };
static const int EXP_B[8] = { 0, 1, 2, 3, 4, 4, 4, 3 };
static const int EXP_C[8] = { 0, 0, 0, 0, 0, -1, -2, -4 };
static const int EXP_D[8] = { 0, 0, 0, -1, -1, -2, -4, -9 };
static const int EXP_E[8] = { 0, 0, 0, -1, -2, -3, -5, -7 };
static const int EXP_F[8] = { 0, -1, -1, -2, -4, -5, -7, -15 };

static double pow10i_(int e)
{
    double r = 1.0;
    int n = e < 0 ? -e : e;
    for (int i = 0; i < n; i++) r *= 10.0;
    return e < 0 ? 1.0 / r : r;
}

/* Sum the mantissas of a numerator array (all eight) and a
 * denominator array (skipping its fixed leading 1.0). */
static double mantissa_sum_(const double *num, const int *num_exp,
                            const double *den, const int *den_exp)
{
    double s = 0.0;
    for (int i = 0; i < 8; i++) s += num[i] / pow10i_(num_exp[i]);
    for (int i = 1; i < 8; i++) s += den[i] / pow10i_(den_exp[i]);
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
    /* ---- Transcription: the paper's mantissa hash sums, computed
     * over the library's own arrays. ------------------------------- */
    {
        double ab = mantissa_sum_(k26rng_internal_qA, EXP_A,
                                  k26rng_internal_qB, EXP_B);
        double cd = mantissa_sum_(k26rng_internal_qC, EXP_C,
                                  k26rng_internal_qD, EXP_D);
        double ef = mantissa_sum_(k26rng_internal_qE, EXP_E,
                                  k26rng_internal_qF, EXP_F);
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
        /* The conformance bound. */
        ASSERT(worst <= 1e-9);
        /* Regression tripwire, distinct from conformance: the measured
         * worst is 1.78e-15, and a change that degrades accuracy by
         * orders of magnitude while staying conformant should be loud,
         * not silent. Loosening this line is a deliberate act. */
        ASSERT(worst <= 5e-15);
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
