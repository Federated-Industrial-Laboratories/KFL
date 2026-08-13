/* test_k26rng_dist.c - distribution primitives: pins, ranges, smoke.
 *
 * Acceptance: the distribution layer is bit-stable (17-digit pinned
 * references reproduce exactly), ranges hold, and first moments sit
 * inside loose envelopes. The pins prove stability across builds and
 * nothing about accuracy; accuracy is demonstrated separately in
 * test_k26rng_quantile.c against external references. Statistical
 * evidence beyond the smoke envelopes belongs to the algorithm's
 * published battery results, not to this tree's test time.
 *
 * Set CAPTURE_REFS in the environment to dump captured outputs in
 * C-source form for re-pinning after an intentional flag change. */
#include "k26rng.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

int main(void)
{
    const char *capture = getenv("CAPTURE_REFS");
    K26RngKey key = k26rng_key(0x4B464C5F524E4701ull);
    K26RngCoords c = { K26RNG_STREAM_RESET_STATE, 1, 2, 3, 4 };

    /* ---- Bit-stability pins (prove stability, not accuracy). ------ */
    {
        uint64_t u64 = k26rng_u64(key, c);
        double u01   = k26rng_uniform01(key, c);
        double uab   = k26rng_uniform(key, c, -2.0, 3.0);
        uint64_t bnd = k26rng_bounded(key, c, 1000000000039ull);
        double nrm   = k26rng_normal(key, c);

        if (capture) {
            printf("        ASSERT(u64 == UINT64_C(%llu));\n",
                   (unsigned long long)u64);
            printf("        ASSERT(u01 == %.17g);\n", u01);
            printf("        ASSERT(uab == %.17g);\n", uab);
            printf("        ASSERT(bnd == UINT64_C(%llu));\n",
                   (unsigned long long)bnd);
            printf("        ASSERT(nrm == %.17g);\n", nrm);
        } else {
            ASSERT(u64 == UINT64_C(16373554204057533500));
            ASSERT(u01 == 0.88761215196741705);
            ASSERT(uab == 2.4380607598370849);
            ASSERT(bnd == UINT64_C(887612152002));
            ASSERT(nrm == 1.2139267523865884);
        }
    }

    /* ---- Ranges. --------------------------------------------------- */
    {
        K26RngCursor cur = k26rng_cursor(key, c);
        for (int i = 0; i < 4096; i++) {
            double u, v;
            uint64_t b;
            ASSERT(k26rng_cursor_uniform01(&cur, &u) == K26RNG_OK);
            ASSERT(u >= 0.0 && u < 1.0);
            ASSERT(k26rng_cursor_uniform(&cur, -2.0, 3.0, &v) == K26RNG_OK);
            ASSERT(v >= -2.0 && v <= 3.0);
            ASSERT(k26rng_cursor_bounded(&cur, 6, &b) == K26RNG_OK);
            ASSERT(b < 6);
        }
        ASSERT(k26rng_bounded(key, c, 0) == 0);
        ASSERT(k26rng_bounded(key, c, 1) == 0);
    }

    /* ---- Statistical smoke: loose envelopes only. ------------------ */
    {
        enum { N = 200000 };
        K26RngCoords cs = { K26RNG_STREAM_EXPERIMENTAL, 0, 0, 0, 0 };
        K26RngCursor cur = k26rng_cursor(key, cs);
        double sum = 0.0, sumsq = 0.0;
        for (int i = 0; i < N; i++) {
            double u;
            ASSERT(k26rng_cursor_uniform01(&cur, &u) == K26RNG_OK);
            sum += u;
            sumsq += u * u;
        }
        double mean = sum / N;
        double var = sumsq / N - mean * mean;
        printf("uniform01 smoke: mean %.6f var %.6f\n", mean, var);
        ASSERT(fabs(mean - 0.5) < 0.01);
        ASSERT(fabs(var - 1.0 / 12.0) < 0.01);

        sum = 0.0;
        sumsq = 0.0;
        for (int i = 0; i < N; i++) {
            double z;
            ASSERT(k26rng_cursor_normal(&cur, &z) == K26RNG_OK);
            sum += z;
            sumsq += z * z;
        }
        mean = sum / N;
        var = sumsq / N - mean * mean;
        printf("normal smoke: mean %.6f var %.6f\n", mean, var);
        ASSERT(fabs(mean) < 0.02);
        ASSERT(fabs(var - 1.0) < 0.03);

        enum { M = 120000, NB = 6 };
        long buckets[NB] = { 0 };
        for (int i = 0; i < M; i++) {
            uint64_t b;
            ASSERT(k26rng_cursor_bounded(&cur, NB, &b) == K26RNG_OK);
            buckets[b]++;
        }
        for (int i = 0; i < NB; i++) {
            printf("bounded smoke: bucket %d count %ld\n", i, buckets[i]);
            ASSERT(labs(buckets[i] - M / NB) < 1000);
        }
    }

    if (capture) {
        fprintf(stderr, "test_k26rng_dist: CAPTURE_REFS set - re-run "
                "without it to validate against the pinned references.\n");
        return 0;
    }

    printf("test_k26rng_dist: all assertions passed "
           "(pins, ranges, moment envelopes)\n");
    return 0;
}
