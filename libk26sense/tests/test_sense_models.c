/* test_sense_models.c: each imperfection model against the behaviour
 * it is defined by.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   A noise arm that only checked the mean would pass for a model
 *   that added nothing at all, since the mean of the addition is
 *   zero. Every noise arm measures the standard deviation, and
 *   measures it against a stated envelope derived from the sample
 *   count rather than against a number chosen after the fact.
 *
 *   A latency arm over a constant signal cannot see a delay, because
 *   every sample equals every other. The signal used is strictly
 *   increasing and distinct at every step, so a delay of d shows as
 *   an exact index shift and a delay of the wrong depth cannot match.
 *
 *   A quantisation arm at positive values only would pass a
 *   truncation towards zero, which differs from the specified
 *   rounding exactly on the negative side. Negative values and exact
 *   ties are both checked, and the tie is checked on both signs.
 *
 *   A dropout arm that only counted how often the value was held
 *   would pass for a model that held at the wrong steps. The arm
 *   recomputes which steps must hold, from the same coordinates the
 *   model uses, and requires the held steps to be exactly those.
 *
 *   A bias-walk arm that checked only the stationary spread would
 *   pass for white noise of the same size, which has no correlation
 *   time at all. The arm recovers the correlation time from the
 *   realised lag-one autocorrelation and checks it against the
 *   declared one, and separately checks a long-lag correlation, so a
 *   process with the right spread and the wrong memory fails.
 *
 *   An arm run at one set of coordinates cannot show that draws are
 *   addressed rather than sequential. The independence arm reads the
 *   same step's noise directly, without producing the steps before
 *   it, and requires it to equal what a full run produces there.
 */
#include "k26sense.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

static double absd_(double x) { return x < 0.0 ? -x : x; }

static K26RngCoords co_(uint16_t ch, uint32_t env, uint32_t ep, uint32_t d)
{
    K26RngCoords c;
    c.stream = K26SENSE_CLASS_SENSOR;
    c.channel = ch; c.environment = env; c.episode = ep; c.draw = d;
    return c;
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    K26RngKey key = k26rng_key(0x5EED1234u);

    /* ---- 1. Additive noise ---------------------------------------- */
    {
        const double sigma = 0.05;
        const int n = 200000;
        K26SenseTerm t = { K26SENSE_ADDITIVE, 0, K26SENSE_NO_CHANNEL,
                           { .additive = { sigma } } };
        K26SenseState st; memset(&st, 0, sizeof st);
        ASSERT(k26sense_chain_reset(&t, 1, &st, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 0.0)
               == K26SENSE_OK);
        double sum = 0.0, sq = 0.0, worst = 0.0;
        for (int k = 0; k < n; k++) {
            double v;
            ASSERT(k26sense_chain_apply(&t, 1, &st, key,
                                        K26SENSE_CLASS_SENSOR, 0, 0,
                                        (uint32_t)k, 7.0, &v) == K26SENSE_OK);
            double e = v - 7.0;
            sum += e; sq += e * e;
            if (absd_(e) > worst) worst = absd_(e);
        }
        double mean = sum / n;
        double sd   = sqrt(sq / n - mean * mean);
        /* The envelope: the sample standard deviation of n draws has a
         * relative standard error of 1/sqrt(2n), so five of those is
         * the band a correct model stays inside. */
        double band = 5.0 / sqrt(2.0 * n);
        printf("gate 1: additive noise, %d samples: mean %+.6e,"
               " sd %.6f, declared %.6f, relative error %.3e,"
               " band %.3e\n", n, mean, sd, sigma,
               absd_(sd - sigma) / sigma, band);
        printf("  the largest excursion was %.4f, so the arm is not"
               " reading a constant\n", worst);
        ASSERT(absd_(sd - sigma) / sigma < band);
        ASSERT(absd_(mean) < 5.0 * sigma / sqrt((double)n));
        ASSERT(worst > 2.0 * sigma);
        n_pass++;
        printf("gate 1: additive noise carries the declared standard"
               " deviation: OK\n");
    }

    /* ---- 2. Scale-factor noise ------------------------------------ */
    {
        const double rel = 0.02, v0 = 12.0;
        const int n = 200000;
        K26SenseTerm t = { K26SENSE_SCALE, 0, K26SENSE_NO_CHANNEL,
                           { .scale = { rel } } };
        K26SenseState st; memset(&st, 0, sizeof st);
        ASSERT(k26sense_chain_reset(&t, 1, &st, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, v0)
               == K26SENSE_OK);
        double sq = 0.0;
        for (int k = 0; k < n; k++) {
            double v;
            ASSERT(k26sense_chain_apply(&t, 1, &st, key,
                                        K26SENSE_CLASS_SENSOR, 0, 0,
                                        (uint32_t)k, v0, &v) == K26SENSE_OK);
            double e = (v - v0) / v0;
            sq += e * e;
        }
        double sd = sqrt(sq / n);
        printf("gate 2: scale-factor noise, %d samples: relative sd"
               " %.6f, declared %.6f\n", n, sd, rel);
        ASSERT(absd_(sd - rel) / rel < 5.0 / sqrt(2.0 * n));
        /* Scale noise is proportional: at zero it must do nothing,
         * which an additive model would not respect. */
        double z;
        ASSERT(k26sense_chain_apply(&t, 1, &st, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 3u, 0.0,
                                    &z) == K26SENSE_OK);
        printf("  at a true value of zero it delivers %.17g\n", z);
        ASSERT(z == 0.0);
        n_pass++;
        printf("gate 2: scale-factor noise is proportional to the"
               " value: OK\n");
    }

    /* ---- 3. Latency ----------------------------------------------- */
    {
        for (uint32_t depth = 0; depth <= 4; depth++) {
            double ring[8];
            K26SenseTerm t = { K26SENSE_LATENCY, K26SENSE_NO_CHANNEL,
                               K26SENSE_NO_CHANNEL,
                               { .latency = { depth } } };
            K26SenseState st; memset(&st, 0, sizeof st);
            st.ring = ring; st.ring_cap = 8;
            uint32_t need = 0;
            ASSERT(k26sense_chain_check(&t, 1, &need) == K26SENSE_OK);
            ASSERT(need == depth);
            ASSERT(k26sense_chain_reset(&t, 1, &st, key,
                                        K26SENSE_CLASS_SENSOR, 0, 0, -1.0)
                   == K26SENSE_OK);
            /* A strictly increasing signal, so a delay is an index
             * shift and no two samples can be confused. */
            int mismatched = 0;
            for (uint32_t k = 0; k < 12; k++) {
                double v;
                ASSERT(k26sense_chain_apply(&t, 1, &st, key,
                                            K26SENSE_CLASS_SENSOR, 0, 0, k,
                                            (double)k, &v) == K26SENSE_OK);
                double want = (k < depth) ? -1.0 : (double)(k - depth);
                if (v != want) mismatched++;
            }
            printf("gate 3: depth %u: %d of 12 steps disagreed with an"
                   " exact %u-step shift\n", depth, mismatched, depth);
            ASSERT(mismatched == 0);
        }
        /* A depth beyond the ring the caller gave is refused, not
         * written past. */
        double small[2];
        K26SenseTerm big = { K26SENSE_LATENCY, K26SENSE_NO_CHANNEL,
                             K26SENSE_NO_CHANNEL, { .latency = { 5 } } };
        K26SenseState st2; memset(&st2, 0, sizeof st2);
        st2.ring = small; st2.ring_cap = 2;
        ASSERT(k26sense_chain_reset(&big, 1, &st2, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 0.0)
               == K26SENSE_E_CAPACITY);
        n_pass++;
        printf("gate 3: latency delays by exactly the declared number"
               " of steps, and refuses a ring too short: OK\n");
    }

    /* ---- 4. Quantisation ------------------------------------------ */
    {
        const double lsb = 0.01;
        struct { double in, want; } cases[] = {
            {  0.0,      0.0   },
            {  0.014,    0.01  },
            {  0.016,    0.02  },
            { -0.014,   -0.01  },
            { -0.016,   -0.02  },
            /* Ties, both signs: half away from zero. */
            {  0.015,    0.02  },
            { -0.015,   -0.02  },
            {  0.025,    0.03  },
            { -0.025,   -0.03  },
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            double got = k26sense_quantise(cases[i].in, lsb, -1e6, 1e6);
            printf("gate 4: q(%+.4f) = %+.4f, want %+.4f\n",
                   cases[i].in, got, cases[i].want);
            ASSERT(absd_(got - cases[i].want) < 1e-12);
        }
        /* Every output is an exact multiple of the step. */
        for (int k = -500; k <= 500; k++) {
            double v = (double)k * 0.0037;
            double q = k26sense_quantise(v, lsb, -1e6, 1e6);
            double n_steps = q / lsb;
            ASSERT(absd_(n_steps - (double)(int64_t)(n_steps < 0 ?
                   n_steps - 0.5 : n_steps + 0.5)) < 1e-9);
        }
        /* The declared clamp is what keeps the integer conversion in
         * range, so it is applied before rounding. */
        ASSERT(k26sense_quantise(5.0, lsb, -1.0, 1.0) == 1.0);
        ASSERT(k26sense_quantise(-5.0, lsb, -1.0, 1.0) == -1.0);
        /* A step of zero is the identity rather than a division. */
        ASSERT(k26sense_quantise(0.1234, 0.0, -1.0, 1.0) == 0.1234);
        n_pass++;
        printf("gate 4: quantisation lands on multiples of the step and"
               " rounds ties away from zero on both signs: OK\n");
    }

    /* ---- 5. Dropout ------------------------------------------------ */
    {
        const double p = 0.25;
        const uint16_t ch = 3;
        const int n = 40000;
        K26SenseTerm t = { K26SENSE_DROPOUT, ch, K26SENSE_NO_CHANNEL,
                           { .dropout = { p } } };
        K26SenseState st; memset(&st, 0, sizeof st);
        ASSERT(k26sense_chain_reset(&t, 1, &st, key,
                                    K26SENSE_CLASS_SENSOR, 1, 2, 0.0)
               == K26SENSE_OK);
        int held = 0, wrong = 0;
        double prev = 0.0;
        for (uint32_t k = 0; k < (uint32_t)n; k++) {
            double v, truth = 1.0 + (double)k;
            ASSERT(k26sense_chain_apply(&t, 1, &st, key,
                                        K26SENSE_CLASS_SENSOR, 1, 2, k,
                                        truth, &v) == K26SENSE_OK);
            /* Recomputed here from the same coordinates the model
             * uses, so the arm knows which steps must hold rather
             * than only how many. */
            int want_hold = k26rng_uniform(key, co_(ch, 1, 2, k), 0.0, 1.0) < p;
            double want = want_hold ? prev : truth;
            if (v != want) wrong++;
            if (want_hold) held++;
            prev = v;
        }
        double rate = (double)held / n;
        printf("gate 5: dropout at p = %.3f: held %d of %d (%.4f),"
               " and %d step(s) delivered the wrong value\n",
               p, held, n, rate, wrong);
        ASSERT(wrong == 0);
        ASSERT(absd_(rate - p) < 5.0 * sqrt(p * (1.0 - p) / n));
        n_pass++;
        printf("gate 5: dropout holds the previous value at exactly the"
               " drawn steps: OK\n");
    }

    /* ---- 6. The bias walk ----------------------------------------- */
    {
        /* The fixture is chosen so the estimate is tight: the run
         * covers 20000 correlation times, so the recovered time has a
         * relative standard error near 1/sqrt(20000), under one per
         * cent. A correlation time of ten seconds at a tenth of a
         * second per step is 100 steps of memory. */
        const double tau = 10.0, dt = 0.1, sigma = 0.004, sigma0 = 0.02;
        const int n = 2000000;
        double phi = 0.0, q = 0.0;
        ASSERT(k26sense_bias_walk_coeffs(tau, dt, sigma, &phi, &q)
               == K26SENSE_OK);
        printf("gate 6: tau %.1f s at dt %.2f s gives phi %.12f,"
               " q %.12e\n", tau, dt, phi, q);
        ASSERT(absd_(phi - exp(-dt / tau)) < 1e-15);
        ASSERT(absd_(q - sigma * sqrt(1.0 - phi * phi)) < 1e-18);

        K26SenseTerm t = { K26SENSE_BIAS_WALK, 5, 6,
                           { .bias_walk = { sigma0, phi, q } } };
        K26SenseState st; memset(&st, 0, sizeof st);
        ASSERT(k26sense_chain_reset(&t, 1, &st, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 0.0)
               == K26SENSE_OK);
        printf("  the turn-on bias drawn at reset is %+.6f against a"
               " declared %.3f\n", st.bias, sigma0);

        double s1 = 0.0, s2 = 0.0, lag1 = 0.0, lag_far = 0.0;
        double prev = 0.0;
        double *hist = (double *)malloc((size_t)n * sizeof(double));
        ASSERT(hist != NULL);
        for (uint32_t k = 0; k < (uint32_t)n; k++) {
            double v;
            ASSERT(k26sense_chain_apply(&t, 1, &st, key,
                                        K26SENSE_CLASS_SENSOR, 0, 0, k,
                                        0.0, &v) == K26SENSE_OK);
            hist[k] = v;
            s1 += v; s2 += v * v;
            if (k > 0) lag1 += v * prev;
            prev = v;
        }
        /* Discard the first ten correlation times, where the turn-on
         * bias has not yet decayed into the stationary distribution. */
        const int burn = 100 * 100;
        double m = 0.0, var = 0.0;
        for (int k = burn; k < n; k++) m += hist[k];
        m /= (n - burn);
        for (int k = burn; k < n; k++) var += (hist[k] - m) * (hist[k] - m);
        var /= (n - burn);
        double sd = sqrt(var);

        double c1 = 0.0;
        for (int k = burn; k < n - 1; k++) c1 += (hist[k] - m) * (hist[k+1] - m);
        c1 /= (n - 1 - burn);
        double rho1 = c1 / var;
        double tau_hat = -dt / log(rho1);

        const int far = 100;   /* one correlation time in steps */
        for (int k = burn; k < n - far; k++) {
            lag_far += (hist[k] - m) * (hist[k + far] - m);
        }
        lag_far /= (n - far - burn);
        double rho_far = lag_far / var;

        printf("  stationary sd %.6e against declared %.6e"
               " (relative %.4f)\n", sd, sigma, absd_(sd - sigma) / sigma);
        printf("  lag-one correlation %.6f recovers a correlation time"
               " of %.4f s against a declared %.1f s\n", rho1, tau_hat, tau);
        printf("  at one correlation time the correlation is %.6f,"
               " want %.6f\n", rho_far, exp(-1.0));
        ASSERT(absd_(sd - sigma) / sigma < 0.05);
        ASSERT(absd_(tau_hat - tau) / tau < 0.05);
        ASSERT(absd_(rho_far - exp(-1.0)) < 0.05);
        /* White noise of the same spread would have no memory at all,
         * which is what the two correlation checks rule out. */
        ASSERT(rho1 > 0.9);
        free(hist);
        n_pass++;
        printf("gate 6: the bias walk has the declared stationary"
               " spread and the declared memory: OK\n");
    }

    /* ---- 7. Deadband ----------------------------------------------- */
    {
        ASSERT(k26sense_deadband(0.05, 0.1, 0) == 0.0);
        ASSERT(k26sense_deadband(-0.05, 0.1, 0) == 0.0);
        ASSERT(k26sense_deadband(0.4, 0.1, 0) == 0.4);
        ASSERT(k26sense_deadband(-0.4, 0.1, 0) == -0.4);
        /* Exactly at the threshold the command passes: the model is
         * "below produces zero", not "at or below". */
        ASSERT(k26sense_deadband(0.1, 0.1, 0) == 0.1);
        /* The declared variant is continuous at the threshold. */
        double lo = k26sense_deadband(0.1, 0.1, 1);
        double hi = k26sense_deadband(1.0, 0.1, 1);
        double mid = k26sense_deadband(0.55, 0.1, 1);
        printf("gate 7: rescaled deadband maps %.2f to %.6f, %.2f to"
               " %.6f, and %.2f to %.6f\n", 0.1, lo, 0.55, mid, 1.0, hi);
        ASSERT(absd_(lo) < 1e-15);
        ASSERT(absd_(hi - 1.0) < 1e-15);
        ASSERT(absd_(mid - 0.5) < 1e-12);
        /* Sign is carried, not lost. */
        ASSERT(k26sense_deadband(-1.0, 0.1, 1) == -1.0);
        n_pass++;
        printf("gate 7: the deadband zeroes below the threshold and its"
               " declared variant is continuous at it: OK\n");
    }

    /* ---- 8. Chain order, and what refuses -------------------------- */
    {
        /* A quantiser before a noise term and after it are different
         * programs, and the chain must respect the order declared. */
        K26SenseTerm noisy_then_q[2] = {
            { K26SENSE_ADDITIVE, 0, K26SENSE_NO_CHANNEL,
              { .additive = { 0.05 } } },
            { K26SENSE_QUANTISE, K26SENSE_NO_CHANNEL, K26SENSE_NO_CHANNEL,
              { .quantise = { 0.01, -100.0, 100.0 } } }
        };
        K26SenseTerm q_then_noisy[2] = { noisy_then_q[1], noisy_then_q[0] };
        K26SenseState a, b;
        memset(&a, 0, sizeof a); memset(&b, 0, sizeof b);
        ASSERT(k26sense_chain_reset(noisy_then_q, 2, &a, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 0.0)
               == K26SENSE_OK);
        ASSERT(k26sense_chain_reset(q_then_noisy, 2, &b, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 0.0)
               == K26SENSE_OK);
        int on_grid_a = 0, on_grid_b = 0;
        for (uint32_t k = 0; k < 200; k++) {
            double va, vb;
            ASSERT(k26sense_chain_apply(noisy_then_q, 2, &a, key,
                                        K26SENSE_CLASS_SENSOR, 0, 0, k,
                                        1.0, &va) == K26SENSE_OK);
            ASSERT(k26sense_chain_apply(q_then_noisy, 2, &b, key,
                                        K26SENSE_CLASS_SENSOR, 0, 0, k,
                                        1.0, &vb) == K26SENSE_OK);
            if (absd_(va / 0.01 - (double)(int64_t)(va / 0.01 + 0.5)) < 1e-9)
                on_grid_a++;
            if (absd_(vb / 0.01 - (double)(int64_t)(vb / 0.01 + 0.5)) < 1e-9)
                on_grid_b++;
        }
        printf("gate 8: quantiser last leaves %d of 200 samples on the"
               " grid; quantiser first leaves %d\n", on_grid_a, on_grid_b);
        ASSERT(on_grid_a == 200);
        ASSERT(on_grid_b < 20);

        /* One state block carries one bias walk and one delay. */
        K26SenseTerm two_bias[2] = {
            { K26SENSE_BIAS_WALK, 0, 2, { .bias_walk = { 0.1, 0.9, 0.1 } } },
            { K26SENSE_BIAS_WALK, 1, 3, { .bias_walk = { 0.1, 0.9, 0.1 } } }
        };
        uint32_t need = 0;
        ASSERT(k26sense_chain_check(two_bias, 2, &need) == K26SENSE_E_CHAIN);
        K26SenseTerm two_delay[2] = {
            { K26SENSE_LATENCY, K26SENSE_NO_CHANNEL, K26SENSE_NO_CHANNEL,
              { .latency = { 1 } } },
            { K26SENSE_LATENCY, K26SENSE_NO_CHANNEL, K26SENSE_NO_CHANNEL,
              { .latency = { 2 } } }
        };
        ASSERT(k26sense_chain_check(two_delay, 2, &need) == K26SENSE_E_CHAIN);

        /* Coefficients outside the model's terms are refused rather
         * than producing an exponential of a nonsense argument. */
        double phi = 0.0, qq = 0.0;
        ASSERT(k26sense_bias_walk_coeffs(0.0, 0.1, 1.0, &phi, &qq)
               == K26SENSE_E_RANGE);
        ASSERT(k26sense_bias_walk_coeffs(10.0, 0.0, 1.0, &phi, &qq)
               == K26SENSE_E_RANGE);
        ASSERT(k26sense_bias_walk_coeffs(10.0, 0.1, -1.0, &phi, &qq)
               == K26SENSE_E_RANGE);
        ASSERT(k26sense_bias_walk_coeffs(10.0, 0.1, 1.0, NULL, &qq)
               == K26SENSE_E_NULL);
        n_pass++;
        printf("gate 8: the chain applies terms in the order declared,"
               " and refuses what one state block cannot carry: OK\n");
    }

    /* ---- 9. The two cadences do not share a channel --------------- */
    {
        /* A bias walk draws once per episode at index 0 and once per
         * step, whose first index is also 0. On one channel those two
         * draws would be the same number: the turn-on bias and the
         * first kick would be locked together for every episode of
         * every environment. Two channels is what the allocation rule
         * requires, and the chain refuses a term that does not carry
         * them. */
        K26SenseTerm same = { K26SENSE_BIAS_WALK, 4, 4,
                              { .bias_walk = { 0.02, 0.99, 0.001 } } };
        uint32_t need = 0;
        ASSERT(k26sense_chain_check(&same, 1, &need) == K26SENSE_E_CHAIN);
        K26SenseTerm none = { K26SENSE_BIAS_WALK, 4, K26SENSE_NO_CHANNEL,
                              { .bias_walk = { 0.02, 0.99, 0.001 } } };
        ASSERT(k26sense_chain_check(&none, 1, &need) == K26SENSE_E_CHAIN);
        K26SenseTerm ok = { K26SENSE_BIAS_WALK, 4, 5,
                            { .bias_walk = { 0.02, 0.99, 0.001 } } };
        ASSERT(k26sense_chain_check(&ok, 1, &need) == K26SENSE_OK);

        /* And with two channels the two draws are different numbers,
         * which is the property the refusal exists to protect. */
        K26SenseState st; memset(&st, 0, sizeof st);
        ASSERT(k26sense_chain_reset(&ok, 1, &st, key,
                                    K26SENSE_CLASS_SENSOR, 0, 0, 0.0)
               == K26SENSE_OK);
        double turn_on = st.bias;
        double first_kick = 0.001 * k26rng_normal(key, co_(4, 0, 0, 0));
        double same_chan  = 0.02 * k26rng_normal(key, co_(4, 0, 0, 0));
        printf("gate 9: turn-on bias %+.9f from channel 5, first step's"
               " kick %+.9f from channel 4\n", turn_on, first_kick);
        printf("  on one channel the turn-on would have been %+.9f,"
               " locked to the kick\n", same_chan);
        ASSERT(turn_on != 0.0);
        ASSERT(absd_(turn_on / 0.02 - first_kick / 0.001) > 1e-9);
        n_pass++;
        printf("gate 9: a term drawing at both cadences holds two"
               " channels, and is refused without them: OK\n");
    }

    printf("test_sense_models: %d gates passed\n", n_pass);
    return 0;
}
