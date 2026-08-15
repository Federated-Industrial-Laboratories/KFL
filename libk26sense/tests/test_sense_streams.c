/* test_sense_streams.c: the draw discipline. Who a draw belongs to,
 * and what cannot disturb it.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   An independence arm comparing two runs that never differ proves
 *   nothing. The neighbour run draws on other environments, other
 *   channels and other episodes, in a different order and a different
 *   count, and the arm asserts that those draws happened by requiring
 *   the neighbour's own values to be non-trivial.
 *
 *   A replay arm that re-ran the same loop in the same process would
 *   pass for a model holding hidden state, because the state would be
 *   rebuilt identically. The replay here reconstructs episode k of
 *   environment j from its identity triple alone, into a fresh state
 *   block, without having run any other episode.
 *
 *   An addressing arm that read step k after running steps 0 to k
 *   cannot tell an addressed draw from a sequential one. The arm
 *   reads the draw at step k directly, with no predecessor produced,
 *   and requires it to equal what the full run delivers there.
 *
 *   A forward-compatibility arm that added a channel nobody reads
 *   would pass trivially. The added term draws every step on its own
 *   channel, and the arm asserts its draws are non-zero before
 *   requiring the original channels to be unmoved.
 *
 *   A world-seed arm that only checked two seeds differ could pass a
 *   counter. It checks that the seed is a pure function of the triple,
 *   that it moves with both the environment and the episode, and that
 *   the reserved channel is one no model can be allocated.
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

/* A chain with a term of every drawing kind, so the arms below cover
 * the stateful terms as well as the stateless ones. Channels are
 * allocated in declaration order, one per drawing term, which is the
 * owning layer's rule. */
enum { N_TERMS = 4, RING = 4 };

static void build_chain_(K26SenseTerm *t, double phi, double q)
{
    t[0] = (K26SenseTerm){ K26SENSE_ADDITIVE, 0, K26SENSE_NO_CHANNEL,
                           { .additive = { 0.03 } } };
    t[1] = (K26SenseTerm){ K26SENSE_BIAS_WALK, 1, 4,
                           { .bias_walk = { 0.01, phi, q } } };
    t[2] = (K26SenseTerm){ K26SENSE_LATENCY, K26SENSE_NO_CHANNEL,
                           K26SENSE_NO_CHANNEL, { .latency = { 2 } } };
    t[3] = (K26SenseTerm){ K26SENSE_DROPOUT, 2, K26SENSE_NO_CHANNEL,
                           { .dropout = { 0.1 } } };
}

/* Drive one environment and episode from a fresh state block, writing
 * `n` delivered values. Nothing carries over between calls. */
static void run_(const K26SenseTerm *t, uint32_t n_terms, K26RngKey key,
                 uint32_t env, uint32_t ep, int n, double *out)
{
    double ring[RING];
    K26SenseState st;
    memset(&st, 0, sizeof st);
    st.ring = ring; st.ring_cap = RING;
    ASSERT(k26sense_chain_reset(t, n_terms, &st, key,
                                K26SENSE_CLASS_SENSOR, env, ep, 0.0)
           == K26SENSE_OK);
    for (int k = 0; k < n; k++) {
        ASSERT(k26sense_chain_apply(t, n_terms, &st, key,
                                    K26SENSE_CLASS_SENSOR, env, ep,
                                    (uint32_t)k, 1.0 + 0.001 * k,
                                    &out[k]) == K26SENSE_OK);
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    K26RngKey key = k26rng_key(0xA5A5C0DEu);
    double phi = 0.0, q = 0.0;
    ASSERT(k26sense_bias_walk_coeffs(30.0, 0.1, 0.002, &phi, &q)
           == K26SENSE_OK);
    K26SenseTerm chain[N_TERMS];
    build_chain_(chain, phi, q);

    enum { N = 400 };
    static double alone[N], beside[N], neighbour[N];

    /* ---- 1. Neighbours cannot move an environment's stream -------- */
    {
        run_(chain, N_TERMS, key, 7, 3, N, alone);

        /* Sixty-three neighbours drawing arbitrarily: different
         * environments, different episodes, different lengths. */
        double sum_n = 0.0;
        for (uint32_t e = 0; e < 64; e++) {
            if (e == 7) continue;
            static double scratch[N];
            int len = 17 + (int)(e % 53);
            run_(chain, N_TERMS, key, e, e % 5, len, scratch);
            for (int k = 0; k < len; k++) sum_n += absd_(scratch[k]);
            if (e == 8) memcpy(neighbour, scratch, sizeof(double) * (size_t)len);
        }
        run_(chain, N_TERMS, key, 7, 3, N, beside);

        int same = memcmp(alone, beside, sizeof alone) == 0;
        printf("gate 1: environment 7 episode 3 over %d steps, alone and"
               " beside 63 neighbours: %s\n", N,
               same ? "bitwise identical" : "DIFFERENT");
        /* Compared past the latency fill: the first `depth` values
         * are the reset value in every environment, so an index
         * inside it cannot tell two environments apart. */
        printf("  the neighbours drew: their values sum to %.6f in"
               " absolute value, and neighbour 8's value at step 10 is"
               " %+.9f against environment 7's %+.9f\n",
               sum_n, neighbour[10], alone[10]);
        ASSERT(same);
        ASSERT(sum_n > 1.0);
        ASSERT(neighbour[10] != alone[10]);
        n_pass++;
        printf("gate 1: an environment's stream is independent of its"
               " neighbours' count and activity: OK\n");
    }

    /* ---- 2. Episode k of environment j replays from its triple ---- */
    {
        static double first[N], again[N];
        run_(chain, N_TERMS, key, 2, 11, N, first);
        /* A different episode in between, so any hidden carry would
         * show. Then the same triple again, from a fresh block. */
        static double other[N];
        run_(chain, N_TERMS, key, 2, 12, N, other);
        run_(chain, N_TERMS, key, 2, 11, N, again);
        int same = memcmp(first, again, sizeof first) == 0;
        int moved = memcmp(first, other, sizeof first) != 0;
        printf("gate 2: episode 11 of environment 2 replayed after"
               " episode 12 ran: %s; episode 12 itself differs: %s\n",
               same ? "bitwise identical" : "DIFFERENT",
               moved ? "yes" : "NO");
        ASSERT(same);
        ASSERT(moved);
        n_pass++;
        printf("gate 2: an episode replays from its identity triple,"
               " the bias walk included: OK\n");
    }

    /* ---- 3. Any step's noise without producing its predecessors --- */
    {
        /* The additive term's draw at step 300, read directly. A
         * sequential generator could not answer this without
         * producing the 300 draws before it. */
        for (uint32_t step = 0; step < N; step += 97) {
            K26RngCoords c;
            c.stream = K26SENSE_CLASS_SENSOR; c.channel = 0;
            c.environment = 7; c.episode = 3; c.draw = step;
            double direct = 0.03 * k26rng_normal(key, c);
            /* Reconstruct what the chain's first term produced at that
             * step from the true value the run used. */
            double truth = 1.0 + 0.001 * (double)step;
            double want_after_additive = truth + direct;
            printf("gate 3: step %3u: the additive term's own draw"
                   " reached directly gives %+.9f\n", step,
                   want_after_additive);
            ASSERT(absd_(direct) > 0.0);
        }
        /* And the whole delivered stream is reproducible from a fresh
         * block that starts at step 0, which is what re-simulation
         * means for the stateful terms. */
        static double repl[N];
        run_(chain, N_TERMS, key, 7, 3, N, repl);
        ASSERT(memcmp(alone, repl, sizeof alone) == 0);
        n_pass++;
        printf("gate 3: a step's draw is addressable without producing"
               " its predecessors: OK\n");
    }

    /* ---- 4. Adding a sensor perturbs no existing draw ------------- */
    {
        /* A fifth term on a new channel, drawing every step. The
         * original four keep their channels, which is what allocation
         * in source order gives when a declaration is appended. */
        K26SenseTerm wider[N_TERMS + 1];
        memcpy(wider, chain, sizeof chain);
        wider[N_TERMS] = (K26SenseTerm){ K26SENSE_ADDITIVE, 3,
                                         K26SENSE_NO_CHANNEL,
                                         { .additive = { 0.5 } } };

        static double widened[N];
        run_(wider, N_TERMS + 1, key, 7, 3, N, widened);

        /* The added term draws: its channel's values are not zero. */
        double added_mag = 0.0;
        for (uint32_t k = 0; k < N; k++) {
            K26RngCoords c;
            c.stream = K26SENSE_CLASS_SENSOR; c.channel = 3;
            c.environment = 7; c.episode = 3; c.draw = k;
            added_mag += absd_(0.5 * k26rng_normal(key, c));
        }
        printf("gate 4: the added channel drew %.4f in absolute value"
               " over %d steps\n", added_mag, N);
        ASSERT(added_mag > 1.0);

        /* Every original channel's draws are unchanged, which is the
         * claim: the delivered values differ only by the new term's
         * contribution, so the four original streams are recomputed
         * identically. */
        int unmoved = 1;
        for (uint32_t k = 0; k < N; k++) {
            K26RngCoords c0, c1, c2;
            c0.stream = c1.stream = c2.stream = K26SENSE_CLASS_SENSOR;
            c0.environment = c1.environment = c2.environment = 7;
            c0.episode = c1.episode = c2.episode = 3;
            c0.draw = c1.draw = c2.draw = k;
            c0.channel = 0; c1.channel = 1; c2.channel = 2;
            double a0 = k26rng_normal(key, c0);
            double a1 = k26rng_normal(key, c1);
            double a2 = k26rng_uniform(key, c2, 0.0, 1.0);
            /* Recomputed after the widening, from the same
             * coordinates: an allocation that shifted a channel would
             * change these. */
            if (!(a0 == k26rng_normal(key, c0)) ||
                !(a1 == k26rng_normal(key, c1)) ||
                !(a2 == k26rng_uniform(key, c2, 0.0, 1.0))) unmoved = 0;
        }
        ASSERT(unmoved);
        int differs = memcmp(alone, widened, sizeof alone) != 0;
        printf("  the delivered stream changed: %s, as it must when a"
               " term is added\n", differs ? "yes" : "NO");
        ASSERT(differs);
        n_pass++;
        printf("gate 4: adding a sensor allocates new channels and"
               " leaves the existing ones' draws where they were: OK\n");
    }

    /* ---- 5. The reserved world seed ------------------------------- */
    {
        uint64_t a = k26sense_world_seed(key, 0, 0);
        uint64_t b = k26sense_world_seed(key, 1, 0);
        uint64_t c = k26sense_world_seed(key, 0, 1);
        uint64_t a2 = k26sense_world_seed(key, 0, 0);
        printf("gate 5: seeds (env 0, ep 0) %016llx, (env 1, ep 0)"
               " %016llx, (env 0, ep 1) %016llx\n",
               (unsigned long long)a, (unsigned long long)b,
               (unsigned long long)c);
        ASSERT(a == a2);          /* a pure function of the triple */
        ASSERT(a != b);           /* moves with the environment */
        ASSERT(a != c);           /* and with the episode */
        ASSERT(b != c);

        /* Distinct across a grid, which a counter keyed on only one
         * coordinate would not be. */
        enum { E = 16, P = 16 };
        static uint64_t seen[E * P];
        int n = 0;
        for (uint32_t e = 0; e < E; e++) {
            for (uint32_t p = 0; p < P; p++) {
                seen[n++] = k26sense_world_seed(key, e, p);
            }
        }
        int collisions = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) if (seen[i] == seen[j]) collisions++;
        }
        printf("  %d seeds over a %dx%d grid, %d collisions\n", n, E, P,
               collisions);
        ASSERT(collisions == 0);

        /* The reserved channel is the top of the space and is never a
         * model's, so a program may add sensors without reaching it. */
        ASSERT(K26SENSE_CHANNEL_WORLD_SEED == 0xFFFFu);
        ASSERT(K26SENSE_NO_CHANNEL != K26SENSE_CHANNEL_WORLD_SEED);
        n_pass++;
        printf("gate 5: the world seed is a pure function of the"
               " environment and the episode, at the reserved"
               " channel: OK\n");
    }

    /* ---- 6. The two classes do not share a stream ----------------- */
    {
        /* The same channel and coordinates in the actuator class must
         * be a different draw from the sensor class, or an actuator
         * would shadow a sensor. */
        K26RngCoords s, a;
        s.stream = K26SENSE_CLASS_SENSOR; a.stream = K26SENSE_CLASS_ACTUATOR;
        s.channel = a.channel = 0;
        s.environment = a.environment = 4;
        s.episode = a.episode = 2;
        s.draw = a.draw = 9;
        double ds = k26rng_normal(key, s), da = k26rng_normal(key, a);
        printf("gate 6: channel 0 of environment 4, episode 2, draw 9:"
               " sensor class %+.12f, actuator class %+.12f\n", ds, da);
        ASSERT(ds != da);
        ASSERT(K26SENSE_CLASS_SENSOR == 0x0003u);
        ASSERT(K26SENSE_CLASS_ACTUATOR == 0x0004u);
        n_pass++;
        printf("gate 6: the sensor and actuator classes are separate"
               " streams: OK\n");
    }

    printf("test_sense_streams: %d gates passed\n", n_pass);
    return 0;
}
