/* test_k26rng_core.c - generator core conformance and coordinate gates.
 *
 * Acceptance: the Philox4x32-10 core reproduces the authors' published
 * known-answer vectors; the coordinate and key mappings are pinned;
 * draws are pure (order-free, state-free) and environment-independent;
 * the cursor equals the pure path and refuses loudly at the sentinel.
 *
 * Known-answer vectors: Random123 tests/kat_vectors (D. E. Shaw
 * Research, BSD licence), the three philox4x32-10 rows, fetched from
 * github.com/DEShawResearch/random123 at intake; the file's sha256 is
 * recorded with the work. The vectors list counter words v[0..3], key
 * words v[0..1], then the expected output words. */
#include "k26rng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

/* Build coordinates straight from raw counter words, inverting the
 * documented mapping (word 3 = stream<<16 | channel, then
 * environment, episode, draw). The known-answer rows exercise the raw
 * core, below the stream-class and sentinel preconditions, which are
 * caller contracts rather than core behaviour. */
static K26RngCoords coords_from_ctr_(uint32_t c0, uint32_t c1,
                                     uint32_t c2, uint32_t c3)
{
    K26RngCoords c;
    c.draw        = c0;
    c.episode     = c1;
    c.environment = c2;
    c.stream      = (uint16_t)(c3 >> 16);
    c.channel     = (uint16_t)(c3 & 0xFFFFu);
    return c;
}

static void expect_block_(uint64_t seed,
                          uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3,
                          uint32_t e0, uint32_t e1, uint32_t e2, uint32_t e3)
{
    K26RngBlock b = k26rng_block(k26rng_key(seed),
                                 coords_from_ctr_(c0, c1, c2, c3));
    ASSERT(b.v[0] == e0);
    ASSERT(b.v[1] == e1);
    ASSERT(b.v[2] == e2);
    ASSERT(b.v[3] == e3);
}

int main(void)
{
    /* ---- Known-answer vectors (philox4x32 10) --------------------- */
    /* ctr {0,0,0,0} key {0,0} */
    expect_block_(0x0000000000000000ull,
                  0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
                  0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u);
    /* ctr {ff..} key {ff..}; key words low-first, so seed is all ones */
    expect_block_(0xFFFFFFFFFFFFFFFFull,
                  0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                  0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu);
    /* ctr {243f6a88 85a308d3 13198a2e 03707344} key {a4093822 299f31d0}:
     * key word 0 is the seed's low half, so seed = 0x299F31D0A4093822.
     * This row also pins the whole coordinate mapping: draw, episode,
     * environment, and stream 0x0370 with channel 0x7344 land on
     * exactly the published counter words. */
    expect_block_(0x299F31D0A4093822ull,
                  0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u,
                  0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u);

    /* ---- u64 composition: first word high. ------------------------ */
    {
        K26RngCoords c = coords_from_ctr_(0, 0, 0, 0);
        uint64_t u = k26rng_u64(k26rng_key(0), c);
        ASSERT(u == ((0x6627e8d5ull << 32) | 0xe169c58dull));
    }

    /* ---- Purity: same coordinates, same bytes, any order. --------- */
    {
        K26RngKey key = k26rng_key(0x4B464C0000000001ull);
        K26RngCoords a = { K26RNG_STREAM_RESET_STATE, 0, 1, 2, 3 };
        K26RngCoords b = { K26RNG_STREAM_DOMAIN_RANDOM, 9, 8, 7, 6 };
        uint64_t a1 = k26rng_u64(key, a);
        uint64_t b1 = k26rng_u64(key, b);
        uint64_t b2 = k26rng_u64(key, b);
        uint64_t a2 = k26rng_u64(key, a);
        ASSERT(a1 == a2);
        ASSERT(b1 == b2);
        ASSERT(a1 != b1);   /* distinct coordinates draw apart */
    }

    /* ---- Vectorisation independence: environment 7's draw sequence
     * is byte-identical computed alone or interleaved with 63
     * neighbours drawing in between. ------------------------------- */
    {
        enum { N_DRAWS = 64, N_ENVS = 64, WATCHED = 7 };
        K26RngKey key = k26rng_key(0x4B464C0000000002ull);
        uint64_t alone[N_DRAWS], beside[N_DRAWS], sink = 0;
        K26RngCoords c = { K26RNG_STREAM_RESET_STATE, 0, WATCHED, 0, 0 };
        for (uint32_t d = 0; d < N_DRAWS; d++) {
            c.draw = d;
            alone[d] = k26rng_u64(key, c);
        }
        for (uint32_t d = 0; d < N_DRAWS; d++) {
            for (uint32_t env = 0; env < N_ENVS; env++) {
                K26RngCoords ci = { K26RNG_STREAM_RESET_STATE, 0, env, 0, d };
                uint64_t u = k26rng_u64(key, ci);
                if (env == WATCHED) beside[d] = u;
                else sink ^= u;
            }
        }
        ASSERT(memcmp(alone, beside, sizeof alone) == 0);
        ASSERT(sink != 0);   /* the neighbours really drew */
    }

    /* ---- Cursor equals the pure path, one tick per primitive. ----- */
    {
        K26RngKey key = k26rng_key(0x4B464C0000000003ull);
        K26RngCoords c0 = { K26RNG_STREAM_TASK, 2, 0, 5, 0 };
        K26RngCursor cur = k26rng_cursor(key, c0);
        K26RngCoords c = c0;
        uint64_t u, up;
        double d, dp;

        ASSERT(k26rng_cursor_u64(&cur, &u) == K26RNG_OK);
        up = k26rng_u64(key, c);           ASSERT(u == up);
        c.draw++;
        ASSERT(k26rng_cursor_uniform01(&cur, &d) == K26RNG_OK);
        dp = k26rng_uniform01(key, c);     ASSERT(d == dp);
        c.draw++;
        ASSERT(k26rng_cursor_uniform(&cur, -2.0, 3.0, &d) == K26RNG_OK);
        dp = k26rng_uniform(key, c, -2.0, 3.0);  ASSERT(d == dp);
        c.draw++;
        ASSERT(k26rng_cursor_bounded(&cur, 1000003, &u) == K26RNG_OK);
        up = k26rng_bounded(key, c, 1000003);    ASSERT(u == up);
        c.draw++;
        ASSERT(k26rng_cursor_normal(&cur, &d) == K26RNG_OK);
        dp = k26rng_normal(key, c);        ASSERT(d == dp);
        ASSERT(cur.c.draw == 5);
    }

    /* ---- Cursor refusals: loud, and state-preserving. ------------- */
    {
        K26RngKey key = k26rng_key(1);
        uint64_t u;
        double d;

        /* Exhaustion sentinel: the draw before it works, the sentinel
         * refuses, and the refused cursor is unchanged. */
        K26RngCoords c = { K26RNG_STREAM_RESET_STATE, 0, 0, 0, 0xFFFFFFFEu };
        K26RngCursor cur = k26rng_cursor(key, c);
        ASSERT(k26rng_cursor_u64(&cur, &u) == K26RNG_OK);
        ASSERT(cur.c.draw == 0xFFFFFFFFu);
        ASSERT(k26rng_cursor_u64(&cur, &u) == K26RNG_E_EXHAUSTED);
        ASSERT(cur.c.draw == 0xFFFFFFFFu);
        ASSERT(k26rng_cursor_normal(&cur, &d) == K26RNG_E_EXHAUSTED);

        /* Invalid stream class refuses. */
        K26RngCoords bad = { K26RNG_STREAM_INVALID, 0, 0, 0, 0 };
        K26RngCursor curb = k26rng_cursor(key, bad);
        ASSERT(k26rng_cursor_u64(&curb, &u) == K26RNG_E_BAD_ARG);
        ASSERT(curb.c.draw == 0);

        /* Zero bound refuses without advancing. */
        K26RngCoords ok = { K26RNG_STREAM_TASK, 0, 0, 0, 0 };
        K26RngCursor curz = k26rng_cursor(key, ok);
        ASSERT(k26rng_cursor_bounded(&curz, 0, &u) == K26RNG_E_BAD_ARG);
        ASSERT(curz.c.draw == 0);

        /* Null arguments refuse without advancing. */
        ASSERT(k26rng_cursor_u64(NULL, &u) == K26RNG_E_NULL);
        ASSERT(k26rng_cursor_u64(&curz, NULL) == K26RNG_E_NULL);
        ASSERT(curz.c.draw == 0);
    }

    /* ---- Status decoder covers every named value. ----------------- */
    ASSERT(strcmp(k26rng_status_str(K26RNG_OK), "ok") == 0);
    ASSERT(k26rng_status_str(K26RNG_E_NULL)[0] != '\0');
    ASSERT(k26rng_status_str(K26RNG_E_EXHAUSTED)[0] != '\0');
    ASSERT(k26rng_status_str(K26RNG_E_BAD_ARG)[0] != '\0');
    ASSERT(k26rng_status_str((K26RngStatus)999)[0] != '\0');

    printf("test_k26rng_core: all assertions passed "
           "(3 published vectors, mapping, purity, independence, cursor)\n");
    return 0;
}
