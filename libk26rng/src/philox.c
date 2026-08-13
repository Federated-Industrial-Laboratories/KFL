/* philox.c - Philox4x32-10 core, coordinate mapping, cursor, status.
 *
 * Constants and round structure transcribed from the authors'
 * reference implementation (Random123 philox.h, D. E. Shaw Research,
 * BSD-licensed); the published known-answer vectors pin the
 * transcription in tests/test_k26rng_core.c. The core is integer
 * only, so bit-identical results follow from the C semantics of
 * fixed-width unsigned arithmetic. */
#include "k26rng.h"

/* Multiplier and Weyl constants, Random123 philox.h. */
#define M0 ((uint32_t)0xD2511F53u)
#define M1 ((uint32_t)0xCD9E8D57u)
#define W0 ((uint32_t)0x9E3779B9u)
#define W1 ((uint32_t)0xBB67AE85u)

K26RngKey k26rng_key(uint64_t seed)
{
    K26RngKey k;
    k.k0 = (uint32_t)seed;
    k.k1 = (uint32_t)(seed >> 32);
    return k;
}

/* One Philox round: two 32x32->64 multiplies, cross-wired with the
 * key. Word order follows the reference exactly. */
static void round_(uint32_t c[4], const uint32_t k[2])
{
    uint64_t p0 = (uint64_t)M0 * c[0];
    uint64_t p1 = (uint64_t)M1 * c[2];
    uint32_t hi0 = (uint32_t)(p0 >> 32), lo0 = (uint32_t)p0;
    uint32_t hi1 = (uint32_t)(p1 >> 32), lo1 = (uint32_t)p1;
    uint32_t n0 = hi1 ^ c[1] ^ k[0];
    uint32_t n1 = lo1;
    uint32_t n2 = hi0 ^ c[3] ^ k[1];
    uint32_t n3 = lo0;
    c[0] = n0; c[1] = n1; c[2] = n2; c[3] = n3;
}

K26RngBlock k26rng_block(K26RngKey key, K26RngCoords c)
{
    uint32_t ctr[4];
    uint32_t k[2];
    K26RngBlock out;

    /* Counter layout, most significant word first: word 3 carries
     * stream class and channel, then environment, episode, draw.
     * Fixed here and pinned by the stability tests; changing it would
     * silently re-map every recorded draw. */
    ctr[0] = c.draw;
    ctr[1] = c.episode;
    ctr[2] = c.environment;
    ctr[3] = ((uint32_t)c.stream << 16) | (uint32_t)c.channel;
    k[0] = key.k0;
    k[1] = key.k1;

    /* Ten rounds: round with the initial key, then bump-and-round
     * nine times, the reference iteration order. */
    round_(ctr, k);
    for (int r = 1; r < 10; r++) {
        k[0] += W0;
        k[1] += W1;
        round_(ctr, k);
    }

    out.v[0] = ctr[0];
    out.v[1] = ctr[1];
    out.v[2] = ctr[2];
    out.v[3] = ctr[3];
    return out;
}

uint64_t k26rng_u64(K26RngKey key, K26RngCoords c)
{
    K26RngBlock b = k26rng_block(key, c);
    return ((uint64_t)b.v[0] << 32) | b.v[1];
}

const char *k26rng_status_str(K26RngStatus s)
{
    switch (s) {
    case K26RNG_OK:          return "ok";
    case K26RNG_E_NULL:      return "null pointer argument";
    case K26RNG_E_EXHAUSTED: return "draw index exhausted";
    case K26RNG_E_BAD_ARG:   return "invalid argument";
    }
    return "unknown status";
}

/* Cursor ----------------------------------------------------------- */

K26RngCursor k26rng_cursor(K26RngKey key, K26RngCoords c)
{
    K26RngCursor cur;
    cur.key = key;
    cur.c = c;
    return cur;
}

/* Shared admission check: every cursor draw refuses the invalid
 * stream class and the exhaustion sentinel before touching anything,
 * so a refused call leaves the cursor exactly as it was. */
static K26RngStatus admit_(const K26RngCursor *cur)
{
    if (!cur) return K26RNG_E_NULL;
    if (cur->c.stream == K26RNG_STREAM_INVALID) return K26RNG_E_BAD_ARG;
    if (cur->c.draw == 0xFFFFFFFFu) return K26RNG_E_EXHAUSTED;
    return K26RNG_OK;
}

K26RngStatus k26rng_cursor_u64(K26RngCursor *cur, uint64_t *out)
{
    K26RngStatus s = admit_(cur);
    if (s != K26RNG_OK) return s;
    if (!out) return K26RNG_E_NULL;
    *out = k26rng_u64(cur->key, cur->c);
    cur->c.draw += 1;
    return K26RNG_OK;
}

K26RngStatus k26rng_cursor_uniform01(K26RngCursor *cur, double *out)
{
    K26RngStatus s = admit_(cur);
    if (s != K26RNG_OK) return s;
    if (!out) return K26RNG_E_NULL;
    *out = k26rng_uniform01(cur->key, cur->c);
    cur->c.draw += 1;
    return K26RNG_OK;
}

K26RngStatus k26rng_cursor_uniform(K26RngCursor *cur, double a, double b,
                                   double *out)
{
    K26RngStatus s = admit_(cur);
    if (s != K26RNG_OK) return s;
    if (!out) return K26RNG_E_NULL;
    *out = k26rng_uniform(cur->key, cur->c, a, b);
    cur->c.draw += 1;
    return K26RNG_OK;
}

K26RngStatus k26rng_cursor_bounded(K26RngCursor *cur, uint64_t n,
                                   uint64_t *out)
{
    K26RngStatus s = admit_(cur);
    if (s != K26RNG_OK) return s;
    if (!out) return K26RNG_E_NULL;
    if (n == 0) return K26RNG_E_BAD_ARG;
    *out = k26rng_bounded(cur->key, cur->c, n);
    cur->c.draw += 1;
    return K26RNG_OK;
}

K26RngStatus k26rng_cursor_normal(K26RngCursor *cur, double *out)
{
    K26RngStatus s = admit_(cur);
    if (s != K26RNG_OK) return s;
    if (!out) return K26RNG_E_NULL;
    *out = k26rng_normal(cur->key, cur->c);
    cur->c.draw += 1;
    return K26RNG_OK;
}
