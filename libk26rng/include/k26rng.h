/* libk26rng - deterministic counter-based sampling (Philox4x32-10).
 *
 * A draw is a pure function of a 64-bit key and a 128-bit counter, so
 * any single draw is addressable by its coordinates without generating
 * a predecessor, parallel consumers never share state, and replay is
 * exact by construction. The generator core is integer-only; the
 * distribution layer uses only IEEE basic operations (add, subtract,
 * multiply, divide, square root), never libm, so results are
 * bit-identical across platforms and libm versions.
 *
 * Algorithm: Philox4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel
 * Random Numbers: As Easy as 1, 2, 3", SC'11, 2011). Ten rounds,
 * fixed. Conformance is anchored to the authors' published
 * known-answer vectors, pinned in this library's tests.
 *
 * Coordinates map onto the counter most significant word first:
 *
 *   word 3   stream class (high 16 bits) | channel (low 16 bits)
 *   word 2   environment index
 *   word 1   episode index
 *   word 0   draw index
 *
 * and the key holds the 64-bit seed, low word first (k0 = seed low 32
 * bits, k1 = seed high 32 bits). Stream class 0x0000 is reserved
 * invalid. Draw index 0xFFFFFFFF is the exhaustion sentinel: the
 * cursor refuses there, and passing it to a pure function is a
 * documented precondition violation, so every (key, coordinate) pair
 * a conforming caller can draw is drawn at most once.
 *
 * Every distribution primitive consumes exactly one counter tick per
 * sample; rejection sampling and other variable-consumption methods
 * are excluded by design, so streams stay decoupled under
 * vectorisation.
 *
 * Threading: everything here is a pure function of its arguments; a
 * K26RngCursor is a local value owned by its caller. No global state
 * exists in this library. */
#ifndef K26RNG_H
#define K26RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define K26RNG_LIB_VERSION "0.1.0"

/* The 64-bit governing seed, split for the generator core. Build one
 * with k26rng_key; the fields are public so a key can be stored and
 * compared, not so callers assemble them by hand. */
typedef struct {
    uint32_t k0;   /* seed bits 0..31 */
    uint32_t k1;   /* seed bits 32..63 */
} K26RngKey;

/* Draw coordinates. All fields are inputs; nothing here is advanced
 * by the pure functions. */
typedef struct {
    uint16_t stream;        /* stream class; 0x0000 reserved invalid */
    uint16_t channel;       /* channel within the stream class */
    uint32_t environment;   /* environment index in a vectorised set */
    uint32_t episode;       /* episode index, counted from reset */
    uint32_t draw;          /* draw index; 0xFFFFFFFF is the sentinel */
} K26RngCoords;

/* One raw generator block: four 32-bit words, exactly the Philox
 * output for (key, coordinates). */
typedef struct {
    uint32_t v[4];
} K26RngBlock;

/* Stream classes assigned so far. Append-only; values are never
 * reused. Channel allocation within a class belongs to the layer that
 * owns the class. */
enum {
    K26RNG_STREAM_INVALID        = 0x0000,
    K26RNG_STREAM_RESET_STATE    = 0x0001,
    K26RNG_STREAM_DOMAIN_RANDOM  = 0x0002,
    K26RNG_STREAM_SENSOR_NOISE   = 0x0003,
    K26RNG_STREAM_ACTUATOR_NOISE = 0x0004,
    K26RNG_STREAM_TASK           = 0x0005,
    K26RNG_STREAM_EXPERIMENTAL   = 0x7FFF
};

typedef enum {
    K26RNG_OK          = 0,
    K26RNG_E_NULL      = 1,   /* null pointer argument */
    K26RNG_E_EXHAUSTED = 2,   /* cursor at the draw-index sentinel */
    K26RNG_E_BAD_ARG   = 3    /* invalid stream class or bound */
} K26RngStatus;

/* Decode any status value to a short stable string. */
const char *k26rng_status_str(K26RngStatus s);

/* Build a key from the governing seed. */
K26RngKey k26rng_key(uint64_t seed);

/* Pure draws. Each consumes exactly one tick (the draw index the
 * caller passes); the caller owns index bookkeeping, or uses the
 * cursor below. Coordinates with stream 0x0000 or draw 0xFFFFFFFF are
 * documented precondition violations for these functions. */

/* The raw block: four 32-bit words. */
K26RngBlock k26rng_block(K26RngKey key, K26RngCoords c);

/* One 64-bit word: the block's first two words, first word high
 * (v[0] << 32 | v[1]). */
uint64_t k26rng_u64(K26RngKey key, K26RngCoords c);

/* Double in [0, 1): (u64 >> 11) * 0x1.0p-53, the 53-bit construction,
 * exact in IEEE double, zero included, one excluded. */
double k26rng_uniform01(K26RngKey key, K26RngCoords c);

/* Double in [a, b): a + uniform01 * (b - a), that exact expression.
 * The caller keeps a < b and both finite; rounding at the upper edge
 * follows IEEE arithmetic of the expression. */
double k26rng_uniform(K26RngKey key, K26RngCoords c, double a, double b);

/* Integer in [0, n): the fixed-cost multiply-high method,
 * (u64 * n) >> 64 in 128-bit arithmetic. Each outcome's probability
 * deviates from 1/n by at most 2^-64 absolute; rejection would be
 * unbiased but has a data-dependent draw count, which this library
 * excludes. n == 0 returns 0. */
uint64_t k26rng_bounded(K26RngKey key, K26RngCoords c, uint64_t n);

/* Standard normal by inverse transform, one tick. The tick's 64-bit
 * word maps to the probability grid with its low grid bit set,
 * ((u64 >> 11) | 1) * 0x1.0p-53, so every draw lies strictly inside
 * (0, 1) where the quantile is finite, with no data-dependent branch.
 * The quantile is Wichura's AS 241 (PPND16), with the logarithm its
 * tail needs computed in-house from IEEE basic operations, so no libm
 * call is made and the result is bit-stable everywhere. Absolute
 * error against the true quantile is demonstrated at or below 1e-9 by
 * this library's tests. */
double k26rng_normal(K26RngKey key, K26RngCoords c);

/* Cursor convenience: a local value wrapping key plus coordinates,
 * advancing the draw index by each primitive's tick budget (one, for
 * every version 1 primitive). The cursor draws at indices 0 to
 * 0xFFFFFFFE; at the sentinel it refuses with K26RNG_E_EXHAUSTED and
 * advances nothing, so exhaustion is loud and no coordinate repeats.
 * A cursor whose stream class is 0x0000 refuses with
 * K26RNG_E_BAD_ARG. */
typedef struct {
    K26RngKey    key;
    K26RngCoords c;
} K26RngCursor;

K26RngCursor k26rng_cursor(K26RngKey key, K26RngCoords c);

K26RngStatus k26rng_cursor_u64(K26RngCursor *cur, uint64_t *out);
K26RngStatus k26rng_cursor_uniform01(K26RngCursor *cur, double *out);
K26RngStatus k26rng_cursor_uniform(K26RngCursor *cur, double a, double b,
                                   double *out);
/* n == 0 refuses with K26RNG_E_BAD_ARG and advances nothing. */
K26RngStatus k26rng_cursor_bounded(K26RngCursor *cur, uint64_t n,
                                   uint64_t *out);
K26RngStatus k26rng_cursor_normal(K26RngCursor *cur, double *out);

#ifdef __cplusplus
}
#endif

#endif /* K26RNG_H */
