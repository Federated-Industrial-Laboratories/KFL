/* k26sense: the imperfection layer. What a sensor reports instead of
 * the truth, and what an actuator does instead of the command.
 *
 * Every model here is a pure function of the true value, its declared
 * parameters, and its draw coordinates. None of them holds a generator:
 * a draw is addressed by (class, channel, environment, episode, draw
 * index) and computed on demand from the counter-based generator, so
 * any step's noise is reachable without producing the step before it,
 * and an environment's noise is independent of how many neighbours it
 * has and of what they draw.
 *
 * Draw budget. Every model states its budget and holds to it: a fixed
 * number of draws per step, never a rejection loop, never a count that
 * depends on the value. That is what makes a step's coordinates
 * predictable and the whole layer replayable.
 *
 * State is state, not draws. The bias walk and the delay rings carry
 * state between steps in caller-provided storage sized at create. An
 * episode's bias at step k is recovered by re-simulating the episode
 * from its identity triple, which the episode format guarantees, and
 * every individual draw inside it stays addressable by its own
 * coordinates.
 *
 * Arithmetic and determinism, per model rather than as a blanket
 * claim. Additive noise, scale-factor noise, latency, dropout,
 * deadband and dispersion use addition, subtraction, multiplication,
 * division and comparison only, all exact or correctly rounded under
 * IEEE-754, so they are reproducible across platforms as well as
 * across runs, given a generator that is. Quantisation adds one more
 * operation, a conversion from double to a signed 64-bit integer,
 * which is exact where it is defined and undefined outside that
 * type's range; the model saturates rather than converting there, so
 * it is total and portable at every input rather than only inside a
 * range a caller has to know about. The bias walk's step is the same
 * five operations; its two coefficients are not, because they come
 * from an exponential. They are computed once, outside the stepping
 * path, by k26sense_bias_walk_coeffs, which is the one function here
 * that calls the platform's exponential and the one place this library
 * carries the per-binary claim rather than the cross-platform one.
 * Nothing on the step path calls the platform's mathematics library:
 * the only calls it makes are into the counter-based generator, which
 * carries the same no-libm rule.
 *
 * Provenance. Nothing here is taken on authority. The bias walk is the
 * first-order Gauss-Markov process, a turn-on bias drawn once per
 * episode and an in-run bias decaying towards zero with a declared
 * correlation time while driven by white noise; its exact
 * discrete-time form is derived in src/models.c from the continuous
 * process, and tests/test_sense_models.c measures the stationary
 * standard deviation it produces and recovers the correlation time
 * from the realised autocorrelation. The other models are definitions
 * rather than results: what additive noise, a scale factor, a delay, a
 * quantiser, a dropout and a deadband do is stated in this header and
 * measured directly.
 */
#ifndef K26SENSE_H
#define K26SENSE_H

#include <stdint.h>

#include "k26rng.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The two stream classes this layer owns. */
#define K26SENSE_CLASS_SENSOR   ((uint16_t)0x0003)
#define K26SENSE_CLASS_ACTUATOR ((uint16_t)0x0004)

/* The top of the sensor channel space, reserved for the per-
 * environment, per-episode seed handed to a world's stateful
 * generator. It is never allocated to a model, so a program may add
 * any number of sensors without reaching it. */
#define K26SENSE_CHANNEL_WORLD_SEED ((uint16_t)0xFFFF)

/* A term that draws nothing carries this in place of a channel. */
#define K26SENSE_NO_CHANNEL ((uint16_t)0xFFFE)

/* Two to the sixty-third, exactly, as a double: the first magnitude a
 * signed 64-bit conversion cannot take. A quantiser saturates here
 * rather than converting, and a caller sizing a declared range against
 * a step uses it to know what the step can reach: a step of `lsb`
 * spans at most this many multiples either side of zero. */
#define K26SENSE_I64_SPAN 9223372036854775808.0

typedef enum {
    K26SENSE_OK             = 0,
    K26SENSE_E_NULL         = 1,  /* null argument */
    K26SENSE_E_RANGE        = 2,  /* a parameter is not a usable finite
                                   * number, or is outside its domain */
    K26SENSE_E_CHAIN        = 3,  /* the chain cannot be served by one
                                   * state block: more than one bias
                                   * walk, or more than one delay */
    K26SENSE_E_CAPACITY     = 4   /* the delay ring given is shorter
                                   * than the declared depth */
} K26SenseStatus;

/**
 * @brief Decode a status value.
 * @param s A status.
 * @return A short English phrase; never NULL.
 */
const char *k26sense_status_str(K26SenseStatus s);

/* ---- The models ---------------------------------------------------- */

typedef enum {
    /* Sensor side. */
    K26SENSE_ADDITIVE   = 1,   /* one normal draw per step */
    K26SENSE_SCALE      = 2,   /* one normal draw per step */
    K26SENSE_BIAS_WALK  = 3,   /* one normal draw per step, plus one
                                * per episode at reset */
    K26SENSE_LATENCY    = 4,   /* no draws */
    K26SENSE_QUANTISE   = 5,   /* no draws */
    K26SENSE_DROPOUT    = 6,   /* one uniform draw per step */
    /* Actuator side. Same primitives, a different stream class. */
    K26SENSE_DEADBAND   = 7,   /* no draws */
    K26SENSE_DISPERSION = 8    /* one normal draw per episode at reset,
                                * and one per step when per-step noise
                                * is declared */
} K26SenseKind;

/* One term of a declared chain, with the channel it draws on.
 *
 * `channel` is allocated by the owning layer in source order, one per
 * term that draws, and a term that draws per episode never shares a
 * channel with one that draws per step, so a per-episode draw at index
 * 0 cannot collide with the first step's draw. */
typedef struct {
    K26SenseKind kind;
    /* The channel a per-step draw takes. */
    uint16_t     channel;
    /* The channel a per-episode draw takes, for the two terms that
     * draw at both cadences. It is a second channel and never the
     * one above, because a per-episode draw takes index 0 and the
     * first step's draw takes index 0 as well: sharing a channel
     * would make an episode's turn-on bias and its first step's kick
     * the same number. Terms that draw at one cadence only leave it
     * at K26SENSE_NO_CHANNEL. */
    uint16_t     channel_ep;
    union {
        /* Standard deviation of the added normal, in the value's own
         * units. */
        struct { double sigma; } additive;
        /* Relative standard deviation: the value is multiplied by
         * (1 + rel_sigma * n). */
        struct { double rel_sigma; } scale;
        /* sigma0 is the turn-on bias standard deviation, drawn once
         * per episode. phi and q are the discrete-time coefficients
         * from k26sense_bias_walk_coeffs. */
        struct { double sigma0, phi, q; } bias_walk;
        /* Whole control periods of delay. A depth of zero is the
         * identity and is legal. */
        struct { uint32_t depth; } latency;
        /* Least significant bit, and the range values clamp to before
         * the integer conversion, so that conversion cannot overflow. */
        struct { double lsb, lo, hi; } quantise;
        /* Probability in [0, 1) that the channel holds its previously
         * delivered value instead of the current one. */
        struct { double p; } dropout;
        /* Commands whose magnitude is below `threshold` produce zero.
         * When `rescale` is non-zero, a command at or above it is
         * mapped from [threshold, 1] onto [0, 1] instead of passing
         * unchanged, which is the declared variant rather than the
         * assumed behaviour. */
        struct { double threshold; int rescale; } deadband;
        /* A multiplicative scale drawn once per episode from
         * 1 + sigma_ep * n, and optional per-step multiplicative noise
         * of relative standard deviation sigma_step. */
        struct { double sigma_ep, sigma_step; } dispersion;
    } u;
} K26SenseTerm;

/* Per-channel state. The delay ring is caller storage so that nothing
 * here allocates; `ring_cap` is what the caller provided and the chain
 * refuses a declared depth larger than it. */
typedef struct {
    double    bias;       /* the bias walk's current value */
    double    disp;       /* the episode's dispersion scale */
    double    last;       /* the last value delivered, for a dropout */
    double   *ring;       /* delay ring, ring_cap entries */
    uint32_t  ring_cap;
    uint32_t  ring_head;
    uint8_t   primed;     /* whether `last` and the ring hold values */
} K26SenseState;

/**
 * @brief The discrete-time coefficients of a first-order Gauss-Markov
 *        process.
 * @param tau    Correlation time, seconds, positive.
 * @param dt     Step interval, seconds, positive.
 * @param sigma  Stationary standard deviation of the process.
 * @param out_phi Decay per step, exp(-dt/tau).
 * @param out_q  Driving term per step, sigma * sqrt(1 - phi*phi).
 * @return K26SENSE_OK, or a status.
 * @note  The process is b' = -b/tau + w, whose exact discrete form
 *        over an interval dt is b <- phi b + q n with the two
 *        coefficients above; that pair makes the stationary standard
 *        deviation exactly `sigma` and the autocorrelation at lag m
 *        steps exactly phi^m, which is exp(-m dt / tau). `sigma` is
 *        therefore the process's own steady-state spread and not the
 *        size of the per-step kick, so a program that changes its
 *        control period keeps the sensor it declared.
 * @note  This is the one function here that calls the platform's
 *        exponential. Call it once, at create; it is not for the
 *        stepping path, and the step below needs only its outputs.
 */
K26SenseStatus k26sense_bias_walk_coeffs(double tau, double dt,
                                         double sigma,
                                         double *out_phi, double *out_q);

/**
 * @brief Check that a chain can be served by one state block.
 * @param terms  The chain, in declaration order.
 * @param n      Term count.
 * @param depth_needed Written with the largest declared delay depth.
 * @return K26SENSE_OK, or K26SENSE_E_CHAIN naming the conflict.
 * @note  One bias walk and one delay per chain, because one state
 *        block carries one of each. Any number of the stateless terms
 *        may appear.
 */
K26SenseStatus k26sense_chain_check(const K26SenseTerm *terms, uint32_t n,
                                    uint32_t *depth_needed);

/**
 * @brief Per-episode setup: the draws a chain takes once, and the
 *        state a step expects to find.
 * @param terms  The chain, in declaration order.
 * @param n      Term count.
 * @param st     State block; `ring` and `ring_cap` are the caller's.
 * @param key    The governing key.
 * @param cls    K26SENSE_CLASS_SENSOR or K26SENSE_CLASS_ACTUATOR.
 * @param env    Environment index.
 * @param episode Episode index.
 * @param value  The value the channel reads at reset, used to fill the
 *               delay ring and to prime the dropout hold.
 * @return K26SENSE_OK, or a status.
 * @note  Every per-episode draw here takes draw index 0 on its own
 *        channel, which no per-step term shares.
 */
K26SenseStatus k26sense_chain_reset(const K26SenseTerm *terms, uint32_t n,
                                    K26SenseState *st, K26RngKey key,
                                    uint16_t cls, uint32_t env,
                                    uint32_t episode, double value);

/**
 * @brief Apply a chain to one value for one step.
 * @param terms  The chain, in declaration order.
 * @param n      Term count.
 * @param st     State block from k26sense_chain_reset.
 * @param key    The governing key.
 * @param cls    K26SENSE_CLASS_SENSOR or K26SENSE_CLASS_ACTUATOR.
 * @param env    Environment index.
 * @param episode Episode index.
 * @param step   Transition index within the episode, which is the draw
 *               index every per-step term uses.
 * @param value  The true value.
 * @param out    The value the channel delivers.
 * @return K26SENSE_OK, or a status.
 * @note  Terms are applied in the order given, which is the order the
 *        program declared them, so a quantiser after a noise term
 *        quantises the noisy value and one before it does not.
 * @note  No allocation, no I/O, and a fixed number of draws.
 */
K26SenseStatus k26sense_chain_apply(const K26SenseTerm *terms, uint32_t n,
                                    K26SenseState *st, K26RngKey key,
                                    uint16_t cls, uint32_t env,
                                    uint32_t episode, uint32_t step,
                                    double value, double *out);

/**
 * @brief The seed a world's stateful generator takes for one
 *        environment and episode.
 * @param key     The governing key.
 * @param env     Environment index.
 * @param episode Episode index.
 * @return A 64-bit seed drawn at the reserved channel.
 * @note  Reserved channel 0xFFFF of the sensor class, draw index 0.
 *        Seeding a stateful generator per environment and per episode
 *        is what stops two environments sharing a sequence when both
 *        were handed the governing seed, and it makes an episode that
 *        uses such a generator replay from its identity triple like
 *        everything else.
 */
uint64_t k26sense_world_seed(K26RngKey key, uint32_t env, uint32_t episode);

/* ---- The models, individually, for callers that want one --------- */

/** @brief v + sigma * n, one normal draw. */
double k26sense_additive(double v, double sigma, K26RngKey key,
                         K26RngCoords c);

/** @brief v * (1 + rel_sigma * n), one normal draw. */
double k26sense_scale(double v, double rel_sigma, K26RngKey key,
                      K26RngCoords c);

/**
 * @brief One step of the bias walk: b <- phi b + q n.
 * @return The new bias, which the caller adds to the value.
 */
double k26sense_bias_walk_step(double bias, double phi, double q,
                               K26RngKey key, K26RngCoords c);

/**
 * @brief Round to a multiple of `lsb` after clamping to [lo, hi].
 * @note  Half away from zero, so that a tie at a negative value rounds
 *        the same distance as its positive mirror. An lsb of zero, or
 *        a value that is not finite, is the identity.
 * @note  A grid position beyond what a signed 64-bit integer holds
 *        saturates at K26SENSE_I64_SPAN multiples of the step. The
 *        conversion is undefined outside that range, so saturating is
 *        what makes this model total: no declared step and no value
 *        can reach undefined behaviour through it. A declared range
 *        should keep the saturation unreachable, and a caller that
 *        wants that guaranteed checks `hi / lsb < K26SENSE_I64_SPAN`
 *        before declaring one.
 */
double k26sense_quantise(double v, double lsb, double lo, double hi);

/**
 * @brief Zero below the threshold, and above it either unchanged or
 *        rescaled from [threshold, 1] onto [0, 1].
 */
double k26sense_deadband(double v, double threshold, int rescale);

#ifdef __cplusplus
}
#endif

#endif /* K26SENSE_H */
