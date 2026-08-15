/* models.c: the imperfection models, and the chain that applies them.
 *
 * The bias walk, derived rather than transcribed.
 * ----------------------------------------------
 * A first-order Gauss-Markov process is the solution of
 *
 *     db/dt = -b/tau + w(t),
 *
 * with w white. Over one interval dt the exact solution is
 *
 *     b(t+dt) = exp(-dt/tau) b(t) + integral of the driving term,
 *
 * and the integral is a zero-mean normal independent of b(t). Writing
 * phi = exp(-dt/tau), the discrete process is b <- phi b + q n with n
 * a standard normal, and the stationary variance V satisfies
 *
 *     V = phi^2 V + q^2   ->   q^2 = V (1 - phi^2).
 *
 * Choosing q = sigma sqrt(1 - phi^2) therefore makes the stationary
 * standard deviation exactly sigma, whatever dt is, and leaves the
 * autocorrelation at lag m steps equal to phi^m = exp(-m dt / tau),
 * which is the declared correlation time expressed in steps. Those two
 * properties are what tests/test_sense_models.c measures: the realised
 * standard deviation against sigma, and the correlation time recovered
 * from the realised lag-one correlation against tau.
 *
 * The alternative reading, in which the declared standard deviation is
 * the size of the per-step kick rather than the process's steady
 * state, is rejected because it is not well posed: under it the
 * stationary spread is sigma / sqrt(1 - phi^2), which grows without
 * bound as the control period shortens, so the same declaration would
 * describe a different sensor at a different control period.
 *
 * Draws, and why a dropout does not disturb them.
 * ----------------------------------------------
 * Draws are addressed, not consumed from a stream: each term computes
 * its own draw from (class, channel, environment, episode, index), so
 * one term never shifts another's coordinates. The chain therefore
 * evolves every stateful term on every step, whatever the dropout
 * does, and a dropout changes only which value is delivered. Without
 * that, a bias trajectory would depend on the dropout history and an
 * episode would not replay from its identity triple.
 */
#include <math.h>

#include "k26sense.h"

const char *k26sense_status_str(K26SenseStatus s)
{
    switch (s) {
    case K26SENSE_OK:         return "ok";
    case K26SENSE_E_NULL:     return "null argument";
    case K26SENSE_E_RANGE:    return "parameter out of range";
    case K26SENSE_E_CHAIN:    return "chain needs more state than one block";
    case K26SENSE_E_CAPACITY: return "delay ring shorter than the depth";
    }
    return "unknown status";
}

/* Finite in the sense every check here needs, written out rather than
 * calling isfinite so the file's arithmetic claim covers every line. */
static int sense_finite_(double x)
{
    return (x == x) && (x - x == 0.0);
}

static K26RngCoords sense_coords_(uint16_t cls, uint16_t channel,
                                  uint32_t env, uint32_t episode,
                                  uint32_t draw)
{
    K26RngCoords c;
    c.stream      = cls;
    c.channel     = channel;
    c.environment = env;
    c.episode     = episode;
    c.draw        = draw;
    return c;
}

/* ---- The models, individually ------------------------------------ */

double k26sense_additive(double v, double sigma, K26RngKey key,
                         K26RngCoords c)
{
    return v + sigma * k26rng_normal(key, c);
}

double k26sense_scale(double v, double rel_sigma, K26RngKey key,
                      K26RngCoords c)
{
    return v * (1.0 + rel_sigma * k26rng_normal(key, c));
}

double k26sense_bias_walk_step(double bias, double phi, double q,
                               K26RngKey key, K26RngCoords c)
{
    return phi * bias + q * k26rng_normal(key, c);
}

double k26sense_quantise(double v, double lsb, double lo, double hi)
{
    if (!(lsb > 0.0) || !sense_finite_(lsb)) return v;
    if (v < lo) v = lo;
    if (v > hi) v = hi;
    /* Half away from zero: a tie at a negative value moves the same
     * distance as its positive mirror, which a truncation towards zero
     * alone would not do. Every operation is exact under IEEE-754 and
     * no rounding mode can move the answer. */
    double scaled = v / lsb;
    double half   = scaled < 0.0 ? -0.5 : 0.5;
    return (double)(int64_t)(scaled + half) * lsb;
}

double k26sense_deadband(double v, double threshold, int rescale)
{
    if (!(threshold > 0.0) || !sense_finite_(threshold)) return v;
    double mag = v < 0.0 ? -v : v;
    if (mag < threshold) return 0.0;
    if (!rescale) return v;
    /* The declared variant: [threshold, 1] maps onto [0, 1], so the
     * command is continuous at the threshold instead of stepping. A
     * threshold at or above one leaves nothing to map and passes the
     * command through. */
    if (!(threshold < 1.0)) return v;
    double sign = v < 0.0 ? -1.0 : 1.0;
    double span = 1.0 - threshold;
    double out  = (mag - threshold) / span;
    return sign * out;
}

K26SenseStatus k26sense_bias_walk_coeffs(double tau, double dt, double sigma,
                                         double *out_phi, double *out_q)
{
    if (!out_phi || !out_q) return K26SENSE_E_NULL;
    if (!sense_finite_(tau) || !sense_finite_(dt) || !sense_finite_(sigma)) {
        return K26SENSE_E_RANGE;
    }
    if (!(tau > 0.0) || !(dt > 0.0) || sigma < 0.0) {
        return K26SENSE_E_RANGE;
    }
    /* The one exponential in this library, and the one place it is
     * called: outside the stepping path, once per handle. */
    double phi = exp(-dt / tau);
    if (!sense_finite_(phi)) return K26SENSE_E_RANGE;
    if (phi < 0.0) phi = 0.0;
    if (phi > 1.0) phi = 1.0;
    double var = 1.0 - phi * phi;
    if (var < 0.0) var = 0.0;
    *out_phi = phi;
    *out_q   = sigma * sqrt(var);
    return K26SENSE_OK;
}

uint64_t k26sense_world_seed(K26RngKey key, uint32_t env, uint32_t episode)
{
    return k26rng_u64(key, sense_coords_(K26SENSE_CLASS_SENSOR,
                                         K26SENSE_CHANNEL_WORLD_SEED,
                                         env, episode, 0u));
}

/* ---- The chain ---------------------------------------------------- */

K26SenseStatus k26sense_chain_check(const K26SenseTerm *terms, uint32_t n,
                                    uint32_t *depth_needed)
{
    if ((!terms && n > 0) || !depth_needed) return K26SENSE_E_NULL;
    uint32_t depth = 0, n_bias = 0, n_delay = 0;
    for (uint32_t i = 0; i < n; i++) {
        switch (terms[i].kind) {
        case K26SENSE_BIAS_WALK:
            n_bias++;
            break;
        case K26SENSE_LATENCY:
            n_delay++;
            if (terms[i].u.latency.depth > depth) {
                depth = terms[i].u.latency.depth;
            }
            break;
        default:
            break;
        }
    }
    if (n_bias > 1 || n_delay > 1) return K26SENSE_E_CHAIN;
    /* A term drawing at both cadences must hold two channels, or its
     * per-episode draw at index 0 and its first step's draw at index 0
     * would be the same number. */
    for (uint32_t i = 0; i < n; i++) {
        if (terms[i].kind != K26SENSE_BIAS_WALK &&
            terms[i].kind != K26SENSE_DISPERSION) continue;
        if (terms[i].channel_ep == terms[i].channel) return K26SENSE_E_CHAIN;
        if (terms[i].channel_ep == K26SENSE_NO_CHANNEL) return K26SENSE_E_CHAIN;
    }
    *depth_needed = depth;
    return K26SENSE_OK;
}

K26SenseStatus k26sense_chain_reset(const K26SenseTerm *terms, uint32_t n,
                                    K26SenseState *st, K26RngKey key,
                                    uint16_t cls, uint32_t env,
                                    uint32_t episode, double value)
{
    if ((!terms && n > 0) || !st) return K26SENSE_E_NULL;

    st->bias     = 0.0;
    st->disp     = 1.0;
    st->last     = value;
    st->ring_head = 0;
    st->primed   = 1;

    for (uint32_t i = 0; i < n; i++) {
        const K26SenseTerm *t = &terms[i];
        /* Every per-episode draw takes index 0 on its own channel,
         * which no per-step term shares. */
        K26RngCoords c = sense_coords_(cls, t->channel_ep, env, episode,
                                       0u);
        switch (t->kind) {
        case K26SENSE_BIAS_WALK:
            st->bias = t->u.bias_walk.sigma0 * k26rng_normal(key, c);
            break;
        case K26SENSE_DISPERSION:
            st->disp = 1.0 + t->u.dispersion.sigma_ep * k26rng_normal(key, c);
            break;
        case K26SENSE_LATENCY: {
            uint32_t d = t->u.latency.depth;
            if (d > st->ring_cap) return K26SENSE_E_CAPACITY;
            if (d > 0 && !st->ring) return K26SENSE_E_NULL;
            for (uint32_t k = 0; k < d; k++) st->ring[k] = value;
            break;
        }
        default:
            break;
        }
    }
    return K26SENSE_OK;
}

K26SenseStatus k26sense_chain_apply(const K26SenseTerm *terms, uint32_t n,
                                    K26SenseState *st, K26RngKey key,
                                    uint16_t cls, uint32_t env,
                                    uint32_t episode, uint32_t step,
                                    double value, double *out)
{
    if ((!terms && n > 0) || !st || !out) return K26SENSE_E_NULL;

    double v = value;
    int    dropped = 0;

    for (uint32_t i = 0; i < n; i++) {
        const K26SenseTerm *t = &terms[i];
        /* A per-step term's draw index is the transition index, so any
         * step's noise is reachable without producing the step before
         * it. */
        K26RngCoords c = sense_coords_(cls, t->channel, env, episode, step);
        switch (t->kind) {
        case K26SENSE_ADDITIVE:
            v = k26sense_additive(v, t->u.additive.sigma, key, c);
            break;
        case K26SENSE_SCALE:
            v = k26sense_scale(v, t->u.scale.rel_sigma, key, c);
            break;
        case K26SENSE_BIAS_WALK:
            /* The state advances on every step, dropout or not, so an
             * episode replays from its identity triple. */
            st->bias = k26sense_bias_walk_step(st->bias,
                                               t->u.bias_walk.phi,
                                               t->u.bias_walk.q, key, c);
            v += st->bias;
            break;
        case K26SENSE_LATENCY: {
            uint32_t d = t->u.latency.depth;
            if (d > st->ring_cap) return K26SENSE_E_CAPACITY;
            if (d > 0) {
                if (!st->ring) return K26SENSE_E_NULL;
                double held = st->ring[st->ring_head];
                st->ring[st->ring_head] = v;
                st->ring_head = (st->ring_head + 1u) % d;
                v = held;
            }
            break;
        }
        case K26SENSE_QUANTISE:
            v = k26sense_quantise(v, t->u.quantise.lsb, t->u.quantise.lo,
                                  t->u.quantise.hi);
            break;
        case K26SENSE_DROPOUT:
            if (k26rng_uniform(key, c, 0.0, 1.0) < t->u.dropout.p) {
                dropped = 1;
            }
            break;
        case K26SENSE_DEADBAND:
            v = k26sense_deadband(v, t->u.deadband.threshold,
                                  t->u.deadband.rescale);
            break;
        case K26SENSE_DISPERSION:
            v *= st->disp;
            if (t->u.dispersion.sigma_step != 0.0) {
                v = k26sense_scale(v, t->u.dispersion.sigma_step, key, c);
            }
            break;
        }
    }

    /* A dropout delivers the value the channel last delivered, whole,
     * rather than a sentinel: that is what a controller sees when a
     * sensor misses an update, and a sentinel in an observation vector
     * is a number a policy learns to exploit. Because it is the
     * delivered value, where the dropout sits in the chain does not
     * change what it produces; every other term is order-sensitive. */
    if (dropped && st->primed) {
        *out = st->last;
    } else {
        *out = v;
        st->last = v;
        st->primed = 1;
    }
    return K26SENSE_OK;
}
