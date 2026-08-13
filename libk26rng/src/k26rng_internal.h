/* k26rng_internal.h - private seams for the distribution layer.
 *
 * Not installed. The accuracy tests reach the quantile and the
 * in-house logarithm directly through this header, so their error
 * bounds are demonstrated against external references at the function
 * that carries them, not inferred through the draw path. */
#ifndef K26RNG_INTERNAL_H
#define K26RNG_INTERNAL_H

/* Natural logarithm from IEEE basic operations only: exact power-of-2
 * decomposition by bit manipulation, then the atanh series on the
 * reduced mantissa. Defined for finite x > 0. */
double k26rng_internal_ln(double x);

/* The AS 241 PPND16 normal quantile (Wichura, Applied Statistics 37
 * no. 3, 1988), with the tail logarithm from k26rng_internal_ln.
 * Defined for p in (0, 1). */
double k26rng_internal_quantile(double p);

#endif /* K26RNG_INTERNAL_H */
