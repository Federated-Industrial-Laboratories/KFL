/* encounter_internal.h — private MERCURIUS handoff helpers.
 *
 * The K(y) smoothing function is the Rein-Tamayo 2019 quintic
 * smoothstep over the transition window [y_inner, y_outer] where
 * y = r_ij / hill_radius(pair). */
#ifndef K26ASTRO_RT_ENCOUNTER_INTERNAL_H
#define K26ASTRO_RT_ENCOUNTER_INTERNAL_H

#include "world_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Quintic smoothstep K(y; y_inner, y_outer). Returns 1.0 when
 * y <= y_inner (force entirely in IAS15 / "near"), 0.0 when
 * y >= y_outer (force entirely in WH / "far"), and a C^2-continuous
 * polynomial in between:
 *   x = (y - y_inner) / (y_outer - y_inner)
 *   K(y) = 1 - (10 x^3 - 15 x^4 + 6 x^5)         x in [0, 1]
 * Coefficients are exact integers; bit-identical across libm. */
double k26astro_mercurius_K(double y, double y_inner, double y_outer);

/* Pairwise mutual Hill radius,
 *   R_hill_ij = ((a_i + a_j) / 2) * cbrt( (m_i + m_j) / (3 * M_central) ),
 * where a_i and a_j are the bodies' osculating semi-major axes about
 * `central` (vis-viva), falling back to the body's current distance
 * from `central` for unbound or degenerate relative states. The full
 * criterion statement and the citation live at the definition
 * (encounter.c). Returns 0 if inputs are degenerate. */
double k26astro_mercurius_hill_radius(const K26AstroBody *i,
                                       const K26AstroBody *j,
                                       const K26AstroBody *central);

/* Scan all pairs; populate world->encounters with pairs whose
 * y < world->mercurius_outer_factor. Each session record carries
 * the per-pair K weight (Rein-Tamayo 2019 eq. 11) for the
 * paper-faithful pair-by-pair force split. Returns the number of
 * active encounters. */
int k26astro_mercurius_detect(K26AstroWorld *world);

/* Size world->encounters and world->pair_weights for the worst case
 * of n_bodies*(n_bodies-1)/2 pairs. Called from the world's body-add
 * path so the per-step detect and the MERCURIUS split build never
 * allocate. Returns 0, or -1 on allocation failure (existing
 * capacities are retained). */
int k26astro_rt_encounter_reserve(K26AstroWorld *world, int n_bodies);

#ifdef __cplusplus
}
#endif

#endif /* K26ASTRO_RT_ENCOUNTER_INTERNAL_H */
