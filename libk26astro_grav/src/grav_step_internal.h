/* grav_step_internal.h — private dispatch helper.
 *
 * The public k26astro_grav_step routes through the event-time
 * wrapper when state->events has registered events; otherwise it
 * dispatches the inner integrator directly. The wrapper in
 * advance_with_events.c uses this helper as its sub-step entry
 * point. */
#ifndef K26ASTRO_GRAV_STEP_INTERNAL_H
#define K26ASTRO_GRAV_STEP_INTERNAL_H

#include <limits.h>

#include "k26astro_grav/grav.h"
#include "k26astro_grav/forces.h"   /* K26AstroPairWeight */

/* Capacity growth target shared by the carry and scratch guards: at
 * least double the current capacity, never less than the immediate
 * need. Growing geometrically makes a body-by-body build-up perform
 * amortised-constant allocator work per add; growing to the exact
 * need would re-copy (and re-zero) every buffer on every add. */
static inline int k26_grav_grow_target_(int cur_cap, int need)
{
    int dbl = (cur_cap > 0 && cur_cap <= INT_MAX / 2) ? cur_cap * 2 : need;
    return dbl > need ? dbl : need;
}

/* Dispatch the configured integrator for a single step of duration
 * `dt`. Mirrors the original k26astro_grav_step body before the
 * event-wrapper split. Does NOT consult the event registry. */
int k26astro_grav_step_inner_dispatch(K26AstroGravState *state, double dt);

/* Step with event-time root-finding. Called by k26astro_grav_step
 * when the event registry is non-empty. */
int k26astro_grav_step_with_events(K26AstroGravState *state, double dt);

/* Wisdom-Holman carry capacity guard (wisdom_holman.c). Allocates
 * only when the carry is missing or under-sized; called from
 * k26astro_grav_state_reserve at non-step times and from the WH
 * step as a fallback. */
int k26astro_grav_wh_carry_ensure(K26AstroGravState *state);

/* MERCURIUS pair-weight lookup (force_direct.c). Returns the K
 * weight recorded for pair (i, j) in either index order, or 0.0 when
 * the pair is not in the list. Shared between the direct-force pair
 * loop and the Wisdom-Holman interaction kick so both consult the
 * split context with identical arithmetic. */
double k26_grav_mercurius_pair_weight(const K26AstroPairWeight *weights,
                                       int n_weights, int i, int j);

#endif /* K26ASTRO_GRAV_STEP_INTERNAL_H */
