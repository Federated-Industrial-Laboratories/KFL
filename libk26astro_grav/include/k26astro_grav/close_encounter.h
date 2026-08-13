/* k26astro_grav/close_encounter.h — Hill-radius proximity detector.
 *
 * Standalone parent-relative proximity scan. NOT consumed by the
 * MERCURIUS handoff orchestration: libk26astro_rt runs its own
 * pairwise detector (encounter.c there), which builds the mutual
 * Hill radius from vis-viva semi-major axes. This query scales each
 * body's CURRENT parent distance instead, so its ratio
 * r / (r * cbrt(gm/3gm_parent)) is independent of the separation:
 * it classifies by mass ratio alone and cannot measure approach.
 * Kept for its API compatibility; new callers wanting a closeness
 * measure should not use it.
 *
 * Returns -1 if no body registers, otherwise the body index with
 * the smallest r/r_Hill ratio under hill_factor. */
#ifndef K26ASTRO_GRAV_CLOSE_ENCOUNTER_H
#define K26ASTRO_GRAV_CLOSE_ENCOUNTER_H

#include "k26astro_grav/forces.h"

#ifdef __cplusplus
extern "C" {
#endif

int k26astro_grav_close_encounter(const K26AstroGravView *view,
                                   double hill_factor);

#ifdef __cplusplus
}
#endif

#endif
