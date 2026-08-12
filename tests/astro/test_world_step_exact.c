/* tests/astro/test_world_step_exact.c - exact simulated-time stepping gate.
 *
 * k26astro_world_step passes its dt through the tick layer's wallclock
 * clamps, so a 3600 s request advances the world 0.5 s. The exact
 * entry k26astro_world_step_exact must advance the world clock by
 * exactly the requested simulated time, and the clamped entry must
 * keep its existing behaviour untouched. */
#include "k26astro_rt/world.h"
#include "k26astro_core/epoch.h"
#include "k26astro_body/body.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static double epoch_s_(const K26AstroEpoch *e)
{
    return (double)e->days_since_J2000 * 86400.0 + e->seconds_of_day;
}

int main(void)
{
    K26AstroWorld *w = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                             K26ASTRO_COORDS_SECTOR_GRID);
    ASSERT(w);

    /* The integrators refuse an empty world (n_bodies < 1), so the
     * clock needs one body to tick against. */
    K26AstroBody sun;
    k26astro_body_init(&sun);
    strncpy(sun.name, "sun", sizeof sun.name - 1);
    sun.mass   = 1.989e30;
    sun.gm     = 1.32712440018e20;
    sun.radius = 6.957e8;
    ASSERT(k26astro_world_add_body(w, sun) >= 0);

    K26AstroEpoch e0, e1, e2;
    ASSERT(k26astro_world_now(w, &e0) == 0);

    /* Exact: one hour requested is one hour advanced. */
    ASSERT(k26astro_world_step_exact(w, 3600.0) == 0);
    ASSERT(k26astro_world_now(w, &e1) == 0);
    double advanced = epoch_s_(&e1) - epoch_s_(&e0);
    printf("step_exact(3600): advanced %.9f s\n", advanced);
    ASSERT(fabs(advanced - 3600.0) < 1e-6);

    /* Clamped entry unchanged: the same request advances at most the
     * 0.5 s hard cap. */
    ASSERT(k26astro_world_step(w, 3600.0) == 0);
    ASSERT(k26astro_world_now(w, &e2) == 0);
    double clamped = epoch_s_(&e2) - epoch_s_(&e1);
    printf("step(3600): advanced %.9f s\n", clamped);
    ASSERT(clamped <= 0.5 + 1e-9);

    /* Exact rejects the arguments the clamped entry rejects. */
    ASSERT(k26astro_world_step_exact(NULL, 1.0) < 0);
    ASSERT(k26astro_world_step_exact(w, -1.0) < 0);

    k26astro_world_destroy(w);
    printf("test_world_step_exact: all assertions passed\n");
    return 0;
}
