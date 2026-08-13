/* test_ias15_wall_budget.c — IAS15 wall-time budget safety net.
 *
 * Verifies that k26astro_grav_ias15_set_wall_budget bounds an
 * otherwise-stalling integration call. Without the budget,
 * pathological chaotic regions (e.g. Burrau-style triple close
 * encounters at impossibly tight tol) can spend minutes accepting
 * picosecond substeps in the IEEE-754 truncation-noise regime;
 * the budget exists as a developer-facing safety net. That
 * truncation-noise regime itself is no longer exercised by any
 * test: the fixture below stalls on legitimate volume of work
 * instead, a consciously accepted coverage reduction (the earlier
 * regime-based fixture's stall came from a since-fixed defect, not
 * from the regime).
 *
 * Strategy: integrate a tight two-body orbit (98-minute period)
 * over a 100-year horizon in one call. The controller legitimately
 * needs millions of substeps, so the unbudgeted call runs for
 * minutes; with a 100 ms wall budget the call must return
 * K26ASTRO_E_TIME_BUDGET (not OK, not NO_CONVERGE).
 *
 * An earlier revision of this test provoked the stall with a
 * near-degenerate close encounter at an absurdly tight tolerance.
 * That fixture's wall time was dominated by the sector fold walking
 * a diverged offset back one sector per loop iteration; with the
 * fold running in constant time the reject cascade finishes in
 * about a millisecond and correctly reports NO_CONVERGE before any
 * budget can expire, so it no longer exercises the budget at all.
 *
 * The budget is NOT a determinism gate; wall-clock measurement is
 * platform-dependent. Deterministic runs must keep budget = 0.0. */
#include "k26astro_grav/grav.h"
#include "k26astro_grav/ias15.h"
#include "k26astro_body/body.h"
#include "k26astro_core/consts.h"
#include "k26astro_core/pos.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    K26AstroBody bodies[2];
    memset(bodies, 0, sizeof(bodies));

    /* A tight circular two-body orbit: 7000 km around an Earth-mass
     * central body, 98-minute period. Legitimately integrable, but
     * a 100-year horizon needs millions of accepted substeps. */
    bodies[0].kind = K26ASTRO_BODY_PLANET;
    bodies[0].gm   = K26A_GM_EARTH;
    bodies[0].pos  = k26astro_pos_zero();
    bodies[0].vel  = (K26V3){ 0.0, 0.0, 0.0 };
    bodies[0].parent_body_idx = -1;

    bodies[1].kind = K26ASTRO_BODY_MOON;
    bodies[1].gm   = 1.0;
    bodies[1].pos  = k26astro_pos_from_m(7.0e6, 0.0, 0.0);
    /* Circular speed sqrt(mu / r) for mu = 3.986004418e14, r = 7e6. */
    bodies[1].vel  = (K26V3){ 0.0, 7546.0533467, 0.0 };
    bodies[1].parent_body_idx = 0;

    K26AstroGravState state;
    assert(k26astro_grav_state_init(&state, bodies, 2) == 0);
    assert(k26astro_grav_set_integrator(&state, K26ASTRO_INTEGRATOR_IAS15) == 0);
    /* 100 ms wall budget, orders of magnitude below the unbudgeted
     * call's runtime. */
    k26astro_grav_ias15_set_wall_budget(&state, 0.1);

    /* Request 100 years in one call. Without the budget this runs
     * for minutes of legitimate substepping. With the budget it
     * returns TIME_BUDGET cleanly within ~100 ms. */
    int rc = k26astro_grav_step(&state, 100.0 * 365.25 * 86400.0);
    fprintf(stderr,
            "test_ias15_wall_budget: rc=%d (TIME_BUDGET=%d) rejects=%u\n",
            rc, K26ASTRO_E_TIME_BUDGET,
            k26astro_grav_ias15_rejected_steps(&state));
    assert(rc == K26ASTRO_E_TIME_BUDGET);

    /* Sanity: with budget = 0 the call falls back to normal
     * behaviour (NO_CONVERGE or completion). Re-init state so the
     * carry-over from the budget-exit doesn't bias this leg. */
    K26AstroBody bodies2[2];
    memcpy(bodies2, bodies, sizeof(bodies));
    K26AstroGravState state2;
    assert(k26astro_grav_state_init(&state2, bodies2, 2) == 0);
    assert(k26astro_grav_set_integrator(&state2, K26ASTRO_INTEGRATOR_IAS15) == 0);
    k26astro_grav_ias15_set_tol(&state2, 1.0e-9);
    k26astro_grav_ias15_set_wall_budget(&state2, 0.0);
    /* Short call so the runtime stays trivial; just verify budget = 0
     * does not short-circuit normal integration. */
    rc = k26astro_grav_step(&state2, 86400.0);
    fprintf(stderr,
            "test_ias15_wall_budget: budget=0 1-day step rc=%d\n", rc);
    assert(rc == K26ASTRO_E_OK || rc == K26ASTRO_E_NO_CONVERGE);
    assert(rc != K26ASTRO_E_TIME_BUDGET);

    k26astro_grav_state_destroy(&state);
    k26astro_grav_state_destroy(&state2);

    printf("test_ias15_wall_budget: OK\n");
    return 0;
}
