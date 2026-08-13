/* orbit_step.c — orbit-channel callback (paper-faithful MERCURIUS).
 *
 * Runs every wallclock advance with the actual elapsed dt. Drives
 * the world's integrator (WH, IAS15, Verlet, RK4, RK45, or MERCURIUS).
 *
 * Paper-faithful MERCURIUS split, after Rein, Hernandez, Tamayo
 * et al. 2019 (MNRAS 485(4):5490-5497) eq. 12-14:
 *   1. Detect close-encounter pairs (k26astro_mercurius_detect).
 *      Each pair carries a K(y) weight computed at detect time.
 *   2. If no encounters or the active integrator is not Verlet:
 *      single full-force step (the integrator handles dynamics
 *      uniformly). A WH base never splits; the admission note at
 *      do_split below says why.
 *   3. Otherwise (paper-faithful split):
 *        a. Set state->mercurius = { FAR, weights, n }.
 *        b. Run the Verlet outer integrator over dt; it sees only
 *           the (1-K)-weighted portion of each encounter pair plus
 *           full-force on non-encounter pairs. This handles the
 *           smooth bulk dynamics.
 *        c. Set state->mercurius = { NEAR, weights, n }.
 *        d. Run IAS15 over the same dt — it sees only the
 *           K-weighted portion of each encounter pair (zero for
 *           non-encounter pairs). This handles the encounter
 *           dynamics with adaptive precision.
 *        e. Clear state->mercurius.
 *
 *      The two integrators contribute additively to position +
 *      velocity because (a) they integrate disjoint pieces of the
 *      total acceleration (Σ((1-K)+K) = Σ identity) and (b) the
 *      MERCURIUS context applies its K filter inside accel_total,
 *      so each integrator's internal kick/drift sees the right
 *      force field. Position and velocity updates accumulate in
 *      place — no separate delta-summation buffer needed. */
#include "encounter_internal.h"

#include "k26astro_grav/grav.h"
#include "k26astro_grav/close_encounter.h"
#include "k26astro_grav/forces.h"
#include "k26astro_grav/ias15.h"
#include "k26astro_vehicle/vehicle.h"

/* Walk the world's vehicle registry and consume each per-substep
 * mass accumulator. Propulsion thrust callbacks add into the
 * accumulator during accel_total; this is the closing step that
 * propagates accumulated dot_m into vehicle.basic_mass_kg and the
 * bound body's mass/GM via k26astro_body_set_mass. */
static void commit_vehicle_mass_(K26AstroWorld *world, double dt_s)
{
    for (int i = 0; i < world->n_vehicles; i++) {
        K26AstroVehicle *v = world->vehicles[i];
        if (v) k26astro_vehicle_commit_mass_step(v, dt_s);
    }
}

/* Fill the world's preallocated pair-weight buffer from
 * world->encounters and return it; *out_n holds the count. The
 * buffer is sized for every distinct body pair at body-add time
 * (k26astro_rt_encounter_reserve), and the in-step fallback grows
 * both session buffers together, so n_encounters cannot exceed the
 * pair-weight capacity; this path neither allocates nor fails.
 * Should that invariant ever break, the fill clamps to the
 * pair-weight capacity: an unweighted pair is simply treated as a
 * non-encounter (full force in the FAR pass, zero in the NEAR
 * pass), which keeps the force split an identity. Returns NULL on
 * n=0. */
static K26AstroPairWeight *build_pair_weights_(K26AstroWorld *world,
                                                int *out_n)
{
    int n = world->n_encounters;
    *out_n = 0;
    if (n <= 0) return NULL;
    if (n > world->cap_pair_weights) n = world->cap_pair_weights;
    if (n <= 0) return NULL;
    K26AstroPairWeight *w = world->pair_weights;
    for (int k = 0; k < n; k++) {
        w[k].i = world->encounters[k].i;
        w[k].j = world->encounters[k].j;
        w[k].k_weight = world->encounters[k].k_weight;
    }
    *out_n = n;
    return w;
}

void k26astro_rt_orbit_step_cb(double dt_s, void *user)
{
    K26AstroWorld *world = (K26AstroWorld *)user;
    if (!world) return;
    if (!(dt_s > 0.0)) return;

    /* No latch check here: the public advance entries guarantee the
     * latch is clear before any dispatch (protocol at the latch's
     * declaration, world_internal.h). This layer only writes it. */

    /* Encounter detection — populates world->encounters with K
     * weights baked in. */
    int n_enc = k26astro_mercurius_detect(world);

    K26AstroIntegrator base = world->grav.integrator;
    /* Split admission: Verlet base only. Verlet evaluates forces
     * through k26astro_grav_accel_total, which applies the MERCURIUS
     * context's K weights (force_direct.c), so its FAR pass
     * genuinely integrates the (1-K) portion. A WH base is excluded
     * because the WH interaction kick computes its own pair sum
     * (interaction_accel_, wisdom_holman.c) and never consults
     * state->mercurius, which breaks the split in both directions:
     * the FAR pass applies the full pair force, so a converging
     * NEAR pass would add the K-weighted portion again on top
     * (double-counting), and a failing NEAR pass fails the substep
     * while having contributed nothing, surfacing an integrator
     * failure on a healthy world. Combined with the detection
     * heuristic's separation-independence (the recorded defect at
     * k26astro_mercurius_hill_radius), any pair above the mass
     * threshold split on every WH step, so both directions were
     * live. Restoring a WH split means teaching the kick the pair
     * weights; that is scoped with the detector follow-up item. */
    int do_split = (n_enc > 0)
        && (base == K26ASTRO_INTEGRATOR_VERLET);

    if (!do_split) {
        /* Standard single-integrator step. On failure the substep
         * did not complete: latch the status and skip the mass
         * commit (its accumulator would step mass past the
         * failure). */
        int rc = k26astro_grav_step(&world->grav, dt_s);
        if (rc != K26ASTRO_E_OK) {
            world->substep_status = rc;
            return;
        }
        commit_vehicle_mass_(world, dt_s);
        return;
    }

    /* Paper-faithful MERCURIUS split (Rein et al. 2019 eq. 12-14).
     * n_enc > 0 here, so the preallocated buffer always comes back
     * non-NULL; the old alloc-failure fallback to a single-
     * integrator step is gone along with the allocation. */
    int n_w = 0;
    K26AstroPairWeight *w = build_pair_weights_(world, &n_w);

    K26AstroMercuriusContext far_ctx  = {
        .mode = K26ASTRO_MERCURIUS_FAR, .pair_weights = w, .n_pair_weights = n_w };
    K26AstroMercuriusContext near_ctx = {
        .mode = K26ASTRO_MERCURIUS_NEAR, .pair_weights = w, .n_pair_weights = n_w };

    /* Step 1: outer (Verlet) on FAR. On failure, stop before
     * the NEAR pass: the split's two integrations are halves of one
     * substep, and running the second half over the first's failed
     * state would step past the failure. The mercurius pointer is
     * always cleared before returning (it aims at stack locals). */
    world->grav.mercurius = &far_ctx;
    int rc = k26astro_grav_step(&world->grav, dt_s);
    if (rc != K26ASTRO_E_OK) {
        world->grav.mercurius = NULL;
        world->substep_status = rc;
        return;
    }

    /* Step 2: IAS15 sub-step on NEAR. Switch the integrator for
     * the inner pass, then restore. The restore and clear run on
     * the failure path too, so a latched failure never leaves the
     * split's temporary integrator or context behind.
     *
     * Epoch accounting: the two passes are halves of ONE substep
     * over the same dt_s, but each integrator bills its own dt
     * against state->t. The FAR pass above already billed the
     * substep's dt, so the NEAR pass's epoch advance is cancelled
     * by restoring t around it (on failure too: a failed NEAR pass
     * with partial internal progress must not add its partial bill
     * on top of the FAR pass's full one). One substep advances
     * simulated time once. */
    K26AstroEpoch t_billed = world->grav.t;
    world->grav.mercurius = &near_ctx;
    (void)k26astro_grav_set_integrator(&world->grav,
                                         K26ASTRO_INTEGRATOR_IAS15);
    rc = k26astro_grav_step(&world->grav, dt_s);
    (void)k26astro_grav_set_integrator(&world->grav, base);
    world->grav.mercurius = NULL;
    world->grav.t = t_billed;
    if (rc != K26ASTRO_E_OK) {
        /* The FAR pass's uncommitted dot_m stays in the vehicle
         * accumulators: the substep did not complete, so its mass
         * step is not taken. */
        world->substep_status = rc;
        return;
    }

    /* MERCURIUS split: commit mass once per outer substep (FAR pass).
     * The NEAR sub-step is internal to IAS15 and shouldn't double-count
     * the dot_m accumulator. */
    commit_vehicle_mass_(world, dt_s);
}
