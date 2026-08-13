/* orbit_step.c — orbit-channel callback (paper-faithful MERCURIUS).
 *
 * Runs every wallclock advance with the actual elapsed dt. Drives
 * the world's integrator (WH, IAS15, Verlet, RK4, RK45, or MERCURIUS).
 *
 * Paper-faithful MERCURIUS split, after Rein, Hernandez, Tamayo
 * et al. 2019 (MNRAS 485(4):5490-5497), section 2:
 *   1. Detect close-encounter pairs (k26astro_mercurius_detect).
 *      Each pair carries a K(y) weight computed at detect time.
 *   2. If no encounters, or the base integrator is not admitted to
 *      the split (admission note at do_split below): single
 *      full-force step.
 *   3. Otherwise, one kick-drift-kick composition of the substep's
 *      dt, splitting the acceleration field into
 *        a_far  = (1-K)-weighted encounter pairs + full weight on
 *                 every other non-central pair + perturbations,
 *        a_near = K-weighted encounter pairs + the central pairs
 *                 at full weight (the drift owns the central
 *                 attraction on every admitted base, per the
 *                 paper's Kepler part; central_plus1 in forces.h),
 *      with a_far + a_near summing to the unsplit total by
 *      construction:
 *        a. Half-kick: v += (dt/2) * a_far at the current
 *           positions (state->mercurius = FAR context).
 *        b. Drift: one IAS15 step over the full dt on the NEAR
 *           context's field. This is the only pass that advances
 *           positions and the epoch; it carries the kinetic motion
 *           and resolves the encounter-pair forces adaptively at
 *           the evolving positions.
 *        c. Half-kick: v += (dt/2) * a_far at the drifted
 *           positions. Clear state->mercurius.
 *
 *      This is the paper's operator structure: the drift owns the
 *      central attraction and the encounter terms, the kick carries
 *      only the (1-K)-weighted pair terms. Two earlier revisions
 *      departed from it. One ran the base integrator over dt (FAR
 *      field) and then IAS15 over the same dt (NEAR field) as two
 *      FULL steps in sequence; the force weights summed to the
 *      identity, but each pass advanced positions by its own
 *      kinetic drift, so every position moved at twice its velocity
 *      on split substeps (measured ratio 2.0000 over one substep).
 *      The other kept the central pairs in the half-kicks on a
 *      Verlet base, inverting the paper's operator assignment
 *      (central attraction impulsive, only the pair term in the
 *      adaptive drift); that shape measured roughly twice as far
 *      from a tight reference as not splitting at all. The
 *      composition here integrates the kinetic term exactly once
 *      and drifts the central attraction on every admitted base.
 *      The force arithmetic (split accelerations summing to the
 *      unsplit total, and the split step landing on the single-step
 *      trajectory) is gated by test_mercurius_force_arithmetic. */
#include "encounter_internal.h"

#include "k26astro_grav/grav.h"
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
    /* Split admission. Verlet: always (it has no hard-wired
     * primary; the composition below never runs the base
     * integrator, and the near drift takes the central pairs at
     * whichever index the detector found). Wisdom-Holman: only when
     * the detector's central body is body 0, the Kepler primary
     * hard-wired into the WH drift (wisdom_holman.c,
     * mu0 = b[0].gm). With the largest mass elsewhere, the base's
     * own unsplit steps and the split's central-pair partition
     * would disagree about which body is primary, so WH takes
     * single full-force steps there, where its primary choice is
     * its own pre-existing approximation and no split arithmetic
     * depends on it (pinned bitwise by gate 6 of
     * test_mercurius_force_arithmetic). IAS15 resolves encounters
     * itself; RK4/RK45 stay whole. */
    int wh_split_ok = (base == K26ASTRO_INTEGRATOR_WH)
        && (world->mercurius_central_idx == 0);
    int do_split = (n_enc > 0)
        && (base == K26ASTRO_INTEGRATOR_VERLET || wh_split_ok);

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

    /* Paper-faithful MERCURIUS split (Rein et al. 2019, section 2).
     * n_enc > 0 here, so the preallocated buffer always comes back
     * non-NULL; the old alloc-failure fallback to a single-
     * integrator step is gone along with the allocation. */
    int n_w = 0;
    K26AstroPairWeight *w = build_pair_weights_(world, &n_w);

    /* The near drift owns the central pairs on every admitted base:
     * the inner integrator carries the central attraction (the
     * paper's Kepler part) while the far kick carries only the
     * (1-K)-weighted pair terms. The detector guarantees
     * mercurius_central_idx >= 0 whenever n_enc > 0, and the WH
     * admission above additionally pins it to 0 on that base. */
    int central_plus1 = world->mercurius_central_idx + 1;
    K26AstroMercuriusContext far_ctx  = {
        .mode = K26ASTRO_MERCURIUS_FAR, .pair_weights = w,
        .n_pair_weights = n_w, .central_plus1 = central_plus1 };
    K26AstroMercuriusContext near_ctx = {
        .mode = K26ASTRO_MERCURIUS_NEAR, .pair_weights = w,
        .n_pair_weights = n_w, .central_plus1 = central_plus1 };

    /* The far-field kick writes through the grav state's kick
     * scratch (the same buffer Verlet's kick uses; no integrator
     * runs while the kick reads it). With the state reserved at
     * body-add time this allocates nothing. */
    int n = world->grav.n_bodies;
    if (world->grav.scratch_cap < n) {
        int rrc = k26astro_grav_state_reserve(&world->grav);
        if (rrc != K26ASTRO_E_OK) {
            world->substep_status = rrc;
            return;
        }
    }
    K26V3 *a_far = world->grav.scratch_accel;
    K26AstroBody *b = world->grav.bodies;

    /* Half-kick on the far field at the current positions. The
     * mercurius pointer aims at stack locals; every return path
     * below clears it. */
    world->grav.mercurius = &far_ctx;
    k26astro_grav_accel_total(&world->grav, a_far);
    for (int i = 0; i < n; i++) {
        b[i].vel.x += 0.5 * dt_s * a_far[i].x;
        b[i].vel.y += 0.5 * dt_s * a_far[i].y;
        b[i].vel.z += 0.5 * dt_s * a_far[i].z;
    }

    /* Drift: one IAS15 step over the substep's full dt on the NEAR
     * field. The only pass that advances positions, and the only
     * one that bills the epoch: the kicks are instantaneous. On
     * failure the substep is incomplete; the latched status stops
     * the advance at this substep, the epoch reflects the drift's
     * completed internal progress (the inner integrator's own
     * contract), and the first half-kick stands as the failed
     * substep's partial write, with no time billed for it. */
    world->grav.mercurius = &near_ctx;
    (void)k26astro_grav_set_integrator(&world->grav,
                                         K26ASTRO_INTEGRATOR_IAS15);
    int rc = k26astro_grav_step(&world->grav, dt_s);
    (void)k26astro_grav_set_integrator(&world->grav, base);
    if (rc != K26ASTRO_E_OK) {
        /* The half-kick's uncommitted dot_m stays in the vehicle
         * accumulators: the substep did not complete, so its mass
         * step is not taken. */
        world->grav.mercurius = NULL;
        world->substep_status = rc;
        return;
    }

    /* Half-kick on the far field at the drifted positions. */
    world->grav.mercurius = &far_ctx;
    k26astro_grav_accel_total(&world->grav, a_far);
    world->grav.mercurius = NULL;
    for (int i = 0; i < n; i++) {
        b[i].vel.x += 0.5 * dt_s * a_far[i].x;
        b[i].vel.y += 0.5 * dt_s * a_far[i].y;
        b[i].vel.z += 0.5 * dt_s * a_far[i].z;
    }

    /* Commit mass once per substep, matching what the same base's
     * unsplit step commits. The split evaluates the far field twice
     * (once per half-kick), so any propulsion dot_m callback ran
     * twice; the NEAR drift is internal to IAS15 and runs no user
     * perturbations (accel_total's rule). Rule, per base: an
     * unsplit Verlet step evaluates the field once and commits dt,
     * so the split on a Verlet base commits dt/2 to keep the
     * committed flow at one evaluation's worth; an unsplit WH step
     * evaluates the interaction field twice (both half-kicks) and
     * commits dt, so the split on a WH base commits dt as well.
     * Gate 5 of test_mercurius_force_arithmetic pins split ==
     * unsplit committed mass on both bases. */
    if (base == K26ASTRO_INTEGRATOR_WH)
        commit_vehicle_mass_(world, dt_s);
    else
        commit_vehicle_mass_(world, 0.5 * dt_s);
}
