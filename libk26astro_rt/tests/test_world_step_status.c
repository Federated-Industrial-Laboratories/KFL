/* test_world_step_status.c - substep failure statuses through the
 * exact stepping entry.
 *
 * The tick callbacks are void, so a failing k26astro_grav_step under
 * the orbit channel reaches the caller through the world's substep
 * latch. Gates:
 *
 *   1. Healthy stepping returns K26ASTRO_RT_OK and advances the
 *      epoch by exactly the requested simulated time.
 *   2. A world whose primary has zero GM drives the Wisdom-Holman
 *      Kepler drift to a genuine NO_CONVERGE on the first substep:
 *      k26astro_world_step_exact returns -K26ASTRO_RT_E_INTEGRATOR,
 *      the epoch does not advance, and the body states are
 *      byte-unchanged (the two-body interaction kick is zero).
 *   3. The latch does not stick: a repeated failing call reports the
 *      same status, and once the primary is repaired the same world
 *      steps cleanly again from where it stood.
 *   4. The clamped 3.1 entry's observable contract is unchanged: on
 *      the same failing world k26astro_world_step returns
 *      K26ASTRO_RT_OK.
 *   5. A mid-run integrator failure keeps completed substeps: IAS15
 *      under a small wall budget fails deep inside a long
 *      propagation. The same configuration driven at the grav
 *      layer directly pins the raw failure as
 *      K26ASTRO_E_TIME_BUDGET; the exact entry reports the world-
 *      path failure as -K26ASTRO_RT_E_INTEGRATOR; the epoch shows
 *      partial, nonzero progress.
 *   6. A vehicle registered on a world whose substep fails commits
 *      no mass on the failed substep: the accumulator survives
 *      untouched and vehicle and body masses hold. The commit runs
 *      on the next successful substep (positive control).
 *   7. Latch hygiene at both public entries: a failing clamped
 *      advance leaves the latch set (that entry never consumes);
 *      each entry's outermost clear is what lets a repaired world
 *      step again. Removing either entry's clear leaves the stale
 *      latch in place, the advance declines to dispatch, and the
 *      epoch stops advancing; the per-entry epoch assertions pin
 *      both clears independently. */
#include "k26astro_rt/world.h"
#include "k26astro_rt/scheduler.h"
#include "k26astro_grav/grav.h"
#include "k26astro_grav/ias15.h"
#include "k26astro_body/body.h"
#include "k26astro_core/pos.h"
#include "k26astro_vehicle/vehicle.h"

#include <math.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

static double epoch_s_(const K26AstroWorld *w)
{
    K26AstroEpoch e;
    assert(k26astro_world_now(w, &e) == K26ASTRO_RT_OK);
    return (double)e.days_since_J2000 * 86400.0 + e.seconds_of_day;
}

/* Earth-mass primary (or a zero-GM one) plus a 7000 km circular
 * craft. Returns the built world. */
static K26AstroWorld *make_two_body_(double primary_gm)
{
    K26AstroWorld *w = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                              K26ASTRO_COORDS_SECTOR_GRID);
    assert(w);

    K26AstroBody b;
    k26astro_body_init(&b);
    strncpy(b.name, "primary", sizeof b.name - 1);
    b.kind = K26ASTRO_BODY_PLANET;
    b.mass = 5.972e24;
    b.gm   = primary_gm;
    b.pos  = k26astro_pos_zero();
    b.parent_body_idx = -1;
    assert(k26astro_world_add_body(w, b) == 0);

    k26astro_body_init(&b);
    strncpy(b.name, "craft", sizeof b.name - 1);
    b.kind = K26ASTRO_BODY_MOON;
    b.mass = 1.0e3;
    b.gm   = 1.0;
    b.pos  = k26astro_pos_from_m(7.0e6, 0.0, 0.0);
    b.vel  = (K26V3){ 0.0, 7546.0533467, 0.0 };
    b.parent_body_idx = 0;
    assert(k26astro_world_add_body(w, b) == 1);
    return w;
}

int main(void)
{
    /* Gate 1: healthy stepping returns OK and advances the epoch. */
    {
        K26AstroWorld *w = make_two_body_(3.986004418e14);
        double t0 = epoch_s_(w);
        for (int i = 0; i < 3; i++) {
            assert(k26astro_world_step_exact(w, 0.1) == K26ASTRO_RT_OK);
        }
        double t1 = epoch_s_(w);
        assert(t1 - t0 == 0.3 || (t1 - t0 > 0.2999999999
                                  && t1 - t0 < 0.3000000001));
        k26astro_world_destroy(w);
        fprintf(stderr, "gate 1: healthy step_exact OK, epoch "
                        "advanced: OK\n");
    }

    /* Gates 2 to 4: zero-GM primary. The Wisdom-Holman drift's
     * universal-variable Kepler solve rejects mu <= 0, so the first
     * substep genuinely fails inside the integrator. */
    {
        K26AstroWorld *w = make_two_body_(0.0);
        double t0 = epoch_s_(w);

        K26AstroBody before[2];
        memcpy(&before[0], k26astro_world_body_at(w, 0), sizeof before[0]);
        memcpy(&before[1], k26astro_world_body_at(w, 1), sizeof before[1]);

        int rc = k26astro_world_step_exact(w, 0.1);
        fprintf(stderr, "gate 2: step_exact on zero-GM primary rc=%d "
                        "(want %d)\n", rc, -K26ASTRO_RT_E_INTEGRATOR);
        assert(rc == -K26ASTRO_RT_E_INTEGRATOR);
        assert(epoch_s_(w) == t0);
        assert(memcmp(&before[0], k26astro_world_body_at(w, 0),
                      sizeof before[0]) == 0);
        assert(memcmp(&before[1], k26astro_world_body_at(w, 1),
                      sizeof before[1]) == 0);
        fprintf(stderr, "gate 2: integrator status surfaced, epoch and "
                        "bodies held: OK\n");

        /* Gate 3: the latch clears per call; repair resumes. */
        rc = k26astro_world_step_exact(w, 0.1);
        assert(rc == -K26ASTRO_RT_E_INTEGRATOR);
        k26astro_world_body_at(w, 0)->gm = 3.986004418e14;
        assert(k26astro_world_step_exact(w, 0.1) == K26ASTRO_RT_OK);
        assert(epoch_s_(w) > t0);
        fprintf(stderr, "gate 3: failure latch clears; repaired world "
                        "steps: OK\n");
        k26astro_world_destroy(w);
    }
    {
        /* Gate 4: the clamped 3.1 entry still returns OK on a failing
         * world; its callers discard the return today and their
         * observable behaviour must hold. */
        K26AstroWorld *w = make_two_body_(0.0);
        assert(k26astro_world_step(w, 0.1) == K26ASTRO_RT_OK);
        assert(k26astro_world_step(w, 0.1) == K26ASTRO_RT_OK);
        k26astro_world_destroy(w);
        fprintf(stderr, "gate 4: clamped entry unchanged on failing "
                        "world: OK\n");
    }

    /* Gate 5: mid-run IAS15 wall-budget exhaustion. 100 simulated
     * days on a 98-minute orbit needs on the order of a million
     * substeps; a 10 ms budget trips deep inside the run, after
     * real progress. The fixed-rate channels are slowed first so
     * the no-op spin/render dispatch stays cheap over the long
     * simulated span. */
    {
        double span = 100.0 * 86400.0;

        /* Pin the raw failure class first. The world path below
         * hands this exact configuration to a single
         * k26astro_grav_step call (two bodies, central pair never
         * an encounter, so no split); driving the grav layer
         * directly with the same configuration shows the raw
         * status is the wall budget, not convergence. */
        {
            K26AstroWorld *w2 = make_two_body_(3.986004418e14);
            K26AstroGravState *g2 = k26astro_world_grav(w2);
            assert(g2);
            assert(k26astro_grav_set_integrator(g2,
                                                 K26ASTRO_INTEGRATOR_IAS15)
                   == K26ASTRO_E_OK);
            k26astro_grav_ias15_set_tol(g2, 1.0e-9);
            k26astro_grav_ias15_set_wall_budget(g2, 0.01);
            int raw = k26astro_grav_step(g2, span);
            fprintf(stderr, "gate 5: raw grav status=%d (want "
                            "TIME_BUDGET=%d)\n", raw,
                            K26ASTRO_E_TIME_BUDGET);
            assert(raw == K26ASTRO_E_TIME_BUDGET);
            k26astro_world_destroy(w2);
        }

        K26AstroWorld *w = make_two_body_(3.986004418e14);
        assert(k26astro_scheduler_set_spin_hz(w, 0.01) == K26ASTRO_RT_OK);
        assert(k26astro_scheduler_set_render_hz(w, 0.01) == K26ASTRO_RT_OK);
        K26AstroGravState *g = k26astro_world_grav(w);
        assert(g);
        assert(k26astro_grav_set_integrator(g, K26ASTRO_INTEGRATOR_IAS15)
               == K26ASTRO_E_OK);
        /* The advanced-API caller configures IAS15 itself. The
         * tolerance is set explicitly so this gate pins the budget
         * failure against a stated controller configuration rather
         * than the world's create-time default (numerically the
         * same 1e-9; k26astro_world_create mirrors the grav
         * library's init defaults). */
        k26astro_grav_ias15_set_tol(g, 1.0e-9);
        k26astro_grav_ias15_set_wall_budget(g, 0.01);

        double t0 = epoch_s_(w);
        int rc = k26astro_world_step_exact(w, span);
        double advanced = epoch_s_(w) - t0;
        fprintf(stderr, "gate 5: budgeted IAS15 rc=%d advanced=%.3f s "
                        "of %.0f s\n", rc, advanced, span);
        /* The raw status just pinned is K26ASTRO_E_TIME_BUDGET (6),
         * numerically equal to K26ASTRO_RT_E_FPU_RACE (6) in this
         * lib's code space. The boundary must translate, never
         * forward raw: a raw forward would surface this budget
         * failure as a phantom FPU-race report. */
        assert(rc == -K26ASTRO_RT_E_INTEGRATOR);
        assert(advanced > 0.0);
        assert(advanced < span);
        k26astro_world_destroy(w);
        fprintf(stderr, "gate 5: budget failure translated with partial "
                        "progress kept: OK\n");
    }

    /* Gate 6: no mass commit on a failed substep. A registered
     * vehicle's accumulator holds thrust-integrated dot_m; the
     * commit that consumes it closes a successful substep only. The
     * accumulator is seeded directly here, standing in for a thrust
     * callback's contribution during the substep's accel
     * evaluations. */
    {
        K26AstroWorld *w = make_two_body_(0.0);
        K26AstroVehicle *v = k26astro_vehicle_new();
        assert(v);
        k26astro_vehicle_set_dry_mass(v, 1000.0);
        k26astro_vehicle_bind_body(v, k26astro_world_body_at(w, 1));
        assert(k26astro_world_register_vehicle(w, v) == K26ASTRO_RT_OK);
        k26astro_vehicle_mass_accum_add(v, -1.0);

        double m_before    = k26astro_vehicle_mass_now(v);
        double body_before = k26astro_world_body_at(w, 1)->mass;
        int rc = k26astro_world_step_exact(w, 0.1);
        assert(rc == -K26ASTRO_RT_E_INTEGRATOR);
        assert(k26astro_vehicle_mass_now(v) == m_before);
        assert(k26astro_world_body_at(w, 1)->mass == body_before);
        assert(k26astro_vehicle_mass_accum_get(v) == -1.0);
        fprintf(stderr, "gate 6: failed substep committed no mass, "
                        "accumulator retained: OK\n");

        /* Positive control: the same accumulator commits on the
         * next successful substep, proving the assertions above
         * bite on a live path. */
        k26astro_world_body_at(w, 0)->gm = 3.986004418e14;
        assert(k26astro_world_step_exact(w, 0.1) == K26ASTRO_RT_OK);
        assert(fabs(k26astro_vehicle_mass_now(v) - 999.9) < 1.0e-9);
        assert(k26astro_vehicle_mass_accum_get(v) == 0.0);
        k26astro_world_unregister_vehicle(w, v);
        k26astro_vehicle_destroy(v);
        k26astro_world_destroy(w);
        fprintf(stderr, "gate 6: successful substep committed the "
                        "accumulator: OK\n");
    }

    /* Gate 7: each public entry's outermost latch clear, pinned
     * separately. The clamped entry never consumes the latch, so a
     * failing clamped advance leaves it set; only an entry's own
     * clear lets the repaired world step again. Removing either
     * entry's clear makes that entry's advance decline against the
     * stale latch and the epoch stops advancing, so each epoch
     * assertion below pins its own entry's clear. */
    {
        K26AstroWorld *w = make_two_body_(0.0);

        /* Stale latch via the clamped entry, then repair and step
         * through the clamped entry again. */
        assert(k26astro_world_step(w, 0.1) == K26ASTRO_RT_OK);
        k26astro_world_body_at(w, 0)->gm = 3.986004418e14;
        double t0 = epoch_s_(w);
        assert(k26astro_world_step(w, 0.1) == K26ASTRO_RT_OK);
        double d0 = epoch_s_(w) - t0;
        assert(d0 > 0.0999999999 && d0 < 0.1000000001);
        fprintf(stderr, "gate 7: clamped entry cleared the stale latch, "
                        "epoch advanced %.9f s: OK\n", d0);

        /* Stale latch again, then repair and step through the exact
         * entry. */
        k26astro_world_body_at(w, 0)->gm = 0.0;
        assert(k26astro_world_step(w, 0.1) == K26ASTRO_RT_OK);
        k26astro_world_body_at(w, 0)->gm = 3.986004418e14;
        double t1 = epoch_s_(w);
        assert(k26astro_world_step_exact(w, 0.1) == K26ASTRO_RT_OK);
        double d1 = epoch_s_(w) - t1;
        assert(d1 > 0.0999999999 && d1 < 0.1000000001);
        k26astro_world_destroy(w);
        fprintf(stderr, "gate 7: exact entry cleared the stale latch, "
                        "epoch advanced %.9f s: OK\n", d1);
    }

    printf("test_world_step_status: OK\n");
    return 0;
}
