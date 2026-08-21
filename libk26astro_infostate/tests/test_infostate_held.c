/* test_infostate_held.c: the held last-known answer.
 *
 * k26astro_infostate_observe_held returns the newest history entry at
 * or before the clock time asked about, exactly as it was pushed, with
 * that entry's age. It is the answer for an observer whose entries
 * arrive intermittently, where a retarded-time solution has nothing to
 * read: at close range the light time is microseconds while the gap
 * between entries is a control period, so the instant the solver would
 * need falls past the newest entry there is.
 *
 * Scenarios:
 *   (1) An empty track, and a clock time before every retained entry:
 *       valid = 0, and nothing else read.
 *   (2) A clock time landing exactly on an entry: that entry, age 0.
 *   (3) A clock time between two entries: the earlier of the two, aged
 *       by the gap, and not an interpolation between them.
 *   (4) A clock time past the newest entry: the newest entry, aged by
 *       the whole gap, where the retarded-time call on the same
 *       history and the same instant reads unavailable. That pair is
 *       the whole reason the second answer exists.
 *   (5) The ring having wrapped: a clock time before the oldest entry
 *       retained reads unavailable, since the entry that would answer
 *       has been dropped.
 *   (6) The retarded-time call is unmoved: one history, two calls,
 *       neither changing what the other returns.
 */

#include "k26astro_infostate/infostate.h"
#include "k26astro_infostate/infostate_consts.h"

#include "k26astro_vehicle/vehicle.h"
#include "k26astro_body/body.h"
#include "k26astro_core/consts.h"
#include "k26astro_core/epoch.h"
#include "k26astro_core/pos.h"
#include "k26m3d.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    return 1; } } while (0)

static K26AstroEpoch at_(double dt_s)
{
    K26AstroEpoch e = k26astro_epoch_j2000_tt();
    k26astro_epoch_add_seconds(&e, dt_s);
    return e;
}

int main(void)
{
    K26AstroBody obs_body, tgt_body;
    K26AstroVehicle *observer, *target;
    K26AstroInfostate *s;
    K26AstroInfostateObservation o;
    /* A kilometre, which is the range craft work at and where the
     * light time is three microseconds against entries a tenth of a
     * second apart. */
    const double range_m = 1000.0;
    const double step_s  = 0.1;
    const double speed   = 50.0;      /* metres a second, along y */

    memset(&obs_body, 0, sizeof(obs_body));
    memset(&tgt_body, 0, sizeof(tgt_body));
    observer = k26astro_vehicle_new();
    target   = k26astro_vehicle_new();
    CHECK(observer != NULL && target != NULL);
    k26astro_vehicle_bind_body(observer, &obs_body);
    k26astro_vehicle_bind_body(target,   &tgt_body);
    obs_body.pos = k26astro_pos_from_m(0.0, 0.0, 0.0);

    s = k26astro_infostate_new(observer, 4);
    CHECK(s != NULL);

    /* ---- 1: nothing pushed --------------------------------------- */
    o = k26astro_infostate_observe_held(s, target, at_(0.0),
            K26ASTRO_INFOSTATE_MODALITY_RADAR);
    CHECK(o.valid == 0);
    CHECK(o.age_s == 0.0);
    CHECK(o.range_m == 0.0);

    /* Four entries a tenth of a second apart, the target crossing at
     * fifty metres a second, ring capacity four so the fifth wraps. */
    for (int i = 0; i < 4; i++) {
        K26V3 r = { range_m, speed * (double)i * step_s, 0.0 };
        K26V3 v = { 0.0, speed, 0.0 };
        k26astro_infostate_target_push(s, target,
            at_((double)i * step_s), r, v);
    }

    /* ---- 1b: before every entry ---------------------------------- */
    o = k26astro_infostate_observe_held(s, target, at_(-0.05),
            K26ASTRO_INFOSTATE_MODALITY_RADAR);
    CHECK(o.valid == 0);

    /* ---- 2: exactly on an entry ---------------------------------- */
    o = k26astro_infostate_observe_held(s, target, at_(2.0 * step_s),
            K26ASTRO_INFOSTATE_MODALITY_RADAR);
    CHECK(o.valid == 1);
    CHECK(o.age_s == 0.0);
    CHECK(o.position.y == speed * 2.0 * step_s);
    CHECK(o.modality == K26ASTRO_INFOSTATE_MODALITY_RADAR);
    CHECK(o.iters == 0);
    CHECK(fabs(o.range_m - sqrt(range_m * range_m
                                + o.position.y * o.position.y)) < 1.0e-6);

    /* ---- 3: between two entries ---------------------------------- */
    /* Three quarters of the way from the second entry to the third.
     * The answer is the second entry, aged by that fraction of the
     * step, and emphatically not the interpolation between them: an
     * interpolated position would carry three quarters of a step's
     * motion, which is nearly four metres away. */
    {
        double when = 2.0 * step_s + 0.75 * step_s;
        o = k26astro_infostate_observe_held(s, target, at_(when),
                K26ASTRO_INFOSTATE_MODALITY_RADAR);
        CHECK(o.valid == 1);
        CHECK(fabs(o.age_s - 0.75 * step_s) < 1.0e-12);
        CHECK(o.position.y == speed * 2.0 * step_s);
        CHECK(fabs(o.position.y - speed * (2.0 + 0.75) * step_s)
              > 1.0);
    }

    /* ---- 4: past the newest entry, where the other call stops ---- */
    {
        double when = 3.0 * step_s + 5.0 * step_s;
        K26AstroInfostateObservation retarded;

        o = k26astro_infostate_observe_held(s, target, at_(when),
                K26ASTRO_INFOSTATE_MODALITY_RADAR);
        CHECK(o.valid == 1);
        CHECK(fabs(o.age_s - 5.0 * step_s) < 1.0e-12);
        CHECK(o.position.y == speed * 3.0 * step_s);
        /* The light time to that position is microseconds while the
         * entry is half a second old, so the retarded-time call has
         * no state to read and says so. */
        CHECK(o.age_s > o.range_m / K26A_C);
        retarded = k26astro_infostate_observe(s, target, at_(when),
                K26ASTRO_INFOSTATE_MODALITY_RADAR);
        CHECK(retarded.valid == 0);
    }

    /* ---- 5: the ring has wrapped -------------------------------- */
    /* A fifth entry drops the first, so an instant the dropped entry
     * would have answered reads unavailable rather than reaching back
     * past what is retained. */
    {
        K26V3 r = { range_m, speed * 4.0 * step_s, 0.0 };
        K26V3 v = { 0.0, speed, 0.0 };
        k26astro_infostate_target_push(s, target,
            at_(4.0 * step_s), r, v);
        CHECK(k26astro_infostate_history_length(s, target) == 4);

        o = k26astro_infostate_observe_held(s, target, at_(0.05),
                K26ASTRO_INFOSTATE_MODALITY_RADAR);
        CHECK(o.valid == 0);

        o = k26astro_infostate_observe_held(s, target, at_(0.15),
                K26ASTRO_INFOSTATE_MODALITY_RADAR);
        CHECK(o.valid == 1);
        CHECK(o.position.y == speed * 1.0 * step_s);
    }

    /* ---- 6: the two answers over one history -------------------- */
    /* A dense history a light time deep, which is the shape the
     * retarded-time call was built for: it answers, the held call
     * answers, and neither has moved the other. */
    {
        K26AstroInfostate *dense = k26astro_infostate_new(observer, 256);
        K26AstroInfostateObservation a, b, again;
        const double far_m = K26A_C;     /* one light second exactly */
        K26V3 v = { 0.0, 0.0, 0.0 };

        CHECK(dense != NULL);
        for (int i = 0; i <= 40; i++) {
            K26V3 r = { far_m, 0.0, 0.0 };
            k26astro_infostate_target_push(dense, target,
                at_(-2.0 + 0.05 * (double)i), r, v);
        }
        a = k26astro_infostate_observe(dense, target, at_(0.0),
                K26ASTRO_INFOSTATE_MODALITY_IR);
        b = k26astro_infostate_observe_held(dense, target, at_(0.0),
                K26ASTRO_INFOSTATE_MODALITY_IR);
        again = k26astro_infostate_observe(dense, target, at_(0.0),
                K26ASTRO_INFOSTATE_MODALITY_IR);
        CHECK(a.valid == 1 && b.valid == 1);
        CHECK(fabs(a.age_s - 1.0) < 1.0e-9);   /* the light second */
        CHECK(b.age_s == 0.0);                 /* an entry at this instant */
        CHECK(again.valid == a.valid);
        CHECK(again.age_s == a.age_s);
        CHECK(again.position.x == a.position.x);
        k26astro_infostate_destroy(dense);
    }

    /* ---- Null and stale inputs ---------------------------------- */
    o = k26astro_infostate_observe_held(NULL, target, at_(0.0),
            K26ASTRO_INFOSTATE_MODALITY_NONE);
    CHECK(o.valid == 0);
    o = k26astro_infostate_observe_held(s, NULL, at_(0.0),
            K26ASTRO_INFOSTATE_MODALITY_NONE);
    CHECK(o.valid == 0);

    k26astro_infostate_destroy(s);
    k26astro_vehicle_destroy(observer);
    k26astro_vehicle_destroy(target);

    printf("test_infostate_held: the held last-known answer, six "
           "scenarios: OK\n");
    return 0;
}
