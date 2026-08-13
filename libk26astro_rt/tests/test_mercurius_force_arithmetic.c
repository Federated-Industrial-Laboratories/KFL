/* test_mercurius_force_arithmetic.c - the split's force arithmetic.
 *
 * The MERCURIUS split's health cannot be read off the fault surface:
 * a split that double-counts a pair force, drops one, or drifts the
 * kinetic term twice still steps "successfully" and reports no
 * fault. These gates check the arithmetic itself.
 *
 *   Gate 1 (field identity): with a genuinely detected encounter
 *     pair, the FAR-context and NEAR-context accelerations sum to
 *     the unsplit acceleration, body by body, for both context
 *     shapes (central_plus1 = 0, the pure pair-weight shape of the
 *     caller-side primitive; central_plus1 set, the production
 *     shape in which the drift owns the central pairs on every
 *     admitted base).
 *
 *   Gate 2 (detector honesty): the same pair of masses is detected
 *     or not detected by closeness, not by mass ratio alone: a
 *     distant pair does not split, a close pair does, a deep pair
 *     carries K = 1, the unbound fallback follows its stated rule
 *     (current central-relative distance), and the mutual Hill
 *     radius is continuous at the parabolic boundary (the bound
 *     side's distance cap and the unbound branch agree at escape
 *     speed).
 *
 *   Gate 3 (perturbation split): with a user perturbation
 *     registered, the FAR and NEAR context fields still sum to the
 *     unsplit total, the perturbation contributes to FAR exactly
 *     once, and the NEAR field is bit-identical with or without the
 *     perturbation registered. This is the gate that goes red on a
 *     perturbation double-count across the passes.
 *
 *   Gate 4 (split-step near-identity): for each admitted base
 *     (Verlet; WH with the largest mass at body 0), a genuinely
 *     splitting trajectory lands on the unsplit high-accuracy IAS15
 *     trajectory of the same configuration within stated per-base
 *     bounds, and bills the same epoch. This is the gate that goes
 *     red on force double-counting (a pass applying full pair force
 *     under an active context), on zero-weighting (a pass dropping
 *     its portion), on kinetic double-drift (two passes each
 *     advancing positions), on central-pair ownership swaps (the
 *     drift must own the central attraction on both bases), and on
 *     kick asymmetry (both half-kicks at dt/2).
 *
 *   Gate 5 (mass commit): the committed vehicle mass of a splitting
 *     substep equals the same base's unsplit commit, on both bases,
 *     within round-off (the per-base rule stated in orbit_step.c).
 *
 *   Gate 6 (split readmission): on a WH base with the largest mass
 *     not at body 0, the split declines even though an encounter is
 *     detected, and the trajectory is bit-identical to the unsplit
 *     WH steps of the same configuration.
 *
 * Gate 4 bounds: measured on the honest tree, both bases sit at a
 * position metric of 8.13e-15 and a velocity metric of 3.09e-12 at
 * this configuration, and the two bases measure identically (the
 * split composition is base-independent once the drift owns the
 * central pairs, so equally tight per-base bounds are themselves
 * part of the pin: an ownership change degrades both). The bounds
 * are set roughly 40x above the measurement (position 3e-13,
 * velocity 1.2e-10), far below the metrics measured under the
 * mutations this gate must catch: a central-pair ownership swap
 * lands at a position metric of 2.4e-11, an asymmetric first-order
 * kick at 4.8e-9, and the retired two-full-integrator composition
 * at a position metric near 1.
 *
 * Reference: Rein, Hernandez, Tamayo et al. (2019), MNRAS
 * 485(4):5490-5497, "Hybrid Symplectic Integrators for Planetary
 * Dynamics." */
#include "k26astro_rt/world.h"
#include "k26astro_grav/grav.h"
#include "k26astro_grav/forces.h"
#include "k26astro_grav/ias15.h"
#include "k26astro_grav/perturb.h"
#include "k26astro_body/body.h"
#include "k26astro_core/pos.h"
#include "k26astro_vehicle/vehicle.h"

#include "encounter_internal.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static const double GM_SUN = 1.32712440018e20;
static const double M_SUN  = 1.98892e30;
static const double AU     = 1.495978707e11;

static int failures_ = 0;
#define CHECK(cond, ...) do { \
        if (!(cond)) { \
            failures_++; \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
        } \
    } while (0)

static K26AstroBody body_(const char *nm, double mass, double gm,
                          double x, double vy, int parent)
{
    K26AstroBody b;
    k26astro_body_init(&b);
    strncpy(b.name, nm, sizeof b.name - 1);
    b.kind = K26ASTRO_BODY_PLANET;
    b.mass = mass;
    b.gm   = gm;
    b.pos  = k26astro_pos_from_m(x, 0.0, 0.0);
    b.vel  = (K26V3){ 0.0, vy, 0.0 };
    b.parent_body_idx = parent;
    return b;
}

/* Sun + two planets of `m_frac` solar masses each, on circular
 * orbits at 1 AU and (1 + sep_au) AU. With both planets on circular
 * central-body orbits, each vis-viva semi-major axis equals its
 * radius, so the honest closeness ratio is
 *   y = sep / (0.5 * (a1 + a2) * cbrt(2 m_frac / 3)).  */
static K26AstroWorld *pair_world_(double m_frac, double sep_au)
{
    K26AstroWorld *w = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                             K26ASTRO_COORDS_SECTOR_GRID);
    if (!w) return NULL;
    double r2 = (1.0 + sep_au) * AU;
    k26astro_world_add_body(w, body_("sun", M_SUN, GM_SUN, 0.0, 0.0, -1));
    k26astro_world_add_body(w, body_("p1", m_frac * M_SUN, m_frac * GM_SUN,
                                     AU, sqrt(GM_SUN / AU), 0));
    k26astro_world_add_body(w, body_("p2", m_frac * M_SUN, m_frac * GM_SUN,
                                     r2, sqrt(GM_SUN / r2), 0));
    return w;
}

static double v3_dist_(K26V3 a, K26V3 b)
{
    K26V3 d = { a.x - b.x, a.y - b.y, a.z - b.z };
    return sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

/* ---- Gate 1: FAR + NEAR = total, per body, both context shapes -- */
static void gate1_field_identity_(void)
{
    /* sep 0.361 AU puts the honest y near 3.5: inside the window,
     * 0 < K < 1, both passes carry force. */
    K26AstroWorld *w = pair_world_(1.0e-3, 0.361);
    int n_enc = k26astro_mercurius_detect(w);
    CHECK(n_enc == 1, "expected 1 encounter, got %d", n_enc);
    CHECK(w->mercurius_central_idx == 0,
          "central idx %d, want 0", w->mercurius_central_idx);
    double K = (n_enc == 1) ? w->encounters[0].k_weight : -1.0;
    CHECK(K > 0.0 && K < 1.0, "K = %.17g not in (0,1)", K);

    K26AstroPairWeight pw = { w->encounters[0].i, w->encounters[0].j, K };
    K26V3 a_full[3], a_far[3], a_near[3];
    k26astro_grav_accel_total(&w->grav, a_full);

    for (int shape = 0; shape < 2; shape++) {
        /* 0 = pure pair-weight shape (caller-side primitive);
         * 1 = production shape, drift-owned central pairs
         *     (central at body 0 in this fixture). */
        int central_plus1 = shape;
        K26AstroMercuriusContext far_ctx = {
            .mode = K26ASTRO_MERCURIUS_FAR, .pair_weights = &pw,
            .n_pair_weights = 1, .central_plus1 = central_plus1 };
        K26AstroMercuriusContext near_ctx = {
            .mode = K26ASTRO_MERCURIUS_NEAR, .pair_weights = &pw,
            .n_pair_weights = 1, .central_plus1 = central_plus1 };
        w->grav.mercurius = &far_ctx;
        k26astro_grav_accel_total(&w->grav, a_far);
        w->grav.mercurius = &near_ctx;
        k26astro_grav_accel_total(&w->grav, a_near);
        w->grav.mercurius = NULL;

        double worst = 0.0;
        for (int i = 0; i < 3; i++) {
            K26V3 sum = { a_far[i].x + a_near[i].x,
                          a_far[i].y + a_near[i].y,
                          a_far[i].z + a_near[i].z };
            double mag = sqrt(a_full[i].x * a_full[i].x
                            + a_full[i].y * a_full[i].y
                            + a_full[i].z * a_full[i].z);
            double resid = v3_dist_(sum, a_full[i]);
            double rel = (mag > 0.0) ? resid / mag : resid;
            if (rel > worst) worst = rel;
        }
        fprintf(stderr, "gate1 shape=%d (central_plus1): "
                "max |a_far+a_near-a_full|/|a_full| = %.3e\n",
                central_plus1, worst);
        CHECK(worst <= 1.0e-14, "field identity residual %.3e", worst);
    }
    k26astro_world_destroy(w);
}

/* ---- Gate 2: detection by closeness, not by mass ratio ---------- */
static void gate2_detector_honesty_(void)
{
    /* Distant: same masses, separation 1e15 m (~6685 AU). */
    {
        K26AstroWorld *w = pair_world_(0.05, 1.0e15 / AU);
        double rh = k26astro_mercurius_hill_radius(
                &w->grav.bodies[1], &w->grav.bodies[2], &w->grav.bodies[0]);
        double d = 1.0e15;
        double y = d / rh;
        int n_enc = k26astro_mercurius_detect(w);
        fprintf(stderr, "gate2 distant pair: d=%.3e m r_hill=%.3e m "
                "y=%.3f -> %d encounter(s)\n", d, rh, y, n_enc);
        CHECK(rh > 0.0, "degenerate hill radius");
        CHECK(y >= w->mercurius_outer_factor, "distant pair y=%.3f "
              "inside the window", y);
        CHECK(n_enc == 0, "distant pair detected (%d)", n_enc);
        k26astro_world_destroy(w);
    }
    /* Transition: y near 3.5 for the 1e-3 solar-mass pair. */
    {
        K26AstroWorld *w = pair_world_(1.0e-3, 0.361);
        int n_enc = k26astro_mercurius_detect(w);
        double K = (n_enc == 1) ? w->encounters[0].k_weight : -1.0;
        double rh = k26astro_mercurius_hill_radius(
                &w->grav.bodies[1], &w->grav.bodies[2], &w->grav.bodies[0]);
        fprintf(stderr, "gate2 transition pair: d=%.3e m r_hill=%.3e m "
                "y=%.3f K=%.4f -> %d encounter(s)\n",
                0.361 * AU, rh, 0.361 * AU / rh, K, n_enc);
        CHECK(n_enc == 1, "transition pair not detected (%d)", n_enc);
        CHECK(K > 0.0 && K < 1.0, "transition K=%.17g not in (0,1)", K);
        k26astro_world_destroy(w);
    }
    /* Deep: y under y_inner carries K = 1 exactly. */
    {
        K26AstroWorld *w = pair_world_(1.0e-3, 0.191);
        int n_enc = k26astro_mercurius_detect(w);
        double K = (n_enc == 1) ? w->encounters[0].k_weight : -1.0;
        fprintf(stderr, "gate2 deep pair: %d encounter(s), K=%.4f\n",
                n_enc, K);
        CHECK(n_enc == 1, "deep pair not detected (%d)", n_enc);
        CHECK(K == 1.0, "deep pair K=%.17g, want 1.0", K);
        k26astro_world_destroy(w);
    }
    /* Unbound fallback: p2 at 1.5x escape speed has no finite
     * semi-major axis; the stated rule scales its Hill contribution
     * by its current central-relative distance instead. */
    {
        K26AstroWorld *w = pair_world_(1.0e-3, 0.361);
        K26AstroBody *p2 = k26astro_world_body_at(w, 2);
        double r2 = (1.0 + 0.361) * AU;
        p2->vel.y = 1.5 * sqrt(2.0 * GM_SUN / r2);
        double rh = k26astro_mercurius_hill_radius(
                &w->grav.bodies[1], &w->grav.bodies[2], &w->grav.bodies[0]);
        double m_sum = w->grav.bodies[1].mass + w->grav.bodies[2].mass;
        double want = 0.5 * (AU + r2) * cbrt(m_sum / (3.0 * M_SUN));
        fprintf(stderr, "gate2 unbound fallback: r_hill=%.6e m "
                "stated-rule value=%.6e m\n", rh, want);
        CHECK(isfinite(rh) && rh > 0.0, "unbound fallback degenerate");
        CHECK(fabs(rh - want) <= 1.0e-12 * want,
              "fallback deviates from its stated rule");
        k26astro_world_destroy(w);
    }
    /* Parabolic-boundary continuity: the bound side's length scale
     * is capped at the current central-relative distance, the same
     * quantity the unbound branch returns, so the mutual Hill
     * radius is IDENTICAL just below escape speed and at escape
     * speed (the rule at hill_length_scale_, encounter.c). Without
     * the cap the just-below-escape orbit's semi-major axis grows
     * without bound and a distant near-parabolic pair classifies as
     * a deep encounter. */
    {
        double rh[2] = { 0.0, 0.0 };
        for (int k = 0; k < 2; k++) {
            K26AstroWorld *w = pair_world_(1.0e-3, 1.0e13 / AU);
            K26AstroBody *p2 = k26astro_world_body_at(w, 2);
            double r2 = AU + 1.0e13;
            double f  = (k == 0) ? (1.0 - 1.0e-12) : 1.0;
            p2->vel.y = f * sqrt(2.0 * GM_SUN / r2);
            rh[k] = k26astro_mercurius_hill_radius(
                    &w->grav.bodies[1], &w->grav.bodies[2],
                    &w->grav.bodies[0]);
            int n_enc = k26astro_mercurius_detect(w);
            CHECK(n_enc == 0, "distant near-parabolic pair split "
                  "(f=%.12f, %d encounter(s))", f, n_enc);
            k26astro_world_destroy(w);
        }
        fprintf(stderr, "gate2 parabolic continuity: r_hill just "
                "below escape %.17g m, at escape %.17g m\n",
                rh[0], rh[1]);
        CHECK(rh[0] == rh[1], "r_hill discontinuous at the "
              "parabolic boundary: %.17g vs %.17g", rh[0], rh[1]);
    }
}

/* ---- Gate 3: perturbations split FAR-only, once ----------------- */

static int  pert_calls_;

/* Constant, recognisable non-gravitational push on body 1. Large
 * against the gate tolerances (the push is about 2e-3 of body 1's
 * gravitational acceleration; a double-count breaks the field
 * identity at that scale, ten orders above the 1e-14 bound). */
static void pert_push_(const K26AstroGravState *s,
                       const K26AstroGravView *v, K26V3 *accel, void *ctx)
{
    (void)s; (void)ctx;
    if (v->n > 1) {
        accel[1].x += 1.0e-5;
        accel[1].y += 2.0e-5;
    }
    pert_calls_++;
}

static void gate3_perturbation_split_(void)
{
    K26AstroWorld *w = pair_world_(1.0e-3, 0.361);
    int n_enc = k26astro_mercurius_detect(w);
    CHECK(n_enc == 1, "expected 1 encounter, got %d", n_enc);
    double K = (n_enc == 1) ? w->encounters[0].k_weight : -1.0;
    CHECK(K > 0.0 && K < 1.0, "K = %.17g not in (0,1)", K);

    K26AstroPairWeight pw = { w->encounters[0].i, w->encounters[0].j, K };
    /* The production context shape: the drift owns the central
     * pairs (orbit_step.c sets central_plus1 on every admitted
     * base). */
    int central_plus1 = w->mercurius_central_idx + 1;
    K26AstroMercuriusContext far_ctx = {
        .mode = K26ASTRO_MERCURIUS_FAR, .pair_weights = &pw,
        .n_pair_weights = 1, .central_plus1 = central_plus1 };
    K26AstroMercuriusContext near_ctx = {
        .mode = K26ASTRO_MERCURIUS_NEAR, .pair_weights = &pw,
        .n_pair_weights = 1, .central_plus1 = central_plus1 };

    /* Context fields before the perturbation exists. */
    K26V3 a_far0[3], a_near0[3];
    w->grav.mercurius = &far_ctx;
    k26astro_grav_accel_total(&w->grav, a_far0);
    w->grav.mercurius = &near_ctx;
    k26astro_grav_accel_total(&w->grav, a_near0);
    w->grav.mercurius = NULL;

    int rc = k26astro_grav_register_perturb(&w->grav, pert_push_, NULL);
    CHECK(rc == 0, "register_perturb rc=%d", rc);

    /* Unsplit total: the perturbation contributes exactly once. */
    K26V3 a_full[3], a_far1[3], a_near1[3];
    pert_calls_ = 0;
    k26astro_grav_accel_total(&w->grav, a_full);
    CHECK(pert_calls_ == 1, "unsplit total ran the perturbation %d "
          "times, want 1", pert_calls_);

    /* FAR carries it once; NEAR must not run it at all. */
    pert_calls_ = 0;
    w->grav.mercurius = &far_ctx;
    k26astro_grav_accel_total(&w->grav, a_far1);
    CHECK(pert_calls_ == 1, "FAR pass ran the perturbation %d times, "
          "want 1", pert_calls_);
    pert_calls_ = 0;
    w->grav.mercurius = &near_ctx;
    k26astro_grav_accel_total(&w->grav, a_near1);
    w->grav.mercurius = NULL;
    CHECK(pert_calls_ == 0, "NEAR pass ran the perturbation %d times, "
          "want 0", pert_calls_);

    /* Field identity with the perturbation registered. A
     * double-count across the passes (the perturbation entering
     * both FAR and NEAR) breaks this at the push's scale. */
    double worst = 0.0;
    for (int i = 0; i < 3; i++) {
        K26V3 sum = { a_far1[i].x + a_near1[i].x,
                      a_far1[i].y + a_near1[i].y,
                      a_far1[i].z + a_near1[i].z };
        double mag = sqrt(a_full[i].x * a_full[i].x
                        + a_full[i].y * a_full[i].y
                        + a_full[i].z * a_full[i].z);
        double resid = v3_dist_(sum, a_full[i]);
        double rel = (mag > 0.0) ? resid / mag : resid;
        if (rel > worst) worst = rel;
    }
    fprintf(stderr, "gate3 perturbed field identity: "
            "max |a_far+a_near-a_full|/|a_full| = %.3e\n", worst);
    CHECK(worst <= 1.0e-14, "perturbed field identity residual %.3e",
          worst);

    /* The NEAR field is bit-identical with or without the
     * perturbation registered; FAR gained exactly the push. */
    CHECK(memcmp(a_near1, a_near0, sizeof a_near0) == 0,
          "NEAR field changed when a perturbation was registered");
    double dx = a_far1[1].x - a_far0[1].x;
    double dy = a_far1[1].y - a_far0[1].y;
    fprintf(stderr, "gate3 FAR delta on body 1: (%.17g, %.17g), "
            "push (1e-05, 2e-05)\n", dx, dy);
    CHECK(fabs(dx - 1.0e-5) <= 1.0e-11 && fabs(dy - 2.0e-5) <= 1.0e-11,
          "FAR pass does not carry the push once: delta (%.3e, %.3e)",
          dx, dy);

    k26astro_world_destroy(w);
}

/* ---- Gate 4: split lands on the unsplit trajectory -------------- */
static double epoch_s_(K26AstroWorld *w)
{
    K26AstroEpoch e;
    k26astro_world_now(w, &e);
    return (double)e.days_since_J2000 * 86400.0 + e.seconds_of_day;
}

static void gate4_near_identity_(K26AstroIntegrator base, const char *name,
                                 double pos_bound, double vel_bound)
{
    const double dt = 60.0;
    const int    steps = 10;

    /* World A: split active on `base`. World B: the same bodies
     * integrated whole by IAS15 at tight tolerance (the equivalent
     * single-integrator trajectory; IAS15 is never admitted to the
     * split). Same tolerance both sides so the inner drift and the
     * reference integrate equally accurately. */
    K26AstroWorld *a = pair_world_(1.0e-3, 0.361);
    K26AstroWorld *b = pair_world_(1.0e-3, 0.361);
    k26astro_grav_set_integrator(k26astro_world_grav(a), base);
    k26astro_grav_ias15_set_tol(k26astro_world_grav(a), 1.0e-11);
    k26astro_grav_set_integrator(k26astro_world_grav(b),
                                 K26ASTRO_INTEGRATOR_IAS15);
    k26astro_grav_ias15_set_tol(k26astro_world_grav(b), 1.0e-11);

    K26V3 b_pos0[3], b_vel0[3];
    for (int i = 0; i < 3; i++) {
        b_pos0[i] = k26astro_pos_to_m_approx(
                &k26astro_world_body_at(b, i)->pos);
        b_vel0[i] = k26astro_world_body_at(b, i)->vel;
    }

    int rc_a = 0, rc_b = 0;
    for (int s = 0; s < steps; s++) {
        int r = k26astro_world_step_exact(a, dt);
        if (r != 0 && rc_a == 0) rc_a = r;
        r = k26astro_world_step_exact(b, dt);
        if (r != 0 && rc_b == 0) rc_b = r;
    }
    CHECK(rc_a == 0, "%s split world rc=%d", name, rc_a);
    CHECK(rc_b == 0, "reference world rc=%d", rc_b);

    /* The split must actually have engaged, with both passes
     * carrying force. */
    CHECK(a->n_encounters == 1, "%s: %d encounters at the last "
          "substep, want 1", name, a->n_encounters);
    if (a->n_encounters == 1) {
        double K = a->encounters[0].k_weight;
        CHECK(K > 0.0 && K < 1.0, "%s: K=%.17g not in (0,1)", name, K);
    }
    if (base == K26ASTRO_INTEGRATOR_WH) {
        CHECK(a->mercurius_central_idx == 0,
              "%s: central idx %d", name, a->mercurius_central_idx);
    }

    /* One substep bills one dt, on both worlds. */
    double ea = epoch_s_(a), eb = epoch_s_(b);
    CHECK(fabs(ea - eb) <= 1.0e-6,
          "%s: epochs differ, %.9f vs %.9f s", name, ea, eb);

    double scale_pos = 0.0, scale_vel = 0.0;
    for (int i = 0; i < 3; i++) {
        K26V3 pb = k26astro_pos_to_m_approx(
                &k26astro_world_body_at(b, i)->pos);
        double dp = v3_dist_(pb, b_pos0[i]);
        double dv = v3_dist_(k26astro_world_body_at(b, i)->vel, b_vel0[i]);
        if (dp > scale_pos) scale_pos = dp;
        if (dv > scale_vel) scale_vel = dv;
    }
    double err_pos = 0.0, err_vel = 0.0;
    for (int i = 0; i < 3; i++) {
        K26V3 pa = k26astro_pos_to_m_approx(
                &k26astro_world_body_at(a, i)->pos);
        K26V3 pb = k26astro_pos_to_m_approx(
                &k26astro_world_body_at(b, i)->pos);
        double dp = v3_dist_(pa, pb);
        double dv = v3_dist_(k26astro_world_body_at(a, i)->vel,
                             k26astro_world_body_at(b, i)->vel);
        if (dp > err_pos) err_pos = dp;
        if (dv > err_vel) err_vel = dv;
    }
    double m_pos = err_pos / scale_pos;
    double m_vel = err_vel / scale_vel;
    fprintf(stderr, "gate4 %s: pos metric %.3e (err %.3e m over "
            "scale %.3e m), vel metric %.3e (err %.3e over %.3e m/s)\n",
            name, m_pos, err_pos, scale_pos, m_vel, err_vel, scale_vel);
    CHECK(m_pos <= pos_bound, "%s position metric %.3e over bound %.1e",
          name, m_pos, pos_bound);
    CHECK(m_vel <= vel_bound, "%s velocity metric %.3e over bound %.1e",
          name, m_vel, vel_bound);

    k26astro_world_destroy(a);
    k26astro_world_destroy(b);
}

/* ---- Gate 5: split and unsplit commit the same vehicle mass ----- */

/* Constant mass flow, added once per perturbation evaluation, the
 * same channel a propulsion callback uses. */
static void dotm_pert_(const K26AstroGravState *s,
                       const K26AstroGravView *v, K26V3 *accel, void *ctx)
{
    (void)s; (void)v; (void)accel;
    k26astro_vehicle_mass_accum_add((K26AstroVehicle *)ctx, -2.5);
}

/* One 60 s substep on `base` with a 1000 kg craft losing mass at a
 * constant 2.5 kg/s per evaluation; returns the committed mass
 * delta. split_on = 0 narrows the transition window so the same
 * bodies take the base's unsplit step. */
static double committed_mass_delta_(K26AstroIntegrator base, int split_on,
                                    const char *name)
{
    K26AstroWorld *w = pair_world_(1.0e-3, 0.361);
    /* A dynamically negligible craft carries the vehicle; its pairs
     * sit far outside the transition window (y > 10), so the p1-p2
     * encounter alone drives the split. */
    k26astro_world_add_body(w, body_("craft", 1.0e3, 6.674e-8,
                                     3.0 * AU, sqrt(GM_SUN / (3.0 * AU)),
                                     0));
    if (!split_on) k26astro_world_set_mercurius_factors(w, 0.05, 0.1);
    k26astro_grav_set_integrator(k26astro_world_grav(w), base);
    k26astro_grav_ias15_set_tol(k26astro_world_grav(w), 1.0e-11);

    K26AstroVehicle *v = k26astro_vehicle_new();
    k26astro_vehicle_bind_body(v, k26astro_world_body_at(w, 3));
    k26astro_vehicle_set_dry_mass(v, 1000.0);
    int rc_reg = k26astro_world_register_vehicle(w, v);
    int rc_pert = k26astro_grav_register_perturb(k26astro_world_grav(w),
                                                 dotm_pert_, v);
    CHECK(rc_reg == 0 && rc_pert == 0,
          "%s: vehicle/perturb registration rc=%d/%d",
          name, rc_reg, rc_pert);

    int rc = k26astro_world_step_exact(w, 60.0);
    CHECK(rc == 0, "%s: step rc=%d", name, rc);
    CHECK(w->n_encounters == (split_on ? 1 : 0),
          "%s: %d encounters, want %d", name, w->n_encounters,
          split_on ? 1 : 0);

    double dm = k26astro_vehicle_mass_now(v) - 1000.0;
    k26astro_world_destroy(w);
    k26astro_vehicle_destroy(v);
    return dm;
}

static void gate5_mass_commit_(void)
{
    struct { K26AstroIntegrator base; const char *name; } cases[2] = {
        { K26ASTRO_INTEGRATOR_VERLET, "verlet-base" },
        { K26ASTRO_INTEGRATOR_WH,     "wh-base" },
    };
    for (int c = 0; c < 2; c++) {
        double dm_split   = committed_mass_delta_(cases[c].base, 1,
                                                  cases[c].name);
        double dm_unsplit = committed_mass_delta_(cases[c].base, 0,
                                                  cases[c].name);
        fprintf(stderr, "gate5 %s: committed mass split %.17g kg, "
                "unsplit %.17g kg\n", cases[c].name, dm_split, dm_unsplit);
        CHECK(dm_split < 0.0 && dm_unsplit < 0.0,
              "%s: no mass was committed", cases[c].name);
        CHECK(fabs(dm_split - dm_unsplit) <= 1.0e-12 * fabs(dm_unsplit),
              "%s: split commits %.17g kg, unsplit %.17g kg",
              cases[c].name, dm_split, dm_unsplit);
    }
}

/* ---- Gate 6: WH split readmission declines off-primary centrals - */
static void gate6_readmission_(void)
{
    /* Largest mass at body 1. The craft-big pair (bodies 0 and 2)
     * is a genuine deep encounter (y near 0.7), so the detector
     * reports it; the WH admission must still decline (the WH
     * drift's Kepler primary is hard-wired to body 0), and every
     * step must be bit-identical to the unsplit WH step of the same
     * configuration (world B, whose narrowed window detects
     * nothing). */
    K26AstroWorld *wa = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                              K26ASTRO_COORDS_SECTOR_GRID);
    K26AstroWorld *wb = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                              K26ASTRO_COORDS_SECTOR_GRID);
    K26AstroWorld *ws[2] = { wa, wb };
    for (int k = 0; k < 2; k++) {
        k26astro_world_add_body(ws[k], body_("craft", 1.0e3, 1.0e-6,
                                             1.05 * AU,
                                             sqrt(GM_SUN / (1.05 * AU)), 1));
        k26astro_world_add_body(ws[k], body_("sun", M_SUN, GM_SUN,
                                             0.0, 0.0, -1));
        k26astro_world_add_body(ws[k], body_("big", 1.0e-3 * M_SUN,
                                             1.0e-3 * GM_SUN,
                                             AU, sqrt(GM_SUN / AU), 1));
        k26astro_grav_set_integrator(k26astro_world_grav(ws[k]),
                                     K26ASTRO_INTEGRATOR_WH);
    }
    k26astro_world_set_mercurius_factors(wb, 0.05, 0.1);

    int rc_a = 0, rc_b = 0;
    for (int s = 0; s < 5; s++) {
        int r = k26astro_world_step_exact(wa, 60.0);
        if (r != 0 && rc_a == 0) rc_a = r;
        r = k26astro_world_step_exact(wb, 60.0);
        if (r != 0 && rc_b == 0) rc_b = r;
    }
    CHECK(rc_a == 0 && rc_b == 0, "gate6 rc=%d/%d", rc_a, rc_b);
    CHECK(wa->n_encounters == 1,
          "gate6: %d encounters, want 1 (the pin needs a detected "
          "encounter)", wa->n_encounters);
    CHECK(wa->mercurius_central_idx == 1,
          "gate6: central idx %d, want 1", wa->mercurius_central_idx);
    CHECK(wb->n_encounters == 0,
          "gate6 reference: %d encounters, want 0", wb->n_encounters);

    int same = 1;
    for (int i = 0; i < 3; i++) {
        K26AstroBody *ba = k26astro_world_body_at(wa, i);
        K26AstroBody *bb = k26astro_world_body_at(wb, i);
        if (memcmp(&ba->vel, &bb->vel, sizeof(K26V3)) != 0) same = 0;
        K26V3 pa = k26astro_pos_to_m_approx(&ba->pos);
        K26V3 pb = k26astro_pos_to_m_approx(&bb->pos);
        if (memcmp(&pa, &pb, sizeof(K26V3)) != 0) same = 0;
    }
    fprintf(stderr, "gate6 off-primary WH vs unsplit WH: %s "
            "(encounters %d, central idx %d)\n",
            same ? "bit-identical" : "DIFFER",
            wa->n_encounters, wa->mercurius_central_idx);
    CHECK(same, "off-primary WH world took a split step");

    double ea = epoch_s_(wa), eb = epoch_s_(wb);
    CHECK(ea == eb, "gate6 epochs differ: %.9f vs %.9f s", ea, eb);

    k26astro_world_destroy(wa);
    k26astro_world_destroy(wb);
}

int main(void)
{
    gate1_field_identity_();
    gate2_detector_honesty_();
    gate3_perturbation_split_();
    /* Per-base bounds per the file header: both bases measure
     * pos 8.13e-15 / vel 3.09e-12 here and MUST measure alike (the
     * composition is base-independent); roughly 40x headroom over
     * the measurement. */
    gate4_near_identity_(K26ASTRO_INTEGRATOR_VERLET, "verlet-base",
                         3.0e-13, 1.2e-10);
    gate4_near_identity_(K26ASTRO_INTEGRATOR_WH, "wh-base",
                         3.0e-13, 1.2e-10);
    gate5_mass_commit_();
    gate6_readmission_();
    if (failures_ != 0) {
        fprintf(stderr, "test_mercurius_force_arithmetic: %d failure(s)\n",
                failures_);
        return 1;
    }
    printf("test_mercurius_force_arithmetic: OK\n");
    return 0;
}
