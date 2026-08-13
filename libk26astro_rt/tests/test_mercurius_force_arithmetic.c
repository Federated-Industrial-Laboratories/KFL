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
 *     shapes (Verlet: central pairs stay in FAR; WH: central pairs
 *     handed to the drift via central_plus1).
 *
 *   Gate 2 (detector honesty): the same pair of masses is detected
 *     or not detected by closeness, not by mass ratio alone: a
 *     distant pair does not split, a close pair does, a deep pair
 *     carries K = 1, and the unbound fallback follows its stated
 *     rule (current central-relative distance).
 *
 *   Gate 3 (WH kick weight consultation): the Wisdom-Holman
 *     interaction kick honours an active split context: a K = 0
 *     entry leaves the step bit-identical to an uncontexted step,
 *     and a K = 1 entry removes exactly the pair's kick.
 *
 *   Gate 4 (split-step near-identity): for each admitted base
 *     (Verlet; WH with the largest mass at body 0), a genuinely
 *     splitting trajectory lands on the unsplit high-accuracy IAS15
 *     trajectory of the same configuration within stated bounds,
 *     and bills the same epoch. This is the gate that goes red on
 *     force double-counting (a pass applying full pair force under
 *     an active context), on zero-weighting (a pass dropping its
 *     portion), and on kinetic double-drift (two passes each
 *     advancing positions).
 *
 * Bounds: the near-identity bound (1e-4) sits far above the honest
 * tree's measured metrics (1e-15 to 1e-11 at this configuration)
 * and well below the metrics measured under deliberate mutations of
 * this tree: force double-counting and zero-weighting land between
 * 1e-3 and 1, and the two-full-integrator composition this gate
 * retired shows a position metric near 1 (each pass drifted every
 * position by its velocity, doubling the kinetic advance).
 *
 * Reference: Rein, Hernandez, Tamayo et al. (2019), MNRAS
 * 485(4):5490-5497, "Hybrid Symplectic Integrators for Planetary
 * Dynamics." */
#include "k26astro_rt/world.h"
#include "k26astro_grav/grav.h"
#include "k26astro_grav/forces.h"
#include "k26astro_grav/ias15.h"
#include "k26astro_body/body.h"
#include "k26astro_core/pos.h"

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
        int central_plus1 = shape;   /* 0 = Verlet shape, 1 = WH shape */
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
}

/* ---- Gate 3: the WH kick honours the split context -------------- */
static void gate3_wh_kick_weights_(void)
{
    K26AstroBody set_a[3], set_b[3], set_c[3];
    for (int k = 0; k < 3; k++) {
        set_a[0] = body_("sun", M_SUN, GM_SUN, 0.0, 0.0, -1);
        set_a[1] = body_("p1", 1.0e-3 * M_SUN, 1.0e-3 * GM_SUN,
                         AU, sqrt(GM_SUN / AU), 0);
        set_a[2] = body_("p2", 1.0e-3 * M_SUN, 1.0e-3 * GM_SUN,
                         1.361 * AU, sqrt(GM_SUN / (1.361 * AU)), 0);
        if (k == 0) memcpy(set_b, set_a, sizeof set_a);
        if (k == 1) memcpy(set_c, set_a, sizeof set_a);
    }

    K26AstroGravState sz, sy, sx;
    k26astro_grav_state_init(&sz, set_a, 3);
    k26astro_grav_state_init(&sy, set_b, 3);
    k26astro_grav_state_init(&sx, set_c, 3);
    k26astro_grav_set_integrator(&sz, K26ASTRO_INTEGRATOR_WH);
    k26astro_grav_set_integrator(&sy, K26ASTRO_INTEGRATOR_WH);
    k26astro_grav_set_integrator(&sx, K26ASTRO_INTEGRATOR_WH);

    double dt = 60.0;
    K26AstroPairWeight pw0 = { 1, 2, 0.0 };
    K26AstroPairWeight pw1 = { 1, 2, 1.0 };
    K26AstroMercuriusContext ctx0 = {
        .mode = K26ASTRO_MERCURIUS_FAR, .pair_weights = &pw0,
        .n_pair_weights = 1, .central_plus1 = 1 };
    K26AstroMercuriusContext ctx1 = {
        .mode = K26ASTRO_MERCURIUS_FAR, .pair_weights = &pw1,
        .n_pair_weights = 1, .central_plus1 = 1 };

    int rcz = k26astro_grav_step(&sz, dt);
    sy.mercurius = &ctx0;
    int rcy = k26astro_grav_step(&sy, dt);
    sy.mercurius = NULL;
    sx.mercurius = &ctx1;
    int rcx = k26astro_grav_step(&sx, dt);
    sx.mercurius = NULL;
    CHECK(rcz == 0 && rcy == 0 && rcx == 0,
          "WH steps rc=%d/%d/%d", rcz, rcy, rcx);

    /* K = 0 in FAR mode is weight 1: bit-identical to no context. */
    int same = 1;
    for (int i = 0; i < 3; i++) {
        if (memcmp(&set_a[i].vel, &set_b[i].vel, sizeof(K26V3)) != 0) same = 0;
        K26V3 pa = k26astro_pos_to_m_approx(&set_a[i].pos);
        K26V3 pb = k26astro_pos_to_m_approx(&set_b[i].pos);
        if (memcmp(&pa, &pb, sizeof(K26V3)) != 0) same = 0;
    }
    fprintf(stderr, "gate3 K=0 context vs no context: %s\n",
            same ? "bit-identical" : "DIFFER");
    CHECK(same, "K=0 far context changed a WH step");

    /* K = 1 in FAR mode removes exactly the pair's kick: p1's
     * velocity difference across the two steps is the pair kick
     * dt * G m2 / d^2 to leading order. */
    double dv = v3_dist_(set_a[1].vel, set_c[1].vel);
    double d12 = 0.361 * AU;
    double kick = dt * (1.0e-3 * GM_SUN) / (d12 * d12);
    fprintf(stderr, "gate3 K=1 pair kick removed: |dv|=%.6e m/s, "
            "analytic pair kick %.6e m/s\n", dv, kick);
    CHECK(dv > 0.0, "K=1 far context did not change the step");
    CHECK(fabs(dv - kick) <= 0.1 * kick,
          "removed kick %.3e not within 10%% of analytic %.3e", dv, kick);

    k26astro_grav_state_destroy(&sz);
    k26astro_grav_state_destroy(&sy);
    k26astro_grav_state_destroy(&sx);
}

/* ---- Gate 4: split lands on the unsplit trajectory -------------- */
static double epoch_s_(K26AstroWorld *w)
{
    K26AstroEpoch e;
    k26astro_world_now(w, &e);
    return (double)e.days_since_J2000 * 86400.0 + e.seconds_of_day;
}

static void gate4_near_identity_(K26AstroIntegrator base, const char *name)
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
    CHECK(m_pos <= 1.0e-4, "%s position metric %.3e", name, m_pos);
    CHECK(m_vel <= 1.0e-4, "%s velocity metric %.3e", name, m_vel);

    k26astro_world_destroy(a);
    k26astro_world_destroy(b);
}

int main(void)
{
    gate1_field_identity_();
    gate2_detector_honesty_();
    gate3_wh_kick_weights_();
    gate4_near_identity_(K26ASTRO_INTEGRATOR_VERLET, "verlet-base");
    gate4_near_identity_(K26ASTRO_INTEGRATOR_WH, "wh-base");
    if (failures_ != 0) {
        fprintf(stderr, "test_mercurius_force_arithmetic: %d failure(s)\n",
                failures_);
        return 1;
    }
    printf("test_mercurius_force_arithmetic: OK\n");
    return 0;
}
