/* grav_state.c — K26AstroGravState lifecycle. */
#include "k26astro_grav/grav.h"
#include "k26astro_grav/forces.h"
#include "k26astro_grav/perturb.h"
#include "k26astro_grav/wisdom_holman.h"
#include "k26astro_grav/ias15.h"
#include "k26astro_grav/verlet.h"
#include "k26compute.h"
#include "grav_step_internal.h"
#include "ias15_internal.h"

#include <stdlib.h>
#include <string.h>

/* Event-list struct definition shared with event_registry.c via this
 * translation unit (the destroy path needs the full layout to free
 * the inner buffer). */
struct K26AstroEventList {
    K26AstroGravEvent *events;
    int                count;
    int                capacity;
};

void k26astro_grav_fpu_pin(void);

/* Defined here so K26AstroIAS15Carry / K26AstroWHCarry's struct
 * definitions in the per-integrator headers are sufficient for the
 * destroy path. */

int k26astro_grav_state_init(K26AstroGravState *state,
                              K26AstroBody *bodies,
                              int n_bodies)
{
    if (!state || !bodies || n_bodies < 1) return K26ASTRO_E_NULL;

    state->bodies      = bodies;
    state->n_bodies    = n_bodies;
    state->integrator  = K26ASTRO_INTEGRATOR_WH;
    state->softening   = 0.0;
    state->t           = k26astro_epoch_j2000_tt();
    state->dt_last     = 0.0;
    state->use_j2      = 0;
    state->use_srp     = 0;
    state->use_gr_ppn1 = 0;
    state->perturbs    = NULL;
    state->events             = NULL;
    state->event_snapshot     = NULL;
    state->event_snapshot_cap = 0;
    /* Default event bisection tolerance: 1e-6 s. Hex-literal IEEE-754
     * for cross-platform determinism. 1e-6 ≈ 0x3EB0C6F7A0B5ED8D. */
    {
        union { double d; uint64_t u; } tol = { .u = 0x3EB0C6F7A0B5ED8DULL };
        state->event_tol_s = tol.d;
    }
    state->ias15_carry = NULL;
    state->wh_carry    = NULL;

    /* IAS15 controller defaults (Rein-Spiegel 2015 §4). epsilon =
     * 1.0e-9 stored as a hex literal for cross-platform determinism
     * (musl/glibc strtod round-trip is 1 ULP but the hex form is
     * exact). 1e-9 ≈ 0x3E112E0BE826D695. */
    {
        union { double d; uint64_t u; } eps = { .u = 0x3E112E0BE826D695ULL };
        state->ias15_tol = eps.d;
    }
    state->ias15_dt_last        = 0.0;
    state->ias15_dt_hint        = 0.0;
    state->ias15_rejected_steps = 0;
    state->ias15_last_pc_iterations  = 0;
    state->ias15_last_eps_b_achieved = 0.0;
    state->ias15_snapshot       = NULL;
    state->ias15_snapshot_cap   = 0;
    state->ias15_wall_budget_s  = 0.0;
    state->mercurius            = NULL;
    state->scratch_accel        = NULL;
    state->scratch_bodies       = NULL;
    state->scratch_y            = NULL;
    state->scratch_ws           = NULL;
    state->scratch_cap          = 0;

    /* Pin FPU rounding + denormal mode for cross-platform determinism. */
    k26astro_grav_fpu_pin();

    /* Preallocate every integrator's carry and scratch so the first
     * step after init touches preallocated memory only. */
    int rc = k26astro_grav_state_reserve(state);
    if (rc != K26ASTRO_E_OK) {
        k26astro_grav_state_destroy(state);
        return rc;
    }

    return K26ASTRO_E_OK;
}

/* Preallocate integrator carries, snapshots, and step scratch for
 * the current n_bodies. See grav.h for the per-body sizing rule.
 * Reached only from non-step times on the supported path (state
 * init, the rt world's body add); the step functions call it as an
 * off-reserve capacity fallback. */
int k26astro_grav_state_reserve(K26AstroGravState *state)
{
    if (!state) return K26ASTRO_E_NULL;
    int n = state->n_bodies;
    if (n < 1) return K26ASTRO_E_OK;

    /* WH carry (p_bary, r_helio, a_int). */
    int rc = k26astro_grav_wh_carry_ensure(state);
    if (rc != 0) return rc;

    /* IAS15 carry: b/e/g coefficient rows plus the predictor's
     * per-call scratch. Grown by allocating the replacement before
     * releasing the old carry, so a failed grow leaves the previous
     * carry sized as before (the contract in grav.h). The
     * replacement starts as a fresh carry: a capacity change resets
     * the predictor seed, exactly as the old release-and-realloc
     * did (a body-count change invalidates the seed anyway). The
     * geometric target makes a body-by-body build-up allocate a new
     * carry a logarithmic number of times instead of every add. */
    if (!state->ias15_carry || state->ias15_carry->capacity < n) {
        int want = k26_grav_grow_target_(
            state->ias15_carry ? state->ias15_carry->capacity : 0, n);
        K26AstroIAS15Carry *fresh = NULL;
        rc = k26_ias15_carry_alloc(&fresh, want);
        if (rc != 0) return rc;
        if (state->ias15_carry) k26_ias15_carry_release(state->ias15_carry);
        state->ias15_carry = fresh;
    }

    /* IAS15 reject-rollback snapshot. */
    if (state->ias15_snapshot_cap < n) {
        int want = k26_grav_grow_target_(state->ias15_snapshot_cap, n);
        K26AstroBody *fresh = realloc(state->ias15_snapshot,
                                       (size_t)want * sizeof(K26AstroBody));
        if (!fresh) return K26ASTRO_E_ALLOC;
        state->ias15_snapshot     = fresh;
        state->ias15_snapshot_cap = want;
    }

    /* Event-time root-finding snapshot. */
    if (state->event_snapshot_cap < n) {
        int want = k26_grav_grow_target_(state->event_snapshot_cap, n);
        K26AstroBody *fresh = realloc(state->event_snapshot,
                                       (size_t)want * sizeof(K26AstroBody));
        if (!fresh) return K26ASTRO_E_ALLOC;
        state->event_snapshot     = fresh;
        state->event_snapshot_cap = want;
    }

    /* Verlet / RK step scratch (sizing rule documented in grav.h). */
    if (state->scratch_cap < n) {
        int want = k26_grav_grow_target_(state->scratch_cap, n);
        K26V3        *accel  = realloc(state->scratch_accel,
                                        (size_t)want * sizeof(K26V3));
        if (!accel) return K26ASTRO_E_ALLOC;
        state->scratch_accel = accel;
        K26AstroBody *bodies = realloc(state->scratch_bodies,
                                        (size_t)want * sizeof(K26AstroBody));
        if (!bodies) return K26ASTRO_E_ALLOC;
        state->scratch_bodies = bodies;
        double       *y      = realloc(state->scratch_y,
                                        (size_t)(6 * want) * sizeof(double));
        if (!y) return K26ASTRO_E_ALLOC;
        state->scratch_y = y;
        double       *ws     = realloc(state->scratch_ws,
            K26C_ODE_RK45_WS((size_t)(6 * want)) * sizeof(double));
        if (!ws) return K26ASTRO_E_ALLOC;
        state->scratch_ws  = ws;
        state->scratch_cap = want;
    }

    return K26ASTRO_E_OK;
}

void k26astro_grav_state_destroy(K26AstroGravState *state)
{
    if (!state) return;

    if (state->perturbs) {
        free(state->perturbs->fns);
        free(state->perturbs->ctxs);
        free(state->perturbs);
        state->perturbs = NULL;
    }

    if (state->events) {
        free(state->events->events);
        free(state->events);
        state->events = NULL;
    }

    if (state->event_snapshot) {
        free(state->event_snapshot);
        state->event_snapshot     = NULL;
        state->event_snapshot_cap = 0;
    }

    if (state->ias15_carry) {
        k26_ias15_carry_release(state->ias15_carry);
        state->ias15_carry = NULL;
    }

    if (state->wh_carry) {
        free(state->wh_carry->p_bary);
        free(state->wh_carry->r_helio);
        free(state->wh_carry->a_int);
        free(state->wh_carry);
        state->wh_carry = NULL;
    }

    if (state->ias15_snapshot) {
        free(state->ias15_snapshot);
        state->ias15_snapshot     = NULL;
        state->ias15_snapshot_cap = 0;
    }

    free(state->scratch_accel);
    free(state->scratch_bodies);
    free(state->scratch_y);
    free(state->scratch_ws);
    state->scratch_accel  = NULL;
    state->scratch_bodies = NULL;
    state->scratch_y      = NULL;
    state->scratch_ws     = NULL;
    state->scratch_cap    = 0;
}

int k26astro_grav_set_integrator(K26AstroGravState *state,
                                  K26AstroIntegrator which)
{
    if (!state) return K26ASTRO_E_NULL;
    if (which < K26ASTRO_INTEGRATOR_WH
     || which > K26ASTRO_INTEGRATOR_MERCURIUS) return K26ASTRO_E_BAD_INPUT;
    state->integrator = which;
    return K26ASTRO_E_OK;
}

int k26astro_grav_set_softening(K26AstroGravState *state,
                                 double softening_m)
{
    if (!state) return K26ASTRO_E_NULL;
    if (softening_m < 0.0) return K26ASTRO_E_BAD_INPUT;
    state->softening = softening_m;
    return K26ASTRO_E_OK;
}

int k26astro_grav_enable_j2(K26AstroGravState *state, int enable)
{
    if (!state) return K26ASTRO_E_NULL;
    state->use_j2 = enable ? 1 : 0;
    return K26ASTRO_E_OK;
}

int k26astro_grav_enable_srp(K26AstroGravState *state, int enable)
{
    if (!state) return K26ASTRO_E_NULL;
    state->use_srp = enable ? 1 : 0;
    return K26ASTRO_E_OK;
}

int k26astro_grav_enable_gr_ppn1(K26AstroGravState *state, int enable)
{
    if (!state) return K26ASTRO_E_NULL;
    state->use_gr_ppn1 = enable ? 1 : 0;
    return K26ASTRO_E_OK;
}

/* ---- IAS15 substep diagnostics --------------------------------- */

int k26astro_grav_last_pc_iterations(const K26AstroGravState *state)
{
    if (!state) return 0;
    return state->ias15_last_pc_iterations;
}

double k26astro_grav_last_dt_taken(const K26AstroGravState *state)
{
    if (!state) return 0.0;
    return state->ias15_dt_last;
}

double k26astro_grav_last_eps_b_achieved(const K26AstroGravState *state)
{
    if (!state) return 0.0;
    return state->ias15_last_eps_b_achieved;
}

uint32_t k26astro_grav_rejected_steps_total(const K26AstroGravState *state)
{
    if (!state) return 0u;
    return state->ias15_rejected_steps;
}

/* ---- Step dispatch ---------------------------------------------- */

/* Inner dispatch: identical to the original k26astro_grav_step body
 * before the event-wrapper split. Used directly when no events are
 * registered, and called recursively by k26astro_grav_step_with_events
 * for each sub-step. */
int k26astro_grav_step_inner_dispatch(K26AstroGravState *state, double dt)
{
    if (!state) return K26ASTRO_E_NULL;
    state->dt_last = dt;
    switch (state->integrator) {
    case K26ASTRO_INTEGRATOR_WH:        return k26astro_grav_step_wh    (state, dt);
    case K26ASTRO_INTEGRATOR_IAS15:     return k26astro_grav_step_ias15 (state, dt);
    case K26ASTRO_INTEGRATOR_VERLET:    return k26astro_grav_step_verlet(state, dt);
    case K26ASTRO_INTEGRATOR_RK4:       /* fall through */
    case K26ASTRO_INTEGRATOR_RK45: {
        int k26astro_grav_step_rk(K26AstroGravState *, double);
        return k26astro_grav_step_rk(state, dt);
    }
    case K26ASTRO_INTEGRATOR_MERCURIUS:
        /* MERCURIUS handoff orchestration lives in libk26astro_rt.
         * Fall through to WH so the call doesn't crash; flag the
         * missing orchestration via K26ASTRO_E_NOT_WIRED. */
        k26astro_grav_step_wh(state, dt);
        return K26ASTRO_E_NOT_WIRED;
    }
    return K26ASTRO_E_BAD_INPUT;
}

int k26astro_grav_step(K26AstroGravState *state, double dt)
{
    if (!state) return K26ASTRO_E_NULL;
    /* Route through the event-time wrapper when the registry has
     * entries. Empty registry → direct dispatch (one NULL check
     * overhead beyond the original path). */
    if (state->events && state->events->count > 0) {
        return k26astro_grav_step_with_events(state, dt);
    }
    return k26astro_grav_step_inner_dispatch(state, dt);
}
