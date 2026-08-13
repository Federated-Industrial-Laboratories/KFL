/* rk_wrapper.c — RK4 + RK45 (Dormand-Prince) wrappers over libk26compute.
 *
 * The K26AstroGravState ↔ K26CVector adapter: flattens N-body state
 * into a 6N-double vector (x, y, z, vx, vy, vz × n_bodies), runs
 * libk26compute's RK driver, copies back. Used for non-conservative
 * force integration where the symplectic property of WH / Verlet
 * isn't relevant.
 *
 * The position storage uses K26AstroPos's sector-grid representation
 * during the K26AstroBody phase; the RK state vector flattens it to
 * a doubles offset relative to each body's sector origin. After the
 * RK step completes, k26astro_pos_normalise rebases each body's
 * sector + offset.
 *
 * Determinism: libk26compute's RK4/RK45 are deterministic; we now
 * patched its Makefile to add -ffp-contract=off so cross-platform
 * bit-identity holds in portable mode. */
#include "k26astro_grav/grav.h"
#include "k26astro_grav/forces.h"
#include "k26astro_core/pos.h"
#include "k26astro_core/epoch.h"
#include "k26compute.h"

#include <string.h>

/* RHS callback: derivative of y = [x, y, z, vx, vy, vz, ...] is
 * dy/dt = [vx, vy, vz, ax, ay, az, ...] where (ax, ay, az) come from
 * a full accel_total evaluation at the current y. */
static int rhs_(double t, const K26CVector *y, K26CVector *dydt, void *user)
{
    (void)t;
    K26AstroGravState *state = (K26AstroGravState *)user;
    int n = state->n_bodies;
    if (!y || !dydt) return 1;
    if (y->n   != (size_t)(6 * n)) return 1;
    if (dydt->n != (size_t)(6 * n)) return 1;

    /* Unpack y into state->bodies (positions as offsets from sector
     * origin; velocities directly). We use a temporary K26AstroBody
     * shallow-copy so we don't disturb the user's state. The copy
     * lives in the state's preallocated step scratch (sized by
     * k26astro_grav_state_reserve; k26astro_grav_step_rk checks the
     * capacity before the driver ever calls back into here). */
    K26AstroBody *scratch = state->scratch_bodies;
    memcpy(scratch, state->bodies, sizeof(K26AstroBody) * (size_t)n);

    for (int i = 0; i < n; i++) {
        double xi = y->data[6*i + 0];
        double yi = y->data[6*i + 1];
        double zi = y->data[6*i + 2];
        scratch[i].pos = k26astro_pos_from_m(xi, yi, zi);
        scratch[i].vel.x = y->data[6*i + 3];
        scratch[i].vel.y = y->data[6*i + 4];
        scratch[i].vel.z = y->data[6*i + 5];
    }

    K26AstroGravState shadow = *state;
    shadow.bodies = scratch;

    /* accel_total zeroes the buffer before writing, so reusing the
     * scratch is bit-identical to a fresh zeroed allocation. */
    K26V3 *accel = state->scratch_accel;
    k26astro_grav_accel_total(&shadow, accel);

    for (int i = 0; i < n; i++) {
        dydt->data[6*i + 0] = scratch[i].vel.x;
        dydt->data[6*i + 1] = scratch[i].vel.y;
        dydt->data[6*i + 2] = scratch[i].vel.z;
        dydt->data[6*i + 3] = accel[i].x;
        dydt->data[6*i + 4] = accel[i].y;
        dydt->data[6*i + 5] = accel[i].z;
    }

    return 0;
}

/* Pack/unpack helpers between K26AstroBody[] and K26CVector. */
static void pack_(const K26AstroBody *bodies, int n, K26CVector *y)
{
    for (int i = 0; i < n; i++) {
        K26V3 r = k26astro_pos_to_m_approx(&bodies[i].pos);
        y->data[6*i + 0] = r.x;
        y->data[6*i + 1] = r.y;
        y->data[6*i + 2] = r.z;
        y->data[6*i + 3] = bodies[i].vel.x;
        y->data[6*i + 4] = bodies[i].vel.y;
        y->data[6*i + 5] = bodies[i].vel.z;
    }
}

static void unpack_(K26AstroBody *bodies, int n, const K26CVector *y)
{
    for (int i = 0; i < n; i++) {
        bodies[i].pos = k26astro_pos_from_m(
            y->data[6*i + 0],
            y->data[6*i + 1],
            y->data[6*i + 2]);
        bodies[i].vel.x = y->data[6*i + 3];
        bodies[i].vel.y = y->data[6*i + 4];
        bodies[i].vel.z = y->data[6*i + 5];
    }
}

int k26astro_grav_step_rk(K26AstroGravState *state, double dt)
{
    if (!state) return K26ASTRO_E_NULL;
    int n = state->n_bodies;

    /* Zero-dimension step: nothing to integrate, but the epoch still
     * advances by dt. The pre-workspace path reached the same result
     * through the ODE driver (its calloc(0) state vector was accepted
     * and every per-component loop was empty); the workspace path
     * would instead hand the driver a NULL data pointer, so the case
     * is short-circuited here to keep the observable behaviour. */
    if (n < 1) {
        k26astro_epoch_add_seconds(&state->t, dt);
        state->dt_last = dt;
        return K26ASTRO_E_OK;
    }

    size_t dim = (size_t)(6 * n);

    /* State vector, RHS shadow, and stage workspace live in the
     * state's preallocated step scratch; the reserve call is the
     * off-reserve fallback and allocates nothing when capacity is
     * sufficient. pack_ writes every entry of y before the driver
     * reads it, so reuse is bit-identical to a fresh zeroed
     * allocation. */
    if (state->scratch_cap < n) {
        int rc = k26astro_grav_state_reserve(state);
        if (rc != K26ASTRO_E_OK) return rc;
    }

    K26CVector y;
    y.n = dim;
    y.data = state->scratch_y;

    pack_(state->bodies, n, &y);

    K26CStatus rc;
    if (state->integrator == K26ASTRO_INTEGRATOR_RK4) {
        rc = k26c_ode_rk4_ws(rhs_, state, 0.0, dt, /*n_steps*/ 1, &y,
                             state->scratch_ws, K26C_ODE_RK45_WS(dim));
    } else {
        /* RK45 (Dormand-Prince) with reasonable default tolerances. */
        rc = k26c_ode_rk45_ws(rhs_, state, 0.0, dt, /*rtol*/ 1e-9,
                              /*atol*/ 1e-12, &y,
                              state->scratch_ws, K26C_ODE_RK45_WS(dim));
    }
    if (rc != K26C_OK) return K26ASTRO_E_NO_CONVERGE;

    unpack_(state->bodies, n, &y);

    k26astro_epoch_add_seconds(&state->t, dt);
    state->dt_last = dt;
    return K26ASTRO_E_OK;
}
