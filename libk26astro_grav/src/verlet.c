/* verlet.c — DKD (Drift-Kick-Drift) Velocity Verlet symplectic integrator.
 *
 * Step:
 *   x(t+dt/2) = x(t)      + (dt/2)·v(t)
 *   v(t+dt)   = v(t)      + dt·a(x(t+dt/2))
 *   x(t+dt)   = x(t+dt/2) + (dt/2)·v(t+dt)
 *
 * Second-order accurate, symplectic, energy bounded. Per-step cost:
 * one acceleration evaluation. */
#include "k26astro_grav/verlet.h"
#include "k26astro_grav/forces.h"
#include "k26astro_grav/grav.h"
#include "k26astro_core/pos.h"
#include "k26astro_core/epoch.h"

int k26astro_grav_step_verlet(K26AstroGravState *state, double dt)
{
    if (!state) return K26ASTRO_E_NULL;
    int n = state->n_bodies;
    K26AstroBody *b = state->bodies;

    /* Kick scratch is preallocated by k26astro_grav_state_reserve;
     * the reserve call here is the off-reserve fallback and
     * allocates nothing when capacity is sufficient. */
    if (state->scratch_cap < n) {
        int rc = k26astro_grav_state_reserve(state);
        if (rc != K26ASTRO_E_OK) return rc;
    }

    /* Half-drift: x → x + (dt/2)·v */
    for (int i = 0; i < n; i++) {
        K26V3 d = { 0.5 * dt * b[i].vel.x,
                    0.5 * dt * b[i].vel.y,
                    0.5 * dt * b[i].vel.z };
        k26astro_pos_add(&b[i].pos, d);
    }

    /* Kick: compute accel at half-drifted position, advance v by dt.
     * accel_total zeroes the buffer before writing, so reusing the
     * scratch is bit-identical to a fresh zeroed allocation. */
    K26V3 *accel = state->scratch_accel;
    k26astro_grav_accel_total(state, accel);
    for (int i = 0; i < n; i++) {
        b[i].vel.x += dt * accel[i].x;
        b[i].vel.y += dt * accel[i].y;
        b[i].vel.z += dt * accel[i].z;
    }

    /* Half-drift: x → x + (dt/2)·v_new */
    for (int i = 0; i < n; i++) {
        K26V3 d = { 0.5 * dt * b[i].vel.x,
                    0.5 * dt * b[i].vel.y,
                    0.5 * dt * b[i].vel.z };
        k26astro_pos_add(&b[i].pos, d);
    }

    k26astro_epoch_add_seconds(&state->t, dt);
    state->dt_last = dt;
    return K26ASTRO_E_OK;
}
