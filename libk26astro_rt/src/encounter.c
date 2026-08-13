/* encounter.c — MERCURIUS close-encounter primitives.
 *
 * Implements the MERCURIUS quintic smoothstep K(y) over the
 * transition window [y_inner, y_outer] in Hill-radius units, plus
 * the per-pair Hill radius and the per-pair encounter session
 * tracker. The orchestration that uses these primitives to actually
 * split the force between the Verlet base and IAS15 lives in
 * orbit_step.c.
 *
 * Reference: Rein, Hernandez, Tamayo et al. (2019), MNRAS
 * 485(4):5490-5497, "Hybrid Symplectic Integrators for Planetary
 * Dynamics." */
#include "encounter_internal.h"

#include "k26astro_core/pos.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

double k26astro_mercurius_K(double y, double y_inner, double y_outer)
{
    if (!(y_outer > y_inner)) return 1.0;
    if (y <= y_inner) return 1.0;
    if (y >= y_outer) return 0.0;
    double x = (y - y_inner) / (y_outer - y_inner);
    /* Quintic smoothstep S(x) = 10 x^3 - 15 x^4 + 6 x^5.
     * K(y) = 1 - S(x) so K = 1 at y_inner (full IAS15) and
     * K = 0 at y_outer (full WH). C^2 at both endpoints. */
    double x2 = x * x;
    double x3 = x2 * x;
    double x4 = x3 * x;
    double x5 = x4 * x;
    double s  = 10.0 * x3 - 15.0 * x4 + 6.0 * x5;
    return 1.0 - s;
}

double k26astro_mercurius_hill_radius(const K26AstroBody *i,
                                       const K26AstroBody *j,
                                       double m_central)
{
    if (!i || !j) return 0.0;
    /* Semi-major axis approximated by the current separation (the
     * exact value would require fitting an osculating conic, which
     * is too expensive per-pair per-step). Known recorded defect:
     * with a_ij equal to the separation, the detector's ratio
     * y = d / r_hill is separation-independent (the distance
     * cancels), so this heuristic classifies pairs by mass ratio
     * alone and never measures closeness; a pair massive enough to
     * cross the threshold splits at every separation. The redesign
     * is scoped as its own follow-up item. */
    K26V3 r = k26astro_pos_sub(&i->pos, &j->pos);
    double a_ij = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    if (!(a_ij > 0.0))      return 0.0;
    if (!(m_central > 0.0)) return 0.0;
    double m_sum = i->mass + j->mass;
    if (!(m_sum > 0.0))     return 0.0;
    return a_ij * cbrt(m_sum / (3.0 * m_central));
}

/* Largest per-buffer entry count the int capacity fields and a
 * size_t byte count can both represent, across both session buffer
 * element types. */
static uint64_t pair_cap_bound_(void)
{
    uint64_t bound   = (uint64_t)INT_MAX;
    uint64_t enc_max = (uint64_t)(SIZE_MAX / sizeof(K26AstroEncounter));
    uint64_t pw_max  = (uint64_t)(SIZE_MAX / sizeof(K26AstroPairWeight));
    if (enc_max < bound) bound = enc_max;
    if (pw_max  < bound) bound = pw_max;
    return bound;
}

/* Grow BOTH MERCURIUS session buffers to at least `need` entries.
 * Growing them together enforces by construction the invariant the
 * pair-weight build relies on: any encounter slot the detect can
 * record has a matching pair-weight slot. The new capacity at least
 * doubles the current one so repeated single-body adds copy an
 * amortised-constant number of entries per add instead of a full
 * quadratic buffer each time. Returns 0, or -1 when `need` is not
 * representable or an allocation fails; on failure the pair-weight
 * buffer keeps its previous size (the encounter list may already
 * have grown, harmlessly: the caller records nothing on failure, so
 * n_encounters never exceeds the smaller capacity) and every
 * capacity field stays truthful. */
static int grow_pair_buffers_(K26AstroWorld *world, uint64_t need)
{
    if (need <= (uint64_t)world->cap_encounters
        && need <= (uint64_t)world->cap_pair_weights) return 0;
    uint64_t bound = pair_cap_bound_();
    if (need > bound) return -1;
    uint64_t cur = (uint64_t)(world->cap_encounters < world->cap_pair_weights
                              ? world->cap_encounters
                              : world->cap_pair_weights);
    uint64_t new_cap = cur * 2u;
    if (new_cap < need)  new_cap = need;
    if (new_cap > bound) new_cap = need;   /* need <= bound, checked above */
    if ((uint64_t)world->cap_encounters < new_cap) {
        K26AstroEncounter *p = (K26AstroEncounter *)realloc(
            world->encounters, (size_t)new_cap * sizeof(K26AstroEncounter));
        if (!p) return -1;
        world->encounters     = p;
        world->cap_encounters = (int)new_cap;
    }
    if ((uint64_t)world->cap_pair_weights < new_cap) {
        K26AstroPairWeight *w = (K26AstroPairWeight *)realloc(
            world->pair_weights, (size_t)new_cap * sizeof(K26AstroPairWeight));
        if (!w) return -1;
        world->pair_weights     = w;
        world->cap_pair_weights = (int)new_cap;
    }
    return 0;
}

int k26astro_rt_encounter_reserve(K26AstroWorld *world, int n_bodies)
{
    if (!world) return -1;
    if (n_bodies < 2) return 0;
    /* Worst case: every distinct pair in the transition region.
     * Computed in 64-bit: n*(n-1)/2 overflows int from n = 46342. A
     * pair count the capacity fields cannot represent refuses the
     * add (the caller maps -1 onto its OOM error). */
    uint64_t pairs = (uint64_t)n_bodies * ((uint64_t)n_bodies - 1u) / 2u;
    return grow_pair_buffers_(world, pairs);
}

int k26astro_mercurius_detect(K26AstroWorld *world)
{
    if (!world) return 0;
    int n = world->grav.n_bodies;
    if (n < 2) { world->n_encounters = 0; return 0; }

    /* Central body = the largest mass. In the solar system this is
     * the Sun; for moon-around-planet sub-systems the caller should
     * set up a hierarchical world (out of scope in v0.1). */
    double m_central   = 0.0;
    int    idx_central = -1;
    for (int k = 0; k < n; k++) {
        double m = world->grav.bodies[k].mass;
        if (m > m_central) { m_central = m; idx_central = k; }
    }
    if (!(m_central > 0.0)) { world->n_encounters = 0; return 0; }

    world->n_encounters = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            /* Pairs involving the central body are never encounter
             * pairs. What the exclusion means depends on the base
             * integrator:
             *
             * - Wisdom-Holman base (currently never admitted to
             *   the split; see the admission note in orbit_step.c):
             *   MERCURIUS K-weights the planet-planet interaction
             *   terms only (Rein et al. 2019, section 2, eqs 4-5).
             *   The central pull is the Kepler part the drift
             *   integrates exactly ONLY when the largest mass is
             *   body 0, the drift's hard-wired primary
             *   (wisdom_holman.c, mu0 = b[0].gm). When the largest
             *   mass sits elsewhere the drift still orbits body 0,
             *   the exclusion is an approximation, and the split's
             *   correctness there is an open question recorded with
             *   the detector follow-up item.
             *
             * - Verlet base: there is no Kepler part; every force
             *   lives in the pair sum. The exclusion is right for
             *   the opposite reason: it keeps the central force in
             *   the FAR pass at full weight. Including central
             *   pairs zero-weighted the central force there (K = 1
             *   near the primary), so the FAR pass integrated a
             *   straight-line drift with no central gravity at all.
             *   Excluding them corrected that, changing Verlet-base
             *   split trajectories; only WH-base configurations are
             *   byte-identical across the change.
             *
             * The mutual-Hill heuristic's degeneracy is structural
             * for ALL pairs, not specific to central ones: see the
             * recorded defect at k26astro_mercurius_hill_radius. */
            if (i == idx_central || j == idx_central) continue;
            const K26AstroBody *bi = &world->grav.bodies[i];
            const K26AstroBody *bj = &world->grav.bodies[j];
            double rh = k26astro_mercurius_hill_radius(bi, bj, m_central);
            if (!(rh > 0.0)) continue;
            K26V3 r = k26astro_pos_sub(&bi->pos, &bj->pos);
            double d = sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
            double y = d / rh;
            if (y >= world->mercurius_outer_factor) continue;
            /* In transition or fully near. Record with K weight. */
            double K = k26astro_mercurius_K(y,
                                              world->mercurius_hill_factor,
                                              world->mercurius_outer_factor);
            int slot = world->n_encounters;
            /* In-step fallback for worlds whose bodies were grown
             * without the body-add reserve; with the reserve it is a
             * no-op. Grows both session buffers together, keeping
             * the pair-weight invariant intact. */
            if (grow_pair_buffers_(world, (uint64_t)slot + 1u) != 0) {
                return slot;
            }
            world->encounters[slot] = (K26AstroEncounter){
                .i = i, .j = j,
                .y_last = y,
                .k_weight = K,
                .active = 1
            };
            world->n_encounters = slot + 1;
        }
    }
    return world->n_encounters;
}
