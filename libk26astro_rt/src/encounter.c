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

/* Length scale of `b` about `central` for the Hill-radius build.
 * Rule, stated: the smaller of the osculating semi-major axis from
 * vis-viva,
 *   a = 1 / (2/r - v^2/mu),   mu = central->gm,
 * and the body's CURRENT distance r from the central body; when the
 * relative state is unbound or degenerate (2/r - v^2/mu <= 0, i.e.
 * parabolic or hyperbolic, or mu or r non-positive, or a non-finite
 * result), the scale is r alone. The cap makes the scale continuous
 * at the parabolic boundary: as the orbit approaches escape from
 * below, a grows without bound, so the bound branch already returns
 * r wherever a >= r, and the unbound branch returns the same r;
 * just-below-escape and at-escape agree. Without the cap a
 * near-parabolic bound orbit inflates a arbitrarily (a to infinity
 * as e approaches 1) and the classification jumps discontinuously
 * at escape speed. The Hill argument is local either way:
 * r * cbrt(m / 3M) is the tidal sphere at the distance the body
 * actually occupies. Both the cap and the fallback use the
 * central-relative distance, never the pair separation, so the
 * closeness ratio y = d / r_hill stays separation-dependent in
 * every branch (limitation noted at the mutual-Hill build below). */
static double hill_length_scale_(const K26AstroBody *b,
                                 const K26AstroBody *central)
{
    K26V3 rel = k26astro_pos_sub(&b->pos, &central->pos);
    double r = sqrt(rel.x * rel.x + rel.y * rel.y + rel.z * rel.z);
    if (!(r > 0.0)) return 0.0;
    double mu = central->gm;
    if (!(mu > 0.0)) return r;
    double vx = b->vel.x - central->vel.x;
    double vy = b->vel.y - central->vel.y;
    double vz = b->vel.z - central->vel.z;
    double v2 = vx * vx + vy * vy + vz * vz;
    double inv_a = 2.0 / r - v2 / mu;
    if (!(inv_a > 0.0)) return r;
    double a = 1.0 / inv_a;
    if (!isfinite(a)) return r;
    return (a < r) ? a : r;
}

double k26astro_mercurius_hill_radius(const K26AstroBody *i,
                                       const K26AstroBody *j,
                                       const K26AstroBody *central)
{
    if (!i || !j || !central) return 0.0;
    /* Mutual Hill radius in the standard form,
     *   R_hill_ij = ((a_i + a_j) / 2) * cbrt((m_i + m_j) / (3 M)),
     * with a_i, a_j the per-body length scales stated at
     * hill_length_scale_ above. The formula is this tree's own
     * construction of that standard form: Rein, Hernandez, Tamayo
     * et al. (2019), MNRAS 485(4):5490-5497, section 3, names "a
     * small multiple of the mutual Hill radius" as the typical
     * switching distance but prints no formula, and Chambers
     * (1999), MNRAS 304:793-799, sets its changeover distance in
     * mutual Hill radii, likewise without printing one.
     *
     * The previous revision approximated the length scale by the
     * pair separation, which cancelled the distance out of
     * y = d / r_hill entirely: pairs were classified by mass ratio
     * alone, at any separation. With the central-relative scales in
     * the unit, y measures closeness for planetary-style
     * architectures: near passes split, co-orbital distant pairs do
     * not. Stated limitation, inherent to Hill-type criteria: for a
     * radially separated pair (one body near the central mass, one
     * far out), the far body's scale tracks the pair distance
     * itself, so y collapses to a mass-ratio-only number again and
     * a sufficiently massive pair splits at any such separation.
     * Architectures that are not roughly planetary must not rely on
     * this detector to keep distant pairs unsplit. */
    double m_central = central->mass;
    if (!(m_central > 0.0)) return 0.0;
    double m_sum = i->mass + j->mass;
    if (!(m_sum > 0.0))     return 0.0;
    double a_i = hill_length_scale_(i, central);
    double a_j = hill_length_scale_(j, central);
    if (!(a_i > 0.0) || !(a_j > 0.0)) return 0.0;
    return 0.5 * (a_i + a_j) * cbrt(m_sum / (3.0 * m_central));
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
    world->mercurius_central_idx = -1;
    int n = world->grav.n_bodies;
    if (n < 2) { world->n_encounters = 0; return 0; }

    /* Central body = the largest mass. In the solar system this is
     * the Sun; for moon-around-planet sub-systems the caller should
     * set up a hierarchical world (out of scope in v0.1). The index
     * is recorded on the world for the split admission: a
     * Wisdom-Holman base may only split when this index is 0, the
     * WH drift's hard-wired Kepler primary (orbit_step.c). */
    double m_central   = 0.0;
    int    idx_central = -1;
    for (int k = 0; k < n; k++) {
        double m = world->grav.bodies[k].mass;
        if (m > m_central) { m_central = m; idx_central = k; }
    }
    if (!(m_central > 0.0)) { world->n_encounters = 0; return 0; }
    world->mercurius_central_idx = idx_central;

    world->n_encounters = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            /* Pairs involving the central body are never encounter
             * pairs: MERCURIUS K-weights the planet-planet
             * interaction terms only (Rein et al. 2019, section 2,
             * eqs 4-5), and the split composition (orbit_step.c)
             * hands every central pair to the near integrator at
             * full weight on both admitted bases (central_plus1 in
             * forces.h), so the drift owns the central attraction
             * and the far kick carries only the weighted pair
             * terms.
             *
             * Wisdom-Holman admission: the WH drift's Kepler
             * primary is hard-wired to body 0 (wisdom_holman.c,
             * mu0 = b[0].gm). When the detector's central body is
             * not body 0, the base's own unsplit steps and the
             * split's central-pair partition would disagree about
             * which body is primary, so orbit_step.c admits a WH
             * base to the split only when mercurius_central_idx is
             * 0; otherwise WH takes single full-force steps, where
             * the primary choice is WH's own pre-existing
             * approximation and no split arithmetic depends on
             * it. */
            if (i == idx_central || j == idx_central) continue;
            const K26AstroBody *bi = &world->grav.bodies[i];
            const K26AstroBody *bj = &world->grav.bodies[j];
            double rh = k26astro_mercurius_hill_radius(
                    bi, bj, &world->grav.bodies[idx_central]);
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
