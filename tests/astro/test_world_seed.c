/* tests/astro/test_world_seed.c - world-seeded RNG gate.
 *
 * The detect, softkill, and impactor noise generators document
 * bit-identical replay under "a world-seeded RNG" (K26CRng*, NULL for
 * the mean-only path). This gate pins the world side of that
 * contract: unseeded worlds expose no RNG, seeding exposes one, a
 * fixed seed gives a bit-identical draw sequence, and reseeding
 * restarts the stream. */
#include "k26astro_rt/world_rng.h"

#include <stdio.h>
#include <stdlib.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#define N_DRAWS 64

int main(void)
{
    K26AstroWorld *w = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                             K26ASTRO_COORDS_SECTOR_GRID);
    ASSERT(w);

    /* Unseeded: no RNG, which keeps the NULL mean-only noise path. */
    ASSERT(k26astro_world_rng(w) == NULL);
    ASSERT(k26astro_world_set_seed(NULL, 1) < 0);

    /* Seeded: draws are deterministic under the seed. */
    ASSERT(k26astro_world_set_seed(w, 0x4B464C5F524Cu) == 0);
    K26CRng *r = k26astro_world_rng(w);
    ASSERT(r != NULL);
    double first[N_DRAWS];
    for (int i = 0; i < N_DRAWS; i++) first[i] = k26c_rng_uniform(r);

    /* Reseeding with the same seed restarts the identical stream. */
    ASSERT(k26astro_world_set_seed(w, 0x4B464C5F524Cu) == 0);
    r = k26astro_world_rng(w);
    ASSERT(r != NULL);
    for (int i = 0; i < N_DRAWS; i++) {
        double d = k26c_rng_uniform(r);
        ASSERT(d == first[i]);
    }

    /* A different seed diverges somewhere in the sequence. */
    ASSERT(k26astro_world_set_seed(w, 0x4B464C5F524Du) == 0);
    r = k26astro_world_rng(w);
    int differs = 0;
    for (int i = 0; i < N_DRAWS; i++) {
        if (k26c_rng_uniform(r) != first[i]) differs = 1;
    }
    ASSERT(differs);

    k26astro_world_destroy(w);
    printf("test_world_seed: all assertions passed\n");
    return 0;
}
