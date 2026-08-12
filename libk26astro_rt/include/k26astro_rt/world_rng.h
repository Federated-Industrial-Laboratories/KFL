/* world_rng.h - the world-seeded deterministic RNG.
 *
 * The detect, softkill, and impactor noise generators take a
 * K26CRng*; detect.h documents bit-identical replay under "a
 * world-seeded RNG", and the other two document the same mechanism
 * in their own words. This header is where that RNG lives. It is
 * separate from world.h so that consumers who never thread noise do
 * not pull libk26compute into their include path.
 *
 * An unseeded world has no RNG: k26astro_world_rng returns NULL, and
 * passing NULL to the noise generators keeps their documented
 * mean-only deterministic path. Nothing changes for existing callers
 * until a seed is set. */
#ifndef K26ASTRO_RT_WORLD_RNG_H
#define K26ASTRO_RT_WORLD_RNG_H

#include <stdint.h>

#include "k26astro_rt/world.h"
#include "k26compute.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Seed the world's RNG. First call creates the stream; a later call
 * reinitialises it, so a fixed seed gives a bit-identical draw
 * sequence from the point of seeding. Returns 0 on success or a
 * negative K26ASTRO_RT_E_* code. */
int k26astro_world_set_seed(K26AstroWorld *world, uint64_t seed);

/* The world-seeded RNG, for threading into the noise generators.
 * NULL until k26astro_world_set_seed is called. The pointer stays
 * valid for the world's lifetime; draws advance its state. */
K26CRng *k26astro_world_rng(K26AstroWorld *world);

#ifdef __cplusplus
}
#endif

#endif /* K26ASTRO_RT_WORLD_RNG_H */
