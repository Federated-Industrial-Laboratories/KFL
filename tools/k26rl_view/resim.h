/* resim.h - deterministic re-simulation, reconstruct and compare only.
 *
 * Given the artifact that produced an episode file, an episode is
 * exactly reproducible from what the file records: its identity
 * triple, the governing seed that triple's ordinal resolves to, and
 * its recorded action stream. This rebuilds one episode that way and
 * compares the reconstruction against the recording, value by value,
 * bitwise.
 *
 * It reconstructs and compares. It does not edit: no action stream
 * this code drives is anything but the one the file recorded, and no
 * interface above it offers a way to change one. The feature is a
 * determinism check a person can run in the field, not a sandbox.
 *
 * The rebuild reaches its episode index through the explicit reset
 * call rather than by stepping through the episodes before it, so no
 * simulated time passes on the way and the reconstruction of episode
 * k does not depend on reproducing episode k-1 first.
 */
#ifndef K26RL_VIEW_RESIM_H
#define K26RL_VIEW_RESIM_H

#include <stdint.h>

#include <string>
#include <vector>

#include "model.h"

namespace k26rl_view {

struct ResimResult {
    bool ran = false;              /* the rebuild completed */
    bool equal = false;            /* every compared value bit-identical */
    std::string message;           /* why not, when either is false */
    uint32_t abi_version = 0;
    uint32_t steps_compared = 0;
    uint32_t first_divergence = 0; /* step index, valid when !equal */
    std::string divergence_kind;   /* which stream diverged first */
    /* The reconstruction itself, in the recording's layout, so a
     * presenter can draw the two side by side. */
    std::vector<double> obs;
    std::vector<double> rewards;
    std::vector<uint32_t> flags;
    std::vector<double> initial_obs;
    /* World-frame body states, steps_compared * body_count * 6, from
     * the artifact's body getter. Empty when the artifact predates
     * that symbol, which a viewer reports rather than hides. */
    std::vector<double> bodies;
    uint32_t body_count = 0;
    bool has_bodies = false;
};

/* Load an artifact and rebuild one episode of a model's file.
 * artifact_path is the compiled environment's shared object. The ABI
 * is probed the way every consumer probes it: major must equal the
 * major this build was written against, minor must be at least it. */
ResimResult resimulate(const Model &model, const Episode &ep,
                       const std::string &artifact_path);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_RESIM_H */
