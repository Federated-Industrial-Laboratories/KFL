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
    /* Attitudes, steps_compared * body_count * 7, from the
     * artifact's attitude getter: the body-to-world quaternion's w,
     * x, y and z, then the body-frame angular velocity. Empty when
     * the artifact predates that symbol, which a viewer reports
     * rather than hides, exactly as it does for the bodies above. */
    std::vector<double> attitudes;
    bool has_attitudes = false;
    /* Actuator drives, steps_compared * actuator_count * 10, from
     * the artifact's actuator getter: body index, kind (0 wheel,
     * 1 magnetorquer, 2 thruster), mounting position (body frame,
     * metres, three), axis or thrust direction (body frame, unit,
     * three), applied magnitude, full-scale magnitude. For a
     * thruster the applied figure is the step-mean imparted force,
     * so force times step duration is the imparted impulse; the
     * command clamped to the limit for the others, per the getter's
     * contract. Empty when the artifact predates that symbol, which
     * a viewer reports rather than hides, exactly as it does for
     * the bodies above. */
    std::vector<double> actuators;
    uint32_t actuator_count = 0;
    bool has_actuators = false;
    /* Datalink state, steps_compared * datalink_count * 5, from the
     * artifact's datalink getter: the transmitting body's index, the
     * receiving body's index, the closure flag at the transmitter's
     * latest broadcast, the margin in decibels against its declared
     * threshold, and the seconds since a closed broadcast last reached
     * the receiver, negative when none has.
     *
     * The two flags are not the same statement and a viewer needs
     * both. `datalink_symbol` says the artifact carries the getter at
     * all, which is what separates an artifact from before ABI minor 7
     * from a program that simply declares no datalink; `has_datalinks`
     * says there are pairs to report. Only the first is worth telling
     * a reader about, and it is told rather than hidden. */
    std::vector<double> datalinks;
    uint32_t datalink_count = 0;
    bool has_datalinks = false;
    bool datalink_symbol = false;
};

/* Load an artifact and rebuild one episode of a model's file.
 * artifact_path is the compiled environment's shared object. The ABI
 * is probed the way every consumer probes it: major must equal the
 * major this build was written against, minor must be at least it.
 *
 * `reference` is the body the recorded positions come back relative
 * to, passed straight to the body getter, and K26RL_BODY_REF_ORIGIN
 * asks it for the world origin. It matters at scale: a
 * position in the runtime is a sector index and a bounded offset, and
 * the getter subtracts exactly in that form, while a caller that took
 * two world-origin positions and subtracted them would have lost what
 * the sector grid exists to keep before it started. */
ResimResult resimulate(const Model &model, const Episode &ep,
                       const std::string &artifact_path,
                       uint32_t reference);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_RESIM_H */
