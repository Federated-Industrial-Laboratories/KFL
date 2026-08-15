/* dump.h - the headless presenter's interface.
 *
 * One panel, one episode selection, one step range, written to a
 * stream with no window anywhere in the process, so the panels can be
 * driven on a machine with no display.
 */
#ifndef K26RL_VIEW_DUMP_H
#define K26RL_VIEW_DUMP_H

#include <stdint.h>
#include <stdio.h>

#include <string>

#include "model.h"
#include "resim.h"

namespace k26rl_view {

struct DumpOptions {
    std::string panel;                 /* meta, timeline, reward, obs,
                                        * action, traj, world, attitude,
                                        * overlay, wireframe, scrub,
                                        * resim, all */
    uint32_t episode = UINT32_MAX;     /* all indexed episodes by default */
    uint32_t step_lo = 0;
    uint32_t step_hi = UINT32_MAX;     /* to the end of the episode */
    std::string artifact;              /* enables the re-simulation panel */
    std::string asset;                 /* enables the wireframe panel */
};

/* The trajectory panel's standing label, shared with the interface so
 * both say the same thing about what the reconstruction is. */
extern const char *const TRAJECTORY_LABEL;

int dump(FILE *f, Model &m, const DumpOptions &o);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_DUMP_H */
