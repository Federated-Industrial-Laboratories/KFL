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
#include <vector>

#include "asset.h"
#include "model.h"
#include "resim.h"
#include "scene.h"

namespace k26rl_view {

/* Watching a run in progress, headlessly. The ring never blocks and
 * never signals, so a consumer decides for itself how often to look
 * and when to give up, and each of the three ways of stopping is
 * separately bounded because they mean different things: the producer
 * finished, the watcher had seen enough, or nothing is arriving any
 * more from a producer that never said it was done. */
struct LiveOptions {
    std::string tap;               /* the ring's name; empty for a file */
    bool from_start = false;       /* join at the oldest surviving frame */
    uint64_t polls = 0;            /* rounds to run; 0 for no bound */
    uint32_t poll_ms = 5;          /* pause between rounds */
    uint64_t idle_polls = 400;     /* consecutive empty rounds tolerated */
};

struct DumpOptions {
    std::string panel;                 /* meta, timeline, reward, obs,
                                        * action, agents, traj, world,
                                        * attitude, overlay, wireframe,
                                        * scene, scrub, resim, all */
    uint32_t episode = UINT32_MAX;     /* all indexed episodes by default */
    uint32_t step_lo = 0;
    uint32_t step_hi = UINT32_MAX;     /* to the end of the episode */
    std::string artifact;              /* enables the re-simulation panel */
    /* The assemblies, one per `--asset`. A recording of two craft
     * binds two, and each says which body it belongs to, because a
     * list position is not a body. */
    std::vector<AssetRequest> assets;
    /* The scene view's own state. It is here and not in the window
     * because every one of these settings changes what is projected,
     * and a setting the headless dump cannot reach is a setting no
     * gate can vary. */
    SceneOptions scene;
    LiveOptions live;
};

/* The trajectory panel's standing label, shared with the interface so
 * both say the same thing about what the reconstruction is. */
extern const char *const TRAJECTORY_LABEL;

/* Take frames from an attached ring until the run ends, the round
 * bound is reached, or nothing has arrived for the idle bound.
 * Returns the reason it stopped: closed, polls, or idle. When report
 * is set it writes one record per round that brought anything, which
 * is what makes the picture's advance a countable fact rather than a
 * claim about a window nobody can see. */
const char *live_watch(FILE *f, Model &m, const LiveOptions &o, bool report);

int dump(FILE *f, Model &m, const DumpOptions &o);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_DUMP_H */
