/* prefs.h - the window's persisted settings.
 *
 * The interactive window keeps its visual settings between runs, in
 * a plain key=value file beside the layout file, in the working
 * directory. What persists is world-independent: element toggles,
 * scales, the depth cue, the camera's mode and shape, the playback
 * rate, and the recently opened files. Reference frames and camera
 * targets name bodies of one recording and do not persist.
 *
 * Two consumers never read this file, and deliberately. The
 * headless dump answers to its command line alone, so a gate's
 * output cannot depend on a file a working directory happens to
 * hold. And a window run with --session-only neither loads nor
 * saves, so a picture can again be reproduced from its arguments
 * alone. Command-line flags override loaded values in every case.
 */
#ifndef K26RL_VIEW_PREFS_H
#define K26RL_VIEW_PREFS_H

#include <string>
#include <vector>

#include "scene.h"

namespace k26rl_view {

struct Prefs {
    /* Where the file lives; empty means never save. */
    std::string path;
    /* Recently opened episode files, newest first. */
    std::vector<std::string> recent;
    double play_rate = 1.0;
};

/* Load the file at `path` into `p` and apply the visual settings it
 * holds to `s`. Missing file or missing keys leave defaults; an
 * unreadable line is skipped. */
void prefs_load(const char *path, Prefs *p, SceneOptions *s);

/* Write `p` and the persisted subset of `s` to p->path. A best
 * effort at exit; failure is reported by return alone. */
bool prefs_save(const Prefs &p, const SceneOptions &s);

/* Put `file` at the head of the recent list, dropping duplicates,
 * keeping at most eight. */
void prefs_touch_recent(Prefs *p, const std::string &file);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_PREFS_H */
