/* gui.h - the graphical presenter's entry point.
 *
 * Separated from the model and from the headless presenter so that
 * nothing above this line links a graphics library: the dump path
 * never enters here, and a build that fails to open a display fails
 * only here.
 */
#ifndef K26RL_VIEW_GUI_H
#define K26RL_VIEW_GUI_H

#include "dump.h"
#include "model.h"
#include "prefs.h"

namespace k26rl_view {

/* Open the replay window over an already-opened model.
 *
 * The options are the same ones the headless dump takes, and
 * deliberately: the artifact enables the re-simulation comparison,
 * the world frame, the attitude panel and the scene's body poses; the
 * asset enables the wireframe and the scene's geometry once its
 * digest matches the one the recording carries; and the scene
 * settings start the view where the command line asked for it. A
 * window additionally starts from its persisted settings file,
 * which the flags override; run with --session-only, or headlessly,
 * and the file is ignored, so a picture is then reproducible by
 * repeating the arguments alone.
 *
 * `prefs` carries the persisted window settings and the recent
 * files, already loaded; null means session-only, and the window
 * then neither loads nor saves. Returns a process exit status. */
int run_gui(Model &model, const DumpOptions &opt, Prefs *prefs);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_GUI_H */
