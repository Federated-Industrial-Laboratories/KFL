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

namespace k26rl_view {

/* Open the replay window over an already-opened model.
 *
 * The options are the same ones the headless dump takes, and
 * deliberately: the artifact enables the re-simulation comparison,
 * the world frame, the attitude panel and the scene's body poses; the
 * asset enables the wireframe and the scene's geometry once its
 * digest matches the one the recording carries; and the scene
 * settings start the view where the command line asked for it, so a
 * picture can be reproduced headlessly by repeating the arguments.
 *
 * Returns a process exit status. */
int run_gui(Model &model, const DumpOptions &opt);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_GUI_H */
