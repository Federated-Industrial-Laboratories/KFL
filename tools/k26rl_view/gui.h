/* gui.h - the graphical presenter's entry point.
 *
 * Separated from the model and from the headless presenter so that
 * nothing above this line links a graphics library: the dump path
 * never enters here, and a build that fails to open a display fails
 * only here.
 */
#ifndef K26RL_VIEW_GUI_H
#define K26RL_VIEW_GUI_H

#include <string>

#include "model.h"

namespace k26rl_view {

/* Open the replay window over an already-opened model. artifact is
 * the compiled environment enabling the re-simulation comparison, or
 * empty when none was supplied. Returns a process exit status. */
int run_gui(Model &model, const std::string &artifact);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_GUI_H */
