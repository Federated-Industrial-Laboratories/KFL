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
 * the compiled environment enabling the re-simulation comparison,
 * the world frame and the attitude panel, or empty when none was
 * supplied; asset is the vehicle assembly the wireframe panel draws
 * once its digest matches the one the recording carries. Returns a
 * process exit status. */
int run_gui(Model &model, const std::string &artifact,
            const std::string &asset);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_GUI_H */
