/* k26rl_view - episode file viewer for the reinforcement learning
 * environment layer (host-only).
 *
 *   k26rl_view FILE                     open the replay window
 *   k26rl_view --dump PANEL [opts] FILE headless dump of one panel
 *
 * Replay mode reads the binary episode format through the format
 * library's reader: the timeline scrubs by step index, episode
 * boundaries are marked, terminated, truncated and faulted endings
 * are distinguished, and a file cut mid-write opens to its readable
 * prefix with the truncation point reported.
 *
 * With an artifact supplied, the recorded episode is re-simulated
 * from its identity triple, its recorded governing seed, and its
 * recorded action stream, and the recorded and reconstructed streams
 * are shown side by side with a bitwise equality verdict. It
 * reconstructs and compares; nothing here edits an action stream.
 *
 * The dump mode exists so the panels are testable without a display.
 * Both presenters read one model, so the numbers checked headlessly
 * are the numbers the window draws.
 *
 * Live attach to a serving simulation is not in this version.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dump.h"
#include "gui.h"
#include "model.h"

static int usage_(const char *prog)
{
    fprintf(stderr,
        "usage: %s FILE\n"
        "       %s --dump PANEL [--episode K] [--steps A:B]\n"
        "               [--artifact PATH] FILE\n"
        "\n"
        "panels: meta timeline reward obs action traj scrub world resim all\n"
        "\n"
        "  --episode K     restrict to the K-th indexed episode\n"
        "  --steps A:B     restrict to steps [A, B) of each episode\n"
        "  --artifact PATH the compiled environment, enabling\n"
        "                  re-simulation and its comparison verdict\n",
        prog, prog);
    return 2;
}

int main(int argc, char **argv)
{
    k26rl_view::DumpOptions opt;
    const char *path = 0;
    bool headless = false;
    std::string err;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "--dump") == 0) {
            if (i + 1 >= argc)
                return usage_(argv[0]);
            headless = true;
            opt.panel = argv[++i];
        } else if (strcmp(a, "--episode") == 0) {
            if (i + 1 >= argc)
                return usage_(argv[0]);
            opt.episode = (uint32_t)strtoul(argv[++i], 0, 10);
        } else if (strcmp(a, "--steps") == 0) {
            const char *colon;
            if (i + 1 >= argc)
                return usage_(argv[0]);
            colon = strchr(argv[i + 1], ':');
            if (!colon)
                return usage_(argv[0]);
            opt.step_lo = (uint32_t)strtoul(argv[i + 1], 0, 10);
            opt.step_hi = (uint32_t)strtoul(colon + 1, 0, 10);
            i++;
        } else if (strcmp(a, "--artifact") == 0) {
            if (i + 1 >= argc)
                return usage_(argv[0]);
            opt.artifact = argv[++i];
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "%s: unknown option `%s`\n", argv[0], a);
            return usage_(argv[0]);
        } else if (!path) {
            path = a;
        } else {
            fprintf(stderr, "%s: more than one file given\n", argv[0]);
            return usage_(argv[0]);
        }
    }
    if (!path)
        return usage_(argv[0]);

    k26rl_view::Model model;
    if (!model.open(path, &err)) {
        fprintf(stderr, "%s: %s\n", argv[0], err.c_str());
        return 1;
    }

    if (headless)
        return k26rl_view::dump(stdout, model, opt);
    return k26rl_view::run_gui(model, opt.artifact);
}
