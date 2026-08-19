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
 * With an artifact and an assembly supplied, the scene view draws the
 * craft in three dimensions: its wireframe, its colliders, its axes,
 * its track, its velocity, its docking ports and its thrusters, from
 * a camera the model resolves. Every transform behind that picture
 * lives in the model, so `--dump scene` writes the projected geometry
 * with no window anywhere in the process.
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
#include "scene.h"

static int usage_(const char *prog)
{
    fprintf(stderr,
        "usage: %s FILE\n"
        "       %s --dump PANEL [--episode K] [--steps A:B]\n"
        "               [--artifact PATH] [--asset PATH] [scene options]\n"
        "               FILE\n"
        "\n"
        "panels: meta timeline reward obs action traj scrub world\n"
        "        attitude overlay wireframe scene resim all\n"
        "\n"
        "  --episode K     restrict to the K-th indexed episode\n"
        "  --steps A:B     restrict to steps [A, B) of each episode\n"
        "  --artifact PATH the compiled environment, enabling\n"
        "                  re-simulation, the world frame, the\n"
        "                  attitude panel and the scene's body poses\n"
        "  --asset PATH    the vehicle assembly, enabling the wireframe\n"
        "                  once its digest matches the one recorded\n"
        "\n"
        "scene options:\n"
        "  --frame BODY            reference body for positions, or\n"
        "                          `origin` for the world origin\n"
        "  --camera MODE           orbit, chase or free\n"
        "  --camera-target BODY    the body the camera orbits or chases\n"
        "  --orbit AZ,EL,R         azimuth and elevation in degrees and\n"
        "                          radius in metres\n"
        "  --chase X,Y,Z           eye offset in the target's own frame\n"
        "  --eye X,Y,Z             free-mode eye, reference frame\n"
        "  --look X,Y,Z            free-mode target point\n"
        "  --up X,Y,Z              up vector\n"
        "  --projection KIND       perspective or orthographic\n"
        "  --fov DEG               vertical field of view\n"
        "  --ortho-height M        orthographic view height\n"
        "  --clip NEAR,FAR         clip planes in metres\n"
        "  --viewport WxH          viewport in pixels\n"
        "  --elements LIST         comma-separated element names, or\n"
        "                          `all` or `none`\n"
        "  --velocity-seconds S    seconds of velocity the vector shows\n"
        "  --axis-length M         length of the body axis lines\n"
        "  --thruster-scale S      metres of line per newton of thrust\n"
        "  --shading               draw the shaded depth cue as well\n"
        "  --light X,Y,Z           view-space light direction for it\n",
        prog, prog);
    return 2;
}

/* Three numbers separated by commas, for the vector options. */
static bool triple_(const char *s, double *out)
{
    char *end = 0;
    for (int i = 0; i < 3; i++) {
        out[i] = strtod(s, &end);
        if (end == s)
            return false;
        s = end;
        if (i < 2) {
            if (*s != ',')
                return false;
            s++;
        }
    }
    return *s == '\0';
}

static bool pair_(const char *s, double *a, double *b)
{
    char *end = 0;
    *a = strtod(s, &end);
    if (end == s || *end != ',')
        return false;
    s = end + 1;
    *b = strtod(s, &end);
    return end != s && *end == '\0';
}

/* The element list. `all` and `none` set every toggle; otherwise the
 * named elements are on and the rest are off, so a gate arm can ask
 * for one element and be certain the others are absent rather than
 * merely unmentioned. */
static bool elements_(const char *list, k26rl_view::SceneOptions *o)
{
    std::string s(list);
    size_t i = 0;

    if (s == "all" || s == "none") {
        for (int k = 0; k < k26rl_view::ELEM_KIND_COUNT; k++)
            o->enabled[k] = (s == "all");
        return true;
    }
    for (int k = 0; k < k26rl_view::ELEM_KIND_COUNT; k++)
        o->enabled[k] = false;
    while (i <= s.size()) {
        size_t j = s.find(',', i);
        std::string name = s.substr(i, j == std::string::npos
                                       ? std::string::npos : j - i);
        int which = k26rl_view::element_by_name(name);
        if (which < 0) {
            fprintf(stderr, "k26rl_view: `%s` is not a scene element\n",
                    name.c_str());
            return false;
        }
        o->enabled[which] = true;
        if (j == std::string::npos)
            break;
        i = j + 1;
    }
    return true;
}

int main(int argc, char **argv)
{
    k26rl_view::DumpOptions opt;
    const char *path = 0;
    bool headless = false;
    std::string frame_name, target_name;
    std::string err;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : 0;

        if (strcmp(a, "--dump") == 0) {
            if (!v)
                return usage_(argv[0]);
            headless = true;
            opt.panel = argv[++i];
        } else if (strcmp(a, "--episode") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.episode = (uint32_t)strtoul(argv[++i], 0, 10);
        } else if (strcmp(a, "--steps") == 0) {
            const char *colon;
            if (!v)
                return usage_(argv[0]);
            colon = strchr(v, ':');
            if (!colon)
                return usage_(argv[0]);
            opt.step_lo = (uint32_t)strtoul(v, 0, 10);
            opt.step_hi = (uint32_t)strtoul(colon + 1, 0, 10);
            i++;
        } else if (strcmp(a, "--asset") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.asset = argv[++i];
        } else if (strcmp(a, "--artifact") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.artifact = argv[++i];
        } else if (strcmp(a, "--frame") == 0) {
            if (!v)
                return usage_(argv[0]);
            frame_name = argv[++i];
        } else if (strcmp(a, "--camera-target") == 0) {
            if (!v)
                return usage_(argv[0]);
            target_name = argv[++i];
        } else if (strcmp(a, "--camera") == 0) {
            if (!v)
                return usage_(argv[0]);
            i++;
            if (strcmp(v, "orbit") == 0) {
                opt.scene.camera.mode = k26rl_view::CAMERA_ORBIT;
            } else if (strcmp(v, "chase") == 0) {
                opt.scene.camera.mode = k26rl_view::CAMERA_CHASE;
            } else if (strcmp(v, "free") == 0) {
                opt.scene.camera.mode = k26rl_view::CAMERA_FREE;
            } else {
                fprintf(stderr, "k26rl_view: `%s` is not a camera mode\n", v);
                return usage_(argv[0]);
            }
        } else if (strcmp(a, "--projection") == 0) {
            if (!v)
                return usage_(argv[0]);
            i++;
            if (strcmp(v, "perspective") == 0) {
                opt.scene.camera.projection =
                    k26rl_view::PROJECTION_PERSPECTIVE;
            } else if (strcmp(v, "orthographic") == 0) {
                opt.scene.camera.projection =
                    k26rl_view::PROJECTION_ORTHOGRAPHIC;
            } else {
                fprintf(stderr, "k26rl_view: `%s` is not a projection\n", v);
                return usage_(argv[0]);
            }
        } else if (strcmp(a, "--orbit") == 0) {
            double t[3];
            if (!v || !triple_(v, t))
                return usage_(argv[0]);
            i++;
            opt.scene.camera.azimuth_deg = t[0];
            opt.scene.camera.elevation_deg = t[1];
            opt.scene.camera.radius = t[2];
        } else if (strcmp(a, "--chase") == 0) {
            if (!v || !triple_(v, opt.scene.camera.chase))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--eye") == 0) {
            if (!v || !triple_(v, opt.scene.camera.eye))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--look") == 0) {
            if (!v || !triple_(v, opt.scene.camera.look))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--up") == 0) {
            if (!v || !triple_(v, opt.scene.camera.up))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--light") == 0) {
            if (!v || !triple_(v, opt.scene.light))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--fov") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.scene.camera.fov_y_deg = strtod(argv[++i], 0);
        } else if (strcmp(a, "--ortho-height") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.scene.camera.ortho_height = strtod(argv[++i], 0);
        } else if (strcmp(a, "--clip") == 0) {
            if (!v || !pair_(v, &opt.scene.camera.near_plane,
                             &opt.scene.camera.far_plane))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--viewport") == 0) {
            const char *x;
            if (!v)
                return usage_(argv[0]);
            x = strchr(v, 'x');
            if (!x)
                return usage_(argv[0]);
            opt.scene.viewport.width = (uint32_t)strtoul(v, 0, 10);
            opt.scene.viewport.height = (uint32_t)strtoul(x + 1, 0, 10);
            i++;
        } else if (strcmp(a, "--elements") == 0) {
            if (!v || !elements_(v, &opt.scene))
                return usage_(argv[0]);
            i++;
        } else if (strcmp(a, "--velocity-seconds") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.scene.velocity_seconds = strtod(argv[++i], 0);
        } else if (strcmp(a, "--axis-length") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.scene.axis_length = strtod(argv[++i], 0);
        } else if (strcmp(a, "--thruster-scale") == 0) {
            if (!v)
                return usage_(argv[0]);
            opt.scene.thruster_scale = strtod(argv[++i], 0);
        } else if (strcmp(a, "--shading") == 0) {
            opt.scene.shading = true;
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

    /* Body names resolve against the file, so they resolve here and
     * not while the arguments are being read. A name the recording
     * does not carry is refused rather than silently taken as the
     * world origin, which would draw a different scene from the one
     * asked for and say nothing. */
    if (!frame_name.empty() &&
        !k26rl_view::scene_body_by_name(model.spec(), frame_name,
                                        &opt.scene.frame)) {
        fprintf(stderr, "%s: no body named `%s` in this recording\n",
                argv[0], frame_name.c_str());
        return 1;
    }
    if (!target_name.empty() &&
        !k26rl_view::scene_body_by_name(model.spec(), target_name,
                                        &opt.scene.camera.target)) {
        fprintf(stderr, "%s: no body named `%s` in this recording\n",
                argv[0], target_name.c_str());
        return 1;
    }

    if (headless)
        return k26rl_view::dump(stdout, model, opt);
    return k26rl_view::run_gui(model, opt);
}
