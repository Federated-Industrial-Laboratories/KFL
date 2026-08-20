/* gui_internal.h - the window's shared spine: the interface state,
 * the palette, the shared table helpers, and the panels the run
 * loop calls. One window, five translation units; each module file
 * says which part of the surface it holds.
 */
#ifndef K26RL_VIEW_GUI_INTERNAL_H
#define K26RL_VIEW_GUI_INTERNAL_H

/* gui.cpp - the replay window.
 *
 * Five panels over the model the headless dump also reads: reward and
 * episode return, observation channels, action traces with their
 * declared bounds, the observer-relative trajectory view, and the run
 * and episode metadata. A sixth appears when an artifact is supplied:
 * the recorded and reconstructed streams side by side with a bitwise
 * verdict.
 *
 * The timeline scrubs by step index. Seeking inside the loaded
 * episode touches no file at all, and changing episode costs the one
 * index lookup and one bounded read the format promises, which is why
 * scrubbing is immediate rather than merely fast.
 *
 * Nothing here writes: no control reaches a simulation, the file is
 * opened read-only by the reader beneath, and the re-simulation panel
 * offers no way to alter a recorded action.
 *
 * With a live source the window watches a run as it happens. Once a
 * frame it takes whatever the ring has published, which costs the
 * producer nothing and cannot be noticed by it, and follows the
 * newest step unless a person takes hold of the timeline, at which
 * point following stops and the run goes on arriving behind them. The
 * live panel states what has arrived and what has not: a producer
 * that outran this window overwrote frames, and the count of them and
 * the holes they left are shown rather than drawn over.
 *
 * The trajectory view plots reconstructed points and stands beside
 * the scene view rather than being replaced by it.
 *
 * The scene view is the three-dimensional one, and it is drawn here
 * and computed nowhere here. Every transform, projection, cull and
 * visibility decision behind it lives in the model, which hands this
 * file a set of local coordinates, one matrix per element, and the
 * list of segments it decided to draw. What follows is buffer
 * uploads, uniform writes and draw calls, and the ImGui widgets that
 * set the view state the model resolves. If a matrix is computed in
 * this file, the split has been broken.
 */

#include "gui.h"

#include "asset.h"

#include "scene.h"

#include <dirent.h>

#include <float.h>

#include <math.h>

#include <stdarg.h>

#include <stdio.h>

#include <string.h>

#include <sys/stat.h>

#include <unistd.h>

#include <algorithm>

#include <map>

#include "dump.h"

#include "pngout.h"

#include "prefs.h"

#include "resim.h"

#include "imgui.h"

#include "imgui_internal.h"

#include "backends/imgui_impl_glfw.h"

#include "backends/imgui_impl_opengl3.h"

#include "implot.h"

#include <GLFW/glfw3.h>

extern "C" {
#include "k26m3d.h"
}

namespace k26rl_view {



struct Ui {
    Model *model;
    std::string artifact;
    /* The assemblies the command line asked for, read once and bound
     * once, because reading them per frame would read a file per
     * frame. The bindings are the model's own answer, taken through
     * the same function the headless dump calls. */
    std::vector<AssetRequest> asset_reqs;
    std::vector<AssetBound> assets;
    bool asset_tried;
    uint32_t episode_index;
    uint32_t step;
    std::vector<char> channel_on;   /* one flag per observation channel */
    int traj_pick;
    float yaw, pitch;
    bool resim_done;
    ResimResult resim;
    std::string error;
    /* The scene view's state. Its rebuild is its own because its
     * reference frame is its own: the world-frame panel asks the body
     * getter for the world origin and the scene asks for whatever
     * frame the view is set to, and one rebuild cannot answer both. */
    SceneOptions scene_opt;
    ResimResult scene_resim;
    bool scene_resim_done;
    bool scene_auto_resim;
    uint32_t scene_resim_ref;
    uint32_t scene_resim_ep;
    Scene scene;
    bool scene_ready;
    /* Live sources only: whether the view rides the newest step as
     * frames arrive. Taking hold of the timeline drops it, because a
     * person who scrubbed back has said where they want to be. */
    bool follow;
    /* Playback: the step advances against the wall clock at the
     * chosen multiple of simulated time, so 1x plays a control
     * period per control period. Window state, like the camera. */
    bool playing;
    double play_rate;
    double play_accum;
    /* The persisted settings, or null when running session-only. */
    Prefs *prefs;
    /* Windows the menu bar opens. */
    bool show_settings;
    bool show_help;
    bool show_open;
    /* The open window's state: the directory listed, the file
     * picked, the artifact to rebuild with, and the last failure. */
    std::string open_dir;
    std::string open_pick;
    char open_artifact[512];
    std::string open_error;
    /* A capture requested this frame: 0 none, 1 the window as
     * rendered, 2 the scene alone, writing to capture_path. The
     * note names the file the last capture wrote, shown in the menu
     * bar for a while. */
    int capture_kind;
    std::string capture_path;
    std::string capture_note;
    int capture_note_frames;
    /* The save window: which capture it will request, where, and
     * under what name. The directory is remembered between runs. */
    bool show_save;
    int save_kind;
    std::string save_dir;
    char save_name[256];
    /* The observation table's filter. */
    char obs_filter[64];
    LiveOptions live;
};

const unsigned COL_WINDOW_BG       = 0x101012u;

const unsigned COL_SURFACE_BG      = 0x595957u;

const unsigned COL_BORDER          = 0x8e8e8cu;

const unsigned COL_TEXT            = 0xc8c8ccu;

const unsigned COL_TEXT_DIM        = 0x9a9a98u;

const unsigned COL_ACCENT          = 0xfbbf24u;

const unsigned COL_ACCENT_SECOND   = 0xb46a00u;

const unsigned COL_INPUT_BG        = 0x2a2a2cu;

const unsigned COL_ERROR           = 0xef4444u;

const unsigned COL_HOVER           = 0x6a6a68u;

/* The scene's draw path: entry points, one shader pair, and buffers.
 *
 * The entry points are resolved through the window system rather than
 * linked, because a 3.3 core context's symbols are the context's and
 * not the library's; a build that links them directly builds on one
 * machine and fails to start on another.
 *
 * One program does both passes. Attribute 0 is the vertex, attribute
 * 1 is the shade a face carries; the line pass leaves attribute 1
 * disabled and sets its constant to one, which is the same shader
 * with the depth cue turned off rather than a second shader to keep
 * in step with the first.
 */

struct SceneGl {
    bool ready;
    std::string error;
    GLuint program;
    GLuint vao;
    GLuint vbo;
    GLuint ebo;
    GLuint shade_vbo;
    GLint u_mvp;
    GLint u_colour;
    /* Buffers whose contents are fixed in a body's own frame are
     * uploaded once and reused, keyed by the element they belong to.
     * A viewer that re-uploaded a craft's mesh every frame is a
     * viewer that will be measured re-uploading a craft's mesh every
     * frame.
     *
     * The uploaded floats are kept beside the buffer and compared,
     * rather than the vertex count alone: a body's axes and its
     * thrusters are drawn at lengths the view sets, so their
     * coordinates change while their count does not, and a cache
     * keyed on the count would show the length the view had when the
     * craft was first drawn. */
    std::map<std::string, GLuint> cached;
    std::map<std::string, std::vector<float> > cached_data;

    PFNGLCREATESHADERPROC CreateShader;
    PFNGLSHADERSOURCEPROC ShaderSource;
    PFNGLCOMPILESHADERPROC CompileShader;
    PFNGLGETSHADERIVPROC GetShaderiv;
    PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog;
    PFNGLDELETESHADERPROC DeleteShader;
    PFNGLCREATEPROGRAMPROC CreateProgram;
    PFNGLATTACHSHADERPROC AttachShader;
    PFNGLLINKPROGRAMPROC LinkProgram;
    PFNGLGETPROGRAMIVPROC GetProgramiv;
    PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog;
    PFNGLDELETEPROGRAMPROC DeleteProgram;
    PFNGLUSEPROGRAMPROC UseProgram;
    PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation;
    PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv;
    PFNGLUNIFORM4FPROC Uniform4f;
    PFNGLGENVERTEXARRAYSPROC GenVertexArrays;
    PFNGLBINDVERTEXARRAYPROC BindVertexArray;
    PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays;
    PFNGLGENBUFFERSPROC GenBuffers;
    PFNGLBINDBUFFERPROC BindBuffer;
    PFNGLBUFFERDATAPROC BufferData;
    PFNGLDELETEBUFFERSPROC DeleteBuffers;
    PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray;
    PFNGLDISABLEVERTEXATTRIBARRAYPROC DisableVertexAttribArray;
    PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer;
    PFNGLVERTEXATTRIB1FPROC VertexAttrib1f;
};

#define GL_LOAD_(g, f) \
    if (!gl_load_(&(g)->f, "gl" #f, &(g)->error)) return false


void kv_(const char *key, const char *fmt, ...);
void panel_action_(Ui &ui, const Episode &ep);
void panel_attitude_(Ui &ui, const Episode &ep);
void panel_live_(Ui &ui);
void panel_meta_(Ui &ui, const Episode &ep);
void panel_obs_(Ui &ui, const Episode &ep);
void panel_resim_(Ui &ui, const Episode &ep);
void panel_reward_(Ui &ui, const Episode &ep);
void panel_timeline_(Ui &ui, const Episode &ep);
void panel_traj_(Ui &ui, const Episode &ep);
void panel_wireframe_(Ui &ui, const Episode &ep);
void panel_world_(Ui &ui, const Episode &ep);
bool table_(const char *id, int cols);
void panel_scene_(Ui &ui, const Episode &ep, SceneGl &gl);
void scene_draw_(SceneGl *gl, const Scene &sc, int fbw, int fbh);
void scene_gl_free_(SceneGl *gl);
bool scene_gl_init_(SceneGl *gl);
void apply_theme_(ImGuiStyle &st);
ImVec4 rgb_(unsigned hex, float a = 1.0f);
void capture_now_(Ui &ui, int w, int h);
void menu_bar_(Ui &ui, GLFWwindow *win, bool *build_layout);
void panel_help_(Ui &ui);
void panel_open_(Ui &ui);
void panel_save_(Ui &ui);
void panel_settings_(Ui &ui);


}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_GUI_INTERNAL_H */
