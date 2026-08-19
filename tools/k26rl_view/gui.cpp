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

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <map>

#include "dump.h"
#include "resim.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "implot.h"

#include <GLFW/glfw3.h>

extern "C" {
#include "k26m3d.h"
}

namespace k26rl_view {

namespace {

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
    uint32_t scene_resim_ref;
    uint32_t scene_resim_ep;
    Scene scene;
    bool scene_ready;
    /* Live sources only: whether the view rides the newest step as
     * frames arrive. Taking hold of the timeline drops it, because a
     * person who scrubbed back has said where they want to be. */
    bool follow;
    LiveOptions live;
};

/* The palette.
 *
 * Grey chrome with one amber accent, and a red reserved for a
 * failure. Hue carries meaning only where something is an accent or
 * an error, so nothing else in the interface introduces one: a
 * reading is told apart from its neighbours by position and weight
 * rather than by colour, which is what lets a panel drop the sentence
 * explaining what it is.
 *
 * The field wells are dark, and deliberately. Dear ImGui takes one
 * text colour per frame, so a light well under this text colour
 * renders at a contrast ratio near one and the field is unreadable
 * without pushing a colour around every widget in the tree.
 *
 * The values are display-encoded bytes over 255, straight alpha.
 */
ImVec4 rgb_(unsigned hex, float a = 1.0f)
{
    return ImVec4((float)((hex >> 16) & 0xFFu) / 255.0f,
                  (float)((hex >> 8) & 0xFFu) / 255.0f,
                  (float)(hex & 0xFFu) / 255.0f, a);
}

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

/* Every slot is written before the named ones, so a colour this build
 * does not name can never bleed a default through. */
void apply_theme_(ImGuiStyle &st)
{
    for (int i = 0; i < ImGuiCol_COUNT; i++)
        st.Colors[i] = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_Text]                  = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_TextDisabled]          = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_WindowBg]              = rgb_(COL_WINDOW_BG);
    st.Colors[ImGuiCol_ChildBg]               = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_PopupBg]               = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_Border]                = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_BorderShadow]          = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_FrameBg]               = rgb_(COL_INPUT_BG);
    st.Colors[ImGuiCol_FrameBgHovered]        = rgb_(COL_INPUT_BG, 0.85f);
    st.Colors[ImGuiCol_FrameBgActive]         = rgb_(COL_ACCENT_SECOND, 0.55f);
    st.Colors[ImGuiCol_TitleBg]               = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TitleBgActive]         = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TitleBgCollapsed]      = rgb_(COL_SURFACE_BG, 0.85f);
    st.Colors[ImGuiCol_MenuBarBg]             = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_ScrollbarBg]           = rgb_(COL_WINDOW_BG);
    st.Colors[ImGuiCol_ScrollbarGrab]         = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_ScrollbarGrabHovered]  = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_ScrollbarGrabActive]   = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_CheckMark]             = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_SliderGrab]            = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_SliderGrabActive]      = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_Button]                = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_ButtonHovered]         = rgb_(COL_HOVER);
    st.Colors[ImGuiCol_ButtonActive]          = rgb_(COL_ACCENT_SECOND, 0.55f);
    st.Colors[ImGuiCol_Header]                = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_HeaderHovered]         = rgb_(COL_HOVER);
    st.Colors[ImGuiCol_HeaderActive]          = rgb_(COL_BORDER, 0.85f);
    st.Colors[ImGuiCol_Separator]             = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_SeparatorHovered]      = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_SeparatorActive]       = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_ResizeGrip]            = rgb_(COL_BORDER, 0.35f);
    st.Colors[ImGuiCol_ResizeGripHovered]     = rgb_(COL_TEXT_DIM, 0.55f);
    st.Colors[ImGuiCol_ResizeGripActive]      = rgb_(COL_ACCENT, 0.55f);
    st.Colors[ImGuiCol_Tab]                   = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TabHovered]            = rgb_(COL_HOVER);
    st.Colors[ImGuiCol_TabActive]             = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TabUnfocused]          = rgb_(COL_WINDOW_BG);
    st.Colors[ImGuiCol_TabUnfocusedActive]    = rgb_(COL_SURFACE_BG, 0.85f);
    /* The docking chrome. The central node is left transparent
     * because the scene is drawn behind it, so the node's own
     * background must not paint over the picture. */
    st.Colors[ImGuiCol_DockingPreview]        = rgb_(COL_ACCENT, 0.35f);
    st.Colors[ImGuiCol_DockingEmptyBg]        = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_PlotLines]             = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_PlotLinesHovered]      = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_PlotHistogram]         = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_PlotHistogramHovered]  = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_TableHeaderBg]         = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TableBorderStrong]     = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_TableBorderLight]      = rgb_(COL_BORDER, 0.55f);
    st.Colors[ImGuiCol_TableRowBg]            = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_TableRowBgAlt]         = rgb_(COL_SURFACE_BG, 0.35f);
    st.Colors[ImGuiCol_TextSelectedBg]        = rgb_(COL_ACCENT_SECOND, 0.35f);
    st.Colors[ImGuiCol_DragDropTarget]        = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_NavHighlight]          = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_NavWindowingHighlight] = rgb_(COL_ACCENT, 0.55f);
    st.Colors[ImGuiCol_NavWindowingDimBg]     = rgb_(COL_WINDOW_BG, 0.55f);
    st.Colors[ImGuiCol_ModalWindowDimBg]      = rgb_(COL_WINDOW_BG, 0.55f);
}

void glfw_error_(int code, const char *desc)
{
    fprintf(stderr, "k26rl_view: glfw error %d: %s\n", code, desc);
}

const char *flag_marks_(uint32_t f, char *buf, size_t n)
{
    buf[0] = '\0';
    if (f & K26RL_FLAG_TERMINATED)
        snprintf(buf + strlen(buf), n - strlen(buf), "terminated ");
    if (f & K26RL_FLAG_TRUNCATED)
        snprintf(buf + strlen(buf), n - strlen(buf), "truncated ");
    if (f & K26RL_FLAG_FAULT)
        snprintf(buf + strlen(buf), n - strlen(buf), "fault ");
    if (f & K26RL_FLAG_RESET_BOUNDARY)
        snprintf(buf + strlen(buf), n - strlen(buf), "reset-boundary ");
    if (!buf[0])
        snprintf(buf, n, "-");
    return buf;
}

/* The ending's colour, so the three the format distinguishes are
 * distinguished here too rather than all reading as "over". */
ImVec4 ending_colour_(uint16_t reason)
{
    switch (reason) {
    case K26RL_END_TERMINATED: return rgb_(COL_TEXT);
    case K26RL_END_TRUNCATED:  return rgb_(COL_ACCENT);
    case K26RL_END_FAULT:      return rgb_(COL_ERROR);
    default:                   return rgb_(COL_TEXT_DIM);
    }
}

void panel_timeline_(Ui &ui, const Episode &ep)
{
    char marks[128];
    ImGui::Begin("Timeline");

    if (ui.model->live())
        ImGui::Text("ring %s", ui.model->info().tap.c_str());
    else
        ImGui::Text("file %s", ui.model->info().path.c_str());
    if (!ui.model->live() && !ui.model->info().clean_close) {
        ImGui::TextColored(rgb_(COL_ACCENT),
                           "unclean close: readable prefix ends at byte %llu "
                           "of %llu",
                           (unsigned long long)ui.model->info().readable_bytes,
                           (unsigned long long)ui.model->info().file_bytes);
    }

    int ep_idx = (int)ui.episode_index;
    if (ImGui::SliderInt("episode", &ep_idx, 0,
                         (int)ui.model->info().episode_count - 1)) {
        ui.episode_index = (uint32_t)ep_idx;
        ui.step = 0;
        ui.resim_done = false;
    }
    int st = (int)ui.step;
    if (ImGui::SliderInt("step", &st, 0,
                         ep.step_count ? (int)ep.step_count - 1 : 0)) {
        ui.step = (uint32_t)st;
        /* Taking hold of the timeline is a person saying where they
         * want to be, so the view stops riding the newest step. */
        ui.follow = false;
    }
    /* What this episode is missing, said where the timeline is read.
     * A step's position in what arrived is not its step number once
     * something did not arrive, so the number is stated too. */
    if (!ep.gaps.empty()) {
        uint32_t missing = 0;
        for (size_t g = 0; g < ep.gaps.size(); g++)
            missing += ep.gaps[g].count;
        ImGui::TextColored(rgb_(COL_ACCENT),
                           "%u step records were overwritten before this "
                           "window read them, in %u run%s. The track "
                           "skips them",
                           (unsigned)missing, (unsigned)ep.gaps.size(),
                           ep.gaps.size() == 1 ? "" : "s");
        for (size_t g = 0; g < ep.gaps.size() && g < 8; g++) {
            ImGui::TextColored(rgb_(COL_ACCENT),
                               "    missing steps %u to %u",
                               ep.gaps[g].first,
                               ep.gaps[g].first + ep.gaps[g].count - 1);
        }
    }
    if (!ep.start_seen) {
        ImGui::TextColored(rgb_(COL_ACCENT),
                           "the opening record never arrived, so the "
                           "initial observation and the randomisation "
                           "draws are not known");
    }
    if (ui.model->live())
        ImGui::Text("step %u of this episode", ep.step_at(ui.step));
    if (!ep.complete) {
        ImGui::TextColored(rgb_(COL_TEXT_DIM),
                           "this episode is still running");
    } else {
        ImGui::TextColored(ending_colour_(ep.end_reason),
                           "ends: %s", end_reason_name(ep.end_reason));
    }
    if (ep.complete && ep.end_reason == K26RL_END_FAULT) {
        ImGui::SameLine();
        ImGui::TextColored(ending_colour_(ep.end_reason),
                           "reason code %u; the final record is not a "
                           "transition",
                           (unsigned)ep.fault_code);
    }
    if (ImGui::Button("jump to ending"))
        ui.step = ep.step_count ? ep.step_count - 1 : 0;
    ImGui::SameLine();
    if (ImGui::Button("jump to start"))
        ui.step = 0;

    ImGui::Text("identity (ordinal %u, env %u, episode %u), %u steps, "
                "%u transitions", ep.ordinal, ep.env, ep.episode,
                ep.step_count, ep.transitions());
    if (ui.step < ep.flags.size()) {
        ImGui::Text("step %u flags: %s", ui.step,
                    flag_marks_(ep.flags[ui.step], marks, sizeof marks));
    }
    ImGui::End();
}

/* The per-agent grouping the observation, action and reward panels
 * take when a recording carries more than one agent.
 *
 * Which channels and which action offsets belong to an agent is the
 * model's answer, read from the spec's slice tags, and this file asks
 * for it rather than working it out: the channel names carry an agent
 * prefix, grouping by that prefix would be easy, and it would put a
 * channel under the wrong agent the first time two agents declare one
 * name. The header shows the name those qualified names publish.
 */
std::string agent_header_(const AgentGroup &g)
{
    char buf[64];
    if (!g.name.empty())
        return g.name;
    snprintf(buf, sizeof buf, "agent %u", g.index);
    return buf;
}

/* One agent's slice, as a reading: where it starts and how wide it
 * is, on each vector. */
void agent_slices_(const AgentGroup &g)
{
    if (g.has_obs_slice)
        ImGui::Text("obs %u+%u", g.obs_offset, g.obs_count);
    else
        ImGui::TextUnformatted("obs slice not published");
    ImGui::SameLine();
    if (g.has_act_slice)
        ImGui::Text("act %u+%u", g.act_offset, g.act_count);
    else
        ImGui::TextUnformatted("act slice not published");
}

/* One agent's reward and return, at that agent's own index into the
 * per-step reward array. */
void plot_reward_(Ui &ui, const Episode &ep, const char *title,
                  uint32_t agent)
{
    const uint32_t agents = ui.model->spec().agent_count
                            ? ui.model->spec().agent_count : 1;
    std::vector<double> ret = Model::returns(ep, agents);
    std::vector<double> xs(ep.step_count);
    std::vector<double> ys(ep.step_count);

    for (uint32_t i = 0; i < ep.step_count; i++)
        xs[i] = (double)i;
    if (!ImPlot::BeginPlot(title, ImVec2(-1, 200)))
        return;
    ImPlot::SetupAxes("step", "reward");
    for (uint32_t i = 0; i < ep.step_count; i++) {
        size_t k = (size_t)i * agents + agent;
        ys[i] = k < ep.rewards.size() ? ep.rewards[k] : 0.0;
    }
    if (ep.step_count)
        ImPlot::PlotLine("reward", &xs[0], &ys[0], (int)ep.step_count);
    for (uint32_t i = 0; i < ep.step_count; i++) {
        size_t k = (size_t)i * agents + agent;
        ys[i] = k < ret.size() ? ret[k] : 0.0;
    }
    if (ep.step_count)
        ImPlot::PlotLine("return", &xs[0], &ys[0], (int)ep.step_count);
    ImPlot::EndPlot();
}

void panel_reward_(Ui &ui, const Episode &ep)
{
    const uint32_t agents = ui.model->spec().agent_count
                            ? ui.model->spec().agent_count : 1;
    std::vector<AgentGroup> g = ui.model->agent_groups();

    ImGui::Begin("Reward");
    if (g.size() > 1) {
        for (size_t a = 0; a < g.size(); a++) {
            std::string h = agent_header_(g[a]);
            ImGui::PushID((int)a);
            if (ImGui::CollapsingHeader(h.c_str(),
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                plot_reward_(ui, ep, "##reward", g[a].index);
                if (g[a].index < ep.terminal_adjustments.size())
                    ImGui::Text("terminal adjustment %g",
                                ep.terminal_adjustments[g[a].index]);
            }
            ImGui::PopID();
        }
        ImGui::End();
        return;
    }
    plot_reward_(ui, ep, "reward and return", 0);
    for (uint32_t a = 0; a < agents && a < ep.terminal_adjustments.size(); a++)
        ImGui::Text("terminal adjustment %g", ep.terminal_adjustments[a]);
    ImGui::End();
}

/* The traces of a set of channels, given as spec positions so the
 * per-channel toggles keep working inside a group. */
void plot_channels_(Ui &ui, const Episode &ep, const char *title,
                    const std::vector<size_t> &pos)
{
    const Spec &sp = ui.model->spec();
    std::vector<double> xs(ep.step_count), ys(ep.step_count);

    for (uint32_t i = 0; i < ep.step_count; i++)
        xs[i] = (double)i;
    for (size_t q = 0; q < pos.size(); q++) {
        size_t c = pos[q];
        bool on = ui.channel_on[c] != 0;
        if (ImGui::Checkbox(sp.channels[c].name.c_str(), &on))
            ui.channel_on[c] = on ? 1 : 0;
        if ((q % 3) != 2 && q + 1 < pos.size())
            ImGui::SameLine();
    }
    if (!ImPlot::BeginPlot(title, ImVec2(-1, 260)))
        return;
    ImPlot::SetupAxes("step", "value");
    for (size_t q = 0; q < pos.size(); q++) {
        size_t c = pos[q];
        uint32_t idx = sp.channels[c].index;
        if (!ui.channel_on[c])
            continue;
        for (uint32_t i = 0; i < ep.step_count; i++) {
            size_t k = (size_t)i * sp.obs_total + idx;
            ys[i] = k < ep.obs.size() ? ep.obs[k] : 0.0;
        }
        if (ep.step_count)
            ImPlot::PlotLine(sp.channels[c].name.c_str(), &xs[0], &ys[0],
                             (int)ep.step_count);
        /* A measured channel is drawn with the truth beside it when
         * the file says it has one. Which channels those are is the
         * model's own answer, the same one the headless dump gets,
         * rather than a predicate written again here: a window
         * overlaying a different pair from the one the gate checks is
         * a window nothing checks. */
        {
            uint32_t t = ui.model->truth_pair_of(idx);
            const char *tname = "truth";
            if (t == K26RL_OBS_PAIR_NONE)
                continue;
            for (size_t d = 0; d < sp.channels.size(); d++) {
                if (sp.channels[d].index == t)
                    tname = sp.channels[d].name.c_str();
            }
            for (uint32_t i = 0; i < ep.step_count; i++) {
                size_t k = (size_t)i * sp.obs_total + t;
                ys[i] = k < ep.obs.size() ? ep.obs[k] : 0.0;
            }
            if (ep.step_count)
                ImPlot::PlotLine(tname, &xs[0], &ys[0], (int)ep.step_count);
        }
    }
    ImPlot::EndPlot();
}

void panel_obs_(Ui &ui, const Episode &ep)
{
    const Spec &sp = ui.model->spec();
    std::vector<AgentGroup> g = ui.model->agent_groups();
    std::vector<size_t> pos;

    ImGui::Begin("Observations");
    if (g.size() > 1) {
        for (size_t a = 0; a < g.size(); a++) {
            std::string h = agent_header_(g[a]);
            ImGui::PushID((int)a);
            if (ImGui::CollapsingHeader(h.c_str(),
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                agent_slices_(g[a]);
                pos.clear();
                for (size_t c = 0; c < sp.channels.size(); c++) {
                    for (size_t q = 0; q < g[a].channels.size(); q++) {
                        if (sp.channels[c].index == g[a].channels[q])
                            pos.push_back(c);
                    }
                }
                plot_channels_(ui, ep, "##obs", pos);
            }
            ImGui::PopID();
        }
        ImGui::End();
        return;
    }
    for (size_t c = 0; c < sp.channels.size(); c++)
        pos.push_back(c);
    plot_channels_(ui, ep, "observation channels", pos);
    ImGui::End();
}

/* The action traces of one contiguous run of offsets, with each
 * channel's declared bounds drawn as the limits they are, so a trace
 * riding its bound is visible as such. */
void plot_actions_(Ui &ui, const Episode &ep, const char *title,
                   uint32_t off, uint32_t count)
{
    const Spec &sp = ui.model->spec();
    std::vector<double> xs(ep.step_count), ys(ep.step_count);

    for (uint32_t i = 0; i < ep.step_count; i++)
        xs[i] = (double)i;
    if (ImPlot::BeginPlot(title, ImVec2(-1, 260))) {
        ImPlot::SetupAxes("step", "value");
        for (uint32_t j = off; j < off + count && j < sp.act_total; j++) {
            char label[48];
            const ActionDecl *decl = 0;
            for (size_t k = 0; k < sp.actions.size(); k++) {
                if (sp.actions[k].offset == j)
                    decl = &sp.actions[k];
            }
            snprintf(label, sizeof label, "action %u%s", j,
                     (decl && decl->has_kind &&
                      decl->kind == K26RL_ACT_KIND_DISCRETE)
                     ? " (discrete)" : "");
            for (uint32_t i = 0; i < ep.step_count; i++) {
                size_t k = (size_t)i * sp.act_total + j;
                ys[i] = k < ep.act.size() ? ep.act[k] : 0.0;
            }
            if (ep.step_count)
                ImPlot::PlotLine(label, &xs[0], &ys[0], (int)ep.step_count);
            if (decl && decl->has_bounds && ep.step_count) {
                double bx[2] = { 0.0, (double)(ep.step_count - 1) };
                double lo[2] = { decl->lo, decl->lo };
                double hi[2] = { decl->hi, decl->hi };
                char blabel[64];
                snprintf(blabel, sizeof blabel, "action %u bounds", j);
                ImPlot::SetNextLineStyle(rgb_(COL_TEXT_DIM, 0.8f));
                ImPlot::PlotLine(blabel, bx, lo, 2);
                ImPlot::SetNextLineStyle(rgb_(COL_TEXT_DIM, 0.8f));
                ImPlot::PlotLine(blabel, bx, hi, 2);
            }
        }
        ImPlot::EndPlot();
    }
    for (size_t k = 0; k < sp.actions.size(); k++) {
        const ActionDecl &a = sp.actions[k];
        if (a.offset < off || a.offset >= off + count)
            continue;
        if (a.has_kind && a.kind == K26RL_ACT_KIND_DISCRETE)
            ImGui::Text("action %u discrete, arity %u", a.offset, a.arity);
        else if (a.has_bounds)
            ImGui::Text("action %u box [%g, %g]", a.offset, a.lo, a.hi);
    }
}

void panel_action_(Ui &ui, const Episode &ep)
{
    const Spec &sp = ui.model->spec();
    std::vector<AgentGroup> g = ui.model->agent_groups();

    ImGui::Begin("Actions");
    if (g.size() > 1) {
        for (size_t a = 0; a < g.size(); a++) {
            std::string h = agent_header_(g[a]);
            ImGui::PushID((int)a);
            if (ImGui::CollapsingHeader(h.c_str(),
                                        ImGuiTreeNodeFlags_DefaultOpen)) {
                if (g[a].act_count)
                    plot_actions_(ui, ep, "##act", g[a].act_offset,
                                  g[a].act_count);
                else
                    ImGui::TextUnformatted("no action channel");
            }
            ImGui::PopID();
        }
        ImGui::End();
        return;
    }
    plot_actions_(ui, ep, "action channels", 0, sp.act_total);
    ImGui::End();
}

/* The trajectory view. Points are reconstructed from the recorded
 * direction and range channels and projected here with the shared 3D
 * maths library; there is no scene, no body, and no claim to a world
 * frame, because the file records observation channels and not world
 * states. */
void panel_traj_(Ui &ui, const Episode &ep)
{
    const std::vector<Trajectory> &tr = ui.model->trajectories();

    ImGui::Begin("Trajectory");
    ImGui::TextWrapped("%s", TRAJECTORY_LABEL);
    if (!tr.empty() && ui.traj_pick < (int)tr.size()) {
        const std::vector<Channel> &ch = ui.model->spec().channels;
        for (size_t i = 0; i < ch.size(); i++) {
            if (ch[i].index != tr[(size_t)ui.traj_pick].dir_x)
                continue;
            ImGui::Text("mode: %s", ch[i].has_mode
                        ? observer_mode_name(ch[i].mode)
                        : "not published by this file");
        }
    }
    if (tr.empty()) {
        ImGui::TextUnformatted(
            "no channel set matches the drawable naming convention. "
            "See the observation panel");
        ImGui::End();
        return;
    }
    if (ui.traj_pick >= (int)tr.size())
        ui.traj_pick = 0;
    for (size_t t = 0; t < tr.size(); t++) {
        if (ImGui::RadioButton(tr[t].base.c_str(), ui.traj_pick == (int)t))
            ui.traj_pick = (int)t;
        if (t + 1 < tr.size())
            ImGui::SameLine();
    }
    ImGui::SliderFloat("yaw", &ui.yaw, -3.14159f, 3.14159f);
    ImGui::SliderFloat("pitch", &ui.pitch, -1.5f, 1.5f);

    {
        const Trajectory &t = tr[(size_t)ui.traj_pick];
        std::vector<double> px(ep.step_count), py(ep.step_count);
        K26M4 rot_y, rot_x, view;
        double scale = 0.0;

        k26m3d_mat4_rotate_y(&rot_y, (double)ui.yaw);
        k26m3d_mat4_rotate_x(&rot_x, (double)ui.pitch);
        k26m3d_mat4_mul(&view, &rot_x, &rot_y);

        for (uint32_t i = 0; i < ep.step_count; i++) {
            double xyz[3];
            K26V4 p, q;
            Model::point(ep, ui.model->spec(), t, i, xyz);
            p = k26m3d_v4(xyz[0], xyz[1], xyz[2], 1.0);
            q = k26m3d_mat4_mul_v4(&view, p);
            px[i] = q.x;
            py[i] = q.y;
            if (fabs(q.x) > scale) scale = fabs(q.x);
            if (fabs(q.y) > scale) scale = fabs(q.y);
        }
        if (ImPlot::BeginPlot("observer-relative track", ImVec2(-1, 320),
                              ImPlotFlags_Equal)) {
            ImPlot::SetupAxes("x (m)", "y (m)");
            if (ep.step_count)
                ImPlot::PlotLine(t.base.c_str(), &px[0], &py[0],
                                 (int)ep.step_count);
            /* The observer sits at the origin of this reconstruction
             * by construction; marking it says so. */
            {
                double ox[1] = { 0.0 }, oy[1] = { 0.0 };
                ImPlot::PlotScatter("observer", ox, oy, 1);
            }
            if (ui.step < ep.step_count) {
                double cx[1] = { px[ui.step] }, cy[1] = { py[ui.step] };
                ImPlot::PlotScatter("current step", cx, cy, 1);
            }
            ImPlot::EndPlot();
        }
        if (ui.step < ep.step_count) {
            double xyz[3];
            Model::point(ep, ui.model->spec(), t, ui.step, xyz);
            ImGui::Text("step %u: (%.6g, %.6g, %.6g) m", ui.step, xyz[0],
                        xyz[1], xyz[2]);
        }
    }
    ImGui::End();
}

/* The world frame, which the file alone cannot give: bodies come from
 * the artifact's body getter as the rebuild runs, so this panel
 * appears only once a re-simulation has been performed. */
void panel_world_(Ui &ui, const Episode &ep)
{
    (void)ep;
    ImGui::Begin("World frame");
    if (ui.artifact.empty()) {
        ImGui::TextWrapped("no artifact: the file records observation "
                           "channels, not world states");
        ImGui::End();
        return;
    }
    if (!ui.resim_done || !ui.resim.ran || !ui.resim.has_bodies) {
        ImGui::TextWrapped("run the reconstruction in the re-simulation "
                           "panel");
        ImGui::End();
        return;
    }
    {
        const std::vector<std::string> &names = ui.model->spec().body_names;
        std::vector<double> px(ui.resim.steps_compared);
        std::vector<double> py(ui.resim.steps_compared);
        if (ImPlot::BeginPlot("bodies, world frame", ImVec2(-1, 320),
                              ImPlotFlags_Equal)) {
            ImPlot::SetupAxes("x (m)", "y (m)");
            for (uint32_t b = 0; b < ui.resim.body_count; b++) {
                char label[64];
                snprintf(label, sizeof label, "%s",
                         b < names.size() ? names[b].c_str() : "body");
                for (uint32_t i = 0; i < ui.resim.steps_compared; i++) {
                    size_t base = ((size_t)i * ui.resim.body_count + b) * 6;
                    px[i] = base + 6 <= ui.resim.bodies.size()
                            ? ui.resim.bodies[base] : 0.0;
                    py[i] = base + 6 <= ui.resim.bodies.size()
                            ? ui.resim.bodies[base + 1] : 0.0;
                }
                if (ui.resim.steps_compared)
                    ImPlot::PlotLine(label, &px[0], &py[0],
                                     (int)ui.resim.steps_compared);
            }
            ImPlot::EndPlot();
        }
        ImGui::Text("%u bodies over %u reconstructed steps",
                    ui.resim.body_count, ui.resim.steps_compared);
    }
    ImGui::End();
}

/* Orientation and body rate, which the file records only if the
 * programme declared an attitude observe, and which the artifact's
 * getter gives for every body whatever the programme declared. Like
 * the world frame this panel exists once a rebuild has run. */
void panel_attitude_(Ui &ui, const Episode &ep)
{
    (void)ep;
    ImGui::Begin("Attitude");
    if (ui.artifact.empty()) {
        ImGui::TextWrapped("no artifact: a body's orientation is an "
                           "observation channel only when the programme "
                           "declares one");
        ImGui::End();
        return;
    }
    if (!ui.resim_done || !ui.resim.ran || !ui.resim.has_attitudes) {
        ImGui::TextWrapped("run the re-simulation panel's reconstruction to "
                           "populate this view");
        ImGui::End();
        return;
    }
    {
        const std::vector<std::string> &names = ui.model->spec().body_names;
        uint32_t n = ui.resim.steps_compared;
        std::vector<double> xs(n), ys(n);
        static const char *const comp[7] = { "w", "x", "y", "z",
                                             "wx", "wy", "wz" };
        for (uint32_t i = 0; i < n; i++)
            xs[i] = (double)i;
        for (uint32_t b = 0; b < ui.resim.body_count; b++) {
            char title[80];
            snprintf(title, sizeof title, "%s: quaternion and body rate",
                     b < names.size() ? names[b].c_str() : "body");
            if (ImPlot::BeginPlot(title, ImVec2(-1, 200))) {
                ImPlot::SetupAxes("step", "value");
                for (int c = 0; c < 7; c++) {
                    for (uint32_t i = 0; i < n; i++) {
                        size_t base = ((size_t)i * ui.resim.body_count + b) * 7;
                        ys[i] = base + 7 <= ui.resim.attitudes.size()
                                ? ui.resim.attitudes[base + c] : 0.0;
                    }
                    if (n)
                        ImPlot::PlotLine(comp[c], &xs[0], &ys[0], (int)n);
                }
                ImPlot::EndPlot();
            }
        }
    }
    ImGui::End();
}

/* The craft's own wireframe, drawn only when the asset on disk
 * digests to what the recording says produced it. A mismatch is
 * reported and nothing is drawn: a wireframe beside a recording is a
 * claim about what flew.
 *
 * One plot per bound assembly, because a recording of two craft binds
 * two and each is a different shape. */
void panel_wireframe_(Ui &ui, const Episode &ep)
{
    (void)ep;
    const Spec &sp = ui.model->spec();

    ImGui::Begin("Wireframe");
    if (ui.asset_reqs.empty()) {
        ImGui::TextWrapped("no assembly supplied. The recording carries "
                           "the digest and not the geometry; run with "
                           "--asset PATH");
        ImGui::End();
        return;
    }
    if (!ui.asset_tried) {
        ui.assets = asset_bind(sp, ui.asset_reqs);
        ui.asset_tried = true;
    }
    for (size_t k = 0; k < ui.assets.size(); k++) {
        const Asset &as = ui.assets[k].asset;
        const AssemblyRef *match = 0;
        char title[128];

        ImGui::PushID((int)k);
        if (!as.loaded) {
            ImGui::TextWrapped("%s", as.error.c_str());
            ImGui::PopID();
            continue;
        }
        /* The same binding function the headless dump calls, so the
         * window cannot draw a craft the dump would refuse. This
         * panel's one safety property is that a craft whose bytes are
         * not the recorded bytes is never drawn, and a property with
         * two implementations holds in whichever was last looked
         * at. */
        if (ui.assets[k].request.body == ASSET_UNBOUND)
            asset_verdict(sp, as, &match);
        else
            asset_verdict_at(sp, as, ui.assets[k].request.body, &match);
        if (ui.assets[k].verdict == ASSET_NO_BODY) {
            ImGui::TextWrapped("%s: no body binds an assembly of that name",
                               as.name.c_str());
            ImGui::PopID();
            continue;
        }
        if (ui.assets[k].verdict != ASSET_DRAWABLE) {
            ImGui::TextWrapped("%s: not the assembly that flew. Recorded "
                               "digest %s, file digest %s",
                               as.name.c_str(),
                               ui.assets[k].verdict == ASSET_NO_DIGEST
                                   ? "none"
                                   : digest_hex(match->digest).c_str(),
                               digest_hex(as.digest).c_str());
            ImGui::PopID();
            continue;
        }
        ImGui::Text("%s on %s, %u vertices, %u edges, %u triangles",
                    as.name.c_str(),
                    scene_body_name(sp, ui.assets[k].body).c_str(),
                    as.mesh_vertices, (unsigned)as.edges.size(),
                    as.mesh_triangles);
        snprintf(title, sizeof title, "%s, body frame", as.name.c_str());
        if (ImPlot::BeginPlot(title, ImVec2(-1, 320), ImPlotFlags_Equal)) {
            ImPlot::SetupAxes("x (m)", "y (m)");
            for (size_t e = 0; e < as.edges.size(); e++) {
                double lx[2], ly[2];
                size_t a = (size_t)as.edges[e].a * 3;
                size_t b = (size_t)as.edges[e].b * 3;
                if (a + 2 >= as.vertices.size() ||
                    b + 2 >= as.vertices.size())
                    continue;
                lx[0] = as.vertices[a];
                ly[0] = as.vertices[a + 1];
                lx[1] = as.vertices[b];
                ly[1] = as.vertices[b + 1];
                ImPlot::PlotLine("##edge", lx, ly, 2);
            }
            ImPlot::EndPlot();
        }
        ImGui::PopID();
    }
    ImGui::End();
}

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

const char *const SCENE_VERT =
    "#version 330 core\n"
    "layout(location = 0) in vec3 a_pos;\n"
    "layout(location = 1) in float a_shade;\n"
    "uniform mat4 u_mvp;\n"
    "out float v_shade;\n"
    "void main()\n"
    "{\n"
    "    v_shade = a_shade;\n"
    "    gl_Position = u_mvp * vec4(a_pos, 1.0);\n"
    "}\n";

const char *const SCENE_FRAG =
    "#version 330 core\n"
    "in float v_shade;\n"
    "uniform vec4 u_colour;\n"
    "out vec4 frag;\n"
    "void main()\n"
    "{\n"
    "    frag = vec4(u_colour.rgb * v_shade, u_colour.a);\n"
    "}\n";

bool gl_load_(void *slot, const char *name, std::string *err)
{
    GLFWglproc p = glfwGetProcAddress(name);
    if (!p) {
        *err = std::string("the graphics context does not export ") + name;
        return false;
    }
    memcpy(slot, &p, sizeof p);
    return true;
}

#define GL_LOAD_(g, f) \
    if (!gl_load_(&(g)->f, "gl" #f, &(g)->error)) return false

bool scene_gl_procs_(SceneGl *gl)
{
    GL_LOAD_(gl, CreateShader);
    GL_LOAD_(gl, ShaderSource);
    GL_LOAD_(gl, CompileShader);
    GL_LOAD_(gl, GetShaderiv);
    GL_LOAD_(gl, GetShaderInfoLog);
    GL_LOAD_(gl, DeleteShader);
    GL_LOAD_(gl, CreateProgram);
    GL_LOAD_(gl, AttachShader);
    GL_LOAD_(gl, LinkProgram);
    GL_LOAD_(gl, GetProgramiv);
    GL_LOAD_(gl, GetProgramInfoLog);
    GL_LOAD_(gl, DeleteProgram);
    GL_LOAD_(gl, UseProgram);
    GL_LOAD_(gl, GetUniformLocation);
    GL_LOAD_(gl, UniformMatrix4fv);
    GL_LOAD_(gl, Uniform4f);
    GL_LOAD_(gl, GenVertexArrays);
    GL_LOAD_(gl, BindVertexArray);
    GL_LOAD_(gl, DeleteVertexArrays);
    GL_LOAD_(gl, GenBuffers);
    GL_LOAD_(gl, BindBuffer);
    GL_LOAD_(gl, BufferData);
    GL_LOAD_(gl, DeleteBuffers);
    GL_LOAD_(gl, EnableVertexAttribArray);
    GL_LOAD_(gl, DisableVertexAttribArray);
    GL_LOAD_(gl, VertexAttribPointer);
    GL_LOAD_(gl, VertexAttrib1f);
    return true;
}

GLuint scene_shader_(SceneGl *gl, GLenum kind, const char *src)
{
    GLuint sh = gl->CreateShader(kind);
    GLint ok = 0;

    gl->ShaderSource(sh, 1, &src, NULL);
    gl->CompileShader(sh);
    gl->GetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        GLsizei n = 0;
        gl->GetShaderInfoLog(sh, (GLsizei)sizeof log, &n, log);
        gl->error = std::string("scene shader: ") + log;
        gl->DeleteShader(sh);
        return 0;
    }
    return sh;
}

bool scene_gl_init_(SceneGl *gl)
{
    GLuint vs, fs;
    GLint ok = 0;

    gl->ready = false;
    gl->program = 0;
    gl->vao = gl->vbo = gl->ebo = gl->shade_vbo = 0;
    if (!scene_gl_procs_(gl))
        return false;
    vs = scene_shader_(gl, GL_VERTEX_SHADER, SCENE_VERT);
    if (!vs)
        return false;
    fs = scene_shader_(gl, GL_FRAGMENT_SHADER, SCENE_FRAG);
    if (!fs) {
        gl->DeleteShader(vs);
        return false;
    }
    gl->program = gl->CreateProgram();
    gl->AttachShader(gl->program, vs);
    gl->AttachShader(gl->program, fs);
    gl->LinkProgram(gl->program);
    gl->GetProgramiv(gl->program, GL_LINK_STATUS, &ok);
    gl->DeleteShader(vs);
    gl->DeleteShader(fs);
    if (!ok) {
        char log[512];
        GLsizei n = 0;
        gl->GetProgramInfoLog(gl->program, (GLsizei)sizeof log, &n, log);
        gl->error = std::string("scene program: ") + log;
        gl->DeleteProgram(gl->program);
        gl->program = 0;
        return false;
    }
    gl->u_mvp = gl->GetUniformLocation(gl->program, "u_mvp");
    gl->u_colour = gl->GetUniformLocation(gl->program, "u_colour");
    gl->GenVertexArrays(1, &gl->vao);
    gl->GenBuffers(1, &gl->vbo);
    gl->GenBuffers(1, &gl->ebo);
    gl->GenBuffers(1, &gl->shade_vbo);
    gl->ready = true;
    return true;
}

/* One colour per element kind, and the axes coloured segment by
 * segment. Colour is the one thing about this picture the window does
 * decide, a colour being a presentation choice and not a geometric
 * one. */
void scene_colour_(ElementKind k, size_t segment, float *rgba)
{
    static const float table[ELEM_KIND_COUNT][4] = {
        { 0.85f, 0.88f, 0.95f, 1.0f },   /* wireframe */
        { 0.95f, 0.65f, 0.25f, 1.0f },   /* collider */
        { 0.90f, 0.30f, 0.30f, 1.0f },   /* axes, overridden below */
        { 0.40f, 0.80f, 0.95f, 1.0f },   /* trajectory */
        { 0.55f, 0.95f, 0.55f, 1.0f },   /* velocity */
        { 0.95f, 0.85f, 0.35f, 1.0f },   /* port */
        { 0.95f, 0.45f, 0.85f, 1.0f },   /* thruster */
        { 0.98f, 0.98f, 0.55f, 1.0f }    /* detection line of sight */
    };
    static const float axes[3][4] = {
        { 0.95f, 0.35f, 0.35f, 1.0f },
        { 0.35f, 0.95f, 0.35f, 1.0f },
        { 0.45f, 0.55f, 0.95f, 1.0f }
    };
    const float *src = (k == ELEM_AXES && segment < 3)
                       ? axes[segment] : table[k];
    for (int i = 0; i < 4; i++)
        rgba[i] = src[i];
}

/* The buffer a body-frame element's vertices live in, uploaded on
 * first sight and reused after. Anything already relative to the
 * camera goes to the shared buffer instead, since its contents change
 * with the eye. */
GLuint scene_vertex_buffer_(SceneGl *gl, const SceneElement &e)
{
    std::string key;
    std::map<std::string, GLuint>::iterator it;
    GLuint buf = 0;

    if (!e.local_static)
        return gl->vbo;
    key = std::string(element_name(e.kind)) + "/" + e.name;
    it = gl->cached.find(key);
    if (it != gl->cached.end()) {
        if (gl->cached_data[key] == e.local)
            return it->second;
        buf = it->second;
    } else {
        gl->GenBuffers(1, &buf);
    }
    gl->BindBuffer(GL_ARRAY_BUFFER, buf);
    gl->BufferData(GL_ARRAY_BUFFER,
                   (GLsizeiptr)(e.local.size() * sizeof(float)),
                   e.local.empty() ? NULL : &e.local[0], GL_STATIC_DRAW);
    gl->cached[key] = buf;
    gl->cached_data[key] = e.local;
    return buf;
}

void scene_draw_(SceneGl *gl, const Scene &sc, int fbw, int fbh)
{
    std::vector<GLuint> idx;
    std::vector<float> tri;
    std::vector<float> shade;

    if (!gl->ready || !sc.available)
        return;
    glViewport(0, 0, fbw, fbh);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClear(GL_DEPTH_BUFFER_BIT);
    gl->UseProgram(gl->program);
    gl->BindVertexArray(gl->vao);

    for (size_t i = 0; i < sc.elements.size(); i++) {
        const SceneElement &e = sc.elements[i];
        GLuint buf;
        float rgba[4];

        if (e.local.empty())
            continue;
        buf = scene_vertex_buffer_(gl, e);
        if (!e.local_static) {
            gl->BindBuffer(GL_ARRAY_BUFFER, buf);
            gl->BufferData(GL_ARRAY_BUFFER,
                           (GLsizeiptr)(e.local.size() * sizeof(float)),
                           &e.local[0], GL_DYNAMIC_DRAW);
        }
        gl->UniformMatrix4fv(gl->u_mvp, 1, GL_FALSE, e.mvp);

        /* The shaded pass first, so the wireframe over it reads as an
         * outline rather than being hidden by its own faces. */
        if (!e.faces.empty()) {
            tri.clear();
            shade.clear();
            for (size_t f = 0; f < e.faces.size(); f++) {
                const SceneFace &sf = e.faces[f];
                uint32_t v[3] = { sf.a, sf.b, sf.c };
                if (!sf.drawn)
                    continue;
                for (int c = 0; c < 3; c++) {
                    size_t base = (size_t)v[c] * 3;
                    if (base + 2 >= e.local.size())
                        continue;
                    tri.push_back(e.local[base]);
                    tri.push_back(e.local[base + 1]);
                    tri.push_back(e.local[base + 2]);
                    shade.push_back(sf.intensity);
                }
            }
            if (!tri.empty()) {
                scene_colour_(e.kind, 0, rgba);
                gl->Uniform4f(gl->u_colour, rgba[0], rgba[1], rgba[2],
                              rgba[3]);
                gl->BindBuffer(GL_ARRAY_BUFFER, gl->vbo);
                gl->BufferData(GL_ARRAY_BUFFER,
                               (GLsizeiptr)(tri.size() * sizeof(float)),
                               &tri[0], GL_DYNAMIC_DRAW);
                gl->EnableVertexAttribArray(0);
                gl->VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                                        3 * sizeof(float), (void *)0);
                gl->BindBuffer(GL_ARRAY_BUFFER, gl->shade_vbo);
                gl->BufferData(GL_ARRAY_BUFFER,
                               (GLsizeiptr)(shade.size() * sizeof(float)),
                               &shade[0], GL_DYNAMIC_DRAW);
                gl->EnableVertexAttribArray(1);
                gl->VertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE,
                                        sizeof(float), (void *)0);
                glDrawArrays(GL_TRIANGLES, 0, (GLsizei)(tri.size() / 3));
                gl->DisableVertexAttribArray(1);
            }
        }

        /* The index buffer is rebuilt every frame and the vertex
         * buffer is not, because the two change for different
         * reasons: a mesh's coordinates are fixed in its own frame,
         * while which of its segments survive the model's cull
         * changes whenever the camera does. */
        gl->BindBuffer(GL_ARRAY_BUFFER, buf);
        gl->EnableVertexAttribArray(0);
        gl->VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                                (void *)0);
        gl->DisableVertexAttribArray(1);
        gl->VertexAttrib1f(1, 1.0f);
        if (e.kind == ELEM_AXES) {
            for (size_t sgi = 0; sgi < e.segments.size(); sgi++) {
                if (!e.segments[sgi].drawn)
                    continue;
                idx.clear();
                idx.push_back(e.segments[sgi].a);
                idx.push_back(e.segments[sgi].b);
                scene_colour_(e.kind, sgi, rgba);
                gl->Uniform4f(gl->u_colour, rgba[0], rgba[1], rgba[2],
                              rgba[3]);
                gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl->ebo);
                gl->BufferData(GL_ELEMENT_ARRAY_BUFFER,
                               (GLsizeiptr)(idx.size() * sizeof(GLuint)),
                               &idx[0], GL_DYNAMIC_DRAW);
                glDrawElements(GL_LINES, (GLsizei)idx.size(),
                               GL_UNSIGNED_INT, (void *)0);
            }
            continue;
        }
        idx.clear();
        for (size_t sgi = 0; sgi < e.segments.size(); sgi++) {
            if (!e.segments[sgi].drawn)
                continue;
            idx.push_back(e.segments[sgi].a);
            idx.push_back(e.segments[sgi].b);
        }
        if (idx.empty())
            continue;
        scene_colour_(e.kind, 0, rgba);
        gl->Uniform4f(gl->u_colour, rgba[0], rgba[1], rgba[2], rgba[3]);
        gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, gl->ebo);
        gl->BufferData(GL_ELEMENT_ARRAY_BUFFER,
                       (GLsizeiptr)(idx.size() * sizeof(GLuint)), &idx[0],
                       GL_DYNAMIC_DRAW);
        glDrawElements(GL_LINES, (GLsizei)idx.size(), GL_UNSIGNED_INT,
                       (void *)0);
    }
    gl->BindVertexArray(0);
    gl->UseProgram(0);
    glDisable(GL_DEPTH_TEST);
}

void scene_gl_free_(SceneGl *gl)
{
    std::map<std::string, GLuint>::iterator it;

    if (!gl->ready)
        return;
    for (it = gl->cached.begin(); it != gl->cached.end(); ++it)
        gl->DeleteBuffers(1, &it->second);
    gl->DeleteBuffers(1, &gl->vbo);
    gl->DeleteBuffers(1, &gl->ebo);
    gl->DeleteBuffers(1, &gl->shade_vbo);
    gl->DeleteVertexArrays(1, &gl->vao);
    gl->DeleteProgram(gl->program);
    gl->ready = false;
}

/* The scene panel: the controls that set the view, and what the model
 * made of them. It computes nothing about the picture; scene_build
 * does that, and the same call with the same settings is what
 * `--dump scene` writes. */
void panel_scene_(Ui &ui, const Episode &ep, SceneGl &gl)
{
    const Spec &sp = ui.model->spec();
    SceneOptions &o = ui.scene_opt;
    SceneInput in;

    ui.scene_ready = false;
    ImGui::Begin("Scene");
    if (!gl.ready) {
        ImGui::TextWrapped("the scene view is unavailable: %s",
                           gl.error.c_str());
        ImGui::End();
        return;
    }
    if (ui.artifact.empty()) {
        ImGui::TextWrapped("no artifact: the recording carries observation "
                           "channels, not body position or attitude");
    } else if (!ui.scene_resim_done || ui.scene_resim_ref != o.frame ||
               ui.scene_resim_ep != ui.episode_index) {
        if (ImGui::Button("reconstruct body poses in this frame")) {
            ui.scene_resim = resimulate(*ui.model, ep, ui.artifact, o.frame);
            ui.scene_resim_done = true;
            ui.scene_resim_ref = o.frame;
            ui.scene_resim_ep = ui.episode_index;
        }
    }

    /* The reference frame, which is the body getter's own reference:
     * the scene shows the frame the getter was asked for. */
    {
        std::string cur = scene_body_name(sp, o.frame);
        if (ImGui::BeginCombo("reference frame", cur.c_str())) {
            if (ImGui::Selectable("origin", o.frame == SCENE_ORIGIN))
                o.frame = SCENE_ORIGIN;
            for (size_t b = 0; b < sp.body_names.size(); b++) {
                if (ImGui::Selectable(sp.body_names[b].c_str(),
                                      o.frame == b))
                    o.frame = (uint32_t)b;
            }
            ImGui::EndCombo();
        }
    }
    {
        int mode = (int)o.camera.mode;
        if (ImGui::Combo("camera", &mode, "orbit\0chase\0free\0"))
            o.camera.mode = (CameraMode)mode;
        std::string cur = scene_body_name(sp, o.camera.target);
        if (ImGui::BeginCombo("camera target", cur.c_str())) {
            if (ImGui::Selectable("origin", o.camera.target == SCENE_ORIGIN))
                o.camera.target = SCENE_ORIGIN;
            for (size_t b = 0; b < sp.body_names.size(); b++) {
                if (ImGui::Selectable(sp.body_names[b].c_str(),
                                      o.camera.target == b))
                    o.camera.target = (uint32_t)b;
            }
            ImGui::EndCombo();
        }
    }
    if (o.camera.mode == CAMERA_ORBIT) {
        float az = (float)o.camera.azimuth_deg;
        float el = (float)o.camera.elevation_deg;
        float r = (float)o.camera.radius;
        if (ImGui::SliderFloat("azimuth (deg)", &az, -180.0f, 180.0f))
            o.camera.azimuth_deg = az;
        if (ImGui::SliderFloat("elevation (deg)", &el, -89.0f, 89.0f))
            o.camera.elevation_deg = el;
        if (ImGui::DragFloat("radius (m)", &r, r * 0.01f + 0.01f, 0.01f,
                             1.0e9f, "%.3f"))
            o.camera.radius = r;
    } else if (o.camera.mode == CAMERA_CHASE) {
        float off[3] = { (float)o.camera.chase[0], (float)o.camera.chase[1],
                         (float)o.camera.chase[2] };
        if (ImGui::DragFloat3("offset in the body frame (m)", off)) {
            for (int c = 0; c < 3; c++)
                o.camera.chase[c] = off[c];
        }
    } else {
        float eye[3] = { (float)o.camera.eye[0], (float)o.camera.eye[1],
                         (float)o.camera.eye[2] };
        float look[3] = { (float)o.camera.look[0], (float)o.camera.look[1],
                          (float)o.camera.look[2] };
        if (ImGui::DragFloat3("eye (m)", eye)) {
            for (int c = 0; c < 3; c++)
                o.camera.eye[c] = eye[c];
        }
        if (ImGui::DragFloat3("look at (m)", look)) {
            for (int c = 0; c < 3; c++)
                o.camera.look[c] = look[c];
        }
    }
    {
        int proj = (int)o.camera.projection;
        if (ImGui::Combo("projection", &proj,
                         "perspective\0orthographic\0"))
            o.camera.projection = (ProjectionKind)proj;
        if (o.camera.projection == PROJECTION_PERSPECTIVE) {
            float fov = (float)o.camera.fov_y_deg;
            if (ImGui::SliderFloat("field of view (deg)", &fov, 5.0f, 120.0f))
                o.camera.fov_y_deg = fov;
        } else {
            float h = (float)o.camera.ortho_height;
            if (ImGui::DragFloat("view height (m)", &h, h * 0.01f + 0.01f,
                                 0.01f, 1.0e9f, "%.3f"))
                o.camera.ortho_height = h;
        }
    }
    ImGui::Separator();
    for (int k = 0; k < ELEM_KIND_COUNT; k++) {
        bool on = o.enabled[k];
        if (ImGui::Checkbox(element_name((ElementKind)k), &on))
            o.enabled[k] = on;
        if ((k % 3) != 2 && k + 1 < ELEM_KIND_COUNT)
            ImGui::SameLine();
    }
    {
        bool on = o.shading;
        float vs = (float)o.velocity_seconds;
        float al = (float)o.axis_length;
        if (ImGui::Checkbox("shading", &on))
            o.shading = on;
        if (o.shading)
            ImGui::TextWrapped("%s", SHADING_LABEL);
        if (ImGui::DragFloat("velocity vector (s)", &vs, 0.1f, 0.0f, 1.0e6f))
            o.velocity_seconds = vs;
        if (ImGui::DragFloat("axis length (m)", &al, 0.05f, 0.0f, 1.0e6f))
            o.axis_length = al;
    }

    if (!ui.asset_tried && !ui.asset_reqs.empty()) {
        ui.assets = asset_bind(sp, ui.asset_reqs);
        ui.asset_tried = true;
    }
    in.model = ui.model;
    in.episode = &ep;
    if (ui.scene_resim_done && ui.scene_resim_ref == o.frame &&
        ui.scene_resim_ep == ui.episode_index)
        in.resim = &ui.scene_resim;
    /* The same binding function the wireframe panel and the dump
     * call. A craft whose bytes are not the recorded bytes is never
     * drawn here either. */
    for (size_t k = 0; k < ui.assets.size(); k++) {
        if (!ui.assets[k].asset.loaded) {
            ImGui::TextWrapped("%s", ui.assets[k].asset.error.c_str());
            continue;
        }
        if (ui.assets[k].verdict == ASSET_DRAWABLE) {
            AssetBinding b;
            b.asset = &ui.assets[k].asset;
            b.body = ui.assets[k].body;
            in.assets.push_back(b);
            continue;
        }
        ImGui::TextWrapped("`%s` draws nothing: %s",
            ui.assets[k].request.path.c_str(),
            ui.assets[k].verdict == ASSET_NO_BODY
                ? "no body binds an assembly of that name"
            : ui.assets[k].verdict == ASSET_NO_DIGEST
                ? "the recording carries no digest for it"
            : "the bytes on disk are not the bytes that flew");
    }

    ui.scene = scene_build(in, o, ui.step);
    ui.scene_ready = true;
    ImGui::Separator();
    ImGui::TextWrapped("%s", ui.scene.message.c_str());
    ImGui::Text("frame %s, %u elements at step %u",
                ui.scene.frame_name.c_str(),
                (unsigned)ui.scene.elements.size(), ui.step);
    ImGui::Text("eye %.3f %.3f %.3f m", ui.scene.eye[0], ui.scene.eye[1],
                ui.scene.eye[2]);
    for (size_t i = 0; i < ui.scene.elements.size(); i++) {
        const SceneElement &e = ui.scene.elements[i];
        size_t drawn = 0;
        for (size_t sgi = 0; sgi < e.segments.size(); sgi++)
            drawn += e.segments[sgi].drawn ? 1 : 0;
        ImGui::BulletText("%s %s: %u of %u segments drawn",
                          element_name(e.kind), e.name.c_str(),
                          (unsigned)drawn, (unsigned)e.segments.size());
    }
    ImGui::End();
}

/* The live panel: where the frames are coming from, how many arrived,
 * how many did not, and what the run is doing now.
 *
 * The loss line is the point of it. The ring overwrites rather than
 * wait for a watcher, so a window that fell behind a fast simulation
 * lost whole steps, and a debugging instrument that showed a smooth
 * track over them would be lying about the flight. The count is here
 * and the holes are marked on the timeline beside it. */
void panel_live_(Ui &ui)
{
    const Model &m = *ui.model;
    const FileInfo &fi = m.info();
    uint64_t lost = m.frames_lost();

    ImGui::Begin("Live");
    ImGui::Text("ring %s", fi.tap.c_str());
    ImGui::Text("%u slots of %u bytes", m.ring_slot_count(),
                m.ring_slot_size());
    ImGui::Text("frames accepted %llu",
                (unsigned long long)m.frames_accepted());
    if (lost) {
        /* The count is the reading. The line beside it is a limit no
         * reading here states: those steps are gone, and the track
         * does not pass through them. */
        ImGui::Text("frames lost %llu", (unsigned long long)lost);
        ImGui::TextUnformatted("those steps are not held and not drawn");
    } else {
        ImGui::TextUnformatted("frames lost 0");
    }
    ImGui::Text("episodes %u", fi.episode_count);
    ImGui::Text("producer %s", m.producer_closed() ? "closed" : "running");
    ImGui::Checkbox("follow the newest step", &ui.follow);
    ImGui::End();
}

void panel_meta_(Ui &ui, const Episode &ep)
{
    const FileInfo &fi = ui.model->info();
    const Spec &sp = ui.model->spec();

    ImGui::Begin("Run and episode");
    ImGui::Text("governing seed at enable: 0x%016llx (ordinal %u)",
                (unsigned long long)fi.governing_seed, fi.rekey_ordinal);
    ImGui::Text("this episode's governing seed: 0x%016llx",
                (unsigned long long)ep.seed);
    ImGui::Text("environments %u, steps per chunk %u, episodes indexed %u",
                fi.n_envs, fi.steps_per_chunk, fi.episode_count);
    if (fi.unindexed_episode_starts) {
        ImGui::Text("episode starts with no indexed episode: %u",
                    fi.unindexed_episode_starts);
    }
    ImGui::Separator();
    ImGui::Text("agents %u, observation channels %u, action channels %u",
                sp.agent_count, sp.obs_total, sp.act_total);
    ImGui::Text("control dt %g s, horizon %u, episode flags 0x%08x",
                sp.control_dt, sp.horizon, sp.episode_flags);
    ImGui::Text("spec blob %u bytes, endian probe 0x%08x",
                (unsigned)sp.raw.size(), sp.endian_probe);
    for (size_t k = 0; k < sp.unknown_tags.size(); k++) {
        ImGui::Text("spec tag 0x%04x (%u bytes) is not known to this build",
                    (unsigned)sp.unknown_tags[k].first,
                    sp.unknown_tags[k].second);
    }
    ImGui::Separator();
    if (!ep.start_seen) {
        /* Not none: unknown. The record carrying them never arrived,
         * and a count of zero here would read as a fact about the
         * episode rather than about this window. */
        ImGui::TextUnformatted("randomisation draws: not known, the "
                               "episode's opening record never arrived");
    } else {
        ImGui::Text("randomisation draws: %u", (unsigned)ep.dr_tags.size());
        for (size_t k = 0; k < ep.dr_tags.size(); k++) {
            ImGui::Text("  parameter %u = %.17g", ep.dr_tags[k],
                        ep.dr_values[k]);
        }
    }
    ImGui::End();
}

void panel_resim_(Ui &ui, const Episode &ep)
{
    ImGui::Begin("Re-simulation");
    if (ui.artifact.empty()) {
        ImGui::TextWrapped(
            "no artifact. Run with --artifact PATH to rebuild this "
            "episode from its recorded seed and action stream");
        ImGui::End();
        return;
    }
    ImGui::Text("artifact %s", ui.artifact.c_str());
    if (!ui.resim_done) {
        if (ImGui::Button("reconstruct and compare")) {
            ui.resim = resimulate(*ui.model, ep, ui.artifact,
                                  K26RL_BODY_REF_ORIGIN);
            ui.resim_done = true;
        }
        ImGui::End();
        return;
    }

    if (!ui.resim.ran) {
        ImGui::TextColored(rgb_(COL_ERROR), "failed: %s",
                           ui.resim.message.c_str());
    } else if (ui.resim.equal) {
        ImGui::TextColored(rgb_(COL_ACCENT),
                           "equal bitwise over %u steps",
                           ui.resim.steps_compared);
    } else {
        ImGui::TextColored(rgb_(COL_ERROR),
                           "diverged at step %u (%s)",
                           ui.resim.first_divergence,
                           ui.resim.divergence_kind.c_str());
    }
    if (ui.resim.ran) {
        const uint32_t agents = ui.model->spec().agent_count
                                ? ui.model->spec().agent_count : 1;
        std::vector<double> xs(ui.resim.steps_compared);
        std::vector<double> a(ui.resim.steps_compared);
        std::vector<double> b(ui.resim.steps_compared);
        for (uint32_t i = 0; i < ui.resim.steps_compared; i++) {
            xs[i] = (double)i;
            size_t k = (size_t)i * agents;
            a[i] = k < ep.rewards.size() ? ep.rewards[k] : 0.0;
            b[i] = k < ui.resim.rewards.size() ? ui.resim.rewards[k] : 0.0;
        }
        if (ui.resim.steps_compared &&
            ImPlot::BeginPlot("recorded against reconstructed reward",
                              ImVec2(-1, 220))) {
            ImPlot::SetupAxes("step", "reward");
            ImPlot::PlotLine("recorded", &xs[0], &a[0],
                             (int)ui.resim.steps_compared);
            ImPlot::PlotLine("reconstructed", &xs[0], &b[0],
                             (int)ui.resim.steps_compared);
            ImPlot::EndPlot();
        }
    }
    if (ImGui::Button("run again"))
        ui.resim_done = false;
    ImGui::End();
}

}  /* namespace */

int run_gui(Model &model, const DumpOptions &opt)
{
    Ui ui;
    /* Value initialised rather than cleared: the entry-point table
     * sits beside a string and a map, and a memset over those would
     * be a memset over their internals. */
    SceneGl gl = SceneGl();
    GLFWwindow *win;

    /* A recording with no complete episode has nothing to show and
     * never will. A run in progress has nothing to show yet, which is
     * a different thing: the window opens and fills as the frames
     * arrive. */
    if (!model.live() && model.info().episode_count == 0) {
        fprintf(stderr, "k26rl_view: the file carries no complete episode; "
                        "try --dump meta to see what it does carry\n");
        return 1;
    }

    ui.model = &model;
    ui.follow = true;
    ui.live = opt.live;
    ui.artifact = opt.artifact;
    ui.asset_reqs = opt.assets;
    ui.asset_tried = false;
    ui.episode_index = 0;
    ui.step = 0;
    ui.traj_pick = 0;
    ui.yaw = 0.0f;
    ui.pitch = 0.0f;
    ui.resim_done = false;
    ui.channel_on.assign(model.spec().channels.size(), 1);
    /* The window starts where the command line asked, so a picture
     * can be reproduced headlessly by repeating the arguments. */
    ui.scene_opt = opt.scene;
    ui.scene_resim_done = false;
    ui.scene_resim_ref = SCENE_ORIGIN;
    ui.scene_resim_ep = 0;
    ui.scene_ready = false;

    glfwSetErrorCallback(glfw_error_);
    if (!glfwInit()) {
        fprintf(stderr, "k26rl_view: cannot initialise the window system; "
                        "--dump works without a display\n");
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    win = glfwCreateWindow(1280, 860, "k26rl_view", NULL, NULL);
    if (!win) {
        fprintf(stderr, "k26rl_view: cannot open a window; "
                        "--dump works without a display\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    /* Dockable panels, so a person arranges the instrument rather
     * than accepting the arrangement it was shipped with. What a
     * viewer is for differs by the hour: a reward curve beside a
     * scene one day, four observation channels stacked the next, and
     * a fixed layout serves whichever of those the author happened to
     * have in mind. Docking is the reason Dear ImGui is vendored here
     * rather than taken from the system: it is not in the release
     * branch. */
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    apply_theme_(ImGui::GetStyle());
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    if (!scene_gl_init_(&gl)) {
        /* Reported in the scene panel and nowhere else: the rest of
         * the window is unaffected by a scene that cannot draw. */
        fprintf(stderr, "k26rl_view: %s\n", gl.error.c_str());
    }

    while (!glfwWindowShouldClose(win)) {
        std::string err;
        const Episode *ep;

        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        /* The dock space covers the window and its centre stays
         * transparent, because the centre is where the scene is. The
         * scene is drawn as the backdrop between the interface's
         * geometry being built and its being rendered, so a dock node
         * that painted its own background would paint over the
         * picture the panels exist to annotate. Panels dock around it
         * or float over it, as they are dragged. */
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport(),
                                     ImGuiDockNodeFlags_PassthruCentralNode);

        {
            /* The viewport the model projects with is the framebuffer
             * the window will draw into, so the aspect ratio the dump
             * reports is the aspect ratio on the screen. */
            int w = 0, h = 0;
            glfwGetFramebufferSize(win, &w, &h);
            if (w > 0 && h > 0) {
                ui.scene_opt.viewport.width = (uint32_t)w;
                ui.scene_opt.viewport.height = (uint32_t)h;
            }
        }
        /* One drain a frame. It is a read of a mapping and a copy of
         * whatever is new, so a window that redraws at the display's
         * rate over a simulation stepping far faster plots the newest
         * of what arrived and counts the rest as lost, which is what
         * the ring's discipline gives and what the live panel says. */
        if (model.live()) {
            uint32_t got = model.poll();
            uint32_t last = model.info().episode_count;
            if (got && ui.follow && last) {
                const Episode *newest = model.load(last - 1, &err);
                ui.episode_index = last - 1;
                if (newest && newest->step_count)
                    ui.step = newest->step_count - 1;
            }
            panel_live_(ui);
        }
        ep = model.load(ui.episode_index, &err);
        if (ep) {
            if (ui.step >= ep->step_count)
                ui.step = ep->step_count ? ep->step_count - 1 : 0;
            panel_timeline_(ui, *ep);
            panel_reward_(ui, *ep);
            panel_obs_(ui, *ep);
            panel_action_(ui, *ep);
            panel_traj_(ui, *ep);
            panel_world_(ui, *ep);
            panel_attitude_(ui, *ep);
            panel_wireframe_(ui, *ep);
            panel_scene_(ui, *ep, gl);
            panel_meta_(ui, *ep);
            panel_resim_(ui, *ep);
        } else {
            ImGui::Begin("Error");
            ImGui::TextUnformatted(err.c_str());
            ImGui::End();
        }

        ImGui::Render();
        {
            int w, h;
            glfwGetFramebufferSize(win, &w, &h);
            glViewport(0, 0, w, h);
            /* The interface's own ground, so the picture
             * behind the panels sits on the same colour they do. */
            glClearColor(16.0f / 255.0f, 16.0f / 255.0f, 18.0f / 255.0f,
                         1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            /* The scene is the backdrop and the panels float over it,
             * which is what a debugging instrument wants: the picture
             * is large and the numbers beside it are readable. */
            if (ui.scene_ready)
                scene_draw_(&gl, ui.scene, w, h);
        }
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    scene_gl_free_(&gl);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

}  /* namespace k26rl_view */
