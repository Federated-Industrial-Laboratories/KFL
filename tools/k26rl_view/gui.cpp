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
 * The trajectory view is drawn with this file's own projection over
 * the shared 3D maths library, not with a scene renderer: version 1
 * plots reconstructed points, and the wireframe scene view belongs to
 * a later version.
 */
#include "gui.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

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
    uint32_t episode_index;
    uint32_t step;
    std::vector<char> channel_on;   /* one flag per observation channel */
    int traj_pick;
    float yaw, pitch;
    bool resim_done;
    ResimResult resim;
    std::string error;
};

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
    case K26RL_END_TERMINATED: return ImVec4(0.45f, 0.85f, 0.45f, 1.0f);
    case K26RL_END_TRUNCATED:  return ImVec4(0.95f, 0.80f, 0.35f, 1.0f);
    case K26RL_END_FAULT:      return ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    default:                   return ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
    }
}

void panel_timeline_(Ui &ui, const Episode &ep)
{
    char marks[128];
    ImGui::Begin("Timeline");

    ImGui::Text("file %s", ui.model->info().path.c_str());
    if (!ui.model->info().clean_close) {
        ImGui::TextColored(ImVec4(0.95f, 0.80f, 0.35f, 1.0f),
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
    }

    ImGui::TextColored(ending_colour_(ep.end_reason),
                       "ends: %s", end_reason_name(ep.end_reason));
    if (ep.end_reason == K26RL_END_FAULT) {
        ImGui::SameLine();
        ImGui::TextColored(ending_colour_(ep.end_reason),
                           "(reason code %u; the final record is the one "
                           "step record that is not a transition)",
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

void panel_reward_(Ui &ui, const Episode &ep)
{
    const uint32_t agents = ui.model->spec().agent_count
                            ? ui.model->spec().agent_count : 1;
    std::vector<double> ret = Model::returns(ep, agents);
    std::vector<double> xs(ep.step_count);
    std::vector<double> ys(ep.step_count);

    for (uint32_t i = 0; i < ep.step_count; i++)
        xs[i] = (double)i;

    ImGui::Begin("Reward");
    if (ImPlot::BeginPlot("per-step reward", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("step", "reward");
        for (uint32_t a = 0; a < agents; a++) {
            char label[32];
            snprintf(label, sizeof label, "agent %u", a);
            for (uint32_t i = 0; i < ep.step_count; i++) {
                size_t k = (size_t)i * agents + a;
                ys[i] = k < ep.rewards.size() ? ep.rewards[k] : 0.0;
            }
            if (ep.step_count)
                ImPlot::PlotLine(label, &xs[0], &ys[0], (int)ep.step_count);
        }
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("episode return", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("step", "return");
        for (uint32_t a = 0; a < agents; a++) {
            char label[32];
            snprintf(label, sizeof label, "agent %u", a);
            for (uint32_t i = 0; i < ep.step_count; i++) {
                size_t k = (size_t)i * agents + a;
                ys[i] = k < ret.size() ? ret[k] : 0.0;
            }
            if (ep.step_count)
                ImPlot::PlotLine(label, &xs[0], &ys[0], (int)ep.step_count);
        }
        ImPlot::EndPlot();
    }
    for (uint32_t a = 0; a < agents && a < ep.terminal_adjustments.size(); a++)
        ImGui::Text("terminal adjustment, agent %u: %g", a,
                    ep.terminal_adjustments[a]);
    ImGui::End();
}

void panel_obs_(Ui &ui, const Episode &ep)
{
    const Spec &sp = ui.model->spec();
    std::vector<double> xs(ep.step_count), ys(ep.step_count);

    for (uint32_t i = 0; i < ep.step_count; i++)
        xs[i] = (double)i;

    ImGui::Begin("Observations");
    ImGui::TextUnformatted("channels");
    for (size_t c = 0; c < sp.channels.size(); c++) {
        bool on = ui.channel_on[c] != 0;
        if (ImGui::Checkbox(sp.channels[c].name.c_str(), &on))
            ui.channel_on[c] = on ? 1 : 0;
        if ((c % 3) != 2 && c + 1 < sp.channels.size())
            ImGui::SameLine();
    }
    if (ImPlot::BeginPlot("observation channels", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("step", "value");
        for (size_t c = 0; c < sp.channels.size(); c++) {
            if (!ui.channel_on[c])
                continue;
            uint32_t idx = sp.channels[c].index;
            for (uint32_t i = 0; i < ep.step_count; i++) {
                size_t k = (size_t)i * sp.obs_total + idx;
                ys[i] = k < ep.obs.size() ? ep.obs[k] : 0.0;
            }
            if (ep.step_count)
                ImPlot::PlotLine(sp.channels[c].name.c_str(), &xs[0], &ys[0],
                                 (int)ep.step_count);
        }
        ImPlot::EndPlot();
    }
    ImGui::End();
}

void panel_action_(Ui &ui, const Episode &ep)
{
    const Spec &sp = ui.model->spec();
    std::vector<double> xs(ep.step_count), ys(ep.step_count);

    for (uint32_t i = 0; i < ep.step_count; i++)
        xs[i] = (double)i;

    ImGui::Begin("Actions");
    if (ImPlot::BeginPlot("action channels", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("step", "value");
        for (uint32_t j = 0; j < sp.act_total; j++) {
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
            /* The declared bounds, drawn as the limits they are, so a
             * trace riding its bound is visible as such. */
            if (decl && decl->has_bounds && ep.step_count) {
                double bx[2] = { 0.0, (double)(ep.step_count - 1) };
                double lo[2] = { decl->lo, decl->lo };
                double hi[2] = { decl->hi, decl->hi };
                char blabel[64];
                snprintf(blabel, sizeof blabel, "action %u bounds", j);
                ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 0.8f));
                ImPlot::PlotLine(blabel, bx, lo, 2);
                ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 0.8f));
                ImPlot::PlotLine(blabel, bx, hi, 2);
            }
        }
        ImPlot::EndPlot();
    }
    for (size_t k = 0; k < sp.actions.size(); k++) {
        const ActionDecl &a = sp.actions[k];
        if (a.has_kind && a.kind == K26RL_ACT_KIND_DISCRETE)
            ImGui::Text("action %u: discrete, arity %u", a.offset, a.arity);
        else if (a.has_bounds)
            ImGui::Text("action %u: box [%g, %g]", a.offset, a.lo, a.hi);
    }
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
            "no channel set matches the drawable naming convention; "
            "every channel is plotted in the observation panel");
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
        ImGui::TextWrapped("the file records observation channels, not world "
                           "states; supply an artifact to reconstruct the "
                           "bodies");
        ImGui::End();
        return;
    }
    if (!ui.resim_done || !ui.resim.ran || !ui.resim.has_bodies) {
        ImGui::TextWrapped("run the re-simulation panel's reconstruction to "
                           "populate this view");
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
    ImGui::Text("randomisation draws: %u", (unsigned)ep.dr_tags.size());
    for (size_t k = 0; k < ep.dr_tags.size(); k++)
        ImGui::Text("  parameter %u = %.17g", ep.dr_tags[k], ep.dr_values[k]);
    ImGui::End();
}

void panel_resim_(Ui &ui, const Episode &ep)
{
    ImGui::Begin("Re-simulation");
    if (ui.artifact.empty()) {
        ImGui::TextWrapped(
            "no artifact supplied; run with --artifact PATH to rebuild "
            "this episode from its recorded seed and action stream and "
            "compare the two bitwise");
        ImGui::End();
        return;
    }
    ImGui::Text("artifact %s", ui.artifact.c_str());
    if (!ui.resim_done) {
        if (ImGui::Button("reconstruct and compare")) {
            ui.resim = resimulate(*ui.model, ep, ui.artifact);
            ui.resim_done = true;
        }
        ImGui::TextWrapped(
            "reconstruct and compare only: the recorded action stream is "
            "replayed as recorded, and nothing here edits it");
        ImGui::End();
        return;
    }

    if (!ui.resim.ran) {
        ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.0f), "failed: %s",
                           ui.resim.message.c_str());
    } else if (ui.resim.equal) {
        ImGui::TextColored(ImVec4(0.45f, 0.85f, 0.45f, 1.0f),
                           "equal bitwise over %u steps",
                           ui.resim.steps_compared);
    } else {
        ImGui::TextColored(ImVec4(0.95f, 0.40f, 0.40f, 1.0f),
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

int run_gui(Model &model, const std::string &artifact)
{
    Ui ui;
    GLFWwindow *win;

    if (model.info().episode_count == 0) {
        fprintf(stderr, "k26rl_view: the file carries no complete episode; "
                        "try --dump meta to see what it does carry\n");
        return 1;
    }

    ui.model = &model;
    ui.artifact = artifact;
    ui.episode_index = 0;
    ui.step = 0;
    ui.traj_pick = 0;
    ui.yaw = 0.0f;
    ui.pitch = 0.0f;
    ui.resim_done = false;
    ui.channel_on.assign(model.spec().channels.size(), 1);

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
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    while (!glfwWindowShouldClose(win)) {
        std::string err;
        const Episode *ep;

        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

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
            glClearColor(0.09f, 0.09f, 0.11f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

}  /* namespace k26rl_view */
