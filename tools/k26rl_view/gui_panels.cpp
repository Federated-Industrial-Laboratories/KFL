/* gui_panels.cpp - the data panels: timeline, reward,
 * channels, actions, trajectory, world, attitude,
 * wireframe, live, run and re-simulation. */
#include "gui_internal.h"

namespace k26rl_view {

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

/* The standard table: alternating row ground, stretch sizing, one
 * look everywhere so the panels read as one instrument. */

bool table_(const char *id, int cols)
{
    return ImGui::BeginTable(id, cols,
                             ImGuiTableFlags_RowBg |
                             ImGuiTableFlags_SizingStretchProp |
                             ImGuiTableFlags_PadOuterX);
}

/* One key-value row of a two-column table. */

void kv_(const char *key, const char *fmt, ...)
{
    va_list ap;
    char val[256];
    va_start(ap, fmt);
    vsnprintf(val, sizeof val, fmt, ap);
    va_end(ap);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", key);
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(val);
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
    /* Transport. Play advances the step against the wall clock at
     * the chosen multiple of simulated time: one control period of
     * recording per control period of watching at 1x. A frame-step
     * pauses first, the way a transport does, and a completed
     * episode pauses at its last step while a live one keeps
     * playing as records arrive. */
    {
        const uint32_t last = ep.step_count ? ep.step_count - 1 : 0;
        if (ImGui::Button("|<")) {
            ui.step = 0;
            ui.follow = false;
        }
        ImGui::SameLine();
        if (ImGui::Button("<") && ui.step > 0) {
            ui.playing = false;
            ui.step--;
            ui.follow = false;
        }
        ImGui::SameLine();
        if (ImGui::Button(ui.playing ? "pause" : "play")) {
            ui.playing = !ui.playing;
            ui.play_accum = 0.0;
            ui.follow = false;
            /* Play at the ending starts over rather than sitting on
             * one frame. */
            if (ui.playing && ep.complete && ui.step >= last)
                ui.step = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button(">") && ui.step < last) {
            ui.playing = false;
            ui.step++;
            ui.follow = false;
        }
        ImGui::SameLine();
        if (ImGui::Button(">|")) {
            ui.step = last;
            ui.follow = false;
        }
        ImGui::SameLine();
        {
            static const double rates[] = { 0.25, 0.5, 1.0, 2.0, 4.0,
                                            8.0 };
            int cur = 2;
            for (int r = 0; r < 6; r++) {
                if (ui.play_rate == rates[r])
                    cur = r;
            }
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 5.0f);
            if (ImGui::Combo("of wall clock", &cur,
                             "0.25x\0" "0.5x\0" "1x\0" "2x\0"
                             "4x\0" "8x\0"))
                ui.play_rate = rates[cur];
        }
        if (ui.playing && ep.step_count) {
            double dt = ui.model->spec().control_dt;
            if (!(dt > 0.0))
                dt = 1.0;
            ui.play_accum += (double)ImGui::GetIO().DeltaTime *
                             ui.play_rate / dt;
            if (ui.play_accum >= 1.0) {
                uint32_t whole = (uint32_t)ui.play_accum;
                ui.play_accum -= (double)whole;
                if (ui.step + whole < last) {
                    ui.step += whole;
                } else {
                    ui.step = last;
                    if (ep.complete)
                        ui.playing = false;
                }
            }
        }
    }

    if (table_("##identity", 2)) {
        kv_("identity", "ordinal %u, env %u, episode %u", ep.ordinal,
            ep.env, ep.episode);
        kv_("steps", "%u (%u transitions)", ep.step_count,
            ep.transitions());
        if (ui.step < ep.flags.size())
            kv_("step flags", "%s",
                flag_marks_(ep.flags[ui.step], marks, sizeof marks));
        ImGui::EndTable();
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
    /* The channel table: which traces plot, and each channel's value
     * at the shown step. The filter narrows by name. */
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.0f);
    ImGui::InputTextWithHint("##filter", "filter", ui.obs_filter,
                             sizeof ui.obs_filter);
    if (ImGui::BeginTable("##channels", 3,
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_SizingStretchProp |
                          ImGuiTableFlags_PadOuterX,
                          ImVec2(0, ImGui::GetFontSize() * 9.0f))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("plot",
                                ImGuiTableColumnFlags_WidthFixed,
                                ImGui::GetFontSize() * 2.5f);
        ImGui::TableSetupColumn("channel");
        ImGui::TableSetupColumn("value at step");
        ImGui::TableHeadersRow();
        for (size_t q = 0; q < pos.size(); q++) {
            size_t c = pos[q];
            const char *name = sp.channels[c].name.c_str();
            if (ui.obs_filter[0] && !strstr(name, ui.obs_filter))
                continue;
            ImGui::PushID((int)c);
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool on = ui.channel_on[c] != 0;
            if (ImGui::Checkbox("##on", &on))
                ui.channel_on[c] = on ? 1 : 0;
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(name);
            ImGui::TableNextColumn();
            {
                size_t k = (size_t)ui.step * sp.obs_total +
                           sp.channels[c].index;
                if (k < ep.obs.size())
                    ImGui::Text("%.6g", ep.obs[k]);
                else
                    ImGui::TextDisabled("-");
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
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
    if (ImGui::BeginTable("##decl", 4,
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_SizingStretchProp |
                          ImGuiTableFlags_PadOuterX,
                          ImVec2(0, ImGui::GetFontSize() * 7.0f))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("channel");
        ImGui::TableSetupColumn("kind");
        ImGui::TableSetupColumn("range");
        ImGui::TableSetupColumn("value at step");
        ImGui::TableHeadersRow();
        for (size_t k = 0; k < sp.actions.size(); k++) {
            const ActionDecl &a = sp.actions[k];
            if (a.offset < off || a.offset >= off + count)
                continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%u", a.offset);
            ImGui::TableNextColumn();
            if (a.has_kind && a.kind == K26RL_ACT_KIND_DISCRETE)
                ImGui::TextUnformatted("discrete");
            else
                ImGui::TextUnformatted("box");
            ImGui::TableNextColumn();
            if (a.has_kind && a.kind == K26RL_ACT_KIND_DISCRETE)
                ImGui::Text("arity %u", a.arity);
            else if (a.has_bounds)
                ImGui::Text("%g to %g", a.lo, a.hi);
            else
                ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            {
                size_t idx = (size_t)ui.step * sp.act_total + a.offset;
                if (idx < ep.act.size())
                    ImGui::Text("%.6g", ep.act[idx]);
                else
                    ImGui::TextDisabled("-");
            }
        }
        ImGui::EndTable();
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
        if (as.edges.empty()) {
            ImGui::TextWrapped(as.mesh_vertices == 0
                ? "this assembly declares no mesh; the scene draws "
                  "its colliders and its axes"
                : "this assembly declares mesh vertices but no faces, "
                  "so no wireframe can be drawn from it; the scene "
                  "draws its colliders and its axes");
            ImGui::PopID();
            continue;
        }
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
    if (table_("##live", 2)) {
        kv_("ring", "%s", fi.tap.c_str());
        kv_("slots", "%u of %u bytes", m.ring_slot_count(),
            m.ring_slot_size());
        kv_("frames accepted", "%llu",
            (unsigned long long)m.frames_accepted());
        /* The count is the reading. The line beside it is a limit no
         * reading here states: those steps are gone, and the track
         * does not pass through them. */
        kv_("frames lost", "%llu", (unsigned long long)lost);
        kv_("episodes", "%u", fi.episode_count);
        kv_("producer", "%s", m.producer_closed() ? "closed"
                                                  : "running");
        ImGui::EndTable();
    }
    if (lost)
        ImGui::TextColored(rgb_(COL_ACCENT),
                           "lost steps are not held and not drawn");
    ImGui::Checkbox("follow the newest step", &ui.follow);
    ImGui::End();
}

void panel_meta_(Ui &ui, const Episode &ep)
{
    const FileInfo &fi = ui.model->info();
    const Spec &sp = ui.model->spec();

    ImGui::Begin("Run and episode");
    if (ImGui::CollapsingHeader("run", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (table_("##run", 2)) {
            kv_("governing seed", "0x%016llx (ordinal %u)",
                (unsigned long long)fi.governing_seed, fi.rekey_ordinal);
            kv_("episode seed", "0x%016llx",
                (unsigned long long)ep.seed);
            kv_("environments", "%u", fi.n_envs);
            kv_("steps per chunk", "%u", fi.steps_per_chunk);
            kv_("episodes indexed", "%u", fi.episode_count);
            if (fi.unindexed_episode_starts)
                kv_("starts not indexed", "%u",
                    fi.unindexed_episode_starts);
            ImGui::EndTable();
        }
    }
    if (ImGui::CollapsingHeader("shape", ImGuiTreeNodeFlags_DefaultOpen)) {
        if (table_("##shape", 2)) {
            kv_("agents", "%u", sp.agent_count);
            kv_("observation channels", "%u", sp.obs_total);
            kv_("action channels", "%u", sp.act_total);
            kv_("control dt", "%g s", sp.control_dt);
            kv_("horizon", "%u", sp.horizon);
            kv_("episode flags", "0x%08x", sp.episode_flags);
            ImGui::EndTable();
        }
    }
    if (ImGui::CollapsingHeader("specification")) {
        if (table_("##spec", 2)) {
            kv_("blob", "%u bytes", (unsigned)sp.raw.size());
            kv_("endian probe", "0x%08x", sp.endian_probe);
            for (size_t k = 0; k < sp.unknown_tags.size(); k++) {
                char key[32];
                snprintf(key, sizeof key, "tag 0x%04x",
                         (unsigned)sp.unknown_tags[k].first);
                kv_(key, "%u bytes, not known to this build",
                    sp.unknown_tags[k].second);
            }
            ImGui::EndTable();
        }
    }
    if (ImGui::CollapsingHeader("randomisation",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        if (!ep.start_seen) {
            /* Not none: unknown. The record carrying them never
             * arrived, and a count of zero here would read as a fact
             * about the episode rather than about this window. */
            ImGui::TextDisabled("not known: the opening record never "
                                "arrived");
        } else if (ep.dr_tags.empty()) {
            ImGui::TextDisabled("no draws");
        } else if (table_("##draws", 2)) {
            ImGui::TableSetupColumn("parameter");
            ImGui::TableSetupColumn("value");
            ImGui::TableHeadersRow();
            for (size_t k = 0; k < ep.dr_tags.size(); k++) {
                char key[32];
                snprintf(key, sizeof key, "%u", ep.dr_tags[k]);
                kv_(key, "%.17g", ep.dr_values[k]);
            }
            ImGui::EndTable();
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
        if (ImGui::Button("rebuild and compare")) {
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

}  /* namespace k26rl_view */
