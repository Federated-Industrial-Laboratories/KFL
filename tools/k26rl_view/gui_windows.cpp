/* gui_windows.cpp - the menu bar and its windows:
 * settings, help, open, save and the capture. */
#include "gui_internal.h"

namespace k26rl_view {

/* The name a capture proposes: the episode file's own base name,
 * the step, and which picture it is. Where it lands is the save
 * window's question, answered by hand. */

std::string capture_name_(const Ui &ui, const char *what)
{
    std::string base = ui.model->info().path;
    size_t cut = base.find_last_of('/');
    if (cut != std::string::npos)
        base = base.substr(cut + 1);
    char name[320];
    snprintf(name, sizeof name, "%s.step%04u.%s.png", base.c_str(),
             ui.step, what);
    return name;
}

/* Read the framebuffer and write it as a PNG. GL rows run bottom
 * up; the file's run top down, so the rows are flipped here. */

void capture_now_(Ui &ui, int w, int h)
{
    if (w <= 0 || h <= 0)
        return;
    std::vector<uint8_t> px((size_t)w * h * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, &px[0]);
    std::vector<uint8_t> flip((size_t)w * h * 3);
    for (int y = 0; y < h; y++) {
        memcpy(&flip[(size_t)y * w * 3],
               &px[(size_t)(h - 1 - y) * w * 3], (size_t)w * 3);
    }
    bool ok = png_write_rgb8(ui.capture_path.c_str(), &flip[0],
                             (uint32_t)w, (uint32_t)h);
    ui.capture_note = (ok ? "saved " : "cannot write ") +
                      ui.capture_path;
    ui.capture_note_frames = 480;
}

/* The settings window: what is drawn and at what scale, one row per
 * element, and the depth cue beside them. Opened from the menu bar
 * and closable, so the main layout stays working surface. */

void panel_settings_(Ui &ui)
{
    SceneOptions &o = ui.scene_opt;

    if (!ui.show_settings)
        return;
    ImGui::SetNextWindowSize(ImVec2(440, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Settings", &ui.show_settings)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginTabBar("##settings")) {
        if (ImGui::BeginTabItem("elements")) {
            if (table_("##elements", 3)) {
                ImGui::TableSetupColumn("show",
                                        ImGuiTableColumnFlags_WidthFixed,
                                        ImGui::GetFontSize() * 3.0f);
                ImGui::TableSetupColumn("element");
                ImGui::TableSetupColumn("scale");
                ImGui::TableHeadersRow();
                for (int k = 0; k < ELEM_KIND_COUNT; k++) {
                    ImGui::PushID(k);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    bool on = o.enabled[k];
                    if (ImGui::Checkbox("##on", &on))
                        o.enabled[k] = on;
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted(
                        element_name((ElementKind)k));
                    ImGui::TableNextColumn();
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    if (k == ELEM_VELOCITY) {
                        float v = (float)o.velocity_seconds;
                        if (ImGui::DragFloat("##vs", &v, 0.1f, 0.0f,
                                             1.0e6f, "%.1f s"))
                            o.velocity_seconds = v;
                    } else if (k == ELEM_AXES) {
                        float v = (float)o.axis_length;
                        if (ImGui::DragFloat("##al", &v, 0.05f, 0.0f,
                                             1.0e6f, "%.2f m"))
                            o.axis_length = v;
                    } else if (k == ELEM_THRUSTER || k == ELEM_FORCE) {
                        float v = (float)o.thruster_scale;
                        if (ImGui::DragFloat("##ts", &v, 0.0005f, 0.0f,
                                             1.0e6f, "%.4f m per N"))
                            o.thruster_scale = v;
                    } else if (k == ELEM_SPIN) {
                        float v = (float)o.spin_scale;
                        if (ImGui::DragFloat("##ss", &v, 0.5f, 0.0f,
                                             1.0e6f, "%.1f m per rad/s"))
                            o.spin_scale = v;
                    } else {
                        ImGui::TextDisabled("-");
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
            ImGui::Spacing();
            if (ImGui::CollapsingHeader("depth cue")) {
                bool on = o.shading;
                if (ImGui::Checkbox("shading", &on))
                    o.shading = on;
                ImGui::TextDisabled("%s", SHADING_LABEL);
            }
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("settings file")) {
            if (table_("##pf", 2)) {
                kv_("state", ui.prefs ? "persists between runs"
                                      : "session only");
                kv_("file", ui.prefs && !ui.prefs->path.empty()
                            ? ui.prefs->path.c_str() : "-");
                ImGui::EndTable();
            }
            ImGui::TextDisabled("command-line flags override the "
                                "file; --session-only ignores it; "
                                "the headless dump never reads it");
            if (ui.prefs && ImGui::Button("save now")) {
                ui.prefs->play_rate = ui.play_rate;
                prefs_save(*ui.prefs, ui.scene_opt);
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

/* The controls window, one table, opened from the help menu. */

void panel_help_(Ui &ui)
{
    if (!ui.show_help)
        return;
    ImGui::SetNextWindowSize(ImVec2(380, 260), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Controls", &ui.show_help)) {
        ImGui::End();
        return;
    }
    if (table_("##keys", 2)) {
        ImGui::TableSetupColumn("key");
        ImGui::TableSetupColumn("function");
        ImGui::TableHeadersRow();
        kv_("space", "play or pause");
        kv_("left arrow", "one step back");
        kv_("right arrow", "one step forward");
        kv_("home", "first step");
        kv_("end", "last step");
        ImGui::EndTable();
    }
    ImGui::TextDisabled("drag a panel by its title to move or dock "
                        "it; view > reset layout restores the "
                        "default");
    ImGui::End();
}

/* Join a directory and a name without doubling the slash at the
 * root, which a persisted path would then carry around. */

std::string path_join_(const std::string &dir, const std::string &name)
{
    if (!dir.empty() && dir[dir.size() - 1] == '/')
        return dir + name;
    return dir + "/" + name;
}

/* Reopen the model over a different file, restoring the old file if
 * the new one refuses, and resetting what a file change invalidates. */

bool reopen_(Ui &ui, const std::string &path, std::string *err)
{
    std::string old = ui.model->info().path;
    ui.model->close();
    if (!ui.model->open(path, err)) {
        std::string e2;
        (void)ui.model->open(old, &e2);
        return false;
    }
    ui.episode_index = 0;
    ui.step = 0;
    ui.follow = true;
    ui.playing = false;
    ui.resim_done = false;
    ui.scene_resim_done = false;
    ui.scene_auto_resim = !ui.artifact.empty();
    ui.asset_tried = false;
    ui.assets.clear();
    ui.channel_on.assign(ui.model->spec().channels.size(), 1);
    ui.traj_pick = 0;
    if (ui.prefs) {
        prefs_touch_recent(ui.prefs, path);
        ui.prefs->play_rate = ui.play_rate;
        prefs_save(*ui.prefs, ui.scene_opt);
    }
    return true;
}

/* The open window: a directory table, episode files selectable, and
 * the artifact the rebuild will use. */

void panel_open_(Ui &ui)
{
    if (!ui.show_open)
        return;
    if (ui.open_dir.empty()) {
        char cwd[512];
        ui.open_dir = getcwd(cwd, sizeof cwd) ? cwd : ".";
    }
    ImGui::SetNextWindowSize(ImVec2(560, 440), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Open", &ui.show_open)) {
        ImGui::End();
        return;
    }
    {
        char buf[512];
        snprintf(buf, sizeof buf, "%s", ui.open_dir.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##dir", buf, sizeof buf,
                             ImGuiInputTextFlags_EnterReturnsTrue))
            ui.open_dir = buf;
    }
    std::vector<std::string> dirs, files;
    {
        DIR *d = opendir(ui.open_dir.c_str());
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != 0) {
                std::string name = e->d_name;
                if (name == "." )
                    continue;
                std::string full = path_join_(ui.open_dir, name);
                struct stat st;
                if (stat(full.c_str(), &st) != 0)
                    continue;
                if (S_ISDIR(st.st_mode)) {
                    dirs.push_back(name);
                } else {
                    size_t n = name.size();
                    if ((n > 6 && name.compare(n - 6, 6, ".k26ep") == 0) ||
                        (n > 7 && name.compare(n - 7, 7, ".k26epi") == 0))
                        files.push_back(name);
                }
            }
            closedir(d);
        } else {
            ImGui::TextDisabled("cannot list this directory");
        }
    }
    std::sort(dirs.begin(), dirs.end());
    std::sort(files.begin(), files.end());
    if (ImGui::BeginTable("##files", 2,
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_SizingStretchProp,
                          ImVec2(0, ImGui::GetFontSize() * 14.0f))) {
        ImGui::TableSetupColumn("name");
        ImGui::TableSetupColumn("type",
                                ImGuiTableColumnFlags_WidthFixed,
                                ImGui::GetFontSize() * 5.0f);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < dirs.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable((dirs[i] + "/").c_str(), false,
                                  ImGuiSelectableFlags_SpanAllColumns)) {
                if (dirs[i] == "..") {
                    size_t cut = ui.open_dir.find_last_of('/');
                    ui.open_dir = cut && cut != std::string::npos
                                  ? ui.open_dir.substr(0, cut) : "/";
                } else {
                    ui.open_dir = path_join_(ui.open_dir, dirs[i]);
                }
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("directory");
        }
        for (size_t i = 0; i < files.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            bool sel = ui.open_pick == files[i];
            if (ImGui::Selectable(files[i].c_str(), sel,
                                  ImGuiSelectableFlags_SpanAllColumns))
                ui.open_pick = files[i];
            ImGui::TableNextColumn();
            ImGui::TextDisabled("episode file");
        }
        ImGui::EndTable();
    }
    if (table_("##openopts", 2)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("artifact");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##artifact", ui.open_artifact,
                         sizeof ui.open_artifact);
        ImGui::EndTable();
    }
    if (!ui.open_error.empty())
        ImGui::TextColored(rgb_(COL_ERROR), "%s", ui.open_error.c_str());
    {
        bool can = !ui.open_pick.empty();
        if (!can)
            ImGui::BeginDisabled();
        if (ImGui::Button("open")) {
            std::string full = path_join_(ui.open_dir, ui.open_pick);
            std::string err;
            if (reopen_(ui, full, &err)) {
                ui.artifact = ui.open_artifact;
                ui.scene_auto_resim = !ui.artifact.empty();
                ui.open_error.clear();
                ui.show_open = false;
            } else {
                ui.open_error = err;
            }
        }
        if (!can)
            ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("cancel"))
            ui.show_open = false;
    }
    ImGui::End();
}

/* The save window: pick the directory by hand, keep or edit the
 * proposed name, and the capture of the next rendered frame lands
 * there. The directory is remembered between runs. */

void panel_save_(Ui &ui)
{
    if (!ui.show_save)
        return;
    if (ui.save_dir.empty()) {
        /* A picture is for a person, so the first default is their
         * home, not the directory the recording happens to live
         * in; every save after remembers where the last one went. */
        const char *home = getenv("HOME");
        if (home && *home) {
            ui.save_dir = home;
        } else {
            std::string base = ui.model->info().path;
            size_t cut = base.find_last_of('/');
            ui.save_dir = cut != std::string::npos
                          ? base.substr(0, cut) : ".";
        }
    }
    ImGui::SetNextWindowSize(ImVec2(560, 420), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Save image", &ui.show_save)) {
        ImGui::End();
        return;
    }
    {
        char buf[512];
        snprintf(buf, sizeof buf, "%s", ui.save_dir.c_str());
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##dir", buf, sizeof buf,
                             ImGuiInputTextFlags_EnterReturnsTrue))
            ui.save_dir = buf;
    }
    std::vector<std::string> dirs;
    {
        DIR *d = opendir(ui.save_dir.c_str());
        if (d) {
            struct dirent *e;
            while ((e = readdir(d)) != 0) {
                std::string name = e->d_name;
                if (name == ".")
                    continue;
                std::string full = path_join_(ui.save_dir, name);
                struct stat st;
                if (stat(full.c_str(), &st) == 0 &&
                    S_ISDIR(st.st_mode))
                    dirs.push_back(name);
            }
            closedir(d);
        } else {
            ImGui::TextDisabled("cannot list this directory");
        }
    }
    std::sort(dirs.begin(), dirs.end());
    if (ImGui::BeginTable("##dirs", 1,
                          ImGuiTableFlags_RowBg |
                          ImGuiTableFlags_ScrollY |
                          ImGuiTableFlags_SizingStretchProp,
                          ImVec2(0, ImGui::GetFontSize() * 10.0f))) {
        ImGui::TableSetupColumn("directory");
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < dirs.size(); i++) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            if (ImGui::Selectable((dirs[i] + "/").c_str(), false)) {
                if (dirs[i] == "..") {
                    size_t cut = ui.save_dir.find_last_of('/');
                    ui.save_dir = cut && cut != std::string::npos
                                  ? ui.save_dir.substr(0, cut) : "/";
                } else {
                    ui.save_dir = path_join_(ui.save_dir, dirs[i]);
                }
            }
        }
        ImGui::EndTable();
    }
    if (table_("##savename", 2)) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("name");
        ImGui::TableNextColumn();
        ImGui::SetNextItemWidth(-FLT_MIN);
        ImGui::InputText("##name", ui.save_name, sizeof ui.save_name);
        ImGui::EndTable();
    }
    {
        std::string full = path_join_(ui.save_dir, ui.save_name);
        struct stat st;
        if (stat(full.c_str(), &st) == 0)
            ImGui::TextColored(rgb_(COL_ACCENT),
                               "this name exists; save replaces it");
        bool can = ui.save_name[0] != '\0';
        if (!can)
            ImGui::BeginDisabled();
        if (ImGui::Button("save")) {
            ui.capture_kind = ui.save_kind;
            ui.capture_path = full;
            ui.show_save = false;
            if (ui.prefs) {
                ui.prefs->save_dir = ui.save_dir;
                prefs_save(*ui.prefs, ui.scene_opt);
            }
        }
        if (!can)
            ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("cancel"))
            ui.show_save = false;
    }
    ImGui::End();
}

/* The menu bar: files, settings, the layout, and help. The capture
 * items ask for a picture of the next rendered frame, which is the
 * one being built now. */

void menu_bar_(Ui &ui, GLFWwindow *win, bool *build_layout)
{
    if (!ImGui::BeginMainMenuBar())
        return;
    if (ImGui::BeginMenu("file")) {
        if (ImGui::MenuItem("open...")) {
            ui.show_open = true;
            ui.open_error.clear();
            snprintf(ui.open_artifact, sizeof ui.open_artifact, "%s",
                     ui.artifact.c_str());
        }
        if (ImGui::BeginMenu("open recent",
                             ui.prefs && !ui.prefs->recent.empty())) {
            for (size_t i = 0;
                 ui.prefs && i < ui.prefs->recent.size(); i++) {
                if (ImGui::MenuItem(ui.prefs->recent[i].c_str())) {
                    std::string err;
                    if (!reopen_(ui, ui.prefs->recent[i], &err)) {
                        ui.open_error = err;
                        ui.show_open = true;
                    }
                }
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("save image, window...")) {
            ui.show_save = true;
            ui.save_kind = 1;
            snprintf(ui.save_name, sizeof ui.save_name, "%s",
                     capture_name_(ui, "window").c_str());
        }
        if (ImGui::MenuItem("save image, scene...")) {
            ui.show_save = true;
            ui.save_kind = 2;
            snprintf(ui.save_name, sizeof ui.save_name, "%s",
                     capture_name_(ui, "scene").c_str());
        }
        ImGui::Separator();
        if (ImGui::MenuItem("quit"))
            glfwSetWindowShouldClose(win, GLFW_TRUE);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("settings")) {
        if (ImGui::MenuItem("visuals..."))
            ui.show_settings = true;
        ImGui::TextDisabled(ui.prefs ? "settings persist between runs"
                                     : "session only");
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("view")) {
        if (ImGui::MenuItem("reset layout"))
            *build_layout = true;
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("help")) {
        if (ImGui::MenuItem("controls..."))
            ui.show_help = true;
        ImGui::EndMenu();
    }
    if (ui.capture_note_frames > 0) {
        ui.capture_note_frames--;
        ImGui::Separator();
        ImGui::TextDisabled("%s", ui.capture_note.c_str());
    }
    ImGui::EndMainMenuBar();
}

}  /* namespace k26rl_view */
