/* gui.cpp - the run loop: the window, the dock layout,
 * the transport keys and the frame. */
#include "gui_internal.h"

namespace k26rl_view {

void glfw_error_(int code, const char *desc)
{
    fprintf(stderr, "k26rl_view: glfw error %d: %s\n", code, desc);
}

int run_gui(Model &model, const DumpOptions &opt, Prefs *prefs)
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
    ui.playing = false;
    ui.play_rate = prefs ? prefs->play_rate : 1.0;
    if (!(ui.play_rate > 0.0))
        ui.play_rate = 1.0;
    ui.play_accum = 0.0;
    ui.prefs = prefs;
    ui.show_settings = false;
    ui.show_help = false;
    ui.show_open = false;
    ui.open_artifact[0] = '\0';
    ui.capture_kind = 0;
    ui.capture_note_frames = 0;
    ui.show_save = false;
    ui.save_kind = 0;
    ui.save_dir = prefs ? prefs->save_dir : "";
    ui.save_name[0] = '\0';
    ui.obs_filter[0] = '\0';
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
    ui.scene_auto_resim = !ui.artifact.empty();
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

    /* The default arrangement is built on a first run, when the
     * working directory holds no imgui.ini to restore one from (the
     * layout file lives where the tool is run from, so each working
     * directory keeps its own), and again on request. Its shape: view controls left, channel panels tabbed
     * right, the timeline and the reward across the bottom, and the
     * centre left open, because the centre is where the scene is. */
    bool build_layout = access("imgui.ini", F_OK) != 0;

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
        menu_bar_(ui, win, &build_layout);
        panel_settings_(ui);
        panel_help_(ui);
        panel_open_(ui);
        panel_save_(ui);
        {
            ImGuiID root = ImGui::DockSpaceOverViewport(
                ImGui::GetMainViewport(),
                ImGuiDockNodeFlags_PassthruCentralNode);
            if (build_layout) {
                build_layout = false;
                ImGui::DockBuilderRemoveNodeDockedWindows(root, true);
                ImGui::DockBuilderRemoveNodeChildNodes(root);
                ImGuiID centre = root;
                ImGuiID left = ImGui::DockBuilderSplitNode(
                    centre, ImGuiDir_Left, 0.24f, NULL, &centre);
                ImGuiID right = ImGui::DockBuilderSplitNode(
                    centre, ImGuiDir_Right, 0.30f, NULL, &centre);
                ImGuiID bottom = ImGui::DockBuilderSplitNode(
                    centre, ImGuiDir_Down, 0.28f, NULL, &centre);
                ImGuiID left_low = ImGui::DockBuilderSplitNode(
                    left, ImGuiDir_Down, 0.40f, NULL, &left);
                ImGuiID bot_right = ImGui::DockBuilderSplitNode(
                    bottom, ImGuiDir_Right, 0.35f, NULL, &bottom);
                ImGui::DockBuilderDockWindow("Scene", left);
                ImGui::DockBuilderDockWindow("Run and episode", left_low);
                ImGui::DockBuilderDockWindow("Re-simulation", left_low);
                ImGui::DockBuilderDockWindow("Live", left_low);
                ImGui::DockBuilderDockWindow("Observations", right);
                ImGui::DockBuilderDockWindow("Actions", right);
                ImGui::DockBuilderDockWindow("Attitude", right);
                ImGui::DockBuilderDockWindow("Trajectory", right);
                ImGui::DockBuilderDockWindow("World frame", right);
                ImGui::DockBuilderDockWindow("Wireframe", right);
                ImGui::DockBuilderDockWindow("Timeline", bottom);
                ImGui::DockBuilderDockWindow("Reward", bot_right);
                ImGui::DockBuilderFinish(root);
            }
        }

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
            /* The transport's keys, listed in the controls window.
             * Text input keeps every key it captures. */
            if (!ImGui::GetIO().WantTextInput && ep->step_count) {
                uint32_t last = ep->step_count - 1;
                if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
                    ui.playing = !ui.playing;
                    ui.play_accum = 0.0;
                    ui.follow = false;
                    if (ui.playing && ep->complete && ui.step >= last)
                        ui.step = 0;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) &&
                    ui.step > 0) {
                    ui.playing = false;
                    ui.step--;
                    ui.follow = false;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) &&
                    ui.step < last) {
                    ui.playing = false;
                    ui.step++;
                    ui.follow = false;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Home, false)) {
                    ui.step = 0;
                    ui.follow = false;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_End, false)) {
                    ui.step = last;
                    ui.follow = false;
                }
            }
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
            /* The scene alone, before the interface is drawn over
             * it. */
            if (ui.capture_kind == 2) {
                capture_now_(ui, w, h);
                ui.capture_kind = 0;
            }
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            /* The window as shown, after everything has drawn. */
            if (ui.capture_kind == 1) {
                capture_now_(ui, w, h);
                ui.capture_kind = 0;
            }
        }
        glfwSwapBuffers(win);
    }

    /* The settings the session ends with are the settings the next
     * one starts with, unless this run was session-only. */
    if (ui.prefs) {
        ui.prefs->play_rate = ui.play_rate;
        prefs_save(*ui.prefs, ui.scene_opt);
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
