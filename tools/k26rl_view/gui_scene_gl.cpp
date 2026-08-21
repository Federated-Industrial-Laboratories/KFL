/* gui_scene_gl.cpp - the scene renderer: the one
 * shader, the buffers, the draw pass and the scene
 * panel. */
#include "gui_internal.h"

namespace k26rl_view {

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
        { 0.98f, 0.98f, 0.55f, 1.0f },   /* detection line of sight */
        { 1.00f, 0.30f, 0.10f, 1.0f },   /* imparted thrust */
        { 0.60f, 0.45f, 0.95f, 1.0f },   /* angular velocity */
        { 0.35f, 0.95f, 0.80f, 1.0f }    /* closed datalink */
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
        /* A datalink line's fade travels as the shade the fragment
         * stage already multiplies its colour by, so a stale link dims
         * towards the scene's own ground; the alternative would be an
         * alpha and a blending state this window does not otherwise
         * keep. The figure is the model's: the window scales a colour
         * by it and decides nothing. */
        gl->VertexAttrib1f(1, e.kind == ELEM_DATALINK
                              ? (float)e.link.fade : 1.0f);
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
    }
    bool mismatch = !ui.scene_resim_done || ui.scene_resim_ref != o.frame ||
                    ui.scene_resim_ep != ui.episode_index;
    bool behind = !mismatch && ep.step_count > ui.scene_resim.steps_compared;
    if (!ui.artifact.empty() && (mismatch || behind)) {
        /* An artifact on the command line is a request to see the
         * flight, so the rebuild runs unasked: at once when the frame
         * or episode changes, and at a bounded pace as a live episode
         * grows, because each rebuild replays the whole episode from
         * its first step. The button stays as the manual override. */
        bool go = ImGui::Button(behind ? "rebuild to the newest step"
                                       : "rebuild poses in this frame");
        if (ui.scene_auto_resim) {
            go = true;
            ui.scene_auto_resim = false;
        }
        static double last_follow = -1.0e9;
        double now = ImGui::GetTime();
        if (mismatch && now - last_follow >= 1.0)
            go = true;
        if (behind && ep.step_count >= ui.scene_resim.steps_compared + 16 &&
            now - last_follow >= 5.0)
            go = true;
        if (go) {
            last_follow = now;
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
    /* What is drawn, and at what scale, lives in the settings
     * window, so this panel stays the working camera surface. */
    if (ImGui::Button("elements and scales..."))
        ui.show_settings = true;

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

    /* A live step past the last rebuild has no pose; showing the
     * newest rebuilt step with a note beats an empty scene. */
    uint32_t sstep = ui.step;
    if (in.resim && in.resim->ran && in.resim->has_bodies &&
        in.resim->steps_compared && sstep >= in.resim->steps_compared)
        sstep = in.resim->steps_compared - 1;
    ui.scene = scene_build(in, o, sstep);
    ui.scene_ready = true;
    if (sstep != ui.step)
        ui.scene.message += " (rebuilt to step " + std::to_string(sstep) +
                            ", the live step is " +
                            std::to_string(ui.step) + ")";
    ImGui::Separator();
    ImGui::TextWrapped("%s", ui.scene.message.c_str());
    if (ImGui::CollapsingHeader("drawn")) {
        if (table_("##drawnkv", 2)) {
            kv_("frame", "%s", ui.scene.frame_name.c_str());
            kv_("elements at step", "%u at %u",
                (unsigned)ui.scene.elements.size(), ui.step);
            kv_("eye", "%.3f %.3f %.3f m", ui.scene.eye[0],
                ui.scene.eye[1], ui.scene.eye[2]);
            ImGui::EndTable();
        }
        if (table_("##drawn", 3)) {
            ImGui::TableSetupColumn("element");
            ImGui::TableSetupColumn("name");
            ImGui::TableSetupColumn("segments");
            ImGui::TableHeadersRow();
            for (size_t i = 0; i < ui.scene.elements.size(); i++) {
                const SceneElement &e = ui.scene.elements[i];
                size_t drawn = 0;
                for (size_t sgi = 0; sgi < e.segments.size(); sgi++)
                    drawn += e.segments[sgi].drawn ? 1 : 0;
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(element_name(e.kind));
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(e.name.c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%u of %u", (unsigned)drawn,
                            (unsigned)e.segments.size());
            }
            ImGui::EndTable();
        }
    }
    ImGui::End();
}

}  /* namespace k26rl_view */
