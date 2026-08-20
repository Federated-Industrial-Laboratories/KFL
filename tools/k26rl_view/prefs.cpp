/* prefs.cpp - the persisted settings declared in prefs.h. */
#include "prefs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace k26rl_view {

namespace {

/* One numeric key. Values are printed with %.17g so a double makes
 * the round trip exactly. */
void put_(FILE *f, const char *key, double v)
{
    fprintf(f, "%s=%.17g\n", key, v);
}

bool key_(const char *line, const char *key, const char **val)
{
    size_t n = strlen(key);
    if (strncmp(line, key, n) != 0 || line[n] != '=')
        return false;
    *val = line + n + 1;
    return true;
}

}  /* namespace */

void prefs_load(const char *path, Prefs *p, SceneOptions *s)
{
    p->path = path ? path : "";
    FILE *f = path ? fopen(path, "r") : 0;
    if (!f)
        return;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        char *nl = strchr(line, '\n');
        if (nl)
            *nl = '\0';
        const char *v;
        if (key_(line, "play_rate", &v)) {
            p->play_rate = atof(v);
        } else if (key_(line, "save_dir", &v)) {
            p->save_dir = v;
        } else if (key_(line, "recent", &v)) {
            if (*v && p->recent.size() < 8)
                p->recent.push_back(v);
        } else if (key_(line, "shading", &v)) {
            s->shading = atoi(v) != 0;
        } else if (key_(line, "velocity_seconds", &v)) {
            s->velocity_seconds = atof(v);
        } else if (key_(line, "axis_length", &v)) {
            s->axis_length = atof(v);
        } else if (key_(line, "thruster_scale", &v)) {
            s->thruster_scale = atof(v);
        } else if (key_(line, "spin_scale", &v)) {
            s->spin_scale = atof(v);
        } else if (key_(line, "camera_mode", &v)) {
            int m = atoi(v);
            if (m >= 0 && m <= 2)
                s->camera.mode = (CameraMode)m;
        } else if (key_(line, "projection", &v)) {
            int m = atoi(v);
            if (m >= 0 && m <= 1)
                s->camera.projection = (ProjectionKind)m;
        } else if (key_(line, "fov_deg", &v)) {
            s->camera.fov_y_deg = atof(v);
        } else if (key_(line, "ortho_height", &v)) {
            s->camera.ortho_height = atof(v);
        } else if (key_(line, "orbit_azimuth_deg", &v)) {
            s->camera.azimuth_deg = atof(v);
        } else if (key_(line, "orbit_elevation_deg", &v)) {
            s->camera.elevation_deg = atof(v);
        } else if (key_(line, "orbit_radius", &v)) {
            s->camera.radius = atof(v);
        } else if (key_(line, "chase_x", &v)) {
            s->camera.chase[0] = atof(v);
        } else if (key_(line, "chase_y", &v)) {
            s->camera.chase[1] = atof(v);
        } else if (key_(line, "chase_z", &v)) {
            s->camera.chase[2] = atof(v);
        } else {
            for (int k = 0; k < ELEM_KIND_COUNT; k++) {
                char name[64];
                snprintf(name, sizeof name, "element_%s",
                         element_name((ElementKind)k));
                if (key_(line, name, &v))
                    s->enabled[k] = atoi(v) != 0;
            }
        }
    }
    fclose(f);
}

bool prefs_save(const Prefs &p, const SceneOptions &s)
{
    if (p.path.empty())
        return false;
    FILE *f = fopen(p.path.c_str(), "w");
    if (!f)
        return false;
    fputs("# k26rl_view window settings. Command-line flags override\n"
          "# these; --session-only ignores this file; the headless\n"
          "# dump never reads it.\n", f);
    put_(f, "play_rate", p.play_rate);
    if (!p.save_dir.empty())
        fprintf(f, "save_dir=%s\n", p.save_dir.c_str());
    put_(f, "shading", s.shading ? 1.0 : 0.0);
    put_(f, "velocity_seconds", s.velocity_seconds);
    put_(f, "axis_length", s.axis_length);
    put_(f, "thruster_scale", s.thruster_scale);
    put_(f, "spin_scale", s.spin_scale);
    put_(f, "camera_mode", (double)s.camera.mode);
    put_(f, "projection", (double)s.camera.projection);
    put_(f, "fov_deg", s.camera.fov_y_deg);
    put_(f, "ortho_height", s.camera.ortho_height);
    put_(f, "orbit_azimuth_deg", s.camera.azimuth_deg);
    put_(f, "orbit_elevation_deg", s.camera.elevation_deg);
    put_(f, "orbit_radius", s.camera.radius);
    put_(f, "chase_x", s.camera.chase[0]);
    put_(f, "chase_y", s.camera.chase[1]);
    put_(f, "chase_z", s.camera.chase[2]);
    for (int k = 0; k < ELEM_KIND_COUNT; k++)
        fprintf(f, "element_%s=%d\n", element_name((ElementKind)k),
                s.enabled[k] ? 1 : 0);
    for (size_t i = 0; i < p.recent.size() && i < 8; i++)
        fprintf(f, "recent=%s\n", p.recent[i].c_str());
    return fclose(f) == 0;
}

void prefs_touch_recent(Prefs *p, const std::string &file)
{
    for (size_t i = 0; i < p->recent.size(); i++) {
        if (p->recent[i] == file) {
            p->recent.erase(p->recent.begin() + i);
            break;
        }
    }
    p->recent.insert(p->recent.begin(), file);
    while (p->recent.size() > 8)
        p->recent.pop_back();
}

}  /* namespace k26rl_view */
