/* scene.cpp - the scene model: camera, frame, geometry and projection.
 *
 * The order of operations is the whole design, so it is stated once
 * here and followed everywhere below.
 *
 *   1. Positions arrive in binary64, relative to a chosen reference
 *      body, from the stepping surface's own body getter. The
 *      reference is the getter's reference: the scene shows the frame
 *      the getter was asked for and does not invent a second one.
 *   2. The camera resolves in that frame, in binary64.
 *   3. Each element gets a model matrix whose translation is the
 *      body's position less the camera eye, formed in binary64. That
 *      difference is small by construction, whatever the scene's
 *      distance from the origin.
 *   4. The view matrix is built with the eye at the origin, since the
 *      eye has already been subtracted.
 *   5. Only then does anything narrow to single precision: the
 *      matrix, and the local coordinates that were always small.
 *   6. The projection runs in single precision, because that is what
 *      the pipeline the numbers are going to runs in, and a dump that
 *      reported binary64 results would be reporting numbers no frame
 *      ever uses.
 *
 * Step 6 is why cross-binary bit identity is not claimed for the
 * projection dump, on the same footing as the closed-form models
 * elsewhere in this tree. Within one binary it is exact.
 */
#include "scene.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "k26m3d.h"
}

namespace k26rl_view {

/* The three standing labels, each shortened to what a reader cannot
 * get from the reading beside it. The first states a distinction a
 * depth cue could otherwise be mistaken for; the other two state
 * where a line is drawn from, which the geometry does not show. */
const char *const SHADING_LABEL =
    "depth cue from a view-space light; not an illumination "
    "calculation";

const char *const SCENE_TRAJECTORY_LABEL =
    "track about the reference origin; exact when the reference body "
    "is the observer";

const char *const SCENE_DETECTION_LABEL =
    "line of sight from the reference origin, drawn while detected; "
    "exact when the reference body carries the payload";

namespace {

/* The angle at which the up vector is too close to the view direction
 * for a right-handed basis to be formed from the two. Below it the
 * camera picks a different up rather than emitting a degenerate view
 * matrix, which is a visibility decision and therefore belongs here
 * and not in the window. */
const double UP_DEGENERATE = 1.0e-6;

const char *const ELEMENT_NAMES[ELEM_KIND_COUNT] = {
    "wireframe", "collider", "axes", "trajectory", "velocity", "port",
    "thruster", "detection", "force", "spin"
};

K26V3 v3_(const double a[3]) { return k26m3d_v3(a[0], a[1], a[2]); }

void store_(double out[3], K26V3 v) { out[0] = v.x; out[1] = v.y; out[2] = v.z; }

void mat_to_double_(double out[16], const K26M4 *m)
{
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++)
            out[c * 4 + r] = m->m[c][r];
    }
}

void mat_to_float_(float out[16], const K26M4 *m)
{
    for (int c = 0; c < 4; c++) {
        for (int r = 0; r < 4; r++)
            out[c * 4 + r] = (float)m->m[c][r];
    }
}

/* The assembly bound to a body, or null.
 *
 * The binding is by body index because a recording of two craft has
 * two of them, and an element built from a list position rather than
 * from the body it belongs to draws the second craft's shape at the
 * first craft's place. Every binding reaching here has already passed
 * its digest verdict. */
const Asset *asset_for_(const SceneInput &in, uint32_t body)
{
    for (size_t i = 0; i < in.assets.size(); i++) {
        if (in.assets[i].body == body && in.assets[i].asset &&
            in.assets[i].asset->loaded)
            return in.assets[i].asset;
    }
    return 0;
}

/* A line set under construction, in one local frame. */
struct Builder {
    std::vector<double> v;
    std::vector<SceneSegment> seg;

    uint32_t point(K26V3 p)
    {
        v.push_back(p.x);
        v.push_back(p.y);
        v.push_back(p.z);
        return (uint32_t)(v.size() / 3 - 1);
    }
    void link(uint32_t a, uint32_t b)
    {
        SceneSegment s;
        s.a = a;
        s.b = b;
        s.drawn = true;
        seg.push_back(s);
    }
    void line(K26V3 a, K26V3 b) { uint32_t i = point(a); link(i, point(b)); }
    bool empty() const { return v.empty(); }
};

/* A closed polygon inscribing the circle of radius r about c in the
 * plane spanned by u and w. Sixteen sides, which reads as a circle at
 * every scale this view is used at and keeps the dump small enough to
 * compare by eye. */
void circle_(Builder *b, K26V3 c, K26V3 u, K26V3 w, double r)
{
    const int n = 16;
    uint32_t first = 0, prev = 0;
    for (int i = 0; i < n; i++) {
        double t = 2.0 * K26M3D_PI * (double)i / (double)n;
        K26V3 p = k26m3d_v3_add(c,
            k26m3d_v3_add(k26m3d_v3_scale(u, r * cos(t)),
                          k26m3d_v3_scale(w, r * sin(t))));
        uint32_t idx = b->point(p);
        if (i == 0)
            first = idx;
        else
            b->link(prev, idx);
        prev = idx;
    }
    b->link(prev, first);
}

/* Half of the same circle, for a capsule's end caps: the arc from u
 * through the axis direction and back to -u. */
void arc_(Builder *b, K26V3 c, K26V3 u, K26V3 w, double r)
{
    const int n = 8;
    uint32_t prev = 0;
    for (int i = 0; i <= n; i++) {
        double t = K26M3D_PI * (double)i / (double)n;
        K26V3 p = k26m3d_v3_add(c,
            k26m3d_v3_add(k26m3d_v3_scale(u, r * cos(t)),
                          k26m3d_v3_scale(w, r * sin(t))));
        uint32_t idx = b->point(p);
        if (i)
            b->link(prev, idx);
        prev = idx;
    }
}

/* Any pair of unit vectors completing a right-handed basis with n.
 * A capsule and a sphere are symmetric about their own axes, so only
 * the axis carries meaning and any completion draws the same shape. */
void basis_(K26V3 n, K26V3 *u, K26V3 *w)
{
    K26V3 seed = (fabs(n.z) > 0.9) ? k26m3d_v3(1, 0, 0) : k26m3d_v3(0, 0, 1);
    *u = k26m3d_v3_norm(k26m3d_v3_cross(seed, n));
    *w = k26m3d_v3_cross(n, *u);
}

void box_(Builder *b, K26V3 c, const double rot[9], const double half[3])
{
    K26V3 ax[3];
    uint32_t idx[8];
    static const int pairs[12][2] = {
        {0,1},{1,3},{3,2},{2,0}, {4,5},{5,7},{7,6},{6,4},
        {0,4},{1,5},{2,6},{3,7}
    };
    for (int i = 0; i < 3; i++)
        ax[i] = k26m3d_v3(rot[0 * 3 + i], rot[1 * 3 + i], rot[2 * 3 + i]);
    for (int i = 0; i < 8; i++) {
        double s[3];
        s[0] = (i & 1) ? half[0] : -half[0];
        s[1] = (i & 2) ? half[1] : -half[1];
        s[2] = (i & 4) ? half[2] : -half[2];
        idx[i] = b->point(k26m3d_v3_add(c,
            k26m3d_v3_add(k26m3d_v3_scale(ax[0], s[0]),
                k26m3d_v3_add(k26m3d_v3_scale(ax[1], s[1]),
                              k26m3d_v3_scale(ax[2], s[2])))));
    }
    for (int i = 0; i < 12; i++)
        b->link(idx[pairs[i][0]], idx[pairs[i][1]]);
}

void sphere_(Builder *b, K26V3 c, double r)
{
    circle_(b, c, k26m3d_v3(1, 0, 0), k26m3d_v3(0, 1, 0), r);
    circle_(b, c, k26m3d_v3(0, 1, 0), k26m3d_v3(0, 0, 1), r);
    circle_(b, c, k26m3d_v3(0, 0, 1), k26m3d_v3(1, 0, 0), r);
}

void capsule_(Builder *b, K26V3 pa, K26V3 pb, double r)
{
    K26V3 n = k26m3d_v3_sub(pb, pa);
    K26V3 u, w;
    double len = k26m3d_v3_len(n);

    n = len > 0.0 ? k26m3d_v3_scale(n, 1.0 / len) : k26m3d_v3(0, 0, 1);
    basis_(n, &u, &w);
    circle_(b, pa, u, w, r);
    circle_(b, pb, u, w, r);
    for (int i = 0; i < 4; i++) {
        double t = 0.5 * K26M3D_PI * (double)i;
        K26V3 off = k26m3d_v3_add(k26m3d_v3_scale(u, r * cos(t)),
                                  k26m3d_v3_scale(w, r * sin(t)));
        b->line(k26m3d_v3_add(pa, off), k26m3d_v3_add(pb, off));
    }
    arc_(b, pa, u, k26m3d_v3_neg(n), r);
    arc_(b, pa, w, k26m3d_v3_neg(n), r);
    arc_(b, pb, u, n, r);
    arc_(b, pb, w, n, r);
}

/* The narrowing, and the projection that follows it. This is step 5
 * and step 6 of the header comment and the only place either
 * happens. */
void finish_(SceneElement *e, const Builder &b, const K26M4 *mv,
             const K26M4 *mvp, bool local_static)
{
    size_t n = b.v.size() / 3;
    float mvf[16];

    e->local_static = local_static;

    mat_to_float_(e->mvp, mvp);
    mat_to_float_(mvf, mv);
    e->local.resize(n * 3);
    e->ndc.resize(n * 3);
    e->behind.resize(n);
    e->segments = b.seg;
    for (size_t i = 0; i < n; i++) {
        float x = (float)b.v[i * 3];
        float y = (float)b.v[i * 3 + 1];
        float z = (float)b.v[i * 3 + 2];
        float vz, cx, cy, cz, cw;

        e->local[i * 3] = x;
        e->local[i * 3 + 1] = y;
        e->local[i * 3 + 2] = z;
        vz = mvf[2] * x + mvf[6] * y + mvf[10] * z + mvf[14];
        cx = e->mvp[0] * x + e->mvp[4] * y + e->mvp[8] * z + e->mvp[12];
        cy = e->mvp[1] * x + e->mvp[5] * y + e->mvp[9] * z + e->mvp[13];
        cz = e->mvp[2] * x + e->mvp[6] * y + e->mvp[10] * z + e->mvp[14];
        cw = e->mvp[3] * x + e->mvp[7] * y + e->mvp[11] * z + e->mvp[15];
        /* The eye plane, not the near plane, and taken in view space
         * rather than from the clip w, so the test means the same
         * thing under both projections: an orthographic camera has a
         * clip w of one everywhere and would flag nothing. */
        e->behind[i] = (vz >= 0.0f) ? 1u : 0u;
        if (cw != 0.0f) {
            e->ndc[i * 3] = cx / cw;
            e->ndc[i * 3 + 1] = cy / cw;
            e->ndc[i * 3 + 2] = cz / cw;
        } else {
            e->ndc[i * 3] = 0.0f;
            e->ndc[i * 3 + 1] = 0.0f;
            e->ndc[i * 3 + 2] = 0.0f;
        }
    }
    for (size_t i = 0; i < e->segments.size(); i++) {
        SceneSegment &s = e->segments[i];
        s.drawn = s.a < n && s.b < n && !e->behind[s.a] && !e->behind[s.b];
    }
}

/* The optional depth cue. The face normal is carried into view space
 * by the model-view rotation and dotted against the light direction;
 * nothing radiometric happens, and SHADING_LABEL says so wherever the
 * result appears. */
void shade_(SceneElement *e, const Asset &as, const K26M4 *mv,
            const double light[3])
{
    K26V3 l = k26m3d_v3_norm(v3_(light));

    e->faces.clear();
    e->faces.reserve(as.faces.size());
    for (size_t i = 0; i < as.faces.size(); i++) {
        const Face &f = as.faces[i];
        SceneFace sf;
        size_t a = (size_t)f.a * 3, b = (size_t)f.b * 3, c = (size_t)f.c * 3;
        if (a + 2 >= as.vertices.size() || b + 2 >= as.vertices.size() ||
            c + 2 >= as.vertices.size())
            continue;
        K26V3 pa = k26m3d_v3(as.vertices[a], as.vertices[a + 1],
                             as.vertices[a + 2]);
        K26V3 pb = k26m3d_v3(as.vertices[b], as.vertices[b + 1],
                             as.vertices[b + 2]);
        K26V3 pc = k26m3d_v3(as.vertices[c], as.vertices[c + 1],
                             as.vertices[c + 2]);
        K26V3 n = k26m3d_v3_norm(k26m3d_v3_cross(k26m3d_v3_sub(pb, pa),
                                                 k26m3d_v3_sub(pc, pa)));
        K26V3 nv = k26m3d_v3_norm(k26m3d_mat4_mul_dir(mv, n));
        double d = -k26m3d_v3_dot(nv, l);
        sf.a = f.a;
        sf.b = f.b;
        sf.c = f.c;
        sf.intensity = (float)(d > 0.0 ? d : 0.0);
        sf.drawn = f.a < e->behind.size() && f.b < e->behind.size() &&
                   f.c < e->behind.size() && !e->behind[f.a] &&
                   !e->behind[f.b] && !e->behind[f.c];
        e->faces.push_back(sf);
    }
}

}  /* namespace */

Camera::Camera()
    : mode(CAMERA_ORBIT), target(SCENE_ORIGIN), azimuth_deg(45.0),
      elevation_deg(20.0), radius(30.0), projection(PROJECTION_PERSPECTIVE),
      fov_y_deg(45.0), ortho_height(40.0), near_plane(0.1),
      far_plane(1.0e9)
{
    chase[0] = -30.0; chase[1] = 0.0; chase[2] = 8.0;
    eye[0] = 30.0; eye[1] = 30.0; eye[2] = 15.0;
    look[0] = 0.0; look[1] = 0.0; look[2] = 0.0;
    up[0] = 0.0; up[1] = 0.0; up[2] = 1.0;
}

SceneOptions::SceneOptions()
    : frame(SCENE_ORIGIN), shading(false), velocity_seconds(1.0),
      axis_length(1.0), thruster_scale(0.0025), spin_scale(10.0)
{
    for (int i = 0; i < ELEM_KIND_COUNT; i++)
        enabled[i] = true;
    /* Down the camera's own view direction, which is where a depth
     * cue wants its light and is the one direction that needs no
     * scene knowledge to choose. */
    light[0] = 0.0; light[1] = 0.0; light[2] = -1.0;
}

const char *element_name(ElementKind k)
{
    return (k >= 0 && k < ELEM_KIND_COUNT) ? ELEMENT_NAMES[k] : "?";
}

int element_by_name(const std::string &name)
{
    for (int i = 0; i < ELEM_KIND_COUNT; i++) {
        if (name == ELEMENT_NAMES[i])
            return i;
    }
    return -1;
}

std::string scene_body_name(const Spec &sp, uint32_t body)
{
    char buf[32];
    if (body == SCENE_ORIGIN)
        return "origin";
    if (body < sp.body_names.size() && !sp.body_names[body].empty())
        return sp.body_names[body];
    snprintf(buf, sizeof buf, "body%u", body);
    return buf;
}

bool scene_body_by_name(const Spec &sp, const std::string &name,
                        uint32_t *out)
{
    if (name == "origin") {
        *out = SCENE_ORIGIN;
        return true;
    }
    for (size_t i = 0; i < sp.body_names.size(); i++) {
        if (sp.body_names[i] == name) {
            *out = (uint32_t)i;
            return true;
        }
    }
    return false;
}

Scene scene_build(const SceneInput &in, const SceneOptions &o, uint32_t step)
{
    Scene sc;
    const Spec &sp = in.model->spec();
    const uint32_t bodies = (in.resim && in.resim->has_bodies)
                            ? in.resim->body_count
                            : (uint32_t)sp.body_names.size();
    std::vector<double> pos(bodies * 3, 0.0);
    std::vector<double> vel(bodies * 3, 0.0);
    std::vector<K26Quat> att(bodies, k26m3d_quat_identity());
    std::vector<K26V3> omg(bodies, k26m3d_v3(0.0, 0.0, 0.0));
    bool have_attitudes = false;
    K26M4 view, proj, pv;
    K26V3 eye, look, up;
    double aspect;

    sc.step = step;
    sc.frame = o.frame;
    sc.frame_name = scene_body_name(sp, o.frame);
    sc.available = true;

    /* Where the bodies are. With no artifact there is no answer: an
     * episode file records observation channels and a body's position
     * is not one of them unless the programme declared an observe for
     * it. The scene then draws the geometry it does have, at the
     * reference origin and unrotated, and says exactly that rather
     * than presenting a pose it does not know. */
    if (in.resim && in.resim->ran && in.resim->has_bodies &&
        step < in.resim->steps_compared) {
        size_t base = (size_t)step * bodies * 6;
        for (uint32_t b = 0; b < bodies; b++) {
            for (int c = 0; c < 3; c++) {
                size_t k = base + (size_t)b * 6 + c;
                if (k + 3 < in.resim->bodies.size()) {
                    pos[b * 3 + c] = in.resim->bodies[k];
                    vel[b * 3 + c] = in.resim->bodies[k + 3];
                }
            }
        }
        sc.pose_from_artifact = true;
        sc.message = "body poses re-simulated from the recorded action "
                     "stream";
    } else if (in.resim && in.resim->ran) {
        sc.message = "this artifact publishes no body state in the chosen "
                     "reference frame; the geometry is drawn at the "
                     "reference origin, unrotated";
    } else if (in.resim) {
        sc.message = "the rebuild did not run (" + in.resim->message +
                     "); the geometry is drawn at the reference origin, "
                     "unrotated";
    } else {
        /* Not "no artifact": the window reaches this branch with an
         * artifact supplied and its rebuild not yet asked for, and a
         * message naming a cause it has not checked would send a
         * reader to the wrong place. */
        sc.message = "no body pose: the recording carries no body position "
                     "or attitude. The geometry is drawn at the reference "
                     "origin, unrotated";
    }
    if (in.resim && in.resim->ran && in.resim->has_attitudes &&
        step < in.resim->steps_compared) {
        size_t base = (size_t)step * bodies * 7;
        for (uint32_t b = 0; b < bodies; b++) {
            size_t k = base + (size_t)b * 7;
            if (k + 3 < in.resim->attitudes.size()) {
                att[b] = k26m3d_quat(in.resim->attitudes[k + 1],
                                     in.resim->attitudes[k + 2],
                                     in.resim->attitudes[k + 3],
                                     in.resim->attitudes[k]);
            }
            if (k + 6 < in.resim->attitudes.size()) {
                omg[b] = k26m3d_v3(in.resim->attitudes[k + 4],
                                   in.resim->attitudes[k + 5],
                                   in.resim->attitudes[k + 6]);
            }
        }
        have_attitudes = true;
    } else if (sc.pose_from_artifact) {
        sc.message += "; this artifact publishes no attitude, so every "
                      "body is drawn unrotated";
    }
    /* The actuator drives at this step, when the artifact publishes
     * them: what the step into this state applied, which is what a
     * force line over this state should draw. An artifact without
     * the getter is reported, not hidden, exactly as an artifact
     * without attitudes is above. */
    if (sc.pose_from_artifact && in.resim && !in.resim->has_actuators) {
        sc.message += "; this artifact publishes no actuator drives, "
                      "so no imparted force is drawn";
    }
    const double *drv = 0;
    uint32_t drv_n = 0;
    if (in.resim && in.resim->ran && in.resim->has_actuators &&
        step < in.resim->steps_compared) {
        size_t base = (size_t)step * in.resim->actuator_count * 10;
        if (base + (size_t)in.resim->actuator_count * 10 <=
            in.resim->actuators.size()) {
            drv = &in.resim->actuators[base];
            drv_n = in.resim->actuator_count;
        }
    }

    /* The camera, resolved in the reference frame and in binary64. */
    {
        K26V3 target = k26m3d_v3(0, 0, 0);
        if (o.camera.target != SCENE_ORIGIN && o.camera.target < bodies) {
            target = k26m3d_v3(pos[o.camera.target * 3],
                               pos[o.camera.target * 3 + 1],
                               pos[o.camera.target * 3 + 2]);
        }
        up = k26m3d_v3_norm(v3_(o.camera.up));
        switch (o.camera.mode) {
        case CAMERA_CHASE: {
            K26Quat q = (o.camera.target != SCENE_ORIGIN &&
                         o.camera.target < bodies)
                        ? att[o.camera.target] : k26m3d_quat_identity();
            eye = k26m3d_v3_add(target,
                                k26m3d_quat_rotate_v3(q, v3_(o.camera.chase)));
            look = target;
            break;
        }
        case CAMERA_FREE:
            eye = v3_(o.camera.eye);
            look = v3_(o.camera.look);
            break;
        case CAMERA_ORBIT:
        default: {
            double az = k26m3d_deg2rad(o.camera.azimuth_deg);
            double el = k26m3d_deg2rad(o.camera.elevation_deg);
            eye = k26m3d_v3_add(target,
                k26m3d_v3(o.camera.radius * cos(el) * cos(az),
                          o.camera.radius * cos(el) * sin(az),
                          o.camera.radius * sin(el)));
            look = target;
            break;
        }
        }
        {
            /* Two degeneracies a view setting can reach, decided here
             * because deciding them is a visibility decision: an eye
             * sitting on its own target, which names no direction at
             * all, and an up vector parallel to the direction it is
             * meant to complete, which names no side. */
            K26V3 f = k26m3d_v3_sub(look, eye);
            if (k26m3d_v3_len(f) <= 0.0) {
                look = k26m3d_v3_add(eye, k26m3d_v3(0, 0, -1));
                f = k26m3d_v3(0, 0, -1);
            }
            f = k26m3d_v3_norm(f);
            if (k26m3d_v3_len(k26m3d_v3_cross(f, up)) < UP_DEGENERATE)
                up = (fabs(f.z) > 0.9) ? k26m3d_v3(1, 0, 0)
                                       : k26m3d_v3(0, 0, 1);
        }
        store_(sc.eye, eye);
        store_(sc.look, look);
        store_(sc.up, up);
    }

    /* The eye is already subtracted, so the view matrix is built at
     * the origin looking along the same direction. */
    k26m3d_mat4_look_at(&view, k26m3d_v3(0, 0, 0),
                        k26m3d_v3_sub(look, eye), up);
    aspect = o.viewport.height ? (double)o.viewport.width /
                                 (double)o.viewport.height : 1.0;
    if (o.camera.projection == PROJECTION_ORTHOGRAPHIC) {
        double h = 0.5 * o.camera.ortho_height;
        k26m3d_mat4_orthographic(&proj, -h * aspect, h * aspect, -h, h,
                                 o.camera.near_plane, o.camera.far_plane);
    } else {
        k26m3d_mat4_perspective(&proj, k26m3d_deg2rad(o.camera.fov_y_deg),
                                aspect, o.camera.near_plane,
                                o.camera.far_plane);
    }
    mat_to_double_(sc.view, &view);
    mat_to_double_(sc.proj, &proj);
    k26m3d_mat4_mul(&pv, &proj, &view);

    /* Per body, in the order a person debugging a flying craft asks
     * the questions: what shape it is, what it collides with, which
     * way it is pointing, where it is going, what it docks with, and
     * what it pushes with. A track belongs to no single body, being a
     * pair of them the spec does not name, so the tracks follow. */
    for (uint32_t b = 0; b < bodies; b++) {
        K26M4 model, rot, trans, mv, mvp;
        const Asset *as = asset_for_(in, b);
        bool drawable = as != 0;

        k26m3d_quat_to_mat4(&rot, att[b]);
        k26m3d_mat4_translate(&trans,
            k26m3d_v3(pos[b * 3] - eye.x, pos[b * 3 + 1] - eye.y,
                      pos[b * 3 + 2] - eye.z));
        k26m3d_mat4_mul(&model, &trans, &rot);
        k26m3d_mat4_mul(&mv, &view, &model);
        k26m3d_mat4_mul(&mvp, &pv, &model);

        if (drawable && o.enabled[ELEM_WIREFRAME] &&
            !as->edges.empty()) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_WIREFRAME;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/" + as->name;
            e.note = "the assembly's meshes, digest verified";
            for (size_t i = 0; i * 3 + 2 < as->vertices.size(); i++) {
                bl.point(k26m3d_v3(as->vertices[i * 3],
                                   as->vertices[i * 3 + 1],
                                   as->vertices[i * 3 + 2]));
            }
            for (size_t i = 0; i < as->edges.size(); i++)
                bl.link(as->edges[i].a, as->edges[i].b);
            finish_(&e, bl, &mv, &mvp, true);
            if (o.shading) {
                shade_(&e, *as, &mv, o.light);
                e.note += "; " + std::string(SHADING_LABEL);
            }
            sc.elements.push_back(e);
        }
        if (drawable && o.enabled[ELEM_COLLIDER] &&
            !as->colliders.empty()) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_COLLIDER;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/colliders";
            e.note = "the declared collision primitives, each "
                     "component's placement applied";
            for (size_t i = 0; i < as->colliders.size(); i++) {
                const Collider &c = as->colliders[i];
                if (c.kind == COLLIDER_BOX)
                    box_(&bl, v3_(c.centre), c.rot, c.a);
                else if (c.kind == COLLIDER_SPHERE)
                    sphere_(&bl, v3_(c.centre), c.radius);
                else
                    capsule_(&bl, v3_(c.a), v3_(c.b), c.radius);
            }
            finish_(&e, bl, &mv, &mvp, true);
            sc.elements.push_back(e);
        }
        if (o.enabled[ELEM_AXES]) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_AXES;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/axes";
            e.note = have_attitudes
                     ? "body frame x, y and z, from the recorded attitude"
                     : "body frame x, y and z; no attitude, so these are "
                       "the reference frame's own";
            for (int c = 0; c < 3; c++) {
                double d[3] = { 0.0, 0.0, 0.0 };
                d[c] = o.axis_length;
                bl.line(k26m3d_v3(0, 0, 0), v3_(d));
            }
            finish_(&e, bl, &mv, &mvp, true);
            sc.elements.push_back(e);
        }
        if (o.enabled[ELEM_PORT] && drawable && !as->ports.empty()) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_PORT;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/ports";
            e.note = "port geometry, with the mating plane and capture "
                     "limits its named envelope publishes";
            for (size_t i = 0; i < as->ports.size(); i++) {
                const Port &p = as->ports[i];
                K26V3 at = v3_(p.at);
                K26V3 ax = k26m3d_v3_norm(v3_(p.axis));
                K26V3 u = v3_(p.roll_ref);
                K26V3 w;
                double reach;
                u = k26m3d_v3_sub(u, k26m3d_v3_scale(ax,
                                                     k26m3d_v3_dot(u, ax)));
                if (k26m3d_v3_len(u) <= 0.0)
                    basis_(ax, &u, &w);
                u = k26m3d_v3_norm(u);
                w = k26m3d_v3_cross(ax, u);
                reach = p.has_envelope ? 0.5 * p.mating_diameter : 1.0;
                bl.line(at, k26m3d_v3_add(at, k26m3d_v3_scale(ax, reach)));
                bl.line(at, k26m3d_v3_add(at, k26m3d_v3_scale(u, reach)));
                if (!p.has_envelope)
                    continue;
                circle_(&bl, at, u, w, reach);
                /* The square plate is what the compiler builds and
                 * therefore what actually meets the other craft; the
                 * circle beside it is what the standard publishes.
                 * Both are drawn because the difference between them
                 * is a real difference in behaviour. */
                {
                    uint32_t c0 = bl.point(k26m3d_v3_add(at,
                        k26m3d_v3_add(k26m3d_v3_scale(u, reach),
                                      k26m3d_v3_scale(w, reach))));
                    uint32_t c1 = bl.point(k26m3d_v3_add(at,
                        k26m3d_v3_add(k26m3d_v3_scale(u, -reach),
                                      k26m3d_v3_scale(w, reach))));
                    uint32_t c2 = bl.point(k26m3d_v3_add(at,
                        k26m3d_v3_add(k26m3d_v3_scale(u, -reach),
                                      k26m3d_v3_scale(w, -reach))));
                    uint32_t c3 = bl.point(k26m3d_v3_add(at,
                        k26m3d_v3_add(k26m3d_v3_scale(u, reach),
                                      k26m3d_v3_scale(w, -reach))));
                    bl.link(c0, c1);
                    bl.link(c1, c2);
                    bl.link(c2, c3);
                    bl.link(c3, c0);
                }
                circle_(&bl, at, u, w, p.lateral_limit);
                {
                    double half = k26m3d_deg2rad(p.pitchyaw_limit_deg);
                    double len = p.mating_diameter;
                    double rr = len * tan(half);
                    K26V3 tip = k26m3d_v3_add(at, k26m3d_v3_scale(ax, len));
                    for (int q = 0; q < 4; q++) {
                        double t = 0.5 * K26M3D_PI * (double)q;
                        K26V3 off = k26m3d_v3_add(
                            k26m3d_v3_scale(u, rr * cos(t)),
                            k26m3d_v3_scale(w, rr * sin(t)));
                        bl.line(at, k26m3d_v3_add(tip, off));
                    }
                    circle_(&bl, tip, u, w, rr);
                }
            }
            finish_(&e, bl, &mv, &mvp, true);
            sc.elements.push_back(e);
        }
        if (o.enabled[ELEM_THRUSTER] && drawable &&
            !as->thrusters.empty()) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_THRUSTER;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/thrusters";
            e.note = "thruster mounting points and force directions, at "
                     "the declared metres of line per newton";
            for (size_t i = 0; i < as->thrusters.size(); i++) {
                const Thruster &t = as->thrusters[i];
                K26V3 at = v3_(t.at);
                K26V3 d = k26m3d_v3_norm(v3_(t.dir));
                bl.line(at, k26m3d_v3_add(at,
                    k26m3d_v3_scale(d, t.thrust * o.thruster_scale)));
            }
            finish_(&e, bl, &mv, &mvp, true);
            sc.elements.push_back(e);
        }
        /* The imparted thrust, from the artifact's actuator getter:
         * the force the step into this state applied, at the same
         * scale the mounts use so the two read against each other.
         * Only thruster records draw; a wheel or a torquer is a
         * command about the centre of mass, not a force at a point,
         * and a line would assert a geometry it does not have. */
        if (o.enabled[ELEM_FORCE] && drv_n) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_FORCE;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/force";
            e.note = "imparted thrust over the step into this state, "
                     "at the declared metres of line per newton";
            for (uint32_t i = 0; i < drv_n; i++) {
                const double *d = drv + (size_t)i * 10;
                if ((uint32_t)d[0] != b || d[1] != 2.0 || d[8] == 0.0)
                    continue;
                K26V3 at = k26m3d_v3(d[2], d[3], d[4]);
                K26V3 dir = k26m3d_v3_norm(k26m3d_v3(d[5], d[6], d[7]));
                bl.line(at, k26m3d_v3_add(at,
                    k26m3d_v3_scale(dir, d[8] * o.thruster_scale)));
            }
            if (!bl.empty()) {
                finish_(&e, bl, &mv, &mvp, true);
                sc.elements.push_back(e);
            }
        }
        /* The angular velocity, drawn in the body frame the getter
         * reports it in: the rotation axis, its length the rate at
         * the declared scale. */
        if (o.enabled[ELEM_SPIN] && have_attitudes &&
            k26m3d_v3_len(omg[b]) > 0.0) {
            SceneElement e;
            Builder bl;
            e.kind = ELEM_SPIN;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/spin";
            e.note = "body-frame angular velocity, at the declared "
                     "metres of line per radian per second";
            bl.line(k26m3d_v3(0.0, 0.0, 0.0),
                    k26m3d_v3_scale(omg[b], o.spin_scale));
            finish_(&e, bl, &mv, &mvp, true);
            sc.elements.push_back(e);
        }
        /* The velocity vector is in the reference frame and not the
         * body's, so it takes the frame's own model matrix. It is the
         * body's velocity relative to the reference body, matching
         * the positions beside it: the getter returns each body's own
         * velocity and says the reference does not affect it, so the
         * relative form is taken here and named in the note. */
        if (o.enabled[ELEM_VELOCITY] && sc.pose_from_artifact) {
            SceneElement e;
            Builder bl;
            K26V3 v = k26m3d_v3(vel[b * 3], vel[b * 3 + 1], vel[b * 3 + 2]);
            K26V3 p = k26m3d_v3(pos[b * 3] - eye.x, pos[b * 3 + 1] - eye.y,
                                pos[b * 3 + 2] - eye.z);
            if (o.frame != SCENE_ORIGIN && o.frame < bodies) {
                v = k26m3d_v3_sub(v, k26m3d_v3(vel[o.frame * 3],
                                               vel[o.frame * 3 + 1],
                                               vel[o.frame * 3 + 2]));
            }
            e.kind = ELEM_VELOCITY;
            e.body = b;
            e.name = scene_body_name(sp, b) + "/velocity";
            e.note = o.frame == SCENE_ORIGIN
                     ? "the body's own velocity, over the declared seconds"
                     : "velocity relative to the reference body, over the "
                       "declared seconds";
            bl.line(p, k26m3d_v3_add(p,
                k26m3d_v3_scale(v, o.velocity_seconds)));
            finish_(&e, bl, &view, &pv, false);
            sc.elements.push_back(e);
        }
    }

    /* The treatment a body with an assembly but no mesh already gets
     * elsewhere: draw what it does have and say what it does not,
     * rather than inventing geometry for it. Said once per such
     * assembly, because a recording may bind more than one. */
    for (size_t i = 0; i < in.assets.size(); i++) {
        const Asset *a = in.assets[i].asset;
        if (a && a->loaded && a->edges.empty()) {
            sc.message += "; " + scene_body_name(sp, in.assets[i].body) +
                          (a->mesh_vertices == 0
                           ? " declares no mesh, so it draws its "
                             "colliders and its axes"
                           : " declares mesh vertices but no faces, so "
                             "no wireframe can be drawn from it; it "
                             "draws its colliders and its axes");
        }
    }
    if (o.enabled[ELEM_TRAJECTORY] && in.episode) {
        const std::vector<Trajectory> &tr = in.model->trajectories();
        for (size_t t = 0; t < tr.size(); t++) {
            SceneElement e;
            Builder bl;
            uint32_t prev = 0;
            uint32_t last = step < in.episode->step_count
                            ? step : in.episode->step_count;
            e.kind = ELEM_TRAJECTORY;
            e.body = SCENE_NO_BODY;
            e.name = tr[t].base;
            e.note = SCENE_TRAJECTORY_LABEL;
            for (uint32_t i = 0; i <= last && i < in.episode->step_count;
                 i++) {
                double xyz[3];
                uint32_t idx;
                Model::point(*in.episode, sp, tr[t], i, xyz);
                idx = bl.point(k26m3d_v3(xyz[0] - eye.x, xyz[1] - eye.y,
                                         xyz[2] - eye.z));
                if (i)
                    bl.link(prev, idx);
                prev = idx;
            }
            /* A one-step track is one point and no line, and it is
             * still an element the model knows about: the dump says
             * so rather than leaving a reader to wonder whether the
             * toggle worked. */
            if (bl.empty())
                continue;
            finish_(&e, bl, &view, &pv, false);
            sc.elements.push_back(e);
        }
    }
    /* The detection line of sight, which is the element that joins
     * the scene to the detection channels: it appears and disappears
     * as a payload gains and loses its target, so a recording plays
     * back the difficulty of the task in one picture.
     *
     * It is built at every step the channels exist for and it is
     * drawn only where the recorded flag is 1.0. Building it always
     * and suppressing the segment is deliberate: an element that
     * vanished would leave a reader unable to tell a target that is
     * not seen from a toggle that is off or a channel set that is
     * absent.
     *
     * Where it starts is a limit rather than a choice. The recording
     * publishes the channels of a detection and never the body that
     * carried the payload, so the line starts at the reference
     * frame's origin, on the same footing as the track above, and the
     * note says the condition under which that is the observer. */
    if (o.enabled[ELEM_DETECTION] && in.episode) {
        const std::vector<Detection> &dt = in.model->detections();
        for (size_t d = 0; d < dt.size(); d++) {
            SceneElement e;
            Builder bl;
            Trajectory ray;
            double xyz[3];
            double flag;
            size_t k;

            if (step >= in.episode->step_count)
                break;
            ray.base = dt[d].base;
            ray.dir_x = dt[d].dir_x;
            ray.dir_y = dt[d].dir_y;
            ray.dir_z = dt[d].dir_z;
            ray.range = dt[d].range;
            Model::point(*in.episode, sp, ray, step, xyz);
            k = (size_t)step * sp.obs_total + dt[d].detected;
            flag = k < in.episode->obs.size() ? in.episode->obs[k] : 0.0;
            e.kind = ELEM_DETECTION;
            e.body = SCENE_NO_BODY;
            e.name = dt[d].base;
            e.note = SCENE_DETECTION_LABEL;
            bl.line(k26m3d_v3(-eye.x, -eye.y, -eye.z),
                    k26m3d_v3(xyz[0] - eye.x, xyz[1] - eye.y,
                              xyz[2] - eye.z));
            finish_(&e, bl, &view, &pv, false);
            if (flag != 1.0) {
                for (size_t s = 0; s < e.segments.size(); s++)
                    e.segments[s].drawn = false;
            }
            sc.elements.push_back(e);
        }
    }
    return sc;
}

}  /* namespace k26rl_view */
