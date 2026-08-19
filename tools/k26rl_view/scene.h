/* scene.h - the three-dimensional scene, entirely in the model.
 *
 * Every transform, every projection, every cull and every visibility
 * decision this view makes happens here, where the headless dump
 * reaches it. The window receives arrays of coordinates and a matrix
 * per element and issues draw calls; it computes nothing.
 *
 * That split is not tidiness. The interesting part of a scene view is
 * exactly the arithmetic between a body's state and a pixel, and
 * arithmetic inside the window is arithmetic no gate can run without
 * a display. Putting it here makes a scene testable on a machine with
 * no graphics at all, which is what the projection dump beside this
 * is for.
 *
 * Scale, and why the scene is camera relative. An orbital scene spans
 * a craft's centimetres and an orbit's thousands of kilometres, and a
 * single-precision vertex pipeline holds neither together with the
 * other: at 7000 km from the origin a 32-bit float's spacing is about
 * half a metre, so a craft's geometry would collapse into its own
 * quantisation before it reached a screen. So positions are held in
 * binary64 in a chosen reference frame, the camera eye is subtracted
 * in binary64, and only the result, which is small by construction,
 * narrows to the single precision the graphics pipeline takes. The
 * narrowing happens once, at the last step.
 */
#ifndef K26RL_VIEW_SCENE_H
#define K26RL_VIEW_SCENE_H

#include <stdint.h>

#include <string>
#include <vector>

#include "asset.h"
#include "model.h"
#include "resim.h"

namespace k26rl_view {

/* The drawn elements, in the order a body's elements are built.
 *
 * Each is independently toggleable and each says where it came from.
 * The list is what a person debugging a flying craft looks at: what
 * shape it is, what it collides with, which way it is pointing, where
 * it has been, where it is going, what it can push with, and what it
 * is trying to dock to. */
enum ElementKind {
    ELEM_WIREFRAME = 0,   /* the assembly's meshes, digest verified */
    ELEM_COLLIDER,        /* the body's collision primitives */
    ELEM_AXES,            /* the body's attitude, as three short lines */
    ELEM_TRAJECTORY,      /* the reconstructed observer-relative track */
    ELEM_VELOCITY,        /* the body's velocity, at a declared scale */
    ELEM_PORT,            /* port geometry and its capture envelope */
    ELEM_THRUSTER,        /* thruster positions and directions */
    ELEM_KIND_COUNT
};

/* The element's name as the toggles and the dump spell it. */
const char *element_name(ElementKind k);

/* The element of that name, or -1. */
int element_by_name(const std::string &name);

/* Three camera modes, which are the three a replay viewer uses. */
enum CameraMode {
    CAMERA_ORBIT = 0,     /* eye on a sphere about a chosen target */
    CAMERA_CHASE,         /* eye offset from a body, in that body's frame */
    CAMERA_FREE           /* eye and target set directly */
};

enum ProjectionKind {
    PROJECTION_PERSPECTIVE = 0,
    PROJECTION_ORTHOGRAPHIC
};

/* The world origin, as a reference frame or as a camera target. It is
 * the stepping surface's own value rather than a second one beside
 * it, so the frame the scene shows is the frame the body getter was
 * asked for. */
#define SCENE_ORIGIN   K26RL_BODY_REF_ORIGIN

/* An element belonging to no body, which a track does. It shares the
 * value above and not its meaning: one says which frame, the other
 * says there is no body to name. */
#define SCENE_NO_BODY  K26RL_BODY_REF_ORIGIN

/* A camera is a pose and a projection, and both are model state. */
struct Camera {
    CameraMode mode;
    uint32_t target;          /* body index, or SCENE_ORIGIN */
    double azimuth_deg;       /* orbit */
    double elevation_deg;     /* orbit */
    double radius;            /* orbit, metres */
    double chase[3];          /* chase, metres in the target's own frame */
    double eye[3];            /* free, metres in the reference frame */
    double look[3];           /* free */
    double up[3];
    ProjectionKind projection;
    double fov_y_deg;         /* perspective */
    double ortho_height;      /* orthographic, metres */
    double near_plane;
    double far_plane;

    Camera();
};

struct Viewport {
    uint32_t width;
    uint32_t height;

    Viewport() : width(1280), height(720) {}
};

/* Everything the view chooses, all of it dumpable. */
struct SceneOptions {
    Camera camera;
    Viewport viewport;
    uint32_t frame;                 /* reference body, or SCENE_ORIGIN */
    bool enabled[ELEM_KIND_COUNT];
    bool shading;
    double light[3];                /* view-space light direction */
    double velocity_seconds;        /* the velocity vector's declared scale */
    double axis_length;             /* the body axes' declared length, m */
    double thruster_scale;          /* metres of line per newton */

    SceneOptions();
};

/* One segment of a line set, and whether the model handed it to the
 * window. A segment with an endpoint at or behind the eye plane is
 * culled here rather than left for the pipeline, so the window never
 * receives geometry it would have to decide about. */
struct SceneSegment {
    uint32_t a;
    uint32_t b;
    bool drawn;
};

/* One triangle of the optional shaded pass, with the depth cue's
 * intensity. Not an illumination calculation: see SHADING_LABEL. */
struct SceneFace {
    uint32_t a;
    uint32_t b;
    uint32_t c;
    float intensity;
    bool drawn;
};

/* One drawn thing: its identity, its geometry in its own local frame,
 * the one matrix that carries that frame to clip space, and the
 * projection of every vertex.
 *
 * `local` is what a vertex buffer holds and it does not change with
 * the camera, so a mesh uploads once and is reused; `mvp` is what
 * changes, and it is a uniform. */
struct SceneElement {
    ElementKind kind;
    uint32_t body;                  /* source body, or SCENE_NO_BODY */
    std::string name;
    std::string note;               /* what this element is, for the panel */
    std::vector<float> local;       /* 3 per vertex */
    /* Whether `local` is fixed in a body's own frame, and therefore a
     * buffer the window uploads once and reuses, or is already
     * relative to the camera eye and changes whenever the eye does.
     * The distinction is the model's because the reason for it is:
     * the camera-relative narrowing has to happen per point for
     * anything whose local frame is the scene's rather than a
     * body's. */
    bool local_static;
    std::vector<SceneSegment> segments;
    std::vector<SceneFace> faces;
    float mvp[16];                  /* column major, as GL takes it */
    std::vector<float> ndc;         /* 3 per vertex */
    std::vector<uint8_t> behind;    /* 1 per vertex, at or behind the eye */
};

struct Scene {
    bool available;
    std::string message;            /* why not, or what limits this scene */
    bool pose_from_artifact;        /* false when bodies sit at the origin */
    uint32_t step;
    uint32_t frame;
    std::string frame_name;
    double eye[3];                  /* resolved, in the reference frame */
    double look[3];
    double up[3];
    double view[16];                /* the camera-relative view matrix */
    double proj[16];
    std::vector<SceneElement> elements;

    Scene() : available(false), pose_from_artifact(false), step(0),
              frame(SCENE_ORIGIN) {}
};

/* What the scene is built from. `resim` supplies the body poses and
 * may be null, in which case the bodies sit at the reference origin
 * with no attitude and the scene says so. `asset` supplies geometry
 * and is drawn only when `asset_body` names the body that binds it,
 * which the digest check decides. */
struct SceneInput {
    const Model *model;
    const Episode *episode;
    const ResimResult *resim;
    const Asset *asset;
    uint32_t asset_body;

    SceneInput() : model(0), episode(0), resim(0), asset(0),
                   asset_body(SCENE_NO_BODY) {}
};

/* The standing statement the shaded view carries, in the panel and in
 * the dump. It is not decoration: this capability contains detection
 * models whose whole purpose is radiometry, and a user who sees a
 * shaded craft beside one of their channels is entitled to assume the
 * two are related unless told they are not. */
extern const char *const SHADING_LABEL;

/* The trajectory element's standing statement, for the same reason
 * the plotted trajectory carries one. */
extern const char *const SCENE_TRAJECTORY_LABEL;

/* Build the scene at one step. Deterministic in its inputs: the same
 * recording, asset bytes, camera, projection and viewport give the
 * same scene, bit for bit, within one binary. */
Scene scene_build(const SceneInput &in, const SceneOptions &o, uint32_t step);

/* A body's name as the spec publishes it, or a stand-in. */
std::string scene_body_name(const Spec &sp, uint32_t body);

/* Resolve a body name. "origin" resolves to SCENE_ORIGIN, the world
 * origin the body getter takes.
 *
 * The answer travels through the out parameter and the verdict
 * through the return, because SCENE_ORIGIN is a valid answer and
 * cannot also mean "no such body": a caller that read one as the
 * other would quietly draw the world origin whenever a name was
 * misspelled. */
bool scene_body_by_name(const Spec &sp, const std::string &name,
                        uint32_t *out);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_SCENE_H */
