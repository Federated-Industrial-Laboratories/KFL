/* test_rl_scene.c: the three-dimensional scene view.
 *
 * A picture is the hardest thing in this tree to gate, because the
 * obvious check is a person looking at it. This gate does not look at
 * it. Everything between a body's state and a pixel lives in the
 * viewer's model and is written out by its projection dump, and that
 * is what is checked here: for a named camera pose, a named
 * projection and a named viewport, every drawn element's vertices in
 * normalised device coordinates, with a per-vertex behind-the-eye
 * flag. No window, no context and no display is opened anywhere.
 *
 * The expected coordinates are derived here from the camera and
 * projection parameters, written out from their definitions, and not
 * recorded from the viewer's own output. Each derivation arm also
 * computes what a plausible wrong implementation would produce and
 * asserts the produced value is not that, so an arm that agreed with
 * everything would fail its own credibility check.
 *
 * Gates:
 *   1. Projection: every vertex of two bodies at two different poses
 *      equals the independently derived coordinate, and does not
 *      equal what a transposed camera basis would give.
 *   2. Rotation side: a non-symmetric craft at a quaternion that is
 *      not a multiple of a right angle projects as the position times
 *      the rotation, not the rotation times the position, and the two
 *      are shown to differ on this fixture.
 *   3. Depth ordering: two bodies order in the depth component by
 *      which is nearer, and moving the camera to the other side
 *      swaps the order.
 *   4. Behind the eye: a body behind the camera has every vertex
 *      flagged and no segment drawn, while a body in front of it has
 *      neither.
 *   5. The camera-relative narrowing: the same scene expressed about
 *      a body and about a world origin 7000 km away projects the
 *      same, which a world-space single-precision path does not, and
 *      the arm computes how far off that path would be.
 *   6. Element toggles: one arm per element, each present when asked
 *      for and absent when not.
 *   7. The shaded depth cue: off by default, and on it varies face to
 *      face and carries, wherever it appears, the statement that it
 *      is a depth cue and not an illumination calculation.
 *   8. The digest: an asset whose bytes are not the recording's draws
 *      nothing at all in the scene, and an assembly with no mesh
 *      draws its colliders and its axes and says so.
 *   9. The scene changes nothing: the recording's bytes are untouched
 *      by every run above, every other panel's output is identical
 *      with the scene at full and with it off, the re-simulation
 *      verdict is unchanged, and two identical runs agree exactly.
 *
 * Skips (77) when the sibling stack archives or the viewer binary are
 * not built.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <math.h>
#include <stdarg.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_scene_test"
#define VIEWER "../tools/k26rl_view/k26rl_view"

/* The fixture's own numbers, in one place so an arm cannot disagree
 * with the command line that produced its dump. */
#define AXIS_LEN   1.0
#define FOV_DEG    40.0
#define VIEW_W     800
#define VIEW_H     600
#define NEAR_PLANE 0.5
#define FAR_PLANE  1.0e9
/* Normalised device coordinates are computed in single precision,
 * because single precision is what the pipeline they are going to
 * runs in. The derivation here is in binary64, so the two agree to
 * about a float epsilon over the scene's own scale and not exactly.
 * Every arm that asserts a disagreement asserts one hundreds of times
 * this size. */
#define NDC_TOL    5.0e-5

/* The chaser's attitude: 37 degrees about (1, 2, 3) normalised, which
 * is not a multiple of a right angle and is not about a coordinate
 * axis, so no wrong rotation order coincides with the right one. */
static const char *const SCENE_KFL =
    "form RL_SCENE\n"
    "fn world scene_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body chaser assembly=\"scene_craft.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=0.94832365520619932 quat_x=0.084803236535420032"
    " quat_y=0.16960647307084006 quat_z=0.25440970960626014"
    " omega_x=0.0 omega_y=0.0 omega_z=0.0\n"
    /* A second body, at a different position and a different
     * attitude. With one body the projection arm cannot tell a right
     * transform from one that is right for a single pose. */
    "    astro_body target assembly=\"scene_target.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=40.0 pos_z=0.0 vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 6\n"
    "        terminated when episode.steps > 4\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    on_step\n"
    "        chaser.vel_x = chaser.vel_x + push * 0.01\n"
    "    end\n"
    "    observe target from chaser mode=geometric as rel\n"
    "    objective\n"
    "        reward rel_range / 100.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* An asymmetric tetrahedron. No reflection and no right-angle
 * rotation leaves it unchanged, so a transform applied on the wrong
 * side moves every vertex of it. */
static const char *const CRAFT_MESH =
    "# an asymmetric tetrahedron, so a wrong transform cannot hide\n"
    "v 2.0 0.0 0.0\n"
    "v -1.5 -1.0 -0.4\n"
    "v -1.5 1.2 -0.4\n"
    "v -1.0 0.0 0.9\n"
    "f 1 2 3\n"
    "f 1 3 4\n"
    "f 1 4 2\n"
    "f 2 4 3\n";

static const char *const POD_MESH =
    "# a second, smaller tetrahedron on a second component\n"
    "v 0.6 0.0 0.0\n"
    "v -0.4 -0.3 -0.2\n"
    "v -0.4 0.35 -0.2\n"
    "v -0.3 0.0 0.45\n"
    "f 1 2 3\n"
    "f 1 3 4\n"
    "f 1 4 2\n"
    "f 2 4 3\n";

/* Three components, two of them with meshes and one without, and one
 * collider of each of the three shapes the format carries: a scene
 * arm that saw one shape would be an arm that had checked one shape.
 * The second component is placed away from the origin AND rotated,
 * because a component at the origin lets a reader that ignores the
 * placement produce the right picture. */
static const char *const CRAFT_ASM =
    "assembly scene_craft\n"
    "    frame x_to_port\n"
    "    provenance mass \"gate fixture, not a craft\" computed\n"
    "    provenance geometry \"gate fixture, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0.0 0.0 0.0\n"
    "        mesh scene_craft.k26mesh\n"
    "        collider box 2.0 1.2 0.9\n"
    "    end\n"
    "    component pod\n"
    "        mass 200.0\n"
    "        at -3.0 0.5 1.0\n"
    "        rotate 0.70710678118654752 0.0 0.70710678118654752 0.0\n"
    "        mesh scene_pod.k26mesh\n"
    "        collider sphere 0.0 0.0 0.0 0.75\n"
    "    end\n"
    "    component boom\n"
    "        mass 50.0\n"
    "        at 0.0 -2.0 0.0\n"
    "        collider capsule 0.0 0.0 0.0 0.0 -1.5 0.0 0.25\n"
    "    end\n"
    "    port forward\n"
    "        at 3.5 0.0 0.0\n"
    "        axis 1.0 0.0 0.0\n"
    "        roll_ref 0.0 1.0 0.0\n"
    "        capture idss_e\n"
    "    end\n"
    "    thruster fore\n"
    "        at -2.0 1.0 0.0\n"
    "        dir 1.0 0.0 0.0\n"
    "        thrust 400.0\n"
    "    end\n"
    "    thruster aft\n"
    "        at 2.0 1.0 0.0\n"
    "        dir -1.0 0.0 0.0\n"
    "        thrust 400.0\n"
    "    end\n"
    "end\n";

/* A body with an assembly and no mesh, which is the case the scene
 * has to treat as drawing its colliders and its axes and saying so.
 */
static const char *const TARGET_ASM =
    "assembly scene_target\n"
    "    frame x_to_port\n"
    "    provenance mass \"gate fixture, not a craft\" computed\n"
    "    provenance geometry \"gate fixture, not a craft\" computed\n"
    "    component hull\n"
    "        mass 5000.0\n"
    "        at 0.0 0.0 0.0\n"
    "        collider box 3.0 3.0 3.0\n"
    "    end\n"
    "    port dock\n"
    "        at -3.1 0.0 0.0\n"
    "        axis -1.0 0.0 0.0\n"
    "        roll_ref 0.0 1.0 0.0\n"
    "        capture idss_e\n"
    "    end\n"
    "end\n";

/* ---- running the viewer and reading its dumps ---------------------- */

static void run_viewer_(const char *args, const char *out_path)
{
    char cmd[4096];
    int n = snprintf(cmd, sizeof cmd, "%s %s > %s 2>%s.err", VIEWER, args,
                     out_path, out_path);
    int rc;

    ASSERT((size_t)n < sizeof cmd);
    rc = system(cmd);
    if (rc != 0) {
        char show[512];
        fprintf(stderr, "viewer failed (rc=%d): %s %s\n", rc, VIEWER, args);
        snprintf(show, sizeof show, "cat %s.err", out_path);
        (void)!system(show);
        exit(1);
    }
}

static char *slurp_(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    char *buf;
    long n;

    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    n = ftell(f);
    ASSERT(n >= 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    buf = malloc((size_t)n + 1);
    ASSERT(buf != NULL);
    ASSERT(fread(buf, 1, (size_t)n, f) == (size_t)n);
    buf[n] = '\0';
    fclose(f);
    if (out_len)
        *out_len = (size_t)n;
    return buf;
}

/* A binary64 written as its bit pattern, which is how every value in
 * these dumps travels. */
static double hexd_(const char *s)
{
    uint64_t bits = strtoull(s, NULL, 16);
    double v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

/* The line beginning with `prefix`, or 0. The prefixes this gate
 * builds carry every leading field, so a match is exact rather than
 * the first line that happens to start the same way. */
static const char *line_(const char *text, const char *prefix)
{
    size_t plen = strlen(prefix);
    const char *p = text;

    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, prefix, plen) == 0)
            return p + plen;
        p = eol ? eol + 1 : NULL;
    }
    return NULL;
}

/* The index of the element of that kind belonging to that body, or -1.
 * The body is spelled as the dump spells it, which is the spec's own
 * name, or "-" for an element that belongs to no body. */
static int element_index_(const char *text, unsigned k, unsigned step,
                          const char *kind, const char *body)
{
    char pre[64];
    const char *p = text;
    size_t plen;

    snprintf(pre, sizeof pre, "scene_element %u %u ", k, step);
    plen = strlen(pre);
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, pre, plen) == 0) {
            unsigned idx = 0;
            char gk[32], gb[64];
            if (sscanf(p + plen, "%u %31s %63s", &idx, gk, gb) == 3 &&
                strcmp(gk, kind) == 0 && strcmp(gb, body) == 0)
                return (int)idx;
        }
        p = eol ? eol + 1 : NULL;
    }
    return -1;
}

static void element_counts_(const char *text, unsigned k, unsigned step,
                            int idx, unsigned *nv, unsigned *nseg,
                            unsigned *nface)
{
    char pre[64];
    const char *p;

    snprintf(pre, sizeof pre, "scene_element %u %u %d ", k, step, idx);
    p = line_(text, pre);
    ASSERT(p != NULL);
    {
        char gk[32], gb[64];
        ASSERT(sscanf(p, "%31s %63s %u %u %u", gk, gb, nv, nseg, nface) == 5);
    }
}

static void vertex_(const char *text, unsigned k, unsigned step, int idx,
                    unsigned v, double ndc[3], int *behind)
{
    char pre[80];
    char a[24], b[24], c[24];
    unsigned bh = 0;
    const char *p;

    snprintf(pre, sizeof pre, "scene_vertex %u %u %d %u ", k, step, idx, v);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%23s %23s %23s %u", a, b, c, &bh) == 4);
    ndc[0] = hexd_(a);
    ndc[1] = hexd_(b);
    ndc[2] = hexd_(c);
    if (behind)
        *behind = (int)bh;
}

static int segment_drawn_(const char *text, unsigned k, unsigned step,
                          int idx, unsigned sgi)
{
    char pre[80];
    unsigned a, b;
    int drawn = -1;
    const char *p;

    snprintf(pre, sizeof pre, "scene_segment %u %u %d %u ", k, step, idx, sgi);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%u %u %d", &a, &b, &drawn) == 3);
    return drawn;
}

static void world_body_(const char *text, unsigned k, unsigned step,
                        unsigned body, double pos[3], double vel[3])
{
    char pre[64];
    char h[6][24];
    const char *p;
    int i;

    snprintf(pre, sizeof pre, "world %u %u %u ", k, step, body);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%23s %23s %23s %23s %23s %23s", h[0], h[1], h[2],
                  h[3], h[4], h[5]) == 6);
    for (i = 0; i < 3; i++) {
        pos[i] = hexd_(h[i]);
        vel[i] = hexd_(h[i + 3]);
    }
}

static void attitude_body_(const char *text, unsigned k, unsigned step,
                           unsigned body, double q[4])
{
    char pre[64];
    char h[7][24];
    const char *p;
    int i;

    snprintf(pre, sizeof pre, "attitude %u %u %u ", k, step, body);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%23s %23s %23s %23s %23s %23s %23s", h[0], h[1], h[2],
                  h[3], h[4], h[5], h[6]) == 7);
    for (i = 0; i < 4; i++)
        q[i] = hexd_(h[i]);
}

static void asset_vertex_(const char *text, unsigned v, double p3[3])
{
    char pre[48];
    char h[3][24];
    const char *p;
    int i;

    snprintf(pre, sizeof pre, "wireframe_vertex %u ", v);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%23s %23s %23s", h[0], h[1], h[2]) == 3);
    for (i = 0; i < 3; i++)
        p3[i] = hexd_(h[i]);
}

/* ---- the derivation, written from its definitions ------------------ */

static void cross_(const double a[3], const double b[3], double out[3])
{
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

static double dot_(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void norm_(double a[3])
{
    double n = sqrt(dot_(a, a));
    int i;
    ASSERT(n > 0.0);
    for (i = 0; i < 3; i++)
        a[i] /= n;
}

/* v rotated by the quaternion (w, x, y, z), scalar first. */
static void qrot_(const double q[4], const double v[3], double out[3])
{
    double u[3] = { q[1], q[2], q[3] };
    double t[3], w[3];
    int i;

    cross_(u, v, t);
    for (i = 0; i < 3; i++)
        t[i] *= 2.0;
    cross_(u, t, w);
    for (i = 0; i < 3; i++)
        out[i] = v[i] + q[0] * t[i] + w[i];
}

/* The camera a projection arm sets, and the coordinates it derives. */
typedef struct {
    double eye[3];
    double look[3];
    double up[3];
    double fov_deg;
    double aspect;
    double near_plane;
    double far_plane;
} Cam;

/* The normalised device coordinate of a world point, from the
 * definitions of a right-handed look-at and a right-handed
 * perspective with the OpenGL depth convention. `transposed` builds
 * the camera basis the wrong way round, which is the arm's own check
 * that it can tell a right transform from a plausible wrong one. */
static void project_(const Cam *c, const double p[3], int transposed,
                     double ndc[3])
{
    double f[3], s[3], u[3], d[3];
    double vx, vy, vz, t, cw, nf;
    int i;

    for (i = 0; i < 3; i++) {
        f[i] = c->look[i] - c->eye[i];
        d[i] = p[i] - c->eye[i];
        u[i] = c->up[i];
    }
    norm_(f);
    cross_(f, u, s);
    norm_(s);
    cross_(s, f, u);
    if (!transposed) {
        vx = dot_(s, d);
        vy = dot_(u, d);
        vz = -dot_(f, d);
    } else {
        double r[3];
        r[0] = s[0]; r[1] = u[0]; r[2] = -f[0];
        vx = dot_(r, d);
        r[0] = s[1]; r[1] = u[1]; r[2] = -f[1];
        vy = dot_(r, d);
        r[0] = s[2]; r[1] = u[2]; r[2] = -f[2];
        vz = dot_(r, d);
    }
    t = 1.0 / tan(c->fov_deg * (M_PI / 180.0) * 0.5);
    nf = 1.0 / (c->near_plane - c->far_plane);
    cw = -vz;
    ASSERT(cw != 0.0);
    ndc[0] = (t / c->aspect) * vx / cw;
    ndc[1] = t * vy / cw;
    ndc[2] = ((c->far_plane + c->near_plane) * nf * vz +
              2.0 * c->far_plane * c->near_plane * nf) / cw;
}

static double worst_(const double a[3], const double b[3])
{
    double m = 0.0;
    int i;
    for (i = 0; i < 3; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > m)
            m = d;
    }
    return m;
}

/* The argument string every projection arm shares, so the camera the
 * dump used is the camera the derivation used. */
static void cam_args_(char *out, size_t n, const Cam *c, const char *tail)
{
    int w = snprintf(out, n,
        "--camera free --eye %.17g,%.17g,%.17g --look %.17g,%.17g,%.17g "
        "--up %.17g,%.17g,%.17g --projection perspective --fov %.17g "
        "--clip %.17g,%.17g --viewport %dx%d --axis-length %.17g %s",
        c->eye[0], c->eye[1], c->eye[2], c->look[0], c->look[1], c->look[2],
        c->up[0], c->up[1], c->up[2], c->fov_deg, c->near_plane,
        c->far_plane, VIEW_W, VIEW_H, AXIS_LEN, tail);
    ASSERT((size_t)w < n);
}

/* Every line beginning with `prefix`, in order, so one panel's output
 * can be lifted out of a dump that carried several. */
static char *lines_with_(const char *text, const char *prefix)
{
    size_t n = strlen(text);
    size_t plen = strlen(prefix);
    char *out = malloc(n + 1);
    size_t w = 0;
    const char *p = text;

    ASSERT(out != NULL);
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) + 1 : strlen(p);
        if (len >= plen && memcmp(p, prefix, plen) == 0) {
            memcpy(out + w, p, len);
            w += len;
        }
        p = eol ? eol + 1 : NULL;
    }
    out[w] = '\0';
    return out;
}

/* Strip the scene's own lines, so what is left is every other panel's
 * output and can be compared byte for byte. */
static char *without_scene_(const char *text)
{
    size_t n = strlen(text);
    char *out = malloc(n + 1);
    size_t w = 0;
    const char *p = text;

    ASSERT(out != NULL);
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) + 1 : strlen(p);
        if (strncmp(p, "scene", 5) != 0) {
            memcpy(out + w, p, len);
            w += len;
        }
        p = eol ? eol + 1 : NULL;
    }
    out[w] = '\0';
    return out;
}

int main(void)
{
    char args[4096], cmd[4096], sel[256];
    char *scene = NULL, *world = NULL, *att = NULL, *wire = NULL;
    double qc[4], qt[4], pc[3], vc[3], pt[3], vt[3];
    const unsigned K = 0, STEP = 2;
    Cam cam;
    int idx, i;

    if (!rl_libs_present_("test_rl_scene"))
        return 77;
    if (!rl_file_exists_(VIEWER)) {
        fprintf(stderr, "test_rl_scene: skip: %s not built\n", VIEWER);
        return 77;
    }
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/scene_craft.k26mesh", CRAFT_MESH);
    rl_write_file_(WORK_DIR "/scene_pod.k26mesh", POD_MESH);
    rl_write_file_(WORK_DIR "/scene_craft.k26asm", CRAFT_ASM);
    rl_write_file_(WORK_DIR "/scene_target.k26asm", TARGET_ASM);
    rl_write_file_(WORK_DIR "/scene.kfl", SCENE_KFL);
    rl_compile_(WORK_DIR "/scene.kfl", WORK_DIR "/scene", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/scene.rlenv.so"));

    /* Two environments and two episodes each, so the panels this gate
     * compares have more than one of everything to disagree about. */
    rl_run_or_die_(WORK_DIR "/scene --envs 2 --episodes 2 --seed 5"
                   " --out " WORK_DIR "/scene.k26epi > " WORK_DIR
                   "/run.log 2>&1");
    ASSERT(rl_file_exists_(WORK_DIR "/scene.k26epi"));

    /* The recording as it stands before the viewer has ever seen it.
     * Gate 8 compares against this. */
    {
        size_t n = 0;
        char *bytes = slurp_(WORK_DIR "/scene.k26epi", &n);
        FILE *f = fopen(WORK_DIR "/before.bin", "wb");
        ASSERT(f != NULL);
        ASSERT(fwrite(bytes, 1, n, f) == n);
        fclose(f);
        free(bytes);
    }

#define ARTIFACT " --artifact " WORK_DIR "/scene.rlenv.so "
#define CRAFT    " --asset " WORK_DIR "/scene_craft.k26asm "
#define EPISODE  WORK_DIR "/scene.k26epi"

    snprintf(cmd, sizeof cmd, "--dump world --episode %u --steps %u:%u"
             ARTIFACT EPISODE, K, STEP, STEP + 1);
    run_viewer_(cmd, WORK_DIR "/world.txt");
    world = slurp_(WORK_DIR "/world.txt", NULL);
    snprintf(cmd, sizeof cmd, "--dump attitude --episode %u --steps %u:%u"
             ARTIFACT EPISODE, K, STEP, STEP + 1);
    run_viewer_(cmd, WORK_DIR "/att.txt");
    att = slurp_(WORK_DIR "/att.txt", NULL);
    snprintf(cmd, sizeof cmd, "--dump wireframe" CRAFT EPISODE);
    run_viewer_(cmd, WORK_DIR "/wire.txt");
    wire = slurp_(WORK_DIR "/wire.txt", NULL);

    world_body_(world, K, STEP, 1, pc, vc);
    world_body_(world, K, STEP, 2, pt, vt);
    attitude_body_(att, K, STEP, 1, qc);
    attitude_body_(att, K, STEP, 2, qt);
    /* The two bodies must actually be at different poses, or an arm
     * that reports two agreements has checked one. */
    ASSERT(fabs(pc[1] - pt[1]) > 1.0);
    ASSERT(fabs(qc[0] - qt[0]) > 0.01);
    printf("test_rl_scene: fixture: two bodies %.3f m apart, attitudes"
           " w=%.6f and w=%.6f: OK\n", fabs(pc[1] - pt[1]), qc[0], qt[0]);

    /* ---- gate 1: the projection, derived independently ------------- */
    {
        unsigned nv, nseg, nface, v;
        double worst = 0.0, worst_wrong = 1.0e30;
        int checked = 0;

        cam.eye[0] = pc[0] + 26.0;
        cam.eye[1] = pc[1] + 18.0;
        cam.eye[2] = pc[2] + 11.0;
        for (i = 0; i < 3; i++)
            cam.look[i] = pc[i];
        cam.up[0] = 0.0; cam.up[1] = 0.0; cam.up[2] = 1.0;
        cam.fov_deg = FOV_DEG;
        cam.aspect = (double)VIEW_W / (double)VIEW_H;
        cam.near_plane = NEAR_PLANE;
        cam.far_plane = FAR_PLANE;
        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements wireframe,axes" ARTIFACT CRAFT,
                 K, STEP, STEP + 1);
        cam_args_(args, sizeof args, &cam, EPISODE);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g1.txt");
        }
        scene = slurp_(WORK_DIR "/g1.txt", NULL);

        /* The craft's own geometry, in the body frame, carried to the
         * world by its position and its attitude. */
        idx = element_index_(scene, K, STEP, "wireframe", "chaser");
        ASSERT(idx >= 0);
        element_counts_(scene, K, STEP, idx, &nv, &nseg, &nface);
        ASSERT(nv == 8 && nseg == 12);
        for (v = 0; v < nv; v++) {
            double body[3], rot[3], world_p[3], want[3], wrong[3], got[3];
            int behind = 0;
            asset_vertex_(wire, v, body);
            qrot_(qc, body, rot);
            for (i = 0; i < 3; i++)
                world_p[i] = pc[i] + rot[i];
            project_(&cam, world_p, 0, want);
            project_(&cam, world_p, 1, wrong);
            vertex_(scene, K, STEP, idx, v, got, &behind);
            ASSERT(behind == 0);
            if (worst_(got, want) > worst)
                worst = worst_(got, want);
            if (worst_(want, wrong) < worst_wrong)
                worst_wrong = worst_(want, wrong);
            ASSERT(worst_(got, want) < NDC_TOL);
            checked++;
        }
        /* Both bodies' axes, which is the second pose. */
        {
            const char *bodies[2] = { "chaser", "target" };
            const double *qs[2] = { qc, qt };
            const double *ps[2] = { pc, pt };
            int b;
            for (b = 0; b < 2; b++) {
                int ai = element_index_(scene, K, STEP, "axes", bodies[b]);
                ASSERT(ai >= 0);
                element_counts_(scene, K, STEP, ai, &nv, &nseg, &nface);
                ASSERT(nv == 6 && nseg == 3);
                for (v = 0; v < nv; v++) {
                    double body[3] = { 0.0, 0.0, 0.0 };
                    double rot[3], world_p[3], want[3], wrong[3], got[3];
                    if (v & 1)
                        body[v / 2] = AXIS_LEN;
                    qrot_(qs[b], body, rot);
                    for (i = 0; i < 3; i++)
                        world_p[i] = ps[b][i] + rot[i];
                    project_(&cam, world_p, 0, want);
                    project_(&cam, world_p, 1, wrong);
                    vertex_(scene, K, STEP, ai, v, got, NULL);
                    if (worst_(got, want) > worst)
                        worst = worst_(got, want);
                    if (worst_(want, wrong) < worst_wrong)
                        worst_wrong = worst_(want, wrong);
                    ASSERT(worst_(got, want) < NDC_TOL);
                    checked++;
                }
            }
        }
        /* The arm's own credibility: the wrong basis it compared
         * against has to actually disagree on this fixture. */
        ASSERT(worst_wrong > 100.0 * NDC_TOL);
        printf("gate 1: %d projected vertices over two bodies at two poses"
               " equal the derived coordinate, worst %.3g, a transposed"
               " basis differing by at least %.3g: OK\n",
               checked, worst, worst_wrong);
        free(scene);
        scene = NULL;
    }

    /* ---- gate 2: which side the rotation is applied ---------------- */
    {
        unsigned nv, nseg, nface, v;
        double worst = 0.0, closest_wrong = 1.0e30;

        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements wireframe" ARTIFACT CRAFT,
                 K, STEP, STEP + 1);
        cam_args_(args, sizeof args, &cam, EPISODE);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g2.txt");
        }
        scene = slurp_(WORK_DIR "/g2.txt", NULL);
        idx = element_index_(scene, K, STEP, "wireframe", "chaser");
        ASSERT(idx >= 0);
        element_counts_(scene, K, STEP, idx, &nv, &nseg, &nface);
        for (v = 0; v < nv; v++) {
            double body[3], rot[3], want[3];
            double conj[4], other[3], wrongp[3], wrong[3];
            asset_vertex_(wire, v, body);
            qrot_(qc, body, rot);
            for (i = 0; i < 3; i++)
                want[i] = pc[i] + rot[i];
            /* The two ways to get this wrong that a reader of the
             * source would not see: the inverse rotation, and the
             * rotation applied to the translated point. */
            conj[0] = qc[0];
            for (i = 1; i < 4; i++)
                conj[i] = -qc[i];
            qrot_(conj, body, other);
            for (i = 0; i < 3; i++)
                wrongp[i] = pc[i] + other[i];
            {
                double wantn[3], gotn[3], wrongn[3], sump[3], rotsum[3];
                project_(&cam, want, 0, wantn);
                project_(&cam, wrongp, 0, wrongn);
                vertex_(scene, K, STEP, idx, v, gotn, NULL);
                if (worst_(gotn, wantn) > worst)
                    worst = worst_(gotn, wantn);
                ASSERT(worst_(gotn, wantn) < NDC_TOL);
                if (worst_(wantn, wrongn) < closest_wrong)
                    closest_wrong = worst_(wantn, wrongn);
                for (i = 0; i < 3; i++)
                    sump[i] = pc[i] + body[i];
                qrot_(qc, sump, rotsum);
                project_(&cam, rotsum, 0, wrong);
                if (worst_(wantn, wrong) < closest_wrong)
                    closest_wrong = worst_(wantn, wrong);
            }
        }
        ASSERT(closest_wrong > 100.0 * NDC_TOL);
        printf("gate 2: the non-symmetric craft at a %.1f degree attitude"
               " projects as position times rotation, worst %.3g; the"
               " inverse rotation and the wrong composition order differ"
               " by at least %.3g: OK\n",
               2.0 * acos(qc[0]) * (180.0 / M_PI), worst, closest_wrong);
        free(scene);
        scene = NULL;
    }

    /* ---- gate 3: depth ordering ------------------------------------ */
    {
        double near_side[3], far_side[3];
        double mid[3];
        int ia, ib;
        double za, zb, za2, zb2;

        for (i = 0; i < 3; i++)
            mid[i] = 0.5 * (pc[i] + pt[i]);
        for (i = 0; i < 3; i++) {
            cam.look[i] = mid[i];
            cam.eye[i] = mid[i];
        }
        /* Along the line joining the two bodies, so one is squarely
         * in front of the other. */
        cam.eye[1] = mid[1] - 120.0;
        cam_args_(args, sizeof args, &cam, EPISODE);
        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements axes" ARTIFACT, K, STEP,
                 STEP + 1);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g3a.txt");
        }
        scene = slurp_(WORK_DIR "/g3a.txt", NULL);
        ia = element_index_(scene, K, STEP, "axes", "chaser");
        ib = element_index_(scene, K, STEP, "axes", "target");
        ASSERT(ia >= 0 && ib >= 0);
        vertex_(scene, K, STEP, ia, 0, near_side, NULL);
        vertex_(scene, K, STEP, ib, 0, far_side, NULL);
        za = near_side[2];
        zb = far_side[2];
        free(scene);

        cam.eye[1] = mid[1] + 120.0;
        cam_args_(args, sizeof args, &cam, EPISODE);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g3b.txt");
        }
        scene = slurp_(WORK_DIR "/g3b.txt", NULL);
        ia = element_index_(scene, K, STEP, "axes", "chaser");
        ib = element_index_(scene, K, STEP, "axes", "target");
        ASSERT(ia >= 0 && ib >= 0);
        vertex_(scene, K, STEP, ia, 0, near_side, NULL);
        vertex_(scene, K, STEP, ib, 0, far_side, NULL);
        za2 = near_side[2];
        zb2 = far_side[2];
        free(scene);
        scene = NULL;
        /* The chaser is the nearer of the two from one side and the
         * farther from the other, and the depth component says so
         * both times. */
        ASSERT(za < zb);
        ASSERT(za2 > zb2);
        ASSERT(fabs(za - zb) > 1.0e-9);
        ASSERT(fabs(za2 - zb2) > 1.0e-9);
        printf("gate 3: depth orders %.9f before %.9f from one side and"
               " %.9f after %.9f from the other: OK\n", za, zb, za2, zb2);
    }

    /* ---- gate 4: behind the eye ------------------------------------ */
    {
        unsigned nv, nseg, nface, v;
        int ic, it, behind, drawn_behind = 0, drawn_front = 0;

        /* Past the chaser and looking on toward the target, so one
         * body is behind the eye and one in front of it in the same
         * picture. */
        for (i = 0; i < 3; i++) {
            cam.eye[i] = pc[i];
            cam.look[i] = pt[i];
        }
        cam.eye[1] = pc[1] + 5.0;
        cam_args_(args, sizeof args, &cam, EPISODE);
        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements axes" ARTIFACT, K, STEP,
                 STEP + 1);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g4.txt");
        }
        scene = slurp_(WORK_DIR "/g4.txt", NULL);
        ic = element_index_(scene, K, STEP, "axes", "chaser");
        it = element_index_(scene, K, STEP, "axes", "target");
        ASSERT(ic >= 0 && it >= 0);
        element_counts_(scene, K, STEP, ic, &nv, &nseg, &nface);
        for (v = 0; v < nv; v++) {
            double ndc[3];
            vertex_(scene, K, STEP, ic, v, ndc, &behind);
            ASSERT(behind == 1);
        }
        for (v = 0; v < nseg; v++)
            drawn_behind += segment_drawn_(scene, K, STEP, ic, v);
        element_counts_(scene, K, STEP, it, &nv, &nseg, &nface);
        for (v = 0; v < nv; v++) {
            double ndc[3];
            vertex_(scene, K, STEP, it, v, ndc, &behind);
            ASSERT(behind == 0);
        }
        for (v = 0; v < nseg; v++)
            drawn_front += segment_drawn_(scene, K, STEP, it, v);
        ASSERT(drawn_behind == 0);
        ASSERT(drawn_front == (int)nseg && nseg > 0);
        printf("gate 4: the body behind the eye has every vertex flagged"
               " and %d of its segments drawn; the body in front has %d of"
               " %u drawn: OK\n", drawn_behind, drawn_front, nseg);
        free(scene);
        scene = NULL;

        /* The same camera under an orthographic projection, where
         * every clip w is one and a test taken from the clip w would
         * flag nothing at all. The flag is taken in view space for
         * exactly this reason, so this arm is what says so. */
        cam_args_(args, sizeof args, &cam,
                  "--projection orthographic --ortho-height 60 " EPISODE);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g4o.txt");
        }
        scene = slurp_(WORK_DIR "/g4o.txt", NULL);
        ASSERT(line_(scene, "scene_projection orthographic") != NULL);
        ic = element_index_(scene, K, STEP, "axes", "chaser");
        it = element_index_(scene, K, STEP, "axes", "target");
        ASSERT(ic >= 0 && it >= 0);
        drawn_behind = 0;
        drawn_front = 0;
        element_counts_(scene, K, STEP, ic, &nv, &nseg, &nface);
        for (v = 0; v < nv; v++) {
            double ndc[3];
            vertex_(scene, K, STEP, ic, v, ndc, &behind);
            ASSERT(behind == 1);
        }
        for (v = 0; v < nseg; v++)
            drawn_behind += segment_drawn_(scene, K, STEP, ic, v);
        element_counts_(scene, K, STEP, it, &nv, &nseg, &nface);
        for (v = 0; v < nv; v++) {
            double ndc[3];
            vertex_(scene, K, STEP, it, v, ndc, &behind);
            ASSERT(behind == 0);
        }
        for (v = 0; v < nseg; v++)
            drawn_front += segment_drawn_(scene, K, STEP, it, v);
        ASSERT(drawn_behind == 0);
        ASSERT(drawn_front == (int)nseg && nseg > 0);
        printf("gate 4: under an orthographic projection, where every clip"
               " w is one, the body behind the eye is still flagged and"
               " still not drawn: OK\n");
        free(scene);
        scene = NULL;
    }

    /* ---- gate 5: the camera-relative narrowing --------------------- */
    {
        char *near_dump, *far_dump;
        Cam near_cam, far_cam;
        double worst = 0.0, worst_world = 0.0;
        int elems, e, checked = 0;
        const char *kinds[5] = { "wireframe", "collider", "axes", "port",
                                 "thruster" };
        const char *bodies[3] = { "earth", "chaser", "target" };

        /* The same scene twice: once about the chaser, where every
         * coordinate is metres, and once about the world origin 7000
         * km away, where a single-precision world-space pipeline
         * cannot hold a craft. */
        near_cam = cam;
        /* Close in, because the failure this arm names is a craft
         * collapsing into the quantisation of its own coordinates,
         * and a camera far enough away hides it. */
        near_cam.eye[0] = 10.0;
        near_cam.eye[1] = 7.0;
        near_cam.eye[2] = 5.0;
        near_cam.look[0] = 0.0;
        near_cam.look[1] = 0.0;
        near_cam.look[2] = 0.0;
        far_cam = near_cam;
        for (i = 0; i < 3; i++) {
            far_cam.eye[i] = near_cam.eye[i] + pc[i];
            far_cam.look[i] = near_cam.look[i] + pc[i];
        }
        ASSERT(sqrt(dot_(pc, pc)) > 6.9e6);

        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame chaser --elements wireframe,collider,axes,port,"
                 "thruster" ARTIFACT CRAFT, K, STEP, STEP + 1);
        cam_args_(args, sizeof args, &near_cam, EPISODE);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g5near.txt");
        }
        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements wireframe,collider,axes,port,"
                 "thruster" ARTIFACT CRAFT, K, STEP, STEP + 1);
        cam_args_(args, sizeof args, &far_cam, EPISODE);
        {
            char full[4096];
            int w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g5far.txt");
        }
        near_dump = slurp_(WORK_DIR "/g5near.txt", NULL);
        far_dump = slurp_(WORK_DIR "/g5far.txt", NULL);
        {
            char pre[64];
            const char *p;
            snprintf(pre, sizeof pre, "scene_elements %u %u ", K, STEP);
            p = line_(near_dump, pre);
            ASSERT(p != NULL);
            elems = atoi(p);
            p = line_(far_dump, pre);
            ASSERT(p != NULL);
            ASSERT(atoi(p) == elems);
        }
        ASSERT(elems > 0);
        for (e = 0; e < elems; e++) {
            unsigned nv, nseg, nface, v;
            element_counts_(near_dump, K, STEP, e, &nv, &nseg, &nface);
            for (v = 0; v < nv; v++) {
                double a[3], b[3];
                int ba = 0, bb = 0;
                vertex_(near_dump, K, STEP, e, v, a, &ba);
                vertex_(far_dump, K, STEP, e, v, b, &bb);
                ASSERT(ba == bb);
                if (worst_(a, b) > worst)
                    worst = worst_(a, b);
                ASSERT(worst_(a, b) < NDC_TOL);
                checked++;
            }
        }
        (void)kinds;
        (void)bodies;
        /* What a world-space single-precision path would have done
         * instead: narrow the position before the eye is subtracted,
         * which is the one step this design exists to avoid. The arm
         * asserts that path is visibly wrong, so the agreement above
         * is a result and not an identity. */
        {
            unsigned nv, nseg, nface, v;
            int wi = element_index_(far_dump, K, STEP, "wireframe",
                                    "chaser");
            ASSERT(wi >= 0);
            element_counts_(far_dump, K, STEP, wi, &nv, &nseg, &nface);
            for (v = 0; v < nv; v++) {
                double body[3], rot[3], world_p[3], flat[3];
                double flatn[3], wantn[3];
                asset_vertex_(wire, v, body);
                qrot_(qc, body, rot);
                for (i = 0; i < 3; i++) {
                    world_p[i] = pc[i] + rot[i];
                    flat[i] = (double)(float)world_p[i];
                }
                project_(&far_cam, world_p, 0, wantn);
                project_(&far_cam, flat, 0, flatn);
                if (worst_(wantn, flatn) > worst_world)
                    worst_world = worst_(wantn, flatn);
            }
        }
        ASSERT(worst_world > 100.0 * NDC_TOL);
        printf("gate 5: %d vertices agree across a %.0f km translation of"
               " the whole scene, worst %.3g; narrowing before the"
               " subtraction would move them by up to %.3g: OK\n",
               checked, sqrt(dot_(pc, pc)) / 1000.0, worst, worst_world);
        free(near_dump);
        free(far_dump);
    }

    /* ---- gate 6: one arm per element toggle ------------------------ */
    {
        static const char *const names[] = {
            "wireframe", "collider", "axes", "trajectory", "velocity",
            "port", "thruster"
        };
        const int n_names = (int)(sizeof names / sizeof names[0]);
        int j;

        for (i = 0; i < 3; i++) {
            cam.eye[i] = pc[i];
            cam.look[i] = pc[i];
        }
        cam.eye[0] += 26.0;
        cam.eye[1] += 18.0;
        cam.eye[2] += 11.0;
        cam_args_(args, sizeof args, &cam, EPISODE);
        for (j = 0; j < n_names; j++) {
            char full[4096];
            int w, present, other = 0;
            snprintf(sel, sizeof sel, "--dump scene --episode %u --steps"
                     " %u:%u --frame origin --elements %s" ARTIFACT CRAFT,
                     K, STEP, STEP + 1, names[j]);
            w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g6.txt");
            scene = slurp_(WORK_DIR "/g6.txt", NULL);
            {
                char pre[80];
                snprintf(pre, sizeof pre, "scene_toggle %s 1", names[j]);
                ASSERT(line_(scene, pre) != NULL);
            }
            present = 0;
            for (i = 0; i < n_names; i++) {
                char pre[96];
                int found;
                snprintf(pre, sizeof pre, " %s ", names[i]);
                /* Element lines name their kind in the fourth field,
                 * which is what this counts; a substring anywhere
                 * else on the line would be a different word. */
                found = 0;
                {
                    const char *p = scene;
                    char head[64];
                    size_t hlen;
                    snprintf(head, sizeof head, "scene_element %u %u ", K,
                             STEP);
                    hlen = strlen(head);
                    while (p && *p) {
                        const char *eol = strchr(p, '\n');
                        size_t len = eol ? (size_t)(eol - p) : strlen(p);
                        if (len >= hlen && memcmp(p, head, hlen) == 0) {
                            unsigned e2;
                            char gk[32];
                            if (sscanf(p + hlen, "%u %31s", &e2, gk) == 2 &&
                                strcmp(gk, names[i]) == 0)
                                found++;
                        }
                        p = eol ? eol + 1 : NULL;
                    }
                }
                if (i == j)
                    present = found;
                else
                    other += found;
            }
            ASSERT(present > 0);
            ASSERT(other == 0);
            free(scene);
            scene = NULL;
            printf("gate 6: `%s` alone yields %d element(s) of that kind"
                   " and none of any other: OK\n", names[j], present);
        }
        {
            char full[4096];
            int w;
            snprintf(sel, sizeof sel, "--dump scene --episode %u --steps"
                     " %u:%u --frame origin --elements none" ARTIFACT CRAFT,
                     K, STEP, STEP + 1);
            w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g6none.txt");
            scene = slurp_(WORK_DIR "/g6none.txt", NULL);
            {
                char pre[64];
                const char *p;
                snprintf(pre, sizeof pre, "scene_elements %u %u ", K, STEP);
                p = line_(scene, pre);
                ASSERT(p != NULL);
                ASSERT(atoi(p) == 0);
            }
            free(scene);
            scene = NULL;
            snprintf(sel, sizeof sel, "--dump scene --episode %u --steps"
                     " %u:%u --frame origin --elements all" ARTIFACT CRAFT,
                     K, STEP, STEP + 1);
            w = snprintf(full, sizeof full, "%s %s", sel, args);
            ASSERT((size_t)w < sizeof full);
            run_viewer_(full, WORK_DIR "/g6all.txt");
            scene = slurp_(WORK_DIR "/g6all.txt", NULL);
            for (j = 0; j < n_names; j++) {
                char pre[80];
                snprintf(pre, sizeof pre, "scene_toggle %s 1", names[j]);
                ASSERT(line_(scene, pre) != NULL);
            }
            free(scene);
            scene = NULL;
            printf("gate 6: every element off yields an empty scene and"
                   " every element on names all %d: OK\n", n_names);
        }
    }

    /* ---- gate 7: the shaded depth cue and its standing label ------- */
    {
        char full[4096];
        int w;
        unsigned nv, nseg, nface, f, shaded_faces;
        double lo = 2.0, hi = -1.0;

        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements wireframe --shading"
                 ARTIFACT CRAFT, K, STEP, STEP + 1);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g7s.txt");
        scene = slurp_(WORK_DIR "/g7s.txt", NULL);
        /* The obligation this arm exists for: the shaded view says
         * what it is wherever it appears, because a shaded craft
         * beside a detection channel reads as radiometry unless it
         * says otherwise. */
        ASSERT(strstr(scene, "scene_shading_label ") != NULL);
        ASSERT(strstr(scene, "not an illumination calculation") != NULL);
        idx = element_index_(scene, K, STEP, "wireframe", "chaser");
        ASSERT(idx >= 0);
        element_counts_(scene, K, STEP, idx, &nv, &nseg, &nface);
        ASSERT(nface > 1);
        shaded_faces = nface;
        for (f = 0; f < nface; f++) {
            char pre[80];
            char h[24];
            unsigned a, b, c;
            int drawn = -1;
            double v;
            const char *p;
            snprintf(pre, sizeof pre, "scene_face %u %u %d %u ", K, STEP,
                     idx, f);
            p = line_(scene, pre);
            ASSERT(p != NULL);
            ASSERT(sscanf(p, "%u %u %u %23s %d", &a, &b, &c, h, &drawn) == 5);
            v = hexd_(h);
            ASSERT(v >= 0.0 && v <= 1.0);
            if (v < lo)
                lo = v;
            if (v > hi)
                hi = v;
        }
        /* A depth cue that gave every face the same value would be a
         * depth cue that cued nothing. */
        ASSERT(hi - lo > 0.01);
        {
            char pre[80];
            const char *p;
            snprintf(pre, sizeof pre, "scene_note %u %u %d ", K, STEP, idx);
            p = line_(scene, pre);
            ASSERT(p != NULL);
            ASSERT(strncmp(p, "the assembly's meshes", 21) == 0);
            ASSERT(strstr(p, "not an illumination calculation") != NULL);
        }
        free(scene);
        scene = NULL;

        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements wireframe" ARTIFACT CRAFT,
                 K, STEP, STEP + 1);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g7n.txt");
        scene = slurp_(WORK_DIR "/g7n.txt", NULL);
        idx = element_index_(scene, K, STEP, "wireframe", "chaser");
        ASSERT(idx >= 0);
        element_counts_(scene, K, STEP, idx, &nv, &nseg, &nface);
        ASSERT(nface == 0);
        ASSERT(strstr(scene, "scene_shading 0 ") != NULL);
        free(scene);
        scene = NULL;
        printf("gate 7: shading is off by default, and on it gives %u faces"
               " with intensities spanning %.3f to %.3f, carrying the"
               " statement that it is a depth cue and not an illumination"
               " calculation: OK\n", shaded_faces, lo, hi);
    }

    /* ---- gate 8: the digest, and an assembly with no mesh ---------- */
    {
        char full[4096];
        int w;

        rl_run_or_die_("sed 's/^v 2.0 0.0 0.0/v 2.0 0.0 0.001/' " WORK_DIR
                       "/scene_craft.k26mesh > " WORK_DIR "/spoiled.k26mesh");
        rl_run_or_die_("sed 's/mesh scene_craft.k26mesh/mesh spoiled"
                       ".k26mesh/' " WORK_DIR "/scene_craft.k26asm > "
                       WORK_DIR "/spoiled.k26asm");
        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements all" ARTIFACT
                 " --asset " WORK_DIR "/spoiled.k26asm ", K, STEP, STEP + 1);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g7.txt");
        scene = slurp_(WORK_DIR "/g7.txt", NULL);
        ASSERT(line_(scene, "scene_asset_verdict mismatch") != NULL);
        ASSERT(element_index_(scene, K, STEP, "wireframe", "chaser") < 0);
        ASSERT(element_index_(scene, K, STEP, "collider", "chaser") < 0);
        ASSERT(element_index_(scene, K, STEP, "port", "chaser") < 0);
        ASSERT(element_index_(scene, K, STEP, "thruster", "chaser") < 0);
        /* The elements that do not come from the asset are unaffected,
         * so the arm is measuring the refusal and not an empty scene. */
        ASSERT(element_index_(scene, K, STEP, "axes", "chaser") >= 0);
        free(scene);
        scene = NULL;
        printf("gate 8: an asset whose bytes are not the recording's draws"
               " no wireframe, no collider, no port and no thruster, and"
               " the axes are still drawn: OK\n");

        snprintf(sel, sizeof sel, "--dump scene --episode %u --steps %u:%u"
                 " --frame origin --elements all" ARTIFACT
                 " --asset " WORK_DIR "/scene_target.k26asm ", K, STEP,
                 STEP + 1);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g7b.txt");
        scene = slurp_(WORK_DIR "/g7b.txt", NULL);
        ASSERT(line_(scene, "scene_asset_verdict drawable target") != NULL);
        ASSERT(element_index_(scene, K, STEP, "wireframe", "target") < 0);
        ASSERT(element_index_(scene, K, STEP, "collider", "target") >= 0);
        ASSERT(element_index_(scene, K, STEP, "port", "target") >= 0);
        ASSERT(strstr(scene, "declares no mesh") != NULL);
        free(scene);
        scene = NULL;
        printf("gate 8: an assembly with no mesh draws its collider outline"
               " and its axes and the scene says so: OK\n");
    }

    /* ---- gate 9: the scene changes nothing ------------------------- */
    {
        char full[4096];
        char *on, *off, *on2, *stripped_on, *stripped_off, *other;
        int w;

        snprintf(sel, sizeof sel, "--dump all --episode %u --elements all"
                 " --shading --frame origin" ARTIFACT CRAFT, K);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g8on.txt");
        run_viewer_(full, WORK_DIR "/g8on2.txt");
        snprintf(sel, sizeof sel, "--dump all --episode %u --elements none"
                 " --frame origin" ARTIFACT CRAFT, K);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g8off.txt");
        snprintf(sel, sizeof sel, "--dump all --episode %u --elements none"
                 " --frame origin" ARTIFACT CRAFT, K + 1);
        w = snprintf(full, sizeof full, "%s %s", sel, args);
        ASSERT((size_t)w < sizeof full);
        run_viewer_(full, WORK_DIR "/g8other.txt");

        on = slurp_(WORK_DIR "/g8on.txt", NULL);
        on2 = slurp_(WORK_DIR "/g8on2.txt", NULL);
        off = slurp_(WORK_DIR "/g8off.txt", NULL);
        other = slurp_(WORK_DIR "/g8other.txt", NULL);
        /* The same run twice is the same bytes, scene included. */
        ASSERT(strcmp(on, on2) == 0);
        stripped_on = without_scene_(on);
        stripped_off = without_scene_(off);
        ASSERT(strcmp(stripped_on, stripped_off) == 0);
        /* Every panel still reports what it reported, and the
         * reconstruction is still bitwise equal to the recording. */
        ASSERT(strstr(stripped_on, "resim_verdict 0 equal-bitwise") != NULL);
        ASSERT(strstr(stripped_off, "resim_verdict 0 equal-bitwise") != NULL);
        /* The comparison's own credibility: it has to be able to say
         * two dumps differ, which a comparison of two equal things
         * never shows. */
        {
            char *stripped_other = without_scene_(other);
            ASSERT(strcmp(stripped_on, stripped_other) != 0);
            free(stripped_other);
        }
        /* And the recording is byte for byte what it was before the
         * viewer had ever opened it. */
        {
            size_t na = 0, nb = 0;
            char *a = slurp_(WORK_DIR "/before.bin", &na);
            char *b = slurp_(WORK_DIR "/scene.k26epi", &nb);
            ASSERT(na == nb);
            ASSERT(memcmp(a, b, na) == 0);
            free(a);
            free(b);
            printf("gate 9: the recording's %u bytes are unchanged by every"
                   " run above: OK\n", (unsigned)na);
        }
        printf("gate 9: the scene at full and the scene off leave every"
               " other panel's %u bytes identical, the reconstruction"
               " bitwise equal, and two identical runs byte for byte the"
               " same; a different episode's dump differs: OK\n",
               (unsigned)strlen(stripped_on));

        /* The scene never built at all, against the scene built at
         * full: the re-simulation panel runs after the scene, so if
         * building a scene left anything behind, this is where it
         * would show. */
        {
            char *alone, *a_lines, *b_lines;
            snprintf(sel, sizeof sel, "--dump resim --episode %u" ARTIFACT
                     EPISODE, K);
            run_viewer_(sel, WORK_DIR "/g8resim.txt");
            alone = slurp_(WORK_DIR "/g8resim.txt", NULL);
            a_lines = lines_with_(alone, "resim");
            b_lines = lines_with_(on, "resim");
            ASSERT(strlen(a_lines) > 0);
            ASSERT(strcmp(a_lines, b_lines) == 0);
            printf("gate 9: the re-simulation panel's %u bytes are the same"
                   " with the scene never built and with it built at full:"
                   " OK\n", (unsigned)strlen(a_lines));
            free(alone);
            free(a_lines);
            free(b_lines);
        }

        /* And the traffic in the other direction: a scene asked for a
         * body's frame gets that frame's body states, whether it is
         * the only panel running or the last of a dozen that asked
         * for a different frame first. */
        {
            char *together, *only, *t_lines, *o_lines;
            char fullb[4096];
            int w2;
            snprintf(sel, sizeof sel, "--dump all --episode %u --frame"
                     " chaser --elements axes" ARTIFACT CRAFT, K);
            w2 = snprintf(fullb, sizeof fullb, "%s %s", sel, args);
            ASSERT((size_t)w2 < sizeof fullb);
            run_viewer_(fullb, WORK_DIR "/g8fa.txt");
            snprintf(sel, sizeof sel, "--dump scene --episode %u --frame"
                     " chaser --elements axes" ARTIFACT CRAFT, K);
            w2 = snprintf(fullb, sizeof fullb, "%s %s", sel, args);
            ASSERT((size_t)w2 < sizeof fullb);
            run_viewer_(fullb, WORK_DIR "/g8fb.txt");
            together = slurp_(WORK_DIR "/g8fa.txt", NULL);
            only = slurp_(WORK_DIR "/g8fb.txt", NULL);
            t_lines = lines_with_(together, "scene_vertex");
            o_lines = lines_with_(only, "scene_vertex");
            ASSERT(strlen(o_lines) > 0);
            ASSERT(strcmp(t_lines, o_lines) == 0);
            printf("gate 9: the scene in a body's frame projects the same"
                   " %u bytes alone and after every other panel has asked"
                   " for the world origin: OK\n", (unsigned)strlen(o_lines));
            free(together);
            free(only);
            free(t_lines);
            free(o_lines);
        }
        free(on);
        free(on2);
        free(off);
        free(other);
        free(stripped_on);
        free(stripped_off);
    }

    free(world);
    free(att);
    free(wire);
    printf("test_rl_scene: 9 gates passed\n");
    return 0;
}
