/* test_rl_docking.c: the docking benchmark, end to end.
 *
 * The programme under test is the shipped example, compiled from the
 * tree rather than written here, so what is measured is what a reader
 * of the documentation can build.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   An envelope arm that read its limits out of the artifact and then
 *   compared them against the artifact would agree with any
 *   transcription error. The limits are written here a second time,
 *   in the units the source document prints them in, and converted
 *   here; the artifact's own constants are read out of the emitted
 *   source and compared against those. A wrong figure in the
 *   compiler's table, or a wrong conversion, fails here.
 *
 *   A limit arm that tested one state well inside and one well
 *   outside would pass for any limit within a wide band. Each is
 *   tested at the limit and one unit of least precision beyond it.
 *
 *   A residual arm that compared the port channels against the port
 *   library would be comparing the artifact with itself. The expected
 *   residuals are computed here from the body-state and attitude
 *   getters and the port geometry, in code that shares nothing with
 *   the library, and the rates are checked a second way, against the
 *   finite difference of the residuals across a step.
 *
 *   An outcome arm that only asserted that something ended would pass
 *   for an environment that ended everything the same way. All four
 *   outcomes are driven, each is asserted on its own flag bit and on
 *   the capture channel, and the docked and impact arms differ only
 *   in the closing rate the same controller holds.
 *
 *   A determinism arm comparing two runs in one process would share
 *   every cache and every allocation. The two runs are two processes.
 */
#include "rl_gate_util.h"

#include "k26astro_coll/coll.h"

#include <math.h>
#include <unistd.h>

#define WORK_DIR "/tmp/kflc_rl_docking_test"
#define BENCH    "examples/docking_benchmark.kfl"

static int n_pass = 0;

static void near_(const char *what, double got, double want, double tol)
{
    double e = got - want;
    if (e < 0.0) e = -e;
    printf("  %-40s %+.12g  want %+.12g  err %.3e\n", what, got, want, e);
    ASSERT(e <= tol);
}

/* ---- the envelope, as the source document prints it --------------- *
 *
 * International Docking System Standard Interface Definition
 * Document, Revision E, October 2016, Table 3.3.1.1-2, Initial
 * Contact Conditions, and section 3.2 for the mating plane diameter.
 * The figures are in the document's own units and the conversion is
 * done here, so a conversion error in the compiler shows up as a
 * disagreement rather than as two copies of one mistake.
 */
static const double IDSS_E_AXIAL_MIN_MPS   = 0.05;
static const double IDSS_E_AXIAL_MAX_MPS   = 0.10;
static const double IDSS_E_LATERAL_MPS     = 0.04;
static const double IDSS_E_PITCHYAW_DEGPS  = 0.20;
static const double IDSS_E_ROLL_DEGPS      = 0.20;
static const double IDSS_E_LATERAL_M       = 0.10;
static const double IDSS_E_PITCHYAW_DEG    = 4.0;
static const double IDSS_E_ROLL_DEG        = 4.0;
static const double IDSS_E_MATING_MM       = 1200.0;

static double deg2rad_(double d)
{
    return d * (3.14159265358979323846 / 180.0);
}

/* ---- reading the artifact's own constants ------------------------ */

/* One port as the compiler emitted it. */
typedef struct {
    int    veh, shape;
    double com[3];
    double at[3];
    double basis[3][3];
    double env[8];
} GatePort;

/* Pull the next numeric token, skipping anything that is not one. A
 * digit inside an identifier is not a number: the emitted table casts
 * its kind through a type whose own name carries digits, and reading
 * those would shift every field that followed. */
static int num_start_(const char *p)
{
    char c = p[-1];
    if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
        (c >= 'A' && c <= 'Z') || c == '_') {
        return 0;
    }
    return 1;
}

static const char *scan_num_(const char *p, double *out)
{
    while (*p) {
        if (num_start_(p) &&
            ((*p >= '0' && *p <= '9') ||
             ((*p == '-' || *p == '+' || *p == '.') &&
              (p[1] >= '0' && p[1] <= '9')))) {
            char *end = NULL;
            *out = strtod(p, &end);
            ASSERT(end != NULL && end != p);
            return end;
        }
        p++;
    }
    ASSERT(0);
    return NULL;
}

static char *read_whole_(const char *path, long *len)
{
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    long n = ftell(f);
    ASSERT(n > 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char *buf = (char *)malloc((size_t)n + 1);
    ASSERT(buf != NULL);
    ASSERT(fread(buf, 1, (size_t)n, f) == (size_t)n);
    buf[n] = '\0';
    fclose(f);
    if (len) *len = n;
    return buf;
}

/* The emitted port table, in declaration order. Each entry is two
 * integers and twenty-three doubles, in the order the emitter writes
 * them, so the reader is a count and not a parser. */
static int read_ports_(const char *csrc, GatePort *out, int max)
{
    const char *p = strstr(csrc, "static const KflrlPort kflrl_ports_[] = {");
    ASSERT(p != NULL);
    p += strlen("static const KflrlPort kflrl_ports_[] = {");
    const char *end = strstr(p, "\n};");
    ASSERT(end != NULL);
    int n = 0;
    while (n < max) {
        const char *q = p;
        while (*q == ' ' || *q == '\n') q++;
        if (q >= end) break;
        double v[25];
        for (int i = 0; i < 25; i++) p = scan_num_(p, &v[i]);
        ASSERT(p <= end);
        GatePort *g = &out[n++];
        g->veh   = (int)v[0];
        g->shape = (int)v[1];
        for (int i = 0; i < 3; i++) g->com[i] = v[2 + i];
        for (int i = 0; i < 3; i++) g->at[i]  = v[5 + i];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) g->basis[i][j] = v[8 + 3 * i + j];
        }
        for (int i = 0; i < 8; i++) g->env[i] = v[17 + i];
        const char *nl = p;
        while (*nl && *nl != '\n') nl++;
        if (nl >= end) break;
    }
    return n;
}

/* ---- fixture vector arithmetic, written here --------------------- */

typedef struct { double x, y, z; } V3;
static V3 v3_(double x, double y, double z) { V3 v = { x, y, z }; return v; }
static V3 add3_(V3 a, V3 b) { return v3_(a.x+b.x, a.y+b.y, a.z+b.z); }
static V3 sub3_(V3 a, V3 b) { return v3_(a.x-b.x, a.y-b.y, a.z-b.z); }
static V3 mul3_(V3 a, double s) { return v3_(a.x*s, a.y*s, a.z*s); }
static double dot3_(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3 cross3_(V3 a, V3 b)
{
    return v3_(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static double len3_(V3 a) { return sqrt(dot3_(a, a)); }
static V3 perp3_(V3 v, V3 n) { return sub3_(v, mul3_(n, dot3_(v, n))); }

/* Rotate a body-frame vector into the world by a quaternion given as
 * w, x, y, z, written out rather than called so the expected values
 * owe nothing to the library under test. */
static V3 qrot_(const double q[4], V3 v)
{
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = sqrt(w*w + x*x + y*y + z*z);
    w /= n; x /= n; y /= n; z /= n;
    V3 u = v3_(x, y, z);
    V3 t = mul3_(cross3_(u, v), 2.0);
    return add3_(add3_(v, mul3_(t, w)), cross3_(u, t));
}

/* The state of the active port with respect to the passive one,
 * computed here from the getters. */
typedef struct {
    double axial, lateral, pitchyaw, roll;
    double v_axial, v_lateral, v_pitchyaw, v_roll, v_lateral_cg;
    /* The rate the axial residual itself changes at, which is not
     * the closing rate: the axis it is measured on belongs to the
     * passive port and turns with that body, so the residual also
     * changes by the separation swept across the turning axis. */
    double d_axial;
} GateState;

static void expect_state_(const GatePort *ap, const double aq[4], V3 apos,
                          V3 avel, V3 aom,
                          const GatePort *pp, const double pq[4], V3 ppos,
                          V3 pvel, V3 pom, GateState *out)
{
    V3 aax[3], pax[3];
    for (int i = 0; i < 3; i++) {
        aax[i] = qrot_(aq, v3_(ap->basis[i][0], ap->basis[i][1],
                               ap->basis[i][2]));
        pax[i] = qrot_(pq, v3_(pp->basis[i][0], pp->basis[i][1],
                               pp->basis[i][2]));
    }
    V3 aorg = add3_(apos, qrot_(aq, v3_(ap->at[0], ap->at[1], ap->at[2])));
    V3 porg = add3_(ppos, qrot_(pq, v3_(pp->at[0], pp->at[1], pp->at[2])));
    V3 acom = add3_(apos, qrot_(aq, v3_(ap->com[0], ap->com[1], ap->com[2])));
    V3 pcom = add3_(ppos, qrot_(pq, v3_(pp->com[0], pp->com[1], pp->com[2])));

    V3 dax[3];
    dax[0] = mul3_(pax[0], -1.0);
    dax[1] = pax[1];
    dax[2] = cross3_(dax[0], dax[1]);

    V3 delta = sub3_(aorg, porg);
    out->axial   = dot3_(delta, pax[0]);
    out->lateral = len3_(perp3_(delta, pax[0]));

    double m[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) m[i][j] = dot3_(dax[i], aax[j]);
    }
    double s = m[2][0];
    if (s > 1.0) s = 1.0;
    if (s < -1.0) s = -1.0;
    double pitch = -asin(s);
    double yaw   = atan2(m[1][0], m[0][0]);
    double roll  = atan2(m[2][1], m[2][2]);
    out->pitchyaw = sqrt(pitch * pitch + yaw * yaw);
    out->roll     = fabs(roll);

    V3 wa = qrot_(aq, aom), wp = qrot_(pq, pom);
    V3 va = add3_(avel, cross3_(wa, sub3_(aorg, acom)));
    V3 vp = add3_(pvel, cross3_(wp, sub3_(porg, pcom)));
    V3 vrel = sub3_(va, vp);
    V3 wrel = sub3_(wa, wp);
    out->v_axial      = -dot3_(vrel, pax[0]);
    out->v_lateral    = len3_(perp3_(vrel, pax[0]));
    out->v_pitchyaw   = len3_(perp3_(wrel, pax[0]));
    out->v_roll       = fabs(dot3_(wrel, pax[0]));
    V3 arm  = sub3_(acom, aorg);
    V3 cvel = add3_(vrel, cross3_(wrel, arm));
    out->v_lateral_cg = len3_(perp3_(cvel, pax[0]));
    out->d_axial = -out->v_axial + dot3_(delta, cross3_(wp, pax[0]));
}

/* ---- the scripted pilot ------------------------------------------ */

/* Channel indices, resolved by name from the spec rather than
 * counted, so a form that changed width would be a lookup failure and
 * not a silently shifted read. */
typedef struct {
    int rel_r_x, rel_r_z, rel_v_x, rel_v_z;
    int att_qw, att_wx;
    int sta_qw, sta_wx;
    int dock_cap, dock_axial, dock_v_axial;
    int touch_hit;
    int total;
} GateChan;

static double clamp01_(double v) { return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v); }
static double clamppm_(double v)
{ return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); }

/* One control step: hold the declared closing rate, null the lateral
 * offset the orbital dynamics build up, and hold the mated attitude
 * the station's own frame defines. The lateral thrusters answer body
 * axes, and under this task's mated attitude body -z is the outward
 * radial direction and body +y the orbit normal.
 *
 * Nothing here is part of the environment: it is one policy among
 * many, written so the gate can show that the task's success
 * condition is reachable. */
static void pilot_(const double *o, const GateChan *c, double v_close,
                   double *act)
{
    memset(act, 0, sizeof(double) * 9);
    double a_ax = 1.0 * (v_close - o[c->dock_v_axial]);
    if (a_ax > 0.0) act[0] = clamp01_(a_ax / 0.08);
    else            act[1] = clamp01_(-a_ax / 0.08);

    double a_rad = 0.01 * o[c->rel_r_x] + 0.2 * o[c->rel_v_x];
    double a_cro = 0.01 * o[c->rel_r_z] + 0.2 * o[c->rel_v_z];
    if (a_rad > 0.0) act[5] = clamp01_(a_rad / 0.08);
    else             act[4] = clamp01_(-a_rad / 0.08);
    if (a_cro > 0.0) act[2] = clamp01_(a_cro / 0.08);
    else             act[3] = clamp01_(-a_cro / 0.08);

    /* The target attitude is the station's, turned half a turn about
     * its own second axis, which is the mated configuration. The
     * error is resolved in the chaser's body frame, and the rate term
     * is the rate ERROR: the target turns once per orbit, and a law
     * that nulled the absolute rate would stop tracking it. */
    const double *cq = &o[c->att_qw];
    const double *sq = &o[c->sta_qw];
    double tw = -sq[2], tx = -sq[3], ty = sq[0], tz = sq[1];
    double iw = cq[0], ix = -cq[1], iy = -cq[2], iz = -cq[3];
    double ew = iw*tw - ix*tx - iy*ty - iz*tz;
    double ex = iw*tx + ix*tw + iy*tz - iz*ty;
    double ey = iw*ty - ix*tz + iy*tw + iz*tx;
    double ez = iw*tz + ix*ty - iy*tx + iz*tw;
    double sgn = ew < 0.0 ? -1.0 : 1.0;
    double err[3] = { 2.0*sgn*ex, 2.0*sgn*ey, 2.0*sgn*ez };
    double om[3] = { o[c->att_wx + 0] + o[c->sta_wx + 0],
                     o[c->att_wx + 1] - o[c->sta_wx + 1],
                     o[c->att_wx + 2] + o[c->sta_wx + 2] };
    const double In[3] = { 17000.0, 42000.0, 42000.0 };
    for (int q = 0; q < 3; q++) {
        double tau = In[q] * (0.004 * err[q] - 0.12 * om[q]);
        act[6 + q] = clamppm_(-tau / 5.0);
    }
}

/* ---- the spec's channel names ------------------------------------ */

static int find_channel_(const uint8_t *blob, uint32_t len, const char *want)
{
    uint32_t off = 0;
    int found = -1;
    while (off + 6 <= len) {
        uint16_t tag = rl_get_u16_(blob + off);
        uint32_t l   = rl_get_u32_(blob + off + 2);
        const uint8_t *val = blob + off + 6;
        if (tag == K26RL_TAG_OBS_CHANNEL_NAME && l >= 4) {
            char name[128];
            uint32_t nl = l - 4;
            if (nl > sizeof name - 1) nl = sizeof name - 1;
            memcpy(name, val + 4, nl);
            name[nl] = '\0';
            if (strcmp(name, want) == 0) found = (int)rl_get_u32_(val);
        }
        off += 6 + l;
    }
    return found;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_docking")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    /* ---- 1. the envelope the compiler carries -------------------- */
    printf("the envelope's own figures, converted once\n");
    rl_run_or_die_("./bin/kflc --emit " BENCH " > " WORK_DIR "/dock.cc 2>"
                   WORK_DIR "/emit.log");
    char *csrc = read_whole_(WORK_DIR "/dock.cc", NULL);
    GatePort ports[8];
    int n_ports = read_ports_(csrc, ports, 8);
    printf("  ports emitted: %d\n", n_ports);
    ASSERT(n_ports == 2);
    const double want_env[8] = {
        IDSS_E_AXIAL_MIN_MPS, IDSS_E_AXIAL_MAX_MPS, IDSS_E_LATERAL_MPS,
        deg2rad_(IDSS_E_PITCHYAW_DEGPS), deg2rad_(IDSS_E_ROLL_DEGPS),
        IDSS_E_LATERAL_M, deg2rad_(IDSS_E_PITCHYAW_DEG),
        deg2rad_(IDSS_E_ROLL_DEG)
    };
    static const char *const env_name[8] = {
        "closing rate lower bound", "closing rate upper bound",
        "lateral rate", "pitch/yaw rate", "roll rate",
        "lateral misalignment", "pitch/yaw misalignment",
        "roll misalignment"
    };
    for (int p = 0; p < n_ports; p++) {
        for (int i = 0; i < 8; i++) {
            char label[80];
            snprintf(label, sizeof label, "port %d %s", p, env_name[i]);
            /* Bitwise: the conversion is one multiplication and the
             * gate does the same one, so anything but equality is a
             * different figure or a different conversion. */
            near_(label, ports[p].env[i], want_env[i], 0.0);
        }
    }
    /* Vehicle slots follow body declaration order, so slot 0 is the
     * station and slot 1 the chaser. They are resolved by slot rather
     * than assumed by position in the table. */
    const GatePort *sta_port = NULL, *cha_port = NULL;
    for (int p = 0; p < n_ports; p++) {
        if (ports[p].veh == 0) sta_port = &ports[p];
        if (ports[p].veh == 1) cha_port = &ports[p];
    }
    ASSERT(sta_port != NULL && cha_port != NULL);
    /* The interface geometry the assets declare, quoted here so a
     * port emitted at the wrong place fails rather than agreeing with
     * itself further down. */
    near_("chaser port on the docking axis", cha_port->at[0], 3.75, 0.0);
    near_("station port on the docking axis", sta_port->at[0], 33.6, 0.0);
    ASSERT(cha_port->at[1] == 0.0 && cha_port->at[2] == 0.0);
    ASSERT(sta_port->at[1] == 0.0 && sta_port->at[2] == 0.0);

    /* The mating plane collider is the envelope's, not the author's:
     * a square plate circumscribing the published circle, thin along
     * the axis and behind the plane. */
    {
        double radius = 0.5 * (IDSS_E_MATING_MM / 1000.0);
        double half   = radius * 0.01;
        static const char *const CTAB =
            "static const K26AstroCollShape kflrl_coll_[] = {";
        const char *cp = strstr(csrc, CTAB);
        ASSERT(cp != NULL);
        const char *cend = strstr(cp, "\n};");
        ASSERT(cend != NULL);
        int plates = 0;
        const char *q = cp + strlen(CTAB);
        while (q < cend) {
            double v[16];
            const char *r = q;
            int ok = 1;
            for (int i = 0; i < 16 && ok; i++) {
                if (!r || r >= cend) { ok = 0; break; }
                r = scan_num_(r, &v[i]);
            }
            if (!ok || r > cend) break;
            q = r;
            if (v[0] == 3.0 && v[14] == radius && v[15] == radius) {
                near_("mating plate half thickness", v[13], half, 0.0);
                plates++;
            }
        }
        printf("  mating plates found: %d\n", plates);
        ASSERT(plates == 2);
    }
    n_pass++;

    /* A port whose frame is not axis aligned. The benchmark's own
     * ports point along a body axis with the roll reference on
     * another, so their basis is the identity and the identity is its
     * own transpose: nothing above can tell a basis carried out
     * correctly from one carried out transposed. This fixture has no
     * two axes alike, and its basis is computed here from the
     * declaration. */
    printf("a port whose frame is not axis aligned keeps its own axes\n");
    {
        rl_write_file_(WORK_DIR "/skew.k26asm",
            "assembly skew\n"
            "    frame x_to_port\n"
            "    provenance mass \"gate fixture, not a craft\" computed\n"
            "    component hull\n"
            "        mass 1000.0\n"
            "        at 0 0 0\n"
            "        collider box 1.0 0.5 0.5\n"
            "    end\n"
            "    port oblique\n"
            "        at 0.7 -0.2 0.5\n"
            "        axis 0.48 0.6 0.64\n"
            "        roll_ref 0.0 1.0 0.0\n"
            "        capture idss_e\n"
            "    end\n"
            "end\n");
        char prog[2048];
        snprintf(prog, sizeof prog,
            "form RL_SKEW\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body one assembly=\"%s/skew.k26asm\" parent=earth"
            " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
            "    astro_body two assembly=\"%s/skew.k26asm\" parent=earth"
            " pos_x=7.0e6 pos_z=20.0 vel_y=7546.0 quat_w=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 4\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port oblique of one as sk\n"
            "    objective\n"
            "        reward sk_axial + sk_lateral + sk_pitchyaw + sk_roll"
            " + sk_v_axial + sk_v_lateral + sk_v_pitchyaw + sk_v_roll"
            " + sk_captured\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR);
        rl_write_file_(WORK_DIR "/skew.kfl", prog);
        rl_run_or_die_("./bin/kflc --emit " WORK_DIR "/skew.kfl > "
                       WORK_DIR "/skew.cc 2> " WORK_DIR "/skew.log");
        char *sk = read_whole_(WORK_DIR "/skew.cc", NULL);
        GatePort sp[4];
        int n_sp = read_ports_(sk, sp, 4);
        ASSERT(n_sp == 2);
        double bx[3] = { 0.48, 0.6, 0.64 };
        double rr[3] = { 0.0, 1.0, 0.0 };
        double d = rr[0]*bx[0] + rr[1]*bx[1] + rr[2]*bx[2];
        double by[3], bz[3], nn = 0.0;
        for (int q = 0; q < 3; q++) by[q] = rr[q] - d * bx[q];
        for (int q = 0; q < 3; q++) nn += by[q] * by[q];
        nn = sqrt(nn);
        for (int q = 0; q < 3; q++) by[q] /= nn;
        bz[0] = bx[1]*by[2] - bx[2]*by[1];
        bz[1] = bx[2]*by[0] - bx[0]*by[2];
        bz[2] = bx[0]*by[1] - bx[1]*by[0];
        for (int q = 0; q < 3; q++) {
            near_("port axis", sp[0].basis[0][q], bx[q], 1e-15);
            near_("port roll reference", sp[0].basis[1][q], by[q], 1e-15);
            near_("port third axis", sp[0].basis[2][q], bz[q], 1e-15);
        }
        free(sk);
    }
    n_pass++;

    /* ---- 2. each limit decides, at the limit and one bit beyond --- */
    printf("each condition at its limit and one bit past it\n");
    {
        K26AstroCollEnvelope e;
        e.axial_rate_min = cha_port->env[0];
        e.axial_rate_max = cha_port->env[1];
        e.lateral_rate   = cha_port->env[2];
        e.pitchyaw_rate  = cha_port->env[3];
        e.roll_rate      = cha_port->env[4];
        e.lateral        = cha_port->env[5];
        e.pitchyaw       = cha_port->env[6];
        e.roll           = cha_port->env[7];

        K26AstroCollPortState base;
        memset(&base, 0, sizeof base);
        base.v_axial = 0.07;   base.v_lateral = 0.01;
        base.v_pitchyaw = 0.001; base.v_roll = 0.001;
        base.lateral = 0.03;   base.pitchyaw = 0.02; base.roll = 0.02;
        base.v_lateral_cg = 0.01;
        ASSERT(k26astro_coll_port_captured(&base, &e) == 1);

        struct { const char *name; size_t off; double lim;
                 int upper; } arm[] = {
            { "closing rate lower bound",
              offsetof(K26AstroCollPortState, v_axial), e.axial_rate_min, 0 },
            { "closing rate upper bound",
              offsetof(K26AstroCollPortState, v_axial), e.axial_rate_max, 1 },
            { "lateral rate",
              offsetof(K26AstroCollPortState, v_lateral), e.lateral_rate, 1 },
            { "pitch/yaw rate",
              offsetof(K26AstroCollPortState, v_pitchyaw), e.pitchyaw_rate, 1 },
            { "roll rate",
              offsetof(K26AstroCollPortState, v_roll), e.roll_rate, 1 },
            { "lateral misalignment",
              offsetof(K26AstroCollPortState, lateral), e.lateral, 1 },
            { "pitch/yaw misalignment",
              offsetof(K26AstroCollPortState, pitchyaw), e.pitchyaw, 1 },
            { "roll misalignment",
              offsetof(K26AstroCollPortState, roll), e.roll, 1 },
            { "lateral rate at the centre of mass",
              offsetof(K26AstroCollPortState, v_lateral_cg),
              e.lateral_rate, 1 },
        };
        for (size_t i = 0; i < sizeof arm / sizeof arm[0]; i++) {
            K26AstroCollPortState at = base, beyond = base;
            double *pa = (double *)((char *)&at + arm[i].off);
            double *pb = (double *)((char *)&beyond + arm[i].off);
            *pa = arm[i].lim;
            *pb = arm[i].upper ? nextafter(arm[i].lim, 2.0)
                               : nextafter(arm[i].lim, -2.0);
            int ok_at = k26astro_coll_port_captured(&at, &e);
            int ok_bey = k26astro_coll_port_captured(&beyond, &e);
            printf("  %-38s %.17g -> %d, %.17g -> %d\n",
                   arm[i].name, *pa, ok_at, *pb, ok_bey);
            ASSERT(ok_at == 1 && ok_bey == 0);
        }

        /* The two vector-sum conditions are vector sums: each part
         * inside its own limit while the sum is outside. */
        K26AstroCollPortState vs = base;
        vs.pitchyaw = e.pitchyaw * 0.8;       /* one part, inside */
        ASSERT(k26astro_coll_port_captured(&vs, &e) == 1);
        vs.pitchyaw = sqrt(2.0) * e.pitchyaw * 0.8;  /* two such parts */
        ASSERT(k26astro_coll_port_captured(&vs, &e) == 0);
        vs = base;
        vs.v_pitchyaw = e.pitchyaw_rate * 0.8;
        ASSERT(k26astro_coll_port_captured(&vs, &e) == 1);
        vs.v_pitchyaw = sqrt(2.0) * e.pitchyaw_rate * 0.8;
        ASSERT(k26astro_coll_port_captured(&vs, &e) == 0);
        printf("  two parts each inside the limit whose vector sum is "
               "outside are refused: OK\n");

        /* The note's own condition, as its own arm: inside every
         * limit at the interface and outside at the centre of mass. */
        K26AstroCollPortState cg = base;
        cg.v_lateral   = e.lateral_rate * 0.5;
        cg.v_pitchyaw  = e.pitchyaw_rate * 0.9;
        cg.v_lateral_cg = e.lateral_rate * 1.4;
        ASSERT(k26astro_coll_port_captured(&cg, &e) == 0);
        cg.v_lateral_cg = e.lateral_rate * 0.5;
        ASSERT(k26astro_coll_port_captured(&cg, &e) == 1);
        printf("  a contact inside every interface limit is refused on the "
               "rate at the centre of mass: OK\n");
    }
    n_pass++;

    char cwd[512];
    ASSERT(getcwd(cwd, sizeof cwd) != NULL);

    /* ---- 3. the artifact ----------------------------------------- */
    rl_compile_(BENCH, WORK_DIR "/dock", WORK_DIR);
    void *so = rl_dlopen_(WORK_DIR "/dock.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);

    K26RlEnv *env = NULL;
    ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
    uint8_t blob[16384];
    int32_t blen = s.spec(env, blob, sizeof blob);
    ASSERT(blen > 0);
    RlSpecView sv;
    rl_parse_spec_(blob, (uint32_t)blen, &sv);

    printf("the port form publishes nine channels at its published names\n");
    static const char *const port_sfx[9] = {
        "_captured", "_axial", "_lateral", "_pitchyaw", "_roll",
        "_v_axial", "_v_lateral", "_v_pitchyaw", "_v_roll"
    };
    int first = -1;
    for (int i = 0; i < 9; i++) {
        char want[64];
        snprintf(want, sizeof want, "dock%s", port_sfx[i]);
        int idx = find_channel_(blob, (uint32_t)blen, want);
        printf("  %-20s channel %d\n", want, idx);
        ASSERT(idx >= 0);
        if (i == 0) first = idx;
        ASSERT(idx == first + i);
    }
    n_pass++;

    GateChan ch;
    ch.total        = (int)sv.obs_total;
    ch.rel_r_x      = find_channel_(blob, (uint32_t)blen, "rel_r_x");
    ch.rel_r_z      = find_channel_(blob, (uint32_t)blen, "rel_r_z");
    ch.rel_v_x      = find_channel_(blob, (uint32_t)blen, "rel_v_x");
    ch.rel_v_z      = find_channel_(blob, (uint32_t)blen, "rel_v_z");
    ch.att_qw       = find_channel_(blob, (uint32_t)blen, "att_quat_w");
    ch.att_wx       = find_channel_(blob, (uint32_t)blen, "att_omega_x");
    ch.sta_qw       = find_channel_(blob, (uint32_t)blen, "sta_quat_w");
    ch.sta_wx       = find_channel_(blob, (uint32_t)blen, "sta_omega_x");
    ch.dock_cap     = find_channel_(blob, (uint32_t)blen, "dock_captured");
    ch.dock_axial   = find_channel_(blob, (uint32_t)blen, "dock_axial");
    ch.dock_v_axial = find_channel_(blob, (uint32_t)blen, "dock_v_axial");
    ch.touch_hit    = find_channel_(blob, (uint32_t)blen, "touch_hit");
    ASSERT(ch.rel_r_x >= 0 && ch.att_qw >= 0 && ch.sta_qw >= 0 &&
           ch.dock_cap >= 0 && ch.touch_hit >= 0);

    double *obs = (double *)calloc((size_t)ch.total, sizeof(double));
    ASSERT(obs != NULL);
    double act[9], rew[1];
    uint32_t flags[1];

    /* ---- 4. the residuals against an independent computation ----- */
    printf("the residuals against the getters, in code sharing nothing\n");
    {
        /* Body order is declaration order: earth, station, chaser.
         * The port table is in vehicle order, which is the order of
         * the bodies that carry an assembly: station, chaser. */
        double bodies[3 * 6], atts[3 * 7];
        GateState want;
        double prev_axial = 0.0, prev_v_axial = 0.0;
        int have_prev = 0;
        memset(act, 0, sizeof act);
        for (int k = 0; k < 6; k++) {
            /* Coasting: the hold point holds, so the residuals move
             * only by what the dynamics do, which is what makes the
             * finite difference below a measurement and not noise. */
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.bodies(env, 1u, bodies, 3 * 6) == 3 * 6);
            ASSERT(s.attitudes(env, atts, 3 * 7) == 3 * 7);
            V3 spos = v3_(bodies[1*6+0], bodies[1*6+1], bodies[1*6+2]);
            V3 svel = v3_(bodies[1*6+3], bodies[1*6+4], bodies[1*6+5]);
            V3 cpos = v3_(bodies[2*6+0], bodies[2*6+1], bodies[2*6+2]);
            V3 cvel = v3_(bodies[2*6+3], bodies[2*6+4], bodies[2*6+5]);
            double sq[4] = { atts[1*7+0], atts[1*7+1],
                             atts[1*7+2], atts[1*7+3] };
            double cq[4] = { atts[2*7+0], atts[2*7+1],
                             atts[2*7+2], atts[2*7+3] };
            V3 som = v3_(atts[1*7+4], atts[1*7+5], atts[1*7+6]);
            V3 com = v3_(atts[2*7+4], atts[2*7+5], atts[2*7+6]);
            expect_state_(cha_port, cq, cpos, cvel, com,
                          sta_port, sq, spos, svel, som, &want);
            if (k == 5) {
                near_("axial", obs[ch.dock_axial + 0], want.axial, 1e-9);
                near_("lateral", obs[ch.dock_axial + 1], want.lateral, 1e-9);
                near_("pitch/yaw", obs[ch.dock_axial + 2], want.pitchyaw, 1e-9);
                near_("roll", obs[ch.dock_axial + 3], want.roll, 1e-9);
                near_("closing rate", obs[ch.dock_v_axial + 0],
                      want.v_axial, 1e-12);
                near_("lateral rate", obs[ch.dock_v_axial + 1],
                      want.v_lateral, 1e-12);
                near_("pitch/yaw rate", obs[ch.dock_v_axial + 2],
                      want.v_pitchyaw, 1e-14);
                near_("roll rate", obs[ch.dock_v_axial + 3],
                      want.v_roll, 1e-14);
                /* The rate a second way: the fall of the axial
                 * residual across one control period, against the
                 * mean of the rates at its two ends. The mean rather
                 * than either end because the rate is changing across
                 * the period, and a one-sided comparison would be
                 * measuring that change rather than the rate. */
                double fd = (obs[ch.dock_axial] - prev_axial) / 0.5;
                near_("the axial residual's own rate against a "
                      "finite difference",
                      0.5 * (prev_v_axial + want.d_axial), fd, 1e-7);
            }
            prev_axial   = obs[ch.dock_axial];
            prev_v_axial = want.d_axial;
            have_prev = 1;
        }
        ASSERT(have_prev);
    }
    n_pass++;
    s.destroy(env);

    /* ---- 5. the four outcomes ------------------------------------ */
    printf("the four outcomes\n");
    struct { const char *what; double v_close; int wheels; } run[3] = {
        { "docked",  0.07, 1 },
        { "impact",  0.20, 1 },
        { "horizon", 0.00, 1 }
    };
    for (int r = 0; r < 3; r++) {
        ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
        int ended = -1, captured = -1, hit = -1;
        double end_v_axial = 0.0, end_axial = 0.0, end_frac = 0.0;
        uint32_t last_flags = 0;
        double last_reward = 0.0;
        int cap_before = 0;
        for (int k = 0; k < 420; k++) {
            ASSERT(s.obs(env, obs) == K26RL_OK);
            if (obs[ch.dock_cap] != 0.0) cap_before++;
            if (r == 2) memset(act, 0, sizeof act);
            else        pilot_(obs, &ch, run[r].v_close, act);
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.flags(env, flags) == K26RL_OK);
            ASSERT(s.reward(env, rew) == K26RL_OK);
            if (flags[0] & 3u) {
                ended = k; last_flags = flags[0]; last_reward = rew[0];
                captured = obs[ch.dock_cap] != 0.0;
                hit = obs[ch.touch_hit] != 0.0;
                end_v_axial = obs[ch.dock_v_axial];
                end_axial   = obs[ch.dock_axial];
                end_frac    = obs[ch.touch_hit + 1];
                break;
            }
        }
        printf("  %-8s ended at step %d flags %u captured %d hit %d "
               "reward %+.3f (capture seen before the end: %d)\n",
               run[r].what, ended, last_flags, captured, hit, last_reward,
               cap_before);
        ASSERT(ended >= 0);
        if (r == 0) {
            ASSERT((last_flags & 1u) != 0);   /* terminated */
            ASSERT(captured == 1 && hit == 1);
            ASSERT(cap_before == 0);          /* set on exactly that step */
            ASSERT(last_reward > 50.0);       /* the terminal bonus landed */
            /* The residuals published on the step that captured are
             * the ones the test was given, taken at the instant of
             * contact. By the end of that transition the resolution
             * has removed the closing rate, so a form that published
             * the later state would read a rate of nearly zero here,
             * outside the envelope it just accepted. */
            printf("           closing rate published on the capture "
                   "step %.6f, axial %.9f, contact fraction %.6f\n",
                   end_v_axial, end_axial, end_frac);
            ASSERT(end_v_axial >= 0.05 && end_v_axial <= 0.10);
            /* And taken at the instant, not at the start of the
             * sub-advance that contained it. The contact fraction is
             * of the whole control period, so where in its own
             * sub-advance the contact fell follows from the declared
             * subdivision, and what a state taken at that
             * sub-advance's start would carry is that much extra
             * separation. The arm states its own discriminating
             * power: the slip has to be several times the residual
             * asserted, or the two configurations coincide and the
             * assertion below is about nothing. */
            {
                double sub_dt = 0.5 / 5.0;
                double f5 = end_frac * 5.0;
                double cc_time = f5 - floor(f5);
                double slip = end_v_axial * sub_dt * cc_time;
                printf("           the contact fell %.4f into its "
                       "sub-advance; a state taken at that "
                       "sub-advance's start would carry %.9f m more\n",
                       cc_time, slip);
                ASSERT(slip > 4.0 * end_axial);
                ASSERT(end_axial >= 0.0 && end_axial < 0.5 * slip);
            }
        } else if (r == 1) {
            ASSERT((last_flags & 1u) != 0);
            ASSERT(captured == 0 && hit == 1);
            ASSERT(last_reward < 0.0);
        } else {
            ASSERT((last_flags & 2u) != 0);   /* truncated */
            ASSERT(captured == 0 && hit == 0);
        }
        s.destroy(env);
    }
    n_pass++;

    /* The fourth outcome, and what it takes to reach it. This
     * benchmark introduces no fault reason of its own, and it turns
     * out no action stream reaches one either: the actuator layer
     * maps a command that is not a number to zero, so an agent cannot
     * drive the physics into a state it cannot advance. That is a
     * property worth pinning rather than working around, so it is the
     * first arm here; the second reaches the fault the way any
     * environment can, through a state the advance cannot use, on a
     * variant of this same programme carrying one added line. */
    printf("the fourth outcome, and the route to it\n");
    {
        ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
        memset(act, 0, sizeof act);
        ASSERT(s.step(env, act) == K26RL_OK);
        for (int i = 0; i < 9; i++) act[i] = NAN;
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.flags(env, flags) == K26RL_OK);
        ASSERT(s.obs(env, obs) == K26RL_OK);
        int finite = 1;
        for (int i = 0; i < ch.total; i++) {
            if (!isfinite(obs[i])) finite = 0;
        }
        printf("  every action not a number: flags %u, observations "
               "finite %d\n", flags[0], finite);
        ASSERT((flags[0] & 4u) == 0);
        ASSERT(finite == 1);
        s.destroy(env);
    }
    {
        /* The variant: the same programme with one action that drives
         * the chaser's attitude quaternion to zero norm, which the
         * advance cannot normalise. The asset paths are made absolute
         * because the copy does not sit beside them. */
        char *src = read_whole_(BENCH, NULL);
        char *out = (char *)malloc(strlen(src) + 4096);
        ASSERT(out != NULL);
        const char *cur = src;
        size_t n = 0;
        for (;;) {
            const char *hit = strstr(cur, "assembly=\"assets/");
            if (!hit) break;
            memcpy(out + n, cur, (size_t)(hit - cur));
            n += (size_t)(hit - cur);
            n += (size_t)snprintf(out + n, 1024,
                                  "assembly=\"%s/examples/assets/", cwd);
            cur = hit + strlen("assembly=\"assets/");
        }
        strcpy(out + n, cur);
        char *body = (char *)malloc(strlen(out) + 4096);
        ASSERT(body != NULL);
        const char *anchor = "    action tq_z box -1.0 1.0 default 0.0\n";
        const char *ap = strstr(out, anchor);
        ASSERT(ap != NULL);
        size_t pre = (size_t)(ap - out) + strlen(anchor);
        memcpy(body, out, pre);
        size_t m = pre;
        m += (size_t)snprintf(body + m, 512,
                              "    action kill box 0.0 1.0 default 0.0\n");
        const char *anchor2 = "        chaser.wheel_z.torque = 5.0 * tq_z\n";
        const char *bp = strstr(out + pre, anchor2);
        ASSERT(bp != NULL);
        memcpy(body + m, out + pre,
               (size_t)(bp - (out + pre)) + strlen(anchor2));
        m += (size_t)(bp - (out + pre)) + strlen(anchor2);
        m += (size_t)snprintf(body + m, 1024,
            "        chaser.quat_w = chaser.quat_w * (1.0 - kill)\n"
            "        chaser.quat_x = chaser.quat_x * (1.0 - kill)\n"
            "        chaser.quat_y = chaser.quat_y * (1.0 - kill)\n"
            "        chaser.quat_z = chaser.quat_z * (1.0 - kill)\n");
        strcpy(body + m, bp + strlen(anchor2));
        rl_write_file_(WORK_DIR "/faultable.kfl", body);
        free(src); free(out); free(body);

        rl_compile_(WORK_DIR "/faultable.kfl", WORK_DIR "/faultable", WORK_DIR);
        void *fso = rl_dlopen_(WORK_DIR "/faultable.rlenv.so");
        RlSurface fs;
        rl_resolve_surface_(fso, &fs);
        K26RlEnv *fe = NULL;
        ASSERT(fs.create(11u, 1u, &fe) == K26RL_OK);
        double fact[10];
        memset(fact, 0, sizeof fact);
        ASSERT(fs.step(fe, fact) == K26RL_OK);
        fact[9] = 1.0;
        ASSERT(fs.step(fe, fact) == K26RL_OK);
        uint32_t ff[1];
        uint16_t fc[1];
        ASSERT(fs.flags(fe, ff) == K26RL_OK);
        ASSERT(fs.fault_codes(fe, fc) == K26RL_OK);
        printf("  a quaternion driven to zero norm: flags %u fault code "
               "%u (%s)\n", ff[0], fc[0], fs.status_str((K26RlStatus)fc[0]));
        ASSERT((ff[0] & 4u) != 0);
        ASSERT(fc[0] == (uint16_t)K26RL_E_DIVERGED);
        fs.destroy(fe);
    }
    n_pass++;

    /* ---- the capture resolution ---------------------------------- *
     *
     * A capture is not one of the resolutions an environment
     * declares, and this is where that is measured. Two programmes
     * are built from the shipped benchmark: one keeps the default
     * arrest, the other declares a bounce at a restitution high
     * enough that an impulse would be unmistakable, and NEITHER
     * terminates on first contact, because what is under test is
     * what happens after it. Without precedence the bounce throws the
     * pair apart on the very step the capture channel reads one.
     *
     * What would make this arm vacuous: a fixture that ends at the
     * contact, which is what the shipped benchmark does and why it
     * cannot serve here; and a separation bound loose enough that a
     * pair merely near each other passes, which is why the bound is
     * on the residuals the port form publishes rather than on the
     * range.
     */
    printf("a capture takes precedence, and the pair becomes one body\n");
    {
        char *src = read_whole_(BENCH, NULL);
        for (int variant = 0; variant < 2; variant++) {
            char *body = (char *)malloc(strlen(src) + 8192);
            ASSERT(body != NULL);
            /* Absolute asset paths, a horizon ending, and for the
             * second variant a declared bounce. */
            const char *cur = src;
            size_t n = 0;
            for (;;) {
                const char *hit = strstr(cur, "assembly=\"assets/");
                if (!hit) break;
                memcpy(body + n, cur, (size_t)(hit - cur));
                n += (size_t)(hit - cur);
                n += (size_t)snprintf(body + n, 1024,
                                      "assembly=\"%s/examples/assets/", cwd);
                cur = hit + strlen("assembly=\"assets/");
            }
            strcpy(body + n, cur);
            char *out2 = (char *)malloc(strlen(body) + 4096);
            ASSERT(out2 != NULL);
            const char *term = "        terminated when touch_hit > 0.5\n";
            const char *tp = strstr(body, term);
            ASSERT(tp != NULL);
            size_t pre = (size_t)(tp - body);
            memcpy(out2, body, pre);
            size_t m2 = pre;
            m2 += (size_t)snprintf(out2 + m2, 512,
                "        terminated when episode.steps > 340\n%s",
                variant ? "        contact bounce restitution 0.9 "
                          "friction 0.1\n" : "");
            strcpy(out2 + m2, tp + strlen(term));
            char path[320], bin[256], so_path[352];
            snprintf(path, sizeof path, "%s/joined%d.kfl", WORK_DIR, variant);
            snprintf(bin, sizeof bin, "%s/joined%d", WORK_DIR, variant);
            rl_write_file_(path, out2);
            free(out2);
            free(body);
            rl_compile_(path, bin, WORK_DIR);
            snprintf(so_path, sizeof so_path, "%s.rlenv.so", bin);
            void *jso = rl_dlopen_(so_path);
            RlSurface js;
            rl_resolve_surface_(jso, &js);
            K26RlEnv *je = NULL;
            ASSERT(js.create(11u, 1u, &je) == K26RL_OK);

            int cap_step = -1;
            double after[8][4];
            int n_after = 0;
            double bodies0[3 * 6], bodies1[3 * 6];
            double sep_at_capture = 0.0;
            for (int k = 0; k < 340; k++) {
                ASSERT(js.obs(je, obs) == K26RL_OK);
                if (cap_step < 0) pilot_(obs, &ch, 0.07, act);
                else              memset(act, 0, sizeof act);
                ASSERT(js.step(je, act) == K26RL_OK);
                ASSERT(js.obs(je, obs) == K26RL_OK);
                ASSERT(js.flags(je, flags) == K26RL_OK);
                ASSERT((flags[0] & 4u) == 0);
                if (cap_step < 0 && obs[ch.dock_cap] != 0.0) {
                    cap_step = k;
                    ASSERT(js.bodies(je, 1u, bodies0, 3 * 6) == 3 * 6);
                    sep_at_capture = sqrt(
                        bodies0[2*6+0]*bodies0[2*6+0] +
                        bodies0[2*6+1]*bodies0[2*6+1] +
                        bodies0[2*6+2]*bodies0[2*6+2]);
                    continue;
                }
                if (cap_step >= 0 && n_after < 8 &&
                    (k - cap_step) % 5 == 0) {
                    after[n_after][0] = obs[ch.dock_axial + 0];
                    after[n_after][1] = obs[ch.dock_axial + 1];
                    after[n_after][2] = obs[ch.dock_v_axial + 0];
                    after[n_after][3] = obs[ch.dock_v_axial + 1];
                    n_after++;
                }
                if (cap_step >= 0 && k > cap_step + 40) break;
            }
            ASSERT(js.bodies(je, 1u, bodies1, 3 * 6) == 3 * 6);
            double sep_after = sqrt(
                bodies1[2*6+0]*bodies1[2*6+0] +
                bodies1[2*6+1]*bodies1[2*6+1] +
                bodies1[2*6+2]*bodies1[2*6+2]);
            printf("  %-7s captured at step %d; separation %.9f m at "
                   "capture, %.9f m forty steps later (drift %.3e m)\n",
                   variant ? "bounce" : "arrest", cap_step,
                   sep_at_capture, sep_after,
                   sep_after - sep_at_capture);
            ASSERT(cap_step >= 0);
            ASSERT(n_after >= 4);
            for (int i = 0; i < n_after; i++) {
                printf("     +%2d steps: axial %+.3e lateral %+.3e "
                       "closing %+.3e lateral rate %+.3e\n",
                       i * 5, after[i][0], after[i][1], after[i][2],
                       after[i][3]);
                /* Mated: the two mating planes stay exactly where
                 * the capture left them. The residuals are not zero
                 * and should not be, the capture having happened at a
                 * small misalignment and the pair having kept it;
                 * what makes the pair one body is that they do not
                 * MOVE. The rates are the rigid rotation carried
                 * across that misalignment, and they are constant for
                 * the same reason.
                 *
                 * A bounce that ran here would part the pair at
                 * centimetres a second, which is metres over these
                 * forty steps, so the bound below is four orders
                 * inside what it has to tell apart. */
                for (int c = 0; c < 4; c++) {
                    ASSERT(fabs(after[i][c] - after[0][c]) < 1e-9);
                }
            }
            /* And the pair holds its separation as a rigid body does,
             * to a bound far below what either resolution would do to
             * it: a bounce at this restitution parts them at
             * centimetres a second, which is metres over forty
             * steps. */
            ASSERT(fabs(sep_after - sep_at_capture) < 1e-6);
            js.destroy(je);
        }
        free(src);
    }
    n_pass++;

    /* ---- the latch tells its two perspectives apart --------------- *
     *
     * The benchmark's own ports lie on their bodies' first axis with
     * an identity basis, so the state of A against B and the state of
     * B against A agree in every channel a contact latches, and
     * nothing above can tell the two apart. This fixture is built so
     * they cannot agree: the two ports sit at different offsets from
     * their own centres, the arriving craft is tilted three degrees
     * off the mating direction, and it carries an angular rate about
     * a skew axis. Both craft publish their own port, and the two
     * published sets are asserted to differ and to be ordered, so
     * exchanging them fails.
     */
    printf("a contact latch resolved from each port's own side\n");
    {
        rl_write_file_(WORK_DIR "/perspa.k26asm",
            "assembly perspa\n"
            "    frame x_to_port\n"
            "    provenance mass \"gate fixture, not a craft\" computed\n"
            "    component hull\n"
            "        mass 1000.0\n"
            "        at 0 0 0\n"
            "        collider box 1.0 0.5 0.5\n"
            "    end\n"
            "    port dock\n"
            "        at 1.2 0.0 0.0\n"
            "        axis 1.0 0.0 0.0\n"
            "        roll_ref 0.0 1.0 0.0\n"
            "        capture idss_e\n"
            "    end\n"
            "end\n");
        rl_write_file_(WORK_DIR "/perspb.k26asm",
            "assembly perspb\n"
            "    frame x_to_port\n"
            "    provenance mass \"gate fixture, not a craft\" computed\n"
            "    component hull\n"
            "        mass 3000.0\n"
            "        at 0 0 0\n"
            "        collider box 1.0 0.5 0.5\n"
            "    end\n"
            "    port dock\n"
            "        at 2.5 0.0 0.0\n"
            "        axis 1.0 0.0 0.0\n"
            "        roll_ref 0.0 1.0 0.0\n"
            "        capture idss_e\n"
            "    end\n"
            "end\n");
        /* Half a turn about the third axis to face the other craft,
         * then three degrees about the second, which is inside the
         * envelope's four and enough that the two mated frames are
         * not the same frame. */
        double half = 3.14159265358979323846 / 2.0;
        double t = 1.5 * 3.14159265358979323846 / 180.0;
        double qy[4] = { cos(half), 0.0, sin(half), 0.0 };   /* face about */
        /* three degrees off the mating direction */
        double qz[4] = { cos(t), 0.0, 0.0, sin(t) };
        double q[4];
        q[0] = qz[0]*qy[0] - qz[1]*qy[1] - qz[2]*qy[2] - qz[3]*qy[3];
        q[1] = qz[0]*qy[1] + qz[1]*qy[0] + qz[2]*qy[3] - qz[3]*qy[2];
        q[2] = qz[0]*qy[2] - qz[1]*qy[3] + qz[2]*qy[0] + qz[3]*qy[1];
        q[3] = qz[0]*qy[3] + qz[1]*qy[2] - qz[2]*qy[1] + qz[3]*qy[0];
        char prog[3072];
        snprintf(prog, sizeof prog,
            "form RL_PERSP\n"
            "fn world w\n"
            "    astro_body one assembly=\"%s/perspa.k26asm\""
            " pos_x=0.0 pos_y=0.0 pos_z=0.0 quat_w=1.0\n"
            "    astro_body two assembly=\"%s/perspb.k26asm\""
            " pos_x=5.3 pos_y=0.03 pos_z=0.01 vel_x=-0.07"
            " quat_w=%.17g quat_x=%.17g quat_y=%.17g quat_z=%.17g"
            " omega_x=0.0010 omega_y=0.0007 omega_z=-0.0005\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        substeps 5\n"
            "        horizon 80\n"
            "        terminated when tc_hit > 0.5\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port dock of one as pa\n"
            "    observe port dock of two as pb\n"
            "    observe contact of one as tc\n"
            "    objective\n"
            "        reward pa_axial + pb_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, q[0], q[1], q[2], q[3]);
        rl_write_file_(WORK_DIR "/persp.kfl", prog);
        rl_compile_(WORK_DIR "/persp.kfl", WORK_DIR "/persp", WORK_DIR);
        void *pso = rl_dlopen_(WORK_DIR "/persp.rlenv.so");
        RlSurface ps;
        rl_resolve_surface_(pso, &ps);
        K26RlEnv *pe = NULL;
        ASSERT(ps.create(3u, 1u, &pe) == K26RL_OK);
        uint8_t pb2[16384];
        int32_t plen = ps.spec(pe, pb2, sizeof pb2);
        ASSERT(plen > 0);
        int a0 = find_channel_(pb2, (uint32_t)plen, "pa_captured");
        int b0 = find_channel_(pb2, (uint32_t)plen, "pb_captured");
        ASSERT(a0 >= 0 && b0 >= 0);
        double pobs[24];
        double pact[1] = { 0.0 };
        int tc = find_channel_(pb2, (uint32_t)plen, "tc_hit");
        ASSERT(tc >= 0);
        int hit_step = -1;
        for (int k = 0; k < 80; k++) {
            ASSERT(ps.step(pe, pact) == K26RL_OK);
            ASSERT(ps.obs(pe, pobs) == K26RL_OK);
            ASSERT(ps.flags(pe, flags) == K26RL_OK);
            if (pobs[tc] != 0.0) { hit_step = k; break; }
        }
        static const char *const nm[8] = {
            "axial", "lateral", "pitchyaw", "roll",
            "v_axial", "v_lateral", "v_pitchyaw", "v_roll"
        };
        printf("  contact at step %d, captured %.0f\n", hit_step,
               pobs[a0]);
        ASSERT(hit_step >= 0);
        int differ = 0;
        for (int i = 0; i < 8; i++) {
            double va = pobs[a0 + 1 + i], vb = pobs[b0 + 1 + i];
            printf("    %-10s one %+.9f   two %+.9f   difference %+.3e\n",
                   nm[i], va, vb, va - vb);
            if (fabs(va - vb) > 1e-6) differ++;
        }
        ASSERT(hit_step >= 0);
        /* The two sides disagree in most of what they publish, which
         * is what makes the assignment measurable at all. */
        printf("  channels that differ between the two sides: %d of 8\n",
               differ);
        ASSERT(differ >= 4);
        /* And they are ordered, in three channels whose measured
         * separation is a thousand times the bound asserted.
         * Exchanging the two sides inverts every one of them, so a
         * latch that stored each state against the wrong port fails
         * here three times over. */
        ASSERT(pobs[a0 + 1] < pobs[b0 + 1] - 1e-3);   /* axial */
        ASSERT(pobs[a0 + 2] > pobs[b0 + 2] + 1e-3);   /* lateral */
        ASSERT(pobs[a0 + 6] > pobs[b0 + 6] + 1e-4);   /* lateral rate */
        printf("  each side's residuals are its own: OK\n");
        ps.destroy(pe);
    }
    n_pass++;

    /* ---- the joint body is the pair's, by momentum ---------------- *
     *
     * In orbit the pair's momentum changes by gravity between one
     * step and the next, which swamps the impulse a capture applies.
     * This fixture has no gravitating body at all: two craft in free
     * space, one closing on the other, so the momentum before the
     * capture and the momentum after it are exactly comparable and
     * the joint velocity is exactly the mass-weighted mean. A joint
     * velocity taken from either craft alone rather than from the
     * pair fails here by the mass ratio.
     */
    printf("the joint body carries the pair's momentum\n");
    {
        char prog[3072];
        snprintf(prog, sizeof prog,
            "form RL_JOINP\n"
            "fn world w\n"
            "    astro_body one assembly=\"%s/perspa.k26asm\""
            " pos_x=0.0 pos_y=0.0 pos_z=0.0 quat_w=1.0\n"
            "    astro_body two assembly=\"%s/perspb.k26asm\""
            " pos_x=5.3 pos_y=0.02 pos_z=0.0 vel_x=-0.07"
            " quat_w=0.0 quat_x=0.0 quat_y=1.0 quat_z=0.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        substeps 5\n"
            "        horizon 90\n"
            "        terminated when episode.steps > 85\n"
            "        contact bounce restitution 0.9 friction 0.1\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port dock of one as pa\n"
            "    observe contact of one as tc\n"
            "    objective\n"
            "        reward pa_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR);
        rl_write_file_(WORK_DIR "/joinp.kfl", prog);
        rl_compile_(WORK_DIR "/joinp.kfl", WORK_DIR "/joinp", WORK_DIR);
        void *jso = rl_dlopen_(WORK_DIR "/joinp.rlenv.so");
        RlSurface js;
        rl_resolve_surface_(jso, &js);
        K26RlEnv *je = NULL;
        ASSERT(js.create(5u, 1u, &je) == K26RL_OK);
        uint8_t jb[16384];
        int32_t jlen = js.spec(je, jb, sizeof jb);
        ASSERT(jlen > 0);
        int ja = find_channel_(jb, (uint32_t)jlen, "pa_captured");
        int jt = find_channel_(jb, (uint32_t)jlen, "tc_hit");
        ASSERT(ja >= 0 && jt >= 0);
        double jobs[16], jact[1] = { 0.0 };
        double before[2 * 6], after2[2 * 6];
        int cap = -1;
        for (int k = 0; k < 90; k++) {
            ASSERT(js.bodies(je, 0u, before, 2 * 6) == 2 * 6);
            ASSERT(js.step(je, jact) == K26RL_OK);
            ASSERT(js.obs(je, jobs) == K26RL_OK);
            if (jobs[jt] != 0.0) {
                cap = k;
                ASSERT(js.bodies(je, 0u, after2, 2 * 6) == 2 * 6);
                break;
            }
        }
        ASSERT(cap >= 0);
        printf("  contact at step %d, captured %.0f\n", cap, jobs[ja]);
        ASSERT(jobs[ja] != 0.0);
        /* The masses are the fixture's own, declared above. */
        const double m1 = 1000.0, m2 = 3000.0, MT = m1 + m2;
        for (int c = 0; c < 3; c++) {
            double p0 = (m1 * before[3 + c] + m2 * before[6 + 3 + c]) / MT;
            double p1 = (m1 * after2[3 + c] + m2 * after2[6 + 3 + c]) / MT;
            printf("    axis %d: before %+.12f and %+.12f, after %+.12f "
                   "and %+.12f, mean %+.12f -> %+.12f\n", c,
                   before[3 + c], before[6 + 3 + c], after2[3 + c],
                   after2[6 + 3 + c], p0, p1);
            /* The pair's momentum is what a capture may not create or
             * destroy: the mass-weighted mean velocity is the same
             * before and after. The two craft do not end at that mean
             * individually, and should not: the capture happened off
             * the line of centres, so the pair turns, and each craft
             * carries the rotation's own velocity at its own centre.
             * What the mean removes is exactly that rotation, since
             * the mass-weighted offsets from the centre of mass sum
             * to zero by definition. */
            /* The bound is not machine precision: the projection
             * runs again on each sub-advance after the capture, and
             * the pair's own rotation carries the two centres a
             * little between them. It is five orders inside what it
             * has to tell apart, a joint velocity taken from one
             * craft alone differing from the mean by the mass ratio
             * times the closing rate, which is 1.75e-2 here. */
            near_("joint momentum conserved", p1, p0, 1e-7);
        }
        /* The pair does turn, so the two craft do not leave at the
         * same velocity, which is what makes the mean above a
         * measurement rather than a restatement. */
        ASSERT(fabs(after2[3] - after2[6 + 3]) > 1e-7);
        /* And the arm can tell the mean from either craft's own
         * velocity: they differ by the mass ratio times the closing
         * rate, which is a thousand times the bound above. */
        ASSERT(fabs(before[3] - before[6 + 3]) > 0.06);
        js.destroy(je);
    }
    n_pass++;

    /* ---- 6. two processes, byte for byte ------------------------- */
    printf("the whole run is byte identical across two processes\n");
    rl_run_or_die_(WORK_DIR "/dock --envs 2 --episodes 2 --seed 11 --out "
                   WORK_DIR "/a.k26epi");
    rl_run_or_die_(WORK_DIR "/dock --envs 2 --episodes 2 --seed 11 --out "
                   WORK_DIR "/b.k26epi");
    ASSERT(rl_files_equal_(WORK_DIR "/a.k26epi", WORK_DIR "/b.k26epi"));
    printf("  two processes at one seed produce identical episode files: OK\n");
    n_pass++;

    free(obs);
    free(csrc);
    printf("test_rl_docking: %d gate(s) passed\n", n_pass);
    return 0;
}
