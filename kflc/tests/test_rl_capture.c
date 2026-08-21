/* test_rl_capture.c: declared envelopes, explicit pairing, and the
 * join chain.
 *
 * The docking gate covers one pairing judged against the compiler's
 * own envelope. This one covers what a world of several small craft
 * needs: envelopes a program declares, statements that say which port
 * they are measured against, and joins that chain rather than stopping
 * at the first pair.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   An envelope arm that read the artifact's constants and compared
 *   them against the artifact would agree with any conversion error.
 *   The declared figures are written here in the units the block takes
 *   and converted here; the artifact's own constants are read out of
 *   the emitted source and compared against those.
 *
 *   A refusal arm that only asserted a non-zero exit status would pass
 *   for a compiler that refused everything. Each names the substring
 *   the diagnostic must carry, and each fixture differs from an
 *   accepted one in exactly the field the refusal is about.
 *
 *   A composite-mass arm that took the pair's mass from the artifact
 *   would agree with any error in it. The masses are the fixtures' own
 *   declarations, written here, and the expected answer is arithmetic
 *   on them; the alternatives the arm rules out are the single-craft
 *   answers, orders of magnitude away from the bound asserted.
 *
 *   A chain arm that only asserted that two joins existed would pass
 *   for a runtime that formed them and then ignored one. The laden
 *   re-dock is measured by what the whole answers to a thrust, which
 *   is the three masses summed and nothing else.
 *
 *   A cargo-visibility arm that only asserted a contact channel would
 *   pass for a runtime reporting the leader's contact rather than the
 *   cargo's. The third body approaches the cargo alone, on a line that
 *   reaches nothing else, and the cargo's own contact channel is what
 *   is read.
 *
 *   A compatibility arm comparing today's binary with itself would
 *   agree with any change. It builds the compiler from the commit
 *   before this work, compiles the same two programs with both, and
 *   compares the episode records byte for byte.
 */
#define _GNU_SOURCE
#include "rl_gate_util.h"

#include <math.h>
#include <sys/wait.h>
#include <unistd.h>

#define WORK_DIR "/tmp/kflc_rl_capture_test"

/* The commit whose compiler the compatibility arm builds. It is the
 * one before the work this gate covers, so the two binaries differ by
 * exactly that work. */
#define PRIOR_COMMIT "3308367"

static int n_pass;

static void near_(const char *what, double got, double want, double tol)
{
    double e = got - want;
    if (e < 0.0) e = -e;
    printf("  %-46s %+.12g  want %+.12g  err %.3e\n", what, got, want, e);
    ASSERT(e <= tol);
}

static int run_(const char *cmd)
{
    int rc = system(cmd);
    if (rc < 0) return -1;
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* ---- reading the artifact's own constants ------------------------ */

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
            double v = strtod(p, &end);
            if (end && end != p) {
                *out = v;
                return end;
            }
        }
        p++;
    }
    return NULL;
}

static char *slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    long sz = ftell(f);
    ASSERT(sz >= 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char *buf = (char *)malloc((size_t)sz + 1);
    ASSERT(buf != NULL);
    if (sz > 0) ASSERT(fread(buf, 1, (size_t)sz, f) == (size_t)sz);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* Emit a program's C++ and hand back the whole of it. */
static char *emit_(const char *kfl)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "./bin/kflc --emit %s > " WORK_DIR "/emit.cc 2> "
             WORK_DIR "/emit.err", kfl);
    if (run_(cmd) != 0) {
        char show[256];
        snprintf(show, sizeof show, "cat " WORK_DIR "/emit.err");
        (void)!system(show);
        fprintf(stderr, "emit failed for %s\n", kfl);
        exit(1);
    }
    return slurp_(WORK_DIR "/emit.cc");
}

/* One port's eight envelope constants, as the compiler emitted them,
 * plus the mating plate's three half extents. */
typedef struct {
    double env[8];
    double plate[3];
} GateEnv;

/* Read port `which` out of the emitted table, and the plate collider
 * the reader built for it. The plate is found by its own shape entry:
 * the port table carries the shape index within the vehicle's slice,
 * and the fixtures below give each craft one hull box and one plate,
 * so the plate is the vehicle's second shape. */
static void read_port_(const char *src, int which, GateEnv *out)
{
    const char *p = strstr(src, "static const KflrlPort kflrl_ports_[] = {");
    ASSERT(p != NULL);
    /* Each entry is: veh, shape, three centre-of-mass components,
     * three port-position components, nine basis components, then the
     * eight envelope limits. */
    for (int i = 0; i <= which; i++) {
        double skip;
        for (int k = 0; k < 17; k++) {
            p = scan_num_(p, &skip);
            ASSERT(p != NULL);
        }
        for (int k = 0; k < 8; k++) {
            p = scan_num_(p, &out->env[k]);
            ASSERT(p != NULL);
        }
    }
    const char *c = strstr(src, "static const K26AstroCollShape kflrl_coll_[] = {");
    ASSERT(c != NULL);
    /* Shape entries are: kind, three centre components, nine axis
     * components, three half extents. The plate of vehicle `which` is
     * that vehicle's second shape, and each vehicle here has two. */
    for (int i = 0; i < 2 * which + 2; i++) {
        double skip;
        for (int k = 0; k < 13; k++) {
            c = scan_num_(c, &skip);
            ASSERT(c != NULL);
        }
        for (int k = 0; k < 3; k++) {
            c = scan_num_(c, &out->plate[k]);
            ASSERT(c != NULL);
        }
    }
}

/* ---- fixtures ---------------------------------------------------- */

/* One craft: a box hull and one port on the positive first axis. */
static void write_craft_(const char *path, const char *name, double mass,
                         double hx, double hy, double hz,
                         const char *port, double at_x, const char *env)
{
    char buf[2048];
    snprintf(buf, sizeof buf,
        "assembly %s\n"
        "    frame x_to_port\n"
        "    provenance mass \"gate fixture, not a craft\" computed\n"
        "    component hull\n"
        "        mass %.17g\n"
        "        at 0 0 0\n"
        "        collider box %.17g %.17g %.17g\n"
        "    end\n"
        "    port %s\n"
        "        at %.17g 0.0 0.0\n"
        "        axis 1.0 0.0 0.0\n"
        "        roll_ref 0.0 1.0 0.0\n"
        "        capture %s\n"
        "    end\n"
        "end\n",
        name, mass, hx, hy, hz, port, at_x, env);
    rl_write_file_(path, buf);
}

/* A plain box with no port at all: something to strike a chain with,
 * whose only geometry is the one collider its contact is taken on. */
static void write_block_(const char *path, const char *name, double mass,
                         double hx, double hy, double hz)
{
    char buf[1024];
    snprintf(buf, sizeof buf,
        "assembly %s\n"
        "    frame x_to_port\n"
        "    provenance mass \"gate fixture, not a craft\" computed\n"
        "    component hull\n"
        "        mass %.17g\n"
        "        at 0 0 0\n"
        "        collider box %.17g %.17g %.17g\n"
        "    end\n"
        "end\n", name, mass, hx, hy, hz);
    rl_write_file_(path, buf);
}

/* The same with a second port on the negative first axis, and an
 * optional thruster on the hull centre line. */
static void write_craft2_(const char *path, const char *name, double mass,
                          double hx, double hy, double hz,
                          const char *port_a, double at_a, const char *env_a,
                          const char *port_b, double at_b, const char *env_b,
                          double thrust, double lat_thrust)
{
    char buf[4096];
    int n = snprintf(buf, sizeof buf,
        "assembly %s\n"
        "    frame x_to_port\n"
        "    provenance mass \"gate fixture, not a craft\" computed\n"
        "    component hull\n"
        "        mass %.17g\n"
        "        at 0 0 0\n"
        "        collider box %.17g %.17g %.17g\n"
        "    end\n"
        "    port %s\n"
        "        at %.17g 0.0 0.0\n"
        "        axis 1.0 0.0 0.0\n"
        "        roll_ref 0.0 1.0 0.0\n"
        "        capture %s\n"
        "    end\n"
        "    port %s\n"
        "        at %.17g 0.0 0.0\n"
        "        axis -1.0 0.0 0.0\n"
        "        roll_ref 0.0 1.0 0.0\n"
        "        capture %s\n"
        "    end\n",
        name, mass, hx, hy, hz, port_a, at_a, env_a, port_b, at_b, env_b);
    if (thrust > 0.0) {
        n += snprintf(buf + n, sizeof buf - (size_t)n,
            "    thruster main\n"
            "        at 0.0 0.0 0.0\n"
            "        dir 1.0 0.0 0.0\n"
            "        thrust %.17g\n"
            "    end\n", thrust);
    }
    /* Across the line of centres, and on the craft's own centre, so
     * its whole lever arm about a composite is that craft's offset
     * from the composite centre and nothing else. */
    if (lat_thrust > 0.0) {
        n += snprintf(buf + n, sizeof buf - (size_t)n,
            "    thruster lat\n"
            "        at 0.0 0.0 0.0\n"
            "        dir 0.0 1.0 0.0\n"
            "        thrust %.17g\n"
            "    end\n", lat_thrust);
    }
    n += snprintf(buf + n, sizeof buf - (size_t)n, "end\n");
    ASSERT((size_t)n < sizeof buf);
    rl_write_file_(path, buf);
}

/* The envelope the chain fixtures declare, in the units the block
 * takes. The limits are wide because a gate fixture closes in tens of
 * steps rather than in the tens of minutes a real approach takes; what
 * they are is the program's own statement and nothing here reads them
 * from anywhere else. */
#define GRASP_BLOCK \
    "    capture_envelope grasp_s\n" \
    "        axial_rate 0.00 0.60\n" \
    "        lateral_rate 0.50\n" \
    "        pitchyaw_rate 30.0\n" \
    "        roll_rate 30.0\n" \
    "        lateral 0.30\n" \
    "        pitchyaw 20.0\n" \
    "        roll 180.0\n" \
    "        diameter 700.0\n" \
    "    end\n"

/* A world with no gravitating body would leave the craft at index 0,
 * and the default integrator treats index 0 as the immobile central
 * mass: its kick and drift loops run from index 1, so a thruster on
 * body 0 pushes nothing. The anchor occupies that index. Its mass is
 * also why it carries a positive gm: the drift is a Kepler
 * propagation about body 0, and a central mass of zero does not
 * converge. The field it makes at the craft is about 1e-24 metres per
 * second squared, which is twenty-three orders below anything
 * measured here. */
#define ANCHOR_BODY \
    "    astro_body anchor gm=1.0e-6 mass=1.0" \
    " pos_x=0.0 pos_y=1.0e9 pos_z=0.0\n"

/* Every craft below carries the same velocity across the line it
 * works on, and it is not decoration.
 *
 * The drift about the anchor is a Kepler propagation, and at this
 * separation the anchor's escape speed is about 4.5e-8 metres per
 * second. A craft left at rest is inside that, its own neighbours
 * pull it through it within a minute of simulated time, and the orbit
 * passes through the parabolic case where a two-body propagation is
 * at its worst conditioned: measured, a craft crossed it and was
 * placed eleven million kilometres away on the next step. One shared
 * velocity seven orders above escape keeps every craft firmly on a
 * hyperbolic path, where the propagation is well behaved, and changes
 * nothing any arm reads: it is common to all of them, so every
 * relative position, rate and residual is what it would be without
 * it, and the deflection the anchor makes over a whole run is 1e-22
 * metres per second. */
#define DRIFT " vel_z=1.0"

static void compile_(const char *stem, const char *src)
{
    char kfl[320], out[320];
    snprintf(kfl, sizeof kfl, WORK_DIR "/%s.kfl", stem);
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    rl_write_file_(kfl, src);
    rl_compile_(kfl, out, WORK_DIR);
}

static void *open_(const char *stem, RlSurface *s)
{
    char so[320];
    snprintf(so, sizeof so, WORK_DIR "/%s.rlenv.so", stem);
    void *h = rl_dlopen_(so);
    rl_resolve_surface_(h, s);
    return h;
}

/* Compile a fixture that must be refused, and assert on what the
 * diagnostic says. */
static void refuse_(const char *what, const char *src, const char *expect)
{
    rl_write_file_(WORK_DIR "/bad.kfl", src);
    int rc = run_("./bin/kflc --check " WORK_DIR "/bad.kfl > "
                  WORK_DIR "/bad.log 2>&1");
    char *log = slurp_(WORK_DIR "/bad.log");
    if (rc == 0 || strstr(log, expect) == NULL) {
        fprintf(stderr, "FAIL %s: rc=%d, log lacks \"%s\"\n---\n%s---\n%s",
                what, rc, expect, log, src);
        exit(1);
    }
    printf("  refused: %-40s (%s)\n", what, expect);
    free(log);
}

int main(void)
{
    if (!rl_libs_present_("test_rl_capture")) return 77;
    ASSERT(run_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR) == 0);

    /* ---- 1. the declared envelope's figures, converted once ------- *
     *
     * The block's own numbers are written here in the units the block
     * takes and converted here; the artifact's constants are read out
     * of the emitted source. A wrong conversion in the compiler, or a
     * field carried into the wrong slot, fails here rather than
     * agreeing with itself.
     */
    printf("a declared envelope's figures, converted once\n");
    {
        /* The figures this fixture declares, and nothing reads them
         * from anywhere else. */
        const double AXIAL_LO = 0.00, AXIAL_HI = 0.05;
        const double LATERAL_RATE = 0.02;
        const double PITCHYAW_RATE_DEG = 3.0, ROLL_RATE_DEG = 3.0;
        const double LATERAL_M = 0.05;
        const double PITCHYAW_DEG = 10.0, ROLL_DEG = 180.0;
        const double DIAMETER_MM = 700.0;
        const double D2R = 3.14159265358979323846 / 180.0;

        write_craft_(WORK_DIR "/d1.k26asm", "d1", 100.0, 0.5, 0.4, 0.4,
                     "grasp", 0.6, "grasp_s");
        write_craft_(WORK_DIR "/d2.k26asm", "d2", 40.0, 0.4, 0.4, 0.4,
                     "face", 0.9, "grasp_s");
        char prog[4096];
        snprintf(prog, sizeof prog,
            "form RL_CAPENV\n"
            "fn world w\n"
            "    capture_envelope grasp_s\n"
            "        axial_rate %.17g %.17g\n"
            "        lateral_rate %.17g\n"
            "        pitchyaw_rate %.17g\n"
            "        roll_rate %.17g\n"
            "        lateral %.17g\n"
            "        pitchyaw %.17g\n"
            "        roll %.17g\n"
            "        diameter %.17g\n"
            "    end\n"
            ANCHOR_BODY
            "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT
            " pos_x=0.0 quat_w=1.0\n"
            "    astro_body rock assembly=\"%s/d2.k26asm\"" DRIFT
            " pos_x=3.0 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 20\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone against face of rock as g1\n"
            "    objective\n"
            "        reward g1_axial\n"
            "    end\n"
            "end\n"
            "end\n",
            AXIAL_LO, AXIAL_HI, LATERAL_RATE, PITCHYAW_RATE_DEG,
            ROLL_RATE_DEG, LATERAL_M, PITCHYAW_DEG, ROLL_DEG, DIAMETER_MM,
            WORK_DIR, WORK_DIR);
        rl_write_file_(WORK_DIR "/capenv.kfl", prog);
        char *src = emit_(WORK_DIR "/capenv.kfl");

        static const char *const NM[8] = {
            "axial rate lower", "axial rate upper", "lateral rate",
            "pitch/yaw rate", "roll rate", "lateral misalignment",
            "pitch/yaw misalignment", "roll misalignment"
        };
        const double want[8] = {
            AXIAL_LO, AXIAL_HI, LATERAL_RATE,
            PITCHYAW_RATE_DEG * D2R, ROLL_RATE_DEG * D2R,
            LATERAL_M, PITCHYAW_DEG * D2R, ROLL_DEG * D2R
        };
        for (int q = 0; q < 2; q++) {
            GateEnv g;
            read_port_(src, q, &g);
            for (int k = 0; k < 8; k++) {
                char what[96];
                snprintf(what, sizeof what, "port %d %s", q, NM[k]);
                near_(what, g.env[k], want[k], 1e-15);
            }
            /* The mating plate is built from the declared diameter and
             * from nothing else: half its own radius across, and a
             * hundredth of that radius thick. */
            double radius = 0.5 * DIAMETER_MM / 1000.0;
            char what[96];
            snprintf(what, sizeof what, "port %d plate half thickness", q);
            near_(what, g.plate[0], radius * 0.01, 1e-15);
            snprintf(what, sizeof what, "port %d plate half width", q);
            near_(what, g.plate[1], radius, 1e-15);
            snprintf(what, sizeof what, "port %d plate half height", q);
            near_(what, g.plate[2], radius, 1e-15);
        }
        free(src);
    }
    n_pass++;

    /* ---- 2. one refusal per rule --------------------------------- *
     *
     * Each fixture below differs from the accepted one above in
     * exactly the field its rule is about, and each asserts on what
     * the diagnostic says rather than only on the exit status.
     */
    printf("one refusal per envelope rule\n");
    {
        static const struct { const char *tag, *block, *expect; } BAD_[] = {
            { "duplicate declaration",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n"
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "an envelope of that name is already declared at line" },
            { "a name the compiler already defines",
              "    capture_envelope idss_e\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "may not shadow one" },
            { "a missing field",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        diameter 700.0\n    end\n",
              "`roll` is not declared, and every field is required" },
            { "a band with its bounds the wrong way about",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.20 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "has its lower bound above its upper" },
            { "a negative rate",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate -0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "`lateral_rate` is -0.02 m/s, and a limit on a rate or a "
              "misalignment is a magnitude, so it cannot be negative" },
            { "a closing band whose lower bound is negative",
              "    capture_envelope grasp_s\n"
              "        axial_rate -0.10 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "`axial_rate`'s lower bound is -0.10000000000000001 m/s" },
            { "a value that is not a finite number",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate inf\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "`lateral_rate` is inf, which is not a finite number" },
            { "an angle past half a turn",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 200.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "`pitchyaw` is 200 degrees, and an angular misalignment "
              "runs from 0 to 180" },
            { "a diameter of nothing",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 0.0\n    end\n",
              "`diameter` is 0 mm, and the mating plane it sizes is a "
              "real interface, so it must be positive" },
            { "a field declared twice",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        lateral_rate 0.03\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "is declared twice, and one field states one limit" },
            { "a field taking the wrong count",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.05\n        lateral_rate 0.02\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "`axial_rate` takes 2 values and 1 was given" },
            { "a field that is not a number",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 20mm\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "takes numbers and `20mm` is not one" },
            { "an unknown field",
              "    capture_envelope grasp_s\n"
              "        axial_rate 0.0 0.05\n        lateral_rate 0.02\n"
              "        yaw_rate 3.0\n"
              "        pitchyaw_rate 3.0\n        roll_rate 3.0\n"
              "        lateral 0.05\n        pitchyaw 10.0\n"
              "        roll 4.0\n        diameter 700.0\n    end\n",
              "unknown field `yaw_rate`" }
        };
        for (size_t i = 0; i < sizeof BAD_ / sizeof BAD_[0]; i++) {
            char prog[4096];
            snprintf(prog, sizeof prog,
                "form RL_CAPBAD\n"
                "fn world w\n"
                "%s"
                ANCHOR_BODY
                "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT
                " pos_x=0.0 quat_w=1.0\n"
                "    astro_body rock assembly=\"%s/d2.k26asm\"" DRIFT
                " pos_x=3.0 quat_w=0.0 quat_y=1.0\n"
                "    episode\n"
                "        control_dt 0.5\n"
                "        horizon 20\n"
                "    end\n"
                "    action push box -1.0 1.0 default 0.0\n"
                "    observe port grasp of drone against face of rock as g1\n"
                "    objective\n"
                "        reward g1_axial\n"
                "    end\n"
                "end\n"
                "end\n", BAD_[i].block, WORK_DIR, WORK_DIR);
            refuse_(BAD_[i].tag, prog, BAD_[i].expect);
        }
        /* And a port naming an envelope no block declared, so the
         * union the mark resolves against is measured from both
         * sides. */
        write_craft_(WORK_DIR "/dz.k26asm", "dz", 100.0, 0.5, 0.4, 0.4,
                     "grasp", 0.6, "grasp_z");
        {
            char prog[4096];
            snprintf(prog, sizeof prog,
                "form RL_CAPMISS\n"
                "fn world w\n"
                GRASP_BLOCK
                ANCHOR_BODY
                "    astro_body drone assembly=\"%s/dz.k26asm\"" DRIFT
                " pos_x=0.0 quat_w=1.0\n"
                "    astro_body rock assembly=\"%s/d2.k26asm\"" DRIFT
                " pos_x=3.0 quat_w=0.0 quat_y=1.0\n"
                "    episode\n"
                "        control_dt 0.5\n"
                "        horizon 20\n"
                "    end\n"
                "    action push box -1.0 1.0 default 0.0\n"
                "    observe port grasp of drone as g1\n"
                "    objective\n"
                "        reward g1_axial\n"
                "    end\n"
                "end\n"
                "end\n", WORK_DIR, WORK_DIR);
            refuse_("a mark naming no envelope in scope", prog,
                    "`capture grasp_z` names no envelope in scope");
        }
    }
    n_pass++;

    /* ---- 3. explicit pairing ------------------------------------- *
     *
     * Three candidate ports on three other bodies. The clause resolves
     * against the one it names, the clause-free statement is refused
     * naming the candidates, and two ports of different envelopes are
     * refused naming both.
     */
    printf("`against` resolves, and what it is needed for is refused\n");
    {
        write_craft_(WORK_DIR "/c1.k26asm", "c1", 100.0, 0.5, 0.4, 0.4,
                     "face", 0.9, "grasp_s");
        write_craft_(WORK_DIR "/c2.k26asm", "c2", 100.0, 0.5, 0.4, 0.4,
                     "face", 0.9, "grasp_s");
        write_craft_(WORK_DIR "/c3.k26asm", "c3", 100.0, 0.5, 0.4, 0.4,
                     "face", 0.9, "grasp_s");
        /* One more, identical but for the envelope it names, so the
         * mismatch arm differs from the accepted fixture in that
         * alone. */
        write_craft_(WORK_DIR "/c4.k26asm", "c4", 100.0, 0.5, 0.4, 0.4,
                     "face", 0.9, "grasp_t");
#define THREE_CANDIDATES \
            ANCHOR_BODY \
            "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT \
            " pos_x=0.0 quat_w=1.0\n" \
            "    astro_body r1 assembly=\"%s/c1.k26asm\"" DRIFT \
            " pos_x=20.0 quat_w=0.0 quat_y=1.0\n" \
            "    astro_body r2 assembly=\"%s/c2.k26asm\"" DRIFT \
            " pos_x=40.0 quat_w=0.0 quat_y=1.0\n" \
            "    astro_body r3 assembly=\"%s/c3.k26asm\"" DRIFT \
            " pos_x=60.0 quat_w=0.0 quat_y=1.0\n"
        char prog[6144];
        snprintf(prog, sizeof prog,
            "form RL_PAIR\n"
            "fn world w\n"
            GRASP_BLOCK
            THREE_CANDIDATES
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 20\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone against face of r2 as g2\n"
            "    objective\n"
            "        reward g2_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR, WORK_DIR);
        compile_("pair", prog);
        RlSurface s;
        void *so = open_("pair", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(3u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int ax = find_channel_(blob, (uint32_t)len, "g2_axial");
        ASSERT(ax >= 0);
        double obs[32], act[1] = { 0.0 };
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, obs) == K26RL_OK);
        /* The drone's port plane stands at 0.6 and r2's at 40 - 0.9,
         * so the axial residual is 38.5 if the clause resolved against
         * r2, 18.5 against r1 and 58.5 against r3. The three answers
         * are twenty metres apart and the arm reads one of them. */
        near_("`against r2` measures against r2", obs[ax], 38.5, 1e-3);
        printf("  the two nearer candidates would read %+.1f and %+.1f\n",
               18.5, 58.5);
        s.destroy(env);
        dlclose(so);

        snprintf(prog, sizeof prog,
            "form RL_PAIRAMB\n"
            "fn world w\n"
            GRASP_BLOCK
            THREE_CANDIDATES
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 20\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone as g2\n"
            "    objective\n"
            "        reward g2_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR, WORK_DIR);
        refuse_("three candidates and no clause", prog,
                "3 ports carrying a capture envelope are declared on "
                "other bodies (`face of r1`, `face of r2`, `face of r3`)");

        snprintf(prog, sizeof prog,
            "form RL_PAIRENV\n"
            "fn world w\n"
            GRASP_BLOCK
            "    capture_envelope grasp_t\n"
            "        axial_rate 0.00 0.60\n"
            "        lateral_rate 0.50\n"
            "        pitchyaw_rate 30.0\n"
            "        roll_rate 30.0\n"
            "        lateral 0.30\n"
            "        pitchyaw 20.0\n"
            "        roll 180.0\n"
            "        diameter 700.0\n"
            "    end\n"
            ANCHOR_BODY
            "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT
            " pos_x=0.0 quat_w=1.0\n"
            "    astro_body r4 assembly=\"%s/c4.k26asm\"" DRIFT
            " pos_x=20.0 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 20\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone against face of r4 as g4\n"
            "    objective\n"
            "        reward g4_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR);
        refuse_("two ports of different envelopes", prog,
                "`grasp of drone` names capture envelope `grasp_s` and "
                "`face of r4` names `grasp_t`");
    }
    n_pass++;

    /* ---- 4. two pairings in one episode -------------------------- *
     *
     * Two disjoint pairs capture at different steps. Both joins hold,
     * the episode runs on to its own clause, and each composite is
     * measured by what it answers to a thrust: the two masses summed,
     * and neither craft's own.
     *
     * The two craft of each pair lie on one line with their thrusters
     * on it, so the line of action passes through the pair's centre of
     * mass and the composite inertia enters with a zero lever arm. The
     * rotational half of the composite is measured on its own, in the
     * docking suite's tensor arm.
     */
    printf("two pairings in one episode, both joined\n");
    {
        /* The fixtures' own numbers, in one place. */
        const double mA1 = 1000.0, mA2 = 3000.0;
        const double mB1 = 500.0,  mB2 = 1500.0;
        const double THRUST = 400.0, DT = 0.5;
        const int BURN = 4;
        enum { ANCHOR = 0, A1 = 1, A2 = 2, B1 = 3, B2 = 4, NB = 5 };
        const int32_t nvals = (int32_t)(NB * 6);

        write_craft2_(WORK_DIR "/a1.k26asm", "a1", mA1, 0.5, 0.4, 0.4,
                      "grasp", 0.6, "grasp_s", "tail", -0.6, "grasp_s",
                      THRUST, THRUST);
        write_craft_(WORK_DIR "/a2.k26asm", "a2", mA2, 0.5, 0.4, 0.4,
                     "face", 0.9, "grasp_s");
        write_craft2_(WORK_DIR "/b1.k26asm", "b1", mB1, 0.5, 0.4, 0.4,
                      "grasp", 0.6, "grasp_s", "tail", -0.6, "grasp_s",
                      THRUST, 0.0);
        write_craft_(WORK_DIR "/b2.k26asm", "b2", mB2, 0.5, 0.4, 0.4,
                     "face", 0.9, "grasp_s");

        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_TWOJOIN\n"
            "fn world w\n"
            GRASP_BLOCK
            ANCHOR_BODY
            "    astro_body a1 assembly=\"%s/a1.k26asm\"" DRIFT
            " pos_x=0.0 pos_y=0.0 quat_w=1.0\n"
            "    astro_body a2 assembly=\"%s/a2.k26asm\"" DRIFT
            " pos_x=3.0 pos_y=0.0 vel_x=-0.3 quat_w=0.0 quat_y=1.0\n"
            "    astro_body b1 assembly=\"%s/b1.k26asm\"" DRIFT
            " pos_x=0.0 pos_y=500.0 quat_w=1.0\n"
            "    astro_body b2 assembly=\"%s/b2.k26asm\"" DRIFT
            " pos_x=5.0 pos_y=500.0 vel_x=-0.3 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt %.17g\n"
            "        substeps 5\n"
            "        horizon 400\n"
            "        terminated when episode.steps > 300\n"
            "        contact bounce restitution 0.9 friction 0.1\n"
            "    end\n"
            "    action pa box 0.0 1.0 default 0.0\n"
            "    action pb box 0.0 1.0 default 0.0\n"
            "    action qa box 0.0 1.0 default 0.0\n"
            "    on_step\n"
            "        a1.main.throttle = pa\n"
            "        b1.main.throttle = pb\n"
            "        a1.lat.throttle = qa\n"
            "    end\n"
            "    observe port grasp of a1 against face of a2 full as ga\n"
            "    observe port grasp of b1 against face of b2 full as gb\n"
            "    objective\n"
            "        reward ga_axial + gb_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR, WORK_DIR, DT);
        compile_("twojoin", prog);
        RlSurface s;
        void *so = open_("twojoin", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(7u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int ca = find_channel_(blob, (uint32_t)len, "ga_captured");
        int cb = find_channel_(blob, (uint32_t)len, "gb_captured");
        int ja = find_channel_(blob, (uint32_t)len, "ga_joined");
        int jb = find_channel_(blob, (uint32_t)len, "gb_joined");
        ASSERT(ca >= 0 && cb >= 0 && ja >= 0 && jb >= 0);

        double obs[64], coast[3] = { 0.0, 0.0, 0.0 };
        uint32_t flags[1];
        int step_a = -1, step_b = -1;
        for (int k = 0; k < 200; k++) {
            ASSERT(s.step(env, coast) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.flags(env, flags) == K26RL_OK);
            if (obs[ca] != 0.0 && step_a < 0) step_a = k;
            if (obs[cb] != 0.0 && step_b < 0) step_b = k;
            if (step_a >= 0 && step_b >= 0) break;
        }
        printf("  captures at steps %d and %d\n", step_a, step_b);
        ASSERT(step_a >= 0 && step_b >= 0);
        ASSERT(step_a != step_b);
        /* The pulses were the two captures; the standing state is the
         * `full` mark's channel, and both hold. */
        ASSERT(obs[ja] == 1.0 && obs[jb] == 1.0);
        /* And the episode did not end on either of them: the endings
         * a program gets are the ones it declared. */
        ASSERT((flags[0] & 0x1u) == 0u);
        printf("  both joins hold and the episode runs on: OK\n");

        /* The composites, by what each answers to its own thruster.
         * The line of action runs through each pair's centre of mass,
         * so the prediction is the mass sum. */
        double s0[NB * 6], s1[NB * 6];
        double fire[3] = { 1.0, 1.0, 0.0 };
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s0, (uint32_t)nvals)
               == nvals);
        for (int k = 0; k < BURN; k++) {
            ASSERT(s.step(env, fire) == K26RL_OK);
        }
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s1, (uint32_t)nvals)
               == nvals);
        {
            double mA = mA1 + mA2, mB = mB1 + mB2;
            double vA0 = (mA1 * s0[A1 * 6 + 3] + mA2 * s0[A2 * 6 + 3]) / mA;
            double vA1 = (mA1 * s1[A1 * 6 + 3] + mA2 * s1[A2 * 6 + 3]) / mA;
            double vB0 = (mB1 * s0[B1 * 6 + 3] + mB2 * s0[B2 * 6 + 3]) / mB;
            double vB1 = (mB1 * s1[B1 * 6 + 3] + mB2 * s1[B2 * 6 + 3]) / mB;
            near_("first pair answers its own summed mass", vA1 - vA0,
                  THRUST * BURN * DT / mA, 1e-6);
            near_("second pair answers its own summed mass", vB1 - vB0,
                  THRUST * BURN * DT / mB, 1e-6);
            /* The four answers this rules out are the two craft of
             * each pair taken alone, the nearest of them a third away
             * from what is measured and the bound five orders below
             * that. */
            ASSERT(fabs((vA1 - vA0) - THRUST * BURN * DT / mA1) > 0.5);
            ASSERT(fabs((vA1 - vA0) - THRUST * BURN * DT / mA2) > 0.06);
            ASSERT(fabs((vB1 - vB0) - THRUST * BURN * DT / mB1) > 1.0);
            ASSERT(fabs((vB1 - vB0) - THRUST * BURN * DT / mB2) > 0.1);
            /* Each pair moved as one and neither pair moved the
             * other, which is what says the two joins are two. */
            ASSERT(fabs((s1[A1 * 6 + 3] - s0[A1 * 6 + 3]) -
                        (s1[A2 * 6 + 3] - s0[A2 * 6 + 3])) < 1e-9);
            ASSERT(fabs((s1[B1 * 6 + 3] - s0[B1 * 6 + 3]) -
                        (s1[B2 * 6 + 3] - s0[B2 * 6 + 3])) < 1e-9);
        }
        /* The frozen geometry: the two mating planes meet, so the
         * craft's centres stand at the sum of the two port arms
         * apart. That figure is arithmetic on the fixture's own
         * declarations and the measurement is the body-state
         * getter's. */
        {
            double sep = s1[A2 * 6] - s1[A1 * 6];
            near_("first pair's centres, at the two port arms", sep,
                  0.6 + 0.9, 2e-3);
        }

        /* Where the composite centre of mass sits on that line, and
         * what the composite turns like about it.
         *
         * A separation alone does not locate the centre: the centre
         * enters behaviour only through the lever arms a rotation is
         * taken on, so it is reachable only by turning the pair. The
         * lighter craft's transverse thruster runs through its own
         * centre and therefore has a lever arm about the composite
         * centre equal to that craft's own offset from it, and both
         * the arm and the tensor are hand-computed from the fixture's
         * declarations. A wrong centre with a correct separation
         * changes the arm and fails here; the arm above cannot see it.
         *
         * The torque is exactly constant as the pair turns: the
         * thrust direction and the lever arm both turn with the
         * lighter craft and both lie across the axis the pair turns
         * about, and a rotation about that axis leaves the composite
         * tensor's third component unchanged.
         */
        {
            const double HX = 0.5, HY = 0.4;   /* box half extents */
            const double sep = 0.6 + 0.9;
            const double mt = mA1 + mA2;
            const double j1 = mA1 / 3.0 * (HX * HX + HY * HY);
            const double j2 = mA2 / 3.0 * (HX * HX + HY * HY);
            const double d1 = mA2 * sep / mt;   /* the light craft's arm */
            const double d2 = mA1 * sep / mt;
            const double j_pair = j1 + mA1 * d1 * d1 + j2 + mA2 * d2 * d2;
            /* The lighter craft sits on the negative side of the
             * composite centre and thrusts along the positive second
             * axis, so the turn is about the negative third. */
            const double torque = -d1 * THRUST;
            const int    turn = 2;
            const double want_dw = torque / j_pair * turn * DT;
            double turn_fire[3] = { 0.0, 0.0, 1.0 };
            double q0[NB * 7], q1[NB * 7];
            const int32_t nquat = (int32_t)(NB * 7);

            ASSERT(s.attitudes(env, q0, (uint32_t)nquat) == nquat);
            for (int k = 0; k < turn; k++) {
                ASSERT(s.step(env, turn_fire) == K26RL_OK);
            }
            ASSERT(s.attitudes(env, q1, (uint32_t)nquat) == nquat);
            printf("  composite centre %.6f m from the lighter craft, "
                   "tensor %.6f kg m^2\n", d1, j_pair);
            near_("first pair turns about its composite centre",
                  q1[A1 * 7 + 6] - q0[A1 * 7 + 6], want_dw, 2e-3);
            /* The two answers this rules out: the centre taken at the
             * midpoint of the line rather than at the mass-weighted
             * point, which is a third of the way off; and the tensor
             * without its parallel-axis terms, which is four times
             * smaller. Both are orders outside the bound asserted. */
            ASSERT(fabs((q1[A1 * 7 + 6] - q0[A1 * 7 + 6]) -
                        (-0.5 * sep * THRUST / j_pair * turn * DT)) > 0.05);
            ASSERT(fabs((q1[A1 * 7 + 6] - q0[A1 * 7 + 6]) -
                        torque / (j1 + j2) * turn * DT) > 0.5);
            /* And the far craft turned with it. Its frame is turned
             * half a turn about its second axis to face the near one,
             * so its third axis is the world's reversed. */
            near_("and the far craft of that pair turns with it",
                  q1[A2 * 7 + 6] - q0[A2 * 7 + 6], -want_dw, 2e-3);
            /* The second pair, which fired nothing, did not turn. */
            for (int k = 0; k < 3; k++) {
                ASSERT(fabs(q1[B1 * 7 + 4 + k]) < 1e-9);
                ASSERT(fabs(q1[B2 * 7 + 4 + k]) < 1e-9);
            }
        }
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 5. the chain -------------------------------------------- *
     *
     * A drone grasps a rock, then the laden drone docks at a host. The
     * composite of the composite answers a thrust with three masses
     * summed, which is what says the second join carried the first
     * one's members with it.
     *
     * What this arm does not cover is stated so it is not read into
     * it. The cargo's own collisions, and what a chain answers to a
     * strike, are the arm on carried cargo below. The two ways a
     * capture forms no join are the two arms after that one. The
     * composite tensor under a thrust of its own is the docking
     * suite's arm, and the composite centre is measured in the arm
     * above this one.
     */
    printf("a laden re-dock, and what the chain still collides with\n");
    {
        const double mHOST = 4000.0, mDRONE = 1000.0, mROCK = 400.0;
        const double THRUST = 400.0, DT = 0.5;
        const int BURN = 4;
        enum { ANCHOR = 0, HOST = 1, DRONE = 2, ROCK = 3, NB = 4 };
        const int32_t nvals = (int32_t)(NB * 6);

        write_craft_(WORK_DIR "/host.k26asm", "host", mHOST, 0.5, 0.4, 0.4,
                     "berth", 0.9, "grasp_s");
        write_craft2_(WORK_DIR "/drone.k26asm", "drone", mDRONE,
                      0.5, 0.4, 0.4,
                      "grasp", 1.2, "grasp_s", "dock", -1.2, "grasp_s",
                      THRUST, 0.0);
        write_craft_(WORK_DIR "/rock.k26asm", "rock", mROCK, 0.4, 0.4, 0.4,
                     "face", 0.9, "grasp_s");

        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_CHAIN\n"
            "fn world w\n"
            GRASP_BLOCK
            ANCHOR_BODY
            "    astro_body host assembly=\"%s/host.k26asm\"" DRIFT
            " pos_x=-6.0 quat_w=1.0\n"
            "    astro_body drone assembly=\"%s/drone.k26asm\"" DRIFT
            " pos_x=0.0 quat_w=1.0\n"
            "    astro_body rock assembly=\"%s/rock.k26asm\"" DRIFT
            " pos_x=4.0 vel_x=-0.3 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt %.17g\n"
            "        substeps 5\n"
            "        horizon 600\n"
            "        terminated when episode.steps > 500\n"
            "        contact bounce restitution 0.9 friction 0.1\n"
            "    end\n"
            "    action pd box 0.0 1.0 default 0.0\n"
            "    on_step\n"
            "        drone.main.throttle = pd\n"
            "    end\n"
            "    observe port grasp of drone against face of rock full "
            "as gr\n"
            "    observe port dock of drone against berth of host full "
            "as gh\n"
            "    objective\n"
            "        reward gr_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR, DT);
        compile_("chain", prog);
        RlSurface s;
        void *so = open_("chain", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(9u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int jr = find_channel_(blob, (uint32_t)len, "gr_joined");
        int jh = find_channel_(blob, (uint32_t)len, "gh_joined");
        ASSERT(jr >= 0 && jh >= 0);

        double obs[64], coast[1] = { 0.0 };
        int step_r = -1, step_h = -1;
        for (int k = 0; k < 400; k++) {
            ASSERT(s.step(env, coast) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            if (obs[jr] != 0.0 && step_r < 0) step_r = k;
            if (obs[jh] != 0.0 && step_h < 0) step_h = k;
            if (step_r >= 0 && step_h >= 0) break;
        }
        printf("  grasp joined at step %d, re-dock at step %d\n",
               step_r, step_h);
        ASSERT(step_r >= 0 && step_h > step_r);
        ASSERT(obs[jr] == 1.0 && obs[jh] == 1.0);

        double s0[NB * 6], s1[NB * 6];
        double fire[1] = { 1.0 };
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s0, (uint32_t)nvals)
               == nvals);
        for (int k = 0; k < BURN; k++) {
            ASSERT(s.step(env, fire) == K26RL_OK);
        }
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s1, (uint32_t)nvals)
               == nvals);
        {
            double M = mHOST + mDRONE + mROCK;
            double v0 = (mHOST * s0[HOST * 6 + 3] +
                         mDRONE * s0[DRONE * 6 + 3] +
                         mROCK * s0[ROCK * 6 + 3]) / M;
            double v1 = (mHOST * s1[HOST * 6 + 3] +
                         mDRONE * s1[DRONE * 6 + 3] +
                         mROCK * s1[ROCK * 6 + 3]) / M;
            near_("three joined craft answer three masses summed",
                  v1 - v0, THRUST * BURN * DT / M, 1e-6);
            /* The answers this rules out are the pair without the
             * rock and the drone alone, the nearer of them a
             * fourteenth away and the bound five orders below it. */
            ASSERT(fabs((v1 - v0) - THRUST * BURN * DT / (mHOST + mDRONE))
                   > 0.01);
            ASSERT(fabs((v1 - v0) - THRUST * BURN * DT / mDRONE) > 0.5);
            /* All three moved as one. */
            ASSERT(fabs((s1[HOST * 6 + 3] - s0[HOST * 6 + 3]) -
                        (s1[DRONE * 6 + 3] - s0[DRONE * 6 + 3])) < 1e-9);
            ASSERT(fabs((s1[DRONE * 6 + 3] - s0[DRONE * 6 + 3]) -
                        (s1[ROCK * 6 + 3] - s0[ROCK * 6 + 3])) < 1e-9);
        }
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 6. carried cargo is still there, and it answers as the
     *         whole it belongs to ---------------------------------- *
     *
     * A drone grasps a rock, and a plain block then strikes the rock
     * across the line the two are joined on. Two things are measured
     * and each rules out a different defect.
     *
     * The contact is reported at all, on the cargo's own channel. A
     * runtime that took a follower's colliders out of the collision
     * pass reports nothing here and the block passes through the
     * cargo unrecorded.
     *
     * And the chain answers as the chain. The linear half gives the
     * impulse, which is the chain's mass times the change in its
     * centre-of-gravity velocity, and the angular half is then that
     * impulse on the lever arm from the CHAIN's centre against the
     * CHAIN's inertia about it. Both figures are hand-computed from
     * the fixture's own declarations. A resolution given the struck
     * craft's own centre answers with almost no turn at all, since
     * the block strikes the rock nearly through its middle; one given
     * the rock's own tensor answers with thirty-four times the turn.
     *
     * The fixture is arranged so the arithmetic is closed. The two
     * craft carry equal and opposite momentum, so the chain is at
     * rest once it forms and the block can be aimed at a fixed place.
     * The block is narrow along the line of centres, so the lever arm
     * of the contact is known to a fiftieth of a metre whatever point
     * on the overlap the kernel reports. And the block arrives square
     * on, with no relative motion across the contact, so no friction
     * impulse enters and the impulse is along the normal alone.
     */
    printf("carried cargo is struck, and the whole answers for it\n");
    {
        const double mDRONE = 1000.0, mROCK = 400.0, mBLOCK = 200.0;
        /* The frozen separation, from the two port arms, and where the
         * chain's centre of gravity sits on it. */
        const double SEP = 1.2 + 0.9;
        const double MT  = mDRONE + mROCK;
        const double dDRONE = mROCK * SEP / MT;
        const double dROCK  = mDRONE * SEP / MT;
        /* Each craft's own third inertia component about its own
         * centre, for a uniform solid box. */
        const double jDRONE = mDRONE / 3.0 * (0.5 * 0.5 + 0.4 * 0.4);
        const double jROCK  = mROCK  / 3.0 * (0.4 * 0.4 + 0.4 * 0.4);
        const double J_CHAIN = jDRONE + mDRONE * dDRONE * dDRONE
                             + jROCK  + mROCK  * dROCK  * dROCK;
        enum { ANCHOR = 0, DRONE = 1, ROCK = 2, BLOCK = 3, NB = 4 };
        const int32_t nvals = (int32_t)(NB * 6);
        const int32_t nquat = (int32_t)(NB * 7);

        write_block_(WORK_DIR "/block.k26asm", "block", mBLOCK,
                     0.02, 0.3, 0.3);
        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_CARGO\n"
            "fn world w\n"
            GRASP_BLOCK
            ANCHOR_BODY
            "    astro_body drone assembly=\"%s/drone.k26asm\"" DRIFT
            " pos_x=0.0 vel_x=0.04 quat_w=1.0\n"
            "    astro_body rock assembly=\"%s/rock.k26asm\"" DRIFT
            " pos_x=5.0 vel_x=-0.10 quat_w=0.0 quat_y=1.0\n"
            "    astro_body block assembly=\"%s/block.k26asm\"" DRIFT
            " pos_x=2.93 pos_y=40.0 vel_y=-1.0 quat_w=1.0\n"
            "    episode\n"
            "        control_dt %.17g\n"
            "        substeps 5\n"
            "        horizon 400\n"
            "        terminated when episode.steps > 300\n"
            "        contact bounce restitution 0.9 friction 0.1\n"
            "    end\n"
            "    action pd box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone against face of rock full "
            "as gr\n"
            "    observe contact of rock as cr\n"
            "    objective\n"
            "        reward gr_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR, 0.5);
        compile_("cargo", prog);
        RlSurface s;
        void *so = open_("cargo", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(13u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int jr = find_channel_(blob, (uint32_t)len, "gr_joined");
        int hr = find_channel_(blob, (uint32_t)len, "cr_hit");
        ASSERT(jr >= 0 && hr >= 0);
        double obs[64], act[1] = { 0.0 };
        double s0[NB * 6], s1[NB * 6], q0[NB * 7], q1[NB * 7];
        double com_x_at_join = 0.0, rock_x_at_join = 0.0;
        double sep_at_join = 0.0;
        int joined_at = -1, cargo_hit = -1;
        for (int k = 0; k < 300; k++) {
            ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s0,
                            (uint32_t)nvals) == nvals);
            ASSERT(s.attitudes(env, q0, (uint32_t)nquat) == nquat);
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            if (obs[jr] != 0.0 && joined_at < 0) {
                joined_at = k;
                ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s1,
                                (uint32_t)nvals) == nvals);
                com_x_at_join = (mDRONE * s1[DRONE * 6] +
                                 mROCK * s1[ROCK * 6]) / MT;
                rock_x_at_join = s1[ROCK * 6];
                sep_at_join = s1[ROCK * 6] - s1[DRONE * 6];
            }
            if (joined_at >= 0 && k > joined_at && obs[hr] != 0.0) {
                cargo_hit = k;
                break;
            }
        }
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, s1,
                        (uint32_t)nvals) == nvals);
        ASSERT(s.attitudes(env, q1, (uint32_t)nquat) == nquat);
        printf("  joined at step %d, the cargo was struck at step %d\n",
               joined_at, cargo_hit);
        ASSERT(joined_at >= 0);
        ASSERT(cargo_hit > joined_at);
        /* The join is still whole: the contact was with a body
         * outside the chain, and nothing about it undid the grasp. */
        ASSERT(obs[jr] == 1.0);
        /* The separation the join froze, read at the step it formed:
         * the two mating planes meet, so it is the sum of the two
         * port arms. It is read there and not at the end, because by
         * then the strike has turned the pair and a separation taken
         * along one axis would be reading the cosine of that turn. */
        near_("the frozen separation is the two port arms",
              sep_at_join, SEP, 2e-3);
        /* And the strike did not stretch the pair: the distance
         * between the two craft is what the join froze, whatever the
         * turn has done to its direction. */
        {
            double dx = s1[ROCK * 6]     - s1[DRONE * 6];
            double dy = s1[ROCK * 6 + 1] - s1[DRONE * 6 + 1];
            double dz = s1[ROCK * 6 + 2] - s1[DRONE * 6 + 2];
            near_("and the pair is still that far apart after the strike",
                  sqrt(dx * dx + dy * dy + dz * dz), sep_at_join, 1e-9);
        }
        /* The chain was at rest before the strike, by construction. */
        {
            double v_before = (mDRONE * s0[DRONE * 6 + 4] +
                               mROCK * s0[ROCK * 6 + 4]) / MT;
            double v_after  = (mDRONE * s1[DRONE * 6 + 4] +
                               mROCK * s1[ROCK * 6 + 4]) / MT;
            double impulse  = MT * (v_before - v_after);
            /* The lever arm the strike had about the chain's centre,
             * from the fixture's own geometry: the block is aimed at
             * the rock's centre and is narrow enough that whatever
             * point on the overlap the kernel reports lies within two
             * hundredths of a metre of it. */
            double lever = rock_x_at_join - com_x_at_join;
            double dw    = q1[DRONE * 7 + 6] - q0[DRONE * 7 + 6];
            double want  = -lever * impulse / J_CHAIN;
            printf("  chain centre %.6f m from the rock, tensor %.6f "
                   "kg m^2, impulse %.6f N s\n", lever, J_CHAIN, impulse);
            near_("the chain's lever arm is the parallel-axis one",
                  lever, dROCK, 0.02);
            ASSERT(impulse > 100.0);
            near_("the chain turns at its own composite tensor's rate",
                  dw, want, 0.02 * fabs(want) + 1e-4);
            /* The rock's own tensor would give thirty-four times the
             * turn, and the rock's own centre almost none of it. */
            ASSERT(fabs(dw - (-lever * impulse / jROCK)) > 0.5);
            ASSERT(fabs(dw) > 0.02);
            /* And the cargo turned with its carrier: the frame of the
             * rock is turned half a turn about its second axis, so
             * its third axis is the world's reversed. */
            near_("and the cargo turns with its carrier",
                  q1[ROCK * 7 + 6] - q0[ROCK * 7 + 6], -dw, 1e-9);
        }
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 7. a capture that can form no join is still resolved ---- *
     *
     * Two chains, each a heavy craft leading a light one. The two
     * light craft then meet nose to nose, inside every limit of the
     * envelope, and no join can form: a body follows at most one
     * leader and both of these already do. The contact is therefore an
     * ordinary one and is resolved as one.
     *
     * This is the arm that a suppressed resolution fails. Suppressing
     * it leaves the two chains closing at the rate they arrived with
     * and passing through each other, which is worse than either
     * answer the environment declares.
     *
     * The other reachable way a capture forms no join is a port
     * already in one, and that is the arm below. The remaining way is
     * not reachable by a trajectory and is stated here rather than
     * gated: a contact between two members of one chain never
     * happens, because such pairs leave the collision pass, and a
     * cycle can only be closed by such a contact. What is left of
     * that rule is the guard in the runtime.
     */
    printf("a capture that can form no join is resolved and not passed "
           "through\n");
    {
        const double mHOST = 4000.0, mDRONE = 1000.0;
        enum { ANCHOR = 0, HA = 1, DA = 2, DB = 3, HB = 4, NB = 5 };
        const int32_t nvals = (int32_t)(NB * 6);

        write_craft_(WORK_DIR "/gantry.k26asm", "gantry", mHOST,
                     0.5, 0.4, 0.4, "berth", 0.9, "grasp_s");
        write_craft2_(WORK_DIR "/tug.k26asm", "tug", mDRONE,
                      0.5, 0.4, 0.4,
                      "fore", 1.2, "grasp_s", "aft", -1.2, "grasp_s",
                      0.0, 0.0);

        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_NOJOIN\n"
            "fn world w\n"
            GRASP_BLOCK
            ANCHOR_BODY
            "    astro_body ha assembly=\"%s/gantry.k26asm\"" DRIFT
            " pos_x=-20.0 vel_x=0.3 quat_w=1.0\n"
            "    astro_body da assembly=\"%s/tug.k26asm\"" DRIFT
            " pos_x=-5.0 quat_w=1.0\n"
            "    astro_body db assembly=\"%s/tug.k26asm\"" DRIFT
            " pos_x=5.0 quat_w=0.0 quat_y=1.0\n"
            "    astro_body hb assembly=\"%s/gantry.k26asm\"" DRIFT
            " pos_x=20.0 vel_x=-0.3 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        substeps 5\n"
            "        horizon 400\n"
            "        terminated when episode.steps > 300\n"
            "    end\n"
            "    action pd box -1.0 1.0 default 0.0\n"
            "    observe port aft of da against berth of ha full as ja\n"
            "    observe port aft of db against berth of hb full as jb\n"
            "    observe port fore of da against fore of db full as jm\n"
            "    objective\n"
            "        reward jm_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR, WORK_DIR);
        compile_("nojoin", prog);
        RlSurface s;
        void *so = open_("nojoin", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(29u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int ja = find_channel_(blob, (uint32_t)len, "ja_joined");
        int jb = find_channel_(blob, (uint32_t)len, "jb_joined");
        int jm = find_channel_(blob, (uint32_t)len, "jm_joined");
        int cm = find_channel_(blob, (uint32_t)len, "jm_captured");
        int am = find_channel_(blob, (uint32_t)len, "jm_axial");
        ASSERT(ja >= 0 && jb >= 0 && jm >= 0 && cm >= 0 && am >= 0);

        double obs[64], act[1] = { 0.0 };
        double before[NB * 6], after[NB * 6];
        int both_joined = -1, met = -1;
        double closing_before = 0.0, axial_at_meeting = 0.0;
        for (int k = 0; k < 300; k++) {
            ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, before,
                            (uint32_t)nvals) == nvals);
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            if (both_joined < 0 && obs[ja] != 0.0 && obs[jb] != 0.0) {
                both_joined = k;
            }
            if (both_joined >= 0 && obs[cm] != 0.0) {
                met = k;
                closing_before = before[DA * 6 + 3] - before[DB * 6 + 3];
                axial_at_meeting = obs[am];
                break;
            }
        }
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, after,
                        (uint32_t)nvals) == nvals);
        printf("  both chains joined by step %d, the two met at step %d\n",
               both_joined, met);
        ASSERT(both_joined >= 0);
        ASSERT(met > both_joined);
        /* The envelope was satisfied: the pulse is the verdict on the
         * contact and it does not depend on whether a join followed. */
        printf("  the meeting satisfied the envelope (axial %+.6f)\n",
               axial_at_meeting);
        /* The two existing joins stand and no third one formed. */
        ASSERT(obs[ja] == 1.0 && obs[jb] == 1.0);
        ASSERT(obs[jm] == 0.0);
        /* And the contact was resolved. The two were closing at nearly
         * half a metre a second; the arrest leaves them with that
         * gone. A suppressed resolution leaves the number where it
         * was. */
        {
            double closing_after = after[DA * 6 + 3] - after[DB * 6 + 3];
            printf("  the two closed at %+.6f and leave at %+.6f\n",
                   closing_before, closing_after);
            ASSERT(closing_before > 0.4);
            ASSERT(fabs(closing_after) < 0.05);
        }
        /* And they do not go on through each other: the gap between
         * the two mating planes never turns negative over the rest of
         * the run. */
        {
            double worst = 1.0e9;
            for (int k = 0; k < 60; k++) {
                ASSERT(s.step(env, act) == K26RL_OK);
                ASSERT(s.obs(env, obs) == K26RL_OK);
                if (obs[am] < worst) worst = obs[am];
            }
            printf("  the closest the two planes come afterwards is "
                   "%+.6f m\n", worst);
            ASSERT(worst > -0.01);
            ASSERT(obs[jm] == 0.0);
        }
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 7b. a capture at a port already in a join ---------------- *
     *
     * Two craft mate. A third then reaches the very interface they
     * mated at, inside every limit of its own envelope, and the port
     * it arrives at is occupied. No second join forms there and the
     * contact is resolved as an ordinary one.
     *
     * Reaching a mated interface takes a geometry, and this one is
     * built rather than found. A mated pair fills its own interface
     * along the approach line, so a third craft has to arrive beside
     * it: the two mated craft are slender and their mating rings are
     * wide, and the arriving craft comes down a line offset far enough
     * across to clear both hulls while its own ring still overlaps the
     * leader's. The follower's ring is the small one, so the arriving
     * craft passes it without touching and meets the leader's, which
     * is the occupied port this arm is about.
     *
     * The envelope is permissive because the arriving craft is
     * measured against the leader's ring from well off its centre
     * line, and a limit that refused that would make the contact an
     * ordinary impact and this arm vacuous. The limits are the
     * program's own statement, which is what a declared envelope is
     * for.
     *
     * The mated pair's two ports name envelopes of different
     * diameters, so no statement pairs them and the join between them
     * is read from behaviour instead: their separation is frozen and
     * their velocities are one. That is what a join is, and a runtime
     * that had not formed one would fail it.
     */
    printf("a capture at a port already in a join forms no join\n");
    {
        const double mLEAD = 1000.0, mFOLLOW = 3000.0, mTHIRD = 500.0;
        enum { ANCHOR = 0, LEAD = 1, FOLLOW = 2, THIRD = 3, NB = 4 };
        const int32_t nvals = (int32_t)(NB * 6);

        /* A slender hull with a wide ring, and the ring standing well
         * proud of it. */
        write_craft_(WORK_DIR "/wlead.k26asm", "wlead", mLEAD,
                     0.5, 0.15, 0.15, "ring", 1.2, "wide_s");
        write_craft_(WORK_DIR "/wfollow.k26asm", "wfollow", mFOLLOW,
                     0.5, 0.15, 0.15, "peg", 1.0, "tiny_s");
        write_craft_(WORK_DIR "/wthird.k26asm", "wthird", mTHIRD,
                     0.5, 0.15, 0.15, "ring", 1.2, "wide_s");

        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_OCCUPIED\n"
            "fn world w\n"
            "    capture_envelope wide_s\n"
            "        axial_rate 0.00 5.00\n"
            "        lateral_rate 5.00\n"
            "        pitchyaw_rate 300.0\n"
            "        roll_rate 300.0\n"
            "        lateral 5.00\n"
            "        pitchyaw 20.0\n"
            "        roll 180.0\n"
            "        diameter 2000.0\n"
            "    end\n"
            "    capture_envelope tiny_s\n"
            "        axial_rate 0.00 5.00\n"
            "        lateral_rate 5.00\n"
            "        pitchyaw_rate 300.0\n"
            "        roll_rate 300.0\n"
            "        lateral 5.00\n"
            "        pitchyaw 20.0\n"
            "        roll 180.0\n"
            "        diameter 200.0\n"
            "    end\n"
            ANCHOR_BODY
            "    astro_body lead assembly=\"%s/wlead.k26asm\"" DRIFT
            " pos_x=0.0 pos_y=0.0 quat_w=1.0\n"
            "    astro_body follow assembly=\"%s/wfollow.k26asm\"" DRIFT
            " pos_x=6.0 pos_y=0.0 vel_x=-0.3 quat_w=0.0 quat_y=1.0\n"
            "    astro_body third assembly=\"%s/wthird.k26asm\"" DRIFT
            " pos_x=15.0 pos_y=1.5 vel_x=-0.5 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        substeps 5\n"
            "        horizon 500\n"
            "        terminated when episode.steps > 400\n"
            "    end\n"
            "    action pd box -1.0 1.0 default 0.0\n"
            "    observe port ring of third against ring of lead full "
            "as gt\n"
            "    observe contact of third as ct\n"
            "    objective\n"
            "        reward gt_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, WORK_DIR);
        compile_("occupied", prog);
        RlSurface s;
        void *so = open_("occupied", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(31u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int gc = find_channel_(blob, (uint32_t)len, "gt_captured");
        int gj = find_channel_(blob, (uint32_t)len, "gt_joined");
        int ga = find_channel_(blob, (uint32_t)len, "gt_axial");
        int ch = find_channel_(blob, (uint32_t)len, "ct_hit");
        ASSERT(gc >= 0 && gj >= 0 && ga >= 0 && ch >= 0);

        double obs[64], act[1] = { 0.0 };
        double st[NB * 6], before[NB * 6], after[NB * 6];
        double mated_sep = 0.0, mated_at = -1.0;
        int mated_step = -1, met = -1;
        for (int k = 0; k < 400; k++) {
            ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, before,
                            (uint32_t)nvals) == nvals);
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, st,
                            (uint32_t)nvals) == nvals);
            /* The mated pair is read from behaviour: the step on which
             * the two stop closing and start moving as one. */
            if (mated_step < 0 &&
                fabs(st[FOLLOW * 6 + 3] - st[LEAD * 6 + 3]) < 1e-12 &&
                fabs(st[LEAD * 6 + 3]) > 1e-9) {
                mated_step = k;
                mated_sep  = st[FOLLOW * 6] - st[LEAD * 6];
            }
            if (mated_step >= 0 && obs[ch] != 0.0) { met = k; break; }
        }
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, after,
                        (uint32_t)nvals) == nvals);
        printf("  the pair mated at step %d, the third craft arrived at "
               "step %d\n", mated_step, met);
        ASSERT(mated_step >= 0);
        ASSERT(met > mated_step);
        /* The mated pair is a join: its separation is frozen and its
         * two velocities are one, before the third craft arrives and
         * after it. */
        mated_at = after[FOLLOW * 6] - after[LEAD * 6];
        printf("  its separation was %+.9f at mating and %+.9f after\n",
               mated_sep, mated_at);
        ASSERT(fabs(mated_at - mated_sep) < 1e-6);
        ASSERT(fabs(after[FOLLOW * 6 + 3] - after[LEAD * 6 + 3]) < 1e-9);
        /* The arriving craft met the envelope: the pulse is the
         * verdict on the contact and does not depend on a join. */
        printf("  the arrival satisfied the envelope (captured %.0f, "
               "axial %+.6f)\n", obs[gc], obs[ga]);
        ASSERT(obs[gc] != 0.0);
        /* And no join formed at the occupied port. */
        ASSERT(obs[gj] == 0.0);
        /* The contact was resolved rather than suppressed: the
         * arriving craft was closing and is not any more. */
        printf("  it closed at %+.6f and leaves at %+.6f\n",
               before[LEAD * 6 + 3] - before[THIRD * 6 + 3],
               after[LEAD * 6 + 3] - after[THIRD * 6 + 3]);
        ASSERT(before[LEAD * 6 + 3] - before[THIRD * 6 + 3] > 0.2);
        ASSERT(after[LEAD * 6 + 3] - after[THIRD * 6 + 3] <
               (before[LEAD * 6 + 3] - before[THIRD * 6 + 3]) - 0.2);
        /* And it does not go on through: the two rings never come
         * closer than they are at the contact. */
        {
            double worst = 1.0e9;
            for (int k = 0; k < 60; k++) {
                ASSERT(s.step(env, act) == K26RL_OK);
                ASSERT(s.obs(env, obs) == K26RL_OK);
                if (obs[ga] < worst) worst = obs[ga];
            }
            printf("  the closest the two rings come afterwards is "
                   "%+.6f m\n", worst);
            ASSERT(worst > -0.05);
            ASSERT(obs[gj] == 0.0);
        }
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 8. the combined rate the `full` mark publishes ------------ *
     *
     * The ninth binding condition is the lateral rate carried along
     * the arm from the interface to the arriving craft's centre of
     * mass. The passive craft here turns about an axis offset from the
     * line of centres, so that rate is not the difference of the two
     * craft's own velocities and a form publishing that difference
     * disagrees.
     *
     * The expected value is computed here from the body-state and
     * attitude getters and the fixture's own port geometry, in code
     * that shares nothing with the library that computes the channel.
     *
     * What that reaches, and what it does not, since a re-implemented
     * formula is only as good as the formula. It catches a channel
     * wired to the wrong index, a sign taken the wrong way, an arm
     * measured from the wrong body, and a component dropped: each
     * would move the number by far more than the bound. It does not
     * catch the two implementations agreeing on a wrong definition of
     * the rate itself, because it is the same definition written
     * twice. What holds that end is the document the definition comes
     * from, and the arm beside it showing the channel differs from
     * the interface rate it would be confused with.
     */
    printf("the combined rate against an independent computation\n");
    {
        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_VCG\n"
            "fn world w\n"
            GRASP_BLOCK
            ANCHOR_BODY
            "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT
            " pos_x=0.0 quat_w=1.0 omega_z=0.004\n"
            "    astro_body rock assembly=\"%s/d2.k26asm\"" DRIFT
            " pos_x=6.0 pos_y=0.05 vel_x=-0.02"
            " quat_w=0.0 quat_y=1.0 omega_x=0.001 omega_y=0.002"
            " omega_z=0.003\n"
            "    episode\n"
            "        control_dt %.17g\n"
            "        substeps 5\n"
            "        horizon 40\n"
            "    end\n"
            "    action pd box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone against face of rock full "
            "as gr\n"
            "    objective\n"
            "        reward gr_axial\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR, 0.5);
        compile_("vcg", prog);
        RlSurface s;
        void *so = open_("vcg", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(19u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        int vcg = find_channel_(blob, (uint32_t)len, "gr_v_cg");
        int vlat = find_channel_(blob, (uint32_t)len, "gr_v_lateral");
        ASSERT(vcg >= 0 && vlat >= 0);
        enum { DR = 1, RK = 2, NB = 3 };
        double obs[64], act[1] = { 0.0 };
        double st[NB * 6], qs[NB * 7];
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, obs) == K26RL_OK);
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, st, NB * 6) == NB * 6);
        ASSERT(s.attitudes(env, qs, NB * 7) == NB * 7);

        /* The independent computation. Both craft's ports sit on their
         * own first body axis at the distances this fixture declared,
         * and each craft's centre of mass is at its own origin, since
         * each is one box centred there. */
        {
            const double AP = 0.6, PP = 0.9;   /* the two port arms */
            double qa[4] = { qs[DR * 7], qs[DR * 7 + 1], qs[DR * 7 + 2],
                             qs[DR * 7 + 3] };
            double qp[4] = { qs[RK * 7], qs[RK * 7 + 1], qs[RK * 7 + 2],
                             qs[RK * 7 + 3] };
            double wa_b[3] = { qs[DR * 7 + 4], qs[DR * 7 + 5],
                               qs[DR * 7 + 6] };
            double wp_b[3] = { qs[RK * 7 + 4], qs[RK * 7 + 5],
                               qs[RK * 7 + 6] };
            /* Rotate a body-frame vector into the world by the
             * quaternion, written out here rather than called. */
#define ROT_(q, v, o) do { \
    double _w = (q)[0], _x = (q)[1], _y = (q)[2], _z = (q)[3]; \
    double _t0 = 2.0 * (_y * (v)[2] - _z * (v)[1]); \
    double _t1 = 2.0 * (_z * (v)[0] - _x * (v)[2]); \
    double _t2 = 2.0 * (_x * (v)[1] - _y * (v)[0]); \
    (o)[0] = (v)[0] + _w * _t0 + (_y * _t2 - _z * _t1); \
    (o)[1] = (v)[1] + _w * _t1 + (_z * _t0 - _x * _t2); \
    (o)[2] = (v)[2] + _w * _t2 + (_x * _t1 - _y * _t0); \
} while (0)
            double abody[3] = { AP, 0.0, 0.0 }, pbody[3] = { PP, 0.0, 0.0 };
            double aarm[3], parm[3], wa[3], wp[3], axis_b[3] = { 1, 0, 0 };
            double paxis[3];
            ROT_(qa, abody, aarm);
            ROT_(qp, pbody, parm);
            ROT_(qa, wa_b, wa);
            ROT_(qp, wp_b, wp);
            ROT_(qp, axis_b, paxis);
            double apos[3], ppos[3];
            for (int k = 0; k < 3; k++) {
                apos[k] = st[DR * 6 + k] + aarm[k];
                ppos[k] = st[RK * 6 + k] + parm[k];
            }
            /* Each port's material velocity, then their difference. */
            double avel[3], pvel[3], vrel[3], wrel[3];
            double ra[3], rp[3];
            for (int k = 0; k < 3; k++) {
                ra[k] = apos[k] - st[DR * 6 + k];
                rp[k] = ppos[k] - st[RK * 6 + k];
                wrel[k] = wa[k] - wp[k];
            }
            avel[0] = st[DR * 6 + 3] + (wa[1] * ra[2] - wa[2] * ra[1]);
            avel[1] = st[DR * 6 + 4] + (wa[2] * ra[0] - wa[0] * ra[2]);
            avel[2] = st[DR * 6 + 5] + (wa[0] * ra[1] - wa[1] * ra[0]);
            pvel[0] = st[RK * 6 + 3] + (wp[1] * rp[2] - wp[2] * rp[1]);
            pvel[1] = st[RK * 6 + 4] + (wp[2] * rp[0] - wp[0] * rp[2]);
            pvel[2] = st[RK * 6 + 5] + (wp[0] * rp[1] - wp[1] * rp[0]);
            for (int k = 0; k < 3; k++) vrel[k] = avel[k] - pvel[k];
            /* The arm from the interface to the arriving craft's
             * centre of mass, and the rate that combination puts
             * there. */
            double arm[3] = { st[DR * 6] - apos[0], st[DR * 6 + 1] - apos[1],
                              st[DR * 6 + 2] - apos[2] };
            double cv[3];
            cv[0] = vrel[0] + (wrel[1] * arm[2] - wrel[2] * arm[1]);
            cv[1] = vrel[1] + (wrel[2] * arm[0] - wrel[0] * arm[2]);
            cv[2] = vrel[2] + (wrel[0] * arm[1] - wrel[1] * arm[0]);
            double dot = cv[0] * paxis[0] + cv[1] * paxis[1] +
                         cv[2] * paxis[2];
            double lat[3];
            for (int k = 0; k < 3; k++) lat[k] = cv[k] - dot * paxis[k];
            double want = sqrt(lat[0] * lat[0] + lat[1] * lat[1] +
                               lat[2] * lat[2]);
#undef ROT_
            near_("the combined rate at the arriving centre of mass",
                  obs[vcg], want, 1e-9);
            /* And it is not the rate at the interface, which is the
             * channel beside it: a form publishing that one instead
             * fails here. */
            printf("  the rate at the interface itself reads %+.9f\n",
                   obs[vlat]);
            ASSERT(fabs(obs[vcg] - obs[vlat]) > 1e-4);
        }
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 9. the unmarked form is nine channels ------------------- *
     *
     * The same statement without the mark publishes nine channels and
     * the mark adds two after them, so a program that adds the mark
     * keeps every index it had.
     */
    printf("the mark adds two channels and moves none\n");
    {
        static const char *const NINE[9] = {
            "_captured", "_axial", "_lateral", "_pitchyaw", "_roll",
            "_v_axial", "_v_lateral", "_v_pitchyaw", "_v_roll"
        };
        int base[2][9], extra[2];
        for (int marked = 0; marked < 2; marked++) {
            char prog[8192];
            snprintf(prog, sizeof prog,
                "form RL_MARK%d\n"
                "fn world w\n"
                GRASP_BLOCK
                ANCHOR_BODY
                "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT
                " pos_x=0.0 quat_w=1.0\n"
                "    astro_body rock assembly=\"%s/d2.k26asm\"" DRIFT
                " pos_x=6.0 quat_w=0.0 quat_y=1.0\n"
                "    episode\n"
                "        control_dt 0.5\n"
                "        horizon 20\n"
                "    end\n"
                "    action pd box -1.0 1.0 default 0.0\n"
                "    observe port grasp of drone against face of rock%s "
                "as gr\n"
                "    observe contact of drone as cd\n"
                "    objective\n"
                "        reward gr_axial\n"
                "    end\n"
                "end\n"
                "end\n", marked, WORK_DIR, WORK_DIR,
                marked ? " full" : "");
            char stem[32];
            snprintf(stem, sizeof stem, "mark%d", marked);
            compile_(stem, prog);
            RlSurface s;
            void *so = open_(stem, &s);
            K26RlEnv *env = NULL;
            ASSERT(s.create(23u, 1u, &env) == K26RL_OK);
            uint8_t blob[16384];
            int32_t len = s.spec(env, blob, sizeof blob);
            ASSERT(len > 0);
            for (int k = 0; k < 9; k++) {
                char name[64];
                snprintf(name, sizeof name, "gr%s", NINE[k]);
                base[marked][k] = find_channel_(blob, (uint32_t)len, name);
                ASSERT(base[marked][k] >= 0);
            }
            extra[marked] = find_channel_(blob, (uint32_t)len, "gr_v_cg");
            /* The contact channel after it moves by two with the mark
             * and by nothing without it, which is what says the two
             * new channels landed after the nine and not among them. */
            int cd = find_channel_(blob, (uint32_t)len, "cd_hit");
            ASSERT(cd >= 0);
            printf("  %s: nine channels at %d..%d, `cd_hit` at %d\n",
                   marked ? "marked  " : "unmarked",
                   base[marked][0], base[marked][8], cd);
            if (!marked) ASSERT(cd == base[marked][8] + 1);
            else         ASSERT(cd == base[marked][8] + 3);
            s.destroy(env);
            dlclose(so);
        }
        for (int k = 0; k < 9; k++) ASSERT(base[0][k] == base[1][k]);
        ASSERT(extra[0] < 0);
        ASSERT(extra[1] == base[1][8] + 1);
        printf("  the nine sit at the same indices either way: OK\n");
    }
    n_pass++;

    /* ---- 9a. the two the mark adds are readable where they matter - *
     *
     * Publishing a channel and admitting it into an expression are two
     * different acts, and this arm exists because they had come apart:
     * the emitter published `_v_cg` and `_joined` and the spec named
     * both, while the pass that decides which names an expression may
     * read carried the nine-channel list for every port observe, mark
     * or no mark. Every program reading either name in a reward, a
     * terminal or a termination predicate was refused as naming
     * something the world does not publish, which is the one thing
     * those two channels exist for.
     *
     * What makes this arm able to fail: the fixture reads both names in
     * all three positions and asserts the compile succeeds. Against the
     * defect it named, the compile is refused and the arm stops here.
     * An arm that only compiled a program declaring the mark would have
     * passed throughout.
     */
    printf("the mark's two channels are readable in an expression\n");
    {
        char prog[8192];
        snprintf(prog, sizeof prog,
            "form RL_MARKREAD\n"
            "fn world w\n"
            GRASP_BLOCK
            ANCHOR_BODY
            "    astro_body drone assembly=\"%s/d1.k26asm\"" DRIFT
            " pos_x=0.0 quat_w=1.0\n"
            "    astro_body rock assembly=\"%s/d2.k26asm\"" DRIFT
            " pos_x=6.0 quat_w=0.0 quat_y=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 20\n"
            "        terminated when gr_joined > 0.5\n"
            "    end\n"
            "    action pd box -1.0 1.0 default 0.0\n"
            "    observe port grasp of drone against face of rock full "
            "as gr\n"
            "    objective\n"
            "        reward 0.0 - gr_v_cg\n"
            "        terminal 10.0 * gr_joined\n"
            "    end\n"
            "end\n"
            "end\n", WORK_DIR, WORK_DIR);
        compile_("markread", prog);
        RlSurface s;
        void *so = open_("markread", &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(29u, 1u, &env) == K26RL_OK);
        uint8_t blob[16384];
        int32_t len = s.spec(env, blob, sizeof blob);
        ASSERT(len > 0);
        ASSERT(find_channel_(blob, (uint32_t)len, "gr_v_cg") >= 0);
        ASSERT(find_channel_(blob, (uint32_t)len, "gr_joined") >= 0);
        printf("  a reward, a terminal and a termination predicate over "
               "`gr_v_cg` and `gr_joined` compile: OK\n");
        s.destroy(env);
        dlclose(so);
    }
    n_pass++;

    /* ---- 10. the shipped programs, against the prior binary ------- *
     *
     * The compiler from the commit before this work is built here and
     * both binaries compile the same two programs. The episode records
     * are compared byte for byte at one seed and one action stream. A
     * comparison of today's binary with itself would agree with any
     * change; this one cannot.
     */
    printf("the shipped programs against the compiler before this work\n");
    if (run_("git -C .. rev-parse --verify --quiet " PRIOR_COMMIT
             "^{commit} > /dev/null 2>&1") != 0) {
        /* A gate that cannot run is not a gate that passed. Without
         * the commit it compares against there is nothing to compare,
         * and reporting that as green would put a tick beside an
         * unmeasured claim. It fails, and the only way past it is an
         * explicit choice that says so in the output. */
        if (getenv("KFLC_CAPTURE_ALLOW_NO_PRIOR")) {
            printf("  NOT MEASURED: " PRIOR_COMMIT " is not in this "
                   "checkout's history, and KFLC_CAPTURE_ALLOW_NO_PRIOR "
                   "is set, so this gate is being stood down by "
                   "request. Nothing about compatibility has been "
                   "checked in this run.\n");
        } else {
            fprintf(stderr,
                "FAIL test_rl_capture: " PRIOR_COMMIT " is not in this "
                "checkout's history, so the compatibility gate has "
                "nothing to compare against and has measured nothing. "
                "Fetch the history, or set "
                "KFLC_CAPTURE_ALLOW_NO_PRIOR to stand the gate down "
                "deliberately.\n");
            exit(1);
        }
    } else {
        ASSERT(run_("rm -rf " WORK_DIR "/prior && mkdir -p "
                    WORK_DIR "/prior") == 0);
        ASSERT(run_("git -C .. archive " PRIOR_COMMIT " kflc | tar -x -C "
                    WORK_DIR "/prior") == 0);
        /* The extracted compiler builds against the libraries beside
         * it in the tree, so the sibling directories are linked in
         * beside the extraction. */
        ASSERT(run_("for d in ../libk26* ../common; do "
                    "ln -sfn \"$(cd $d && pwd)\" " WORK_DIR "/prior/; "
                    "done") == 0);
        ASSERT(run_("make -C " WORK_DIR "/prior/kflc bin/kflc > "
                    WORK_DIR "/prior/build.log 2>&1") == 0);
        printf("  built the compiler at " PRIOR_COMMIT "\n");

        /* The two shipped programs a port observe reaches: the docking
         * benchmark, and the two-statement fixture the docking gate
         * drives, written here as that gate writes it. */
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
        {
            double half = 3.14159265358979323846 / 2.0;
            double t = 1.5 * 3.14159265358979323846 / 180.0;
            double qy[4] = { cos(half), 0.0, sin(half), 0.0 };
            double qz[4] = { cos(t), 0.0, 0.0, sin(t) };
            double q[4];
            q[0] = qz[0]*qy[0] - qz[1]*qy[1] - qz[2]*qy[2] - qz[3]*qy[3];
            q[1] = qz[0]*qy[1] + qz[1]*qy[0] + qz[2]*qy[3] - qz[3]*qy[2];
            q[2] = qz[0]*qy[2] - qz[1]*qy[3] + qz[2]*qy[0] + qz[3]*qy[1];
            q[3] = qz[0]*qy[3] + qz[1]*qy[2] - qz[2]*qy[1] + qz[3]*qy[0];
            char prog[4096];
            snprintf(prog, sizeof prog,
                "form RL_PERSP\n"
                "fn world w\n"
                "    astro_body one assembly=\"%s/perspa.k26asm\"" DRIFT
                " pos_x=0.0 pos_y=0.0 pos_z=0.0 quat_w=1.0\n"
                "    astro_body two assembly=\"%s/perspb.k26asm\"" DRIFT
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
        }

        static const struct { const char *tag, *kfl, *args; } SHIPPED_[] = {
            { "the docking benchmark", "examples/docking_benchmark.kfl",
              "--envs 2 --episodes 3 --seed 11" },
            { "the two-statement fixture", WORK_DIR "/persp.kfl",
              "--envs 2 --episodes 3 --seed 11" }
        };
        for (size_t i = 0; i < sizeof SHIPPED_ / sizeof SHIPPED_[0]; i++) {
            char out_old[256], out_new[256], cmd[16384];
            snprintf(out_old, sizeof out_old, WORK_DIR "/old%zu", i);
            snprintf(out_new, sizeof out_new, WORK_DIR "/new%zu", i);
            /* rl_compile_ drives ./bin/kflc; the prior binary is
             * driven the same way, with the same flags, by reusing
             * the same environment through a shell that names it. */
            rl_compile_(SHIPPED_[i].kfl, out_new, WORK_DIR);
            {
                (void)run_("cp " WORK_DIR "/kflc.log "
                           WORK_DIR "/new.log");
            }
            /* The prior compiler, with the same include and archive
             * lists rl_compile_ builds. */
            {
                char cflags[4096], ldlibs[4096];
                int n = snprintf(cflags, sizeof cflags,
                    "-O2 -g -std=c++11 -Wno-format-truncation "
                    "-ffp-contract=off -fexcess-precision=standard");
                for (int k = 0; RL_INCLUDE_DIRS_[k]; k++) {
                    n += snprintf(cflags + n, sizeof cflags - (size_t)n,
                                  " -I%s", RL_INCLUDE_DIRS_[k]);
                }
                ASSERT((size_t)n < sizeof cflags);
                n = 0;
                for (int k = 0; RL_LINK_LIBS_[k]; k++) {
                    n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n,
                                  "%s%s", k ? " " : "", RL_LINK_LIBS_[k]);
                }
                n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n,
                              " -lgfortran -lm");
                ASSERT((size_t)n < sizeof ldlibs);
                snprintf(cmd, sizeof cmd,
                    "KFLC_CFLAGS=\"%s\" KFLC_LDLIBS=\"%s\" "
                    WORK_DIR "/prior/kflc/bin/kflc %s -o %s > "
                    WORK_DIR "/old.log 2>&1",
                    cflags, ldlibs, SHIPPED_[i].kfl, out_old);
                if (run_(cmd) != 0) {
                    (void)!system("cat " WORK_DIR "/old.log");
                    fprintf(stderr, "the prior compiler failed on %s\n",
                            SHIPPED_[i].kfl);
                    exit(1);
                }
            }
            snprintf(cmd, sizeof cmd, "%s %s --out %s.k26epi > /dev/null 2>&1",
                     out_old, SHIPPED_[i].args, out_old);
            ASSERT(run_(cmd) == 0);
            snprintf(cmd, sizeof cmd, "%s %s --out %s.k26epi > /dev/null 2>&1",
                     out_new, SHIPPED_[i].args, out_new);
            ASSERT(run_(cmd) == 0);
            char pa[320], pb[320];
            snprintf(pa, sizeof pa, "%s.k26epi", out_old);
            snprintf(pb, sizeof pb, "%s.k26epi", out_new);
            {
                struct stat sa;
                ASSERT(stat(pa, &sa) == 0);
                printf("  %-28s %ld bytes, identical: %s\n",
                       SHIPPED_[i].tag, (long)sa.st_size,
                       rl_files_equal_(pa, pb) ? "yes" : "NO");
            }
            ASSERT(rl_files_equal_(pa, pb));
        }
    }
    n_pass++;

    printf("test_rl_capture: %d gate(s) passed\n", n_pass);
    return 0;
}
