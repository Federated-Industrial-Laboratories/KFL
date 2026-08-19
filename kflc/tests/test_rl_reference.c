/* test_rl_reference.c: the plan a craft flies, through the stepping
 * surface.
 *
 * The plan format's own gates, in the record library, measure the
 * bytes: the digest over the whole file, the refusals, the ordering
 * rule and the current-knot rule against a clock with no craft in it.
 * This one measures what only an artifact can show: that the eight
 * channels are published under the names and in the order the grammar
 * promises, that they carry an error against where the craft actually
 * is rather than the plan's own numbers, that the knot advances on
 * the clock and not on arrival, that nothing beyond the current knot
 * can move a published value, and that a run's plan comes back out of
 * its recording.
 *
 * What would make each arm vacuous, and how each is ruled out.
 *
 *   A fixture with one craft cannot tell an error from an absolute:
 *   with the craft at the frame's origin the two are the same numbers.
 *   The fixture therefore carries two craft at different positions
 *   reading one plan, and the arm requires their published positions
 *   to differ by exactly their separation and neither to equal the
 *   knot the plan holds.
 *
 *   A fixture that always arrives on time cannot tell an advancing
 *   rule from a satisfied one. The two craft above are placed so that
 *   one is inside every knot's tolerance at every boundary and the
 *   other is outside every knot's tolerance at every boundary, both
 *   against the same plan, and the arm requires the advance to happen
 *   on identical steps for both. The distances are printed, so a
 *   fixture that stopped separating the two cases would say so.
 *
 *   Comparing the published channels against the same library that
 *   computed them would compare a thing with itself. The expected
 *   values are rebuilt here from the body-state getter, with the
 *   frame constructed in this file from the axis definitions in
 *   words, and nothing in that path calls the proximity library.
 *
 *   An arm requiring two artifacts to agree proves nothing unless the
 *   same comparison can disagree. The arm that alters a knot beyond
 *   the current one and requires no change is paired with the same
 *   two artifacts driven on past the boundary, where the altered knot
 *   is current and the streams are required to differ.
 *
 *   An arm requiring a refusal proves nothing unless the accepted
 *   counterpart compiles. Every refusal below is written as one
 *   change to a source that compiles, and the accepted source is
 *   compiled first.
 *
 * Arms:
 *   1. The eight published components, their names and their order,
 *      and the observe declared after them starting where an
 *      eight-wide form leaves it.
 *   2. Errors and not absolutes, over sixty steps, against a frame
 *      and a subtraction rebuilt here.
 *   3. The knot advances on the clock: at the exact boundary the knot
 *      has not passed, one step later it has, and the craft that is
 *      nowhere near the plan advances on the same steps as the one
 *      sitting on it. Past the last knot the last knot stays current
 *      and the seconds-to channel goes negative without end.
 *   4. Nothing beyond the current knot reaches the published values,
 *      with its matched disagreement.
 *   5. An altered plan file is refused where the program names it,
 *      including an alteration to the provenance text alone, which is
 *      what shows the text is inside the hashed region.
 *   6. The refusals: a frame naming a body the world does not
 *      declare, knots out of time order, a local-vertical frame on a
 *      body that orbits nothing, a plan written in the observing
 *      craft's own frame, a plan that asks for nothing, and a
 *      reference observe of a craft carrying no plan.
 *   7. The record: the plan comes back out of the recording byte for
 *      byte, parses, and agrees with the digest the record carries
 *      beside it; a record whose plan bytes are altered no longer
 *      does.
 *   8. Determinism under a plan: two processes at one seed write
 *      identical episode files.
 *   9. What a planner emits: a `plan` block declares eight action
 *      channels per slot with the bounds it named, and the actions of
 *      the step that ended the episode become a plan file and a plan
 *      frame that agree byte for byte, with the absent slot dropped
 *      and the rest in time order.
 *  10. Both new spellings survive a parse, print and parse again,
 *      which the plan block is what puts at risk: it declares action
 *      channels of its own, and a printer emitting both the block and
 *      the channels would declare each of them twice.
 *
 * The no-allocation half of the plan's stepping-path obligation is
 * measured where every other addition to that path is measured, in
 * test_rl_hotpath, whose collision fixture now carries a plan.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>
#include <signal.h>
#include <unistd.h>

#include "rl_gate_util.h"

#include "k26rl_episode.h"
#include "k26rl_ref.h"

#define WORK_DIR "/tmp/kflc_rl_reference_test"

static int g_arms;

static const char *rl_stage_name_ = "startup";

static void rl_deadline_fired_(int sig)
{
    (void)sig;
    const char *a = "test_rl_reference: DEADLINE EXCEEDED at stage: ";
    (void)!write(2, a, strlen(a));
    (void)!write(2, rl_stage_name_, strlen(rl_stage_name_));
    (void)!write(2, "\n", 1);
    _exit(1);
}

static void rl_stage_(const char *name, unsigned secs)
{
    rl_stage_name_ = name;
    alarm(secs);
}

static void rl_stage_done_(void) { alarm(0); }

/* ---- The fixture ---------------------------------------------------- *
 *
 * One plan, in the target's local-vertical local-horizontal frame,
 * and two craft reading it. `near` sits five metres along track from
 * the target and is inside every knot's tolerance at every boundary;
 * `far` sits five kilometres along track and is outside every knot's
 * tolerance at every boundary. Everything else about the two is the
 * same, so the only thing that can separate their published knot
 * advance is a rule that reads where they are.
 *
 * The line-of-sight observe after the plan is there so that a form
 * publishing its eight values into seven slots, or nine, displaces
 * something the arm can see.
 */
#define REF_KFL(tag, plan_file) \
    "form RL_REFERENCE" tag "\n" \
    "fn world ref_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body target mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    astro_body near mass=1.0 parent=earth reference=\"" plan_file "\"" \
    " pos_x=7.0e6 pos_y=5.0 vel_y=7546.049108166324\n" \
    "    astro_body far mass=1.0 parent=earth reference=\"" plan_file "\"" \
    " pos_x=7.0e6 pos_y=-5000.0 vel_y=7546.049108166324\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 60\n" \
    "        substeps 1\n" \
    "    end\n" \
    "    action idle box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        near.vel_x = near.vel_x + idle * 0.0\n" \
    "    end\n" \
    "    observe reference of near as pn\n" \
    "    observe reference of far as pf\n" \
    "    observe near from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward pn_r_y\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* The planner's side: a world whose actions are knot slots and whose
 * plan is written at the episode's end.
 *
 * Four slots, one of them given a tolerance of zero so the absent-slot
 * rule is exercised, and the times given out of order so the ordering
 * rule is too. A fixture whose slots arrived in order and were all
 * present could not tell either rule from no rule.
 */
#define PLANNER_KFL(dir) \
    "form RL_REF_PLANNER\n" \
    "fn world planner_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body target mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 3\n" \
    "    end\n" \
    "    plan approach\n" \
    "        file \"" dir "/approach\"\n" \
    "        frame target lvlh\n" \
    "        slots 4\n" \
    "        time 0.0 600.0\n" \
    "        position -5000.0 5000.0\n" \
    "        velocity -10.0 10.0\n" \
    "        tolerance 0.0 500.0\n" \
    "        epoch 7.0\n" \
    "        provenance \"the gate's planner\"\n" \
    "    end\n" \
    "    observe target from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward trk_range * 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

#define PLAN_SLOTS 4
#define PLAN_ACT   (PLAN_SLOTS * 8)

/* The slots the gate drives, in the order the action vector holds
 * them. Slot 2 is absent by its zero tolerance; the times are out of
 * order, so what the file holds is slots 1, 0 and 3, at 50, 100 and
 * 200 seconds. */
static const double PLAN_DRIVE[PLAN_SLOTS][8] = {
    { 100.0,  1.0,  2.0,  3.0,  0.1,  0.2,  0.3, 10.0 },
    {  50.0,  4.0,  5.0,  6.0,  0.4,  0.5,  0.6, 20.0 },
    { 300.0,  7.0,  8.0,  9.0,  0.7,  0.8,  0.9,  0.0 },
    { 200.0, -1.0, -2.0, -3.0, -0.1, -0.2, -0.3, 30.0 }
};
static const int PLAN_EXPECT[3] = { 1, 0, 3 };

/* Eight plan channels, eight more, five line-of-sight ones. */
#define REF_OBS   21
#define PN_BASE    0
#define PF_BASE    8
#define TRK_BASE  16
#define N_BODIES   4
#define B_EARTH    0
#define B_TARGET   1
#define B_NEAR     2
#define B_FAR      3

#define KNOTS 3

/* The plan. Times at 10, 20 and 30 seconds on an episode clock whose
 * zero is the episode's start, so at a control period of one second
 * the boundaries fall on steps 10, 20 and 30 exactly. The tolerances
 * grow, so an implementation that published a fixed one cannot pass
 * arm 1, and the positions differ, so one that published a fixed knot
 * cannot pass arm 3. */
static const double PLAN_T[KNOTS]   = { 10.0, 20.0, 30.0 };
static const double PLAN_RY[KNOTS]  = { 0.0, -2.0, -4.0 };
static const double PLAN_VY[KNOTS]  = { 0.0, 0.1, 0.2 };
static const double PLAN_TOL[KNOTS] = { 100.0, 200.0, 300.0 };

static void fill_plan_(K26RlRefKnot *ks, int last_altered)
{
    int i;

    for (i = 0; i < KNOTS; i++) {
        ks[i].t = PLAN_T[i];
        ks[i].r[0] = 0.0; ks[i].r[1] = PLAN_RY[i]; ks[i].r[2] = 0.0;
        ks[i].v[0] = 0.0; ks[i].v[1] = PLAN_VY[i]; ks[i].v[2] = 0.0;
        ks[i].tolerance = PLAN_TOL[i];
    }
    if (last_altered) {
        /* Only the last knot, and only in what it asks for. Its time
         * is untouched, so the two plans advance on identical steps
         * and the arm is about what is published rather than about
         * when. */
        ks[KNOTS - 1].r[1] = -400.0;
        ks[KNOTS - 1].v[1] = 9.0;
        ks[KNOTS - 1].tolerance = 999.0;
    }
}

static void write_plan_(const char *path, const char *frame, uint32_t kind,
                        int last_altered)
{
    K26RlRefKnot ks[KNOTS];
    K26RlRefPlan plan;

    fill_plan_(ks, last_altered);
    plan.frame_kind = kind;
    plan.frame_name = frame;
    plan.provenance = "test_rl_reference fixture";
    plan.epoch      = 0.0;
    plan.knots      = ks;
    plan.knot_count = KNOTS;
    ASSERT(k26rl_ref_write(path, &plan) == K26RL_REF_OK);
}

/* ---- Independent geometry -------------------------------------------- */

typedef struct { double x, y, z; } V3;

static V3 v3_(double x, double y, double z) { V3 v = { x, y, z }; return v; }
static V3 sub_(V3 a, V3 b) { return v3_(a.x - b.x, a.y - b.y, a.z - b.z); }
static double dot_(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static V3 cross_(V3 a, V3 b)
{
    return v3_(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static V3 scale_(V3 a, double s) { return v3_(a.x*s, a.y*s, a.z*s); }
static double len_(V3 a) { return sqrt(dot_(a, a)); }
static double absd_(double x) { return x < 0.0 ? -x : x; }

static V3 body_pos_(const double *b, int i)
{
    return v3_(b[i * 6 + 0], b[i * 6 + 1], b[i * 6 + 2]);
}

static V3 body_vel_(const double *b, int i)
{
    return v3_(b[i * 6 + 3], b[i * 6 + 4], b[i * 6 + 5]);
}

/* The craft's state in the target's local-vertical local-horizontal
 * frame, built here from the axis definitions in words: the first
 * axis points from the body the target orbits out to the target, the
 * third along the orbital angular momentum, the second completes the
 * right-handed set; the velocity is the rate an observer riding the
 * frame sees. Nothing here calls the library the artifact used. */
static void craft_in_frame_(const double *bodies, int craft,
                            double out_r[3], double out_v[3])
{
    V3 rt = body_pos_(bodies, B_TARGET);
    V3 vt = sub_(body_vel_(bodies, B_TARGET), body_vel_(bodies, B_EARTH));
    V3 rc = body_pos_(bodies, craft);
    V3 vc = body_vel_(bodies, craft);
    V3 h  = cross_(rt, vt);
    V3 e1 = scale_(rt, 1.0 / len_(rt));
    V3 e3 = scale_(h, 1.0 / len_(h));
    V3 e2 = cross_(e3, e1);
    V3 om = scale_(h, 1.0 / dot_(rt, rt));
    V3 rho = sub_(rc, rt);
    V3 dv  = sub_(vc, body_vel_(bodies, B_TARGET));
    V3 vrot = sub_(dv, cross_(om, rho));

    out_r[0] = dot_(rho, e1);
    out_r[1] = dot_(rho, e2);
    out_r[2] = dot_(rho, e3);
    out_v[0] = dot_(vrot, e1);
    out_v[1] = dot_(vrot, e2);
    out_v[2] = dot_(vrot, e3);
}

/* Which knot the rule makes current at a time, computed here from the
 * rule in words: the earliest whose time has not passed, and the last
 * one once every time has. */
static int expect_knot_(double t)
{
    int i;

    for (i = 0; i < KNOTS; i++) {
        if (PLAN_T[i] >= t) return i;
    }
    return KNOTS - 1;
}

/* ---- Refusals -------------------------------------------------------- */

static void expect_refusal_(const char *name, const char *src,
                            const char *needle)
{
    char path[512], cmd[2048];
    int rc;

    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", name);
    rl_write_file_(path, src);
    snprintf(cmd, sizeof cmd,
             "./bin/kflc --check %s > " WORK_DIR "/%s.log 2>&1", path, name);
    rc = system(cmd);
    if (rc == 0) {
        fprintf(stderr, "FAIL: %s was accepted\n", name);
        snprintf(cmd, sizeof cmd, "cat " WORK_DIR "/%s.log", name);
        (void)!system(cmd);
        exit(1);
    }
    snprintf(cmd, sizeof cmd, "grep -q -- '%s' " WORK_DIR "/%s.log",
             needle, name);
    if (system(cmd) != 0) {
        fprintf(stderr, "FAIL: %s was refused without naming `%s`\n",
                name, needle);
        snprintf(cmd, sizeof cmd, "cat " WORK_DIR "/%s.log", name);
        (void)!system(cmd);
        exit(1);
    }
    printf("  refused: %s\n", name);
}

static void expect_accepted_(const char *name, const char *src)
{
    char path[512], cmd[2048];

    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", name);
    rl_write_file_(path, src);
    snprintf(cmd, sizeof cmd,
             "./bin/kflc --check %s > " WORK_DIR "/%s.log 2>&1", path, name);
    if (system(cmd) != 0) {
        fprintf(stderr, "FAIL: %s was refused and should not have been\n",
                name);
        snprintf(cmd, sizeof cmd, "cat " WORK_DIR "/%s.log", name);
        (void)!system(cmd);
        exit(1);
    }
    printf("  accepted: %s\n", name);
}

/* A body reading a plan, with one field of the source substituted, so
 * every refusal below differs from an accepted program in exactly the
 * thing it is about. */
#define ONE_CRAFT_KFL(tag, extra_body, ref_file, observed)                \
    "form RL_REF_" tag "\n"                                               \
    "fn world w\n"                                                        \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"              \
    "    astro_body target mass=1.0 parent=earth"                         \
    " pos_x=7.0e6 vel_y=7546.049108166324\n"                              \
    extra_body                                                            \
    "    astro_body near mass=1.0 parent=earth"                           \
    " reference=\"" ref_file "\""                                         \
    " pos_x=7.0e6 pos_y=5.0 vel_y=7546.049108166324\n"                    \
    "    episode\n"                                                       \
    "        control_dt 1.0\n"                                            \
    "        horizon 8\n"                                                 \
    "    end\n"                                                           \
    "    action idle box -1.0 1.0 default 0.0\n"                          \
    "    observe reference of " observed " as pn\n"                       \
    "    objective\n"                                                     \
    "        reward pn_r_y + idle\n"                                      \
    "    end\n"                                                           \
    "end\n"                                                               \
    "end\n"

/* ---- Files ----------------------------------------------------------- */

static uint8_t *read_all_(const char *path, uint64_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long end;
    uint8_t *b;

    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    end = ftell(f);
    ASSERT(end > 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    b = (uint8_t *)malloc((size_t)end);
    ASSERT(b != NULL);
    ASSERT(fread(b, 1, (size_t)end, f) == (size_t)end);
    fclose(f);
    *out_len = (uint64_t)end;
    return b;
}

static void write_all_(const char *path, const uint8_t *b, uint64_t n)
{
    FILE *f = fopen(path, "wb");

    ASSERT(f != NULL);
    ASSERT(fwrite(b, 1, (size_t)n, f) == (size_t)n);
    ASSERT(fclose(f) == 0);
}

/* Recompute the file's own digest after a hand alteration, so a
 * refusal measured afterwards is the altered field's and not the
 * digest's. The rule is the format's: the whole file with its own 32
 * digest bytes read as zero. */
static void redigest_(uint8_t *b, uint64_t n)
{
    static const uint8_t zeros[K26RL_SHA256_BYTES] = { 0 };
    K26RlSha256 s;
    uint8_t d[K26RL_SHA256_BYTES];

    k26rl_sha256_init(&s);
    k26rl_sha256_update(&s, b, (uint64_t)K26RL_REF_DIGEST_OFFSET);
    k26rl_sha256_update(&s, zeros, (uint64_t)K26RL_SHA256_BYTES);
    k26rl_sha256_update(&s, b + K26RL_REF_DIGEST_OFFSET + K26RL_SHA256_BYTES,
                        n - (K26RL_REF_DIGEST_OFFSET + K26RL_SHA256_BYTES));
    k26rl_sha256_final(&s, d);
    memcpy(b + K26RL_REF_DIGEST_OFFSET, d, K26RL_SHA256_BYTES);
}

static void put_f64_(uint8_t *p, double v)
{
    uint64_t bits;
    int i;

    memcpy(&bits, &v, sizeof bits);
    for (i = 0; i < 8; i++) p[i] = (uint8_t)((bits >> (8 * i)) & 0xFFu);
}

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGALRM, rl_deadline_fired_);

    if (!rl_libs_present_("test_rl_reference")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    write_plan_(WORK_DIR "/plan.k26ref", "target", K26RL_REF_FRAME_LVLH, 0);
    rl_write_file_(WORK_DIR "/ref.kfl", REF_KFL("", "plan.k26ref"));

    rl_stage_("compiling the plan artifact", 900u);
    rl_compile_(WORK_DIR "/ref.kfl", WORK_DIR "/ref", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/ref.rlenv.so"));
    rl_stage_done_();

    void *so = rl_dlopen_(WORK_DIR "/ref.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    K26RlEnv *env = NULL;
    ASSERT(s.create(23u, 1u, &env) == K26RL_OK);

    /* ---- Arm 1: the published names and their order ------------- */
    rl_stage_("reading the spec", 60u);
    {
        int32_t need = s.spec(env, NULL, 0);
        uint8_t *blob;
        RlSpecView v;
        static const char *const want[] = {
            "pn_r_x", "pn_r_y", "pn_r_z", "pn_v_x", "pn_v_y", "pn_v_z",
            "pn_time_to", "pn_tolerance",
            "pf_r_x", "pf_r_y", "pf_r_z", "pf_v_x", "pf_v_y", "pf_v_z",
            "pf_time_to", "pf_tolerance",
            "trk_dir_x"
        };
        size_t k;

        ASSERT(need > 0);
        blob = (uint8_t *)malloc((size_t)need);
        ASSERT(blob != NULL);
        ASSERT(s.spec(env, blob, (uint32_t)need) == need);
        rl_parse_spec_(blob, (uint32_t)need, &v);
        printf("arm 1: obs_total %u (eight plan channels twice, then five"
               " line-of-sight ones)\n", v.obs_total);
        ASSERT(v.obs_total == REF_OBS);
        ASSERT(v.n_chan_names >= (int)(sizeof want / sizeof want[0]));
        for (k = 0; k < sizeof want / sizeof want[0]; k++) {
            if (strcmp(v.chan_names[k], want[k]) != 0) {
                fprintf(stderr, "FAIL: channel %d is `%s`, expected `%s`\n",
                        (int)k, v.chan_names[k], want[k]);
                exit(1);
            }
        }
        /* Every plan channel is geometric: there is no observer, no
         * light time and no correction of any kind. */
        for (k = 0; k < 16; k++) {
            ASSERT(v.modes[k] == K26RL_OBS_MODE_GEOMETRIC);
        }
        printf("arm 1: eight components each, in order, and the observe"
               " after them starts at %d: OK\n", TRK_BASE);
        free(blob);
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arms 2 and 3: errors, and the advance rule ------------- */
    rl_stage_("driving the plan artifact", 300u);
    {
        static double obs[REF_OBS];
        static double bodies[N_BODIES * 6];
        double act[1] = { 0.0 };
        int advance_step[KNOTS];
        int near_inside = 0, far_outside = 0, late_steps = 0;
        int k, t;

        for (k = 0; k < KNOTS; k++) advance_step[k] = -1;

        for (t = 0; t <= 45; t++) {
            double nr[3], nv[3], fr[3], fv[3];
            double time_now = (double)t;
            int want_knot = expect_knot_(time_now);
            double dn, df;
            int j;

            ASSERT(s.obs(env, obs) == K26RL_OK);
            ASSERT(s.bodies(env, B_EARTH, bodies, N_BODIES * 6) ==
                   N_BODIES * 6);
            craft_in_frame_(bodies, B_NEAR, nr, nv);
            craft_in_frame_(bodies, B_FAR, fr, fv);

            /* Arm 2: the published position is the knot less where the
             * craft is, in the plan's own frame, and the same for the
             * velocity. Rebuilt here from the body states. */
            for (j = 0; j < 3; j++) {
                double want_r = (j == 1 ? PLAN_RY[want_knot] : 0.0) - nr[j];
                double want_v = (j == 1 ? PLAN_VY[want_knot] : 0.0) - nv[j];

                ASSERT(absd_(obs[PN_BASE + j] - want_r) < 1.0e-9);
                ASSERT(absd_(obs[PN_BASE + 3 + j] - want_v) < 1.0e-12);
                want_r = (j == 1 ? PLAN_RY[want_knot] : 0.0) - fr[j];
                want_v = (j == 1 ? PLAN_VY[want_knot] : 0.0) - fv[j];
                ASSERT(absd_(obs[PF_BASE + j] - want_r) < 1.0e-6);
                ASSERT(absd_(obs[PF_BASE + 3 + j] - want_v) < 1.0e-12);
            }
            /* The two craft read one plan. Their published positions
             * must therefore differ by exactly their separation in the
             * frame, and neither may equal the plan's own number: an
             * implementation publishing the knot rather than the error
             * gives both craft identical channels. */
            for (j = 0; j < 3; j++) {
                double sep = fr[j] - nr[j];

                ASSERT(absd_((obs[PN_BASE + j] - obs[PF_BASE + j]) - sep) <
                       1.0e-6);
            }
            ASSERT(absd_(obs[PN_BASE + 1] - obs[PF_BASE + 1]) > 1.0e3);
            ASSERT(absd_(obs[PN_BASE + 1] - PLAN_RY[want_knot]) > 1.0);
            ASSERT(absd_(obs[PF_BASE + 1] - PLAN_RY[want_knot]) > 1.0e3);

            /* Arm 3: which knot is current, read off the published
             * tolerance, which the plan makes distinct per knot. */
            ASSERT(obs[PN_BASE + 7] == PLAN_TOL[want_knot]);
            ASSERT(obs[PF_BASE + 7] == PLAN_TOL[want_knot]);
            ASSERT(absd_(obs[PN_BASE + 6] -
                         (PLAN_T[want_knot] - time_now)) < 1.0e-9);
            ASSERT(absd_(obs[PF_BASE + 6] -
                         (PLAN_T[want_knot] - time_now)) < 1.0e-9);
            if (advance_step[want_knot] < 0) advance_step[want_knot] = t;
            if (obs[PN_BASE + 6] < 0.0) late_steps++;

            /* The separation that makes the arm about time. At every
             * boundary one craft is inside the tolerance and the other
             * is outside it. */
            dn = sqrt((PLAN_RY[want_knot] - nr[1]) *
                      (PLAN_RY[want_knot] - nr[1]) +
                      nr[0] * nr[0] + nr[2] * nr[2]);
            df = sqrt((PLAN_RY[want_knot] - fr[1]) *
                      (PLAN_RY[want_knot] - fr[1]) +
                      fr[0] * fr[0] + fr[2] * fr[2]);
            if (t == 10 || t == 20 || t == 30) {
                printf("arm 3: at t=%2d knot %d is current;"
                       " near is %8.3f m from it (tolerance %5.1f),"
                       " far is %10.3f m from it\n",
                       t, want_knot, dn, PLAN_TOL[want_knot], df);
                ASSERT(dn < PLAN_TOL[want_knot]);
                ASSERT(df > PLAN_TOL[want_knot]);
                near_inside++;
                far_outside++;
            }
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        /* The knot in force changes exactly at the step after each
         * boundary, for both craft, and the plan does not run out: the
         * last knot stays current and the seconds-to channel goes on
         * falling. */
        printf("arm 3: knot 0 from step %d, knot 1 from step %d,"
               " knot 2 from step %d; %d late steps\n",
               advance_step[0], advance_step[1], advance_step[2],
               late_steps);
        ASSERT(advance_step[0] == 0);
        ASSERT(advance_step[1] == 11);
        ASSERT(advance_step[2] == 21);
        ASSERT(near_inside == 3 && far_outside == 3);
        ASSERT(late_steps == 15);
        printf("arm 2: two craft, one plan, positions differing by their"
               " separation and neither equal to the plan's own: OK\n");
        printf("arm 3: the knot advances on the clock, identically for a"
               " craft on the plan and one five kilometres from it: OK\n");
        g_arms += 2;
        s.destroy(env);
    }
    rl_stage_done_();

    /* ---- Arm 4: nothing beyond the current knot ----------------- */
    rl_stage_("compiling the altered-knot artifact", 900u);
    {
        static double a_obs[REF_OBS], b_obs[REF_OBS];
        double act[1] = { 0.0 };
        K26RlEnv *ea = NULL, *eb = NULL;
        void *so_b;
        RlSurface sb;
        int t, differing = 0;

        write_plan_(WORK_DIR "/plan2.k26ref", "target",
                    K26RL_REF_FRAME_LVLH, 1);
        /* The same world, naming the altered plan and nothing else
         * changed but the form's name, which two forms in one
         * directory may not share. */
        rl_write_file_(WORK_DIR "/ref2.kfl", REF_KFL("2", "plan2.k26ref"));
        rl_compile_(WORK_DIR "/ref2.kfl", WORK_DIR "/ref2", WORK_DIR);
        ASSERT(rl_file_exists_(WORK_DIR "/ref2.rlenv.so"));
        so_b = rl_dlopen_(WORK_DIR "/ref2.rlenv.so");
        rl_resolve_surface_(so_b, &sb);

        ASSERT(s.create(23u, 1u, &ea) == K26RL_OK);
        ASSERT(sb.create(23u, 1u, &eb) == K26RL_OK);
        for (t = 0; t <= 40; t++) {
            int j, same = 1;

            ASSERT(s.obs(ea, a_obs) == K26RL_OK);
            ASSERT(sb.obs(eb, b_obs) == K26RL_OK);
            for (j = 0; j < REF_OBS; j++) {
                if (memcmp(&a_obs[j], &b_obs[j], sizeof a_obs[j]) != 0)
                    same = 0;
            }
            if (t <= 20) {
                /* Knot 0 or knot 1 is current. The plans differ only
                 * in knot 2, so nothing may differ here. */
                if (!same) {
                    fprintf(stderr, "FAIL: a knot beyond the current one "
                            "moved a published value at step %d\n", t);
                    for (j = 0; j < REF_OBS; j++) {
                        if (a_obs[j] != b_obs[j])
                            fprintf(stderr, "  channel %d: %.17g vs %.17g\n",
                                    j, a_obs[j], b_obs[j]);
                    }
                    exit(1);
                }
            } else if (!same) {
                differing++;
            }
            ASSERT(s.step(ea, act) == K26RL_OK);
            ASSERT(sb.step(eb, act) == K26RL_OK);
        }
        /* The matched disagreement. Once the altered knot is current
         * the two artifacts must part, or the comparison above was
         * comparing two things that could never differ. */
        printf("arm 4: identical through step 20, then %d of 20 steps"
               " differ once the altered knot is current\n", differing);
        ASSERT(differing == 20);
        s.destroy(ea);
        sb.destroy(eb);
        dlclose(so_b);
        printf("arm 4: altering a knot beyond the current one changes"
               " nothing an executor can see: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arm 5: an altered plan file ---------------------------- */
    rl_stage_("the altered-file refusals", 300u);
    {
        uint64_t n = 0;
        uint8_t *bytes = read_all_(WORK_DIR "/plan.k26ref", &n);
        const char *acc =
            ONE_CRAFT_KFL("ACC", "", "plan.k26ref", "near");

        expect_accepted_("ref_accepted", acc);

        /* A knot's own bytes. */
        {
            uint8_t *copy = (uint8_t *)malloc((size_t)n);

            ASSERT(copy != NULL);
            memcpy(copy, bytes, (size_t)n);
            copy[n - 1u] ^= 0x01u;
            write_all_(WORK_DIR "/altered_knot.k26ref", copy, n);
            expect_refusal_("ref_altered_knot",
                ONE_CRAFT_KFL("AK", "", "altered_knot.k26ref", "near"),
                "digest disagrees");
            free(copy);
        }
        /* The provenance text alone, which is what shows the text is
         * inside the hashed region rather than beside it. */
        {
            uint8_t *copy = (uint8_t *)malloc((size_t)n);
            uint64_t at = K26RL_REF_HEADER_BYTES + strlen("target");

            ASSERT(copy != NULL);
            memcpy(copy, bytes, (size_t)n);
            copy[at] = (uint8_t)(copy[at] ^ 0x20u);
            write_all_(WORK_DIR "/altered_prov.k26ref", copy, n);
            expect_refusal_("ref_altered_prov",
                ONE_CRAFT_KFL("AP", "", "altered_prov.k26ref", "near"),
                "digest disagrees");
            free(copy);
        }
        free(bytes);
        printf("arm 5: an altered plan is refused where the program names"
               " it, the provenance text included: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arm 6: the remaining refusals -------------------------- */
    rl_stage_("the declaration refusals", 300u);
    {
        uint64_t n = 0;
        uint8_t *bytes;

        /* A frame naming a body this world does not declare. */
        write_plan_(WORK_DIR "/plan_ghost.k26ref", "ghost",
                    K26RL_REF_FRAME_LVLH, 0);
        expect_refusal_("ref_ghost_frame",
            ONE_CRAFT_KFL("GF", "", "plan_ghost.k26ref", "near"),
            "no astro_body of that name is declared");

        /* Knots out of time order. The writer orders them, so the file
         * is built by hand from an accepted one and re-digested. */
        bytes = read_all_(WORK_DIR "/plan.k26ref", &n);
        {
            uint64_t at = K26RL_REF_HEADER_BYTES + strlen("target") +
                          strlen("test_rl_reference fixture");

            put_f64_(bytes + at + K26RL_REF_KNOT_BYTES, 1.0);
            redigest_(bytes, n);
            write_all_(WORK_DIR "/plan_unordered.k26ref", bytes, n);
        }
        free(bytes);
        expect_refusal_("ref_unordered",
            ONE_CRAFT_KFL("UO", "", "plan_unordered.k26ref", "near"),
            "do not strictly increase");

        /* A local-vertical frame on a body that orbits nothing. */
        write_plan_(WORK_DIR "/plan_orphan.k26ref", "drifter",
                    K26RL_REF_FRAME_LVLH, 0);
        expect_refusal_("ref_orphan_frame",
            ONE_CRAFT_KFL("OF",
                "    astro_body drifter mass=1.0"
                " pos_x=7.1e6 vel_y=7500.0\n",
                "plan_orphan.k26ref", "near"),
            "declares no `parent=`");
        /* The same plan in an inertial frame asks nothing of the body
         * it names, so the accepted counterpart is the same source
         * with the frame kind changed and nothing else. */
        write_plan_(WORK_DIR "/plan_orphan_i.k26ref", "drifter",
                    K26RL_REF_FRAME_INERTIAL, 0);
        expect_accepted_("ref_orphan_inertial",
            ONE_CRAFT_KFL("OI",
                "    astro_body drifter mass=1.0"
                " pos_x=7.1e6 vel_y=7500.0\n",
                "plan_orphan_i.k26ref", "near"));

        /* A plan written in the observing craft's own frame, whose
         * published components would be the plan's own numbers. */
        write_plan_(WORK_DIR "/plan_self.k26ref", "near",
                    K26RL_REF_FRAME_INERTIAL, 0);
        expect_refusal_("ref_self_frame",
            ONE_CRAFT_KFL("SF", "", "plan_self.k26ref", "near"),
            "itself");

        /* A plan whose every slot is absent, written through the
         * writer, which drops absent slots: the file then carries no
         * knot at all and the reader says so.
         *
         * And a plan whose slots are all present in the file and all
         * absent by tolerance, built by hand because the writer will
         * not produce one. The two refusals are different and both
         * are reachable, which is why both are here: one is the
         * format's and one is this compiler's. */
        {
            K26RlRefKnot ks[2];
            K26RlRefPlan plan;
            int i;

            for (i = 0; i < 2; i++) {
                ks[i].t = (double)(i + 1);
                ks[i].r[0] = ks[i].r[1] = ks[i].r[2] = 0.0;
                ks[i].v[0] = ks[i].v[1] = ks[i].v[2] = 0.0;
                ks[i].tolerance = 0.0;
            }
            plan.frame_kind = K26RL_REF_FRAME_LVLH;
            plan.frame_name = "target";
            plan.provenance = "empty";
            plan.epoch = 0.0;
            plan.knots = ks;
            plan.knot_count = 2;
            ASSERT(k26rl_ref_write(WORK_DIR "/plan_empty.k26ref", &plan) ==
                   K26RL_REF_OK);
        }
        expect_refusal_("ref_empty",
            ONE_CRAFT_KFL("EM", "", "plan_empty.k26ref", "near"),
            "carries no knot");

        bytes = read_all_(WORK_DIR "/plan.k26ref", &n);
        {
            uint64_t at = K26RL_REF_HEADER_BYTES + strlen("target") +
                          strlen("test_rl_reference fixture");
            int k;

            for (k = 0; k < KNOTS; k++) {
                put_f64_(bytes + at + (uint64_t)k * K26RL_REF_KNOT_BYTES +
                         7u * 8u, 0.0);
            }
            redigest_(bytes, n);
            write_all_(WORK_DIR "/plan_absent.k26ref", bytes, n);
        }
        free(bytes);
        expect_refusal_("ref_absent",
            ONE_CRAFT_KFL("AB", "", "plan_absent.k26ref", "near"),
            "asks for nothing");

        /* A reference observe of a craft that declares no plan. */
        expect_refusal_("ref_no_plan",
            ONE_CRAFT_KFL("NP", "", "plan.k26ref", "target"),
            "declares no `reference=`");

        printf("arm 6: every declaration refusal fires, and the accepted"
               " counterparts compile: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arm 7: the plan in the record -------------------------- */
    rl_stage_("recording and recovering the plan", 300u);
    {
        K26RlEnv *er = NULL;
        K26RlEpisodeReader *rd = NULL;
        K26RlEpisodePlan got;
        uint32_t plans = 0;
        double act[1] = { 0.0 };
        uint64_t plan_len = 0;
        uint8_t *plan_bytes = read_all_(WORK_DIR "/plan.k26ref", &plan_len);
        int t;

        rl_run_or_die_("rm -f " WORK_DIR "/run.k26epi");
        ASSERT(s.create(31u, 1u, &er) == K26RL_OK);
        ASSERT(s.output(er, WORK_DIR "/run.k26epi") == K26RL_OK);
        for (t = 0; t < 70; t++) ASSERT(s.step(er, act) == K26RL_OK);
        s.destroy(er);

        ASSERT(k26rl_episode_reader_open(WORK_DIR "/run.k26epi", &rd) ==
               K26RL_OK);
        ASSERT(k26rl_episode_reader_plans(rd, &plans) == K26RL_OK);
        printf("arm 7: the recording carries %u plan frame(s) for the"
               " %d craft that fly one\n", plans, 2);
        ASSERT(plans == 2);
        for (uint32_t p = 0; p < plans; p++) {
            K26RlRef *back = NULL;
            K26RlRefInfo info;
            int k;

            ASSERT(k26rl_episode_reader_plan(rd, p, &got) == K26RL_OK);
            ASSERT(got.role == K26RL_PLAN_ROLE_FLOWN);
            ASSERT(got.env == K26RL_PLAN_ALL);
            ASSERT(got.episode == K26RL_PLAN_ALL);
            /* Byte for byte the file the program named. */
            ASSERT(got.len == (uint32_t)plan_len);
            ASSERT(memcmp(got.bytes, plan_bytes, (size_t)plan_len) == 0);
            /* And a plan again, agreeing with the digest the record
             * carries beside it rather than only with the one inside
             * it. */
            ASSERT(k26rl_ref_parse(got.bytes, got.len, &back) ==
                   K26RL_REF_OK);
            ASSERT(k26rl_ref_info(back, &info) == K26RL_REF_OK);
            ASSERT(memcmp(info.digest, got.digest, K26RL_SHA256_BYTES) == 0);
            ASSERT(info.present_count == KNOTS);
            ASSERT(strcmp(k26rl_ref_frame_name(back, NULL), "target") == 0);
            for (k = 0; k < KNOTS; k++) {
                K26RlRefKnot kn;

                ASSERT(k26rl_ref_knot(back, (uint32_t)k, &kn) ==
                       K26RL_REF_OK);
                ASSERT(kn.t == PLAN_T[k]);
                ASSERT(kn.r[1] == PLAN_RY[k]);
                ASSERT(kn.v[1] == PLAN_VY[k]);
                ASSERT(kn.tolerance == PLAN_TOL[k]);
            }
            k26rl_ref_close(back);

            /* The matched failure: a plan altered inside the record no
             * longer agrees with the digest the record carries, so a
             * recording flown against a plan of the same name is
             * distinguishable from one flown against the plan. */
            got.bytes[got.len - 1u] ^= 0x01u;
            back = NULL;
            ASSERT(k26rl_ref_parse(got.bytes, got.len, &back) ==
                   K26RL_REF_E_DIGEST);
            k26rl_episode_plan_free(&got);
        }
        k26rl_episode_reader_close(rd);
        free(plan_bytes);
        printf("arm 7: a run's plan is recoverable from its recording"
               " alone and checkable against it: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arm 8: determinism under a plan ------------------------ */
    rl_stage_("determinism under a plan", 600u);
    {
        char cmd[2048];
        int i;

        /* Two separate processes at one seed, each writing its own
         * episode file through the batch executable, compared byte for
         * byte. A plan that reached a generator, a clock or an address
         * would separate them. */
        for (i = 0; i < 2; i++) {
            snprintf(cmd, sizeof cmd,
                     "cd " WORK_DIR " && rm -f det%d.k26epi && "
                     "./ref --envs 2 --episodes 3 --seed 31 "
                     "--out det%d.k26epi > det%d.log 2>&1", i, i, i);
            rl_run_or_die_(cmd);
        }
        ASSERT(rl_files_equal_(WORK_DIR "/det0.k26epi",
                               WORK_DIR "/det1.k26epi"));
        printf("arm 8: two processes at one seed wrote identical episode"
               " files for a plan-bearing program: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arm 9: what a planner emits ---------------------------- */
    rl_stage_("compiling and driving the planner", 900u);
    {
        void *pso;
        RlSurface ps;
        K26RlEnv *pe = NULL;
        double act[PLAN_ACT];
        K26RlEpisodeReader *rd = NULL;
        K26RlEpisodePlan got;
        K26RlRef *back = NULL;
        K26RlRefInfo info;
        uint32_t plans = 0;
        uint64_t flen = 0;
        uint8_t *fbytes;
        int i, k;

        rl_run_or_die_("mkdir -p " WORK_DIR "/plans");
        rl_write_file_(WORK_DIR "/planner.kfl",
                       PLANNER_KFL(WORK_DIR "/plans"));
        rl_compile_(WORK_DIR "/planner.kfl", WORK_DIR "/planner", WORK_DIR);
        ASSERT(rl_file_exists_(WORK_DIR "/planner.rlenv.so"));
        pso = rl_dlopen_(WORK_DIR "/planner.rlenv.so");
        rl_resolve_surface_(pso, &ps);

        /* The block declares its own action channels, eight to a slot,
         * with the bounds the block gave. A spec that published fewer,
         * or in another order, is a planner whose policy cannot be
         * shaped to it. */
        ASSERT(ps.create(3u, 1u, &pe) == K26RL_OK);
        {
            int32_t need = ps.spec(pe, NULL, 0);
            uint8_t *blob = (uint8_t *)malloc((size_t)need);
            RlSpecView v;
            uint32_t off = 0;
            int seen = 0;

            ASSERT(need > 0 && blob != NULL);
            ASSERT(ps.spec(pe, blob, (uint32_t)need) == need);
            rl_parse_spec_(blob, (uint32_t)need, &v);
            ASSERT(v.act_total == PLAN_ACT);
            /* The declared bounds, walked from the tag stream. */
            while (off + 6 <= (uint32_t)need) {
                uint16_t tag = rl_get_u16_(blob + off);
                uint32_t l   = rl_get_u32_(blob + off + 2);
                const uint8_t *val = blob + off + 6;

                if (tag == K26RL_TAG_ACT_BOUNDS && l == 20u) {
                    uint32_t ch = rl_get_u32_(val);
                    double lo = rl_get_f64_(val + 4);
                    double hi = rl_get_f64_(val + 12);

                    if (ch == 0) {
                        ASSERT(lo == 0.0 && hi == 600.0);
                        seen++;
                    }
                    if (ch == 7) {
                        ASSERT(lo == 0.0 && hi == 500.0);
                        seen++;
                    }
                    if (ch == 1) {
                        ASSERT(lo == -5000.0 && hi == 5000.0);
                        seen++;
                    }
                    if (ch == 4) {
                        ASSERT(lo == -10.0 && hi == 10.0);
                        seen++;
                    }
                }
                off += 6 + l;
            }
            /* Four channels of three different bound pairs, so a
             * spec publishing one pair for every channel cannot
             * pass. */
            ASSERT(seen == 4);
            free(blob);
        }
        printf("arm 9: the block declared %d action channels with the"
               " bounds it named\n", PLAN_ACT);

        rl_run_or_die_("rm -f " WORK_DIR "/plans/*" K26RL_REF_SUFFIX
                       " " WORK_DIR "/planner.k26epi");
        ASSERT(ps.output(pe, WORK_DIR "/planner.k26epi") == K26RL_OK);
        for (i = 0; i < PLAN_SLOTS; i++) {
            for (k = 0; k < 8; k++) act[i * 8 + k] = PLAN_DRIVE[i][k];
        }
        for (i = 0; i < 3; i++) ASSERT(ps.step(pe, act) == K26RL_OK);
        ps.destroy(pe);

        /* The file the world wrote, named for the episode it came out
         * of, and holding exactly the present slots in time order. */
        fbytes = read_all_(WORK_DIR "/plans/approach-0-0-0" K26RL_REF_SUFFIX,
                           &flen);
        ASSERT(k26rl_ref_parse(fbytes, flen, &back) == K26RL_REF_OK);
        ASSERT(k26rl_ref_info(back, &info) == K26RL_REF_OK);
        ASSERT(info.frame_kind == K26RL_REF_FRAME_LVLH);
        ASSERT(strcmp(k26rl_ref_frame_name(back, NULL), "target") == 0);
        ASSERT(strcmp(k26rl_ref_provenance(back, NULL),
                      "the gate's planner") == 0);
        ASSERT(info.epoch == 7.0);
        ASSERT(info.knot_count == 3 && info.present_count == 3);
        for (k = 0; k < 3; k++) {
            const double *want_k = PLAN_DRIVE[PLAN_EXPECT[k]];
            K26RlRefKnot kn;
            int q;

            ASSERT(k26rl_ref_knot(back, (uint32_t)k, &kn) == K26RL_REF_OK);
            ASSERT(kn.t == want_k[0]);
            for (q = 0; q < 3; q++) {
                ASSERT(kn.r[q] == want_k[1 + q]);
                ASSERT(kn.v[q] == want_k[4 + q]);
            }
            ASSERT(kn.tolerance == want_k[7]);
        }
        k26rl_ref_close(back);
        printf("arm 9: the emitted plan holds the three present slots at"
               " %g, %g and %g seconds, the absent one dropped\n",
               PLAN_DRIVE[PLAN_EXPECT[0]][0], PLAN_DRIVE[PLAN_EXPECT[1]][0],
               PLAN_DRIVE[PLAN_EXPECT[2]][0]);

        /* And the recording carries the same plan, marked as one this
         * run produced rather than one it flew. */
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/planner.k26epi", &rd) ==
               K26RL_OK);
        ASSERT(k26rl_episode_reader_plans(rd, &plans) == K26RL_OK);
        ASSERT(plans == 1);
        ASSERT(k26rl_episode_reader_plan(rd, 0, &got) == K26RL_OK);
        ASSERT(got.role == K26RL_PLAN_ROLE_EMITTED);
        ASSERT(got.env == 0 && got.episode == 0);
        ASSERT(got.len == (uint32_t)flen);
        ASSERT(memcmp(got.bytes, fbytes, (size_t)flen) == 0);
        back = NULL;
        ASSERT(k26rl_ref_parse(got.bytes, got.len, &back) == K26RL_REF_OK);
        ASSERT(k26rl_ref_info(back, &info) == K26RL_REF_OK);
        ASSERT(memcmp(info.digest, got.digest, K26RL_SHA256_BYTES) == 0);
        k26rl_ref_close(back);
        k26rl_episode_plan_free(&got);
        k26rl_episode_reader_close(rd);
        free(fbytes);
        dlclose(pso);
        printf("arm 9: a planner's actions become a plan file and a plan"
               " frame that agree byte for byte: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    /* ---- Arm 10: the spelling survives a round trip -------------- */
    rl_stage_("round-tripping both new spellings", 120u);
    {
        /* Both constructs are new spellings, and a program that
         * parses, prints and parses again must give the same tree. The
         * plan block is the one at risk: it declares action channels
         * of its own, and a printer that emitted both the block and
         * the channels would declare each of them twice on the way
         * back in. The round-trip corpus does not reach either
         * construct, so the check is taken here over this gate's own
         * fixtures. */
        static const char *const srcs[] = {
            WORK_DIR "/ref.kfl", WORK_DIR "/planner.kfl", NULL
        };
        int k;

        for (k = 0; srcs[k]; k++) {
            char cmd[1024];

            snprintf(cmd, sizeof cmd,
                     "./tests/round_trip %s > " WORK_DIR "/rt%d.log 2>&1",
                     srcs[k], k);
            if (system(cmd) != 0) {
                fprintf(stderr, "FAIL: %s did not round-trip\n", srcs[k]);
                snprintf(cmd, sizeof cmd, "cat " WORK_DIR "/rt%d.log", k);
                (void)!system(cmd);
                exit(1);
            }
        }
        printf("arm 10: both spellings parse, print and parse again to"
               " the same tree: OK\n");
        g_arms++;
    }
    rl_stage_done_();

    dlclose(so);
    ASSERT(g_arms == 10);
    printf("test_rl_reference: %d arms passed\n", g_arms);
    return 0;
}
