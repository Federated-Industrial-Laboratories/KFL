/* test_rl_contextual_keywords.c: a construct word is still a name.
 *
 * `agent`, `sensor`, `on_step`, `astro_payload`, `capture_envelope`
 * and `engage` introduce constructs at statement position inside a
 * `fn world` body, and `at`, `effect`, `against` and `full` are read
 * as connectives or marks inside three of those statements. None of
 * the ten is a reserved word: a Grammar 3.1 program that binds one of
 * them as an ordinary identifier compiles and behaves as it always
 * did, because each opens its construct only when what follows is
 * what that construct's form requires.
 *
 * Gates, one arm per keyword per shape:
 *   1. The identifier readings. `<word> = 2.0`, `<word>(3.0)` as a
 *      bare statement, `<word>[0] = 1.0`, and `let <word>: double`
 *      compile, and the value the program computes through the name is
 *      the value it should be, so the word is not merely tolerated but
 *      read as the binding it names.
 *   1b. The same readings inside an `on_step` body, which is the block
 *      `engage` is live in and therefore the block where a greedy
 *      reading of it would bite. A world-prefix arm alone could not
 *      see that, because `engage` is not a construct there at all.
 *   2. The construct readings. `agent <name>`, `sensor <name>`,
 *      `astro_payload <name> ...`, `engage <payload> at <target>`,
 *      `observe effect <payload> as <name>` and a bare `on_step` still
 *      open their constructs in a program that uses them.
 *   2b. The two connectives. A body called `at` is engaged by name,
 *      and a body called `effect` keeps its line-of-sight observe,
 *      which is the shape the effector form is told apart from.
 *
 * Both halves are needed and neither is sufficient. A fixture holding
 * only the block form cannot tell a disambiguating parser from a
 * greedy one, which is how the identifier readings were lost in the
 * first place; a fixture holding only the identifier form cannot tell
 * a disambiguating parser from one that has dropped the construct.
 *
 * `episode`, `action` and `objective` are deliberately not here. They
 * were placed on the reserved-name table before their constructs
 * landed, so a program binding one of them was warned in advance, and
 * they are greedy by that decision rather than by oversight.
 *
 * Pattern: run ./bin/kflc via system() on generated sources and assert
 * on the exit status and, for the identifier readings, on the batch
 * program's own printed output. Needs only the built kflc for the
 * check arms; the run arms skip (77) with the drive gates when the
 * sibling stack archives are absent.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_ctxkw_test"

/* The words this gate is about. The first four open blocks in the
 * world prefix; `engage` opens a statement inside `on_step`; `at` and
 * `effect` are connectives inside two statements and are here because
 * a word read anywhere is a word that can be lost everywhere. */
static const char *const WORDS[] = {
    "agent", "sensor", "on_step", "astro_payload", "capture_envelope",
    "engage", "at", "effect", "against", "full", NULL
};
#define WORD_COUNT 10

static int g_arms;

static int check_(const char *src, char **out_log)
{
    rl_write_file_(WORK_DIR "/case.kfl", src);
    int rc = system("./bin/kflc --check " WORK_DIR "/case.kfl > "
                    WORK_DIR "/case.log 2>&1");
    if (out_log) {
        FILE *f = fopen(WORK_DIR "/case.log", "rb");
        ASSERT(f != NULL);
        static char buf[8192];
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = '\0';
        fclose(f);
        *out_log = buf;
    }
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* A fixture the compiler must refuse, and what its diagnostic must
 * say. The identifier arms show a word surviving; this shows a word
 * not being claimed where it has no business being claimed, which is
 * the other half of a contextual reading and the half a fixture of
 * accepted programs cannot see. */
static void must_refuse_(const char *what, const char *src,
                         const char *expect)
{
    char *log = NULL;
    int rc = check_(src, &log);
    if (rc == 0 || strstr(log, expect) == NULL) {
        fprintf(stderr, "FAIL %s: rc=%d, log lacks \"%s\"\n---\n%s---\n%s",
                what, rc, expect, log, src);
        exit(1);
    }
    g_arms++;
    printf("  refused: %s\n", what);
}

static void must_compile_(const char *what, const char *src)
{
    char *log = NULL;
    int rc = check_(src, &log);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s: refused\n---\n%s---\n%s", what, log, src);
        exit(1);
    }
    g_arms++;
    printf("  compiles: %s\n", what);
}

/* Build the batch executable and run it, returning its stdout. The
 * identifier readings are checked by what the program computes, not
 * only by whether it compiled: a parser that swallowed the word and
 * silently dropped the statement would still compile. */
static void must_print_(const char *what, const char *src,
                        const char *expect)
{
    rl_write_file_(WORK_DIR "/run.kfl", src);
    rl_compile_(WORK_DIR "/run.kfl", WORK_DIR "/run", WORK_DIR);
    int rc = system(WORK_DIR "/run > " WORK_DIR "/run.out 2>&1");
    ASSERT(rc == 0);
    FILE *f = fopen(WORK_DIR "/run.out", "rb");
    ASSERT(f != NULL);
    char out[4096];
    size_t n = fread(out, 1, sizeof out - 1, f);
    out[n] = '\0';
    fclose(f);
    if (strstr(out, expect) == NULL) {
        fprintf(stderr, "FAIL %s: output lacks \"%s\"\n---\n%s---\n",
                what, expect, out);
        exit(1);
    }
    g_arms++;
    printf("  runs: %s (printed %s)\n", what, expect);
}

/* ---- The identifier readings ---------------------------------------- */

/* `<word> = 2.0`, with the value printed so the assignment is known to
 * have happened rather than to have been parsed away. */
static void arm_assignment_(const char *w, int run)
{
    char src[2048], what[128];
    snprintf(src, sizeof src,
        "form CTX\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    let %s: double = 1.0\n"
        "    %s = 2.0\n"
        "    print %s\n"
        "end\n"
        "end\n", w, w, w);
    snprintf(what, sizeof what, "`%s` assigned as a name", w);
    must_compile_(what, src);
    if (run) must_print_(what, src, "2");
}

/* `<word>(3.0)` as a bare statement, with a second call whose result
 * is printed, so the name resolves to the program's own function. */
static void arm_call_(const char *w, int run)
{
    char src[2048], what[128];
    snprintf(src, sizeof src,
        "form CTX\n"
        "fn double %s(double x)\n"
        "    return x + 1.0\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    %s(3.0)\n"
        "    print %s(41.0)\n"
        "end\n"
        "end\n", w, w, w);
    snprintf(what, sizeof what, "`%s` called as a function", w);
    must_compile_(what, src);
    if (run) must_print_(what, src, "42");
}

/* `<word>[0] = 1.0`, the index-assignment shape. */
static void arm_index_(const char *w, int run)
{
    char src[2048], what[128];
    snprintf(src, sizeof src,
        "form CTX\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    let %s: vector = zeros(3)\n"
        "    %s[0] = 7.0\n"
        "    print %s[0]\n"
        "end\n"
        "end\n", w, w, w);
    snprintf(what, sizeof what, "`%s` indexed as a vector", w);
    must_compile_(what, src);
    if (run) must_print_(what, src, "7");
}

/* `let <word>: double = ...` with no later use, the declaration shape
 * on its own. */
static void arm_declaration_(const char *w)
{
    char src[2048], what[128];
    snprintf(src, sizeof src,
        "form CTX\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    let %s: double = 1.0\n"
        "    print 1.0\n"
        "end\n"
        "end\n", w);
    snprintf(what, sizeof what, "`%s` bound and not used", w);
    must_compile_(what, src);
}

/* Build a program, drive one step, and return the published range.
 * The step arms compare two artifacts that differ only in the value a
 * binding holds, so a parser that swallowed the binding would make the
 * two agree. */
static double step_range_(const char *src, const char *stem)
{
    char path[512], out[512], so[512];
    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    rl_write_file_(path, src);
    rl_compile_(path, out, WORK_DIR);
    snprintf(so, sizeof so, WORK_DIR "/%s.rlenv.so", stem);
    void *h = rl_dlopen_(so);
    RlSurface s;
    rl_resolve_surface_(h, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    ASSERT(s.step(env, act) == K26RL_OK);
    double o[32];
    ASSERT(s.obs(env, o) == K26RL_OK);
    double range = o[3];
    s.destroy(env);
    dlclose(h);
    return range;
}

/* ---- The identifier readings inside a step body ---------------------- */

/* The world an `on_step` arm is built on. The step body is where
 * `engage` is a construct, so it is where a greedy reading of the word
 * would take a binding away, and the world-prefix arms above cannot
 * see that at all. */
#define CTX_STEP_HEAD \
    "form CTXSTEP\n" \
    "fn double %s(double x)\n" \
    "    return x + 1.0\n" \
    "end\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft assembly=\"calibration_box.k26asm\"" \
    " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n" \
    "    episode\n" \
    "        control_dt 0.5\n" \
    "        horizon 8\n" \
    "    end\n" \
    "    action thrust box -1.0 1.0 default 0.0\n" \
    "    observe craft from earth mode=geometric as los\n"

#define CTX_STEP_TAIL \
    "    objective\n" \
    "        reward los_range\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* The identifier shapes, inside the step body. The bound value is
 * written into a body's velocity, so a run arm can tell a binding that
 * was read from one that was parsed away: the two artifacts differ only
 * in the number the binding holds. */
static void arm_step_shapes_(const char *w, int run)
{
    /* Three shapes, not four. A vector binding is refused on the
     * stepping path whatever it is called, because it owns heap
     * storage, so the index shape has no reading here to lose; the
     * world-prefix arm above is where it is covered. The call shape
     * takes its place, since an identifier followed by `(` is the
     * shape a greedy word would swallow. */
    static const char *const SHAPES[] = {
        "    let %s: double = 1.0\n"
        "    %s = %s + 2.0\n"
        "    craft.vel_x = craft.vel_x + %s\n",
        "    let %s: double = 3.0\n"
        "    craft.vel_x = craft.vel_x + %s\n",
        "    craft.vel_x = craft.vel_x + %s(2.0)\n",
        NULL
    };
    static const char *const NAMES[] = {
        "assigned as a name", "bound and read", "called as a function",
        NULL
    };
    for (int k = 0; SHAPES[k]; k++) {
        char body[1024], src[4096], what[160];
        snprintf(body, sizeof body, SHAPES[k], w, w, w, w);
        char head[2048];
        snprintf(head, sizeof head, CTX_STEP_HEAD, w);
        snprintf(src, sizeof src, "%s    on_step\n%s    end\n%s",
                 head, body, CTX_STEP_TAIL);
        snprintf(what, sizeof what, "`%s` %s inside on_step", w, NAMES[k]);
        must_compile_(what, src);
        if (!run) continue;
        /* The run half. The same program with the bound value at zero
         * must give a different range, which is what says the binding
         * reached the world rather than being parsed away. */
        char zero_body[1024], zero_src[4096];
        snprintf(zero_body, sizeof zero_body, SHAPES[k], w, w, w, w);
        char *p3 = strstr(zero_body, "3.0");
        char *p2 = strstr(zero_body, "2.0");
        if (p3) memcpy(p3, "0.0", 3);
        if (p2) memcpy(p2, "0.0", 3);
        snprintf(zero_src, sizeof zero_src,
                 "%s    on_step\n%s    end\n%s",
                 head, zero_body, CTX_STEP_TAIL);
        double live = step_range_(src, "ctxlive");
        double dead = step_range_(zero_src, "ctxdead");
        if (live == dead) {
            fprintf(stderr, "FAIL %s: the binding did not reach the "
                    "world (both ranges %.17g)\n", what, live);
            exit(1);
        }
        g_arms++;
        printf("  reaches the world: %s (%.9g against %.9g)\n", what,
               live, dead);
    }
}

/* ---- The block readings --------------------------------------------- */

/* Each word still opens its block where the block form is written.
 * One program carries all three, so a change that kept one reading and
 * lost another cannot pass by covering for itself. */
static void arm_blocks_(void)
{
    static const char *const SRC =
        "form CTXBLK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0\n"
        "    astro_body mark assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.02e6 vel_y=7535.0\n"
        "    astro_payload eye body=craft kind=detect_radar"
        " p_tx_w=2000.0 g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10"
        " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0"
        " noise_figure=2.0 snr_threshold=10.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n"
        "    sensor rf\n"
        "        noise normal 0.0 1.0\n"
        "    end\n"
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric through rf"
        " as trk\n"
        "        observe detect eye of mark as look\n"
        "        objective\n"
        "            reward 0.0 - pilot.trk_range\n"
        "        end\n"
        "    end\n"
        "    on_step\n"
        "        craft.vel_x = craft.vel_x + thrust\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("all four construct forms in one program", SRC);
}

/* A word that opens a block, and the same word bound as a name in the
 * same world. The two readings coexist, which is what "contextual"
 * means and what a reservation could not give. */
static void arm_both_readings_(void)
{
    static const char *const SRC =
        "form CTXBOTH\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0\n"
        "    astro_body mark assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.02e6 vel_y=7535.0\n"
        "    astro_payload eye body=craft kind=detect_radar"
        " p_tx_w=2000.0 g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10"
        " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0"
        " noise_figure=2.0 snr_threshold=10.0\n"
        "    let sensor: double = 3.0\n"
        "    sensor = sensor + 1.0\n"
        "    let astro_payload: double = 5.0\n"
        "    astro_payload = astro_payload + 1.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n"
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "        objective\n"
        "            reward 0.0 - pilot.trk_range\n"
        "        end\n"
        "    end\n"
        "    agent watcher\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "        observe detect eye of mark as look\n"
        "        objective\n"
        "            reward watcher.look_range\n"
        "        end\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("a bound name and a block of that word in one world",
                  SRC);
}

/* The two statements this phase adds, in one program that also binds
 * `engage`, `at` and `effect` as ordinary names, so the construct
 * readings and the identifier readings are shown to coexist. `at` is
 * also a body name here and is engaged by that name, which is the one
 * position where the word is read as a connective. */
static void arm_effector_forms_(void)
{
    static const char *const SRC =
        "form CTXEFF\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    astro_body at assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=2.0e3 vel_y=7546.0"
        " quat_w=1.0\n"
        "    astro_payload beam body=craft kind=laser"
        " primary_diam_m=1.5 wavelength_nm=1064.0 p_output_w=1.0e6"
        " m_squared=1.2 pointing_jitter_rad=1.0e-7"
        " rms_wavefront_m=5.0e-8 plasma_attn_k=1.0"
        " target_material=aluminum target_reflectivity=0.2\n"
        "    let engage: double = 3.0\n"
        "    engage = engage + 1.0\n"
        "    let effect: double = 5.0\n"
        "    effect = effect + 1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 8\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at at\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("the effector statements beside bindings of their own "
                  "words, with a body called `at` engaged by name", SRC);
}

/* A body genuinely called `effect` keeps its line-of-sight observe.
 * That is the shape the effector form is told apart from: `observe
 * effect <payload> as` has a payload name where this has `from`. */
static void arm_effect_as_body_(void)
{
    static const char *const SRC =
        "form CTXEFFBODY\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body effect assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 8\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe effect from earth mode=geometric as los\n"
        "    on_step\n"
        "        effect.vel_x = effect.vel_x + thrust\n"
        "    end\n"
        "    objective\n"
        "        reward los_range\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("a body called `effect` keeps its line-of-sight "
                  "observe", SRC);
}

/* The `capture_envelope` block opens where its form is written, and
 * the port observe's two words are read as the clause and the mark
 * they are. */
static void arm_capture_forms_(void)
{
    static const char *const SRC =
        "form CTXCAPENV\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    capture_envelope grasp_s\n"
        "        axial_rate 0.00 0.60\n"
        "        lateral_rate 0.50\n"
        "        pitchyaw_rate 30.0\n"
        "        roll_rate 30.0\n"
        "        lateral 0.30\n"
        "        pitchyaw 20.0\n"
        "        roll 180.0\n"
        "        diameter 700.0\n"
        "    end\n"
        "    astro_body one assembly=\"ctx_port_a.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    astro_body two assembly=\"ctx_port_b.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=30.0 vel_y=7546.0"
        " quat_w=0.0 quat_y=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 8\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe port grasp of one against face of two full as gr\n"
        "    objective\n"
        "        reward gr_axial\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("the capture_envelope block, the `against` clause and "
                  "the `full` mark in one program", SRC);
}

/* Bodies and sensors genuinely called `against` and `full`, reached
 * through the port observe's own trailing-clause scan. A greedy
 * reading of either word takes these names away; a fixture holding
 * only the clause form could not tell the two apart. */
static void arm_port_clause_names_(void)
{
    static const char *const HEAD =
        "form CTXPORTNAME\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    capture_envelope grasp_s\n"
        "        axial_rate 0.00 0.60\n"
        "        lateral_rate 0.50\n"
        "        pitchyaw_rate 30.0\n"
        "        roll_rate 30.0\n"
        "        lateral 0.30\n"
        "        pitchyaw 20.0\n"
        "        roll 180.0\n"
        "        diameter 700.0\n"
        "    end\n";
    /* One case per shape: the passive body named by each word, and a
     * sensor named by each word routing the same statement. */
    static const struct { const char *what, *body, *stmt; } CASE_[] = {
        { "a body called `against` carrying the active port",
          "two",
          "    observe port grasp of against against face of two as gr\n" },
        { "a body called `full` carrying the active port, with the mark",
          "two",
          "    observe port grasp of full against face of two full "
          "as gr\n" },
        { "a body called `against` carrying the passive port",
          "against",
          "    observe port grasp of one against face of against as gr\n" },
        { "a body called `full` carrying the passive port",
          "full",
          "    observe port grasp of one against face of full as gr\n" },
        { "a sensor called `against` on a marked port observe",
          "two",
          "    observe port grasp of one against face of two full "
          "through against as gr\n" },
        { "a sensor called `full` on a marked port observe",
          "two",
          "    observe port grasp of one against face of two full "
          "through full as gr\n" }
    };
    for (size_t i = 0; i < sizeof CASE_ / sizeof CASE_[0]; i++) {
        char src[4096];
        /* The craft carrying the active port takes the name the case
         * names, so the same fixture covers a body called `against`
         * or `full` on either side of the pairing. */
        const char *active = (i < 2) ? (i == 0 ? "against" : "full")
                                     : "one";
        snprintf(src, sizeof src,
            "%s"
            "    astro_body %s assembly=\"ctx_port_a.k26asm\""
            " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
            "    astro_body %s assembly=\"ctx_port_b.k26asm\""
            " parent=earth pos_x=7.0e6 pos_y=30.0 vel_y=7546.0"
            " quat_w=0.0 quat_y=1.0\n"
            "    sensor against\n"
            "        noise normal 0.0 0.001\n"
            "    end\n"
            "    sensor full\n"
            "        noise normal 0.0 0.001\n"
            "    end\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 8\n"
            "    end\n"
            "    action thrust box -1.0 1.0 default 0.0\n"
            "%s"
            "    objective\n"
            "        reward gr_axial\n"
            "    end\n"
            "end\n"
            "end\n", HEAD, active, CASE_[i].body, CASE_[i].stmt);
        must_compile_(CASE_[i].what, src);
    }
}

/* `against` and `full` are the port observe's own clause and mark, and
 * they are read there and nowhere else. These arms put both words at
 * the position a clause is read from, on forms that have no such
 * clause, and require the compiler to refuse them as it refuses any
 * other bare word there.
 *
 * They are the red half of the two arms above. Deleting the
 * form guard that makes the reading contextual leaves every
 * identifier arm passing, because those words are still bindable
 * names; what changes is that the two clauses start being claimed on
 * every observe form in the language, and only a fixture that writes
 * them where they do not belong can see it.
 */
static void arm_port_clause_not_greedy_(void)
{
    static const struct { const char *what, *stmt; } CASE_[] = {
        { "`against` at a clause position on a line-of-sight observe",
          "    observe craft from earth against face of mark as los\n" },
        { "`full` at a clause position on a line-of-sight observe",
          "    observe craft from earth full as los\n" },
        { "`against` at a clause position on an attitude observe",
          "    observe attitude of craft against face of mark as los\n" },
        { "`full` at a clause position on an attitude observe",
          "    observe attitude of craft full as los\n" }
    };
    for (size_t i = 0; i < sizeof CASE_ / sizeof CASE_[0]; i++) {
        char src[2048];
        snprintf(src, sizeof src,
            "form CTXNOTGREEDY\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
            "    astro_body mark assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=7.02e6 vel_y=7535.0 quat_w=1.0\n"
            "    episode\n"
            "        control_dt 0.5\n"
            "        horizon 8\n"
            "    end\n"
            "    action thrust box -1.0 1.0 default 0.0\n"
            "%s"
            "    objective\n"
            "        reward 1.0\n"
            "    end\n"
            "end\n"
            "end\n", CASE_[i].stmt);
        must_refuse_(CASE_[i].what, src,
                     "expected `name=value` after observer");
    }
}

int main(void)
{
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    /* Two craft with a docking port each, for the port form's arms.
     * The port stands proud of its own hull and the mating plate is
     * narrower than the hull it sits on, which is what the grammar
     * asks of a port that is meant to be met first. */
    rl_write_file_(WORK_DIR "/ctx_port_a.k26asm",
        "assembly ctx_port_a\n"
        "    frame x_to_port\n"
        "    provenance mass \"gate fixture, not a craft\" computed\n"
        "    component hull\n"
        "        mass 1000.0\n"
        "        at 0 0 0\n"
        "        collider box 0.5 0.4 0.4\n"
        "    end\n"
        "    port grasp\n"
        "        at 0.9 0.0 0.0\n"
        "        axis 1.0 0.0 0.0\n"
        "        roll_ref 0.0 1.0 0.0\n"
        "        capture grasp_s\n"
        "    end\n"
        "end\n");
    rl_write_file_(WORK_DIR "/ctx_port_b.k26asm",
        "assembly ctx_port_b\n"
        "    frame x_to_port\n"
        "    provenance mass \"gate fixture, not a craft\" computed\n"
        "    component hull\n"
        "        mass 2000.0\n"
        "        at 0 0 0\n"
        "        collider box 0.5 0.4 0.4\n"
        "    end\n"
        "    port face\n"
        "        at 0.9 0.0 0.0\n"
        "        axis 1.0 0.0 0.0\n"
        "        roll_ref 0.0 1.0 0.0\n"
        "        capture grasp_s\n"
        "    end\n"
        "end\n");
    /* The construct arms declare a payload, which needs a body that
     * carries a vehicle, so the assembly and its mesh travel with the
     * generated sources: an assembly path resolves against the
     * directory of the file that names it. */
    rl_run_or_die_("cp examples/assets/calibration_box.k26asm "
                   "examples/assets/calibration_box.k26mesh "
                   WORK_DIR "/");
    /* The check arms need only the compiler. The run arms link the
     * stack, so they are the part that stands down when it is absent,
     * and the gate says which half it ran. */
    int run = rl_libs_present_("test_rl_contextual_keywords");

    printf("test_rl_contextual_keywords: construct words as names\n");
    for (int i = 0; WORDS[i]; i++) {
        arm_assignment_(WORDS[i], run);
        arm_call_(WORDS[i], run);
        arm_index_(WORDS[i], run);
        arm_declaration_(WORDS[i]);
        arm_step_shapes_(WORDS[i], run);
    }
    arm_blocks_();
    arm_both_readings_();
    arm_effector_forms_();
    arm_effect_as_body_();
    arm_capture_forms_();
    arm_port_clause_names_();
    arm_port_clause_not_greedy_();

    printf("test_rl_contextual_keywords: %d arm(s) passed over %d "
           "word(s), run arms %s\n", g_arms, WORD_COUNT,
           run ? "included" : "stood down (stack archives absent)");
    return 0;
}
