/* test_rl_grammar.c: Grammar 3.2 reinforcement learning front end.
 *
 * Positive gate: a full RL world (episode with control_dt / horizon /
 * terminated when / reset lines, box + discrete actions with
 * defaults, an on_step body, observe-as, an objective with reward and
 * terminal) parses and checks clean.
 *
 * Negative gates pin the diagnostics: step inside an RL world,
 * missing episode, missing control_dt, duplicate objective, duplicate
 * action name, reset with a bad state key, a distribution expression
 * outside its two valid positions, an unknown name in `terminated
 * when`, and a statement the on_step body rejects. The never-ending
 * episode warns but still checks clean.
 *
 * Emit-level gates: a top-level world `let`/`const` scalar is
 * readable in `terminated when` / `reward` / `terminal` (the emitter
 * captures it at create), and the nested-block and non-scalar
 * refusals carry precise diagnostics.
 *
 * Pattern: write a small .kfl fixture to a tmpfile, run
 *   ./bin/kflc --check <tmpfile>   (or --emit for the emit gates)
 * capture stderr + exit code, assert on the captured state.
 *
 * Wire: see kflc/Makefile RL_GRAMMAR_TEST + test target.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static void write_fixture_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(content, f);
    fclose(f);
}

/* Runs `kflc <mode> <fixture>`. Returns exit code; writes stderr
 * text to *err_out (caller frees). */
static int run_mode_(const char *mode, const char *fixture, char **err_out)
{
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "./bin/kflc %s %s >/dev/null 2>/tmp/kflc_rl_err.log",
             mode, fixture);
    int rc = system(cmd);
    FILE *f = fopen("/tmp/kflc_rl_err.log", "rb");
    if (!f) { *err_out = strdup(""); return WEXITSTATUS(rc); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) sz = 0;
    fseek(f, 0, SEEK_SET);
    *err_out = (char *)malloc((size_t)sz + 1);
    if (sz > 0) (void)!fread(*err_out, 1, (size_t)sz, f);
    (*err_out)[sz] = '\0';
    fclose(f);
    return WEXITSTATUS(rc);
}

static int n_pass = 0;

/* One fixture, one expectation under the given kflc mode:
 * `expect_rc` on exit, `expect_msg` present on stderr (NULL skips the
 * message check), `absent_msg` absent from stderr (NULL skips). */
static void expect_mode_(const char *mode, const char *tag,
                         const char *src, int expect_rc,
                         const char *expect_msg, const char *absent_msg)
{
    char path[128];
    snprintf(path, sizeof path, "/tmp/kflc_rl_%s.kfl", tag);
    write_fixture_(path, src);
    char *err = NULL;
    int rc = run_mode_(mode, path, &err);
    if (rc != expect_rc ||
        (expect_msg && strstr(err, expect_msg) == NULL) ||
        (absent_msg && strstr(err, absent_msg) != NULL))
    {
        fprintf(stderr, "case `%s`: rc=%d (expected %d)\nstderr:\n%s\n",
                tag, rc, expect_rc, err);
    }
    ASSERT(rc == expect_rc);
    if (expect_msg) ASSERT(strstr(err, expect_msg) != NULL);
    if (absent_msg) ASSERT(strstr(err, absent_msg) == NULL);
    free(err);
    unlink(path);
    n_pass++;
}

static void expect_(const char *tag, const char *src, int expect_rc,
                    const char *expect_msg, const char *absent_msg)
{
    expect_mode_("--check", tag, src, expect_rc, expect_msg, absent_msg);
}

int main(void)
{
    /* Positive: the full RL surface parses and checks clean. */
    expect_("full",
        "form RL_FULL\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=uniform(-1.0,1.0) pos_y=0.0 pos_z=0.0"
        " vel_x=0.0 vel_y=normal(0.0,0.1) vel_z=0.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 1000\n"
        "        terminated when episode.steps > 900\n"
        "        reset craft.pos_x uniform(-1.0, 1.0)\n"
        "        reset craft.vel_y normal(0.0, 0.1)\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    action gear discrete 3 default 0\n"
        "    on_step\n"
        "        let damping: double = 0.99\n"
        "    end\n"
        "    observe craft from earth mode=astrometric as track\n"
        "    objective\n"
        "        reward 0.0 - track_range + thrust\n"
        "        terminal track_range < 0.5\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* step / propagate belong to the episode machinery in RL worlds. */
    expect_("step",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    step 0.1\n"
        "end\n"
        "end\n",
        1, "the stepping belongs to the episode machinery", NULL);

    /* An RL construct without an episode block. */
    expect_("noepisode",
        "form P\n"
        "fn world w\n"
        "    action thrust box -1.0 1.0\n"
        "end\n"
        "end\n",
        1, "requires an `episode` block with `control_dt`", NULL);

    /* Episode without the required control_dt. */
    expect_("nodt",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        horizon 100\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "episode: missing required `control_dt <expr>`", NULL);

    /* Duplicate objective block. */
    expect_("dupobjective",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    objective\n"
        "        reward 1.0\n"
        "    end\n"
        "    objective\n"
        "        reward 2.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "duplicate `objective` block", NULL);

    /* Duplicate action name. */
    expect_("dupaction",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    action a box -1.0 1.0\n"
        "    action a discrete 4\n"
        "end\n"
        "end\n",
        1, "action `a`: duplicate action name", NULL);

    /* Reset with a state key outside the six scalar keys. */
    expect_("badkey",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "        reset craft.pos_w uniform(0.0, 1.0)\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "episode reset: unknown state key `pos_w`", NULL);

    /* Distribution expression outside its two valid positions. */
    expect_("distpos",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    let x: double = uniform(0.0, 1.0)\n"
        "end\n"
        "end\n",
        1, "only valid in astro_body attribute values and episode "
           "reset lines", NULL);

    /* Unknown name in the terminated-when expression. */
    expect_("unknownname",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when wibble > 1.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "terminated when: unknown name `wibble`", NULL);

    /* No horizon, no terminated-when: warns but checks clean. */
    expect_("neverends",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "    end\n"
        "end\n"
        "end\n",
        0, "the episode can never end", "error");

    /* on_step rejects stepping statements. */
    expect_("onstepstep",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    on_step\n"
        "        step 0.1\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "on_step: `step` is not allowed inside an on_step block", NULL);

    /* Grammar 3.1 compatibility: a program with its own function
     * named `uniform` (used in an astro_body value and in an
     * ordinary call) is not an RL program and stays clean. */
    expect_("userfn",
        "form P\n"
        "    fn double uniform(double a, double b)\n"
        "        return a + b\n"
        "    end\n"
        "    fn world w\n"
        "        astro_body earth gm=uniform(1.0,2.0)\n"
        "        step 0.1\n"
        "    end\n"
        "end\n",
        0, NULL, "error");

    /* World scalar bindings in the objective and termination scope:
     * a top-level world `let`/`const` is readable in `terminated
     * when`, `reward`, and `terminal`, checked here through the
     * emitter (the environment emitter captures the values at
     * create). */
    expect_mode_("--emit", "wscalpos",
        "form WSCAL_POS\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let target_range: double = 7.2e6\n"
        "    const bonus: double = 2.5\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "        terminated when track_range > target_range\n"
        "    end\n"
        "    action push box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward bonus - track_range / target_range\n"
        "        terminal bonus * 2.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* A binding declared inside a nested block is not in that scope;
     * the emitter says so precisely instead of failing on an unknown
     * identifier. */
    expect_mode_("--emit", "wscalnested",
        "form WSCAL_NEG1\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    if 1.0 > 0.0\n"
        "        let hidden: double = 1.0\n"
        "    end\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward hidden - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`hidden` is declared inside a nested block", NULL);

    /* A non-scalar world binding is not readable there either. */
    expect_mode_("--emit", "wscalvec",
        "form WSCAL_NEG2\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let gains: vector = [1.0, 2.0, 3.0]\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward gains - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`gains` is not a scalar", NULL);

    /* A negative horizon is a checker error, not a silent unbounded
     * episode; the never-ends warning must not fire beside it. */
    expect_("neghorizon",
        "form NEG_H\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon -3\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward 0.0 - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`horizon` must be non-negative", "can never end");

    /* A fractional horizon has no meaning as a step count. */
    expect_("frachorizon",
        "form FRAC_H\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 2.5\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward 0.0 - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "whole number of steps", NULL);

    /* An action name colliding with a derived observation component
     * is a KFL diagnostic, not a C++ redeclaration error. */
    expect_("nscollide",
        "form NS_COLLIDE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action track_range box -1.0 1.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with the `track_range` component", NULL);

    /* A channel name too long for the spec's 64-byte name entries is
     * refused at compile time, never silently truncated. The bound is
     * 64 less the longest derived suffix, `_range_rate`. */
    expect_("longchan",
        "form LONG_CHAN\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " channel_name_padded_out_to_be_conspicuously_longer_than_53_bytes\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "longer than 53 bytes", NULL);

    /* A world binding silently shadowed by an observation component
     * would read back the wrong value with no diagnostic; the
     * collision is an error instead. */
    expect_("wbindcollide",
        "form WB_COLLIDE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let track_range: double = 1.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with a world binding", NULL);

    /* Same rule between an action and a form argument. */
    expect_("argcollide",
        "form ARG_COLLIDE\n"
        "arg thrust default 9.0\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward thrust\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with form argument `thrust`", NULL);

    /* And between an action and a top-level world binding. */
    expect_("actbindcollide",
        "form AB_COLLIDE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let thrust: double = 1.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward thrust\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with a world binding", NULL);

    /* A binding inside a nested block was never readable in the
     * evaluators, so it may share an action's name freely. */
    expect_("nestedbindok",
        "form NB_OK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    if 1.0 > 0.0\n"
        "        let thrust: double = 1.0\n"
        "    end\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward thrust\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "collides");

    /* ---- Body state inside on_step ------------------------------- */

    /* The shape of every case below: one world, one action, one
     * observation channel, with STATE substituted into the on_step
     * body or elsewhere. */
#define STATE_WORLD(WHERE_PREFIX, WHERE_ON_STEP, WHERE_OBJECTIVE) \
        "form RL_STATE\n" \
        "fn world w\n" \
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
        "    astro_body craft gm=1.0 parent=earth" \
        " pos_x=7.0e6 vel_y=7546.0\n" \
        WHERE_PREFIX \
        "    episode\n" \
        "        control_dt 1.0\n" \
        "        horizon 4\n" \
        "    end\n" \
        "    action a box -1.0 1.0 default 0.0\n" \
        "    on_step\n" \
        WHERE_ON_STEP \
        "    end\n" \
        "    observe craft from earth mode=geometric as trk\n" \
        "    objective\n" \
        "        reward " WHERE_OBJECTIVE "\n" \
        "    end\n" \
        "end\n" \
        "end\n"

    /* Every state key is readable and assignable in the block. */
    expect_("state_all_keys",
        STATE_WORLD("",
            "        craft.pos_x = craft.pos_x + a\n"
            "        craft.pos_y = craft.pos_y + a\n"
            "        craft.pos_z = craft.pos_z + a\n"
            "        craft.vel_x = craft.vel_x + a\n"
            "        craft.vel_y = craft.vel_y + a\n"
            "        craft.vel_z = craft.vel_z + a\n", "0.0"),
        0, NULL, "error");

    /* Outside the block the dot is not a name: in the world prefix it
     * stays the lexical error it was before the block existed. */
    expect_("state_in_prefix",
        STATE_WORLD("    craft.vel_x = 1.0\n", "        let z: double = a\n",
                    "0.0"),
        1, "unexpected character", NULL);

    /* An objective is told where body state lives instead. */
    expect_("state_in_objective",
        STATE_WORLD("", "        let z: double = a\n", "craft.vel_x"),
        1, "only inside an on_step block", NULL);

    /* A termination predicate is refused the same way. */
    expect_("state_in_terminated",
        "form RL_STATE_T\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        terminated when craft.vel_x > 1.0\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "only inside an on_step block", NULL);

    /* An ordinary fn body is not a stepping block either. */
    expect_("state_in_fn",
        "form RL_STATE_F\n"
        "fn double helper()\n"
        "    craft.vel_x = 1.0\n"
        "    return 1.0\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "unexpected character", NULL);

    /* An unknown body, an unknown key, and `episode.steps` each name
     * what they found. */
    expect_("state_unknown_body",
        STATE_WORLD("", "        rocket.vel_x = a\n", "0.0"),
        1, "no astro_body named `rocket`", NULL);

    expect_("state_unknown_key",
        STATE_WORLD("", "        craft.spin_x = a\n", "0.0"),
        1, "is not a body state key", NULL);

    expect_("state_episode_steps",
        STATE_WORLD("", "        let z: double = episode.steps\n", "0.0"),
        1, "not in on_step", NULL);

    /* The assigned expression must stay re-evaluable, so a call to a
     * builtin that is not marked pure is refused, whether it is made
     * directly or through a fn. */
    expect_("state_impure_builtin",
        STATE_WORLD("",
            "        craft.vel_x = a + astro_world_body_count(world)\n",
            "0.0"),
        1, "is not a pure builtin", NULL);

    expect_("state_impure_through_fn",
        "form RL_STATE_P\n"
        "fn double reach(world w)\n"
        "    return astro_world_body_count(w)\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.vel_x = a + reach(world)\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "reaches", NULL);

    /* A pure builtin is admitted, which is what makes the refusals
     * above about purity rather than about calls. */
    expect_("state_pure_builtin",
        STATE_WORLD("", "        craft.vel_x = sqrt(a * a) + 1.0\n", "0.0"),
        0, NULL, "error");

    /* The stepping path performs no I/O, and a print one call away is
     * still I/O on the stepping path. */
    expect_("state_print_through_fn",
        "form RL_STATE_IO\n"
        "fn double chatty(double x)\n"
        "    print \"step\"\n"
        "    return x\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        let z: double = chatty(a)\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "performs no I/O", NULL);

    /* A body named `episode` would make `episode.steps` ambiguous. */
    expect_("state_body_named_episode",
        "form RL_STATE_E\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body episode gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe episode from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "the name is taken by `episode.steps`", NULL);

    /* The channel-name bound is the spec's 64-byte name entry less
     * the longest derived suffix, so 53 accepted and 54 refused. */
    expect_("chan_name_53",
        "form RL_CHAN53\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " chan_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuv\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "longer than");

    expect_("chan_name_54",
        "form RL_CHAN54\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " chan_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvw\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "longer than 53 bytes", NULL);

    /* The range-rate component is a readable channel like the four
     * beside it. */
    expect_("chan_range_rate",
        STATE_WORLD("", "        let z: double = a\n", "trk_range_rate"),
        0, NULL, "error");

#undef STATE_WORLD

    printf("test_rl_grammar: %d case(s) passed\n", n_pass);
    return 0;
}
