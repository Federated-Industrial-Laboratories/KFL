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
 * Pattern: write a small .kfl fixture to a tmpfile, run
 *   ./bin/kflc --check <tmpfile>
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

/* Runs `kflc --check <fixture>`. Returns exit code; writes stderr
 * text to *err_out (caller frees). */
static int run_check_(const char *fixture, char **err_out)
{
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "./bin/kflc --check %s 2>/tmp/kflc_rl_err.log", fixture);
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

/* One fixture, one expectation: `expect_rc` on exit, `expect_msg`
 * present on stderr (NULL skips the message check), `absent_msg`
 * absent from stderr (NULL skips). */
static void expect_(const char *tag, const char *src, int expect_rc,
                    const char *expect_msg, const char *absent_msg)
{
    char path[128];
    snprintf(path, sizeof path, "/tmp/kflc_rl_%s.kfl", tag);
    write_fixture_(path, src);
    char *err = NULL;
    int rc = run_check_(path, &err);
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

    printf("test_rl_grammar: %d case(s) passed\n", n_pass);
    return 0;
}
