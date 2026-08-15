/* test_rl_purity.c: what a stepping path may reach.
 *
 * The `on_step` body runs once per external step of every
 * environment, so every expression in it has to be side-effect free:
 * an impure call there is an allocation, an input or output, or a
 * world mutation on a path whose contract is that it performs none of
 * those, and a recorded episode replayed from its recorded inputs
 * would not reproduce it.
 *
 * Registry gates (in process, against libkflc):
 *   1. A builtin registered without a purity declaration is impure,
 *      and one registered with an explicit declaration carries it.
 *      The default is the point: registering and forgetting to
 *      declare must not admit the entry to a stepping path.
 *   2. A name that was never registered is neither known nor pure, so
 *      a lookup miss cannot admit anything either.
 *   3. The static table's own answers: `sqrt` pure, `concat` impure
 *      because it allocates.
 *   4. Every world-mutating builtin the astro registrar publishes is
 *      impure, which is what makes the refusals below fall out of the
 *      general rule rather than out of a special case naming them.
 *
 * Position gates (subprocess, against the built kflc): one program
 * per position an expression can occupy inside the block, each
 * calling a known impure builtin, each requiring a refusal that names
 * the position, the line and the builtin:
 *
 *   assignment, `let` initialiser, expression statement, `if`
 *   condition, `while` condition, a statement inside a `for_each`
 *   body, an index expression, and a builtin call argument.
 *
 * Every one of those fixtures holds two statements of the position's
 * own form, one pure and one impure, and the assertion pins the line
 * of the impure one: a check that refused the block wholesale, or
 * that reported the first statement of the form rather than the
 * offending one, fails here rather than passing.
 *
 * Each position is gated in both polarities. The positive arms hold
 * the pure statement alone and require a clean compile, which is what
 * keeps the negative arms about purity rather than about calls.
 *
 * Two further arms: one indirection through a user `fn`, which the
 * sweep follows; and the leak itself, an `on_step` that creates a
 * world on every step of every environment and never closes it.
 *
 * Every position is asserted under `--check` and under `--emit`,
 * since a refusal that only the checker performs does not stop a
 * compile.
 *
 * Wire: see kflc/Makefile RL_PURITY_TEST + test target.
 */
#define _GNU_SOURCE
#include "kflc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

/* ---- Registry ------------------------------------------------------- */

/* The builtins that create, destroy, advance or otherwise mutate a
 * world. None may be reachable from a stepping path, and none is
 * named anywhere in the compiler's refusal path: they are refused
 * because the registry says they are impure. */
static const char *const WORLD_MUTATING_[] = {
    "astro_world_open",
    "astro_world_close",
    "astro_world_add_body",
    "astro_world_step",
    "astro_world_snapshot_load",
    "astro_world_snapshot_save",
    "astro_world_set_spin_hz",
    "astro_world_set_render_hz",
    "astro_world_set_observer_mode",
    "astro_world_set_mercurius",
    "astro_world_set_ephem",
    "astro_world_rng",
    NULL
};

static void registry_gates_(void)
{
    kflc_clear_builtins();

    /* A registration that declares nothing is impure. */
    ASSERT(kflc_register_builtin("gate_undeclared", "c_undeclared", 1) == 0);
    ASSERT(kflc_builtin_known("gate_undeclared") == 1);
    ASSERT(kflc_builtin_is_pure("gate_undeclared") == 0);

    /* A registration that declares carries what it declared, both
     * ways round, so the flag is read rather than assumed. */
    ASSERT(kflc_register_builtin_pure("gate_pure", "c_pure", 1, 1) == 0);
    ASSERT(kflc_register_builtin_pure("gate_impure", "c_impure", 1, 0) == 0);
    ASSERT(kflc_builtin_is_pure("gate_pure") == 1);
    ASSERT(kflc_builtin_is_pure("gate_impure") == 0);

    /* A name nobody registered is neither known nor pure. */
    ASSERT(kflc_builtin_known("gate_absent") == 0);
    ASSERT(kflc_builtin_is_pure("gate_absent") == 0);

    /* The static table's own answers. */
    ASSERT(kflc_builtin_is_pure("sqrt") == 1);
    ASSERT(kflc_builtin_is_pure("concat") == 0);

    kflc_clear_builtins();
    printf("a builtin that does not declare is impure: OK\n");
    n_pass++;

    kflc_register_astro_builtins();
    for (int i = 0; WORLD_MUTATING_[i]; i++) {
        if (!kflc_builtin_known(WORLD_MUTATING_[i]) ||
            kflc_builtin_is_pure(WORLD_MUTATING_[i]))
        {
            fprintf(stderr, "FAIL: %s known=%d pure=%d\n",
                    WORLD_MUTATING_[i],
                    kflc_builtin_known(WORLD_MUTATING_[i]),
                    kflc_builtin_is_pure(WORLD_MUTATING_[i]));
            exit(1);
        }
    }
    kflc_clear_builtins();
    printf("every world-mutating builtin is impure in the registry"
           " (%d checked): OK\n", (int)(sizeof WORLD_MUTATING_ /
                                        sizeof WORLD_MUTATING_[0]) - 1);
    n_pass++;
}

/* ---- Position fixtures ---------------------------------------------- */

static void write_fixture_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(content, f);
    fclose(f);
}

/* Runs `kflc <mode> <fixture>`; returns the exit code and writes the
 * captured stderr to *err_out, which the caller frees. */
static int run_mode_(const char *mode, const char *fixture, char **err_out)
{
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "./bin/kflc %s %s >/dev/null 2>/tmp/kflc_purity_err.log",
             mode, fixture);
    int rc = system(cmd);
    FILE *f = fopen("/tmp/kflc_purity_err.log", "rb");
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

/* The 1-based line of the first occurrence of `needle` in `src`, or
 * -1. Each negative fixture names its impure builtin exactly once, so
 * this is the line the diagnostic has to report. */
static int line_of_(const char *src, const char *needle)
{
    const char *hit = strstr(src, needle);
    if (!hit) return -1;
    int line = 1;
    for (const char *p = src; p < hit; p++) {
        if (*p == '\n') line++;
    }
    return line;
}

/* One program per escaping position. `position` is the phrase the
 * refusal must use, `builtin` the impure name it must report, and the
 * line is derived from the fixture rather than written down twice.
 * Asserted under both modes: a refusal only the checker performs does
 * not stop a compile. */
static void expect_refusal_(const char *tag, const char *src,
                            const char *position, const char *builtin)
{
    int line = line_of_(src, builtin);
    ASSERT(line > 0);

    char want[512];
    snprintf(want, sizeof want,
             ":%d: error: %s must be side-effect free, and `%s` is not a "
             "pure builtin", line, position, builtin);

    const char *modes[] = { "--check", "--emit", NULL };
    for (int i = 0; modes[i]; i++) {
        char path[160];
        snprintf(path, sizeof path, "/tmp/kflc_purity_%s_%d.kfl", tag, i);
        write_fixture_(path, src);
        char *err = NULL;
        int rc = run_mode_(modes[i], path, &err);
        if (rc != 1 || strstr(err, want) == NULL) {
            fprintf(stderr,
                    "case `%s` under %s: rc=%d (expected 1)\n"
                    "wanted: %s\nstderr:\n%s\n",
                    tag, modes[i], rc, want, err);
        }
        ASSERT(rc == 1);
        ASSERT(strstr(err, want) != NULL);
        free(err);
        unlink(path);
    }
    printf("%-14s refused, naming the position and line %d: OK\n", tag, line);
    n_pass++;
}

/* The same position with the impure statement removed compiles clean,
 * which is what makes the refusal above about purity and not about
 * calls. */
static void expect_clean_(const char *tag, const char *src)
{
    const char *modes[] = { "--check", "--emit", NULL };
    for (int i = 0; modes[i]; i++) {
        char path[160];
        snprintf(path, sizeof path, "/tmp/kflc_purity_ok_%s_%d.kfl", tag, i);
        write_fixture_(path, src);
        char *err = NULL;
        int rc = run_mode_(modes[i], path, &err);
        if (rc != 0 || strstr(err, "error") != NULL) {
            fprintf(stderr, "case `%s` under %s: rc=%d (expected 0)\n"
                            "stderr:\n%s\n", tag, modes[i], rc, err);
        }
        ASSERT(rc == 0);
        ASSERT(strstr(err, "error") == NULL);
        free(err);
        unlink(path);
    }
    printf("%-14s pure in the same position: OK\n", tag);
    n_pass++;
}

/* One world, one block, whatever the arm puts in it. */
#define WORLD(BODY) \
    "form RL_PURITY\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft gm=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.0\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 4\n" \
    "    end\n" \
    "    action a box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    BODY \
    "    end\n" \
    "    observe craft from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* Each pair below is the same body twice: the pure half alone, and
 * the pure half followed by the impure one. */
#define ASSIGN_OK \
    "        let z: double = 0.0\n" \
    "        z = sqrt(a * a)\n" \
    "        craft.vel_x = craft.vel_x + z\n"
#define ASSIGN_BAD \
    "        let z: double = 0.0\n" \
    "        z = sqrt(a * a)\n" \
    "        z = astro_world_body_count(world)\n" \
    "        craft.vel_x = craft.vel_x + z\n"

#define LET_OK \
    "        let y: double = sqrt(a * a)\n" \
    "        craft.vel_x = craft.vel_x + y\n"
#define LET_BAD \
    "        let y: double = sqrt(a * a)\n" \
    "        let z: double = astro_world_body_count(world)\n" \
    "        craft.vel_x = craft.vel_x + y + z\n"

#define EXPR_OK \
    "        sqrt(a * a)\n" \
    "        craft.vel_x = craft.vel_x + a\n"
#define EXPR_BAD \
    "        sqrt(a * a)\n" \
    "        astro_world_add_body(world, world)\n" \
    "        craft.vel_x = craft.vel_x + a\n"

#define IF_OK \
    "        if sqrt(a * a) > 0.0\n" \
    "            craft.vel_x = craft.vel_x + a\n" \
    "        end\n"
#define IF_BAD \
    "        if sqrt(a * a) > 0.0\n" \
    "            craft.vel_x = craft.vel_x + a\n" \
    "        end\n" \
    "        if astro_world_body_count(world) > 0.0\n" \
    "            craft.vel_x = craft.vel_x + a\n" \
    "        end\n"

#define WHILE_OK \
    "        while sqrt(a * a) > 1.0e9\n" \
    "            craft.vel_x = craft.vel_x + a\n" \
    "        end\n"
#define WHILE_BAD \
    "        while sqrt(a * a) > 1.0e9\n" \
    "            craft.vel_x = craft.vel_x + a\n" \
    "        end\n" \
    "        while astro_world_body_count(world) > 1.0e9\n" \
    "            craft.vel_x = craft.vel_x + a\n" \
    "        end\n"

#define FOREACH_OK \
    "        for_each b in world\n" \
    "            let p: double = sqrt(a * a)\n" \
    "            craft.vel_x = craft.vel_x + p\n" \
    "        end\n"
#define FOREACH_BAD \
    "        for_each b in world\n" \
    "            let p: double = sqrt(a * a)\n" \
    "            let q: double = astro_body_name(b)\n" \
    "            craft.vel_x = craft.vel_x + p + q\n" \
    "        end\n"

/* An index expression is evaluated as surely as the value beside it,
 * so it is a position of its own rather than part of the assignment. */
#define INDEX_OK \
    "        let xs: vector = zeros(2)\n" \
    "        xs[0] = sqrt(a * a)\n" \
    "        craft.vel_x = craft.vel_x + xs[0]\n"
#define INDEX_BAD \
    "        let xs: vector = zeros(2)\n" \
    "        xs[0] = sqrt(a * a)\n" \
    "        xs[astro_world_body_count(world)] = 1.0\n" \
    "        craft.vel_x = craft.vel_x + xs[0]\n"

#define ARG_OK \
    "        let y: double = max(1.0, sqrt(a * a))\n" \
    "        craft.vel_x = craft.vel_x + y\n"
#define ARG_BAD \
    "        let y: double = max(1.0, sqrt(a * a))\n" \
    "        let z: double = max(1.0, sqrt(astro_world_body_count(world)))\n" \
    "        craft.vel_x = craft.vel_x + y + z\n"

/* One indirection: the sweep follows a call into a user fn body, so
 * wrapping the impure call does not launder it. */
static const char *const THROUGH_FN =
    "form RL_PURITY_FN\n"
    "fn double reach(world ww)\n"
    "    return astro_world_body_count(ww)\n"
    "end\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 4\n"
    "    end\n"
    "    action a box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        let y: double = sqrt(a * a)\n"
    "        let z: double = reach(world)\n"
    "        craft.vel_x = craft.vel_x + y + z\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

int main(void)
{
    registry_gates_();

    /* Every position an expression can occupy in the block, each in
     * both polarities. */
    expect_refusal_("assignment", WORLD(ASSIGN_BAD),
                    "on_step: the assignment to `z`",
                    "astro_world_body_count");
    expect_clean_("assignment", WORLD(ASSIGN_OK));

    expect_refusal_("initialiser", WORLD(LET_BAD),
                    "on_step: the initialiser of `z`",
                    "astro_world_body_count");
    expect_clean_("initialiser", WORLD(LET_OK));

    /* This one is also the world-mutating arm: `astro_world_add_body`
     * is refused by the general rule, reported like any other impure
     * builtin, with nothing in the compiler naming it. */
    expect_refusal_("expr-stmt", WORLD(EXPR_BAD),
                    "on_step: an expression statement",
                    "astro_world_add_body");
    expect_clean_("expr-stmt", WORLD(EXPR_OK));

    expect_refusal_("if-cond", WORLD(IF_BAD),
                    "on_step: an `if` condition",
                    "astro_world_body_count");
    expect_clean_("if-cond", WORLD(IF_OK));

    expect_refusal_("while-cond", WORLD(WHILE_BAD),
                    "on_step: a `while` condition",
                    "astro_world_body_count");
    expect_clean_("while-cond", WORLD(WHILE_OK));

    expect_refusal_("for-each", WORLD(FOREACH_BAD),
                    "on_step: the initialiser of `q` in a `for_each` body",
                    "astro_body_name");
    expect_clean_("for-each", WORLD(FOREACH_OK));

    expect_refusal_("index-expr", WORLD(INDEX_BAD),
                    "on_step: an index expression",
                    "astro_world_body_count");
    expect_clean_("index-expr", WORLD(INDEX_OK));

    expect_refusal_("call-arg", WORLD(ARG_BAD),
                    "on_step: the initialiser of `z`",
                    "astro_world_body_count");
    expect_clean_("call-arg", WORLD(ARG_OK));

    /* The indirection arm carries its own message shape, so it is
     * asserted here rather than through expect_refusal_. */
    {
        char path[160];
        int line = line_of_(THROUGH_FN, "reach(world)");
        ASSERT(line > 0);
        char want[512];
        snprintf(want, sizeof want,
                 ":%d: error: on_step: the initialiser of `z` must be "
                 "side-effect free, and `fn reach` called here reaches "
                 "`astro_world_body_count`, which is not pure", line);
        const char *modes[] = { "--check", "--emit", NULL };
        for (int i = 0; modes[i]; i++) {
            snprintf(path, sizeof path, "/tmp/kflc_purity_fn_%d.kfl", i);
            write_fixture_(path, THROUGH_FN);
            char *err = NULL;
            int rc = run_mode_(modes[i], path, &err);
            if (rc != 1 || strstr(err, want) == NULL) {
                fprintf(stderr, "case `through-fn` under %s: rc=%d\n"
                                "wanted: %s\nstderr:\n%s\n",
                        modes[i], rc, want, err);
            }
            ASSERT(rc == 1);
            ASSERT(strstr(err, want) != NULL);
            free(err);
            unlink(path);
        }
        printf("through-fn     refused, naming the fn and the builtin:"
               " OK\n");
        n_pass++;
    }

    /* The leak this rule was written for: a block that creates a
     * world on every step of every environment and never closes it.
     * It passed a clean check before the rule covered the block. */
    expect_refusal_("world-leak",
        WORLD("        let leaked: world = astro_world_open(0, 0)\n"
              "        craft.vel_x = craft.vel_x + a\n"),
        "on_step: the initialiser of `leaked`",
        "astro_world_open");

    printf("test_rl_purity: %d gate(s) passed\n", n_pass);
    return 0;
}
