/* test_rl_purity.c: what a stepping path may reach.
 *
 * The `on_step` body runs once per control step of every environment,
 * so nothing it reaches may allocate, perform input or output, or
 * change the world: such a call would break the path's contract, and
 * a recorded episode replayed from its recorded inputs would not
 * reproduce it. Reaching is reaching whether the last hop is an
 * expression or a statement, and this gate holds both.
 *
 * Registry arms, in process against libkflc:
 *   - a builtin registered without a purity declaration is impure,
 *     and one registered with a declaration carries it either way;
 *   - a name nobody registered is neither known nor pure;
 *   - the static table's own answers, `sqrt` pure and `concat` impure;
 *   - every world-mutating builtin is impure, which is what makes the
 *     refusals below fall out of the general rule rather than out of a
 *     special case naming them;
 *   - the two read-only world queries are pure, which is what keeps
 *     the rule from being "no library call at all".
 *
 * Manifest arm: a manifest written for this run, loaded through
 * K26_KFL_BUILTIN_PATH, declaring one builtin `pure` and one not.
 * The declared one is admitted in the block and the undeclared one is
 * refused, so rule and manifest are exercised end to end rather than
 * through the in-process registrar alone.
 *
 * Position arms: one program per position an expression can occupy in
 * the block, each holding two statements of that position's own form,
 * one pure and one impure, so the assertion can pin the line of the
 * offending one. A check that refused the block wholesale, or that
 * named the wrong position or the wrong line, fails here. Each
 * position is also gated in the other polarity, with the impure
 * statement removed and a clean compile required, which is what keeps
 * these arms about purity rather than about calls.
 *
 * The dotted-assignment arm carries a claim of its own: the check runs
 * before the rewrite that turns `craft.vel_x = ...` into a call to an
 * emitted accessor, so the position named is the one the source wrote.
 * Moving the check after that rewrite degrades the message to "an
 * expression statement", and this arm is what notices.
 *
 * Two of the position arms, the index descent and the vector-literal
 * descent, use fixtures that a later pass would refuse anyway for an
 * unrelated reason. They are here because the descents are what stop
 * an impure call hiding inside an index or a literal, and the arm
 * pins which refusal fires, so a descent cannot be removed unnoticed.
 *
 * Statement arms: `astro_body`, `step`, `propagate`, `observe` and
 * `print` reached through a user function, which is the shape that
 * escaped an expression-only walk. One arm holds a two-function chain
 * and requires the diagnostic to name the whole route.
 *
 * Allocation arms: a vector or matrix binding, a function returning
 * one, and a call the compiler cannot classify, each refused because
 * the storage behind them is heap allocated and the stepping path
 * allocates nothing.
 *
 * Depth arm: a chain longer than the walk can follow is refused
 * rather than passed, because a limit that answers "clean" when it
 * runs out of room is a limit an author can step over.
 *
 * Every arm is asserted under `--check` and under `--emit`, since a
 * refusal that only the checker performs does not stop a compile, and
 * every positive arm is taken on through `kflc -o` to a real artifact
 * where the sibling archives are built, since an arm that certifies a
 * program `-o` cannot build certifies nothing.
 *
 * Wire: see kflc/Makefile RL_PURITY_TEST + test target.
 */
#define _GNU_SOURCE
#include "kflc.h"
/* For the stack's include and archive lists and the full-artifact
 * compile the positive arms take, so those lists live in one place. */
#include "rl_gate_util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>

static int n_pass = 0;
/* 1 when the sibling archives are built, so a positive arm can be
 * taken all the way to an artifact rather than stopping at --emit. */
static int libs_ok_ = 0;

/* ---- Registry -------------------------------------------------------- */

/* The builtins that create, destroy, advance or otherwise change a
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
    "astro_world_body_at",
    "astro_body_name",
    NULL
};

/* The read-only queries, declared pure by the runtime's manifest and
 * by the in-process registrar. Both read a const world, write
 * nothing, allocate nothing and return an int by value. */
static const char *const WORLD_READING_[] = {
    "astro_world_body_count",
    "astro_world_find_body",
    NULL
};

static void registry_gates_(void)
{
    kflc_clear_builtins();

    /* A registration that declares nothing is impure. */
    ASSERT(kflc_register_builtin("gate_undeclared", "c_undeclared", 1) == 0);
    ASSERT(kflc_builtin_known("gate_undeclared") == 1);
    ASSERT(kflc_builtin_is_pure("gate_undeclared") == 0);

    /* A registration that declares carries what it declared, both ways
     * round, so the flag is read rather than assumed. */
    ASSERT(kflc_register_builtin_pure("gate_pure", "c_pure", 1, 1) == 0);
    ASSERT(kflc_register_builtin_pure("gate_impure", "c_impure", 1, 0) == 0);
    ASSERT(kflc_builtin_is_pure("gate_pure") == 1);
    ASSERT(kflc_builtin_is_pure("gate_impure") == 0);

    /* A name nobody registered is neither known nor pure. */
    ASSERT(kflc_builtin_known("gate_absent") == 0);
    ASSERT(kflc_builtin_is_pure("gate_absent") == 0);

    /* The static table's own answers. `concat` is impure because its
     * result is a pointer into a per-callsite mutable static that the
     * next evaluation at that site overwrites. */
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
    printf("every world-mutating builtin is impure in the registry"
           " (%d checked): OK\n", (int)(sizeof WORLD_MUTATING_ /
                                        sizeof WORLD_MUTATING_[0]) - 1);
    n_pass++;

    for (int i = 0; WORLD_READING_[i]; i++) {
        if (!kflc_builtin_known(WORLD_READING_[i]) ||
            !kflc_builtin_is_pure(WORLD_READING_[i]))
        {
            fprintf(stderr, "FAIL: %s known=%d pure=%d\n",
                    WORLD_READING_[i],
                    kflc_builtin_known(WORLD_READING_[i]),
                    kflc_builtin_is_pure(WORLD_READING_[i]));
            exit(1);
        }
    }
    printf("the read-only world queries are pure (%d checked): OK\n",
           (int)(sizeof WORLD_READING_ / sizeof WORLD_READING_[0]) - 1);
    n_pass++;

    kflc_clear_builtins();
}

/* ---- Fixture plumbing ------------------------------------------------ */

static void write_fixture_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(content, f);
    fclose(f);
}

/* Runs `[env] kflc <mode> <fixture>`; returns the exit code and writes
 * the captured stderr to *err_out, which the caller frees. `env` is a
 * prefix such as a K26_KFL_BUILTIN_PATH setting, or NULL. */
static int run_mode_(const char *env, const char *mode, const char *fixture,
                     char **err_out)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "%s ./bin/kflc %s %s >/dev/null 2>/tmp/kflc_purity_err.log",
             env ? env : "", mode, fixture);
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

/* The 1-based line of the first occurrence of `needle`, or -1. */
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

/* Copies `tail` into `buf`, replacing each "@L" with `inner`. The
 * substitution keeps a fixture's own line numbers out of the arm's
 * expectation, so inserting a line into a fixture cannot silently
 * turn an assertion into a weaker one. */
static void expand_(char *buf, size_t n, const char *tail, int inner)
{
    size_t o = 0;
    for (const char *p = tail; *p && o + 1 < n; ) {
        if (p[0] == '@' && p[1] == 'L') {
            o += (size_t)snprintf(buf + o, n - o, "%d", inner);
            p += 2;
        } else {
            buf[o++] = *p++;
        }
    }
    buf[o < n ? o : n - 1] = '\0';
}

/* One arm. `pos_needle` locates the line the diagnostic must report;
 * `inner_needle` locates the line named inside a called function, or
 * is NULL when the arm's message names none; `tail` is everything the
 * diagnostic must say after "error: ", with "@L" standing for the
 * inner line. Asserted under both modes. */
static void expect_refusal_(const char *tag, const char *src,
                            const char *pos_needle,
                            const char *inner_needle,
                            const char *tail)
{
    int pos_line = line_of_(src, pos_needle);
    ASSERT(pos_line > 0);
    int inner = inner_needle ? line_of_(src, inner_needle) : 0;
    ASSERT(!inner_needle || inner > 0);

    char body[1024];
    expand_(body, sizeof body, tail, inner);

    char want[1200];
    snprintf(want, sizeof want, ":%d: error: %s", pos_line, body);

    const char *modes[] = { "--check", "--emit", NULL };
    for (int i = 0; modes[i]; i++) {
        char path[160];
        snprintf(path, sizeof path, "/tmp/kflc_purity_%s_%d.kfl", tag, i);
        write_fixture_(path, src);
        char *err = NULL;
        int rc = run_mode_(NULL, modes[i], path, &err);
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
    printf("%-22s refused at line %d, naming the position: OK\n",
           tag, pos_line);
    n_pass++;
}

/* A weaker arm for the cases whose message carries no fixture line,
 * used by the depth arm alone. */
static void expect_contains_(const char *tag, const char *src,
                             const char *needle)
{
    const char *modes[] = { "--check", "--emit", NULL };
    for (int i = 0; modes[i]; i++) {
        char path[160];
        snprintf(path, sizeof path, "/tmp/kflc_purity_%s_%d.kfl", tag, i);
        write_fixture_(path, src);
        char *err = NULL;
        int rc = run_mode_(NULL, modes[i], path, &err);
        if (rc != 1 || strstr(err, needle) == NULL) {
            fprintf(stderr, "case `%s` under %s: rc=%d\nwanted: %s\n"
                            "stderr:\n%s\n", tag, modes[i], rc, needle, err);
        }
        ASSERT(rc == 1);
        ASSERT(strstr(err, needle) != NULL);
        free(err);
        unlink(path);
    }
    printf("%-22s refused, naming the limit it hit: OK\n", tag);
    n_pass++;
}

/* The same position with the impure statement removed compiles clean,
 * and, where the sibling archives are built, all the way to an
 * artifact. Stopping at --emit is not enough: a positive arm that
 * passes --check and --emit while `kflc -o` fails certifies a program
 * nobody can build, which is how a construct that no artifact can
 * carry once became the only fixture for a whole position. */
static void expect_clean_(const char *tag, const char *src)
{
    const char *modes[] = { "--check", "--emit", NULL };
    for (int i = 0; modes[i]; i++) {
        char path[160];
        snprintf(path, sizeof path, "/tmp/kflc_purity_ok_%s_%d.kfl", tag, i);
        write_fixture_(path, src);
        char *err = NULL;
        int rc = run_mode_(NULL, modes[i], path, &err);
        if (rc != 0 || strstr(err, "error") != NULL) {
            fprintf(stderr, "case `%s` under %s: rc=%d (expected 0)\n"
                            "stderr:\n%s\n", tag, modes[i], rc, err);
        }
        ASSERT(rc == 0);
        ASSERT(strstr(err, "error") == NULL);
        free(err);
        unlink(path);
    }
    if (libs_ok_) {
        char path[160], out[160];
        snprintf(path, sizeof path, "/tmp/kflc_purity_art_%s.kfl", tag);
        snprintf(out, sizeof out, "/tmp/kflc_purity_art_%s", tag);
        write_fixture_(path, src);
        rl_compile_(path, out, "/tmp");
        if (!rl_file_exists_(out)) {
            fprintf(stderr, "case `%s`: kflc -o produced no artifact at %s\n",
                    tag, out);
        }
        ASSERT(rl_file_exists_(out));
        /* The companion shared object lands beside the executable and
         * is removed with it, so a later run cannot read a stale one. */
        char so[192];
        snprintf(so, sizeof so, "%s.rlenv.so", out);
        unlink(path);
        unlink(out);
        unlink(so);
        printf("%-22s pure in the same position, and builds: OK\n", tag);
    } else {
        printf("%-22s pure in the same position (artifact step skipped,"
               " archives not built): OK\n", tag);
    }
    n_pass++;
}

/* ---- Fixtures -------------------------------------------------------- */

/* One world, one block, whatever the arm puts in it. PRE goes in the
 * world prefix, FNS above the world, BODY inside `on_step`. */
#define WORLD_FULL(FNS, PRE, BODY, REWARD) \
    "form RL_PURITY\n" \
    FNS \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft gm=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.0\n" \
    PRE \
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
    "        reward " REWARD "\n" \
    "    end\n" \
    "end\n" \
    "end\n"

#define WORLD(BODY) WORLD_FULL("", "", BODY, "0.0")

/* The impure builtin the position arms reach. It reads a file, so why
 * it may not run on a stepping path is legible from its name. */
#define IMP "astro_world_snapshot_load(world)"

int main(void)
{
    libs_ok_ = rl_libs_present_("test_rl_purity");

    registry_gates_();

    /* ---- The positions -------------------------------------------- */

    expect_refusal_("assign-plain",
        WORLD("        let z: double = 0.0\n"
              "        z = sqrt(a * a)\n"
              "        z = " IMP "\n"
              "        craft.vel_x = craft.vel_x + z\n"),
        "z = " IMP, NULL,
        "on_step: the assignment to `z` must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("assign-plain",
        WORLD("        let z: double = 0.0\n"
              "        z = sqrt(a * a)\n"
              "        craft.vel_x = craft.vel_x + z\n"));

    /* The ordering claim: the check runs before the dotted name is
     * rewritten into an accessor call, so the position named is the
     * one the source wrote. After the rewrite this reads "an
     * expression statement" and this arm is what notices. */
    expect_refusal_("assign-dotted",
        WORLD("        craft.vel_y = craft.vel_y + sqrt(a * a)\n"
              "        craft.vel_x = craft.vel_x + " IMP "\n"),
        "craft.vel_x = craft.vel_x + " IMP, NULL,
        "on_step: the assignment to `craft.vel_x` must be side-effect "
        "free, and `astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("assign-dotted",
        WORLD("        craft.vel_y = craft.vel_y + sqrt(a * a)\n"
              "        craft.vel_x = craft.vel_x + a\n"));

    expect_refusal_("initialiser-let",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        let z: double = " IMP "\n"
              "        craft.vel_x = craft.vel_x + y + z\n"),
        "let z: double", NULL,
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("initialiser-let",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        let z: double = sqrt(y)\n"
              "        craft.vel_x = craft.vel_x + y + z\n"));

    expect_refusal_("initialiser-const",
        WORLD("        const p: double = sqrt(a * a)\n"
              "        const q: double = " IMP "\n"
              "        craft.vel_x = craft.vel_x + p + q\n"),
        "const q: double", NULL,
        "on_step: the initialiser of `q` must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("initialiser-const",
        WORLD("        const p: double = sqrt(a * a)\n"
              "        const q: double = 2.0\n"
              "        craft.vel_x = craft.vel_x + p + q\n"));

    expect_refusal_("expression-stmt",
        WORLD("        sqrt(a * a)\n"
              "        astro_world_add_body(world, world)\n"
              "        craft.vel_x = craft.vel_x + a\n"),
        "astro_world_add_body", NULL,
        "on_step: an expression statement must be side-effect free, and "
        "`astro_world_add_body` is not a pure builtin");
    expect_clean_("expression-stmt",
        WORLD("        sqrt(a * a)\n"
              "        craft.vel_x = craft.vel_x + a\n"));

    expect_refusal_("if-condition",
        WORLD("        if sqrt(a * a) > 0.0\n"
              "            craft.vel_x = craft.vel_x + a\n"
              "        end\n"
              "        if " IMP " > 0.0\n"
              "            craft.vel_x = craft.vel_x + a\n"
              "        end\n"),
        "if " IMP, NULL,
        "on_step: an `if` condition must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("if-condition",
        WORLD("        if sqrt(a * a) > 0.0\n"
              "            craft.vel_x = craft.vel_x + a\n"
              "        end\n"));

    expect_refusal_("while-condition",
        WORLD("        while sqrt(a * a) > 1.0e9\n"
              "            craft.vel_x = craft.vel_x + a\n"
              "        end\n"
              "        while " IMP " > 1.0e9\n"
              "            craft.vel_x = craft.vel_x + a\n"
              "        end\n"),
        "while " IMP, NULL,
        "on_step: a `while` condition must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("while-condition",
        WORLD("        while sqrt(a * a) > 1.0e9\n"
              "            craft.vel_x = craft.vel_x + a\n"
              "        end\n"));

    /* The else branch is walked with a name of its own, so an impure
     * call there is not reported as if it sat in the then branch. */
    expect_refusal_("else-body",
        WORLD("        if a > 0.0\n"
              "            craft.vel_x = craft.vel_x + sqrt(a * a)\n"
              "        else\n"
              "            craft.vel_x = " IMP "\n"
              "        end\n"),
        "craft.vel_x = " IMP, NULL,
        "on_step: the assignment to `craft.vel_x` in an `else` body must "
        "be side-effect free, and `astro_world_snapshot_load` is not a "
        "pure builtin");
    expect_clean_("else-body",
        WORLD("        if a > 0.0\n"
              "            craft.vel_x = craft.vel_x + sqrt(a * a)\n"
              "        else\n"
              "            craft.vel_x = craft.vel_x - a\n"
              "        end\n"));

    /* Two loops, so a check that named the first one regardless fails
     * here rather than passing. */
    expect_refusal_("for-each-body",
        WORLD("        for_each b in world\n"
              "            let p: double = sqrt(a * a)\n"
              "            craft.vel_x = craft.vel_x + p\n"
              "        end\n"
              "        for_each c in world\n"
              "            let q: double = " IMP "\n"
              "            craft.vel_x = craft.vel_x + q\n"
              "        end\n"),
        "let q: double", NULL,
        "on_step: the initialiser of `q` in a `for_each` body must be "
        "side-effect free, and `astro_world_snapshot_load` is not a pure "
        "builtin");
    expect_clean_("for-each-body",
        WORLD("        for_each b in world\n"
              "            let p: double = sqrt(a * a)\n"
              "            craft.vel_x = craft.vel_x + p\n"
              "        end\n"
              "        for_each c in world\n"
              "            let q: double = sqrt(a)\n"
              "            craft.vel_x = craft.vel_x + q\n"
              "        end\n"));

    expect_refusal_("call-argument",
        WORLD("        let y: double = max(1.0, sqrt(a * a))\n"
              "        let z: double = max(1.0, sqrt(" IMP "))\n"
              "        craft.vel_x = craft.vel_x + y + z\n"),
        "let z: double", NULL,
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");
    expect_clean_("call-argument",
        WORLD("        let y: double = max(1.0, sqrt(a * a))\n"
              "        let z: double = max(1.0, sqrt(y))\n"
              "        craft.vel_x = craft.vel_x + y + z\n"));

    expect_refusal_("returned-expr",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        return " IMP "\n"),
        "return " IMP, NULL,
        "on_step: a returned expression must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");

    /* The index descent, the index slot of an indexed assignment, and
     * the generalised lvalue slots. Each fixture would be refused by a
     * later pass for an unrelated reason; the arm pins which refusal
     * fires, so the descent cannot be removed unnoticed. */
    expect_refusal_("index-read",
        WORLD_FULL("", "    let xs: vector = zeros(4)\n",
              "        craft.vel_y = xs[0] + a\n"
              "        craft.vel_x = xs[" IMP "]\n", "0.0"),
        "craft.vel_x = xs[", NULL,
        "on_step: the assignment to `craft.vel_x` must be side-effect "
        "free, and `astro_world_snapshot_load` is not a pure builtin");

    expect_refusal_("index-slot",
        WORLD_FULL("", "    let xs: vector = zeros(4)\n",
              "        xs[0] = sqrt(a * a)\n"
              "        xs[" IMP "] = 1.0\n", "0.0"),
        "xs[" IMP "]", NULL,
        "on_step: an index expression must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");

    expect_refusal_("lvalue-slot",
        WORLD_FULL("", "    let mm: matrix = zeros(2, 2)\n",
              "        mm[0][0] = sqrt(a * a)\n"
              "        mm[1][1] = " IMP "\n", "0.0"),
        "mm[1][1]", NULL,
        "on_step: an assigned expression must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");

    /* The vector-literal descent, reached through a reward expression,
     * which is the one position a literal survives long enough to be
     * swept. */
    expect_refusal_("vector-literal",
        WORLD_FULL("", "", "        craft.vel_x = craft.vel_x + a\n",
              "[sqrt(trk_range), astro_world_snapshot_load(trk_range)]"),
        "reward [", NULL,
        "the `reward` expression must be side-effect free, and "
        "`astro_world_snapshot_load` is not a pure builtin");

    /* ---- Statements reached through a call ------------------------ */

    /* The shape an expression-only walk missed entirely: the statement
     * is legal where it is written and forbidden where it runs. Each
     * function holds an admissible statement beside the offending one,
     * so the line the message carries is pinned. */
#define VIA_FN(FNBODY, CALL) \
    WORLD_FULL("fn double reach(world ww)\n" FNBODY "    return 1.0\n" \
               "end\n", "", \
               "        let y: double = sqrt(a * a)\n" \
               "        let z: double = " CALL "\n" \
               "        craft.vel_x = craft.vel_x + y + z\n", "0.0")

    expect_refusal_("stmt-astro_body",
        VIA_FN("    let g: double = 1.0\n"
               "    astro_body extra gm=1.0 pos_x=1.0e7 vel_y=100.0\n",
               "reach(world)"),
        "let z: double", "astro_body extra",
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`fn reach` called here reaches `astro_body` at line @L, which is "
        "not allowed on the stepping path: it adds a body to the world, "
        "which allocates and can move every body already in it");

    expect_refusal_("stmt-step",
        VIA_FN("    let g: double = 1.0\n"
               "    step 1.0\n", "reach(world)"),
        "let z: double", "    step 1.0",
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`fn reach` called here reaches `step` at line @L, which is not "
        "allowed on the stepping path: it advances the world, and the "
        "episode machinery owns stepping in these programs");

    expect_refusal_("stmt-propagate",
        VIA_FN("    let g: double = 1.0\n"
               "    propagate craft for 1.0\n", "reach(world)"),
        "let z: double", "propagate craft",
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`fn reach` called here reaches `propagate` at line @L, which is "
        "not allowed on the stepping path: it advances a body, and the "
        "episode machinery owns stepping in these programs");

    expect_refusal_("stmt-observe",
        VIA_FN("    let g: double = 1.0\n"
               "    observe craft from earth mode=geometric\n",
               "reach(world)"),
        "let z: double", "    observe craft from earth",
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`fn reach` called here reaches `observe` at line @L, which is not "
        "allowed on the stepping path: it runs the world's observer "
        "pipeline and sets the world's observer mode");

    expect_refusal_("stmt-print-via-fn",
        VIA_FN("    let g: double = 1.0\n"
               "    print \"step\"\n", "reach(world)"),
        "let z: double", "    print \"step\"",
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`fn reach` called here reaches `print` at line @L, which is not "
        "allowed on the stepping path: the stepping path performs no I/O");

    /* Written straight into the block, judged by the same rule. */
    expect_refusal_("stmt-print-direct",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        print \"step\"\n"
              "        craft.vel_x = craft.vel_x + y\n"),
        "print \"step\"", NULL,
        "on_step: `print` is not allowed on the stepping path: the "
        "stepping path performs no I/O");

    /* Inside a nested block, where the diagnostic names the block. */
    expect_refusal_("stmt-print-nested",
        WORLD("        if a > 0.0\n"
              "            print \"step\"\n"
              "        end\n"
              "        craft.vel_x = craft.vel_x + a\n"),
        "print \"step\"", NULL,
        "on_step: `print` is not allowed on the stepping path in an `if` "
        "body: the stepping path performs no I/O");

    /* Two functions deep, so the diagnostic has a route to render
     * rather than a single name. The callee is declared first because
     * the emitter writes user functions in source order without
     * forward declarations, so a call to one declared later does not
     * build; that is a property of the emitter rather than of this
     * rule, and the clean arm below would fail on it. */
    expect_refusal_("stmt-chain-of-two",
        WORLD_FULL(
            "fn double inner(world ww)\n"
            "    let g: double = 1.0\n"
            "    astro_body extra gm=1.0 pos_x=1.0e7 vel_y=100.0\n"
            "    return 1.0\n"
            "end\n"
            "fn double outer(world ww)\n"
            "    return inner(ww)\n"
            "end\n", "",
            "        let y: double = sqrt(a * a)\n"
            "        let z: double = outer(world)\n"
            "        craft.vel_x = craft.vel_x + y + z\n", "0.0"),
        "let z: double", "astro_body extra",
        "on_step: the initialiser of `z` must be side-effect free, and "
        "the call chain `fn outer` -> `fn inner` from here reaches "
        "`astro_body` at line @L, which is not allowed on the stepping "
        "path: it adds a body to the world, which allocates and can move "
        "every body already in it");

    /* A function whose body is clean is followed and admitted, which
     * is what keeps the arms above about what was reached. */
    expect_clean_("stmt-chain-clean",
        WORLD_FULL(
            "fn double inner(double x)\n"
            "    let g: double = sqrt(x * x)\n"
            "    return g\n"
            "end\n"
            "fn double outer(double x)\n"
            "    return inner(x) + 1.0\n"
            "end\n", "",
            "        let z: double = outer(a)\n"
            "        craft.vel_x = craft.vel_x + z\n", "0.0"));

    /* ---- Allocation ----------------------------------------------- */

    expect_refusal_("alloc-vector-builder",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        let xs: vector = zeros(64)\n"
              "        craft.vel_x = craft.vel_x + y\n"),
        "let xs: vector", NULL,
        "on_step: `xs` is not allowed on the stepping path: a vector or "
        "matrix binding owns heap storage, and the stepping path "
        "allocates nothing");

    expect_refusal_("alloc-vector-literal",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        let xs: vector = [1.0, 2.0]\n"
              "        craft.vel_x = craft.vel_x + y\n"),
        "let xs: vector", NULL,
        "on_step: `xs` is not allowed on the stepping path: a vector or "
        "matrix binding owns heap storage, and the stepping path "
        "allocates nothing");

    expect_refusal_("alloc-matrix",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        let mm: matrix = zeros(2, 2)\n"
              "        craft.vel_x = craft.vel_x + y\n"),
        "let mm: matrix", NULL,
        "on_step: `mm` is not allowed on the stepping path: a vector or "
        "matrix binding owns heap storage, and the stepping path "
        "allocates nothing");

    /* A function that returns one allocates in the callee, so the call
     * is refused where it is made. */
    expect_refusal_("alloc-fn-return",
        WORLD_FULL("fn vector build(double x)\n"
                   "    let xs: vector = zeros(4)\n"
                   "    return xs\n"
                   "end\n", "",
                   "        let y: double = sqrt(a * a)\n"
                   "        let z: double = build(a)\n"
                   "        craft.vel_x = craft.vel_x + y\n", "0.0"),
        "let z: double", NULL,
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`build` is not allowed on the stepping path: it returns a vector "
        "or matrix, which owns heap storage, and the stepping path "
        "allocates nothing");

    /* A call the compiler can classify as neither a builtin nor a user
     * function is a form it lowers itself, and some of those allocate.
     * Refused by name rather than passed over. */
    expect_refusal_("alloc-unknown-call",
        WORLD("        let y: double = sqrt(a * a)\n"
              "        let z: double = linspace(0.0, 1.0, 8)\n"
              "        craft.vel_x = craft.vel_x + y + z\n"),
        "let z: double", NULL,
        "on_step: the initialiser of `z` must be side-effect free, and "
        "`linspace` is not allowed on the stepping path: the compiler "
        "cannot show that this call is free of effects, and the forms it "
        "lowers itself include ones that allocate");

    /* ---- Depth ----------------------------------------------------- */

    /* A chain longer than the walk follows. Before the walk failed
     * closed this compiled at exit 0 with the impure call live at the
     * end of it. */
    {
        static char src[65536];
        static char fns[49152];
        size_t o = 0;
        const int n_fns = 70;
        for (int i = 0; i < n_fns; i++) {
            if (i + 1 < n_fns) {
                o += (size_t)snprintf(fns + o, sizeof fns - o,
                    "fn double f%d(world ww)\n    return f%d(ww)\nend\n",
                    i, i + 1);
            } else {
                o += (size_t)snprintf(fns + o, sizeof fns - o,
                    "fn double f%d(world ww)\n    return "
                    "astro_world_snapshot_load(ww)\nend\n", i);
            }
        }
        ASSERT(o < sizeof fns);
        int k = snprintf(src, sizeof src, WORLD_FULL("%s", "",
            "        let z: double = f0(world)\n"
            "        craft.vel_x = craft.vel_x + z\n", "0.0"), fns);
        ASSERT(k > 0 && (size_t)k < sizeof src);
        expect_contains_("depth-limit", src,
            "the compiler follows at most 64 nested user functions");
    }

    /* ---- The manifest path ---------------------------------------- */

    /* Rule 2 end to end: a library declares purity in its manifest,
     * the loader carries the declaration into the registry, and the
     * block admits or refuses on the strength of it. Every other arm
     * here draws its builtin from the in-process registrar, so without
     * this one the manifest half of the rule is never compiled. */
    {
        char dir[] = "/tmp/kflc_purity_mf_XXXXXX";
        ASSERT(mkdtemp(dir) != NULL);
        char mf[512];
        snprintf(mf, sizeof mf, "%s/gate.kflbi", dir);
        write_fixture_(mf,
            "schema 1\n"
            "library libgate\n"
            "version 0.1.0\n"
            "builtin gate_reads   c_gate_reads   1 pure\n"
            "builtin gate_writes  c_gate_writes  1\n");

        char env[640];
        snprintf(env, sizeof env, "K26_KFL_BUILTIN_PATH=%s", dir);

        const char *refused = WORLD(
            "        let y: double = sqrt(a * a)\n"
            "        let z: double = gate_writes(a)\n"
            "        craft.vel_x = craft.vel_x + y + z\n");
        const char *admitted = WORLD(
            "        let y: double = sqrt(a * a)\n"
            "        let z: double = gate_reads(a)\n"
            "        craft.vel_x = craft.vel_x + y + z\n");

        char want[512];
        snprintf(want, sizeof want,
                 ":%d: error: on_step: the initialiser of `z` must be "
                 "side-effect free, and `gate_writes` is not a pure builtin",
                 line_of_(refused, "gate_writes(a)"));

        const char *modes[] = { "--check", "--emit", NULL };
        for (int i = 0; modes[i]; i++) {
            char path[160];
            char *err = NULL;

            snprintf(path, sizeof path, "/tmp/kflc_purity_mfbad_%d.kfl", i);
            write_fixture_(path, refused);
            int rc = run_mode_(env, modes[i], path, &err);
            if (rc != 1 || strstr(err, want) == NULL) {
                fprintf(stderr, "manifest refusal under %s: rc=%d\n"
                                "wanted: %s\nstderr:\n%s\n",
                        modes[i], rc, want, err);
            }
            ASSERT(rc == 1);
            ASSERT(strstr(err, want) != NULL);
            free(err);
            unlink(path);

            snprintf(path, sizeof path, "/tmp/kflc_purity_mfok_%d.kfl", i);
            write_fixture_(path, admitted);
            err = NULL;
            rc = run_mode_(env, modes[i], path, &err);
            if (rc != 0 || strstr(err, "error") != NULL) {
                fprintf(stderr, "manifest admission under %s: rc=%d\n"
                                "stderr:\n%s\n", modes[i], rc, err);
            }
            ASSERT(rc == 0);
            ASSERT(strstr(err, "error") == NULL);
            free(err);
            unlink(path);
        }
        unlink(mf);
        rmdir(dir);
        printf("manifest-declared      pure admitted, undeclared refused:"
               " OK\n");
        n_pass++;
    }

    /* ---- The reads the rule must not cost ------------------------- */

    /* The declared-pure query is admitted in the block, which is what
     * keeps this a purity rule rather than a ban on library calls. */
    expect_clean_("declared-pure-read",
        WORLD("        let n: double = astro_world_body_count(world)\n"
              "        craft.vel_x = craft.vel_x + a * n\n"));

    /* ---- The leak this rule was written for ----------------------- */

    expect_refusal_("world-leak",
        WORLD("        let leaked: world = astro_world_open(0, 0)\n"
              "        craft.vel_x = craft.vel_x + a\n"),
        "astro_world_open", NULL,
        "on_step: the initialiser of `leaked` must be side-effect free, "
        "and `astro_world_open` is not a pure builtin");

    printf("test_rl_purity: %d gate(s) passed\n", n_pass);
    return 0;
}
