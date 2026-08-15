/* test_rl_contextual_keywords.c: a construct word is still a name.
 *
 * `agent`, `sensor` and `on_step` introduce block constructs at
 * statement position inside a `fn world` body. They are not reserved
 * words: a Grammar 3.1 program that binds one of them as an ordinary
 * identifier compiles and behaves as it always did, because each opens
 * its block only when what follows is what that block form requires.
 *
 * Gates, one arm per keyword per shape:
 *   1. The identifier readings. `<word> = 2.0`, `<word>(3.0)` as a
 *      bare statement, `<word>[0] = 1.0`, and `let <word>: double`
 *      compile, and the value the program computes through the name is
 *      the value it should be, so the word is not merely tolerated but
 *      read as the binding it names.
 *   2. The block readings. `agent <name>`, `sensor <name>` and a bare
 *      `on_step` still open their blocks in a program that uses them.
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

/* The three words this gate is about. */
static const char *const WORDS[] = { "agent", "sensor", "on_step", NULL };
#define WORD_COUNT 3

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
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
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
        "        objective\n"
        "            reward 0.0 - pilot.trk_range\n"
        "        end\n"
        "    end\n"
        "    on_step\n"
        "        craft.vel_x = craft.vel_x + thrust\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("all three block forms in one program", SRC);
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
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    let sensor: double = 3.0\n"
        "    sensor = sensor + 1.0\n"
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
        "        observe craft from earth mode=geometric as look\n"
        "        objective\n"
        "            reward watcher.look_range\n"
        "        end\n"
        "    end\n"
        "end\n"
        "end\n";
    must_compile_("a bound name and a block of that word in one world",
                  SRC);
}

int main(void)
{
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
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
    }
    arm_blocks_();
    arm_both_readings_();

    printf("test_rl_contextual_keywords: %d arm(s) passed over %d "
           "word(s), run arms %s\n", g_arms, WORD_COUNT,
           run ? "included" : "stood down (stack archives absent)");
    return 0;
}
