/* test_rl_31_baseline.c: Grammar 3.1 baseline byte-identity.
 *
 * Every 3.1 example and integration fixture emits byte-identical C++
 * through today's kflc and through kflc built from the pre-RL base
 * commit (53a4452). A fixture that fails to emit (opaque types
 * provided by installed manifests) must fail identically on both.
 *
 * Exits 77 (the harness skip code) when the repository history does
 * not carry the base commit, so a skip is reported as a skip and can
 * never be mistaken for a pass; the always-runnable 3.1 pins live in
 * test_rl_31_pin.
 *
 * Pattern: run ./bin/kflc via system() on fixtures, assert on the
 * captured output (test_rl_grammar's harness style).
 */
#define _GNU_SOURCE
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#define WORK_DIR "/tmp/kflc_rl_31_baseline_test"

#define BASE_COMMIT "53a4452"

static int run_(const char *cmd)
{
    int rc = system(cmd);
    if (rc < 0) return -1;
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* Whole file into a NUL-terminated heap buffer; caller frees. */
static char *slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    long sz = ftell(f);
    ASSERT(sz >= 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char *buf = malloc((size_t)sz + 1);
    ASSERT(buf != NULL);
    if (sz > 0) ASSERT(fread(buf, 1, (size_t)sz, f) == (size_t)sz);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static int files_equal_(const char *a, const char *b)
{
    char *ca = slurp_(a), *cb = slurp_(b);
    int equal = (strcmp(ca, cb) == 0);
    free(ca);
    free(cb);
    return equal;
}

/* Emit a fixture through the given kflc binary; returns the exit
 * code, with stdout and stderr captured to files. */
static int emit_(const char *kflc, const char *kfl, const char *out,
                 const char *err)
{
    char cmd[1024];
    int n = snprintf(cmd, sizeof cmd, "%s --emit %s > %s 2> %s",
                     kflc, kfl, out, err);
    ASSERT((size_t)n < sizeof cmd);
    return run_(cmd);
}

int main(void)
{
    if (run_("git -C .. rev-parse --verify --quiet "
             BASE_COMMIT "^{commit} > /dev/null 2>&1") != 0) {
        printf("test_rl_31_baseline: SKIP (base commit " BASE_COMMIT
               " not available in this checkout's history)\n");
        return 77;
    }

    ASSERT(run_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR) == 0);
    ASSERT(run_("git -C .. archive " BASE_COMMIT " kflc | tar -x -C "
                WORK_DIR) == 0);
    ASSERT(run_("make -C " WORK_DIR "/kflc bin/kflc > "
                WORK_DIR "/oldbuild.log 2>&1") == 0);

    glob_t g;
    memset(&g, 0, sizeof g);
    ASSERT(glob("examples/*.kfl", 0, NULL, &g) == 0);
    ASSERT(glob("integration_tests/astro_w*.kfl", GLOB_APPEND, NULL, &g)
           == 0);
    int n_same = 0, n_diag = 0, n_newer = 0;
    for (size_t i = 0; i < g.gl_pathc; i++) {
        const char *kfl = g.gl_pathv[i];
        int rc_old = emit_(WORK_DIR "/kflc/bin/kflc", kfl,
                           WORK_DIR "/old.cc", WORK_DIR "/old.err");
        int rc_new = emit_("./bin/kflc", kfl,
                           WORK_DIR "/new.cc", WORK_DIR "/new.err");
        if (rc_old != 0 && rc_new == 0) {
            /* A programme the base grammar has no constructs for.
             * There is no baseline for it to be identical to: the
             * base compiler cannot emit it at all, which is what
             * makes it a later grammar's fixture rather than this
             * gate's. It is counted and named rather than skipped
             * quietly, and the count is pinned below, so a 3.1
             * fixture that started failing on the base would land
             * here and fail this gate rather than disappear from it. */
            fprintf(stderr, "%s: accepted by today's grammar and not by"
                    " the base\n", kfl);
            n_newer++;
            continue;
        }
        if (rc_old != rc_new) {
            fprintf(stderr, "%s: exit %d (base) vs %d (now)\n", kfl,
                    rc_old, rc_new);
            ASSERT(0);
        }
        if (rc_old == 0) {
            if (!files_equal_(WORK_DIR "/old.cc", WORK_DIR "/new.cc")) {
                fprintf(stderr, "%s: emitted C++ differs from the"
                        " base\n", kfl);
                ASSERT(0);
            }
            n_same++;
        } else {
            /* Both refuse; the diagnostics must match too. */
            if (!files_equal_(WORK_DIR "/old.err", WORK_DIR "/new.err")) {
                fprintf(stderr, "%s: emit diagnostics differ from the"
                        " base\n", kfl);
                ASSERT(0);
            }
            n_diag++;
        }
    }
    globfree(&g);
    ASSERT(n_same >= 12);
    /* One fixture in this tree is a later grammar's: the docking
     * benchmark. */
    ASSERT(n_newer == 1);
    printf("test_rl_31_baseline: %d fixture(s) byte-identical to the"
           " pre-RL base, %d refused identically on both, %d beyond the"
           " base grammar: OK\n", n_same, n_diag, n_newer);
    return 0;
}
