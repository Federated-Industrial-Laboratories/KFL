/* test_reserved_keywords.c: reserved-keyword warning gate.
 *
 * `action`, `objective`, and `episode` are reserved for a future KFL
 * grammar version, alongside the concurrency set. A program using one
 * as an identifier must still compile (the reservation is
 * warn-before-enforce) and must get the deprecation-style warning, so
 * authors rename before the keyword lands and binds.
 *
 * Pattern: write a small .kfl fixture to a tmpfile, run
 *   ./bin/kflc --check <tmpfile>
 * capture stderr + exit code, assert on the captured state.
 *
 * Wire: see kflc/Makefile RESERVED_KW_TEST + test target.
 */
#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

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
             "./bin/kflc --check %s 2>/tmp/kflc_reserved_err.log", fixture);
    int rc = system(cmd);
    FILE *f = fopen("/tmp/kflc_reserved_err.log", "rb");
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

static void expect_warns_(const char *name)
{
    char src[256], path[128];
    snprintf(src, sizeof src,
             "form PROBE\n"
             "    fn double f()\n"
             "        let %s = 1.0\n"
             "        return %s\n"
             "    end\n"
             "end\n", name, name);
    snprintf(path, sizeof path, "/tmp/kflc_reserved_%s.kfl", name);
    write_fixture_(path, src);

    char *err = NULL;
    int rc = run_check_(path, &err);
    /* Warn, not enforce: the program still checks clean. */
    assert(rc == 0);
    assert(strstr(err, "reserved for a future KFL") != NULL);
    assert(strstr(err, name) != NULL);
    free(err);
    unlink(path);
    n_pass++;
}

int main(void)
{
    expect_warns_("action");
    expect_warns_("objective");
    expect_warns_("episode");
    /* The pre-existing set still warns. */
    expect_warns_("thread");

    /* An unreserved name draws no reservation warning. */
    write_fixture_("/tmp/kflc_reserved_clean.kfl",
                   "form PROBE\n"
                   "    fn double f()\n"
                   "        let motion = 1.0\n"
                   "        return motion\n"
                   "    end\n"
                   "end\n");
    char *err = NULL;
    int rc = run_check_("/tmp/kflc_reserved_clean.kfl", &err);
    assert(rc == 0);
    assert(strstr(err, "reserved for a future KFL") == NULL);
    free(err);
    unlink("/tmp/kflc_reserved_clean.kfl");
    n_pass++;

    printf("test_reserved_keywords: %d case(s) passed\n", n_pass);
    return 0;
}
