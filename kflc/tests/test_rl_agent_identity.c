/* test_rl_agent_identity.c: a single-agent program's spec is what it
 * was.
 *
 * The multi-agent surface is additive, and the whole of what that
 * means for a program that declares no `agent` block is this: it is
 * agent count 1, and its published environment spec is the same bytes
 * the compiler produced before the block existed. This gate builds
 * every reinforcement learning programme in the tree twice, once with
 * the compiler at the base commit and once with today's, drives each
 * artifact through the frozen surface, and compares the two spec blobs
 * byte for byte.
 *
 * The comparison is of the blobs and not of the emitted source. The
 * emitted source does differ: rewards are per agent now, so the
 * reward function is indexed and the reward buffer is env-major over
 * agents. What must not differ is the artifact's published contract,
 * which is what a consumer reads and what an episode file embeds.
 *
 * Exits 77 (the harness skip code) when the repository history does
 * not carry the base commit, or when the sibling stack archives are
 * absent, so a skip is reported as a skip and can never be mistaken
 * for a pass.
 *
 * Pattern: build the base compiler from a git archive of the base
 * commit, then rl_gate_util.h for everything else.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_agent_identity_test"

/* The commit the `agent` block landed on. */
#define BASE_COMMIT "932839a"

/* Every reinforcement learning programme in the tree. The count is
 * pinned below so that a programme dropping out of this list fails the
 * gate rather than shrinking it in silence. */
static const char *const RL_PROGRAMS[] = {
    "integration_tests/rl_pointing.kfl",
    "integration_tests/stationkeeping.kfl",
    "integration_tests/orbit_transfer.kfl",
    "examples/docking_benchmark.kfl",
    NULL
};

#define RL_PROGRAM_COUNT 4

static int run_(const char *cmd)
{
    int rc = system(cmd);
    if (rc < 0) return -1;
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* Compile through a named kflc rather than through ./bin/kflc, which
 * is the one thing rl_compile_ fixes. The flags are the gate helper's
 * with the optimisation level dropped: the compiler computes the
 * emitted constants itself, so the C++ optimiser cannot move a byte of
 * the spec, and eight builds of the docking programme at -O2 would buy
 * nothing for the wait. */
static void compile_with_(const char *kflc, const char *kfl_path,
                          const char *out_path)
{
    char cflags[4096];
    int n = snprintf(cflags, sizeof cflags,
        "-O0 -g0 -std=c++11 -Wno-format-truncation "
        "-ffp-contract=off -fexcess-precision=standard");
    for (int i = 0; RL_INCLUDE_DIRS_[i]; i++) {
        n += snprintf(cflags + n, sizeof cflags - (size_t)n, " -I%s",
                      RL_INCLUDE_DIRS_[i]);
    }
    ASSERT((size_t)n < sizeof cflags);

    char ldlibs[4096];
    n = 0;
    for (int i = 0; RL_LINK_LIBS_[i]; i++) {
        n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n, "%s%s",
                      i ? " " : "", RL_LINK_LIBS_[i]);
    }
    n += snprintf(ldlibs + n, sizeof ldlibs - (size_t)n, " -lgfortran -lm");
    ASSERT((size_t)n < sizeof ldlibs);

    char cmd[16384];
    n = snprintf(cmd, sizeof cmd,
        "KFLC_CFLAGS=\"%s\" KFLC_LDLIBS=\"%s\" %s %s -o %s "
        "> %s/kflc.log 2>&1",
        cflags, ldlibs, kflc, kfl_path, out_path, WORK_DIR);
    ASSERT((size_t)n < sizeof cmd);
    int rc = system(cmd);
    if (rc != 0) {
        char show[512];
        snprintf(show, sizeof show, "cat %s/kflc.log", WORK_DIR);
        (void)!system(show);
        fprintf(stderr, "kflc failed (rc=%d): %s %s\n", rc, kflc, kfl_path);
        exit(1);
    }
}

/* Open an artifact and copy out its spec blob. Caller frees. */
static uint8_t *spec_of_(const char *so_path, uint32_t *out_len,
                         uint32_t *out_agents)
{
    void *dso = rl_dlopen_(so_path);
    RlSurface s;
    rl_resolve_surface_(dso, &s);
    K26RlEnv *h = NULL;
    ASSERT(s.create(4242u, 3u, &h) == K26RL_OK);
    int32_t need = s.spec(h, NULL, 0);
    ASSERT(need > 0 && need < 1048576);
    uint8_t *blob = (uint8_t *)malloc((size_t)need);
    ASSERT(blob != NULL);
    ASSERT(s.spec(h, blob, (uint32_t)need) == need);
    RlSpecView v;
    rl_parse_spec_(blob, (uint32_t)need, &v);
    *out_agents = v.agent_count;
    *out_len = (uint32_t)need;
    s.destroy(h);
    dlclose(dso);
    return blob;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_agent_identity")) return 77;

    if (run_("git -C .. rev-parse --verify --quiet " BASE_COMMIT
             "^{commit} > /dev/null 2>&1") != 0) {
        printf("test_rl_agent_identity: SKIP (base commit " BASE_COMMIT
               " not available in this checkout's history)\n");
        return 77;
    }

    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    /* The whole tree at the base commit, because the compiler there
     * links three sibling libraries and they must be that commit's
     * too. */
    ASSERT(run_("git -C .. archive " BASE_COMMIT " | tar -x -C "
                WORK_DIR) == 0);
    ASSERT(run_("make -C " WORK_DIR "/libk26rng > "
                WORK_DIR "/oldbuild.log 2>&1") == 0);
    ASSERT(run_("make -C " WORK_DIR "/libk26sense >> "
                WORK_DIR "/oldbuild.log 2>&1") == 0);
    ASSERT(run_("make -C " WORK_DIR "/libk26rl >> "
                WORK_DIR "/oldbuild.log 2>&1") == 0);
    ASSERT(run_("make -C " WORK_DIR "/kflc bin/kflc >> "
                WORK_DIR "/oldbuild.log 2>&1") == 0);

    int n_checked = 0;
    for (int i = 0; RL_PROGRAMS[i]; i++) {
        const char *kfl = RL_PROGRAMS[i];
        char stem[64];
        snprintf(stem, sizeof stem, "p%d", i);

        char out_new[256], out_old[256], so_new[256], so_old[256];
        snprintf(out_new, sizeof out_new, "%s/%s_new", WORK_DIR, stem);
        snprintf(out_old, sizeof out_old, "%s/%s_old", WORK_DIR, stem);
        snprintf(so_new, sizeof so_new, "%s/%s_new.rlenv.so", WORK_DIR,
                 stem);
        snprintf(so_old, sizeof so_old, "%s/%s_old.rlenv.so", WORK_DIR,
                 stem);

        compile_with_("./bin/kflc", kfl, out_new);
        compile_with_(WORK_DIR "/kflc/bin/kflc", kfl, out_old);

        uint32_t len_new = 0, len_old = 0, ag_new = 0, ag_old = 0;
        uint8_t *sp_new = spec_of_(so_new, &len_new, &ag_new);
        uint8_t *sp_old = spec_of_(so_old, &len_old, &ag_old);

        if (ag_old != 1 || ag_new != 1) {
            fprintf(stderr, "FAIL %s: agent count %u (base) vs %u (now); "
                    "a programme with no agent block is agent count 1\n",
                    kfl, ag_old, ag_new);
            exit(1);
        }
        if (len_new != len_old) {
            fprintf(stderr, "FAIL %s: spec is %u bytes (base) vs %u "
                    "(now)\n", kfl, len_old, len_new);
            exit(1);
        }
        for (uint32_t b = 0; b < len_new; b++) {
            if (sp_new[b] == sp_old[b]) continue;
            fprintf(stderr, "FAIL %s: spec byte %u is 0x%02x (base) vs "
                    "0x%02x (now), of %u\n", kfl, b, sp_old[b], sp_new[b],
                    len_new);
            exit(1);
        }
        printf("  %s: %u spec bytes identical, agent count 1\n", kfl,
               len_new);
        free(sp_new);
        free(sp_old);
        n_checked++;
    }

    /* The fixture set does not shrink. A programme removed from the
     * list above would otherwise leave this gate green while measuring
     * less than it was written to measure. */
    ASSERT(n_checked == RL_PROGRAM_COUNT);
    printf("test_rl_agent_identity: %d programme(s) byte-identical "
           "against " BASE_COMMIT ": OK\n", n_checked);
    return 0;
}
