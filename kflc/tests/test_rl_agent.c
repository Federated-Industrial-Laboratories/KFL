/* test_rl_agent.c: the `agent` block, the multi-agent grammar surface.
 *
 * Gates:
 *   1. Slice partition. A two-agent program publishes one observation
 *      slice and one action slice per agent, contiguous from zero,
 *      summing to the totals and overlapping nowhere. The fixture's
 *      two agents have *different* channel counts on both vectors,
 *      which is the whole of this gate: two agents of equal counts
 *      cannot tell a correct partition from one that has swapped the
 *      two agents' offsets.
 *   2. Published names. With more than one agent a channel publishes
 *      as `<agent>.<channel>`; with one it publishes the name it was
 *      declared with, which is what makes a single-agent program's
 *      spec identical to the one it had before agents existed. The
 *      one-agent case is held by a program that declares exactly one
 *      `agent` block as well as by one that declares none, since
 *      publication turns on the count rather than on a block being
 *      absent and those two conditions come apart only there.
 *   2a. An agent with no `objective` publishes an all-zero reward
 *      stream for its own agent, beside one that declares a reward.
 *   2b. The scope an objective expression is emitted into holds the
 *      channels the expression reads and nothing else, each declared
 *      at the index the channel occupies in the observation vector.
 *   3. The agent-name bound. A name of exactly 31 bytes compiles and
 *      one of 32 is refused naming the bound; a combination that
 *      overflows the 96-byte entry is refused naming both parts and
 *      the arithmetic rather than only the total.
 *   4. Mixing and collision refusals. A world-level `action`,
 *      `objective` or `as`-bound `observe` beside any agent block is
 *      refused naming both sites; an agent name equal to an
 *      `astro_body` name is refused naming both declarations.
 *   5. Unqualified resolution inside `on_step`. A name unique across
 *      the blocks resolves; a name two blocks declare is refused
 *      naming both declaration sites and the qualified form; the
 *      qualified form itself resolves, and it commands the agent it
 *      names, which is read back off the bodies rather than inferred.
 *   6. Name scoping. A channel name repeats across blocks and not
 *      within one.
 *   7. Block contents. At most one `objective` per block; a block with
 *      no `action` is legal (observation only); a block with neither
 *      an action nor an objective is refused; a block holding
 *      anything else is refused; two blocks of one name are refused.
 *   8. Cross-agent reads. An `objective` reading another agent's
 *      channel by qualified name reads that agent's channel and not
 *      its own, driven against a fixture whose two agents' channels
 *      differ in value at every step, and the zero-sum pair the
 *      clause exists for holds exactly.
 *   9. The episode's own scope. `terminated when` sits in no block, so
 *      it names the agent it reads.
 *  10. Batch mode. A multi-agent program is batch runnable with every
 *      action channel at its declared default, and the episode file it
 *      writes carries the agent count and one reward stream per agent.
 *
 * Pattern: rl_gate_util.h. Refusal arms drive ./bin/kflc --check and
 * assert on the captured diagnostic; behaviour arms compile the
 * artifact and drive the frozen surface.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_agent_test"

/* Arms run, so the suite line carries a count rather than a bare
 * word: a gate that stopped running half of what it holds would
 * otherwise print the same line it prints when it runs all of it. */
static int g_arms;

/* ---- Sources -------------------------------------------------------- */

/* Two agents of deliberately different channel counts: alpha owns two
 * actions and one observation channel (five components), beta one
 * action and two observation channels (ten). The two craft sit at
 * different radii and never meet, so alpha's range and beta's range
 * differ at every step and a cross-agent read that resolved to the
 * wrong agent cannot look like one that resolved right. */
static const char *const TWO_KFL =
    "form TWO\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body alpha_craft gm=1.0 parent=earth pos_x=7.0e6"
    " vel_y=7546.0\n"
    "    astro_body beta_craft gm=1.0 parent=earth pos_x=1.1e7"
    " vel_y=6020.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 6\n"
    "    end\n"
    "    agent alpha\n"
    "        action thrust box -1.0 1.0 default 0.25\n"
    "        action yaw box -1.0 1.0 default 0.5\n"
    "        observe alpha_craft from earth mode=geometric as trk\n"
    "        objective\n"
    "            reward beta.trk_range - alpha.trk_range\n"
    "        end\n"
    "    end\n"
    "    agent beta\n"
    "        action thrust box -1.0 1.0 default 0.75\n"
    "        observe beta_craft from earth mode=geometric as trk\n"
    "        observe beta_craft from alpha_craft mode=geometric as rel\n"
    "        objective\n"
    "            reward 0.0 - (beta.trk_range - alpha.trk_range)\n"
    "        end\n"
    "    end\n"
    "    on_step\n"
    "        alpha_craft.vel_x = alpha_craft.vel_x + alpha.thrust\n"
    "        beta_craft.vel_x = beta_craft.vel_x + beta.thrust\n"
    "        alpha_craft.vel_z = alpha_craft.vel_z + yaw\n"
    "    end\n"
    "end\n"
    "end\n";

/* The same world with no agent block: one agent, whole-vector
 * slices, unqualified names. */
static const char *const ONE_KFL =
    "form ONE\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 4\n"
    "    end\n"
    "    action thrust box -1.0 1.0 default 0.0\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0 - trk_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* An observation-only agent beside one that acts: the act slice of
 * agent 1 is legally empty. */
static const char *const OBS_ONLY_KFL =
    "form OBSONLY\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0\n"
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
    "        observe craft from earth mode=geometric as look\n"
    "        objective\n"
    "            reward watcher.look_range\n"
    "        end\n"
    "    end\n"
    "end\n"
    "end\n";

/* Each agent's qualified action written straight onto its own craft's
 * out-of-plane velocity, which nothing else in the world touches. The
 * two agents drive different values, so a qualified read that resolved
 * to the wrong agent puts the wrong number on a body and the body says
 * so. */
static const char *const APPLY_KFL =
    "form APPLY\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body alpha_craft gm=1.0 parent=earth pos_x=7.0e6"
    " vel_y=7546.0\n"
    "    astro_body beta_craft gm=1.0 parent=earth pos_x=1.1e7"
    " vel_y=6020.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 8\n"
    "    end\n"
    "    agent alpha\n"
    "        action push box -1000.0 1000.0 default 0.0\n"
    "        observe alpha_craft from earth mode=geometric as trk\n"
    "    end\n"
    "    agent beta\n"
    "        action push box -1000.0 1000.0 default 0.0\n"
    "        observe beta_craft from earth mode=geometric as trk\n"
    "    end\n"
    "    on_step\n"
    "        alpha_craft.vel_z = alpha.push\n"
    "        beta_craft.vel_z = beta.push\n"
    "    end\n"
    "end\n"
    "end\n";

/* Exactly one `agent` block. Publication is conditioned on the agent
 * count and not on the absence of a block, and this is the only shape
 * where those two conditions come apart: one block is agent count 1,
 * so its channels publish bare and its whole-vector slices are the
 * ones a program with no block would have had. */
static const char *const ONE_BLOCK_KFL =
    "form ONEBLOCK\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 4\n"
    "    end\n"
    "    agent pilot\n"
    "        action thrust box -1.0 1.0 default 0.0\n"
    "        observe craft from earth mode=geometric as trk\n"
    "        objective\n"
    "            reward 0.0 - trk_range\n"
    "        end\n"
    "    end\n"
    "end\n"
    "end\n";

/* One agent with an objective beside one without. The second agent
 * publishes an all-zero reward stream, which is the rule a program
 * with no objective already follows; a defect handing it its
 * neighbour's reward would be invisible to every fixture in which
 * both agents declare one. The first agent's reward is a constant, so
 * the two streams are told apart by value and not by coincidence. */
static const char *const HALF_OBJECTIVE_KFL =
    "form HALFOBJ\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7546.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 4\n"
    "    end\n"
    "    agent paid\n"
    "        action thrust box -1.0 1.0 default 0.0\n"
    "        observe craft from earth mode=geometric as trk\n"
    "        objective\n"
    "            reward 5.0\n"
    "        end\n"
    "    end\n"
    "    agent unpaid\n"
    "        action brake box -1.0 1.0 default 0.0\n"
    "        observe craft from earth mode=geometric as look\n"
    "    end\n"
    "end\n"
    "end\n";

/* ---- Harness -------------------------------------------------------- */

static char *slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    long sz = ftell(f);
    ASSERT(sz >= 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char *buf = (char *)malloc((size_t)sz + 1);
    ASSERT(buf != NULL);
    if (sz > 0) ASSERT(fread(buf, 1, (size_t)sz, f) == (size_t)sz);
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

/* Run `kflc --check` over a source written to WORK_DIR, returning the
 * exit status with the diagnostics in *out_log (caller frees). */
static int check_src_(const char *src, char **out_log)
{
    rl_write_file_(WORK_DIR "/case.kfl", src);
    int rc = system("./bin/kflc --check " WORK_DIR "/case.kfl > "
                    WORK_DIR "/case.log 2>&1");
    *out_log = slurp_(WORK_DIR "/case.log");
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

/* A refusal arm: the compiler must refuse, and its diagnostic must
 * carry every fragment the rule promises. A refusal for the wrong
 * reason is not a pass, which is why the fragments are asserted one
 * at a time and the whole log is printed on failure. */
static void must_refuse_(const char *what, const char *src,
                         const char *const *fragments)
{
    char *log = NULL;
    int rc = check_src_(src, &log);
    if (rc == 0) {
        fprintf(stderr, "FAIL %s: accepted, expected a refusal\n", what);
        exit(1);
    }
    for (int i = 0; fragments[i]; i++) {
        if (strstr(log, fragments[i]) == NULL) {
            fprintf(stderr, "FAIL %s: diagnostic lacks \"%s\"\n---\n%s---\n",
                    what, fragments[i], log);
            exit(1);
        }
    }
    free(log);
    g_arms++;
    printf("  refused: %s\n", what);
}

static void must_accept_(const char *what, const char *src)
{
    char *log = NULL;
    int rc = check_src_(src, &log);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s: refused, expected acceptance\n---\n%s---\n",
                what, log);
        exit(1);
    }
    free(log);
    g_arms++;
    printf("  accepted: %s\n", what);
}

/* Compile a source string to an artifact and read its spec. */
static void *build_and_open_(const char *src, const char *stem,
                             RlSpecView *view, RlSurface *s)
{
    char kfl[256], out[256], so[256];
    snprintf(kfl, sizeof kfl, "%s/%s.kfl", WORK_DIR, stem);
    snprintf(out, sizeof out, "%s/%s", WORK_DIR, stem);
    snprintf(so,  sizeof so,  "%s/%s.rlenv.so", WORK_DIR, stem);
    rl_write_file_(kfl, src);
    rl_compile_(kfl, out, WORK_DIR);
    void *dso = rl_dlopen_(so);
    rl_resolve_surface_(dso, s);

    K26RlEnv *h = NULL;
    ASSERT(s->create(11u, 1u, &h) == K26RL_OK);
    int32_t need = s->spec(h, NULL, 0);
    ASSERT(need > 0 && need < 65536);
    uint8_t *blob = (uint8_t *)malloc((size_t)need);
    ASSERT(blob != NULL);
    ASSERT(s->spec(h, blob, (uint32_t)need) == need);
    rl_parse_spec_(blob, (uint32_t)need, view);
    free(blob);
    s->destroy(h);
    return dso;
}

/* The slices must be a partition: one per agent, contiguous from
 * zero, summing to the total, overlapping nowhere. */
static void assert_partition_(const char *what, uint32_t (*slice)[2],
                              int n, uint32_t total)
{
    uint32_t next = 0;
    for (int a = 0; a < n; a++) {
        if (slice[a][0] != next) {
            fprintf(stderr, "FAIL %s: agent %d offset %u, expected %u\n",
                    what, a, slice[a][0], next);
            exit(1);
        }
        next += slice[a][1];
    }
    if (next != total) {
        fprintf(stderr, "FAIL %s: slices cover %u of %u\n", what, next,
                total);
        exit(1);
    }
}

/* A name of exactly `n` bytes made of ASCII letters. */
static char *name_of_(int n)
{
    char *s = (char *)malloc((size_t)n + 1);
    ASSERT(s != NULL);
    for (int i = 0; i < n; i++) s[i] = (char)('a' + (i % 26));
    s[n] = '\0';
    return s;
}

/* A one-agent program whose agent name and channel base name are the
 * given lengths, for the name-bound arms. */
static char *bound_src_(int agent_len, int base_len)
{
    char *an = name_of_(agent_len);
    char *bn = name_of_(base_len);
    char *src = (char *)malloc(4096);
    ASSERT(src != NULL);
    snprintf(src, 4096,
        "form BOUND\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n"
        "    agent %s\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as %s\n"
        "    end\n"
        "    agent second\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as look\n"
        "    end\n"
        "end\n"
        "end\n", an, bn);
    free(an);
    free(bn);
    return src;
}

/* ---- Gates ---------------------------------------------------------- */

static void gate_slices_and_names_(void)
{
    RlSpecView two, one;
    RlSurface  s2, s1;
    void *d2 = build_and_open_(TWO_KFL, "two", &two, &s2);
    void *d1 = build_and_open_(ONE_KFL, "one", &one, &s1);

    ASSERT(two.agent_count == 2);
    ASSERT(one.agent_count == 1);
    ASSERT(two.n_obs_slices == 2 && two.n_act_slices == 2);
    ASSERT(one.n_obs_slices == 1 && one.n_act_slices == 1);

    /* The counts differ between the two agents on both vectors. This
     * is the condition that makes the partition test able to fail: with
     * equal counts, swapping the two agents' offsets produces slices
     * that still partition. */
    ASSERT(two.obs_slice[0][1] != two.obs_slice[1][1]);
    ASSERT(two.act_slice[0][1] != two.act_slice[1][1]);

    assert_partition_("two-agent observation", two.obs_slice, 2,
                      two.obs_total);
    assert_partition_("two-agent action", two.act_slice, 2, two.act_total);
    assert_partition_("one-agent observation", one.obs_slice, 1,
                      one.obs_total);
    assert_partition_("one-agent action", one.act_slice, 1, one.act_total);

    /* Source order decides which agent is which, so the counts are
     * asserted against the declarations rather than against each
     * other: alpha declares one observation channel and two actions,
     * beta two channels and one action. */
    ASSERT(two.obs_slice[0][1] == 5);
    ASSERT(two.obs_slice[1][1] == 10);
    ASSERT(two.act_slice[0][1] == 2);
    ASSERT(two.act_slice[1][1] == 1);

    /* Published names: qualified above one agent, bare at one. */
    ASSERT(two.n_chan_names == 15);
    ASSERT(strcmp(two.chan_names[3],  "alpha.trk_range") == 0);
    ASSERT(strcmp(two.chan_names[8],  "beta.trk_range") == 0);
    ASSERT(strcmp(two.chan_names[13], "beta.rel_range") == 0);
    ASSERT(one.n_chan_names == 5);
    ASSERT(strcmp(one.chan_names[3], "trk_range") == 0);

    g_arms++;
    printf("  slices: obs [%u,%u)+[%u,%u) act [%u,%u)+[%u,%u) of %u/%u\n",
           two.obs_slice[0][0], two.obs_slice[0][0] + two.obs_slice[0][1],
           two.obs_slice[1][0], two.obs_slice[1][0] + two.obs_slice[1][1],
           two.act_slice[0][0], two.act_slice[0][0] + two.act_slice[0][1],
           two.act_slice[1][0], two.act_slice[1][0] + two.act_slice[1][1],
           two.obs_total, two.act_total);
    dlclose(d1);
    dlclose(d2);
}

/* An observation-only agent: legal, and its action slice is empty
 * without displacing the other agent's. */
static void gate_observation_only_(void)
{
    RlSpecView v;
    RlSurface  s;
    void *d = build_and_open_(OBS_ONLY_KFL, "obsonly", &v, &s);
    ASSERT(v.agent_count == 2);
    ASSERT(v.act_slice[0][0] == 0 && v.act_slice[0][1] == 1);
    ASSERT(v.act_slice[1][0] == 1 && v.act_slice[1][1] == 0);
    assert_partition_("observation-only action", v.act_slice, 2,
                      v.act_total);
    assert_partition_("observation-only observation", v.obs_slice, 2,
                      v.obs_total);
    g_arms++;
    printf("  observation-only agent: act slice [1,1), obs [5,10)\n");
    dlclose(d);
}

/* One block is agent count 1, which is the case that tells a rule
 * conditioned on the count from one conditioned on a block being
 * absent. */
static void gate_one_block_(void)
{
    RlSpecView v;
    RlSurface  s;
    void *d = build_and_open_(ONE_BLOCK_KFL, "oneblock", &v, &s);
    ASSERT(v.agent_count == 1);
    ASSERT(v.n_obs_slices == 1 && v.n_act_slices == 1);
    assert_partition_("one-block observation", v.obs_slice, 1,
                      v.obs_total);
    assert_partition_("one-block action", v.act_slice, 1, v.act_total);
    /* Bare, not `pilot.trk_range`: with one agent there is nothing for
     * a qualifier to tell apart, and this is what keeps a one-agent
     * program's names the names it declared. */
    ASSERT(v.n_chan_names == 5);
    ASSERT(strcmp(v.chan_names[3], "trk_range") == 0);
    g_arms++;
    printf("  one agent block: count 1, names bare, slices whole\n");
    dlclose(d);
}

/* An agent with no `objective` publishes an all-zero reward stream for
 * its own agent, and its neighbour's reward is unaffected. */
static void gate_missing_objective_(void)
{
    RlSpecView v;
    RlSurface  s;
    void *dso = build_and_open_(HALF_OBJECTIVE_KFL, "halfobj", &v, &s);
    ASSERT(v.agent_count == 2);

    K26RlEnv *h = NULL;
    ASSERT(s.create(19u, 1u, &h) == K26RL_OK);
    double act[2] = { 0.0, 0.0 };
    double rew[2] = { -1.0, -1.0 };
    for (int t = 0; t < 3; t++) {
        ASSERT(s.step(h, act) == K26RL_OK);
        ASSERT(s.reward(h, rew) == K26RL_OK);
        if (rew[0] != 5.0 || rew[1] != 0.0) {
            fprintf(stderr, "FAIL missing objective: step %d rewards "
                    "%.17g and %.17g, expected 5 and 0\n", t, rew[0],
                    rew[1]);
            exit(1);
        }
    }
    g_arms++;
    printf("  an agent with no objective: rewards %g and %g\n",
           rew[0], rew[1]);
    s.destroy(h);
    dlclose(dso);
}

/* The emitted source is part of what the compiler produces, and a
 * reader of a two-agent artifact meets both its geometry macro and
 * the reward buffer's own description. They have to agree: the
 * description said "agent count 1" while the macro beside it said 2
 * and the allocation multiplied by it, which is a statement the
 * artifact contradicts three lines later. */
static void gate_emitted_text_(void)
{
    rl_write_file_(WORK_DIR "/text.kfl", TWO_KFL);
    rl_run_or_die_("./bin/kflc --emit " WORK_DIR "/text.kfl > "
                   WORK_DIR "/text.cc 2> " WORK_DIR "/text.err");
    FILE *f = fopen(WORK_DIR "/text.cc", "rb");
    ASSERT(f != NULL);
    static char cc[1 << 20];
    size_t n = fread(cc, 1, sizeof cc - 1, f);
    cc[n] = '\0';
    fclose(f);
    ASSERT(strstr(cc, "#define KFLRL_N_AGENTS 2") != NULL);
    if (strstr(cc, "agent count 1") != NULL) {
        fprintf(stderr, "FAIL emitted text: a two-agent artifact "
                "describes something in it as agent count 1\n");
        exit(1);
    }
    g_arms++;
    printf("  emitted text: nothing in a two-agent artifact claims "
           "agent count 1\n");
}

/* ---- The scope an objective expression is emitted into -------------- */

/* Three agents, twenty observation channels and three actions between
 * them, with each objective expression reading a named few. Every
 * shape the scope has to get right appears once: a channel read
 * unqualified by its owner, a channel of another agent read qualified,
 * an action read unqualified, an expression that reads no channel at
 * all, an agent with no objective, and the episode's own predicate,
 * which sits in no block.
 *
 * The declared indices are part of what is asserted: `atrk_range` is
 * the fourth channel of the first observe and `beta.btrk_range` the
 * fourth of the second, so a scope that computed a channel's place in
 * the vector by anything other than the widths before it would fail
 * here rather than in the arithmetic downstream of it. */
static const char *const SCOPE_KFL =
    "form SCOPE\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body alpha_craft gm=1.0 parent=earth pos_x=7.0e6"
    " vel_y=7546.0\n"
    "    astro_body beta_craft gm=1.0 parent=earth pos_x=1.1e7"
    " vel_y=6020.0\n"
    "    astro_body gamma_craft gm=1.0 parent=earth pos_x=1.3e7"
    " vel_y=5535.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 6\n"
    "        terminated when beta.btrk_range > 1.0e9\n"
    "    end\n"
    "    agent alpha\n"
    "        action thrust box -1.0 1.0 default 0.25\n"
    "        observe alpha_craft from earth mode=geometric as atrk\n"
    "        objective\n"
    "            reward atrk_range - beta.btrk_range\n"
    "            terminal thrust\n"
    "        end\n"
    "    end\n"
    "    agent beta\n"
    "        action brake box -1.0 1.0 default 0.5\n"
    "        observe beta_craft from earth mode=geometric as btrk\n"
    "        observe beta_craft from alpha_craft mode=geometric as brel\n"
    "        objective\n"
    "            reward 0.0 - episode.steps\n"
    "        end\n"
    "    end\n"
    "    agent gamma\n"
    "        action idle box -1.0 1.0 default 0.0\n"
    "        observe gamma_craft from earth mode=geometric as gtrk\n"
    "    end\n"
    "end\n"
    "end\n";

/* The channel and action declarations one emitted function opens with,
 * as one line each, in the order they were written. `fn_head` is the
 * function's own first line; the body runs to the first line that is a
 * closing brace alone. */
static int scope_decls_(const char *cc, const char *fn_head,
                        char decls[][160], int cap)
{
    const char *p = strstr(cc, fn_head);
    if (!p) {
        fprintf(stderr, "FAIL scope: the emitted source has no `%s`\n",
                fn_head);
        exit(1);
    }
    int n = 0;
    for (const char *line = strchr(p, '\n'); line; ) {
        line++;
        const char *end = strchr(line, '\n');
        if (!end) break;
        if (end - line == 1 && line[0] == '}') break;
        /* Every scope declaration reads one of the two vectors the
         * function is handed; nothing else in a body does. */
        const char *eq = strstr(line, " = _kfl_obs_v[");
        if (!eq || eq > end) eq = strstr(line, " = _kfl_act_v ?");
        if (eq && eq < end) {
            size_t len = (size_t)(end - line);
            ASSERT(n < cap);
            ASSERT(len < 160);
            memcpy(decls[n], line, len);
            decls[n][len] = '\0';
            n++;
        }
        line = end;
    }
    return n;
}

/* One emitted function's scope, against the exact lines it should
 * hold. A count alone would pass for a scope that declared the right
 * number of the wrong channels, so the lines are compared whole:
 * the name, the vector it reads and the index in it. */
static void scope_is_(const char *cc, const char *fn_head,
                      const char *const *want, int n_want)
{
    char got[64][160];
    int n = scope_decls_(cc, fn_head, got, 64);
    int bad = (n != n_want);
    for (int i = 0; !bad && i < n; i++) {
        if (strcmp(got[i], want[i]) != 0) bad = 1;
    }
    if (bad) {
        fprintf(stderr, "FAIL scope: `%s` declares %d channel(s), "
                "wanted %d\n", fn_head, n, n_want);
        for (int i = 0; i < n; i++) {
            fprintf(stderr, "  got  %s\n", got[i]);
        }
        for (int i = 0; i < n_want; i++) {
            fprintf(stderr, "  want %s\n", want[i]);
        }
        exit(1);
    }
    printf("  %-26s %d declaration(s), each the channel it reads\n",
           fn_head, n);
}

/* The scope holds what the expression reads and nothing else.
 *
 * What would make this arm vacuous, and how it is ruled out. An arm
 * that only counted declarations would pass for a scope holding the
 * right number of the wrong channels; the lines are compared whole. An
 * arm over one function would pass for an emitter that got one case
 * right; all seven the fixture produces are named, including the three
 * that read nothing. And an arm with no figure for the alternative
 * would not say what it is holding down: the fixture's twenty-three
 * channels are what a whole-program scope declares in each of the
 * seven, which is what this arm exists to prevent and what it prints.
 *
 * The values these functions return are held by the cross-agent arm
 * below, which drives the same shape of program and reads the numbers
 * off the surface. */
static void gate_scope_holds_what_is_read_(void)
{
    rl_write_file_(WORK_DIR "/scope.kfl", SCOPE_KFL);
    rl_run_or_die_("./bin/kflc --emit " WORK_DIR "/scope.kfl > "
                   WORK_DIR "/scope.cc 2> " WORK_DIR "/scope.err");
    FILE *f = fopen(WORK_DIR "/scope.cc", "rb");
    ASSERT(f != NULL);
    static char cc[1 << 22];
    size_t n = fread(cc, 1, sizeof cc - 1, f);
    ASSERT(n > 0 && n < sizeof cc - 1);
    cc[n] = '\0';
    fclose(f);

    printf("the scope an objective is emitted into\n");

    static const char *const r0[] = {
        "    const double atrk_range = _kfl_obs_v[3]; (void)atrk_range;",
        "    const double _kfl_q1_btrk_range = _kfl_obs_v[8]; "
        "(void)_kfl_q1_btrk_range;"
    };
    scope_is_(cc, "kflrl_reward_0_", r0, 2);

    static const char *const t0[] = {
        "    const double thrust = _kfl_act_v ? _kfl_act_v[0] : 0.0; "
        "(void)thrust;"
    };
    scope_is_(cc, "kflrl_terminal_0_", t0, 1);

    /* Beta's reward reads the step count and no channel; gamma
     * declares no objective at all, so both of its functions take
     * their documented default. */
    scope_is_(cc, "kflrl_reward_1_", NULL, 0);
    scope_is_(cc, "kflrl_terminal_1_", NULL, 0);
    scope_is_(cc, "kflrl_reward_2_", NULL, 0);
    scope_is_(cc, "kflrl_terminal_2_", NULL, 0);

    static const char *const tw[] = {
        "    const double _kfl_q1_btrk_range = _kfl_obs_v[8]; "
        "(void)_kfl_q1_btrk_range;"
    };
    scope_is_(cc, "kflrl_terminated_", tw, 1);

    printf("  the program declares 20 observation channels and 3 "
           "actions; a scope of the whole program would carry 23 "
           "declarations in each of these 7 functions\n");
    g_arms++;
}

static void gate_name_bound_(void)
{
    /* Exactly at the bound, with a channel name short enough that the
     * combination fits: accepted. */
    char *at31 = bound_src_(31, 3);
    must_accept_("agent name of exactly 31 bytes", at31);
    free(at31);

    /* One byte over: refused, and the diagnostic names the bound. */
    char *at32 = bound_src_(32, 3);
    const char *over[] = { "32 bytes", "at most 31", NULL };
    must_refuse_("agent name of 32 bytes", at32, over);
    free(at32);

    /* At the bound, with a channel base name at its own declarable
     * bound of 53. 31 + 1 + 53 + 11 + 1 is 97 and the entry holds 96,
     * so the combination overflows by one byte; the diagnostic must
     * decompose it rather than report only the total. */
    char *combo = bound_src_(31, 53);
    const char *parts[] = {
        "97 bytes", "entry holds 96",
        "agent name is 31", "the dot is 1", "channel name is 53",
        "_range_rate", NULL
    };
    must_refuse_("a qualified name that overflows the entry", combo, parts);
    free(combo);
}

static void gate_mixing_and_collisions_(void)
{
    static const char *const HEAD =
        "form MIX\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n";
    char src[4096];

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    action rogue box -1.0 1.0 default 0.0\n"
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f1[] = { "action `rogue`", "world level", "agent pilot",
                         NULL };
    must_refuse_("a world-level action beside an agent block", src, f1);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    observe craft from earth mode=geometric as loose\n"
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f2[] = { "observe ... as `loose`", "world level",
                         "agent pilot", NULL };
    must_refuse_("a world-level observe-as beside an agent block", src, f2);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    objective\n"
        "        reward 1.0\n"
        "    end\n"
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f3[] = { "`objective` block", "world level", "agent pilot",
                         NULL };
    must_refuse_("a world-level objective beside an agent block", src, f3);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent craft\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f4[] = { "agent `craft`", "astro_body craft", "rename",
                         NULL };
    must_refuse_("an agent name equal to a body name", src, f4);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "    end\n"
        "    agent pilot\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f5[] = { "agent `pilot`", "already declared", NULL };
    must_refuse_("two agents of one name", src, f5);
}

static void gate_block_contents_(void)
{
    static const char *const HEAD =
        "form BLK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n";
    char src[4096];

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent idle\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f1[] = { "agent `idle`", "neither an `action` nor an "
                         "`objective`", NULL };
    must_refuse_("a block with neither an action nor an objective", src, f1);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        objective\n"
        "            reward 1.0\n"
        "        end\n"
        "        objective\n"
        "            reward 2.0\n"
        "        end\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f2[] = { "agent `pilot`", "duplicate `objective`", NULL };
    must_refuse_("two objectives in one block", src, f2);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        let stray: double = 1.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f3[] = { "agent `pilot`", "and nothing else", NULL };
    must_refuse_("a binding inside an agent block", src, f3);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f4[] = { "agent `pilot`", "needs an `as <name>` clause",
                         NULL };
    must_refuse_("an unnamed observe inside an agent block", src, f4);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    if 1.0 > 0.0\n"
        "        agent pilot\n"
        "            action thrust box -1.0 1.0 default 0.0\n"
        "        end\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f5[] = { "must be top level", "source order", NULL };
    must_refuse_("an agent block inside a conditional", src, f5);
}

static void gate_on_step_resolution_(void)
{
    static const char *const HEAD =
        "form STEP\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n"
        "    agent alpha\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        action yaw box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "    agent beta\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as look\n"
        "    end\n";
    char src[4096];

    /* A name two blocks declare, used unqualified: refused, naming
     * both declaration sites and the qualified form. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    on_step\n"
        "        craft.vel_x = craft.vel_x + thrust\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f1[] = { "`thrust` is ambiguous", "agent `alpha` declares "
                         "it at line 10", "agent `beta` declares it at "
                         "line 15", "`alpha.thrust`", "`beta.thrust`",
                         NULL };
    must_refuse_("an ambiguous unqualified action name in on_step",
                 src, f1);

    /* The qualified form of the same name resolves, and so does a
     * name only one block declares. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    on_step\n"
        "        craft.vel_x = craft.vel_x + alpha.thrust\n"
        "        craft.vel_y = craft.vel_y + beta.thrust\n"
        "        craft.vel_z = craft.vel_z + yaw\n"
        "    end\n"
        "end\n"
        "end\n");
    must_accept_("qualified and uniquely named actions in on_step", src);

    /* A qualified name whose channel the agent does not declare. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    on_step\n"
        "        craft.vel_x = craft.vel_x + alpha.brake\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f2[] = { "agent `alpha` declares no action called `brake`",
                         NULL };
    must_refuse_("a qualified name for an action nobody declares", src, f2);

    /* An assignment to a channel, which is a reading and not a
     * command. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    on_step\n"
        "        alpha.thrust = 1.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f3[] = { "read", "only", NULL };
    must_refuse_("an assignment to an action channel", src, f3);
}

/* Channel names are unique within their owner and may repeat across
 * owners. Both halves are asserted: the fixtures above already carry
 * two blocks declaring one action name, and the refusals here are what
 * keeps the rule from having been widened to nothing. */
static void gate_name_scoping_(void)
{
    static const char *const HEAD =
        "form SCOPE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n";
    char src[4096];

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        action thrust box -2.0 2.0 default 0.0\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f1[] = { "action `thrust`", "duplicate action name", NULL };
    must_refuse_("one action name twice in one block", src, f1);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent pilot\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "        observe craft from earth mode=astrometric as trk\n"
        "    end\n"
        "end\n"
        "end\n");
    const char *f2[] = { "observe ... as `trk`", "duplicate channel name",
                         NULL };
    must_refuse_("one channel name twice in one block", src, f2);

    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent one\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "    agent two\n"
        "        action thrust box -2.0 2.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "end\n"
        "end\n");
    must_accept_("one action and channel name in each of two blocks", src);
}

/* The episode's own `terminated when` belongs to no block, so it
 * names the agent it reads whenever there is more than one. */
static void gate_episode_scope_(void)
{
    static const char *const HEAD =
        "form EPI\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n";
    static const char *const TAIL =
        "    agent alpha\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "    agent beta\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as look\n"
        "    end\n"
        "end\n"
        "end\n";
    char src[4096];

    snprintf(src, sizeof src,
        "%s"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "        terminated when trk_range < 1.0\n"
        "    end\n"
        "%s", HEAD, TAIL);
    const char *f1[] = { "`trk_range` is a channel of agent `alpha`",
                         "`alpha.trk_range`", NULL };
    must_refuse_("an unqualified channel in terminated when", src, f1);

    snprintf(src, sizeof src,
        "%s"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "        terminated when alpha.trk_range < 1.0"
        " || beta.look_range < 1.0\n"
        "    end\n"
        "%s", HEAD, TAIL);
    must_accept_("qualified channels in terminated when", src);
}

/* ---- Cross-agent reads, driven ------------------------------------- */

static void gate_cross_agent_reads_(void)
{
    RlSpecView v;
    RlSurface  s;
    void *dso = build_and_open_(TWO_KFL, "cross", &v, &s);

    K26RlEnv *h = NULL;
    ASSERT(s.create(2024u, 2u, &h) == K26RL_OK);

    const uint32_t n_envs = 2;
    double *act = (double *)calloc(n_envs * v.act_total, sizeof(double));
    double *obs = (double *)calloc(n_envs * v.obs_total, sizeof(double));
    double *rew = (double *)calloc(n_envs * v.agent_count, sizeof(double));
    ASSERT(act && obs && rew);

    /* alpha's range is channel 3, beta's is channel 8, by the slices
     * asserted above. */
    const uint32_t ALPHA_RANGE = 3, BETA_RANGE = 8;
    int steps = 0;
    for (uint32_t t = 0; t < 5; t++) {
        for (uint32_t i = 0; i < n_envs * v.act_total; i++) {
            act[i] = 0.1 * (double)(i + t);
        }
        ASSERT(s.step(h, act) == K26RL_OK);
        ASSERT(s.obs(h, obs) == K26RL_OK);
        ASSERT(s.reward(h, rew) == K26RL_OK);
        for (uint32_t e = 0; e < n_envs; e++) {
            const double *o = obs + (size_t)e * v.obs_total;
            double a_range = o[ALPHA_RANGE];
            double b_range = o[BETA_RANGE];
            /* The fixture is built so the two differ at every step;
             * without that a read of the wrong agent's channel would
             * look exactly like a read of the right one. */
            ASSERT(fabs(a_range - b_range) > 1.0);
            double want = b_range - a_range;
            double got_a = rew[(size_t)e * v.agent_count + 0];
            double got_b = rew[(size_t)e * v.agent_count + 1];
            if (got_a != want) {
                fprintf(stderr, "FAIL cross-agent: env %u step %u: "
                        "alpha reward %.17g, expected %.17g "
                        "(alpha range %.17g, beta range %.17g)\n",
                        e, t, got_a, want, a_range, b_range);
                exit(1);
            }
            /* The zero-sum pair the clause exists for: written once as
             * the negation of the other, the two are the same quantity
             * by construction. */
            if (got_b != -got_a) {
                fprintf(stderr, "FAIL cross-agent: env %u step %u: "
                        "beta reward %.17g is not the negation of "
                        "alpha's %.17g\n", e, t, got_b, got_a);
                exit(1);
            }
        }
        steps++;
    }
    g_arms++;
    printf("  cross-agent reads: %d steps x %u environments, zero sum "
           "exact\n", steps, n_envs);
    free(act); free(obs); free(rew);
    s.destroy(h);
    dlclose(dso);
}

/* A qualified action name inside `on_step` commands the agent that
 * owns it. Reading the rewards cannot show this: an action reaches
 * the world through a body-state write, so the bodies are what is
 * read back, through the frozen body-state getter. */
static void gate_qualified_application_(void)
{
    RlSpecView v;
    RlSurface  s;
    void *dso = build_and_open_(APPLY_KFL, "apply", &v, &s);
    ASSERT(v.agent_count == 2);
    ASSERT(v.act_total == 2);

    K26RlEnv *h = NULL;
    ASSERT(s.create(31u, 1u, &h) == K26RL_OK);

    /* Two values far apart, so a command landing on the wrong craft
     * cannot look like one landing on the right craft. */
    const double ALPHA_PUSH = 111.0, BETA_PUSH = 222.0;
    double act[2] = { ALPHA_PUSH, BETA_PUSH };
    ASSERT(s.step(h, act) == K26RL_OK);

    int32_t need = s.bodies(h, K26RL_BODY_REF_ORIGIN, NULL, 0);
    ASSERT(need == 3 * 6);
    double b[18];
    ASSERT(s.bodies(h, K26RL_BODY_REF_ORIGIN, b, 18) == need);

    /* Body order is declaration order: earth, alpha_craft,
     * beta_craft; six doubles each, position then velocity. Both
     * craft start in the x-y plane with no out-of-plane motion, so
     * the z velocity after one second is what `on_step` wrote plus
     * the small z acceleration the displacement itself creates. */
    double vz_alpha = b[1 * 6 + 5];
    double vz_beta  = b[2 * 6 + 5];
    if (fabs(vz_alpha - ALPHA_PUSH) > 1.0e-2 ||
        fabs(vz_beta - BETA_PUSH) > 1.0e-2) {
        fprintf(stderr, "FAIL qualified application: alpha craft vz "
                "%.17g (expected %.17g), beta craft vz %.17g (expected "
                "%.17g)\n", vz_alpha, ALPHA_PUSH, vz_beta, BETA_PUSH);
        exit(1);
    }
    g_arms++;
    printf("  qualified actions in on_step: alpha craft vz %.6f, beta "
           "craft vz %.6f\n", vz_alpha, vz_beta);
    s.destroy(h);
    dlclose(dso);
}

/* ---- Batch mode and the episode file --------------------------------- */

static void gate_batch_(void)
{
    char cmd[1024];
    (void)!system("rm -f " WORK_DIR "/batch.k26ep");
    snprintf(cmd, sizeof cmd,
        WORK_DIR "/two --envs 2 --episodes 2 --seed 7 --out "
        WORK_DIR "/batch.k26ep > " WORK_DIR "/batch.log 2>&1");
    int rc = system(cmd);
    ASSERT(rc == 0);

    K26RlEpisodeReader *r = NULL;
    ASSERT(k26rl_episode_reader_open(WORK_DIR "/batch.k26ep", &r) ==
           K26RL_OK);
    K26RlEpisodeInfo info;
    ASSERT(k26rl_episode_reader_info(r, &info) == K26RL_OK);
    ASSERT(info.agent_count == 2);
    ASSERT(info.n_envs == 2);
    ASSERT(info.episode_count >= 4);

    K26RlEpisodeData d;
    ASSERT(k26rl_episode_read(r, 0u, 0u, 0u, &d) == K26RL_OK);
    ASSERT(d.step_count > 0);
    /* Batch mode drives every action channel of every agent at its
     * declared default for the whole run: 0.25 and 0.5 for alpha,
     * 0.75 for beta. */
    for (uint32_t t = 0; t < d.step_count; t++) {
        const double *a = d.act + (size_t)t * info.act_total;
        ASSERT(a[0] == 0.25);
        ASSERT(a[1] == 0.5);
        ASSERT(a[2] == 0.75);
    }
    /* One reward stream per agent, and the zero-sum pair holds in the
     * recording as it does through the surface. */
    for (uint32_t t = 0; t < d.step_count; t++) {
        const double *w = d.rewards + (size_t)t * info.agent_count;
        ASSERT(w[1] == -w[0]);
    }
    g_arms++;
    printf("  batch: %u episodes, %u steps in the first, defaults held, "
           "rewards per agent\n", info.episode_count, d.step_count);
    k26rl_episode_free(&d);
    k26rl_episode_reader_close(r);
}

int main(void)
{
    if (!rl_libs_present_("test_rl_agent")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    printf("test_rl_agent: the agent block\n");
    gate_slices_and_names_();
    gate_one_block_();
    gate_observation_only_();
    gate_missing_objective_();
    gate_emitted_text_();
    gate_scope_holds_what_is_read_();
    gate_name_bound_();
    gate_mixing_and_collisions_();
    gate_block_contents_();
    gate_on_step_resolution_();
    gate_name_scoping_();
    gate_episode_scope_();
    gate_cross_agent_reads_();
    gate_qualified_application_();
    gate_batch_();
    printf("test_rl_agent: %d arm(s) passed\n", g_arms);
    return 0;
}
