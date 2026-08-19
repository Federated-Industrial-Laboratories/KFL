/* test_rl_multiagent_view.c: the viewer over a recording with more
 * than one agent and more than one craft.
 *
 * Three things join here and each has its own way of being wrong.
 * Panels grouped per agent can take their grouping from the channel
 * names instead of from the spec's slice tags, and the names carry an
 * agent prefix that makes the wrong reading look right. An assembly
 * bound to a body can be bound to the other body and draw the wrong
 * shape in the right place. A detection's line of sight can be drawn
 * on a step where nothing was detected, which is the one step it must
 * not appear on.
 *
 * Everything is driven through the headless dump. No window, no
 * graphics context and no display is opened anywhere.
 *
 * The fixture is built so that each of those wrong readings gives a
 * different answer from the right one, and each arm computes the
 * wrong answer as well as the right one and shows the two apart.
 *
 * Gates:
 *   1. Agent grouping. Each agent's channels are exactly its
 *      published slice, the two slices partition the observation
 *      vector, and the counts are the tags' rather than an equal
 *      division. Both agents declare a channel base of the same name,
 *      so a grouping keyed on the unqualified name gives a different
 *      partition, which the arm computes and shows.
 *   2. Per-agent reward. Each agent's stream is its own column: agent
 *      0's reward is its objective recomputed here from the
 *      observation channel it reads, bit for bit, and agent 1's is
 *      the negation of agent 0's, which the fixture makes true and a
 *      stream shown twice makes false. The returns accumulate.
 *   3. Body-qualified assets. Two assemblies of different geometry,
 *      bound to two bodies, each draw on their own body; binding each
 *      to the other body is refused by the digest and draws nothing;
 *      and one binding alone leaves the other body without geometry.
 *   4. The detection line of sight. Drawn on exactly the steps where
 *      the recorded flag is 1.0, over two detections whose flags
 *      differ, with both values present in the run. Its far end is
 *      the reconstructed point of its own base at its own step, bit
 *      for bit, and another step's point differs.
 *   5. The element toggle. `--elements detection` yields detection
 *      elements and no others, and every element on yields both.
 *   6. Watching changes nothing. The recording's bytes are what they
 *      were before the viewer first read them, and two identical runs
 *      agree byte for byte.
 *
 * Skips (77) when the sibling stack archives or the viewer binary are
 * not built.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <math.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_multiagent_view_test"
#define VIEWER "../tools/k26rl_view/k26rl_view"

/* The fixture's own numbers, in one place so an arm cannot disagree
 * with the command line that produced its dump. */
#define STEPS       8
#define OBS_TOTAL   19
#define ACT_TOTAL   3
#define ALPHA_OBS   12
#define BETA_OBS    7
#define ALPHA_ACT   1
#define BETA_ACT    2
/* The observation channel agent 0's objective reads, and the divisor
 * it divides by. Written here from the source below, not read back
 * from the viewer. */
#define ALPHA_RANGE_CHANNEL 2
#define REWARD_DIVISOR      1000.0
/* The last step on which the wider payload still meets its threshold.
 * Measured from the fixture and asserted rather than trusted: the arm
 * requires both a detected step and an undetected one. */
#define LAST_DETECTED 2

/* Two agents, two craft, and two detections of one target.
 *
 * Both agents declare an observation called `trk`, so the published
 * names collide on their unqualified part and only the slice tags can
 * say which channel belongs to which agent. The two agents also carry
 * different channel counts on both vectors, so an equal division of
 * either vector puts channels under the wrong agent.
 *
 * The two payloads differ only in aperture and threshold: the wider
 * one detects the target until the range grows past its threshold and
 * the narrow one never detects it at all. That gives one detection
 * whose flag changes during the episode and one whose flag does not,
 * which is what an arm needs to tell a flag being read from a flag
 * being assumed.
 */
static const char *const MA_KFL =
    "form MAVIEW\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body chaser assembly=\"ma_chaser.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0\n"
    "    astro_body target assembly=\"ma_target.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=120.0 pos_z=0.0 vel_x=0.0 vel_y=7946.0 vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0\n"
    "    astro_payload eye body=chaser kind=detect_ir aperture_m=1.0"
    " integration_s=0.5 passband_lo_um=3.0 passband_hi_um=12.0"
    " throughput=0.5 snr_threshold=2.5e7 target_temp_k=300.0"
    " target_emissivity=0.9 t_optics_k=280.0 optics_emissivity=0.05\n"
    "    astro_payload squint body=chaser kind=detect_ir aperture_m=0.01"
    " integration_s=0.5 passband_lo_um=3.0 passband_hi_um=12.0"
    " throughput=0.5 snr_threshold=1.0e9 target_temp_k=300.0"
    " target_emissivity=0.9 t_optics_k=280.0 optics_emissivity=0.05\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 8\n"
    "    end\n"
    "    agent alpha\n"
    "        action push box -1.0 1.0 default 0.25\n"
    "        observe detect eye of target as trk\n"
    "        observe target from chaser mode=geometric as rel\n"
    "        objective\n"
    "            reward alpha.trk_range / 1000.0\n"
    "        end\n"
    "    end\n"
    "    agent beta\n"
    "        action dodge box -1.0 1.0 default 0.5\n"
    "        action spin box -1.0 1.0 default 0.1\n"
    "        observe detect squint of target as trk\n"
    "        objective\n"
    "            reward 0.0 - (alpha.trk_range / 1000.0)\n"
    "        end\n"
    "    end\n"
    "    on_step\n"
    "        chaser.vel_y = chaser.vel_y + alpha.push * 0.01\n"
    "        target.vel_y = target.vel_y + beta.dodge * 0.01\n"
    "    end\n"
    "end\n"
    "end\n";

/* An asymmetric tetrahedron, four vertices and six edges. */
static const char *const CHASER_MESH =
    "# an asymmetric tetrahedron\n"
    "v 2.0 0.0 0.0\n"
    "v -1.5 -1.0 -0.4\n"
    "v -1.5 1.2 -0.4\n"
    "v -1.0 0.0 0.9\n"
    "f 1 2 3\n"
    "f 1 3 4\n"
    "f 1 4 2\n"
    "f 2 4 3\n";

/* A box, eight vertices and eighteen edges. It is a different shape
 * and a different size from the tetrahedron, so an element carrying
 * the wrong craft's geometry is visible as a wrong vertex count
 * before any coordinate is compared. */
static const char *const TARGET_MESH =
    "# a box, and not the tetrahedron\n"
    "v -1.0 -1.0 -1.0\n"
    "v 1.0 -1.0 -1.0\n"
    "v 1.0 1.0 -1.0\n"
    "v -1.0 1.0 -1.0\n"
    "v -1.0 -1.0 1.0\n"
    "v 1.0 -1.0 1.0\n"
    "v 1.0 1.0 1.0\n"
    "v -1.0 1.0 1.0\n"
    "f 1 3 2\n"
    "f 1 4 3\n"
    "f 5 6 7\n"
    "f 5 7 8\n"
    "f 1 2 6\n"
    "f 1 6 5\n"
    "f 2 3 7\n"
    "f 2 7 6\n"
    "f 3 4 8\n"
    "f 3 8 7\n"
    "f 4 1 5\n"
    "f 4 5 8\n";

static const char *const CHASER_ASM =
    "assembly ma_chaser\n"
    "    frame x_to_port\n"
    "    provenance mass \"gate fixture, not a craft\" computed\n"
    "    provenance geometry \"gate fixture, not a craft\" computed\n"
    "    component hull\n"
    "        mass 900.0\n"
    "        at 0.0 0.0 0.0\n"
    "        mesh ma_chaser.k26mesh\n"
    "        collider box 2.0 1.2 0.9\n"
    "    end\n"
    "end\n";

static const char *const TARGET_ASM =
    "assembly ma_target\n"
    "    frame x_to_port\n"
    "    provenance mass \"gate fixture, not a craft\" computed\n"
    "    provenance geometry \"gate fixture, not a craft\" computed\n"
    "    component hull\n"
    "        mass 4000.0\n"
    "        at 0.0 0.0 0.0\n"
    "        mesh ma_target.k26mesh\n"
    "        collider sphere 0.0 0.0 0.0 1.8\n"
    "    end\n"
    "end\n";

/* The expected geometry, written here from the mesh sources above so
 * an arm holds the dump against a count it derived rather than
 * against the count the dump reported. */
#define CHASER_VERTS 4
#define CHASER_EDGES 6
#define TARGET_VERTS 8
#define TARGET_EDGES 18

static int g_arms;

/* ---- running the viewer and reading its dumps ---------------------- */

static void run_viewer_(const char *args, const char *out_path)
{
    char cmd[8192];
    int n = snprintf(cmd, sizeof cmd, "%s %s > %s 2>%s.err", VIEWER, args,
                     out_path, out_path);
    int rc;

    ASSERT((size_t)n < sizeof cmd);
    rc = system(cmd);
    if (rc != 0) {
        char show[512];
        fprintf(stderr, "viewer failed (rc=%d): %s %s\n", rc, VIEWER, args);
        snprintf(show, sizeof show, "cat %s.err", out_path);
        (void)!system(show);
        exit(1);
    }
}

static char *slurp_(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    char *buf;
    long n;

    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    n = ftell(f);
    ASSERT(n >= 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    buf = malloc((size_t)n + 1);
    ASSERT(buf != NULL);
    ASSERT(fread(buf, 1, (size_t)n, f) == (size_t)n);
    buf[n] = '\0';
    fclose(f);
    if (out_len)
        *out_len = (size_t)n;
    return buf;
}

/* A binary64 written as its bit pattern, which is how every value in
 * these dumps travels. */
static double hexd_(const char *s)
{
    uint64_t bits = strtoull(s, NULL, 16);
    double v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

static uint64_t bits_(double v)
{
    uint64_t b;
    memcpy(&b, &v, sizeof b);
    return b;
}

/* The tail of the line beginning with `prefix`, or 0. Prefixes carry
 * every leading field, so a match is exact rather than the first line
 * that happens to start the same way. */
static const char *line_(const char *text, const char *prefix)
{
    size_t plen = strlen(prefix);
    const char *p = text;

    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, prefix, plen) == 0)
            return p + plen;
        p = eol ? eol + 1 : NULL;
    }
    return NULL;
}

/* Every line beginning with `prefix`, in order, with the prefix
 * removed. Returns how many were found. */
static int lines_(const char *text, const char *prefix, char out[][256],
                  int max)
{
    size_t plen = strlen(prefix);
    const char *p = text;
    int n = 0;

    while (p && *p && n < max) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, prefix, plen) == 0) {
            size_t take = len - plen;
            ASSERT(take < 256);
            memcpy(out[n], p + plen, take);
            out[n][take] = '\0';
            n++;
        }
        p = eol ? eol + 1 : NULL;
    }
    return n;
}

/* The index of the element of that kind with that name, or -1. */
static int element_index_(const char *text, unsigned k, unsigned step,
                          const char *kind, const char *name)
{
    char pre[64];
    const char *p = text;
    size_t plen;

    snprintf(pre, sizeof pre, "scene_element %u %u ", k, step);
    plen = strlen(pre);
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, pre, plen) == 0) {
            unsigned idx = 0, nv = 0, nseg = 0, nface = 0;
            char gk[32], gb[64], gn[128];
            if (sscanf(p + plen, "%u %31s %63s %u %u %u %127s", &idx, gk, gb,
                       &nv, &nseg, &nface, gn) == 7 &&
                strcmp(gk, kind) == 0 && strcmp(gn, name) == 0)
                return (int)idx;
        }
        p = eol ? eol + 1 : NULL;
    }
    return -1;
}

/* How many elements of that kind the step carries. Element lines
 * name their kind in the fourth field, which is what this reads: a
 * substring search would match the toggle lines and the standing
 * labels in the header, which are not elements. */
static int kind_count_(const char *text, unsigned k, unsigned step,
                       const char *kind)
{
    char pre[64];
    const char *p = text;
    size_t plen;
    int n = 0;

    snprintf(pre, sizeof pre, "scene_element %u %u ", k, step);
    plen = strlen(pre);
    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, pre, plen) == 0) {
            unsigned idx = 0;
            char gk[32];
            if (sscanf(p + plen, "%u %31s", &idx, gk) == 2 &&
                strcmp(gk, kind) == 0)
                n++;
        }
        p = eol ? eol + 1 : NULL;
    }
    return n;
}

static void element_counts_(const char *text, unsigned k, unsigned step,
                            int idx, unsigned *nv, unsigned *nseg)
{
    char pre[64];
    const char *p;
    char gk[32], gb[64];
    unsigned nface = 0;

    snprintf(pre, sizeof pre, "scene_element %u %u %d ", k, step, idx);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%31s %63s %u %u %u", gk, gb, nv, nseg, &nface) == 5);
}

/* One vertex's normalised device coordinate, as three bit patterns. */
static void vertex_(const char *text, unsigned k, unsigned step, int idx,
                    unsigned v, uint64_t out[3])
{
    char pre[96];
    char a[24], b[24], c[24];
    unsigned bh = 0;
    const char *p;

    snprintf(pre, sizeof pre, "scene_vertex %u %u %d %u ", k, step, idx, v);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%23s %23s %23s %u", a, b, c, &bh) == 4);
    out[0] = strtoull(a, NULL, 16);
    out[1] = strtoull(b, NULL, 16);
    out[2] = strtoull(c, NULL, 16);
}

/* Whether the one segment of an element was handed to the window. */
static int segment_drawn_(const char *text, unsigned k, unsigned step,
                          int idx)
{
    char pre[96];
    const char *p;
    unsigned a = 0, b = 0;
    int drawn = -1;

    snprintf(pre, sizeof pre, "scene_segment %u %u %d 0 ", k, step, idx);
    p = line_(text, pre);
    ASSERT(p != NULL);
    ASSERT(sscanf(p, "%u %u %d", &a, &b, &drawn) == 3);
    ASSERT(drawn == 0 || drawn == 1);
    return drawn;
}

/* ---- the arms -------------------------------------------------------- */

/* Gate 1: the grouping comes from the slice tags.
 *
 * The right answer and the wrong one are both computed here. The
 * right answer is the slices the spec publishes. The wrong one is the
 * grouping a viewer would reach by keying on a channel's unqualified
 * name, which this fixture makes different because both agents
 * declare a base called `trk`.
 */
static void gate_grouping_(const char *meta, const char *agents)
{
    char rows[64][256];
    int n, i;
    unsigned off[2] = { 0, 0 }, cnt[2] = { 0, 0 };
    unsigned seen[OBS_TOTAL];
    unsigned owner[OBS_TOTAL];
    unsigned equal_split;
    int misplaced_by_equal_split = 0;
    int bases_shared = 0;

    /* The published slices, straight from the spec's own tags. */
    n = lines_(meta, "obs_slice ", rows, 64);
    ASSERT(n == 2);
    for (i = 0; i < n; i++) {
        unsigned a = 0, o = 0, c = 0;
        ASSERT(sscanf(rows[i], "%u %u %u", &a, &o, &c) == 3);
        ASSERT(a < 2);
        off[a] = o;
        cnt[a] = c;
    }
    ASSERT(off[0] == 0 && cnt[0] == ALPHA_OBS);
    ASSERT(off[1] == ALPHA_OBS && cnt[1] == BETA_OBS);
    ASSERT(cnt[0] + cnt[1] == OBS_TOTAL);

    /* What the panel put under each agent. */
    for (i = 0; i < OBS_TOTAL; i++) {
        seen[i] = 0;
        owner[i] = 0xFFFFFFFFu;
    }
    n = lines_(agents, "agent_channel ", rows, 64);
    ASSERT(n == OBS_TOTAL);
    for (i = 0; i < n; i++) {
        unsigned a = 0, ch = 0;
        char name[128];
        ASSERT(sscanf(rows[i], "%u %u %127s", &a, &ch, name) == 3);
        ASSERT(a < 2);
        ASSERT(ch < OBS_TOTAL);
        ASSERT(seen[ch] == 0);       /* no channel under two agents */
        seen[ch] = 1;
        owner[ch] = a;
        /* Inside the published slice, and inside no other. */
        ASSERT(ch >= off[a] && ch - off[a] < cnt[a]);
    }
    for (i = 0; i < OBS_TOTAL; i++)
        ASSERT(seen[i] == 1);        /* and every channel under one */

    /* The equal division a viewer might reach instead, and how many
     * channels it would misplace. A fixture whose agents had equal
     * counts could not tell the two apart. */
    equal_split = OBS_TOTAL / 2;
    for (i = 0; i < OBS_TOTAL; i++) {
        unsigned by_split = ((unsigned)i < equal_split) ? 0u : 1u;
        if (by_split != owner[i])
            misplaced_by_equal_split++;
    }
    ASSERT(misplaced_by_equal_split > 0);

    /* The unqualified names the two agents share. A grouping keyed on
     * a channel's name after its agent prefix would put each of these
     * pairs in one group, which is not the partition above. */
    n = lines_(agents, "agent_channel ", rows, 64);
    for (i = 0; i < n; i++) {
        unsigned ai = 0, chi = 0;
        char ni[128];
        const char *bi;
        int j;
        ASSERT(sscanf(rows[i], "%u %u %127s", &ai, &chi, ni) == 3);
        bi = strchr(ni, '.');
        ASSERT(bi != NULL);          /* qualified, as more than one agent */
        for (j = 0; j < n; j++) {
            unsigned aj = 0, chj = 0;
            char nj[128];
            const char *bj;
            ASSERT(sscanf(rows[j], "%u %u %127s", &aj, &chj, nj) == 3);
            bj = strchr(nj, '.');
            ASSERT(bj != NULL);
            if (aj != ai && strcmp(bi + 1, bj + 1) == 0)
                bases_shared++;
        }
    }
    ASSERT(bases_shared > 0);

    /* The agent names come from those same qualified names. */
    ASSERT(line_(agents, "agent 0 alpha") != NULL);
    ASSERT(line_(agents, "agent 1 beta") != NULL);
    /* And the action slices, which are the other tag. */
    ASSERT(line_(agents, "agent_act_slice 0 0 1 1") != NULL);
    ASSERT(line_(agents, "agent_act_slice 1 1 2 1") != NULL);
    ASSERT(ALPHA_ACT + BETA_ACT == ACT_TOTAL);

    g_arms++;
    printf("gate 1: %d observation channels partition into %u and %u by the"
           " slice tags, %d of them shared an unqualified name across the"
           " two agents and %d would be misplaced by an equal division:"
           " OK\n", OBS_TOTAL, cnt[0], cnt[1], bases_shared,
           misplaced_by_equal_split);
}

/* Gate 2: each agent's reward stream is its own column.
 *
 * Agent 0's objective is recomputed here from the observation channel
 * it reads, which is a different panel and a different code path from
 * the reward the agents panel reports. Agent 1's is the negation of
 * agent 0's, which the fixture's own source makes true and which a
 * viewer showing one stream twice makes false.
 */
static void gate_reward_(const char *obs, const char *agents)
{
    char rows[256][256];
    int n, i;
    double r0[STEPS], r1[STEPS], ret0[STEPS];
    double range[STEPS];
    double running = 0.0;
    int differ = 0;

    n = lines_(obs, "obs 0 ", rows, 256);
    ASSERT(n == STEPS * OBS_TOTAL);
    for (i = 0; i < STEPS; i++)
        range[i] = 0.0;
    for (i = 0; i < n; i++) {
        unsigned st = 0, ch = 0;
        char v[24];
        ASSERT(sscanf(rows[i], "%u %u %23s", &st, &ch, v) == 3);
        if (ch == ALPHA_RANGE_CHANNEL) {
            ASSERT(st < STEPS);
            range[st] = hexd_(v);
        }
    }

    n = lines_(agents, "agent_reward 0 ", rows, 256);
    ASSERT(n == 2 * STEPS);
    for (i = 0; i < n; i++) {
        unsigned a = 0, st = 0;
        char rv[24], sv[24];
        ASSERT(sscanf(rows[i], "%u %u %23s %23s", &a, &st, rv, sv) == 4);
        ASSERT(a < 2 && st < STEPS);
        if (a == 0) {
            r0[st] = hexd_(rv);
            ret0[st] = hexd_(sv);
        } else {
            r1[st] = hexd_(rv);
        }
    }

    for (i = 0; i < STEPS; i++) {
        /* The objective, written out here from the source above. */
        double want = range[i] / REWARD_DIVISOR;
        if (bits_(r0[i]) != bits_(want)) {
            fprintf(stderr, "FAIL step %d: agent 0 reward %016" PRIx64
                    ", the objective recomputed gives %016" PRIx64 "\n",
                    i, bits_(r0[i]), bits_(want));
            exit(1);
        }
        /* The second agent's objective is the first's negated, so a
         * stream shown twice would make this false. */
        if (bits_(r1[i]) != bits_(-r0[i])) {
            fprintf(stderr, "FAIL step %d: agent 1 reward %016" PRIx64
                    " is not the negation of agent 0's %016" PRIx64 "\n",
                    i, bits_(r1[i]), bits_(r0[i]));
            exit(1);
        }
        if (bits_(r1[i]) != bits_(r0[i]))
            differ++;
        running += r0[i];
        if (bits_(ret0[i]) != bits_(running)) {
            fprintf(stderr, "FAIL step %d: agent 0 return %016" PRIx64
                    ", accumulating its own rewards gives %016" PRIx64 "\n",
                    i, bits_(ret0[i]), bits_(running));
            exit(1);
        }
    }
    ASSERT(differ == STEPS);

    g_arms++;
    printf("gate 2: %d steps of agent 0's reward equal its objective"
           " recomputed from the observation channel, its return equals"
           " their running sum, and agent 1's stream differs from agent"
           " 0's at every one of them: OK\n", STEPS);
}

/* Gate 3: an assembly draws on the body it was bound to.
 *
 * The right binding puts each craft's own geometry on its own body,
 * which the vertex counts show because the two craft are different
 * shapes. The swap is the defect this arm exists for, and the digest
 * refuses it: the bytes the recording carries for a body are the
 * bytes that must match, so an assembly bound to the wrong body is
 * reported and drawn nowhere.
 */
static void gate_assets_(const char *both, const char *swapped,
                         const char *one)
{
    int ic, it;
    unsigned nv, nseg;

    ASSERT(CHASER_VERTS != TARGET_VERTS);   /* else nothing to tell apart */

    ic = element_index_(both, 0, 0, "wireframe", "chaser/ma_chaser");
    it = element_index_(both, 0, 0, "wireframe", "target/ma_target");
    ASSERT(ic >= 0);
    ASSERT(it >= 0);
    element_counts_(both, 0, 0, ic, &nv, &nseg);
    ASSERT(nv == CHASER_VERTS);
    ASSERT(nseg == CHASER_EDGES);
    element_counts_(both, 0, 0, it, &nv, &nseg);
    ASSERT(nv == TARGET_VERTS);
    ASSERT(nseg == TARGET_EDGES);
    /* And neither craft's shape appears on the other's body. */
    ASSERT(element_index_(both, 0, 0, "wireframe", "chaser/ma_target") < 0);
    ASSERT(element_index_(both, 0, 0, "wireframe", "target/ma_chaser") < 0);
    ASSERT(line_(both, "scene_asset_verdict 0 drawable chaser chaser")
           != NULL);
    ASSERT(line_(both, "scene_asset_verdict 1 drawable target target")
           != NULL);

    /* The swap: each assembly bound to the other body. Both verdicts
     * are a digest mismatch and no wireframe is drawn at all. */
    ASSERT(line_(swapped, "scene_asset_verdict 0 mismatch chaser") != NULL);
    ASSERT(line_(swapped, "scene_asset_verdict 1 mismatch target") != NULL);
    ASSERT(kind_count_(swapped, 0, 0, "wireframe") == 0);
    ASSERT(kind_count_(swapped, 0, 0, "collider") == 0);
    /* The axes are still drawn, which is what says the scene ran and
     * refused the geometry rather than failing to build at all. */
    ASSERT(kind_count_(swapped, 0, 0, "axes") > 0);

    /* One binding alone leaves the other body without geometry, which
     * is what makes the arm above a statement about binding and not
     * about the assemblies being present. */
    ASSERT(element_index_(one, 0, 0, "wireframe", "target/ma_target") >= 0);
    ASSERT(kind_count_(one, 0, 0, "wireframe") == 1);

    g_arms++;
    printf("gate 3: two assemblies bound by body draw %u and %u vertices on"
           " their own bodies, the swap is refused by the digest and draws"
           " no wireframe, and one binding leaves the other body bare:"
           " OK\n", (unsigned)CHASER_VERTS, (unsigned)TARGET_VERTS);
}

/* Gate 4: the line of sight follows the recorded flag.
 *
 * Two detections, one whose flag changes during the episode and one
 * whose flag never rises. The arm requires both values to occur,
 * because a run in which everything was detected would let a line
 * always drawn pass.
 */
static void gate_detection_(const char *obs, const char *scene)
{
    char rows[256][256];
    int n, i;
    double flag[2][STEPS];
    unsigned chan[2] = { 0, 0 };
    int drawn_true = 0, drawn_false = 0;
    unsigned step;

    /* The flag channels, as the scene dump itself declares them, held
     * against the names the fixture wrote. */
    ASSERT(line_(scene, "scene_detection 0 alpha.trk ") != NULL);
    ASSERT(line_(scene, "scene_detection 1 beta.trk ") != NULL);
    n = lines_(scene, "scene_detection ", rows, 256);
    ASSERT(n == 2);
    for (i = 0; i < n; i++) {
        unsigned d = 0, det = 0, dx = 0, dy = 0, dz = 0, rg = 0;
        char base[64];
        ASSERT(sscanf(rows[i], "%u %63s %u %u %u %u %u", &d, base, &det,
                      &dx, &dy, &dz, &rg) == 7);
        ASSERT(d < 2);
        chan[d] = det;
    }
    ASSERT(chan[0] != chan[1]);

    n = lines_(obs, "obs 0 ", rows, 256);
    ASSERT(n == STEPS * OBS_TOTAL);
    for (i = 0; i < n; i++) {
        unsigned st = 0, ch = 0;
        char v[24];
        ASSERT(sscanf(rows[i], "%u %u %23s", &st, &ch, v) == 3);
        if (st < STEPS && ch == chan[0])
            flag[0][st] = hexd_(v);
        if (st < STEPS && ch == chan[1])
            flag[1][st] = hexd_(v);
    }

    for (step = 0; step < STEPS; step++) {
        static const char *const NAMES[2] = { "alpha.trk", "beta.trk" };
        int d;
        for (d = 0; d < 2; d++) {
            int idx = element_index_(scene, 0, step, "detection", NAMES[d]);
            int drawn;
            int want;
            uint64_t los[3], trk[3];
            int tidx;

            ASSERT(idx >= 0);        /* built at every step, drawn or not */
            drawn = segment_drawn_(scene, 0, step, idx);
            want = (flag[d][step] == 1.0) ? 1 : 0;
            if (drawn != want) {
                fprintf(stderr, "FAIL %s at step %u: drawn %d, the recorded"
                        " flag is %g\n", NAMES[d], step, drawn,
                        flag[d][step]);
                exit(1);
            }
            if (want)
                drawn_true++;
            else
                drawn_false++;

            /* The far end is the reconstructed point of this base at
             * this step, which the trajectory element of the same
             * name carries at the same index. */
            tidx = element_index_(scene, 0, step, "trajectory", NAMES[d]);
            ASSERT(tidx >= 0);
            vertex_(scene, 0, step, idx, 1, los);
            vertex_(scene, 0, step, tidx, step, trk);
            for (i = 0; i < 3; i++) {
                if (los[i] != trk[i]) {
                    fprintf(stderr, "FAIL %s at step %u: the line ends at"
                            " %016" PRIx64 " and the track's own point for"
                            " that step is %016" PRIx64 "\n", NAMES[d],
                            step, los[i], trk[i]);
                    exit(1);
                }
            }
            /* And a different step's point is a different point, so
             * the equality above is not an equality of everything. */
            if (step > 0) {
                uint64_t other[3];
                int differ = 0;
                vertex_(scene, 0, step, tidx, step - 1, other);
                for (i = 0; i < 3; i++)
                    differ += (other[i] != los[i]) ? 1 : 0;
                ASSERT(differ > 0);
            }
        }
    }
    /* Both outcomes occurred, so the comparison compared something. */
    ASSERT(drawn_true > 0);
    ASSERT(drawn_false > 0);
    /* And the flag moved within one detection rather than only across
     * the two, which is what tells a per-step read from a per-element
     * one. */
    ASSERT(flag[0][0] == 1.0);
    ASSERT(flag[0][LAST_DETECTED] == 1.0);
    ASSERT(flag[0][LAST_DETECTED + 1] == 0.0);
    ASSERT(flag[1][0] == 0.0);

    g_arms++;
    printf("gate 4: over %d steps and two detections, %d lines were drawn"
           " and %d were not, each matching its recorded flag, and each"
           " drawn line ends at its own base's reconstructed point:"
           " OK\n", STEPS, drawn_true, drawn_false);
}

/* Gate 5: the detection element's own toggle. */
static void gate_toggle_(const char *only, const char *all)
{
    ASSERT(element_index_(only, 0, 0, "detection", "alpha.trk") >= 0);
    ASSERT(element_index_(only, 0, 0, "detection", "beta.trk") >= 0);
    ASSERT(kind_count_(only, 0, 0, "detection") == 2);
    ASSERT(kind_count_(only, 0, 0, "wireframe") == 0);
    ASSERT(kind_count_(only, 0, 0, "trajectory") == 0);
    ASSERT(kind_count_(only, 0, 0, "axes") == 0);
    ASSERT(kind_count_(only, 0, 0, "collider") == 0);
    ASSERT(kind_count_(only, 0, 0, "velocity") == 0);
    ASSERT(line_(only, "scene_toggle detection 1") != NULL);
    ASSERT(line_(all, "scene_toggle detection 1") != NULL);
    ASSERT(element_index_(all, 0, 0, "detection", "alpha.trk") >= 0);
    ASSERT(element_index_(all, 0, 0, "wireframe", "chaser/ma_chaser") >= 0);

    g_arms++;
    printf("gate 5: `--elements detection` yields the two detections and no"
           " other element, and every element on yields both: OK\n");
}

int main(void)
{
    char cmd[8192], asm_args[1024], swap_args[1024];
    char *meta = NULL, *agents = NULL, *obs = NULL;
    char *both = NULL, *swapped = NULL, *one = NULL;
    char *only = NULL, *all = NULL, *again = NULL;
    size_t before_n = 0, after_n = 0;
    char *before = NULL, *after = NULL;

    if (!rl_libs_present_("test_rl_multiagent_view"))
        return 77;
    if (!rl_file_exists_(VIEWER)) {
        fprintf(stderr, "test_rl_multiagent_view: skip: %s not built\n",
                VIEWER);
        return 77;
    }
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/ma_chaser.k26mesh", CHASER_MESH);
    rl_write_file_(WORK_DIR "/ma_target.k26mesh", TARGET_MESH);
    rl_write_file_(WORK_DIR "/ma_chaser.k26asm", CHASER_ASM);
    rl_write_file_(WORK_DIR "/ma_target.k26asm", TARGET_ASM);
    rl_write_file_(WORK_DIR "/ma.kfl", MA_KFL);
    rl_compile_(WORK_DIR "/ma.kfl", WORK_DIR "/ma", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/ma.rlenv.so"));

    rl_run_or_die_(WORK_DIR "/ma --envs 1 --episodes 1 --seed 5"
                   " --out " WORK_DIR "/ma.k26epi > " WORK_DIR
                   "/run.log 2>&1");
    ASSERT(rl_file_exists_(WORK_DIR "/ma.k26epi"));
    before = slurp_(WORK_DIR "/ma.k26epi", &before_n);

#define EPISODE  WORK_DIR "/ma.k26epi"
#define CAMERA   " --camera orbit --camera-target chaser --orbit 30,20,3000"
#define ARTIFACT " --artifact " WORK_DIR "/ma.rlenv.so "

    /* The two bindings, and the two the swap gives. */
    snprintf(asm_args, sizeof asm_args,
             " --asset chaser=" WORK_DIR "/ma_chaser.k26asm"
             " --asset target=" WORK_DIR "/ma_target.k26asm ");
    snprintf(swap_args, sizeof swap_args,
             " --asset chaser=" WORK_DIR "/ma_target.k26asm"
             " --asset target=" WORK_DIR "/ma_chaser.k26asm ");

    run_viewer_("--dump meta " EPISODE, WORK_DIR "/meta.txt");
    meta = slurp_(WORK_DIR "/meta.txt", NULL);
    run_viewer_("--dump agents --episode 0 " EPISODE, WORK_DIR "/agents.txt");
    agents = slurp_(WORK_DIR "/agents.txt", NULL);
    run_viewer_("--dump obs --episode 0 " EPISODE, WORK_DIR "/obs.txt");
    obs = slurp_(WORK_DIR "/obs.txt", NULL);

    snprintf(cmd, sizeof cmd, "--dump scene --episode 0 --steps 0:1"
             " --frame chaser" CAMERA ARTIFACT "%s" EPISODE, asm_args);
    run_viewer_(cmd, WORK_DIR "/both.txt");
    both = slurp_(WORK_DIR "/both.txt", NULL);

    snprintf(cmd, sizeof cmd, "--dump scene --episode 0 --steps 0:1"
             " --frame chaser" CAMERA ARTIFACT "%s" EPISODE, swap_args);
    run_viewer_(cmd, WORK_DIR "/swapped.txt");
    swapped = slurp_(WORK_DIR "/swapped.txt", NULL);

    snprintf(cmd, sizeof cmd, "--dump scene --episode 0 --steps 0:1"
             " --frame chaser" CAMERA ARTIFACT
             " --asset target=" WORK_DIR "/ma_target.k26asm " EPISODE);
    run_viewer_(cmd, WORK_DIR "/one.txt");
    one = slurp_(WORK_DIR "/one.txt", NULL);

    snprintf(cmd, sizeof cmd, "--dump scene --episode 0 --steps 0:%d"
             " --frame chaser" CAMERA ARTIFACT "%s" EPISODE, STEPS,
             asm_args);
    run_viewer_(cmd, WORK_DIR "/full.txt");
    all = slurp_(WORK_DIR "/full.txt", NULL);

    snprintf(cmd, sizeof cmd, "--dump scene --episode 0 --steps 0:1"
             " --frame chaser --elements detection" CAMERA ARTIFACT "%s"
             EPISODE, asm_args);
    run_viewer_(cmd, WORK_DIR "/only.txt");
    only = slurp_(WORK_DIR "/only.txt", NULL);

    gate_grouping_(meta, agents);
    gate_reward_(obs, agents);
    gate_assets_(both, swapped, one);
    gate_detection_(obs, all);
    gate_toggle_(only, all);

    /* Gate 6: watching changed nothing. The recording is what it was
     * before the viewer first read it, and the same command run twice
     * gives the same bytes. */
    {
        size_t n2 = 0;
        snprintf(cmd, sizeof cmd, "--dump scene --episode 0 --steps 0:%d"
                 " --frame chaser" CAMERA ARTIFACT "%s" EPISODE, STEPS,
                 asm_args);
        run_viewer_(cmd, WORK_DIR "/full2.txt");
        again = slurp_(WORK_DIR "/full2.txt", &n2);
        ASSERT(strlen(all) == n2);
        ASSERT(memcmp(all, again, n2) == 0);

        after = slurp_(WORK_DIR "/ma.k26epi", &after_n);
        ASSERT(after_n == before_n);
        ASSERT(memcmp(before, after, before_n) == 0);
        g_arms++;
        printf("gate 6: the recording's %u bytes are unchanged by every run"
               " above and two identical runs agree over %u bytes: OK\n",
               (unsigned)before_n, (unsigned)n2);
    }

    free(meta);
    free(agents);
    free(obs);
    free(both);
    free(swapped);
    free(one);
    free(only);
    free(all);
    free(again);
    free(before);
    free(after);
    printf("test_rl_multiagent_view: %d gates passed\n", g_arms);
    return 0;
}
