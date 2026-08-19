/* test_rl_view.c: the episode viewer's panels.
 *
 * A panel whose numbers nothing checks is an assertion, so every
 * panel the viewer draws is driven here through its headless dump and
 * held against the episode file read directly through the format
 * library's reader. The viewer's two presenters share one model, so
 * what is checked here is what the window draws.
 *
 * The value panels are checked by generating the dump this file
 * should produce, from the reader, and comparing the whole text: a
 * single wrong value, a missing line, or an extra one fails, which a
 * field-by-field spot check would not.
 *
 * Gates:
 *   1. Metadata: file geometry, spec totals, channel names and kinds,
 *      action bounds and kinds, and the drawable trajectory grouping,
 *      all against the reader and a spec walk.
 *   2. Value panels: observations, rewards and returns, actions, and
 *      the timeline's per-step flags and applied dt, each compared as
 *      whole text against what the reader says.
 *   3. Trajectory reconstruction: the observer-relative point of each
 *      step equals range times direction, computed independently
 *      here, and the panel's label states what the reconstruction is
 *      rather than claiming an observer mode the spec never
 *      published.
 *   4. Scrubbing cost: a seek inside the loaded episode costs no
 *      reader traffic at all, and a seek that changes episode costs
 *      exactly one index lookup and one read. Counted, never timed.
 *   5. A file cut mid-write opens to its readable prefix, reports the
 *      truncation point, and still serves the episodes it does carry.
 *   6. A file whose channel names do not match the drawable
 *      convention yields no trajectory and still plots every channel
 *      in the observation panel.
 *   7. Re-simulation: with the producing artifact supplied, every
 *      recorded episode is rebuilt from its identity triple, recorded
 *      seed and recorded action stream, and the verdict is bitwise
 *      equality.
 *
 * Skips (77) when the sibling stack archives or the viewer binary are
 * not built.
 */
#define _GNU_SOURCE
#include <inttypes.h>
#include <stdarg.h>
#include <sys/wait.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_view_test"
#define VIEWER "../tools/k26rl_view/k26rl_view"

/* Two reset draws so episode-start frames carry a randomisation
 * record, a horizon that ends episodes inside a short run, and an
 * action that reaches the dynamics so the streams differ step to
 * step. A termination predicate that some episodes meet and a
 * terminal reward that only a terminated episode carries, so the
 * timeline has both endings to distinguish and the episode-return
 * curve has a terminal adjustment to fold in. */
static const char *const VIEW_KFL =
    "form RL_VIEW\n"
    "fn world view_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0,10.0) vel_z=0.0\n"
    "    let range_scale: double = 7.0e6\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 8\n"
    "        terminated when trk_range > 7.05e6\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 10.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    action gear discrete 3 default 1\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range / range_scale + push\n"
    "        terminal 5.0\n"
    "    end\n"
    "end\n"
    "end\n";

enum { VIEW_OBS = 5, VIEW_ACT = 2 };

/* The second fixture, for the panels that consume what the physics
 * work added: a body binding an assembly, so it has an attitude and a
 * wireframe to draw; a sensor with the truth published beside the
 * measurement, so the overlay has pairs to draw; and an angular rate,
 * so the attitude is not the identity forever.
 *
 * The noise is deliberately large against the observed range. A
 * sensor that changed nothing would let an overlay panel plot the
 * measured channel twice and pass, and the arm below asserts the two
 * halves differ for exactly that reason.
 *
 * The assembly it binds is written by the gate rather than taken from
 * the shipped assets, and its component is placed away from the body
 * origin AND rotated. A component at the origin would let a reader
 * that ignored the placement altogether produce the right wireframe,
 * which is a fixture multiplying a wrong answer by zero. */
static const char *const VIEW_ASM_KFL =
    "form RL_VIEW_ASM\n"
    "fn world view_asm_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft assembly=\"rich_box.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=7546.0 vel_z=0.0"
    " quat_w=1.0 omega_x=0.01 omega_y=0.02 omega_z=0.03\n"
    /* A second body binding a DIFFERENT assembly. Without it the
     * wireframe arm cannot tell a viewer that matches the asset's
     * name against the recording from one that takes whichever
     * assembly it happens to reach last, there being only one. */
    "    astro_body probe assembly=\"other_box.k26asm\" parent=earth"
    " pos_x=7.2e6 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=7440.0 vel_z=0.0"
    " quat_w=1.0 omega_x=0.0 omega_y=0.0 omega_z=0.005\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 6\n"
    "        terminated when episode.steps > 5\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    sensor nav\n"
    "        noise normal 0.0 500.0\n"
    "        latency 1\n"
    "    end\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "    end\n"
    "    observe craft from earth mode=geometric through nav with truth"
    " as trk\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward trk_range / 7.0e6\n"
    "    end\n"
    "end\n"
    "end\n";

/* ---- Running the viewer -------------------------------------------- */

static void run_viewer_(const char *args, const char *out_path)
{
    char cmd[2048];
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

/* Compare a produced dump against the text this gate says it should
 * be, reporting the first line that differs rather than only that it
 * did. */
static void expect_text_(const char *produced_path, const char *expected)
{
    size_t plen = 0;
    char *got = slurp_(produced_path, &plen);

    if (strcmp(got, expected) != 0) {
        const char *a = got, *b = expected;
        int line = 1;
        while (*a && *b) {
            const char *ae = strchr(a, '\n');
            const char *be = strchr(b, '\n');
            size_t alen = ae ? (size_t)(ae - a) : strlen(a);
            size_t blen = be ? (size_t)(be - b) : strlen(b);
            if (alen != blen || memcmp(a, b, alen) != 0) {
                fprintf(stderr, "dump differs at line %d\n  got:      %.*s\n"
                                "  expected: %.*s\n",
                        line, (int)alen, a, (int)blen, b);
                exit(1);
            }
            if (!ae || !be)
                break;
            a = ae + 1;
            b = be + 1;
            line++;
        }
        fprintf(stderr, "dump differs in length at line %d\n", line);
        exit(1);
    }
    free(got);
}

/* A line-oriented field fetch, for the panels checked by assertion
 * rather than by whole-text equality. */
static int find_line_(const char *text, const char *prefix, char *out,
                      size_t out_n)
{
    const char *p = text;
    size_t plen = strlen(prefix);

    while (p && *p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len >= plen && memcmp(p, prefix, plen) == 0) {
            if (len >= out_n)
                len = out_n - 1;
            memcpy(out, p, len);
            out[len] = '\0';
            return 1;
        }
        p = eol ? eol + 1 : NULL;
    }
    return 0;
}

static void append_(char **buf, size_t *len, size_t *cap, const char *fmt, ...)
{
    va_list ap;
    int n;

    for (;;) {
        va_start(ap, fmt);
        n = vsnprintf(*buf + *len, *cap - *len, fmt, ap);
        va_end(ap);
        ASSERT(n >= 0);
        if ((size_t)n < *cap - *len)
            break;
        *cap *= 2;
        *buf = realloc(*buf, *cap);
        ASSERT(*buf != NULL);
    }
    *len += (size_t)n;
}

static uint64_t bits_(double v)
{
    uint64_t b;
    memcpy(&b, &v, sizeof b);
    return b;
}

/* The episode header line every per-episode panel emits. */
static void append_episode_header_(char **b, size_t *l, size_t *c, uint32_t k,
                                   const K26RlEpisodeData *e, uint64_t seed)
{
    const char *reason = e->end_reason == K26RL_END_TERMINATED ? "terminated"
                       : e->end_reason == K26RL_END_TRUNCATED ? "truncated"
                       : e->end_reason == K26RL_END_FAULT ? "fault" : "unknown";
    uint32_t transitions = (e->end_reason == K26RL_END_FAULT && e->step_count)
                           ? e->step_count - 1 : e->step_count;
    append_(b, l, c, "episode %u %u %u %u %u %u %s %u %016" PRIx64 "\n", k,
            e->rekey_ordinal, e->env, e->episode, e->step_count, transitions,
            reason, (unsigned)e->fault_code, seed);
}

int main(void)
{
    if (!rl_libs_present_("test_rl_view"))
        return 77;
    if (!rl_file_exists_(VIEWER)) {
        fprintf(stderr, "test_rl_view: skip: %s not built\n", VIEWER);
        return 77;
    }
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/view.kfl", VIEW_KFL);
    rl_compile_(WORK_DIR "/view.kfl", WORK_DIR "/view", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/view.rlenv.so"));

    /* A batch run of the fixture: two environments, three episodes
     * each, so the file carries several complete episodes and the
     * timeline has boundaries to mark. */
    rl_run_or_die_(WORK_DIR "/view --envs 2 --episodes 3 --seed 11"
                   " --out " WORK_DIR "/view.k26epi > " WORK_DIR "/run.log 2>&1");
    ASSERT(rl_file_exists_(WORK_DIR "/view.k26epi"));

    K26RlEpisodeReader *rd = NULL;
    K26RlEpisodeInfo info;
    ASSERT(k26rl_episode_reader_open(WORK_DIR "/view.k26epi", &rd) == K26RL_OK);
    ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
    ASSERT(info.episode_count >= 4);
    ASSERT(info.obs_total == VIEW_OBS && info.act_total == VIEW_ACT);

    size_t cap = 1 << 16, len = 0;
    char *exp = malloc(cap);
    ASSERT(exp != NULL);
    exp[0] = '\0';

    /* ---- Gate 1: metadata ------------------------------------------ */
    {
        char line[512];
        char *got;

        run_viewer_("--dump meta " WORK_DIR "/view.k26epi",
                    WORK_DIR "/meta.txt");
        got = slurp_(WORK_DIR "/meta.txt", NULL);

        ASSERT(find_line_(got, "clean_close ", line, sizeof line));
        ASSERT(strcmp(line, "clean_close 1") == 0);
        snprintf(line, sizeof line, "episode_count %u", info.episode_count);
        {
            char found[512];
            ASSERT(find_line_(got, "episode_count ", found, sizeof found));
            ASSERT(strcmp(found, line) == 0);
        }
        snprintf(line, sizeof line, "governing_seed %016" PRIx64,
                 info.governing_seed);
        {
            char found[512];
            ASSERT(find_line_(got, "governing_seed ", found, sizeof found));
            ASSERT(strcmp(found, line) == 0);
        }
        {
            char found[512];
            snprintf(line, sizeof line, "spec obs_total %u", info.obs_total);
            ASSERT(find_line_(got, "spec obs_total ", found, sizeof found));
            ASSERT(strcmp(found, line) == 0);
            snprintf(line, sizeof line, "spec act_total %u", info.act_total);
            ASSERT(find_line_(got, "spec act_total ", found, sizeof found));
            ASSERT(strcmp(found, line) == 0);
        }
        /* The fixture's one as-bound observe publishes five channels
         * whose names follow the convention, so exactly one drawable
         * trajectory is grouped, over the first four of them. */
        {
            char found[512];
            ASSERT(find_line_(got, "trajectory_count ", found, sizeof found));
            ASSERT(strcmp(found, "trajectory_count 1") == 0);
            ASSERT(find_line_(got, "trajectory trk ", found, sizeof found));
            ASSERT(strcmp(found, "trajectory trk 0 1 2 3") == 0);
            ASSERT(find_line_(got, "channel 0 ", found, sizeof found));
            ASSERT(strcmp(found, "channel 0 0 trk_dir_x") == 0);
            ASSERT(find_line_(got, "channel 4 ", found, sizeof found));
            ASSERT(strcmp(found, "channel 4 0 trk_range_rate") == 0);
        }
        /* The declared action bounds and kinds the action panel draws. */
        {
            char found[512];
            ASSERT(find_line_(got, "action 0 ", found, sizeof found));
            ASSERT(strcmp(found, "action 0 0 0 bff0000000000000"
                                 " 3ff0000000000000") == 0);
            ASSERT(find_line_(got, "action 1 ", found, sizeof found));
            ASSERT(strcmp(found, "action 1 1 3 0000000000000000"
                                 " 4000000000000000") == 0);
        }
        /* The trajectory panel's label states the construction and
         * makes no claim about an observer mode, which the spec does
         * not publish. */
        {
            char found[1024];
            ASSERT(find_line_(got, "trajectory_label ", found, sizeof found));
            ASSERT(strstr(found, "observer-relative") != NULL);
            ASSERT(strstr(found, "geometric") != NULL);
        }
        free(got);
        printf("gate 1: metadata, spec, channels, bounds, grouping: OK\n");
    }

    /* ---- Gate 2: the value panels, as whole text ------------------- */
    {
        /* Observations. */
        len = 0;
        exp[0] = '\0';
        append_(&exp, &len, &cap, "k26rl_view dump 1\npanel obs\n");
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord, env, epi;
            uint64_t seed = 0;
            K26RlEpisodeData e;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &epi) == K26RL_OK);
            ASSERT(k26rl_episode_reader_seed(rd, ord, &seed) == K26RL_OK);
            ASSERT(k26rl_episode_read(rd, ord, env, epi, &e) == K26RL_OK);
            append_episode_header_(&exp, &len, &cap, k, &e, seed);
            for (uint32_t j = 0; j < info.obs_total; j++) {
                append_(&exp, &len, &cap, "initial_obs %u %u %016" PRIx64 "\n",
                        k, j, bits_(e.initial_obs[j]));
            }
            for (uint32_t j = 0; j < e.dr_count; j++) {
                append_(&exp, &len, &cap, "dr %u %u %u %016" PRIx64 "\n", k, j,
                        e.dr_tags[j], bits_(e.dr_values[j]));
            }
            for (uint32_t i = 0; i < e.step_count; i++) {
                for (uint32_t j = 0; j < info.obs_total; j++) {
                    append_(&exp, &len, &cap, "obs %u %u %u %016" PRIx64 "\n",
                            k, i, j, bits_(e.obs[(size_t)i * info.obs_total + j]));
                }
            }
            k26rl_episode_free(&e);
        }
        append_(&exp, &len, &cap, "end\n");
        run_viewer_("--dump obs " WORK_DIR "/view.k26epi", WORK_DIR "/obs.txt");
        expect_text_(WORK_DIR "/obs.txt", exp);

        /* Rewards, returns, and the terminal adjustment. */
        len = 0;
        exp[0] = '\0';
        append_(&exp, &len, &cap, "k26rl_view dump 1\npanel reward\n");
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord, env, epi;
            uint64_t seed = 0;
            K26RlEpisodeData e;
            double acc = 0.0;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &epi) == K26RL_OK);
            ASSERT(k26rl_episode_reader_seed(rd, ord, &seed) == K26RL_OK);
            ASSERT(k26rl_episode_read(rd, ord, env, epi, &e) == K26RL_OK);
            append_episode_header_(&exp, &len, &cap, k, &e, seed);
            for (uint32_t i = 0; i < e.step_count; i++) {
                double ret;
                acc += e.rewards[i];
                ret = acc;
                /* The terminal adjustment belongs to the return the
                 * last step reports, not to a step's own reward. */
                if (i + 1 == e.step_count)
                    ret += e.terminal_adjustments[0];
                append_(&exp, &len, &cap, "reward %u %u 0 %016" PRIx64 "\n", k,
                        i, bits_(e.rewards[i]));
                append_(&exp, &len, &cap, "return %u %u 0 %016" PRIx64 "\n", k,
                        i, bits_(ret));
            }
            append_(&exp, &len, &cap, "terminal_adj %u 0 %016" PRIx64 "\n", k,
                    bits_(e.terminal_adjustments[0]));
            k26rl_episode_free(&e);
        }
        append_(&exp, &len, &cap, "end\n");
        run_viewer_("--dump reward " WORK_DIR "/view.k26epi",
                    WORK_DIR "/reward.txt");
        expect_text_(WORK_DIR "/reward.txt", exp);

        /* Action traces, with the declared bounds the panel draws
         * beside them. */
        len = 0;
        exp[0] = '\0';
        append_(&exp, &len, &cap, "k26rl_view dump 1\npanel action\n");
        append_(&exp, &len, &cap, "action_bound 0 0 0 bff0000000000000"
                " 3ff0000000000000\n");
        append_(&exp, &len, &cap, "action_bound 1 1 3 0000000000000000"
                " 4000000000000000\n");
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord, env, epi;
            uint64_t seed = 0;
            K26RlEpisodeData e;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &epi) == K26RL_OK);
            ASSERT(k26rl_episode_reader_seed(rd, ord, &seed) == K26RL_OK);
            ASSERT(k26rl_episode_read(rd, ord, env, epi, &e) == K26RL_OK);
            append_episode_header_(&exp, &len, &cap, k, &e, seed);
            for (uint32_t i = 0; i < e.step_count; i++) {
                for (uint32_t j = 0; j < info.act_total; j++) {
                    append_(&exp, &len, &cap, "act %u %u %u %016" PRIx64 "\n",
                            k, i, j,
                            bits_(e.act[(size_t)i * info.act_total + j]));
                }
            }
            k26rl_episode_free(&e);
        }
        append_(&exp, &len, &cap, "end\n");
        run_viewer_("--dump action " WORK_DIR "/view.k26epi",
                    WORK_DIR "/action.txt");
        expect_text_(WORK_DIR "/action.txt", exp);

        /* The timeline's per-step flags and applied dt. */
        len = 0;
        exp[0] = '\0';
        append_(&exp, &len, &cap, "k26rl_view dump 1\npanel timeline\n");
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord, env, epi;
            uint64_t seed = 0;
            K26RlEpisodeData e;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &epi) == K26RL_OK);
            ASSERT(k26rl_episode_reader_seed(rd, ord, &seed) == K26RL_OK);
            ASSERT(k26rl_episode_read(rd, ord, env, epi, &e) == K26RL_OK);
            append_episode_header_(&exp, &len, &cap, k, &e, seed);
            for (uint32_t i = 0; i < e.step_count; i++) {
                append_(&exp, &len, &cap, "flag %u %u %08x\n", k, i, e.flags[i]);
                append_(&exp, &len, &cap, "dt %u %u %016" PRIx64 "\n", k, i,
                        bits_(e.applied_dt[i]));
            }
            k26rl_episode_free(&e);
        }
        append_(&exp, &len, &cap, "end\n");
        run_viewer_("--dump timeline " WORK_DIR "/view.k26epi",
                    WORK_DIR "/timeline.txt");
        expect_text_(WORK_DIR "/timeline.txt", exp);
        printf("gate 2: observation, reward, action and timeline panels"
               " equal the file's decoded content: OK\n");
    }

    /* ---- Gate 3: the trajectory reconstruction --------------------- */
    {
        len = 0;
        exp[0] = '\0';
        append_(&exp, &len, &cap, "k26rl_view dump 1\npanel traj\n");
        append_(&exp, &len, &cap, "trajectory_count 1\n");
        append_(&exp, &len, &cap, "trajectory_label observer-relative;"
                " exact under a geometric observe, otherwise an apparent"
                " direction at a geometric range\n");
        append_(&exp, &len, &cap, "trajectory trk 0 1 2 3\n");
        append_(&exp, &len, &cap, "trajectory_mode trk geometric\n");
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord, env, epi;
            uint64_t seed = 0;
            K26RlEpisodeData e;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &env, &epi) == K26RL_OK);
            ASSERT(k26rl_episode_reader_seed(rd, ord, &seed) == K26RL_OK);
            ASSERT(k26rl_episode_read(rd, ord, env, epi, &e) == K26RL_OK);
            append_episode_header_(&exp, &len, &cap, k, &e, seed);
            for (uint32_t i = 0; i < e.step_count; i++) {
                const double *o = e.obs + (size_t)i * info.obs_total;
                /* The reconstruction: the point at the recorded range
                 * along the recorded direction, computed here from
                 * the file rather than taken from the viewer. */
                append_(&exp, &len, &cap,
                        "traj trk %u %u %016" PRIx64 " %016" PRIx64
                        " %016" PRIx64 "\n", k, i,
                        bits_(o[0] * o[3]), bits_(o[1] * o[3]),
                        bits_(o[2] * o[3]));
            }
            k26rl_episode_free(&e);
        }
        append_(&exp, &len, &cap, "end\n");
        run_viewer_("--dump traj " WORK_DIR "/view.k26epi",
                    WORK_DIR "/traj.txt");
        expect_text_(WORK_DIR "/traj.txt", exp);
        printf("gate 3: the trajectory panel's points are range times"
               " direction, bitwise: OK\n");
    }

    /* ---- Gate 4: scrubbing cost, counted --------------------------- */
    {
        char *got;
        char line[256];
        uint64_t prev_lookups = 0, prev_reads = 0;
        int first = 1;

        run_viewer_("--dump scrub " WORK_DIR "/view.k26epi",
                    WORK_DIR "/scrub.txt");
        got = slurp_(WORK_DIR "/scrub.txt", NULL);

        /* Three samples per episode: within one episode the counters
         * must not move at all, and each change of episode must move
         * both by exactly one. */
        {
            const char *p = got;
            uint32_t seen_episode = 0xFFFFFFFFu;
            int samples_this_episode = 0;
            uint32_t episodes_seen = 0;
            while (p && *p) {
                const char *eol = strchr(p, '\n');
                if (strncmp(p, "scrub_sample ", 13) == 0) {
                    uint32_t k, step;
                    unsigned long long lk, rdc;
                    ASSERT(sscanf(p, "scrub_sample %u %u %llu %llu", &k, &step,
                                  &lk, &rdc) == 4);
                    if (k != seen_episode) {
                        if (!first) {
                            ASSERT(samples_this_episode == 3);
                            ASSERT(lk == prev_lookups + 1);
                            ASSERT(rdc == prev_reads + 1);
                        }
                        seen_episode = k;
                        samples_this_episode = 0;
                        episodes_seen++;
                        first = 0;
                    } else {
                        /* Same episode: a seek costs nothing. */
                        ASSERT(lk == prev_lookups);
                        ASSERT(rdc == prev_reads);
                    }
                    prev_lookups = lk;
                    prev_reads = rdc;
                    samples_this_episode++;
                }
                p = eol ? eol + 1 : NULL;
            }
            ASSERT(episodes_seen == info.episode_count);
        }
        snprintf(line, sizeof line, "scrub_index_lookups %u",
                 info.episode_count);
        {
            char found[256];
            ASSERT(find_line_(got, "scrub_index_lookups ", found, sizeof found));
            ASSERT(strcmp(found, line) == 0);
        }
        snprintf(line, sizeof line, "scrub_episode_reads %u",
                 info.episode_count);
        {
            char found[256];
            ASSERT(find_line_(got, "scrub_episode_reads ", found, sizeof found));
            ASSERT(strcmp(found, line) == 0);
        }
        free(got);
        printf("gate 4: %u episodes scrubbed at one index lookup and one read"
               " each, and nothing for a step inside one: OK\n",
               info.episode_count);
    }

    /* ---- Gate 7: re-simulation ------------------------------------- */
    {
        char *got;
        const char *p;
        uint32_t verdicts = 0;

        run_viewer_("--dump resim --artifact " WORK_DIR "/view.rlenv.so "
                    WORK_DIR "/view.k26epi", WORK_DIR "/resim.txt");
        got = slurp_(WORK_DIR "/resim.txt", NULL);
        p = got;
        while (p && *p) {
            const char *eol = strchr(p, '\n');
            if (strncmp(p, "resim_verdict ", 14) == 0) {
                char v[64];
                uint32_t k;
                ASSERT(sscanf(p, "resim_verdict %u %63s", &k, v) == 2);
                ASSERT(strcmp(v, "equal-bitwise") == 0);
                verdicts++;
            }
            p = eol ? eol + 1 : NULL;
        }
        ASSERT(verdicts == info.episode_count);
        free(got);
        printf("gate 7: %u episodes reconstructed from their identity triple"
               " and recorded action stream, every verdict bitwise equal:"
               " OK\n", info.episode_count);
    }

    k26rl_episode_reader_close(rd);

    /* ---- Gate 5: a file cut mid-write ------------------------------ */
    {
        char cmd[512];
        char *got;
        char found[256];
        FILE *src, *dst;
        long full;
        long keep;
        size_t n;
        char buf[4096];

        src = fopen(WORK_DIR "/view.k26epi", "rb");
        ASSERT(src != NULL);
        ASSERT(fseek(src, 0, SEEK_END) == 0);
        full = ftell(src);
        ASSERT(full > 0);
        ASSERT(fseek(src, 0, SEEK_SET) == 0);
        /* Two thirds of the file: past several complete episodes and
         * well before the index and trailer. */
        keep = full * 2 / 3;
        dst = fopen(WORK_DIR "/cut.k26epi", "wb");
        ASSERT(dst != NULL);
        while (keep > 0 && (n = fread(buf, 1, (size_t)(keep < (long)sizeof buf
                                                       ? keep : (long)sizeof buf),
                                      src)) > 0) {
            ASSERT(fwrite(buf, 1, n, dst) == n);
            keep -= (long)n;
        }
        fclose(src);
        fclose(dst);

        snprintf(cmd, sizeof cmd, "--dump meta " WORK_DIR "/cut.k26epi");
        run_viewer_(cmd, WORK_DIR "/cut.txt");
        got = slurp_(WORK_DIR "/cut.txt", NULL);
        ASSERT(find_line_(got, "clean_close ", found, sizeof found));
        ASSERT(strcmp(found, "clean_close 0") == 0);
        ASSERT(find_line_(got, "truncation_point ", found, sizeof found));
        /* The prefix is readable and its end is reported; the file
         * still carries whole episodes, so the viewer has something
         * to show. */
        {
            unsigned long long tp = 0;
            ASSERT(sscanf(found, "truncation_point %llu", &tp) == 1);
            ASSERT(tp <= (unsigned long long)(full * 2 / 3));
        }
        ASSERT(find_line_(got, "episode_count ", found, sizeof found));
        {
            unsigned ec = 0;
            ASSERT(sscanf(found, "episode_count %u", &ec) == 1);
            ASSERT(ec > 0);
            ASSERT(ec < info.episode_count);
        }
        free(got);
        printf("gate 5: a file cut mid-write opens to its readable prefix and"
               " reports the truncation point: OK\n");
    }

    /* ---- Gate 6: channel names outside the convention -------------- */
    {
        K26RlEpisodeWriter *w = NULL;
        K26RlEpisodeGeom g;
        uint8_t spec[256];
        uint32_t sl = 0;
        double obs[3] = { 1.0, 2.0, 3.0 };
        double act[1] = { 0.5 };
        double rew[1] = { 0.25 };
        double adj[1] = { 0.0 };
        char *got;
        char found[256];
        static const char *const names[3] = { "alpha", "beta", "gamma" };

        /* A hand-built spec whose channels are named nothing like the
         * convention, so no drawable triple can be grouped. */
        for (int i = 0; i < 3; i++) {
            uint32_t nl = (uint32_t)strlen(names[i]);
            spec[sl++] = (uint8_t)(K26RL_TAG_OBS_CHANNEL_NAME & 0xFF);
            spec[sl++] = (uint8_t)(K26RL_TAG_OBS_CHANNEL_NAME >> 8);
            spec[sl++] = (uint8_t)((4 + nl) & 0xFF);
            spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
            spec[sl++] = (uint8_t)i; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
            memcpy(spec + sl, names[i], nl);
            sl += nl;
        }
        /* The totals the panels size themselves from. */
        spec[sl++] = (uint8_t)(K26RL_TAG_OBS_TOTAL & 0xFF);
        spec[sl++] = (uint8_t)(K26RL_TAG_OBS_TOTAL >> 8);
        spec[sl++] = 4; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
        spec[sl++] = 3; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
        spec[sl++] = (uint8_t)(K26RL_TAG_ACT_TOTAL & 0xFF);
        spec[sl++] = (uint8_t)(K26RL_TAG_ACT_TOTAL >> 8);
        spec[sl++] = 4; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
        spec[sl++] = 1; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
        spec[sl++] = (uint8_t)(K26RL_TAG_AGENT_COUNT & 0xFF);
        spec[sl++] = (uint8_t)(K26RL_TAG_AGENT_COUNT >> 8);
        spec[sl++] = 4; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;
        spec[sl++] = 1; spec[sl++] = 0; spec[sl++] = 0; spec[sl++] = 0;

        g.n_envs = 1;
        g.agent_count = 1;
        g.obs_total = 3;
        g.act_total = 1;
        g.steps_per_chunk = 8;
        g.dr_max = 0;
        ASSERT(k26rl_episode_writer_open(WORK_DIR "/named.k26epi", &g, 7, 0,
                                         "3.2", "test", spec, sl, &w) ==
               K26RL_OK);
        ASSERT(k26rl_episode_writer_start(w, 0, 0, obs, NULL, NULL, 0) ==
               K26RL_OK);
        ASSERT(k26rl_episode_writer_step(w, 0, obs, act, rew,
                                         K26RL_FLAG_TERMINATED, 0.1) ==
               K26RL_OK);
        ASSERT(k26rl_episode_writer_end(w, 0, K26RL_END_TERMINATED, 0, adj) ==
               K26RL_OK);
        ASSERT(k26rl_episode_writer_close(w) == K26RL_OK);

        run_viewer_("--dump meta " WORK_DIR "/named.k26epi",
                    WORK_DIR "/named.txt");
        got = slurp_(WORK_DIR "/named.txt", NULL);
        ASSERT(find_line_(got, "trajectory_count ", found, sizeof found));
        ASSERT(strcmp(found, "trajectory_count 0") == 0);
        /* Every channel is still published to the observation panel:
         * a name outside the convention is undrawable in three
         * dimensions, not invisible. */
        ASSERT(find_line_(got, "channel 0 ", found, sizeof found));
        ASSERT(strcmp(found, "channel 0 0 alpha") == 0);
        ASSERT(find_line_(got, "channel 1 ", found, sizeof found));
        ASSERT(strcmp(found, "channel 1 0 beta") == 0);
        ASSERT(find_line_(got, "channel 2 ", found, sizeof found));
        ASSERT(strcmp(found, "channel 2 0 gamma") == 0);
        free(got);

        run_viewer_("--dump traj " WORK_DIR "/named.k26epi",
                    WORK_DIR "/named_traj.txt");
        got = slurp_(WORK_DIR "/named_traj.txt", NULL);
        ASSERT(strstr(got, "\ntraj ") == NULL);
        free(got);

        run_viewer_("--dump obs " WORK_DIR "/named.k26epi",
                    WORK_DIR "/named_obs.txt");
        got = slurp_(WORK_DIR "/named_obs.txt", NULL);
        ASSERT(find_line_(got, "obs 0 0 2 ", found, sizeof found));
        free(got);
        printf("gate 6: channels outside the drawable convention plot in the"
               " observation panel and nowhere in three dimensions: OK\n");
    }

    /* ---- Gate 8: the world-frame panel ----------------------------- */
    {
        char *got;
        char found[256];
        void *so;
        RlSurface vs;
        K26RlEnv *env = NULL;
        K26RlEpisodeReader *rd2 = NULL;
        K26RlEpisodeInfo i2;
        K26RlEpisodeData e0;
        uint32_t ord, envi, epi;
        uint64_t seed = 0;
        int32_t bwant;
        double *bodies, *act;
        uint32_t nb, lines = 0;

        /* Without an artifact the panel says so: the file alone
         * records observation channels and cannot produce a world. */
        run_viewer_("--dump world " WORK_DIR "/view.k26epi",
                    WORK_DIR "/world_none.txt");
        got = slurp_(WORK_DIR "/world_none.txt", NULL);
        ASSERT(find_line_(got, "world unavailable ", found, sizeof found));
        ASSERT(strstr(got, "\nworld 0 ") == NULL);
        free(got);

        /* The channel modes are now facts read from the spec, not
         * conditions the reader must resolve. */
        run_viewer_("--dump traj --episode 0 --steps 0:1 "
                    WORK_DIR "/view.k26epi", WORK_DIR "/traj_mode.txt");
        got = slurp_(WORK_DIR "/traj_mode.txt", NULL);
        ASSERT(find_line_(got, "trajectory_mode trk ", found, sizeof found));
        ASSERT(strcmp(found, "trajectory_mode trk geometric") == 0);
        free(got);

        /* With an artifact, the panel's values must equal the
         * artifact's own getter, driven here independently along the
         * same rebuild the viewer performs. */
        run_viewer_("--dump world --episode 1 --artifact "
                    WORK_DIR "/view.rlenv.so " WORK_DIR "/view.k26epi",
                    WORK_DIR "/world.txt");
        got = slurp_(WORK_DIR "/world.txt", NULL);
        ASSERT(find_line_(got, "body 0 ", found, sizeof found));
        ASSERT(strcmp(found, "body 0 earth") == 0);
        ASSERT(find_line_(got, "body 1 ", found, sizeof found));
        ASSERT(strcmp(found, "body 1 craft") == 0);

        ASSERT(k26rl_episode_reader_open(WORK_DIR "/view.k26epi", &rd2) ==
               K26RL_OK);
        ASSERT(k26rl_episode_reader_info(rd2, &i2) == K26RL_OK);
        ASSERT(k26rl_episode_reader_at(rd2, 1, &ord, &envi, &epi) == K26RL_OK);
        ASSERT(k26rl_episode_reader_seed(rd2, ord, &seed) == K26RL_OK);
        ASSERT(k26rl_episode_read(rd2, ord, envi, epi, &e0) == K26RL_OK);

        so = rl_dlopen_(WORK_DIR "/view.rlenv.so");
        rl_resolve_surface_(so, &vs);
        ASSERT(vs.abi_version() == K26RL_ABI_VERSION);
        ASSERT(vs.create(seed, i2.n_envs, &env) == K26RL_OK);
        for (uint32_t k = 0; k < epi; k++)
            ASSERT(vs.reset(env) == K26RL_OK);
        bwant = vs.bodies(env, K26RL_BODY_REF_ORIGIN, NULL, 0);
        ASSERT(bwant > 0);
        nb = (uint32_t)bwant / (i2.n_envs * 6u);
        bodies = malloc(sizeof(double) * (size_t)bwant);
        act = malloc(sizeof(double) * (size_t)i2.n_envs * i2.act_total);
        ASSERT(bodies && act);

        for (uint32_t t = 0; t < e0.step_count; t++) {
            for (uint32_t j = 0; j < i2.n_envs; j++) {
                memcpy(act + (size_t)j * i2.act_total,
                       e0.act + (size_t)t * i2.act_total,
                       sizeof(double) * i2.act_total);
            }
            ASSERT(vs.step(env, act) == K26RL_OK);
            ASSERT(vs.bodies(env, K26RL_BODY_REF_ORIGIN, bodies,
                             (uint32_t)bwant) == bwant);
            for (uint32_t b = 0; b < nb; b++) {
                char key[64], line[512], want[512];
                const double *src = bodies +
                    ((size_t)envi * nb + b) * 6;
                int off;
                snprintf(key, sizeof key, "world 1 %u %u ", t, b);
                ASSERT(find_line_(got, key, line, sizeof line));
                off = snprintf(want, sizeof want, "world 1 %u %u", t, b);
                for (int c = 0; c < 6; c++) {
                    off += snprintf(want + off, sizeof want - (size_t)off,
                                    " %016" PRIx64, bits_(src[c]));
                }
                ASSERT(strcmp(line, want) == 0);
                lines++;
            }
        }
        ASSERT(lines == e0.step_count * nb);

        free(bodies);
        free(act);
        k26rl_episode_free(&e0);
        vs.destroy(env);
        dlclose(so);
        k26rl_episode_reader_close(rd2);
        free(got);
        printf("gate 8: %u world-frame rows equal the artifact's own body"
               " getter bitwise, and the channel modes are read from the"
               " spec: OK\n", lines);
    }

    /* ---- Gate 9: the panels the physics work made possible --------- *
     *
     * Three panels, and for each the fixture that would make its arm
     * vacuous, ruled out.
     *
     *   An attitude panel over bodies that never turn would pass while
     *   printing the identity quaternion forever, so the fixture spins
     *   its craft on all three axes and the arm asserts the values
     *   move between steps and differ between the two bodies before it
     *   compares any of them.
     *
     *   An overlay panel over a sensor that changed nothing would pass
     *   while plotting the measured channel twice, so the fixture's
     *   noise is large against the observed range and the arm asserts
     *   the two halves of every pair differ.
     *
     *   A wireframe arm that compared the panel against the panel
     *   would pass for any geometry, so the expected vertex count and
     *   the expected vertex values are read here out of the mesh file
     *   itself, and the digest the panel checks against is the one the
     *   artifact recorded rather than the one the asset computes.
     */
    {
        char *got;
        char found[512];
        void *so;
        RlSurface vs;
        K26RlEnv *env = NULL;
        K26RlEpisodeReader *rd2 = NULL;
        K26RlEpisodeInfo i2;
        K26RlEpisodeData e0;
        uint32_t ord, envi, epi;
        uint64_t seed = 0;
        int32_t await;
        double *atts, *act;
        uint32_t na, lines = 0, pick;
        char epsel[256];
        uint8_t blob[16384];
        int32_t blen;

        /* The assets travel to the work directory so the fixture can
         * name them beside itself, and so the mismatch arm below has
         * a copy of its own to spoil. */
        rl_run_or_die_("cp examples/assets/calibration_box.k26mesh "
                       WORK_DIR "/");
        /* The wireframe fixture, built to be one that a reader with
         * any of four plausible defects gets wrong.
         *
         * TWO components with TWO meshes, because with one of each a
         * reader that rebased the second mesh's vertex indices by the
         * running total and one that did not agree, and so do one
         * that applies each mesh's own placement and one that applies
         * the first placement to everything.
         *
         * The first component names its mesh BEFORE its placement,
         * because the compiler parses the whole block and derives
         * afterwards, so key order inside a block is free for it; a
         * reader that snapshotted the placement at the `mesh` line
         * would lose it here and draw raw mesh coordinates while the
         * digest, which is over bytes, still matched.
         *
         * Its path is QUOTED, because the compiler's tokenizer strips
         * quotes and a reader that kept them would refuse a file the
         * compiler compiled.
         *
         * And the second mesh writes its FACES before its VERTICES,
         * because the compiler counts vertices over the whole file
         * before bounding any index, and a reader bounding by what it
         * has seen so far would refuse that file too. */
        rl_write_file_(WORK_DIR "/faces_first.k26mesh",
            "# the same box, with its faces written before its vertices\n"
            "f 1 4 3\n"
            "f 1 3 2\n"
            "f 5 6 7\n"
            "f 5 7 8\n"
            "f 1 2 6\n"
            "f 1 6 5\n"
            "f 2 3 7\n"
            "f 2 7 6\n"
            "f 3 4 8\n"
            "f 3 8 7\n"
            "f 4 1 5\n"
            "f 4 5 8\n"
            "v -1.0 -0.5 -0.5\n"
            "v 1.0 -0.5 -0.5\n"
            "v 1.0 0.5 -0.5\n"
            "v -1.0 0.5 -0.5\n"
            "v -1.0 -0.5 0.5\n"
            "v 1.0 -0.5 0.5\n"
            "v 1.0 0.5 0.5\n"
            "v -1.0 0.5 0.5\n");
        rl_write_file_(WORK_DIR "/rich_box.k26asm",
            "assembly rich_box\n"
            "    frame x_to_port\n"
            "    provenance mass \"gate fixture, not a craft\" computed\n"
            "    component first\n"
            "        mass 1000.0\n"
            "        mesh \"calibration_box.k26mesh\"\n"
            "        at 0.25 -0.5 0.75\n"
            "        rotate 0.70710678118654752 0.0 0.70710678118654752"
            " 0.0\n"
            "    end\n"
            "    component second\n"
            "        mass 400.0\n"
            "        at -6.0 1.5 0.5\n"
            "        mesh faces_first.k26mesh\n"
            "    end\n"
            "end\n");
        rl_write_file_(WORK_DIR "/other_box.k26asm",
            "assembly other_box\n"
            "    frame x_to_port\n"
            "    provenance mass \"gate fixture, not a craft\" computed\n"
            "    component hull\n"
            "        mass 500.0\n"
            "        at -1.5 0.25 0.0\n"
            "        mesh calibration_box.k26mesh\n"
            "    end\n"
            "end\n");
        rl_write_file_(WORK_DIR "/view_asm.kfl", VIEW_ASM_KFL);
        rl_compile_(WORK_DIR "/view_asm.kfl", WORK_DIR "/view_asm", WORK_DIR);
        /* Three environments, so the episode this arm reads is not
         * environment 0. A single-environment file would let a panel
         * that read the wrong environment's slice pass, the two
         * slices being the same one. */
        rl_run_or_die_(WORK_DIR "/view_asm --envs 3 --episodes 2 --seed 9 "
                       "--out " WORK_DIR "/view_asm.k26epi");

        /* -- the attitude panel -- */
        run_viewer_("--dump attitude " WORK_DIR "/view_asm.k26epi",
                    WORK_DIR "/att_none.txt");
        got = slurp_(WORK_DIR "/att_none.txt", NULL);
        ASSERT(find_line_(got, "attitude unavailable ", found, sizeof found));
        ASSERT(strstr(got, "\nattitude 0 ") == NULL);
        free(got);

        ASSERT(k26rl_episode_reader_open(WORK_DIR "/view_asm.k26epi", &rd2) ==
               K26RL_OK);
        ASSERT(k26rl_episode_reader_info(rd2, &i2) == K26RL_OK);
        /* The first indexed episode that did not come from
         * environment 0, and the arm asserts it found one rather than
         * assuming: without that the environment slice below is
         * untested. */
        pick = UINT32_MAX;
        for (uint32_t k = 0; k < i2.episode_count; k++) {
            ASSERT(k26rl_episode_reader_at(rd2, k, &ord, &envi, &epi) ==
                   K26RL_OK);
            if (envi != 0) { pick = k; break; }
        }
        ASSERT(pick != UINT32_MAX);
        ASSERT(envi != 0);
        printf("gate 9: the attitude arm reads episode %u, which is"
               " environment %u and not environment 0: OK\n", pick, envi);
        ASSERT(k26rl_episode_reader_seed(rd2, ord, &seed) == K26RL_OK);
        ASSERT(k26rl_episode_read(rd2, ord, envi, epi, &e0) == K26RL_OK);

        snprintf(epsel, sizeof epsel, "--dump attitude --episode %u"
                 " --artifact " WORK_DIR "/view_asm.rlenv.so "
                 WORK_DIR "/view_asm.k26epi", pick);
        run_viewer_(epsel, WORK_DIR "/att.txt");
        got = slurp_(WORK_DIR "/att.txt", NULL);
        ASSERT(find_line_(got, "body 1 ", found, sizeof found));
        ASSERT(strcmp(found, "body 1 craft") == 0);

        so = rl_dlopen_(WORK_DIR "/view_asm.rlenv.so");
        rl_resolve_surface_(so, &vs);
        ASSERT(vs.create(seed, i2.n_envs, &env) == K26RL_OK);
        for (uint32_t k = 0; k < epi; k++)
            ASSERT(vs.reset(env) == K26RL_OK);
        await = vs.attitudes(env, NULL, 0);
        ASSERT(await > 0);
        na = (uint32_t)await / (i2.n_envs * 7u);
        atts = malloc(sizeof(double) * (size_t)await);
        act = malloc(sizeof(double) * (size_t)i2.n_envs * i2.act_total);
        ASSERT(atts && act);

        {
            /* The fixture has to be one an identity panel would fail:
             * two bodies whose attitudes differ, and a craft whose
             * own attitude moves from step to step. */
            double first[7], later[7], earth0[7];
            for (uint32_t t = 0; t < 3; t++) {
                for (uint32_t j = 0; j < i2.n_envs; j++) {
                    memcpy(act + (size_t)j * i2.act_total,
                           e0.act + (size_t)t * i2.act_total,
                           sizeof(double) * i2.act_total);
                }
                ASSERT(vs.step(env, act) == K26RL_OK);
                ASSERT(vs.attitudes(env, atts, (uint32_t)await) == await);
                if (t == 0) {
                    memcpy(first, atts + ((size_t)envi * na + 1) * 7,
                           sizeof first);
                    memcpy(earth0, atts + ((size_t)envi * na + 0) * 7,
                           sizeof earth0);
                }
                if (t == 2)
                    memcpy(later, atts + ((size_t)envi * na + 1) * 7,
                           sizeof later);
            }
            ASSERT(memcmp(first, later, sizeof first) != 0);
            ASSERT(memcmp(first, earth0, sizeof first) != 0);
            printf("gate 9: the fixture's craft turns (quaternion moves "
                   "between steps) and differs from its parent: OK\n");
        }

        /* Re-create and drive from the top, so the comparison below
         * runs the same rebuild the viewer runs. */
        vs.destroy(env);
        ASSERT(vs.create(seed, i2.n_envs, &env) == K26RL_OK);
        for (uint32_t k = 0; k < epi; k++)
            ASSERT(vs.reset(env) == K26RL_OK);
        for (uint32_t t = 0; t < e0.step_count; t++) {
            for (uint32_t j = 0; j < i2.n_envs; j++) {
                memcpy(act + (size_t)j * i2.act_total,
                       e0.act + (size_t)t * i2.act_total,
                       sizeof(double) * i2.act_total);
            }
            ASSERT(vs.step(env, act) == K26RL_OK);
            ASSERT(vs.attitudes(env, atts, (uint32_t)await) == await);
            for (uint32_t b = 0; b < na; b++) {
                char key[64], line[512], want[512];
                const double *src = atts + ((size_t)envi * na + b) * 7;
                int off;
                snprintf(key, sizeof key, "attitude %u %u %u ", pick,
                         t, b);
                ASSERT(find_line_(got, key, line, sizeof line));
                off = snprintf(want, sizeof want, "attitude %u %u %u",
                               pick, t, b);
                for (int c = 0; c < 7; c++) {
                    off += snprintf(want + off, sizeof want - (size_t)off,
                                    " %016" PRIx64, bits_(src[c]));
                }
                ASSERT(strcmp(line, want) == 0);
                lines++;
            }
        }
        ASSERT(lines == e0.step_count * na);
        printf("gate 9: %u attitude rows equal the artifact's own attitude"
               " getter bitwise: OK\n", lines);

        /* -- the overlay -- */
        {
            char *ov;
            uint32_t pairs = 0, drawn = 0;
            uint32_t off = 0;

            ASSERT((blen = vs.spec(env, blob, sizeof blob)) > 0);
            snprintf(epsel, sizeof epsel, "--dump overlay --episode %u "
                     WORK_DIR "/view_asm.k26epi", pick);
            run_viewer_(epsel, WORK_DIR "/ov.txt");
            ov = slurp_(WORK_DIR "/ov.txt", NULL);
            /* Every pair the source tag declares is a pair the panel
             * draws, read out of the spec here rather than out of the
             * panel's own output. */
            while (off + 6 <= (uint32_t)blen) {
                uint16_t tag = rl_get_u16_(blob + off);
                uint32_t l = rl_get_u32_(blob + off + 2);
                const uint8_t *v = blob + off + 6;
                if (tag == K26RL_TAG_OBS_CHANNEL_SOURCE && l >= 10 &&
                    rl_get_u16_(v + 4) == K26RL_OBS_SOURCE_MEASURED &&
                    rl_get_u32_(v + 6) != K26RL_OBS_PAIR_NONE) {
                    char key[64];
                    snprintf(key, sizeof key, "overlay_pair %u %u ",
                             rl_get_u32_(v), rl_get_u32_(v + 6));
                    ASSERT(find_line_(ov, key, found, sizeof found));
                    pairs++;
                }
                off += 6 + l;
            }
            ASSERT(pairs == VIEW_OBS);
            /* And the values are the file's, at the channels the tag
             * named. A pair whose halves agreed everywhere would make
             * this arm vacuous, so the two are asserted to differ. */
            for (uint32_t t = 0; t < e0.step_count; t++) {
                for (uint32_t p = 0; p < pairs; p++) {
                    char key[64], line[512], want[512];
                    const double *o = e0.obs + (size_t)t * i2.obs_total;
                    snprintf(key, sizeof key, "overlay %u %u %u ", pick,
                             t, p);
                    ASSERT(find_line_(ov, key, line, sizeof line));
                    snprintf(want, sizeof want, "overlay %u %u %u %016"
                             PRIx64 " %016" PRIx64, pick, t, p,
                             bits_(o[p]), bits_(o[p + VIEW_OBS]));
                    ASSERT(strcmp(line, want) == 0);
                    if (o[p] != o[p + VIEW_OBS])
                        drawn++;
                }
            }
            ASSERT(drawn > 0);
            printf("gate 9: %u paired channels, %u of %u overlaid values"
                   " differ between the measurement and the truth: OK\n",
                   pairs, drawn, e0.step_count * pairs);
            free(ov);
        }

        free(atts);
        free(act);
        k26rl_episode_free(&e0);
        vs.destroy(env);
        dlclose(so);
        k26rl_episode_reader_close(rd2);
        free(got);

        /* -- the wireframe, and the digest that licenses drawing it -- */
        {
            char *wf;

            run_viewer_("--dump wireframe " WORK_DIR "/view_asm.k26epi",
                        WORK_DIR "/wf_none.txt");
            wf = slurp_(WORK_DIR "/wf_none.txt", NULL);
            /* Two assemblies in the recording, and the panel picks
             * the one the asset names. */
            ASSERT(find_line_(wf, "assembly 1 rich_box ", found,
                              sizeof found));
            ASSERT(find_line_(wf, "assembly 2 other_box ", found,
                              sizeof found));
            ASSERT(find_line_(wf, "wireframe unavailable ", found,
                              sizeof found));
            ASSERT(strstr(wf, "\nwireframe_vertex ") == NULL);
            free(wf);

            /* The expected geometry comes from the mesh files and
             * the assembly's own placements, walked here in the same
             * order the assembly declares its components. A panel
             * drawing nothing fails on the counts; a panel drawing
             * the second mesh at the first placement, or without
             * rebasing its indices, fails on the values. */
            struct { const char *mesh; double at[3]; double q[4]; } comp[2];
            uint32_t verts = 0, tris = 0, edges = 0;
            uint32_t vbase = 0;
            char *wf2;

            comp[0].mesh = WORK_DIR "/calibration_box.k26mesh";
            comp[0].at[0] = 0.25; comp[0].at[1] = -0.5; comp[0].at[2] = 0.75;
            comp[0].q[0] = 0.70710678118654752; comp[0].q[1] = 0.0;
            comp[0].q[2] = 0.70710678118654752; comp[0].q[3] = 0.0;
            comp[1].mesh = WORK_DIR "/faces_first.k26mesh";
            comp[1].at[0] = -6.0; comp[1].at[1] = 1.5; comp[1].at[2] = 0.5;
            comp[1].q[0] = 1.0; comp[1].q[1] = 0.0;
            comp[1].q[2] = 0.0; comp[1].q[3] = 0.0;

            run_viewer_("--dump wireframe --asset "
                        WORK_DIR "/rich_box.k26asm "
                        WORK_DIR "/view_asm.k26epi", WORK_DIR "/wf.txt");
            wf2 = slurp_(WORK_DIR "/wf.txt", NULL);
            ASSERT(find_line_(wf2, "wireframe_digest 0 match ", found,
                              sizeof found));
            ASSERT(find_line_(wf2, "wireframe_body ", found, sizeof found));
            ASSERT(strcmp(found, "wireframe_body 0 1 craft") == 0);
            ASSERT(find_line_(wf2, "wireframe_meshes ", found,
                              sizeof found));
            ASSERT(strcmp(found, "wireframe_meshes 0 2") == 0);

            for (int ci = 0; ci < 2; ci++) {
                char *mesh = slurp_(comp[ci].mesh, NULL);
                const char *line;
                uint32_t local = 0, ltris = 0;
                uint32_t ea[64], eb[64], ne = 0;

                for (line = mesh; *line; ) {
                    if (line[0] == 'v' && line[1] == ' ') {
                        double v[3], t[3], w[3], qw, qx, qy, qz;
                        char key[64], got_line[256], want[256];

                        ASSERT(sscanf(line + 2, "%lf %lf %lf", &v[0], &v[1],
                                      &v[2]) == 3);
                        qw = comp[ci].q[0]; qx = comp[ci].q[1];
                        qy = comp[ci].q[2]; qz = comp[ci].q[3];
                        /* v + 2w(u x v) + 2u x (u x v), written out
                         * here rather than taken from a library. */
                        t[0] = 2.0 * (qy * v[2] - qz * v[1]);
                        t[1] = 2.0 * (qz * v[0] - qx * v[2]);
                        t[2] = 2.0 * (qx * v[1] - qy * v[0]);
                        w[0] = v[0] + qw * t[0] + (qy * t[2] - qz * t[1]);
                        w[1] = v[1] + qw * t[1] + (qz * t[0] - qx * t[2]);
                        w[2] = v[2] + qw * t[2] + (qx * t[1] - qy * t[0]);
                        snprintf(key, sizeof key, "wireframe_vertex 0 %u ",
                                 vbase + local);
                        ASSERT(find_line_(wf2, key, got_line,
                                          sizeof got_line));
                        snprintf(want, sizeof want, "wireframe_vertex 0 %u "
                                 "%016" PRIx64 " %016" PRIx64
                                 " %016" PRIx64, vbase + local,
                                 bits_(w[0] + comp[ci].at[0]),
                                 bits_(w[1] + comp[ci].at[1]),
                                 bits_(w[2] + comp[ci].at[2]));
                        ASSERT(strcmp(got_line, want) == 0);
                        local++;
                    }
                    while (*line && *line != '\n') line++;
                    if (*line) line++;
                }
                /* The edge set of this mesh, rebased by the vertices
                 * the meshes before it contributed. */
                for (line = mesh; *line; ) {
                    if (line[0] == 'f' && line[1] == ' ') {
                        unsigned fv[3];
                        ASSERT(sscanf(line + 2, "%u %u %u", &fv[0], &fv[1],
                                      &fv[2]) == 3);
                        for (int c = 0; c < 3; c++) {
                            uint32_t x = vbase + fv[c] - 1u;
                            uint32_t y = vbase + fv[(c + 1) % 3] - 1u;
                            uint32_t lo = x < y ? x : y;
                            uint32_t hi = x < y ? y : x;
                            uint32_t j;
                            for (j = 0; j < ne; j++) {
                                if (ea[j] == lo && eb[j] == hi)
                                    break;
                            }
                            if (j == ne) {
                                ASSERT(ne < 64);
                                ea[ne] = lo;
                                eb[ne] = hi;
                                ne++;
                            }
                        }
                        ltris++;
                    }
                    while (*line && *line != '\n') line++;
                    if (*line) line++;
                }
                for (uint32_t j = 0; j < ne; j++) {
                    char key[64], line2[128], want[128];
                    snprintf(key, sizeof key, "wireframe_edge 0 %u ",
                             edges + j);
                    ASSERT(find_line_(wf2, key, line2, sizeof line2));
                    snprintf(want, sizeof want, "wireframe_edge 0 %u %u %u",
                             edges + j, ea[j], eb[j]);
                    ASSERT(strcmp(line2, want) == 0);
                }
                ASSERT(local > 0 && ltris > 0);
                verts += local;
                tris += ltris;
                edges += ne;
                vbase += local;
                free(mesh);
            }
            ASSERT(find_line_(wf2, "wireframe_counts ", found,
                              sizeof found));
            {
                char want[128];
                snprintf(want, sizeof want, "wireframe_counts 0 %u %u %u",
                         verts, edges, tris);
                ASSERT(strcmp(found, want) == 0);
            }
            wf = wf2;
            printf("gate 9: %u vertices and %u edges over two components and"
                   " two meshes are the asset's own, each carried by its own"
                   " placement, and the digest is the recording's: OK\n",
                   verts, edges);
            free(wf);

            /* An asset whose bytes are not the bytes that flew is
             * reported and not drawn. One byte of the mesh moves,
             * which the digest covers because the mesh's bytes are
             * inside it. */
            rl_run_or_die_("sed 's/^v -1.0 -0.5 -0.5/v -1.0 -0.5 -0.4/' "
                           WORK_DIR "/faces_first.k26mesh > "
                           WORK_DIR "/spoiled.k26mesh");
            rl_run_or_die_("sed 's/mesh faces_first.k26mesh/"
                           "mesh spoiled.k26mesh/' "
                           WORK_DIR "/rich_box.k26asm > "
                           WORK_DIR "/spoiled.k26asm");
            run_viewer_("--dump wireframe --asset "
                        WORK_DIR "/spoiled.k26asm "
                        WORK_DIR "/view_asm.k26epi", WORK_DIR "/wf_bad.txt");
            wf = slurp_(WORK_DIR "/wf_bad.txt", NULL);
            ASSERT(find_line_(wf, "wireframe_digest 0 mismatch ", found,
                              sizeof found));
            ASSERT(strstr(wf, "\nwireframe_vertex 0 ") == NULL);
            ASSERT(strstr(wf, "\nwireframe_counts 0 ") == NULL);
            printf("gate 9: an asset whose bytes are not the recording's is"
                   " reported and not drawn: OK\n");
            free(wf);
        }
    }

    free(exp);
    printf("test_rl_view: 9 gates passed\n");
    return 0;
}
