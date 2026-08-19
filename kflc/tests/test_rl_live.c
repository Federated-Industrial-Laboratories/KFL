/* test_rl_live.c: watching a simulation while it runs.
 *
 * The viewer reads a finished recording from a file and a running one
 * from the telemetry ring the serving surface publishes into. This
 * gates the second: that watching cannot change the run, that the
 * picture actually advances while the run advances, and that a
 * watcher the producer outran says what went past it.
 *
 * The isolation arm is the one that matters most, because the
 * property it holds is fixed rather than desirable: a simulation
 * behaves identically whether or not anybody is watching it. It is
 * written so that it can fail. The comparison is over whole file
 * bytes; a third run at a different seed must make the same
 * comparison say `differ`, so the arm cannot pass by comparing
 * nothing with nothing; the watcher must be shown to have consumed
 * something, so it cannot pass by attaching and doing nothing; and
 * the watcher's own mapping of the ring is read out of its process
 * and required to carry no write permission, so a viewer that mapped
 * the ring writable fails here rather than passing quietly.
 *
 * The loss arm induces the overwrite rather than waiting for one. The
 * harness owns a ring of its own, sized so its slot count is the
 * floor, publishes more frames than it holds, and only then lets the
 * viewer attach at the oldest surviving frame. The number of lost
 * frames is therefore arithmetic and not luck, and it is checked
 * against the producer's own published count rather than against the
 * consumer's opinion of itself. Its credibility arm publishes fewer
 * frames than the ring holds and requires the same viewer to report
 * no loss at all, so a viewer that always claimed loss fails as
 * surely as one that never did.
 *
 * Gates:
 *   1. Watching changes nothing. One run with a viewer attached for
 *      its whole length and one with none produce byte-identical
 *      episode files; a run at another seed differs; the viewer is
 *      shown to have accepted frames; and its mapping is read only.
 *   2. The picture advances as the frames arrive. The viewer's own
 *      account of what it took, round by round, ends where the run
 *      ends and begins strictly before it, over more than one round.
 *   3. One schema, two sources. Every value the viewer draws from the
 *      ring equals what it draws from the file the same run wrote,
 *      bitwise, with a credibility arm requiring a different episode
 *      to differ.
 *   4. Loss is reported, exactly. A lapped ring costs the viewer a
 *      known number of frames; it reports that number, reports the
 *      steps missing from the episode and the opening record it never
 *      saw, and still holds the surviving records at their own step
 *      numbers.
 *   5. Loss is not reported when there is none, which is what makes
 *      gate 4 a measurement.
 *
 * Each of the four arms can be run alone with --arm=NAME, and each is
 * paired with a build that carries a deliberate defect it must
 * reject: RL_LIVE_INJECT selects the defect and the pairing is in the
 * makefile beside the run. A gate that has never been shown failing
 * is a gate nobody has measured.
 *
 * Skips (77) when the sibling stack archives or the viewer binary are
 * not built.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"
#include "k26rl_tap.h"

#define WORK_DIR "/tmp/kflc_rl_live_test"
#define VIEWER   "../tools/k26rl_view/k26rl_view"

/* The fixture's own numbers, in one place so an arm cannot disagree
 * with the run that produced its dump. One environment, because the
 * arms compare episode by index and one environment makes the order
 * episodes begin in the order they end in. */
enum { LIVE_ENVS = 1, LIVE_ACT = 2, LIVE_OBS = 5, LIVE_AGENTS = 1 };
/* Ninety driver steps. It is not ninety recorded steps: a horizon
 * ends an episode and the driver step that follows opens the next one
 * without recording a transition, so the run is seven whole episodes
 * of twelve recorded steps with six boundary steps between them. The
 * arms below take the counts from the run rather than assuming them,
 * and check them against each other. */
enum { LIVE_STEPS = 90 };
enum { PRODUCER_STEP_US = 2000 };

/* The randomisation width the loss arm's own ring declares. It buys
 * nothing but size: the slot is sized from the largest frame the
 * geometry admits, and a slot this wide brings the slot count down to
 * its floor, so a lap of the ring is a thousand frames rather than a
 * hundred thousand and the arm is cheap as well as certain. */
enum { LOSS_DR_MAX = 700 };
enum { LOSS_EXTRA = 5 };        /* frames published past a full lap */

/* Two reset draws, so an episode-start frame carries a randomisation
 * record; a horizon short enough that a run of ninety steps crosses
 * several episode boundaries, so the arms see starts and ends and not
 * one long middle. */
static const char *const LIVE_KFL =
    "form RL_LIVE\n"
    "fn world live_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0,10.0) vel_z=0.0\n"
    "    let range_scale: double = 7.0e6\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 12\n"
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
    "    end\n"
    "end\n"
    "end\n";

static double act_(uint32_t t, uint32_t ch)
{
    if (ch == 0)
        return ((double)((t * 7u) % 13u)) / 13.0 * 2.0 - 1.0;
    return (double)(t % 3u);
}

/* ---- files, text, and the viewer ----------------------------------- */

static char *slurp_(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    long n;
    char *buf;

    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    n = ftell(f);
    ASSERT(n >= 0);
    rewind(f);
    buf = (char *)malloc((size_t)n + 1);
    ASSERT(buf != NULL);
    ASSERT(fread(buf, 1, (size_t)n, f) == (size_t)n);
    buf[n] = '\0';
    fclose(f);
    if (out_len)
        *out_len = (size_t)n;
    return buf;
}

/* The first line beginning with prefix, or NULL. Prefixes carry every
 * leading field, so a match is exact rather than nearly so. */
static const char *line_(const char *text, const char *prefix)
{
    size_t n = strlen(prefix);
    const char *p = text;

    while (p && *p) {
        if (strncmp(p, prefix, n) == 0)
            return p;
        p = strchr(p, '\n');
        if (p)
            p++;
    }
    return NULL;
}

/* Every line with this prefix says the same thing, and there is at
 * least one. The dump repeats an episode's header once per panel, so
 * an arm that expected one copy would be measuring the panel count;
 * what it means to check is that the panels agree. Returns the number
 * of copies, or 0 when they disagree or none exist. */
static int lines_agree_(const char *text, const char *prefix)
{
    size_t n = strlen(prefix);
    const char *p = text, *first = NULL;
    size_t first_len = 0;
    int c = 0;

    while (p && *p) {
        if (strncmp(p, prefix, n) == 0) {
            const char *nl = strchr(p, '\n');
            size_t len = nl ? (size_t)(nl - p) : strlen(p);
            if (!first) {
                first = p;
                first_len = len;
            } else if (len != first_len ||
                       strncmp(p, first, first_len) != 0) {
                return 0;
            }
            c++;
        }
        p = strchr(p, '\n');
        if (p)
            p++;
    }
    return c;
}

static int count_lines_(const char *text, const char *prefix)
{
    size_t n = strlen(prefix);
    const char *p = text;
    int c = 0;

    while (p && *p) {
        if (strncmp(p, prefix, n) == 0)
            c++;
        p = strchr(p, '\n');
        if (p)
            p++;
    }
    return c;
}

/* Every line whose first word is one of the named keywords, in order.
 * The comparison arms run over this rather than over whole dumps,
 * because a live source and a file honestly differ about where they
 * came from and must not differ about a single value. */
static char *lines_with_(const char *text, const char *const *keywords)
{
    size_t cap = strlen(text) + 1;
    char *out = (char *)malloc(cap);
    size_t w = 0;
    const char *p = text;

    ASSERT(out != NULL);
    while (p && *p) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) + 1 : strlen(p);
        int i;
        for (i = 0; keywords[i]; i++) {
            size_t kn = strlen(keywords[i]);
            if (strncmp(p, keywords[i], kn) == 0 && p[kn] == ' ') {
                memcpy(out + w, p, len);
                w += len;
                break;
            }
        }
        p = nl ? nl + 1 : NULL;
    }
    out[w] = '\0';
    return out;
}

static void run_viewer_(char *const argv[], const char *out_path)
{
    pid_t pid;
    int status = 0;

    /* The buffer goes out before the fork, or the child inherits it
     * and writes this process's lines into the dump it is about to
     * make. */
    fflush(stdout);
    pid = fork();
    ASSERT(pid >= 0);
    if (pid == 0) {
        char err[512];
        snprintf(err, sizeof err, "%s.err", out_path);
        ASSERT(freopen(out_path, "w", stdout) != NULL);
        ASSERT(freopen(err, "w", stderr) != NULL);
        execv(VIEWER, argv);
        _exit(127);
    }
    ASSERT(waitpid(pid, &status, 0) == pid);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        char show[512];
        fprintf(stderr, "viewer failed (status=%d)\n", status);
        snprintf(show, sizeof show, "cat %s.err", out_path);
        (void)!system(show);
        exit(1);
    }
}

static pid_t spawn_viewer_(char *const argv[], const char *out_path)
{
    pid_t pid;

    fflush(stdout);
    pid = fork();
    ASSERT(pid >= 0);
    if (pid == 0) {
        char err[512];
        snprintf(err, sizeof err, "%s.err", out_path);
        ASSERT(freopen(out_path, "w", stdout) != NULL);
        ASSERT(freopen(err, "w", stderr) != NULL);
        execv(VIEWER, argv);
        _exit(127);
    }
    return pid;
}

static void reap_viewer_(pid_t pid, const char *out_path)
{
    int status = 0;

    ASSERT(waitpid(pid, &status, 0) == pid);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        char show[512];
        fprintf(stderr, "watching viewer failed (status=%d)\n", status);
        snprintf(show, sizeof show, "cat %s.err", out_path);
        (void)!system(show);
        exit(1);
    }
}

/* The permission field of the watcher's own mapping of the ring, read
 * out of its process. This is the read-only claim made mechanically:
 * a viewer that mapped the object writable reports a write bit here
 * and fails, where a viewer that merely never wrote still reports
 * none. Every mapping of the ring in the process is scanned and a
 * writable one wins the report, so a second writable mapping beside a
 * read-only one cannot hide behind it.
 *
 * The process is required to be the viewer before any of that is
 * believed, and that requirement is not decoration. A watcher is
 * forked and then executes the viewer, and between those two moments
 * its address space is still the producer's, ring mapping and write
 * permission included. An arm that read the maps in that window would
 * be reading the producer's own mapping and reporting it as the
 * viewer's, which is a gate measuring the wrong process. It was
 * written that way first and this is what it did.
 *
 * Returns 0 until the viewer holds a mapping, which is also how an
 * arm knows the watcher has attached. */
static int mapping_perms_(pid_t pid, const char *tag, char *perms, size_t n)
{
    char path[64], line[1024], exe[512];
    ssize_t got;
    FILE *f;
    int found = 0;

    snprintf(path, sizeof path, "/proc/%ld/exe", (long)pid);
    got = readlink(path, exe, sizeof exe - 1);
    if (got <= 0)
        return 0;
    exe[got] = '\0';
    if (strlen(exe) < 10 || strcmp(exe + strlen(exe) - 10, "k26rl_view") != 0)
        return 0;               /* still the forked producer, not yet the viewer */

    snprintf(path, sizeof path, "/proc/%ld/maps", (long)pid);
    f = fopen(path, "r");
    if (!f)
        return 0;
    while (fgets(line, sizeof line, f)) {
        char *sp;
        if (!strstr(line, tag))
            continue;
        sp = strchr(line, ' ');
        if (!sp || strlen(sp) < 5)
            continue;
        if (!found || sp[2] == 'w')
            snprintf(perms, n, "%.4s", sp + 1);
        found = 1;
        if (sp[2] == 'w')
            break;
    }
    fclose(f);
    return found;
}

static void sleep_us_(long us)
{
    struct timespec ts;
    ts.tv_sec = us / 1000000L;
    ts.tv_nsec = (us % 1000000L) * 1000L;
    nanosleep(&ts, NULL);
}

/* ---- the serving run ----------------------------------------------- */

/* One scripted session. The file is written when out_path is given
 * and the ring published when tap_name is; when a watcher is asked
 * for, it is started after the ring exists and the run does not begin
 * until the watcher's own mapping of that ring is visible, so an arm
 * claiming a viewer was attached for the whole run knows one was.
 *
 * The pause between steps is the producer being slower than the
 * watcher, which is what a run against a wall clock is. Without it a
 * ninety-step run would be over before a watcher looked twice and the
 * advance the next gate measures would be a single round. */
static void drive_(const RlSurface *s, uint64_t seed, uint32_t steps,
                   const char *out_path, const char *tap_name,
                   char *const watcher_argv[], const char *watcher_out,
                   char *out_perms, size_t perms_n, int paced)
{
    K26RlEnv *env = NULL;
    double act[LIVE_ENVS * LIVE_ACT];
    pid_t watcher = -1;
    uint32_t t;

    ASSERT(s->create(seed, LIVE_ENVS, &env) == K26RL_OK);
    if (out_path)
        ASSERT(s->output(env, out_path) == K26RL_OK);
    if (tap_name)
        ASSERT(s->tap(env, tap_name) == K26RL_OK);

    if (watcher_argv) {
        watcher = spawn_viewer_(watcher_argv, watcher_out);
        /* An arm that wants the watcher attached for the whole run
         * waits for its mapping before the first step, and takes the
         * mapping's permissions while it is there. An arm that does
         * not ask for the permissions is not making that claim and
         * does not wait. */
        if (out_perms) {
            char tag[128];
            int tries;

            snprintf(tag, sizeof tag, "k26rl.%s", tap_name);
            for (tries = 0; tries < 20000; tries++) {
                if (mapping_perms_(watcher, tag, out_perms, perms_n))
                    break;
                sleep_us_(1000);
            }
            ASSERT(tries < 20000);
        }
    }

    for (t = 0; t < steps; t++) {
        uint32_t c;
        for (c = 0; c < LIVE_ACT; c++)
            act[c] = act_(t, c);
        ASSERT(s->step(env, act) == K26RL_OK);
        if (paced)
            sleep_us_(PRODUCER_STEP_US);
    }
    s->destroy(env);
    if (watcher >= 0)
        reap_viewer_(watcher, watcher_out);
}

/* ---- the arms ------------------------------------------------------ */

static char name_store_[4][64];

static const char *tap_name_(int slot, const char *tag)
{
    snprintf(name_store_[slot], sizeof name_store_[slot], "live.%ld.%s",
             (long)getpid(), tag);
    return name_store_[slot];
}

/* The keyword set the two sources must agree on to the bit: every
 * recorded value and the episode identity that carries it. What is
 * left out is what the two honestly differ about, which is where the
 * records came from. */
static const char *const VALUE_KEYS_[] = {
    "episode", "flag", "dt", "reward", "return", "terminal_adj",
    "initial_obs", "dr", "obs", "act", "traj", NULL
};

/* The defect a red build carries. Zero is the ordinary build.
 *
 *   1  the watched run is driven at a different seed from the
 *      unwatched one, so the two recordings differ and the isolation
 *      arm's whole-file comparison has to say so.
 *   2  the watcher looks exactly once, so nothing about the advance of
 *      the picture can be seen and the liveness arm has to say so.
 *   3  the file half of the comparison is read from a different run,
 *      so the two sources disagree and the equality arm has to say so.
 *   4  the watcher joins a lapped ring at the producer's current
 *      position instead of at the oldest surviving frame, so it is
 *      owed nothing, reports no loss, and the loss arm has to say so.
 */
#ifndef RL_LIVE_INJECT
#define RL_LIVE_INJECT 0
#endif

static void *so_;
static RlSurface s_;
static uint8_t *spec_;
static int32_t spec_len_;
static int drives_done_;
static char watch_perms_[8];

/* The three runs the first three arms read: unwatched, watched for
 * its whole length, and one at another seed for the comparisons to
 * have something to differ from. Performed once however many arms
 * ask for them. */
static void drives_(void)
{
    const char *n = tap_name_(0, "iso");
    static char tapopt[80];
    char *argv[] = {
        (char *)VIEWER, (char *)"--dump", (char *)"all",
        (char *)"--tap", tapopt,
        (char *)"--from-start",
        (char *)"--poll-ms", (char *)"1",
#if RL_LIVE_INJECT == 2
        (char *)"--polls", (char *)"1",
#endif
        (char *)"--idle-polls", (char *)"20000",
        NULL
    };
    uint64_t watched_seed = 7;

#if RL_LIVE_INJECT == 1
    watched_seed = 8;
#endif
    if (drives_done_)
        return;
    snprintf(tapopt, sizeof tapopt, "%s", n);
    memset(watch_perms_, 0, sizeof watch_perms_);

    drive_(&s_, 7, LIVE_STEPS, WORK_DIR "/unwatched.k26epi", NULL,
           NULL, NULL, NULL, 0, 0);
#if RL_LIVE_INJECT == 2
    /* A watcher that looks once, and is not waited for: whatever it
     * takes, it takes in a single round. */
    drive_(&s_, watched_seed, LIVE_STEPS, WORK_DIR "/watched.k26epi", n,
           argv, WORK_DIR "/watched.txt", NULL, 0, 1);
#else
    drive_(&s_, watched_seed, LIVE_STEPS, WORK_DIR "/watched.k26epi", n,
           argv, WORK_DIR "/watched.txt", watch_perms_,
           sizeof watch_perms_, 1);
#endif
    drive_(&s_, 8, LIVE_STEPS, WORK_DIR "/other.k26epi", NULL,
           NULL, NULL, NULL, 0, 0);
    drives_done_ = 1;
}

static void arm_isolation_(void)
{
    {
        const char *perms = watch_perms_;
        char *watched;

        drives_();
        ASSERT(rl_files_equal_(WORK_DIR "/unwatched.k26epi",
                               WORK_DIR "/watched.k26epi"));
        ASSERT(!rl_files_equal_(WORK_DIR "/unwatched.k26epi",
                                WORK_DIR "/other.k26epi"));

        /* The watcher has to have watched. An arm that compared two
         * runs while the viewer sat idle would compare nothing. */
        watched = slurp_(WORK_DIR "/watched.txt", NULL);
        {
            const char *e = line_(watched, "live_end ");
            char reason[32];
            unsigned long long rounds = 0, accepted = 0, lost = 0;
            ASSERT(e != NULL);
            ASSERT(sscanf(e, "live_end %31s %llu %llu %llu", reason, &rounds,
                          &accepted, &lost) == 4);
            ASSERT(accepted > (unsigned long long)LIVE_STEPS);
            ASSERT(strcmp(reason, "closed") == 0);
            printf("gate 1: the watched and unwatched runs are byte"
                   " identical over %llu frames the viewer took, a run at"
                   " another seed differs, and the viewer ended on the"
                   " producer's own close after %llu rounds: OK\n",
                   accepted, rounds);
        }
        /* Read only, mechanically: the permission bits of the
         * watcher's own mapping, read out of its process while it
         * held it. */
        ASSERT(perms[0] == 'r');
        ASSERT(perms[1] == '-');
        printf("gate 1: the viewer's mapping of the ring carried"
               " permissions `%s`, with no write bit: OK\n", perms);
        free(watched);
    }

}

static void arm_liveness_(void)
{
    {
        char *watched;

        drives_();
        watched = slurp_(WORK_DIR "/watched.txt", NULL);
        const char *p = watched;
        unsigned long long first_steps = 0, last_steps = 0, total = 0;
        unsigned long long prev = 0, accepted = 0;
        unsigned episodes = 0;
        int rounds = 0;

        while (p && *p) {
            if (strncmp(p, "live_poll ", 10) == 0) {
                unsigned long long round, lost, acc, lost_total, held;
                unsigned frames, eps;
                ASSERT(sscanf(p, "live_poll %llu %u %llu %llu %llu %u %llu",
                              &round, &frames, &lost, &acc, &lost_total,
                              &eps, &held) == 7);
                if (!rounds)
                    first_steps = held;
                /* Never backwards: what has arrived cannot unarrive. */
                ASSERT(held >= prev);
                prev = held;
                last_steps = held;
                rounds++;
            }
            p = strchr(p, '\n');
            if (p)
                p++;
        }
        {
            const char *e = line_(watched, "live_end ");
            char reason[32];
            unsigned long long r, a, l, held;
            unsigned eps;
            ASSERT(e != NULL);
            ASSERT(sscanf(e, "live_end %31s %llu %llu %llu %u %llu", reason,
                          &r, &a, &l, &eps, &held) == 6);
            total = held;
            episodes = eps;
            accepted = a;
            ASSERT(l == 0);   /* the ring holds far more than this run */
        }
        /* More than one round, and strictly more held at the end than
         * at the first: the frames arrived while the run ran rather
         * than in one draught after it. A viewer that only ever
         * looked once would report one round and fail here. */
        ASSERT(rounds >= 2);
        ASSERT(last_steps > first_steps);
        ASSERT(total == last_steps);
        ASSERT(total > 0);
        /* Every frame is accounted for: one opening and one closing
         * frame per episode, and a step record for everything else.
         * A viewer that dropped frames without saying so, or counted
         * one twice, fails this rather than agreeing with itself. */
        ASSERT(accepted == total + 2 * (unsigned long long)episodes);
        printf("gate 2: the viewer's held step count rose from %llu to %llu"
               " over %d rounds, and its %llu frames are %llu step records"
               " and the opening and closing frames of %u episodes: OK\n",
               first_steps, last_steps, rounds, accepted, total, episodes);
        free(watched);
    }

}

static void arm_sources_(void)
{
    {
        char *from_ring, *from_file, *other;
        char *a, *b, *c;
#if RL_LIVE_INJECT == 3
        /* The file half read from another run entirely. */
        char *argv[] = {
            (char *)VIEWER, (char *)"--dump", (char *)"all",
            (char *)WORK_DIR "/other.k26epi", NULL
        };
#else
        char *argv[] = {
            (char *)VIEWER, (char *)"--dump", (char *)"all",
            (char *)WORK_DIR "/watched.k26epi", NULL
        };
#endif
        char *argv_other[] = {
            (char *)VIEWER, (char *)"--dump", (char *)"all",
            (char *)"--episode", (char *)"1",
            (char *)WORK_DIR "/watched.k26epi", NULL
        };

        drives_();
        from_ring = slurp_(WORK_DIR "/watched.txt", NULL);
        run_viewer_(argv, WORK_DIR "/fromfile.txt");
        from_file = slurp_(WORK_DIR "/fromfile.txt", NULL);
        run_viewer_(argv_other, WORK_DIR "/otherep.txt");
        other = slurp_(WORK_DIR "/otherep.txt", NULL);

        a = lines_with_(from_ring, VALUE_KEYS_);
        b = lines_with_(from_file, VALUE_KEYS_);
        c = lines_with_(other, VALUE_KEYS_);

        /* Not a comparison of two empty strings. */
        ASSERT(strlen(a) > 1000);
        ASSERT(strcmp(a, b) == 0);
        /* The comparison's own credibility: it has to be able to say
         * two dumps differ, which comparing two equal things never
         * shows. */
        ASSERT(strcmp(a, c) != 0);
        printf("gate 3: %u bytes of recorded values read from the ring"
               " equal the same values read from the file the run wrote,"
               " and one episode's differ from all of them: OK\n",
               (unsigned)strlen(a));
        free(a);
        free(b);
        free(c);
        free(from_ring);
        free(from_file);
        free(other);
    }

}

static void arm_loss_(void)
{
    {
        static const struct {
            const char *tag;
            int lap;              /* publish past a full lap of the ring */
        } arms[2] = { { "lap", 1 }, { "nolap", 0 } };
        int arm;

        for (arm = 0; arm < 2; arm++) {
            const char *n = tap_name_(1 + arm, arms[arm].tag);
            K26RlEpisodeGeom geom;
            K26RlTap *tap = NULL;
            uint32_t slot_size = 0, slot_count = 0;
            uint32_t steps, t;
            uint64_t published, expect_lost, expect_accept;
            double obs[LIVE_OBS], act[LIVE_ACT], rew[LIVE_AGENTS];
            uint32_t dr_tags[2];
            double dr_values[2];
            char out[128], tapopt[80];
            char *argv[] = {
                (char *)VIEWER, (char *)"--dump", (char *)"all",
                (char *)"--tap", tapopt,
#if RL_LIVE_INJECT != 4
                (char *)"--from-start",
#endif
                (char *)"--steps", (char *)"0:3",
                (char *)"--poll-ms", (char *)"1",
                (char *)"--idle-polls", (char *)"200",
                NULL
            };
            char *text;
            unsigned long long accepted = 0, lost = 0;

            memset(&geom, 0, sizeof geom);
            geom.n_envs = LIVE_ENVS;
            geom.agent_count = LIVE_AGENTS;
            geom.obs_total = LIVE_OBS;
            geom.act_total = LIVE_ACT;
            geom.steps_per_chunk = 1024;
            geom.dr_max = LOSS_DR_MAX;
            ASSERT(k26rl_tap_open(n, &geom, 4242, 0, "3.2", "test",
                                  spec_, (uint32_t)spec_len_,
                                  &tap) == K26RL_OK);
            {
                K26RlTapReader *probe_r = NULL;
                ASSERT(k26rl_tap_attach(n, 0, &probe_r) == K26RL_OK);
                ASSERT(k26rl_tap_reader_info(probe_r, &slot_size,
                                             &slot_count) == K26RL_OK);
                k26rl_tap_detach(probe_r);
            }
            /* The declared randomisation width was chosen to bring
             * the slot count to its floor; an arm whose ring turned
             * out larger would need a different number of frames to
             * lap it and is refused rather than quietly weakened. */
            ASSERT(slot_count == K26RL_TAP_SLOTS_MIN);

            steps = arms[arm].lap ? slot_count + LOSS_EXTRA
                                  : slot_count / 4;
            for (t = 0; t < LIVE_OBS; t++)
                obs[t] = 0.0;
            for (t = 0; t < LIVE_ACT; t++)
                act[t] = 0.5;
            rew[0] = 0.25;
            dr_tags[0] = 1;  dr_values[0] = 6.95e6;
            dr_tags[1] = 2;  dr_values[1] = 7351.0;
            k26rl_tap_start(tap, 0, 0, obs, dr_tags, dr_values, 2);
            for (t = 0; t < steps; t++) {
                /* The first channel carries the step's own number, so
                 * an arm can say the records held are the records
                 * published and not merely as many of them. */
                obs[0] = (double)t;
                k26rl_tap_step(tap, 0, obs, act, rew, 0u, 0.1);
            }
            k26rl_tap_end(tap, 0, K26RL_END_TRUNCATED, 0, rew);
            published = k26rl_tap_published(tap);
            ASSERT(published == (uint64_t)steps + 2);

            expect_accept = published < slot_count ? published : slot_count;
            expect_lost = published - expect_accept;

            snprintf(tapopt, sizeof tapopt, "%s", n);
            snprintf(out, sizeof out, WORK_DIR "/loss_%s.txt",
                     arms[arm].tag);
            run_viewer_(argv, out);
            k26rl_tap_close(tap);

            text = slurp_(out, NULL);
            ASSERT(sscanf(line_(text, "frames_accepted "),
                          "frames_accepted %llu", &accepted) == 1);
            ASSERT(sscanf(line_(text, "frames_lost "),
                          "frames_lost %llu", &lost) == 1);
            /* Held against the producer's own count, not against the
             * consumer's arithmetic about itself. */
            ASSERT(accepted + lost == published);
            ASSERT(accepted == expect_accept);
            ASSERT(lost == expect_lost);

            if (arms[arm].lap) {
                const char *g = line_(text, "episode_gap 0 ");
                unsigned k, first, count, held_step;
                double v;
                uint64_t bits;
                const char *o;

                /* The opening frame was among the overwritten, so the
                 * episode says it has no initial observation rather
                 * than presenting its earliest held values as one. */
                ASSERT(line_(text, "episode_start_missing 0") != NULL);
                ASSERT(g != NULL);
                ASSERT(sscanf(g, "episode_gap %u %u %u", &k, &first,
                              &count) == 3);
                ASSERT(k == 0);
                ASSERT(first == 0);
                /* Exactly the step records that went past: the frames
                 * lost less the opening frame that was one of them. */
                ASSERT((uint64_t)count == lost - 1);
                /* One hole, and every panel says the same about it. */
                ASSERT(lines_agree_(text, "episode_gap ") ==
                       count_lines_(text, "episode_gap "));
                ASSERT(count_lines_(text, "episode_gap ") >= 1);
                /* And no value was invented in place of what the
                 * opening record would have carried. */
                ASSERT(line_(text, "initial_obs ") == NULL);
                ASSERT(line_(text, "dr ") == NULL);

                /* And the records held are the right records: the
                 * first one held carries its own step number in its
                 * first channel, at the step number the viewer
                 * reports for it. */
                o = line_(text, "obs 0 ");
                ASSERT(o != NULL);
                ASSERT(sscanf(o, "obs 0 %u 0 %16" SCNx64, &held_step,
                              &bits) == 2);
                memcpy(&v, &bits, sizeof v);
                ASSERT(held_step == count);
                ASSERT(v == (double)held_step);
                printf("gate 4: a ring of %u slots lapped by %llu published"
                       " frames cost the viewer %llu of them; it reported"
                       " that count, the %u missing steps, the opening"
                       " record it never saw, and held step %u at its own"
                       " number: OK\n", slot_count,
                       (unsigned long long)published, lost, count,
                       held_step);
            } else {
                ASSERT(lost == 0);
                ASSERT(line_(text, "episode_gap ") == NULL);
                ASSERT(line_(text, "episode_start_missing") == NULL);
                ASSERT(line_(text, "initial_obs 0 0 ") != NULL);
                printf("gate 5: the same viewer over a ring of %u slots"
                       " carrying %llu frames reported no loss, no gap and"
                       " no missing opening record: OK\n", slot_count,
                       (unsigned long long)published);
            }
            free(text);
        }
    }
}

static void setup_(void)
{
    K26RlEnv *probe = NULL;
    RlSpecView v;

    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/live.kfl", LIVE_KFL);
    rl_compile_(WORK_DIR "/live.kfl", WORK_DIR "/live", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/live.rlenv.so"));

    so_ = rl_dlopen_(WORK_DIR "/live.rlenv.so");
    rl_resolve_surface_(so_, &s_);

    /* The spec, taken once, so the arm that owns a ring of its own
     * publishes the artifact's real declaration rather than a
     * fabrication of the same shape. */
    ASSERT(s_.create(1, LIVE_ENVS, &probe) == K26RL_OK);
    spec_len_ = s_.spec(probe, NULL, 0);
    ASSERT(spec_len_ > 0);
    spec_ = (uint8_t *)malloc((size_t)spec_len_);
    ASSERT(spec_ != NULL);
    ASSERT(s_.spec(probe, spec_, (uint32_t)spec_len_) == spec_len_);
    rl_parse_spec_(spec_, (uint32_t)spec_len_, &v);
    ASSERT(v.obs_total == LIVE_OBS);
    ASSERT(v.act_total == LIVE_ACT);
    ASSERT(v.agent_count == LIVE_AGENTS);
    printf("test_rl_live: fixture: %u observation channels, %u action"
           " channels, %u agent: OK\n", v.obs_total, v.act_total,
           v.agent_count);
    s_.destroy(probe);
}

int main(int argc, char **argv)
{
    const char *arm = NULL;
    int i, ran = 0;

    for (i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--arm=", 6) == 0)
            arm = argv[i] + 6;
        else {
            fprintf(stderr, "usage: test_rl_live [--arm=NAME]\n");
            return 2;
        }
    }
    if (!rl_libs_present_("test_rl_live"))
        return 77;
    if (!rl_file_exists_(VIEWER)) {
        fprintf(stderr, "test_rl_live: skip: %s not built\n", VIEWER);
        return 77;
    }
    setup_();

    if (!arm || strcmp(arm, "isolation") == 0) { arm_isolation_(); ran++; }
    if (!arm || strcmp(arm, "liveness") == 0)  { arm_liveness_();  ran++; }
    if (!arm || strcmp(arm, "sources") == 0)   { arm_sources_();   ran++; }
    if (!arm || strcmp(arm, "loss") == 0)      { arm_loss_();      ran++; }
    if (!ran) {
        fprintf(stderr, "test_rl_live: no arm named `%s`\n", arm);
        return 2;
    }

    free(spec_);
    dlclose(so_);
    if (arm)
        printf("test_rl_live: arm %s passed\n", arm);
    else
        printf("test_rl_live: 5 gates passed\n");
    return 0;
}
