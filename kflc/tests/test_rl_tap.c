/* test_rl_tap.c: the telemetry tap on the serve surface.
 *
 * The tap publishes the record format's frames on a shared memory
 * ring while a simulation serves. Two things have to be true at once:
 * a consumer must receive the same records the episode file receives,
 * and the simulation must not be able to tell whether anyone is
 * watching. The gates below are numbered as the design numbers them.
 *
 * Gates:
 *   8.  Cross-transport equality. One serving run with both the
 *       episode file and the tap enabled, a consumer process
 *       recording every frame it accepts. For every step record the
 *       consumer received, its observations, actions, rewards, flag
 *       word, and applied dt equal the file's bitwise for the same
 *       identity triple and step index; sequence values and frame
 *       boundaries are excluded, being per-transport by definition.
 *       The gate cannot pass vacuously: the consumer's accepted set
 *       must cover a whole episode, its start and end frames
 *       included. The two transports' file-header payloads are
 *       compared byte for byte in the same gate, which is where "one
 *       schema, two transports" is actually checked. A second arm
 *       drives a deliberately induced fault on a second artifact and
 *       checks the awkward case rather than only the easy one: the
 *       final step record of a faulted episode completes no
 *       transition, and both transports must carry it identically,
 *       zero reward, zero applied dt, the faulting call's actions,
 *       the fault bit alone, and the same end reason and registry
 *       code.
 *   9a. Bit identity, tap absent against tap enabled: two runs at one
 *       seed and one scripted action stream, output enabled in both,
 *       one never calling the tap and one enabling it at create.
 *       Episode files byte-identical, per-step getter dumps bitwise
 *       equal.
 *   9b. Consumer absent, attached, and stalled: three tapped runs,
 *       one unwatched, one with a consumer reading continuously, one
 *       with a consumer that accepts a single frame and then blocks
 *       for the rest of the run. All three files identical to each
 *       other and to 9a's.
 *   9c. Attach, detach, and death mid-run: a consumer that attaches
 *       and detaches repeatedly, and one killed outright while its
 *       mapping is live. The file stays identical.
 *   9d. Read-only, mechanically: the consumer's mapping is PROT_READ
 *       and a store through it faults, proven by catching the fault
 *       rather than by asserting the flag that was passed; and the
 *       object's mode is 0600.
 *   12. Enable surface: the version word, the timing rule, every name
 *       refusal with its bound in the decoded message, exclusivity,
 *       disablement and re-enablement, no object left behind at
 *       destroy, no object created at all by a run that never taps,
 *       and no rendering library in the artifact's needs.
 *   14. Late attach: a consumer joining mid-run recovers the spec
 *       blob from the preamble bitwise equal to the artifact's own,
 *       claims nothing about the episode already in flight, and the
 *       first episode it sees whole matches the file's.
 *
 * Gate 13 is the ring protocol's own and lives with the library
 * (libk26rl/tests/test_k26rl_ring.c); gate 9e is the hot-path cost
 * and lives with the interposition gate (test_rl_hotpath.c), where
 * the untapped drive it must equal is measured in the same process.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "rl_gate_util.h"
#include "k26rl_tap.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_tap_test"

/* The determinism gate's fixture: two bodies, a horizon that ends
 * episodes inside a short drive, two action channels, and two reset
 * draws so episode-start frames carry a randomisation record. */
static const char *const TAP_KFL =
    "form RL_TAP\n"
    "fn world tap_world\n"
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

/* The fault fixture: the reward divides by zero when a caller drives
 * the first action channel to 1.0, so a fault is induced at a step of
 * the driver's choosing. Same channel shape as the fixture above, so
 * the one decoder reads both artifacts' frames. */
static const char *const FAULT_KFL =
    "form RL_TAPFAULT\n"
    "fn world tapfault_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 6\n"
    "    end\n"
    "    action push box 0.0 4.0 default 0.0\n"
    "    action gear discrete 3 default 1\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0 / (1.0 - push)\n"
    "    end\n"
    "end\n"
    "end\n";

enum { TAP_ENVS = 2, TAP_ACT = 2, TAP_OBS = 5, TAP_STEPS = 40 };
enum { FAULT_AT = 3 };

static double act_(uint32_t t, uint32_t e, uint32_t ch)
{
    if (ch == 0)
        return ((double)((t * 7u + e * 3u) % 13u)) / 13.0 * 2.0 - 1.0;
    return (double)((t + e) % 3u);
}

/* ---- Shared-memory bookkeeping ------------------------------------- */

static int shm_objects_(void)
{
    DIR *d = opendir("/dev/shm");
    struct dirent *de;
    int n = 0;

    if (!d)
        return -1;
    while ((de = readdir(d)) != NULL) {
        if (strncmp(de->d_name, "k26rl.", 6) == 0)
            n++;
    }
    closedir(d);
    return n;
}

static char name_store_[8][64];

static const char *tap_name_(int slot, const char *tag)
{
    snprintf(name_store_[slot], sizeof name_store_[slot], "tap.%ld.%s",
             (long)getpid(), tag);
    return name_store_[slot];
}

/* ---- Driving the serve surface ------------------------------------- */

/* One scripted run. out_path enables the episode file when non-null,
 * tap_name enables the tap when non-null, and dump_path records the
 * getter outputs of every step when non-null. Returns after destroy,
 * so the file is closed and the ring released. */
static void wait_ready_(int fd)
{
    char b = 0;

    if (fd < 0)
        return;
    /* One byte, written by the consumer only once it holds a mapping.
     * Without this the consumer races the producer's whole window and
     * can lose it, and an arm that claims to run with a consumer
     * attached must know that one is. */
    ASSERT(read(fd, &b, 1) == 1);
    ASSERT(b == 'a');
    close(fd);
}

static void drive_(const RlSurface *s, uint64_t seed, int steps,
                   const char *out_path, const char *tap_name,
                   const char *dump_path, int ready_fd)
{
    K26RlEnv *env = NULL;
    double act[TAP_ENVS * TAP_ACT];
    double obs[TAP_ENVS * TAP_OBS];
    double rew[TAP_ENVS];
    uint32_t flags[TAP_ENVS];
    uint16_t fault[TAP_ENVS];
    FILE *dump = NULL;

    ASSERT(s->create(seed, TAP_ENVS, &env) == K26RL_OK);
    if (out_path)
        ASSERT(s->output(env, out_path) == K26RL_OK);
    if (tap_name)
        ASSERT(s->tap(env, tap_name) == K26RL_OK);
    if (dump_path) {
        dump = fopen(dump_path, "wb");
        ASSERT(dump != NULL);
    }
    /* The tap exists now, so a waiting consumer can attach; block
     * until it has before any step is taken. */
    wait_ready_(ready_fd);
    for (int t = 0; t < steps; t++) {
        for (uint32_t e = 0; e < TAP_ENVS; e++)
            for (uint32_t c = 0; c < TAP_ACT; c++)
                act[e * TAP_ACT + c] = act_((uint32_t)t, e, c);
        ASSERT(s->step(env, act) == K26RL_OK);
        if (dump) {
            ASSERT(s->obs(env, obs) == K26RL_OK);
            ASSERT(s->reward(env, rew) == K26RL_OK);
            ASSERT(s->flags(env, flags) == K26RL_OK);
            ASSERT(s->fault_codes(env, fault) == K26RL_OK);
            ASSERT(fwrite(obs, sizeof obs, 1, dump) == 1);
            ASSERT(fwrite(rew, sizeof rew, 1, dump) == 1);
            ASSERT(fwrite(flags, sizeof flags, 1, dump) == 1);
            ASSERT(fwrite(fault, sizeof fault, 1, dump) == 1);
        }
    }
    if (dump)
        fclose(dump);
    s->destroy(env);
}

/* ---- Consumer processes -------------------------------------------- */

typedef enum {
    CONSUME_ALL,      /* read continuously, dump every accepted frame */
    CONSUME_STALL,    /* accept one frame, then block until the end */
    CONSUME_FLAP,     /* attach and detach repeatedly */
    CONSUME_DIE       /* attach, then wait to be killed while mapped */
} ConsumeMode;

/* Frames are dumped length-prefixed so the parent can walk them back
 * without knowing the geometry. */
static pid_t spawn_consumer_(ConsumeMode mode, const char *name,
                             const char *dump_path, int *out_ready)
{
    int ready[2];
    pid_t pid;

    ASSERT(pipe(ready) == 0);
    pid = fork();
    ASSERT(pid >= 0);
    if (pid != 0) {
        close(ready[1]);
        if (out_ready)
            *out_ready = ready[0];
        else
            close(ready[0]);
        return pid;
    }

    {
        close(ready[0]);
        K26RlTapReader *r = NULL;
        FILE *dump = NULL;
        uint8_t *buf;
        uint32_t slot = 0;
        int tries;

        /* The producer enables the tap after this process starts, so
         * the first attach waits for the object rather than spinning
         * a fixed number of times and giving up. */
        for (tries = 0; tries < 10000 && !r; tries++) {
            struct timespec ts;
            if (k26rl_tap_attach(name, 1, &r) == K26RL_OK)
                break;
            ts.tv_sec = 0;
            ts.tv_nsec = 1000000;   /* 1 ms; 10 s in total */
            nanosleep(&ts, NULL);
        }
        if (!r)
            _exit(2);
        if (k26rl_tap_reader_info(r, &slot, NULL) != K26RL_OK)
            _exit(2);
        buf = (uint8_t *)malloc(slot);
        if (!buf)
            _exit(2);
        if (dump_path) {
            dump = fopen(dump_path, "wb");
            if (!dump)
                _exit(2);
        }
        /* Attached and ready to read: the producer may start. */
        if (write(ready[1], "a", 1) != 1)
            _exit(2);
        close(ready[1]);

        for (;;) {
            uint32_t len = 0;

            if (k26rl_tap_read(r, buf, slot, &len, NULL) != K26RL_OK)
                _exit(2);
            if (len) {
                if (dump) {
                    if (fwrite(&len, sizeof len, 1, dump) != 1 ||
                        fwrite(buf, len, 1, dump) != 1)
                        _exit(2);
                }
                if (mode == CONSUME_STALL) {
                    /* One frame, then nothing: a consumer that stops
                     * consuming must not stop the producer. */
                    for (;;)
                        sleep(3600);
                }
            }
            if (mode == CONSUME_FLAP) {
                k26rl_tap_detach(r);
                r = NULL;
                for (tries = 0; tries < 100000; tries++) {
                    if (k26rl_tap_attach(name, 0, &r) == K26RL_OK)
                        break;
                }
                if (!r)
                    break;   /* the producer closed and unlinked */
            }
            if (mode == CONSUME_DIE) {
                for (;;)
                    sleep(3600);
            }
            if (!len && k26rl_tap_reader_closed(r))
                break;
        }
        if (dump)
            fclose(dump);
        if (r)
            k26rl_tap_detach(r);
        _exit(0);
    }
}

/* ---- Frame decoding, one decoder over both transports -------------- */

typedef struct {
    uint32_t ordinal, env, episode, step;
    double obs[TAP_OBS];
    double act[TAP_ACT];
    double reward;
    uint32_t flags;
    double dt;
} StepRec;

typedef struct {
    uint32_t ordinal, env, episode, steps;
    uint16_t end_reason, fault_code;
} EndRec;

typedef struct {
    StepRec *recs;
    uint32_t count, cap;
    uint32_t starts, ends;
    EndRec last_end;
    uint32_t fault_steps;
} Decoded;

static void rec_push_(Decoded *d, const StepRec *r)
{
    if (d->count == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 256;
        d->recs = (StepRec *)realloc(d->recs, (size_t)d->cap * sizeof *d->recs);
        ASSERT(d->recs != NULL);
    }
    d->recs[d->count++] = *r;
}

/* A step-chunk payload, whatever its count: the file writes chunks of
 * up to steps_per_chunk in column-major order and the ring writes one
 * step per frame, which is the same layout at a count of one. One
 * decoder reads both, which is the point of the schema. */
static void decode_chunk_(Decoded *d, const uint8_t *p, uint32_t plen,
                          uint32_t episode_of_env[TAP_ENVS])
{
    uint32_t ord = rl_get_u32_(p);
    uint32_t env = rl_get_u32_(p + 4);
    uint32_t first = rl_get_u32_(p + 8);
    uint32_t K = rl_get_u32_(p + 12);
    const uint8_t *cols = p + 16;
    uint32_t ncols8 = TAP_OBS + TAP_ACT + 1;
    uint32_t i, j;

    ASSERT(env < TAP_ENVS);
    ASSERT(plen == 16 + K * (ncols8 * 8 + 12));
    for (i = 0; i < K; i++) {
        StepRec r;
        memset(&r, 0, sizeof r);
        r.ordinal = ord;
        r.env = env;
        r.episode = episode_of_env[env];
        r.step = first + i;
        for (j = 0; j < TAP_OBS; j++)
            r.obs[j] = rl_get_f64_(cols + ((size_t)j * K + i) * 8);
        for (j = 0; j < TAP_ACT; j++)
            r.act[j] = rl_get_f64_(cols + ((size_t)(TAP_OBS + j) * K + i) * 8);
        r.reward = rl_get_f64_(cols +
            ((size_t)(TAP_OBS + TAP_ACT) * K + i) * 8);
        r.flags = rl_get_u32_(cols + (size_t)ncols8 * K * 8 + (size_t)i * 4);
        r.dt = rl_get_f64_(cols + (size_t)ncols8 * K * 8 + (size_t)K * 4 +
                           (size_t)i * 8);
        if (r.flags & K26RL_FLAG_FAULT)
            d->fault_steps++;
        rec_push_(d, &r);
    }
}

/* Walk a frame stream, whichever transport produced it. The caller
 * supplies frames one at a time; episode indices come from the
 * episode-start frames, which is how a step record's episode is known
 * on both transports. */
static void decode_frame_(Decoded *d, const uint8_t *frame, uint32_t total,
                          uint32_t episode_of_env[TAP_ENVS])
{
    uint16_t kind = rl_get_u16_(frame + 0);
    uint32_t plen = rl_get_u32_(frame + 4);
    const uint8_t *p = frame + K26RL_EPISODE_FRAME_HEADER_SIZE;

    ASSERT(total == K26RL_EPISODE_FRAME_HEADER_SIZE + plen);
    switch (kind) {
    case K26RL_FRAME_EPISODE_START: {
        uint32_t env = rl_get_u32_(p + 4);
        ASSERT(env < TAP_ENVS);
        episode_of_env[env] = rl_get_u32_(p + 8);
        d->starts++;
        break;
    }
    case K26RL_FRAME_STEP_CHUNK:
        decode_chunk_(d, p, plen, episode_of_env);
        break;
    case K26RL_FRAME_EPISODE_END:
        d->last_end.ordinal = rl_get_u32_(p);
        d->last_end.env = rl_get_u32_(p + 4);
        d->last_end.episode = rl_get_u32_(p + 8);
        d->last_end.steps = rl_get_u32_(p + 12);
        d->last_end.end_reason = rl_get_u16_(p + 16);
        d->last_end.fault_code = rl_get_u16_(p + 18);
        d->ends++;
        break;
    default:
        break;
    }
}

static void decode_dump_(const char *path, Decoded *d)
{
    FILE *f = fopen(path, "rb");
    uint32_t episode_of_env[TAP_ENVS];
    uint8_t *buf = NULL;
    uint32_t cap = 0;

    ASSERT(f != NULL);
    memset(episode_of_env, 0, sizeof episode_of_env);
    for (;;) {
        uint32_t len;
        if (fread(&len, sizeof len, 1, f) != 1)
            break;
        if (len > cap) {
            cap = len;
            buf = (uint8_t *)realloc(buf, cap);
            ASSERT(buf != NULL);
        }
        ASSERT(fread(buf, len, 1, f) == 1);
        decode_frame_(d, buf, len, episode_of_env);
    }
    free(buf);
    fclose(f);
}

/* The episode file, read through the library's own reader and flattened
 * into the same records. */
static void decode_file_(const char *path, Decoded *d)
{
    K26RlEpisodeReader *r = NULL;
    K26RlEpisodeInfo info;
    uint32_t k;

    ASSERT(k26rl_episode_reader_open(path, &r) == K26RL_OK);
    ASSERT(k26rl_episode_reader_info(r, &info) == K26RL_OK);
    ASSERT(info.obs_total == TAP_OBS && info.act_total == TAP_ACT);
    for (k = 0; k < info.episode_count; k++) {
        uint32_t ord, env, ep, i, j;
        K26RlEpisodeData ed;

        ASSERT(k26rl_episode_reader_at(r, k, &ord, &env, &ep) == K26RL_OK);
        ASSERT(k26rl_episode_read(r, ord, env, ep, &ed) == K26RL_OK);
        d->starts++;
        d->ends++;
        for (i = 0; i < ed.step_count; i++) {
            StepRec rec;
            memset(&rec, 0, sizeof rec);
            rec.ordinal = ord;
            rec.env = env;
            rec.episode = ep;
            rec.step = i;
            for (j = 0; j < TAP_OBS; j++)
                rec.obs[j] = ed.obs[(size_t)i * TAP_OBS + j];
            for (j = 0; j < TAP_ACT; j++)
                rec.act[j] = ed.act[(size_t)i * TAP_ACT + j];
            rec.reward = ed.rewards[i];
            rec.flags = ed.flags[i];
            rec.dt = ed.applied_dt[i];
            rec_push_(d, &rec);
        }
        k26rl_episode_free(&ed);
    }
    k26rl_episode_reader_close(r);
}

static const StepRec *find_(const Decoded *d, const StepRec *key)
{
    uint32_t i;

    for (i = 0; i < d->count; i++) {
        const StepRec *c = &d->recs[i];
        if (c->ordinal == key->ordinal && c->env == key->env &&
            c->episode == key->episode && c->step == key->step)
            return c;
    }
    return NULL;
}

/* ---- Gate 12: the enable surface ------------------------------------ */

static void gate_enable_(const RlSurface *s)
{
    K26RlEnv *env = NULL;
    char over[K26RL_TAP_NAME_MAX + 3];
    const char *n = tap_name_(0, "enable");
    int before, after;

    ASSERT(s->abi_version() == K26RL_ABI_VERSION);
    /* Major equality and minor at-least, which is the check every
     * consumer makes: the tap arrived at minor 1, and a later minor
     * still carries it. */
    ASSERT((s->abi_version() >> 16) == 1 && (s->abi_version() & 0xFFFF) >= 1);

    /* A run that never asks for a tap creates nothing. */
    before = shm_objects_();
    ASSERT(before >= 0);
    drive_(s, 7, 5, NULL, NULL, NULL, -1);
    after = shm_objects_();
    ASSERT(after == before);

    /* Timing: at create, and after a reset before the next step, both
     * accepted; after a step, refused. */
    ASSERT(s->create(7, TAP_ENVS, &env) == K26RL_OK);
    ASSERT(s->tap(env, n) == K26RL_OK);
    ASSERT(s->tap(env, NULL) == K26RL_OK);
    ASSERT(s->reset(env) == K26RL_OK);
    ASSERT(s->tap(env, n) == K26RL_OK);
    ASSERT(s->tap(env, NULL) == K26RL_OK);
    {
        double act[TAP_ENVS * TAP_ACT];
        memset(act, 0, sizeof act);
        ASSERT(s->step(env, act) == K26RL_OK);
    }
    ASSERT(s->tap(env, n) == K26RL_E_OUTPUT_TIMING);
    s->destroy(env);

    /* Name refusals, and the bound stated in the decoded message. */
    ASSERT(s->create(7, TAP_ENVS, &env) == K26RL_OK);
    ASSERT(s->tap(env, "") == K26RL_E_TAP_NAME);
    ASSERT(s->tap(env, "has/separator") == K26RL_E_TAP_NAME);
    memset(over, 'a', K26RL_TAP_NAME_MAX + 1);
    over[K26RL_TAP_NAME_MAX + 1] = '\0';
    ASSERT(s->tap(env, over) == K26RL_E_TAP_NAME);
    ASSERT(strstr(s->status_str(K26RL_E_TAP_NAME), "63") != NULL);
    /* Exactly at the bound is not a refusal. */
    over[K26RL_TAP_NAME_MAX] = '\0';
    ASSERT(s->tap(env, over) == K26RL_OK);
    ASSERT(s->tap(env, NULL) == K26RL_OK);

    /* Exclusivity, disablement, and re-enablement of the same name. */
    ASSERT(s->tap(env, n) == K26RL_OK);
    {
        K26RlEnv *other = NULL;
        ASSERT(s->create(9, TAP_ENVS, &other) == K26RL_OK);
        ASSERT(s->tap(other, n) == K26RL_E_TAP_EXISTS);
        s->destroy(other);
    }
    ASSERT(s->tap(env, NULL) == K26RL_OK);
    ASSERT(s->tap(env, n) == K26RL_OK);

    /* The ceiling declaration is read on this surface. Measure the
     * ring this fixture makes, assert one MiB is genuinely below it,
     * then watch a one-MiB declaration refuse the same enable, a
     * malformed one refuse it too, and the withdrawn declaration
     * admit it again. */
    {
        K26RlTapReader *rd = NULL;
        uint32_t slot = 0, cnt = 0;

        ASSERT(k26rl_tap_attach(n, 1, &rd) == K26RL_OK);
        ASSERT(k26rl_tap_reader_info(rd, &slot, &cnt) == K26RL_OK);
        k26rl_tap_detach(rd);
        ASSERT((uint64_t)slot * cnt > (uint64_t)1 << 20);
        ASSERT(s->tap(env, NULL) == K26RL_OK);
        ASSERT(setenv(K26RL_TAP_CEILING_ENV, "1", 1) == 0);
        ASSERT(s->tap(env, n) == K26RL_E_GEOMETRY);
        ASSERT(setenv(K26RL_TAP_CEILING_ENV, "512M", 1) == 0);
        ASSERT(s->tap(env, n) == K26RL_E_GEOMETRY);
        ASSERT(unsetenv(K26RL_TAP_CEILING_ENV) == 0);
        ASSERT(s->tap(env, n) == K26RL_OK);
    }

    /* Destroy releases the name and leaves no object behind. */
    before = shm_objects_();
    s->destroy(env);
    after = shm_objects_();
    ASSERT(after == before - 1);

    printf("gate 12: enable surface, refusals, and disablement: OK\n");
}

static void gate_no_render_deps_(void)
{
    /* The compiled artifact links no rendering machinery: the tap is
     * shared memory, and every graphics dependency belongs to a
     * viewer that is not this. */
    int rc = system("readelf -d " WORK_DIR "/tap.rlenv.so"
                    " | grep -i NEEDED"
                    " | grep -qiE 'libGL|glfw|imgui|X11|EGL'");
    ASSERT(rc != -1);
    ASSERT(WIFEXITED(rc) && WEXITSTATUS(rc) == 1);   /* grep found none */
    printf("gate 12: artifact needs no rendering library: OK\n");
}

/* ---- Gate 9a to 9c: the simulation cannot tell ---------------------- */

static void gate_identity_(const RlSurface *s)
{
    const char *n = tap_name_(1, "ident");
    pid_t pid;
    int st = 0;
    int ready = -1;

    /* 9a: never tapped, against tapped with nobody listening. */
    drive_(s, 11, TAP_STEPS, WORK_DIR "/untapped.k26epi", NULL,
           WORK_DIR "/untapped.dump", -1);
    drive_(s, 11, TAP_STEPS, WORK_DIR "/tapped.k26epi", n,
           WORK_DIR "/tapped.dump", -1);
    ASSERT(rl_files_equal_(WORK_DIR "/untapped.k26epi",
                           WORK_DIR "/tapped.k26epi"));
    ASSERT(rl_files_equal_(WORK_DIR "/untapped.dump",
                           WORK_DIR "/tapped.dump"));
    printf("gate 9a: episode file and getter streams identical with the"
           " tap enabled: OK\n");

    /* 9b: a consumer reading continuously, then one that accepts a
     * single frame and blocks for the remainder of the run. Neither
     * may move a byte, and the stalled one is the arm that proves a
     * slow consumer cannot hold a step. */
    pid = spawn_consumer_(CONSUME_ALL, n, WORK_DIR "/watch.frames", &ready);
    drive_(s, 11, TAP_STEPS, WORK_DIR "/watched.k26epi", n, NULL, ready);
    ASSERT(waitpid(pid, &st, 0) == pid);
    ASSERT(WIFEXITED(st) && WEXITSTATUS(st) == 0);
    ASSERT(rl_files_equal_(WORK_DIR "/untapped.k26epi",
                           WORK_DIR "/watched.k26epi"));

    pid = spawn_consumer_(CONSUME_STALL, n, NULL, &ready);
    drive_(s, 11, TAP_STEPS, WORK_DIR "/stalled.k26epi", n, NULL, ready);
    ASSERT(kill(pid, SIGKILL) == 0);
    ASSERT(waitpid(pid, &st, 0) == pid);
    ASSERT(rl_files_equal_(WORK_DIR "/untapped.k26epi",
                           WORK_DIR "/stalled.k26epi"));
    printf("gate 9b: consumer absent, reading, and stalled all leave the"
           " run identical: OK\n");

    /* 9c: a consumer attaching and detaching throughout, and one
     * killed outright with its mapping live. */
    pid = spawn_consumer_(CONSUME_FLAP, n, NULL, &ready);
    drive_(s, 11, TAP_STEPS, WORK_DIR "/flapped.k26epi", n, NULL, ready);
    (void)kill(pid, SIGKILL);
    ASSERT(waitpid(pid, &st, 0) == pid);
    ASSERT(rl_files_equal_(WORK_DIR "/untapped.k26epi",
                           WORK_DIR "/flapped.k26epi"));

    pid = spawn_consumer_(CONSUME_DIE, n, NULL, &ready);
    {
        K26RlEnv *env = NULL;
        double act[TAP_ENVS * TAP_ACT];

        ASSERT(s->create(11, TAP_ENVS, &env) == K26RL_OK);
        ASSERT(s->output(env, WORK_DIR "/killed.k26epi") == K26RL_OK);
        ASSERT(s->tap(env, n) == K26RL_OK);
        wait_ready_(ready);
        for (int t = 0; t < TAP_STEPS; t++) {
            for (uint32_t e = 0; e < TAP_ENVS; e++)
                for (uint32_t c = 0; c < TAP_ACT; c++)
                    act[e * TAP_ACT + c] = act_((uint32_t)t, e, c);
            /* Kill the consumer with its mapping live, mid-run. */
            if (t == TAP_STEPS / 2) {
                ASSERT(kill(pid, SIGKILL) == 0);
                ASSERT(waitpid(pid, &st, 0) == pid);
                ASSERT(WIFSIGNALED(st));
            }
            ASSERT(s->step(env, act) == K26RL_OK);
        }
        s->destroy(env);
    }
    ASSERT(rl_files_equal_(WORK_DIR "/untapped.k26epi",
                           WORK_DIR "/killed.k26epi"));
    printf("gate 9c: attach, detach, and a consumer killed mid-run leave"
           " the run identical: OK\n");
}

/* ---- Gate 9d: read-only, proven by the fault ------------------------ */

static sigjmp_buf fault_jmp_;
static volatile sig_atomic_t faulted_;

static void segv_(int sig)
{
    (void)sig;
    faulted_ = 1;
    siglongjmp(fault_jmp_, 1);
}

static void gate_readonly_(const RlSurface *s)
{
    K26RlEnv *env = NULL;
    const char *n = tap_name_(2, "ro");
    char object[K26RL_TAP_NAME_MAX + 16];
    struct sigaction sa, old;
    struct stat sb;
    volatile uint8_t *map;
    int fd;

    ASSERT(s->create(13, TAP_ENVS, &env) == K26RL_OK);
    ASSERT(s->tap(env, n) == K26RL_OK);

    snprintf(object, sizeof object, "%s%s", K26RL_TAP_OBJECT_PREFIX, n);
    fd = shm_open(object, O_RDONLY, 0);
    ASSERT(fd >= 0);
    ASSERT(fstat(fd, &sb) == 0);
    /* Telemetry from a training run is the owner's, not the world's. */
    ASSERT((sb.st_mode & 0777) == 0600);
    map = (volatile uint8_t *)mmap(NULL, (size_t)sb.st_size, PROT_READ,
                                   MAP_SHARED, fd, 0);
    close(fd);
    ASSERT(map != MAP_FAILED);

    /* Reading is fine. */
    ASSERT(map[0] == 'K');

    /* Writing is not, and that is proven by catching the fault rather
     * than by trusting the flag that was passed to mmap. A change that
     * mapped this read-write would pass silently without this. */
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = segv_;
    sigemptyset(&sa.sa_mask);
    ASSERT(sigaction(SIGSEGV, &sa, &old) == 0);
    faulted_ = 0;
    if (sigsetjmp(fault_jmp_, 1) == 0) {
        map[K26RL_TAP_OFF_HEAD] = 0xFF;   /* must fault */
        /* Reached only if the store succeeded, which is the defect
         * this gate exists to catch. */
        ASSERT(0 && "a store through the consumer mapping succeeded");
    }
    ASSERT(sigaction(SIGSEGV, &old, NULL) == 0);
    ASSERT(faulted_ == 1);

    munmap((void *)map, (size_t)sb.st_size);
    s->destroy(env);
    printf("gate 9d: consumer mapping is read-only, proven by the fault,"
           " and the object is mode 0600: OK\n");
}

/* ---- Gate 8: cross-transport equality ------------------------------- */

static void gate_cross_transport_(const RlSurface *s)
{
    const char *n = tap_name_(3, "xport");
    Decoded ring, file;
    pid_t pid;
    int st = 0, ready = -1;
    uint32_t i, matched = 0;

    memset(&ring, 0, sizeof ring);
    memset(&file, 0, sizeof file);

    pid = spawn_consumer_(CONSUME_ALL, n, WORK_DIR "/xport.frames", &ready);
    drive_(s, 17, TAP_STEPS, WORK_DIR "/xport.k26epi", n, NULL, ready);
    ASSERT(waitpid(pid, &st, 0) == pid);
    ASSERT(WIFEXITED(st) && WEXITSTATUS(st) == 0);

    decode_dump_(WORK_DIR "/xport.frames", &ring);
    decode_file_(WORK_DIR "/xport.k26epi", &file);

    /* Not permitted to pass vacuously: the consumer must have covered
     * whole episodes, boundary frames included. */
    ASSERT(ring.count > 0);
    ASSERT(ring.starts >= TAP_ENVS + 1);
    ASSERT(ring.ends >= 1);
    ASSERT(file.count > 0);

    for (i = 0; i < ring.count; i++) {
        const StepRec *a = &ring.recs[i];
        const StepRec *b = find_(&file, a);
        uint32_t j;

        if (!b)
            continue;   /* the file closed before this episode ended */
        for (j = 0; j < TAP_OBS; j++)
            ASSERT(memcmp(&a->obs[j], &b->obs[j], sizeof(double)) == 0);
        for (j = 0; j < TAP_ACT; j++)
            ASSERT(memcmp(&a->act[j], &b->act[j], sizeof(double)) == 0);
        ASSERT(memcmp(&a->reward, &b->reward, sizeof(double)) == 0);
        ASSERT(a->flags == b->flags);
        ASSERT(memcmp(&a->dt, &b->dt, sizeof(double)) == 0);
        matched++;
    }
    /* A whole episode's worth of steps at the least, so the equality
     * above is a statement about a run and not about one frame. */
    ASSERT(matched >= 12);
    printf("gate 8: %u ring step records matched the file bitwise"
           " (%u ring records, %u file records, %u starts, %u ends): OK\n",
           matched, ring.count, file.count, ring.starts, ring.ends);

    /* One schema, two transports, checked where it is actually
     * claimed: the file-header payload the ring publishes in its
     * preamble is the payload the file carries in its first frame. */
    {
        K26RlTapReader *r = NULL;
        K26RlEnv *env = NULL;
        const uint8_t *tap_frame = NULL, *spec = NULL;
        uint32_t tap_len = 0, spec_len = 0;
        FILE *f;
        uint8_t fhdr[24];
        uint8_t *fpay;
        uint32_t fplen;
        const char *n2 = tap_name_(4, "hdr");

        ASSERT(s->create(17, TAP_ENVS, &env) == K26RL_OK);
        ASSERT(s->tap(env, n2) == K26RL_OK);
        ASSERT(k26rl_tap_attach(n2, 1, &r) == K26RL_OK);
        ASSERT(k26rl_tap_reader_header(r, &tap_frame, &tap_len) == K26RL_OK);

        f = fopen(WORK_DIR "/xport.k26epi", "rb");
        ASSERT(f != NULL);
        ASSERT(fseek(f, 8, SEEK_SET) == 0);   /* past the file magic */
        ASSERT(fread(fhdr, sizeof fhdr, 1, f) == 1);
        ASSERT(rl_get_u16_(fhdr) == K26RL_FRAME_FILE_HEADER);
        fplen = rl_get_u32_(fhdr + 4);
        ASSERT(fplen == tap_len - K26RL_EPISODE_FRAME_HEADER_SIZE);
        fpay = (uint8_t *)malloc(fplen);
        ASSERT(fpay != NULL);
        ASSERT(fread(fpay, fplen, 1, f) == 1);
        fclose(f);
        ASSERT(memcmp(fpay, tap_frame + K26RL_EPISODE_FRAME_HEADER_SIZE,
                      fplen) == 0);
        free(fpay);

        /* And the spec the ring publishes is the artifact's own. */
        ASSERT(k26rl_tap_reader_spec(r, &spec, &spec_len) == K26RL_OK);
        {
            int32_t want = s->spec(env, NULL, 0);
            uint8_t *blob;
            ASSERT(want > 0 && (uint32_t)want == spec_len);
            blob = (uint8_t *)malloc((size_t)want);
            ASSERT(blob != NULL);
            ASSERT(s->spec(env, blob, (uint32_t)want) == want);
            ASSERT(memcmp(blob, spec, spec_len) == 0);
            free(blob);
        }
        k26rl_tap_detach(r);
        s->destroy(env);
    }
    printf("gate 8: both transports publish the same file-header payload"
           " and the same spec blob: OK\n");

    free(ring.recs);
    free(file.recs);
}

/* ---- Gate 14: late attach ------------------------------------------- */

static void gate_late_attach_(const RlSurface *s)
{
    const char *n = tap_name_(5, "late");
    K26RlEnv *env = NULL;
    K26RlTapReader *r = NULL;
    Decoded ring, file;
    double act[TAP_ENVS * TAP_ACT];
    uint32_t episode_of_env[TAP_ENVS];
    uint8_t *buf = NULL;
    uint32_t slot = 0;
    uint64_t lost_total = 0;
    int saw_start = 0, saw_end = 0;
    uint32_t i;

    memset(&ring, 0, sizeof ring);
    memset(&file, 0, sizeof file);
    memset(episode_of_env, 0, sizeof episode_of_env);

    ASSERT(s->create(19, TAP_ENVS, &env) == K26RL_OK);
    ASSERT(s->output(env, WORK_DIR "/late.k26epi") == K26RL_OK);
    ASSERT(s->tap(env, n) == K26RL_OK);

    /* Run well into the first episodes before anyone attaches. */
    for (int t = 0; t < 8; t++) {
        for (uint32_t e = 0; e < TAP_ENVS; e++)
            for (uint32_t c = 0; c < TAP_ACT; c++)
                act[e * TAP_ACT + c] = act_((uint32_t)t, e, c);
        ASSERT(s->step(env, act) == K26RL_OK);
    }

    ASSERT(k26rl_tap_attach(n, 0, &r) == K26RL_OK);
    ASSERT(k26rl_tap_reader_info(r, &slot, NULL) == K26RL_OK);
    buf = (uint8_t *)malloc(slot);
    ASSERT(buf != NULL);

    /* The preamble outlives every overwrite, so the spec comes back
     * whole however late the attach. */
    {
        const uint8_t *spec = NULL;
        uint32_t spec_len = 0;
        int32_t want = s->spec(env, NULL, 0);
        uint8_t *blob;

        ASSERT(k26rl_tap_reader_spec(r, &spec, &spec_len) == K26RL_OK);
        ASSERT(want > 0 && (uint32_t)want == spec_len);
        blob = (uint8_t *)malloc((size_t)want);
        ASSERT(blob != NULL);
        ASSERT(s->spec(env, blob, (uint32_t)want) == want);
        ASSERT(memcmp(blob, spec, spec_len) == 0);
        free(blob);
    }

    for (int t = 8; t < TAP_STEPS; t++) {
        for (uint32_t e = 0; e < TAP_ENVS; e++)
            for (uint32_t c = 0; c < TAP_ACT; c++)
                act[e * TAP_ACT + c] = act_((uint32_t)t, e, c);
        ASSERT(s->step(env, act) == K26RL_OK);
        for (;;) {
            uint32_t len = 0;
            uint64_t lost = 0;
            ASSERT(k26rl_tap_read(r, buf, slot, &len, &lost) == K26RL_OK);
            lost_total += lost;
            if (!len)
                break;
            if (rl_get_u16_(buf) == K26RL_FRAME_EPISODE_START)
                saw_start = 1;
            if (rl_get_u16_(buf) == K26RL_FRAME_EPISODE_END && saw_start)
                saw_end = 1;
            /* Records before the first episode-start this reader saw
             * belong to an episode it cannot name, and it claims
             * nothing about them. */
            if (saw_start)
                decode_frame_(&ring, buf, len, episode_of_env);
        }
    }
    /* A late reader that kept up loses nothing: it simply was not
     * there for what came before. */
    ASSERT(lost_total == 0);
    ASSERT(saw_start && saw_end);
    ASSERT(ring.count > 0);

    s->destroy(env);
    k26rl_tap_detach(r);
    free(buf);

    decode_file_(WORK_DIR "/late.k26epi", &file);
    for (i = 0; i < ring.count; i++) {
        const StepRec *a = &ring.recs[i];
        const StepRec *b = find_(&file, a);
        uint32_t j;

        if (!b)
            continue;
        for (j = 0; j < TAP_OBS; j++)
            ASSERT(memcmp(&a->obs[j], &b->obs[j], sizeof(double)) == 0);
        ASSERT(memcmp(&a->reward, &b->reward, sizeof(double)) == 0);
        ASSERT(a->flags == b->flags);
    }
    printf("gate 14: late attach recovers the spec and matches the file"
           " from its first whole episode (%u records): OK\n", ring.count);

    free(ring.recs);
    free(file.recs);
}

/* ---- Gate 8, fault arm: the one step record that is not a
 *      transition ------------------------------------------------- */

/* A faulted episode is where the format's rules are least ordinary:
 * the final step record completes no transition, carries the faulting
 * call's actions with the pre-step observations, zero reward and zero
 * applied dt, and the fault bit alone, and the episode-end frame
 * carries the reason from the status registry. The tap publishes that
 * record through the same call the file writer takes, so this arm
 * checks that both transports agree on the awkward case rather than
 * only on the easy one. */
static void gate_fault_(void)
{
    const char *n = tap_name_(6, "fault");
    void *so;
    RlSurface s;
    K26RlEnv *env = NULL;
    Decoded ring, file;
    double act[TAP_ENVS * TAP_ACT];
    pid_t pid;
    int st = 0, ready = -1;
    uint32_t i;
    uint16_t ring_code;
    int saw = 0;

    memset(&ring, 0, sizeof ring);
    memset(&file, 0, sizeof file);

    rl_write_file_(WORK_DIR "/tapfault.kfl", FAULT_KFL);
    rl_compile_(WORK_DIR "/tapfault.kfl", WORK_DIR "/tapfault", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/tapfault.rlenv.so"));
    so = rl_dlopen_(WORK_DIR "/tapfault.rlenv.so");
    rl_resolve_surface_(so, &s);

    pid = spawn_consumer_(CONSUME_ALL, n, WORK_DIR "/fault.frames", &ready);

    ASSERT(s.create(23, TAP_ENVS, &env) == K26RL_OK);
    ASSERT(s.output(env, WORK_DIR "/fault.k26epi") == K26RL_OK);
    ASSERT(s.tap(env, n) == K26RL_OK);
    wait_ready_(ready);
    for (int t = 0; t < 10; t++) {
        memset(act, 0, sizeof act);
        /* Environment 0 divides by zero at a step of our choosing;
         * environment 1 keeps stepping normally beside it. */
        if (t == FAULT_AT)
            act[0] = 1.0;
        ASSERT(s.step(env, act) == K26RL_OK);
    }
    s.destroy(env);
    ASSERT(waitpid(pid, &st, 0) == pid);
    ASSERT(WIFEXITED(st) && WEXITSTATUS(st) == 0);

    decode_dump_(WORK_DIR "/fault.frames", &ring);
    ASSERT(ring.fault_steps == 1);

    /* The ring's fault record: zero reward, zero applied dt, the
     * fault bit and nothing else. */
    for (i = 0; i < ring.count; i++) {
        const StepRec *a = &ring.recs[i];
        if (!(a->flags & K26RL_FLAG_FAULT))
            continue;
        ASSERT(a->env == 0);
        ASSERT(a->flags == K26RL_FLAG_FAULT);
        ASSERT(a->reward == 0.0);
        ASSERT(a->dt == 0.0);
        /* The faulting call's action reached the record. */
        ASSERT(a->act[0] == 1.0);
        saw = 1;
    }
    ASSERT(saw);

    /* And the file agrees, record for record and reason for reason. */
    decode_file_(WORK_DIR "/fault.k26epi", &file);
    saw = 0;
    for (i = 0; i < ring.count; i++) {
        const StepRec *a = &ring.recs[i];
        const StepRec *b = find_(&file, a);
        uint32_t j;

        if (!b)
            continue;
        for (j = 0; j < TAP_OBS; j++)
            ASSERT(memcmp(&a->obs[j], &b->obs[j], sizeof(double)) == 0);
        for (j = 0; j < TAP_ACT; j++)
            ASSERT(memcmp(&a->act[j], &b->act[j], sizeof(double)) == 0);
        ASSERT(memcmp(&a->reward, &b->reward, sizeof(double)) == 0);
        ASSERT(a->flags == b->flags);
        ASSERT(memcmp(&a->dt, &b->dt, sizeof(double)) == 0);
        if (a->flags & K26RL_FLAG_FAULT)
            saw = 1;
    }
    ASSERT(saw);   /* the fault record itself was one of the matches */

    /* The end frame's reason and code, from the ring, against the
     * file read back through the library's own reader. */
    {
        K26RlEpisodeReader *r = NULL;
        K26RlEpisodeData ed;
        uint32_t k, found = 0;

        ring_code = 0;
        for (i = 0; i < ring.count; i++) {
            if (ring.recs[i].flags & K26RL_FLAG_FAULT)
                break;
        }
        ASSERT(i < ring.count);

        ASSERT(k26rl_episode_reader_open(WORK_DIR "/fault.k26epi", &r) ==
               K26RL_OK);
        {
            K26RlEpisodeInfo info;
            ASSERT(k26rl_episode_reader_info(r, &info) == K26RL_OK);
            for (k = 0; k < info.episode_count; k++) {
                uint32_t ord, e2, ep;
                ASSERT(k26rl_episode_reader_at(r, k, &ord, &e2, &ep) ==
                       K26RL_OK);
                ASSERT(k26rl_episode_read(r, ord, e2, ep, &ed) == K26RL_OK);
                if (ed.end_reason == K26RL_END_FAULT) {
                    ASSERT(ed.fault_code != 0);
                    ASSERT(e2 == ring.recs[i].env);
                    ASSERT(ep == ring.recs[i].episode);
                    /* The step count includes the fault record, which
                     * is the format's one non-transition. */
                    ASSERT(ed.step_count == ring.recs[i].step + 1);
                    ring_code = ed.fault_code;
                    found++;
                }
                k26rl_episode_free(&ed);
            }
        }
        k26rl_episode_reader_close(r);
        ASSERT(found == 1);
        ASSERT(ring_code != 0);
    }

    printf("gate 8: the faulted episode agrees across both transports"
           " (fault record and end reason, code %u): OK\n",
           (unsigned)ring_code);

    free(ring.recs);
    free(file.recs);
    dlclose(so);
}

int main(void)
{
    if (!rl_libs_present_("test_rl_tap"))
        return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/tap.kfl", TAP_KFL);
    rl_compile_(WORK_DIR "/tap.kfl", WORK_DIR "/tap", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/tap.rlenv.so"));

    {
        void *so = rl_dlopen_(WORK_DIR "/tap.rlenv.so");
        RlSurface s;

        rl_resolve_surface_(so, &s);
        gate_enable_(&s);
        gate_no_render_deps_();
        gate_identity_(&s);
        gate_readonly_(&s);
        gate_cross_transport_(&s);
        gate_late_attach_(&s);
        dlclose(so);
    }
    gate_fault_();
    printf("test_rl_tap: gates 8 (transitions and the fault record),"
           " 9a to 9d, 12, and 14 passed\n");
    return 0;
}
