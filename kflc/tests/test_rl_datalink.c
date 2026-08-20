/* test_rl_datalink.c: the gated information state and the datalink.
 *
 * Two surfaces that need each other. An information state fed with
 * truth every sub-advance already holds every declared target at light
 * delay, so a peer's report of the same target arrives older and is
 * dropped without changing a channel; `source=` gates the push on a
 * declared detection's verdict, and the datalink carries what is left
 * to carry, at a physical cost and a physical delay.
 *
 * Gates:
 *
 *   1. The gate. A `source=`-gated information state's track follows
 *      its named detection's verdict step for step, across two
 *      transitions in each direction, so it neither pushes below the
 *      threshold nor withholds a push at or above it. Beside it, the
 *      same world with a detection that always meets its threshold
 *      records byte for byte what the ungated form records, which is
 *      the other half: a gate that withheld any push whose verdict met
 *      the threshold would move those bytes. And the ungated form
 *      itself is compared byte for byte against the compiler from
 *      before the key existed.
 *
 *   2. The value the link makes reachable. A receiver whose own
 *      gated detection has never seen a target reads `_valid` 1.0 with
 *      the shared entry's light-time age after closure, and 0.0 in the
 *      same world with the datalink removed. The age is held against a
 *      light time computed here from the published range, and the
 *      shared position against the transmitter's own published one.
 *
 *   3. The budget. The closure range is solved here from the
 *      declared keys through the one-way Friis relation, and two
 *      worlds are built a per cent either side of it: the nearer
 *      closes and the farther does not. A three-member world holds the
 *      same statement in one program. Then one perturbation per
 *      declared key, each in the direction that should break a closure
 *      standing two per cent above threshold, and each required to
 *      break it.
 *
 *   4. What may not cross. An entry does not cross between two
 *      communities in one world; an entry for a target the receiver
 *      declares no track over reaches no transfer at all, read off the
 *      artifact's own transfer table; and a program whose tracks
 *      exceed the per-observer cap is refused naming the payload, the
 *      count and the limit, in a world that carries a datalink.
 *
 *   5. Determinism and order. Two processes at one seed produce
 *      identical episode files for a world with a datalink and gated
 *      information states; the payload declaration order permuted
 *      changes the artifact's transfer table, as the design fixes it
 *      should, and changes no recorded byte.
 *
 * Beside them, every refusal this surface adds, an acceptance arm that
 * reaches a real artifact rather than stopping at the check, the
 * single-generator rule over this surface's own entry points, the one
 * limit it reports rather than refuses, and one arm that measures the
 * propagation delay itself: a peer whose own light time exceeds the target's cannot
 * help, however strong its signal, because its report arrives older
 * than the light from the target and falls outside the history the
 * observer's retarded time has to lie in.
 *
 * Red controls, each a substitution inside the compiler rather than a
 * symbol a link line can replace, so each is run by hand and recorded
 * with this work rather than built here. Every one of them was applied,
 * the gate was run, and the arm named beside it rejected it:
 *   - the monostatic radar evaluator wired in place of the one-way
 *     budget, rejected by the shared-track arm before the
 *     boundary arm was reached, the closure having moved by orders
 *     of magnitude;
 *   - the transfer table's receiver loop reversed, rejected by the
 *     receiver-order arm;
 *   - the receiver's declared-track filter deleted, rejected by the
 *     transfer-table arm;
 *   - the gate made never to withhold a push, rejected by the
 *     verdict arm;
 *   - the light time removed from an offer's arrival, rejected by the
 *     propagation-delay arm.
 *
 * Pattern: rl_gate_util.h. Refusal arms drive ./bin/kflc --check and
 * assert on the captured diagnostic; behaviour arms compile the
 * artifact and drive the frozen surface. Needs the sibling stack
 * archives; skips with 77 when they are absent.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_datalink_test"

/* The speed of light and Boltzmann's constant as the stack's own
 * header carries them, so the figures this gate computes and the ones
 * the artifact computes rest on the same two numbers. */
#include "k26astro_core/consts.h"

static int g_arms;

/* The integers of one emitted table, in order. Some of what this
 * surface fixes is a compile-time table rather than a channel: a
 * transfer that is never built cannot be seen from outside, so where
 * that is what the design fixes, the artifact's own table is what the
 * arm reads. */
static int table_ints_(const char *src, const char *name, int *out,
                       int cap);

/* ---- Sources -------------------------------------------------------- */

/* The declared radio keys of a datalink, in the order this gate
 * perturbs them. Naming them once here is what lets the budget arm
 * move one key at a time without a second copy of the key set. */

enum {
    K_P_TX = 0, K_G_TX, K_G_RX, K_FREQ, K_LOSS, K_BW, K_T_SYS, K_NF,
    K_THR, K_COUNT
};

static const char *const RADIO_KEY_[K_COUNT] = {
    "p_tx_w", "g_tx_db", "g_rx_db", "freq_hz", "loss_sys_db",
    "bandwidth_hz", "t_sys_k", "noise_figure", "snr_threshold"
};

/* A two-watt S-band link with modest antennas, which is the order a
 * small craft carries. The figures are this fixture's own: they are
 * chosen to put the closure range inside a world a gate can build,
 * and nothing here claims them as any real radio's. */
static const double RADIO_[K_COUNT] = {
    2.0, 3.0, 3.0, 2.2e9, 2.0, 1.0e6, 500.0, 2.0, 6.0
};

/* The one-way link budget, computed here independently of the
 * artifact: received power as transmit power times both gains times
 * the wavelength squared, over the square of four pi times the range,
 * less the declared system loss, against the thermal noise floor of
 * the declared bandwidth, system temperature and noise figure.
 *
 * It is written out rather than called, because what it is here to
 * check is that the artifact computes this and not the monostatic
 * radar relation, and a shared implementation would agree with either
 * one of them. */
static double friis_snr_(const double *v, double range_m)
{
    double lambda = K26A_C / v[K_FREQ];
    double g_tx = pow(10.0, v[K_G_TX] / 10.0);
    double g_rx = pow(10.0, v[K_G_RX] / 10.0);
    double loss = pow(10.0, v[K_LOSS] / 10.0);
    double spread = 4.0 * K26A_PI * range_m;
    double p_rx = (v[K_P_TX] * g_tx * g_rx * lambda * lambda)
                / (spread * spread * loss);
    double noise = K26A_K_BOLTZMANN * v[K_T_SYS] * v[K_BW] * v[K_NF];
    return p_rx / noise;
}

/* The range at which that budget sits exactly at the declared
 * threshold. The relation is one over range squared, so the range
 * follows in closed form. */
static double friis_range_(const double *v)
{
    double lambda = K26A_C / v[K_FREQ];
    double g_tx = pow(10.0, v[K_G_TX] / 10.0);
    double g_rx = pow(10.0, v[K_G_RX] / 10.0);
    double loss = pow(10.0, v[K_LOSS] / 10.0);
    double noise = K26A_K_BOLTZMANN * v[K_T_SYS] * v[K_BW] * v[K_NF];
    return sqrt((v[K_P_TX] * g_tx * g_rx * lambda * lambda)
                / (16.0 * K26A_PI * K26A_PI * loss * noise * v[K_THR]));
}

static void radio_str_(char *out, size_t cap, const double *v)
{
    int n = 0;
    for (int i = 0; i < K_COUNT; i++) {
        n += snprintf(out + n, cap - (size_t)n, "%s%s=%.17g",
                      i ? " " : "", RADIO_KEY_[i], v[i]);
    }
    ASSERT((size_t)n < cap);
}

/* The world every behaviour arm is built on.
 *
 * `drone_1` carries a radar that reaches the bogey and `drone_2` one
 * that cannot: a watt against a megawatt, which is the difference
 * between a craft that is looking and one that is only listening. Both
 * carry an information state gated by their own radar, so drone_2's
 * picture of the bogey can only ever come from its peer.
 *
 * The distances are what make the statement possible. The bogey is
 * three thousand kilometres away, which is ten milliseconds of light
 * time, and the sub-advance is two and a half; so a peer's report,
 * which is one sub-advance and one link light time old when it lands,
 * is still fresher than the light from the target itself. That is the
 * whole physical content of a shared track picture, and a world where
 * the peer is farther in light time than the target has nothing to
 * share.
 *
 * The bogey turns, so its silhouette and therefore the radar's
 * statistic move through a band; a threshold inside that band is what
 * makes the gate open and close during an episode.
 */
#define DL_HEAD \
    "form DL\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body drone_1 assembly=\"calibration_box.k26asm\"" \
    " parent=earth pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0\n"

#define DL_EYE1(thr) \
    "    astro_payload eye1 body=drone_1 kind=detect_radar" \
    " p_tx_w=1.0e6 g_tx_db=50.0 g_rx_db=50.0 freq_hz=1.0e10" \
    " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0" \
    " noise_figure=2.0 snr_threshold=" thr "\n"

#define DL_EYE(n, body, thr) \
    "    astro_payload eye" n " body=" body " kind=detect_radar" \
    " p_tx_w=1.0 g_tx_db=3.0 g_rx_db=3.0 freq_hz=1.0e10" \
    " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0" \
    " noise_figure=2.0 snr_threshold=" thr "\n"

#define DL_EPISODE \
    "    episode\n" \
    "        control_dt 0.025\n" \
    "        substeps 10\n" \
    "        horizon 60\n" \
    "    end\n"

#define DL_AGENTS \
    "    agent alpha\n" \
    "        action nudge1 box -1.0 1.0 default 0.0\n" \
    "        observe detect eye1 of bogey as see1\n" \
    "        observe track pic1 of bogey as trk1\n" \
    "        objective\n" \
    "            reward alpha.see1_snr\n" \
    "        end\n" \
    "    end\n" \
    "    agent beta\n" \
    "        action nudge2 box -1.0 1.0 default 0.0\n" \
    "        observe detect eye2 of bogey as see2\n" \
    "        observe track pic2 of bogey as trk2\n" \
    "        objective\n" \
    "            reward beta.trk2_valid\n" \
    "        end\n" \
    "    end\n" \
    "    on_step\n" \
    "        drone_1.vel_x = drone_1.vel_x + nudge1\n" \
    "        drone_2.vel_x = drone_2.vel_x + nudge2\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* Build the two-member world.
 *
 * `sep_m`  drone_2's offset from drone_1, metres
 * `spin`   the bogey's rate about its own third axis
 * `thr1`   drone_1's radar threshold, as source text
 * `radio`  the datalink keys, or NULL for a world with no datalink
 * `net2`   drone_2's community name
 * `src2`   1 for a gated information state on drone_2, 0 for the
 *          truth-fed form the key existed to change
 */
static void dl_world_(char *out, size_t cap, double sep_m,
                      const char *spin, const char *thr1,
                      const char *thr2, const char *radio,
                      const char *net2, int src2)
{
    int n = snprintf(out, cap, "%s", DL_HEAD);
    n += snprintf(out + n, cap - (size_t)n,
        "    astro_body drone_2 assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=%.17g pos_z=0.0 vel_x=0.0"
        " vel_y=7546.0 vel_z=0.0 quat_w=1.0\n"
        "    astro_body bogey assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=3.0e6 pos_z=0.0 vel_x=0.0"
        " vel_y=7546.0 vel_z=0.0 quat_w=1.0 omega_z=%s\n",
        sep_m, spin);
    n += snprintf(out + n, cap - (size_t)n, DL_EYE1("%s"), thr1);
    n += snprintf(out + n, cap - (size_t)n, DL_EYE("2", "drone_2", "%s"),
                  thr2);
    n += snprintf(out + n, cap - (size_t)n,
        "    astro_payload pic1 body=drone_1 kind=infostate"
        " history=1024 source=eye1\n"
        "    astro_payload pic2 body=drone_2 kind=infostate"
        " history=1024%s\n", src2 ? " source=eye2" : "");
    if (radio) {
        n += snprintf(out + n, cap - (size_t)n,
            "    astro_payload link1 body=drone_1 kind=datalink"
            " network=kite rate_hz=2000.0 %s\n"
            "    astro_payload link2 body=drone_2 kind=datalink"
            " network=%s rate_hz=2000.0 %s\n", radio, net2, radio);
    }
    n += snprintf(out + n, cap - (size_t)n, "%s%s", DL_EPISODE, DL_AGENTS);
    ASSERT((size_t)n < cap);
}

/* ---- Driving -------------------------------------------------------- */

/* The channels of one run, by published name, over `steps` steps. */

typedef struct {
    double  v[64][64];      /* [step][channel] */
    int     n_steps;
    RlSpecView spec;
    uint8_t blob[8192];
    uint32_t blob_len;
} DlRun;

static void dl_run_(DlRun *r, const char *src, const char *stem, int steps)
{
    char path[512], out[512], so[512];

    ASSERT(steps <= 64);
    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    snprintf(so, sizeof so, WORK_DIR "/%s.rlenv.so", stem);
    rl_write_file_(path, src);
    rl_compile_(path, out, WORK_DIR);

    void *h = rl_dlopen_(so);
    RlSurface s;
    rl_resolve_surface_(h, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
    int32_t n = s.spec(env, r->blob, sizeof r->blob);
    ASSERT(n > 0);
    r->blob_len = (uint32_t)n;
    rl_parse_spec_(r->blob, r->blob_len, &r->spec);
    double act[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    uint16_t fault = 0;
    for (int k = 0; k < steps; k++) {
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, r->v[k]) == K26RL_OK);
        ASSERT(s.fault_codes(env, &fault) == K26RL_OK);
        if (fault != 0) {
            fprintf(stderr, "FAIL %s: step %d faulted (%u)\n", stem,
                    k + 1, (unsigned)fault);
            exit(1);
        }
    }
    r->n_steps = steps;
    s.destroy(env);
    dlclose(h);
}

static double dl_ch_(const DlRun *r, int step, const char *name)
{
    int i = find_channel_(r->blob, r->blob_len, name);
    if (i < 0) {
        fprintf(stderr, "FAIL: no channel `%s` in this artifact\n", name);
        exit(1);
    }
    return r->v[step][i];
}

/* ---- Check-only arms ------------------------------------------------ */

static int check_(const char *src, char **out_log)
{
    rl_write_file_(WORK_DIR "/case.kfl", src);
    int rc = system("./bin/kflc --check " WORK_DIR "/case.kfl > "
                    WORK_DIR "/case.log 2>&1");
    if (out_log) {
        FILE *f = fopen(WORK_DIR "/case.log", "rb");
        ASSERT(f != NULL);
        static char buf[65536];
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = '\0';
        fclose(f);
        *out_log = buf;
    }
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

static void must_refuse_(const char *what, const char *src,
                         const char *const *fragments)
{
    char *log = NULL;
    int rc = check_(src, &log);
    if (rc == 0) {
        fprintf(stderr, "FAIL %s: accepted\n---\n%s", what, src);
        exit(1);
    }
    for (int i = 0; fragments[i]; i++) {
        if (strstr(log, fragments[i])) continue;
        fprintf(stderr, "FAIL %s: diagnostic lacks \"%s\"\n---\n%s",
                what, fragments[i], log);
        exit(1);
    }
    g_arms++;
    printf("  refused: %s\n", what);
}

static void must_accept_(const char *what, const char *src)
{
    char *log = NULL;
    if (check_(src, &log) != 0) {
        fprintf(stderr, "FAIL %s: refused\n---\n%s", what, log);
        exit(1);
    }
    g_arms++;
    printf("  accepted: %s\n", what);
}

/* ---- The refusals --------------------------------------------------- */

static void gate_refusals_(void)
{
    static char src[16384];
    char radio[1024];

    radio_str_(radio, sizeof radio, RADIO_);

    /* `source=` naming nothing, a payload of the wrong kind, and a
     * payload of another body. Each names both statements, because a
     * reader has two places to look and the diagnostic says which. */
    dl_world_(src, sizeof src, 3.0e4, "6.0", "150.0", "10.0", radio,
              "kite", 1);
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "source=eye1");
        ASSERT(p != NULL);
        memcpy(p, "source=nope", 11);
        const char *f[] = { "astro_payload `pic1`", "`source=nope`",
                            "names no astro_payload", NULL };
        must_refuse_("`source=` naming no payload of this world", bad, f);
    }
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "source=eye1");
        ASSERT(p != NULL);
        memcpy(p, "source=pic2", 11);
        const char *f[] = { "astro_payload `pic1`", "`source=pic2`",
                            "which is of kind `infostate`",
                            "publishes no detection verdict", NULL };
        must_refuse_("`source=` naming a payload of a kind that decides "
                     "nothing", bad, f);
    }
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "source=eye1");
        ASSERT(p != NULL);
        memcpy(p, "source=eye2", 11);
        const char *f[] = { "astro_payload `pic1`", "`source=eye2`",
                            "carried by `drone_2`",
                            "carried by `drone_1`", NULL };
        must_refuse_("`source=` naming another craft's detection", bad, f);
    }

    /* A body carrying two datalinks, and a datalink whose carrier
     * holds no information state. */
    {
        static char bad[24576];
        char *p;
        int n = snprintf(bad, sizeof bad, "%s", src);
        (void)n;
        p = strstr(bad, "    episode\n");
        ASSERT(p != NULL);
        static char tail[8192];
        snprintf(tail, sizeof tail, "%s", p);
        snprintf(p, sizeof bad - (size_t)(p - bad),
            "    astro_payload link3 body=drone_1 kind=datalink"
            " network=kite rate_hz=2000.0 %s\n%s", radio, tail);
        const char *f[] = { "astro_payload `link3`",
                            "`drone_1` already carries the datalink "
                            "`link1`", "carries at most one", NULL };
        must_refuse_("a second datalink on one craft", bad, f);
    }
    {
        static char bad[24576];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "    astro_payload pic2 body=drone_2");
        ASSERT(p != NULL);
        /* Take drone_2's information state away and leave its
         * datalink, which then has nothing to broadcast. Its track
         * observe goes with it. */
        char *eol = strchr(p, '\n');
        ASSERT(eol != NULL);
        memmove(p, eol + 1, strlen(eol + 1) + 1);
        p = strstr(bad, "        observe track pic2 of bogey as trk2\n");
        ASSERT(p != NULL);
        memmove(p, p + strlen("        observe track pic2 of bogey"
                              " as trk2\n"),
                strlen(p + strlen("        observe track pic2 of bogey"
                                  " as trk2\n")) + 1);
        p = strstr(bad, "reward beta.trk2_valid");
        ASSERT(p != NULL);
        memcpy(p, "reward beta.see2_snr  ", 22);
        const char *f[] = { "astro_payload `link2`",
                            "`drone_2` carries no payload of kind "
                            "`infostate`",
                            "nothing to broadcast", NULL };
        must_refuse_("a datalink on a craft that keeps no information "
                     "state", bad, f);
    }

    /* A cadence that names no instant. */
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "rate_hz=2000.0");
        ASSERT(p != NULL);
        memcpy(p, "rate_hz=0000.0", 14);
        const char *f[] = { "astro_payload `link1`", "`rate_hz=0000.0`",
                            "names no broadcast instant", NULL };
        must_refuse_("a datalink cadence of nought", bad, f);
    }

    /* A key of the datalink written on another kind, and one of
     * another kind written on a datalink: each refused naming the kind
     * it belongs to rather than reported as unknown. */
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "history=1024 source=eye1");
        ASSERT(p != NULL);
        memcpy(p, "history=1024 network=kit", 24);
        const char *f[] = { "astro_payload `pic1`",
                            "`network=` belongs to kind `datalink`",
                            NULL };
        must_refuse_("`network=` on an information state", bad, f);
    }
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "network=kite rate_hz=2000.0");
        ASSERT(p != NULL);
        memcpy(p, "network=kite history=64444.", 27);
        const char *f[] = { "astro_payload `link1`",
                            "`history=` belongs to kind `infostate`",
                            NULL };
        must_refuse_("`history=` on a datalink", bad, f);
    }

    /* A datalink handed to a form that takes another kind of payload.
     * It publishes no channel of its own, which is the design's
     * position: what it changes is what a carrier's information state
     * knows, never whose channels an agent reads. */
    {
        static char bad[16384];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "observe track pic1 of bogey as trk1");
        ASSERT(p != NULL);
        memcpy(p, "observe track link1 of bogey as trk1", 35);
        const char *f[] = { "observe track link1 of bogey",
                            "is of kind `datalink`",
                            "takes a payload of kind `infostate`", NULL };
        must_refuse_("a datalink read as an information state", bad, f);
    }

    /* And the whole world as written, so the refusals above are shown
     * to be about what they name rather than about the fixture. */
    must_accept_("the two-member world the arms below are built on", src);
}

/* ---- 31d: the per-observer cap, in a world that carries a link ------- */

static void gate_cap_(void)
{
    static char src[262144];
    char radio[1024];
    int n;

    radio_str_(radio, sizeof radio, RADIO_);
    /* Shared and own knowledge count alike against the cap, and a
     * shared entry only lands for a target the receiver declares a
     * track over, so what the cap binds on is the declared count. The
     * refusal is exercised here in a world that carries a datalink,
     * which is where a reader would look for it. */
    for (int over = 1; over >= 0; over--) {
        int targets = 64 + over;
        n = snprintf(src, sizeof src, "%s", DL_HEAD);
        n += snprintf(src + n, sizeof src - (size_t)n,
            "    astro_body drone_2 assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=7.0e6 pos_y=3.0e4 vel_y=7546.0"
            " quat_w=1.0\n");
        for (int i = 0; i < targets; i++) {
            n += snprintf(src + n, sizeof src - (size_t)n,
                "    astro_body t%d assembly=\"calibration_box.k26asm\""
                " parent=earth pos_x=%d.0e6 vel_y=7000.0 quat_w=1.0\n",
                i, 7 + i % 3);
        }
        n += snprintf(src + n, sizeof src - (size_t)n,
            "%s%s"
            "    astro_payload pic1 body=drone_1 kind=infostate"
            " source=eye1\n"
            "    astro_payload pic2 body=drone_2 kind=infostate"
            " source=eye2\n"
            "    astro_payload link1 body=drone_1 kind=datalink"
            " network=kite rate_hz=1.0 %s\n"
            "    astro_payload link2 body=drone_2 kind=datalink"
            " network=kite rate_hz=1.0 %s\n"
            "%s"
            "    agent alpha\n"
            "        action nudge1 box -1.0 1.0 default 0.0\n",
            DL_EYE1("10.0"), DL_EYE("2", "drone_2", "10.0"), radio, radio,
            DL_EPISODE);
        for (int i = 0; i < targets; i++) {
            n += snprintf(src + n, sizeof src - (size_t)n,
                "        observe track pic1 of t%d as k%d\n", i, i);
        }
        n += snprintf(src + n, sizeof src - (size_t)n,
            "        objective\n            reward 1.0\n        end\n"
            "    end\nend\nend\n");
        ASSERT((size_t)n < sizeof src);
        if (over) {
            const char *f[] = { "astro_payload `pic1`",
                                "65 targets are tracked",
                                "per-observer limit is 64", NULL };
            must_refuse_("more tracks than the information state admits, "
                         "in a world carrying a datalink", src, f);
        } else {
            must_accept_("exactly as many tracks as it admits, in a "
                         "world carrying a datalink", src);
        }
    }
}

/* ---- 31a: the gate follows the verdict ------------------------------ */

static void gate_verdict_(void)
{
    static char src[16384];
    char radio[1024];
    DlRun run;
    int rises = 0, falls = 0, prev = -1;

    radio_str_(radio, sizeof radio, RADIO_);
    /* The bogey turns, and drone_1's threshold sits inside the band
     * its statistic sweeps, so the verdict changes several times in an
     * episode. */
    dl_world_(src, sizeof src, 3.0e4, "6.0", "150.0", "10.0", radio,
              "kite", 1);
    dl_run_(&run, src, "verdict", 24);

    for (int k = 0; k < run.n_steps; k++) {
        double det = dl_ch_(&run, k, "alpha.see1_detected");
        double val = dl_ch_(&run, k, "alpha.trk1_valid");
        if (det != val) {
            fprintf(stderr, "FAIL: step %d publishes verdict %.1f and "
                    "track validity %.1f\n", k + 1, det, val);
            exit(1);
        }
        if (prev >= 0 && det > 0.5 && prev == 0) rises++;
        if (prev >= 0 && det < 0.5 && prev == 1) falls++;
        prev = det > 0.5 ? 1 : 0;
    }
    if (rises < 1 || falls < 1) {
        fprintf(stderr, "FAIL: the verdict did not change in both "
                "directions (%d rises, %d falls)\n", rises, falls);
        exit(1);
    }
    g_arms++;
    printf("  the gated track follows its detection's verdict on all "
           "%d steps (%d rise(s), %d fall(s))\n", run.n_steps, rises,
           falls);

    /* And the shared half of the same statement: drone_2, which never
     * sees the bogey itself, holds a track exactly while its peer's
     * verdict holds. */
    for (int k = 0; k < run.n_steps; k++) {
        double det = dl_ch_(&run, k, "alpha.see1_detected");
        double own = dl_ch_(&run, k, "beta.see2_detected");
        double val = dl_ch_(&run, k, "beta.trk2_valid");
        ASSERT(own == 0.0);
        if (det != val) {
            fprintf(stderr, "FAIL: step %d, the peer's verdict is %.1f "
                    "and the receiver's track validity %.1f\n",
                    k + 1, det, val);
            exit(1);
        }
    }
    g_arms++;
    printf("  the receiving craft's track follows the transmitter's "
           "verdict, its own detection reading 0.0 throughout\n");
}

/* ---- 31a: the ungated form is unchanged ----------------------------- */

/* Two programs that differ only in the `source=` key, with a detection
 * that meets its threshold at every instant, record the same bytes. A
 * gate that withheld any push whose verdict met the threshold would
 * move them; so would a gate that changed the order or the epoch of a
 * push. */

static void gate_ungated_identity_(void)
{
    static char gated[16384], plain[16384];
    char radio[1024];
    char cmd[1024];

    radio_str_(radio, sizeof radio, RADIO_);
    /* No spin and a threshold far below the statistic, so the verdict
     * is 1.0 at every instant of the run. */
    dl_world_(gated, sizeof gated, 3.0e4, "0.0", "1.0", "1.0e-30",
              radio, "kite", 1);
    snprintf(plain, sizeof plain, "%s", gated);
    {
        char *p = strstr(plain, " source=eye1");
        ASSERT(p != NULL);
        memset(p, ' ', strlen(" source=eye1"));
        p = strstr(plain, " source=eye2");
        ASSERT(p != NULL);
        memset(p, ' ', strlen(" source=eye2"));
    }

    rl_write_file_(WORK_DIR "/gated.kfl", gated);
    rl_write_file_(WORK_DIR "/plain.kfl", plain);
    rl_compile_(WORK_DIR "/gated.kfl", WORK_DIR "/gated", WORK_DIR);
    rl_compile_(WORK_DIR "/plain.kfl", WORK_DIR "/plain", WORK_DIR);
    for (int i = 0; i < 2; i++) {
        snprintf(cmd, sizeof cmd,
                 WORK_DIR "/%s --seed 909 --envs 2 --episodes 2 --out "
                 WORK_DIR "/%s.k26ep > " WORK_DIR "/%s.log 2>&1",
                 i ? "plain" : "gated", i ? "plain" : "gated",
                 i ? "plain" : "gated");
        rl_run_or_die_(cmd);
    }
    if (!rl_files_equal_(WORK_DIR "/gated.k26ep",
                         WORK_DIR "/plain.k26ep")) {
        fprintf(stderr, "FAIL: a gate whose verdict holds at every "
                "instant changed what the run recorded\n");
        exit(1);
    }
    struct stat st;
    ASSERT(stat(WORK_DIR "/gated.k26ep", &st) == 0);
    g_arms++;
    printf("  a gate that never closes records what the ungated form "
           "records, byte for byte (%lld bytes)\n",
           (long long)st.st_size);
}

/* ---- 31a: the ungated form against the compiler before the key ------ */

/* The commit this work started from. A program that declares no
 * `source=` and no datalink must record what it recorded before either
 * word existed, and the only witness that can say so is the compiler
 * from before them. */
#define BASE_COMMIT "7d6cdf2"

static void dl_compile_with_(const char *kflc, const char *kfl_path,
                             const char *out_path)
{
    char cflags[4096];
    int n = snprintf(cflags, sizeof cflags,
        "-O2 -g -std=c++11 -Wno-format-truncation "
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
        "KFLC_CFLAGS=\"%s\" KFLC_LDLIBS=\"%s\" %s %s -o %s > "
        WORK_DIR "/base.log 2>&1", cflags, ldlibs, kflc, kfl_path,
        out_path);
    ASSERT((size_t)n < sizeof cmd);
    if (system(cmd) != 0) {
        (void)!system("cat " WORK_DIR "/base.log");
        fprintf(stderr, "FAIL: %s could not compile %s\n", kflc, kfl_path);
        exit(1);
    }
}

static void gate_base_identity_(void)
{
    static char src[16384];
    char cmd[2048];

    if (system("git -C .. rev-parse --verify --quiet " BASE_COMMIT
               "^{commit} > /dev/null 2>&1") != 0) {
        printf("  SKIP: the base commit " BASE_COMMIT " is not in this "
               "checkout's history, so the ungated form has no witness "
               "to be compared against\n");
        return;
    }
    rl_run_or_die_("rm -rf " WORK_DIR "/base && mkdir -p "
                   WORK_DIR "/base");
    rl_run_or_die_("git -C .. archive " BASE_COMMIT " kflc | tar -x -C "
                   WORK_DIR "/base");
    /* The archived tree carries the compiler alone, and its build
     * reads the sibling libraries' headers, so the siblings of this
     * checkout are put where it looks for them. */
    rl_run_or_die_("for d in ../lib* ../common; do ln -sfn \"$(cd $d && "
                   "pwd)\" " WORK_DIR "/base/; done");
    rl_run_or_die_("make -C " WORK_DIR "/base/kflc bin/kflc > "
                   WORK_DIR "/base/build.log 2>&1");

    /* A world with neither word in it. */
    dl_world_(src, sizeof src, 3.0e4, "6.0", "150.0", "10.0", NULL,
              "kite", 0);
    {
        char *p = strstr(src, " source=eye1");
        ASSERT(p != NULL);
        memset(p, ' ', strlen(" source=eye1"));
    }
    rl_write_file_(WORK_DIR "/base.kfl", src);
    dl_compile_with_(WORK_DIR "/base/kflc/bin/kflc", WORK_DIR "/base.kfl",
                     WORK_DIR "/basebin");
    dl_compile_with_("./bin/kflc", WORK_DIR "/base.kfl",
                     WORK_DIR "/nowbin");
    for (int i = 0; i < 2; i++) {
        snprintf(cmd, sizeof cmd,
                 WORK_DIR "/%s --seed 5150 --envs 2 --episodes 2 --out "
                 WORK_DIR "/%s.k26ep > " WORK_DIR "/%s.log 2>&1",
                 i ? "nowbin" : "basebin", i ? "nowbin" : "basebin",
                 i ? "nowbin" : "basebin");
        rl_run_or_die_(cmd);
    }
    if (!rl_files_equal_(WORK_DIR "/basebin.k26ep",
                         WORK_DIR "/nowbin.k26ep")) {
        fprintf(stderr, "FAIL: a program declaring neither `source=` nor "
                "a datalink records different bytes than it did before "
                "either existed\n");
        exit(1);
    }
    struct stat st;
    ASSERT(stat(WORK_DIR "/nowbin.k26ep", &st) == 0);
    g_arms++;
    printf("  the ungated form records what the compiler at " BASE_COMMIT
           " recorded, byte for byte (%lld bytes)\n",
           (long long)st.st_size);
}

/* ---- 31b: the value the link makes reachable ------------------------ */

static void gate_shared_track_(void)
{
    static char with_link[16384], without[16384];
    char radio[1024];
    DlRun a, b;

    radio_str_(radio, sizeof radio, RADIO_);
    dl_world_(with_link, sizeof with_link, 3.0e4, "0.0", "10.0", "10.0",
              radio, "kite", 1);
    dl_world_(without, sizeof without, 3.0e4, "0.0", "10.0", "10.0", NULL,
              "kite", 1);
    dl_run_(&a, with_link, "shared", 8);
    dl_run_(&b, without, "unshared", 8);

    for (int k = 0; k < a.n_steps; k++) {
        double own = dl_ch_(&a, k, "beta.see2_detected");
        double val = dl_ch_(&a, k, "beta.trk2_valid");
        double ctl = dl_ch_(&b, k, "beta.trk2_valid");
        ASSERT(own == 0.0);
        if (val != 1.0 || ctl != 0.0) {
            fprintf(stderr, "FAIL: step %d reads validity %.1f with the "
                    "link and %.1f without it\n", k + 1, val, ctl);
            exit(1);
        }
        /* The eight channels beside it zero-fill without the link,
         * which is the form an unavailable observation takes. */
        static const char *const REST[] = {
            "beta.trk2_pos_x", "beta.trk2_pos_y", "beta.trk2_pos_z",
            "beta.trk2_vel_x", "beta.trk2_vel_y", "beta.trk2_vel_z",
            "beta.trk2_range", "beta.trk2_age", NULL
        };
        for (int i = 0; REST[i]; i++) {
            ASSERT(dl_ch_(&b, k, REST[i]) == 0.0);
        }
    }
    g_arms++;
    printf("  a target its own detection has never seen reads valid "
           "with the link and unavailable without it, over %d steps\n",
           a.n_steps);

    /* The age against a light time computed here, and the shared
     * position against the transmitter's own published one. */
    for (int k = 1; k < a.n_steps; k++) {
        double range = dl_ch_(&a, k, "beta.trk2_range");
        double age   = dl_ch_(&a, k, "beta.trk2_age");
        double want  = range / K26A_C;
        double err   = fabs(age - want);
        if (err > 1.0e-12) {
            fprintf(stderr, "FAIL: step %d publishes age %.17g against "
                    "a light time of %.17g\n", k + 1, age, want);
            exit(1);
        }
        /* The peer's report and the transmitter's own picture are the
         * same sample of the same target, so the two published
         * positions agree to the metre at three thousand kilometres.
         * They are not identical: the two craft solve their own
         * retarded times from their own positions. */
        double dx = dl_ch_(&a, k, "beta.trk2_pos_x")
                  - dl_ch_(&a, k, "alpha.trk1_pos_x");
        double dy = dl_ch_(&a, k, "beta.trk2_pos_y")
                  - dl_ch_(&a, k, "alpha.trk1_pos_y");
        if (fabs(dx) > 1.0 || fabs(dy) > 1.0) {
            fprintf(stderr, "FAIL: step %d, the shared position differs "
                    "from the transmitter's by (%.6g, %.6g) m\n",
                    k + 1, dx, dy);
            exit(1);
        }
    }
    g_arms++;
    printf("  the shared age is the light time to the reported "
           "position, and the reported position is the transmitter's\n");

    /* The entry the receiver holds is older than the transmitter's own
     * by at least the link's light time, which is what says the report
     * travelled rather than appearing. The two ages are taken against
     * the same instant, so their difference is the queue and the
     * link. */
    {
        double lt_link = 3.0e4 / K26A_C;
        double age2 = dl_ch_(&a, a.n_steps - 1, "beta.trk2_age");
        double age1 = dl_ch_(&a, a.n_steps - 1, "alpha.trk1_age");
        /* Both ages are light times to the same target from craft
         * thirty kilometres apart, so they differ by geometry alone;
         * what the link costs shows in the sample the receiver holds,
         * which is one sub-advance and one link light time old. */
        printf("    transmitter age %.9g s, receiver age %.9g s, link "
               "light time %.9g s\n", age1, age2, lt_link);
    }
}

/* ---- 31c: the budget ------------------------------------------------ */

static void gate_budget_(void)
{
    static char src[16384];
    char radio[1024];
    DlRun run;
    double rstar = friis_range_(RADIO_);

    radio_str_(radio, sizeof radio, RADIO_);
    printf("    the declared keys put the closure range at %.6g m\n",
           rstar);

    /* A per cent either side of the range the hand figures give. The
     * monostatic relation is a fourth power in range with a cube of
     * four pi and a target cross-section in it, so an evaluator wired
     * in this one's place puts the boundary somewhere else entirely
     * and both halves of this arm fail. */
    for (int side = 0; side < 2; side++) {
        double sep = rstar * (side ? 1.01 : 0.99);
        double snr = friis_snr_(RADIO_, sep);
        dl_world_(src, sizeof src, sep, "0.0", "10.0", "10.0", radio,
                  "kite", 1);
        dl_run_(&run, src, side ? "beyond" : "inside", 6);
        double val = dl_ch_(&run, run.n_steps - 1, "beta.trk2_valid");
        double want = side ? 0.0 : 1.0;
        if (val != want) {
            fprintf(stderr, "FAIL: at %.6g m the budget gives %.6g "
                    "against a threshold of %.6g, and the receiver "
                    "reads validity %.1f\n", sep, snr, RADIO_[K_THR],
                    val);
            exit(1);
        }
        g_arms++;
        printf("  %s the closure range: %.6g m, budget %.6g against "
               "threshold %.6g, receiver reads %.1f\n",
               side ? "beyond" : "inside", sep, snr, RADIO_[K_THR], val);
    }

    /* One perturbation per declared key. The separation stands two per
     * cent inside the closure range, and each key is moved by the
     * factor that halves the budget or doubles the threshold, so a
     * budget that did not read that key would keep the closure it had.
     */
    {
        double sep = rstar * 0.98;
        for (int key = 0; key < K_COUNT; key++) {
            double v[K_COUNT];
            char perturbed[1024];

            memcpy(v, RADIO_, sizeof v);
            switch (key) {
            case K_P_TX: v[key] = RADIO_[key] / 2.0;   break;
            case K_G_TX:
            case K_G_RX: v[key] = RADIO_[key] - 3.0;   break;
            case K_FREQ: v[key] = RADIO_[key] * 2.0;   break;
            case K_LOSS: v[key] = RADIO_[key] + 3.0;   break;
            case K_BW:
            case K_T_SYS:
            case K_NF:   v[key] = RADIO_[key] * 2.0;   break;
            default:     v[key] = RADIO_[key] * 2.0;   break;
            }
            double snr = friis_snr_(v, sep);
            ASSERT(snr < v[K_THR]);
            radio_str_(perturbed, sizeof perturbed, v);
            dl_world_(src, sizeof src, sep, "0.0", "10.0", "10.0",
                      perturbed, "kite", 1);
            char stem[64];
            snprintf(stem, sizeof stem, "pert%d", key);
            dl_run_(&run, src, stem, 4);
            double val = dl_ch_(&run, run.n_steps - 1, "beta.trk2_valid");
            if (val != 0.0) {
                fprintf(stderr, "FAIL: %s perturbed to %.17g leaves the "
                        "budget at %.6g against threshold %.6g and the "
                        "receiver still reads valid\n",
                        RADIO_KEY_[key], v[key], snr, v[K_THR]);
                exit(1);
            }
            g_arms++;
            printf("  %s moves the budget: %.6g against %.6g, closure "
                   "lost\n", RADIO_KEY_[key], snr, v[K_THR]);
        }
        /* And the unperturbed world at the same separation closes, so
         * the nine above are about their keys and not about the
         * separation. */
        dl_world_(src, sizeof src, sep, "0.0", "10.0", "10.0", radio,
                  "kite", 1);
        dl_run_(&run, src, "unpert", 4);
        ASSERT(dl_ch_(&run, run.n_steps - 1, "beta.trk2_valid") == 1.0);
        g_arms++;
        printf("  the same world with every key as declared closes "
               "(budget %.6g)\n", friis_snr_(RADIO_, sep));
    }
}

/* One transmitter and two listening members, one inside the closure
 * range and one beyond it, in a single program. The pair of worlds
 * above says the boundary is where the hand figures put it; this says
 * it in one world, where the two members differ in nothing but their
 * distance from the transmitter. */

static void gate_two_receivers_(void)
{
    static char src[24576];
    char radio[1024];
    DlRun run;
    double rstar = friis_range_(RADIO_);
    double near_m = rstar * 0.99;
    double far_m  = rstar * 1.01;

    radio_str_(radio, sizeof radio, RADIO_);
    int n = snprintf(src, sizeof src, "%s", DL_HEAD);
    n += snprintf(src + n, sizeof src - (size_t)n,
        "    astro_body drone_2 assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=%.17g vel_y=7546.0"
        " quat_w=1.0\n"
        "    astro_body drone_3 assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=%.17g vel_y=7546.0"
        " quat_w=1.0\n"
        "    astro_body bogey assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=3.0e6 vel_y=7546.0"
        " quat_w=1.0\n"
        "%s%s%s"
        "    astro_payload pic1 body=drone_1 kind=infostate"
        " history=1024 source=eye1\n"
        "    astro_payload pic2 body=drone_2 kind=infostate"
        " history=1024 source=eye2\n"
        "    astro_payload pic3 body=drone_3 kind=infostate"
        " history=1024 source=eye3\n"
        "    astro_payload link1 body=drone_1 kind=datalink"
        " network=kite rate_hz=2000.0 %s\n"
        "    astro_payload link2 body=drone_2 kind=datalink"
        " network=kite rate_hz=2000.0 %s\n"
        "    astro_payload link3 body=drone_3 kind=datalink"
        " network=kite rate_hz=2000.0 %s\n"
        "%s"
        "    agent alpha\n"
        "        action nudge1 box -1.0 1.0 default 0.0\n"
        "        observe track pic1 of bogey as trk1\n"
        "        objective\n            reward 1.0\n        end\n"
        "    end\n"
        "    agent beta\n"
        "        action nudge2 box -1.0 1.0 default 0.0\n"
        "        observe track pic2 of bogey as trk2\n"
        "        objective\n            reward beta.trk2_valid\n"
        "        end\n"
        "    end\n"
        "    agent gamma\n"
        "        action nudge3 box -1.0 1.0 default 0.0\n"
        "        observe track pic3 of bogey as trk3\n"
        "        objective\n            reward gamma.trk3_valid\n"
        "        end\n"
        "    end\n"
        "end\nend\n",
        near_m, -far_m,
        DL_EYE1("10.0"), DL_EYE("2", "drone_2", "10.0"),
        DL_EYE("3", "drone_3", "10.0"),
        radio, radio, radio, DL_EPISODE);
    ASSERT((size_t)n < sizeof src);
    dl_run_(&run, src, "pair", 6);
    for (int k = 0; k < run.n_steps; k++) {
        double near_v = dl_ch_(&run, k, "beta.trk2_valid");
        double far_v  = dl_ch_(&run, k, "gamma.trk3_valid");
        if (near_v != 1.0 || far_v != 0.0) {
            fprintf(stderr, "FAIL: step %d, the near member reads %.1f "
                    "and the far member %.1f\n", k + 1, near_v, far_v);
            exit(1);
        }
    }
    g_arms++;
    printf("  in one world, the member at %.6g m receives and the "
           "member at %.6g m receives nothing (budgets %.6g and %.6g "
           "against %.6g)\n", near_m, far_m,
           friis_snr_(RADIO_, near_m), friis_snr_(RADIO_, far_m),
           RADIO_[K_THR]);

    /* The same world is the only one here that holds an entry offered
     * to more than one receiver, so it is where the order the design
     * fixes for receivers can be read at all: the transmitter's entry
     * for the bogey reaches both listening members, and it reaches
     * them in the order the program declares them. A loop that walked
     * the members the other way round would put the same two
     * transfers in the other order, and the drop-older rule would
     * then be settled by which loop the compiler happened to write. */
    {
        static char emitted[4194304];
        char cmd[1024];
        int rx[64], ent[64], edge0[64], nedge[64], first[8], count[8];

        snprintf(cmd, sizeof cmd,
                 "./bin/kflc --emit " WORK_DIR "/pair.kfl > " WORK_DIR
                 "/pair.cc 2>/dev/null");
        rl_run_or_die_(cmd);
        FILE *f = fopen(WORK_DIR "/pair.cc", "rb");
        ASSERT(f != NULL);
        size_t len = fread(emitted, 1, sizeof emitted - 1, f);
        emitted[len] = '\0';
        fclose(f);
        int nr = table_ints_(emitted, "kflrl_ledge_rx_[]", rx, 64);
        int nt = table_ints_(emitted, "kflrl_ledge_ent_[]", ent, 64);
        int n0 = table_ints_(emitted, "kflrl_lent_edge0_[]", edge0, 64);
        int nn = table_ints_(emitted, "kflrl_lent_nedge_[]", nedge, 64);
        int nf = table_ints_(emitted, "kflrl_link_ent0_[]", first, 8);
        int nc = table_ints_(emitted, "kflrl_link_nent_[]", count, 8);
        ASSERT(nr > 0 && nr == nt && n0 > 0 && n0 == nn);
        ASSERT(nf == 3 && nc == 3);
        /* The first datalink's entry for the bogey: its own state is
         * entry 0 and the bogey is entry 1. */
        int e_bogey = first[0] + 1;
        ASSERT(count[0] == 2);
        if (nedge[e_bogey] != 2) {
            fprintf(stderr, "FAIL: the transmitter's entry for the "
                    "bogey reaches %d receiver(s), not two\n",
                    nedge[e_bogey]);
            exit(1);
        }
        int k = edge0[e_bogey];
        if (!(rx[k] == 1 && rx[k + 1] == 2 && ent[k] == e_bogey &&
              ent[k + 1] == e_bogey)) {
            fprintf(stderr, "FAIL: the two transfers of one entry are "
                    "not in the program's own receiver order "
                    "(%d, %d)\n", rx[k], rx[k + 1]);
            exit(1);
        }
        g_arms++;
        printf("  one entry reaching two members transfers to them in "
               "declaration order (links %d then %d)\n", rx[k],
               rx[k + 1]);
    }
}

/* The link's own light time, measured rather than assumed.
 *
 * A shared entry is only usable while it is fresher than the light
 * from the target itself: the retarded time the observer solves for
 * has to fall inside the history it holds, and an entry newer than
 * that time is a report of light that has not arrived yet. So a peer
 * far enough away in light time cannot help, however strong its
 * signal, and that is what says the transfer pays a propagation delay
 * rather than landing where it was sent from.
 *
 * Two worlds differing in one number: the receiver's distance from
 * the transmitter. The target stands still in both, so its own light
 * time to the receiver barely moves, and the radio is strong enough
 * that both links close.
 */

static void gate_delay_(void)
{
    static char src[24576];
    char radio[1024];
    double v[K_COUNT];
    DlRun run;
    /* Twenty-decibel antennas, which put the closure range past four
     * thousand kilometres, so the far world below is a statement
     * about light time and not about the budget. */
    memcpy(v, RADIO_, sizeof v);
    v[K_G_TX] = 20.0;
    v[K_G_RX] = 20.0;
    radio_str_(radio, sizeof radio, v);

    for (int far = 0; far < 2; far++) {
        double sep = far ? 3.0e6 : 3.0e4;
        int n = snprintf(src, sizeof src, "%s", DL_HEAD);
        n += snprintf(src + n, sizeof src - (size_t)n,
            "    astro_body drone_2 assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=7.0e6 pos_z=%.17g vel_y=7546.0"
            " quat_w=1.0\n"
            "    astro_body bogey assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=7.0e6 pos_y=1.5e6 vel_y=7546.0"
            " quat_w=1.0\n"
            "%s%s"
            "    astro_payload pic1 body=drone_1 kind=infostate"
            " history=1024 source=eye1\n"
            "    astro_payload pic2 body=drone_2 kind=infostate"
            " history=1024 source=eye2\n"
            "    astro_payload link1 body=drone_1 kind=datalink"
            " network=kite rate_hz=2000.0 %s\n"
            "    astro_payload link2 body=drone_2 kind=datalink"
            " network=kite rate_hz=2000.0 %s\n"
            "%s"
            "    agent alpha\n"
            "        action nudge1 box -1.0 1.0 default 0.0\n"
            "        observe track pic1 of bogey as trk1\n"
            "        objective\n            reward 1.0\n        end\n"
            "    end\n"
            "    agent beta\n"
            "        action nudge2 box -1.0 1.0 default 0.0\n"
            "        observe track pic2 of bogey as trk2\n"
            "        objective\n            reward beta.trk2_valid\n"
            "        end\n"
            "    end\n"
            "end\nend\n",
            sep, DL_EYE1("10.0"), DL_EYE("2", "drone_2", "10.0"),
            radio, radio, DL_EPISODE);
        ASSERT((size_t)n < sizeof src);
        double link_lt = sep / K26A_C;
        ASSERT(friis_snr_(v, sep) > v[K_THR]);
        dl_run_(&run, src, far ? "delayfar" : "delaynear", 6);
        double val = dl_ch_(&run, run.n_steps - 1, "beta.trk2_valid");
        double want = far ? 0.0 : 1.0;
        if (val != want) {
            fprintf(stderr, "FAIL: with the peer %.6g m away (link "
                    "light time %.9g s, budget %.6g) the receiver "
                    "reads %.1f\n", sep, link_lt, friis_snr_(v, sep),
                    val);
            exit(1);
        }
        g_arms++;
        printf("  peer at %.6g m (link light time %.9g s, budget "
               "%.6g): receiver reads %.1f\n", sep, link_lt,
               friis_snr_(v, sep), val);
    }
    printf("    the far peer's signal closes and its report is still "
           "useless: it arrives older than the light from the target "
           "itself\n");
}

/* ---- 31d: what may not cross ---------------------------------------- */

static void gate_networks_(void)
{
    static char src[16384];
    char radio[1024];
    DlRun run;

    radio_str_(radio, sizeof radio, RADIO_);
    dl_world_(src, sizeof src, 3.0e4, "0.0", "10.0", "10.0", radio,
              "other", 1);
    dl_run_(&run, src, "twonets", 6);
    for (int k = 0; k < run.n_steps; k++) {
        if (dl_ch_(&run, k, "beta.trk2_valid") != 0.0) {
            fprintf(stderr, "FAIL: step %d, an entry crossed between "
                    "two communities\n", k + 1);
            exit(1);
        }
    }
    g_arms++;
    printf("  no entry crosses between two communities in one world, "
           "at a separation that closes on one\n");
}

/* An entry for a target the receiver declares no track over reaches no
 * transfer at all. It is read off the artifact's own transfer table,
 * because the consequence of not dropping it is a ring slot the
 * receiver never asked for, and a ring slot publishes no channel: the
 * table is where the rule is visible. */

static int table_ints_(const char *src, const char *name, int *out, int cap)
{
    const char *p = strstr(src, name);
    int n = 0;

    if (!p) return -1;
    p = strchr(p, '{');
    if (!p) return -1;
    p++;
    while (*p && *p != '}') {
        if ((*p >= '0' && *p <= '9') || *p == '-') {
            if (n >= cap) return -1;
            out[n++] = (int)strtol(p, (char **)&p, 10);
            continue;
        }
        p++;
    }
    return n;
}

static void gate_undeclared_target_(void)
{
    static char src[16384];
    static char emitted[4194304];
    char radio[1024];
    char cmd[1024];

    radio_str_(radio, sizeof radio, RADIO_);
    /* drone_1 tracks the bogey and a mule; drone_2 tracks the bogey
     * alone. The mule is inside the community's reach and is offered
     * by nobody, because the receiver holds no ring slot for it. */
    int n = snprintf(src, sizeof src, "%s", DL_HEAD);
    n += snprintf(src + n, sizeof src - (size_t)n,
        "    astro_body drone_2 assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=3.0e4 vel_y=7546.0"
        " quat_w=1.0\n"
        "    astro_body bogey assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=3.0e6 vel_y=7546.0"
        " quat_w=1.0\n"
        "    astro_body mule assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=2.0e6 vel_y=7546.0"
        " quat_w=1.0\n"
        "%s%s"
        "    astro_payload pic1 body=drone_1 kind=infostate"
        " history=1024 source=eye1\n"
        "    astro_payload pic2 body=drone_2 kind=infostate"
        " history=1024 source=eye2\n"
        "    astro_payload link1 body=drone_1 kind=datalink"
        " network=kite rate_hz=2000.0 %s\n"
        "    astro_payload link2 body=drone_2 kind=datalink"
        " network=kite rate_hz=2000.0 %s\n"
        "%s"
        "    agent alpha\n"
        "        action nudge1 box -1.0 1.0 default 0.0\n"
        "        observe track pic1 of bogey as trk1\n"
        "        observe track pic1 of mule as trkm\n"
        "        objective\n            reward 1.0\n        end\n"
        "    end\n"
        "    agent beta\n"
        "        action nudge2 box -1.0 1.0 default 0.0\n"
        "        observe track pic2 of bogey as trk2\n"
        "        objective\n            reward beta.trk2_valid\n"
        "        end\n"
        "    end\n"
        "end\nend\n",
        DL_EYE1("10.0"), DL_EYE("2", "drone_2", "10.0"), radio, radio,
        DL_EPISODE);
    ASSERT((size_t)n < sizeof src);
    rl_write_file_(WORK_DIR "/mule.kfl", src);
    snprintf(cmd, sizeof cmd,
             "./bin/kflc --emit " WORK_DIR "/mule.kfl > " WORK_DIR
             "/mule.cc 2> " WORK_DIR "/mule.log");
    rl_run_or_die_(cmd);
    FILE *f = fopen(WORK_DIR "/mule.cc", "rb");
    ASSERT(f != NULL);
    size_t len = fread(emitted, 1, sizeof emitted - 1, f);
    emitted[len] = '\0';
    fclose(f);

    int veh[64], self[64], nedge[64], first[8], count[8];
    int nv = table_ints_(emitted, "kflrl_lent_veh_[]", veh, 64);
    int ns = table_ints_(emitted, "kflrl_lent_self_[]", self, 64);
    int ne = table_ints_(emitted, "kflrl_lent_nedge_[]", nedge, 64);
    int nf = table_ints_(emitted, "kflrl_link_ent0_[]", first, 8);
    int nc = table_ints_(emitted, "kflrl_link_nent_[]", count, 8);
    ASSERT(nv > 0 && nv == ns && nv == ne);
    ASSERT(nf == 2 && nc == 2);

    /* Vehicle slots are the assembly-bearing bodies in declaration
     * order: drone_1, drone_2, bogey, mule. The transmitter is the
     * first datalink, and its entries are its own state, the bogey and
     * the mule, in that order. */
    int base = first[0];
    ASSERT(count[0] == 3);
    ASSERT(self[base] == 1 && veh[base] == 0);
    ASSERT(self[base + 1] == 0 && veh[base + 1] == 2);
    ASSERT(self[base + 2] == 0 && veh[base + 2] == 3);
    if (nedge[base + 1] != 1 || nedge[base + 2] != 0) {
        fprintf(stderr, "FAIL: the bogey's entry has %d transfer(s) and "
                "the mule's %d; the mule is tracked by nobody that "
                "receives\n", nedge[base + 1], nedge[base + 2]);
        exit(1);
    }
    /* The transmitter's own state is offered to nobody either, because
     * no member of the community declares a track over it. */
    ASSERT(nedge[base] == 0);
    g_arms++;
    printf("  an entry for a target the receiver declares no track over "
           "reaches no transfer (bogey %d, mule %d, own state %d)\n",
           nedge[base + 1], nedge[base + 2], nedge[base]);
}

/* The one limit this surface reports rather than refuses.
 *
 * A cadence high enough that more than the depth of one edge's queue
 * is in flight at once cannot be refused where the program is
 * written: the depth a cadence needs is its rate times a light time
 * the run decides. So the environment faults on it, rather than
 * dropping a transmission and letting the link claim a reach it has
 * not got. The arm reaches it, since a branch nothing can take is a
 * branch nothing measures, and shows the same world inside the limit
 * running clean.
 *
 * At a sub-advance of two and a half milliseconds an offer to a peer
 * thirty milliseconds away is in flight for twelve boundaries, which
 * is past the depth; one to a peer at three hundred kilometres is in
 * flight for one.
 */

static void gate_inflight_(void)
{
    static char src[24576];
    char radio[1024];
    double v[K_COUNT];

    memcpy(v, RADIO_, sizeof v);
    v[K_G_TX] = 25.0;
    v[K_G_RX] = 25.0;
    radio_str_(radio, sizeof radio, v);

    for (int over = 1; over >= 0; over--) {
        double sep = over ? 9.0e6 : 3.0e5;
        char path[512], out[512], so[512];
        const char *stem = over ? "flood" : "trickle";

        ASSERT(friis_snr_(v, sep) > v[K_THR]);
        dl_world_(src, sizeof src, sep, "0.0", "10.0", "10.0", radio,
                  "kite", 1);
        snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
        snprintf(out, sizeof out, WORK_DIR "/%s", stem);
        snprintf(so, sizeof so, WORK_DIR "/%s.rlenv.so", stem);
        rl_write_file_(path, src);
        rl_compile_(path, out, WORK_DIR);
        void *h = rl_dlopen_(so);
        RlSurface s;
        rl_resolve_surface_(h, &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(11u, 1u, &env) == K26RL_OK);
        double act[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        uint16_t fault = 0;
        int faulted = 0;
        for (int k = 0; k < 6; k++) {
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.fault_codes(env, &fault) == K26RL_OK);
            if (fault != 0) faulted = 1;
        }
        s.destroy(env);
        dlclose(h);
        if (faulted != over) {
            fprintf(stderr, "FAIL: with the peer %.6g m away (%.6g "
                    "boundaries of light time) the environment %s\n",
                    sep, sep / K26A_C / 0.0025,
                    faulted ? "faulted" : "did not fault");
            exit(1);
        }
        g_arms++;
        printf("  peer at %.6g m, %.3g boundaries of light time: %s\n",
               sep, sep / K26A_C / 0.0025,
               faulted ? "the environment faults on the offers in "
                         "flight, as it must"
                       : "the run is clean");
    }
}

/* ---- 37a: one generator, over this surface's own entry points ------- */

/* The single-generator rule is this capability's, and it extends over
 * every entry point added to the stepping path. The transfer pass and
 * the gate are both deterministic functions of the world's state and
 * the declared parameters, so an artifact carrying a datalink and a
 * gated information state must name the tier's generator nowhere at
 * all, and the compiled unit must leave no reference to it.
 *
 * The perturbation beside each arm is what makes it a check rather
 * than a hope: a check that cannot fail on the defect it names
 * measures nothing. */

static void gate_one_generator_(void)
{
    static char src[16384];
    static char emitted[4194304];
    char radio[1024], cmd[1024];

    radio_str_(radio, sizeof radio, RADIO_);
    dl_world_(src, sizeof src, 3.0e4, "6.0", "150.0", "10.0", radio,
              "kite", 1);
    rl_write_file_(WORK_DIR "/gen.kfl", src);
    rl_run_or_die_("./bin/kflc --emit " WORK_DIR "/gen.kfl > "
                   WORK_DIR "/gen.cc 2> " WORK_DIR "/gen.err");
    FILE *f = fopen(WORK_DIR "/gen.cc", "rb");
    ASSERT(f != NULL);
    size_t len = fread(emitted, 1, sizeof emitted - 1, f);
    emitted[len] = '\0';
    fclose(f);
    if (strstr(emitted, "K26CRng") || strstr(emitted, "k26c_rng_")) {
        fprintf(stderr, "FAIL: an artifact carrying a datalink and a "
                "gated information state names the tier's generator\n");
        exit(1);
    }
    g_arms++;
    printf("  an artifact with a datalink and a gated information "
           "state names neither K26CRng nor k26c_rng_ (%zu bytes "
           "scanned)\n", len);

    /* The perturbation: a generator planted in the same emitted source
     * is caught by the same check. */
    rl_run_or_die_("sed 's/^static void kflrl_link_pass_/"
                   "static K26CRng _kfl_probe;\\nstatic void "
                   "kflrl_link_pass_/' " WORK_DIR "/gen.cc > "
                   WORK_DIR "/genp.cc");
    f = fopen(WORK_DIR "/genp.cc", "rb");
    ASSERT(f != NULL);
    len = fread(emitted, 1, sizeof emitted - 1, f);
    emitted[len] = '\0';
    fclose(f);
    if (!strstr(emitted, "K26CRng")) {
        fprintf(stderr, "FAIL: a generator planted beside the transfer "
                "pass was not caught\n");
        exit(1);
    }
    g_arms++;
    printf("  perturbation: a generator planted beside the transfer "
           "pass is caught\n");

    /* And the compiled unit, which is what a direct call would show as
     * an undefined reference. */
    snprintf(cmd, sizeof cmd,
             "grep -c 'k26c_rng_' " WORK_DIR "/gen.cc > /dev/null");
    ASSERT(system(cmd) != 0);
    g_arms++;
    printf("  the emitted source calls no entry point of the tier's "
           "generator\n");
}

/* ---- 37a: determinism, and the order the design fixes --------------- */

static void gate_determinism_(void)
{
    static char src[16384], swapped[16384];
    static char emit_a[4194304];
    static char emit_b[4194304];
    char radio[1024], cmd[1024];

    radio_str_(radio, sizeof radio, RADIO_);
    dl_world_(src, sizeof src, 3.0e4, "6.0", "150.0", "10.0", radio,
              "kite", 1);
    rl_write_file_(WORK_DIR "/det.kfl", src);
    rl_compile_(WORK_DIR "/det.kfl", WORK_DIR "/det", WORK_DIR);
    for (int i = 0; i < 2; i++) {
        snprintf(cmd, sizeof cmd,
                 WORK_DIR "/det --seed 4242 --envs 2 --episodes 2 --out "
                 WORK_DIR "/run%d.k26ep > " WORK_DIR "/run%d.log 2>&1",
                 i, i);
        rl_run_or_die_(cmd);
    }
    if (!rl_files_equal_(WORK_DIR "/run0.k26ep", WORK_DIR "/run1.k26ep")) {
        fprintf(stderr, "FAIL: two processes at one seed wrote different "
                "episode files\n");
        exit(1);
    }
    struct stat st;
    ASSERT(stat(WORK_DIR "/run0.k26ep", &st) == 0);
    g_arms++;
    printf("  two processes at one seed produce identical episode files "
           "for a world with a datalink (%lld bytes)\n",
           (long long)st.st_size);

    /* The same world with the two datalinks declared the other way
     * round. The transfer table is the program's own order, so it
     * changes; nothing the run records does. */
    {
        static char line1[2048], line2[2048];
        const char *a = strstr(src, "    astro_payload link1 ");
        ASSERT(a != NULL);
        const char *b = strchr(a, '\n');
        ASSERT(b != NULL);
        snprintf(line1, sizeof line1, "%.*s\n", (int)(b - a), a);
        const char *c = b + 1;
        const char *d = strchr(c, '\n');
        ASSERT(d != NULL);
        snprintf(line2, sizeof line2, "%.*s\n", (int)(d - c), c);
        int n = snprintf(swapped, sizeof swapped, "%.*s%s%s%s",
                         (int)(a - src), src, line2, line1, d + 1);
        ASSERT((size_t)n < sizeof swapped);
    }
    rl_write_file_(WORK_DIR "/swap.kfl", swapped);
    rl_compile_(WORK_DIR "/swap.kfl", WORK_DIR "/swap", WORK_DIR);
    snprintf(cmd, sizeof cmd,
             WORK_DIR "/swap --seed 4242 --envs 2 --episodes 2 --out "
             WORK_DIR "/run2.k26ep > " WORK_DIR "/run2.log 2>&1");
    rl_run_or_die_(cmd);
    if (!rl_files_equal_(WORK_DIR "/run0.k26ep", WORK_DIR "/run2.k26ep")) {
        fprintf(stderr, "FAIL: permuting the payload declaration order "
                "changed what the run recorded\n");
        exit(1);
    }
    g_arms++;
    printf("  the payload declaration order permuted records the same "
           "bytes\n");

    /* And the table itself, which is what the order rule is about. The
     * transmitters appear in the order the program declares them, so
     * the craft each table row speaks for are the two programs' orders
     * and are each other's reverse. The payload slots themselves
     * cannot show it: a slot is the declaration order too, so the
     * first datalink declared holds the lower slot in either
     * program. */
    {
        snprintf(cmd, sizeof cmd,
                 "./bin/kflc --emit " WORK_DIR "/det.kfl > " WORK_DIR
                 "/det.cc 2>/dev/null");
        rl_run_or_die_(cmd);
        snprintf(cmd, sizeof cmd,
                 "./bin/kflc --emit " WORK_DIR "/swap.kfl > " WORK_DIR
                 "/swap.cc 2>/dev/null");
        rl_run_or_die_(cmd);
        FILE *f = fopen(WORK_DIR "/det.cc", "rb");
        ASSERT(f != NULL);
        size_t la = fread(emit_a, 1, sizeof emit_a - 1, f);
        emit_a[la] = '\0';
        fclose(f);
        f = fopen(WORK_DIR "/swap.cc", "rb");
        ASSERT(f != NULL);
        size_t lb = fread(emit_b, 1, sizeof emit_b - 1, f);
        emit_b[lb] = '\0';
        fclose(f);
        int pa[8], pb[8];
        int na = table_ints_(emit_a, "kflrl_link_veh_[]", pa, 8);
        int nb = table_ints_(emit_b, "kflrl_link_veh_[]", pb, 8);
        ASSERT(na == 2 && nb == 2);
        if (!(pa[0] == 0 && pa[1] == 1 && pb[0] == 1 && pb[1] == 0)) {
            fprintf(stderr, "FAIL: the transfer table is not the "
                    "program's own declaration order (%d,%d against "
                    "%d,%d)\n", pa[0], pa[1], pb[0], pb[1]);
            exit(1);
        }
        g_arms++;
        printf("  the transfer table is the declaration order: the "
               "transmitters are craft %d,%d against %d,%d\n", pa[0],
               pa[1], pb[0], pb[1]);
    }
}

int main(void)
{
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_run_or_die_("cp examples/assets/calibration_box.k26asm "
                   "examples/assets/calibration_box.k26mesh "
                   WORK_DIR "/");

    printf("test_rl_datalink: the gated information state and the "
           "datalink\n");
    gate_refusals_();
    gate_cap_();

    if (!rl_libs_present_("test_rl_datalink")) {
        printf("test_rl_datalink: %d arm(s) passed, drive arms stood "
               "down (stack archives absent)\n", g_arms);
        return 77;
    }
    gate_verdict_();
    gate_ungated_identity_();
    gate_base_identity_();
    gate_shared_track_();
    gate_budget_();
    gate_two_receivers_();
    gate_delay_();
    gate_networks_();
    gate_undeclared_target_();
    gate_inflight_();
    gate_one_generator_();
    gate_determinism_();

    printf("test_rl_datalink: %d arm(s) passed\n", g_arms);
    return 0;
}
