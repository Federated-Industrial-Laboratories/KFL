/* test_rl_track_held.c: how a gated information state answers.
 *
 * An information state with no `source=` is pushed the target's true
 * state every sub-advance, so its history is dense and complete and
 * the only thing between the observer and it is the light time. That
 * form answers at the retarded time and has always done so.
 *
 * A `source=` gates the push on a declared detection's verdict, and a
 * datalink adds entries a peer made and sent. Such a history is not
 * dense: entries arrive when the target was seen and when a report
 * landed, and nothing fills the gaps between them. At the ranges craft
 * work at, the light time is microseconds while the gap between
 * entries is a control period, so a retarded-time answer over such a
 * history reads unavailable almost always: the instant it has to read
 * the history at falls past the newest entry there is. A gated state
 * therefore answers with the newest entry at or before the instant
 * asked about, and publishes the age of that entry. Nothing is
 * extrapolated; the position published is one the target genuinely
 * occupied, and the age says how long ago.
 *
 * The fixture is a swarm at working range: two craft fifty metres
 * apart in low orbit, a fragment a kilometre off, a radio that closes
 * between the craft at eighty-four kilometres, and a control period of
 * half a second cut into sub-advances of fifty milliseconds. The light
 * time to the fragment is three microseconds. That asymmetry, four
 * orders of magnitude between the light time and the interval between
 * entries, is the whole subject.
 *
 * Gates:
 *
 *   1. An own-blind receiver holds an aged track through its peer. The
 *      wing craft's own radar never meets its threshold against the
 *      fragment in the whole run; the lead's always does; the radio
 *      closes. The same fixture is compiled by the compiler at the
 *      commit this work started from and driven with the same actions,
 *      and the two are counted against each other: the older artifact
 *      holds a valid track on no step at all, and today's holds one on
 *      every step, its age being the instant asked about less the
 *      instant the peer's entry was made, which the arm checks against
 *      the declared broadcast cadence.
 *
 *   2. The age grows through a gap and drops on the entry that ends
 *      it. The lead is driven away from the fragment until its own
 *      radar loses it for several control periods and then brought
 *      back. Through the gap the age has to grow by exactly one
 *      control period a step, and on the step the verdict returns it
 *      has to read nought.
 *
 *   3. The ungated form is untouched. The same world with no `source=`
 *      records what the compiler at the same commit recorded, byte for
 *      byte.
 *
 *   4. Two processes at one seed record identical episode files for
 *      the gated world with its datalink.
 *
 *   5. The rule this one replaces is shown absent. The old rule read
 *      unavailable once the newest entry was older than the light time
 *      to the target; the arm counts the steps standing in exactly
 *      that condition, requires there to be some, and requires each of
 *      them to publish a valid track with an honest age.
 *
 * Pattern: rl_gate_util.h. Needs the sibling stack archives; skips
 * with 77 when they are absent. The arms that compare against an
 * archived compiler fail rather than skip when the commit is not in
 * this checkout's history, on that helper's own rule, unless
 * KFLRL_ALLOW_NO_PRIOR stands them down aloud.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "rl_gate_util.h"

/* The speed of light as the stack's own header carries it, so the
 * light times this gate computes and the ones the artifact computes
 * rest on one number. */
#include "k26astro_core/consts.h"

#define WORK_DIR "/tmp/kflc_rl_track_held_test"

/* The commit this work started from: the witness for what a gated
 * state answered before this rule, and for the ungated form being
 * unmoved by it. */
#define BASE_COMMIT "327448c"

static int g_arms;

/* ---- The fixture ---------------------------------------------------- */

/* The clock, and the two figures every arm does arithmetic with. They
 * are stringified from the same macros the episode block is built out
 * of, so a world and the numbers an arm holds it against cannot say
 * different things about it. */
#define TH_CONTROL_DT 0.5
#define TH_SUBSTEPS   10
#define TH_STR__(x)   #x
#define TH_STR_(x)    TH_STR__(x)
#define TH_SUB_S      (TH_CONTROL_DT / (double)TH_SUBSTEPS)

/* The broadcast cadence, one a second, which is the Kite worlds' own
 * and is slow against the control period on purpose: a report that
 * arrived every step would make the age hard to tell from nought. */
#define TH_RATE_HZ    1.0

/* The geometry. The two craft sit fifty metres apart and the fragment
 * a kilometre off the lead, along a third axis so that driving the
 * lead sideways lengthens the range to it. */
#define TH_WING_SEP_M 50.0
#define TH_FRAG_M     1000.0

/* The radios. Both craft carry the same radar; the wing's transmits a
 * millionth of the lead's power, which is what makes it blind to the
 * fragment at this range rather than a threshold chosen to make it so.
 * The link keys are the Kite worlds' own. */
#define TH_LEAD_TX "0.5"
#define TH_WING_TX "5.0e-7"
#define TH_RADAR_THRESHOLD "20.0"

#define TH_LINK \
    " network=kite_ops rate_hz=" TH_STR_(TH_RATE_HZ) \
    " p_tx_w=2.0 g_tx_db=3.0 g_rx_db=3.0 freq_hz=2.2e9" \
    " loss_sys_db=2.0 bandwidth_hz=1.0e6 t_sys_k=500.0" \
    " noise_figure=2.0 snr_threshold=6.0\n"

/* `gated` puts a `source=` on both information states; `link` declares
 * the two datalinks. The gravitating primary is declared first, as
 * every world in this tree declares it. */
static void th_world_(char *out, size_t cap, int gated, int link)
{
    int n = snprintf(out, cap,
        "form HELD\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body lead assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=6921000.0 pos_y=0.0 pos_z=0.0"
        " vel_x=0.0 vel_y=7588.998434594857 vel_z=0.0 quat_w=1.0\n"
        "    astro_body wing assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=6921000.0 pos_y=%.17g pos_z=0.0"
        " vel_x=0.0 vel_y=7588.998434594857 vel_z=0.0 quat_w=1.0\n"
        "    astro_body frag assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=6921000.0 pos_y=0.0 pos_z=%.17g"
        " vel_x=0.0 vel_y=7588.998434594857 vel_z=0.0 quat_w=1.0\n"
        "    astro_payload lead_eye body=lead kind=detect_radar"
        " p_tx_w=" TH_LEAD_TX " g_tx_db=16.0 g_rx_db=16.0"
        " freq_hz=1.7e10 loss_sys_db=3.0 bandwidth_hz=2.0e7"
        " t_sys_k=500.0 noise_figure=3.0"
        " snr_threshold=" TH_RADAR_THRESHOLD "\n"
        "    astro_payload wing_eye body=wing kind=detect_radar"
        " p_tx_w=" TH_WING_TX " g_tx_db=16.0 g_rx_db=16.0"
        " freq_hz=1.7e10 loss_sys_db=3.0 bandwidth_hz=2.0e7"
        " t_sys_k=500.0 noise_figure=3.0"
        " snr_threshold=" TH_RADAR_THRESHOLD "\n"
        "    astro_payload lead_pic body=lead kind=infostate"
        " history=64%s\n"
        "    astro_payload wing_pic body=wing kind=infostate"
        " history=64%s\n"
        "%s%s"
        "    episode\n"
        "        control_dt " TH_STR_(TH_CONTROL_DT) "\n"
        "        substeps " TH_STR_(TH_SUBSTEPS) "\n"
        "        horizon 60\n"
        "    end\n"
        "    agent lead_pilot\n"
        "        action lead_push box -400.0 400.0 default 0.0\n"
        "        observe detect lead_eye of frag as look\n"
        "        observe track lead_pic of frag as pic\n"
        "        objective\n"
        "            reward lead_pilot.look_snr\n"
        "        end\n"
        "    end\n"
        "    agent wing_pilot\n"
        "        action wing_push box -1.0 1.0 default 0.0\n"
        "        observe detect wing_eye of frag as look\n"
        "        observe track wing_pic of frag as pic\n"
        "        objective\n"
        "            reward wing_pilot.pic_valid\n"
        "        end\n"
        "    end\n"
        /* The lead's drift across the fragment's line of sight is set
         * outright rather than added to, so a step's range is this
         * gate's own arithmetic and not an accumulation it would have
         * to track. */
        "    on_step\n"
        "        lead.vel_x = lead_push\n"
        "        wing.vel_x = wing.vel_x + wing_push\n"
        "    end\n"
        "end\n"
        "end\n",
        TH_WING_SEP_M, TH_FRAG_M,
        gated ? " source=lead_eye" : "",
        gated ? " source=wing_eye" : "",
        link ? "    astro_payload lead_link body=lead kind=datalink"
               TH_LINK : "",
        link ? "    astro_payload wing_link body=wing kind=datalink"
               TH_LINK : "");
    ASSERT((size_t)n < cap);
}

/* ---- Driving -------------------------------------------------------- */

#define TH_MAX_STEPS 64

typedef struct {
    double   v[TH_MAX_STEPS][64];
    int      n_steps;
    uint8_t  blob[8192];
    uint32_t blob_len;
} ThRun;

/* Drive one already-built artifact through `steps` steps, writing the
 * lead's drift from `push` and leaving every other action at nought. */
static void th_drive_(ThRun *r, const char *so_path, const double *push,
                      int steps)
{
    void *h = rl_dlopen_(so_path);
    RlSurface s;
    K26RlEnv *env = NULL;
    uint16_t fault = 0;

    ASSERT(steps <= TH_MAX_STEPS);
    rl_resolve_surface_(h, &s);
    ASSERT(s.create(17u, 1u, &env) == K26RL_OK);
    {
        int32_t n = s.spec(env, r->blob, sizeof r->blob);
        ASSERT(n > 0);
        r->blob_len = (uint32_t)n;
    }
    for (int k = 0; k < steps; k++) {
        double act[8] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        act[0] = push ? push[k] : 0.0;
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, r->v[k]) == K26RL_OK);
        ASSERT(s.fault_codes(env, &fault) == K26RL_OK);
        if (fault != 0) {
            fprintf(stderr, "FAIL: %s faulted at step %d (%u)\n",
                    so_path, k + 1, (unsigned)fault);
            exit(1);
        }
    }
    r->n_steps = steps;
    s.destroy(env);
    dlclose(h);
}

static double th_ch_(const ThRun *r, int step, const char *name)
{
    int i = find_channel_(r->blob, r->blob_len, name);
    if (i < 0) {
        fprintf(stderr, "FAIL: no channel `%s` in this artifact\n", name);
        exit(1);
    }
    return r->v[step][i];
}

/* Compile a source through today's compiler and hand back the shared
 * object's path in `so`. */
static void th_build_(const char *src, const char *stem, char *so,
                      size_t cap)
{
    char path[512], out[512];

    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    snprintf(so, cap, WORK_DIR "/%s.rlenv.so", stem);
    rl_write_file_(path, src);
    rl_compile_(path, out, WORK_DIR);
}

/* ---- Gates 1 and 5: the receiver, and the rule this replaces -------- */

/* Whether the entry a held answer published was made at one of the
 * declared broadcast instants. Each cadence instant takes effect at
 * the first sub-advance boundary at or after it, so a broadcast leaves
 * somewhere in the sub-advance beginning at a whole multiple of the
 * cadence period; the transmitter's own verdict holds at every
 * boundary in this fixture, so the entry it offers was made at the
 * boundary it broadcast on.
 *
 * A tenth of a nanosecond is the allowance: the environment reaches a
 * boundary by accumulating sub-advances and this reaches it by
 * multiplying, and the quantities being told apart are seconds. */
static int th_on_cadence_(double made_at)
{
    const double period = 1.0 / TH_RATE_HZ;
    double n = floor(made_at / period + 1.0e-9);
    double at = n * period;

    return n >= 0.0 && made_at >= at - 1.0e-10
        && made_at <= at + TH_SUB_S + 1.0e-10;
}

static void gate_receiver_(int have_base)
{
    static char src[16384];
    char so_now[512];
    ThRun now, base;
    int held = 0, past_light = 0, base_valid = 0;
    /* The oldest a delivered report can be: one cadence period waiting
     * for the next broadcast, one sub-advance for the boundary that
     * broadcast leaves on, one for the boundary its arrival is taken
     * on, and the light time between the craft. */
    const double oldest = 1.0 / TH_RATE_HZ + 2.0 * TH_SUB_S
                        + TH_WING_SEP_M / K26A_C;

    th_world_(src, sizeof src, 1, 1);
    th_build_(src, "linked", so_now, sizeof so_now);
    th_drive_(&now, so_now, NULL, 24);

    for (int k = 0; k < now.n_steps; k++) {
        double own  = th_ch_(&now, k, "wing_pilot.look_detected");
        double val  = th_ch_(&now, k, "wing_pilot.pic_valid");
        double age  = th_ch_(&now, k, "wing_pilot.pic_age");
        double rng  = th_ch_(&now, k, "wing_pilot.pic_range");
        double elapsed = (double)(k + 1) * TH_CONTROL_DT;
        double made_at = elapsed - age;

        /* The receiver is blind on its own account, or the arm is
         * about a craft that could see the fragment anyway. */
        if (own != 0.0) {
            fprintf(stderr, "FAIL: step %d, the receiving craft's own "
                    "radar met its threshold, so this fixture does not "
                    "hold the statement the arm makes\n", k + 1);
            exit(1);
        }
        if (val != 1.0) {
            fprintf(stderr, "FAIL: step %d, the receiver publishes "
                    "validity %.1f\n", k + 1, val);
            exit(1);
        }
        if (age < 0.0 || age > elapsed + 1.0e-10) {
            fprintf(stderr, "FAIL: step %d publishes an age of %.17g "
                    "against an elapsed episode of %.17g\n", k + 1, age,
                    elapsed);
            exit(1);
        }
        /* An age equal to the elapsed episode is the epoch seed, which
         * every ring holds; anything younger came over the link, and
         * it has to have been made at a declared broadcast instant and
         * within the delivery the cadence and the distance cost. */
        if (fabs(age - elapsed) <= 1.0e-10) continue;
        if (age > oldest || !th_on_cadence_(made_at)) {
            fprintf(stderr, "FAIL: step %d publishes an age of %.17g, "
                    "putting the entry at %.17g s, which is neither a "
                    "broadcast instant nor inside the %.9g s a delivery "
                    "costs\n", k + 1, age, made_at, oldest);
            exit(1);
        }
        held++;
        /* Gate 5's condition, computed from the artifact's own
         * published range: the entry is older than the light from the
         * target, which is where the rule this replaces read
         * unavailable. */
        if (age > rng / K26A_C) past_light++;
    }
    if (held < 1) {
        fprintf(stderr, "FAIL: the receiver never held anything but its "
                "epoch seed, so nothing crossed the link and the arm "
                "has measured nothing\n");
        exit(1);
    }
    if (past_light != held) {
        fprintf(stderr, "FAIL: %d of the %d peer reports stand past the "
                "light time to the reported position; the fixture was "
                "built so that all of them would, and an arm about the "
                "old rule needs them to\n", past_light, held);
        exit(1);
    }
    g_arms++;
    printf("  a receiver blind on its own account holds its peer's "
           "report on %d of %d steps, each made at a declared broadcast "
           "instant and at most %.6g s old\n", held, now.n_steps,
           oldest);
    g_arms++;
    printf("  every one of those %d reports is older than the light "
           "time to the position it reports (%.3g s at a kilometre), "
           "which is the condition the rule this replaces read "
           "unavailable on\n", past_light, TH_FRAG_M / K26A_C);

    if (!have_base) {
        printf("  NOT MEASURED: the count under the compiler at "
               BASE_COMMIT " was not taken, its arm having been stood "
               "down above\n");
        return;
    }

    /* The same fixture through the compiler at the commit this work
     * started from, driven identically. What that artifact counted is
     * the measurement the design was changed on. */
    rl_base_compile_(WORK_DIR, WORK_DIR "/linked.kfl",
                     WORK_DIR "/linkedbase", 0);
    th_drive_(&base, WORK_DIR "/linkedbase.rlenv.so", NULL, 24);
    for (int k = 0; k < base.n_steps; k++) {
        if (th_ch_(&base, k, "wing_pilot.pic_valid") != 0.0) base_valid++;
    }
    if (base_valid != 0) {
        fprintf(stderr, "FAIL: the artifact at " BASE_COMMIT " held a "
                "valid track on %d step(s), so the count this change "
                "was made on is not what this fixture reproduces and "
                "the comparison says nothing\n", base_valid);
        exit(1);
    }
    g_arms++;
    printf("  the same fixture at " BASE_COMMIT " held a valid track on "
           "%d of %d steps against %d today\n", base_valid,
           base.n_steps, now.n_steps);
}

/* ---- Gate 2: the age through a measured gap ------------------------- */

/* The lead is driven across the fragment's line of sight and back. The
 * range grows as the drift squared over the offset, and the radar's
 * statistic falls as the fourth power of the range, so a few hundred
 * metres of drift takes it under its threshold and brings it back.
 *
 * Through the gap the ring gains nothing, so the age has to grow by
 * exactly one control period a step; on the step the verdict returns,
 * a push lands at the instant the observation is taken and the age has
 * to read nought. */

static void gate_gap_(void)
{
    static char src[16384];
    static double push[24];
    char so[512];
    ThRun run;
    int gap = 0, longest = 0, drops = 0;
    double prev_age = -1.0;
    int prev_det = -1;

    for (int k = 0; k < 24; k++) {
        /* Out for eight steps, back for eight, still for the rest. */
        push[k] = k < 4 ? 0.0 : (k < 12 ? 400.0 : (k < 20 ? -400.0 : 0.0));
    }
    th_world_(src, sizeof src, 1, 0);
    th_build_(src, "gapped", so, sizeof so);
    th_drive_(&run, so, push, 24);

    for (int k = 0; k < run.n_steps; k++) {
        double det = th_ch_(&run, k, "lead_pilot.look_detected");
        double age = th_ch_(&run, k, "lead_pilot.pic_age");
        ASSERT(th_ch_(&run, k, "lead_pilot.pic_valid") == 1.0);
        if (det > 0.5) {
            if (age != 0.0) {
                fprintf(stderr, "FAIL: step %d meets its threshold and "
                        "publishes an age of %.17g rather than nought\n",
                        k + 1, age);
                exit(1);
            }
            if (prev_det == 0) {
                /* The entry that ends a gap: the age drops from a gap
                 * that had grown to something worth calling one. */
                if (!(prev_age >= 2.0 * TH_CONTROL_DT)) {
                    fprintf(stderr, "FAIL: step %d ends a gap that had "
                            "only reached %.17g s\n", k + 1, prev_age);
                    exit(1);
                }
                drops++;
                if (gap > longest) longest = gap;
                gap = 0;
            }
        } else {
            double want = prev_age + TH_CONTROL_DT;
            if (prev_det != 0) {
                /* The first step of a gap. The step before it met its
                 * threshold at its own last sub-advance, so a push
                 * landed there at the latest: the age is over nought
                 * and at most one control period. */
                if (!(age > 0.0) || age > TH_CONTROL_DT + 1.0e-10) {
                    fprintf(stderr, "FAIL: step %d opens a gap with an "
                            "age of %.17g, outside the control period "
                            "the last push must lie in\n", k + 1, age);
                    exit(1);
                }
            } else if (fabs(age - want) > 1.0e-10) {
                fprintf(stderr, "FAIL: step %d publishes an age of "
                        "%.17g where the step before it published "
                        "%.17g and no entry has arrived since\n",
                        k + 1, age, prev_age);
                exit(1);
            }
            gap++;
        }
        prev_age = age;
        prev_det = det > 0.5 ? 1 : 0;
    }
    if (gap > longest) longest = gap;
    if (longest < 4 || drops < 1) {
        fprintf(stderr, "FAIL: the longest gap this run drove was %d "
                "step(s) and %d ended in a re-acquisition; the arm "
                "needs a gap of several control periods and an entry "
                "to end it\n", longest, drops);
        exit(1);
    }
    g_arms++;
    printf("  the age grows by one control period a step through a gap "
           "of %d steps (%.3g s) and reads nought on the step the "
           "verdict returns, %d re-acquisition(s) in the run\n",
           longest, (double)longest * TH_CONTROL_DT, drops);
}

/* ---- Gate 3: the ungated form is untouched -------------------------- */

/* A world declaring no `source=` records what the compiler at the
 * commit this work started from recorded. Its datalink is left
 * declared, so the transfer pass runs and its offers meet a truth-fed
 * ring exactly as they did before; what the arm is about is the
 * answering, and the answering of an ungated state has not moved. */

static void gate_ungated_identity_(void)
{
    static char src[16384];
    char cmd[2048];
    struct stat st;

    th_world_(src, sizeof src, 0, 1);
    rl_write_file_(WORK_DIR "/plain.kfl", src);
    rl_base_compile_(WORK_DIR, WORK_DIR "/plain.kfl",
                     WORK_DIR "/plainbase", 0);
    rl_compile_(WORK_DIR "/plain.kfl", WORK_DIR "/plainnow", WORK_DIR);
    for (int i = 0; i < 2; i++) {
        const char *stem = i ? "plainnow" : "plainbase";
        snprintf(cmd, sizeof cmd,
                 WORK_DIR "/%s --seed 5150 --envs 2 --episodes 2 --out "
                 WORK_DIR "/%s.k26ep > " WORK_DIR "/%s.log 2>&1",
                 stem, stem, stem);
        rl_run_or_die_(cmd);
    }
    if (!rl_files_equal_(WORK_DIR "/plainbase.k26ep",
                         WORK_DIR "/plainnow.k26ep")) {
        fprintf(stderr, "FAIL: a world declaring no `source=` records "
                "different bytes than it did at " BASE_COMMIT "\n");
        exit(1);
    }
    ASSERT(stat(WORK_DIR "/plainnow.k26ep", &st) == 0);
    g_arms++;
    printf("  the ungated form records what the compiler at "
           BASE_COMMIT " recorded, byte for byte (%lld bytes)\n",
           (long long)st.st_size);
}

/* ---- Gate 4: two processes ------------------------------------------ */

static void gate_two_processes_(void)
{
    char cmd[2048];
    struct stat st;

    /* The gated linked world the receiver arm built and left behind. */
    for (int i = 0; i < 2; i++) {
        snprintf(cmd, sizeof cmd,
                 WORK_DIR "/linked --seed 4242 --envs 2 --episodes 2 "
                 "--out " WORK_DIR "/proc%d.k26ep > " WORK_DIR
                 "/proc%d.log 2>&1", i, i);
        rl_run_or_die_(cmd);
    }
    if (!rl_files_equal_(WORK_DIR "/proc0.k26ep",
                         WORK_DIR "/proc1.k26ep")) {
        fprintf(stderr, "FAIL: two processes at one seed wrote "
                "different episode files for a gated world with a "
                "datalink\n");
        exit(1);
    }
    ASSERT(stat(WORK_DIR "/proc0.k26ep", &st) == 0);
    g_arms++;
    printf("  two processes at one seed record identical episode files "
           "for a gated world with a datalink (%lld bytes)\n",
           (long long)st.st_size);
}

int main(void)
{
    int have_base;

    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_run_or_die_("cp examples/assets/calibration_box.k26asm "
                   "examples/assets/calibration_box.k26mesh "
                   WORK_DIR "/");

    printf("test_rl_track_held: how a gated information state "
           "answers\n");
    if (!rl_libs_present_("test_rl_track_held")) {
        printf("test_rl_track_held: stood down (stack archives "
               "absent)\n");
        return 77;
    }

    /* Built once and used by two arms; the helper fails the gate on
     * absent history unless it is stood down aloud. */
    have_base = rl_base_build_(BASE_COMMIT, WORK_DIR);

    gate_receiver_(have_base);
    gate_gap_();
    if (have_base) {
        gate_ungated_identity_();
    } else {
        printf("  NOT MEASURED: the ungated form was not compared "
               "against " BASE_COMMIT ", its arm having been stood "
               "down above\n");
    }
    gate_two_processes_();

    printf("test_rl_track_held: %d arm(s) passed\n", g_arms);
    return 0;
}
