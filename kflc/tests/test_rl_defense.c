/* test_rl_defense.c: the defense payload statement, detection, and the
 * information state.
 *
 * Gates:
 *   1. Refusals. Every rule the payload statement carries is written
 *      as a program that breaks it: an unknown kind, a missing
 *      required key, a key that belongs to another kind (refused
 *      naming that kind rather than reported as unknown), a body that
 *      carries no vehicle, a target that carries none, a payload
 *      observed by the wrong form, a payload observing its own
 *      platform, a distribution on a key fixed at construction, an
 *      unknown modality, a modality on the form that has none, and a
 *      declaration inside a conditional.
 *   2. Every agent block declares an observation, above one agent.
 *      The same program at one agent compiles, which is what tells the
 *      rule from a blanket refusal, and a two-agent program in which
 *      every block declares one compiles beside it.
 *   3. The published channels. Seven suffixes for a detection observe
 *      and nine for a track observe, in the design's order, qualified
 *      by the agent that owns them, with the solver's iteration count
 *      absent from both.
 *   4. Validity. The first observation of an episode is unavailable
 *      because the retarded time precedes the only sample, and the
 *      remaining eight channels zero-fill; the steps after it are
 *      valid and the age is the light time over the range to the
 *      metre. A long-range fixture holds the other half: while the
 *      elapsed time is shorter than the light time the observation is
 *      unavailable, and it becomes available on the step the history
 *      first covers the lag. A second episode reproduces the first
 *      bit for bit, which is what says the history the ring still
 *      holds from the first does not reach the second.
 *   5. Aspect. A target whose silhouette is genuinely aspect dependent
 *      moves all three detection channels as it turns, by the ratio
 *      its geometry predicts and not merely in some direction; a
 *      spherical target under the same rotation moves none of them,
 *      which is what tells a channel that honours aspect from one
 *      that reads something else that happens to change.
 *   6. One generator. An artifact carrying every payload kind names
 *      the tier's generator nowhere, hands every tier evaluator a null
 *      one, and leaves no undefined reference to it once compiled.
 *      Three arms because the rule has three ways of being broken and
 *      one check catches only one of them; each carries a perturbation
 *      of the same emitted source and is required to catch it.
 *   7. The per-observer target cap. A program tracking more targets
 *      against one information state than the library admits is
 *      refused naming the payload, the count and the limit.
 *   8. Determinism. Two processes at one seed produce identical
 *      episode files for a two-agent program with every payload kind
 *      of this surface bound.
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
#include <sys/wait.h>

#include "rl_gate_util.h"
/* The modality mapping is checked against the library's own
 * enumeration rather than against a number restated here. */
#include "k26astro_infostate/infostate.h"

#define WORK_DIR "/tmp/kflc_rl_defense_test"

static int g_arms;

/* ---- Sources -------------------------------------------------------- */

/* The world every behaviour arm is built on. `watcher` carries all
 * four payloads; `mover` is a two-by-one-by-one metre box turning
 * about its z axis at 0.05 rad/s, so its silhouette along a fixed
 * line of sight is a function of time with a closed form this gate
 * computes independently.
 *
 * The two craft sit twenty kilometres apart along the y axis, which
 * puts the light time at 66.7 microseconds against a control period
 * of half a second: the information state is consistent rather than
 * late at this range, which is the regime the design says these
 * environments live in. */
#define DEF_HEAD \
    "form DEF\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body watcher assembly=\"crew_vehicle_10t.k26asm\"" \
    " parent=earth pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0\n"

#define DEF_MOVER(asset, spin) \
    "    astro_body mover assembly=\"" asset "\"" \
    " parent=earth pos_x=7.0e6 pos_y=2.0e4 pos_z=0.0 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=" spin "\n"

#define DEF_PAYLOADS \
    "    astro_payload eye body=watcher kind=detect_ir aperture_m=1.0" \
    " integration_s=0.5 passband_lo_um=3.0 passband_hi_um=12.0" \
    " throughput=0.5 snr_threshold=5.0 target_temp_k=300.0" \
    " target_emissivity=0.9 optics_temp_k=280.0" \
    " optics_emissivity=0.05\n" \
    "    astro_payload rf body=watcher kind=detect_radar p_tx_w=2000.0" \
    " g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10 loss_sys_db=3.0" \
    " bandwidth_hz=1.0e6 t_sys_k=290.0 noise_figure=2.0" \
    " snr_threshold=10.0\n" \
    "    astro_payload beam body=watcher kind=detect_lidar" \
    " pulse_energy_j=0.1 wavelength_nm=1064.0 aperture_rx_m=0.5" \
    " atmospheric_tx=1.0 detector_efficiency=0.3 snr_threshold=5.0" \
    " target_albedo=0.2\n" \
    "    astro_payload eye_wide body=watcher kind=detect_ir" \
    " aperture_m=2.0" \
    " integration_s=0.5 passband_lo_um=3.0 passband_hi_um=12.0" \
    " throughput=0.5 snr_threshold=5.0 target_temp_k=300.0" \
    " target_emissivity=0.9 optics_temp_k=280.0" \
    " optics_emissivity=0.05\n" \
    "    astro_payload picture body=watcher kind=infostate history=256\n"

#define DEF_EPISODE \
    "    episode\n" \
    "        control_dt 0.5\n" \
    "        substeps 4\n" \
    "        horizon 8\n" \
    "    end\n"

#define DEF_AGENTS \
    "    agent hunter\n" \
    "        action nudge box -1.0 1.0 default 0.0\n" \
    "        observe detect eye of mover as ir\n" \
    "        observe detect rf of mover as radar\n" \
    "        observe detect beam of mover as lidar\n" \
    "        observe detect eye_wide of mover as ir2\n" \
    "        observe track picture of mover modality=radar as trk\n" \
    "        objective\n" \
    "            reward hunter.ir_snr\n" \
    "        end\n" \
    "    end\n" \
    "    agent quarry\n" \
    "        action dodge box -1.0 1.0 default 0.0\n" \
    "        observe mover from watcher mode=geometric as los\n" \
    "        objective\n" \
    "            reward 0.0 - hunter.ir_snr\n" \
    "        end\n" \
    "    end\n" \
    "    on_step\n" \
    "        watcher.vel_x = watcher.vel_x + nudge\n" \
    "        mover.vel_x = mover.vel_x + dodge\n" \
    "    end\n" \
    "end\n" \
    "end\n"

static const char *const BOX_KFL =
    DEF_HEAD DEF_MOVER("calibration_box.k26asm", "0.05")
    DEF_PAYLOADS DEF_EPISODE DEF_AGENTS;

/* The same world with a spherical target and the same rotation. A
 * sphere presents its great circle at every aspect, so every
 * detection channel must sit still while the box's move: this is the
 * control that says the arm above measures aspect rather than the
 * passage of time. */
static const char *const BALL_KFL =
    DEF_HEAD DEF_MOVER("scratch_ball.k26asm", "0.05")
    DEF_PAYLOADS DEF_EPISODE DEF_AGENTS;

/* A spherical collider of the same characteristic size as the box, so
 * the two fixtures differ in shape and not in scale. */
static const char *const BALL_ASM =
    "# scratch_ball.k26asm - a sphere, for the aspect control arm.\n"
    "#\n"
    "# Not a craft. It exists so a detection channel that honours\n"
    "# aspect and one that ignores it can be told apart: a sphere\n"
    "# presents the same area from every direction, so its signature\n"
    "# is the one silhouette in the format that cannot move.\n"
    "assembly scratch_ball\n"
    "    frame x_to_port\n"
    "    provenance mass \"the shape's own definition\" computed\n"
    "    provenance inertia \"derived from the geometry by the compiler\""
    " computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0.0 0.0 0.0\n"
    "        collider sphere 0.0 0.0 0.0 0.7978845608028654\n"
    "    end\n"
    "end\n";

/* ---- Refusal plumbing ------------------------------------------------ */

static int check_(const char *src, char **out_log)
{
    rl_write_file_(WORK_DIR "/case.kfl", src);
    int rc = system("./bin/kflc --check " WORK_DIR "/case.kfl > "
                    WORK_DIR "/case.log 2>&1");
    static char buf[16384];
    FILE *f = fopen(WORK_DIR "/case.log", "rb");
    ASSERT(f != NULL);
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    buf[n] = '\0';
    fclose(f);
    if (out_log) *out_log = buf;
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

static void must_refuse_(const char *what, const char *src,
                         const char *const *needles)
{
    char *log = NULL;
    int rc = check_(src, &log);
    if (rc == 0) {
        fprintf(stderr, "FAIL %s: accepted, expected a refusal\n", what);
        exit(1);
    }
    for (int i = 0; needles[i]; i++) {
        if (strstr(log, needles[i]) == NULL) {
            fprintf(stderr, "FAIL %s: diagnostic lacks \"%s\"\n---\n%s---\n",
                    what, needles[i], log);
            exit(1);
        }
    }
    g_arms++;
    printf("  refused: %s\n", what);
}

static void must_accept_(const char *what, const char *src)
{
    char *log = NULL;
    int rc = check_(src, &log);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s: refused\n---\n%s---\n", what, log);
        exit(1);
    }
    g_arms++;
    printf("  accepted: %s\n", what);
}

/* ---- Gate 1: the refusals ------------------------------------------- */

static void gate_refusals_(void)
{
    char src[16384];

    /* An unknown kind, with the admissible set named. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_payload eye body=watcher kind=detect_sonar\n",
        DEF_EPISODE "end\nend\n");
    const char *f1[] = { "unknown kind `detect_sonar`", "detect_ir",
                         "detect_radar", "detect_lidar", "infostate", NULL };
    must_refuse_("an unknown payload kind", src, f1);

    /* A missing required key, named. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_payload eye body=watcher kind=detect_ir"
        " aperture_m=1.0 integration_s=0.5 passband_lo_um=3.0"
        " passband_hi_um=12.0 throughput=0.5\n",
        DEF_EPISODE "end\nend\n");
    const char *f2[] = { "kind `detect_ir` requires `snr_threshold=`",
                         "requires `target_temp_k=`",
                         "requires `target_emissivity=`", NULL };
    must_refuse_("a payload missing required keys", src, f2);

    /* A key of another kind, refused naming that kind. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_payload eye body=watcher kind=detect_ir"
        " aperture_m=1.0 integration_s=0.5 passband_lo_um=3.0"
        " passband_hi_um=12.0 throughput=0.5 snr_threshold=5.0"
        " target_temp_k=300.0 target_emissivity=0.9 freq_hz=1.0e10\n",
        DEF_EPISODE "end\nend\n");
    const char *f3[] = { "`freq_hz=` belongs to kind `detect_radar`",
                         "not to `detect_ir`", NULL };
    must_refuse_("a key that belongs to another kind", src, f3);

    /* A key of no kind at all, refused as unknown. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_payload eye body=watcher kind=infostate"
        " history=256 colour=blue\n",
        DEF_EPISODE "end\nend\n");
    const char *f4[] = { "unknown key `colour=`", "kind `infostate`", NULL };
    must_refuse_("a key belonging to no kind", src, f4);

    /* A body that carries no vehicle, refused naming both statements. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_body plain gm=1.0 parent=earth pos_x=7.1e6"
        " vel_y=7500.0\n"
        "    astro_payload eye body=plain kind=infostate\n",
        DEF_EPISODE "end\nend\n");
    const char *f5[] = { "`body=plain` declares no `assembly=`",
                         "carries no vehicle", NULL };
    must_refuse_("a payload on a body with no vehicle", src, f5);

    /* A target that carries none. */
    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_body plain gm=1.0 parent=earth pos_x=7.1e6"
        " vel_y=7500.0\n",
        DEF_PAYLOADS DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe detect eye of plain as ir\n"
        "    end\n"
        "end\nend\n");
    const char *f6[] = { "`plain` declares no `assembly=`",
                         "signature is computed from", NULL };
    must_refuse_("a detection target with no geometry", src, f6);

    /* A detection payload put to the track form, and the reverse. */
    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_PAYLOADS, DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe track eye of mover as trk\n"
        "    end\n"
        "end\nend\n");
    const char *f7[] = { "`eye` is of kind `detect_ir`",
                         "takes a payload of kind `infostate`", NULL };
    must_refuse_("an information-state form over a detection payload",
                 src, f7);

    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_PAYLOADS, DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe detect picture of mover as ir\n"
        "    end\n"
        "end\nend\n");
    const char *f8[] = { "`picture` is of kind `infostate`",
                         "takes a detection payload", NULL };
    must_refuse_("a detection form over an information state", src, f8);

    /* A payload observing the platform that carries it. */
    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_PAYLOADS, DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe detect eye of watcher as ir\n"
        "    end\n"
        "end\nend\n");
    const char *f9[] = { "carried by `watcher` itself",
                         "does not observe its own platform", NULL };
    must_refuse_("a payload observing its own platform", src, f9);

    /* A distribution on a key that is fixed when the ring is built. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_payload picture body=watcher kind=infostate"
        " history=uniform(64.0,256.0)\n",
        DEF_EPISODE "end\nend\n");
    const char *f10[] = { "fixed when the payload is constructed",
                          "admits no distribution form", NULL };
    must_refuse_("a distribution on a construction-fixed key", src, f10);

    /* An unknown modality, and a modality on the form that has none. */
    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_PAYLOADS, DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe track picture of mover modality=sonar as trk\n"
        "    end\n"
        "end\nend\n");
    const char *f11[] = { "unknown modality `sonar`",
                          "none, ir, radar, lidar and ephem", NULL };
    must_refuse_("an unknown track modality", src, f11);

    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_PAYLOADS, DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe detect eye of mover modality=radar as ir\n"
        "    end\n"
        "end\nend\n");
    const char *f12[] = { "`modality=` belongs to the `observe track` form",
                          NULL };
    must_refuse_("a modality on the detection form", src, f12);

    /* A payload named by nothing, and a name naming no payload. */
    snprintf(src, sizeof src, "%s%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_PAYLOADS, DEF_EPISODE,
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe detect nothere of mover as ir\n"
        "    end\n"
        "end\nend\n");
    const char *f13[] = { "no astro_payload of that name", NULL };
    must_refuse_("an observe naming no payload", src, f13);

    /* Two payloads of one name. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    astro_payload picture body=watcher kind=infostate\n"
        "    astro_payload picture body=watcher kind=infostate\n",
        DEF_EPISODE "end\nend\n");
    const char *f14[] = { "a payload of that name is already declared",
                          NULL };
    must_refuse_("two payloads of one name", src, f14);

    /* A payload inside a conditional: the set is part of the
     * program's identity and cannot be conditional. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        "    if 1.0 > 0.0\n"
        "        astro_payload picture body=watcher kind=infostate\n"
        "    end\n",
        DEF_EPISODE "end\nend\n");
    const char *f15[] = { "must be top level",
                          "part of the compiled program's identity", NULL };
    must_refuse_("a payload declaration inside a conditional", src, f15);

    /* A payload inside `on_step`, which is where the design says it
     * may not be: the block calls evaluators and constructs nothing. */
    snprintf(src, sizeof src, "%s%s%s%s",
        DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
        DEF_EPISODE,
        "    on_step\n"
        "        astro_payload picture body=watcher kind=infostate\n"
        "    end\n"
        "end\nend\n");
    const char *f16[] = { "`astro_payload` is not allowed inside an "
                          "on_step block", NULL };
    must_refuse_("a payload declaration inside on_step", src, f16);

    /* And the whole fixture, accepted. */
    must_accept_("every payload kind of this surface in one world",
                 BOX_KFL);
}

/* ---- Gate 2: every agent block declares an observation --------------- */

static void gate_agent_observation_(void)
{
    static const char *const HEAD =
        "form OBSRULE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6"
        " vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 10.0\n"
        "        horizon 4\n"
        "    end\n";
    char src[4096];

    /* Two agents, the second declaring no observation: refused. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent alpha\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "    agent beta\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "        objective\n"
        "            reward 1.0\n"
        "        end\n"
        "    end\n"
        "end\nend\n");
    const char *f1[] = { "agent `beta`", "declares no `observe ... as`",
                         "qualifier on its own observation channels",
                         NULL };
    must_refuse_("a silent agent block above one agent", src, f1);

    /* The same block alone: one agent, accepted, because at one agent
     * the name publishes unqualified and nothing depends on it. This
     * is what tells the rule from a blanket refusal. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent beta\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "        objective\n"
        "            reward 1.0\n"
        "        end\n"
        "    end\n"
        "end\nend\n");
    must_accept_("the same silent block alone, at one agent", src);

    /* Two agents that each declare one: accepted. */
    snprintf(src, sizeof src, "%s%s", HEAD,
        "    agent alpha\n"
        "        action thrust box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as trk\n"
        "    end\n"
        "    agent beta\n"
        "        action brake box -1.0 1.0 default 0.0\n"
        "        observe craft from earth mode=geometric as look\n"
        "    end\n"
        "end\nend\n");
    must_accept_("two agents that each declare an observation", src);
}

/* ---- Gate 7: the per-observer target cap ----------------------------- */

static void gate_target_cap_(void)
{
    /* One observer and sixty-five targets, which is one past the
     * library's per-observer cap. */
    static char src[262144];
    int n = snprintf(src, sizeof src, "%s", DEF_HEAD);
    for (int i = 0; i < 65; i++) {
        n += snprintf(src + n, sizeof src - (size_t)n,
            "    astro_body t%d assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=%d.0e6 vel_y=7000.0\n", i, 7 + i % 3);
    }
    n += snprintf(src + n, sizeof src - (size_t)n, "%s",
        "    astro_payload picture body=watcher kind=infostate\n"
        "    episode\n        control_dt 0.5\n        horizon 4\n    end\n"
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n");
    for (int i = 0; i < 65; i++) {
        n += snprintf(src + n, sizeof src - (size_t)n,
            "        observe track picture of t%d as k%d\n", i, i);
    }
    n += snprintf(src + n, sizeof src - (size_t)n, "%s",
        "    end\nend\nend\n");
    ASSERT((size_t)n < sizeof src);
    const char *f[] = { "astro_payload `picture`", "65 targets are tracked",
                        "per-observer limit is 64", NULL };
    must_refuse_("more targets than the information state admits", src, f);

    /* Sixty-four is admitted, so the refusal is a cap and not a
     * blanket refusal of a large program. */
    n = snprintf(src, sizeof src, "%s", DEF_HEAD);
    for (int i = 0; i < 64; i++) {
        n += snprintf(src + n, sizeof src - (size_t)n,
            "    astro_body t%d assembly=\"calibration_box.k26asm\""
            " parent=earth pos_x=%d.0e6 vel_y=7000.0\n", i, 7 + i % 3);
    }
    n += snprintf(src + n, sizeof src - (size_t)n, "%s",
        "    astro_payload picture body=watcher kind=infostate\n"
        "    episode\n        control_dt 0.5\n        horizon 4\n    end\n"
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n");
    for (int i = 0; i < 64; i++) {
        n += snprintf(src + n, sizeof src - (size_t)n,
            "        observe track picture of t%d as k%d\n", i, i);
    }
    n += snprintf(src + n, sizeof src - (size_t)n, "%s",
        "    end\nend\nend\n");
    ASSERT((size_t)n < sizeof src);
    must_accept_("exactly as many targets as it admits", src);
}

/* ---- Driving plumbing ----------------------------------------------- */

typedef struct {
    void      *so;
    RlSurface  s;
    K26RlEnv  *env;
    RlSpecView spec;
} Driven;

static void open_(Driven *d, const char *kfl, const char *stem,
                  uint64_t seed)
{
    char path[512], out[512];
    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    rl_write_file_(path, kfl);
    rl_compile_(path, out, WORK_DIR);
    snprintf(path, sizeof path, WORK_DIR "/%s.rlenv.so", stem);
    d->so = rl_dlopen_(path);
    rl_resolve_surface_(d->so, &d->s);
    ASSERT(d->s.create(seed, 1u, &d->env) == K26RL_OK);
    static uint8_t blob[65536];
    int32_t need = d->s.spec(d->env, blob, (uint32_t)sizeof blob);
    ASSERT(need > 0 && (size_t)need <= sizeof blob);
    rl_parse_spec_(blob, (uint32_t)need, &d->spec);
}

static void close_(Driven *d)
{
    d->s.destroy(d->env);
    dlclose(d->so);
}

static int chan_(const Driven *d, const char *name)
{
    for (int i = 0; i < d->spec.n_chan_names; i++) {
        if (strcmp(d->spec.chan_names[i], name) == 0) return i;
    }
    fprintf(stderr, "FAIL: no channel called %s\n", name);
    for (int i = 0; i < d->spec.n_chan_names; i++) {
        fprintf(stderr, "  %2d %s\n", i, d->spec.chan_names[i]);
    }
    exit(1);
}

/* ---- Gate 3: the published channels ---------------------------------- */

static void gate_channels_(Driven *d)
{
    static const char *const DET[] = {
        "_detected", "_snr", "_range", "_dir_x", "_dir_y", "_dir_z",
        "_aspect", NULL
    };
    static const char *const TRK[] = {
        "_valid", "_pos_x", "_pos_y", "_pos_z", "_vel_x", "_vel_y",
        "_vel_z", "_range", "_age", NULL
    };
    static const char *const BASES[] = { "hunter.ir", "hunter.radar",
                                         "hunter.lidar", NULL };
    for (int b = 0; BASES[b]; b++) {
        int first = -1;
        for (int k = 0; DET[k]; k++) {
            char want[96];
            snprintf(want, sizeof want, "%s%s", BASES[b], DET[k]);
            int c = chan_(d, want);
            if (first < 0) first = c;
            if (c != first + k) {
                fprintf(stderr, "FAIL %s at channel %d, expected %d\n",
                        want, c, first + k);
                exit(1);
            }
        }
        g_arms++;
        printf("  %s publishes 7 channels from %d, in order\n",
               BASES[b], first);
    }
    int first = -1;
    for (int k = 0; TRK[k]; k++) {
        char want[96];
        snprintf(want, sizeof want, "hunter.trk%s", TRK[k]);
        int c = chan_(d, want);
        if (first < 0) first = c;
        ASSERT(c == first + k);
    }
    g_arms++;
    printf("  hunter.trk publishes 9 channels from %d, in order\n", first);

    /* The solver's iteration count is a diagnostic and is published to
     * no agent. */
    for (int i = 0; i < d->spec.n_chan_names; i++) {
        if (strstr(d->spec.chan_names[i], "iter") != NULL) {
            fprintf(stderr, "FAIL: channel %s publishes the solver's "
                    "iteration count\n", d->spec.chan_names[i]);
            exit(1);
        }
    }
    g_arms++;
    printf("  no channel publishes the solver's iteration count\n");
}

/* ---- Gate 4: validity ------------------------------------------------ */

static void gate_validity_(void)
{
    Driven d;
    open_(&d, BOX_KFL, "box", 11u);
    gate_channels_(&d);

    int c_valid = chan_(&d, "hunter.trk_valid");
    int c_age   = chan_(&d, "hunter.trk_age");
    int c_rng   = chan_(&d, "hunter.trk_range");
    int c_px    = chan_(&d, "hunter.trk_pos_x");
    int c_det   = chan_(&d, "hunter.ir_range");

    double v[128], act[2] = { 0.0, 0.0 };
    ASSERT(d.spec.obs_total <= 128);

    /* At the epoch the only sample is the epoch itself and the
     * retarded time is strictly earlier, so the observation is
     * genuinely unavailable and the other eight zero-fill. The
     * detection channels are unaffected: they are geometric and
     * carry no light time. */
    ASSERT(d.s.obs(d.env, v) == K26RL_OK);
    ASSERT(v[c_valid] == 0.0);
    ASSERT(v[c_age] == 0.0 && v[c_rng] == 0.0 && v[c_px] == 0.0);
    ASSERT(v[c_det] > 19000.0);
    g_arms++;
    printf("  epoch: track invalid and zero filled, range %.1f m still "
           "published by detection\n", v[c_det]);

    /* The three detection payloads are three different instruments
     * against one target, so their ratios are far from one. An
     * implementation that read one payload's parameters for all three
     * would put the same number in all three channels; requiring them
     * to differ by more than a factor of two is what tells the
     * per-payload parameter store from a shared one. */
    {
        double snr[3] = { v[chan_(&d, "hunter.ir_snr")],
                          v[chan_(&d, "hunter.radar_snr")],
                          v[chan_(&d, "hunter.lidar_snr")] };
        for (int a = 0; a < 3; a++) {
            for (int b = a + 1; b < 3; b++) {
                double r = snr[a] > snr[b] ? snr[a] / snr[b]
                                           : snr[b] / snr[a];
                if (!(r > 2.0)) {
                    fprintf(stderr, "FAIL: channels %d and %d differ by "
                            "only %.4f\n", a, b, r);
                    exit(1);
                }
            }
        }
        g_arms++;
        printf("  three instruments read %.6g, %.6g and %.6g against one "
               "target\n", snr[0], snr[1], snr[2]);
    }

    /* Two payloads of one kind differing only in aperture. This is the
     * arm that tells a per-payload parameter store from a shared one:
     * three kinds against one target differ whatever parameters they
     * are handed, because the three models differ, but two instances
     * of one model handed one payload's parameters read exactly the
     * same number. A wider aperture collects more signal against a
     * background that does not grow with it, so the wider one must
     * read higher and not merely differently. */
    {
        double narrow = v[chan_(&d, "hunter.ir_snr")];
        double wide   = v[chan_(&d, "hunter.ir2_snr")];
        if (!(wide > narrow)) {
            fprintf(stderr, "FAIL: a 1 m aperture reads %.9g and a 2 m "
                    "aperture reads %.9g against one target\n",
                    narrow, wide);
            exit(1);
        }
        g_arms++;
        printf("  one model, two apertures: 1 m reads %.6g and 2 m reads "
               "%.6g\n", narrow, wide);
    }

    /* Every step after it is valid, and the age is the light time
     * over the range. */
    double first_age = 0.0;
    for (int s = 1; s <= 8; s++) {
        ASSERT(d.s.step(d.env, act) == K26RL_OK);
        ASSERT(d.s.obs(d.env, v) == K26RL_OK);
        if (s > 4) continue;   /* the horizon boundary is exercised below */
        ASSERT(v[c_valid] == 1.0);
        double want = v[c_rng] / 299792458.0;
        double err = fabs(v[c_age] - want);
        if (!(err < 1.0e-15)) {
            fprintf(stderr, "FAIL step %d: age %.12e, range/c %.12e\n",
                    s, v[c_age], want);
            exit(1);
        }
        if (s == 1) first_age = v[c_age];
    }
    g_arms++;
    printf("  steps 1 to 4 valid, age equals range over c "
           "(step 1: %.6e s)\n", first_age);

    /* A second episode reproduces the first bit for bit. The ring
     * still holds the first episode's samples, so this is what says
     * they do not reach the second. */
    double ep1[16][128];
    ASSERT(d.s.reset(d.env) == K26RL_OK);
    for (int s = 0; s <= 6; s++) {
        ASSERT(d.s.obs(d.env, ep1[s]) == K26RL_OK);
        if (s < 6) ASSERT(d.s.step(d.env, act) == K26RL_OK);
    }
    double ep2[16][128];
    ASSERT(d.s.reset(d.env) == K26RL_OK);
    for (int s = 0; s <= 6; s++) {
        ASSERT(d.s.obs(d.env, ep2[s]) == K26RL_OK);
        if (s < 6) ASSERT(d.s.step(d.env, act) == K26RL_OK);
    }
    for (int s = 0; s <= 6; s++) {
        for (uint32_t c = 0; c < d.spec.obs_total; c++) {
            if (ep1[s][c] == ep2[s][c]) continue;
            fprintf(stderr, "FAIL: episode 2 differs at step %d, channel "
                    "%u (%s): %.17g against %.17g\n", s, c,
                    d.spec.chan_names[c], ep1[s][c], ep2[s][c]);
            exit(1);
        }
    }
    ASSERT(ep1[0][c_valid] == 0.0 && ep1[1][c_valid] == 1.0);
    g_arms++;
    printf("  two episodes bit identical over 7 steps, each invalid at "
           "its own epoch\n");
    close_(&d);
}

/* The other half of the validity rule: at a range whose light time
 * exceeds the control period, the observation is unavailable until the
 * episode's own history covers the lag, and then it is available. This
 * is the arm a binding that published whatever the ring happened to
 * hold would fail. */
static void gate_validity_long_(void)
{
    /* Forty thousand kilometres between the two craft puts the light
     * time at 0.1335 s against a control period of 0.05 s, so the
     * first two steps cannot be answered and the third can. */
    static const char *const LONG_KFL =
        "form DEFLONG\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body watcher assembly=\"crew_vehicle_10t.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0\n"
        "    astro_body mover assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=4.0e7 vel_y=7546.0\n"
        "    astro_payload picture body=watcher kind=infostate"
        " history=1024\n"
        "    episode\n"
        "        control_dt 0.05\n"
        "        substeps 1\n"
        "        horizon 40\n"
        "    end\n"
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe track picture of mover as trk\n"
        "    end\n"
        "    agent quarry\n"
        "        action dodge box -1.0 1.0 default 0.0\n"
        "        observe mover from watcher mode=geometric as los\n"
        "    end\n"
        "end\n"
        "end\n";
    Driven d;
    open_(&d, LONG_KFL, "far", 13u);
    int c_valid = chan_(&d, "hunter.trk_valid");
    int c_age   = chan_(&d, "hunter.trk_age");
    double v[128], act[2] = { 0.0, 0.0 };

    /* Twice: once on the fresh handle, whose history begins at create,
     * and once after a reset, whose history begins at the reset. The
     * two must agree, and at this range they can only agree if the
     * episode's own first sample sits at its epoch rather than a
     * sub-advance later: a lag of 0.1335 s against a step of 0.05 s
     * makes one sub-advance of slack the difference between the third
     * step and the fourth. */
    int first_valid[2] = { -1, -1 };
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) ASSERT(d.s.reset(d.env) == K26RL_OK);
        for (int s = 0; s <= 8; s++) {
            ASSERT(d.s.obs(d.env, v) == K26RL_OK);
            if (v[c_valid] != 0.0 && first_valid[pass] < 0) {
                first_valid[pass] = s;
            }
            if (v[c_valid] == 0.0) ASSERT(v[c_age] == 0.0);
            if (s < 8) ASSERT(d.s.step(d.env, act) == K26RL_OK);
        }
    }
    /* The lag is 0.1335 s and a step is 0.05 s, so the first step
     * whose elapsed time covers it is step 3. */
    if (first_valid[0] != 3 || first_valid[1] != 3) {
        fprintf(stderr, "FAIL: first valid step %d on a fresh handle and "
                "%d after a reset, expected 3 for both\n",
                first_valid[0], first_valid[1]);
        exit(1);
    }
    g_arms++;
    printf("  at 40000 km the track is unavailable until step %d on a "
           "fresh handle and step %d after a reset, each the first "
           "whose history covers the lag\n", first_valid[0],
           first_valid[1]);
    close_(&d);
}

/* The third validity arm, and the one the seeding rule is really
 * about. At a range whose light time is shorter than one sub-advance
 * but not negligible against it, the retarded time of an episode's
 * first step falls between that episode's epoch and its first stepped
 * sample. A history that begins at the epoch answers it from inside
 * the episode; one that begins a sub-advance later has to reach back
 * across the reset into the previous episode, and would answer with a
 * position that episode never had.
 *
 * The two passes are the measurement: the first runs on a fresh
 * handle, whose history begins when the world is built, and the
 * second after a reset. They must agree channel for channel. */
static void gate_validity_seed_(void)
{
    /* Twelve thousand kilometres is a light time of 0.04003 s against
     * a sub-advance of 0.05 s. */
    static const char *const SEED_KFL =
        "form DEFSEED\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body watcher assembly=\"crew_vehicle_10t.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0\n"
        "    astro_body mover assembly=\"calibration_box.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=1.2e7 vel_y=1000.0\n"
        "    astro_payload picture body=watcher kind=infostate"
        " history=1024\n"
        "    episode\n"
        "        control_dt 0.05\n"
        "        substeps 1\n"
        "        horizon 40\n"
        "    end\n"
        "    agent hunter\n"
        "        action nudge box -1.0 1.0 default 0.0\n"
        "        observe track picture of mover as trk\n"
        "    end\n"
        "    agent quarry\n"
        "        action dodge box -1.0 1.0 default 0.0\n"
        "        observe mover from watcher mode=geometric as los\n"
        "    end\n"
        "end\n"
        "end\n";
    Driven d;
    open_(&d, SEED_KFL, "seed", 17u);
    int c_valid = chan_(&d, "hunter.trk_valid");
    int c_age   = chan_(&d, "hunter.trk_age");
    double act[2] = { 0.0, 0.0 };
    double pass[2][8][128];
    for (int p = 0; p < 2; p++) {
        if (p == 1) ASSERT(d.s.reset(d.env) == K26RL_OK);
        for (int s = 0; s < 8; s++) {
            ASSERT(d.s.obs(d.env, pass[p][s]) == K26RL_OK);
            if (s < 7) ASSERT(d.s.step(d.env, act) == K26RL_OK);
        }
    }
    /* The arm measures nothing unless the lag really is inside one
     * sub-advance and the channel really is answered, so both are
     * asserted before the comparison. */
    ASSERT(pass[0][1][c_valid] == 1.0);
    ASSERT(pass[0][1][c_age] > 0.03 && pass[0][1][c_age] < 0.05);
    for (int s = 0; s < 8; s++) {
        for (uint32_t c = 0; c < d.spec.obs_total; c++) {
            if (pass[0][s][c] == pass[1][s][c]) continue;
            fprintf(stderr, "FAIL: with a light time of %.6f s inside a "
                    "sub-advance of 0.05 s, the episode after a reset "
                    "differs at step %d, channel %u (%s): %.17g against "
                    "%.17g\n", pass[0][1][c_age], s, c,
                    d.spec.chan_names[c], pass[0][s][c], pass[1][s][c]);
            exit(1);
        }
    }
    g_arms++;
    printf("  a light time of %.6f s inside a sub-advance of 0.05 s: the "
           "episode after a reset matches the first over 8 steps\n",
           pass[0][1][c_age]);
    close_(&d);
}

/* ---- Gate 5: aspect -------------------------------------------------- */

/* The projected area of the box target along a look direction in its
 * own frame, computed here from the asset's own dimensions so the arm
 * has an expectation of its own rather than a recorded number. */
static double box_area_(double theta)
{
    double s = fabs(sin(theta)), c = fabs(cos(theta));
    return 4.0 * (0.25 * s + 0.5 * c);
}

static void gate_aspect_(void)
{
    Driven box, ball;
    open_(&box, BOX_KFL, "box_a", 5u);
    open_(&ball, BALL_KFL, "ball_a", 5u);

    double vb[128], vs[128], act[2] = { 0.0, 0.0 };
    int b_ir = chan_(&box, "hunter.ir_snr");
    int b_rd = chan_(&box, "hunter.radar_snr");
    int b_ld = chan_(&box, "hunter.lidar_snr");
    int s_ir = chan_(&ball, "hunter.ir_snr");
    int s_rd = chan_(&ball, "hunter.radar_snr");
    int s_ld = chan_(&ball, "hunter.lidar_snr");

    ASSERT(box.s.obs(box.env, vb) == K26RL_OK);
    ASSERT(ball.s.obs(ball.env, vs) == K26RL_OK);
    double b0[3] = { vb[b_ir], vb[b_rd], vb[b_ld] };
    double s0[3] = { vs[s_ir], vs[s_rd], vs[s_ld] };
    ASSERT(b0[0] > 0.0 && b0[1] > 0.0 && b0[2] > 0.0);
    ASSERT(s0[0] > 0.0 && s0[1] > 0.0 && s0[2] > 0.0);

    for (int s = 0; s < 6; s++) {
        ASSERT(box.s.step(box.env, act) == K26RL_OK);
        ASSERT(ball.s.step(ball.env, act) == K26RL_OK);
    }
    ASSERT(box.s.obs(box.env, vb) == K26RL_OK);
    ASSERT(ball.s.obs(ball.env, vs) == K26RL_OK);

    /* The box has turned 0.05 rad/s for 3 s. Photon-count and
     * shot-noise detection give a ratio in the square root of the
     * area; the radar's flat-plate cross-section is in its square. */
    double ratio = box_area_(0.05 * 3.0) / box_area_(0.0);
    struct { const char *name; double got; double want; } arms[3] = {
        { "infrared", vb[b_ir] / b0[0], sqrt(ratio) },
        { "radar",    vb[b_rd] / b0[1], ratio * ratio },
        { "lidar",    vb[b_ld] / b0[2], sqrt(ratio) }
    };
    for (int i = 0; i < 3; i++) {
        double err = fabs(arms[i].got - arms[i].want) / arms[i].want;
        if (!(err < 2.0e-3)) {
            fprintf(stderr, "FAIL aspect %s: measured %.9f, predicted "
                    "%.9f\n", arms[i].name, arms[i].got, arms[i].want);
            exit(1);
        }
        g_arms++;
        printf("  aspect %s: signal-to-noise moves by %.6f, the geometry "
               "predicts %.6f\n", arms[i].name, arms[i].got, arms[i].want);
    }

    /* The control. A sphere presents the same silhouette from every
     * direction, so nothing about the same rotation may move its
     * channels; the only change is the geometry the two craft share,
     * which at this separation is below a part in ten thousand. */
    double moved[3] = { fabs(vs[s_ir] / s0[0] - 1.0),
                        fabs(vs[s_rd] / s0[1] - 1.0),
                        fabs(vs[s_ld] / s0[2] - 1.0) };
    for (int i = 0; i < 3; i++) {
        if (!(moved[i] < 1.0e-4)) {
            fprintf(stderr, "FAIL aspect control %d: a spherical target's "
                    "channel moved by %.6e\n", i, moved[i]);
            exit(1);
        }
    }
    g_arms++;
    printf("  aspect control: a spherical target's three channels move "
           "by at most %.2e under the same rotation\n",
           moved[0] > moved[1] ? (moved[0] > moved[2] ? moved[0] : moved[2])
                               : (moved[1] > moved[2] ? moved[1] : moved[2]));
    close_(&box);
    close_(&ball);
}

/* ---- Gate 5a: the modality mapping ---------------------------------- */

/* The modality is recorded into the library's observation snapshot and
 * nothing publishes it, so no channel can be made to move by getting
 * it wrong. What can be checked is the mapping itself, against the
 * library's own enumeration rather than against a number restated
 * here: the emitted call must carry the value the header declares for
 * the name the program wrote. */
static void gate_modality_map_(void)
{
    static const struct { const char *name; int value; } MODS[] = {
        { "none",  (int)K26ASTRO_INFOSTATE_MODALITY_NONE  },
        { "ir",    (int)K26ASTRO_INFOSTATE_MODALITY_IR    },
        { "radar", (int)K26ASTRO_INFOSTATE_MODALITY_RADAR },
        { "lidar", (int)K26ASTRO_INFOSTATE_MODALITY_LIDAR },
        { "ephem", (int)K26ASTRO_INFOSTATE_MODALITY_EPHEM }
    };
    for (int i = 0; i < 5; i++) {
        char src[16384];
        char clause[64];
        snprintf(clause, sizeof clause,
                 "        observe track picture of mover modality=%s"
                 " as trk\n", MODS[i].name);
        snprintf(src, sizeof src, "%s%s%s%s%s%s",
            DEF_HEAD, DEF_MOVER("calibration_box.k26asm", "0.0"),
            DEF_PAYLOADS, DEF_EPISODE,
            "    agent hunter\n"
            "        action nudge box -1.0 1.0 default 0.0\n",
            clause);
        size_t used = strlen(src);
        snprintf(src + used, sizeof src - used, "%s",
            "    end\n"
            "    agent quarry\n"
            "        action dodge box -1.0 1.0 default 0.0\n"
            "        observe mover from watcher mode=geometric as los\n"
            "    end\n"
            "end\nend\n");
        rl_write_file_(WORK_DIR "/mod.kfl", src);
        int rc = system("./bin/kflc --emit " WORK_DIR "/mod.kfl > "
                        WORK_DIR "/mod.cc 2> " WORK_DIR "/mod.err");
        ASSERT(WIFEXITED(rc) && WEXITSTATUS(rc) == 0);
        char want[128];
        snprintf(want, sizeof want, "(K26AstroInfostateModality)%d)",
                 MODS[i].value);
        char cmd[512];
        snprintf(cmd, sizeof cmd, "grep -q -- '%s' " WORK_DIR "/mod.cc",
                 want);
        if (system(cmd) != 0) {
            fprintf(stderr, "FAIL: modality `%s` does not reach the "
                    "library as %d\n", MODS[i].name, MODS[i].value);
            exit(1);
        }
        g_arms++;
        printf("  modality %s reaches the library as %d, the value its "
               "header declares\n", MODS[i].name, MODS[i].value);
    }
}

/* ---- Gate 6: one generator ------------------------------------------- */

/* The tier's evaluators take an optional generator as their last
 * argument, and the design's rule is that this binding always passes
 * none: an imperfection on a defense channel goes through the sensor
 * layer, whose draws already carry the coordinates replay rests on. A
 * second generator on the same path, advanced sequentially, would have
 * none of those properties and would break the first.
 *
 * Three arms, because the rule has three ways of being broken and one
 * check catches only one of them. Each carries its own perturbation,
 * and each perturbation is asserted to have reached the file before
 * its verdict is believed. */

static char *slurp_(const char *path)
{
    FILE *f = fopen(path, "rb");
    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    long n = ftell(f);
    ASSERT(n > 0);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    char *buf = (char *)malloc((size_t)n + 1);
    ASSERT(buf != NULL);
    ASSERT(fread(buf, 1, (size_t)n, f) == (size_t)n);
    buf[n] = '\0';
    fclose(f);
    return buf;
}

static int count_(const char *hay, const char *needle)
{
    int n = 0;
    for (const char *p = strstr(hay, needle); p;
         p = strstr(p + 1, needle)) {
        n++;
    }
    return n;
}

/* The last argument of the first call to `fn` in `src`, written into
 * `out`. Returns 0 when there is no such call. The scan tracks
 * parenthesis depth, so it reads the call's own argument list rather
 * than the first comma it meets. */
static int last_arg_(const char *src, const char *fn, char *out,
                     size_t cap)
{
    char pat[128];
    snprintf(pat, sizeof pat, "%s(", fn);
    const char *p = strstr(src, pat);
    if (!p) return 0;
    p += strlen(pat);
    int depth = 1;
    const char *last = p;
    for (; *p; p++) {
        if (*p == '(') depth++;
        else if (*p == ')') { if (--depth == 0) break; }
        else if (*p == ',' && depth == 1) last = p + 1;
    }
    ASSERT(depth == 0);
    while (last < p && (*last == ' ' || *last == '\n' || *last == '\t')) {
        last++;
    }
    size_t n = (size_t)(p - last);
    if (n >= cap) n = cap - 1;
    memcpy(out, last, n);
    out[n] = '\0';
    while (n > 0 && (out[n - 1] == ' ' || out[n - 1] == '\n' ||
                     out[n - 1] == '\t')) {
        out[--n] = '\0';
    }
    return 1;
}

static const char *const TIER_EVALS[] = {
    "k26astro_detect_ir_passive_with_optics",
    "k26astro_detect_radar_active",
    "k26astro_detect_lidar_active",
    NULL
};

/* Whether the emitted source names the tier's generator anywhere. */
static int names_generator_(const char *src)
{
    return count_(src, "K26CRng") > 0 || count_(src, "k26c_rng_") > 0;
}

/* Whether every tier evaluator in the source is handed a null
 * generator. Returns the number of calls inspected through
 * `out_seen`, so an arm that inspected nothing cannot read as clean. */
static int all_null_(const char *src, int *out_seen)
{
    int seen = 0, ok = 1;
    for (int i = 0; TIER_EVALS[i]; i++) {
        char arg[128];
        if (!last_arg_(src, TIER_EVALS[i], arg, sizeof arg)) continue;
        seen++;
        if (strcmp(arg, "NULL") != 0) ok = 0;
    }
    *out_seen = seen;
    return ok;
}

/* Compile a translation unit on its own and report whether it leaves
 * an undefined reference to the tier's generator, which is what a
 * direct call would produce. */
static int refers_to_rng_(const char *cc_path, const char *obj_path)
{
    char cmd[16384];
    int n = snprintf(cmd, sizeof cmd, "c++ -O0 -std=c++11 -c -o %s %s",
                     obj_path, cc_path);
    for (int i = 0; RL_INCLUDE_DIRS_[i]; i++) {
        n += snprintf(cmd + n, sizeof cmd - (size_t)n, " -I%s",
                      RL_INCLUDE_DIRS_[i]);
    }
    n += snprintf(cmd + n, sizeof cmd - (size_t)n,
                  " > " WORK_DIR "/obj.log 2>&1");
    ASSERT((size_t)n < sizeof cmd);
    if (system(cmd) != 0) {
        (void)!system("cat " WORK_DIR "/obj.log");
        fprintf(stderr, "FAIL: could not compile %s\n", cc_path);
        exit(1);
    }
    snprintf(cmd, sizeof cmd, "nm -u %s | grep -q k26c_rng_", obj_path);
    return system(cmd) == 0;
}

static void run_or_die_(const char *cmd)
{
    int rc = system(cmd);
    if (!(WIFEXITED(rc) && WEXITSTATUS(rc) == 0)) {
        fprintf(stderr, "FAIL: command failed: %s\n", cmd);
        exit(1);
    }
}

static void gate_one_generator_(void)
{
    rl_write_file_(WORK_DIR "/gen.kfl", BOX_KFL);
    run_or_die_("./bin/kflc --emit " WORK_DIR "/gen.kfl > "
                WORK_DIR "/gen.cc 2> " WORK_DIR "/gen.err");
    char *src = slurp_(WORK_DIR "/gen.cc");

    /* Arm 1: the artifact names the tier's generator nowhere, so it
     * can neither construct one nor hold one. */
    if (names_generator_(src)) {
        fprintf(stderr, "FAIL: the artifact names the tier's generator\n");
        exit(1);
    }
    g_arms++;
    printf("  the artifact names neither K26CRng nor k26c_rng_\n");

    /* Arm 2: every tier evaluator is handed a null generator, and the
     * count of calls inspected is reported so an arm that found none
     * cannot read as an arm that found them all null. */
    int seen = 0;
    if (!all_null_(src, &seen) || seen != 3) {
        fprintf(stderr, "FAIL: %d tier call(s) inspected, expected 3, "
                "and not all pass a null generator\n", seen);
        exit(1);
    }
    g_arms++;
    printf("  all %d tier evaluator calls pass a null generator\n", seen);

    /* Arm 3: the compiled unit leaves no undefined reference to the
     * generator, which is what a direct call would produce. */
    if (refers_to_rng_(WORK_DIR "/gen.cc", WORK_DIR "/gen.o")) {
        fprintf(stderr, "FAIL: the compiled artifact refers to the "
                "tier's generator\n");
        exit(1);
    }
    g_arms++;
    printf("  the compiled artifact leaves no undefined k26c_rng_ "
           "reference\n");
    free(src);

    /* The three perturbations. Each is applied to a copy of the same
     * emitted source, asserted to have reached the file, and required
     * to be caught by the arm it is aimed at; a check that cannot fail
     * on the defect it names measures nothing. */
    run_or_die_("sed 's/^static double kflrl_sig_area_/"
                "static K26CRng _kfl_probe;\\nstatic double "
                "kflrl_sig_area_/' " WORK_DIR "/gen.cc > "
                WORK_DIR "/p1.cc");
    char *p1 = slurp_(WORK_DIR "/p1.cc");
    ASSERT(count_(p1, "_kfl_probe") == 1);
    if (!names_generator_(p1)) {
        fprintf(stderr, "FAIL: a generator held by the artifact was not "
                "caught\n");
        exit(1);
    }
    g_arms++;
    printf("  perturbation: a generator held by the artifact is caught\n");
    free(p1);

    run_or_die_("sed 's/_kfl_pp\\[7\\], _kfl_pp\\[8\\], NULL)/"
                "_kfl_pp[7], _kfl_pp[8], \\&_kfl_probe)/' "
                WORK_DIR "/gen.cc > " WORK_DIR "/p2.cc");
    char *p2 = slurp_(WORK_DIR "/p2.cc");
    ASSERT(count_(p2, "&_kfl_probe") == 1);
    if (all_null_(p2, &seen)) {
        fprintf(stderr, "FAIL: a generator threaded into a tier call was "
                "not caught\n");
        exit(1);
    }
    ASSERT(seen == 3);
    g_arms++;
    printf("  perturbation: a generator threaded into a tier call is "
           "caught\n");
    free(p2);

    run_or_die_("sed 's/^    double area = 0.0;$/"
                "    double area = 0.0;\\n    { K26CRng r; "
                "k26c_rng_init(\\&r, 1u); area += 0.0 * "
                "k26c_rng_uniform(\\&r); }/' " WORK_DIR "/gen.cc > "
                WORK_DIR "/p3.cc");
    char *p3 = slurp_(WORK_DIR "/p3.cc");
    ASSERT(count_(p3, "k26c_rng_uniform") == 1);
    free(p3);
    if (!refers_to_rng_(WORK_DIR "/p3.cc", WORK_DIR "/p3.o")) {
        fprintf(stderr, "FAIL: a generator advanced by the artifact was "
                "not caught\n");
        exit(1);
    }
    g_arms++;
    printf("  perturbation: a generator advanced by the artifact is "
           "caught\n");
}

/* ---- Gate 8: determinism --------------------------------------------- */

static void gate_determinism_(void)
{
    rl_write_file_(WORK_DIR "/det.kfl", BOX_KFL);
    rl_compile_(WORK_DIR "/det.kfl", WORK_DIR "/det", WORK_DIR);
    for (int i = 0; i < 2; i++) {
        char cmd[512];
        snprintf(cmd, sizeof cmd,
                 WORK_DIR "/det --seed 4242 --envs 2 --episodes 3 "
                 "--out " WORK_DIR "/run%d.k26ep > " WORK_DIR
                 "/run%d.log 2>&1", i, i);
        int rc = system(cmd);
        if (rc != 0) {
            (void)!system("cat " WORK_DIR "/run0.log");
            fprintf(stderr, "FAIL: batch run %d exited %d\n", i, rc);
            exit(1);
        }
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
           "(%lld bytes)\n", (long long)st.st_size);
}

int main(void)
{
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_run_or_die_("cp examples/assets/calibration_box.k26asm "
                   "examples/assets/calibration_box.k26mesh "
                   "examples/assets/crew_vehicle_10t.k26asm "
                   WORK_DIR "/");
    rl_write_file_(WORK_DIR "/scratch_ball.k26asm", BALL_ASM);

    printf("test_rl_defense: payloads, detection, and the information "
           "state\n");
    gate_refusals_();
    gate_agent_observation_();
    gate_target_cap_();

    if (!rl_libs_present_("test_rl_defense")) {
        printf("test_rl_defense: %d arm(s) passed, drive arms stood down "
               "(stack archives absent)\n", g_arms);
        return 77;
    }
    gate_validity_();
    gate_validity_long_();
    gate_validity_seed_();
    gate_aspect_();
    gate_modality_map_();
    gate_one_generator_();
    gate_determinism_();

    printf("test_rl_defense: %d arm(s) passed\n", g_arms);
    return 0;
}
