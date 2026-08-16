/* test_rl_softkill.c: the countermeasure effectors.
 *
 * `decoy` and `jammer` are the two payload kinds in the countermeasure
 * library, and they are the one effector class whose result lands on
 * another craft's payloads rather than on a body. That is what most of
 * this gate is about: an engagement here has to be visible in the
 * victim's own published detection channels, or it is a decoration
 * whatever it publishes about itself.
 *
 * Chaff is the third thing that library carries and it is not a
 * payload: four free functions, no handle, no registry tag, and no
 * bind. It arrives instead as two optional reference-target keys on the
 * radar detection payload, raising the cross-section the radar sees.
 *
 * Gates:
 *   1. Refusals. Every rule the two kinds and the chaff keys carry,
 *      written as a program that breaks it, including the kind the
 *      defense registry names and no library implements.
 *   2. Acceptance. Both kinds through compile, link, dlopen, step,
 *      reset and step, so "accepted" means an artifact exists and runs
 *      rather than that a checker did not object.
 *   3. The published channels: the names and the order of both
 *      component sets, read from the artifact's own spec blob.
 *   4. The jammer's effect on a payload's capability. The victim's own
 *      radar signal-to-noise ratio and its detection flag are compared
 *      with and without the engagement, over one seed and one action
 *      stream. Its mutation follows: with the degradation write deleted
 *      from the emitted source the victim's channels return to the
 *      control's, bit for bit, and the arm must fail.
 *   5. The decoy's, likewise, and with two victim payloads declaring
 *      different discriminator regimes, since a fixture holding one
 *      regime cannot tell a regime that is read from one that is
 *      ignored.
 *   6. The other edge. A jammer announces its own position with every
 *      watt it transmits, so the victim's infrared payload detects the
 *      emitter it could not see before: the flag moves from 0 to 1 in
 *      the engaged run and stays 0 in the control.
 *   7. The decoy's deploy is a body effect as well: a mass leaves the
 *      host at the separation velocity, so the host's velocity and its
 *      mass both move. The mass write's route to the dynamics is
 *      measured at two separations rather than assumed, and no
 *      ordering between them is asserted.
 *   8. Reach. An engagement publishes how many of the victim's
 *      detection payloads it wrote to, and the arm drives that count
 *      to a non-zero value and to zero, the second being the
 *      decoration this class has to be able to report.
 *   9. Chaff moves the radar channels, and moves them by the law it
 *      claims: the radar statistic is linear in the cross-section, so
 *      three strip counts give increments in exactly the ratio of the
 *      counts.
 *  10. And a program that declares no chaff key publishes the radar
 *      channels the compiler at the base commit published for it, bit
 *      for bit, which is what makes the addition additive rather than
 *      a change to everything.
 *  11. The degradation is an act of one step. A step whose body engaged
 *      nothing publishes zero on every countermeasure channel and
 *      leaves the victim's channels at their undegraded values, and so
 *      does the observation a reset produces.
 *  12. Every published component moves, one fixture per component per
 *      kind; a component nothing can move is a component nothing can
 *      be shaped against.
 *  13. One engagement per payload per step, the runtime half, for both
 *      kinds: a single statement reached twice through a loop faults
 *      the environment.
 *  14. The words this change adds are values and key names rather than
 *      construct words, which is measured against the compiler at the
 *      base commit rather than argued: every new word is driven
 *      through both binaries in each position an identifier can
 *      appear.
 *
 * Pattern: rl_gate_util.h. Refusal arms drive ./bin/kflc --check and
 * assert on the captured diagnostic; behaviour arms compile the
 * artifact and drive the frozen surface. The mutation arms compile the
 * emitted source directly, so they carry their own link line. Needs
 * the sibling stack archives; skips with 77 when they are absent.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_softkill_test"

/* The commit this change opened at. Arms 10 and 14 build the compiler
 * there and compare, so that "unchanged" and "still a name" are
 * measured against the compiler that was, not against a reading of the
 * source. */
#define BASE_COMMIT "5d3c298"

static int g_arms;
static int g_base_ok;          /* the base compiler built */

/* ---- Sources -------------------------------------------------------- */

/* The world every arm is built on. `watcher` carries the detection
 * payloads and `mover` carries the countermeasures, so an engagement
 * runs from the craft being looked at towards the craft looking, which
 * is the direction a countermeasure is used in.
 *
 * `mover` is the two-by-one-by-one metre box, turning about its z axis,
 * so the silhouette it presents along a fixed line is a function of
 * time: the jammer's own radar cross-section is that silhouette, so the
 * quantity the jamming ratio divides by moves as the craft turns.
 */
#define SK_EARTH \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"

#define SK_WATCHER \
    "    astro_body watcher assembly=\"crew_vehicle_10t.k26asm\"" \
    " parent=earth pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0\n"

/* The separation is a parameter because one of the two routes the
 * decoy's mass write takes to the dynamics is a function of it. */
#define SK_MOVER(sep) \
    "    astro_body mover assembly=\"calibration_box.k26asm\"" \
    " parent=earth pos_x=7.0e6 pos_y=" sep " pos_z=0.0 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.05\n"

/* A third craft carrying nothing, for the arm that drives the reach
 * count to zero. */
#define SK_BYSTANDER \
    "    astro_body bystander assembly=\"crew_vehicle_10t.k26asm\"" \
    " parent=earth pos_x=7.0e6 pos_y=0.0 pos_z=3.0e4 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0\n"

/* The radar payload exactly as it could be written before these keys,
 * so a program built from it is one the compiler at the base commit
 * can compile too. That is what the no-chaff identity arm needs. */
#define SK_RADAR_BASE(nm, extra) \
    "    astro_payload " nm " body=watcher kind=detect_radar" \
    " p_tx_w=2000.0 g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10" \
    " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0" \
    " noise_figure=2.0 snr_threshold=10.0" extra "\n"

#define SK_RADAR(nm, regime, extra) \
    SK_RADAR_BASE(nm, " discriminator_regime=" regime extra)

/* A telescope that sees the target's own warmth easily. */
#define SK_IR_WARM(nm) \
    "    astro_payload " nm " body=watcher kind=detect_ir" \
    " aperture_m=1.0 integration_s=0.5 passband_lo_um=3.0" \
    " passband_hi_um=12.0 throughput=0.5 snr_threshold=5.0" \
    " target_temp_k=300.0 target_emissivity=0.9 t_optics_k=280.0" \
    " optics_emissivity=0.05\n"

/* A small, briefly-dwelling telescope against a cold, dark target: the
 * skin signature is far below the threshold, so this instrument sees
 * nothing at all until the target transmits. */
#define SK_IR_COLD(nm) \
    "    astro_payload " nm " body=watcher kind=detect_ir" \
    " aperture_m=0.05 integration_s=0.02 passband_lo_um=3.0" \
    " passband_hi_um=12.0 throughput=0.5 snr_threshold=5.0" \
    " target_temp_k=90.0 target_emissivity=0.02 t_optics_k=290.0" \
    " optics_emissivity=0.3\n"

#define SK_JAMMER(nm, power) \
    "    astro_payload " nm " body=mover kind=jammer mode=noise" \
    " p_j_w=" power " g_j_db=10.0 freq_hz=1.0e10 bandwidth_hz=1.0e6" \
    " snr_threshold=10.0 radiator_temp_k=320.0\n"

#define SK_DECOY(nm, mode, irq) \
    "    astro_payload " nm " body=mover kind=decoy mode=" mode \
    " dry_mass_kg=5.0 deploy_dv_mps=2.0 ir_match_quality=" irq \
    " rcs_match_quality=0.7 accel_match_quality=0.1\n"

#define SK_EPISODE \
    "    episode\n" \
    "        control_dt 0.5\n" \
    "        substeps 4\n" \
    "        horizon 32\n" \
    "    end\n"

#define SK_ACTION "    action nudge box -1.0 1.0 default 0.0\n"

/* ---- Plumbing -------------------------------------------------------- */

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

static void must_accept_check_(const char *what, const char *src)
{
    char *log = NULL;
    int rc = check_(src, &log);
    if (rc != 0) {
        fprintf(stderr, "FAIL %s: refused by the checker\n---\n%s---\n",
                what, log);
        exit(1);
    }
    g_arms++;
    printf("  accepted by the checker: %s\n", what);
}

typedef struct {
    void      *so;
    RlSurface  s;
    K26RlEnv  *env;
} SkArt;

static void art_open_(SkArt *a, const char *so_path, uint64_t seed,
                      uint32_t n_envs)
{
    a->so = rl_dlopen_(so_path);
    rl_resolve_surface_(a->so, &a->s);
    a->env = NULL;
    ASSERT(a->s.create(seed, n_envs, &a->env) == K26RL_OK);
}

static void art_close_(SkArt *a)
{
    a->s.destroy(a->env);
    dlclose(a->so);
}

static void build_(const char *src, const char *stem)
{
    char path[512], out[512];
    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    rl_write_file_(path, src);
    rl_compile_(path, out, WORK_DIR);
}

static void so_path_(char *buf, size_t cap, const char *stem)
{
    snprintf(buf, cap, WORK_DIR "/%s.rlenv.so", stem);
}

static void build_emitted_(const char *cc_path, const char *so_path)
{
    char cmd[16384];
    int n = snprintf(cmd, sizeof cmd,
        "c++ -O2 -g -std=c++11 -Wno-format-truncation -ffp-contract=off "
        "-fexcess-precision=standard -fPIC -shared -o %s %s",
        so_path, cc_path);
    for (int i = 0; RL_INCLUDE_DIRS_[i]; i++) {
        n += snprintf(cmd + n, sizeof cmd - (size_t)n, " -I%s",
                      RL_INCLUDE_DIRS_[i]);
    }
    for (int i = 0; RL_LINK_LIBS_[i]; i++) {
        n += snprintf(cmd + n, sizeof cmd - (size_t)n, " %s",
                      RL_LINK_LIBS_[i]);
    }
    n += snprintf(cmd + n, sizeof cmd - (size_t)n,
                  " -lgfortran -lm > " WORK_DIR "/mut.log 2>&1");
    ASSERT((size_t)n < sizeof cmd);
    if (system(cmd) != 0) {
        (void)!system("cat " WORK_DIR "/mut.log");
        fprintf(stderr, "FAIL: could not build the perturbed source %s\n",
                cc_path);
        exit(1);
    }
}

static void emit_with_(const char *kflc, const char *stem)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "%s --emit " WORK_DIR "/%s.kfl > " WORK_DIR
             "/%s.cc 2> " WORK_DIR "/%s.emit.log", kflc, stem, stem, stem);
    if (system(cmd) != 0) {
        char show[512];
        snprintf(show, sizeof show, "cat " WORK_DIR "/%s.emit.log", stem);
        (void)!system(show);
        fprintf(stderr, "FAIL: --emit failed for %s\n", stem);
        exit(1);
    }
}

static void emit_(const char *stem) { emit_with_("./bin/kflc", stem); }

/* Apply a sed script to an emitted source and assert that it changed
 * the file. A mutation that never reached the file cannot be reported
 * at all, and one that never reached the build cannot be reported as a
 * survivor. */
static void mutate_(const char *stem, const char *out_stem,
                    const char *sed_script, const char *needle,
                    int before, int after)
{
    char cmd[4096];
    snprintf(cmd, sizeof cmd, "sed '%s' " WORK_DIR "/%s.cc > " WORK_DIR
             "/%s.cc", sed_script, stem, out_stem);
    rl_run_or_die_(cmd);
    snprintf(cmd, sizeof cmd,
             "grep -c -- '%s' " WORK_DIR "/%s.cc > " WORK_DIR "/n0.txt; "
             "grep -c -- '%s' " WORK_DIR "/%s.cc > " WORK_DIR "/n1.txt",
             needle, stem, needle, out_stem);
    (void)!system(cmd);
    FILE *f0 = fopen(WORK_DIR "/n0.txt", "rb");
    FILE *f1 = fopen(WORK_DIR "/n1.txt", "rb");
    ASSERT(f0 && f1);
    int n0 = 0, n1 = 0;
    ASSERT(fscanf(f0, "%d", &n0) == 1);
    ASSERT(fscanf(f1, "%d", &n1) == 1);
    fclose(f0);
    fclose(f1);
    if (n0 != before || n1 != after) {
        fprintf(stderr, "FAIL: mutation of `%s` did not reach the file: "
                "%d occurrence(s) before and %d after, expected %d and "
                "%d\n", needle, n0, n1, before, after);
        exit(1);
    }
}

/* ---- Reading an artifact --------------------------------------------- */

static void spec_of_(const char *so_path, RlSpecView *v)
{
    SkArt a;
    art_open_(&a, so_path, 4242u, 1u);
    uint8_t blob[65536];
    int32_t n = a.s.spec(a.env, blob, (uint32_t)sizeof blob);
    ASSERT(n > 0 && (uint32_t)n <= sizeof blob);
    rl_parse_spec_(blob, (uint32_t)n, v);
    art_close_(&a);
}

static int chan_(const RlSpecView *v, const char *name)
{
    for (int i = 0; i < v->n_chan_names; i++) {
        if (strcmp(v->chan_names[i], name) == 0) return i;
    }
    fprintf(stderr, "FAIL: no published channel named `%s`\n", name);
    exit(1);
}

/* Step an artifact `n` times with a constant action and return the
 * whole observation vector of the last step. */
static void run_obs_(const char *so_path, int n, double act0, double *out,
                     int width)
{
    SkArt a;
    art_open_(&a, so_path, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    act[0] = act0;
    double o[256];
    for (int i = 0; i < n; i++) ASSERT(a.s.step(a.env, act) == K26RL_OK);
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    for (int i = 0; i < width; i++) out[i] = o[i];
    art_close_(&a);
}

/* The observation a fresh handle's reset produces, before any step. */
static void reset_obs_(const char *so_path, double *out, int width)
{
    SkArt a;
    art_open_(&a, so_path, 4242u, 1u);
    ASSERT(a.s.reset(a.env) == K26RL_OK);
    double o[256];
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    for (int i = 0; i < width; i++) out[i] = o[i];
    art_close_(&a);
}

/* Body state after `n` steps. Body order is declaration order, so
 * `mover` is index 2 in every fixture that declares three bodies. */
static void run_body_(const char *so_path, int n, int body, double out6[6])
{
    SkArt a;
    art_open_(&a, so_path, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < n; i++) ASSERT(a.s.step(a.env, act) == K26RL_OK);
    double b[64];
    int32_t need = a.s.bodies(a.env, 0u, b, 64u);
    ASSERT(need >= (body + 1) * 6);
    memcpy(out6, b + body * 6, sizeof(double) * 6);
    art_close_(&a);
}

static double dist6_(const double a[6], const double b[6])
{
    double s = 0.0;
    for (int i = 0; i < 6; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(s);
}

/* ---- Gate 1: the refusals -------------------------------------------- */

static void gate_refusals_(void)
{
    char src[8192];

    /* The kind the registry names and no library implements. The
     * message has to say that rather than "unknown kind": a reader who
     * found the tag in the registry header did not misspell it. */
    snprintf(src, sizeof src,
        "form SKR\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload glare body=mover kind=dazzler\n"
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "kind `dazzler`",
                                  "K26ASTRO_DEFENSE_KIND_DAZZLER",
                                  "no library in this tree implements it",
                                  NULL };
        must_refuse_("kind=dazzler, which the registry names and nothing "
                     "implements", src, n);
    }

    /* And a kind that is not in the registry at all still gets the
     * message for a misspelling, so the arm above is measuring the
     * unimplemented path rather than the general one. */
    snprintf(src, sizeof src,
        "form SKR2\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload glare body=mover kind=blinder\n"
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "unknown kind `blinder`", NULL };
        must_refuse_("an unregistered kind is still unknown", src, n);
    }

    /* A required key of each new kind. */
    snprintf(src, sizeof src,
        "form SKR3\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload jam body=mover kind=jammer mode=noise"
        " p_j_w=200.0 g_j_db=10.0 freq_hz=1.0e10 bandwidth_hz=1.0e6"
        " snr_threshold=10.0\n"
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "kind `jammer` requires "
                                  "`radiator_temp_k=`", NULL };
        must_refuse_("a jammer with no radiator temperature", src, n);
    }

    snprintf(src, sizeof src,
        "form SKR4\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload flare body=mover kind=decoy mode=passive"
        " dry_mass_kg=5.0 ir_match_quality=0.5 rcs_match_quality=0.5"
        " accel_match_quality=0.5\n"
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "kind `decoy` requires "
                                  "`deploy_dv_mps=`", NULL };
        must_refuse_("a decoy with no separation velocity", src, n);
    }

    /* A word key given a word it does not take, one arm per kind. */
    snprintf(src, sizeof src,
        "form SKR5\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_JAMMER("jam", "200.0")
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        char bad[8192];
        char *p;
        snprintf(bad, sizeof bad, "%s", src);
        p = strstr(bad, "mode=noise");
        ASSERT(p != NULL);
        memcpy(p, "mode=barra", 10);
        const char *const n[] = { "`mode=barra` is not one of the words",
                                  "noise, cover_pulse, deception", NULL };
        must_refuse_("a jammer mode that is not one of the library's",
                     bad, n);
    }

    snprintf(src, sizeof src,
        "form SKR6\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_DECOY("flare", "warm", "0.8")
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "`mode=warm` is not one of the words",
                                  "passive, active", NULL };
        must_refuse_("a decoy mode that is not one of the library's",
                     src, n);
    }

    /* A number where a word belongs, which would be a program reading
     * the library's internal numbering. */
    snprintf(src, sizeof src,
        "form SKR7\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload jam body=mover kind=jammer mode=1"
        " p_j_w=200.0 g_j_db=10.0 freq_hz=1.0e10 bandwidth_hz=1.0e6"
        " snr_threshold=10.0 radiator_temp_k=320.0\n"
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "is not one of the words this key takes",
                                  NULL };
        must_refuse_("a number on the jammer's mode", src, n);
    }

    /* An unknown discriminator regime on a detection payload. */
    snprintf(src, sizeof src,
        "form SKR8\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_RADAR("rf", "ir_and_everything", "")
        SK_EPISODE SK_ACTION
        "    observe detect rf of mover as radar\n"
        "    objective\n        reward radar_snr\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "is not one of the words this key takes",
                                  "ir_only, ir_plus_rcs, ir_rcs_accel",
                                  NULL };
        must_refuse_("a discriminator regime the library does not define",
                     src, n);
    }

    /* The chaff pair, in the one direction that is refused. */
    snprintf(src, sizeof src,
        "form SKR9\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_RADAR("rf", "ir_only", " target_chaff_sigma_dipole_m2=2.0e-3")
        SK_EPISODE SK_ACTION
        "    observe detect rf of mover as radar\n"
        "    objective\n        reward radar_snr\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "target_chaff_sigma_dipole_m2",
                                  "target_chaff_n_strips",
                                  "nothing would read it", NULL };
        must_refuse_("a dipole cross-section with no strip count", src, n);
    }

    /* And the other direction is admitted, taking the library's own
     * default for the dipole: a fixture that only refused could not
     * tell a pair rule from a rule that refuses the key outright. */
    snprintf(src, sizeof src,
        "form SKR10\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_RADAR("rf", "ir_only", " target_chaff_n_strips=100000.0")
        SK_EPISODE SK_ACTION
        "    observe detect rf of mover as radar\n"
        "    objective\n        reward radar_snr\n    end\n"
        "end\nend\n");
    must_accept_check_("a strip count with no dipole cross-section", src);

    /* A chaff key on a detection kind whose model has no cross-section
     * in it, which names the kind the key belongs to. */
    snprintf(src, sizeof src,
        "form SKR11\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload eye body=watcher kind=detect_ir"
        " aperture_m=1.0 integration_s=0.5 passband_lo_um=3.0"
        " passband_hi_um=12.0 throughput=0.5 snr_threshold=5.0"
        " target_temp_k=300.0 target_emissivity=0.9"
        " target_chaff_n_strips=100.0\n"
        SK_EPISODE SK_ACTION
        "    observe detect eye of mover as ir\n"
        "    objective\n        reward ir_snr\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "belongs to kind `detect_radar`", NULL };
        must_refuse_("a chaff key on the infrared kind", src, n);
    }

    /* A countermeasure engaged at its own platform. */
    snprintf(src, sizeof src,
        "form SKR12\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_JAMMER("jam", "200.0")
        SK_EPISODE SK_ACTION
        "    observe effect jam as ew\n"
        "    on_step\n        engage jam at mover\n    end\n"
        "    objective\n        reward ew_effect\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "an effector does not engage its own "
                                  "platform", NULL };
        must_refuse_("a jammer engaged at the craft that carries it",
                     src, n);
    }

    /* Two engagements of one countermeasure in one step body, which is
     * the compile-time half of the one-per-step rule. */
    snprintf(src, sizeof src,
        "form SKR13\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_DECOY("flare", "active", "0.8")
        SK_EPISODE SK_ACTION
        "    observe effect flare as dec\n"
        "    on_step\n"
        "        engage flare at watcher\n"
        "        engage flare at watcher\n"
        "    end\n"
        "    objective\n        reward dec_effect\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "is already engaged at line",
                                  "engaged at most once", NULL };
        must_refuse_("a decoy engaged twice in one step body", src, n);
    }

    /* A jammer whose host has no silhouette. The jamming ratio divides
     * by the protected craft's radar cross-section, so the host needs
     * a collision primitive even though the victim does not. */
    snprintf(src, sizeof src,
        "form SKR14\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER
        "    astro_body mover assembly=\"scratch_bare.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=2.0e4 pos_z=0.0 vel_x=0.0"
        " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0"
        " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0\n"
        SK_JAMMER("jam", "200.0")
        SK_EPISODE SK_ACTION
        "    observe effect jam as ew\n"
        "    on_step\n        engage jam at watcher\n    end\n"
        "    objective\n        reward ew_effect\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "engage of a jammer carried by `mover`",
                                  "declares no `collider`", NULL };
        must_refuse_("a jammer on a craft with no silhouette", src, n);
    }

    /* And a decoy on the same craft is admitted, because a decoy takes
     * no area of anything: a fixture that refused both could not tell
     * the rule from a rule about countermeasures in general. */
    snprintf(src, sizeof src,
        "form SKR15\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER
        "    astro_body mover assembly=\"scratch_bare.k26asm\""
        " parent=earth pos_x=7.0e6 pos_y=2.0e4 pos_z=0.0 vel_x=0.0"
        " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0"
        " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0\n"
        SK_DECOY("flare", "active", "0.8")
        SK_EPISODE SK_ACTION
        "    observe effect flare as dec\n"
        "    on_step\n        engage flare at watcher\n    end\n"
        "    objective\n        reward dec_effect\n    end\n"
        "end\nend\n");
    must_accept_check_("a decoy on a craft with no silhouette", src);

    /* `engage` at a body carrying no vehicle, whose reason for a
     * countermeasure is that there is no payload to reach. */
    snprintf(src, sizeof src,
        "form SKR16\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_JAMMER("jam", "200.0")
        SK_EPISODE SK_ACTION
        "    observe effect jam as ew\n"
        "    on_step\n        engage jam at earth\n    end\n"
        "    objective\n        reward ew_effect\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "no payload for a countermeasure to "
                                  "reach", NULL };
        must_refuse_("a countermeasure engaged at a body with no vehicle",
                     src, n);
    }

    /* A distribution on a word-valued key. */
    snprintf(src, sizeof src,
        "form SKR17\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        "    astro_payload flare body=mover kind=decoy"
        " mode=uniform(1.0, 2.0) dry_mass_kg=5.0 deploy_dv_mps=2.0"
        " ir_match_quality=0.8 rcs_match_quality=0.7"
        " accel_match_quality=0.1\n"
        SK_EPISODE SK_ACTION
        "    observe mover from watcher mode=geometric as los\n"
        "    objective\n        reward los_range\n    end\n"
        "end\nend\n");
    {
        const char *const n[] = { "is not one of the words this key takes",
                                  NULL };
        must_refuse_("a distribution on the decoy's mode", src, n);
    }
}

/* ---- The fixtures the behaviour arms drive --------------------------- */

/* Both countermeasures engaged at the watcher, whose radar and warm
 * telescope are the payloads they reach. The two radars declare
 * different discriminator regimes, so the decoy's derate differs
 * between them and a regime that was ignored could not agree with one
 * that is read. */
static const char *const ENGAGED_KFL =
    "form SKENG\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_RADAR("rf2", "ir_rcs_accel", "")
    SK_IR_WARM("eye")
    SK_JAMMER("jam", "200.0")
    SK_DECOY("flare", "active", "0.8")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe detect rf2 of mover as radar2\n"
    "    observe detect eye of mover as ir\n"
    "    observe effect jam as ew\n"
    "    observe effect flare as dec\n"
    "    on_step\n"
    "        engage jam at watcher\n"
    "        engage flare at watcher\n"
    "    end\n"
    "    objective\n"
    "        reward radar_snr + ew_effect + dec_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* The matched control: the same world, the same craft, the same
 * payloads, the same period, and no engagement. Its step body writes a
 * body state key scaled by zero, so it has a body of the same shape
 * without moving anything. */
static const char *const CONTROL_KFL =
    "form SKCTL\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_RADAR("rf2", "ir_rcs_accel", "")
    SK_IR_WARM("eye")
    SK_JAMMER("jam", "200.0")
    SK_DECOY("flare", "active", "0.8")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe detect rf2 of mover as radar2\n"
    "    observe detect eye of mover as ir\n"
    "    observe effect jam as ew\n"
    "    observe effect flare as dec\n"
    "    on_step\n"
    "        watcher.vel_x = watcher.vel_x + nudge * 0.0\n"
    "    end\n"
    "    objective\n"
    "        reward radar_snr\n"
    "    end\n"
    "end\n"
    "end\n";

/* The jammer alone, so the arm that measures jamming is not measuring
 * the decoy beside it. */
static const char *const JAM_ONLY_KFL =
    "form SKJAM\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_JAMMER("jam", "200.0")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe effect jam as ew\n"
    "    on_step\n        engage jam at watcher\n    end\n"
    "    objective\n        reward radar_snr + ew_effect\n    end\n"
    "end\n"
    "end\n";

static const char *const JAM_CTRL_KFL =
    "form SKJAMC\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_JAMMER("jam", "200.0")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe effect jam as ew\n"
    "    on_step\n"
    "        watcher.vel_x = watcher.vel_x + nudge * 0.0\n"
    "    end\n"
    "    objective\n        reward radar_snr\n    end\n"
    "end\n"
    "end\n";

/* The decoy alone, against two radars of different regimes. */
static const char *const DEC_ONLY_KFL =
    "form SKDEC\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_RADAR("rf2", "ir_rcs_accel", "")
    SK_DECOY("flare", "active", "0.8")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe detect rf2 of mover as radar2\n"
    "    observe effect flare as dec\n"
    "    on_step\n        engage flare at watcher\n    end\n"
    "    objective\n        reward radar_snr + dec_effect\n    end\n"
    "end\n"
    "end\n";

/* Two decoys against one observer, otherwise the same world. A fixture
 * holding one decoy cannot tell the composition the library's own
 * discrimination model uses from a plain assignment: with one of a
 * thing the two agree. */
static const char *const DEC_TWO_KFL =
    "form SKDEC2\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_DECOY("flare", "active", "0.8")
    SK_DECOY("flare2", "active", "0.8")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe effect flare as dec\n"
    "    on_step\n"
    "        engage flare at watcher\n"
    "        engage flare2 at watcher\n"
    "    end\n"
    "    objective\n        reward radar_snr + dec_effect\n    end\n"
    "end\n"
    "end\n";

static const char *const DEC_CTRL_KFL =
    "form SKDECC\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_RADAR("rf2", "ir_rcs_accel", "")
    SK_DECOY("flare", "active", "0.8")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe detect rf2 of mover as radar2\n"
    "    observe effect flare as dec\n"
    "    on_step\n"
    "        watcher.vel_x = watcher.vel_x + nudge * 0.0\n"
    "    end\n"
    "    objective\n        reward radar_snr\n    end\n"
    "end\n"
    "end\n";

/* The counter-detection pair: a cold, dark target and a small
 * telescope, so the skin signature is far below the threshold and the
 * only thing that can raise it is the emitter's own transmission. */
#define CTR_KFL(form_name, power) \
    "form " form_name "\n" \
    "fn world w\n" \
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4") \
    SK_IR_COLD("eye") \
    SK_JAMMER("jam", power) \
    SK_EPISODE SK_ACTION \
    "    observe detect eye of mover as ir\n" \
    "    observe effect jam as ew\n" \
    "    on_step\n        engage jam at watcher\n    end\n" \
    "    objective\n        reward ir_snr + ew_effect\n    end\n" \
    "end\n" \
    "end\n"

static const char *const CTR_ON_KFL = CTR_KFL("SKCTR", "200.0");

/* The same emitter at a ten-thousandth of a watt, whose
 * counter-detection range falls inside the separation: the signal is
 * present at the victim and below its threshold. */
static const char *const CTR_QUIET_KFL = CTR_KFL("SKCTRQ", "1.0e-4");

static const char *const CTR_OFF_KFL =
    "form SKCTRC\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_IR_COLD("eye")
    SK_JAMMER("jam", "200.0")
    SK_EPISODE SK_ACTION
    "    observe detect eye of mover as ir\n"
    "    observe effect jam as ew\n"
    "    on_step\n"
    "        watcher.vel_x = watcher.vel_x + nudge * 0.0\n"
    "    end\n"
    "    objective\n        reward ir_snr\n    end\n"
    "end\n"
    "end\n";

/* A jammer aimed at a craft that carries no detection payload at all,
 * which is the decoration this class has to be able to report. */
static const char *const REACH0_KFL =
    "form SKREACH0\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4") SK_BYSTANDER
    SK_RADAR("rf", "ir_only", "")
    SK_JAMMER("jam", "200.0")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe effect jam as ew\n"
    "    on_step\n        engage jam at bystander\n    end\n"
    "    objective\n        reward ew_effect\n    end\n"
    "end\n"
    "end\n";

/* The chaff fixtures, three strip counts over one otherwise identical
 * world. No countermeasure is declared: chaff is not one, and mixing
 * the two would leave the arm unable to say which moved what. */
#define CHAFF_KFL(form_name, chaff) \
    "form " form_name "\n" \
    "fn world w\n" \
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4") \
    SK_RADAR_BASE("rf", chaff) \
    SK_EPISODE SK_ACTION \
    "    observe detect rf of mover as radar\n" \
    "    on_step\n" \
    "        watcher.vel_x = watcher.vel_x + nudge * 0.0\n" \
    "    end\n" \
    "    objective\n        reward radar_snr\n    end\n" \
    "end\n" \
    "end\n"

static const char *const CHAFF_NONE_KFL = CHAFF_KFL("SKCH0", "");
static const char *const CHAFF_1X_KFL =
    CHAFF_KFL("SKCH1", " target_chaff_n_strips=2.0e7"
                       " target_chaff_sigma_dipole_m2=1.0e-3");
static const char *const CHAFF_2X_KFL =
    CHAFF_KFL("SKCH2", " target_chaff_n_strips=4.0e7"
                       " target_chaff_sigma_dipole_m2=1.0e-3");

/* An assembly with a mesh and no collision primitive, for the arm that
 * refuses a jammer host with no silhouette. */
static const char *const BARE_ASM =
    "# scratch_bare.k26asm - geometry without a collider.\n"
    "#\n"
    "# Not a craft. It exists so a body that presents no area can be\n"
    "# written down: mass properties derive from the mesh and no\n"
    "# collision primitive is declared.\n"
    "assembly scratch_bare\n"
    "    frame x_to_port\n"
    "    provenance mass \"the shape's own definition\" computed\n"
    "    provenance inertia \"derived from the geometry by the"
    " compiler\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0.0 0.0 0.0\n"
    "        mesh calibration_box.k26mesh\n"
    "    end\n"
    "end\n";

/* ---- Gate 2: acceptance reaches an artifact -------------------------- */

static void must_build_and_step_(const char *what, const char *src,
                                 const char *stem)
{
    char so[512];
    build_(src, stem);
    so_path_(so, sizeof so, stem);
    SkArt a;
    art_open_(&a, so, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    ASSERT(a.s.step(a.env, act) == K26RL_OK);
    ASSERT(a.s.reset(a.env) == K26RL_OK);
    ASSERT(a.s.step(a.env, act) == K26RL_OK);
    art_close_(&a);
    g_arms++;
    printf("  accepted, built and stepped: %s\n", what);
}

static void gate_acceptance_(void)
{
    must_build_and_step_("the jammer", JAM_ONLY_KFL, "jam");
    must_build_and_step_("the decoy", DEC_ONLY_KFL, "dec");
    must_build_and_step_("both together", ENGAGED_KFL, "eng");
}

/* ---- Gate 3: the published channels ---------------------------------- */

static void gate_channels_(void)
{
    static const char *const JAM_C[] = {
        "ew_engaged", "ew_effect", "ew_reached", "ew_range", "ew_rcs",
        "ew_burn_through", "ew_self_signature", "ew_counter_range",
        "ew_counter_detected", NULL
    };
    static const char *const DEC_C[] = {
        "dec_engaged", "dec_effect", "dec_reached", "dec_p_discriminated",
        "dec_range", "dec_dv", "dec_mass_loss", NULL
    };
    char so[512];
    RlSpecView v;

    so_path_(so, sizeof so, "jam");
    spec_of_(so, &v);
    int base = chan_(&v, "ew_engaged");
    for (int i = 0; JAM_C[i]; i++) {
        if (strcmp(v.chan_names[base + i], JAM_C[i]) != 0) {
            fprintf(stderr, "FAIL: jammer channel %d is `%s`, expected "
                    "`%s`\n", i, v.chan_names[base + i], JAM_C[i]);
            exit(1);
        }
    }
    g_arms++;
    printf("  the jammer publishes 9 components in order\n");

    so_path_(so, sizeof so, "dec");
    spec_of_(so, &v);
    base = chan_(&v, "dec_engaged");
    for (int i = 0; DEC_C[i]; i++) {
        if (strcmp(v.chan_names[base + i], DEC_C[i]) != 0) {
            fprintf(stderr, "FAIL: decoy channel %d is `%s`, expected "
                    "`%s`\n", i, v.chan_names[base + i], DEC_C[i]);
            exit(1);
        }
    }
    g_arms++;
    printf("  the decoy publishes 7 components in order\n");

    /* And the checker's own name table agrees with the emitter's. The
     * two live in different files and the arms above read the spec blob,
     * which only the emitter writes, so a drift between them is
     * invisible to those arms and visible here: a program whose
     * objective reads every published component is accepted only if the
     * checker knows all sixteen names. */
    {
        char src[16384];
        int n = snprintf(src, sizeof src,
            "form SKNAMES\n"
            "fn world w\n"
            SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
            SK_RADAR("rf", "ir_only", "")
            SK_JAMMER("jam", "200.0")
            SK_DECOY("flare", "active", "0.8")
            SK_EPISODE SK_ACTION
            "    observe detect rf of mover as radar\n"
            "    observe effect jam as ew\n"
            "    observe effect flare as dec\n"
            "    on_step\n"
            "        engage jam at watcher\n"
            "        engage flare at watcher\n"
            "    end\n"
            "    objective\n"
            "        reward 0.0");
        for (int i = 0; JAM_C[i]; i++) {
            n += snprintf(src + n, sizeof src - (size_t)n, " + %s",
                          JAM_C[i]);
        }
        for (int i = 0; DEC_C[i]; i++) {
            n += snprintf(src + n, sizeof src - (size_t)n, " + %s",
                          DEC_C[i]);
        }
        n += snprintf(src + n, sizeof src - (size_t)n,
                      "\n    end\n"
                      "end\n"
                      "end\n");
        ASSERT((size_t)n < sizeof src);
        must_accept_check_("an objective reading all 16 published "
                           "components of the two kinds", src);
    }
}

/* ---- Gate 4: the jammer changes a payload's capability --------------- */

/* The victim's own two channels, read from the artifact after `n`
 * steps. The quantity is the detection payload's, not the effector's:
 * an effector's own channel moving says only that the effector ran. */
static void victim_radar_(const char *so_path, const char *stem, int n,
                          double *snr, double *det)
{
    char path[512];
    RlSpecView v;
    snprintf(path, sizeof path, "%s", so_path);
    spec_of_(path, &v);
    int i_snr = chan_(&v, "radar_snr");
    int i_det = chan_(&v, "radar_detected");
    double o[256];
    run_obs_(path, n, 0.0, o, 256);
    *snr = o[i_snr];
    *det = o[i_det];
    (void)stem;
}

static void gate_jammer_capability_(void)
{
    char so_on[512], so_off[512];
    build_(JAM_ONLY_KFL, "jamon");
    build_(JAM_CTRL_KFL, "jamoff");
    so_path_(so_on, sizeof so_on, "jamon");
    so_path_(so_off, sizeof so_off, "jamoff");

    double s_on, d_on, s_off, d_off;
    victim_radar_(so_on, "jamon", 3, &s_on, &d_on);
    victim_radar_(so_off, "jamoff", 3, &s_off, &d_off);

    if (!(s_on < s_off)) {
        fprintf(stderr, "FAIL: jamming did not lower the victim's "
                "statistic: %.10g engaged against %.10g control\n",
                s_on, s_off);
        exit(1);
    }
    if (!(d_off == 1.0 && d_on == 0.0)) {
        fprintf(stderr, "FAIL: the victim's detection flag did not move: "
                "%.1f engaged against %.1f control\n", d_on, d_off);
        exit(1);
    }
    g_arms++;
    printf("  jamming moves the victim's own radar statistic from "
           "%.10g to %.10g and its flag from %.0f to %.0f\n",
           s_off, s_on, d_off, d_on);

    /* The mutation. With the degradation write deleted the engagement
     * still runs and still publishes its own channels, and the victim
     * sees nothing: the arm above must fail on it. */
    emit_("jamon");
    mutate_("jamon", "jamon_mut",
            "s/eng->deg\\[0 \\* KFLRL_DEG_STRIDE + 2\\].js += [^;]*;//",
            "\\.js += ", 1, 0);
    char so_mut[512];
    snprintf(so_mut, sizeof so_mut, WORK_DIR "/jamon_mut.so");
    build_emitted_(WORK_DIR "/jamon_mut.cc", so_mut);

    RlSpecView v;
    spec_of_(so_on, &v);
    int i_snr = chan_(&v, "radar_snr");
    double a[256], b[256];
    run_obs_(so_mut, 3, 0.0, a, 256);
    run_obs_(so_off, 3, 0.0, b, 256);
    if (a[i_snr] != b[i_snr]) {
        fprintf(stderr, "FAIL: with the degradation write deleted the "
                "victim's statistic is %.17g and the control's %.17g; "
                "the arm above is measuring something else\n",
                a[i_snr], b[i_snr]);
        exit(1);
    }
    g_arms++;
    printf("  and with that write deleted the victim's statistic is the "
           "control's bit for bit (%.17g)\n", a[i_snr]);
}

/* ---- Gate 5: the decoy changes a payload's capability ---------------- */

static void gate_decoy_capability_(void)
{
    char so_on[512], so_off[512];
    build_(DEC_ONLY_KFL, "decon");
    build_(DEC_CTRL_KFL, "decoff");
    so_path_(so_on, sizeof so_on, "decon");
    so_path_(so_off, sizeof so_off, "decoff");

    RlSpecView v;
    spec_of_(so_on, &v);
    int i1 = chan_(&v, "radar_snr");
    int i2 = chan_(&v, "radar2_snr");
    double on[256], off[256];
    run_obs_(so_on, 3, 0.0, on, 256);
    run_obs_(so_off, 3, 0.0, off, 256);

    if (!(on[i1] < off[i1]) || !(on[i2] < off[i2])) {
        fprintf(stderr, "FAIL: the decoy did not lower both victims: "
                "%.10g/%.10g engaged against %.10g/%.10g control\n",
                on[i1], on[i2], off[i1], off[i2]);
        exit(1);
    }
    /* The two victims declare different discriminator regimes, so the
     * ratios they are derated by must differ. Equal ratios would be a
     * regime nothing reads. */
    double r1 = on[i1] / off[i1];
    double r2 = on[i2] / off[i2];
    if (!(r2 > r1)) {
        fprintf(stderr, "FAIL: the two regimes gave derates %.10g and "
                "%.10g; the stronger discriminator must lose less\n",
                r1, r2);
        exit(1);
    }
    g_arms++;
    printf("  the decoy derates the two victims by %.10g and %.10g, the "
           "regime deciding which\n", r1, r2);

    /* Two decoys against one observer. The observer has to see through
     * both, so the probabilities of not seeing through each multiply,
     * and the statistic falls by the second factor again. A fixture
     * holding one decoy cannot tell that composition from a plain
     * assignment of the last one's degradation.
     *
     * It runs before the mutation below rather than after it, so that a
     * defect in the composition is caught by an arm that measures it
     * rather than by the mutation harness failing to find its needle in
     * a line the defect has already changed. */
    build_(DEC_TWO_KFL, "dectwo");
    char so_two[512];
    so_path_(so_two, sizeof so_two, "dectwo");
    RlSpecView vt;
    spec_of_(so_two, &vt);
    int j1 = chan_(&vt, "radar_snr");
    double two[256];
    run_obs_(so_two, 3, 0.0, two, 256);
    double one_ratio = on[i1] / off[i1];
    double two_ratio = two[j1] / off[i1];
    if (!(two_ratio < one_ratio * 0.02)) {
        fprintf(stderr, "FAIL: two decoys derated by %.10g against one "
                "decoy's %.10g; the second is not composing\n",
                two_ratio, one_ratio);
        exit(1);
    }
    g_arms++;
    printf("  two decoys derate by %.10g where one derates by %.10g, "
           "which is the composition and not the last one\n",
           two_ratio, one_ratio);

    /* The mutation. Deleting the derate write cannot be judged against
     * the control bit for bit the way the jammer's was: a decoy also
     * deploys, and the deploy moves the host, so the geometry the two
     * programs are evaluated at differs by that much. What it can be
     * judged against is the two victims themselves. They differ in the
     * declared regime and in nothing else, so undegraded they are
     * bit-identical and degraded they are not; the mutant must put them
     * back together exactly, and must leave each of them within the
     * deploy's own perturbation of the control rather than at the
     * hundredfold and twofold derates above. */
    /* The perturbation targets the statement rather than its exact
     * text, so a defect that has already changed that line does not
     * leave this harness reporting a missing needle instead of the
     * arm above reporting the defect. */
    emit_("decon");
    mutate_("decon", "decon_mut",
            "s/^\\( *\\)\\*_kfl_sl = .*;$//",
            "^ *\\*_kfl_sl = ", 2, 0);
    char so_mut[512];
    snprintf(so_mut, sizeof so_mut, WORK_DIR "/decon_mut.so");
    build_emitted_(WORK_DIR "/decon_mut.cc", so_mut);
    double mut[256];
    run_obs_(so_mut, 3, 0.0, mut, 256);
    if (!(on[i1] != on[i2])) {
        fprintf(stderr, "FAIL: the two regimes gave equal statistics, so "
                "the mutation below cannot discriminate\n");
        exit(1);
    }
    if (mut[i1] != mut[i2]) {
        fprintf(stderr, "FAIL: with the derate write deleted the two "
                "victims still differ: %.17g against %.17g\n",
                mut[i1], mut[i2]);
        exit(1);
    }
    double rel = fabs(mut[i1] - off[i1]) / off[i1];
    if (!(rel < 1.0e-4)) {
        fprintf(stderr, "FAIL: with the derate write deleted the victim "
                "is %.17g against the control's %.17g, a relative %.3g\n",
                mut[i1], off[i1], rel);
        exit(1);
    }
    g_arms++;
    printf("  and with that write deleted the two victims are equal to "
           "the last digit (%.17g) and within %.3g of the control, the "
           "residual being the deploy the mutation left in place\n",
           mut[i1], rel);
}

/* ---- Gate 6: the other edge ------------------------------------------ */

static void gate_counter_detection_(void)
{
    char so_on[512], so_off[512];
    build_(CTR_ON_KFL, "ctron");
    build_(CTR_OFF_KFL, "ctroff");
    so_path_(so_on, sizeof so_on, "ctron");
    so_path_(so_off, sizeof so_off, "ctroff");

    RlSpecView v;
    spec_of_(so_on, &v);
    int i_snr = chan_(&v, "ir_snr");
    int i_det = chan_(&v, "ir_detected");
    int i_cr  = chan_(&v, "ew_counter_range");
    int i_cd  = chan_(&v, "ew_counter_detected");
    int i_ss  = chan_(&v, "ew_self_signature");
    double on[256], off[256];
    run_obs_(so_on, 3, 0.0, on, 256);
    run_obs_(so_off, 3, 0.0, off, 256);

    if (!(off[i_det] == 0.0 && on[i_det] == 1.0)) {
        fprintf(stderr, "FAIL: counter-detection did not move the "
                "victim's flag: %.1f engaged against %.1f control\n",
                on[i_det], off[i_det]);
        exit(1);
    }
    if (!(on[i_snr] > off[i_snr])) {
        fprintf(stderr, "FAIL: counter-detection did not raise the "
                "victim's statistic\n");
        exit(1);
    }
    if (!(on[i_ss] > 0.0) || !(on[i_cr] > 0.0) || on[i_cd] != 1.0) {
        fprintf(stderr, "FAIL: the jammer published no self-signature or "
                "no counter-detection range\n");
        exit(1);
    }
    g_arms++;
    printf("  the jammer's own emission raises the victim's infrared "
           "statistic from %.10g to %.10g and its flag from 0 to 1, at a "
           "self-signature of %.10g W and a counter range of %.10g m\n",
           off[i_snr], on[i_snr], on[i_ss], on[i_cr]);

    emit_("ctron");
    mutate_("ctron", "ctron_mut",
            "s/if (_kfl_cs > _kfl_snr) _kfl_snr = _kfl_cs;//",
            "_kfl_cs > _kfl_snr", 1, 0);
    char so_mut[512];
    snprintf(so_mut, sizeof so_mut, WORK_DIR "/ctron_mut.so");
    build_emitted_(WORK_DIR "/ctron_mut.cc", so_mut);
    double mut[256];
    run_obs_(so_mut, 3, 0.0, mut, 256);
    if (mut[i_det] != 0.0) {
        fprintf(stderr, "FAIL: with the counter-detection raise deleted "
                "the victim still reads detected\n");
        exit(1);
    }
    g_arms++;
    printf("  and with that raise deleted the victim reads 0 again\n");
}

/* ---- Gate 7: the decoy's deploy is a body effect ---------------------- */

/* The two writes the deploy makes, and the routes each takes to the
 * dynamics, at one separation. Follows the shape the directed-energy
 * item established: the velocity write and the mass write are deleted
 * separately and together, and what is asserted is that each is not
 * inert, not which of them dominates. */
static void deploy_routes_(const char *src, const char *stem,
                           double *d_vel, double *d_mass, double *d_both)
{
    char so[512], mut[512];
    build_(src, stem);
    so_path_(so, sizeof so, stem);
    emit_(stem);

    double full[6], a[6];
    run_body_(so, 6, 2, full);

    char out[128];
    snprintf(out, sizeof out, "%s_v", stem);
    mutate_(stem, out, "s/_kfl_eb->vel.x [-+]= [^;]*;//",
            "vel.x [-+]= _kfl_dv", 1, 0);
    snprintf(mut, sizeof mut, WORK_DIR "/%s.so", out);
    {
        char cc[512];
        snprintf(cc, sizeof cc, WORK_DIR "/%s.cc", out);
        build_emitted_(cc, mut);
    }
    run_body_(mut, 6, 2, a);
    *d_vel = dist6_(full, a);

    snprintf(out, sizeof out, "%s_m", stem);
    mutate_(stem, out,
            "s/k26astro_body_set_mass(_kfl_eb[^;]*;//",
            "set_mass(_kfl_eb", 1, 0);
    snprintf(mut, sizeof mut, WORK_DIR "/%s.so", out);
    {
        char cc[512];
        snprintf(cc, sizeof cc, WORK_DIR "/%s.cc", out);
        build_emitted_(cc, mut);
    }
    run_body_(mut, 6, 2, a);
    *d_mass = dist6_(full, a);

    snprintf(out, sizeof out, "%s_b", stem);
    mutate_(stem, out,
            "s/_kfl_eb->vel.x [-+]= [^;]*;//;"
            "s/k26astro_body_set_mass(_kfl_eb[^;]*;//",
            "vel.x [-+]= _kfl_dv", 1, 0);
    snprintf(mut, sizeof mut, WORK_DIR "/%s.so", out);
    {
        char cc[512];
        snprintf(cc, sizeof cc, WORK_DIR "/%s.cc", out);
        build_emitted_(cc, mut);
    }
    run_body_(mut, 6, 2, a);
    *d_both = dist6_(full, a);
}

#define DEPLOY_KFL(nm, sep) \
    "form " nm "\n" \
    "fn world w\n" \
    SK_EARTH SK_WATCHER SK_MOVER(sep) \
    SK_RADAR("rf", "ir_only", "") \
    SK_DECOY("flare", "active", "0.8") \
    SK_EPISODE SK_ACTION \
    "    observe detect rf of mover as radar\n" \
    "    observe effect flare as dec\n" \
    "    on_step\n        engage flare at watcher\n    end\n" \
    "    objective\n        reward dec_effect\n    end\n" \
    "end\n" \
    "end\n"

static const char *const DEPLOY_FAR_KFL  = DEPLOY_KFL("SKDPF", "2.0e3");
static const char *const DEPLOY_NEAR_KFL = DEPLOY_KFL("SKDPN", "1.5e2");

static void gate_decoy_body_(void)
{
    char so_on[512], so_off[512];
    build_(DEC_ONLY_KFL, "decon2");
    build_(DEC_CTRL_KFL, "decoff2");
    so_path_(so_on, sizeof so_on, "decon2");
    so_path_(so_off, sizeof so_off, "decoff2");

    double on[6], off[6];
    run_body_(so_on, 6, 2, on);
    run_body_(so_off, 6, 2, off);
    double d = dist6_(on, off);
    if (!(d > 0.0)) {
        fprintf(stderr, "FAIL: deploying a decoy left the host's state "
                "identical to the control's\n");
        exit(1);
    }
    RlSpecView v;
    spec_of_(so_on, &v);
    int i_dv = chan_(&v, "dec_dv");
    int i_ml = chan_(&v, "dec_mass_loss");
    double o[256];
    run_obs_(so_on, 3, 0.0, o, 256);
    if (!(o[i_dv] > 0.0) || !(o[i_ml] > 0.0)) {
        fprintf(stderr, "FAIL: the decoy published no increment or no "
                "mass loss\n");
        exit(1);
    }
    g_arms++;
    printf("  the deploy moves the host by %.10g in its own state, at an "
           "increment of %.10g m/s and a mass loss of %.10g kg\n",
           d, o[i_dv], o[i_ml]);

    /* And a magnitude cannot see a sign. The decoy is placed between
     * the host and the observer it is meant to fool, so the host
     * recoils away from that observer; a recoil reversed leaves every
     * published component bit-identical and moves the host the wrong
     * way, which the distance above cannot tell. */
    {
        double wv[6];
        run_body_(so_on, 6, 1, wv);
        double dir[3];
        for (int k = 0; k < 3; k++) dir[k] = wv[k] - on[k];
        double n2 = sqrt(dir[0] * dir[0] + dir[1] * dir[1] +
                         dir[2] * dir[2]);
        ASSERT(n2 > 0.0);
        double proj = 0.0;
        for (int k = 0; k < 3; k++) {
            proj += (on[3 + k] - off[3 + k]) * dir[k] / n2;
        }
        if (!(proj < 0.0)) {
            fprintf(stderr, "FAIL: the host's change in velocity projects "
                    "%.10g onto the line towards the victim; the recoil "
                    "must be away from it\n", proj);
            exit(1);
        }
        g_arms++;
        printf("  and it recoils away from the victim: the change in "
               "velocity projects %.10g m/s onto the line towards it\n",
               proj);
    }

    double fv, fm, fb, nv, nm, nb;
    deploy_routes_(DEPLOY_FAR_KFL, "dpfar", &fv, &fm, &fb);
    deploy_routes_(DEPLOY_NEAR_KFL, "dpnear", &nv, &nm, &nb);
    if (!(fv > 0.0) || !(nv > 0.0)) {
        fprintf(stderr, "FAIL: deleting the velocity write changed "
                "nothing\n");
        exit(1);
    }
    if (!(fm > 0.0) || !(nm > 0.0)) {
        fprintf(stderr, "FAIL: the mass write is a dead store at one of "
                "the two separations: %.10g far, %.10g near\n", fm, nm);
        exit(1);
    }
    if (!(fb > 0.0) || !(nb > 0.0)) {
        fprintf(stderr, "FAIL: deleting both writes changed nothing\n");
        exit(1);
    }
    g_arms++;
    printf("  and both writes reach the dynamics: velocity %.6g far and "
           "%.6g near, mass %.6g far and %.6g near, both together %.6g "
           "and %.6g. No ordering between them is asserted\n",
           fv, nv, fm, nm, fb, nb);
}

/* ---- Gate 8: the reach count, and the two together ------------------- */

static void gate_reach_(void)
{
    char so[512], so_c[512];
    RlSpecView v;
    double o[256], c[256];

    /* Both countermeasures on one victim, against the matched control.
     * The two compose rather than one masking the other, which a
     * fixture holding one of them cannot show. */
    build_(CONTROL_KFL, "engctl");
    so_path_(so, sizeof so, "eng");
    so_path_(so_c, sizeof so_c, "engctl");
    spec_of_(so, &v);
    {
        int i_r = chan_(&v, "radar_snr");
        int i_d = chan_(&v, "radar_detected");
        run_obs_(so, 3, 0.0, o, 256);
        run_obs_(so_c, 3, 0.0, c, 256);
        if (!(o[i_r] < c[i_r] * 1.0e-6) || o[i_d] != 0.0 ||
            c[i_d] != 1.0) {
            fprintf(stderr, "FAIL: the two together read %.10g/%.0f "
                    "against the control's %.10g/%.0f\n",
                    o[i_r], o[i_d], c[i_r], c[i_d]);
            exit(1);
        }
        g_arms++;
        printf("  both countermeasures on one victim take its statistic "
               "from %.10g to %.10g and its flag from 1 to 0\n",
               c[i_r], o[i_r]);
    }

    so_path_(so, sizeof so, "eng");
    spec_of_(so, &v);
    int i_ew = chan_(&v, "ew_reached");
    int i_dc = chan_(&v, "dec_reached");
    run_obs_(so, 3, 0.0, o, 256);
    /* Three detection payloads on the victim: two radars and one
     * telescope. The jammer reaches both radars and the telescope, the
     * decoy reaches all three. */
    if (o[i_ew] != 3.0 || o[i_dc] != 3.0) {
        fprintf(stderr, "FAIL: reach counts are %.0f and %.0f, expected "
                "3 and 3\n", o[i_ew], o[i_dc]);
        exit(1);
    }
    g_arms++;
    printf("  an engagement at a craft carrying three detection payloads "
           "reaches 3 and 3\n");

    build_(REACH0_KFL, "reach0");
    so_path_(so, sizeof so, "reach0");
    spec_of_(so, &v);
    i_ew = chan_(&v, "ew_reached");
    int i_eff = chan_(&v, "ew_effect");
    int i_eng = chan_(&v, "ew_engaged");
    run_obs_(so, 3, 0.0, o, 256);
    if (o[i_eng] != 1.0) {
        fprintf(stderr, "FAIL: the engagement did not run\n");
        exit(1);
    }
    if (o[i_ew] != 0.0 || o[i_eff] != 0.0) {
        fprintf(stderr, "FAIL: an engagement at a craft with no detection "
                "payload reached %.0f payloads at an effect of %.10g\n",
                o[i_ew], o[i_eff]);
        exit(1);
    }
    g_arms++;
    printf("  and one at a craft carrying none reaches 0, with the "
           "engagement flag still 1\n");
}

/* ---- Gate 9: chaff moves the radar, by the law it claims ------------- */

static void gate_chaff_(void)
{
    char s0[512], s1[512], s2[512];
    build_(CHAFF_NONE_KFL, "ch0");
    build_(CHAFF_1X_KFL, "ch1");
    build_(CHAFF_2X_KFL, "ch2");
    so_path_(s0, sizeof s0, "ch0");
    so_path_(s1, sizeof s1, "ch1");
    so_path_(s2, sizeof s2, "ch2");

    RlSpecView v;
    spec_of_(s0, &v);
    int i_snr = chan_(&v, "radar_snr");
    double a[256], b[256], c[256];
    run_obs_(s0, 3, 0.0, a, 256);
    run_obs_(s1, 3, 0.0, b, 256);
    run_obs_(s2, 3, 0.0, c, 256);

    if (!(b[i_snr] > a[i_snr]) || !(c[i_snr] > b[i_snr])) {
        fprintf(stderr, "FAIL: chaff did not raise the radar statistic: "
                "%.10g, %.10g, %.10g\n", a[i_snr], b[i_snr], c[i_snr]);
        exit(1);
    }
    /* The radar statistic is linear in the cross-section, and the mean
     * cloud cross-section is linear in the strip count, so doubling the
     * count doubles the increment. A term added anywhere else in the
     * chain would not obey that. */
    double d1 = b[i_snr] - a[i_snr];
    double d2 = c[i_snr] - a[i_snr];
    double ratio = d2 / d1;
    if (!(fabs(ratio - 2.0) < 1.0e-9)) {
        fprintf(stderr, "FAIL: doubling the strip count multiplied the "
                "increment by %.17g, not 2\n", ratio);
        exit(1);
    }
    g_arms++;
    printf("  chaff raises the radar statistic from %.10g to %.10g and "
           "%.10g, the increment doubling with the count at a ratio of "
           "%.17g\n", a[i_snr], b[i_snr], c[i_snr], ratio);

    emit_("ch1");
    mutate_("ch1", "ch1_mut",
            "s/_kfl_rcs += k26astro_chaff_mean_rcs(/_kfl_rcs += 0.0 * "
            "k26astro_chaff_mean_rcs(/",
            "_kfl_rcs += k26astro_chaff_mean_rcs(", 1, 0);
    char so_mut[512];
    snprintf(so_mut, sizeof so_mut, WORK_DIR "/ch1_mut.so");
    build_emitted_(WORK_DIR "/ch1_mut.cc", so_mut);
    double mut[256];
    run_obs_(so_mut, 3, 0.0, mut, 256);
    if (mut[i_snr] != a[i_snr]) {
        fprintf(stderr, "FAIL: with the chaff term zeroed the statistic "
                "is %.17g against the no-chaff program's %.17g\n",
                mut[i_snr], a[i_snr]);
        exit(1);
    }
    g_arms++;
    printf("  and with the chaff term zeroed it is the no-chaff "
           "program's bit for bit (%.17g)\n", mut[i_snr]);
}

/* ---- Gate 10: no chaff, no change ------------------------------------ */

/* The addition is additive only if a program that declares none
 * publishes what it published before. That is measured against the
 * compiler at the base commit rather than argued from the source: the
 * base compiler emits the program, the emitted source is built against
 * the same archives, and every channel is compared. */
static void gate_chaff_additive_(void)
{
    if (!g_base_ok) {
        printf("  base compiler unavailable: the no-chaff identity arm "
               "stands down\n");
        return;
    }
    build_(CHAFF_NONE_KFL, "ch0b");
    emit_with_(WORK_DIR "/base/kflc/bin/kflc", "ch0b");
    char so_base[512];
    snprintf(so_base, sizeof so_base, WORK_DIR "/ch0b_base.so");
    build_emitted_(WORK_DIR "/ch0b.cc", so_base);

    char so_head[512];
    so_path_(so_head, sizeof so_head, "ch0");

    RlSpecView v;
    spec_of_(so_head, &v);
    int n = v.n_chan_names;
    ASSERT(n > 0);
    double head[256], base[256];
    int differing = 0;
    for (int step = 1; step <= 4; step++) {
        run_obs_(so_head, step, 0.0, head, 256);
        run_obs_(so_base, step, 0.0, base, 256);
        for (int i = 0; i < n; i++) {
            if (head[i] != base[i]) {
                fprintf(stderr, "FAIL: channel `%s` at step %d reads "
                        "%.17g at head and %.17g at " BASE_COMMIT "\n",
                        v.chan_names[i], step, head[i], base[i]);
                differing++;
            }
        }
    }
    if (differing) exit(1);
    g_arms++;
    printf("  a program declaring no chaff key publishes all %d channels "
           "bit-identically to " BASE_COMMIT " over 4 steps\n", n);

    /* And the reason: the term is not in the artifact at all. */
    emit_("ch0");
    int rc = system("grep -q k26astro_chaff " WORK_DIR "/ch0.cc");
    if (WIFEXITED(rc) && WEXITSTATUS(rc) == 0) {
        fprintf(stderr, "FAIL: the no-chaff program's emitted source "
                "names the chaff routine\n");
        exit(1);
    }
    g_arms++;
    printf("  and its emitted source names no chaff routine at all\n");
}

/* ---- Gate 11: a degradation is an act of one step -------------------- */

/* The same fixture engaging only on alternate steps, so one artifact
 * and one episode carry both a degraded step and an undegraded one. */
static const char *const ALT_KFL =
    "form SKALT\n"
    "fn world w\n"
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
    SK_RADAR("rf", "ir_only", "")
    SK_JAMMER("jam", "200.0")
    SK_EPISODE SK_ACTION
    "    observe detect rf of mover as radar\n"
    "    observe effect jam as ew\n"
    "    on_step\n"
    "        if nudge > 0.5\n"
    "            engage jam at watcher\n"
    "        end\n"
    "    end\n"
    "    objective\n        reward radar_snr + ew_effect\n    end\n"
    "end\n"
    "end\n";

static void gate_clear_(void)
{
    build_(ALT_KFL, "alt");
    char so[512];
    so_path_(so, sizeof so, "alt");
    RlSpecView v;
    spec_of_(so, &v);
    int i_snr = chan_(&v, "radar_snr");
    int i_eng = chan_(&v, "ew_engaged");
    int i_eff = chan_(&v, "ew_effect");
    int i_rch = chan_(&v, "ew_reached");

    SkArt a;
    art_open_(&a, so, 4242u, 1u);
    double on_act[4] = { 1.0, 0.0, 0.0, 0.0 };
    double off_act[4] = { 0.0, 0.0, 0.0, 0.0 };
    double o[256];
    double snr_on = 0.0, snr_off = 0.0;

    ASSERT(a.s.step(a.env, on_act) == K26RL_OK);
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    snr_on = o[i_snr];
    ASSERT(o[i_eng] == 1.0 && o[i_rch] == 1.0 && o[i_eff] > 0.0);

    ASSERT(a.s.step(a.env, off_act) == K26RL_OK);
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    snr_off = o[i_snr];
    if (!(o[i_eng] == 0.0 && o[i_eff] == 0.0 && o[i_rch] == 0.0)) {
        fprintf(stderr, "FAIL: a step that engaged nothing published "
                "%.1f, %.10g, %.0f\n", o[i_eng], o[i_eff], o[i_rch]);
        exit(1);
    }
    if (!(snr_off > snr_on * 1.0e3)) {
        fprintf(stderr, "FAIL: the degradation survived into the next "
                "step: %.10g engaged, %.10g not\n", snr_on, snr_off);
        exit(1);
    }
    art_close_(&a);
    g_arms++;
    printf("  the degradation lasts one step: %.10g engaged, %.10g on the "
           "step after, on one artifact and one episode\n",
           snr_on, snr_off);

    /* And a reset clears it too: the observation a fresh episode opens
     * with carries nothing from the episode before. */
    art_open_(&a, so, 4242u, 1u);
    ASSERT(a.s.step(a.env, on_act) == K26RL_OK);
    ASSERT(a.s.reset(a.env) == K26RL_OK);
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    if (!(o[i_eng] == 0.0 && o[i_eff] == 0.0 && o[i_rch] == 0.0)) {
        fprintf(stderr, "FAIL: a reset left an engagement published\n");
        exit(1);
    }
    double after_reset = o[i_snr];
    art_close_(&a);
    double fresh[256];
    reset_obs_(so, fresh, 256);
    if (after_reset != fresh[i_snr]) {
        fprintf(stderr, "FAIL: the observation after a reset is %.17g "
                "against a fresh handle's %.17g\n", after_reset,
                fresh[i_snr]);
        exit(1);
    }
    g_arms++;
    printf("  and a reset returns the victim's statistic to a fresh "
           "handle's, bit for bit (%.17g)\n", after_reset);
}

/* ---- Gate 12: every published component moves ------------------------ */

/* The spread of each component of one observe over a run, so a
 * component that never moves is named rather than assumed to. */
static void spread_(const char *so_path, int first, int width, int n,
                    double *lo, double *hi)
{
    SkArt a;
    art_open_(&a, so_path, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    double o[256];
    for (int c = 0; c < width; c++) { lo[c] = 1.0e300; hi[c] = -1.0e300; }
    for (int i = 0; i < n; i++) {
        ASSERT(a.s.step(a.env, act) == K26RL_OK);
        ASSERT(a.s.obs(a.env, o) == K26RL_OK);
        for (int c = 0; c < width; c++) {
            double v = o[first + c];
            if (v < lo[c]) lo[c] = v;
            if (v > hi[c]) hi[c] = v;
        }
    }
    art_close_(&a);
}

static void gate_components_move_(void)
{
    /* The jammer's set falls in two. Four components are functions of
     * the geometry and must move as the craft move and turn. The other
     * five are functions of what was declared and hold within a run by
     * construction: `_engaged` and `_reached` are what the statement
     * did, `_self_signature` is the declared power, `_counter_range` is
     * that power against the victim instrument's own optics, and
     * `_counter_detected` is a comparison of that range against the
     * separation. Those are checked for the value they must hold here
     * and then driven between two fixtures below, since a component
     * that cannot move anywhere is a component nothing can be shaped
     * against. */
    static const char *const JAM_MOVES[] = {
        "ew_effect", "ew_range", "ew_rcs", "ew_burn_through", NULL
    };
    static const char *const DEC_MOVES[] = {
        "dec_range", "dec_dv", NULL
    };
    char so[512];
    RlSpecView v;
    double lo[32], hi[32];

    so_path_(so, sizeof so, "eng");
    spec_of_(so, &v);
    int base = chan_(&v, "ew_engaged");
    spread_(so, base, 9, 12, lo, hi);
    for (int i = 0; JAM_MOVES[i]; i++) {
        int c = chan_(&v, JAM_MOVES[i]) - base;
        if (!(hi[c] > lo[c])) {
            fprintf(stderr, "FAIL: `%s` held still at %.17g over 12 "
                    "steps\n", JAM_MOVES[i], lo[c]);
            exit(1);
        }
    }
    {
        int c = chan_(&v, "ew_self_signature") - base;
        ASSERT(lo[c] == hi[c] && lo[c] == 200.0);
        c = chan_(&v, "ew_counter_detected") - base;
        ASSERT(lo[c] == 1.0 && hi[c] == 1.0);
        c = chan_(&v, "ew_counter_range") - base;
        ASSERT(lo[c] == hi[c] && lo[c] > 0.0);
    }
    g_arms++;
    printf("  the jammer's four geometric components each move over 12 "
           "steps, and its three declared ones hold\n");

    /* The three declared components, driven between two fixtures that
     * differ in the transmitted power alone. At a ten-thousandth of a
     * watt the counter-detection range falls inside the separation, so
     * the emitter's own signal is present at the victim and below its
     * threshold: the flag reads 0 where the loud jammer's reads 1. */
    build_(CTR_ON_KFL, "ctrloud");
    build_(CTR_QUIET_KFL, "ctrquiet");
    {
        char so_l[512], so_q[512];
        so_path_(so_l, sizeof so_l, "ctrloud");
        so_path_(so_q, sizeof so_q, "ctrquiet");
        RlSpecView vl;
        spec_of_(so_l, &vl);
        int i_ss = chan_(&vl, "ew_self_signature");
        int i_cr = chan_(&vl, "ew_counter_range");
        int i_cd = chan_(&vl, "ew_counter_detected");
        int i_de = chan_(&vl, "ir_detected");
        double l[256], q[256];
        run_obs_(so_l, 3, 0.0, l, 256);
        run_obs_(so_q, 3, 0.0, q, 256);
        if (!(q[i_ss] < l[i_ss]) || !(q[i_cr] < l[i_cr]) ||
            l[i_cd] != 1.0 || q[i_cd] != 0.0 ||
            l[i_de] != 1.0 || q[i_de] != 0.0) {
            fprintf(stderr, "FAIL: the declared components did not move "
                    "between the two powers: %.10g/%.10g/%.0f/%.0f loud "
                    "against %.10g/%.10g/%.0f/%.0f quiet\n",
                    l[i_ss], l[i_cr], l[i_cd], l[i_de],
                    q[i_ss], q[i_cr], q[i_cd], q[i_de]);
            exit(1);
        }
        g_arms++;
        printf("  and its declared three move with the power: a counter "
               "range of %.10g m at %.10g W against %.10g m at %.10g W, "
               "the flag falling to 0 with the victim's\n",
               l[i_cr], l[i_ss], q[i_cr], q[i_ss]);
    }

    base = chan_(&v, "dec_engaged");
    spread_(so, base, 7, 12, lo, hi);
    for (int i = 0; DEC_MOVES[i]; i++) {
        int c = chan_(&v, DEC_MOVES[i]) - base;
        if (!(hi[c] > lo[c])) {
            fprintf(stderr, "FAIL: `%s` held still at %.17g over 12 "
                    "steps\n", DEC_MOVES[i], lo[c]);
            exit(1);
        }
    }
    /* The mass loss is the declared dry mass every time, and the
     * increment moves because the host's mass falls with each deploy:
     * the two together are what says the deploy is cumulative. */
    {
        int c = chan_(&v, "dec_mass_loss") - base;
        ASSERT(lo[c] == hi[c] && lo[c] == 5.0);
        c = chan_(&v, "dec_dv") - base;
        ASSERT(hi[c] > lo[c]);
    }
    g_arms++;
    printf("  the decoy's range and increment move while its mass loss "
           "holds at the declared 5 kg, which is the deploy being "
           "cumulative\n");

    /* And the two components that describe the discrimination move
     * when the decoy's own match quality does, which the effector's own
     * channels are the only place to see. */
    build_(
        "form SKDQ\n"
        "fn world w\n"
        SK_EARTH SK_WATCHER SK_MOVER("2.0e4")
        SK_RADAR("rf", "ir_rcs_accel", "")
        SK_DECOY("flare", "passive", "0.1")
        SK_EPISODE SK_ACTION
        "    observe detect rf of mover as radar\n"
        "    observe effect flare as dec\n"
        "    on_step\n        engage flare at watcher\n    end\n"
        "    objective\n        reward dec_effect\n    end\n"
        "end\nend\n", "decq");
    char so2[512];
    so_path_(so2, sizeof so2, "decq");
    RlSpecView v2;
    spec_of_(so2, &v2);
    double o1[256], o2[256];
    run_obs_(so, 3, 0.0, o1, 256);
    run_obs_(so2, 3, 0.0, o2, 256);
    double p1 = o1[chan_(&v, "dec_p_discriminated")];
    double p2 = o2[chan_(&v2, "dec_p_discriminated")];
    double e1 = o1[chan_(&v, "dec_effect")];
    double e2 = o2[chan_(&v2, "dec_effect")];
    if (!(p2 > p1) || !(e2 < e1)) {
        fprintf(stderr, "FAIL: a poorly matched passive decoy against the "
                "full discriminator gave %.10g/%.10g against a well "
                "matched active one's %.10g/%.10g\n", p2, e2, p1, e1);
        exit(1);
    }
    g_arms++;
    printf("  and a poorly matched passive decoy is discriminated at "
           "%.10g against a well matched active one's %.10g\n", p2, p1);
}

/* ---- Gate 13: one engagement per payload per step, at runtime -------- */

/* One artifact whose loop bound is driven by the action, so the run
 * that engages once and the run that engages twice are the same
 * compiled code and differ in nothing else. A fixture holding only the
 * faulting case could not tell the rule from a fault that always
 * fires. */
#define LOOP_KFL(payload, form_name) \
    "form " form_name "\n" \
    "fn world w\n" \
    SK_EARTH SK_WATCHER SK_MOVER("2.0e4") \
    SK_RADAR("rf", "ir_only", "") \
    payload \
    SK_EPISODE SK_ACTION \
    "    observe detect rf of mover as radar\n" \
    "    on_step\n" \
    "        let n: double = 1.0\n" \
    "        if nudge > 0.5\n" \
    "            n = 2.0\n" \
    "        end\n" \
    "        let i: double = 0.0\n" \
    "        while i < n\n" \
    "            engage cm at watcher\n" \
    "            i = i + 1.0\n" \
    "        end\n" \
    "    end\n" \
    "    objective\n        reward radar_snr\n    end\n" \
    "end\n" \
    "end\n"

static void loop_arm_(const char *kind, const char *src, const char *stem)
{
    build_(src, stem);
    char so[512];
    so_path_(so, sizeof so, stem);
    SkArt a;
    art_open_(&a, so, 4242u, 1u);
    double once[4]  = { 0.0, 0.0, 0.0, 0.0 };
    double twice[4] = { 1.0, 0.0, 0.0, 0.0 };
    uint16_t f = 0;
    uint32_t fl = 0;

    ASSERT(a.s.step(a.env, once) == K26RL_OK);
    ASSERT(a.s.fault_codes(a.env, &f) == K26RL_OK);
    if (f != 0) {
        fprintf(stderr, "FAIL: one %s engagement in a loop faulted with "
                "%u\n", kind, (unsigned)f);
        exit(1);
    }
    g_arms++;
    printf("  a loop engaging one %s once does not fault\n", kind);

    ASSERT(a.s.step(a.env, twice) == K26RL_OK);
    ASSERT(a.s.fault_codes(a.env, &f) == K26RL_OK);
    if (f != (uint16_t)K26RL_E_ENV_INTERNAL) {
        fprintf(stderr, "FAIL: a second %s engagement in one step "
                "reported fault code %u, expected %u\n", kind,
                (unsigned)f, (unsigned)K26RL_E_ENV_INTERNAL);
        exit(1);
    }
    ASSERT(a.s.flags(a.env, &fl) == K26RL_OK);
    ASSERT((fl & K26RL_FLAG_FAULT) != 0u);
    art_close_(&a);
    g_arms++;
    printf("  and the same %s statement reached twice faults with "
           "K26RL_E_ENV_INTERNAL (%u)\n", kind,
           (unsigned)K26RL_E_ENV_INTERNAL);
}

static void gate_runtime_second_engage_(void)
{
    loop_arm_("jammer", LOOP_KFL(SK_JAMMER("cm", "200.0"), "SKL1"),
              "loopj");
    loop_arm_("decoy", LOOP_KFL(SK_DECOY("cm", "active", "0.8"), "SKL2"),
              "loopd");
}

/* ---- Gate 14: the new words are still names -------------------------- */

/* Every word this change adds, in each position an identifier can appear,
 * through the compiler at the base commit and through this one. None of
 * them opens a construct: they are `kind=` values, `mode=` values,
 * regime words and key names, all of which sit after a `=` or inside a
 * statement that already existed. That is the claim, and it is measured
 * rather than read off the source. */
static const char *const NEW_WORDS[] = {
    "decoy", "jammer", "dazzler", "noise", "cover_pulse", "deception",
    "passive", "active", "ir_only", "ir_plus_rcs", "ir_rcs_accel",
    "p_j_w", "g_j_db", "radiator_temp_k", "dry_mass_kg",
    "deploy_dv_mps", "ir_match_quality", "rcs_match_quality",
    "accel_match_quality", "discriminator_regime",
    "target_chaff_n_strips", "target_chaff_sigma_dipole_m2", NULL
};

static int probe_(const char *kflc, const char *word, int shape)
{
    static const char *const SHAPES[] = {
        "    %s = 2.0\n    print \"%s\"\n",
        "    let %s: double = 3.0\n    print \"%s\"\n",
        "    %s = 1.0\n    %s = %s + 1.0\n"
    };
    char body[1024];
    if (shape == 2) {
        snprintf(body, sizeof body, SHAPES[2], word, word, word);
    } else {
        snprintf(body, sizeof body, SHAPES[shape], word, word);
    }
    char src[4096];
    snprintf(src, sizeof src,
             "form PROBE\n"
             "fn run\n"
             "%s"
             "end\n"
             "end\n", body);
    rl_write_file_(WORK_DIR "/probe.kfl", src);
    char cmd[1024];
    snprintf(cmd, sizeof cmd, "%s --check " WORK_DIR "/probe.kfl "
             "> /dev/null 2>&1", kflc);
    int rc = system(cmd);
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

static void gate_contextual_words_(void)
{
    if (!g_base_ok) {
        printf("  base compiler unavailable: the identifier probes stand "
               "down\n");
        return;
    }
    int n = 0, regress = 0, newly = 0;
    for (int i = 0; NEW_WORDS[i]; i++) {
        for (int s = 0; s < 3; s++) {
            int base = probe_(WORK_DIR "/base/kflc/bin/kflc",
                              NEW_WORDS[i], s);
            int head = probe_("./bin/kflc", NEW_WORDS[i], s);
            n++;
            if (base == 0 && head != 0) {
                fprintf(stderr, "FAIL: `%s` in shape %d compiled at "
                        BASE_COMMIT " and does not at head\n",
                        NEW_WORDS[i], s);
                regress++;
            }
            if (base != 0 && head == 0) newly++;
        }
    }
    if (regress) exit(1);
    g_arms++;
    printf("  %d identifier probes over %d new words in 3 shapes: %d "
           "regressions against " BASE_COMMIT ", %d newly accepted\n",
           n, (int)(sizeof NEW_WORDS / sizeof NEW_WORDS[0]) - 1,
           regress, newly);
}

/* ---- The base compiler ----------------------------------------------- */

static int run_(const char *cmd)
{
    int rc = system(cmd);
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : -1;
}

static void build_base_(void)
{
    if (run_("git -C .. rev-parse --verify --quiet " BASE_COMMIT
             "^{commit} > /dev/null 2>&1") != 0) {
        printf("  base commit " BASE_COMMIT " is not in this checkout's "
               "history\n");
        return;
    }
    if (run_("mkdir -p " WORK_DIR "/base && git -C .. archive "
             BASE_COMMIT " | tar -x -C " WORK_DIR "/base") != 0) {
        printf("  could not extract " BASE_COMMIT "\n");
        return;
    }
    if (run_("make -C " WORK_DIR "/base/libk26rng > " WORK_DIR
             "/base.log 2>&1") != 0 ||
        run_("make -C " WORK_DIR "/base/libk26sense >> " WORK_DIR
             "/base.log 2>&1") != 0 ||
        run_("make -C " WORK_DIR "/base/libk26rl >> " WORK_DIR
             "/base.log 2>&1") != 0 ||
        run_("make -C " WORK_DIR "/base/kflc bin/kflc >> " WORK_DIR
             "/base.log 2>&1") != 0) {
        (void)!system("tail -20 " WORK_DIR "/base.log");
        printf("  the compiler at " BASE_COMMIT " did not build\n");
        return;
    }
    g_base_ok = 1;
    printf("  the compiler at " BASE_COMMIT " is built\n");
}

int main(void)
{
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_run_or_die_("cp examples/assets/calibration_box.k26asm "
                   "examples/assets/calibration_box.k26mesh "
                   "examples/assets/crew_vehicle_10t.k26asm "
                   WORK_DIR "/");
    rl_write_file_(WORK_DIR "/scratch_bare.k26asm", BARE_ASM);

    printf("test_rl_softkill: the countermeasure effectors\n");
    gate_refusals_();

    if (!rl_libs_present_("test_rl_softkill")) {
        printf("test_rl_softkill: %d arm(s) passed, drive arms stood down "
               "(stack archives absent)\n", g_arms);
        return 77;
    }
    build_base_();

    gate_acceptance_();
    gate_channels_();
    gate_jammer_capability_();
    gate_decoy_capability_();
    gate_counter_detection_();
    gate_decoy_body_();
    gate_reach_();
    gate_chaff_();
    gate_chaff_additive_();
    gate_clear_();
    gate_components_move_();
    gate_runtime_second_engage_();
    gate_contextual_words_();

    printf("test_rl_softkill: %d arm(s) passed\n", g_arms);
    return 0;
}
