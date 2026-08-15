/* test_rl_grammar.c: Grammar 3.2 reinforcement learning front end.
 *
 * Positive gate: a full RL world (episode with control_dt / horizon /
 * terminated when / reset lines, box + discrete actions with
 * defaults, an on_step body, observe-as, an objective with reward and
 * terminal) parses and checks clean.
 *
 * Negative gates pin the diagnostics: step inside an RL world,
 * missing episode, missing control_dt, duplicate objective, duplicate
 * action name, reset with a bad state key, a distribution expression
 * outside its two valid positions, an unknown name in `terminated
 * when`, and a statement the on_step body rejects. The never-ending
 * episode warns but still checks clean.
 *
 * Emit-level gates: a top-level world `let`/`const` scalar is
 * readable in `terminated when` / `reward` / `terminal` (the emitter
 * captures it at create), and the nested-block and non-scalar
 * refusals carry precise diagnostics.
 *
 * Purity gates: an expression that has to reproduce on replay is
 * refused when it reaches a builtin that is not marked pure, in a
 * body state assignment and in all three objective-side positions
 * (`reward`, `terminal`, `terminated when`), directly or through a
 * user fn, while pure builtins stay admitted in every one of them.
 *
 * Mode gates: the two refusals that guard published bytes, an
 * over-long channel name and a body named `episode`, are asserted on
 * the artifact-producing paths as well as under `--check`, since a
 * refusal that only `--check` performs does not stop a compile.
 *
 * Pattern: write a small .kfl fixture to a tmpfile, run
 *   ./bin/kflc --check <tmpfile>   (or --emit / -o for the gates that
 * pin a mode) capture stderr + exit code, assert on the captured
 * state.
 *
 * Wire: see kflc/Makefile RL_GRAMMAR_TEST + test target.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static void write_fixture_(const char *path, const char *content)
{
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); exit(2); }
    fputs(content, f);
    fclose(f);
}

/* Runs `kflc <mode> <fixture>`. Returns exit code; writes stderr
 * text to *err_out (caller frees). */
static int run_mode_(const char *mode, const char *fixture, char **err_out)
{
    char cmd[512];
    snprintf(cmd, sizeof cmd,
             "./bin/kflc %s %s >/dev/null 2>/tmp/kflc_rl_err.log",
             mode, fixture);
    int rc = system(cmd);
    FILE *f = fopen("/tmp/kflc_rl_err.log", "rb");
    if (!f) { *err_out = strdup(""); return WEXITSTATUS(rc); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) sz = 0;
    fseek(f, 0, SEEK_SET);
    *err_out = (char *)malloc((size_t)sz + 1);
    if (sz > 0) (void)!fread(*err_out, 1, (size_t)sz, f);
    (*err_out)[sz] = '\0';
    fclose(f);
    return WEXITSTATUS(rc);
}

/* Attitude state is only accepted on a body that binds a vehicle
 * assembly, because the advance needs the inertia tensor an assembly
 * derives. The fixtures below that carry attitude keys bind this one,
 * written beside them so the relative path resolves. */
static void write_attitude_asset_(void)
{
    write_fixture_("/tmp/kflc_rl_att.k26asm",
        "assembly grammar_box\n"
        "    frame x_to_port\n"
        "    provenance mass \"calibration shape, not a craft\" computed\n"
        "    component hull\n"
        "        mass 1000.0\n"
        "        at 0 0 0\n"
        "        collider box 1.0 0.5 0.5\n"
        "    end\n"
        "end\n");
}

static int n_pass = 0;

/* One fixture, one expectation under the given kflc mode:
 * `expect_rc` on exit, `expect_msg` present on stderr (NULL skips the
 * message check), `absent_msg` absent from stderr (NULL skips). */
static void expect_mode_(const char *mode, const char *tag,
                         const char *src, int expect_rc,
                         const char *expect_msg, const char *absent_msg)
{
    char path[128];
    snprintf(path, sizeof path, "/tmp/kflc_rl_%s.kfl", tag);
    write_fixture_(path, src);
    char *err = NULL;
    int rc = run_mode_(mode, path, &err);
    if (rc != expect_rc ||
        (expect_msg && strstr(err, expect_msg) == NULL) ||
        (absent_msg && strstr(err, absent_msg) != NULL))
    {
        fprintf(stderr, "case `%s`: rc=%d (expected %d)\nstderr:\n%s\n",
                tag, rc, expect_rc, err);
    }
    ASSERT(rc == expect_rc);
    if (expect_msg) ASSERT(strstr(err, expect_msg) != NULL);
    if (absent_msg) ASSERT(strstr(err, absent_msg) == NULL);
    free(err);
    unlink(path);
    n_pass++;
}

static void expect_(const char *tag, const char *src, int expect_rc,
                    const char *expect_msg, const char *absent_msg)
{
    expect_mode_("--check", tag, src, expect_rc, expect_msg, absent_msg);
}

/* The same expectation on the checking path and on the emitting one.
 * A refusal only the checker performs does not stop a compile, which
 * is the defect class the mode gates below already pin; every
 * actuator case is therefore asserted twice rather than trusted to
 * one path. */
static void expect_both_(const char *tag, const char *src, int expect_rc,
                         const char *expect_msg, const char *absent_msg)
{
    char t[96];
    snprintf(t, sizeof t, "%s_chk", tag);
    expect_mode_("--check", t, src, expect_rc, expect_msg, absent_msg);
    snprintf(t, sizeof t, "%s_emit", tag);
    expect_mode_("--emit", t, src, expect_rc, expect_msg, absent_msg);
}

/* An assembly carrying one of each commandable component, so the
 * refusals below can be about the name or the field rather than
 * about the asset. */
static void write_actuator_asset_(void)
{
    write_fixture_("/tmp/kflc_rl_act.k26asm",
        "assembly grammar_act\n"
        "    frame x_to_port\n"
        "    provenance mass \"calibration shape, not a craft\" computed\n"
        "    component hull\n"
        "        mass 1000.0\n"
        "        at 0 0 0\n"
        "        collider box 1.0 0.5 0.5\n"
        "    end\n"
        "    wheel yaw\n"
        "        axis 0.0 0.0 1.0\n"
        "        spin_inertia 0.05\n"
        "        max_momentum 15.0\n"
        "        max_torque 0.20\n"
        "    end\n"
        "    magnetorquer m_y\n"
        "        axis 0.0 1.0 0.0\n"
        "        max_dipole 30.0\n"
        "    end\n"
        "    thruster rcs_py\n"
        "        at 1.05 0.92 0.0\n"
        "        dir 0.0 -1.0 0.0\n"
        "        thrust 400.0\n"
        "    end\n"
        "end\n");
}

/* A world with `craft` bound to that assembly and `probe` bound to
 * none, so the no-assembly refusal has a body to name. STEP is the
 * on_step body and REWARD the reward expression.
 *
 * `earth` carries its NAIF id because the assembly declares a
 * magnetorquer, and a magnetorquer's parent must name a rotation
 * model: that is what the field chain is evaluated in. The refusal
 * that enforces it is gated below. */
#define ACT_WORLD(STEP, REWARD) \
    "form RL_ACTG\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24" \
    " ephem_naif_id=399\n" \
    "    astro_body craft assembly=\"kflc_rl_act.k26asm\" parent=earth" \
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n" \
    "    astro_body probe gm=1.0 parent=earth" \
    " pos_x=8.0e6 vel_y=7000.0\n" \
    "    episode\n" \
    "        control_dt 0.5\n" \
    "        horizon 4\n" \
    "    end\n" \
    "    action a box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    STEP \
    "    end\n" \
    "    observe craft from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward " REWARD "\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* The three-part command surface: one accepted shape per component
 * kind and the six refusals. The positive case is what keeps the
 * refusals from passing vacuously: a build that rejected every
 * three-part name would fail it. */
static void actuator_cases_(void)
{
    write_actuator_asset_();

    expect_both_("act_ok",
        ACT_WORLD(
        "        craft.yaw.torque = a * 0.2\n"
        "        craft.m_y.dipole = a * 30.0\n"
        "        craft.rcs_py.throttle = a * 0.0 + 0.5\n"
        "        let h: double = craft.yaw.momentum\n"
        "        let r: double = craft.yaw.rate\n"
        "        craft.omega_x = h * 0.0 + r * 0.0\n", "0.0"),
        0, NULL, "error");

    expect_both_("act_unknown_body",
        ACT_WORLD("        ghost.yaw.torque = a\n", "0.0"),
        1, "no astro_body named `ghost` is declared in this world", NULL);

    expect_both_("act_no_assembly",
        ACT_WORLD("        probe.yaw.torque = a\n", "0.0"),
        1, "`probe` declares no `assembly=`", NULL);

    expect_both_("act_unknown_component",
        ACT_WORLD("        craft.pitch.torque = a\n", "0.0"),
        1, "has no wheel, magnetorquer or thruster named `pitch`", NULL);

    expect_both_("act_wheel_field",
        ACT_WORLD("        craft.yaw.spin = a\n", "0.0"),
        1, "a wheel takes `torque` and reads `momentum` and `rate`, "
           "not `spin`", NULL);

    expect_both_("act_torquer_field",
        ACT_WORLD("        craft.m_y.torque = a\n", "0.0"),
        1, "a magnetorquer takes `dipole`, not `torque`", NULL);

    expect_both_("act_thruster_field",
        ACT_WORLD("        craft.rcs_py.dipole = a\n", "0.0"),
        1, "a thruster takes `throttle`, not `dipole`", NULL);

    expect_both_("act_readonly",
        ACT_WORLD("        craft.yaw.momentum = a\n", "0.0"),
        1, "`momentum` is a reading, not a command", NULL);

    /* The magnetorquer's frame chain, as a precondition on the
     * declaration rather than a zero at run time. Without a rotation
     * model on the parent there is no frame the field is defined in,
     * and a magnetorquer that compiled, accepted commands and
     * produced no torque would be a training run wasted rather than a
     * compile failed. The positive case is ACT_WORLD itself, which
     * every case above compiles or refuses for another reason, so
     * these three are refusals of exactly this precondition. */
    expect_both_("act_mag_parent_no_naif",
        "form RL_ACTN\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"kflc_rl_act.k26asm\" parent=earth"
        " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.m_y.dipole = a * 30.0\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "declares no `ephem_naif_id=`", NULL);

    expect_both_("act_mag_no_parent",
        "form RL_ACTP\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24"
        " ephem_naif_id=399\n"
        "    astro_body craft assembly=\"kflc_rl_act.k26asm\""
        " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.m_y.dipole = a * 30.0\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "declares no `parent=`", NULL);

    /* The refusal is about the magnetorquer and not about assemblies
     * in general: the same world, with an assembly carrying only a
     * wheel, compiles with no id on the parent. */
    write_fixture_("/tmp/kflc_rl_wheelonly.k26asm",
        "assembly grammar_wheel\n"
        "    frame x_to_port\n"
        "    provenance mass \"calibration shape\" computed\n"
        "    component hull\n"
        "        mass 1000.0\n"
        "        at 0 0 0\n"
        "        collider box 1.0 0.5 0.5\n"
        "    end\n"
        "    wheel yaw\n"
        "        axis 0.0 0.0 1.0\n"
        "        spin_inertia 0.05\n"
        "        max_momentum 15.0\n"
        "        max_torque 0.20\n"
        "    end\n"
        "end\n");
    expect_both_("act_wheel_needs_no_naif",
        "form RL_ACTW\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"kflc_rl_wheelonly.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.yaw.torque = a * 0.2\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* The contact observe form. Three channels rather than a flag
     * bit, and a body that carries no assembly is refused rather than
     * given three channels that could never be anything but zero.
     * The positive case is `obs_contact_channels_readable` below,
     * which a build that refused every contact form would fail while
     * passing both refusals here. */
    expect_both_("obs_contact_unknown_body",
        "form RL_CONU\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"kflc_rl_wheelonly.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    observe contact of ghost as hit\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward a * 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "no astro_body of that name is declared", NULL);

    expect_both_("obs_contact_no_assembly",
        "form RL_CONA\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body probe gm=1.0 parent=earth"
        " pos_x=8.0e6 vel_y=7000.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    observe contact of probe as hit\n"
        "    observe probe from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward a * 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "carries no colliders and can report no contact", NULL);

    /* The three channels are readable by name in the objective, which
     * is the whole route by which a contact reaches a reward or a
     * termination: nothing widens the objective's scope. */
    expect_both_("obs_contact_channels_readable",
        "form RL_CONR\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"kflc_rl_wheelonly.k26asm\""
        " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "        terminated when hit_hit > 0.5\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    observe contact of craft as hit\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward hit_hit + hit_fraction + hit_speed + a * 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* A body genuinely named `contact` still takes the ordinary form,
     * because `observe contact from earth` has no `of` after the
     * name. Without this the keyword would have taken a name away
     * from every program that used it. */
    expect_both_("obs_contact_is_not_a_keyword",
        "form RL_CONK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body contact gm=1.0 parent=earth"
        " pos_x=8.0e6 vel_y=7000.0\n"
        "    episode\n"
        "        control_dt 0.5\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    observe contact from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward trk_range + a * 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* The contact resolution line. One optional line inside the
     * episode, because the resolution is a property of how the
     * episode ends rather than of any one body; absent means arrest,
     * which is the default, so no program written before the line
     * existed changes meaning.
     *
     * The three accepted shapes run first. Without them every
     * refusal below would be satisfied by a build that rejected
     * `contact` lines outright, which is the difference between a
     * gate on the coefficients and a gate on the keyword. */
#define CONTACT_WORLD(LINE) \
    "form RL_CRES\n" \
    "fn world w\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft assembly=\"kflc_rl_wheelonly.k26asm\"" \
    " parent=earth pos_x=7.0e6 vel_y=7546.0 quat_w=1.0\n" \
    "    episode\n" \
    "        control_dt 0.5\n" \
    "        horizon 4\n" \
    LINE \
    "    end\n" \
    "    action a box -1.0 1.0 default 0.0\n" \
    "    observe contact of craft as hit\n" \
    "    observe craft from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward hit_hit + a * 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

    expect_both_("contact_absent",  CONTACT_WORLD(""), 0, NULL, "error");
    expect_both_("contact_arrest",
        CONTACT_WORLD("        contact arrest\n"), 0, NULL, "error");
    expect_both_("contact_bounce",
        CONTACT_WORLD("        contact bounce restitution 0.4"
                      " friction 0.25\n"), 0, NULL, "error");
    /* Both coefficients are compile-time expressions, not only
     * literals, so a program can name its own constants. */
    expect_both_("contact_bounce_expr",
        CONTACT_WORLD("        contact bounce restitution 0.2 + 0.2"
                      " friction 1.0 / 4.0\n"), 0, NULL, "error");

    expect_both_("contact_unknown_kind",
        CONTACT_WORLD("        contact squish\n"),
        1, "takes `arrest` or `bounce`", NULL);
    expect_both_("contact_arrest_trailing",
        CONTACT_WORLD("        contact arrest restitution 0.5\n"),
        1, "takes no further words", NULL);
    expect_both_("contact_bounce_bare",
        CONTACT_WORLD("        contact bounce\n"),
        1, "requires `restitution <expr> friction <expr>`", NULL);
    expect_both_("contact_bounce_no_friction",
        CONTACT_WORLD("        contact bounce restitution 0.5\n"),
        1, "requires `friction <expr>` after the restitution", NULL);
    expect_both_("contact_restitution_high",
        CONTACT_WORLD("        contact bounce restitution 1.5"
                      " friction 0.2\n"),
        1, "restitution` is 1.5", NULL);
    expect_both_("contact_restitution_negative",
        CONTACT_WORLD("        contact bounce restitution -0.1"
                      " friction 0.2\n"),
        1, "closed interval 0 to 1", NULL);
    expect_both_("contact_friction_negative",
        CONTACT_WORLD("        contact bounce restitution 0.5"
                      " friction -0.2\n"),
        1, "friction` is -0.2", NULL);
    expect_both_("contact_duplicate",
        CONTACT_WORLD("        contact arrest\n"
                      "        contact bounce restitution 0.5"
                      " friction 0.2\n"),
        1, "duplicate `contact`", NULL);
    /* The boundaries of the closed interval are inside it, which a
     * check written with the wrong comparison would refuse. */
    expect_both_("contact_restitution_zero",
        CONTACT_WORLD("        contact bounce restitution 0.0"
                      " friction 0.0\n"), 0, NULL, "error");
    expect_both_("contact_restitution_one",
        CONTACT_WORLD("        contact bounce restitution 1.0"
                      " friction 0.0\n"), 0, NULL, "error");
#undef CONTACT_WORLD

    /* Outside on_step: the same name in the objective is refused,
     * because a reward read of a command surface would be a read of
     * state the step has already moved on from. */
    expect_both_("act_outside_on_step",
        ACT_WORLD("        craft.yaw.torque = a\n",
                  "craft.yaw.momentum"),
        1, "only inside an on_step block", NULL);
}

int main(void)
{
    write_attitude_asset_();
    actuator_cases_();

    /* Positive: the full RL surface parses and checks clean. */
    expect_("full",
        "form RL_FULL\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=uniform(-1.0,1.0) pos_y=0.0 pos_z=0.0"
        " vel_x=0.0 vel_y=normal(0.0,0.1) vel_z=0.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 1000\n"
        "        terminated when episode.steps > 900\n"
        "        reset craft.pos_x uniform(-1.0, 1.0)\n"
        "        reset craft.vel_y normal(0.0, 0.1)\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    action gear discrete 3 default 0\n"
        "    on_step\n"
        "        let damping: double = 0.99\n"
        "    end\n"
        "    observe craft from earth mode=astrometric as track\n"
        "    objective\n"
        "        reward 0.0 - track_range + thrust\n"
        "        terminal track_range < 0.5\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* step / propagate belong to the episode machinery in RL worlds. */
    expect_("step",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    step 0.1\n"
        "end\n"
        "end\n",
        1, "the stepping belongs to the episode machinery", NULL);

    /* An RL construct without an episode block. */
    expect_("noepisode",
        "form P\n"
        "fn world w\n"
        "    action thrust box -1.0 1.0\n"
        "end\n"
        "end\n",
        1, "requires an `episode` block with `control_dt`", NULL);

    /* Episode without the required control_dt. */
    expect_("nodt",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        horizon 100\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "episode: missing required `control_dt <expr>`", NULL);

    /* Duplicate objective block. */
    expect_("dupobjective",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    objective\n"
        "        reward 1.0\n"
        "    end\n"
        "    objective\n"
        "        reward 2.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "duplicate `objective` block", NULL);

    /* Duplicate action name. */
    expect_("dupaction",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    action a box -1.0 1.0\n"
        "    action a discrete 4\n"
        "end\n"
        "end\n",
        1, "action `a`: duplicate action name", NULL);

    /* Reset with a state key outside the six scalar keys. */
    expect_("badkey",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "        reset craft.pos_w uniform(0.0, 1.0)\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "episode reset: unknown state key `pos_w`", NULL);

    /* Distribution expression outside its two valid positions. */
    expect_("distpos",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    let x: double = uniform(0.0, 1.0)\n"
        "end\n"
        "end\n",
        1, "only valid in astro_body attribute values and episode "
           "reset lines", NULL);

    /* Unknown name in the terminated-when expression. */
    expect_("unknownname",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when wibble > 1.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "terminated when: unknown name `wibble`", NULL);

    /* No horizon, no terminated-when: warns but checks clean. */
    expect_("neverends",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "    end\n"
        "end\n"
        "end\n",
        0, "the episode can never end", "error");

    /* on_step rejects stepping statements. */
    expect_("onstepstep",
        "form P\n"
        "fn world w\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        terminated when episode.steps > 10\n"
        "    end\n"
        "    on_step\n"
        "        step 0.1\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "on_step: `step` is not allowed inside an on_step block", NULL);

    /* Grammar 3.1 compatibility: a program with its own function
     * named `uniform` (used in an astro_body value and in an
     * ordinary call) is not an RL program and stays clean. */
    expect_("userfn",
        "form P\n"
        "    fn double uniform(double a, double b)\n"
        "        return a + b\n"
        "    end\n"
        "    fn world w\n"
        "        astro_body earth gm=uniform(1.0,2.0)\n"
        "        step 0.1\n"
        "    end\n"
        "end\n",
        0, NULL, "error");

    /* World scalar bindings in the objective and termination scope:
     * a top-level world `let`/`const` is readable in `terminated
     * when`, `reward`, and `terminal`, checked here through the
     * emitter (the environment emitter captures the values at
     * create). */
    expect_mode_("--emit", "wscalpos",
        "form WSCAL_POS\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let target_range: double = 7.2e6\n"
        "    const bonus: double = 2.5\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "        terminated when track_range > target_range\n"
        "    end\n"
        "    action push box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward bonus - track_range / target_range\n"
        "        terminal bonus * 2.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* A binding declared inside a nested block is not in that scope;
     * the emitter says so precisely instead of failing on an unknown
     * identifier. */
    expect_mode_("--emit", "wscalnested",
        "form WSCAL_NEG1\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    if 1.0 > 0.0\n"
        "        let hidden: double = 1.0\n"
        "    end\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward hidden - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`hidden` is declared inside a nested block", NULL);

    /* A non-scalar world binding is not readable there either. */
    expect_mode_("--emit", "wscalvec",
        "form WSCAL_NEG2\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let gains: vector = [1.0, 2.0, 3.0]\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward gains - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`gains` is not a scalar", NULL);

    /* A negative horizon is a checker error, not a silent unbounded
     * episode; the never-ends warning must not fire beside it. */
    expect_("neghorizon",
        "form NEG_H\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon -3\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward 0.0 - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`horizon` must be non-negative", "can never end");

    /* A fractional horizon has no meaning as a step count. */
    expect_("frachorizon",
        "form FRAC_H\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 2.5\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward 0.0 - track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "whole number of steps", NULL);

    /* An action name colliding with a derived observation component
     * is a KFL diagnostic, not a C++ redeclaration error. */
    expect_("nscollide",
        "form NS_COLLIDE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action track_range box -1.0 1.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with the `track_range` component", NULL);

    /* A channel name too long for the spec's 64-byte name entries is
     * refused at compile time, never silently truncated. The bound is
     * 64 less the longest derived suffix, `_range_rate`. */
    expect_("longchan",
        "form LONG_CHAN\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " channel_name_padded_out_to_be_conspicuously_longer_than_53_bytes\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "longer than 53 bytes", NULL);

    /* A world binding silently shadowed by an observation component
     * would read back the wrong value with no diagnostic; the
     * collision is an error instead. */
    expect_("wbindcollide",
        "form WB_COLLIDE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let track_range: double = 1.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward track_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with a world binding", NULL);

    /* Same rule between an action and a form argument. */
    expect_("argcollide",
        "form ARG_COLLIDE\n"
        "arg thrust default 9.0\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward thrust\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with form argument `thrust`", NULL);

    /* And between an action and a top-level world binding. */
    expect_("actbindcollide",
        "form AB_COLLIDE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    let thrust: double = 1.0\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward thrust\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "collides with a world binding", NULL);

    /* A binding inside a nested block was never readable in the
     * evaluators, so it may share an action's name freely. */
    expect_("nestedbindok",
        "form NB_OK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7350.0\n"
        "    if 1.0 > 0.0\n"
        "        let thrust: double = 1.0\n"
        "    end\n"
        "    episode\n"
        "        control_dt 0.1\n"
        "        horizon 20\n"
        "    end\n"
        "    action thrust box -1.0 1.0 default 0.0\n"
        "    observe craft from earth mode=geometric as track\n"
        "    objective\n"
        "        reward thrust\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "collides");

    /* ---- Body state inside on_step ------------------------------- */

    /* The shape of every case below: one world, one action, one
     * observation channel, with STATE substituted into the on_step
     * body or elsewhere. */
#define STATE_WORLD(WHERE_PREFIX, WHERE_ON_STEP, WHERE_OBJECTIVE) \
        "form RL_STATE\n" \
        "fn world w\n" \
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
        "    astro_body craft gm=1.0 parent=earth" \
        " pos_x=7.0e6 vel_y=7546.0\n" \
        WHERE_PREFIX \
        "    episode\n" \
        "        control_dt 1.0\n" \
        "        horizon 4\n" \
        "    end\n" \
        "    action a box -1.0 1.0 default 0.0\n" \
        "    on_step\n" \
        WHERE_ON_STEP \
        "    end\n" \
        "    observe craft from earth mode=geometric as trk\n" \
        "    objective\n" \
        "        reward " WHERE_OBJECTIVE "\n" \
        "    end\n" \
        "end\n" \
        "end\n"

    /* Every state key is readable and assignable in the block. */
    expect_("state_all_keys",
        STATE_WORLD("",
            "        craft.pos_x = craft.pos_x + a\n"
            "        craft.pos_y = craft.pos_y + a\n"
            "        craft.pos_z = craft.pos_z + a\n"
            "        craft.vel_x = craft.vel_x + a\n"
            "        craft.vel_y = craft.vel_y + a\n"
            "        craft.vel_z = craft.vel_z + a\n", "0.0"),
        0, NULL, "error");

    /* Outside the block the dot is not a name: in the world prefix it
     * stays the lexical error it was before the block existed. */
    expect_("state_in_prefix",
        STATE_WORLD("    craft.vel_x = 1.0\n", "        let z: double = a\n",
                    "0.0"),
        1, "unexpected character", NULL);

    /* An objective is told where body state lives instead. */
    expect_("state_in_objective",
        STATE_WORLD("", "        let z: double = a\n", "craft.vel_x"),
        1, "only inside an on_step block", NULL);

    /* A termination predicate is refused the same way. */
    expect_("state_in_terminated",
        "form RL_STATE_T\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        terminated when craft.vel_x > 1.0\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "only inside an on_step block", NULL);

    /* An ordinary fn body is not a stepping block either. */
    expect_("state_in_fn",
        "form RL_STATE_F\n"
        "fn double helper()\n"
        "    craft.vel_x = 1.0\n"
        "    return 1.0\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "unexpected character", NULL);

    /* An unknown body, an unknown key, and `episode.steps` each name
     * what they found. */
    expect_("state_unknown_body",
        STATE_WORLD("", "        rocket.vel_x = a\n", "0.0"),
        1, "no astro_body named `rocket`", NULL);

    expect_("state_unknown_key",
        STATE_WORLD("", "        craft.spin_x = a\n", "0.0"),
        1, "is not a body state key", NULL);

    expect_("state_episode_steps",
        STATE_WORLD("", "        let z: double = episode.steps\n", "0.0"),
        1, "not in on_step", NULL);

    /* The assigned expression must stay re-evaluable, so a call to a
     * builtin that is not marked pure is refused, whether it is made
     * directly or through a fn. */
    expect_("state_impure_builtin",
        STATE_WORLD("",
            "        craft.vel_x = a + astro_world_body_count(world)\n",
            "0.0"),
        1, "is not a pure builtin", NULL);

    expect_("state_impure_through_fn",
        "form RL_STATE_P\n"
        "fn double reach(world w)\n"
        "    return astro_world_body_count(w)\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.vel_x = a + reach(world)\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "reaches", NULL);

    /* A pure builtin is admitted, which is what makes the refusals
     * above about purity rather than about calls. */
    expect_("state_pure_builtin",
        STATE_WORLD("", "        craft.vel_x = sqrt(a * a) + 1.0\n", "0.0"),
        0, NULL, "error");

    /* The same rule on the three objective-side expressions. Each is
     * evaluated once per transition and has to reproduce on replay,
     * so an impure call is refused there for the reason it is refused
     * in a state assignment. One macro, three positions. */
#define OBJ_WORLD(WHERE_EPISODE, WHERE_OBJECTIVE) \
        "form RL_OBJ\n" \
        "fn world w\n" \
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
        "    astro_body craft gm=1.0 parent=earth" \
        " pos_x=7.0e6 vel_y=7546.0\n" \
        "    episode\n" \
        "        control_dt 1.0\n" \
        "        horizon 4\n" \
        WHERE_EPISODE \
        "    end\n" \
        "    observe craft from earth mode=geometric as trk\n" \
        "    objective\n" \
        WHERE_OBJECTIVE \
        "    end\n" \
        "end\n" \
        "end\n"

    expect_("reward_impure_builtin",
        OBJ_WORLD("",
            "        reward 0.0 - astro_world_body_count(trk_range)\n"),
        1, "the `reward` expression must be side-effect free", NULL);

    expect_("terminal_impure_builtin",
        OBJ_WORLD("",
            "        reward 0.0\n"
            "        terminal astro_world_body_count(trk_range)\n"),
        1, "the `terminal` expression must be side-effect free", NULL);

    expect_("terminated_when_impure_builtin",
        OBJ_WORLD(
            "        terminated when"
            " astro_world_body_count(trk_range) > 0.0\n",
            "        reward 0.0\n"),
        1, "the `terminated when` expression must be side-effect free",
        NULL);

    /* One indirection does not launder it: the sweep follows calls
     * into user fn bodies, and the diagnostic names the fn. */
    expect_("reward_impure_through_fn",
        "form RL_OBJ_P\n"
        "    fn double reach(double x)\n"
        "        return astro_world_body_count(x)\n"
        "    end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0 - reach(trk_range)\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "`fn reach` called here reaches", NULL);

    /* Pure builtins stay admitted in all three positions, which is
     * what keeps the refusals above about purity: the shipped
     * environments compute their reward with `abs` and `sqrt`. */
    expect_("objective_pure_builtins",
        OBJ_WORLD(
            "        terminated when sqrt(trk_range * trk_range) > 1.0\n",
            "        reward 0.0 - abs(trk_range)\n"
            "        terminal max(trk_range, 1.0)\n"),
        0, NULL, "error");

#undef OBJ_WORLD

    /* The stepping path performs no I/O, and a print one call away is
     * still I/O on the stepping path. */
    expect_("state_print_through_fn",
        "form RL_STATE_IO\n"
        "fn double chatty(double x)\n"
        "    print \"step\"\n"
        "    return x\n"
        "end\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        let z: double = chatty(a)\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "performs no I/O", NULL);

    /* A body named `episode` would make `episode.steps` ambiguous. */
    expect_("state_body_named_episode",
        "form RL_STATE_E\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body episode gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe episode from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "the name is taken by `episode.steps`", NULL);

    /* The channel-name bound is the spec's 64-byte name entry less
     * the longest derived suffix, so 53 accepted and 54 refused. */
    expect_("chan_name_53",
        "form RL_CHAN53\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " chan_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuv\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "longer than");

    expect_("chan_name_54",
        "form RL_CHAN54\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " chan_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvw\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "longer than 53 bytes", NULL);

    /* Both refusals above reach a plain compile, not only `--check`.
     * They guard published bytes: an over-long channel name is
     * truncated by the emitted spec writer rather than reported, and
     * a body named `episode` collides with `episode.steps` in the
     * expression scope, so a compile that accepted either would ship
     * the defect into an artifact.
     *
     * Two modes per case. `--emit` produces the C++ and nothing else,
     * so its exit code is the compiler front end's own answer. `-o`
     * additionally runs the host C++ compiler, which in a bare
     * environment fails for its own reasons, so that arm asserts on
     * the message: the refusal text present and `compile failed`
     * absent proves kflc refused before it ever reached the host
     * compiler. */
    expect_mode_("--emit", "chan_name_54_emit",
        "form RL_CHAN54E\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " chan_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvw\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "longer than 53 bytes", NULL);

    expect_mode_("-o /tmp/kflc_rl_compile.out", "chan_name_54_compile",
        "form RL_CHAN54O\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as"
        " chan_abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvw\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "longer than 53 bytes", "compile failed");

    expect_mode_("--emit", "body_episode_emit",
        "form RL_BODYEPE\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body episode gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe episode from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "the name is taken by `episode.steps`", NULL);

    expect_mode_("-o /tmp/kflc_rl_compile.out", "body_episode_compile",
        "form RL_BODYEPO\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body episode gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe episode from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "the name is taken by `episode.steps`", "compile failed");

    /* A program the checker accepts still compiles: the gate above
     * must not have made every compile run a refusal. `--emit` on the
     * positive fixture is the same front end that refused the two
     * cases above. */
    expect_mode_("--emit", "compile_path_positive",
        "form RL_OKC\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0 - abs(trk_range)\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* The seven attitude state keys, on the same three paths the six
     * translation keys take: an attribute, a reset target, and a read
     * or assignment inside on_step. */
    expect_("attitude_keys_all",
        "form RL_ATTK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"kflc_rl_att.k26asm\" parent=earth"
        " pos_x=7.0e6 vel_y=7546.0"
        " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"
        " omega_x=0.0 omega_y=0.0 omega_z=0.05\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        reset craft.quat_w normal(1.0, 0.0)\n"
        "        reset craft.quat_x normal(0.0, 0.0)\n"
        "        reset craft.quat_y normal(0.0, 0.0)\n"
        "        reset craft.quat_z normal(0.0, 0.0)\n"
        "        reset craft.omega_x uniform(-0.1, 0.1)\n"
        "        reset craft.omega_y uniform(-0.1, 0.1)\n"
        "        reset craft.omega_z uniform(-0.1, 0.1)\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.quat_w = craft.quat_w\n"
        "        craft.quat_x = craft.quat_x\n"
        "        craft.quat_y = craft.quat_y\n"
        "        craft.quat_z = craft.quat_z\n"
        "        craft.omega_x = craft.omega_x + a * 0.01\n"
        "        craft.omega_y = craft.omega_y\n"
        "        craft.omega_z = craft.omega_z\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* Attitude state on a body that cannot be advanced. Without an
     * assembly there is no inertia tensor, so the advance never runs
     * and the body would report the rate it was given while its
     * orientation stood still. Refused on all three paths that can
     * set it. */
    expect_("attitude_attr_without_assembly",
        "form RL_ATTNA\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0 omega_z=0.4\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "its attitude is never advanced", NULL);

    expect_("attitude_reset_without_assembly",
        "form RL_ATTNR\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        reset craft.omega_z uniform(-0.1, 0.1)\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "its attitude is never advanced", NULL);

    expect_("attitude_on_step_without_assembly",
        "form RL_ATTNO\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.omega_z = a\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "its attitude is never advanced", NULL);

    /* Translation keys are untouched by that rule: a body with no
     * assembly still takes position and velocity, which is every
     * program written before this surface existed. */
    expect_("translation_without_assembly",
        "form RL_ATTNT\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
        "    end\n"
        "    action a box -1.0 1.0 default 0.0\n"
        "    on_step\n"
        "        craft.vel_x = a * 0.01\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* A key that looks like one of them is still refused, and the
     * diagnostic names the whole set. */
    expect_("attitude_key_unknown",
        STATE_WORLD("", "        craft.quat_v = a\n", "0.0"),
        1, "is not a body state key", NULL);

    expect_("attitude_key_outside_on_step",
        "form RL_ATTO\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    craft.omega_z = 0.5\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, NULL, NULL);

    expect_("attitude_reset_unknown_key",
        "form RL_ATTR\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        reset craft.spin_z uniform(-0.1, 0.1)\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "unknown state key", NULL);

    /* The attitude observe form publishes seven channels under the
     * names the emitter and the checker both know, and they are
     * readable in the objective like any other channel. */
    expect_("attitude_observe",
        "form RL_ATTOBS\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft assembly=\"kflc_rl_att.k26asm\" parent=earth"
        " pos_x=7.0e6 vel_y=7546.0 omega_z=0.05\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    observe attitude of craft as att\n"
        "    objective\n"
        "        reward att_omega_z + att_quat_w * 0.0 + trk_range * 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    /* A channel the attitude form does not publish is still refused,
     * so the two suffix sets do not leak into each other. */
    expect_("attitude_observe_wrong_channel",
        "form RL_ATTOBSW\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe attitude of craft as att\n"
        "    objective\n"
        "        reward att_range\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "unknown name `att_range`", NULL);

    expect_("attitude_observe_unknown_body",
        "form RL_ATTOBSB\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "    end\n"
        "    observe attitude of ghost as att\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, NULL, NULL);

    /* The subdivision: accepted as a whole number, refused otherwise,
     * and the diagnostic says what it evaluates to. */
    expect_("substeps_ok",
        "form RL_SUBOK\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        substeps 8\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        0, NULL, "error");

    expect_("substeps_fractional",
        "form RL_SUBF\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        substeps 2.5\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "whole number of at least 1", NULL);

    expect_("substeps_zero",
        "form RL_SUBZ\n"
        "fn world w\n"
        "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
        "    astro_body craft gm=1.0 parent=earth"
        " pos_x=7.0e6 vel_y=7546.0\n"
        "    episode\n"
        "        control_dt 1.0\n"
        "        horizon 4\n"
        "        substeps 0\n"
        "    end\n"
        "    observe craft from earth mode=geometric as trk\n"
        "    objective\n"
        "        reward 0.0\n"
        "    end\n"
        "end\n"
        "end\n",
        1, "whole number of at least 1", NULL);

    /* The range-rate component is a readable channel like the four
     * beside it. */
    expect_("chan_range_rate",
        STATE_WORLD("", "        let z: double = a\n", "trk_range_rate"),
        0, NULL, "error");

#undef STATE_WORLD

    printf("test_rl_grammar: %d case(s) passed\n", n_pass);
    return 0;
}
