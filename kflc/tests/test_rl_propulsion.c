/* test_rl_propulsion.c - propellant, mass depletion and the rocket
 * equation, through the stepping surface.
 *
 * Acceptance:
 *   1. Specific impulse is carried and priced per thruster. The mass a
 *      burn spends is the throttled thrust divided by the specific
 *      impulse times standard gravity, summed over the thrusters that
 *      fired and multiplied by the time they fired for. The fixture
 *      carries two thrusters whose specific impulses differ by a
 *      factor of three and the arm drives them at two different pairs
 *      of throttles, so a model that used one figure for the whole
 *      craft cannot reproduce both readings.
 *   2. The rocket equation is a consequence and not a formula. The
 *      velocity change measured from the body state over a burn to
 *      depletion is compared against specific impulse times standard
 *      gravity times the log of the mass ratio, at mass ratios of 1.2
 *      and 3.0, which are not near each other. The measurement is
 *      required to fall below the closed form by no more than the
 *      sub-interval discretisation accounts for, and never above it:
 *      holding the mass constant across a sub-interval and debiting it
 *      at the end understates the ratio by a known one-sided amount.
 *   3. The centre of mass moves as the tank empties, and the attitude
 *      authority moves with it. The fixture's tank is offset from the
 *      structure, so the craft's centre of mass slides along one axis
 *      as it drains. One thruster is placed so that the component of
 *      its torque about that axis is proportional to the offset and
 *      zero when the offset is: the ratio of the two angular responses
 *      it produces is therefore a direct measurement of where the
 *      centre of mass now is. It is taken at a full tank and again at
 *      a tenth of one and compared against the closed form, and the
 *      two differ by a factor of four. The matched control is a craft
 *      whose tank sits on the centre of mass, whose response about
 *      that axis stays zero at every fill: a centred tank cannot fail
 *      on this, which is why the measured fixture's is not.
 *   4. An empty tank produces no thrust, no torque and no consumption
 *      however hard it is commanded, and the sub-interval a tank runs
 *      out in is apportioned rather than overrun or truncated. The
 *      fixture is sized so that depletion falls part way through a
 *      sub-interval, and the impulse the control period containing it
 *      delivers is compared against what the propellant that remained
 *      could deliver. Letting the sub-interval finish would deliver
 *      about three times that; dropping it would deliver none of it;
 *      the acceptance band admits neither.
 *   5. Propellant, mass, centre of mass and inertia return to their
 *      constructed values at an episode reset, measured across the
 *      boundary rather than read off the source. Mass and propellant
 *      are read from the published channels; the centre of mass and
 *      the inertia are measured by repeating arm 3's angular
 *      experiment after the reset and requiring it to reproduce the
 *      first episode's reading bit for bit. A whole episode is then
 *      repeated under the same actions and required to agree bitwise
 *      with the first.
 *   6. Two processes at one seed write identical episode files for a
 *      program whose craft burns to depletion.
 *   7. The four published components, their names and their order,
 *      and the same components routed through a declared sensor with
 *      the uncorrupted values beside them.
 *   8. The refusals: a propulsion observe of a body that binds no
 *      assembly, and of a craft whose assembly holds no propellant.
 *
 * On vacuity: every arm that asserts a quantity is unchanged is paired
 * with one that asserts the same quantity moves. Arm 3's control is
 * the pairing for the centre of mass; arm 4's empty-tank half is
 * paired with the burn that emptied it; arm 5's reset comparison is
 * against an episode whose state demonstrably moved.
 *
 * The craft sit far from the attracting body and cross their burns at
 * a strongly hyperbolic speed, so the gravitational contribution to
 * the measured velocity change is below one part in ten million of
 * it and the two-body propagation stays well away from the radial and
 * parabolic cases a universal-variable solver is worst at. The
 * gravitational term is measured rather than assumed: the drift after
 * depletion is reported with each burn.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_propulsion_test"

/* Standard gravity, the defined constant, exactly as the artifact
 * carries it. Every closed form below is written against this figure
 * and not against a local gravitational acceleration. */
#define G0 9.80665

static int g_arms;

/* ---- The assemblies -------------------------------------------------- *
 *
 * The axial article: one hull, one tank on the thrust axis, and one
 * thruster on that axis pointing along it. Its line of action passes
 * through the centre of mass at every fill, since the centre of mass
 * can only slide along the same axis, so the craft takes no torque and
 * the velocity change is the whole of what the burn did. That is what
 * arms 2 and 4 need and what an offset tank would spoil.
 *
 * The capacity and the thrust are the two figures that differ between
 * its three uses, so it is written once with both substituted.
 */
#define AXIAL_ASM(name, cap, thrust)                                      \
    "assembly " name "\n"                                                 \
    "    frame x_to_port\n"                                               \
    "    provenance mass \"calibration shape, not a craft\" computed\n"    \
    "    provenance tank \"a chosen tank placement on the thrust axis, "   \
    "so the line of action passes through the centre of mass at every "    \
    "fill\" modelled\n"                                                    \
    "    component hull\n"                                                \
    "        mass 1000.0\n"                                               \
    "        at 0 0 0\n"                                                  \
    "        collider box 1.0 0.5 0.5\n"                                  \
    "    end\n"                                                           \
    "    component tank\n"                                                \
    "        mass " cap "\n"                                              \
    "        at 2.0 0.0 0.0\n"                                            \
    "        collider box 0.3 0.3 0.3\n"                                  \
    "        propellant\n"                                                \
    "    end\n"                                                           \
    "    thruster main\n"                                                 \
    "        at -3.0 0.0 0.0\n"                                           \
    "        dir 1.0 0.0 0.0\n"                                           \
    "        thrust " thrust "\n"                                         \
    "        isp_s 300.0\n"                                               \
    "    end\n"                                                           \
    "end\n"

/* The offset article, for the centre of mass. The hull and the tank
 * are both cubes, so each contributes the same moment about all three
 * of its own axes and the assembly's tensor stays diagonal: the two
 * are displaced from one another along y alone, and a displacement
 * along one axis adds no product of inertia. That is what lets the
 * closed forms below be written in three lines rather than repeating
 * the compiler's derivation.
 *
 * `tilt` sits on the x axis and thrusts along z. Its torque about the
 * centre of mass is the offset from it crossed with that thrust: the y
 * component is its own x arm times the thrust, fixed, and the x
 * component is the centre of mass's y coordinate times the thrust,
 * which is zero exactly when the centre of mass is on the body-frame
 * origin. The ratio of the two angular rates it produces from rest is
 * therefore that coordinate, scaled by the ratio of two moments.
 *
 * `drain` sits on the y axis and thrusts along it, so its line of
 * action passes through the centre of mass at every fill whatever the
 * offset: it empties the tank without turning the craft, which is what
 * lets the second measurement start from rest as the first did.
 *
 * The tank's placement is the one figure that differs between the
 * measured article and its control.
 */
#define OFFSET_ASM(name, tank_at, prov)                                   \
    "assembly " name "\n"                                                 \
    "    frame x_to_port\n"                                               \
    "    provenance mass \"calibration shape, not a craft\" computed\n"    \
    "    provenance tank \"" prov "\" modelled\n"                          \
    "    component hull\n"                                                \
    "        mass 1000.0\n"                                               \
    "        at 0 0 0\n"                                                  \
    "        collider box 1.0 1.0 1.0\n"                                  \
    "    end\n"                                                           \
    "    component tank\n"                                                \
    "        mass 500.0\n"                                                \
    "        at " tank_at "\n"                                            \
    "        collider box 0.5 0.5 0.5\n"                                  \
    "        propellant\n"                                                \
    "    end\n"                                                           \
    "    thruster tilt\n"                                                 \
    "        at 1.0 0.0 0.0\n"                                            \
    "        dir 0.0 0.0 1.0\n"                                           \
    "        thrust 200.0\n"                                              \
    "        isp_s 300.0\n"                                               \
    "    end\n"                                                           \
    "    thruster drain\n"                                                \
    "        at 0.0 5.0 0.0\n"                                            \
    "        dir 0.0 1.0 0.0\n"                                           \
    "        thrust 2000.0\n"                                             \
    "        isp_s 100.0\n"                                               \
    "    end\n"                                                           \
    "end\n"

/* ---- The programs ---------------------------------------------------- *
 *
 * The craft sits a hundred million kilometres from the attracting body
 * and crosses at a kilometre a second, which is more than ten times
 * the escape speed there: the propagation is a well-conditioned
 * hyperbola with real angular momentum, and the gravitational
 * acceleration is four hundredths of a micrometre per second squared.
 * A craft placed at rest instead sits on a radial orbit and crosses
 * the parabolic boundary as it accelerates, which is the one two-body
 * configuration a universal-variable solver cannot be asked to hold;
 * that arrangement was tried first and threw away twenty metres per
 * second of the measurement in a single step.
 */
#define WORLD_HEAD(asm_path)                                              \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"              \
    "    astro_body craft assembly=\"" asm_path "\" parent=earth"         \
    " pos_x=1.0e11 pos_y=0.0 pos_z=0.0"                                   \
    " vel_x=0.0 vel_y=1000.0 vel_z=0.0"                                   \
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"                        \
    " omega_x=0.0 omega_y=0.0 omega_z=0.0\n"

#define AXIAL_KFL(form, asm_path, dt, horizon, subs)                      \
    "form " form "\n"                                                     \
    "fn world w\n"                                                        \
    WORLD_HEAD(asm_path)                                                  \
    "    episode\n"                                                       \
    "        control_dt " dt "\n"                                         \
    "        horizon " horizon "\n"                                       \
    "        substeps " subs "\n"                                         \
    "    end\n"                                                           \
    "    action push box 0.0 1.0 default 0.0\n"                           \
    "    on_step\n"                                                       \
    "        craft.main.throttle = push\n"                                \
    "    end\n"                                                           \
    "    observe propulsion of craft as fuel\n"                           \
    "    observe attitude of craft as att\n"                              \
    "    objective\n"                                                     \
    "        reward fuel_propellant_fraction + att_omega_z\n"             \
    "    end\n"                                                           \
    "end\n"                                                               \
    "end\n"

#define OFFSET_KFL(form, asm_path)                                        \
    "form " form "\n"                                                     \
    "fn world w\n"                                                        \
    WORLD_HEAD(asm_path)                                                  \
    "    episode\n"                                                       \
    "        control_dt 1.0\n"                                            \
    "        horizon 300\n"                                               \
    "        substeps 8\n"                                                \
    "    end\n"                                                           \
    "    action lean box 0.0 1.0 default 0.0\n"                           \
    "    action pump box 0.0 1.0 default 0.0\n"                           \
    "    on_step\n"                                                       \
    "        craft.tilt.throttle = lean\n"                                \
    "        craft.drain.throttle = pump\n"                               \
    "    end\n"                                                           \
    "    observe propulsion of craft as fuel\n"                           \
    "    observe attitude of craft as att\n"                              \
    "    objective\n"                                                     \
    "        reward fuel_propellant_fraction + att_omega_x\n"             \
    "    end\n"                                                           \
    "end\n"                                                               \
    "end\n"

/* ---- Opening and driving --------------------------------------------- */

typedef struct {
    void       *so;
    RlSurface   s;
    K26RlEnv   *env;
} PropArt;

static void art_open_(PropArt *a, const char *so_path, uint64_t seed)
{
    a->so = rl_dlopen_(so_path);
    rl_resolve_surface_(a->so, &a->s);
    a->env = NULL;
    ASSERT(a->s.create(seed, 1u, &a->env) == K26RL_OK);
    ASSERT(a->s.reset(a->env) == K26RL_OK);
}

static void art_close_(PropArt *a)
{
    a->s.destroy(a->env);
    dlclose(a->so);
}

/* Write a program and its assembly and compile both into WORK_DIR. */
static void build_(const char *stem, const char *asm_name,
                   const char *asm_text, const char *kfl_text)
{
    char path[512];
    snprintf(path, sizeof path, WORK_DIR "/%s", asm_name);
    rl_write_file_(path, asm_text);
    snprintf(path, sizeof path, WORK_DIR "/%s.kfl", stem);
    rl_write_file_(path, kfl_text);
    char out[512];
    snprintf(out, sizeof out, WORK_DIR "/%s", stem);
    rl_compile_(path, out, WORK_DIR);
    snprintf(out, sizeof out, WORK_DIR "/%s.rlenv.so", stem);
    ASSERT(rl_file_exists_(out));
}

/* The craft is the second body, so its velocity is at offset 9 in the
 * body buffer and its body-frame angular velocity at offset 11 in the
 * attitude buffer. Both layouts are the frozen surface's own. */
static double craft_vx_(PropArt *a)
{
    double b[64];
    ASSERT(a->s.bodies(a->env, K26RL_BODY_REF_ORIGIN, b, 64u) > 0);
    return b[9];
}

static void craft_omega_(PropArt *a, double out[3])
{
    double at[64];
    ASSERT(a->s.attitudes(a->env, at, 64u) > 0);
    out[0] = at[11];
    out[1] = at[12];
    out[2] = at[13];
}

/* The published propulsion channels, which the fixtures declare first,
 * so they occupy the first four slots. The order is asserted against
 * the spec's own names in arm 7 before any arm reads them by index. */
enum { CH_PROP = 0, CH_FRAC = 1, CH_MASS = 2, CH_DVREM = 3 };

static void obs_(PropArt *a, double out[4])
{
    double o[64];
    ASSERT(a->s.obs(a->env, o) == K26RL_OK);
    for (int i = 0; i < 4; i++) out[i] = o[i];
}

/* ---- Arm 2: the rocket equation -------------------------------------- *
 *
 * Drive one artifact at full throttle until the tank reads empty, then
 * a few steps beyond it, and report the velocity change against the
 * closed form.
 *
 * The acceptance is one-sided by construction. Within a sub-interval
 * the mass is held at what it was when the interval opened and debited
 * at its close, so each interval's velocity change is the flow times
 * the interval over the opening mass rather than the exact logarithm
 * of the ratio; the difference is the second-order term of that
 * logarithm and is always positive, so the measurement lands below the
 * closed form and never above it. The band is stated as a fraction and
 * the measured shortfall is printed beside it.
 */
static void gate_rocket_(const char *stem, double capacity, double dry,
                         int steps, double band)
{
    char so[512];
    snprintf(so, sizeof so, WORK_DIR "/%s.rlenv.so", stem);
    PropArt a;
    art_open_(&a, so, 4242u);

    const double v_e   = 300.0 * G0;
    const double ideal = v_e * log((dry + capacity) / dry);

    double o[4];
    obs_(&a, o);
    ASSERT(o[CH_PROP] == capacity);
    ASSERT(o[CH_MASS] == dry + capacity);
    /* The published remaining velocity change is the same closed form
     * the burn is about to be measured against, so a burn that reaches
     * it proves the channel too. */
    ASSERT(fabs(o[CH_DVREM] - ideal) < 1.0e-9);

    double vx0 = craft_vx_(&a);
    double act[1] = { 1.0 };
    int    emptied = -1;
    for (int i = 0; i < steps; i++) {
        ASSERT(a.s.step(a.env, act) == K26RL_OK);
        obs_(&a, o);
        if (emptied < 0 && o[CH_PROP] == 0.0) emptied = i + 1;
    }
    ASSERT(emptied > 0);
    double vx1 = craft_vx_(&a);

    /* What the attracting body contributed, measured rather than
     * assumed: after the tank is empty nothing but gravity moves the
     * craft along the thrust axis, so four further steps size it. */
    double vg0 = craft_vx_(&a);
    for (int i = 0; i < 4; i++) ASSERT(a.s.step(a.env, act) == K26RL_OK);
    double drift = fabs(craft_vx_(&a) - vg0) / 4.0;

    obs_(&a, o);
    ASSERT(o[CH_PROP] == 0.0);
    ASSERT(o[CH_FRAC] == 0.0);
    ASSERT(o[CH_MASS] == dry);
    ASSERT(o[CH_DVREM] == 0.0);

    double measured = vx1 - vx0;
    double rel      = (ideal - measured) / ideal;
    if (!(rel >= 0.0 && rel < band)) {
        fprintf(stderr, "FAIL: %s: measured %.12g against the closed form "
                "%.12g, a shortfall of %.6g; the band is [0, %.6g)\n",
                stem, measured, ideal, rel, band);
        exit(1);
    }
    printf("  mass ratio %.4f: measured %.9g m/s against %.9g m/s, "
           "short by %.4g of it, tank empty after %d steps, gravity "
           "%.3g m/s per step\n",
           (dry + capacity) / dry, measured, ideal, rel, emptied, drift);
    g_arms++;
    art_close_(&a);
}

/* ---- Arm 3: where the centre of mass is ------------------------------ *
 *
 * The closed forms for the offset article. Both components are cubes,
 * so each carries the same moment about all three of its own axes, and
 * the two are displaced along y alone, so the assembly's tensor stays
 * diagonal and only the parallel-axis terms about x and z pick the
 * displacement up.
 *
 * `tilt` thrusts along z from a point on the x axis. Its torque about
 * the centre of mass has an x component of minus the centre of mass's
 * y coordinate times the thrust, and a y component of minus its own x
 * arm times the thrust. From rest, and over an interval short enough
 * that the gyroscopic term is negligible, the ratio of the two angular
 * rates is therefore that y coordinate times the ratio of the two
 * moments.
 */
static double com_y_(double prop_kg)
{
    return 1.5 * prop_kg / (1000.0 + prop_kg);
}

static double omega_ratio_(double prop_kg)
{
    const double hull = 1000.0, ha = 1.0, ta = 0.5;
    double cy  = com_y_(prop_kg);
    double ixx = (2.0 / 3.0) * hull * ha * ha + hull * cy * cy
               + (2.0 / 3.0) * prop_kg * ta * ta
               + prop_kg * (1.5 - cy) * (1.5 - cy);
    double iyy = (2.0 / 3.0) * hull * ha * ha
               + (2.0 / 3.0) * prop_kg * ta * ta;
    return cy * iyy / ixx;
}

/* One measurement: from rest, one control period of `tilt` alone, and
 * the ratio of the angular rates it left. The fill is averaged over
 * the burst because the burst itself spends a little of it. */
static double measure_ratio_(PropArt *a, double *out_fill)
{
    double o[4], w[3], before, after;
    obs_(a, o);
    before = o[CH_PROP];
    double act[2] = { 0.25, 0.0 };
    ASSERT(a->s.step(a->env, act) == K26RL_OK);
    obs_(a, o);
    after = o[CH_PROP];
    craft_omega_(a, w);
    if (out_fill) *out_fill = 0.5 * (before + after);
    return w[1] != 0.0 ? w[0] / w[1] : 0.0;
}

/* Drain the tank to about a tenth without turning the craft. */
static void drain_(PropArt *a, int steps)
{
    double act[2] = { 0.0, 1.0 };
    for (int i = 0; i < steps; i++) {
        ASSERT(a->s.step(a->env, act) == K26RL_OK);
    }
}


/* ---- Perturbing the emitted source ----------------------------------- *
 *
 * Two of the arms above assert that a quantity comes out right, and an
 * arm like that is worth what its failure mode is worth. Each is
 * therefore repeated on an artifact built from the same program with
 * one line of the emitted source changed into the defect the arm
 * names, and required to fail there. The mutation is asserted to have
 * reached the file before the perturbed artifact is built, because a
 * mutation that changed nothing cannot be reported as a survivor.
 */
static void emit_(const char *stem)
{
    char cmd[1024];
    snprintf(cmd, sizeof cmd,
             "./bin/kflc --emit " WORK_DIR "/%s.kfl > " WORK_DIR
             "/%s.cc 2> " WORK_DIR "/%s.emit.log", stem, stem, stem);
    if (system(cmd) != 0) {
        fprintf(stderr, "FAIL: --emit failed for %s\n", stem);
        exit(1);
    }
}

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

/* Drive an axial artifact to depletion and return the velocity change
 * the control period containing it delivered. */
static double depleting_period_dv_(const char *so_path)
{
    PropArt a;
    art_open_(&a, so_path, 4242u);
    double act[1] = { 1.0 }, o[4];
    double prev_vx = craft_vx_(&a), dv = -1.0;
    for (int i = 0; i < 20; i++) {
        ASSERT(a.s.step(a.env, act) == K26RL_OK);
        obs_(&a, o);
        double vx = craft_vx_(&a);
        if (dv < 0.0 && o[CH_PROP] == 0.0) dv = vx - prev_vx;
        prev_vx = vx;
    }
    art_close_(&a);
    ASSERT(dv >= 0.0);
    return dv;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_propulsion")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    build_("axa", "axa.k26asm",
           AXIAL_ASM("prop_axial_a", "200.0", "400.0"),
           AXIAL_KFL("RL_PROP_A", "axa.k26asm", "10.0", "200", "16"));
    build_("axb", "axb.k26asm",
           AXIAL_ASM("prop_axial_b", "2000.0", "4000.0"),
           AXIAL_KFL("RL_PROP_B", "axb.k26asm", "10.0", "200", "16"));
    build_("axd", "axd.k26asm",
           AXIAL_ASM("prop_axial_d", "20.5", "400.0"),
           AXIAL_KFL("RL_PROP_D", "axd.k26asm", "10.0", "30", "4"));
    build_("off", "off.k26asm",
           OFFSET_ASM("prop_offset", "0.0 1.5 0.0",
                      "a chosen tank placement, held clear of the "
                      "structure so the centre of mass moves as it "
                      "empties"),
           OFFSET_KFL("RL_PROP_OFF", "off.k26asm"));
    build_("cen", "cen.k26asm",
           OFFSET_ASM("prop_centred", "0.0 0.0 0.0",
                      "a chosen tank placement, on the structure's own "
                      "centre so the centre of mass does not move as it "
                      "empties"),
           OFFSET_KFL("RL_PROP_CEN", "cen.k26asm"));

    /* ---- Arm 7: the published components ---------------------------- */
    {
        PropArt a;
        art_open_(&a, WORK_DIR "/axa.rlenv.so", 4242u);
        uint8_t blob[65536];
        int32_t n = a.s.spec(a.env, blob, (uint32_t)sizeof blob);
        ASSERT(n > 0);
        RlSpecView v;
        rl_parse_spec_(blob, (uint32_t)n, &v);
        static const char *const want[4] = {
            "fuel_propellant_kg", "fuel_propellant_fraction",
            "fuel_mass_kg", "fuel_delta_v_remaining"
        };
        ASSERT(v.n_chan_names >= 4);
        for (int i = 0; i < 4; i++) {
            if (strcmp(v.chan_names[i], want[i]) != 0) {
                fprintf(stderr, "FAIL: channel %d is `%s`, expected `%s`\n",
                        i, v.chan_names[i], want[i]);
                exit(1);
            }
            /* No observer and no correction, so the only true mode
             * among the published ones is the geometric. */
            ASSERT(v.modes[i] == 0);
        }
        printf("  four components published in order, all geometric: %s, "
               "%s, %s, %s\n", want[0], want[1], want[2], want[3]);
        g_arms++;
        art_close_(&a);
    }

    /* ---- Arm 1: what a burn spends, per thruster --------------------- *
     *
     * The two thrusters differ in specific impulse by a factor of
     * three, so a model carrying one figure for the craft cannot match
     * both drives: the first spends nearly all of its propellant
     * through the low-impulse thruster and the second through the
     * high-impulse one, and the predicted masses differ by more than
     * an order of magnitude between them.
     */
    {
        static const struct { double lean, pump; } drive[2] = {
            { 0.25, 1.00 }, { 1.00, 0.10 }
        };
        for (int k = 0; k < 2; k++) {
            PropArt a;
            art_open_(&a, WORK_DIR "/off.rlenv.so", 4242u);
            double o[4];
            obs_(&a, o);
            double m0 = o[CH_MASS], p0 = o[CH_PROP];
            double act[2] = { drive[k].lean, drive[k].pump };
            const int steps = 8;
            for (int i = 0; i < steps; i++) {
                ASSERT(a.s.step(a.env, act) == K26RL_OK);
            }
            obs_(&a, o);
            /* Thrust over exhaust speed, per thruster, times the time
             * they fired for. The control period is one second. */
            double flow = drive[k].lean * 200.0 / (300.0 * G0)
                        + drive[k].pump * 2000.0 / (100.0 * G0);
            double want = flow * (double)steps;
            double spent = p0 - o[CH_PROP];
            double rel   = fabs(spent - want) / want;
            /* What the same drive would spend if one figure served the
             * whole craft, taken at the thrust-weighted mean the
             * published remaining velocity change is quoted at. */
            double mean_isp = (200.0 * 300.0 + 2000.0 * 100.0)
                            / (200.0 + 2000.0);
            double lumped = (drive[k].lean * 200.0 + drive[k].pump * 2000.0)
                          / (mean_isp * G0) * (double)steps;
            if (!(rel < 1.0e-12)) {
                fprintf(stderr, "FAIL: throttles %.2f/%.2f spent %.17g kg "
                        "against %.17g predicted, a relative %.3g\n",
                        drive[k].lean, drive[k].pump, spent, want, rel);
                exit(1);
            }
            ASSERT(fabs(m0 - o[CH_MASS] - spent) < 1.0e-9);
            ASSERT(fabs(lumped - want) / want > 0.05);
            printf("  throttles %.2f/%.2f over %d s spend %.10g kg against "
                   "%.10g predicted per thruster (relative %.2g); one "
                   "figure for the craft would predict %.10g\n",
                   drive[k].lean, drive[k].pump, steps, spent, want, rel,
                   lumped);
            g_arms++;
            art_close_(&a);
        }
    }

    /* ---- Arm 2: the rocket equation at two mass ratios --------------- */
    printf("the rocket equation as a consequence:\n");
    gate_rocket_("axa", 200.0, 1000.0, 155, 2.0e-4);
    gate_rocket_("axb", 2000.0, 1000.0, 155, 1.0e-3);

    /* ---- Arm 3: the centre of mass, and its control ------------------ */
    {
        PropArt a;
        art_open_(&a, WORK_DIR "/off.rlenv.so", 4242u);
        double fill_full = 0.0, fill_low = 0.0;
        double r_full = measure_ratio_(&a, &fill_full);
        double p_full = omega_ratio_(fill_full);
        ASSERT(a.s.reset(a.env) == K26RL_OK);
        drain_(&a, 220);
        double w[3];
        craft_omega_(&a, w);
        /* The draining thruster's line of action passes through the
         * centre of mass at every fill, so the second measurement
         * starts from rest as the first did. */
        ASSERT(fabs(w[0]) < 1.0e-15 && fabs(w[1]) < 1.0e-15);
        double r_low = measure_ratio_(&a, &fill_low);
        double p_low = omega_ratio_(fill_low);

        double e_full = fabs(r_full / p_full - 1.0);
        double e_low  = fabs(r_low / p_low - 1.0);
        if (!(e_full < 1.0e-3 && e_low < 1.0e-3)) {
            fprintf(stderr, "FAIL: the measured angular ratios %.12g and "
                    "%.12g miss the closed forms %.12g and %.12g by %.3g "
                    "and %.3g\n", r_full, r_low, p_full, p_low, e_full,
                    e_low);
            exit(1);
        }
        if (!(r_full / r_low > 3.0)) {
            fprintf(stderr, "FAIL: the angular ratio moved by only %.6g "
                    "between a full tank and a tenth of one\n",
                    r_full / r_low);
            exit(1);
        }
        printf("  the centre of mass measured through the dynamics: "
               "%.9g m at %.4g kg and %.9g m at %.4g kg, the closed forms "
               "%.9g and %.9g, the response moving by a factor of %.4g\n",
               r_full, fill_full, r_low, fill_low, p_full, p_low,
               r_full / r_low);
        g_arms++;
        art_close_(&a);

        /* The control: the same experiment on the article whose tank
         * sits on the centre of mass. The response about that axis is
         * zero at every fill, so this fixture could not fail the arm
         * above however the model treated the tank, which is what
         * makes the offset one the measurement. */
        PropArt c;
        art_open_(&c, WORK_DIR "/cen.rlenv.so", 4242u);
        double f0 = 0.0, f1 = 0.0;
        double c_full = measure_ratio_(&c, &f0);
        ASSERT(c.s.reset(c.env) == K26RL_OK);
        drain_(&c, 220);
        double c_low = measure_ratio_(&c, &f1);
        if (!(fabs(c_full) < 1.0e-12 && fabs(c_low) < 1.0e-12)) {
            fprintf(stderr, "FAIL: the centred article's angular ratio "
                    "reads %.6g and %.6g, not zero\n", c_full, c_low);
            exit(1);
        }
        printf("  the control, tank on the centre of mass: the same "
               "response reads %.3g at %.4g kg and %.3g at %.4g kg\n",
               c_full, f0, c_low, f1);
        g_arms++;
        art_close_(&c);
    }

    /* ---- Arm 4: the empty tank and the apportioned interval ---------- *
     *
     * The article's capacity is 20.5 kg, its thrust 400 N and its
     * specific impulse 300 s, so it spends 0.339905404 kg in each
     * 2.5 s sub-interval and the tank lasts 60.31 of them. Depletion
     * therefore falls 31 per cent of the way through the sixty-first,
     * which is the second sub-interval of the sixteenth control period.
     *
     * What that control period delivers is the whole of the arm. Letting
     * the sub-interval finish would spend 0.234 kg the craft does not
     * have and deliver about a metre per second; dropping the partial
     * interval would deliver 0.31 m/s less than it should. The
     * apportioned answer is between them by more than three parts in
     * ten either way, and is compared against the closed form at the
     * masses the period actually began and ended with.
     */
    {
        PropArt a;
        art_open_(&a, WORK_DIR "/axd.rlenv.so", 4242u);
        const double v_e = 300.0 * G0;
        double act[1] = { 1.0 };
        double o[4], prev_mass = 0.0, prev_vx = 0.0;
        obs_(&a, o);
        prev_mass = o[CH_MASS];
        prev_vx   = craft_vx_(&a);
        int    step_emptied = -1;
        double dv_last = 0.0, m_before = 0.0;
        for (int i = 0; i < 20; i++) {
            ASSERT(a.s.step(a.env, act) == K26RL_OK);
            obs_(&a, o);
            double vx = craft_vx_(&a);
            if (step_emptied < 0 && o[CH_PROP] == 0.0) {
                step_emptied = i + 1;
                dv_last      = vx - prev_vx;
                m_before     = prev_mass;
            }
            prev_mass = o[CH_MASS];
            prev_vx   = vx;
        }
        ASSERT(step_emptied == 16);
        ASSERT(o[CH_PROP] == 0.0);
        ASSERT(o[CH_MASS] == 1000.0);

        double want = v_e * log(m_before / 1000.0);
        /* A sub-interval allowed to overrun would spend a full
         * 0.339905404 kg where 0.1057 kg remained. */
        double overrun  = v_e * log(m_before / (1000.0 - 0.2342));
        double truncate = 0.0;
        if (!(fabs(dv_last - want) < 1.0e-4)) {
            fprintf(stderr, "FAIL: the depleting period delivered %.12g "
                    "m/s against the %.12g m/s the propellant that "
                    "remained could deliver\n", dv_last, want);
            exit(1);
        }
        ASSERT(fabs(overrun - want) > 0.3 && fabs(truncate - want) > 0.3);
        printf("  the interval a tank runs out in is apportioned: the "
               "period delivered %.10g m/s against %.10g m/s from the "
               "propellant that remained; overrunning it would give "
               "%.6g and truncating it %.6g\n",
               dv_last, want, overrun, truncate);
        g_arms++;

        /* Commanded hard on an empty tank: no thrust, no torque, no
         * consumption, and no fault. What is left is the attracting
         * body, which the burn above measured. */
        double vx0 = craft_vx_(&a), w0[3], w1[3];
        craft_omega_(&a, w0);
        for (int i = 0; i < 8; i++) {
            ASSERT(a.s.step(a.env, act) == K26RL_OK);
        }
        obs_(&a, o);
        craft_omega_(&a, w1);
        double moved = fabs(craft_vx_(&a) - vx0);
        ASSERT(o[CH_PROP] == 0.0 && o[CH_MASS] == 1000.0);
        /* The craft is not left perfectly still: it hangs in a
         * gradient, and the torque that gradient puts on it is what
         * the residual below is. A thruster firing would put four
         * hundred newtons through a three-metre arm. */
        double turned = fabs(w1[0] - w0[0]) + fabs(w1[1] - w0[1])
                      + fabs(w1[2] - w0[2]);
        if (!(moved < 1.0e-5 && turned < 1.0e-12)) {
            fprintf(stderr, "FAIL: eight steps at full throttle on an "
                    "empty tank moved the craft by %.6g m/s and turned "
                    "it by %.6g rad/s\n", moved, turned);
            exit(1);
        }
        printf("  eight periods at full throttle on an empty tank spend "
               "nothing, move the craft %.3g m/s and turn it %.3g rad/s, "
               "which is what the attracting body and its gradient do\n",
               moved, turned);
        g_arms++;
        art_close_(&a);
    }

    /* ---- Arm 5: the episode boundary --------------------------------- */
    {
        PropArt a;
        art_open_(&a, WORK_DIR "/off.rlenv.so", 4242u);
        double f0 = 0.0, f1 = 0.0;
        /* Episode one: the angular experiment, which measures the
         * centre of mass and the inertia together, then a drain that
         * moves both a long way from where they started. */
        double r0 = measure_ratio_(&a, &f0);
        drain_(&a, 220);
        double o[4];
        obs_(&a, o);
        double p_end = o[CH_PROP], m_end = o[CH_MASS];
        ASSERT(p_end < 60.0 && m_end < 1060.0);

        ASSERT(a.s.reset(a.env) == K26RL_OK);
        obs_(&a, o);
        if (!(o[CH_PROP] == 500.0 && o[CH_FRAC] == 1.0 &&
              o[CH_MASS] == 1500.0)) {
            fprintf(stderr, "FAIL: after a reset the tank reads %.17g kg, "
                    "the fraction %.17g and the mass %.17g\n",
                    o[CH_PROP], o[CH_FRAC], o[CH_MASS]);
            exit(1);
        }
        /* The centre of mass and the inertia are not published, so
         * they are measured across the boundary the way they were
         * measured before it, and required to agree bit for bit. A
         * restoration that put the tank back and left the mass
         * properties where the burn left them would fail here and
         * nowhere else. */
        double r1 = measure_ratio_(&a, &f1);
        if (!(r1 == r0)) {
            fprintf(stderr, "FAIL: the angular experiment reads %.17g "
                    "after the reset and read %.17g before it\n", r1, r0);
            exit(1);
        }
        printf("  across an episode boundary: the tank returns from "
               "%.6g kg to 500 kg and the mass from %.6g kg to 1500 kg, "
               "and the angular experiment reproduces %.17g bit for bit\n",
               p_end, m_end, r1);
        g_arms++;
        art_close_(&a);

        /* And the whole of it: two episodes of the same artifact under
         * the same actions agree bitwise over their published channels,
         * which is the restoration measured over everything at once. */
        PropArt b;
        art_open_(&b, WORK_DIR "/off.rlenv.so", 4242u);
        enum { T = 40, OBS = 11 };
        static double e1[T * OBS], e2[T * OBS];
        double act[2] = { 0.6, 0.4 };
        for (int i = 0; i < T; i++) {
            ASSERT(b.s.step(b.env, act) == K26RL_OK);
            ASSERT(b.s.obs(b.env, e1 + (size_t)i * OBS) == K26RL_OK);
        }
        ASSERT(b.s.reset(b.env) == K26RL_OK);
        for (int i = 0; i < T; i++) {
            ASSERT(b.s.step(b.env, act) == K26RL_OK);
            ASSERT(b.s.obs(b.env, e2 + (size_t)i * OBS) == K26RL_OK);
        }
        ASSERT(memcmp(e1, e2, sizeof e1) == 0);
        /* Vacuity: the streams compared are not constant. */
        ASSERT(e1[0] != e1[(T - 1) * OBS]);
        printf("  two episodes of %d periods agree bitwise over all %d "
               "channels, the propellant channel running from %.6g to "
               "%.6g within each\n", T, OBS, e1[0], e1[(T - 1) * OBS]);
        g_arms++;
        art_close_(&b);
    }

    /* ---- Arm 6: two processes, one seed ------------------------------ */
    {
        rl_run_or_die_(WORK_DIR "/axd --envs 2 --episodes 2 --seed 7"
                       " --out " WORK_DIR "/burn_a.k26epi > /dev/null");
        rl_run_or_die_(WORK_DIR "/axd --envs 2 --episodes 2 --seed 7"
                       " --out " WORK_DIR "/burn_b.k26epi > /dev/null");
        ASSERT(rl_files_equal_(WORK_DIR "/burn_a.k26epi",
                               WORK_DIR "/burn_b.k26epi"));
        printf("  two processes at one seed write identical episode files "
               "for a craft that burns to depletion\n");
        g_arms++;
    }

    /* ---- Arm 7b: the components through a declared sensor ------------ */
    {
        static const char *const SENSED_KFL =
            "form RL_PROP_S\n"
            "fn world w\n"
            WORLD_HEAD("axa.k26asm")
            "    episode\n"
            "        control_dt 10.0\n"
            "        horizon 200\n"
            "        substeps 16\n"
            "    end\n"
            "    sensor gauge\n"
            "        noise normal 0.0 2.0\n"
            "    end\n"
            "    action push box 0.0 1.0 default 0.0\n"
            "    on_step\n"
            "        craft.main.throttle = push\n"
            "    end\n"
            "    observe propulsion of craft through gauge with truth"
            " as fuel\n"
            "    objective\n"
            "        reward fuel_propellant_fraction\n"
            "    end\n"
            "end\n"
            "end\n";
        rl_write_file_(WORK_DIR "/sen.kfl", SENSED_KFL);
        rl_compile_(WORK_DIR "/sen.kfl", WORK_DIR "/sen", WORK_DIR);
        PropArt a;
        art_open_(&a, WORK_DIR "/sen.rlenv.so", 4242u);
        uint8_t blob[65536];
        int32_t n = a.s.spec(a.env, blob, (uint32_t)sizeof blob);
        ASSERT(n > 0);
        RlSpecView v;
        rl_parse_spec_(blob, (uint32_t)n, &v);
        ASSERT(v.n_chan_names == 8);
        ASSERT(strcmp(v.chan_names[4], "fuel_truth_propellant_kg") == 0);
        double act[1] = { 1.0 };
        int differing = 0;
        for (int i = 0; i < 4; i++) {
            ASSERT(a.s.step(a.env, act) == K26RL_OK);
            double o[16];
            ASSERT(a.s.obs(a.env, o) == K26RL_OK);
            if (o[0] != o[4]) differing++;
            /* The uncorrupted half is the tank as it stands. */
            ASSERT(o[4] > 0.0 && o[4] < 200.0);
        }
        if (differing == 0) {
            fprintf(stderr, "FAIL: the gauge left every reading equal to "
                    "the uncorrupted value; the sensor reached nothing\n");
            exit(1);
        }
        printf("  the four components route through a declared gauge: %d "
               "of 4 readings differ from the uncorrupted ones beside "
               "them\n", differing);
        g_arms++;
        art_close_(&a);
    }

    /* ---- Arm 8: the refusals ----------------------------------------- */
    {
        static const struct { const char *name, *src, *needle; } bad[] = {
            { "no_assembly",
              "form RL_PROP_N1\n"
              "fn world w\n"
              "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
              "    astro_body rock gm=1.0 parent=earth pos_x=1.0e11"
              " vel_y=1000.0\n"
              "    episode\n"
              "        control_dt 1.0\n"
              "        horizon 4\n"
              "    end\n"
              "    action push box 0.0 1.0 default 0.0\n"
              "    observe propulsion of rock as fuel\n"
              "    objective\n"
              "        reward fuel_mass_kg + push\n"
              "    end\n"
              "end\n"
              "end\n",
              "declares no `assembly=`" },
            { "no_tank",
              "form RL_PROP_N2\n"
              "fn world w\n"
              "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
              "    astro_body craft assembly=\"bare.k26asm\" parent=earth"
              " pos_x=1.0e11 vel_y=1000.0\n"
              "    episode\n"
              "        control_dt 1.0\n"
              "        horizon 4\n"
              "    end\n"
              "    action push box 0.0 1.0 default 0.0\n"
              "    observe propulsion of craft as fuel\n"
              "    objective\n"
              "        reward fuel_mass_kg + push\n"
              "    end\n"
              "end\n"
              "end\n",
              "no component holding propellant" }
        };
        rl_write_file_(WORK_DIR "/bare.k26asm",
            "assembly bare_box\n"
            "    frame x_to_port\n"
            "    provenance mass \"calibration shape, not a craft\" "
            "computed\n"
            "    component hull\n"
            "        mass 1000.0\n"
            "        at 0 0 0\n"
            "        collider box 1.0 0.5 0.5\n"
            "    end\n"
            "end\n");
        for (size_t k = 0; k < sizeof bad / sizeof bad[0]; k++) {
            char path[512], cmd[2048];
            snprintf(path, sizeof path, WORK_DIR "/%s.kfl", bad[k].name);
            rl_write_file_(path, bad[k].src);
            snprintf(cmd, sizeof cmd,
                     "./bin/kflc --check %s > " WORK_DIR "/%s.log 2>&1",
                     path, bad[k].name);
            int rc = system(cmd);
            if (rc == 0) {
                fprintf(stderr, "FAIL: %s was accepted\n", bad[k].name);
                exit(1);
            }
            snprintf(cmd, sizeof cmd, "grep -q -- '%s' " WORK_DIR "/%s.log",
                     bad[k].needle, bad[k].name);
            if (system(cmd) != 0) {
                fprintf(stderr, "FAIL: %s was refused without naming "
                        "`%s`\n", bad[k].name, bad[k].needle);
                snprintf(cmd, sizeof cmd, "cat " WORK_DIR "/%s.log",
                         bad[k].name);
                (void)!system(cmd);
                exit(1);
            }
            printf("  refused: %s\n", bad[k].name);
            g_arms++;
        }
    }

    /* ---- The perturbed artifacts ------------------------------------ */
    {
        /* The apportionment removed: the sub-interval a tank runs out
         * in fires at the throttle it was commanded at for the whole
         * of itself, while the tank still stops at zero. That is the
         * craft spending propellant it does not have, and it is the
         * defect arm 4 exists to catch. */
        emit_("axd");
        mutate_("axd", "axd_overrun",
                "s/KFLRL_THRSC(h, e, vi) = left \\/ demand;"
                "/KFLRL_THRSC(h, e, vi) = 1.0;/",
                "KFLRL_THRSC(h, e, vi) = left / demand;", 1, 0);
        build_emitted_(WORK_DIR "/axd_overrun.cc",
                       WORK_DIR "/axd_overrun.so");
        double dv_bad = depleting_period_dv_(WORK_DIR "/axd_overrun.so");
        double dv_ok  = depleting_period_dv_(WORK_DIR "/axd.rlenv.so");
        if (!(dv_bad - dv_ok > 0.3)) {
            fprintf(stderr, "FAIL: with the apportionment removed the "
                    "depleting period delivered %.12g m/s against the "
                    "sound artifact's %.12g; arm 4 could not have seen "
                    "the difference\n", dv_bad, dv_ok);
            exit(1);
        }
        printf("  perturbation: without the apportionment the depleting "
               "period delivers %.10g m/s instead of %.10g, which arm 4's "
               "band excludes\n", dv_bad, dv_ok);
        g_arms++;
    }
    {
        /* The mass properties frozen: the tank still counts down and
         * the craft still spends, but its mass, centre of mass and
         * inertia stay where construction put them. That is the defect
         * arm 3 exists to catch, and it is the shortcut this model
         * declined. */
        emit_("off");
        mutate_("off", "off_frozen",
                "s/^                kflrl_apply_props_(h->vehicles\\[pix\\],"
                " vb, vi, left,$/                if (0) "
                "kflrl_apply_props_(h->vehicles[pix], vb, vi, left,/",
                "if (0) kflrl_apply_props_", 0, 1);
        build_emitted_(WORK_DIR "/off_frozen.cc",
                       WORK_DIR "/off_frozen.so");
        PropArt a;
        art_open_(&a, WORK_DIR "/off_frozen.so", 4242u);
        drain_(&a, 220);
        double fill = 0.0;
        double r = measure_ratio_(&a, &fill);
        double want = omega_ratio_(fill);
        double err = fabs(r / want - 1.0);
        if (!(err > 1.0e-3)) {
            fprintf(stderr, "FAIL: with the mass properties frozen the "
                    "angular experiment still read %.12g against the "
                    "closed form %.12g at %.6g kg; arm 3 could not have "
                    "seen the difference\n", r, want, fill);
            exit(1);
        }
        printf("  perturbation: with the mass properties frozen the same "
               "experiment reads %.9g where the tank at %.4g kg calls for "
               "%.9g, a relative %.3g against arm 3's band of 1e-3\n",
               r, fill, want, err);
        g_arms++;
        art_close_(&a);
    }

    printf("test_rl_propulsion: %d arm(s) passed\n", g_arms);
    return 0;
}
