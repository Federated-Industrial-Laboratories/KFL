/* test_rl_effectors.c: the kinetic and directed-energy effectors.
 *
 * `engage <payload> at <target>` fires an effector payload, and
 * `observe effect <payload> as <name>` publishes what the engagement
 * did. The substance of the surface is that the engagement changes the
 * world, so most of this gate is about the world and not about the
 * channels.
 *
 * Gates:
 *   1. Refusals. Every rule the two statements carry, written as a
 *      program that breaks it: `engage` outside a step body, an
 *      unknown payload, a payload of a kind that is not an effector, a
 *      payload engaged at its own platform, an unknown target, a
 *      target with no assembly and one whose assembly declares no
 *      collider, a numeric value on a keyword-valued key, an unknown
 *      word on one, a distribution on one, the swarm keys against the
 *      single pattern and missing against the swarm one, and a second
 *      `engage` of one payload in one step body.
 *   2. Acceptance. Both kinds through compile, link, dlopen, step,
 *      reset and step, at both release patterns, so "accepted" means
 *      an artifact exists and runs rather than that a checker did not
 *      object.
 *   3. The published channels: the names and the order of both
 *      component sets, read from the artifact's own spec blob.
 *   4. The effect on the world, one arm per kind. The target's
 *      trajectory is compared with and without the engagement, over
 *      the same seed and the same action stream, and the arm requires
 *      them to differ. Its own mutation follows: with the velocity
 *      write deleted from the emitted source the trajectory returns to
 *      the control's, bit for bit, and the arm must fail. A gate that
 *      cannot fail on the defect it names measures nothing.
 *   4b. The ablated mass is not a dead store. Deleting the mass write
 *      alone moves the trajectory, because the reduced mass divides
 *      the next engagement's velocity increment. The route is measured
 *      rather than assumed, at two separations, because the answer
 *      depends on one: with the velocity write also deleted a
 *      residual remains, the target's own gravitational parameter
 *      reaching the integrator's step control, and it is invisible two
 *      kilometres out and measurable at a hundred and fifty metres.
 *      The arm requires the increment to be the larger route rather
 *      than requiring the other to be absent.
 *   5. The hit test, three arms that no two of which can be collapsed:
 *      a closing intercept hits; the same geometry receding does not,
 *      though its predicted closest approach is the same number; and a
 *      closing pass offset beyond the target's silhouette radius does
 *      not, though its time to closest approach is positive.
 *   6. Swarm against single. The two patterns give genuinely different
 *      transferred momentum through the footprint fraction, and the
 *      fraction moves with range.
 *   6b. The silhouette follows the line the effector acts on. Two
 *      targets differing only in attitude, nose on and broadside, are
 *      required to differ by the ratio the asset's own box dimensions
 *      predict. An arm that only asks whether an area-dependent
 *      quantity moves cannot tell a silhouette taken along the closing
 *      direction from one taken along a fixed axis.
 *   6c. Each kind's principal result is recomputed from the other
 *      published components. An increment that lost the target's mass
 *      would still be non-zero and would still move, so a magnitude is
 *      what pins it.
 *   6f. The release count decides what one arriving unit is. Three
 *      swarms of equal total mass and different counts: the momentum
 *      must not move and the penetration must, since the first is the
 *      landed total's and the second is a unit's.
 *   6e. The bumper's thickness and the rear wall's are different keys
 *      feeding different branches, measured by two fixtures differing
 *      in that one key: the penetration analysis must not move and the
 *      delivered energy must.
 *   6d. And a magnitude cannot see a sign. The target's change in
 *      velocity is projected onto the emitter-to-target line and
 *      required positive, because a recoil reversed leaves every
 *      published component bit-identical and pushes the target the
 *      wrong way.
 *   7. Every published component moves. For each component of each
 *      kind, a fixture in which it moves and the measured spread; a
 *      component nothing can move is a component nothing can be shaped
 *      against.
 *   8. Zero fill. A step whose body did not engage reads `_engaged`
 *      0.0 with every other component 0.0, measured on the same
 *      artifact and the same episode as a step that did engage.
 *   9. One engagement per payload per step, the runtime half. A single
 *      statement reached twice through a loop faults the environment
 *      with K26RL_E_ENV_INTERNAL; the same loop running once does not,
 *      which is what tells the rule from a fault that always fires.
 *      A loop is the only route to that path: `engage` is a statement
 *      of the step body and an ordinary identifier everywhere else, so
 *      a `fn` the body calls cannot hold one and there is no
 *      twice-called function to reach it through.
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

#define WORK_DIR "/tmp/kflc_rl_effectors_test"

static int g_arms;

/* ---- Sources -------------------------------------------------------- */

/* The world every arm is built on. `shooter` carries both effectors;
 * `mover` is a two-by-one-by-one metre box turning about its z axis, so
 * the silhouette it presents along a fixed direction is a function of
 * time with a closed form.
 *
 * The box starts nose on to the closing direction. That is not
 * decoration: the library's impact-angle cosine is clamped at zero,
 * because a surface is not struck from behind, so a target whose first
 * body axis points away from the projectile reports zero and the
 * penetration components stay at zero with it. Starting nose on and
 * turning away is the one arrangement in which that cosine, and the
 * two components that depend on it, are seen to move.
 */
#define EFF_EARTH \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"

#define EFF_SHOOTER(vy) \
    "    astro_body shooter assembly=\"crew_vehicle_10t.k26asm\"" \
    " parent=earth pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0" \
    " vel_y=" vy " vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0\n"

/* The separation is a parameter because the kinetic effector is a
 * terminal-phase one: its intercept must fall inside the step about to
 * be integrated, so how far apart the pair starts decides whether a
 * given step lands. At the closing speed these fixtures use, 200
 * metres per second against a half-second period, an intercept is
 * inside the step below a hundred metres of separation and outside it
 * above. Every fixture states which side of that it sits on. */
#define EFF_MOVER(asset, py, pz) \
    "    astro_body mover assembly=\"" asset "\"" \
    " parent=earth pos_x=7.0e6 pos_y=" py " pos_z=" pz " vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=0.7071067811865476 quat_x=0.0" \
    " quat_y=0.0 quat_z=-0.7071067811865476 omega_x=0.0 omega_y=0.0" \
    " omega_z=0.05\n"

/* The same target lying broadside instead of nose on. Its collision
 * primitive is a box of 2.0 by 1.0 by 1.0 metres, so along its own
 * long axis it presents 1.0 square metre and along either short axis
 * 2.0: the two attitudes differ by a factor of two in silhouette and
 * in nothing else. */
#define EFF_MOVER_BROADSIDE(asset, py, pz) \
    "    astro_body mover assembly=\"" asset "\"" \
    " parent=earth pos_x=7.0e6 pos_y=" py " pos_z=" pz " vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0" \
    " quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.05\n"

/* The directed-energy payload. One megawatt at a metre and a half of
 * aperture is an emitter a reader could describe; the arm that
 * measures the ablated mass declares a larger one and says so. */
#define EFF_LASER(power) \
    "    astro_payload beam body=shooter kind=laser primary_diam_m=1.5" \
    " wavelength_nm=1064.0 p_output_w=" power " m_squared=1.2" \
    " pointing_jitter_rad=1.0e-7 rms_wavefront_m=5.0e-8" \
    " plasma_attn_k=1.0 target_material=aluminum" \
    " target_reflectivity=0.2\n"

/* The kinetic payload, with the whole of a shielded target's Whipple
 * description so the penetration components have a branch to run and
 * the coupling routine has the two layers it distinguishes: a bumper
 * of its own thickness, a stand-off, and the rear wall behind it. */
#define EFF_IMPACTOR_SINGLE \
    "    astro_payload rock body=shooter kind=impactor pattern=single" \
    " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0" \
    " projectile_diameter_m=0.2 target_wall_thickness_m=0.002" \
    " target_bumper_thickness_m=0.0016" \
    " target_bumper_density_kg_per_m3=2700.0" \
    " target_bumper_spacing_m=0.1 target_wall_yield_stress_ksi=40.0" \
    " target_inner_thickness_m=0.003\n"

/* The swarm payload declares both structure branches, the Whipple
 * parameters and the monolithic ones. The library runs the two
 * independently, so one payload exercises both and neither branch's
 * components can be left at zero by an arm that only ever described
 * one kind of target. */
#define EFF_IMPACTOR_SWARM \
    "    astro_payload rock body=shooter kind=impactor pattern=swarm" \
    " swarm_count=20 swarm_half_angle_rad=0.02" \
    " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0" \
    " projectile_diameter_m=0.2 target_wall_thickness_m=0.002" \
    " target_bumper_thickness_m=0.0016" \
    " target_bumper_density_kg_per_m3=2700.0" \
    " target_bumper_spacing_m=0.1 target_wall_yield_stress_ksi=40.0" \
    " target_inner_thickness_m=0.003" \
    " target_brinell_hardness=95.0 target_density_kg_per_m3=2700.0" \
    " target_speed_of_sound_m_per_s=5100.0" \
    " target_monolithic_thickness_m=0.02\n"

#define EFF_EPISODE \
    "    episode\n" \
    "        control_dt 0.5\n" \
    "        substeps 4\n" \
    "        horizon 32\n" \
    "    end\n"

#define EFF_ACTION "    action fire box -1.0 1.0 default 0.0\n"

/* Both effectors engaged every step, both results published. The
 * reward reads one component of each so nothing in the pair is a
 * channel with no consumer. */
static const char *const BOTH_KFL =
    "form EFFBOTH\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_LASER("1.0e6") EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage beam at mover\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect + kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* The laser alone, and the matched control with the same world, the
 * same craft and the same period and no engagement. Without the
 * control a trajectory says only where the craft went, not what the
 * engagement did to it. */
static const char *const LAS_KFL =
    "form EFFLAS\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_LASER("1.0e6")
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    on_step\n"
    "        engage beam at mover\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect\n"
    "    end\n"
    "end\n"
    "end\n";

static const char *const KIN_KFL =
    "form EFFKIN\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* The control declares the same two craft and the same episode frame
 * and engages nothing. Its step body writes a body state key scaled by
 * zero, so it has a body of the same shape without moving anything. */
static const char *const CTRL_KFL =
    "form EFFCTRL\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_EPISODE EFF_ACTION
    "    observe mover from shooter mode=geometric as los\n"
    "    on_step\n"
    "        shooter.vel_x = shooter.vel_x + fire * 0.0\n"
    "    end\n"
    "    objective\n"
    "        reward los_range\n"
    "    end\n"
    "end\n"
    "end\n";

/* The swarm pattern against the single one, in otherwise identical
 * worlds, so the transferred momentum differs by the footprint
 * fraction and by nothing else. */
static const char *const SWARM_KFL =
    "form EFFSWARM\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_IMPACTOR_SWARM
    EFF_EPISODE EFF_ACTION
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* The target receding: the same separation and the same relative speed
 * with the sign reversed. The predicted closest-approach distance is
 * the same small number as the closing case, so an arm that tested
 * that alone could not tell the two apart. */
static const char *const RECEDE_KFL =
    "form EFFRECEDE\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7346.0")
    EFF_MOVER("calibration_box.k26asm", "5.0e1", "0.0")
    EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* A genuine collision course two kilometres out. The predicted closest
 * approach is zero and the time to it is ten seconds, which is twenty
 * control periods, so the intercept is real and is nowhere near the
 * step about to be integrated. Nothing may be transferred.
 *
 * This is the fixture the terminal-window bound exists for. Without it
 * the engagement lands here, and lands again on every following step,
 * each time delivering a whole projectile's momentum for an
 * intercept that has not happened. */
static const char *const LONG_KFL =
    "form EFFLONG\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0")
    EFF_MOVER("calibration_box.k26asm", "2.0e3", "0.0")
    EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* A control's body: the same craft and the same episode frame with no
 * engagement in it, its step body writing a body state key scaled by
 * zero so it has a block of the same shape without moving anything.
 * One per geometry, because a control at a different separation is not
 * a control at all. */
#define EFF_CTRL_BODY \
    "    observe mover from shooter mode=geometric as los\n" \
    "    on_step\n" \
    "        shooter.vel_x = shooter.vel_x + fire * 0.0\n" \
    "    end\n" \
    "    objective\n" \
    "        reward los_range\n" \
    "    end\n" \
    "end\n" \
    "end\n"

static const char *const CTRL_LONG_KFL =
    "form EFFCTRLLONG\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0")
    EFF_MOVER("calibration_box.k26asm", "2.0e3", "0.0")
    EFF_EPISODE EFF_ACTION EFF_CTRL_BODY;

/* The receding fixture's own control. */
static const char *const CTRL_RECEDE_KFL =
    "form EFFCTRLRECEDE\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7346.0")
    EFF_MOVER("calibration_box.k26asm", "5.0e1", "0.0")
    EFF_EPISODE EFF_ACTION EFF_CTRL_BODY;

/* A pass inside the step whose closest approach falls in the band
 * between the silhouette's true radius and its square root.
 *
 * The target lies broadside, so it presents two square metres and its
 * effective radius is sqrt(2/pi), about 0.798 m. A radius taken as
 * sqrt(area) instead would be 1.414 m, and every other fixture here
 * sits far outside that band: at zero, or at five metres. One metre is
 * inside it, so this is the fixture that measures the boundary rather
 * than the two sides of it. */
static const char *const BAND_KFL =
    "form EFFBAND\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0")
    EFF_MOVER_BROADSIDE("calibration_box.k26asm", "5.0e1", "1.0")
    EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* A closing pass that misses: the target is offset five metres out of
 * the closing plane, against a silhouette radius of about half a
 * metre, so the time to closest approach is positive and the intercept
 * still does not land. */
static const char *const OFFSET_KFL =
    "form EFFOFFSET\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "5.0e1", "5.0")
    EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* An engagement under a condition on the action, so one artifact and
 * one episode carry both a step that engaged and a step that did not.
 * A fixture with two artifacts could not tell a cleared block from two
 * different programs. */
static const char *const COND_KFL =
    "form EFFCOND\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_LASER("1.0e6") EFF_IMPACTOR_SINGLE
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        if fire > 0.5\n"
    "            engage beam at mover\n"
    "            engage rock at mover\n"
    "        end\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect + kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* One `engage` statement inside a loop. The bound is an action, so the
 * same artifact runs the loop once on one step and twice on the next:
 * a fixture whose loop always ran twice could not tell the rule from a
 * fault that fires whatever the program does. */
static const char *const LOOP_KFL =
    "form EFFLOOP\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
    EFF_LASER("1.0e6")
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    on_step\n"
    "        let n: double = 1.0\n"
    "        if fire > 0.5\n"
    "            n = 2.0\n"
    "        end\n"
    "        let i: double = 0.0\n"
    "        while i < n\n"
    "            engage beam at mover\n"
    "            i = i + 1.0\n"
    "        end\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* A hotter emitter against a faster-turning target. Two of the
 * emitter's components report the plasma regime, and a fluence that
 * never reaches the material's ignition threshold leaves both of them
 * still: the transmissivity reads 1.0 because there is no plasma to
 * attenuate the beam, and the ignition flag reads 0.0. This fixture
 * puts the fluence across that threshold and turns the target fast
 * enough for the silhouette to carry it back and forth, so both are
 * seen to move and the hard flag is seen to follow the continuous
 * quantity beside it. Three hundred megawatts is a modelling choice of
 * this gate, not a claim about any emitter. */
#define EFF_MOVER_FAST(asset) \
    "    astro_body mover assembly=\"" asset "\"" \
    " parent=earth pos_x=7.0e6 pos_y=2.0e3 pos_z=0.0 vel_x=0.0" \
    " vel_y=7546.0 vel_z=0.0 quat_w=0.7071067811865476 quat_x=0.0" \
    " quat_y=0.0 quat_z=-0.7071067811865476 omega_x=0.0 omega_y=0.0" \
    " omega_z=0.5\n"

static const char *const HOT_KFL =
    "form EFFHOT\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0") EFF_MOVER_FAST("calibration_box.k26asm")
    EFF_LASER("3.0e8")
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    on_step\n"
    "        engage beam at mover\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* The same swarm and the same emitter against a target lying broadside
 * to the closing direction rather than nose on. The target's collision
 * primitive is a box of 2.0 by 1.0 by 1.0 metres, so the area it
 * presents along its own long axis is 1.0 square metre and the area
 * along either short axis is 2.0: the two fixtures differ by a factor
 * of two in silhouette and in nothing else.
 *
 * They exist because a silhouette taken along a fixed axis rather than
 * along the direction the effector acts on survives every arm that
 * only asks whether a quantity moves. Two attitudes with a predicted
 * ratio between them is what tells the two apart, and the ratio comes
 * from the asset's own declared dimensions rather than from a figure
 * restated here. */

static const char *const BROADSIDE_KFL =
    "form EFFBROAD\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0")
    EFF_MOVER_BROADSIDE("calibration_box.k26asm", "2.0e3", "0.0")
    EFF_LASER("1.0e6") EFF_IMPACTOR_SWARM
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage beam at mover\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect + kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* The nose-on half of the same pair, identical but for the target's
 * initial attitude. */
static const char *const NOSEON_KFL =
    "form EFFNOSE\n"
    "fn world w\n"
    EFF_EARTH EFF_SHOOTER("7746.0")
    EFF_MOVER("calibration_box.k26asm", "2.0e3", "0.0")
    EFF_LASER("1.0e6") EFF_IMPACTOR_SWARM
    EFF_EPISODE EFF_ACTION
    "    observe effect beam as las\n"
    "    observe effect rock as kin\n"
    "    on_step\n"
    "        engage beam at mover\n"
    "        engage rock at mover\n"
    "    end\n"
    "    objective\n"
    "        reward las_effect + kin_effect\n"
    "    end\n"
    "end\n"
    "end\n";

/* An assembly with a mesh and no collision primitive: geometry that a
 * defense payload cannot take an area of. */
static const char *const BARE_ASM =
    "# scratch_bare.k26asm - geometry without a collider.\n"
    "#\n"
    "# Not a craft. It exists so a body an effector is pointed at with\n"
    "# no silhouette can be written down: mass properties derive from\n"
    "# the mesh and no collision primitive is declared.\n"
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

/* ---- Component tables ----------------------------------------------- */

/* The two published component sets, in order. They live here as well
 * as in the compiler, and this gate is what compares them: a set that
 * drifted between the two would publish one order and document
 * another. */
static const char *const LAS_COMPS[] = {
    "_engaged", "_effect", "_dv", "_mass_loss", "_range", "_spot",
    "_encircled", "_fluence", "_transmissivity", "_p_coupled",
    "_ignited", NULL
};
static const char *const KIN_COMPS[] = {
    "_engaged", "_effect", "_hit", "_closing_speed", "_t_close",
    "_miss", "_fraction", "_cos_angle", "_penetrates",
    "_critical_diameter", "_penetration", "_energy", NULL
};
#define LAS_N 11
#define KIN_N 12

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

/* One built artifact, opened and resolved. */
typedef struct {
    void      *so;
    RlSurface  s;
    K26RlEnv  *env;
} EffArt;

static void art_open_(EffArt *a, const char *so_path, uint64_t seed,
                      uint32_t n_envs)
{
    a->so = rl_dlopen_(so_path);
    rl_resolve_surface_(a->so, &a->s);
    a->env = NULL;
    ASSERT(a->s.create(seed, n_envs, &a->env) == K26RL_OK);
}

static void art_close_(EffArt *a)
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

/* Compile a translation unit of this artifact's own emitted source
 * into a shared object directly, which is what the mutation arms need:
 * they perturb the emitted source and must build exactly that, not
 * re-run the compiler over the program. */
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

/* Apply a sed script to an emitted source and assert that it changed
 * the file. A mutation that never reached the build cannot be reported
 * as a survivor, and a mutation that never reached the file cannot be
 * reported at all. */
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

/* Step an artifact `n` times with a constant action and return the
 * target body's position and velocity. Body order is declaration
 * order, so the target is index 2 in every fixture here. */
static void run_body_(const char *so_path, int n, double act0,
                      double out6[6])
{
    EffArt a;
    art_open_(&a, so_path, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    act[0] = act0;
    for (int i = 0; i < n; i++) ASSERT(a.s.step(a.env, act) == K26RL_OK);
    double b[64];
    int32_t need = a.s.bodies(a.env, 0u, b, 64u);
    ASSERT(need == 18);
    memcpy(out6, b + 12, sizeof(double) * 6);
    art_close_(&a);
}

static double dist6_(const double a[6], const double b[6])
{
    double s = 0.0;
    for (int i = 0; i < 6; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return sqrt(s);
}

/* ---- Gate 1: the refusals ------------------------------------------- */

static void gate_refusals_(void)
{
    char src[8192];

    /* `engage` in the world prefix. */
    snprintf(src, sizeof src,
        "form EFFR\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    engage beam at mover\n"
        "    observe effect beam as las\n"
        "    on_step\n"
        "        shooter.vel_x = shooter.vel_x + fire * 0.0\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage", "act of a step",
                            "`on_step` block only", NULL };
        must_refuse_("`engage` in the world prefix", src, n);
    }

    /* An unknown payload. */
    snprintf(src, sizeof src, "%s",
        "form EFFR2\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage gun at mover\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage gun at mover",
                            "no astro_payload of that name", NULL };
        must_refuse_("`engage` naming no payload", src, n);
    }

    /* A payload of a kind that is not an effector, engaged and
     * observed. Two arms, because the two statements refuse it for
     * their own reasons and one diagnostic would not cover both. */
    snprintf(src, sizeof src, "%s",
        "form EFFR3\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload eye body=shooter kind=detect_radar"
        " p_tx_w=2000.0 g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10"
        " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0"
        " noise_figure=2.0 snr_threshold=10.0\n"
        EFF_EPISODE EFF_ACTION
        "    observe detect eye of mover as look\n"
        "    on_step\n"
        "        engage eye at mover\n"
        "    end\n"
        "    objective\n"
        "        reward look_snr\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage eye at mover", "detect_radar",
                            "not an effector", NULL };
        must_refuse_("`engage` on a detection payload", src, n);
    }

    snprintf(src, sizeof src, "%s",
        "form EFFR4\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload eye body=shooter kind=detect_radar"
        " p_tx_w=2000.0 g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10"
        " loss_sys_db=3.0 bandwidth_hz=1.0e6 t_sys_k=290.0"
        " noise_figure=2.0 snr_threshold=10.0\n"
        EFF_EPISODE EFF_ACTION
        "    observe effect eye as look\n"
        "    on_step\n"
        "        shooter.vel_x = shooter.vel_x + fire * 0.0\n"
        "    end\n"
        "    objective\n"
        "        reward look_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "observe effect eye", "detect_radar",
                            "no engagement event", NULL };
        must_refuse_("`observe effect` on a detection payload", src, n);
    }

    /* A payload engaged at its own platform. */
    snprintf(src, sizeof src, "%s",
        "form EFFR5\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at shooter\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage beam at `shooter`",
                            "own platform", NULL };
        must_refuse_("a payload engaged at its own platform", src, n);
    }

    /* An unknown target, and a target with no assembly. */
    snprintf(src, sizeof src, "%s",
        "form EFFR6\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at ghost\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage beam at `ghost`",
                            "no astro_body of that name", NULL };
        must_refuse_("`engage` naming no body", src, n);
    }

    snprintf(src, sizeof src, "%s",
        "form EFFR7\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at earth\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage beam at `earth`",
                            "declares no `assembly=`", NULL };
        must_refuse_("a target that binds no assembly", src, n);
    }

    /* A target whose assembly declares no collider. Without this the
     * checker accepts a program that presents no area and the build
     * fails against a generated file. */
    snprintf(src, sizeof src, "%s",
        "form EFFR8\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("scratch_bare.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at mover\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "engage ... at `mover`",
                            "declares no `collider`",
                            "presents no area along a line of sight",
                            NULL };
        must_refuse_("an engaged body whose assembly has no collider",
                     src, n);
    }

    /* A second `engage` of one payload in one step body. */
    snprintf(src, sizeof src, "%s",
        "form EFFR9\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at mover\n"
        "        engage beam at mover\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "already engaged at line",
                            "at most once per step", NULL };
        must_refuse_("one payload engaged twice in one step body", src, n);
    }

    /* The same, with the second engagement inside a conditional, which
     * the collection walks into: a rule that stopped at the top level
     * of the block would miss it. */
    snprintf(src, sizeof src, "%s",
        "form EFFR10\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        EFF_LASER("1.0e6")
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at mover\n"
        "        if fire > 0.5\n"
        "            engage beam at mover\n"
        "        end\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "already engaged at line",
                            "at most once per step", NULL };
        must_refuse_("the second engagement inside a conditional", src, n);
    }

    /* The keyword-valued keys: a number, an unknown word, and a
     * distribution form. */
    snprintf(src, sizeof src, "%s",
        "form EFFR11\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload rock body=shooter kind=impactor pattern=2"
        " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0"
        " projectile_diameter_m=0.2\n"
        EFF_EPISODE EFF_ACTION
        "    observe effect rock as kin\n"
        "    on_step\n"
        "        engage rock at mover\n"
        "    end\n"
        "    objective\n"
        "        reward kin_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "`pattern=2`", "single, swarm",
                            "library constant rather than a number",
                            NULL };
        must_refuse_("a number where a release pattern belongs", src, n);
    }

    snprintf(src, sizeof src, "%s",
        "form EFFR12\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload beam body=shooter kind=laser"
        " primary_diam_m=1.5 wavelength_nm=1064.0 p_output_w=1.0e6"
        " m_squared=1.2 pointing_jitter_rad=1.0e-7"
        " rms_wavefront_m=5.0e-8 plasma_attn_k=1.0"
        " target_material=none target_reflectivity=0.2\n"
        EFF_EPISODE EFF_ACTION
        "    observe effect beam as las\n"
        "    on_step\n"
        "        engage beam at mover\n"
        "    end\n"
        "    objective\n"
        "        reward las_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "`target_material=none`",
                            "aluminum, steel, titanium", NULL };
        must_refuse_("a target material the grammar does not admit",
                     src, n);
    }

    snprintf(src, sizeof src, "%s",
        "form EFFR13\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload rock body=shooter kind=impactor"
        " pattern=uniform(1.0, 2.0)"
        " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0"
        " projectile_diameter_m=0.2\n"
        EFF_EPISODE EFF_ACTION
        "    observe effect rock as kin\n"
        "    on_step\n"
        "        engage rock at mover\n"
        "    end\n"
        "    objective\n"
        "        reward kin_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "pattern=uniform(1.0, 2.0)",
                            "library constant rather than a number",
                            NULL };
        must_refuse_("a distribution on a keyword-valued key", src, n);
    }

    /* The swarm keys, required by the pattern and refused against the
     * other one. Both directions, since a rule that only ever added a
     * requirement would pass the first arm alone. */
    snprintf(src, sizeof src, "%s",
        "form EFFR14\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload rock body=shooter kind=impactor pattern=swarm"
        " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0"
        " projectile_diameter_m=0.2\n"
        EFF_EPISODE EFF_ACTION
        "    observe effect rock as kin\n"
        "    on_step\n"
        "        engage rock at mover\n"
        "    end\n"
        "    objective\n"
        "        reward kin_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "`pattern=swarm` requires `swarm_count=`",
                            NULL };
        must_refuse_("a swarm with no count", src, n);
    }

    snprintf(src, sizeof src, "%s",
        "form EFFR15\n"
        "fn world w\n"
        EFF_EARTH EFF_SHOOTER("7746.0")
        EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0")
        "    astro_payload rock body=shooter kind=impactor"
        " pattern=single swarm_count=20"
        " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0"
        " projectile_diameter_m=0.2\n"
        EFF_EPISODE EFF_ACTION
        "    observe effect rock as kin\n"
        "    on_step\n"
        "        engage rock at mover\n"
        "    end\n"
        "    objective\n"
        "        reward kin_effect\n"
        "    end\n"
        "end\n"
        "end\n");
    {
        const char *n[] = { "`swarm_count=` belongs to `pattern=swarm`",
                            "pattern=single", NULL };
        must_refuse_("a swarm count on a single projectile", src, n);
    }
}

/* ---- Gate 2: acceptance reaches an artifact -------------------------- */

static void must_build_and_step_(const char *what, const char *src,
                                 const char *stem)
{
    char so[512];
    build_(src, stem);
    so_path_(so, sizeof so, stem);
    EffArt a;
    art_open_(&a, so, 3u, 1u);
    double act[4] = { 1.0, 0.0, 0.0, 0.0 };
    ASSERT(a.s.step(a.env, act) == K26RL_OK);
    ASSERT(a.s.reset(a.env) == K26RL_OK);
    ASSERT(a.s.step(a.env, act) == K26RL_OK);
    art_close_(&a);
    g_arms++;
    printf("  built, opened, stepped, reset and stepped: %s\n", what);
}

/* ---- Gate 3: the published channels ---------------------------------- */

static void spec_of_(const char *stem, RlSpecView *v)
{
    char so[512];
    so_path_(so, sizeof so, stem);
    EffArt a;
    art_open_(&a, so, 3u, 1u);
    uint8_t blob[8192];
    int32_t n = a.s.spec(a.env, blob, sizeof blob);
    ASSERT(n > 0 && (size_t)n <= sizeof blob);
    rl_parse_spec_(blob, (uint32_t)n, v);
    art_close_(&a);
}

static void gate_channels_(void)
{
    RlSpecView v;
    spec_of_("both", &v);
    ASSERT(v.obs_total == (uint32_t)(LAS_N + KIN_N));
    for (int i = 0; LAS_COMPS[i]; i++) {
        char want[96];
        snprintf(want, sizeof want, "las%s", LAS_COMPS[i]);
        if (strcmp(v.chan_names[i], want) != 0) {
            fprintf(stderr, "FAIL: channel %d is `%s`, expected `%s`\n",
                    i, v.chan_names[i], want);
            exit(1);
        }
    }
    for (int i = 0; KIN_COMPS[i]; i++) {
        char want[96];
        snprintf(want, sizeof want, "kin%s", KIN_COMPS[i]);
        if (strcmp(v.chan_names[LAS_N + i], want) != 0) {
            fprintf(stderr, "FAIL: channel %d is `%s`, expected `%s`\n",
                    LAS_N + i, v.chan_names[LAS_N + i], want);
            exit(1);
        }
    }
    /* Both are published under the geometric observer-mode tag: an
     * engagement applies no light-time correction and no aberration,
     * so the default astrometric value would assert a correction that
     * is not made. */
    ASSERT(v.n_modes == LAS_N + KIN_N);
    for (int i = 0; i < v.n_modes; i++) ASSERT(v.modes[i] == 0);
    g_arms++;
    printf("  %d published components in the design's order, %d for the "
           "emitter and %d for the impactor, all geometric\n",
           LAS_N + KIN_N, LAS_N, KIN_N);
}

/* ---- Gate 4: the effect on the world --------------------------------- */

#define EFF_STEPS 6

static void gate_effect_world_(void)
{
    double ctrl[6], las[6], kin[6];
    char so[512];

    build_(CTRL_KFL, "ctrl");
    build_(LAS_KFL, "las");
    build_(KIN_KFL, "kin");

    so_path_(so, sizeof so, "ctrl");
    run_body_(so, EFF_STEPS, 0.0, ctrl);
    so_path_(so, sizeof so, "las");
    run_body_(so, EFF_STEPS, 0.0, las);
    so_path_(so, sizeof so, "kin");
    run_body_(so, EFF_STEPS, 0.0, kin);

    double d_las = dist6_(las, ctrl);
    double d_kin = dist6_(kin, ctrl);
    if (!(d_las > 0.0)) {
        fprintf(stderr, "FAIL: the directed-energy engagement left the "
                "target's trajectory unchanged\n");
        exit(1);
    }
    if (!(d_kin > 0.0)) {
        fprintf(stderr, "FAIL: the kinetic engagement left the target's "
                "trajectory unchanged\n");
        exit(1);
    }
    g_arms += 2;
    printf("  the target's state after %d steps differs from the "
           "no-engagement control by %.6g (emitter) and %.6g "
           "(impactor), in metres and metres per second combined\n",
           EFF_STEPS, d_las, d_kin);

    /* The mutations. Each deletes the whole of that kind's effect on
     * the world, from that artifact's own emitted source, and requires
     * the trajectory to return to the control's bit for bit. An arm
     * that still passed would be measuring something other than the
     * effect.
     *
     * For the emitter that is both writes and not only the velocity
     * one. The ablated mass reaches the trajectory by two routes, and
     * the second of them is measured in its own gate below; deleting
     * one write and demanding bit-for-bit equality would be asserting
     * that the other route is absent, which at close separations it is
     * not. */
    emit_("las");
    mutate_("las", "las_noeffect",
            "s/^    _kfl_tb->vel\\.\\([xyz]\\) += _kfl_dv \\* "
            "_kfl_u\\.\\([xyz]\\);$/    (void)0;/;"
            "s/^        k26astro_body_set_mass(_kfl_tb, _kfl_mass - "
            "_kfl_loss);$/        (void)0;/",
            "_kfl_tb->vel", 3, 0);
    build_emitted_(WORK_DIR "/las_noeffect.cc",
                   WORK_DIR "/las_noeffect.so");
    double las_noeffect[6];
    run_body_(WORK_DIR "/las_noeffect.so", EFF_STEPS, 0.0, las_noeffect);
    if (dist6_(las_noeffect, ctrl) != 0.0) {
        fprintf(stderr, "FAIL: with the emitter's two state writes "
                "deleted the trajectory still differs from the control "
                "by %.6g; the arm above is measuring something else\n",
                dist6_(las_noeffect, ctrl));
        exit(1);
    }
    g_arms++;
    printf("  mutation: the emitter's two state writes deleted return the "
           "trajectory to the control's, bit for bit\n");

    /* The needle is the increment rather than the field, because the
     * impactor's own geometry reads the target's velocity too and a
     * count over the field would not be a count of the writes. */
    emit_("kin");
    mutate_("kin", "kin_novel",
            "s/^    _kfl_tb->vel\\.\\([xyz]\\) += _kfl_dv \\* "
            "_kfl_w\\.\\([xyz]\\);$/    (void)0;/",
            "_kfl_dv \\* _kfl_w", 3, 0);
    build_emitted_(WORK_DIR "/kin_novel.cc", WORK_DIR "/kin_novel.so");
    double kin_novel[6];
    run_body_(WORK_DIR "/kin_novel.so", EFF_STEPS, 0.0, kin_novel);
    if (dist6_(kin_novel, ctrl) != 0.0) {
        fprintf(stderr, "FAIL: with the impactor's velocity write deleted "
                "the trajectory still differs from the control by %.6g; "
                "the arm above is measuring something else\n",
                dist6_(kin_novel, ctrl));
        exit(1);
    }
    g_arms++;
    printf("  mutation: the impactor's velocity write deleted returns the "
           "trajectory to the control's, bit for bit\n");
}

/* ---- Gate 4b: the ablated mass is consumed --------------------------- */

/* A high-power emitter, declared so the ablated fraction is large
 * enough to see in six steps. The figure is a modelling choice of this
 * gate and not a claim about any real emitter; what the arm measures
 * is the route the mass takes, which does not depend on the power.
 *
 * Two separations, because the answer depends on one. Far apart, the
 * mass reaches the trajectory only by dividing the next engagement's
 * velocity increment. Close in, a second and much weaker route opens:
 * the target's own mass sets its gravitational parameter, which enters
 * the integrator's own step control through the close-encounter and
 * predictor machinery, and at close separation that machinery is live.
 * A single-separation fixture would report one of those two answers
 * and call it the answer. */
#define MASSY_KFL_AT(form, py, power) \
    "form " form "\n" \
    "fn world w\n" \
    EFF_EARTH EFF_SHOOTER("7746.0") \
    EFF_MOVER("calibration_box.k26asm", py, "0.0") \
    EFF_LASER(power) \
    EFF_EPISODE EFF_ACTION \
    "    observe effect beam as las\n" \
    "    on_step\n" \
    "        engage beam at mover\n" \
    "    end\n" \
    "    objective\n" \
    "        reward las_effect\n" \
    "    end\n" \
    "end\n" \
    "end\n"

static const char *const MASSY_KFL =
    MASSY_KFL_AT("EFFMASS", "2.0e3", "1.0e9");
static const char *const MASSY_NEAR_KFL =
    MASSY_KFL_AT("EFFMASSNEAR", "1.5e2", "1.0e9");

/* Measure, for one fixture, how far the mass write moves the target on
 * its own and how far it moves it once the velocity write is gone.
 * The first is the whole of what the mass does; the second is what is
 * left when the route through the increment is closed. */
static void mass_routes_(const char *src, const char *stem,
                         double *out_total, double *out_field)
{
    char so[512], cc[512], mso[512];
    build_(src, stem);
    emit_(stem);
    so_path_(so, sizeof so, stem);

    double base[6], nomass[6], novel[6], neither[6];
    run_body_(so, EFF_STEPS, 0.0, base);

    char a[128], b[128], c[128];
    snprintf(a, sizeof a, "%s_nomass", stem);
    snprintf(b, sizeof b, "%s_novel", stem);
    snprintf(c, sizeof c, "%s_neither", stem);

    mutate_(stem, a,
            "s/^        k26astro_body_set_mass(_kfl_tb, _kfl_mass - "
            "_kfl_loss);$/        (void)0;/",
            /* The needle names the ablation's own mass write and not
             * every call to that entry: the emitted source has other
             * callers, and a count over all of them would move when
             * one of those changed and leave this arm asserting
             * something about a site it does not mutate. */
            "k26astro_body_set_mass(_kfl_tb", 1, 0);
    snprintf(cc, sizeof cc, WORK_DIR "/%s.cc", a);
    snprintf(mso, sizeof mso, WORK_DIR "/%s.so", a);
    build_emitted_(cc, mso);
    run_body_(mso, EFF_STEPS, 0.0, nomass);

    mutate_(stem, b,
            "s/^    _kfl_tb->vel\\.\\([xyz]\\) += _kfl_dv \\* "
            "_kfl_u\\.\\([xyz]\\);$/    (void)0;/",
            "_kfl_tb->vel", 3, 0);
    snprintf(cc, sizeof cc, WORK_DIR "/%s.cc", b);
    snprintf(mso, sizeof mso, WORK_DIR "/%s.so", b);
    build_emitted_(cc, mso);
    run_body_(mso, EFF_STEPS, 0.0, novel);

    mutate_(b, c,
            "s/^        k26astro_body_set_mass(_kfl_tb, _kfl_mass - "
            "_kfl_loss);$/        (void)0;/",
            /* The needle names the ablation's own mass write and not
             * every call to that entry: the emitted source has other
             * callers, and a count over all of them would move when
             * one of those changed and leave this arm asserting
             * something about a site it does not mutate. */
            "k26astro_body_set_mass(_kfl_tb", 1, 0);
    snprintf(cc, sizeof cc, WORK_DIR "/%s.cc", c);
    snprintf(mso, sizeof mso, WORK_DIR "/%s.so", c);
    build_emitted_(cc, mso);
    run_body_(mso, EFF_STEPS, 0.0, neither);

    *out_total = dist6_(base, nomass);
    *out_field = dist6_(novel, neither);
}

static void gate_mass_consumed_(void)
{
    double far_total = 0.0, far_field = 0.0;
    double near_total = 0.0, near_field = 0.0;

    mass_routes_(MASSY_KFL, "massy", &far_total, &far_field);
    mass_routes_(MASSY_NEAR_KFL, "massy_near", &near_total, &near_field);

    if (!(far_total > 0.0) || !(near_total > 0.0)) {
        fprintf(stderr, "FAIL: deleting the mass write changed nothing "
                "(%.6g far, %.6g near); the published mass loss would be "
                "a dead store\n", far_total, near_total);
        exit(1);
    }
    g_arms += 2;
    printf("  deleting the mass write moves the target by %.6g at 2 km "
           "and %.6g at 150 m: the ablated mass is read back, not merely "
           "written\n", far_total, near_total);

    /* The mass takes two routes to the trajectory and which of them
     * carries it depends on the separation, which is why this arm
     * measures two. Closing the route through the velocity increment
     * leaves nothing at all two kilometres out, and leaves a residual
     * of the same order as the whole effect at a hundred and fifty
     * metres: that residual is the target's own gravitational
     * parameter reaching the integrator's step control, which is live
     * when the pair is close and not when it is far.
     *
     * The arm asserts both halves of that, so it fails if the
     * residual ever appears at long range or vanishes at short. It
     * does not assert an ordering between the two routes: at the
     * closer separation they are the same size and partly cancel, and
     * an ordering claim would have been a claim this gate measured to
     * be false. */
    if (far_field != 0.0) {
        fprintf(stderr, "FAIL: two kilometres out, the mass write moves "
                "the target by %.6g with the velocity write deleted; the "
                "field route was measured to be absent at this "
                "separation\n", far_field);
        exit(1);
    }
    if (!(near_field > 0.0)) {
        fprintf(stderr, "FAIL: at a hundred and fifty metres the mass "
                "write moves nothing with the velocity write deleted, so "
                "the second route this arm reports is not present and "
                "the fixture does not measure it\n");
        exit(1);
    }
    g_arms += 2;
    printf("  the mass takes two routes and the separation decides "
           "which: with the velocity write also deleted it moves the "
           "target by %.6g at 2 km and %.6g at 150 m, against %.6g and "
           "%.6g with it, so the increment carries it far out and the "
           "integrator's own step control carries it close in\n",
           far_field, near_field, far_total, near_total);
}

/* ---- Gate 5: the hit test -------------------------------------------- */

/* Read one observation vector after `n` steps. */
static void run_obs_(const char *so_path, int n, double act0,
                     double *out, int width)
{
    EffArt a;
    art_open_(&a, so_path, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    act[0] = act0;
    for (int i = 0; i < n; i++) ASSERT(a.s.step(a.env, act) == K26RL_OK);
    double o[128];
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    memcpy(out, o, sizeof(double) * (size_t)width);
    art_close_(&a);
}

/* The step the kinetic fixtures land on. They start a hundred and
 * fifty metres apart closing at two hundred metres per second, so the
 * first step's intercept is three quarters of a second ahead, which is
 * outside the half-second period, and the second step's is a quarter
 * of a second ahead, which is inside it. The pair then crosses and
 * recedes. One artifact therefore carries a step before the window, a
 * step inside it, and steps after it. */
#define EFF_LAND_STEP 2

static void gate_hit_test_(void)
{
    char so[512];
    double before[KIN_N], landing[KIN_N], recede[KIN_N];
    double offset[KIN_N], longr[KIN_N], band[KIN_N];

    build_(RECEDE_KFL, "recede");
    build_(OFFSET_KFL, "offset");
    build_(LONG_KFL, "long");
    build_(CTRL_LONG_KFL, "ctrl_long");
    build_(CTRL_RECEDE_KFL, "ctrl_recede");
    build_(BAND_KFL, "band");

    so_path_(so, sizeof so, "kin");
    run_obs_(so, 1, 0.0, before, KIN_N);
    run_obs_(so, EFF_LAND_STEP, 0.0, landing, KIN_N);
    so_path_(so, sizeof so, "recede");
    run_obs_(so, 1, 0.0, recede, KIN_N);
    so_path_(so, sizeof so, "offset");
    run_obs_(so, 1, 0.0, offset, KIN_N);
    so_path_(so, sizeof so, "long");
    run_obs_(so, 1, 0.0, longr, KIN_N);
    so_path_(so, sizeof so, "band");
    run_obs_(so, 1, 0.0, band, KIN_N);

    /* Inside the window: a hit, and momentum transferred. */
    ASSERT(landing[0] == 1.0);
    if (landing[2] != 1.0) {
        fprintf(stderr, "FAIL: an intercept %.6g s ahead of a %.6g s "
                "step did not land\n", landing[4], 0.5);
        exit(1);
    }
    ASSERT(landing[1] > 0.0);
    ASSERT(landing[4] > 0.0 && landing[4] <= 0.5);

    /* One step earlier, on the same artifact and the same episode: the
     * same collision course, the same predicted miss, an intercept
     * three quarters of a second ahead, and no transfer. This is the
     * boundary measured from the inside. */
    ASSERT(before[0] == 1.0);
    if (before[2] != 0.0) {
        fprintf(stderr, "FAIL: an intercept %.6g s ahead of a 0.5 s step "
                "landed\n", before[4]);
        exit(1);
    }
    if (!(before[4] > 0.5)) {
        fprintf(stderr, "FAIL: the step before the landing has its "
                "intercept %.6g s ahead, which is inside the period, so "
                "it does not measure the bound\n", before[4]);
        exit(1);
    }
    ASSERT(before[1] == 0.0);
    g_arms += 2;
    printf("  the intercept must fall inside the step: %.9g s ahead does "
           "not land, %.9g s ahead does, on one artifact and one episode "
           "at a period of 0.5 s\n", before[4], landing[4]);

    /* Far outside the window on a genuine collision course. The
     * predicted closest approach is zero, so only the horizon can
     * refuse it. */
    ASSERT(longr[0] == 1.0);
    if (longr[2] != 0.0) {
        fprintf(stderr, "FAIL: an intercept %.6g s ahead reported a hit\n",
                longr[4]);
        exit(1);
    }
    if (!(longr[4] > 5.0) || !(longr[5] < 1.0e-6)) {
        fprintf(stderr, "FAIL: the long-horizon fixture is %.6g s ahead "
                "with a predicted miss of %.6g m, so it is not a distant "
                "intercept on a collision course\n", longr[4], longr[5]);
        exit(1);
    }
    ASSERT(longr[1] == 0.0);

    /* And nothing reached the world. A channel reading zero is not the
     * same claim as a target that did not move. */
    double ctrl_long[6], long_body[6];
    so_path_(so, sizeof so, "ctrl_long");
    run_body_(so, EFF_STEPS, 0.0, ctrl_long);
    so_path_(so, sizeof so, "long");
    run_body_(so, EFF_STEPS, 0.0, long_body);
    if (dist6_(long_body, ctrl_long) != 0.0) {
        fprintf(stderr, "FAIL: a long-horizon engagement moved the target "
                "by %.6g over %d steps\n",
                dist6_(long_body, ctrl_long), EFF_STEPS);
        exit(1);
    }
    g_arms += 2;
    printf("  a collision course %.9g s ahead transfers nothing: `_hit` "
           "0, `_effect` 0, and the target's state after %d steps is the "
           "control's bit for bit\n", longr[4], EFF_STEPS);

    /* Receding: engaged, inside the distance window, and not a hit.
     * The discriminating quantity is the sign of the time and nothing
     * else. */
    ASSERT(recede[0] == 1.0);
    if (recede[2] != 0.0) {
        fprintf(stderr, "FAIL: a receding target reported a hit\n");
        exit(1);
    }
    if (!(recede[4] < 0.0)) {
        fprintf(stderr, "FAIL: the receding fixture does not have a "
                "negative time to closest approach (%.6g), so it is not "
                "the case this arm exists to tell apart\n", recede[4]);
        exit(1);
    }
    ASSERT(recede[1] == 0.0);

    /* Offset: engaged, inside the time window, and still not a hit,
     * because the predicted closest approach is outside the
     * silhouette. */
    ASSERT(offset[0] == 1.0);
    if (offset[2] != 0.0) {
        fprintf(stderr, "FAIL: a pass at %.6g m reported a hit\n",
                offset[5]);
        exit(1);
    }
    if (!(offset[4] > 0.0 && offset[4] <= 0.5)) {
        fprintf(stderr, "FAIL: the offset fixture's intercept is %.6g s "
                "ahead, so the horizon and not the distance is what "
                "refuses it\n", offset[4]);
        exit(1);
    }
    ASSERT(offset[1] == 0.0);
    g_arms += 2;
    printf("  receding does not land (miss %.6g m, time %.6g s); a pass "
           "at %.6g m inside the step does not land (time %.6g s)\n",
           recede[5], recede[4], offset[5], offset[4]);

    /* The band between the silhouette's radius and its square root.
     * The target is broadside, so it presents two square metres: the
     * radius is sqrt(2/pi) = 0.7979 m and the square root of the area
     * is 1.4142 m. A pass at one metre is inside that band, and is the
     * only fixture here that tells the two apart. */
    ASSERT(band[0] == 1.0);
    if (!(band[5] > sqrt(2.0 / 3.14159265358979323846) &&
          band[5] < sqrt(2.0))) {
        fprintf(stderr, "FAIL: the band fixture passes at %.6g m, which "
                "is not between %.6g and %.6g, so it does not bracket "
                "the radius\n", band[5],
                sqrt(2.0 / 3.14159265358979323846), sqrt(2.0));
        exit(1);
    }
    if (!(band[4] > 0.0 && band[4] <= 0.5)) {
        fprintf(stderr, "FAIL: the band fixture's intercept is %.6g s "
                "ahead, so the horizon and not the distance refuses "
                "it\n", band[4]);
        exit(1);
    }
    if (band[2] != 0.0) {
        fprintf(stderr, "FAIL: a pass at %.6g m against a silhouette of "
                "radius %.6g m reported a hit\n", band[5],
                sqrt(2.0 / 3.14159265358979323846));
        exit(1);
    }
    ASSERT(band[1] == 0.0);
    /* The band fixture is broadside, so its first body axis is square
     * to the closing direction and the impact cosine is the edge-on
     * case: the library clamps by comparison, which leaves a negative
     * zero where it finds one, and the published range is [0, 1]. */
    if (band[7] != 0.0) {
        fprintf(stderr, "FAIL: the broadside fixture's impact cosine is "
                "%.17g, so it is not the edge-on case this arm reads\n",
                band[7]);
        exit(1);
    }
    if (signbit(band[7])) {
        fprintf(stderr, "FAIL: the impact cosine published a negative "
                "zero, which is outside its documented range\n");
        exit(1);
    }
    g_arms++;
    printf("  the impact cosine publishes no negative zero at the edge-on "
           "geometry\n");
    g_arms++;
    printf("  the radius is the disc's, not the area's: a pass at %.9g m "
           "against a silhouette of radius %.9g m does not land, and "
           "%.9g m is inside the band an area-rooted radius would "
           "admit\n", band[5], sqrt(2.0 / 3.14159265358979323846),
           band[5]);

    /* And the effect follows the test: a miss transfers none, measured
     * on the world rather than on the channel. */
    double ctrl[6], rec[6];
    so_path_(so, sizeof so, "ctrl_recede");
    run_body_(so, EFF_STEPS, 0.0, ctrl);
    so_path_(so, sizeof so, "recede");
    run_body_(so, EFF_STEPS, 0.0, rec);
    if (dist6_(rec, ctrl) != 0.0) {
        fprintf(stderr, "FAIL: a receding engagement moved the target by "
                "%.6g\n", dist6_(rec, ctrl));
        exit(1);
    }
    g_arms++;
    printf("  a miss transfers no momentum: the receding fixture's target "
           "state is the control's, bit for bit\n");
}

/* ---- Gate 6: swarm against single ------------------------------------ */

static void gate_swarm_(void)
{
    char so[512];
    double single1[KIN_N], swarm1[KIN_N], swarm5[KIN_N];

    build_(SWARM_KFL, "swarm");

    /* Sampled on the step the intercept lands, since a step outside
     * the terminal window transfers nothing whatever the pattern and
     * the two would agree at zero. The fraction is sampled a step
     * earlier as well, where it is published without a landing, which
     * is what lets the arm see it move with the range. */
    so_path_(so, sizeof so, "kin");
    run_obs_(so, EFF_LAND_STEP, 0.0, single1, KIN_N);
    so_path_(so, sizeof so, "swarm");
    run_obs_(so, 1, 0.0, swarm5, KIN_N);
    run_obs_(so, EFF_LAND_STEP, 0.0, swarm1, KIN_N);

    ASSERT(single1[2] == 1.0);
    ASSERT(swarm1[2] == 1.0);
    ASSERT(single1[6] == 1.0);
    if (!(swarm1[6] > 0.0 && swarm1[6] < 1.0)) {
        fprintf(stderr, "FAIL: the swarm fraction is %.6g, which is not a "
                "spread footprint\n", swarm1[6]);
        exit(1);
    }
    if (!(swarm1[1] < single1[1])) {
        fprintf(stderr, "FAIL: the swarm transferred %.6g and the single "
                "projectile %.6g; the two patterns do not differ\n",
                swarm1[1], single1[1]);
        exit(1);
    }
    /* The fraction rises as the range closes, because the footprint
     * shrinks with it: a fraction that did not move with range would be
     * a constant wearing a channel's name. */
    if (!(swarm1[6] > swarm5[6])) {
        fprintf(stderr, "FAIL: the swarm fraction did not move with "
                "range: %.6g a step out and %.6g at the landing\n",
                swarm5[6], swarm1[6]);
        exit(1);
    }
    /* The delivered energy carries the landed fraction too. Unscaled
     * it would report a swarm's whole release as arriving, and a
     * policy shaped on it could not tell the two patterns apart. */
    if (!(swarm1[11] < single1[11])) {
        fprintf(stderr, "FAIL: the swarm delivered %.17g and the single "
                "projectile %.17g; the energy does not carry the landed "
                "fraction\n", swarm1[11], single1[11]);
        exit(1);
    }
    double e_want = swarm1[6] * single1[11];
    double e_rel  = fabs(swarm1[11] - e_want) / e_want;
    if (e_rel > 1.0e-9) {
        fprintf(stderr, "FAIL: the swarm delivered %.17g where the "
                "single projectile's %.17g scaled by the landed "
                "fraction %.17g gives %.17g\n",
                swarm1[11], single1[11], swarm1[6], e_want);
        exit(1);
    }
    g_arms += 5;
    printf("  swarm against single: fraction %.6g against 1.0, "
           "increment %.6g against %.6g m/s, energy %.6g against %.6g J "
           "which is that fraction of it to %.3g, fraction rising from "
           "%.6g as the range closes\n",
           swarm1[6], swarm1[1], single1[1], swarm1[11], single1[11],
           e_rel, swarm5[6]);
}

/* ---- Gate 6b: the silhouette follows the line of action -------------- */

/* The target's declared mass, read from the assembly this gate uses:
 * `calibration_box.k26asm` states 1000 kg for its single component.
 * The frozen surface publishes no mass getter, so the figure travels
 * from the asset rather than from the artifact, and the arm that uses
 * it says so. */
#define EFF_TARGET_MASS_KG 1000.0

static void gate_silhouette_(void)
{
    double nose[LAS_N + KIN_N], broad[LAS_N + KIN_N];
    char so[512];

    build_(NOSEON_KFL, "noseon");
    build_(BROADSIDE_KFL, "broadside");
    so_path_(so, sizeof so, "noseon");
    run_obs_(so, 1, 0.0, nose, LAS_N + KIN_N);
    so_path_(so, sizeof so, "broadside");
    run_obs_(so, 1, 0.0, broad, LAS_N + KIN_N);

    /* The impactor's footprint fraction is the silhouette over the
     * cone's footprint, and both fixtures are at the same range on the
     * step sampled, so the fraction's ratio is the silhouette's. */
    double f_ratio = broad[LAS_N + 6] / nose[LAS_N + 6];
    if (fabs(f_ratio - 2.0) > 1.0e-9) {
        fprintf(stderr, "FAIL: the swarm fraction ratio between a "
                "broadside and a nose-on target is %.17g, and the box's "
                "own dimensions predict 2.0; the silhouette is not being "
                "taken along the closing direction\n", f_ratio);
        exit(1);
    }
    g_arms++;
    printf("  the impactor's silhouette follows the closing direction: "
           "fraction %.9g broadside against %.9g nose on, a ratio of "
           "%.12g where the asset's 2.0 by 1.0 by 1.0 metre box predicts "
           "2\n", broad[LAS_N + 6], nose[LAS_N + 6], f_ratio);

    /* The emitter's fluence is its power over the area it lands on, so
     * a target of twice the silhouette takes half the fluence. The
     * ratio is not exactly two because the encircled fraction differs
     * slightly between a one and a two square metre target, which is a
     * real difference and is what the tolerance below allows for. */
    double e_ratio = nose[7] / broad[7];
    if (fabs(e_ratio - 2.0) > 1.0e-3) {
        fprintf(stderr, "FAIL: the emitter's fluence ratio between a "
                "nose-on and a broadside target is %.17g, and the box's "
                "own dimensions predict 2.0\n", e_ratio);
        exit(1);
    }
    g_arms++;
    printf("  the emitter's silhouette follows the line of sight: "
           "fluence %.9g nose on against %.9g broadside, a ratio of "
           "%.12g against the geometry's 2, the remainder being the "
           "encircled fraction (%.9g against %.9g)\n",
           nose[7], broad[7], e_ratio, nose[6], broad[6]);
}

/* ---- Gate 6c: the increment is the momentum balance ------------------ */

/* Both kinds publish a principal result whose arithmetic is fixed by
 * the design, and an arm that only asks whether it is non-zero cannot
 * tell a velocity increment from a momentum wearing its name. Each is
 * recomputed here from the other published components and the two
 * declared figures the statement carries. */
static void gate_increment_arithmetic_(void)
{
    double kin[KIN_N], las[LAS_N];
    char so[512];

    /* The impactor: the transferred momentum is the fraction that
     * landed times the projectile mass times the closing speed, and the
     * increment is that over the target's mass. */
    so_path_(so, sizeof so, "swarm");
    run_obs_(so, EFF_LAND_STEP, 0.0, kin, KIN_N);
    ASSERT(kin[2] == 1.0);
    double want = kin[6] * 50.0 * kin[3] / EFF_TARGET_MASS_KG;
    double rel  = fabs(kin[1] - want) / want;
    if (rel > 1.0e-12) {
        fprintf(stderr, "FAIL: the impactor published an increment of "
                "%.17g where the momentum balance over the published "
                "fraction (%.17g), the declared projectile mass "
                "(50 kg) and the published closing speed (%.17g) gives "
                "%.17g\n", kin[1], kin[6], kin[3], want);
        exit(1);
    }
    g_arms++;
    printf("  the impactor's increment is the momentum balance: %.9g "
           "against %.9g recomputed from the published fraction, the "
           "declared projectile mass and the published closing speed "
           "(relative difference %.3g)\n", kin[1], want, rel);

    /* The emitter: the increment is the published impulse over the
     * target's mass. The first engagement is the one to check, because
     * from the second onward the divisor is the mass the earlier
     * ablations left rather than the declared one, and that coupling
     * is what the mass arm above measures. Checking a later step here
     * would either restate that coupling or need a tolerance wide
     * enough to hide the defect this arm exists to catch. */
    so_path_(so, sizeof so, "las");
    run_obs_(so, 1, 0.0, las, LAS_N);
    double want_dv = las[1] / EFF_TARGET_MASS_KG;
    double rel_dv  = fabs(las[2] - want_dv) / want_dv;
    if (rel_dv > 1.0e-12) {
        fprintf(stderr, "FAIL: the emitter published an increment of "
                "%.17g where the published impulse (%.17g) over the "
                "target's declared mass gives %.17g\n",
                las[2], las[1], want_dv);
        exit(1);
    }
    g_arms++;
    printf("  the emitter's increment is the impulse over the target's "
           "mass: %.9g against %.9g (relative difference %.3g)\n",
           las[2], want_dv, rel_dv);
}

/* ---- Gate 6f: the release count decides the unit ---------------------- */

/* Three swarms of the same total mass, the same spread and the same
 * geometry, differing only in how many units the release is divided
 * into. The momentum is a property of the landed total and must not
 * move; the penetration analysis is a question about one arriving
 * unit and must.
 *
 * Without this the count is a required key nothing reads back, which
 * is what it was: three counts gave bit-identical channels and
 * bit-identical body state. */
#define EFF_SWARM_N_KFL(form, count) \
    "form " form "\n" \
    "fn world w\n" \
    EFF_EARTH EFF_SHOOTER("7746.0") \
    EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0") \
    "    astro_payload rock body=shooter kind=impactor pattern=swarm" \
    " swarm_count=" count " swarm_half_angle_rad=0.02" \
    " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0" \
    " projectile_diameter_m=0.2 target_wall_thickness_m=0.002" \
    " target_bumper_thickness_m=0.0016" \
    " target_bumper_density_kg_per_m3=2700.0" \
    " target_bumper_spacing_m=0.1 target_wall_yield_stress_ksi=40.0" \
    " target_brinell_hardness=95.0 target_density_kg_per_m3=2700.0" \
    " target_speed_of_sound_m_per_s=5100.0" \
    " target_monolithic_thickness_m=0.02\n" \
    EFF_EPISODE EFF_ACTION \
    "    observe effect rock as kin\n" \
    "    on_step\n" \
    "        engage rock at mover\n" \
    "    end\n" \
    "    objective\n" \
    "        reward kin_effect\n" \
    "    end\n" \
    "end\n" \
    "end\n"

static const char *const SWARM_N2_KFL    = EFF_SWARM_N_KFL("EFFN2", "2");
static const char *const SWARM_N12_KFL   = EFF_SWARM_N_KFL("EFFN12", "12");
static const char *const SWARM_N5000_KFL = EFF_SWARM_N_KFL("EFFN5000", "5000");

static void gate_swarm_count_(void)
{
    double n2[KIN_N], n12[KIN_N], n5000[KIN_N];
    char so[512];

    build_(SWARM_N2_KFL, "n2");
    build_(SWARM_N12_KFL, "n12");
    build_(SWARM_N5000_KFL, "n5000");
    so_path_(so, sizeof so, "n2");
    run_obs_(so, EFF_LAND_STEP, 0.0, n2, KIN_N);
    so_path_(so, sizeof so, "n12");
    run_obs_(so, EFF_LAND_STEP, 0.0, n12, KIN_N);
    so_path_(so, sizeof so, "n5000");
    run_obs_(so, EFF_LAND_STEP, 0.0, n5000, KIN_N);

    ASSERT(n2[2] == 1.0 && n12[2] == 1.0 && n5000[2] == 1.0);

    /* The momentum is the landed total's, so it does not move with the
     * count. An arm that only asked for a difference somewhere would
     * pass on a count that wrongly changed this too. */
    if (n2[1] != n12[1] || n2[1] != n5000[1]) {
        fprintf(stderr, "FAIL: the transferred momentum moved with the "
                "release count: %.17g, %.17g, %.17g at equal total "
                "mass\n", n2[1], n12[1], n5000[1]);
        exit(1);
    }

    /* The penetration depth is a unit's, so it falls as the release is
     * divided further. */
    if (!(n2[10] > n12[10] && n12[10] > n5000[10])) {
        fprintf(stderr, "FAIL: the penetration depth did not fall with "
                "the release count: %.17g, %.17g, %.17g\n",
                n2[10], n12[10], n5000[10]);
        exit(1);
    }

    /* And far enough down, a unit stops getting through the shield,
     * which moves the delivered energy with it. */
    if (!(n2[8] == 1.0 && n5000[8] == 0.0)) {
        fprintf(stderr, "FAIL: dividing the release into 5000 units did "
                "not stop it perforating: %.17g against %.17g at two "
                "units\n", n5000[8], n2[8]);
        exit(1);
    }
    if (!(n5000[11] < n2[11])) {
        fprintf(stderr, "FAIL: a release that no longer perforates "
                "delivered %.17g against %.17g\n", n5000[11], n2[11]);
        exit(1);
    }

    /* The critical diameter does not move, and that is not an
     * oversight. The ballistic-limit equation takes the projectile's
     * density and not its size, so the threshold a unit is compared
     * against is the same for every division of one release; what
     * moves is the unit's own diameter on the other side of that
     * comparison. Asserted so that a change making it move would be
     * noticed rather than welcomed. */
    if (n2[9] != n12[9] || n2[9] != n5000[9]) {
        fprintf(stderr, "FAIL: the critical diameter moved with the "
                "release count: %.17g, %.17g, %.17g; the ballistic-limit "
                "equation does not take the projectile's size\n",
                n2[9], n12[9], n5000[9]);
        exit(1);
    }
    g_arms += 4;
    printf("  the release count decides the unit: at equal total mass "
           "the increment holds at %.9g m/s and the critical diameter at "
           "%.9g m, while the penetration falls %.6g, %.6g, %.6g over "
           "counts 2, 12 and 5000, perforation stops, and the delivered "
           "energy falls from %.6g to %.6g J\n",
           n2[1], n2[9], n2[10], n12[10], n5000[10], n2[11], n5000[11]);
}

/* ---- Gate 6e: the bumper's thickness is its own key ------------------ */

/* The impact analysis takes the rear wall's thickness and the coupling
 * routine takes the bumper's, and until this pass one key was fed to
 * both. That is not a documentation question: the two decide different
 * branches, so a program declaring only a rear wall was silently
 * claiming a bumper of the same thickness and getting a different
 * delivered energy for it.
 *
 * Two fixtures differing in that one key, and in nothing else, with an
 * inner wall declared in both so the coupling routine's Whipple branch
 * and its fallback give different answers. If they agreed, the key
 * would have no consumer and the distinction would be a comment. */
#define EFF_SHIELD_KFL(form, bumper) \
    "form " form "\n" \
    "fn world w\n" \
    EFF_EARTH EFF_SHOOTER("7746.0") \
    EFF_MOVER("calibration_box.k26asm", "1.5e2", "0.0") \
    "    astro_payload rock body=shooter kind=impactor pattern=single" \
    " projectile_mass_kg=50.0 projectile_density_kg_per_m3=7800.0" \
    " projectile_diameter_m=0.2 target_wall_thickness_m=0.002" \
    bumper \
    " target_bumper_density_kg_per_m3=2700.0" \
    " target_bumper_spacing_m=0.1 target_wall_yield_stress_ksi=40.0" \
    " target_inner_thickness_m=0.003\n" \
    EFF_EPISODE EFF_ACTION \
    "    observe effect rock as kin\n" \
    "    on_step\n" \
    "        engage rock at mover\n" \
    "    end\n" \
    "    objective\n" \
    "        reward kin_effect\n" \
    "    end\n" \
    "end\n" \
    "end\n"

static const char *const SHIELD_WALL_KFL =
    EFF_SHIELD_KFL("EFFSHIELDWALL", "");
static const char *const SHIELD_BUMPER_KFL =
    EFF_SHIELD_KFL("EFFSHIELDBUMPER", " target_bumper_thickness_m=0.0016");

static void gate_bumper_key_(void)
{
    double wall[KIN_N], bumper[KIN_N];
    char so[512];

    build_(SHIELD_WALL_KFL, "shield_wall");
    build_(SHIELD_BUMPER_KFL, "shield_bumper");
    so_path_(so, sizeof so, "shield_wall");
    run_obs_(so, EFF_LAND_STEP, 0.0, wall, KIN_N);
    so_path_(so, sizeof so, "shield_bumper");
    run_obs_(so, EFF_LAND_STEP, 0.0, bumper, KIN_N);

    ASSERT(wall[2] == 1.0 && bumper[2] == 1.0);
    /* The penetration analysis reads the rear wall and not the bumper,
     * so the two fixtures must agree on it: a key that moved this as
     * well would still be feeding two layers. */
    if (wall[9] != bumper[9] || wall[8] != bumper[8]) {
        fprintf(stderr, "FAIL: declaring the bumper's thickness moved "
                "the penetration analysis (critical diameter %.17g "
                "against %.17g), which reads the rear wall\n",
                wall[9], bumper[9]);
        exit(1);
    }
    if (wall[11] == bumper[11]) {
        fprintf(stderr, "FAIL: declaring the bumper's thickness left the "
                "delivered energy at %.17g, so the key has no consumer\n",
                wall[11]);
        exit(1);
    }
    g_arms += 2;
    printf("  the bumper's thickness is its own key: declaring it leaves "
           "the critical diameter at %.9g and moves the delivered energy "
           "from %.9g to %.9g J\n", wall[9], wall[11], bumper[11]);
}

/* ---- Gate 6d: the effect points the way it should -------------------- */

/* Every arm so far has measured a magnitude, and a magnitude cannot
 * see a sign. Reversing the emitter's line-of-sight unit vector leaves
 * every published component bit-identical and pushes the target the
 * wrong way, which is a defect nothing above can fail on.
 *
 * The ablation plume leaves along the beam, so the recoil pushes the
 * target away from the emitter. This arm projects the target's change
 * in velocity, against a matched control, onto the emitter-to-target
 * line, and requires it positive and of the published magnitude. */
static void gate_effect_direction_(void)
{
    double las[6], ctrl[6];
    char so[512];

    so_path_(so, sizeof so, "las");
    run_body_(so, 1, 0.0, las);
    so_path_(so, sizeof so, "ctrl");
    run_body_(so, 1, 0.0, ctrl);

    /* The bodies are read relative to earth, so the emitter-to-target
     * line is the difference of their two positions. The shooter is
     * body 1 and the target body 2, and `run_body_` returns the
     * target's six, so the shooter's are fetched here. */
    EffArt a;
    so_path_(so, sizeof so, "las");
    art_open_(&a, so, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    ASSERT(a.s.step(a.env, act) == K26RL_OK);
    double b[64];
    ASSERT(a.s.bodies(a.env, 0u, b, 64u) == 18);
    double ux = las[0] - b[6], uy = las[1] - b[7], uz = las[2] - b[8];
    double un = sqrt(ux * ux + uy * uy + uz * uz);
    ASSERT(un > 0.0);
    ux /= un; uy /= un; uz /= un;

    double o[128];
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    double published = o[2];            /* the emitter's `_dv` */
    art_close_(&a);

    double dvx = las[3] - ctrl[3];
    double dvy = las[4] - ctrl[4];
    double dvz = las[5] - ctrl[5];
    double along = dvx * ux + dvy * uy + dvz * uz;

    if (!(along > 0.0)) {
        fprintf(stderr, "FAIL: the emitter's recoil pushed the target "
                "%.17g along the line from emitter to target, so it "
                "pushed it toward the emitter\n", along);
        exit(1);
    }
    double rel = fabs(along - published) / published;
    if (rel > 1.0e-6) {
        fprintf(stderr, "FAIL: the target's velocity change along the "
                "line of sight is %.17g and the published increment is "
                "%.17g\n", along, published);
        exit(1);
    }
    g_arms += 2;
    printf("  the recoil pushes the target away from the emitter: the "
           "target's velocity change projects %+.9g m/s on the "
           "emitter-to-target line, against a published increment of "
           "%.9g (relative difference %.3g)\n", along, published, rel);
}

/* ---- Gate 7: every published component moves ------------------------- */

/* The spread of each component over a run, so a component that cannot
 * move can be named rather than assumed to be fine. */
static void spread_(const char *stem, int n, int width, double *lo,
                    double *hi)
{
    char so[512];
    so_path_(so, sizeof so, stem);
    EffArt a;
    art_open_(&a, so, 4242u, 1u);
    double act[4] = { 0.0, 0.0, 0.0, 0.0 };
    for (int c = 0; c < width; c++) { lo[c] = 1.0e300; hi[c] = -1.0e300; }
    for (int i = 0; i < n; i++) {
        ASSERT(a.s.step(a.env, act) == K26RL_OK);
        double o[128];
        ASSERT(a.s.obs(a.env, o) == K26RL_OK);
        for (int c = 0; c < width; c++) {
            double v = o[c];
            if (v < lo[c]) lo[c] = v;
            if (v > hi[c]) hi[c] = v;
        }
    }
    art_close_(&a);
}

static void gate_components_move_(void)
{
    /* Three fixtures, because two of the components need a regime the
     * first one does not reach. The emitter's plasma components move
     * only once the fluence crosses the material's ignition threshold,
     * which the hot fixture is declared to straddle; the impactor's
     * footprint fraction is 1.0 for a single projectile by
     * construction and moves in the swarm fixture. Every component is
     * required to move in one of them, and the fixture that moved it
     * is named, so a component nothing can move is reported by name
     * rather than passed over.
     *
     * `_engaged` is the exception, and its own arm follows: it is 1.0
     * on every step of a fixture that engages every step, and the
     * conditional fixture is where it moves. */
    double lo_b[128], hi_b[128], lo_h[128], hi_h[128];
    double lo_s[128], hi_s[128];
    int moved = 0;

    build_(HOT_KFL, "hot");
    spread_("both", 30, LAS_N + KIN_N, lo_b, hi_b);
    spread_("hot", 12, LAS_N, lo_h, hi_h);
    spread_("swarm", 30, KIN_N, lo_s, hi_s);

    printf("    emitter spreads, the standing fixture then the hot one:");
    for (int c = 1; c < LAS_N; c++) {
        double sb = hi_b[c] - lo_b[c];
        double sh = hi_h[c] - lo_h[c];
        printf(" %s %.6g/%.6g", LAS_COMPS[c], sb, sh);
        if (sb > 0.0 || sh > 0.0) { moved++; continue; }
        printf("\n");
        fprintf(stderr, "FAIL: the emitter's `%s` held %.17g in the "
                "standing fixture and %.17g in the hot one\n",
                LAS_COMPS[c], lo_b[c], lo_h[c]);
        exit(1);
    }
    printf("\n    impactor spreads, the single fixture then the swarm:");
    for (int c = 1; c < KIN_N; c++) {
        double sb = hi_b[LAS_N + c] - lo_b[LAS_N + c];
        double ss = hi_s[c] - lo_s[c];
        printf(" %s %.6g/%.6g", KIN_COMPS[c], sb, ss);
        if (sb > 0.0 || ss > 0.0) { moved++; continue; }
        printf("\n");
        fprintf(stderr, "FAIL: the impactor's `%s` held %.17g in the "
                "single fixture and %.17g in the swarm one\n",
                KIN_COMPS[c], lo_b[LAS_N + c], lo_s[c]);
        exit(1);
    }
    printf("\n");
    g_arms++;
    printf("  %d of the %d published components move under a program a "
           "reader can write; `_engaged` is the remaining one and moves "
           "in the conditional fixture below\n",
           moved, LAS_N + KIN_N);
}

/* ---- Gate 8: zero fill on an unengaged step -------------------------- */

static void gate_zero_fill_(void)
{
    build_(COND_KFL, "cond");
    char so[512];
    so_path_(so, sizeof so, "cond");

    EffArt a;
    art_open_(&a, so, 4242u, 1u);
    double on[4]  = { 1.0, 0.0, 0.0, 0.0 };
    double off[4] = { 0.0, 0.0, 0.0, 0.0 };
    double o_on[128], o_off[128];

    ASSERT(a.s.step(a.env, on) == K26RL_OK);
    ASSERT(a.s.obs(a.env, o_on) == K26RL_OK);
    ASSERT(a.s.step(a.env, off) == K26RL_OK);
    ASSERT(a.s.obs(a.env, o_off) == K26RL_OK);
    art_close_(&a);

    ASSERT(o_on[0] == 1.0);
    ASSERT(o_on[LAS_N] == 1.0);
    int nonzero = 0;
    for (int c = 1; c < LAS_N + KIN_N; c++) {
        if (c == LAS_N) continue;
        if (o_on[c] != 0.0) nonzero++;
    }
    if (nonzero == 0) {
        fprintf(stderr, "FAIL: the engaged step published nothing, so the "
                "unengaged step below proves nothing\n");
        exit(1);
    }
    for (int c = 0; c < LAS_N + KIN_N; c++) {
        if (o_off[c] == 0.0) continue;
        fprintf(stderr, "FAIL: on a step that engaged nothing, component "
                "%d read %.17g\n", c, o_off[c]);
        exit(1);
    }
    g_arms += 2;
    printf("  a step whose body engaged reads `_engaged` 1.0 with %d "
           "further components non-zero; the next step of the same "
           "episode engages nothing and every one of the %d reads 0.0\n",
           nonzero, LAS_N + KIN_N);
}

/* ---- Gate 9: one engagement per payload per step, at runtime --------- */

static void gate_runtime_second_engage_(void)
{
    build_(LOOP_KFL, "loop");
    char so[512];
    so_path_(so, sizeof so, "loop");

    EffArt a;
    art_open_(&a, so, 4242u, 1u);
    double once[4]  = { 0.0, 0.0, 0.0, 0.0 };
    double twice[4] = { 1.0, 0.0, 0.0, 0.0 };
    uint16_t f = 0;
    uint32_t fl = 0;

    /* The loop runs once: no fault, and the engagement happened. */
    ASSERT(a.s.step(a.env, once) == K26RL_OK);
    ASSERT(a.s.fault_codes(a.env, &f) == K26RL_OK);
    if (f != 0) {
        fprintf(stderr, "FAIL: one engagement in a loop faulted with %u\n",
                (unsigned)f);
        exit(1);
    }
    double o[128];
    ASSERT(a.s.obs(a.env, o) == K26RL_OK);
    ASSERT(o[0] == 1.0);
    g_arms++;
    printf("  a loop engaging once does not fault\n");

    /* The same statement reached twice: the fault the compiler cannot
     * raise statically. */
    ASSERT(a.s.step(a.env, twice) == K26RL_OK);
    ASSERT(a.s.fault_codes(a.env, &f) == K26RL_OK);
    if (f != (uint16_t)K26RL_E_ENV_INTERNAL) {
        fprintf(stderr, "FAIL: a second engagement of one payload in one "
                "step reported fault code %u, expected %u\n",
                (unsigned)f, (unsigned)K26RL_E_ENV_INTERNAL);
        exit(1);
    }
    ASSERT(a.s.flags(a.env, &fl) == K26RL_OK);
    ASSERT((fl & K26RL_FLAG_FAULT) != 0u);
    art_close_(&a);
    g_arms++;
    printf("  the same statement reached twice through a loop faults with "
           "K26RL_E_ENV_INTERNAL (%u)\n", (unsigned)K26RL_E_ENV_INTERNAL);
}

int main(void)
{
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_run_or_die_("cp examples/assets/calibration_box.k26asm "
                   "examples/assets/calibration_box.k26mesh "
                   "examples/assets/crew_vehicle_10t.k26asm "
                   WORK_DIR "/");
    rl_write_file_(WORK_DIR "/scratch_bare.k26asm", BARE_ASM);

    printf("test_rl_effectors: the kinetic and directed-energy "
           "effectors\n");
    gate_refusals_();

    if (!rl_libs_present_("test_rl_effectors")) {
        printf("test_rl_effectors: %d arm(s) passed, drive arms stood "
               "down (stack archives absent)\n", g_arms);
        return 77;
    }

    must_build_and_step_("both kinds in one program", BOTH_KFL, "both");
    must_build_and_step_("the swarm release pattern", SWARM_KFL,
                         "swarm_accept");
    gate_channels_();
    gate_effect_world_();
    gate_mass_consumed_();
    gate_hit_test_();
    gate_swarm_();
    gate_silhouette_();
    gate_increment_arithmetic_();
    gate_swarm_count_();
    gate_bumper_key_();
    gate_effect_direction_();
    gate_components_move_();
    gate_zero_fill_();
    gate_runtime_second_engage_();

    printf("test_rl_effectors: %d arm(s) passed\n", g_arms);
    return 0;
}
