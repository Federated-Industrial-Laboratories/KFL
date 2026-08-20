/* emit_rl_internal.h - the reinforcement learning emitter's shared
 * spine: the program model, its limits, and the functions that cross
 * a module line. One emitter, nine translation units; the module
 * files each say which part of the pipeline they hold.
 */
#ifndef KFLC_EMIT_RL_INTERNAL_H
#define KFLC_EMIT_RL_INTERNAL_H

/* kflc - reinforcement learning environment emitter.
 *
 * A Grammar 3.2 program that declares reinforcement learning
 * constructs compiles to two artifacts from one emitted translation
 * unit: the ordinary batch executable and a companion shared object
 * exporting exactly the k26rl_ stepping surface (k26rl_env.h). This
 * file emits that translation unit. The driver compiles it twice,
 * once with KFLC_RL_BATCH_MAIN defined (adds main() and the batch
 * loop) and once as a shared object linked with a version script, so
 * batch and serve share every step of generated code by construction.
 *
 * The emitted environment:
 *   - builds n_envs worlds from the fn world prefix statements,
 *     exactly as the batch emitter would, one world per environment;
 *   - draws distribution-valued astro_body attributes from the
 *     domain-randomisation stream (class 0x0002) and episode reset
 *     lines from the reset-state stream (class 0x0001), one channel
 *     per parameter in source order, draw index 0 at
 *     (class, channel, environment, episode), through libk26rng;
 *   - resets an environment in place by restoring a body-state and
 *     epoch baseline captured at create and clearing the integrator
 *     transients, so episode k of environment j is a pure function of
 *     (seed, j, k) and the per-step path never allocates;
 *   - computes observation channels from `observe ... as` statements
 *     (dir_x, dir_y, dir_z, range per channel, source order), the
 *     reward from the objective block, termination from the episode
 *     block, and applies action channels in the on_step block;
 *   - captures the world prefix's top-level scalar bindings per
 *     environment at create, so objective and termination expressions
 *     read them with no allocation and no prefix re-run on the step
 *     path;
 *   - emits the episode record format through the libk26rl writer
 *     when output is enabled, and performs no I/O otherwise.
 */

#include "kflc.h"

#include "internal.h"

/* The imperfection layer's own kinds and channel conventions, so the
 * emitted tables carry the library's values rather than copies of
 * them. */

#include "k26sense.h"

#include "k26rl_env.h"

/* The plan format's own reader. A `reference=` on a body is read when
 * the program is compiled, exactly as an assembly is, so the emitted
 * artifact carries the knots as constants and a running simulation
 * opens no plan file. One decoder serves the compiler, the record and
 * any consumer, which is what keeps them from disagreeing about what
 * a plan says. */

#include "k26rl_ref.h"

/* The per-observer target cap the information state documents. The
 * compiler refuses a program that would exceed it, so the limit is
 * read from the library's own header rather than restated here. */

#include "k26astro_infostate/infostate_consts.h"

#include "assembly.h"

#include "capture.h"

#include <ctype.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

#include <math.h>

/* ---- Limits -------------------------------------------------------- */

/* Emit-time model sizes. These bound compiler-side scratch tables,
 * not the emitted program; a program over a limit gets a diagnostic
 * naming it rather than silent truncation. */

#define RL_MAX_BODIES   256

#define RL_MAX_ACTIONS  256

#define RL_MAX_OBSERVES 256

#define RL_MAX_RESETS   256

#define RL_MAX_DR       256

#define RL_MAX_WSCAL    256

#define RL_MAX_BS       128

#define RL_MAX_AGENTS   64

/* Bodies that may bind a plan. One plan addresses one craft, so this
 * bounds the craft under a plan rather than the plans in a file. */

#define RL_MAX_REF      16

/* The longest an agent name may be, in bytes, and the arithmetic it
 * comes out of. A published channel name entry holds KFLC_OBS_NAME_MAX
 * bytes; a declared `as` base name may be KFLC_OBS_AS_MAX of them; the
 * suffix figure below is 11, which was `_range_rate` when this bound
 * was set; and a qualified name spends one more byte on the separating
 * dot. What is left is the agent name's, and neither declarable bound
 * moves, because a name that compiles today must keep compiling. The
 * suffix figure does not move either, for the same reason: it is what
 * this bound was published at. Later forms have brought longer
 * suffixes, the kinetic effector's `_critical_diameter` at 18 being
 * the longest, and the combination test below is what holds for them.
 *
 * This is a bound on the name alone and not on the combination. A
 * paired channel inserts `_truth` before its component suffix, so a
 * `with truth` observe can derive 17 suffix bytes rather than 11, and
 * a name at this bound beside a base name at its own can still
 * overflow the entry. Every published name is therefore measured
 * where it is written as well, and the refusal there names both parts
 * rather than only the sum. */

#define RL_AGENT_SUFFIX_MAX 11

#define RL_AGENT_NAME_MAX \
    (KFLC_OBS_NAME_MAX - KFLC_OBS_AS_MAX - RL_AGENT_SUFFIX_MAX - 1)

/* The components one `observe ... as` contributes, in observation
 * vector order. One table, so the width and the names cannot drift
 * apart between the spec, the scope prelude, and the emitted
 * recompute. */

#define RL_OBS_COMPS 5

/* An attitude observe publishes a body's own orientation and rate
 * instead of a line of sight, so its channel set is its own. Widths
 * differ per observe from here on, which is why offsets are carried
 * rather than computed as an index times a constant. */

#define RL_ATT_COMPS 7

/* A contact observe publishes what a transition did rather than where
 * a body is: whether it contained a contact, at what fraction of the
 * control period the first one happened, and how fast the pair was
 * closing along the contact normal. Three channels rather than a flag
 * bit, so that a consumer gets when and how fast as well as whether,
 * and so that the frozen flag word is untouched. */

#define RL_CON_COMPS 3

/* A propulsion observe publishes what a craft has left to spend. The
 * fraction is beside the kilograms because it is the scale-free form
 * a policy conditions on more readily, and the current mass is beside
 * both because a craft's acceleration per unit of thrust is a
 * function of it. The fourth is the rocket equation read forwards:
 * specific impulse times standard gravity times the log of current
 * mass over the mass with an empty tank, which is what a planner
 * actually reasons with, since a manoeuvre is affordable or it is
 * not. */

#define RL_PROP_COMPS 4

/* A reference observe publishes where the craft's plan says it should
 * be next, as an error against where it is. The first six are errors
 * and not absolutes on purpose: a controller conditioned on where it
 * should be relative to where it is transfers between missions, and
 * one conditioned on an absolute position in a frame learns the
 * mission it was trained on. `_time_to` is the scale-free channel
 * among them and goes negative when the craft is late; `_tolerance`
 * is what makes the error interpretable rather than merely large or
 * small, since it separates a state that must be hit from one that
 * may be passed near.
 *
 * Only the current knot is published, whatever the plan's length.
 * That is what keeps the input width fixed, what keeps the controller
 * ignorant of the mission it is flying, and what lets a plan be
 * replaced without the controller knowing. */

#define RL_REF_COMPS 8

/* A relative observe publishes where the target is from the chief and
 * how fast it is moving there, in the chief's own local-vertical
 * local-horizontal axes: radial, along-track, cross-track. A target
 * thirty metres ahead and a target thirty metres below are different
 * problems, and the line-of-sight channels cannot tell them apart. */

/* The longest component suffix a channel can carry, with room for the
 * `_truth` a paired channel inserts and the terminator. */

#define RL_COMP_MAX 32

#define RL_REL_COMPS 6

/* A port observe publishes what a docking interface is doing: whether
 * the contact it just made satisfied every condition of its declared
 * capture envelope, the four residuals that condition is judged on,
 * and their four rates. The residuals are published every step, since
 * that is what a shaping term needs to fly an approach; on a step
 * whose transition contained a contact at this port they carry the
 * values the capture test itself was given, so nothing about the test
 * is invisible to the agent or to the record. */

#define RL_PORT_COMPS 9

/* The `full` mark adds two more: the rate the envelope's own note
 * binds, carried to the arriving vehicle's centre of mass, which
 * decides captures and was published nowhere; and whether the two
 * ports are joined, which is the standing state the capture channel
 * deliberately does not carry, being a pulse on the step of the
 * contact. Both are additive: an unmarked form publishes the nine
 * above and nothing else, so no program's observation vector changes
 * width or meaning when these arrive. */

#define RL_PORT_FULL_COMPS 11

/* A detection observe publishes whether the target was seen, the
 * continuous quantity that decision was taken on, and the geometry it
 * was taken from. Both the hard decision and the continuous quantity
 * are published so a program can shape a reward on one and terminate
 * on the other without deriving either from the other. The aspect
 * cosine is there because the signature models are aspect dependent:
 * without it an agent sees detections come and go with no channel
 * that explains them. These channels are geometric and carry no
 * light-time correction; the corrected view is the track form's. */

#define RL_DET_COMPS 7

/* An information-state observe publishes where the target was when
 * the light left it, how far away that was, and how old the picture
 * is, with an explicit validity channel because a history that does
 * not reach back to the retarded time produces no observation at all.
 * The solver's iteration count is deliberately absent: it is a
 * convergence diagnostic rather than a state of the world, and an
 * agent that could see it could learn the shape of the solver rather
 * than the shape of the problem. */

#define RL_TRK_COMPS 9

/* The defense payload kinds, one entry per `kind=` word this grammar
 * binds. Declared here rather than beside the key tables because an
 * effect observe's channel set is its payload's kind's, and the widths
 * are what the agent slices are computed from. */

typedef enum {
    RL_PAY_DETECT_IR    = 0,
    RL_PAY_DETECT_RADAR = 1,
    RL_PAY_DETECT_LIDAR = 2,
    RL_PAY_INFOSTATE    = 3,
    RL_PAY_IMPACTOR     = 4,
    RL_PAY_LASER        = 5,
    RL_PAY_DECOY        = 6,
    RL_PAY_JAMMER       = 7,
    /* The datalink, added with the shared information state. It is
     * declared through the same statement for surface uniformity and
     * it is the one kind that binds to no slot of the tier: the
     * payload machinery keys every slot by a registry tag, the
     * registry is not this layer's to change, and no tier evaluator
     * consumes a datalink. What it drives is this layer's own
     * transfer pass, beside the push discipline it extends. */
    RL_PAY_DATALINK     = 8
} RlPayloadKind;

#define RL_PAY_KINDS 9

/* An effector observe publishes what the last engagement of its
 * payload did. Two channels are common to every effector kind:
 * `_engaged`, which says whether the payload was engaged on this step,
 * and `_effect`, the scalar magnitude the kind's event reports as its
 * principal result. The rest are the kind's own, because an ablation
 * event and an impact event share no fields.
 *
 * Every channel here can be moved by a program a reader can write, and
 * four of the ablation event's thirteen fields are deliberately absent
 * for the opposite reason, each because nothing in a program can move
 * it: the power at the aperture is the declared output power times a
 * Strehl ratio computed from two declared constants; the threshold
 * fluence is a constant of the declared material; the effective dwell
 * is the control period; and the on-target intensity is the published
 * fluence divided by that period. A channel that cannot move is a
 * channel nothing can be gated on. The coupled power is published,
 * because it does move: it carries the plasma transmissivity and the
 * encircled fraction, and both of those move. */

#define RL_EFF_LAS_COMPS 11

#define RL_EFF_IMP_COMPS 12

/* The countermeasure kinds publish what they did to another craft's
 * payloads rather than what they did to its body, which is what makes
 * them the one effector class whose result is not a change of state.
 * `_reached` is the count of the victim's detection payloads the
 * engagement actually wrote to, and it is published rather than
 * inferred because an engagement aimed at a craft that carries nothing
 * to degrade is exactly the decoration this surface has to be able to
 * report. It reads zero on such a step, and the rest of the set reads
 * zero with it.
 *
 * The jammer's set carries both edges. `_effect` and `_reached` are the
 * benefit, `_self_signature` and the two counter-detection components
 * the cost: every watt transmitted announces the emitter's position to
 * a passive infrared observer, so a craft that jams becomes easier to
 * see while it does so, and both halves are on the same statement. */

#define RL_EFF_JAM_COMPS 9

/* `_deployed` is its own channel rather than something a reader infers
 * from a zero. A decoy's engagement is bounded by the host's own mass,
 * so a statement that ran and deployed nothing is a state a program can
 * reach and has to be able to see. */

#define RL_EFF_DEC_COMPS 8

/* The widest effector channel set, which is the per-payload stride of
 * the store the step writes and the observation reads. */

#define RL_EFF_MAX_COMPS RL_EFF_IMP_COMPS

/* Which form an observe is. The marker attribute the parser leaves is
 * what decides it, so the four are told apart in one place and the
 * width and the names cannot drift apart between them. */

typedef enum {
    RL_OBS_LOS  = 0,
    RL_OBS_ATT  = 1,
    RL_OBS_CON  = 2,
    RL_OBS_REL  = 3,
    RL_OBS_PORT = 4,
    RL_OBS_DET  = 5,
    RL_OBS_TRK  = 6,
    RL_OBS_EFF  = 7,
    RL_OBS_PROP = 8,
    RL_OBS_REF  = 9
} RlObserveForm;

/* ---- Program model -------------------------------------------------- */

typedef struct {
    const KflcNode *body;      /* KFLN_STMT_ASTRO_BODY */
    int             index;     /* world body index (source order) */
} RlBody;

/* Actuator descriptors, copied out of each body's assembly at model
 * build so the emitter needs no assembly afterwards. `veh` is the
 * vehicle slot, which is the order assembly-bearing bodies appear in;
 * `body` is the model body index; `name` is what a program commands
 * it by. */

#define RL_MAX_ACT 64

/* The bound on assembly-bearing bodies, which is separate from the
 * actuator bound above because the two count different things. It was
 * the actuator bound until this item, and at that value a program
 * could not reach the information state's own per-observer target cap
 * of 64: one observer plus 64 targets is 65 vehicles, so the vehicle
 * bound refused first and the cap's refusal could never fire. A limit
 * whose diagnostic cannot be produced is a limit nothing gates. */

#define RL_MAX_VEH 128

/* A declared sensor: its name, and the model terms it applies in the
 * order they were written. The terms are resolved from the block's
 * children once, at model build, so the emitter needs no parse tree
 * afterwards. A term that draws holds the channels the owning layer
 * allocated to it; one that draws at both cadences holds two, since a
 * per-episode draw and the first step's draw both take index 0. */

#define RL_MAX_SENSORS 32

#define RL_MAX_TERMS   16

typedef struct {
    int    kind;          /* a K26SenseKind value */
    double p0, p1, p2;    /* per kind, as k26sense.h documents */
    int    draws_step;
    int    draws_ep;
    int    line;
} RlSenseTerm;

typedef struct {
    const KflcNode *node;
    const char     *name;
    RlSenseTerm     terms[RL_MAX_TERMS];
    int             n_terms;
    int             depth;      /* the declared delay, in steps */
} RlSensor;

/* Colliders across every collidable body in one program. The
 * assembly reader's own limit is per assembly; this one is the
 * program's total, and it is a fixed size because the per-step
 * working set is preallocated and the loops are over fixed counts. */

#define RL_MAX_COLL 256

/* One collision primitive, already in its body's frame with the
 * component placement baked in by the assembly reader. The axes are
 * carried out in full rather than as a quaternion because that is the
 * form the kernels take and converting per step would be arithmetic
 * on the hot path for no gain. */

typedef struct {
    int    veh;
    int    body;
    int    kind;              /* the collision library's own kinds */
    double centre[3];
    double axis[3][3];
    double half[3];
} RlCollider;

/* ---- Defense payloads ----------------------------------------------- *
 *
 * One `astro_payload` statement covers the whole tier because the
 * libraries share one payload slot and one kind-tag registry, so the
 * admissible key set is a function of `kind=` and lives in the table
 * below rather than in a parser case per kind.
 *
 * Parameter order is this table's order, and it is the order the
 * per-environment parameter store is written and read in, so a key
 * cannot reach the wrong argument of a library call. The required keys
 * are the tier constructor's own parameters of the same name.
 *
 * Three parameters are not any constructor's. A detection evaluator
 * needs the target's radiometric properties as well as the
 * instrument's, and the tier's constructors do not carry them because
 * they describe what is being looked at rather than the instrument.
 * They are declared on the payload for two reasons: they are the
 * reference-target properties a detection threshold is specified
 * against, and declaring them here puts them on the same
 * domain-randomisation stream as every other payload parameter. The
 * consequence is stated rather than hidden: one detection payload
 * models one target class, and a program observing two dissimilar
 * targets declares one payload per class.
 *
 * Radar needs none of them for radiometry, its cross-section being
 * geometric; it does take the two that describe a chaff cloud around
 * the target, which raise that cross-section.
 *
 * The kind enumeration itself is declared with the effector channel
 * tables above, because an effect observe's width is its payload's
 * kind's and the widths are needed before this point. */

#define RL_PAY_MAXP     20

#define RL_MAX_PAYLOADS 32

/* A key whose value is a word rather than a number, and the constant
 * the word stands for. Two of the tier's constructor parameters are
 * enumerations, and a program that wrote the number would be depending
 * on a library's internal numbering that no document promises it. */

typedef struct {
    const char *word;
    const char *code;
} RlPayWord;

typedef struct {
    const char *key;
    int         required;
    const char *dflt;      /* emitted verbatim when the key is absent */
    /* NULL for a numeric key; otherwise the admissible words, and the
     * key takes one of them and no distribution form. */
    const RlPayWord *words;
    int              n_words;
    /* An open identifier this grammar resolves itself: the name of
     * another declaration, or the name of a community of them. It
     * takes no distribution form, because there is nothing between
     * two names to draw from, and it reaches the artifact as a
     * compile-time table rather than as a number in the parameter
     * store. */
    int              ident;
} RlPayKey;

/* The set of discriminators the observing craft runs against a decoy.
 * It is a property of the observer rather than of the decoy, which is
 * why it is declared on the detection payload and not on the payload
 * that deploys the countermeasure: two craft looking at one decoy may
 * run different discriminators and reach different answers about it.
 *
 * The words are the enumeration's own names. Nothing in the detection
 * library carries this: the type is declared in the countermeasure
 * library's constants header and is taken there by the discrimination
 * routine alone, so the value has to be declared somewhere, and the
 * observer's own statement is where it belongs. */

#define RL_PAY_REGIME_KEY \
    { "discriminator_regime", 0, "K26ASTRO_DISC_IR_ONLY", \
      RL_PAY_REGIME_, \
      (int)(sizeof RL_PAY_REGIME_ / sizeof RL_PAY_REGIME_[0]), 0 }

/* The two chaff keys are the cloud statistics routine's own parameters
 * of the same name, prefixed like every other key that describes what
 * is being looked at rather than the instrument. A chaff cloud is not
 * a payload of this tier: the countermeasure library gives it four free
 * functions, no handle and no registry tag, and its only entry point
 * that takes a generator is the per-sample draw, which this layer never
 * calls. What it does have is a deterministic mean cross-section, and
 * that is a property of the target a radar looks at, so it is declared
 * beside the other reference-target properties.
 *
 * Both are optional and the pair is all-or-nothing in one direction:
 * the strip count is what says a cloud is there, and a dipole
 * cross-section declared without one would describe strips that do not
 * exist. */

/* Where each detection kind keeps the discriminator regime, since the
 * three kinds have different key counts and the decoy engagement reads
 * one slot per victim payload. One place, so a key appended to a kind
 * above cannot leave the engagement reading a neighbour. */

#define RL_PAY_IR_REGIME    10

#define RL_PAY_RADAR_REGIME  9

#define RL_PAY_LIDAR_REGIME  7

/* The chaff pair's slots on the radar kind, for the same reason. */

#define RL_PAY_RADAR_CHAFF_N     10

#define RL_PAY_RADAR_CHAFF_SIG   11

/* The release pattern, which decides whether the swarm keys below are
 * required or refused and whether a spread footprint is computed at
 * all. It is a word rather than a number because the enumeration's
 * values are the library's own numbering. */

/* The kinetic impactor.
 *
 * The first four keys and the two swarm keys are the constructor's own
 * parameters of the same name. The nine `target_` keys describe the
 * structure of what is being hit rather than the projectile, and they
 * are optional: the library's own analysis skips the Whipple branch
 * unless a wall thickness is declared and the monolithic branch unless
 * a hardness, a density and a speed of sound are, and this grammar
 * mirrors that rather than inventing required keys. The penetration
 * channels read zero where the analysis was skipped, and the effect on
 * the target's motion needs none of them: momentum transfer is a
 * function of the projectile and the closing geometry alone.
 *
 * Seven of the ten are `k26astro_impactor_analyse_impact`'s own
 * parameters, spelled as it spells them. The other three are fields of
 * the target-structure specification the delivered-energy routine
 * takes, prefixed `target_` for consistency with the seven beside
 * them: the bumper the stand-off carries, the wall behind it, and the
 * thickness the monolithic penetration depth is judged against.
 *
 * The bumper's thickness is its own key and not the wall's. The
 * ballistic-limit equation's first parameter is the *rear wall's*
 * thickness, which is what `target_wall_thickness_m` means and what
 * the impact analysis is given; the coupling routine's
 * `outer_thickness_m` is the *bumper's*, which is a different layer of
 * the same shield. Feeding one key to both would make a program that
 * declared a rear wall silently claim a bumper of the same thickness,
 * and the two decide different things. */

/* The target material, which selects the coupling coefficient, the
 * plasma-ignition threshold and the specific ablation energy the
 * ablation chain uses.
 *
 * The library's own enumeration carries a seventh entry, `NONE`, whose
 * documented meaning is a conservative generic-metal anchor rather
 * than an absence of material. This grammar does not admit it: a
 * program that wrote `target_material=none` would reasonably expect no
 * ablation and would get steel's numbers, so the six materials are
 * named and the anchor is written as the material it actually is. The
 * words are the enumeration's own names, including its spelling of
 * aluminium, because they name that enumeration's entries. */

/* The directed-energy emitter. The first seven keys are the
 * constructor's own parameters of the same name; the two `target_`
 * keys are the evaluator's, and they describe what is being fired at
 * rather than the emitter, which is why no constructor carries them.
 * As with the detection kinds, one laser payload models one target
 * class. */

/* The decoy's modality, which selects the discrimination table the
 * probability is read from: a cold balloon and a heated, thrusting
 * surrogate are caught by different discriminators at very different
 * rates. The words are the enumeration's own names. */

/* The decoy. Every key is the constructor's own parameter of the same
 * name. There are no `target_` keys: what a decoy has to match is its
 * own host, which it is deployed from, and the match qualities are how
 * well it does so on each of the three discriminator channels. */

/* The jammer's waveform. The words are the enumeration's own names. */

/* The jammer. The first six keys are the constructor's own parameters
 * of the same name. `radiator_temp_k` is not any constructor's: it is
 * the emitter radiator temperature the counter-detection routine takes,
 * and it is required rather than optional because it is what prices the
 * engagement. A jammer announces its own position with every watt it
 * transmits, and a payload that published the benefit and let the cost
 * be left undeclared would model an advantage that costs nothing. */

/* The jammer's own slots, read by the engagement. */

#define RL_PAY_JAMMER_RADIATOR 6

/* The datalink's own slots. `network` is an identifier and `rate_hz`
 * is the broadcast cadence; the nine below them are the one-way link
 * budget's, read by this layer's own kernel and by nothing in the
 * tier. They are spelled as the radar row spells them, less the
 * cross-section a one-way link has no use for, so a reader who knows
 * one radio row knows the other. */
#define RL_PAY_LINK_NETWORK   0
#define RL_PAY_LINK_RATE      1
#define RL_PAY_LINK_P_TX      2
#define RL_PAY_LINK_G_TX      3
#define RL_PAY_LINK_G_RX      4
#define RL_PAY_LINK_FREQ      5
#define RL_PAY_LINK_LOSS      6
#define RL_PAY_LINK_BW        7
#define RL_PAY_LINK_T_SYS     8
#define RL_PAY_LINK_NF        9
#define RL_PAY_LINK_THRESHOLD 10

/* The information state's `source` slot, which names the detection
 * payload whose verdict gates the push. */
#define RL_PAY_INFO_SOURCE 1

typedef struct {
    const char     *name;         /* the `kind=` value */
    /* The registry's own tag name, or NULL for a kind that binds to
     * no slot of the tier and therefore carries no tag. */
    const char     *tag;
    const RlPayKey *keys;
    int             n_keys;
    int             is_detect;
    int             is_effector;  /* the registry's effector class */
    /* An effector whose result lands on another payload's capability
     * rather than on a body. The two are not exclusive: a decoy also
     * takes mass off its host. */
    int             is_softkill;
    /* A kind this layer implements itself, with no handle constructed
     * and no slot of the tier taken. The construction loop, the
     * teardown switch and the registry-tag cross-check all stand
     * down for it, since there is nothing to construct, free or
     * check. */
    int             is_link;
} RlPayKindDesc;

/* A kind the tier's registry names and no library in the tree
 * implements. The registry allocates its tag and reserves it against
 * reassignment, and the effector-class range includes it, so a reader
 * of that header would reasonably expect to be able to write it. There
 * is no constructor, no evaluator and no event struct behind it, so
 * this grammar refuses it with what is actually the matter rather than
 * with the message for a misspelling. */

typedef struct {
    const char *name;
    const char *tag;
} RlPayUnimplemented;

/* The kinds this grammar binds, for the diagnostics that list them.
 * One string rather than a sentence per refusal, so a kind added to
 * the table above cannot be missing from half the messages. */

#define RL_PAY_KIND_LIST \
    "detect_ir, detect_radar, detect_lidar, infostate, impactor, laser, " \
    "decoy, jammer and datalink"

typedef struct {
    const KflcNode *node;
    const char     *name;
    int             kind;                 /* an RlPayloadKind */
    int             body;                 /* index into the model's bodies */
    int             veh;                  /* vehicle slot of that body */
    const KflcAttr *attr[RL_PAY_MAXP];    /* declared key, or NULL */
    KflcExpr       *dist[RL_PAY_MAXP];    /* distribution form, or NULL */
    int             dr[RL_PAY_MAXP];      /* draw slot, or -1 */
    /* The detection payload an information state's `source=` names,
     * or -1 where the key is absent and the push is truth-fed. */
    int             src_pay;
    /* A datalink's community, as an index into the model's network
     * table, and the information state its carrier holds. Both -1 on
     * every other kind. */
    int             net;
    int             info_pay;
    int             line;
} RlPayload;

/* One `engage <payload> at <target>` inside the step body, resolved to
 * the payload it fires and the body it is aimed at. The statements are
 * collected in source order over the whole block, including the
 * branches of any conditional in it, and the emitted helper for each
 * carries that index: an engagement's geometry is a function of two
 * bodies that are fixed at compile time, so nothing about it is looked
 * up while stepping. */

#define RL_MAX_ENGAGE 32

typedef struct {
    KflcNode *node;
    int       payload;
    int       target;
    int       line;
} RlEngage;

/* One primitive of a detection target's silhouette, in that body's own
 * frame. The projected area of the set along the line of sight is the
 * aspect-dependent geometry the tier's signature models take, and it
 * is derived from the colliders the assembly already declares rather
 * than from a second description of the same craft. */

#define RL_MAX_SIG 256

typedef struct {
    int    body;
    int    kind;              /* 1 sphere, 2 capsule, 3 box */
    double axis[3][3];
    double half[3];
} RlSigPrim;

typedef struct {
    int    veh, body;
    char   name[KFLC_ASM_NAME_MAX];
    double axis[3];
    double spin_inertia, max_momentum, max_torque;
    double viscous, coulomb, dead_rate;
} RlWheel;

typedef struct {
    int    veh, body;
    char   name[KFLC_ASM_NAME_MAX];
    double axis[3];
    double max_dipole;
} RlTorquer;

typedef struct {
    int    veh, body;
    char   name[KFLC_ASM_NAME_MAX];
    double at[3], dir[3];
    double max_thrust;
    double isp_s;
} RlThruster;

/* One docking port that named a capture envelope, copied out of its
 * assembly at model build. `coll` is the index, within this program's
 * whole collider set, of the mating plane the assembly reader built
 * from the envelope's published diameter: it is what tells a contact
 * at this interface from a contact anywhere else on the craft. The
 * basis is orthonormal and body-frame, its first axis the outward
 * normal of the mating plane. The envelope travels in SI, converted
 * once from the printed figures. */

typedef struct {
    int    veh, body, coll;
    char   name[KFLC_ASM_NAME_MAX];
    /* The envelope the port's `capture` mark named. Two ports of one
     * pairing must name the same one, which is a comparison of names
     * and not of the limits behind them: two envelopes with equal
     * figures are still two interfaces. */
    char   env_name[KFLC_CAPTURE_NAME_MAX];
    double com[3];
    double at[3];
    double basis[3][3];
    double axial_rate_min, axial_rate_max;
    double lateral_rate, pitchyaw_rate, roll_rate;
    double lateral, pitchyaw, roll;
    double diameter;
} RlPort;

/* One resolved actuator command or read inside on_step. */

typedef struct {
    int kind;      /* 0 wheel, 1 magnetorquer, 2 thruster */
    int index;     /* into the model's array for that kind */
    int field;     /* 0 command, 1 momentum, 2 rate */
    int written;
} RlActRef;

/* A distribution-valued declaration on the domain-randomisation
 * stream. Exactly one of `body` and `payload` is a real index and the
 * other is -1: a draw belongs to the body attribute or to the payload
 * key that declared it. Channels are allocated in source order over
 * the bodies and then over the payloads, which for a program that
 * declares no payload is the allocation it always had, so no existing
 * program's draw coordinates move. */

typedef struct {
    int             body;      /* index into bodies[], or -1 */
    int             payload;   /* index into payloads[], or -1 */
    int             param;     /* parameter slot when payload >= 0 */
    const KflcAttr *attr;      /* the distribution-valued attribute */
    KflcExpr       *dist;      /* parsed uniform/normal call */
    int             channel;   /* class 0x0002 channel */
} RlDrParam;

/* One (body, state key) pair an on_step body reads or assigns. */

typedef struct {
    int body;      /* index into the model's bodies[] */
    int key;       /* index into the shared body state key table */
    int written;   /* 1 when an assignment targets it */
} RlStateRef;

/* One agent: the block that declares it, and the ranges of the flat
 * declaration arrays it owns. Blocks are collected in source order and
 * each block's declarations in source order within it, so an agent's
 * ranges are contiguous and the concatenation over agents is the whole
 * of each vector. That is what keeps `obs_total` and `act_total`
 * meaning what they mean, and it is what makes each published slice a
 * partition rather than a lookup.
 *
 * A world with no `agent` block is agent count 1 with one implicit
 * agent owning everything and publishing unqualified names, which is
 * the shape every program had before the block existed. */

typedef struct {
    const KflcNode *node;       /* the block, or NULL when implicit */
    const char     *name;       /* NULL for the implicit agent */
    int             obs_first, n_obs;   /* range in the model's observes */
    int             act_first, n_act;   /* range in the model's actions */
    int             obs_off, obs_count; /* observation vector slice */
    int             act_off, act_count; /* action vector slice */
    const KflcNode *objective;  /* or NULL: an all-zero reward stream */
    const KflcAttr *reward;
    const KflcAttr *terminal;
} RlAgent;

/* A plan bound to a body by `reference=`. The file is read here, when
 * the program is compiled, and what the emitted artifact carries is
 * the decoded knots as constants beside the file's own bytes: the
 * knots because the stepping path must open nothing and allocate
 * nothing, and the bytes because the record has to carry the plan a
 * run flew against. Both come out of one reader, so the plan the
 * record names and the plan the craft flew cannot be two things. */

typedef struct {
    int             body;        /* the craft the plan addresses */
    int             frame_body;  /* the body the plan's frame centres on */
    uint32_t        frame_kind;
    double          epoch;
    int             n_knots;
    const double   *knots;       /* n_knots by 8, arena owned */
    const uint8_t  *bytes;       /* the file, verbatim, arena owned */
    uint32_t        len;
    uint8_t         digest[K26RL_SHA256_BYTES];
} RlReference;

/* A `plan` block: the knot slots a planner emits and where the plan
 * they make goes. `act_first` is the index of the first of the
 * block's action channels in the environment's action vector, which
 * is where the episode-end write reads its knots from. */

typedef struct {
    const KflcNode *node;
    const char     *name;
    const char     *file;        /* path prefix, unquoted */
    const char     *provenance;  /* unquoted, or "" */
    int             frame_body;
    uint32_t        frame_kind;
    int             slots;
    double          epoch;
    int             act_first;
} RlPlanOut;

typedef struct {
    const KflcNode *world;
    const KflcNode *episode;
    const KflcNode *on_step;    /* or NULL */
    const KflcNode *objective;  /* world level, or NULL */

    /* Agent count is n_agents when any `agent` block is declared and 1
     * otherwise, the implicit agent occupying slot 0 either way. */
    RlAgent         agents[RL_MAX_AGENTS];
    int             n_agents;      /* declared blocks; 0 when implicit */
    int             agent_count;   /* published count, never below 1 */

    /* Which block owns each declaration, by index into agents[], or -1
     * for one written at world level. Mixing the two is refused, so
     * after collection these are all -1 or all non-negative. */
    int             act_owner[RL_MAX_ACTIONS];
    int             obs_owner[RL_MAX_OBSERVES];

    RlBody          bodies[RL_MAX_BODIES];
    int             n_bodies;

    const KflcNode *actions[RL_MAX_ACTIONS];
    int             n_actions;

    const KflcNode *observes[RL_MAX_OBSERVES];   /* observe ... as */
    int             n_observes;

    RlSensor        sensors[RL_MAX_SENSORS];
    int             n_sensors;
    /* Per observe, the sensor it declared with `through`, or -1. */
    int             obs_sensor[RL_MAX_OBSERVES];

    const KflcNode *resets[RL_MAX_RESETS];       /* episode reset lines */
    int             n_resets;

    RlDrParam       dr[RL_MAX_DR];
    int             n_dr;

    /* Top-level scalar (double, int, bool) let/const bindings of the
     * world prefix. Their values are captured per environment when the
     * world is built at create, so the objective and termination
     * expressions can read them without re-running the prefix and
     * without allocating on the step path. */
    const KflcNode *wscal[RL_MAX_WSCAL];
    int             n_wscal;

    /* Body state the on_step body reaches, in first-mention order.
     * One accessor pair is emitted per entry. */
    RlStateRef      bs[RL_MAX_BS];
    int             n_bs;

    const KflcAttr *control_dt;
    const KflcAttr *substeps;
    const KflcAttr *contact_kind;   /* `arrest` or `bounce`, or NULL */
    const KflcAttr *restitution;
    const KflcAttr *friction;
    RlWheel     wheels[RL_MAX_ACT];
    int         n_wheels;
    RlTorquer   torquers[RL_MAX_ACT];
    int         n_torquers;
    RlThruster  thrusters[RL_MAX_ACT];
    int         n_thrusters;
    RlPort      ports[RL_MAX_ACT];
    int         n_ports;
    /* The `capture_envelope` blocks this program declared, in source
     * order. The values themselves live in the compiler's envelope
     * table, which is what a port's `capture` mark resolves against;
     * what is kept here is the declaration, so a second block of the
     * same name can be refused naming the first one's line. */
    const KflcNode *capenv[KFLC_CAPTURE_MAX_DECLARED];
    int         n_capenv;
    double      veh_com[RL_MAX_VEH][3];
    double      veh_bound[RL_MAX_VEH];
    /* Per vehicle, the mass properties as a function of the
     * propellant left, copied out of its assembly. A vehicle whose
     * assembly declares no tank has a capacity of zero, aggregates
     * covering the whole of it, and never enters the consumption
     * path. `veh_isp` is the thrust-weighted mean specific impulse of
     * the vehicle's thrusters, which is the figure the published
     * remaining velocity change is taken at where they differ; it is
     * a constant because the weights are the declared maxima. */
    double      veh_prop_cap[RL_MAX_VEH];
    double      veh_struct_mass[RL_MAX_VEH];
    double      veh_struct_moment[RL_MAX_VEH][3];
    double      veh_struct_inertia[RL_MAX_VEH][6];
    double      veh_prop_centroid[RL_MAX_VEH][3];
    double      veh_prop_inertia[RL_MAX_VEH][6];
    double      veh_isp[RL_MAX_VEH];
    int         veh_has_prop[RL_MAX_VEH];
    int         n_prop_veh;
    int         n_veh;
    RlCollider  colliders[RL_MAX_COLL];
    int         n_colliders;
    RlPayload   payloads[RL_MAX_PAYLOADS];
    int         n_payloads;
    /* The datalink communities this program names, in the order they
     * are first declared. A community is a name and nothing else:
     * every datalink carrying it participates, with no addressing and
     * no acknowledgement, so the table exists to tell one community
     * from another and for no other purpose. */
    const char *nets[RL_MAX_PAYLOADS];
    int         n_nets;
    /* Per observe of a defense form: the payload it names and the body
     * it observes, resolved once so the emitter needs no lookups. Both
     * are -1 for every other form. */
    int         obs_payload[RL_MAX_OBSERVES];
    int         obs_target[RL_MAX_OBSERVES];
    /* The plans bound to bodies by `reference=`, and, per observe, the
     * plan a reference observe reads, or -1. */
    RlReference refs[RL_MAX_REF];
    int         n_refs;
    int         obs_ref[RL_MAX_OBSERVES];
    RlPlanOut   plans[RL_MAX_REF];
    int         n_plans;
    RlEngage    engages[RL_MAX_ENGAGE];
    int         n_engages;
    RlSigPrim   sig[RL_MAX_SIG];
    int         n_sig;
    int         pay_nparam;         /* widest parameter set in use */
    RlActRef    acts[RL_MAX_ACT];
    int         n_acts;
    const KflcAttr *horizon;          /* or NULL */
    const KflcAttr *terminated_when;  /* or NULL */
    const KflcAttr *reward;           /* or NULL */
    const KflcAttr *terminal;         /* or NULL */
} RlModel;

/* One vehicle's emission data, captured while its body is emitted and
 * spent once every body has been added.
 *
 * A vehicle binds a `K26AstroBody *` taken from the world, and that
 * pointer is stable across everything the world does except a further
 * `k26astro_world_add_body`, which reallocates the body array
 * (k26astro_rt/world.h). Binding while bodies are still arriving
 * therefore leaves every earlier vehicle pointing into freed memory,
 * so the binds are deferred to a second pass and the derived constants
 * are carried here across the assembly arena's release. */

typedef struct {
    const char *body_name;
    double      mass;
    double      com[3];
    double      inertia[6];
    /* A vehicle that carries propellant takes its mass properties
     * from the one function that evaluates them at a fill, so that
     * construction, reset and the burn cannot come out differently at
     * the same fill. One that carries none keeps the constants the
     * assembly derived, unchanged, and never enters that path. */
    int         has_prop;
    double      prop_cap;
} RlVehEmit;

/* ---- What a stepping path may reach ---------------------------------- */

/* One sweep answers the whole question, because reaching is reaching
 * whether the last hop is an expression or a statement. From an
 * expression that runs on the stepping path it follows operands, call
 * arguments, and calls into user function bodies to any depth; inside
 * those bodies it judges every statement as well as every expression.
 * A forbidden statement one indirection away is the same defect as a
 * forbidden call written in place, and an earlier form of this walk
 * judged expressions only, so `astro_body` inside a called function
 * added a body to the world on every step of every environment and
 * checked clean.
 *
 * Three things end a sweep, and each is reported with what was
 * reached and the chain of functions that got there:
 *   - a builtin the registry does not declare pure;
 *   - a statement or binding the stepping path may not carry;
 *   - a call the compiler cannot classify at all.
 *
 * The walk fails closed on all three counts. A call name that is
 * neither a registered builtin nor a user function is a form the
 * compiler lowers itself, and some of those allocate, so it is
 * refused rather than passed over. A statement kind that is neither
 * admitted nor named below is refused for the same reason. And a
 * chain deeper than the walk can follow is refused rather than
 * assumed pure, because a limit that returns "clean" when it runs out
 * of room is a limit an author can step over. */

#define RL_SWEEP_MAX_FNS 64

typedef struct {
    const KflcNode *form;
    const char     *found;      /* the name that ended the sweep */
    const char     *why;        /* why it may not be reached; NULL for
                                 * an undeclared builtin, whose wording
                                 * is fixed */
    int             depth;      /* the sweep ran out of room */
    int             line;       /* line of the offending statement */
    const char     *chain[RL_SWEEP_MAX_FNS];   /* fns entered, in order */
    int             n_chain;
    const char     *seen[RL_SWEEP_MAX_FNS];    /* fns already walked */
    int             n_seen;
} RlSweep;

/* ---- The positions inside the per-step block ------------------------- */

/* The rule above, applied to every position an expression can occupy
 * in the on_step body: assignments, initialisers, expression
 * statements, conditions, index expressions, attribute expressions,
 * and anything nested inside a block. Enforcing it on the assigned
 * expression alone left a `let`, a bare expression statement and a
 * condition able to reach the same builtins, so it is a property of
 * the block rather than of one statement form in it.
 *
 * Following an initialiser is what follows the dataflow into a
 * binding: a name bound to a rejected call is refused where it is
 * bound, so no later read of it has to be traced.
 *
 * Each statement is judged as a statement first and then for the
 * expressions it evaluates, which is the same order the walk above
 * uses inside a called function, so a form written here and a form
 * reached through a call get the same answer.
 *
 * Positions are named in the diagnostic because they fail in
 * different-looking ways, and a reader told only that the block is
 * impure has to find the call for themselves. */

#define RL_STEP_POS_MAX 192

/* ---- Entry ---------------------------------------------------------- */


extern const RlPayKindDesc RL_PAY_KIND_[RL_PAY_KINDS];

/* ---- The functions that cross a module line ---- */

int rl_act_agent(const RlModel *m, int i);
const KflcAttr *rl_attr(const KflcNode *n, const char *key);
KflcExpr *rl_attr_dist(const KflcAttr *a, const KflcNode *form,
                               KflcArena *arena, KflcDiag *diag,
                               int *out_error);
const KflcAttr *rl_body_attr(const KflcNode *body, const char *name);
int rl_body_has_assembly(const RlModel *m, int bi);
int rl_body_index_of(const RlModel *m, const char *name);
int rl_collect(RlModel *m, const KflcNode *form,
                       KflcArena *arena, KflcDiag *diag);
const KflcNode *rl_find_world_binding(const KflcNode *stmts,
                                              const char *name, int depth,
                                              int *out_top);
int rl_is_scalar_type(KflcType t);
int rl_is_state_key(const char *k);
int rl_obs_agent(const RlModel *m, int i);
int rl_obs_offset(const KflcNode *const *obs, int i);
int rl_obs_pub_name(const RlModel *m, int i, int c,
                            char *out, size_t cap, KflcDiag *diag);
int rl_obs_total(const KflcNode *const *obs, int n);
const char *rl_observe_as(const KflcNode *n);
int rl_observe_base_width(const KflcNode *n);
const char *rl_observe_comp(const KflcNode *n, int c,
                                    char *buf, size_t cap);
RlObserveForm rl_observe_form(const KflcNode *n);
int rl_observe_has_truth(const KflcNode *n);
int rl_observe_is_full(const KflcNode *n);
int rl_observe_is_attitude(const KflcNode *n);
uint16_t rl_observe_mode(const KflcNode *n);
const char *rl_observe_payload(const KflcNode *n);
int rl_observe_width(const KflcNode *n);
int rl_scope_name_taken(const RlModel *m, const char *name);
int rl_veh_slot_of(const RlModel *m, int body);
int rl_collect_plans(RlModel *m, KflcArena *arena, KflcDiag *diag);
int rl_collect_references(RlModel *m, KflcArena *arena,
                                  KflcDiag *diag);
int rl_finish_defense(RlModel *m, KflcDiag *diag);
int rl_finish_payloads(RlModel *m, const KflcNode *form,
                               KflcArena *arena, KflcDiag *diag);
const char *rl_pay_attr_text(const KflcAttr *a);
const char *rl_pay_ctype(int kind);
const char *rl_pay_dtor(int kind);
int rl_pay_kind_from_name(const char *s);
void rl_stamp_effect_kinds(RlModel *m, KflcArena *arena);
int rl_track_modality(const KflcNode *n, const char **out_bad);
void rl_emit_batch_main(FILE *out, const KflcNode *form);
void rl_emit_env_core(FILE *out);
void rl_action_sites(const RlModel *m, const char *name,
                             int *a0, int *l0, int *a1, int *l1);
int rl_agent_has_channel(const RlModel *m, int ag, const char *name);
int rl_agent_index_of(const RlModel *m, const char *name);
void rl_bs_fn_name(const RlModel *m, int slot, int set,
                           char *out, size_t cap);
int rl_bs_rewrite_stmts(RlModel *m, KflcNode *stmts,
                                KflcArena *arena, KflcDiag *diag);
int rl_channel_owner(const RlModel *m, const char *name,
                             int *n_owners);
void rl_collect_form_args(const KflcNode *form, KflcArena *arena,
                                  KflcExprBinding **live,
                                  int *live_n, int *live_cap);
void rl_collect_lets(const KflcNode *n, KflcArena *arena,
                             KflcExprBinding **live, int *live_n,
                             int *live_cap);
int rl_detect_observes(const RlModel *m, int p, int b);
int rl_dotted_split(const char *name, char *lhs, size_t lcap,
                            char *rhs, size_t rcap);
void rl_emit_body_write(FILE *out, int indent, const char *lv,
                                const char *key, const char *value_text);
void rl_emit_body_write_var(FILE *out, int indent, const char *lv,
                                    const char *key, const char *var);
int rl_emit_draw(FILE *out, const KflcExpr *dist,
                         unsigned cls, int channel,
                         const KflcExprCtx *ctx, KflcDiag *diag);
void rl_emit_scope_prelude(FILE *out, const RlModel *m, int ag,
                                   int indent);
void rl_emit_string_literal(FILE *out, const char *s);
int rl_has_relative_observe(const RlModel *m);
int rl_n_chaff(const RlModel *m);
int rl_n_detect(const RlModel *m);
int rl_n_effector(const RlModel *m);
int rl_n_pay_kind(const RlModel *m, int kind);
int rl_n_softkill_engage(const RlModel *m);
int rl_n_vehicles(const RlModel *m);
void rl_push_binding(KflcArena *arena, KflcExprBinding **live,
                             int *live_n, int *live_cap,
                             const char *name, KflcType type);
void rl_qual_ident(int agent, const char *chan, char *out,
                           size_t cap);
void rl_qual_rewrite_expr(const RlModel *m, KflcExpr *e,
                                  KflcArena *arena);
void rl_rewrite_steps(KflcExpr *e, KflcArena *arena);
void rl_scope_bindings(const RlModel *m, int ag,
                               const KflcNode *form,
                               KflcArena *arena, KflcExprBinding **live,
                               int *live_n, int *live_cap);
int rl_softkill_reaches(const RlModel *m, int p, int t,
                                int want_jammer);
int rl_track_pairs(const RlModel *m, int *pay, int *veh, int cap);
int rl_n_link(const RlModel *m);
int rl_track_gate_pay(const RlModel *m, int payload);
int rl_emit_link_tables(FILE *out, const RlModel *m, KflcDiag *diag);
void rl_emit_detect_eval(FILE *out, const RlModel *m, int p, int tgt);
int rl_emit_observe(FILE *out, const RlModel *m,
                            KflcDiag *diag);
int rl_emit_objective(FILE *out, const RlModel *m,
                              const KflcNode *form, KflcArena *arena,
                              KflcExprFn *user_fn_arr, int n_user_fns,
                              KflcDiag *diag);
int rl_emit_on_step(FILE *out, RlModel *m,
                            const KflcNode *form, KflcArena *arena,
                            KflcExprFn *user_fn_arr, int n_user_fns,
                            KflcDiag *diag);
void rl_emit_actuators(FILE *out, const RlModel *m);
void rl_emit_mass_tables(FILE *out, const RlModel *m);
int rl_emit_payload_tables(FILE *out, const RlModel *m);
void rl_emit_plan_tables(FILE *out, const RlModel *m);
void rl_emit_reference_tables(FILE *out, const RlModel *m);
int rl_emit_sensors(FILE *out, const RlModel *m, KflcDiag *diag);
int rl_n_sensed(const RlModel *m);
int rl_emit_apply_draws(FILE *out, const RlModel *m,
                                const KflcExprCtx *arg_ctx, KflcDiag *diag);
int rl_emit_build_world(FILE *out, const RlModel *m,
                                const KflcNode *form, KflcArena *arena,
                                KflcExprFn *user_fn_arr, int n_user_fns,
                                KflcDiag *diag);
void rl_emit_form_args(FILE *out, const KflcNode *form);
int rl_emit_params(FILE *out, const RlModel *m,
                           const KflcExprCtx *arg_ctx, KflcDiag *diag);
int rl_emit_prologue(FILE *out, const RlModel *m,
                             const KflcNode *form, KflcDiag *diag);
int rl_emit_user_fns(FILE *out, const KflcNode *form,
                             KflcArena *arena, KflcExprFn *user_fn_arr,
                             int n_user_fns, KflcDiag *diag);
#endif /* KFLC_EMIT_RL_INTERNAL_H */
