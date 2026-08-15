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

/* The published observer-mode value of an as-bound observe. The
 * grammar's default when no mode= attribute is given is the runtime's
 * default, astrometric, which is what the observation path selects. */
static uint16_t rl_observe_mode_(const KflcNode *n)
{
    if (!n) return 1;
    /* An attitude observe reports a body's own orientation and rate.
     * There is no observer, no light time and no aberration, so the
     * only true answer among the published modes is the geometric
     * one; the default astrometric value would claim a correction
     * that is not applied. */
    for (const KflcAttr *k = n->attrs; k; k = k->next) {
        if (!k->name) continue;
        if (strcmp(k->name, "attitude") == 0) return 0;
        /* A contact observe has no observer either, and applies no
         * correction of any kind: it reports what a transition did.
         * Geometric is true of it in the one sense the tag was minted
         * for, and astrometric would assert a correction that is not
         * made. */
        if (strcmp(k->name, "contact") == 0) return 0;
        /* A relative observe has an observer, the chief, but applies
         * no light-time correction and no aberration to the state it
         * publishes: it resolves a state the integrator already
         * produced onto a set of axes. Geometric is exactly what that
         * is; astrometric would assert a correction that is not
         * made. */
        if (strcmp(k->name, "relative") == 0) return 0;
        /* A port observe reports the geometry of one docking interface
         * against another, both of which the integrator already
         * produced. No correction of any kind is applied to it. */
        if (strcmp(k->name, "port") == 0) return 0;
    }
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "mode") == 0 &&
            a->value.kind == KFLV_IDENT && a->value.u.s) {
            if (strcmp(a->value.u.s, "geometric") == 0) return 0;
            if (strcmp(a->value.u.s, "astrometric") == 0) return 1;
            if (strcmp(a->value.u.s, "apparent") == 0) return 2;
            if (strcmp(a->value.u.s, "topocentric") == 0) return 3;
        }
    }
    return 1;
}

static int rl_is_state_key_(const char *k)
{
    return kflc_body_state_key_index(k) >= 0;
}

/* The components one `observe ... as` contributes, in observation
 * vector order. One table, so the width and the names cannot drift
 * apart between the spec, the scope prelude, and the emitted
 * recompute. */
#define RL_OBS_COMPS 5
static const char *const RL_OBS_COMP_[RL_OBS_COMPS] = {
    "_dir_x", "_dir_y", "_dir_z", "_range", "_range_rate"
};

/* An attitude observe publishes a body's own orientation and rate
 * instead of a line of sight, so its channel set is its own. Widths
 * differ per observe from here on, which is why offsets are carried
 * rather than computed as an index times a constant. */
#define RL_ATT_COMPS 7
static const char *const RL_ATT_COMP_[RL_ATT_COMPS] = {
    "_quat_w", "_quat_x", "_quat_y", "_quat_z",
    "_omega_x", "_omega_y", "_omega_z"
};

/* A contact observe publishes what a transition did rather than where
 * a body is: whether it contained a contact, at what fraction of the
 * control period the first one happened, and how fast the pair was
 * closing along the contact normal. Three channels rather than a flag
 * bit, so that a consumer gets when and how fast as well as whether,
 * and so that the frozen flag word is untouched. */
#define RL_CON_COMPS 3
static const char *const RL_CON_COMP_[RL_CON_COMPS] = {
    "_hit", "_fraction", "_speed"
};

/* A relative observe publishes where the target is from the chief and
 * how fast it is moving there, in the chief's own local-vertical
 * local-horizontal axes: radial, along-track, cross-track. A target
 * thirty metres ahead and a target thirty metres below are different
 * problems, and the line-of-sight channels cannot tell them apart. */
/* The longest component suffix a channel can carry, with room for the
 * `_truth` a paired channel inserts and the terminator. */
#define RL_COMP_MAX 32

#define RL_REL_COMPS 6
static const char *const RL_REL_COMP_[RL_REL_COMPS] = {
    "_r_x", "_r_y", "_r_z", "_v_x", "_v_y", "_v_z"
};

/* A port observe publishes what a docking interface is doing: whether
 * the contact it just made satisfied every condition of its declared
 * capture envelope, the four residuals that condition is judged on,
 * and their four rates. The residuals are published every step, since
 * that is what a shaping term needs to fly an approach; on a step
 * whose transition contained a contact at this port they carry the
 * values the capture test itself was given, so nothing about the test
 * is invisible to the agent or to the record. */
#define RL_PORT_COMPS 9
static const char *const RL_PORT_COMP_[RL_PORT_COMPS] = {
    "_captured", "_axial", "_lateral", "_pitchyaw", "_roll",
    "_v_axial", "_v_lateral", "_v_pitchyaw", "_v_roll"
};

/* Which form an observe is. The marker attribute the parser leaves is
 * what decides it, so the four are told apart in one place and the
 * width and the names cannot drift apart between them. */
typedef enum {
    RL_OBS_LOS  = 0,
    RL_OBS_ATT  = 1,
    RL_OBS_CON  = 2,
    RL_OBS_REL  = 3,
    RL_OBS_PORT = 4
} RlObserveForm;

static RlObserveForm rl_observe_form_(const KflcNode *n)
{
    if (!n) return RL_OBS_LOS;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (!a->name) continue;
        if (strcmp(a->name, "attitude") == 0) return RL_OBS_ATT;
        if (strcmp(a->name, "contact") == 0)  return RL_OBS_CON;
        if (strcmp(a->name, "relative") == 0) return RL_OBS_REL;
        if (strcmp(a->name, "port") == 0)     return RL_OBS_PORT;
    }
    return RL_OBS_LOS;
}

static int rl_observe_is_attitude_(const KflcNode *n)
{
    return rl_observe_form_(n) == RL_OBS_ATT;
}

/* The two switches below name every form and carry no default, so a
 * form added to the enum without a width and a name table is a
 * compiler warning rather than five silent line-of-sight channels. The
 * return after each switch is what the language requires, not a
 * fallback the code relies on. */
/* Whether the observe declared `through <sensor>`, and whether it
 * asked for the uncorrupted values beside the measured ones. */
static const char *rl_observe_through_(const KflcNode *n)
{
    if (!n) return NULL;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "through") == 0 &&
            a->value.kind == KFLV_IDENT) {
            return a->value.u.s;
        }
    }
    return NULL;
}

static int rl_observe_has_truth_(const KflcNode *n)
{
    if (!n) return 0;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "truth") == 0) return 1;
    }
    return 0;
}

/* The components one form publishes, before any pairing. */
static int rl_observe_base_width_(const KflcNode *n)
{
    switch (rl_observe_form_(n)) {
    case RL_OBS_ATT: return RL_ATT_COMPS;
    case RL_OBS_CON: return RL_CON_COMPS;
    case RL_OBS_REL: return RL_REL_COMPS;
    case RL_OBS_PORT: return RL_PORT_COMPS;
    case RL_OBS_LOS: return RL_OBS_COMPS;
    }
    return RL_OBS_COMPS;
}

/* What the observe contributes to the observation vector. `with truth`
 * publishes the uncorrupted components as a paired set beside the
 * measured ones, so the observe is twice as wide: the measured half
 * first, then the truth half, each in component order. Every offset,
 * the total, and the base of every write follow from this one
 * function, so the pairing cannot displace a neighbouring observe. */
static int rl_observe_width_(const KflcNode *n)
{
    int w = rl_observe_base_width_(n);
    return rl_observe_has_truth_(n) ? 2 * w : w;
}

static const char *rl_observe_base_comp_(const KflcNode *n, int c)
{
    switch (rl_observe_form_(n)) {
    case RL_OBS_ATT: return RL_ATT_COMP_[c];
    case RL_OBS_CON: return RL_CON_COMP_[c];
    case RL_OBS_REL: return RL_REL_COMP_[c];
    case RL_OBS_PORT: return RL_PORT_COMP_[c];
    case RL_OBS_LOS: return RL_OBS_COMP_[c];
    }
    return RL_OBS_COMP_[c];
}

/* The suffix of channel `c` of this observe, written into `buf`. A
 * channel in the truth half carries `_truth` before its component, so
 * a reader sees `<name>_truth_range` beside `<name>_range`. The buffer
 * is the caller's because two suffixes are live at once wherever a
 * pair is emitted. */
static const char *rl_observe_comp_(const KflcNode *n, int c,
                                    char *buf, size_t cap)
{
    int w = rl_observe_base_width_(n);
    if (c < w) return rl_observe_base_comp_(n, c);
    snprintf(buf, cap, "_truth%s", rl_observe_base_comp_(n, c - w));
    return buf;
}

/* The first channel index of observe `i`, and the total width. Both
 * walk the declarations in source order, which is the order channels
 * are allocated in. */
static int rl_obs_offset_(const KflcNode *const *obs, int i)
{
    int off = 0;
    for (int k = 0; k < i; k++) off += rl_observe_width_(obs[k]);
    return off;
}

static int rl_obs_total_(const KflcNode *const *obs, int n)
{
    return rl_obs_offset_(obs, n);
}

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

typedef struct {
    int             body;      /* index into bodies[] */
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

typedef struct {
    const KflcNode *world;
    const KflcNode *episode;
    const KflcNode *on_step;    /* or NULL */
    const KflcNode *objective;  /* or NULL */

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
    double      veh_com[RL_MAX_ACT][3];
    double      veh_bound[RL_MAX_ACT];
    int         n_veh;
    RlCollider  colliders[RL_MAX_COLL];
    int         n_colliders;
    RlActRef    acts[RL_MAX_ACT];
    int         n_acts;
    const KflcAttr *horizon;          /* or NULL */
    const KflcAttr *terminated_when;  /* or NULL */
    const KflcAttr *reward;           /* or NULL */
    const KflcAttr *terminal;         /* or NULL */
} RlModel;

static const KflcAttr *rl_attr_(const KflcNode *n, const char *key)
{
    for (const KflcAttr *a = n ? n->attrs : NULL; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0) return a;
    }
    return NULL;
}

static const char *rl_observe_as_(const KflcNode *n)
{
    const KflcAttr *a = rl_attr_(n, "as");
    if (a && a->value.kind == KFLV_IDENT && a->value.u.s) return a->value.u.s;
    return NULL;
}

/* A user-declared fn shadows the distribution reading of its name,
 * mirroring the checker's rule. */
static int rl_form_declares_fn_(const KflcNode *form, const char *name)
{
    if (!form || !name) return 0;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_FN && c->name && strcmp(c->name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Word-boundary scan of attribute value text for a call to `name`,
 * matching the checker's classification so emission and check agree
 * on what counts as a distribution position. */
static int rl_text_has_call_(const char *text, const char *name)
{
    if (!text || !name) return 0;
    size_t nl = strlen(name);
    for (const char *p = text; *p; p++) {
        if (strncmp(p, name, nl) != 0) continue;
        if (p != text) {
            char b = p[-1];
            if (b == '_' || isalnum((unsigned char)b)) continue;
        }
        const char *q = p + nl;
        if (*q == '_' || isalnum((unsigned char)*q)) continue;
        while (*q == ' ' || *q == '\t') q++;
        if (*q == '(') return 1;
    }
    return 0;
}

/* Parse an astro_body attribute's verbatim value text and return the
 * expression when the whole value is a two-argument uniform/normal
 * call whose name is not shadowed; NULL otherwise. A value that
 * merely contains a distribution call somewhere inside a larger
 * expression is reported: a drawn parameter is the whole value or
 * nothing. */
static KflcExpr *rl_attr_dist_(const KflcAttr *a, const KflcNode *form,
                               KflcArena *arena, KflcDiag *diag,
                               int *out_error)
{
    if (a->value.kind != KFLV_IDENT || !a->value.u.s) return NULL;
    const char *text = a->value.u.s;
    int has_uniform = rl_text_has_call_(text, "uniform") &&
                      !rl_form_declares_fn_(form, "uniform");
    int has_normal  = rl_text_has_call_(text, "normal") &&
                      !rl_form_declares_fn_(form, "normal");
    if (!has_uniform && !has_normal) return NULL;

    KflcDiag scratch;
    kflc_diag_init(&scratch, "", stderr);
    /* Parse errors fall through to verbatim splicing; the downstream
     * C++ compile reports them as it always has for attribute text. */
    FILE *devnull = fopen("/dev/null", "w");
    if (devnull) scratch.errstream = devnull;
    KflcExpr *e = kflc_parse_expr(text, arena, &scratch, a->line);
    if (devnull) fclose(devnull);

    int is_dist = (e && scratch.errors == 0 &&
                   e->kind == KFLE_CALL && e->u.call.name &&
                   (strcmp(e->u.call.name, "uniform") == 0 ||
                    strcmp(e->u.call.name, "normal") == 0) &&
                   e->u.call.n_args == 2 &&
                   !rl_form_declares_fn_(form, e->u.call.name));
    if (is_dist) return e;

    /* A distribution call inside a larger expression is not a
     * drawable parameter. */
    {
        kflc_diag_errorf(diag, a->line,
            "astro_body %s: a distribution must be the whole attribute "
            "value (`%s=uniform(lo, hi)` or `%s=normal(mu, sigma)`), not "
            "part of a larger expression", a->name, a->name, a->name);
        *out_error = 1;
    }
    return NULL;
}

static int rl_is_scalar_type_(KflcType t)
{
    return t == KFLT_DOUBLE || t == KFLT_INT || t == KFLT_BOOL;
}

/* True when `name` is already claimed by the RL expression scope: an
 * action channel, an observation channel component, or the episode
 * step counter. A world binding with such a name is shadowed there and
 * never captured. */
static int rl_scope_name_taken_(const RlModel *m, const char *name)
{
    if (!name) return 0;
    if (strcmp(name, "episode.steps") == 0) return 1;
    for (int i = 0; i < m->n_actions; i++) {
        const char *an = m->actions[i]->name;
        if (an && strcmp(an, name) == 0) return 1;
    }
    for (int i = 0; i < m->n_observes; i++) {
        const char *base = rl_observe_as_(m->observes[i]);
        if (!base) continue;
        size_t bl = strlen(base);
        if (strncmp(name, base, bl) != 0) continue;
        for (int c = 0; c < rl_observe_width_(m->observes[i]); c++) {
            char cb[RL_COMP_MAX];
            if (strcmp(name + bl,
                       rl_observe_comp_(m->observes[i], c, cb, sizeof cb))
                == 0) {
                return 1;
            }
        }
    }
    return 0;
}

/* Find a let/const named `name` anywhere in the world subtree.
 * out_top reports whether the hit is a direct child of the world
 * (the prefix's own scope) rather than nested inside a block. */
static const KflcNode *rl_find_world_binding_(const KflcNode *stmts,
                                              const char *name, int depth,
                                              int *out_top)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        if ((s->kind == KFLN_STMT_LET || s->kind == KFLN_STMT_CONST) &&
            s->name && strcmp(s->name, name) == 0) {
            *out_top = (depth == 0);
            return s;
        }
        const KflcNode *hit =
            rl_find_world_binding_(s->children, name, depth + 1, out_top);
        if (hit) return hit;
        hit = rl_find_world_binding_(s->else_children, name, depth + 1,
                                     out_top);
        if (hit) return hit;
    }
    return NULL;
}

/* Collect the program model from the form. Returns 0 on success;
 * nonzero after reporting diagnostics. */
/* Whether a model body binds a vehicle assembly. Attitude state is
 * only advanced for a body that does, since the advance needs the
 * inertia tensor the assembly derives, so the three places that can
 * set attitude all ask this before accepting it. */
static int rl_body_has_assembly_(const RlModel *m, int bi)
{
    if (bi < 0 || bi >= m->n_bodies) return 0;
    for (const KflcAttr *a = m->bodies[bi].body->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "assembly") == 0) return 1;
    }
    return 0;
}

/* Copy each assembly-bearing body's actuators into the model, in
 * vehicle order and, within a vehicle, in the order the assembly
 * declares them. That order is the command surface's order and the
 * emitted tables', so a program's actuators are named by the asset
 * and indexed the same way everywhere. */
/* The attribute of that name on a body, or NULL. */
static const KflcAttr *rl_body_attr_(const KflcNode *body, const char *name)
{
    if (!body || !name) return NULL;
    for (const KflcAttr *a = body->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, name) == 0) return a;
    }
    return NULL;
}

/* Read one `sensor` block into the model. Each child is one term; the
 * operands were parsed as plain numbers and are checked here against
 * the term's own domain, because a negative standard deviation or a
 * probability above one is a declaration nobody can mean. */
static double rl_term_num_(const KflcNode *t, int i, int *have)
{
    char key[8];
    snprintf(key, sizeof key, "n%d", i);
    for (const KflcAttr *a = t->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0) {
            if (have) *have = 1;
            return a->value.u.f;
        }
    }
    if (have) *have = 0;
    return 0.0;
}

static int rl_collect_sensor_(RlModel *m, const KflcNode *n, KflcDiag *diag)
{
    if (m->n_sensors >= RL_MAX_SENSORS) {
        kflc_diag_errorf(diag, n->line,
            "more than %d sensors in this program", RL_MAX_SENSORS);
        return 1;
    }
    for (int i = 0; i < m->n_sensors; i++) {
        if (m->sensors[i].name && n->name &&
            strcmp(m->sensors[i].name, n->name) == 0) {
            kflc_diag_errorf(diag, n->line,
                "sensor `%s` is declared twice", n->name);
            return 1;
        }
    }
    RlSensor *sn = &m->sensors[m->n_sensors];
    memset(sn, 0, sizeof *sn);
    sn->node = n;
    sn->name = n->name;

    for (const KflcNode *c = n->children; c; c = c->next) {
        if (c->kind != KFLN_STMT_SENSOR_TERM) continue;
        if (sn->n_terms >= RL_MAX_TERMS) {
            kflc_diag_errorf(diag, c->line,
                "sensor `%s`: more than %d terms", sn->name, RL_MAX_TERMS);
            return 1;
        }
        RlSenseTerm *t = &sn->terms[sn->n_terms];
        memset(t, 0, sizeof *t);
        t->line = c->line;
        const char *kw = c->name ? c->name : "";
        int h0 = 0, h1 = 0, h2 = 0;
        double v0 = rl_term_num_(c, 0, &h0);
        double v1 = rl_term_num_(c, 1, &h1);
        double v2 = rl_term_num_(c, 2, &h2);

        if (strcmp(kw, "noise") == 0) {
            const char *dist = NULL;
            for (const KflcAttr *a = c->attrs; a; a = a->next) {
                if (a->name && strcmp(a->name, "dist") == 0 &&
                    a->value.kind == KFLV_IDENT) dist = a->value.u.s;
            }
            if (!dist || strcmp(dist, "normal") != 0) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `noise` takes `normal`, the only "
                    "distribution this layer offers", sn->name);
                return 1;
            }
            if (!h0 || !h1) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `noise normal` takes a mean and a "
                    "standard deviation", sn->name);
                return 1;
            }
            if (v0 != 0.0) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `noise normal` takes a mean of zero; "
                    "a constant offset is a bias, not noise", sn->name);
                return 1;
            }
            if (!(v1 >= 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: a standard deviation cannot be "
                    "negative", sn->name);
                return 1;
            }
            t->kind = K26SENSE_ADDITIVE;
            t->p0 = v1;
            t->draws_step = 1;
        } else if (strcmp(kw, "scale") == 0) {
            if (!h0 || !(v0 >= 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `scale` takes one relative standard "
                    "deviation, not negative", sn->name);
                return 1;
            }
            t->kind = K26SENSE_SCALE;
            t->p0 = v0;
            t->draws_step = 1;
        } else if (strcmp(kw, "bias_walk") == 0) {
            if (!h0 || !h1 || !h2) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `bias_walk` takes a turn-on standard "
                    "deviation, a correlation time in seconds, and an "
                    "in-run standard deviation", sn->name);
                return 1;
            }
            if (!(v0 >= 0.0) || !(v1 > 0.0) || !(v2 >= 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `bias_walk` needs a positive "
                    "correlation time and standard deviations that are "
                    "not negative", sn->name);
                return 1;
            }
            t->kind = K26SENSE_BIAS_WALK;
            t->p0 = v0; t->p1 = v1; t->p2 = v2;
            t->draws_step = 1;
            t->draws_ep = 1;
        } else if (strcmp(kw, "latency") == 0) {
            if (!h0 || v0 < 0.0 || v0 != (double)(int)v0 || v0 > 1024.0) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `latency` takes a whole number of "
                    "control periods, from 0 to 1024", sn->name);
                return 1;
            }
            t->kind = K26SENSE_LATENCY;
            t->p0 = v0;
            if ((int)v0 > sn->depth) sn->depth = (int)v0;
        } else if (strcmp(kw, "quantise") == 0) {
            if (!h0 || !(v0 > 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `quantise` takes a positive step",
                    sn->name);
                return 1;
            }
            if (h1 != h2) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `quantise` takes a step alone, or a "
                    "step with both ends of its range", sn->name);
                return 1;
            }
            if (h1 && !(v1 < v2)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `quantise` needs a range whose low "
                    "end is below its high end", sn->name);
                return 1;
            }
            /* A range and a step together decide whether the
             * quantiser's grid position stays inside a signed 64-bit
             * integer. Beyond that the conversion is undefined, and a
             * program must not be able to reach undefined behaviour by
             * declaring a fine step, so the pair is checked here and
             * the library saturates if anything ever slips past.
             *
             * The implied range is the narrower of a working ceiling
             * and what the step itself can reach: a step of `lsb`
             * spans at most 2^63 multiples either side of zero, so a
             * step of 1e-14 reaches about 9.2e4 and not 1e15. Taking
             * 1e15 unconditionally is what let a true value of seven
             * million be published as minus ninety-two thousand. */
            double span = v0 * K26SENSE_I64_SPAN;
            double cap  = span < 1.0e15 ? span : 1.0e15;
            if (h1) {
                double reach = (v2 > -v1 ? v2 : -v1) / v0;
                if (!(reach < K26SENSE_I64_SPAN)) {
                    kflc_diag_errorf(diag, c->line,
                        "sensor `%s`: `quantise` with a step of %g over "
                        "a range reaching %g needs %g grid positions, "
                        "and a 64-bit integer holds %g; widen the step "
                        "or narrow the range", sn->name, v0,
                        (v2 > -v1 ? v2 : -v1), reach,
                        K26SENSE_I64_SPAN);
                    return 1;
                }
            }
            t->kind = K26SENSE_QUANTISE;
            t->p0 = v0;
            t->p1 = h1 ? v1 : -cap;
            t->p2 = h1 ? v2 :  cap;
        } else if (strcmp(kw, "dropout") == 0) {
            if (!h0 || !(v0 >= 0.0) || !(v0 < 1.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `dropout` takes a probability from 0 "
                    "up to but not including 1", sn->name);
                return 1;
            }
            t->kind = K26SENSE_DROPOUT;
            t->p0 = v0;
            t->draws_step = 1;
        } else {
            kflc_diag_errorf(diag, c->line,
                "sensor `%s`: unknown model term `%s`", sn->name, kw);
            return 1;
        }
        sn->n_terms++;
    }

    if (sn->n_terms == 0) {
        kflc_diag_errorf(diag, n->line,
            "sensor `%s` declares no model terms, so it would leave "
            "every channel it touches unchanged", sn->name);
        return 1;
    }
    int n_bias = 0, n_delay = 0;
    for (int i = 0; i < sn->n_terms; i++) {
        if (sn->terms[i].kind == K26SENSE_BIAS_WALK) n_bias++;
        if (sn->terms[i].kind == K26SENSE_LATENCY)   n_delay++;
    }
    if (n_bias > 1 || n_delay > 1) {
        kflc_diag_errorf(diag, n->line,
            "sensor `%s`: one bias walk and one latency per sensor, "
            "since one carries one state of each", sn->name);
        return 1;
    }
    m->n_sensors++;
    return 0;
}

static int rl_collect_actuators_(RlModel *m, KflcDiag *diag)
{
    int veh = 0;
    for (int i = 0; i < m->n_bodies; i++) {
        if (!rl_body_has_assembly_(m, i)) continue;
        KflcArena *ar = kflc_arena_create();
        if (!ar) return 1;
        KflcAssembly *a = NULL;
        if (kflc_assembly_for_body(m->bodies[i].body, diag->path, ar, diag,
                                   &a) || !a) {
            kflc_arena_release(ar);
            return 1;
        }
        if (veh >= RL_MAX_ACT) {
            kflc_diag_errorf(diag, m->bodies[i].body->line,
                "more than %d bodies carry an assembly", RL_MAX_ACT);
            kflc_arena_release(ar);
            return 1;
        }
        for (int k = 0; k < 3; k++) m->veh_com[veh][k] = a->com[k];
        m->veh_bound[veh] = a->bound_radius;

        /* The collider set, taken in declaration order, which is the
         * order the narrowphase tests them in and therefore the order
         * the selection rule's tie-break is defined over. The reader
         * has already baked the component placements in, so these are
         * body-frame and comparable across components. */
        for (int k = 0; k < a->n_colliders; k++) {
            const KflcAsmCollider *cl = &a->colliders[k];
            if (m->n_colliders >= RL_MAX_COLL) {
                kflc_diag_errorf(diag, cl->line,
                    "more than %d colliders in this program", RL_MAX_COLL);
                kflc_arena_release(ar);
                return 1;
            }
            RlCollider *rc = &m->colliders[m->n_colliders++];
            memset(rc, 0, sizeof *rc);
            rc->veh = veh; rc->body = i;
            for (int q = 0; q < 3; q++) {
                rc->centre[q] = cl->centre[q];
                for (int w = 0; w < 3; w++) rc->axis[q][w] = cl->rot[w][q];
            }
            if (cl->kind == KFLC_SHAPE_SPHERE) {
                rc->kind = 1;
                rc->half[0] = cl->r;
            } else if (cl->kind == KFLC_SHAPE_CAPSULE) {
                /* The kernels take a capsule as a centre, a direction,
                 * a radius, and half the segment's length, so the
                 * reader's two endpoints are converted here once
                 * rather than on every step. */
                rc->kind = 2;
                double d[3] = { cl->b[0] - cl->a[0], cl->b[1] - cl->a[1],
                                cl->b[2] - cl->a[2] };
                double len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                rc->half[0] = cl->r;
                rc->half[2] = 0.5 * len;
                if (len > 0.0) {
                    for (int q = 0; q < 3; q++) rc->axis[2][q] = d[q] / len;
                    /* Any pair completing the frame will do: a capsule
                     * is symmetric about its axis, so only the third
                     * direction carries meaning. */
                    double up[3] = { 0.0, 0.0, 1.0 };
                    if (rc->axis[2][2] > 0.9 || rc->axis[2][2] < -0.9) {
                        up[0] = 1.0; up[2] = 0.0;
                    }
                    double ax[3] = {
                        up[1] * rc->axis[2][2] - up[2] * rc->axis[2][1],
                        up[2] * rc->axis[2][0] - up[0] * rc->axis[2][2],
                        up[0] * rc->axis[2][1] - up[1] * rc->axis[2][0] };
                    double al = sqrt(ax[0] * ax[0] + ax[1] * ax[1] +
                                     ax[2] * ax[2]);
                    for (int q = 0; q < 3; q++) rc->axis[0][q] = ax[q] / al;
                    rc->axis[1][0] = rc->axis[2][1] * rc->axis[0][2]
                                   - rc->axis[2][2] * rc->axis[0][1];
                    rc->axis[1][1] = rc->axis[2][2] * rc->axis[0][0]
                                   - rc->axis[2][0] * rc->axis[0][2];
                    rc->axis[1][2] = rc->axis[2][0] * rc->axis[0][1]
                                   - rc->axis[2][1] * rc->axis[0][0];
                }
            } else {
                rc->kind = 3;
                for (int q = 0; q < 3; q++) rc->half[q] = cl->a[q];
            }
        }

        for (int f = 0; f < a->n_features; f++) {
            const KflcAsmFeature *ft = &a->features[f];
            if (ft->kind == KFLC_FEAT_WHEEL) {
                if (m->n_wheels >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d wheels in this program", RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlWheel *w = &m->wheels[m->n_wheels++];
                w->veh = veh; w->body = i;
                snprintf(w->name, sizeof w->name, "%s", ft->name);
                for (int k = 0; k < 3; k++) w->axis[k] = ft->axis[k];
                w->spin_inertia = ft->spin_inertia;
                w->max_momentum = ft->max_momentum;
                w->max_torque   = ft->max_torque;
                w->viscous      = ft->viscous;
                w->coulomb      = ft->coulomb;
                w->dead_rate    = ft->dead_rate;
            } else if (ft->kind == KFLC_FEAT_TORQUER) {
                if (m->n_torquers >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d magnetorquers in this program",
                        RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlTorquer *q = &m->torquers[m->n_torquers++];
                q->veh = veh; q->body = i;
                snprintf(q->name, sizeof q->name, "%s", ft->name);
                for (int k = 0; k < 3; k++) q->axis[k] = ft->axis[k];
                q->max_dipole = ft->max_dipole;
            } else if (ft->kind == KFLC_FEAT_THRUSTER) {
                if (m->n_thrusters >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d thrusters in this program",
                        RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlThruster *t = &m->thrusters[m->n_thrusters++];
                t->veh = veh; t->body = i;
                snprintf(t->name, sizeof t->name, "%s", ft->name);
                for (int k = 0; k < 3; k++) {
                    t->at[k]  = ft->at[k];
                    t->dir[k] = ft->dir[k];
                }
                t->max_thrust = ft->thrust;
            } else if (ft->kind == KFLC_FEAT_PORT && ft->collider >= 0) {
                /* Only a port that named a capture envelope reaches
                 * here: the reader gives one a mating plane collider
                 * and leaves the index at -1 otherwise, so a port
                 * declared without an envelope is geometry the
                 * program can describe and nothing this emitter has a
                 * test to apply to. */
                const KflcCaptureEnvelope *env =
                    kflc_capture_envelope(ft->capture);
                if (!env) {
                    kflc_diag_errorf(diag, ft->line,
                        "port `%s` names capture envelope `%s`, which is "
                        "not defined", ft->name, ft->capture);
                    kflc_arena_release(ar);
                    return 1;
                }
                if (m->n_ports >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d docking ports in this program",
                        RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlPort *pt = &m->ports[m->n_ports++];
                memset(pt, 0, sizeof *pt);
                pt->veh = veh; pt->body = i;
                for (int k = 0; k < 3; k++) pt->com[k] = a->com[k];
                /* The pass reports a shape index within the body's
                 * own slice, which is the assembly's collider order,
                 * so the reader's index is what a contact is
                 * compared against. */
                pt->coll = ft->collider;
                snprintf(pt->name, sizeof pt->name, "%s", ft->name);
                const KflcAsmCollider *plate = &a->colliders[ft->collider];
                for (int k = 0; k < 3; k++) {
                    pt->at[k] = ft->at[k];
                    /* The reader built the plate on the port's own
                     * frame, so the basis is read back from it rather
                     * than orthonormalised a second time here, which
                     * would be a second place for it to be wrong. */
                    for (int w = 0; w < 3; w++) {
                        pt->basis[k][w] = plate->rot[w][k];
                    }
                }
                /* The one conversion from the envelope's printed
                 * units to the units the artifact carries. */
                pt->axial_rate_min = env->axial_rate_min;
                pt->axial_rate_max = env->axial_rate_max;
                pt->lateral_rate   = env->lateral_rate;
                pt->pitchyaw_rate  = kflc_capture_deg_to_rad(env->pitchyaw_rate_deg);
                pt->roll_rate      = kflc_capture_deg_to_rad(env->roll_rate_deg);
                pt->lateral        = env->lateral;
                pt->pitchyaw       = kflc_capture_deg_to_rad(env->pitchyaw_deg);
                pt->roll           = kflc_capture_deg_to_rad(env->roll_deg);
                pt->diameter       = kflc_capture_mm_to_m(env->mating_diameter_mm);
            }
        }

        /* A magnetorquer works against the local magnetic field, and
         * the field model is defined at a geodetic position on a
         * rotating body. Reaching one needs the parent's rotation
         * model, and the only key a program can supply for it is the
         * parent's NAIF id: the model table is keyed by that id and
         * by a name of its own form, which a grammar identifier can
         * never be. A declaration that cannot reach a field is
         * refused here rather than run: the alternative is a
         * magnetorquer that compiles, accepts commands, and produces
         * no torque, which is the kind of wrong answer that costs a
         * training run rather than a compile. */
        {
            int has_torquer = 0;
            for (int q = 0; q < m->n_torquers; q++) {
                if (m->torquers[q].body == i) has_torquer = 1;
            }
            if (has_torquer) {
                const KflcNode *b = m->bodies[i].body;
                const KflcAttr *pa = rl_body_attr_(b, "parent");
                const char *pn = (pa && pa->value.kind == KFLV_IDENT)
                               ? pa->value.u.s : NULL;
                int pi = -1;
                for (int j = 0; pn && j < m->n_bodies; j++) {
                    const char *nm = m->bodies[j].body->name;
                    if (nm && strcmp(nm, pn) == 0) pi = j;
                }
                if (!pn) {
                    kflc_diag_errorf(diag, b->line,
                        "`%s` carries a magnetorquer but declares no "
                        "`parent=`, so there is no rotating body whose "
                        "field it could work against", b->name);
                    kflc_arena_release(ar);
                    return 1;
                }
                if (pi < 0) {
                    kflc_diag_errorf(diag, b->line,
                        "`%s` carries a magnetorquer and names `%s` as its "
                        "parent, but no astro_body of that name is declared "
                        "in this world", b->name, pn);
                    kflc_arena_release(ar);
                    return 1;
                }
                if (!rl_body_attr_(m->bodies[pi].body, "ephem_naif_id")) {
                    kflc_diag_errorf(diag, m->bodies[pi].body->line,
                        "`%s` carries a magnetorquer, and its parent `%s` "
                        "declares no `ephem_naif_id=`; that is what names "
                        "the parent's rotation model, without which the "
                        "field has no frame and the magnetorquer would "
                        "produce no torque (Earth is 399)",
                        b->name, pn);
                    kflc_arena_release(ar);
                    return 1;
                }
            }
        }
        veh++;
        kflc_arena_release(ar);
    }
    m->n_veh = veh;
    return 0;
}

static int rl_collect_(RlModel *m, const KflcNode *form,
                       KflcArena *arena, KflcDiag *diag)
{
    memset(m, 0, sizeof *m);

    for (const KflcNode *c = form->children; c; c = c->next) {
        switch (c->kind) {
        case KFLN_FN_WORLD:
            if (m->world) {
                kflc_diag_errorf(diag, c->line,
                    "fn world %s: a reinforcement learning program "
                    "compiles to one environment and admits exactly one "
                    "`fn world`", c->name ? c->name : "?");
                return 1;
            }
            m->world = c;
            break;
        case KFLN_FN_DATA:
        case KFLN_WIDGET:
            kflc_diag_errorf(diag, c->line,
                "plot surfaces are not supported in a reinforcement "
                "learning program; the compiled environment has no "
                "figure output");
            return 1;
        case KFLN_ARENA:
            kflc_diag_errorf(diag, c->line,
                "arena declarations are not supported in a "
                "reinforcement learning program");
            return 1;
        default:
            break;
        }
    }
    if (!m->world) {
        kflc_diag_errorf(diag, form->line,
            "reinforcement learning emission requires a `fn world` block");
        return 1;
    }

    int err = 0;
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        switch (s->kind) {
        case KFLN_STMT_ASTRO_BODY:
            if (m->n_bodies == RL_MAX_BODIES) {
                kflc_diag_errorf(diag, s->line,
                    "too many astro_body declarations (limit %d)",
                    RL_MAX_BODIES);
                return 1;
            }
            if (!s->name) {
                kflc_diag_errorf(diag, s->line,
                    "astro_body requires a name in a reinforcement "
                    "learning world");
                return 1;
            }
            m->bodies[m->n_bodies].body  = s;
            m->bodies[m->n_bodies].index = m->n_bodies;
            m->n_bodies++;
            break;
        case KFLN_STMT_EPISODE:
            m->episode = s;
            for (const KflcNode *r = s->children; r; r = r->next) {
                if (r->kind != KFLN_STMT_EPISODE_RESET) continue;
                if (m->n_resets == RL_MAX_RESETS) {
                    kflc_diag_errorf(diag, r->line,
                        "too many reset lines (limit %d)", RL_MAX_RESETS);
                    return 1;
                }
                m->resets[m->n_resets++] = r;
            }
            break;
        case KFLN_STMT_ACTION:
            if (m->n_actions == RL_MAX_ACTIONS) {
                kflc_diag_errorf(diag, s->line,
                    "too many action declarations (limit %d)",
                    RL_MAX_ACTIONS);
                return 1;
            }
            m->actions[m->n_actions++] = s;
            break;
        case KFLN_STMT_ON_STEP:
            m->on_step = s;
            break;
        case KFLN_STMT_OBJECTIVE:
            m->objective = s;
            break;
        case KFLN_STMT_SENSOR:
            if (rl_collect_sensor_(m, s, diag)) err = 1;
            break;
        case KFLN_STMT_OBSERVE:
            if (rl_observe_as_(s)) {
                if (m->n_observes == RL_MAX_OBSERVES) {
                    kflc_diag_errorf(diag, s->line,
                        "too many observation channels (limit %d)",
                        RL_MAX_OBSERVES);
                    return 1;
                }
                m->observes[m->n_observes++] = s;
            }
            break;
        default:
            break;
        }
    }

    /* Constructs and body declarations must be top level: body
     * indices, draw channels, and the episode machinery are part of
     * the compiled program's identity and cannot be conditional. */
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        if (s->kind != KFLN_STMT_IF && s->kind != KFLN_STMT_WHILE &&
            s->kind != KFLN_STMT_FOR_EACH) {
            continue;
        }
        const KflcNode *stack[2] = { s->children, s->else_children };
        for (int b = 0; b < 2; b++) {
            for (const KflcNode *c = stack[b]; c; c = c->next) {
                if (c->kind == KFLN_STMT_ASTRO_BODY) {
                    kflc_diag_errorf(diag, c->line,
                        "astro_body declarations in a reinforcement "
                        "learning world must be top level: the body set "
                        "is part of the compiled program's identity");
                    err = 1;
                }
            }
        }
    }

    if (!m->episode) {
        /* The checker enforces this; emission cannot proceed without
         * a control period, so it is re-checked rather than assumed. */
        kflc_diag_errorf(diag, m->world->line,
            "fn world %s: reinforcement learning emission requires an "
            "`episode` block with `control_dt`",
            m->world->name ? m->world->name : "?");
        return 1;
    }
    m->control_dt      = rl_attr_(m->episode, "control_dt");
    m->horizon         = rl_attr_(m->episode, "horizon");
    m->substeps        = rl_attr_(m->episode, "substeps");
    m->contact_kind    = rl_attr_(m->episode, "contact");
    m->restitution     = rl_attr_(m->episode, "restitution");
    m->friction        = rl_attr_(m->episode, "friction");
    m->terminated_when = rl_attr_(m->episode, "terminated_when");
    if (m->objective) {
        m->reward   = rl_attr_(m->objective, "reward");
        m->terminal = rl_attr_(m->objective, "terminal");
    }
    if (!m->control_dt || !m->control_dt->expr) {
        kflc_diag_errorf(diag, m->episode->line,
            "episode: missing required `control_dt <expr>`");
        return 1;
    }

    /* Domain-randomisation parameters: distribution-valued astro_body
     * attributes, channels in source order within class 0x0002. */
    for (int i = 0; i < m->n_bodies; i++) {
        for (const KflcAttr *a = m->bodies[i].body->attrs; a; a = a->next) {
            if (!a->name || strcmp(a->name, "parent") == 0) continue;
            KflcExpr *d = rl_attr_dist_(a, form, arena, diag, &err);
            if (!d) continue;
            if (m->n_dr == RL_MAX_DR) {
                kflc_diag_errorf(diag, a->line,
                    "too many domain-randomisation parameters (limit %d)",
                    RL_MAX_DR);
                return 1;
            }
            m->dr[m->n_dr].body    = i;
            m->dr[m->n_dr].attr    = a;
            m->dr[m->n_dr].dist    = d;
            m->dr[m->n_dr].channel = m->n_dr;
            m->n_dr++;
        }
    }

    /* World scalars readable by the objective and termination
     * expressions: top-level prefix let/const bindings of scalar type,
     * captured per environment at create. Names the action channels or
     * observation components already claim are shadowed there and not
     * captured. A duplicate top-level name is left to the emitted
     * C++'s own redeclaration error, as the batch emitter leaves it. */
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        if (s->kind != KFLN_STMT_LET && s->kind != KFLN_STMT_CONST) {
            continue;
        }
        if (!s->name || !rl_is_scalar_type_(s->type)) continue;
        if (rl_scope_name_taken_(m, s->name)) continue;
        int dup = 0;
        for (int j = 0; j < m->n_wscal; j++) {
            const char *wn = m->wscal[j]->name;
            if (wn && strcmp(wn, s->name) == 0) dup = 1;
        }
        if (dup) continue;
        if (m->n_wscal == RL_MAX_WSCAL) {
            kflc_diag_errorf(diag, s->line,
                "too many world scalar bindings (limit %d)",
                RL_MAX_WSCAL);
            return 1;
        }
        m->wscal[m->n_wscal++] = s;
    }

    /* Reset lines must name a declared top-level body. */
    for (int i = 0; i < m->n_resets; i++) {
        const KflcNode *r = m->resets[i];
        int found = -1;
        for (int j = 0; j < m->n_bodies; j++) {
            const char *bn = m->bodies[j].body->name;
            if (bn && r->name && strcmp(bn, r->name) == 0) found = j;
        }
        if (found < 0) {
            kflc_diag_errorf(diag, r->line,
                "episode reset: unknown body `%s` (reset lines target "
                "bodies declared with astro_body in this world)",
                r->name ? r->name : "?");
            err = 1;
            continue;
        }
        const char *rk = (r->position.kind == KFLV_IDENT)
                         ? r->position.u.s : NULL;
        if (rk && kflc_body_state_is_attitude(rk) &&
            !rl_body_has_assembly_(m, found)) {
            kflc_diag_errorf(diag, r->line,
                "episode reset: `%s.%s` sets attitude state, but `%s` "
                "declares no `assembly=`, so it has no inertia tensor and "
                "its attitude is never advanced; bind an assembly or drop "
                "the attitude keys", r->name, rk, r->name);
            err = 1;
        }
    }

    /* Resolve each observe's `through <sensor>` to a declared sensor.
     * A name that resolves to nothing is refused where it is written:
     * a program that misspells a sensor would otherwise compile and
     * silently publish uncorrupted values. */
    for (int i = 0; i < m->n_observes; i++) {
        m->obs_sensor[i] = -1;
        const char *want = rl_observe_through_(m->observes[i]);
        if (!want) continue;
        for (int k = 0; k < m->n_sensors; k++) {
            if (m->sensors[k].name && strcmp(m->sensors[k].name, want) == 0) {
                m->obs_sensor[i] = k;
            }
        }
        if (m->obs_sensor[i] < 0) {
            kflc_diag_errorf(diag, m->observes[i]->line,
                "observe ... through `%s`: no sensor of that name is "
                "declared in this world", want);
            err = 1;
        }
    }
    /* `with truth` without a sensor would publish a channel beside an
     * identical one. It is refused rather than allowed to double an
     * observation vector for nothing. */
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_observe_has_truth_(m->observes[i]) && m->obs_sensor[i] < 0) {
            kflc_diag_errorf(diag, m->observes[i]->line,
                "observe ... with truth: this observe declares no "
                "`through <sensor>`, so its measured and true values "
                "would be the same numbers");
            err = 1;
        }
    }

    if (!err && rl_collect_actuators_(m, diag)) err = 1;

    return err;
}

static int rl_body_index_of_(const RlModel *m, const char *name)
{
    if (!name) return -1;
    for (int i = 0; i < m->n_bodies; i++) {
        const char *bn = m->bodies[i].body->name;
        if (bn && strcmp(bn, name) == 0) return i;
    }
    return -1;
}

/* ---- Emit-side helpers ---------------------------------------------- */

static void rl_emit_indent_(FILE *out, int n)
{
    for (int i = 0; i < n; i++) fputc(' ', out);
}

static void rl_emit_string_literal_(FILE *out, const char *s)
{
    fputc('"', out);
    for (const unsigned char *p = (const unsigned char *)s; p && *p; p++) {
        unsigned char c = *p;
        switch (c) {
        case '"':  fputs("\\\"", out); break;
        case '\\': fputs("\\\\", out); break;
        case '\n': fputs("\\n",  out); break;
        case '\t': fputs("\\t",  out); break;
        case '\r': fputs("\\r",  out); break;
        default:
            if (c < 0x20) fprintf(out, "\\x%02x", c);
            else          fputc(c, out);
            break;
        }
    }
    fputc('"', out);
}

/* Form-argument bindings for expression contexts, mirroring the batch
 * emitter's type heuristic so `kfl_arg_<name>` reads type-consistently
 * in both emitters. */
static void rl_collect_form_args_(const KflcNode *form, KflcArena *arena,
                                  KflcExprBinding **live,
                                  int *live_n, int *live_cap)
{
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_ARG || !c->name) continue;
        if (*live_n == *live_cap) {
            int nc = (*live_cap ? *live_cap * 2 : 8);
            KflcExprBinding *nl = (KflcExprBinding *)kflc_arena_alloc(
                arena, sizeof(KflcExprBinding) * (size_t)nc);
            if (*live_n > 0) {
                memcpy(nl, *live,
                       sizeof(KflcExprBinding) * (size_t)(*live_n));
            }
            *live = nl;
            *live_cap = nc;
        }
        KflcExprBinding *b = &(*live)[*live_n];
        memset(b, 0, sizeof *b);
        b->name = c->name;
        if (c->position.kind == KFLV_STR) {
            b->type = KFLT_STRING;
        } else if (c->position.kind == KFLV_FLOAT) {
            b->type = KFLT_DOUBLE;
        } else if (c->position.kind == KFLV_INT &&
                   (c->position.u.i < 0 || c->position.u.i > 1)) {
            b->type = KFLT_INT;
        } else {
            b->type = KFLT_BOOL;
        }
        b->is_form_arg = 1;
        b->borrow_source_idx = -1;
        (*live_n)++;
    }
}

static void rl_push_binding_(KflcArena *arena, KflcExprBinding **live,
                             int *live_n, int *live_cap,
                             const char *name, KflcType type)
{
    if (*live_n == *live_cap) {
        int nc = (*live_cap ? *live_cap * 2 : 8);
        KflcExprBinding *nl = (KflcExprBinding *)kflc_arena_alloc(
            arena, sizeof(KflcExprBinding) * (size_t)nc);
        if (*live_n > 0) {
            memcpy(nl, *live, sizeof(KflcExprBinding) * (size_t)(*live_n));
        }
        *live = nl;
        *live_cap = nc;
    }
    KflcExprBinding *b = &(*live)[*live_n];
    memset(b, 0, sizeof *b);
    b->name = name;
    b->type = type;
    b->borrow_source_idx = -1;
    (*live_n)++;
}

/* Scalar let/const bindings anywhere in a statement subtree. Used to
 * pre-populate binding tables so forward references resolve, exactly
 * as the batch emitter does. */
static void rl_collect_lets_(const KflcNode *n, KflcArena *arena,
                             KflcExprBinding **live, int *live_n,
                             int *live_cap)
{
    if (!n) return;
    if ((n->kind == KFLN_STMT_LET || n->kind == KFLN_STMT_CONST) &&
        n->name) {
        rl_push_binding_(arena, live, live_n, live_cap, n->name, n->type);
        (*live)[*live_n - 1].type_subtype = n->type_subtype;
        (*live)[*live_n - 1].lifetime_qualifier = n->lifetime_qualifier;
    }
    for (const KflcNode *c = n->children; c; c = c->next) {
        rl_collect_lets_(c, arena, live, live_n, live_cap);
    }
    for (const KflcNode *c = n->else_children; c; c = c->next) {
        rl_collect_lets_(c, arena, live, live_n, live_cap);
    }
}

/* Rewrite `episode.steps` identifiers to a C-compatible name. The
 * expression AST is not reused after emission, so an in-place rename
 * is safe; every other identifier stays untouched. */
static void rl_rewrite_steps_(KflcExpr *e, KflcArena *arena)
{
    if (!e) return;
    switch (e->kind) {
    case KFLE_IDENT:
        if (e->u.ident && strcmp(e->u.ident, "episode.steps") == 0) {
            e->u.ident = kflc_arena_strdup(arena, "_kfl_episode_steps");
        }
        return;
    case KFLE_UNARY:
        rl_rewrite_steps_(e->u.un.operand, arena);
        return;
    case KFLE_BINARY:
        rl_rewrite_steps_(e->u.bin.lhs, arena);
        rl_rewrite_steps_(e->u.bin.rhs, arena);
        return;
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            rl_rewrite_steps_(e->u.call.args[i], arena);
        }
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            rl_rewrite_steps_(e->u.vec.elems[i], arena);
        }
        return;
    case KFLE_INDEX:
        rl_rewrite_steps_(e->u.index.base, arena);
        rl_rewrite_steps_(e->u.index.idx, arena);
        return;
    default:
        return;
    }
}

/* ---- RL expression scope -------------------------------------------- *
 *
 * The `terminated when`, `reward`, and `terminal` expressions read
 * action channels, observation channel components, `episode.steps`,
 * world scalar bindings, and form arguments. Emission declares one
 * const double local per action, channel component, and captured
 * world scalar so the expression emitter resolves the KFL names as
 * ordinary scalar bindings. */

static void rl_emit_scope_prelude_(FILE *out, const RlModel *m, int indent)
{
    for (int i = 0; i < m->n_actions; i++) {
        rl_emit_indent_(out, indent);
        fprintf(out,
            "const double %s = _kfl_act_v ? _kfl_act_v[%d] : 0.0; "
            "(void)%s;\n",
            m->actions[i]->name, i, m->actions[i]->name);
    }
    for (int i = 0; i < m->n_observes; i++) {
        const char *base = rl_observe_as_(m->observes[i]);
        for (int c = 0; c < rl_observe_width_(m->observes[i]); c++) {
            char cb[RL_COMP_MAX];
            const char *cmp = rl_observe_comp_(m->observes[i], c, cb,
                                               sizeof cb);
            rl_emit_indent_(out, indent);
            fprintf(out,
                "const double %s%s = _kfl_obs_v[%d]; (void)%s%s;\n",
                base, cmp, rl_obs_offset_(m->observes, i) + c, base, cmp);
        }
    }
    for (int i = 0; i < m->n_wscal; i++) {
        rl_emit_indent_(out, indent);
        fprintf(out,
            "const double %s = _kfl_world_v[%d]; (void)%s;\n",
            m->wscal[i]->name, i, m->wscal[i]->name);
    }
    rl_emit_indent_(out, indent);
    fputs("const double _kfl_episode_steps = (double)_kfl_nsteps; "
          "(void)_kfl_episode_steps;\n", out);
}

/* Bindings matching rl_emit_scope_prelude_ plus form arguments. */
static void rl_scope_bindings_(const RlModel *m, const KflcNode *form,
                               KflcArena *arena, KflcExprBinding **live,
                               int *live_n, int *live_cap)
{
    rl_collect_form_args_(form, arena, live, live_n, live_cap);
    for (int i = 0; i < m->n_actions; i++) {
        rl_push_binding_(arena, live, live_n, live_cap,
                         m->actions[i]->name, KFLT_DOUBLE);
    }
    for (int i = 0; i < m->n_observes; i++) {
        const char *base = rl_observe_as_(m->observes[i]);
        if (!base) continue;
        for (int c = 0; c < rl_observe_width_(m->observes[i]); c++) {
            char cb[RL_COMP_MAX];
            const char *cmp = rl_observe_comp_(m->observes[i], c, cb,
                                               sizeof cb);
            size_t bl = strlen(base), sl = strlen(cmp);
            char *nm = (char *)kflc_arena_alloc(arena, bl + sl + 1);
            memcpy(nm, base, bl);
            memcpy(nm + bl, cmp, sl + 1);
            rl_push_binding_(arena, live, live_n, live_cap, nm,
                             KFLT_DOUBLE);
        }
    }
    for (int i = 0; i < m->n_wscal; i++) {
        rl_push_binding_(arena, live, live_n, live_cap,
                         m->wscal[i]->name, KFLT_DOUBLE);
    }
    rl_push_binding_(arena, live, live_n, live_cap,
                     "_kfl_episode_steps", KFLT_DOUBLE);
}

/* ---- Draw and state-write emission ----------------------------------- */

/* Emit the draw expression for a distribution call at the given
 * stream class and channel, with `envi` and `ep` the in-scope
 * environment and episode variables. Arguments are emitted in the
 * given expression context (form arguments only in practice). */
static int rl_emit_draw_(FILE *out, const KflcExpr *dist,
                         unsigned cls, int channel,
                         const KflcExprCtx *ctx, KflcDiag *diag)
{
    int is_uniform = (strcmp(dist->u.call.name, "uniform") == 0);
    fprintf(out, "kflrl_%s_(_kfl_key, 0x%04xu, %du, _kfl_envi, _kfl_ep, ",
            is_uniform ? "uniform" : "normal", cls, channel);
    if (kflc_emit_expr(out, dist->u.call.args[0], ctx, diag)) return 1;
    fputs(", ", out);
    if (kflc_emit_expr(out, dist->u.call.args[1], ctx, diag)) return 1;
    fputs(")", out);
    return 0;
}

/* Emit a write of `value_text` into body field `key` on the body
 * lvalue `lv` (either `_kfl_b.` for a struct or `_kfl_bp->` for a
 * pointer). The six scalar state keys map onto the compound
 * position and velocity fields; position writes re-normalise the
 * sector fold so the component behaves as metres from the origin. */
static void rl_emit_body_write_(FILE *out, int indent, const char *lv,
                                const char *key, const char *value_text)
{
    if (rl_is_state_key_(key)) {
        kflc_emit_body_state_write(out, indent, lv, key, value_text);
        return;
    }
    rl_emit_indent_(out, indent);
    fprintf(out, "%s%s = (%s);\n", lv, key, value_text);
}

/* As rl_emit_body_write_, with the value already emitted into a
 * named double variable. */
static void rl_emit_body_write_var_(FILE *out, int indent, const char *lv,
                                    const char *key, const char *var)
{
    rl_emit_body_write_(out, indent, lv, key, var);
}

/* ---- Generated-code sections ----------------------------------------- */

/* The number of bodies in this program that carry a vehicle
 * assembly. Each gets one vehicle per environment, owned by the
 * handle. */
static int rl_n_vehicles_(const RlModel *m)
{
    int n = 0;
    for (int i = 0; i < m->n_bodies; i++) {
        for (const KflcAttr *a = m->bodies[i].body->attrs; a; a = a->next) {
            if (a->name && strcmp(a->name, "assembly") == 0) { n++; break; }
        }
    }
    return n;
}

/* Whether any declared channel is a relative observe. What it gates is
 * the proximity library's header and its archive: a program that never
 * asks for a relative state must not acquire the dependency. */
static int rl_has_relative_observe_(const RlModel *m)
{
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_observe_form_(m->observes[i]) == RL_OBS_REL) return 1;
    }
    return 0;
}

/* The actuator tables and the per-environment block their commands
 * and wheel momenta live in. The tables are what the assemblies
 * declared and never change; the block is state, one per environment,
 * allocated at create and reset with the episode. */
static void rl_emit_actuators_(FILE *out, const RlModel *m)
{
    fprintf(out,
        "#define KFLRL_N_WHEELS %d\n"
        "#define KFLRL_N_TORQUERS %d\n"
        "#define KFLRL_N_THRUSTERS %d\n"
        "#define KFLRL_N_CMD (KFLRL_N_WHEELS + KFLRL_N_TORQUERS + "
        "KFLRL_N_THRUSTERS)\n\n",
        m->n_wheels, m->n_torquers, m->n_thrusters);

    if (m->n_wheels > 0) {
        fputs("static const K26AstroAttWheel kflrl_wheel_desc_[] = {\n",
              out);
        for (int i = 0; i < m->n_wheels; i++) {
            const RlWheel *w = &m->wheels[i];
            fprintf(out,
                "    { { %.17g, %.17g, %.17g }, %.17g, %.17g, %.17g, "
                "%.17g, %.17g, %.17g, 0.0, 0.0 },\n",
                w->axis[0], w->axis[1], w->axis[2], w->spin_inertia,
                w->max_momentum, w->max_torque, w->viscous, w->coulomb,
                w->dead_rate);
        }
        fputs("};\n", out);
        fputs("static const int kflrl_wheel_veh_[] = {\n", out);
        for (int i = 0; i < m->n_wheels; i++) {
            fprintf(out, "    %d,\n", m->wheels[i].veh);
        }
        fputs("};\n\n", out);
    }
    if (m->n_torquers > 0) {
        fputs("static const K26AstroAttTorquer kflrl_torquer_desc_[] = {\n",
              out);
        for (int i = 0; i < m->n_torquers; i++) {
            const RlTorquer *q = &m->torquers[i];
            fprintf(out, "    { { %.17g, %.17g, %.17g }, %.17g, 0.0 },\n",
                    q->axis[0], q->axis[1], q->axis[2], q->max_dipole);
        }
        fputs("};\n", out);
        fputs("static const int kflrl_torquer_veh_[] = {\n", out);
        for (int i = 0; i < m->n_torquers; i++) {
            fprintf(out, "    %d,\n", m->torquers[i].veh);
        }
        fputs("};\n\n", out);
    }
    if (m->n_thrusters > 0) {
        fputs("static const K26AstroAttThruster kflrl_thruster_desc_[] "
              "= {\n", out);
        for (int i = 0; i < m->n_thrusters; i++) {
            const RlThruster *t = &m->thrusters[i];
            fprintf(out,
                "    { { %.17g, %.17g, %.17g }, { %.17g, %.17g, %.17g }, "
                "%.17g, 0.0 },\n",
                t->at[0], t->at[1], t->at[2], t->dir[0], t->dir[1],
                t->dir[2], t->max_thrust);
        }
        fputs("};\n", out);
        fputs("static const int kflrl_thruster_veh_[] = {\n", out);
        for (int i = 0; i < m->n_thrusters; i++) {
            fprintf(out, "    %d,\n", m->thrusters[i].veh);
        }
        fputs("};\n\n", out);
    }
    if (m->n_veh > 0) {
        fputs("static const double kflrl_veh_com_[][3] = {\n", out);
        for (int i = 0; i < m->n_veh; i++) {
            fprintf(out, "    { %.17g, %.17g, %.17g },\n",
                    m->veh_com[i][0], m->veh_com[i][1], m->veh_com[i][2]);
        }
        fputs("};\n\n", out);
    }

    fputs(
"/* One environment's actuator state. Commands are written by on_step\n"
" * once per control period and held across the sub-advances, which is\n"
" * what a zero-order hold means; wheel momentum is the only state that\n"
" * persists between transitions, and the episode reset clears both. */\n"
"typedef struct {\n"
"    double wheel_h[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];\n"
"    double cmd[KFLRL_N_CMD > 0 ? KFLRL_N_CMD : 1];\n"
"} KflrlAct;\n\n", out);


    /* The accessors a command or reading in on_step lowers to. */
    for (int i = 0; i < m->n_acts; i++) {
        const RlActRef *r = &m->acts[i];
        int base = (r->kind == 0) ? r->index
                 : (r->kind == 1) ? m->n_wheels + r->index
                                  : m->n_wheels + m->n_torquers + r->index;
        if (r->field == 0) {
            fprintf(out,
                "static void kflrl_act_set_%d(KflrlAct *a, double v)\n"
                "{\n    a->cmd[%d] = v;\n}\n\n", i, base);
        } else if (r->field == 1) {
            fprintf(out,
                "static double kflrl_act_get_%d(const KflrlAct *a)\n"
                "{\n    return a->wheel_h[%d];\n}\n\n", i, r->index);
        } else {
            fprintf(out,
                "static double kflrl_act_get_%d(const KflrlAct *a)\n"
                "{\n    return a->wheel_h[%d] / %.17g;\n}\n\n",
                i, r->index, m->wheels[r->index].spin_inertia);
        }
    }

    if (m->n_torquers > 0) {
        fputs(
"/* The magnetic field a magnetorquer works against, in the body\n"
" * frame. The chain is where this model would go wrong quietly, so\n"
" * it is written out step by step:\n"
" *\n"
" *   the vehicle's position relative to the body it orbits, in the\n"
" *   world frame; rotated into that body's own rotating frame by its\n"
" *   rotation model, which is what makes a longitude mean anything;\n"
" *   converted to a geodetic latitude, longitude and height on the\n"
" *   ellipsoid, which is where the field model is defined; evaluated\n"
" *   there, giving north, east and down in tesla; turned into east,\n"
" *   north and up, rotated back out to the rotating frame, then to\n"
" *   the world frame, then into the body frame by the conjugate of\n"
" *   the vehicle's own orientation.\n"
" *\n"
" * The parent's rotation model is found by its `ephem_naif_id=`,\n"
" * which the compiler requires on the parent of any body carrying a\n"
" * magnetorquer: the body library's table is keyed by that id and by\n"
" * a model name of its own form, and a name declared in a program is\n"
" * a grammar identifier, so the id is the only key a program can\n"
" * supply.\n"
" *\n"
" * The zero returns below are therefore defensive and not a\n"
" * behaviour a program can reach: the compiler refuses a\n"
" * magnetorquer whose parent carries no id, so the lookup here\n"
" * cannot fail for that reason. A zero would be the honest answer\n"
" * rather than a field taken in the wrong frame, but the refusal is\n"
" * the answer the author gets. */\n"
"static K26V3 kflrl_field_body_(K26AstroWorld *w, K26AstroVehicle *v,\n"
"                               int veh)\n"
"{\n"
"    K26V3 zero = { 0.0, 0.0, 0.0 };\n"
"    if (!w || !v) return zero;\n"
"    const K26AstroBody *vb = k26astro_world_body_at(\n"
"        w, kflrl_body_idx_[kflrl_vehicle_body_[veh]]);\n"
"    if (!vb || vb->parent_body_idx < 0) return zero;\n"
"    const K26AstroBody *pb = k26astro_world_body_at(\n"
"        w, vb->parent_body_idx);\n"
"    if (!pb) return zero;\n"
"    if (pb->ephem_naif_id == 0) return zero;\n"
"    const K26AstroIAURotation *rot =\n"
"        k26astro_rotation_by_naif(pb->ephem_naif_id);\n"
"    if (!rot) return zero;\n"
"    K26AstroGravState *g = k26astro_world_grav(w);\n"
"    if (!g) return zero;\n"
"\n"
"    K26V3 r_world = k26astro_pos_sub(&vb->pos, &pb->pos);\n"
"    K26Quat fixed_from_world = k26astro_rotation_quaternion(rot, &g->t);\n"
"    K26V3 r_fixed = k26m3d_quat_rotate_v3(fixed_from_world, r_world);\n"
"\n"
"    double lat = 0.0, lon = 0.0, alt = 0.0;\n"
"    if (k26astro_att_geodetic(r_fixed, &lat, &lon, &alt) !=\n"
"        K26ASTRO_ATT_OK) return zero;\n"
"\n"
"    /* The field model's epoch argument is years past J2000. */\n"
"    double yrs = ((double)g->t.days_since_J2000 +\n"
"                  g->t.seconds_of_day / 86400.0) / 365.25;\n"
"    K26V3 ned = k26astro_geomag_field_v3(lat, lon, alt, yrs);\n"
"    /* North, east, down as the model reports it, into east, north\n"
"     * and up as the rotation below expects. The sign on the third\n"
"     * component is the whole of the conversion. */\n"
"    K26V3 enu = { ned.y, ned.x, -ned.z };\n"
"    K26V3 b_fixed = k26astro_att_enu_to_ecef(enu, lat, lon);\n"
"    K26V3 b_world = k26m3d_quat_rotate_v3(\n"
"        k26m3d_quat_conj(fixed_from_world), b_fixed);\n"
"    return k26m3d_quat_rotate_v3(k26m3d_quat_conj(vb->attitude),\n"
"                                 b_world);\n"
"}\n\n", out);
    }
    if (m->n_veh > 0 && m->n_torquers == 0) {
        fputs(
"/* No magnetorquer is declared, so no field is needed and none is\n"
" * computed. This is what keeps the Fortran-backed field model out of\n"
" * a program that does not ask for it. */\n"
"static K26V3 kflrl_field_body_(K26AstroWorld *w, K26AstroVehicle *v,\n"
"                               int veh)\n"
"{\n"
"    (void)w; (void)v; (void)veh;\n"
"    return k26m3d_v3(0.0, 0.0, 0.0);\n"
"}\n\n", out);
    }
    if (m->n_veh > 0) {
        fputs(
"/* Build one vehicle's actuator view over the environment's state.\n"
" * The descriptors are constants and the mutable parts are copied in\n"
" * and out around the call, so nothing here allocates and the state\n"
" * stays where the handle can reset it. */\n"
"static void kflrl_act_view_(KflrlAct *a, int veh,\n"
"                            K26AstroAttWheel *wh, K26AstroAttTorquer *tq,\n"
"                            K26AstroAttThruster *th,\n"
"                            K26AstroAttActuators *out, int *w_map)\n"
"{\n"
"    int nw = 0, nq = 0, nt = 0;\n"
"#if KFLRL_N_WHEELS > 0\n"
"    for (int i = 0; i < KFLRL_N_WHEELS; i++) {\n"
"        if (kflrl_wheel_veh_[i] != veh) continue;\n"
"        wh[nw] = kflrl_wheel_desc_[i];\n"
"        wh[nw].momentum = a->wheel_h[i];\n"
"        wh[nw].command  = a->cmd[i];\n"
"        w_map[nw] = i;\n"
"        nw++;\n"
"    }\n"
"#endif\n"
"#if KFLRL_N_TORQUERS > 0\n"
"    for (int i = 0; i < KFLRL_N_TORQUERS; i++) {\n"
"        if (kflrl_torquer_veh_[i] != veh) continue;\n"
"        tq[nq] = kflrl_torquer_desc_[i];\n"
"        tq[nq].command = a->cmd[KFLRL_N_WHEELS + i];\n"
"        nq++;\n"
"    }\n"
"#endif\n"
"#if KFLRL_N_THRUSTERS > 0\n"
"    for (int i = 0; i < KFLRL_N_THRUSTERS; i++) {\n"
"        if (kflrl_thruster_veh_[i] != veh) continue;\n"
"        th[nt] = kflrl_thruster_desc_[i];\n"
"        th[nt].command =\n"
"            a->cmd[KFLRL_N_WHEELS + KFLRL_N_TORQUERS + i];\n"
"        nt++;\n"
"    }\n"
"#endif\n"
"    out->wheels = wh; out->n_wheels = nw;\n"
"    out->torquers = tq; out->n_torquers = nq;\n"
"    out->thrusters = th; out->n_thrusters = nt;\n"
"    out->com = k26m3d_v3(kflrl_veh_com_[veh][0], kflrl_veh_com_[veh][1],\n"
"                         kflrl_veh_com_[veh][2]);\n"
"}\n\n"
"/* Copy the wheel momenta back, which is the only part of the view\n"
" * that is state rather than description. */\n"
"static void kflrl_act_store_(KflrlAct *a, const K26AstroAttActuators *v,\n"
"                             const int *w_map)\n"
"{\n"
"    for (int i = 0; i < v->n_wheels; i++) {\n"
"        a->wheel_h[w_map[i]] = v->wheels[i].momentum;\n"
"    }\n"
"}\n\n", out);
    }
}

/* The sensed channels, their model terms, and the draw channels the
 * owning layer allocates to them.
 *
 * One sensed channel is one numeric component of one observe that
 * declared a sensor. Channels are allocated in source order, one per
 * term that draws, and a term drawing at both cadences takes two, so
 * that a per-episode draw at index 0 cannot be the same number as the
 * first step's draw. Adding a sensor to a program therefore allocates
 * channels above every channel already allocated and perturbs no
 * existing draw.
 *
 * The rows are emitted as plain numbers and assembled into the
 * library's typed terms at create, because the artifact is C++11 and
 * cannot initialise a union member by name, and because the bias
 * walk's two coefficients are not known until the control period is
 * read. */
static int rl_n_sensed_(const RlModel *m)
{
    int n = 0;
    for (int i = 0; i < m->n_observes; i++) {
        if (m->obs_sensor[i] < 0) continue;
        n += rl_observe_base_width_(m->observes[i]);
    }
    return n;
}

static int rl_emit_sensors_(FILE *out, const RlModel *m, KflcDiag *diag)
{
    int n_sensed = rl_n_sensed_(m);
    int n_terms = 0, depth = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        n_terms += m->sensors[si].n_terms * rl_observe_base_width_(m->observes[i]);
        if (m->sensors[si].depth > depth) depth = m->sensors[si].depth;
    }
    fprintf(out, "#define KFLRL_N_SENSED %d\n", n_sensed);
    fprintf(out, "#define KFLRL_N_STERMS %d\n", n_terms);
    fprintf(out, "#define KFLRL_SENSE_RING %d\n\n", depth);
    if (n_sensed == 0) return 0;

    fputs("/* kind, per-step channel, per-episode channel, three\n"
          " * parameters. The parameters' meaning per kind is\n"
          " * k26sense.h's. */\n"
          "static const struct {\n"
          "    int      kind;\n"
          "    uint16_t ch, ch_ep;\n"
          "    double   p0, p1, p2;\n"
          "} kflrl_sterm_[KFLRL_N_STERMS] = {\n", out);
    int chan = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        const RlSensor *sn = &m->sensors[si];
        int w = rl_observe_base_width_(m->observes[i]);
        for (int c = 0; c < w; c++) {
            /* The chain as the library will see it, assembled here so
             * that the library's own precondition check is what passes
             * it. Without a consumer that refusal measures nothing:
             * the rule it enforces, that a term drawing at both
             * cadences holds two channels, would otherwise rest on the
             * two lines below and on nothing else. */
            K26SenseTerm probe[RL_MAX_TERMS];
            for (int t = 0; t < sn->n_terms; t++) {
                const RlSenseTerm *tm = &sn->terms[t];
                int ch = tm->draws_step ? chan++ : K26SENSE_NO_CHANNEL;
                int ce = tm->draws_ep   ? chan++ : K26SENSE_NO_CHANNEL;
                memset(&probe[t], 0, sizeof probe[t]);
                probe[t].kind       = (K26SenseKind)tm->kind;
                probe[t].channel    = (uint16_t)ch;
                probe[t].channel_ep = (uint16_t)ce;
                if (tm->kind == K26SENSE_LATENCY) {
                    probe[t].u.latency.depth = (uint32_t)tm->p0;
                }
                fprintf(out,
                    "    { %d, %uu, %uu, %.17g, %.17g, %.17g },\n",
                    tm->kind, (unsigned)ch, (unsigned)ce,
                    tm->p0, tm->p1, tm->p2);
            }
            uint32_t need = 0;
            K26SenseStatus cst = k26sense_chain_check(
                probe, (uint32_t)sn->n_terms, &need);
            if (cst != K26SENSE_OK) {
                kflc_diag_errorf(diag, sn->node->line,
                    "sensor `%s`: the assembled chain is refused by the "
                    "imperfection layer: %s", sn->name,
                    k26sense_status_str(cst));
                return 1;
            }
        }
    }
    fputs("};\n\n", out);

    fputs("static const int kflrl_sensed_first_[KFLRL_N_SENSED] = {\n", out);
    int first = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        int w = rl_observe_base_width_(m->observes[i]);
        for (int c = 0; c < w; c++) {
            fprintf(out, "    %d,\n", first);
            first += m->sensors[si].n_terms;
        }
    }
    fputs("};\n\nstatic const int kflrl_sensed_count_[KFLRL_N_SENSED] = {\n",
          out);
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        int w = rl_observe_base_width_(m->observes[i]);
        for (int c = 0; c < w; c++) {
            fprintf(out, "    %d,\n", m->sensors[si].n_terms);
        }
    }
    /* The observation slot each sensed channel rewrites: the measured
     * half of its observe, in component order. */
    fputs("};\n\nstatic const int kflrl_sensed_slot_[KFLRL_N_SENSED] = {\n",
          out);
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        int off = rl_obs_offset_(m->observes, i);
        int w = rl_observe_base_width_(m->observes[i]);
        for (int c = 0; c < w; c++) fprintf(out, "    %d,\n", off + c);
    }
    fputs("};\n\n", out);
    return 0;
}

static int rl_emit_prologue_(FILE *out, const RlModel *m,
                             const KflcNode *form, KflcDiag *diag)
{
    fprintf(out,
        "/* AUTO-GENERATED by kflc. DO NOT EDIT BY HAND.\n"
        " * Source form: %s\n"
        " *\n"
        " * Reinforcement learning environment core. One translation\n"
        " * unit, two artifacts: compiled with KFLC_RL_BATCH_MAIN it is\n"
        " * the batch executable; compiled as a shared object it exports\n"
        " * exactly the k26rl_ stepping surface. Batch mode drives the\n"
        " * same create/step/reset code through the same functions, so\n"
        " * the two entries are bit-identical by construction. */\n\n",
        form->name ? form->name : "FORM");

    fputs(
        "#include <k26astro_rt/world.h>\n"
        "#include <k26astro_rt/observer.h>\n"
        "#include <k26astro_rt/world_rng.h>\n"
        "#include <k26astro_grav/grav.h>\n"
        "#include <k26astro_grav/ias15.h>\n"
        "#include <k26astro_grav/perturb.h>\n"
        "#include <k26astro_body/body.h>\n"
        "#include <k26astro_core/pos.h>\n"
        "#include <k26astro_vehicle/vehicle.h>\n"
        "#include <k26astro_att/att.h>\n", out);
    if (m->n_colliders > 0) {
        fputs("#include <k26astro_coll/coll.h>\n", out);
    }
    if (rl_has_relative_observe_(m)) {
        /* Only a program that declares a relative observe pulls in the
         * proximity library, for the reason the field model is
         * conditional below: a dependency follows a declaration. */
        fputs("#include <k26astro_prox/prox.h>\n", out);
    }
    if (rl_n_sensed_(m) > 0) {
        /* Only a program that declares a sensor pulls in the
         * imperfection layer, for the reason the field model and the
         * proximity library are conditional: a dependency follows a
         * declaration, and a program that declares no sensor must not
         * acquire a new archive on its link line. */
        fputs("#include <k26sense.h>\n", out);
    }
    if (m->n_torquers > 0) {
        /* Only a program that declares a magnetorquer pulls in the
         * field model, which is Fortran-backed: the dependency is a
         * consequence of a declaration and not of the tier existing. */
        fputs("#include <k26astro_geomag/geomag.h>\n"
              "#include <k26astro_body/rotation_model.h>\n", out);
    }
    fputs(
        "#include \"k26rl_env.h\"\n"
        "#include \"k26rl_episode.h\"\n"
        "#include \"k26rl_tap.h\"\n"
        "#include \"k26rng.h\"\n"
        "\n"
        "#include <stdint.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        "#include <string.h>\n"
        "#include <cmath>\n"
        "\n", out);

    /* Program geometry. */
    fprintf(out,
        "#define KFLRL_N_BODIES %d\n"
        "#define KFLRL_OBS_TOTAL %du\n"
        "#define KFLRL_ACT_TOTAL %du\n"
        "#define KFLRL_N_RESET %d\n"
        "#define KFLRL_N_DR %d\n"
        "#define KFLRL_N_REC %d\n"
        "#define KFLRL_N_WSCAL %d\n"
        "#define KFLRL_HAS_TERMINATED %d\n"
        "#define KFLRL_N_VEHICLES %d\n"
        "#define KFLRL_MAGIC 0x4b524c45u\n"
        "\n",
        m->n_bodies, rl_obs_total_(m->observes, m->n_observes),
        m->n_actions,
        m->n_resets, m->n_dr, m->n_resets + m->n_dr,
        m->n_wscal, m->terminated_when ? 1 : 0,
        rl_n_vehicles_(m));

    /* Domain-randomisation record tags, ascending (class, channel):
     * reset-state entries (class 0x0001) then domain-randomisation
     * entries (class 0x0002). */
    fputs("#if KFLRL_N_REC > 0\n"
          "static const uint32_t kflrl_dr_tags_[KFLRL_N_REC] = {", out);
    for (int i = 0; i < m->n_resets; i++) {
        fprintf(out, "%s0x%08xu", i ? ", " : " ",
                (0x0001u << 16) | (unsigned)i);
    }
    for (int i = 0; i < m->n_dr; i++) {
        fprintf(out, "%s0x%08xu", (i || m->n_resets) ? ", " : " ",
                (0x0002u << 16) | (unsigned)m->dr[i].channel);
    }
    fputs(" };\n#endif\n\n", out);

    /* Observation channel names, source order, one entry per
     * component of each observe-as statement. */
    if (m->n_observes > 0) {
        fputs("static const char *const kflrl_obs_names_"
              "[KFLRL_OBS_TOTAL] = {\n", out);
        for (int i = 0; i < m->n_observes; i++) {
            const char *base = rl_observe_as_(m->observes[i]);
            for (int c = 0; c < rl_observe_width_(m->observes[i]); c++) {
                char cb[RL_COMP_MAX];
                const char *cmp = rl_observe_comp_(m->observes[i], c, cb,
                                                   sizeof cb);
                /* A derived name that would not fit the entry is
                 * refused here rather than truncated in the artifact.
                 * The declarable bound and the entry are sized so this
                 * cannot fire (internal.h states the arithmetic); it
                 * exists so that a suffix added later moves the entry
                 * instead of silently shortening a channel's name. */
                if (strlen(base) + strlen(cmp) >= KFLC_OBS_NAME_MAX) {
                    kflc_diag_errorf(diag, m->observes[i]->line,
                        "observe ... as `%s`: the derived channel name "
                        "`%s%s` is %zu bytes and the spec entry holds "
                        "%d", base, base, cmp,
                        strlen(base) + strlen(cmp), KFLC_OBS_NAME_MAX);
                    return 1;
                }
                fprintf(out, "    \"%s%s\",\n", base, cmp);
            }
        }
        fputs("};\n\n", out);

        /* The mode each channel's observe declared, published per
         * channel so a consumer reads it without reconstructing
         * observe grouping. */
        fputs("static const uint16_t kflrl_obs_modes_"
              "[KFLRL_OBS_TOTAL] = {\n", out);
        for (int i = 0; i < m->n_observes; i++) {
            uint16_t md = rl_observe_mode_(m->observes[i]);
            for (int c = 0; c < rl_observe_width_(m->observes[i]); c++) {
                fprintf(out, "    %u,\n", (unsigned)md);
            }
        }
        fputs("};\n\n", out);

        /* What each channel is, and which channel it is paired with.
         * A program declaring no sensor publishes every channel as
         * measured and unpaired, which is what it is: the value the
         * simulation produced, with nothing between. */
        fputs("static const uint16_t kflrl_obs_source_"
              "[KFLRL_OBS_TOTAL] = {\n", out);
        for (int i = 0; i < m->n_observes; i++) {
            int w = rl_observe_base_width_(m->observes[i]);
            int t = rl_observe_has_truth_(m->observes[i]);
            for (int c = 0; c < w; c++) {
                fprintf(out, "    %u,\n",
                        (unsigned)K26RL_OBS_SOURCE_MEASURED);
            }
            if (!t) continue;
            for (int c = 0; c < w; c++) {
                fprintf(out, "    %u,\n", (unsigned)K26RL_OBS_SOURCE_TRUTH);
            }
        }
        fputs("};\n\nstatic const uint32_t kflrl_obs_pair_"
              "[KFLRL_OBS_TOTAL] = {\n", out);
        for (int i = 0; i < m->n_observes; i++) {
            int off = rl_obs_offset_(m->observes, i);
            int w   = rl_observe_base_width_(m->observes[i]);
            int t   = rl_observe_has_truth_(m->observes[i]);
            for (int c = 0; c < w; c++) {
                if (t) fprintf(out, "    %uu,\n", (unsigned)(off + w + c));
                else   fprintf(out, "    %uu,\n",
                               (unsigned)K26RL_OBS_PAIR_NONE);
            }
            if (!t) continue;
            for (int c = 0; c < w; c++) {
                fprintf(out, "    %uu,\n", (unsigned)(off + c));
            }
        }
        fputs("};\n\n", out);
    }

    /* Body names, declaration order, which is the order the body-state
     * getter returns and the order the body-name tags index. */
    if (m->n_bodies > 0) {
        fputs("static const char *const kflrl_body_names_"
              "[KFLRL_N_BODIES] = {\n", out);
        for (int i = 0; i < m->n_bodies; i++) {
            fprintf(out, "    \"%s\",\n",
                    m->bodies[i].body->name ? m->bodies[i].body->name : "?");
        }
        fputs("};\n\n", out);
    }

    /* Vehicle slot to body index, in the order the vehicles are
     * created, so the step can find each vehicle's own body and the
     * body it is attracted by without searching. */
    {
        int n_v = 0;
        for (int i = 0; i < m->n_bodies; i++) {
            if (!rl_body_has_assembly_(m, i)) continue;
            if (n_v == 0) {
                fputs("static const int kflrl_vehicle_body_[] = {\n", out);
            }
            fprintf(out, "    %d,\n", i);
            n_v++;
        }
        if (n_v > 0) fputs("};\n\n", out);
    }

    /* ---- The collider set ---------------------------------------- *
     *
     * Counts and shapes are fixed at compile time, so the per-step
     * working set is a fixed-size array and the loops are over
     * constants. The primitives are already in their bodies' frames;
     * only the rotation into the world frame is left to the step, and
     * that is one quaternion per body rather than one per primitive.
     */
    fprintf(out, "#define KFLRL_N_COLL %d\n", m->n_colliders);
    /* The contact block is one entry per collidable body, and at
     * least one, so that a program with no vehicles still allocates
     * something a pointer check can be made against. */
    fprintf(out, "#define KFLRL_OBS_NAME_MAX %d\n", KFLC_OBS_NAME_MAX);
    if (rl_emit_sensors_(out, m, diag)) return 1;
    fprintf(out, "#define KFLRL_N_CONTACT %d\n\n",
            m->n_veh > 0 ? m->n_veh : 1);
    fputs(
"/* What a contact observe form publishes: whether this transition\n"
" * contained one, at what fraction of the control period the first\n"
" * one occurred, and how fast the pair was closing along the contact\n"
" * normal when it did. Zero on a transition with no contact, which\n"
" * is what makes the three channels readable without a fourth saying\n"
" * whether the other three mean anything. */\n"
"typedef struct {\n"
"    double hit;\n"
"    double fraction;\n"
"    double speed;\n"
"    /* What a port observe form publishes on a transition whose\n"
"     * contact was between two docking interfaces: whether it\n"
"     * satisfied every condition of the declared envelope, and the\n"
"     * residuals the test was given. They are latched because the\n"
"     * test is a statement about the instant of contact, and by the\n"
"     * end of the transition the resolution has already removed the\n"
"     * relative velocity the test read. */\n"
"    double port_hit;\n"
"    double captured;\n"
"    double axial;\n"
"    double lateral;\n"
"    double pitchyaw;\n"
"    double roll;\n"
"    double v_axial;\n"
"    double v_lateral;\n"
"    double v_pitchyaw;\n"
"    double v_roll;\n"
"} KflrlContact;\n\n", out);
    fputs(
"/* A pair joined by a capture.\n"
" *\n"
" * A capture is not a resolution the environment declares. It takes\n"
" * precedence over the declared one, because a programme told its\n"
" * craft is docked and then shown it flung away has been told two\n"
" * things that cannot both be true; and it does not merely stop the\n"
" * pair, it makes it one body. Mass is summed, inertia is summed\n"
" * about the joint centre of mass by the parallel-axis theorem, and\n"
" * the joint body is carried by whichever of the two has the greater\n"
" * mass, since one of them must carry it and the joint centre of mass\n"
" * lies nearer that one.\n"
" *\n"
" * Thereafter the pair is projected back onto one rigid motion at the\n"
" * end of every sub-advance: the follower is placed from the carrier\n"
" * at the frozen offset, and the two momenta are summed and turned\n"
" * back into one velocity and one rate. Projecting rather than\n"
" * slaving is what keeps a thruster on either craft accelerating the\n"
" * pair, which is what a task that continues past docking needs. The\n"
" * follower's own colliders leave the pass, the pair being one body\n"
" * and the pass a test between bodies.\n"
" *\n"
" * One join per environment. A second capture while joined is left\n"
" * alone rather than nested: the construction below describes a pair\n"
" * and says so. */\n"
"typedef struct {\n"
"    int     active;\n"
"    int     carrier;      /* vehicle slot carrying the joint body */\n"
"    int     follower;\n"
"    K26V3   offset;       /* follower origin, carrier body frame */\n"
"    K26Quat rel;          /* follower attitude = carrier's times this */\n"
"} KflrlJoin;\n\n", out);
    if (m->n_colliders > 0) {
        fputs("static const K26AstroCollShape kflrl_coll_[] = {\n", out);
        for (int i = 0; i < m->n_colliders; i++) {
            const RlCollider *c = &m->colliders[i];
            fprintf(out,
                "    { (K26AstroCollKind)%d, { %.17g, %.17g, %.17g },\n"
                "      { { %.17g, %.17g, %.17g }, { %.17g, %.17g, %.17g },\n"
                "        { %.17g, %.17g, %.17g } },\n"
                "      { %.17g, %.17g, %.17g } },\n",
                c->kind, c->centre[0], c->centre[1], c->centre[2],
                c->axis[0][0], c->axis[0][1], c->axis[0][2],
                c->axis[1][0], c->axis[1][1], c->axis[1][2],
                c->axis[2][0], c->axis[2][1], c->axis[2][2],
                c->half[0], c->half[1], c->half[2]);
        }
        fputs("};\n\n", out);

        /* Where each vehicle's primitives begin and how many it has,
         * so the pass slices one array rather than following
         * pointers. */
        fputs("static const int kflrl_coll_first_[] = {\n", out);
        for (int v = 0; v < m->n_veh; v++) {
            int first = -1;
            for (int i = 0; i < m->n_colliders; i++) {
                if (m->colliders[i].veh == v && first < 0) first = i;
            }
            fprintf(out, "    %d,\n", first < 0 ? 0 : first);
        }
        fputs("};\n\nstatic const int kflrl_coll_count_[] = {\n", out);
        for (int v = 0; v < m->n_veh; v++) {
            int n = 0;
            for (int i = 0; i < m->n_colliders; i++) {
                if (m->colliders[i].veh == v) n++;
            }
            fprintf(out, "    %d,\n", n);
        }
        fputs("};\n\n", out);

        fputs("static const double kflrl_veh_bound_[] = {\n", out);
        for (int v = 0; v < m->n_veh; v++) {
            fprintf(out, "    %.17g,\n", m->veh_bound[v]);
        }
        fputs("};\n\n", out);
    }

    /* ---- Docking ports ---------------------------------------- *
     *
     * One entry per port that named a capture envelope. `shape` is
     * the index, within its own vehicle's collider slice, of the
     * mating plane the assembly reader built from the envelope's
     * published diameter, which is what tells a contact at the
     * docking interface from a contact anywhere else on the craft.
     *
     * The envelope arrives here already in SI. Its printed figures
     * and the single conversion that turns them into these constants
     * live in the compiler's capture table, so a citation stays exact
     * and the arithmetic that changed the units is in one place
     * rather than spread through the artifact.
     */
    fprintf(out, "#define KFLRL_N_PORTS %d\n\n", m->n_ports);
    if (m->n_ports > 0) {
        fputs(
"typedef struct {\n"
"    int                  veh;\n"
"    int                  shape;\n"
"    K26V3                com;\n"
"    K26AstroCollPort     geom;\n"
"    K26AstroCollEnvelope env;\n"
"} KflrlPort;\n\n", out);
        fputs("static const KflrlPort kflrl_ports_[] = {\n", out);
        for (int i = 0; i < m->n_ports; i++) {
            const RlPort *pt = &m->ports[i];
            fprintf(out,
                "    { %d, %d,\n"
                "      { %.17g, %.17g, %.17g },\n"
                "      { { %.17g, %.17g, %.17g },\n"
                "        { { %.17g, %.17g, %.17g }, { %.17g, %.17g, %.17g },\n"
                "          { %.17g, %.17g, %.17g } } },\n"
                "      { %.17g, %.17g, %.17g, %.17g, %.17g,\n"
                "        %.17g, %.17g, %.17g } },\n",
                pt->veh, pt->coll,
                pt->com[0], pt->com[1], pt->com[2],
                pt->at[0], pt->at[1], pt->at[2],
                pt->basis[0][0], pt->basis[0][1], pt->basis[0][2],
                pt->basis[1][0], pt->basis[1][1], pt->basis[1][2],
                pt->basis[2][0], pt->basis[2][1], pt->basis[2][2],
                pt->axial_rate_min, pt->axial_rate_max, pt->lateral_rate,
                pt->pitchyaw_rate, pt->roll_rate,
                pt->lateral, pt->pitchyaw, pt->roll);
        }
        fputs("};\n\n", out);
        fputs(
"/* One body as the port geometry takes it, at rest over a zero\n"
" * interval: the residuals a port observe publishes between contacts\n"
" * are of the state as it stands, so both endpoints of the interval\n"
" * are the same state and the fraction asked for is zero. */\n"
"static void kflrl_port_snap_(const K26AstroBody *b, K26V3 com, K26V3 pos,\n"
"                             K26AstroCollBody *out)\n"
"{\n"
"    memset(out, 0, sizeof *out);\n"
"    out->pos0 = pos;\n"
"    out->pos1 = pos;\n"
"    out->orientation = b->attitude;\n"
"    out->vel0 = b->vel;\n"
"    out->vel1 = b->vel;\n"
"    out->omega = b->omega;\n"
"    out->mass = b->mass;\n"
"    out->com_offset = com;\n"
"}\n\n", out);
    }

    /* Assembly identity, per body that carries one. The assembly is
     * read again here rather than passed across from the body
     * emitter: a digest is a pure function of the asset bytes, so two
     * reads agree by construction, and the alternative is a naming
     * contract between two files for no gain. Bodies without an
     * assembly appear in neither table and publish neither tag. */
    {
        int n_asm = 0;
        for (int i = 0; i < m->n_bodies; i++) {
            const KflcNode *b = m->bodies[i].body;
            KflcArena *ar = kflc_arena_create();
            if (!ar) return 1;
            KflcAssembly *asmb = NULL;
            if (kflc_assembly_for_body(b, diag->path, ar, diag, &asmb)) {
                kflc_arena_release(ar);
                return 1;
            }
            if (!asmb) { kflc_arena_release(ar); continue; }
            if (n_asm == 0) {
                fputs("static const struct {\n"
                      "    uint32_t body;\n"
                      "    const char *name;\n"
                      "    uint8_t digest[32];\n"
                      "} kflrl_assemblies_[] = {\n", out);
            }
            fprintf(out, "    { %d, \"%s\", {", i, asmb->name);
            for (int k = 0; k < KFLC_ASM_DIGEST; k++) {
                fprintf(out, "%s0x%02x", k ? "," : "",
                        (unsigned)asmb->digest[k]);
            }
            fputs("} },\n", out);
            n_asm++;
            kflc_arena_release(ar);
        }
        if (n_asm > 0) fputs("};\n", out);
        fprintf(out, "#define KFLRL_N_ASSEMBLIES %d\n\n", n_asm);
    }

    /* Draw helpers: draw index 0 at (class, channel, environment,
     * episode); a parameter consumes exactly one draw, and drawing is
     * a pure function of its coordinates. normal(mu, sigma) is the
     * affine transform of the standard normal primitive. */
    fputs(
        "static double kflrl_uniform_(K26RngKey _kfl_key, uint16_t cls,\n"
        "                             uint16_t ch, uint32_t envi,\n"
        "                             uint32_t ep, double a, double b)\n"
        "{\n"
        "    K26RngCoords c;\n"
        "    c.stream = cls; c.channel = ch;\n"
        "    c.environment = envi; c.episode = ep; c.draw = 0;\n"
        "    return k26rng_uniform(_kfl_key, c, a, b);\n"
        "}\n"
        "\n"
        "static double kflrl_normal_(K26RngKey _kfl_key, uint16_t cls,\n"
        "                            uint16_t ch, uint32_t envi,\n"
        "                            uint32_t ep, double mu, double sigma)\n"
        "{\n"
        "    K26RngCoords c;\n"
        "    c.stream = cls; c.channel = ch;\n"
        "    c.environment = envi; c.episode = ep; c.draw = 0;\n"
        "    return mu + sigma * k26rng_normal(_kfl_key, c);\n"
        "}\n"
        "\n", out);

    /* Body indices as add_body returns them; worlds are built by the
     * same code in the same order, so the mapping is identical across
     * environments and captured once. */
    /* The seed a world's own stateful generator takes for one
     * environment and episode. It is a counter draw at the
     * imperfection layer's reserved channel, which that layer never
     * allocates to a model; the two numbers are the library's own
     * constants, read at compile time so that a program declaring no
     * sensor carries no dependency on it, and an artifact gate checks
     * the emitted seed against the library's function. */
    fprintf(out,
        "static uint64_t kflrl_world_seed_(K26RngKey _kfl_key,\n"
        "                                  uint32_t e, uint32_t ep)\n"
        "{\n"
        "    K26RngCoords c;\n"
        "    c.stream = 0x%04xu; c.channel = 0x%04xu;\n"
        "    c.environment = e; c.episode = ep; c.draw = 0u;\n"
        "    return k26rng_u64(_kfl_key, c);\n"
        "}\n\n",
        (unsigned)K26SENSE_CLASS_SENSOR,
        (unsigned)K26SENSE_CHANNEL_WORLD_SEED);

    fputs("static int kflrl_body_idx_[KFLRL_N_BODIES > 0 ? "
          "KFLRL_N_BODIES : 1];\n\n", out);
    return 0;
}

/* Form-argument globals, exactly the batch emitter's shape, so RL
 * expressions and the batch CLI read the same storage. */
static void rl_emit_form_args_(FILE *out, const KflcNode *form)
{
    int nargs = 0;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_ARG) nargs++;
    }
    if (nargs == 0) return;

    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_ARG || c->position.kind != KFLV_STR) continue;
        int ro = (c->flags & KFL_NF_READONLY) ? 1 : 0;
        fprintf(out, "static %schar kfl_arg_%s_buf[2048] = ",
                ro ? "const " : "", c->name);
        rl_emit_string_literal_(out, c->position.u.s ? c->position.u.s : "");
        fputs(";\n", out);
    }
    fputs("extern \"C\" {\n", out);
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_ARG) continue;
        int ro = (c->flags & KFL_NF_READONLY) ? 1 : 0;
        if (c->position.kind == KFLV_STR) {
            fprintf(out, "    const char *%skfl_arg_%s = kfl_arg_%s_buf;\n",
                    ro ? "const " : "", c->name, c->name);
        } else if (c->position.kind == KFLV_FLOAT) {
            fprintf(out, "    %sdouble kfl_arg_%s = %.17g;\n",
                    ro ? "const " : "", c->name, c->position.u.f);
        } else if (c->position.kind == KFLV_INT &&
                   (c->position.u.i < 0 || c->position.u.i > 1)) {
            fprintf(out, "    %sint kfl_arg_%s = %ld;\n",
                    ro ? "const " : "", c->name, c->position.u.i);
        } else {
            int dflt = (c->position.kind == KFLV_INT)
                       ? (int)c->position.u.i : 0;
            fprintf(out, "    %sbool kfl_arg_%s = %s;\n",
                    ro ? "const " : "", c->name, dflt ? "true" : "false");
        }
    }
    fputs("}\n\n", out);
}

/* User fns at form scope, mirroring the batch emitter so calls from
 * world prefixes and reinforcement learning expressions resolve. */
static int rl_emit_user_fns_(FILE *out, const KflcNode *form,
                             KflcArena *arena, KflcExprFn *user_fn_arr,
                             int n_user_fns, KflcDiag *diag)
{
    for (const KflcNode *fn = form->children; fn; fn = fn->next) {
        if (fn->kind != KFLN_FN) continue;
        const char *ret_cxx = kflc_type_cxx(fn->type, fn->type_subtype);
        if (!ret_cxx) ret_cxx = "double";
        fputs("extern \"C\" ", out);
        fprintf(out, "%s %s(", ret_cxx, fn->name ? fn->name : "_anon");
        int aidx = 0;
        for (const KflcNode *a = fn->children; a; a = a->next) {
            if (a->kind != KFLN_FN_ARG) continue;
            if (aidx > 0) fputs(", ", out);
            const char *aty = kflc_type_cxx(a->type, a->type_subtype);
            fprintf(out, "%s %s", aty ? aty : "double",
                    a->name ? a->name : "_a");
            aidx++;
        }
        if (aidx == 0) fputs("void", out);
        fputs(") {\n", out);

        KflcExprBinding *live = NULL;
        int live_n = 0, live_cap = 0;
        for (const KflcNode *a = fn->children; a; a = a->next) {
            if (a->kind != KFLN_FN_ARG) continue;
            rl_push_binding_(arena, &live, &live_n, &live_cap,
                             a->name, a->type);
            live[live_n - 1].type_subtype = a->type_subtype;
            live[live_n - 1].lifetime_qualifier = a->lifetime_qualifier;
        }
        for (const KflcNode *s = fn->children; s; s = s->next) {
            if (s->kind == KFLN_FN_ARG) continue;
            rl_collect_lets_(s, arena, &live, &live_n, &live_cap);
        }
        rl_collect_form_args_(form, arena, &live, &live_n, &live_cap);

        KflcExprCtx body_ctx;
        memset(&body_ctx, 0, sizeof body_ctx);
        body_ctx.bindings   = live;
        body_ctx.n_bindings = live_n;
        body_ctx.fns        = user_fn_arr;
        body_ctx.n_fns      = n_user_fns;
        body_ctx.form       = form;

        kfl_emit_stmt_reset_scopes_ex(arena, fn->type, fn->type_subtype);
        int last_was_return = 0;
        for (const KflcNode *s = fn->children; s; s = s->next) {
            if (s->kind == KFLN_FN_ARG) continue;
            last_was_return = (s->kind == KFLN_STMT_RETURN);
            (void)kfl_emit_stmt(out, s, &body_ctx, diag, 4);
        }
        if (!last_was_return) kfl_emit_stmt_drain_root(out, 4);
        fputs("}\n\n", out);
    }
    return diag->errors ? 1 : 0;
}

/* Program parameters evaluated over form arguments at create. */
static int rl_emit_params_(FILE *out, const RlModel *m,
                           const KflcExprCtx *arg_ctx, KflcDiag *diag)
{
    fputs("static double kflrl_control_dt_(void)\n{\n    return (double)(",
          out);
    if (kflc_emit_expr(out, m->control_dt->expr, arg_ctx, diag)) return 1;
    fputs(");\n}\n\n", out);

    /* The subdivision of a control period. Emitted as an accessor
     * beside the horizon, for the horizon's reason: it is a declared
     * expression, and create is where a declared value is read and
     * checked once. */
    fputs("static uint32_t kflrl_substeps_(void)\n{\n    return (uint32_t)(",
          out);
    if (m->substeps && m->substeps->expr) {
        if (kflc_emit_expr(out, m->substeps->expr, arg_ctx, diag)) return 1;
    } else {
        fputs("1", out);
    }
    fputs(");\n}\n\n", out);

    /* The contact resolution and, for bounce, its two coefficients.
     * Emitted as accessors for the same reason as the subdivision:
     * they are declared expressions, and create is where a declared
     * value is read once. A program that declares no `contact` line
     * gets arrest, which is the default the design fixes, so the
     * absent case and the `contact arrest` case emit the same
     * artifact. */
    {
        int bounce = m->contact_kind &&
                     m->contact_kind->value.kind == KFLV_IDENT &&
                     m->contact_kind->value.u.s &&
                     strcmp(m->contact_kind->value.u.s, "bounce") == 0;
        fprintf(out, "#define KFLRL_CONTACT_BOUNCE %d\n\n", bounce ? 1 : 0);
        fputs("static double kflrl_restitution_(void)\n{\n    return (",
              out);
        if (bounce && m->restitution && m->restitution->expr) {
            if (kflc_emit_expr(out, m->restitution->expr, arg_ctx, diag)) {
                return 1;
            }
        } else {
            fputs("0.0", out);
        }
        fputs(");\n}\n\n", out);
        fputs("static double kflrl_friction_(void)\n{\n    return (", out);
        if (bounce && m->friction && m->friction->expr) {
            if (kflc_emit_expr(out, m->friction->expr, arg_ctx, diag)) {
                return 1;
            }
        } else {
            fputs("0.0", out);
        }
        fputs(");\n}\n\n", out);
    }

    fputs("static uint32_t kflrl_horizon_(void)\n{\n    return (uint32_t)(",
          out);
    if (m->horizon && m->horizon->expr) {
        if (kflc_emit_expr(out, m->horizon->expr, arg_ctx, diag)) return 1;
    } else {
        fputs("0", out);
    }
    fputs(");\n}\n\n", out);

    /* Action channel parameters: bounds, kinds, arities, defaults.
     * Discrete channels publish bounds 0 to arity - 1 so a consumer
     * reading only the bounds tag still sees the legal range. */
    fputs("static void kflrl_act_params_(double *lo, double *hi,\n"
          "                              uint32_t *arity, uint16_t *kind)\n"
          "{\n"
          "    (void)lo; (void)hi; (void)arity; (void)kind;\n", out);
    for (int i = 0; i < m->n_actions; i++) {
        const KflcNode *a = m->actions[i];
        int is_box = (a->position.kind == KFLV_IDENT && a->position.u.s &&
                      strcmp(a->position.u.s, "box") == 0);
        if (is_box) {
            fprintf(out, "    kind[%d] = K26RL_ACT_KIND_BOX; "
                         "arity[%d] = 0;\n", i, i);
            fprintf(out, "    lo[%d] = (double)(", i);
            if (kflc_emit_expr(out, a->expr, arg_ctx, diag)) return 1;
            fputs(");\n", out);
            fprintf(out, "    hi[%d] = (double)(", i);
            if (kflc_emit_expr(out, a->expr2, arg_ctx, diag)) return 1;
            fputs(");\n", out);
        } else {
            fprintf(out, "    kind[%d] = K26RL_ACT_KIND_DISCRETE;\n", i);
            fprintf(out, "    arity[%d] = (uint32_t)(", i);
            if (kflc_emit_expr(out, a->expr, arg_ctx, diag)) return 1;
            fputs(");\n", out);
            fprintf(out, "    lo[%d] = 0.0;\n", i);
            fprintf(out, "    hi[%d] = (double)(arity[%d] > 0 ? "
                         "arity[%d] - 1 : 0);\n", i, i, i);
        }
    }
    fputs("}\n\n", out);

    /* Batch-mode action defaults: every channel holds its declared
     * default for the whole run (0 for box, choice 0 for discrete,
     * when no default is declared). */
    fputs("static void kflrl_act_defaults_(double *out_v)\n"
          "{\n"
          "    (void)out_v;\n", out);
    for (int i = 0; i < m->n_actions; i++) {
        const KflcAttr *d = rl_attr_(m->actions[i], "default");
        fprintf(out, "    out_v[%d] = (double)(", i);
        if (d && d->expr) {
            if (kflc_emit_expr(out, d->expr, arg_ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(");\n", out);
    }
    fputs("}\n\n", out);
    return 0;
}

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
} RlVehEmit;

/* World construction: the fn world prefix statements in source order.
 * astro_body declarations are emitted here (with the six scalar state
 * keys mapped onto the compound fields and distribution values drawn
 * at the environment's episode-0 coordinates); reinforcement learning
 * constructs and as-bound observes are the episode machinery's and
 * are skipped; everything else is the ordinary statement emitter. */
static int rl_emit_build_world_(FILE *out, const RlModel *m,
                                const KflcNode *form, KflcArena *arena,
                                KflcExprFn *user_fn_arr, int n_user_fns,
                                KflcDiag *diag)
{
    fputs("static int kflrl_build_world_(K26AstroWorld *world, "
          "K26RngKey _kfl_key,\n"
          "                              uint32_t _kfl_envi, "
          "double *_kfl_wscal,\n"
          "                              double *_kfl_dr0,\n"
          "                              K26AstroVehicle **_kfl_veh)\n"
          "{\n"
          "    const uint32_t _kfl_ep = 0;\n"
          "    (void)_kfl_key; (void)_kfl_envi; (void)_kfl_ep; "
          "(void)_kfl_wscal; (void)_kfl_dr0; (void)_kfl_veh;\n", out);

    /* Known-body index locals, the batch emitter's convention, so the
     * shared statement emitter resolves parent/observe targets. A
     * name declared twice shares one local, as the batch emitter's
     * dedup does. */
    for (int i = 0; i < m->n_bodies; i++) {
        const char *bn = m->bodies[i].body->name;
        int dup = 0;
        for (int j = 0; j < i; j++) {
            const char *pn = m->bodies[j].body->name;
            if (pn && strcmp(pn, bn) == 0) dup = 1;
        }
        if (dup) continue;
        fprintf(out, "    int _kfl_body_%s_idx = -1; "
                     "(void)_kfl_body_%s_idx;\n", bn, bn);
    }

    /* Binding table: form args, prefix lets, and the world handle. */
    KflcExprBinding *live = NULL;
    int live_n = 0, live_cap = 0;
    rl_collect_form_args_(form, arena, &live, &live_n, &live_cap);
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        if (s->kind == KFLN_STMT_EPISODE || s->kind == KFLN_STMT_ACTION ||
            s->kind == KFLN_STMT_ON_STEP || s->kind == KFLN_STMT_OBJECTIVE) {
            continue;
        }
        rl_collect_lets_(s, arena, &live, &live_n, &live_cap);
    }
    rl_push_binding_(arena, &live, &live_n, &live_cap, "world",
                     KFLT_OPAQUE);
    live[live_n - 1].type_subtype = "world";

    const char *known[RL_MAX_BODIES];
    int n_known = 0;
    for (int i = 0; i < m->n_bodies; i++) {
        const char *bn = m->bodies[i].body->name;
        int dup = 0;
        for (int j = 0; j < n_known; j++) {
            if (bn && strcmp(known[j], bn) == 0) dup = 1;
        }
        if (bn && !dup) known[n_known++] = bn;
    }

    KflcExprCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.bindings           = live;
    ctx.n_bindings         = live_n;
    ctx.fns                = user_fn_arr;
    ctx.n_fns              = n_user_fns;
    ctx.known_body_names   = known;
    ctx.n_known_body_names = n_known;
    ctx.form               = form;
    ctx.headless           = 1;

    kfl_emit_stmt_reset_scopes(arena, KFLT_VOID);

    int body_i = 0;
    int veh_i  = 0;
    RlVehEmit veh[RL_MAX_ACT];
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        switch (s->kind) {
        case KFLN_STMT_EPISODE:
        case KFLN_STMT_ACTION:
        case KFLN_STMT_ON_STEP:
        case KFLN_STMT_OBJECTIVE:
        case KFLN_STMT_SENSOR:
            continue;
        case KFLN_STMT_OBSERVE:
            if (rl_observe_as_(s)) continue;   /* channel, not a print */
            (void)kfl_emit_stmt(out, s, &ctx, diag, 4);
            continue;
        case KFLN_STMT_ASTRO_BODY: {
            /* The binding's rules and its numbers come from the one
             * resolver, so this emitter and the batch one cannot
             * disagree about what an assembly means. */
            KflcAssembly *asmb  = NULL;
            KflcArena    *asm_a = kflc_arena_create();
            if (!asm_a) return 1;
            if (kflc_assembly_for_body(s, diag->path, asm_a, diag, &asmb)) {
                kflc_arena_release(asm_a);
                return 1;
            }
            fputs("    {\n"
                  "        K26AstroBody _kfl_b; "
                  "k26astro_body_init(&_kfl_b);\n", out);
            fprintf(out,
                "        snprintf(_kfl_b.name, sizeof _kfl_b.name, "
                "\"%%s\", \"%s\");\n", s->name ? s->name : "_anon");
            if (asmb) {
                char hex[2 * KFLC_ASM_DIGEST + 1];
                kflc_assembly_digest_hex(asmb->digest, hex);
                fprintf(out, "        /* assembly `%s`, digest %s */\n",
                        asmb->name, hex);
                fprintf(out,
                        "        k26astro_body_set_mass(&_kfl_b, %.17g);\n",
                        asmb->mass);
                fprintf(out,
                        "        static const double _kfl_asm_com[3] = "
                        "{ %.17g, %.17g, %.17g };\n",
                        asmb->com[0], asmb->com[1], asmb->com[2]);
                fprintf(out,
                        "        static const double _kfl_asm_inertia[6] = "
                        "{ %.17g, %.17g, %.17g, %.17g, %.17g, %.17g };\n",
                        asmb->inertia[0], asmb->inertia[1], asmb->inertia[2],
                        asmb->inertia[3], asmb->inertia[4], asmb->inertia[5]);
                fputs("        (void)_kfl_asm_com; (void)_kfl_asm_inertia;\n",
                      out);
            }
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                if (!a->name) continue;
                if (strcmp(a->name, "assembly") == 0) continue;
                const char *val =
                    (a->value.kind == KFLV_IDENT && a->value.u.s)
                    ? a->value.u.s : "0";
                if (strcmp(a->name, "parent") == 0) {
                    if (rl_body_index_of_(m, val) >= 0) {
                        fprintf(out, "        _kfl_b.parent_body_idx = "
                                     "_kfl_body_%s_idx;\n", val);
                    } else {
                        fprintf(out, "        _kfl_b.parent_body_idx = "
                                     "k26astro_world_find_body(world, "
                                     "\"%s\");\n", val);
                    }
                    continue;
                }
                /* Distribution-valued attribute: episode-0 draw at
                 * this parameter's coordinates, the coordinate's only
                 * consumption at create; the value is handed out
                 * through _kfl_dr0 so kflrl_apply_draws_ records it
                 * without re-drawing. */
                int dr_slot = -1;
                for (int d = 0; d < m->n_dr; d++) {
                    if (m->dr[d].attr == a) dr_slot = d;
                }
                if (dr_slot >= 0) {
                    fputs("        {\n            double _kfl_v = ", out);
                    if (rl_emit_draw_(out, m->dr[dr_slot].dist, 0x0002u,
                                      m->dr[dr_slot].channel, &ctx, diag)) {
                        return 1;
                    }
                    fputs(";\n", out);
                    fprintf(out, "            if (_kfl_dr0) "
                                 "_kfl_dr0[%d] = _kfl_v;\n", dr_slot);
                    rl_emit_body_write_var_(out, 12, "_kfl_b.", a->name,
                                            "_kfl_v");
                    fputs("        }\n", out);
                } else {
                    rl_emit_body_write_(out, 8, "_kfl_b.", a->name, val);
                }
            }
            fprintf(out,
                "        _kfl_body_%s_idx = "
                "k26astro_world_add_body(world, _kfl_b);\n"
                "        if (_kfl_body_%s_idx < 0) return -1;\n"
                "        kflrl_body_idx_[%d] = _kfl_body_%s_idx;\n",
                s->name, s->name, body_i, s->name);
            if (asmb) {
                if (veh_i >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, s->line,
                        "more than %d bodies carry an assembly",
                        RL_MAX_ACT);
                    kflc_arena_release(asm_a);
                    return 1;
                }
                veh[veh_i].body_name = s->name;
                veh[veh_i].mass      = asmb->mass;
                memcpy(veh[veh_i].com, asmb->com, sizeof veh[veh_i].com);
                memcpy(veh[veh_i].inertia, asmb->inertia,
                       sizeof veh[veh_i].inertia);
                veh_i++;
            }
            fputs("    }\n", out);
            kflc_arena_release(asm_a);
            body_i++;
            continue;
        }
        default:
            (void)kfl_emit_stmt(out, s, &ctx, diag, 4);
            continue;
        }
    }
    kfl_emit_stmt_drain_root(out, 4);

    /* The vehicles, after the last body has been added. Each carries
     * the derived inertia tensor and the centre-of-mass offset, and
     * binds the body the world now owns; the bind is only valid here,
     * for the reason RlVehEmit states. The world's registry is
     * non-owning, so the handle owns the vehicle and destroys it; this
     * is the one place that knows both when a world appears and when
     * it goes. Slot order is body declaration order, which is the
     * order every per-vehicle table in this artifact is written in. */
    for (int i = 0; i < veh_i; i++) {
        const RlVehEmit *ve = &veh[i];
        fprintf(out,
            "    {\n"
            "        K26AstroVehicle *_kfl_v = k26astro_vehicle_new();\n"
            "        if (!_kfl_v) return -1;\n"
            "        k26astro_vehicle_set_dry_mass(_kfl_v, %.17g);\n"
            "        k26astro_vehicle_set_com_offset(_kfl_v, "
            "%.17g, %.17g, %.17g);\n",
                ve->mass, ve->com[0], ve->com[1], ve->com[2]);
        fprintf(out,
            "        K26M3 _kfl_I;\n"
            "        _kfl_I.m[0][0] = %.17g; _kfl_I.m[0][1] = %.17g; "
            "_kfl_I.m[0][2] = %.17g;\n"
            "        _kfl_I.m[1][0] = %.17g; _kfl_I.m[1][1] = %.17g; "
            "_kfl_I.m[1][2] = %.17g;\n"
            "        _kfl_I.m[2][0] = %.17g; _kfl_I.m[2][1] = %.17g; "
            "_kfl_I.m[2][2] = %.17g;\n"
            "        k26astro_vehicle_set_inertia_full(_kfl_v, _kfl_I);\n",
                ve->inertia[0], ve->inertia[3], ve->inertia[4],
                ve->inertia[3], ve->inertia[1], ve->inertia[5],
                ve->inertia[4], ve->inertia[5], ve->inertia[2]);
        fprintf(out,
            "        k26astro_vehicle_bind_body(_kfl_v, "
            "k26astro_world_body_at(world, _kfl_body_%s_idx));\n"
            "        if (k26astro_world_register_vehicle(world, "
            "_kfl_v) != 0) {\n"
            "            k26astro_vehicle_destroy(_kfl_v);\n"
            "            return -1;\n"
            "        }\n"
            "        if (_kfl_veh) _kfl_veh[%d] = _kfl_v;\n"
            "    }\n",
                ve->body_name, i);
    }

    /* Capture the world scalars for the objective and termination
     * evaluators: the prefix runs once per environment at create, and
     * these are its final values. */
    for (int i = 0; i < m->n_wscal; i++) {
        fprintf(out, "    _kfl_wscal[%d] = (double)(%s);\n",
                i, m->wscal[i]->name);
    }
    fputs("    return 0;\n}\n\n", out);
    return diag->errors ? 1 : 0;
}

/* Per-episode draws: every drawn parameter is computed first into the
 * record slots (ascending tag order), then applied, attribute draws
 * before reset lines so a reset line targeting the same field wins,
 * as initial state overrides the built value. */
static int rl_emit_apply_draws_(FILE *out, const RlModel *m,
                                const KflcExprCtx *arg_ctx, KflcDiag *diag)
{
    fputs("static void kflrl_apply_draws_(K26AstroWorld *world, "
          "K26RngKey _kfl_key,\n"
          "                               uint32_t _kfl_envi, "
          "uint32_t _kfl_ep,\n"
          "                               double *rec,\n"
          "                               const double *_kfl_dr0)\n"
          "{\n"
          "    (void)world; (void)_kfl_key; (void)_kfl_envi; "
          "(void)_kfl_ep; (void)rec; (void)_kfl_dr0;\n", out);

    for (int i = 0; i < m->n_resets; i++) {
        fprintf(out, "    rec[%d] = ", i);
        if (rl_emit_draw_(out, m->resets[i]->expr, 0x0001u, i,
                          arg_ctx, diag)) {
            return 1;
        }
        fputs(";\n", out);
    }
    /* At create the attribute draws were consumed while the world was
     * built; _kfl_dr0 hands their values in so each (seed, coordinate)
     * pair is drawn exactly once per handle. Boundary resets pass
     * null and draw at the new episode index. */
    for (int i = 0; i < m->n_dr; i++) {
        fprintf(out, "    rec[%d] = _kfl_dr0 ? _kfl_dr0[%d] : (",
                m->n_resets + i, i);
        if (rl_emit_draw_(out, m->dr[i].dist, 0x0002u, m->dr[i].channel,
                          arg_ctx, diag)) {
            return 1;
        }
        fputs(");\n", out);
    }

    for (int i = 0; i < m->n_dr; i++) {
        char var[32];
        snprintf(var, sizeof var, "rec[%d]", m->n_resets + i);
        fprintf(out, "    {\n        K26AstroBody *_kfl_bp = "
                     "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                     "        if (_kfl_bp) {\n", m->dr[i].body);
        rl_emit_body_write_var_(out, 12, "_kfl_bp->", m->dr[i].attr->name,
                                var);
        fputs("        }\n    }\n", out);
    }
    for (int i = 0; i < m->n_resets; i++) {
        const KflcNode *r = m->resets[i];
        int bidx = rl_body_index_of_(m, r->name);
        const char *key = (r->position.kind == KFLV_IDENT)
                          ? r->position.u.s : NULL;
        if (bidx < 0 || !key) continue;   /* reported at collect */
        char var[32];
        snprintf(var, sizeof var, "rec[%d]", i);
        fprintf(out, "    {\n        K26AstroBody *_kfl_bp = "
                     "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                     "        if (_kfl_bp) {\n", bidx);
        rl_emit_body_write_var_(out, 12, "_kfl_bp->", key, var);
        fputs("        }\n    }\n", out);
    }
    fputs("}\n\n", out);
    return 0;
}

/* Observation recompute: the as-bound observes in source order, four
 * channels each. The range channel is the magnitude of the relative
 * position vector between the corrected target position and the
 * observer, in metres. */
static int rl_emit_observe_(FILE *out, const RlModel *m,
                            KflcDiag *diag)
{
    fputs("/* The observation vector for one environment. `ct` is that\n"
          " * environment's latched contact block, which a contact\n"
          " * observe reads: contact is a fact about the transition\n"
          " * just taken rather than about where a body is, so it is\n"
          " * passed in rather than read back out of the world. */\n"
          "static void kflrl_observe_(K26AstroWorld *world, "
          "double *out_v,\n"
          "                           const KflrlContact *ct,\n"
          "                           const KflrlJoin *jn)\n"
          "{\n"
          "    (void)world; (void)out_v; (void)ct; (void)jn;\n", out);
    for (int i = 0; i < m->n_observes; i++) {
        const KflcNode *s = m->observes[i];
        int off = rl_obs_offset_(m->observes, i);
        if (rl_observe_form_(s) == RL_OBS_PORT) {
            /* The form names a port on a body, and the state it
             * publishes is that port's with respect to the one it
             * faces. Which port it faces is not written in the
             * statement, so it is resolved here: the one port
             * declared on any other body. A world with none has
             * nothing to measure against, and a world with several
             * leaves the pairing to an accident of declaration
             * order, so both are refused where they are written
             * rather than publishing nine channels about an
             * arbitrary choice. */
            const char *pname = NULL;
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                if (a->name && strcmp(a->name, "port") == 0 &&
                    a->value.kind == KFLV_IDENT) {
                    pname = a->value.u.s;
                }
            }
            int tgt = rl_body_index_of_(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of `%s`: no astro_body of that name "
                    "is declared in this world", pname ? pname : "?",
                    s->name);
                return 1;
            }
            if (!rl_body_has_assembly_(m, tgt)) {
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of `%s`: `%s` declares no "
                    "`assembly=`, so it carries no docking ports",
                    pname ? pname : "?", s->name, s->name);
                return 1;
            }
            int active = -1;
            for (int q = 0; q < m->n_ports; q++) {
                if (m->ports[q].body == tgt && pname &&
                    strcmp(m->ports[q].name, pname) == 0) {
                    active = q;
                }
            }
            if (active < 0) {
                char have[256];
                size_t used = 0;
                int    seen = 0;
                have[0] = '\0';
                for (int q = 0; q < m->n_ports; q++) {
                    if (m->ports[q].body != tgt) continue;
                    int wrote = snprintf(have + used, sizeof have - used,
                                         "%s`%s`", seen++ ? ", " : "",
                                         m->ports[q].name);
                    if (wrote < 0 || (size_t)wrote >= sizeof have - used) break;
                    used += (size_t)wrote;
                }
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of %s: `%s` declares no port of that "
                    "name carrying a capture envelope; it carries %s",
                    pname ? pname : "?", s->name, s->name,
                    seen ? have : "none");
                return 1;
            }
            int passive = -1, others = 0;
            for (int q = 0; q < m->n_ports; q++) {
                if (m->ports[q].body == tgt) continue;
                passive = q;
                others++;
            }
            if (others != 1) {
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of %s: the state this form publishes "
                    "is against the port it faces, and %d ports carrying a "
                    "capture envelope are declared on other bodies; exactly "
                    "one is needed", pname, s->name, others);
                return 1;
            }
            int aslot = m->ports[active].veh;
            int pbody = m->ports[passive].body;
            fprintf(out,
                "    {\n"
                "        const K26AstroBody *_kfl_pa = "
                "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                "        const K26AstroBody *_kfl_pp = "
                "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                "        K26AstroCollPortState _kfl_ps;\n"
                "        double _kfl_cap = 0.0;\n"
                "        memset(&_kfl_ps, 0, sizeof _kfl_ps);\n"
                /* A transition that ended at this interface publishes
                 * the state the capture test was given; any other
                 * step publishes the state as it stands, which is
                 * what an approach is flown on. */
                "        if (ct && ct[%d].port_hit != 0.0) {\n"
                "            _kfl_cap           = ct[%d].captured;\n"
                "            _kfl_ps.axial      = ct[%d].axial;\n"
                "            _kfl_ps.lateral    = ct[%d].lateral;\n"
                "            _kfl_ps.pitchyaw   = ct[%d].pitchyaw;\n"
                "            _kfl_ps.roll       = ct[%d].roll;\n"
                "            _kfl_ps.v_axial    = ct[%d].v_axial;\n"
                "            _kfl_ps.v_lateral  = ct[%d].v_lateral;\n"
                "            _kfl_ps.v_pitchyaw = ct[%d].v_pitchyaw;\n"
                "            _kfl_ps.v_roll     = ct[%d].v_roll;\n"
                "        } else if (_kfl_pa && _kfl_pp) {\n"
                "            K26AstroCollBody _kfl_ba, _kfl_bp;\n"
                "            kflrl_port_snap_(_kfl_pa, "
                "kflrl_ports_[%d].com,\n"
                "                k26astro_pos_sub(&_kfl_pa->pos, "
                "&_kfl_pp->pos), &_kfl_ba);\n"
                "            kflrl_port_snap_(_kfl_pp, "
                "kflrl_ports_[%d].com,\n"
                "                k26m3d_v3(0.0, 0.0, 0.0), &_kfl_bp);\n"
                "            (void)k26astro_coll_port_state(&_kfl_ba,\n"
                "                &kflrl_ports_[%d].geom, &_kfl_bp,\n"
                "                &kflrl_ports_[%d].geom, 0.0, &_kfl_ps);\n"
                "        }\n"
                "        out_v[%d] = _kfl_cap;\n"
                "        out_v[%d] = _kfl_ps.axial;\n"
                "        out_v[%d] = _kfl_ps.lateral;\n"
                "        out_v[%d] = _kfl_ps.pitchyaw;\n"
                "        out_v[%d] = _kfl_ps.roll;\n"
                "        out_v[%d] = _kfl_ps.v_axial;\n"
                "        out_v[%d] = _kfl_ps.v_lateral;\n"
                "        out_v[%d] = _kfl_ps.v_pitchyaw;\n"
                "        out_v[%d] = _kfl_ps.v_roll;\n"
                "    }\n",
                tgt, pbody,
                aslot, aslot, aslot, aslot, aslot, aslot, aslot, aslot,
                aslot, aslot,
                active, passive,
                active, passive,
                off, off + 1, off + 2, off + 3, off + 4, off + 5,
                off + 6, off + 7, off + 8);
            continue;
        }
        if (rl_observe_form_(s) == RL_OBS_CON) {
            /* The body must be one the pass can report on, which is a
             * body that binds an assembly: without one it has no
             * colliders and would publish three channels that could
             * never be anything but zero. */
            int tgt = rl_body_index_of_(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe contact of `%s`: no astro_body of that name "
                    "is declared in this world", s->name);
                return 1;
            }
            int slot = -1, seen = 0;
            for (int b = 0; b < m->n_bodies; b++) {
                if (!rl_body_has_assembly_(m, b)) continue;
                if (b == tgt) slot = seen;
                seen++;
            }
            if (slot < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe contact of `%s`: `%s` declares no "
                    "`assembly=`, so it carries no colliders and can "
                    "report no contact", s->name, s->name);
                return 1;
            }
            fprintf(out,
                "    if (ct) {\n"
                "        out_v[%d] = ct[%d].hit;\n"
                "        out_v[%d] = ct[%d].fraction;\n"
                "        out_v[%d] = ct[%d].speed;\n"
                "    } else {\n"
                "        out_v[%d] = 0.0; out_v[%d] = 0.0; "
                "out_v[%d] = 0.0;\n"
                "    }\n",
                off, slot, off + 1, slot, off + 2, slot,
                off, off + 1, off + 2);
            continue;
        }
        if (rl_observe_form_(s) == RL_OBS_REL) {
            /* Both bodies must be declared here, and the chief must
             * name a parent, because the chief's frame is built from
             * its state relative to the body it orbits and there is no
             * other way to know which body that is. A chief without a
             * parent is a declaration that cannot produce the frame at
             * all, so it is refused where it is written rather than
             * publishing six channels that could only ever be zero. */
            const char *chief = NULL;
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                if (a->name && strcmp(a->name, "observer") == 0 &&
                    a->value.kind == KFLV_IDENT) {
                    chief = a->value.u.s;
                }
            }
            int tgt = rl_body_index_of_(m, s->name);
            int chf = chief ? rl_body_index_of_(m, chief) : -1;
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative `%s`: no astro_body of that name "
                    "is declared in this world", s->name);
                return 1;
            }
            if (chf < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative %s from `%s`: no astro_body of "
                    "that name is declared in this world", s->name,
                    chief ? chief : "?");
                return 1;
            }
            if (chf == tgt) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative %s from `%s`: a body has no "
                    "relative state with respect to itself", s->name,
                    chief);
                return 1;
            }
            if (!rl_body_attr_(m->bodies[chf].body, "parent")) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative %s from `%s`: `%s` declares no "
                    "`parent=`, so the body it orbits is unknown and "
                    "its local-vertical local-horizontal frame cannot "
                    "be built", s->name, chief, chief);
                return 1;
            }
            fprintf(out,
                "    {\n"
                "        K26AstroBody *_kfl_cb = k26astro_world_body_at("
                "world, kflrl_body_idx_[%d]);\n"
                "        K26AstroBody *_kfl_tb = k26astro_world_body_at("
                "world, kflrl_body_idx_[%d]);\n"
                "        K26AstroBody *_kfl_pb = (_kfl_cb && "
                "_kfl_cb->parent_body_idx >= 0)\n"
                "            ? k26astro_world_body_at(world, "
                "_kfl_cb->parent_body_idx) : NULL;\n"
                "        K26AstroProxRel _kfl_rr;\n"
                "        _kfl_rr.r = k26m3d_v3(0.0, 0.0, 0.0);\n"
                "        _kfl_rr.v = k26m3d_v3(0.0, 0.0, 0.0);\n"
                /* A state with no frame publishes zeros: the chief
                 * sitting at its parent's centre, or moving straight
                 * at it, names no direction of motion. That is a
                 * configuration and not a declaration, so it cannot be
                 * refused at compile time and is reported as the
                 * absence of a measurement. */
                "        if (_kfl_cb && _kfl_tb && _kfl_pb) {\n"
                "            K26AstroProxFrame _kfl_f;\n"
                "            if (k26astro_prox_frame(&_kfl_pb->pos, "
                "_kfl_pb->vel,\n"
                "                                    &_kfl_cb->pos, "
                "_kfl_cb->vel,\n"
                "                                    &_kfl_f) == "
                "K26ASTRO_PROX_OK) {\n"
                "                (void)k26astro_prox_relative(&_kfl_f,\n"
                "                    &_kfl_cb->pos, _kfl_cb->vel,\n"
                "                    &_kfl_tb->pos, _kfl_tb->vel, "
                "&_kfl_rr);\n"
                "            }\n"
                "        }\n"
                "        out_v[%d] = _kfl_rr.r.x;\n"
                "        out_v[%d] = _kfl_rr.r.y;\n"
                "        out_v[%d] = _kfl_rr.r.z;\n"
                "        out_v[%d] = _kfl_rr.v.x;\n"
                "        out_v[%d] = _kfl_rr.v.y;\n"
                "        out_v[%d] = _kfl_rr.v.z;\n"
                "    }\n",
                chf, tgt, off, off + 1, off + 2, off + 3, off + 4,
                off + 5);
            continue;
        }
        if (rl_observe_is_attitude_(s)) {
            /* A body reporting itself: the orientation and the rate
             * as they stand after the advance, with no observer, no
             * light-time correction and no aberration to apply. */
            int tgt = rl_body_index_of_(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe attitude of `%s`: no astro_body of that name "
                    "is declared in this world", s->name);
                return 1;
            }
            fprintf(out,
                "    {\n"
                "        const K26AstroBody *_kfl_b = "
                "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                "        if (_kfl_b) {\n"
                "            out_v[%d] = _kfl_b->attitude.w;\n"
                "            out_v[%d] = _kfl_b->attitude.x;\n"
                "            out_v[%d] = _kfl_b->attitude.y;\n"
                "            out_v[%d] = _kfl_b->attitude.z;\n"
                "            out_v[%d] = _kfl_b->omega.x;\n"
                "            out_v[%d] = _kfl_b->omega.y;\n"
                "            out_v[%d] = _kfl_b->omega.z;\n"
                "        } else {\n"
                "            out_v[%d] = 1.0;\n"
                "            out_v[%d] = 0.0; out_v[%d] = 0.0; "
                "out_v[%d] = 0.0;\n"
                "            out_v[%d] = 0.0; out_v[%d] = 0.0; "
                "out_v[%d] = 0.0;\n"
                "        }\n"
                "    }\n",
                tgt, off, off + 1, off + 2, off + 3, off + 4, off + 5,
                off + 6, off, off + 1, off + 2, off + 3, off + 4,
                off + 5, off + 6);
            continue;
        }
        const char *observer = "_observer";
        const char *mode_kw = NULL;
        for (const KflcAttr *a = s->attrs; a; a = a->next) {
            if (!a->name) continue;
            if (strcmp(a->name, "observer") == 0 &&
                a->value.kind == KFLV_IDENT && a->value.u.s) {
                observer = a->value.u.s;
            } else if (strcmp(a->name, "mode") == 0 &&
                       a->value.kind == KFLV_IDENT) {
                mode_kw = a->value.u.s;
            }
        }
        int tgt = rl_body_index_of_(m, s->name);
        int obs = rl_body_index_of_(m, observer);
        fputs("    {\n", out);
        if (mode_kw) {
            const char *en = "K26ASTRO_OBS_ASTROMETRIC";
            if      (strcmp(mode_kw, "geometric") == 0)
                en = "K26ASTRO_OBS_GEOMETRIC";
            else if (strcmp(mode_kw, "apparent") == 0)
                en = "K26ASTRO_OBS_APPARENT";
            else if (strcmp(mode_kw, "topocentric") == 0)
                en = "K26ASTRO_OBS_TOPOCENTRIC";
            fprintf(out, "        (void)k26astro_world_set_observer_mode"
                         "(world, %s);\n", en);
        }
        if (tgt >= 0) {
            fprintf(out, "        int _kfl_t = kflrl_body_idx_[%d];\n",
                    tgt);
        } else {
            fprintf(out, "        int _kfl_t = k26astro_world_find_body"
                         "(world, \"%s\");\n", s->name ? s->name : "?");
        }
        if (obs >= 0) {
            fprintf(out, "        int _kfl_o = kflrl_body_idx_[%d];\n",
                    obs);
        } else {
            fprintf(out, "        int _kfl_o = k26astro_world_find_body"
                         "(world, \"%s\");\n", observer);
        }
        fprintf(out,
            "        K26AstroPos _kfl_p; memset(&_kfl_p, 0, sizeof "
            "_kfl_p);\n"
            "        K26V3 _kfl_d; _kfl_d.x = _kfl_d.y = _kfl_d.z = "
            "0.0;\n"
            "        double _kfl_range = 0.0;\n"
            "        double _kfl_rrate = 0.0;\n"
            "        if (_kfl_t >= 0 && _kfl_o >= 0) {\n"
            "            (void)k26astro_world_observe(world, _kfl_t, "
            "_kfl_o, &_kfl_p, &_kfl_d);\n"
            "            K26AstroBody *_kfl_ob = "
            "k26astro_world_body_at(world, _kfl_o);\n"
            "            K26AstroBody *_kfl_tb = "
            "k26astro_world_body_at(world, _kfl_t);\n"
            "            if (_kfl_ob) {\n"
            "                K26V3 _kfl_r = k26astro_pos_sub(&_kfl_p, "
            "&_kfl_ob->pos);\n"
            "                _kfl_range = std::sqrt(_kfl_r.x * _kfl_r.x "
            "+ _kfl_r.y * _kfl_r.y + _kfl_r.z * _kfl_r.z);\n"
            "            }\n"
            /* The range rate is the geometric one: the observation
             * mode corrects a position, and there is no corrected
             * velocity to differentiate, so the rate is taken from
             * the two bodies' current state. Zero separation yields
             * 0.0 rather than a quotient of zeroes. */
            "            if (_kfl_ob && _kfl_tb) {\n"
            "                K26V3 _kfl_gr = k26astro_pos_sub("
            "&_kfl_tb->pos, &_kfl_ob->pos);\n"
            "                K26V3 _kfl_gv;\n"
            "                _kfl_gv.x = _kfl_tb->vel.x - _kfl_ob->vel.x;\n"
            "                _kfl_gv.y = _kfl_tb->vel.y - _kfl_ob->vel.y;\n"
            "                _kfl_gv.z = _kfl_tb->vel.z - _kfl_ob->vel.z;\n"
            "                double _kfl_gm = std::sqrt("
            "_kfl_gr.x * _kfl_gr.x + _kfl_gr.y * _kfl_gr.y"
            " + _kfl_gr.z * _kfl_gr.z);\n"
            "                if (_kfl_gm != 0.0) {\n"
            "                    _kfl_rrate = (_kfl_gr.x * _kfl_gv.x"
            " + _kfl_gr.y * _kfl_gv.y + _kfl_gr.z * _kfl_gv.z)"
            " / _kfl_gm;\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        out_v[%d] = _kfl_d.x;\n"
            "        out_v[%d] = _kfl_d.y;\n"
            "        out_v[%d] = _kfl_d.z;\n"
            "        out_v[%d] = _kfl_range;\n"
            "        out_v[%d] = _kfl_rrate;\n"
            "    }\n",
            off, off + 1, off + 2, off + 3, off + 4);
    }
    /* The truth half. A paired observe publishes each component twice,
     * and both halves leave this function carrying the same
     * uncorrupted value; the sensor pass below rewrites the measured
     * half in place. Copying here rather than computing twice is what
     * keeps the two halves the same number before any noise, which is
     * the property `with truth` exists to give a consumer. */
    for (int i = 0; i < m->n_observes; i++) {
        if (!rl_observe_has_truth_(m->observes[i])) continue;
        int off = rl_obs_offset_(m->observes, i);
        int w   = rl_observe_base_width_(m->observes[i]);
        fprintf(out,
            "    for (int k = 0; k < %d; k++) out_v[%d + k] = out_v[%d + k];\n",
            w, off + w, off);
    }
    fputs("}\n\n", out);

    return 0;
}

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

static int rl_sweep_stmts_(RlSweep *sw, const KflcNode *stmts);

static const KflcNode *rl_find_user_fn_(const KflcNode *form,
                                        const char *name)
{
    if (!form || !name) return NULL;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_FN && c->name && strcmp(c->name, name) == 0) {
            return c;
        }
    }
    return NULL;
}

/* A vector or matrix value owns heap storage: every way of making one
 * (a literal, `zeros`, `ones`, `linspace`, `arange`) lowers to
 * k26c_vec_alloc, k26c_mat_alloc or k26c_vec_from, and all three
 * allocate. The type is therefore judged rather than the constructor,
 * so no list of constructor names has to be kept in step with the
 * emitter's. */
static int rl_type_allocates_(KflcType t)
{
    return t == KFLT_VECTOR || t == KFLT_MATRIX;
}

/* Statements the stepping path may carry, and the reason each of the
 * rest may not. The admissible set is closed: a kind that is neither
 * admitted here nor named with a reason is refused, because the
 * compiler cannot show it is free of effects and admitting the
 * unfamiliar by default is exactly how this surface was left open.
 *
 * A statement form whose effect is declared and bounded joins the
 * admitted group, which is an addition to this rule rather than an
 * exception to it. Returns NULL when `s` is admissible, otherwise the
 * reason, with *what set to the name to report. */
static const char *rl_step_stmt_why_(const KflcNode *s, const char **what)
{
    /* A binding whose storage is heap allocated is refused whatever
     * its initialiser, since the allocation is the binding's. */
    if ((s->kind == KFLN_STMT_LET || s->kind == KFLN_STMT_CONST ||
         s->kind == KFLN_FN_ARG) && rl_type_allocates_(s->type)) {
        *what = s->name ? s->name : "?";
        return "a vector or matrix binding owns heap storage, and the "
               "stepping path allocates nothing";
    }

    switch (s->kind) {
    /* Pure computation, control flow, and the function plumbing that
     * carries them. */
    case KFLN_STMT_LET:
    case KFLN_STMT_CONST:
    case KFLN_STMT_ASSIGN:
    case KFLN_STMT_INDEX_ASSIGN:
    case KFLN_STMT_LVALUE_ASSIGN:
    case KFLN_STMT_RETURN:
    case KFLN_STMT_EXPR:
    case KFLN_STMT_IF:
    case KFLN_STMT_WHILE:
    case KFLN_STMT_FOR_EACH:
    case KFLN_FN_ARG:
        return NULL;

    case KFLN_STMT_PRINT:
        *what = "print";
        return "the stepping path performs no I/O";
    case KFLN_STMT_ASTRO_BODY:
        *what = "astro_body";
        return "it adds a body to the world, which allocates and can "
               "move every body already in it";
    case KFLN_STMT_STEP:
        *what = "step";
        return "it advances the world, and the episode machinery owns "
               "stepping in these programs";
    case KFLN_STMT_PROPAGATE:
        *what = "propagate";
        return "it advances a body, and the episode machinery owns "
               "stepping in these programs";
    case KFLN_STMT_OBSERVE:
        *what = "observe";
        return "it runs the world's observer pipeline and sets the "
               "world's observer mode";
    case KFLN_STMT_SERIES:
        *what = "series";
        return "it packages plot data, which is not something a step "
               "produces";
    case KFLN_ALLOCATOR_BIND:
        *what = "allocator";
        return "it binds an arena, and the stepping path allocates "
               "nothing";
    case KFLN_STMT_EPISODE:
    case KFLN_STMT_EPISODE_RESET:
    case KFLN_STMT_ACTION:
    case KFLN_STMT_ON_STEP:
    case KFLN_STMT_OBJECTIVE:
    case KFLN_STMT_SENSOR:
    case KFLN_STMT_SENSOR_TERM:
        *what = "a reinforcement learning construct";
        return "these are declarations of the environment, not acts of "
               "a step";
    default:
        *what = "this statement";
        return "the compiler cannot show that it is free of effects";
    }
}

static int rl_sweep_expr_(RlSweep *sw, const KflcExpr *e)
{
    if (!e) return 0;
    switch (e->kind) {
    case KFLE_UNARY:
        return rl_sweep_expr_(sw, e->u.un.operand);
    case KFLE_BINARY:
        return rl_sweep_expr_(sw, e->u.bin.lhs) ||
               rl_sweep_expr_(sw, e->u.bin.rhs);
    case KFLE_INDEX:
        return rl_sweep_expr_(sw, e->u.index.base) ||
               rl_sweep_expr_(sw, e->u.index.idx);
    case KFLE_VEC_LIT:
        /* The elements are swept even though a literal in a scalar
         * context is already an error, so that the walk does not
         * depend on another pass having run first. */
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            if (rl_sweep_expr_(sw, e->u.vec.elems[i])) return 1;
        }
        return 0;
    case KFLE_CALL: {
        for (int i = 0; i < e->u.call.n_args; i++) {
            if (rl_sweep_expr_(sw, e->u.call.args[i])) return 1;
        }
        const char *nm = e->u.call.name;
        if (!nm) return 0;
        if (kflc_builtin_known(nm)) {
            if (kflc_builtin_is_pure(nm)) return 0;
            sw->found = nm;
            sw->why   = NULL;
            return 1;
        }
        const KflcNode *fn = rl_find_user_fn_(sw->form, nm);
        if (!fn) {
            /* Neither a builtin nor a user function: a form the
             * compiler lowers itself, such as the vector builders,
             * which allocate. Nothing here can show it is free of
             * effects, so it is refused by name. */
            sw->found = nm;
            sw->why   = "the compiler cannot show that this call is free "
                        "of effects, and the forms it lowers itself "
                        "include ones that allocate";
            return 1;
        }
        for (int i = 0; i < sw->n_seen; i++) {
            if (strcmp(sw->seen[i], nm) == 0) return 0;   /* walked */
        }
        if (sw->n_seen >= RL_SWEEP_MAX_FNS ||
            sw->n_chain >= RL_SWEEP_MAX_FNS) {
            sw->found = nm;
            sw->depth = 1;
            return 1;
        }
        if (rl_type_allocates_(fn->type)) {
            sw->found = nm;
            sw->why   = "it returns a vector or matrix, which owns heap "
                        "storage, and the stepping path allocates nothing";
            return 1;
        }
        sw->seen[sw->n_seen++]   = nm;
        sw->chain[sw->n_chain++] = nm;
        if (rl_sweep_stmts_(sw, fn->children)) return 1;
        sw->n_chain--;
        return 0;
    }
    default:
        return 0;
    }
}

static int rl_sweep_stmts_(RlSweep *sw, const KflcNode *stmts)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        const char *what = NULL;
        const char *why  = rl_step_stmt_why_(s, &what);
        sw->line = s->line;
        if (why) {
            sw->found = what;
            sw->why   = why;
            return 1;
        }
        if (rl_sweep_expr_(sw, s->expr)) return 1;
        if (rl_sweep_expr_(sw, s->expr2)) return 1;
        if (rl_sweep_stmts_(sw, s->children)) return 1;
        if (rl_sweep_stmts_(sw, s->else_children)) return 1;
    }
    return 0;
}

static void rl_sweep_init_(RlSweep *sw, const KflcNode *form)
{
    memset(sw, 0, sizeof *sw);
    sw->form = form;
}

/* "`fn a`" for one, "`fn a` -> `fn b`" for a chain, so a reader is
 * told the whole route rather than its first step. */
static void rl_sweep_chain_(const RlSweep *sw, char *buf, size_t n)
{
    size_t off = 0;
    buf[0] = '\0';
    for (int i = 0; i < sw->n_chain && off + 1 < n; i++) {
        int k = snprintf(buf + off, n - off, "%s`fn %s`",
                         i ? " -> " : "", sw->chain[i]);
        if (k < 0 || (size_t)k >= n - off) break;
        off += (size_t)k;
    }
}

/* Reports the finding `sw` holds. `what` names the position and
 * completes the sentence "<what> must be side-effect free"; `line` is
 * the position's own line, used when the reach ended there rather
 * than inside a called function. */
static void rl_sweep_report_(const RlSweep *sw, const char *what, int line,
                             KflcDiag *diag)
{
    char chain[512];
    rl_sweep_chain_(sw, chain, sizeof chain);
    /* Always reported at the position's own line, which is the line
     * the author edits; when the reach ended inside a called function
     * the message carries that function's line as well, so neither
     * end of the route has to be hunted for. */
    int at = line;

    if (sw->depth) {
        kflc_diag_errorf(diag, at,
            "%s must be side-effect free, and the compiler follows at "
            "most %d nested user functions from a position; this one "
            "goes deeper at `fn %s`, so nothing here can show it is, "
            "and it is refused rather than assumed",
            what, RL_SWEEP_MAX_FNS, sw->found);
        return;
    }

    if (sw->n_chain == 0) {
        if (sw->why) {
            kflc_diag_errorf(diag, at,
                "%s must be side-effect free, and `%s` is not allowed on "
                "the stepping path: %s", what, sw->found, sw->why);
        } else {
            kflc_diag_errorf(diag, at,
                "%s must be side-effect free, and `%s` is not a pure "
                "builtin", what, sw->found);
        }
        return;
    }

    if (sw->n_chain == 1) {
        if (sw->why) {
            kflc_diag_errorf(diag, at,
                "%s must be side-effect free, and %s called here reaches "
                "`%s` at line %d, which is not allowed on the stepping "
                "path: %s", what, chain, sw->found, sw->line, sw->why);
        } else {
            kflc_diag_errorf(diag, at,
                "%s must be side-effect free, and %s called here reaches "
                "`%s` at line %d, which is not pure",
                what, chain, sw->found, sw->line);
        }
        return;
    }

    if (sw->why) {
        kflc_diag_errorf(diag, at,
            "%s must be side-effect free, and the call chain %s from here "
            "reaches `%s` at line %d, which is not allowed on the stepping "
            "path: %s", what, chain, sw->found, sw->line, sw->why);
    } else {
        kflc_diag_errorf(diag, at,
            "%s must be side-effect free, and the call chain %s from here "
            "reaches `%s` at line %d, which is not pure",
            what, chain, sw->found, sw->line);
    }
}

/* An expression the runtime has to be able to re-evaluate is refused
 * when it reaches anything the stepping path may not carry, calls into
 * user fns followed to any depth. `what` names the position and
 * completes the sentence "<what> must be side-effect free". Every
 * position that replays, whether by re-simulation or by being
 * evaluated once per step, uses this: a reach there would make a
 * recorded episode irreproducible from its recorded inputs. */
static int rl_reject_impure_(const KflcNode *form, const KflcExpr *e,
                             const char *what, int line, KflcDiag *diag)
{
    RlSweep sw;
    rl_sweep_init_(&sw, form);
    if (!rl_sweep_expr_(&sw, e)) return 0;
    rl_sweep_report_(&sw, what, line, diag);
    return 1;
}

/* A statement written directly in the block, judged by the same rule
 * that judges one reached through a call. `block` names the enclosing
 * block for the diagnostic, or is NULL at the top of the body. */
static int rl_reject_stmt_(const KflcNode *s, const char *block,
                           KflcDiag *diag)
{
    const char *what = NULL;
    const char *why  = rl_step_stmt_why_(s, &what);
    if (!why) return 0;
    if (block) {
        kflc_diag_errorf(diag, s->line,
            "on_step: `%s` is not allowed on the stepping path in %s: %s",
            what, block, why);
    } else {
        kflc_diag_errorf(diag, s->line,
            "on_step: `%s` is not allowed on the stepping path: %s",
            what, why);
    }
    return 1;
}

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

/* The name of the block `s` opens, for the diagnostics of the
 * statements inside it; `outer` for a statement that opens none. */
static const char *rl_step_block_(const KflcNode *s, const char *outer)
{
    switch (s->kind) {
    case KFLN_STMT_IF:       return "an `if` body";
    case KFLN_STMT_WHILE:    return "a `while` body";
    case KFLN_STMT_FOR_EACH: return "a `for_each` body";
    default:                 return outer;
    }
}

/* Writes the name of the position that expression slot `slot` of `s`
 * occupies, completing the sentence "<position> must be side-effect
 * free". Slot 0 is KflcNode::expr and slot 1 is KflcNode::expr2, whose
 * meanings differ by statement kind. `inside` names the enclosing
 * block, or is NULL at the top of the body. */
static void rl_step_position_(char *buf, size_t n, const KflcNode *s,
                              int slot, const char *inside)
{
    const char *nm = s->name ? s->name : "?";
    char base[128];

    switch (s->kind) {
    case KFLN_STMT_LET:
    case KFLN_STMT_CONST:
        snprintf(base, sizeof base, "the initialiser of `%s`", nm);
        break;
    case KFLN_STMT_ASSIGN:
        snprintf(base, sizeof base, "the assignment to `%s`", nm);
        break;
    case KFLN_STMT_INDEX_ASSIGN:
        if (slot == 1) snprintf(base, sizeof base, "an index expression");
        else snprintf(base, sizeof base, "the assignment to `%s`", nm);
        break;
    case KFLN_STMT_LVALUE_ASSIGN:
        if (slot == 1) snprintf(base, sizeof base, "an assigned expression");
        else snprintf(base, sizeof base, "an assignment target");
        break;
    case KFLN_STMT_EXPR:
        snprintf(base, sizeof base, "an expression statement");
        break;
    case KFLN_STMT_IF:
        snprintf(base, sizeof base, "an `if` condition");
        break;
    case KFLN_STMT_WHILE:
        snprintf(base, sizeof base, "a `while` condition");
        break;
    case KFLN_STMT_RETURN:
        snprintf(base, sizeof base, "a returned expression");
        break;
    default:
        snprintf(base, sizeof base, "an expression");
        break;
    }

    if (inside) snprintf(buf, n, "on_step: %s in %s", base, inside);
    else        snprintf(buf, n, "on_step: %s", base);
}

/* Walks the block, refusing the first reach it finds. Each statement
 * is judged as a statement, then both of its expression slots are
 * judged with the name of the position each occupies, and nested
 * blocks are walked with the name of the block they sit in.
 *
 * Arguments need no case of their own: the sweep this calls descends
 * through a call's arguments before it judges the call, so an impure
 * builtin passed to a pure one is found wherever the outer call sits.
 * Attributes need none either: no statement kind the stepping path
 * admits carries an attribute expression, and the closed admissible
 * list above is what keeps that true, since a kind that is not
 * admitted is refused before its attributes are reached. */
static int rl_reject_impure_block_(const KflcNode *form,
                                   const KflcNode *stmts,
                                   const char *inside, KflcDiag *diag)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        char pos[RL_STEP_POS_MAX];
        if (rl_reject_stmt_(s, inside, diag)) return 1;
        if (s->expr) {
            rl_step_position_(pos, sizeof pos, s, 0, inside);
            if (rl_reject_impure_(form, s->expr, pos, s->line, diag)) return 1;
        }
        if (s->expr2) {
            rl_step_position_(pos, sizeof pos, s, 1, inside);
            if (rl_reject_impure_(form, s->expr2, pos, s->line, diag)) {
                return 1;
            }
        }
        const char *body = rl_step_block_(s, inside);
        if (rl_reject_impure_block_(form, s->children, body, diag)) return 1;
        if (s->else_children) {
            const char *els = (s->kind == KFLN_STMT_IF) ? "an `else` body"
                                                        : body;
            if (rl_reject_impure_block_(form, s->else_children, els, diag)) {
                return 1;
            }
        }
    }
    return 0;
}

/* ---- Body state inside on_step -------------------------------------- */

/* Split a name at its single dot. Returns 1 when exactly one dot sits
 * between two non-empty parts. */
static int rl_dotted_split_(const char *name, char *lhs, size_t lcap,
                            char *rhs, size_t rcap)
{
    if (!name) return 0;
    const char *dot = strchr(name, '.');
    if (!dot || dot == name || dot[1] == '\0') return 0;
    if (strchr(dot + 1, '.')) return 0;
    size_t ln = (size_t)(dot - name);
    if (ln + 1 > lcap || strlen(dot + 1) + 1 > rcap) return 0;
    memcpy(lhs, name, ln);
    lhs[ln] = '\0';
    snprintf(rhs, rcap, "%s", dot + 1);
    return 1;
}

static int rl_state_key_index_(const char *k)
{
    return kflc_body_state_key_index(k);
}

/* Record a (body, key) reference, one slot per pair, in first-mention
 * order. Returns the slot or -2 when the model's table is full. */
/* Split a three-part dotted name. Returns 1 when the name has
 * exactly three identifier parts. The two-part splitter beside this
 * one refuses those, so the two resolvers never see each other's
 * shapes. */
static int rl_dotted_split3_(const char *name, char *a, size_t acap,
                             char *b, size_t bcap, char *c, size_t ccap)
{
    if (!name) return 0;
    const char *d1 = strchr(name, '.');
    if (!d1 || d1 == name) return 0;
    const char *d2 = strchr(d1 + 1, '.');
    if (!d2 || d2 == d1 + 1 || d2[1] == '\0') return 0;
    if (strchr(d2 + 1, '.')) return 0;
    size_t la = (size_t)(d1 - name), lb = (size_t)(d2 - d1 - 1);
    if (la + 1 > acap || lb + 1 > bcap || strlen(d2 + 1) + 1 > ccap) return 0;
    memcpy(a, name, la); a[la] = '\0';
    memcpy(b, d1 + 1, lb); b[lb] = '\0';
    snprintf(c, ccap, "%s", d2 + 1);
    return 1;
}

/* Resolve `<body>.<component>.<field>` to an actuator slot. Returns
 * the slot, -1 when the name is not a three-part one, and -2 when it
 * is but is refused, with the diagnostic already raised. The refusals
 * mirror the body-state form's: an unknown body, a body with no
 * assembly, an unknown component, an unknown field for that kind, and
 * a write to a read-only field. */
static int rl_act_resolve_(RlModel *m, const char *name, int write,
                           int line, KflcDiag *diag)
{
    char bn[128], cn[KFLC_ASM_NAME_MAX], fn[64];
    if (!rl_dotted_split3_(name, bn, sizeof bn, cn, sizeof cn,
                           fn, sizeof fn)) {
        return -1;
    }
    int bi = rl_body_index_of_(m, bn);
    if (bi < 0) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: no astro_body named `%s` is declared in this "
            "world", name, bn);
        return -2;
    }
    if (!rl_body_has_assembly_(m, bi)) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: `%s` declares no `assembly=`, so it has no "
            "components to command", name, bn);
        return -2;
    }
    int kind = -1, idx = -1;
    for (int i = 0; i < m->n_wheels; i++) {
        if (m->wheels[i].body == bi && strcmp(m->wheels[i].name, cn) == 0) {
            kind = 0; idx = i;
        }
    }
    for (int i = 0; i < m->n_torquers; i++) {
        if (m->torquers[i].body == bi &&
            strcmp(m->torquers[i].name, cn) == 0) { kind = 1; idx = i; }
    }
    for (int i = 0; i < m->n_thrusters; i++) {
        if (m->thrusters[i].body == bi &&
            strcmp(m->thrusters[i].name, cn) == 0) { kind = 2; idx = i; }
    }
    if (kind < 0) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: `%s` has no wheel, magnetorquer or thruster "
            "named `%s`; the commandable names are the assembly's own",
            name, bn, cn);
        return -2;
    }

    int field = -1, readonly = 0;
    if (kind == 0) {
        if (strcmp(fn, "torque") == 0)        field = 0;
        else if (strcmp(fn, "momentum") == 0) { field = 1; readonly = 1; }
        else if (strcmp(fn, "rate") == 0)     { field = 2; readonly = 1; }
        else {
            kflc_diag_errorf(diag, line,
                "on_step: `%s`: a wheel takes `torque` and reads "
                "`momentum` and `rate`, not `%s`", name, fn);
            return -2;
        }
    } else if (kind == 1) {
        if (strcmp(fn, "dipole") == 0) field = 0;
        else {
            kflc_diag_errorf(diag, line,
                "on_step: `%s`: a magnetorquer takes `dipole`, not `%s`",
                name, fn);
            return -2;
        }
    } else {
        if (strcmp(fn, "throttle") == 0) field = 0;
        else {
            kflc_diag_errorf(diag, line,
                "on_step: `%s`: a thruster takes `throttle`, not `%s`",
                name, fn);
            return -2;
        }
    }
    if (write && readonly) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: `%s` is a reading, not a command", name, fn);
        return -2;
    }

    for (int i = 0; i < m->n_acts; i++) {
        if (m->acts[i].kind == kind && m->acts[i].index == idx &&
            m->acts[i].field == field) {
            if (write) m->acts[i].written = 1;
            return i;
        }
    }
    if (m->n_acts >= RL_MAX_ACT) {
        kflc_diag_errorf(diag, line,
            "on_step: more than %d distinct actuator references",
            RL_MAX_ACT);
        return -2;
    }
    m->acts[m->n_acts].kind    = kind;
    m->acts[m->n_acts].index   = idx;
    m->acts[m->n_acts].field   = field;
    m->acts[m->n_acts].written = write;
    return m->n_acts++;
}

static int rl_bs_slot_(RlModel *m, int body, int key, int write, int line,
                       KflcDiag *diag)
{
    const char *kn = kflc_body_state_key_name(key);
    if (kn && kflc_body_state_is_attitude(kn) &&
        !rl_body_has_assembly_(m, body)) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s.%s` is attitude state, but `%s` declares no "
            "`assembly=`, so it has no inertia tensor and its attitude is "
            "never advanced; bind an assembly or drop the attitude keys",
            m->bodies[body].body->name, kn, m->bodies[body].body->name);
        return -2;
    }
    for (int i = 0; i < m->n_bs; i++) {
        if (m->bs[i].body == body && m->bs[i].key == key) {
            if (write) m->bs[i].written = 1;
            return i;
        }
    }
    if (m->n_bs >= RL_MAX_BS) {
        kflc_diag_errorf(diag, line,
            "on_step: more than %d distinct body state references",
            RL_MAX_BS);
        return -2;
    }
    m->bs[m->n_bs].body    = body;
    m->bs[m->n_bs].key     = key;
    m->bs[m->n_bs].written = write;
    return m->n_bs++;
}

static void rl_bs_fn_name_(const RlModel *m, int slot, int set,
                           char *out, size_t cap)
{
    const RlStateRef *r = &m->bs[slot];
    snprintf(out, cap, "kflrl_bs_%s_%s_%s", set ? "set" : "get",
             m->bodies[r->body].body->name,
             kflc_body_state_key_name(r->key));
}

/* Resolve a dotted name used inside on_step. Returns the slot, -1 when
 * the name carries no dot and is somebody else's to resolve, or -2 when
 * it is dotted and refused with a diagnostic. */
static int rl_bs_resolve_(RlModel *m, const char *name, int write,
                          int line, KflcDiag *diag)
{
    if (!name || !strchr(name, '.')) return -1;
    if (strcmp(name, "episode.steps") == 0) {
        kflc_diag_errorf(diag, line,
            "on_step: `episode.steps` is readable in the objective and "
            "termination expressions, not in on_step");
        return -2;
    }
    char b[128], k[64];
    if (!rl_dotted_split_(name, b, sizeof b, k, sizeof k)) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s` is not a body state reference (expected "
            "`<body>.<key>`)", name);
        return -2;
    }
    int bi = rl_body_index_of_(m, b);
    if (bi < 0) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: no astro_body named `%s` is declared in this "
            "world", name, b);
        return -2;
    }
    int ki = rl_state_key_index_(k);
    if (ki < 0) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: `%s` is not a body state key (expected "
            "pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, "
            "quat_y, quat_z, omega_x, omega_y, or omega_z)", name, k);
        return -2;
    }
    return rl_bs_slot_(m, bi, ki, write, line, diag);
}

/* Rewrite dotted reads in an expression into calls on the emitted
 * state accessors. The identifier text becomes the call itself, which
 * the expression emitter passes through for a double binding, so the
 * read is live: it sees writes made earlier in the same body. */
static int rl_bs_rewrite_expr_(RlModel *m, KflcExpr *e, int line,
                               KflcArena *arena, KflcDiag *diag)
{
    if (!e) return 0;
    switch (e->kind) {
    case KFLE_IDENT: {
        int aslot = rl_act_resolve_(m, e->u.ident, 0, line, diag);
        if (aslot == -2) return 1;
        if (aslot >= 0) {
            char call[128];
            snprintf(call, sizeof call, "kflrl_act_get_%d(_kfl_a)", aslot);
            e->u.ident = kflc_arena_strdup(arena, call);
            return 0;
        }
        int slot = rl_bs_resolve_(m, e->u.ident, 0, line, diag);
        if (slot == -1) return 0;
        if (slot < 0) return 1;
        char fn[192], call[256];
        rl_bs_fn_name_(m, slot, 0, fn, sizeof fn);
        snprintf(call, sizeof call, "%s(world)", fn);
        e->u.ident = kflc_arena_strdup(arena, call);
        return 0;
    }
    case KFLE_UNARY:
        return rl_bs_rewrite_expr_(m, e->u.un.operand, line, arena, diag);
    case KFLE_BINARY:
        return rl_bs_rewrite_expr_(m, e->u.bin.lhs, line, arena, diag) ||
               rl_bs_rewrite_expr_(m, e->u.bin.rhs, line, arena, diag);
    case KFLE_INDEX:
        return rl_bs_rewrite_expr_(m, e->u.index.base, line, arena, diag) ||
               rl_bs_rewrite_expr_(m, e->u.index.idx, line, arena, diag);
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            if (rl_bs_rewrite_expr_(m, e->u.vec.elems[i], line, arena,
                                    diag)) return 1;
        }
        return 0;
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            if (rl_bs_rewrite_expr_(m, e->u.call.args[i], line, arena,
                                    diag)) return 1;
        }
        return 0;
    default:
        return 0;
    }
}

/* Rewrite one statement list: dotted reads become accessor calls, and
 * an assignment to a dotted name becomes an expression statement
 * calling the state setter. Nested blocks are rewritten too, so the
 * form works wherever an ordinary assignment does. */
static int rl_bs_rewrite_stmts_(RlModel *m, KflcNode *stmts,
                                KflcArena *arena, KflcDiag *diag)
{
    for (KflcNode *s = stmts; s; s = s->next) {
        if (s->kind == KFLN_STMT_ASSIGN && s->name &&
            strchr(s->name, '.')) {
            int aslot = rl_act_resolve_(m, s->name, 1, s->line, diag);
            if (aslot == -2) return 1;
            if (aslot >= 0) {
                if (rl_bs_rewrite_expr_(m, s->expr, s->line, arena, diag)) {
                    return 1;
                }
                char fn[64];
                snprintf(fn, sizeof fn, "kflrl_act_set_%d", aslot);
                KflcExpr *blk = (KflcExpr *)kflc_arena_alloc(arena,
                                                             sizeof *blk);
                memset(blk, 0, sizeof *blk);
                blk->kind    = KFLE_IDENT;
                blk->line    = s->line;
                blk->u.ident = kflc_arena_strdup(arena, "_kfl_a");
                KflcExpr **args = (KflcExpr **)kflc_arena_alloc(
                    arena, 2 * sizeof *args);
                args[0] = blk;
                args[1] = s->expr;
                KflcExpr *call = (KflcExpr *)kflc_arena_alloc(arena,
                                                              sizeof *call);
                memset(call, 0, sizeof *call);
                call->kind          = KFLE_CALL;
                call->line          = s->line;
                call->u.call.name   = kflc_arena_strdup(arena, fn);
                call->u.call.args   = args;
                call->u.call.n_args = 2;
                s->kind = KFLN_STMT_EXPR;
                s->name = NULL;
                s->expr = call;
                continue;
            }
            int slot = rl_bs_resolve_(m, s->name, 1, s->line, diag);
            if (slot < 0) return 1;
            if (rl_bs_rewrite_expr_(m, s->expr, s->line, arena, diag)) {
                return 1;
            }
            char fn[192];
            rl_bs_fn_name_(m, slot, 1, fn, sizeof fn);
            KflcExpr *world = (KflcExpr *)kflc_arena_alloc(arena,
                                                           sizeof *world);
            memset(world, 0, sizeof *world);
            world->kind    = KFLE_IDENT;
            world->line    = s->line;
            world->u.ident = kflc_arena_strdup(arena, "world");
            KflcExpr **args = (KflcExpr **)kflc_arena_alloc(
                arena, 2 * sizeof *args);
            args[0] = world;
            args[1] = s->expr;
            KflcExpr *call = (KflcExpr *)kflc_arena_alloc(arena,
                                                          sizeof *call);
            memset(call, 0, sizeof *call);
            call->kind           = KFLE_CALL;
            call->line           = s->line;
            call->u.call.name    = kflc_arena_strdup(arena, fn);
            call->u.call.args    = args;
            call->u.call.n_args  = 2;

            s->kind = KFLN_STMT_EXPR;
            s->name = NULL;
            s->expr = call;
            continue;
        }
        if (rl_bs_rewrite_expr_(m, s->expr, s->line, arena, diag)) return 1;
        if (rl_bs_rewrite_expr_(m, s->expr2, s->line, arena, diag)) return 1;
        if (rl_bs_rewrite_stmts_(m, s->children, arena, diag)) {
            return 1;
        }
        if (rl_bs_rewrite_stmts_(m, s->else_children, arena, diag)) {
            return 1;
        }
    }
    return 0;
}

/* The state accessors, one pair per referenced (body, key). A key
 * means metres or metres per second from the world origin wherever it
 * is written, so the setter emits the same write the reset path emits,
 * sector fold included, and the getter is its exact inverse. */
static void rl_emit_state_accessors_(FILE *out, const RlModel *m)
{
    for (int i = 0; i < m->n_bs; i++) {
        const RlStateRef *r = &m->bs[i];
        const char *key = kflc_body_state_key_name(r->key);
        char fn[192];

        rl_bs_fn_name_(m, i, 0, fn, sizeof fn);
        fprintf(out,
            "static double %s(K26AstroWorld *world)\n"
            "{\n"
            "    K26AstroBody *b = k26astro_world_body_at(world, "
            "kflrl_body_idx_[%d]);\n"
            "    if (!b) return 0.0;\n", fn, r->body);
        kflc_emit_body_state_read(out, "b->", key);
        fputs("}\n\n", out);

        if (!r->written) continue;
        rl_bs_fn_name_(m, i, 1, fn, sizeof fn);
        fprintf(out,
            "static void %s(K26AstroWorld *world, double v)\n"
            "{\n"
            "    K26AstroBody *b = k26astro_world_body_at(world, "
            "kflrl_body_idx_[%d]);\n"
            "    if (!b) return;\n", fn, r->body);
        rl_emit_body_write_(out, 4, "b->", key, "v");
        fputs("}\n\n", out);
    }
}

/* The on_step body: action names in scope as read-only scalars, body
 * state readable and assignable by dotted name, run once per external
 * step before the world advances, identically in both modes. */
static int rl_emit_on_step_(FILE *out, RlModel *m,
                            const KflcNode *form, KflcArena *arena,
                            KflcExprFn *user_fn_arr, int n_user_fns,
                            KflcDiag *diag)
{
    KflcExprFn *fns = user_fn_arr;
    int n_fns = n_user_fns;

    if (m->on_step) {
        /* Before the rewrite below, so that the positions the
         * diagnostic names are the ones the source wrote rather than
         * the accessor calls they lower to: after it a body state
         * assignment is an expression statement calling an accessor,
         * and the diagnostic would name that instead of the source. */
        if (rl_reject_impure_block_(form, m->on_step->children, NULL,
                                    diag)) {
            return 1;
        }
        if (rl_bs_rewrite_stmts_(m, m->on_step->children, arena, diag)) {
            return 1;
        }
        rl_emit_state_accessors_(out, m);
        /* After the rewrite above, which is what populates the
         * actuator reference table this emits accessors for. */
        rl_emit_actuators_(out, m);

        /* The accessors resolve as ordinary calls in this body alone:
         * they are emitted in this translation unit and named only
         * here. */
        if (m->n_bs > 0 || m->n_acts > 0) {
            n_fns = n_user_fns + 2 * m->n_bs + m->n_acts;
            fns = (KflcExprFn *)kflc_arena_alloc(
                arena, (size_t)n_fns * sizeof *fns);
            for (int i = 0; i < n_user_fns; i++) fns[i] = user_fn_arr[i];
            int at = n_user_fns;
            for (int i = 0; i < m->n_bs; i++) {
                char nm[192];
                rl_bs_fn_name_(m, i, 0, nm, sizeof nm);
                fns[at].name  = kflc_arena_strdup(arena, nm);
                fns[at].arity = 1;
                at++;
                rl_bs_fn_name_(m, i, 1, nm, sizeof nm);
                fns[at].name  = kflc_arena_strdup(arena, nm);
                fns[at].arity = 2;
                at++;
            }
            /* The actuator accessors resolve the same way: emitted in
             * this translation unit, named only here, and reachable
             * only from the block that was rewritten to call them. */
            for (int i = 0; i < m->n_acts; i++) {
                char nm[64];
                if (m->acts[i].field == 0) {
                    snprintf(nm, sizeof nm, "kflrl_act_set_%d", i);
                    fns[at].arity = 2;
                } else {
                    snprintf(nm, sizeof nm, "kflrl_act_get_%d", i);
                    fns[at].arity = 1;
                }
                fns[at].name = kflc_arena_strdup(arena, nm);
                at++;
            }
            n_fns = at;
        }
    } else {
        /* No on_step body, so nothing was rewritten and no actuator
         * is referenced; the block is still emitted because the
         * handle carries it and the signature below takes it either
         * way. It comes out with the counts at zero and costs one
         * unused pointer, which is cheaper than making every use of
         * the type conditional on the declaration. */
        rl_emit_actuators_(out, m);
    }

    fputs("static void kflrl_on_step_(K26AstroWorld *world, "
          "const double *_kfl_act_v,\n"
          "                           KflrlAct *_kfl_a)\n"
          "{\n"
          "    (void)world; (void)_kfl_act_v; (void)_kfl_a;\n", out);
    if (m->on_step) {
        for (int i = 0; i < m->n_actions; i++) {
            fprintf(out,
                "    const double %s = _kfl_act_v ? _kfl_act_v[%d] : 0.0;"
                " (void)%s;\n",
                m->actions[i]->name, i, m->actions[i]->name);
        }

        KflcExprBinding *live = NULL;
        int live_n = 0, live_cap = 0;
        rl_collect_form_args_(form, arena, &live, &live_n, &live_cap);
        for (int i = 0; i < m->n_actions; i++) {
            rl_push_binding_(arena, &live, &live_n, &live_cap,
                             m->actions[i]->name, KFLT_DOUBLE);
        }
        for (const KflcNode *s = m->on_step->children; s; s = s->next) {
            rl_collect_lets_(s, arena, &live, &live_n, &live_cap);
        }
        for (int i = 0; i < m->n_bs; i++) {
            char nm[192], call[256];
            rl_bs_fn_name_(m, i, 0, nm, sizeof nm);
            snprintf(call, sizeof call, "%s(world)", nm);
            rl_push_binding_(arena, &live, &live_n, &live_cap,
                             kflc_arena_strdup(arena, call), KFLT_DOUBLE);
        }
        /* An actuator reading resolves the same way a body state
         * reading does: the rewrite turned the dotted name into the
         * accessor call's text, and that text is what the expression
         * scope knows as a double. */
        for (int i = 0; i < m->n_acts; i++) {
            if (m->acts[i].field == 0) continue;
            char call[64];
            snprintf(call, sizeof call, "kflrl_act_get_%d(_kfl_a)", i);
            rl_push_binding_(arena, &live, &live_n, &live_cap,
                             kflc_arena_strdup(arena, call), KFLT_DOUBLE);
        }
        rl_push_binding_(arena, &live, &live_n, &live_cap, "world",
                         KFLT_OPAQUE);
        live[live_n - 1].type_subtype = "world";
        /* The environment's actuator block, which the rewritten
         * commands and readings are called with. Like `world` it is a
         * name the block never sees in source: a program writes
         * `<body>.<component>.<field>` and the rewrite supplies this. */
        rl_push_binding_(arena, &live, &live_n, &live_cap, "_kfl_a",
                         KFLT_OPAQUE);
        live[live_n - 1].type_subtype = "world";

        KflcExprCtx ctx;
        memset(&ctx, 0, sizeof ctx);
        ctx.bindings   = live;
        ctx.n_bindings = live_n;
        ctx.fns        = fns;
        ctx.n_fns      = n_fns;
        ctx.form       = form;
        ctx.headless   = 1;

        kfl_emit_stmt_reset_scopes(arena, KFLT_VOID);
        for (const KflcNode *s = m->on_step->children; s; s = s->next) {
            (void)kfl_emit_stmt(out, s, &ctx, diag, 4);
        }
        kfl_emit_stmt_drain_root(out, 4);
    }
    fputs("}\n\n", out);
    return diag->errors ? 1 : 0;
}

/* Names an objective or termination expression may read that the
 * emitted scope resolves. A world binding the capture pass skipped
 * (nested in a block, or not scalar) gets a precise diagnostic here
 * instead of an unknown-identifier failure downstream. Every other
 * unknown name is left to the expression emitter's own resolution
 * (builtins, user fns). */
static void rl_check_objective_names_(const RlModel *m,
                                      const KflcNode *form,
                                      const KflcExpr *e,
                                      const char *ctx_word, int line,
                                      KflcDiag *diag)
{
    if (!e) return;
    switch (e->kind) {
    case KFLE_IDENT: {
        const char *id = e->u.ident;
        if (!id) return;
        if (strcmp(id, "episode.steps") == 0 ||
            strcmp(id, "true") == 0 || strcmp(id, "false") == 0) {
            return;
        }
        if (strchr(id, '.')) {
            /* Body state is addressed by dotted name inside on_step
             * and nowhere else; state reaches an objective through
             * observation channels. */
            kflc_diag_errorf(diag, line,
                "%s: `%s`: body state is readable and assignable only "
                "inside an on_step block; an objective reads state "
                "through `observe ... as` channels", ctx_word, id);
            return;
        }
        if (rl_scope_name_taken_(m, id)) return;   /* action / channel */
        for (int i = 0; i < m->n_wscal; i++) {
            const char *wn = m->wscal[i]->name;
            if (wn && strcmp(wn, id) == 0) return;
        }
        for (const KflcNode *c = form->children; c; c = c->next) {
            if (c->kind == KFLN_ARG && c->name &&
                strcmp(c->name, id) == 0) {
                return;
            }
        }
        int top = 0;
        const KflcNode *b =
            rl_find_world_binding_(m->world->children, id, 0, &top);
        if (b && !top) {
            kflc_diag_errorf(diag, line,
                "%s: `%s` is declared inside a nested block of the "
                "world body and is not in scope here; declare it at "
                "the top level of `fn world` to read it", ctx_word, id);
        } else if (b && !rl_is_scalar_type_(b->type)) {
            kflc_diag_errorf(diag, line,
                "%s: `%s` is not a scalar; only double, int, and bool "
                "world bindings are readable here", ctx_word, id);
        }
        return;
    }
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            rl_check_objective_names_(m, form, e->u.call.args[i],
                                      ctx_word, line, diag);
        }
        return;
    case KFLE_UNARY:
        rl_check_objective_names_(m, form, e->u.un.operand, ctx_word,
                                  line, diag);
        return;
    case KFLE_BINARY:
        rl_check_objective_names_(m, form, e->u.bin.lhs, ctx_word,
                                  line, diag);
        rl_check_objective_names_(m, form, e->u.bin.rhs, ctx_word,
                                  line, diag);
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            rl_check_objective_names_(m, form, e->u.vec.elems[i],
                                      ctx_word, line, diag);
        }
        return;
    case KFLE_INDEX:
        rl_check_objective_names_(m, form, e->u.index.base, ctx_word,
                                  line, diag);
        rl_check_objective_names_(m, form, e->u.index.idx, ctx_word,
                                  line, diag);
        return;
    default:
        return;
    }
}

/* Reward, terminal adjustment, and termination predicate. Absent
 * blocks give the documented defaults: an all-zero reward stream, no
 * terminal adjustment, no predicate termination. */
static int rl_emit_objective_(FILE *out, const RlModel *m,
                              const KflcNode *form, KflcArena *arena,
                              KflcExprFn *user_fn_arr, int n_user_fns,
                              KflcDiag *diag)
{
    KflcExprBinding *live = NULL;
    int live_n = 0, live_cap = 0;
    rl_scope_bindings_(m, form, arena, &live, &live_n, &live_cap);
    KflcExprCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.bindings   = live;
    ctx.n_bindings = live_n;
    ctx.fns        = user_fn_arr;
    ctx.n_fns      = n_user_fns;
    ctx.form       = form;

    struct {
        const char *name;
        const char *ret;
        const KflcAttr *attr;
        const char *absent;
        const char *word;
    } fns[3] = {
        { "kflrl_reward_",     "double", m->reward,          "0.0",
          "reward" },
        { "kflrl_terminal_",   "double", m->terminal,        "0.0",
          "terminal" },
        { "kflrl_terminated_", "int",    m->terminated_when, "0",
          "terminated when" },
    };
    for (int i = 0; i < 3; i++) {
        fprintf(out,
            "static %s %s(const double *_kfl_obs_v,\n"
            "        const double *_kfl_act_v, uint32_t _kfl_nsteps,\n"
            "        const double *_kfl_world_v)\n"
            "{\n"
            "    (void)_kfl_obs_v; (void)_kfl_act_v; (void)_kfl_nsteps; "
            "(void)_kfl_world_v;\n",
            fns[i].ret, fns[i].name);
        if (fns[i].attr && fns[i].attr->expr) {
            rl_check_objective_names_(m, form, fns[i].attr->expr,
                                      fns[i].word, fns[i].attr->line,
                                      diag);
            if (diag->errors) return 1;
            char what[80];
            snprintf(what, sizeof what, "the `%s` expression",
                     fns[i].word);
            if (rl_reject_impure_(form, fns[i].attr->expr, what,
                                  fns[i].attr->line, diag)) {
                return 1;
            }
            rl_emit_scope_prelude_(out, m, 4);
            rl_rewrite_steps_(fns[i].attr->expr, arena);
            if (strcmp(fns[i].ret, "int") == 0) {
                fputs("    return (", out);
                if (kflc_emit_expr(out, fns[i].attr->expr, &ctx, diag)) {
                    return 1;
                }
                fputs(") ? 1 : 0;\n", out);
            } else {
                fputs("    return (double)(", out);
                if (kflc_emit_expr(out, fns[i].attr->expr, &ctx, diag)) {
                    return 1;
                }
                fputs(");\n", out);
            }
        } else {
            fprintf(out, "    return %s;\n", fns[i].absent);
        }
        fputs("}\n\n", out);
    }
    return 0;
}

/* The environment handle and the frozen surface. This block is
 * program-independent apart from the geometry macros, so it is
 * emitted as one literal. */
static void rl_emit_env_core_(FILE *out)
{
    fputs(
"/* ---- Environment handle ------------------------------------------ */\n"
"\n"
"/* The context a thrust perturbation is registered with: the handle\n"
" * and which environment it speaks for. Both are fixed at create. */\n"
"typedef struct KflrlThrustCtx {\n"
"    struct K26RlEnv *h;\n"
"    uint32_t         e;\n"
"} KflrlThrustCtx;\n"
"\n"
"struct K26RlEnv {\n"
"    uint32_t magic;\n"
"    uint32_t n_envs;\n"
"    uint64_t seed;\n"
"    K26RngKey key;\n"
"    uint32_t rekey_ordinal;\n"
"    uint64_t *seen_seeds;\n"
"    uint32_t n_seen, cap_seen;\n"
"    K26AstroWorld **worlds;\n"
"    /* One vehicle slot per assembly-bearing body per environment,\n"
"     * allocated at create and destroyed at destroy. The world holds\n"
"     * a non-owning pointer to each, so the handle owns them: this is\n"
"     * the one place that can, since it is the one thing that outlives\n"
"     * a world and knows when the world goes. */\n"
"    K26AstroVehicle **vehicles;\n"
"    /* One actuator block per environment: commands written by\n"
"     * on_step and the wheel momenta that persist between them. */\n"
"    KflrlAct *act;\n"
"    /* One latched contact per collidable body per environment. A\n"
"     * contact is a fact about a transition, so it is cleared at the\n"
"     * start of each one and the first contact in the transition is\n"
"     * the one the channels report. The block exists whether or not\n"
"     * the program declares a collider, because the observation\n"
"     * function reads it either way and one signature is cheaper than\n"
"     * a conditional one. */\n"
"    KflrlContact *contact;\n"
"    KflrlJoin *join;             /* one per environment */\n"
"    struct KflrlThrustCtx *thrust_ctx;\n"
"    K26AstroBody *baseline;      /* n_envs * KFLRL_N_BODIES */\n"
"    K26AstroEpoch *baseline_t;   /* n_envs */\n"
"    uint32_t *episode;\n"
"    uint32_t *steps;             /* transitions in the current episode */\n"
"    uint8_t  *ended;\n"
"    double   *obs;               /* n_envs * KFLRL_OBS_TOTAL */\n"
"    double   *rew;               /* n_envs (agent count 1) */\n"
"    uint32_t *flags;\n"
"    uint16_t *fault;\n"
"    double   *dr_vals;           /* n_envs * KFLRL_N_REC */\n"
"    double   *wscal;             /* n_envs * KFLRL_N_WSCAL, world\n"
"                                  * scalars captured at create */\n"
"    double   *scratch;           /* KFLRL_OBS_TOTAL */\n"
#if 1
"#if KFLRL_N_SENSED > 0\n"
"    /* The imperfection layer. `sterm` is the resolved model\n"
"     * chain, built once at create because the bias walk's two\n"
"     * coefficients follow from the control period; `sense` is\n"
"     * one state block per sensed channel per environment, and\n"
"     * `sense_ring` is the delay storage those blocks point\n"
"     * into. All three are sized at create, so the step path\n"
"     * allocates nothing. */\n"
"    K26SenseTerm  *sterm;\n"
"    K26SenseState *sense;\n"
"    double        *sense_ring;\n"
"#endif\n"
#endif

"    double    control_dt;\n"
"    uint32_t  horizon;\n"
"    /* The declared subdivision of a control period, and the interval\n"
"     * each sub-advance takes. The last sub-advance of a transition\n"
"     * takes the remainder instead, so the advanced time sums to\n"
"     * control_dt exactly however the division rounded. */\n"
"    uint32_t  substeps;\n"
"    double    restitution;       /* contact bounce, 0 for arrest */\n"
"    double    friction;\n"
"    double    sub_dt;\n", out);
    fputs(
"    double    act_lo[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];\n"
"    double    act_hi[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];\n"
"    uint32_t  act_arity[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];\n"
"    uint16_t  act_kind[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];\n"
"    uint8_t  *spec;\n"
"    uint32_t  spec_len;\n"
"    K26RlEpisodeWriter *writer;\n"
"    K26RlTap *tap;               /* the telemetry ring, when enabled */\n"
"    uint8_t   at_boundary;\n"
"};\n"
"\n"
"static int kflrl_live_(const K26RlEnv *h)\n"
"{\n"
"    return h && h->magic == KFLRL_MAGIC;\n"
"}\n"
"\n"
"/* k26rl_status_str is the episode library's; nothing else in this\n"
" * translation unit calls it, so this anchor makes the link pull the\n"
" * archive member in and the shared object export the full frozen\n"
" * surface. */\n"
"static const void *const kflrl_keep_status_str_\n"
"    __attribute__((used)) = (const void *)&k26rl_status_str;\n"
"\n", out);

    /* The imperfection layer's two entries. Both are emitted whatever
     * the program declares, so the call sites need no conditional; with
     * no sensor they compile to nothing. */
    fputs(
"/* Per-episode setup: the draws a chain takes once, and the state a\n"
" * step expects to find. The value handed in is the channel's true\n"
" * value at the boundary, which fills a delay ring and primes a\n"
" * dropout hold. */\n"
"static void kflrl_sense_reset_(K26RlEnv *h, uint32_t e, uint32_t ep,\n"
"                               const double *obs_v)\n"
"{\n"
"    (void)h; (void)e; (void)ep; (void)obs_v;\n"
"#if KFLRL_N_SENSED > 0\n"
"    for (int c = 0; c < KFLRL_N_SENSED; c++) {\n"
"        (void)k26sense_chain_reset(\n"
"            h->sterm + kflrl_sensed_first_[c],\n"
"            (uint32_t)kflrl_sensed_count_[c],\n"
"            &h->sense[(size_t)e * KFLRL_N_SENSED + (size_t)c],\n"
"            h->key, K26SENSE_CLASS_SENSOR, e, ep,\n"
"            obs_v[kflrl_sensed_slot_[c]]);\n"
"    }\n"
"#endif\n"
"}\n"
"\n"
"/* One transition's corruption, in place over the measured half. The\n"
" * draw index is the transition index within the episode, so any\n"
" * step's noise is addressable without producing the step before it.\n"
" * No allocation, no I/O, and a fixed number of draws per term. */\n"
"static void kflrl_sense_apply_(K26RlEnv *h, uint32_t e, uint32_t ep,\n"
"                               uint32_t step, double *obs_v)\n"
"{\n"
"    (void)h; (void)e; (void)ep; (void)step; (void)obs_v;\n"
"#if KFLRL_N_SENSED > 0\n"
"    for (int c = 0; c < KFLRL_N_SENSED; c++) {\n"
"        double v = obs_v[kflrl_sensed_slot_[c]];\n"
"        (void)k26sense_chain_apply(\n"
"            h->sterm + kflrl_sensed_first_[c],\n"
"            (uint32_t)kflrl_sensed_count_[c],\n"
"            &h->sense[(size_t)e * KFLRL_N_SENSED + (size_t)c],\n"
"            h->key, K26SENSE_CLASS_SENSOR, e, ep, step, v, &v);\n"
"        obs_v[kflrl_sensed_slot_[c]] = v;\n"
"    }\n"
"#endif\n"
"}\n"
"\n", out);

    fputs(
"#if KFLRL_N_PORTS > 1\n"
"/* Form the joint body from a captured pair, at the impact\n"
" * configuration the sweep computed.\n"
" *\n"
" * Mass is summed. Linear momentum is conserved, so the pair leaves\n"
" * at the velocity of their common centre of mass. Angular momentum\n"
" * about that centre is conserved too, which is what decides the\n"
" * joint rate: each body brings its own spin and the moment of its\n"
" * own motion about the joint centre, and the joint inertia turns the\n"
" * total into a rate. The inertia is the sum of the two tensors, each\n"
" * carried to the joint centre by the parallel-axis theorem.\n"
" *\n"
" * Nothing here allocates: the tensors are set in place through the\n"
" * vehicle's own setters, which recompute the inverse where they\n"
" * stand. */\n"
"static K26M3 kflrl_join_world_inertia_(const K26AstroVehicle *v,\n"
"                                       K26Quat q)\n"
"{\n"
"    K26M3 out;\n"
"    memset(&out, 0, sizeof out);\n"
"    const K26AstroAttitudeStateExt *x = v\n"
"        ? k26astro_vehicle_attitude_ext((K26AstroVehicle *)v) : NULL;\n"
"    if (!x) return out;\n"
"    K26V3 e0 = k26m3d_quat_rotate_v3(q, k26m3d_v3(1, 0, 0));\n"
"    K26V3 e1 = k26m3d_quat_rotate_v3(q, k26m3d_v3(0, 1, 0));\n"
"    K26V3 e2 = k26m3d_quat_rotate_v3(q, k26m3d_v3(0, 0, 1));\n"
"    K26M3 R;\n"
"    R.m[0][0] = e0.x; R.m[0][1] = e1.x; R.m[0][2] = e2.x;\n"
"    R.m[1][0] = e0.y; R.m[1][1] = e1.y; R.m[1][2] = e2.y;\n"
"    R.m[2][0] = e0.z; R.m[2][1] = e1.z; R.m[2][2] = e2.z;\n"
"    K26M3 tmp;\n"
"    for (int i = 0; i < 3; i++) {\n"
"        for (int j = 0; j < 3; j++) {\n"
"            double a = 0.0;\n"
"            for (int k = 0; k < 3; k++) a += R.m[i][k] * x->inertia.m[k][j];\n"
"            tmp.m[i][j] = a;\n"
"        }\n"
"    }\n"
"    for (int i = 0; i < 3; i++) {\n"
"        for (int j = 0; j < 3; j++) {\n"
"            double a = 0.0;\n"
"            for (int k = 0; k < 3; k++) a += tmp.m[i][k] * R.m[j][k];\n"
"            out.m[i][j] = a;\n"
"        }\n"
"    }\n"
"    return out;\n"
"}\n\n", out);
    fputs(
"static K26V3 kflrl_m3_mul_(K26M3 m, K26V3 v)\n"
"{\n"
"    return k26m3d_v3(m.m[0][0]*v.x + m.m[0][1]*v.y + m.m[0][2]*v.z,\n"
"                     m.m[1][0]*v.x + m.m[1][1]*v.y + m.m[1][2]*v.z,\n"
"                     m.m[2][0]*v.x + m.m[2][1]*v.y + m.m[2][2]*v.z);\n"
"}\n\n"
"/* Make the joined pair one body again, at the end of a sub-advance\n"
" * in which the two were integrated separately.\n"
" *\n"
" * Configuration first: the follower is placed from the carrier at\n"
" * the offset and relative attitude the capture froze, which is what\n"
" * makes the pair rigid rather than merely close.\n"
" *\n"
" * Motion second, and this is why the pair is projected rather than\n"
" * slaved: both bodies keep their own mass and their own forces, so\n"
" * a thruster on either of them accelerates the pair. Their momenta\n"
" * are summed and the sum is turned back into one rigid motion. Mass\n"
" * is summed; linear momentum gives the joint velocity; angular\n"
" * momentum about the joint centre of mass, each body contributing\n"
" * its own spin and the moment of its own motion, gives the joint\n"
" * rate against the joint inertia, which is the two tensors summed\n"
" * about that centre by the parallel-axis theorem. Both quantities\n"
" * are conserved exactly across the projection; the relative kinetic\n"
" * energy is not, which is what a capture latch does. */\n"
"static void kflrl_join_impose_(K26RlEnv *h, uint32_t e,\n"
"                               const K26AstroPos *cref)\n"
"{\n"
"    KflrlJoin *j = &h->join[e];\n"
"    if (!j->active) return;\n"
"    K26AstroBody *cb = k26astro_world_body_at(h->worlds[e],\n"
"        kflrl_body_idx_[kflrl_vehicle_body_[j->carrier]]);\n"
"    K26AstroBody *fb = k26astro_world_body_at(h->worlds[e],\n"
"        kflrl_body_idx_[kflrl_vehicle_body_[j->follower]]);\n"
"    if (!cb || !fb) return;\n"
"\n"
"    fb->attitude = k26m3d_quat_norm(k26m3d_quat_mul(cb->attitude,\n"
"                                                    j->rel));\n"
"    K26V3 arm  = k26m3d_quat_rotate_v3(cb->attitude, j->offset);\n"
"    K26V3 cpos = k26astro_pos_sub(&cb->pos, cref);\n"
"    K26V3 fpos = k26astro_pos_sub(&fb->pos, cref);\n"
"    k26astro_pos_add(&fb->pos,\n"
"        k26m3d_v3(cpos.x + arm.x - fpos.x, cpos.y + arm.y - fpos.y,\n"
"                  cpos.z + arm.z - fpos.z));\n"
"    fpos = k26m3d_v3(cpos.x + arm.x, cpos.y + arm.y, cpos.z + arm.z);\n"
"", out);
    fputs(
"    K26V3 rc = k26m3d_quat_rotate_v3(cb->attitude,\n"
"        k26m3d_v3(kflrl_veh_com_[j->carrier][0],\n"
"                  kflrl_veh_com_[j->carrier][1],\n"
"                  kflrl_veh_com_[j->carrier][2]));\n"
"    K26V3 rf = k26m3d_quat_rotate_v3(fb->attitude,\n"
"        k26m3d_v3(kflrl_veh_com_[j->follower][0],\n"
"                  kflrl_veh_com_[j->follower][1],\n"
"                  kflrl_veh_com_[j->follower][2]));\n"
"    K26V3 comc = k26m3d_v3(cpos.x + rc.x, cpos.y + rc.y, cpos.z + rc.z);\n"
"    K26V3 comf = k26m3d_v3(fpos.x + rf.x, fpos.y + rf.y, fpos.z + rf.z);\n"
"    double mc = cb->mass > 0.0 ? cb->mass : 0.0;\n"
"    double mf = fb->mass > 0.0 ? fb->mass : 0.0;\n"
"    double M = mc + mf;\n"
"    if (!(M > 0.0)) return;\n"
"    K26V3 R = k26m3d_v3((mc * comc.x + mf * comf.x) / M,\n"
"                        (mc * comc.y + mf * comf.y) / M,\n"
"                        (mc * comc.z + mf * comf.z) / M);\n"
"    K26V3 V = k26m3d_v3((mc * cb->vel.x + mf * fb->vel.x) / M,\n"
"                        (mc * cb->vel.y + mf * fb->vel.y) / M,\n"
"                        (mc * cb->vel.z + mf * fb->vel.z) / M);\n"
"    K26M3 Ic = kflrl_join_world_inertia_(\n"
"        h->vehicles[(size_t)e * KFLRL_N_VEHICLES + j->carrier],\n"
"        cb->attitude);\n"
"    K26M3 If = kflrl_join_world_inertia_(\n"
"        h->vehicles[(size_t)e * KFLRL_N_VEHICLES + j->follower],\n"
"        fb->attitude);\n"
"    K26V3 wc = k26m3d_quat_rotate_v3(cb->attitude, cb->omega);\n"
"    K26V3 wf = k26m3d_quat_rotate_v3(fb->attitude, fb->omega);\n"
"    K26V3 dc = k26m3d_v3(comc.x - R.x, comc.y - R.y, comc.z - R.z);\n"
"    K26V3 df = k26m3d_v3(comf.x - R.x, comf.y - R.y, comf.z - R.z);\n"
"    K26V3 Lc = kflrl_m3_mul_(Ic, wc);\n"
"    K26V3 Lf = kflrl_m3_mul_(If, wf);\n"
"    K26V3 mc_v = k26m3d_v3_cross(dc,\n"
"        k26m3d_v3(mc * (cb->vel.x - V.x), mc * (cb->vel.y - V.y),\n"
"                  mc * (cb->vel.z - V.z)));\n"
"    K26V3 mf_v = k26m3d_v3_cross(df,\n"
"        k26m3d_v3(mf * (fb->vel.x - V.x), mf * (fb->vel.y - V.y),\n"
"                  mf * (fb->vel.z - V.z)));\n"
"    K26V3 L = k26m3d_v3(Lc.x + Lf.x + mc_v.x + mf_v.x,\n"
"                        Lc.y + Lf.y + mc_v.y + mf_v.y,\n"
"                        Lc.z + Lf.z + mc_v.z + mf_v.z);\n"
"", out);
    fputs(
"    K26M3 J;\n"
"    double dcc = k26m3d_v3_dot(dc, dc), dff = k26m3d_v3_dot(df, df);\n"
"    double dcv[3] = { dc.x, dc.y, dc.z }, dfv[3] = { df.x, df.y, df.z };\n"
"    for (int i = 0; i < 3; i++) {\n"
"        for (int k = 0; k < 3; k++) {\n"
"            double kron = (i == k) ? 1.0 : 0.0;\n"
"            J.m[i][k] = Ic.m[i][k] + If.m[i][k]\n"
"                      + mc * (kron * dcc - dcv[i] * dcv[k])\n"
"                      + mf * (kron * dff - dfv[i] * dfv[k]);\n"
"        }\n"
"    }\n"
"    /* The joint rate is that momentum against the joint inertia.\n"
"     * The inverse is written out here because the tensor is the\n"
"     * pair's and no vehicle carries it. A singular one means the\n"
"     * pair has no rotational answer, and the projection is left\n"
"     * undone rather than continued with a fabricated one. */\n"
"    K26M3 Jinv;\n"
"    {\n"
"        double c00 = J.m[1][1]*J.m[2][2] - J.m[1][2]*J.m[2][1];\n"
"        double c01 = J.m[1][2]*J.m[2][0] - J.m[1][0]*J.m[2][2];\n"
"        double c02 = J.m[1][0]*J.m[2][1] - J.m[1][1]*J.m[2][0];\n"
"        double det = J.m[0][0]*c00 + J.m[0][1]*c01 + J.m[0][2]*c02;\n"
"        if (!(det > 0.0) && !(det < 0.0)) return;\n"
"        double id = 1.0 / det;\n"
"        Jinv.m[0][0] = c00 * id;\n"
"        Jinv.m[1][0] = c01 * id;\n"
"        Jinv.m[2][0] = c02 * id;\n"
"        Jinv.m[0][1] = (J.m[0][2]*J.m[2][1] - J.m[0][1]*J.m[2][2]) * id;\n"
"        Jinv.m[1][1] = (J.m[0][0]*J.m[2][2] - J.m[0][2]*J.m[2][0]) * id;\n"
"        Jinv.m[2][1] = (J.m[0][1]*J.m[2][0] - J.m[0][0]*J.m[2][1]) * id;\n"
"        Jinv.m[0][2] = (J.m[0][1]*J.m[1][2] - J.m[0][2]*J.m[1][1]) * id;\n"
"        Jinv.m[1][2] = (J.m[0][2]*J.m[1][0] - J.m[0][0]*J.m[1][2]) * id;\n"
"        Jinv.m[2][2] = (J.m[0][0]*J.m[1][1] - J.m[0][1]*J.m[1][0]) * id;\n"
"    }\n"
"    K26V3 W = kflrl_m3_mul_(Jinv, L);\n"
"    K26V3 vc = k26m3d_v3_cross(W, dc);\n"
"    K26V3 vf = k26m3d_v3_cross(W, df);\n"
"    cb->vel = k26m3d_v3(V.x + vc.x, V.y + vc.y, V.z + vc.z);\n"
"    fb->vel = k26m3d_v3(V.x + vf.x, V.y + vf.y, V.z + vf.z);\n"
"    cb->omega = k26m3d_quat_rotate_v3(k26m3d_quat_conj(cb->attitude), W);\n"
"    fb->omega = k26m3d_quat_rotate_v3(k26m3d_quat_conj(fb->attitude), W);\n"
"}\n\n", out);
    fputs(
"/* Form the join at the impact configuration the sweep computed: put\n"
" * both bodies there, freeze the relative configuration, and project\n"
" * the motion. The carrier is the heavier of the two, since one of\n"
" * them has to hold the pair's pose and the joint centre of mass\n"
" * lies nearer that one; a tie goes to the lower vehicle slot, which\n"
" * is declaration order, so the choice is never an accident. */\n"
"static void kflrl_join_form_(K26RlEnv *h, uint32_t e,\n"
"                             const K26AstroCollBody *cbody,\n"
"                             int va, int vb, double time,\n"
"                             const K26AstroPos *cref)\n"
"{\n"
"    KflrlJoin *j = &h->join[e];\n"
"    if (j->active) return;\n"
"    K26AstroBody *ba = k26astro_world_body_at(h->worlds[e],\n"
"        kflrl_body_idx_[kflrl_vehicle_body_[va]]);\n"
"    K26AstroBody *bb = k26astro_world_body_at(h->worlds[e],\n"
"        kflrl_body_idx_[kflrl_vehicle_body_[vb]]);\n"
"    if (!ba || !bb) return;\n"
"    int carrier = va, follower = vb;\n"
"    if (bb->mass > ba->mass || (bb->mass == ba->mass && vb < va)) {\n"
"        carrier = vb; follower = va;\n"
"    }\n"
"    K26AstroBody *cb = (carrier == va) ? ba : bb;\n"
"    K26AstroBody *fb = (carrier == va) ? bb : ba;\n"
"    const K26AstroCollBody *cc = &cbody[carrier];\n"
"    const K26AstroCollBody *cf = &cbody[follower];\n"
"    K26V3 pc = k26m3d_v3(cc->pos0.x + (cc->pos1.x - cc->pos0.x) * time,\n"
"                         cc->pos0.y + (cc->pos1.y - cc->pos0.y) * time,\n"
"                         cc->pos0.z + (cc->pos1.z - cc->pos0.z) * time);\n"
"    K26V3 pf = k26m3d_v3(cf->pos0.x + (cf->pos1.x - cf->pos0.x) * time,\n"
"                         cf->pos0.y + (cf->pos1.y - cf->pos0.y) * time,\n"
"                         cf->pos0.z + (cf->pos1.z - cf->pos0.z) * time);\n"
"    K26V3 nowc = k26astro_pos_sub(&cb->pos, cref);\n"
"    k26astro_pos_add(&cb->pos, k26m3d_v3(pc.x - nowc.x, pc.y - nowc.y,\n"
"                                         pc.z - nowc.z));\n"
"    K26V3 nowf = k26astro_pos_sub(&fb->pos, cref);\n"
"    k26astro_pos_add(&fb->pos, k26m3d_v3(pf.x - nowf.x, pf.y - nowf.y,\n"
"                                         pf.z - nowf.z));\n"
"    K26Quat qc_conj = k26m3d_quat_conj(cb->attitude);\n"
"    j->carrier  = carrier;\n"
"    j->follower = follower;\n"
"    j->rel      = k26m3d_quat_norm(k26m3d_quat_mul(qc_conj,\n"
"                                                   fb->attitude));\n"
"    j->offset   = k26m3d_quat_rotate_v3(qc_conj,\n"
"        k26m3d_v3(pf.x - pc.x, pf.y - pc.y, pf.z - pc.z));\n"
"    j->active   = 1;\n"
"    kflrl_join_impose_(h, e, cref);\n"
"}\n"
"#endif\n\n", out);
    fputs(
"/* Reset one environment in place to episode `ep`: restore the\n"
" * body-state and epoch baseline captured at create, clear the\n"
" * integrator transients a step leaves behind, re-seed the world's\n"
" * runtime noise stream, apply the episode's draws, and recompute\n"
" * the initial observation. No allocation on this path. */\n"
"static void kflrl_reset_env_(K26RlEnv *h, uint32_t e, uint32_t ep)\n"
"{\n"
"    K26AstroWorld *w = h->worlds[e];\n"
"#if KFLRL_N_VEHICLES > 0\n"
"    /* Actuator state is episode state: a wheel's stored momentum and\n"
"     * every standing command belong to the episode that produced\n"
"     * them, so both are cleared here with the rest of the baseline.\n"
"     * A wheel left spun up across a reset would make episode k+1 a\n"
"     * function of episode k, which is exactly what the identity\n"
"     * triple says it is not. */\n"
"    memset(&h->act[e], 0, sizeof h->act[e]);\n"
"#endif\n"
"    /* The latched contact belongs to the episode that produced it,\n"
"     * so it is cleared here with the rest of the baseline. */\n"
"    memset(&h->contact[(size_t)e * KFLRL_N_CONTACT], 0,\n"
"           sizeof(KflrlContact) * KFLRL_N_CONTACT);\n"
"    /* A pair joined by a capture belongs to its episode too. The\n"
"     * join holds no property of either craft, only the pose that\n"
"     * ties them, so dropping it is the whole of undoing it. */\n"
"    memset(&h->join[e], 0, sizeof h->join[e]);\n"
"#if KFLRL_N_BODIES > 0\n"
"    K26AstroBody *b0 = k26astro_world_body_at(w, 0);\n"
"    if (b0) {\n"
"        memcpy(b0, h->baseline + (size_t)e * KFLRL_N_BODIES,\n"
"               sizeof(K26AstroBody) * KFLRL_N_BODIES);\n"
"    }\n"
"#endif\n"
"    K26AstroGravState *g = k26astro_world_grav(w);\n"
"    if (g) {\n"
"        g->t = h->baseline_t[e];\n"
"        g->dt_last = 0.0;\n"
"        g->ias15_dt_last = 0.0;\n"
"        k26astro_grav_ias15_reset(g);\n"
"    }\n"
"    /* The world's own stateful generator is seeded from a counter\n"
"     * draw at this environment's and episode's coordinates,\n"
"     * rather than from the governing seed. Handing every world\n"
"     * the same seed would correlate any model that drew from\n"
"     * it across the whole vectorised set, and would repeat the\n"
"     * same sequence in every episode. */\n"
"    (void)k26astro_world_set_seed(\n"
"        w, kflrl_world_seed_(h->key, e, ep));\n"
"#if KFLRL_N_REC > 0\n"
"    kflrl_apply_draws_(w, h->key, e, ep,\n"
"                       h->dr_vals + (size_t)e * KFLRL_N_REC, NULL);\n"
"#else\n"
"    kflrl_apply_draws_(w, h->key, e, ep, NULL, NULL);\n"
"#endif\n"
"    h->episode[e] = ep;\n"
"    h->steps[e]   = 0;\n"
"    h->ended[e]   = 0;\n"
"    h->rew[e]     = 0.0;\n"
"    h->fault[e]   = 0;\n"
"    kflrl_observe_(w, h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"                   &h->contact[(size_t)e * KFLRL_N_CONTACT],\n"
"                       &h->join[e]);\n"
"    kflrl_sense_reset_(h, e, ep,\n"
"                       h->obs + (size_t)e * KFLRL_OBS_TOTAL);\n"
"}\n"
"\n"
"", out);
    fputs(
"/* ---- Spec blob ---------------------------------------------------- */\n"
"\n"
"static void kflrl_put_u16_(uint8_t *p, uint16_t v)\n"
"{\n"
"    p[0] = (uint8_t)(v & 0xff);\n"
"    p[1] = (uint8_t)(v >> 8);\n"
"}\n"
"\n"
"static void kflrl_put_u32_(uint8_t *p, uint32_t v)\n"
"{\n"
"    p[0] = (uint8_t)(v & 0xff);\n"
"    p[1] = (uint8_t)((v >> 8) & 0xff);\n"
"    p[2] = (uint8_t)((v >> 16) & 0xff);\n"
"    p[3] = (uint8_t)((v >> 24) & 0xff);\n"
"}\n"
"\n"
"static void kflrl_put_u64_(uint8_t *p, uint64_t v)\n"
"{\n"
"    kflrl_put_u32_(p, (uint32_t)(v & 0xffffffffu));\n"
"    kflrl_put_u32_(p + 4, (uint32_t)(v >> 32));\n"
"}\n"
"\n"
"static uint64_t kflrl_f64_bits_(double d)\n"
"{\n"
"    uint64_t u;\n"
"    memcpy(&u, &d, sizeof u);\n"
"    return u;\n"
"}\n"
"\n"
"/* One TLV entry; p advances past it. NULL p sizes only. */\n"
"static uint32_t kflrl_tlv_(uint8_t **p, uint16_t tag, uint32_t len,\n"
"                           const uint8_t *val)\n"
"{\n"
"    if (*p) {\n"
"        kflrl_put_u16_(*p, tag);\n"
"        kflrl_put_u32_(*p + 2, len);\n"
"        if (len) memcpy(*p + 6, val, len);\n"
"        *p += 6 + len;\n"
"    }\n"
"    return 6 + len;\n"
"}\n"
"\n"
"static uint32_t kflrl_spec_write_(uint8_t *buf, const K26RlEnv *h)\n"
"{\n"
"    uint8_t *p = buf;\n"
"    uint8_t v[20];\n"
"    uint32_t total = 0;\n"
"    kflrl_put_u32_(v, K26RL_ABI_VERSION);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_ABI_VERSION, 4, v);\n"
"    kflrl_put_u32_(v, 0x01020304u);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_ENDIAN_PROBE, 4, v);\n"
"    kflrl_put_u32_(v, 1u);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_AGENT_COUNT, 4, v);\n"
"    kflrl_put_u32_(v, h->n_envs);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_N_ENVS, 4, v);\n"
"    kflrl_put_u64_(v, kflrl_f64_bits_(h->control_dt));\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_CONTROL_DT, 8, v);\n"
"    kflrl_put_u32_(v, h->substeps);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_SUBSTEPS, 4, v);\n"
"    kflrl_put_u32_(v, h->horizon);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_HORIZON, 4, v);\n"
"    kflrl_put_u32_(v, KFLRL_OBS_TOTAL);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_OBS_TOTAL, 4, v);\n"
"    kflrl_put_u32_(v, KFLRL_ACT_TOTAL);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_ACT_TOTAL, 4, v);\n"
"    kflrl_put_u32_(v, 0u);\n"
"    kflrl_put_u32_(v + 4, 0u);\n"
"    kflrl_put_u32_(v + 8, KFLRL_OBS_TOTAL);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_AGENT_OBS_SLICE, 12, v);\n"
"    kflrl_put_u32_(v + 8, KFLRL_ACT_TOTAL);\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_AGENT_ACT_SLICE, 12, v);\n"
"    for (uint32_t i = 0; i < KFLRL_ACT_TOTAL; i++) {\n"
"        kflrl_put_u32_(v, i);\n"
"        kflrl_put_u64_(v + 4, kflrl_f64_bits_(h->act_lo[i]));\n"
"        kflrl_put_u64_(v + 12, kflrl_f64_bits_(h->act_hi[i]));\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_ACT_BOUNDS, 20, v);\n"
"    }\n"
"    for (uint32_t i = 0; i < KFLRL_ACT_TOTAL; i++) {\n"
"        kflrl_put_u32_(v, i);\n"
"        kflrl_put_u16_(v + 4, h->act_kind[i]);\n"
"        if (h->act_kind[i] == K26RL_ACT_KIND_DISCRETE) {\n"
"            kflrl_put_u32_(v + 6, h->act_arity[i]);\n"
"            total += kflrl_tlv_(&p, K26RL_TAG_ACT_KIND, 10, v);\n"
"        } else {\n"
"            total += kflrl_tlv_(&p, K26RL_TAG_ACT_KIND, 6, v);\n"
"        }\n"
"    }\n"
"#if KFLRL_OBS_TOTAL > 0\n"
"    for (uint32_t i = 0; i < KFLRL_OBS_TOTAL; i++) {\n"
"        uint8_t nv[4 + KFLRL_OBS_NAME_MAX];\n"
"        uint32_t nl = (uint32_t)strlen(kflrl_obs_names_[i]);\n"
"        /* The compiler refuses a name that would not fit, so this\n"
"         * cannot truncate; it is the belt on the braces. */\n"
"        if (nl > KFLRL_OBS_NAME_MAX) nl = KFLRL_OBS_NAME_MAX;\n"
"        kflrl_put_u32_(nv, i);\n"
"        memcpy(nv + 4, kflrl_obs_names_[i], nl);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_NAME, 4 + nl, nv);\n"
"    }\n"
"    for (uint32_t i = 0; i < KFLRL_OBS_TOTAL; i++) {\n"
"        kflrl_put_u32_(v, i);\n"
"        kflrl_put_u16_(v + 4, K26RL_OBS_KIND_VECTOR);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_KIND, 6, v);\n"
"    }\n"
"    for (uint32_t i = 0; i < KFLRL_OBS_TOTAL; i++) {\n"
"        kflrl_put_u32_(v, i);\n"
"        kflrl_put_u16_(v + 4, kflrl_obs_modes_[i]);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_MODE, 6, v);\n"
"        kflrl_put_u16_(v + 4, kflrl_obs_source_[i]);\n"
"        kflrl_put_u32_(v + 6, kflrl_obs_pair_[i]);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_SOURCE, 10, v);\n"
"    }\n"
"#endif\n"
"", out);
    fputs(
"#if KFLRL_N_BODIES > 0\n"
"    for (uint32_t i = 0; i < (uint32_t)KFLRL_N_BODIES; i++) {\n"
"        uint8_t nv[4 + 64];\n"
"        uint32_t nl = (uint32_t)strlen(kflrl_body_names_[i]);\n"
"        if (nl > 64) nl = 64;\n"
"        kflrl_put_u32_(nv, i);\n"
"        memcpy(nv + 4, kflrl_body_names_[i], nl);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_BODY_NAME, 4 + nl, nv);\n"
"    }\n"
"#endif\n"
"#if KFLRL_N_ASSEMBLIES > 0\n"
"    for (uint32_t i = 0; i < (uint32_t)KFLRL_N_ASSEMBLIES; i++) {\n"
"        uint8_t dv[4 + 32];\n"
"        kflrl_put_u32_(dv, kflrl_assemblies_[i].body);\n"
"        memcpy(dv + 4, kflrl_assemblies_[i].digest, 32);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_ASSEMBLY_DIGEST, 4 + 32, dv);\n"
"    }\n"
"    for (uint32_t i = 0; i < (uint32_t)KFLRL_N_ASSEMBLIES; i++) {\n"
"        uint8_t nv2[4 + 64];\n"
"        uint32_t nl2 = (uint32_t)strlen(kflrl_assemblies_[i].name);\n"
"        if (nl2 > 64) nl2 = 64;\n"
"        kflrl_put_u32_(nv2, kflrl_assemblies_[i].body);\n"
"        memcpy(nv2 + 4, kflrl_assemblies_[i].name, nl2);\n"
"        total += kflrl_tlv_(&p, K26RL_TAG_ASSEMBLY_NAME, 4 + nl2, nv2);\n"
"    }\n"
"#endif\n"
"    kflrl_put_u32_(v, 1u);   /* bit 0: auto-reset, always on */\n"
"    total += kflrl_tlv_(&p, K26RL_TAG_EPISODE_FLAGS, 4, v);\n"
"    return total;\n"
"}\n"
"\n"
"", out);
    fputs(
"/* ---- Frozen surface ----------------------------------------------- */\n"
"\n"
"extern \"C\" uint32_t k26rl_abi_version(void)\n"
"{\n"
"    return K26RL_ABI_VERSION;\n"
"}\n"
"\n"
"static void kflrl_free_handle_(K26RlEnv *h)\n"
"{\n"
"    if (!h) return;\n"
"#if KFLRL_N_VEHICLES > 0\n"
"    /* Vehicles first: a vehicle's teardown notifies its subsystems\n"
"     * and unregisters from the world, so the world must still be\n"
"     * there when it runs. */\n"
"    if (h->vehicles) {\n"
"        for (uint32_t i = 0; i < h->n_envs * KFLRL_N_VEHICLES; i++) {\n"
"            if (h->vehicles[i]) {\n"
"                uint32_t e = i / KFLRL_N_VEHICLES;\n"
"                if (h->worlds && h->worlds[e]) {\n"
"                    k26astro_world_unregister_vehicle(h->worlds[e],\n"
"                                                      h->vehicles[i]);\n"
"                }\n"
"                k26astro_vehicle_destroy(h->vehicles[i]);\n"
"            }\n"
"        }\n"
"    }\n"
"    free(h->vehicles);\n"
"    free(h->act);\n"
"    free(h->contact);\n"
"    free(h->join);\n"
"    free(h->thrust_ctx);\n"
"#endif\n"
"    if (h->worlds) {\n"
"        for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"            if (h->worlds[e]) k26astro_world_destroy(h->worlds[e]);\n"
"        }\n"
"    }\n"
"    free(h->worlds);\n"
"    free(h->baseline);\n"
"    free(h->baseline_t);\n"
"    free(h->episode);\n"
"    free(h->steps);\n"
"    free(h->ended);\n"
"    free(h->obs);\n"
"    free(h->rew);\n"
"    free(h->flags);\n"
"    free(h->fault);\n"
"    free(h->dr_vals);\n"
"    free(h->wscal);\n"
"#if KFLRL_N_SENSED > 0\n"
"    free(h->sterm);\n"
"    free(h->sense);\n"
"    free(h->sense_ring);\n"
"#endif\n"
"    free(h->scratch);\n"
"    free(h->spec);\n"
"    free(h->seen_seeds);\n"
"    free(h);\n"
"}\n"
"\n"
"", out);
    fputs(
"#if KFLRL_N_THRUSTERS > 0\n"
"/* Thrust reaches translation as an acceleration on the gravity\n"
" * state's perturbation registry, which the integrator evaluates at\n"
" * its own stages: the thrust is integrated with everything else\n"
" * rather than added to a finished step. The registry is additive and\n"
" * is dispatched in registration order, which is fixed at create.\n"
" *\n"
" * The same declaration that gives a thruster its torque gives it its\n"
" * force, so a program cannot have one without the other. */\n"
"static void kflrl_thrust_perturb_(const K26AstroGravState *st,\n"
"                                  const K26AstroGravView *vw,\n"
"                                  K26V3 *accel, void *ctx)\n"
"{\n"
"    (void)st; (void)vw;\n"
"    KflrlThrustCtx *c = (KflrlThrustCtx *)ctx;\n"
"    if (!c || !c->h || !accel) return;\n"
"    for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {\n"
"        K26AstroVehicle *veh =\n"
"            c->h->vehicles[(size_t)c->e * KFLRL_N_VEHICLES + vi];\n"
"        if (!veh) continue;\n"
"        K26AstroAttWheel wbuf[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];\n"
"        K26AstroAttTorquer qbuf[KFLRL_N_TORQUERS > 0 ?\n"
"                                KFLRL_N_TORQUERS : 1];\n"
"        K26AstroAttThruster tbuf[KFLRL_N_THRUSTERS > 0 ?\n"
"                                 KFLRL_N_THRUSTERS : 1];\n"
"        int wmap[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];\n"
"        K26AstroAttActuators view;\n"
"        kflrl_act_view_(&c->h->act[c->e], vi, wbuf, qbuf, tbuf, &view,\n"
"                        wmap);\n"
"        K26V3 f_body;\n"
"        if (k26astro_att_thrusters_wrench(&view, &f_body, NULL) !=\n"
"            K26ASTRO_ATT_OK) continue;\n"
"        if (f_body.x == 0.0 && f_body.y == 0.0 && f_body.z == 0.0) {\n"
"            continue;\n"
"        }\n"
"        K26AstroBody *b = k26astro_vehicle_body(veh);\n"
"        if (!b || !(b->mass > 0.0)) continue;\n"
"        K26V3 f_world = k26m3d_quat_rotate_v3(b->attitude, f_body);\n"
"        int bi = kflrl_body_idx_[kflrl_vehicle_body_[vi]];\n"
"        if (bi < 0) continue;\n"
"        accel[bi].x += f_world.x / b->mass;\n"
"        accel[bi].y += f_world.y / b->mass;\n"
"        accel[bi].z += f_world.z / b->mass;\n"
"    }\n"
"}\n"
"#endif\n"
"\n", out);
    fputs(

"extern \"C\" K26RlStatus k26rl_env_create(uint64_t seed, uint32_t n_envs,\n"
"                                         K26RlEnv **out_env)\n"
"{\n"
"    if (!out_env) return K26RL_E_NULL;\n"
"    *out_env = NULL;\n"
"    if (n_envs == 0) return K26RL_E_GEOMETRY;\n"
"\n"
"    K26RlEnv *h = (K26RlEnv *)calloc(1, sizeof *h);\n"
"    if (!h) return K26RL_E_INTERNAL;\n"
"    h->n_envs = n_envs;\n"
"    h->seed = seed;\n"
"    h->key = k26rng_key(seed);\n"
"    h->rekey_ordinal = 0;\n"
"    h->control_dt = kflrl_control_dt_();\n"
"    h->horizon = kflrl_horizon_();\n"
"    h->substeps = kflrl_substeps_();\n"
"    h->restitution = kflrl_restitution_();\n"
"    h->friction    = kflrl_friction_();\n"
"    if (h->substeps < 1u) h->substeps = 1u;\n", out);
    fputs(
"    /* The control period is checked first and keeps the status it\n"
"     * has always returned; the subdivision's own check follows, so\n"
"     * that adding one cannot change what a bad period reports. */\n"
"    if (!(h->control_dt > 0.0) || !std::isfinite(h->control_dt)) {\n"
"        free(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"    h->sub_dt = h->control_dt / (double)h->substeps;\n"
"    if (!(h->sub_dt > 0.0) || !std::isfinite(h->sub_dt)) {\n"
"        kflrl_free_handle_(h);\n"
"        return K26RL_E_GEOMETRY;\n"
"    }\n"
"    kflrl_act_params_(h->act_lo, h->act_hi, h->act_arity, h->act_kind);\n"
"\n", out);
    fputs(
"    h->cap_seen = 4;\n"
"    h->seen_seeds = (uint64_t *)malloc(h->cap_seen * sizeof(uint64_t));\n"
"    h->worlds = (K26AstroWorld **)calloc(n_envs, sizeof(*h->worlds));\n"
"#if KFLRL_N_VEHICLES > 0\n"
"    h->vehicles = (K26AstroVehicle **)calloc(\n"
"        (size_t)n_envs * KFLRL_N_VEHICLES, sizeof(*h->vehicles));\n"
"    h->act = (KflrlAct *)calloc(n_envs, sizeof(*h->act));\n"
"    h->thrust_ctx = (KflrlThrustCtx *)calloc(n_envs,\n"
"                                             sizeof(*h->thrust_ctx));\n"
"#endif\n"
"    h->baseline = (K26AstroBody *)calloc(\n"
"        (size_t)n_envs * (KFLRL_N_BODIES ? KFLRL_N_BODIES : 1),\n"
"        sizeof(K26AstroBody));\n"
"    h->baseline_t = (K26AstroEpoch *)calloc(n_envs, sizeof(K26AstroEpoch));\n"
"    h->episode = (uint32_t *)calloc(n_envs, sizeof(uint32_t));\n"
"    h->steps = (uint32_t *)calloc(n_envs, sizeof(uint32_t));\n"
"    h->ended = (uint8_t *)calloc(n_envs, 1);\n"
"    h->obs = (double *)calloc(\n"
"        (size_t)n_envs * (KFLRL_OBS_TOTAL ? KFLRL_OBS_TOTAL : 1),\n"
"        sizeof(double));\n"
"    h->rew = (double *)calloc(n_envs, sizeof(double));\n"
"    h->flags = (uint32_t *)calloc(n_envs, sizeof(uint32_t));\n"
"    h->fault = (uint16_t *)calloc(n_envs, sizeof(uint16_t));\n"
"    h->dr_vals = (double *)calloc(\n"
"        (size_t)n_envs * (KFLRL_N_REC ? KFLRL_N_REC : 1),\n"
"        sizeof(double));\n"
"    h->wscal = (double *)calloc(\n"
"        (size_t)n_envs * (KFLRL_N_WSCAL ? KFLRL_N_WSCAL : 1),\n"
"        sizeof(double));\n"
"    /* The contact block is allocated for every program, not only\n"
"     * for one that declares a vehicle, because the observation\n"
"     * function takes it either way and a conditional signature\n"
"     * would buy nothing but a second shape to keep agreeing. */\n"
"    h->join = (KflrlJoin *)calloc(n_envs, sizeof(*h->join));\n"
"    h->contact = (KflrlContact *)calloc(\n"
"        (size_t)n_envs * KFLRL_N_CONTACT, sizeof(KflrlContact));\n"
"#if KFLRL_N_SENSED > 0\n"
"    h->sterm = (K26SenseTerm *)calloc(KFLRL_N_STERMS,\n"
"                                      sizeof(K26SenseTerm));\n"
"    h->sense = (K26SenseState *)calloc(\n"
"        (size_t)n_envs * KFLRL_N_SENSED, sizeof(K26SenseState));\n"
"    h->sense_ring = (double *)calloc(\n"
"        (size_t)n_envs * KFLRL_N_SENSED *\n"
"        (KFLRL_SENSE_RING > 0 ? KFLRL_SENSE_RING : 1),\n"
"        sizeof(double));\n"
"    if (!h->sterm || !h->sense || !h->sense_ring) {\n"
"        kflrl_free_handle_(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"#endif\n"
"    h->scratch = (double *)calloc(\n"
"        KFLRL_OBS_TOTAL ? KFLRL_OBS_TOTAL : 1, sizeof(double));\n"
"    if (!h->seen_seeds || !h->worlds || !h->baseline || !h->baseline_t ||\n"
"        !h->episode || !h->steps || !h->ended || !h->obs || !h->rew ||\n"
"        !h->flags || !h->fault || !h->dr_vals || !h->wscal ||\n"
"#if KFLRL_N_VEHICLES > 0\n"
"        !h->vehicles || !h->act || !h->thrust_ctx ||\n"
"#endif\n"
"        !h->join ||\n"
"        !h->contact || !h->scratch) {\n"
"        kflrl_free_handle_(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"    h->seen_seeds[0] = seed;\n"
"    h->n_seen = 1;\n"
"\n", out);
    fputs(
"    for (uint32_t e = 0; e < n_envs; e++) {\n"
"        h->worlds[e] = k26astro_world_create(K26ASTRO_MODE_PORTABLE,\n"
"                                             K26ASTRO_COORDS_SECTOR_GRID);\n"
"        if (!h->worlds[e]) {\n"
"            kflrl_free_handle_(h);\n"
"            return K26RL_E_INTERNAL;\n"
"        }\n"
"        (void)k26astro_world_set_seed(\n"
"            h->worlds[e], kflrl_world_seed_(h->key, e, 0u));\n"
"#if KFLRL_N_DR > 0\n"
"        double dr0[KFLRL_N_DR];\n"
"#else\n"
"        double *dr0 = NULL;\n"
"#endif\n"
"        if (kflrl_build_world_(h->worlds[e], h->key, e,\n"
"                h->wscal + (size_t)e * KFLRL_N_WSCAL, dr0,\n"
"#if KFLRL_N_VEHICLES > 0\n"
"                h->vehicles + (size_t)e * KFLRL_N_VEHICLES\n"
"#else\n"
"                NULL\n"
"#endif\n"
"                ) != 0) {\n"
"            kflrl_free_handle_(h);\n"
"            return K26RL_E_INTERNAL;\n"
"        }\n"
"#if KFLRL_N_THRUSTERS > 0\n"
"        /* Registered here rather than at world build because the\n"
"         * context is the handle and this environment's index, and\n"
"         * the handle is what owns them. Registration order is fixed\n"
"         * by this loop. */\n"
"        h->thrust_ctx[e].h = h;\n"
"        h->thrust_ctx[e].e = e;\n"
"        if (k26astro_grav_register_perturb(\n"
"                k26astro_world_grav(h->worlds[e]),\n"
"                kflrl_thrust_perturb_, &h->thrust_ctx[e]) != 0) {\n"
"            kflrl_free_handle_(h);\n"
"            return K26RL_E_INTERNAL;\n"
"        }\n"
"#endif\n"
"#if KFLRL_N_BODIES > 0\n"
"        {\n"
"            K26AstroBody *b0 = k26astro_world_body_at(h->worlds[e], 0);\n"
"            if (b0) {\n"
"                memcpy(h->baseline + (size_t)e * KFLRL_N_BODIES, b0,\n"
"                       sizeof(K26AstroBody) * KFLRL_N_BODIES);\n"
"            }\n"
"        }\n"
"#endif\n"
"        {\n"
"            K26AstroGravState *g = k26astro_world_grav(h->worlds[e]);\n"
"            if (g) h->baseline_t[e] = g->t;\n"
"        }\n"
"#if KFLRL_N_REC > 0\n"
"        kflrl_apply_draws_(h->worlds[e], h->key, e, 0,\n"
"                           h->dr_vals + (size_t)e * KFLRL_N_REC, dr0);\n"
"#else\n"
"        kflrl_apply_draws_(h->worlds[e], h->key, e, 0, NULL, dr0);\n"
"#endif\n"
"        kflrl_observe_(h->worlds[e],\n"
"                       h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"                       &h->contact[(size_t)e * KFLRL_N_CONTACT],\n"
"                       &h->join[e]);\n"
"    }\n"
"\n"
"    h->spec_len = kflrl_spec_write_(NULL, h);\n", out);
    fputs(
"    h->spec = (uint8_t *)malloc(h->spec_len);\n"
"    if (!h->spec) {\n"
"        kflrl_free_handle_(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"    (void)kflrl_spec_write_(h->spec, h);\n"
"\n", out);
    fputs(
"#if KFLRL_N_SENSED > 0\n"
"    /* The declared terms become the library's typed terms\n"
"     * here: the artifact is C++11 and cannot name a union\n"
"     * member in an initialiser, and the bias walk's decay and\n"
"     * driving coefficients follow from the control period,\n"
"     * which is read once, here, and never on the step path. */\n"
"    for (int t = 0; t < KFLRL_N_STERMS; t++) {\n"
"        K26SenseTerm *d = &h->sterm[t];\n"
"        d->kind       = (K26SenseKind)kflrl_sterm_[t].kind;\n"
"        d->channel    = kflrl_sterm_[t].ch;\n"
"        d->channel_ep = kflrl_sterm_[t].ch_ep;\n"
"        double p0 = kflrl_sterm_[t].p0;\n"
"        double p1 = kflrl_sterm_[t].p1;\n"
"        double p2 = kflrl_sterm_[t].p2;\n"
"        switch (d->kind) {\n"
"        case K26SENSE_ADDITIVE: d->u.additive.sigma = p0; break;\n"
"        case K26SENSE_SCALE: d->u.scale.rel_sigma = p0; break;\n"
"        case K26SENSE_BIAS_WALK:\n"
"            d->u.bias_walk.sigma0 = p0;\n"
"            if (k26sense_bias_walk_coeffs(\n"
"                    p1, h->control_dt, p2,\n"
"                    &d->u.bias_walk.phi,\n"
"                    &d->u.bias_walk.q) != K26SENSE_OK) {\n"
"                kflrl_free_handle_(h);\n"
"                return K26RL_E_INTERNAL;\n"
"            }\n"
"            break;\n"
"        case K26SENSE_LATENCY:\n"
"            d->u.latency.depth = (uint32_t)p0; break;\n"
"        case K26SENSE_QUANTISE:\n"
"            d->u.quantise.lsb = p0;\n"
"            d->u.quantise.lo = p1;\n"
"            d->u.quantise.hi = p2; break;\n"
"        case K26SENSE_DROPOUT: d->u.dropout.p = p0; break;\n"
"        default: break;\n"
"        }\n"
"    }\n"
"    for (uint32_t e = 0; e < n_envs; e++) {\n"
"        for (int c = 0; c < KFLRL_N_SENSED; c++) {\n"
"            size_t k = (size_t)e * KFLRL_N_SENSED + (size_t)c;\n"
"            h->sense[k].ring = h->sense_ring + k *\n"
"                (KFLRL_SENSE_RING > 0 ? KFLRL_SENSE_RING : 1);\n"
"            h->sense[k].ring_cap = KFLRL_SENSE_RING;\n"
"        }\n"
"        kflrl_sense_reset_(h, e, 0u,\n"
"                           h->obs + (size_t)e * KFLRL_OBS_TOTAL);\n"
"    }\n"
"#endif\n"
"    h->at_boundary = 1;\n"
"    h->magic = KFLRL_MAGIC;\n"
"    *out_env = h;\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"", out);
    fputs(
"extern \"C\" K26RlStatus k26rl_env_output(K26RlEnv *h, const char *path)\n"
"{\n"
"    if (!h) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    if (!h->at_boundary) return K26RL_E_OUTPUT_TIMING;\n"
"    if (h->writer) {\n"
"        K26RlStatus cst = k26rl_episode_writer_close(h->writer);\n"
"        h->writer = NULL;\n"
"        if (cst != K26RL_OK) return K26RL_E_INTERNAL;\n"
"    }\n"
"    if (!path) return K26RL_OK;\n"
"\n"
"    K26RlEpisodeGeom geom;\n"
"    geom.n_envs = h->n_envs;\n"
"    geom.agent_count = 1;\n"
"    geom.obs_total = KFLRL_OBS_TOTAL;\n"
"    geom.act_total = KFLRL_ACT_TOTAL;\n"
"    geom.steps_per_chunk = 1024;\n"
"    geom.dr_max = KFLRL_N_REC;\n"
"    K26RlStatus st = k26rl_episode_writer_open(\n"
"        path, &geom, h->seed, h->rekey_ordinal, \"3.2\",\n"
"        K26ASTRO_RT_LIB_VERSION, h->spec, h->spec_len, &h->writer);\n"
"    if (st != K26RL_OK) return st;\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        st = k26rl_episode_writer_start(\n"
"            h->writer, e, h->episode[e],\n"
"            h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"#if KFLRL_N_REC > 0\n"
"            kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,\n"
"#else\n"
"            NULL, NULL,\n"
"#endif\n"
"            KFLRL_N_REC);\n"
"        if (st != K26RL_OK) {\n"
"            (void)k26rl_episode_writer_close(h->writer);\n"
"            h->writer = NULL;\n"
"            return K26RL_E_INTERNAL;\n"
"        }\n"
"    }\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"", out);
    fputs(
"/* The telemetry tap: the same frames the episode writer emits,\n"
" * published on a shared memory ring instead of into a file. It is\n"
" * enabled the same way, at the same moments, with the same refusals,\n"
" * and it is a separate producer sharing no buffer and no counter\n"
" * with the writer, so a run's episode file is identical whether or\n"
" * not this is on. Publication is best-effort: nothing below can fail\n"
" * and nothing below can slow a step, so no status here reaches a\n"
" * stepping caller. */\n"
"extern \"C\" K26RlStatus k26rl_env_tap(K26RlEnv *h, const char *name)\n"
"{\n"
"    if (!h) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    if (!h->at_boundary) return K26RL_E_OUTPUT_TIMING;\n"
"    if (h->tap) {\n"
"        k26rl_tap_close(h->tap);\n"
"        h->tap = NULL;\n"
"    }\n"
"    if (!name) return K26RL_OK;\n"
"\n"
"    K26RlEpisodeGeom geom;\n"
"    geom.n_envs = h->n_envs;\n"
"    geom.agent_count = 1;\n"
"    geom.obs_total = KFLRL_OBS_TOTAL;\n"
"    geom.act_total = KFLRL_ACT_TOTAL;\n"
"    geom.steps_per_chunk = 1024;\n"
"    geom.dr_max = KFLRL_N_REC;\n"
"    K26RlStatus st = k26rl_tap_open(\n"
"        name, &geom, h->seed, h->rekey_ordinal, \"3.2\",\n"
"        K26ASTRO_RT_LIB_VERSION, h->spec, h->spec_len, &h->tap);\n"
"    if (st != K26RL_OK) return st;\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        k26rl_tap_start(\n"
"            h->tap, e, h->episode[e],\n"
"            h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"#if KFLRL_N_REC > 0\n"
"            kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,\n"
"#else\n"
"            NULL, NULL,\n"
"#endif\n"
"            KFLRL_N_REC);\n"
"    }\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"", out);
    fputs(
"/* An environment's episode ends by fault: no transition completes,\n"
" * the public observation slice keeps the pre-step values, the fault\n"
" * record and episode-end frame travel the file when enabled. */\n"
"static K26RlStatus kflrl_fault_(K26RlEnv *h, uint32_t e,\n"
"                                const double *aslice, uint16_t reason)\n"
"{\n"
"    h->rew[e] = 0.0;\n"
"    h->flags[e] = K26RL_FLAG_FAULT;\n"
"    h->fault[e] = reason;\n"
"    h->ended[e] = 1;\n"
"    if (h->writer) {\n"
"        double zero_reward = 0.0;\n"
"        double zero_adj = 0.0;\n"
"        K26RlStatus st = k26rl_episode_writer_step(\n"
"            h->writer, e, h->obs + (size_t)e * KFLRL_OBS_TOTAL, aslice,\n"
"            &zero_reward, K26RL_FLAG_FAULT, 0.0);\n"
"        if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"        st = k26rl_episode_writer_end(h->writer, e, K26RL_END_FAULT,\n"
"                                      reason, &zero_adj);\n"
"        if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"    }\n"
"    if (h->tap) {\n"
"        double zero_reward = 0.0;\n"
"        double zero_adj = 0.0;\n"
"        k26rl_tap_step(h->tap, e, h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"                       aslice, &zero_reward, K26RL_FLAG_FAULT, 0.0);\n"
"        k26rl_tap_end(h->tap, e, K26RL_END_FAULT, reason, &zero_adj);\n"
"    }\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"", out);
    fputs(
"extern \"C\" K26RlStatus k26rl_env_step(K26RlEnv *h, const double *actions)\n"
"{\n"
"    if (!h) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    if (KFLRL_ACT_TOTAL != 0 && !actions) return K26RL_E_NULL;\n"
"\n"
"    /* A boundary reset that would exhaust its episode coordinates\n"
"     * refuses the whole call before anything advances. */\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        if (h->ended[e] && h->episode[e] == 0xFFFFFFFFu) {\n"
"            return K26RL_E_RNG_EXHAUSTED;\n"
"        }\n"
"    }\n"
"    h->at_boundary = 0;\n"
"\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        const double *aslice = actions\n"
"            ? actions + (size_t)e * KFLRL_ACT_TOTAL : NULL;\n"
"        const double *wslice = h->wscal + (size_t)e * KFLRL_N_WSCAL;\n"
"        (void)wslice;\n"
"\n"
"        if (h->ended[e]) {\n"
"            /* Boundary reset: no transition, no time advance, the\n"
"             * new episode's initial observation, only the\n"
"             * reset-boundary bit. */\n"
"            kflrl_reset_env_(h, e, h->episode[e] + 1u);\n"
"            h->flags[e] = K26RL_FLAG_RESET_BOUNDARY;\n"
"            if (h->writer) {\n"
"                K26RlStatus st = k26rl_episode_writer_start(\n"
"                    h->writer, e, h->episode[e],\n"
"                    h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"#if KFLRL_N_REC > 0\n"
"                    kflrl_dr_tags_,\n"
"                    h->dr_vals + (size_t)e * KFLRL_N_REC,\n"
"#else\n"
"                    NULL, NULL,\n"
"#endif\n"
"                    KFLRL_N_REC);\n"
"                if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"            }\n"
"            if (h->tap) {\n"
"                k26rl_tap_start(\n"
"                    h->tap, e, h->episode[e],\n"
"                    h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"#if KFLRL_N_REC > 0\n"
"                    kflrl_dr_tags_,\n"
"                    h->dr_vals + (size_t)e * KFLRL_N_REC,\n"
"#else\n"
"                    NULL, NULL,\n"
"#endif\n"
"                    KFLRL_N_REC);\n"
"            }\n"
"            continue;\n"
"        }\n"
"\n"
"", out);
    fputs(
"        kflrl_on_step_(h->worlds[e], aslice, &h->act[e]);\n"
"        /* A contact is a fact about one transition, so the latch is\n"
"         * cleared here and whatever the sub-advances below find is\n"
"         * what this step reports. */\n"
"        memset(&h->contact[(size_t)e * KFLRL_N_CONTACT], 0,\n"
"               sizeof(KflrlContact) * KFLRL_N_CONTACT);\n"
"        /* One transition is `substeps` sub-advances. Translation\n"
"         * advances first, then attitude by the same interval with\n"
"         * the torque held at its start, which is the splitting the\n"
"         * attitude library documents and its gates measure.\n"
"         *\n"
"         * The last sub-advance takes the remainder rather than the\n"
"         * quotient, so the simulated time a transition advances sums\n"
"         * to control_dt exactly however the division rounded. That\n"
"         * subtraction is exact: by the last sub-advance at least\n"
"         * half the period has been advanced, so the two operands lie\n"
"         * within a factor of two of each other and the difference is\n"
"         * representable.\n"
"         */\n"
"        int rc = 0;\n"
"        uint16_t att_reason = 0;\n"
"        double advanced = 0.0;\n"
"        for (uint32_t sub = 0; sub < h->substeps; sub++) {\n"
"            double step_dt = (sub + 1u == h->substeps)\n"
"                           ? (h->control_dt - advanced)\n"
"                           : h->sub_dt;\n"
"#if KFLRL_N_COLL > 0\n"
"            /* The sweep needs both endpoints of the sub-advance, so\n"
"             * the starting configuration is taken before the world\n"
"             * moves. Fixed-size buffers over compile-time counts:\n"
"             * nothing here allocates. */\n"
"            K26AstroPos csnap[KFLRL_N_VEHICLES];\n"
"            K26V3       cvel0[KFLRL_N_VEHICLES];\n"
"            K26Quat     cquat[KFLRL_N_VEHICLES];\n"
"            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {\n"
"                const K26AstroBody *cb = k26astro_world_body_at(\n"
"                    h->worlds[e], kflrl_body_idx_[kflrl_vehicle_body_[vi]]);\n"
"                if (!cb) { memset(&csnap[vi], 0, sizeof csnap[vi]);\n"
"                           cvel0[vi] = k26m3d_v3(0.0, 0.0, 0.0);\n"
"                           cquat[vi] = k26m3d_quat_identity(); continue; }\n"
"                csnap[vi] = cb->pos;\n"
"                cvel0[vi] = cb->vel;\n"
"                cquat[vi] = cb->attitude;\n"
"            }\n"
"#endif\n"
"            rc = k26astro_world_step_exact(h->worlds[e], step_dt);\n"
"            if (rc != 0) break;\n"
"            advanced += step_dt;\n", out);
    fputs(
"#if KFLRL_N_VEHICLES > 0\n"
"            /* The gravity-gradient torque is part of the torque sum\n"
"             * and is computed per vehicle from its own separation\n"
"             * from the body it orbits. A vehicle whose body declares\n"
"             * no parent has no attractor named and takes no torque\n"
"             * from this source. */\n"
"            K26V3 gg[KFLRL_N_VEHICLES];\n"
"            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {\n"
"                gg[vi].x = gg[vi].y = gg[vi].z = 0.0;\n"
"                K26AstroVehicle *veh =\n"
"                    h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi];\n"
"                if (!veh) continue;\n"
"                const K26AstroBody *vb = k26astro_world_body_at(\n"
"                    h->worlds[e], kflrl_body_idx_[kflrl_vehicle_body_[vi]]);\n"
"                if (!vb || vb->parent_body_idx < 0) continue;\n"
"                const K26AstroBody *pb = k26astro_world_body_at(\n"
"                    h->worlds[e], vb->parent_body_idx);\n"
"                if (!pb) continue;\n"
"                K26V3 rw = k26astro_pos_sub(&pb->pos, &vb->pos);\n"
"                (void)k26astro_att_gravity_gradient(veh, rw, pb->gm,\n"
"                                                    &gg[vi]);\n"
"            }\n"
"            /* Each vehicle advances with its own actuators, its own\n"
"             * gravity-gradient torque, and the local magnetic field\n"
"             * a magnetorquer needs. The actuator view is built over\n"
"             * the environment's state and the momenta are stored\n"
"             * back, so the state lives where the reset can clear it\n"
"             * and nothing here allocates. */\n", out);
    fputs(
"            K26AstroAttStatus ast = K26ASTRO_ATT_OK;\n"
"            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {\n"
"                K26AstroVehicle *veh =\n"
"                    h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi];\n"
"                if (!veh) continue;\n"
"                K26AstroAttWheel wbuf[KFLRL_N_WHEELS > 0 ?\n"
"                                      KFLRL_N_WHEELS : 1];\n"
"                K26AstroAttTorquer qbuf[KFLRL_N_TORQUERS > 0 ?\n"
"                                        KFLRL_N_TORQUERS : 1];\n"
"                K26AstroAttThruster tbuf[KFLRL_N_THRUSTERS > 0 ?\n"
"                                         KFLRL_N_THRUSTERS : 1];\n"
"                int wmap[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];\n"
"                K26AstroAttActuators view;\n"
"                kflrl_act_view_(&h->act[e], vi, wbuf, qbuf, tbuf,\n"
"                                &view, wmap);\n"
"                K26V3 bfield = kflrl_field_body_(h->worlds[e], veh, vi);\n"
"                ast = k26astro_att_step_actuated(veh, &view, gg[vi],\n"
"                                                 bfield, step_dt);\n"
"                /* The wheel momenta are written back only when the\n"
"                 * advance stood. A failed advance leaves the\n"
"                 * orientation and rate as they were, so writing the\n"
"                 * momenta back would leave the two halves of one\n"
"                 * step disagreeing about whether it happened. */\n"
"                if (ast != K26ASTRO_ATT_OK) break;\n"
"                kflrl_act_store_(&h->act[e], &view, wmap);\n"
"            }\n"
"            if (ast != K26ASTRO_ATT_OK) {\n"
"                att_reason = (ast == K26ASTRO_ATT_E_DIVERGED)\n"
"                    ? (uint16_t)K26RL_E_DIVERGED\n"
"                    : (uint16_t)K26RL_E_ENV_INTERNAL;\n"
"                break;\n"
"            }\n"
"#endif\n"
"#if KFLRL_N_COLL > 0\n"
"            /* The collision pass, between the sub-advances. A body\n"
"             * that crosses a target inside one control period is not\n"
"             * found by comparing the period's endpoints, and\n"
"             * shortening the step to catch it would make the applied\n"
"             * duration a function of the geometry; sweeping inside\n"
"             * the sub-advance finds it and leaves the duration\n"
"             * exactly what was declared.\n"
"             *\n"
"             * Every position is taken as an exact difference from\n"
"             * one reference rather than as a flattened coordinate,\n"
"             * because a position here is a sector index and a\n"
"             * bounded offset, and flattening two of them before\n"
"             * subtracting throws away the precision the sector grid\n"
"             * exists to keep. */\n"
"", out);
    fputs(
"            {\n"
"                K26AstroCollBody cbody[KFLRL_N_VEHICLES];\n"
"                const K26AstroPos *cref = &csnap[0];\n"
"                for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {\n"
"                    memset(&cbody[vi], 0, sizeof cbody[vi]);\n"
"                    const K26AstroBody *cb = k26astro_world_body_at(\n"
"                        h->worlds[e],\n"
"                        kflrl_body_idx_[kflrl_vehicle_body_[vi]]);\n"
"                    if (!cb) continue;\n"
"                    cbody[vi].pos0 = k26astro_pos_sub(&csnap[vi], cref);\n"
"                    cbody[vi].pos1 = k26astro_pos_sub(&cb->pos, cref);\n"
"                    cbody[vi].orientation = cquat[vi];\n"
"                    cbody[vi].vel0 = cvel0[vi];\n"
"                    cbody[vi].vel1 = cb->vel;\n"
"                    cbody[vi].shapes = &kflrl_coll_[kflrl_coll_first_[vi]];\n"
"                    cbody[vi].n_shapes = kflrl_coll_count_[vi];\n"
"                    cbody[vi].bound_radius = kflrl_veh_bound_[vi];\n"
"                    cbody[vi].mass = cb->mass;\n"
"                    cbody[vi].com_offset = k26m3d_v3(\n"
"                        kflrl_veh_com_[vi][0], kflrl_veh_com_[vi][1],\n"
"                        kflrl_veh_com_[vi][2]);\n"
"#if KFLRL_N_PORTS > 1\n"
"                    /* A joined pair is one body, and the pass tests\n"
"                     * pairs of bodies, so the follower's primitives\n"
"                     * leave it; left in, the join itself would be\n"
"                     * reported as a contact on every sub-advance. */\n"
"                    if (h->join[e].active &&\n"
"                        h->join[e].follower == vi) {\n"
"                        cbody[vi].n_shapes = 0;\n"
"                    }\n"
"#endif\n"
"                    cbody[vi].omega = cb->omega;\n"
"", out);
    fputs(
"", out);
    fputs(
"                    /* The inverse inertia the angular half of an\n"
"                     * impulse turns on, taken from the vehicle's own\n"
"                     * attitude state, which is where the assembly's\n"
"                     * derived tensor was installed and inverted. Left\n"
"                     * at the zero matrix by the clearing above, an\n"
"                     * off-centre impact would produce no spin at all\n"
"                     * while every other term looked right. */\n"
"                    {\n"
"                        K26AstroVehicle *cvh =\n"
"                            h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi];\n"
"                        const K26AstroAttitudeStateExt *cx = cvh\n"
"                            ? k26astro_vehicle_attitude_ext(cvh) : NULL;\n"
"                        if (cx) cbody[vi].inv_inertia = cx->inertia_inverse;\n"
"                    }\n"
"                }\n"
"", out);
    fputs(
"                K26AstroCollContact cc;\n"
"                if (k26astro_coll_pass(cbody, KFLRL_N_VEHICLES, step_dt,\n"
"                                       &cc) == K26ASTRO_COLL_OK &&\n"
"                    cc.hit) {\n"
"                    int captured = 0;\n"
"#if KFLRL_N_PORTS > 1\n"
"                    /* Was this contact between two docking\n"
"                     * interfaces, and did it satisfy the envelope?\n"
"                     * Both are decided BEFORE any resolution runs,\n"
"                     * because a capture takes precedence over the\n"
"                     * environment's declared resolution: a\n"
"                     * programme cannot be told its craft docked and\n"
"                     * then shown it flung away.\n"
"                     *\n"
"                     * The verdict is the pair\'s, not one port\'s.\n"
"                     * Either port can be read as the active one and\n"
"                     * the two readings differ slightly, so a\n"
"                     * capture requires both: the conservative\n"
"                     * reading, and the one that leaves the two\n"
"                     * ports\' channels agreeing about a fact of the\n"
"                     * pair. */\n"
"", out);
    fputs(
"                    int pidx[2] = { -1, -1 };\n"
"                    K26AstroCollPortState pst[2];\n"
"                    int pair0[2] = { cc.body_a, cc.body_b };\n"
"                    for (int q = 0; q < KFLRL_N_PORTS; q++) {\n"
"                        if (kflrl_ports_[q].veh == cc.body_a &&\n"
"                            kflrl_ports_[q].shape == cc.shape_a) pidx[0] = q;\n"
"                        if (kflrl_ports_[q].veh == cc.body_b &&\n"
"                            kflrl_ports_[q].shape == cc.shape_b) pidx[1] = q;\n"
"                    }\n"
"", out);
    fputs(
"                    memset(pst, 0, sizeof pst);\n"
"                    if (pidx[0] >= 0 && pidx[1] >= 0) {\n"
"                        captured = 1;\n"
"                        for (int q = 0; q < 2; q++) {\n"
"                            if (k26astro_coll_port_state(\n"
"                                    &cbody[pair0[q]],\n"
"                                    &kflrl_ports_[pidx[q]].geom,\n"
"                                    &cbody[pair0[1 - q]],\n"
"                                    &kflrl_ports_[pidx[1 - q]].geom,\n"
"                                    cc.time, &pst[q]) != K26ASTRO_COLL_OK) {\n"
"                                pidx[0] = -1;\n"
"                                captured = 0;\n"
"                                break;\n"
"                            }\n"
"                            if (!k26astro_coll_port_captured(\n"
"                                    &pst[q], &kflrl_ports_[pidx[q]].env)) {\n"
"                                captured = 0;\n"
"                            }\n"
"                        }\n"
"                    }\n"
"#endif\n"
"                    /* Arrest, the default resolution: the pair is\n"
"                     * placed at the sweep's own interpolated\n"
"                     * configuration, so the reported contact and the\n"
"                     * recorded state agree exactly, and the relative\n"
"                     * velocity is removed by a momentum-conserving\n"
"                     * merge. The remainder of the control period\n"
"                     * advances with the pair moving together. */\n"
"                    K26V3 apos, avel, bpos, bvel;\n"
"                    K26V3 awb = cbody[cc.body_a].omega;\n"
"                    K26V3 bwb = cbody[cc.body_b].omega;\n"
"#if KFLRL_CONTACT_BOUNCE\n"
"                    /* Bounce: one impulse at the contact point,\n"
"                     * using the effective mass there, so an\n"
"                     * off-centre impact spins the body by the\n"
"                     * amount the geometry gives. */\n"
"                    K26AstroCollStatus cst = captured\n"
"                        ? K26ASTRO_COLL_OK\n"
"                        : k26astro_coll_bounce(\n"
"                        &cbody[cc.body_a], &cbody[cc.body_b], &cc,\n"
"                        h->restitution, h->friction,\n"
"                        &apos, &avel, &awb, &bpos, &bvel, &bwb);\n"
"#else\n"
"                    K26AstroCollStatus cst = captured\n"
"                        ? K26ASTRO_COLL_OK\n"
"                        : k26astro_coll_arrest(\n"
"                        &cbody[cc.body_a], &cbody[cc.body_b], cc.time,\n"
"                        &apos, &avel, &bpos, &bvel);\n"
"#endif\n"
"", out);
    fputs(
"                    /* A capture is not one of the declared\n"
"                     * resolutions and does not run either of them.\n"
"                     * It makes the pair one body instead. */\n"
"                    if (captured) cst = K26ASTRO_COLL_E_NULL;\n"
"#if KFLRL_N_PORTS > 1\n"
"                    if (captured) {\n"
"                        kflrl_join_form_(h, e, cbody, cc.body_a,\n"
"                                         cc.body_b, cc.time, cref);\n"
"                    }\n"
"#endif\n"
"                    if (cst == K26ASTRO_COLL_OK) {\n"
"                        int pair[2] = { cc.body_a, cc.body_b };\n"
"                        K26V3 np[2] = { apos, bpos };\n"
"                        K26V3 nv[2] = { avel, bvel };\n"
"                        K26V3 nw[2] = { awb, bwb };\n"
"                        for (int q = 0; q < 2; q++) {\n"
"                            int vi = pair[q];\n"
"                            K26AstroBody *cb = k26astro_world_body_at(\n"
"                                h->worlds[e],\n"
"                                kflrl_body_idx_[kflrl_vehicle_body_[vi]]);\n"
"                            if (!cb) continue;\n"
"                            /* The correction is applied as a delta so\n"
"                             * the sector representation is preserved\n"
"                             * rather than rebuilt from a flattened\n"
"                             * coordinate. */\n"
"                            K26V3 now = k26astro_pos_sub(&cb->pos, cref);\n"
"                            k26astro_pos_add(&cb->pos, k26m3d_v3(\n"
"                                np[q].x - now.x, np[q].y - now.y,\n"
"                                np[q].z - now.z));\n"
"                            cb->vel = nv[q];\n"
"                            cb->omega = nw[q];\n"
"                        }\n"
"                    }\n"
"", out);
    fputs(
"                    /* The fraction the channels publish is of the\n"
"                     * whole control period, not of this sub-advance,\n"
"                     * and the first contact of the transition is the\n"
"                     * one that is kept. */\n"
"                    double cfrac = h->control_dt > 0.0\n"
"                        ? ((advanced - step_dt) + cc.time * step_dt)\n"
"                          / h->control_dt\n"
"                        : 0.0;\n"
"                    int pair[2] = { cc.body_a, cc.body_b };\n"
"", out);
    fputs(
"                    for (int q = 0; q < 2; q++) {\n"
"                        KflrlContact *ct = &h->contact[\n"
"                            (size_t)e * KFLRL_N_CONTACT + pair[q]];\n"
"                        if (ct->hit != 0.0) continue;\n"
"                        ct->hit      = 1.0;\n"
"                        ct->fraction = cfrac;\n"
"                        ct->speed    = cc.speed;\n"
"#if KFLRL_N_PORTS > 1\n"
"                        if (pidx[0] < 0 || pidx[1] < 0) continue;\n"
"                        /* The residuals are of this body\'s own port\n"
"                         * as the active one, which is the sense the\n"
"                         * form that reads them names, taken at the\n"
"                         * impact configuration and computed above,\n"
"                         * before any resolution reached the world.\n"
"                         * The capture flag is the pair\'s and is the\n"
"                         * same on both ports. */\n"
"", out);
    fputs(
"                        ct->port_hit   = 1.0;\n"
"                        ct->captured   = captured ? 1.0 : 0.0;\n"
"                        ct->axial      = pst[q].axial;\n"
"                        ct->lateral    = pst[q].lateral;\n"
"                        ct->pitchyaw   = pst[q].pitchyaw;\n"
"                        ct->roll       = pst[q].roll;\n"
"                        ct->v_axial    = pst[q].v_axial;\n"
"                        ct->v_lateral  = pst[q].v_lateral;\n"
"                        ct->v_pitchyaw = pst[q].v_pitchyaw;\n"
"                        ct->v_roll     = pst[q].v_roll;\n"
"#endif\n"
"                    }\n"
"                }\n"
"#if KFLRL_N_PORTS > 1\n"
"                /* A joined pair advances as one body: the follower\n"
"                 * is placed from the carrier at the end of every\n"
"                 * sub-advance, including the one that formed the\n"
"                 * join, rather than integrated on its own. */\n"
"                kflrl_join_impose_(h, e, cref);\n"
"#endif\n"
"            }\n"
"#endif\n"
"        }\n"
"        if (att_reason != 0) {\n"
"            K26RlStatus fst = kflrl_fault_(h, e, aslice, att_reason);\n"
"            if (fst != K26RL_OK) return fst;\n"
"            continue;\n"
"        }\n"
"        if (rc != 0) {\n"
"            int code = rc < 0 ? -rc : rc;\n"
"            if (code == K26ASTRO_RT_E_FPU_RACE) return K26RL_E_FPU_RACE;\n"
"            if (code == K26ASTRO_RT_E_OOM) return K26RL_E_INTERNAL;\n"
"            uint16_t reason = (code == K26ASTRO_RT_E_INTEGRATOR)\n"
"                ? (uint16_t)K26RL_E_DIVERGED\n"
"                : (uint16_t)K26RL_E_ENV_INTERNAL;\n"
"            K26RlStatus fst = kflrl_fault_(h, e, aslice, reason);\n"
"            if (fst != K26RL_OK) return fst;\n"
"            continue;\n"
"        }\n"
"\n"
"        kflrl_observe_(h->worlds[e], h->scratch,\n"
"                       &h->contact[(size_t)e * KFLRL_N_CONTACT],\n"
"                       &h->join[e]);\n"
"        /* The transition index is the draw index every per-step term\n"
"         * uses, and h->steps[e] is still the count before this\n"
"         * transition, so the first transition of an episode draws at\n"
"         * index 0. */\n"
"        kflrl_sense_apply_(h, e, h->episode[e], h->steps[e],\n"
"                           h->scratch);\n"
"        int finite = 1;\n"
"        for (uint32_t j = 0; j < KFLRL_OBS_TOTAL; j++) {\n"
"            if (!std::isfinite(h->scratch[j])) finite = 0;\n"
"        }\n"
"        if (!finite) {\n"
"            K26RlStatus fst = kflrl_fault_(h, e, aslice,\n"
"                                           (uint16_t)K26RL_E_DIVERGED);\n"
"            if (fst != K26RL_OK) return fst;\n"
"            continue;\n"
"        }\n"
"\n"
"        uint32_t ns = h->steps[e] + 1u;\n"
"        double r = kflrl_reward_(h->scratch, aslice, ns, wslice);\n"
"        if (!std::isfinite(r)) {\n"
"            K26RlStatus fst = kflrl_fault_(\n"
"                h, e, aslice, (uint16_t)K26RL_E_ENV_INTERNAL);\n"
"            if (fst != K26RL_OK) return fst;\n"
"            continue;\n"
"        }\n"
"\n"
"        /* Terminal adjustment on termination only; truncation\n"
"         * carries none. Evaluated before the transition commits so\n"
"         * a non-finite adjusted reward faults like a non-finite\n"
"         * reward, never reaching the recorded stream. */\n"
"        int term = kflrl_terminated_(h->scratch, aslice, ns, wslice);\n"
"        double tadj = 0.0;\n"
"        if (term) {\n"
"            tadj = kflrl_terminal_(h->scratch, aslice, ns, wslice);\n"
"            r += tadj;\n"
"            if (!std::isfinite(r)) {\n"
"                K26RlStatus fst = kflrl_fault_(\n"
"                    h, e, aslice, (uint16_t)K26RL_E_ENV_INTERNAL);\n"
"                if (fst != K26RL_OK) return fst;\n"
"                continue;\n"
"            }\n"
"        }\n"
"\n"
"", out);
    fputs(
"        /* The transition commits. Termination and truncation are\n"
"         * distinct outcomes: termination wins when both land on one\n"
"         * step, and the flag word agrees in kind with the episode\n"
"         * file's end reason. */\n"
"        h->steps[e] = ns;\n"
"        memcpy(h->obs + (size_t)e * KFLRL_OBS_TOTAL, h->scratch,\n"
"               sizeof(double) * KFLRL_OBS_TOTAL);\n"
"        uint32_t f = 0;\n"
"        if (term) {\n"
"            f |= K26RL_FLAG_TERMINATED;\n"
"        } else if (h->horizon != 0 && ns >= h->horizon) {\n"
"            f |= K26RL_FLAG_TRUNCATED;\n"
"        }\n"
"        h->rew[e] = r;\n"
"        h->flags[e] = f;\n"
"        h->fault[e] = 0;\n"
"        if (f & (K26RL_FLAG_TERMINATED | K26RL_FLAG_TRUNCATED)) {\n"
"            h->ended[e] = 1;\n"
"        }\n"
"        if (h->writer) {\n"
"            K26RlStatus st = k26rl_episode_writer_step(\n"
"                h->writer, e, h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"                aslice, &h->rew[e], f, h->control_dt);\n"
"            if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"            if (h->ended[e]) {\n"
"                st = k26rl_episode_writer_end(\n"
"                    h->writer, e,\n"
"                    term ? K26RL_END_TERMINATED : K26RL_END_TRUNCATED,\n"
"                    0, &tadj);\n"
"                if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"            }\n"
"        }\n"
"        if (h->tap) {\n"
"            k26rl_tap_step(h->tap, e,\n"
"                           h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"                           aslice, &h->rew[e], f, h->control_dt);\n"
"            if (h->ended[e]) {\n"
"                k26rl_tap_end(\n"
"                    h->tap, e,\n"
"                    term ? K26RL_END_TERMINATED : K26RL_END_TRUNCATED,\n"
"                    0, &tadj);\n"
"            }\n"
"        }\n"
"    }\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"", out);
    fputs(
"/* Shared by the two explicit reset calls. With output enabled, an\n"
" * episode cut mid-flight records a truncated end (the recording\n"
" * ended; the old ordinal's frames precede any rekey frame), then\n"
" * every environment restarts and re-emits its episode-start. */\n"
"static K26RlStatus kflrl_reset_all_(K26RlEnv *h, int rekey,\n"
"                                    uint64_t new_seed)\n"
"{\n"
"    if (!rekey) {\n"
"        for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"            if (h->episode[e] == 0xFFFFFFFFu) {\n"
"                return K26RL_E_RNG_EXHAUSTED;\n"
"            }\n"
"        }\n"
"    }\n"
"    if (h->writer) {\n"
"        double zero_adj = 0.0;\n"
"        for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"            if (h->ended[e]) continue;   /* end frame already written */\n"
"            K26RlStatus st = k26rl_episode_writer_end(\n"
"                h->writer, e, K26RL_END_TRUNCATED, 0, &zero_adj);\n"
"            if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"        }\n"
"    }\n"
"    if (h->tap) {\n"
"        double zero_adj = 0.0;\n"
"        for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"            if (h->ended[e]) continue;   /* end frame already published */\n"
"            k26rl_tap_end(h->tap, e, K26RL_END_TRUNCATED, 0, &zero_adj);\n"
"        }\n"
"    }\n"
"    if (rekey) {\n"
"        h->seed = new_seed;\n"
"        h->key = k26rng_key(new_seed);\n"
"        h->rekey_ordinal += 1u;\n"
"        if (h->writer) {\n"
"            K26RlStatus st = k26rl_episode_writer_rekey(\n"
"                h->writer, new_seed, NULL);\n"
"            if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"        }\n"
"        if (h->tap) k26rl_tap_rekey(h->tap, new_seed);\n"
"    }\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        uint32_t ep = rekey ? 0u : h->episode[e] + 1u;\n"
"        kflrl_reset_env_(h, e, ep);\n"
"        h->flags[e] = 0;\n"
"    }\n"
"    h->at_boundary = 1;\n"
"    if (h->writer) {\n"
"        for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"            K26RlStatus st = k26rl_episode_writer_start(\n"
"                h->writer, e, h->episode[e],\n"
"                h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"#if KFLRL_N_REC > 0\n"
"                kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,\n"
"#else\n"
"                NULL, NULL,\n"
"#endif\n"
"                KFLRL_N_REC);\n"
"            if (st != K26RL_OK) return K26RL_E_INTERNAL;\n"
"        }\n"
"    }\n"
"    if (h->tap) {\n"
"        for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"            k26rl_tap_start(\n"
"                h->tap, e, h->episode[e],\n"
"                h->obs + (size_t)e * KFLRL_OBS_TOTAL,\n"
"#if KFLRL_N_REC > 0\n"
"                kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,\n"
"#else\n"
"                NULL, NULL,\n"
"#endif\n"
"                KFLRL_N_REC);\n"
"        }\n"
"    }\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"", out);
    fputs(
"extern \"C\" K26RlStatus k26rl_env_reset(K26RlEnv *h)\n"
"{\n"
"    if (!h) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    return kflrl_reset_all_(h, 0, 0);\n"
"}\n"
"\n"
"extern \"C\" K26RlStatus k26rl_env_reset_seeded(K26RlEnv *h, uint64_t seed)\n"
"{\n"
"    if (!h) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    for (uint32_t i = 0; i < h->n_seen; i++) {\n"
"        if (h->seen_seeds[i] == seed) return K26RL_E_SEED_REUSE;\n"
"    }\n"
"    if (h->n_seen == h->cap_seen) {\n"
"        uint32_t nc = h->cap_seen * 2u;\n"
"        uint64_t *ns = (uint64_t *)realloc(h->seen_seeds,\n"
"                                           nc * sizeof(uint64_t));\n"
"        if (!ns) return K26RL_E_INTERNAL;\n"
"        h->seen_seeds = ns;\n"
"        h->cap_seen = nc;\n"
"    }\n"
"    h->seen_seeds[h->n_seen++] = seed;\n"
"    return kflrl_reset_all_(h, 1, seed);\n"
"}\n"
"\n"
"", out);
    fputs(
"extern \"C\" K26RlStatus k26rl_env_obs(const K26RlEnv *h, double *out)\n"
"{\n"
"    if (!h || !out) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    memcpy(out, h->obs,\n"
"           sizeof(double) * (size_t)h->n_envs * KFLRL_OBS_TOTAL);\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"extern \"C\" K26RlStatus k26rl_env_reward(const K26RlEnv *h, double *out)\n"
"{\n"
"    if (!h || !out) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    memcpy(out, h->rew, sizeof(double) * h->n_envs);\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"extern \"C\" K26RlStatus k26rl_env_flags(const K26RlEnv *h, uint32_t *out)\n"
"{\n"
"    if (!h || !out) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    memcpy(out, h->flags, sizeof(uint32_t) * h->n_envs);\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"extern \"C\" K26RlStatus k26rl_env_fault_codes(const K26RlEnv *h,\n"
"                                              uint16_t *out)\n"
"{\n"
"    if (!h || !out) return K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;\n"
"    memcpy(out, h->fault, sizeof(uint16_t) * h->n_envs);\n"
"    return K26RL_OK;\n"
"}\n"
"\n"
"extern \"C\" int32_t k26rl_env_spec(const K26RlEnv *h, uint8_t *out,\n"
"                                   uint32_t capacity)\n"
"{\n"
"    if (!h) return -(int32_t)K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;\n"
"    if (capacity >= h->spec_len) {\n"
"        if (!out) return -(int32_t)K26RL_E_NULL;\n"
"        memcpy(out, h->spec, h->spec_len);\n"
"    }\n"
"    return (int32_t)h->spec_len;\n"
"}\n"
"\n"
"/* Body states: positions relative to the reference body, computed\n"
" * with the runtime's exact position subtraction rather than by\n"
" * flattening two absolute coordinates, and the bodies' own\n"
" * velocities. Sized like the spec getter, env-major like every\n"
" * other buffer, and a pure read that allocates nothing. */\n"
"extern \"C\" int32_t k26rl_env_bodies(const K26RlEnv *h, uint32_t reference,\n"
"                                     double *out, uint32_t capacity)\n"
"{\n"
"    if (!h) return -(int32_t)K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;\n"
"#if KFLRL_N_BODIES <= 0\n"
"    (void)reference; (void)out; (void)capacity;\n"
"    return 0;\n"
"#else\n"
"    if (reference != K26RL_BODY_REF_ORIGIN &&\n"
"        reference >= (uint32_t)KFLRL_N_BODIES) {\n"
"        return -(int32_t)K26RL_E_GEOMETRY;\n"
"    }\n"
"    uint64_t need = (uint64_t)h->n_envs * (uint32_t)KFLRL_N_BODIES * 6u;\n"
"    if (need > 0x7FFFFFFFu) return -(int32_t)K26RL_E_GEOMETRY;\n"
"    if (capacity < need) return (int32_t)need;\n"
"    if (!out) return -(int32_t)K26RL_E_NULL;\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        K26AstroWorld *w = h->worlds[e];\n"
"        const K26AstroBody *rb = NULL;\n"
"        if (reference != K26RL_BODY_REF_ORIGIN) {\n"
"            rb = k26astro_world_body_at(w, kflrl_body_idx_[reference]);\n"
"        }\n"
"        for (uint32_t b = 0; b < (uint32_t)KFLRL_N_BODIES; b++) {\n"
"            const K26AstroBody *bd =\n"
"                k26astro_world_body_at(w, kflrl_body_idx_[b]);\n"
"            double *o = out + ((size_t)e * (uint32_t)KFLRL_N_BODIES + b) * 6;\n"
"            K26V3 r;\n"
"            if (!bd) {\n"
"                o[0] = o[1] = o[2] = o[3] = o[4] = o[5] = 0.0;\n"
"                continue;\n"
"            }\n"
"            if (rb) {\n"
"                r = k26astro_pos_sub(&bd->pos, &rb->pos);\n"
"            } else {\n"
"                K26AstroPos origin;\n"
"                memset(&origin, 0, sizeof origin);\n"
"                r = k26astro_pos_sub(&bd->pos, &origin);\n"
"            }\n"
"            o[0] = r.x; o[1] = r.y; o[2] = r.z;\n"
"            o[3] = bd->vel.x; o[4] = bd->vel.y; o[5] = bd->vel.z;\n"
"        }\n"
"    }\n"
"    return (int32_t)need;\n"
"#endif\n"
"}\n"
"\n", out);
    fputs(
"extern \"C\" int32_t k26rl_env_attitudes(const K26RlEnv *h, double *out,\n"
"                                        uint32_t capacity)\n"
"{\n"
"    if (!h) return -(int32_t)K26RL_E_NULL;\n"
"    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;\n"
"#if KFLRL_N_BODIES <= 0\n"
"    (void)out; (void)capacity;\n"
"    return 0;\n"
"#else\n"
"    uint64_t need = (uint64_t)h->n_envs * (uint32_t)KFLRL_N_BODIES * 7u;\n"
"    if (need > 0x7FFFFFFFu) return -(int32_t)K26RL_E_GEOMETRY;\n"
"    if (capacity < need) return (int32_t)need;\n"
"    if (!out) return -(int32_t)K26RL_E_NULL;\n"
"    for (uint32_t e = 0; e < h->n_envs; e++) {\n"
"        K26AstroWorld *w = h->worlds[e];\n"
"        for (uint32_t b = 0; b < (uint32_t)KFLRL_N_BODIES; b++) {\n"
"            const K26AstroBody *bd =\n"
"                k26astro_world_body_at(w, kflrl_body_idx_[b]);\n"
"            double *o = out + ((size_t)e * (uint32_t)KFLRL_N_BODIES + b) * 7;\n"
"            if (!bd) {\n"
"                /* No body, no orientation: the identity, which is\n"
"                 * what an untracked attitude holds. */\n"
"                o[0] = 1.0; o[1] = o[2] = o[3] = 0.0;\n"
"                o[4] = o[5] = o[6] = 0.0;\n"
"                continue;\n"
"            }\n"
"            o[0] = bd->attitude.w; o[1] = bd->attitude.x;\n"
"            o[2] = bd->attitude.y; o[3] = bd->attitude.z;\n"
"            o[4] = bd->omega.x; o[5] = bd->omega.y; o[6] = bd->omega.z;\n"
"        }\n"
"    }\n"
"    return (int32_t)need;\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" void k26rl_env_destroy(K26RlEnv *h)\n"
"{\n"
"    if (!h || !kflrl_live_(h)) return;\n"
"    if (h->writer) {\n"
"        (void)k26rl_episode_writer_close(h->writer);\n"
"        h->writer = NULL;\n"
"    }\n"
"    if (h->tap) {\n"
"        k26rl_tap_close(h->tap);\n"
"        h->tap = NULL;\n"
"    }\n"
"    h->magic = 0;\n"
"    kflrl_free_handle_(h);\n"
"}\n"
"\n", out);
}

/* The batch entry: the episode machinery at declared default actions,
 * stepping until every environment has recorded at least K completed
 * episodes; early finishers keep stepping and record their extras. */
static void rl_emit_batch_main_(FILE *out, const KflcNode *form)
{
    fputs(
"#ifdef KFLC_RL_BATCH_MAIN\n"
"\n"
"static int kflrl_usage_(const char *prog)\n"
"{\n"
"    fprintf(stderr,\n"
"        \"usage: %s [--envs <N>] [--episodes <K>] [--seed <S>]\"\n"
"        \" [--out <path>]\\n\", prog);\n"
"    return 2;\n"
"}\n"
"\n"
"/* Full-token non-negative integer, or a clear diagnostic. A run\n"
" * silently defaulting a mistyped seed would be indistinguishable\n"
" * from the intended one. */\n"
"static int kflrl_arg_u64_(const char *prog, const char *flag,\n"
"                          const char *val, unsigned long long *out_v)\n"
"{\n"
"    char *end = NULL;\n"
"    if (val && val[0] >= '0' && val[0] <= '9') {\n"
"        unsigned long long v = strtoull(val, &end, 10);\n"
"        if (end && *end == '\\0') { *out_v = v; return 0; }\n"
"    }\n"
"    fprintf(stderr, \"%s: %s expects a non-negative integer,\"\n"
"            \" got `%s`\\n\", prog, flag, val ? val : \"\");\n"
"    return -1;\n"
"}\n"
"\n"
"int main(int argc, char **argv)\n"
"{\n"
"    uint32_t n_envs = 1;\n"
"    unsigned long long episodes = 1;\n"
"    uint64_t seed = 0;\n"
"    const char *out_path = NULL;\n"
"    for (int i = 1; i < argc; i++) {\n"
"        if (strcmp(argv[i], \"--envs\") == 0 ||\n"
"            strcmp(argv[i], \"--episodes\") == 0 ||\n"
"            strcmp(argv[i], \"--seed\") == 0 ||\n"
"            strcmp(argv[i], \"--out\") == 0) {\n"
"            if (i + 1 >= argc) {\n"
"                fprintf(stderr, \"%s: %s is missing its value\\n\",\n"
"                        argv[0], argv[i]);\n"
"                return kflrl_usage_(argv[0]);\n"
"            }\n"
"        }\n"
"        if (strcmp(argv[i], \"--envs\") == 0) {\n"
"            unsigned long long v = 0;\n"
"            if (kflrl_arg_u64_(argv[0], \"--envs\", argv[++i], &v)\n"
"                != 0) return 2;\n"
"            if (v > 0xFFFFFFFFull) {\n"
"                fprintf(stderr, \"%s: --envs exceeds the largest\"\n"
"                        \" supported environment count\\n\", argv[0]);\n"
"                return 2;\n"
"            }\n"
"            n_envs = (uint32_t)v;\n"
"        } else if (strcmp(argv[i], \"--episodes\") == 0) {\n"
"            if (kflrl_arg_u64_(argv[0], \"--episodes\", argv[++i],\n"
"                               &episodes) != 0) return 2;\n"
"        } else if (strcmp(argv[i], \"--seed\") == 0) {\n"
"            unsigned long long v = 0;\n"
"            if (kflrl_arg_u64_(argv[0], \"--seed\", argv[++i], &v)\n"
"                != 0) return 2;\n"
"            seed = (uint64_t)v;\n"
"        } else if (strcmp(argv[i], \"--out\") == 0) {\n"
"            out_path = argv[++i];\n"
"        }\n", out);
    /* Form-argument overrides keep their batch spelling. */
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_ARG || (c->flags & KFL_NF_READONLY)) continue;
        if (c->position.kind == KFLV_STR) {
            fprintf(out,
"        else if (strcmp(argv[i], \"--%s\") == 0 && i + 1 < argc) {\n"
"            kfl_arg_%s = argv[++i];\n"
"        }\n", c->name, c->name);
        } else if (c->position.kind == KFLV_FLOAT) {
            fprintf(out,
"        else if (strcmp(argv[i], \"--%s\") == 0 && i + 1 < argc) {\n"
"            kfl_arg_%s = strtod(argv[++i], NULL);\n"
"        }\n", c->name, c->name);
        } else if (c->position.kind == KFLV_INT &&
                   (c->position.u.i < 0 || c->position.u.i > 1)) {
            fprintf(out,
"        else if (strcmp(argv[i], \"--%s\") == 0 && i + 1 < argc) {\n"
"            kfl_arg_%s = (int)strtol(argv[++i], NULL, 10);\n"
"        }\n", c->name, c->name);
        } else {
            fprintf(out,
"        else if (strcmp(argv[i], \"--%s\") == 0) {\n"
"            kfl_arg_%s = true;\n"
"        } else if (strcmp(argv[i], \"--no-%s\") == 0) {\n"
"            kfl_arg_%s = false;\n"
"        }\n", c->name, c->name, c->name, c->name);
        }
    }
    fputs(
"        else {\n"
"            fprintf(stderr, \"%s: unknown argument `%s`\\n\",\n"
"                    argv[0], argv[i]);\n"
"            return kflrl_usage_(argv[0]);\n"
"        }\n"
"    }\n"
"    if (n_envs == 0) {\n"
"        fprintf(stderr, \"%s: --envs must be at least 1\\n\", argv[0]);\n"
"        return 2;\n"
"    }\n"
"\n"
"    /* An episode that can never end would run forever; the batch\n"
"     * entry refuses it, while the shared object accepts it for\n"
"     * consumers that reset externally. */\n"
"    if (!KFLRL_HAS_TERMINATED && kflrl_horizon_() == 0) {\n"
"        fprintf(stderr,\n"
"            \"%s: this program's episode can never end (no positive\"\n"
"            \" horizon and no termination condition); refusing to\"\n"
"            \" run\\n\", argv[0]);\n"
"        return 1;\n"
"    }\n"
"\n"
"    K26RlEnv *env = NULL;\n"
"    K26RlStatus st = k26rl_env_create(seed, n_envs, &env);\n"
"    if (st != K26RL_OK) {\n"
"        fprintf(stderr, \"%s: create failed: %s\\n\", argv[0],\n"
"                k26rl_status_str(st));\n"
"        return 1;\n"
"    }\n"
"    if (out_path) {\n"
"        st = k26rl_env_output(env, out_path);\n"
"        if (st != K26RL_OK) {\n"
"            fprintf(stderr, \"%s: output failed: %s\\n\", argv[0],\n"
"                    k26rl_status_str(st));\n"
"            k26rl_env_destroy(env);\n"
"            return 1;\n"
"        }\n"
"    }\n"
"\n"
"    double *act = NULL;\n"
"    if (KFLRL_ACT_TOTAL > 0) {\n"
"        act = (double *)malloc(\n"
"            sizeof(double) * (size_t)n_envs * KFLRL_ACT_TOTAL);\n"
"        if (!act) {\n"
"            k26rl_env_destroy(env);\n"
"            return 1;\n"
"        }\n"
"        for (uint32_t e = 0; e < n_envs; e++) {\n"
"            kflrl_act_defaults_(act + (size_t)e * KFLRL_ACT_TOTAL);\n"
"        }\n"
"    }\n"
"    unsigned long long *done = (unsigned long long *)calloc(\n"
"        n_envs, sizeof(unsigned long long));\n"
"    uint32_t *fl = (uint32_t *)calloc(n_envs, sizeof(uint32_t));\n"
"    if (!done || !fl) {\n"
"        free(act); free(done); free(fl);\n"
"        k26rl_env_destroy(env);\n"
"        return 1;\n"
"    }\n"
"\n"
"    for (;;) {\n"
"        int all_done = 1;\n"
"        for (uint32_t e = 0; e < n_envs; e++) {\n"
"            if (done[e] < episodes) all_done = 0;\n"
"        }\n"
"        if (all_done) break;\n"
"        st = k26rl_env_step(env, act);\n"
"        if (st != K26RL_OK) {\n"
"            fprintf(stderr, \"%s: step failed: %s\\n\", argv[0],\n"
"                    k26rl_status_str(st));\n"
"            free(act); free(done); free(fl);\n"
"            k26rl_env_destroy(env);\n"
"            return 1;\n"
"        }\n"
"        (void)k26rl_env_flags(env, fl);\n"
"        for (uint32_t e = 0; e < n_envs; e++) {\n"
"            if (fl[e] & (K26RL_FLAG_TERMINATED | K26RL_FLAG_TRUNCATED\n"
"                         | K26RL_FLAG_FAULT)) {\n"
"                done[e]++;\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    k26rl_env_destroy(env);\n"
"    free(act);\n"
"    free(done);\n"
"    free(fl);\n"
"    return 0;\n"
"}\n"
"\n"
"#endif /* KFLC_RL_BATCH_MAIN */\n", out);
}

/* ---- Entry ---------------------------------------------------------- */

static int kfl_emit_rl_cxx_inner_(FILE *out, const KflcNode *form,
                                  KflcDiag *diag);

int kfl_emit_rl_cxx(FILE *out, const KflcNode *form, KflcDiag *diag)
{
    if (!form || form->kind != KFLN_FORM) {
        kflc_diag_errorf(diag, 0, "emit: not a form node");
        return 1;
    }
    /* Every expression in this translation unit is environment code
     * (reward, termination, on_step, prefix), so float literals are
     * pinned for the whole emission; the flag is cleared on every
     * return so 3.1 emission never sees it. */
    kfl_expr_set_float_pin(1);
    int rl_rc_ = kfl_emit_rl_cxx_inner_(out, form, diag);
    kfl_expr_set_float_pin(0);
    return rl_rc_;
}

static int kfl_emit_rl_cxx_inner_(FILE *out, const KflcNode *form,
                                  KflcDiag *diag)
{

    KflcArena *arena = kflc_arena_create();
    RlModel m;
    if (rl_collect_(&m, form, arena, diag)) {
        kflc_arena_release(arena);
        return 1;
    }

    /* User fn table for expression resolution. */
    int n_user_fns = 0;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_FN) n_user_fns++;
    }
    KflcExprFn *user_fn_arr = NULL;
    if (n_user_fns > 0) {
        user_fn_arr = (KflcExprFn *)kflc_arena_alloc(
            arena, sizeof(KflcExprFn) * (size_t)n_user_fns);
        int idx = 0;
        for (const KflcNode *c = form->children; c; c = c->next) {
            if (c->kind != KFLN_FN) continue;
            int arity = 0;
            for (const KflcNode *a = c->children; a; a = a->next) {
                if (a->kind == KFLN_FN_ARG) arity++;
            }
            user_fn_arr[idx].name  = c->name;
            user_fn_arr[idx].arity = arity;
            idx++;
        }
    }

    /* Expression context over form arguments alone, for program
     * parameters and distribution arguments. */
    KflcExprBinding *arg_live = NULL;
    int arg_n = 0, arg_cap = 0;
    rl_collect_form_args_(form, arena, &arg_live, &arg_n, &arg_cap);
    KflcExprCtx arg_ctx;
    memset(&arg_ctx, 0, sizeof arg_ctx);
    arg_ctx.bindings   = arg_live;
    arg_ctx.n_bindings = arg_n;
    arg_ctx.fns        = user_fn_arr;
    arg_ctx.n_fns      = n_user_fns;
    arg_ctx.form       = form;

    if (rl_emit_prologue_(out, &m, form, diag)) return 1;
    rl_emit_form_args_(out, form);
    if (rl_emit_user_fns_(out, form, arena, user_fn_arr, n_user_fns,
                          diag) ||
        rl_emit_params_(out, &m, &arg_ctx, diag) ||
        rl_emit_build_world_(out, &m, form, arena, user_fn_arr,
                             n_user_fns, diag) ||
        rl_emit_apply_draws_(out, &m, &arg_ctx, diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    if (rl_emit_observe_(out, &m, diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    if (rl_emit_on_step_(out, &m, form, arena, user_fn_arr, n_user_fns,
                         diag) ||
        rl_emit_objective_(out, &m, form, arena, user_fn_arr, n_user_fns,
                           diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    rl_emit_env_core_(out);
    rl_emit_batch_main_(out, form);

    kflc_arena_release(arena);
    return diag->errors ? 1 : 0;
}
