/* kflc: semantic checks beyond the parser (kflc_check).
 *
 * Grammar 3.2 adds reinforcement learning constructs to `fn world`
 * bodies: episode / action / on_step / objective blocks, the
 * `observe ... as <name>` channel clause, and distribution
 * expressions (uniform / normal) in astro_body attribute values and
 * episode reset lines. Code emission for these arrives in a later
 * compiler pass, so `kflc --check` routes RL programs through this
 * walk instead of the emit-through-/dev/null pass it keeps for
 * Grammar 3.1 programs.
 *
 * Classification: a world containing any RL construct is an RL world.
 * A distribution call counts as a construct only in its two valid
 * positions, and never when the program declares its own function by
 * that name, so Grammar 3.1 programs (including ones with a function
 * named `uniform` or `normal`) are unaffected.
 */

#include "kflc.h"
#include "internal.h"

#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ---- Small helpers ------------------------------------------------ */

static int is_dist_name_(const char *s)
{
    return s && (strcmp(s, "uniform") == 0 || strcmp(s, "normal") == 0);
}

/* A user-declared `fn <name>` shadows the distribution reading of
 * `<name>` everywhere in the program (no new global names in 3.2). */
static int form_declares_fn_(const KflcNode *form, const char *name)
{
    if (!form || !name) return 0;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_FN) continue;
        if (c->name && strcmp(c->name, name) == 0) return 1;
    }
    return 0;
}

static const KflcAttr *node_attr_(const KflcNode *n, const char *key)
{
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0) return a;
    }
    return NULL;
}

static const char *observe_as_name_(const KflcNode *n)
{
    const KflcAttr *a = node_attr_(n, "as");
    if (a && a->value.kind == KFLV_IDENT && a->value.u.s) return a->value.u.s;
    return NULL;
}

/* Word-boundary scan of raw astro_body attribute text for a call to
 * `name` (e.g. "uniform("). Lexical on purpose: astro_body values are
 * stored verbatim and only re-parsed at emit time, and classification
 * must not depend on the emitter. */
static int text_has_call_(const char *text, const char *name)
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

/* astro_body statement whose attribute text uses an unshadowed
 * distribution call. This is the one valid distribution position that
 * can make a world RL on its own (reset lines already sit inside an
 * episode block). */
static int astro_body_has_dist_(const KflcNode *s, const KflcNode *form)
{
    for (const KflcAttr *a = s->attrs; a; a = a->next) {
        if (a->value.kind != KFLV_IDENT || !a->value.u.s) continue;
        if (text_has_call_(a->value.u.s, "uniform") &&
            !form_declares_fn_(form, "uniform")) return 1;
        if (text_has_call_(a->value.u.s, "normal") &&
            !form_declares_fn_(form, "normal")) return 1;
    }
    return 0;
}

/* ---- RL world classification -------------------------------------- */

static int stmts_have_rl_(const KflcNode *stmts, const KflcNode *form)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        switch (s->kind) {
        case KFLN_STMT_EPISODE:
        case KFLN_STMT_EPISODE_RESET:
        case KFLN_STMT_ACTION:
        case KFLN_STMT_ON_STEP:
        case KFLN_STMT_OBJECTIVE:
        case KFLN_STMT_AGENT:
        case KFLN_STMT_ASTRO_PAYLOAD:
            return 1;
        case KFLN_STMT_OBSERVE:
            if (observe_as_name_(s)) return 1;
            break;
        case KFLN_STMT_ASTRO_BODY:
            if (astro_body_has_dist_(s, form)) return 1;
            break;
        default:
            break;
        }
        if (stmts_have_rl_(s->children, form))      return 1;
        if (stmts_have_rl_(s->else_children, form)) return 1;
    }
    return 0;
}

int kflc_form_has_rl(const KflcNode *form)
{
    if (!form || form->kind != KFLN_FORM) return 0;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_FN_WORLD) continue;
        if (stmts_have_rl_(c->children, form)) return 1;
    }
    return 0;
}

/* ---- Collection ---------------------------------------------------- */

typedef struct {
    const KflcNode **items;
    int n, cap;
} NodeList;

static void nodelist_push_(NodeList *l, const KflcNode *node, KflcArena *arena)
{
    if (l->n == l->cap) {
        int nc = l->cap ? l->cap * 2 : 4;
        const KflcNode **ni = (const KflcNode **)kflc_arena_alloc(
            arena, sizeof(*ni) * (size_t)nc);
        if (l->n > 0) memcpy(ni, l->items, sizeof(*ni) * (size_t)l->n);
        l->items = ni;
        l->cap   = nc;
    }
    l->items[l->n++] = node;
}

typedef struct {
    const char **names;
    int n, cap;
} NameList;

static void namelist_push_(NameList *l, const char *name, KflcArena *arena)
{
    if (!name) return;
    if (l->n == l->cap) {
        int nc = l->cap ? l->cap * 2 : 8;
        const char **nn = (const char **)kflc_arena_alloc(
            arena, sizeof(*nn) * (size_t)nc);
        if (l->n > 0) memcpy(nn, l->names, sizeof(*nn) * (size_t)l->n);
        l->names = nn;
        l->cap   = nc;
    }
    l->names[l->n++] = name;
}

static int namelist_has_(const NameList *l, const char *name)
{
    for (int i = 0; i < l->n; i++) {
        if (strcmp(l->names[i], name) == 0) return 1;
    }
    return 0;
}

/* Declarations, and the `agent` block each was written in, or NULL for
 * one written at world level. Names are unique within an owner and may
 * repeat across owners, which is the whole point of the block: two
 * agents may each declare a channel called `rel`. The owner lists run
 * in step with the declaration lists they qualify. */
typedef struct {
    NodeList episodes;
    NodeList on_steps;
    NodeList objectives;
    NodeList obj_owner;
    NodeList actions;
    NodeList act_owner;
    NodeList observes_as;   /* observe statements carrying `as` */
    NodeList obs_owner;
    NodeList agents;
} RlStats;

static void collect_stats_(const KflcNode *stmts, RlStats *st,
                           const KflcNode *owner, KflcArena *arena)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        const KflcNode *inner = owner;
        switch (s->kind) {
        case KFLN_STMT_EPISODE:   nodelist_push_(&st->episodes,   s, arena); break;
        case KFLN_STMT_ON_STEP:   nodelist_push_(&st->on_steps,   s, arena); break;
        case KFLN_STMT_OBJECTIVE:
            nodelist_push_(&st->objectives, s, arena);
            nodelist_push_(&st->obj_owner, owner, arena);
            break;
        case KFLN_STMT_ACTION:
            nodelist_push_(&st->actions, s, arena);
            nodelist_push_(&st->act_owner, owner, arena);
            break;
        case KFLN_STMT_AGENT:
            nodelist_push_(&st->agents, s, arena);
            inner = s;
            break;
        case KFLN_STMT_OBSERVE:
            if (observe_as_name_(s)) {
                nodelist_push_(&st->observes_as, s, arena);
                nodelist_push_(&st->obs_owner, owner, arena);
            }
            break;
        default:
            break;
        }
        collect_stats_(s->children, st, inner, arena);
        collect_stats_(s->else_children, st, inner, arena);
    }
}

static void collect_binding_names_(const KflcNode *stmts, NameList *nl,
                                   KflcArena *arena)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        if ((s->kind == KFLN_STMT_LET || s->kind == KFLN_STMT_CONST) &&
            s->name)
        {
            namelist_push_(nl, s->name, arena);
        }
        collect_binding_names_(s->children, nl, arena);
        collect_binding_names_(s->else_children, nl, arena);
    }
}

/* ---- Per-rule checks ------------------------------------------------ */

static void check_no_stepping_(const KflcNode *stmts, const char *world_name,
                               KflcDiag *diag)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        if (s->kind == KFLN_STMT_STEP || s->kind == KFLN_STMT_PROPAGATE) {
            kflc_diag_errorf(diag, s->line,
                "`%s` is not allowed in a reinforcement learning world: "
                "the stepping belongs to the episode machinery in these "
                "programs (fn world %s)",
                s->kind == KFLN_STMT_STEP ? "step" : "propagate",
                world_name);
        }
        check_no_stepping_(s->children, world_name, diag);
        check_no_stepping_(s->else_children, world_name, diag);
    }
}

static void dist_scan_expr_(const KflcExpr *e, const KflcNode *form,
                            int line, KflcDiag *diag)
{
    if (!e) return;
    switch (e->kind) {
    case KFLE_CALL:
        if (is_dist_name_(e->u.call.name) &&
            !form_declares_fn_(form, e->u.call.name))
        {
            kflc_diag_errorf(diag, line,
                "`%s(...)` is a distribution expression; distribution "
                "expressions are only valid in astro_body attribute "
                "values and episode reset lines", e->u.call.name);
            return;
        }
        for (int i = 0; i < e->u.call.n_args; i++) {
            dist_scan_expr_(e->u.call.args[i], form, line, diag);
        }
        return;
    case KFLE_UNARY:
        dist_scan_expr_(e->u.un.operand, form, line, diag);
        return;
    case KFLE_BINARY:
        dist_scan_expr_(e->u.bin.lhs, form, line, diag);
        dist_scan_expr_(e->u.bin.rhs, form, line, diag);
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            dist_scan_expr_(e->u.vec.elems[i], form, line, diag);
        }
        return;
    case KFLE_INDEX:
        dist_scan_expr_(e->u.index.base, form, line, diag);
        dist_scan_expr_(e->u.index.idx,  form, line, diag);
        return;
    default:
        return;
    }
}

/* Distribution expressions are valid in exactly two positions:
 * astro_body attribute values (raw text, skipped here) and episode
 * reset lines (their whole expression subtree is the blessed
 * position). Everything else in an RL world is scanned. */
static void check_dist_positions_(const KflcNode *stmts,
                                  const KflcNode *form, KflcDiag *diag)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        if (s->kind == KFLN_STMT_EPISODE_RESET) continue;
        dist_scan_expr_(s->expr,  form, s->line, diag);
        dist_scan_expr_(s->expr2, form, s->line, diag);
        if (s->kind != KFLN_STMT_ASTRO_BODY) {
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                dist_scan_expr_(a->expr, form, a->line, diag);
            }
        }
        check_dist_positions_(s->children, form, diag);
        check_dist_positions_(s->else_children, form, diag);
    }
}

static void resolve_expr_names_(const KflcExpr *e, const NameList *allowed,
                                const char *ctx_word, int line,
                                KflcDiag *diag)
{
    if (!e) return;
    switch (e->kind) {
    case KFLE_IDENT:
        if (e->u.ident && !namelist_has_(allowed, e->u.ident)) {
            /* A dotted name here is body state, which is addressed
             * inside on_step and nowhere else; say so rather than
             * listing the scope. */
            if (strchr(e->u.ident, '.')) {
                kflc_diag_errorf(diag, line,
                    "%s: `%s`: body state is readable and assignable "
                    "only inside an on_step block; an objective reads "
                    "state through `observe ... as` channels",
                    ctx_word, e->u.ident);
                return;
            }
            kflc_diag_errorf(diag, line,
                "%s: unknown name `%s`; readable names here are declared "
                "actions, observe-as channel components, `episode.steps`, "
                "and in-scope bindings", ctx_word, e->u.ident);
        }
        return;
    case KFLE_CALL:
        /* Call names resolve against the builtin registry at emit
         * time; only the arguments carry readable names. */
        for (int i = 0; i < e->u.call.n_args; i++) {
            resolve_expr_names_(e->u.call.args[i], allowed, ctx_word,
                                line, diag);
        }
        return;
    case KFLE_UNARY:
        resolve_expr_names_(e->u.un.operand, allowed, ctx_word, line, diag);
        return;
    case KFLE_BINARY:
        resolve_expr_names_(e->u.bin.lhs, allowed, ctx_word, line, diag);
        resolve_expr_names_(e->u.bin.rhs, allowed, ctx_word, line, diag);
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            resolve_expr_names_(e->u.vec.elems[i], allowed, ctx_word,
                                line, diag);
        }
        return;
    case KFLE_INDEX:
        resolve_expr_names_(e->u.index.base, allowed, ctx_word, line, diag);
        resolve_expr_names_(e->u.index.idx,  allowed, ctx_word, line, diag);
        return;
    default:
        return;
    }
}

/* `<agent>.<channel>`, the qualified form of a channel name. */
static const char *qualified_(KflcArena *arena, const char *agent,
                              const char *chan)
{
    size_t al = strlen(agent), cl = strlen(chan);
    char *out = (char *)kflc_arena_alloc(arena, al + 1 + cl + 1);
    memcpy(out, agent, al);
    out[al] = '.';
    memcpy(out + al + 1, chan, cl + 1);
    return out;
}

static const char *suffixed_(KflcArena *arena, const char *base,
                             const char *suffix)
{
    size_t bl = strlen(base), sl = strlen(suffix);
    char *out = (char *)kflc_arena_alloc(arena, bl + sl + 1);
    memcpy(out, base, bl);
    memcpy(out + bl, suffix, sl + 1);
    return out;
}

/* Constant-fold a horizon expression. The horizon is an episode-count
 * bound baked into the artifact's spec, so it must be decidable at
 * compile time: literals and + - * / arithmetic over them. Returns 1
 * with the value in *out, 0 when the expression is not constant. */
/* The channel suffixes an observe contributes. A line-of-sight
 * observe publishes five; an attitude observe publishes its body's
 * own orientation and rate, which is seven; a contact observe
 * publishes what the transition did, which is three; a relative
 * observe publishes a position and a velocity in the chief's frame,
 * which is six. The lists live here and at the emitter, and the gates
 * compare the published names against both. */
static const char *const OBS_SFX_LOS_[] =
    { "_dir_x", "_dir_y", "_dir_z", "_range", "_range_rate", NULL };
static const char *const OBS_SFX_ATT_[] =
    { "_quat_w", "_quat_x", "_quat_y", "_quat_z",
      "_omega_x", "_omega_y", "_omega_z", NULL };
static const char *const OBS_SFX_CON_[] =
    { "_hit", "_fraction", "_speed", NULL };
static const char *const OBS_SFX_REL_[] =
    { "_r_x", "_r_y", "_r_z", "_v_x", "_v_y", "_v_z", NULL };
/* A port observe publishes whether the contact it just made was a
 * capture, the four alignment residuals a capture is judged on, and
 * their four rates, which is nine. */
static const char *const OBS_SFX_PORT_[] =
    { "_captured", "_axial", "_lateral", "_pitchyaw", "_roll",
      "_v_axial", "_v_lateral", "_v_pitchyaw", "_v_roll", NULL };
/* A detection observe publishes whether the target was seen, the
 * continuous quantity the decision was taken on, and the geometry it
 * was taken from, which is seven. The aspect cosine is among them
 * because the signature models are aspect dependent, so an agent
 * without it cannot see why its detections come and go. */
static const char *const OBS_SFX_DETECT_[] =
    { "_detected", "_snr", "_range", "_dir_x", "_dir_y", "_dir_z",
      "_aspect", NULL };
/* An information-state observe publishes the target's state at the
 * retarded time, the range and the age of what it is looking at, with
 * an explicit validity channel, which is nine. The solver's iteration
 * count is not among them: it is a convergence diagnostic and not a
 * state of the world. */
static const char *const OBS_SFX_TRACK_[] =
    { "_valid", "_pos_x", "_pos_y", "_pos_z", "_vel_x", "_vel_y",
      "_vel_z", "_range", "_age", NULL };
/* An effector observe publishes what the last engagement of its
 * payload did. Two suffixes are common to every effector kind,
 * `_engaged` and `_effect`; the rest are the kind's own, because an
 * ablation event and an impact event share no fields. The set is
 * therefore a function of the payload's `kind=`, which is why the
 * lookup below reads the world rather than the statement alone. */
static const char *const OBS_SFX_EFF_LASER_[] =
    { "_engaged", "_effect", "_dv", "_mass_loss", "_range", "_spot",
      "_encircled", "_fluence", "_transmissivity", "_p_coupled",
      "_ignited", NULL };
static const char *const OBS_SFX_EFF_IMPACTOR_[] =
    { "_engaged", "_effect", "_hit", "_closing_speed", "_t_close",
      "_miss", "_fraction", "_cos_angle", "_penetrates",
      "_critical_diameter", "_penetration", "_energy", NULL };
static const char *const OBS_SFX_EFF_JAMMER_[] =
    { "_engaged", "_effect", "_reached", "_range", "_rcs",
      "_burn_through", "_self_signature", "_counter_range",
      "_counter_detected", NULL };
static const char *const OBS_SFX_EFF_DECOY_[] =
    { "_engaged", "_deployed", "_effect", "_reached",
      "_p_discriminated", "_range", "_dv", "_mass_loss", NULL };

static int observe_marker_(const KflcNode *n, const char *marker)
{
    if (!n) return 0;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, marker) == 0) return 1;
    }
    return 0;
}

/* Whether the observe asked for the uncorrupted values beside the
 * measured ones. A paired observe publishes each component twice, the
 * second carrying `_truth` before the component, so every name the
 * program may read is one of two per component. */
static int observe_has_truth_(const KflcNode *n)
{
    return observe_marker_(n, "truth");
}

/* The value of an attribute carried as an identifier. */
static const char *attr_ident_(const KflcNode *n, const char *key)
{
    for (const KflcAttr *a = n ? n->attrs : NULL; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0 &&
            a->value.kind == KFLV_IDENT) {
            return a->value.u.s;
        }
    }
    return NULL;
}

/* The `kind=` of the payload an effect observe names, read straight
 * off the `astro_payload` statement that declares it. The emitter
 * resolves the same thing for the channel widths; this pass needs it
 * so that the names it accepts are the names the build will publish,
 * and a program the checker accepts is one that builds. */
static const char *effect_payload_kind_(const KflcNode *world,
                                        const KflcNode *n)
{
    const char *pn = attr_ident_(n, "effect");
    if (!pn) return NULL;
    for (const KflcNode *s = world ? world->children : NULL; s; s = s->next) {
        if (s->kind != KFLN_STMT_ASTRO_PAYLOAD) continue;
        if (!s->name || strcmp(s->name, pn) != 0) continue;
        return attr_ident_(s, "kind");
    }
    return NULL;
}

static const char *const *observe_suffixes_(const KflcNode *world,
                                            const KflcNode *n)
{
    if (observe_marker_(n, "detect"))   return OBS_SFX_DETECT_;
    if (observe_marker_(n, "track"))    return OBS_SFX_TRACK_;
    if (observe_marker_(n, "effect")) {
        const char *k = effect_payload_kind_(world, n);
        if (k && strcmp(k, "impactor") == 0) return OBS_SFX_EFF_IMPACTOR_;
        if (k && strcmp(k, "jammer") == 0)   return OBS_SFX_EFF_JAMMER_;
        if (k && strcmp(k, "decoy") == 0)    return OBS_SFX_EFF_DECOY_;
        return OBS_SFX_EFF_LASER_;
    }
    if (observe_marker_(n, "port"))     return OBS_SFX_PORT_;
    if (observe_marker_(n, "contact"))  return OBS_SFX_CON_;
    if (observe_marker_(n, "attitude")) return OBS_SFX_ATT_;
    if (observe_marker_(n, "relative")) return OBS_SFX_REL_;
    return OBS_SFX_LOS_;
}

/* Push every channel name one observe publishes: the measured
 * components, and the paired truth components when it declares them.
 * Both name sets live here and at the emitter, and the gates compare
 * the published names against both. */
static void observe_push_names_(NameList *dst, const KflcNode *world,
                                const KflcNode *n,
                                const char *base, KflcArena *arena)
{
    const char *const *sfx = observe_suffixes_(world, n);
    for (int k = 0; sfx[k]; k++) {
        namelist_push_(dst, suffixed_(arena, base, sfx[k]), arena);
    }
    if (!observe_has_truth_(n)) return;
    for (int k = 0; sfx[k]; k++) {
        char t[80];
        snprintf(t, sizeof t, "_truth%s", sfx[k]);
        namelist_push_(dst, suffixed_(arena, base, t), arena);
    }
}

/* The bound on an `as` name. It is a fixed number rather than one
 * derived from the tables above: a program that compiles today was
 * written against it, so it does not move when a form adds a suffix.
 * What moves instead is the emitter's name entry, which internal.h
 * sizes to hold this bound plus the longest suffix any form derives,
 * with headroom. */
static size_t observe_as_bound_(void)
{
    /* The bound is a compatibility promise and does not follow from
     * the tables: it is fixed at 53, and the emitter's name entry is
     * sized to hold it plus the longest suffix any form derives, with
     * headroom. internal.h states that arithmetic. What is checked
     * here is the direction that could bite, that no suffix has
     * outgrown the entry. */
    size_t longest = 0;
    const char *const *lists[] = { OBS_SFX_LOS_, OBS_SFX_ATT_,
                                   OBS_SFX_CON_, OBS_SFX_REL_,
                                   OBS_SFX_PORT_, OBS_SFX_DETECT_,
                                   OBS_SFX_TRACK_, OBS_SFX_EFF_LASER_,
                                   OBS_SFX_EFF_IMPACTOR_,
                                   OBS_SFX_EFF_JAMMER_,
                                   OBS_SFX_EFF_DECOY_ };
    for (size_t i = 0; i < sizeof lists / sizeof lists[0]; i++) {
        for (int k = 0; lists[i][k]; k++) {
            /* `_truth` is what a paired channel inserts, so the
             * longest name any form can derive is a suffix plus it. */
            size_t n = strlen(lists[i][k]) + 6;
            if (n > longest) longest = n;
        }
    }
    if (KFLC_OBS_AS_MAX + longest + 1 > KFLC_OBS_NAME_MAX) {
        /* Unreachable while the arithmetic in internal.h holds; it is
         * here so that a new suffix moves the entry rather than
         * truncating a name in silence. */
        return KFLC_OBS_NAME_MAX - longest - 1;
    }
    return KFLC_OBS_AS_MAX;
}

static int horizon_const_eval_(const KflcExpr *e, double *out)
{
    if (!e) return 0;
    switch (e->kind) {
    case KFLE_INT_LIT:   *out = (double)e->u.i; return 1;
    case KFLE_FLOAT_LIT: *out = e->u.f;         return 1;
    case KFLE_UNARY: {
        double v;
        if (!horizon_const_eval_(e->u.un.operand, &v)) return 0;
        if (e->u.un.op == KFLOP_NEG) { *out = -v; return 1; }
        if (e->u.un.op == KFLOP_POS) { *out =  v; return 1; }
        return 0;
    }
    case KFLE_BINARY: {
        double a, b;
        if (!horizon_const_eval_(e->u.bin.lhs, &a)) return 0;
        if (!horizon_const_eval_(e->u.bin.rhs, &b)) return 0;
        switch (e->u.bin.op) {
        case KFLOP_ADD: *out = a + b; return 1;
        case KFLOP_SUB: *out = a - b; return 1;
        case KFLOP_MUL: *out = a * b; return 1;
        case KFLOP_DIV: if (b == 0.0) return 0; *out = a / b; return 1;
        default: return 0;
        }
    }
    default:
        return 0;
    }
}

/* ---- Per-world check ------------------------------------------------ */

static void check_world_(const KflcNode *world, const KflcNode *form,
                         KflcArena *arena, KflcDiag *diag)
{
    if (!stmts_have_rl_(world->children, form)) return;

    const char *wname = world->name ? world->name : "?";
    RlStats st;
    memset(&st, 0, sizeof st);
    collect_stats_(world->children, &st, NULL, arena);

    if (st.episodes.n == 0) {
        kflc_diag_errorf(diag, world->line,
            "fn world %s: this world uses reinforcement learning "
            "constructs and requires an `episode` block with `control_dt`",
            wname);
    }
    for (int i = 1; i < st.episodes.n; i++) {
        kflc_diag_errorf(diag, st.episodes.items[i]->line,
            "fn world %s: duplicate `episode` block (allowed at most "
            "once per world)", wname);
    }
    for (int i = 1; i < st.on_steps.n; i++) {
        kflc_diag_errorf(diag, st.on_steps.items[i]->line,
            "fn world %s: duplicate `on_step` block (allowed at most "
            "once per world)", wname);
    }
    /* At most one objective at world level. Two agents each declaring
     * their own is the ordinary multi-agent shape and is not a
     * duplicate; how many an `agent` block may hold is that block's
     * rule and is enforced where the rest of the block's contents
     * are, so that one rule has one place. */
    for (int i = 0; i < st.objectives.n; i++) {
        if (st.obj_owner.items[i] != NULL) continue;
        for (int j = 0; j < i; j++) {
            if (st.obj_owner.items[j] != NULL) continue;
            kflc_diag_errorf(diag, st.objectives.items[i]->line,
                "fn world %s: duplicate `objective` block (allowed at "
                "most once per world)", wname);
            break;
        }
    }

    /* Channel names are unique within their owner. Across owners they
     * may repeat, and the qualified form is what tells them apart. */
    for (int i = 0; i < st.actions.n; i++) {
        for (int j = 0; j < i; j++) {
            if (st.act_owner.items[i] != st.act_owner.items[j]) continue;
            const char *ni = st.actions.items[i]->name;
            const char *nj = st.actions.items[j]->name;
            if (ni && nj && strcmp(ni, nj) == 0) {
                kflc_diag_errorf(diag, st.actions.items[i]->line,
                    "action `%s`: duplicate action name", ni);
                break;
            }
        }
    }
    for (int i = 0; i < st.observes_as.n; i++) {
        for (int j = 0; j < i; j++) {
            if (st.obs_owner.items[i] != st.obs_owner.items[j]) continue;
            const char *ni = observe_as_name_(st.observes_as.items[i]);
            const char *nj = observe_as_name_(st.observes_as.items[j]);
            if (ni && nj && strcmp(ni, nj) == 0) {
                kflc_diag_errorf(diag, st.observes_as.items[i]->line,
                    "observe ... as `%s`: duplicate channel name", ni);
                break;
            }
        }
    }

    /* `episode.steps` occupies that name in the same read space an
     * on_step body addresses bodies through, so a body called
     * `episode` would make `episode.steps` ambiguous. */
    for (const KflcNode *s = world->children; s; s = s->next) {
        if (s->kind == KFLN_STMT_ASTRO_BODY && s->name &&
            strcmp(s->name, "episode") == 0)
        {
            kflc_diag_errorf(diag, s->line,
                "astro_body `episode`: the name is taken by "
                "`episode.steps` in this program's expression scope; "
                "rename the body");
        }
    }

    /* Channel names are published in the artifact's spec, and the
     * entry that carries one is sized to hold this bound plus the
     * longest suffix any form derives. The bound itself stays at 53,
     * so a program that compiles today keeps compiling however the
     * suffix set grows; internal.h states the arithmetic. */
    {
        size_t bound = observe_as_bound_();
        for (int i = 0; i < st.observes_as.n; i++) {
            const char *ni = observe_as_name_(st.observes_as.items[i]);
            if (ni && strlen(ni) > bound) {
                kflc_diag_errorf(diag, st.observes_as.items[i]->line,
                    "observe ... as `%s`: channel name is longer than %d "
                    "bytes, which is the most a declared channel name "
                    "may be", ni, (int)bound);
            }
        }
    }

    /* Actions, derived observation components, world bindings, and
     * form arguments all share the evaluator scope (and the generated
     * code's namespace), so a name may appear in at most one of the
     * sets the episode machinery introduces, and neither of the two
     * pre-existing sets may reuse one of those names: an action or a
     * component silently shadowing a user's `let` or `arg` would read
     * back the wrong value with no diagnostic at all. */
    NameList comps;
    memset(&comps, 0, sizeof comps);
    for (int j = 0; j < st.observes_as.n; j++) {
        const char *base = observe_as_name_(st.observes_as.items[j]);
        if (!base) continue;
        observe_push_names_(&comps, world, st.observes_as.items[j], base,
                            arena);
    }
    /* An action and an observation component share a scope only when
     * one owner declares both, since it is the owner's own names that
     * an expression reads unqualified. */
    for (int i = 0; i < st.actions.n; i++) {
        const char *an = st.actions.items[i]->name;
        if (!an) continue;
        for (int j = 0; j < st.observes_as.n; j++) {
            if (st.act_owner.items[i] != st.obs_owner.items[j]) continue;
            const char *base = observe_as_name_(st.observes_as.items[j]);
            if (!base) continue;
            NameList own;
            memset(&own, 0, sizeof own);
            observe_push_names_(&own, world, st.observes_as.items[j], base,
                                arena);
            for (int k = 0; k < own.n; k++) {
                if (strcmp(an, own.names[k]) != 0) continue;
                kflc_diag_errorf(diag, st.actions.items[i]->line,
                    "action `%s`: name collides with the `%s` component "
                    "of an observation channel; rename one",
                    an, own.names[k]);
            }
        }
    }
    /* Top-level bindings only: those are the ones the evaluator scope
     * captures, so only they can be silently shadowed. A binding
     * inside a nested block was never readable there and may share a
     * name freely. */
    NameList wbind;
    memset(&wbind, 0, sizeof wbind);
    for (const KflcNode *s = world->children; s; s = s->next) {
        if ((s->kind == KFLN_STMT_LET || s->kind == KFLN_STMT_CONST) &&
            s->name)
        {
            namelist_push_(&wbind, s->name, arena);
        }
    }
    NameList argn;
    memset(&argn, 0, sizeof argn);
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_ARG && c->name) {
            namelist_push_(&argn, c->name, arena);
        }
    }
    for (int i = 0; i < st.actions.n; i++) {
        const char *an = st.actions.items[i]->name;
        if (!an) continue;
        for (int j = 0; j < wbind.n; j++) {
            if (strcmp(an, wbind.names[j]) == 0) {
                kflc_diag_errorf(diag, st.actions.items[i]->line,
                    "action `%s`: name collides with a world binding of "
                    "the same name; rename one", an);
                break;
            }
        }
        for (int j = 0; j < argn.n; j++) {
            if (strcmp(an, argn.names[j]) == 0) {
                kflc_diag_errorf(diag, st.actions.items[i]->line,
                    "action `%s`: name collides with form argument `%s`; "
                    "rename one", an, an);
                break;
            }
        }
    }
    for (int i = 0; i < comps.n; i++) {
        for (int j = 0; j < wbind.n; j++) {
            if (strcmp(comps.names[i], wbind.names[j]) == 0) {
                kflc_diag_errorf(diag, world->line,
                    "observation channel component `%s` collides with a "
                    "world binding of the same name; rename one",
                    comps.names[i]);
                break;
            }
        }
        for (int j = 0; j < argn.n; j++) {
            if (strcmp(comps.names[i], argn.names[j]) == 0) {
                kflc_diag_errorf(diag, world->line,
                    "observation channel component `%s` collides with "
                    "form argument `%s`; rename one",
                    comps.names[i], comps.names[i]);
                break;
            }
        }
    }

    check_no_stepping_(world->children, wname, diag);
    check_dist_positions_(world->children, form, diag);

    /* Names readable in `terminated when` / `reward` / `terminal`:
     * declared actions, the four components of each observe-as
     * channel, `episode.steps`, and ordinary in-scope names (world
     * bindings, form args, the boolean literals). */
    NameList allowed;
    memset(&allowed, 0, sizeof allowed);
    namelist_push_(&allowed, "episode.steps", arena);
    namelist_push_(&allowed, "true",  arena);
    namelist_push_(&allowed, "false", arena);
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_ARG && c->name) {
            namelist_push_(&allowed, c->name, arena);
        }
    }
    collect_binding_names_(world->children, &allowed, arena);
    for (int i = 0; i < st.actions.n; i++) {
        namelist_push_(&allowed, st.actions.items[i]->name, arena);
    }
    for (int i = 0; i < st.observes_as.n; i++) {
        const char *base = observe_as_name_(st.observes_as.items[i]);
        if (!base) continue;
        observe_push_names_(&allowed, world, st.observes_as.items[i], base,
                            arena);
    }
    /* The qualified forms, `<agent>.<channel>`, which are dotted names
     * that are not body state. Which of the two forms an expression in
     * a given position may use is decided where the agent set is
     * known; what this list settles is that a qualified name is a name
     * at all, so that one is not reported here as body state. */
    for (int i = 0; i < st.actions.n; i++) {
        const KflcNode *own = st.act_owner.items[i];
        if (!own || !own->name || !st.actions.items[i]->name) continue;
        namelist_push_(&allowed,
                       qualified_(arena, own->name,
                                  st.actions.items[i]->name), arena);
    }
    for (int i = 0; i < st.observes_as.n; i++) {
        const KflcNode *own  = st.obs_owner.items[i];
        const char     *base = observe_as_name_(st.observes_as.items[i]);
        if (!own || !own->name || !base) continue;
        NameList own_comps;
        memset(&own_comps, 0, sizeof own_comps);
        observe_push_names_(&own_comps, world, st.observes_as.items[i],
                            base, arena);
        for (int k = 0; k < own_comps.n; k++) {
            namelist_push_(&allowed,
                           qualified_(arena, own->name, own_comps.names[k]),
                           arena);
        }
    }

    for (int i = 0; i < st.episodes.n; i++) {
        const KflcAttr *tw = node_attr_(st.episodes.items[i],
                                        "terminated_when");
        if (tw) {
            resolve_expr_names_(tw->expr, &allowed, "terminated when",
                                tw->line, diag);
        }
    }
    for (int i = 0; i < st.objectives.n; i++) {
        const KflcAttr *rw = node_attr_(st.objectives.items[i], "reward");
        const KflcAttr *tm = node_attr_(st.objectives.items[i], "terminal");
        if (rw) resolve_expr_names_(rw->expr, &allowed, "reward",
                                    rw->line, diag);
        if (tm) resolve_expr_names_(tm->expr, &allowed, "terminal",
                                    tm->line, diag);
    }

    /* Horizon validity and termination reachability. The horizon is a
     * step-count bound published in the artifact's spec, so it must
     * const-evaluate to a non-negative integer at compile time; a
     * negative or fractional bound has no meaning there. With no
     * positive horizon and no `terminated when`, nothing ever ends
     * the episode: warning, not error, since the program still
     * checks clean. */
    if (st.episodes.n > 0) {
        const KflcNode *ep = st.episodes.items[0];
        /* The subdivision of a control period. It changes the physics,
         * so it is part of the program's identity and is published in
         * the spec; it must therefore const-evaluate to a positive
         * whole number, like the horizon beside it. */
        const KflcAttr *sb = node_attr_(ep, "substeps");
        if (sb && sb->expr) {
            double sv = 0.0;
            if (!horizon_const_eval_(sb->expr, &sv)) {
                kflc_diag_errorf(diag, sb->line,
                    "episode: `substeps` must be a compile-time constant "
                    "expression");
            } else if (!(sv >= 1.0) || sv != (double)(long)sv) {
                kflc_diag_errorf(diag, sb->line,
                    "episode: `substeps` must be a whole number of at "
                    "least 1 (evaluates to %g)", sv);
            } else if (sv > 65536.0) {
                kflc_diag_errorf(diag, sb->line,
                    "episode: `substeps` exceeds 65536 (evaluates to %g); "
                    "a subdivision that fine costs more than it buys",
                    sv);
            }
        }
        /* The contact resolution. Both coefficients are physical
         * quantities with ranges, and a value outside one would not
         * misbehave visibly: a restitution above one adds energy at
         * every impact and a negative friction drives the surfaces
         * apart, either of which reads as an unstable task rather
         * than as a typed number out of range. They are therefore
         * refused here, with the value named, rather than clamped. */
        const KflcAttr *cr = node_attr_(ep, "restitution");
        if (cr && cr->expr) {
            double rv = 0.0;
            if (!horizon_const_eval_(cr->expr, &rv)) {
                kflc_diag_errorf(diag, cr->line,
                    "episode: `contact bounce restitution` must be a "
                    "compile-time constant expression");
            } else if (!isfinite(rv) || !(rv >= 0.0) || !(rv <= 1.0)) {
                /* %.17g rather than %g: six significant digits reports
                 * 1.0000001 as "is 1", so a diagnostic required to
                 * name the offending value would name a legal one
                 * instead. The requirement is met by a representation
                 * that cannot round the offence away. */
                kflc_diag_errorf(diag, cr->line,
                    "episode: `contact bounce restitution` is %.17g; it "
                    "is the fraction of the approach speed a surface "
                    "returns and lies in the closed interval 0 to 1",
                    rv);
            }
        }
        const KflcAttr *cf = node_attr_(ep, "friction");
        if (cf && cf->expr) {
            double fv = 0.0;
            if (!horizon_const_eval_(cf->expr, &fv)) {
                kflc_diag_errorf(diag, cf->line,
                    "episode: `contact bounce friction` must be a "
                    "compile-time constant expression");
            } else if (!isfinite(fv) || !(fv >= 0.0)) {
                /* isfinite first: an infinity satisfies `>= 0.0` and
                 * would otherwise be accepted, but the declared range
                 * is the half-open interval from zero and does not
                 * contain it. The restitution above was caught only
                 * by its upper bound, which is luck rather than a
                 * check. */
                kflc_diag_errorf(diag, cf->line,
                    "episode: `contact bounce friction` is %.17g; a "
                    "friction coefficient opposes sliding, is never "
                    "negative, and is a finite number", fv);
            }
        }

        const KflcAttr *hz = node_attr_(ep, "horizon");
        const KflcAttr *tw = node_attr_(ep, "terminated_when");
        double hv = 0.0;
        int hz_bad = 0;
        if (hz && hz->expr) {
            if (!horizon_const_eval_(hz->expr, &hv)) {
                kflc_diag_errorf(diag, hz->line,
                    "episode: `horizon` must be a compile-time constant "
                    "expression");
                hz_bad = 1;
            } else if (hv < 0.0) {
                kflc_diag_errorf(diag, hz->line,
                    "episode: `horizon` must be non-negative (evaluates "
                    "to %g)", hv);
                hz_bad = 1;
            } else if (hv > 4294967295.0) {
                kflc_diag_errorf(diag, hz->line,
                    "episode: `horizon` exceeds the largest supported "
                    "step count (evaluates to %g)", hv);
                hz_bad = 1;
            } else if (hv != (double)(unsigned long long)hv) {
                kflc_diag_errorf(diag, hz->line,
                    "episode: `horizon` must be a whole number of steps "
                    "(evaluates to %g)", hv);
                hz_bad = 1;
            }
        }
        if (!tw && !hz_bad && (!hz || hv <= 0.0)) {
            kflc_diag_warnf(diag, ep->line,
                "episode: without a positive `horizon` or a "
                "`terminated when` condition the episode can never end");
        }
    }
}

/* ---- Entry point ---------------------------------------------------- */

int kflc_check(const KflcNode *form, KflcDiag *diag)
{
    if (!form || form->kind != KFLN_FORM) return 0;
    int errs_before = diag ? diag->errors : 0;
    KflcArena *arena = kflc_arena_create();
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind != KFLN_FN_WORLD) continue;
        check_world_(c, form, arena, diag);
    }
    kflc_arena_release(arena);
    return (diag && diag->errors != errs_before) ? 1 : 0;
}
