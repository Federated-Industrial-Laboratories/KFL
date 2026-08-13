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

#include <ctype.h>
#include <stdio.h>
#include <string.h>

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

typedef struct {
    NodeList episodes;
    NodeList on_steps;
    NodeList objectives;
    NodeList actions;
    NodeList observes_as;   /* observe statements carrying `as` */
} RlStats;

static void collect_stats_(const KflcNode *stmts, RlStats *st,
                           KflcArena *arena)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        switch (s->kind) {
        case KFLN_STMT_EPISODE:   nodelist_push_(&st->episodes,   s, arena); break;
        case KFLN_STMT_ON_STEP:   nodelist_push_(&st->on_steps,   s, arena); break;
        case KFLN_STMT_OBJECTIVE: nodelist_push_(&st->objectives, s, arena); break;
        case KFLN_STMT_ACTION:    nodelist_push_(&st->actions,    s, arena); break;
        case KFLN_STMT_OBSERVE:
            if (observe_as_name_(s)) nodelist_push_(&st->observes_as, s, arena);
            break;
        default:
            break;
        }
        collect_stats_(s->children, st, arena);
        collect_stats_(s->else_children, st, arena);
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

static const char *suffixed_(KflcArena *arena, const char *base,
                             const char *suffix)
{
    size_t bl = strlen(base), sl = strlen(suffix);
    char *out = (char *)kflc_arena_alloc(arena, bl + sl + 1);
    memcpy(out, base, bl);
    memcpy(out + bl, suffix, sl + 1);
    return out;
}

/* Constant-zero horizon: `horizon 0` (or 0.0) can never end the
 * episode any more than an absent horizon can. */
static int horizon_is_zero_(const KflcAttr *hz)
{
    if (!hz || !hz->expr) return 0;
    if (hz->expr->kind == KFLE_INT_LIT   && hz->expr->u.i == 0)   return 1;
    if (hz->expr->kind == KFLE_FLOAT_LIT && hz->expr->u.f == 0.0) return 1;
    return 0;
}

/* ---- Per-world check ------------------------------------------------ */

static void check_world_(const KflcNode *world, const KflcNode *form,
                         KflcArena *arena, KflcDiag *diag)
{
    if (!stmts_have_rl_(world->children, form)) return;

    const char *wname = world->name ? world->name : "?";
    RlStats st;
    memset(&st, 0, sizeof st);
    collect_stats_(world->children, &st, arena);

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
    for (int i = 1; i < st.objectives.n; i++) {
        kflc_diag_errorf(diag, st.objectives.items[i]->line,
            "fn world %s: duplicate `objective` block (allowed at most "
            "once per world)", wname);
    }

    for (int i = 0; i < st.actions.n; i++) {
        for (int j = 0; j < i; j++) {
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
            const char *ni = observe_as_name_(st.observes_as.items[i]);
            const char *nj = observe_as_name_(st.observes_as.items[j]);
            if (ni && nj && strcmp(ni, nj) == 0) {
                kflc_diag_errorf(diag, st.observes_as.items[i]->line,
                    "observe ... as `%s`: duplicate channel name", ni);
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
        namelist_push_(&allowed, suffixed_(arena, base, "_dir_x"), arena);
        namelist_push_(&allowed, suffixed_(arena, base, "_dir_y"), arena);
        namelist_push_(&allowed, suffixed_(arena, base, "_dir_z"), arena);
        namelist_push_(&allowed, suffixed_(arena, base, "_range"), arena);
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

    /* Termination reachability: with no positive horizon and no
     * `terminated when`, nothing ever ends the episode. Warning, not
     * error: the program still checks clean. */
    if (st.episodes.n > 0) {
        const KflcNode *ep = st.episodes.items[0];
        const KflcAttr *hz = node_attr_(ep, "horizon");
        const KflcAttr *tw = node_attr_(ep, "terminated_when");
        if (!tw && (!hz || horizon_is_zero_(hz))) {
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
