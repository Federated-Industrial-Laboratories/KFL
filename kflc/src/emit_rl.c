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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* The six scalar state keys astro_body, reset lines, and on_step
 * assignments share. Order is the key index used by the state
 * accessors; membership is what the other callers ask about. */
static const char *const RL_STATE_KEYS_[6] = {
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"
};

/* The published observer-mode value of an as-bound observe. The
 * grammar's default when no mode= attribute is given is the runtime's
 * default, astrometric, which is what the observation path selects. */
static uint16_t rl_observe_mode_(const KflcNode *n)
{
    if (!n) return 1;
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
    if (!k) return 0;
    for (int i = 0; i < 6; i++) {
        if (strcmp(k, RL_STATE_KEYS_[i]) == 0) return 1;
    }
    return 0;
}

/* The components one `observe ... as` contributes, in observation
 * vector order. One table, so the width and the names cannot drift
 * apart between the spec, the scope prelude, and the emitted
 * recompute. */
#define RL_OBS_COMPS 5
static const char *const RL_OBS_COMP_[RL_OBS_COMPS] = {
    "_dir_x", "_dir_y", "_dir_z", "_range", "_range_rate"
};

/* ---- Program model -------------------------------------------------- */

typedef struct {
    const KflcNode *body;      /* KFLN_STMT_ASTRO_BODY */
    int             index;     /* world body index (source order) */
} RlBody;

typedef struct {
    int             body;      /* index into bodies[] */
    const KflcAttr *attr;      /* the distribution-valued attribute */
    KflcExpr       *dist;      /* parsed uniform/normal call */
    int             channel;   /* class 0x0002 channel */
} RlDrParam;

/* One (body, state key) pair an on_step body reads or assigns. */
typedef struct {
    int body;      /* index into the model's bodies[] */
    int key;       /* index into RL_STATE_KEYS_ */
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
        for (int c = 0; c < RL_OBS_COMPS; c++) {
            if (strcmp(name + bl, RL_OBS_COMP_[c]) == 0) return 1;
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
        }
    }

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
        for (int c = 0; c < RL_OBS_COMPS; c++) {
            rl_emit_indent_(out, indent);
            fprintf(out,
                "const double %s%s = _kfl_obs_v[%d]; (void)%s%s;\n",
                base, RL_OBS_COMP_[c], i * RL_OBS_COMPS + c,
                base, RL_OBS_COMP_[c]);
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
        for (int c = 0; c < RL_OBS_COMPS; c++) {
            size_t bl = strlen(base), sl = strlen(RL_OBS_COMP_[c]);
            char *nm = (char *)kflc_arena_alloc(arena, bl + sl + 1);
            memcpy(nm, base, bl);
            memcpy(nm + bl, RL_OBS_COMP_[c], sl + 1);
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
        char axis = key[4];             /* x, y, or z */
        if (strncmp(key, "pos_", 4) == 0) {
            rl_emit_indent_(out, indent);
            fprintf(out, "%spos.s%c = 0; %spos.l%c = (%s); "
                         "k26astro_pos_normalise(&%spos);\n",
                    lv, axis, lv, axis, value_text, lv);
        } else {
            rl_emit_indent_(out, indent);
            fprintf(out, "%svel.%c = (%s);\n", lv, axis, value_text);
        }
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

static void rl_emit_prologue_(FILE *out, const RlModel *m,
                              const KflcNode *form)
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
        "#include <k26astro_body/body.h>\n"
        "#include <k26astro_core/pos.h>\n"
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
        "#define KFLRL_MAGIC 0x4b524c45u\n"
        "\n",
        m->n_bodies, m->n_observes * RL_OBS_COMPS, m->n_actions,
        m->n_resets, m->n_dr, m->n_resets + m->n_dr,
        m->n_wscal, m->terminated_when ? 1 : 0);

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
            for (int c = 0; c < RL_OBS_COMPS; c++) {
                fprintf(out, "    \"%s%s\",\n", base, RL_OBS_COMP_[c]);
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
            for (int c = 0; c < RL_OBS_COMPS; c++) {
                fprintf(out, "    %u,\n", (unsigned)md);
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
    fputs("static int kflrl_body_idx_[KFLRL_N_BODIES > 0 ? "
          "KFLRL_N_BODIES : 1];\n\n", out);
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
          "                              double *_kfl_dr0)\n"
          "{\n"
          "    const uint32_t _kfl_ep = 0;\n"
          "    (void)_kfl_key; (void)_kfl_envi; (void)_kfl_ep; "
          "(void)_kfl_wscal; (void)_kfl_dr0;\n", out);

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
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        switch (s->kind) {
        case KFLN_STMT_EPISODE:
        case KFLN_STMT_ACTION:
        case KFLN_STMT_ON_STEP:
        case KFLN_STMT_OBJECTIVE:
            continue;
        case KFLN_STMT_OBSERVE:
            if (rl_observe_as_(s)) continue;   /* channel, not a print */
            (void)kfl_emit_stmt(out, s, &ctx, diag, 4);
            continue;
        case KFLN_STMT_ASTRO_BODY: {
            fputs("    {\n"
                  "        K26AstroBody _kfl_b; "
                  "k26astro_body_init(&_kfl_b);\n", out);
            fprintf(out,
                "        snprintf(_kfl_b.name, sizeof _kfl_b.name, "
                "\"%%s\", \"%s\");\n", s->name ? s->name : "_anon");
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                if (!a->name) continue;
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
                "        kflrl_body_idx_[%d] = _kfl_body_%s_idx;\n"
                "    }\n",
                s->name, s->name, body_i, s->name);
            body_i++;
            continue;
        }
        default:
            (void)kfl_emit_stmt(out, s, &ctx, diag, 4);
            continue;
        }
    }
    kfl_emit_stmt_drain_root(out, 4);
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
static void rl_emit_observe_(FILE *out, const RlModel *m)
{
    fputs("static void kflrl_observe_(K26AstroWorld *world, "
          "double *out_v)\n"
          "{\n"
          "    (void)world; (void)out_v;\n", out);
    for (int i = 0; i < m->n_observes; i++) {
        const KflcNode *s = m->observes[i];
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
            i * RL_OBS_COMPS, i * RL_OBS_COMPS + 1,
            i * RL_OBS_COMPS + 2, i * RL_OBS_COMPS + 3,
            i * RL_OBS_COMPS + 4);
    }
    fputs("}\n\n", out);
}

/* ---- Statements and calls the stepping path forbids ----------------- */

/* A sweep follows calls into user fn bodies, so a forbidden statement
 * cannot reach the stepping path one indirection away. Two sweeps
 * share the walk: RL_SWEEP_PRINT for I/O, RL_SWEEP_IMPURE for a call
 * to a builtin that is not marked pure. */
#define RL_SWEEP_PRINT   0
#define RL_SWEEP_IMPURE  1
#define RL_SWEEP_MAX_FNS 64

typedef struct {
    const KflcNode *form;
    int             kind;              /* RL_SWEEP_* */
    const char     *found;             /* offending name, when found */
    const char     *via;               /* user fn it was reached through */
    int             line;              /* line of the offending statement */
    const char     *seen[RL_SWEEP_MAX_FNS];
    int             n_seen;
} RlSweep;

static int rl_sweep_stmts_(RlSweep *sw, const KflcNode *stmts,
                           const char *via);

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

static int rl_sweep_expr_(RlSweep *sw, const KflcExpr *e, const char *via)
{
    if (!e) return 0;
    switch (e->kind) {
    case KFLE_UNARY:
        return rl_sweep_expr_(sw, e->u.un.operand, via);
    case KFLE_BINARY:
        return rl_sweep_expr_(sw, e->u.bin.lhs, via) ||
               rl_sweep_expr_(sw, e->u.bin.rhs, via);
    case KFLE_INDEX:
        return rl_sweep_expr_(sw, e->u.index.base, via) ||
               rl_sweep_expr_(sw, e->u.index.idx, via);
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            if (rl_sweep_expr_(sw, e->u.vec.elems[i], via)) return 1;
        }
        return 0;
    case KFLE_CALL: {
        for (int i = 0; i < e->u.call.n_args; i++) {
            if (rl_sweep_expr_(sw, e->u.call.args[i], via)) return 1;
        }
        const char *nm = e->u.call.name;
        if (!nm) return 0;
        if (sw->kind == RL_SWEEP_IMPURE && kflc_builtin_known(nm) &&
            !kflc_builtin_is_pure(nm)) {
            sw->found = nm;
            sw->via   = via;
            return 1;
        }
        const KflcNode *fn = rl_find_user_fn_(sw->form, nm);
        if (!fn) return 0;
        for (int i = 0; i < sw->n_seen; i++) {
            if (strcmp(sw->seen[i], nm) == 0) return 0;   /* recursion */
        }
        if (sw->n_seen >= RL_SWEEP_MAX_FNS) return 0;
        sw->seen[sw->n_seen++] = nm;
        return rl_sweep_stmts_(sw, fn->children, via ? via : nm);
    }
    default:
        return 0;
    }
}

static int rl_sweep_stmts_(RlSweep *sw, const KflcNode *stmts,
                           const char *via)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        if (sw->kind == RL_SWEEP_PRINT && s->kind == KFLN_STMT_PRINT) {
            sw->found = "print";
            sw->via   = via;
            sw->line  = s->line;
            return 1;
        }
        sw->line = s->line;
        if (rl_sweep_expr_(sw, s->expr, via)) return 1;
        if (rl_sweep_expr_(sw, s->expr2, via)) return 1;
        if (rl_sweep_stmts_(sw, s->children, via)) return 1;
        if (rl_sweep_stmts_(sw, s->else_children, via)) return 1;
    }
    return 0;
}

static void rl_sweep_init_(RlSweep *sw, const KflcNode *form, int kind)
{
    memset(sw, 0, sizeof *sw);
    sw->form = form;
    sw->kind = kind;
}

/* The stepping hot path performs no I/O; print anywhere the on_step
 * body reaches, nested blocks and called fns included, is rejected. */
static int rl_reject_print_(const KflcNode *form, const KflcNode *stmts,
                            KflcDiag *diag)
{
    RlSweep sw;
    rl_sweep_init_(&sw, form, RL_SWEEP_PRINT);
    if (!rl_sweep_stmts_(&sw, stmts, NULL)) return 0;
    if (sw.via) {
        kflc_diag_errorf(diag, sw.line,
            "print is not allowed in on_step: the stepping hot path "
            "performs no I/O, and `fn %s` called from here prints",
            sw.via);
    } else {
        kflc_diag_errorf(diag, sw.line,
            "print is not allowed in on_step: the stepping hot "
            "path performs no I/O");
    }
    return 1;
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
    for (int i = 0; i < 6; i++) {
        if (k && strcmp(k, RL_STATE_KEYS_[i]) == 0) return i;
    }
    return -1;
}

/* Record a (body, key) reference, one slot per pair, in first-mention
 * order. Returns the slot or -2 when the model's table is full. */
static int rl_bs_slot_(RlModel *m, int body, int key, int write, int line,
                       KflcDiag *diag)
{
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
             m->bodies[r->body].body->name, RL_STATE_KEYS_[r->key]);
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
            "pos_x, pos_y, pos_z, vel_x, vel_y, or vel_z)", name, k);
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

/* The assigned expression must stay re-evaluable, so a call to a
 * builtin that is not marked pure is refused, calls into user fns
 * followed. */
static int rl_bs_check_pure_(const KflcNode *form, const KflcExpr *e,
                             int line, KflcDiag *diag)
{
    RlSweep sw;
    rl_sweep_init_(&sw, form, RL_SWEEP_IMPURE);
    if (!rl_sweep_expr_(&sw, e, NULL)) return 0;
    if (sw.via) {
        kflc_diag_errorf(diag, line,
            "on_step: a body state assignment must be side-effect free, "
            "and `fn %s` called here reaches `%s`, which is not pure",
            sw.via, sw.found);
    } else {
        kflc_diag_errorf(diag, line,
            "on_step: a body state assignment must be side-effect free, "
            "and `%s` is not a pure builtin", sw.found);
    }
    return 1;
}

/* Rewrite one statement list: dotted reads become accessor calls, and
 * an assignment to a dotted name becomes an expression statement
 * calling the state setter. Nested blocks are rewritten too, so the
 * form works wherever an ordinary assignment does. */
static int rl_bs_rewrite_stmts_(RlModel *m, KflcNode *stmts,
                                const KflcNode *form, KflcArena *arena,
                                KflcDiag *diag)
{
    for (KflcNode *s = stmts; s; s = s->next) {
        if (s->kind == KFLN_STMT_ASSIGN && s->name &&
            strchr(s->name, '.')) {
            int slot = rl_bs_resolve_(m, s->name, 1, s->line, diag);
            if (slot < 0) return 1;
            if (rl_bs_rewrite_expr_(m, s->expr, s->line, arena, diag)) {
                return 1;
            }
            if (rl_bs_check_pure_(form, s->expr, s->line, diag)) return 1;

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
        if (rl_bs_rewrite_stmts_(m, s->children, form, arena, diag)) {
            return 1;
        }
        if (rl_bs_rewrite_stmts_(m, s->else_children, form, arena, diag)) {
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
        const char *key = RL_STATE_KEYS_[r->key];
        char axis = key[4];
        char fn[192];

        rl_bs_fn_name_(m, i, 0, fn, sizeof fn);
        fprintf(out,
            "static double %s(K26AstroWorld *world)\n"
            "{\n"
            "    K26AstroBody *b = k26astro_world_body_at(world, "
            "kflrl_body_idx_[%d]);\n"
            "    if (!b) return 0.0;\n", fn, r->body);
        if (strncmp(key, "pos_", 4) == 0) {
            fprintf(out,
                "    return (double)b->pos.s%c * K26ASTRO_SECTOR_EDGE_M"
                " + b->pos.l%c;\n", axis, axis);
        } else {
            fprintf(out, "    return b->vel.%c;\n", axis);
        }
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
        /* The stepping hot path performs no I/O, so a print here, or
         * in anything this body calls, would falsify the artifact's
         * contract. */
        if (rl_reject_print_(form, m->on_step->children, diag)) return 1;
        if (rl_bs_rewrite_stmts_(m, m->on_step->children, form, arena,
                                 diag)) {
            return 1;
        }
        rl_emit_state_accessors_(out, m);

        /* The accessors resolve as ordinary calls in this body alone:
         * they are emitted in this translation unit and named only
         * here. */
        if (m->n_bs > 0) {
            n_fns = n_user_fns + 2 * m->n_bs;
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
            n_fns = at;
        }
    }

    fputs("static void kflrl_on_step_(K26AstroWorld *world, "
          "const double *_kfl_act_v)\n"
          "{\n"
          "    (void)world; (void)_kfl_act_v;\n", out);
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
        rl_push_binding_(arena, &live, &live_n, &live_cap, "world",
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
"struct K26RlEnv {\n"
"    uint32_t magic;\n"
"    uint32_t n_envs;\n"
"    uint64_t seed;\n"
"    K26RngKey key;\n"
"    uint32_t rekey_ordinal;\n"
"    uint64_t *seen_seeds;\n"
"    uint32_t n_seen, cap_seen;\n"
"    K26AstroWorld **worlds;\n"
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
"    double    control_dt;\n"
"    uint32_t  horizon;\n"
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
"\n"
"/* Reset one environment in place to episode `ep`: restore the\n"
" * body-state and epoch baseline captured at create, clear the\n"
" * integrator transients a step leaves behind, re-seed the world's\n"
" * runtime noise stream, apply the episode's draws, and recompute\n"
" * the initial observation. No allocation on this path. */\n"
"static void kflrl_reset_env_(K26RlEnv *h, uint32_t e, uint32_t ep)\n"
"{\n"
"    K26AstroWorld *w = h->worlds[e];\n"
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
"    (void)k26astro_world_set_seed(w, h->seed);\n"
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
"    kflrl_observe_(w, h->obs + (size_t)e * KFLRL_OBS_TOTAL);\n"
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
"        uint8_t nv[4 + 64];\n"
"        uint32_t nl = (uint32_t)strlen(kflrl_obs_names_[i]);\n"
"        if (nl > 64) nl = 64;\n"
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
"    free(h->scratch);\n"
"    free(h->spec);\n"
"    free(h->seen_seeds);\n"
"    free(h);\n"
"}\n"
"\n"
"", out);
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
"    if (!(h->control_dt > 0.0) || !std::isfinite(h->control_dt)) {\n"
"        free(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"    kflrl_act_params_(h->act_lo, h->act_hi, h->act_arity, h->act_kind);\n"
"\n"
"    h->cap_seen = 4;\n"
"    h->seen_seeds = (uint64_t *)malloc(h->cap_seen * sizeof(uint64_t));\n"
"    h->worlds = (K26AstroWorld **)calloc(n_envs, sizeof(*h->worlds));\n"
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
"    h->scratch = (double *)calloc(\n"
"        KFLRL_OBS_TOTAL ? KFLRL_OBS_TOTAL : 1, sizeof(double));\n"
"    if (!h->seen_seeds || !h->worlds || !h->baseline || !h->baseline_t ||\n"
"        !h->episode || !h->steps || !h->ended || !h->obs || !h->rew ||\n"
"        !h->flags || !h->fault || !h->dr_vals || !h->wscal ||\n"
"        !h->scratch) {\n"
"        kflrl_free_handle_(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"    h->seen_seeds[0] = seed;\n"
"    h->n_seen = 1;\n"
"\n"
"    for (uint32_t e = 0; e < n_envs; e++) {\n"
"        h->worlds[e] = k26astro_world_create(K26ASTRO_MODE_PORTABLE,\n"
"                                             K26ASTRO_COORDS_SECTOR_GRID);\n"
"        if (!h->worlds[e]) {\n"
"            kflrl_free_handle_(h);\n"
"            return K26RL_E_INTERNAL;\n"
"        }\n"
"        (void)k26astro_world_set_seed(h->worlds[e], seed);\n"
"#if KFLRL_N_DR > 0\n"
"        double dr0[KFLRL_N_DR];\n"
"#else\n"
"        double *dr0 = NULL;\n"
"#endif\n"
"        if (kflrl_build_world_(h->worlds[e], h->key, e,\n"
"                h->wscal + (size_t)e * KFLRL_N_WSCAL, dr0) != 0) {\n"
"            kflrl_free_handle_(h);\n"
"            return K26RL_E_INTERNAL;\n"
"        }\n"
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
"                       h->obs + (size_t)e * KFLRL_OBS_TOTAL);\n"
"    }\n"
"\n"
"    h->spec_len = kflrl_spec_write_(NULL, h);\n"
"    h->spec = (uint8_t *)malloc(h->spec_len);\n"
"    if (!h->spec) {\n"
"        kflrl_free_handle_(h);\n"
"        return K26RL_E_INTERNAL;\n"
"    }\n"
"    (void)kflrl_spec_write_(h->spec, h);\n"
"\n"
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
"        kflrl_on_step_(h->worlds[e], aslice);\n"
"        int rc = k26astro_world_step_exact(h->worlds[e], h->control_dt);\n"
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
"        kflrl_observe_(h->worlds[e], h->scratch);\n"
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

    rl_emit_prologue_(out, &m, form);
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
    rl_emit_observe_(out, &m);
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
