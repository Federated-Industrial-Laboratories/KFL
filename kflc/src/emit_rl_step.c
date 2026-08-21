/* emit_rl_step.c - the step path: purity sweeps, body-state resolution,
 * engagements, on_step and the objective. */
#include "emit_rl_internal.h"

static const KflcNode *rl_find_user_fn_(const KflcNode *form,
                                        const char *name);
static int rl_type_allocates_(KflcType t);
static const char *rl_step_stmt_why_(const KflcNode *s, const char **what);
static int rl_sweep_expr_(RlSweep *sw, const KflcExpr *e);
static int rl_sweep_stmts_(RlSweep *sw, const KflcNode *stmts);
static void rl_sweep_init_(RlSweep *sw, const KflcNode *form);
static void rl_sweep_chain_(const RlSweep *sw, char *buf, size_t n);
static void rl_sweep_report_(const RlSweep *sw, const char *what, int line,
                             KflcDiag *diag);
static int rl_reject_impure_(const KflcNode *form, const KflcExpr *e,
                             const char *what, int line, KflcDiag *diag);
static int rl_reject_stmt_(const KflcNode *s, const char *block,
                           KflcDiag *diag);
static const char *rl_step_block_(const KflcNode *s, const char *outer);
static void rl_step_position_(char *buf, size_t n, const KflcNode *s,
                              int slot, const char *inside);
static int rl_reject_impure_block_(const KflcNode *form,
                                   const KflcNode *stmts,
                                   const char *inside, KflcDiag *diag);
static void rl_emit_state_accessors_(FILE *out, const RlModel *m);
static int rl_agent_has_action_(const RlModel *m, int ag, const char *name);
static int rl_action_declarers_(const RlModel *m, const char *name);
static int rl_agent_scan_expr_(const RlModel *m, KflcExpr *e, int line,
                               KflcArena *arena, KflcDiag *diag);
static int rl_agent_scan_stmts_(const RlModel *m, KflcNode *stmts,
                                KflcArena *arena, KflcDiag *diag);
static void rl_emit_engage_softkill_(FILE *out, const RlModel *m, int e);
static void rl_emit_engage_one_(FILE *out, const RlModel *m, int e);
static void rl_check_objective_names_(const RlModel *m, int ag,
                                      const KflcNode *form,
                                      const KflcExpr *e,
                                      const char *ctx_word, int line,
                                      KflcDiag *diag);
static int rl_emit_objective_one_(FILE *out, const RlModel *m, int ag,
                                  const KflcNode *form, KflcArena *arena,
                                  const char *name, const char *ret,
                                  const KflcAttr *attr, const char *absent,
                                  const char *word,
                                  KflcExprFn *user_fn_arr, int n_user_fns,
                                  KflcDiag *diag);

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

    /* An engagement is an act of a step by construction: it lowers to
     * one call over compile-time indices, it reads no heap storage and
     * takes none, and its effect on the world is the point of the
     * statement rather than a side effect of one. */
    case KFLN_STMT_ENGAGE:
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
    case KFLN_STMT_AGENT:
    case KFLN_STMT_PLAN:
    case KFLN_STMT_CAPTURE_ENVELOPE:
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
    /* An identifier reaches nothing this sweep judges, and neither
     * does a literal. They are named rather than left to a default so
     * that a ninth expression kind is a compile error here. */
    case KFLE_IDENT:
    case KFLE_INT_LIT:
    case KFLE_FLOAT_LIT:
        return 0;
    }
    return 0;
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

        rl_bs_fn_name(m, i, 0, fn, sizeof fn);
        fprintf(out,
            "static double %s(K26AstroWorld *world)\n"
            "{\n"
            "    K26AstroBody *b = k26astro_world_body_at(world, "
            "kflrl_body_idx_[%d]);\n"
            "    if (!b) return 0.0;\n", fn, r->body);
        kflc_emit_body_state_read(out, "b->", key);
        fputs("}\n\n", out);

        if (!r->written) continue;
        rl_bs_fn_name(m, i, 1, fn, sizeof fn);
        fprintf(out,
            "static void %s(K26AstroWorld *world, double v)\n"
            "{\n"
            "    K26AstroBody *b = k26astro_world_body_at(world, "
            "kflrl_body_idx_[%d]);\n"
            "    if (!b) return;\n", fn, r->body);
        rl_emit_body_write(out, 4, "b->", key, "v");
        fputs("}\n\n", out);
    }
}

/* ---- Agent names inside on_step -------------------------------------- */

static int rl_agent_has_action_(const RlModel *m, int ag, const char *name)
{
    if (!name) return 0;
    for (int i = 0; i < m->n_actions; i++) {
        if (rl_act_agent(m, i) != ag) continue;
        const char *an = m->actions[i]->name;
        if (an && strcmp(an, name) == 0) return 1;
    }
    return 0;
}

/* How many blocks declare an action of this name. */

static int rl_action_declarers_(const RlModel *m, const char *name)
{
    int n = 0;
    for (int a = 0; a < m->agent_count; a++) {
        if (rl_agent_has_action_(m, a, name)) n++;
    }
    return n;
}

/* Resolve the agent names an `on_step` body reads. Every agent's
 * action channels are in scope there: an unqualified name resolves
 * when it is unique across every block, and `<agent>.<action>` always
 * resolves. The refusal fires exactly where the resolution would
 * otherwise be a guess, and it names both declaration sites and the
 * qualified form that separates them.
 *
 * Runs before the body-state rewrite, so a qualified action name never
 * reaches the body-state resolver, which would report it as a body
 * that does not exist. */

static int rl_agent_scan_expr_(const RlModel *m, KflcExpr *e, int line,
                               KflcArena *arena, KflcDiag *diag)
{
    if (!e) return 0;
    switch (e->kind) {
    case KFLE_IDENT: {
        const char *id = e->u.ident;
        if (!id) return 0;
        char an[KFLC_OBS_NAME_MAX], cn[KFLC_OBS_NAME_MAX];
        if (rl_dotted_split(id, an, sizeof an, cn, sizeof cn)) {
            int ag = rl_agent_index_of(m, an);
            if (ag < 0) return 0;   /* somebody else's dotted shape */
            if (rl_agent_has_action_(m, ag, cn)) {
                char q[KFLC_OBS_NAME_MAX + 32];
                rl_qual_ident(ag, cn, q, sizeof q);
                e->u.ident = kflc_arena_strdup(arena, q);
                return 0;
            }
            if (rl_agent_has_channel(m, ag, cn)) {
                kflc_diag_errorf(diag, line,
                    "on_step: `%s`: `%s` is an observation channel of "
                    "agent `%s`, and observation channels are readable "
                    "in the objective and termination expressions, not "
                    "in on_step", id, cn, an);
                return 1;
            }
            kflc_diag_errorf(diag, line,
                "on_step: `%s`: agent `%s` declares no action called "
                "`%s`", id, an, cn);
            return 1;
        }
        if (m->agent_count > 1 && rl_action_declarers_(m, id) > 1) {
            int a0 = -1, l0 = 0, a1 = -1, l1 = 0;
            rl_action_sites(m, id, &a0, &l0, &a1, &l1);
            kflc_diag_errorf(diag, line,
                "on_step: `%s` is ambiguous: agent `%s` declares it at "
                "line %d and agent `%s` declares it at line %d; write "
                "`%s.%s` or `%s.%s`",
                id, m->agents[a0].name, l0, m->agents[a1].name, l1,
                m->agents[a0].name, id, m->agents[a1].name, id);
            return 1;
        }
        return 0;
    }
    case KFLE_UNARY:
        return rl_agent_scan_expr_(m, e->u.un.operand, line, arena, diag);
    case KFLE_BINARY:
        return rl_agent_scan_expr_(m, e->u.bin.lhs, line, arena, diag) ||
               rl_agent_scan_expr_(m, e->u.bin.rhs, line, arena, diag);
    case KFLE_INDEX:
        return rl_agent_scan_expr_(m, e->u.index.base, line, arena, diag) ||
               rl_agent_scan_expr_(m, e->u.index.idx, line, arena, diag);
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            if (rl_agent_scan_expr_(m, e->u.vec.elems[i], line, arena,
                                    diag)) return 1;
        }
        return 0;
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            if (rl_agent_scan_expr_(m, e->u.call.args[i], line, arena,
                                    diag)) return 1;
        }
        return 0;
    case KFLE_INT_LIT:
    case KFLE_FLOAT_LIT:
        return 0;
    }
    return 0;
}

static int rl_agent_scan_stmts_(const RlModel *m, KflcNode *stmts,
                                KflcArena *arena, KflcDiag *diag)
{
    for (KflcNode *s = stmts; s; s = s->next) {
        if (rl_agent_scan_expr_(m, s->expr, s->line, arena, diag))  return 1;
        if (rl_agent_scan_expr_(m, s->expr2, s->line, arena, diag)) return 1;
        for (KflcAttr *a = s->attrs; a; a = a->next) {
            if (rl_agent_scan_expr_(m, a->expr, a->line ? a->line : s->line,
                                    arena, diag)) return 1;
        }
        if (rl_agent_scan_stmts_(m, s->children, arena, diag))      return 1;
        if (rl_agent_scan_stmts_(m, s->else_children, arena, diag)) return 1;
    }
    return 0;
}

/* One engagement, as a function of the two bodies' current state.
 *
 * The whole of an engagement's geometry is derived here rather than
 * declared: the range is the distance between the payload's own body
 * and the target, the dwell is the control period, and the area the
 * target presents is the projected area of its collision primitives
 * along the line the effector acts on. That last is the same helper
 * and the same silhouette the detection channels take, because a laser
 * spot and an infrared signature are the same projection of the same
 * craft; nothing here is a second description of the geometry.
 *
 * The effect on the world is the substance of the statement. An
 * effector that published a result and changed nothing would be a
 * decoration, so each kind's principal result is applied to the target
 * before this returns, and the step's own advance is what integrates
 * it: `on_step` runs before the world moves, which is the same
 * relation a body state write has.
 *
 * Both kinds are closed form over the declared parameters and the
 * current state, with no draw and no iteration, and both are handed no
 * generator: an imperfection on an effector channel arrives through
 * the declared sensor layer, at the coordinates a replay reproduces.
 */

/* The body of a countermeasure engagement, which is the one effector
 * class whose result lands on another payload rather than on a body.
 *
 * Both kinds write into the per-step degradation store, indexed by the
 * victim payload and by the craft the countermeasure protects, and the
 * victim's own detection observe reads it later in the same step. The
 * engagement runs before the world advances and the observation is
 * computed after it, so a degradation written here is in force for that
 * step's observation and for no other: the store is cleared at the top
 * of every step and at every reset.
 *
 * Which victim payloads an engagement can reach is fixed at compile
 * time, since a payload's body and an engagement's target are both
 * declarations, so the loop below is unrolled here and nothing is
 * searched for while stepping. What is not fixed is how many it
 * actually reached, which is published: a countermeasure aimed at a
 * craft that carries nothing to degrade reaches none, and that is the
 * decoration this class has to be able to report rather than hide. */

static void rl_emit_engage_softkill_(FILE *out, const RlModel *m, int e)
{
    const RlEngage  *en = &m->engages[e];
    const RlPayload *py = &m->payloads[en->payload];
    int host = py->body;

    if (py->kind == RL_PAY_JAMMER) {
        fputs(
        "    (void)dt;\n"
        "    (void)_kfl_mass;\n"
        /* Every watt transmitted announces the emitter's position, and
         * the library publishes that power as its own quantity. It is
         * the input to the counter-detection range below and it is
         * published beside it, so a program can see what jamming costs
         * on the same statement that shows what it buys. */
        "    double _kfl_self = k26astro_jammer_self_signature_W(_kfl_h);\n"
        "    K26V3 _kfl_u = k26m3d_v3(_kfl_d.x / _kfl_rng,\n"
        "                             _kfl_d.y / _kfl_rng,\n"
        "                             _kfl_d.z / _kfl_rng);\n"
        /* The cross-section in the jamming-to-signal ratio is the
         * protected craft's, which for self-protection jamming is the
         * jammer's own host: this library exposes that geometry and
         * says so. It is the silhouette the host presents along the
         * line the victim looks down, taken in the host's own frame, so
         * a craft that turns changes both what it returns to the radar
         * and how well its jammer masks that return. */
        "    K26V3 _kfl_look = k26m3d_quat_rotate_v3(\n"
        "        k26m3d_quat_conj(_kfl_eb->attitude),\n"
        "        k26m3d_v3(-_kfl_u.x, -_kfl_u.y, -_kfl_u.z));\n", out);
        fprintf(out,
        "    double _kfl_area = kflrl_sig_area_(%d, _kfl_look);\n", host);
        fputs(
        "    K26V3 _kfl_nrm = k26m3d_v3(-_kfl_look.x, -_kfl_look.y,\n"
        "                               -_kfl_look.z);\n"
        "    int    _kfl_reach = 0;\n"
        "    double _kfl_best = 0.0, _kfl_rcs = 0.0, _kfl_bt = 0.0;\n"
        "    double _kfl_ctr = 0.0;\n", out);

        for (int q = 0; q < m->n_payloads; q++) {
            if (m->payloads[q].body != en->target) continue;
            if (m->payloads[q].kind == RL_PAY_DETECT_RADAR) {
                fprintf(out,
        "    {\n"
        "        const double *vp = payp + %d * KFLRL_PAY_NPARAM;\n"
        "        double _kfl_lam = vp[3] > 0.0 ? (K26A_C / vp[3]) : 0.0;\n"
        "        double _kfl_s = k26astro_signature_rcs_monostatic(\n"
        "            1, &_kfl_nrm, &_kfl_area, _kfl_look, _kfl_lam);\n", q);
                /* The cloud this victim declared around the craft the
                 * jammer protects belongs in the jamming ratio as well
                 * as in the radar equation. A cross-section the radar
                 * sees and the jamming ratio does not would have the
                 * two equations describing different craft, and the
                 * ratio would overstate the jamming by whatever the
                 * cloud returns. */
                if (m->payloads[q].attr[RL_PAY_RADAR_CHAFF_N] &&
                    rl_detect_observes(m, q, host)) {
                    fprintf(out,
        "        _kfl_s += k26astro_chaff_mean_rcs(\n"
        "            kflrl_chaff_strips_(vp[%d]), vp[%d]);\n",
                        RL_PAY_RADAR_CHAFF_N, RL_PAY_RADAR_CHAFF_SIG);
                }
                fprintf(out,
        "        double _kfl_js = k26astro_jammer_js_ratio(_kfl_h,\n"
        "            vp[0], vp[1], _kfl_s, _kfl_rng);\n"
        /* Two jammers on one victim add their power at the receiver,
         * so the ratios add. */
        "        if (_kfl_js > 0.0) {\n"
        "            eng->deg[%d * KFLRL_DEG_STRIDE + %d].js += _kfl_js;\n"
        "            _kfl_reach++;\n"
        "            if (_kfl_js > _kfl_best) {\n"
        "                _kfl_best = _kfl_js;\n"
        "                _kfl_rcs  = _kfl_s;\n"
        /* The range at which this victim's radar burns through, which
         * is the range at which its published statistic crosses its own
         * threshold under the degradation this binding applies. The
         * library's own helper answers a different question: it solves
         * the ratio against the *jammer's* declared threshold and never
         * sees the victim's, so its figure and this artifact's
         * detection flag disagree by a factor of the victim's threshold.
         * A channel a policy steers on has to mean what the flag beside
         * it means.
         *
         * The statistic falls as one over range to the fourth and the
         * ratio grows as range squared, so writing the degradation law
         * at the crossover and substituting both gives a quadratic in
         * range squared whose positive root is the boundary. It is
         * taken in the form that divides rather than subtracts, the
         * subtracting form losing every significant digit whenever the
         * jamming dominates, which is the regime the channel exists
         * for. */
        "                double _kfl_snr0 =\n"
        "                    k26astro_detect_radar_active(\n"
        "                        vp[0], vp[1], vp[2], vp[3], _kfl_s,\n"
        "                        _kfl_rng, vp[4], vp[5], vp[6], vp[7],\n"
        "                        vp[8], NULL).snr;\n"
        "                double _kfl_r2 = _kfl_rng * _kfl_rng;\n"
        "                double _kfl_kk = _kfl_snr0 * _kfl_r2 * _kfl_r2;\n"
        "                double _kfl_jj = _kfl_js / _kfl_r2;\n"
        "                _kfl_bt = 0.0;\n"
        "                if (_kfl_kk > 0.0 && vp[8] > 0.0) {\n"
        "                    double _kfl_bb = _kfl_jj * _kfl_kk;\n"
        "                    double _kfl_cc = _kfl_kk / vp[8];\n"
        "                    double _kfl_den = _kfl_bb\n"
        "                        + sqrt(_kfl_bb * _kfl_bb + 4.0 * _kfl_cc);\n"
        "                    if (_kfl_den > 0.0) {\n"
        "                        _kfl_bt = sqrt(2.0 * _kfl_cc / _kfl_den);\n"
        "                    }\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n", q, host);
            } else if (m->payloads[q].kind == RL_PAY_DETECT_IR) {
                /* The other edge. A passive infrared observer sees the
                 * thermalised transmit power, and the detection
                 * library's counter-detection routine returns the range
                 * at which that observer's own aperture, dwell,
                 * pass-band, throughput and threshold put it at its
                 * detection threshold. The largest such range over the
                 * victim's infrared payloads is what the engagement
                 * publishes, and each payload's own range is what that
                 * payload's observation is raised by. */
                fprintf(out,
        "    {\n"
        "        const double *vp = payp + %d * KFLRL_PAY_NPARAM;\n"
        "        double _kfl_rc = k26astro_counter_detect_ir_range(\n"
        "            _kfl_self, pp[%d], vp[0], vp[1], vp[2], vp[3],\n"
        "            vp[4], vp[5]);\n"
        "        if (_kfl_rc > 0.0) {\n"
        "            double *_kfl_sl =\n"
        "                &eng->deg[%d * KFLRL_DEG_STRIDE + %d].ctr;\n"
        "            if (_kfl_rc > *_kfl_sl) *_kfl_sl = _kfl_rc;\n"
        "            _kfl_reach++;\n"
        "            if (_kfl_rc > _kfl_ctr) _kfl_ctr = _kfl_rc;\n"
        "        }\n"
        "    }\n", q, RL_PAY_JAMMER_RADIATOR, q, host);
            }
        }

        fputs(
        "    ch[1] = _kfl_best;\n"
        "    ch[2] = (double)_kfl_reach;\n"
        "    ch[3] = _kfl_rng;\n"
        "    ch[4] = _kfl_rcs;\n"
        "    ch[5] = _kfl_bt;\n"
        "    ch[6] = _kfl_self;\n"
        "    ch[7] = _kfl_ctr;\n"
        "    ch[8] = (_kfl_ctr > 0.0 && _kfl_rng <= _kfl_ctr)\n"
        "          ? 1.0 : 0.0;\n"
        "}\n\n", out);
        return;
    }

    /* The decoy, whose engagement is two effects rather than one. The
     * deploy is a change to the craft that carries it: a mass leaves at
     * a separation velocity, so the host takes the opposite momentum
     * and loses that mass. The decoy is placed between the host and the
     * observer it is meant to fool, which fixes the direction the
     * separation is taken along and therefore the direction the host
     * recoils in.
     *
     * **The engagement is bounded by conservation rather than by a
     * declared count.** The momentum the host takes is derived from the
     * mass that leaves it, so the two cannot be decided separately: a
     * deploy the host cannot supply must impart no momentum either, or
     * the statement pays a benefit out of mass that never left. Since
     * nothing here counts rounds, a program that engages on every step
     * walks its host down to the declared dry mass, and from there on
     * the host's own mass is what refuses the deploy. That is a real
     * magazine derived from what the tier already models, and it is why
     * this is the one effector kind whose engagements run out.
     *
     * Nothing is deployed unless the host is strictly heavier than the
     * decoy: no momentum, no mass loss, and no degradation on any
     * victim, since a decoy that never left the craft cannot be
     * confused with it. The strict comparison is the same rule the
     * ablated mass follows, a massless body in the integrator not being
     * a state this layer will produce. A separate channel says so
     * plainly rather than leaving a reader to read it out of a zero. */
    fputs(
        "    (void)dt;\n"
        "    (void)_kfl_mass;\n"
        "    K26V3 _kfl_u = k26m3d_v3(_kfl_d.x / _kfl_rng,\n"
        "                             _kfl_d.y / _kfl_rng,\n"
        "                             _kfl_d.z / _kfl_rng);\n"
        "    double _kfl_hm = _kfl_eb->mass;\n"
        "    double _kfl_dm = pp[1];\n"
        "    int    _kfl_dep = (_kfl_dm > 0.0 && _kfl_hm > _kfl_dm)\n"
        "                    ? 1 : 0;\n"
        "    double _kfl_dv = 0.0, _kfl_loss = 0.0;\n"
        "    if (_kfl_dep) {\n"
        /* The separation velocity is between the decoy and the host
         * after the deploy, which is what makes this exact: writing
         * momentum conservation with that convention gives the host an
         * increment of the released mass times the separation velocity
         * over the mass the host had before it, and no approximation
         * enters. The alternative convention, the decoy leaving at the
         * declared speed in the pre-deploy frame, divides by the mass
         * left behind instead; both are defensible and this one is
         * chosen because it is the one that closes exactly. */
        "        _kfl_dv   = _kfl_dm * pp[2] / _kfl_hm;\n"
        "        _kfl_loss = _kfl_dm;\n"
        "        _kfl_eb->vel.x -= _kfl_dv * _kfl_u.x;\n"
        "        _kfl_eb->vel.y -= _kfl_dv * _kfl_u.y;\n"
        "        _kfl_eb->vel.z -= _kfl_dv * _kfl_u.z;\n"
        "        k26astro_body_set_mass(_kfl_eb, _kfl_hm - _kfl_loss);\n"
        "    }\n"
        "    int    _kfl_reach = 0;\n"
        "    double _kfl_best = 0.0, _kfl_pd = 0.0;\n"
        "    if (_kfl_dep) {\n", out);

    for (int q = 0; q < m->n_payloads; q++) {
        if (m->payloads[q].body != en->target) continue;
        int slot;
        switch (m->payloads[q].kind) {
        case RL_PAY_DETECT_IR:    slot = RL_PAY_IR_REGIME;    break;
        case RL_PAY_DETECT_RADAR: slot = RL_PAY_RADAR_REGIME; break;
        case RL_PAY_DETECT_LIDAR: slot = RL_PAY_LIDAR_REGIME; break;
        default: continue;
        }
        /* The regime is the victim payload's own declaration, so two
         * observers running different discriminators against one decoy
         * reach different answers about it, which is what the library's
         * dispatch is for. */
        fprintf(out,
        "    {\n"
        "        const double *vp = payp + %d * KFLRL_PAY_NPARAM;\n"
        "        double _kfl_pdq =\n"
        "            k26astro_decoy_probability_discriminated(_kfl_h,\n"
        "                (K26AstroDiscriminatorRegime)(int)vp[%d]);\n"
        "        double _kfl_dg = 1.0 - _kfl_pdq;\n"
        "        if (_kfl_dg < 0.0) _kfl_dg = 0.0;\n"
        "        if (_kfl_dg > 0.0) {\n"
        "            double *_kfl_sl =\n"
        "                &eng->deg[%d * KFLRL_DEG_STRIDE + %d].dec;\n"
        /* Two decoys against one observer compose the way the
         * library's own discrimination model composes its channels.
         * The observer must discriminate every decoy to be undeceived,
         * so the probabilities of discriminating each multiply, and
         * the confidence the observer keeps is that product: two
         * decoys leave it less than one does. */
        "            *_kfl_sl = *_kfl_sl + (1.0 - *_kfl_sl) * _kfl_dg;\n"
        "            _kfl_reach++;\n"
        "            if (_kfl_dg > _kfl_best) {\n"
        "                _kfl_best = _kfl_dg;\n"
        "                _kfl_pd   = _kfl_pdq;\n"
        "            }\n"
        "        }\n"
        "    }\n", q, slot, q, host);
    }

    fputs(
        "    }\n"
        "    ch[1] = _kfl_dep ? 1.0 : 0.0;\n"
        "    ch[2] = _kfl_best;\n"
        "    ch[3] = (double)_kfl_reach;\n"
        "    ch[4] = _kfl_pd;\n"
        "    ch[5] = _kfl_rng;\n"
        "    ch[6] = _kfl_dv;\n"
        "    ch[7] = _kfl_loss;\n"
        "}\n\n", out);
}

static void rl_emit_engage_one_(FILE *out, const RlModel *m, int e)
{
    const RlEngage  *en = &m->engages[e];
    const RlPayload *py = &m->payloads[en->payload];
    int p   = en->payload;
    int eb  = py->body;
    int tb  = en->target;
    int imp = py->kind == RL_PAY_IMPACTOR;

    fprintf(out,
        "/* `engage %s at %s`, line %d. */\n"
        "static void kflrl_engage_%d_(K26AstroWorld *world,\n"
        "                             void *const *pay,\n"
        "                             void *const *payu,\n"
        "                             const double *payp, double dt,\n"
        "                             KflrlEng *eng)\n"
        "{\n"
        "    if (!eng) return;\n"
        /* The runtime half of the one-engagement-per-step rule. The
         * compiler refuses two statements naming one payload; this
         * catches a single statement reached twice, which is the shape
         * a loop or a twice-called function produces and which no
         * static test can see. */
        "    if (eng->used[%d]) { eng->fault = 1; return; }\n"
        "    eng->used[%d] = 1;\n"
        "    double *ch = eng->ch + %d * KFLRL_EFF_STRIDE;\n"
        "    ch[0] = 1.0;\n"
        "    K26AstroBody *_kfl_eb = k26astro_world_body_at(\n"
        "        world, kflrl_body_idx_[%d]);\n"
        "    K26AstroBody *_kfl_tb = k26astro_world_body_at(\n"
        "        world, kflrl_body_idx_[%d]);\n"
        "    %s *_kfl_h = pay ? (%s *)pay[%d] : NULL;\n"
        "    (void)payu;\n"
        "    const double *pp = payp ? payp + %d * KFLRL_PAY_NPARAM\n"
        "                            : NULL;\n"
        "    if (!_kfl_eb || !_kfl_tb || !_kfl_h || !pp) return;\n"
        "    K26V3 _kfl_d = k26astro_pos_sub(&_kfl_tb->pos, &_kfl_eb->pos);\n"
        "    double _kfl_rng = k26m3d_v3_len(_kfl_d);\n"
        "    if (!(_kfl_rng > 0.0)) return;\n"
        "    double _kfl_mass = _kfl_tb->mass;\n",
        en->node->name ? en->node->name : "?",
        m->bodies[tb].body->name, en->line, e, p, p, p,
        eb, tb, rl_pay_ctype(py->kind), rl_pay_ctype(py->kind), p, p);

    if (py->kind == RL_PAY_DECOY || py->kind == RL_PAY_JAMMER) {
        rl_emit_engage_softkill_(out, m, e);
        return;
    }

    if (!imp) {
        fputs(
        "    K26V3 _kfl_u = k26m3d_v3(_kfl_d.x / _kfl_rng,\n"
        "                             _kfl_d.y / _kfl_rng,\n"
        "                             _kfl_d.z / _kfl_rng);\n"
        /* The silhouette is taken in the target's own frame along the
         * line the beam travels, which is what makes the spot's
         * encircled fraction and the fluence aspect dependent. */
        "    K26V3 _kfl_look = k26m3d_quat_rotate_v3(\n"
        "        k26m3d_quat_conj(_kfl_tb->attitude), _kfl_u);\n"
        "    double _kfl_area = kflrl_sig_area_(", out);
        fprintf(out, "%d, _kfl_look);\n", tb);
        fputs(
        "    if (!(_kfl_area > 0.0)) return;\n"
        "    K26AstroLaserAblationEvent ev = k26astro_laser_engage(\n"
        "        _kfl_h, (K26AstroLaserMaterial)(int)pp[7], _kfl_area,\n"
        "        pp[8], _kfl_rng, dt);\n"
        /* The ablation plume leaves along the beam, so the recoil
         * pushes the target away from the emitter: the impulse acts
         * along the unit vector from emitter to target. The velocity
         * increment is taken against the mass the target had when the
         * light arrived, and the ablated mass is removed after it. */
        "    double _kfl_dv = (_kfl_mass > 0.0)\n"
        "                   ? ev.impulse_N_s / _kfl_mass : 0.0;\n"
        "    _kfl_tb->vel.x += _kfl_dv * _kfl_u.x;\n"
        "    _kfl_tb->vel.y += _kfl_dv * _kfl_u.y;\n"
        "    _kfl_tb->vel.z += _kfl_dv * _kfl_u.z;\n"
        /* The published mass loss is the mass actually removed. A step
         * that would ablate the whole body is outside this model's
         * range, and removing all of it would leave a massless body in
         * the integrator, so the removal is bounded and the channel
         * reports what was applied rather than what was predicted. */
        "    double _kfl_loss = ev.mass_loss_kg;\n"
        "    if (_kfl_loss < 0.0) _kfl_loss = 0.0;\n"
        "    if (_kfl_loss >= _kfl_mass) _kfl_loss = 0.0;\n"
        "    if (_kfl_loss > 0.0) {\n"
        "        k26astro_body_set_mass(_kfl_tb, _kfl_mass - _kfl_loss);\n"
        "    }\n"
        "    ch[1] = ev.impulse_N_s;\n"
        "    ch[2] = _kfl_dv;\n"
        "    ch[3] = _kfl_loss;\n"
        "    ch[4] = ev.range_m;\n"
        "    ch[5] = ev.spot_diameter_m;\n"
        "    ch[6] = ev.encircled_fraction;\n"
        "    ch[7] = ev.fluence_J_per_m2;\n"
        "    ch[8] = ev.plasma_transmissivity;\n"
        "    ch[9] = ev.p_coupled_W;\n"
        "    ch[10] = ev.plasma_ignited ? 1.0 : 0.0;\n"
        "}\n\n", out);
        return;
    }

    fprintf(out,
        /* The intercept prediction, from the relative state of the two
         * bodies. Positions travel as an exact difference rather than
         * as flattened coordinates, because a position here is a sector
         * index and a bounded offset and flattening throws away the
         * precision the sector grid exists to keep. */
        "    K26V3 _kfl_rp = k26astro_pos_sub(&_kfl_eb->pos,\n"
        "                                     &_kfl_tb->pos);\n"
        "    K26V3 _kfl_rt = k26m3d_v3(0.0, 0.0, 0.0);\n"
        "    K26V3 _kfl_vp = _kfl_eb->vel;\n"
        "    K26V3 _kfl_vt = _kfl_tb->vel;\n"
        "    double _kfl_vc = k26astro_impactor_closing_speed(_kfl_vp,\n"
        "                                                     _kfl_vt);\n"
        "    double _kfl_tca = k26astro_impactor_time_to_closest_approach(\n"
        "        _kfl_rp, _kfl_vp, _kfl_rt, _kfl_vt);\n"
        "    double _kfl_miss = k26astro_impactor_closest_approach_distance(\n"
        "        _kfl_rp, _kfl_vp, _kfl_rt, _kfl_vt);\n"
        /* The projectile is released with its launcher's own state and
         * flies ballistically, so the direction it arrives from is the
         * closing velocity. The area the target presents to it is the
         * projection along that direction in the target's own frame,
         * which is the same silhouette a detector would see from the
         * same direction. */
        "    K26V3 _kfl_w = k26m3d_v3_sub(_kfl_vp, _kfl_vt);\n"
        "    if (_kfl_vc > 0.0) {\n"
        "        _kfl_w = k26m3d_v3(_kfl_w.x / _kfl_vc,\n"
        "                           _kfl_w.y / _kfl_vc,\n"
        "                           _kfl_w.z / _kfl_vc);\n"
        "    }\n"
        "    K26V3 _kfl_look = k26m3d_quat_rotate_v3(\n"
        "        k26m3d_quat_conj(_kfl_tb->attitude), _kfl_w);\n"
        "    double _kfl_area = kflrl_sig_area_(%d, _kfl_look);\n"
        "    if (!(_kfl_area > 0.0)) return;\n"
        /* The hit test, in two parts. The target's effective silhouette
         * radius is the radius of a disc of the projected area, and the
         * predicted closest approach must fall inside it.
         *
         * And the predicted intercept must fall inside the step about
         * to be integrated. The prediction is a straight line through
         * both bodies' current states, which is defensible only over a
         * horizon short enough that neither trajectory curves
         * appreciably, and one control period is that horizon by
         * construction: it is the interval the artifact is about to
         * integrate. Without the bound an engagement lands whenever a
         * straight line says the paths cross at any time in the
         * future, which for two craft in nearby orbits is almost
         * always, and it lands again on every step until they meet.
         * The bound is what makes this a terminal-phase effector, and
         * it is why the reach scales with closing speed rather than
         * being a fixed distance. */
        "    double _kfl_rad = sqrt(_kfl_area / K26A_PI);\n"
        "    int _kfl_hit = (_kfl_tca > 0.0 && _kfl_tca <= dt &&\n"
        "                    _kfl_miss <= _kfl_rad) ? 1 : 0;\n"
        /* The first body axis is the axis the assembly format runs
         * along the craft, and the axis the projected area's own first
         * term is taken against. */
        "    K26V3 _kfl_n = k26m3d_quat_rotate_v3(_kfl_tb->attitude,\n"
        "                                         k26m3d_v3(1.0, 0.0, 0.0));\n"
        "    double _kfl_cos = k26astro_impactor_impact_cos_angle(_kfl_w,\n"
        "                                                         _kfl_n);\n"
        /* The library clamps a negative cosine to zero by comparison,
         * which leaves a negative zero as it found it. The published
         * range is [0, 1] and negative zero is not in it, so the sign
         * is normalised where the channel is written rather than left
         * for a consumer to discover. */
        "    if (_kfl_cos == 0.0) _kfl_cos = 0.0;\n"
        /* A swarm and a single projectile differ in what reaches the
         * target. The cone's footprint at the engagement range is the
         * area the released units are spread over, and the fraction of
         * them that land is the target's silhouette over that
         * footprint, capped at one. The fraction is a function of the
         * declared cone and the geometry alone: no direction is
         * sampled, because sampling would put a second generator on
         * this path. */
        "    double _kfl_frac = 1.0;\n"
        "    if ((int)pp[0] == (int)K26ASTRO_IMPACTOR_PATTERN_SWARM) {\n"
        "        double _kfl_fp = k26astro_swarm_footprint_m2(_kfl_rng,\n"
        "                                                     pp[5]);\n"
        "        _kfl_frac = (_kfl_fp > 0.0) ? (_kfl_area / _kfl_fp) : 1.0;\n"
        "        if (_kfl_frac > 1.0) _kfl_frac = 1.0;\n"
        "    }\n"
        "    ch[3] = _kfl_vc;\n"
        "    ch[4] = _kfl_tca;\n"
        "    ch[5] = _kfl_miss;\n"
        "    ch[6] = _kfl_frac;\n"
        "    ch[7] = _kfl_cos;\n"
        "    ch[2] = _kfl_hit ? 1.0 : 0.0;\n"
        "    if (!_kfl_hit) return;\n"
        /* Momentum transfer, and nothing beyond it. The projectile
         * arrives carrying its mass times the closing speed in the
         * target's frame, and the target takes that momentum along the
         * closing direction. The transferred momentum is scaled by the
         * fraction that landed. No ejecta enhancement is claimed: the
         * momentum enhancement factor here is exactly one, where the
         * published deflection literature reports it above one for a
         * cratering impact into a rubble body, and this model does not
         * carry the ejecta mass and speed that figure comes from. */
        "    double _kfl_dv = (_kfl_mass > 0.0)\n"
        "        ? (_kfl_frac * pp[1] * _kfl_vc / _kfl_mass) : 0.0;\n"
        "    _kfl_tb->vel.x += _kfl_dv * _kfl_w.x;\n"
        "    _kfl_tb->vel.y += _kfl_dv * _kfl_w.y;\n"
        "    _kfl_tb->vel.z += _kfl_dv * _kfl_w.z;\n"
        "    ch[1] = _kfl_dv;\n"
        /* The penetration analysis, which is a question about one
         * arriving unit rather than about the release as a whole:
         * whether a projectile gets through this shield. For a swarm
         * that unit is not the declared projectile but one of the
         * units it was divided into, so the analysis runs against the
         * per-unit handle built beside the payload at create. For a
         * single projectile the two are the same handle.
         *
         * The library's own header documents which branch runs on
         * which parameters and skips the rest, so the declared target
         * keys are passed straight through and an undeclared branch
         * leaves its channels at zero. */
        "    const K26AstroImpactor *_kfl_u = payu && payu[%d]\n"
        "        ? (const K26AstroImpactor *)payu[%d] : _kfl_h;\n"
        "    K26AstroImpactEvent ev = k26astro_impactor_analyse_impact(\n"
        "        _kfl_u, _kfl_vc, _kfl_cos,\n"
        "        pp[6], pp[8], pp[9], pp[10], pp[11], pp[12], pp[13]);\n"
        /* Only the four fields this version's decision logic reads are
         * filled. Writing the others would put a mapping in the
         * artifact that nothing checks and nothing consumes, which is
         * the shape of the defect the bumper key above exists to
         * correct. */
        "    K26AstroTargetStructureSpec spec;\n"
        "    memset(&spec, 0, sizeof spec);\n"
        "    spec.outer_thickness_m      = pp[7];\n"
        "    spec.bumper_gap_m           = pp[9];\n"
        "    spec.inner_thickness_m      = pp[14];\n"
        "    spec.monolithic_thickness_m = pp[15];\n"
        "    ch[8]  = ev.penetrates ? 1.0 : 0.0;\n"
        "    ch[9]  = ev.critical_diameter_m;\n"
        "    ch[10] = ev.monolithic_penetration_m;\n"
        /* The delivered energy is the landed total's, so it is
         * computed from the whole declared projectile and then scaled
         * by the fraction that landed, exactly as the momentum is.
         * Unscaled it would report a swarm's whole release as arriving
         * on the target and a policy could not tell one pattern from
         * the other. */
        "    ch[11] = _kfl_frac *\n"
        "        k26astro_impactor_energy_delivered_j(_kfl_h, &ev, &spec);\n"
        "}\n\n", tb, p, p);
}

/* The on_step body: action names in scope as read-only scalars, body
 * state readable and assignable by dotted name, run once per external
 * step before the world advances, identically in both modes. */

int rl_emit_on_step(FILE *out, RlModel *m,
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
        /* Before the body-state rewrite too, and for the same reason:
         * a qualified action name is a dotted shape, and the
         * body-state resolver would report it as an unknown body. */
        if (rl_agent_scan_stmts_(m, m->on_step->children, arena, diag)) {
            return 1;
        }
        if (rl_bs_rewrite_stmts(m, m->on_step->children, arena, diag)) {
            return 1;
        }
        rl_emit_state_accessors_(out, m);
        /* After the rewrite above, which is what populates the
         * actuator reference table this emits accessors for. */
        rl_emit_actuators(out, m);
        /* One helper per engagement, and the index of each left on the
         * statement so the shared statement emitter calls the right
         * one. Engagements were collected in source order, which is
         * the order they run in. */
        for (int i = 0; i < m->n_engages; i++) {
            m->engages[i].node->position.kind = KFLV_INT;
            m->engages[i].node->position.u.i  = (long)i;
            rl_emit_engage_one_(out, m, i);
        }

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
                rl_bs_fn_name(m, i, 0, nm, sizeof nm);
                fns[at].name  = kflc_arena_strdup(arena, nm);
                fns[at].arity = 1;
                at++;
                rl_bs_fn_name(m, i, 1, nm, sizeof nm);
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
        rl_emit_actuators(out, m);
    }

    /* The payload handles, the parameter store, the control period and
     * the engagement block travel into the step body because an
     * `engage` statement reads all four: the period is the dwell an
     * engagement lasts, and the block is where its result is recorded
     * for the observation to publish. A program with no effector passes
     * nulls through and the block is never touched. */
    fputs("static void kflrl_on_step_(K26AstroWorld *world, "
          "const double *_kfl_act_v,\n"
          "                           KflrlAct *_kfl_a, "
          "void *const *_kfl_pay,\n"
          "                           void *const *_kfl_payu,\n"
          "                           const double *_kfl_payp,\n"
          "                           double _kfl_dt, KflrlEng *_kfl_eng)\n"
          "{\n"
          "    (void)world; (void)_kfl_act_v; (void)_kfl_a;\n"
          "    (void)_kfl_pay; (void)_kfl_payu; (void)_kfl_payp;\n"
          "    (void)_kfl_dt; (void)_kfl_eng;\n", out);
    if (m->on_step) {
        /* Every agent's action channels are in scope. A name unique
         * across the blocks is bound unqualified as well as
         * qualified; a name two blocks share is bound qualified only,
         * and an unqualified use of it was refused above rather than
         * resolved to one of the two. */
        for (int i = 0; i < m->n_actions; i++) {
            const char *nm = m->actions[i]->name;
            if (rl_action_declarers_(m, nm) <= 1) {
                fprintf(out,
                    "    const double %s = _kfl_act_v ? _kfl_act_v[%d] : "
                    "0.0; (void)%s;\n", nm, i, nm);
            }
            if (m->n_agents == 0) continue;
            char q[KFLC_OBS_NAME_MAX + 32];
            rl_qual_ident(rl_act_agent(m, i), nm, q, sizeof q);
            fprintf(out,
                "    const double %s = _kfl_act_v ? _kfl_act_v[%d] : 0.0;"
                " (void)%s;\n", q, i, q);
        }

        KflcExprBinding *live = NULL;
        int live_n = 0, live_cap = 0;
        rl_collect_form_args(form, arena, &live, &live_n, &live_cap);
        for (int i = 0; i < m->n_actions; i++) {
            const char *nm = m->actions[i]->name;
            if (rl_action_declarers_(m, nm) <= 1) {
                rl_push_binding(arena, &live, &live_n, &live_cap, nm,
                                 KFLT_DOUBLE);
            }
            if (m->n_agents == 0) continue;
            char q[KFLC_OBS_NAME_MAX + 32];
            rl_qual_ident(rl_act_agent(m, i), nm, q, sizeof q);
            rl_push_binding(arena, &live, &live_n, &live_cap,
                             kflc_arena_strdup(arena, q), KFLT_DOUBLE);
        }
        for (const KflcNode *s = m->on_step->children; s; s = s->next) {
            rl_collect_lets(s, arena, &live, &live_n, &live_cap);
        }
        for (int i = 0; i < m->n_bs; i++) {
            char nm[192], call[256];
            rl_bs_fn_name(m, i, 0, nm, sizeof nm);
            snprintf(call, sizeof call, "%s(world)", nm);
            rl_push_binding(arena, &live, &live_n, &live_cap,
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
            rl_push_binding(arena, &live, &live_n, &live_cap,
                             kflc_arena_strdup(arena, call), KFLT_DOUBLE);
        }
        rl_push_binding(arena, &live, &live_n, &live_cap, "world",
                         KFLT_OPAQUE);
        live[live_n - 1].type_subtype = "world";
        /* The environment's actuator block, which the rewritten
         * commands and readings are called with. Like `world` it is a
         * name the block never sees in source: a program writes
         * `<body>.<component>.<field>` and the rewrite supplies this. */
        rl_push_binding(arena, &live, &live_n, &live_cap, "_kfl_a",
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

static void rl_check_objective_names_(const RlModel *m, int ag,
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
            char an[KFLC_OBS_NAME_MAX], cn[KFLC_OBS_NAME_MAX];
            if (rl_dotted_split(id, an, sizeof an, cn, sizeof cn)) {
                int qa = rl_agent_index_of(m, an);
                if (qa >= 0 && rl_agent_has_channel(m, qa, cn)) return;
                if (qa >= 0) {
                    kflc_diag_errorf(diag, line,
                        "%s: `%s`: agent `%s` declares no channel called "
                        "`%s`; a qualified name names one of that "
                        "agent's own actions or observation components",
                        ctx_word, id, an, cn);
                    return;
                }
            }
            /* Body state is addressed by dotted name inside on_step
             * and nowhere else; state reaches an objective through
             * observation channels. */
            kflc_diag_errorf(diag, line,
                "%s: `%s`: body state is readable and assignable only "
                "inside an on_step block; an objective reads state "
                "through `observe ... as` channels", ctx_word, id);
            return;
        }
        if (m->agent_count > 1) {
            if (ag >= 0 && rl_agent_has_channel(m, ag, id)) return;
            int n_owners = 0;
            int owner = rl_channel_owner(m, id, &n_owners);
            if (owner >= 0) {
                kflc_diag_errorf(diag, line,
                    "%s: `%s` is a channel of agent `%s` and this "
                    "expression is %s, so it names the agent it reads: "
                    "write `%s.%s`",
                    ctx_word, id, m->agents[owner].name,
                    ag >= 0 ? "another agent's" : "the episode's, which "
                              "belongs to no agent",
                    m->agents[owner].name, id);
                return;
            }
        } else if (rl_scope_name_taken(m, id)) {
            return;   /* action / channel */
        }
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
            rl_find_world_binding(m->world->children, id, 0, &top);
        if (b && !top) {
            kflc_diag_errorf(diag, line,
                "%s: `%s` is declared inside a nested block of the "
                "world body and is not in scope here; declare it at "
                "the top level of `fn world` to read it", ctx_word, id);
        } else if (b && !rl_is_scalar_type(b->type)) {
            kflc_diag_errorf(diag, line,
                "%s: `%s` is not a scalar; only double, int, and bool "
                "world bindings are readable here", ctx_word, id);
        }
        return;
    }
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            rl_check_objective_names_(m, ag, form, e->u.call.args[i],
                                      ctx_word, line, diag);
        }
        return;
    case KFLE_UNARY:
        rl_check_objective_names_(m, ag, form, e->u.un.operand, ctx_word,
                                  line, diag);
        return;
    case KFLE_BINARY:
        rl_check_objective_names_(m, ag, form, e->u.bin.lhs, ctx_word,
                                  line, diag);
        rl_check_objective_names_(m, ag, form, e->u.bin.rhs, ctx_word,
                                  line, diag);
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            rl_check_objective_names_(m, ag, form, e->u.vec.elems[i],
                                      ctx_word, line, diag);
        }
        return;
    case KFLE_INDEX:
        rl_check_objective_names_(m, ag, form, e->u.index.base, ctx_word,
                                  line, diag);
        rl_check_objective_names_(m, ag, form, e->u.index.idx, ctx_word,
                                  line, diag);
        return;
    case KFLE_INT_LIT:
    case KFLE_FLOAT_LIT:
        return;
    }
}

/* Reward, terminal adjustment, and termination predicate. Absent
 * blocks give the documented defaults: an all-zero reward stream, no
 * terminal adjustment, no predicate termination. */

static int rl_emit_objective_one_(FILE *out, const RlModel *m, int ag,
                                  const KflcNode *form, KflcArena *arena,
                                  const char *name, const char *ret,
                                  const KflcAttr *attr, const char *absent,
                                  const char *word,
                                  KflcExprFn *user_fn_arr, int n_user_fns,
                                  KflcDiag *diag)
{
    fprintf(out,
        "static %s %s(const double *_kfl_obs_v,\n"
        "        const double *_kfl_act_v, uint32_t _kfl_nsteps,\n"
        "        const double *_kfl_world_v)\n"
        "{\n"
        "    (void)_kfl_obs_v; (void)_kfl_act_v; (void)_kfl_nsteps; "
        "(void)_kfl_world_v;\n",
        ret, name);
    if (!attr || !attr->expr) {
        /* An absent block takes its documented default and reads
         * nothing, so it needs neither a scope nor the bindings that
         * would resolve one. */
        fprintf(out, "    return %s;\n", absent);
        fputs("}\n\n", out);
        return 0;
    }

    rl_check_objective_names_(m, ag, form, attr->expr, word,
                              attr->line, diag);
    if (diag->errors) return 1;
    char what[80];
    snprintf(what, sizeof what, "the `%s` expression", word);
    if (rl_reject_impure_(form, attr->expr, what, attr->line, diag)) {
        return 1;
    }

    /* Both rewrites run before the scope is built, because the scope
     * is built from the identifiers the expression holds and these
     * two are what settle their final spelling: a qualified read
     * becomes the identifier the prelude declares, and `episode.steps`
     * becomes the step count's own name. The two checks above run
     * first and are unaffected, reporting the names the source wrote. */
    rl_qual_rewrite_expr(m, attr->expr, arena);
    rl_rewrite_steps(attr->expr, arena);

    RlUsedNames used;
    if (rl_used_names_build(&used, attr->expr, arena)) {
        kflc_diag_errorf(diag, attr->line,
            "%s: out of memory collecting the names the `%s` "
            "expression reads", word, word);
        return 1;
    }

    KflcExprBinding *live = NULL;
    int live_n = 0, live_cap = 0;
    rl_scope_bindings(m, ag, form, arena, &live, &live_n, &live_cap,
                      &used);
    KflcExprCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.bindings   = live;
    ctx.n_bindings = live_n;
    ctx.fns        = user_fn_arr;
    ctx.n_fns      = n_user_fns;
    ctx.form       = form;

    rl_emit_scope_prelude(out, m, ag, 4, &used);

    if (strcmp(ret, "int") == 0) {
        fputs("    return (", out);
        if (kflc_emit_expr(out, attr->expr, &ctx, diag)) return 1;
        fputs(") ? 1 : 0;\n", out);
    } else {
        fputs("    return (double)(", out);
        if (kflc_emit_expr(out, attr->expr, &ctx, diag)) return 1;
        fputs(");\n", out);
    }
    fputs("}\n\n", out);
    return 0;
}

int rl_emit_objective(FILE *out, const RlModel *m,
                              const KflcNode *form, KflcArena *arena,
                              KflcExprFn *user_fn_arr, int n_user_fns,
                              KflcDiag *diag)
{
    /* One reward and one terminal adjustment per agent. Termination is
     * per environment and there is one of it: the episode ends for
     * every agent at once, on `terminated when`, on horizon
     * truncation, or on fault. */
    for (int a = 0; a < m->agent_count; a++) {
        char rn[64], tn[64];
        snprintf(rn, sizeof rn, "kflrl_reward_%d_", a);
        snprintf(tn, sizeof tn, "kflrl_terminal_%d_", a);
        if (rl_emit_objective_one_(out, m, a, form, arena, rn, "double",
                                   m->agents[a].reward, "0.0", "reward",
                                   user_fn_arr, n_user_fns, diag)) {
            return 1;
        }
        if (rl_emit_objective_one_(out, m, a, form, arena, tn, "double",
                                   m->agents[a].terminal, "0.0", "terminal",
                                   user_fn_arr, n_user_fns, diag)) {
            return 1;
        }
    }
    if (rl_emit_objective_one_(out, m, -1, form, arena, "kflrl_terminated_",
                               "int", m->terminated_when, "0",
                               "terminated when",
                               user_fn_arr, n_user_fns, diag)) {
        return 1;
    }

    /* The two per-agent streams, gathered in agent-index order. The
     * calls are written out rather than reached through a table of
     * pointers, so the step path carries no indirect call. */
    fputs("static void kflrl_rewards_(const double *_kfl_obs_v,\n"
          "        const double *_kfl_act_v, uint32_t _kfl_nsteps,\n"
          "        const double *_kfl_world_v, double *out)\n"
          "{\n", out);
    for (int a = 0; a < m->agent_count; a++) {
        fprintf(out,
            "    out[%d] = kflrl_reward_%d_(_kfl_obs_v, _kfl_act_v, "
            "_kfl_nsteps, _kfl_world_v);\n", a, a);
    }
    fputs("}\n\n", out);
    fputs("static void kflrl_terminals_(const double *_kfl_obs_v,\n"
          "        const double *_kfl_act_v, uint32_t _kfl_nsteps,\n"
          "        const double *_kfl_world_v, double *out)\n"
          "{\n", out);
    for (int a = 0; a < m->agent_count; a++) {
        fprintf(out,
            "    out[%d] = kflrl_terminal_%d_(_kfl_obs_v, _kfl_act_v, "
            "_kfl_nsteps, _kfl_world_v);\n", a, a);
    }
    fputs("}\n\n", out);
    return 0;
}
