/* emit_rl_helpers.c - emit-side helpers: scope preludes, qualified names,
 * draws, state writes and the model counts. */
#include "emit_rl_internal.h"

static void rl_emit_indent_(FILE *out, int n);
static const char *rl_obs_chan_name_(const RlModel *m, int i, int c,
                                     char *buf, size_t cap);
static int rl_state_key_index_(const char *k);
static int rl_dotted_split3_(const char *name, char *a, size_t acap,
                             char *b, size_t bcap, char *c, size_t ccap);
static int rl_act_resolve_(RlModel *m, const char *name, int write,
                           int line, KflcDiag *diag);
static int rl_bs_slot_(RlModel *m, int body, int key, int write, int line,
                       KflcDiag *diag);
static int rl_bs_resolve_(RlModel *m, const char *name, int write,
                          int line, KflcDiag *diag);
static int rl_bs_rewrite_expr_(RlModel *m, KflcExpr *e, int line,
                               KflcArena *arena, KflcDiag *diag);

/* ---- Emit-side helpers ---------------------------------------------- */

static void rl_emit_indent_(FILE *out, int n)
{
    for (int i = 0; i < n; i++) fputc(' ', out);
}

void rl_emit_string_literal(FILE *out, const char *s)
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

void rl_collect_form_args(const KflcNode *form, KflcArena *arena,
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

void rl_push_binding(KflcArena *arena, KflcExprBinding **live,
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

void rl_collect_lets(const KflcNode *n, KflcArena *arena,
                             KflcExprBinding **live, int *live_n,
                             int *live_cap)
{
    if (!n) return;
    if ((n->kind == KFLN_STMT_LET || n->kind == KFLN_STMT_CONST) &&
        n->name) {
        rl_push_binding(arena, live, live_n, live_cap, n->name, n->type);
        (*live)[*live_n - 1].type_subtype = n->type_subtype;
        (*live)[*live_n - 1].lifetime_qualifier = n->lifetime_qualifier;
    }
    for (const KflcNode *c = n->children; c; c = c->next) {
        rl_collect_lets(c, arena, live, live_n, live_cap);
    }
    for (const KflcNode *c = n->else_children; c; c = c->next) {
        rl_collect_lets(c, arena, live, live_n, live_cap);
    }
}

/* Rewrite `episode.steps` identifiers to a C-compatible name. The
 * expression AST is not reused after emission, so an in-place rename
 * is safe; every other identifier stays untouched. */

void rl_rewrite_steps(KflcExpr *e, KflcArena *arena)
{
    if (!e) return;
    switch (e->kind) {
    case KFLE_IDENT:
        if (e->u.ident && strcmp(e->u.ident, "episode.steps") == 0) {
            e->u.ident = kflc_arena_strdup(arena, "_kfl_episode_steps");
        }
        return;
    case KFLE_UNARY:
        rl_rewrite_steps(e->u.un.operand, arena);
        return;
    case KFLE_BINARY:
        rl_rewrite_steps(e->u.bin.lhs, arena);
        rl_rewrite_steps(e->u.bin.rhs, arena);
        return;
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            rl_rewrite_steps(e->u.call.args[i], arena);
        }
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            rl_rewrite_steps(e->u.vec.elems[i], arena);
        }
        return;
    case KFLE_INDEX:
        rl_rewrite_steps(e->u.index.base, arena);
        rl_rewrite_steps(e->u.index.idx, arena);
        return;
    default:
        return;
    }
}

/* The identifier the emitted scope binds a qualified read to. The
 * agent's index rather than its name: two agent names differ, but a
 * name joined to a channel name need not, and a mangled identifier
 * that could collide would silently read the wrong agent. */

/* ---- Qualified channel names ---------------------------------------- */

void rl_qual_ident(int agent, const char *chan, char *out,
                           size_t cap)
{
    snprintf(out, cap, "_kfl_q%d_%s", agent, chan);
}

/* The channel name each `as`-bound observe component is written as in
 * source, which is the declared base name with the component suffix,
 * unqualified. */

static const char *rl_obs_chan_name_(const RlModel *m, int i, int c,
                                     char *buf, size_t cap)
{
    const char *base = rl_observe_as(m->observes[i]);
    char        cb[RL_COMP_MAX];
    const char *cmp  = rl_observe_comp(m->observes[i], c, cb, sizeof cb);
    snprintf(buf, cap, "%s%s", base ? base : "?", cmp);
    return buf;
}

int rl_agent_index_of(const RlModel *m, const char *name)
{
    if (!name) return -1;
    for (int i = 0; i < m->n_agents; i++) {
        if (strcmp(m->agents[i].name, name) == 0) return i;
    }
    return -1;
}

/* Whether agent `ag` declares a channel called `name`: one of its
 * actions, or one component of one of its observation channels.
 *
 * Every qualified read in every objective expression asks this once,
 * and the walk below reads every declaration in the program to answer
 * it, so on a program of many agents and many channels the two counts
 * multiply. The index the collector builds answers the same question
 * by lookup, and is filled by the same two loops the walk uses so that
 * the two agree by construction; the walk remains for a caller that
 * asks before the declarations are all in. */

int rl_agent_has_channel(const RlModel *m, int ag, const char *name)
{
    if (!name) return 0;
    if (m->agent_names_built && ag >= 0 && ag < m->agent_count) {
        return rl_used_names_has(&m->agent_names[ag], name);
    }
    for (int i = 0; i < m->n_actions; i++) {
        if (rl_act_agent(m, i) != ag) continue;
        const char *an = m->actions[i]->name;
        if (an && strcmp(an, name) == 0) return 1;
    }
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_obs_agent(m, i) != ag) continue;
        for (int c = 0; c < rl_observe_width(m->observes[i]); c++) {
            char nb[KFLC_OBS_NAME_MAX];
            if (strcmp(rl_obs_chan_name_(m, i, c, nb, sizeof nb),
                       name) == 0) {
                return 1;
            }
        }
    }
    return 0;
}

/* The per-agent name index, built once the declarations are all
 * collected. It holds exactly the names the walk above would match:
 * each agent's action names and every component name of each of its
 * observation channels, in the same spelling. Returns 0 on success and
 * 1 when a table cannot be allocated, which the caller reports.
 *
 * Two passes because the tables are sized from what they will hold and
 * never grow: the first counts each agent's names, the second stores
 * them. */

int rl_build_agent_names(RlModel *m, KflcArena *arena)
{
    long count[RL_MAX_AGENTS];
    for (int a = 0; a < RL_MAX_AGENTS; a++) count[a] = 0;
    for (int i = 0; i < m->n_actions; i++) {
        int ag = rl_act_agent(m, i);
        if (ag >= 0 && ag < m->agent_count) count[ag]++;
    }
    for (int i = 0; i < m->n_observes; i++) {
        int ag = rl_obs_agent(m, i);
        if (ag >= 0 && ag < m->agent_count) {
            count[ag] += rl_observe_width(m->observes[i]);
        }
    }
    for (int a = 0; a < m->agent_count; a++) {
        if (rl_used_names_reserve(&m->agent_names[a], arena, count[a])) {
            return 1;
        }
    }
    for (int i = 0; i < m->n_actions; i++) {
        int ag = rl_act_agent(m, i);
        const char *an = m->actions[i]->name;
        if (ag < 0 || ag >= m->agent_count || !an) continue;
        rl_used_names_add(&m->agent_names[ag], an);
    }
    for (int i = 0; i < m->n_observes; i++) {
        int ag = rl_obs_agent(m, i);
        if (ag < 0 || ag >= m->agent_count) continue;
        int width = rl_observe_width(m->observes[i]);
        for (int c = 0; c < width; c++) {
            char nb[KFLC_OBS_NAME_MAX];
            const char *nm = rl_obs_chan_name_(m, i, c, nb, sizeof nb);
            const char *kept = kflc_arena_strdup(arena, nm);
            if (!kept) return 1;
            rl_used_names_add(&m->agent_names[ag], kept);
        }
    }
    m->agent_names_built = 1;
    return 0;
}

/* The agent that owns the only channel called `name`, or -1 when no
 * agent declares it. `*n_owners` reports how many do, which is what
 * tells an unambiguous unqualified read from one that would be a
 * guess. */

int rl_channel_owner(const RlModel *m, const char *name,
                             int *n_owners)
{
    int owner = -1, n = 0;
    for (int a = 0; a < m->agent_count; a++) {
        if (!rl_agent_has_channel(m, a, name)) continue;
        if (n == 0) owner = a;
        n++;
    }
    if (n_owners) *n_owners = n;
    return owner;
}

/* The two declaration sites of an action name two blocks share, for
 * the diagnostic that refuses an unqualified read of it. */

void rl_action_sites(const RlModel *m, const char *name,
                             int *a0, int *l0, int *a1, int *l1)
{
    int seen = 0;
    for (int i = 0; i < m->n_actions; i++) {
        const char *an = m->actions[i]->name;
        if (!an || strcmp(an, name) != 0) continue;
        if (seen == 0) { *a0 = rl_act_agent(m, i); *l0 = m->actions[i]->line; }
        else if (seen == 1) {
            *a1 = rl_act_agent(m, i); *l1 = m->actions[i]->line;
        }
        seen++;
    }
}

/* The scope one reward, terminal or termination expression reads.
 * `ag` is the agent whose objective is being emitted, or -1 for the
 * episode's own `terminated when`, which sits in no block and so has
 * no agent to default to.
 *
 * A channel is bound unqualified when the reading expression owns it,
 * and when the program declares at most one agent, in which case there
 * is nothing for a qualifier to disambiguate. Every channel of every
 * agent is bound under its qualified identifier as well, which is what
 * lets a zero-sum reward be written once as the negation of the other
 * agent's rather than twice as two expressions a later edit can pull
 * apart. */

/* ---- The names an objective expression reads ------------------------ */

/* FNV-1a over the identifier's bytes. The set holds identifiers, which
 * are short and share long prefixes with one another, and this mixes
 * every byte rather than the first few. */

static unsigned rl_name_hash_(const char *s)
{
    unsigned h = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        h ^= (unsigned)*p;
        h *= 16777619u;
    }
    return h;
}

/* Size a table for `count` names and take its storage from the arena,
 * which is what frees it: the sets outlive the walks that fill them
 * and are released with everything else this compilation allocated.
 * Twice the count, rounded up to a power of two, so the table is at
 * most half full and linear probing terminates. Returns 0 on success
 * and 1 when the storage cannot be taken. */

int rl_used_names_reserve(RlUsedNames *set, KflcArena *arena, long count)
{
    set->slot = NULL;
    set->cap  = 0;
    set->n    = 0;
    if (count <= 0) return 0;
    long cap = 16;
    while (cap < 2 * count) {
        if (cap > (long)1 << 28) return 1;
        cap *= 2;
    }
    set->slot = (const char **)kflc_arena_alloc(
        arena, (size_t)cap * sizeof *set->slot);
    if (!set->slot) return 1;
    set->cap = (int)cap;
    return 0;
}

/* Store `name`, which is borrowed and must outlive the set. Storing
 * more names than the table was reserved for would fill it and hang
 * the probe, so the count the caller reserved for is the count it may
 * add; a name already present is not stored twice. */

void rl_used_names_add(RlUsedNames *set, const char *name)
{
    if (!set->cap || !name) return;
    unsigned i = rl_name_hash_(name) & (unsigned)(set->cap - 1);
    while (set->slot[i]) {
        if (strcmp(set->slot[i], name) == 0) return;
        i = (i + 1u) & (unsigned)(set->cap - 1);
    }
    set->slot[i] = name;
    set->n++;
}

/* Whether `name` was stored. An empty set holds nothing, which is the
 * right answer for an expression that names no identifier and for an
 * agent that declares no channel. */

int rl_used_names_has(const RlUsedNames *set, const char *name)
{
    if (!set || set->cap == 0 || !name) return 0;
    unsigned i = rl_name_hash_(name) & (unsigned)(set->cap - 1);
    while (set->slot[i]) {
        if (strcmp(set->slot[i], name) == 0) return 1;
        i = (i + 1u) & (unsigned)(set->cap - 1);
    }
    return 0;
}

/* Every identifier in the subtree, counted when `set` is NULL and
 * inserted when it is not. The two passes walk the same shapes, which
 * is what makes the count an upper bound on what the insert pass
 * stores. */

static long rl_walk_idents_(const KflcExpr *e, RlUsedNames *set)
{
    if (!e) return 0;
    switch (e->kind) {
    case KFLE_IDENT:
        if (!e->u.ident) return 0;
        if (set) rl_used_names_add(set, e->u.ident);
        return 1;
    case KFLE_UNARY:
        return rl_walk_idents_(e->u.un.operand, set);
    case KFLE_BINARY:
        return rl_walk_idents_(e->u.bin.lhs, set) +
               rl_walk_idents_(e->u.bin.rhs, set);
    case KFLE_CALL: {
        long n = 0;
        for (int i = 0; i < e->u.call.n_args; i++) {
            n += rl_walk_idents_(e->u.call.args[i], set);
        }
        return n;
    }
    case KFLE_VEC_LIT: {
        long n = 0;
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            n += rl_walk_idents_(e->u.vec.elems[i], set);
        }
        return n;
    }
    case KFLE_INDEX:
        return rl_walk_idents_(e->u.index.base, set) +
               rl_walk_idents_(e->u.index.idx, set);
    default:
        return 0;
    }
}

/* Build the set of identifiers `e` names. Returns 0 on success and 1
 * when the table cannot be allocated, which the caller reports rather
 * than emitting a scope that would be missing what the expression
 * reads. The identifiers are borrowed from the expression, which
 * outlives the set. The first walk counts them and the second stores
 * them, so the table is sized from an upper bound on what it holds. */

int rl_used_names_build(RlUsedNames *set, const KflcExpr *e,
                                KflcArena *arena)
{
    long count = rl_walk_idents_(e, NULL);
    if (rl_used_names_reserve(set, arena, count)) return 1;
    rl_walk_idents_(e, set);
    return 0;
}

/* ---- RL expression scope -------------------------------------------- *
 *
 * The `terminated when`, `reward`, and `terminal` expressions read
 * action channels, observation channel components, `episode.steps`,
 * world scalar bindings, and form arguments. Emission declares one
 * const double local per action, channel component, and captured
 * world scalar so the expression emitter resolves the KFL names as
 * ordinary scalar bindings.
 *
 * `used` holds the identifiers the expression contains. Only the
 * channels it names are declared: a local the expression cannot
 * mention changes no emitted arithmetic, and on a program of many
 * agents the ones it cannot mention are nearly all of them. World
 * scalars and the step count are declared whatever is read, both being
 * bounded by a small declared count rather than by the channel
 * product. */

void rl_emit_scope_prelude(FILE *out, const RlModel *m, int ag,
                                   int indent, const RlUsedNames *used)
{
    int have_q   = (m->n_agents > 0);
    int bare_all = (m->agent_count <= 1);
    for (int i = 0; i < m->n_actions; i++) {
        const char *nm = m->actions[i]->name;
        int owner = rl_act_agent(m, i);
        if ((bare_all || owner == ag) && rl_used_names_has(used, nm)) {
            rl_emit_indent_(out, indent);
            fprintf(out,
                "const double %s = _kfl_act_v ? _kfl_act_v[%d] : 0.0; "
                "(void)%s;\n", nm, i, nm);
        }
        if (!have_q) continue;
        char q[KFLC_OBS_NAME_MAX + 32];
        rl_qual_ident(owner, nm, q, sizeof q);
        if (!rl_used_names_has(used, q)) continue;
        rl_emit_indent_(out, indent);
        fprintf(out,
            "const double %s = _kfl_act_v ? _kfl_act_v[%d] : 0.0; "
            "(void)%s;\n", q, i, q);
    }
    /* The first channel index of each observe, carried across the walk
     * rather than recomputed per observe: rl_obs_offset sums the widths
     * before its argument, so asking it once per channel makes the walk
     * quadratic in the declaration count. */
    int base_off = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int owner = rl_obs_agent(m, i);
        int width = rl_observe_width(m->observes[i]);
        for (int c = 0; c < width; c++) {
            char nb[KFLC_OBS_NAME_MAX];
            const char *nm = rl_obs_chan_name_(m, i, c, nb, sizeof nb);
            int off = base_off + c;
            if ((bare_all || owner == ag) && rl_used_names_has(used, nm)) {
                rl_emit_indent_(out, indent);
                fprintf(out,
                    "const double %s = _kfl_obs_v[%d]; (void)%s;\n",
                    nm, off, nm);
            }
            if (!have_q) continue;
            char q[KFLC_OBS_NAME_MAX + 32];
            rl_qual_ident(owner, nm, q, sizeof q);
            if (!rl_used_names_has(used, q)) continue;
            rl_emit_indent_(out, indent);
            fprintf(out,
                "const double %s = _kfl_obs_v[%d]; (void)%s;\n",
                q, off, q);
        }
        base_off += width;
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

/* Bindings matching rl_emit_scope_prelude plus form arguments, held
 * down to the same `used` set for the same reason: a binding the
 * expression cannot name resolves nothing, and the expression emitter
 * scans this table for every identifier it emits. Names outside the
 * set are dropped whole, so the first-match order among the names that
 * remain is the order they had before. */

void rl_scope_bindings(const RlModel *m, int ag,
                               const KflcNode *form,
                               KflcArena *arena, KflcExprBinding **live,
                               int *live_n, int *live_cap,
                               const RlUsedNames *used)
{
    int have_q   = (m->n_agents > 0);
    int bare_all = (m->agent_count <= 1);
    rl_collect_form_args(form, arena, live, live_n, live_cap);
    for (int i = 0; i < m->n_actions; i++) {
        int owner = rl_act_agent(m, i);
        const char *an = m->actions[i]->name;
        if ((bare_all || owner == ag) && rl_used_names_has(used, an)) {
            rl_push_binding(arena, live, live_n, live_cap,
                             an, KFLT_DOUBLE);
        }
        if (!have_q) continue;
        char q[KFLC_OBS_NAME_MAX + 32];
        rl_qual_ident(owner, an, q, sizeof q);
        if (!rl_used_names_has(used, q)) continue;
        rl_push_binding(arena, live, live_n, live_cap,
                         kflc_arena_strdup(arena, q), KFLT_DOUBLE);
    }
    for (int i = 0; i < m->n_observes; i++) {
        if (!rl_observe_as(m->observes[i])) continue;
        int owner = rl_obs_agent(m, i);
        int width = rl_observe_width(m->observes[i]);
        for (int c = 0; c < width; c++) {
            char nb[KFLC_OBS_NAME_MAX];
            const char *nm = rl_obs_chan_name_(m, i, c, nb, sizeof nb);
            if ((bare_all || owner == ag) && rl_used_names_has(used, nm)) {
                rl_push_binding(arena, live, live_n, live_cap,
                                 kflc_arena_strdup(arena, nm), KFLT_DOUBLE);
            }
            if (!have_q) continue;
            char q[KFLC_OBS_NAME_MAX + 32];
            rl_qual_ident(owner, nm, q, sizeof q);
            if (!rl_used_names_has(used, q)) continue;
            rl_push_binding(arena, live, live_n, live_cap,
                             kflc_arena_strdup(arena, q), KFLT_DOUBLE);
        }
    }
    for (int i = 0; i < m->n_wscal; i++) {
        rl_push_binding(arena, live, live_n, live_cap,
                         m->wscal[i]->name, KFLT_DOUBLE);
    }
    rl_push_binding(arena, live, live_n, live_cap,
                     "_kfl_episode_steps", KFLT_DOUBLE);
}

/* Rewrite `<agent>.<channel>` reads into the identifiers the prelude
 * binds. It runs after the names have been checked and before every
 * other resolution, so a qualified read never reaches the body-state
 * resolver, which would report it as an unknown body. */

void rl_qual_rewrite_expr(const RlModel *m, KflcExpr *e,
                                  KflcArena *arena)
{
    if (!e) return;
    switch (e->kind) {
    case KFLE_IDENT: {
        char an[KFLC_OBS_NAME_MAX], cn[KFLC_OBS_NAME_MAX];
        if (!rl_dotted_split(e->u.ident, an, sizeof an, cn, sizeof cn)) {
            return;
        }
        int ag = rl_agent_index_of(m, an);
        if (ag < 0 || !rl_agent_has_channel(m, ag, cn)) return;
        char q[KFLC_OBS_NAME_MAX + 32];
        rl_qual_ident(ag, cn, q, sizeof q);
        e->u.ident = kflc_arena_strdup(arena, q);
        return;
    }
    case KFLE_UNARY:
        rl_qual_rewrite_expr(m, e->u.un.operand, arena);
        return;
    case KFLE_BINARY:
        rl_qual_rewrite_expr(m, e->u.bin.lhs, arena);
        rl_qual_rewrite_expr(m, e->u.bin.rhs, arena);
        return;
    case KFLE_INDEX:
        rl_qual_rewrite_expr(m, e->u.index.base, arena);
        rl_qual_rewrite_expr(m, e->u.index.idx, arena);
        return;
    case KFLE_VEC_LIT:
        for (int i = 0; i < e->u.vec.n_elems; i++) {
            rl_qual_rewrite_expr(m, e->u.vec.elems[i], arena);
        }
        return;
    case KFLE_CALL:
        for (int i = 0; i < e->u.call.n_args; i++) {
            rl_qual_rewrite_expr(m, e->u.call.args[i], arena);
        }
        return;
    default:
        return;
    }
}

/* ---- Draw and state-write emission ----------------------------------- */

/* Emit the draw expression for a distribution call at the given
 * stream class and channel, with `envi` and `ep` the in-scope
 * environment and episode variables. Arguments are emitted in the
 * given expression context (form arguments only in practice). */

int rl_emit_draw(FILE *out, const KflcExpr *dist,
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

void rl_emit_body_write(FILE *out, int indent, const char *lv,
                                const char *key, const char *value_text)
{
    if (rl_is_state_key(key)) {
        kflc_emit_body_state_write(out, indent, lv, key, value_text);
        return;
    }
    rl_emit_indent_(out, indent);
    fprintf(out, "%s%s = (%s);\n", lv, key, value_text);
}

/* As rl_emit_body_write, with the value already emitted into a
 * named double variable. */

void rl_emit_body_write_var(FILE *out, int indent, const char *lv,
                                    const char *key, const char *var)
{
    rl_emit_body_write(out, indent, lv, key, var);
}

/* ---- Generated-code sections ----------------------------------------- */

/* The number of bodies in this program that carry a vehicle
 * assembly. Each gets one vehicle per environment, owned by the
 * handle. */

int rl_n_vehicles(const RlModel *m)
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

int rl_has_relative_observe(const RlModel *m)
{
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_observe_form(m->observes[i]) == RL_OBS_REL) return 1;
    }
    /* A plan written in a body's local-vertical local-horizontal frame
     * is resolved through the same routine the relative observe uses,
     * so it pulls in the same library on the same terms. A plan in an
     * inertial frame needs none of it. */
    for (int k = 0; k < m->n_refs; k++) {
        if (m->refs[k].frame_kind == K26RL_REF_FRAME_LVLH) return 1;
    }
    return 0;
}

int rl_n_pay_kind(const RlModel *m, int kind)
{
    int n = 0;
    for (int p = 0; p < m->n_payloads; p++) {
        if (m->payloads[p].kind == kind) n++;
    }
    return n;
}

int rl_n_detect(const RlModel *m)
{
    int n = 0;
    for (int p = 0; p < m->n_payloads; p++) {
        if (m->payloads[p].kind >= 0 &&
            RL_PAY_KIND_[m->payloads[p].kind].is_detect) n++;
    }
    return n;
}

int rl_n_effector(const RlModel *m)
{
    int n = 0;
    for (int p = 0; p < m->n_payloads; p++) {
        if (m->payloads[p].kind >= 0 &&
            RL_PAY_KIND_[m->payloads[p].kind].is_effector) n++;
    }
    return n;
}

/* Radar payloads that declared a chaff cloud around their reference
 * target. The count decides whether the cloud's contribution is
 * computed at all: a payload that declared no strip count has no chaff
 * term in the emitted source, so its cross-section is the expression it
 * was before this surface existed, which is what makes this addition
 * provably additive rather than additive by arithmetic. */

int rl_n_chaff(const RlModel *m)
{
    int n = 0;
    for (int p = 0; p < m->n_payloads; p++) {
        if (m->payloads[p].kind != RL_PAY_DETECT_RADAR) continue;
        if (m->payloads[p].attr[RL_PAY_RADAR_CHAFF_N]) n++;
    }
    return n;
}

/* Engagements whose result lands on another payload rather than on a
 * body. The count decides whether the per-step block carries the
 * degradation store at all: a program that engages no countermeasure
 * has the store absent rather than present and empty, so its detection
 * channels are computed by exactly the code they were computed by
 * before this surface existed. */

int rl_n_softkill_engage(const RlModel *m)
{
    int n = 0;
    for (int e = 0; e < m->n_engages; e++) {
        int p = m->engages[e].payload;
        if (p < 0 || m->payloads[p].kind < 0) continue;
        if (RL_PAY_KIND_[m->payloads[p].kind].is_softkill) n++;
    }
    return n;
}

/* Whether a countermeasure engagement can reach detection payload `p`
 * observing body `t`. A countermeasure degrades the victim's view of
 * the craft that carries it, so the pair is reached when some
 * engagement of a countermeasure hosted on `t` is aimed at the body
 * that carries `p`. Both halves are fixed at compile time, which is
 * what keeps the read out of every program that cannot have one.
 *
 * `want_jammer` selects the class: a jammer reaches radar and infrared
 * payloads by different routes, a decoy reaches every detection kind. */

int rl_softkill_reaches(const RlModel *m, int p, int t,
                                int want_jammer)
{
    if (p < 0 || t < 0) return 0;
    for (int e = 0; e < m->n_engages; e++) {
        int ep = m->engages[e].payload;
        if (ep < 0 || m->payloads[ep].kind < 0) continue;
        int k = m->payloads[ep].kind;
        if (!RL_PAY_KIND_[k].is_softkill) continue;
        if (want_jammer != (k == RL_PAY_JAMMER)) continue;
        if (m->payloads[ep].body != t) continue;
        if (m->engages[e].target != m->payloads[p].body) continue;
        return 1;
    }
    return 0;
}

/* Whether detection payload `p` has a detect observe of body `b`. The
 * jamming ratio divides by the protected craft's cross-section, and the
 * chaff keys on a radar payload describe a cloud around *that payload's
 * reference target*, so the cloud belongs in that ratio exactly when
 * the craft the jammer protects is the craft the payload is pointed at.
 * Both are declarations, so this is settled here rather than while
 * stepping. */

int rl_detect_observes(const RlModel *m, int p, int b)
{
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_observe_form(m->observes[i]) != RL_OBS_DET) continue;
        if (m->obs_payload[i] != p) continue;
        if (m->obs_target[i] == b) return 1;
    }
    return 0;
}

/* The (information state, target) pairs the binding pushes a sample
 * for. One pair per distinct target of each infostate payload, taken
 * in observe declaration order, so the push order is the program's own
 * and the same on every step and in every environment. */

/* How many datalinks this program declares. Declaration order is the
 * order transfers are offered in, and is what decides the drop-older
 * outcome where two transmitters offer one target on one instant. */

int rl_n_link(const RlModel *m)
{
    return rl_n_pay_kind(m, RL_PAY_DATALINK);
}

/* The detection payload gating one track pair's push, or -1 where the
 * pair's information state declares no `source=`. */

int rl_track_gate_pay(const RlModel *m, int payload)
{
    if (payload < 0 || payload >= m->n_payloads) return -1;
    return m->payloads[payload].src_pay;
}

int rl_track_pairs(const RlModel *m, int *pay, int *veh, int cap)
{
    int n = 0;
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_observe_form(m->observes[i]) != RL_OBS_TRK) continue;
        int p = m->obs_payload[i];
        int b = m->obs_target[i];
        if (p < 0 || b < 0) continue;
        int v = rl_veh_slot_of(m, b);
        int dup = 0;
        for (int q = 0; q < n; q++) {
            if (pay[q] == p && veh[q] == v) dup = 1;
        }
        if (dup || n >= cap) continue;
        pay[n] = p;
        veh[n] = v;
        n++;
    }
    return n;
}

/* ---- Body state inside on_step -------------------------------------- */

/* Split a name at its single dot. Returns 1 when exactly one dot sits
 * between two non-empty parts. */

int rl_dotted_split(const char *name, char *lhs, size_t lcap,
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
    int bi = rl_body_index_of(m, bn);
    if (bi < 0) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: no astro_body named `%s` is declared in this "
            "world", name, bn);
        return -2;
    }
    if (!rl_body_has_assembly(m, bi)) {
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
        !rl_body_has_assembly(m, body)) {
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

void rl_bs_fn_name(const RlModel *m, int slot, int set,
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
    if (!rl_dotted_split(name, b, sizeof b, k, sizeof k)) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s` is not a body state reference (expected "
            "`<body>.<key>`)", name);
        return -2;
    }
    /* A qualified read was rewritten before this point, so a
     * qualified name reaching here is an assignment to one. */
    int qa = rl_agent_index_of(m, b);
    if (qa >= 0 && rl_agent_has_channel(m, qa, k)) {
        kflc_diag_errorf(diag, line,
            "on_step: `%s`: `%s` is a channel of agent `%s` and is read "
            "only; an action reaches the dynamics through a body state "
            "assignment", name, k, b);
        return -2;
    }
    int bi = rl_body_index_of(m, b);
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
        rl_bs_fn_name(m, slot, 0, fn, sizeof fn);
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

int rl_bs_rewrite_stmts(RlModel *m, KflcNode *stmts,
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
            rl_bs_fn_name(m, slot, 1, fn, sizeof fn);
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
        if (rl_bs_rewrite_stmts(m, s->children, arena, diag)) {
            return 1;
        }
        if (rl_bs_rewrite_stmts(m, s->else_children, arena, diag)) {
            return 1;
        }
    }
    return 0;
}
