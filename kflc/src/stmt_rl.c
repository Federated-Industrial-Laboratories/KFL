/* stmt_rl.c - the Grammar 3.2 construct parsers: plan,
 * sensor, episode, action, on_step, objective, agent,
 * astro_payload and engage. */
#include "stmt_internal.h"

/* Statement-parse context. `g_rl_world_ctx` is raised by parser.c
 * around `fn world` body parses (kfl_stmt_set_world_ctx); the RL
 * statement keywords bind as constructs only while it is set, so those
 * words keep their ordinary-identifier reading (and the reserved-future
 * warning) in every other fn body. Inside a `fn world` body three of
 * them are decided by what follows as well, which
 * kfl_stmt_rl_word_is_construct below states. `g_rl_on_step_depth` tracks
 * nesting inside `on_step` bodies so the statements the per-step body
 * rejects can be diagnosed at parse time. */

int g_rl_world_ctx     = 0;

int g_rl_on_step_depth = 0;

static int is_reset_state_key_(const char *s);
static int is_dist_call_(const KflcExpr *e);

/* Whether a word at statement position inside a `fn world` body is a
 * Grammar 3.2 construct keyword here, or an ordinary identifier.
 *
 * Three of the words are decided by what follows them rather than by
 * the word alone, so that a Grammar 3.1 program which binds one of
 * them as a name keeps compiling: `agent alpha` and `sensor rf` open
 * their blocks, while `agent = 2.0`, `agent(3.0)`, `agent[0]` and a
 * bare `agent` stay what they were. The discrimination is exact
 * rather than a heuristic. A name-led block is followed by an
 * identifier, and no statement form in this language has the shape
 * `<identifier> <identifier>`: an assignment, an index assignment and
 * a call all put punctuation there, and juxtaposition is not
 * application in any expression. A body-led block is followed by the
 * end of its line, which is the one shape the ordinary reading also
 * has, and that shape is refused by the grammar these words joined,
 * so nothing that compiled before is lost to it.
 *
 * `episode`, `action` and `objective` are not treated this way. They
 * were placed on the reserved-name table before the constructs
 * landed, so a program binding one of them has been warned that the
 * word was going to be taken.
 *
 * `engage` joins the name-led group: `engage beam at mover` is the
 * statement and `engage = 1.0`, `engage(2.0)`, `engage[0]` and a bare
 * `engage` stay the identifier they were. It is the only word here
 * that is a construct in one block and not in another, which the
 * dispatch below decides rather than this function: what is a
 * construct is a property of the word and what follows it, and where
 * it is admissible is a property of the block. */

int kfl_stmt_rl_word_is_construct(Lexer *L, const char *s)
{
    if (strcmp(s, "episode") == 0 || strcmp(s, "action") == 0 ||
        strcmp(s, "objective") == 0) {
        return 1;
    }
    TokenKind k = T_EOF;
    if (strcmp(s, "on_step") == 0) {
        return kfl_stmt_peek_kind(L, &k) && (k == T_NEWLINE || k == T_EOF);
    }
    if (strcmp(s, "sensor") == 0 || strcmp(s, "agent") == 0 ||
        strcmp(s, "astro_payload") == 0 || strcmp(s, "engage") == 0 ||
        strcmp(s, "plan") == 0) {
        return kfl_stmt_peek_kind(L, &k) && k == T_IDENT;
    }
    return 0;
}

/* World construction, stepping and observation, which the `on_step`
 * body rejects because the episode machinery owns them in these
 * programs. The nested constructs it also rejects are decided by
 * kfl_stmt_rl_word_is_construct beside this, so a word that is an ordinary
 * identifier there is not reported as a construct. */

int kfl_stmt_is_on_step_world_stmt(const char *s)
{
    return strcmp(s, "astro_body") == 0 || strcmp(s, "step") == 0 ||
           strcmp(s, "propagate")  == 0 || strcmp(s, "observe") == 0;
}

/* Scalar state keys accepted on an episode reset line. The same keys
 * are accepted as astro_body attributes (parsed generically there;
 * the emission pass maps them). Six are translation; the other seven
 * are the body-to-world quaternion's components and the body-frame
 * angular velocity. */

static int is_reset_state_key_(const char *s)
{
    return kflc_body_state_key_index(s) >= 0;
}

/* Returns 1 when `e` is a well-formed distribution call:
 * `uniform(<low>, <high>)` or `normal(<mean>, <stddev>)`. */

static int is_dist_call_(const KflcExpr *e)
{
    return e && e->kind == KFLE_CALL && e->u.call.name &&
           (strcmp(e->u.call.name, "uniform") == 0 ||
            strcmp(e->u.call.name, "normal")  == 0) &&
           e->u.call.n_args == 2;
}

/* Error recovery: discard the rest of the current source line as raw
 * bytes, then step past the newline. Raw discard (rather than
 * token-by-token draining) because the rest of the line may hold
 * expression characters the main lexer does not tokenise. Call only
 * when `cur` has not yet reached the line's newline. */

void kfl_stmt_rl_drain_line(Lexer *L, Token *cur,
                           KflcArena *arena, int *had_error)
{
    (void)kfl_stmt_take_line_remainder(L, arena);
    kfl_stmt_advance(L, cur, had_error);
    if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
}

/* `sensor <name> ... end`. One model term per line, in the order the
 * chain applies them:
 *
 *   noise normal <mean> <sigma>       additive, one draw per step
 *   scale <relative sigma>            multiplicative, one draw per step
 *   bias_walk <sigma0> <tau> <sigma>  turn-on bias plus an in-run walk
 *   latency <whole control periods>   a fixed-depth delay, no draws
 *   quantise <lsb> [<lo> <hi>]        a step, with a declared clamp
 *   dropout <probability>             hold the last delivered value
 *
 * Order is load-bearing and is kept: a quantiser declared after a
 * noise term quantises the noisy value and one declared before it does
 * not, so the terms are carried as ordered children rather than as
 * attributes, which have no order a reader can rely on.
 *
 * Operands are plain numbers rather than expressions. A sensor's
 * parameters become compile-time tables and a coefficient the artifact
 * computes once at create, so there is nothing for an expression to
 * close over; the assembly reader's rule of a fully consumed number is
 * used, so a unit suffix is a loud refusal rather than a silent
 * truncation.
 *
 * `cur` is the `sensor` keyword on entry. */

/* `plan <name> ... end`, the knot slots a planner emits.
 *
 * The block is a declaration of an action space and of where the plan
 * those actions make is written. Its lines are:
 *
 *     file "<prefix>"          required, the path a plan is written to
 *     frame <body> <kind>      required, `lvlh` or `inertial`
 *     slots <n>                required
 *     time <lo> <hi>           required, seconds from the epoch
 *     position <lo> <hi>       required, metres
 *     velocity <lo> <hi>       required, metres per second
 *     tolerance <lo> <hi>      required, metres
 *     epoch <value>            optional, default 0
 *     provenance "<text>"      optional
 *
 * The block declares eight action channels per slot and appends them
 * as ordinary `action` statements, chained after this node so the
 * block builder takes the whole run. Everything downstream therefore
 * sees a fixed-width action space with declared bounds, and nothing
 * about a knot slot is a special case in the spec, the agent slicing
 * or the batch driver.
 *
 * The bounds are the block's rather than a convention because a
 * planner's action space is the mission's scale: a transfer's knots
 * are megametres apart and a docking approach's are metres apart, and
 * a single hard-coded pair would be wrong for both. */

KflcNode *kfl_stmt_parse_plan(Lexer *L, Token *cur,
                             KflcArena *arena, KflcDiag *diag,
                             int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0, "plan: expected a name");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_PLAN, line0);
    n->name = cur->str;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "plan `%s`: expected end of line after the name", n->name);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    } else if (kfl_stmt_at_nl(cur)) {
        kfl_stmt_advance(L, cur, had_error);
    }

    for (;;) {
        kfl_stmt_skip_newlines(L, cur, had_error);
        if (kfl_stmt_at_eof2(cur)) {
            kflc_diag_errorf(diag, line0,
                "plan `%s`: unexpected EOF (missing `end`)", n->name);
            *had_error = 1;
            return n;
        }
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            break;
        }
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, cur->line,
                "plan `%s`: expected a plan line or `end`", n->name);
            *had_error = 1;
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
            continue;
        }
        char *kw = cur->str;
        int lineK = cur->line;
        char *rest = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

        char *p = kfl_stmt_trim(rest);
        if (strcmp(kw, "file") == 0 || strcmp(kw, "provenance") == 0 ||
            strcmp(kw, "slots") == 0 || strcmp(kw, "epoch") == 0) {
            KflcValue v;
            memset(&v, 0, sizeof v);
            v.kind = KFLV_IDENT;
            v.u.s  = kflc_arena_strdup(arena, p);
            if (p[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "plan `%s`: `%s` takes a value", n->name, kw);
                *had_error = 1;
                continue;
            }
            kfl_stmt_append_attr(arena, n, kflc_arena_strdup(arena, kw), v,
                             lineK);
            continue;
        }
        if (strcmp(kw, "frame") == 0) {
            char *body = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            if (*p) { *p = '\0'; p++; }
            char *kind = kfl_stmt_trim(p);
            if (body[0] == '\0' || kind[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "plan `%s`: `frame` takes a body name and either "
                    "`lvlh` or `inertial`", n->name);
                *had_error = 1;
                continue;
            }
            KflcValue bv, kv;
            memset(&bv, 0, sizeof bv);
            memset(&kv, 0, sizeof kv);
            bv.kind = KFLV_IDENT; bv.u.s = kflc_arena_strdup(arena, body);
            kv.kind = KFLV_IDENT; kv.u.s = kflc_arena_strdup(arena, kind);
            kfl_stmt_append_attr(arena, n, "frame", bv, lineK);
            kfl_stmt_append_attr(arena, n, "kind", kv, lineK);
            continue;
        }
        if (strcmp(kw, "time") == 0 || strcmp(kw, "position") == 0 ||
            strcmp(kw, "velocity") == 0 || strcmp(kw, "tolerance") == 0) {
            char *lo = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            if (*p) { *p = '\0'; p++; }
            char *hi = kfl_stmt_trim(p);
            if (lo[0] == '\0' || hi[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "plan `%s`: `%s` takes a lower and an upper bound",
                    n->name, kw);
                *had_error = 1;
                continue;
            }
            char key[32];
            KflcValue lv, hv;
            memset(&lv, 0, sizeof lv);
            memset(&hv, 0, sizeof hv);
            lv.kind = KFLV_IDENT; lv.u.s = kflc_arena_strdup(arena, lo);
            hv.kind = KFLV_IDENT; hv.u.s = kflc_arena_strdup(arena, hi);
            snprintf(key, sizeof key, "%s_lo", kw);
            kfl_stmt_append_attr(arena, n, kflc_arena_strdup(arena, key), lv,
                             lineK);
            snprintf(key, sizeof key, "%s_hi", kw);
            kfl_stmt_append_attr(arena, n, kflc_arena_strdup(arena, key), hv,
                             lineK);
            continue;
        }
        kflc_diag_errorf(diag, lineK,
            "plan `%s`: unknown line `%s`; the block takes `file`, "
            "`frame`, `slots`, `time`, `position`, `velocity`, "
            "`tolerance`, `epoch` and `provenance`", n->name, kw);
        *had_error = 1;
    }

    /* The action channels this block declares. They are appended to
     * the chain this statement returns, which the block builder walks
     * to its end, so the world body carries them as ordinary actions
     * in the position the plan was written. */
    {
        const char *slots_s = NULL;
        long slots = 0;
        static const struct { const char *suffix, *bound; } CH_[] = {
            { "_t",   "time" },      { "_r_x", "position" },
            { "_r_y", "position" },  { "_r_z", "position" },
            { "_v_x", "velocity" },  { "_v_y", "velocity" },
            { "_v_z", "velocity" },  { "_tol", "tolerance" }
        };
        for (const KflcAttr *a = n->attrs; a; a = a->next) {
            if (a->name && strcmp(a->name, "slots") == 0 &&
                a->value.kind == KFLV_IDENT) {
                slots_s = a->value.u.s;
            }
        }
        if (slots_s) {
            char *end = NULL;
            slots = strtol(slots_s, &end, 10);
            if (!end || *end != '\0') slots = 0;
        }
        if (slots < 1 || slots > 4096) {
            kflc_diag_errorf(diag, line0,
                "plan `%s`: `slots` takes a whole number of knot slots "
                "from 1 to 4096", n->name);
            *had_error = 1;
            return n;
        }
        KflcNode *tail = n;
        for (long k = 0; k < slots; k++) {
            for (int c = 0; c < 8; c++) {
                char nm[128], lo[8], hi[8];
                const KflcAttr *alo = NULL, *ahi = NULL;

                snprintf(lo, sizeof lo, "_lo");
                snprintf(hi, sizeof hi, "_hi");
                for (const KflcAttr *a = n->attrs; a; a = a->next) {
                    if (!a->name || a->value.kind != KFLV_IDENT) continue;
                    size_t bl = strlen(CH_[c].bound);
                    if (strncmp(a->name, CH_[c].bound, bl) != 0) continue;
                    if (strcmp(a->name + bl, "_lo") == 0) alo = a;
                    if (strcmp(a->name + bl, "_hi") == 0) ahi = a;
                }
                if (!alo || !ahi) {
                    kflc_diag_errorf(diag, line0,
                        "plan `%s`: `%s` bounds are required, since a knot "
                        "slot is an action channel and an action channel "
                        "declares its own range", n->name, CH_[c].bound);
                    *had_error = 1;
                    return n;
                }
                snprintf(nm, sizeof nm, "%s_k%ld%s", n->name, k,
                         CH_[c].suffix);
                KflcNode *act = kfl_stmt_new_node(arena, KFLN_STMT_ACTION, line0);
                KflcValue mark;

                act->name = kflc_arena_strdup(arena, nm);
                /* The mark says the channel is the block's rather than
                 * a line an author wrote, so a round trip prints the
                 * block and not the channels the block would declare
                 * again on the way back in. */
                memset(&mark, 0, sizeof mark);
                mark.kind = KFLV_IDENT;
                mark.u.s  = kflc_arena_strdup(arena, n->name);
                kfl_stmt_append_attr(arena, act, "plan", mark, line0);
                act->position.kind = KFLV_IDENT;
                act->position.u.s  = kflc_arena_strdup(arena, "box");
                act->expr  = kflc_parse_expr(alo->value.u.s, arena, diag,
                                             line0);
                act->expr2 = kflc_parse_expr(ahi->value.u.s, arena, diag,
                                             line0);
                if (!act->expr || !act->expr2) {
                    *had_error = 1;
                    return n;
                }
                tail->next = act;
                tail = act;
            }
        }
    }
    return n;
}

KflcNode *kfl_stmt_parse_sensor(Lexer *L, Token *cur,
                               KflcArena *arena, KflcDiag *diag,
                               int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0, "sensor: expected a name");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_SENSOR, line0);
    n->name = cur->str;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "sensor `%s`: expected end of line after the name", n->name);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    } else if (kfl_stmt_at_nl(cur)) {
        kfl_stmt_advance(L, cur, had_error);
    }

    for (;;) {
        kfl_stmt_skip_newlines(L, cur, had_error);
        if (kfl_stmt_at_eof2(cur)) {
            kflc_diag_errorf(diag, line0,
                "sensor `%s`: unexpected EOF (missing `end`)", n->name);
            *had_error = 1;
            return n;
        }
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            break;
        }
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, cur->line,
                "sensor `%s`: expected a model term or `end`", n->name);
            *had_error = 1;
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
            continue;
        }

        /* The remainder is taken while the cursor still sits on the
         * keyword, which is the convention the episode block uses:
         * advancing first would consume the first operand into the
         * cursor and leave it out of the remainder. */
        char *kw = cur->str;
        int lineK = cur->line;
        char *rest = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

        KflcNode *t = kfl_stmt_new_node(arena, KFLN_STMT_SENSOR_TERM, lineK);
        t->name = kw;

        char *p = kfl_stmt_trim(rest);
        int    n_num = 0, bad = 0;
        double num[4];
        char  *dist = NULL;

        if (strcmp(kw, "noise") == 0) {
            char *w = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            if (*p) { *p = '\0'; p++; }
            if (*w == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "sensor `%s`: `noise` takes a distribution name",
                    n->name);
                *had_error = 1;
                bad = 1;
            } else {
                dist = kflc_arena_strdup(arena, w);
            }
        }
        while (!bad && *p) {
            while (*p == ' ' || *p == '\t') p++;
            if (!*p) break;
            if (n_num == 4) {
                kflc_diag_errorf(diag, lineK,
                    "sensor `%s`: `%s` takes at most four values",
                    n->name, kw);
                *had_error = 1;
                bad = 1;
                break;
            }
            char *w = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            char saved = *p;
            *p = '\0';
            char *endp = NULL;
            double v = strtod(w, &endp);
            if (!endp || *endp != '\0' || endp == w) {
                kflc_diag_errorf(diag, lineK,
                    "sensor `%s`: `%s` is not a number", n->name, w);
                *had_error = 1;
                bad = 1;
            } else {
                num[n_num++] = v;
            }
            if (saved) { *p = saved; p++; } else break;
        }
        if (bad) { kfl_stmt_append_child(n, t); continue; }

        if (dist) {
            KflcValue dv;
            memset(&dv, 0, sizeof dv);
            dv.kind = KFLV_IDENT; dv.u.s = dist;
            kfl_stmt_append_attr(arena, t, "dist", dv, lineK);
        }
        for (int i = 0; i < n_num; i++) {
            KflcValue nv;
            memset(&nv, 0, sizeof nv);
            nv.kind = KFLV_FLOAT; nv.u.f = num[i];
            char akey[8];
            snprintf(akey, sizeof akey, "n%d", i);
            kfl_stmt_append_attr(arena, t, akey, nv, lineK);
        }
        kfl_stmt_append_child(n, t);
    }
    return n;
}

/* `episode ... end`. Body lines, each at most once except `reset`:
 *   control_dt <expr>           (required)
 *   horizon <expr>
 *   terminated when <expr>
 *   reset <body>.<key> <distribution>
 * `cur` is the `episode` keyword on entry. */

KflcNode *kfl_stmt_parse_episode(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "episode: expected end of line after `episode`");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    } else if (kfl_stmt_at_nl(cur)) {
        kfl_stmt_advance(L, cur, had_error);
    }

    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_EPISODE, line0);
    for (;;) {
        kfl_stmt_skip_newlines(L, cur, had_error);
        if (kfl_stmt_at_eof2(cur)) {
            kflc_diag_errorf(diag, cur->line,
                "episode: unexpected EOF (missing `end`)");
            *had_error = 1;
            break;
        }
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            break;
        }
        if (cur->kind != T_IDENT || !cur->str) {
            kflc_diag_errorf(diag, cur->line,
                "episode: expected a keyword line (control_dt, horizon, "
                "substeps, contact, `terminated when`, reset, or end)");
            *had_error = 1;
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
            continue;
        }

        if (kfl_stmt_is_ident_named(cur, "control_dt") ||
            kfl_stmt_is_ident_named(cur, "horizon") ||
            kfl_stmt_is_ident_named(cur, "substeps"))
        {
            const char *key = cur->str;
            int lineK = cur->line;
            char *src = kfl_stmt_take_line_remainder(L, arena);
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_find_attr(n, key)) {
                kflc_diag_errorf(diag, lineK,
                    "episode: duplicate `%s` (allowed at most once)", key);
                *had_error = 1;
                continue;
            }
            char *t = kfl_stmt_trim(src);
            if (t[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "episode: `%s` requires an expression", key);
                *had_error = 1;
                continue;
            }
            KflcValue none;
            memset(&none, 0, sizeof none);
            KflcAttr *a = kfl_stmt_append_attr(arena, n, key, none, lineK);
            a->expr = kflc_parse_expr(t, arena, diag, lineK);
            if (!a->expr) *had_error = 1;
            continue;
        }

        /* `contact arrest`
         * `contact bounce restitution <expr> friction <expr>`
         *
         * One optional line, because the resolution is a property of
         * how the episode ends rather than of any one body. Absent
         * means arrest, which is the default the design already
         * fixed, so no program written before this line existed
         * changes meaning.
         *
         * `bounce` requires both coefficients and neither is
         * defaulted: a restitution nobody declared is a number this
         * compiler would have invented, and inventing physical
         * constants is the habit these rules exist to prevent. */
        if (kfl_stmt_is_ident_named(cur, "contact")) {
            int lineK = cur->line;
            char *src = kfl_stmt_take_line_remainder(L, arena);
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_find_attr(n, "contact")) {
                kflc_diag_errorf(diag, lineK,
                    "episode: duplicate `contact` (allowed at most once)");
                *had_error = 1;
                continue;
            }
            char *t = kfl_stmt_trim(src);
            char *rest = t;
            while (*rest && *rest != ' ' && *rest != '\t') rest++;
            int is_arrest = (rest - t) == 6 && strncmp(t, "arrest", 6) == 0;
            int is_bounce = (rest - t) == 6 && strncmp(t, "bounce", 6) == 0;
            if (!is_arrest && !is_bounce) {
                kflc_diag_errorf(diag, lineK,
                    "episode: `contact` takes `arrest` or `bounce`, not "
                    "`%s`", t[0] ? t : "nothing");
                *had_error = 1;
                continue;
            }
            KflcValue kind;
            memset(&kind, 0, sizeof kind);
            kind.kind = KFLV_IDENT;
            kind.u.s  = kflc_arena_strdup(arena, is_arrest ? "arrest"
                                                           : "bounce");
            kfl_stmt_append_attr(arena, n, "contact", kind, lineK);
            if (is_arrest) {
                if (*kfl_stmt_trim(rest) != '\0') {
                    kflc_diag_errorf(diag, lineK,
                        "episode: `contact arrest` takes no further "
                        "words, and `%s` follows it", kfl_stmt_trim(rest));
                    *had_error = 1;
                }
                continue;
            }
            /* `bounce` takes the two coefficients in a fixed order,
             * each introduced by its own keyword so that a program
             * cannot silently swap them. */
            char *p2 = kfl_stmt_trim(rest);
            if (strncmp(p2, "restitution", 11) != 0) {
                kflc_diag_errorf(diag, lineK,
                    "episode: `contact bounce` requires `restitution "
                    "<expr> friction <expr>`; neither coefficient is "
                    "defaulted, because a coefficient nobody declared "
                    "is one this compiler invented");
                *had_error = 1;
                continue;
            }
            p2 = kfl_stmt_trim(p2 + 11);
            char *fr = strstr(p2, "friction");
            if (!fr || fr == p2) {
                kflc_diag_errorf(diag, lineK,
                    "episode: `contact bounce` requires `friction "
                    "<expr>` after the restitution");
                *had_error = 1;
                continue;
            }
            char *rest_src = kflc_arena_strdup(arena, p2);
            rest_src[fr - p2] = '\0';
            char *rr = kfl_stmt_trim(rest_src);
            char *ff = kfl_stmt_trim(fr + 8);
            if (rr[0] == '\0' || ff[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "episode: `contact bounce` requires an expression "
                    "for each of `restitution` and `friction`");
                *had_error = 1;
                continue;
            }
            KflcValue none;
            memset(&none, 0, sizeof none);
            KflcAttr *ra = kfl_stmt_append_attr(arena, n, "restitution", none,
                                            lineK);
            ra->expr = kflc_parse_expr(rr, arena, diag, lineK);
            if (!ra->expr) *had_error = 1;
            KflcAttr *fa = kfl_stmt_append_attr(arena, n, "friction", none,
                                            lineK);
            fa->expr = kflc_parse_expr(ff, arena, diag, lineK);
            if (!fa->expr) *had_error = 1;
            continue;
        }

        if (kfl_stmt_is_ident_named(cur, "terminated")) {
            int lineK = cur->line;
            kfl_stmt_advance(L, cur, had_error);
            if (!kfl_stmt_is_ident_named(cur, "when")) {
                kflc_diag_errorf(diag, lineK,
                    "episode: expected `when` after `terminated`");
                *had_error = 1;
                if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
                    kfl_stmt_rl_drain_line(L, cur, arena, had_error);
                } else if (kfl_stmt_at_nl(cur)) {
                    kfl_stmt_advance(L, cur, had_error);
                }
                continue;
            }
            char *src = kfl_stmt_take_line_remainder(L, arena);
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_find_attr(n, "terminated_when")) {
                kflc_diag_errorf(diag, lineK,
                    "episode: duplicate `terminated when` "
                    "(allowed at most once)");
                *had_error = 1;
                continue;
            }
            char *t = kfl_stmt_trim(src);
            if (t[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "episode: `terminated when` requires an expression");
                *had_error = 1;
                continue;
            }
            KflcValue none;
            memset(&none, 0, sizeof none);
            KflcAttr *a = kfl_stmt_append_attr(arena, n, "terminated_when",
                                           none, lineK);
            a->expr = kflc_parse_expr(t, arena, diag, lineK);
            if (!a->expr) *had_error = 1;
            continue;
        }

        if (kfl_stmt_is_ident_named(cur, "reset")) {
            int lineK = cur->line;
            char *raw = kfl_stmt_take_line_remainder(L, arena);
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            char *p = kfl_stmt_trim(raw);
            char *q = p;
            while (*q && *q != '.' && *q != ' ' && *q != '\t') q++;
            if (*q != '.' || q == p) {
                kflc_diag_errorf(diag, lineK,
                    "episode reset: expected `<body>.<key> <distribution>`");
                *had_error = 1;
                continue;
            }
            *q = '\0';
            char *body_name = kflc_arena_strdup(arena, p);
            char *k = q + 1;
            q = k;
            while (*q && *q != ' ' && *q != '\t') q++;
            int have_more = (*q != '\0');
            *q = '\0';
            char *key = kflc_arena_strdup(arena, k);
            char *expr_src = have_more ? q + 1 : q;
            if (!is_reset_state_key_(key)) {
                kflc_diag_errorf(diag, lineK,
                    "episode reset: unknown state key `%s` (expected pos_x, "
                    "pos_y, pos_z, vel_x, vel_y, vel_z, quat_w, quat_x, "
                    "quat_y, quat_z, omega_x, omega_y, or omega_z)", key);
                *had_error = 1;
                continue;
            }
            expr_src = kfl_stmt_trim(expr_src);
            if (expr_src[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "episode reset: expected a distribution expression "
                    "after `%s.%s`", body_name, key);
                *had_error = 1;
                continue;
            }
            KflcExpr *dist = kflc_parse_expr(expr_src, arena, diag, lineK);
            if (!dist) {
                *had_error = 1;
                continue;
            }
            if (!is_dist_call_(dist)) {
                kflc_diag_errorf(diag, lineK,
                    "episode reset: expected a distribution expression "
                    "`uniform(<low>, <high>)` or `normal(<mean>, <stddev>)`");
                *had_error = 1;
                continue;
            }
            KflcNode *r = kfl_stmt_new_node(arena, KFLN_STMT_EPISODE_RESET, lineK);
            r->name          = body_name;
            r->position.kind = KFLV_IDENT;
            r->position.u.s  = key;
            r->expr          = dist;
            kfl_stmt_append_child(n, r);
            continue;
        }

        kflc_diag_errorf(diag, cur->line,
            "episode: unknown keyword `%s` (expected control_dt, horizon, "
            "substeps, contact, `terminated when`, reset, or end)",
            cur->str);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    }

    if (!kfl_stmt_find_attr(n, "control_dt")) {
        kflc_diag_errorf(diag, line0,
            "episode: missing required `control_dt <expr>`");
        *had_error = 1;
    }
    return n;
}

/* `action <name> box <low> <high> [default <expr>]`
 * `action <name> discrete <count> [default <expr>]`
 * The bounds / count / default are whitespace-separated expressions
 * (balanced `()` / `[]` keep a spaced expression together, matching
 * the astro_body value convention). `cur` is the `action` keyword. */

KflcNode *kfl_stmt_parse_action(Lexer *L, Token *cur,
                               KflcArena *arena, KflcDiag *diag,
                               int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0, "action: expected action name");
        *had_error = 1;
        if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        } else if (kfl_stmt_at_nl(cur)) {
            kfl_stmt_advance(L, cur, had_error);
        }
        return NULL;
    }
    char *name = cur->str;
    kfl_stmt_advance(L, cur, had_error);
    int is_box = kfl_stmt_is_ident_named(cur, "box");
    if (!is_box && !kfl_stmt_is_ident_named(cur, "discrete")) {
        kflc_diag_errorf(diag, line0,
            "action %s: expected `box <low> <high>` or `discrete <count>`",
            name);
        *had_error = 1;
        if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        } else if (kfl_stmt_at_nl(cur)) {
            kfl_stmt_advance(L, cur, had_error);
        }
        return NULL;
    }
    char *raw = kfl_stmt_take_line_remainder(L, arena);
    kfl_stmt_advance(L, cur, had_error);
    if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

    /* Split into whitespace-separated, paren/bracket-balanced chunks:
     * the expressions plus the optional `default` marker word. */
    enum { ACTION_MAX_CHUNKS = 5 };
    char *chunks[ACTION_MAX_CHUNKS];
    int   n_chunks = 0;
    int   overflow = 0;
    char *p = kfl_stmt_trim(raw);
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        char *beg = p;
        int paren = 0, brack = 0;
        while (*p) {
            if      (*p == '(') paren++;
            else if (*p == ')') paren--;
            else if (*p == '[') brack++;
            else if (*p == ']') brack--;
            else if ((*p == ' ' || *p == '\t') && paren == 0 && brack == 0)
                break;
            p++;
        }
        int more = (*p != '\0');
        *p = '\0';
        if (n_chunks < ACTION_MAX_CHUNKS) chunks[n_chunks++] = beg;
        else overflow = 1;
        if (more) p++;
    }

    int expect = is_box ? 2 : 1;
    if (n_chunks < expect) {
        kflc_diag_errorf(diag, line0,
            is_box ? "action %s: box requires `<low> <high>`"
                   : "action %s: discrete requires `<count>`",
            name);
        *had_error = 1;
        return NULL;
    }
    int have_default = 0;
    if (overflow || n_chunks > expect) {
        if (overflow || n_chunks != expect + 2 ||
            strcmp(chunks[expect], "default") != 0)
        {
            kflc_diag_errorf(diag, line0,
                "action %s: expected optional `default <expr>` after the %s",
                name, is_box ? "bounds" : "count");
            *had_error = 1;
            return NULL;
        }
        have_default = 1;
    }

    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_ACTION, line0);
    n->name          = name;
    n->position.kind = KFLV_IDENT;
    n->position.u.s  = kflc_arena_strdup(arena, is_box ? "box" : "discrete");
    n->expr = kflc_parse_expr(chunks[0], arena, diag, line0);
    if (!n->expr) *had_error = 1;
    if (is_box) {
        n->expr2 = kflc_parse_expr(chunks[1], arena, diag, line0);
        if (!n->expr2) *had_error = 1;
    }
    if (have_default) {
        KflcValue none;
        memset(&none, 0, sizeof none);
        KflcAttr *a = kfl_stmt_append_attr(arena, n, "default", none, line0);
        a->expr = kflc_parse_expr(chunks[expect + 1], arena, diag, line0);
        if (!a->expr) *had_error = 1;
    }
    return n;
}

/* `on_step ... end`. The body is a plain statement block; the
 * forbidden-statement check at the top of kfl_stmt_parse_stmt fires while
 * `g_rl_on_step_depth` is raised. `cur` is the `on_step` keyword. */

KflcNode *kfl_stmt_parse_on_step(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "on_step: expected end of line after `on_step`");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    } else if (kfl_stmt_at_nl(cur)) {
        kfl_stmt_advance(L, cur, had_error);
    }

    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_ON_STEP, line0);
    const char *brk[] = { "end", NULL };
    g_rl_on_step_depth++;
    KflcNode *blk = kfl_parse_stmt_block(L, cur, arena, diag, had_error,
                                         "end", brk);
    g_rl_on_step_depth--;
    n->children = blk ? blk->children : NULL;
    if (kfl_stmt_is_ident_named(cur, "end")) {
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
    }
    return n;
}

/* `objective ... end` with `reward <expr>` (required) and
 * `terminal <expr>` (optional), each at most once. `cur` is the
 * `objective` keyword. */

KflcNode *kfl_stmt_parse_objective(Lexer *L, Token *cur,
                                  KflcArena *arena, KflcDiag *diag,
                                  int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "objective: expected end of line after `objective`");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    } else if (kfl_stmt_at_nl(cur)) {
        kfl_stmt_advance(L, cur, had_error);
    }

    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_OBJECTIVE, line0);
    for (;;) {
        kfl_stmt_skip_newlines(L, cur, had_error);
        if (kfl_stmt_at_eof2(cur)) {
            kflc_diag_errorf(diag, cur->line,
                "objective: unexpected EOF (missing `end`)");
            *had_error = 1;
            break;
        }
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            break;
        }
        if (cur->kind == T_IDENT && cur->str &&
            (strcmp(cur->str, "reward") == 0 ||
             strcmp(cur->str, "terminal") == 0))
        {
            const char *key = cur->str;
            int lineK = cur->line;
            char *src = kfl_stmt_take_line_remainder(L, arena);
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_find_attr(n, key)) {
                kflc_diag_errorf(diag, lineK,
                    "objective: duplicate `%s` (allowed at most once)", key);
                *had_error = 1;
                continue;
            }
            char *t = kfl_stmt_trim(src);
            if (t[0] == '\0') {
                kflc_diag_errorf(diag, lineK,
                    "objective: `%s` requires an expression", key);
                *had_error = 1;
                continue;
            }
            KflcValue none;
            memset(&none, 0, sizeof none);
            KflcAttr *a = kfl_stmt_append_attr(arena, n, key, none, lineK);
            a->expr = kflc_parse_expr(t, arena, diag, lineK);
            if (!a->expr) *had_error = 1;
            continue;
        }
        kflc_diag_errorf(diag, cur->line,
            "objective: unknown keyword `%s` (expected reward, terminal, "
            "or end)",
            (cur->kind == T_IDENT && cur->str) ? cur->str : "(non-ident)");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    }

    if (!kfl_stmt_find_attr(n, "reward")) {
        kflc_diag_errorf(diag, line0,
            "objective: missing required `reward <expr>`");
        *had_error = 1;
    }
    return n;
}

/* `agent <name> ... end`. The block is a scope over constructs that
 * already exist, so its body is parsed by the ordinary block parser
 * and the `action`, `observe ... as` and `objective` statements inside
 * it come out as the same nodes they are at world level. Which
 * statements an agent block may hold is decided where the rest of the
 * environment model is built, so the admissible set lives in one place
 * rather than in the parser and again in the emitter. `cur` is the
 * `agent` keyword. */

KflcNode *kfl_stmt_parse_agent(Lexer *L, Token *cur,
                              KflcArena *arena, KflcDiag *diag,
                              int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0, "agent: expected a name");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_AGENT, line0);
    n->name = cur->str;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "agent `%s`: expected end of line after the name", n->name);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
    } else if (kfl_stmt_at_nl(cur)) {
        kfl_stmt_advance(L, cur, had_error);
    }

    const char *brk[] = { "end", NULL };
    KflcNode *blk = kfl_parse_stmt_block(L, cur, arena, diag, had_error,
                                         "end", brk);
    n->children = blk ? blk->children : NULL;
    if (kfl_stmt_is_ident_named(cur, "end")) {
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
    } else {
        kflc_diag_errorf(diag, line0,
            "agent `%s`: unexpected EOF (missing `end`)", n->name);
        *had_error = 1;
    }
    return n;
}

/* `astro_payload <name> body=<body> kind=<kind> [key=value ...]`.
 *
 * The shape is astro_body's, deliberately: a name and whitespace
 * separated `key=value` pairs whose values are kept as verbatim
 * expression text and validated at emit time, where the kind is known
 * and where the distribution forms are already recognised. One
 * statement covers every defense payload kind because the libraries
 * share one payload slot and one kind-tag registry, so the parse has
 * one case and the key schema is a function of `kind=` rather than of
 * the statement word. */

KflcNode *kfl_stmt_parse_astro_payload(Lexer *L, Token *cur,
                                      KflcArena *arena, KflcDiag *diag,
                                      int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0,
            "astro_payload: expected payload name");
        *had_error = 1;
        while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        return NULL;
    }
    char *pay_name = cur->str;
    char *raw = kfl_stmt_take_line_remainder(L, arena);
    kfl_stmt_advance(L, cur, had_error);
    if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_ASTRO_PAYLOAD, line0);
    n->name = pay_name;

    char *p = kfl_stmt_trim(raw);
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        char *kbeg = p;
        while (*p && *p != '=' && *p != ' ' && *p != '\t') p++;
        if (*p != '=') {
            kflc_diag_errorf(diag, line0,
                "astro_payload %s: expected `key=value` (got `%s`)",
                pay_name, kbeg);
            *had_error = 1;
            return n;
        }
        char *kend = p;
        *kend = '\0';
        char *key = kflc_arena_strdup(arena, kbeg);
        p++;  /* past '=' */
        char *vbeg = p;
        int paren = 0, brack = 0;
        while (*p) {
            if      (*p == '(') paren++;
            else if (*p == ')') paren--;
            else if (*p == '[') brack++;
            else if (*p == ']') brack--;
            else if ((*p == ' ' || *p == '\t') && paren == 0 && brack == 0) break;
            p++;
        }
        char saved_v = *p; *p = '\0';
        char *val = kflc_arena_strdup(arena, vbeg);
        if (saved_v) { *p = saved_v; }
        KflcValue v;
        memset(&v, 0, sizeof v);
        v.kind = KFLV_IDENT;
        v.u.s  = val;
        kfl_stmt_append_attr(arena, n, key, v, line0);
    }
    return n;
}

/* `engage <payload> at <target>`.
 *
 * Three identifiers and one connective, with nothing else admitted on
 * the line: an engagement names what fires and what it fires at, and
 * every other quantity it needs is already declared on the payload or
 * derived from the two bodies' state. `at` is read as a connective
 * here alone, so a body or a binding called `at` keeps its name
 * everywhere else. */

KflcNode *kfl_stmt_parse_engage(Lexer *L, Token *cur,
                               KflcArena *arena, KflcDiag *diag,
                               int *had_error)
{
    int line0 = cur->line;
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0,
            "engage: expected the name of a payload to engage");
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    char *pay = cur->str;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_is_ident_named(cur, "at")) {
        kflc_diag_errorf(diag, line0,
            "engage %s: expected `at` and the name of the body engaged",
            pay);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    kfl_stmt_advance(L, cur, had_error);
    if (cur->kind != T_IDENT) {
        kflc_diag_errorf(diag, line0,
            "engage %s at: expected the name of a body", pay);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    char *tgt = cur->str;
    kfl_stmt_advance(L, cur, had_error);
    if (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) {
        kflc_diag_errorf(diag, line0,
            "engage %s at %s: the statement takes a payload and a body "
            "and nothing else", pay, tgt);
        *had_error = 1;
        kfl_stmt_rl_drain_line(L, cur, arena, had_error);
        return NULL;
    }
    if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_ENGAGE, line0);
    n->name = pay;
    KflcValue v;
    memset(&v, 0, sizeof v);
    v.kind = KFLV_IDENT;
    v.u.s  = tgt;
    kfl_stmt_append_attr(arena, n, "at", v, line0);
    return n;
}
