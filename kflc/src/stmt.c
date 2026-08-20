/* stmt.c - the statement parser: line plumbing, node
 * construction, the type helpers, and the dispatcher. */
#include "stmt_internal.h"

static long find_assign(const char *s);
static long find_colon(const char *s);
int kflc_type_from_str(const char *name, const char **out_opaque_name);
const char *kflc_type_kfl_str(KflcType t, const char *opaque_name);
void kfl_stmt_set_world_ctx(int in_world);

/* Helpers exported to parser.c. Declared in internal.h. */

/* Read raw characters from the lexer until end-of-line (or EOF),
 * returning an arena-owned NUL-terminated string. Leading whitespace
 * is skipped; trailing whitespace is preserved (the expression parser
 * tolerates it). Used to capture the right-hand side of let/const/
 * return/assign + the condition of if/while.
 *
 * Does NOT consume the closing newline. */

char *kfl_stmt_take_line_remainder(Lexer *L, KflcArena *arena)
{
    /* Skip horizontal whitespace. */
    while (L->pos < L->len) {
        int c = (unsigned char)L->src[L->pos];
        if (c == ' ' || c == '\t' || c == '\r') { L->pos++; continue; }
        break;
    }
    size_t start = L->pos;
    while (L->pos < L->len && L->src[L->pos] != '\n') L->pos++;
    size_t n = L->pos - start;
    /* Strip trailing horizontal whitespace. */
    while (n > 0) {
        char c = L->src[start + n - 1];
        if (c == ' ' || c == '\t' || c == '\r') { n--; continue; }
        break;
    }
    char *out = (char *)kflc_arena_alloc(arena, n + 1);
    if (n > 0) memcpy(out, L->src + start, n);
    out[n] = '\0';
    return out;
}

/* Find the first un-bracketed `=` (not `==`) in a string. Returns
 * its index, or -1 if not found. Counts both `(...)` and `[...]` so
 * an `=` inside an index expression (`xs[i] = ...` — outer) is
 * preferred over any `==` that might appear in the index. */

static long find_assign(const char *s)
{
    int paren = 0;
    int brack = 0;
    for (long i = 0; s[i]; i++) {
        char c = s[i];
        if      (c == '(') paren++;
        else if (c == ')') paren--;
        else if (c == '[') brack++;
        else if (c == ']') brack--;
        else if (c == '=' && paren == 0 && brack == 0) {
            if (s[i + 1] == '=') { i++; continue; }    /* skip == */
            return i;
        }
    }
    return -1;
}

/* Find the first unparenthesised `:` in a string. */

static long find_colon(const char *s)
{
    int depth = 0;
    for (long i = 0; s[i]; i++) {
        if (s[i] == '(') depth++;
        else if (s[i] == ')') depth--;
        else if (s[i] == ':' && depth == 0) return i;
    }
    return -1;
}

/* Trim a NUL-terminated string in place (returns pointer to first
 * non-space char; nul-terminates after last non-space). */

char *kfl_stmt_trim(char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t')) { s[n - 1] = '\0'; n--; }
    return s;
}

/* ---- Public type helpers ----------------------------------------- */

const char *kflc_type_cxx(KflcType t, const char *opaque_name)
{
    switch (t) {
    case KFLT_VOID:   return "void";
    case KFLT_DOUBLE: return "double";
    case KFLT_INT:    return "int";
    case KFLT_BOOL:   return "bool";
    case KFLT_STRING: return "const char *";
    case KFLT_VECTOR: return "K26CVector";
    case KFLT_MATRIX: return "K26CMatrix";
    case KFLT_OPAQUE:
        if (!opaque_name) return NULL;
        return kflc_opaque_cxx(opaque_name);
    }
    return NULL;
}

int kflc_type_from_str(const char *name, const char **out_opaque_name)
{
    if (out_opaque_name) *out_opaque_name = NULL;
    if (!name) return -1;
    if (strcmp(name, "void")   == 0) return KFLT_VOID;
    if (strcmp(name, "double") == 0) return KFLT_DOUBLE;
    if (strcmp(name, "int")    == 0) return KFLT_INT;
    if (strcmp(name, "bool")   == 0) return KFLT_BOOL;
    if (strcmp(name, "string") == 0) return KFLT_STRING;
    if (strcmp(name, "vector") == 0) return KFLT_VECTOR;
    if (strcmp(name, "matrix") == 0) return KFLT_MATRIX;
    /* Check the opaque registry only when the caller can
     * receive the matched name. Without `out_opaque_name` the type
     * stays ambiguous and we report "not a type" rather than risk
     * the caller losing the subtype identity. */
    if (out_opaque_name) {
        const char *cxx = kflc_opaque_cxx(name);
        if (cxx) {
            /* Resolve to the registry's stable pointer for `name` —
             * kflc_opaque_cxx already validates registration; the
             * caller will write the surface name back to its node so
             * we hand it the registry-owned literal where possible. */
            *out_opaque_name = name;
            return KFLT_OPAQUE;
        }
    }
    return -1;
}

const char *kflc_type_kfl_str(KflcType t, const char *opaque_name)
{
    switch (t) {
    case KFLT_VOID:   return "void";
    case KFLT_DOUBLE: return "double";
    case KFLT_INT:    return "int";
    case KFLT_BOOL:   return "bool";
    case KFLT_STRING: return "string";
    case KFLT_VECTOR: return "vector";
    case KFLT_MATRIX: return "matrix";
    case KFLT_OPAQUE: return opaque_name;   /* may be NULL on malformed input */
    }
    return NULL;
}

int kfl_stmt_is_ident_named(const Token *t, const char *name)
{
    return t->kind == T_IDENT && t->str && strcmp(t->str, name) == 0;
}

int kfl_stmt_at_nl  (const Token *t) { return t->kind == T_NEWLINE; }
int kfl_stmt_at_eof2(const Token *t) { return t->kind == T_EOF; }

void kfl_stmt_advance(Lexer *L, Token *cur, int *had_error)
{
    if (!lex_next(L, cur)) *had_error = 1;
}

void kfl_stmt_skip_newlines(Lexer *L, Token *cur, int *had_error)
{
    while (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
}

KflcNode *kfl_stmt_new_node(KflcArena *arena, KflcNodeKind k, int line)
{
    KflcNode *n = (KflcNode *)kflc_arena_alloc(arena, sizeof(*n));
    /* Zero-init defensively so all KflcNode fields have a predictable
     * default; the arena is malloc-backed and does not pre-zero
     * allocations. */
    memset(n, 0, sizeof(*n));
    n->kind = k;
    n->line = line;
    /* lifetime_qualifier defaults to KFL_LQ_NONE (= 0); explicit
     * assignment for clarity. */
    n->lifetime_qualifier = KFL_LQ_NONE;
    return n;
}

/* Append a single attribute to a statement node's linked attr list.
 * Returns the appended attr so callers can stash extras (e.g. a
 * pre-parsed KflcExpr * on the attr's `expr` field for `label`). */

KflcAttr *kfl_stmt_append_attr(KflcArena *arena, KflcNode *n,
                                   const char *key, KflcValue val, int line)
{
    KflcAttr *a = (KflcAttr *)kflc_arena_alloc(arena, sizeof *a);
    a->name = kflc_arena_strdup(arena, key);
    a->value = val;
    a->line = line;
    a->expr = NULL;
    a->next = NULL;
    if (!n->attrs) { n->attrs = a; return a; }
    KflcAttr *t = n->attrs;
    while (t->next) t = t->next;
    t->next = a;
    return a;
}

void kfl_stmt_append_child(KflcNode *parent, KflcNode *child)
{
    if (!parent->children) { parent->children = child; return; }
    KflcNode *p = parent->children;
    while (p->next) p = p->next;
    p->next = child;
}

void kfl_stmt_set_world_ctx(int in_world)
{
    g_rl_world_ctx = in_world;
    if (!in_world) g_rl_on_step_depth = 0;
}

/* The kind of the token after the current one, without consuming it.
 * The lexer's whole state is restored, and its diagnostic sink is
 * detached for the duration, so a speculative look at a character the
 * lexer would refuse reports nothing and leaves nothing behind. The
 * arena bytes a speculative identifier takes are not reclaimed, which
 * costs one word per statement that begins with one of the words
 * below and buys back the programs the alternative gives up.
 *
 * Returns the lexer's own success, which is what tells the end of the
 * input from a character the lexer will not read: both leave T_EOF in
 * the token, and the expression sub-parser reads several characters
 * this lexer refuses, so treating a refusal as an end of line would
 * make `on_step = 2.0` open a block. */

int kfl_stmt_peek_kind(Lexer *L, TokenKind *out)
{
    Lexer save = *L;
    Token next;
    memset(&next, 0, sizeof next);
    L->diag = NULL;
    int ok = lex_next(L, &next);
    *out = next.kind;
    *L = save;
    return ok;
}

const KflcAttr *kfl_stmt_find_attr(const KflcNode *n, const char *key)
{
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0) return a;
    }
    return NULL;
}

/* Parse a single statement on the current line. Consumes the trailing
 * newline. Returns NULL on parse error.
 *
 * The main lexer doesn't tokenise expression operators (- + * / etc),
 * so every statement that contains an expression captures the rest of
 * the source line as raw bytes after the keyword token and hands it
 * to the expression sub-parser (which has its own lexer). After the
 * capture L->pos sits at the newline; we then advance the main lexer
 * once to refresh `cur` to T_NEWLINE, and again to skip it. */

KflcNode *kfl_stmt_parse_stmt(Lexer *L, Token *cur,
                             KflcArena *arena, KflcDiag *diag,
                             int *had_error)
{
    int line = cur->line;

    /* Grammar 3.2 RL statements. The keywords bind as constructs only
     * at statement position inside `fn world` bodies; everywhere else
     * they keep the ordinary-identifier reading (and reserved-future
     * warning behaviour) of Grammar 3.1. Inside an `on_step` body the
     * world-construction / stepping statements and nested RL
     * constructs are rejected here so the diagnostic names the
     * offending keyword. */
    if (g_rl_world_ctx && cur->kind == T_IDENT && cur->str) {
        int construct = kfl_stmt_rl_word_is_construct(L, cur->str);
        /* `engage` runs the other way from every word beside it: it is
         * an act of a step, so the per-step body is the one block that
         * admits it and the world prefix is where it is refused. */
        int engaging = construct && strcmp(cur->str, "engage") == 0;
        if (g_rl_on_step_depth > 0 && !engaging &&
            (construct || kfl_stmt_is_on_step_world_stmt(cur->str))) {
            kflc_diag_errorf(diag, line,
                "on_step: `%s` is not allowed inside an on_step block; "
                "the per-step body admits only ordinary statements",
                cur->str);
            *had_error = 1;
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
            return NULL;
        }
        if (engaging && g_rl_on_step_depth == 0) {
            kflc_diag_errorf(diag, line,
                "engage: an engagement is an act of a step and is "
                "admissible inside an `on_step` block only");
            *had_error = 1;
            kfl_stmt_rl_drain_line(L, cur, arena, had_error);
            return NULL;
        }
        if (construct) {
            if (engaging)
                return kfl_stmt_parse_engage(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "episode") == 0)
                return kfl_stmt_parse_episode(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "action") == 0)
                return kfl_stmt_parse_action(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "on_step") == 0)
                return kfl_stmt_parse_on_step(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "objective") == 0)
                return kfl_stmt_parse_objective(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "sensor") == 0)
                return kfl_stmt_parse_sensor(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "agent") == 0)
                return kfl_stmt_parse_agent(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "astro_payload") == 0)
                return kfl_stmt_parse_astro_payload(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "plan") == 0)
                return kfl_stmt_parse_plan(L, cur, arena, diag, had_error);
            if (strcmp(cur->str, "capture_envelope") == 0)
                return kfl_stmt_parse_capture_envelope(L, cur, arena, diag,
                                                       had_error);
        }
    }

    /* `return [<expr>]` */
    if (kfl_stmt_is_ident_named(cur, "return")) {
        char *body = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);   /* cur becomes T_NEWLINE */
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_RETURN, line);
        char *trimmed = kfl_stmt_trim(body);
        if (trimmed[0] != '\0') {
            n->expr = kflc_parse_expr(trimmed, arena, diag, line);
            if (!n->expr) *had_error = 1;
        }
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        return n;
    }

    /* `print <arg>[, <arg>...]` — write each arg to stdout, then a
     * newline. Args are comma-separated; a leading `"` marks a string
     * literal (printed verbatim), otherwise the arg is a numeric
     * expression (printed as a double). Commas inside a quoted string
     * do not split. */
    if (kfl_stmt_is_ident_named(cur, "print")) {
        char *body = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);   /* cur becomes T_NEWLINE */
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_PRINT, line);
        char *b = kfl_stmt_trim(body);
        while (*b) {
            while (*b == ' ' || *b == '\t') b++;
            if (!*b) break;
            char *start = b;
            int in_str = 0, depth = 0;
            while (*b && (in_str || depth > 0 || *b != ',')) {
                if (*b == '"') in_str = !in_str;
                else if (!in_str && (*b == '(' || *b == '[')) depth++;
                else if (!in_str && (*b == ')' || *b == ']')) depth--;
                b++;
            }
            char *end = b;                 /* points at ',' or '\0' */
            if (*b == ',') b++;            /* consume separator */
            while (end > start && (end[-1] == ' ' || end[-1] == '\t')) end--;
            if (end == start) continue;    /* empty arg */
            KflcNode *arg = kfl_stmt_new_node(arena, KFLN_STMT_EXPR, line);
            if (start[0] == '"') {
                /* string literal: content between the quotes */
                char *s = start + 1;
                char *e = end;
                if (e > s && e[-1] == '"') e--;
                size_t len = (size_t)(e - s);
                char *lit = kflc_arena_alloc(arena, len + 1);
                memcpy(lit, s, len);
                lit[len] = '\0';
                arg->position.kind = KFLV_STR;
                arg->position.u.s  = lit;
            } else {
                size_t len = (size_t)(end - start);
                char *src = kflc_arena_alloc(arena, len + 1);
                memcpy(src, start, len);
                src[len] = '\0';
                arg->expr = kflc_parse_expr(src, arena, diag, line);
                if (!arg->expr) *had_error = 1;
            }
            kfl_stmt_append_child(n, arg);
        }
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        return n;
    }

    /* `let <name>: <type> = <expr>` and `const <name>: <type> = <expr>`.
     * The lexer doesn't tokenise `:` or `=`, so after consuming the
     * name we capture the rest of the line as raw bytes and split
     * manually. Cur is left on the trailing newline so the block
     * parser can skip it. */
    if (kfl_stmt_is_ident_named(cur, "let") || kfl_stmt_is_ident_named(cur, "const")) {
        int is_const = kfl_stmt_is_ident_named(cur, "const");
        kfl_stmt_advance(L, cur, had_error);
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, line,
                             "%s: expected name identifier",
                             is_const ? "const" : "let");
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *name = cur->str;
        if (kfl_is_reserved_future(name)) {
            kflc_diag_warnf(diag, line,
                "%s `%s` shadows a name reserved for a future KFL "
                "keyword; consider renaming to avoid breakage when "
                "that keyword lands",
                is_const ? "const" : "let", name);
        }

        /* Capture from immediately after `name` to EOL — should look
         * like `: <type> = <expr>` or `= <expr>`. Then refill cur to
         * read the newline that terminates the statement. */
        char *rest = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);   /* now cur is the newline */
        rest = kfl_stmt_trim(rest);

        KflcType              ty         = KFLT_DOUBLE;
        const char           *ty_opaque  = NULL;
        KflcLifetimeQualifier lq         = KFL_LQ_NONE;
        long colon_at = (rest[0] == ':') ? 0 : find_colon(rest);
        if (colon_at >= 0) {
            char *p = rest + colon_at + 1;
            while (*p == ' ' || *p == '\t') p++;
            /* Parse `: [<lifetime>] <type>` where lifetime is one of
             * `own` / `borrow` / `ptr`. If the first word names a
             * qualifier, consume it and advance to the next word for
             * the type; otherwise the first word IS the type. */
            char *first_start = p;
            while (*p && *p != ' ' && *p != '\t' && *p != '=') p++;
            char saved = *p;
            *p = '\0';
            int lq_check = kflc_lifetime_qualifier_from_str(first_start);
            char *type_start;
            if (lq_check >= 0) {
                lq = (KflcLifetimeQualifier)lq_check;
                *p = saved;
                while (*p == ' ' || *p == '\t') p++;
                type_start = p;
                while (*p && *p != ' ' && *p != '\t' && *p != '=') p++;
                saved = *p;
                *p = '\0';
            } else {
                type_start = first_start;
            }
            const char *opaque_raw = NULL;
            int t = kflc_type_from_str(type_start, &opaque_raw);
            /* Snapshot the opaque name before restoring the delimiter —
             * type_start points into the mutated `rest` buffer. */
            if (opaque_raw)
                ty_opaque = kflc_arena_strdup(arena, opaque_raw);
            if (t < 0) {
                kflc_diag_errorf(diag, line,
                                 "%s: unknown type `%s`",
                                 is_const ? "const" : "let", type_start);
                *had_error = 1;
            } else {
                ty = (KflcType)t;
            }
            *p = saved;
            rest = p;
        }
        long eq_at = find_assign(rest);
        if (eq_at < 0) {
            kflc_diag_errorf(diag, line,
                             "%s: expected `=` followed by expression",
                             is_const ? "const" : "let");
            *had_error = 1;
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *expr_src = rest + eq_at + 1;
        KflcNode *n = kfl_stmt_new_node(arena,
            is_const ? KFLN_STMT_CONST : KFLN_STMT_LET, line);
        n->name = name;
        n->type = ty;
        n->type_subtype = ty_opaque;
        n->lifetime_qualifier = lq;
        n->expr = kflc_parse_expr(expr_src, arena, diag, line);
        if (!n->expr) *had_error = 1;
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        return n;
    }

    /* `if <expr> ... [else] end` */
    if (kfl_stmt_is_ident_named(cur, "if")) {
        char *body_src = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        char *trimmed = kfl_stmt_trim(body_src);
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_IF, line);
        n->expr = kflc_parse_expr(trimmed, arena, diag, line);
        if (!n->expr) *had_error = 1;

        const char *brk[] = { "else", "end", NULL };
        KflcNode *blk = kfl_parse_stmt_block(L, cur, arena, diag, had_error,
                                              "end", brk);
        n->children = blk ? blk->children : NULL;
        if (kfl_stmt_is_ident_named(cur, "else")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            const char *brk2[] = { "end", NULL };
            KflcNode *eb = kfl_parse_stmt_block(L, cur, arena, diag, had_error,
                                                 "end", brk2);
            n->else_children = eb ? eb->children : NULL;
        }
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        }
        return n;
    }

    /* `series_<kind> "<label>" <xs_ident> <ys_ident>`. Only
     * meaningful inside a `fn data` body — but we parse it here for
     * any function-body context and the emitter raises if used in a
     * non-data fn. The kind ident maps to a K26PSeriesKind enum
     * value stored in position.u.i. */
    if (cur->kind == T_IDENT && cur->str &&
        strncmp(cur->str, "series_", 7) == 0)
    {
        int series_kind = -1;
        const char *suffix = cur->str + 7;
        if      (strcmp(suffix, "line")     == 0) series_kind = 0;  /* K26P_LINE */
        else if (strcmp(suffix, "scatter")  == 0) series_kind = 1;  /* K26P_SCATTER */
        else if (strcmp(suffix, "errorbar") == 0) series_kind = 2;
        else if (strcmp(suffix, "histogram")== 0) series_kind = 3;
        else if (strcmp(suffix, "box")      == 0) series_kind = 4;
        else if (strcmp(suffix, "heatmap")  == 0) series_kind = 5;
        if (series_kind >= 0) {
            int line0 = cur->line;
            kfl_stmt_advance(L, cur, had_error);   /* consume keyword; now cur = label string */
            if (cur->kind != T_STRING) {
                kflc_diag_errorf(diag, line0,
                    "series_%s: expected label string", suffix);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            char *label = cur->str;
            kfl_stmt_advance(L, cur, had_error);
            /* Heatmap takes a single `matrix` identifier (row-major
             * data + rows/cols), not an xs/ys vector pair. The grid is
             * placed in index space [0,cols]×[0,rows]; color auto-fits. */
            if (series_kind == 5) {
                if (cur->kind != T_IDENT) {
                    kflc_diag_errorf(diag, line0,
                        "series_heatmap: expected matrix identifier");
                    *had_error = 1;
                    while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                    if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                    return NULL;
                }
                char *mat_name = cur->str;
                kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                KflcNode *hn = kfl_stmt_new_node(arena, KFLN_STMT_SERIES, line0);
                hn->name          = label;
                hn->position.kind = KFLV_INT;
                hn->position.u.i  = series_kind;
                KflcAttr *am = (KflcAttr *)kflc_arena_alloc(arena, sizeof(*am));
                am->name       = (char *)"mat";
                am->value.kind = KFLV_IDENT;
                am->value.u.s  = mat_name;
                am->next       = NULL;
                am->line       = line0;
                hn->attrs = am;
                return hn;
            }
            if (cur->kind != T_IDENT) {
                kflc_diag_errorf(diag, line0,
                    "series_%s: expected xs vector identifier", suffix);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            char *xs_name = cur->str;
            kfl_stmt_advance(L, cur, had_error);
            if (cur->kind != T_IDENT) {
                kflc_diag_errorf(diag, line0,
                    "series_%s: expected ys vector identifier", suffix);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            char *ys_name = cur->str;
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_SERIES, line0);
            n->name             = label;
            n->position.kind    = KFLV_INT;
            n->position.u.i     = series_kind;
            KflcAttr *ax = (KflcAttr *)kflc_arena_alloc(arena, sizeof(*ax));
            ax->name = (char *)"xs";
            ax->value.kind = KFLV_IDENT;
            ax->value.u.s  = xs_name;
            ax->next = NULL;
            ax->line = line0;
            KflcAttr *ay = (KflcAttr *)kflc_arena_alloc(arena, sizeof(*ay));
            ay->name = (char *)"ys";
            ay->value.kind = KFLV_IDENT;
            ay->value.u.s  = ys_name;
            ay->next = NULL;
            ay->line = line0;
            n->attrs = ax;
            ax->next = ay;
            return n;
        }
    }

    /* ---- Astro statements (inside `fn world` bodies) ------------ *
     * Each parser consumes the keyword + its argument shape and returns
     * a KFLN_STMT_* node. Per-statement raw-line capture handles named-
     * argument syntax (e.g. `mode=astrometric`) — the lexer does not
     * tokenise `=`, so we string-find within the captured remainder.
     * Round-trip emission lives in serialize.c; C++ codegen lives in
     * the kfl_emit_stmt dispatch further down this file. */

    /* `astro_body <name> gm=<expr> pos=<expr> vel=<expr> [mass=<expr>]
     *                    [radius=<expr>] [j2=<expr>] [on_rails=<bool>]
     *                    [parent=<ident>]`
     *
     * Each `<key>=<expr>` lands as a stmt-attr whose value carries the
     * verbatim expression text (KFLV_IDENT) for emit-time parsing
     * against the active fn-world ctx. Keys are stable for serialize
     * round-trip; emit time validates the schema. The lexer doesn't
     * tokenise `=`, so we read the line remainder raw and split. */
    if (kfl_stmt_is_ident_named(cur, "astro_body")) {
        int line0 = cur->line;
        kfl_stmt_advance(L, cur, had_error);
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, line0, "astro_body: expected body name");
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *body_name = cur->str;
        char *raw = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_ASTRO_BODY, line0);
        n->name = body_name;

        /* Whitespace-separated `key=value` pairs. Values are
         * paren-balanced expressions; whitespace inside `()`/`[]` is
         * preserved. */
        char *p = kfl_stmt_trim(raw);
        while (*p) {
            while (*p == ' ' || *p == '\t') p++;
            if (!*p) break;
            char *kbeg = p;
            while (*p && *p != '=' && *p != ' ' && *p != '\t') p++;
            if (*p != '=') {
                kflc_diag_errorf(diag, line0,
                    "astro_body %s: expected `key=value` (got `%s`)",
                    body_name, kbeg);
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

    /* `step <dt_expr>` — drive one scheduler tick. */
    if (kfl_stmt_is_ident_named(cur, "step")) {
        int line0 = cur->line;
        char *body = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_STEP, line0);
        char *trimmed = kfl_stmt_trim(body);
        if (trimmed[0] == '\0') {
            kflc_diag_errorf(diag, line0, "step: expected dt expression");
            *had_error = 1;
            return n;
        }
        n->expr = kflc_parse_expr(trimmed, arena, diag, line0);
        if (!n->expr) *had_error = 1;
        return n;
    }

    /* `propagate <body_ident> for <dt_expr>`: single-body Kepler /
     * integrator step (conics on-rails or grav.step under the hood). */
    if (kfl_stmt_is_ident_named(cur, "propagate")) {
        int line0 = cur->line;
        kfl_stmt_advance(L, cur, had_error);
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, line0,
                "propagate: expected body identifier");
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *body_ident = cur->str;
        kfl_stmt_advance(L, cur, had_error);
        if (!kfl_stmt_is_ident_named(cur, "for")) {
            kflc_diag_errorf(diag, line0,
                "propagate %s: expected `for` keyword", body_ident);
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *dt_src = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_PROPAGATE, line0);
        n->name = body_ident;
        char *trimmed = kfl_stmt_trim(dt_src);
        n->expr = kflc_parse_expr(trimmed, arena, diag, line0);
        if (!n->expr) *had_error = 1;
        return n;
    }

    /* `for_each <ident> in <world_ident> ... end` — read-only
     * iteration over the world's bodies. Block-bearing statement;
     * reuses kfl_parse_stmt_block with `end` terminator (matches
     * the existing while/if precedent — the lexer doesn't tokenise
     * `{`/`}`, so we use `end` as the block closer). */
    if (kfl_stmt_is_ident_named(cur, "for_each")) {
        int line0 = cur->line;
        kfl_stmt_advance(L, cur, had_error);
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, line0, "for_each: expected iterator name");
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *iter_name = cur->str;
        kfl_stmt_advance(L, cur, had_error);
        if (!kfl_stmt_is_ident_named(cur, "in")) {
            kflc_diag_errorf(diag, line0,
                "for_each %s: expected `in` keyword", iter_name);
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        kfl_stmt_advance(L, cur, had_error);
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, line0,
                "for_each %s in: expected world identifier", iter_name);
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *world_ident = cur->str;
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_FOR_EACH, line0);
        n->name = iter_name;
        KflcValue wv;
        memset(&wv, 0, sizeof wv);
        wv.kind = KFLV_IDENT;
        wv.u.s  = world_ident;
        kfl_stmt_append_attr(arena, n, "world", wv, line0);

        const char *brk[] = { "end", NULL };
        KflcNode *blk = kfl_parse_stmt_block(L, cur, arena, diag,
                                              had_error, "end", brk);
        n->children = blk ? blk->children : NULL;
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        }
        return n;
    }

    /* `observe <target> from <observer> [<name>=<value>]*` — invoke
     * the world's observer-correction pipeline. Trailing `name=value`
     * pairs (mode=astrometric etc.) are captured by raw-line scan. */
    if (kfl_stmt_is_ident_named(cur, "observe")) {
        int line0 = cur->line;
        kfl_stmt_advance(L, cur, had_error);
        if (cur->kind != T_IDENT) {
            kflc_diag_errorf(diag, line0, "observe: expected target ident");
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *target_ident = cur->str;
        kfl_stmt_advance(L, cur, had_error);
        /* `observe attitude of <body> as <name>` publishes a body's
         * own orientation and rate, rather than a line of sight from
         * one body to another. It is spelled as a distinct form
         * because it is one: there is no observer and no target, only
         * a body reporting itself. The two forms are told apart by
         * what follows the first name, not by the name itself: a body
         * genuinely called `attitude` is still observed by the
         * ordinary form, because `observe attitude from earth` has no
         * `of` after the name and takes the other branch. */
        /* `observe contact of <body> as <name>` reads the same way and
         * for the same reason: a body reporting a fact about itself,
         * with no observer to name. `observe propulsion of <body> as
         * <name>` is the third of them, reporting what the craft has
         * left to spend, and `observe reference of <body> as <name>`
         * the fourth, reporting where the craft's plan says it should
         * be. The self-reporting forms share this branch rather than
         * each growing one, so a fifth is a table entry and not a
         * fifth copy of the parse. */
        static const char *const SELF_FORMS_[] = {
            "attitude", "contact", "propulsion", "reference", NULL
        };
        int attitude_form = 0;
        const char *self_marker = NULL;
        /* `observe relative <target> from <chief> as <name>` publishes
         * the target's position and velocity in the chief's own
         * local-vertical local-horizontal frame, rather than a line of
         * sight. It names two bodies like the ordinary form, so it is
         * told apart by the same rule the two above use: what follows
         * the first name. The ordinary form has `from` there; this one
         * has the target's name. A body genuinely called `relative` is
         * therefore still observed by the ordinary form, because
         * `observe relative from earth` has `from` after the name. */
        int relative_form = 0;
        if (strcmp(target_ident, "relative") == 0 &&
            cur->kind == T_IDENT && !kfl_stmt_is_ident_named(cur, "from"))
        {
            relative_form = 1;
            target_ident  = cur->str;
            kfl_stmt_advance(L, cur, had_error);
        }
        /* `observe port <port> of <body> as <name>` publishes how far
         * a named docking port on a body is from mated with the port
         * it faces, and whether the contact it just made was a
         * capture. It is the only form naming something inside an
         * assembly rather than a body, so it carries two names, and
         * the same rule tells it apart: a body genuinely called
         * `port` still takes the ordinary form, because `observe port
         * from earth` has `from` where this form has a port name. */
        int         port_form = 0;
        const char *port_ident = NULL;
        if (!relative_form && strcmp(target_ident, "port") == 0 &&
            cur->kind == T_IDENT && !kfl_stmt_is_ident_named(cur, "from"))
        {
            port_ident = cur->str;
            kfl_stmt_advance(L, cur, had_error);
            if (!kfl_stmt_is_ident_named(cur, "of")) {
                /* A body genuinely called `port` reaches here when
                 * the line-of-sight form is written with a trailing
                 * key rather than `from` first, so the diagnostic
                 * names both readings rather than assuming this one. */
                kflc_diag_errorf(diag, line0,
                    "observe port %s: expected `of` and the name of the "
                    "body that carries the port; if `port` is a body "
                    "here, its line-of-sight form is `observe port from "
                    "<observer>` with `from` before any key",
                    port_ident);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            kfl_stmt_advance(L, cur, had_error);
            if (cur->kind != T_IDENT) {
                kflc_diag_errorf(diag, line0,
                    "observe port %s of: expected a body name", port_ident);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            /* As the self-reporting forms below: the cursor stays on
             * the body name, which is where the trailing-clause scan
             * takes the remainder of the line from. */
            target_ident  = cur->str;
            port_form     = 1;
            attitude_form = 1;
        }
        /* `observe detect <payload> of <target> as <name>` and
         * `observe track <payload> of <target> [modality=<m>] as <name>`
         * name a payload and a body rather than an observer and a
         * target, so they carry two names like the port form and are
         * told apart by the same rule: a body genuinely called `detect`
         * or `track` still takes the line-of-sight form, because
         * `observe detect from earth` has `from` where these forms have
         * a payload name. */
        int         defense_form = 0;   /* 1 detect, 2 track, 3 effect */
        const char *payload_ident = NULL;
        /* `observe effect <payload> as <name>` names a payload and no
         * body: an effector's result belongs to the effector, and the
         * body it was aimed at was named by the `engage` that produced
         * it. A body genuinely called `effect` keeps its line-of-sight
         * form, which puts `from` or `as` where this form has a
         * payload name. */
        if (!relative_form && !port_form &&
            strcmp(target_ident, "effect") == 0 &&
            cur->kind == T_IDENT && !kfl_stmt_is_ident_named(cur, "from") &&
            !kfl_stmt_is_ident_named(cur, "as"))
        {
            defense_form  = 3;
            payload_ident = cur->str;
            /* The cursor stays on the payload name, which is where the
             * trailing-clause scan takes the remainder of the line
             * from: this form's only clause is `as`. */
            target_ident  = cur->str;
            attitude_form = 1;
        }
        if (!relative_form && !port_form && !defense_form &&
            (strcmp(target_ident, "detect") == 0 ||
             strcmp(target_ident, "track") == 0) &&
            cur->kind == T_IDENT && !kfl_stmt_is_ident_named(cur, "from"))
        {
            defense_form  = strcmp(target_ident, "detect") == 0 ? 1 : 2;
            payload_ident = cur->str;
            kfl_stmt_advance(L, cur, had_error);
            if (!kfl_stmt_is_ident_named(cur, "of")) {
                kflc_diag_errorf(diag, line0,
                    "observe %s %s: expected `of` and the name of the "
                    "body observed; if `%s` is a body here, its "
                    "line-of-sight form is `observe %s from <observer>` "
                    "with `from` before any key",
                    defense_form == 1 ? "detect" : "track", payload_ident,
                    defense_form == 1 ? "detect" : "track",
                    defense_form == 1 ? "detect" : "track");
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            kfl_stmt_advance(L, cur, had_error);
            if (cur->kind != T_IDENT) {
                kflc_diag_errorf(diag, line0,
                    "observe %s %s of: expected a body name",
                    defense_form == 1 ? "detect" : "track", payload_ident);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            /* As the port form: the cursor stays on the body name,
             * which is where the trailing-clause scan starts. */
            target_ident  = cur->str;
            attitude_form = 1;
        }
        /* Not after the relative branch has consumed a target name: a
         * target that happens to be called `attitude`, `contact`,
         * `propulsion` or `reference` is a name here, not a form, and
         * re-entering the branch below would rewrite the target a
         * second time and leave the diagnostic naming the wrong
         * body. */
        if (!relative_form && !port_form && !defense_form &&
            kfl_stmt_is_ident_named(cur, "of"))
        {
            for (int k = 0; SELF_FORMS_[k]; k++) {
                if (strcmp(target_ident, SELF_FORMS_[k]) == 0) {
                    self_marker = SELF_FORMS_[k];
                }
            }
        }
        if (self_marker) {
            kfl_stmt_advance(L, cur, had_error);
            if (cur->kind != T_IDENT) {
                kflc_diag_errorf(diag, line0,
                    "observe %s of: expected a body name", self_marker);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            /* The cursor stays on the body name, where the ordinary
             * form leaves it on the observer: the trailing-clause
             * scan below takes the remainder of the line from there. */
            target_ident   = cur->str;
            attitude_form  = 1;
        }
        if (!attitude_form && !kfl_stmt_is_ident_named(cur, "from")) {
            if (relative_form) {
                kflc_diag_errorf(diag, line0,
                    "observe relative %s: expected `from` after the "
                    "target name", target_ident);
            } else {
                kflc_diag_errorf(diag, line0,
                    "observe %s: expected `from` keyword", target_ident);
            }
            *had_error = 1;
            while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        char *observer_ident = target_ident;
        if (!attitude_form) {
            kfl_stmt_advance(L, cur, had_error);
            if (cur->kind != T_IDENT) {
                kflc_diag_errorf(diag, line0,
                    "observe %s from: expected observer ident",
                    target_ident);
                *had_error = 1;
                while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
                if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
                return NULL;
            }
            observer_ident = cur->str;
        }
        /* Capture rest of line for trailing named-args. */
        char *raw = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_OBSERVE, line0);
        n->name = target_ident;
        KflcValue ov;
        memset(&ov, 0, sizeof ov);
        ov.kind = KFLV_IDENT; ov.u.s = observer_ident;
        kfl_stmt_append_attr(arena, n, "observer", ov, line0);
        if (attitude_form || relative_form) {
            KflcValue kv;
            memset(&kv, 0, sizeof kv);
            kv.kind = KFLV_IDENT;
            /* The port form's marker carries the port's own name
             * rather than a bare 1, because that name is the second
             * thing the statement said and the round trip has to
             * print it back. */
            kv.u.s  = kflc_arena_strdup(arena,
                          port_form    ? port_ident
                        : defense_form ? payload_ident : "1");
            const char *marker = relative_form   ? "relative"
                               : port_form       ? "port"
                               : defense_form == 1 ? "detect"
                               : defense_form == 2 ? "track"
                               : defense_form == 3 ? "effect"
                               : self_marker     ? self_marker : "attitude";
            kfl_stmt_append_attr(arena, n, marker, kv, line0);
        }

        /* Parse trailing `key=value` pairs (whitespace-separated).
         * Each value lexes as an identifier (KFLV_IDENT). A trailing
         * `as <name>` clause (Grammar 3.2) names the observation
         * channel; it must be the last clause on the line and lands
         * as the `as` attr. */
        char *p = kfl_stmt_trim(raw);
        while (*p) {
            while (*p == ' ' || *p == '\t') p++;
            if (!*p) break;
            char *kbeg = p;
            while (*p && *p != '=' && *p != ' ' && *p != '\t') p++;
            if (*p != '=') {
                /* `against <port> of <body>` names the passive port a
                 * port observe measures against, and `full` asks that
                 * form for its two optional channels. Both are read
                 * only on the port form: a body or a sensor called
                 * `against` or `full` is reached through the words
                 * that already introduce one, `of` and `through`, so
                 * claiming these two everywhere would take names away
                 * from every other form for no gain. They must precede
                 * `as`, which remains the last clause on the line. */
                if (port_form && (size_t)(p - kbeg) == 7 &&
                    strncmp(kbeg, "against", 7) == 0) {
                    char *pbeg, *obeg, *bbeg, *pend, *oend, *bend;
                    char  saved_b;
                    while (*p == ' ' || *p == '\t') p++;
                    pbeg = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    pend = p;
                    while (*p == ' ' || *p == '\t') p++;
                    obeg = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    oend = p;
                    while (*p == ' ' || *p == '\t') p++;
                    bbeg = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    bend = p;
                    saved_b = *bend;
                    *pend = '\0'; *oend = '\0'; *bend = '\0';
                    if (pbeg[0] == '\0' || bbeg[0] == '\0' ||
                        strcmp(obeg, "of") != 0) {
                        kflc_diag_errorf(diag, line0,
                            "observe port %s of %s: `against` names the "
                            "port this one is measured against and the "
                            "body that carries it, as `against <port> of "
                            "<body>`", port_ident, target_ident);
                        *had_error = 1;
                        return n;
                    }
                    {
                        KflcValue av, bv;
                        memset(&av, 0, sizeof av);
                        memset(&bv, 0, sizeof bv);
                        av.kind = KFLV_IDENT;
                        av.u.s  = kflc_arena_strdup(arena, pbeg);
                        bv.kind = KFLV_IDENT;
                        bv.u.s  = kflc_arena_strdup(arena, bbeg);
                        kfl_stmt_append_attr(arena, n, "against", av, line0);
                        kfl_stmt_append_attr(arena, n, "against_body", bv,
                                             line0);
                    }
                    p = bend;
                    if (saved_b) { *p = saved_b; p++; }
                    continue;
                }
                if (port_form && (size_t)(p - kbeg) == 4 &&
                    strncmp(kbeg, "full", 4) == 0) {
                    char saved_f = *p;
                    KflcValue fv;
                    memset(&fv, 0, sizeof fv);
                    fv.kind = KFLV_IDENT;
                    fv.u.s  = kflc_arena_strdup(arena, "1");
                    kfl_stmt_append_attr(arena, n, "full", fv, line0);
                    if (saved_f) p++;
                    continue;
                }
                /* `through <sensor>` routes the observe's numeric
                 * components through a declared model chain, and
                 * `with truth` publishes the uncorrupted components
                 * beside them. Both are bare words rather than
                 * `key=value` pairs because that is how the design
                 * spells them, and both must precede `as`, which
                 * remains the last clause on the line. */
                if ((size_t)(p - kbeg) == 7 &&
                    strncmp(kbeg, "through", 7) == 0) {
                    while (*p == ' ' || *p == '\t') p++;
                    char *nbeg = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    char saved_t = *p;
                    *p = '\0';
                    if (nbeg[0] == '\0') {
                        kflc_diag_errorf(diag, line0,
                            "observe %s: `through` requires a sensor "
                            "name", target_ident);
                        *had_error = 1;
                        return n;
                    }
                    KflcValue tv;
                    memset(&tv, 0, sizeof tv);
                    tv.kind = KFLV_IDENT;
                    tv.u.s  = kflc_arena_strdup(arena, nbeg);
                    kfl_stmt_append_attr(arena, n, "through", tv, line0);
                    if (saved_t) { *p = saved_t; p++; }
                    continue;
                }
                if ((size_t)(p - kbeg) == 4 &&
                    strncmp(kbeg, "with", 4) == 0) {
                    while (*p == ' ' || *p == '\t') p++;
                    char *nbeg = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    char saved_w = *p;
                    *p = '\0';
                    if (strcmp(nbeg, "truth") != 0) {
                        kflc_diag_errorf(diag, line0,
                            "observe %s: `with` takes `truth` and "
                            "nothing else", target_ident);
                        *had_error = 1;
                        return n;
                    }
                    KflcValue wv;
                    memset(&wv, 0, sizeof wv);
                    wv.kind = KFLV_IDENT;
                    wv.u.s  = kflc_arena_strdup(arena, "1");
                    kfl_stmt_append_attr(arena, n, "truth", wv, line0);
                    if (saved_w) { *p = saved_w; p++; }
                    continue;
                }
                if ((size_t)(p - kbeg) == 2 && strncmp(kbeg, "as", 2) == 0) {
                    while (*p == ' ' || *p == '\t') p++;
                    char *nbeg = p;
                    while (*p && *p != ' ' && *p != '\t') p++;
                    int more = (*p != '\0');
                    *p = '\0';
                    if (nbeg[0] == '\0') {
                        kflc_diag_errorf(diag, line0,
                            "observe %s: `as` requires a channel name",
                            target_ident);
                        *had_error = 1;
                        return n;
                    }
                    if (more) {
                        p++;
                        while (*p == ' ' || *p == '\t') p++;
                        if (*p) {
                            kflc_diag_errorf(diag, line0,
                                "observe %s: `as %s` must be the last "
                                "clause on the line", target_ident, nbeg);
                            *had_error = 1;
                            return n;
                        }
                    }
                    KflcValue av;
                    memset(&av, 0, sizeof av);
                    av.kind = KFLV_IDENT;
                    av.u.s  = kflc_arena_strdup(arena, nbeg);
                    kfl_stmt_append_attr(arena, n, "as", av, line0);
                    return n;
                }
                kflc_diag_errorf(diag, line0,
                    "observe %s: expected `name=value` after observer",
                    target_ident);
                *had_error = 1;
                return n;
            }
            char *kend = p;
            *kend = '\0';
            char *key = kflc_arena_strdup(arena, kbeg);
            p++; /* past '=' */
            char *vbeg = p;
            while (*p && *p != ' ' && *p != '\t') p++;
            char saved_v = *p; *p = '\0';
            char *val = kflc_arena_strdup(arena, vbeg);
            if (saved_v) { *p = saved_v; }
            KflcValue v;
            memset(&v, 0, sizeof v);
            v.kind = KFLV_IDENT; v.u.s = val;
            kfl_stmt_append_attr(arena, n, key, v, line0);
        }
        /* Reached only when no `as` clause was found; the clause
         * returns above. The line-of-sight form without one is a print
         * statement and stays one. The relative form has no print
         * spelling: without `as` it would fall through to the
         * line-of-sight print and report a different physical quantity
         * than the one it names, between two bodies the reader has
         * named on purpose, so it is refused instead. */
        if (relative_form) {
            kflc_diag_errorf(diag, line0,
                "observe relative %s from %s: this form publishes an "
                "observation channel and requires `as <name>`",
                target_ident, observer_ident);
            *had_error = 1;
        }
        return n;
    }

    /* `allocator = <arena_name>` fn-prologue binding. Lives in fn
     * bodies and parses to a KFLN_ALLOCATOR_BIND node whose `name`
     * is the arena identifier. Emit injects the arena handle as an
     * implicit C-local of the fn body, replacing malloc/heap
     * allocations within the fn with arena_alloc calls. Parsed here
     * (statement context) rather than fn-header so the full fn body
     * parser owns the dispatch; emit enforces the "must appear
     * before non-binding statements" rule. */
    if (kfl_stmt_is_ident_named(cur, "allocator")) {
        int line_a = cur->line;
        /* `=` is not in the lexer's token alphabet; same as
         * let/const, we capture the rest of the line as raw text
         * and string-parse `= <ident>`. */
        char *raw = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);   /* now cur = newline */
        char *p = kfl_stmt_trim(raw);
        if (*p != '=') {
            kflc_diag_errorf(diag, line_a,
                "allocator: expected `= <arena_name>`");
            *had_error = 1;
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        p++;
        while (*p == ' ' || *p == '\t') p++;
        char *name_start = p;
        while (*p && !(*p == ' ' || *p == '\t')) p++;
        *p = '\0';
        if (*name_start == '\0') {
            kflc_diag_errorf(diag, line_a,
                "allocator: expected arena name identifier after `=`");
            *had_error = 1;
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_ALLOCATOR_BIND, line_a);
        n->name = kflc_arena_strdup(arena, name_start);
        return n;
    }

    /* `while <expr> ... end` */
    if (kfl_stmt_is_ident_named(cur, "while")) {
        char *body_src = kfl_stmt_take_line_remainder(L, arena);
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        char *trimmed = kfl_stmt_trim(body_src);
        KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_WHILE, line);
        n->expr = kflc_parse_expr(trimmed, arena, diag, line);
        if (!n->expr) *had_error = 1;

        const char *brk[] = { "end", NULL };
        KflcNode *blk = kfl_parse_stmt_block(L, cur, arena, diag, had_error,
                                              "end", brk);
        n->children = blk ? blk->children : NULL;
        if (kfl_stmt_is_ident_named(cur, "end")) {
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        }
        return n;
    }

    /* Assignment or bare expression: `<name> = <expr>` or just `<expr>`.
     * Distinguish by looking at the whole line for an unparenthesised `=`. */
    char prefix[64] = "";
    if (cur->kind == T_IDENT && cur->str) snprintf(prefix, sizeof prefix, "%s ", cur->str);
    else if (cur->kind == T_INT)          snprintf(prefix, sizeof prefix, "%ld ", cur->i);
    else if (cur->kind == T_FLOAT)        snprintf(prefix, sizeof prefix, "%g ", cur->f);
    else if (cur->kind == T_STRING) {
        /* Bare strings as a statement are nonsense; emit a diagnostic. */
        kflc_diag_errorf(diag, line, "stmt: unexpected string literal at start of statement");
        *had_error = 1;
        while (!kfl_stmt_at_nl(cur) && !kfl_stmt_at_eof2(cur)) kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
        return NULL;
    }
    char *rest = kfl_stmt_take_line_remainder(L, arena);
    /* The leading token is re-joined to the rest of the line with a
     * separating space, which would split a dotted name the
     * expression lexer folds into one identifier (`craft.vel_x`).
     * Drop the separator inside an on_step body, where body state is
     * addressed that way, so the name reaches the expression parser
     * whole. Everywhere else the dot stays a lexical error, which is
     * what it is today. Only the tight form joins: `craft . vel_x`
     * keeps its spaces and stays an error there too. */
    if (g_rl_on_step_depth > 0 && rest[0] == '.' &&
        cur->kind == T_IDENT && cur->str) {
        size_t plen = strlen(prefix);
        if (plen > 0 && prefix[plen - 1] == ' ') prefix[plen - 1] = '\0';
    }
    size_t pn = strlen(prefix), rn = strlen(rest);
    char *line_src = (char *)kflc_arena_alloc(arena, pn + rn + 1);
    memcpy(line_src, prefix, pn);
    memcpy(line_src + pn, rest, rn + 1);

    long eq = find_assign(line_src);
    if (eq >= 0 && cur->kind == T_IDENT && cur->str) {
        /* `<lhs> = <rhs>` assignment. The LHS must be either:
         *   - a single ident (regular scalar assign), OR
         *   - `<ident>[<expr>]` indexed assign into a vector.
         * We parse the LHS through the expression sub-parser and
         * dispatch on the resulting AST shape. */
        char *lhs_buf = (char *)kflc_arena_alloc(arena, (size_t)eq + 1);
        memcpy(lhs_buf, line_src, (size_t)eq);
        lhs_buf[eq] = '\0';
        char *lhs_trim = kfl_stmt_trim(lhs_buf);
        KflcExpr *lhs_expr = kflc_parse_expr(lhs_trim, arena, diag, line);
        if (!lhs_expr) {
            *had_error = 1;
            kfl_stmt_advance(L, cur, had_error);
            if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
            return NULL;
        }
        KflcExpr *rhs_expr = kflc_parse_expr(line_src + eq + 1, arena, diag, line);
        if (!rhs_expr) *had_error = 1;
        kfl_stmt_advance(L, cur, had_error);
        if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);

        if (lhs_expr->kind == KFLE_IDENT) {
            KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_ASSIGN, line);
            n->name = lhs_expr->u.ident;
            n->expr = rhs_expr;
            return n;
        }
        if (lhs_expr->kind == KFLE_INDEX &&
            lhs_expr->u.index.base &&
            lhs_expr->u.index.base->kind == KFLE_IDENT)
        {
            /* Single-level vector index. Kept on STMT_INDEX_ASSIGN so
             * the B5 KFLC_VEC_W bounds-check macro applies. */
            KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_INDEX_ASSIGN, line);
            n->name  = lhs_expr->u.index.base->u.ident;
            n->expr  = rhs_expr;
            n->expr2 = lhs_expr->u.index.idx;
            return n;
        }
        if (lhs_expr->kind == KFLE_INDEX) {
            /* B2: deeper index chain (e.g. matrix `m[i][j]`). The
             * shape check is deferred to kflc_emit_lvalue at emit
             * time so error messages cite the actual ident + type. */
            KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_LVALUE_ASSIGN, line);
            n->expr  = lhs_expr;    /* full LHS expr */
            n->expr2 = rhs_expr;    /* RHS expr */
            return n;
        }
        kflc_diag_errorf(diag, line,
            "stmt: assignment LHS must be a name, `<name>[<expr>]`, "
            "or matrix `<name>[<row>][<col>]`");
        *had_error = 1;
        return NULL;
    }

    /* Bare expression statement. Generates `(void)expr;` in C++. */
    KflcNode *n = kfl_stmt_new_node(arena, KFLN_STMT_EXPR, line);
    n->expr = kflc_parse_expr(line_src, arena, diag, line);
    if (!n->expr) *had_error = 1;
    kfl_stmt_advance(L, cur, had_error);
    if (kfl_stmt_at_nl(cur)) kfl_stmt_advance(L, cur, had_error);
    return n;
}

/* Parse a sequence of statements until we hit `terminator` (or one of
 * the `also_break` keywords). The terminator itself stays as `cur` —
 * the caller decides whether to consume it. Returns a parent node
 * whose `children` is the linked list of statements, or NULL on error.
 */

KflcNode *kfl_parse_stmt_block(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error,
                                const char *terminator,
                                const char **also_break)
{
    KflcNode *parent = kfl_stmt_new_node(arena, KFLN_FN, 0);   /* placeholder kind */
    for (;;) {
        kfl_stmt_skip_newlines(L, cur, had_error);
        if (kfl_stmt_at_eof2(cur)) {
            kflc_diag_errorf(diag, cur->line,
                             "unexpected EOF inside fn body (missing `%s`)",
                             terminator);
            *had_error = 1;
            return parent;
        }
        if (kfl_stmt_is_ident_named(cur, terminator)) return parent;
        if (also_break) {
            for (int i = 0; also_break[i]; i++) {
                if (kfl_stmt_is_ident_named(cur, also_break[i])) return parent;
            }
        }
        KflcNode *s = kfl_stmt_parse_stmt(L, cur, arena, diag, had_error);
        if (s) kfl_stmt_append_child(parent, s);
    }
}
