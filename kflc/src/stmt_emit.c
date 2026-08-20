/* stmt_emit.c - the statement emitter and the block-scope
 * tracker for heap-typed lets. */
#include "stmt_internal.h"

BlockScope g_b1_scopes[B1_MAX_SCOPE_DEPTH];

int        g_b1_depth = 0;        /* 0 == fn-body root */
static KflcArena *g_b1_arena = NULL;

KflcType   g_b1_fn_return_type = KFLT_VOID;

/* ---- Grammar 3.2 reinforcement learning statements ---------------- */

void kfl_emit_stmt_reset_scopes(KflcArena *arena, KflcType fn_return_type);
void kfl_emit_stmt_reset_scopes_ex(KflcArena *arena, KflcType fn_return_type,
                                   const char *fn_return_subtype);
static void b1_scope_push_(void);
static void b1_scope_pop_(void);
static void b1_scope_add_let_(const char *name, KflcType type);
static void b1_emit_free_one_(FILE *out, const char *name,
                              KflcType type, int indent);
static void b1_emit_frees_current_(FILE *out, int indent);
void kfl_emit_stmt_drain_root(FILE *out, int indent);
static int b1_total_live_(void);
static const char *observed_cell_base_(const KflcExpr *e,
                                        const KflcExprCtx *ctx);
static void emit_observed_cell_notify_(FILE *out, const KflcExpr *lhs,
                                        const KflcExprCtx *ctx, int indent);
static void b1_emit_frees_all_(FILE *out, int indent);
static void emit_indent(FILE *out, int n);
static int emit_expr_stmt_value(FILE *out, const KflcExpr *e,
                                 KflcType target_type,
                                 const KflcExprCtx *ctx, KflcDiag *diag);
static void emit_c_str_lit_(FILE *out, const char *s);

void kfl_emit_stmt_reset_scopes(KflcArena *arena, KflcType fn_return_type)
{
    kfl_emit_stmt_reset_scopes_ex(arena, fn_return_type, NULL);
}

void kfl_emit_stmt_reset_scopes_ex(KflcArena *arena, KflcType fn_return_type,
                                   const char *fn_return_subtype)
{
    g_b1_depth = 0;
    g_b1_arena = arena;
    g_b1_fn_return_type = fn_return_type;
    g_b1_fn_return_subtype = fn_return_subtype;
    for (int i = 0; i < B1_MAX_SCOPE_DEPTH; i++) {
        g_b1_scopes[i].n = 0;
        /* names/types/cap intentionally NOT cleared — the arena owns
         * the backing storage across fn boundaries; resetting n is
         * enough to "empty" the scope, and we can reuse the cap. */
    }
}

static void b1_scope_push_(void)
{
    if (g_b1_depth + 1 >= B1_MAX_SCOPE_DEPTH) {
        /* Hard cap. Realistic KFL forms don't nest 16 deep; if they
         * do, just stop tracking; the deeper scopes will leak their
         * heap-typed lets until the process exits. */
        return;
    }
    g_b1_depth++;
    g_b1_scopes[g_b1_depth].n = 0;
}

static void b1_scope_pop_(void)
{
    if (g_b1_depth == 0) return;
    g_b1_depth--;
}

/* Register a let in the CURRENT scope, if it's heap-typed. Depth 0
 * is the fn-body root and is tracked here too; emit.c does not keep
 * a parallel `frees[]` registry. */

static void b1_scope_add_let_(const char *name, KflcType type)
{
    if (type != KFLT_VECTOR && type != KFLT_MATRIX) return;
    BlockScope *s = &g_b1_scopes[g_b1_depth];
    if (s->n == s->cap) {
        int nc = s->cap ? s->cap * 2 : 4;
        const char **nn = (const char **)kflc_arena_alloc(
            g_b1_arena, sizeof(char *) * (size_t)nc);
        KflcType *nt = (KflcType *)kflc_arena_alloc(
            g_b1_arena, sizeof(KflcType) * (size_t)nc);
        if (s->n > 0) {
            memcpy(nn, s->names, sizeof(char *)   * (size_t)s->n);
            memcpy(nt, s->types, sizeof(KflcType) * (size_t)s->n);
        }
        s->names = nn;
        s->types = nt;
        s->cap   = nc;
    }
    s->names[s->n] = name;
    s->types[s->n] = type;
    s->n++;
}

/* Emit a free for a single binding. Centralises the
 * `if (KFLT_MATRIX) k26c_mat_free else k26c_vec_free` repetition. */

static void b1_emit_free_one_(FILE *out, const char *name,
                              KflcType type, int indent)
{
    emit_indent(out, indent);
    if (type == KFLT_MATRIX) {
        fprintf(out, "k26c_mat_free(&%s);\n", name);
    } else {
        fprintf(out, "k26c_vec_free(&%s);\n", name);
    }
}

/* Emit frees for the CURRENT scope only (in reverse declaration
 * order). Used at the end of each block body, i.e. before the
 * closing `}` of a while-loop or if-branch. Per-iteration semantic.
 * For depth 0 (fn-body root), callers should use
 * kfl_emit_stmt_drain_root from emit.c at fall-through end. */

static void b1_emit_frees_current_(FILE *out, int indent)
{
    const BlockScope *s = &g_b1_scopes[g_b1_depth];
    for (int i = s->n - 1; i >= 0; i--) {
        b1_emit_free_one_(out, s->names[i], s->types[i], indent);
    }
}

/* Public helper: emit frees for depth 0 (the fn-body root). Called
 * from emit.c at fn-body fall-through end to release any heap-typed
 * lets that didn't escape via a return. */

void kfl_emit_stmt_drain_root(FILE *out, int indent)
{
    /* Walk depth 0 specifically; deeper scopes have already drained
     * via their own end-of-block hooks (b1_emit_frees_current_). */
    const BlockScope *s = &g_b1_scopes[0];
    for (int i = s->n - 1; i >= 0; i--) {
        b1_emit_free_one_(out, s->names[i], s->types[i], indent);
    }
}

/* Total live heap-typed let count across all currently-open scopes.
 * Used by the RETURN handler to decide whether to bother with the
 * `_kfl_rv` temp-save dance (no heap lets → no risk of returning a
 * dangling reference → emit the plain `return expr;` form). */

static int b1_total_live_(void)
{
    int total = 0;
    for (int d = 0; d <= g_b1_depth; d++) total += g_b1_scopes[d].n;
    return total;
}

/* Walk an lvalue expression to its base IDENT; return the name if
 * the base resolves to an observed-cell form-arg in ctx->bindings,
 * NULL otherwise. The walker uses this after each assignment
 * statement to decide whether to fan out to subscribers via
 * `kfl_cell_notify(&_kfl_cell_<name>)`. */

static const char *observed_cell_base_(const KflcExpr *e,
                                        const KflcExprCtx *ctx)
{
    if (!e || !ctx || !ctx->bindings) return NULL;
    /* Walk index chains down to the root identifier. */
    while (e && e->kind == KFLE_INDEX) {
        e = e->u.index.base;
    }
    if (!e || e->kind != KFLE_IDENT || !e->u.ident) return NULL;
    for (int i = ctx->n_bindings - 1; i >= 0; i--) {
        if (strcmp(ctx->bindings[i].name, e->u.ident) == 0) {
            if (ctx->bindings[i].is_form_arg &&
                ctx->bindings[i].is_observed_cell) {
                return e->u.ident;
            }
            return NULL;
        }
    }
    return NULL;
}

static void emit_observed_cell_notify_(FILE *out, const KflcExpr *lhs,
                                        const KflcExprCtx *ctx, int indent)
{
    const char *name = observed_cell_base_(lhs, ctx);
    if (!name) return;
    emit_indent(out, indent);
    fprintf(out, "kfl_cell_notify(&_kfl_cell_%s);\n", name);
}

/* Emit frees for ALL scopes (depths 0..current), in reverse depth
 * order, reverse declaration order within each level. Includes
 * depth 0 (the fn-body root) so nested returns inside if/while
 * drain every heap-typed let, not just the inner ones. */

static void b1_emit_frees_all_(FILE *out, int indent)
{
    for (int d = g_b1_depth; d >= 0; d--) {
        const BlockScope *s = &g_b1_scopes[d];
        for (int i = s->n - 1; i >= 0; i--) {
            b1_emit_free_one_(out, s->names[i], s->types[i], indent);
        }
    }
}

static void emit_indent(FILE *out, int n)
{
    for (int i = 0; i < n; i++) fputc(' ', out);
}

static int emit_expr_stmt_value(FILE *out, const KflcExpr *e,
                                 KflcType target_type,
                                 const KflcExprCtx *ctx, KflcDiag *diag)
{
    if (target_type == KFLT_INT) {
        fputs("(int)(", out);
        if (kflc_emit_expr(out, e, ctx, diag)) return 1;
        fputs(")", out);
    } else if (target_type == KFLT_BOOL) {
        fputs("(bool)(", out);
        if (kflc_emit_expr(out, e, ctx, diag)) return 1;
        fputs(")", out);
    } else if (target_type == KFLT_STRING) {
        /* Strings aren't part of the expr grammar; emit literal "". */
        fputs("\"\"", out);
        (void)ctx; (void)diag; (void)e;
    } else {
        if (kflc_emit_expr(out, e, ctx, diag)) return 1;
    }
    return 0;
}

/* Emit `s` as a C++ double-quoted string literal (escapes decoded in
 * the KflcValue are re-escaped for the emitted source). Used by the
 * print statement; kept local because emit.c's emit_string_literal is
 * file-static there. */

static void emit_c_str_lit_(FILE *out, const char *s)
{
    fputc('"', out);
    for (; s && *s; s++) {
        switch (*s) {
        case '"':  fputs("\\\"", out); break;
        case '\\': fputs("\\\\", out); break;
        case '\n': fputs("\\n",  out); break;
        case '\t': fputs("\\t",  out); break;
        case '\r': fputs("\\r",  out); break;
        default:   fputc(*s, out);     break;
        }
    }
    fputc('"', out);
}

int kfl_emit_stmt(FILE *out, const KflcNode *s,
                   const KflcExprCtx *ctx, KflcDiag *diag,
                   int indent)
{
    if (!s) return 0;
    switch (s->kind) {
    case KFLN_STMT_ENGAGE: {
        /* The environment emitter resolved this statement to one
         * engagement helper and left its index on the node, so the
         * payload, the target and every table index they need are
         * compile-time constants and nothing is looked up here. The
         * statement is admissible in an `on_step` body alone, which is
         * the one place these names are in scope. */
        emit_indent(out, indent);
        fprintf(out, "kflrl_engage_%ld_(world, _kfl_pay, _kfl_payu, "
                     "_kfl_payp, _kfl_dt, _kfl_eng);\n",
                s->position.kind == KFLV_INT ? s->position.u.i : 0L);
        return 0;
    }

    case KFLN_STMT_LET:
    case KFLN_STMT_CONST: {
        /* Register heap-typed lets with the unified scope tracker
         * (depth 0 = fn-body root, depths >= 1 = if/while bodies).
         * Registration happens before the type-specific emit so a
         * `return` mid-body that follows still sees the binding. */
        b1_scope_add_let_(s->name, s->type);
        /* Vector / matrix locals: zero-init the struct, then call
         * k26c_vec_from / k26c_mat_from with a stack-array temporary.
         * The expression must be a vector literal: nested for
         * matrices, flat for vectors. */
        if (s->type == KFLT_VECTOR) {
            emit_indent(out, indent);
            fprintf(out, "K26CVector %s = {0};\n", s->name);
            if (!s->expr) {
                kflc_diag_errorf(diag, s->line,
                    "let %s: vector requires a `[...]` or builder initializer",
                    s->name);
                return 1;
            }
            if (s->expr->kind == KFLE_VEC_LIT) {
                int n = s->expr->u.vec.n_elems;
                emit_indent(out, indent);
                fputs("{ double _ka[] = {", out);
                for (int i = 0; i < n; i++) {
                    if (i > 0) fputs(", ", out);
                    if (kflc_emit_expr(out, s->expr->u.vec.elems[i], ctx, diag)) return 1;
                }
                fprintf(out, "}; k26c_vec_from(&%s, _ka, %d); }\n", s->name, n);
                return 0;
            }
            /* Vector builder calls: linspace / zeros / ones /
             * arange. Each emits a `k26c_vec_alloc` + a fill loop. */
            if (s->expr->kind == KFLE_CALL &&
                s->expr->u.call.name)
            {
                const char *bn = s->expr->u.call.name;
                int n_args = s->expr->u.call.n_args;
                if (strcmp(bn, "zeros") == 0 && n_args == 1) {
                    emit_indent(out, indent);
                    fprintf(out, "{ size_t _n = (size_t)((int)(");
                    if (kflc_emit_expr(out, s->expr->u.call.args[0], ctx, diag)) return 1;
                    fprintf(out, ")); k26c_vec_alloc(&%s, _n); for (size_t _i = 0; _i < _n; _i++) %s.data[_i] = 0.0; }\n",
                            s->name, s->name);
                    return 0;
                }
                if (strcmp(bn, "ones") == 0 && n_args == 1) {
                    emit_indent(out, indent);
                    fprintf(out, "{ size_t _n = (size_t)((int)(");
                    if (kflc_emit_expr(out, s->expr->u.call.args[0], ctx, diag)) return 1;
                    fprintf(out, ")); k26c_vec_alloc(&%s, _n); for (size_t _i = 0; _i < _n; _i++) %s.data[_i] = 1.0; }\n",
                            s->name, s->name);
                    return 0;
                }
                if (strcmp(bn, "linspace") == 0 && n_args == 3) {
                    emit_indent(out, indent);
                    fputs("{ double _a = ", out);
                    if (kflc_emit_expr(out, s->expr->u.call.args[0], ctx, diag)) return 1;
                    fputs("; double _b = ", out);
                    if (kflc_emit_expr(out, s->expr->u.call.args[1], ctx, diag)) return 1;
                    fputs("; size_t _n = (size_t)((int)(", out);
                    if (kflc_emit_expr(out, s->expr->u.call.args[2], ctx, diag)) return 1;
                    fprintf(out, ")); k26c_vec_alloc(&%s, _n); if (_n == 1) { %s.data[0] = _a; } else { for (size_t _i = 0; _i < _n; _i++) %s.data[_i] = _a + (_b - _a) * (double)_i / (double)(_n - 1); } }\n",
                            s->name, s->name, s->name);
                    return 0;
                }
                if (strcmp(bn, "arange") == 0 && n_args == 3) {
                    emit_indent(out, indent);
                    fputs("{ double _a = ", out);
                    if (kflc_emit_expr(out, s->expr->u.call.args[0], ctx, diag)) return 1;
                    fputs("; double _b = ", out);
                    if (kflc_emit_expr(out, s->expr->u.call.args[1], ctx, diag)) return 1;
                    fputs("; double _st = ", out);
                    if (kflc_emit_expr(out, s->expr->u.call.args[2], ctx, diag)) return 1;
                    fprintf(out, "; size_t _n = (size_t)((_b - _a) / _st + 0.5); k26c_vec_alloc(&%s, _n); for (size_t _i = 0; _i < _n; _i++) %s.data[_i] = _a + (double)_i * _st; }\n",
                            s->name, s->name);
                    return 0;
                }
            }
            kflc_diag_errorf(diag, s->line,
                "let %s: vector init must be `[...]`, `zeros(N)`, "
                "`ones(N)`, `linspace(a, b, N)`, or `arange(a, b, step)`",
                s->name);
            return 1;
        }
        if (s->type == KFLT_MATRIX) {
            emit_indent(out, indent);
            fprintf(out, "K26CMatrix %s = {0};\n", s->name);
            /* Runtime `zeros(rows, cols)` constructor — the 2-arg form
             * allocates a rows×cols matrix (the 1-arg form is a vector).
             * Cells are then filled by `m[i][j] = ...` assignment. */
            if (s->expr && s->expr->kind == KFLE_CALL && s->expr->u.call.name &&
                strcmp(s->expr->u.call.name, "zeros") == 0 &&
                s->expr->u.call.n_args == 2)
            {
                emit_indent(out, indent);
                fputs("{ size_t _mr = (size_t)((int)(", out);
                if (kflc_emit_expr(out, s->expr->u.call.args[0], ctx, diag)) return 1;
                fputs(")); size_t _mc = (size_t)((int)(", out);
                if (kflc_emit_expr(out, s->expr->u.call.args[1], ctx, diag)) return 1;
                fprintf(out, ")); k26c_mat_alloc(&%s, _mr, _mc); "
                             "for (size_t _i = 0; _i < _mr * _mc; _i++) "
                             "%s.data[_i] = 0.0; }\n", s->name, s->name);
                return 0;
            }
            if (!s->expr || s->expr->kind != KFLE_VEC_LIT) {
                kflc_diag_errorf(diag, s->line,
                    "let %s: matrix requires a `[[...], [...]]` initializer "
                    "or `zeros(rows, cols)`",
                    s->name);
                return 1;
            }
            int rows = s->expr->u.vec.n_elems;
            int cols = 0;
            if (rows > 0) {
                const KflcExpr *first_row = s->expr->u.vec.elems[0];
                if (!first_row || first_row->kind != KFLE_VEC_LIT) {
                    kflc_diag_errorf(diag, s->line,
                        "let %s: matrix rows must themselves be `[...]` literals",
                        s->name);
                    return 1;
                }
                cols = first_row->u.vec.n_elems;
            }
            /* Validate every row has the same column count. */
            for (int r = 0; r < rows; r++) {
                const KflcExpr *row = s->expr->u.vec.elems[r];
                if (!row || row->kind != KFLE_VEC_LIT || row->u.vec.n_elems != cols) {
                    kflc_diag_errorf(diag, s->line,
                        "let %s: matrix row %d has wrong shape (expected %d cols)",
                        s->name, r, cols);
                    return 1;
                }
            }
            emit_indent(out, indent);
            fputs("{ double _ka[] = {", out);
            int first = 1;
            for (int r = 0; r < rows; r++) {
                const KflcExpr *row = s->expr->u.vec.elems[r];
                for (int c = 0; c < cols; c++) {
                    if (!first) fputs(", ", out);
                    first = 0;
                    if (kflc_emit_expr(out, row->u.vec.elems[c], ctx, diag)) return 1;
                }
            }
            fprintf(out, "}; k26c_mat_from(&%s, _ka, %d, %d); }\n",
                    s->name, rows, cols);
            return 0;
        }
        const char *cxx_ty = kflc_type_cxx(s->type, s->type_subtype);
        if (!cxx_ty) cxx_ty = "double";
        /* Borrow source resolution + emit type adjustment.
         *
         * For `let v: borrow T = src`, look up src's binding index in
         * the current ctx and store it on the new binding's
         * borrow_source_idx so the read-side use-after-move check
         * (expr.c KFLE_IDENT) can trace from a borrow read back to
         * its source binding and detect "source was moved" cases.
         *
         * Also enforce the "borrow needs an owning source" rule:
         * borrow RHS must be a bare identifier resolving to an
         * own / borrow / ptr / none-opaque binding (not a call
         * result, not a value-typed scalar). */
        int borrow_source_idx = -1;
        if (s->type == KFLT_OPAQUE
            && s->lifetime_qualifier == KFL_LQ_BORROW
            && s->expr
            && ctx && ctx->bindings) {
            if (s->expr->kind != KFLE_IDENT || !s->expr->u.ident) {
                kflc_diag_errorf(diag, s->line,
                    "let %s: borrow RHS must be a bare identifier "
                    "(borrow needs a named source for scope tracking)",
                    s->name);
                return 1;
            }
            int found = -1;
            for (int i = 0; i < ctx->n_bindings; i++) {
                if (!ctx->bindings[i].name) continue;
                if (strcmp(ctx->bindings[i].name, s->expr->u.ident) != 0) continue;
                found = i;
                break;
            }
            if (found < 0) {
                kflc_diag_errorf(diag, s->line,
                    "let %s: borrow source identifier `%s` is unknown",
                    s->name, s->expr->u.ident);
                return 1;
            }
            if (ctx->bindings[found].moved_from) {
                kflc_diag_errorf(diag, s->line,
                    "let %s: borrow source `%s` has already been moved",
                    s->name, s->expr->u.ident);
                return 1;
            }
            borrow_source_idx = found;
            /* Record on the newly-declared binding (find by name). */
            for (int i = 0; i < ctx->n_bindings; i++) {
                if (!ctx->bindings[i].name) continue;
                if (strcmp(ctx->bindings[i].name, s->name) != 0) continue;
                ((KflcExprBinding *)ctx->bindings)[i].borrow_source_idx =
                    borrow_source_idx;
                break;
            }
        }
        (void)borrow_source_idx;
        /* "Explicit move required" enforcement. When the LHS binding
         * is `own`-qualified and the RHS is a bare identifier
         * resolving to an own-qualified source, require the user to
         * wrap with `move(src)` so the ownership transfer is
         * syntactically visible. Aliasing two `own` bindings to the
         * same heap value would silently break the single-ownership
         * invariant; explicit move() makes the intent (and the
         * implicit invalidation of the source) clear.
         *
         * Pass-through cases (no error):
         *  - LHS is none/borrow/ptr qualified, no ownership invariant
         *  - RHS is not a bare identifier (call result, expression, NULL)
         *  - RHS identifier resolves to a non-own binding (form-arg,
         *    borrow, ptr, scalar copy) */
        if (s->type == KFLT_OPAQUE
            && s->lifetime_qualifier == KFL_LQ_OWN
            && s->expr
            && s->expr->kind == KFLE_IDENT
            && s->expr->u.ident
            && ctx && ctx->bindings) {
            for (int i = 0; i < ctx->n_bindings; i++) {
                if (!ctx->bindings[i].name) continue;
                if (strcmp(ctx->bindings[i].name, s->expr->u.ident) != 0) continue;
                if (ctx->bindings[i].lifetime_qualifier == KFL_LQ_OWN) {
                    kflc_diag_errorf(diag, s->line,
                        "let %s: own RHS is a bare `own` binding `%s`. "
                        "Wrap with `move(%s)` to make the ownership "
                        "transfer explicit.",
                        s->name, s->expr->u.ident, s->expr->u.ident);
                    return 1;
                }
                break;
            }
        }
        emit_indent(out, indent);
        if (s->kind == KFLN_STMT_CONST) fputs("const ", out);
        /* Borrow-qualified opaques emit as `const T *` so downstream
         * code can't mutate through the borrow handle (compile-error
         * from the C compiler if attempted). own / ptr / none
         * qualifiers fall through to the existing `T *` shape. */
        if (s->type == KFLT_OPAQUE
            && s->lifetime_qualifier == KFL_LQ_BORROW) {
            fprintf(out, "const %s %s = ", cxx_ty, s->name);
        } else {
            fprintf(out, "%s %s = ", cxx_ty, s->name);
        }
        if (s->expr) {
            if (emit_expr_stmt_value(out, s->expr, s->type, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(";\n", out);
        return 0;
    }
    /* Unified lvalue write. All three assignment
     * shapes (scalar bare IDENT, single-level vector index, full
     * lvalue chain) route through kflc_emit_lvalue, which decides
     * whether to emit a `kfl_arg_<name>` prefix, KFLC_VEC_W /
     * KFLC_MAT_W, or a deeper chain. The legacy three-arm dispatch
     * is preserved at the AST level (parser still classifies into
     * STMT_ASSIGN / STMT_INDEX_ASSIGN / STMT_LVALUE_ASSIGN to keep
     * round-trip serialisation honest), but each arm now synthesises
     * a KflcExpr LHS on the stack and hands it to the walker so the
     * write path has one source of truth. After the write we also
     * fire `kfl_cell_notify(&_kfl_cell_<base>)` if the base resolves
     * to an observed form-arg (E7) — covers handler-fn writes via
     * `kfl_arg_X = ...` and future vector/matrix cell writes via
     * `xs[i] = ...` per the §4.3 sketch in 12-binding-model.md. */
    case KFLN_STMT_ASSIGN: {
        emit_indent(out, indent);
        KflcExpr lhs;
        memset(&lhs, 0, sizeof lhs);
        lhs.kind     = KFLE_IDENT;
        lhs.line     = s->line;
        lhs.u.ident  = s->name;
        if (kflc_emit_lvalue(out, &lhs, ctx, diag, KFLC_LV_WRITE)) return 1;
        fputs(" = ", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(";\n", out);
        emit_observed_cell_notify_(out, &lhs, ctx, indent);
        return 0;
    }
    case KFLN_STMT_INDEX_ASSIGN: {
        emit_indent(out, indent);
        KflcExpr base, lhs;
        memset(&base, 0, sizeof base);
        memset(&lhs,  0, sizeof lhs);
        base.kind        = KFLE_IDENT;
        base.line        = s->line;
        base.u.ident     = s->name;
        lhs.kind         = KFLE_INDEX;
        lhs.line         = s->line;
        lhs.u.index.base = &base;
        lhs.u.index.idx  = s->expr2;
        if (kflc_emit_lvalue(out, &lhs, ctx, diag, KFLC_LV_WRITE)) return 1;
        fputs(" = ", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(";\n", out);
        emit_observed_cell_notify_(out, &lhs, ctx, indent);
        return 0;
    }
    case KFLN_STMT_LVALUE_ASSIGN:
        emit_indent(out, indent);
        if (!s->expr) {
            kflc_diag_errorf(diag, s->line,
                "lvalue-assign: missing LHS expression");
            return 1;
        }
        if (kflc_emit_lvalue(out, s->expr, ctx, diag, KFLC_LV_WRITE)) return 1;
        fputs(" = ", out);
        if (s->expr2) {
            if (kflc_emit_expr(out, s->expr2, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(";\n", out);
        emit_observed_cell_notify_(out, s->expr, ctx, indent);
        return 0;
    case KFLN_STMT_SERIES: {
        /* Multi-series: each statement slots its K26PSeries into
         * the enclosing fn-data's _kfl_all[] array at index
         * _kfl_count, then bumps the counter. The xs/ys caches at the
         * same index hold deep copies of the fn-local vectors so the
         * series stays valid across calls (callee may free its
         * locals). Bounded by _KFL_MAX_SERIES (16). */
        /* Heatmap: matrix-backed 2-D field. Deep-copies the matrix's
         * row-major data into the persistent _kfl_hh cache (the fn-data
         * local matrix is freed before k26plot_render runs), then wires
         * heat_data / heat_rows / heat_cols + index-space extents
         * [0,cols]×[0,rows]; NaN vmin/vmax auto-fits the color scale. */
        if (s->position.u.i == 5) {
            const KflcAttr *amat = NULL;
            for (const KflcAttr *a = s->attrs; a; a = a->next)
                if (strcmp(a->name, "mat") == 0) amat = a;
            if (!amat || amat->value.kind != KFLV_IDENT) {
                kflc_diag_errorf(diag, s->line,
                    "series_heatmap: missing matrix identifier");
                return 1;
            }
            const char *mat_name = amat->value.u.s;
            emit_indent(out, indent);
            fputs("if (_kfl_count < _KFL_MAX_SERIES) {\n", out);
            emit_indent(out, indent + 4);
            fprintf(out, "size_t _hn = %s.rows * %s.cols;\n", mat_name, mat_name);
            emit_indent(out, indent + 4);
            fputs("k26c_vec_free(&_kfl_hh[_kfl_count]);\n", out);
            emit_indent(out, indent + 4);
            fputs("k26c_vec_alloc(&_kfl_hh[_kfl_count], _hn);\n", out);
            emit_indent(out, indent + 4);
            fprintf(out, "for (size_t _i = 0; _i < _hn; _i++) "
                         "_kfl_hh[_kfl_count].data[_i] = %s.data[_i];\n", mat_name);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].kind      = K26P_HEATMAP;\n", out);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].label     = ", out);
            fputc('"', out);
            for (const unsigned char *p = (const unsigned char *)(s->name ? s->name : "");
                 p && *p; p++) {
                if (*p == '"' || *p == '\\') fputc('\\', out);
                fputc((char)*p, out);
            }
            fputs("\";\n", out);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].heat_data = _kfl_hh[_kfl_count].data;\n", out);
            emit_indent(out, indent + 4);
            fprintf(out, "_kfl_all[_kfl_count].heat_rows = %s.rows;\n", mat_name);
            emit_indent(out, indent + 4);
            fprintf(out, "_kfl_all[_kfl_count].heat_cols = %s.cols;\n", mat_name);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].heat_x0   = 0.0;\n", out);
            emit_indent(out, indent + 4);
            fprintf(out, "_kfl_all[_kfl_count].heat_x1   = (double)%s.cols;\n", mat_name);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].heat_y0   = 0.0;\n", out);
            emit_indent(out, indent + 4);
            fprintf(out, "_kfl_all[_kfl_count].heat_y1   = (double)%s.rows;\n", mat_name);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].heat_vmin = (double)NAN;\n", out);
            emit_indent(out, indent + 4);
            fputs("_kfl_all[_kfl_count].heat_vmax = (double)NAN;\n", out);
            emit_indent(out, indent + 4);
            fputs("_kfl_count++;\n", out);
            emit_indent(out, indent);
            fputs("}\n", out);
            return 0;
        }
        const KflcAttr *axs = NULL, *ays = NULL;
        for (const KflcAttr *a = s->attrs; a; a = a->next) {
            if (strcmp(a->name, "xs") == 0) axs = a;
            if (strcmp(a->name, "ys") == 0) ays = a;
        }
        if (!axs || !ays ||
            axs->value.kind != KFLV_IDENT || ays->value.kind != KFLV_IDENT)
        {
            kflc_diag_errorf(diag, s->line, "series: missing xs/ys identifiers");
            return 1;
        }
        const char *xs_name = axs->value.u.s;
        const char *ys_name = ays->value.u.s;
        const char *kind_name =
            (s->position.u.i == 0) ? "K26P_LINE" :
            (s->position.u.i == 1) ? "K26P_SCATTER" :
            (s->position.u.i == 2) ? "K26P_ERRORBAR" :
            (s->position.u.i == 3) ? "K26P_HISTOGRAM" :
            (s->position.u.i == 4) ? "K26P_BOX" :
                                     "K26P_HEATMAP";
        emit_indent(out, indent);
        fputs("if (_kfl_count < _KFL_MAX_SERIES) {\n", out);
        emit_indent(out, indent + 4);
        fputs("k26c_vec_free(&_kfl_xh[_kfl_count]);\n", out);
        emit_indent(out, indent + 4);
        fputs("k26c_vec_free(&_kfl_yh[_kfl_count]);\n", out);
        emit_indent(out, indent + 4);
        fprintf(out, "k26c_vec_copy(&_kfl_xh[_kfl_count], &%s);\n", xs_name);
        emit_indent(out, indent + 4);
        fprintf(out, "k26c_vec_copy(&_kfl_yh[_kfl_count], &%s);\n", ys_name);
        emit_indent(out, indent + 4);
        fprintf(out, "_kfl_all[_kfl_count].kind  = %s;\n", kind_name);
        emit_indent(out, indent + 4);
        fputs("_kfl_all[_kfl_count].label = ", out);
        /* Emit label literal. */
        fputc('"', out);
        for (const unsigned char *p = (const unsigned char *)(s->name ? s->name : "");
             p && *p; p++)
        {
            if (*p == '"' || *p == '\\') fputc('\\', out);
            fputc((char)*p, out);
        }
        fputs("\";\n", out);
        emit_indent(out, indent + 4);
        fputs("_kfl_all[_kfl_count].xs        = _kfl_xh[_kfl_count].data;\n", out);
        emit_indent(out, indent + 4);
        fputs("_kfl_all[_kfl_count].ys        = _kfl_yh[_kfl_count].data;\n", out);
        emit_indent(out, indent + 4);
        fputs("_kfl_all[_kfl_count].n         = _kfl_xh[_kfl_count].n;\n", out);
        emit_indent(out, indent + 4);
        fputs("_kfl_all[_kfl_count].linewidth = 2.0;\n", out);
        emit_indent(out, indent + 4);
        fputs("_kfl_count++;\n", out);
        emit_indent(out, indent);
        fputs("}\n", out);
        return 0;
    }

    case KFLN_STMT_RETURN: {
        /* "No-cross-fn-borrow-escape" check.
         *
         * Returning a borrow whose source is a fn-local binding
         * (not form-arg, not fn-arg) is a dangling-reference bug:
         * the source goes out of scope at fn return, leaving the
         * caller holding a borrow that points at freed memory.
         *
         * Form-args (is_form_arg=1) outlive the fn body, so borrows
         * to them are safe to return. Bindings without a tracked
         * borrow_source_idx are assumed safe (they aren't borrows). */
        if (s->expr && s->expr->kind == KFLE_IDENT && s->expr->u.ident
            && ctx && ctx->bindings) {
            for (int i = 0; i < ctx->n_bindings; i++) {
                if (!ctx->bindings[i].name) continue;
                if (strcmp(ctx->bindings[i].name, s->expr->u.ident) != 0) continue;
                if (ctx->bindings[i].lifetime_qualifier != KFL_LQ_BORROW) break;
                int src_idx = ctx->bindings[i].borrow_source_idx;
                if (src_idx < 0 || src_idx >= ctx->n_bindings) break;
                if (!ctx->bindings[src_idx].is_form_arg) {
                    kflc_diag_errorf(diag, s->line,
                        "return: cannot return borrow `%s`; its source "
                        "`%s` is a fn-local binding and would dangle "
                        "(no-cross-fn-borrow-escape rule).",
                        s->expr->u.ident,
                        ctx->bindings[src_idx].name);
                    return 1;
                }
                break;
            }
        }
        /* Drain every live heap-typed let across all open scopes
         * (including depth 0, the fn-body root). The drain happens
         * BEFORE
         * the actual `return`, but if the return value is itself a
         * function of one of the about-to-be-freed lets (e.g.
         * `return xs[0]` after `let xs: vector = ...`), evaluating
         * the expression after the free would be use-after-free.
         * Save the value into a temporary first, then free, then
         * return the temp. The fn return type comes from
         * kfl_emit_stmt_reset_scopes (g_b1_fn_return_type). */
        int live = b1_total_live_();
        if (s->expr && live > 0 && g_b1_fn_return_type != KFLT_VOID) {
            const char *rt = kflc_type_cxx(g_b1_fn_return_type,
                                            g_b1_fn_return_subtype);
            emit_indent(out, indent);
            fprintf(out, "%s _kfl_rv = ", rt ? rt : "double");
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
            fputs(";\n", out);
            b1_emit_frees_all_(out, indent);
            emit_indent(out, indent);
            fputs("return _kfl_rv;\n", out);
        } else {
            b1_emit_frees_all_(out, indent);
            emit_indent(out, indent);
            if (s->expr) {
                fputs("return ", out);
                if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
                fputs(";\n", out);
            } else {
                fputs("return;\n", out);
            }
        }
        return 0;
    }
    case KFLN_STMT_EXPR:
        emit_indent(out, indent);
        fputs("(void)(", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(");\n", out);
        return 0;
    case KFLN_STMT_PRINT: {
        /* Each arg emits its own matched-argument fprintf/fputs (no
         * unmatched-specifier musl UB), then a trailing newline. String
         * args print verbatim; numeric args print as a double (%.6g). */
        for (const KflcNode *arg = s->children; arg; arg = arg->next) {
            emit_indent(out, indent);
            if (arg->position.kind == KFLV_STR) {
                fputs("fputs(", out);
                emit_c_str_lit_(out, arg->position.u.s);
                fputs(", stdout);\n", out);
            } else if (arg->expr) {
                fputs("fprintf(stdout, \"%.6g\", (double)(", out);
                if (kflc_emit_expr(out, arg->expr, ctx, diag)) return 1;
                fputs("));\n", out);
            }
        }
        emit_indent(out, indent);
        fputs("fputc('\\n', stdout);\n", out);
        return 0;
    }
    case KFLN_STMT_IF: {
        emit_indent(out, indent);
        fputs("if (", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(") {\n", out);
        /* B1: push a scope for the then-branch so heap-typed lets
         * declared inside are tracked + freed at branch exit. Same
         * shape for the else-branch below. The drain happens just
         * before the closing `}` so a fall-through cleans up; if the
         * branch's last statement is a `return`, the RETURN case
         * already drained, so skip — emitting both would leave
         * unreachable `k26c_vec_free` calls in the generated C++. */
        b1_scope_push_();
        int last_was_return = 0;
        for (const KflcNode *c = s->children; c; c = c->next) {
            if (kfl_emit_stmt(out, c, ctx, diag, indent + 4)) return 1;
            last_was_return = (c->kind == KFLN_STMT_RETURN);
        }
        if (!last_was_return) b1_emit_frees_current_(out, indent + 4);
        b1_scope_pop_();
        emit_indent(out, indent);
        fputs("}", out);
        if (s->else_children) {
            fputs(" else {\n", out);
            b1_scope_push_();
            last_was_return = 0;
            for (const KflcNode *c = s->else_children; c; c = c->next) {
                if (kfl_emit_stmt(out, c, ctx, diag, indent + 4)) return 1;
                last_was_return = (c->kind == KFLN_STMT_RETURN);
            }
            if (!last_was_return) b1_emit_frees_current_(out, indent + 4);
            b1_scope_pop_();
            emit_indent(out, indent);
            fputs("}\n", out);
        } else {
            fputs("\n", out);
        }
        return 0;
    }
    case KFLN_STMT_WHILE: {
        emit_indent(out, indent);
        fputs("while (", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(") {\n", out);
        /* B1: push a scope; the drain fires before the closing `}`
         * which is the loop's natural back-edge. Net effect: each
         * iteration sees a freshly-zeroed K26CVector / K26CMatrix
         * for any heap-typed `let` declared in the body, and the
         * prior iteration's heap buffer is freed before the next
         * one allocates. Skip the back-edge drain when the loop
         * body's last statement is `return` — the RETURN case
         * already drained, and the back-edge is unreachable. */
        b1_scope_push_();
        int last_was_return = 0;
        for (const KflcNode *c = s->children; c; c = c->next) {
            if (kfl_emit_stmt(out, c, ctx, diag, indent + 4)) return 1;
            last_was_return = (c->kind == KFLN_STMT_RETURN);
        }
        if (!last_was_return) b1_emit_frees_current_(out, indent + 4);
        b1_scope_pop_();
        emit_indent(out, indent);
        fputs("}\n", out);
        return 0;
    }
    /* ---- Astro statements ---------------------------------------- *
     * Emitted inside `fn world <name>` bodies. The handle `world`
     * (typed K26AstroWorld * by the fn-world prologue) is in scope.
     * Attribute values were captured at parse time as verbatim
     * expression text (KFLV_IDENT); we splice them straight into
     * the C++ output. Callers must `#include <k26astro_rt/world.h>`
     * + `<k26astro_rt/observer.h>` + `<k26astro_body/body.h>` in
     * the linking translation unit. */
    case KFLN_STMT_ASTRO_BODY: {
        /* assembly= binds a vehicle assembly to the body: the
         * compiler reads the asset, derives the mass properties from
         * the geometry, and writes them here as constants, so nothing
         * opens an asset file while a simulation runs. The binding's
         * rules live in one place (assembly.c), because the
         * environment emitter has its own body emission and the two
         * must not be able to disagree. */
        KflcAssembly *asmb = NULL;
        KflcArena    *asm_arena = kflc_arena_create();
        if (!asm_arena) return 1;
        if (kflc_assembly_for_body(s, diag->path, asm_arena, diag, &asmb)) {
            kflc_arena_release(asm_arena);
            return 1;
        }

        emit_indent(out, indent);
        fputs("{\n", out);
        emit_indent(out, indent + 4);
        fputs("K26AstroBody _kfl_b; k26astro_body_init(&_kfl_b);\n", out);
        emit_indent(out, indent + 4);
        fprintf(out,
            "snprintf(_kfl_b.name, sizeof _kfl_b.name, \"%%s\", \"%s\");\n",
            s->name ? s->name : "_anon");
        if (asmb) {
            char hex[2 * KFLC_ASM_DIGEST + 1];
            kflc_assembly_digest_hex(asmb->digest, hex);
            emit_indent(out, indent + 4);
            fprintf(out, "/* assembly `%s` from %s\n", asmb->name,
                    asmb->path);
            emit_indent(out, indent + 4);
            fprintf(out, " * digest %s */\n", hex);
            emit_indent(out, indent + 4);
            fprintf(out, "k26astro_body_set_mass(&_kfl_b, %.17g);\n",
                    asmb->mass);
            /* The centre of mass and the inertia tensor are derived
             * here and emitted as constants beside the body. The
             * attitude path is what consumes them; they are written
             * now so the numbers in an artifact are the asset's. */
            emit_indent(out, indent + 4);
            fprintf(out, "static const double _kfl_asm_com[3] = "
                    "{ %.17g, %.17g, %.17g };\n",
                    asmb->com[0], asmb->com[1], asmb->com[2]);
            emit_indent(out, indent + 4);
            fprintf(out, "static const double _kfl_asm_inertia[6] = "
                    "{ %.17g, %.17g, %.17g, %.17g, %.17g, %.17g };\n",
                    asmb->inertia[0], asmb->inertia[1], asmb->inertia[2],
                    asmb->inertia[3], asmb->inertia[4], asmb->inertia[5]);
            emit_indent(out, indent + 4);
            fputs("(void)_kfl_asm_com; (void)_kfl_asm_inertia;\n", out);
        }
        for (const KflcAttr *a = s->attrs; a; a = a->next) {
            if (!a->name) continue;
            if (strcmp(a->name, "assembly") == 0) continue;
            /* A plan is read when the program is compiled, like an
             * assembly, and what it contributes lands as constants
             * rather than as a field on the body. */
            if (strcmp(a->name, "reference") == 0) continue;
            const char *val = (a->value.kind == KFLV_IDENT && a->value.u.s)
                              ? a->value.u.s : "0";
            /* parent= is a name string → resolve at runtime via find_body
             * or, if the parent is a parse-time-known body, via the
             * cached idx. */
            if (strcmp(a->name, "parent") == 0) {
                emit_indent(out, indent + 4);
                if (kflc_body_idx_known(ctx, val)) {
                    fprintf(out, "_kfl_b.parent_body_idx = "
                            "_kfl_body_%s_idx;\n", val);
                } else {
                    fprintf(out, "_kfl_b.parent_body_idx = "
                            "k26astro_world_find_body(world, \"%s\");\n",
                            val);
                }
            } else if (kflc_body_state_key_index(a->name) >= 0) {
                /* The scalar state keys map onto compound fields, so
                 * they take the shared write rather than the generic
                 * passthrough below; every emitter that touches body
                 * state uses that one emission. */
                kflc_emit_body_state_write(out, indent + 4, "_kfl_b.",
                                           a->name, val);
            } else {
                emit_indent(out, indent + 4);
                fprintf(out, "_kfl_b.%s = (%s);\n", a->name, val);
            }
        }
        emit_indent(out, indent + 4);
        /* Parse-time-known body name → capture idx in the
         * fn-prologue-declared `_kfl_body_<NAME>_idx` variable so
         * subsequent `propagate`/`observe` calls can deref via
         * `k26astro_world_body_at`. */
        if (s->name && kflc_body_idx_known(ctx, s->name)) {
            fprintf(out, "_kfl_body_%s_idx = "
                    "k26astro_world_add_body(world, _kfl_b);\n",
                    s->name);
        } else {
            fputs("(void)k26astro_world_add_body(world, _kfl_b);\n", out);
        }
        emit_indent(out, indent);
        fputs("}\n", out);
        kflc_arena_release(asm_arena);
        return 0;
    }

    case KFLN_STMT_STEP: {
        emit_indent(out, indent);
        fputs("(void)k26astro_world_step(world, ", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(");\n", out);
        return 0;
    }

    case KFLN_STMT_PROPAGATE: {
        /* Per-body Kepler advance via the dedicated runtime API
         * k26astro_world_body_step (libk26astro_rt). The body's SOI
         * parent (body->parent_body_idx) sets the central body for
         * the Kepler step. Other bodies in the world are NOT touched
         * and the world clock is NOT advanced; that is `step`
         * semantics, not `propagate`.
         *
         * Use the parse-time-known body idx when the name is
         * registered; fall back to runtime find_body otherwise. */
        emit_indent(out, indent);
        fputs("{\n", out);
        emit_indent(out, indent + 4);
        if (s->name && kflc_body_idx_known(ctx, s->name)) {
            fprintf(out, "int _kfl_idx = _kfl_body_%s_idx;\n", s->name);
        } else {
            fprintf(out, "int _kfl_idx = "
                    "k26astro_world_find_body(world, \"%s\");\n",
                    s->name ? s->name : "_anon");
        }
        emit_indent(out, indent + 4);
        fputs("(void)k26astro_world_body_step(world, _kfl_idx, ", out);
        if (s->expr) {
            if (kflc_emit_expr(out, s->expr, ctx, diag)) return 1;
        } else {
            fputs("0", out);
        }
        fputs(");\n", out);
        emit_indent(out, indent);
        fputs("}\n", out);
        return 0;
    }

    case KFLN_STMT_FOR_EACH: {
        /* The iterator binds as a `body` opaque (K26AstroBody *) for
         * the duration of one iteration. add_body() inside the loop
         * body is not realloc-safe; the loop pointer would become
         * stale. The surface treats the iterator as read-only. */
        const char *iter = s->name ? s->name : "_kfl_b";
        emit_indent(out, indent);
        fputs("{\n", out);
        emit_indent(out, indent + 4);
        fputs("int _kfl_n = k26astro_world_body_count(world);\n", out);
        emit_indent(out, indent + 4);
        fputs("for (int _kfl_i = 0; _kfl_i < _kfl_n; _kfl_i++) {\n", out);
        emit_indent(out, indent + 8);
        fprintf(out, "K26AstroBody *%s = "
                     "k26astro_world_body_at(world, _kfl_i);\n", iter);
        emit_indent(out, indent + 8);
        fprintf(out, "(void)%s;\n", iter);
        for (const KflcNode *c = s->children; c; c = c->next) {
            if (kfl_emit_stmt(out, c, ctx, diag, indent + 8)) return 1;
        }
        emit_indent(out, indent + 4);
        fputs("}\n", out);
        emit_indent(out, indent);
        fputs("}\n", out);
        return 0;
    }

    case KFLN_STMT_OBSERVE: {
        /* Resolve target + observer by name, optionally set the
         * world's observer mode from the named-arg, then call
         * k26astro_world_observe. The position + apparent direction
         * are currently discarded; consumers can capture them once a
         * KFL-side observation type is added. */
        const char *observer = "_observer";
        const char *mode_kw  = NULL;
        for (const KflcAttr *a = s->attrs; a; a = a->next) {
            if (!a->name) continue;
            if (strcmp(a->name, "observer") == 0 && a->value.kind == KFLV_IDENT)
                observer = a->value.u.s ? a->value.u.s : observer;
            else if (strcmp(a->name, "mode") == 0 && a->value.kind == KFLV_IDENT)
                mode_kw = a->value.u.s;
        }
        emit_indent(out, indent);
        fputs("{\n", out);
        if (mode_kw) {
            const char *enum_name = "K26ASTRO_OBS_ASTROMETRIC";
            if      (strcmp(mode_kw, "geometric")   == 0) enum_name = "K26ASTRO_OBS_GEOMETRIC";
            else if (strcmp(mode_kw, "astrometric") == 0) enum_name = "K26ASTRO_OBS_ASTROMETRIC";
            else if (strcmp(mode_kw, "apparent")    == 0) enum_name = "K26ASTRO_OBS_APPARENT";
            else if (strcmp(mode_kw, "topocentric") == 0) enum_name = "K26ASTRO_OBS_TOPOCENTRIC";
            emit_indent(out, indent + 4);
            fprintf(out, "(void)k26astro_world_set_observer_mode(world, %s);\n",
                    enum_name);
        }
        /* Route through parse-time-known idx tables when available. */
        emit_indent(out, indent + 4);
        if (s->name && kflc_body_idx_known(ctx, s->name)) {
            fprintf(out, "int _kfl_t = _kfl_body_%s_idx;\n", s->name);
        } else {
            fprintf(out, "int _kfl_t = "
                    "k26astro_world_find_body(world, \"%s\");\n",
                    s->name ? s->name : "_target");
        }
        emit_indent(out, indent + 4);
        if (kflc_body_idx_known(ctx, observer)) {
            fprintf(out, "int _kfl_o = _kfl_body_%s_idx;\n", observer);
        } else {
            fprintf(out, "int _kfl_o = "
                    "k26astro_world_find_body(world, \"%s\");\n",
                    observer);
        }
        emit_indent(out, indent + 4);
        fputs("K26AstroPos _kfl_p;\n", out);
        emit_indent(out, indent + 4);
        fputs("K26V3 _kfl_d;\n", out);
        emit_indent(out, indent + 4);
        fputs("if (_kfl_t >= 0 && _kfl_o >= 0) "
              "(void)k26astro_world_observe(world, _kfl_t, _kfl_o, "
              "&_kfl_p, &_kfl_d);\n", out);
        if (ctx && ctx->headless) {
            /* Headless artifact output: print the observation (range +
             * direction) to stdout instead of feeding a render. The
             * emit-time format carries matched %s args (target,
             * observer); the %% escapes become the runtime %.6e
             * specifiers, so this emit-time fprintf is not the
             * unmatched-specifier musl UB case. */
            emit_indent(out, indent + 4);
            fprintf(out,
                "if (_kfl_t >= 0 && _kfl_o >= 0) fprintf(stdout, "
                "\"observe %s from %s: range=%%.6e m  "
                "dir=(%%.6e, %%.6e, %%.6e)\\n\", "
                "std::sqrt(_kfl_d.x*_kfl_d.x + _kfl_d.y*_kfl_d.y "
                "+ _kfl_d.z*_kfl_d.z), "
                "_kfl_d.x, _kfl_d.y, _kfl_d.z);\n",
                s->name ? s->name : "_target", observer);
            emit_indent(out, indent + 4);
            fputs("(void)_kfl_p;\n", out);
        } else {
            emit_indent(out, indent + 4);
            fputs("(void)_kfl_p; (void)_kfl_d;\n", out);
        }
        emit_indent(out, indent);
        fputs("}\n", out);
        return 0;
    }

    case KFLN_ALLOCATOR_BIND: {
        /* Bind the fn body to the form-level arena named `<s->name>`.
         * The arena's static handle (`_kfl_arena_<name>`) was declared
         * at form scope by the form_has_arena emit sweep; here we
         * install it as the fn-local active arena alias.
         *
         * reset_mode handling: look up the matching arena decl in the
         * form (via ctx->form) and check its `reset_mode` attr:
         *   - fn / fn_exit / unspecified (default): emit with
         *     __attribute__((cleanup(_kfl_arena_reset_cleanup_))) so
         *     the arena auto-resets on every fn return path.
         *   - manual: skip the cleanup attribute; user calls
         *     k26kfl_arena_reset(_kfl_active_arena) explicitly when
         *     they want it.
         *   - form / frame / tick: not yet implemented; treated as
         *     manual with a diagnostic note. */
        const char *reset_mode = "fn_exit";   /* default */
        if (ctx && ctx->form && s->name) {
            for (const KflcNode *c = ctx->form->children; c; c = c->next) {
                if (c->kind != KFLN_ARENA || !c->name) continue;
                if (strcmp(c->name, s->name) != 0) continue;
                for (const KflcAttr *a = c->attrs; a; a = a->next) {
                    if (strcmp(a->name, "reset_mode") != 0) continue;
                    if (a->value.kind == KFLV_IDENT && a->value.u.s) {
                        reset_mode = a->value.u.s;
                    }
                    break;
                }
                break;
            }
        }
        const int use_cleanup =
            (strcmp(reset_mode, "fn") == 0
             || strcmp(reset_mode, "fn_exit") == 0);
        emit_indent(out, indent);
        if (use_cleanup) {
            fprintf(out,
                "K26KflArena *_kfl_active_arena "
                "__attribute__((cleanup(_kfl_arena_reset_cleanup_))) "
                "= _kfl_arena_%s;\n",
                s->name ? s->name : "anon");
        } else {
            fprintf(out,
                "K26KflArena *_kfl_active_arena = _kfl_arena_%s;\n",
                s->name ? s->name : "anon");
            if (strcmp(reset_mode, "manual") != 0) {
                emit_indent(out, indent);
                fprintf(out,
                    "/* unhandled reset_mode `%s`; defaulting to manual */\n",
                    reset_mode);
            }
        }
        emit_indent(out, indent);
        fputs("(void)_kfl_active_arena;\n", out);
        return 0;
    }

    case KFLN_STMT_EPISODE:
    case KFLN_STMT_EPISODE_RESET:
    case KFLN_STMT_ACTION:
    case KFLN_STMT_ON_STEP:
    case KFLN_STMT_OBJECTIVE:
    case KFLN_STMT_AGENT:
        /* The reinforcement learning constructs are emitted by the
         * environment emitter (emit_rl.c), which a form using them is
         * routed to before this dispatch can see them. Reaching this
         * arm means the routing failed. */
        kflc_diag_errorf(diag, s->line,
            "internal error: reinforcement learning construct outside "
            "the environment emitter");
        return 1;

    default:
        kflc_diag_errorf(diag, s->line,
                         "emit_stmt: unexpected node kind %d", s->kind);
        return 1;
    }
}
