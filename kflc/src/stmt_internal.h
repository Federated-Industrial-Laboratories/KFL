/* stmt_internal.h - the statement layer's shared spine: the node
 * plumbing, the type helpers, the scope state, and the functions
 * that cross a module line. stmt.c parses, stmt_rl.c parses the
 * Grammar 3.2 constructs, stmt_emit.c emits.
 */
#ifndef KFLC_STMT_INTERNAL_H
#define KFLC_STMT_INTERNAL_H

/* kflc , function-body parser + emitter.
 *
 * Inline KFL functions hold a body of statements: let / const / assign /
 * return / expression / if / while. This module parses them off the
 * main line-oriented token stream, and (separately) walks the parsed
 * AST and emits the corresponding C++ statements.
 *
 * Statement grammar (newline-terminated, blocks closed with `end`):
 *
 *   let_stmt    = "let"   IDENT ":" TYPE "=" EXPR
 *   const_stmt  = "const" IDENT ":" TYPE "=" EXPR
 *   assign_stmt = IDENT "=" EXPR
 *   return_stmt = "return" [EXPR]
 *   expr_stmt   = EXPR
 *   if_stmt     = "if" EXPR NEWLINE
 *                   stmt*
 *                 [ "else" NEWLINE stmt* ]
 *                 "end"
 *   while_stmt  = "while" EXPR NEWLINE
 *                   stmt*
 *                 "end"
 *
 * EXPR / TYPE re-use the shared expression sub-language (expr.c) and
 * the KflcType enum from kflc.h.
 *
 * The parser is given the same Token-stream `Lexer` used by parser.c.
 * It reads a single physical line of source as an attribute-style value
 * stream, then takes the *rest* of that logical line (everything after
 * the keyword that introduced the statement) and hands it to the
 * expression parser. This sidesteps having to grow the main lexer with
 * operator tokens.
 */

#include "kflc.h"

#include "internal.h"

#include "assembly.h"

/* ---- Block-scope tracker for heap-typed lets --------------------- */

#include <string.h>   /* memcpy for scope-array growth */

/* Per-scope record: heap-typed `let` names declared in this scope,
 * with their types so we know whether to emit `k26c_vec_free` or
 * `k26c_mat_free`. Storage is arena-backed (lifetime = the per-compile
 * temp arena set by kfl_emit_stmt_reset_scopes). */

typedef struct {
    const char **names;
    KflcType    *types;
    int          n;
    int          cap;
} BlockScope;

enum { B1_MAX_SCOPE_DEPTH = 16 };


#include <ctype.h>

#include <stdio.h>

#include <stdlib.h>

#include <string.h>

/* ---- Statement parser -------------------------------------------- */

/* Forward decls. The fn-body parser is mutually recursive: an if/while
 * statement contains a body of further statements. */

KflcNode *kfl_parse_stmt_block(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error,
                                const char *terminator,
                                const char **also_break);

/* ---- Statement emitter ------------------------------------------ */

/* Emit a statement subtree to `out`. Indent is in spaces. Returns 0
 * on success, nonzero on emit error. */

int kfl_emit_stmt(FILE *out, const KflcNode *s,
                   const KflcExprCtx *ctx, KflcDiag *diag,
                   int indent);

extern BlockScope g_b1_scopes[B1_MAX_SCOPE_DEPTH];
extern int        g_b1_depth;
extern KflcType   g_b1_fn_return_type;
extern int g_rl_world_ctx;
extern int g_rl_on_step_depth;

void kfl_stmt_append_child(KflcNode *parent, KflcNode *child);
int kfl_stmt_at_nl  (const Token *t);
int kfl_stmt_is_ident_named(const Token *t, const char *name);
KflcNode *kfl_parse_stmt_block(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error,
                                const char *terminator,
                                const char **also_break);
const char *kflc_type_cxx(KflcType t, const char *opaque_name);
KflcNode *kfl_stmt_new_node(KflcArena *arena, KflcNodeKind k, int line);
KflcNode *kfl_stmt_parse_stmt(Lexer *L, Token *cur,
                             KflcArena *arena, KflcDiag *diag,
                             int *had_error);
int kfl_stmt_peek_kind(Lexer *L, TokenKind *out);
void kfl_stmt_skip_newlines(Lexer *L, Token *cur, int *had_error);
KflcAttr *kfl_stmt_append_attr(KflcArena *arena, KflcNode *n,
                                   const char *key, KflcValue val, int line);
const KflcAttr *kfl_stmt_find_attr(const KflcNode *n, const char *key);
char *kfl_stmt_take_line_remainder(Lexer *L, KflcArena *arena);
char *kfl_stmt_trim(char *s);
int kfl_stmt_is_on_step_world_stmt(const char *s);
KflcNode *kfl_stmt_parse_action(Lexer *L, Token *cur,
                               KflcArena *arena, KflcDiag *diag,
                               int *had_error);
KflcNode *kfl_stmt_parse_agent(Lexer *L, Token *cur,
                              KflcArena *arena, KflcDiag *diag,
                              int *had_error);
KflcNode *kfl_stmt_parse_astro_payload(Lexer *L, Token *cur,
                                      KflcArena *arena, KflcDiag *diag,
                                      int *had_error);
KflcNode *kfl_stmt_parse_engage(Lexer *L, Token *cur,
                               KflcArena *arena, KflcDiag *diag,
                               int *had_error);
KflcNode *kfl_stmt_parse_episode(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error);
KflcNode *kfl_stmt_parse_objective(Lexer *L, Token *cur,
                                  KflcArena *arena, KflcDiag *diag,
                                  int *had_error);
KflcNode *kfl_stmt_parse_on_step(Lexer *L, Token *cur,
                                KflcArena *arena, KflcDiag *diag,
                                int *had_error);
KflcNode *kfl_stmt_parse_plan(Lexer *L, Token *cur,
                             KflcArena *arena, KflcDiag *diag,
                             int *had_error);
KflcNode *kfl_stmt_parse_sensor(Lexer *L, Token *cur,
                               KflcArena *arena, KflcDiag *diag,
                               int *had_error);
KflcNode *kfl_stmt_parse_capture_envelope(Lexer *L, Token *cur,
                                         KflcArena *arena, KflcDiag *diag,
                                         int *had_error);
void kfl_stmt_rl_drain_line(Lexer *L, Token *cur,
                           KflcArena *arena, int *had_error);
int kfl_stmt_rl_word_is_construct(Lexer *L, const char *s);
void kfl_stmt_advance(Lexer *L, Token *cur, int *had_error);
int kfl_stmt_at_eof2(const Token *t);

#endif /* KFLC_STMT_INTERNAL_H */
