/* emit_rl.c - the reinforcement learning emitter's entry:
 * one call, two artifacts from one emitted translation
 * unit. The pipeline lives in the emit_rl_*.c modules and
 * their shared spine emit_rl_internal.h. */
#include "emit_rl_internal.h"

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
    if (rl_collect(&m, form, arena, diag)) {
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
    rl_collect_form_args(form, arena, &arg_live, &arg_n, &arg_cap);
    KflcExprCtx arg_ctx;
    memset(&arg_ctx, 0, sizeof arg_ctx);
    arg_ctx.bindings   = arg_live;
    arg_ctx.n_bindings = arg_n;
    arg_ctx.fns        = user_fn_arr;
    arg_ctx.n_fns      = n_user_fns;
    arg_ctx.form       = form;

    if (rl_emit_prologue(out, &m, form, diag)) return 1;
    /* The mass-property tables come before the world is built, not
     * with the actuators: the world prefix installs a vehicle's
     * constructed properties through them, and the actuator block is
     * emitted from inside the per-step body further down. */
    rl_emit_mass_tables(out, &m);
    rl_emit_reference_tables(out, &m);
    rl_emit_plan_tables(out, &m);
    rl_emit_form_args(out, form);
    if (rl_emit_user_fns(out, form, arena, user_fn_arr, n_user_fns,
                          diag) ||
        rl_emit_params(out, &m, &arg_ctx, diag) ||
        rl_emit_build_world(out, &m, form, arena, user_fn_arr,
                             n_user_fns, diag) ||
        rl_emit_apply_draws(out, &m, &arg_ctx, diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    /* The datalink's tables and the information state's gates come
     * after the world prefix, because a gate is the detection block
     * and that block reads the body-index table the prefix declares,
     * and before the observation function, because the environment
     * core beneath both walks the tables. */
    if (rl_emit_link_tables(out, &m, diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    if (rl_emit_observe(out, &m, diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    if (rl_emit_on_step(out, &m, form, arena, user_fn_arr, n_user_fns,
                         diag) ||
        rl_emit_objective(out, &m, form, arena, user_fn_arr, n_user_fns,
                           diag)) {
        kflc_arena_release(arena);
        return 1;
    }
    rl_emit_env_core(out);
    rl_emit_batch_main(out, form);

    kflc_arena_release(arena);
    return diag->errors ? 1 : 0;
}
