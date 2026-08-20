/* emit_rl_core.c - the environment core the artifact carries verbatim, and
 * the batch entry. */
#include "emit_rl_internal.h"
#include "emitted/env_core_inc.h"

/* The environment handle and the frozen surface. This block is
 * program-independent apart from the geometry macros, so it is
 * emitted as one literal. */

void rl_emit_env_core(FILE *out)
{
    /* The core is carried verbatim from its template file, where it
     * reads as the C++ it is; the build turns the template into the
     * byte array included above, and author notes marked #@ in the
     * template stay with the code they describe without being
     * emitted. */
    fputs(kflrl_tpl_env_core, out);
}

/* The batch entry: the episode machinery at declared default actions,
 * stepping until every environment has recorded at least K completed
 * episodes; early finishers keep stepping and record their extras. */

void rl_emit_batch_main(FILE *out, const KflcNode *form)
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
