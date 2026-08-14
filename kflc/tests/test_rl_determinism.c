/* test_rl_determinism.c: determinism and equivalence gates for the
 * Grammar 3.2 dual-mode artifact.
 *
 * Gates:
 *   1. Two-process batch determinism: the same compiled batch binary
 *      run twice in separate processes with identical seed, envs, and
 *      episodes produces byte-identical episode files.
 *   2. Two-session serve determinism: two separate dlopen sessions
 *      (distinct copies of the shared object, so no loader-level
 *      state is shared) driven with the same seed and the same
 *      scripted action stream produce bitwise-identical observation,
 *      reward, and flag streams.
 *   3. Vector independence, batch: environment 0's recorded episodes
 *      are bitwise identical between an --envs 1 run and an --envs 64
 *      run at the same seed, across every recorded stream (initial
 *      observation, observations, actions, rewards, flags, applied
 *      dt, randomisation draws).
 *   4. Vector independence, serve: environment 0's observation,
 *      reward, and flag streams are bitwise unchanged when 63
 *      neighbours run beside it and are driven with different
 *      actions.
 *   5. Spec round trip: parse the TLV blob, verify every declared
 *      total, rebuild the drive buffers from the parsed geometry (the
 *      serve gates above size their buffers from the parsed spec),
 *      and re-parse after splicing an unknown tag into the blob: the
 *      unknown tag is skipped by length and every recovered value is
 *      unchanged.
 *   6. Batch/serve equivalence as whole files: driving the serve
 *      surface with the batch stopping rule and the declared default
 *      actions writes a byte-identical episode file. The two entries
 *      share the stepping machinery by construction, so this gate
 *      exercises the entry code: argument handling, the stopping
 *      rule, and output enabling.
 *   7. Two-process serve determinism: the same serve drive run as
 *      two separate processes (this binary re-executed in driver
 *      mode), each writing its observation, reward, and flag
 *      streams to a file, produces bitwise-identical files; a third
 *      in-process drive matches the same bytes. Gate 2 only ever
 *      compared serve sessions inside one process.
 *
 * The fixture's reward reads a world scalar binding and an action
 * channel, so the reward stream witnesses both the world-binding
 * capture and the action stream; its observation channels witness
 * the dynamics. Actions do not feed back into the dynamics here,
 * which the fault and re-simulation gates do not need either; the
 * streams they shape (reward) are compared bitwise all the same.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_det_test"

/* Horizon 24 with no termination predicate: every episode truncates
 * after exactly 24 transitions, so runs of different --envs finish in
 * lockstep and each environment records exactly the requested number
 * of episodes. */
static const char *const DET_KFL =
    "form RL_DET\n"
    "fn world det_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0,10.0) vel_z=0.0\n"
    "    let range_scale: double = 7.0e6\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 24\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 10.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    action gear discrete 3 default 1\n"
    "    on_step\n"
    "        let scale: double = 1.0 + push * 0.001\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range / range_scale + push\n"
    "    end\n"
    "end\n"
    "end\n";

/* The fixture's declared action defaults, used by the batch entry and
 * by the serve replica in gate 6. */
static const double DET_DEFAULTS_[2] = { 0.25, 1.0 };

/* Scripted action streams, pure functions of (step, environment,
 * channel) so two runs compute identical inputs. The alt stream is
 * everywhere different from the base stream's neighbour values. */
static double act_base_(uint32_t t, uint32_t e, uint32_t ch)
{
    if (ch == 0) {
        return ((double)((t * 7u + e * 3u) % 13u)) / 13.0 * 2.0 - 1.0;
    }
    return (double)((t + e) % 3u);
}

static double act_alt_(uint32_t t, uint32_t e, uint32_t ch)
{
    if (ch == 0) {
        return ((double)((t * 5u + e * 11u) % 7u)) / 7.0;
    }
    return (double)((t + 2u * e + 1u) % 3u);
}

/* Drive one dlopen session for T steps and record every environment's
 * observation, reward, and flag streams. Environment 0 always takes
 * the base action stream; neighbours take the base stream or, with
 * alt_neighbors, the alternative one. Buffer strides come from the
 * artifact's own parsed spec (gate 5's geometry rebuild). Layout:
 * obs_out[t * n_envs * obs_total ...], rew_out[t * n_envs + e],
 * fl_out[t * n_envs + e]. */
static void drive_(const char *so_path, uint64_t seed, uint32_t n_envs,
                   uint32_t T, int alt_neighbors,
                   double *obs_out, double *rew_out, uint32_t *fl_out)
{
    void *so = rl_dlopen_(so_path);
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    K26RlEnv *env = NULL;
    ASSERT(s.create(seed, n_envs, &env) == K26RL_OK);

    int32_t spec_len = s.spec(env, NULL, 0);
    ASSERT(spec_len > 0);
    uint8_t *blob = malloc((size_t)spec_len);
    ASSERT(blob != NULL);
    ASSERT(s.spec(env, blob, (uint32_t)spec_len) == spec_len);
    RlSpecView v;
    rl_parse_spec_(blob, (uint32_t)spec_len, &v);
    free(blob);
    ASSERT(v.n_envs == n_envs);
    const uint32_t obs_total = v.obs_total;
    const uint32_t act_total = v.act_total;
    ASSERT(obs_total == 5 && act_total == 2);

    double *act = malloc(sizeof(double) * (size_t)n_envs * act_total);
    ASSERT(act != NULL);

    for (uint32_t t = 0; t < T; t++) {
        for (uint32_t e = 0; e < n_envs; e++) {
            for (uint32_t ch = 0; ch < act_total; ch++) {
                double a = (e > 0 && alt_neighbors)
                    ? act_alt_(t, e, ch) : act_base_(t, e, ch);
                act[(size_t)e * act_total + ch] = a;
            }
        }
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, obs_out + (size_t)t * n_envs * obs_total)
               == K26RL_OK);
        ASSERT(s.reward(env, rew_out + (size_t)t * n_envs) == K26RL_OK);
        ASSERT(s.flags(env, fl_out + (size_t)t * n_envs) == K26RL_OK);
    }

    free(act);
    s.destroy(env);
    dlclose(so);
}

/* Serve-driver subprocess entry (gate 7): drive one session with the
 * deterministic base action stream and write the raw observation,
 * reward, and flag streams to out_path. The parent launches two of
 * these as separate processes and compares the files bitwise. The
 * observation width is the fixture's (5, pinned by gate 5's spec
 * walk). */
static int serve_driver_main_(const char *so_path, const char *seed_s,
                              const char *T_s, const char *n_s,
                              const char *out_path)
{
    uint64_t seed = strtoull(seed_s, NULL, 10);
    uint32_t T    = (uint32_t)strtoul(T_s, NULL, 10);
    uint32_t N    = (uint32_t)strtoul(n_s, NULL, 10);
    const uint32_t OBS = 5;
    ASSERT(T > 0 && N > 0);

    double   *o = malloc(sizeof(double)   * (size_t)T * N * OBS);
    double   *r = malloc(sizeof(double)   * (size_t)T * N);
    uint32_t *f = malloc(sizeof(uint32_t) * (size_t)T * N);
    ASSERT(o != NULL && r != NULL && f != NULL);
    drive_(so_path, seed, N, T, 0, o, r, f);

    FILE *out = fopen(out_path, "wb");
    ASSERT(out != NULL);
    ASSERT(fwrite(o, sizeof(double), (size_t)T * N * OBS, out)
           == (size_t)T * N * OBS);
    ASSERT(fwrite(r, sizeof(double), (size_t)T * N, out)
           == (size_t)T * N);
    ASSERT(fwrite(f, sizeof(uint32_t), (size_t)T * N, out)
           == (size_t)T * N);
    ASSERT(fclose(out) == 0);
    free(o);
    free(r);
    free(f);
    return 0;
}

/* Bitwise comparison of two decoded episodes across every recorded
 * stream. */
static void episodes_equal_(const K26RlEpisodeData *a,
                            const K26RlEpisodeData *b,
                            uint32_t obs_total, uint32_t act_total,
                            uint32_t agent_count)
{
    ASSERT(a->step_count == b->step_count);
    ASSERT(a->end_reason == b->end_reason);
    ASSERT(a->fault_code == b->fault_code);
    ASSERT(memcmp(a->initial_obs, b->initial_obs,
                  sizeof(double) * obs_total) == 0);
    ASSERT(a->dr_count == b->dr_count);
    ASSERT(memcmp(a->dr_tags, b->dr_tags,
                  sizeof(uint32_t) * a->dr_count) == 0);
    ASSERT(memcmp(a->dr_values, b->dr_values,
                  sizeof(double) * a->dr_count) == 0);
    size_t n = a->step_count;
    ASSERT(memcmp(a->obs, b->obs, sizeof(double) * n * obs_total) == 0);
    ASSERT(memcmp(a->act, b->act, sizeof(double) * n * act_total) == 0);
    ASSERT(memcmp(a->rewards, b->rewards,
                  sizeof(double) * n * agent_count) == 0);
    ASSERT(memcmp(a->flags, b->flags, sizeof(uint32_t) * n) == 0);
    ASSERT(memcmp(a->applied_dt, b->applied_dt, sizeof(double) * n) == 0);
    ASSERT(memcmp(a->terminal_adjustments, b->terminal_adjustments,
                  sizeof(double) * agent_count) == 0);
}

int main(int argc, char **argv)
{
    /* Subprocess driver mode for gate 7. Launched by this same
     * binary; the fixture's shared object already exists then. */
    if (argc == 7 && strcmp(argv[1], "--serve-driver") == 0) {
        return serve_driver_main_(argv[2], argv[3], argv[4], argv[5],
                                  argv[6]);
    }

    if (!rl_libs_present_("test_rl_determinism")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/det.kfl", DET_KFL);
    rl_compile_(WORK_DIR "/det.kfl", WORK_DIR "/det", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/det"));
    ASSERT(rl_file_exists_(WORK_DIR "/det.rlenv.so"));

    /* Gate 1: two processes, memcmp of the episode files. */
    rl_run_or_die_(WORK_DIR "/det --envs 2 --episodes 2 --seed 42"
                   " --out " WORK_DIR "/run_a.k26epi > /dev/null");
    rl_run_or_die_(WORK_DIR "/det --envs 2 --episodes 2 --seed 42"
                   " --out " WORK_DIR "/run_b.k26epi > /dev/null");
    ASSERT(rl_files_equal_(WORK_DIR "/run_a.k26epi",
                           WORK_DIR "/run_b.k26epi"));
    printf("gate 1: two-process batch determinism (memcmp): OK\n");

    /* Gate 2: two dlopen sessions on distinct copies of the shared
     * object, same seed and scripted action stream, bitwise-identical
     * streams. T=60 crosses two truncation boundaries, so the
     * comparison covers boundary resets too. */
    {
        enum { T = 60, N = 4, OBS = 5 };
        rl_run_or_die_("cp " WORK_DIR "/det.rlenv.so "
                       WORK_DIR "/det_s1.so && cp " WORK_DIR
                       "/det.rlenv.so " WORK_DIR "/det_s2.so");
        static double o1[T * N * OBS], o2[T * N * OBS];
        static double r1[T * N], r2[T * N];
        static uint32_t f1[T * N], f2[T * N];
        drive_(WORK_DIR "/det_s1.so", 42, N, T, 0, o1, r1, f1);
        drive_(WORK_DIR "/det_s2.so", 42, N, T, 0, o2, r2, f2);
        ASSERT(memcmp(o1, o2, sizeof o1) == 0);
        ASSERT(memcmp(r1, r2, sizeof r1) == 0);
        ASSERT(memcmp(f1, f2, sizeof f1) == 0);
    }
    printf("gate 2: two-session serve determinism over 60 steps"
           " x 4 environments: OK\n");

    /* Gate 3: vector independence, batch. Environment 0's episodes at
     * --envs 1 equal environment 0's episodes at --envs 64, stream by
     * stream. */
    {
        rl_run_or_die_(WORK_DIR "/det --envs 1 --episodes 2 --seed 7"
                       " --out " WORK_DIR "/v1.k26epi > /dev/null");
        rl_run_or_die_(WORK_DIR "/det --envs 64 --episodes 2 --seed 7"
                       " --out " WORK_DIR "/v64.k26epi > /dev/null");
        K26RlEpisodeReader *ra = NULL, *rb = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/v1.k26epi", &ra)
               == K26RL_OK);
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/v64.k26epi", &rb)
               == K26RL_OK);
        K26RlEpisodeInfo ia, ib;
        ASSERT(k26rl_episode_reader_info(ra, &ia) == K26RL_OK);
        ASSERT(k26rl_episode_reader_info(rb, &ib) == K26RL_OK);
        ASSERT(ia.n_envs == 1 && ib.n_envs == 64);
        ASSERT(ia.obs_total == ib.obs_total);
        ASSERT(ia.act_total == ib.act_total);
        ASSERT(ia.agent_count == ib.agent_count);
        for (uint32_t k = 0; k < 2; k++) {
            K26RlEpisodeData ea, eb;
            ASSERT(k26rl_episode_read(ra, 0, 0, k, &ea) == K26RL_OK);
            ASSERT(k26rl_episode_read(rb, 0, 0, k, &eb) == K26RL_OK);
            ASSERT(ea.end_reason == K26RL_END_TRUNCATED);
            ASSERT(ea.step_count == 24);
            episodes_equal_(&ea, &eb, ia.obs_total, ia.act_total,
                            ia.agent_count);
            k26rl_episode_free(&ea);
            k26rl_episode_free(&eb);
        }
        /* The reward stream witnesses the captured world scalar: at
         * default actions it is range/7.0e6 + 0.25 exactly. */
        {
            K26RlEpisodeData ea;
            ASSERT(k26rl_episode_read(ra, 0, 0, 0, &ea) == K26RL_OK);
            for (uint32_t t = 0; t < ea.step_count; t++) {
                double want = ea.obs[(size_t)t * ia.obs_total + 3]
                              / 7.0e6 + 0.25;
                ASSERT(memcmp(&ea.rewards[t], &want, sizeof want) == 0);
            }
            k26rl_episode_free(&ea);
        }
        k26rl_episode_reader_close(ra);
        k26rl_episode_reader_close(rb);
    }
    printf("gate 3: vector independence, batch (1 vs 64 environments,"
           " all streams): OK\n");

    /* Gate 4: vector independence, serve, neighbours driven with
     * different actions. */
    {
        enum { T = 60, N = 64, OBS = 5 };
        static double o1[T * 1 * OBS], oN[T * N * OBS];
        static double r1[T * 1], rN[T * N];
        static uint32_t f1[T * 1], fN[T * N];
        drive_(WORK_DIR "/det.rlenv.so", 9, 1, T, 0, o1, r1, f1);
        drive_(WORK_DIR "/det.rlenv.so", 9, N, T, 1, oN, rN, fN);
        for (uint32_t t = 0; t < T; t++) {
            ASSERT(memcmp(o1 + (size_t)t * OBS,
                          oN + (size_t)t * N * OBS,
                          sizeof(double) * OBS) == 0);
            ASSERT(memcmp(&r1[t], &rN[(size_t)t * N],
                          sizeof(double)) == 0);
            ASSERT(f1[t] == fN[(size_t)t * N]);
        }
    }
    printf("gate 4: vector independence, serve (1 vs 64, different"
           " neighbour actions): OK\n");

    /* Gate 5: spec round trip with unknown-tag injection. The drive
     * runs above already rebuilt their buffer geometry from the
     * parsed blob; here the declared totals are pinned and an unknown
     * tag is spliced in after the first entry. */
    {
        void *so = rl_dlopen_(WORK_DIR "/det.rlenv.so");
        RlSurface s;
        rl_resolve_surface_(so, &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(3, 2, &env) == K26RL_OK);
        int32_t len = s.spec(env, NULL, 0);
        ASSERT(len > 0);
        uint8_t *blob = malloc((size_t)len);
        ASSERT(blob != NULL);
        ASSERT(s.spec(env, blob, (uint32_t)len) == len);

        RlSpecView v;
        rl_parse_spec_(blob, (uint32_t)len, &v);
        ASSERT(v.abi_version == K26RL_ABI_VERSION);
        ASSERT(v.endian_probe == 0x01020304u);
        ASSERT(v.agent_count == 1);
        ASSERT(v.n_envs == 2);
        ASSERT(v.obs_total == 5);
        ASSERT(v.act_total == 2);
        ASSERT(v.horizon == 24);
        {
            double dt;
            uint64_t bits = v.control_dt_bits;
            memcpy(&dt, &bits, sizeof dt);
            ASSERT(dt == 0.1);
        }
        ASSERT(v.saw_bounds && v.saw_kind && v.saw_names);
        ASSERT(v.episode_flags & 1u);

        /* Splice an unknown tag (0x7F00, 5 payload bytes) after the
         * first TLV entry; a conforming walk skips it by length and
         * recovers every value unchanged. */
        uint32_t first = 6u + rl_get_u32_(blob + 2);
        uint32_t len2 = (uint32_t)len + 6u + 5u;
        uint8_t *blob2 = malloc(len2);
        ASSERT(blob2 != NULL);
        memcpy(blob2, blob, first);
        blob2[first] = 0x00; blob2[first + 1] = 0x7F;   /* tag */
        blob2[first + 2] = 5; blob2[first + 3] = 0;     /* length */
        blob2[first + 4] = 0; blob2[first + 5] = 0;
        memcpy(blob2 + first + 6, "\x01\x02\x03\x04\x05", 5);
        memcpy(blob2 + first + 11, blob + first, (size_t)len - first);
        RlSpecView v2;
        rl_parse_spec_(blob2, len2, &v2);
        ASSERT(memcmp(&v, &v2, sizeof v) == 0);
        free(blob2);
        free(blob);
        s.destroy(env);
        dlclose(so);
    }
    printf("gate 5: spec round trip + unknown-tag injection: OK\n");

    /* Gate 6: batch/serve equivalence as whole files. The serve side
     * drives the batch stopping rule (step until every environment
     * has completed at least K episodes) at the declared default
     * actions and writes through the same enable call. */
    {
        rl_run_or_die_(WORK_DIR "/det --envs 3 --episodes 2 --seed 11"
                       " --out " WORK_DIR "/eq_batch.k26epi > /dev/null");

        void *so = rl_dlopen_(WORK_DIR "/det.rlenv.so");
        RlSurface s;
        rl_resolve_surface_(so, &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(11, 3, &env) == K26RL_OK);
        ASSERT(s.output(env, WORK_DIR "/eq_serve.k26epi") == K26RL_OK);

        double act[3 * 2];
        for (int e = 0; e < 3; e++) {
            act[e * 2 + 0] = DET_DEFAULTS_[0];
            act[e * 2 + 1] = DET_DEFAULTS_[1];
        }
        unsigned long long done[3] = { 0, 0, 0 };
        uint32_t fl[3];
        for (;;) {
            int all_done = 1;
            for (int e = 0; e < 3; e++) {
                if (done[e] < 2) all_done = 0;
            }
            if (all_done) break;
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.flags(env, fl) == K26RL_OK);
            for (int e = 0; e < 3; e++) {
                if (fl[e] & (K26RL_FLAG_TERMINATED | K26RL_FLAG_TRUNCATED
                             | K26RL_FLAG_FAULT)) {
                    done[e]++;
                }
            }
        }
        s.destroy(env);
        dlclose(so);

        ASSERT(rl_files_equal_(WORK_DIR "/eq_batch.k26epi",
                               WORK_DIR "/eq_serve.k26epi"));
    }
    printf("gate 6: batch/serve equivalence, whole-file memcmp: OK\n");

    /* Gate 7: two-process serve determinism. Two separate processes
     * (fresh address spaces, fresh loader state) run the same serve
     * drive and write their streams to files; the files must match
     * bitwise, and an in-process drive of the same script must
     * match the same bytes. This closes the note that serve
     * determinism was only ever checked within one process. */
    {
        enum { T = 60, N = 4, OBS = 5 };
        char cmd[1024];
        int n = snprintf(cmd, sizeof cmd,
                         "%s --serve-driver " WORK_DIR "/det.rlenv.so"
                         " 42 %d %d " WORK_DIR "/serve_p1.bin",
                         argv[0], T, N);
        ASSERT(n > 0 && (size_t)n < sizeof cmd);
        rl_run_or_die_(cmd);
        n = snprintf(cmd, sizeof cmd,
                     "%s --serve-driver " WORK_DIR "/det.rlenv.so"
                     " 42 %d %d " WORK_DIR "/serve_p2.bin",
                     argv[0], T, N);
        ASSERT(n > 0 && (size_t)n < sizeof cmd);
        rl_run_or_die_(cmd);
        ASSERT(rl_files_equal_(WORK_DIR "/serve_p1.bin",
                               WORK_DIR "/serve_p2.bin"));

        /* Third leg: in-process drive against the subprocess bytes. */
        static double   o[T * N * OBS], of[T * N * OBS];
        static double   r[T * N], rf[T * N];
        static uint32_t f[T * N], ff[T * N];
        drive_(WORK_DIR "/det.rlenv.so", 42, N, T, 0, o, r, f);
        FILE *fp = fopen(WORK_DIR "/serve_p1.bin", "rb");
        ASSERT(fp != NULL);
        ASSERT(fread(of, sizeof(double), T * N * OBS, fp)
               == (size_t)T * N * OBS);
        ASSERT(fread(rf, sizeof(double), T * N, fp) == (size_t)T * N);
        ASSERT(fread(ff, sizeof(uint32_t), T * N, fp) == (size_t)T * N);
        ASSERT(fgetc(fp) == EOF);   /* stream file has no tail */
        fclose(fp);
        ASSERT(memcmp(o, of, sizeof o) == 0);
        ASSERT(memcmp(r, rf, sizeof r) == 0);
        ASSERT(memcmp(f, ff, sizeof f) == 0);
    }
    printf("gate 7: two-process serve determinism, stream files"
           " bitwise: OK\n");

    printf("test_rl_determinism: 7 gates passed\n");
    return 0;
}
