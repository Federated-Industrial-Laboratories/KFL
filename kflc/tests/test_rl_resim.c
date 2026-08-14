/* test_rl_resim.c: re-simulation gate for the Grammar 3.2 dual-mode
 * artifact and its episode files.
 *
 * Every indexed episode of a recorded file is rebuilt from the file
 * alone, addressed directly by its identity triple (rekey ordinal,
 * env index, episode index): the ordinal resolves to its recorded
 * governing seed, a fresh environment handle is created with that
 * seed, the episode index is reached through the explicit reset call
 * (never by stepping through preceding episodes under auto-reset),
 * and the recorded action stream is replayed. The episode-start
 * frame's initial observation and the per-step observation, reward,
 * and flag streams must match bitwise.
 *
 * Reaching the initial observation without any prior stepping pins
 * the reset boundary's zero time advance: the recording produced
 * episode k's start after stepping through episode k-1, the rebuild
 * produces it from resets alone, and the two must be bit-equal.
 *
 * Coverage: terminated, truncated, and externally cut episodes with
 * per-step scripted actions and per-episode reset draws (fixture 1,
 * recorded through the serve surface across a mid-run reseed, so
 * ordinal 1 episodes are rebuilt from the rekey frame's seed);
 * faulted episodes whose final fault record replays the faulting
 * call's actions and reproduces the fault (fixture 2, recorded by
 * the batch executable). Both fixtures write body state in on_step,
 * so what replays includes the writes the actions drove.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_resim_test"

/* Episodes whose ending depends on the per-episode draw: a start
 * above the termination threshold terminates at transition 1, the
 * rest truncate at the horizon. The reward reads the action channel,
 * so the replayed action stream shapes the compared reward stream. */
static const char *const RESIM_KFL =
    "form RL_RESIM\n"
    "fn world resim_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=7350.0 vel_z=0.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 5\n"
    "        terminated when trk_range > 7.05e6\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 5.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.1\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward trk_range * 1.0e-6 + push\n"
    "    end\n"
    "end\n"
    "end\n";

/* reward 1.0 / (3.0 - episode.steps): every episode faults at
 * transition 3, so the batch file records fault records to rebuild.
 * The block writes body state every step, so the episodes recorded
 * after a fault also pin the discard rule: a write made on the
 * faulting step must reach neither the next episode's opening state
 * nor its streams, or the rebuild from resets alone would differ. */
static const char *const RESIM_FAULT_KFL =
    "form RL_RESIM_FAULT\n"
    "fn world resim_fault_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 6\n"
    "        reset craft.vel_y normal(7350.0, 5.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push + 5.0\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0 / (3.0 - episode.steps)\n"
    "    end\n"
    "end\n"
    "end\n";

/* Scripted recording actions, a pure function of (step, env). */
static double act_r1_(uint32_t t, uint32_t e)
{
    return ((double)((t * 3u + e * 5u) % 11u)) / 11.0 * 2.0 - 1.0;
}

/* Step the recording session until every environment has ended at
 * least `want` episodes since the counters were zeroed, feeding the
 * scripted stream; asserts progress within a step budget. */
static void record_until_(const RlSurface *s, K26RlEnv *env,
                          uint32_t n_envs, uint32_t want,
                          uint32_t *t_io)
{
    uint32_t ends[8] = { 0 };
    double act[8];
    uint32_t fl[8];
    ASSERT(n_envs <= 8);
    for (uint32_t budget = 0; budget < 200; budget++) {
        int all = 1;
        for (uint32_t e = 0; e < n_envs; e++) {
            if (ends[e] < want) all = 0;
        }
        if (all) return;
        for (uint32_t e = 0; e < n_envs; e++) {
            act[e] = act_r1_(*t_io, e);
        }
        (*t_io)++;
        ASSERT(s->step(env, act) == K26RL_OK);
        ASSERT(s->flags(env, fl) == K26RL_OK);
        for (uint32_t e = 0; e < n_envs; e++) {
            if (fl[e] & (K26RL_FLAG_TERMINATED | K26RL_FLAG_TRUNCATED
                         | K26RL_FLAG_FAULT)) {
                ends[e]++;
            }
        }
    }
    ASSERT(0 && "recording made no progress");
}

/* Rebuild one indexed episode from the file alone and compare every
 * stream bitwise. Returns the episode's end reason. */
static uint16_t resim_one_(K26RlEpisodeReader *rd, const char *so_path,
                           uint32_t ord, uint32_t e, uint32_t epi)
{
    K26RlEpisodeInfo info;
    ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
    uint64_t seed = 0;
    ASSERT(k26rl_episode_reader_seed(rd, ord, &seed) == K26RL_OK);
    K26RlEpisodeData ep;
    ASSERT(k26rl_episode_read(rd, ord, e, epi, &ep) == K26RL_OK);

    const uint32_t n = info.n_envs;
    const uint32_t obs_total = info.obs_total;
    const uint32_t act_total = info.act_total;

    void *so = rl_dlopen_(so_path);
    RlSurface s;
    rl_resolve_surface_(so, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(seed, n, &env) == K26RL_OK);

    /* Direct addressing: the explicit reset advances every episode
     * index by one without stepping, so index `epi` is reached with
     * no simulated time and no auto-reset traversal. */
    for (uint32_t k = 0; k < epi; k++) {
        ASSERT(s.reset(env) == K26RL_OK);
    }

    double *obs = malloc(sizeof(double) * (size_t)n * obs_total);
    double *rew = malloc(sizeof(double) * n);
    double *act = malloc(sizeof(double) * (size_t)n * act_total);
    uint32_t *fl = malloc(sizeof(uint32_t) * n);
    ASSERT(obs && rew && act && fl);

    ASSERT(s.obs(env, obs) == K26RL_OK);
    ASSERT(memcmp(obs + (size_t)e * obs_total, ep.initial_obs,
                  sizeof(double) * obs_total) == 0);

    for (uint32_t t = 0; t < ep.step_count; t++) {
        /* Every environment gets the target's recorded action row;
         * the target's streams are unaffected by the neighbours
         * (vector independence gate). */
        for (uint32_t j = 0; j < n; j++) {
            memcpy(act + (size_t)j * act_total,
                   ep.act + (size_t)t * act_total,
                   sizeof(double) * act_total);
        }
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.obs(env, obs) == K26RL_OK);
        ASSERT(s.reward(env, rew) == K26RL_OK);
        ASSERT(s.flags(env, fl) == K26RL_OK);
        ASSERT(memcmp(obs + (size_t)e * obs_total,
                      ep.obs + (size_t)t * obs_total,
                      sizeof(double) * obs_total) == 0);
        ASSERT(memcmp(&rew[e], &ep.rewards[t], sizeof(double)) == 0);
        ASSERT(fl[e] == ep.flags[t]);
    }

    uint16_t reason = ep.end_reason;
    free(obs);
    free(rew);
    free(act);
    free(fl);
    k26rl_episode_free(&ep);
    s.destroy(env);
    dlclose(so);
    return reason;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_resim")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/resim.kfl", RESIM_KFL);
    rl_compile_(WORK_DIR "/resim.kfl", WORK_DIR "/resim", WORK_DIR);
    rl_write_file_(WORK_DIR "/resim_fault.kfl", RESIM_FAULT_KFL);
    rl_compile_(WORK_DIR "/resim_fault.kfl", WORK_DIR "/resim_fault",
                WORK_DIR);

    /* Recording 1: serve session, scripted actions, two episodes per
     * environment under the create seed, a reseed, one more episode
     * per environment under the new key. */
    {
        void *so = rl_dlopen_(WORK_DIR "/resim.rlenv.so");
        RlSurface s;
        rl_resolve_surface_(so, &s);
        K26RlEnv *env = NULL;
        ASSERT(s.create(101, 2, &env) == K26RL_OK);
        ASSERT(s.output(env, WORK_DIR "/resim.k26epi") == K26RL_OK);
        uint32_t t = 0;
        record_until_(&s, env, 2, 2, &t);
        ASSERT(s.reset_seeded(env, 202) == K26RL_OK);
        record_until_(&s, env, 2, 1, &t);
        s.destroy(env);
        dlclose(so);
    }

    /* Recording 2: the batch executable, whose episodes fault at
     * transition 3. */
    rl_run_or_die_(WORK_DIR "/resim_fault --envs 2 --episodes 2"
                   " --seed 33 --out " WORK_DIR "/resim_fault.k26epi"
                   " > /dev/null");

    /* Rebuild every indexed episode of both files. */
    int n_term = 0, n_trunc = 0, n_fault = 0, n_ord1 = 0;
    {
        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/resim.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeInfo info;
        ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
        ASSERT(info.episode_count >= 6);
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord = 0, e = 0, epi = 0;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &e, &epi)
                   == K26RL_OK);
            uint16_t reason = resim_one_(rd, WORK_DIR "/resim.rlenv.so",
                                         ord, e, epi);
            if (reason == K26RL_END_TERMINATED) n_term++;
            if (reason == K26RL_END_TRUNCATED)  n_trunc++;
            if (ord == 1) n_ord1++;
        }
        /* The rekey frame's seed resolves from the file alone. */
        uint64_t s0 = 0, s1 = 0;
        ASSERT(k26rl_episode_reader_seed(rd, 0, &s0) == K26RL_OK);
        ASSERT(k26rl_episode_reader_seed(rd, 1, &s1) == K26RL_OK);
        ASSERT(s0 == 101 && s1 == 202);
        k26rl_episode_reader_close(rd);
    }
    {
        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/resim_fault.k26epi",
                                         &rd) == K26RL_OK);
        K26RlEpisodeInfo info;
        ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord = 0, e = 0, epi = 0;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &e, &epi)
                   == K26RL_OK);
            uint16_t reason = resim_one_(
                rd, WORK_DIR "/resim_fault.rlenv.so", ord, e, epi);
            ASSERT(reason == K26RL_END_FAULT);
            n_fault++;
        }
        k26rl_episode_reader_close(rd);
    }

    /* The sweep must have exercised every ending kind and a post-
     * reseed episode; these counts are deterministic for the fixed
     * fixtures and seeds. */
    ASSERT(n_term >= 1);
    ASSERT(n_trunc >= 1);
    ASSERT(n_fault >= 4);
    ASSERT(n_ord1 >= 2);
    printf("test_rl_resim: rebuilt %d terminated, %d truncated, %d"
           " faulted, %d post-reseed episode(s), all bitwise equal\n",
           n_term, n_trunc, n_fault, n_ord1);
    return 0;
}
