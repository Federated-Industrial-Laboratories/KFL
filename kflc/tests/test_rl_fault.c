/* test_rl_fault.c: fault and auto-reset gates for the Grammar 3.2
 * dual-mode artifact.
 *
 * Gates:
 *   1. Fault in one environment of a stepping set: a deterministically
 *      induced fault (a non-finite reward from the objective, driven
 *      by one environment's action) in environment 1 of 3. The step
 *      call returns OK; flag bit 2, the fault code, the held pre-step
 *      observations, and the zero reward appear for exactly the
 *      faulted environment; the neighbours advance normally.
 *   2. The recorded fault carriers: the episode file's fault record
 *      carries its five fixed column families (the faulting call's
 *      actions, the pre-step observations, zero rewards, zero applied
 *      dt, the fault flag alone), and the episode-end frame carries
 *      the same registry reason with zero terminal adjustments.
 *   3. Auto-reset boundaries for all three endings: after a fault,
 *      after a truncation, and after a termination, the environment's
 *      next step is a boundary reset carrying only the reset-boundary
 *      bit, a zero reward, and a cleared fault code, while the
 *      neighbours step normally in the same call.
 *   4. Termination bookkeeping: the terminal adjustment is added to
 *      the terminating step's reward and recorded in the episode-end
 *      frame; a truncated episode records a zero adjustment.
 *   5. Episode index monotonicity: a batch run's file indexes each
 *      environment's episodes at ordinal 0 with indices 0, 1, 2 in
 *      file order.
 *   8. Integrator-divergence arm: a world whose declared dynamics
 *      drive the runtime's integrator to a genuine failure status
 *      (a zero-GM primary makes the Wisdom-Holman Kepler drift
 *      reject its solve, a NO_CONVERGE-class failure) maps to the
 *      K26RL_E_DIVERGED fault: the step call returns OK, the flag
 *      word carries the fault bit, the fault code is DIVERGED, the
 *      pre-step observations hold, and the episode ends faulted in
 *      the file with the same code.
 *   9. No phantom divergence on a healthy heavy-satellite world: a
 *      Pluto/Charon/probe system (JPL GM and mass values) whose
 *      satellite pair crosses the close-encounter detector's mass
 *      threshold at every separation runs its full horizon on the
 *      default Wisdom-Holman base with no fault and a truncated
 *      ending. Pins the regression where the WH base was admitted
 *      to the MERCURIUS split (whose NEAR pass the WH kick ignores)
 *      and the world's unset IAS15 tolerance made that pass fail
 *      structurally, faulting a healthy environment DIVERGED at
 *      step 1.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"
#include "k26rl_episode.h"

#define WORK_DIR "/tmp/kflc_rl_fault_test"

/* reward 1.0 / (1.0 - push): finite at the default action (0.0),
 * division by zero when a caller drives push to 1.0, so the fault
 * lands in exactly the environment whose action slice carries 1.0.
 * Horizon 6 truncates the environments that never fault. */
static const char *const FAULTV_KFL =
    "form RL_FAULTV\n"
    "fn world fault_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 6\n"
    "    end\n"
    "    action push box 0.0 4.0 default 0.0\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0 / (1.0 - push)\n"
    "    end\n"
    "end\n"
    "end\n";

/* Termination with a terminal adjustment: every episode terminates at
 * transition 3 with reward 1.0 + 5.0. */
static const char *const TERM_KFL =
    "form RL_TERM\n"
    "fn world term_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 100\n"
    "        terminated when episode.steps > 2\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0\n"
    "        terminal 5.0\n"
    "    end\n"
    "end\n"
    "end\n";

int main(void)
{
    if (!rl_libs_present_("test_rl_fault")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/faultv.kfl", FAULTV_KFL);
    rl_compile_(WORK_DIR "/faultv.kfl", WORK_DIR "/faultv", WORK_DIR);
    rl_write_file_(WORK_DIR "/term.kfl", TERM_KFL);
    rl_compile_(WORK_DIR "/term.kfl", WORK_DIR "/term", WORK_DIR);

    /* Gates 1 to 3 (fault and truncation halves): one serve session
     * over 3 environments with output enabled. */
    void *so = rl_dlopen_(WORK_DIR "/faultv.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    K26RlEnv *env = NULL;
    ASSERT(s.create(21, 3, &env) == K26RL_OK);
    ASSERT(s.output(env, WORK_DIR "/faultv.k26epi") == K26RL_OK);

    double act[3] = { 0.0, 0.0, 0.0 };
    double obs[3 * 4], obs_pre[3 * 4], fault_obs[4];
    double rew[3];
    uint32_t fl[3];
    uint16_t fc[3];

    /* Two clean transitions for everyone. */
    for (int t = 0; t < 2; t++) {
        ASSERT(s.step(env, act) == K26RL_OK);
        ASSERT(s.flags(env, fl) == K26RL_OK);
        ASSERT(fl[0] == 0 && fl[1] == 0 && fl[2] == 0);
    }
    ASSERT(s.obs(env, obs_pre) == K26RL_OK);

    /* Transition 3: environment 1's action drives its reward
     * non-finite. The call itself succeeds; the fault is confined to
     * environment 1 through all three carriers; its observations hold
     * the pre-step values; the neighbours advance. */
    act[1] = 1.0;
    ASSERT(s.step(env, act) == K26RL_OK);
    act[1] = 0.0;
    ASSERT(s.flags(env, fl) == K26RL_OK);
    ASSERT(fl[0] == 0 && fl[1] == K26RL_FLAG_FAULT && fl[2] == 0);
    ASSERT(s.fault_codes(env, fc) == K26RL_OK);
    ASSERT(fc[0] == 0 && fc[1] == (uint16_t)K26RL_E_ENV_INTERNAL &&
           fc[2] == 0);
    ASSERT(s.obs(env, obs) == K26RL_OK);
    ASSERT(memcmp(obs + 4, obs_pre + 4, 4 * sizeof(double)) == 0);
    ASSERT(memcmp(obs + 0, obs_pre + 0, 4 * sizeof(double)) != 0);
    ASSERT(memcmp(obs + 8, obs_pre + 8, 4 * sizeof(double)) != 0);
    ASSERT(s.reward(env, rew) == K26RL_OK);
    ASSERT(rew[0] == 1.0 && rew[1] == 0.0 && rew[2] == 1.0);
    memcpy(fault_obs, obs + 4, sizeof fault_obs);
    printf("gate 1: fault confined to one environment of three: OK\n");

    /* Boundary after the fault: only the reset-boundary bit, a
     * cleared fault code, a zero reward; the neighbours take their
     * transition 4 in the same call. */
    ASSERT(s.step(env, act) == K26RL_OK);
    ASSERT(s.flags(env, fl) == K26RL_OK);
    ASSERT(fl[0] == 0 && fl[1] == K26RL_FLAG_RESET_BOUNDARY && fl[2] == 0);
    ASSERT(s.fault_codes(env, fc) == K26RL_OK);
    ASSERT(fc[1] == 0);
    ASSERT(s.reward(env, rew) == K26RL_OK);
    ASSERT(rew[1] == 0.0);

    /* Two more transitions truncate environments 0 and 2 at the
     * horizon while environment 1 is mid-episode. */
    for (int t = 0; t < 2; t++) {
        ASSERT(s.step(env, act) == K26RL_OK);
    }
    ASSERT(s.flags(env, fl) == K26RL_OK);
    ASSERT(fl[0] == K26RL_FLAG_TRUNCATED && fl[1] == 0 &&
           fl[2] == K26RL_FLAG_TRUNCATED);

    /* Boundary after truncation, neighbours stepping normally. */
    ASSERT(s.step(env, act) == K26RL_OK);
    ASSERT(s.flags(env, fl) == K26RL_OK);
    ASSERT(fl[0] == K26RL_FLAG_RESET_BOUNDARY && fl[1] == 0 &&
           fl[2] == K26RL_FLAG_RESET_BOUNDARY);
    ASSERT(s.reward(env, rew) == K26RL_OK);
    ASSERT(rew[0] == 0.0 && rew[2] == 0.0);

    s.destroy(env);
    env = NULL;
    printf("gate 3a: boundary resets after fault and truncation,"
           " neighbours unaffected: OK\n");

    /* Gate 2: the recorded fault carriers. */
    {
        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/faultv.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeInfo info;
        ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
        ASSERT(info.n_envs == 3);

        K26RlEpisodeData ep;
        ASSERT(k26rl_episode_read(rd, 0, 1, 0, &ep) == K26RL_OK);
        ASSERT(ep.end_reason == K26RL_END_FAULT);
        ASSERT(ep.fault_code == (uint16_t)K26RL_E_ENV_INTERNAL);
        /* Two transitions plus the one fault record. */
        ASSERT(ep.step_count == 3);
        /* The five fixed column families of the fault record. */
        ASSERT(ep.act[2] == 1.0);                     /* faulting call */
        ASSERT(memcmp(ep.obs + 2 * 4, ep.obs + 1 * 4,
                      4 * sizeof(double)) == 0);      /* pre-step obs */
        ASSERT(memcmp(ep.obs + 2 * 4, fault_obs,
                      4 * sizeof(double)) == 0);      /* getter equality */
        ASSERT(ep.rewards[2] == 0.0);
        ASSERT(ep.applied_dt[2] == 0.0);
        ASSERT(ep.flags[2] == K26RL_FLAG_FAULT);
        /* Transitions before the fault are ordinary records. */
        ASSERT(ep.applied_dt[0] == 0.1 && ep.applied_dt[1] == 0.1);
        ASSERT(ep.flags[0] == 0 && ep.flags[1] == 0);
        ASSERT(ep.rewards[0] == 1.0 && ep.rewards[1] == 1.0);
        ASSERT(ep.terminal_adjustments[0] == 0.0);
        k26rl_episode_free(&ep);

        /* A truncated neighbour: full horizon, zero adjustment. */
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep) == K26RL_OK);
        ASSERT(ep.end_reason == K26RL_END_TRUNCATED);
        ASSERT(ep.fault_code == 0);
        ASSERT(ep.step_count == 6);
        ASSERT(ep.flags[5] == K26RL_FLAG_TRUNCATED);
        ASSERT(ep.terminal_adjustments[0] == 0.0);
        for (uint32_t t = 0; t < 6; t++) {
            ASSERT(ep.applied_dt[t] == 0.1);
        }
        k26rl_episode_free(&ep);

        /* Every environment's second episode was open at close (the
         * final boundary step restarted environments 0 and 2, and
         * environment 1 was mid-episode); the starts stay unindexed,
         * and nothing invents an ending for them. */
        ASSERT(info.unindexed_episode_starts == 3);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 2: fault record column families + episode-end"
           " reason: OK\n");

    /* Gates 3 (termination half) and 4: terminated episodes carry the
     * terminal adjustment; the boundary after termination is a
     * boundary like the others. */
    {
        void *so2 = rl_dlopen_(WORK_DIR "/term.rlenv.so");
        RlSurface t;
        rl_resolve_surface_(so2, &t);
        K26RlEnv *te = NULL;
        ASSERT(t.create(5, 1, &te) == K26RL_OK);
        ASSERT(t.output(te, WORK_DIR "/term.k26epi") == K26RL_OK);
        double a1[1] = { 0.0 };
        double r1;
        uint32_t f1;
        for (int k = 0; k < 2; k++) {
            ASSERT(t.step(te, a1) == K26RL_OK);
            ASSERT(t.reward(te, &r1) == K26RL_OK);
            ASSERT(r1 == 1.0);
        }
        ASSERT(t.step(te, a1) == K26RL_OK);
        ASSERT(t.flags(te, &f1) == K26RL_OK);
        ASSERT(f1 == K26RL_FLAG_TERMINATED);
        ASSERT(t.reward(te, &r1) == K26RL_OK);
        ASSERT(r1 == 6.0);   /* reward 1.0 + terminal 5.0 */
        ASSERT(t.step(te, a1) == K26RL_OK);
        ASSERT(t.flags(te, &f1) == K26RL_OK);
        ASSERT(f1 == K26RL_FLAG_RESET_BOUNDARY);
        ASSERT(t.reward(te, &r1) == K26RL_OK);
        ASSERT(r1 == 0.0);
        ASSERT(t.step(te, a1) == K26RL_OK);
        ASSERT(t.flags(te, &f1) == K26RL_OK);
        ASSERT(f1 == 0);
        t.destroy(te);
        dlclose(so2);

        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/term.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeData ep;
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep) == K26RL_OK);
        ASSERT(ep.end_reason == K26RL_END_TERMINATED);
        ASSERT(ep.step_count == 3);
        ASSERT(ep.rewards[2] == 6.0);
        ASSERT(ep.terminal_adjustments[0] == 5.0);
        ASSERT(ep.flags[2] == K26RL_FLAG_TERMINATED);
        k26rl_episode_free(&ep);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 3b/4: termination boundary + terminal adjustment"
           " recorded: OK\n");

    /* Gate 5: episode index monotonicity in a batch run's file. */
    {
        rl_run_or_die_(WORK_DIR "/faultv --envs 2 --episodes 3 --seed 8"
                       " --out " WORK_DIR "/mono.k26epi > /dev/null");
        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/mono.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeInfo info;
        ASSERT(k26rl_episode_reader_info(rd, &info) == K26RL_OK);
        ASSERT(info.episode_count == 6);
        uint32_t next_ep[2] = { 0, 0 };
        for (uint32_t k = 0; k < info.episode_count; k++) {
            uint32_t ord = 9, e = 9, epi = 9;
            ASSERT(k26rl_episode_reader_at(rd, k, &ord, &e, &epi)
                   == K26RL_OK);
            ASSERT(ord == 0);
            ASSERT(e < 2);
            ASSERT(epi == next_ep[e]);
            next_ep[e]++;
        }
        ASSERT(next_ep[0] == 3 && next_ep[1] == 3);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 5: episode index monotonicity (0,1,2 per"
           " environment): OK\n");

    /* Gate 6: termination and truncation landing on one step are one
     * outcome, not two. Termination wins; the flag word is exactly
     * the terminated bit and agrees in kind with the recorded end
     * reason. */
    {
        static const char *const cofire_kfl =
            "form RL_COFIRE\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft gm=1.0 parent=earth"
            " pos_x=7.0e6 vel_y=7350.0\n"
            "    episode\n"
            "        control_dt 0.1\n"
            "        horizon 3\n"
            "        terminated when episode.steps > 2\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe craft from earth mode=geometric as trk\n"
            "    objective\n"
            "        reward 1.0\n"
            "    end\n"
            "end\n"
            "end\n";
        rl_write_file_(WORK_DIR "/cofire.kfl", cofire_kfl);
        rl_compile_(WORK_DIR "/cofire.kfl", WORK_DIR "/cofire", WORK_DIR);
        void *so3 = rl_dlopen_(WORK_DIR "/cofire.rlenv.so");
        RlSurface c;
        rl_resolve_surface_(so3, &c);
        K26RlEnv *ce = NULL;
        ASSERT(c.create(7, 1, &ce) == K26RL_OK);
        ASSERT(c.output(ce, WORK_DIR "/cofire.k26epi") == K26RL_OK);
        double a1[1] = { 0.0 };
        uint32_t f1 = 0;
        for (int k = 0; k < 3; k++) ASSERT(c.step(ce, a1) == K26RL_OK);
        ASSERT(c.flags(ce, &f1) == K26RL_OK);
        ASSERT(f1 == K26RL_FLAG_TERMINATED);
        c.destroy(ce);
        dlclose(so3);

        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/cofire.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeData ep;
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep) == K26RL_OK);
        ASSERT(ep.end_reason == K26RL_END_TERMINATED);
        ASSERT(ep.flags[2] == K26RL_FLAG_TERMINATED);
        k26rl_episode_free(&ep);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 6: simultaneous termination and truncation resolve"
           " to terminated alone: OK\n");

    /* Gate 7: user arithmetic over fractionless float literals is
     * double arithmetic (reward 1.0 / 2.0 is exactly 0.5 through the
     * surface), and a non-finite terminal adjustment faults the
     * episode instead of crashing the process or reaching the
     * recorded stream. */
    {
        static const char *const arith_kfl =
            "form RL_ARITH\n"
            "fn world w\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft gm=1.0 parent=earth"
            " pos_x=7.0e6 vel_y=7350.0\n"
            "    episode\n"
            "        control_dt 0.1\n"
            "        horizon 100\n"
            "        terminated when episode.steps > 1\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe craft from earth mode=geometric as trk\n"
            "    objective\n"
            "        reward 1.0 / 2.0\n"
            "        terminal 1.0 / 0.0\n"
            "    end\n"
            "end\n"
            "end\n";
        rl_write_file_(WORK_DIR "/arith.kfl", arith_kfl);
        rl_compile_(WORK_DIR "/arith.kfl", WORK_DIR "/arith", WORK_DIR);
        void *so4 = rl_dlopen_(WORK_DIR "/arith.rlenv.so");
        RlSurface a;
        rl_resolve_surface_(so4, &a);
        K26RlEnv *ae = NULL;
        ASSERT(a.create(11, 1, &ae) == K26RL_OK);
        double a1[1] = { 0.0 };
        double r1 = 0.0;
        uint32_t f1 = 0;
        uint16_t c1 = 0;
        ASSERT(a.step(ae, a1) == K26RL_OK);
        ASSERT(a.reward(ae, &r1) == K26RL_OK);
        ASSERT(r1 == 0.5);
        ASSERT(a.step(ae, a1) == K26RL_OK);
        ASSERT(a.flags(ae, &f1) == K26RL_OK);
        ASSERT(f1 == K26RL_FLAG_FAULT);
        ASSERT(a.fault_codes(ae, &c1) == K26RL_OK);
        ASSERT(c1 == (uint16_t)K26RL_E_ENV_INTERNAL);
        ASSERT(a.reward(ae, &r1) == K26RL_OK);
        ASSERT(r1 == 0.0);
        a.destroy(ae);
        dlclose(so4);
    }
    printf("gate 7: fractionless literals stay double arithmetic;"
           " a non-finite terminal faults: OK\n");

    /* Gate 8: the integrator-divergence arm. The primary's zero GM
     * is the declared dynamics; the Wisdom-Holman drift's universal-
     * variable Kepler solve genuinely rejects it inside the runtime
     * on the first substep, the runtime returns the integrator
     * status through the exact stepping entry, and the environment
     * maps it to a per-environment K26RL_E_DIVERGED fault. Nothing
     * here injects a status: the failure travels the full stack. */
    {
        static const char *const diverge_kfl =
            "form RL_DIVERGE\n"
            "fn world w\n"
            "    astro_body core gm=0.0 mass=5.972e24\n"
            "    astro_body craft gm=1.0 parent=core"
            " pos_x=7.0e6 vel_y=7350.0\n"
            "    episode\n"
            "        control_dt 0.1\n"
            "        horizon 6\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe craft from core mode=geometric as trk\n"
            "    objective\n"
            "        reward 1.0\n"
            "    end\n"
            "end\n"
            "end\n";
        rl_write_file_(WORK_DIR "/diverge.kfl", diverge_kfl);
        rl_compile_(WORK_DIR "/diverge.kfl", WORK_DIR "/diverge", WORK_DIR);
        void *so5 = rl_dlopen_(WORK_DIR "/diverge.rlenv.so");
        RlSurface d;
        rl_resolve_surface_(so5, &d);
        K26RlEnv *de = NULL;
        ASSERT(d.create(13, 1, &de) == K26RL_OK);
        ASSERT(d.output(de, WORK_DIR "/diverge.k26epi") == K26RL_OK);

        double pre[4], post[4];
        double a1[1] = { 0.0 };
        double r1 = -1.0;
        uint32_t f1 = 0;
        uint16_t c1 = 0;
        ASSERT(d.obs(de, pre) == K26RL_OK);

        /* The faulting step: the call itself succeeds; the fault is
         * carried per environment. */
        ASSERT(d.step(de, a1) == K26RL_OK);
        ASSERT(d.flags(de, &f1) == K26RL_OK);
        ASSERT(f1 == K26RL_FLAG_FAULT);
        ASSERT(d.fault_codes(de, &c1) == K26RL_OK);
        ASSERT(c1 == (uint16_t)K26RL_E_DIVERGED);
        ASSERT(d.obs(de, post) == K26RL_OK);
        ASSERT(memcmp(post, pre, sizeof post) == 0);   /* held */
        ASSERT(d.reward(de, &r1) == K26RL_OK);
        ASSERT(r1 == 0.0);

        /* Boundary after the fault clears the code. */
        ASSERT(d.step(de, a1) == K26RL_OK);
        ASSERT(d.flags(de, &f1) == K26RL_OK);
        ASSERT(f1 == K26RL_FLAG_RESET_BOUNDARY);
        ASSERT(d.fault_codes(de, &c1) == K26RL_OK);
        ASSERT(c1 == 0);
        d.destroy(de);
        dlclose(so5);

        /* The recorded ending. */
        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/diverge.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeData ep;
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep) == K26RL_OK);
        ASSERT(ep.end_reason == K26RL_END_FAULT);
        ASSERT(ep.fault_code == (uint16_t)K26RL_E_DIVERGED);
        ASSERT(ep.step_count == 1);
        ASSERT(ep.flags[0] == K26RL_FLAG_FAULT);
        ASSERT(ep.rewards[0] == 0.0);
        ASSERT(ep.applied_dt[0] == 0.0);
        ASSERT(memcmp(ep.obs, ep.initial_obs, 4 * sizeof(double)) == 0);
        ASSERT(memcmp(ep.obs, pre, sizeof pre) == 0);
        k26rl_episode_free(&ep);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 8: integrator divergence surfaces as a DIVERGED"
           " fault, end to end: OK\n");

    /* Gate 9: the healthy counterpart of gate 8. Pluto, Charon, and
     * a small probe with JPL GM and mass values. The Charon+probe
     * pair is a genuine close pair by the semi-major-axis Hill
     * criterion (y near 1.16, inside the transition window's inner
     * edge), so on the default Wisdom-Holman base with Pluto at
     * body 0 the MERCURIUS split runs on every step, carrying the
     * pair force in the IAS15 drift at full weight. The episode
     * must run its declared horizon with no fault and end
     * truncated. This gate sits at the compiled-environment level
     * because that is where the historical defect surfaced (a
     * failing near pass manufacturing failure statuses on a healthy
     * world): it covers detection, split admission, the split
     * composition, the exact stepping status, and the fault mapping
     * in one path. */
    {
        static const char *const pluto_kfl =
            "form RL_PLUTO\n"
            "fn world w\n"
            "    astro_body pluto gm=8.696e11 mass=1.303e22\n"
            "    astro_body charon gm=1.058e11 mass=1.586e21"
            " parent=pluto pos_x=1.9595e7 vel_y=222.0\n"
            "    astro_body probe gm=1.0 mass=1.0e3 parent=pluto"
            " pos_x=3.0e7 vel_y=170.2\n"
            "    episode\n"
            "        control_dt 1.0\n"
            "        horizon 8\n"
            "    end\n"
            "    action push box -1.0 1.0 default 0.0\n"
            "    observe probe from pluto mode=geometric as trk\n"
            "    objective\n"
            "        reward 1.0\n"
            "    end\n"
            "end\n"
            "end\n";
        rl_write_file_(WORK_DIR "/pluto.kfl", pluto_kfl);
        rl_compile_(WORK_DIR "/pluto.kfl", WORK_DIR "/pluto", WORK_DIR);
        void *so6 = rl_dlopen_(WORK_DIR "/pluto.rlenv.so");
        RlSurface p;
        rl_resolve_surface_(so6, &p);
        K26RlEnv *pe = NULL;
        ASSERT(p.create(17, 1, &pe) == K26RL_OK);
        ASSERT(p.output(pe, WORK_DIR "/pluto.k26epi") == K26RL_OK);

        double a1[1] = { 0.0 };
        double r1;
        uint32_t f1;
        uint16_t c1;
        for (int t = 0; t < 8; t++) {
            ASSERT(p.step(pe, a1) == K26RL_OK);
            ASSERT(p.flags(pe, &f1) == K26RL_OK);
            ASSERT(f1 == (t < 7 ? 0u : K26RL_FLAG_TRUNCATED));
            ASSERT(p.fault_codes(pe, &c1) == K26RL_OK);
            ASSERT(c1 == 0);
            ASSERT(p.reward(pe, &r1) == K26RL_OK);
            ASSERT(r1 == 1.0);
        }
        p.destroy(pe);
        dlclose(so6);

        K26RlEpisodeReader *rd = NULL;
        ASSERT(k26rl_episode_reader_open(WORK_DIR "/pluto.k26epi", &rd)
               == K26RL_OK);
        K26RlEpisodeData ep;
        ASSERT(k26rl_episode_read(rd, 0, 0, 0, &ep) == K26RL_OK);
        ASSERT(ep.end_reason == K26RL_END_TRUNCATED);
        ASSERT(ep.fault_code == 0);
        ASSERT(ep.step_count == 8);
        for (uint32_t t = 0; t < 8; t++) {
            ASSERT(ep.applied_dt[t] == 1.0);
        }
        k26rl_episode_free(&ep);
        k26rl_episode_reader_close(rd);
    }
    printf("gate 9: healthy heavy-satellite world runs its full"
           " horizon unfaulted: OK\n");

    dlclose(so);
    printf("test_rl_fault: 9 gates passed\n");
    return 0;
}
