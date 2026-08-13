/* test_k26rl_hotpath.c - the writer's no-allocation contract.
 *
 * The library promises that every buffer the writer touches between
 * open and close is allocated at open, once. The proof is mechanical:
 * this binary defines malloc, calloc, realloc, and free, so these
 * definitions preempt the C library's at link time and every
 * allocation in the process is counted before forwarding to the
 * glibc internals. Acceptance: the counters stay exactly zero from
 * the moment open returns until close is called, across plain steps,
 * chunk-boundary flushes, episode starts and ends, a refused
 * over-budget start, and a rekey; open and close themselves move the
 * counters, which proves the interposition is live on both sides of
 * the window; the closed file reopens clean with every episode
 * indexed and decodable. */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#include "episode_fixture.h"

/* The glibc-internal entry points, reachable beneath the
 * interposing definitions below. */
extern void *__libc_malloc(size_t size);
extern void *__libc_calloc(size_t nmemb, size_t size);
extern void *__libc_realloc(void *ptr, size_t size);
extern void __libc_free(void *ptr);

static unsigned long n_malloc_;
static unsigned long n_calloc_;
static unsigned long n_realloc_;
static unsigned long n_free_;

void *malloc(size_t size)
{
    n_malloc_++;
    return __libc_malloc(size);
}

void *calloc(size_t nmemb, size_t size)
{
    n_calloc_++;
    return __libc_calloc(nmemb, size);
}

void *realloc(void *ptr, size_t size)
{
    n_realloc_++;
    return __libc_realloc(ptr, size);
}

void free(void *ptr)
{
    n_free_++;
    __libc_free(ptr);
}

static unsigned long counters_total_(void)
{
    return n_malloc_ + n_calloc_ + n_realloc_ + n_free_;
}

static void counters_arm_(void)
{
    n_malloc_ = 0;
    n_calloc_ = 0;
    n_realloc_ = 0;
    n_free_ = 0;
}

/* One over dr_max, to prove the refusal; the first HOT_DR_MAX
 * entries serve the accepted starts. */
#define HOT_DR_MAX 4u

static const double obs_[FIX_OBS] = { 1.5, 2.5, 3.5 };
static const double act_[FIX_ACT] = { 0.25, 0.75 };
static const double rew_[FIX_AGENTS] = { 1.0, -1.0 };
static const double adj_[FIX_AGENTS] = { 0.5, -0.5 };
static const uint32_t dr_tags_[HOT_DR_MAX + 1] = { 11, 12, 13, 14, 15 };
static const double dr_vals_[HOT_DR_MAX + 1] = { 1.5, 2.5, 3.5, 4.5, 5.5 };

int main(void)
{
    Fixture fx;
    char path[512];
    K26RlEpisodeGeom g;
    K26RlEpisodeWriter *w = NULL;
    K26RlEpisodeReader *r = NULL;
    K26RlEpisodeInfo info;
    K26RlEpisodeData d;
    uint32_t ord = 0;
    uint32_t s;

    fix_build_(&fx);
    fix_path_(path, sizeof path, "hotpath.k26epi");
    remove(path);

    g.n_envs = FIX_N_ENVS;
    g.agent_count = FIX_AGENTS;
    g.obs_total = FIX_OBS;
    g.act_total = FIX_ACT;
    g.steps_per_chunk = FIX_CHUNK;
    g.dr_max = HOT_DR_MAX;

    counters_arm_();
    ASSERT(k26rl_episode_writer_open(path, &g, FIX_SEED, 0, "3.2",
                                     "kfl-rl-0.1", fx.spec, fx.spec_len,
                                     &w) == K26RL_OK);
    /* Open allocates. A zero here would mean the interposition is
     * not seeing the library's allocations and the zeros asserted
     * below would be vacuous. */
    ASSERT(counters_total_() > 0);

    counters_arm_();

    /* A start carrying more pairs than dr_max is refused without
     * touching the file or the heap. */
    ASSERT(k26rl_episode_writer_start(w, 0, 0, obs_, dr_tags_, dr_vals_,
                                      HOT_DR_MAX + 1) == K26RL_E_GEOMETRY);

    /* Episode 0 on each environment: env 0 crosses a chunk boundary
     * at step 4 and ends on a partial chunk; env 1, started with
     * exactly dr_max pairs, ends exactly at a chunk boundary. */
    ASSERT(k26rl_episode_writer_start(w, 0, 0, obs_, dr_tags_, dr_vals_,
                                      2) == K26RL_OK);
    ASSERT(k26rl_episode_writer_start(w, 1, 0, obs_, dr_tags_, dr_vals_,
                                      HOT_DR_MAX) == K26RL_OK);
    for (s = 0; s < 6; s++)
        ASSERT(k26rl_episode_writer_step(w, 0, obs_, act_, rew_, 0,
                                         0.02) == K26RL_OK);
    for (s = 0; s < 4; s++)
        ASSERT(k26rl_episode_writer_step(w, 1, obs_, act_, rew_, 0,
                                         0.02) == K26RL_OK);
    ASSERT(k26rl_episode_writer_end(w, 0, K26RL_END_TERMINATED, 0,
                                    adj_) == K26RL_OK);
    ASSERT(k26rl_episode_writer_end(w, 1, K26RL_END_TRUNCATED, 0,
                                    adj_) == K26RL_OK);

    /* A fresh episode, a rekey at the boundary, then another episode
     * under the new key crossing one more chunk boundary. */
    ASSERT(k26rl_episode_writer_start(w, 0, 1, obs_, NULL, NULL,
                                      0) == K26RL_OK);
    for (s = 0; s < 2; s++)
        ASSERT(k26rl_episode_writer_step(w, 0, obs_, act_, rew_, 0,
                                         0.02) == K26RL_OK);
    ASSERT(k26rl_episode_writer_end(w, 0, K26RL_END_TERMINATED, 0,
                                    adj_) == K26RL_OK);
    ASSERT(k26rl_episode_writer_rekey(w, FIX_SEED2, &ord) == K26RL_OK);
    ASSERT(ord == 1);
    ASSERT(k26rl_episode_writer_start(w, 0, 0, obs_, dr_tags_, dr_vals_,
                                      1) == K26RL_OK);
    for (s = 0; s < 5; s++)
        ASSERT(k26rl_episode_writer_step(w, 0, obs_, act_, rew_, 0,
                                         0.02) == K26RL_OK);
    ASSERT(k26rl_episode_writer_end(w, 0, K26RL_END_TERMINATED, 0,
                                    adj_) == K26RL_OK);

    /* The contract: nothing allocated or freed since open returned. */
    ASSERT(n_malloc_ == 0);
    ASSERT(n_calloc_ == 0);
    ASSERT(n_realloc_ == 0);
    ASSERT(n_free_ == 0);

    counters_arm_();
    ASSERT(k26rl_episode_writer_close(w) == K26RL_OK);
    /* Close assembles the index and releases the open-time buffers,
     * so the counters move; with the assertion after open this
     * brackets the zero window with live interposition on both
     * sides. */
    ASSERT(counters_total_() > 0);

    /* The closed file is whole: clean close, every episode indexed
     * by the close-time pass, and the recorded offsets decode. */
    ASSERT(k26rl_episode_reader_open(path, &r) == K26RL_OK);
    ASSERT(k26rl_episode_reader_info(r, &info) == K26RL_OK);
    ASSERT(info.clean_close == 1);
    ASSERT(info.episode_count == 4);
    ASSERT(info.unindexed_episode_starts == 0);
    ASSERT(k26rl_episode_read(r, 0, 0, 0, &d) == K26RL_OK);
    ASSERT(d.step_count == 6);
    ASSERT(d.dr_count == 2);
    k26rl_episode_free(&d);
    ASSERT(k26rl_episode_read(r, 1, 0, 0, &d) == K26RL_OK);
    ASSERT(d.step_count == 5);
    k26rl_episode_free(&d);
    k26rl_episode_reader_close(r);

    remove(path);
    printf("test_k26rl_hotpath: ok\n");
    return 0;
}
