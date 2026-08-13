/* episode_fixture.h - the scripted dataset shared by the format tests.
 *
 * Two environments, two agents, three observation channels, two
 * action channels, four steps per chunk. Every double comes from one
 * running counter, so each value is distinct, exact in binary64, and
 * reproduced identically on every run. Episodes: env 0 terminates
 * after 6 steps (one full chunk plus a final chunk of 2), env 1
 * truncates after 4 steps (exactly one full chunk), then a rekey,
 * then env 0 faults on its third step under the new key. The faulted
 * episode's final record carries the faulting call's actions, the
 * pre-step observations, zero rewards, zero applied dt, and the
 * fault flag alone; its terminal adjustments are zero.
 *
 * Requires an ASSERT macro defined before inclusion. */
#ifndef EPISODE_FIXTURE_H
#define EPISODE_FIXTURE_H

#ifndef ASSERT
#error "define ASSERT before including episode_fixture.h"
#endif

#include "k26rl_episode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define FIX_N_ENVS  2u
#define FIX_AGENTS  2u
#define FIX_OBS     3u
#define FIX_ACT     2u
#define FIX_CHUNK   4u
#define FIX_SEED    0x1122334455667788ULL
#define FIX_SEED2   0x99AABBCCDDEEFF00ULL
#define FIX_FAULT_CODE 10u   /* K26RL_E_DIVERGED, a fault reason */
#define FIX_MAX_STEPS 6u
#define FIX_MAX_DR  2u
#define FIX_EP_COUNT 3u

typedef struct {
    uint32_t ordinal;
    uint32_t env;
    uint32_t episode;
    uint32_t step_count;
    uint16_t end_reason;
    uint16_t fault_code;
    double initial_obs[FIX_OBS];
    uint32_t dr_count;
    uint32_t dr_tags[FIX_MAX_DR];
    double dr_values[FIX_MAX_DR];
    double obs[FIX_MAX_STEPS * FIX_OBS];
    double act[FIX_MAX_STEPS * FIX_ACT];
    double rewards[FIX_MAX_STEPS * FIX_AGENTS];
    uint32_t flags[FIX_MAX_STEPS];
    double applied_dt[FIX_MAX_STEPS];
    double terminal_adjustments[FIX_AGENTS];
} FixEpisode;

typedef struct {
    FixEpisode eps[FIX_EP_COUNT];
    uint8_t spec[64];
    uint32_t spec_len;
} Fixture;

/* Identity triples in file (episode-end) order. */
static const uint32_t fix_order_[FIX_EP_COUNT][3] = {
    { 0, 0, 0 },
    { 0, 1, 0 },
    { 1, 0, 0 },
};

/* Counter values are small integers, exact in binary64; the half
 * step keeps them off the integer lattice. Every value is distinct
 * and every run reproduces the same bit patterns. */
static inline double fix_next_(uint64_t *ctr)
{
    *ctr += 1;
    return (double)*ctr * 0.5;
}

static inline void fix_spec_put_(uint8_t *p, uint16_t tag, uint32_t v)
{
    p[0] = (uint8_t)(tag & 0xFFu);
    p[1] = (uint8_t)(tag >> 8);
    p[2] = 4;
    p[3] = 0;
    p[4] = 0;
    p[5] = 0;
    p[6] = (uint8_t)(v & 0xFFu);
    p[7] = (uint8_t)((v >> 8) & 0xFFu);
    p[8] = (uint8_t)((v >> 16) & 0xFFu);
    p[9] = (uint8_t)(v >> 24);
}

/* The spec blob: the three geometry tags the reader needs, plus one
 * unknown tag in the middle so skip-by-length is exercised. */
static inline void fix_spec_build_(Fixture *fx)
{
    uint8_t *p = fx->spec;
    static const uint8_t unknown[11] = {
        0xF0, 0x00,               /* tag 0x00F0, unassigned */
        0x05, 0x00, 0x00, 0x00,   /* length 5 */
        0x01, 0x02, 0x03, 0x04, 0x05
    };

    fix_spec_put_(p, K26RL_TAG_AGENT_COUNT, FIX_AGENTS);
    p += 10;
    memcpy(p, unknown, sizeof unknown);
    p += sizeof unknown;
    fix_spec_put_(p, K26RL_TAG_OBS_TOTAL, FIX_OBS);
    p += 10;
    fix_spec_put_(p, K26RL_TAG_ACT_TOTAL, FIX_ACT);
    p += 10;
    fx->spec_len = (uint32_t)(p - fx->spec);
}

static inline void fix_build_(Fixture *fx)
{
    uint64_t ctr = 0;
    FixEpisode *e;
    uint32_t s, j;

    memset(fx, 0, sizeof *fx);
    fix_spec_build_(fx);

    /* Episode (0, 0, 0): terminated after 6 steps. */
    e = &fx->eps[0];
    e->ordinal = 0;
    e->env = 0;
    e->episode = 0;
    e->step_count = 6;
    e->end_reason = K26RL_END_TERMINATED;
    e->fault_code = 0;
    for (j = 0; j < FIX_OBS; j++)
        e->initial_obs[j] = fix_next_(&ctr);
    e->dr_count = 2;
    e->dr_tags[0] = 7;
    e->dr_values[0] = fix_next_(&ctr);
    e->dr_tags[1] = 9;
    e->dr_values[1] = fix_next_(&ctr);
    for (s = 0; s < e->step_count; s++) {
        for (j = 0; j < FIX_OBS; j++)
            e->obs[s * FIX_OBS + j] = fix_next_(&ctr);
        for (j = 0; j < FIX_ACT; j++)
            e->act[s * FIX_ACT + j] = fix_next_(&ctr);
        for (j = 0; j < FIX_AGENTS; j++)
            e->rewards[s * FIX_AGENTS + j] = fix_next_(&ctr);
        e->flags[s] = (s + 1 == e->step_count) ? K26RL_FLAG_TERMINATED : 0;
        e->applied_dt[s] = fix_next_(&ctr);
    }
    for (j = 0; j < FIX_AGENTS; j++)
        e->terminal_adjustments[j] = fix_next_(&ctr);

    /* Episode (0, 1, 0): truncated after 4 steps. */
    e = &fx->eps[1];
    e->ordinal = 0;
    e->env = 1;
    e->episode = 0;
    e->step_count = 4;
    e->end_reason = K26RL_END_TRUNCATED;
    e->fault_code = 0;
    for (j = 0; j < FIX_OBS; j++)
        e->initial_obs[j] = fix_next_(&ctr);
    e->dr_count = 1;
    e->dr_tags[0] = 3;
    e->dr_values[0] = fix_next_(&ctr);
    for (s = 0; s < e->step_count; s++) {
        for (j = 0; j < FIX_OBS; j++)
            e->obs[s * FIX_OBS + j] = fix_next_(&ctr);
        for (j = 0; j < FIX_ACT; j++)
            e->act[s * FIX_ACT + j] = fix_next_(&ctr);
        for (j = 0; j < FIX_AGENTS; j++)
            e->rewards[s * FIX_AGENTS + j] = fix_next_(&ctr);
        e->flags[s] = (s + 1 == e->step_count) ? K26RL_FLAG_TRUNCATED : 0;
        e->applied_dt[s] = fix_next_(&ctr);
    }
    for (j = 0; j < FIX_AGENTS; j++)
        e->terminal_adjustments[j] = fix_next_(&ctr);

    /* Episode (1, 0, 0): faulted on the third step. */
    e = &fx->eps[2];
    e->ordinal = 1;
    e->env = 0;
    e->episode = 0;
    e->step_count = 3;
    e->end_reason = K26RL_END_FAULT;
    e->fault_code = FIX_FAULT_CODE;
    for (j = 0; j < FIX_OBS; j++)
        e->initial_obs[j] = fix_next_(&ctr);
    e->dr_count = 0;
    for (s = 0; s < 2; s++) {
        for (j = 0; j < FIX_OBS; j++)
            e->obs[s * FIX_OBS + j] = fix_next_(&ctr);
        for (j = 0; j < FIX_ACT; j++)
            e->act[s * FIX_ACT + j] = fix_next_(&ctr);
        for (j = 0; j < FIX_AGENTS; j++)
            e->rewards[s * FIX_AGENTS + j] = fix_next_(&ctr);
        e->flags[s] = 0;
        e->applied_dt[s] = fix_next_(&ctr);
    }
    /* The fault record: fresh actions, pre-step observations, zero
     * rewards, zero applied dt, the fault flag alone. */
    for (j = 0; j < FIX_ACT; j++)
        e->act[2 * FIX_ACT + j] = fix_next_(&ctr);
    for (j = 0; j < FIX_OBS; j++)
        e->obs[2 * FIX_OBS + j] = e->obs[1 * FIX_OBS + j];
    for (j = 0; j < FIX_AGENTS; j++)
        e->rewards[2 * FIX_AGENTS + j] = 0.0;
    e->flags[2] = K26RL_FLAG_FAULT;
    e->applied_dt[2] = 0.0;
    /* terminal_adjustments stay zero for a faulted episode. */
}

/* Write the dataset. The two live episodes are stepped interleaved
 * so the per-environment chunk buffers fill concurrently. */
static inline void fix_write_(const Fixture *fx, const char *path)
{
    K26RlEpisodeGeom g;
    K26RlEpisodeWriter *w = NULL;
    const FixEpisode *e0 = &fx->eps[0];
    const FixEpisode *e1 = &fx->eps[1];
    const FixEpisode *e2 = &fx->eps[2];
    uint32_t ord = 0;
    uint32_t s;

    g.n_envs = FIX_N_ENVS;
    g.agent_count = FIX_AGENTS;
    g.obs_total = FIX_OBS;
    g.act_total = FIX_ACT;
    g.steps_per_chunk = FIX_CHUNK;
    g.dr_max = FIX_MAX_DR;
    ASSERT(k26rl_episode_writer_open(path, &g, FIX_SEED, 0, "3.2",
                                     "kfl-rl-0.1", fx->spec, fx->spec_len,
                                     &w) == K26RL_OK);
    ASSERT(k26rl_episode_writer_start(w, 0, 0, e0->initial_obs, e0->dr_tags,
                                      e0->dr_values,
                                      e0->dr_count) == K26RL_OK);
    ASSERT(k26rl_episode_writer_start(w, 1, 0, e1->initial_obs, e1->dr_tags,
                                      e1->dr_values,
                                      e1->dr_count) == K26RL_OK);
    for (s = 0; s < e0->step_count; s++) {
        ASSERT(k26rl_episode_writer_step(w, 0, &e0->obs[s * FIX_OBS],
                                         &e0->act[s * FIX_ACT],
                                         &e0->rewards[s * FIX_AGENTS],
                                         e0->flags[s],
                                         e0->applied_dt[s]) == K26RL_OK);
        if (s < e1->step_count)
            ASSERT(k26rl_episode_writer_step(w, 1, &e1->obs[s * FIX_OBS],
                                             &e1->act[s * FIX_ACT],
                                             &e1->rewards[s * FIX_AGENTS],
                                             e1->flags[s],
                                             e1->applied_dt[s]) == K26RL_OK);
    }
    ASSERT(k26rl_episode_writer_end(w, 0, e0->end_reason, e0->fault_code,
                                    e0->terminal_adjustments) == K26RL_OK);
    ASSERT(k26rl_episode_writer_end(w, 1, e1->end_reason, e1->fault_code,
                                    e1->terminal_adjustments) == K26RL_OK);
    ASSERT(k26rl_episode_writer_rekey(w, FIX_SEED2, &ord) == K26RL_OK);
    ASSERT(ord == 1);
    ASSERT(k26rl_episode_writer_start(w, 0, 0, e2->initial_obs, NULL, NULL,
                                      0) == K26RL_OK);
    for (s = 0; s < e2->step_count; s++)
        ASSERT(k26rl_episode_writer_step(w, 0, &e2->obs[s * FIX_OBS],
                                         &e2->act[s * FIX_ACT],
                                         &e2->rewards[s * FIX_AGENTS],
                                         e2->flags[s],
                                         e2->applied_dt[s]) == K26RL_OK);
    ASSERT(k26rl_episode_writer_end(w, 0, e2->end_reason, e2->fault_code,
                                    e2->terminal_adjustments) == K26RL_OK);
    ASSERT(k26rl_episode_writer_close(w) == K26RL_OK);
}

/* Bitwise comparison of one decoded episode against what was fed. */
static inline void fix_check_episode_(const Fixture *fx, uint32_t idx,
                                      const K26RlEpisodeData *d)
{
    const FixEpisode *e = &fx->eps[idx];

    ASSERT(d->rekey_ordinal == e->ordinal);
    ASSERT(d->env == e->env);
    ASSERT(d->episode == e->episode);
    ASSERT(d->step_count == e->step_count);
    ASSERT(d->end_reason == e->end_reason);
    ASSERT(d->fault_code == e->fault_code);
    ASSERT(memcmp(d->initial_obs, e->initial_obs,
                  FIX_OBS * sizeof(double)) == 0);
    ASSERT(d->dr_count == e->dr_count);
    if (e->dr_count) {
        ASSERT(memcmp(d->dr_tags, e->dr_tags,
                      e->dr_count * sizeof(uint32_t)) == 0);
        ASSERT(memcmp(d->dr_values, e->dr_values,
                      e->dr_count * sizeof(double)) == 0);
    }
    ASSERT(memcmp(d->obs, e->obs,
                  (size_t)e->step_count * FIX_OBS * sizeof(double)) == 0);
    ASSERT(memcmp(d->act, e->act,
                  (size_t)e->step_count * FIX_ACT * sizeof(double)) == 0);
    ASSERT(memcmp(d->rewards, e->rewards,
                  (size_t)e->step_count * FIX_AGENTS * sizeof(double)) == 0);
    ASSERT(memcmp(d->flags, e->flags,
                  (size_t)e->step_count * sizeof(uint32_t)) == 0);
    ASSERT(memcmp(d->applied_dt, e->applied_dt,
                  (size_t)e->step_count * sizeof(double)) == 0);
    ASSERT(memcmp(d->terminal_adjustments, e->terminal_adjustments,
                  FIX_AGENTS * sizeof(double)) == 0);
}

/* Scratch paths under TMPDIR (or /tmp), distinct per process. */
static inline void fix_path_(char *out, size_t cap, const char *name)
{
    const char *dir = getenv("TMPDIR");

    if (!dir || !dir[0])
        dir = "/tmp";
    snprintf(out, cap, "%s/k26rl_%ld_%s", dir, (long)getpid(), name);
}

static inline uint8_t *fix_read_file_(const char *path, uint64_t *out_size)
{
    FILE *f = fopen(path, "rb");
    long sz;
    uint8_t *buf;

    ASSERT(f != NULL);
    ASSERT(fseek(f, 0, SEEK_END) == 0);
    sz = ftell(f);
    ASSERT(sz >= 0);
    buf = malloc(sz ? (size_t)sz : 1);
    ASSERT(buf != NULL);
    ASSERT(fseek(f, 0, SEEK_SET) == 0);
    ASSERT(fread(buf, 1, (size_t)sz, f) == (size_t)sz);
    ASSERT(fclose(f) == 0);
    *out_size = (uint64_t)sz;
    return buf;
}

#endif /* EPISODE_FIXTURE_H */
