/* test_k26rl_roundtrip.c - write, reopen, and decode the scripted set.
 *
 * Acceptance: the writer is byte-deterministic (identical call
 * sequences yield byte-identical files); an existing path is refused;
 * a clean reopen reports the recorded geometry, clean close, and
 * episode count; every decoded value is bitwise identical to what was
 * fed; both rekey ordinals resolve to their seeds; enumeration covers
 * the index and a missing identity refuses. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#include "episode_fixture.h"

int main(void)
{
    Fixture fx;
    char pa[512], pb[512];
    uint8_t *ba, *bb;
    uint64_t sa, sb, seed;
    K26RlEpisodeReader *r = NULL;
    K26RlEpisodeInfo info;
    K26RlEpisodeData d;
    const uint8_t *spec;
    uint32_t spec_len, i, o, e, p;

    fix_build_(&fx);
    fix_path_(pa, sizeof pa, "roundtrip_a.k26epi");
    fix_path_(pb, sizeof pb, "roundtrip_b.k26epi");
    remove(pa);
    remove(pb);

    /* (a) identical call sequences yield byte-identical files. */
    fix_write_(&fx, pa);
    fix_write_(&fx, pb);
    ba = fix_read_file_(pa, &sa);
    bb = fix_read_file_(pb, &sb);
    ASSERT(sa == sb);
    ASSERT(memcmp(ba, bb, (size_t)sa) == 0);
    free(ba);
    free(bb);

    /* An existing path is refused. */
    {
        K26RlEpisodeWriter *w2 = NULL;
        K26RlEpisodeGeom g;
        g.n_envs = FIX_N_ENVS;
        g.agent_count = FIX_AGENTS;
        g.obs_total = FIX_OBS;
        g.act_total = FIX_ACT;
        g.steps_per_chunk = FIX_CHUNK;
        ASSERT(k26rl_episode_writer_open(pa, &g, FIX_SEED, 0, "3.2",
                                         "kfl-rl-0.1", fx.spec, fx.spec_len,
                                         &w2) == K26RL_E_OUTPUT_EXISTS);
        ASSERT(w2 == NULL);
    }

    /* (b) reopen: recorded fields, clean close, episode count. */
    ASSERT(k26rl_episode_reader_open(pa, &r) == K26RL_OK);
    ASSERT(k26rl_episode_reader_info(r, &info) == K26RL_OK);
    ASSERT(info.format_version == K26RL_EPISODE_FORMAT_VERSION);
    ASSERT(info.governing_seed == FIX_SEED);
    ASSERT(info.rekey_ordinal == 0);
    ASSERT(info.n_envs == FIX_N_ENVS);
    ASSERT(info.steps_per_chunk == FIX_CHUNK);
    ASSERT(info.agent_count == FIX_AGENTS);
    ASSERT(info.obs_total == FIX_OBS);
    ASSERT(info.act_total == FIX_ACT);
    ASSERT(info.clean_close == 1);
    ASSERT(info.episode_count == FIX_EP_COUNT);

    /* The embedded spec blob is verbatim. */
    ASSERT(k26rl_episode_reader_spec(r, &spec, &spec_len) == K26RL_OK);
    ASSERT(spec_len == fx.spec_len);
    ASSERT(memcmp(spec, fx.spec, spec_len) == 0);

    /* (c) every decoded value is bitwise what was fed. */
    for (i = 0; i < FIX_EP_COUNT; i++) {
        ASSERT(k26rl_episode_read(r, fix_order_[i][0], fix_order_[i][1],
                                  fix_order_[i][2], &d) == K26RL_OK);
        fix_check_episode_(&fx, i, &d);
        k26rl_episode_free(&d);
        ASSERT(d.obs == NULL);
        ASSERT(d.step_count == 0);
    }

    /* (d) both rekey ordinals resolve to their seeds. */
    ASSERT(k26rl_episode_reader_seed(r, 0, &seed) == K26RL_OK);
    ASSERT(seed == FIX_SEED);
    ASSERT(k26rl_episode_reader_seed(r, 1, &seed) == K26RL_OK);
    ASSERT(seed == FIX_SEED2);
    ASSERT(k26rl_episode_reader_seed(r, 7, &seed) != K26RL_OK);

    /* (e) enumeration covers all three, in file order. */
    for (i = 0; i < FIX_EP_COUNT; i++) {
        ASSERT(k26rl_episode_reader_at(r, i, &o, &e, &p) == K26RL_OK);
        ASSERT(o == fix_order_[i][0]);
        ASSERT(e == fix_order_[i][1]);
        ASSERT(p == fix_order_[i][2]);
    }
    ASSERT(k26rl_episode_reader_at(r, FIX_EP_COUNT, &o, &e, &p) != K26RL_OK);

    /* (f) a missing identity triple refuses. */
    ASSERT(k26rl_episode_read(r, 0, 1, 5, &d) != K26RL_OK);
    ASSERT(k26rl_episode_read(r, 2, 0, 0, &d) != K26RL_OK);

    k26rl_episode_reader_close(r);
    remove(pa);
    remove(pb);
    printf("test_k26rl_roundtrip: ok\n");
    return 0;
}
