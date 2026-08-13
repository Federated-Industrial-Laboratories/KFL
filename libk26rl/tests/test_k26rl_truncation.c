/* test_k26rl_truncation.c - crash-shaped and corrupted files.
 *
 * Acceptance: a file cut at any frame boundary, mid-header, or
 * mid-payload still opens, reports clean_close 0 with readable_bytes
 * at the last complete frame, and yields exactly the episodes whose
 * frames are complete; a corrupted payload byte ends the readable
 * prefix before its frame; an unknown frame kind with a valid
 * checksum is skipped by length on the sequential pass. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#include "episode_fixture.h"

typedef struct {
    uint64_t start;
    uint64_t end;
    uint16_t kind;
    uint32_t plen;
} TFrame_;

static uint16_t le16_(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t le32_(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put16_(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)(v >> 8);
}

static void put32_(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)(v >> 24);
}

static void put64_(uint8_t *p, uint64_t v)
{
    put32_(p, (uint32_t)(v & 0xFFFFFFFFu));
    put32_(p + 4, (uint32_t)(v >> 32));
}

static void write_file_(const char *path, const uint8_t *buf, uint64_t n)
{
    FILE *f;

    remove(path);
    f = fopen(path, "wb");
    ASSERT(f != NULL);
    ASSERT(fwrite(buf, 1, (size_t)n, f) == (size_t)n);
    ASSERT(fclose(f) == 0);
}

/* Walk the frame stream in memory by header arithmetic alone, and
 * check that what follows the last frame is exactly the trailer. */
static uint32_t walk_(const uint8_t *buf, uint64_t size, TFrame_ *frames,
                      uint32_t max)
{
    uint64_t off = 8;
    uint32_t n = 0;

    while (off + 24 <= size) {
        uint32_t plen = le32_(buf + off + 4);
        if (off + 24 + plen > size)
            break;
        ASSERT(n < max);
        frames[n].start = off;
        frames[n].kind = le16_(buf + off);
        frames[n].plen = plen;
        frames[n].end = off + 24 + plen;
        off = frames[n].end;
        n++;
    }
    ASSERT(size - off == 16);
    ASSERT(memcmp(buf + off, K26RL_EPISODE_TRAILER, 8) == 0);
    return n;
}

/* Episodes complete at cut c: episode-end frames wholly inside the
 * prefix, in file order, so the first n identities of fix_order_. */
static uint32_t complete_at_(const TFrame_ *fr, uint32_t nf, uint64_t c)
{
    uint32_t i, n = 0;

    for (i = 0; i < nf; i++)
        if (fr[i].kind == K26RL_FRAME_EPISODE_END && fr[i].end <= c)
            n++;
    return n;
}

static uint64_t boundary_at_(const TFrame_ *fr, uint32_t nf, uint64_t c)
{
    uint64_t b = 8;
    uint32_t i;

    for (i = 0; i < nf; i++)
        if (fr[i].end <= c)
            b = fr[i].end;
    return b;
}

/* Open a cut or modified file and check the reported prefix and the
 * decodable episode set against expectation. */
static void check_prefix_(const Fixture *fx, const char *path,
                          uint64_t want_readable, uint32_t want_eps)
{
    K26RlEpisodeReader *r = NULL;
    K26RlEpisodeInfo info;
    K26RlEpisodeData d;
    uint32_t i, o, e, p;

    ASSERT(k26rl_episode_reader_open(path, &r) == K26RL_OK);
    ASSERT(k26rl_episode_reader_info(r, &info) == K26RL_OK);
    ASSERT(info.clean_close == 0);
    ASSERT(info.readable_bytes == want_readable);
    ASSERT(info.episode_count == want_eps);
    for (i = 0; i < want_eps; i++) {
        ASSERT(k26rl_episode_reader_at(r, i, &o, &e, &p) == K26RL_OK);
        ASSERT(o == fix_order_[i][0]);
        ASSERT(e == fix_order_[i][1]);
        ASSERT(p == fix_order_[i][2]);
        ASSERT(k26rl_episode_read(r, o, e, p, &d) == K26RL_OK);
        fix_check_episode_(fx, i, &d);
        k26rl_episode_free(&d);
    }
    ASSERT(k26rl_episode_reader_at(r, want_eps, &o, &e, &p) != K26RL_OK);
    k26rl_episode_reader_close(r);
}

int main(void)
{
    Fixture fx;
    char base[512], cut[512], mod[512];
    uint8_t *buf;
    uint64_t size, c;
    TFrame_ fr[32];
    uint32_t nf, i;

    fix_build_(&fx);
    fix_path_(base, sizeof base, "trunc_base.k26epi");
    fix_path_(cut, sizeof cut, "trunc_cut.k26epi");
    fix_path_(mod, sizeof mod, "trunc_mod.k26epi");
    remove(base);
    fix_write_(&fx, base);
    buf = fix_read_file_(base, &size);
    nf = walk_(buf, size, fr, 32);

    /* The writer's frame layout for this dataset is pinned: header,
     * two starts, the interleaved full chunks, env 0's short final
     * chunk, two ends, the rekey, then the faulted episode and the
     * index. */
    {
        static const uint16_t kinds[13] = {
            K26RL_FRAME_FILE_HEADER,
            K26RL_FRAME_EPISODE_START, K26RL_FRAME_EPISODE_START,
            K26RL_FRAME_STEP_CHUNK, K26RL_FRAME_STEP_CHUNK,
            K26RL_FRAME_STEP_CHUNK,
            K26RL_FRAME_EPISODE_END, K26RL_FRAME_EPISODE_END,
            K26RL_FRAME_REKEY,
            K26RL_FRAME_EPISODE_START, K26RL_FRAME_STEP_CHUNK,
            K26RL_FRAME_EPISODE_END,
            K26RL_FRAME_INDEX
        };
        ASSERT(nf == 13);
        for (i = 0; i < nf; i++)
            ASSERT(fr[i].kind == kinds[i]);
    }

    /* Cut at every frame boundary, from bare magic through the full
     * frame stream with the trailer gone. */
    for (i = 0; i <= nf; i++) {
        c = (i == 0) ? 8 : fr[i - 1].end;
        write_file_(cut, buf, c);
        check_prefix_(&fx, cut, c, complete_at_(fr, nf, c));
    }

    /* A mid-header cut: the readable prefix ends at the previous
     * frame boundary. */
    c = fr[4].start + 10;
    write_file_(cut, buf, c);
    check_prefix_(&fx, cut, boundary_at_(fr, nf, c),
                  complete_at_(fr, nf, boundary_at_(fr, nf, c)));

    /* A mid-payload cut. */
    c = fr[3].start + 24 + fr[3].plen / 2;
    write_file_(cut, buf, c);
    check_prefix_(&fx, cut, boundary_at_(fr, nf, c),
                  complete_at_(fr, nf, boundary_at_(fr, nf, c)));

    /* One corrupted payload byte in a middle frame of a full copy:
     * the readable prefix ends before that frame even though the
     * trailer is intact. The rekey frame is corrupted, so ordinal 0
     * still resolves and ordinal 1 must not. */
    {
        uint8_t *b2 = malloc((size_t)size);
        K26RlEpisodeReader *r = NULL;
        K26RlEpisodeData d;
        uint64_t seed;

        ASSERT(b2 != NULL);
        memcpy(b2, buf, (size_t)size);
        b2[fr[8].start + 24] ^= 0x01;
        write_file_(mod, b2, size);
        check_prefix_(&fx, mod, fr[8].start,
                      complete_at_(fr, nf, fr[8].start));
        ASSERT(k26rl_episode_reader_open(mod, &r) == K26RL_OK);
        ASSERT(k26rl_episode_reader_seed(r, 0, &seed) == K26RL_OK);
        ASSERT(seed == FIX_SEED);
        ASSERT(k26rl_episode_reader_seed(r, 1, &seed) != K26RL_OK);
        ASSERT(k26rl_episode_read(r, 1, 0, 0, &d) != K26RL_OK);
        k26rl_episode_reader_close(r);
        free(b2);
    }

    /* An unknown frame kind with a valid checksum, inserted mid-file
     * through a helper that mimics the writer's framing, is skipped
     * by length on the sequential pass: every episode stays
     * reachable. The insertion shifts the real frames, so the
     * trailer's stored index offset goes stale and the open must fall
     * back to reconstruction. */
    {
        uint8_t syn[24 + 5];
        uint64_t nsize = size + sizeof syn;
        uint8_t *b3 = malloc((size_t)nsize);
        K26RlEpisodeReader *r = NULL;
        uint64_t seed;
        uint32_t crc;

        ASSERT(b3 != NULL);
        put16_(syn + 0, 0x7FFF);      /* unassigned kind */
        put16_(syn + 2, 0);           /* flags */
        put32_(syn + 4, 5);           /* payload length */
        put64_(syn + 8, 12345);       /* sequence, ignored by readers */
        put32_(syn + 16, 0);          /* crc, patched below */
        put32_(syn + 20, 0);          /* reserved */
        syn[24] = 0xDE;
        syn[25] = 0xAD;
        syn[26] = 0xBE;
        syn[27] = 0xEF;
        syn[28] = 0x5A;
        crc = k26rl_crc32c(K26RL_CRC32C_INIT, syn, sizeof syn);
        put32_(syn + 16, crc);

        memcpy(b3, buf, (size_t)fr[8].start);
        memcpy(b3 + fr[8].start, syn, sizeof syn);
        memcpy(b3 + fr[8].start + sizeof syn, buf + fr[8].start,
               (size_t)(size - fr[8].start));
        write_file_(mod, b3, nsize);
        check_prefix_(&fx, mod, nsize - 16, FIX_EP_COUNT);
        ASSERT(k26rl_episode_reader_open(mod, &r) == K26RL_OK);
        ASSERT(k26rl_episode_reader_seed(r, 0, &seed) == K26RL_OK);
        ASSERT(seed == FIX_SEED);
        ASSERT(k26rl_episode_reader_seed(r, 1, &seed) == K26RL_OK);
        ASSERT(seed == FIX_SEED2);
        k26rl_episode_reader_close(r);
        free(b3);
    }

    free(buf);
    remove(base);
    remove(cut);
    remove(mod);
    printf("test_k26rl_truncation: ok\n");
    return 0;
}
