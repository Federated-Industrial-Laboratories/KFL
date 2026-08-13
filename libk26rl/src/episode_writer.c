/* episode_writer.c - buffered writer for the episode file format.
 *
 * Layouts follow k26rl_episode.h: every integer is assembled
 * little-endian field by field, doubles travel as their binary64 bit
 * patterns, and every frame carries the 24-byte CRC-framed header.
 * Every buffer touched between open and close is allocated at open,
 * once: the per-step path copies into the environment's chunk buffer,
 * episode starts and ends build in the environment's boundary
 * scratch, and each finished frame leaves in a single fwrite. The
 * index is assembled at close by one sequential pass over the file
 * this writer wrote, so no bookkeeping grows while running. All
 * output is a pure function of the call sequence, so identical
 * sequences yield byte-identical files. */
#include "k26rl_episode.h"
#include "k26rl_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int open;
    uint32_t ordinal;       /* rekey ordinal captured at episode start */
    uint32_t episode;
    uint32_t steps_total;
    uint32_t chunk_first;   /* first step index of the buffered chunk */
    uint32_t chunk_fill;
    uint8_t *buf;           /* chunk frame image: header plus payload */
    uint8_t *boundary;      /* start and end frame image scratch */
} EnvState_;

struct K26RlEpisodeWriter {
    FILE *f;
    K26RlEpisodeGeom geom;
    uint64_t offset;        /* absolute file offset of the next byte */
    uint64_t sequence;      /* next frame sequence number */
    uint32_t rekey_ordinal;
    uint64_t ncols8;        /* 8-byte columns per chunk: obs+act+agents */
    EnvState_ *envs;
};

/* Emit one frame whose payload already sits at buf + 24. The header
 * is written here with the crc field zero, the CRC is computed over
 * the whole image as the format requires, patched in, and the frame
 * leaves in a single fwrite. */
static K26RlStatus emit_buf_(K26RlEpisodeWriter *w, uint16_t kind,
                             uint8_t *buf, uint32_t payload_len,
                             uint64_t *out_offset)
{
    size_t total = (size_t)K26RL_EPISODE_FRAME_HEADER_SIZE + payload_len;
    uint32_t crc;

    k26rl_frame_header_write_(buf, kind, payload_len, w->sequence);
    crc = k26rl_crc32c(K26RL_CRC32C_INIT, buf, total);
    k26rl_put_u32_(buf + K26RL_FH_OFF_CRC, crc);
    if (out_offset)
        *out_offset = w->offset;
    if (fwrite(buf, 1, total, w->f) != total)
        return K26RL_E_INTERNAL;
    w->sequence++;
    w->offset += total;
    return K26RL_OK;
}

/* Flush an environment's buffered chunk, if any. Columns are laid
 * out at full-chunk stride while filling so the step path is a plain
 * strided copy; a short final chunk is compacted in place here,
 * outside the step path. Moves run in ascending region order and
 * every destination starts at or before its source, so memmove per
 * region cannot clobber unread bytes. */
static K26RlStatus flush_chunk_(K26RlEpisodeWriter *w, uint32_t env)
{
    EnvState_ *e = &w->envs[env];
    const uint64_t C = w->geom.steps_per_chunk;
    const uint64_t K = e->chunk_fill;
    uint8_t *pay = e->buf + K26RL_EPISODE_FRAME_HEADER_SIZE;
    uint32_t plen;

    if (K == 0)
        return K26RL_OK;
    if (K < C) {
        uint64_t c;
        for (c = 0; c < w->ncols8; c++)
            memmove(pay + 16 + c * K * 8, pay + 16 + c * C * 8,
                    (size_t)(K * 8));
        memmove(pay + 16 + w->ncols8 * K * 8, pay + 16 + w->ncols8 * C * 8,
                (size_t)(K * 4));
        memmove(pay + 16 + w->ncols8 * K * 8 + K * 4,
                pay + 16 + w->ncols8 * C * 8 + C * 4, (size_t)(K * 8));
    }
    k26rl_put_u32_(pay, e->ordinal);
    k26rl_put_u32_(pay + 4, env);
    k26rl_put_u32_(pay + 8, e->chunk_first);
    k26rl_put_u32_(pay + 12, (uint32_t)K);
    plen = (uint32_t)(16 + K * (w->ncols8 * 8 + 12));
    e->chunk_first += (uint32_t)K;
    e->chunk_fill = 0;
    return emit_buf_(w, K26RL_FRAME_STEP_CHUNK, e->buf, plen, NULL);
}

K26RlStatus k26rl_episode_writer_open(const char *path,
                                      const K26RlEpisodeGeom *geom,
                                      uint64_t governing_seed,
                                      uint32_t rekey_ordinal,
                                      const char *grammar_version,
                                      const char *runtime_version,
                                      const uint8_t *spec, uint32_t spec_len,
                                      K26RlEpisodeWriter **out)
{
    K26RlEpisodeWriter *w = NULL;
    FILE *f;
    int fd;
    size_t glen, rlen;
    uint64_t chunk_payload, start_payload, end_payload, boundary_size;
    uint64_t plen64;
    uint32_t i;
    K26RlStatus st = K26RL_E_INTERNAL;

    if (!path || !geom || !grammar_version || !runtime_version || !out)
        return K26RL_E_NULL;
    if (spec_len && !spec)
        return K26RL_E_NULL;
    *out = NULL;
    if (geom->n_envs == 0 || geom->steps_per_chunk == 0)
        return K26RL_E_GEOMETRY;
    glen = strlen(grammar_version);
    rlen = strlen(runtime_version);
    if (glen > 0xFFFF || rlen > 0xFFFF)
        return K26RL_E_GEOMETRY;
    chunk_payload = 16 + (uint64_t)geom->steps_per_chunk *
        (((uint64_t)geom->obs_total + geom->act_total +
          geom->agent_count) * 8 + 12);
    /* The boundary scratch holds one frame image, sized for the
     * larger of an episode-start carrying dr_max pairs and an
     * episode-end, so starts and ends build in place and allocate
     * nothing. */
    start_payload = 16 + (uint64_t)geom->obs_total * 8 +
        (uint64_t)geom->dr_max * 12;
    end_payload = 20 + (uint64_t)geom->agent_count * 8;
    if (chunk_payload > UINT32_MAX || start_payload > UINT32_MAX ||
        end_payload > UINT32_MAX)
        return K26RL_E_GEOMETRY;
    boundary_size = K26RL_EPISODE_FRAME_HEADER_SIZE +
        (start_payload > end_payload ? start_payload : end_payload);

    /* O_EXCL makes creation the existence check: nothing truncates a
     * completed run and no race window separates test from create.
     * Read-write because close re-reads the file to assemble the
     * index. */
    fd = open(path, O_RDWR | O_CREAT | O_EXCL, 0644);
    if (fd < 0)
        return errno == EEXIST ? K26RL_E_OUTPUT_EXISTS : K26RL_E_INTERNAL;
    f = fdopen(fd, "w+b");
    if (!f) {
        close(fd);
        remove(path);
        return K26RL_E_INTERNAL;
    }

    w = calloc(1, sizeof *w);
    if (!w)
        goto fail;
    w->f = f;
    w->geom = *geom;
    w->rekey_ordinal = rekey_ordinal;
    w->ncols8 = (uint64_t)geom->obs_total + geom->act_total +
                geom->agent_count;
    w->envs = calloc(geom->n_envs, sizeof *w->envs);
    if (!w->envs)
        goto fail;
    for (i = 0; i < geom->n_envs; i++) {
        w->envs[i].buf = malloc((size_t)K26RL_EPISODE_FRAME_HEADER_SIZE +
                                (size_t)chunk_payload);
        w->envs[i].boundary = malloc((size_t)boundary_size);
        if (!w->envs[i].buf || !w->envs[i].boundary)
            goto fail;
    }

    if (fwrite(K26RL_EPISODE_MAGIC, 1, 8, f) != 8)
        goto fail;
    w->offset = 8;

    plen64 = 24 + 2 + (uint64_t)glen + 2 + (uint64_t)rlen + 4 + spec_len;
    if (plen64 > UINT32_MAX) {
        st = K26RL_E_GEOMETRY;
        goto fail;
    }
    {
        uint32_t plen = (uint32_t)plen64;
        uint8_t *frame =
            malloc((size_t)K26RL_EPISODE_FRAME_HEADER_SIZE + plen);
        uint8_t *p;
        if (!frame)
            goto fail;
        p = frame + K26RL_EPISODE_FRAME_HEADER_SIZE;
        k26rl_put_u32_(p, K26RL_EPISODE_FORMAT_VERSION);
        k26rl_put_u64_(p + 4, governing_seed);
        k26rl_put_u32_(p + 12, rekey_ordinal);
        k26rl_put_u32_(p + 16, geom->n_envs);
        k26rl_put_u32_(p + 20, geom->steps_per_chunk);
        p += 24;
        k26rl_put_u16_(p, (uint16_t)glen);
        p += 2;
        memcpy(p, grammar_version, glen);
        p += glen;
        k26rl_put_u16_(p, (uint16_t)rlen);
        p += 2;
        memcpy(p, runtime_version, rlen);
        p += rlen;
        k26rl_put_u32_(p, spec_len);
        p += 4;
        if (spec_len)
            memcpy(p, spec, spec_len);
        st = emit_buf_(w, K26RL_FRAME_FILE_HEADER, frame, plen, NULL);
        free(frame);
        if (st != K26RL_OK)
            goto fail;
    }
    *out = w;
    return K26RL_OK;

fail:
    /* Remove the just-created file so a retry is not refused over a
     * path this failed call itself created. */
    if (w) {
        if (w->envs) {
            for (i = 0; i < geom->n_envs; i++) {
                free(w->envs[i].buf);
                free(w->envs[i].boundary);
            }
            free(w->envs);
        }
        free(w);
    }
    fclose(f);
    remove(path);
    return st;
}

K26RlStatus k26rl_episode_writer_start(K26RlEpisodeWriter *w,
                                       uint32_t env, uint32_t episode,
                                       const double *initial_obs,
                                       const uint32_t *dr_tags,
                                       const double *dr_values,
                                       uint32_t dr_count)
{
    EnvState_ *e;
    uint8_t *p;
    uint32_t plen, j;
    K26RlStatus st;

    if (!w)
        return K26RL_E_NULL;
    if (w->geom.obs_total && !initial_obs)
        return K26RL_E_NULL;
    if (dr_count && (!dr_tags || !dr_values))
        return K26RL_E_NULL;
    if (env >= w->geom.n_envs)
        return K26RL_E_GEOMETRY;
    /* The boundary scratch was sized for dr_max pairs at open; a
     * larger record cannot be built without allocating, so it is
     * refused as a geometry disagreement. */
    if (dr_count > w->geom.dr_max)
        return K26RL_E_GEOMETRY;
    e = &w->envs[env];
    if (e->open)
        return K26RL_E_OUTPUT_TIMING;

    /* Open bounded the dr_max payload to a uint32 and dr_count is
     * within dr_max, so this cannot overflow. */
    plen = (uint32_t)(16 + (uint64_t)w->geom.obs_total * 8 +
                      (uint64_t)dr_count * 12);
    p = e->boundary + K26RL_EPISODE_FRAME_HEADER_SIZE;
    k26rl_put_u32_(p, w->rekey_ordinal);
    k26rl_put_u32_(p + 4, env);
    k26rl_put_u32_(p + 8, episode);
    p += 12;
    for (j = 0; j < w->geom.obs_total; j++, p += 8)
        k26rl_put_f64_(p, initial_obs[j]);
    k26rl_put_u32_(p, dr_count);
    p += 4;
    for (j = 0; j < dr_count; j++, p += 12) {
        k26rl_put_u32_(p, dr_tags[j]);
        k26rl_put_f64_(p + 4, dr_values[j]);
    }
    st = emit_buf_(w, K26RL_FRAME_EPISODE_START, e->boundary, plen, NULL);
    if (st != K26RL_OK)
        return st;

    e->open = 1;
    e->ordinal = w->rekey_ordinal;
    e->episode = episode;
    e->steps_total = 0;
    e->chunk_first = 0;
    e->chunk_fill = 0;
    return K26RL_OK;
}

K26RlStatus k26rl_episode_writer_step(K26RlEpisodeWriter *w, uint32_t env,
                                      const double *obs, const double *act,
                                      const double *rewards, uint32_t flags,
                                      double applied_dt)
{
    EnvState_ *e;
    uint8_t *cols;
    uint64_t C, i;
    uint32_t j;

    if (!w)
        return K26RL_E_NULL;
    if ((w->geom.obs_total && !obs) || (w->geom.act_total && !act) ||
        (w->geom.agent_count && !rewards))
        return K26RL_E_NULL;
    if (env >= w->geom.n_envs)
        return K26RL_E_GEOMETRY;
    e = &w->envs[env];
    if (!e->open)
        return K26RL_E_OUTPUT_TIMING;

    C = w->geom.steps_per_chunk;
    i = e->chunk_fill;
    cols = e->buf + K26RL_EPISODE_FRAME_HEADER_SIZE + 16;
    for (j = 0; j < w->geom.obs_total; j++)
        k26rl_put_f64_(cols + (j * C + i) * 8, obs[j]);
    for (j = 0; j < w->geom.act_total; j++)
        k26rl_put_f64_(cols + ((w->geom.obs_total + j) * C + i) * 8, act[j]);
    for (j = 0; j < w->geom.agent_count; j++)
        k26rl_put_f64_(cols + (((uint64_t)w->geom.obs_total +
                                w->geom.act_total + j) * C + i) * 8,
                       rewards[j]);
    k26rl_put_u32_(cols + w->ncols8 * C * 8 + i * 4, flags);
    k26rl_put_f64_(cols + w->ncols8 * C * 8 + C * 4 + i * 8, applied_dt);
    e->chunk_fill++;
    e->steps_total++;
    if (e->chunk_fill == w->geom.steps_per_chunk)
        return flush_chunk_(w, env);
    return K26RL_OK;
}

K26RlStatus k26rl_episode_writer_end(K26RlEpisodeWriter *w, uint32_t env,
                                     uint16_t end_reason, uint16_t fault_code,
                                     const double *terminal_adjustments)
{
    EnvState_ *e;
    uint8_t *p;
    uint32_t plen, j;
    K26RlStatus st;

    if (!w)
        return K26RL_E_NULL;
    if (w->geom.agent_count && !terminal_adjustments)
        return K26RL_E_NULL;
    if (env >= w->geom.n_envs)
        return K26RL_E_GEOMETRY;
    e = &w->envs[env];
    if (!e->open)
        return K26RL_E_OUTPUT_TIMING;

    st = flush_chunk_(w, env);
    if (st != K26RL_OK)
        return st;

    plen = (uint32_t)(20 + (uint64_t)w->geom.agent_count * 8);
    p = e->boundary + K26RL_EPISODE_FRAME_HEADER_SIZE;
    k26rl_put_u32_(p, e->ordinal);
    k26rl_put_u32_(p + 4, env);
    k26rl_put_u32_(p + 8, e->episode);
    k26rl_put_u32_(p + 12, e->steps_total);
    k26rl_put_u16_(p + 16, end_reason);
    k26rl_put_u16_(p + 18, fault_code);
    p += 20;
    for (j = 0; j < w->geom.agent_count; j++, p += 8)
        k26rl_put_f64_(p, terminal_adjustments[j]);
    st = emit_buf_(w, K26RL_FRAME_EPISODE_END, e->boundary, plen, NULL);
    if (st != K26RL_OK)
        return st;

    e->open = 0;
    return K26RL_OK;
}

K26RlStatus k26rl_episode_writer_rekey(K26RlEpisodeWriter *w,
                                       uint64_t new_seed,
                                       uint32_t *out_ordinal)
{
    uint8_t frame[K26RL_EPISODE_FRAME_HEADER_SIZE + 12];
    uint32_t i;
    K26RlStatus st;

    if (!w)
        return K26RL_E_NULL;
    /* Episode identity is (ordinal, env, episode) and indices restart
     * at zero under the new key, so a rekey inside an open episode
     * would split its identity. The producing surface rekeys through
     * a seeded reset, which sits at an episode boundary for every
     * environment, so the same boundary is required here. */
    for (i = 0; i < w->geom.n_envs; i++)
        if (w->envs[i].open)
            return K26RL_E_OUTPUT_TIMING;

    k26rl_put_u64_(frame + K26RL_EPISODE_FRAME_HEADER_SIZE, new_seed);
    k26rl_put_u32_(frame + K26RL_EPISODE_FRAME_HEADER_SIZE + 8,
                   w->rekey_ordinal + 1);
    st = emit_buf_(w, K26RL_FRAME_REKEY, frame, 12, NULL);
    if (st != K26RL_OK)
        return st;
    w->rekey_ordinal++;
    if (out_ordinal)
        *out_ordinal = w->rekey_ordinal;
    return K26RL_OK;
}

/* Close-time index assembly. Close is not the hot path: the entry
 * and offset arrays here grow freely. */
typedef struct {
    uint32_t ordinal;
    uint32_t env;
    uint32_t episode;
    uint32_t chunk_count;
    uint64_t start_off;
    uint64_t end_off;
    uint64_t *chunk_offs;
} IndexEntry_;

typedef struct {
    int open;
    uint32_t ordinal;
    uint32_t episode;
    uint64_t start_off;
    uint64_t *chunk_offs;
    uint32_t chunk_count;
    uint32_t chunk_cap;
} Pending_;

static int u64_append_(uint64_t **arr, uint32_t *count, uint32_t *cap,
                       uint64_t v)
{
    if (*count == *cap) {
        uint32_t ncap = *cap ? *cap * 2 : 8;
        uint64_t *n = realloc(*arr, (size_t)ncap * sizeof **arr);
        if (!n)
            return -1;
        *arr = n;
        *cap = ncap;
    }
    (*arr)[(*count)++] = v;
    return 0;
}

static int index_append_(IndexEntry_ **arr, uint32_t *count, uint32_t *cap,
                         const IndexEntry_ *e)
{
    if (*count == *cap) {
        uint32_t ncap = *cap ? *cap * 2 : 8;
        IndexEntry_ *n = realloc(*arr, (size_t)ncap * sizeof *n);
        if (!n)
            return -1;
        *arr = n;
        *cap = ncap;
    }
    (*arr)[(*count)++] = *e;
    return 0;
}

static void index_free_(IndexEntry_ *idx, uint32_t count)
{
    uint32_t i;

    if (!idx)
        return;
    for (i = 0; i < count; i++)
        free(idx[i].chunk_offs);
    free(idx);
}

/* One sequential pass over the frames this writer wrote, collecting
 * per-episode start, chunk, and end offsets for episodes whose start
 * and end are both present. The caller flushed the stream and every
 * byte below w->offset was written by this process moments ago, so
 * the walk trusts header arithmetic, skips payloads by length, and
 * reads only the identity prefix of the three episode frame kinds;
 * any disagreement with what this writer emits is an internal
 * failure, not a parse decision. */
static K26RlStatus build_index_(K26RlEpisodeWriter *w, IndexEntry_ **out_idx,
                                uint32_t *out_count)
{
    uint8_t hdr[K26RL_EPISODE_FRAME_HEADER_SIZE];
    uint8_t pfx[12];
    Pending_ *pend;
    IndexEntry_ *idx = NULL;
    uint32_t idx_count = 0, idx_cap = 0;
    uint64_t off = 8;
    uint32_t i;
    K26RlStatus st = K26RL_OK;

    *out_idx = NULL;
    *out_count = 0;
    pend = calloc(w->geom.n_envs, sizeof *pend);
    if (!pend)
        return K26RL_E_INTERNAL;

    while (st == K26RL_OK &&
           off + K26RL_EPISODE_FRAME_HEADER_SIZE <= w->offset) {
        uint16_t kind;
        uint32_t plen;

        if (fseeko(w->f, (off_t)off, SEEK_SET) != 0 ||
            fread(hdr, 1, sizeof hdr, w->f) != sizeof hdr) {
            st = K26RL_E_INTERNAL;
            break;
        }
        kind = k26rl_get_u16_(hdr + K26RL_FH_OFF_KIND);
        plen = k26rl_get_u32_(hdr + K26RL_FH_OFF_LENGTH);
        if (plen > w->offset - off - K26RL_EPISODE_FRAME_HEADER_SIZE) {
            st = K26RL_E_INTERNAL;
            break;
        }
        if (kind == K26RL_FRAME_EPISODE_START ||
            kind == K26RL_FRAME_STEP_CHUNK ||
            kind == K26RL_FRAME_EPISODE_END) {
            uint32_t env;
            Pending_ *pe;

            if (plen < sizeof pfx ||
                fread(pfx, 1, sizeof pfx, w->f) != sizeof pfx) {
                st = K26RL_E_INTERNAL;
                break;
            }
            env = k26rl_get_u32_(pfx + 4);
            if (env >= w->geom.n_envs) {
                st = K26RL_E_INTERNAL;
                break;
            }
            pe = &pend[env];
            if (kind == K26RL_FRAME_EPISODE_START) {
                pe->open = 1;
                pe->ordinal = k26rl_get_u32_(pfx);
                pe->episode = k26rl_get_u32_(pfx + 8);
                pe->start_off = off;
                pe->chunk_count = 0;
            } else if (kind == K26RL_FRAME_STEP_CHUNK) {
                if (!pe->open ||
                    u64_append_(&pe->chunk_offs, &pe->chunk_count,
                                &pe->chunk_cap, off) != 0) {
                    st = K26RL_E_INTERNAL;
                    break;
                }
            } else {
                IndexEntry_ entry;

                if (!pe->open) {
                    st = K26RL_E_INTERNAL;
                    break;
                }
                entry.ordinal = pe->ordinal;
                entry.env = env;
                entry.episode = pe->episode;
                entry.chunk_count = pe->chunk_count;
                entry.start_off = pe->start_off;
                entry.end_off = off;
                entry.chunk_offs = pe->chunk_offs;
                if (index_append_(&idx, &idx_count, &idx_cap,
                                  &entry) != 0) {
                    st = K26RL_E_INTERNAL;
                    break;
                }
                /* The entry owns the offset list now. */
                pe->chunk_offs = NULL;
                pe->chunk_cap = 0;
                pe->chunk_count = 0;
                pe->open = 0;
            }
        }
        off += K26RL_EPISODE_FRAME_HEADER_SIZE + (uint64_t)plen;
    }

    for (i = 0; i < w->geom.n_envs; i++)
        free(pend[i].chunk_offs);
    free(pend);
    if (st != K26RL_OK) {
        index_free_(idx, idx_count);
        return st;
    }
    *out_idx = idx;
    *out_count = idx_count;
    return K26RL_OK;
}

K26RlStatus k26rl_episode_writer_close(K26RlEpisodeWriter *w)
{
    K26RlStatus st = K26RL_OK, s2;
    IndexEntry_ *idx = NULL;
    uint32_t idx_count = 0;
    uint64_t plen64, idx_off = 0;
    uint32_t i, j;

    if (!w)
        return K26RL_E_NULL;

    /* An episode still open at close is the crash-adjacent shape:
     * flush what is buffered so its frames stay readable on a
     * sequential pass, fabricate no episode-end, and leave it out of
     * the index. A consumer of the cleanly closed file sees complete
     * episodes in the index; the unfinished episode's frames remain
     * reachable by walking the file. */
    for (i = 0; i < w->geom.n_envs; i++) {
        if (w->envs[i].open) {
            s2 = flush_chunk_(w, i);
            if (s2 != K26RL_OK && st == K26RL_OK)
                st = s2;
        }
    }

    /* The index pass reads the file back, so buffered output must
     * reach the file first; after a write failure above w->offset no
     * longer matches the file, so no index is attempted over it. */
    if (st == K26RL_OK && fflush(w->f) != 0)
        st = K26RL_E_INTERNAL;
    if (st == K26RL_OK)
        st = build_index_(w, &idx, &idx_count);
    /* Reposition at the logical end: the pass moved the stream, and
     * an update stream requires a seek between the reads and the
     * writes that follow. */
    if (st == K26RL_OK && fseeko(w->f, (off_t)w->offset, SEEK_SET) != 0)
        st = K26RL_E_INTERNAL;

    if (st == K26RL_OK) {
        plen64 = 4;
        for (i = 0; i < idx_count; i++)
            plen64 += 32 + (uint64_t)idx[i].chunk_count * 8;
        if (plen64 > UINT32_MAX) {
            st = K26RL_E_INTERNAL;
        } else {
            uint32_t plen = (uint32_t)plen64;
            uint8_t *frame =
                malloc((size_t)K26RL_EPISODE_FRAME_HEADER_SIZE + plen);
            if (!frame) {
                st = K26RL_E_INTERNAL;
            } else {
                uint8_t *p = frame + K26RL_EPISODE_FRAME_HEADER_SIZE;
                k26rl_put_u32_(p, idx_count);
                p += 4;
                for (i = 0; i < idx_count; i++) {
                    const IndexEntry_ *e = &idx[i];
                    k26rl_put_u32_(p, e->ordinal);
                    k26rl_put_u32_(p + 4, e->env);
                    k26rl_put_u32_(p + 8, e->episode);
                    k26rl_put_u32_(p + 12, e->chunk_count);
                    k26rl_put_u64_(p + 16, e->start_off);
                    p += 24;
                    for (j = 0; j < e->chunk_count; j++, p += 8)
                        k26rl_put_u64_(p, e->chunk_offs[j]);
                    k26rl_put_u64_(p, e->end_off);
                    p += 8;
                }
                s2 = emit_buf_(w, K26RL_FRAME_INDEX, frame, plen, &idx_off);
                free(frame);
                if (s2 == K26RL_OK) {
                    uint8_t tr[16];
                    memcpy(tr, K26RL_EPISODE_TRAILER, 8);
                    k26rl_put_u64_(tr + 8, idx_off);
                    if (fwrite(tr, 1, sizeof tr, w->f) != sizeof tr)
                        s2 = K26RL_E_INTERNAL;
                }
                if (s2 != K26RL_OK)
                    st = s2;
            }
        }
    }

    if (fclose(w->f) != 0 && st == K26RL_OK)
        st = K26RL_E_INTERNAL;
    index_free_(idx, idx_count);
    for (i = 0; i < w->geom.n_envs; i++) {
        free(w->envs[i].buf);
        free(w->envs[i].boundary);
    }
    free(w->envs);
    free(w);
    return st;
}
