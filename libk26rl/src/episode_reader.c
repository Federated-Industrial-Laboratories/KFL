/* episode_reader.c - index-directed reader for the episode file format.
 *
 * Open verifies the whole frame stream. The trailer, when valid,
 * authenticates the index frame, but corruption anywhere in the file
 * must surface as a shortened readable prefix rather than a clean
 * report, and rekey seed resolution needs the mid-file rekey frames
 * the index does not carry, so every open runs the sequential CRC
 * pass. clean_close is set only when the trailer located a valid
 * index and the pass reached it intact. Only a CRC failure or header
 * arithmetic running past end of file ends the readable prefix; a
 * CRC-valid frame that does not parse is skipped by length like an
 * unknown kind.
 *
 * Status mapping on this read-only surface: an identity not present
 * in the file (an unknown episode triple, an out-of-range enumeration
 * index, an unrecorded rekey ordinal, a spec asked of a file with no
 * file-header frame) refuses with K26RL_E_GEOMETRY; I/O failures and
 * corruption refuse with K26RL_E_INTERNAL. */
#include "k26rl_episode.h"
#include "k26rl_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t ordinal;
    uint32_t env;
    uint32_t episode;
    uint32_t chunk_count;
    uint64_t start_off;
    uint64_t end_off;
    uint64_t *chunk_offs;
} EpEntry_;

typedef struct {
    uint32_t ordinal;
    uint64_t seed;
} KeyEntry_;

struct K26RlEpisodeReader {
    FILE *f;
    uint64_t file_size;
    int have_header;
    K26RlEpisodeInfo info;
    uint64_t ncols8;        /* 8-byte columns per chunk: obs+act+agents */
    uint8_t *spec;
    uint32_t spec_len;
    KeyEntry_ *keys;
    uint32_t key_count;
    uint32_t key_cap;
    EpEntry_ *eps;
    uint32_t ep_count;
    uint32_t ep_cap;
};

typedef struct {
    uint8_t *p;
    uint64_t cap;
} Scratch_;

static int scratch_reserve_(Scratch_ *s, uint64_t n)
{
    uint8_t *np;
    uint64_t ncap;

    if (n <= s->cap)
        return 0;
    ncap = s->cap ? s->cap : 256;
    while (ncap < n)
        ncap *= 2;
    np = realloc(s->p, (size_t)ncap);
    if (!np)
        return -1;
    s->p = np;
    s->cap = ncap;
    return 0;
}

/* Overflow-safe array allocation; a zero-byte request still yields a
 * distinct block so freshly decoded arrays are always non-null. */
static void *alloc_n_(uint64_t count, uint64_t elem)
{
    uint64_t total;

    if (elem && count > UINT64_MAX / elem)
        return NULL;
    total = count * elem;
    return malloc(total ? (size_t)total : 1);
}

static void free_entries_(EpEntry_ *eps, uint32_t count)
{
    uint32_t i;

    if (!eps)
        return;
    for (i = 0; i < count; i++)
        free(eps[i].chunk_offs);
    free(eps);
}

static int eps_append_(K26RlEpisodeReader *r, const EpEntry_ *e)
{
    if (r->ep_count == r->ep_cap) {
        uint32_t ncap = r->ep_cap ? r->ep_cap * 2 : 8;
        EpEntry_ *n = realloc(r->eps, (size_t)ncap * sizeof *n);
        if (!n)
            return -1;
        r->eps = n;
        r->ep_cap = ncap;
    }
    r->eps[r->ep_count++] = *e;
    return 0;
}

static int key_add_(K26RlEpisodeReader *r, uint32_t ordinal, uint64_t seed)
{
    if (r->key_count == r->key_cap) {
        uint32_t ncap = r->key_cap ? r->key_cap * 2 : 8;
        KeyEntry_ *n = realloc(r->keys, (size_t)ncap * sizeof *n);
        if (!n)
            return -1;
        r->keys = n;
        r->key_cap = ncap;
    }
    r->keys[r->key_count].ordinal = ordinal;
    r->keys[r->key_count].seed = seed;
    r->key_count++;
    return 0;
}

/* Read and CRC-check the frame at off. Returns 0 with the payload in
 * scratch, or -1 on exactly the conditions that end a readable
 * prefix: the header does not fit, the arithmetic runs past end of
 * file, I/O fails, or the CRC disagrees. */
static int read_frame_at_(FILE *f, uint64_t file_size, uint64_t off,
                          Scratch_ *s, uint16_t *out_kind,
                          uint32_t *out_len)
{
    uint8_t hdr[K26RL_EPISODE_FRAME_HEADER_SIZE];
    uint32_t len, stored, crc;
    uint16_t kind;

    if (off > file_size || file_size - off < sizeof hdr)
        return -1;
    if (fseeko(f, (off_t)off, SEEK_SET) != 0)
        return -1;
    if (fread(hdr, 1, sizeof hdr, f) != sizeof hdr)
        return -1;
    kind = k26rl_get_u16_(hdr + K26RL_FH_OFF_KIND);
    len = k26rl_get_u32_(hdr + K26RL_FH_OFF_LENGTH);
    if (file_size - off - sizeof hdr < len)
        return -1;
    if (scratch_reserve_(s, len ? len : 1) != 0)
        return -1;
    if (len && fread(s->p, 1, len, f) != len)
        return -1;
    stored = k26rl_get_u32_(hdr + K26RL_FH_OFF_CRC);
    k26rl_put_u32_(hdr + K26RL_FH_OFF_CRC, 0);
    crc = k26rl_crc32c(K26RL_CRC32C_INIT, hdr, sizeof hdr);
    crc = k26rl_crc32c(crc, s->p, len);
    if (crc != stored)
        return -1;
    *out_kind = kind;
    *out_len = len;
    return 0;
}

/* Geometry parsed from the embedded spec blob. Unknown tags are
 * skipped by length; a truncated tag ends the walk. The file header's
 * own n_envs and steps-per-chunk fields are authoritative for those
 * two, so only the three per-record sizes come from here. */
static void parse_spec_(K26RlEpisodeReader *r)
{
    uint64_t pos = 0;

    while (pos + 6 <= r->spec_len) {
        uint16_t tag = k26rl_get_u16_(r->spec + pos);
        uint32_t len = k26rl_get_u32_(r->spec + pos + 2);
        const uint8_t *val = r->spec + pos + 6;

        if (pos + 6 + len > r->spec_len)
            break;
        if (len >= 4) {
            if (tag == K26RL_TAG_AGENT_COUNT)
                r->info.agent_count = k26rl_get_u32_(val);
            else if (tag == K26RL_TAG_OBS_TOTAL)
                r->info.obs_total = k26rl_get_u32_(val);
            else if (tag == K26RL_TAG_ACT_TOTAL)
                r->info.act_total = k26rl_get_u32_(val);
        }
        pos += 6 + len;
    }
}

/* Returns 0 parsed, 1 malformed (skipped by length), -1 allocation
 * failure. */
static int parse_file_header_(K26RlEpisodeReader *r, const uint8_t *p,
                              uint32_t plen)
{
    uint64_t pos;
    uint32_t glen, rlen, slen;

    if (plen < 24)
        return 1;
    pos = 24;
    if (pos + 2 > plen)
        return 1;
    glen = k26rl_get_u16_(p + pos);
    pos += 2;
    if (pos + glen > plen)
        return 1;
    pos += glen;
    if (pos + 2 > plen)
        return 1;
    rlen = k26rl_get_u16_(p + pos);
    pos += 2;
    if (pos + rlen > plen)
        return 1;
    pos += rlen;
    if (pos + 4 > plen)
        return 1;
    slen = k26rl_get_u32_(p + pos);
    pos += 4;
    if (pos + slen != plen)
        return 1;

    r->info.format_version = k26rl_get_u32_(p);
    r->info.governing_seed = k26rl_get_u64_(p + 4);
    r->info.rekey_ordinal = k26rl_get_u32_(p + 12);
    r->info.n_envs = k26rl_get_u32_(p + 16);
    r->info.steps_per_chunk = k26rl_get_u32_(p + 20);
    r->spec = malloc(slen ? slen : 1);
    if (!r->spec)
        return -1;
    memcpy(r->spec, p + pos, slen);
    r->spec_len = slen;
    parse_spec_(r);
    r->ncols8 = (uint64_t)r->info.obs_total + r->info.act_total +
                r->info.agent_count;
    if (key_add_(r, r->info.rekey_ordinal, r->info.governing_seed) != 0)
        return -1;
    r->have_header = 1;
    return 0;
}

typedef struct {
    uint32_t ordinal;
    uint32_t env;
    uint32_t episode;
    uint32_t dr_count;
    const uint8_t *obs;     /* obs_total binary64 bit patterns */
    const uint8_t *dr;      /* dr_count (tag u32, bits u64) pairs */
} StartView_;

static int parse_start_(const K26RlEpisodeReader *r, const uint8_t *p,
                        uint32_t plen, StartView_ *v)
{
    uint64_t need = 12 + (uint64_t)r->info.obs_total * 8 + 4;

    if (plen < need)
        return -1;
    v->ordinal = k26rl_get_u32_(p);
    v->env = k26rl_get_u32_(p + 4);
    v->episode = k26rl_get_u32_(p + 8);
    v->obs = p + 12;
    v->dr_count = k26rl_get_u32_(p + 12 + (size_t)r->info.obs_total * 8);
    if ((uint64_t)plen != need + (uint64_t)v->dr_count * 12)
        return -1;
    v->dr = p + need;
    return 0;
}

typedef struct {
    uint32_t ordinal;
    uint32_t env;
    uint32_t episode;
    uint32_t step_count;
    uint16_t end_reason;
    uint16_t fault_code;
    const uint8_t *adjustments;   /* agent_count binary64 bit patterns */
} EndView_;

static int parse_end_(const K26RlEpisodeReader *r, const uint8_t *p,
                      uint32_t plen, EndView_ *v)
{
    if ((uint64_t)plen != 20 + (uint64_t)r->info.agent_count * 8)
        return -1;
    v->ordinal = k26rl_get_u32_(p);
    v->env = k26rl_get_u32_(p + 4);
    v->episode = k26rl_get_u32_(p + 8);
    v->step_count = k26rl_get_u32_(p + 12);
    v->end_reason = k26rl_get_u16_(p + 16);
    v->fault_code = k26rl_get_u16_(p + 18);
    v->adjustments = p + 20;
    return 0;
}

typedef struct {
    uint32_t ordinal;
    uint32_t env;
    uint32_t first;
    uint32_t count;
    const uint8_t *cols;
} ChunkView_;

static int parse_chunk_(const K26RlEpisodeReader *r, const uint8_t *p,
                        uint32_t plen, ChunkView_ *v)
{
    if (plen < 16)
        return -1;
    v->ordinal = k26rl_get_u32_(p);
    v->env = k26rl_get_u32_(p + 4);
    v->first = k26rl_get_u32_(p + 8);
    v->count = k26rl_get_u32_(p + 12);
    if (v->count == 0 || v->count > r->info.steps_per_chunk)
        return -1;
    if ((uint64_t)plen != 16 + (uint64_t)v->count * (r->ncols8 * 8 + 12))
        return -1;
    v->cols = p + 16;
    return 0;
}

typedef struct {
    int open;
    uint32_t ordinal;
    uint32_t episode;
    uint32_t steps_seen;
    uint64_t start_off;
    uint64_t *chunk_offs;
    uint32_t chunk_count;
    uint32_t chunk_cap;
} Pending_;

static int pending_chunk_append_(Pending_ *pe, uint64_t off)
{
    if (pe->chunk_count == pe->chunk_cap) {
        uint32_t ncap = pe->chunk_cap ? pe->chunk_cap * 2 : 8;
        uint64_t *n = realloc(pe->chunk_offs, (size_t)ncap * sizeof *n);
        if (!n)
            return -1;
        pe->chunk_offs = n;
        pe->chunk_cap = ncap;
    }
    pe->chunk_offs[pe->chunk_count++] = off;
    return 0;
}

/* The sequential pass: validate every frame, note where the readable
 * prefix ends, collect the rekey seed table, and reconstruct an index
 * from the episode frames actually seen. Only episodes with both a
 * start and a matching end whose chunk arithmetic adds up are
 * indexed, so every indexed episode is fully decodable. */
static K26RlStatus scan_(K26RlEpisodeReader *r, Scratch_ *s,
                         uint64_t *out_end)
{
    uint64_t off = 8;
    Pending_ *pend = NULL;
    uint32_t n_pend = 0;
    uint32_t i;
    K26RlStatus st = K26RL_OK;

    for (;;) {
        uint16_t kind;
        uint32_t plen;

        if (read_frame_at_(r->f, r->file_size, off, s, &kind, &plen) != 0)
            break;

        if (kind == K26RL_FRAME_FILE_HEADER && !r->have_header) {
            int pr = parse_file_header_(r, s->p, plen);
            if (pr < 0) {
                st = K26RL_E_INTERNAL;
                goto out;
            }
            if (pr == 0 && r->info.n_envs) {
                pend = calloc(r->info.n_envs, sizeof *pend);
                if (!pend) {
                    st = K26RL_E_INTERNAL;
                    goto out;
                }
                n_pend = r->info.n_envs;
            }
        } else if (kind == K26RL_FRAME_EPISODE_START) {
            StartView_ v;
            if (r->have_header && parse_start_(r, s->p, plen, &v) == 0 &&
                v.env < n_pend) {
                Pending_ *pe = &pend[v.env];
                pe->open = 1;
                pe->ordinal = v.ordinal;
                pe->episode = v.episode;
                pe->steps_seen = 0;
                pe->start_off = off;
                pe->chunk_count = 0;
            }
        } else if (kind == K26RL_FRAME_STEP_CHUNK) {
            ChunkView_ v;
            if (r->have_header && parse_chunk_(r, s->p, plen, &v) == 0 &&
                v.env < n_pend && pend[v.env].open &&
                pend[v.env].ordinal == v.ordinal &&
                pend[v.env].steps_seen == v.first) {
                if (pending_chunk_append_(&pend[v.env], off) != 0) {
                    st = K26RL_E_INTERNAL;
                    goto out;
                }
                pend[v.env].steps_seen += v.count;
            }
        } else if (kind == K26RL_FRAME_EPISODE_END) {
            EndView_ v;
            if (r->have_header && parse_end_(r, s->p, plen, &v) == 0 &&
                v.env < n_pend && pend[v.env].open &&
                pend[v.env].ordinal == v.ordinal &&
                pend[v.env].episode == v.episode &&
                pend[v.env].steps_seen == v.step_count) {
                Pending_ *pe = &pend[v.env];
                EpEntry_ entry;
                uint64_t *offs = NULL;
                if (pe->chunk_count) {
                    offs = malloc((size_t)pe->chunk_count * sizeof *offs);
                    if (!offs) {
                        st = K26RL_E_INTERNAL;
                        goto out;
                    }
                    memcpy(offs, pe->chunk_offs,
                           (size_t)pe->chunk_count * sizeof *offs);
                }
                entry.ordinal = v.ordinal;
                entry.env = v.env;
                entry.episode = v.episode;
                entry.chunk_count = pe->chunk_count;
                entry.start_off = pe->start_off;
                entry.end_off = off;
                entry.chunk_offs = offs;
                if (eps_append_(r, &entry) != 0) {
                    free(offs);
                    st = K26RL_E_INTERNAL;
                    goto out;
                }
                pe->open = 0;
                pe->chunk_count = 0;
            }
        } else if (kind == K26RL_FRAME_REKEY) {
            if (plen >= 12) {
                if (key_add_(r, k26rl_get_u32_(s->p + 8),
                             k26rl_get_u64_(s->p)) != 0) {
                    st = K26RL_E_INTERNAL;
                    goto out;
                }
            }
        }
        /* Index frames and unknown kinds are skipped by length; the
         * reconstruction never trusts a stored index. */

        off += K26RL_EPISODE_FRAME_HEADER_SIZE + (uint64_t)plen;
    }
    *out_end = off;
out:
    if (pend) {
        for (i = 0; i < n_pend; i++)
            free(pend[i].chunk_offs);
        free(pend);
    }
    return st;
}

static int parse_index_payload_(const uint8_t *p, uint32_t plen,
                                EpEntry_ **out, uint32_t *out_count)
{
    uint64_t pos;
    uint32_t count, i, j;
    EpEntry_ *eps;

    if (plen < 4)
        return -1;
    count = k26rl_get_u32_(p);
    pos = 4;
    /* Every entry takes at least 32 bytes; a count beyond that bound
     * cannot be honest and must not size an allocation. */
    if ((uint64_t)count > ((uint64_t)plen - 4) / 32)
        return -1;
    eps = calloc(count ? count : 1, sizeof *eps);
    if (!eps)
        return -1;
    for (i = 0; i < count; i++) {
        EpEntry_ *e = &eps[i];
        uint64_t need;
        if (pos + 16 > plen)
            goto fail;
        e->ordinal = k26rl_get_u32_(p + pos);
        e->env = k26rl_get_u32_(p + pos + 4);
        e->episode = k26rl_get_u32_(p + pos + 8);
        e->chunk_count = k26rl_get_u32_(p + pos + 12);
        pos += 16;
        need = 8 + (uint64_t)e->chunk_count * 8 + 8;
        if (need > plen - pos)
            goto fail;
        e->start_off = k26rl_get_u64_(p + pos);
        pos += 8;
        e->chunk_offs = alloc_n_(e->chunk_count, sizeof *e->chunk_offs);
        if (!e->chunk_offs)
            goto fail;
        for (j = 0; j < e->chunk_count; j++, pos += 8)
            e->chunk_offs[j] = k26rl_get_u64_(p + pos);
        e->end_off = k26rl_get_u64_(p + pos);
        pos += 8;
    }
    if (pos != plen)
        goto fail;
    *out = eps;
    *out_count = count;
    return 0;
fail:
    free_entries_(eps, count);
    return -1;
}

/* Try the clean-close path: a well-formed trailer whose index offset
 * names a CRC-valid index frame that is the last content before the
 * trailer, and whose payload parses. Returns 1 with the parsed
 * entries on success, 0 otherwise. */
static int try_trailer_(K26RlEpisodeReader *r, Scratch_ *s,
                        EpEntry_ **out_eps, uint32_t *out_count)
{
    uint8_t tr[16];
    uint64_t idx_off;
    uint16_t kind;
    uint32_t plen;

    *out_eps = NULL;
    *out_count = 0;
    if (r->file_size < 8 + K26RL_EPISODE_FRAME_HEADER_SIZE + 16)
        return 0;
    if (fseeko(r->f, (off_t)(r->file_size - 16), SEEK_SET) != 0)
        return 0;
    if (fread(tr, 1, sizeof tr, r->f) != sizeof tr)
        return 0;
    if (memcmp(tr, K26RL_EPISODE_TRAILER, 8) != 0)
        return 0;
    idx_off = k26rl_get_u64_(tr + 8);
    if (idx_off < 8 ||
        idx_off > r->file_size - 16 - K26RL_EPISODE_FRAME_HEADER_SIZE)
        return 0;
    if (read_frame_at_(r->f, r->file_size, idx_off, s, &kind, &plen) != 0)
        return 0;
    if (kind != K26RL_FRAME_INDEX)
        return 0;
    if (idx_off + K26RL_EPISODE_FRAME_HEADER_SIZE + plen !=
        r->file_size - 16)
        return 0;
    return parse_index_payload_(s->p, plen, out_eps, out_count) == 0;
}

K26RlStatus k26rl_episode_reader_open(const char *path,
                                      K26RlEpisodeReader **out)
{
    FILE *f;
    off_t sz;
    uint8_t magic[8];
    K26RlEpisodeReader *r;
    Scratch_ s = { NULL, 0 };
    uint64_t readable_end = 8;
    EpEntry_ *tr_eps;
    uint32_t tr_count;
    int trailer_ok;
    K26RlStatus st;

    if (!path || !out)
        return K26RL_E_NULL;
    *out = NULL;
    f = fopen(path, "rb");
    if (!f)
        return K26RL_E_INTERNAL;
    if (fseeko(f, 0, SEEK_END) != 0 || (sz = ftello(f)) < 0 ||
        (uint64_t)sz < 8) {
        fclose(f);
        return K26RL_E_INTERNAL;
    }
    if (fseeko(f, 0, SEEK_SET) != 0 || fread(magic, 1, 8, f) != 8 ||
        memcmp(magic, K26RL_EPISODE_MAGIC, 8) != 0) {
        fclose(f);
        return K26RL_E_INTERNAL;
    }
    r = calloc(1, sizeof *r);
    if (!r) {
        fclose(f);
        return K26RL_E_INTERNAL;
    }
    r->f = f;
    r->file_size = (uint64_t)sz;

    st = scan_(r, &s, &readable_end);
    if (st != K26RL_OK) {
        free(s.p);
        k26rl_episode_reader_close(r);
        return st;
    }
    trailer_ok = try_trailer_(r, &s, &tr_eps, &tr_count);
    if (trailer_ok && readable_end == r->file_size - 16) {
        free_entries_(r->eps, r->ep_count);
        r->eps = tr_eps;
        r->ep_count = tr_count;
        r->ep_cap = tr_count;
        r->info.clean_close = 1;
        r->info.readable_bytes = r->file_size;
    } else {
        if (trailer_ok)
            free_entries_(tr_eps, tr_count);
        r->info.clean_close = 0;
        r->info.readable_bytes = readable_end;
    }
    r->info.episode_count = r->ep_count;
    free(s.p);
    *out = r;
    return K26RL_OK;
}

K26RlStatus k26rl_episode_reader_info(const K26RlEpisodeReader *r,
                                      K26RlEpisodeInfo *out)
{
    if (!r || !out)
        return K26RL_E_NULL;
    *out = r->info;
    return K26RL_OK;
}

K26RlStatus k26rl_episode_reader_spec(const K26RlEpisodeReader *r,
                                      const uint8_t **out, uint32_t *out_len)
{
    if (!r || !out || !out_len)
        return K26RL_E_NULL;
    if (!r->have_header)
        return K26RL_E_GEOMETRY;
    *out = r->spec;
    *out_len = r->spec_len;
    return K26RL_OK;
}

K26RlStatus k26rl_episode_reader_seed(const K26RlEpisodeReader *r,
                                      uint32_t ordinal, uint64_t *out_seed)
{
    uint32_t i;

    if (!r || !out_seed)
        return K26RL_E_NULL;
    for (i = 0; i < r->key_count; i++) {
        if (r->keys[i].ordinal == ordinal) {
            *out_seed = r->keys[i].seed;
            return K26RL_OK;
        }
    }
    return K26RL_E_GEOMETRY;
}

K26RlStatus k26rl_episode_read(K26RlEpisodeReader *r,
                               uint32_t rekey_ordinal, uint32_t env,
                               uint32_t episode, K26RlEpisodeData *out)
{
    const EpEntry_ *e = NULL;
    Scratch_ s = { NULL, 0 };
    uint16_t kind;
    uint32_t plen, i, j, k;
    uint32_t OT, AT, AC;
    uint64_t acc = 0;
    StartView_ sv;
    EndView_ ev;

    if (!r || !out)
        return K26RL_E_NULL;
    memset(out, 0, sizeof *out);
    for (i = 0; i < r->ep_count; i++) {
        if (r->eps[i].ordinal == rekey_ordinal && r->eps[i].env == env &&
            r->eps[i].episode == episode) {
            e = &r->eps[i];
            break;
        }
    }
    if (!e)
        return K26RL_E_GEOMETRY;
    OT = r->info.obs_total;
    AT = r->info.act_total;
    AC = r->info.agent_count;

    if (read_frame_at_(r->f, r->file_size, e->start_off, &s, &kind,
                       &plen) != 0 ||
        kind != K26RL_FRAME_EPISODE_START ||
        parse_start_(r, s.p, plen, &sv) != 0 ||
        sv.ordinal != e->ordinal || sv.env != e->env ||
        sv.episode != e->episode)
        goto fail;
    out->initial_obs = alloc_n_(OT, sizeof(double));
    out->dr_tags = alloc_n_(sv.dr_count, sizeof(uint32_t));
    out->dr_values = alloc_n_(sv.dr_count, sizeof(double));
    if (!out->initial_obs || !out->dr_tags || !out->dr_values)
        goto fail;
    out->dr_count = sv.dr_count;
    for (j = 0; j < OT; j++)
        out->initial_obs[j] = k26rl_get_f64_(sv.obs + (size_t)j * 8);
    for (j = 0; j < sv.dr_count; j++) {
        out->dr_tags[j] = k26rl_get_u32_(sv.dr + (size_t)j * 12);
        out->dr_values[j] = k26rl_get_f64_(sv.dr + (size_t)j * 12 + 4);
    }

    /* The end frame is read before the chunks: its step count sizes
     * the step arrays. */
    if (read_frame_at_(r->f, r->file_size, e->end_off, &s, &kind,
                       &plen) != 0 ||
        kind != K26RL_FRAME_EPISODE_END ||
        parse_end_(r, s.p, plen, &ev) != 0 ||
        ev.ordinal != e->ordinal || ev.env != e->env ||
        ev.episode != e->episode)
        goto fail;
    out->step_count = ev.step_count;
    out->end_reason = ev.end_reason;
    out->fault_code = ev.fault_code;
    out->terminal_adjustments = alloc_n_(AC, sizeof(double));
    if (!out->terminal_adjustments)
        goto fail;
    for (j = 0; j < AC; j++)
        out->terminal_adjustments[j] =
            k26rl_get_f64_(ev.adjustments + (size_t)j * 8);

    out->obs = alloc_n_((uint64_t)ev.step_count * OT, sizeof(double));
    out->act = alloc_n_((uint64_t)ev.step_count * AT, sizeof(double));
    out->rewards = alloc_n_((uint64_t)ev.step_count * AC, sizeof(double));
    out->flags = alloc_n_(ev.step_count, sizeof(uint32_t));
    out->applied_dt = alloc_n_(ev.step_count, sizeof(double));
    if (!out->obs || !out->act || !out->rewards || !out->flags ||
        !out->applied_dt)
        goto fail;

    for (i = 0; i < e->chunk_count; i++) {
        ChunkView_ cv;
        uint64_t K;
        if (read_frame_at_(r->f, r->file_size, e->chunk_offs[i], &s, &kind,
                           &plen) != 0 ||
            kind != K26RL_FRAME_STEP_CHUNK ||
            parse_chunk_(r, s.p, plen, &cv) != 0 ||
            cv.ordinal != e->ordinal || cv.env != e->env ||
            cv.first != acc || acc + cv.count > ev.step_count)
            goto fail;
        K = cv.count;
        for (j = 0; j < OT; j++)
            for (k = 0; k < K; k++)
                out->obs[(size_t)(acc + k) * OT + j] =
                    k26rl_get_f64_(cv.cols + (j * K + k) * 8);
        for (j = 0; j < AT; j++)
            for (k = 0; k < K; k++)
                out->act[(size_t)(acc + k) * AT + j] =
                    k26rl_get_f64_(cv.cols + ((OT + j) * K + k) * 8);
        for (j = 0; j < AC; j++)
            for (k = 0; k < K; k++)
                out->rewards[(size_t)(acc + k) * AC + j] =
                    k26rl_get_f64_(cv.cols +
                                   (((uint64_t)OT + AT + j) * K + k) * 8);
        for (k = 0; k < K; k++)
            out->flags[acc + k] =
                k26rl_get_u32_(cv.cols + r->ncols8 * K * 8 + (size_t)k * 4);
        for (k = 0; k < K; k++)
            out->applied_dt[acc + k] =
                k26rl_get_f64_(cv.cols + r->ncols8 * K * 8 + K * 4 +
                               (size_t)k * 8);
        acc += K;
    }
    if (acc != ev.step_count)
        goto fail;

    out->rekey_ordinal = e->ordinal;
    out->env = e->env;
    out->episode = e->episode;
    free(s.p);
    return K26RL_OK;

fail:
    free(s.p);
    k26rl_episode_free(out);
    return K26RL_E_INTERNAL;
}

void k26rl_episode_free(K26RlEpisodeData *data)
{
    if (!data)
        return;
    free(data->initial_obs);
    free(data->dr_tags);
    free(data->dr_values);
    free(data->obs);
    free(data->act);
    free(data->rewards);
    free(data->flags);
    free(data->applied_dt);
    free(data->terminal_adjustments);
    memset(data, 0, sizeof *data);
}

K26RlStatus k26rl_episode_reader_at(const K26RlEpisodeReader *r, uint32_t k,
                                    uint32_t *out_ordinal, uint32_t *out_env,
                                    uint32_t *out_episode)
{
    if (!r || !out_ordinal || !out_env || !out_episode)
        return K26RL_E_NULL;
    if (k >= r->ep_count)
        return K26RL_E_GEOMETRY;
    *out_ordinal = r->eps[k].ordinal;
    *out_env = r->eps[k].env;
    *out_episode = r->eps[k].episode;
    return K26RL_OK;
}

void k26rl_episode_reader_close(K26RlEpisodeReader *r)
{
    if (!r)
        return;
    if (r->f)
        fclose(r->f);
    free(r->spec);
    free(r->keys);
    free_entries_(r->eps, r->ep_count);
    free(r);
}
