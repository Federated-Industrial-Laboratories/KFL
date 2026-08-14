/* tap_ring.c - the telemetry tap's producer.
 *
 * Publishes the record format's frames into the shared memory ring
 * k26rl_tap.h describes. The frames are the episode writer's frames:
 * same kinds, same payload layouts, same values, published at the
 * same points by a caller that calls both. What differs is the
 * transport, and only the transport: this producer keeps its own
 * sequence numbering, its own mapping, and no state whatever in
 * common with the file writer, which is what makes a run's episode
 * file identical whether or not the tap is enabled.
 *
 * Every byte of the ring is established at open: the object is
 * created, sized, and mapped there, and the preamble is written
 * there. Publication afterwards is one bounded copy into a slot, one
 * checksum over the same bytes, and two release stores. It allocates
 * nothing, makes no system call, and branches on nothing a consumer
 * does, so a step costs the same whether a consumer is fast, slow,
 * absent, or dead. */
#include "k26rl_tap.h"
#include "k26rl_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    int open;
    uint32_t ordinal;       /* rekey ordinal captured at episode start */
    uint32_t episode;
    uint32_t steps;         /* transitions recorded in this episode */
} TapEnv_;

struct K26RlTap {
    uint8_t *base;          /* the mapping */
    size_t   bytes;         /* its length */
    char     object[K26RL_TAP_NAME_MAX + 16];
    K26RlEpisodeGeom geom;
    uint64_t ncols8;        /* obs + act + agents, the 8-byte columns */
    uint32_t slot_size;
    uint32_t slot_count;    /* power of two */
    uint32_t preamble;      /* bytes to the first slot */
    uint32_t rekey_ordinal;
    uint64_t sequence;      /* next slot frame's sequence; 0 is the preamble */
    TapEnv_ *envs;          /* n_envs, trailing this struct in one block */
};

/* ---- Naming and geometry ------------------------------------------ */

static K26RlStatus name_check_(const char *name)
{
    size_t i, n;

    n = strlen(name);
    if (n == 0 || n > (size_t)K26RL_TAP_NAME_MAX)
        return K26RL_E_TAP_NAME;
    for (i = 0; i < n; i++) {
        char c = name[i];
        int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                 (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok)
            return K26RL_E_TAP_NAME;
    }
    return K26RL_OK;
}

static uint64_t round_up_(uint64_t v, uint64_t quantum)
{
    uint64_t r = v % quantum;

    return r ? v + (quantum - r) : v;
}

/* Slot size is the largest frame the declared geometry can produce,
 * so every frame fits one slot and a slot's address is its sequence
 * masked. Slot count is the largest power of two inside the byte
 * budget, floored so a wide environment still gets usable history and
 * refused when even the floor would exceed the ceiling. */
static K26RlStatus geometry_(const K26RlEpisodeGeom *g, uint32_t *out_slot,
                             uint32_t *out_count)
{
    uint64_t ncols8 = (uint64_t)g->obs_total + g->act_total + g->agent_count;
    uint64_t step_pay = 28 + ncols8 * 8;
    uint64_t start_pay = 16 + (uint64_t)g->obs_total * 8 +
                         (uint64_t)g->dr_max * 12;
    uint64_t end_pay = 20 + (uint64_t)g->agent_count * 8;
    uint64_t max_pay = step_pay;
    uint64_t slot, count;

    if (start_pay > max_pay)
        max_pay = start_pay;
    if (end_pay > max_pay)
        max_pay = end_pay;
    /* The rekey frame's 12-byte payload cannot be the largest. */
    if (max_pay > (uint64_t)UINT32_MAX - K26RL_EPISODE_FRAME_HEADER_SIZE -
                  K26RL_TAP_LINE)
        return K26RL_E_GEOMETRY;
    slot = round_up_(K26RL_EPISODE_FRAME_HEADER_SIZE + max_pay,
                     K26RL_TAP_LINE);

    count = 1;
    while (count * 2 * slot <= K26RL_TAP_BUDGET_DEFAULT)
        count *= 2;
    if (count < K26RL_TAP_SLOTS_MIN)
        count = K26RL_TAP_SLOTS_MIN;
    if (count * slot > K26RL_TAP_BYTES_MAX)
        return K26RL_E_GEOMETRY;

    *out_slot = (uint32_t)slot;
    *out_count = (uint32_t)count;
    return K26RL_OK;
}

/* ---- Publication -------------------------------------------------- */

static uint8_t *slot_of_(const K26RlTap *t, uint64_t sequence)
{
    return t->base + t->preamble +
           (size_t)(sequence & (t->slot_count - 1)) * t->slot_size;
}

/* Finish a frame whose payload already sits at slot + 24: build the
 * header with the true sequence in a local word, checksum header and
 * payload together, copy every header byte except the sequence field
 * into the slot, then publish by release-storing the sequence and the
 * head cursor.
 *
 * The sequence field is the publication flag and is written last. Its
 * counterpart is the invalidation the caller performed before
 * touching the body: a slot being rewritten reads back as sequence
 * zero throughout, so a consumer that raced the rewrite fails its
 * re-read rather than accepting a frame stitched from two. */
static void finish_(K26RlTap *t, uint8_t *slot, uint16_t kind, uint32_t plen)
{
    uint8_t hdr[K26RL_EPISODE_FRAME_HEADER_SIZE];
    uint64_t seq = t->sequence;
    uint32_t crc;

    k26rl_frame_header_write_(hdr, kind, plen, seq);
    crc = k26rl_crc32c(K26RL_CRC32C_INIT, hdr, sizeof hdr);
    crc = k26rl_crc32c(crc, slot + K26RL_EPISODE_FRAME_HEADER_SIZE, plen);
    k26rl_put_u32_(hdr + K26RL_FH_OFF_CRC, crc);

    memcpy(slot, hdr, K26RL_FH_OFF_SEQUENCE);
    memcpy(slot + K26RL_FH_OFF_CRC, hdr + K26RL_FH_OFF_CRC,
           sizeof hdr - K26RL_FH_OFF_CRC);
    k26rl_pub_store_u64_(slot + K26RL_FH_OFF_SEQUENCE, seq);
    k26rl_pub_store_u64_(t->base + K26RL_TAP_OFF_HEAD, seq);
    t->sequence++;
}

/* Claim the slot for the next frame and return where its payload
 * goes, or NULL when the payload cannot fit the slot the geometry
 * sized. An unfittable frame consumes its sequence number without
 * being written, so the loss reaches consumers as a gap rather than
 * as a frame that quietly is not there. */
static uint8_t *claim_(K26RlTap *t, uint32_t plen)
{
    uint8_t *slot = slot_of_(t, t->sequence);

    if ((uint64_t)plen + K26RL_EPISODE_FRAME_HEADER_SIZE > t->slot_size) {
        k26rl_pub_store_u64_(slot + K26RL_FH_OFF_SEQUENCE, 0);
        t->sequence++;
        return NULL;
    }
    k26rl_pub_store_u64_(slot + K26RL_FH_OFF_SEQUENCE, 0);
    return slot;
}

/* ---- Open and close ----------------------------------------------- */

K26RlStatus k26rl_tap_open(const char *name, const K26RlEpisodeGeom *geom,
                           uint64_t governing_seed, uint32_t rekey_ordinal,
                           const char *grammar_version,
                           const char *runtime_version,
                           const uint8_t *spec, uint32_t spec_len,
                           K26RlTap **out)
{
    K26RlTap *t = NULL;
    size_t glen, rlen;
    uint64_t hdr_plen, preamble, bytes;
    uint32_t slot_size, slot_count;
    uint8_t *p;
    uint32_t crc;
    int fd = -1;
    K26RlStatus st;

    if (!name || !geom || !grammar_version || !runtime_version || !out)
        return K26RL_E_NULL;
    if (spec_len && !spec)
        return K26RL_E_NULL;
    *out = NULL;
    if (geom->n_envs == 0)
        return K26RL_E_GEOMETRY;
    st = name_check_(name);
    if (st != K26RL_OK)
        return st;
    glen = strlen(grammar_version);
    rlen = strlen(runtime_version);
    if (glen > 0xFFFF || rlen > 0xFFFF)
        return K26RL_E_GEOMETRY;
    st = geometry_(geom, &slot_size, &slot_count);
    if (st != K26RL_OK)
        return st;

    hdr_plen = 24 + 2 + (uint64_t)glen + 2 + (uint64_t)rlen + 4 + spec_len;
    if (hdr_plen > UINT32_MAX)
        return K26RL_E_GEOMETRY;
    preamble = round_up_(K26RL_TAP_OFF_HEADER +
                         K26RL_EPISODE_FRAME_HEADER_SIZE + hdr_plen,
                         K26RL_TAP_LINE);
    /* Both parts are bounded well inside a size_t on every target the
     * tree builds for: the ring by the byte ceiling geometry_ already
     * enforced, the preamble by the same ceiling here. */
    if (preamble > K26RL_TAP_BYTES_MAX)
        return K26RL_E_GEOMETRY;
    bytes = preamble + (uint64_t)slot_count * slot_size;

    /* One allocation holds the producer's state and its per-environment
     * episode bookkeeping, so publication touches no memory this call
     * did not obtain. */
    t = calloc(1, sizeof *t + (size_t)geom->n_envs * sizeof(TapEnv_));
    if (!t)
        return K26RL_E_TAP_UNAVAILABLE;
    t->envs = (TapEnv_ *)(void *)(t + 1);
    if ((size_t)snprintf(t->object, sizeof t->object, "%s%s",
                         K26RL_TAP_OBJECT_PREFIX, name) >= sizeof t->object) {
        free(t);
        return K26RL_E_TAP_NAME;
    }

    /* O_EXCL makes creation the existence check, as it does for an
     * episode file: two producers on one ring would interleave frames
     * into one sequence space and make every gap a consumer computed
     * a fiction. Mode 0600 because a training run's telemetry is not
     * world-readable by default. */
    fd = shm_open(t->object, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        st = (errno == EEXIST) ? K26RL_E_TAP_EXISTS : K26RL_E_TAP_UNAVAILABLE;
        free(t);
        return st;
    }
    if (ftruncate(fd, (off_t)bytes) != 0) {
        close(fd);
        shm_unlink(t->object);
        free(t);
        return K26RL_E_TAP_UNAVAILABLE;
    }
    t->base = mmap(NULL, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                   fd, 0);
    /* The descriptor has done its work: the mapping holds the object
     * open, so nothing later needs the descriptor. */
    close(fd);
    if (t->base == MAP_FAILED) {
        t->base = NULL;
        shm_unlink(t->object);
        free(t);
        return K26RL_E_TAP_UNAVAILABLE;
    }

    t->bytes = (size_t)bytes;
    t->geom = *geom;
    t->ncols8 = (uint64_t)geom->obs_total + geom->act_total +
                geom->agent_count;
    t->slot_size = slot_size;
    t->slot_count = slot_count;
    t->preamble = (uint32_t)preamble;
    t->rekey_ordinal = rekey_ordinal;
    t->sequence = 1;   /* sequence 0 is the preamble's file-header frame */

    /* The control block. A consumer reads these before it trusts any
     * other byte of the object. */
    memcpy(t->base + K26RL_TAP_OFF_MAGIC, K26RL_TAP_MAGIC, 8);
    k26rl_put_u32_(t->base + K26RL_TAP_OFF_LAYOUT, K26RL_TAP_LAYOUT_VERSION);
    k26rl_put_u32_(t->base + K26RL_TAP_OFF_FORMAT,
                   K26RL_EPISODE_FORMAT_VERSION);
    k26rl_put_u32_(t->base + K26RL_TAP_OFF_SLOT_SIZE, slot_size);
    k26rl_put_u32_(t->base + K26RL_TAP_OFF_SLOT_COUNT, slot_count);
    k26rl_put_u32_(t->base + K26RL_TAP_OFF_PREAMBLE, (uint32_t)preamble);
    k26rl_put_u32_(t->base + K26RL_TAP_OFF_CLOSED, 0);
    k26rl_pub_store_u64_(t->base + K26RL_TAP_OFF_HEAD, 0);

    /* The file-header frame, whole and in the ordinary frame
     * encoding, at sequence 0 where no overwrite reaches it. Its
     * payload is the episode file's file-header payload, field for
     * field, so the two transports publish one schema. */
    p = t->base + K26RL_TAP_OFF_HEADER + K26RL_EPISODE_FRAME_HEADER_SIZE;
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

    p = t->base + K26RL_TAP_OFF_HEADER;
    k26rl_frame_header_write_(p, K26RL_FRAME_FILE_HEADER,
                              (uint32_t)hdr_plen, 0);
    crc = k26rl_crc32c(K26RL_CRC32C_INIT, p,
                       (size_t)K26RL_EPISODE_FRAME_HEADER_SIZE + hdr_plen);
    k26rl_put_u32_(p + K26RL_FH_OFF_CRC, crc);

    *out = t;
    return K26RL_OK;
}

void k26rl_tap_close(K26RlTap *t)
{
    if (!t)
        return;
    /* Mark closed before the name goes, so a consumer that is mapped
     * learns the run ended rather than waiting on a cursor that will
     * never move again. Removing the name leaves existing mappings
     * valid, which is what lets a consumer finish reading. */
    k26rl_pub_store_u32_(t->base + K26RL_TAP_OFF_CLOSED, 1);
    shm_unlink(t->object);
    munmap(t->base, t->bytes);
    free(t);
}

uint64_t k26rl_tap_published(const K26RlTap *t)
{
    return t ? t->sequence - 1 : 0;
}

/* ---- The four publication calls ------------------------------------ */

void k26rl_tap_start(K26RlTap *t, uint32_t env, uint32_t episode,
                     const double *initial_obs, const uint32_t *dr_tags,
                     const double *dr_values, uint32_t dr_count)
{
    uint8_t *slot, *p;
    TapEnv_ *e;
    uint32_t plen, j;

    if (!t || env >= t->geom.n_envs)
        return;
    if ((t->geom.obs_total && !initial_obs) ||
        (dr_count && (!dr_tags || !dr_values)))
        return;

    /* The episode's identity is captured whether or not its frame
     * fits, so the step and end frames that follow carry the same
     * ordinal and episode index the file's do. */
    e = &t->envs[env];
    e->open = 1;
    e->ordinal = t->rekey_ordinal;
    e->episode = episode;
    e->steps = 0;

    plen = (uint32_t)(16 + (uint64_t)t->geom.obs_total * 8 +
                      (uint64_t)dr_count * 12);
    slot = claim_(t, plen);
    if (!slot)
        return;
    p = slot + K26RL_EPISODE_FRAME_HEADER_SIZE;
    k26rl_put_u32_(p, t->rekey_ordinal);
    k26rl_put_u32_(p + 4, env);
    k26rl_put_u32_(p + 8, episode);
    p += 12;
    for (j = 0; j < t->geom.obs_total; j++, p += 8)
        k26rl_put_f64_(p, initial_obs[j]);
    k26rl_put_u32_(p, dr_count);
    p += 4;
    for (j = 0; j < dr_count; j++, p += 12) {
        k26rl_put_u32_(p, dr_tags[j]);
        k26rl_put_f64_(p + 4, dr_values[j]);
    }
    finish_(t, slot, K26RL_FRAME_EPISODE_START, plen);
}

/* One step record, published as the step-chunk kind with a count of
 * one. At K equal to one the format's column-major payload is the
 * same bytes as row order, so this is the file's layout with no
 * second encoding: observations, actions, rewards, the flag word,
 * then the applied dt. */
void k26rl_tap_step(K26RlTap *t, uint32_t env, const double *obs,
                    const double *act, const double *rewards,
                    uint32_t flags, double applied_dt)
{
    uint8_t *slot, *p;
    TapEnv_ *e;
    uint32_t plen, j;

    if (!t || env >= t->geom.n_envs)
        return;
    if ((t->geom.obs_total && !obs) || (t->geom.act_total && !act) ||
        (t->geom.agent_count && !rewards))
        return;
    e = &t->envs[env];

    plen = (uint32_t)(28 + t->ncols8 * 8);
    slot = claim_(t, plen);
    if (!slot) {
        e->steps++;
        return;
    }
    p = slot + K26RL_EPISODE_FRAME_HEADER_SIZE;
    k26rl_put_u32_(p, e->ordinal);
    k26rl_put_u32_(p + 4, env);
    k26rl_put_u32_(p + 8, e->steps);
    k26rl_put_u32_(p + 12, 1);
    p += 16;
    for (j = 0; j < t->geom.obs_total; j++, p += 8)
        k26rl_put_f64_(p, obs[j]);
    for (j = 0; j < t->geom.act_total; j++, p += 8)
        k26rl_put_f64_(p, act[j]);
    for (j = 0; j < t->geom.agent_count; j++, p += 8)
        k26rl_put_f64_(p, rewards[j]);
    k26rl_put_u32_(p, flags);
    p += 4;
    k26rl_put_f64_(p, applied_dt);
    e->steps++;
    finish_(t, slot, K26RL_FRAME_STEP_CHUNK, plen);
}

void k26rl_tap_end(K26RlTap *t, uint32_t env, uint16_t end_reason,
                   uint16_t fault_code, const double *terminal_adjustments)
{
    uint8_t *slot, *p;
    TapEnv_ *e;
    uint32_t plen, j;

    if (!t || env >= t->geom.n_envs)
        return;
    if (t->geom.agent_count && !terminal_adjustments)
        return;
    e = &t->envs[env];

    plen = (uint32_t)(20 + (uint64_t)t->geom.agent_count * 8);
    slot = claim_(t, plen);
    if (!slot) {
        e->open = 0;
        return;
    }
    p = slot + K26RL_EPISODE_FRAME_HEADER_SIZE;
    k26rl_put_u32_(p, e->ordinal);
    k26rl_put_u32_(p + 4, env);
    k26rl_put_u32_(p + 8, e->episode);
    k26rl_put_u32_(p + 12, e->steps);
    k26rl_put_u16_(p + 16, end_reason);
    k26rl_put_u16_(p + 18, fault_code);
    p += 20;
    for (j = 0; j < t->geom.agent_count; j++, p += 8)
        k26rl_put_f64_(p, terminal_adjustments[j]);
    e->open = 0;
    finish_(t, slot, K26RL_FRAME_EPISODE_END, plen);
}

void k26rl_tap_rekey(K26RlTap *t, uint64_t new_seed)
{
    uint8_t *slot;

    if (!t)
        return;
    slot = claim_(t, 12);
    if (slot) {
        k26rl_put_u64_(slot + K26RL_EPISODE_FRAME_HEADER_SIZE, new_seed);
        k26rl_put_u32_(slot + K26RL_EPISODE_FRAME_HEADER_SIZE + 8,
                       t->rekey_ordinal + 1u);
        finish_(t, slot, K26RL_FRAME_REKEY, 12);
    }
    t->rekey_ordinal++;
}
