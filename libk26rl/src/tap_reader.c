/* tap_reader.c - the telemetry tap's consumer.
 *
 * Attaches read-only to a ring a serving simulation publishes and
 * delivers whole frames, or reports the frames the ring overwrote
 * before this reader reached them. Everything the reader touches in
 * the mapping it treats as hostile in exactly one sense: the producer
 * may overwrite any slot at any moment, so a frame is accepted only
 * when the slot's sequence field reads the same value before and
 * after the copy and the frame's own checksum verifies over what was
 * copied.
 *
 * The reader writes nothing. The mapping is PROT_READ, the object is
 * opened O_RDONLY, and the read position lives here rather than in
 * the ring, so no consumer can signal, delay, or perturb a producer,
 * and no control path runs from here into a running simulation. */
#include "k26rl_tap.h"
#include "k26rl_internal.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct K26RlTapReader {
    const uint8_t *base;
    size_t   bytes;
    uint32_t slot_size;
    uint32_t slot_count;    /* power of two */
    uint32_t preamble;
    uint32_t header_len;    /* the file-header frame, header and payload */
    uint64_t next;          /* sequence this reader wants next */
    uint64_t accepted;
    uint64_t lost;
};

static const uint8_t *slot_of_(const K26RlTapReader *r, uint64_t sequence)
{
    return r->base + r->preamble +
           (size_t)(sequence & (r->slot_count - 1)) * r->slot_size;
}

/* Verify a frame image the caller already copied out of the mapping:
 * the checksum is recomputed over the frame with its checksum field
 * zeroed, which is how the format defines it. */
static int frame_ok_(const uint8_t *frame, uint32_t total, uint64_t sequence)
{
    uint8_t hdr[K26RL_EPISODE_FRAME_HEADER_SIZE];
    uint32_t want, got;

    if (total < K26RL_EPISODE_FRAME_HEADER_SIZE)
        return 0;
    if (k26rl_get_u64_(frame + K26RL_FH_OFF_SEQUENCE) != sequence)
        return 0;
    want = k26rl_get_u32_(frame + K26RL_FH_OFF_CRC);
    memcpy(hdr, frame, sizeof hdr);
    k26rl_put_u32_(hdr + K26RL_FH_OFF_CRC, 0);
    got = k26rl_crc32c(K26RL_CRC32C_INIT, hdr, sizeof hdr);
    got = k26rl_crc32c(got, frame + K26RL_EPISODE_FRAME_HEADER_SIZE,
                       total - K26RL_EPISODE_FRAME_HEADER_SIZE);
    return got == want;
}

K26RlStatus k26rl_tap_attach(const char *name, int from_start,
                             K26RlTapReader **out)
{
    char object[K26RL_TAP_NAME_MAX + 16];
    K26RlTapReader *r;
    struct stat sb;
    const uint8_t *base;
    uint32_t slot_size, slot_count, preamble, plen;
    uint64_t need;
    int fd;

    if (!name || !out)
        return K26RL_E_NULL;
    *out = NULL;
    if (strlen(name) == 0 || strlen(name) > (size_t)K26RL_TAP_NAME_MAX)
        return K26RL_E_TAP_NAME;
    if ((size_t)snprintf(object, sizeof object, "%s%s",
                         K26RL_TAP_OBJECT_PREFIX, name) >= sizeof object)
        return K26RL_E_TAP_NAME;

    fd = shm_open(object, O_RDONLY, 0);
    if (fd < 0)
        return K26RL_E_TAP_UNAVAILABLE;
    if (fstat(fd, &sb) != 0 || sb.st_size <= 0) {
        close(fd);
        return K26RL_E_TAP_UNAVAILABLE;
    }
    base = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED)
        return K26RL_E_TAP_UNAVAILABLE;

    if ((uint64_t)sb.st_size < K26RL_TAP_OFF_HEADER +
                               K26RL_EPISODE_FRAME_HEADER_SIZE ||
        memcmp(base + K26RL_TAP_OFF_MAGIC, K26RL_TAP_MAGIC, 8) != 0 ||
        k26rl_get_u32_(base + K26RL_TAP_OFF_LAYOUT) !=
            K26RL_TAP_LAYOUT_VERSION ||
        k26rl_get_u32_(base + K26RL_TAP_OFF_FORMAT) !=
            K26RL_EPISODE_FORMAT_VERSION)
        goto reject;

    slot_size = k26rl_get_u32_(base + K26RL_TAP_OFF_SLOT_SIZE);
    slot_count = k26rl_get_u32_(base + K26RL_TAP_OFF_SLOT_COUNT);
    preamble = k26rl_get_u32_(base + K26RL_TAP_OFF_PREAMBLE);
    if (slot_size < K26RL_EPISODE_FRAME_HEADER_SIZE || slot_count == 0 ||
        (slot_count & (slot_count - 1)) != 0 ||
        preamble < K26RL_TAP_OFF_HEADER + K26RL_EPISODE_FRAME_HEADER_SIZE)
        goto reject;
    need = (uint64_t)preamble + (uint64_t)slot_count * slot_size;
    if ((uint64_t)sb.st_size < need)
        goto reject;

    /* The file-header frame must be whole and must verify before this
     * reader reports anything about the run it describes. */
    plen = k26rl_get_u32_(base + K26RL_TAP_OFF_HEADER + K26RL_FH_OFF_LENGTH);
    if ((uint64_t)K26RL_TAP_OFF_HEADER + K26RL_EPISODE_FRAME_HEADER_SIZE +
        plen > preamble)
        goto reject;
    if (k26rl_get_u16_(base + K26RL_TAP_OFF_HEADER + K26RL_FH_OFF_KIND) !=
            K26RL_FRAME_FILE_HEADER ||
        !frame_ok_(base + K26RL_TAP_OFF_HEADER,
                   K26RL_EPISODE_FRAME_HEADER_SIZE + plen, 0))
        goto reject;

    r = calloc(1, sizeof *r);
    if (!r) {
        munmap((void *)(uintptr_t)base, (size_t)sb.st_size);
        return K26RL_E_TAP_UNAVAILABLE;
    }
    r->base = base;
    r->bytes = (size_t)sb.st_size;
    r->slot_size = slot_size;
    r->slot_count = slot_count;
    r->preamble = preamble;
    r->header_len = K26RL_EPISODE_FRAME_HEADER_SIZE + plen;
    /* From the start means the first slot frame; otherwise join where
     * the producer is, which is the live-attach case and reports no
     * loss for what ran before this reader existed. */
    r->next = from_start
        ? 1
        : k26rl_pub_load_u64_(base + K26RL_TAP_OFF_HEAD) + 1;
    *out = r;
    return K26RL_OK;

reject:
    munmap((void *)(uintptr_t)base, (size_t)sb.st_size);
    return K26RL_E_GEOMETRY;
}

K26RlStatus k26rl_tap_read(K26RlTapReader *r, uint8_t *buf, uint32_t cap,
                           uint32_t *out_len, uint64_t *out_lost)
{
    const uint8_t *slot;
    uint64_t head, seq0, seq1, lost = 0;
    uint32_t plen, total;

    if (!r || !buf || !out_len)
        return K26RL_E_NULL;
    *out_len = 0;
    if (out_lost)
        *out_lost = 0;
    if (cap < r->slot_size)
        return K26RL_E_GEOMETRY;

    head = k26rl_pub_load_u64_(r->base + K26RL_TAP_OFF_HEAD);
    if (head < r->next)
        return K26RL_OK;               /* nothing published since last read */

    /* Fallen further behind than the ring is deep: everything older
     * than the oldest slot still standing is gone, counted, and
     * stepped over. This is the whole of the loss accounting; a
     * consumer's arithmetic on sequence numbers is the only signal
     * the format gives it, and it is exact. */
    if (head - r->next >= r->slot_count) {
        uint64_t oldest = head - r->slot_count + 1;
        lost = oldest - r->next;
        r->next = oldest;
    }

    slot = slot_of_(r, r->next);
    seq0 = k26rl_pub_load_u64_(slot + K26RL_FH_OFF_SEQUENCE);
    if (seq0 != r->next) {
        /* The producer is rewriting this slot, so the frame this
         * reader wanted is on its way out. Report what is already
         * known lost and let the caller poll again; the fall-behind
         * arm above resolves it once the cursor moves. */
        r->lost += lost;
        if (out_lost)
            *out_lost = lost;
        return K26RL_OK;
    }

    plen = k26rl_get_u32_(slot + K26RL_FH_OFF_LENGTH);
    total = K26RL_EPISODE_FRAME_HEADER_SIZE + plen;
    if (plen > r->slot_size - K26RL_EPISODE_FRAME_HEADER_SIZE) {
        /* A length no honest producer of this geometry can write:
         * the slot is being rewritten under the copy. Discard it. */
        lost += 1;
        r->next++;
        r->lost += lost;
        if (out_lost)
            *out_lost = lost;
        return K26RL_OK;
    }
    memcpy(buf, slot, total);

    seq1 = k26rl_pub_load_u64_(slot + K26RL_FH_OFF_SEQUENCE);
    if (seq1 != r->next || !frame_ok_(buf, total, r->next)) {
        /* The copy raced an overwrite: the re-read catches it, and
         * the checksum catches it independently. Either way the frame
         * is lost rather than delivered half old and half new. */
        lost += 1;
        r->next++;
        r->lost += lost;
        if (out_lost)
            *out_lost = lost;
        return K26RL_OK;
    }

    r->next++;
    r->accepted++;
    r->lost += lost;
    if (out_lost)
        *out_lost = lost;
    *out_len = total;
    return K26RL_OK;
}

K26RlStatus k26rl_tap_reader_header(const K26RlTapReader *r,
                                    const uint8_t **out_frame,
                                    uint32_t *out_len)
{
    if (!r || !out_frame || !out_len)
        return K26RL_E_NULL;
    *out_frame = r->base + K26RL_TAP_OFF_HEADER;
    *out_len = r->header_len;
    return K26RL_OK;
}

/* The spec blob inside the preamble's file-header payload. The
 * payload's fixed prefix is 24 bytes, then the two length-prefixed
 * version strings, then the blob with its own length. */
K26RlStatus k26rl_tap_reader_spec(const K26RlTapReader *r,
                                  const uint8_t **out_spec,
                                  uint32_t *out_len)
{
    const uint8_t *p, *end;
    uint32_t len;
    uint16_t n;

    if (!r || !out_spec || !out_len)
        return K26RL_E_NULL;
    p = r->base + K26RL_TAP_OFF_HEADER + K26RL_EPISODE_FRAME_HEADER_SIZE;
    end = r->base + K26RL_TAP_OFF_HEADER + r->header_len;
    if ((uint64_t)(end - p) < 24 + 2u)
        return K26RL_E_GEOMETRY;
    p += 24;
    n = k26rl_get_u16_(p);
    p += 2;
    if ((uint64_t)(end - p) < (uint64_t)n + 2)
        return K26RL_E_GEOMETRY;
    p += n;
    n = k26rl_get_u16_(p);
    p += 2;
    if ((uint64_t)(end - p) < (uint64_t)n + 4)
        return K26RL_E_GEOMETRY;
    p += n;
    len = k26rl_get_u32_(p);
    p += 4;
    if ((uint64_t)(end - p) < len)
        return K26RL_E_GEOMETRY;
    *out_spec = p;
    *out_len = len;
    return K26RL_OK;
}

K26RlStatus k26rl_tap_reader_info(const K26RlTapReader *r,
                                  uint32_t *out_slot_size,
                                  uint32_t *out_slot_count)
{
    if (!r)
        return K26RL_E_NULL;
    if (out_slot_size)
        *out_slot_size = r->slot_size;
    if (out_slot_count)
        *out_slot_count = r->slot_count;
    return K26RL_OK;
}

int k26rl_tap_reader_closed(const K26RlTapReader *r)
{
    if (!r)
        return 1;
    return k26rl_pub_load_u32_(r->base + K26RL_TAP_OFF_CLOSED) != 0;
}

uint64_t k26rl_tap_reader_accepted(const K26RlTapReader *r)
{
    return r ? r->accepted : 0;
}

uint64_t k26rl_tap_reader_lost(const K26RlTapReader *r)
{
    return r ? r->lost : 0;
}

void k26rl_tap_detach(K26RlTapReader *r)
{
    if (!r)
        return;
    munmap((void *)(uintptr_t)r->base, r->bytes);
    free(r);
}
