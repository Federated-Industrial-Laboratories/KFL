/* crc32c.c - CRC-32C (Castagnoli), the episode file frame checksum.
 *
 * Table-driven over the reflected polynomial 0x82F63B78, initial and
 * final value 0xFFFFFFFF. The inversions are folded into the call
 * boundary rather than kept in the running state: the argument and
 * the return value are always the finished CRC, so
 * k26rl_crc32c(0, data, len) is the standard CRC-32C of data and
 * k26rl_crc32c(prev, more, len) continues it, because re-inverting a
 * finished CRC recovers the internal register exactly. The published
 * check values are pinned in tests/test_k26rl_crc.c. */
#include "k26rl_episode.h"

#include <stdatomic.h>

/* The 256-entry table is built from the polynomial on first use. The
 * ready flag is atomic: the release store below pairs with the
 * acquire load on the fast path, so a thread that observes the flag
 * set also observes the table stores that precede it. Concurrent
 * first calls may build the table more than once, which stays
 * harmless: every builder computes and stores the identical values,
 * so a double build writes the same bytes. */
static uint32_t table_[256];
static atomic_int table_ready_;

static void table_init_(void)
{
    uint32_t i;
    for (i = 0; i < 256; i++) {
        uint32_t c = i;
        int k;
        for (k = 0; k < 8; k++)
            c = (c & 1u) ? (0x82F63B78u ^ (c >> 1)) : (c >> 1);
        table_[i] = c;
    }
    atomic_store_explicit(&table_ready_, 1, memory_order_release);
}

uint32_t k26rl_crc32c(uint32_t crc_in, const void *data, uint64_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = crc_in ^ 0xFFFFFFFFu;
    uint64_t i;

    if (!atomic_load_explicit(&table_ready_, memory_order_acquire))
        table_init_();
    for (i = 0; i < len; i++)
        c = table_[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
