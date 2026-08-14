/* k26rl_internal.h - shared byte assembly for the episode format.
 *
 * Every multi-byte integer in an episode file is little-endian and is
 * assembled or parsed field by field through these helpers, never by
 * writing a struct, so the on-disk bytes are identical on every host
 * regardless of endianness or padding. Doubles travel as their
 * IEEE-754 binary64 bit patterns carried in uint64 values. */
#ifndef K26RL_INTERNAL_H
#define K26RL_INTERNAL_H

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "k26rl_env.h"

/* Frame header byte offsets within the 24-byte header. */
#define K26RL_FH_OFF_KIND     0
#define K26RL_FH_OFF_FLAGS    2
#define K26RL_FH_OFF_LENGTH   4
#define K26RL_FH_OFF_SEQUENCE 8
#define K26RL_FH_OFF_CRC      16
#define K26RL_FH_OFF_RESERVED 20

static inline void k26rl_put_u16_(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)(v >> 8);
}

static inline void k26rl_put_u32_(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)(v >> 24);
}

static inline void k26rl_put_u64_(uint8_t *p, uint64_t v)
{
    k26rl_put_u32_(p, (uint32_t)(v & 0xFFFFFFFFu));
    k26rl_put_u32_(p + 4, (uint32_t)(v >> 32));
}

static inline void k26rl_put_f64_(uint8_t *p, double v)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    k26rl_put_u64_(p, bits);
}

static inline uint16_t k26rl_get_u16_(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t k26rl_get_u32_(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t k26rl_get_u64_(const uint8_t *p)
{
    return (uint64_t)k26rl_get_u32_(p) |
           ((uint64_t)k26rl_get_u32_(p + 4) << 32);
}

static inline double k26rl_get_f64_(const uint8_t *p)
{
    uint64_t bits = k26rl_get_u64_(p);
    double v;
    memcpy(&v, &bits, sizeof v);
    return v;
}

/* Write a frame header into out (24 bytes) with the crc field zero,
 * as the checksum definition requires: the CRC is computed over these
 * bytes plus the payload and patched in afterwards. */
static inline void k26rl_frame_header_write_(uint8_t *out, uint16_t kind,
                                             uint32_t length,
                                             uint64_t sequence)
{
    k26rl_put_u16_(out + K26RL_FH_OFF_KIND, kind);
    k26rl_put_u16_(out + K26RL_FH_OFF_FLAGS, 0);
    k26rl_put_u32_(out + K26RL_FH_OFF_LENGTH, length);
    k26rl_put_u64_(out + K26RL_FH_OFF_SEQUENCE, sequence);
    k26rl_put_u32_(out + K26RL_FH_OFF_CRC, 0);
    k26rl_put_u32_(out + K26RL_FH_OFF_RESERVED, 0);
}

/* Shared-memory publication word access, used by the telemetry tap's
 * producer and its readers.
 *
 * Two disciplines meet in these four helpers. The bytes in shared
 * memory are little-endian whatever the host is, as everywhere else
 * in this format, so the value is assembled into a local word first
 * and that word's memory image is what crosses. The store and the
 * load are also the publication protocol's ordering points, so they
 * are atomic with release and acquire ordering: a reader that
 * observes a published sequence also observes the frame bytes written
 * before it. The addresses these are used on are naturally aligned by
 * the ring's layout, every slot and the cursor sitting on a cache
 * line boundary. */
static inline void k26rl_pub_store_u64_(uint8_t *p, uint64_t v)
{
    uint64_t word;

    k26rl_put_u64_((uint8_t *)&word, v);
    atomic_store_explicit((_Atomic uint64_t *)(void *)p, word,
                          memory_order_release);
}

static inline uint64_t k26rl_pub_load_u64_(const uint8_t *p)
{
    uint64_t word = atomic_load_explicit(
        (const _Atomic uint64_t *)(const void *)p, memory_order_acquire);

    return k26rl_get_u64_((const uint8_t *)&word);
}

static inline void k26rl_pub_store_u32_(uint8_t *p, uint32_t v)
{
    uint32_t word;

    k26rl_put_u32_((uint8_t *)&word, v);
    atomic_store_explicit((_Atomic uint32_t *)(void *)p, word,
                          memory_order_release);
}

static inline uint32_t k26rl_pub_load_u32_(const uint8_t *p)
{
    uint32_t word = atomic_load_explicit(
        (const _Atomic uint32_t *)(const void *)p, memory_order_acquire);

    return k26rl_get_u32_((const uint8_t *)&word);
}

/* The tap name rule, shared by the producer that creates a ring and
 * the reader that attaches to one, so a name either side rejects is
 * rejected for the same reason with the same status. */
static inline K26RlStatus k26rl_tap_name_ok_(const char *name)
{
    size_t i, n;

    if (!name)
        return K26RL_E_NULL;
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

#endif /* K26RL_INTERNAL_H */
