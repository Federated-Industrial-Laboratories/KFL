/* k26rl_internal.h - shared byte assembly for the episode format.
 *
 * Every multi-byte integer in an episode file is little-endian and is
 * assembled or parsed field by field through these helpers, never by
 * writing a struct, so the on-disk bytes are identical on every host
 * regardless of endianness or padding. Doubles travel as their
 * IEEE-754 binary64 bit patterns carried in uint64 values. */
#ifndef K26RL_INTERNAL_H
#define K26RL_INTERNAL_H

#include <stdint.h>
#include <string.h>

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

#endif /* K26RL_INTERNAL_H */
