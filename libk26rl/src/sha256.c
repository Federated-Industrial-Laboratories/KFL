/* sha256.c - SHA-256 (FIPS 180-4) and the asset identity digest.
 *
 * A direct transcription of the standard's own description: the
 * message schedule of section 6.2.2, the eight working variables, and
 * the padding of section 5.1.1. Integer arithmetic on fixed-width
 * unsigned types throughout, so the result is exact on every platform
 * and does not depend on any floating-point setting.
 *
 * The round constants are the first thirty-two bits of the fractional
 * parts of the cube roots of the first sixty-four primes, and the
 * initial hash value is the same of the square roots of the first
 * eight; both are the standard's tables, and both are pinned by the
 * published example digests in tests/test_k26rl_digest.c rather than
 * by trust in this transcription.
 */
#include "k26rl_digest.h"

#include <string.h>

static const uint32_t K_[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t ror_(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32u - n));
}

static void block_(uint32_t h[8], const uint8_t p[64])
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
               ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ror_(w[i - 15], 7) ^ ror_(w[i - 15], 18) ^
                      (w[i - 15] >> 3);
        uint32_t s1 = ror_(w[i - 2], 17) ^ ror_(w[i - 2], 19) ^
                      (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1  = ror_(e, 6) ^ ror_(e, 11) ^ ror_(e, 25);
        uint32_t ch  = (e & f) ^ ((~e) & g);
        uint32_t t1  = hh + S1 + ch + K_[i] + w[i];
        uint32_t S0  = ror_(a, 2) ^ ror_(a, 13) ^ ror_(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2  = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

void k26rl_sha256_init(K26RlSha256 *s)
{
    if (!s) return;
    s->h[0] = 0x6a09e667u; s->h[1] = 0xbb67ae85u;
    s->h[2] = 0x3c6ef372u; s->h[3] = 0xa54ff53au;
    s->h[4] = 0x510e527fu; s->h[5] = 0x9b05688cu;
    s->h[6] = 0x1f83d9abu; s->h[7] = 0x5be0cd19u;
    s->len_bytes = 0;
    s->buf_len   = 0;
    memset(s->buf, 0, sizeof s->buf);
}

void k26rl_sha256_update(K26RlSha256 *s, const void *p, uint64_t len)
{
    if (!s || (!p && len)) return;
    const uint8_t *b = (const uint8_t *)p;
    s->len_bytes += len;
    while (len > 0) {
        uint64_t room = 64u - s->buf_len;
        uint64_t take = len < room ? len : room;
        memcpy(s->buf + s->buf_len, b, (size_t)take);
        s->buf_len += (uint32_t)take;
        b          += take;
        len        -= take;
        if (s->buf_len == 64u) {
            block_(s->h, s->buf);
            s->buf_len = 0;
        }
    }
}

void k26rl_sha256_final(K26RlSha256 *s, uint8_t out[K26RL_SHA256_BYTES])
{
    if (!s || !out) return;
    /* Section 5.1.1: append one 1 bit, then zeros, then the message
     * length in bits as a 64-bit big-endian value. The buffer holds
     * at most 63 bytes here, since a full block is consumed as soon
     * as it fills, so the first store below always has room. */
    uint64_t bits = s->len_bytes * 8u;
    s->buf[s->buf_len++] = 0x80u;
    if (s->buf_len > 56u) {
        memset(s->buf + s->buf_len, 0, 64u - s->buf_len);
        block_(s->h, s->buf);
        s->buf_len = 0;
    }
    memset(s->buf + s->buf_len, 0, 56u - s->buf_len);
    for (int i = 0; i < 8; i++) {
        s->buf[56 + i] = (uint8_t)(bits >> (56 - 8 * i));
    }
    block_(s->h, s->buf);
    s->buf_len = 0;
    for (int i = 0; i < 8; i++) {
        out[4 * i]     = (uint8_t)(s->h[i] >> 24);
        out[4 * i + 1] = (uint8_t)(s->h[i] >> 16);
        out[4 * i + 2] = (uint8_t)(s->h[i] >> 8);
        out[4 * i + 3] = (uint8_t)(s->h[i]);
    }
}

void k26rl_sha256(const void *p, uint64_t len,
                  uint8_t out[K26RL_SHA256_BYTES])
{
    K26RlSha256 s;
    k26rl_sha256_init(&s);
    k26rl_sha256_update(&s, p, len);
    k26rl_sha256_final(&s, out);
}

void k26rl_sha256_hex(const uint8_t d[K26RL_SHA256_BYTES],
                      char out[K26RL_SHA256_HEX])
{
    static const char HEX[] = "0123456789abcdef";
    if (!d || !out) return;
    for (int i = 0; i < K26RL_SHA256_BYTES; i++) {
        out[2 * i]     = HEX[(d[i] >> 4) & 0x0fu];
        out[2 * i + 1] = HEX[d[i] & 0x0fu];
    }
    out[2 * K26RL_SHA256_BYTES] = '\0';
}

void k26rl_digest_begin(K26RlSha256 *s)
{
    k26rl_sha256_init(s);
}

void k26rl_digest_add(K26RlSha256 *s, const void *p, uint64_t len)
{
    if (!s) return;
    uint8_t le[8];
    for (int i = 0; i < 8; i++) le[i] = (uint8_t)(len >> (8 * i));
    k26rl_sha256_update(s, le, 8);
    k26rl_sha256_update(s, p, len);
}

void k26rl_digest_final(K26RlSha256 *s, uint8_t out[K26RL_SHA256_BYTES])
{
    k26rl_sha256_final(s, out);
}
