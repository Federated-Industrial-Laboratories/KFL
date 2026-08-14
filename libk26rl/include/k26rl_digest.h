/* k26rl_digest.h - SHA-256 and the asset identity digest.
 *
 * Two layers, one implementation.
 *
 * The lower layer is plain SHA-256 (FIPS 180-4), dependency-free C,
 * anchored by the standard's own published example digests in
 * tests/test_k26rl_digest.c.
 *
 * The upper layer is the identity digest for compiled-in assets. An
 * asset is a set of files: a vehicle assembly and the mesh files it
 * references. Their digest is what makes a changed asset a changed
 * program, so it must not be possible for two different file sets to
 * produce one digest by concatenating differently. Each contribution
 * is therefore framed by its own byte length before its bytes, so a
 * rename, a reordering, or a byte moved across a file boundary all
 * change the digest.
 *
 * The digest is computed once at compile time and travels in the
 * compiled program; nothing here runs on a stepping path.
 */
#ifndef K26RL_DIGEST_H
#define K26RL_DIGEST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define K26RL_SHA256_BYTES 32
#define K26RL_SHA256_HEX   65   /* 64 hex digits and the terminator */

/* Streaming SHA-256 state. Callers treat it as opaque; the layout is
 * exposed so it can live on the stack. */
typedef struct {
    uint32_t h[8];
    uint64_t len_bytes;         /* total message length fed so far */
    uint8_t  buf[64];           /* partial block */
    uint32_t buf_len;
} K26RlSha256;

/**
 * @brief Start a SHA-256 computation.
 * @param s State to initialise.
 */
void k26rl_sha256_init(K26RlSha256 *s);

/**
 * @brief Feed bytes into a SHA-256 computation.
 * @param s   State from k26rl_sha256_init.
 * @param p   Bytes to absorb; may be NULL when len is 0.
 * @param len Byte count.
 */
void k26rl_sha256_update(K26RlSha256 *s, const void *p, uint64_t len);

/**
 * @brief Finish a SHA-256 computation.
 * @param s   State; unusable afterwards without a fresh init.
 * @param out Receives the 32-byte digest.
 */
void k26rl_sha256_final(K26RlSha256 *s, uint8_t out[K26RL_SHA256_BYTES]);

/**
 * @brief One-shot SHA-256 over a single buffer.
 * @param p   Bytes; may be NULL when len is 0.
 * @param len Byte count.
 * @param out Receives the 32-byte digest.
 */
void k26rl_sha256(const void *p, uint64_t len,
                  uint8_t out[K26RL_SHA256_BYTES]);

/**
 * @brief Render a digest as lowercase hexadecimal.
 * @param d   The 32-byte digest.
 * @param out Receives 64 hex digits and a terminator.
 */
void k26rl_sha256_hex(const uint8_t d[K26RL_SHA256_BYTES],
                      char out[K26RL_SHA256_HEX]);

/**
 * @brief Start an asset identity digest.
 * @param s State to initialise.
 * @note  Contributions are framed by length; see k26rl_digest_add.
 */
void k26rl_digest_begin(K26RlSha256 *s);

/**
 * @brief Add one file's bytes to an asset identity digest.
 * @param s   State from k26rl_digest_begin.
 * @param p   The file's bytes; may be NULL when len is 0.
 * @param len The file's byte count.
 * @note  The length is absorbed first, as a little-endian 64-bit
 *        value, so two different file sets cannot share a digest by
 *        concatenating to the same byte sequence.
 */
void k26rl_digest_add(K26RlSha256 *s, const void *p, uint64_t len);

/**
 * @brief Finish an asset identity digest.
 * @param s   State; unusable afterwards without a fresh begin.
 * @param out Receives the 32-byte digest.
 */
void k26rl_digest_final(K26RlSha256 *s, uint8_t out[K26RL_SHA256_BYTES]);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_DIGEST_H */
