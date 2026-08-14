/* test_k26rl_digest.c - SHA-256 and the asset identity digest.
 *
 * Acceptance:
 *   1. Known answers. The digests of the standard's own published
 *      example messages, plus the empty message and the one-million-
 *      character message, match byte for byte. A transcription error
 *      in the round constants or the schedule fails here rather than
 *      shifting every digest the compiler stamps into an artifact.
 *   2. Streaming equals one-shot. The same bytes fed in one call, in
 *      single-byte calls, and split across a block boundary give one
 *      digest, so a caller's chunking cannot change an identity.
 *   3. Framing. The identity digest's length prefixes do what they
 *      exist for: two file sets whose bytes concatenate to the same
 *      sequence have different digests, and a byte moved across a
 *      file boundary changes the digest.
 *
 * Provenance of the expected values: the four messages and their
 * digests are the examples published with the standard; each was also
 * recomputed here with the system's own sha256sum, an independent
 * implementation, and the two agree. The recomputation is what the
 * comment above means by anchored, and it is repeatable with
 *   printf 'abc' | sha256sum
 *
 * Wire: see libk26rl/Makefile.
 */
#include "k26rl_digest.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* NDEBUG-immune: a gate built with release flags must still gate. */
#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

static int n_pass = 0;

static void expect_hex_(const char *tag, const void *p, uint64_t len,
                        const char *want)
{
    uint8_t d[K26RL_SHA256_BYTES];
    char    hex[K26RL_SHA256_HEX];
    k26rl_sha256(p, len, d);
    k26rl_sha256_hex(d, hex);
    if (strcmp(hex, want) != 0) {
        fprintf(stderr, "case `%s`:\n  got  %s\n  want %s\n", tag, hex, want);
    }
    ASSERT(strcmp(hex, want) == 0);
    printf("  %-14s %s: OK\n", tag, hex);
    n_pass++;
}

int main(void)
{
    printf("known answers (the standard's published examples):\n");

    expect_hex_("empty", "", 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    expect_hex_("abc", "abc", 3,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    expect_hex_("two-block",
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56,
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    {
        /* One million 'a', the standard's long example: the case that
         * exercises the block loop and the 64-bit length field. */
        char *big = (char *)malloc(1000000);
        ASSERT(big != NULL);
        memset(big, 'a', 1000000);
        expect_hex_("million-a", big, 1000000,
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        free(big);
    }

    /* ---- Streaming equals one-shot ---------------------------- */
    {
        static const char msg[] =
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            "the framing of an asset digest must not depend on how the "
            "caller happened to chunk the file it was reading";
        uint64_t n = (uint64_t)(sizeof msg - 1);

        uint8_t one[K26RL_SHA256_BYTES];
        k26rl_sha256(msg, n, one);

        K26RlSha256 s;
        uint8_t byte_at_a_time[K26RL_SHA256_BYTES];
        k26rl_sha256_init(&s);
        for (uint64_t i = 0; i < n; i++) k26rl_sha256_update(&s, msg + i, 1);
        k26rl_sha256_final(&s, byte_at_a_time);
        ASSERT(memcmp(one, byte_at_a_time, sizeof one) == 0);

        /* Split at 63, 64, and 65 bytes: either side of a block. */
        for (uint64_t cut = 63; cut <= 65; cut++) {
            uint8_t split[K26RL_SHA256_BYTES];
            k26rl_sha256_init(&s);
            k26rl_sha256_update(&s, msg, cut);
            k26rl_sha256_update(&s, msg + cut, n - cut);
            k26rl_sha256_final(&s, split);
            ASSERT(memcmp(one, split, sizeof one) == 0);
        }
        printf("  streaming equals one-shot, byte-wise and across a block "
               "boundary: OK\n");
        n_pass++;
    }

    /* ---- The identity digest's framing ------------------------- */
    {
        /* Two file sets whose bytes concatenate identically. Without
         * the length prefixes these would share a digest, and a
         * renamed asset would compile to the same program identity as
         * the one it replaced. */
        K26RlSha256 a, b;
        uint8_t da[K26RL_SHA256_BYTES], db[K26RL_SHA256_BYTES];

        k26rl_digest_begin(&a);
        k26rl_digest_add(&a, "assembly", 8);
        k26rl_digest_add(&a, "mesh", 4);
        k26rl_digest_final(&a, da);

        k26rl_digest_begin(&b);
        k26rl_digest_add(&b, "assembly" "m", 9);
        k26rl_digest_add(&b, "esh", 3);
        k26rl_digest_final(&b, db);

        ASSERT(memcmp(da, db, sizeof da) != 0);
        printf("  a byte moved across a file boundary changes the "
               "digest: OK\n");
        n_pass++;

        /* Order is part of the identity too. */
        K26RlSha256 c;
        uint8_t dc[K26RL_SHA256_BYTES];
        k26rl_digest_begin(&c);
        k26rl_digest_add(&c, "mesh", 4);
        k26rl_digest_add(&c, "assembly", 8);
        k26rl_digest_final(&c, dc);
        ASSERT(memcmp(da, dc, sizeof da) != 0);
        printf("  reordering the contributions changes the digest: OK\n");
        n_pass++;

        /* And the same set in the same order is stable. */
        K26RlSha256 e;
        uint8_t de[K26RL_SHA256_BYTES];
        k26rl_digest_begin(&e);
        k26rl_digest_add(&e, "assembly", 8);
        k26rl_digest_add(&e, "mesh", 4);
        k26rl_digest_final(&e, de);
        ASSERT(memcmp(da, de, sizeof da) == 0);
        printf("  the same contributions in the same order are stable: "
               "OK\n");
        n_pass++;

        /* An empty contribution is a contribution: a referenced but
         * empty mesh file is not the same asset as no mesh at all. */
        K26RlSha256 f, g;
        uint8_t df[K26RL_SHA256_BYTES], dg[K26RL_SHA256_BYTES];
        k26rl_digest_begin(&f);
        k26rl_digest_add(&f, "assembly", 8);
        k26rl_digest_final(&f, df);
        k26rl_digest_begin(&g);
        k26rl_digest_add(&g, "assembly", 8);
        k26rl_digest_add(&g, NULL, 0);
        k26rl_digest_final(&g, dg);
        ASSERT(memcmp(df, dg, sizeof df) != 0);
        printf("  an empty contribution still changes the digest: OK\n");
        n_pass++;
    }

    printf("test_k26rl_digest: %d check(s) passed\n", n_pass);
    return 0;
}
