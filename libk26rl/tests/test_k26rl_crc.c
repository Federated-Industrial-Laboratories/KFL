/* test_k26rl_crc.c - CRC-32C conformance and continuation.
 *
 * Acceptance: the checksum reproduces the published CRC-32C check
 * values and continues correctly across split inputs, since the frame
 * CRC is computed incrementally over header then payload.
 *
 * Known-answer vectors: RFC 3720 appendix B.4 ("CRC Examples"). The
 * RFC lists each CRC as the four bytes appended to the message; those
 * bytes are the CRC value little-endian, so the listed "aa 36 91 8a"
 * for 32 bytes of zero pins the value 0x8A9136AA. */
#include "k26rl_episode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

int main(void)
{
    uint8_t buf[32];
    uint32_t i, whole;
    uint64_t split;

    /* 32 bytes of zero -> 0x8A9136AA. */
    memset(buf, 0, sizeof buf);
    ASSERT(k26rl_crc32c(K26RL_CRC32C_INIT, buf, 32) == 0x8A9136AAu);

    /* 32 bytes of 0xFF -> 0x62A8AB43. */
    memset(buf, 0xFF, sizeof buf);
    ASSERT(k26rl_crc32c(K26RL_CRC32C_INIT, buf, 32) == 0x62A8AB43u);

    /* 0x00 up to 0x1F: the RFC prints the CRC bytes "4e 79 dd 46",
     * which read little-endian is the value 0x46DD794E. */
    for (i = 0; i < 32; i++)
        buf[i] = (uint8_t)i;
    ASSERT(k26rl_crc32c(K26RL_CRC32C_INIT, buf, 32) == 0x46DD794Eu);

    /* 0x1F down to 0x00: printed "5c db 3f 11", value 0x113FDB5C. */
    for (i = 0; i < 32; i++)
        buf[i] = (uint8_t)(31 - i);
    ASSERT(k26rl_crc32c(K26RL_CRC32C_INIT, buf, 32) == 0x113FDB5Cu);

    /* The RFC's 48-byte iSCSI SCSI Read (10) command PDU
     * -> 0xD9963A56. */
    {
        static const uint8_t pdu[48] = {
            0x01, 0xC0, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x14, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x04, 0x00,
            0x00, 0x00, 0x00, 0x14,
            0x00, 0x00, 0x00, 0x18,
            0x28, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00
        };
        ASSERT(k26rl_crc32c(K26RL_CRC32C_INIT, pdu, sizeof pdu)
               == 0xD9963A56u);

        /* Continuation: crc(a + b) == crc(crc(a), b) at every split
         * of the PDU. */
        whole = k26rl_crc32c(K26RL_CRC32C_INIT, pdu, sizeof pdu);
        for (split = 0; split <= sizeof pdu; split++) {
            uint32_t part = k26rl_crc32c(K26RL_CRC32C_INIT, pdu, split);
            ASSERT(k26rl_crc32c(part, pdu + split, sizeof pdu - split)
                   == whole);
        }
    }

    /* Continuation on an irregular byte pattern as well. */
    for (i = 0; i < 32; i++)
        buf[i] = (uint8_t)(i * 7 + 3);
    whole = k26rl_crc32c(K26RL_CRC32C_INIT, buf, sizeof buf);
    for (split = 0; split <= sizeof buf; split++) {
        uint32_t part = k26rl_crc32c(K26RL_CRC32C_INIT, buf, split);
        ASSERT(k26rl_crc32c(part, buf + split, sizeof buf - split) == whole);
    }

    printf("test_k26rl_crc: ok\n");
    return 0;
}
