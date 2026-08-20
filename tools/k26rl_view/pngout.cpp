/* pngout.cpp - the PNG writer declared in pngout.h. */
#include "pngout.h"

#include <stdio.h>
#include <string.h>

#include <vector>

namespace k26rl_view {

namespace {

/* CRC-32 (ISO 3309), the PNG chunk checksum, table built once. */
uint32_t crc_table_[256];
bool crc_ready_ = false;

void crc_init_()
{
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++)
            c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
        crc_table_[n] = c;
    }
    crc_ready_ = true;
}

uint32_t crc_(uint32_t c, const uint8_t *p, size_t n)
{
    if (!crc_ready_)
        crc_init_();
    c ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++)
        c = crc_table_[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void be32_(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}

/* One chunk: length, type, payload, CRC over type and payload. */
bool chunk_(FILE *f, const char type[4], const uint8_t *data, size_t n)
{
    uint8_t head[8], tail[4];
    uint32_t c;

    be32_(head, (uint32_t)n);
    memcpy(head + 4, type, 4);
    /* Chaining works because the finalising xor of one call is
     * undone by the initialising xor of the next. */
    c = crc_(0, head + 4, 4);
    if (n)
        c = crc_(c, data, n);
    be32_(tail, c);
    return fwrite(head, 1, 8, f) == 8 &&
           (n == 0 || fwrite(data, 1, n, f) == n) &&
           fwrite(tail, 1, 4, f) == 4;
}

}  /* namespace */

bool png_write_rgb8(const char *path, const uint8_t *pixels,
                    uint32_t width, uint32_t height)
{
    if (!path || !pixels || !width || !height)
        return false;

    /* The zlib stream: header, stored deflate blocks over the
     * filtered scanlines (a zero filter byte before each row), and
     * the adler32 of the filtered bytes. Stored blocks carry at
     * most 65535 bytes each. */
    const size_t row = (size_t)width * 3 + 1;
    std::vector<uint8_t> raw(row * height);
    for (uint32_t y = 0; y < height; y++) {
        raw[y * row] = 0;
        memcpy(&raw[y * row + 1], pixels + (size_t)y * width * 3,
               (size_t)width * 3);
    }
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < raw.size(); i++) {
        a = (a + raw[i]) % 65521u;
        b = (b + a) % 65521u;
    }
    std::vector<uint8_t> z;
    z.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    z.push_back(0x78);
    z.push_back(0x01);
    size_t off = 0;
    while (off < raw.size()) {
        size_t n = raw.size() - off;
        if (n > 65535)
            n = 65535;
        int last = (off + n == raw.size()) ? 1 : 0;
        z.push_back((uint8_t)last);
        z.push_back((uint8_t)(n & 0xFF));
        z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 0xFF));
        z.push_back((uint8_t)((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + n);
        off += n;
    }
    uint8_t adler[4];
    be32_(adler, (b << 16) | a);
    z.insert(z.end(), adler, adler + 4);

    FILE *f = fopen(path, "wb");
    if (!f)
        return false;
    static const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    uint8_t ihdr[13];
    be32_(ihdr, width);
    be32_(ihdr + 4, height);
    ihdr[8] = 8;   /* bit depth */
    ihdr[9] = 2;   /* colour type: truecolour */
    ihdr[10] = 0;  /* compression */
    ihdr[11] = 0;  /* filter */
    ihdr[12] = 0;  /* interlace */
    bool ok = fwrite(sig, 1, 8, f) == 8 &&
              chunk_(f, "IHDR", ihdr, sizeof ihdr) &&
              chunk_(f, "IDAT", z.empty() ? 0 : &z[0], z.size()) &&
              chunk_(f, "IEND", 0, 0);
    if (fclose(f) != 0)
        ok = false;
    if (!ok)
        remove(path);
    return ok;
}

}  /* namespace k26rl_view */
