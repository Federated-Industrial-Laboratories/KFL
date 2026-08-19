/* test_k26rl_ref.c - the `.k26ref` plan format, its writer and its
 * reader.
 *
 * Acceptance:
 *   1. A plan encoded and read back reports what was put in: the
 *      frame, the epoch, the provenance, and every knot bit for bit.
 *      Two plans are used, differing in frame kind and knot count, so
 *      a reader that returned one fixed answer cannot pass.
 *   2. Identity. The digest covers the whole file with its own bytes
 *      read as zero, so altering any byte outside the digest field is
 *      refused, and so is altering a byte of the digest itself. Every
 *      byte of the file is tried, one at a time, rather than a chosen
 *      few: the claim is that no byte is outside the hashed region,
 *      and a fixture that flipped two bytes could not tell that from
 *      most of them being covered. The provenance and the frame name
 *      are inside it, which is checked by re-encoding with each
 *      changed and requiring a different digest.
 *   3. The refusals, each measured on a file that differs from an
 *      accepted one in exactly the field under test: the magic, the
 *      format version, the header size, a set flag bit, a non-zero
 *      reserved word, a frame kind outside the list, an empty frame
 *      name, a knot count of zero, a length short of what the
 *      sections imply, a length past it, knots out of time order, and
 *      a knot carrying a value that is not usable.
 *   4. Absent slots. A slot with a zero tolerance is absent, a
 *      producer's slots are ordered by time whatever order they were
 *      given in, and a slot that does not advance the clock is
 *      dropped. A negative tolerance is refused rather than read as
 *      absence.
 *   5. The current knot advances by time and not by arrival. It is
 *      taken either side of every boundary and exactly on it, and the
 *      plan advances at each although nothing in the fixture ever
 *      arrives anywhere: the reader has no craft to compare against,
 *      which is the strongest available statement that arrival is not
 *      what moves it. After the last knot's time the last knot stays
 *      current.
 *   6. The plan frame in the episode record. A reference written into
 *      a record comes back byte for byte, with the digest the frame
 *      carries; a record whose reference bytes were altered still
 *      carries the digest of what was intended, so the two disagree
 *      and the alteration is visible from the record alone.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    exit(1); } } while (0)

#include "k26rl_episode.h"
#include "k26rl_ref.h"

/* Header field offsets, restated here rather than shared with the
 * reader: a test that took the offsets from the code under test could
 * not fail on an offset that moved. */
#define OFF_MAGIC      0
#define OFF_VERSION    8
#define OFF_HEADER     12
#define OFF_DIGEST     16
#define OFF_FLAGS      48
#define OFF_FRAME_KIND 52
#define OFF_FRAME_LEN  56
#define OFF_KNOTS      60
#define OFF_PROVENANCE 64
#define OFF_RESERVED   68
#define OFF_EPOCH      72
#define HEADER_BYTES   80
#define KNOT_BYTES     64

static int g_arms;

static void put32_(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)(v >> 24);
}

static uint32_t get32_(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put64_(uint8_t *p, uint64_t v)
{
    put32_(p, (uint32_t)(v & 0xFFFFFFFFu));
    put32_(p + 4, (uint32_t)(v >> 32));
}

static void putf64_(uint8_t *p, double v)
{
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    put64_(p, bits);
}

static K26RlRefKnot knot_(double t, double x, double y, double z,
                          double vx, double vy, double vz, double tol)
{
    K26RlRefKnot k;

    k.t = t;
    k.r[0] = x; k.r[1] = y; k.r[2] = z;
    k.v[0] = vx; k.v[1] = vy; k.v[2] = vz;
    k.tolerance = tol;
    return k;
}

/* ---- arm 1: round trip ---------------------------------------------- */

static void arm_round_trip_(void)
{
    K26RlRefKnot ks[3];
    K26RlRefPlan plan;
    uint8_t *bytes = NULL;
    uint64_t len = 0;
    K26RlRef *ref = NULL;
    K26RlRefInfo info;
    uint32_t i, n;
    const char *s;

    ks[0] = knot_(10.0, 100.0, -20.0, 3.5, 1.0, -0.5, 0.25, 5.0);
    ks[1] = knot_(120.0, 0.0, -200.0, 0.0, 0.0, -1.5, 0.0, 25.0);
    ks[2] = knot_(600.0, -3.0, -10.0, 1.0, 0.0, -0.1, 0.0, 0.5);

    plan.frame_kind = K26RL_REF_FRAME_LVLH;
    plan.frame_name = "station";
    plan.provenance = "test fixture, arm 1";
    plan.epoch      = 42.5;
    plan.knots      = ks;
    plan.knot_count = 3;

    ASSERT(k26rl_ref_encode(&plan, &bytes, &len) == K26RL_REF_OK);
    ASSERT(len == (uint64_t)HEADER_BYTES + strlen(plan.frame_name) +
                  strlen(plan.provenance) + 3u * KNOT_BYTES);
    ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_OK);
    ASSERT(k26rl_ref_info(ref, &info) == K26RL_REF_OK);
    ASSERT(info.format_version == K26RL_REF_FORMAT_VERSION);
    ASSERT(info.frame_kind == K26RL_REF_FRAME_LVLH);
    ASSERT(info.knot_count == 3 && info.present_count == 3);
    ASSERT(info.epoch == 42.5);

    s = k26rl_ref_frame_name(ref, &n);
    ASSERT(n == strlen("station") && strcmp(s, "station") == 0);
    s = k26rl_ref_provenance(ref, &n);
    ASSERT(n == strlen(plan.provenance) && strcmp(s, plan.provenance) == 0);

    for (i = 0; i < 3; i++) {
        K26RlRefKnot got;
        int j;

        ASSERT(k26rl_ref_knot(ref, i, &got) == K26RL_REF_OK);
        ASSERT(got.t == ks[i].t && got.tolerance == ks[i].tolerance);
        for (j = 0; j < 3; j++)
            ASSERT(got.r[j] == ks[i].r[j] && got.v[j] == ks[i].v[j]);
    }
    ASSERT(k26rl_ref_knot(ref, 3, &(K26RlRefKnot){ 0 }) ==
           K26RL_REF_E_RANGE);

    /* The plan keeps the bytes it was given, which is what lets a
     * consumer put a reference into a record without re-reading the
     * file it came from. */
    {
        uint64_t bl = 0;
        const uint8_t *b = k26rl_ref_bytes(ref, &bl);

        ASSERT(b != NULL && bl == len && memcmp(b, bytes, (size_t)len) == 0);
    }
    k26rl_ref_close(ref);
    free(bytes);

    /* A second plan differing in frame kind and knot count, so the
     * arm cannot pass on a reader that returns one fixed answer. */
    plan.frame_kind = K26RL_REF_FRAME_INERTIAL;
    plan.frame_name = "earth";
    plan.provenance = NULL;
    plan.epoch      = 0.0;
    plan.knot_count = 1;
    ASSERT(k26rl_ref_encode(&plan, &bytes, &len) == K26RL_REF_OK);
    ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_OK);
    ASSERT(k26rl_ref_info(ref, &info) == K26RL_REF_OK);
    ASSERT(info.frame_kind == K26RL_REF_FRAME_INERTIAL);
    ASSERT(info.knot_count == 1 && info.present_count == 1);
    ASSERT(info.provenance_bytes == 0);
    s = k26rl_ref_provenance(ref, &n);
    ASSERT(s != NULL && n == 0 && s[0] == '\0');
    s = k26rl_ref_frame_name(ref, &n);
    ASSERT(strcmp(s, "earth") == 0);
    k26rl_ref_close(ref);
    free(bytes);
    g_arms++;
}

/* ---- arm 2: identity ------------------------------------------------ */

static void encode_fixture_(uint8_t **out, uint64_t *out_len,
                            const char *frame, const char *prov)
{
    K26RlRefKnot ks[2];
    K26RlRefPlan plan;

    ks[0] = knot_(30.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 10.0);
    ks[1] = knot_(90.0, 4.0, 5.0, 6.0, 0.4, 0.5, 0.6, 2.0);
    plan.frame_kind = K26RL_REF_FRAME_LVLH;
    plan.frame_name = frame;
    plan.provenance = prov;
    plan.epoch      = 7.0;
    plan.knots      = ks;
    plan.knot_count = 2;
    ASSERT(k26rl_ref_encode(&plan, out, out_len) == K26RL_REF_OK);
}

static void arm_identity_(void)
{
    uint8_t *bytes = NULL, *other = NULL;
    uint64_t len = 0, olen = 0;
    K26RlRef *ref = NULL;
    uint64_t i;

    encode_fixture_(&bytes, &len, "station", "arm 2");
    ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_OK);
    k26rl_ref_close(ref);

    /* Every byte, one at a time. A byte inside the digest field
     * refuses because the stored digest no longer matches what the
     * file hashes to; every other byte refuses because the file
     * hashes to something else. Both are the same refusal and that is
     * the point: no byte of the file sits outside its identity. */
    for (i = 0; i < len; i++) {
        uint8_t *copy = malloc((size_t)len);
        K26RlRefStatus st;

        ASSERT(copy != NULL);
        memcpy(copy, bytes, (size_t)len);
        copy[i] = (uint8_t)(copy[i] ^ 0xFFu);
        ref = NULL;
        st = k26rl_ref_parse(copy, len, &ref);
        /* The magic and the version are checked before the digest, so
         * those eight and four bytes refuse by their own code; every
         * other byte must refuse as an altered file. */
        if (i < 8)
            ASSERT(st == K26RL_REF_E_MAGIC);
        else if (i < 16)
            ASSERT(st == K26RL_REF_E_VERSION);
        else
            ASSERT(st == K26RL_REF_E_DIGEST);
        ASSERT(ref == NULL);
        free(copy);
    }

    /* The frame name and the provenance are inside the hashed region,
     * so changing either changes the file's identity. */
    encode_fixture_(&other, &olen, "station", "arm 2, relabelled");
    ASSERT(olen != len ||
           memcmp(bytes + OFF_DIGEST, other + OFF_DIGEST, 32) != 0);
    ASSERT(memcmp(bytes + OFF_DIGEST, other + OFF_DIGEST, 32) != 0);
    free(other);
    encode_fixture_(&other, &olen, "chaser1", "arm 2");
    ASSERT(memcmp(bytes + OFF_DIGEST, other + OFF_DIGEST, 32) != 0);
    free(other);

    free(bytes);
    g_arms++;
}

/* ---- arm 3: the refusals -------------------------------------------- */

/* Rebuild the digest over a mutated buffer, so a refusal measured
 * after the mutation is the field's own and not the digest's. */
static void redigest_(uint8_t *b, uint64_t len)
{
    K26RlSha256 s;
    static const uint8_t zeros[32] = { 0 };
    uint8_t d[32];

    k26rl_sha256_init(&s);
    k26rl_sha256_update(&s, b, (uint64_t)OFF_DIGEST);
    k26rl_sha256_update(&s, zeros, 32);
    k26rl_sha256_update(&s, b + OFF_DIGEST + 32, len - (OFF_DIGEST + 32));
    k26rl_sha256_final(&s, d);
    memcpy(b + OFF_DIGEST, d, 32);
}

static K26RlRefStatus parse_mutated_(const uint8_t *src, uint64_t len,
                                     uint64_t new_len,
                                     void (*mutate)(uint8_t *, uint64_t))
{
    uint8_t *copy = calloc((size_t)(new_len > len ? new_len : len), 1u);
    K26RlRef *ref = NULL;
    K26RlRefStatus st;

    ASSERT(copy != NULL);
    memcpy(copy, src, (size_t)(new_len < len ? new_len : len));
    if (mutate)
        mutate(copy, new_len);
    redigest_(copy, new_len);
    st = k26rl_ref_parse(copy, new_len, &ref);
    if (st == K26RL_REF_OK)
        k26rl_ref_close(ref);
    free(copy);
    return st;
}

static void mut_magic_(uint8_t *b, uint64_t n)   { (void)n; b[3] = 'X'; }
static void mut_version_(uint8_t *b, uint64_t n) { (void)n;
                                                   put32_(b + OFF_VERSION, 2); }
static void mut_header_(uint8_t *b, uint64_t n)  { (void)n;
                                                   put32_(b + OFF_HEADER, 96); }
static void mut_flags_(uint8_t *b, uint64_t n)   { (void)n;
                                                   put32_(b + OFF_FLAGS, 1); }
static void mut_reserved_(uint8_t *b, uint64_t n){ (void)n;
                                                   put32_(b + OFF_RESERVED, 9); }
static void mut_frame_kind_(uint8_t *b, uint64_t n) { (void)n;
                                                   put32_(b + OFF_FRAME_KIND,
                                                          7); }
static void mut_epoch_(uint8_t *b, uint64_t n)   { (void)n;
                                                   putf64_(b + OFF_EPOCH,
                                                           (double)(1.0 /
                                                                    0.0)); }

/* An empty frame name, with the file's length and its knot section
 * moved down so that the length still equals what the sections imply:
 * the refusal must be the empty name and not the arithmetic. */
static void arm_refusals_(void)
{
    uint8_t *bytes = NULL;
    uint64_t len = 0;
    uint32_t frame_len, prov_len;
    uint64_t body;

    encode_fixture_(&bytes, &len, "station", "arm 3");
    frame_len = get32_(bytes + OFF_FRAME_LEN);
    prov_len  = get32_(bytes + OFF_PROVENANCE);
    body = (uint64_t)HEADER_BYTES + frame_len + prov_len;

    ASSERT(parse_mutated_(bytes, len, len, NULL) == K26RL_REF_OK);
    ASSERT(parse_mutated_(bytes, len, len, mut_magic_) == K26RL_REF_E_MAGIC);
    ASSERT(parse_mutated_(bytes, len, len, mut_version_) ==
           K26RL_REF_E_VERSION);
    ASSERT(parse_mutated_(bytes, len, len, mut_header_) ==
           K26RL_REF_E_VERSION);
    ASSERT(parse_mutated_(bytes, len, len, mut_flags_) == K26RL_REF_E_FLAGS);
    ASSERT(parse_mutated_(bytes, len, len, mut_reserved_) ==
           K26RL_REF_E_RESERVED);
    ASSERT(parse_mutated_(bytes, len, len, mut_frame_kind_) ==
           K26RL_REF_E_FRAME);
    ASSERT(parse_mutated_(bytes, len, len, mut_epoch_) == K26RL_REF_E_VALUE);

    /* An empty frame name. The knots move up by the name's length and
     * the file shrinks by it, so every other field still agrees. */
    {
        uint8_t *copy = malloc((size_t)len);
        K26RlRef *ref = NULL;
        uint64_t nlen = len - frame_len;

        ASSERT(copy != NULL);
        memcpy(copy, bytes, (size_t)HEADER_BYTES);
        memcpy(copy + HEADER_BYTES, bytes + HEADER_BYTES + frame_len,
               (size_t)(len - HEADER_BYTES - frame_len));
        put32_(copy + OFF_FRAME_LEN, 0);
        redigest_(copy, nlen);
        ASSERT(k26rl_ref_parse(copy, nlen, &ref) == K26RL_REF_E_FRAME);
        free(copy);
    }

    /* A knot count of zero, with the knot section removed so the
     * arithmetic still holds. */
    {
        uint8_t *copy = malloc((size_t)body);
        K26RlRef *ref = NULL;

        ASSERT(copy != NULL);
        memcpy(copy, bytes, (size_t)body);
        put32_(copy + OFF_KNOTS, 0);
        redigest_(copy, body);
        ASSERT(k26rl_ref_parse(copy, body, &ref) == K26RL_REF_E_KNOTS);
        free(copy);
    }

    /* One byte short of what the sections imply, and one byte past
     * it. Both are refused, so no byte of a plan is outside the
     * digest. */
    ASSERT(parse_mutated_(bytes, len, len - 1u, NULL) ==
           K26RL_REF_E_TRUNCATED);
    ASSERT(parse_mutated_(bytes, len, len + 1u, NULL) ==
           K26RL_REF_E_TRAILING);

    /* Knots out of time order, written straight into the file rather
     * than through the writer, which orders them. */
    {
        uint8_t *copy = malloc((size_t)len);
        K26RlRef *ref = NULL;

        ASSERT(copy != NULL);
        memcpy(copy, bytes, (size_t)len);
        putf64_(copy + body + KNOT_BYTES, 5.0);   /* second knot earlier */
        redigest_(copy, len);
        ASSERT(k26rl_ref_parse(copy, len, &ref) == K26RL_REF_E_ORDER);
        /* Equal times are refused too: two knots at one instant say
         * two things about it. */
        putf64_(copy + body + KNOT_BYTES, 30.0);
        redigest_(copy, len);
        ASSERT(k26rl_ref_parse(copy, len, &ref) == K26RL_REF_E_ORDER);
        free(copy);
    }

    /* A knot carrying a value that is not usable: a position that is
     * not a number, and a negative tolerance, which is refused rather
     * than read as absence. */
    {
        uint8_t *copy = malloc((size_t)len);
        K26RlRef *ref = NULL;

        ASSERT(copy != NULL);
        memcpy(copy, bytes, (size_t)len);
        putf64_(copy + body + 8, (double)(0.0 / 0.0));
        redigest_(copy, len);
        ASSERT(k26rl_ref_parse(copy, len, &ref) == K26RL_REF_E_VALUE);
        memcpy(copy, bytes, (size_t)len);
        putf64_(copy + body + 7u * 8u, -1.0);
        redigest_(copy, len);
        ASSERT(k26rl_ref_parse(copy, len, &ref) == K26RL_REF_E_VALUE);
        free(copy);
    }

    free(bytes);
    g_arms++;
}

/* ---- arm 4: absent slots and ordering -------------------------------- */

static void arm_slots_(void)
{
    K26RlRefKnot ks[5];
    K26RlRefPlan plan;
    uint8_t *bytes = NULL;
    uint64_t len = 0;
    K26RlRef *ref = NULL;
    K26RlRefInfo info;
    K26RlRefKnot got;

    /* Given out of order, with two absent slots and one that repeats
     * a time already taken. What survives is three knots at 10, 40 and
     * 90, in that order. */
    ks[0] = knot_(40.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0);
    ks[1] = knot_(0.0, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 0.0);   /* absent */
    ks[2] = knot_(10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0);
    ks[3] = knot_(40.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);  /* repeats 40 */
    ks[4] = knot_(90.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0);

    plan.frame_kind = K26RL_REF_FRAME_INERTIAL;
    plan.frame_name = "earth";
    plan.provenance = "arm 4";
    plan.epoch      = 0.0;
    plan.knots      = ks;
    plan.knot_count = 5;
    ASSERT(k26rl_ref_encode(&plan, &bytes, &len) == K26RL_REF_OK);
    ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_OK);
    ASSERT(k26rl_ref_info(ref, &info) == K26RL_REF_OK);
    ASSERT(info.knot_count == 3 && info.present_count == 3);
    ASSERT(k26rl_ref_knot(ref, 0, &got) == K26RL_REF_OK);
    ASSERT(got.t == 10.0 && got.r[0] == 1.0 && got.tolerance == 7.0);
    ASSERT(k26rl_ref_knot(ref, 1, &got) == K26RL_REF_OK);
    /* The first slot given at time 40 is the one kept, which is what
     * makes one set of slots encode to one file. */
    ASSERT(got.t == 40.0 && got.r[0] == 4.0);
    ASSERT(k26rl_ref_knot(ref, 2, &got) == K26RL_REF_OK);
    ASSERT(got.t == 90.0 && got.r[0] == 9.0);
    k26rl_ref_close(ref);

    /* One set of slots, encoded twice, is one file. */
    {
        uint8_t *again = NULL;
        uint64_t alen = 0;

        ASSERT(k26rl_ref_encode(&plan, &again, &alen) == K26RL_REF_OK);
        ASSERT(alen == len && memcmp(again, bytes, (size_t)len) == 0);
        free(again);
    }
    free(bytes);

    /* Every slot absent encodes to a file with no knot, which is a
     * producer that asked for nothing; loading it says so. */
    {
        K26RlRefKnot none[2];

        none[0] = knot_(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        none[1] = knot_(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        plan.knots = none;
        plan.knot_count = 2;
        ASSERT(k26rl_ref_encode(&plan, &bytes, &len) == K26RL_REF_OK);
        ASSERT(get32_(bytes + OFF_KNOTS) == 0);
        ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_E_KNOTS);
        free(bytes);
    }

    /* A value that is not usable is refused at encode rather than
     * written into a file the reader would then refuse. */
    {
        K26RlRefKnot bad[1];

        bad[0] = knot_(1.0, (double)(1.0 / 0.0), 0.0, 0.0, 0.0, 0.0, 0.0,
                       5.0);
        plan.knots = bad;
        plan.knot_count = 1;
        ASSERT(k26rl_ref_encode(&plan, &bytes, &len) == K26RL_REF_E_VALUE);
    }
    g_arms++;
}

/* ---- arm 5: the current knot ----------------------------------------- */

static void arm_current_(void)
{
    K26RlRefKnot ks[3];
    K26RlRefPlan plan;
    uint8_t *bytes = NULL;
    uint64_t len = 0;
    K26RlRef *ref = NULL;
    uint32_t cur;

    /* Times 10, 40 and 90 past an epoch of 100, so the absolute
     * boundaries are 110, 140 and 190. The epoch is not zero on
     * purpose: a reader that compared the offset alone would pass
     * every check below at an epoch of zero. */
    ks[0] = knot_(10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    ks[1] = knot_(40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    ks[2] = knot_(90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    plan.frame_kind = K26RL_REF_FRAME_INERTIAL;
    plan.frame_name = "earth";
    plan.provenance = "arm 5";
    plan.epoch      = 100.0;
    plan.knots      = ks;
    plan.knot_count = 3;
    ASSERT(k26rl_ref_encode(&plan, &bytes, &len) == K26RL_REF_OK);
    ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_OK);

    ASSERT(k26rl_ref_current(ref, 0.0, &cur) == K26RL_REF_OK && cur == 0);
    ASSERT(k26rl_ref_current(ref, 109.5, &cur) == K26RL_REF_OK && cur == 0);
    /* Exactly on the boundary the knot has not passed. */
    ASSERT(k26rl_ref_current(ref, 110.0, &cur) == K26RL_REF_OK && cur == 0);
    ASSERT(k26rl_ref_current(ref, 110.5, &cur) == K26RL_REF_OK && cur == 1);
    ASSERT(k26rl_ref_current(ref, 139.5, &cur) == K26RL_REF_OK && cur == 1);
    ASSERT(k26rl_ref_current(ref, 140.0, &cur) == K26RL_REF_OK && cur == 1);
    ASSERT(k26rl_ref_current(ref, 140.5, &cur) == K26RL_REF_OK && cur == 2);
    ASSERT(k26rl_ref_current(ref, 190.0, &cur) == K26RL_REF_OK && cur == 2);
    /* Past the last knot's time the last knot stays current. */
    ASSERT(k26rl_ref_current(ref, 1.0e6, &cur) == K26RL_REF_OK && cur == 2);

    k26rl_ref_close(ref);
    free(bytes);
    g_arms++;
}

/* ---- arm 6: the plan frame in a record ------------------------------- */

static void arm_record_(void)
{
    const char *path = "/tmp/k26rl_ref_record.k26epi";
    K26RlEpisodeGeom geom;
    K26RlEpisodeWriter *w = NULL;
    K26RlEpisodeReader *r = NULL;
    K26RlEpisodePlan got;
    K26RlRefInfo info;
    K26RlRef *ref = NULL;
    uint8_t *bytes = NULL;
    uint64_t len = 0;
    uint32_t count = 0;
    uint8_t spec[8] = { 0 };
    double obs[2] = { 0.0, 0.0 };

    encode_fixture_(&bytes, &len, "station", "arm 6");
    ASSERT(k26rl_ref_parse(bytes, len, &ref) == K26RL_REF_OK);
    ASSERT(k26rl_ref_info(ref, &info) == K26RL_REF_OK);

    remove(path);
    geom.n_envs = 1;
    geom.agent_count = 1;
    geom.obs_total = 2;
    geom.act_total = 1;
    geom.steps_per_chunk = 4;
    geom.dr_max = 0;
    ASSERT(k26rl_episode_writer_open(path, &geom, 7u, 0u, "3.2", "0.1",
                                     spec, (uint32_t)sizeof spec,
                                     &w) == K26RL_OK);
    ASSERT(k26rl_episode_writer_plan(w, K26RL_PLAN_ROLE_FLOWN,
                                     K26RL_PLAN_ALL, K26RL_PLAN_ALL,
                                     info.digest, bytes,
                                     (uint32_t)len) == K26RL_OK);
    ASSERT(k26rl_episode_writer_start(w, 0, 0, obs, NULL, NULL, 0) ==
           K26RL_OK);
    {
        double adj[1] = { 0.0 };

        ASSERT(k26rl_episode_writer_end(w, 0, K26RL_END_TRUNCATED, 0,
                                        adj) == K26RL_OK);
    }
    ASSERT(k26rl_episode_writer_close(w) == K26RL_OK);

    ASSERT(k26rl_episode_reader_open(path, &r) == K26RL_OK);
    ASSERT(k26rl_episode_reader_plans(r, &count) == K26RL_OK);
    ASSERT(count == 1);
    ASSERT(k26rl_episode_reader_plan(r, 0, &got) == K26RL_OK);
    ASSERT(got.role == K26RL_PLAN_ROLE_FLOWN);
    ASSERT(got.env == K26RL_PLAN_ALL && got.episode == K26RL_PLAN_ALL);
    ASSERT(got.len == (uint32_t)len);
    ASSERT(memcmp(got.bytes, bytes, (size_t)len) == 0);
    ASSERT(memcmp(got.digest, info.digest, 32) == 0);

    /* The recovered bytes are a plan again, and they carry the digest
     * the frame carries: the recording is sufficient. */
    {
        K26RlRef *back = NULL;
        K26RlRefInfo binfo;

        ASSERT(k26rl_ref_parse(got.bytes, got.len, &back) == K26RL_REF_OK);
        ASSERT(k26rl_ref_info(back, &binfo) == K26RL_REF_OK);
        ASSERT(memcmp(binfo.digest, got.digest, 32) == 0);
        k26rl_ref_close(back);
    }

    /* A reference altered inside the record no longer agrees with the
     * digest the record carries beside it, so a recording flown
     * against a plan with the same name is distinguishable from one
     * flown against the plan itself. */
    {
        K26RlRef *back = NULL;

        got.bytes[got.len - 1u] ^= 0xFFu;
        ASSERT(k26rl_ref_parse(got.bytes, got.len, &back) ==
               K26RL_REF_E_DIGEST);
        ASSERT(back == NULL);
    }
    k26rl_episode_plan_free(&got);
    k26rl_episode_reader_close(r);
    k26rl_ref_close(ref);
    free(bytes);
    remove(path);
    g_arms++;
}

int main(void)
{
    arm_round_trip_();
    arm_identity_();
    arm_refusals_();
    arm_slots_();
    arm_current_();
    arm_record_();
    ASSERT(g_arms == 6);
    printf("test_k26rl_ref: ok (%d arms)\n", g_arms);
    return 0;
}
