/* test_rl_policy.c: the trained-policy file format, its reader and
 * its evaluator.
 *
 * Arms:
 *   1. Round trip. A policy built here loads, and every field it
 *      declared comes back: geometry, layer shapes, activations,
 *      provenance text, and the action log standard deviation.
 *   2. The forward pass, against arithmetic written out longhand in
 *      this file rather than looped. Two layers of different widths
 *      and two different activations, so a reversed layer order
 *      cannot even chain and swapped activations cannot agree; the
 *      swap is its own arm and must change the result.
 *   3. Standardisation and clamping. Each is held against the same
 *      longhand, and each is shown to bite: the clip clamps a channel
 *      the raw observation drives outside it, and the clamp holds an
 *      output that would otherwise leave its bounds.
 *   4. Identity. A single bit flipped anywhere in the file is
 *      refused, one arm per region of the layout, including the
 *      provenance text, which is inside the hashed region precisely
 *      so that relabelling a policy is not free. A file re-sealed
 *      after being lengthened or shortened is refused as trailing or
 *      truncated rather than accepted.
 *   5. The file path and the null contract: a policy read from disk
 *      acts identically to the same bytes read from memory, an absent
 *      path is an input error, every entry point refuses a null
 *      argument, and an unknown status still decodes. Beside it, a
 *      policy whose slices start away from zero reads and writes its
 *      own slice of a shared vector and leaves the rest of it alone.
 *   6. Structural refusals, each on a file whose digest was
 *      recomputed after the change, so the arm measures the check it
 *      names and not the digest.
 *   7. The artifact check, one arm per field. A two-agent world gives
 *      each agent a slice that is a strict sub-range of the totals,
 *      which is what lets an offset be moved while every other field
 *      stays right; a fixture that could only vary a width could not
 *      tell a field-wise check from a check on totals. Each arm
 *      requires that field's own refusal code, and the refusal must
 *      name both numbers.
 *   8. Agreement with the framework that produced the weights, over
 *      real observations, at the tolerance policy_fixture.h states
 *      and justifies. A format that round-trips its bytes and then
 *      computes a different action is worse than useless.
 *   9. Repeatability. Evaluating a second policy between two
 *      evaluations of the first leaves the first's answer bit for
 *      bit, and the same weights over the same observations give the
 *      same bits in two separate processes, which the gate obtains by
 *      re-running itself.
 *
 * Every arm but the seventh needs no compiled world and always runs;
 * the seventh needs the sibling stack archives and reports itself
 * skipped without them, rather than taking the whole gate down with
 * it and leaving the format unchecked.
 */
#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rl_gate_util.h"

#include "k26rl_policy.h"
#include "policy_fixture.h"

#define WORK_DIR "/tmp/kflc_rl_policy_test"

/* Arms run, so the closing line carries a count: a gate that stopped
 * running half of what it holds would otherwise print the line it
 * prints when it runs all of it. */
static int g_arms;

/* ---- This file's own encoder ----------------------------------------
 *
 * The gate writes the bytes itself rather than calling anything the
 * library ships, so that a layout the reader gets wrong cannot be
 * cancelled by a writer that gets it wrong the same way. The digest
 * is the library's SHA-256, which is anchored separately against the
 * published vectors of its own standard, so nothing about the layout
 * rests on it. */

typedef struct {
    uint32_t in_width;
    uint32_t out_width;
    uint16_t activation;
    uint16_t reserved;
    const double *weights;      /* out_width rows of in_width */
    const double *biases;       /* out_width */
} PolLayerSpec;

typedef struct {
    uint32_t flags;
    uint32_t agent_count, agent_index;
    uint32_t obs_total, act_total;
    uint32_t obs_offset, obs_width;
    uint32_t act_offset, act_width;
    uint32_t layer_count;
    const PolLayerSpec *layers;
    const char *provenance;
    const double *mean, *variance;      /* obs_width each */
    double epsilon, clip;
    const double *log_std;              /* act_width */
    const double *lower, *upper;        /* act_width each */
} PolBuild;

static void put_u16_(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8) & 0xFFu);
}

static void put_u32_(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xFFu);
    p[1] = (unsigned char)((v >> 8) & 0xFFu);
    p[2] = (unsigned char)((v >> 16) & 0xFFu);
    p[3] = (unsigned char)((v >> 24) & 0xFFu);
}

static void put_f64_(unsigned char *p, double v)
{
    uint64_t bits;
    int i;

    memcpy(&bits, &v, sizeof bits);
    for (i = 0; i < 8; i++)
        p[i] = (unsigned char)((bits >> (8 * i)) & 0xFFu);
}

/* Recompute the digest a file carries, over its own bytes with the
 * digest field read as zero. */
static void pol_seal_(unsigned char *buf, size_t len)
{
    static const unsigned char zeros[K26RL_SHA256_BYTES] = { 0 };
    K26RlSha256 s;
    unsigned char digest[K26RL_SHA256_BYTES];

    ASSERT(len >= K26RL_POLICY_HEADER_BYTES);
    k26rl_sha256_init(&s);
    k26rl_sha256_update(&s, buf, (uint64_t)K26RL_POLICY_DIGEST_OFFSET);
    k26rl_sha256_update(&s, zeros, (uint64_t)K26RL_SHA256_BYTES);
    k26rl_sha256_update(&s,
        buf + K26RL_POLICY_DIGEST_OFFSET + K26RL_SHA256_BYTES,
        (uint64_t)len - (K26RL_POLICY_DIGEST_OFFSET + K26RL_SHA256_BYTES));
    k26rl_sha256_final(&s, digest);
    memcpy(buf + K26RL_POLICY_DIGEST_OFFSET, digest, sizeof digest);
}

static size_t pol_encode_(const PolBuild *b, unsigned char *out, size_t cap)
{
    size_t at = 0;
    uint32_t i, o, k;
    size_t prov_len = b->provenance ? strlen(b->provenance) : 0;

    ASSERT(cap > K26RL_POLICY_HEADER_BYTES + prov_len);
    memset(out, 0, K26RL_POLICY_HEADER_BYTES);
    memcpy(out, K26RL_POLICY_MAGIC, 8);
    put_u32_(out + 8, K26RL_POLICY_FORMAT_VERSION);
    put_u32_(out + 12, K26RL_POLICY_HEADER_BYTES);
    put_u32_(out + 48, b->flags);
    put_u32_(out + 52, b->agent_count);
    put_u32_(out + 56, b->agent_index);
    put_u32_(out + 60, b->obs_total);
    put_u32_(out + 64, b->act_total);
    put_u32_(out + 68, b->obs_offset);
    put_u32_(out + 72, b->obs_width);
    put_u32_(out + 76, b->act_offset);
    put_u32_(out + 80, b->act_width);
    put_u32_(out + 84, b->layer_count);
    put_u32_(out + 88, (uint32_t)prov_len);
    put_u32_(out + 92, 0);
    at = K26RL_POLICY_HEADER_BYTES;
    memcpy(out + at, b->provenance ? b->provenance : "", prov_len);
    at += prov_len;

    for (i = 0; i < b->layer_count; i++) {
        const PolLayerSpec *l = &b->layers[i];

        ASSERT(at + 12 + (size_t)8 *
               ((size_t)l->in_width * l->out_width + l->out_width) <= cap);
        put_u32_(out + at, l->in_width);
        put_u32_(out + at + 4, l->out_width);
        put_u16_(out + at + 8, l->activation);
        put_u16_(out + at + 10, l->reserved);
        at += 12;
        for (o = 0; o < l->out_width; o++)
            for (k = 0; k < l->in_width; k++, at += 8)
                put_f64_(out + at, l->weights[(size_t)o * l->in_width + k]);
        for (o = 0; o < l->out_width; o++, at += 8)
            put_f64_(out + at, l->biases[o]);
    }
    if (b->flags & K26RL_POLICY_FLAG_STANDARDISE) {
        ASSERT(at + (size_t)8 * (2u * b->obs_width + 2u) <= cap);
        for (k = 0; k < b->obs_width; k++, at += 8)
            put_f64_(out + at, b->mean[k]);
        for (k = 0; k < b->obs_width; k++, at += 8)
            put_f64_(out + at, b->variance[k]);
        put_f64_(out + at, b->epsilon);
        put_f64_(out + at + 8, b->clip);
        at += 16;
    }
    if (b->flags & K26RL_POLICY_FLAG_LOG_STD) {
        ASSERT(at + (size_t)8 * b->act_width <= cap);
        for (k = 0; k < b->act_width; k++, at += 8)
            put_f64_(out + at, b->log_std[k]);
    }
    if (b->flags & K26RL_POLICY_FLAG_CLAMP) {
        ASSERT(at + (size_t)16 * b->act_width <= cap);
        for (k = 0; k < b->act_width; k++, at += 8)
            put_f64_(out + at, b->lower[k]);
        for (k = 0; k < b->act_width; k++, at += 8)
            put_f64_(out + at, b->upper[k]);
    }
    pol_seal_(out, at);
    return at;
}

static void expect_status_(const unsigned char *bytes, size_t len,
                           K26RlPolicyStatus want, const char *what)
{
    K26RlPolicy *p = NULL;
    K26RlPolicyStatus got = k26rl_policy_parse(bytes, (uint64_t)len, &p);

    g_arms++;
    if (got != want) {
        fprintf(stderr, "FAIL %s: expected %d (%s), got %d (%s)\n", what,
                (int)want, k26rl_policy_status_str(want), (int)got,
                k26rl_policy_status_str(got));
        exit(1);
    }
    k26rl_policy_close(p);
}

/* ---- The fixture network -------------------------------------------- */

/* Three inputs, four hidden units, two outputs: widths that differ at
 * every layer, so a reversed order does not chain and a transposed
 * matrix does not fit. */
static const double W1[12] = {
     0.5, -1.5,  2.0,
    -0.25, 0.75, 1.25,
     2.5,  0.0, -0.5,
     1.0,  1.0,  1.0
};
static const double B1[4] = { 0.125, -0.5, 0.25, 0.0 };
static const double W2[8] = {
     1.0, -2.0,  0.5, 0.25,
     0.0,  1.5, -1.0, 2.0
};
static const double B2[2] = { -0.75, 0.5 };
static const double OBS[3] = { 0.3, -0.7, 1.1 };
static const double LOG_STD[2] = { -1.25, -0.5 };

static PolLayerSpec net_layers_[2] = {
    { 3, 4, K26RL_POLICY_ACT_TANH, 0, W1, B1 },
    { 4, 2, K26RL_POLICY_ACT_LOGISTIC, 0, W2, B2 }
};

static void net_build_(PolBuild *b)
{
    memset(b, 0, sizeof *b);
    b->flags = K26RL_POLICY_FLAG_LOG_STD;
    b->agent_count = 1;
    b->agent_index = 0;
    b->obs_total = 3;
    b->act_total = 2;
    b->obs_width = 3;
    b->act_width = 2;
    b->layer_count = 2;
    b->layers = net_layers_;
    b->provenance = "gate fixture";
    b->log_std = LOG_STD;
}

/* The forward pass over OBS, written out rather than looped, so that
 * agreement is agreement with a second statement of the arithmetic
 * and not with a second call into the same loop. */
static void net_expect_(const double *x, double *out)
{
    double h0 = tanh(0.125 + 0.5 * x[0] + -1.5 * x[1] + 2.0 * x[2]);
    double h1 = tanh(-0.5 + -0.25 * x[0] + 0.75 * x[1] + 1.25 * x[2]);
    double h2 = tanh(0.25 + 2.5 * x[0] + 0.0 * x[1] + -0.5 * x[2]);
    double h3 = tanh(0.0 + 1.0 * x[0] + 1.0 * x[1] + 1.0 * x[2]);

    out[0] = 1.0 / (1.0 + exp(-(-0.75 + 1.0 * h0 + -2.0 * h1 +
                                0.5 * h2 + 0.25 * h3)));
    out[1] = 1.0 / (1.0 + exp(-(0.5 + 0.0 * h0 + 1.5 * h1 +
                                -1.0 * h2 + 2.0 * h3)));
}

/* ---- Arms 1 to 3 ---------------------------------------------------- */

static void arm_round_trip_and_forward_(void)
{
    unsigned char buf[4096];
    PolBuild b;
    PolLayerSpec swapped[2];
    K26RlPolicy *p = NULL;
    K26RlPolicyInfo info;
    K26RlPolicyLayer layer;
    const double *log_std;
    double got[2], want[2], other[2];
    uint32_t prov_len = 0;
    size_t len;

    net_build_(&b);
    len = pol_encode_(&b, buf, sizeof buf);
    ASSERT(k26rl_policy_parse(buf, len, &p) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_info(p, &info) == K26RL_POLICY_OK);
    ASSERT(info.format_version == K26RL_POLICY_FORMAT_VERSION);
    ASSERT(info.agent_count == 1 && info.agent_index == 0);
    ASSERT(info.obs_total == 3 && info.act_total == 2);
    ASSERT(info.obs_offset == 0 && info.obs_width == 3);
    ASSERT(info.act_offset == 0 && info.act_width == 2);
    ASSERT(info.layer_count == 2);
    ASSERT(info.flags == K26RL_POLICY_FLAG_LOG_STD);
    ASSERT(memcmp(info.digest, buf + K26RL_POLICY_DIGEST_OFFSET,
                  K26RL_SHA256_BYTES) == 0);
    g_arms++;

    ASSERT(k26rl_policy_layer(p, 0, &layer) == K26RL_POLICY_OK);
    ASSERT(layer.in_width == 3 && layer.out_width == 4);
    ASSERT(layer.activation == K26RL_POLICY_ACT_TANH);
    ASSERT(k26rl_policy_layer(p, 1, &layer) == K26RL_POLICY_OK);
    ASSERT(layer.in_width == 4 && layer.out_width == 2);
    ASSERT(layer.activation == K26RL_POLICY_ACT_LOGISTIC);
    ASSERT(k26rl_policy_layer(p, 2, &layer) == K26RL_POLICY_E_LAYERS);
    g_arms++;

    ASSERT(strcmp(k26rl_policy_provenance(p, &prov_len),
                  "gate fixture") == 0);
    ASSERT(prov_len == strlen("gate fixture"));
    log_std = k26rl_policy_log_std(p);
    ASSERT(log_std != NULL);
    ASSERT(log_std[0] == LOG_STD[0] && log_std[1] == LOG_STD[1]);
    g_arms++;

    ASSERT(k26rl_policy_act(p, OBS, got) == K26RL_POLICY_OK);
    net_expect_(OBS, want);
    ASSERT(got[0] == want[0] && got[1] == want[1]);
    g_arms++;
    k26rl_policy_close(p);

    /* The two activations swapped: the same weights in the same
     * order, so anything that agreed here would be reading neither. */
    swapped[0] = net_layers_[0];
    swapped[1] = net_layers_[1];
    swapped[0].activation = K26RL_POLICY_ACT_LOGISTIC;
    swapped[1].activation = K26RL_POLICY_ACT_TANH;
    b.layers = swapped;
    len = pol_encode_(&b, buf, sizeof buf);
    p = NULL;
    ASSERT(k26rl_policy_parse(buf, len, &p) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(p, OBS, other) == K26RL_POLICY_OK);
    ASSERT(fabs(other[0] - want[0]) > 1e-3);
    ASSERT(fabs(other[1] - want[1]) > 1e-3);
    g_arms++;
    k26rl_policy_close(p);
    b.layers = net_layers_;

    /* A policy carrying no log standard deviation publishes none,
     * rather than publishing zeros a caller would sample from. */
    b.flags = 0;
    len = pol_encode_(&b, buf, sizeof buf);
    p = NULL;
    ASSERT(k26rl_policy_parse(buf, len, &p) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_log_std(p) == NULL);
    g_arms++;
    k26rl_policy_close(p);
}

static void arm_standardise_and_clamp_(void)
{
    /* These numbers are chosen, not arbitrary: on all three channels
     * the declared division by the square root and a multiplication
     * by its reciprocal give different last bits. The comparison
     * below is exact, so this arm holds the arithmetic the format
     * declares rather than an algebraically equal substitute, which
     * is the whole of what lets two implementations agree bit for
     * bit. */
    static const double MEAN[3] = { 0.1, -0.2, 0.3 };
    static const double VAR[3]  = { 3.0, 0.9, 3.0 };
    static const double LOWER[2] = { 0.0, 0.62 };
    static const double UPPER[2] = { 1.0, 0.64 };
    static const double FAR[3] = { 40.0, -0.7, 1.1 };
    unsigned char buf[4096];
    PolBuild b;
    K26RlPolicy *p = NULL;
    double got[2], want[2], standardised[3];
    size_t len;
    int i;

    net_build_(&b);
    b.flags |= K26RL_POLICY_FLAG_STANDARDISE;
    b.mean = MEAN;
    b.variance = VAR;
    b.epsilon = 1e-8;
    b.clip = 10.0;
    len = pol_encode_(&b, buf, sizeof buf);
    ASSERT(k26rl_policy_parse(buf, len, &p) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(p, OBS, got) == K26RL_POLICY_OK);
    for (i = 0; i < 3; i++)
        standardised[i] = (OBS[i] - MEAN[i]) / sqrt(VAR[i] + 1e-8);
    net_expect_(standardised, want);
    ASSERT(got[0] == want[0] && got[1] == want[1]);
    g_arms++;

    /* The clip is not decoration: an observation far outside the
     * trained range must reach the network as the clip, not as
     * itself. */
    ASSERT((FAR[0] - MEAN[0]) / sqrt(VAR[0] + 1e-8) > 10.0);
    ASSERT(k26rl_policy_act(p, FAR, got) == K26RL_POLICY_OK);
    standardised[0] = 10.0;
    for (i = 1; i < 3; i++)
        standardised[i] = (FAR[i] - MEAN[i]) / sqrt(VAR[i] + 1e-8);
    net_expect_(standardised, want);
    ASSERT(got[0] == want[0] && got[1] == want[1]);
    g_arms++;
    k26rl_policy_close(p);

    /* The clamp, on bounds narrow enough to bite one output and not
     * the other. */
    net_build_(&b);
    b.flags |= K26RL_POLICY_FLAG_CLAMP;
    b.lower = LOWER;
    b.upper = UPPER;
    len = pol_encode_(&b, buf, sizeof buf);
    p = NULL;
    ASSERT(k26rl_policy_parse(buf, len, &p) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(p, OBS, got) == K26RL_POLICY_OK);
    net_expect_(OBS, want);
    ASSERT(want[0] > LOWER[0] && want[0] < UPPER[0]);
    ASSERT(want[1] > UPPER[1]);
    ASSERT(got[0] == want[0]);
    ASSERT(got[1] == UPPER[1]);
    g_arms++;
    k26rl_policy_close(p);
}

/* ---- Arm 4: identity ------------------------------------------------ */

static void arm_digest_(void)
{
    static const double MEAN[3] = { 0.1, -0.2, 0.0 };
    static const double VAR[3]  = { 4.0, 0.25, 1.0 };
    static const double LOWER[2] = { -1.0, -1.0 };
    static const double UPPER[2] = { 2.0, 2.0 };
    unsigned char buf[4096], mutated[4096];
    struct { size_t off; const char *what; } spots[12];
    PolBuild b;
    size_t len, i, prov, layer0, layer1, stats, log_std, clamp;

    net_build_(&b);
    b.flags |= K26RL_POLICY_FLAG_STANDARDISE | K26RL_POLICY_FLAG_CLAMP;
    b.mean = MEAN;
    b.variance = VAR;
    b.epsilon = 1e-8;
    b.clip = 10.0;
    b.lower = LOWER;
    b.upper = UPPER;
    len = pol_encode_(&b, buf, sizeof buf);
    expect_status_(buf, len, K26RL_POLICY_OK, "the sealed file");

    /* The section starts, derived from the layout rather than written
     * down: a hardcoded offset that drifted into the wrong section
     * would still flip a bit and still fail the digest, so the arm
     * would keep passing while its coverage rotted and its names
     * lied. Every section of the file is named once here, so the set
     * below covers the file rather than a sample of it. */
    prov    = K26RL_POLICY_HEADER_BYTES;
    layer0  = prov + strlen(b.provenance);
    layer1  = layer0 + 12u + 8u * (3u * 4u + 4u);
    stats   = layer1 + 12u + 8u * (4u * 2u + 2u);
    log_std = stats + 8u * (2u * 3u + 2u);
    clamp   = log_std + 8u * 2u;
    ASSERT(clamp + 8u * 2u * 2u == len);

    i = 0;
#define SPOT_(offset, name) do { \
        spots[i].off = (offset); \
        spots[i].what = (name); \
        i++; \
    } while (0)
    SPOT_(48,                        "the flags word");
    SPOT_(prov + 1,                  "the provenance text");
    SPOT_(layer0 + 12 + 8,           "a first-layer weight");
    SPOT_(layer0 + 12 + 8 * 12 + 8,  "a first-layer bias");
    SPOT_(layer1 + 12 + 8,           "a second-layer weight");
    SPOT_(layer1 + 12 + 8 * 8 + 8,   "a second-layer bias");
    SPOT_(stats + 8,                 "a standardisation mean");
    SPOT_(stats + 8 * 3 + 8,         "a standardisation variance");
    SPOT_(stats + 8 * 6,             "the standardisation epsilon");
    SPOT_(stats + 8 * 7,             "the standardisation clip");
    SPOT_(log_std + 8,               "the log standard deviation");
    SPOT_(clamp + 8 * 2 + 8,         "an upper action bound");
#undef SPOT_
    ASSERT(i == sizeof spots / sizeof spots[0]);

    for (i = 0; i < sizeof spots / sizeof spots[0]; i++) {
        char what[128];

        ASSERT(spots[i].off < len);
        memcpy(mutated, buf, len);
        mutated[spots[i].off] ^= 0x01u;
        snprintf(what, sizeof what, "one bit flipped in %s",
                 spots[i].what);
        expect_status_(mutated, len, K26RL_POLICY_E_DIGEST, what);
    }

    /* A file lengthened or shortened and then re-sealed passes its
     * digest and must still be refused, because the declared sections
     * account for every byte. */
    memcpy(mutated, buf, len);
    mutated[len] = 0x5Au;
    pol_seal_(mutated, len + 1);
    expect_status_(mutated, len + 1, K26RL_POLICY_E_TRAILING,
                   "a re-sealed file with a byte appended");
    memcpy(mutated, buf, len);
    pol_seal_(mutated, len - 8);
    expect_status_(mutated, len - 8, K26RL_POLICY_E_TRUNCATED,
                   "a re-sealed file with eight bytes removed");
}

/* ---- Arm 5: the file path and the null contract --------------------- */

static void arm_file_and_nulls_(void)
{
    unsigned char buf[4096];
    PolBuild b;
    K26RlPolicy *p = NULL, *q = NULL;
    K26RlPolicyInfo info;
    char path[512];
    double from_file[2], from_memory[2];
    size_t len;
    FILE *f;

    net_build_(&b);
    len = pol_encode_(&b, buf, sizeof buf);
    rl_run_or_die_("mkdir -p " WORK_DIR);
    snprintf(path, sizeof path, "%s/gate" K26RL_POLICY_SUFFIX, WORK_DIR);
    f = fopen(path, "wb");
    ASSERT(f != NULL);
    ASSERT(fwrite(buf, 1, len, f) == len);
    ASSERT(fclose(f) == 0);

    ASSERT(k26rl_policy_open(path, &p) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_parse(buf, len, &q) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_info(p, &info) == K26RL_POLICY_OK);
    ASSERT(memcmp(info.digest, buf + K26RL_POLICY_DIGEST_OFFSET,
                  K26RL_SHA256_BYTES) == 0);
    ASSERT(k26rl_policy_act(p, OBS, from_file) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(q, OBS, from_memory) == K26RL_POLICY_OK);
    ASSERT(from_file[0] == from_memory[0] && from_file[1] == from_memory[1]);
    k26rl_policy_close(p);
    p = NULL;
    g_arms++;

    /* A path that is not there is an input error, not a crash and not
     * a policy of zero layers. A refused load leaves the caller's
     * pointer alone, so it is still null here. */
    ASSERT(k26rl_policy_open(WORK_DIR "/absent" K26RL_POLICY_SUFFIX, &p) ==
           K26RL_POLICY_E_IO);
    ASSERT(p == NULL);
    g_arms++;

    ASSERT(k26rl_policy_open(NULL, &p) == K26RL_POLICY_E_NULL);
    ASSERT(k26rl_policy_parse(NULL, 0, &p) == K26RL_POLICY_E_NULL);
    ASSERT(k26rl_policy_info(NULL, &info) == K26RL_POLICY_E_NULL);
    ASSERT(k26rl_policy_act(q, NULL, from_memory) == K26RL_POLICY_E_NULL);
    ASSERT(k26rl_policy_act_env(q, OBS, NULL) == K26RL_POLICY_E_NULL);
    ASSERT(k26rl_policy_check_spec(q, NULL, 0, NULL, 0) ==
           K26RL_POLICY_E_NULL);
    ASSERT(k26rl_policy_provenance(NULL, NULL) == NULL);
    ASSERT(k26rl_policy_log_std(NULL) == NULL);
    k26rl_policy_close(NULL);
    g_arms++;

    /* An unknown status decodes to one definite string rather than
     * to nothing, since the registry only ever grows. */
    ASSERT(strcmp(k26rl_policy_status_str((K26RlPolicyStatus)9999),
                  "unknown policy status") == 0);
    ASSERT(strcmp(k26rl_policy_status_str(K26RL_POLICY_OK), "ok") == 0);
    g_arms++;

    k26rl_policy_close(q);
}

/* ---- Arm 5a: driving one agent's slice of a shared vector ----------- */

static void arm_slices_(void)
{
    unsigned char buf[4096];
    PolBuild b;
    K26RlPolicy *p = NULL;
    double env_obs[6], env_act[5], direct[2];
    size_t len;
    int i;

    /* A slice that starts away from zero on both vectors, so an
     * evaluator that ignored the offsets would read the wrong
     * observations and write over another agent's actions. */
    net_build_(&b);
    b.obs_total = 6;
    b.obs_offset = 2;
    b.act_total = 5;
    b.act_offset = 1;
    b.agent_count = 2;
    b.agent_index = 1;
    len = pol_encode_(&b, buf, sizeof buf);
    ASSERT(k26rl_policy_parse(buf, len, &p) == K26RL_POLICY_OK);

    for (i = 0; i < 6; i++)
        env_obs[i] = -9.0;
    for (i = 0; i < 3; i++)
        env_obs[2 + i] = OBS[i];
    for (i = 0; i < 5; i++)
        env_act[i] = -9.0;
    ASSERT(k26rl_policy_act_env(p, env_obs, env_act) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(p, OBS, direct) == K26RL_POLICY_OK);
    ASSERT(env_act[1] == direct[0] && env_act[2] == direct[1]);
    ASSERT(env_act[0] == -9.0 && env_act[3] == -9.0 && env_act[4] == -9.0);
    g_arms++;
    k26rl_policy_close(p);
}

/* ---- Arm 6: structural refusals ------------------------------------- */

static void arm_structure_(void)
{
    static const double MEAN[3] = { 0.1, -0.2, 0.0 };
    static const double VAR[3]  = { 4.0, 0.25, 1.0 };
    static const double LOWER[2] = { -1.0, -1.0 };
    static const double UPPER[2] = { 2.0, 2.0 };
    unsigned char buf[4096], m[4096];
    PolBuild b;
    PolLayerSpec broken[2];
    size_t len, layer0, stats;

    net_build_(&b);
    b.flags |= K26RL_POLICY_FLAG_STANDARDISE | K26RL_POLICY_FLAG_CLAMP;
    b.mean = MEAN;
    b.variance = VAR;
    b.epsilon = 1e-8;
    b.clip = 10.0;
    b.lower = LOWER;
    b.upper = UPPER;
    len = pol_encode_(&b, buf, sizeof buf);
    layer0 = K26RL_POLICY_HEADER_BYTES + strlen("gate fixture");
    stats = layer0 + 12 + 8 * (3 * 4 + 4) + 12 + 8 * (4 * 2 + 2);

#define MUTATE_(body, want, what) do { \
        memcpy(m, buf, len); \
        body; \
        pol_seal_(m, len); \
        expect_status_(m, len, (want), (what)); \
    } while (0)

    MUTATE_(m[3] = 'X', K26RL_POLICY_E_MAGIC, "a corrupted magic");
    MUTATE_(put_u32_(m + 8, 2), K26RL_POLICY_E_VERSION,
            "an unserved format version");
    MUTATE_(put_u32_(m + 12, 100), K26RL_POLICY_E_VERSION,
            "a header length this version does not use");
    MUTATE_(put_u32_(m + 92, 1), K26RL_POLICY_E_RESERVED,
            "a reserved header field set");
    MUTATE_(put_u32_(m + 48, b.flags | (1u << 3)), K26RL_POLICY_E_FLAGS,
            "an unknown flag bit");
    MUTATE_(put_u16_(m + layer0 + 8, 9), K26RL_POLICY_E_ACTIVATION,
            "an activation outside the closed list");
    MUTATE_(put_u16_(m + layer0 + 10, 1), K26RL_POLICY_E_RESERVED,
            "a reserved layer field set");
    MUTATE_(put_u32_(m + 84, 0), K26RL_POLICY_E_LAYERS,
            "a policy declaring no layer");
    MUTATE_(put_u32_(m + 72, 0), K26RL_POLICY_E_SIZE,
            "an observation slice of width zero");
    MUTATE_(put_u32_(m + 56, 1), K26RL_POLICY_E_SLICE,
            "an agent index outside the declared agent count");
    MUTATE_(put_u32_(m + 68, 1), K26RL_POLICY_E_SLICE,
            "an observation slice running past its total");
    MUTATE_(put_f64_(m + stats + 8 * 3, -1.0), K26RL_POLICY_E_STATISTICS,
            "a negative variance");
    MUTATE_(put_f64_(m + stats + 8 * 7, 0.0), K26RL_POLICY_E_STATISTICS,
            "a clip of zero");
    MUTATE_(put_f64_(m + stats + 8 * 10, 5.0), K26RL_POLICY_E_STATISTICS,
            "a lower action bound above its upper bound");

#undef MUTATE_

    /* Widths that do not chain: the second layer takes one input
     * fewer than the first gives. The digest is recomputed, so it is
     * the chain check that refuses. */
    broken[0] = net_layers_[0];
    broken[1] = net_layers_[1];
    broken[1].in_width = 3;
    b.layers = broken;
    b.flags = K26RL_POLICY_FLAG_LOG_STD;
    len = pol_encode_(&b, buf, sizeof buf);
    expect_status_(buf, len, K26RL_POLICY_E_LAYERS,
                   "layers whose widths do not chain");

    /* A first layer that does not take the observation slice. */
    broken[1] = net_layers_[1];
    broken[0].in_width = 2;
    len = pol_encode_(&b, buf, sizeof buf);
    expect_status_(buf, len, K26RL_POLICY_E_LAYERS,
                   "a first layer narrower than the observation slice");

    /* A last layer that does not fill the action slice. */
    broken[0] = net_layers_[0];
    broken[1].out_width = 1;
    len = pol_encode_(&b, buf, sizeof buf);
    expect_status_(buf, len, K26RL_POLICY_E_LAYERS,
                   "a last layer narrower than the action slice");

    expect_status_((const unsigned char *)"K26POL\0\0", 8,
                   K26RL_POLICY_E_TRUNCATED, "a file shorter than a header");
}

/* ---- Arm 8: agreement with the producing framework ------------------ */

static void arm_fixture_one_(const unsigned char *bytes, size_t len,
                             const double *expected, const char *what)
{
    K26RlPolicy *p = NULL;
    K26RlPolicyInfo info;
    double action[POLICY_FIXTURE_ACT];
    double worst = 0.0;
    K26RlPolicyStatus st;
    int c, j;

    st = k26rl_policy_parse(bytes, (uint64_t)len, &p);
    if (st != K26RL_POLICY_OK) {
        fprintf(stderr, "FAIL %s: load refused: %s\n", what,
                k26rl_policy_status_str(st));
        exit(1);
    }
    ASSERT(k26rl_policy_info(p, &info) == K26RL_POLICY_OK);
    ASSERT(info.obs_width == POLICY_FIXTURE_OBS);
    ASSERT(info.act_width == POLICY_FIXTURE_ACT);
    ASSERT(info.flags & K26RL_POLICY_FLAG_STANDARDISE);
    ASSERT(info.flags & K26RL_POLICY_FLAG_LOG_STD);
    ASSERT(info.flags & K26RL_POLICY_FLAG_CLAMP);
    for (c = 0; c < POLICY_FIXTURE_CASES; c++) {
        ASSERT(k26rl_policy_act(
                   p, POLICY_FIXTURE_OBSERVATIONS + c * POLICY_FIXTURE_OBS,
                   action) == K26RL_POLICY_OK);
        for (j = 0; j < POLICY_FIXTURE_ACT; j++) {
            double d = fabs(action[j] -
                            expected[c * POLICY_FIXTURE_ACT + j]);

            if (d > worst)
                worst = d;
        }
    }
    printf("  %s: largest disagreement %.3g over %d observations\n",
           what, worst, POLICY_FIXTURE_CASES);
    if (!(worst <= POLICY_FIXTURE_TOLERANCE)) {
        fprintf(stderr, "FAIL %s: %.17g exceeds the tolerance %.17g\n",
                what, worst, (double)POLICY_FIXTURE_TOLERANCE);
        exit(1);
    }
    g_arms++;
    k26rl_policy_close(p);
}

static void arm_fixture_(void)
{
    arm_fixture_one_(POLICY_FIXTURE_TANH, sizeof POLICY_FIXTURE_TANH,
                     POLICY_FIXTURE_TANH_ACTIONS,
                     "a policy with hyperbolic tangent hidden layers");
    arm_fixture_one_(POLICY_FIXTURE_RELU, sizeof POLICY_FIXTURE_RELU,
                     POLICY_FIXTURE_RELU_ACTIONS,
                     "a policy with rectified hidden layers");
}

/* ---- Arm 9: the same weights and observation, twice over ------------ */

/* Print one policy's actions over the fixture observations, exactly.
 * The gate re-runs itself in this mode so the comparison crosses a
 * process boundary rather than only a loop. */
static void policy_dump_(void)
{
    K26RlPolicy *p = NULL;
    double action[POLICY_FIXTURE_ACT];
    int c, j;

    if (k26rl_policy_parse(POLICY_FIXTURE_TANH,
                           sizeof POLICY_FIXTURE_TANH,
                           &p) != K26RL_POLICY_OK)
        exit(1);
    for (c = 0; c < POLICY_FIXTURE_CASES; c++) {
        if (k26rl_policy_act(
                p, POLICY_FIXTURE_OBSERVATIONS + c * POLICY_FIXTURE_OBS,
                action) != K26RL_POLICY_OK)
            exit(1);
        for (j = 0; j < POLICY_FIXTURE_ACT; j++)
            printf("%.17g\n", action[j]);
    }
    k26rl_policy_close(p);
}

static void read_dump_(const char *self, char *out, size_t cap)
{
    char command[1024];
    FILE *pipe;
    size_t got;

    snprintf(command, sizeof command, "%s --dump", self);
    pipe = popen(command, "r");
    ASSERT(pipe != NULL);
    got = fread(out, 1, cap - 1, pipe);
    ASSERT(pclose(pipe) == 0);
    ASSERT(got > 0 && got < cap - 1);
    out[got] = '\0';
}

static void arm_repeatability_(const char *self)
{
    unsigned char buf[4096];
    PolBuild b;
    K26RlPolicy *a = NULL, *other = NULL;
    double first[2], second[2], scratch[POLICY_FIXTURE_ACT];
    static char run_a[65536], run_b[65536];
    size_t len;

    /* A second policy evaluated between two evaluations of the first
     * must leave the first's answer alone: the working memory an
     * evaluation uses belongs to the policy that owns it. */
    net_build_(&b);
    len = pol_encode_(&b, buf, sizeof buf);
    ASSERT(k26rl_policy_parse(buf, len, &a) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(a, OBS, first) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_parse(POLICY_FIXTURE_RELU,
                              sizeof POLICY_FIXTURE_RELU,
                              &other) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(other, POLICY_FIXTURE_OBSERVATIONS,
                            scratch) == K26RL_POLICY_OK);
    ASSERT(k26rl_policy_act(a, OBS, second) == K26RL_POLICY_OK);
    ASSERT(first[0] == second[0] && first[1] == second[1]);
    g_arms++;
    k26rl_policy_close(other);
    k26rl_policy_close(a);

    /* And the same weights and the same observations in two separate
     * processes give the same bits. */
    read_dump_(self, run_a, sizeof run_a);
    read_dump_(self, run_b, sizeof run_b);
    ASSERT(strcmp(run_a, run_b) == 0);
    g_arms++;
}

/* ---- Arm 7: the artifact check -------------------------------------- */

static const char *const TWO_KFL =
    "form TWO\n"
    "fn world w\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body alpha_craft gm=1.0 parent=earth pos_x=7.0e6"
    " vel_y=7546.0\n"
    "    astro_body beta_craft gm=1.0 parent=earth pos_x=1.1e7"
    " vel_y=6020.0\n"
    "    episode\n"
    "        control_dt 10.0\n"
    "        horizon 6\n"
    "    end\n"
    "    agent alpha\n"
    "        action thrust box -1.0 1.0 default 0.25\n"
    "        action yaw box -1.0 1.0 default 0.5\n"
    "        observe alpha_craft from earth mode=geometric as trk\n"
    "        objective\n"
    "            reward 0.0 - alpha.trk_range\n"
    "        end\n"
    "    end\n"
    "    agent beta\n"
    "        action thrust box -1.0 1.0 default 0.75\n"
    "        observe beta_craft from earth mode=geometric as trk\n"
    "        observe beta_craft from alpha_craft mode=geometric as rel\n"
    "        objective\n"
    "            reward 0.0 - beta.trk_range\n"
    "        end\n"
    "    end\n"
    "    on_step\n"
    "        alpha_craft.vel_x = alpha_craft.vel_x + alpha.thrust\n"
    "        beta_craft.vel_x = beta_craft.vel_x + beta.thrust\n"
    "        alpha_craft.vel_z = alpha_craft.vel_z + yaw\n"
    "    end\n"
    "end\n"
    "end\n";

/* A policy matching one agent's declared slices, with weights that
 * are only required to exist: this arm measures the check, and every
 * arithmetic property is measured elsewhere. */
typedef struct {
    PolBuild build;
    PolLayerSpec layer;
    double *weights;
    double *biases;
    double *lower;
    double *upper;
} FittedPolicy;

static void fitted_make_(FittedPolicy *f, const RlSpecView *v, uint32_t agent,
                         uint32_t obs_width, uint32_t act_width)
{
    uint32_t i;

    memset(f, 0, sizeof *f);
    f->weights = calloc((size_t)obs_width * act_width, sizeof *f->weights);
    f->biases = calloc(act_width, sizeof *f->biases);
    f->lower = calloc(act_width, sizeof *f->lower);
    f->upper = calloc(act_width, sizeof *f->upper);
    ASSERT(f->weights && f->biases && f->lower && f->upper);
    for (i = 0; i < (uint32_t)obs_width * act_width; i++)
        f->weights[i] = 0.01 * (double)(i % 7) - 0.03;
    for (i = 0; i < act_width; i++) {
        f->biases[i] = 0.1 * (double)i;
        f->lower[i] = -0.5;
        f->upper[i] = 0.5;
    }
    f->layer.in_width = obs_width;
    f->layer.out_width = act_width;
    f->layer.activation = K26RL_POLICY_ACT_TANH;
    f->layer.weights = f->weights;
    f->layer.biases = f->biases;
    f->build.flags = K26RL_POLICY_FLAG_CLAMP;
    f->build.agent_count = v->agent_count;
    f->build.agent_index = agent;
    f->build.obs_total = v->obs_total;
    f->build.act_total = v->act_total;
    f->build.obs_offset = v->obs_slice[agent][0];
    f->build.obs_width = obs_width;
    f->build.act_offset = v->act_slice[agent][0];
    f->build.act_width = act_width;
    f->build.layer_count = 1;
    f->build.layers = &f->layer;
    f->build.provenance = "artifact check";
    f->build.lower = f->lower;
    f->build.upper = f->upper;
}

static void fitted_free_(FittedPolicy *f)
{
    free(f->weights);
    free(f->biases);
    free(f->lower);
    free(f->upper);
}

/* A different slice offset that still leaves the slice inside its
 * total, so the arm that moves an offset is refused for the offset
 * rather than for running off the end. */
static uint32_t alt_offset_(uint32_t current, uint32_t width, uint32_t total)
{
    ASSERT(width < total);
    return current == 0 ? 1u : current - 1u;
}

static void expect_check_(const unsigned char *bytes, size_t len,
                          const uint8_t *spec, uint32_t spec_len,
                          K26RlPolicyStatus want, const char *what)
{
    K26RlPolicy *p = NULL;
    char detail[256];
    K26RlPolicyStatus got;

    ASSERT(k26rl_policy_parse(bytes, (uint64_t)len, &p) == K26RL_POLICY_OK);
    got = k26rl_policy_check_spec(p, spec, spec_len, detail, sizeof detail);
    g_arms++;
    if (got != want) {
        fprintf(stderr, "FAIL %s: expected %d (%s), got %d (%s): %s\n", what,
                (int)want, k26rl_policy_status_str(want), (int)got,
                k26rl_policy_status_str(got), detail);
        exit(1);
    }
    /* A refusal that named neither number would leave a caller with a
     * code and no way to see which side is wrong. */
    if (want != K26RL_POLICY_OK)
        ASSERT(strlen(detail) > 0);
    k26rl_policy_close(p);
}

static void arm_artifact_(void)
{
    unsigned char buf[8192];
    char kfl_path[512], so_path[512];
    void *so;
    RlSurface s;
    RlSpecView v;
    K26RlEnv *env = NULL;
    uint8_t *spec;
    int32_t spec_len;
    FittedPolicy f;
    uint32_t agent = 0, obs_width, act_width;
    size_t len;

    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    snprintf(kfl_path, sizeof kfl_path, "%s/two.kfl", WORK_DIR);
    snprintf(so_path, sizeof so_path, "%s/two", WORK_DIR);
    rl_write_file_(kfl_path, TWO_KFL);
    rl_compile_(kfl_path, so_path, WORK_DIR);
    strncat(so_path, ".rlenv.so", sizeof so_path - strlen(so_path) - 1);
    so = rl_dlopen_(so_path);
    rl_resolve_surface_(so, &s);
    ASSERT(s.create(7, 1, &env) == K26RL_OK);
    spec_len = s.spec(env, NULL, 0);
    ASSERT(spec_len > 0);
    spec = malloc((size_t)spec_len);
    ASSERT(spec != NULL);
    ASSERT(s.spec(env, spec, (uint32_t)spec_len) == spec_len);
    rl_parse_spec_(spec, (uint32_t)spec_len, &v);

    /* The fixture must give each agent a strict sub-range of the
     * totals, or an offset cannot be moved on its own and the offset
     * arms would be measuring a width. */
    ASSERT(v.agent_count == 2);
    ASSERT(v.n_obs_slices == 2 && v.n_act_slices == 2);
    ASSERT(v.obs_slice[0][1] < v.obs_total);
    ASSERT(v.act_slice[0][1] < v.act_total);
    obs_width = v.obs_slice[agent][1];
    act_width = v.act_slice[agent][1];
    printf("  world: %u agents, observation total %u, action total %u; "
           "agent 0 slices [%u,%u) and [%u,%u)\n",
           v.agent_count, v.obs_total, v.act_total,
           v.obs_slice[0][0], v.obs_slice[0][0] + v.obs_slice[0][1],
           v.act_slice[0][0], v.act_slice[0][0] + v.act_slice[0][1]);

    fitted_make_(&f, &v, agent, obs_width, act_width);
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len, K26RL_POLICY_OK,
                  "a policy matching agent 0");

    /* One arm per field, each varying that field alone. */
    f.build.agent_count = v.agent_count + 1u;
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_AGENT_COUNT, "a wrong agent count");
    f.build.agent_count = v.agent_count;

    f.build.obs_total = v.obs_total + 1u;
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_OBS_TOTAL, "a wrong observation total");
    f.build.obs_total = v.obs_total;

    f.build.act_total = v.act_total + 1u;
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_ACT_TOTAL, "a wrong action total");
    f.build.act_total = v.act_total;

    f.build.obs_offset = alt_offset_(v.obs_slice[agent][0], obs_width,
                                     v.obs_total);
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_OBS_OFFSET,
                  "a wrong observation slice offset");
    f.build.obs_offset = v.obs_slice[agent][0];

    f.build.act_offset = alt_offset_(v.act_slice[agent][0], act_width,
                                     v.act_total);
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_ACT_OFFSET, "a wrong action slice offset");
    f.build.act_offset = v.act_slice[agent][0];

    fitted_free_(&f);
    fitted_make_(&f, &v, agent, obs_width - 1u, act_width);
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_OBS_WIDTH,
                  "a wrong observation slice width");
    fitted_free_(&f);

    fitted_make_(&f, &v, agent, obs_width, act_width - 1u);
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_ACT_WIDTH, "a wrong action slice width");
    fitted_free_(&f);

    /* The clamp against the world's own declared bounds: narrower is
     * allowed, wider is not. */
    fitted_make_(&f, &v, agent, obs_width, act_width);
    f.lower[0] = -2.0;
    f.upper[0] = 2.0;
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len,
                  K26RL_POLICY_E_ACT_BOUNDS,
                  "a clamp wider than the world's action bounds");
    f.lower[0] = -0.25;
    f.upper[0] = 0.25;
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len, K26RL_POLICY_OK,
                  "a clamp narrower than the world's action bounds");
    fitted_free_(&f);

    /* The other agent's policy, so that the check is not passing on
     * the one agent whose offsets happen to be zero. */
    fitted_make_(&f, &v, 1, v.obs_slice[1][1], v.act_slice[1][1]);
    len = pol_encode_(&f.build, buf, sizeof buf);
    expect_check_(buf, len, spec, (uint32_t)spec_len, K26RL_POLICY_OK,
                  "a policy matching agent 1");
    fitted_free_(&f);

    free(spec);
    s.destroy(env);
    dlclose(so);
}

int main(int argc, char **argv)
{
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        policy_dump_();
        return 0;
    }
    printf("test_rl_policy: the trained-policy file format\n");
    arm_round_trip_and_forward_();
    arm_standardise_and_clamp_();
    arm_digest_();
    arm_file_and_nulls_();
    arm_slices_();
    arm_structure_();
    arm_fixture_();
    arm_repeatability_(argv[0]);
    if (rl_libs_present_("test_rl_policy")) {
        arm_artifact_();
    } else {
        printf("  artifact check arms: SKIP (stack libraries not built)\n");
    }
    printf("test_rl_policy: OK (%d arms)\n", g_arms);
    return 0;
}
