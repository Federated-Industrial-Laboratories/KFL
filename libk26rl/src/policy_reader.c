/* policy_reader.c - loading and evaluating a `.k26pol` policy.
 *
 * The layout, the evaluation rule and the identity rule are declared
 * in k26rl_policy.h; this file implements exactly those. Two
 * disciplines govern it.
 *
 * Nothing is indexed before it is verified. The file's bytes are read
 * whole, the digest is checked over them, and every declared size is
 * bounds-checked in 64-bit arithmetic against the byte count before
 * any of it is used to walk. A file that declares a layer wider than
 * the file is refused rather than trusted for one read.
 *
 * The arithmetic is the header's, literally. Standardisation divides
 * by a square root rather than multiplying by a reciprocal, and the
 * accumulation runs the declared order one operation at a time,
 * because the format's value is that two implementations agree and
 * either substitution moves the last bits.
 */
#include "k26rl_policy.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k26rl_env.h"
#include "k26rl_internal.h"

/* Header field byte offsets. */
#define POL_OFF_MAGIC       0
#define POL_OFF_VERSION     8
#define POL_OFF_HEADER      12
#define POL_OFF_DIGEST      16
#define POL_OFF_FLAGS       48
#define POL_OFF_AGENT_COUNT 52
#define POL_OFF_AGENT_INDEX 56
#define POL_OFF_OBS_TOTAL   60
#define POL_OFF_ACT_TOTAL   64
#define POL_OFF_OBS_OFFSET  68
#define POL_OFF_OBS_WIDTH   72
#define POL_OFF_OBS_CHANS   76
#define POL_OFF_ACT_OFFSET  80
#define POL_OFF_ACT_WIDTH   84
#define POL_OFF_LAYERS      88
#define POL_OFF_PROVENANCE  92
#define POL_OFF_RESERVED    96

/* Per-layer record bytes before the weights. */
#define POL_LAYER_HEADER 12

typedef struct {
    uint32_t in_width;
    uint32_t out_width;
    uint16_t activation;
    uint64_t weight_at;    /* index into the reals block */
    uint64_t bias_at;
} PolLayer;

struct K26RlPolicy {
    K26RlPolicyInfo info;
    char     *provenance;   /* terminated copy; never null once loaded */
    uint32_t *channels;     /* the observation channels, in file order */
    PolLayer *layers;
    double   *reals;        /* parameters, statistics, bounds, scratch */
    uint64_t  gather_at;    /* where act_env gathers the channels it reads */
    uint64_t  mean_at;
    uint64_t  scale_at;     /* sqrt(variance + epsilon), one per input */
    uint64_t  log_std_at;
    uint64_t  lower_at;
    uint64_t  upper_at;
    uint64_t  scratch_a;
    uint64_t  scratch_b;
    double    clip;
};

const char *k26rl_policy_status_str(K26RlPolicyStatus status)
{
    switch (status) {
    case K26RL_POLICY_OK:            return "ok";
    case K26RL_POLICY_E_NULL:        return "null pointer argument";
    case K26RL_POLICY_E_IO:          return "the policy file could not be "
                                            "read";
    case K26RL_POLICY_E_MEMORY:      return "allocation failed";
    case K26RL_POLICY_E_MAGIC:       return "not a policy file";
    case K26RL_POLICY_E_VERSION:     return "policy format version not served";
    case K26RL_POLICY_E_TRUNCATED:   return "policy file is shorter than it "
                                            "declares";
    case K26RL_POLICY_E_TRAILING:    return "policy file carries bytes past "
                                            "its declared end";
    case K26RL_POLICY_E_DIGEST:      return "policy digest disagrees with the "
                                            "file's own bytes";
    case K26RL_POLICY_E_SIZE:        return "a declared size exceeds this "
                                            "format's ceiling";
    case K26RL_POLICY_E_RESERVED:    return "a reserved field is not zero";
    case K26RL_POLICY_E_FLAGS:       return "an unknown flag bit is set";
    case K26RL_POLICY_E_ACTIVATION:  return "activation outside the format's "
                                            "closed list";
    case K26RL_POLICY_E_LAYERS:      return "layer widths do not chain from "
                                            "the observation slice to the "
                                            "action slice";
    case K26RL_POLICY_E_STATISTICS:  return "a declared statistic or bound is "
                                            "unusable";
    case K26RL_POLICY_E_SLICE:       return "a declared slice runs past its "
                                            "total";
    case K26RL_POLICY_E_SPEC:        return "the artifact's spec blob is "
                                            "malformed";
    case K26RL_POLICY_E_AGENT_COUNT: return "agent count disagrees with the "
                                            "artifact";
    case K26RL_POLICY_E_OBS_TOTAL:   return "observation total disagrees with "
                                            "the artifact";
    case K26RL_POLICY_E_ACT_TOTAL:   return "action total disagrees with the "
                                            "artifact";
    case K26RL_POLICY_E_OBS_OFFSET:  return "observation slice offset "
                                            "disagrees with the artifact";
    case K26RL_POLICY_E_OBS_WIDTH:   return "observation slice width "
                                            "disagrees with the artifact";
    case K26RL_POLICY_E_ACT_OFFSET:  return "action slice offset disagrees "
                                            "with the artifact";
    case K26RL_POLICY_E_ACT_WIDTH:   return "action slice width disagrees "
                                            "with the artifact";
    case K26RL_POLICY_E_ACT_BOUNDS:  return "the policy's action clamp "
                                            "reaches outside the artifact's "
                                            "declared bounds";
    case K26RL_POLICY_E_CHANNELS:    return "the observation channel list "
                                            "repeats a channel or leaves the "
                                            "declared observation slice";
    case K26RL_POLICY_E_OBS_CHANNELS: return "the observation channels the "
                                            "policy reads are not the ones "
                                            "the artifact publishes for it, "
                                            "or not in that order";
    }
    return "unknown policy status";
}

/* ---- loading -------------------------------------------------------- */

static int pol_activation_known_(uint16_t a)
{
    return a == K26RL_POLICY_ACT_IDENTITY || a == K26RL_POLICY_ACT_TANH ||
           a == K26RL_POLICY_ACT_RELU || a == K26RL_POLICY_ACT_LOGISTIC;
}

static int pol_width_ok_(uint32_t w)
{
    return w >= 1u && w <= K26RL_POLICY_MAX_WIDTH;
}

/* The digest as the format defines it: the whole file with its own 32
 * digest bytes read as zero. Provenance and every declared field sit
 * inside that region, so relabelling a policy changes its identity. */
static void pol_digest_(const uint8_t *b, uint64_t len,
                        uint8_t out[K26RL_SHA256_BYTES])
{
    static const uint8_t zeros[K26RL_SHA256_BYTES] = { 0 };
    K26RlSha256 s;

    k26rl_sha256_init(&s);
    k26rl_sha256_update(&s, b, (uint64_t)POL_OFF_DIGEST);
    k26rl_sha256_update(&s, zeros, (uint64_t)K26RL_SHA256_BYTES);
    k26rl_sha256_update(&s, b + POL_OFF_DIGEST + K26RL_SHA256_BYTES,
                        len - (uint64_t)(POL_OFF_DIGEST +
                                         K26RL_SHA256_BYTES));
    k26rl_sha256_final(&s, out);
}

/* Decode and check the observation channel list. Every entry must
 * lie inside the agent's declared slice and appear once: a repeated
 * channel would give the network two inputs from one measurement and
 * two standardisation entries for it, and a channel outside the slice
 * would have the policy read another agent's observation. The marker
 * array keeps that to one pass over the list; a file declares its own
 * channel count, so a hostile one must not be able to buy the square
 * of the ceiling in comparisons. */
static K26RlPolicyStatus pol_channels_(const uint8_t *bytes, uint64_t at,
                                       const K26RlPolicyInfo *in,
                                       uint32_t *out)
{
    unsigned char *seen;
    K26RlPolicyStatus st = K26RL_POLICY_OK;
    uint32_t i;

    seen = calloc((size_t)in->obs_width, 1u);
    if (!seen)
        return K26RL_POLICY_E_MEMORY;
    for (i = 0; i < in->obs_channel_count; i++) {
        uint32_t c = k26rl_get_u32_(bytes + at + (uint64_t)i * 4u);

        if (c < in->obs_offset || c - in->obs_offset >= in->obs_width ||
            seen[c - in->obs_offset]) {
            st = K26RL_POLICY_E_CHANNELS;
            break;
        }
        seen[c - in->obs_offset] = 1u;
        out[i] = c;
    }
    free(seen);
    return st;
}

static void pol_free_(K26RlPolicy *p)
{
    if (!p)
        return;
    free(p->provenance);
    free(p->channels);
    free(p->layers);
    free(p->reals);
    free(p);
}

K26RlPolicyStatus k26rl_policy_parse(const uint8_t *bytes, uint64_t len,
                                     K26RlPolicy **out)
{
    K26RlPolicy *p;
    K26RlPolicyInfo *in;
    K26RlPolicyStatus st;
    uint8_t computed[K26RL_SHA256_BYTES];
    uint64_t at, reals_n, params_n, scratch_w;
    uint32_t i;

    if (!bytes || !out)
        return K26RL_POLICY_E_NULL;
    if (len > K26RL_POLICY_MAX_BYTES)
        return K26RL_POLICY_E_SIZE;
    if (len < (uint64_t)K26RL_POLICY_HEADER_BYTES)
        return K26RL_POLICY_E_TRUNCATED;
    if (memcmp(bytes + POL_OFF_MAGIC, K26RL_POLICY_MAGIC, 8) != 0)
        return K26RL_POLICY_E_MAGIC;
    if (k26rl_get_u32_(bytes + POL_OFF_VERSION) !=
        K26RL_POLICY_FORMAT_VERSION)
        return K26RL_POLICY_E_VERSION;
    if (k26rl_get_u32_(bytes + POL_OFF_HEADER) != K26RL_POLICY_HEADER_BYTES)
        return K26RL_POLICY_E_VERSION;

    /* Before anything the file declares is used to walk it, the bytes
     * are checked against the identity they carry. An altered file is
     * then refused as an altered file, rather than as whatever its
     * corrupted arithmetic happens to say next. */
    pol_digest_(bytes, len, computed);
    if (memcmp(computed, bytes + POL_OFF_DIGEST, K26RL_SHA256_BYTES) != 0)
        return K26RL_POLICY_E_DIGEST;

    if (k26rl_get_u32_(bytes + POL_OFF_RESERVED) != 0)
        return K26RL_POLICY_E_RESERVED;

    p = calloc(1, sizeof *p);
    if (!p)
        return K26RL_POLICY_E_MEMORY;
    in = &p->info;
    in->format_version    = K26RL_POLICY_FORMAT_VERSION;
    in->flags             = k26rl_get_u32_(bytes + POL_OFF_FLAGS);
    in->agent_count       = k26rl_get_u32_(bytes + POL_OFF_AGENT_COUNT);
    in->agent_index       = k26rl_get_u32_(bytes + POL_OFF_AGENT_INDEX);
    in->obs_total         = k26rl_get_u32_(bytes + POL_OFF_OBS_TOTAL);
    in->act_total         = k26rl_get_u32_(bytes + POL_OFF_ACT_TOTAL);
    in->obs_offset        = k26rl_get_u32_(bytes + POL_OFF_OBS_OFFSET);
    in->obs_width         = k26rl_get_u32_(bytes + POL_OFF_OBS_WIDTH);
    in->obs_channel_count = k26rl_get_u32_(bytes + POL_OFF_OBS_CHANS);
    in->act_offset        = k26rl_get_u32_(bytes + POL_OFF_ACT_OFFSET);
    in->act_width         = k26rl_get_u32_(bytes + POL_OFF_ACT_WIDTH);
    in->layer_count       = k26rl_get_u32_(bytes + POL_OFF_LAYERS);
    in->provenance_bytes  = k26rl_get_u32_(bytes + POL_OFF_PROVENANCE);
    memcpy(in->digest, bytes + POL_OFF_DIGEST, K26RL_SHA256_BYTES);

#define POL_FAIL_(code) do { pol_free_(p); return (code); } while (0)

    if (in->flags & ~(uint32_t)K26RL_POLICY_FLAGS_KNOWN)
        POL_FAIL_(K26RL_POLICY_E_FLAGS);
    if (in->layer_count == 0)
        POL_FAIL_(K26RL_POLICY_E_LAYERS);
    if (in->layer_count > K26RL_POLICY_MAX_LAYERS ||
        in->provenance_bytes > K26RL_POLICY_MAX_PROVENANCE ||
        !pol_width_ok_(in->obs_width) || !pol_width_ok_(in->act_width) ||
        !pol_width_ok_(in->obs_channel_count) ||
        !pol_width_ok_(in->obs_total) || !pol_width_ok_(in->act_total))
        POL_FAIL_(K26RL_POLICY_E_SIZE);
    if (in->agent_count < 1u || in->agent_index >= in->agent_count)
        POL_FAIL_(K26RL_POLICY_E_SLICE);
    if ((uint64_t)in->obs_offset + in->obs_width > (uint64_t)in->obs_total ||
        (uint64_t)in->act_offset + in->act_width > (uint64_t)in->act_total)
        POL_FAIL_(K26RL_POLICY_E_SLICE);
    /* Distinct channels of one slice cannot outnumber the slice, so a
     * count above its width is refused before a byte of the list is
     * read. */
    if (in->obs_channel_count > in->obs_width)
        POL_FAIL_(K26RL_POLICY_E_CHANNELS);

    p->layers = calloc(in->layer_count, sizeof *p->layers);
    if (!p->layers)
        POL_FAIL_(K26RL_POLICY_E_MEMORY);

    /* First pass: walk the layer records for their shapes, bounding
     * every read against the byte count as it goes, and total up the
     * reals the body carries. Nothing is read out of the body until
     * the whole walk has landed inside the file. */
    at = (uint64_t)K26RL_POLICY_HEADER_BYTES + in->provenance_bytes;
    if (at > len)
        POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
    if ((uint64_t)in->obs_channel_count * 4u > len - at)
        POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
    p->channels = calloc(in->obs_channel_count, sizeof *p->channels);
    if (!p->channels)
        POL_FAIL_(K26RL_POLICY_E_MEMORY);
    st = pol_channels_(bytes, at, in, p->channels);
    if (st != K26RL_POLICY_OK)
        POL_FAIL_(st);
    at += (uint64_t)in->obs_channel_count * 4u;
    params_n = 0;
    scratch_w = in->obs_channel_count;
    for (i = 0; i < in->layer_count; i++) {
        PolLayer *l = &p->layers[i];
        uint64_t n;

        if (at + (uint64_t)POL_LAYER_HEADER > len)
            POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
        l->in_width   = k26rl_get_u32_(bytes + at);
        l->out_width  = k26rl_get_u32_(bytes + at + 4);
        l->activation = k26rl_get_u16_(bytes + at + 8);
        if (k26rl_get_u16_(bytes + at + 10) != 0)
            POL_FAIL_(K26RL_POLICY_E_RESERVED);
        if (!pol_width_ok_(l->in_width) || !pol_width_ok_(l->out_width))
            POL_FAIL_(K26RL_POLICY_E_SIZE);
        if (!pol_activation_known_(l->activation))
            POL_FAIL_(K26RL_POLICY_E_ACTIVATION);
        if (l->in_width != (i == 0 ? in->obs_channel_count
                                   : p->layers[i - 1].out_width))
            POL_FAIL_(K26RL_POLICY_E_LAYERS);
        n = (uint64_t)l->in_width * l->out_width + l->out_width;
        l->weight_at = params_n;
        l->bias_at   = params_n + (uint64_t)l->in_width * l->out_width;
        params_n += n;
        at += (uint64_t)POL_LAYER_HEADER;
        if (n > (len - at) / 8u)
            POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
        at += n * 8u;
        if (l->out_width > scratch_w)
            scratch_w = l->out_width;
    }
    if (p->layers[in->layer_count - 1].out_width != in->act_width)
        POL_FAIL_(K26RL_POLICY_E_LAYERS);

    reals_n = params_n;
    p->mean_at = p->scale_at = p->log_std_at = 0;
    p->lower_at = p->upper_at = 0;
    if (in->flags & K26RL_POLICY_FLAG_STANDARDISE) {
        uint64_t need = 2u * (uint64_t)in->obs_channel_count + 2u;

        if (need > (len - at) / 8u)
            POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
        p->mean_at  = reals_n;
        p->scale_at = reals_n + in->obs_channel_count;
        reals_n += 2u * (uint64_t)in->obs_channel_count;
        at += need * 8u;
    }
    if (in->flags & K26RL_POLICY_FLAG_LOG_STD) {
        if ((uint64_t)in->act_width > (len - at) / 8u)
            POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
        p->log_std_at = reals_n;
        reals_n += in->act_width;
        at += (uint64_t)in->act_width * 8u;
    }
    if (in->flags & K26RL_POLICY_FLAG_CLAMP) {
        if (2u * (uint64_t)in->act_width > (len - at) / 8u)
            POL_FAIL_(K26RL_POLICY_E_TRUNCATED);
        p->lower_at = reals_n;
        p->upper_at = reals_n + in->act_width;
        reals_n += 2u * (uint64_t)in->act_width;
        at += 2u * (uint64_t)in->act_width * 8u;
    }
    if (at != len)
        POL_FAIL_(at < len ? K26RL_POLICY_E_TRAILING
                           : K26RL_POLICY_E_TRUNCATED);

    p->scratch_a = reals_n;
    p->scratch_b = reals_n + scratch_w;
    reals_n += 2u * scratch_w;
    /* Where the whole-vector entry point gathers the channels it
     * reads. It is part of the load's allocation so that a decision
     * on a deadline allocates nothing. */
    p->gather_at = reals_n;
    reals_n += in->obs_channel_count;
    p->reals = calloc((size_t)reals_n, sizeof *p->reals);
    p->provenance = calloc((size_t)in->provenance_bytes + 1u, 1u);
    if (!p->reals || !p->provenance)
        POL_FAIL_(K26RL_POLICY_E_MEMORY);
    memcpy(p->provenance, bytes + K26RL_POLICY_HEADER_BYTES,
           in->provenance_bytes);

    /* Second pass: decode the reals, now that the walk has proved
     * every one of them lies inside the file. It restarts at the
     * first layer record, which is past the provenance text and the
     * channel list both. */
    at = (uint64_t)K26RL_POLICY_HEADER_BYTES + in->provenance_bytes +
         (uint64_t)in->obs_channel_count * 4u;
    for (i = 0; i < in->layer_count; i++) {
        PolLayer *l = &p->layers[i];
        uint64_t n = (uint64_t)l->in_width * l->out_width + l->out_width;
        uint64_t k;

        at += (uint64_t)POL_LAYER_HEADER;
        for (k = 0; k < n; k++)
            p->reals[l->weight_at + k] = k26rl_get_f64_(bytes + at + k * 8u);
        at += n * 8u;
    }
    if (in->flags & K26RL_POLICY_FLAG_STANDARDISE) {
        double epsilon;
        uint32_t j;

        for (j = 0; j < in->obs_channel_count; j++)
            p->reals[p->mean_at + j] = k26rl_get_f64_(bytes + at + j * 8u);
        epsilon = k26rl_get_f64_(bytes + at +
                                 (uint64_t)2u * in->obs_channel_count * 8u);
        p->clip = k26rl_get_f64_(bytes + at +
                                 ((uint64_t)2u * in->obs_channel_count + 1u) *
                                 8u);
        if (!(epsilon >= 0.0) || !isfinite(epsilon))
            POL_FAIL_(K26RL_POLICY_E_STATISTICS);
        if (isnan(p->clip) || p->clip <= 0.0)
            POL_FAIL_(K26RL_POLICY_E_STATISTICS);
        for (j = 0; j < in->obs_channel_count; j++) {
            double var = k26rl_get_f64_(bytes + at +
                                        ((uint64_t)in->obs_channel_count + j) *
                                        8u);
            double scale;

            if (!isfinite(p->reals[p->mean_at + j]) || !isfinite(var) ||
                var < 0.0)
                POL_FAIL_(K26RL_POLICY_E_STATISTICS);
            /* The declared rule divides by this square root; it is
             * stored, not inverted, so that the division the header
             * states is the division performed. */
            scale = sqrt(var + epsilon);
            if (!isfinite(scale) || scale <= 0.0)
                POL_FAIL_(K26RL_POLICY_E_STATISTICS);
            p->reals[p->scale_at + j] = scale;
        }
        at += ((uint64_t)2u * in->obs_channel_count + 2u) * 8u;
    }
    if (in->flags & K26RL_POLICY_FLAG_LOG_STD) {
        uint32_t j;

        for (j = 0; j < in->act_width; j++) {
            double v = k26rl_get_f64_(bytes + at + j * 8u);

            if (!isfinite(v))
                POL_FAIL_(K26RL_POLICY_E_STATISTICS);
            p->reals[p->log_std_at + j] = v;
        }
        at += (uint64_t)in->act_width * 8u;
    }
    if (in->flags & K26RL_POLICY_FLAG_CLAMP) {
        uint32_t j;

        for (j = 0; j < in->act_width; j++) {
            double lo = k26rl_get_f64_(bytes + at + j * 8u);
            double hi = k26rl_get_f64_(bytes + at +
                                       ((uint64_t)in->act_width + j) * 8u);

            if (isnan(lo) || isnan(hi) || !(lo <= hi))
                POL_FAIL_(K26RL_POLICY_E_STATISTICS);
            p->reals[p->lower_at + j] = lo;
            p->reals[p->upper_at + j] = hi;
        }
    }

#undef POL_FAIL_

    *out = p;
    return K26RL_POLICY_OK;
}

K26RlPolicyStatus k26rl_policy_open(const char *path, K26RlPolicy **out)
{
    FILE *f;
    long end;
    uint64_t len;
    uint8_t *bytes;
    K26RlPolicyStatus st;

    if (!path || !out)
        return K26RL_POLICY_E_NULL;
    f = fopen(path, "rb");
    if (!f)
        return K26RL_POLICY_E_IO;
    if (fseek(f, 0, SEEK_END) != 0 || (end = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return K26RL_POLICY_E_IO;
    }
    len = (uint64_t)end;
    if (len > K26RL_POLICY_MAX_BYTES) {
        fclose(f);
        return K26RL_POLICY_E_SIZE;
    }
    bytes = malloc((size_t)len + 1u);
    if (!bytes) {
        fclose(f);
        return K26RL_POLICY_E_MEMORY;
    }
    if (len != 0 && fread(bytes, 1, (size_t)len, f) != (size_t)len) {
        free(bytes);
        fclose(f);
        return K26RL_POLICY_E_IO;
    }
    fclose(f);
    st = k26rl_policy_parse(bytes, len, out);
    free(bytes);
    return st;
}

void k26rl_policy_close(K26RlPolicy *policy)
{
    pol_free_(policy);
}

K26RlPolicyStatus k26rl_policy_info(const K26RlPolicy *policy,
                                    K26RlPolicyInfo *out)
{
    if (!policy || !out)
        return K26RL_POLICY_E_NULL;
    *out = policy->info;
    return K26RL_POLICY_OK;
}

K26RlPolicyStatus k26rl_policy_layer(const K26RlPolicy *policy,
                                     uint32_t index, K26RlPolicyLayer *out)
{
    if (!policy || !out)
        return K26RL_POLICY_E_NULL;
    if (index >= policy->info.layer_count)
        return K26RL_POLICY_E_LAYERS;
    out->in_width   = policy->layers[index].in_width;
    out->out_width  = policy->layers[index].out_width;
    out->activation = policy->layers[index].activation;
    return K26RL_POLICY_OK;
}

const char *k26rl_policy_provenance(const K26RlPolicy *policy,
                                    uint32_t *out_len)
{
    if (!policy)
        return NULL;
    if (out_len)
        *out_len = policy->info.provenance_bytes;
    return policy->provenance;
}

const uint32_t *k26rl_policy_obs_channels(const K26RlPolicy *policy,
                                          uint32_t *out_len)
{
    if (!policy)
        return NULL;
    if (out_len)
        *out_len = policy->info.obs_channel_count;
    return policy->channels;
}

const double *k26rl_policy_log_std(const K26RlPolicy *policy)
{
    if (!policy || !(policy->info.flags & K26RL_POLICY_FLAG_LOG_STD))
        return NULL;
    return policy->reals + policy->log_std_at;
}

/* ---- the spec check ------------------------------------------------- */

/* What the check needs out of the artifact's blob. The slice fields
 * carry a seen flag because a one-agent artifact is permitted to
 * describe its vectors by their totals alone. */
typedef struct {
    uint32_t agent_count;
    uint32_t obs_total;
    uint32_t act_total;
    uint32_t obs_offset, obs_width;
    uint32_t act_offset, act_width;
    int      saw_agent_count, saw_obs_total, saw_act_total;
    int      saw_obs_slice, saw_act_slice;
} PolSpecView;

/* Per-channel action bounds live in their own repeated tag, so they
 * are looked up on a second pass over the blob rather than gathered
 * into a table whose size the blob has not yet declared. */
static int pol_spec_bounds_(const uint8_t *spec, uint32_t spec_len,
                            uint32_t channel, double *lo, double *hi)
{
    uint32_t off = 0;

    while (off + 6u <= spec_len) {
        uint16_t tag = k26rl_get_u16_(spec + off);
        uint32_t l   = k26rl_get_u32_(spec + off + 2);

        if ((uint64_t)off + 6u + l > (uint64_t)spec_len)
            return 0;
        if (tag == K26RL_TAG_ACT_BOUNDS && l == 20u &&
            k26rl_get_u32_(spec + off + 6) == channel) {
            *lo = k26rl_get_f64_(spec + off + 10);
            *hi = k26rl_get_f64_(spec + off + 18);
            return 1;
        }
        off += 6u + l;
    }
    return 0;
}

/* What the artifact says observation channel `channel` carries. A
 * blob that publishes no source tag for it was written before the tag
 * existed, and every channel of such an artifact is a measurement,
 * which is also what an artifact declaring no sensor publishes
 * explicitly. A source kind this version does not know is refused
 * rather than assumed to be one of the two it does: guessing there
 * would be guessing which channels a policy may read. */
static K26RlPolicyStatus pol_spec_source_(const uint8_t *spec,
                                          uint32_t spec_len,
                                          uint32_t channel, uint16_t *out)
{
    uint32_t off = 0;

    *out = K26RL_OBS_SOURCE_MEASURED;
    while (off + 6u <= spec_len) {
        uint16_t tag = k26rl_get_u16_(spec + off);
        uint32_t l   = k26rl_get_u32_(spec + off + 2);

        if ((uint64_t)off + 6u + l > (uint64_t)spec_len)
            return K26RL_POLICY_E_SPEC;
        if (tag == K26RL_TAG_OBS_CHANNEL_SOURCE && l == 10u &&
            k26rl_get_u32_(spec + off + 6) == channel) {
            uint16_t source = k26rl_get_u16_(spec + off + 10);

            if (source != K26RL_OBS_SOURCE_MEASURED &&
                source != K26RL_OBS_SOURCE_TRUTH)
                return K26RL_POLICY_E_SPEC;
            *out = source;
            return K26RL_POLICY_OK;
        }
        off += 6u + l;
    }
    return K26RL_POLICY_OK;
}

/* The channels the policy reads against the channels the artifact
 * publishes to it: the measured ones of the agent's slice, ascending.
 * A count that disagrees and a position that disagrees are separate
 * refusals of the same field, because a policy reading five of six
 * channels and a policy reading the right five in the wrong order are
 * different defects and a caller cannot act on a code that says only
 * that something about the channels is wrong. */
static K26RlPolicyStatus pol_check_channels_(const K26RlPolicy *policy,
                                             const uint8_t *spec,
                                             uint32_t spec_len,
                                             uint32_t obs_offset,
                                             uint32_t obs_width,
                                             char *detail,
                                             uint32_t detail_capacity)
{
    const uint32_t declared = policy->info.obs_channel_count;
    uint32_t measured = 0, bad_at = 0, bad_channel = 0, c;
    int saw_bad = 0;

    for (c = obs_offset; c < obs_offset + obs_width; c++) {
        uint16_t source;
        K26RlPolicyStatus st = pol_spec_source_(spec, spec_len, c, &source);

        if (st != K26RL_POLICY_OK)
            return st;
        if (source != K26RL_OBS_SOURCE_MEASURED)
            continue;
        if (!saw_bad && measured < declared &&
            policy->channels[measured] != c) {
            saw_bad = 1;
            bad_at = measured;
            bad_channel = c;
        }
        measured++;
    }
    if (measured != declared) {
        if (detail && detail_capacity > 0)
            snprintf(detail, (size_t)detail_capacity,
                     "policy reads %lu observation channels, artifact "
                     "publishes %lu measured channels for this agent",
                     (unsigned long)declared, (unsigned long)measured);
        return K26RL_POLICY_E_OBS_CHANNELS;
    }
    if (saw_bad) {
        if (detail && detail_capacity > 0)
            snprintf(detail, (size_t)detail_capacity,
                     "policy reads observation channel %lu at position %lu, "
                     "artifact publishes channel %lu there",
                     (unsigned long)policy->channels[bad_at],
                     (unsigned long)bad_at, (unsigned long)bad_channel);
        return K26RL_POLICY_E_OBS_CHANNELS;
    }
    return K26RL_POLICY_OK;
}

static K26RlPolicyStatus pol_spec_view_(const uint8_t *spec, uint32_t spec_len,
                                        uint32_t agent, PolSpecView *v)
{
    uint32_t off = 0;

    memset(v, 0, sizeof *v);
    while (off + 6u <= spec_len) {
        uint16_t tag = k26rl_get_u16_(spec + off);
        uint32_t l   = k26rl_get_u32_(spec + off + 2);
        const uint8_t *val = spec + off + 6;

        if ((uint64_t)off + 6u + l > (uint64_t)spec_len)
            return K26RL_POLICY_E_SPEC;
        switch (tag) {
        case K26RL_TAG_AGENT_COUNT:
            if (l != 4u)
                return K26RL_POLICY_E_SPEC;
            v->agent_count = k26rl_get_u32_(val);
            v->saw_agent_count = 1;
            break;
        case K26RL_TAG_OBS_TOTAL:
            if (l != 4u)
                return K26RL_POLICY_E_SPEC;
            v->obs_total = k26rl_get_u32_(val);
            v->saw_obs_total = 1;
            break;
        case K26RL_TAG_ACT_TOTAL:
            if (l != 4u)
                return K26RL_POLICY_E_SPEC;
            v->act_total = k26rl_get_u32_(val);
            v->saw_act_total = 1;
            break;
        case K26RL_TAG_AGENT_OBS_SLICE:
            if (l != 12u)
                return K26RL_POLICY_E_SPEC;
            if (k26rl_get_u32_(val) == agent) {
                v->obs_offset = k26rl_get_u32_(val + 4);
                v->obs_width  = k26rl_get_u32_(val + 8);
                v->saw_obs_slice = 1;
            }
            break;
        case K26RL_TAG_AGENT_ACT_SLICE:
            if (l != 12u)
                return K26RL_POLICY_E_SPEC;
            if (k26rl_get_u32_(val) == agent) {
                v->act_offset = k26rl_get_u32_(val + 4);
                v->act_width  = k26rl_get_u32_(val + 8);
                v->saw_act_slice = 1;
            }
            break;
        default:
            break;      /* unknown tags skipped by length */
        }
        off += 6u + l;
    }
    if (off != spec_len)
        return K26RL_POLICY_E_SPEC;
    if (!v->saw_agent_count || !v->saw_obs_total || !v->saw_act_total)
        return K26RL_POLICY_E_SPEC;
    return K26RL_POLICY_OK;
}

static void pol_detail_(char *detail, uint32_t cap, const char *what,
                        unsigned long mine, unsigned long theirs)
{
    if (!detail || cap == 0)
        return;
    snprintf(detail, (size_t)cap,
             "policy declares %s %lu, artifact declares %lu",
             what, mine, theirs);
}

K26RlPolicyStatus k26rl_policy_check_spec(const K26RlPolicy *policy,
                                          const uint8_t *spec,
                                          uint32_t spec_len,
                                          char *detail,
                                          uint32_t detail_capacity)
{
    const K26RlPolicyInfo *in;
    PolSpecView v;
    K26RlPolicyStatus st;
    uint32_t j;

    if (!policy || !spec)
        return K26RL_POLICY_E_NULL;
    if (detail && detail_capacity > 0)
        detail[0] = '\0';
    in = &policy->info;
    st = pol_spec_view_(spec, spec_len, in->agent_index, &v);
    if (st != K26RL_POLICY_OK)
        return st;

    /* Each field is its own comparison and its own refusal. A check
     * that agreed on totals while disagreeing on a slice would be a
     * check that cannot fail on the defect it is here to catch. */
    if (in->agent_count != v.agent_count) {
        pol_detail_(detail, detail_capacity, "agent count",
                    in->agent_count, v.agent_count);
        return K26RL_POLICY_E_AGENT_COUNT;
    }
    if (in->obs_total != v.obs_total) {
        pol_detail_(detail, detail_capacity, "observation total",
                    in->obs_total, v.obs_total);
        return K26RL_POLICY_E_OBS_TOTAL;
    }
    if (in->act_total != v.act_total) {
        pol_detail_(detail, detail_capacity, "action total",
                    in->act_total, v.act_total);
        return K26RL_POLICY_E_ACT_TOTAL;
    }
    /* A one-agent artifact may describe its vectors by their totals
     * alone, which is the same statement as a slice covering the
     * whole of each. */
    if (!v.saw_obs_slice) {
        if (v.agent_count != 1u)
            return K26RL_POLICY_E_SPEC;
        v.obs_offset = 0;
        v.obs_width = v.obs_total;
    }
    if (!v.saw_act_slice) {
        if (v.agent_count != 1u)
            return K26RL_POLICY_E_SPEC;
        v.act_offset = 0;
        v.act_width = v.act_total;
    }
    if (in->obs_offset != v.obs_offset) {
        pol_detail_(detail, detail_capacity, "observation slice offset",
                    in->obs_offset, v.obs_offset);
        return K26RL_POLICY_E_OBS_OFFSET;
    }
    if (in->obs_width != v.obs_width) {
        pol_detail_(detail, detail_capacity, "observation slice width",
                    in->obs_width, v.obs_width);
        return K26RL_POLICY_E_OBS_WIDTH;
    }
    /* The slice agreeing does not make the input agree: the policy
     * reads the measured channels of that slice, and which of them
     * those are is the artifact's to say. */
    st = pol_check_channels_(policy, spec, spec_len, v.obs_offset,
                             v.obs_width, detail, detail_capacity);
    if (st != K26RL_POLICY_OK)
        return st;
    if (in->act_offset != v.act_offset) {
        pol_detail_(detail, detail_capacity, "action slice offset",
                    in->act_offset, v.act_offset);
        return K26RL_POLICY_E_ACT_OFFSET;
    }
    if (in->act_width != v.act_width) {
        pol_detail_(detail, detail_capacity, "action slice width",
                    in->act_width, v.act_width);
        return K26RL_POLICY_E_ACT_WIDTH;
    }
    if (!(in->flags & K26RL_POLICY_FLAG_CLAMP))
        return K26RL_POLICY_OK;
    for (j = 0; j < in->act_width; j++) {
        double lo, hi;

        if (!pol_spec_bounds_(spec, spec_len, in->act_offset + j, &lo, &hi))
            continue;   /* an unbounded channel constrains nothing */
        if (policy->reals[policy->lower_at + j] < lo ||
            policy->reals[policy->upper_at + j] > hi) {
            if (detail && detail_capacity > 0)
                snprintf(detail, (size_t)detail_capacity,
                         "policy clamps action channel %lu to [%.17g, %.17g], "
                         "artifact declares [%.17g, %.17g]",
                         (unsigned long)(in->act_offset + j),
                         policy->reals[policy->lower_at + j],
                         policy->reals[policy->upper_at + j], lo, hi);
            return K26RL_POLICY_E_ACT_BOUNDS;
        }
    }
    return K26RL_POLICY_OK;
}

/* ---- evaluation ----------------------------------------------------- */

static double pol_activate_(uint16_t activation, double v)
{
    switch (activation) {
    case K26RL_POLICY_ACT_TANH:
        return tanh(v);
    case K26RL_POLICY_ACT_RELU:
        /* A NaN propagates rather than becoming zero: an input that
         * has already gone wrong should be visible in the action, not
         * hidden by the rectifier. */
        return (v > 0.0 || v != v) ? v : 0.0;
    case K26RL_POLICY_ACT_LOGISTIC:
        return 1.0 / (1.0 + exp(-v));
    default:
        return v;
    }
}

K26RlPolicyStatus k26rl_policy_act(const K26RlPolicy *policy,
                                   const double *obs, double *out)
{
    const K26RlPolicyInfo *in;
    double *x, *y;
    uint32_t i, j;

    if (!policy || !obs || !out)
        return K26RL_POLICY_E_NULL;
    in = &policy->info;
    x = policy->reals + policy->scratch_a;
    y = policy->reals + policy->scratch_b;

    if (in->flags & K26RL_POLICY_FLAG_STANDARDISE) {
        for (j = 0; j < in->obs_channel_count; j++) {
            double v = (obs[j] - policy->reals[policy->mean_at + j]) /
                       policy->reals[policy->scale_at + j];

            if (v < -policy->clip)
                v = -policy->clip;
            else if (v > policy->clip)
                v = policy->clip;
            x[j] = v;
        }
    } else {
        for (j = 0; j < in->obs_channel_count; j++)
            x[j] = obs[j];
    }

    for (i = 0; i < in->layer_count; i++) {
        const PolLayer *l = &policy->layers[i];
        const double *w = policy->reals + l->weight_at;
        const double *b = policy->reals + l->bias_at;
        double *swap;
        uint32_t o, k;

        for (o = 0; o < l->out_width; o++) {
            double acc = b[o];
            const double *row = w + (size_t)o * l->in_width;

            for (k = 0; k < l->in_width; k++)
                acc += row[k] * x[k];
            y[o] = pol_activate_(l->activation, acc);
        }
        swap = x;
        x = y;
        y = swap;
    }

    if (in->flags & K26RL_POLICY_FLAG_CLAMP) {
        for (j = 0; j < in->act_width; j++) {
            double v = x[j];

            if (v < policy->reals[policy->lower_at + j])
                v = policy->reals[policy->lower_at + j];
            else if (v > policy->reals[policy->upper_at + j])
                v = policy->reals[policy->upper_at + j];
            out[j] = v;
        }
    } else {
        for (j = 0; j < in->act_width; j++)
            out[j] = x[j];
    }
    return K26RL_POLICY_OK;
}

K26RlPolicyStatus k26rl_policy_act_env(const K26RlPolicy *policy,
                                       const double *env_obs, double *env_act)
{
    double *gathered;
    uint32_t j;

    if (!policy || !env_obs || !env_act)
        return K26RL_POLICY_E_NULL;
    /* The channels the policy reads, in the order it reads them. The
     * load proved every index lies inside the agent's slice and so
     * inside the observation total, and the buffer is the policy's
     * own, so the gather neither allocates nor reads outside the
     * vector the caller passed. */
    gathered = policy->reals + policy->gather_at;
    for (j = 0; j < policy->info.obs_channel_count; j++)
        gathered[j] = env_obs[policy->channels[j]];
    return k26rl_policy_act(policy, gathered,
                            env_act + policy->info.act_offset);
}
