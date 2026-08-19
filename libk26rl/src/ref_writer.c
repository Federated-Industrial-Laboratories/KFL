/* ref_writer.c - encoding a `.k26ref` plan.
 *
 * The layout and the identity rule are declared in k26rl_ref.h; this
 * file writes exactly what ref_reader.c reads, through the same
 * little-endian field assembly every other format in this library
 * uses.
 *
 * The selection and ordering rule lives here rather than in the
 * producer, and that is the point of the file. A planner's output is
 * a fixed number of slots because a policy with a variable-width
 * output is not a policy that trains, and a fixed-width output can
 * neither guarantee that its slots are in time order nor that it
 * filled them all. Putting the rule here means every plan such a
 * producer can emit encodes, and what it encodes to is a file the
 * reader accepts.
 */
#include "k26rl_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k26rl_internal.h"

/* Header field byte offsets, the reader's. */
#define REF_OFF_MAGIC      0
#define REF_OFF_VERSION    8
#define REF_OFF_HEADER     12
#define REF_OFF_DIGEST     16
#define REF_OFF_FLAGS      48
#define REF_OFF_FRAME_KIND 52
#define REF_OFF_FRAME_LEN  56
#define REF_OFF_KNOTS      60
#define REF_OFF_PROVENANCE 64
#define REF_OFF_RESERVED   68
#define REF_OFF_EPOCH      72

static void ref_digest_(const uint8_t *b, uint64_t len,
                        uint8_t out[K26RL_SHA256_BYTES])
{
    static const uint8_t zeros[K26RL_SHA256_BYTES] = { 0 };
    K26RlSha256 s;

    k26rl_sha256_init(&s);
    k26rl_sha256_update(&s, b, (uint64_t)REF_OFF_DIGEST);
    k26rl_sha256_update(&s, zeros, (uint64_t)K26RL_SHA256_BYTES);
    k26rl_sha256_update(&s, b + REF_OFF_DIGEST + K26RL_SHA256_BYTES,
                        len - (uint64_t)(REF_OFF_DIGEST +
                                         K26RL_SHA256_BYTES));
    k26rl_sha256_final(&s, out);
}

static int ref_knot_finite_(const K26RlRefKnot *k)
{
    int j;

    if (!isfinite(k->t) || !isfinite(k->tolerance))
        return 0;
    for (j = 0; j < 3; j++) {
        if (!isfinite(k->r[j]) || !isfinite(k->v[j]))
            return 0;
    }
    return 1;
}

K26RlRefStatus k26rl_ref_encode_into(const K26RlRefPlan *plan,
                                     K26RlRefKnot *scratch,
                                     uint8_t *out, uint64_t cap,
                                     uint64_t *out_len)
{
    K26RlRefKnot *kept = scratch;
    uint8_t *b = out;
    uint64_t len, at;
    uint32_t frame_len, prov_len, n, i, j;

    if (!plan || !out || !out_len || !plan->frame_name)
        return K26RL_REF_E_NULL;
    if (plan->knot_count != 0 && (!plan->knots || !scratch))
        return K26RL_REF_E_NULL;
    if (plan->frame_kind != K26RL_REF_FRAME_LVLH &&
        plan->frame_kind != K26RL_REF_FRAME_INERTIAL)
        return K26RL_REF_E_FRAME;
    if (!isfinite(plan->epoch))
        return K26RL_REF_E_VALUE;

    frame_len = (uint32_t)strlen(plan->frame_name);
    prov_len  = plan->provenance ? (uint32_t)strlen(plan->provenance) : 0u;
    if (frame_len == 0)
        return K26RL_REF_E_FRAME;
    if (frame_len > K26RL_REF_MAX_FRAME ||
        prov_len > K26RL_REF_MAX_PROVENANCE ||
        plan->knot_count > K26RL_REF_MAX_KNOTS)
        return K26RL_REF_E_SIZE;

    /* A slot whose tolerance is not above zero is absent. Every other
     * value must be a usable number: an infinity or a not-a-number
     * anywhere in a present slot is a producer that has already gone
     * wrong, and encoding it would put that in a file the reader must
     * then refuse. */
    n = 0;
    for (i = 0; i < plan->knot_count; i++) {
        const K26RlRefKnot *k = &plan->knots[i];

        if (isnan(k->tolerance) || k->tolerance <= 0.0)
            continue;
        if (!ref_knot_finite_(k))
            return K26RL_REF_E_VALUE;
        kept[n++] = *k;
    }

    /* Insertion sort by time, ties broken by the order the producer
     * gave, so one set of slots always encodes to one file. */
    for (i = 1; i < n; i++) {
        K26RlRefKnot pivot = kept[i];

        j = i;
        while (j > 0 && kept[j - 1].t > pivot.t) {
            kept[j] = kept[j - 1];
            j--;
        }
        kept[j] = pivot;
    }
    /* A slot that does not strictly advance the clock could never be
     * current, so it is dropped rather than written into a file the
     * reader would refuse for its order. */
    j = 0;
    for (i = 0; i < n; i++) {
        if (j > 0 && !(kept[i].t > kept[j - 1].t))
            continue;
        kept[j++] = kept[i];
    }
    n = j;

    len = (uint64_t)K26RL_REF_HEADER_BYTES + frame_len + prov_len +
          (uint64_t)n * K26RL_REF_KNOT_BYTES;
    if (len > K26RL_REF_MAX_BYTES || len > cap)
        return K26RL_REF_E_SIZE;
    memset(b, 0, (size_t)len);

    memcpy(b + REF_OFF_MAGIC, K26RL_REF_MAGIC, 8);
    k26rl_put_u32_(b + REF_OFF_VERSION, K26RL_REF_FORMAT_VERSION);
    k26rl_put_u32_(b + REF_OFF_HEADER, K26RL_REF_HEADER_BYTES);
    k26rl_put_u32_(b + REF_OFF_FLAGS, 0);
    k26rl_put_u32_(b + REF_OFF_FRAME_KIND, plan->frame_kind);
    k26rl_put_u32_(b + REF_OFF_FRAME_LEN, frame_len);
    k26rl_put_u32_(b + REF_OFF_KNOTS, n);
    k26rl_put_u32_(b + REF_OFF_PROVENANCE, prov_len);
    k26rl_put_u32_(b + REF_OFF_RESERVED, 0);
    k26rl_put_f64_(b + REF_OFF_EPOCH, plan->epoch);

    at = (uint64_t)K26RL_REF_HEADER_BYTES;
    memcpy(b + at, plan->frame_name, frame_len);
    at += frame_len;
    if (prov_len)
        memcpy(b + at, plan->provenance, prov_len);
    at += prov_len;
    for (i = 0; i < n; i++) {
        uint8_t *k = b + at + (uint64_t)i * K26RL_REF_KNOT_BYTES;

        k26rl_put_f64_(k, kept[i].t);
        for (j = 0; j < 3; j++)
            k26rl_put_f64_(k + (uint64_t)(1u + j) * 8u, kept[i].r[j]);
        for (j = 0; j < 3; j++)
            k26rl_put_f64_(k + (uint64_t)(4u + j) * 8u, kept[i].v[j]);
        k26rl_put_f64_(k + 7u * 8u, kept[i].tolerance);
    }

    /* The digest is taken last, over the finished bytes with its own
     * field still zero, and patched in. */
    {
        uint8_t d[K26RL_SHA256_BYTES];

        ref_digest_(b, len, d);
        memcpy(b + REF_OFF_DIGEST, d, K26RL_SHA256_BYTES);
    }
    *out_len = len;
    return K26RL_REF_OK;
}

K26RlRefStatus k26rl_ref_encode(const K26RlRefPlan *plan, uint8_t **out,
                                uint64_t *out_len)
{
    K26RlRefKnot *kept;
    uint8_t *b;
    uint64_t len, cap;
    uint32_t frame_len, prov_len;
    K26RlRefStatus st;

    if (!plan || !out || !out_len || !plan->frame_name)
        return K26RL_REF_E_NULL;
    if (plan->knot_count != 0 && !plan->knots)
        return K26RL_REF_E_NULL;

    frame_len = (uint32_t)strlen(plan->frame_name);
    prov_len  = plan->provenance ? (uint32_t)strlen(plan->provenance) : 0u;
    if (frame_len > K26RL_REF_MAX_FRAME ||
        prov_len > K26RL_REF_MAX_PROVENANCE ||
        plan->knot_count > K26RL_REF_MAX_KNOTS)
        return K26RL_REF_E_SIZE;

    /* The worst case, which is every slot present. */
    cap = (uint64_t)K26RL_REF_HEADER_BYTES + frame_len + prov_len +
          (uint64_t)plan->knot_count * K26RL_REF_KNOT_BYTES;
    kept = plan->knot_count
         ? calloc((size_t)plan->knot_count, sizeof *kept) : NULL;
    if (plan->knot_count && !kept)
        return K26RL_REF_E_MEMORY;
    b = calloc((size_t)cap, 1u);
    if (!b) {
        free(kept);
        return K26RL_REF_E_MEMORY;
    }
    st = k26rl_ref_encode_into(plan, kept, b, cap, &len);
    free(kept);
    if (st != K26RL_REF_OK) {
        free(b);
        return st;
    }
    *out = b;
    *out_len = len;
    return K26RL_REF_OK;
}

K26RlRefStatus k26rl_ref_write(const char *path, const K26RlRefPlan *plan)
{
    uint8_t *bytes = NULL;
    uint64_t len = 0;
    K26RlRefStatus st;
    FILE *f;

    if (!path)
        return K26RL_REF_E_NULL;
    st = k26rl_ref_encode(plan, &bytes, &len);
    if (st != K26RL_REF_OK)
        return st;
    f = fopen(path, "wb");
    if (!f) {
        free(bytes);
        return K26RL_REF_E_IO;
    }
    if (len != 0 && fwrite(bytes, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(bytes);
        return K26RL_REF_E_IO;
    }
    free(bytes);
    if (fclose(f) != 0)
        return K26RL_REF_E_IO;
    return K26RL_REF_OK;
}
