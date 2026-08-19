/* ref_reader.c - loading a `.k26ref` plan.
 *
 * The layout and the identity rule are declared in k26rl_ref.h; this
 * file implements exactly those, on the discipline policy_reader.c
 * already follows: nothing is indexed before it is verified. The
 * file's bytes are read whole, the digest is checked over them, and
 * every declared size is bounds-checked in 64-bit arithmetic against
 * the byte count before any of it is used to walk.
 */
#include "k26rl_ref.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k26rl_internal.h"

/* Header field byte offsets. */
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

struct K26RlRef {
    K26RlRefInfo  info;
    char         *frame;        /* terminated copy; never null once loaded */
    char         *provenance;   /* terminated copy; never null once loaded */
    K26RlRefKnot *knots;        /* the present knots, in time order */
    uint8_t      *bytes;        /* the file's own bytes, unchanged */
    uint64_t      len;
};

const char *k26rl_ref_status_str(K26RlRefStatus status)
{
    switch (status) {
    case K26RL_REF_OK:          return "ok";
    case K26RL_REF_E_NULL:      return "null pointer argument";
    case K26RL_REF_E_IO:        return "the reference file could not be "
                                       "read or written";
    case K26RL_REF_E_MEMORY:    return "allocation failed";
    case K26RL_REF_E_MAGIC:     return "not a reference file";
    case K26RL_REF_E_VERSION:   return "reference format version not served";
    case K26RL_REF_E_TRUNCATED: return "reference file is shorter than it "
                                       "declares";
    case K26RL_REF_E_TRAILING:  return "reference file carries bytes past "
                                       "its declared end";
    case K26RL_REF_E_DIGEST:    return "reference digest disagrees with the "
                                       "file's own bytes";
    case K26RL_REF_E_SIZE:      return "a declared size exceeds this "
                                       "format's ceiling";
    case K26RL_REF_E_RESERVED:  return "a reserved field is not zero";
    case K26RL_REF_E_FLAGS:     return "an unknown flag bit is set";
    case K26RL_REF_E_FRAME:     return "the frame kind is outside the "
                                       "format's closed list, or the frame "
                                       "names no body";
    case K26RL_REF_E_KNOTS:     return "the plan carries no knot";
    case K26RL_REF_E_ORDER:     return "the plan's knot times do not "
                                       "strictly increase";
    case K26RL_REF_E_VALUE:     return "a knot carries a value that is not "
                                       "a usable finite number";
    case K26RL_REF_E_RANGE:     return "an index is outside what the plan "
                                       "holds";
    }
    return "unknown reference status";
}

/* The digest as the format defines it: the whole file with its own 32
 * digest bytes read as zero. The frame, the epoch, the provenance and
 * every knot sit inside that region, so relabelling a plan changes
 * its identity. */
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

static void ref_free_(K26RlRef *r)
{
    if (!r)
        return;
    free(r->frame);
    free(r->provenance);
    free(r->knots);
    free(r->bytes);
    free(r);
}

K26RlRefStatus k26rl_ref_parse(const uint8_t *bytes, uint64_t len,
                               K26RlRef **out)
{
    K26RlRef *r;
    K26RlRefInfo *in;
    uint8_t computed[K26RL_SHA256_BYTES];
    uint64_t at, need;
    uint32_t i, kept;

    if (!bytes || !out)
        return K26RL_REF_E_NULL;
    if (len > K26RL_REF_MAX_BYTES)
        return K26RL_REF_E_SIZE;
    if (len < (uint64_t)K26RL_REF_HEADER_BYTES)
        return K26RL_REF_E_TRUNCATED;
    if (memcmp(bytes + REF_OFF_MAGIC, K26RL_REF_MAGIC, 8) != 0)
        return K26RL_REF_E_MAGIC;
    if (k26rl_get_u32_(bytes + REF_OFF_VERSION) != K26RL_REF_FORMAT_VERSION)
        return K26RL_REF_E_VERSION;
    if (k26rl_get_u32_(bytes + REF_OFF_HEADER) != K26RL_REF_HEADER_BYTES)
        return K26RL_REF_E_VERSION;

    /* Before anything the file declares is used to walk it, the bytes
     * are checked against the identity they carry. An altered file is
     * then refused as an altered file, rather than as whatever its
     * corrupted arithmetic happens to say next. */
    ref_digest_(bytes, len, computed);
    if (memcmp(computed, bytes + REF_OFF_DIGEST, K26RL_SHA256_BYTES) != 0)
        return K26RL_REF_E_DIGEST;

    if (k26rl_get_u32_(bytes + REF_OFF_RESERVED) != 0)
        return K26RL_REF_E_RESERVED;

    r = calloc(1, sizeof *r);
    if (!r)
        return K26RL_REF_E_MEMORY;
    in = &r->info;
    in->format_version   = K26RL_REF_FORMAT_VERSION;
    in->flags            = k26rl_get_u32_(bytes + REF_OFF_FLAGS);
    in->frame_kind       = k26rl_get_u32_(bytes + REF_OFF_FRAME_KIND);
    in->frame_name_bytes = k26rl_get_u32_(bytes + REF_OFF_FRAME_LEN);
    in->knot_count       = k26rl_get_u32_(bytes + REF_OFF_KNOTS);
    in->provenance_bytes = k26rl_get_u32_(bytes + REF_OFF_PROVENANCE);
    in->epoch            = k26rl_get_f64_(bytes + REF_OFF_EPOCH);
    memcpy(in->digest, bytes + REF_OFF_DIGEST, K26RL_SHA256_BYTES);

#define REF_FAIL_(code) do { ref_free_(r); return (code); } while (0)

    if (in->flags & ~(uint32_t)K26RL_REF_FLAGS_KNOWN)
        REF_FAIL_(K26RL_REF_E_FLAGS);
    if (in->frame_kind != K26RL_REF_FRAME_LVLH &&
        in->frame_kind != K26RL_REF_FRAME_INERTIAL)
        REF_FAIL_(K26RL_REF_E_FRAME);
    if (in->frame_name_bytes == 0)
        REF_FAIL_(K26RL_REF_E_FRAME);
    if (in->frame_name_bytes > K26RL_REF_MAX_FRAME ||
        in->provenance_bytes > K26RL_REF_MAX_PROVENANCE ||
        in->knot_count > K26RL_REF_MAX_KNOTS)
        REF_FAIL_(K26RL_REF_E_SIZE);
    if (in->knot_count == 0)
        REF_FAIL_(K26RL_REF_E_KNOTS);
    if (!isfinite(in->epoch))
        REF_FAIL_(K26RL_REF_E_VALUE);

    /* The length the declared sections imply, computed whole in
     * 64-bit arithmetic before any of it is used to index. */
    need = (uint64_t)K26RL_REF_HEADER_BYTES + in->frame_name_bytes +
           in->provenance_bytes +
           (uint64_t)in->knot_count * K26RL_REF_KNOT_BYTES;
    if (len != need)
        REF_FAIL_(len < need ? K26RL_REF_E_TRUNCATED
                             : K26RL_REF_E_TRAILING);

    r->frame      = calloc((size_t)in->frame_name_bytes + 1u, 1u);
    r->provenance = calloc((size_t)in->provenance_bytes + 1u, 1u);
    r->knots      = calloc((size_t)in->knot_count, sizeof *r->knots);
    r->bytes      = malloc((size_t)len);
    if (!r->frame || !r->provenance || !r->knots || !r->bytes)
        REF_FAIL_(K26RL_REF_E_MEMORY);
    memcpy(r->bytes, bytes, (size_t)len);
    r->len = len;

    at = (uint64_t)K26RL_REF_HEADER_BYTES;
    memcpy(r->frame, bytes + at, in->frame_name_bytes);
    at += in->frame_name_bytes;
    memcpy(r->provenance, bytes + at, in->provenance_bytes);
    at += in->provenance_bytes;

    /* A knot whose tolerance is not above zero is absent, which is how
     * a producer with a fixed slot count emits fewer knots than it has
     * slots. The present ones are kept in file order and their times
     * must strictly increase; a negative tolerance is a value no
     * producer can mean and is refused rather than read as absence. */
    kept = 0;
    for (i = 0; i < in->knot_count; i++) {
        const uint8_t *k = bytes + at + (uint64_t)i * K26RL_REF_KNOT_BYTES;
        K26RlRefKnot kn;
        int j;

        kn.t = k26rl_get_f64_(k);
        for (j = 0; j < 3; j++)
            kn.r[j] = k26rl_get_f64_(k + (uint64_t)(1 + j) * 8u);
        for (j = 0; j < 3; j++)
            kn.v[j] = k26rl_get_f64_(k + (uint64_t)(4 + j) * 8u);
        kn.tolerance = k26rl_get_f64_(k + 7u * 8u);

        if (isnan(kn.tolerance) || kn.tolerance < 0.0)
            REF_FAIL_(K26RL_REF_E_VALUE);
        if (kn.tolerance == 0.0)
            continue;                       /* absent */
        if (!isfinite(kn.t) || !isfinite(kn.tolerance))
            REF_FAIL_(K26RL_REF_E_VALUE);
        for (j = 0; j < 3; j++) {
            if (!isfinite(kn.r[j]) || !isfinite(kn.v[j]))
                REF_FAIL_(K26RL_REF_E_VALUE);
        }
        if (kept > 0 && !(kn.t > r->knots[kept - 1].t))
            REF_FAIL_(K26RL_REF_E_ORDER);
        r->knots[kept++] = kn;
    }
    in->present_count = kept;

#undef REF_FAIL_

    *out = r;
    return K26RL_REF_OK;
}

K26RlRefStatus k26rl_ref_open(const char *path, K26RlRef **out)
{
    FILE *f;
    long end;
    uint64_t len;
    uint8_t *bytes;
    K26RlRefStatus st;

    if (!path || !out)
        return K26RL_REF_E_NULL;
    f = fopen(path, "rb");
    if (!f)
        return K26RL_REF_E_IO;
    if (fseek(f, 0, SEEK_END) != 0 || (end = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return K26RL_REF_E_IO;
    }
    len = (uint64_t)end;
    if (len > K26RL_REF_MAX_BYTES) {
        fclose(f);
        return K26RL_REF_E_SIZE;
    }
    bytes = malloc((size_t)len + 1u);
    if (!bytes) {
        fclose(f);
        return K26RL_REF_E_MEMORY;
    }
    if (len != 0 && fread(bytes, 1, (size_t)len, f) != (size_t)len) {
        free(bytes);
        fclose(f);
        return K26RL_REF_E_IO;
    }
    fclose(f);
    st = k26rl_ref_parse(bytes, len, out);
    free(bytes);
    return st;
}

void k26rl_ref_close(K26RlRef *ref)
{
    ref_free_(ref);
}

K26RlRefStatus k26rl_ref_info(const K26RlRef *ref, K26RlRefInfo *out)
{
    if (!ref || !out)
        return K26RL_REF_E_NULL;
    *out = ref->info;
    return K26RL_REF_OK;
}

const char *k26rl_ref_frame_name(const K26RlRef *ref, uint32_t *out_len)
{
    if (!ref)
        return NULL;
    if (out_len)
        *out_len = ref->info.frame_name_bytes;
    return ref->frame;
}

const char *k26rl_ref_provenance(const K26RlRef *ref, uint32_t *out_len)
{
    if (!ref)
        return NULL;
    if (out_len)
        *out_len = ref->info.provenance_bytes;
    return ref->provenance;
}

K26RlRefStatus k26rl_ref_knot(const K26RlRef *ref, uint32_t index,
                              K26RlRefKnot *out)
{
    if (!ref || !out)
        return K26RL_REF_E_NULL;
    if (index >= ref->info.present_count)
        return K26RL_REF_E_RANGE;
    *out = ref->knots[index];
    return K26RL_REF_OK;
}

K26RlRefStatus k26rl_ref_current(const K26RlRef *ref, double t, uint32_t *out)
{
    uint32_t i;

    if (!ref || !out)
        return K26RL_REF_E_NULL;
    if (ref->info.present_count == 0)
        return K26RL_REF_E_KNOTS;
    /* The earliest knot whose time has not passed. A knot's time
     * passing without the craft inside its tolerance is not an error
     * and ends nothing; the plan advances and the craft is late. When
     * every knot's time has passed the last one stays current, which
     * is the plan's final state still being the one asked for. */
    for (i = 0; i < ref->info.present_count; i++) {
        if (ref->info.epoch + ref->knots[i].t >= t) {
            *out = i;
            return K26RL_REF_OK;
        }
    }
    *out = ref->info.present_count - 1u;
    return K26RL_REF_OK;
}

const uint8_t *k26rl_ref_bytes(const K26RlRef *ref, uint64_t *out_len)
{
    if (!ref)
        return NULL;
    if (out_len)
        *out_len = ref->len;
    return ref->bytes;
}
