/* k26rl_ref.h - the reference file a planner emits and an executor
 * reads.
 *
 * A `.k26ref` file is a plan: a sequence of states a craft should be
 * in, and when. It is not a trajectory to be tracked pointwise, not a
 * control schedule and not a set of instructions. It carries no code,
 * so loading one is loading data.
 *
 * The file is the interface between two models that are trained
 * separately and run together. One of them produces a plan on the
 * mission clock; the other flies the craft on the vehicle clock,
 * reading the plan beside its own sensors. Neither has to know
 * anything about the other beyond this file, which is why the file is
 * versioned, digested, and self-describing rather than an
 * arrangement between two programs that happen to agree.
 *
 * Layout. It follows the policy file's conventions rather than
 * inventing new ones, because two formats in one tree that disagree
 * about magic, versioning and digests are two formats to get wrong.
 * Every multi-byte integer is little-endian and is assembled field by
 * field, never by writing a struct; every real is an IEEE-754
 * binary64 bit pattern. The fixed header is K26RL_REF_HEADER_BYTES
 * bytes:
 *
 *     0   8  magic, K26RL_REF_MAGIC
 *     8   4  format version, uint32
 *    12   4  header bytes, uint32
 *    16  32  digest, SHA-256 (see below)
 *    48   4  flags, uint32
 *    52   4  frame kind, uint32 from the closed list below
 *    56   4  frame name bytes, uint32
 *    60   4  knot count, uint32
 *    64   4  provenance bytes, uint32
 *    68   4  reserved, uint32 zero
 *    72   8  epoch, binary64 seconds
 *
 * The body follows immediately, in this order and with no padding:
 *
 *   1. The frame name: `frame name bytes` of UTF-8, unterminated.
 *   2. The provenance text: `provenance bytes` of UTF-8,
 *      unterminated.
 *   3. `knot count` knots, each K26RL_REF_KNOT_BYTES bytes: eight
 *      binary64 values in the order time offset, position x, y, z,
 *      velocity x, y, z, tolerance radius.
 *
 * The file's length is exactly the total those sections imply. A
 * shorter file is truncated and a longer one carries trailing bytes;
 * both are refused, so no byte of a `.k26ref` is outside the digest.
 *
 * Identity. The digest is SHA-256 over the whole file with the 32
 * digest bytes themselves read as zero, which is the frame checksum's
 * convention in k26rl_episode.h applied to a whole file, and the
 * convention k26rl_policy.h already uses. The frame name, the epoch,
 * the provenance text and every knot therefore sit inside the hashed
 * region: a plan relabelled, re-framed, or altered in one knot is a
 * different plan and says so at load.
 *
 * What a knot means. The time offset is seconds from the epoch. The
 * position and the velocity are the state the craft should be in at
 * that time, in the frame the header names. The tolerance radius is
 * the distance inside which the craft is considered to have met the
 * knot, in metres; it is what makes a knot a requirement rather than
 * a suggestion, and it is the plan's statement of how much it cares.
 *
 * A knot whose tolerance is zero is absent. That is how a producer
 * with a fixed number of slots to fill emits fewer knots than it has
 * slots, and this reader drops such knots on load: the knot count in
 * the header is what the file carries, k26rl_ref_info's
 * `present_count` is what the plan asks for, and every accessor below
 * indexes the present knots. A negative tolerance is refused rather
 * than treated as absent, because it is not a plan a producer can
 * mean.
 *
 * The present knots' time offsets strictly increase. A plan whose
 * knots do not is refused: two knots at one time say two things about
 * one instant, and a knot before the one preceding it can never be
 * current.
 *
 * Attitude is deliberately not in a knot. A plan written on the
 * mission clock has no business commanding an attitude on the
 * vehicle clock, and a controller that must point itself to fly a
 * translation should decide that for itself. A later format version
 * may add an optional attitude and a flag saying whether it is
 * required; that is an addition and not a change.
 *
 * One plan addresses one craft. A formation is not expressible here
 * and would want a frame this format does not have.
 *
 * Nothing here runs on a stepping path: a plan is loaded once, before
 * stepping, and reading a knot afterwards is an indexed read of
 * memory the load allocated. */
#ifndef K26RL_REF_H
#define K26RL_REF_H

#include <stdint.h>

#include "k26rl_digest.h"

#ifdef __cplusplus
extern "C" {
#endif

#define K26RL_REF_MAGIC          "K26REF\0\0"   /* 8 bytes, file start */
#define K26RL_REF_SUFFIX         ".k26ref"
#define K26RL_REF_FORMAT_VERSION ((uint32_t)1)
#define K26RL_REF_HEADER_BYTES   ((uint32_t)80)
#define K26RL_REF_KNOT_BYTES     ((uint32_t)64)

/* Byte offset of the digest field, which reads as zero while the
 * digest is computed. Exposed because a writer needs the same number
 * the reader uses. */
#define K26RL_REF_DIGEST_OFFSET  ((uint32_t)16)

/* Ceilings. A file declares its own sizes, so a corrupt or hostile
 * one must not be able to ask for an unbounded allocation before
 * anything has been verified. These are the bounds a refusal names. */
#define K26RL_REF_MAX_KNOTS      ((uint32_t)4096)
#define K26RL_REF_MAX_FRAME      ((uint32_t)256)
#define K26RL_REF_MAX_PROVENANCE ((uint32_t)4096)
#define K26RL_REF_MAX_BYTES      ((uint64_t)4194304)   /* 4 MiB */

/* Frame kinds, a closed list. The frame is named by a body, and the
 * kind says which set of axes centred on that body the knots are
 * written in.
 *
 * K26RL_REF_FRAME_LVLH is the named body's local-vertical
 * local-horizontal frame: the first axis outward from the body it
 * orbits, the third along the orbital angular momentum, the second
 * completing the right-handed set along the direction of motion. It
 * is the frame a close approach is naturally written in, and it
 * requires the named body to be orbiting something.
 *
 * K26RL_REF_FRAME_INERTIAL is the non-rotating frame centred on the
 * named body, on the world's own axes. It is the frame a transfer is
 * naturally written in, and it asks nothing of the named body beyond
 * existing. */
#define K26RL_REF_FRAME_LVLH     ((uint32_t)0)
#define K26RL_REF_FRAME_INERTIAL ((uint32_t)1)

/* Flag bits. Version 1 defines none: the format has no optional
 * section, so every bit is unknown and a file that sets one is
 * refused rather than partly read. The constant is here so that the
 * check reads the same way the policy format's does. */
#define K26RL_REF_FLAGS_KNOWN    ((uint32_t)0)

/* Status registry for this header. Append-only: values are never
 * renumbered.
 *
 * It is a registry of its own rather than an extension of
 * k26rl_env.h's, because that surface is frozen and adding to its
 * enum would be a change to it. The policy layer set that precedent
 * and this one follows it. */
typedef enum {
    K26RL_REF_OK          = 0,
    K26RL_REF_E_NULL      = 1,  /* null pointer argument */
    K26RL_REF_E_IO        = 2,  /* the file could not be read or written */
    K26RL_REF_E_MEMORY    = 3,  /* allocation failed */
    K26RL_REF_E_MAGIC     = 4,  /* not a reference file */
    K26RL_REF_E_VERSION   = 5,  /* format version not served */
    K26RL_REF_E_TRUNCATED = 6,  /* file shorter than it declares */
    K26RL_REF_E_TRAILING  = 7,  /* bytes past the declared end */
    K26RL_REF_E_DIGEST    = 8,  /* digest disagrees with the bytes */
    K26RL_REF_E_SIZE      = 9,  /* a declared size exceeds a ceiling */
    K26RL_REF_E_RESERVED  = 10, /* a reserved field is not zero */
    K26RL_REF_E_FLAGS     = 11, /* an unknown flag bit is set */
    K26RL_REF_E_FRAME     = 12, /* frame kind outside the list, or the
                                 * frame name is empty */
    K26RL_REF_E_KNOTS     = 13, /* the file declares no knot at all */
    K26RL_REF_E_ORDER     = 14, /* the present knots' times do not
                                 * strictly increase */
    K26RL_REF_E_VALUE     = 15, /* a knot carries a value that is not a
                                 * usable finite number */
    K26RL_REF_E_RANGE     = 16  /* an index is outside what the plan
                                 * holds */
} K26RlRefStatus;

/* A loaded reference. The layout is private: a caller reads it
 * through the accessors below. */
typedef struct K26RlRef K26RlRef;

/* One knot: the state the craft should be in, and when. */
typedef struct {
    double t;            /* seconds from the epoch */
    double r[3];         /* position in the header's frame, metres */
    double v[3];         /* velocity in the header's frame, m/s */
    double tolerance;    /* metres; strictly positive for a present knot */
} K26RlRefKnot;

/* What a loaded reference declares about itself. */
typedef struct {
    uint32_t format_version;
    uint32_t flags;
    uint32_t frame_kind;
    uint32_t frame_name_bytes;
    uint32_t knot_count;      /* knots the file carries */
    uint32_t present_count;   /* knots with a tolerance above zero */
    uint32_t provenance_bytes;
    double   epoch;           /* seconds; a knot's time is epoch + offset */
    uint8_t  digest[K26RL_SHA256_BYTES];
} K26RlRefInfo;

/* A plan on its way to a file: what a producer hands the writer.
 * `knots` are slots in the producer's own order, absent ones
 * included; the writer selects and orders them by the rule
 * k26rl_ref_encode documents. */
typedef struct {
    uint32_t            frame_kind;
    const char         *frame_name;   /* not terminated by the file */
    const char         *provenance;   /* may be null for none */
    double              epoch;
    const K26RlRefKnot *knots;
    uint32_t            knot_count;   /* slots, present and absent */
} K26RlRefPlan;

/**
 * @brief Decode a reference status code.
 * @param status The code.
 * @return A stable string; unknown values decode to one string rather
 *         than failing, as the registry is append-only.
 */
const char *k26rl_ref_status_str(K26RlRefStatus status);

/**
 * @brief Load a plan from a `.k26ref` file.
 * @param path File to read.
 * @param out  Receives the loaded plan on success, untouched
 *             otherwise; release it with k26rl_ref_close.
 * @return K26RL_REF_OK, or the refusal.
 * @note  The whole file is read, its digest verified over its own
 *        bytes, and every declared size checked before anything is
 *        indexed. A file above K26RL_REF_MAX_BYTES is refused without
 *        being read.
 */
K26RlRefStatus k26rl_ref_open(const char *path, K26RlRef **out);

/**
 * @brief Load a plan from bytes already in memory.
 * @param bytes The file's bytes; the plan copies what it keeps, so
 *              the caller may free them on return.
 * @param len   Byte count.
 * @param out   Receives the loaded plan on success, untouched
 *              otherwise; release it with k26rl_ref_close.
 * @return K26RL_REF_OK, or the refusal.
 */
K26RlRefStatus k26rl_ref_parse(const uint8_t *bytes, uint64_t len,
                               K26RlRef **out);

/**
 * @brief Release a loaded plan. Null is accepted and ignored.
 * @param ref The plan.
 */
void k26rl_ref_close(K26RlRef *ref);

/**
 * @brief What the plan declares about itself.
 * @param ref The plan.
 * @param out Receives the declarations.
 * @return K26RL_REF_OK, or K26RL_REF_E_NULL.
 */
K26RlRefStatus k26rl_ref_info(const K26RlRef *ref, K26RlRefInfo *out);

/**
 * @brief The body the plan's frame is centred on.
 * @param ref     The plan.
 * @param out_len Receives the byte count, or is null.
 * @return The name, terminated by the loader for convenience and not
 *         by the file; null when the plan is null.
 */
const char *k26rl_ref_frame_name(const K26RlRef *ref, uint32_t *out_len);

/**
 * @brief The provenance text, naming what produced this plan.
 * @param ref     The plan.
 * @param out_len Receives the byte count, or is null.
 * @return The text, terminated by the loader and not by the file;
 *         null when the plan is null. A plan carrying none returns an
 *         empty string with a length of zero.
 */
const char *k26rl_ref_provenance(const K26RlRef *ref, uint32_t *out_len);

/**
 * @brief One present knot, in time order.
 * @param ref   The plan.
 * @param index Below `present_count`.
 * @param out   Receives the knot, with its time offset as the file
 *              carries it rather than added to the epoch.
 * @return K26RL_REF_OK, K26RL_REF_E_NULL, or K26RL_REF_E_RANGE.
 */
K26RlRefStatus k26rl_ref_knot(const K26RlRef *ref, uint32_t index,
                              K26RlRefKnot *out);

/**
 * @brief The current knot at a time.
 * @param ref   The plan.
 * @param t     The time, on the same clock the epoch is measured on.
 * @param out   Receives the index of the current knot.
 * @return K26RL_REF_OK, or K26RL_REF_E_NULL; K26RL_REF_E_KNOTS when
 *         the plan holds no present knot.
 * @note  The current knot is the earliest present knot whose time has
 *        not passed, which is the earliest whose epoch plus offset is
 *        at least `t`. A knot's time passing without the craft inside
 *        its tolerance is not an error and ends nothing: the plan
 *        advances and the craft is late.
 * @note  When every knot's time has passed, the last knot stays
 *        current. A plan that has run out is a plan whose final state
 *        is the one still asked for, and publishing nothing instead
 *        would be indistinguishable from a craft sitting exactly on a
 *        knot.
 * @note  The comparison is against the knot time computed as epoch
 *        plus offset in binary64, one addition, so two
 *        implementations agree on which knot is current at a boundary
 *        rather than on a value near it.
 */
K26RlRefStatus k26rl_ref_current(const K26RlRef *ref, double t,
                                 uint32_t *out);

/**
 * @brief The plan's own file bytes.
 * @param ref     The plan.
 * @param out_len Receives the byte count, or is null.
 * @return The bytes the plan was loaded from, unchanged; null when
 *         the plan is null. They belong to the plan and live until it
 *         is closed.
 * @note  Here so that a consumer writing a plan into a record does
 *        not have to re-read the file it came from, and so that what
 *        the record carries is the bytes the digest was taken over.
 */
const uint8_t *k26rl_ref_bytes(const K26RlRef *ref, uint64_t *out_len);

/**
 * @brief Encode a plan into `.k26ref` bytes.
 * @param plan    The plan.
 * @param out     Receives a malloc'd buffer the caller frees.
 * @param out_len Receives its byte count.
 * @return K26RL_REF_OK, or the refusal.
 * @note  Slots are selected and ordered here rather than by the
 *        producer, because a producer whose slots are a fixed-width
 *        output has no way to guarantee either. The rule: a slot
 *        whose tolerance is not above zero is absent; the remaining
 *        slots are sorted by time offset, ties broken by the order
 *        they were given in; and a slot whose time offset does not
 *        strictly exceed the last one kept is dropped, because it
 *        could never be current.
 * @note  A value that is not finite is refused with
 *        K26RL_REF_E_VALUE rather than encoded. A plan left with no
 *        knot at all still encodes, to a file whose knot count is
 *        zero; loading that file is refused with K26RL_REF_E_KNOTS,
 *        which is the honest report of a producer that asked for
 *        nothing and is why the encode does not refuse it here.
 */
K26RlRefStatus k26rl_ref_encode(const K26RlRefPlan *plan, uint8_t **out,
                                uint64_t *out_len);

/**
 * @brief Encode a plan into buffers the caller owns.
 * @param plan    The plan.
 * @param scratch Room for `plan->knot_count` knots, used for the
 *                selection and ordering and not read afterwards.
 * @param out     Room for the encoded file.
 * @param cap     Bytes available at `out`. The worst case is
 *                K26RL_REF_HEADER_BYTES plus the frame name, the
 *                provenance text and `knot_count` times
 *                K26RL_REF_KNOT_BYTES; a smaller capacity is refused
 *                with K26RL_REF_E_SIZE rather than truncating.
 * @param out_len Receives the bytes written.
 * @return K26RL_REF_OK, or the refusal.
 * @note  Here so that a producer working under a no-allocation
 *        contract can write a plan without leaving it. The selection
 *        and ordering rule is k26rl_ref_encode's, which this is the
 *        body of.
 */
K26RlRefStatus k26rl_ref_encode_into(const K26RlRefPlan *plan,
                                     K26RlRefKnot *scratch,
                                     uint8_t *out, uint64_t cap,
                                     uint64_t *out_len);

/**
 * @brief Encode a plan and write it to a file.
 * @param path Path to create or replace.
 * @param plan The plan.
 * @return K26RL_REF_OK, or the refusal.
 * @note  Replaces an existing file rather than refusing it: a plan is
 *        a product of one episode of one run, and the identity that
 *        keeps two plans apart belongs in the path the caller chose.
 */
K26RlRefStatus k26rl_ref_write(const char *path, const K26RlRefPlan *plan);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_REF_H */
