/* k26rl_tap.h - the telemetry tap's shared memory ring.
 *
 * The second transport for the record format k26rl_episode.h defines.
 * A serving simulation publishes the same frame kinds with the same
 * payloads it would write to an episode file; a consumer attaches
 * passively and reads them. The file transport writes every frame;
 * this one is lossy by design, so a slow, absent, or dead consumer
 * never blocks a step and never changes what a step computes.
 *
 * Layout of the shared memory object:
 *
 *   preamble    control block, then the head cursor on its own cache
 *               line, then one complete file-header frame in the
 *               ordinary frame encoding. Written once when the tap is
 *               enabled; no overwrite ever reaches it, so a consumer
 *               attaching an hour into a run still recovers the format
 *               version, the geometry, and the spec blob.
 *   slots       slot_count fixed-size slots, each holding one frame,
 *               each 64-byte aligned. slot_count is a power of two, so
 *               a frame's slot is its sequence masked, never divided.
 *
 * Every multi-byte integer in the object is little-endian and is
 * assembled field by field, exactly as the file transport does, so a
 * consumer reads the same bytes on any host.
 *
 * Sequence numbering is this producer's own and is independent of the
 * file transport's: the preamble's file-header frame is sequence 0 and
 * slot frames run from 1, so a zero sequence in a slot means the slot
 * has never been written. A consumer detects loss by arithmetic on
 * the sequence alone.
 *
 * Publication protocol, single producer, many independent consumers:
 * the producer invalidates the target slot's sequence field, writes
 * the frame body, then release-stores the true sequence, then
 * release-stores the head cursor. A consumer acquire-loads a slot's
 * sequence, copies the frame out, re-loads the sequence, and requires
 * the two loads to agree; the frame's CRC is verified as a second and
 * independent tear check. Because the sequence field is invalidated
 * before the body is touched and restored only after it is whole, a
 * copy that raced an overwrite cannot pass the re-read.
 *
 * The consumer's read position lives in the consumer's own memory.
 * The object carries no read cursor, no acknowledgement, and no field
 * a consumer may write, and consumers map it read-only: no control
 * path runs from a consumer into a running simulation. */
#ifndef K26RL_TAP_H
#define K26RL_TAP_H

#include <stdint.h>

#include "k26rl_env.h"
#include "k26rl_episode.h"

#ifdef __cplusplus
extern "C" {
#endif

#define K26RL_TAP_MAGIC "K26TAP\0\0"          /* 8 bytes, object start */
#define K26RL_TAP_LAYOUT_VERSION ((uint32_t)1)

/* Shared memory object names are the caller's tap name under this
 * prefix, which is how one host's rings are told from another's. */
#define K26RL_TAP_OBJECT_PREFIX "/k26rl."

/* Alignment quantum for the control block, the head cursor, the
 * header frame, and every slot: one cache line on the targets this
 * runs on, so the producer's cursor store never shares a line with
 * the preamble a consumer is reading. */
#define K26RL_TAP_LINE ((uint32_t)64)

/* Ring sizing. The slot count is the largest power of two whose
 * product with the slot size fits the budget, floored at
 * K26RL_TAP_SLOTS_MIN so a wide environment still gets a usable ring,
 * and refused with K26RL_E_GEOMETRY when even the floor would exceed
 * K26RL_TAP_BYTES_MAX. */
#define K26RL_TAP_BUDGET_DEFAULT ((uint64_t)16 * 1024 * 1024)
#define K26RL_TAP_SLOTS_MIN      ((uint32_t)1024)
#define K26RL_TAP_BYTES_MAX      ((uint64_t)256 * 1024 * 1024)

/* Control block field offsets, little-endian, from the object start. */
#define K26RL_TAP_OFF_MAGIC        0    /* 8 bytes */
#define K26RL_TAP_OFF_LAYOUT       8    /* uint32 */
#define K26RL_TAP_OFF_FORMAT      12    /* uint32, K26RL_EPISODE_FORMAT_VERSION */
#define K26RL_TAP_OFF_SLOT_SIZE   16    /* uint32 */
#define K26RL_TAP_OFF_SLOT_COUNT  20    /* uint32, power of two */
#define K26RL_TAP_OFF_PREAMBLE    24    /* uint32, bytes to the first slot */
#define K26RL_TAP_OFF_CLOSED      28    /* uint32, set when the producer ends */
#define K26RL_TAP_OFF_HEAD        64    /* uint64, own line: last published */
#define K26RL_TAP_OFF_HEADER     128    /* the file-header frame */

/* Producer -------------------------------------------------------- */

typedef struct K26RlTap K26RlTap;

/* Create the ring and publish the file-header frame into its
 * preamble. Takes the same geometry, seed, versions, and spec blob
 * the episode writer takes, because it publishes the same frames.
 * Every buffer and mapping the producer will ever touch is
 * established here, once, so publication allocates nothing.
 *
 * Refusals: K26RL_E_TAP_NAME for a name outside the rule stated at
 * k26rl_env_tap, K26RL_E_TAP_EXISTS when a ring of that name already
 * exists, K26RL_E_GEOMETRY for a geometry the size ceiling cannot
 * hold, K26RL_E_TAP_UNAVAILABLE when the platform refuses to create,
 * size, or map the object. */
K26RlStatus k26rl_tap_open(const char *name, const K26RlEpisodeGeom *geom,
                           uint64_t governing_seed, uint32_t rekey_ordinal,
                           const char *grammar_version,
                           const char *runtime_version,
                           const uint8_t *spec, uint32_t spec_len,
                           K26RlTap **out);

/* The four publication calls, mirroring the episode writer's so the
 * producing surface calls them at the same points with the same
 * values. They return nothing and cannot fail: after open the ring's
 * geometry is fixed and publication is a bounded copy into a mapping,
 * so there is no failure a caller could act on and no way for the tap
 * to alter a step's control flow. A null tap is a no-op, which is how
 * a disabled tap costs one pointer test.
 *
 * The one frame that cannot be published is one whose payload exceeds
 * the slot the geometry sized, which only a domain-randomisation
 * record larger than the declared dr_max can produce. Such a frame is
 * skipped and its sequence number is consumed, so the loss shows up
 * as a gap rather than as a silent substitution. */
void k26rl_tap_start(K26RlTap *t, uint32_t env, uint32_t episode,
                     const double *initial_obs, const uint32_t *dr_tags,
                     const double *dr_values, uint32_t dr_count);
void k26rl_tap_step(K26RlTap *t, uint32_t env, const double *obs,
                    const double *act, const double *rewards,
                    uint32_t flags, double applied_dt);
void k26rl_tap_end(K26RlTap *t, uint32_t env, uint16_t end_reason,
                   uint16_t fault_code, const double *terminal_adjustments);
void k26rl_tap_rekey(K26RlTap *t, uint64_t new_seed);

/* Mark the ring closed, remove its name, and release the producer's
 * mapping. Consumers already attached keep valid mappings, because
 * removing the name does not remove the object, and see the closed
 * mark. */
void k26rl_tap_close(K26RlTap *t);

/* Slot frames published so far, the preamble's file-header frame not
 * counted. A consumer that attached before the first slot frame and
 * drained to the close should account for exactly this many: what it
 * accepted plus what it reports lost. */
uint64_t k26rl_tap_published(const K26RlTap *t);

/* Consumer -------------------------------------------------------- */

typedef struct K26RlTapReader K26RlTapReader;

/* Attach read-only to a named ring. from_start begins at the first
 * slot frame; otherwise the reader joins at the producer's current
 * position, which is the live-attach case. Refuses a name that does
 * not resolve, a mapping the platform will not give, or an object
 * whose control block does not describe a ring this build understands
 * (K26RL_E_TAP_UNAVAILABLE, K26RL_E_GEOMETRY). */
K26RlStatus k26rl_tap_attach(const char *name, int from_start,
                             K26RlTapReader **out);

/* Copy the next frame into buf. Returns K26RL_OK with *out_len zero
 * when nothing is available yet, which a caller distinguishes from
 * the end of the run by k26rl_tap_reader_closed. *out_lost receives
 * the number of frames overwritten before the one returned, zero when
 * none were, and the reader accumulates the same count. cap smaller
 * than the ring's slot size is refused with K26RL_E_GEOMETRY. */
K26RlStatus k26rl_tap_read(K26RlTapReader *r, uint8_t *buf, uint32_t cap,
                           uint32_t *out_len, uint64_t *out_lost);

/* The preamble's file-header frame, whole and in the frame encoding,
 * and the spec blob carried inside its payload. Both point into the
 * read-only mapping and stay valid until detach. */
K26RlStatus k26rl_tap_reader_header(const K26RlTapReader *r,
                                    const uint8_t **out_frame,
                                    uint32_t *out_len);
K26RlStatus k26rl_tap_reader_spec(const K26RlTapReader *r,
                                  const uint8_t **out_spec,
                                  uint32_t *out_len);

/* Ring geometry, for sizing a read buffer. */
K26RlStatus k26rl_tap_reader_info(const K26RlTapReader *r,
                                  uint32_t *out_slot_size,
                                  uint32_t *out_slot_count);

int      k26rl_tap_reader_closed(const K26RlTapReader *r);
uint64_t k26rl_tap_reader_accepted(const K26RlTapReader *r);
uint64_t k26rl_tap_reader_lost(const K26RlTapReader *r);

void k26rl_tap_detach(K26RlTapReader *r);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_TAP_H */
