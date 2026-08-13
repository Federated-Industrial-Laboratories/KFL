/* k26rl_episode.h - the binary episode and telemetry record format.
 *
 * One schema, written natively by batch runs and published unchanged
 * on the live telemetry ring, so an episode file is an offline
 * dataset and live visibility consumes records, never the simulation.
 * This header is the file transport: writer and reader.
 *
 * A file is an 8-byte magic "K26EPI\0\0", a stream of frames, then
 * optionally an index frame as the final frame and a fixed 16-byte
 * trailer ("K26EPIX\0" plus the uint64 offset of the index frame).
 * Every multi-byte integer is little-endian, assembled and read field
 * by field on every host; doubles travel as their IEEE-754 binary64
 * bit patterns in 64-bit integers. A file killed mid-write is
 * readable up to its last complete frame: a frame whose CRC fails or
 * whose header arithmetic runs past end-of-file ends the readable
 * prefix, and everything before it is valid.
 *
 * Every frame carries a 24-byte header: kind (uint16, append-only
 * registry below), flags (uint16, kind-specific, zero where unused),
 * length (uint32 payload bytes), sequence (uint64, monotone per
 * producer), crc (uint32, CRC-32C over the header bytes with this
 * field zero, then the payload), reserved (uint32 zero). The CRC
 * covering the header means a corrupted length cannot silently
 * misalign the walk.
 *
 * Episode identity in a file is the triple (rekey ordinal, env index,
 * episode index); every ordinal appearing in any frame resolves to a
 * seed recorded in that same file, so an episode is exactly
 * reproducible from the file alone given the producing artifact. A
 * step record is a transition, with one exception: a faulted
 * episode's final step record carries the faulting call's actions,
 * the pre-step observations, zero rewards, zero applied dt, and the
 * fault flag alone. Step-chunk payloads are column-major (a channel
 * is one contiguous run per chunk); within an episode every chunk
 * carries exactly steps-per-chunk steps except the final one.
 *
 * The writer's per-step path appends to memory preallocated at open
 * and performs no I/O; a bounded buffered flush occurs at chunk
 * boundaries and episode ends. Readers skip unknown frame kinds by
 * length and unknown spec tags by length. */
#ifndef K26RL_EPISODE_H
#define K26RL_EPISODE_H

#include <stdint.h>

#include "k26rl_env.h"

#ifdef __cplusplus
extern "C" {
#endif

#define K26RL_EPISODE_FORMAT_VERSION ((uint32_t)1)
#define K26RL_EPISODE_MAGIC   "K26EPI\0\0"   /* 8 bytes, file start */
#define K26RL_EPISODE_TRAILER "K26EPIX\0"    /* 8 bytes + uint64 offset */
#define K26RL_EPISODE_FRAME_HEADER_SIZE ((uint32_t)24)

/* Frame kinds. Append-only: kinds are never renumbered or reused. */
#define K26RL_FRAME_FILE_HEADER   ((uint16_t)0x0001)
#define K26RL_FRAME_EPISODE_START ((uint16_t)0x0002)
#define K26RL_FRAME_STEP_CHUNK    ((uint16_t)0x0003)
#define K26RL_FRAME_EPISODE_END   ((uint16_t)0x0004)
#define K26RL_FRAME_INDEX         ((uint16_t)0x0005)
#define K26RL_FRAME_REKEY         ((uint16_t)0x0006)

/* Episode end reasons (episode-end frame, uint16). Fault is distinct
 * from termination and truncation: it means the physics could not
 * honestly advance, and the frame's fault code carries the reason
 * from the k26rl_env.h status registry. */
#define K26RL_END_TERMINATED ((uint16_t)0)
#define K26RL_END_TRUNCATED  ((uint16_t)1)
#define K26RL_END_FAULT      ((uint16_t)2)

/* CRC-32C (Castagnoli), the frame checksum: reflected polynomial,
 * initial and final value 0xFFFFFFFF, table-driven, in-tree. Exposed
 * so tests can pin the published check values. */
uint32_t k26rl_crc32c(uint32_t crc_in, const void *data, uint64_t len);
#define K26RL_CRC32C_INIT ((uint32_t)0)   /* pass 0 to start a new CRC */

/* Fixed per-file geometry, stated by the producer at open. The spec
 * blob is embedded verbatim in the file header so a reader needs no
 * side channel, but the writer takes the numbers it sizes buffers by
 * explicitly. Chunk and boundary buffers are per environment and
 * sized from these numbers at open, so resident memory scales as
 * n_envs times the chunk frame; the producer chooses steps_per_chunk
 * with that in mind. */
typedef struct {
    uint32_t n_envs;
    uint32_t agent_count;
    uint32_t obs_total;        /* doubles per environment */
    uint32_t act_total;        /* doubles per environment */
    uint32_t steps_per_chunk;  /* any positive value; 1024 is the default */
    uint32_t dr_max;           /* most domain-randomisation pairs any
                                * episode-start carries; starts refuse
                                * above it */
} K26RlEpisodeGeom;

/* Writer ----------------------------------------------------------- */

typedef struct K26RlEpisodeWriter K26RlEpisodeWriter;

/* Open a new episode file. Refuses an existing path with
 * K26RL_E_OUTPUT_EXISTS: nothing truncates a completed run and no
 * file carries two file-headers. Writes the file-header frame
 * recording the governing seed and rekey ordinal in force at enable,
 * the geometry, both version strings, and the spec blob verbatim.
 * Every buffer the writer will ever touch between open and close is
 * allocated here, once: the per-environment chunk buffers and the
 * per-environment boundary scratch, so steps, episode starts, and
 * episode ends allocate nothing. The index is assembled at close by
 * one pass over the file the writer wrote, so no per-chunk or
 * per-episode bookkeeping grows while running. */
K26RlStatus k26rl_episode_writer_open(const char *path,
                                      const K26RlEpisodeGeom *geom,
                                      uint64_t governing_seed,
                                      uint32_t rekey_ordinal,
                                      const char *grammar_version,
                                      const char *runtime_version,
                                      const uint8_t *spec, uint32_t spec_len,
                                      K26RlEpisodeWriter **out);

/* Begin an environment's episode: emits the episode-start frame with
 * the initial observation (obs_total values) and the
 * domain-randomisation record (count pairs of tag and value bits).
 * For each environment, its episode-end frame precedes its next
 * episode-start in file order. */
K26RlStatus k26rl_episode_writer_start(K26RlEpisodeWriter *w,
                                       uint32_t env, uint32_t episode,
                                       const double *initial_obs,
                                       const uint32_t *dr_tags,
                                       const double *dr_values,
                                       uint32_t dr_count);

/* Append one step record to the environment's current episode. The
 * hot path: copies into the preallocated chunk buffer only; a full
 * chunk flushes as one buffered write. Observations are the
 * post-step values; a faulted episode's final record carries what the
 * format fixes for it and is appended through this same call. Flag
 * words are recorded as given: bit 3 and every higher bit are the
 * producer's to keep zero, per the format. */
K26RlStatus k26rl_episode_writer_step(K26RlEpisodeWriter *w, uint32_t env,
                                      const double *obs, const double *act,
                                      const double *rewards, uint32_t flags,
                                      double applied_dt);

/* End the environment's episode: flushes its partial chunk, emits the
 * episode-end frame (step count, end reason, fault code zero unless
 * fault, per-agent terminal reward adjustments, zero for a faulted
 * episode), and records the episode in the index. */
K26RlStatus k26rl_episode_writer_end(K26RlEpisodeWriter *w, uint32_t env,
                                     uint16_t end_reason, uint16_t fault_code,
                                     const double *terminal_adjustments);

/* Record a rekey: emits the rekey frame carrying the new seed and the
 * next ordinal, after which episode indices restart at zero under the
 * new key. Returns the new ordinal through out_ordinal when non-null. */
K26RlStatus k26rl_episode_writer_rekey(K26RlEpisodeWriter *w,
                                       uint64_t new_seed,
                                       uint32_t *out_ordinal);

/* Clean close: writes the index frame, then the trailer, then closes.
 * A file without them (killed mid-write) is still readable to its
 * last complete frame. */
K26RlStatus k26rl_episode_writer_close(K26RlEpisodeWriter *w);

/* Reader ----------------------------------------------------------- */

typedef struct K26RlEpisodeReader K26RlEpisodeReader;

/* What open() learned about the file. clean_close says whether a
 * valid trailer located the index in one seek; when it is zero the
 * index was reconstructed by one sequential pass and readable_bytes
 * marks where the valid prefix ended. */
typedef struct {
    uint32_t format_version;
    uint64_t governing_seed;    /* at enable */
    uint32_t rekey_ordinal;     /* at enable */
    uint32_t n_envs;
    uint32_t steps_per_chunk;
    uint32_t agent_count;       /* parsed from the spec blob */
    uint32_t obs_total;
    uint32_t act_total;
    uint64_t readable_bytes;
    uint32_t episode_count;     /* complete episodes in the index */
    uint32_t unindexed_episode_starts; /* episode-start frames in the
                                * readable prefix with no indexed
                                * episode: a producer closed with an
                                * episode open, or the tail was cut */
    uint8_t  clean_close;
} K26RlEpisodeInfo;

K26RlStatus k26rl_episode_reader_open(const char *path,
                                      K26RlEpisodeReader **out);
K26RlStatus k26rl_episode_reader_info(const K26RlEpisodeReader *r,
                                      K26RlEpisodeInfo *out);

/* The embedded spec blob, verbatim. The pointer stays valid for the
 * reader's lifetime. */
K26RlStatus k26rl_episode_reader_spec(const K26RlEpisodeReader *r,
                                      const uint8_t **out, uint32_t *out_len);

/* The seed a rekey ordinal resolves to, from the file alone. */
K26RlStatus k26rl_episode_reader_seed(const K26RlEpisodeReader *r,
                                      uint32_t ordinal, uint64_t *out_seed);

/* One decoded episode, addressed directly by its identity triple
 * through the index, never by scanning. Arrays are allocated by the
 * call and freed with k26rl_episode_free; step-major layouts
 * (obs[step * obs_total + channel], and so on). step_count includes a
 * faulted episode's final fault record. */
typedef struct {
    uint32_t rekey_ordinal;
    uint32_t env;
    uint32_t episode;
    uint32_t step_count;
    uint16_t end_reason;
    uint16_t fault_code;
    double  *initial_obs;          /* obs_total */
    uint32_t dr_count;
    uint32_t *dr_tags;
    double  *dr_values;
    double  *obs;                  /* step_count * obs_total */
    double  *act;                  /* step_count * act_total */
    double  *rewards;              /* step_count * agent_count */
    uint32_t *flags;               /* step_count */
    double  *applied_dt;           /* step_count */
    double  *terminal_adjustments; /* agent_count */
} K26RlEpisodeData;

K26RlStatus k26rl_episode_read(K26RlEpisodeReader *r,
                               uint32_t rekey_ordinal, uint32_t env,
                               uint32_t episode, K26RlEpisodeData *out);
void k26rl_episode_free(K26RlEpisodeData *data);

/* Identity of the k-th indexed episode, for enumeration. */
K26RlStatus k26rl_episode_reader_at(const K26RlEpisodeReader *r, uint32_t k,
                                    uint32_t *out_ordinal, uint32_t *out_env,
                                    uint32_t *out_episode);

void k26rl_episode_reader_close(K26RlEpisodeReader *r);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_EPISODE_H */
