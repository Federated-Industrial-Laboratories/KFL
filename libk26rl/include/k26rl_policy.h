/* k26rl_policy.h - the trained-policy file format and its evaluator.
 *
 * A `.k26pol` file is a feedforward policy: the weights that turn the
 * observation channels one agent reads into that agent's action
 * slice, plus the declarations needed to check the file against the
 * world it was trained on and to reproduce its arithmetic anywhere.
 * It carries no code, so loading one is loading data, and evaluating
 * one is this library's arithmetic rather than a property of whatever
 * runtime produced the weights.
 *
 * The file is an inference artifact, not a training checkpoint. It
 * holds what the action path needs and nothing else: a value head,
 * an optimiser state, and a replay buffer are all absent by design,
 * and a trainer that wants to resume keeps its own checkpoint.
 *
 * A policy is exported by the Python package that already marshals
 * for the stepping surface. Nothing in this header knows what
 * produced the weights, and nothing here writes a file.
 *
 * What a policy reads. An agent's observation slice is one run of the
 * environment's observation vector, but the channels a policy is
 * trained on need not be all of it. A world that routes an observe
 * through a sensor publishes the measured channel a policy may read
 * and the ground-truth channel beside it, which it may not, and with
 * more than one such observe the two kinds interleave: no offset and
 * width names the measured ones. So the file names its input channel
 * by channel, as a list of indices into the environment's observation
 * vector in the order the network takes them. The list is channels
 * and nothing else: no expression, no computed index, no channel
 * named twice, and every entry inside the agent's own slice. A file
 * that says otherwise is refused rather than read.
 *
 * Layout. Every multi-byte integer is little-endian and is assembled
 * field by field, never by writing a struct; every real is an
 * IEEE-754 binary64 bit pattern. The fixed header is
 * K26RL_POLICY_HEADER_BYTES bytes:
 *
 *     0   8  magic, K26RL_POLICY_MAGIC
 *     8   4  format version, uint32
 *    12   4  header bytes, uint32
 *    16  32  digest, SHA-256 (see below)
 *    48   4  flags, uint32
 *    52   4  agent count, uint32
 *    56   4  agent index, uint32
 *    60   4  observation total, uint32
 *    64   4  action total, uint32
 *    68   4  observation slice offset, uint32
 *    72   4  observation slice width, uint32
 *    76   4  observation channel count, uint32
 *    80   4  action slice offset, uint32
 *    84   4  action slice width, uint32
 *    88   4  layer count, uint32
 *    92   4  provenance bytes, uint32
 *    96   4  reserved, uint32 zero
 *
 * The observation slice is the run the world gives the agent; the
 * observation channel count is how many of that run's channels this
 * policy reads, which the list in the body names.
 *
 * The body follows immediately, in this order and with no padding:
 *
 *   1. The provenance text: `provenance bytes` of UTF-8, unterminated.
 *   2. The observation channel list: `observation channel count`
 *      uint32 channel indices, in the order the network takes them.
 *      Each lies inside the declared observation slice and each
 *      appears once.
 *   3. `layer count` layers, each an input width (uint32), an output
 *      width (uint32), an activation (uint16 from the closed list
 *      below), a reserved uint16 zero, then output*input binary64
 *      weights and `output` binary64 biases. Weights are output-major:
 *      the weight from input i to output o sits at o*input + i.
 *   4. Present when the standardise flag is set: `observation channel
 *      count` binary64 means, the same count of variances, then the
 *      binary64 epsilon and the binary64 clip.
 *   5. Present when the log-std flag is set: `action slice width`
 *      binary64 values.
 *   6. Present when the clamp flag is set: `action slice width`
 *      binary64 lower bounds then the same count of upper bounds.
 *
 * The file's length is exactly the total those sections imply. A
 * shorter file is truncated and a longer one carries trailing bytes;
 * both are refused, so no byte of a `.k26pol` is outside the digest.
 *
 * Identity. The digest is SHA-256 over the whole file with the 32
 * digest bytes themselves read as zero, which is the frame checksum's
 * convention in k26rl_episode.h applied to a whole file. The
 * provenance text, the spec fields, the channel list, the weights and
 * every flag therefore sit inside the hashed region: a policy
 * relabelled, repointed at another world, pointed at other channels
 * of the same world, or altered in one weight is a different policy
 * and says so at load.
 *
 * Evaluation, declared so that two implementations can agree:
 *
 *   x is what the policy reads: `observation channel count` binary64
 *   values, channel list entry j giving x[j]. A caller holding a
 *   whole environment observation vector has them gathered for it;
 *   one holding them already gathered passes them straight in.
 *
 *   Standardisation, when present: x[i] becomes
 *   (x[i] - mean[i]) / sqrt(variance[i] + epsilon), then clamped to
 *   [-clip, +clip]. A clip of positive infinity clamps nothing;
 *   a clip that is zero, negative, or not a number is refused at
 *   load, as is a variance that is negative or not finite. The
 *   network's input is not the world's observation, which is why
 *   these numbers travel inside the file rather than beside it: a
 *   policy loaded without the statistics it was trained against
 *   produces a confident wrong action instead of an error.
 *
 *   Each layer in file order: y[o] is the running binary64 sum that
 *   starts at bias[o] and adds weight[o][i] * x[i] for i ascending,
 *   one multiplication and one addition at a time; then the layer's
 *   activation is applied to every y[o]; then x becomes y.
 *
 *   Clamping, when present: the result's channel j is clamped to
 *   [lower[j], upper[j]].
 *
 * The accumulation order is part of the format, not an implementation
 * detail, and this library builds with contraction off and standard
 * excess precision so that the declared order is the executed one.
 *
 * The action log standard deviation, when present, is the per-channel
 * log standard deviation of the diagonal Gaussian whose mean the
 * network computes. Evaluation here is deterministic and returns that
 * mean, so nothing in this file reads the value; it is carried
 * because it is a learned parameter of the trained policy that
 * belongs to no layer, and a format that dropped it would make an
 * export quietly lossy. It is published through
 * k26rl_policy_log_std for a caller that wants to sample, which needs
 * a random source this library does not own.
 *
 * Nothing here runs on a stepping path: a policy is loaded once, and
 * evaluation reads preallocated memory and performs no I/O and no
 * allocation. Evaluation writes its intermediate activations into
 * working memory the load allocated, so one loaded policy is
 * evaluated by one thread at a time; concurrent evaluation loads one
 * policy per thread, which is the stepping surface's handle rule
 * applied to the same kind of object.
 *
 * What a caller working to a deadline can rely on. An evaluation
 * allocates nothing, performs no I/O and takes no lock, so what it
 * costs does not depend on what the rest of the process has been
 * doing. The work is a fixed count of multiplications and additions
 * set by the policy's shape alone, so the cost grows in proportion to
 * the parameter count and a policy can be sized against a control
 * period before it is trained. The first evaluations after a load are
 * dearer than the ones that follow, because they are the ones that
 * page the weights in and fill the caches, so a caller with a deadline
 * evaluates once before its first period rather than paying that
 * inside one. */
#ifndef K26RL_POLICY_H
#define K26RL_POLICY_H

#include <stdint.h>

#include "k26rl_digest.h"

#ifdef __cplusplus
extern "C" {
#endif

#define K26RL_POLICY_MAGIC          "K26POL\0\0"   /* 8 bytes, file start */
#define K26RL_POLICY_SUFFIX         ".k26pol"
/* Version 2 replaced version 1's single observation run with the
 * channel list, and this reader serves version 2 alone. A version-1
 * file declares its input as an offset and a width, so serving it
 * would mean inventing the list it does not carry, and a reader that
 * guesses at a field is the thing this format exists to avoid. */
#define K26RL_POLICY_FORMAT_VERSION ((uint32_t)2)
#define K26RL_POLICY_HEADER_BYTES   ((uint32_t)100)

/* Byte offset of the digest field, which reads as zero while the
 * digest is computed. Exposed because a writer needs the same
 * number the reader uses. */
#define K26RL_POLICY_DIGEST_OFFSET  ((uint32_t)16)

/* Ceilings. A file declares its own sizes, so a corrupt or hostile
 * one must not be able to ask for an unbounded allocation before
 * anything has been verified. These are the bounds a refusal names;
 * they are far above any feedforward policy this format is for. */
#define K26RL_POLICY_MAX_LAYERS     ((uint32_t)64)
#define K26RL_POLICY_MAX_WIDTH      ((uint32_t)65536)
#define K26RL_POLICY_MAX_PROVENANCE ((uint32_t)4096)
#define K26RL_POLICY_MAX_BYTES      ((uint64_t)268435456)   /* 256 MiB */

/* Flag bits. Each names a body section; an unknown bit means an
 * unknown section, so the byte arithmetic no longer holds and the
 * file is refused rather than partly read. */
#define K26RL_POLICY_FLAG_STANDARDISE ((uint32_t)1u << 0)
#define K26RL_POLICY_FLAG_LOG_STD     ((uint32_t)1u << 1)
#define K26RL_POLICY_FLAG_CLAMP       ((uint32_t)1u << 2)
#define K26RL_POLICY_FLAGS_KNOWN                                       \
    (K26RL_POLICY_FLAG_STANDARDISE | K26RL_POLICY_FLAG_LOG_STD |       \
     K26RL_POLICY_FLAG_CLAMP)

/* Activations, a closed list. This version expresses feedforward
 * networks over these four; a policy needing another arrives under a
 * later format version rather than through a value nothing here can
 * evaluate. */
#define K26RL_POLICY_ACT_IDENTITY ((uint16_t)0)
#define K26RL_POLICY_ACT_TANH     ((uint16_t)1)
#define K26RL_POLICY_ACT_RELU     ((uint16_t)2)
#define K26RL_POLICY_ACT_LOGISTIC ((uint16_t)3)

/* Status registry for this header. Append-only: values are never
 * renumbered.
 *
 * It is a registry of its own rather than an extension of
 * k26rl_env.h's, because that surface is frozen and adding to its
 * enum would be a change to it. The eight spec fields have eight
 * codes on purpose: a check that compared totals alone could not
 * report which field disagreed, so the code a refusal carries is
 * itself the evidence that the check is field-wise. There is no code
 * for an out-of-range agent index, because the load already bounds
 * the index by the policy's own agent count and the check requires
 * that count to equal the artifact's, so no file can reach one; a
 * status nothing can return is a status no gate can fail on. */
typedef enum {
    K26RL_POLICY_OK               = 0,
    K26RL_POLICY_E_NULL           = 1,  /* null pointer argument */
    K26RL_POLICY_E_IO             = 2,  /* the file could not be read */
    K26RL_POLICY_E_MEMORY         = 3,  /* allocation failed */
    K26RL_POLICY_E_MAGIC          = 4,  /* not a policy file */
    K26RL_POLICY_E_VERSION        = 5,  /* format version not served */
    K26RL_POLICY_E_TRUNCATED      = 6,  /* file shorter than it declares */
    K26RL_POLICY_E_TRAILING       = 7,  /* bytes past the declared end */
    K26RL_POLICY_E_DIGEST         = 8,  /* digest disagrees with the bytes */
    K26RL_POLICY_E_SIZE           = 9,  /* a declared size exceeds a ceiling */
    K26RL_POLICY_E_RESERVED       = 10, /* a reserved field is not zero */
    K26RL_POLICY_E_FLAGS          = 11, /* an unknown flag bit is set */
    K26RL_POLICY_E_ACTIVATION     = 12, /* activation outside the list */
    K26RL_POLICY_E_LAYERS         = 13, /* layer widths do not chain */
    K26RL_POLICY_E_STATISTICS     = 14, /* a declared statistic or bound
                                         * is unusable */
    K26RL_POLICY_E_SLICE          = 15, /* a slice runs past its total */
    K26RL_POLICY_E_SPEC           = 16, /* the spec blob is malformed */
    K26RL_POLICY_E_AGENT_COUNT    = 17, /* agent count disagrees */
    K26RL_POLICY_E_OBS_TOTAL      = 18, /* observation total disagrees */
    K26RL_POLICY_E_ACT_TOTAL      = 19, /* action total disagrees */
    K26RL_POLICY_E_OBS_OFFSET     = 20, /* observation slice offset
                                         * disagrees */
    K26RL_POLICY_E_OBS_WIDTH      = 21, /* observation slice width disagrees */
    K26RL_POLICY_E_ACT_OFFSET     = 22, /* action slice offset disagrees */
    K26RL_POLICY_E_ACT_WIDTH      = 23, /* action slice width disagrees */
    K26RL_POLICY_E_ACT_BOUNDS     = 24, /* clamp reaches outside the world's
                                         * declared action bounds */
    K26RL_POLICY_E_CHANNELS       = 25, /* the observation channel list
                                         * repeats a channel or leaves the
                                         * declared slice */
    K26RL_POLICY_E_OBS_CHANNELS   = 26  /* the channels the policy reads are
                                         * not the ones the artifact
                                         * publishes for it, or not in that
                                         * order */
} K26RlPolicyStatus;

/* A loaded policy. The layout is private: a caller reads it through
 * the accessors below. */
typedef struct K26RlPolicy K26RlPolicy;

/* What a loaded policy declares about itself and the world it was
 * trained against. */
typedef struct {
    uint32_t format_version;
    uint32_t flags;
    uint32_t agent_count;
    uint32_t agent_index;
    uint32_t obs_total;      /* doubles per environment */
    uint32_t act_total;      /* doubles per environment */
    uint32_t obs_offset;     /* the agent's observation slice */
    uint32_t obs_width;
    uint32_t obs_channel_count;  /* channels of that slice this policy
                                  * reads, and the network's input width */
    uint32_t act_offset;     /* the agent's action slice */
    uint32_t act_width;
    uint32_t layer_count;
    uint32_t provenance_bytes;
    uint8_t  digest[K26RL_SHA256_BYTES];
} K26RlPolicyInfo;

/* One layer's shape. */
typedef struct {
    uint32_t in_width;
    uint32_t out_width;
    uint16_t activation;
} K26RlPolicyLayer;

/**
 * @brief Decode a policy status code.
 * @param status The code.
 * @return A stable string; unknown values decode to one string rather
 *         than failing, as the registry is append-only.
 */
const char *k26rl_policy_status_str(K26RlPolicyStatus status);

/**
 * @brief Load a policy from a `.k26pol` file.
 * @param path File to read.
 * @param out  Receives the loaded policy on success, untouched
 *             otherwise; release it with k26rl_policy_close.
 * @return K26RL_POLICY_OK, or the refusal.
 * @note  The whole file is read, its digest verified over its own
 *        bytes, and every declared size checked before anything is
 *        indexed. A file above K26RL_POLICY_MAX_BYTES is refused
 *        without being read.
 */
K26RlPolicyStatus k26rl_policy_open(const char *path, K26RlPolicy **out);

/**
 * @brief Load a policy from bytes already in memory.
 * @param bytes The file's bytes; the policy copies what it keeps, so
 *              the caller may free them on return.
 * @param len   Byte count.
 * @param out   Receives the loaded policy on success, untouched
 *              otherwise; release it with k26rl_policy_close.
 * @return K26RL_POLICY_OK, or the refusal.
 */
K26RlPolicyStatus k26rl_policy_parse(const uint8_t *bytes, uint64_t len,
                                     K26RlPolicy **out);

/**
 * @brief Release a loaded policy. Null is accepted and ignored.
 * @param policy The policy.
 */
void k26rl_policy_close(K26RlPolicy *policy);

/**
 * @brief What the policy declares about itself.
 * @param policy The policy.
 * @param out    Receives the declarations.
 * @return K26RL_POLICY_OK, or K26RL_POLICY_E_NULL.
 */
K26RlPolicyStatus k26rl_policy_info(const K26RlPolicy *policy,
                                    K26RlPolicyInfo *out);

/**
 * @brief One layer's shape, in file order.
 * @param policy The policy.
 * @param index  Layer index below the declared layer count.
 * @param out    Receives the shape.
 * @return K26RL_POLICY_OK, K26RL_POLICY_E_NULL, or
 *         K26RL_POLICY_E_LAYERS when the index is out of range.
 */
K26RlPolicyStatus k26rl_policy_layer(const K26RlPolicy *policy,
                                     uint32_t index,
                                     K26RlPolicyLayer *out);

/**
 * @brief The provenance text.
 * @param policy The policy.
 * @param out_len Receives the byte count, or is null.
 * @return The text, terminated by the loader for convenience and not
 *         by the file; null when the policy is null. The bytes belong
 *         to the policy and live until it is closed.
 */
const char *k26rl_policy_provenance(const K26RlPolicy *policy,
                                    uint32_t *out_len);

/**
 * @brief The observation channels the policy reads.
 * @param policy  The policy.
 * @param out_len Receives the channel count, or is null.
 * @return `observation channel count` indices into the environment's
 *         observation vector, in the order the network takes them;
 *         null when the policy is null. The values belong to the
 *         policy and live until it is closed. A caller gathering the
 *         network's input itself reads them here rather than assuming
 *         a contiguous run.
 */
const uint32_t *k26rl_policy_obs_channels(const K26RlPolicy *policy,
                                          uint32_t *out_len);

/**
 * @brief The action log standard deviation.
 * @param policy The policy.
 * @return `action slice width` values, or null when the file carries
 *         none. Nothing in this library reads them; they are here so
 *         a caller that samples has the parameter the trainer learned.
 */
const double *k26rl_policy_log_std(const K26RlPolicy *policy);

/**
 * @brief Check a policy against a loaded artifact's env_spec blob.
 * @param policy   The policy.
 * @param spec     The blob k26rl_env_spec returned.
 * @param spec_len Its byte count.
 * @param detail   Receives a sentence naming both values on a
 *                 disagreement, or is null.
 * @param detail_capacity Bytes available at `detail`, terminator
 *                 included; nothing is written when it is zero.
 * @return K26RL_POLICY_OK when the policy can drive this artifact,
 *         otherwise the code for the field that disagreed.
 * @note  Every field is compared separately and the code names which
 *        one failed, so a policy that is wrong in one field is refused
 *        by that field rather than by a total that happens to differ.
 *        The observation channel list is compared against the measured
 *        channels the artifact publishes inside the agent's slice, in
 *        ascending channel order: a policy naming a channel the
 *        artifact does not publish to it, naming one the artifact
 *        publishes as ground truth, or naming the right channels in
 *        another order is refused, because each of those has the
 *        inference tier feed the network something other than what it
 *        was trained on. The action clamp, when the file carries one,
 *        must lie inside the artifact's declared action bounds: a
 *        policy may be more conservative than the world, never wider
 *        than it.
 */
K26RlPolicyStatus k26rl_policy_check_spec(const K26RlPolicy *policy,
                                          const uint8_t *spec,
                                          uint32_t spec_len,
                                          char *detail,
                                          uint32_t detail_capacity);

/**
 * @brief Evaluate the policy over the channels it reads.
 * @param policy The policy.
 * @param obs    `observation channel count` values, the channels
 *               k26rl_policy_obs_channels names, in that order.
 * @param out    Receives `action slice width` values.
 * @return K26RL_POLICY_OK, or K26RL_POLICY_E_NULL.
 * @note  Deterministic: the result is the mean of the policy's action
 *        distribution, standardised, propagated and clamped exactly as
 *        this header declares. It allocates nothing and performs no
 *        I/O. `obs` and `out` may not overlap, and one loaded policy
 *        is evaluated by one thread at a time.
 */
K26RlPolicyStatus k26rl_policy_act(const K26RlPolicy *policy,
                                   const double *obs, double *out);

/**
 * @brief Evaluate the policy over a whole environment's vectors.
 * @param policy  The policy.
 * @param env_obs `observation total` values, the buffer k26rl_env_obs
 *                filled for one environment.
 * @param env_act `action total` values, the buffer k26rl_env_step will
 *                read for one environment.
 * @return K26RL_POLICY_OK, or K26RL_POLICY_E_NULL.
 * @note  The policy gathers the channels it declares out of `env_obs`
 *        and writes its own action slice; every other channel of
 *        `env_act` is left as the caller had it, so several policies
 *        can fill one action vector for a multi-agent world. The
 *        gather buffer is the policy's own, so this allocates nothing
 *        and one loaded policy is evaluated by one thread at a time.
 */
K26RlPolicyStatus k26rl_policy_act_env(const K26RlPolicy *policy,
                                       const double *env_obs,
                                       double *env_act);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_POLICY_H */
