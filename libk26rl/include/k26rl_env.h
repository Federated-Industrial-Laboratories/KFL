/* k26rl_env.h - the external stepping surface of a compiled simulation.
 *
 * A KFL program that declares reinforcement-learning constructs
 * compiles, in one emission, to its ordinary batch executable and a
 * companion shared object exporting exactly this k26rl_ surface. The
 * shared object is the world's single implementation; batch mode
 * drives the same code. A consumer dlopens the shared object, checks
 * k26rl_abi_version (major must match, minor at least), and steps
 * every environment in a handle through flat, fixed-width,
 * spec-described buffers.
 *
 * Version 1.0. This surface is frozen: any change to an existing
 * symbol's signature or semantics, or to an existing spec tag's
 * payload, increments the major version; additions land as new
 * symbols, new tags, and new status codes under a minor increment,
 * and consumers probe symbols with dlsym and skip unknown tags.
 *
 * Determinism contract: given the same artifact, the same seed, and
 * the same action stream, the observation, reward, and flag streams
 * are bit-identical across runs, and an environment's streams are
 * unchanged by the presence, count, or activity of its neighbours in
 * the handle, with one bounded exception noted at k26rl_env_step.
 *
 * Threading: a handle is single-threaded. Concurrent handles in one
 * process are legal; the runtime's FPU mode-conflict detection
 * applies.
 *
 * The stepping hot loop performs no allocation and no I/O: the
 * per-step path touches preallocated memory only, and when episode
 * output is enabled a bounded buffered flush occurs once per chunk
 * boundary and at episode ends. With output disabled the step path
 * performs no I/O of any kind. */
#ifndef K26RL_ENV_H
#define K26RL_ENV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Major in the high 16 bits, minor in the low 16. */
#define K26RL_ABI_VERSION ((uint32_t)0x00010000u)

/* One handle owns n_envs worlds; layout is private to the artifact. */
typedef struct K26RlEnv K26RlEnv;

/* Status registry. Append-only: new codes are added under the minor
 * version rule, values are never renumbered, and every value fits in
 * 16 bits so the two uint16 fault carriers (k26rl_env_fault_codes and
 * the episode file's episode-end fault code) can hold any of them.
 *
 * Fault reasons are the per-environment codes an episode can end
 * with: K26RL_E_DIVERGED, K26RL_E_ENV_INTERNAL, and
 * K26RL_E_RNG_EXHAUSTED when the exhaustion is met inside a
 * transition. K26RL_E_INTERNAL is handle-level only and never appears
 * as a fault reason; K26RL_E_DIVERGED and K26RL_E_ENV_INTERNAL never
 * appear as a call status; K26RL_E_RNG_EXHAUSTED is the registry's
 * one dual-use value, a fault reason inside a transition and a call
 * status when a reset's draws exhaust. */
typedef enum {
    K26RL_OK                 = 0,
    K26RL_E_NULL             = 1,   /* null pointer argument */
    K26RL_E_GEOMETRY         = 2,   /* buffer geometry disagrees with spec */
    K26RL_E_RNG_EXHAUSTED    = 3,   /* draw coordinates exhausted */
    K26RL_E_SEED_REUSE       = 4,   /* seed already used by this handle */
    K26RL_E_OUTPUT_TIMING    = 5,   /* output call not at an episode boundary */
    K26RL_E_OUTPUT_EXISTS    = 6,   /* output path already exists */
    K26RL_E_FPU_RACE         = 7,   /* FPU mode conflict between live worlds */
    K26RL_E_USE_AFTER_DESTROY = 8,  /* handle already destroyed */
    K26RL_E_INTERNAL         = 9,   /* handle-level internal failure */
    K26RL_E_DIVERGED         = 10,  /* fault reason: integrator divergence */
    K26RL_E_ENV_INTERNAL     = 11   /* fault reason: internal error confined
                                     * to one environment's step */
} K26RlStatus;

/* Flag word bits, shared with the episode file's recorded flag words.
 * Bit 3 appears only in this ABI's flags word: the reset boundary is
 * the episode file's episode-start frame, never a step record.
 * Remaining bits are reserved zero. */
#define K26RL_FLAG_TERMINATED     ((uint32_t)1u << 0)
#define K26RL_FLAG_TRUNCATED      ((uint32_t)1u << 1)
#define K26RL_FLAG_FAULT          ((uint32_t)1u << 2)
#define K26RL_FLAG_RESET_BOUNDARY ((uint32_t)1u << 3)

/* env_spec TLV tags. The blob is a sequence of (tag uint16, length
 * uint32, value) triples, little-endian, no alignment padding,
 * terminated by the blob length the call returned; unknown tags are
 * skipped by length. Append-only: tags are never renumbered or
 * reassigned. Multiple slice, bounds, kind, and name tags appear, one
 * per agent or channel. */
#define K26RL_TAG_ABI_VERSION       ((uint16_t)0x0001) /* uint32, equals k26rl_abi_version() */
#define K26RL_TAG_ENDIAN_PROBE      ((uint16_t)0x0002) /* uint32 0x01020304 */
#define K26RL_TAG_AGENT_COUNT       ((uint16_t)0x0003) /* uint32 */
#define K26RL_TAG_N_ENVS            ((uint16_t)0x0004) /* uint32, restates create */
#define K26RL_TAG_CONTROL_DT        ((uint16_t)0x0005) /* binary64 bits, simulated s/step */
#define K26RL_TAG_HORIZON           ((uint16_t)0x0006) /* uint32 max transitions, 0 unbounded */
#define K26RL_TAG_OBS_TOTAL         ((uint16_t)0x0007) /* uint32 doubles per environment */
#define K26RL_TAG_ACT_TOTAL         ((uint16_t)0x0008) /* uint32 doubles per environment */
#define K26RL_TAG_AGENT_OBS_SLICE   ((uint16_t)0x0009) /* agent u32, offset u32, count u32 */
#define K26RL_TAG_AGENT_ACT_SLICE   ((uint16_t)0x000A) /* agent u32, offset u32, count u32 */
#define K26RL_TAG_ACT_BOUNDS        ((uint16_t)0x000B) /* offset u32, lo bits u64, hi bits u64 */
#define K26RL_TAG_ACT_KIND          ((uint16_t)0x000C) /* offset u32, kind u16, arity u32 */
#define K26RL_TAG_OBS_CHANNEL_NAME  ((uint16_t)0x000D) /* channel u32, UTF-8 name */
#define K26RL_TAG_OBS_CHANNEL_KIND  ((uint16_t)0x000E) /* channel u32, kind u16 */
#define K26RL_TAG_EPISODE_FLAGS     ((uint16_t)0x000F) /* uint32; bit 0 auto-reset */
#define K26RL_TAG_REWARD_COMPONENTS ((uint16_t)0x0010) /* reserved */

/* Action kinds for K26RL_TAG_ACT_KIND. */
#define K26RL_ACT_KIND_BOX      ((uint16_t)0)
#define K26RL_ACT_KIND_DISCRETE ((uint16_t)1)

/* Observation channel kinds for K26RL_TAG_OBS_CHANNEL_KIND. */
#define K26RL_OBS_KIND_VECTOR ((uint16_t)0)
/* kind 1 is reserved for image channels. */

/* The surface. All integers are stdint.h fixed-width; all reals are
 * IEEE-754 binary64. Callable before any create: k26rl_abi_version
 * and k26rl_status_str. A status is the caller's to check.
 *
 * k26rl_env_create completes the initial reset of every environment
 * before it returns: episode indices are 0, initial-state and
 * randomisation draws are taken at episode 0 coordinates, the getters
 * are valid immediately, and the first step steps normally.
 *
 * k26rl_env_step advances every environment in the handle by exactly
 * the spec's control-dt of simulated time, except an environment
 * performing its boundary reset in that call and an environment whose
 * step faults, both advancing none. After an environment's episode
 * ends, its next step performs its reset instead of stepping: action
 * slice ignored, episode index incremented, draws at the new episode
 * coordinates, observation slice carrying the new episode's initial
 * observation, reward zero, flag word carrying only the
 * reset-boundary bit. Environments reset independently; the others
 * step normally in the same call. The step call's own status is
 * independent of per-environment faults: it returns K26RL_OK whenever
 * the call itself completed, however many environments faulted, and
 * nonzero step statuses are reserved for handle-level failures. A
 * fault is carried by flag bit 2, k26rl_env_fault_codes, and the
 * episode file's episode-end frame; a faulting step completes no
 * transition, so the getters deliver the pre-step observations with
 * zero rewards and the fault bit set. A boundary reset whose draw
 * coordinates would exhaust refuses the whole call with
 * K26RL_E_RNG_EXHAUSTED before anything advances, nothing stepped and
 * nothing emitted; that refusal is the one way a neighbour's
 * condition stops another environment's stream, and it changes no
 * value in any stream.
 *
 * k26rl_env_reset increments every environment's episode index and
 * resets them all; it re-reads nothing from the outside world.
 * k26rl_env_reset_seeded replaces the key and zeroes all episode
 * indices; the handle remembers every seed it has ever held, the
 * create seed included, and refuses any of them with
 * K26RL_E_SEED_REUSE. A reset cannot fault: a reset that cannot
 * complete is a handle-level error returned as the call's status,
 * with no frame emitted and no episode consumed.
 *
 * k26rl_env_output enables episode-file emission to path, or disables
 * it when path is null. Callable only when every environment sits at
 * an episode boundary: after create, immediately after a reset call,
 * and before the first step that follows; any other timing is refused
 * with K26RL_E_OUTPUT_TIMING. A path that already exists is refused
 * with K26RL_E_OUTPUT_EXISTS. Emission is off until enabled.
 *
 * Buffer geometry is spec-driven and env-major: actions is
 * n_envs * act_total doubles, out for observations
 * n_envs * obs_total, rewards n_envs * agent_count, flags one uint32
 * per environment, fault codes one uint16 per environment. The
 * artifact never allocates or frees caller memory. Getters copy from
 * internal state written by the latest step or reset; they are pure
 * reads, callable repeatedly. k26rl_env_fault_codes writes zero
 * unless that environment's current episode ended by fault, in which
 * case it is the fault reason from the status registry.
 *
 * k26rl_env_spec returns the required byte count as a positive value;
 * when capacity is at least that it writes the blob, and when
 * capacity is smaller it writes nothing and still returns the
 * requirement, so capacity 0 sizes it. On error it returns the
 * negated K26RlStatus. The blob never changes after create. */
uint32_t     k26rl_abi_version(void);
K26RlStatus  k26rl_env_create(uint64_t seed, uint32_t n_envs,
                              K26RlEnv **out_env);
K26RlStatus  k26rl_env_output(K26RlEnv *env, const char *path);
K26RlStatus  k26rl_env_reset(K26RlEnv *env);
K26RlStatus  k26rl_env_reset_seeded(K26RlEnv *env, uint64_t seed);
K26RlStatus  k26rl_env_step(K26RlEnv *env, const double *actions);
K26RlStatus  k26rl_env_obs(const K26RlEnv *env, double *out);
K26RlStatus  k26rl_env_reward(const K26RlEnv *env, double *out);
K26RlStatus  k26rl_env_flags(const K26RlEnv *env, uint32_t *out);
K26RlStatus  k26rl_env_fault_codes(const K26RlEnv *env, uint16_t *out);
int32_t      k26rl_env_spec(const K26RlEnv *env, uint8_t *out,
                            uint32_t capacity);
const char  *k26rl_status_str(K26RlStatus status);
void         k26rl_env_destroy(K26RlEnv *env);

#ifdef __cplusplus
}
#endif

#endif /* K26RL_ENV_H */
