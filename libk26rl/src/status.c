/* status.c - status code to string decoding.
 *
 * Callable before any create, so it depends on nothing but the value.
 * Unknown values decode to one stable string rather than failing:
 * the registry is append-only, so a newer producer's code reaching an
 * older consumer must still print something definite. */
#include "k26rl_env.h"

const char *k26rl_status_str(K26RlStatus status)
{
    switch (status) {
    case K26RL_OK:                  return "ok";
    case K26RL_E_NULL:              return "null pointer argument";
    case K26RL_E_GEOMETRY:          return "buffer geometry disagrees with spec";
    case K26RL_E_RNG_EXHAUSTED:     return "draw coordinates exhausted";
    case K26RL_E_SEED_REUSE:        return "seed already used by this handle";
    case K26RL_E_OUTPUT_TIMING:     return "output call not at an episode boundary";
    case K26RL_E_OUTPUT_EXISTS:     return "output path already exists";
    case K26RL_E_FPU_RACE:          return "FPU mode conflict between live worlds";
    case K26RL_E_USE_AFTER_DESTROY: return "handle already destroyed";
    case K26RL_E_INTERNAL:          return "handle-level internal failure";
    case K26RL_E_DIVERGED:          return "integrator divergence";
    case K26RL_E_ENV_INTERNAL:      return "internal error confined to one environment";
    }
    return "unknown status";
}
