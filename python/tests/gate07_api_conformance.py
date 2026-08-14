"""Gate 7: API conformance.

Gymnasium's environment checker over the single shape of the
validation fixture: reset-seed repetition, space containment, the
five-tuple, and the reset signature all pass the ecosystem's own
check. The checker applies to gymnasium.Env only, so the vector
shape gets a direct conformance pass here: batched spaces, batched
reset and step shapes and dtypes, and containment.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import sys
import warnings

import _gateutil as g

GATE = "gate07_api_conformance"


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    from gymnasium.utils.env_checker import check_env
    from k26rl.env import K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_pointing")

    env = K26RlEnv(so, seed=31)
    # The checker's warnings (an unregistered spec, unbounded
    # observation space) are advisory; a failed check raises.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env)
    env.close()
    print("gymnasium check_env passed on the single shape")

    # ---- vector conformance ------------------------------------------
    n = 3
    venv = K26RlVectorEnv(so, seed=32, n_envs=n)
    g.check(venv.num_envs == n, "num_envs")
    obs, info = venv.reset(seed=33)
    g.check(obs.shape == (n, venv.env_spec.obs_total)
            and obs.dtype == np.float64,
            "vector reset observation shape %s dtype %s"
            % (obs.shape, obs.dtype))
    g.check(obs in venv.observation_space,
            "vector reset observation is outside the batched space")
    g.check(isinstance(info, dict), "vector reset info")
    venv.action_space.seed(34)
    for _ in range(3):
        actions = venv.action_space.sample()
        obs, rew, term, trunc, infos = venv.step(actions)
        g.check(obs in venv.observation_space,
                "vector step observation is outside the batched space")
        g.check(rew.shape == (n,) and rew.dtype == np.float64,
                "vector reward shape %s dtype %s"
                % (rew.shape, rew.dtype))
        g.check(term.shape == (n,) and term.dtype == bool
                and trunc.shape == (n,) and trunc.dtype == bool,
                "vector ending flag arrays")
        g.check(isinstance(infos, dict), "vector step info")
    venv.close()

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
