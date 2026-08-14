"""Gate 3: fault surfacing.

A deterministically induced fault (a division by zero in the
objective, driven by one environment's action value) surfaced through
the package by a consumer that never touches the getters. The
faulted step reports truncated true with the fault_code and
fault_reason info keys for exactly the faulted environment; default
mode raises K26RlFaultError after that stream is formed, carrying the
environment index and registry code with the artifact-decoded reason
and the step's results; truncate mode returns the same stream without
raising. In the vector shape the following step is the boundary reset
(the new episode's initial observation, reward zero, neither ending
bit), legal because the faulted step reported an ending; in the
single shape a following step is refused with the reset-needed error
and reset() proceeds. Construction with an unknown on_fault value is
refused naming the two accepted values.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import ctypes
import sys

import _gateutil as g

GATE = "gate03_fault"
SEED = 5
E_ENV_INTERNAL = 11


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    import gymnasium
    from k26rl import K26RlError, K26RlFaultError
    from k26rl.env import K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_shimfault", g.FAULT_KFL)

    # Independent decode reference: the artifact's own status_str,
    # reached directly, not through the package.
    ref = ctypes.CDLL(str(so))
    ref.k26rl_status_str.restype = ctypes.c_char_p
    ref.k26rl_status_str.argtypes = [ctypes.c_int]
    reason_ref = ref.k26rl_status_str(E_ENV_INTERNAL).decode()

    def benign(n):
        return np.full((n, 1), 0.2, dtype=np.float64)

    # ---- vector shape, default (raise) mode --------------------------
    env = K26RlVectorEnv(so, seed=SEED, n_envs=3)
    initial, _ = env.reset()
    prev_obs, prev_rew, _, _, _ = env.step(benign(3))

    faulting = np.array([[0.2], [1.0], [0.3]])
    try:
        env.step(faulting)
    except K26RlFaultError as exc:
        fault = exc
    else:
        g.check(False, "faulting step did not raise in default mode")

    g.check(fault.indices == (1,), "fault indices %r" % (fault.indices,))
    g.check(fault.codes == (E_ENV_INTERNAL,),
            "fault codes %r" % (fault.codes,))
    g.check(fault.reasons == (reason_ref,),
            "fault reason %r, artifact says %r"
            % (fault.reasons, reason_ref))

    obs, rew, term, trunc, infos = fault.results
    g.check(list(term) == [False, False, False],
            "fault rendered as termination")
    g.check(list(trunc) == [False, True, False],
            "truncation flags %s" % trunc)
    g.check(rew[1] == 0.0, "faulted reward %r" % rew[1])
    g.check(obs[1].tobytes() == prev_obs[1].tobytes(),
            "faulted observation is not the pre-step observation")
    g.check(rew[0] != 0.0 and rew[2] != 0.0,
            "neighbours did not advance")
    g.check(list(infos["_fault_code"]) == [False, True, False],
            "fault_code mask %s" % infos.get("_fault_code"))
    g.check(int(infos["fault_code"][1]) == E_ENV_INTERNAL,
            "info fault_code %r" % infos["fault_code"])
    g.check(infos["fault_reason"][1] == reason_ref,
            "info fault_reason %r" % infos["fault_reason"])

    # The next step is the boundary reset for the faulted environment:
    # a reset step arrives only after a reported ending.
    obs2, rew2, term2, trunc2, infos2 = env.step(benign(3))
    g.check(not term2.any() and not trunc2.any(),
            "boundary step carries ending flags")
    g.check(rew2[1] == 0.0, "boundary reward %r" % rew2[1])
    # The fixture draws nothing at reset, so the new episode's initial
    # observation equals episode 0's, bitwise.
    g.check(obs2[1].tobytes() == initial[1].tobytes(),
            "boundary observation is not the new episode's initial one")
    g.check("fault_code" not in infos2,
            "boundary step still carries fault info")
    env.close()

    # ---- vector shape, truncate mode: same stream, no raise ----------
    env_t = K26RlVectorEnv(so, seed=SEED, n_envs=3,
                           on_fault="truncate")
    env_t.reset()
    obs_a, rew_a, _, _, _ = env_t.step(benign(3))
    g.check(obs_a.tobytes() == prev_obs.tobytes()
            and rew_a.tobytes() == prev_rew.tobytes(),
            "truncate-mode session diverged before the fault")
    obs_b, rew_b, term_b, trunc_b, infos_b = env_t.step(faulting)
    g.check(obs_b.tobytes() == obs.tobytes()
            and rew_b.tobytes() == rew.tobytes(),
            "truncate-mode fault stream differs from raise-mode")
    g.check(list(trunc_b) == [False, True, False]
            and not term_b.any(),
            "truncate-mode flags")
    g.check(int(infos_b["fault_code"][1]) == E_ENV_INTERNAL
            and infos_b["fault_reason"][1] == reason_ref,
            "truncate-mode info keys")
    env_t.close()

    # ---- single shape ------------------------------------------------
    senv = K26RlEnv(so, seed=SEED)
    s_initial, _ = senv.reset()
    senv.step(np.array([0.2]))
    try:
        senv.step(np.array([1.0]))
    except K26RlFaultError as exc:
        g.check(exc.results[3] is True, "single-shape truncation")
        g.check(exc.results[4]["fault_code"] == E_ENV_INTERNAL
                and exc.results[4]["fault_reason"] == reason_ref,
                "single-shape info keys")
    else:
        g.check(False, "single-shape fault did not raise")
    try:
        senv.step(np.array([0.2]))
    except gymnasium.error.ResetNeeded:
        pass
    else:
        g.check(False, "step after a caught fault was not refused")
    s_obs, _ = senv.reset()
    g.check(s_obs.tobytes() == s_initial.tobytes(),
            "single-shape reset after fault")
    senv.step(np.array([0.2]))
    senv.close()

    # ---- on_fault validation -----------------------------------------
    for bad in ("drop", "ignore", ""):
        try:
            K26RlVectorEnv(so, seed=1, n_envs=1, on_fault=bad)
        except ValueError as exc:
            g.check("raise" in str(exc) and "truncate" in str(exc),
                    "on_fault refusal does not name the accepted "
                    "values: %s" % exc)
        else:
            g.check(False, "on_fault=%r accepted" % bad)

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
