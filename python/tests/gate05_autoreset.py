"""Gate 5: autoreset conformance.

The vector shape against gymnasium's reset-on-next-step autoreset
convention: the step that ends an episode delivers the final
observation and the terminal-adjusted reward with its ending bit; the
following step delivers the new episode's initial observation with
reward zero and neither bit. The single shape refuses a step after an
ending with gymnasium's reset-needed error. Termination is exercised
on a fixture whose episodes terminate at transition 3 with terminal
adjustment 5.0 on a step reward of 1.0; truncation on a fixture whose
horizon truncates at transition 6. Neither fixture draws at reset, so
every episode's initial observation is bitwise equal to episode 0's,
which pins the boundary observation exactly.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import sys

import _gateutil as g

GATE = "gate05_autoreset"


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    import gymnasium
    from k26rl.env import K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    term_so = g.compile_fixture("rl_shimterm", g.TERM_KFL)
    trunc_so = g.compile_fixture("rl_shimfault", g.FAULT_KFL)

    def zeros(n):
        return np.zeros((n, 1), dtype=np.float64)

    # ---- vector shape, termination with terminal adjustment ----------
    env = K26RlVectorEnv(term_so, seed=3, n_envs=2)
    initial, _ = env.reset()
    for step in (1, 2):
        _, rew, term, trunc, _ = env.step(zeros(2))
        g.check(not term.any() and not trunc.any(),
                "step %d carries an ending bit" % step)
        g.check(list(rew) == [1.0, 1.0], "step %d reward %s"
                % (step, rew))
    final_obs, rew, term, trunc, _ = env.step(zeros(2))
    g.check(list(term) == [True, True], "termination bits %s" % term)
    g.check(not trunc.any(), "termination also truncated")
    g.check(list(rew) == [6.0, 6.0],
            "terminal-adjusted reward %s, expected 1.0 + 5.0" % rew)
    b_obs, b_rew, b_term, b_trunc, _ = env.step(zeros(2))
    g.check(not b_term.any() and not b_trunc.any(),
            "boundary step carries ending bits")
    g.check(list(b_rew) == [0.0, 0.0], "boundary reward %s" % b_rew)
    g.check(b_obs.tobytes() == initial.tobytes(),
            "boundary observation is not the new episode's initial one")
    g.check(final_obs.tobytes() != b_obs.tobytes(),
            "final observation equals the boundary observation; the "
            "ending step did not deliver the final observation")
    _, rew5, term5, trunc5, _ = env.step(zeros(2))
    g.check(list(rew5) == [1.0, 1.0] and not term5.any()
            and not trunc5.any(),
            "the new episode does not step normally")
    env.close()

    # ---- vector shape, truncation at the horizon ---------------------
    env = K26RlVectorEnv(trunc_so, seed=4, n_envs=2)
    initial, _ = env.reset()
    for step in range(1, 6):
        _, _, term, trunc, _ = env.step(zeros(2))
        g.check(not term.any() and not trunc.any(),
                "truncation fixture ended early at step %d" % step)
    _, rew, term, trunc, _ = env.step(zeros(2))
    g.check(list(trunc) == [True, True], "truncation bits %s" % trunc)
    g.check(not term.any(), "truncation also terminated")
    g.check(list(rew) == [1.0, 1.0],
            "truncated final reward %s carries an adjustment" % rew)
    b_obs, b_rew, b_term, b_trunc, _ = env.step(zeros(2))
    g.check(not b_term.any() and not b_trunc.any()
            and list(b_rew) == [0.0, 0.0]
            and b_obs.tobytes() == initial.tobytes(),
            "truncation boundary step")
    env.close()

    # ---- single shape refuses step-after-end -------------------------
    for so, endings in ((term_so, 3), (trunc_so, 6)):
        senv = K26RlEnv(so, seed=9)
        s_initial, _ = senv.reset()
        for _ in range(endings - 1):
            _, _, term, trunc, _ = senv.step(np.zeros(1))
            g.check(not term and not trunc, "single shape ended early")
        _, _, term, trunc, _ = senv.step(np.zeros(1))
        g.check(term or trunc, "single shape did not end")
        try:
            senv.step(np.zeros(1))
        except gymnasium.error.ResetNeeded:
            pass
        else:
            g.check(False, "step after an ending was not refused")
        s_obs, _ = senv.reset()
        g.check(s_obs.tobytes() == s_initial.tobytes(),
                "single-shape reset after an ending")
        _, _, term, trunc, _ = senv.step(np.zeros(1))
        g.check(not term and not trunc,
                "the new single-shape episode does not step")
        senv.close()

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
