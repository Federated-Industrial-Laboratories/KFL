"""Gate 15: the telemetry tap, armed from Python.

A training host watches a run while it trains, and it arms the ring
itself rather than asking anything else to. This gate holds the
package to the two properties that makes worth having.

The first is that the ring actually carries the run. The tap is armed
on a two-environment artifact from the package's vector shape, an
independent reader attaches to the named ring, and the frames it
drains satisfy the arithmetic a lossless drain must: the first slot
frame is sequence 1, every later one is one past the last, and the
reader reports no loss. The reader is the record library's own, run in
its own process, so the gate does not check the ring against a
reimplementation of the ring.

The second is that watching changes nothing. A run with the tap armed
and a run without it, same binary, same seed, same action stream,
write byte-identical episode files. Two controls stand behind that
comparison, because a comparison of nothing with nothing passes
without measuring: a third run at a different seed must produce a
different file, and the reader must be shown to have accepted frames,
so the tap cannot pass by never having been armed.

Beside them, the arming call's own refusals reach the package's error
mapping: a name outside the tap's name rule, and a call made when the
handle is not at an episode boundary.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import os
import subprocess
import sys

import _gateutil as g

GATE = "gate15_tap_armed"
SEED = 20260820
OTHER_SEED = 20260821
N_ENVS = 2
T = 24
DRAIN_TIMEOUT = "60"


def act(t, e):
    """A stream both runs feed identically, and one that reaches the
    fixture's world rather than sitting unread in the action buffer."""
    return ((t * 5 + e * 3) % 11) / 11.0 * 2.0 - 1.0


def actions(t):
    import numpy as np

    return np.array([[act(t, e)] for e in range(N_ENVS)],
                    dtype=np.float64)


def drive(so, out_path, seed, tap_name=None, drain=None):
    """One run of the fixture through the vector shape, optionally
    with the ring armed and a reader draining it. Returns the reader's
    completed process, or None where none was asked for."""
    from k26rl.vector import K26RlVectorEnv

    proc = None
    env = K26RlVectorEnv(so, seed=seed, n_envs=N_ENVS)
    try:
        env.set_output(out_path)
        if tap_name is not None:
            env.tap(tap_name)
            proc = subprocess.Popen(
                [str(drain), tap_name, DRAIN_TIMEOUT],
                stdout=subprocess.PIPE, text=True)
            # The reader says when it holds the ring. Stepping before
            # that would let a short run finish and take the ring's
            # name with it before the reader ever saw it.
            line = proc.stdout.readline().strip()
            g.check(line == "attached",
                    "the reader did not report attaching: %r" % line)
        for t in range(T):
            env.step(actions(t))
    finally:
        env.close()
    if proc is not None:
        rest = proc.stdout.read()
        proc.stdout.close()
        proc.wait()
        proc.report = rest
    return proc


def report_values(text):
    """The reader's report as a mapping, one word pair per line."""
    values = {}
    for line in text.splitlines():
        words = line.split()
        for i in range(len(words) - 1):
            if (not words[i].isdigit()
                    and words[i + 1].lstrip("-").isdigit()):
                values[words[i]] = int(words[i + 1])
    return values


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    from k26rl import K26RlError, K26RlOutputError
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("rl_shimterm", g.TERM_KFL)
    drain = g.build_tap_drain()

    tapped = g.WORK / "gate15_tapped.episode"
    untapped = g.WORK / "gate15_untapped.episode"
    other = g.WORK / "gate15_other_seed.episode"
    for f in (tapped, untapped, other):
        if f.exists():
            f.unlink()

    # The name carries this process's identifier so a run cannot be
    # refused by a ring an earlier one left behind, and so two runs on
    # one machine do not collide.
    tap_name = "k26rl_gate15_%d" % os.getpid()

    # ---- the ring carries the run -----------------------------------
    proc = drive(so, str(tapped), SEED, tap_name, drain)
    print(proc.report.strip())
    g.check(proc.returncode == 0,
            "the reader reported an unclean drain (rc=%d)"
            % proc.returncode)
    values = report_values(proc.report)
    g.check(values.get("accepted", 0) > 0,
            "the reader accepted no frames: %r" % values)
    g.check(values.get("lost", 1) == 0,
            "the reader reported lost frames: %r" % values)
    g.check(values.get("first_seq") == 1,
            "the drain did not begin at the first slot frame: %r"
            % values)
    g.check(values.get("last_seq") - values.get("first_seq") + 1
            == values.get("accepted"),
            "the accepted count does not span the sequence range: %r"
            % values)
    g.check(values.get("closed") == 1,
            "the reader never saw the producer close: %r" % values)
    # The run's shape reached the ring, not merely some frames: every
    # environment opened an episode, the fixture terminates inside the
    # run so episodes also ended, and steps were published between.
    g.check(values.get("starts", 0) >= N_ENVS,
            "fewer episode-start frames than environments: %r" % values)
    g.check(values.get("chunks", 0) > 0,
            "no step frames reached the ring: %r" % values)
    g.check(values.get("ends", 0) > 0,
            "no episode-end frames reached the ring: %r" % values)
    print("%s: %d frames drained in sequence, %d starts, %d steps, "
          "%d ends" % (GATE, values["accepted"], values["starts"],
                       values["chunks"], values["ends"]))

    # The reader's own control: pointed at a ring nobody armed, it
    # must fail rather than report a clean drain, or its verdict above
    # would be worth nothing.
    absent = subprocess.run(
        [str(drain), tap_name + "_absent", "1"],
        stdout=subprocess.PIPE, text=True)
    g.check(absent.returncode != 0,
            "the reader reported success against a ring nobody armed")
    print("%s: the reader refuses a ring nobody armed (rc=%d)"
          % (GATE, absent.returncode))

    # ---- watching changes nothing ------------------------------------
    drive(so, str(untapped), SEED)
    drive(so, str(other), OTHER_SEED)
    tapped_bytes = tapped.read_bytes()
    untapped_bytes = untapped.read_bytes()
    other_bytes = other.read_bytes()
    g.check(len(tapped_bytes) > 0, "the tapped run wrote nothing")
    g.check(tapped_bytes == untapped_bytes,
            "the episode file of a tapped run differs from an "
            "untapped one at the same seed and action stream")
    # The control: the same comparison must be able to say `differ`.
    g.check(tapped_bytes != other_bytes,
            "a run at a different seed produced the same episode file, "
            "so the comparison above measures nothing")
    print("%s: tapped and untapped runs wrote %d identical bytes; "
          "another seed differs" % (GATE, len(tapped_bytes)))

    # ---- the arming call's refusals ----------------------------------
    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    try:
        env.tap("has a space")
    except K26RlError as exc:
        g.check(type(exc) is K26RlError,
                "a bad tap name raised %s, expected the base type"
                % type(exc).__name__)
        print("%s: a name outside the rule is refused: %s" % (GATE, exc))
    else:
        g.check(False, "a tap name carrying a space was accepted")

    env.step(actions(0))
    try:
        env.tap(tap_name + "_mid")
    except K26RlOutputError as exc:
        print("%s: arming away from a boundary is refused: %s"
              % (GATE, exc))
    else:
        g.check(False, "the tap was armed in the middle of an episode")
    env.close()

    # A tap name is a string or None; anything else is an argument
    # defect this package catches before the call.
    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    try:
        env.tap(7)
    except TypeError:
        pass
    else:
        g.check(False, "a non-string tap name was accepted")
    env.close()

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
