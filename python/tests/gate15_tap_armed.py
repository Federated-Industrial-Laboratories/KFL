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

The third is that an armed ring is not taken away in silence. A
seeded reset to a seed the environment has already held needs a fresh
handle, and destroying the old one closes the ring and removes its
name, so that reset is refused while a tap is armed, exactly as it is
while episode output is enabled. The arm drives the refusal, goes on
stepping past it, and then counts the drained frames against the
fixture's own arithmetic, so the frames published before the refused
reset are shown to have survived it and publication is shown to have
continued after it. Disarming then lets the same reset through, and
the same ring name arms again and is drained a second time.

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
# Driver steps per episode of the fixture: three recorded transitions
# and the boundary step that follows them.
EPISODE_STEPS = 4
# The refused-reset arm's two stretches, before and after the refusal.
# They sum to a whole number of episodes, so no environment is
# already ended when the explicit reset that follows them arrives and
# the frames that reset publishes are one end and one start each.
T_BEFORE = 5
T_AFTER = 7
# Steps driven through the second arming of the same ring name.
T_REARM = 4
DRAIN_TIMEOUT = "60"


def act(t, e):
    """A stream both runs feed identically, and one that reaches the
    fixture's world rather than sitting unread in the action buffer."""
    return ((t * 5 + e * 3) % 11) / 11.0 * 2.0 - 1.0


def expected_frames(steps, resets=0):
    """What the fixture publishes over ``steps`` driver steps from a
    fresh arming, by the fixture's own arithmetic rather than by
    whatever the run happened to produce.

    An episode of this fixture is three recorded transitions; the
    fourth driver step records no transition and instead ends that
    episode and opens the next, so a cycle is four driver steps
    carrying three step frames, one end and one start. Arming itself
    publishes one start per environment.

    ``resets`` counts explicit reset calls, each of which cuts the
    open episode with a truncated end and opens the next with a
    start, one of each per environment. That is the count only where
    no environment was already ended when the reset arrived, which is
    why the stretches this gate drives are whole episodes."""
    boundaries = steps // EPISODE_STEPS
    return {
        "starts": N_ENVS * (1 + boundaries + resets),
        "chunks": N_ENVS * (steps - boundaries),
        "ends": N_ENVS * (boundaries + resets),
    }


def check_drain(proc, steps, what, resets=0):
    """One reader's report: a clean drain, and exactly the frames the
    fixture's arithmetic says that many steps publish."""
    print(proc.report.strip())
    g.check(proc.returncode == 0,
            "%s: the reader reported an unclean drain (rc=%d)"
            % (what, proc.returncode))
    values = report_values(proc.report)
    g.check(values.get("lost", 1) == 0,
            "%s: the reader reported lost frames: %r" % (what, values))
    g.check(values.get("first_seq") == 1,
            "%s: the drain did not begin at the first slot frame: %r"
            % (what, values))
    g.check(values.get("closed") == 1,
            "%s: the reader never saw the producer close: %r"
            % (what, values))
    want = expected_frames(steps, resets)
    for kind, count in sorted(want.items()):
        g.check(values.get(kind) == count,
                "%s: %d %s frames reached the ring, expected %d over "
                "%d steps: %r" % (what, values.get(kind), kind, count,
                                  steps, values))
    total = sum(want.values())
    g.check(values.get("accepted") == total,
            "%s: %r frames accepted, expected %d"
            % (what, values.get("accepted"), total))
    g.check(values.get("last_seq") - values.get("first_seq") + 1
            == values.get("accepted"),
            "%s: the accepted count does not span the sequence range: "
            "%r" % (what, values))
    return values


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
    values = check_drain(proc, T, "the tapped run")
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

    # ---- an armed ring is not taken away in silence ------------------
    #
    # reset(seed=S) to a seed this environment has already held needs
    # a fresh handle, and destroying the old one closes the ring and
    # removes its name. That reset is refused while a tap is armed,
    # as it is while episode output is enabled.
    live_name = tap_name + "_live"
    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    env.tap(live_name)
    g.check(env.tap_name == live_name,
            "the armed ring's name is not recorded: %r" % (env.tap_name,))
    reader = subprocess.Popen([str(drain), live_name, DRAIN_TIMEOUT],
                              stdout=subprocess.PIPE, text=True)
    g.check(reader.stdout.readline().strip() == "attached",
            "the reader did not report attaching to the live ring")
    for t in range(T_BEFORE):
        env.step(actions(t))
    # The construction seed is one this environment has held, and the
    # handle has been stepped, so this reset takes the recreate route.
    try:
        env.reset(seed=SEED)
    except K26RlError as exc:
        text = str(exc)
        g.check("tap" in text and "fresh handle" in text,
                "the refusal does not say the tap is why: %s" % text)
        print("%s: a seeded reset needing a fresh handle is refused "
              "while the tap is armed: %s" % (GATE, text))
    else:
        g.check(False, "a seeded reset recreated the handle and took "
                "the armed ring with it")
    g.check(env.tap_name == live_name,
            "the refused reset changed the armed ring's name: %r"
            % (env.tap_name,))
    # Publication continued past the refusal, and what was published
    # before it is still in the ring: the count below covers both
    # stretches.
    for t in range(T_AFTER):
        env.step(actions(T_BEFORE + t))
    # Disarming follows the arming call's own timing rule, so it needs
    # a boundary, and the only boundary a stepped handle reaches is
    # the one an explicit reset makes. That reset cuts each open
    # episode and opens the next, which the count below accounts for.
    env.reset()
    env.tap(None)
    g.check(env.tap_name is None,
            "disarming did not clear the recorded ring name: %r"
            % (env.tap_name,))
    reader.report = reader.stdout.read()
    reader.stdout.close()
    reader.wait()
    check_drain(reader, T_BEFORE + T_AFTER, "the ring kept past a refusal",
                resets=1)
    print("%s: the frames of %d steps before a refused reset and %d "
          "after it all reached the ring" % (GATE, T_BEFORE, T_AFTER))

    # Disarmed, the same reset goes through, and the name the disarm
    # released arms again and carries a second run.
    env.reset(seed=SEED)
    env.tap(live_name)
    rearmed = subprocess.Popen([str(drain), live_name, DRAIN_TIMEOUT],
                               stdout=subprocess.PIPE, text=True)
    g.check(rearmed.stdout.readline().strip() == "attached",
            "the reader did not report attaching to the re-armed ring")
    for t in range(T_REARM):
        env.step(actions(t))
    env.close()
    rearmed.report = rearmed.stdout.read()
    rearmed.stdout.close()
    rearmed.wait()
    check_drain(rearmed, T_REARM, "the re-armed ring")
    print("%s: disarming let the same reset through, and the released "
          "name armed again" % GATE)

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
