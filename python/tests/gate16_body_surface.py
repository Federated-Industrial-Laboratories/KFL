"""Gate 16: the world behind the observation channels.

A training host reads the bodies to draw the scene it is watching and
the actuators to see what its policy actually commanded. Both are
readbacks of the same world the observation channels are views of, so
the gate that matters is not that they return plausible numbers but
that they return the same world.

The arms:

* the truth pin, bitwise. With the reference set to a geometric
  observe's observer, the target body's reported vector reproduces
  that observe's own range and direction channels exactly, in the
  runtime's own expressions and its own order. A getter reading a
  different world, a stale world, or the right world in the wrong
  frame fails here.
* shape and layout against a hand-computed fixture. The fixture
  declares three bodies with a distinct value on every axis and draws
  nothing at reset, so at construction every one of the eighteen
  numbers per environment is known in advance; a transposed layout, a
  wrong stride, or positions and velocities exchanged fails on the
  values themselves rather than on the shape alone.
* the refusals. A reference naming no body is the getter's own,
  surfaced as the mapped exception with the artifact's own decode; an
  unresolvable body name is caught before the call and names what it
  could not resolve; the world-origin constant is accepted.
* the actuator readback against the same kind of hand-computed
  fixture: descriptors in the getter's declared order with the body
  each binds, commands clamped as the actuator library clamps them,
  each environment's record at its own slot, and a program declaring
  no actuator at all reporting a well-formed empty set rather than
  refusing or returning something of another shape.

Skips (77) when the built compiler, the stack archives, or gymnasium
are absent.
"""

import sys

import _gateutil as g

GATE = "gate16_body_surface"
SEED = 20260820
N_ENVS = 2

# Three bodies, every declared component distinct and no reset draw,
# so the state at construction is known in advance to the bit. The
# observe is geometric: no light-time correction, so the vector the
# observation path measures is exactly the position subtraction the
# getter reports.
BODIES_KFL = """\
form RL_G16BOD
fn world w
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 pos_y=1.1e5 pos_z=-2.3e5 vel_x=3.5 vel_y=7350.0 vel_z=-11.25
    astro_body probe gm=1.0 parent=earth pos_x=-6.4e6 pos_y=3.7e5 pos_z=1.9e5 vel_x=-2.75 vel_y=-7100.0 vel_z=5.5
    episode
        control_dt 0.1
        horizon 20
    end
    action push box -1.0 1.0 default 0.0
    on_step
        craft.vel_x = craft.vel_x + push * 0.01
    end
    observe craft from earth mode=geometric as trk
    objective
        reward trk_range * 1.0e-7 + push
    end
end
end
"""

# The declared state, in declaration order, as the getter lays it out:
# three position components then three velocity components.
DECLARED = (
    ("earth", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("craft", (7.0e6, 1.1e5, -2.3e5, 3.5, 7350.0, -11.25)),
    ("probe", (-6.4e6, 3.7e5, 1.9e5, -2.75, -7100.0, 5.5)),
)

# A wheel about the third axis and a thruster mounted off-centre, so
# each actuator's descriptor is distinct in every field that could be
# confused with another's.
ACT_ASM = """\
assembly gate16_box
    frame x_to_port
    provenance mass "gate fixture, not a craft" computed
    component hull
        mass 1000.0
        at 0 0 0
        collider box 1.0 0.5 0.5
    end
    wheel yaw
        axis 0.0 0.0 1.0
        spin_inertia 0.05
        max_momentum 15.0
        max_torque 0.20
    end
    thruster rcs_py
        at 1.05 0.92 0.0
        dir 0.0 -1.0 0.0
        thrust 400.0
    end
end
"""

ACT_KFL = """\
form RL_G16ACT
fn world w
    astro_body earth gm=3.986004418e14 mass=5.972e24
    astro_body craft assembly="%s/gate16_box.k26asm" parent=earth pos_x=7.0e6 pos_y=0.0 pos_z=0.0 vel_x=0.0 vel_y=7546.0 vel_z=0.0 quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0 omega_x=0.0 omega_y=0.0 omega_z=0.0
    episode
        control_dt 0.5
        horizon 8
        substeps 4
    end
    action spin box -1.0 1.0 default 0.0
    action push box 0.0 1.0 default 0.0
    on_step
        craft.yaw.torque = spin * 0.2
        craft.rcs_py.throttle = push
    end
    observe attitude of craft as att
    objective
        reward att_omega_z
    end
end
end
"""

# Actuator kinds, as the getter's second value reports them.
KIND_WHEEL = 0.0
KIND_THRUSTER = 2.0


def main():
    g.require_stack(GATE)
    g.require_gymnasium(GATE)

    import numpy as np
    from k26rl import BODY_REF_ORIGIN, K26RlError
    from k26rl.env import K26RlEnv
    from k26rl.vector import K26RlVectorEnv

    so = g.compile_fixture("gate16_bodies", BODIES_KFL)

    # ---- shape and layout, against the declared state ----------------
    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    names = env.body_names
    g.check(names == {i: n for i, (n, _) in enumerate(DECLARED)},
            "declared body names %r" % (names,))

    state = env.bodies(BODY_REF_ORIGIN)
    g.check(state.shape == (N_ENVS, len(DECLARED), 6),
            "body array shape %r, expected %r"
            % (state.shape, (N_ENVS, len(DECLARED), 6)))
    g.check(state.dtype == np.float64,
            "body array dtype %r, expected float64" % (state.dtype,))
    for e in range(N_ENVS):
        for b, (name, want) in enumerate(DECLARED):
            got = tuple(state[e, b])
            g.check(got == want,
                    "environment %d body %s reads %r, declared %r"
                    % (e, name, got, want))
    print("%s: %d bodies of %d environments read their declared state"
          % (GATE, len(DECLARED), N_ENVS))

    # A body reports itself at the origin of its own frame, which is
    # the cheapest check that the frame is the one claimed, and the
    # reference shifts every other body by the whole offset.
    rel = env.bodies("craft")
    g.check(tuple(rel[0, 1, 0:3]) == (0.0, 0.0, 0.0),
            "the reference body is not at the origin of its own frame: "
            "%r" % (rel[0, 1, 0:3],))
    g.check(tuple(rel[0, 0, 0:3]) ==
            tuple(-np.asarray(DECLARED[1][1][0:3])),
            "the reference did not shift the other bodies: %r"
            % (rel[0, 0, 0:3],))
    # Velocities are the bodies' own and the reference does not touch
    # them, which is the half of the contract a shifting getter would
    # break silently.
    g.check((rel[:, :, 3:6] == state[:, :, 3:6]).all(),
            "the reference changed the reported velocities")
    # An index and the declared name reach the same body.
    g.check((env.bodies(1) == rel).all(),
            "a body name and its index disagree")

    # ---- the refusals -------------------------------------------------
    try:
        env.bodies(len(DECLARED))
    except K26RlError as exc:
        g.check(type(exc) is K26RlError,
                "a reference naming no body raised %s, expected the "
                "base type" % type(exc).__name__)
        g.check(exc.status is not None,
                "the refusal carries no artifact status: %r" % exc)
        print("%s: a reference naming no body is refused: %s"
              % (GATE, exc))
    else:
        g.check(False, "a reference past the last body was accepted")

    try:
        env.bodies("nothing_is_named_this")
    except ValueError as exc:
        g.check("nothing_is_named_this" in str(exc),
                "the unresolvable-name refusal does not name the "
                "reference: %s" % exc)
        for name, _ in DECLARED:
            g.check(name in str(exc),
                    "the unresolvable-name refusal does not say what "
                    "the artifact declares: %s" % exc)
    else:
        g.check(False, "a name naming no body was accepted")

    for bad in (None, 1.5, [1]):
        try:
            env.bodies(bad)
        except TypeError:
            pass
        else:
            g.check(False, "a reference of type %s was accepted"
                    % type(bad).__name__)
    env.close()

    # ---- the truth pin, bitwise ---------------------------------------
    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    channel = {name: i for i, name in env.obs_channel_names.items()}
    for want in ("trk_dir_x", "trk_dir_y", "trk_dir_z", "trk_range"):
        g.check(want in channel, "the fixture publishes no %s" % want)
    # The returned observation carries the measured channels; with no
    # sensor declared that is every channel, and the mapping from
    # declared channel to returned component is the shape's own.
    column = {c: i for i, c in enumerate(env.policy_channels)}
    dir_x = column[channel["trk_dir_x"]]
    dir_y = column[channel["trk_dir_y"]]
    dir_z = column[channel["trk_dir_z"]]
    rng = column[channel["trk_range"]]

    compared = 0
    seen = set()
    for t in range(12):
        actions = np.array([[((t * 3 + e * 5) % 7) / 7.0 - 0.5]
                            for e in range(N_ENVS)], dtype=np.float64)
        obs, _, _, _, _ = env.step(actions)
        # The reference is the geometric observe's observer, so the
        # target's reported vector is the very vector the observation
        # path measured.
        state = env.bodies("earth")
        for e in range(N_ENVS):
            r = state[e, 1]
            # The runtime's own expressions, in its own order: the
            # magnitude as a sum of squares under one square root, and
            # the direction as the vector times the reciprocal of that
            # magnitude, never divided.
            magnitude = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            inv = 1.0 / magnitude
            for value, got in ((magnitude, obs[e][rng]),
                               (r[0] * inv, obs[e][dir_x]),
                               (r[1] * inv, obs[e][dir_y]),
                               (r[2] * inv, obs[e][dir_z])):
                g.check(np.float64(value).tobytes()
                        == np.float64(got).tobytes(),
                        "step %d environment %d: the getter's vector "
                        "gives %r where the observation channel reads "
                        "%r" % (t, e, value, got))
            seen.add(float(magnitude))
            compared += 1
    g.check(compared == 12 * N_ENVS,
            "compared %d readings, expected %d" % (compared, 12 * N_ENVS))
    # The compared quantity moved through the run and differs between
    # the two environments, so this is a comparison and not a
    # restatement of one constant.
    g.check(len(seen) == compared,
            "the compared range repeated a value: %d distinct of %d"
            % (len(seen), compared))
    print("%s: %d readings where the getter's vector reproduces the "
          "observed range and direction bitwise" % (GATE, compared))
    env.close()

    # ---- the same surface on the single shape -------------------------
    single = K26RlEnv(so, seed=SEED)
    one = single.bodies(BODY_REF_ORIGIN)
    g.check(one.shape == (1, len(DECLARED), 6),
            "the single shape's body array shape %r" % (one.shape,))
    vec_one = K26RlVectorEnv(so, seed=SEED, n_envs=1)
    g.check((one == vec_one.bodies(BODY_REF_ORIGIN)).all(),
            "the two shapes report different bodies for one handle")
    single.close()
    vec_one.close()

    # ---- the actuator readback ----------------------------------------
    act_so = g.compile_fixture(
        "gate16_act", ACT_KFL % (g.WORK,),
        {"gate16_box.k26asm": ACT_ASM})
    env = K26RlVectorEnv(act_so, seed=SEED, n_envs=N_ENVS)
    craft = 1.0

    # The descriptors, present before anything has been commanded, in
    # the getter's declared order: every wheel, then every thruster.
    wheel = (craft, KIND_WHEEL, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    thruster = (craft, KIND_THRUSTER, 1.05, 0.92, 0.0, 0.0, -1.0, 0.0)
    before = env.actuators()
    g.check(before.shape == (N_ENVS, 2, 10),
            "actuator array shape %r, expected %r"
            % (before.shape, (N_ENVS, 2, 10)))
    g.check(before.dtype == np.float64,
            "actuator array dtype %r, expected float64"
            % (before.dtype,))
    for e in range(N_ENVS):
        g.check(tuple(before[e, 0, 0:8]) == wheel,
                "environment %d wheel descriptor %r, expected %r"
                % (e, tuple(before[e, 0, 0:8]), wheel))
        g.check(tuple(before[e, 1, 0:8]) == thruster,
                "environment %d thruster descriptor %r, expected %r"
                % (e, tuple(before[e, 1, 0:8]), thruster))
        # A reset leaves zero commands, and the full-scale figures are
        # the declaration's.
        g.check(before[e, 0, 8] == 0.0 and before[e, 1, 8] == 0.0,
                "environment %d reports a command before any step: %r"
                % (e, before[e, :, 8]))
        g.check(before[e, 0, 9] == 0.20 and before[e, 1, 9] == 400.0,
                "environment %d full-scale figures %r"
                % (e, before[e, :, 9]))

    # The body index and the kind are exact small integers in binary64
    # and are not converted anywhere: they arrive as the doubles the
    # getter wrote.
    for e in range(N_ENVS):
        for a in range(2):
            for field in (0, 1):
                value = before[e, a, field]
                g.check(isinstance(value, np.float64)
                        and float(value).is_integer(),
                        "environment %d actuator %d field %d is not an "
                        "exact small double: %r" % (e, a, field, value))

    # Commands the actuator library clamps, driven past the declared
    # bounds because this package passes action values through
    # unchecked: environment 1's wheel demand is three times its
    # limit and its throttle twice its own.
    env.step(np.array([[0.5, 0.25], [3.0, 2.0]], dtype=np.float64))
    after = env.actuators()
    # A thruster's applied figure is the throttle clamped to the unit
    # interval, times the fraction of the step its tank had propellant
    # for, times its maximum thrust. This fixture declares no
    # propellant, so nothing is spent and that fraction is 1.0
    # throughout; the thruster figures below rest on that premise, and
    # the assertion after them pins it rather than leaving it assumed.
    expected = ((0.5 * 0.2, 0.25 * 400.0), (0.20, 400.0))
    for e in range(N_ENVS):
        for a, want in enumerate(expected[e]):
            got = after[e, a, 8]
            g.check(got == want,
                    "environment %d actuator %d applied %r, expected %r"
                    % (e, a, got, want))
    for e, throttle in ((0, 0.25), (1, 2.0)):
        clamped = min(max(throttle, 0.0), 1.0)
        g.check(after[e, 1, 8] == clamped * after[e, 1, 9],
                "environment %d thrust %r is not the clamped throttle "
                "%r times the full scale %r, so the step's propellant "
                "fraction was not 1"
                % (e, after[e, 1, 8], clamped, after[e, 1, 9]))
        # The descriptors do not move when the commands do.
        g.check(tuple(after[e, 0, 0:8]) == wheel
                and tuple(after[e, 1, 0:8]) == thruster,
                "environment %d descriptors changed under command" % e)
    # The two environments hold different records, so a readback that
    # returned one environment's slot for both fails here.
    g.check((after[0, :, 8] != after[1, :, 8]).all(),
            "the two environments report the same commands: %r"
            % (after[:, :, 8],))
    print("%s: two actuators of two environments report their "
          "descriptors and their clamped commands" % GATE)
    env.close()

    # A program declaring no actuator reports an empty set of the
    # right shape, not a refusal and not something of another shape:
    # the body fixture above declares none.
    env = K26RlVectorEnv(so, seed=SEED, n_envs=N_ENVS)
    empty = env.actuators()
    g.check(empty.shape == (N_ENVS, 0, 10),
            "an artifact declaring no actuator reports %r, expected %r"
            % (empty.shape, (N_ENVS, 0, 10)))
    g.check(empty.dtype == np.float64,
            "the empty actuator array's dtype is %r" % (empty.dtype,))
    env.close()
    print("%s: a program with no actuator reports an empty set of the "
          "getter's own shape" % GATE)

    g.ok(GATE)


if __name__ == "__main__":
    sys.exit(main())
