"""Single-environment Gymnasium mapping over a compiled artifact.

This module also carries the pieces the API shapes share: the spaces
built from the parsed spec, the action marshalling onto the flat
double vector, and the session object that owns the handle, the
buffers, the seed record, and the fault rendering. The vectorised
shape in :mod:`k26rl.vector` and the multi-agent shape in
:mod:`k26rl.parallel` build on the same session.

Every shape returns the measured channels alone. An artifact publishes
what each observation channel is, and where a program routes an
observe through a declared sensor it publishes the uncorrupted value
beside the measured one as a paired ground-truth channel. A policy
reads its craft and its surroundings through the declared sensors,
imperfections and all, so the environment shapes here build their
observation space over the measured channels and return those; the
ground-truth channels come back beside the observation, under
:data:`INFO_TRUTH_OBS`, where a privileged critic reads them and a
policy does not. That is what these shapes do, not what they can be
asked to do: a rule a trainer has to remember to switch on is a rule
nothing keeps.
"""

import ctypes
import warnings

import numpy as np
import gymnasium
from gymnasium import spaces as gym_spaces

from . import _abi, _spec
from ._errors import K26RlError, K26RlFaultError

_SEED_LIMIT = 2 ** 64

#: Info key carrying the ground-truth channels of the observation the
#: same call returned, in channel order. Present exactly when the
#: artifact declares at least one ground-truth channel, which is
#: exactly when ``truth_observation_space`` is not None.
INFO_TRUTH_OBS = "truth_obs"


# ---- spaces from the spec, and nothing else -------------------------

def build_observation_space(spec):
    """Box(-inf, +inf) over the channels a policy reads: every
    declared channel except the ground-truth half of a paired one.
    The spec declares no observation bounds in version 1, so none are
    invented."""
    measured, _truth = _spec.split_channels(spec, 0, spec.obs_total)
    return build_observation_space_of(len(measured))


def build_truth_observation_space(channels):
    """Box(-inf, +inf) over a run of ground-truth channels, or None
    where there are none. The privileged half of an observation is
    shaped like the rest of it, and a consumer building a critic's
    input needs its width without counting names."""
    if not channels:
        return None
    return build_observation_space_of(len(channels))


def build_observation_space_of(count):
    """Box(-inf, +inf) over ``count`` components: the whole vector for
    a single-agent shape, one agent's observation slice width for the
    parallel shape."""
    return gym_spaces.Box(low=-np.inf, high=np.inf, shape=(count,),
                          dtype=np.float64)


def build_action_space(spec):
    """The action space over every declared channel."""
    return build_action_space_of(spec.act_channels)


def build_action_space_of(channels):
    """The action space from a run of declared channels, in declared
    order: all box gives one Box with per-channel bounds, a single
    discrete channel gives Discrete, all discrete gives
    MultiDiscrete, and a mixture gives a Tuple of per-channel
    spaces. The run is the whole action vector for a single-agent
    shape and one agent's action slice for the parallel shape."""
    kinds = [c.kind for c in channels]
    if all(k == _spec.ACT_KIND_BOX for k in kinds):
        lo = np.array([c.lo for c in channels], dtype=np.float64)
        hi = np.array([c.hi for c in channels], dtype=np.float64)
        return gym_spaces.Box(low=lo, high=hi,
                              shape=(len(channels),), dtype=np.float64)
    if all(k == _spec.ACT_KIND_DISCRETE for k in kinds):
        if len(channels) == 1:
            return gym_spaces.Discrete(channels[0].arity)
        return gym_spaces.MultiDiscrete([c.arity for c in channels])
    parts = []
    for c in channels:
        if c.kind == _spec.ACT_KIND_BOX:
            parts.append(gym_spaces.Box(
                low=np.array([c.lo], dtype=np.float64),
                high=np.array([c.hi], dtype=np.float64),
                dtype=np.float64))
        else:
            parts.append(gym_spaces.Discrete(c.arity))
    return gym_spaces.Tuple(parts)


# ---- action marshalling ---------------------------------------------
#
# Incoming actions are coerced to contiguous float64 with the shape
# checked against the spec; the coercion of well-formed input is
# bit-preserving, and a discrete choice index converts to double
# exactly. Values are never checked, clipped, or wrapped against
# bounds or arity: bounds belong to the environment's own physics and
# the training loop, so an out-of-range value passes through as
# given.

def _as_float_batch(value, shape, what):
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError("%s has shape %s, expected %s"
                         % (what, arr.shape, shape))
    if not (np.issubdtype(arr.dtype, np.floating)
            or np.issubdtype(arr.dtype, np.integer)):
        raise TypeError("%s has dtype %s, expected a real numeric "
                        "dtype" % (what, arr.dtype))
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_index_batch(value, shape, what):
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError("%s has shape %s, expected %s"
                         % (what, arr.shape, shape))
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("%s has dtype %s, expected an integer dtype "
                        "for discrete choice indices" % (what, arr.dtype))
    return arr.astype(np.float64)


def _as_index_scalar(value, what):
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise TypeError("%s is a bool; discrete choice indices are "
                        "integers" % what)
    if not isinstance(value, (int, np.integer)):
        raise TypeError("%s is %s; discrete choice indices are "
                        "integers" % (what, type(value).__name__))
    return float(int(value))


def flatten_action(action, space, spec):
    """One environment's action onto its flat double slice."""
    flat = np.empty(spec.act_total, dtype=np.float64)
    flatten_action_into(flat, 0, action, space, spec.act_channels)
    return flat


def flatten_action_into(flat, base, action, space, channels, what="action"):
    """One agent's action onto ``flat`` at the channels' own offsets.

    ``base`` is where the run starts in ``flat`` and ``channels``
    carries the run's declared kinds and bounds; the two coincide for
    a single-agent shape and are the agent's action slice for the
    parallel shape. Channel offsets are absolute, as the spec
    publishes them."""
    count = len(channels)
    if isinstance(space, gym_spaces.Box):
        flat[base:base + count] = _as_float_batch(action, (count,), what)
        return
    if isinstance(space, gym_spaces.Discrete):
        flat[base] = _as_index_scalar(action, what)
        return
    if isinstance(space, gym_spaces.MultiDiscrete):
        flat[base:base + count] = _as_index_batch(action, (count,), what)
        return
    # Tuple of per-channel Box and Discrete entries, declared order.
    if not isinstance(action, (tuple, list)) or len(action) != count:
        raise ValueError(
            "%s must be a sequence of %d per-channel entries"
            % (what, count))
    for chan, part in zip(channels, action):
        if chan.kind == _spec.ACT_KIND_BOX:
            arr = np.asarray(part, dtype=np.float64)
            if arr.shape not in ((), (1,)):
                raise ValueError(
                    "%s channel %d has shape %s, expected a scalar"
                    % (what, chan.offset, arr.shape))
            flat[chan.offset] = float(arr.reshape(()))
        else:
            flat[chan.offset] = _as_index_scalar(
                part, "%s channel %d" % (what, chan.offset))


def flatten_action_batch(actions, space, spec, n_envs):
    """A whole handle's actions onto the env-major flat vector. The
    expected structure is the batched form of the single space, as
    gymnasium's batch_space shapes it."""
    act_total = spec.act_total
    if isinstance(space, gym_spaces.Box):
        arr = _as_float_batch(actions, (n_envs, act_total), "actions")
        return np.ascontiguousarray(arr.reshape(-1))
    if isinstance(space, gym_spaces.Discrete):
        flat = np.empty(n_envs, dtype=np.float64)
        flat[:] = _as_index_batch(actions, (n_envs,), "actions")
        return flat
    if isinstance(space, gym_spaces.MultiDiscrete):
        arr = _as_index_batch(actions, (n_envs, act_total), "actions")
        return np.ascontiguousarray(arr.reshape(-1))
    if not isinstance(actions, (tuple, list)) or \
            len(actions) != len(spec.act_channels):
        raise ValueError(
            "actions must be a sequence of %d per-channel batches"
            % len(spec.act_channels))
    flat = np.empty(n_envs * act_total, dtype=np.float64)
    for chan, part in zip(spec.act_channels, actions):
        what = "actions channel %d" % chan.offset
        if chan.kind == _spec.ACT_KIND_BOX:
            arr = np.asarray(part)
            if arr.shape == (n_envs, 1):
                arr = arr.reshape(n_envs)
            col = _as_float_batch(arr, (n_envs,), what)
        else:
            col = _as_index_batch(part, (n_envs,), what)
        flat[chan.offset::act_total] = col
    return flat


# ---- seed and options checks ----------------------------------------

def check_seed(seed):
    """Seeds are explicit integers in [0, 2^64); there is no default,
    because a defaulted seed invites accidental stream reuse across
    runs."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer in [0, 2**64), not %s"
                        % type(seed).__name__)
    if not 0 <= int(seed) < _SEED_LIMIT:
        raise ValueError("seed %d is outside [0, 2**64)" % int(seed))
    return int(seed)


def check_options(options, refuse=True):
    """None and an empty mapping are accepted and ignored. Version 1
    defines no options, and silently ignoring an explicit request
    would mislead, so a non-empty mapping is refused naming its keys.
    Returns the ignored keys, which is an empty list on every path
    that ignores nothing.

    ``refuse`` is False for the multi-agent shape, where the same rule
    is enforced at a lower strength: that framework's conformance
    check passes a non-empty mapping in to establish that reset
    accepts the argument at all, and this package chose to be
    conformant to it and to pay for the choice here. The mapping is
    still not ignored silently, and not only through the warnings
    machinery, which a consumer can filter away: the keys are returned
    for the caller to publish in what it hands back, so the record of
    what was discarded survives ``-W ignore``."""
    if options is None:
        return []
    if not isinstance(options, dict):
        raise TypeError("options must be a mapping or None, not %s"
                        % type(options).__name__)
    if not options:
        return []
    keys = sorted(map(str, options.keys()))
    if refuse:
        raise ValueError(
            "version 1 defines no reset options; refusing options "
            "with keys %s" % keys)
    # Three frames out is the consumer's own reset call: this frame,
    # the API method that called it, and the caller of that.
    warnings.warn(
        "version 1 defines no reset options; ignoring options with "
        "keys %s" % keys, stacklevel=3)
    return keys


# ---- the session: handle, buffers, seed record, fault rendering -----

class _Session:
    """Owns one artifact handle and everything stateful around it.

    Not thread-safe, any number per process: a handle is
    single-threaded, and concurrent handles in one process are legal.
    """

    def __init__(self, artifact_path, seed, n_envs, on_fault,
                 multi_agent=False):
        if on_fault not in ("raise", "truncate"):
            raise ValueError(
                "on_fault must be \"raise\" or \"truncate\", not %r"
                % (on_fault,))
        seed = check_seed(seed)
        if isinstance(n_envs, bool) or not isinstance(n_envs, (int,
                                                               np.integer)):
            raise TypeError("n_envs must be an integer, not %s"
                            % type(n_envs).__name__)
        n_envs = int(n_envs)
        if n_envs < 1:
            raise ValueError("n_envs must be at least 1, not %d" % n_envs)

        self.artifact = _abi.Artifact(artifact_path)
        self.on_fault = on_fault
        self.n_envs = n_envs
        self._handle = self.artifact.create(seed, n_envs)
        self._closed = False

        try:
            blob = self.artifact.spec_blob(self._handle)
            self.spec = _spec.parse(blob)
            _spec.validate(self.spec, self.artifact.abi_version, n_envs)
            # How many agents this shape can serve is the shape's own
            # question; both refusals name the count found.
            if multi_agent:
                _spec.require_multi_agent(self.spec)
            else:
                _spec.require_single_agent(self.spec)
        except Exception:
            self.artifact.destroy(self._handle)
            self._closed = True
            raise

        # The cumulative seed record: every seed any of this object's
        # handles has held, carried across recreates so a seed held
        # before a recreate still routes as held. Per session object;
        # two sessions over one artifact each start empty.
        self._seeds_held = {seed}
        self._current_key = seed
        # Untouched: sitting at episode 0 of the current key with no
        # step and no episode-advancing reset since the key was
        # established. Creation completes episode 0's initial reset,
        # so a fresh handle is untouched by construction.
        self._untouched = True
        self._output_path = None
        self._tap_name = None

        # The two halves of the observation vector, as absolute
        # channel indices. Read once here because they are a property
        # of the artifact, not of a step, and applied by every shape:
        # the measured channels are what an observation is, and the
        # ground-truth channels travel beside it.
        measured, truth = _spec.split_channels(self.spec, 0,
                                               self.spec.obs_total)
        self.policy_channels = tuple(measured)
        self.truth_channels = tuple(truth)
        self._policy_index = np.asarray(measured, dtype=np.intp)
        self._truth_index = np.asarray(truth, dtype=np.intp)

        obs_n = n_envs * self.spec.obs_total
        self._obs_buf = np.empty(obs_n, dtype=np.float64)
        self._rew_buf = np.empty(n_envs * self.spec.agent_count,
                                 dtype=np.float64)
        self._flags_buf = np.empty(n_envs, dtype=np.uint32)
        self._fault_buf = np.empty(n_envs, dtype=np.uint16)

    # ---- guards and small readers ------------------------------------

    @property
    def closed(self):
        return self._closed

    @property
    def seeds_held(self):
        """Read-only view of the cumulative seed record."""
        return frozenset(self._seeds_held)

    @property
    def output_path(self):
        return self._output_path

    @property
    def tap_name(self):
        return self._tap_name

    def ensure_open(self):
        if self._closed:
            raise K26RlError(
                None, "environment is closed; no further calls reach "
                "the artifact")

    @staticmethod
    def _ptr(buf, ctype):
        return buf.ctypes.data_as(ctypes.POINTER(ctype))

    def read_obs(self):
        """The whole declared observation vector, ground-truth
        channels included; :meth:`policy_obs` and :meth:`truth_obs`
        cut it into the half a policy reads and the half it does not.

        Fresh copy each call, never a view over the preallocated
        buffer: consumers store returned arrays, and an aliased view
        would be silently overwritten one step later."""
        self.artifact.obs(self._handle,
                          self._ptr(self._obs_buf, ctypes.c_double))
        return self._obs_buf.reshape(
            self.n_envs, self.spec.obs_total).copy()

    def policy_obs(self, obs):
        """The measured channels of a whole handle's observation, as
        ``(n_envs, len(policy_channels))``. An artifact with no
        ground-truth channel drops nothing, and the read's own fresh
        copy is returned unchanged; otherwise the selection makes
        one."""
        if not self.truth_channels:
            return obs
        return obs[:, self._policy_index]

    def truth_obs(self, obs):
        """The ground-truth channels of that same observation, as
        ``(n_envs, len(truth_channels))`` in channel order."""
        return obs[:, self._truth_index]

    def read_reward(self):
        self.artifact.reward(self._handle,
                             self._ptr(self._rew_buf, ctypes.c_double))
        return self._rew_buf.copy()

    def read_flags(self):
        self.artifact.flags(self._handle,
                            self._ptr(self._flags_buf, ctypes.c_uint32))
        return self._flags_buf.copy()

    def read_fault_codes(self):
        self.artifact.fault_codes(
            self._handle, self._ptr(self._fault_buf, ctypes.c_uint16))
        return self._fault_buf.copy()

    # ---- stepping -----------------------------------------------------

    def step_flat(self, flat_actions):
        """One step call plus the per-step reads. The flag words are
        read on every step, and the fault codes whenever any fault
        bit is set, so no consumer of this package can under-read a
        fault whatever its own habits."""
        self._untouched = False
        flat_actions = np.ascontiguousarray(flat_actions,
                                            dtype=np.float64)
        self.artifact.step(self._handle,
                           self._ptr(flat_actions, ctypes.c_double))
        obs = self.read_obs()
        rew = self.read_reward()
        flags = self.read_flags()
        fault_mask = (flags & _abi.FLAG_FAULT) != 0
        codes = self.read_fault_codes() if fault_mask.any() else None
        return obs, rew, flags, fault_mask, codes

    def fault_details(self, fault_mask, codes):
        indices = [int(i) for i in np.flatnonzero(fault_mask)]
        picked = [int(codes[i]) for i in indices]
        reasons = [self.artifact.status_message(c) for c in picked]
        return indices, picked, reasons

    # ---- seeding routes ----------------------------------------------

    def reset_routed(self, seed):
        """The reset mapping. Plain reset preserves episode 0 on an
        untouched handle and advances every episode index otherwise.
        A seeded reset delivers episode 0 of the requested key by one
        of three routes chosen from the handle's state; all three
        yield bit-identical streams."""
        if seed is None:
            if not self._untouched:
                self.artifact.reset(self._handle)
            # On an untouched handle construction already completed
            # episode 0's initial reset, so the construct-then-reset
            # idiom delivers episode 0 rather than discarding it.
            return self.read_obs()

        seed = check_seed(seed)
        if self._untouched and self._current_key == seed:
            # Route 1: already sitting untouched at episode 0 of this
            # key, however the handle came to hold it. No call, no
            # episode consumed; a repeated seeded reset is a no-op.
            return self.read_obs()
        if seed not in self._seeds_held:
            # Route 2: a key this session has never held rekeys the
            # live handle and zeroes its episode indices.
            self.artifact.reset_seeded(self._handle, seed)
            self._seeds_held.add(seed)
            self._current_key = seed
            self._untouched = True
            return self.read_obs()
        # Route 3: the handle would refuse a seed it has held, so the
        # session recreates. Refused while episode output is enabled:
        # recreating would close the file mid-recording and the same
        # path would then be refused as existing; nothing silently
        # splits or truncates a recording.
        if self._output_path is not None:
            raise K26RlError(
                None,
                "reset(seed=%d) needs a fresh handle because this "
                "environment has already held that seed, and episode "
                "output is enabled; disable output with "
                "set_output(None) at an episode boundary first, or "
                "choose a seed not yet held" % seed)
        # And refused while the tap is armed, for the same kind of
        # reason: destroying the handle closes the ring and removes
        # its name, so the fresh handle would publish nothing while a
        # watching host went on stepping into a ring that is gone.
        # Nothing silently stops publishing.
        if self._tap_name is not None:
            raise K26RlError(
                None,
                "reset(seed=%d) needs a fresh handle because this "
                "environment has already held that seed, and the "
                "telemetry tap is armed; disarm it with tap(None) at "
                "an episode boundary first, or choose a seed not yet "
                "held" % seed)
        self.artifact.destroy(self._handle)
        self._handle = None
        try:
            self._handle = self.artifact.create(seed, self.n_envs)
        except Exception as exc:
            # The old handle is already destroyed, so this object
            # cannot honestly continue; it closes, and every later
            # call gets the use-after-close refusal instead of the
            # artifact's null-handle error.
            self._closed = True
            raise K26RlError(
                None,
                "recreate for seed %d failed after the previous "
                "handle was destroyed; the environment is now closed"
                % seed) from exc
        self._current_key = seed
        self._untouched = True
        return self.read_obs()

    # ---- the training-host surface -------------------------------------
    #
    # Three getters that postdate the frozen thirteen, bound here for
    # a host that drives this package directly: the tap it arms to
    # watch a run, and the two readbacks it takes of the world the
    # observation channels are views of. Each is present on the
    # session whatever the artifact reports; the binding layer says
    # the absence of one the loaded artifact does not carry.

    def tap(self, name):
        """Arm the telemetry ring under ``name``, or disarm it with
        None. Arming is the host's own act and publishes nothing back
        into the simulation: the ring is read-only to its consumers,
        and a run is bit-identical whether or not one is watching.

        The armed name is recorded because the handle can outlive it:
        a seeded reset that recreates the handle would take the ring
        with it, so the reset routes consult this."""
        if name is not None and not isinstance(name, str):
            raise TypeError("tap name must be a str or None, not %s"
                            % type(name).__name__)
        self.artifact.tap(self._handle, name)
        self._tap_name = None if name is None else str(name)

    def resolve_body_reference(self, reference):
        """The body getter's reference argument from a body index, a
        declared body name, or the origin constant.

        A name is resolved here because only the spec blob carries the
        mapping; an index and the origin constant are passed through
        untouched, so a reference naming no body is refused by the
        getter itself rather than by a guess made here."""
        if isinstance(reference, str):
            for index, name in self.spec.body_names.items():
                if name == reference:
                    return index
            known = sorted(self.spec.body_names.values())
            raise ValueError(
                "no body is named %r; this artifact declares %s"
                % (reference, known if known else "no body names"))
        if isinstance(reference, bool) or not isinstance(
                reference, (int, np.integer)):
            raise TypeError(
                "body reference must be an index, a declared body "
                "name, or BODY_REF_ORIGIN, not %s"
                % type(reference).__name__)
        index = int(reference)
        if not 0 <= index <= 0xFFFFFFFF:
            raise ValueError(
                "body reference %d is outside the surface's unsigned "
                "32-bit reference range" % index)
        return index

    def read_bodies(self, reference):
        """Every body of every environment, env-major, as
        ``(n_envs, body_count, 6)``: three position components then
        three velocity components, positions relative to
        ``reference``.

        The array is freshly allocated for the call and returned as
        the getter laid it out, reshaped and not reordered."""
        index = self.resolve_body_reference(reference)
        need = self.artifact.bodies(self._handle, index, None, 0)
        return self._shaped_read(
            "k26rl_env_bodies", _abi.BODY_STRIDE, need,
            lambda buf, count: self.artifact.bodies(
                self._handle, index,
                self._ptr(buf, ctypes.c_double), count))

    def read_actuators(self):
        """The actuator set as the latest step drove it, env-major, as
        ``(n_envs, actuator_count, 10)``.

        The ten values are the getter's own and cross untouched: the
        bound body's index, the kind, the mounting position, the axis
        or thrust direction, the applied magnitude, and the full-scale
        magnitude. The first two are exact small integers in binary64
        and are not converted here."""
        need = self.artifact.actuators(self._handle, None, 0)
        return self._shaped_read(
            "k26rl_env_actuators", _abi.ACTUATOR_STRIDE, need,
            lambda buf, count: self.artifact.actuators(
                self._handle, self._ptr(buf, ctypes.c_double), count))

    def _shaped_read(self, symbol, stride, need, fill):
        """One sized read of a getter following the spec getter's
        sizing convention, shaped ``(n_envs, need // (n_envs * stride),
        stride)``. A requirement that is not a whole number of rows
        per environment is an artifact defect worth naming rather than
        reshaping around."""
        row = self.n_envs * stride
        if need % row:
            raise K26RlError(
                None,
                "%s requires %d doubles, which is not %d environments "
                "times a whole number of rows of %d"
                % (symbol, need, self.n_envs, stride))
        count = need // row
        buf = np.empty(need, dtype=np.float64)
        got = fill(buf, need)
        if got != need:
            # The count is fixed after create, so two sizing answers
            # disagreeing is an artifact defect worth naming.
            raise K26RlError(
                None,
                "%s sized its output at %d doubles and then at %d"
                % (symbol, need, got))
        return buf.reshape(self.n_envs, count, stride)

    # ---- episode output ----------------------------------------------

    def set_output(self, path):
        """Enable episode-file emission to path, or disable it with
        None. The writer is artifact-side; this only throws the
        switch, and the timing and existing-path refusals surface as
        K26RlOutputError."""
        self.artifact.output(self._handle, path)
        self._output_path = None if path is None else str(path)

    # ---- teardown -----------------------------------------------------

    def close(self):
        if not self._closed:
            self._closed = True
            self.artifact.destroy(self._handle)
            self._handle = None


# ---- the single-environment shape -----------------------------------

class K26RlEnv(gymnasium.Env):
    """One environment behind gymnasium.Env.

    Construction loads the artifact at ``artifact_path``, checks its
    ABI version, creates the handle with the required explicit
    ``seed``, and builds the spaces from the artifact's spec blob
    alone. The environment is live from birth: construction completes
    episode 0's initial reset.

    The observation is the artifact's measured channels. Where the
    program routed an observe through a declared sensor, the
    uncorrupted values beside it come back under
    :data:`INFO_TRUTH_OBS` instead, sized by
    ``truth_observation_space``; ``policy_channels`` and
    ``truth_channels`` name which declared channel each component of
    the two arrays is.

    ``on_fault`` selects the fault mode: ``"raise"`` (the default)
    raises :class:`K26RlFaultError` after a faulting step's results
    are formed; ``"truncate"`` returns the same results without
    raising. Both render a fault as a truncated episode ending with
    the ``fault_code`` and ``fault_reason`` info keys set; no mode
    drops a fault.

    After any episode ending, a caught fault included, the next call
    is ``reset()``; stepping a finished episode raises
    ``gymnasium.error.ResetNeeded``.

    Not thread-safe; any number of instances per process.
    """

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, artifact_path, seed, on_fault="raise"):
        self._session = _Session(artifact_path, seed, 1, on_fault)
        spec = self._session.spec
        self.observation_space = build_observation_space(spec)
        self.action_space = build_action_space(spec)
        self.truth_observation_space = build_truth_observation_space(
            self._session.truth_channels)
        self._needs_reset = False

    # ---- spec data exposed for consumers and tooling ------------------
    #
    # Like the methods, the spec-reading properties refuse after
    # close(). The two record properties, seeds_held and output_path,
    # stay readable: they are plain Python records of what the run
    # did, wanted precisely after it, and reading them touches no
    # artifact state.

    @property
    def env_spec(self):
        """The parsed spec value object."""
        self._session.ensure_open()
        return self._session.spec

    @property
    def control_dt(self):
        """Simulated seconds per external step."""
        self._session.ensure_open()
        return self._session.spec.control_dt

    @property
    def obs_channel_names(self):
        self._session.ensure_open()
        return dict(self._session.spec.obs_channel_names)

    @property
    def obs_channel_kinds(self):
        self._session.ensure_open()
        return dict(self._session.spec.obs_channel_kinds)

    @property
    def policy_channels(self):
        """The declared channels this environment's observation
        carries, in the order it carries them: the measured ones."""
        self._session.ensure_open()
        return self._session.policy_channels

    @property
    def truth_channels(self):
        """The declared channels the observation does not carry, in
        the order the ``INFO_TRUTH_OBS`` array carries them: the
        ground truth paired with the measured channels above."""
        self._session.ensure_open()
        return self._session.truth_channels

    @property
    def on_fault(self):
        self._session.ensure_open()
        return self._session.on_fault

    @property
    def seeds_held(self):
        """Every seed this environment object's handles have held,
        cumulative across recreates. Read-only, and still readable
        after close(): the seed record of a finished run is exactly
        what reproduction needs."""
        return self._session.seeds_held

    # ---- the API ------------------------------------------------------

    def _truth_into(self, info, obs):
        """The ground-truth channels of ``obs`` put beside the
        measured ones in that call's info mapping. Both halves are cut
        from the one vector the call read, so they cannot drift a step
        apart."""
        if self._session.truth_channels:
            info[INFO_TRUTH_OBS] = self._session.truth_obs(obs)[0]
        return info

    def reset(self, *, seed=None, options=None):
        self._session.ensure_open()
        check_options(options)
        if seed is not None:
            seed = check_seed(seed)
        # The base reset keeps np_random available as the ecosystem
        # expects; nothing here ever draws from it, because every
        # draw in an environment's life is artifact-side.
        super().reset(seed=seed)
        obs = self._session.reset_routed(seed)
        self._needs_reset = False
        return (self._session.policy_obs(obs)[0],
                self._truth_into({}, obs))

    def step(self, action):
        self._session.ensure_open()
        if self._needs_reset:
            raise gymnasium.error.ResetNeeded(
                "the episode has ended; call reset() before stepping")
        flat = flatten_action(action, self.action_space,
                              self._session.spec)
        obs, rew, flags, fault_mask, codes = \
            self._session.step_flat(flat)

        word = int(flags[0])
        terminated = bool(word & _abi.FLAG_TERMINATED)
        truncated = bool(word & _abi.FLAG_TRUNCATED)
        info = self._truth_into({}, obs)
        faulted = bool(word & _abi.FLAG_FAULT)
        if faulted:
            # A fault is not termination, and the episode ended for a
            # reason outside the task's own dynamics, so truncated is
            # the rendering; the faulting step completed no
            # transition, so the observation is the pre-step one with
            # reward zero, as the artifact delivers it.
            truncated = True
            indices, picked, reasons = \
                self._session.fault_details(fault_mask, codes)
            info["fault_code"] = picked[0]
            info["fault_reason"] = reasons[0]

        if terminated or truncated:
            self._needs_reset = True
        results = (self._session.policy_obs(obs)[0],
                   float(rew[0]), terminated, truncated, info)
        if faulted and self._session.on_fault == "raise":
            raise K26RlFaultError(indices, picked, reasons, results)
        return results

    def set_output(self, path):
        """Enable episode-file emission to path, or disable it with
        None."""
        self._session.ensure_open()
        self._session.set_output(path)

    # ---- the training-host surface ------------------------------------
    #
    # The three getters that postdate the frozen set. Each method
    # exists whatever the loaded artifact reports; one the artifact is
    # too old to carry refuses by naming the symbol, the ABI minor it
    # arrived at, and the minor the artifact reports.

    def tap(self, name):
        """Arm the telemetry ring under ``name``, or disarm it with
        None, so a viewer can watch this environment while it runs.

        Callable only at an episode boundary, as ``set_output`` is:
        after construction, immediately after a reset, and before the
        step that follows. Watching cannot change the run.

        While a tap is armed, a ``reset(seed=S)`` that would need a
        fresh handle is refused rather than taking the ring with it;
        disarm first or choose a seed this environment has not
        held."""
        self._session.ensure_open()
        self._session.tap(name)

    def bodies(self, reference):
        """Every body's position and velocity, as
        ``(1, body_count, 6)``: three position components then three
        velocity components, positions taken relative to
        ``reference``.

        The leading axis is the handle's environment count, which is
        one for this shape, so the layout is the getter's own on
        either shape and a consumer reads row 0.

        ``reference`` is a body index, a declared body name, or
        :data:`k26rl.BODY_REF_ORIGIN` for the world origin. Positions
        relative to a body are the runtime's exact subtraction;
        relative to the origin they are the flattened coordinate and
        carry that form's precision limit."""
        self._session.ensure_open()
        return self._session.read_bodies(reference)

    def actuators(self):
        """The actuator set as the latest step drove it, as
        ``(1, actuator_count, 10)`` and in the getter's own order.

        The ten values per actuator are the bound body's index, the
        kind, the mounting position, the axis or thrust direction, the
        applied magnitude, and the full-scale magnitude. They cross
        untouched, the first two as the exact small integers the
        getter writes."""
        self._session.ensure_open()
        return self._session.read_actuators()

    @property
    def body_names(self):
        """Declared body name by index, in the order
        :meth:`bodies` reports them. Empty for an artifact that
        publishes no body names."""
        self._session.ensure_open()
        return dict(self._session.spec.body_names)

    @property
    def output_path(self):
        """The enabled episode-output path, or None. Still readable
        after close(): it locates the recorded file."""
        return self._session.output_path

    @property
    def tap_name(self):
        """The armed telemetry ring's name, or None. Still readable
        after close(): it names the ring the run published to."""
        return self._session.tap_name

    def close(self):
        # Idempotent, as the ecosystem expects; everything else on a
        # closed environment is refused before any artifact call,
        # except the seeds_held and output_path records.
        self._session.close()
