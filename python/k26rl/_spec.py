"""Decoder for the artifact's env_spec TLV blob.

The blob is a sequence of (tag uint16, length uint32, value) triples,
little-endian, no alignment padding; unknown tags are skipped by
length. Everything this package knows about an environment it learns
here, so the parse is strict about what it reads and silent about
what it does not know.

No numpy, no gymnasium: the parsed value object is plain Python, and
the spaces built from it live with the environment classes.
"""

import struct

from ._errors import K26RlError

TAG_ABI_VERSION = 0x0001
TAG_ENDIAN_PROBE = 0x0002
TAG_AGENT_COUNT = 0x0003
TAG_N_ENVS = 0x0004
TAG_CONTROL_DT = 0x0005
TAG_HORIZON = 0x0006
TAG_OBS_TOTAL = 0x0007
TAG_ACT_TOTAL = 0x0008
TAG_AGENT_OBS_SLICE = 0x0009
TAG_AGENT_ACT_SLICE = 0x000A
TAG_ACT_BOUNDS = 0x000B
TAG_ACT_KIND = 0x000C
TAG_OBS_CHANNEL_NAME = 0x000D
TAG_OBS_CHANNEL_KIND = 0x000E
TAG_EPISODE_FLAGS = 0x000F
TAG_REWARD_COMPONENTS = 0x0010
TAG_OBS_CHANNEL_SOURCE = 0x0016

ACT_KIND_BOX = 0
ACT_KIND_DISCRETE = 1

OBS_KIND_VECTOR = 0

# What an observation channel is. A measured channel carries the value
# a policy is allowed to read: the simulation's value once every
# declared imperfection has been applied to it, or the simulation's
# value itself where the program declared no sensor. A ground-truth
# channel carries the uncorrupted value beside it, for a privileged
# critic and for the episode record.
OBS_SOURCE_MEASURED = 0
OBS_SOURCE_TRUTH = 1

# The paired-channel value of a channel that has no pair.
OBS_PAIR_NONE = 0xFFFFFFFF

ENDIAN_PROBE_VALUE = 0x01020304

EPISODE_FLAG_AUTO_RESET = 1 << 0

_INF = float("inf")


class ActChannel:
    """One action scalar: its offset in the flat action vector, its
    kind, its bounds (box), and its arity (discrete)."""

    __slots__ = ("offset", "kind", "arity", "lo", "hi")

    def __init__(self, offset, kind, arity, lo, hi):
        self.offset = offset
        self.kind = kind
        self.arity = arity
        self.lo = lo
        self.hi = hi


class Spec:
    """The parsed spec: scalar tags as attributes, per-channel and
    per-agent tags as lists and mappings. Values default to None (or
    empty containers) until their tag is seen."""

    def __init__(self):
        self.abi_version = None
        self.endian_probe = None
        self.agent_count = None
        self.n_envs = None
        self.control_dt = None
        self.horizon = None
        self.obs_total = None
        self.act_total = None
        self.agent_obs_slices = []
        self.agent_act_slices = []
        self.act_bounds = {}
        self.act_kinds = {}
        self.obs_channel_names = {}
        self.obs_channel_kinds = {}
        self.obs_channel_sources = {}
        self.obs_channel_pairs = {}
        self.episode_flags = None
        self.act_channels = []


def _malformed(detail):
    raise K26RlError(None, "spec blob is malformed: " + detail)


def _need_len(tag, length, expect):
    if length != expect:
        _malformed("tag 0x%04x carries %d bytes, expected %d"
                   % (tag, length, expect))


def parse(blob):
    """Parse a spec blob into a :class:`Spec`. Unknown tags are
    skipped by length; a truncated entry is refused."""
    spec = Spec()
    off = 0
    end = len(blob)
    while off < end:
        if off + 6 > end:
            _malformed("entry header at byte %d runs past the end" % off)
        tag, length = struct.unpack_from("<HI", blob, off)
        value_off = off + 6
        if value_off + length > end:
            _malformed("tag 0x%04x at byte %d runs past the end"
                       % (tag, off))

        if tag == TAG_ABI_VERSION:
            _need_len(tag, length, 4)
            (spec.abi_version,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_ENDIAN_PROBE:
            _need_len(tag, length, 4)
            (spec.endian_probe,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_AGENT_COUNT:
            _need_len(tag, length, 4)
            (spec.agent_count,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_N_ENVS:
            _need_len(tag, length, 4)
            (spec.n_envs,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_CONTROL_DT:
            _need_len(tag, length, 8)
            # The payload is the binary64 bit pattern; unpacking it as
            # a little-endian double reads those bits exactly.
            (spec.control_dt,) = struct.unpack_from("<d", blob, value_off)
        elif tag == TAG_HORIZON:
            _need_len(tag, length, 4)
            (spec.horizon,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_OBS_TOTAL:
            _need_len(tag, length, 4)
            (spec.obs_total,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_ACT_TOTAL:
            _need_len(tag, length, 4)
            (spec.act_total,) = struct.unpack_from("<I", blob, value_off)
        elif tag == TAG_AGENT_OBS_SLICE:
            _need_len(tag, length, 12)
            spec.agent_obs_slices.append(
                struct.unpack_from("<III", blob, value_off))
        elif tag == TAG_AGENT_ACT_SLICE:
            _need_len(tag, length, 12)
            spec.agent_act_slices.append(
                struct.unpack_from("<III", blob, value_off))
        elif tag == TAG_ACT_BOUNDS:
            _need_len(tag, length, 20)
            offset, lo, hi = struct.unpack_from("<Idd", blob, value_off)
            spec.act_bounds[offset] = (lo, hi)
        elif tag == TAG_ACT_KIND:
            # 6 bytes for a box channel; 10 with the trailing arity
            # for a discrete one.
            if length == 6:
                offset, kind = struct.unpack_from("<IH", blob, value_off)
                arity = 0
            elif length == 10:
                offset, kind, arity = struct.unpack_from(
                    "<IHI", blob, value_off)
            else:
                _malformed("tag 0x%04x carries %d bytes, expected 6 or 10"
                           % (tag, length))
            spec.act_kinds[offset] = (kind, arity)
        elif tag == TAG_OBS_CHANNEL_NAME:
            if length < 4:
                _malformed("tag 0x%04x carries %d bytes, expected at "
                           "least 4" % (tag, length))
            (channel,) = struct.unpack_from("<I", blob, value_off)
            name = blob[value_off + 4:value_off + length]
            spec.obs_channel_names[channel] = name.decode("utf-8",
                                                          "replace")
        elif tag == TAG_OBS_CHANNEL_KIND:
            _need_len(tag, length, 6)
            channel, kind = struct.unpack_from("<IH", blob, value_off)
            spec.obs_channel_kinds[channel] = kind
        elif tag == TAG_OBS_CHANNEL_SOURCE:
            _need_len(tag, length, 10)
            channel, source, pair = struct.unpack_from(
                "<IHI", blob, value_off)
            spec.obs_channel_sources[channel] = source
            spec.obs_channel_pairs[channel] = pair
        elif tag == TAG_EPISODE_FLAGS:
            _need_len(tag, length, 4)
            (spec.episode_flags,) = struct.unpack_from("<I", blob,
                                                       value_off)
        # Any other tag, the reserved reward-components tag included,
        # is skipped by length: additions arrive as new tags under the
        # minor version rule and older consumers step over them.

        off = value_off + length

    if spec.act_total is not None:
        for offset in range(spec.act_total):
            kind, arity = spec.act_kinds.get(offset, (ACT_KIND_BOX, 0))
            lo, hi = spec.act_bounds.get(offset, (-_INF, _INF))
            spec.act_channels.append(
                ActChannel(offset, kind, arity, lo, hi))
    return spec


# ---- measured channels, and the truth beside them --------------------
#
# Every observation channel publishes what it is. A measured channel
# carries what a policy is allowed to read: the value the simulation
# produced once every imperfection the program declared has been
# applied to it, or that value itself where the program declared no
# sensor. A ground-truth channel carries the uncorrupted value beside
# a measured one, for a privileged critic and for the episode record,
# and a policy never reads it.
#
# A blob that publishes no source tag for a channel was written before
# the tag existed; that channel is measured and unpaired, which is
# what a program declaring no sensor publishes explicitly.


def channel_source(spec, channel):
    """What observation channel ``channel`` carries."""
    return spec.obs_channel_sources.get(channel, OBS_SOURCE_MEASURED)


def channel_pair(spec, channel):
    """The channel paired with ``channel``, or :data:`OBS_PAIR_NONE`
    when it has none."""
    return spec.obs_channel_pairs.get(channel, OBS_PAIR_NONE)


def split_channels(spec, offset, count):
    """The channels of ``[offset, offset + count)`` split into the
    measured ones a policy reads and the ground-truth ones it does
    not, each in ascending channel order.

    The range is the whole observation vector for a single-agent
    shape and one agent's observation slice for the parallel shape."""
    measured = []
    truth = []
    for channel in range(offset, offset + count):
        if channel_source(spec, channel) == OBS_SOURCE_TRUTH:
            truth.append(channel)
        else:
            measured.append(channel)
    return measured, truth


def _check_channel_sources(spec):
    """A declared pairing must be a pairing: a source kind this
    version knows, an in-range partner of the opposite kind, and the
    partner naming the channel back. Keeping ground truth out of a
    policy's observation rests entirely on these tags, so they are
    checked against the blob rather than assumed of it, and a blob
    that half declares a pair is refused rather than resolved by
    guesswork."""
    for channel in sorted(spec.obs_channel_sources):
        source = spec.obs_channel_sources[channel]
        if channel >= spec.obs_total:
            raise K26RlError(
                None,
                "a source tag names observation channel %d, outside "
                "the declared observation total %d"
                % (channel, spec.obs_total))
        if source not in (OBS_SOURCE_MEASURED, OBS_SOURCE_TRUTH):
            raise K26RlError(
                None,
                "observation channel %d declares source kind %d, which "
                "is neither measured (%d) nor ground truth (%d)"
                % (channel, source, OBS_SOURCE_MEASURED,
                   OBS_SOURCE_TRUTH))
        pair = channel_pair(spec, channel)
        if pair == OBS_PAIR_NONE:
            if source == OBS_SOURCE_TRUTH:
                raise K26RlError(
                    None,
                    "observation channel %d publishes as ground truth "
                    "with no measured channel paired with it" % channel)
            continue
        if pair >= spec.obs_total:
            raise K26RlError(
                None,
                "observation channel %d is paired with channel %d, "
                "outside the declared observation total %d"
                % (channel, pair, spec.obs_total))
        if pair == channel:
            raise K26RlError(
                None,
                "observation channel %d is paired with itself" % channel)
        if channel_source(spec, pair) == source:
            raise K26RlError(
                None,
                "observation channels %d and %d are paired and both "
                "publish source kind %d; a pair is one measured "
                "channel and the ground truth beside it"
                % (channel, pair, source))
        if channel_pair(spec, pair) != channel:
            raise K26RlError(
                None,
                "observation channel %d is paired with channel %d, "
                "which is paired with %d; a pair names itself from "
                "both ends"
                % (channel, pair, channel_pair(spec, pair)))


def validate(spec, abi_version, n_envs):
    """Refuse a spec this package cannot honestly serve. The checks
    are the load-time contract: probe, version echo, geometry echo,
    agent count, slice arithmetic, channel pairing, and auto-reset
    stepping.

    How many agents a given API shape can serve is that shape's
    question, not this one's: :func:`require_single_agent` and
    :func:`require_multi_agent` answer it."""
    if spec.endian_probe != ENDIAN_PROBE_VALUE:
        raise K26RlError(
            None,
            "spec endian probe reads 0x%08x, expected 0x%08x"
            % (spec.endian_probe or 0, ENDIAN_PROBE_VALUE))
    if spec.abi_version != abi_version:
        raise K26RlError(
            None,
            "spec declares ABI version 0x%08x but the artifact "
            "reports 0x%08x" % (spec.abi_version or 0, abi_version))
    if spec.n_envs != n_envs:
        raise K26RlError(
            None,
            "spec declares %s environments but the handle was created "
            "with %d" % (spec.n_envs, n_envs))
    if spec.obs_total is None or spec.act_total is None:
        raise K26RlError(
            None, "spec declares no observation or action totals")
    if spec.act_total == 0:
        raise K26RlError(
            None,
            "spec declares action total 0; an environment with no "
            "action channels cannot be stepped")
    if spec.agent_count is None or spec.agent_count < 1:
        raise K26RlError(
            None,
            "spec declares agent count %s; every environment has at "
            "least one agent" % spec.agent_count)
    _check_slices(spec.agent_obs_slices, spec.obs_total, "observation",
                  spec.agent_count)
    _check_slices(spec.agent_act_slices, spec.act_total, "action",
                  spec.agent_count)
    _check_channel_sources(spec)
    for chan in spec.act_channels:
        if chan.kind == ACT_KIND_DISCRETE and chan.arity < 1:
            raise K26RlError(
                None,
                "discrete action channel at offset %d declares arity "
                "%d" % (chan.offset, chan.arity))
        elif chan.kind not in (ACT_KIND_BOX, ACT_KIND_DISCRETE):
            raise K26RlError(
                None,
                "action channel at offset %d declares unknown kind %d"
                % (chan.offset, chan.kind))
    if spec.episode_flags is None or \
            not spec.episode_flags & EPISODE_FLAG_AUTO_RESET:
        raise K26RlError(
            None,
            "artifact declares episode flags 0x%x with the auto-reset "
            "bit clear; this package maps auto-reset stepping only "
            "and has no non-auto-reset mapping in version 1"
            % (spec.episode_flags or 0))


def _check_slices(slices, total, what, agent_count):
    """The per-agent slices must partition the vector: one slice per
    declared agent, contiguous in agent-index order from zero, ending
    exactly at the declared total. Everything a multi-agent consumer
    reads out of the flat vectors rests on this, so it is checked
    against the blob rather than assumed of it."""
    if not slices:
        if agent_count > 1:
            raise K26RlError(
                None,
                "spec declares agent count %d but carries no per-agent "
                "%s slice" % (agent_count, what))
        # A one-agent vector is fully described by its total.
        return
    if len(slices) != agent_count:
        raise K26RlError(
            None,
            "spec declares agent count %d but carries %d per-agent %s "
            "slices" % (agent_count, len(slices), what))
    seen = {}
    for agent, offset, count in slices:
        if agent >= agent_count:
            raise K26RlError(
                None,
                "an %s slice names agent %d, outside the declared "
                "agent count %d" % (what, agent, agent_count))
        if agent in seen:
            raise K26RlError(
                None, "two %s slices name agent %d" % (what, agent))
        seen[agent] = (offset, count)
    covered = 0
    for agent in range(agent_count):
        offset, count = seen[agent]
        if offset + count > total:
            raise K26RlError(
                None,
                "agent %d %s slice [%d, %d) runs past the declared "
                "total %d" % (agent, what, offset, offset + count,
                              total))
        if offset != covered:
            raise K26RlError(
                None,
                "agent %d %s slice starts at %d, but the slices before "
                "it cover [0, %d); per-agent slices are contiguous in "
                "agent order" % (agent, what, offset, covered))
        covered += count
    if covered != total:
        raise K26RlError(
            None,
            "agent %s slices cover %d of the declared total %d"
            % (what, covered, total))


def slices_by_agent(slices, agent_count):
    """The per-agent slices as ``(offset, count)`` pairs indexed by
    agent, once :func:`validate` has established that they partition
    the vector."""
    table = {agent: (offset, count) for agent, offset, count in slices}
    return [table[agent] for agent in range(agent_count)]


def require_single_agent(spec):
    """The refusal the single-agent API shapes make. A Gymnasium
    environment reports one observation, one action and one reward per
    step, so it cannot serve more than one agent's; a multi-agent
    artifact's shape is the parallel wrapper."""
    if spec.agent_count != 1:
        raise K26RlError(
            None,
            "artifact declares agent count %s; a single-agent API "
            "cannot serve a %s-agent artifact, and K26RlParallelEnv "
            "is the shape that does"
            % (spec.agent_count, spec.agent_count))


def require_multi_agent(spec):
    """The refusal the parallel wrapper makes. At agent count 1
    observation channel names publish unqualified, so the artifact
    carries no agent name anywhere, and this package invents none."""
    if spec.agent_count is None or spec.agent_count < 2:
        raise K26RlError(
            None,
            "artifact declares agent count %s; at agent count 1 "
            "observation channel names publish unqualified, so the "
            "artifact publishes no agent name to key the per-agent "
            "dictionaries by, and K26RlEnv is the shape that serves "
            "it" % spec.agent_count)


def agent_names(spec):
    """The published agent names, in agent-index order.

    Above one agent every observation channel name publishes
    qualified, as ``<agent>.<channel>``, so an agent's name is the
    qualifier its own observation slice's channels carry. Action
    channel names are published nowhere and no tag carries them, so an
    agent that declares no observation channel has no published name;
    such an artifact is refused rather than given an invented one.
    """
    names = []
    for agent, (offset, count) in enumerate(
            slices_by_agent(spec.agent_obs_slices, spec.agent_count)):
        if count == 0:
            raise K26RlError(
                None,
                "agent %d declares no observation channel, so the "
                "artifact publishes no name for it: an agent name "
                "reaches this package only as the qualifier on an "
                "observation channel name, and no tag carries an "
                "action channel name" % agent)
        found = set()
        for channel in range(offset, offset + count):
            text = spec.obs_channel_names.get(channel)
            if text is None:
                raise K26RlError(
                    None,
                    "observation channel %d carries no published name, "
                    "so agent %d's name cannot be read from it"
                    % (channel, agent))
            qualifier, sep, _rest = text.partition(".")
            if not sep or not qualifier:
                raise K26RlError(
                    None,
                    "observation channel %d publishes as %r, which "
                    "carries no <agent>.<channel> qualifier; above one "
                    "agent every observation channel name is qualified"
                    % (channel, text))
            found.add(qualifier)
        if len(found) != 1:
            raise K26RlError(
                None,
                "agent %d's observation channels [%d, %d) publish "
                "under %d different qualifiers %s; one agent's "
                "channels carry one name"
                % (agent, offset, offset + count, len(found),
                   sorted(found)))
        names.append(found.pop())
    if len(set(names)) != len(names):
        raise K26RlError(
            None, "two agents publish under one name: %s" % (names,))
    return names
