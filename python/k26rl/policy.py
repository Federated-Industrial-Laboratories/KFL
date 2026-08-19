"""Writer for the trained-policy file format.

A ``.k26pol`` file holds a feedforward policy: the weights that turn
one agent's observation slice into that agent's action slice, plus
the declarations a loader needs to check the file against the world
it was trained on and to reproduce its arithmetic. The layout, the
evaluation rule and the identity rule are the ones the C header
states; this module writes exactly those bytes and reads nothing.

It knows nothing about any training framework. It takes plain
numbers, in the order the format declares them, and returns bytes.
The module that knows what produced the weights is :mod:`k26rl.sb3`,
and it is the only one.

The file is an inference artifact, not a training checkpoint: it
carries what the action path needs and nothing else. A value head, an
optimiser state and a replay buffer are all absent by design, and a
trainer that wants to resume keeps its own checkpoint.

No numpy: every value is taken through ``float``, so numpy arrays,
lists and tuples all serialise the same way, and the package's
binding layers stay importable where only the marshalling is wanted.
"""

import hashlib
import math
import struct

from ._errors import K26RlError

MAGIC = b"K26POL\0\0"
SUFFIX = ".k26pol"
FORMAT_VERSION = 1
HEADER_BYTES = 96
DIGEST_OFFSET = 16
DIGEST_BYTES = 32

MAX_LAYERS = 64
MAX_WIDTH = 65536
MAX_PROVENANCE = 4096

FLAG_STANDARDISE = 1 << 0
FLAG_LOG_STD = 1 << 1
FLAG_CLAMP = 1 << 2

ACT_IDENTITY = 0
ACT_TANH = 1
ACT_RELU = 2
ACT_LOGISTIC = 3

ACTIVATIONS = (ACT_IDENTITY, ACT_TANH, ACT_RELU, ACT_LOGISTIC)

ACTIVATION_NAMES = {
    "identity": ACT_IDENTITY,
    "linear": ACT_IDENTITY,
    "tanh": ACT_TANH,
    "relu": ACT_RELU,
    "logistic": ACT_LOGISTIC,
    "sigmoid": ACT_LOGISTIC,
}


class Layer:
    """One dense layer: an output-major weight matrix, a bias vector,
    and an activation from the format's closed list.

    ``weights`` is a sequence of ``out_width`` rows, each of
    ``in_width`` numbers: row ``o`` holds the weights that reach
    output ``o``. That is the orientation the file declares and the
    orientation a row-major framework weight matrix already has, so
    nothing here transposes anything and no transposition can hide.
    """

    __slots__ = ("weights", "biases", "activation", "in_width", "out_width")

    def __init__(self, weights, biases, activation):
        rows = [[float(w) for w in row] for row in weights]
        self.biases = [float(b) for b in biases]
        self.activation = _activation_code(activation)
        self.out_width = len(rows)
        if self.out_width == 0:
            _refuse("a layer declares no output")
        widths = {len(row) for row in rows}
        if len(widths) != 1:
            _refuse("a layer's weight rows are of %d different lengths %s"
                    % (len(widths), sorted(widths)))
        self.in_width = widths.pop()
        if self.in_width == 0:
            _refuse("a layer declares no input")
        if len(self.biases) != self.out_width:
            _refuse("a layer declares %d outputs and %d biases"
                    % (self.out_width, len(self.biases)))
        _check_width(self.in_width, "layer input width")
        _check_width(self.out_width, "layer output width")
        self.weights = rows


class Standardisation:
    """The affine standardisation applied to the observation slice
    before the first layer: per channel a mean and a variance, then
    one epsilon and one clip shared by every channel.

    These numbers are part of the policy, not of the environment. The
    network's input is not the world's observation, so a file that
    left them outside itself would not describe a reproducible
    policy: loaded without them a policy produces a confident wrong
    action rather than an error, and loaded with somebody else's it
    produces a different one silently.
    """

    __slots__ = ("mean", "variance", "epsilon", "clip")

    def __init__(self, mean, variance, epsilon, clip):
        self.mean = [float(v) for v in mean]
        self.variance = [float(v) for v in variance]
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        if len(self.mean) != len(self.variance):
            _refuse("standardisation declares %d means and %d variances"
                    % (len(self.mean), len(self.variance)))
        for v in self.mean:
            if not math.isfinite(v):
                _refuse("a standardisation mean is not finite")
        for v in self.variance:
            if not math.isfinite(v) or v < 0.0:
                _refuse("a standardisation variance is negative or not "
                        "finite")
        if not math.isfinite(self.epsilon) or self.epsilon < 0.0:
            _refuse("the standardisation epsilon is negative or not finite")
        if math.isnan(self.clip) or self.clip <= 0.0:
            _refuse("the standardisation clip is zero, negative, or not a "
                    "number; positive infinity is how a policy declares "
                    "that it clips nothing")
        for m, v in zip(self.mean, self.variance):
            scale = math.sqrt(v + self.epsilon)
            if not math.isfinite(scale) or scale <= 0.0:
                _refuse("a standardisation variance and epsilon give the "
                        "unusable scale %r" % (scale,))


def _refuse(detail):
    raise K26RlError(None, "policy export refused: " + detail)


def _check_width(value, what):
    if value < 1 or value > MAX_WIDTH:
        _refuse("%s is %d, outside 1 to %d" % (what, value, MAX_WIDTH))


def _activation_code(activation):
    """An activation as its format code. A name is accepted for the
    caller's readability; anything outside the closed list is
    refused here rather than written for a loader to refuse."""
    if isinstance(activation, str):
        code = ACTIVATION_NAMES.get(activation.lower())
        if code is None:
            _refuse("activation %r is outside the format's closed list %s"
                    % (activation, sorted(ACTIVATION_NAMES)))
        return code
    code = int(activation)
    if code not in ACTIVATIONS:
        _refuse("activation code %d is outside the format's closed list %s"
                % (code, sorted(ACTIVATIONS)))
    return code


def _bounds(action_bounds, act_width):
    lower = [float(v) for v in action_bounds[0]]
    upper = [float(v) for v in action_bounds[1]]
    if len(lower) != act_width or len(upper) != act_width:
        _refuse("action bounds declare %d lower and %d upper values for %d "
                "action channels" % (len(lower), len(upper), act_width))
    for lo, hi in zip(lower, upper):
        if math.isnan(lo) or math.isnan(hi) or not lo <= hi:
            _refuse("an action bound pair is [%r, %r]" % (lo, hi))
    return lower, upper


def encode(layers, agent_count=1, agent_index=0, obs_total=None,
           act_total=None, obs_offset=0, act_offset=0, standardisation=None,
           log_std=None, action_bounds=None, provenance=""):
    """The file's bytes.

    Args:
        layers: the dense layers in evaluation order. The first
            layer's input width is the agent's observation slice
            width and the last layer's output width is its action
            slice width, so neither is restated and neither can
            disagree with the weights.
        agent_count: agents the artifact declares.
        agent_index: which of them this policy drives.
        obs_total: observation doubles per environment; defaults to
            the policy's own observation slice width, which is the
            single-agent case.
        act_total: action doubles per environment; defaults likewise.
        obs_offset: where the agent's observation slice starts.
        act_offset: where the agent's action slice starts.
        standardisation: a :class:`Standardisation`, or None.
        log_std: per action channel log standard deviation of the
            policy's action distribution, or None. Deterministic
            evaluation does not read it; it is carried because it is
            a learned parameter that belongs to no layer, and an
            export that dropped it would be lossy with nothing
            saying so.
        action_bounds: ``(lower, upper)`` sequences the evaluated
            action is clamped to, or None for no clamp.
        provenance: UTF-8 text recording what produced the file. It
            sits inside the hashed region, so a relabelled policy is
            a different policy.

    Returns:
        The complete file as ``bytes``, digest included.
    """
    layers = list(layers)
    if not layers:
        _refuse("a policy declares no layer")
    if len(layers) > MAX_LAYERS:
        _refuse("a policy declares %d layers, above the format's %d"
                % (len(layers), MAX_LAYERS))
    for i in range(1, len(layers)):
        if layers[i].in_width != layers[i - 1].out_width:
            _refuse("layer %d takes %d inputs but layer %d gives %d outputs"
                    % (i, layers[i].in_width, i - 1, layers[i - 1].out_width))

    obs_width = layers[0].in_width
    act_width = layers[-1].out_width
    obs_total = obs_width if obs_total is None else int(obs_total)
    act_total = act_width if act_total is None else int(act_total)
    agent_count = int(agent_count)
    agent_index = int(agent_index)
    obs_offset = int(obs_offset)
    act_offset = int(act_offset)
    _check_width(obs_total, "observation total")
    _check_width(act_total, "action total")
    if agent_count < 1 or not 0 <= agent_index < agent_count:
        _refuse("agent index %d is outside an agent count of %d"
                % (agent_index, agent_count))
    if obs_offset < 0 or obs_offset + obs_width > obs_total:
        _refuse("observation slice [%d, %d) runs past the total %d"
                % (obs_offset, obs_offset + obs_width, obs_total))
    if act_offset < 0 or act_offset + act_width > act_total:
        _refuse("action slice [%d, %d) runs past the total %d"
                % (act_offset, act_offset + act_width, act_total))

    flags = 0
    body = bytearray()

    text = provenance.encode("utf-8") if isinstance(provenance, str) \
        else bytes(provenance)
    if len(text) > MAX_PROVENANCE:
        _refuse("provenance is %d bytes, above the format's %d"
                % (len(text), MAX_PROVENANCE))
    body += text

    for layer in layers:
        body += struct.pack("<IIHH", layer.in_width, layer.out_width,
                            layer.activation, 0)
        for row in layer.weights:
            body += struct.pack("<%dd" % layer.in_width, *row)
        body += struct.pack("<%dd" % layer.out_width, *layer.biases)

    if standardisation is not None:
        if len(standardisation.mean) != obs_width:
            _refuse("standardisation declares %d channels for an "
                    "observation slice of %d"
                    % (len(standardisation.mean), obs_width))
        flags |= FLAG_STANDARDISE
        body += struct.pack("<%dd" % obs_width, *standardisation.mean)
        body += struct.pack("<%dd" % obs_width, *standardisation.variance)
        body += struct.pack("<dd", standardisation.epsilon,
                            standardisation.clip)

    if log_std is not None:
        values = [float(v) for v in log_std]
        if len(values) != act_width:
            _refuse("log standard deviation declares %d values for %d "
                    "action channels" % (len(values), act_width))
        for v in values:
            if not math.isfinite(v):
                _refuse("a log standard deviation is not finite")
        flags |= FLAG_LOG_STD
        body += struct.pack("<%dd" % act_width, *values)

    if action_bounds is not None:
        lower, upper = _bounds(action_bounds, act_width)
        flags |= FLAG_CLAMP
        body += struct.pack("<%dd" % act_width, *lower)
        body += struct.pack("<%dd" % act_width, *upper)

    header = bytearray(MAGIC)
    header += struct.pack("<II", FORMAT_VERSION, HEADER_BYTES)
    header += b"\0" * DIGEST_BYTES
    header += struct.pack("<IIIIIIIIIII", flags, agent_count, agent_index,
                          obs_total, act_total, obs_offset, obs_width,
                          act_offset, act_width, len(layers), len(text))
    header += struct.pack("<I", 0)
    if len(header) != HEADER_BYTES:
        raise AssertionError("policy header assembled to %d bytes, not %d"
                             % (len(header), HEADER_BYTES))

    # The identity is a digest over the file's own bytes with the
    # digest field read as zero, so the provenance and every declared
    # field are inside the hashed region.
    blob = bytearray(header) + body
    digest = hashlib.sha256(bytes(blob)).digest()
    blob[DIGEST_OFFSET:DIGEST_OFFSET + DIGEST_BYTES] = digest
    return bytes(blob)


def write(path, layers, **kwargs):
    """Encode a policy and write it to ``path``.

    Returns the bytes written, so a caller that wants the file's
    digest need not read it back.
    """
    blob = encode(layers, **kwargs)
    with open(path, "wb") as handle:
        handle.write(blob)
    return blob


def digest_of(blob):
    """The digest a policy file's bytes carry, recomputed from the
    bytes themselves. A caller checking a file it did not write
    compares this against ``blob[16:48]``."""
    if len(blob) < HEADER_BYTES:
        _refuse("a policy file is %d bytes, below the header's %d"
                % (len(blob), HEADER_BYTES))
    zeroed = bytearray(blob)
    zeroed[DIGEST_OFFSET:DIGEST_OFFSET + DIGEST_BYTES] = b"\0" * DIGEST_BYTES
    return hashlib.sha256(bytes(zeroed)).digest()
