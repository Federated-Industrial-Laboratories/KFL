"""Export a Stable-Baselines3 actor-critic policy to ``.k26pol``.

This module is the only part of the package that knows what
Stable-Baselines3 is. The format itself does not: :mod:`k26rl.policy`
takes plain numbers and writes bytes, and everything framework-shaped
is translated here.

What is exported, and what is not
---------------------------------

An actor-critic model carries more than a forward pass needs.

*The action network is exported*: the policy hidden layers and the
output layer that produces the action distribution's mean, in
evaluation order.

*The value network is not*. It is not on the action path; inference
and a viewer never call it, and carrying it would roughly double the
file for a quantity whose only consumer is a trainer that still holds
its own checkpoint. A ``.k26pol`` is an inference artifact, not a
resumable model, and that is a property of the format rather than an
omission here.

*The action log standard deviation is exported*, although a
deterministic evaluation does not read it. It is a learned parameter
that belongs to no layer, so a format carrying layers alone would
drop it with nothing saying so, and a policy sampled without it is
not the policy that was trained.

*The observation statistics are exported* when a running-statistics
wrapper is supplied. The trained network's input is the standardised
observation, not the world's; a policy exported without the
statistics it was trained against would load happily and act wrongly.

*The action bounds are exported* as the evaluated action's clamp,
because the framework's own deterministic prediction clips to the
action space before returning, and a policy that did not would emit
actions the trained one never emitted.

Refusals
--------

Anything the format cannot express is refused by name rather than
approximated: state-dependent exploration, squashed output, discrete
or mixed action spaces, non-flat observations, a feature extractor
that is not the identity, a recurrent policy, an activation outside
the format's closed list, and an agent with no measured observation
channel at all, which leaves a policy nothing it is allowed to read.

Measured channels that interleave with ground-truth ones are not a
refusal: the file names the channels a policy reads one by one, so a
world that senses more than one thing exports like any other.
"""

from . import policy as _policy
from . import _spec
from ._errors import K26RlError

#: Torch activation module names this format can express.
_ACTIVATIONS = {
    "Tanh": _policy.ACT_TANH,
    "ReLU": _policy.ACT_RELU,
    "Sigmoid": _policy.ACT_LOGISTIC,
    "Identity": _policy.ACT_IDENTITY,
}


def _refuse(detail):
    raise K26RlError(None, "policy export refused: " + detail)


def _linear_layer(module, activation):
    """One torch ``Linear`` as a format layer. The weight tensor is
    already output-major, which is the orientation the file declares,
    so nothing is transposed on the way out and a transposition
    cannot hide in the export."""
    weight = module.weight.detach().cpu().numpy()
    bias = (module.bias.detach().cpu().numpy() if module.bias is not None
            else [0.0] * weight.shape[0])
    return _policy.Layer(weight.tolist(), list(bias), activation)


def _walk_sequential(net):
    """A torch ``Sequential`` of alternating linear and activation
    modules as format layers. A module that is neither is refused: a
    normalisation or dropout layer changes the arithmetic and the
    format has no way to say so."""
    layers = []
    pending = None
    for module in net:
        name = type(module).__name__
        if name == "Linear":
            if pending is not None:
                layers.append(_linear_layer(pending, _policy.ACT_IDENTITY))
            pending = module
            continue
        if name in _ACTIVATIONS:
            if pending is None:
                _refuse("the policy network applies %s before any linear "
                        "layer" % name)
            layers.append(_linear_layer(pending, _ACTIVATIONS[name]))
            pending = None
            continue
        _refuse("the policy network carries a %s module, which this format "
                "cannot express" % name)
    if pending is not None:
        layers.append(_linear_layer(pending, _policy.ACT_IDENTITY))
    return layers


def _statistics(normaliser, input_width):
    """The observation standardisation from a running-statistics
    wrapper, or None when it standardises nothing."""
    if normaliser is None:
        return None
    if isinstance(normaliser, str):
        import pickle

        with open(normaliser, "rb") as handle:
            normaliser = pickle.load(handle)
    if not getattr(normaliser, "norm_obs", False):
        return None
    if getattr(normaliser, "norm_obs_keys", None):
        _refuse("the observation statistics are keyed by name, which this "
                "format's flat observation slice cannot express")
    rms = normaliser.obs_rms
    mean = list(rms.mean)
    if len(mean) != input_width:
        _refuse("the observation statistics carry %d channels for a policy "
                "input of %d" % (len(mean), input_width))
    return _policy.Standardisation(mean, list(rms.var),
                                   float(normaliser.epsilon),
                                   float(normaliser.clip_obs))


def _slices(spec, agent):
    """The agent's slice geometry, and the channels of it a policy
    reads, from a parsed artifact spec; None when no spec is given,
    which leaves the caller its single-agent defaults."""
    if spec is None:
        return None
    count = spec.agent_count
    if count is None or count < 1:
        _refuse("the artifact spec declares agent count %r" % (count,))
    if not 0 <= agent < count:
        _refuse("agent %d is outside the artifact's agent count %d"
                % (agent, count))
    obs = {a: (o, c) for a, o, c in spec.agent_obs_slices}
    act = {a: (o, c) for a, o, c in spec.agent_act_slices}
    if count == 1:
        obs.setdefault(0, (0, spec.obs_total))
        act.setdefault(0, (0, spec.act_total))
    if agent not in obs or agent not in act:
        _refuse("the artifact spec carries no observation or action slice "
                "for agent %d" % agent)
    return {
        "agent_count": count,
        "agent_index": agent,
        "obs_total": spec.obs_total,
        "act_total": spec.act_total,
        "obs_offset": obs[agent][0],
        "obs_width": obs[agent][1],
        "obs_channels": _policy_channels(spec, agent, *obs[agent]),
        "act_offset": act[agent][0],
        "act_width": act[agent][1],
    }


def _policy_channels(spec, agent, offset, count):
    """The channels of the agent's observation slice a policy reads:
    its measured ones, ascending, as the file lists them.

    The trained network's input is the measured channels, not the
    whole slice, so those are what the file names; a file naming the
    whole slice would have the inference tier feed a policy the ground
    truth beside each measurement, in place of the measurements it was
    trained on, with the widths agreeing and nothing raised.

    Any arrangement of measured and ground-truth channels is
    expressible, interleaved ones included, because the file lists
    channels rather than one run of them. An agent with no measured
    channel is refused: there is nothing a policy may read."""
    measured, _truth = _spec.split_channels(spec, offset, count)
    if not measured:
        _refuse("agent %d declares no measured observation channel, so "
                "there is nothing a policy may read" % agent)
    return measured


def export_policy(model, path, spec=None, agent=0, normaliser=None,
                  provenance=None):
    """Write a trained actor-critic model to a ``.k26pol`` file.

    Args:
        model: the trained model. Its policy must be the on-policy
            actor-critic shape with a diagonal Gaussian action
            distribution.
        path: the file to write.
        spec: the artifact's parsed spec, as an environment's
            ``env_spec`` attribute carries it. Without one the export
            declares the single-agent geometry, in which the policy's
            own widths are the environment's totals; with one, the
            declared slice geometry is written and a later load can
            check it field by field. The channels the file names are
            the agent's measured ones, which is what the network was
            trained on and what the inference tier must feed it.
        agent: which agent of a multi-agent artifact this policy
            drives.
        normaliser: the running observation statistics the model was
            trained against, either the wrapper object or the path to
            the file it was saved to. Omitting it for a model that
            was trained with normalisation produces a policy that
            loads and acts wrongly, so it is omitted only for a model
            that was trained without.
        provenance: text recorded inside the hashed region. Defaults
            to the model's class name and the package version.

    Returns:
        The bytes written.
    """
    pol = getattr(model, "policy", None)
    if pol is None:
        _refuse("the model carries no policy")
    if getattr(model, "use_sde", False) or getattr(pol, "use_sde", False):
        _refuse("the policy uses state-dependent exploration, whose action "
                "noise is a network this format has no layer for")
    if getattr(pol, "squash_output", False):
        _refuse("the policy squashes and rescales its output, an affine "
                "transform this format's activation list cannot express")
    if hasattr(pol, "lstm_actor"):
        _refuse("the policy is recurrent; version 1 of this format "
                "expresses feedforward networks only")
    dist = type(getattr(pol, "action_dist", None)).__name__
    if dist != "DiagGaussianDistribution":
        _refuse("the policy's action distribution is %s; version 1 of this "
                "format expresses a diagonal Gaussian over a continuous "
                "action vector, whose mean is the deterministic action"
                % dist)

    space = model.action_space
    if type(space).__name__ != "Box" or len(space.shape) != 1:
        _refuse("the action space is %r; this format's action slice is a "
                "flat vector of continuous channels" % (space,))
    obs_space = model.observation_space
    if type(obs_space).__name__ != "Box" or len(obs_space.shape) != 1:
        _refuse("the observation space is %r; this format's observation "
                "slice is a flat vector" % (obs_space,))
    extractor = type(getattr(pol, "features_extractor", None)).__name__
    if extractor != "FlattenExtractor":
        _refuse("the policy extracts features with %s, which is not the "
                "identity this format assumes between the observation "
                "slice and the first layer" % extractor)

    layers = _walk_sequential(pol.mlp_extractor.policy_net)
    layers.append(_linear_layer(pol.action_net, _policy.ACT_IDENTITY))
    # How many channels the policy reads, which is not the width of
    # the slice they come out of once a world declares a sensor.
    input_width = layers[0].in_width
    act_width = layers[-1].out_width
    if input_width != int(obs_space.shape[0]):
        _refuse("the first layer takes %d inputs for an observation space of "
                "%d" % (input_width, int(obs_space.shape[0])))
    if act_width != int(space.shape[0]):
        _refuse("the output layer gives %d values for an action space of %d"
                % (act_width, int(space.shape[0])))

    log_std = getattr(pol, "log_std", None)
    if log_std is not None:
        log_std = list(log_std.detach().cpu().numpy())

    geometry = _slices(spec, agent)
    if geometry is None:
        geometry = {"agent_count": 1, "agent_index": 0,
                    "obs_total": input_width, "act_total": act_width,
                    "obs_offset": 0, "obs_width": input_width,
                    "obs_channels": list(range(input_width)),
                    "act_offset": 0, "act_width": act_width}
    if len(geometry["obs_channels"]) != input_width:
        _refuse("the artifact gives agent %d %d measured observation "
                "channels, and the policy takes %d inputs"
                % (agent, len(geometry["obs_channels"]), input_width))
    if geometry["act_width"] != act_width:
        _refuse("the artifact gives agent %d an action slice of %d, and the "
                "policy gives %d outputs"
                % (agent, geometry["act_width"], act_width))

    if provenance is None:
        from . import __version__

        provenance = "%s exported by k26rl %s" % (type(model).__name__,
                                                  __version__)

    return _policy.write(
        path, layers,
        agent_count=geometry["agent_count"],
        agent_index=geometry["agent_index"],
        obs_total=geometry["obs_total"], act_total=geometry["act_total"],
        obs_offset=geometry["obs_offset"], obs_width=geometry["obs_width"],
        obs_channels=geometry["obs_channels"],
        act_offset=geometry["act_offset"],
        standardisation=_statistics(normaliser, input_width),
        log_std=log_std,
        action_bounds=(list(space.low), list(space.high)),
        provenance=provenance)
