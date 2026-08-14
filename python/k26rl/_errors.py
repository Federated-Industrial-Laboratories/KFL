"""Exception types for the k26rl package.

Every message that reports an artifact status value is obtained from
the loaded artifact's own k26rl_status_str at raise time. The package
carries no status-message table, so a status value minted after this
package was written still decodes to its true name through the
artifact that minted it.
"""


class K26RlError(Exception):
    """Base error for the package.

    ``status`` is the artifact's status registry value behind the
    raise, or None for refusals raised on the Python side before any
    artifact call (a closed environment, a malformed spec blob, an
    ABI version mismatch).
    """

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


class K26RlSeedReuseError(K26RlError):
    """The artifact refused a seed its handle has already held."""


class K26RlOutputError(K26RlError):
    """Episode output was refused: the call was not at an episode
    boundary, or the output path already exists."""


class K26RlFaultError(K26RlError):
    """One or more environments faulted during a step.

    Not a call-status error: the step call itself completed and every
    other environment advanced before the raise. In the default fault
    mode this is raised after the step's results are formed; those
    results already report each fault as a truncation with the
    ``fault_code`` and ``fault_reason`` info keys set, so a catching
    consumer can inspect, record, and continue.

    Attributes:
        indices: faulted environment indices, ascending.
        codes: registry fault code per faulted environment.
        reasons: artifact-decoded reason per faulted environment.
        results: the completed step call's return value.
    """

    def __init__(self, indices, codes, reasons, results):
        detail = "; ".join(
            "environment %d: %s (status %d)" % (i, r, c)
            for i, c, r in zip(indices, codes, reasons)
        )
        super().__init__(None, "step faulted: " + detail)
        self.indices = tuple(indices)
        self.codes = tuple(codes)
        self.reasons = tuple(reasons)
        self.results = results
