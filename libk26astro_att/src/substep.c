/* substep.c - how finely one attitude advance is subdivided.
 *
 * The rotation the advance resolves in one sub-interval is the whole
 * of what its step can represent. A rate that turns the body a large
 * angle in one sub-interval asks the integration to follow a curve it
 * has only sampled the start of, and the error feeds the rate that
 * produced it: the rate grows, the angle per sub-interval grows with
 * it, and enough sub-intervals later the state is not a number. A
 * caller cannot avoid that by declaring a subdivision, because the
 * rate a policy will command is not knowable when the program is
 * written. So the advance subdivides itself, by the rate it is
 * actually given.
 *
 * The whole of the rule is one line of arithmetic, and that is
 * deliberate. The count decides how much work the integration does,
 * which makes it part of the physics: the same state must give the
 * same count on every run, in every process and on every machine, or
 * two runs of one artifact stop agreeing bit for bit. A rule small
 * enough to read is a rule whose purity can be seen rather than
 * argued, and this file is kept to that one function so that a gate
 * can replace it at link time and prove the property has teeth.
 *
 * What the count may read: the body rate and the interval, both
 * passed in. What it may not read, and does not: a clock, an elapsed
 * time, a budget of work already spent, an error accumulated across
 * calls, an environment variable, an address, or anything else that
 * two runs of one artifact could disagree about. It holds no state
 * between calls and it writes none.
 *
 * What this buys, stated as measured. Holding the angle turned in one
 * sub-interval inside the bound is what keeps the rotation
 * representable at all. On the first-order step this rule was built
 * against, that was the difference between an answer and a state that
 * was not a number: a body at ten radians a second, given intervals
 * of a tenth of a second, kept its angular momentum inside a factor
 * of three over thirty seconds where the unsubdivided advance had
 * left the number line by sixteen.
 *
 * The step underneath is no longer that step, and the division of
 * labour is now clean. The step's own accuracy is the body library's
 * and is stated there: over nine hundred intervals of a tenth of a
 * second, torque free, on a crew vehicle's tensor, the drift in
 * angular momentum runs from 2.6e-09 at one radian a second to
 * 7.7e-08 at thirty, against 0.28 to 41 before. What this rule
 * contributes is that the sub-interval stays inside the range where
 * that accuracy is the accuracy: a step is only as good as the
 * rotation it samples, and a caller who cannot know what rate a
 * policy will command cannot keep it there by declaring a number.
 *
 * The arithmetic is multiplication, division, square root and a
 * ceiling. All four are correctly rounded under IEEE-754, so the same
 * inputs give the same count on any host that keeps to it, which is
 * the same restriction the swept contact kernels observe and is a
 * stronger claim than reproducibility on one machine.
 */
#include "k26astro_att/att.h"

#include <math.h>

int k26astro_att_substep_count(K26V3 omega_body, double dt)
{
    if (!(dt > 0.0) || !isfinite(dt)) return 1;

    double w2 = omega_body.x * omega_body.x
              + omega_body.y * omega_body.y
              + omega_body.z * omega_body.z;
    double theta = sqrt(w2) * dt;

    /* A rate that is not a number cannot be resolved by subdividing
     * the interval it turns through, so the interval is taken whole
     * and the advance reports the divergence it finds. Spending the
     * ceiling of sub-intervals on a state that no count can rescue
     * would turn one faulted step into the slowest step in the run. */
    if (!isfinite(theta)) return 1;
    if (theta <= K26ASTRO_ATT_SUBSTEP_ANGLE) return 1;

    double want = ceil(theta / K26ASTRO_ATT_SUBSTEP_ANGLE);
    if (!(want < (double)K26ASTRO_ATT_SUBSTEP_MAX)) {
        return K26ASTRO_ATT_SUBSTEP_MAX;
    }
    return (int)want;
}
