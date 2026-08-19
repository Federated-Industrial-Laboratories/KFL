/* substep.c - how finely one attitude advance is subdivided.
 *
 * The advance integrates the body rate with an explicit first-order
 * step, so the rotation it resolves in one sub-interval is the whole
 * of its stability. A rate that turns the body a large angle in one
 * sub-interval leaves the gyroscopic term evaluated at an orientation
 * the body has already left, and the error feeds the rate that
 * produced it: the rate grows, the angle per sub-interval grows with
 * it, and a few sub-intervals later the state is not a number. A
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
 * sub-interval inside the bound removes the runaway: a body at ten
 * radians a second, given intervals of a tenth of a second, keeps its
 * angular momentum inside a factor of three over thirty seconds where
 * the unsubdivided advance has left the number line by sixteen. It
 * does not make a first-order step unconditionally stable, because no
 * step size does: the residual drift falls in proportion to the
 * sub-interval rather than to zero, and closing that is a
 * higher-order step and not a finer one. The subdivision is the
 * remedy for a state the interval cannot represent, which is the
 * failure a world's author cannot see coming; the accuracy of a step
 * that can represent it is a separate question with its own answer.
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
