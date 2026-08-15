/* k26astro_coll: swept contact detection between the sub-advances of
 * an exact step.
 *
 * This is not the runtime's close-encounter detector and does not
 * touch it. That one is a gravitational test on mutual Hill radii
 * which never reads a body's radius and exists to hand integration to
 * a different method. Contact is a separate question and is answered
 * by separate machinery; the two are named apart so that nobody later
 * assumes one covers the other.
 *
 * Arithmetic. Every kernel here uses addition, subtraction,
 * multiplication, division and square root, and nothing else. Those
 * five are correctly rounded under IEEE-754, so a kernel's result is
 * reproducible across platforms and not only across runs on one.
 * Comparison, selection and negation appear freely: they are exact
 * and round nothing, so they do not weaken that claim. No library
 * function beyond sqrt is called, in particular no fabs, fmin, fmax,
 * or any transcendental.
 */
#ifndef K26ASTRO_COLL_H
#define K26ASTRO_COLL_H

#include <k26m3d.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Shapes ------------------------------------------------------ */

typedef enum {
    K26ASTRO_COLL_SPHERE  = 1,
    K26ASTRO_COLL_CAPSULE = 2,
    K26ASTRO_COLL_BOX     = 3
} K26AstroCollKind;

/* One collision primitive, in the frame of the body that carries it,
 * with the component placement that positioned it already baked in.
 *
 * `centre` is the primitive's centre. `axis` are its three local axes,
 * unit and right-handed; a sphere ignores them, a capsule uses the
 * third as its segment direction. `half` is the box's half-extent
 * along each local axis; a sphere uses half[0] as its radius; a
 * capsule uses half[0] as its radius and half[2] as half the distance
 * between its two segment endpoints. */
typedef struct {
    K26AstroCollKind kind;
    K26V3            centre;
    K26V3            axis[3];
    double           half[3];
} K26AstroCollShape;

/* One collidable body over one sub-advance. The translation is taken
 * as linear between the endpoint positions and the orientation is
 * held at the interval's start, which is what makes every kernel's
 * relative motion a straight line and its contact time exactly
 * solvable. `pos0` and `pos1` are the body's positions at the start
 * and end of the interval, in a frame common to the pair and near
 * enough to it that a difference of positions is not a difference of
 * large numbers. `shapes` point into storage the caller owns and are
 * never retained past a call. */
typedef struct {
    K26V3                    pos0;
    K26V3                    pos1;
    K26Quat                  orientation;
    const K26AstroCollShape *shapes;
    int                      n_shapes;
    double                   bound_radius;   /* covers every shape */
    K26V3                    vel0;           /* start of the interval */
    K26V3                    vel1;           /* end of the interval */

    /* What resolution needs, and nothing more. `com_offset` is the
     * centre of mass in the body frame, so a body whose mass is not
     * centred on its origin spins about the right point; the caller
     * supplies it because the assembly derived it. A mass of zero or
     * less means immovable, which is how a body a task treats as
     * fixed is expressed without a special case. */
    double                   mass;           /* kg */
    K26V3                    com_offset;
    K26V3                    omega;          /* body frame, rad/s */
    K26M3                    inv_inertia;    /* body frame */
} K26AstroCollBody;

/* ---- Reports ----------------------------------------------------- */

typedef struct {
    int    hit;            /* 0 or 1 */
    double time;           /* fraction of the interval, in [0, 1] */
    K26V3  normal;         /* unit, from the first body to the second */
    K26V3  point;          /* contact point at the impact configuration */
    int    shape_a;        /* index into the first body's shapes */
    int    shape_b;
} K26AstroCollHit;

typedef struct {
    int    hit;
    double time;
    K26V3  normal;
    K26V3  point;
    int    body_a;
    int    body_b;
    int    shape_a;
    int    shape_b;
    /* Closing speed along the normal, metres per second, positive
     * when the pair is approaching. Taken at the start of the
     * interval: by its end the pair has already passed through the
     * contact, so the velocity there is a state the resolution
     * discards rather than one the report should carry. */
    double speed;
} K26AstroCollContact;

typedef enum {
    K26ASTRO_COLL_OK       = 0,
    K26ASTRO_COLL_E_NULL   = 1,
    K26ASTRO_COLL_E_BAD_DT = 2,
    K26ASTRO_COLL_E_LIMIT  = 3
} K26AstroCollStatus;

const char *k26astro_coll_status_str(K26AstroCollStatus s);

/* ---- Kernels ----------------------------------------------------- *
 *
 * Each takes two primitives already expressed in a frame the pair
 * shares, the second one's centre offset from the first at the start
 * of the interval, and the second one's displacement relative to the
 * first over the whole interval. Each reports the least interval
 * fraction at which the two touch.
 *
 * A pair already overlapping at the start reports a hit at time zero.
 * That is not an error: a reset may place two bodies overlapping, and
 * the first step is then expected to report contact rather than to
 * search for one.
 */

/* Two spheres. The squared separation is a quadratic in the interval
 * fraction, solved for its first root at the sum of the radii. */
int k26astro_coll_sweep_sphere_sphere(double ra, double rb,
                                      K26V3 offset, K26V3 disp,
                                      K26AstroCollHit *out);

/* A moving point against a static segment, with a combined radius:
 * the kernel a sphere against a capsule reduces to exactly, and the
 * one a capsule against a capsule is built from. `a` and `b` are the
 * segment's endpoints. */
int k26astro_coll_sweep_point_segment(K26V3 a, K26V3 b, double radius,
                                      K26V3 offset, K26V3 disp,
                                      K26AstroCollHit *out);

/* Two capsules, each a segment and a radius. Exact: the closest
 * approach of two segments is attained either between their interiors
 * or at an endpoint of one against the other, and both cases are
 * solved in closed form. A sphere is the degenerate case whose two
 * endpoints coincide, so this kernel also serves sphere against
 * capsule. */
int k26astro_coll_sweep_capsule_capsule(K26V3 a0, K26V3 a1, double ra,
                                        K26V3 b0, K26V3 b1, double rb,
                                        K26V3 offset, K26V3 disp,
                                        K26AstroCollHit *out);

/* The separating-axis test carried over the interval, for any pair in
 * which at least one primitive is a box. Along each candidate axis
 * the two projections are intervals moving at a constant relative
 * rate, which gives an entry time and an exit time; the impact time
 * is the greatest entry time, and there is no contact when that
 * exceeds the least exit time or falls outside the interval.
 *
 * Candidate axes are the boxes' face normals, a capsule's own axis,
 * and the cross products of the pairs of directions, in a fixed
 * order. A cross product of a near-parallel pair is skipped below the
 * stated bound, which can only report a contact that a full test
 * would separate and never miss one that a full test would find.
 *
 * For a sphere or a capsule against a box the axis set is the box's
 * own directions and, for a capsule, its axis and the cross products.
 * That set is complete for a box against a box and conservative for a
 * round primitive against a box's corner or edge region, where the
 * true separating axis runs from the feature to the round
 * primitive's centre and is not among the candidates. The test can
 * therefore report contact slightly early there. It is documented
 * rather than hidden because a conservative contact time is a
 * different number, not a wrong answer about whether contact
 * happened, and reporting one early is the safe direction. */
int k26astro_coll_sweep_sat(const K26AstroCollShape *a,
                            const K26AstroCollShape *b,
                            K26V3 offset, K26V3 disp,
                            K26AstroCollHit *out);

/* The kernel this pair of kinds needs, dispatched on the two kinds in
 * a fixed order. The primitives are in a shared frame. */
int k26astro_coll_sweep_pair(const K26AstroCollShape *a,
                             const K26AstroCollShape *b,
                             K26V3 offset, K26V3 disp,
                             K26AstroCollHit *out);

/* Below this squared length a cross product of two directions is
 * treated as degenerate and its axis is skipped. The two directions
 * are unit, so this is the square of the sine of the angle between
 * them: the bound corresponds to about six ten-thousandths of a
 * degree, small enough that a genuinely separating edge-edge axis is
 * never discarded at any scale this capability runs at, and large
 * enough that the axis is never normalised from noise. */
#define K26ASTRO_COLL_PARALLEL_EPS2 1.0e-16

/* ---- The pass ---------------------------------------------------- */

/* Test every ordered pair of collidable bodies, in declaration order,
 * and report the single contact that governs the sub-advance.
 *
 * The broadphase is the swept bounding spheres with each radius
 * inflated by a curvature margin, half the magnitude of the relative
 * acceleration over the interval times the interval squared, the
 * relative acceleration taken as the change in relative velocity
 * across the interval divided by the interval. That margin is what
 * makes a linear sweep conservative against the true curved path.
 *
 * When more than one primitive pair reports a contact, the reported
 * one is the least impact time; ties break on the first body index,
 * then the second, then the first primitive index, then the second,
 * which is a total order over the candidates and leaves nothing to
 * iteration accident.
 *
 * `dt` is the sub-advance duration in seconds and is used only for
 * the margin and for the reported speed. Allocates nothing. */
K26AstroCollStatus k26astro_coll_pass(const K26AstroCollBody *bodies,
                                      int n_bodies, double dt,
                                      K26AstroCollContact *out);

/* The curvature margin the broadphase inflates each bounding sphere
 * by: half the magnitude of the relative acceleration over the
 * interval times the interval squared, with the acceleration taken as
 * the change in relative velocity across the interval divided by it,
 * which reduces to half the change in relative speed times the
 * interval. That is the bound on how far the true curved path can
 * depart from the straight one the sweep assumes.
 *
 * It is exposed because it cannot be measured through the pass. With
 * bounds that honestly cover their primitives, any pair the
 * narrowphase reports is a pair whose bounds already overlap, so
 * inflating them changes no outcome the pass can report and a gate
 * driving the pass cannot tell a correct margin from none at all.
 * The value is therefore gated here directly, at the one place it is
 * computed. */
double k26astro_coll_curvature_margin(K26V3 dv, double dt);

/* ---- Resolution -------------------------------------------------- */

/* Arrest: place the pair at the impact configuration the sweep itself
 * computed, and remove the pair's relative translational velocity by
 * a momentum-conserving merge. Writes the two bodies' positions and
 * velocities at the impact configuration.
 *
 * The positions written are the linear interpolation the sweep used,
 * so the reported contact and the recorded state agree exactly rather
 * than nearly. */
K26AstroCollStatus k26astro_coll_arrest(const K26AstroCollBody *a,
                                        const K26AstroCollBody *b,
                                        double time,
                                        K26V3 *pos_a, K26V3 *vel_a,
                                        K26V3 *pos_b, K26V3 *vel_b);

/* Bounce: one impulse at the impact configuration along the contact
 * normal, with a restitution coefficient in the closed unit interval.
 * Writes the two bodies' positions and post-impulse velocities.
 *
 * The friction coefficient is Coulomb: the tangential impulse
 * opposes the sliding at the contact point, and its magnitude is
 * whatever would stop the sliding outright, or the coefficient times
 * the normal impulse, whichever is smaller. Below that bound the
 * surfaces grip and above it they slide, which is the whole of the
 * model and is why a single coefficient is enough.
 *
 * The impulse uses the effective mass at the contact point, so an
 * off-centre impact spins the body by the amount the geometry gives
 * rather than by too much. Omitting that term is the recorded defect
 * in the prior art this capability deliberately does not carry.
 *
 * Linear momentum is conserved exactly by construction: the two
 * bodies receive equal and opposite impulses. Angular momentum about
 * any fixed point follows from the same construction, since the two
 * impulses act at one common point. A separating pair is left alone,
 * so a contact reported by a conservative kernel while the pair is
 * already moving apart does not inject energy. */
K26AstroCollStatus k26astro_coll_bounce(const K26AstroCollBody *a,
                                        const K26AstroCollBody *b,
                                        const K26AstroCollContact *hit,
                                        double restitution,
                                        double friction,
                                        K26V3 *pos_a, K26V3 *vel_a,
                                        K26V3 *omega_a,
                                        K26V3 *pos_b, K26V3 *vel_b,
                                        K26V3 *omega_b);

#ifdef __cplusplus
}
#endif

#endif /* K26ASTRO_COLL_H */
