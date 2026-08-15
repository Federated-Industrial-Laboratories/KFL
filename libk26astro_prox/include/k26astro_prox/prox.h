/* k26astro_prox: relative motion between two craft in proximity.
 *
 * Two things live here, and they are deliberately not the same thing.
 *
 * The frame and the relative state are measurement. They answer "where
 * is the target, from where the chief sits, and how fast is it
 * moving", in the chief's local-vertical local-horizontal axes, and
 * they are what a rendezvous policy reads every step. They are exact
 * for any orbit, perturbed or not, because they are algebra on the
 * states the integrator already produced.
 *
 * The Clohessy-Wiltshire propagator is a tool, not the dynamics.
 * Nothing in this capability integrates it. It exists to construct an
 * initial condition from a declared hold point, so that a reset can
 * place a chaser on a stated standoff rather than at a hand-computed
 * state vector, and to give the gates an analytic trajectory to
 * measure the frame and the relative state against. Running a
 * linearised model beside the integrated one as physics would give one
 * scenario two dynamics whose disagreement nobody owns.
 *
 * Arithmetic and determinism. The frame and the relative state use
 * addition, subtraction, multiplication, division and square root, and
 * nothing else; those five are correctly rounded under IEEE-754, so
 * this path is reproducible across platforms and not only across runs
 * on one. The state transition matrix calls the platform's sine and
 * cosine, which are not correctly rounded, so it carries the per-binary
 * claim only. That costs nothing, because the matrix is not on the
 * stepping path: the hold-point construction below reaches its answer
 * without a transcendental, and the matrix is otherwise the gates'.
 *
 * Provenance. The linearised equations of relative motion about a
 * circular reference orbit are
 *
 *     x'' = 3 n^2 x + 2 n y',      y'' = -2 n x',      z'' = -n^2 z
 *
 * with x radial, y along-track, z along the orbital angular momentum,
 * and n the mean motion. They are derived in `src/cw.c` from the
 * rotating-frame acceleration and the first-order expansion of the
 * gravity difference, and they agree with the form published for the
 * Clohessy-Wiltshire equations. The state transition matrix below is
 * solved from those equations in that file rather than transcribed:
 * every entry is obtained by integrating them, and the result is
 * checked in tests/test_prox_cw.c against the matrix exponential of
 * the system matrix, against a unit determinant, against the identity
 * at zero interval, and against the semigroup property. The reference
 * for the equations is Clohessy and Wiltshire, "Terminal Guidance
 * System for Satellite Rendezvous", Journal of the Aerospace Sciences,
 * volume 27, number 9, 1960, pages 653 to 658. That paper is not read
 * here and the implementation does not rest on it: it rests on the
 * derivation and on the checks named above.
 */
#ifndef K26ASTRO_PROX_H
#define K26ASTRO_PROX_H

#include "k26astro_core/pos.h"
#include "k26m3d.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    K26ASTRO_PROX_OK           = 0,
    K26ASTRO_PROX_E_NULL       = 1,  /* null argument */
    K26ASTRO_PROX_E_DEGENERATE = 2,  /* no frame: coincident centres,
                                      * or motion with no angular
                                      * momentum about the centre */
    K26ASTRO_PROX_E_RANGE      = 3   /* mean motion or interval not a
                                      * usable finite number */
} K26AstroProxStatus;

/**
 * @brief Decode a status value.
 * @param s A status.
 * @return A short English phrase; never NULL.
 */
const char *k26astro_prox_status_str(K26AstroProxStatus s);

/**
 * The chief's local-vertical local-horizontal basis, in world axes.
 *
 * `e1` is the unit radial direction, outward from the central body to
 * the chief. `e3` is the unit orbital angular momentum direction. `e2`
 * completes the right-handed set, `e3` cross `e1`, which for any
 * motion with a positive transverse component points along the
 * direction of motion. `omega` is the frame's angular velocity in
 * world axes and `radius` the chief's distance from the central body.
 */
typedef struct {
    K26V3  e1;
    K26V3  e2;
    K26V3  e3;
    K26V3  omega;
    double radius;
} K26AstroProxFrame;

/** A relative state in the chief's frame: metres and metres/second. */
typedef struct {
    K26V3 r;
    K26V3 v;
} K26AstroProxRel;

/**
 * @brief Build the chief's local-vertical local-horizontal frame.
 * @param central_pos Central body position.
 * @param central_vel Central body velocity, world axes, m/s.
 * @param chief_pos   Chief position.
 * @param chief_vel   Chief velocity, world axes, m/s.
 * @param out         Frame written on success; untouched otherwise.
 * @return K26ASTRO_PROX_OK, or a status.
 * @note  Positions are differenced with `k26astro_pos_sub`, the exact
 *        sector-aware subtraction, never through flattened
 *        coordinates: the pair may sit anywhere in the system and the
 *        separation is the small quantity, which is precisely the case
 *        a flattened difference destroys.
 * @note  The frame's angular velocity is `(r cross v) / |r|^2`, the
 *        rate at which the radial direction turns about the angular
 *        momentum direction. The component along `e1` is taken as
 *        zero, which is the standard convention and is exact whenever
 *        the angular momentum direction holds still; under a
 *        perturbation that tilts the orbit plane it neglects that
 *        tilt's rate, which is smaller than the retained term by the
 *        ratio of the out-of-plane force to the central attraction.
 * @note  Refused as degenerate when the chief sits at the central
 *        body's centre, or when the two are in purely radial motion,
 *        because neither case names a direction of motion.
 */
K26AstroProxStatus k26astro_prox_frame(const K26AstroPos *central_pos,
                                       K26V3 central_vel,
                                       const K26AstroPos *chief_pos,
                                       K26V3 chief_vel,
                                       K26AstroProxFrame *out);

/**
 * @brief The target's state relative to the chief, in the chief's
 *        frame.
 * @param f          A frame from k26astro_prox_frame.
 * @param chief_pos  Chief position.
 * @param chief_vel  Chief velocity, world axes, m/s.
 * @param target_pos Target position.
 * @param target_vel Target velocity, world axes, m/s.
 * @param out        Relative state written on success.
 * @return K26ASTRO_PROX_OK, or a status.
 * @note  The velocity is the rate of change an observer riding the
 *        chief's frame sees, `(v_target - v_chief) - omega cross r`,
 *        and not the inertial velocity difference merely resolved onto
 *        the axes. The difference is not a detail: two craft holding
 *        station on the same circular orbit, one ahead of the other,
 *        have a non-zero inertial velocity difference and a zero
 *        relative velocity, and it is the zero that a station-keeping
 *        policy and the propagator below both mean.
 */
K26AstroProxStatus k26astro_prox_relative(const K26AstroProxFrame *f,
                                          const K26AstroPos *chief_pos,
                                          K26V3 chief_vel,
                                          const K26AstroPos *target_pos,
                                          K26V3 target_vel,
                                          K26AstroProxRel *out);

/**
 * @brief Mean motion of a circular orbit.
 * @param mu     Gravitational parameter of the central body, m^3/s^2.
 * @param radius Orbit radius, metres.
 * @return sqrt(mu / radius^3), or 0 for a non-positive or non-finite
 *         argument.
 */
double k26astro_prox_mean_motion(double mu, double radius);

/**
 * @brief The Clohessy-Wiltshire state transition matrix.
 * @param n   Mean motion of the reference orbit, rad/s, positive.
 * @param t   Interval, seconds; may be negative.
 * @param phi 36 doubles, row-major 6 by 6, written on success. The
 *            state order is r_x, r_y, r_z, v_x, v_y, v_z in the
 *            chief's frame.
 * @return K26ASTRO_PROX_OK, or a status.
 * @note  Valid for a circular reference orbit and a separation small
 *        against its radius. The neglected term is second order in
 *        that ratio; tests/test_prox_cw.c measures the resulting error
 *        against integrated two-body motion, shows it falling as the
 *        square of the separation, and shows it leaving the stated
 *        bound where the linearisation stops holding.
 */
K26AstroProxStatus k26astro_prox_cw_stm(double n, double t, double *phi);

/**
 * @brief Propagate a relative state with the matrix above.
 * @param n   Mean motion, rad/s, positive.
 * @param t   Interval, seconds.
 * @param in  Relative state at the interval's start.
 * @param out Relative state at its end; may alias `in`.
 * @return K26ASTRO_PROX_OK, or a status.
 */
K26AstroProxStatus k26astro_prox_cw_propagate(double n, double t,
                                              const K26AstroProxRel *in,
                                              K26AstroProxRel *out);

/**
 * @brief The relative velocity that holds a declared standoff.
 * @param n     Mean motion, rad/s, positive.
 * @param hold  The hold point in the chief's frame, metres.
 * @param out_v Relative velocity written on success, m/s.
 * @return K26ASTRO_PROX_OK, or a status.
 * @note  Returns the velocity that leaves the relative motion bounded,
 *        `(0, -2 n x, 0)`, which removes the secular along-track drift
 *        the linearised solution otherwise carries at the rate
 *        `-6 n x - 3 v_y`. A hold point on the along-track axis alone
 *        therefore holds still exactly, which is the station-keeping
 *        case; a hold point with a radial component travels a closed
 *        two-by-one ellipse about it once per orbit, which is the
 *        smallest honest answer, since no non-zero radial offset is a
 *        fixed point of the relative motion.
 * @note  No sine or cosine is called, so a reset that places a craft
 *        through this function is reproducible across platforms.
 */
K26AstroProxStatus k26astro_prox_cw_hold(double n, K26V3 hold,
                                         K26V3 *out_v);

#ifdef __cplusplus
}
#endif

#endif /* K26ASTRO_PROX_H */
