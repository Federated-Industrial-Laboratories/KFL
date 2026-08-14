/* k26astro_att/att.h - spacecraft attitude on the exact stepping path.
 *
 * What this library is, given what already exists. The rigid-body
 * attitude core is in libk26astro_body: the quaternion exponential
 * map, Euler's rotational equation in a full inertia tensor, the
 * gravity-gradient torque, and the torque-source registry, together
 * with the K26AstroAttitudeStateExt a spacecraft uses. Nothing here
 * reimplements any of that. This library is the actuator and
 * control-authority layer, and the advance that puts attitude on the
 * same stepping path as translation.
 *
 * The advance is the piece the tree lacked. A body's attitude fields
 * exist and are serialised, and nothing integrates them: the
 * runtime's spin channel is a declared no-op, and it is rate-driven,
 * so an attitude advanced there would be a function of a scheduler
 * setting rather than of the declared physics. Attitude is therefore
 * advanced by an explicit call, over the same interval as translation
 * and at the same subdivision of it, by whoever owns the stepping
 * loop.
 *
 * Ordering, per sub-advance of duration h: translation advances
 * first, then attitude advances by the same h, with the torque
 * evaluated at the interval's start. That splitting is first order in
 * h, as is the angular-velocity update the body library performs, so
 * the local error in attitude is of order h squared. It is measured
 * rather than asserted: tests/test_att_advance.c compares against
 * closed-form solutions and checks that the error falls at the rate a
 * first-order method gives when the subdivision doubles.
 *
 * Provenance. This library introduces no new equation of motion. The
 * forms it composes are implemented and cited by libk26astro_body,
 * whose sources give Markley and Crassidis (2014), Hughes (1986), and
 * Wertz (1978) for the quaternion exponential map, Euler's rotational
 * equation in non-principal axes, and the inertia tensor's
 * properties; the section numbers there are the ones to check against
 * those texts. What this file adds is sequencing and the write-back
 * to the body, both of which the analytic gates cover.
 */
#ifndef K26ASTRO_ATT_H
#define K26ASTRO_ATT_H

#include "k26astro_body/attitude.h"
#include "k26astro_vehicle/vehicle.h"
#include "k26m3d.h"

#ifdef __cplusplus
extern "C" {
#endif

#define K26ASTRO_ATT_LIB_VERSION "0.1.0"

/* Typed status, following the tree's enum-plus-decoder pattern. */
typedef enum {
    K26ASTRO_ATT_OK          = 0,
    K26ASTRO_ATT_E_NULL      = 1,   /* null vehicle or missing state */
    K26ASTRO_ATT_E_BAD_DT    = 2,   /* dt not finite, or negative */
    K26ASTRO_ATT_E_SINGULAR  = 3,   /* inertia tensor has no inverse */
    K26ASTRO_ATT_E_DIVERGED  = 4    /* the step produced a non-finite state */
} K26AstroAttStatus;

/**
 * @brief Decode a status value.
 * @param s The status.
 * @return A stable short string; never NULL.
 */
const char *k26astro_att_status_str(K26AstroAttStatus s);

/**
 * @brief Advance one vehicle's attitude by dt under a body-frame torque.
 * @param v      The vehicle; its Ext attitude state carries the
 *               orientation, the angular velocity, and the inertia
 *               tensor set from the vehicle's assembly.
 * @param torque Body-frame torque for this interval, held constant
 *               across it, which is what a control loop at a fixed
 *               rate applies.
 * @param dt     Interval, seconds; zero is accepted and does nothing.
 * @return K26ASTRO_ATT_OK, or a status describing why nothing moved.
 * @note  The bound body is the state. Its orientation and rate are
 *        loaded before the step and written back after, so anything
 *        that writes a body, a declaration, a reset draw, or a
 *        step-time assignment, is what the advance integrates, and
 *        consumers that read the body see the result. With no body
 *        bound the vehicle's own attitude state stands alone. The
 *        orientation is normalised at the point of use, since a
 *        quaternion is written one component at a time; one with zero
 *        norm is reported as divergence rather than propagated. A step that produces a
 *        non-finite orientation or rate is reported as diverged and
 *        the body is left holding the last finite state, so a caller
 *        can end an episode honestly rather than publishing a value
 *        that is not a number.
 */
K26AstroAttStatus k26astro_att_step(K26AstroVehicle *v, K26V3 torque,
                                    double dt);

/**
 * @brief Advance a set of vehicles in the order given.
 * @param v   Array of vehicle pointers; null entries are skipped.
 * @param n   Entry count.
 * @param dt  Interval, seconds.
 * @return The first non-OK status, or K26ASTRO_ATT_OK.
 * @note  Registration order is the iteration order and is fixed by
 *        the caller, which is what makes a run reproducible.
 *        Torque-free: this entry exists for the common case of a set
 *        of bodies with no actuator commanded this interval.
 */
K26AstroAttStatus k26astro_att_step_all(K26AstroVehicle *const *v, int n,
                                        double dt);

/**
 * @brief Total angular momentum of a vehicle in the world frame.
 * @param v The vehicle.
 * @param out Receives the angular momentum vector, kg m^2 per second.
 * @return K26ASTRO_ATT_OK, or K26ASTRO_ATT_E_NULL.
 * @note  The inertia tensor times the body-frame angular velocity,
 *        rotated into the world frame by the orientation. Torque-free
 *        motion conserves it, which is what the analytic gates use to
 *        measure the integrator rather than trusting it.
 */
K26AstroAttStatus k26astro_att_momentum_world(const K26AstroVehicle *v,
                                              K26V3 *out);

#ifdef __cplusplus
}
#endif

#endif /* K26ASTRO_ATT_H */
