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
 * A caller that subdivides a control period should know what a
 * failure part-way through means. When an advance fails at the k-th
 * sub-interval, the k sub-intervals before it have already been
 * applied, so the world has moved while no transition completed. The
 * honest reading is that the record states the transition, not the
 * world: a failed transition records an applied duration of zero, and
 * what makes that true of the world as well is the caller's boundary
 * reset, which restores the episode's baseline whole. The reinforcement
 * learning layer does exactly that, and gates it: after a fault at a
 * sub-advance the next episode begins at the declared state exactly.
 * A caller that does not reset after a failure keeps a partially
 * advanced world, and that is the caller's to handle.
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
 *        norm is reported as divergence rather than propagated. A
 *        step that produces a non-finite orientation or rate is
 *        reported as diverged and the body is left holding the last
 *        finite state, so a caller
 *        can end an episode honestly rather than publishing a value
 *        that is not a number.
 */
K26AstroAttStatus k26astro_att_step(K26AstroVehicle *v, K26V3 torque,
                                    double dt);

/**
 * @brief Gravity-gradient torque on a vehicle, in its body frame.
 * @param v       The vehicle; its orientation rotates the separation
 *                into the body frame and its inertia tensor scales
 *                the result.
 * @param r_world Vector from the vehicle's body to the attracting
 *                body, in the world frame, metres.
 * @param mu      The attracting body's gravitational parameter,
 *                cubic metres per second squared.
 * @param out     Receives the torque, newton metres.
 * @return K26ASTRO_ATT_OK, or a status describing why it is zero.
 * @note  The torque is three mu over the cube of the separation,
 *        times the cross product of the unit separation with the
 *        inertia tensor applied to it. libk26astro_body implements
 *        the same expression for a diagonal inertia
 *        (k26astro_torque_gravity_gradient) and cites Wertz (1978)
 *        section 17.2 for it; this entry exists because a vehicle
 *        built from an assembly carries a full tensor whose products
 *        of inertia that entry cannot see, and dropping them would
 *        quietly change the torque on any vehicle that is not
 *        symmetric. The two agree exactly when the tensor is
 *        diagonal, which is gated.
 */
K26AstroAttStatus k26astro_att_gravity_gradient(const K26AstroVehicle *v,
                                                K26V3 r_world, double mu,
                                                K26V3 *out);

/**
 * @brief Advance a set of vehicles in the order given.
 * @param v   Array of vehicle pointers; null entries are skipped.
 * @param n   Entry count.
 * @param dt  Interval, seconds.
 * @return The first non-OK status, or K26ASTRO_ATT_OK.
 * @note  Registration order is the iteration order and is fixed by
 *        the caller, which is what makes a run reproducible. The
 *        torque applied to each is the caller's array entry, so a
 *        caller that has computed gravity-gradient torques passes
 *        them and one that has not passes zeroes.
 */
K26AstroAttStatus k26astro_att_step_all(K26AstroVehicle *const *v, int n,
                                        const K26V3 *torques, double dt);

/* ---- Actuators ---------------------------------------------------- *
 *
 * A vehicle's actuators are described by the assembly it was built
 * from and their state lives in the caller's storage, not in a heap
 * object of this library's: the stepping path allocates nothing, and
 * a set of plain arrays is what an environment can hold per
 * environment and reset without freeing anything.
 *
 * Commands are separate from state. A control loop writes a command
 * once per control period and the advance consumes it at each
 * sub-interval, which is what a zero-order hold means.
 */

/* A reaction wheel. The axis is a unit vector in the body frame; the
 * momentum is the state, along that axis. Friction is a viscous term
 * proportional to wheel rate plus a constant Coulomb term opposing
 * motion, the latter switched off below the dead rate so a wheel at
 * rest does not chatter across a sign change. */
typedef struct {
    K26V3  axis;
    double spin_inertia;   /* kg m^2, about the spin axis */
    double max_momentum;   /* N m s */
    double max_torque;     /* N m */
    double viscous;        /* N m per rad/s */
    double coulomb;        /* N m */
    double dead_rate;      /* rad/s */
    double momentum;       /* state, N m s */
    double command;        /* commanded torque, N m */
} K26AstroAttWheel;

/* A magnetorquer. The command is the signed dipole magnitude along
 * the declared axis. */
typedef struct {
    K26V3  axis;
    double max_dipole;     /* A m^2 */
    double command;        /* A m^2, signed along the axis */
} K26AstroAttTorquer;

/* A thruster. Position and direction are in the body frame; the
 * command is a throttle in the closed unit interval. */
typedef struct {
    K26V3  at;
    K26V3  dir;
    double max_thrust;     /* N */
    double command;        /* throttle, 0 to 1 */
} K26AstroAttThruster;

/* One vehicle's actuators, and the centre of mass their torques are
 * taken about. Any count may be zero. */
typedef struct {
    K26AstroAttWheel    *wheels;
    int                  n_wheels;
    K26AstroAttTorquer  *torquers;
    int                  n_torquers;
    K26AstroAttThruster *thrusters;
    int                  n_thrusters;
    K26V3                com;          /* body frame, metres */
} K26AstroAttActuators;

/**
 * @brief Advance the wheels and report their reaction on the body.
 * @param act    The actuator set; wheel momenta are updated in place.
 * @param dt     Interval, seconds.
 * @param torque Receives the reaction torque on the body, which is
 *               the negated rate of change of the stored momentum.
 * @param stored Receives the total stored momentum in the body frame,
 *               after the update; may be NULL.
 * @return K26ASTRO_ATT_OK, or a status.
 * @note  The commanded torque is clamped to the wheel's limit.
 *        Saturation clamps the state, not the command: at maximum
 *        momentum the part of the rate of change that would increase
 *        the magnitude is dropped, so the wheel delivers no further
 *        torque in that direction while friction can still slow it,
 *        and the body's momentum has to be dumped by something else.
 */
K26AstroAttStatus k26astro_att_wheels_step(K26AstroAttActuators *act,
                                           double dt, K26V3 *torque,
                                           K26V3 *stored);

/**
 * @brief Torque from the magnetorquers, in the body frame.
 * @param act    The actuator set.
 * @param b_body The local magnetic flux density in the body frame,
 *               tesla.
 * @param out    Receives the torque, newton metres.
 * @return K26ASTRO_ATT_OK, or a status.
 * @note  The torque is the commanded dipole crossed with the field,
 *        so it is exactly zero along the field: a magnetorquer set
 *        cannot control the axis parallel to the field, and a task
 *        built on them alone is a three-axis problem with a
 *        time-varying uncontrollable direction. That is a property of
 *        the actuator, not a limitation of this implementation.
 */
K26AstroAttStatus k26astro_att_torquers_torque(
    const K26AstroAttActuators *act, K26V3 b_body, K26V3 *out);

/**
 * @brief Force and torque from the thrusters, in the body frame.
 * @param act   The actuator set.
 * @param force Receives the summed force, newtons; may be NULL.
 * @param out   Receives the torque about the centre of mass, newton
 *              metres; may be NULL.
 * @return K26ASTRO_ATT_OK, or a status.
 * @note  Each thruster's throttle is clamped to the closed unit
 *        interval. One declaration serves translation and rotation
 *        together: the force is the throttle times the maximum thrust
 *        along the direction, and the torque is the offset from the
 *        centre of mass crossed with that force, so a thruster
 *        through the centre of mass produces none.
 */
K26AstroAttStatus k26astro_att_thrusters_wrench(
    const K26AstroAttActuators *act, K26V3 *force, K26V3 *out);

/**
 * @brief Geodetic latitude, longitude and height from an Earth-fixed
 *        Cartesian position.
 * @param ecef Position in the Earth-fixed frame, metres.
 * @param lat  Receives geodetic latitude, radians.
 * @param lon  Receives longitude, radians.
 * @param alt  Receives height above the ellipsoid, metres.
 * @return K26ASTRO_ATT_OK, or K26ASTRO_ATT_E_NULL.
 * @note  This exists because the magnetic field model is evaluated at
 *        a geodetic position and nothing in this tree converted to
 *        one. Bowring's method, iterated to convergence, on the WGS84
 *        ellipsoid.
 *
 *        The two defining constants are corroborated inside this tree
 *        rather than asserted: the equatorial radius 6378137 m is what
 *        libk26geo states as the WGS84 equatorial radius, and the
 *        flattening 1/298.257223563 is what libk26astro_core's
 *        constants carry marked WGS84. The rounded equatorial radius
 *        in that same header is not used here, since pairing it with
 *        the exact flattening would misplace a point by of order a
 *        hundred metres. The standard itself is the authority for
 *        both and is verified at intake.
 */
K26AstroAttStatus k26astro_att_geodetic(K26V3 ecef, double *lat,
                                        double *lon, double *alt);

/**
 * @brief Rotate a local east-north-up vector into the Earth-fixed
 *        frame.
 * @param enu Vector in the local frame, east then north then up.
 * @param lat Geodetic latitude, radians.
 * @param lon Longitude, radians.
 * @return The same vector in the Earth-fixed frame.
 * @note  The field model reports north, east and down, so a caller
 *        converts to east, north and up before calling this. The
 *        sign of the down component is the trap that convention sets,
 *        and it is the caller's to get right; the gate that exercises
 *        this pair does so through a field whose direction is known.
 */
K26V3 k26astro_att_enu_to_ecef(K26V3 enu, double lat, double lon);

/**
 * @brief Advance a vehicle's attitude with stored momentum.
 * @param v     The vehicle.
 * @param act   Its actuators, or NULL for a body with none.
 * @param extra Any further body-frame torque for this interval, such
 *              as the gravity gradient.
 * @param dt    Interval, seconds.
 * @return K26ASTRO_ATT_OK, or a status.
 * @note  The equation is the momentum-exchange form of Euler's
 *        rotational equation: the inertia times the angular
 *        acceleration equals the applied torque, less the cross
 *        product of the angular velocity with the sum of the body's
 *        own angular momentum and the wheels' stored momentum, less
 *        the rate of change of that stored momentum. The last two
 *        terms are what a wheel does to a spacecraft and what the
 *        unaugmented equation omits.
 *
 *        Provenance: libk26astro_body implements and cites the
 *        unaugmented equation (Markley and Crassidis 2014 section
 *        3.6.2, Hughes 1986 section 4.5, Wertz 1978 section 16.3).
 *        The augmented form is the same equation with the wheel
 *        momentum carried inside the gyroscopic term, and is standard
 *        in all three; the section citation for it is verified at
 *        intake rather than asserted here. What is not left to
 *        citation is its behaviour: the gates check that total
 *        angular momentum is conserved under wheel commands alone,
 *        which is the property the augmented terms exist to produce
 *        and which the unaugmented equation does not have.
 */
K26AstroAttStatus k26astro_att_step_actuated(K26AstroVehicle *v,
                                             K26AstroAttActuators *act,
                                             K26V3 extra, K26V3 b_body,
                                             double dt);

/**
 * @brief Total angular momentum of a vehicle in the world frame.
 * @param v The vehicle.
 * @param out Receives the angular momentum vector, kg m^2 per second.
 * @return K26ASTRO_ATT_OK, or K26ASTRO_ATT_E_NULL.
 * @note  The inertia tensor times the body-frame angular velocity,
 *        rotated into the world frame by the orientation. Torque-free
 *        motion conserves it, which is what the analytic gates use to
 *        measure the integrator rather than trusting it. Like the
 *        advance, this reads the bound body first, so it reports the
 *        momentum of the state a body actually holds rather than of
 *        a working copy that may predate the last write to it.
 */
K26AstroAttStatus k26astro_att_momentum_world(const K26AstroVehicle *v,
                                              K26V3 *out);

#ifdef __cplusplus
}
#endif

#endif /* K26ASTRO_ATT_H */
