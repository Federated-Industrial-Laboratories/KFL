/* capture.h - the named docking capture envelopes.
 *
 * A port names the envelope its contact is judged against. An
 * envelope is a set of limits on the state of one docking interface
 * with respect to the other at first contact, plus the diameter of
 * the mating plane those limits are stated at.
 *
 * The values are held here in the units the source document prints,
 * and the conversion to the SI units the artifact carries happens in
 * one place, in capture.c. That is deliberate: a citation stays exact
 * when the figure beside it is the figure the document shows, and a
 * conversion written once can be checked once.
 */
#ifndef KFLC_CAPTURE_H
#define KFLC_CAPTURE_H

/* One envelope, in printed units: metres and metres per second for
 * the distances and rates, degrees and degrees per second for the
 * angles, millimetres for the mating plane. */
typedef struct {
    const char *name;
    double axial_rate_min;      /* m/s, closing */
    double axial_rate_max;      /* m/s */
    double lateral_rate;        /* m/s */
    double pitchyaw_rate_deg;   /* deg/s, vector sum of pitch and yaw */
    double roll_rate_deg;       /* deg/s */
    double lateral;             /* m */
    double pitchyaw_deg;        /* deg, vector sum of pitch and yaw */
    double roll_deg;            /* deg */
    double mating_diameter_mm;  /* mm */
    const char *source;         /* document, revision, and table */
} KflcCaptureEnvelope;

/**
 * @brief Look up a named capture envelope.
 * @param name Envelope name as an assembly's port declared it.
 * @return The envelope, or NULL when no envelope carries that name.
 */
const KflcCaptureEnvelope *kflc_capture_envelope(const char *name);

/**
 * @brief The names this version defines, for a diagnostic.
 * @param i Index from 0.
 * @return The name, or NULL past the last.
 */
const char *kflc_capture_name_at(int i);

/**
 * @brief Convert a printed angle to radians.
 * @param deg Angle in degrees.
 * @return The angle in radians.
 * @note  The one place the compiler turns an envelope's printed
 *        angles into the units the artifact carries.
 */
double kflc_capture_deg_to_rad(double deg);

/**
 * @brief Convert a printed length to metres.
 * @param mm Length in millimetres.
 * @return The length in metres.
 */
double kflc_capture_mm_to_m(double mm);

/**
 * @brief The axial half thickness of a port's mating plane collider.
 * @param diameter_m Mating plane diameter in metres.
 * @return The half thickness in metres.
 * @note  A mating plane has no thickness, and a box with none is
 *        degenerate for the separating-axis kernel. The plate is
 *        given a hundredth of its own radius and sits entirely
 *        behind the mating plane, so the instant at which two mating
 *        planes meet is a function of the plane and not of the plate.
 */
double kflc_capture_plate_half(double diameter_m);

#endif /* KFLC_CAPTURE_H */
