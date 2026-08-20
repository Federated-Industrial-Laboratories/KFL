/* capture.h - the docking capture envelopes, named and declared.
 *
 * A port names the envelope its contact is judged against. An
 * envelope is a set of limits on the state of one docking interface
 * with respect to the other at first contact, plus the diameter of
 * the mating plane those limits are stated at.
 *
 * There are two sources. The built-in table carries the envelopes
 * this version defines from published documents; a program may
 * declare its own with a `capture_envelope` block, and a port's
 * `capture` mark resolves against the two together. A declaration
 * may not take a built-in name, so one name is one envelope whichever
 * source it came from.
 *
 * The values are held here in the units the source document prints,
 * and the conversion to the SI units the artifact carries happens in
 * one place, in capture.c. That is deliberate: a citation stays exact
 * when the figure beside it is the figure the document shows, and a
 * conversion written once can be checked once. A declared envelope
 * arrives in the same printed units and takes the same conversion,
 * so a program's own figures and a published table's are read the
 * same way.
 */
#ifndef KFLC_CAPTURE_H
#define KFLC_CAPTURE_H

/* How many envelopes one program may declare. */
#define KFLC_CAPTURE_MAX_DECLARED 32

/* The longest envelope name, matching the assembly reader's own name
 * limit, since a port's `capture` mark is where these names are
 * spelled a second time. */
#define KFLC_CAPTURE_NAME_MAX 64

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
 * @brief Look up a capture envelope by name.
 * @param name Envelope name as an assembly's port declared it.
 * @return The envelope, or NULL when no envelope carries that name.
 * @note  Resolves against the built-in table and the envelopes the
 *        program being compiled declared, in that order. A
 *        declaration may not take a built-in name, so the order
 *        settles nothing a program can observe.
 */
const KflcCaptureEnvelope *kflc_capture_envelope(const char *name);

/**
 * @brief Look up a capture envelope in the built-in table alone.
 * @param name Envelope name.
 * @return The envelope, or NULL when the built-in table has no such
 *         name.
 * @note  This is what a declaration is checked against before it is
 *        admitted, so that a program cannot redefine a published
 *        envelope under its own name.
 */
const KflcCaptureEnvelope *kflc_capture_builtin(const char *name);

/**
 * @brief The names in scope, for a diagnostic.
 * @param i Index from 0.
 * @return The name, or NULL past the last.
 * @note  The built-in names first, then the program's declared ones
 *        in declaration order.
 */
const char *kflc_capture_name_at(int i);

/**
 * @brief Forget every declared envelope.
 * @note  Called at the start of each program's collection, so a
 *        compiler that emits twice from one process resolves the
 *        second emission against that program's declarations alone.
 */
void kflc_capture_declared_reset(void);

/**
 * @brief Admit one program-declared envelope.
 * @param e The envelope in printed units; its name and source strings
 *          are copied, so the caller's storage need not outlive the
 *          call.
 * @return 0 when admitted, 1 when the table is full.
 * @note  The caller checks the values and the name; this records what
 *        it is given.
 */
int kflc_capture_declare(const KflcCaptureEnvelope *e);

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
