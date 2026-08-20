/* capture.c: the capture envelopes and the one conversion.
 *
 * The figures below are transcribed from the document each entry
 * names, in the units that document prints them in. Nothing here is
 * rounded, averaged, or adjusted: a transcription that changes a
 * figure is a transcription error, and the only way to keep that
 * checkable is to carry the printed number and convert it once.
 *
 * A program's own declarations join them in the table below the
 * built-in one. They arrive in the same printed units and leave
 * through the same conversion, so a declared envelope and a
 * transcribed one differ in where their figures came from and in
 * nothing else.
 */

#include "capture.h"

#include <stdio.h>
#include <string.h>

/* The International Docking System Standard's recommended initial
 * contact conditions, and the mating plane those conditions are
 * stated at. The standard publishes them as recommendations for
 * interoperability between docking mechanisms; a program that names
 * this envelope adopts them as the condition its own task treats as
 * a capture, which is a use of the figures and not a claim about
 * what the standard requires. */
static const KflcCaptureEnvelope kflc_envelopes_[] = {
    {
        "idss_e",
        0.05,   /* closing (axial) rate, m/sec, lower end of the band */
        0.10,   /* closing (axial) rate, m/sec, upper end */
        0.04,   /* lateral (radial) rate, m/sec */
        0.20,   /* pitch/yaw rate, deg/sec, vector sum */
        0.20,   /* roll rate, deg/sec */
        0.10,   /* lateral (radial) misalignment, m */
        4.0,    /* pitch/yaw misalignment, deg, vector sum */
        4.0,    /* roll misalignment, deg */
        1200.0, /* soft capture mating plane diameter, mm */
        "International Docking System Standard Interface Definition "
        "Document, Revision E, October 2016: Table 3.3.1.1-2 for the "
        "contact conditions, section 3.2 for the mating plane"
    }
};

#define KFLC_N_ENVELOPES \
    ((int)(sizeof kflc_envelopes_ / sizeof kflc_envelopes_[0]))

/* What a program declared, this compilation. The name and the source
 * note are held here rather than pointed at, because the tree they
 * were read from is released before the last consumer of an envelope
 * has finished with it. */
typedef struct {
    KflcCaptureEnvelope env;
    char                name[KFLC_CAPTURE_NAME_MAX];
    char                source[128];
} KflcDeclaredEnvelope;

static KflcDeclaredEnvelope kflc_declared_[KFLC_CAPTURE_MAX_DECLARED];

static int kflc_n_declared_;

const KflcCaptureEnvelope *kflc_capture_builtin(const char *name)
{
    if (!name) return NULL;
    for (int i = 0; i < KFLC_N_ENVELOPES; i++) {
        if (strcmp(kflc_envelopes_[i].name, name) == 0) {
            return &kflc_envelopes_[i];
        }
    }
    return NULL;
}

const KflcCaptureEnvelope *kflc_capture_envelope(const char *name)
{
    const KflcCaptureEnvelope *b = kflc_capture_builtin(name);
    if (b) return b;
    if (!name) return NULL;
    for (int i = 0; i < kflc_n_declared_; i++) {
        if (strcmp(kflc_declared_[i].name, name) == 0) {
            return &kflc_declared_[i].env;
        }
    }
    return NULL;
}

const char *kflc_capture_name_at(int i)
{
    if (i < 0) return NULL;
    if (i < KFLC_N_ENVELOPES) return kflc_envelopes_[i].name;
    i -= KFLC_N_ENVELOPES;
    if (i >= kflc_n_declared_) return NULL;
    return kflc_declared_[i].name;
}

void kflc_capture_declared_reset(void)
{
    kflc_n_declared_ = 0;
}

int kflc_capture_declare(const KflcCaptureEnvelope *e)
{
    if (!e || !e->name) return 1;
    if (kflc_n_declared_ >= KFLC_CAPTURE_MAX_DECLARED) return 1;
    KflcDeclaredEnvelope *d = &kflc_declared_[kflc_n_declared_];
    d->env = *e;
    snprintf(d->name, sizeof d->name, "%s", e->name);
    snprintf(d->source, sizeof d->source, "%s",
             e->source ? e->source : "declared in this program");
    d->env.name   = d->name;
    d->env.source = d->source;
    kflc_n_declared_++;
    return 0;
}

double kflc_capture_deg_to_rad(double deg)
{
    return deg * (3.14159265358979323846 / 180.0);
}

double kflc_capture_mm_to_m(double mm)
{
    return mm / 1000.0;
}

double kflc_capture_plate_half(double diameter_m)
{
    return diameter_m * 0.5 * 0.01;
}
