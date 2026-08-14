/* body_state.c - the body state keys, and the one place that writes
 * them.
 *
 * Three emitters reach a body's scalar state: the batch emitter's
 * `astro_body` statement, the environment emitter's own version of
 * it, and the environment emitter's `on_step` state setters. Each
 * used to carry its own idea of which keys exist and how each one
 * lands on the struct, and that is how a key comes to mean two things
 * in one compiler: an assembly binding once reached an artifact
 * through one emitter while the other emitted an assignment to a
 * struct field that does not exist. The key table and the write live
 * here, once, and every emitter calls them.
 *
 * The six translation keys are metres and metres per second in the
 * world frame. A position key lands in the sector grid's local offset
 * with the sector index zeroed and then re-normalised, which is the
 * same construction the reset path uses, so a written position means
 * metres from the world origin whatever sector it lands in.
 *
 * The seven attitude keys are the body-to-world quaternion's four
 * components and the body-frame angular velocity's three, in radians
 * per second. A quaternion is written component by component like any
 * other scalar: the alternative, a single four-valued write, would be
 * a new statement form for one type. It is normalised where it is
 * used rather than where it is written, so a program that writes
 * three components and then the fourth is never observed part-way
 * through, and a quaternion written to zero norm produces a
 * non-finite orientation whose consequence the advance reports
 * through the fault path that already exists.
 */
#include "kflc.h"
#include "internal.h"

#include <stdio.h>
#include <string.h>

static const char *const BODY_STATE_KEYS_[] = {
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "omega_x", "omega_y", "omega_z"
};

int kflc_body_state_key_count(void)
{
    return (int)(sizeof BODY_STATE_KEYS_ / sizeof BODY_STATE_KEYS_[0]);
}

const char *kflc_body_state_key_name(int i)
{
    if (i < 0 || i >= kflc_body_state_key_count()) return NULL;
    return BODY_STATE_KEYS_[i];
}

/* Whether a key names attitude state rather than translation. The
 * distinction matters at the binding: attitude is advanced from an
 * inertia tensor, which only an assembly supplies. */
int kflc_body_state_is_attitude(const char *k)
{
    if (!k) return 0;
    return strncmp(k, "quat_", 5) == 0 || strncmp(k, "omega_", 6) == 0;
}

int kflc_body_state_key_index(const char *k)
{
    if (!k) return -1;
    for (int i = 0; i < kflc_body_state_key_count(); i++) {
        if (strcmp(k, BODY_STATE_KEYS_[i]) == 0) return i;
    }
    return -1;
}

/* The component character of a state key: the axis for a translation
 * or rate key, and w, x, y or z for a quaternion key. */
char kflc_body_state_key_comp(const char *k)
{
    if (!k) return '\0';
    if (strncmp(k, "quat_", 5) == 0)  return k[5];
    if (strncmp(k, "omega_", 6) == 0) return k[6];
    return k[4];
}

void kflc_emit_body_state_write(FILE *out, int indent, const char *lv,
                                const char *key, const char *value_text)
{
    char comp = kflc_body_state_key_comp(key);
    for (int i = 0; i < indent; i++) fputc(' ', out);
    if (strncmp(key, "pos_", 4) == 0) {
        fprintf(out, "%spos.s%c = 0; %spos.l%c = (%s); "
                     "k26astro_pos_normalise(&%spos);\n",
                lv, comp, lv, comp, value_text, lv);
    } else if (strncmp(key, "vel_", 4) == 0) {
        fprintf(out, "%svel.%c = (%s);\n", lv, comp, value_text);
    } else if (strncmp(key, "quat_", 5) == 0) {
        fprintf(out, "%sattitude.%c = (%s);\n", lv, comp, value_text);
    } else {
        fprintf(out, "%somega.%c = (%s);\n", lv, comp, value_text);
    }
}

void kflc_emit_body_state_read(FILE *out, const char *lv, const char *key)
{
    char comp = kflc_body_state_key_comp(key);
    if (strncmp(key, "pos_", 4) == 0) {
        fprintf(out, "    return (double)%spos.s%c * "
                     "K26ASTRO_SECTOR_EDGE_M + %spos.l%c;\n",
                lv, comp, lv, comp);
    } else if (strncmp(key, "vel_", 4) == 0) {
        fprintf(out, "    return %svel.%c;\n", lv, comp);
    } else if (strncmp(key, "quat_", 5) == 0) {
        fprintf(out, "    return %sattitude.%c;\n", lv, comp);
    } else {
        fprintf(out, "    return %somega.%c;\n", lv, comp);
    }
}
