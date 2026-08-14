/* att_internal.h: predicates shared by this library's two entry
 * points. They are here rather than duplicated in each because both
 * entries advance the same state under the same preconditions, and a
 * guard that exists in one translation unit and not the other is how
 * the actuated path came to accept a singular tensor the unactuated
 * path refused.
 *
 * Not installed and not part of the public interface.
 */
#ifndef K26ASTRO_ATT_INTERNAL_H
#define K26ASTRO_ATT_INTERNAL_H

#include "k26astro_att/att.h"

/* The inertia inverse is zeroed by the body library when the tensor it
 * was given is singular, which is exactly the state in which a torque
 * step would silently do nothing. Detecting it turns that into a
 * status. */
static inline int att_inverse_is_zero_(const K26M3 *inv)
{
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (inv->m[i][j] != 0.0) return 0;
        }
    }
    return 1;
}

#endif /* K26ASTRO_ATT_INTERNAL_H */
