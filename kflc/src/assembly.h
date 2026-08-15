/* assembly.h - vehicle assemblies and their derived mass properties.
 *
 * An assembly is the data-driven description of a vehicle: components
 * with placements and masses, collision primitives, docking ports,
 * thrusters, and momentum devices. The compiler reads it, derives the
 * vehicle's mass, centre of mass, and inertia tensor from it, and
 * writes those numbers into the emitted source as constants, so a
 * running simulation never opens an asset file.
 *
 * Two properties this interface exists to keep:
 *
 * Identity. The bytes of the assembly and of every mesh it references
 * are hashed into one digest that travels in the compiled program, so
 * a changed asset is a changed program and a recorded run can be
 * traced back to the exact geometry that produced it.
 *
 * Determinism. The derivation walks components in source order and
 * triangles in file order, and its arithmetic is addition,
 * subtraction, multiplication, division, and one square root, all of
 * which IEEE-754 requires to be correctly rounded; nothing here
 * depends on evaluation order or on a library approximation. Under
 * the float-control flags in this directory's Makefile the derived
 * constants are therefore a function of the asset bytes alone.
 */
#ifndef KFLC_ASSEMBLY_H
#define KFLC_ASSEMBLY_H

#include "kflc.h"

#include <stdint.h>

#define KFLC_ASM_NAME_MAX   64
#define KFLC_ASM_PATH_MAX   512
#define KFLC_ASM_SRC_MAX    256
#define KFLC_ASM_MAX_COMP   64
#define KFLC_ASM_MAX_COLL   128
#define KFLC_ASM_MAX_FEAT   128
#define KFLC_ASM_MAX_PROV   64
#define KFLC_ASM_DIGEST     32

/* Collision primitive kinds. Declared, never inferred from a mesh:
 * an author who wants a collider says so, which keeps the collision
 * cost visible in the source. */
typedef enum {
    KFLC_SHAPE_SPHERE  = 1,
    KFLC_SHAPE_CAPSULE = 2,
    KFLC_SHAPE_BOX     = 3
} KflcShapeKind;

/* A collider is declared in its component's frame and is left there
 * for the mass-property derivation, which walks components one at a
 * time. After that derivation the component placement is baked in and
 * every collider is in the assembly's body frame, which is the frame
 * a collision pass wants and the only one in which two components'
 * colliders can be compared. `rot` carries the component's
 * orientation, since a box needs its axes and not only its centre;
 * it is the identity for a component that declares no rotation. */
typedef struct {
    KflcShapeKind kind;
    double        a[3];      /* sphere centre, capsule end A, box half extents */
    double        b[3];      /* capsule end B; unused otherwise */
    double        r;         /* sphere and capsule radius; unused for a box */
    double        rot[3][3]; /* body frame from component frame */
    double        centre[3]; /* body frame, after baking */
    int           component; /* owning component index */
    int           line;
} KflcAsmCollider;

/* Features carry no mass of their own: a thruster's or a wheel's mass
 * belongs to the component that holds it, which is where an author
 * declares it. Refusing a mass key on a feature keeps one place per
 * quantity. */
typedef enum {
    KFLC_FEAT_PORT     = 1,
    KFLC_FEAT_THRUSTER = 2,
    KFLC_FEAT_WHEEL    = 3,
    KFLC_FEAT_TORQUER  = 4
} KflcAsmFeatureKind;

typedef struct {
    KflcAsmFeatureKind kind;
    char   name[KFLC_ASM_NAME_MAX];
    double at[3];            /* port, thruster */
    double axis[3];          /* port, wheel, magnetorquer */
    double roll_ref[3];      /* port */
    double dir[3];           /* thruster */
    char   capture[KFLC_ASM_NAME_MAX];   /* port capture envelope name */
    double thrust;           /* thruster, N */
    double spin_inertia;     /* wheel, kg m^2 */
    double max_momentum;     /* wheel, N m s */
    double max_torque;       /* wheel, N m */
    double viscous;          /* wheel, N m per rad/s */
    double coulomb;          /* wheel, N m */
    double dead_rate;        /* wheel, rad/s */
    double max_dipole;       /* magnetorquer, A m^2 */
    /* The mating plane collider a port with a named capture envelope
     * is given, as an index into the assembly's collider array, or -1
     * for every other feature and for a port that names none. The
     * plate is synthesised from the envelope's published mating plane
     * rather than declared, so an author cannot get the geometry the
     * capture test is judged at wrong, and it is appended after the
     * mass-property derivation has run so that it adds no mass. */
    int    collider;
    int    line;
} KflcAsmFeature;

typedef struct {
    char   name[KFLC_ASM_NAME_MAX];
    double mass;             /* kg, declared */
    double at[3];            /* placement in the body frame, m */
    double rot[4];           /* w, x, y, z; identity when absent */
    int    has_rot;
    char   mesh[KFLC_ASM_PATH_MAX];
    int    has_mesh;
    int    line;
    /* Derived, in the component's own frame. */
    double volume;           /* m^3 */
    double com[3];           /* m, from the component origin */
    double inertia[6];       /* about the component centre of mass:
                                xx, yy, zz, xy, xz, yz, the tensor
                                form with products already negated */
} KflcAsmComponent;

/* Provenance status words. The set is closed: an asset says where a
 * number came from in one of four ways or is refused.
 *
 * `cited` is read from the named source. `computed` is derived here
 * from cited inputs, and the derivation is the source. `modelled` is
 * an arrangement chosen rather than found: the source names what was
 * chosen and what is not published, and the choice is disclosed
 * rather than dressed as a derivation. `unverified` is a working
 * figure whose source has not been checked.
 *
 * The fourth word exists because the first three left a real case
 * with nowhere truthful to sit. A thruster layout placed on a hull
 * whose dimensions are cited is not derived from those dimensions in
 * any sense that would let a reader reconstruct it, so calling it
 * computed overstates; and it is not an unchecked working figure
 * either, since there is nothing to check it against. Marking it
 * either way made the status word disagree with the asset's own
 * prose, which is the disagreement this word removes. */
typedef enum {
    KFLC_PROV_CITED      = 1,
    KFLC_PROV_COMPUTED   = 2,
    KFLC_PROV_UNVERIFIED = 3,
    KFLC_PROV_MODELLED   = 4
} KflcAsmProvStatus;

typedef struct {
    char              property[KFLC_ASM_NAME_MAX];
    char              source[KFLC_ASM_SRC_MAX];
    KflcAsmProvStatus status;
    int               line;
} KflcAsmProvenance;

typedef struct {
    char    name[KFLC_ASM_NAME_MAX];
    char    path[KFLC_ASM_PATH_MAX];    /* as resolved and read */
    /* Derived totals, body frame. */
    double  mass;                       /* kg */
    double  com[3];                     /* m */
    double  inertia[6];                 /* about the assembly centre of
                                           mass: xx, yy, zz, xy, xz, yz */
    uint8_t digest[KFLC_ASM_DIGEST];
    /* One sphere about the body-frame origin covering every collider,
     * which is what a broadphase tests before it tests primitives.
     * Derived here because this is where the collider geometry and
     * the component placements are both in hand. */
    double  bound_radius;               /* m */
    int     n_components;
    int     n_colliders;
    int     n_features;
    int     n_provenance;
    int     n_unverified;
    int     n_modelled;
    KflcAsmComponent  components[KFLC_ASM_MAX_COMP];
    KflcAsmCollider   colliders[KFLC_ASM_MAX_COLL];
    KflcAsmFeature    features[KFLC_ASM_MAX_FEAT];
    KflcAsmProvenance provenance[KFLC_ASM_MAX_PROV];
} KflcAssembly;

/**
 * @brief Read an assembly, derive its mass properties, and digest it.
 * @param path     Assembly path as written in the program.
 * @param src_path Path of the source file that named it; the assembly
 *                 path resolves relative to its directory.
 * @param line     Source line, for diagnostics.
 * @param arena    Arena the result is allocated from.
 * @param diag     Diagnostics; every refusal names a file and a line.
 * @return The derived assembly, or NULL when it was refused.
 * @note  Warns once per property whose provenance status is
 *        unverified and once per property that is modelled; the
 *        counts are left in n_unverified and n_modelled so a caller
 *        can apply a stricter rule to either. The rule this tree
 *        applies is that a shipped asset carries no unverified
 *        property and may carry modelled ones, which are reported.
 */
KflcAssembly *kflc_assembly_load(const char *path, const char *src_path,
                                 int line, KflcArena *arena,
                                 KflcDiag *diag);

/**
 * @brief Resolve the assembly an `astro_body` statement binds, if any.
 * @param body     The KFLN_STMT_ASTRO_BODY node.
 * @param src_path Path of the source file, for relative resolution.
 * @param arena    Arena the result is allocated from.
 * @param diag     Diagnostics.
 * @param out      Receives the derived assembly, or NULL when the body
 *                 binds none.
 * @return 0 when the body is acceptable, 1 when it was refused.
 * @note  This is the one place the binding's rules live, because the
 *        batch emitter and the environment emitter each have their own
 *        body emission and must not be able to disagree about them.
 *        Declaring `mass=` or `gm=` beside `assembly=` is refused
 *        here, naming both sites.
 */
int kflc_assembly_for_body(const KflcNode *body, const char *src_path,
                           KflcArena *arena, KflcDiag *diag,
                           KflcAssembly **out);

/**
 * @brief Strip the quotes an attribute value still carries.
 * @param raw    Attribute text as the parser captured it.
 * @param out    Receives the unquoted path.
 * @param out_sz Capacity of out.
 * @return 0 on success, 1 when the value is absent or does not fit.
 * @note  Attribute values arrive as verbatim source text, so a quoted
 *        path arrives with its quotes attached.
 */
int kflc_assembly_unquote(const char *raw, char *out, size_t out_sz);

/**
 * @brief Render a digest as lowercase hexadecimal.
 * @param d   The digest bytes.
 * @param out Receives 64 hex digits and a terminator.
 */
void kflc_assembly_digest_hex(const uint8_t d[KFLC_ASM_DIGEST],
                              char out[2 * KFLC_ASM_DIGEST + 1]);

#endif /* KFLC_ASSEMBLY_H */
