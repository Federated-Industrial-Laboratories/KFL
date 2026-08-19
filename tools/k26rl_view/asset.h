/* asset.h - the vehicle assembly a recording names, read for drawing.
 *
 * An assembly is the text description of a vehicle that the compiler
 * derived a body's mass properties from, and the compiled artifact
 * carries the digest of its bytes. This reads the same asset from
 * disk, recomputes that digest through the format library's own
 * function, and builds the wireframe its meshes describe.
 *
 * The digest is the whole point of reading it here. A wireframe drawn
 * beside a recording is a claim that this is the craft that flew, and
 * the only thing that makes the claim true is that the bytes on disk
 * are the bytes the artifact was built from. When they are not, this
 * viewer says so and draws nothing, which is the design's own rule:
 * report the mismatch rather than render a craft that is not the
 * craft that flew.
 *
 * What is read here is a subset of what the compiler reads, and
 * deliberately so: the viewer needs what it draws. That is the
 * assembly's name, the meshes its components reference in the order
 * the compiler contributes them and the geometry those meshes hold,
 * and the shapes the scene view draws beside them, which are the
 * collision primitives, the docking ports and the thrusters. It does
 * not read masses, inertias, or momentum devices, and it derives no
 * mass property of any kind. A subset reader can disagree with the
 * compiler about what an assembly references, and the consequence of
 * disagreeing is bounded by construction: the digest then fails to
 * match and the viewer refuses to draw. It cannot draw the wrong
 * craft quietly.
 */
#ifndef K26RL_VIEW_ASSET_H
#define K26RL_VIEW_ASSET_H

#include <stdint.h>

#include <string>
#include <vector>

#include "model.h"

namespace k26rl_view {

/* One line of the wireframe, as two vertex indices. */
struct Edge {
    uint32_t a;
    uint32_t b;
};

/* One triangle, in the winding the mesh declared it in. The edges
 * above lose the winding, and a face normal has no sign without one,
 * so the shaded depth cue reads its faces from here instead. */
struct Face {
    uint32_t a;
    uint32_t b;
    uint32_t c;
};

/* The three collision primitives version 1 of the assembly format
 * carries. */
enum ColliderKind {
    COLLIDER_SPHERE  = 1,
    COLLIDER_CAPSULE = 2,
    COLLIDER_BOX     = 3
};

/* One collision primitive, in the assembly's body frame.
 *
 * A collider is declared inside a component and in that component's
 * frame; the compiler bakes the component placement in before
 * anything compares two components' colliders, and this does the same
 * for the same reason, a scene drawing one craft rather than one
 * component at a time. The bake is the placement this reader already
 * applies to the mesh vertices beside it. */
struct Collider {
    ColliderKind kind;
    double centre[3];     /* body frame */
    double a[3];          /* capsule end A, body frame; box half extents */
    double b[3];          /* capsule end B, body frame */
    double radius;        /* sphere and capsule; unused for a box */
    double rot[9];        /* body from component, row major */
};

/* One docking port, in the body frame, with what its named capture
 * envelope publishes.
 *
 * The envelope's figures are read from the compiler's own envelope
 * table rather than transcribed here: a port's mating plane is the
 * envelope's diameter and not the author's, and a second copy of that
 * number is a picture that disagrees with the program the moment
 * either moves. */
struct Port {
    std::string name;
    double at[3];
    double axis[3];
    double roll_ref[3];
    std::string envelope;       /* the name the port declared */
    bool has_envelope;          /* whether that name resolved */
    double mating_diameter;     /* m */
    double lateral_limit;       /* m, capture misalignment limit */
    double pitchyaw_limit_deg;  /* deg, capture misalignment limit */
};

/* One thruster, in the body frame. */
struct Thruster {
    std::string name;
    double at[3];
    double dir[3];
    double thrust;              /* N */
};

/* An assembly read from disk: what it is called, what its bytes
 * digest to, and the wireframe its meshes make.
 *
 * Vertices are in the assembly's body frame, the component placement
 * applied, because that is the frame the craft's own geometry is
 * expressed in and the frame a viewer draws it in. */
struct Asset {
    bool loaded = false;
    std::string error;                  /* why not, when !loaded */
    std::string path;
    std::string name;                   /* the `assembly <name>` line */
    uint8_t digest[32];
    std::vector<std::string> meshes;    /* referenced, in contribution order */
    uint32_t mesh_vertices = 0;         /* as the mesh files declare them */
    uint32_t mesh_triangles = 0;
    std::vector<double> vertices;       /* 3 per vertex, body frame */
    std::vector<Edge> edges;            /* unique, undirected */
    std::vector<Face> faces;            /* as declared, winding kept */
    std::vector<Collider> colliders;    /* body frame, placement baked */
    std::vector<Port> ports;            /* body frame */
    std::vector<Thruster> thrusters;    /* body frame */
};

/* Read an assembly and every mesh it references, recompute the
 * identity digest over their bytes, and build the wireframe. A
 * missing file, a malformed number, or a mesh directive outside the
 * accepted subset leaves `loaded` false with `error` saying which. */
Asset asset_load(const std::string &path);

/* Lowercase hexadecimal, for a panel and for a diagnostic. */
std::string digest_hex(const uint8_t d[32]);

/* Whether an asset on disk is the one a recording was made with. */
enum AssetVerdict {
    ASSET_DRAWABLE = 0,   /* the bytes are the bytes that flew */
    ASSET_NO_BODY,        /* no body binds an assembly of this name */
    ASSET_NO_DIGEST,      /* the recording carries none for it */
    ASSET_MISMATCH        /* it carries one and the bytes differ */
};

/* Match an asset against a recording's spec: which body binds it, and
 * whether its digest is the recorded one.
 *
 * One implementation, called by both presenters. The safety property
 * this whole panel exists for is that a craft whose bytes are not the
 * recorded bytes is never drawn, and a property implemented twice is
 * a property that holds in whichever copy was last looked at. On any
 * verdict but ASSET_DRAWABLE the caller draws nothing.
 *
 * @param sp    The recording's spec.
 * @param a     A loaded asset.
 * @param out   Receives the matching body's record when one is found.
 * @return The verdict.
 */
AssetVerdict asset_verdict(const Spec &sp, const Asset &a,
                           const AssemblyRef **out);

}  /* namespace k26rl_view */

#endif /* K26RL_VIEW_ASSET_H */
