/* assembly.c - the vehicle assembly reader and its mass-property
 * derivation.
 *
 * Three passes, in this order, because each needs the one before it:
 * read every file's bytes and digest them; parse the assembly and its
 * meshes; derive the mass properties.
 *
 * The mass-property integral is the divergence theorem applied to a
 * closed triangulated surface. Each triangle forms a tetrahedron with
 * the component origin whose signed volume is one sixth of the
 * determinant of its three vertices; summing those signed volumes
 * over a closed surface gives the enclosed volume, and the first and
 * second moments follow from the same determinant with the vertex
 * products the tetrahedron integral supplies. The method is standard.
 * A common shortcut is to compute the volume this way and then take
 * the inertia from a bounding box, which is wrong by tens of per cent
 * for anything long and thin; the full integral costs one more
 * accumulator per moment and is what this file does.
 *
 * The coefficients below are checked against closed forms in
 * tests/test_assembly.c rather than trusted: a triangulated box,
 * sphere, and capsule each reproduce the analytic inertia of the
 * solid they approximate, so a transcription error fails a gate
 * rather than shifting every vehicle's dynamics by a few per cent.
 *
 * Density is derived, never declared: a component states its mass,
 * and the uniform density that reproduces that mass over the computed
 * volume scales the second moments. Authors know and publish masses;
 * nobody publishes a spacecraft's mean density, and a default one
 * would be an invented number sitting inside every derived tensor.
 *
 * Convention: the derived tensor is the one that gives angular
 * momentum when applied to an angular velocity, so its off-diagonal
 * entries are the negated products of inertia. Published tables often
 * print the positive integrals instead, and the docking standard's
 * mass-property table says so explicitly; a cross-check against one
 * negates them first.
 *
 * Arithmetic is addition, subtraction, multiplication, division, and,
 * at one site, square root. Every one of those is correctly rounded
 * under IEEE-754, so under this directory's float-control flags the
 * derived constants are a function of the asset bytes alone. The one
 * root is the length of a capsule's axis, which its closed form needs
 * and which no rearrangement removes.
 */
#include "assembly.h"
#include "internal.h"

#include "k26rl_digest.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Ratio of a circle's circumference to its diameter, to the last bit
 * binary64 carries. Used only by the primitive closed forms; the mesh
 * path never needs it. */
#define ASM_PI 3.14159265358979311599796346854418516159057617187500

/* A declared rotation must already be a unit quaternion: this reader
 * refuses one that is not rather than normalising it, so a typed
 * quaternion is a diagnostic instead of a silently different
 * orientation. The bound is on the squared norm, which needs no root. */
#define ASM_QUAT_TOL 1.0e-9

/* A declared direction is a direction, and this reader refuses one
 * that is not of unit length on the same terms as the quaternion
 * above. The consequence of normalising instead is not cosmetic: a
 * thruster direction of twice unit length would deliver twice the
 * thrust the author declared, and a wheel axis of twice unit length
 * would put twice the torque on the body while the wheel's own rate
 * and its saturation clamp still used the unscaled scalar, so the
 * declaration and the behaviour would disagree in two directions at
 * once. The bound is on the squared norm, which needs no root. */
#define ASM_AXIS_TOL 1.0e-9

/* ---- File bytes ----------------------------------------------------- */

typedef struct {
    char  *bytes;
    size_t len;
} AsmFile;

static int asm_read_file_(const char *path, KflcArena *arena, AsmFile *out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return 1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 1; }
    char *buf = (char *)kflc_arena_alloc(arena, (size_t)sz + 1);
    if (!buf) { fclose(f); return 1; }
    size_t got = sz > 0 ? fread(buf, 1, (size_t)sz, f) : 0;
    fclose(f);
    if (got != (size_t)sz) return 1;
    buf[sz] = '\0';
    out->bytes = buf;
    out->len   = (size_t)sz;
    return 0;
}

/* Resolve `path` against the directory of `src_path`. An absolute
 * path is taken as it stands. */
static void asm_resolve_(const char *path, const char *src_path,
                         char *out, size_t out_sz)
{
    const char *slash = src_path ? strrchr(src_path, '/') : NULL;
    if (path[0] == '/' || !slash) {
        snprintf(out, out_sz, "%s", path);
        return;
    }
    snprintf(out, out_sz, "%.*s/%s", (int)(slash - src_path), src_path, path);
}

/* ---- Line scanning -------------------------------------------------- */

/* One logical line: the text between newlines with any comment
 * removed. Tokens are whitespace-separated, except that a token may
 * be a double-quoted string, which is how a provenance source travels
 * with its spaces intact. */
typedef struct {
    const char *p;
    const char *end;
    int         line;
} AsmScan;

typedef struct {
    char text[KFLC_ASM_SRC_MAX];
    int  quoted;
} AsmTok;

static void asm_scan_init_(AsmScan *s, const AsmFile *f)
{
    s->p    = f->bytes;
    s->end  = f->bytes + f->len;
    s->line = 0;
}

/* Advance to the next non-blank line and split it into at most `max`
 * tokens. Returns the token count, or -1 at end of file. */
static int asm_next_line_(AsmScan *s, AsmTok *tok, int max)
{
    while (s->p < s->end) {
        const char *nl   = memchr(s->p, '\n', (size_t)(s->end - s->p));
        const char *lend = nl ? nl : s->end;
        const char *lp   = s->p;
        s->p = nl ? nl + 1 : s->end;
        s->line++;

        int         in_q = 0;
        const char *cut  = lend;
        for (const char *q = lp; q < lend; q++) {
            if (*q == '"') in_q = !in_q;
            else if (*q == '#' && !in_q) { cut = q; break; }
        }

        int         n = 0;
        const char *q = lp;
        while (q < cut && n < max) {
            while (q < cut && isspace((unsigned char)*q)) q++;
            if (q >= cut) break;
            size_t len = 0;
            if (*q == '"') {
                q++;
                tok[n].quoted = 1;
                while (q < cut && *q != '"' && len + 1 < sizeof tok[n].text) {
                    tok[n].text[len++] = *q++;
                }
                if (q < cut && *q == '"') q++;
            } else {
                tok[n].quoted = 0;
                while (q < cut && !isspace((unsigned char)*q) &&
                       len + 1 < sizeof tok[n].text) {
                    tok[n].text[len++] = *q++;
                }
            }
            tok[n].text[len] = '\0';
            n++;
        }
        if (n > 0) return n;
    }
    return -1;
}

/* Copy a declared name, refusing one that does not fit rather than
 * truncating it. A truncated name would collide with another that
 * shares its first bytes, and the collision would surface as two
 * components quietly becoming one. Returns 1 when it does not fit. */
static int asm_copy_name_(char *dst, size_t cap, const char *src)
{
    size_t n = strlen(src);
    if (n + 1 > cap) return 1;
    memcpy(dst, src, n + 1);
    return 0;
}

/* Parse one token as a number, requiring the whole token to be
 * consumed. The strict consumption is not fussiness: it catches a
 * typed unit suffix, and it turns a locale whose decimal separator is
 * not a point into a loud refusal rather than a silently truncated
 * value. */
static int asm_num_(const char *tok, double *out)
{
    if (!tok || !*tok) return 1;
    char  *endp = NULL;
    double v    = strtod(tok, &endp);
    if (!endp || *endp != '\0') return 1;
    *out = v;
    return 0;
}

/* ---- Mesh ----------------------------------------------------------- */

typedef struct {
    double *v;
    int     n_v;
    int    *t;
    int    *t_line;
    int     n_t;
} AsmMesh;

typedef struct { int u, v, dir, line; } AsmEdge;

static int asm_edge_cmp_(const void *pa, const void *pb)
{
    const AsmEdge *a = (const AsmEdge *)pa, *b = (const AsmEdge *)pb;
    if (a->u != b->u)       return a->u < b->u ? -1 : 1;
    if (a->v != b->v)       return a->v < b->v ? -1 : 1;
    if (a->dir != b->dir)   return a->dir < b->dir ? -1 : 1;
    if (a->line != b->line) return a->line < b->line ? -1 : 1;
    return 0;
}

/* Every edge of a closed, consistently wound surface appears exactly
 * twice, once in each direction. Anything else is refused: the
 * integral below is defined over a closed oriented surface and
 * returns arithmetic without meaning over anything else. The
 * comparator is a total order down to the source line, so the
 * diagnostic names the same face on every run. */
static int asm_mesh_check_closed_(const AsmMesh *m, const char *path,
                                  KflcArena *arena, KflcDiag *diag)
{
    if (m->n_t == 0) {
        kflc_diag_errorf(diag, 0, "%s: mesh carries no faces", path);
        return 1;
    }
    int      n = m->n_t * 3;
    AsmEdge *e = (AsmEdge *)kflc_arena_alloc(arena, (size_t)n * sizeof *e);
    if (!e) return 1;
    int k = 0;
    for (int i = 0; i < m->n_t; i++) {
        for (int j = 0; j < 3; j++) {
            int a = m->t[3 * i + j];
            int b = m->t[3 * i + (j + 1) % 3];
            e[k].u    = a < b ? a : b;
            e[k].v    = a < b ? b : a;
            e[k].dir  = a < b ? 0 : 1;
            e[k].line = m->t_line[i];
            k++;
        }
    }
    qsort(e, (size_t)n, sizeof *e, asm_edge_cmp_);
    for (int i = 0; i < n; ) {
        int j = i;
        while (j < n && e[j].u == e[i].u && e[j].v == e[i].v) j++;
        int fwd = 0, rev = 0;
        for (int q = i; q < j; q++) {
            if (e[q].dir) rev++; else fwd++;
        }
        if (j - i != 2) {
            kflc_diag_errorf(diag, e[i].line,
                "%s: mesh is not closed: the edge between vertices %d "
                "and %d is used by %d face(s), not 2",
                path, e[i].u + 1, e[i].v + 1, j - i);
            return 1;
        }
        if (fwd != 1 || rev != 1) {
            kflc_diag_errorf(diag, e[i].line,
                "%s: mesh winding is inconsistent: the edge between "
                "vertices %d and %d is traversed the same way by both "
                "its faces, so the outward normal is not defined",
                path, e[i].u + 1, e[i].v + 1);
            return 1;
        }
        i = j;
    }
    return 0;
}

/* The accepted mesh form is a strict subset of the widely authored
 * vertex-and-face line format: `v x y z` and `f i j k` with positive
 * one-based indices and triangles only. Every other directive is
 * refused by name, so a file carrying material or normal data is a
 * diagnostic rather than a silent partial read. */
static int asm_mesh_load_(const char *path, KflcArena *arena,
                          KflcDiag *diag, AsmMesh *out, AsmFile *bytes)
{
    if (asm_read_file_(path, arena, bytes)) {
        kflc_diag_errorf(diag, 0, "cannot open mesh `%s`", path);
        return 1;
    }
    AsmScan s;
    AsmTok  tok[8];
    int     n_v = 0, n_t = 0, n;

    asm_scan_init_(&s, bytes);
    while ((n = asm_next_line_(&s, tok, 8)) >= 0) {
        if (strcmp(tok[0].text, "v") == 0)      n_v++;
        else if (strcmp(tok[0].text, "f") == 0) n_t++;
        else {
            kflc_diag_errorf(diag, s.line,
                "%s: `%s` is not part of the accepted mesh subset, which "
                "is `v x y z` and `f i j k` with comments",
                path, tok[0].text);
            return 1;
        }
    }
    out->n_v    = n_v;
    out->n_t    = n_t;
    out->v      = (double *)kflc_arena_alloc(arena,
                      (size_t)(n_v > 0 ? n_v : 1) * 3 * sizeof(double));
    out->t      = (int *)kflc_arena_alloc(arena,
                      (size_t)(n_t > 0 ? n_t : 1) * 3 * sizeof(int));
    out->t_line = (int *)kflc_arena_alloc(arena,
                      (size_t)(n_t > 0 ? n_t : 1) * sizeof(int));
    if (!out->v || !out->t || !out->t_line) return 1;

    int iv = 0, it = 0;
    asm_scan_init_(&s, bytes);
    while ((n = asm_next_line_(&s, tok, 8)) >= 0) {
        if (strcmp(tok[0].text, "v") == 0) {
            if (n != 4) {
                kflc_diag_errorf(diag, s.line,
                    "%s: `v` takes exactly three coordinates", path);
                return 1;
            }
            for (int i = 0; i < 3; i++) {
                if (asm_num_(tok[1 + i].text, &out->v[3 * iv + i])) {
                    kflc_diag_errorf(diag, s.line,
                        "%s: `%s` is not a number", path, tok[1 + i].text);
                    return 1;
                }
            }
            iv++;
        } else {
            if (n != 4) {
                kflc_diag_errorf(diag, s.line,
                    "%s: `f` takes exactly three vertex indices: this "
                    "subset carries triangles only", path);
                return 1;
            }
            for (int i = 0; i < 3; i++) {
                double d;
                if (asm_num_(tok[1 + i].text, &d)) {
                    kflc_diag_errorf(diag, s.line,
                        "%s: `%s` is not a vertex index", path,
                        tok[1 + i].text);
                    return 1;
                }
                int idx = (int)d;
                if ((double)idx != d || idx < 1 || idx > n_v) {
                    kflc_diag_errorf(diag, s.line,
                        "%s: vertex index %s is not a whole number in 1 "
                        "to %d", path, tok[1 + i].text, n_v);
                    return 1;
                }
                out->t[3 * it + i] = idx - 1;
            }
            out->t_line[it] = s.line;
            it++;
        }
    }
    return asm_mesh_check_closed_(out, path, arena, diag);
}

/* ---- Mass properties ------------------------------------------------ */

/* Volume, centre of mass, and inertia at unit density. The inertia is
 * about the centre of mass, in the frame the geometry was given in. */
typedef struct {
    double vol;
    double com[3];
    double mom[6];      /* xx, yy, zz, xy, xz, yz */
} AsmProps;

/* Add the parallel-axis term for a body of mass m whose centre of
 * mass sits at offset d from the point the tensor is wanted about. */
static void asm_shift_(double m, const double d[3], double mom[6])
{
    double dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    mom[0] += m * (dd - d[0] * d[0]);
    mom[1] += m * (dd - d[1] * d[1]);
    mom[2] += m * (dd - d[2] * d[2]);
    mom[3] += m * (-d[0] * d[1]);
    mom[4] += m * (-d[0] * d[2]);
    mom[5] += m * (-d[1] * d[2]);
}

static void asm_mesh_props_(const AsmMesh *m, AsmProps *out)
{
    double sv    = 0.0;
    double s1[3] = { 0.0, 0.0, 0.0 };
    double s2[3] = { 0.0, 0.0, 0.0 };
    double sp[3] = { 0.0, 0.0, 0.0 };
    static const int P[3][2] = { { 0, 1 }, { 0, 2 }, { 1, 2 } };

    for (int i = 0; i < m->n_t; i++) {
        const double *a = &m->v[3 * m->t[3 * i]];
        const double *b = &m->v[3 * m->t[3 * i + 1]];
        const double *c = &m->v[3 * m->t[3 * i + 2]];

        double det = a[0] * (b[1] * c[2] - b[2] * c[1])
                   + a[1] * (b[2] * c[0] - b[0] * c[2])
                   + a[2] * (b[0] * c[1] - b[1] * c[0]);
        sv += det;
        for (int k = 0; k < 3; k++) {
            s1[k] += det * (a[k] + b[k] + c[k]);
            s2[k] += det * (a[k] * a[k] + b[k] * b[k] + c[k] * c[k]
                            + a[k] * b[k] + a[k] * c[k] + b[k] * c[k]);
        }
        for (int k = 0; k < 3; k++) {
            int x = P[k][0], y = P[k][1];
            sp[k] += det * (2.0 * (a[x] * a[y] + b[x] * b[y] + c[x] * c[y])
                            + a[x] * b[y] + b[x] * a[y]
                            + a[x] * c[y] + c[x] * a[y]
                            + b[x] * c[y] + c[x] * b[y]);
        }
    }

    double V = sv / 6.0;
    out->vol = V;
    if (V == 0.0) {
        for (int k = 0; k < 3; k++) out->com[k] = 0.0;
        for (int k = 0; k < 6; k++) out->mom[k] = 0.0;
        return;
    }
    double Mxx = s2[0] / 60.0,  Myy = s2[1] / 60.0,  Mzz = s2[2] / 60.0;
    double Mxy = sp[0] / 120.0, Mxz = sp[1] / 120.0, Myz = sp[2] / 120.0;
    for (int k = 0; k < 3; k++) out->com[k] = (s1[k] / 24.0) / V;

    out->mom[0] = Myy + Mzz;
    out->mom[1] = Mxx + Mzz;
    out->mom[2] = Mxx + Myy;
    out->mom[3] = -Mxy;
    out->mom[4] = -Mxz;
    out->mom[5] = -Myz;
    /* Shift from the frame origin to the centre of mass: subtract the
     * parallel-axis term a point mass of the same volume would add. */
    double neg[3] = { out->com[0], out->com[1], out->com[2] };
    double back[6] = { 0, 0, 0, 0, 0, 0 };
    asm_shift_(V, neg, back);
    for (int k = 0; k < 6; k++) out->mom[k] -= back[k];
}

/* Closed forms for the declared primitives, at unit density, about
 * each primitive's own centre of mass. A component with no mesh
 * derives its mass properties from the primitives it declares. */
static void asm_shape_props_(const KflcAsmCollider *s, AsmProps *out)
{
    for (int k = 0; k < 6; k++) out->mom[k] = 0.0;
    for (int k = 0; k < 3; k++) out->com[k] = 0.0;

    if (s->kind == KFLC_SHAPE_SPHERE) {
        double r = s->r, r2 = r * r;
        out->vol = (4.0 / 3.0) * ASM_PI * r2 * r;
        double i = (2.0 / 5.0) * out->vol * r2;
        out->mom[0] = out->mom[1] = out->mom[2] = i;
        for (int k = 0; k < 3; k++) out->com[k] = s->a[k];
        return;
    }
    if (s->kind == KFLC_SHAPE_BOX) {
        double hx = s->a[0], hy = s->a[1], hz = s->a[2];
        out->vol = 8.0 * hx * hy * hz;
        double m = out->vol;
        out->mom[0] = m * (hy * hy + hz * hz) / 3.0;
        out->mom[1] = m * (hx * hx + hz * hz) / 3.0;
        out->mom[2] = m * (hx * hx + hy * hy) / 3.0;
        return;
    }

    /* Capsule: a cylinder of radius r and length L capped by two
     * hemispheres. Assembled about the capsule centre. The cylinder
     * contributes its own moments; each hemisphere contributes its
     * moment about its own centre of mass, which sits 3r/8 from its
     * flat face, shifted by the distance from there to the capsule
     * centre. The axial and transverse moments are then combined into
     * the component frame through the axis direction, which is where
     * the one square root in this file is spent. */
    double d[3];
    for (int k = 0; k < 3; k++) {
        d[k] = s->b[k] - s->a[k];
        out->com[k] = 0.5 * (s->a[k] + s->b[k]);
    }
    double L2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    double r  = s->r, r2 = r * r;
    if (L2 == 0.0) {
        /* Coincident endpoints are a sphere, and are treated as one
         * rather than as a degenerate capsule. */
        KflcAsmCollider sph = *s;
        sph.kind = KFLC_SHAPE_SPHERE;
        double keep[3] = { out->com[0], out->com[1], out->com[2] };
        asm_shape_props_(&sph, out);
        for (int k = 0; k < 3; k++) out->com[k] = keep[k];
        return;
    }
    double L      = sqrt(L2);
    double v_cyl  = ASM_PI * r2 * L;
    double v_hemi = (2.0 / 3.0) * ASM_PI * r2 * r;
    out->vol = v_cyl + 2.0 * v_hemi;

    double axial = v_cyl * r2 / 2.0 + 2.0 * (2.0 / 5.0) * v_hemi * r2;
    double arm   = L / 2.0 + 3.0 * r / 8.0;
    double trans = v_cyl * (3.0 * r2 + L * L) / 12.0
                 + 2.0 * ((83.0 / 320.0) * v_hemi * r2 + v_hemi * arm * arm);

    double u[3] = { d[0] / L, d[1] / L, d[2] / L };
    double k    = axial - trans;
    out->mom[0] = trans + k * u[0] * u[0];
    out->mom[1] = trans + k * u[1] * u[1];
    out->mom[2] = trans + k * u[2] * u[2];
    out->mom[3] = k * u[0] * u[1];
    out->mom[4] = k * u[0] * u[2];
    out->mom[5] = k * u[1] * u[2];
}

/* ---- Rotation -------------------------------------------------------- */

static void asm_quat_matrix_(const double q[4], double R[3][3])
{
    double w = q[0], x = q[1], y = q[2], z = q[3];
    R[0][0] = 1.0 - 2.0 * (y * y + z * z);
    R[0][1] = 2.0 * (x * y - w * z);
    R[0][2] = 2.0 * (x * z + w * y);
    R[1][0] = 2.0 * (x * y + w * z);
    R[1][1] = 1.0 - 2.0 * (x * x + z * z);
    R[1][2] = 2.0 * (y * z - w * x);
    R[2][0] = 2.0 * (x * z - w * y);
    R[2][1] = 2.0 * (y * z + w * x);
    R[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

static void asm_rot_vec_(double R[3][3], const double v[3], double out[3])
{
    for (int i = 0; i < 3; i++) {
        out[i] = R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2];
    }
}

/* Rotate a symmetric tensor: R I R^T, in the six-component form. */
static void asm_rot_tensor_(double R[3][3], const double in[6], double out[6])
{
    double I[3][3] = {
        { in[0], in[3], in[4] },
        { in[3], in[1], in[5] },
        { in[4], in[5], in[2] }
    };
    double T[3][3], O[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            T[i][j] = R[i][0] * I[0][j] + R[i][1] * I[1][j]
                    + R[i][2] * I[2][j];
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            O[i][j] = T[i][0] * R[j][0] + T[i][1] * R[j][1]
                    + T[i][2] * R[j][2];
        }
    }
    out[0] = O[0][0]; out[1] = O[1][1]; out[2] = O[2][2];
    out[3] = O[0][1]; out[4] = O[0][2]; out[5] = O[1][2];
}

/* ---- Assembly parsing ------------------------------------------------ */

typedef enum {
    ASM_BLK_NONE = 0,
    ASM_BLK_COMPONENT,
    ASM_BLK_PORT,
    ASM_BLK_THRUSTER,
    ASM_BLK_WHEEL,
    ASM_BLK_TORQUER
} AsmBlock;

static int asm_name_taken_(const KflcAssembly *a, AsmBlock blk,
                           const char *name)
{
    if (blk == ASM_BLK_COMPONENT) {
        for (int i = 0; i < a->n_components; i++) {
            if (strcmp(a->components[i].name, name) == 0) return 1;
        }
        return 0;
    }
    KflcAsmFeatureKind k = blk == ASM_BLK_PORT     ? KFLC_FEAT_PORT
                         : blk == ASM_BLK_THRUSTER ? KFLC_FEAT_THRUSTER
                         : blk == ASM_BLK_WHEEL    ? KFLC_FEAT_WHEEL
                                                   : KFLC_FEAT_TORQUER;
    for (int i = 0; i < a->n_features; i++) {
        if (a->features[i].kind == k &&
            strcmp(a->features[i].name, name) == 0) return 1;
    }
    return 0;
}

/* Every refusal below names the assembly file, the line, and what was
 * found. An assembly is program identity, so a partially understood
 * one is not a program and is never accepted with a warning. */
KflcAssembly *kflc_assembly_load(const char *path, const char *src_path,
                                 int line, KflcArena *arena, KflcDiag *diag)
{
    char resolved[KFLC_ASM_PATH_MAX];
    asm_resolve_(path, src_path, resolved, sizeof resolved);

    AsmFile af;
    if (asm_read_file_(resolved, arena, &af)) {
        kflc_diag_errorf(diag, line, "cannot open assembly `%s`", resolved);
        return NULL;
    }

    KflcAssembly *a = (KflcAssembly *)kflc_arena_alloc(arena, sizeof *a);
    if (!a) return NULL;
    memset(a, 0, sizeof *a);
    snprintf(a->path, sizeof a->path, "%s", resolved);

    /* The identity digest opens with the assembly's own bytes; each
     * mesh follows in first-reference order as it is read. */
    K26RlSha256 dig;
    k26rl_digest_begin(&dig);
    k26rl_digest_add(&dig, af.bytes, (uint64_t)af.len);

    AsmScan  s;
    AsmTok   tok[16];
    int      n;
    AsmBlock blk      = ASM_BLK_NONE;
    int      in_asm   = 0;
    int      have_frame = 0;
    KflcAsmComponent *comp = NULL;
    KflcAsmFeature   *feat = NULL;
    asm_scan_init_(&s, &af);
    while ((n = asm_next_line_(&s, tok, 16)) >= 0) {
        const char *kw = tok[0].text;

#define ASM_ERR(...) do { \
        kflc_diag_errorf(diag, s.line, __VA_ARGS__); return NULL; } while (0)
#define ASM_NEED(cnt) do { if (n != (cnt)) \
        ASM_ERR("%s: `%s` takes %d value(s)", resolved, kw, (cnt) - 1); \
    } while (0)
#define ASM_NUM(i, dst) do { if (asm_num_(tok[i].text, dst)) \
        ASM_ERR("%s: `%s` is not a number", resolved, tok[i].text); \
    } while (0)
#define ASM_VEC3(first, dst) do { \
        ASM_NEED((first) + 3); \
        for (int q_ = 0; q_ < 3; q_++) ASM_NUM((first) + q_, &(dst)[q_]); \
    } while (0)

        if (!in_asm) {
            if (strcmp(kw, "assembly") != 0) {
                ASM_ERR("%s: expected `assembly <name>`, found `%s`",
                        resolved, kw);
            }
            ASM_NEED(2);
            if (asm_copy_name_(a->name, sizeof a->name, tok[1].text)) {
                ASM_ERR("%s: assembly name is longer than %d bytes",
                        resolved, (int)sizeof a->name - 1);
            }
            in_asm = 1;
            continue;
        }

        if (strcmp(kw, "end") == 0) {
            if (blk != ASM_BLK_NONE) {
                if (blk == ASM_BLK_COMPONENT && comp && comp->mass <= 0.0) {
                    ASM_ERR("%s: component `%s` declares no positive mass",
                            resolved, comp->name);
                }
                blk  = ASM_BLK_NONE;
                comp = NULL;
                feat = NULL;
            } else if (in_asm) {
                in_asm = 2;
            }
            continue;
        }

        if (blk == ASM_BLK_NONE) {
            if (strcmp(kw, "frame") == 0) {
                ASM_NEED(2);
                if (strcmp(tok[1].text, "x_to_port") != 0) {
                    ASM_ERR("%s: `frame %s` is not a defined body frame; "
                            "version 1 defines `x_to_port`, whose x axis "
                            "runs along the vehicle's longitudinal axis "
                            "positive toward the docking interface",
                            resolved, tok[1].text);
                }
                have_frame = 1;
                continue;
            }
            if (strcmp(kw, "provenance") == 0) {
                if (n != 4) {
                    ASM_ERR("%s: `provenance` takes a property, a quoted "
                            "source, and a status", resolved);
                }
                if (a->n_provenance >= KFLC_ASM_MAX_PROV) {
                    ASM_ERR("%s: more than %d provenance lines", resolved,
                            KFLC_ASM_MAX_PROV);
                }
                KflcAsmProvenance *p = &a->provenance[a->n_provenance++];
                if (asm_copy_name_(p->property, sizeof p->property,
                                   tok[1].text)) {
                    ASM_ERR("%s: provenance property name is longer than "
                            "%d bytes", resolved,
                            (int)sizeof p->property - 1);
                }
                snprintf(p->source, sizeof p->source, "%s", tok[2].text);
                p->line = s.line;
                if (strcmp(tok[3].text, "cited") == 0) {
                    p->status = KFLC_PROV_CITED;
                } else if (strcmp(tok[3].text, "computed") == 0) {
                    p->status = KFLC_PROV_COMPUTED;
                } else if (strcmp(tok[3].text, "unverified") == 0) {
                    p->status = KFLC_PROV_UNVERIFIED;
                    a->n_unverified++;
                } else {
                    ASM_ERR("%s: `%s` is not a provenance status; the set "
                            "is cited, computed, unverified",
                            resolved, tok[3].text);
                }
                continue;
            }
            if (strcmp(kw, "component") == 0 || strcmp(kw, "port") == 0 ||
                strcmp(kw, "thruster") == 0 || strcmp(kw, "wheel") == 0 ||
                strcmp(kw, "magnetorquer") == 0) {
                ASM_NEED(2);
                blk = strcmp(kw, "component") == 0 ? ASM_BLK_COMPONENT
                    : strcmp(kw, "port") == 0      ? ASM_BLK_PORT
                    : strcmp(kw, "thruster") == 0  ? ASM_BLK_THRUSTER
                    : strcmp(kw, "wheel") == 0     ? ASM_BLK_WHEEL
                                                   : ASM_BLK_TORQUER;
                if (asm_name_taken_(a, blk, tok[1].text)) {
                    ASM_ERR("%s: a %s named `%s` is already declared",
                            resolved, kw, tok[1].text);
                }
                if (blk == ASM_BLK_COMPONENT) {
                    if (a->n_components >= KFLC_ASM_MAX_COMP) {
                        ASM_ERR("%s: more than %d components", resolved,
                                KFLC_ASM_MAX_COMP);
                    }
                    comp = &a->components[a->n_components++];
                    if (asm_copy_name_(comp->name, sizeof comp->name,
                                       tok[1].text)) {
                        ASM_ERR("%s: component name is longer than %d "
                                "bytes", resolved,
                                (int)sizeof comp->name - 1);
                    }
                    comp->rot[0] = 1.0;
                    comp->line   = s.line;
                } else {
                    if (a->n_features >= KFLC_ASM_MAX_FEAT) {
                        ASM_ERR("%s: more than %d features", resolved,
                                KFLC_ASM_MAX_FEAT);
                    }
                    feat = &a->features[a->n_features++];
                    memset(feat, 0, sizeof *feat);
                    feat->kind = blk == ASM_BLK_PORT     ? KFLC_FEAT_PORT
                               : blk == ASM_BLK_THRUSTER ? KFLC_FEAT_THRUSTER
                               : blk == ASM_BLK_WHEEL    ? KFLC_FEAT_WHEEL
                                                         : KFLC_FEAT_TORQUER;
                    if (asm_copy_name_(feat->name, sizeof feat->name,
                                       tok[1].text)) {
                        ASM_ERR("%s: %s name is longer than %d bytes",
                                resolved, kw, (int)sizeof feat->name - 1);
                    }
                    feat->line = s.line;
                }
                continue;
            }
            ASM_ERR("%s: `%s` is not an assembly-level key", resolved, kw);
        }

        if (blk == ASM_BLK_COMPONENT) {
            if (strcmp(kw, "mass") == 0) {
                ASM_NEED(2);
                ASM_NUM(1, &comp->mass);
                if (!(comp->mass > 0.0)) {
                    ASM_ERR("%s: component mass must be positive", resolved);
                }
                continue;
            }
            if (strcmp(kw, "at") == 0)     { ASM_VEC3(1, comp->at); continue; }
            if (strcmp(kw, "rotate") == 0) {
                ASM_NEED(5);
                for (int i = 0; i < 4; i++) ASM_NUM(1 + i, &comp->rot[i]);
                double nn = comp->rot[0] * comp->rot[0]
                          + comp->rot[1] * comp->rot[1]
                          + comp->rot[2] * comp->rot[2]
                          + comp->rot[3] * comp->rot[3];
                double dev = nn - 1.0;
                if (dev < 0.0) dev = -dev;
                if (dev > ASM_QUAT_TOL) {
                    ASM_ERR("%s: `rotate` is not a unit quaternion (its "
                            "squared norm is %.17g); this reader refuses "
                            "one rather than normalising it, so a typed "
                            "value is a diagnostic instead of a different "
                            "orientation", resolved, nn);
                }
                comp->has_rot = 1;
                continue;
            }
            if (strcmp(kw, "mesh") == 0) {
                ASM_NEED(2);
                snprintf(comp->mesh, sizeof comp->mesh, "%s", tok[1].text);
                comp->has_mesh = 1;
                continue;
            }
            if (strcmp(kw, "collider") == 0) {
                if (a->n_colliders >= KFLC_ASM_MAX_COLL) {
                    ASM_ERR("%s: more than %d colliders", resolved,
                            KFLC_ASM_MAX_COLL);
                }
                KflcAsmCollider *c = &a->colliders[a->n_colliders];
                memset(c, 0, sizeof *c);
                c->component = a->n_components - 1;
                c->line      = s.line;
                if (n < 2) ASM_ERR("%s: `collider` needs a shape", resolved);
                if (strcmp(tok[1].text, "sphere") == 0) {
                    ASM_NEED(6);
                    c->kind = KFLC_SHAPE_SPHERE;
                    for (int i = 0; i < 3; i++) ASM_NUM(2 + i, &c->a[i]);
                    ASM_NUM(5, &c->r);
                } else if (strcmp(tok[1].text, "capsule") == 0) {
                    ASM_NEED(9);
                    c->kind = KFLC_SHAPE_CAPSULE;
                    for (int i = 0; i < 3; i++) ASM_NUM(2 + i, &c->a[i]);
                    for (int i = 0; i < 3; i++) ASM_NUM(5 + i, &c->b[i]);
                    ASM_NUM(8, &c->r);
                } else if (strcmp(tok[1].text, "box") == 0) {
                    ASM_NEED(5);
                    c->kind = KFLC_SHAPE_BOX;
                    for (int i = 0; i < 3; i++) ASM_NUM(2 + i, &c->a[i]);
                } else {
                    ASM_ERR("%s: `%s` is not a collider shape; version 1 "
                            "carries sphere, capsule, and box",
                            resolved, tok[1].text);
                }
                if (c->kind != KFLC_SHAPE_BOX && !(c->r > 0.0)) {
                    ASM_ERR("%s: collider radius must be positive", resolved);
                }
                if (c->kind == KFLC_SHAPE_BOX &&
                    !(c->a[0] > 0.0 && c->a[1] > 0.0 && c->a[2] > 0.0)) {
                    ASM_ERR("%s: box half extents must be positive",
                            resolved);
                }
                a->n_colliders++;
                continue;
            }
            ASM_ERR("%s: `%s` is not a component key", resolved, kw);
        }

        /* Feature blocks. Mass is refused on all of them: a feature's
         * mass belongs to the component that holds it, so there is one
         * declaration site per quantity. */
        if (strcmp(kw, "mass") == 0) {
            ASM_ERR("%s: a feature carries no mass of its own; declare it "
                    "on the component that holds this one", resolved);
        }
        if (blk == ASM_BLK_PORT) {
            if (strcmp(kw, "at") == 0)       { ASM_VEC3(1, feat->at); continue; }
            if (strcmp(kw, "axis") == 0)     { ASM_VEC3(1, feat->axis); continue; }
            if (strcmp(kw, "roll_ref") == 0) { ASM_VEC3(1, feat->roll_ref); continue; }
            if (strcmp(kw, "capture") == 0) {
                ASM_NEED(2);
                if (asm_copy_name_(feat->capture, sizeof feat->capture,
                                   tok[1].text)) {
                    ASM_ERR("%s: capture envelope name is longer than %d "
                            "bytes", resolved,
                            (int)sizeof feat->capture - 1);
                }
                continue;
            }
            ASM_ERR("%s: `%s` is not a port key", resolved, kw);
        }
        if (blk == ASM_BLK_THRUSTER) {
            if (strcmp(kw, "at") == 0)  { ASM_VEC3(1, feat->at); continue; }
            if (strcmp(kw, "dir") == 0) { ASM_VEC3(1, feat->dir); continue; }
            if (strcmp(kw, "thrust") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->thrust); continue;
            }
            ASM_ERR("%s: `%s` is not a thruster key", resolved, kw);
        }
        if (blk == ASM_BLK_WHEEL) {
            if (strcmp(kw, "axis") == 0) { ASM_VEC3(1, feat->axis); continue; }
            if (strcmp(kw, "spin_inertia") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->spin_inertia); continue;
            }
            if (strcmp(kw, "max_momentum") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->max_momentum); continue;
            }
            if (strcmp(kw, "max_torque") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->max_torque); continue;
            }
            if (strcmp(kw, "viscous") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->viscous); continue;
            }
            if (strcmp(kw, "coulomb") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->coulomb); continue;
            }
            if (strcmp(kw, "dead_rate") == 0) {
                ASM_NEED(2); ASM_NUM(1, &feat->dead_rate); continue;
            }
            ASM_ERR("%s: `%s` is not a wheel key", resolved, kw);
        }
        if (strcmp(kw, "axis") == 0) { ASM_VEC3(1, feat->axis); continue; }
        if (strcmp(kw, "max_dipole") == 0) {
            ASM_NEED(2); ASM_NUM(1, &feat->max_dipole); continue;
        }
        ASM_ERR("%s: `%s` is not a magnetorquer key", resolved, kw);
    }

    if (in_asm != 2) {
        kflc_diag_errorf(diag, line,
            "%s: the assembly block is not closed by `end`", resolved);
        return NULL;
    }
    if (!have_frame) {
        kflc_diag_errorf(diag, line,
            "%s: no `frame` declared; version 1 defines `x_to_port`",
            resolved);
        return NULL;
    }
    if (a->n_components == 0) {
        kflc_diag_errorf(diag, line,
            "%s: assembly declares no components, so it has no mass "
            "properties to derive", resolved);
        return NULL;
    }

    /* ---- Actuator declarations ------------------------------------ *
     *
     * Every quantity an actuator model divides by, clamps to, or
     * scales a command by is checked here, at the one place it is
     * declared, rather than left to produce a plausible number
     * downstream. Each of these was a silent wrong answer before it
     * was a diagnostic: a negative maximum inverts the clamp's bounds
     * so that a command of zero produces permanent uncommanded
     * torque, and a zero momentum limit disables saturation
     * altogether, which is the momentum-management task quietly
     * removed rather than solved. A wheel with no momentum limit is
     * not a wheel this model can represent. */
    for (int i = 0; i < a->n_features; i++) {
        const KflcAsmFeature *f = &a->features[i];
        if (f->kind == KFLC_FEAT_PORT) continue;

        if (f->kind == KFLC_FEAT_THRUSTER) {
            double nn = f->dir[0] * f->dir[0] + f->dir[1] * f->dir[1]
                      + f->dir[2] * f->dir[2];
            double dev = nn - 1.0;
            if (dev < 0.0) dev = -dev;
            if (dev > ASM_AXIS_TOL) {
                kflc_diag_errorf(diag, f->line,
                    "%s: thruster `%s`: `dir` is not a unit vector (its "
                    "squared norm is %.17g); this reader refuses one "
                    "rather than normalising it, so a typed direction is "
                    "a diagnostic instead of a different thrust",
                    resolved, f->name, nn);
                return NULL;
            }
            if (!(f->thrust > 0.0)) {
                kflc_diag_errorf(diag, f->line,
                    "%s: thruster `%s`: `thrust` is %.17g; a maximum is a "
                    "positive quantity, and a negative one inverts the "
                    "throttle clamp", resolved, f->name, f->thrust);
                return NULL;
            }
            continue;
        }

        {
            double nn = f->axis[0] * f->axis[0] + f->axis[1] * f->axis[1]
                      + f->axis[2] * f->axis[2];
            double dev = nn - 1.0;
            if (dev < 0.0) dev = -dev;
            if (dev > ASM_AXIS_TOL) {
                kflc_diag_errorf(diag, f->line,
                    "%s: %s `%s`: `axis` is not a unit vector (its "
                    "squared norm is %.17g); this reader refuses one "
                    "rather than normalising it, so a typed axis is a "
                    "diagnostic instead of a different torque",
                    resolved,
                    f->kind == KFLC_FEAT_WHEEL ? "wheel" : "magnetorquer",
                    f->name, nn);
                return NULL;
            }
        }

        if (f->kind == KFLC_FEAT_WHEEL) {
            if (!(f->spin_inertia > 0.0)) {
                kflc_diag_errorf(diag, f->line,
                    "%s: wheel `%s`: `spin_inertia` is %.17g; the wheel's "
                    "rate is its momentum divided by this, so it must be "
                    "positive", resolved, f->name, f->spin_inertia);
                return NULL;
            }
            if (!(f->max_torque > 0.0)) {
                kflc_diag_errorf(diag, f->line,
                    "%s: wheel `%s`: `max_torque` is %.17g; a maximum is "
                    "a positive quantity, and a negative one inverts the "
                    "command clamp", resolved, f->name, f->max_torque);
                return NULL;
            }
            if (!(f->max_momentum > 0.0)) {
                kflc_diag_errorf(diag, f->line,
                    "%s: wheel `%s`: `max_momentum` is %.17g; a wheel "
                    "with no momentum limit never saturates and never "
                    "needs dumping, which is not a wheel this model "
                    "represents", resolved, f->name, f->max_momentum);
                return NULL;
            }
            if (f->viscous < 0.0 || f->coulomb < 0.0 || f->dead_rate < 0.0) {
                kflc_diag_errorf(diag, f->line,
                    "%s: wheel `%s`: `viscous`, `coulomb` and `dead_rate` "
                    "describe friction and are never negative (they are "
                    "%.17g, %.17g and %.17g); a negative one drives the "
                    "wheel instead of slowing it",
                    resolved, f->name, f->viscous, f->coulomb,
                    f->dead_rate);
                return NULL;
            }
            continue;
        }

        if (!(f->max_dipole > 0.0)) {
            kflc_diag_errorf(diag, f->line,
                "%s: magnetorquer `%s`: `max_dipole` is %.17g; a maximum "
                "is a positive quantity, and a negative one inverts the "
                "command clamp", resolved, f->name, f->max_dipole);
            return NULL;
        }
    }

    /* ---- Derivation, components first, in source order ------------ */
    for (int i = 0; i < a->n_components; i++) {
        KflcAsmComponent *c = &a->components[i];
        AsmProps          p;
        memset(&p, 0, sizeof p);

        if (c->has_mesh) {
            char mpath[KFLC_ASM_PATH_MAX];
            asm_resolve_(c->mesh, resolved, mpath, sizeof mpath);
            AsmMesh mesh;
            AsmFile mbytes;
            memset(&mesh, 0, sizeof mesh);
            if (asm_mesh_load_(mpath, arena, diag, &mesh, &mbytes)) {
                return NULL;
            }
            k26rl_digest_add(&dig, mbytes.bytes, (uint64_t)mbytes.len);
            asm_mesh_props_(&mesh, &p);
            if (!(p.vol > 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "%s: mesh `%s` encloses no positive volume; a surface "
                    "wound inside out gives a negative one",
                    resolved, c->mesh);
                return NULL;
            }
        } else {
            /* No mesh: the declared primitives are the solid. */
            int found = 0;
            double vsum = 0.0, csum[3] = { 0.0, 0.0, 0.0 };
            AsmProps parts[KFLC_ASM_MAX_COLL];
            int      n_parts = 0;
            for (int k = 0; k < a->n_colliders; k++) {
                if (a->colliders[k].component != i) continue;
                asm_shape_props_(&a->colliders[k], &parts[n_parts]);
                vsum += parts[n_parts].vol;
                for (int q = 0; q < 3; q++) {
                    csum[q] += parts[n_parts].vol * parts[n_parts].com[q];
                }
                n_parts++;
                found = 1;
            }
            if (!found) {
                kflc_diag_errorf(diag, c->line,
                    "%s: component `%s` declares neither a mesh nor a "
                    "collider, so its mass properties cannot be derived",
                    resolved, c->name);
                return NULL;
            }
            p.vol = vsum;
            for (int q = 0; q < 3; q++) p.com[q] = csum[q] / vsum;
            for (int q = 0; q < 6; q++) p.mom[q] = 0.0;
            for (int k = 0; k < n_parts; k++) {
                double d[3];
                for (int q = 0; q < 3; q++) d[q] = parts[k].com[q] - p.com[q];
                for (int q = 0; q < 6; q++) p.mom[q] += parts[k].mom[q];
                asm_shift_(parts[k].vol, d, p.mom);
            }
        }

        /* Density is the declared mass over the computed volume. */
        double rho = c->mass / p.vol;
        c->volume = p.vol;
        for (int q = 0; q < 3; q++) c->com[q] = p.com[q];
        for (int q = 0; q < 6; q++) c->inertia[q] = rho * p.mom[q];
    }

    /* ---- Assembly totals ------------------------------------------ */
    double total = 0.0, csum[3] = { 0.0, 0.0, 0.0 };
    double wcom[KFLC_ASM_MAX_COMP][3];
    for (int i = 0; i < a->n_components; i++) {
        KflcAsmComponent *c = &a->components[i];
        double R[3][3];
        asm_quat_matrix_(c->rot, R);
        double rc[3];
        asm_rot_vec_(R, c->com, rc);
        for (int q = 0; q < 3; q++) {
            wcom[i][q] = c->at[q] + rc[q];
            csum[q]   += c->mass * wcom[i][q];
        }
        total += c->mass;
    }
    a->mass = total;
    for (int q = 0; q < 3; q++) a->com[q] = csum[q] / total;
    for (int q = 0; q < 6; q++) a->inertia[q] = 0.0;
    for (int i = 0; i < a->n_components; i++) {
        KflcAsmComponent *c = &a->components[i];
        double R[3][3], rot[6], d[3];
        asm_quat_matrix_(c->rot, R);
        asm_rot_tensor_(R, c->inertia, rot);
        for (int q = 0; q < 6; q++) a->inertia[q] += rot[q];
        for (int q = 0; q < 3; q++) d[q] = wcom[i][q] - a->com[q];
        asm_shift_(c->mass, d, a->inertia);
    }
    k26rl_digest_final(&dig, a->digest);

    /* The compiler reports what an asset says it has not checked. The
     * rule that a shipped asset carries none of these is the asset
     * gate's, not the language's: an author's own working figure is
     * theirs to keep. */
    for (int i = 0; i < a->n_provenance; i++) {
        if (a->provenance[i].status == KFLC_PROV_UNVERIFIED) {
            kflc_diag_warnf(diag, line,
                "%s: provenance of `%s` is unverified (%s)",
                resolved, a->provenance[i].property, a->provenance[i].source);
        }
    }
    return a;

#undef ASM_ERR
#undef ASM_NEED
#undef ASM_NUM
#undef ASM_VEC3
}

int kflc_assembly_for_body(const KflcNode *body, const char *src_path,
                           KflcArena *arena, KflcDiag *diag,
                           KflcAssembly **out)
{
    if (out) *out = NULL;
    if (!body) return 0;
    const KflcAttr *asm_attr = NULL, *mass_attr = NULL, *gm_attr = NULL;
    const KflcAttr *att_attr = NULL;
    for (const KflcAttr *a = body->attrs; a; a = a->next) {
        if (!a->name) continue;
        if (strcmp(a->name, "assembly") == 0)  asm_attr  = a;
        else if (strcmp(a->name, "mass") == 0) mass_attr = a;
        else if (strcmp(a->name, "gm") == 0)   gm_attr   = a;
        else if (!att_attr && kflc_body_state_is_attitude(a->name)) {
            att_attr = a;
        }
    }
    if (!asm_attr) {
        /* Attitude on a body that cannot be advanced. The advance
         * needs an inertia tensor, which comes from an assembly, so a
         * body without one never rotates however it is declared: it
         * would report the angular velocity it was given while its
         * orientation stood still, and a policy would train against a
         * body that is spinning and static at once. Refused rather
         * than warned, because the program cannot mean what it says.
         *
         * The refusal is additive: no Grammar 3.1 program carries an
         * attitude key, since the keys arrive with this surface. */
        if (att_attr) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: `%s=` sets attitude state, but the body "
                "declares no `assembly=`, so it has no inertia tensor and "
                "its attitude is never advanced; bind an assembly or drop "
                "the attitude keys",
                body->name ? body->name : "_anon", att_attr->name);
            return 1;
        }
        return 0;
    }

    const KflcAttr *clash = mass_attr ? mass_attr : gm_attr;
    if (clash) {
        kflc_diag_errorf(diag, body->line,
            "astro_body `%s`: `%s=` on line %d and `assembly=` on line %d "
            "both set the body's mass; the assembly derives it from the "
            "geometry, so drop the other one",
            body->name ? body->name : "_anon", clash->name, clash->line,
            asm_attr->line);
        return 1;
    }
    const char *raw = (asm_attr->value.kind == KFLV_STR ||
                       asm_attr->value.kind == KFLV_IDENT)
                      ? asm_attr->value.u.s : NULL;
    char path[KFLC_ASM_PATH_MAX];
    if (kflc_assembly_unquote(raw, path, sizeof path) || !*path) {
        kflc_diag_errorf(diag, body->line,
            "astro_body `%s`: `assembly=` takes a path",
            body->name ? body->name : "_anon");
        return 1;
    }
    KflcAssembly *a = kflc_assembly_load(path, src_path, asm_attr->line,
                                         arena, diag);
    if (!a) return 1;
    if (out) *out = a;
    return 0;
}

int kflc_assembly_unquote(const char *raw, char *out, size_t out_sz)
{
    if (!raw || !*raw || !out || out_sz == 0) return 1;
    size_t n = strlen(raw);
    if (n >= 2 && raw[0] == '"' && raw[n - 1] == '"') {
        if (n - 1 > out_sz) return 1;
        memcpy(out, raw + 1, n - 2);
        out[n - 2] = '\0';
        return 0;
    }
    if (n + 1 > out_sz) return 1;
    memcpy(out, raw, n + 1);
    return 0;
}

void kflc_assembly_digest_hex(const uint8_t d[KFLC_ASM_DIGEST],
                              char out[2 * KFLC_ASM_DIGEST + 1])
{
    k26rl_sha256_hex(d, out);
}
