/* emit_rl_collect.c - the program model built from the tree: observe shapes,
 * sensors, actuators, agents, and the collect pass itself. */
#include "emit_rl_internal.h"

static const char *const RL_OBS_COMP_[RL_OBS_COMPS] = {
    "_dir_x", "_dir_y", "_dir_z", "_range", "_range_rate"
};

static const char *const RL_ATT_COMP_[RL_ATT_COMPS] = {
    "_quat_w", "_quat_x", "_quat_y", "_quat_z",
    "_omega_x", "_omega_y", "_omega_z"
};

static const char *const RL_CON_COMP_[RL_CON_COMPS] = {
    "_hit", "_fraction", "_speed"
};

static const char *const RL_PROP_COMP_[RL_PROP_COMPS] = {
    "_propellant_kg", "_propellant_fraction", "_mass_kg",
    "_delta_v_remaining"
};

static const char *const RL_REF_COMP_[RL_REF_COMPS] = {
    "_r_x", "_r_y", "_r_z", "_v_x", "_v_y", "_v_z",
    "_time_to", "_tolerance"
};

static const char *const RL_REL_COMP_[RL_REL_COMPS] = {
    "_r_x", "_r_y", "_r_z", "_v_x", "_v_y", "_v_z"
};

static const char *const RL_PORT_COMP_[RL_PORT_FULL_COMPS] = {
    "_captured", "_axial", "_lateral", "_pitchyaw", "_roll",
    "_v_axial", "_v_lateral", "_v_pitchyaw", "_v_roll",
    /* The two the `full` mark adds, after the nine, so a marked form
     * publishes the unmarked one's channels at the unmarked one's
     * offsets and a program that adds the mark keeps every index it
     * had. */
    "_v_cg", "_joined"
};

static const char *const RL_DET_COMP_[RL_DET_COMPS] = {
    "_detected", "_snr", "_range", "_dir_x", "_dir_y", "_dir_z",
    "_aspect"
};

static const char *const RL_TRK_COMP_[RL_TRK_COMPS] = {
    "_valid", "_pos_x", "_pos_y", "_pos_z", "_vel_x", "_vel_y",
    "_vel_z", "_range", "_age"
};

static const char *const RL_EFF_LAS_COMP_[RL_EFF_LAS_COMPS] = {
    "_engaged", "_effect", "_dv", "_mass_loss", "_range", "_spot",
    "_encircled", "_fluence", "_transmissivity", "_p_coupled",
    "_ignited"
};

static const char *const RL_EFF_IMP_COMP_[RL_EFF_IMP_COMPS] = {
    "_engaged", "_effect", "_hit", "_closing_speed", "_t_close",
    "_miss", "_fraction", "_cos_angle", "_penetrates",
    "_critical_diameter", "_penetration", "_energy"
};

static const char *const RL_EFF_JAM_COMP_[RL_EFF_JAM_COMPS] = {
    "_engaged", "_effect", "_reached", "_range", "_rcs",
    "_burn_through", "_self_signature", "_counter_range",
    "_counter_detected"
};

static const char *const RL_EFF_DEC_COMP_[RL_EFF_DEC_COMPS] = {
    "_engaged", "_deployed", "_effect", "_reached", "_p_discriminated",
    "_range", "_dv", "_mass_loss"
};

static int rl_observe_eff_kind_(const KflcNode *n);
static int rl_eff_comps_(int kind);
static const char *rl_eff_comp_(int kind, int c);
static const char *rl_observe_through_(const KflcNode *n);
static const char *rl_observe_base_comp_(const KflcNode *n, int c);
static int rl_form_declares_fn_(const KflcNode *form, const char *name);
static int rl_text_has_call_(const char *text, const char *name);
static double rl_term_num_(const KflcNode *t, int i, int *have);
static int rl_collect_sensor_(RlModel *m, const KflcNode *n, KflcDiag *diag);
static int rl_collect_actuators_(RlModel *m, KflcDiag *diag);
static int rl_add_action_(RlModel *m, const KflcNode *s, int owner,
                          KflcDiag *diag);
static int rl_add_observe_(RlModel *m, const KflcNode *s, int owner,
                           KflcDiag *diag);
/* ---- Agents ---------------------------------------------------------- */

static int rl_collect_agent_(RlModel *m, const KflcNode *s, KflcDiag *diag);
static int rl_finish_agents_(RlModel *m, KflcDiag *diag);

/* The published observer-mode value of an as-bound observe. The
 * grammar's default when no mode= attribute is given is the runtime's
 * default, astrometric, which is what the observation path selects. */

uint16_t rl_observe_mode(const KflcNode *n)
{
    if (!n) return 1;
    /* An attitude observe reports a body's own orientation and rate.
     * There is no observer, no light time and no aberration, so the
     * only true answer among the published modes is the geometric
     * one; the default astrometric value would claim a correction
     * that is not applied. */
    for (const KflcAttr *k = n->attrs; k; k = k->next) {
        if (!k->name) continue;
        if (strcmp(k->name, "attitude") == 0) return 0;
        /* A contact observe has no observer either, and applies no
         * correction of any kind: it reports what a transition did.
         * Geometric is true of it in the one sense the tag was minted
         * for, and astrometric would assert a correction that is not
         * made. */
        if (strcmp(k->name, "contact") == 0) return 0;
        /* A relative observe has an observer, the chief, but applies
         * no light-time correction and no aberration to the state it
         * publishes: it resolves a state the integrator already
         * produced onto a set of axes. Geometric is exactly what that
         * is; astrometric would assert a correction that is not
         * made. */
        if (strcmp(k->name, "relative") == 0) return 0;
        /* A port observe reports the geometry of one docking interface
         * against another, both of which the integrator already
         * produced. No correction of any kind is applied to it. */
        if (strcmp(k->name, "port") == 0) return 0;
        /* An effector observe reports what an engagement did, from the
         * two craft's current state and the declared payload. There is
         * no observer and no light time, so geometric is the only true
         * answer among the published modes. */
        if (strcmp(k->name, "effect") == 0) return 0;
        /* A propulsion observe reports a craft's own remaining
         * propellant and mass. There is no observer and nothing is
         * corrected, so geometric is the only true answer here too. */
        if (strcmp(k->name, "propulsion") == 0) return 0;
        /* A reference observe reports where a craft's plan says it
         * should be, against where the integrator has put it. Nothing
         * is observed across a distance and nothing is corrected, so
         * geometric is the only true answer here as well. */
        if (strcmp(k->name, "reference") == 0) return 0;
    }
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "mode") == 0 &&
            a->value.kind == KFLV_IDENT && a->value.u.s) {
            if (strcmp(a->value.u.s, "geometric") == 0) return 0;
            if (strcmp(a->value.u.s, "astrometric") == 0) return 1;
            if (strcmp(a->value.u.s, "apparent") == 0) return 2;
            if (strcmp(a->value.u.s, "topocentric") == 0) return 3;
        }
    }
    return 1;
}

int rl_is_state_key(const char *k)
{
    return kflc_body_state_key_index(k) >= 0;
}

RlObserveForm rl_observe_form(const KflcNode *n)
{
    if (!n) return RL_OBS_LOS;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (!a->name) continue;
        if (strcmp(a->name, "detect") == 0)   return RL_OBS_DET;
        if (strcmp(a->name, "track") == 0)    return RL_OBS_TRK;
        if (strcmp(a->name, "effect") == 0)   return RL_OBS_EFF;
        if (strcmp(a->name, "attitude") == 0) return RL_OBS_ATT;
        if (strcmp(a->name, "propulsion") == 0) return RL_OBS_PROP;
        if (strcmp(a->name, "reference") == 0) return RL_OBS_REF;
        if (strcmp(a->name, "contact") == 0)  return RL_OBS_CON;
        if (strcmp(a->name, "relative") == 0) return RL_OBS_REL;
        if (strcmp(a->name, "port") == 0)     return RL_OBS_PORT;
    }
    return RL_OBS_LOS;
}

/* The payload a defense observe names, which its marker attribute
 * carries for the same reason the port marker carries the port's own
 * name. */

const char *rl_observe_payload(const KflcNode *n)
{
    if (!n) return NULL;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (!a->name || a->value.kind != KFLV_IDENT) continue;
        if (strcmp(a->name, "detect") == 0 ||
            strcmp(a->name, "track")  == 0 ||
            strcmp(a->name, "effect") == 0) {
            return a->value.u.s;
        }
    }
    return NULL;
}

/* Which effector kind an effect observe publishes the channel set of.
 * The kind is resolved onto the statement during collection, before the
 * agent slices are computed, because the width of this form is its
 * payload's kind's and the slices are a function of the widths. An
 * unstamped statement is one whose payload did not resolve, which is
 * an error already reported; the laser kind is returned so the widths
 * stay consistent while the diagnostics are collected. */
static int rl_observe_eff_kind_(const KflcNode *n)
{
    if (!n) return RL_PAY_LASER;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (!a->name || strcmp(a->name, "effect_kind") != 0) continue;
        if (a->value.kind != KFLV_IDENT || !a->value.u.s) break;
        int k = rl_pay_kind_from_name(a->value.u.s);
        if (k >= 0) return k;
        break;
    }
    return RL_PAY_LASER;
}

/* The component table and its width, per effector kind. One switch, so
 * a kind cannot have a name table without a width or the other way
 * about. */

static int rl_eff_comps_(int kind)
{
    switch (kind) {
    case RL_PAY_IMPACTOR: return RL_EFF_IMP_COMPS;
    case RL_PAY_DECOY:    return RL_EFF_DEC_COMPS;
    case RL_PAY_JAMMER:   return RL_EFF_JAM_COMPS;
    default:              return RL_EFF_LAS_COMPS;
    }
}

static const char *rl_eff_comp_(int kind, int c)
{
    switch (kind) {
    case RL_PAY_IMPACTOR: return RL_EFF_IMP_COMP_[c];
    case RL_PAY_DECOY:    return RL_EFF_DEC_COMP_[c];
    case RL_PAY_JAMMER:   return RL_EFF_JAM_COMP_[c];
    default:              return RL_EFF_LAS_COMP_[c];
    }
}

int rl_observe_is_attitude(const KflcNode *n)
{
    return rl_observe_form(n) == RL_OBS_ATT;
}

/* The two switches below name every form and carry no default, so a
 * form added to the enum without a width and a name table is a
 * compiler warning rather than five silent line-of-sight channels. The
 * return after each switch is what the language requires, not a
 * fallback the code relies on. */

/* Whether the observe declared `through <sensor>`, and whether it
 * asked for the uncorrupted values beside the measured ones. */

static const char *rl_observe_through_(const KflcNode *n)
{
    if (!n) return NULL;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "through") == 0 &&
            a->value.kind == KFLV_IDENT) {
            return a->value.u.s;
        }
    }
    return NULL;
}

int rl_observe_has_truth(const KflcNode *n)
{
    if (!n) return 0;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "truth") == 0) return 1;
    }
    return 0;
}

/* Whether a port observe carries the `full` mark. */

int rl_observe_is_full(const KflcNode *n)
{
    if (!n) return 0;
    for (const KflcAttr *a = n->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "full") == 0) return 1;
    }
    return 0;
}

/* The components one form publishes, before any pairing. */

int rl_observe_base_width(const KflcNode *n)
{
    switch (rl_observe_form(n)) {
    case RL_OBS_ATT: return RL_ATT_COMPS;
    case RL_OBS_PROP: return RL_PROP_COMPS;
    case RL_OBS_REF: return RL_REF_COMPS;
    case RL_OBS_CON: return RL_CON_COMPS;
    case RL_OBS_REL: return RL_REL_COMPS;
    case RL_OBS_PORT:
        return rl_observe_is_full(n) ? RL_PORT_FULL_COMPS : RL_PORT_COMPS;
    case RL_OBS_DET: return RL_DET_COMPS;
    case RL_OBS_TRK: return RL_TRK_COMPS;
    case RL_OBS_EFF: return rl_eff_comps_(rl_observe_eff_kind_(n));
    case RL_OBS_LOS: return RL_OBS_COMPS;
    }
    return RL_OBS_COMPS;
}

/* What the observe contributes to the observation vector. `with truth`
 * publishes the uncorrupted components as a paired set beside the
 * measured ones, so the observe is twice as wide: the measured half
 * first, then the truth half, each in component order. Every offset,
 * the total, and the base of every write follow from this one
 * function, so the pairing cannot displace a neighbouring observe. */

int rl_observe_width(const KflcNode *n)
{
    int w = rl_observe_base_width(n);
    return rl_observe_has_truth(n) ? 2 * w : w;
}

static const char *rl_observe_base_comp_(const KflcNode *n, int c)
{
    switch (rl_observe_form(n)) {
    case RL_OBS_ATT: return RL_ATT_COMP_[c];
    case RL_OBS_PROP: return RL_PROP_COMP_[c];
    case RL_OBS_REF: return RL_REF_COMP_[c];
    case RL_OBS_CON: return RL_CON_COMP_[c];
    case RL_OBS_REL: return RL_REL_COMP_[c];
    case RL_OBS_PORT: return RL_PORT_COMP_[c];
    case RL_OBS_DET: return RL_DET_COMP_[c];
    case RL_OBS_TRK: return RL_TRK_COMP_[c];
    case RL_OBS_EFF: return rl_eff_comp_(rl_observe_eff_kind_(n), c);
    case RL_OBS_LOS: return RL_OBS_COMP_[c];
    }
    return RL_OBS_COMP_[c];
}

/* The suffix of channel `c` of this observe, written into `buf`. A
 * channel in the truth half carries `_truth` before its component, so
 * a reader sees `<name>_truth_range` beside `<name>_range`. The buffer
 * is the caller's because two suffixes are live at once wherever a
 * pair is emitted. */

const char *rl_observe_comp(const KflcNode *n, int c,
                                    char *buf, size_t cap)
{
    int w = rl_observe_base_width(n);
    if (c < w) return rl_observe_base_comp_(n, c);
    snprintf(buf, cap, "_truth%s", rl_observe_base_comp_(n, c - w));
    return buf;
}

/* The first channel index of observe `i`, and the total width. Both
 * walk the declarations in source order, which is the order channels
 * are allocated in. */

int rl_obs_offset(const KflcNode *const *obs, int i)
{
    int off = 0;
    for (int k = 0; k < i; k++) off += rl_observe_width(obs[k]);
    return off;
}

int rl_obs_total(const KflcNode *const *obs, int n)
{
    return rl_obs_offset(obs, n);
}

const KflcAttr *rl_attr(const KflcNode *n, const char *key)
{
    for (const KflcAttr *a = n ? n->attrs : NULL; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0) return a;
    }
    return NULL;
}

const char *rl_observe_as(const KflcNode *n)
{
    const KflcAttr *a = rl_attr(n, "as");
    if (a && a->value.kind == KFLV_IDENT && a->value.u.s) return a->value.u.s;
    return NULL;
}

/* A user-declared fn shadows the distribution reading of its name,
 * mirroring the checker's rule. */

static int rl_form_declares_fn_(const KflcNode *form, const char *name)
{
    if (!form || !name) return 0;
    for (const KflcNode *c = form->children; c; c = c->next) {
        if (c->kind == KFLN_FN && c->name && strcmp(c->name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

/* Word-boundary scan of attribute value text for a call to `name`,
 * matching the checker's classification so emission and check agree
 * on what counts as a distribution position. */

static int rl_text_has_call_(const char *text, const char *name)
{
    if (!text || !name) return 0;
    size_t nl = strlen(name);
    for (const char *p = text; *p; p++) {
        if (strncmp(p, name, nl) != 0) continue;
        if (p != text) {
            char b = p[-1];
            if (b == '_' || isalnum((unsigned char)b)) continue;
        }
        const char *q = p + nl;
        if (*q == '_' || isalnum((unsigned char)*q)) continue;
        while (*q == ' ' || *q == '\t') q++;
        if (*q == '(') return 1;
    }
    return 0;
}

/* Parse an astro_body attribute's verbatim value text and return the
 * expression when the whole value is a two-argument uniform/normal
 * call whose name is not shadowed; NULL otherwise. A value that
 * merely contains a distribution call somewhere inside a larger
 * expression is reported: a drawn parameter is the whole value or
 * nothing. */

KflcExpr *rl_attr_dist(const KflcAttr *a, const KflcNode *form,
                               KflcArena *arena, KflcDiag *diag,
                               int *out_error)
{
    if (a->value.kind != KFLV_IDENT || !a->value.u.s) return NULL;
    const char *text = a->value.u.s;
    int has_uniform = rl_text_has_call_(text, "uniform") &&
                      !rl_form_declares_fn_(form, "uniform");
    int has_normal  = rl_text_has_call_(text, "normal") &&
                      !rl_form_declares_fn_(form, "normal");
    if (!has_uniform && !has_normal) return NULL;

    KflcDiag scratch;
    kflc_diag_init(&scratch, "", stderr);
    /* Parse errors fall through to verbatim splicing; the downstream
     * C++ compile reports them as it always has for attribute text. */
    FILE *devnull = fopen("/dev/null", "w");
    if (devnull) scratch.errstream = devnull;
    KflcExpr *e = kflc_parse_expr(text, arena, &scratch, a->line);
    if (devnull) fclose(devnull);

    int is_dist = (e && scratch.errors == 0 &&
                   e->kind == KFLE_CALL && e->u.call.name &&
                   (strcmp(e->u.call.name, "uniform") == 0 ||
                    strcmp(e->u.call.name, "normal") == 0) &&
                   e->u.call.n_args == 2 &&
                   !rl_form_declares_fn_(form, e->u.call.name));
    if (is_dist) return e;

    /* A distribution call inside a larger expression is not a
     * drawable parameter. */
    {
        kflc_diag_errorf(diag, a->line,
            "astro_body %s: a distribution must be the whole attribute "
            "value (`%s=uniform(lo, hi)` or `%s=normal(mu, sigma)`), not "
            "part of a larger expression", a->name, a->name, a->name);
        *out_error = 1;
    }
    return NULL;
}

int rl_is_scalar_type(KflcType t)
{
    return t == KFLT_DOUBLE || t == KFLT_INT || t == KFLT_BOOL;
}

/* True when `name` is already claimed by the RL expression scope: an
 * action channel, an observation channel component, or the episode
 * step counter. A world binding with such a name is shadowed there and
 * never captured. */

int rl_scope_name_taken(const RlModel *m, const char *name)
{
    if (!name) return 0;
    if (strcmp(name, "episode.steps") == 0) return 1;
    for (int i = 0; i < m->n_actions; i++) {
        const char *an = m->actions[i]->name;
        if (an && strcmp(an, name) == 0) return 1;
    }
    for (int i = 0; i < m->n_observes; i++) {
        const char *base = rl_observe_as(m->observes[i]);
        if (!base) continue;
        size_t bl = strlen(base);
        if (strncmp(name, base, bl) != 0) continue;
        for (int c = 0; c < rl_observe_width(m->observes[i]); c++) {
            char cb[RL_COMP_MAX];
            if (strcmp(name + bl,
                       rl_observe_comp(m->observes[i], c, cb, sizeof cb))
                == 0) {
                return 1;
            }
        }
    }
    return 0;
}

/* Find a let/const named `name` anywhere in the world subtree.
 * out_top reports whether the hit is a direct child of the world
 * (the prefix's own scope) rather than nested inside a block. */

const KflcNode *rl_find_world_binding(const KflcNode *stmts,
                                              const char *name, int depth,
                                              int *out_top)
{
    for (const KflcNode *s = stmts; s; s = s->next) {
        if ((s->kind == KFLN_STMT_LET || s->kind == KFLN_STMT_CONST) &&
            s->name && strcmp(s->name, name) == 0) {
            *out_top = (depth == 0);
            return s;
        }
        const KflcNode *hit =
            rl_find_world_binding(s->children, name, depth + 1, out_top);
        if (hit) return hit;
        hit = rl_find_world_binding(s->else_children, name, depth + 1,
                                     out_top);
        if (hit) return hit;
    }
    return NULL;
}

/* Collect the program model from the form. Returns 0 on success;
 * nonzero after reporting diagnostics. */

/* Whether a model body binds a vehicle assembly. Attitude state is
 * only advanced for a body that does, since the advance needs the
 * inertia tensor the assembly derives, so the three places that can
 * set attitude all ask this before accepting it. */

int rl_body_has_assembly(const RlModel *m, int bi)
{
    if (bi < 0 || bi >= m->n_bodies) return 0;
    for (const KflcAttr *a = m->bodies[bi].body->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, "assembly") == 0) return 1;
    }
    return 0;
}

/* Copy each assembly-bearing body's actuators into the model, in
 * vehicle order and, within a vehicle, in the order the assembly
 * declares them. That order is the command surface's order and the
 * emitted tables', so a program's actuators are named by the asset
 * and indexed the same way everywhere. */

/* The attribute of that name on a body, or NULL. */

const KflcAttr *rl_body_attr(const KflcNode *body, const char *name)
{
    if (!body || !name) return NULL;
    for (const KflcAttr *a = body->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, name) == 0) return a;
    }
    return NULL;
}

/* Read one `sensor` block into the model. Each child is one term; the
 * operands were parsed as plain numbers and are checked here against
 * the term's own domain, because a negative standard deviation or a
 * probability above one is a declaration nobody can mean. */

static double rl_term_num_(const KflcNode *t, int i, int *have)
{
    char key[8];
    snprintf(key, sizeof key, "n%d", i);
    for (const KflcAttr *a = t->attrs; a; a = a->next) {
        if (a->name && strcmp(a->name, key) == 0) {
            if (have) *have = 1;
            return a->value.u.f;
        }
    }
    if (have) *have = 0;
    return 0.0;
}

static int rl_collect_sensor_(RlModel *m, const KflcNode *n, KflcDiag *diag)
{
    if (m->n_sensors >= RL_MAX_SENSORS) {
        kflc_diag_errorf(diag, n->line,
            "more than %d sensors in this program", RL_MAX_SENSORS);
        return 1;
    }
    for (int i = 0; i < m->n_sensors; i++) {
        if (m->sensors[i].name && n->name &&
            strcmp(m->sensors[i].name, n->name) == 0) {
            kflc_diag_errorf(diag, n->line,
                "sensor `%s` is declared twice", n->name);
            return 1;
        }
    }
    RlSensor *sn = &m->sensors[m->n_sensors];
    memset(sn, 0, sizeof *sn);
    sn->node = n;
    sn->name = n->name;

    for (const KflcNode *c = n->children; c; c = c->next) {
        if (c->kind != KFLN_STMT_SENSOR_TERM) continue;
        if (sn->n_terms >= RL_MAX_TERMS) {
            kflc_diag_errorf(diag, c->line,
                "sensor `%s`: more than %d terms", sn->name, RL_MAX_TERMS);
            return 1;
        }
        RlSenseTerm *t = &sn->terms[sn->n_terms];
        memset(t, 0, sizeof *t);
        t->line = c->line;
        const char *kw = c->name ? c->name : "";
        int h0 = 0, h1 = 0, h2 = 0;
        double v0 = rl_term_num_(c, 0, &h0);
        double v1 = rl_term_num_(c, 1, &h1);
        double v2 = rl_term_num_(c, 2, &h2);

        if (strcmp(kw, "noise") == 0) {
            const char *dist = NULL;
            for (const KflcAttr *a = c->attrs; a; a = a->next) {
                if (a->name && strcmp(a->name, "dist") == 0 &&
                    a->value.kind == KFLV_IDENT) dist = a->value.u.s;
            }
            if (!dist || strcmp(dist, "normal") != 0) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `noise` takes `normal`, the only "
                    "distribution this layer offers", sn->name);
                return 1;
            }
            if (!h0 || !h1) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `noise normal` takes a mean and a "
                    "standard deviation", sn->name);
                return 1;
            }
            if (v0 != 0.0) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `noise normal` takes a mean of zero; "
                    "a constant offset is a bias, not noise", sn->name);
                return 1;
            }
            if (!(v1 >= 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: a standard deviation cannot be "
                    "negative", sn->name);
                return 1;
            }
            t->kind = K26SENSE_ADDITIVE;
            t->p0 = v1;
            t->draws_step = 1;
        } else if (strcmp(kw, "scale") == 0) {
            if (!h0 || !(v0 >= 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `scale` takes one relative standard "
                    "deviation, not negative", sn->name);
                return 1;
            }
            t->kind = K26SENSE_SCALE;
            t->p0 = v0;
            t->draws_step = 1;
        } else if (strcmp(kw, "bias_walk") == 0) {
            if (!h0 || !h1 || !h2) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `bias_walk` takes a turn-on standard "
                    "deviation, a correlation time in seconds, and an "
                    "in-run standard deviation", sn->name);
                return 1;
            }
            if (!(v0 >= 0.0) || !(v1 > 0.0) || !(v2 >= 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `bias_walk` needs a positive "
                    "correlation time and standard deviations that are "
                    "not negative", sn->name);
                return 1;
            }
            t->kind = K26SENSE_BIAS_WALK;
            t->p0 = v0; t->p1 = v1; t->p2 = v2;
            t->draws_step = 1;
            t->draws_ep = 1;
        } else if (strcmp(kw, "latency") == 0) {
            if (!h0 || v0 < 0.0 || v0 != (double)(int)v0 || v0 > 1024.0) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `latency` takes a whole number of "
                    "control periods, from 0 to 1024", sn->name);
                return 1;
            }
            t->kind = K26SENSE_LATENCY;
            t->p0 = v0;
            if ((int)v0 > sn->depth) sn->depth = (int)v0;
        } else if (strcmp(kw, "quantise") == 0) {
            if (!h0 || !(v0 > 0.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `quantise` takes a positive step",
                    sn->name);
                return 1;
            }
            if (h1 != h2) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `quantise` takes a step alone, or a "
                    "step with both ends of its range", sn->name);
                return 1;
            }
            if (h1 && !(v1 < v2)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `quantise` needs a range whose low "
                    "end is below its high end", sn->name);
                return 1;
            }
            /* A range and a step together decide whether the
             * quantiser's grid position stays inside a signed 64-bit
             * integer. Beyond that the conversion is undefined, and a
             * program must not be able to reach undefined behaviour by
             * declaring a fine step, so the pair is checked here and
             * the library saturates if anything ever slips past.
             *
             * The implied range is the narrower of a working ceiling
             * and what the step itself can reach: a step of `lsb`
             * spans at most 2^63 multiples either side of zero, so a
             * step of 1e-14 reaches about 9.2e4 and not 1e15. Taking
             * 1e15 unconditionally is what let a true value of seven
             * million be published as minus ninety-two thousand. */
            double span = v0 * K26SENSE_I64_SPAN;
            double cap  = span < 1.0e15 ? span : 1.0e15;
            if (h1) {
                double reach = (v2 > -v1 ? v2 : -v1) / v0;
                if (!(reach < K26SENSE_I64_SPAN)) {
                    kflc_diag_errorf(diag, c->line,
                        "sensor `%s`: `quantise` with a step of %g over "
                        "a range reaching %g needs %g grid positions, "
                        "and a 64-bit integer holds %g; widen the step "
                        "or narrow the range", sn->name, v0,
                        (v2 > -v1 ? v2 : -v1), reach,
                        K26SENSE_I64_SPAN);
                    return 1;
                }
            }
            t->kind = K26SENSE_QUANTISE;
            t->p0 = v0;
            t->p1 = h1 ? v1 : -cap;
            t->p2 = h1 ? v2 :  cap;
        } else if (strcmp(kw, "dropout") == 0) {
            if (!h0 || !(v0 >= 0.0) || !(v0 < 1.0)) {
                kflc_diag_errorf(diag, c->line,
                    "sensor `%s`: `dropout` takes a probability from 0 "
                    "up to but not including 1", sn->name);
                return 1;
            }
            t->kind = K26SENSE_DROPOUT;
            t->p0 = v0;
            t->draws_step = 1;
        } else {
            kflc_diag_errorf(diag, c->line,
                "sensor `%s`: unknown model term `%s`", sn->name, kw);
            return 1;
        }
        sn->n_terms++;
    }

    if (sn->n_terms == 0) {
        kflc_diag_errorf(diag, n->line,
            "sensor `%s` declares no model terms, so it would leave "
            "every channel it touches unchanged", sn->name);
        return 1;
    }
    int n_bias = 0, n_delay = 0;
    for (int i = 0; i < sn->n_terms; i++) {
        if (sn->terms[i].kind == K26SENSE_BIAS_WALK) n_bias++;
        if (sn->terms[i].kind == K26SENSE_LATENCY)   n_delay++;
    }
    if (n_bias > 1 || n_delay > 1) {
        kflc_diag_errorf(diag, n->line,
            "sensor `%s`: one bias walk and one latency per sensor, "
            "since one carries one state of each", sn->name);
        return 1;
    }
    m->n_sensors++;
    return 0;
}

static int rl_collect_actuators_(RlModel *m, KflcDiag *diag)
{
    int veh = 0;
    for (int i = 0; i < m->n_bodies; i++) {
        if (!rl_body_has_assembly(m, i)) continue;
        KflcArena *ar = kflc_arena_create();
        if (!ar) return 1;
        KflcAssembly *a = NULL;
        if (kflc_assembly_for_body(m->bodies[i].body, diag->path, ar, diag,
                                   &a) || !a) {
            kflc_arena_release(ar);
            return 1;
        }
        if (veh >= RL_MAX_VEH) {
            kflc_diag_errorf(diag, m->bodies[i].body->line,
                "more than %d bodies carry an assembly", RL_MAX_VEH);
            kflc_arena_release(ar);
            return 1;
        }
        for (int k = 0; k < 3; k++) m->veh_com[veh][k] = a->com[k];
        m->veh_bound[veh] = a->bound_radius;

        /* The mass properties at any fill, and the tank the fill
         * belongs to. An assembly with no tank leaves the capacity at
         * zero and its aggregates covering the whole vehicle, so a
         * consumer indexes the same tables either way. */
        m->veh_has_prop[veh]  = a->propellant >= 0;
        m->veh_prop_cap[veh]  = a->prop_capacity;
        m->veh_struct_mass[veh] = a->struct_mass;
        for (int k = 0; k < 3; k++) {
            m->veh_struct_moment[veh][k]  = a->struct_moment[k];
            m->veh_prop_centroid[veh][k]  = a->prop_centroid[k];
        }
        for (int k = 0; k < 6; k++) {
            m->veh_struct_inertia[veh][k] = a->struct_inertia[k];
            m->veh_prop_inertia[veh][k]   = a->prop_inertia[k];
        }
        if (a->propellant >= 0) m->n_prop_veh++;

        /* The collider set, taken in declaration order, which is the
         * order the narrowphase tests them in and therefore the order
         * the selection rule's tie-break is defined over. The reader
         * has already baked the component placements in, so these are
         * body-frame and comparable across components. */
        for (int k = 0; k < a->n_colliders; k++) {
            const KflcAsmCollider *cl = &a->colliders[k];
            if (m->n_colliders >= RL_MAX_COLL) {
                kflc_diag_errorf(diag, cl->line,
                    "more than %d colliders in this program", RL_MAX_COLL);
                kflc_arena_release(ar);
                return 1;
            }
            RlCollider *rc = &m->colliders[m->n_colliders++];
            memset(rc, 0, sizeof *rc);
            rc->veh = veh; rc->body = i;
            for (int q = 0; q < 3; q++) {
                rc->centre[q] = cl->centre[q];
                for (int w = 0; w < 3; w++) rc->axis[q][w] = cl->rot[w][q];
            }
            if (cl->kind == KFLC_SHAPE_SPHERE) {
                rc->kind = 1;
                rc->half[0] = cl->r;
            } else if (cl->kind == KFLC_SHAPE_CAPSULE) {
                /* The kernels take a capsule as a centre, a direction,
                 * a radius, and half the segment's length, so the
                 * reader's two endpoints are converted here once
                 * rather than on every step. */
                rc->kind = 2;
                double d[3] = { cl->b[0] - cl->a[0], cl->b[1] - cl->a[1],
                                cl->b[2] - cl->a[2] };
                double len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                rc->half[0] = cl->r;
                rc->half[2] = 0.5 * len;
                if (len > 0.0) {
                    for (int q = 0; q < 3; q++) rc->axis[2][q] = d[q] / len;
                    /* Any pair completing the frame will do: a capsule
                     * is symmetric about its axis, so only the third
                     * direction carries meaning. */
                    double up[3] = { 0.0, 0.0, 1.0 };
                    if (rc->axis[2][2] > 0.9 || rc->axis[2][2] < -0.9) {
                        up[0] = 1.0; up[2] = 0.0;
                    }
                    double ax[3] = {
                        up[1] * rc->axis[2][2] - up[2] * rc->axis[2][1],
                        up[2] * rc->axis[2][0] - up[0] * rc->axis[2][2],
                        up[0] * rc->axis[2][1] - up[1] * rc->axis[2][0] };
                    double al = sqrt(ax[0] * ax[0] + ax[1] * ax[1] +
                                     ax[2] * ax[2]);
                    for (int q = 0; q < 3; q++) rc->axis[0][q] = ax[q] / al;
                    rc->axis[1][0] = rc->axis[2][1] * rc->axis[0][2]
                                   - rc->axis[2][2] * rc->axis[0][1];
                    rc->axis[1][1] = rc->axis[2][2] * rc->axis[0][0]
                                   - rc->axis[2][0] * rc->axis[0][2];
                    rc->axis[1][2] = rc->axis[2][0] * rc->axis[0][1]
                                   - rc->axis[2][1] * rc->axis[0][0];
                }
            } else {
                rc->kind = 3;
                for (int q = 0; q < 3; q++) rc->half[q] = cl->a[q];
            }
        }

        for (int f = 0; f < a->n_features; f++) {
            const KflcAsmFeature *ft = &a->features[f];
            if (ft->kind == KFLC_FEAT_WHEEL) {
                if (m->n_wheels >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d wheels in this program", RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlWheel *w = &m->wheels[m->n_wheels++];
                w->veh = veh; w->body = i;
                snprintf(w->name, sizeof w->name, "%s", ft->name);
                for (int k = 0; k < 3; k++) w->axis[k] = ft->axis[k];
                w->spin_inertia = ft->spin_inertia;
                w->max_momentum = ft->max_momentum;
                w->max_torque   = ft->max_torque;
                w->viscous      = ft->viscous;
                w->coulomb      = ft->coulomb;
                w->dead_rate    = ft->dead_rate;
            } else if (ft->kind == KFLC_FEAT_TORQUER) {
                if (m->n_torquers >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d magnetorquers in this program",
                        RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlTorquer *q = &m->torquers[m->n_torquers++];
                q->veh = veh; q->body = i;
                snprintf(q->name, sizeof q->name, "%s", ft->name);
                for (int k = 0; k < 3; k++) q->axis[k] = ft->axis[k];
                q->max_dipole = ft->max_dipole;
            } else if (ft->kind == KFLC_FEAT_THRUSTER) {
                if (m->n_thrusters >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d thrusters in this program",
                        RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlThruster *t = &m->thrusters[m->n_thrusters++];
                t->veh = veh; t->body = i;
                snprintf(t->name, sizeof t->name, "%s", ft->name);
                for (int k = 0; k < 3; k++) {
                    t->at[k]  = ft->at[k];
                    t->dir[k] = ft->dir[k];
                }
                t->max_thrust = ft->thrust;
                t->isp_s      = ft->isp_s;
            } else if (ft->kind == KFLC_FEAT_PORT && ft->collider >= 0) {
                /* Only a port that named a capture envelope reaches
                 * here: the reader gives one a mating plane collider
                 * and leaves the index at -1 otherwise, so a port
                 * declared without an envelope is geometry the
                 * program can describe and nothing this emitter has a
                 * test to apply to. */
                const KflcCaptureEnvelope *env =
                    kflc_capture_envelope(ft->capture);
                if (!env) {
                    kflc_diag_errorf(diag, ft->line,
                        "port `%s` names capture envelope `%s`, which is "
                        "not defined", ft->name, ft->capture);
                    kflc_arena_release(ar);
                    return 1;
                }
                if (m->n_ports >= RL_MAX_ACT) {
                    kflc_diag_errorf(diag, ft->line,
                        "more than %d docking ports in this program",
                        RL_MAX_ACT);
                    kflc_arena_release(ar);
                    return 1;
                }
                RlPort *pt = &m->ports[m->n_ports++];
                memset(pt, 0, sizeof *pt);
                pt->veh = veh; pt->body = i;
                for (int k = 0; k < 3; k++) pt->com[k] = a->com[k];
                /* The pass reports a shape index within the body's
                 * own slice, which is the assembly's collider order,
                 * so the reader's index is what a contact is
                 * compared against. */
                pt->coll = ft->collider;
                snprintf(pt->name, sizeof pt->name, "%s", ft->name);
                snprintf(pt->env_name, sizeof pt->env_name, "%s",
                         ft->capture);
                const KflcAsmCollider *plate = &a->colliders[ft->collider];
                for (int k = 0; k < 3; k++) {
                    pt->at[k] = ft->at[k];
                    /* The reader built the plate on the port's own
                     * frame, so the basis is read back from it rather
                     * than orthonormalised a second time here, which
                     * would be a second place for it to be wrong. */
                    for (int w = 0; w < 3; w++) {
                        pt->basis[k][w] = plate->rot[w][k];
                    }
                }
                /* The one conversion from the envelope's printed
                 * units to the units the artifact carries. */
                pt->axial_rate_min = env->axial_rate_min;
                pt->axial_rate_max = env->axial_rate_max;
                pt->lateral_rate   = env->lateral_rate;
                pt->pitchyaw_rate  = kflc_capture_deg_to_rad(env->pitchyaw_rate_deg);
                pt->roll_rate      = kflc_capture_deg_to_rad(env->roll_rate_deg);
                pt->lateral        = env->lateral;
                pt->pitchyaw       = kflc_capture_deg_to_rad(env->pitchyaw_deg);
                pt->roll           = kflc_capture_deg_to_rad(env->roll_deg);
                pt->diameter       = kflc_capture_mm_to_m(env->mating_diameter_mm);
            }
        }

        /* The specific impulse the vehicle's remaining velocity
         * change is published at. Thrusters may differ in it, so one
         * figure has to be chosen and the choice is the thrust
         * weighted mean: an engine that can spend the tank fastest
         * weighs most in what the tank is worth. The weights are the
         * declared maxima, so the figure is a constant of the
         * assembly and not of the throttle. */
        {
            double wsum = 0.0, isum = 0.0;
            for (int q = 0; q < m->n_thrusters; q++) {
                if (m->thrusters[q].veh != veh) continue;
                wsum += m->thrusters[q].max_thrust;
                isum += m->thrusters[q].max_thrust * m->thrusters[q].isp_s;
            }
            m->veh_isp[veh] = wsum > 0.0 ? isum / wsum : 0.0;
        }

        /* A magnetorquer works against the local magnetic field, and
         * the field model is defined at a geodetic position on a
         * rotating body. Reaching one needs the parent's rotation
         * model, and the only key a program can supply for it is the
         * parent's NAIF id: the model table is keyed by that id and
         * by a name of its own form, which a grammar identifier can
         * never be. A declaration that cannot reach a field is
         * refused here rather than run: the alternative is a
         * magnetorquer that compiles, accepts commands, and produces
         * no torque, which is the kind of wrong answer that costs a
         * training run rather than a compile. */
        {
            int has_torquer = 0;
            for (int q = 0; q < m->n_torquers; q++) {
                if (m->torquers[q].body == i) has_torquer = 1;
            }
            if (has_torquer) {
                const KflcNode *b = m->bodies[i].body;
                const KflcAttr *pa = rl_body_attr(b, "parent");
                const char *pn = (pa && pa->value.kind == KFLV_IDENT)
                               ? pa->value.u.s : NULL;
                int pi = -1;
                for (int j = 0; pn && j < m->n_bodies; j++) {
                    const char *nm = m->bodies[j].body->name;
                    if (nm && strcmp(nm, pn) == 0) pi = j;
                }
                if (!pn) {
                    kflc_diag_errorf(diag, b->line,
                        "`%s` carries a magnetorquer but declares no "
                        "`parent=`, so there is no rotating body whose "
                        "field it could work against", b->name);
                    kflc_arena_release(ar);
                    return 1;
                }
                if (pi < 0) {
                    kflc_diag_errorf(diag, b->line,
                        "`%s` carries a magnetorquer and names `%s` as its "
                        "parent, but no astro_body of that name is declared "
                        "in this world", b->name, pn);
                    kflc_arena_release(ar);
                    return 1;
                }
                if (!rl_body_attr(m->bodies[pi].body, "ephem_naif_id")) {
                    kflc_diag_errorf(diag, m->bodies[pi].body->line,
                        "`%s` carries a magnetorquer, and its parent `%s` "
                        "declares no `ephem_naif_id=`; that is what names "
                        "the parent's rotation model, without which the "
                        "field has no frame and the magnetorquer would "
                        "produce no torque (Earth is 399)",
                        b->name, pn);
                    kflc_arena_release(ar);
                    return 1;
                }
            }
        }
        veh++;
        kflc_arena_release(ar);
    }
    m->n_veh = veh;
    return 0;
}

/* Record one action or one `as`-bound observe against the agent that
 * owns it, -1 meaning it was written at world level. Declarations are
 * appended in the order they are read, which is source order, so each
 * agent's range in these arrays is contiguous. */

static int rl_add_action_(RlModel *m, const KflcNode *s, int owner,
                          KflcDiag *diag)
{
    if (m->n_actions == RL_MAX_ACTIONS) {
        kflc_diag_errorf(diag, s->line,
            "too many action declarations (limit %d)", RL_MAX_ACTIONS);
        return 1;
    }
    m->act_owner[m->n_actions] = owner;
    m->actions[m->n_actions++] = s;
    return 0;
}

static int rl_add_observe_(RlModel *m, const KflcNode *s, int owner,
                           KflcDiag *diag)
{
    if (m->n_observes == RL_MAX_OBSERVES) {
        kflc_diag_errorf(diag, s->line,
            "too many observation channels (limit %d)", RL_MAX_OBSERVES);
        return 1;
    }
    m->obs_owner[m->n_observes] = owner;
    m->observes[m->n_observes++] = s;
    return 0;
}

/* One `agent <name> ... end` block. The block is a scope over the
 * three constructs and holds nothing else: any other declaration
 * inside it would have an ownership the grammar does not define, and
 * inventing one here is how an ambiguity a reader has to look up gets
 * built in. */

static int rl_collect_agent_(RlModel *m, const KflcNode *s, KflcDiag *diag)
{
    if (m->n_agents == RL_MAX_AGENTS) {
        kflc_diag_errorf(diag, s->line,
            "too many agent blocks (limit %d)", RL_MAX_AGENTS);
        return 1;
    }
    int idx  = m->n_agents++;
    RlAgent *a = &m->agents[idx];
    memset(a, 0, sizeof *a);
    a->node      = s;
    a->name      = s->name ? s->name : "?";
    a->act_first = m->n_actions;
    a->obs_first = m->n_observes;

    int err = 0;
    for (const KflcNode *c = s->children; c; c = c->next) {
        switch (c->kind) {
        case KFLN_STMT_ACTION:
            if (rl_add_action_(m, c, idx, diag)) return 1;
            break;
        case KFLN_STMT_OBSERVE:
            if (!rl_observe_as(c)) {
                kflc_diag_errorf(diag, c->line,
                    "agent `%s`: an `observe` inside an agent block "
                    "declares one of that agent's observation channels "
                    "and needs an `as <name>` clause; an observe that "
                    "only prints belongs at world level", a->name);
                err = 1;
                break;
            }
            if (rl_add_observe_(m, c, idx, diag)) return 1;
            break;
        case KFLN_STMT_OBJECTIVE:
            if (a->objective) {
                kflc_diag_errorf(diag, c->line,
                    "agent `%s`: duplicate `objective` block; an agent "
                    "declares at most one, and the first is at line %d",
                    a->name, a->objective->line);
                err = 1;
                break;
            }
            a->objective = c;
            break;
        default:
            kflc_diag_errorf(diag, c->line,
                "agent `%s`: an agent block holds `action`, "
                "`observe ... as` and `objective` declarations and "
                "nothing else", a->name);
            err = 1;
            break;
        }
    }
    a->n_act = m->n_actions  - a->act_first;
    a->n_obs = m->n_observes - a->obs_first;
    if (a->objective) {
        a->reward   = rl_attr(a->objective, "reward");
        a->terminal = rl_attr(a->objective, "terminal");
    }
    /* An agent with no actions is an observation-only agent, which is
     * legal for the same reason an action total of zero is. An agent
     * with no actions and no objective declares nothing that has an
     * effect, and is more likely a mistake than an intention. */
    if (!err && a->n_act == 0 && !a->objective) {
        kflc_diag_errorf(diag, s->line,
            "agent `%s`: this block declares neither an `action` nor an "
            "`objective`, so it publishes no action channel and no "
            "reward and has no effect on the program", a->name);
        err = 1;
    }
    return err;
}

/* Everything about the agent set that can only be judged once all of
 * it has been read: that world level and block level are not mixed,
 * that the names are usable, and what each agent's slices are. */

static int rl_finish_agents_(RlModel *m, KflcDiag *diag)
{
    int err = 0;
    m->agent_count = m->n_agents > 0 ? m->n_agents : 1;

    /* Mixing. A world-level declaration beside a block has two
     * possible readings, that it belongs to every agent or that it
     * belongs to agent 0, and both would be inventions. */
    if (m->n_agents > 0) {
        const KflcNode *blk = m->agents[0].node;
        const char     *bn  = m->agents[0].name;
        for (int i = 0; i < m->n_actions; i++) {
            if (m->act_owner[i] >= 0) continue;
            kflc_diag_errorf(diag, m->actions[i]->line,
                "action `%s` is declared at world level, and this world "
                "declares `agent %s` at line %d; once any agent block is "
                "present every action, objective and `observe ... as` "
                "belongs to one of them, so move this into the block that "
                "owns it",
                m->actions[i]->name ? m->actions[i]->name : "?",
                bn, blk->line);
            err = 1;
        }
        for (int i = 0; i < m->n_observes; i++) {
            if (m->obs_owner[i] >= 0) continue;
            kflc_diag_errorf(diag, m->observes[i]->line,
                "observe ... as `%s` is declared at world level, and this "
                "world declares `agent %s` at line %d; once any agent "
                "block is present every action, objective and "
                "`observe ... as` belongs to one of them, so move this "
                "into the block that owns it",
                rl_observe_as(m->observes[i]), bn, blk->line);
            err = 1;
        }
        if (m->objective) {
            kflc_diag_errorf(diag, m->objective->line,
                "this `objective` block is declared at world level, and "
                "this world declares `agent %s` at line %d; once any "
                "agent block is present every action, objective and "
                "`observe ... as` belongs to one of them, so move this "
                "into the block that owns it", bn, blk->line);
            err = 1;
        }
        if (err) return 1;
    }

    if (m->n_agents == 0) {
        /* The implicit agent: one agent owning every declaration and
         * publishing unqualified names, which is what a program with
         * no block has always been. */
        RlAgent *a = &m->agents[0];
        memset(a, 0, sizeof *a);
        a->n_act     = m->n_actions;
        a->n_obs     = m->n_observes;
        a->objective = m->objective;
        a->reward    = m->reward;
        a->terminal  = m->terminal;
    }

    for (int i = 0; i < m->n_agents; i++) {
        const RlAgent *a = &m->agents[i];
        size_t len = strlen(a->name);
        if (len > (size_t)RL_AGENT_NAME_MAX) {
            kflc_diag_errorf(diag, a->node->line,
                "agent `%s`: the name is %zu bytes and an agent name may "
                "be at most %d; a published channel name entry holds %d "
                "bytes, a declared channel name may be %d of them, the "
                "longest component suffix is %d, and the qualifying dot "
                "takes one more",
                a->name, len, RL_AGENT_NAME_MAX, KFLC_OBS_NAME_MAX,
                KFLC_OBS_AS_MAX, RL_AGENT_SUFFIX_MAX);
            err = 1;
        }
        for (int j = 0; j < i; j++) {
            if (strcmp(m->agents[j].name, a->name) != 0) continue;
            kflc_diag_errorf(diag, a->node->line,
                "agent `%s`: an agent of that name is already declared at "
                "line %d; agent names are what qualified channel names "
                "are built from and must tell the agents apart",
                a->name, m->agents[j].node->line);
            err = 1;
            break;
        }
        /* An agent name and a body name share the read space inside
         * `on_step`, where `<body>.<key>` is already a dotted shape,
         * so one name cannot stand for both. */
        int bi = rl_body_index_of(m, a->name);
        if (bi >= 0) {
            kflc_diag_errorf(diag, a->node->line,
                "agent `%s`: `astro_body %s` at line %d already claims "
                "that name; both are read by dotted name in this "
                "program's expression scope, so `%s.x` would have two "
                "meanings; rename one",
                a->name, a->name, m->bodies[bi].body->line, a->name);
            err = 1;
        }
        if (strcmp(a->name, "episode") == 0) {
            kflc_diag_errorf(diag, a->node->line,
                "agent `episode`: the name is taken by `episode.steps` in "
                "this program's expression scope; rename the agent");
            err = 1;
        }
    }
    if (err) return 1;

    /* Every block declares at least one observation, above one agent.
     * An agent's name reaches a consumer only through the qualifier on
     * its own observation channel names, so a block with none
     * publishes a slice of (n, 0, 0) and its name does not survive
     * compilation at all: it is absent from the spec blob, from the
     * artifact and therefore from the episode file, and a per-agent
     * API has nothing to key it by. Refusing at the one place the name
     * still exists puts the diagnostic where the author can act on it.
     * At one agent the name publishes unqualified and nothing depends
     * on it, so the rule is not applied there. */
    if (m->agent_count > 1) {
        for (int i = 0; i < m->n_agents; i++) {
            if (m->agents[i].n_obs > 0) continue;
            kflc_diag_errorf(diag, m->agents[i].node->line,
                "agent `%s`: this block declares no `observe ... as`, and "
                "an agent's name reaches a consumer only through the "
                "qualifier on its own observation channels, so above one "
                "agent such a block publishes a name nothing can read; "
                "declare at least one observation",
                m->agents[i].name ? m->agents[i].name : "?");
            err = 1;
        }
    }

    /* Slices. Channels allocate in source order within a block and
     * blocks in source order, so each agent's slice is contiguous and
     * the concatenation over agents is the whole vector. */
    for (int i = 0; i < m->agent_count; i++) {
        RlAgent *a = &m->agents[i];
        a->obs_off   = rl_obs_offset(m->observes, a->obs_first);
        a->obs_count = rl_obs_offset(m->observes, a->obs_first + a->n_obs)
                       - a->obs_off;
        a->act_off   = a->act_first;
        a->act_count = a->n_act;
    }
    return 0;
}

/* The agent that owns observe `i`, or action `i`. With no block
 * declared the implicit agent owns everything. */

int rl_obs_agent(const RlModel *m, int i)
{
    return m->n_agents > 0 ? m->obs_owner[i] : 0;
}

int rl_act_agent(const RlModel *m, int i)
{
    return m->n_agents > 0 ? m->act_owner[i] : 0;
}

/* The published name of channel `c` of observe `i`, written into
 * `out`. With one agent a channel publishes the name it was declared
 * with, which is what makes a single-agent program's spec identical to
 * the one it had before agents existed. With more than one it
 * publishes `<agent>.<channel>`, because two agents may each declare a
 * channel called `rel` and the published name must still tell them
 * apart.
 *
 * A combination the entry cannot hold is refused here rather than
 * truncated in the artifact, and the diagnostic names both parts and
 * the arithmetic: a reader given only the total has to go back to the
 * source and do the subtraction themselves. */

int rl_obs_pub_name(const RlModel *m, int i, int c,
                            char *out, size_t cap, KflcDiag *diag)
{
    const KflcNode *n    = m->observes[i];
    const char     *base = rl_observe_as(n);
    char            cb[RL_COMP_MAX];
    const char     *cmp  = rl_observe_comp(n, c, cb, sizeof cb);
    const char     *qual = NULL;
    size_t          qlen = 0;

    if (m->agent_count > 1) {
        qual = m->agents[rl_obs_agent(m, i)].name;
        qlen = strlen(qual) + 1;
    }
    size_t need = qlen + strlen(base) + strlen(cmp) + 1;
    if (need > cap || need > (size_t)KFLC_OBS_NAME_MAX) {
        if (qual) {
            kflc_diag_errorf(diag, n->line,
                "agent `%s` and `observe ... as %s`: the published "
                "channel name `%s.%s%s` needs %zu bytes and the spec "
                "entry holds %d; the agent name is %zu, the dot is 1, the "
                "channel name is %zu, the suffix `%s` is %zu, and the "
                "terminator is 1",
                qual, base, qual, base, cmp, need, KFLC_OBS_NAME_MAX,
                strlen(qual), strlen(base), cmp, strlen(cmp));
        } else {
            kflc_diag_errorf(diag, n->line,
                "observe ... as `%s`: the derived channel name `%s%s` "
                "needs %zu bytes and the spec entry holds %d",
                base, base, cmp, need, KFLC_OBS_NAME_MAX);
        }
        return 1;
    }
    /* Assembled by copy rather than by format, because the lengths
     * are the ones the test above accepted: a format that could
     * truncate would put a shortened name in the artifact after a
     * check that said it fitted. */
    char *w = out;
    if (qual) {
        size_t ql = strlen(qual);
        memcpy(w, qual, ql); w += ql;
        *w++ = '.';
    }
    size_t bl = strlen(base), cl = strlen(cmp);
    memcpy(w, base, bl); w += bl;
    memcpy(w, cmp, cl);  w += cl;
    *w = '\0';
    return 0;
}

/* The vehicle slot of a body, which is its position among the
 * assembly-bearing bodies in declaration order: the order every
 * per-vehicle table in the artifact is written in. Returns -1 for a
 * body that binds no assembly. */

int rl_veh_slot_of(const RlModel *m, int body)
{
    int seen = 0;
    for (int b = 0; b < m->n_bodies; b++) {
        if (!rl_body_has_assembly(m, b)) continue;
        if (b == body) return seen;
        seen++;
    }
    return -1;
}

/* One `capture_envelope` block, checked and admitted.
 *
 * The parser has already refused anything about the block's shape: a
 * missing field, a repeated one, a field taking the wrong count, a
 * value that is not a number. What is left is what the numbers mean,
 * and the one thing only this pass knows, which is whether the name
 * is already a published envelope's.
 *
 * The bounds are checked in the units they were written in, which is
 * the point of checking them here rather than after conversion: an
 * author reading the diagnostic sees the figure they typed.
 */

static double rl_capenv_field_(const KflcNode *n, const char *key,
                               int *line)
{
    const KflcAttr *a = rl_attr(n, key);
    if (!a) { if (line) *line = n->line; return 0.0; }
    if (line) *line = a->line;
    return a->value.u.f;
}

static int rl_collect_capture_envelope_(RlModel *m, const KflcNode *n,
                                        KflcDiag *diag)
{
    const char *nm = n->name ? n->name : "?";
    if (kflc_capture_builtin(nm)) {
        kflc_diag_errorf(diag, n->line,
            "capture_envelope `%s`: that name belongs to an envelope "
            "this compiler defines, and a declaration may not shadow "
            "one; a program's own envelope needs its own name", nm);
        return 1;
    }
    for (int i = 0; i < m->n_capenv; i++) {
        if (strcmp(m->capenv[i]->name ? m->capenv[i]->name : "", nm) != 0) {
            continue;
        }
        kflc_diag_errorf(diag, n->line,
            "capture_envelope `%s`: an envelope of that name is already "
            "declared at line %d; a port's `capture` mark names one "
            "envelope, so one name is one set of limits",
            nm, m->capenv[i]->line);
        return 1;
    }
    if (m->n_capenv >= KFLC_CAPTURE_MAX_DECLARED) {
        kflc_diag_errorf(diag, n->line,
            "capture_envelope `%s`: more than %d envelopes declared in "
            "one program", nm, KFLC_CAPTURE_MAX_DECLARED);
        return 1;
    }

    /* The eight fields, in the order the block writes them, with the
     * rule each is held to beside it. `angle` marks a misalignment,
     * which is bounded above as well as below: a misalignment past
     * half a turn names the same configuration as its complement and
     * a limit stated there admits every attitude there is. A rate in
     * degrees per second carries no such bound, since a craft may
     * legitimately turn faster than that. */
    static const struct {
        const char *key, *kw, *unit;
        int         angle;
    } F_[] = {
        { "axial_rate_lo", "axial_rate",    "m/s",     0 },
        { "axial_rate_hi", "axial_rate",    "m/s",     0 },
        { "lateral_rate",  "lateral_rate",  "m/s",     0 },
        { "pitchyaw_rate", "pitchyaw_rate", "deg/s",   0 },
        { "roll_rate",     "roll_rate",     "deg/s",   0 },
        { "lateral",       "lateral",       "m",       0 },
        { "pitchyaw",      "pitchyaw",      "deg",     1 },
        { "roll",          "roll",          "deg",     1 }
    };
    int err = 0;
    for (int i = 0; i < (int)(sizeof F_ / sizeof F_[0]); i++) {
        int    line = n->line;
        double v = rl_capenv_field_(n, F_[i].key, &line);
        if (!(v >= 0.0) || !(v < 1.0e300)) {
            kflc_diag_errorf(diag, line,
                "capture_envelope `%s`: `%s` is %.17g %s, and a limit on "
                "a rate or a misalignment is a magnitude, so it cannot "
                "be negative", nm, F_[i].kw, v, F_[i].unit);
            err = 1;
            continue;
        }
        if (F_[i].angle && v > 180.0) {
            kflc_diag_errorf(diag, line,
                "capture_envelope `%s`: `%s` is %.17g degrees, and an "
                "angular misalignment runs from 0 to 180; a limit past "
                "half a turn admits every attitude there is",
                nm, F_[i].kw, v);
            err = 1;
        }
    }
    {
        int    line_lo = n->line, line_hi = n->line;
        double lo = rl_capenv_field_(n, "axial_rate_lo", &line_lo);
        double hi = rl_capenv_field_(n, "axial_rate_hi", &line_hi);
        if (lo > hi) {
            kflc_diag_errorf(diag, line_lo,
                "capture_envelope `%s`: `axial_rate %.17g %.17g` has its "
                "lower bound above its upper, so no closing rate is "
                "inside it and no contact could ever capture", nm, lo, hi);
            err = 1;
        }
    }
    {
        int    line_d = n->line;
        double d = rl_capenv_field_(n, "diameter", &line_d);
        if (!(d > 0.0) || !(d < 1.0e300)) {
            kflc_diag_errorf(diag, line_d,
                "capture_envelope `%s`: `diameter` is %.17g mm, and the "
                "mating plane it sizes is a real interface, so it must "
                "be positive", nm, d);
            err = 1;
        }
    }
    if (err) return 1;

    KflcCaptureEnvelope e;
    memset(&e, 0, sizeof e);
    e.name               = nm;
    e.axial_rate_min     = rl_capenv_field_(n, "axial_rate_lo", NULL);
    e.axial_rate_max     = rl_capenv_field_(n, "axial_rate_hi", NULL);
    e.lateral_rate       = rl_capenv_field_(n, "lateral_rate",  NULL);
    e.pitchyaw_rate_deg  = rl_capenv_field_(n, "pitchyaw_rate", NULL);
    e.roll_rate_deg      = rl_capenv_field_(n, "roll_rate",     NULL);
    e.lateral            = rl_capenv_field_(n, "lateral",       NULL);
    e.pitchyaw_deg       = rl_capenv_field_(n, "pitchyaw",      NULL);
    e.roll_deg           = rl_capenv_field_(n, "roll",          NULL);
    e.mating_diameter_mm = rl_capenv_field_(n, "diameter",      NULL);
    /* Where a built-in envelope carries its document, revision and
     * table, a declared one carries the fact that the figures are the
     * program's: the program's own review is what defends them. */
    e.source = "declared by the program that names it";
    if (kflc_capture_declare(&e)) {
        kflc_diag_errorf(diag, n->line,
            "capture_envelope `%s`: more than %d envelopes declared in "
            "one program", nm, KFLC_CAPTURE_MAX_DECLARED);
        return 1;
    }
    m->capenv[m->n_capenv++] = n;
    return 0;
}

int rl_collect(RlModel *m, const KflcNode *form,
                       KflcArena *arena, KflcDiag *diag)
{
    memset(m, 0, sizeof *m);
    /* The declared envelopes are this program's, so the table they
     * land in is emptied before this program fills it. */
    kflc_capture_declared_reset();

    for (const KflcNode *c = form->children; c; c = c->next) {
        switch (c->kind) {
        case KFLN_FN_WORLD:
            if (m->world) {
                kflc_diag_errorf(diag, c->line,
                    "fn world %s: a reinforcement learning program "
                    "compiles to one environment and admits exactly one "
                    "`fn world`", c->name ? c->name : "?");
                return 1;
            }
            m->world = c;
            break;
        case KFLN_FN_DATA:
        case KFLN_WIDGET:
            kflc_diag_errorf(diag, c->line,
                "plot surfaces are not supported in a reinforcement "
                "learning program; the compiled environment has no "
                "figure output");
            return 1;
        case KFLN_ARENA:
            kflc_diag_errorf(diag, c->line,
                "arena declarations are not supported in a "
                "reinforcement learning program");
            return 1;
        default:
            break;
        }
    }
    if (!m->world) {
        kflc_diag_errorf(diag, form->line,
            "reinforcement learning emission requires a `fn world` block");
        return 1;
    }

    int err = 0;
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        switch (s->kind) {
        case KFLN_STMT_ASTRO_BODY:
            if (m->n_bodies == RL_MAX_BODIES) {
                kflc_diag_errorf(diag, s->line,
                    "too many astro_body declarations (limit %d)",
                    RL_MAX_BODIES);
                return 1;
            }
            if (!s->name) {
                kflc_diag_errorf(diag, s->line,
                    "astro_body requires a name in a reinforcement "
                    "learning world");
                return 1;
            }
            m->bodies[m->n_bodies].body  = s;
            m->bodies[m->n_bodies].index = m->n_bodies;
            m->n_bodies++;
            break;
        case KFLN_STMT_EPISODE:
            m->episode = s;
            for (const KflcNode *r = s->children; r; r = r->next) {
                if (r->kind != KFLN_STMT_EPISODE_RESET) continue;
                if (m->n_resets == RL_MAX_RESETS) {
                    kflc_diag_errorf(diag, r->line,
                        "too many reset lines (limit %d)", RL_MAX_RESETS);
                    return 1;
                }
                m->resets[m->n_resets++] = r;
            }
            break;
        case KFLN_STMT_ACTION:
            if (rl_add_action_(m, s, -1, diag)) return 1;
            break;
        case KFLN_STMT_ON_STEP:
            m->on_step = s;
            break;
        case KFLN_STMT_OBJECTIVE:
            m->objective = s;
            break;
        case KFLN_STMT_AGENT:
            if (rl_collect_agent_(m, s, diag)) err = 1;
            break;
        case KFLN_STMT_SENSOR:
            if (rl_collect_sensor_(m, s, diag)) err = 1;
            break;
        case KFLN_STMT_CAPTURE_ENVELOPE:
            if (rl_collect_capture_envelope_(m, s, diag)) err = 1;
            break;
        case KFLN_STMT_PLAN:
            /* The block's own record. Its action channels were
             * appended after it by the parser and are collected by
             * the ordinary action case above, so the only thing kept
             * here is where the plan goes and what frame it is in. */
            if (m->n_plans == RL_MAX_REF) {
                kflc_diag_errorf(diag, s->line,
                    "too many plan blocks (limit %d)", RL_MAX_REF);
                return 1;
            }
            m->plans[m->n_plans].node = s;
            m->plans[m->n_plans].name = s->name;
            m->n_plans++;
            break;
        case KFLN_STMT_ASTRO_PAYLOAD:
            if (m->n_payloads == RL_MAX_PAYLOADS) {
                kflc_diag_errorf(diag, s->line,
                    "too many astro_payload declarations (limit %d)",
                    RL_MAX_PAYLOADS);
                return 1;
            }
            if (!s->name) {
                kflc_diag_errorf(diag, s->line,
                    "astro_payload requires a name");
                return 1;
            }
            m->payloads[m->n_payloads].node = s;
            m->payloads[m->n_payloads].name = s->name;
            m->payloads[m->n_payloads].line = s->line;
            m->n_payloads++;
            break;
        case KFLN_STMT_OBSERVE:
            if (rl_observe_as(s)) {
                if (rl_add_observe_(m, s, -1, diag)) return 1;
            }
            break;
        default:
            break;
        }
    }

    /* Constructs and body declarations must be top level: body
     * indices, draw channels, and the episode machinery are part of
     * the compiled program's identity and cannot be conditional. */
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        if (s->kind != KFLN_STMT_IF && s->kind != KFLN_STMT_WHILE &&
            s->kind != KFLN_STMT_FOR_EACH) {
            continue;
        }
        const KflcNode *stack[2] = { s->children, s->else_children };
        for (int b = 0; b < 2; b++) {
            for (const KflcNode *c = stack[b]; c; c = c->next) {
                if (c->kind == KFLN_STMT_ASTRO_BODY) {
                    kflc_diag_errorf(diag, c->line,
                        "astro_body declarations in a reinforcement "
                        "learning world must be top level: the body set "
                        "is part of the compiled program's identity");
                    err = 1;
                }
                /* Agent index is source order, and the slices it
                 * decides are published in the artifact's spec, so the
                 * agent set cannot be conditional either. */
                if (c->kind == KFLN_STMT_AGENT) {
                    kflc_diag_errorf(diag, c->line,
                        "agent blocks in a reinforcement learning world "
                        "must be top level: agent index is source order "
                        "and is part of the compiled program's identity");
                    err = 1;
                }
                /* A payload is constructed once per environment in the
                 * world prefix and lives as long as the environment,
                 * so its declaration cannot be conditional either. */
                if (c->kind == KFLN_STMT_ASTRO_PAYLOAD) {
                    kflc_diag_errorf(diag, c->line,
                        "astro_payload declarations in a reinforcement "
                        "learning world must be top level: the payload "
                        "set is part of the compiled program's identity");
                    err = 1;
                }
                /* An envelope's limits become compile-time constants
                 * of every port that names it, so the set of them
                 * cannot be conditional either. */
                if (c->kind == KFLN_STMT_CAPTURE_ENVELOPE) {
                    kflc_diag_errorf(diag, c->line,
                        "capture_envelope declarations in a "
                        "reinforcement learning world must be top "
                        "level: an envelope's limits are compile-time "
                        "constants of the artifact");
                    err = 1;
                }
            }
        }
    }

    if (!m->episode) {
        /* The checker enforces this; emission cannot proceed without
         * a control period, so it is re-checked rather than assumed. */
        kflc_diag_errorf(diag, m->world->line,
            "fn world %s: reinforcement learning emission requires an "
            "`episode` block with `control_dt`",
            m->world->name ? m->world->name : "?");
        return 1;
    }
    m->control_dt      = rl_attr(m->episode, "control_dt");
    m->horizon         = rl_attr(m->episode, "horizon");
    m->substeps        = rl_attr(m->episode, "substeps");
    m->contact_kind    = rl_attr(m->episode, "contact");
    m->restitution     = rl_attr(m->episode, "restitution");
    m->friction        = rl_attr(m->episode, "friction");
    m->terminated_when = rl_attr(m->episode, "terminated_when");
    if (m->objective) {
        m->reward   = rl_attr(m->objective, "reward");
        m->terminal = rl_attr(m->objective, "terminal");
    }
    if (!m->control_dt || !m->control_dt->expr) {
        kflc_diag_errorf(diag, m->episode->line,
            "episode: missing required `control_dt <expr>`");
        return 1;
    }
    /* Before the agent pass, which computes each agent's slice from
     * the channel widths: an effect observe's width is its payload's
     * kind's, so the kind has to be on the statement before any width
     * is asked for. */
    rl_stamp_effect_kinds(m, arena);
    if (rl_finish_agents_(m, diag)) return 1;
    if (rl_finish_payloads(m, form, arena, diag)) err = 1;

    /* Domain-randomisation parameters: distribution-valued astro_body
     * attributes, channels in source order within class 0x0002. */
    for (int i = 0; i < m->n_bodies; i++) {
        for (const KflcAttr *a = m->bodies[i].body->attrs; a; a = a->next) {
            if (!a->name || strcmp(a->name, "parent") == 0) continue;
            KflcExpr *d = rl_attr_dist(a, form, arena, diag, &err);
            if (!d) continue;
            if (m->n_dr == RL_MAX_DR) {
                kflc_diag_errorf(diag, a->line,
                    "too many domain-randomisation parameters (limit %d)",
                    RL_MAX_DR);
                return 1;
            }
            m->dr[m->n_dr].body    = i;
            m->dr[m->n_dr].payload = -1;
            m->dr[m->n_dr].param   = -1;
            m->dr[m->n_dr].attr    = a;
            m->dr[m->n_dr].dist    = d;
            m->dr[m->n_dr].channel = m->n_dr;
            m->n_dr++;
        }
    }
    /* Payload keys, after the bodies and in source order within the
     * payload set. A program that declares no payload allocates
     * exactly the channels it allocated before, so no existing draw
     * coordinate moves. */
    for (int p = 0; p < m->n_payloads; p++) {
        for (int k = 0; k < RL_PAY_MAXP; k++) {
            if (!m->payloads[p].dist[k]) continue;
            if (m->n_dr == RL_MAX_DR) {
                kflc_diag_errorf(diag, m->payloads[p].line,
                    "too many domain-randomisation parameters (limit %d)",
                    RL_MAX_DR);
                return 1;
            }
            m->dr[m->n_dr].body    = -1;
            m->dr[m->n_dr].payload = p;
            m->dr[m->n_dr].param   = k;
            m->dr[m->n_dr].attr    = m->payloads[p].attr[k];
            m->dr[m->n_dr].dist    = m->payloads[p].dist[k];
            m->dr[m->n_dr].channel = m->n_dr;
            m->payloads[p].dr[k]   = m->n_dr;
            m->n_dr++;
        }
    }

    /* World scalars readable by the objective and termination
     * expressions: top-level prefix let/const bindings of scalar type,
     * captured per environment at create. Names the action channels or
     * observation components already claim are shadowed there and not
     * captured. A duplicate top-level name is left to the emitted
     * C++'s own redeclaration error, as the batch emitter leaves it. */
    for (const KflcNode *s = m->world->children; s; s = s->next) {
        if (s->kind != KFLN_STMT_LET && s->kind != KFLN_STMT_CONST) {
            continue;
        }
        if (!s->name || !rl_is_scalar_type(s->type)) continue;
        if (rl_scope_name_taken(m, s->name)) continue;
        int dup = 0;
        for (int j = 0; j < m->n_wscal; j++) {
            const char *wn = m->wscal[j]->name;
            if (wn && strcmp(wn, s->name) == 0) dup = 1;
        }
        if (dup) continue;
        if (m->n_wscal == RL_MAX_WSCAL) {
            kflc_diag_errorf(diag, s->line,
                "too many world scalar bindings (limit %d)",
                RL_MAX_WSCAL);
            return 1;
        }
        m->wscal[m->n_wscal++] = s;
    }

    /* Reset lines must name a declared top-level body. */
    for (int i = 0; i < m->n_resets; i++) {
        const KflcNode *r = m->resets[i];
        int found = -1;
        for (int j = 0; j < m->n_bodies; j++) {
            const char *bn = m->bodies[j].body->name;
            if (bn && r->name && strcmp(bn, r->name) == 0) found = j;
        }
        if (found < 0) {
            kflc_diag_errorf(diag, r->line,
                "episode reset: unknown body `%s` (reset lines target "
                "bodies declared with astro_body in this world)",
                r->name ? r->name : "?");
            err = 1;
            continue;
        }
        const char *rk = (r->position.kind == KFLV_IDENT)
                         ? r->position.u.s : NULL;
        if (rk && kflc_body_state_is_attitude(rk) &&
            !rl_body_has_assembly(m, found)) {
            kflc_diag_errorf(diag, r->line,
                "episode reset: `%s.%s` sets attitude state, but `%s` "
                "declares no `assembly=`, so it has no inertia tensor and "
                "its attitude is never advanced; bind an assembly or drop "
                "the attitude keys", r->name, rk, r->name);
            err = 1;
        }
    }

    /* Resolve each observe's `through <sensor>` to a declared sensor.
     * A name that resolves to nothing is refused where it is written:
     * a program that misspells a sensor would otherwise compile and
     * silently publish uncorrupted values. */
    for (int i = 0; i < m->n_observes; i++) {
        m->obs_sensor[i] = -1;
        const char *want = rl_observe_through_(m->observes[i]);
        if (!want) continue;
        for (int k = 0; k < m->n_sensors; k++) {
            if (m->sensors[k].name && strcmp(m->sensors[k].name, want) == 0) {
                m->obs_sensor[i] = k;
            }
        }
        if (m->obs_sensor[i] < 0) {
            kflc_diag_errorf(diag, m->observes[i]->line,
                "observe ... through `%s`: no sensor of that name is "
                "declared in this world", want);
            err = 1;
        }
    }
    /* `with truth` without a sensor would publish a channel beside an
     * identical one. It is refused rather than allowed to double an
     * observation vector for nothing. */
    for (int i = 0; i < m->n_observes; i++) {
        if (rl_observe_has_truth(m->observes[i]) && m->obs_sensor[i] < 0) {
            kflc_diag_errorf(diag, m->observes[i]->line,
                "observe ... with truth: this observe declares no "
                "`through <sensor>`, so its measured and true values "
                "would be the same numbers");
            err = 1;
        }
    }

    if (!err && rl_collect_actuators_(m, diag)) err = 1;
    if (!err && rl_finish_defense(m, diag)) err = 1;
    if (!err && rl_collect_references(m, arena, diag)) err = 1;
    if (!err && rl_collect_plans(m, arena, diag)) err = 1;

    return err;
}

int rl_body_index_of(const RlModel *m, const char *name)
{
    if (!name) return -1;
    for (int i = 0; i < m->n_bodies; i++) {
        const char *bn = m->bodies[i].body->name;
        if (bn && strcmp(bn, name) == 0) return i;
    }
    return -1;
}
