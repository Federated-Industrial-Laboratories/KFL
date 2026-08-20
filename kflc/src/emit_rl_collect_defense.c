/* emit_rl_collect_defense.c - the model of the defense tier: payloads,
 * engagements, references and plans. */
#include "emit_rl_internal.h"

static const RlPayWord RL_PAY_REGIME_[] = {
    { "ir_only",      "K26ASTRO_DISC_IR_ONLY" },
    { "ir_plus_rcs",  "K26ASTRO_DISC_IR_PLUS_RCS" },
    { "ir_rcs_accel", "K26ASTRO_DISC_IR_RCS_ACCEL" }
};

static const RlPayKey RL_PAY_IR_[] = {
    { "aperture_m",        1, "0.0", NULL, 0 },
    { "integration_s",     1, "0.0", NULL, 0 },
    { "passband_lo_um",    1, "0.0", NULL, 0 },
    { "passband_hi_um",    1, "0.0", NULL, 0 },
    { "throughput",        1, "0.0", NULL, 0 },
    { "snr_threshold",     1, "0.0", NULL, 0 },
    { "target_temp_k",     1, "0.0", NULL, 0 },
    { "target_emissivity", 1, "0.0", NULL, 0 },
    /* Optional, and zero by default, which is what the library
     * documents as recovering its cosmic-background-only behaviour.
     * They are offered because that behaviour is background free for
     * a warm instrument looking in its own emission band, and the
     * library says so; a program that wants the honest noise floor
     * declares its optics rather than rebuilding the model. */
    { "t_optics_k",        0, "0.0", NULL, 0 },
    { "optics_emissivity", 0, "0.0", NULL, 0 },
    RL_PAY_REGIME_KEY
};

static const RlPayKey RL_PAY_RADAR_[] = {
    { "p_tx_w",        1, "0.0", NULL, 0 },
    { "g_tx_db",       1, "0.0", NULL, 0 },
    { "g_rx_db",       1, "0.0", NULL, 0 },
    { "freq_hz",       1, "0.0", NULL, 0 },
    { "loss_sys_db",   1, "0.0", NULL, 0 },
    { "bandwidth_hz",  1, "0.0", NULL, 0 },
    { "t_sys_k",       1, "0.0", NULL, 0 },
    { "noise_figure",  1, "0.0", NULL, 0 },
    { "snr_threshold", 1, "0.0", NULL, 0 },
    RL_PAY_REGIME_KEY,
    { "target_chaff_n_strips",        0, "0.0", NULL, 0 },
    { "target_chaff_sigma_dipole_m2", 0,
      "K26ASTRO_SOFTKILL_DIPOLE_RCS_DEFAULT_M2", NULL, 0 }
};

static const RlPayKey RL_PAY_LIDAR_[] = {
    { "pulse_energy_j",      1, "0.0", NULL, 0 },
    { "wavelength_nm",       1, "0.0", NULL, 0 },
    { "aperture_rx_m",       1, "0.0", NULL, 0 },
    { "atmospheric_tx",      1, "0.0", NULL, 0 },
    { "detector_efficiency", 1, "0.0", NULL, 0 },
    { "snr_threshold",       1, "0.0", NULL, 0 },
    { "target_albedo",       1, "0.0", NULL, 0 },
    RL_PAY_REGIME_KEY
};

static const RlPayKey RL_PAY_INFO_[] = {
    { "history", 0, "1024", NULL, 0 }
};

static const RlPayWord RL_PAY_PATTERN_[] = {
    { "single", "K26ASTRO_IMPACTOR_PATTERN_SINGLE" },
    { "swarm",  "K26ASTRO_IMPACTOR_PATTERN_SWARM" }
};

static const RlPayKey RL_PAY_IMPACTOR_[] = {
    { "pattern",                      1, "0.0",
      RL_PAY_PATTERN_,
      (int)(sizeof RL_PAY_PATTERN_ / sizeof RL_PAY_PATTERN_[0]) },
    { "projectile_mass_kg",           1, "0.0", NULL, 0 },
    { "projectile_density_kg_per_m3", 1, "0.0", NULL, 0 },
    { "projectile_diameter_m",        1, "0.0", NULL, 0 },
    /* Required when `pattern=swarm` and refused when `pattern=single`,
     * which the resolution below applies: the requirement is a
     * function of another key's value and not of the kind. */
    { "swarm_count",                  0, "0.0", NULL, 0 },
    { "swarm_half_angle_rad",         0, "0.0", NULL, 0 },
    { "target_wall_thickness_m",         0, "0.0", NULL, 0 },
    { "target_bumper_thickness_m",       0, "0.0", NULL, 0 },
    { "target_bumper_density_kg_per_m3", 0, "0.0", NULL, 0 },
    { "target_bumper_spacing_m",         0, "0.0", NULL, 0 },
    { "target_wall_yield_stress_ksi",    0, "0.0", NULL, 0 },
    { "target_brinell_hardness",         0, "0.0", NULL, 0 },
    { "target_density_kg_per_m3",        0, "0.0", NULL, 0 },
    { "target_speed_of_sound_m_per_s",   0, "0.0", NULL, 0 },
    { "target_inner_thickness_m",        0, "0.0", NULL, 0 },
    { "target_monolithic_thickness_m",   0, "0.0", NULL, 0 }
};

static const RlPayWord RL_PAY_MATERIAL_[] = {
    { "aluminum",     "K26ASTRO_LASER_MAT_ALUMINUM" },
    { "steel",        "K26ASTRO_LASER_MAT_STEEL" },
    { "titanium",     "K26ASTRO_LASER_MAT_TITANIUM" },
    { "copper",       "K26ASTRO_LASER_MAT_COPPER" },
    { "composite",    "K26ASTRO_LASER_MAT_COMPOSITE" },
    { "fused_silica", "K26ASTRO_LASER_MAT_FUSED_SILICA" }
};

static const RlPayKey RL_PAY_LASER_[] = {
    { "primary_diam_m",      1, "0.0", NULL, 0 },
    { "wavelength_nm",       1, "0.0", NULL, 0 },
    { "p_output_w",          1, "0.0", NULL, 0 },
    { "m_squared",           1, "0.0", NULL, 0 },
    { "pointing_jitter_rad", 1, "0.0", NULL, 0 },
    { "rms_wavefront_m",     1, "0.0", NULL, 0 },
    { "plasma_attn_k",       1, "0.0", NULL, 0 },
    { "target_material",     1, "0.0",
      RL_PAY_MATERIAL_,
      (int)(sizeof RL_PAY_MATERIAL_ / sizeof RL_PAY_MATERIAL_[0]) },
    { "target_reflectivity", 1, "0.0", NULL, 0 }
};

static const RlPayWord RL_PAY_DECOY_MODE_[] = {
    { "passive", "K26ASTRO_DECOY_MODE_PASSIVE" },
    { "active",  "K26ASTRO_DECOY_MODE_ACTIVE" }
};

static const RlPayKey RL_PAY_DECOY_[] = {
    { "mode", 1, "0.0",
      RL_PAY_DECOY_MODE_,
      (int)(sizeof RL_PAY_DECOY_MODE_ / sizeof RL_PAY_DECOY_MODE_[0]) },
    { "dry_mass_kg",         1, "0.0", NULL, 0 },
    { "deploy_dv_mps",       1, "0.0", NULL, 0 },
    { "ir_match_quality",    1, "0.0", NULL, 0 },
    { "rcs_match_quality",   1, "0.0", NULL, 0 },
    { "accel_match_quality", 1, "0.0", NULL, 0 }
};

static const RlPayWord RL_PAY_JAMMER_MODE_[] = {
    { "noise",       "K26ASTRO_JAMMER_MODE_NOISE" },
    { "cover_pulse", "K26ASTRO_JAMMER_MODE_COVER_PULSE" },
    { "deception",   "K26ASTRO_JAMMER_MODE_DECEPTION" }
};

static const RlPayKey RL_PAY_JAMMER_[] = {
    { "mode", 1, "0.0",
      RL_PAY_JAMMER_MODE_,
      (int)(sizeof RL_PAY_JAMMER_MODE_ / sizeof RL_PAY_JAMMER_MODE_[0]) },
    { "p_j_w",           1, "0.0", NULL, 0 },
    { "g_j_db",          1, "0.0", NULL, 0 },
    { "freq_hz",         1, "0.0", NULL, 0 },
    { "bandwidth_hz",    1, "0.0", NULL, 0 },
    { "snr_threshold",   1, "0.0", NULL, 0 },
    { "radiator_temp_k", 1, "0.0", NULL, 0 }
};

const RlPayKindDesc RL_PAY_KIND_[RL_PAY_KINDS] = {
    { "detect_ir",    "K26ASTRO_DEFENSE_KIND_DETECT_SENSOR",
      RL_PAY_IR_,    (int)(sizeof RL_PAY_IR_    / sizeof RL_PAY_IR_[0]),
      1, 0, 0 },
    { "detect_radar", "K26ASTRO_DEFENSE_KIND_DETECT_SENSOR",
      RL_PAY_RADAR_, (int)(sizeof RL_PAY_RADAR_ / sizeof RL_PAY_RADAR_[0]),
      1, 0, 0 },
    { "detect_lidar", "K26ASTRO_DEFENSE_KIND_DETECT_SENSOR",
      RL_PAY_LIDAR_, (int)(sizeof RL_PAY_LIDAR_ / sizeof RL_PAY_LIDAR_[0]),
      1, 0, 0 },
    { "infostate",    "K26ASTRO_DEFENSE_KIND_INFOSTATE",
      RL_PAY_INFO_,  (int)(sizeof RL_PAY_INFO_  / sizeof RL_PAY_INFO_[0]),
      0, 0, 0 },
    { "impactor",     "K26ASTRO_DEFENSE_KIND_IMPACTOR",
      RL_PAY_IMPACTOR_,
      (int)(sizeof RL_PAY_IMPACTOR_ / sizeof RL_PAY_IMPACTOR_[0]),
      0, 1, 0 },
    { "laser",        "K26ASTRO_DEFENSE_KIND_LASER",
      RL_PAY_LASER_,
      (int)(sizeof RL_PAY_LASER_ / sizeof RL_PAY_LASER_[0]), 0, 1, 0 },
    { "decoy",        "K26ASTRO_DEFENSE_KIND_DECOY",
      RL_PAY_DECOY_,
      (int)(sizeof RL_PAY_DECOY_ / sizeof RL_PAY_DECOY_[0]), 0, 1, 1 },
    { "jammer",       "K26ASTRO_DEFENSE_KIND_JAMMER",
      RL_PAY_JAMMER_,
      (int)(sizeof RL_PAY_JAMMER_ / sizeof RL_PAY_JAMMER_[0]), 0, 1, 1 }
};

static const RlPayUnimplemented RL_PAY_UNIMPLEMENTED_[] = {
    { "dazzler", "K26ASTRO_DEFENSE_KIND_DAZZLER" }
};

static void rl_node_add_ident_(KflcArena *arena, KflcNode *n,
                               const char *key, const char *value);
static void rl_stamp_effect_kind_(RlModel *m, KflcNode *o, KflcArena *arena);
static int rl_collect_engages_(RlModel *m, KflcNode *stmts, KflcDiag *diag);
static int rl_ref_resolve_(const char *path, const char *src_path,
                           char *out, size_t out_sz);
static const char *rl_unquote_(KflcArena *arena, const char *raw);

int rl_pay_kind_from_name(const char *s)
{
    if (!s) return -1;
    for (int k = 0; k < RL_PAY_KINDS; k++) {
        if (strcmp(RL_PAY_KIND_[k].name, s) == 0) return k;
    }
    return -1;
}

/* The library type a payload handle points at, and the function that
 * frees it. One place, so the construction and the teardown of a kind
 * cannot name different types. */

const char *rl_pay_ctype(int kind)
{
    switch (kind) {
    case RL_PAY_INFOSTATE: return "K26AstroInfostate";
    case RL_PAY_IMPACTOR:  return "K26AstroImpactor";
    case RL_PAY_LASER:     return "K26AstroLaser";
    case RL_PAY_DECOY:     return "K26AstroDecoy";
    case RL_PAY_JAMMER:    return "K26AstroJammer";
    default:               return "K26AstroDetectSensor";
    }
}

const char *rl_pay_dtor(int kind)
{
    switch (kind) {
    case RL_PAY_INFOSTATE: return "k26astro_infostate_destroy";
    case RL_PAY_IMPACTOR:  return "k26astro_impactor_destroy";
    case RL_PAY_LASER:     return "k26astro_laser_destroy";
    case RL_PAY_DECOY:     return "k26astro_decoy_destroy";
    case RL_PAY_JAMMER:    return "k26astro_jammer_destroy";
    default:               return "k26astro_detect_sensor_destroy";
    }
}

const char *rl_pay_attr_text(const KflcAttr *a)
{
    if (a && a->value.kind == KFLV_IDENT && a->value.u.s) return a->value.u.s;
    return NULL;
}

/* Resolve every `astro_payload`: its kind, the body it is carried by,
 * and its key set. Every refusal names what a reader has to change,
 * which is why a key belonging to another kind is refused naming that
 * kind rather than reported as unknown: the second diagnostic sends a
 * reader to the grammar reference and the first answers them. */

int rl_finish_payloads(RlModel *m, const KflcNode *form,
                               KflcArena *arena, KflcDiag *diag)
{
    int err = 0;
    for (int p = 0; p < m->n_payloads; p++) {
        RlPayload *py = &m->payloads[p];
        py->kind = -1;
        py->body = -1;
        py->veh  = -1;
        for (int k = 0; k < RL_PAY_MAXP; k++) {
            py->attr[k] = NULL;
            py->dist[k] = NULL;
            py->dr[k]   = -1;
        }
        for (int q = 0; q < p; q++) {
            if (m->payloads[q].name && py->name &&
                strcmp(m->payloads[q].name, py->name) == 0) {
                kflc_diag_errorf(diag, py->line,
                    "astro_payload `%s`: a payload of that name is already "
                    "declared at line %d", py->name, m->payloads[q].line);
                err = 1;
            }
        }

        const KflcAttr *kind_a = rl_attr(py->node, "kind");
        const KflcAttr *body_a = rl_attr(py->node, "body");
        const char *kind_s = rl_pay_attr_text(kind_a);
        const char *body_s = rl_pay_attr_text(body_a);
        if (!kind_s) {
            kflc_diag_errorf(diag, py->line,
                "astro_payload `%s`: missing required `kind=`; the kinds "
                "this grammar binds are " RL_PAY_KIND_LIST, py->name);
            err = 1;
        } else {
            py->kind = rl_pay_kind_from_name(kind_s);
            if (py->kind < 0) {
                /* A kind the tier's own registry names and no library
                 * in the tree implements is answered with that, rather
                 * than with the message for a misspelling: a reader who
                 * found the tag in the registry header did not get the
                 * name wrong, and telling them it is unknown would send
                 * them back to check a spelling that is correct. */
                const char *unimp = NULL;
                for (int u = 0;
                     u < (int)(sizeof RL_PAY_UNIMPLEMENTED_ /
                               sizeof RL_PAY_UNIMPLEMENTED_[0]); u++) {
                    if (strcmp(RL_PAY_UNIMPLEMENTED_[u].name, kind_s) == 0) {
                        unimp = RL_PAY_UNIMPLEMENTED_[u].tag;
                    }
                }
                if (unimp) {
                    kflc_diag_errorf(diag, py->line,
                        "astro_payload `%s`: kind `%s` is named by the "
                        "defense kind registry as `%s`, but no library in "
                        "this tree implements it: there is no constructor, "
                        "no evaluator and no event for it, so this grammar "
                        "does not offer the kind. The kinds it binds are "
                        RL_PAY_KIND_LIST, py->name, kind_s, unimp);
                } else {
                    kflc_diag_errorf(diag, py->line,
                        "astro_payload `%s`: unknown kind `%s`; the kinds "
                        "this grammar binds are " RL_PAY_KIND_LIST,
                        py->name, kind_s);
                }
                err = 1;
            }
        }
        if (!body_s) {
            kflc_diag_errorf(diag, py->line,
                "astro_payload `%s`: missing required `body=`, which names "
                "the astro_body that carries the payload", py->name);
            err = 1;
        } else {
            py->body = rl_body_index_of(m, body_s);
            if (py->body < 0) {
                kflc_diag_errorf(diag, py->line,
                    "astro_payload `%s`: `body=%s` names no astro_body "
                    "declared in this world", py->name, body_s);
                err = 1;
            } else if (!rl_body_has_assembly(m, py->body)) {
                kflc_diag_errorf(diag, py->line,
                    "astro_payload `%s`: `body=%s` declares no `assembly=` "
                    "at line %d, so it carries no vehicle and nothing can "
                    "be bound to it", py->name, body_s,
                    m->bodies[py->body].body->line);
                err = 1;
            } else {
                py->veh = rl_veh_slot_of(m, py->body);
            }
        }
        if (py->kind < 0) continue;

        const RlPayKindDesc *kd = &RL_PAY_KIND_[py->kind];
        for (const KflcAttr *a = py->node->attrs; a; a = a->next) {
            if (!a->name) continue;
            if (strcmp(a->name, "kind") == 0) continue;
            if (strcmp(a->name, "body") == 0) continue;
            int slot = -1;
            for (int k = 0; k < kd->n_keys; k++) {
                if (strcmp(kd->keys[k].key, a->name) == 0) slot = k;
            }
            if (slot < 0) {
                const char *owner = NULL;
                for (int k = 0; k < RL_PAY_KINDS && !owner; k++) {
                    if (k == py->kind) continue;
                    for (int j = 0; j < RL_PAY_KIND_[k].n_keys; j++) {
                        if (strcmp(RL_PAY_KIND_[k].keys[j].key,
                                   a->name) == 0) {
                            owner = RL_PAY_KIND_[k].name;
                            break;
                        }
                    }
                }
                if (owner) {
                    kflc_diag_errorf(diag, a->line,
                        "astro_payload `%s`: `%s=` belongs to kind `%s`, "
                        "not to `%s`", py->name, a->name, owner, kd->name);
                } else {
                    kflc_diag_errorf(diag, a->line,
                        "astro_payload `%s`: unknown key `%s=` for kind "
                        "`%s`", py->name, a->name, kd->name);
                }
                err = 1;
                continue;
            }
            py->attr[slot] = a;
            /* A keyword-valued key takes one of its own words and no
             * distribution: the value stands for a library constant,
             * so there is nothing between two of them to draw from,
             * and a number there would be a program depending on an
             * internal numbering no document promises it. */
            if (kd->keys[slot].words) {
                const char *v = rl_pay_attr_text(a);
                int known = 0;
                for (int w = 0; w < kd->keys[slot].n_words; w++) {
                    if (v && strcmp(kd->keys[slot].words[w].word, v) == 0) {
                        known = 1;
                    }
                }
                if (!known) {
                    char list[256];
                    size_t used = 0;
                    list[0] = '\0';
                    for (int w = 0; w < kd->keys[slot].n_words; w++) {
                        int wrote = snprintf(list + used, sizeof list - used,
                                             "%s%s", w ? ", " : "",
                                             kd->keys[slot].words[w].word);
                        if (wrote < 0 || (size_t)wrote >= sizeof list - used) {
                            break;
                        }
                        used += (size_t)wrote;
                    }
                    kflc_diag_errorf(diag, a->line,
                        "astro_payload `%s`: `%s=%s` is not one of the "
                        "words this key takes, which are %s; the value "
                        "names a library constant rather than a number",
                        py->name, a->name, v ? v : "?", list);
                    err = 1;
                }
                continue;
            }
            int derr = 0;
            KflcExpr *d = rl_attr_dist(a, form, arena, diag, &derr);
            if (derr) err = 1;
            if (d && py->kind == RL_PAY_INFOSTATE) {
                /* The history capacity is fixed when the ring is
                 * allocated, and the ring is allocated once in the
                 * world prefix, so a per-episode draw could not reach
                 * it. Refusing says so rather than drawing a number
                 * nothing applies. */
                kflc_diag_errorf(diag, a->line,
                    "astro_payload `%s`: `%s=` is fixed when the payload "
                    "is constructed and admits no distribution form",
                    py->name, a->name);
                err = 1;
                continue;
            }
            py->dist[slot] = d;
        }
        for (int k = 0; k < kd->n_keys; k++) {
            if (kd->keys[k].required && !py->attr[k]) {
                kflc_diag_errorf(diag, py->line,
                    "astro_payload `%s`: kind `%s` requires `%s=`",
                    py->name, kd->name, kd->keys[k].key);
                err = 1;
            }
        }
        if (kd->n_keys > m->pay_nparam) m->pay_nparam = kd->n_keys;

        /* The impactor's two swarm keys are required by the release
         * pattern rather than by the kind: a swarm without a count and
         * a cone angle has no footprint, and a single projectile has
         * no use for either, so declaring one there would be a number
         * nothing reads. The constructor refuses the first case by
         * returning null, which would surface as a world that failed
         * to build; refusing here names the key instead. */
        if (py->kind == RL_PAY_IMPACTOR && py->attr[0]) {
            const char *pat = rl_pay_attr_text(py->attr[0]);
            int swarm = pat && strcmp(pat, "swarm") == 0;
            for (int k = 4; k <= 5; k++) {
                if (swarm && !py->attr[k]) {
                    kflc_diag_errorf(diag, py->line,
                        "astro_payload `%s`: `pattern=swarm` requires "
                        "`%s=`, which a single projectile has no use for "
                        "and a swarm has no footprint without",
                        py->name, kd->keys[k].key);
                    err = 1;
                }
                if (!swarm && py->attr[k]) {
                    kflc_diag_errorf(diag, py->attr[k]->line,
                        "astro_payload `%s`: `%s=` belongs to "
                        "`pattern=swarm` and this payload declares "
                        "`pattern=%s`, where nothing would read it",
                        py->name, kd->keys[k].key, pat ? pat : "?");
                    err = 1;
                }
            }
        }

        /* The chaff pair, on the same rule as the swarm keys: a dipole
         * cross-section without a strip count describes strips that do
         * not exist, and the mean cross-section of nought strips is
         * nought whatever each one of them would have returned, so the
         * declared figure would be a number nothing reads. The strip
         * count alone is admitted, taking the library's own X-band
         * default for the dipole. */
        if (py->kind == RL_PAY_DETECT_RADAR &&
            py->attr[RL_PAY_RADAR_CHAFF_SIG] &&
            !py->attr[RL_PAY_RADAR_CHAFF_N]) {
            kflc_diag_errorf(diag, py->attr[RL_PAY_RADAR_CHAFF_SIG]->line,
                "astro_payload `%s`: `%s=` describes one strip of a chaff "
                "cloud and this payload declares no `%s=`, so there is no "
                "cloud for it to describe and nothing would read it",
                py->name, kd->keys[RL_PAY_RADAR_CHAFF_SIG].key,
                kd->keys[RL_PAY_RADAR_CHAFF_N].key);
            err = 1;
        }

        /* And the strip count against the range the cloud statistics
         * routine's own parameter has. Past it the conversion is
         * undefined and what it produces reads as a cloud of no strips,
         * which is a declared figure nothing reads: the same ground the
         * pair rule above stands on, and the same ground the history
         * minimum stands on. A distribution is admissible here and only
         * a literal can be judged now, so what can be judged is. */
        if (py->kind == RL_PAY_DETECT_RADAR &&
            py->attr[RL_PAY_RADAR_CHAFF_N]) {
            const char *txt = rl_pay_attr_text(py->attr[RL_PAY_RADAR_CHAFF_N]);
            char *end = NULL;
            double v = txt ? strtod(txt, &end) : 0.0;
            if (txt && end && *end == '\0' &&
                (v < 0.0 || v > 2147483647.0)) {
                kflc_diag_errorf(diag,
                    py->attr[RL_PAY_RADAR_CHAFF_N]->line,
                    "astro_payload `%s`: `%s=%s` is outside the range the "
                    "cloud statistics routine counts strips in, which is 0 "
                    "to 2147483647; a count past it describes a cloud the "
                    "routine cannot represent and would be read as no "
                    "cloud at all",
                    py->name, kd->keys[RL_PAY_RADAR_CHAFF_N].key, txt);
                err = 1;
            }
        }

        /* At most one information state per body. It binds through the
         * vehicle's singleton payload slot, so a second evicts the
         * first and the eviction nulls the evicted one's observer: its
         * nine channels then read invalid for the rest of the run,
         * which is exactly the silence the target cap above exists to
         * prevent. Detection payloads bind through the list slot and
         * are unaffected, so the rule is this kind's alone. */
        if (py->kind == RL_PAY_INFOSTATE && py->body >= 0) {
            for (int q = 0; q < p; q++) {
                if (m->payloads[q].kind != RL_PAY_INFOSTATE) continue;
                if (m->payloads[q].body != py->body) continue;
                kflc_diag_errorf(diag, py->line,
                    "astro_payload `%s`: `%s` already carries the "
                    "information state `%s` declared at line %d, and a "
                    "body carries at most one: the second would evict "
                    "the first, whose channels would then report "
                    "nothing for the rest of the run",
                    py->name, m->bodies[py->body].body->name,
                    m->payloads[q].name, m->payloads[q].line);
                err = 1;
                break;
            }
        }

        /* D7's other half: the history capacity the library documents
         * a minimum for. The library clamps a smaller value silently;
         * a reader who has been told the minimum takes it for a
         * refusal, so it is one. A distribution is already refused
         * above, so the value here is a literal or an expression the
         * emitted code evaluates, and only a literal can be judged
         * now: what can be judged is judged. */
        if (py->kind == RL_PAY_INFOSTATE && py->attr[0]) {
            const char *txt = rl_pay_attr_text(py->attr[0]);
            char *end = NULL;
            double v = txt ? strtod(txt, &end) : 0.0;
            if (txt && end && *end == '\0' &&
                v < (double)K26ASTRO_INFOSTATE_MIN_HISTORY_CAPACITY) {
                kflc_diag_errorf(diag, py->attr[0]->line,
                    "astro_payload `%s`: `history=%s` is below the "
                    "minimum of %d the information state needs to "
                    "interpolate between two samples",
                    py->name, txt,
                    K26ASTRO_INFOSTATE_MIN_HISTORY_CAPACITY);
                err = 1;
            }
        }
    }
    return err;
}

/* The modality an `observe track` declared. The library records it
 * into the observation snapshot unchanged and applies no per-modality
 * processing, and says so; the grammar admits the five it names and
 * refuses anything else rather than passing a number through. */

int rl_track_modality(const KflcNode *n, const char **out_bad)
{
    const KflcAttr *a = rl_attr(n, "modality");
    if (!a) return 0;
    const char *v = rl_pay_attr_text(a);
    if (!v) { *out_bad = "?"; return -1; }
    if (strcmp(v, "none") == 0)  return 0;
    if (strcmp(v, "ir") == 0)    return 1;
    if (strcmp(v, "radar") == 0) return 2;
    if (strcmp(v, "lidar") == 0) return 3;
    if (strcmp(v, "ephem") == 0) return 4;
    *out_bad = v;
    return -1;
}

/* Append one identifier-valued attribute to a statement. The parser's
 * own helper is private to it, and this pass adds an attribute the
 * author did not write: the effector kind an effect observe publishes
 * the channels of. */

static void rl_node_add_ident_(KflcArena *arena, KflcNode *n,
                               const char *key, const char *value)
{
    KflcAttr *a = (KflcAttr *)kflc_arena_alloc(arena, sizeof *a);
    if (!a) return;
    memset(a, 0, sizeof *a);
    a->name       = kflc_arena_strdup(arena, key);
    a->value.kind = KFLV_IDENT;
    a->value.u.s  = kflc_arena_strdup(arena, value);
    a->line       = n->line;
    if (!n->attrs) { n->attrs = a; return; }
    KflcAttr *t = n->attrs;
    while (t->next) t = t->next;
    t->next = a;
}

/* Resolve each effect observe's payload kind onto the statement.
 *
 * The two effector kinds publish different channel sets, so this
 * form's width is a function of its payload's kind. The agent slices
 * are computed from the widths and are computed before the payloads
 * are resolved, so the kind is read straight off the `astro_payload`
 * statement here and stamped where the width function can see it. A
 * payload that does not resolve is left unstamped and is refused with
 * its own diagnostic further down. */

static void rl_stamp_effect_kind_(RlModel *m, KflcNode *o, KflcArena *arena)
{
    if (o->kind != KFLN_STMT_OBSERVE) return;
    if (rl_observe_form(o) != RL_OBS_EFF) return;
    const char *pn = rl_observe_payload(o);
    if (!pn) return;
    for (int p = 0; p < m->n_payloads; p++) {
        if (!m->payloads[p].name) continue;
        if (strcmp(m->payloads[p].name, pn) != 0) continue;
        const char *kind_s =
            rl_pay_attr_text(rl_attr(m->payloads[p].node, "kind"));
        if (kind_s) rl_node_add_ident_(arena, o, "effect_kind", kind_s);
        return;
    }
}

void rl_stamp_effect_kinds(RlModel *m, KflcArena *arena)
{
    for (KflcNode *s = m->world ? m->world->children : NULL; s; s = s->next) {
        if (s->kind == KFLN_STMT_AGENT) {
            for (KflcNode *o = s->children; o; o = o->next) {
                rl_stamp_effect_kind_(m, o, arena);
            }
            continue;
        }
        rl_stamp_effect_kind_(m, s, arena);
    }
}

/* Collect every `engage` in the step body, in source order over the
 * whole block including the branches of any conditional in it. Source
 * order is the order the statements run in and therefore the order the
 * effects apply in, which is already the rule for a body state write.
 * Returns non-zero when the block holds more than the fixed limit. */

static int rl_collect_engages_(RlModel *m, KflcNode *stmts, KflcDiag *diag)
{
    for (KflcNode *s = stmts; s; s = s->next) {
        if (s->kind == KFLN_STMT_ENGAGE) {
            if (m->n_engages == RL_MAX_ENGAGE) {
                kflc_diag_errorf(diag, s->line,
                    "too many `engage` statements in one step body "
                    "(limit %d)", RL_MAX_ENGAGE);
                return 1;
            }
            RlEngage *e = &m->engages[m->n_engages++];
            e->node    = s;
            e->payload = -1;
            e->target  = -1;
            e->line    = s->line;
            continue;
        }
        if (rl_collect_engages_(m, s->children, diag)) return 1;
        if (rl_collect_engages_(m, s->else_children, diag)) return 1;
    }
    return 0;
}

/* Resolve the three defense observation forms and every engagement,
 * and derive the silhouette of every body either needs. Runs after the
 * actuator pass, which is what puts the assemblies' colliders in the
 * model. */

int rl_finish_defense(RlModel *m, KflcDiag *diag)
{
    int err = 0;
    for (int i = 0; i < m->n_observes; i++) {
        m->obs_payload[i] = -1;
        m->obs_target[i]  = -1;
    }
    /* The effect form names a payload and no body: an effector's
     * result belongs to the effector, and the body it was aimed at was
     * named by the `engage` that produced it. */
    for (int i = 0; i < m->n_observes; i++) {
        const KflcNode *s = m->observes[i];
        if (rl_observe_form(s) != RL_OBS_EFF) continue;
        const char *pn = rl_observe_payload(s);
        int p = -1;
        for (int q = 0; q < m->n_payloads; q++) {
            if (m->payloads[q].name && pn &&
                strcmp(m->payloads[q].name, pn) == 0) p = q;
        }
        if (p < 0) {
            kflc_diag_errorf(diag, s->line,
                "observe effect %s: no astro_payload of that name is "
                "declared in this world", pn ? pn : "?");
            err = 1;
            continue;
        }
        if (m->payloads[p].kind < 0) { err = 1; continue; }
        const RlPayKindDesc *kd = &RL_PAY_KIND_[m->payloads[p].kind];
        if (!kd->is_effector) {
            kflc_diag_errorf(diag, s->line,
                "observe effect %s: `%s` is of kind `%s`, which produces "
                "no engagement event; this form takes an effector "
                "payload", pn, pn, kd->name);
            err = 1;
            continue;
        }
        if (rl_attr(s, "modality")) {
            kflc_diag_errorf(diag, s->line,
                "observe effect %s: `modality=` belongs to the "
                "`observe track` form", pn);
            err = 1;
        }
        m->obs_payload[i] = p;
    }
    for (int i = 0; i < m->n_observes; i++) {
        const KflcNode *s = m->observes[i];
        RlObserveForm f = rl_observe_form(s);
        if (f != RL_OBS_DET && f != RL_OBS_TRK) continue;
        const char *word = (f == RL_OBS_DET) ? "detect" : "track";
        const char *pn = rl_observe_payload(s);
        int p = -1;
        for (int q = 0; q < m->n_payloads; q++) {
            if (m->payloads[q].name && pn &&
                strcmp(m->payloads[q].name, pn) == 0) p = q;
        }
        if (p < 0) {
            kflc_diag_errorf(diag, s->line,
                "observe %s %s of %s: no astro_payload of that name is "
                "declared in this world", word, pn ? pn : "?",
                s->name ? s->name : "?");
            err = 1;
            continue;
        }
        if (m->payloads[p].kind < 0) { err = 1; continue; }
        const RlPayKindDesc *kd = &RL_PAY_KIND_[m->payloads[p].kind];
        if (f == RL_OBS_DET && !kd->is_detect) {
            kflc_diag_errorf(diag, s->line,
                "observe detect %s of %s: `%s` is of kind `%s`; this form "
                "takes a detection payload", pn, s->name ? s->name : "?",
                pn, kd->name);
            err = 1;
            continue;
        }
        if (f == RL_OBS_TRK && m->payloads[p].kind != RL_PAY_INFOSTATE) {
            kflc_diag_errorf(diag, s->line,
                "observe track %s of %s: `%s` is of kind `%s`; this form "
                "takes a payload of kind `infostate`", pn,
                s->name ? s->name : "?", pn, kd->name);
            err = 1;
            continue;
        }
        const char *bad = NULL;
        if (f == RL_OBS_DET && rl_attr(s, "modality")) {
            kflc_diag_errorf(diag, s->line,
                "observe detect %s of %s: `modality=` belongs to the "
                "`observe track` form, which records it into the "
                "information state's own snapshot", pn,
                s->name ? s->name : "?");
            err = 1;
        }
        if (f == RL_OBS_TRK && rl_track_modality(s, &bad) < 0) {
            kflc_diag_errorf(diag, s->line,
                "observe track %s of %s: unknown modality `%s`; the "
                "modalities are none, ir, radar, lidar and ephem", pn,
                s->name ? s->name : "?", bad ? bad : "?");
            err = 1;
        }
        int tgt = rl_body_index_of(m, s->name);
        if (tgt < 0) {
            kflc_diag_errorf(diag, s->line,
                "observe %s %s of `%s`: no astro_body of that name is "
                "declared in this world", word, pn,
                s->name ? s->name : "?");
            err = 1;
            continue;
        }
        if (!rl_body_has_assembly(m, tgt)) {
            kflc_diag_errorf(diag, s->line,
                "observe %s %s of `%s`: `%s` declares no `assembly=`, so "
                "it carries neither the geometry a signature is computed "
                "from nor the vehicle a track is kept against", word, pn,
                s->name, s->name);
            err = 1;
            continue;
        }
        if (tgt == m->payloads[p].body) {
            kflc_diag_errorf(diag, s->line,
                "observe %s %s of `%s`: `%s` is carried by `%s` itself, "
                "and a payload does not observe its own platform", word,
                pn, s->name, pn, s->name);
            err = 1;
            continue;
        }
        m->obs_payload[i] = p;
        m->obs_target[i]  = tgt;
    }
    if (err) return 1;

    /* The per-observer target cap the library documents. Past it a
     * push is a silent no-op, which would leave a channel reading
     * invalid forever for a reason nothing reports, so it is refused
     * here naming the payload, the count and the limit. */
    for (int p = 0; p < m->n_payloads; p++) {
        if (m->payloads[p].kind != RL_PAY_INFOSTATE) continue;
        int seen[RL_MAX_BODIES];
        int n_seen = 0;
        for (int i = 0; i < m->n_observes; i++) {
            if (m->obs_payload[i] != p) continue;
            if (rl_observe_form(m->observes[i]) != RL_OBS_TRK) continue;
            int dup = 0;
            for (int q = 0; q < n_seen; q++) {
                if (seen[q] == m->obs_target[i]) dup = 1;
            }
            if (!dup && n_seen < RL_MAX_BODIES) seen[n_seen++] = m->obs_target[i];
        }
        if (n_seen > K26ASTRO_INFOSTATE_MAX_TARGETS) {
            kflc_diag_errorf(diag, m->payloads[p].line,
                "astro_payload `%s`: %d targets are tracked against this "
                "information state and the per-observer limit is %d",
                m->payloads[p].name, n_seen,
                K26ASTRO_INFOSTATE_MAX_TARGETS);
            err = 1;
        }
    }

    /* Every engagement, resolved to the payload it fires and the body
     * it is aimed at. */
    if (m->on_step &&
        rl_collect_engages_(m, m->on_step->children, diag)) {
        return 1;
    }
    for (int e = 0; e < m->n_engages; e++) {
        RlEngage *en = &m->engages[e];
        const char *pn = en->node->name;
        const KflcAttr *at = rl_attr(en->node, "at");
        const char *tn = rl_pay_attr_text(at);
        for (int q = 0; q < m->n_payloads; q++) {
            if (m->payloads[q].name && pn &&
                strcmp(m->payloads[q].name, pn) == 0) en->payload = q;
        }
        if (en->payload < 0) {
            kflc_diag_errorf(diag, en->line,
                "engage %s at %s: no astro_payload of that name is "
                "declared in this world", pn ? pn : "?", tn ? tn : "?");
            err = 1;
            continue;
        }
        if (m->payloads[en->payload].kind < 0) { err = 1; continue; }
        const RlPayKindDesc *kd =
            &RL_PAY_KIND_[m->payloads[en->payload].kind];
        if (!kd->is_effector) {
            kflc_diag_errorf(diag, en->line,
                "engage %s at %s: `%s` is of kind `%s`, which is not an "
                "effector; only an effector payload is engaged",
                pn, tn ? tn : "?", pn, kd->name);
            err = 1;
            continue;
        }
        /* At most one engagement of a payload per step. Two would make
         * the published result depend on statement order with no
         * channel reporting which one it came from, and both would
         * have applied their effect to the world. Refused here when
         * both statements name the payload directly; the emitted
         * artifact faults on the second call for the cases this test
         * cannot see, such as one inside a loop. */
        for (int q = 0; q < e; q++) {
            if (m->engages[q].payload != en->payload) continue;
            kflc_diag_errorf(diag, en->line,
                "engage %s at %s: `%s` is already engaged at line %d in "
                "this step body, and a payload is engaged at most once "
                "per step: the published result would depend on which "
                "statement ran last with nothing reporting it",
                pn, tn ? tn : "?", pn, m->engages[q].line);
            err = 1;
            break;
        }
        en->target = rl_body_index_of(m, tn);
        if (en->target < 0) {
            kflc_diag_errorf(diag, en->line,
                "engage %s at `%s`: no astro_body of that name is "
                "declared in this world", pn, tn ? tn : "?");
            err = 1;
            continue;
        }
        if (!rl_body_has_assembly(m, en->target)) {
            /* The reason differs by class and the diagnostic says
             * which. A kinetic or directed-energy effector acts on the
             * area the target presents; a countermeasure acts on the
             * target's payloads, and a body with no assembly carries no
             * vehicle and therefore no payload at all. */
            kflc_diag_errorf(diag, en->line,
                kd->is_softkill
                    ? "engage %s at `%s`: `%s` declares no `assembly=`, "
                      "so it carries no vehicle and no payload for a "
                      "countermeasure to reach"
                    : "engage %s at `%s`: `%s` declares no `assembly=`, "
                      "so it presents no geometry for an effector to "
                      "act on",
                pn, tn, tn);
            err = 1;
            continue;
        }
        if (en->target == m->payloads[en->payload].body) {
            kflc_diag_errorf(diag, en->line,
                "engage %s at `%s`: `%s` is carried by `%s` itself, and "
                "an effector does not engage its own platform", pn, tn,
                pn, tn);
            err = 1;
            continue;
        }
    }

    /* Every effector payload's result is published or it is not
     * computed at all. A payload engaged with no `observe effect` on it
     * still changes the world, so the engagement is not a decoration;
     * but the channels the design fixes would be written where nothing
     * reads them, and a reader who wrote the statement expecting to see
     * a result would see nothing. It is not an error, and nothing here
     * refuses it: the effect is the point and the channels are how it
     * is watched. */

    /* The silhouette of every body a defense form takes an area of:
     * every detection target, and every body engaged. Both use the same
     * projected-area helper, since a laser spot and a detection
     * signature are the same silhouette along the same line of sight,
     * and an engagement's hit test is that silhouette's own radius.
     * Bodies are walked in declaration order so the emitted table's
     * order is the program's own and not an accident of which statement
     * was written first. */
    for (int b = 0; b < m->n_bodies; b++) {
        int wanted = 0;
        const char *why = "observe detect ... of";
        for (int i = 0; i < m->n_observes; i++) {
            if (rl_observe_form(m->observes[i]) != RL_OBS_DET) continue;
            if (m->obs_target[i] == b) wanted = 1;
        }
        for (int e = 0; e < m->n_engages; e++) {
            const RlEngage *en = &m->engages[e];
            if (en->payload < 0 || m->payloads[en->payload].kind < 0) {
                continue;
            }
            const RlPayKindDesc *ekd =
                &RL_PAY_KIND_[m->payloads[en->payload].kind];
            /* A countermeasure acts on the victim's payloads and takes
             * no area of the victim at all. What the jammer does take
             * an area of is its own host: the jamming-to-signal ratio
             * divides by the radar cross-section of the craft the
             * jammer is protecting, which is the silhouette that craft
             * presents to the victim. */
            int host = m->payloads[en->payload].body;
            if (ekd->is_softkill) {
                if (en->payload >= 0 &&
                    m->payloads[en->payload].kind == RL_PAY_JAMMER &&
                    host == b) {
                    if (!wanted) why = "engage of a jammer carried by";
                    wanted = 1;
                }
                continue;
            }
            if (en->target != b) continue;
            if (!wanted) why = "engage ... at";
            wanted = 1;
        }
        if (!wanted) continue;
        int have = 0;
        for (int c = 0; c < m->n_colliders; c++) {
            if (m->colliders[c].body == b) have++;
        }
        if (have == 0) {
            /* An assembly may declare components and no collider, and
             * such a body has no silhouette. Without this the program
             * passes the check and then fails to build, against a
             * generated source, naming neither the body nor what is
             * missing. */
            const KflcAttr *a = rl_attr(m->bodies[b].body, "assembly");
            char path[KFLC_ASM_PATH_MAX];
            const char *raw = a ? rl_pay_attr_text(a) : NULL;
            if (!raw || kflc_assembly_unquote(raw, path, sizeof path)) {
                snprintf(path, sizeof path, "%s", raw ? raw : "?");
            }
            kflc_diag_errorf(diag, m->bodies[b].body->line,
                "%s `%s`: the assembly `%s` declares no `collider`, so "
                "`%s` presents no area along a line of sight and its "
                "silhouette would be zero at every aspect; a body whose "
                "silhouette a defense payload takes needs at least one "
                "collision primitive", why, m->bodies[b].body->name,
                path, m->bodies[b].body->name);
            err = 1;
            continue;
        }
        for (int c = 0; c < m->n_colliders; c++) {
            if (m->colliders[c].body != b) continue;
            if (m->n_sig >= RL_MAX_SIG) {
                kflc_diag_errorf(diag, m->bodies[b].body->line,
                    "more than %d signature primitives in this program",
                    RL_MAX_SIG);
                return 1;
            }
            RlSigPrim *sp = &m->sig[m->n_sig++];
            sp->body = b;
            sp->kind = m->colliders[c].kind;
            memcpy(sp->axis, m->colliders[c].axis, sizeof sp->axis);
            memcpy(sp->half, m->colliders[c].half, sizeof sp->half);
        }
    }
    return err;
}

/* Resolve `path` against the directory of the source file that named
 * it, which is where every other asset a program names is looked for.
 * An absolute path is taken as it stands. */

/* Assembled rather than formatted, so a path that will not fit is
 * reported as one instead of being quietly cut and then failing to
 * open under a name nobody wrote. Returns nonzero when it will not
 * fit. */

static int rl_ref_resolve_(const char *path, const char *src_path,
                           char *out, size_t out_sz)
{
    const char *slash = src_path ? strrchr(src_path, '/') : NULL;
    size_t dir, pl;

    if (!path) return 1;
    pl = strlen(path);
    if (path[0] == '/' || !slash) {
        if (pl + 1u > out_sz) return 1;
        memcpy(out, path, pl + 1u);
        return 0;
    }
    dir = (size_t)(slash - src_path);
    if (dir + pl + 2u > out_sz) return 1;
    memcpy(out, src_path, dir);
    out[dir] = '/';
    memcpy(out + dir + 1u, path, pl + 1u);
    return 0;
}

/* Read every `reference=` a body declares.
 *
 * The plan is read here rather than by the running artifact, for the
 * reason an assembly is: a running simulation opens no asset file, and
 * a plan whose bytes could change between the build and the run would
 * make the compiled program's identity a claim about a file nobody
 * checked. What the emitter carries away is the decoded knots, the
 * file's own bytes for the record, and its digest.
 *
 * Four things are refused where they are written, because each would
 * otherwise publish eight channels that could only ever be wrong: a
 * plan the reader will not have, a plan with nothing in it, a frame
 * naming a body this world does not declare, and a local-vertical
 * frame on a body that orbits nothing. */

int rl_collect_references(RlModel *m, KflcArena *arena,
                                  KflcDiag *diag)
{
    int err = 0;

    for (int i = 0; i < RL_MAX_OBSERVES; i++) m->obs_ref[i] = -1;

    for (int b = 0; b < m->n_bodies; b++) {
        const KflcNode *body = m->bodies[b].body;
        const KflcAttr *a = rl_body_attr(body, "reference");
        const char *raw = a ? rl_pay_attr_text(a) : NULL;
        char rel[KFLC_ASM_PATH_MAX], full[KFLC_ASM_PATH_MAX];
        K26RlRef *ref = NULL;
        K26RlRefInfo info;
        K26RlRefStatus st;
        RlReference *r;
        const char *frame;
        uint64_t blen = 0;
        const uint8_t *bytes;
        double *knots;

        if (!a) continue;
        if (m->n_refs >= RL_MAX_REF) {
            kflc_diag_errorf(diag, body->line,
                "more than %d bodies carry a `reference=`", RL_MAX_REF);
            return 1;
        }
        if (!raw || kflc_assembly_unquote(raw, rel, sizeof rel)) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: `reference=` takes a quoted path to a "
                "`%s` file", body->name ? body->name : "?",
                K26RL_REF_SUFFIX);
            err = 1;
            continue;
        }
        if (rl_ref_resolve_(rel, diag->path, full, sizeof full)) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: the reference path `%s` does not fit "
                "in %d bytes once resolved against this source file",
                body->name ? body->name : "?", rel,
                (int)sizeof full);
            err = 1;
            continue;
        }
        st = k26rl_ref_open(full, &ref);
        if (st != K26RL_REF_OK) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: reference `%s` was not read: %s",
                body->name ? body->name : "?", full,
                k26rl_ref_status_str(st));
            err = 1;
            continue;
        }
        (void)k26rl_ref_info(ref, &info);
        if (info.present_count == 0) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: reference `%s` asks for nothing, since "
                "every knot it carries has a tolerance of zero and is "
                "therefore absent", body->name ? body->name : "?", full);
            k26rl_ref_close(ref);
            err = 1;
            continue;
        }
        frame = k26rl_ref_frame_name(ref, NULL);
        r = &m->refs[m->n_refs];
        memset(r, 0, sizeof *r);
        r->body       = b;
        r->frame_body = rl_body_index_of(m, frame);
        r->frame_kind = info.frame_kind;
        r->epoch      = info.epoch;
        r->n_knots    = (int)info.present_count;
        memcpy(r->digest, info.digest, K26RL_SHA256_BYTES);
        if (r->frame_body < 0) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: reference `%s` is written in the frame "
                "of `%s`, and no astro_body of that name is declared in "
                "this world", body->name ? body->name : "?", full,
                frame ? frame : "?");
            k26rl_ref_close(ref);
            err = 1;
            continue;
        }
        if (r->frame_body == b) {
            /* Every published component would then be the knot itself,
             * whatever the craft did, which is an absolute wearing the
             * name of an error. */
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: reference `%s` is written in the frame "
                "of `%s` itself, so its published components would be "
                "the plan's own numbers rather than the craft's error "
                "against them", body->name ? body->name : "?", full,
                body->name ? body->name : "?");
            k26rl_ref_close(ref);
            err = 1;
            continue;
        }
        if (r->frame_kind == K26RL_REF_FRAME_LVLH &&
            !rl_body_attr(m->bodies[r->frame_body].body, "parent")) {
            kflc_diag_errorf(diag, body->line,
                "astro_body `%s`: reference `%s` is written in the "
                "local-vertical local-horizontal frame of `%s`, and `%s` "
                "declares no `parent=`, so the body it orbits is unknown "
                "and that frame cannot be built",
                body->name ? body->name : "?", full, frame, frame);
            k26rl_ref_close(ref);
            err = 1;
            continue;
        }

        bytes = k26rl_ref_bytes(ref, &blen);
        r->len   = (uint32_t)blen;
        r->bytes = (const uint8_t *)kflc_arena_alloc(arena, (size_t)blen);
        memcpy((void *)r->bytes, bytes, (size_t)blen);
        knots = (double *)kflc_arena_alloc(
            arena, sizeof(double) * (size_t)r->n_knots * 8u);
        for (int k = 0; k < r->n_knots; k++) {
            K26RlRefKnot kn;

            (void)k26rl_ref_knot(ref, (uint32_t)k, &kn);
            knots[k * 8 + 0] = kn.t;
            for (int q = 0; q < 3; q++) {
                knots[k * 8 + 1 + q] = kn.r[q];
                knots[k * 8 + 4 + q] = kn.v[q];
            }
            knots[k * 8 + 7] = kn.tolerance;
        }
        r->knots = knots;
        k26rl_ref_close(ref);
        m->n_refs++;
    }

    /* Bind each reference observe to the plan its body carries. A
     * craft with no plan has nothing to publish, so it is refused
     * where the statement is written. */
    for (int i = 0; i < m->n_observes; i++) {
        const KflcNode *s = m->observes[i];
        int tgt;

        if (rl_observe_form(s) != RL_OBS_REF) continue;
        tgt = rl_body_index_of(m, s->name);
        if (tgt < 0) {
            kflc_diag_errorf(diag, s->line,
                "observe reference of `%s`: no astro_body of that name is "
                "declared in this world", s->name ? s->name : "?");
            err = 1;
            continue;
        }
        for (int k = 0; k < m->n_refs; k++) {
            if (m->refs[k].body == tgt) m->obs_ref[i] = k;
        }
        if (m->obs_ref[i] < 0) {
            kflc_diag_errorf(diag, s->line,
                "observe reference of `%s`: `%s` declares no "
                "`reference=`, so it is flying no plan and there is "
                "nothing to publish", s->name, s->name);
            err = 1;
        }
    }
    return err;
}

/* Strip a pair of quotes, as an assembly path is stripped, so the two
 * spellings mean the same thing. */

static const char *rl_unquote_(KflcArena *arena, const char *raw)
{
    size_t n = raw ? strlen(raw) : 0;
    char *out;

    if (!raw) return "";
    if (n >= 2 && raw[0] == '"' && raw[n - 1] == '"') {
        out = (char *)kflc_arena_alloc(arena, n - 1);
        memcpy(out, raw + 1, n - 2);
        out[n - 2] = '\0';
        return out;
    }
    return raw;
}

/* Resolve every `plan` block: the frame it writes its knots in, the
 * slot count, and where the block's own action channels sit in the
 * environment's action vector.
 *
 * The refusals are the reference's own, taken here instead of at load
 * because a plan this world emits has no file to read yet: a frame
 * naming a body the world does not declare, and a local-vertical
 * frame on a body that orbits nothing. */

int rl_collect_plans(RlModel *m, KflcArena *arena, KflcDiag *diag)
{
    int err = 0;

    for (int p = 0; p < m->n_plans; p++) {
        RlPlanOut *po = &m->plans[p];
        const KflcNode *s = po->node;
        const char *frame = NULL, *kind = NULL, *slots = NULL;
        const char *epoch = NULL;
        char *end = NULL;

        po->file       = "";
        po->provenance = "";
        po->epoch      = 0.0;
        for (const KflcAttr *a = s->attrs; a; a = a->next) {
            if (!a->name || a->value.kind != KFLV_IDENT) continue;
            if (strcmp(a->name, "frame") == 0) frame = a->value.u.s;
            else if (strcmp(a->name, "kind") == 0) kind = a->value.u.s;
            else if (strcmp(a->name, "slots") == 0) slots = a->value.u.s;
            else if (strcmp(a->name, "epoch") == 0) epoch = a->value.u.s;
            else if (strcmp(a->name, "file") == 0)
                po->file = rl_unquote_(arena, a->value.u.s);
            else if (strcmp(a->name, "provenance") == 0)
                po->provenance = rl_unquote_(arena, a->value.u.s);
        }
        if (po->file[0] == '\0') {
            kflc_diag_errorf(diag, s->line,
                "plan `%s`: `file` is required and names the path a plan "
                "is written to", po->name ? po->name : "?");
            err = 1;
            continue;
        }
        if (!frame || !kind) {
            kflc_diag_errorf(diag, s->line,
                "plan `%s`: `frame` is required and takes a body name and "
                "either `lvlh` or `inertial`", po->name ? po->name : "?");
            err = 1;
            continue;
        }
        if (strcmp(kind, "lvlh") == 0) {
            po->frame_kind = K26RL_REF_FRAME_LVLH;
        } else if (strcmp(kind, "inertial") == 0) {
            po->frame_kind = K26RL_REF_FRAME_INERTIAL;
        } else {
            kflc_diag_errorf(diag, s->line,
                "plan `%s`: frame kind `%s` is not one this format "
                "carries; it is `lvlh` or `inertial`",
                po->name ? po->name : "?", kind);
            err = 1;
            continue;
        }
        po->frame_body = rl_body_index_of(m, frame);
        if (po->frame_body < 0) {
            kflc_diag_errorf(diag, s->line,
                "plan `%s`: the frame names `%s`, and no astro_body of "
                "that name is declared in this world",
                po->name ? po->name : "?", frame);
            err = 1;
            continue;
        }
        if (po->frame_kind == K26RL_REF_FRAME_LVLH &&
            !rl_body_attr(m->bodies[po->frame_body].body, "parent")) {
            kflc_diag_errorf(diag, s->line,
                "plan `%s`: the local-vertical local-horizontal frame of "
                "`%s` cannot be built, because `%s` declares no "
                "`parent=` and the body it orbits is unknown",
                po->name ? po->name : "?", frame, frame);
            err = 1;
            continue;
        }
        po->slots = slots ? (int)strtol(slots, &end, 10) : 0;
        if (po->slots < 1 || po->slots > (int)K26RL_REF_MAX_KNOTS) {
            kflc_diag_errorf(diag, s->line,
                "plan `%s`: `slots` takes a whole number of knot slots "
                "from 1 to %u", po->name ? po->name : "?",
                (unsigned)K26RL_REF_MAX_KNOTS);
            err = 1;
            continue;
        }
        if (epoch) {
            po->epoch = strtod(epoch, &end);
            if (!end || *end != '\0') {
                kflc_diag_errorf(diag, s->line,
                    "plan `%s`: `epoch` takes a number of seconds",
                    po->name ? po->name : "?");
                err = 1;
                continue;
            }
        }

        /* The first of the block's own action channels, found by the
         * name the block gave it. Its index in the action vector is
         * what the episode-end write reads from, and the eight
         * channels of a slot are contiguous because the parser
         * appended them in that order. */
        {
            char first[160];
            int found = -1;

            snprintf(first, sizeof first, "%s_k0_t",
                     po->name ? po->name : "?");
            for (int a = 0; a < m->n_actions; a++) {
                const char *an = m->actions[a] ? m->actions[a]->name : NULL;

                if (an && strcmp(an, first) == 0) found = a;
            }
            if (found < 0) {
                kflc_diag_errorf(diag, s->line,
                    "plan `%s`: its action channels were not declared",
                    po->name ? po->name : "?");
                err = 1;
                continue;
            }
            po->act_first = found;
        }
    }
    return err;
}
