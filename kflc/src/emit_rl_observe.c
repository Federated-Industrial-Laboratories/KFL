/* emit_rl_observe.c - the observation path: geometric, defense and degraded
 * observes. */
#include "emit_rl_internal.h"

static void rl_emit_detect_degrade_(FILE *out, const RlModel *m,
                                    int p, int tgt, int kind);
static void rl_emit_observe_defense_(FILE *out, const RlModel *m,
                                     int i, int off);

/* Observation recompute: the as-bound observes in source order, four
 * channels each. The range channel is the magnitude of the relative
 * position vector between the corrected target position and the
 * observer, in metres. */

/* The three defense observation forms.
 *
 * Detection is closed form over the two bodies' current state: no
 * integration, no draw, no iteration, and no light-time correction.
 * The corrected view is the track form's, which is the division of
 * labour the two libraries already have. Both call the tier's
 * evaluators with a null generator, so every imperfection on these
 * channels arrives through the sensor layer, which has the
 * draw-coordinate discipline replay rests on.
 *
 * The signature is aspect dependent because the tier's models are:
 * the area the target presents is computed along the line of sight in
 * the target's own frame, so a craft that turns changes what its
 * observer sees. The published aspect channel is the cosine between
 * the line of sight and the target's own first body axis, which is the
 * axis that silhouette is taken against.
 *
 * The effector form reads rather than computes: the engagement itself
 * ran in the step body, before the world advanced, and left its
 * channels in the environment's engagement block. Reading them here
 * puts an effector's result in the same vector, at the same step
 * boundary, as every other observation. */

/* What a countermeasure engaged earlier in this step did to this
 * detection payload's view of this target.
 *
 * Nothing is emitted unless some engagement in this program can reach
 * the pair, which is a compile-time question: a countermeasure degrades
 * its victim's view of the craft that carries it, and both bodies are
 * declarations. So a program with no countermeasure, or one whose
 * countermeasures are aimed elsewhere, publishes the channels it
 * published before this surface existed, computed by the same
 * expressions.
 *
 * Three effects, applied in a fixed order and the threshold taken once
 * at the end:
 *
 *   the counter-detection signal, which raises the statistic rather
 *   than lowering it. A jammer's thermalised transmit power is a
 *   separate signal in the same band as the target's own emission, and
 *   the observer detects on whichever is stronger; the library's
 *   counter-detection range is the range at which that signal sits at
 *   this observer's threshold, and its own derivation is the
 *   shot-noise regime in which the statistic falls as one over range,
 *   which is what the scaling below is;
 *
 *   the jamming, which raises the noise. The library returns the
 *   jamming-to-signal ratio at the receiver, and a statistic of signal
 *   over noise becomes signal over noise plus jamming, which in terms
 *   of that ratio is the expression below and nothing else;
 *
 *   the decoy, which lowers confidence rather than signal. The library
 *   returns the probability the observer's discriminators correctly
 *   identify the decoy, and the derate is the complement: an observer
 *   that always tells the decoy from the craft loses nothing, one that
 *   never does loses all of it. That mapping from a discrimination
 *   probability onto a detection statistic is this layer's, not the
 *   library's, which is why it is written down here. */

static void rl_emit_detect_degrade_(FILE *out, const RlModel *m,
                                    int p, int tgt, int kind)
{
    int jam = rl_softkill_reaches(m, p, tgt, 1);
    int dec = rl_softkill_reaches(m, p, tgt, 0);
    if (!jam && !dec) return;

    int thr = kind == RL_PAY_DETECT_RADAR ? 8 : 5;
    int any = 0;

    if (jam && kind == RL_PAY_DETECT_IR) {
        fprintf(out,
        "                {\n"
        "                    double _kfl_cr =\n"
        "                        eng->deg[%d * KFLRL_DEG_STRIDE + %d].ctr;\n"
        "                    if (_kfl_cr > 0.0) {\n"
        "                        double _kfl_cs = _kfl_pp[%d] * _kfl_cr\n"
        "                                       / _kfl_rng;\n"
        "                        if (_kfl_cs > _kfl_snr) _kfl_snr = _kfl_cs;\n"
        "                    }\n"
        "                }\n", p, tgt, thr);
        any = 1;
    }
    if (jam && kind == RL_PAY_DETECT_RADAR) {
        fprintf(out,
        "                {\n"
        "                    double _kfl_j =\n"
        "                        eng->deg[%d * KFLRL_DEG_STRIDE + %d].js;\n"
        "                    if (_kfl_j > 0.0 && _kfl_snr > 0.0) {\n"
        "                        _kfl_snr = _kfl_snr\n"
        "                                 / (1.0 + _kfl_j * _kfl_snr);\n"
        "                    }\n"
        "                }\n", p, tgt);
        any = 1;
    }
    if (dec) {
        fprintf(out,
        "                {\n"
        "                    double _kfl_dc =\n"
        "                        eng->deg[%d * KFLRL_DEG_STRIDE + %d].dec;\n"
        "                    if (_kfl_dc > 0.0) {\n"
        "                        _kfl_snr *= (1.0 - _kfl_dc);\n"
        "                    }\n"
        "                }\n", p, tgt);
        any = 1;
    }
    if (!any) return;
    /* The flag rests on the statistic, so it is taken again after the
     * statistic has moved. Leaving the library's own decision in place
     * would publish a detection whose continuous quantity is below the
     * threshold it was taken at. */
    fprintf(out,
        "                _kfl_det = (_kfl_snr >= _kfl_pp[%d])\n"
        "                         ? 1.0 : 0.0;\n", thr);
}

static void rl_emit_observe_defense_(FILE *out, const RlModel *m,
                                     int i, int off)
{
    const KflcNode *s = m->observes[i];
    int p   = m->obs_payload[i];
    int tgt = m->obs_target[i];
    const RlPayload *py = &m->payloads[p];

    if (rl_observe_form(s) == RL_OBS_EFF) {
        /* The block is read without a null test. This form exists only
         * in a program that declares an effector payload, and such a
         * program's handle always carries a block: a test here would be
         * a branch nothing can take, which is a branch no gate can
         * measure. What makes an unengaged step read zero is the clear
         * at the top of the step, not a fallback here. */
        int w = rl_observe_base_width(s);
        fprintf(out, "    {\n"
                     "        const double *_kfl_ec =\n"
                     "            eng->ch + %d * KFLRL_EFF_STRIDE;\n", p);
        for (int c = 0; c < w; c++) {
            fprintf(out, "        out_v[%d] = _kfl_ec[%d];\n", off + c, c);
        }
        fputs("    }\n", out);
        return;
    }

    if (rl_observe_form(s) == RL_OBS_TRK) {
        const char *bad = NULL;
        int mod = rl_track_modality(s, &bad);
        fprintf(out,
            "    {\n"
            "        double _kfl_val = 0.0;\n"
            "        double _kfl_px = 0.0, _kfl_py = 0.0, _kfl_pz = 0.0;\n"
            "        double _kfl_vx = 0.0, _kfl_vy = 0.0, _kfl_vz = 0.0;\n"
            "        double _kfl_rr = 0.0, _kfl_ag = 0.0;\n"
            "        K26AstroInfostate *_kfl_is = pay\n"
            "            ? (K26AstroInfostate *)pay[%d] : NULL;\n"
            "        K26AstroVehicle *_kfl_tv = veh ? veh[%d] : NULL;\n"
            "        if (_kfl_is && _kfl_tv) {\n"
            "            K26AstroInfostateObservation _kfl_o =\n"
            "                k26astro_infostate_observe(_kfl_is, _kfl_tv,\n"
            "                    kflrl_info_epoch_(t_day, t_info),\n"
            "                    (K26AstroInfostateModality)%d);\n"
            /* The age test is what keeps one episode out of the next.
             * The ring carries the previous episode's samples, since
             * the library offers no way to empty it, so an
             * observation whose retarded time precedes this episode's
             * epoch was interpolated across the reset. It is
             * published as unavailable, which is what it is. */
            /* The retarded epoch must lie in this episode. The ring
             * still holds the previous episode's samples, so an
             * observation solved across the reset comes back valid
             * with that episode's state blended in; the day index the
             * epochs carry separates the two exactly, where comparing
             * the reported age against the elapsed time leaves a band
             * of a few nanoseconds in which the iterate crossed the
             * epoch and the reported age did not. */
            "            if (_kfl_o.valid &&\n"
            "                _kfl_o.t_retarded.days_since_J2000 >= t_day) {\n"
            "                _kfl_val = 1.0;\n"
            "                _kfl_px = _kfl_o.position.x;\n"
            "                _kfl_py = _kfl_o.position.y;\n"
            "                _kfl_pz = _kfl_o.position.z;\n"
            "                _kfl_vx = _kfl_o.velocity.x;\n"
            "                _kfl_vy = _kfl_o.velocity.y;\n"
            "                _kfl_vz = _kfl_o.velocity.z;\n"
            "                _kfl_rr = _kfl_o.range_m;\n"
            "                _kfl_ag = _kfl_o.age_s;\n"
            "            }\n"
            "        }\n"
            "        out_v[%d] = _kfl_val;\n"
            "        out_v[%d] = _kfl_px;\n"
            "        out_v[%d] = _kfl_py;\n"
            "        out_v[%d] = _kfl_pz;\n"
            "        out_v[%d] = _kfl_vx;\n"
            "        out_v[%d] = _kfl_vy;\n"
            "        out_v[%d] = _kfl_vz;\n"
            "        out_v[%d] = _kfl_rr;\n"
            "        out_v[%d] = _kfl_ag;\n"
            "    }\n",
            p, rl_veh_slot_of(m, tgt), mod < 0 ? 0 : mod,
            off, off + 1, off + 2, off + 3, off + 4, off + 5, off + 6,
            off + 7, off + 8);
        return;
    }

    fprintf(out,
        "    {\n"
        "        const K26AstroBody *_kfl_ob = k26astro_world_body_at(\n"
        "            world, kflrl_body_idx_[%d]);\n"
        "        const K26AstroBody *_kfl_tb = k26astro_world_body_at(\n"
        "            world, kflrl_body_idx_[%d]);\n"
        "        double _kfl_det = 0.0, _kfl_snr = 0.0, _kfl_rng = 0.0;\n"
        "        double _kfl_ux = 0.0, _kfl_uy = 0.0, _kfl_uz = 0.0;\n"
        "        double _kfl_asp = 0.0;\n"
        "        if (_kfl_ob && _kfl_tb) {\n"
        "            K26V3 _kfl_d = k26astro_pos_sub(&_kfl_tb->pos,\n"
        "                                            &_kfl_ob->pos);\n"
        "            _kfl_rng = k26m3d_v3_len(_kfl_d);\n"
        "            if (_kfl_rng > 0.0) {\n"
        "                _kfl_ux = _kfl_d.x / _kfl_rng;\n"
        "                _kfl_uy = _kfl_d.y / _kfl_rng;\n"
        "                _kfl_uz = _kfl_d.z / _kfl_rng;\n"
        "            }\n"
        "            K26V3 _kfl_look = k26m3d_quat_rotate_v3(\n"
        "                k26m3d_quat_conj(_kfl_tb->attitude),\n"
        "                k26m3d_v3(_kfl_ux, _kfl_uy, _kfl_uz));\n"
        /* The aspect the silhouette is actually taken at: the line of
         * sight resolved in the target's own frame, against the first
         * body axis, which the assembly format runs along the craft.
         * It is the first component of the look vector by
         * construction. A velocity-referenced cosine would name the
         * same angle for a craft flying nose forward and would sit
         * still while the signature moved for one that is not. */
        "            _kfl_asp = _kfl_look.x;\n"
        "            double _kfl_area = kflrl_sig_area_(%d, _kfl_look);\n"
        "            const double *_kfl_pp = payp\n"
        "                ? payp + %d * KFLRL_PAY_NPARAM : NULL;\n"
        "            if (_kfl_pp && _kfl_rng > 0.0) {\n",
        py->body, tgt, tgt, p);

    if (py->kind == RL_PAY_DETECT_IR) {
        fputs(
        "                double _kfl_pw =\n"
        "                    k26astro_signature_ir_planck_inband(\n"
        "                        _kfl_area, _kfl_pp[7], _kfl_pp[6],\n"
        "                        _kfl_pp[2] * 1.0e-6,\n"
        "                        _kfl_pp[3] * 1.0e-6, 64);\n"
        "                K26AstroDetectIrEvent _kfl_ev =\n"
        "                    k26astro_detect_ir_passive_with_optics(\n"
        "                        _kfl_pw, _kfl_pp[6], _kfl_rng,\n"
        "                        _kfl_pp[0], _kfl_pp[1], _kfl_pp[2],\n"
        "                        _kfl_pp[3], _kfl_pp[4], _kfl_pp[5],\n"
        "                        _kfl_pp[8], _kfl_pp[9], NULL);\n"
        "                _kfl_snr = _kfl_ev.snr;\n"
        "                _kfl_det = _kfl_ev.detected ? 1.0 : 0.0;\n", out);
    } else if (py->kind == RL_PAY_DETECT_RADAR) {
        /* The target is taken as a flat plate of its own projected
         * area facing the observer, which is the geometric-optics
         * form the signature library implements and the one its
         * facet routine returns for a facet the observer looks
         * squarely at. The aspect dependence is in the area. */
        fputs(
        "                double _kfl_lam = _kfl_pp[3] > 0.0\n"
        "                    ? (K26A_C / _kfl_pp[3]) : 0.0;\n"
        "                K26V3 _kfl_nrm = k26m3d_v3(-_kfl_look.x,\n"
        "                                           -_kfl_look.y,\n"
        "                                           -_kfl_look.z);\n"
        "                double _kfl_rcs =\n"
        "                    k26astro_signature_rcs_monostatic(\n"
        "                        1, &_kfl_nrm, &_kfl_area, _kfl_look,\n"
        "                        _kfl_lam);\n", out);
        if (py->attr[RL_PAY_RADAR_CHAFF_N]) {
            /* A chaff cloud around the target returns as well as the
             * target does, and the strips are uncorrelated, so the two
             * cross-sections add: the cloud's mean is the strip count
             * times one strip's, which is what the library's
             * deterministic mean returns. Its sampling entry point,
             * which takes a generator, is not called: an imperfection
             * on a detection channel arrives through the declared
             * sensor layer, at coordinates a replay reproduces.
             *
             * The term is emitted only where a strip count was
             * declared, so a program that declares no cloud computes
             * the cross-section it computed before this key existed. */
            fprintf(out,
        "                _kfl_rcs += k26astro_chaff_mean_rcs(\n"
        "                    kflrl_chaff_strips_(_kfl_pp[%d]),\n"
        "                    _kfl_pp[%d]);\n",
                RL_PAY_RADAR_CHAFF_N, RL_PAY_RADAR_CHAFF_SIG);
        }
        fputs(
        "                K26AstroDetectRadarEvent _kfl_ev =\n"
        "                    k26astro_detect_radar_active(\n"
        "                        _kfl_pp[0], _kfl_pp[1], _kfl_pp[2],\n"
        "                        _kfl_pp[3], _kfl_rcs, _kfl_rng,\n"
        "                        _kfl_pp[4], _kfl_pp[5], _kfl_pp[6],\n"
        "                        _kfl_pp[7], _kfl_pp[8], NULL);\n"
        "                _kfl_snr = _kfl_ev.snr;\n"
        "                _kfl_det = _kfl_ev.detected ? 1.0 : 0.0;\n", out);
    } else {
        /* The transmit gain is the diffraction-limited figure for the
         * declared aperture at the declared wavelength, which is the
         * relation the library's own header states, rather than a
         * separate declaration that could disagree with the aperture
         * beside it. The view-angle cosine is 1 because the area
         * already carries the projection. */
        fputs(
        "                double _kfl_lam = _kfl_pp[1] * 1.0e-9;\n"
        "                double _kfl_gt = 0.0;\n"
        "                if (_kfl_lam > 0.0 && _kfl_pp[2] > 0.0) {\n"
        "                    double _kfl_g = K26A_PI * _kfl_pp[2]\n"
        "                                  / _kfl_lam;\n"
        "                    _kfl_gt = 10.0 * log10(_kfl_g * _kfl_g);\n"
        "                }\n"
        "                K26AstroDetectLidarEvent _kfl_ev =\n"
        "                    k26astro_detect_lidar_active(\n"
        "                        _kfl_pp[0], _kfl_pp[1], _kfl_pp[2],\n"
        "                        _kfl_gt, _kfl_pp[6], _kfl_area, 1.0,\n"
        "                        _kfl_rng, _kfl_pp[3], _kfl_pp[4],\n"
        "                        _kfl_pp[5], NULL);\n"
        "                _kfl_snr = _kfl_ev.snr;\n"
        "                _kfl_det = _kfl_ev.detected ? 1.0 : 0.0;\n", out);
    }

    rl_emit_detect_degrade_(out, m, p, tgt, py->kind);

    fprintf(out,
        "            }\n"
        "        }\n"
        "        out_v[%d] = _kfl_det;\n"
        "        out_v[%d] = _kfl_snr;\n"
        "        out_v[%d] = _kfl_rng;\n"
        "        out_v[%d] = _kfl_ux;\n"
        "        out_v[%d] = _kfl_uy;\n"
        "        out_v[%d] = _kfl_uz;\n"
        "        out_v[%d] = _kfl_asp;\n"
        "    }\n",
        off, off + 1, off + 2, off + 3, off + 4, off + 5, off + 6);
}

int rl_emit_observe(FILE *out, const RlModel *m,
                            KflcDiag *diag)
{
    fputs("/* The observation vector for one environment. `ct` is that\n"
          " * environment's latched contact block, which a contact\n"
          " * observe reads: contact is a fact about the transition\n"
          " * just taken rather than about where a body is, so it is\n"
          " * passed in rather than read back out of the world.\n"
          " * `eng` is the environment's engagement block, read by an\n"
          " * effector observe for the same reason: an engagement is a\n"
          " * fact about the act the step took, not about where a body\n"
          " * is. */\n"
          "static void kflrl_observe_(K26AstroWorld *world, "
          "double *out_v,\n"
          "                           const KflrlContact *ct,\n"
          "                           const KflrlJoin *jn,\n"
          "                           const KflrlPortLatch *pl,\n"
          "                           void *const *pay,\n"
          "                           const double *payp,\n"
          "                           K26AstroVehicle *const *veh,\n"
          "                           const double *prop,\n"
          "                           const double *coms,\n"
          "                           int64_t t_day, double t_info,\n"
          "                           const KflrlEng *eng,\n"
          /* The simulated seconds this episode has advanced, which is
           * the clock a plan's knot times are read on. It is passed in
           * rather than read back out of the world for the reason the
           * contact block is: it is a fact about the transition just
           * taken, and the caller is the only thing that knows how
           * many have been taken. */
          "                           double t_ep)\n"
          "{\n"
          "    (void)world; (void)out_v; (void)ct; (void)jn; (void)pl;\n"
          "    (void)pay; (void)payp; (void)veh; (void)t_day; "
          "(void)t_info;\n"
          "    (void)eng; (void)prop; (void)coms; (void)t_ep;\n", out);
    for (int i = 0; i < m->n_observes; i++) {
        const KflcNode *s = m->observes[i];
        int off = rl_obs_offset(m->observes, i);
        if (rl_observe_form(s) == RL_OBS_DET ||
            rl_observe_form(s) == RL_OBS_TRK ||
            rl_observe_form(s) == RL_OBS_EFF) {
            rl_emit_observe_defense_(out, m, i, off);
            continue;
        }
        if (rl_observe_form(s) == RL_OBS_PORT) {
            /* The form names a port on a body, and the state it
             * publishes is that port's with respect to the one it
             * faces.
             *
             * Which port it faces is written in the statement when the
             * statement says so, with `against <port> of <body>`, and
             * resolved here when it does not: the one port declared on
             * any other body. A world with none has nothing to measure
             * against; a world with several leaves the pairing to an
             * accident of declaration order, so it is refused naming
             * the candidates rather than publishing channels about an
             * arbitrary choice. The rule is per statement and not per
             * world, so several enveloped bodies with a clause each
             * are ordinary. */
            const char *pname = NULL;
            const char *agst = NULL, *agst_body = NULL;
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                if (!a->name || a->value.kind != KFLV_IDENT) continue;
                if (strcmp(a->name, "port") == 0)  pname = a->value.u.s;
                if (strcmp(a->name, "against") == 0) agst = a->value.u.s;
                if (strcmp(a->name, "against_body") == 0) {
                    agst_body = a->value.u.s;
                }
            }
            int tgt = rl_body_index_of(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of `%s`: no astro_body of that name "
                    "is declared in this world", pname ? pname : "?",
                    s->name);
                return 1;
            }
            if (!rl_body_has_assembly(m, tgt)) {
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of `%s`: `%s` declares no "
                    "`assembly=`, so it carries no docking ports",
                    pname ? pname : "?", s->name, s->name);
                return 1;
            }
            int active = -1;
            for (int q = 0; q < m->n_ports; q++) {
                if (m->ports[q].body == tgt && pname &&
                    strcmp(m->ports[q].name, pname) == 0) {
                    active = q;
                }
            }
            if (active < 0) {
                char have[256];
                size_t used = 0;
                int    seen = 0;
                have[0] = '\0';
                for (int q = 0; q < m->n_ports; q++) {
                    if (m->ports[q].body != tgt) continue;
                    int wrote = snprintf(have + used, sizeof have - used,
                                         "%s`%s`", seen++ ? ", " : "",
                                         m->ports[q].name);
                    if (wrote < 0 || (size_t)wrote >= sizeof have - used) break;
                    used += (size_t)wrote;
                }
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of %s: `%s` declares no port of that "
                    "name carrying a capture envelope; it carries %s",
                    pname ? pname : "?", s->name, s->name,
                    seen ? have : "none");
                return 1;
            }
            int passive = -1;
            if (agst) {
                int abody = rl_body_index_of(m, agst_body);
                if (abody < 0) {
                    kflc_diag_errorf(diag, s->line,
                        "observe port %s of %s against %s of `%s`: no "
                        "astro_body of that name is declared in this "
                        "world", pname, s->name, agst, agst_body);
                    return 1;
                }
                if (abody == tgt) {
                    kflc_diag_errorf(diag, s->line,
                        "observe port %s of %s against %s of %s: a "
                        "pairing is between two bodies, and both of "
                        "these ports are on `%s`",
                        pname, s->name, agst, agst_body, s->name);
                    return 1;
                }
                for (int q = 0; q < m->n_ports; q++) {
                    if (m->ports[q].body == abody &&
                        strcmp(m->ports[q].name, agst) == 0) {
                        passive = q;
                    }
                }
                if (passive < 0) {
                    char have[256];
                    size_t used = 0;
                    int    seen = 0;
                    have[0] = '\0';
                    for (int q = 0; q < m->n_ports; q++) {
                        if (m->ports[q].body != abody) continue;
                        int wrote = snprintf(have + used, sizeof have - used,
                                             "%s`%s`", seen++ ? ", " : "",
                                             m->ports[q].name);
                        if (wrote < 0 ||
                            (size_t)wrote >= sizeof have - used) break;
                        used += (size_t)wrote;
                    }
                    kflc_diag_errorf(diag, s->line,
                        "observe port %s of %s against %s of %s: `%s` "
                        "declares no port of that name carrying a capture "
                        "envelope; it carries %s",
                        pname, s->name, agst, agst_body, agst_body,
                        seen ? have : "none");
                    return 1;
                }
            } else {
                char cand[512];
                size_t used = 0;
                int    others = 0;
                cand[0] = '\0';
                for (int q = 0; q < m->n_ports; q++) {
                    if (m->ports[q].body == tgt) continue;
                    passive = q;
                    int wrote = snprintf(cand + used, sizeof cand - used,
                                         "%s`%s of %s`", others++ ? ", " : "",
                                         m->ports[q].name,
                                         m->bodies[m->ports[q].body].body->name);
                    if (wrote < 0 || (size_t)wrote >= sizeof cand - used) break;
                    used += (size_t)wrote;
                }
                if (others != 1) {
                    kflc_diag_errorf(diag, s->line,
                        "observe port %s of %s: the state this form "
                        "publishes is against the port it faces, and %d "
                        "ports carrying a capture envelope are declared on "
                        "other bodies (%s); name one with `against <port> "
                        "of <body>`", pname, s->name, others,
                        others ? cand : "none");
                    return 1;
                }
            }
            if (strcmp(m->ports[active].env_name,
                       m->ports[passive].env_name) != 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe port %s of %s: `%s of %s` names capture "
                    "envelope `%s` and `%s of %s` names `%s`; the two "
                    "ports of a pairing are one interface and are judged "
                    "against one envelope",
                    pname, s->name, pname, s->name,
                    m->ports[active].env_name,
                    m->ports[passive].name,
                    m->bodies[m->ports[passive].body].body->name,
                    m->ports[passive].env_name);
                return 1;
            }
            int pbody = m->ports[passive].body;
            fprintf(out,
                "    {\n"
                "        const K26AstroBody *_kfl_pa = "
                "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                "        const K26AstroBody *_kfl_pp = "
                "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                "        K26AstroCollPortState _kfl_ps;\n"
                "        double _kfl_cap = 0.0;\n"
                "        memset(&_kfl_ps, 0, sizeof _kfl_ps);\n"
                /* A transition that ended at THIS pairing publishes
                 * the state the capture test was given; any other
                 * step, and any contact this port made with some
                 * other port, publishes the state as it stands, which
                 * is what an approach is flown on. */
                "        if (pl && pl[%d].hit != 0.0 && "
                "pl[%d].partner == %d) {\n"
                "            _kfl_cap             = pl[%d].captured;\n"
                "            _kfl_ps.axial        = pl[%d].axial;\n"
                "            _kfl_ps.lateral      = pl[%d].lateral;\n"
                "            _kfl_ps.pitchyaw     = pl[%d].pitchyaw;\n"
                "            _kfl_ps.roll         = pl[%d].roll;\n"
                "            _kfl_ps.v_axial      = pl[%d].v_axial;\n"
                "            _kfl_ps.v_lateral    = pl[%d].v_lateral;\n"
                "            _kfl_ps.v_pitchyaw   = pl[%d].v_pitchyaw;\n"
                "            _kfl_ps.v_roll       = pl[%d].v_roll;\n"
                "            _kfl_ps.v_lateral_cg = pl[%d].v_cg;\n"
                "        } else if (_kfl_pa && _kfl_pp) {\n"
                "            K26AstroCollBody _kfl_ba, _kfl_bp;\n"
                /* The centres of mass are read from the state
                 * rather than from the port table: a craft's port
                 * sits at a declared place in the body frame, and
                 * where that is with respect to the centre of mass
                 * moves as the craft burns. */
                "            kflrl_port_snap_(_kfl_pa,\n"
                "                k26m3d_v3(coms[%d], coms[%d], coms[%d]),\n"
                "                k26astro_pos_sub(&_kfl_pa->pos, "
                "&_kfl_pp->pos), &_kfl_ba);\n"
                "            kflrl_port_snap_(_kfl_pp,\n"
                "                k26m3d_v3(coms[%d], coms[%d], coms[%d]),\n"
                "                k26m3d_v3(0.0, 0.0, 0.0), &_kfl_bp);\n"
                "            (void)k26astro_coll_port_state(&_kfl_ba,\n"
                "                &kflrl_ports_[%d].geom, &_kfl_bp,\n"
                "                &kflrl_ports_[%d].geom, 0.0, &_kfl_ps);\n"
                "        }\n"
                "        out_v[%d] = _kfl_cap;\n"
                "        out_v[%d] = _kfl_ps.axial;\n"
                "        out_v[%d] = _kfl_ps.lateral;\n"
                "        out_v[%d] = _kfl_ps.pitchyaw;\n"
                "        out_v[%d] = _kfl_ps.roll;\n"
                "        out_v[%d] = _kfl_ps.v_axial;\n"
                "        out_v[%d] = _kfl_ps.v_lateral;\n"
                "        out_v[%d] = _kfl_ps.v_pitchyaw;\n"
                "        out_v[%d] = _kfl_ps.v_roll;\n",
                tgt, pbody,
                active, active, passive,
                active, active, active, active, active, active, active,
                active, active, active,
                3 * m->ports[active].veh, 3 * m->ports[active].veh + 1,
                3 * m->ports[active].veh + 2,
                3 * m->ports[passive].veh, 3 * m->ports[passive].veh + 1,
                3 * m->ports[passive].veh + 2,
                active, passive,
                off, off + 1, off + 2, off + 3, off + 4, off + 5,
                off + 6, off + 7, off + 8);
            /* The two the `full` mark adds. The combined rate is the
             * one condition an unmarked form leaves invisible: it
             * decides captures and a craft inside every published
             * limit can be refused by it with no channel that says
             * so. The join state is the standing fact the capture
             * pulse deliberately is not. */
            if (rl_observe_is_full(s)) {
                fprintf(out,
                    "        out_v[%d] = _kfl_ps.v_lateral_cg;\n"
                    "        out_v[%d] = kflrl_ports_joined_(jn, %d, %d);\n",
                    off + 9, off + 10, active, passive);
            }
            fputs("    }\n", out);
            continue;
        }
        if (rl_observe_form(s) == RL_OBS_CON) {
            /* The body must be one the pass can report on, which is a
             * body that binds an assembly: without one it has no
             * colliders and would publish three channels that could
             * never be anything but zero. */
            int tgt = rl_body_index_of(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe contact of `%s`: no astro_body of that name "
                    "is declared in this world", s->name);
                return 1;
            }
            int slot = -1, seen = 0;
            for (int b = 0; b < m->n_bodies; b++) {
                if (!rl_body_has_assembly(m, b)) continue;
                if (b == tgt) slot = seen;
                seen++;
            }
            if (slot < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe contact of `%s`: `%s` declares no "
                    "`assembly=`, so it carries no colliders and can "
                    "report no contact", s->name, s->name);
                return 1;
            }
            fprintf(out,
                "    if (ct) {\n"
                "        out_v[%d] = ct[%d].hit;\n"
                "        out_v[%d] = ct[%d].fraction;\n"
                "        out_v[%d] = ct[%d].speed;\n"
                "    } else {\n"
                "        out_v[%d] = 0.0; out_v[%d] = 0.0; "
                "out_v[%d] = 0.0;\n"
                "    }\n",
                off, slot, off + 1, slot, off + 2, slot,
                off, off + 1, off + 2);
            continue;
        }
        if (rl_observe_form(s) == RL_OBS_REL) {
            /* Both bodies must be declared here, and the chief must
             * name a parent, because the chief's frame is built from
             * its state relative to the body it orbits and there is no
             * other way to know which body that is. A chief without a
             * parent is a declaration that cannot produce the frame at
             * all, so it is refused where it is written rather than
             * publishing six channels that could only ever be zero. */
            const char *chief = NULL;
            for (const KflcAttr *a = s->attrs; a; a = a->next) {
                if (a->name && strcmp(a->name, "observer") == 0 &&
                    a->value.kind == KFLV_IDENT) {
                    chief = a->value.u.s;
                }
            }
            int tgt = rl_body_index_of(m, s->name);
            int chf = chief ? rl_body_index_of(m, chief) : -1;
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative `%s`: no astro_body of that name "
                    "is declared in this world", s->name);
                return 1;
            }
            if (chf < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative %s from `%s`: no astro_body of "
                    "that name is declared in this world", s->name,
                    chief ? chief : "?");
                return 1;
            }
            if (chf == tgt) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative %s from `%s`: a body has no "
                    "relative state with respect to itself", s->name,
                    chief);
                return 1;
            }
            if (!rl_body_attr(m->bodies[chf].body, "parent")) {
                kflc_diag_errorf(diag, s->line,
                    "observe relative %s from `%s`: `%s` declares no "
                    "`parent=`, so the body it orbits is unknown and "
                    "its local-vertical local-horizontal frame cannot "
                    "be built", s->name, chief, chief);
                return 1;
            }
            fprintf(out,
                "    {\n"
                "        K26AstroBody *_kfl_cb = k26astro_world_body_at("
                "world, kflrl_body_idx_[%d]);\n"
                "        K26AstroBody *_kfl_tb = k26astro_world_body_at("
                "world, kflrl_body_idx_[%d]);\n"
                "        K26AstroBody *_kfl_pb = (_kfl_cb && "
                "_kfl_cb->parent_body_idx >= 0)\n"
                "            ? k26astro_world_body_at(world, "
                "_kfl_cb->parent_body_idx) : NULL;\n"
                "        K26AstroProxRel _kfl_rr;\n"
                "        _kfl_rr.r = k26m3d_v3(0.0, 0.0, 0.0);\n"
                "        _kfl_rr.v = k26m3d_v3(0.0, 0.0, 0.0);\n"
                /* A state with no frame publishes zeros: the chief
                 * sitting at its parent's centre, or moving straight
                 * at it, names no direction of motion. That is a
                 * configuration and not a declaration, so it cannot be
                 * refused at compile time and is reported as the
                 * absence of a measurement. */
                "        if (_kfl_cb && _kfl_tb && _kfl_pb) {\n"
                "            K26AstroProxFrame _kfl_f;\n"
                "            if (k26astro_prox_frame(&_kfl_pb->pos, "
                "_kfl_pb->vel,\n"
                "                                    &_kfl_cb->pos, "
                "_kfl_cb->vel,\n"
                "                                    &_kfl_f) == "
                "K26ASTRO_PROX_OK) {\n"
                "                (void)k26astro_prox_relative(&_kfl_f,\n"
                "                    &_kfl_cb->pos, _kfl_cb->vel,\n"
                "                    &_kfl_tb->pos, _kfl_tb->vel, "
                "&_kfl_rr);\n"
                "            }\n"
                "        }\n"
                "        out_v[%d] = _kfl_rr.r.x;\n"
                "        out_v[%d] = _kfl_rr.r.y;\n"
                "        out_v[%d] = _kfl_rr.r.z;\n"
                "        out_v[%d] = _kfl_rr.v.x;\n"
                "        out_v[%d] = _kfl_rr.v.y;\n"
                "        out_v[%d] = _kfl_rr.v.z;\n"
                "    }\n",
                chf, tgt, off, off + 1, off + 2, off + 3, off + 4,
                off + 5);
            continue;
        }
        if (rl_observe_form(s) == RL_OBS_REF) {
            /* A craft reporting where its plan says it should be, as
             * an error against where it is. Everything this needs was
             * resolved when the plan was read at compile time, so
             * there is no diagnostic here: the refusals are at the
             * `reference=` and at this statement, both taken before
             * any of the tables below were written.
             *
             * Only the current knot is read, and the index of it comes
             * from the clock alone. That is the whole of what keeps a
             * controller from seeing past it: there is no expression
             * below in which a later knot appears, so altering one
             * cannot move a published number. */
            const RlReference *rf = &m->refs[m->obs_ref[i]];
            int ri  = m->obs_ref[i];
            int tgt = rf->body;
            int fb  = rf->frame_body;

            fputs(
                "    {\n"
                "        double _kfl_qr[3] = { 0.0, 0.0, 0.0 };\n"
                "        double _kfl_qv[3] = { 0.0, 0.0, 0.0 };\n", out);
            fprintf(out,
                "        K26AstroBody *_kfl_qc = k26astro_world_body_at("
                "world, kflrl_body_idx_[%d]);\n"
                "        K26AstroBody *_kfl_qf = k26astro_world_body_at("
                "world, kflrl_body_idx_[%d]);\n", tgt, fb);
            if (rf->frame_kind == K26RL_REF_FRAME_LVLH) {
                fputs(
                "        K26AstroBody *_kfl_qp = (_kfl_qf && "
                "_kfl_qf->parent_body_idx >= 0)\n"
                "            ? k26astro_world_body_at(world, "
                "_kfl_qf->parent_body_idx) : NULL;\n"
                /* A frame body sitting at its parent's centre, or
                 * moving straight at it, names no direction of motion
                 * and so no frame. That is a configuration rather than
                 * a declaration and cannot be refused when the program
                 * is compiled, so the six error components read zero,
                 * which is what the relative form does in the same
                 * state and for the same reason. Publishing the knot
                 * unreduced there would be an absolute wearing the
                 * name of an error. */
                "        if (_kfl_qc && _kfl_qf && _kfl_qp) {\n"
                "            K26AstroProxFrame _kfl_qw;\n"
                "            K26AstroProxRel _kfl_qz;\n"
                "            if (k26astro_prox_frame(&_kfl_qp->pos, "
                "_kfl_qp->vel,\n"
                "                                    &_kfl_qf->pos, "
                "_kfl_qf->vel,\n"
                "                                    &_kfl_qw) == "
                "K26ASTRO_PROX_OK &&\n"
                "                k26astro_prox_relative(&_kfl_qw,\n"
                "                    &_kfl_qf->pos, _kfl_qf->vel,\n"
                "                    &_kfl_qc->pos, _kfl_qc->vel,\n"
                "                    &_kfl_qz) == K26ASTRO_PROX_OK) {\n"
                "                _kfl_qr[0] = _kfl_qz.r.x;\n"
                "                _kfl_qr[1] = _kfl_qz.r.y;\n"
                "                _kfl_qr[2] = _kfl_qz.r.z;\n"
                "                _kfl_qv[0] = _kfl_qz.v.x;\n"
                "                _kfl_qv[1] = _kfl_qz.v.y;\n"
                "                _kfl_qv[2] = _kfl_qz.v.z;\n"
                "            }\n"
                "        }\n", out);
            } else {
                fputs(
                /* The non-rotating frame centred on the named body, on
                 * the world's own axes. The separation is taken with
                 * the exact sector-aware subtraction rather than by
                 * flattening two absolute coordinates, because the
                 * pair may sit anywhere in the system and the
                 * separation is the small quantity. */
                "        if (_kfl_qc && _kfl_qf) {\n"
                "            K26V3 _kfl_qd = k26astro_pos_sub(&_kfl_qc->pos,"
                " &_kfl_qf->pos);\n"
                "            _kfl_qr[0] = _kfl_qd.x;\n"
                "            _kfl_qr[1] = _kfl_qd.y;\n"
                "            _kfl_qr[2] = _kfl_qd.z;\n"
                "            _kfl_qv[0] = _kfl_qc->vel.x - _kfl_qf->vel.x;\n"
                "            _kfl_qv[1] = _kfl_qc->vel.y - _kfl_qf->vel.y;\n"
                "            _kfl_qv[2] = _kfl_qc->vel.z - _kfl_qf->vel.z;\n"
                "        }\n", out);
            }
            fprintf(out,
                "        int _kfl_qi = kflrl_ref_current_(%d, t_ep);\n"
                "        const double *_kfl_qk = kflrl_ref_knots_[%d]\n"
                "            + (size_t)_kfl_qi * 8;\n", ri, ri);
            for (int c = 0; c < 3; c++) {
                fprintf(out, "        out_v[%d] = _kfl_qk[%d] - "
                             "_kfl_qr[%d];\n", off + c, 1 + c, c);
            }
            for (int c = 0; c < 3; c++) {
                fprintf(out, "        out_v[%d] = _kfl_qk[%d] - "
                             "_kfl_qv[%d];\n", off + 3 + c, 4 + c, c);
            }
            fprintf(out,
                "        out_v[%d] = (kflrl_ref_epoch_[%d] + _kfl_qk[0])"
                " - t_ep;\n"
                "        out_v[%d] = _kfl_qk[7];\n"
                "    }\n", off + 6, ri, off + 7);
            continue;
        }
        if (rl_observe_form(s) == RL_OBS_PROP) {
            /* A craft reporting what it has left to spend. The body
             * must bind an assembly that holds propellant: without a
             * tank there is nothing to report, and four channels that
             * could only ever read zero are worse than a diagnostic
             * at the line that asked for them. */
            int tgt = rl_body_index_of(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe propulsion of `%s`: no astro_body of that "
                    "name is declared in this world", s->name);
                return 1;
            }
            int slot = -1, seen = 0;
            for (int b = 0; b < m->n_bodies; b++) {
                if (!rl_body_has_assembly(m, b)) continue;
                if (b == tgt) slot = seen;
                seen++;
            }
            if (slot < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe propulsion of `%s`: `%s` declares no "
                    "`assembly=`, so it carries neither thrusters nor "
                    "propellant", s->name, s->name);
                return 1;
            }
            if (!m->veh_has_prop[slot]) {
                kflc_diag_errorf(diag, s->line,
                    "observe propulsion of `%s`: its assembly declares "
                    "no component holding propellant, so there is no "
                    "tank to report; mark the component that holds it "
                    "with `propellant`, whose declared mass is a full "
                    "tank", s->name);
                return 1;
            }
            /* The remaining velocity change is the rocket equation
             * read forwards, at the thrust-weighted mean specific
             * impulse of the craft's thrusters. An empty tank leaves
             * the log at zero, so the channel reads zero without the
             * quotient being taken at a mass ratio of one. */
            fprintf(out,
                "    {\n"
                "        double _kfl_pk = prop ? prop[%d] : 0.0;\n"
                "        double _kfl_cap = %.17g;\n"
                "        double _kfl_dry = %.17g;\n"
                "        double _kfl_m = 0.0, _kfl_c[3], _kfl_I6[6];\n"
                "        kflrl_mass_props_(%d, _kfl_pk, &_kfl_m, _kfl_c,\n"
                "                          _kfl_I6);\n"
                "        out_v[%d] = _kfl_pk;\n"
                "        out_v[%d] = _kfl_pk / _kfl_cap;\n"
                "        out_v[%d] = _kfl_m;\n"
                "        out_v[%d] = (_kfl_pk > 0.0 && _kfl_dry > 0.0)\n"
                "            ? %.17g * KFLRL_G0 * std::log(_kfl_m / "
                "_kfl_dry)\n"
                "            : 0.0;\n"
                "    }\n",
                slot, m->veh_prop_cap[slot], m->veh_struct_mass[slot],
                slot, off, off + 1, off + 2, off + 3,
                m->veh_isp[slot]);
            continue;
        }
        if (rl_observe_is_attitude(s)) {
            /* A body reporting itself: the orientation and the rate
             * as they stand after the advance, with no observer, no
             * light-time correction and no aberration to apply. */
            int tgt = rl_body_index_of(m, s->name);
            if (tgt < 0) {
                kflc_diag_errorf(diag, s->line,
                    "observe attitude of `%s`: no astro_body of that name "
                    "is declared in this world", s->name);
                return 1;
            }
            fprintf(out,
                "    {\n"
                "        const K26AstroBody *_kfl_b = "
                "k26astro_world_body_at(world, kflrl_body_idx_[%d]);\n"
                "        if (_kfl_b) {\n"
                "            out_v[%d] = _kfl_b->attitude.w;\n"
                "            out_v[%d] = _kfl_b->attitude.x;\n"
                "            out_v[%d] = _kfl_b->attitude.y;\n"
                "            out_v[%d] = _kfl_b->attitude.z;\n"
                "            out_v[%d] = _kfl_b->omega.x;\n"
                "            out_v[%d] = _kfl_b->omega.y;\n"
                "            out_v[%d] = _kfl_b->omega.z;\n"
                "        } else {\n"
                "            out_v[%d] = 1.0;\n"
                "            out_v[%d] = 0.0; out_v[%d] = 0.0; "
                "out_v[%d] = 0.0;\n"
                "            out_v[%d] = 0.0; out_v[%d] = 0.0; "
                "out_v[%d] = 0.0;\n"
                "        }\n"
                "    }\n",
                tgt, off, off + 1, off + 2, off + 3, off + 4, off + 5,
                off + 6, off, off + 1, off + 2, off + 3, off + 4,
                off + 5, off + 6);
            continue;
        }
        const char *observer = "_observer";
        const char *mode_kw = NULL;
        for (const KflcAttr *a = s->attrs; a; a = a->next) {
            if (!a->name) continue;
            if (strcmp(a->name, "observer") == 0 &&
                a->value.kind == KFLV_IDENT && a->value.u.s) {
                observer = a->value.u.s;
            } else if (strcmp(a->name, "mode") == 0 &&
                       a->value.kind == KFLV_IDENT) {
                mode_kw = a->value.u.s;
            }
        }
        int tgt = rl_body_index_of(m, s->name);
        int obs = rl_body_index_of(m, observer);
        fputs("    {\n", out);
        if (mode_kw) {
            const char *en = "K26ASTRO_OBS_ASTROMETRIC";
            if      (strcmp(mode_kw, "geometric") == 0)
                en = "K26ASTRO_OBS_GEOMETRIC";
            else if (strcmp(mode_kw, "apparent") == 0)
                en = "K26ASTRO_OBS_APPARENT";
            else if (strcmp(mode_kw, "topocentric") == 0)
                en = "K26ASTRO_OBS_TOPOCENTRIC";
            fprintf(out, "        (void)k26astro_world_set_observer_mode"
                         "(world, %s);\n", en);
        }
        if (tgt >= 0) {
            fprintf(out, "        int _kfl_t = kflrl_body_idx_[%d];\n",
                    tgt);
        } else {
            fprintf(out, "        int _kfl_t = k26astro_world_find_body"
                         "(world, \"%s\");\n", s->name ? s->name : "?");
        }
        if (obs >= 0) {
            fprintf(out, "        int _kfl_o = kflrl_body_idx_[%d];\n",
                    obs);
        } else {
            fprintf(out, "        int _kfl_o = k26astro_world_find_body"
                         "(world, \"%s\");\n", observer);
        }
        fprintf(out,
            "        K26AstroPos _kfl_p; memset(&_kfl_p, 0, sizeof "
            "_kfl_p);\n"
            "        K26V3 _kfl_d; _kfl_d.x = _kfl_d.y = _kfl_d.z = "
            "0.0;\n"
            "        double _kfl_range = 0.0;\n"
            "        double _kfl_rrate = 0.0;\n"
            "        if (_kfl_t >= 0 && _kfl_o >= 0) {\n"
            "            (void)k26astro_world_observe(world, _kfl_t, "
            "_kfl_o, &_kfl_p, &_kfl_d);\n"
            "            K26AstroBody *_kfl_ob = "
            "k26astro_world_body_at(world, _kfl_o);\n"
            "            K26AstroBody *_kfl_tb = "
            "k26astro_world_body_at(world, _kfl_t);\n"
            "            if (_kfl_ob) {\n"
            "                K26V3 _kfl_r = k26astro_pos_sub(&_kfl_p, "
            "&_kfl_ob->pos);\n"
            "                _kfl_range = std::sqrt(_kfl_r.x * _kfl_r.x "
            "+ _kfl_r.y * _kfl_r.y + _kfl_r.z * _kfl_r.z);\n"
            "            }\n"
            /* The range rate is the geometric one: the observation
             * mode corrects a position, and there is no corrected
             * velocity to differentiate, so the rate is taken from
             * the two bodies' current state. Zero separation yields
             * 0.0 rather than a quotient of zeroes. */
            "            if (_kfl_ob && _kfl_tb) {\n"
            "                K26V3 _kfl_gr = k26astro_pos_sub("
            "&_kfl_tb->pos, &_kfl_ob->pos);\n"
            "                K26V3 _kfl_gv;\n"
            "                _kfl_gv.x = _kfl_tb->vel.x - _kfl_ob->vel.x;\n"
            "                _kfl_gv.y = _kfl_tb->vel.y - _kfl_ob->vel.y;\n"
            "                _kfl_gv.z = _kfl_tb->vel.z - _kfl_ob->vel.z;\n"
            "                double _kfl_gm = std::sqrt("
            "_kfl_gr.x * _kfl_gr.x + _kfl_gr.y * _kfl_gr.y"
            " + _kfl_gr.z * _kfl_gr.z);\n"
            "                if (_kfl_gm != 0.0) {\n"
            "                    _kfl_rrate = (_kfl_gr.x * _kfl_gv.x"
            " + _kfl_gr.y * _kfl_gv.y + _kfl_gr.z * _kfl_gv.z)"
            " / _kfl_gm;\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "        out_v[%d] = _kfl_d.x;\n"
            "        out_v[%d] = _kfl_d.y;\n"
            "        out_v[%d] = _kfl_d.z;\n"
            "        out_v[%d] = _kfl_range;\n"
            "        out_v[%d] = _kfl_rrate;\n"
            "    }\n",
            off, off + 1, off + 2, off + 3, off + 4);
    }
    /* The truth half. A paired observe publishes each component twice,
     * and both halves leave this function carrying the same
     * uncorrupted value; the sensor pass below rewrites the measured
     * half in place. Copying here rather than computing twice is what
     * keeps the two halves the same number before any noise, which is
     * the property `with truth` exists to give a consumer. */
    for (int i = 0; i < m->n_observes; i++) {
        if (!rl_observe_has_truth(m->observes[i])) continue;
        int off = rl_obs_offset(m->observes, i);
        int w   = rl_observe_base_width(m->observes[i]);
        fprintf(out,
            "    for (int k = 0; k < %d; k++) out_v[%d + k] = out_v[%d + k];\n",
            w, off + w, off);
    }
    fputs("}\n\n", out);

    return 0;
}
