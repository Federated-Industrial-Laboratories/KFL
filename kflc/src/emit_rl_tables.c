/* emit_rl_tables.c - the static tables an artifact carries: masses,
 * actuators, payloads, sensors, references and plans. */
#include "emit_rl_internal.h"

/* The actuator tables and the per-environment block their commands
 * and wheel momenta live in. The tables are what the assemblies
 * declared and never change; the block is state, one per environment,
 * allocated at create and reset with the episode. */

/* The mass properties of every vehicle as a function of the
 * propellant it has left, and the one function that evaluates them.
 *
 * The assembly derives a vehicle's mass, centre of mass and inertia
 * from its components by closed form. Those forms are linear in a
 * component's mass once its geometry is fixed, and the tank's
 * geometry does not move as it empties, so the whole family follows
 * from two sets of numbers: the structure's aggregates, and the
 * tank's per unit of mass. What is emitted are those, not a table of
 * fills, and the evaluation below is the same closed form the
 * compiler applied, with the tank at whatever it now holds.
 *
 * The tensors are summed about the body-frame origin rather than
 * about the centre of mass, because the centre of mass is itself a
 * function of the fill and a tensor taken about it cannot be summed
 * once and scaled after. The shift at the end takes it to where the
 * centre of mass has moved to, which is where the rotational equation
 * wants it.
 *
 * A vehicle whose assembly declares no tank has a capacity of zero
 * and aggregates covering the whole of it, so this evaluates to its
 * constructed properties for any argument it is ever given. It is
 * never given one: such a vehicle keeps the constants the assembly
 * derived, bit for bit.
 */

/* The plans this program carries, as constants.
 *
 * Two things travel per plan and each has its own reason. The knots
 * are what the published channels are computed from, so they are here
 * as a table the stepping path indexes: it opens nothing, allocates
 * nothing, and takes no branch on whether a file was found. The
 * file's own bytes are here so that the episode record can carry the
 * plan a run flew against, which is what makes a recording sufficient
 * to explain the run; they are read at an episode boundary and never
 * on a step.
 *
 * A knot is eight numbers in the file's order: the time offset from
 * the plan's epoch, three position components, three velocity
 * components, and the tolerance radius. */

void rl_emit_reference_tables(FILE *out, const RlModel *m)
{
    fprintf(out,
        "/* Plans bound to craft by `reference=`. Zero keeps every\n"
        " * plan-related declaration out of a program that flies\n"
        " * none. */\n"
        "#define KFLRL_N_REF %d\n\n", m->n_refs);
    if (m->n_refs == 0) return;

    for (int i = 0; i < m->n_refs; i++) {
        const RlReference *r = &m->refs[i];
        char hex[K26RL_SHA256_HEX];

        k26rl_sha256_hex(r->digest, hex);
        fprintf(out,
            "/* `%s`, in the %s frame of `%s`, digest %s */\n",
            m->bodies[r->body].body->name
                ? m->bodies[r->body].body->name : "?",
            r->frame_kind == K26RL_REF_FRAME_LVLH
                ? "local-vertical local-horizontal" : "inertial",
            m->bodies[r->frame_body].body->name
                ? m->bodies[r->frame_body].body->name : "?", hex);
        fprintf(out, "static const double kflrl_ref_knots_%d_[][8] = {\n",
                i);
        for (int k = 0; k < r->n_knots; k++) {
            const double *kn = r->knots + (size_t)k * 8u;

            fprintf(out,
                "    { %.17g, %.17g, %.17g, %.17g,"
                " %.17g, %.17g, %.17g, %.17g },\n",
                kn[0], kn[1], kn[2], kn[3], kn[4], kn[5], kn[6], kn[7]);
        }
        fputs("};\n", out);
        fprintf(out,
            "static const unsigned char kflrl_ref_bytes_%d_[] = {\n", i);
        for (uint32_t b = 0; b < r->len; b++) {
            fprintf(out, "%s0x%02x,", (b % 12u) == 0 ? "    " : " ",
                    (unsigned)r->bytes[b]);
            if ((b % 12u) == 11u || b + 1u == r->len) fputc('\n', out);
        }
        fputs("};\n", out);
        fprintf(out,
            "static const unsigned char kflrl_ref_digest_%d_[32] = {\n",
            i);
        for (int b = 0; b < K26RL_SHA256_BYTES; b++) {
            fprintf(out, "%s0x%02x,", (b % 12) == 0 ? "    " : " ",
                    (unsigned)r->digest[b]);
            if ((b % 12) == 11 || b + 1 == K26RL_SHA256_BYTES)
                fputc('\n', out);
        }
        fputs("};\n\n", out);
    }

    fputs("static const double *const kflrl_ref_knots_[] = {\n", out);
    for (int i = 0; i < m->n_refs; i++) {
        fprintf(out, "    &kflrl_ref_knots_%d_[0][0],\n", i);
    }
    fputs("};\nstatic const int kflrl_ref_count_[] = {\n", out);
    for (int i = 0; i < m->n_refs; i++) {
        fprintf(out, "    %d,\n", m->refs[i].n_knots);
    }
    fputs("};\nstatic const double kflrl_ref_epoch_[] = {\n", out);
    for (int i = 0; i < m->n_refs; i++) {
        fprintf(out, "    %.17g,\n", m->refs[i].epoch);
    }
    fputs("};\nstatic const unsigned char *const kflrl_ref_bytes_[] = {\n",
          out);
    for (int i = 0; i < m->n_refs; i++) {
        fprintf(out, "    kflrl_ref_bytes_%d_,\n", i);
    }
    fputs("};\nstatic const unsigned int kflrl_ref_len_[] = {\n", out);
    for (int i = 0; i < m->n_refs; i++) {
        fprintf(out, "    %uu,\n", (unsigned)m->refs[i].len);
    }
    fputs("};\nstatic const unsigned char *const kflrl_ref_digest_[] = {\n",
          out);
    for (int i = 0; i < m->n_refs; i++) {
        fprintf(out, "    kflrl_ref_digest_%d_,\n", i);
    }
    fputs("};\n\n", out);

    fputs(
"/* The current knot: the earliest whose time has not passed.\n"
" *\n"
" * The rule is time and not arrival. A knot's time passing without\n"
" * the craft inside its tolerance is not an error and ends nothing;\n"
" * the plan advances, the craft is late, and the seconds-to channel\n"
" * goes negative on the knot after it. Nothing here reads where the\n"
" * craft is, which is what makes that true rather than merely\n"
" * intended.\n"
" *\n"
" * When every knot's time has passed the last one stays current: a\n"
" * plan that has run out is one whose final state is still the state\n"
" * asked for, and publishing nothing instead would read exactly like\n"
" * a craft sitting on a knot. */\n"
"static int kflrl_ref_current_(int r, double t)\n"
"{\n"
"    const double *k = kflrl_ref_knots_[r];\n"
"    int n = kflrl_ref_count_[r];\n"
"    for (int i = 0; i < n; i++) {\n"
"        if (kflrl_ref_epoch_[r] + k[(size_t)i * 8] >= t) return i;\n"
"    }\n"
"    return n - 1;\n"
"}\n\n", out);
}

/* The plan blocks this program emits, as constants.
 *
 * A plan a world writes is the actions of the step that ended the
 * episode, read out of the action vector at the offset this table
 * carries. The buffers the encode needs are sized here, from the
 * widest plan the program declares, and allocated once with the
 * handle: an episode end is an output point, not a place to start
 * allocating. */

void rl_emit_plan_tables(FILE *out, const RlModel *m)
{
    int max_slots = 0;
    uint64_t max_bytes = 0;

    for (int p = 0; p < m->n_plans; p++) {
        const RlPlanOut *po = &m->plans[p];
        uint64_t need = (uint64_t)K26RL_REF_HEADER_BYTES +
            strlen(m->bodies[po->frame_body].body->name
                   ? m->bodies[po->frame_body].body->name : "") +
            strlen(po->provenance) +
            (uint64_t)po->slots * K26RL_REF_KNOT_BYTES;

        if (po->slots > max_slots) max_slots = po->slots;
        if (need > max_bytes) max_bytes = need;
    }
    fprintf(out,
        "/* Plans this world emits. Zero keeps every plan-writing\n"
        " * declaration out of a program that emits none. */\n"
        "#define KFLRL_N_PLAN %d\n", m->n_plans);
    if (m->n_plans == 0) {
        fputs("\n", out);
        return;
    }
    fprintf(out,
        "#define KFLRL_PLAN_SLOTS %d\n"
        "#define KFLRL_PLAN_BYTES %lluu\n\n",
        max_slots, (unsigned long long)max_bytes);

    fputs("static const char *const kflrl_plan_file_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        fputs("    ", out);
        rl_emit_string_literal(out, m->plans[p].file);
        fputs(",\n", out);
    }
    fputs("};\nstatic const char *const kflrl_plan_frame_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        const char *fn = m->bodies[m->plans[p].frame_body].body->name;

        fputs("    ", out);
        rl_emit_string_literal(out, fn ? fn : "");
        fputs(",\n", out);
    }
    fputs("};\nstatic const char *const kflrl_plan_prov_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        fputs("    ", out);
        rl_emit_string_literal(out, m->plans[p].provenance);
        fputs(",\n", out);
    }
    fputs("};\nstatic const unsigned kflrl_plan_kind_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        fprintf(out, "    %uu,\n", (unsigned)m->plans[p].frame_kind);
    }
    fputs("};\nstatic const int kflrl_plan_nslots_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        fprintf(out, "    %d,\n", m->plans[p].slots);
    }
    fputs("};\nstatic const double kflrl_plan_epoch_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        fprintf(out, "    %.17g,\n", m->plans[p].epoch);
    }
    fputs("};\nstatic const int kflrl_plan_act_[] = {\n", out);
    for (int p = 0; p < m->n_plans; p++) {
        fprintf(out, "    %d,\n", m->plans[p].act_first);
    }
    fputs("};\n\n", out);
}

void rl_emit_mass_tables(FILE *out, const RlModel *m)
{
    fprintf(out,
        "/* Vehicles carrying propellant. Zero keeps the consumption\n"
        " * arithmetic out of a program whose craft spend nothing. */\n"
        "#define KFLRL_N_PROPELLANT %d\n"
        "/* Standard gravity, the defined constant, exactly. It is what\n"
        " * makes a specific impulse in seconds mean a speed, and it is\n"
        " * a definition rather than a measurement: it is not the local\n"
        " * gravitational acceleration anywhere in this world. */\n"
        "#define KFLRL_G0 9.80665\n\n", m->n_prop_veh);
    if (m->n_veh == 0) return;

    fputs("static const double kflrl_veh_prop_cap_[] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    %.17g,\n", m->veh_prop_cap[i]);
    }
    fputs("};\n", out);
    fputs("static const double kflrl_veh_struct_mass_[] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    %.17g,\n", m->veh_struct_mass[i]);
    }
    fputs("};\n", out);
    fputs("static const double kflrl_veh_isp_[] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    %.17g,\n", m->veh_isp[i]);
    }
    fputs("};\n", out);
    fputs("static const double kflrl_veh_struct_moment_[][3] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    { %.17g, %.17g, %.17g },\n",
                m->veh_struct_moment[i][0], m->veh_struct_moment[i][1],
                m->veh_struct_moment[i][2]);
    }
    fputs("};\n", out);
    fputs("static const double kflrl_veh_prop_centroid_[][3] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    { %.17g, %.17g, %.17g },\n",
                m->veh_prop_centroid[i][0], m->veh_prop_centroid[i][1],
                m->veh_prop_centroid[i][2]);
    }
    fputs("};\n", out);
    fputs("static const double kflrl_veh_struct_inertia_[][6] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    { %.17g, %.17g, %.17g, %.17g, %.17g, %.17g },\n",
                m->veh_struct_inertia[i][0], m->veh_struct_inertia[i][1],
                m->veh_struct_inertia[i][2], m->veh_struct_inertia[i][3],
                m->veh_struct_inertia[i][4], m->veh_struct_inertia[i][5]);
    }
    fputs("};\n", out);
    fputs("static const double kflrl_veh_prop_inertia_[][6] = {\n", out);
    for (int i = 0; i < m->n_veh; i++) {
        fprintf(out, "    { %.17g, %.17g, %.17g, %.17g, %.17g, %.17g },\n",
                m->veh_prop_inertia[i][0], m->veh_prop_inertia[i][1],
                m->veh_prop_inertia[i][2], m->veh_prop_inertia[i][3],
                m->veh_prop_inertia[i][4], m->veh_prop_inertia[i][5]);
    }
    fputs("};\n\n", out);

    fputs(
"/* One vehicle's mass, centre of mass and inertia at a given fill.\n"
" * The inertia is returned in the six-component form the assembly\n"
" * uses: xx, yy, zz, then the products already negated, about the\n"
" * centre of mass this same call computes. */\n"
"static void kflrl_mass_props_(int veh, double prop_kg, double *mass,\n"
"                              double *com, double *inertia)\n"
"{\n"
"    const double *sm = kflrl_veh_struct_moment_[veh];\n"
"    const double *si = kflrl_veh_struct_inertia_[veh];\n"
"    const double *pc = kflrl_veh_prop_centroid_[veh];\n"
"    const double *pi = kflrl_veh_prop_inertia_[veh];\n"
"    double mtot = kflrl_veh_struct_mass_[veh] + prop_kg;\n"
"    double c[3];\n"
"    double I[6];\n"
"    for (int q = 0; q < 3; q++) c[q] = (sm[q] + prop_kg * pc[q]) / mtot;\n"
"    for (int q = 0; q < 6; q++) I[q] = si[q] + prop_kg * pi[q];\n"
"    /* Parallel axis, from the body-frame origin to the centre of\n"
"     * mass: subtract what a point mass of the whole vehicle sitting\n"
"     * there would have contributed. */\n"
"    double dd = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];\n"
"    I[0] -= mtot * (dd - c[0] * c[0]);\n"
"    I[1] -= mtot * (dd - c[1] * c[1]);\n"
"    I[2] -= mtot * (dd - c[2] * c[2]);\n"
"    I[3] -= mtot * (-c[0] * c[1]);\n"
"    I[4] -= mtot * (-c[0] * c[2]);\n"
"    I[5] -= mtot * (-c[1] * c[2]);\n"
"    *mass = mtot;\n"
"    for (int q = 0; q < 3; q++) com[q] = c[q];\n"
"    for (int q = 0; q < 6; q++) inertia[q] = I[q];\n"
"}\n\n"
"/* Install those properties on the body and on the vehicle. This is\n"
" * the one place they are written, so construction, episode reset\n"
" * and the burn itself cannot disagree about what a given fill\n"
" * means; the reset restores by recomputing at the declared\n"
" * capacity rather than by remembering a number, which is what makes\n"
" * the restoration exact.\n"
" *\n"
" * The body carries the mass, and its gravitational parameter\n"
" * follows through the setter. The vehicle carries the basic mass\n"
" * its own library reports, the centre of mass every thruster torque\n"
" * is taken about, and the inertia tensor, whose inverse the\n"
" * rotational equation needs and which the setter recomputes. */\n"
"static void kflrl_apply_props_(K26AstroVehicle *v, K26AstroBody *b,\n"
"                               int veh, double prop_kg, double *com_out)\n"
"{\n"
"    double mtot, c[3], I6[6];\n"
"    kflrl_mass_props_(veh, prop_kg, &mtot, c, I6);\n"
"    if (b) k26astro_body_set_mass(b, mtot);\n"
"    if (v) {\n"
"        k26astro_vehicle_set_dry_mass(v, mtot);\n"
"        k26astro_vehicle_set_com_offset(v, c[0], c[1], c[2]);\n"
"        K26M3 I;\n"
"        I.m[0][0] = I6[0]; I.m[0][1] = I6[3]; I.m[0][2] = I6[4];\n"
"        I.m[1][0] = I6[3]; I.m[1][1] = I6[1]; I.m[1][2] = I6[5];\n"
"        I.m[2][0] = I6[4]; I.m[2][1] = I6[5]; I.m[2][2] = I6[2];\n"
"        k26astro_vehicle_set_inertia_full(v, I);\n"
"    }\n"
"    if (com_out) {\n"
"        com_out[0] = c[0]; com_out[1] = c[1]; com_out[2] = c[2];\n"
"    }\n"
"}\n\n", out);
}

void rl_emit_actuators(FILE *out, const RlModel *m)
{
    fprintf(out,
        "#define KFLRL_N_WHEELS %d\n"
        "#define KFLRL_N_TORQUERS %d\n"
        "#define KFLRL_N_THRUSTERS %d\n"
        "#define KFLRL_N_CMD (KFLRL_N_WHEELS + KFLRL_N_TORQUERS + "
        "KFLRL_N_THRUSTERS)\n\n",
        m->n_wheels, m->n_torquers, m->n_thrusters);

    if (m->n_wheels > 0) {
        fputs("static const K26AstroAttWheel kflrl_wheel_desc_[] = {\n",
              out);
        for (int i = 0; i < m->n_wheels; i++) {
            const RlWheel *w = &m->wheels[i];
            fprintf(out,
                "    { { %.17g, %.17g, %.17g }, %.17g, %.17g, %.17g, "
                "%.17g, %.17g, %.17g, 0.0, 0.0 },\n",
                w->axis[0], w->axis[1], w->axis[2], w->spin_inertia,
                w->max_momentum, w->max_torque, w->viscous, w->coulomb,
                w->dead_rate);
        }
        fputs("};\n", out);
        fputs("static const int kflrl_wheel_veh_[] = {\n", out);
        for (int i = 0; i < m->n_wheels; i++) {
            fprintf(out, "    %d,\n", m->wheels[i].veh);
        }
        fputs("};\n\n", out);
    }
    if (m->n_torquers > 0) {
        fputs("static const K26AstroAttTorquer kflrl_torquer_desc_[] = {\n",
              out);
        for (int i = 0; i < m->n_torquers; i++) {
            const RlTorquer *q = &m->torquers[i];
            fprintf(out, "    { { %.17g, %.17g, %.17g }, %.17g, 0.0 },\n",
                    q->axis[0], q->axis[1], q->axis[2], q->max_dipole);
        }
        fputs("};\n", out);
        fputs("static const int kflrl_torquer_veh_[] = {\n", out);
        for (int i = 0; i < m->n_torquers; i++) {
            fprintf(out, "    %d,\n", m->torquers[i].veh);
        }
        fputs("};\n\n", out);
    }
    if (m->n_thrusters > 0) {
        fputs("static const K26AstroAttThruster kflrl_thruster_desc_[] "
              "= {\n", out);
        for (int i = 0; i < m->n_thrusters; i++) {
            const RlThruster *t = &m->thrusters[i];
            fprintf(out,
                "    { { %.17g, %.17g, %.17g }, { %.17g, %.17g, %.17g }, "
                "%.17g, 0.0, %.17g },\n",
                t->at[0], t->at[1], t->at[2], t->dir[0], t->dir[1],
                t->dir[2], t->max_thrust, t->isp_s);
        }
        fputs("};\n", out);
        fputs("static const int kflrl_thruster_veh_[] = {\n", out);
        for (int i = 0; i < m->n_thrusters; i++) {
            fprintf(out, "    %d,\n", m->thrusters[i].veh);
        }
        fputs("};\n\n", out);
    }

    fputs(
"/* One environment's actuator state. Commands are written by on_step\n"
" * once per control period and held across the sub-advances, which is\n"
" * what a zero-order hold means; wheel momentum is the only state that\n"
" * persists between transitions, and the episode reset clears both. */\n"
"typedef struct {\n"
"    double wheel_h[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];\n"
"    double cmd[KFLRL_N_CMD > 0 ? KFLRL_N_CMD : 1];\n"
"} KflrlAct;\n\n", out);


    /* The accessors a command or reading in on_step lowers to. */
    for (int i = 0; i < m->n_acts; i++) {
        const RlActRef *r = &m->acts[i];
        int base = (r->kind == 0) ? r->index
                 : (r->kind == 1) ? m->n_wheels + r->index
                                  : m->n_wheels + m->n_torquers + r->index;
        if (r->field == 0) {
            fprintf(out,
                "static void kflrl_act_set_%d(KflrlAct *a, double v)\n"
                "{\n    a->cmd[%d] = v;\n}\n\n", i, base);
        } else if (r->field == 1) {
            fprintf(out,
                "static double kflrl_act_get_%d(const KflrlAct *a)\n"
                "{\n    return a->wheel_h[%d];\n}\n\n", i, r->index);
        } else {
            fprintf(out,
                "static double kflrl_act_get_%d(const KflrlAct *a)\n"
                "{\n    return a->wheel_h[%d] / %.17g;\n}\n\n",
                i, r->index, m->wheels[r->index].spin_inertia);
        }
    }

    if (m->n_torquers > 0) {
        fputs(
"/* The magnetic field a magnetorquer works against, in the body\n"
" * frame. The chain is where this model would go wrong quietly, so\n"
" * it is written out step by step:\n"
" *\n"
" *   the vehicle's position relative to the body it orbits, in the\n"
" *   world frame; rotated into that body's own rotating frame by its\n"
" *   rotation model, which is what makes a longitude mean anything;\n"
" *   converted to a geodetic latitude, longitude and height on the\n"
" *   ellipsoid, which is where the field model is defined; evaluated\n"
" *   there, giving north, east and down in tesla; turned into east,\n"
" *   north and up, rotated back out to the rotating frame, then to\n"
" *   the world frame, then into the body frame by the conjugate of\n"
" *   the vehicle's own orientation.\n"
" *\n"
" * The parent's rotation model is found by its `ephem_naif_id=`,\n"
" * which the compiler requires on the parent of any body carrying a\n"
" * magnetorquer: the body library's table is keyed by that id and by\n"
" * a model name of its own form, and a name declared in a program is\n"
" * a grammar identifier, so the id is the only key a program can\n"
" * supply.\n"
" *\n"
" * The zero returns below are therefore defensive and not a\n"
" * behaviour a program can reach: the compiler refuses a\n"
" * magnetorquer whose parent carries no id, so the lookup here\n"
" * cannot fail for that reason. A zero would be the honest answer\n"
" * rather than a field taken in the wrong frame, but the refusal is\n"
" * the answer the author gets. */\n"
"static K26V3 kflrl_field_body_(K26AstroWorld *w, K26AstroVehicle *v,\n"
"                               int veh)\n"
"{\n"
"    K26V3 zero = { 0.0, 0.0, 0.0 };\n"
"    if (!w || !v) return zero;\n"
"    const K26AstroBody *vb = k26astro_world_body_at(\n"
"        w, kflrl_body_idx_[kflrl_vehicle_body_[veh]]);\n"
"    if (!vb || vb->parent_body_idx < 0) return zero;\n"
"    const K26AstroBody *pb = k26astro_world_body_at(\n"
"        w, vb->parent_body_idx);\n"
"    if (!pb) return zero;\n"
"    if (pb->ephem_naif_id == 0) return zero;\n"
"    const K26AstroIAURotation *rot =\n"
"        k26astro_rotation_by_naif(pb->ephem_naif_id);\n"
"    if (!rot) return zero;\n"
"    K26AstroGravState *g = k26astro_world_grav(w);\n"
"    if (!g) return zero;\n"
"\n"
"    K26V3 r_world = k26astro_pos_sub(&vb->pos, &pb->pos);\n"
"    K26Quat fixed_from_world = k26astro_rotation_quaternion(rot, &g->t);\n"
"    K26V3 r_fixed = k26m3d_quat_rotate_v3(fixed_from_world, r_world);\n"
"\n"
"    double lat = 0.0, lon = 0.0, alt = 0.0;\n"
"    if (k26astro_att_geodetic(r_fixed, &lat, &lon, &alt) !=\n"
"        K26ASTRO_ATT_OK) return zero;\n"
"\n"
"    /* The field model's epoch argument is years past J2000. */\n"
"    double yrs = ((double)g->t.days_since_J2000 +\n"
"                  g->t.seconds_of_day / 86400.0) / 365.25;\n"
"    K26V3 ned = k26astro_geomag_field_v3(lat, lon, alt, yrs);\n"
"    /* North, east, down as the model reports it, into east, north\n"
"     * and up as the rotation below expects. The sign on the third\n"
"     * component is the whole of the conversion. */\n"
"    K26V3 enu = { ned.y, ned.x, -ned.z };\n"
"    K26V3 b_fixed = k26astro_att_enu_to_ecef(enu, lat, lon);\n"
"    K26V3 b_world = k26m3d_quat_rotate_v3(\n"
"        k26m3d_quat_conj(fixed_from_world), b_fixed);\n"
"    return k26m3d_quat_rotate_v3(k26m3d_quat_conj(vb->attitude),\n"
"                                 b_world);\n"
"}\n\n", out);
    }
    if (m->n_veh > 0 && m->n_torquers == 0) {
        fputs(
"/* No magnetorquer is declared, so no field is needed and none is\n"
" * computed. This is what keeps the Fortran-backed field model out of\n"
" * a program that does not ask for it. */\n"
"static K26V3 kflrl_field_body_(K26AstroWorld *w, K26AstroVehicle *v,\n"
"                               int veh)\n"
"{\n"
"    (void)w; (void)v; (void)veh;\n"
"    return k26m3d_v3(0.0, 0.0, 0.0);\n"
"}\n\n", out);
    }
    if (m->n_veh > 0) {
        fputs(
"/* Build one vehicle's actuator view over the environment's state.\n"
" * The descriptors are constants and the mutable parts are copied in\n"
" * and out around the call, so nothing here allocates and the state\n"
" * stays where the handle can reset it.\n"
" *\n"
" * The centre of mass is passed in rather than read from a constant\n"
" * table because it is state: a tank offset from it moves it as the\n"
" * craft burns, and every thruster torque is taken about it, so a\n"
" * craft's attitude authority changes over a long burn. */\n"
"static void kflrl_act_view_(KflrlAct *a, int veh, const double *com,\n"
"                            double thr_scale,\n"
"                            K26AstroAttWheel *wh, K26AstroAttTorquer *tq,\n"
"                            K26AstroAttThruster *th,\n"
"                            K26AstroAttActuators *out, int *w_map)\n"
"{\n"
"    int nw = 0, nq = 0, nt = 0;\n"
"    (void)veh;\n"
"#if KFLRL_N_WHEELS > 0\n"
"    for (int i = 0; i < KFLRL_N_WHEELS; i++) {\n"
"        if (kflrl_wheel_veh_[i] != veh) continue;\n"
"        wh[nw] = kflrl_wheel_desc_[i];\n"
"        wh[nw].momentum = a->wheel_h[i];\n"
"        wh[nw].command  = a->cmd[i];\n"
"        w_map[nw] = i;\n"
"        nw++;\n"
"    }\n"
"#endif\n"
"#if KFLRL_N_TORQUERS > 0\n"
"    for (int i = 0; i < KFLRL_N_TORQUERS; i++) {\n"
"        if (kflrl_torquer_veh_[i] != veh) continue;\n"
"        tq[nq] = kflrl_torquer_desc_[i];\n"
"        tq[nq].command = a->cmd[KFLRL_N_WHEELS + i];\n"
"        nq++;\n"
"    }\n"
"#endif\n"
"#if KFLRL_N_THRUSTERS > 0\n"
"    for (int i = 0; i < KFLRL_N_THRUSTERS; i++) {\n"
"        if (kflrl_thruster_veh_[i] != veh) continue;\n"
"        th[nt] = kflrl_thruster_desc_[i];\n"
"        /* Clamped here and then scaled, not the other way about: a\n"
"         * throttle beyond the unit interval is a throttle of one,\n"
"         * and the fraction of the sub-interval propellant lasted\n"
"         * applies to that. Scaling first would let an out-of-range\n"
"         * command buy back the part of the interval the tank could\n"
"         * not pay for. */\n"
"        {\n"
"            double u_ =\n"
"                a->cmd[KFLRL_N_WHEELS + KFLRL_N_TORQUERS + i];\n"
"            u_ = u_ < 0.0 ? 0.0 : (u_ > 1.0 ? 1.0 : u_);\n"
"            th[nt].command = u_ * thr_scale;\n"
"        }\n"
"        nt++;\n"
"    }\n"
"#endif\n"
"    out->wheels = wh; out->n_wheels = nw;\n"
"    out->torquers = tq; out->n_torquers = nq;\n"
"    out->thrusters = th; out->n_thrusters = nt;\n"
"    out->com = k26m3d_v3(com[0], com[1], com[2]);\n"
"}\n\n"
"/* Copy the wheel momenta back, which is the only part of the view\n"
" * that is state rather than description. */\n"
"static void kflrl_act_store_(KflrlAct *a, const K26AstroAttActuators *v,\n"
"                             const int *w_map)\n"
"{\n"
"    for (int i = 0; i < v->n_wheels; i++) {\n"
"        a->wheel_h[w_map[i]] = v->wheels[i].momentum;\n"
"    }\n"
"}\n\n", out);
    }
}

/* The sensed channels, their model terms, and the draw channels the
 * owning layer allocates to them.
 *
 * One sensed channel is one numeric component of one observe that
 * declared a sensor. Channels are allocated in source order, one per
 * term that draws, and a term drawing at both cadences takes two, so
 * that a per-episode draw at index 0 cannot be the same number as the
 * first step's draw. Adding a sensor to a program therefore allocates
 * channels above every channel already allocated and perturbs no
 * existing draw.
 *
 * The rows are emitted as plain numbers and assembled into the
 * library's typed terms at create, because the artifact is C++11 and
 * cannot initialise a union member by name, and because the bias
 * walk's two coefficients are not known until the control period is
 * read. */

/* The defense payload set, the pairs the information state is pushed
 * for, and the silhouette of every detection target.
 *
 * Everything here is a compile-time constant table, so the stepping
 * path indexes fixed arrays and constructs nothing: the payloads are
 * built once per environment in the world prefix and the step calls
 * evaluators only. */

int rl_emit_payload_tables(FILE *out, const RlModel *m)
{
    fprintf(out, "#define KFLRL_N_PAYLOAD %d\n", m->n_payloads);
    fprintf(out, "#define KFLRL_PAY_NPARAM %d\n",
            m->pay_nparam > 0 ? m->pay_nparam : 1);
    int tpay[RL_MAX_PAYLOADS * RL_MAX_BODIES];
    int tveh[RL_MAX_PAYLOADS * RL_MAX_BODIES];
    int n_pairs = rl_track_pairs(m, tpay, tveh,
                                  (int)(sizeof tpay / sizeof tpay[0]));
    fprintf(out, "#define KFLRL_N_TRACKPAIR %d\n", n_pairs);
    fprintf(out, "#define KFLRL_N_SIG %d\n", m->n_sig);
    fprintf(out, "#define KFLRL_N_EFFECTOR %d\n", rl_n_effector(m));
    fprintf(out, "#define KFLRL_EFF_STRIDE %d\n", RL_EFF_MAX_COMPS);
    /* The degradation store is indexed by (detection payload, body):
     * a countermeasure degrades one victim payload's view of one craft,
     * the craft that carries the countermeasure. Both indices are the
     * program's own counts rather than the compiler's limits, so a
     * program declaring five payloads over four bodies carries sixty
     * doubles per environment. At the limits the grammar admits, 32
     * payloads over 256 bodies, it would be 24576 doubles, which is
     * 192 kilobytes per environment: worth knowing before a program is
     * written that large, and not a shape any fixture here reaches. */
    fprintf(out, "#define KFLRL_N_SOFTKILL %d\n",
            rl_n_softkill_engage(m));
    {
        int stride = m->n_bodies > 0 ? m->n_bodies : 1;
        int n_deg = m->n_payloads * stride;
        fprintf(out, "#define KFLRL_DEG_STRIDE %d\n", stride);
        fprintf(out, "#define KFLRL_N_DEG %d\n\n", n_deg > 0 ? n_deg : 1);
    }
    fputs(
"/* This environment's slice of the payload handle array and of the\n"
" * parameter store. Both are macros so a program with no payload\n"
" * passes a null pointer through the same call sites rather than\n"
" * carrying a second shape of them. */\n"
"#if KFLRL_N_PAYLOAD > 0\n"
"#define KFLRL_PAYP(h, e) ((h)->payp + (size_t)(e) * KFLRL_N_PAYLOAD \\\n"
"                          * KFLRL_PAY_NPARAM)\n"
"#define KFLRL_PAYH(h, e) ((h)->payloads + (size_t)(e) * KFLRL_N_PAYLOAD)\n"
"#define KFLRL_PAYU(h, e) ((h)->payu + (size_t)(e) * KFLRL_N_PAYLOAD)\n"
"#define KFLRL_INFOT(h, e)  ((h)->info_t[(e)])\n"
"#define KFLRL_INFODAY(h, e) ((h)->info_day[(e)])\n"
"#else\n"
"#define KFLRL_PAYP(h, e) ((double *)0)\n"
"#define KFLRL_PAYH(h, e) ((void **)0)\n"
"#define KFLRL_PAYU(h, e) ((void **)0)\n"
"#define KFLRL_INFOT(h, e)  (0.0)\n"
"#define KFLRL_INFODAY(h, e) ((int64_t)0)\n"
"#endif\n"
"#if KFLRL_N_VEHICLES > 0\n"
"#define KFLRL_VEHS(h, e) ((h)->vehicles + (size_t)(e) * KFLRL_N_VEHICLES)\n"
/* The centre of mass and the propellant left are per environment and
 * per vehicle, because both are state that a burn moves. A program
 * whose craft carry no tank still holds them: the values never leave
 * what the assembly derived, and one shape is cheaper to keep right
 * than two. */
"#define KFLRL_COM(h, e, v) ((h)->veh_com + \\\n"
"    ((size_t)(e) * KFLRL_N_VEHICLES + (size_t)(v)) * 3)\n"
"#define KFLRL_PROP(h, e) ((h)->prop + (size_t)(e) * KFLRL_N_VEHICLES)\n"
/* The fraction of the current sub-interval this vehicle's thrusters
 * had propellant for. It is 1.0 everywhere except the sub-interval a
 * tank runs out in, and it is 1.0 for the whole run of a craft that
 * carries no tank. */
"#define KFLRL_THRSC(h, e, v) \\\n"
"    ((h)->thr_scale[(size_t)(e) * KFLRL_N_VEHICLES + (size_t)(v)])\n"
"/* The step mean of the fraction above: the integral the step kept,\n"
" * over the time it advanced, or 1.0 before any step has run. */\n"
"#define KFLRL_THRMEAN(h, e, v) \\\n"
"    ((h)->thr_itime[(e)] > 0.0 \\\n"
"     ? (h)->thr_isum[(size_t)(e) * KFLRL_N_VEHICLES + (size_t)(v)] / \\\n"
"       (h)->thr_itime[(e)] \\\n"
"     : 1.0)\n"
"#else\n"
"#define KFLRL_VEHS(h, e) ((K26AstroVehicle **)0)\n"
"#define KFLRL_COM(h, e, v) ((const double *)0)\n"
"#define KFLRL_PROP(h, e) ((const double *)0)\n"
"#define KFLRL_THRSC(h, e, v) (1.0)\n"
"#define KFLRL_THRMEAN(h, e, v) (1.0)\n"
"#endif\n"
"#if KFLRL_N_EFFECTOR > 0\n"
"#define KFLRL_ENG(h, e) (&(h)->eng[(e)])\n"
"#else\n"
"#define KFLRL_ENG(h, e) ((KflrlEng *)0)\n"
"#endif\n\n", out);
    fputs(
"/* One environment's engagement block: the channels every effector\n"
" * payload published on this step, and the record of which payloads\n"
" * were engaged.\n"
" *\n"
" * `ch` is indexed by payload slot and channel, at the widest\n"
" * effector channel set's stride, so a payload's channels sit where\n"
" * the observation reads them without a per-kind offset table. The\n"
" * whole block is cleared at the top of every step, which is what\n"
" * makes `_engaged` read 0.0 and the rest zero-fill on a step whose\n"
" * body engaged nothing.\n"
" *\n"
" * `fault` is raised by a second engagement of one payload inside one\n"
" * step. The compiler refuses that where both statements name the\n"
" * payload directly; this catches the cases it cannot see statically,\n"
" * such as a statement inside a loop or inside a function called\n"
" * twice, and the step reports it as a fault rather than publishing a\n"
" * result whose value depends on which call ran last.\n"
" *\n"
" * The type is declared whatever the program carries, so the step\n"
" * body's signature has one shape; a program with no effector passes\n"
" * a null pointer through it and its handle carries no block. */\n"
"typedef struct {\n"
"    double  ch[KFLRL_N_PAYLOAD > 0\n"
"               ? KFLRL_N_PAYLOAD * KFLRL_EFF_STRIDE : 1];\n"
"    uint8_t used[KFLRL_N_PAYLOAD > 0 ? KFLRL_N_PAYLOAD : 1];\n"
"    uint8_t fault;\n"
"#if KFLRL_N_SOFTKILL > 0\n"
"    /* What a countermeasure engaged on this step did to another\n"
"     * craft's detection payloads, indexed by (payload, body): the\n"
"     * victim payload whose view is degraded, and the craft it is\n"
"     * degraded about, which is the craft carrying the countermeasure.\n"
"     *\n"
"     * `js` is the jamming-to-signal ratio delivered at that payload's\n"
"     * receiver, `ctr` the range at which that payload detects the\n"
"     * jammer's own emission, and `dec` the confidence degradation a\n"
"     * decoy delivered, in [0, 1). All three are zero when nothing was\n"
"     * engaged, which the clear at the top of every step and at every\n"
"     * reset is what provides: a degradation is an act of one step and\n"
"     * is not in force on the next.\n"
"     *\n"
"     * The engagement runs in the step body, before the world\n"
"     * advances, and the observation is computed after it, so a\n"
"     * degradation written here is still in force when the victim's\n"
"     * detection observe is evaluated later in the same step. That is\n"
"     * the same relation a body state write has. */\n"
"    struct { double js, ctr, dec; } deg[KFLRL_N_DEG];\n"
"#endif\n"
"} KflrlEng;\n\n", out);
    if (m->n_payloads == 0) return 0;

    if (rl_n_pay_kind(m, RL_PAY_INFOSTATE) > 0) {
        fputs(
"/* The clock the information state is pushed and observed on.\n"
" *\n"
" * It is this layer's own rather than the world's: the world's clock\n"
" * returns to the episode baseline at every reset, and a push older\n"
" * than the ring's newest sample is dropped, so pushing on the\n"
" * world's clock would silence the history for the rest of the run.\n"
" * The library needs only that its epochs are consistent and\n"
" * increasing, which this is by construction; the retarded-time\n"
" * solution depends on differences of them and not on their origin.\n"
" *\n"
" * The day index only ever increases, one step per reset, and the\n"
" * seconds restart at each episode's epoch. So two samples of one\n"
" * episode differ by the same arithmetic on the same operands\n"
" * however many episodes have run before it, and the day index tells\n"
" * an epoch of this episode from one of any earlier episode exactly.\n"
" *\n"
" * The clock is what lets the pushes continue across a reset. It is\n"
" * not on its own what keeps the episodes apart: the ring is never\n"
" * emptied, so the retarded time of an early step falls inside\n"
" * retained history and the interpolator will bracket across the\n"
" * reset and return a valid observation with the previous episode's\n"
" * sample blended into it. What rejects that is the day test at the\n"
" * point of publication. */\n"
"static K26AstroEpoch kflrl_info_epoch_(int64_t day, double t_s)\n"
"{\n"
"    K26AstroEpoch t = k26astro_epoch_j2000_tt();\n"
"    t.days_since_J2000 = day;\n"
"    t.seconds_of_day   = 0.0;\n"
"    k26astro_epoch_add_seconds(&t, t_s);\n"
"    return t;\n"
"}\n\n", out);
    }

    /* Teardown per payload slot. The slot decides the destructor
     * because the handle is carried as a void pointer: one array
     * holds every kind, which is what makes the construction and
     * teardown loops one loop each rather than one per kind. */
    fputs("static void kflrl_payload_destroy_(int slot, void *p)\n{\n"
          "    switch (slot) {\n", out);
    for (int p = 0; p < m->n_payloads; p++) {
        fprintf(out, "    case %d: %s((%s *)p); break;\n", p,
                rl_pay_dtor(m->payloads[p].kind),
                rl_pay_ctype(m->payloads[p].kind));
    }
    fputs("    default: break;\n    }\n}\n\n", out);

    /* The kind tag each payload's base must carry. It is written from
     * the tier's own registry and checked against the constructed
     * payload at build, so the grammar's claim to mirror the registry
     * is a mechanical check rather than a comment. */
    fputs("static const uint32_t kflrl_pay_tag_[] = {\n", out);
    for (int p = 0; p < m->n_payloads; p++) {
        fprintf(out, "    (uint32_t)%s,\n", RL_PAY_KIND_[m->payloads[p].kind].tag);
    }
    fputs("};\n\n", out);
    fputs("static const int kflrl_pay_veh_[] = {\n", out);
    for (int p = 0; p < m->n_payloads; p++) {
        fprintf(out, "    %d,\n", m->payloads[p].veh);
    }
    fputs("};\n\n", out);

    if (n_pairs > 0) {
        fputs("/* One (information state, target vehicle) pair per push,\n"
              " * in observe declaration order. */\n"
              "static const int kflrl_track_pay_[] = {\n", out);
        for (int i = 0; i < n_pairs; i++) fprintf(out, "    %d,\n", tpay[i]);
        fputs("};\n\nstatic const int kflrl_track_veh_[] = {\n", out);
        for (int i = 0; i < n_pairs; i++) fprintf(out, "    %d,\n", tveh[i]);
        fputs("};\n\n", out);
    }

    if (m->n_sig > 0) {
        fputs(
"/* One primitive of a detection target's silhouette, in that body's\n"
" * own frame, with the component placement already baked in by the\n"
" * assembly reader. `axis` holds the primitive's own axes as columns\n"
" * in that frame; `half` holds a sphere's radius, a capsule's radius\n"
" * and half length, or a box's three half extents. */\n"
"typedef struct {\n"
"    int    kind;\n"
"    double axis[3][3];\n"
"    double half[3];\n"
"} KflrlSigPrim;\n\n"
"static const KflrlSigPrim kflrl_sig_[] = {\n", out);
        for (int i = 0; i < m->n_sig; i++) {
            const RlSigPrim *sp = &m->sig[i];
            fprintf(out, "    { %d, { { %.17g, %.17g, %.17g },"
                         " { %.17g, %.17g, %.17g },"
                         " { %.17g, %.17g, %.17g } },"
                         " { %.17g, %.17g, %.17g } },\n",
                    sp->kind,
                    sp->axis[0][0], sp->axis[0][1], sp->axis[0][2],
                    sp->axis[1][0], sp->axis[1][1], sp->axis[1][2],
                    sp->axis[2][0], sp->axis[2][1], sp->axis[2][2],
                    sp->half[0], sp->half[1], sp->half[2]);
        }
        fputs("};\n\nstatic const int kflrl_sig_first_[] = {\n", out);
        for (int b = 0; b < m->n_bodies; b++) {
            int first = -1;
            for (int i = 0; i < m->n_sig; i++) {
                if (m->sig[i].body == b && first < 0) first = i;
            }
            fprintf(out, "    %d,\n", first < 0 ? 0 : first);
        }
        fputs("};\n\nstatic const int kflrl_sig_count_[] = {\n", out);
        for (int b = 0; b < m->n_bodies; b++) {
            int n = 0;
            for (int i = 0; i < m->n_sig; i++) {
                if (m->sig[i].body == b) n++;
            }
            fprintf(out, "    %d,\n", n);
        }
        fputs("};\n\n", out);
        if (rl_n_chaff(m) > 0) {
            fputs(
"/* A declared chaff strip count, as the count the cloud statistics\n"
" * routine takes. The key is a scalar like every other, so its value\n"
" * may be an expression or a per-episode draw and need not land inside\n"
" * the integer range the routine's parameter has. Converting a double\n"
" * outside that range is undefined, and what it does here is read as a\n"
" * cloud of no strips, which is indistinguishable from declaring none:\n"
" * a declared figure nothing reads. The compiler refuses a literal\n"
" * outside the range, naming the bound; this is what stops an\n"
" * expression or a draw from reaching the conversion at all.\n"
" *\n"
" * The bound is the routine's own parameter type, and it saturates\n"
" * rather than wrapping, so a count past it is the largest cloud the\n"
" * routine can describe rather than no cloud. */\n"
"static int kflrl_chaff_strips_(double n)\n"
"{\n"
"    if (!(n > 0.0)) return 0;\n"
"    if (n >= 2147483647.0) return 2147483647;\n"
"    return (int)n;\n"
"}\n\n", out);
        }
        fputs(
"/* The area a target presents along a look direction, the direction\n"
" * given in the target's own frame. The three convex primitives have\n"
" * closed forms: a sphere presents its great circle at every aspect,\n"
" * a capsule its two caps as one sphere plus the side of its\n"
" * cylinder, and a box the three face pairs each weighted by how\n"
" * squarely it faces the observer. Overlapping primitives are summed\n"
" * rather than unioned, which overstates the area of a craft whose\n"
" * colliders interpenetrate and is stated rather than corrected: the\n"
" * union of arbitrary primitives has no closed form and an iterative\n"
" * one would not belong on this path.\n"
" *\n"
" * The arithmetic is addition, multiplication and one square root,\n"
" * all correctly rounded under IEEE-754, so the area is a function of\n"
" * the asset bytes and the orientation alone. */\n"
"static double kflrl_sig_area_(int body, K26V3 look_body)\n"
"{\n"
"    double area = 0.0;\n"
"    int first = kflrl_sig_first_[body];\n"
"    int n     = kflrl_sig_count_[body];\n"
"    for (int i = first; i < first + n; i++) {\n"
"        const KflrlSigPrim *s = &kflrl_sig_[i];\n"
"        if (s->kind == 1) {\n"
"            area += K26A_PI * s->half[0] * s->half[0];\n"
"        } else if (s->kind == 2) {\n"
"            double c = look_body.x * s->axis[2][0]\n"
"                     + look_body.y * s->axis[2][1]\n"
"                     + look_body.z * s->axis[2][2];\n"
"            double s2 = 1.0 - c * c;\n"
"            if (s2 < 0.0) s2 = 0.0;\n"
"            area += K26A_PI * s->half[0] * s->half[0]\n"
"                  + 2.0 * s->half[0] * (2.0 * s->half[2]) * sqrt(s2);\n"
"        } else {\n"
"            for (int k = 0; k < 3; k++) {\n"
"                double c = look_body.x * s->axis[k][0]\n"
"                         + look_body.y * s->axis[k][1]\n"
"                         + look_body.z * s->axis[k][2];\n"
"                if (c < 0.0) c = -c;\n"
"                double e0 = s->half[(k + 1) % 3];\n"
"                double e1 = s->half[(k + 2) % 3];\n"
"                area += 4.0 * e0 * e1 * c;\n"
"            }\n"
"        }\n"
"    }\n"
"    return area;\n"
"}\n\n", out);
    }
    return 0;
}

int rl_n_sensed(const RlModel *m)
{
    int n = 0;
    for (int i = 0; i < m->n_observes; i++) {
        if (m->obs_sensor[i] < 0) continue;
        n += rl_observe_base_width(m->observes[i]);
    }
    return n;
}

int rl_emit_sensors(FILE *out, const RlModel *m, KflcDiag *diag)
{
    int n_sensed = rl_n_sensed(m);
    int n_terms = 0, depth = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        n_terms += m->sensors[si].n_terms * rl_observe_base_width(m->observes[i]);
        if (m->sensors[si].depth > depth) depth = m->sensors[si].depth;
    }
    fprintf(out, "#define KFLRL_N_SENSED %d\n", n_sensed);
    fprintf(out, "#define KFLRL_N_STERMS %d\n", n_terms);
    fprintf(out, "#define KFLRL_SENSE_RING %d\n\n", depth);
    if (n_sensed == 0) return 0;

    fputs("/* kind, per-step channel, per-episode channel, three\n"
          " * parameters. The parameters' meaning per kind is\n"
          " * k26sense.h's. */\n"
          "static const struct {\n"
          "    int      kind;\n"
          "    uint16_t ch, ch_ep;\n"
          "    double   p0, p1, p2;\n"
          "} kflrl_sterm_[KFLRL_N_STERMS] = {\n", out);
    int chan = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        const RlSensor *sn = &m->sensors[si];
        int w = rl_observe_base_width(m->observes[i]);
        for (int c = 0; c < w; c++) {
            /* The chain as the library will see it, assembled here so
             * that the library's own precondition check is what passes
             * it. Without a consumer that refusal measures nothing:
             * the rule it enforces, that a term drawing at both
             * cadences holds two channels, would otherwise rest on the
             * two lines below and on nothing else. */
            K26SenseTerm probe[RL_MAX_TERMS];
            for (int t = 0; t < sn->n_terms; t++) {
                const RlSenseTerm *tm = &sn->terms[t];
                int ch = tm->draws_step ? chan++ : K26SENSE_NO_CHANNEL;
                int ce = tm->draws_ep   ? chan++ : K26SENSE_NO_CHANNEL;
                memset(&probe[t], 0, sizeof probe[t]);
                probe[t].kind       = (K26SenseKind)tm->kind;
                probe[t].channel    = (uint16_t)ch;
                probe[t].channel_ep = (uint16_t)ce;
                if (tm->kind == K26SENSE_LATENCY) {
                    probe[t].u.latency.depth = (uint32_t)tm->p0;
                }
                fprintf(out,
                    "    { %d, %uu, %uu, %.17g, %.17g, %.17g },\n",
                    tm->kind, (unsigned)ch, (unsigned)ce,
                    tm->p0, tm->p1, tm->p2);
            }
            uint32_t need = 0;
            K26SenseStatus cst = k26sense_chain_check(
                probe, (uint32_t)sn->n_terms, &need);
            if (cst != K26SENSE_OK) {
                kflc_diag_errorf(diag, sn->node->line,
                    "sensor `%s`: the assembled chain is refused by the "
                    "imperfection layer: %s", sn->name,
                    k26sense_status_str(cst));
                return 1;
            }
        }
    }
    fputs("};\n\n", out);

    fputs("static const int kflrl_sensed_first_[KFLRL_N_SENSED] = {\n", out);
    int first = 0;
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        int w = rl_observe_base_width(m->observes[i]);
        for (int c = 0; c < w; c++) {
            fprintf(out, "    %d,\n", first);
            first += m->sensors[si].n_terms;
        }
    }
    fputs("};\n\nstatic const int kflrl_sensed_count_[KFLRL_N_SENSED] = {\n",
          out);
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        int w = rl_observe_base_width(m->observes[i]);
        for (int c = 0; c < w; c++) {
            fprintf(out, "    %d,\n", m->sensors[si].n_terms);
        }
    }
    /* The observation slot each sensed channel rewrites: the measured
     * half of its observe, in component order. */
    fputs("};\n\nstatic const int kflrl_sensed_slot_[KFLRL_N_SENSED] = {\n",
          out);
    for (int i = 0; i < m->n_observes; i++) {
        int si = m->obs_sensor[i];
        if (si < 0) continue;
        int off = rl_obs_offset(m->observes, i);
        int w = rl_observe_base_width(m->observes[i]);
        for (int c = 0; c < w; c++) fprintf(out, "    %d,\n", off + c);
    }
    fputs("};\n\n", out);
    return 0;
}
