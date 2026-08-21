/* emit_rl_link.c - the datalink: the gate a `source=` puts on the
 * information state's push, and the tables the transfer pass walks. */
#include "emit_rl_internal.h"

/* One gated (detection payload, target body) pair. A program may hold
 * several track pairs whose gate is the same detection against the
 * same target, and one evaluator serves all of them: the verdict is a
 * function of that pair alone. */

typedef struct {
    int src;      /* the detection payload named by `source=` */
    int tgt;      /* the target body its verdict is taken against */
} RlGatePair;

/* The table bounds, each derived from a limit this compiler already
 * carries rather than chosen. One entry per datalink for its carrier's
 * own state, plus one per track its carrier's information state holds,
 * and a track belongs to one information state and a body carries at
 * most one of each, so the tracks across every transmitter cannot
 * outnumber the program's track observes. An edge is one entry offered
 * to one other member of the same community. */

#define RL_LINK_MAX_ENT  (RL_MAX_PAYLOADS + RL_MAX_OBSERVES)
#define RL_LINK_MAX_EDGE (RL_LINK_MAX_ENT * (RL_MAX_PAYLOADS - 1))

/* The gates this program needs, and, per track pair, which of them
 * decides its push. Returns the number of distinct gates. */

static int rl_collect_gates_(const RlModel *m, const int *tpay,
                             const int *tveh, int n_pairs,
                             RlGatePair *out, int *pair_gate, int cap)
{
    int n = 0;

    for (int i = 0; i < n_pairs; i++) {
        pair_gate[i] = -1;
        int src = rl_track_gate_pay(m, tpay[i]);
        if (src < 0) continue;
        int tgt = -1;
        for (int b = 0; b < m->n_bodies; b++) {
            if (rl_veh_slot_of(m, b) == tveh[i]) { tgt = b; break; }
        }
        if (tgt < 0) continue;
        for (int q = 0; q < n; q++) {
            if (out[q].src == src && out[q].tgt == tgt) pair_gate[i] = q;
        }
        if (pair_gate[i] >= 0) continue;
        if (n >= cap) continue;
        out[n].src = src;
        out[n].tgt = tgt;
        pair_gate[i] = n;
        n++;
    }
    return n;
}

/* The entries one datalink offers: its carrier's own state first, then
 * every target its carrier's information state holds a track over, in
 * that program's own track order. The carrier's state is offered
 * because a swarm's members are each other's targets, and a member
 * that told its peers nothing about itself would be the one craft in
 * the world no track could ever cover. */

static int rl_link_entries_(const RlModel *m, int link_pay,
                            const int *tpay, const int *tveh, int n_pairs,
                            int *out_veh, int *out_self, int cap)
{
    const RlPayload *py = &m->payloads[link_pay];
    int n = 0;

    if (cap < 1) return 0;
    out_veh[n]  = py->veh;
    out_self[n] = 1;
    n++;
    for (int i = 0; i < n_pairs; i++) {
        if (tpay[i] != py->info_pay) continue;
        if (n >= cap) break;
        out_veh[n]  = tveh[i];
        out_self[n] = 0;
        n++;
    }
    return n;
}

/* Does this receiving datalink's carrier hold a track over `veh`? An
 * entry for a target the receiver declares no `observe track` over is
 * dropped, so no edge is built for it at all: the declared-target
 * universe bounds shared knowledge exactly as it bounds sensed
 * knowledge, and the bound is applied where it can be seen rather than
 * on the stepping path where it could not. */

static int rl_link_receives_(const RlModel *m, int rx_pay, int veh,
                             const int *tpay, const int *tveh, int n_pairs)
{
    const RlPayload *rx = &m->payloads[rx_pay];

    for (int i = 0; i < n_pairs; i++) {
        if (tpay[i] == rx->info_pay && tveh[i] == veh) return 1;
    }
    return 0;
}

int rl_emit_link_tables(FILE *out, const RlModel *m, KflcDiag *diag)
{
    int tpay[RL_MAX_OBSERVES];
    int tveh[RL_MAX_OBSERVES];
    int n_pairs = rl_track_pairs(m, tpay, tveh, RL_MAX_OBSERVES);
    RlGatePair gate[RL_MAX_OBSERVES];
    int pair_gate[RL_MAX_OBSERVES];
    int n_gates = rl_collect_gates_(m, tpay, tveh, n_pairs, gate,
                                    pair_gate, RL_MAX_OBSERVES);
    int n_links = rl_n_link(m);

    if (n_gates > 0 || n_links > 0) {
        fputs(
"/* ---- The gated information state and the datalink ---------------- *\n"
" *\n"
" * Two surfaces that need each other. An information state fed with\n"
" * truth every sub-advance already knows every declared target's\n"
" * state at light delay, so a peer's report of the same target would\n"
" * arrive with an older timestamp and be dropped by the ring's own\n"
" * drop-older rule without changing any channel. The `source=` key\n"
" * gates the push on a declared detection's verdict, which is what\n"
" * leaves a receiver with something to learn; the datalink is what\n"
" * carries it, at a physical cost and a physical delay. */\n\n",
            out);
    }

    fprintf(out, "#define KFLRL_N_GATE %d\n", n_gates);
    fprintf(out, "#define KFLRL_N_LINK %d\n", n_links);

    /* The datalink's own key slots, so the kernel below and the
     * transfer pass read the parameter store by name rather than by a
     * number repeated in two places. */
    fprintf(out,
        "#define KFLRL_LINK_RATE %d\n"
        "#define KFLRL_LINK_P_TX %d\n"
        "#define KFLRL_LINK_G_TX %d\n"
        "#define KFLRL_LINK_G_RX %d\n"
        "#define KFLRL_LINK_FREQ %d\n"
        "#define KFLRL_LINK_LOSS %d\n"
        "#define KFLRL_LINK_BW %d\n"
        "#define KFLRL_LINK_T_SYS %d\n"
        "#define KFLRL_LINK_NF %d\n"
        "#define KFLRL_LINK_THRESHOLD %d\n",
        RL_PAY_LINK_RATE, RL_PAY_LINK_P_TX, RL_PAY_LINK_G_TX,
        RL_PAY_LINK_G_RX, RL_PAY_LINK_FREQ, RL_PAY_LINK_LOSS,
        RL_PAY_LINK_BW, RL_PAY_LINK_T_SYS, RL_PAY_LINK_NF,
        RL_PAY_LINK_THRESHOLD);

    /* How many offers one edge may carry in flight at once. An offer
     * is in flight from the instant it is broadcast until the instant
     * it arrives, so the depth a program needs is its cadence times
     * the light time to its peer, which is below one for every
     * cadence and separation an engagement bubble holds. Past the
     * depth the environment faults rather than dropping the offer,
     * because a silently dropped transmission is a link that reports
     * a range it does not have. */
    fputs("#define KFLRL_LINK_QCAP 8\n", out);

    if (n_gates > 0) {
        fputs(
"\n/* One gate per (detection payload, target) pair the information\n"
" * state's `source=` names. The body of each is the emitter's own\n"
" * detection block, the same statements in the same order as the\n"
" * `_detected` channel's, so the gate cannot open on a rule that\n"
" * channel disagrees with. */\n", out);
        for (int g = 0; g < n_gates; g++) {
            fprintf(out,
                "static int kflrl_gate_%d_(K26AstroWorld *world,\n"
                "                         const double *payp,\n"
                "                         const KflrlEng *eng)\n"
                "{\n"
                "    (void)eng;\n", g);
            rl_emit_detect_eval(out, m, gate[g].src, gate[g].tgt);
            fputs(
                "        (void)_kfl_snr; (void)_kfl_asp;\n"
                "        return _kfl_det > 0.0;\n"
                "    }\n"
                "}\n\n", out);
        }
        fputs(
"/* Which gate decides one track pair's push, in the pair order the\n"
" * push loop walks. A pair whose information state declares no\n"
" * `source=` has no gate and pushes as it did before the key\n"
" * existed. */\n"
"static int kflrl_track_gate_(K26AstroWorld *world, const double *payp,\n"
"                             const KflrlEng *eng, int pair)\n"
"{\n"
"    switch (pair) {\n", out);
        for (int i = 0; i < n_pairs; i++) {
            if (pair_gate[i] < 0) continue;
            fprintf(out, "    case %d: return kflrl_gate_%d_("
                         "world, payp, eng);\n", i, pair_gate[i]);
        }
        fputs("    default: break;\n"
              "    }\n"
              "    return 1;\n"
              "}\n\n", out);
    }

    if (n_links == 0) {
        fputs("#define KFLRL_N_LINKENT 0\n"
              "#define KFLRL_N_LINKEDGE 0\n"
              "#define KFLRL_N_LINKPAIR 0\n\n", out);
        return 0;
    }

    /* The link table, and the entry and edge tables beneath it. The
     * three are walked in index order and index order is declaration
     * order: transmitters as the program declares them, entries in
     * each transmitter's own track order, receivers as the program
     * declares them. What the drop-older rule keeps where two offers
     * of one target meet is therefore decided by the program and not
     * by which loop the compiler happened to write first. */
    int lpay[RL_MAX_PAYLOADS], lveh[RL_MAX_PAYLOADS];
    int linfo[RL_MAX_PAYLOADS], lnet[RL_MAX_PAYLOADS];
    int n = 0;
    for (int p = 0; p < m->n_payloads && n < RL_MAX_PAYLOADS; p++) {
        if (m->payloads[p].kind != RL_PAY_DATALINK) continue;
        lpay[n]  = p;
        lveh[n]  = m->payloads[p].veh;
        linfo[n] = m->payloads[p].info_pay;
        lnet[n]  = m->payloads[p].net;
        n++;
    }

    /* The entry and edge tables come from the heap, and the reason is
     * arithmetic rather than taste. An edge is one entry offered to
     * one other member of a community, so the edge bound is the entry
     * bound times the payload bound, and at the sizes a swarm needs
     * that product is over a million: three tables of it are tens of
     * megabytes, which is not a stack frame. The bounds themselves are
     * unchanged and so are the two refusals that name them, so a
     * program over either is still refused by name. */
    size_t ent_cap  = (size_t)RL_LINK_MAX_ENT;
    size_t edge_cap = (size_t)RL_LINK_MAX_EDGE;
    size_t pair_cap = (size_t)RL_MAX_PAYLOADS * (size_t)RL_MAX_PAYLOADS;
    int *link_store = (int *)calloc(
        4 * ent_cap + 3 * edge_cap + 3 * pair_cap, sizeof(int));
    if (!link_store) {
        kflc_diag_errorf(diag, 0,
            "out of memory building this program's datalink tables");
        return 1;
    }
    int *ent_veh   = link_store;
    int *ent_self  = ent_veh + ent_cap;
    int *ent_edge0 = ent_self + ent_cap;
    int *ent_nedge = ent_edge0 + ent_cap;
    int *edge_rx   = ent_nedge + ent_cap;
    int *edge_ent  = edge_rx + edge_cap;
    int *edge_pair = edge_ent + edge_cap;
    int *pair_tx   = edge_pair + edge_cap;
    int *pair_rx   = pair_tx + pair_cap;
    int *pair_of   = pair_rx + pair_cap;
    int link_ent0[RL_MAX_PAYLOADS], link_nent[RL_MAX_PAYLOADS];
    int n_ent = 0, n_edge = 0;

    /* The ordered transmitter and receiver pairs of each community,
     * which is what the datalink getter reports over. A pair is not an
     * edge: an edge carries one entry to one receiver and exists only
     * where that receiver declares a track over that entry's target,
     * while a pair is the radio link itself and exists for every
     * ordered pair of one network, whether or not anything is ever
     * offered across it. The budget is the transmitter's own, so the
     * two directions between one pair of craft are two pairs here.
     *
     * Contiguous per transmitter and in declaration order within it,
     * so the transfer pass writes a transmitter's records with one
     * bounded walk and the getter's order is the program's. */
    int link_pair0[RL_MAX_PAYLOADS], link_npair[RL_MAX_PAYLOADS];
    int n_pair = 0;

    for (int i = 0; i < n; i++) {
        for (int r = 0; r < n; r++) {
            pair_of[(size_t)i * RL_MAX_PAYLOADS + (size_t)r] = -1;
        }
    }
    for (int i = 0; i < n; i++) {
        link_pair0[i] = n_pair;
        link_npair[i] = 0;
        for (int r = 0; r < n; r++) {
            if (r == i) continue;
            if (lnet[r] < 0 || lnet[r] != lnet[i]) continue;
            pair_tx[n_pair] = i;
            pair_rx[n_pair] = r;
            pair_of[(size_t)i * RL_MAX_PAYLOADS + (size_t)r] = n_pair;
            n_pair++;
            link_npair[i]++;
        }
    }

    for (int i = 0; i < n; i++) {
        int veh[RL_MAX_OBSERVES + 1], self[RL_MAX_OBSERVES + 1];
        int ne = rl_link_entries_(m, lpay[i], tpay, tveh, n_pairs,
                                  veh, self, RL_MAX_OBSERVES + 1);
        link_ent0[i] = n_ent;
        link_nent[i] = 0;
        for (int j = 0; j < ne; j++) {
            if (n_ent >= RL_LINK_MAX_ENT) {
                kflc_diag_errorf(diag, m->payloads[lpay[i]].line,
                    "more than %d datalink entries in this program",
                    RL_LINK_MAX_ENT);
                free(link_store);
                return 1;
            }
            ent_veh[n_ent]   = veh[j];
            ent_self[n_ent]  = self[j];
            ent_edge0[n_ent] = n_edge;
            ent_nedge[n_ent] = 0;
            for (int r = 0; r < n; r++) {
                if (r == i) continue;
                if (lnet[r] < 0 || lnet[r] != lnet[i]) continue;
                if (!rl_link_receives_(m, lpay[r], veh[j], tpay, tveh,
                                       n_pairs)) {
                    continue;
                }
                if (n_edge >= RL_LINK_MAX_EDGE) {
                    kflc_diag_errorf(diag, m->payloads[lpay[i]].line,
                        "more than %d datalink transfers in this "
                        "program", RL_LINK_MAX_EDGE);
                    free(link_store);
                    return 1;
                }
                edge_rx[n_edge]   = r;
                edge_ent[n_edge]  = n_ent;
                edge_pair[n_edge] =
                    pair_of[(size_t)i * RL_MAX_PAYLOADS + (size_t)r];
                n_edge++;
                ent_nedge[n_ent]++;
            }
            n_ent++;
            link_nent[i]++;
        }
    }

    fprintf(out, "#define KFLRL_N_LINKENT %d\n", n_ent);
    fprintf(out, "#define KFLRL_N_LINKEDGE %d\n", n_edge);
    fprintf(out, "#define KFLRL_N_LINKPAIR %d\n\n", n_pair);

    fputs("static const int kflrl_link_pay_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", lpay[i]);
    fputs("};\n\nstatic const int kflrl_link_veh_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", lveh[i]);
    fputs("};\n\nstatic const int kflrl_link_info_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", linfo[i]);
    fputs("};\n\nstatic const int kflrl_link_ent0_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", link_ent0[i]);
    fputs("};\n\nstatic const int kflrl_link_nent_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", link_nent[i]);
    fputs("};\n\nstatic const int kflrl_link_pair0_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", link_pair0[i]);
    fputs("};\n\nstatic const int kflrl_link_npair_[] = {\n", out);
    for (int i = 0; i < n; i++) fprintf(out, "    %d,\n", link_npair[i]);
    fputs("};\n\n", out);

    if (n_pair > 0) {
        fputs("static const int kflrl_lpair_tx_[] = {\n", out);
        for (int p = 0; p < n_pair; p++)
            fprintf(out, "    %d,\n", pair_tx[p]);
        fputs("};\n\nstatic const int kflrl_lpair_rx_[] = {\n", out);
        for (int p = 0; p < n_pair; p++)
            fprintf(out, "    %d,\n", pair_rx[p]);
        fputs("};\n\n", out);
    }

    if (n_ent > 0) {
        fputs("static const int kflrl_lent_veh_[] = {\n", out);
        for (int j = 0; j < n_ent; j++) fprintf(out, "    %d,\n", ent_veh[j]);
        fputs("};\n\nstatic const int kflrl_lent_self_[] = {\n", out);
        for (int j = 0; j < n_ent; j++) fprintf(out, "    %d,\n", ent_self[j]);
        fputs("};\n\nstatic const int kflrl_lent_edge0_[] = {\n", out);
        for (int j = 0; j < n_ent; j++) fprintf(out, "    %d,\n", ent_edge0[j]);
        fputs("};\n\nstatic const int kflrl_lent_nedge_[] = {\n", out);
        for (int j = 0; j < n_ent; j++) fprintf(out, "    %d,\n", ent_nedge[j]);
        fputs("};\n\n", out);
    }
    if (n_edge > 0) {
        fputs("static const int kflrl_ledge_rx_[] = {\n", out);
        for (int k = 0; k < n_edge; k++) fprintf(out, "    %d,\n", edge_rx[k]);
        fputs("};\n\nstatic const int kflrl_ledge_ent_[] = {\n", out);
        for (int k = 0; k < n_edge; k++) fprintf(out, "    %d,\n", edge_ent[k]);
        /* Which pair an arriving offer belongs to, so the transfer
         * pass records the arrival where the getter reads it without
         * searching for the transmitter that owns the entry. */
        fputs("};\n\nstatic const int kflrl_ledge_pair_[] = {\n", out);
        for (int k = 0; k < n_edge; k++)
            fprintf(out, "    %d,\n", edge_pair[k]);
        fputs("};\n\n", out);
    }
    free(link_store);
    return 0;
}
