/* ---- Environment handle ------------------------------------------ */

/* The context a thrust perturbation is registered with: the handle
 * and which environment it speaks for. Both are fixed at create. */
typedef struct KflrlThrustCtx {
    struct K26RlEnv *h;
    uint32_t         e;
} KflrlThrustCtx;

struct K26RlEnv {
    uint32_t magic;
    uint32_t n_envs;
    uint64_t seed;
    K26RngKey key;
    uint32_t rekey_ordinal;
    uint64_t *seen_seeds;
    uint32_t n_seen, cap_seen;
    K26AstroWorld **worlds;
    /* One vehicle slot per assembly-bearing body per environment,
     * allocated at create and destroyed at destroy. The world holds
     * a non-owning pointer to each, so the handle owns them: this is
     * the one place that can, since it is the one thing that outlives
     * a world and knows when the world goes. */
    K26AstroVehicle **vehicles;
    /* One actuator block per environment: commands written by
     * on_step and the wheel momenta that persist between them. */
    KflrlAct *act;
    /* One latched contact per collidable body per environment. A
     * contact is a fact about a transition, so it is cleared at the
     * start of each one and the first contact in the transition is
     * the one the channels report. The block exists whether or not
     * the program declares a collider, because the observation
     * function reads it either way and one signature is cheaper than
     * a conditional one. */
    KflrlContact *contact;
    KflrlJoin *join;             /* one per environment */
    struct KflrlThrustCtx *thrust_ctx;
    /* Propellant remaining and the centre of mass it puts the craft
     * at, one of each per vehicle per environment. Both are episode
     * state: a burn spends the one and moves the other, and the
     * reset puts both back where construction left them. A vehicle
     * whose assembly declares no tank holds a capacity of zero here
     * and a centre of mass that never moves.
     *
     * The ledger is the closed-system account of what the burns
     * threw overboard: the energy and momentum the propellant left
     * with, and the chemistry that put it there. It is registered
     * into and never read back, so nothing in the simulation is a
     * function of it; one per environment, since the environments
     * are separate systems. */
    double *prop;                /* n_envs * KFLRL_N_VEHICLES, kg */
    double *veh_com;             /* n_envs * KFLRL_N_VEHICLES * 3 */
    double *thr_scale;           /* n_envs * KFLRL_N_VEHICLES */
    /* The step integral of thr_scale and the time it covers, kept
     * for the actuator getter: the fraction is per sub-interval and
     * the getter reports the step, so the mean is what it reports.
     * Written by the step, read by nothing on the stepping path. */
    double *thr_isum;            /* n_envs * KFLRL_N_VEHICLES */
    double *thr_itime;           /* n_envs */
    K26AstroRtConservationLedger **ledger;   /* n_envs */
    K26AstroBody *baseline;      /* n_envs * KFLRL_N_BODIES */
    K26AstroEpoch *baseline_t;   /* n_envs */
    uint32_t *episode;
    uint32_t *steps;             /* transitions in the current episode */
    uint8_t  *ended;
    double   *obs;               /* n_envs * KFLRL_OBS_TOTAL */
    double   *rew;               /* n_envs * KFLRL_N_AGENTS, env major */
    uint32_t *flags;
    uint16_t *fault;
    double   *dr_vals;           /* n_envs * KFLRL_N_REC */
    double   *wscal;             /* n_envs * KFLRL_N_WSCAL, world
                                  * scalars captured at create */
    double   *scratch;           /* KFLRL_OBS_TOTAL */
#if KFLRL_N_SENSED > 0
    /* The imperfection layer. `sterm` is the resolved model
     * chain, built once at create because the bias walk's two
     * coefficients follow from the control period; `sense` is
     * one state block per sensed channel per environment, and
     * `sense_ring` is the delay storage those blocks point
     * into. All three are sized at create, so the step path
     * allocates nothing. */
    K26SenseTerm  *sterm;
    K26SenseState *sense;
    double        *sense_ring;
#endif
#if KFLRL_N_PAYLOAD > 0
    /* The defense payloads, one set per environment, constructed
     * once in the world prefix and destroyed with the handle;
     * `payp` is the parameter store the evaluators read, since the
     * tier's handles publish no accessors for what they were built
     * with and a per-episode draw has to reach them somewhere.
     *
     * The information state is pushed and observed on a clock of
     * this layer's own, in two parts. `info_t` is the seconds
     * elapsed since the current episode's epoch and starts each
     * episode at zero; `info_day` is a day index that only ever
     * increases, one step per reset.
     *
     * The split is what makes an episode reproduce its
     * predecessor bit for bit. The history ring drops a push
     * older than its newest sample and the library offers no way
     * to empty it, so the epochs must increase across a reset;
     * but the retarded-time solution is a function of differences
     * between them, and a difference taken between two seconds
     * counts that grew with the run would round differently in
     * each episode. Putting the growth in the day index and the
     * within-episode time in the seconds field leaves every
     * difference inside an episode the same arithmetic on the
     * same operands in every episode.
     *
     * The clock alone does not keep the episodes apart, and the
     * observation path does not rely on it to. The ring is never
     * emptied, so an early step's retarded time falls inside
     * retained history and the interpolator brackets across the
     * reset and returns a valid observation with the previous
     * episode's sample blended in. What rejects that is the day
     * index test where the channels are published: an observation
     * whose retarded epoch falls before this episode's epoch day
     * is published as unavailable, because reporting it would make
     * episode k+1 a function of episode k. */
    void   **payloads;
#@ The per-unit handles a swarm release needs, one slot per payload
#@ and null for every payload that is not a swarm. A swarm's
#@ penetration analysis is a question about one arriving unit, and
#@ the unit is not the declared projectile, so the handle carrying
#@ the divided geometry is built once beside the payload rather
#@ than derived on the stepping path.
    void   **payu;
    double  *payp;
    double  *info_t;
    int64_t *info_day;
#endif
#if KFLRL_N_EFFECTOR > 0
    /* One engagement block per environment, holding the channels
     * this step's engagements published and the record of which
     * payloads were engaged. Sized at create like every other
     * per-environment store, so the stepping path writes into
     * memory that already exists; clearing it at the top of a step
     * is what makes an unengaged step read zero. */
    KflrlEng *eng;
#endif
    double    control_dt;
    uint32_t  horizon;
    /* The declared subdivision of a control period, and the interval
     * each sub-advance takes. The last sub-advance of a transition
     * takes the remainder instead, so the advanced time sums to
     * control_dt exactly however the division rounded. */
    uint32_t  substeps;
    double    restitution;       /* contact bounce, 0 for arrest */
    double    friction;
    double    sub_dt;
    double    act_lo[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];
    double    act_hi[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];
    uint32_t  act_arity[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];
    uint16_t  act_kind[KFLRL_ACT_TOTAL ? KFLRL_ACT_TOTAL : 1];
    uint8_t  *spec;
    uint32_t  spec_len;
#if KFLRL_N_PLAN > 0
    /* The scratch a plan write needs, allocated with the handle and
     * sized from the widest plan this program declares. An episode
     * end is an output point rather than a place to start
     * allocating, and a handle is single-threaded, so one set of
     * buffers per handle is what the write needs. */
    K26RlRefKnot  *plan_slot;
    K26RlRefKnot  *plan_scratch;
    unsigned char *plan_buf;
#endif
    K26RlEpisodeWriter *writer;
    K26RlTap *tap;               /* the telemetry ring, when enabled */
    uint8_t   at_boundary;
};

static int kflrl_live_(const K26RlEnv *h)
{
    return h && h->magic == KFLRL_MAGIC;
}

/* k26rl_status_str is the episode library's; nothing else in this
 * translation unit calls it, so this anchor makes the link pull the
 * archive member in and the shared object export the full frozen
 * surface. */
static const void *const kflrl_keep_status_str_
    __attribute__((used)) = (const void *)&k26rl_status_str;

#@ The imperfection layer's two entries. Both are emitted whatever
#@ the program declares, so the call sites need no conditional; with
#@ no sensor they compile to nothing.
/* Per-episode setup: the draws a chain takes once, and the state a
 * step expects to find. The value handed in is the channel's true
 * value at the boundary, which fills a delay ring and primes a
 * dropout hold. */
static void kflrl_sense_reset_(K26RlEnv *h, uint32_t e, uint32_t ep,
                               const double *obs_v)
{
    (void)h; (void)e; (void)ep; (void)obs_v;
#if KFLRL_N_SENSED > 0
    for (int c = 0; c < KFLRL_N_SENSED; c++) {
        (void)k26sense_chain_reset(
            h->sterm + kflrl_sensed_first_[c],
            (uint32_t)kflrl_sensed_count_[c],
            &h->sense[(size_t)e * KFLRL_N_SENSED + (size_t)c],
            h->key, K26SENSE_CLASS_SENSOR, e, ep,
            obs_v[kflrl_sensed_slot_[c]]);
    }
#endif
}

/* One transition's corruption, in place over the measured half. The
 * draw index is the transition index within the episode, so any
 * step's noise is addressable without producing the step before it.
 * No allocation, no I/O, and a fixed number of draws per term. */
static void kflrl_sense_apply_(K26RlEnv *h, uint32_t e, uint32_t ep,
                               uint32_t step, double *obs_v)
{
    (void)h; (void)e; (void)ep; (void)step; (void)obs_v;
#if KFLRL_N_SENSED > 0
    for (int c = 0; c < KFLRL_N_SENSED; c++) {
        double v = obs_v[kflrl_sensed_slot_[c]];
        (void)k26sense_chain_apply(
            h->sterm + kflrl_sensed_first_[c],
            (uint32_t)kflrl_sensed_count_[c],
            &h->sense[(size_t)e * KFLRL_N_SENSED + (size_t)c],
            h->key, K26SENSE_CLASS_SENSOR, e, ep, step, v, &v);
        obs_v[kflrl_sensed_slot_[c]] = v;
    }
#endif
}

#if KFLRL_N_PORTS > 1
/* Form the joint body from a captured pair, at the impact
 * configuration the sweep computed.
 *
 * Mass is summed. Linear momentum is conserved, so the pair leaves
 * at the velocity of their common centre of mass. Angular momentum
 * about that centre is conserved too, which is what decides the
 * joint rate: each body brings its own spin and the moment of its
 * own motion about the joint centre, and the joint inertia turns the
 * total into a rate. The inertia is the sum of the two tensors, each
 * carried to the joint centre by the parallel-axis theorem.
 *
 * Nothing here allocates: the tensors are set in place through the
 * vehicle's own setters, which recompute the inverse where they
 * stand. */
static K26M3 kflrl_join_world_inertia_(const K26AstroVehicle *v,
                                       K26Quat q)
{
    K26M3 out;
    memset(&out, 0, sizeof out);
    const K26AstroAttitudeStateExt *x = v
        ? k26astro_vehicle_attitude_ext((K26AstroVehicle *)v) : NULL;
    if (!x) return out;
    K26V3 e0 = k26m3d_quat_rotate_v3(q, k26m3d_v3(1, 0, 0));
    K26V3 e1 = k26m3d_quat_rotate_v3(q, k26m3d_v3(0, 1, 0));
    K26V3 e2 = k26m3d_quat_rotate_v3(q, k26m3d_v3(0, 0, 1));
    K26M3 R;
    R.m[0][0] = e0.x; R.m[0][1] = e1.x; R.m[0][2] = e2.x;
    R.m[1][0] = e0.y; R.m[1][1] = e1.y; R.m[1][2] = e2.y;
    R.m[2][0] = e0.z; R.m[2][1] = e1.z; R.m[2][2] = e2.z;
    K26M3 tmp;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double a = 0.0;
            for (int k = 0; k < 3; k++) a += R.m[i][k] * x->inertia.m[k][j];
            tmp.m[i][j] = a;
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double a = 0.0;
            for (int k = 0; k < 3; k++) a += tmp.m[i][k] * R.m[j][k];
            out.m[i][j] = a;
        }
    }
    return out;
}

static K26V3 kflrl_m3_mul_(K26M3 m, K26V3 v)
{
    return k26m3d_v3(m.m[0][0]*v.x + m.m[0][1]*v.y + m.m[0][2]*v.z,
                     m.m[1][0]*v.x + m.m[1][1]*v.y + m.m[1][2]*v.z,
                     m.m[2][0]*v.x + m.m[2][1]*v.y + m.m[2][2]*v.z);
}

/* Make the joined pair one body again, at the end of a sub-advance
 * in which the two were integrated separately.
 *
 * Configuration first: the follower is placed from the carrier at
 * the offset and relative attitude the capture froze, which is what
 * makes the pair rigid rather than merely close.
 *
 * Motion second, and this is why the pair is projected rather than
 * slaved: both bodies keep their own mass and their own forces, so
 * a thruster on either of them accelerates the pair. Their momenta
 * are summed and the sum is turned back into one rigid motion. Mass
 * is summed; linear momentum gives the joint velocity; angular
 * momentum about the joint centre of mass, each body contributing
 * its own spin and the moment of its own motion, gives the joint
 * rate against the joint inertia, which is the two tensors summed
 * about that centre by the parallel-axis theorem. Both quantities
 * are conserved exactly across the projection; the relative kinetic
 * energy is not, which is what a capture latch does. */
static void kflrl_join_impose_(K26RlEnv *h, uint32_t e,
                               const K26AstroPos *cref)
{
    KflrlJoin *j = &h->join[e];
    if (!j->active) return;
    K26AstroBody *cb = k26astro_world_body_at(h->worlds[e],
        kflrl_body_idx_[kflrl_vehicle_body_[j->carrier]]);
    K26AstroBody *fb = k26astro_world_body_at(h->worlds[e],
        kflrl_body_idx_[kflrl_vehicle_body_[j->follower]]);
    if (!cb || !fb) return;

    fb->attitude = k26m3d_quat_norm(k26m3d_quat_mul(cb->attitude,
                                                    j->rel));
    K26V3 arm  = k26m3d_quat_rotate_v3(cb->attitude, j->offset);
    K26V3 cpos = k26astro_pos_sub(&cb->pos, cref);
    K26V3 fpos = k26astro_pos_sub(&fb->pos, cref);
    k26astro_pos_add(&fb->pos,
        k26m3d_v3(cpos.x + arm.x - fpos.x, cpos.y + arm.y - fpos.y,
                  cpos.z + arm.z - fpos.z));
    fpos = k26m3d_v3(cpos.x + arm.x, cpos.y + arm.y, cpos.z + arm.z);
    const double *cc_ = KFLRL_COM(h, e, j->carrier);
    const double *cf_ = KFLRL_COM(h, e, j->follower);
    K26V3 rc = k26m3d_quat_rotate_v3(cb->attitude,
        k26m3d_v3(cc_[0], cc_[1], cc_[2]));
    K26V3 rf = k26m3d_quat_rotate_v3(fb->attitude,
        k26m3d_v3(cf_[0], cf_[1], cf_[2]));
    K26V3 comc = k26m3d_v3(cpos.x + rc.x, cpos.y + rc.y, cpos.z + rc.z);
    K26V3 comf = k26m3d_v3(fpos.x + rf.x, fpos.y + rf.y, fpos.z + rf.z);
    double mc = cb->mass > 0.0 ? cb->mass : 0.0;
    double mf = fb->mass > 0.0 ? fb->mass : 0.0;
    double M = mc + mf;
    if (!(M > 0.0)) return;
    K26V3 R = k26m3d_v3((mc * comc.x + mf * comf.x) / M,
                        (mc * comc.y + mf * comf.y) / M,
                        (mc * comc.z + mf * comf.z) / M);
    K26V3 V = k26m3d_v3((mc * cb->vel.x + mf * fb->vel.x) / M,
                        (mc * cb->vel.y + mf * fb->vel.y) / M,
                        (mc * cb->vel.z + mf * fb->vel.z) / M);
    K26M3 Ic = kflrl_join_world_inertia_(
        h->vehicles[(size_t)e * KFLRL_N_VEHICLES + j->carrier],
        cb->attitude);
    K26M3 If = kflrl_join_world_inertia_(
        h->vehicles[(size_t)e * KFLRL_N_VEHICLES + j->follower],
        fb->attitude);
    K26V3 wc = k26m3d_quat_rotate_v3(cb->attitude, cb->omega);
    K26V3 wf = k26m3d_quat_rotate_v3(fb->attitude, fb->omega);
    K26V3 dc = k26m3d_v3(comc.x - R.x, comc.y - R.y, comc.z - R.z);
    K26V3 df = k26m3d_v3(comf.x - R.x, comf.y - R.y, comf.z - R.z);
    K26V3 Lc = kflrl_m3_mul_(Ic, wc);
    K26V3 Lf = kflrl_m3_mul_(If, wf);
    K26V3 mc_v = k26m3d_v3_cross(dc,
        k26m3d_v3(mc * (cb->vel.x - V.x), mc * (cb->vel.y - V.y),
                  mc * (cb->vel.z - V.z)));
    K26V3 mf_v = k26m3d_v3_cross(df,
        k26m3d_v3(mf * (fb->vel.x - V.x), mf * (fb->vel.y - V.y),
                  mf * (fb->vel.z - V.z)));
    K26V3 L = k26m3d_v3(Lc.x + Lf.x + mc_v.x + mf_v.x,
                        Lc.y + Lf.y + mc_v.y + mf_v.y,
                        Lc.z + Lf.z + mc_v.z + mf_v.z);
    K26M3 J;
    double dcc = k26m3d_v3_dot(dc, dc), dff = k26m3d_v3_dot(df, df);
    double dcv[3] = { dc.x, dc.y, dc.z }, dfv[3] = { df.x, df.y, df.z };
    for (int i = 0; i < 3; i++) {
        for (int k = 0; k < 3; k++) {
            double kron = (i == k) ? 1.0 : 0.0;
            J.m[i][k] = Ic.m[i][k] + If.m[i][k]
                      + mc * (kron * dcc - dcv[i] * dcv[k])
                      + mf * (kron * dff - dfv[i] * dfv[k]);
        }
    }
    /* The joint rate is that momentum against the joint inertia.
     * The inverse is written out here because the tensor is the
     * pair's and no vehicle carries it. A singular one means the
     * pair has no rotational answer, and the projection is left
     * undone rather than continued with a fabricated one. */
    K26M3 Jinv;
    {
        double c00 = J.m[1][1]*J.m[2][2] - J.m[1][2]*J.m[2][1];
        double c01 = J.m[1][2]*J.m[2][0] - J.m[1][0]*J.m[2][2];
        double c02 = J.m[1][0]*J.m[2][1] - J.m[1][1]*J.m[2][0];
        double det = J.m[0][0]*c00 + J.m[0][1]*c01 + J.m[0][2]*c02;
        if (!(det > 0.0) && !(det < 0.0)) return;
        double id = 1.0 / det;
        Jinv.m[0][0] = c00 * id;
        Jinv.m[1][0] = c01 * id;
        Jinv.m[2][0] = c02 * id;
        Jinv.m[0][1] = (J.m[0][2]*J.m[2][1] - J.m[0][1]*J.m[2][2]) * id;
        Jinv.m[1][1] = (J.m[0][0]*J.m[2][2] - J.m[0][2]*J.m[2][0]) * id;
        Jinv.m[2][1] = (J.m[0][1]*J.m[2][0] - J.m[0][0]*J.m[2][1]) * id;
        Jinv.m[0][2] = (J.m[0][1]*J.m[1][2] - J.m[0][2]*J.m[1][1]) * id;
        Jinv.m[1][2] = (J.m[0][2]*J.m[1][0] - J.m[0][0]*J.m[1][2]) * id;
        Jinv.m[2][2] = (J.m[0][0]*J.m[1][1] - J.m[0][1]*J.m[1][0]) * id;
    }
    K26V3 W = kflrl_m3_mul_(Jinv, L);
    K26V3 vc = k26m3d_v3_cross(W, dc);
    K26V3 vf = k26m3d_v3_cross(W, df);
    cb->vel = k26m3d_v3(V.x + vc.x, V.y + vc.y, V.z + vc.z);
    fb->vel = k26m3d_v3(V.x + vf.x, V.y + vf.y, V.z + vf.z);
    cb->omega = k26m3d_quat_rotate_v3(k26m3d_quat_conj(cb->attitude), W);
    fb->omega = k26m3d_quat_rotate_v3(k26m3d_quat_conj(fb->attitude), W);
}

/* Form the join at the impact configuration the sweep computed: put
 * both bodies there, freeze the relative configuration, and project
 * the motion. The carrier is the heavier of the two, since one of
 * them has to hold the pair's pose and the joint centre of mass
 * lies nearer that one; a tie goes to the lower vehicle slot, which
 * is declaration order, so the choice is never an accident. */
static void kflrl_join_form_(K26RlEnv *h, uint32_t e,
                             const K26AstroCollBody *cbody,
                             int va, int vb, double time, double rem,
                             const K26AstroPos *cref)
{
    KflrlJoin *j = &h->join[e];
    if (j->active) return;
    K26AstroBody *ba = k26astro_world_body_at(h->worlds[e],
        kflrl_body_idx_[kflrl_vehicle_body_[va]]);
    K26AstroBody *bb = k26astro_world_body_at(h->worlds[e],
        kflrl_body_idx_[kflrl_vehicle_body_[vb]]);
    if (!ba || !bb) return;
    int carrier = va, follower = vb;
    if (bb->mass > ba->mass || (bb->mass == ba->mass && vb < va)) {
        carrier = vb; follower = va;
    }
    K26AstroBody *cb = (carrier == va) ? ba : bb;
    K26AstroBody *fb = (carrier == va) ? bb : ba;
    const K26AstroCollBody *cc = &cbody[carrier];
    const K26AstroCollBody *cf = &cbody[follower];
    K26V3 pc = k26m3d_v3(cc->pos0.x + (cc->pos1.x - cc->pos0.x) * time,
                         cc->pos0.y + (cc->pos1.y - cc->pos0.y) * time,
                         cc->pos0.z + (cc->pos1.z - cc->pos0.z) * time);
    K26V3 pf = k26m3d_v3(cf->pos0.x + (cf->pos1.x - cf->pos0.x) * time,
                         cf->pos0.y + (cf->pos1.y - cf->pos0.y) * time,
                         cf->pos0.z + (cf->pos1.z - cf->pos0.z) * time);
    /* The capture answers at the contact instant, part way through
     * the sub-advance, and the world stands at its end, which is
     * where the observation is taken. The pair's own configuration is
     * the one the contact found, and the offset below holds it; what
     * the capture changed of the pair's motion is the carrier's
     * velocity, from the one it arrived with to the joint one the
     * projection below gives it, so the carrier is moved by the
     * displacement that change makes over the rest of the
     * sub-advance and keeps the gravity and thrust of it. Left at the
     * contact configuration instead, the pair would sit at a moment
     * the rest of the world had left, and every relative observe
     * taken from it would carry the travel it never made.
     *
     * The follower is placed from the carrier below, so it needs no
     * correction of its own. */
    K26V3 vcs = k26m3d_v3(cc->vel0.x + (cc->vel1.x - cc->vel0.x) * time,
                          cc->vel0.y + (cc->vel1.y - cc->vel0.y) * time,
                          cc->vel0.z + (cc->vel1.z - cc->vel0.z) * time);
    double mcj = cb->mass > 0.0 ? cb->mass : 0.0;
    double mfj = fb->mass > 0.0 ? fb->mass : 0.0;
    double mtj = mcj + mfj;
    K26V3 vjn = cb->vel;
    if (mtj > 0.0) {
        vjn = k26m3d_v3((mcj * cb->vel.x + mfj * fb->vel.x) / mtj,
                        (mcj * cb->vel.y + mfj * fb->vel.y) / mtj,
                        (mcj * cb->vel.z + mfj * fb->vel.z) / mtj);
    }
    k26astro_pos_add(&cb->pos, k26m3d_v3((vjn.x - vcs.x) * rem,
                                         (vjn.y - vcs.y) * rem,
                                         (vjn.z - vcs.z) * rem));
    K26Quat qc_conj = k26m3d_quat_conj(cb->attitude);
    j->carrier  = carrier;
    j->follower = follower;
    j->rel      = k26m3d_quat_norm(k26m3d_quat_mul(qc_conj,
                                                   fb->attitude));
    j->offset   = k26m3d_quat_rotate_v3(qc_conj,
        k26m3d_v3(pf.x - pc.x, pf.y - pc.y, pf.z - pc.z));
    j->active   = 1;
    kflrl_join_impose_(h, e, cref);
}
#endif

/* Reset one environment in place to episode `ep`: restore the
 * body-state and epoch baseline captured at create, clear the
 * integrator transients a step leaves behind, re-seed the world's
 * runtime noise stream, apply the episode's draws, and recompute
 * the initial observation. No allocation on this path. */
#if KFLRL_N_PAYLOAD > 0
/* One sample per (information state, target) pair at the clock's
 * current reading. Called once at the episode epoch before any
 * stepping, which is what allocates each target's ring in the
 * prefix and what gives the interpolator its first sample, and once
 * per sub-advance thereafter, so the history is finer than the
 * light-time lag rather than coarser than it. The rings exist from
 * the first call onward, so no call after the first allocates. */
static void kflrl_info_push_(K26RlEnv *h, uint32_t e)
{
#if KFLRL_N_TRACKPAIR > 0
    K26AstroEpoch t = kflrl_info_epoch_(h->info_day[e], h->info_t[e]);
    K26AstroPos origin = k26astro_pos_zero();
    for (int i = 0; i < KFLRL_N_TRACKPAIR; i++) {
        K26AstroInfostate *is = (K26AstroInfostate *)
            h->payloads[(size_t)e * KFLRL_N_PAYLOAD
                        + kflrl_track_pay_[i]];
        int tv_slot = kflrl_track_veh_[i];
        K26AstroVehicle *tv =
            h->vehicles[(size_t)e * KFLRL_N_VEHICLES + tv_slot];
        if (!is || !tv) continue;
        const K26AstroBody *tb = k26astro_world_body_at(
            h->worlds[e],
            kflrl_body_idx_[kflrl_vehicle_body_[tv_slot]]);
        if (!tb) continue;
        k26astro_infostate_target_push(
            is, tv, t, k26astro_pos_sub(&tb->pos, &origin), tb->vel);
    }
#else
    (void)h; (void)e;
#endif
}
#endif

static void kflrl_reset_env_(K26RlEnv *h, uint32_t e, uint32_t ep)
{
    K26AstroWorld *w = h->worlds[e];
#if KFLRL_N_VEHICLES > 0
    /* Actuator state is episode state: a wheel's stored momentum and
     * every standing command belong to the episode that produced
     * them, so both are cleared here with the rest of the baseline.
     * A wheel left spun up across a reset would make episode k+1 a
     * function of episode k, which is exactly what the identity
     * triple says it is not. */
    memset(&h->act[e], 0, sizeof h->act[e]);
#endif
    /* The latched contact belongs to the episode that produced it,
     * so it is cleared here with the rest of the baseline. */
    memset(&h->contact[(size_t)e * KFLRL_N_CONTACT], 0,
           sizeof(KflrlContact) * KFLRL_N_CONTACT);
    /* A pair joined by a capture belongs to its episode too. The
     * join holds no property of either craft, only the pose that
     * ties them, so dropping it is the whole of undoing it. */
    memset(&h->join[e], 0, sizeof h->join[e]);
#if KFLRL_N_BODIES > 0
    K26AstroBody *b0 = k26astro_world_body_at(w, 0);
    if (b0) {
        memcpy(b0, h->baseline + (size_t)e * KFLRL_N_BODIES,
               sizeof(K26AstroBody) * KFLRL_N_BODIES);
    }
#endif
#if KFLRL_N_PROPELLANT > 0
    /* The tank, and everything that hangs off it. The body baseline
     * above carries the mass, but the centre of mass and the inertia
     * live on the vehicle rather than on the body and would survive
     * a reset untouched: a craft that ended an episode empty would
     * start the next one with a full tank and the mass properties of
     * an empty one, and episode k+1 would be a function of episode
     * k. They are restored by recomputing at the declared capacity,
     * which is the same call construction made, so the restored
     * values are the constructed ones exactly rather than nearly. */
    for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
        double cap = kflrl_veh_prop_cap_[vi];
        KFLRL_THRSC(h, e, vi) = 1.0;
        if (!(cap > 0.0)) continue;
        h->prop[(size_t)e * KFLRL_N_VEHICLES + vi] = cap;
        kflrl_apply_props_(
            h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi],
            k26astro_world_body_at(
                w, kflrl_body_idx_[kflrl_vehicle_body_[vi]]),
            vi, cap, KFLRL_COM(h, e, vi));
    }
#endif
    K26AstroGravState *g = k26astro_world_grav(w);
    if (g) {
        g->t = h->baseline_t[e];
        g->dt_last = 0.0;
        g->ias15_dt_last = 0.0;
        k26astro_grav_ias15_reset(g);
    }
    /* The world's own stateful generator is seeded from a counter
     * draw at this environment's and episode's coordinates,
     * rather than from the governing seed. Handing every world
     * the same seed would correlate any model that drew from
     * it across the whole vectorised set, and would repeat the
     * same sequence in every episode. */
    (void)k26astro_world_set_seed(
        w, kflrl_world_seed_(h->key, e, ep));
#if KFLRL_N_REC > 0
    kflrl_apply_draws_(w, h->key, e, ep,
                       h->dr_vals + (size_t)e * KFLRL_N_REC, NULL,
                       KFLRL_PAYP(h, e));
#else
    kflrl_apply_draws_(w, h->key, e, ep, NULL, NULL, KFLRL_PAYP(h, e));
#endif
#if KFLRL_N_PAYLOAD > 0
    /* The new episode's epoch: a whole day past the last sample
     * pushed, and past any day the last episode itself ran into, so
     * the seed is strictly newer than what the ring holds and is
     * appended rather than replacing it. */
    h->info_day[e] += 1 + (int64_t)floor(h->info_t[e] / 86400.0);
    h->info_t[e]    = 0.0;
    kflrl_info_push_(h, e);
#endif
    h->episode[e] = ep;
    h->steps[e]   = 0;
    h->ended[e]   = 0;
    for (int a = 0; a < KFLRL_N_AGENTS; a++) {
        h->rew[(size_t)e * KFLRL_N_AGENTS + a] = 0.0;
    }
    h->fault[e]   = 0;
#if KFLRL_N_EFFECTOR > 0
    /* Nothing has been engaged in the new episode, so every effector
     * channel reads zero in its initial observation. */
    memset(&h->eng[e], 0, sizeof h->eng[e]);
#endif
    kflrl_observe_(w, h->obs + (size_t)e * KFLRL_OBS_TOTAL,
                   &h->contact[(size_t)e * KFLRL_N_CONTACT],
                   &h->join[e], KFLRL_PAYH(h, e), KFLRL_PAYP(h, e),
                   KFLRL_VEHS(h, e), KFLRL_PROP(h, e),
                   KFLRL_COM(h, e, 0), KFLRL_INFODAY(h, e),
                   KFLRL_INFOT(h, e), KFLRL_ENG(h, e), 0.0);
    kflrl_sense_reset_(h, e, ep,
                       h->obs + (size_t)e * KFLRL_OBS_TOTAL);
}

/* ---- Spec blob ---------------------------------------------------- */

static void kflrl_put_u16_(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)(v >> 8);
}

static void kflrl_put_u32_(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static void kflrl_put_u64_(uint8_t *p, uint64_t v)
{
    kflrl_put_u32_(p, (uint32_t)(v & 0xffffffffu));
    kflrl_put_u32_(p + 4, (uint32_t)(v >> 32));
}

static uint64_t kflrl_f64_bits_(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof u);
    return u;
}

/* One TLV entry; p advances past it. NULL p sizes only. */
static uint32_t kflrl_tlv_(uint8_t **p, uint16_t tag, uint32_t len,
                           const uint8_t *val)
{
    if (*p) {
        kflrl_put_u16_(*p, tag);
        kflrl_put_u32_(*p + 2, len);
        if (len) memcpy(*p + 6, val, len);
        *p += 6 + len;
    }
    return 6 + len;
}

static uint32_t kflrl_spec_write_(uint8_t *buf, const K26RlEnv *h)
{
    uint8_t *p = buf;
    uint8_t v[20];
    uint32_t total = 0;
    kflrl_put_u32_(v, K26RL_ABI_VERSION);
    total += kflrl_tlv_(&p, K26RL_TAG_ABI_VERSION, 4, v);
    kflrl_put_u32_(v, 0x01020304u);
    total += kflrl_tlv_(&p, K26RL_TAG_ENDIAN_PROBE, 4, v);
    kflrl_put_u32_(v, (uint32_t)KFLRL_N_AGENTS);
    total += kflrl_tlv_(&p, K26RL_TAG_AGENT_COUNT, 4, v);
    kflrl_put_u32_(v, h->n_envs);
    total += kflrl_tlv_(&p, K26RL_TAG_N_ENVS, 4, v);
    kflrl_put_u64_(v, kflrl_f64_bits_(h->control_dt));
    total += kflrl_tlv_(&p, K26RL_TAG_CONTROL_DT, 8, v);
    kflrl_put_u32_(v, h->substeps);
    total += kflrl_tlv_(&p, K26RL_TAG_SUBSTEPS, 4, v);
    kflrl_put_u32_(v, h->horizon);
    total += kflrl_tlv_(&p, K26RL_TAG_HORIZON, 4, v);
    kflrl_put_u32_(v, KFLRL_OBS_TOTAL);
    total += kflrl_tlv_(&p, K26RL_TAG_OBS_TOTAL, 4, v);
    kflrl_put_u32_(v, KFLRL_ACT_TOTAL);
    total += kflrl_tlv_(&p, K26RL_TAG_ACT_TOTAL, 4, v);
    for (uint32_t a = 0; a < (uint32_t)KFLRL_N_AGENTS; a++) {
        kflrl_put_u32_(v, a);
        kflrl_put_u32_(v + 4, kflrl_agent_slices_[a][0]);
        kflrl_put_u32_(v + 8, kflrl_agent_slices_[a][1]);
        total += kflrl_tlv_(&p, K26RL_TAG_AGENT_OBS_SLICE, 12, v);
        kflrl_put_u32_(v + 4, kflrl_agent_slices_[a][2]);
        kflrl_put_u32_(v + 8, kflrl_agent_slices_[a][3]);
        total += kflrl_tlv_(&p, K26RL_TAG_AGENT_ACT_SLICE, 12, v);
    }
    for (uint32_t i = 0; i < KFLRL_ACT_TOTAL; i++) {
        kflrl_put_u32_(v, i);
        kflrl_put_u64_(v + 4, kflrl_f64_bits_(h->act_lo[i]));
        kflrl_put_u64_(v + 12, kflrl_f64_bits_(h->act_hi[i]));
        total += kflrl_tlv_(&p, K26RL_TAG_ACT_BOUNDS, 20, v);
    }
    for (uint32_t i = 0; i < KFLRL_ACT_TOTAL; i++) {
        kflrl_put_u32_(v, i);
        kflrl_put_u16_(v + 4, h->act_kind[i]);
        if (h->act_kind[i] == K26RL_ACT_KIND_DISCRETE) {
            kflrl_put_u32_(v + 6, h->act_arity[i]);
            total += kflrl_tlv_(&p, K26RL_TAG_ACT_KIND, 10, v);
        } else {
            total += kflrl_tlv_(&p, K26RL_TAG_ACT_KIND, 6, v);
        }
    }
#if KFLRL_OBS_TOTAL > 0
    for (uint32_t i = 0; i < KFLRL_OBS_TOTAL; i++) {
        uint8_t nv[4 + KFLRL_OBS_NAME_MAX];
        uint32_t nl = (uint32_t)strlen(kflrl_obs_names_[i]);
        /* The compiler refuses a name that would not fit, so this
         * cannot truncate; it is the belt on the braces. */
        if (nl > KFLRL_OBS_NAME_MAX) nl = KFLRL_OBS_NAME_MAX;
        kflrl_put_u32_(nv, i);
        memcpy(nv + 4, kflrl_obs_names_[i], nl);
        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_NAME, 4 + nl, nv);
    }
    for (uint32_t i = 0; i < KFLRL_OBS_TOTAL; i++) {
        kflrl_put_u32_(v, i);
        kflrl_put_u16_(v + 4, K26RL_OBS_KIND_VECTOR);
        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_KIND, 6, v);
    }
    for (uint32_t i = 0; i < KFLRL_OBS_TOTAL; i++) {
        kflrl_put_u32_(v, i);
        kflrl_put_u16_(v + 4, kflrl_obs_modes_[i]);
        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_MODE, 6, v);
        kflrl_put_u16_(v + 4, kflrl_obs_source_[i]);
        kflrl_put_u32_(v + 6, kflrl_obs_pair_[i]);
        total += kflrl_tlv_(&p, K26RL_TAG_OBS_CHANNEL_SOURCE, 10, v);
    }
#endif
#if KFLRL_N_BODIES > 0
    for (uint32_t i = 0; i < (uint32_t)KFLRL_N_BODIES; i++) {
        uint8_t nv[4 + 64];
        uint32_t nl = (uint32_t)strlen(kflrl_body_names_[i]);
        if (nl > 64) nl = 64;
        kflrl_put_u32_(nv, i);
        memcpy(nv + 4, kflrl_body_names_[i], nl);
        total += kflrl_tlv_(&p, K26RL_TAG_BODY_NAME, 4 + nl, nv);
    }
#endif
#if KFLRL_N_ASSEMBLIES > 0
    for (uint32_t i = 0; i < (uint32_t)KFLRL_N_ASSEMBLIES; i++) {
        uint8_t dv[4 + 32];
        kflrl_put_u32_(dv, kflrl_assemblies_[i].body);
        memcpy(dv + 4, kflrl_assemblies_[i].digest, 32);
        total += kflrl_tlv_(&p, K26RL_TAG_ASSEMBLY_DIGEST, 4 + 32, dv);
    }
    for (uint32_t i = 0; i < (uint32_t)KFLRL_N_ASSEMBLIES; i++) {
        uint8_t nv2[4 + 64];
        uint32_t nl2 = (uint32_t)strlen(kflrl_assemblies_[i].name);
        if (nl2 > 64) nl2 = 64;
        kflrl_put_u32_(nv2, kflrl_assemblies_[i].body);
        memcpy(nv2 + 4, kflrl_assemblies_[i].name, nl2);
        total += kflrl_tlv_(&p, K26RL_TAG_ASSEMBLY_NAME, 4 + nl2, nv2);
    }
#endif
    kflrl_put_u32_(v, 1u);   /* bit 0: auto-reset, always on */
    total += kflrl_tlv_(&p, K26RL_TAG_EPISODE_FLAGS, 4, v);
    return total;
}

/* ---- Frozen surface ----------------------------------------------- */

extern "C" uint32_t k26rl_abi_version(void)
{
    return K26RL_ABI_VERSION;
}

static void kflrl_free_handle_(K26RlEnv *h)
{
    if (!h) return;
#if KFLRL_N_PLAN > 0
    free(h->plan_slot);
    free(h->plan_scratch);
    free(h->plan_buf);
#endif
#if KFLRL_N_PAYLOAD > 0
    /* Payloads before vehicles: each `_destroy` unlinks itself from
     * the vehicle's payload slot before releasing its storage, so
     * the vehicle must still be there when it runs. This is the
     * other half of the tier's allocation discipline: the
     * constructors ran once in the world prefix and the destructors
     * run once here, and nothing between them allocates. */
    if (h->payloads) {
        for (uint32_t i = 0; i < h->n_envs * KFLRL_N_PAYLOAD; i++) {
            if (!h->payloads[i]) continue;
            kflrl_payload_destroy_((int)(i % KFLRL_N_PAYLOAD),
                                   h->payloads[i]);
        }
    }
#@ The per-unit handles are the same kind as the payload whose slot
#@ they sit in, so one destructor table serves both walks. They bind
#@ to no vehicle, so nothing else has to be told they are going.
    if (h->payu) {
        for (uint32_t i = 0; i < h->n_envs * KFLRL_N_PAYLOAD; i++) {
            if (!h->payu[i]) continue;
            kflrl_payload_destroy_((int)(i % KFLRL_N_PAYLOAD),
                                   h->payu[i]);
        }
    }
    free(h->payloads);
    free(h->payu);
    free(h->payp);
    free(h->info_t);
    free(h->info_day);
#endif
#if KFLRL_N_EFFECTOR > 0
    free(h->eng);
#endif
#if KFLRL_N_VEHICLES > 0
    /* Vehicles first: a vehicle's teardown notifies its subsystems
     * and unregisters from the world, so the world must still be
     * there when it runs. */
    if (h->vehicles) {
        for (uint32_t i = 0; i < h->n_envs * KFLRL_N_VEHICLES; i++) {
            if (h->vehicles[i]) {
                uint32_t e = i / KFLRL_N_VEHICLES;
                if (h->worlds && h->worlds[e]) {
                    k26astro_world_unregister_vehicle(h->worlds[e],
                                                      h->vehicles[i]);
                }
                k26astro_vehicle_destroy(h->vehicles[i]);
            }
        }
    }
    free(h->vehicles);
    free(h->act);
    free(h->contact);
    free(h->join);
    free(h->thrust_ctx);
    free(h->prop);
    free(h->veh_com);
    free(h->thr_scale);
    free(h->thr_isum);
    free(h->thr_itime);
#if KFLRL_N_PROPELLANT > 0
    if (h->ledger) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            k26astro_rt_ledger_destroy(h->ledger[e]);
        }
    }
#endif
    free(h->ledger);
#endif
    if (h->worlds) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            if (h->worlds[e]) k26astro_world_destroy(h->worlds[e]);
        }
    }
    free(h->worlds);
    free(h->baseline);
    free(h->baseline_t);
    free(h->episode);
    free(h->steps);
    free(h->ended);
    free(h->obs);
    free(h->rew);
    free(h->flags);
    free(h->fault);
    free(h->dr_vals);
    free(h->wscal);
#if KFLRL_N_SENSED > 0
    free(h->sterm);
    free(h->sense);
    free(h->sense_ring);
#endif
    free(h->scratch);
    free(h->spec);
    free(h->seen_seeds);
    free(h);
}

#if KFLRL_N_THRUSTERS > 0
/* Thrust reaches translation as an acceleration on the gravity
 * state's perturbation registry, which the integrator evaluates at
 * its own stages: the thrust is integrated with everything else
 * rather than added to a finished step. The registry is additive and
 * is dispatched in registration order, which is fixed at create.
 *
 * The same declaration that gives a thruster its torque gives it its
 * force, so a program cannot have one without the other. */
static void kflrl_thrust_perturb_(const K26AstroGravState *st,
                                  const K26AstroGravView *vw,
                                  K26V3 *accel, void *ctx)
{
    (void)st; (void)vw;
    KflrlThrustCtx *c = (KflrlThrustCtx *)ctx;
    if (!c || !c->h || !accel) return;
    for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
        K26AstroVehicle *veh =
            c->h->vehicles[(size_t)c->e * KFLRL_N_VEHICLES + vi];
        if (!veh) continue;
        K26AstroAttWheel wbuf[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];
        K26AstroAttTorquer qbuf[KFLRL_N_TORQUERS > 0 ?
                                KFLRL_N_TORQUERS : 1];
        K26AstroAttThruster tbuf[KFLRL_N_THRUSTERS > 0 ?
                                 KFLRL_N_THRUSTERS : 1];
        int wmap[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];
        K26AstroAttActuators view;
        kflrl_act_view_(&c->h->act[c->e], vi,
                        KFLRL_COM(c->h, c->e, vi),
                        KFLRL_THRSC(c->h, c->e, vi), wbuf, qbuf, tbuf,
                        &view, wmap);
        K26V3 f_body;
        if (k26astro_att_thrusters_wrench(&view, &f_body, NULL) !=
            K26ASTRO_ATT_OK) continue;
        if (f_body.x == 0.0 && f_body.y == 0.0 && f_body.z == 0.0) {
            continue;
        }
        K26AstroBody *b = k26astro_vehicle_body(veh);
        if (!b || !(b->mass > 0.0)) continue;
        K26V3 f_world = k26m3d_quat_rotate_v3(b->attitude, f_body);
        int bi = kflrl_body_idx_[kflrl_vehicle_body_[vi]];
        if (bi < 0) continue;
        accel[bi].x += f_world.x / b->mass;
        accel[bi].y += f_world.y / b->mass;
        accel[bi].z += f_world.z / b->mass;
    }
}
#endif

extern "C" K26RlStatus k26rl_env_create(uint64_t seed, uint32_t n_envs,
                                         K26RlEnv **out_env)
{
    if (!out_env) return K26RL_E_NULL;
    *out_env = NULL;
    if (n_envs == 0) return K26RL_E_GEOMETRY;

    K26RlEnv *h = (K26RlEnv *)calloc(1, sizeof *h);
    if (!h) return K26RL_E_INTERNAL;
    h->n_envs = n_envs;
    h->seed = seed;
    h->key = k26rng_key(seed);
    h->rekey_ordinal = 0;
    h->control_dt = kflrl_control_dt_();
    h->horizon = kflrl_horizon_();
    h->substeps = kflrl_substeps_();
    h->restitution = kflrl_restitution_();
    h->friction    = kflrl_friction_();
    if (h->substeps < 1u) h->substeps = 1u;
    /* The control period is checked first and keeps the status it
     * has always returned; the subdivision's own check follows, so
     * that adding one cannot change what a bad period reports. */
    if (!(h->control_dt > 0.0) || !std::isfinite(h->control_dt)) {
        free(h);
        return K26RL_E_INTERNAL;
    }
    h->sub_dt = h->control_dt / (double)h->substeps;
    if (!(h->sub_dt > 0.0) || !std::isfinite(h->sub_dt)) {
        kflrl_free_handle_(h);
        return K26RL_E_GEOMETRY;
    }
    kflrl_act_params_(h->act_lo, h->act_hi, h->act_arity, h->act_kind);

    h->cap_seen = 4;
    h->seen_seeds = (uint64_t *)malloc(h->cap_seen * sizeof(uint64_t));
    h->worlds = (K26AstroWorld **)calloc(n_envs, sizeof(*h->worlds));
#if KFLRL_N_VEHICLES > 0
    h->vehicles = (K26AstroVehicle **)calloc(
        (size_t)n_envs * KFLRL_N_VEHICLES, sizeof(*h->vehicles));
    h->act = (KflrlAct *)calloc(n_envs, sizeof(*h->act));
    h->thrust_ctx = (KflrlThrustCtx *)calloc(n_envs,
                                             sizeof(*h->thrust_ctx));
    h->prop = (double *)calloc(
        (size_t)n_envs * KFLRL_N_VEHICLES, sizeof(double));
    h->veh_com = (double *)calloc(
        (size_t)n_envs * KFLRL_N_VEHICLES * 3, sizeof(double));
    h->thr_scale = (double *)calloc(
        (size_t)n_envs * KFLRL_N_VEHICLES, sizeof(double));
    h->thr_isum = (double *)calloc(
        (size_t)n_envs * KFLRL_N_VEHICLES, sizeof(double));
    h->thr_itime = (double *)calloc(n_envs, sizeof(double));
    h->ledger = (K26AstroRtConservationLedger **)calloc(
        n_envs, sizeof(*h->ledger));
    if (!h->prop || !h->veh_com || !h->thr_scale || !h->thr_isum ||
        !h->thr_itime || !h->ledger) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
#if KFLRL_N_PROPELLANT > 0
    /* One ledger per environment, taken here because a burn is
     * recorded on the stepping path and that path allocates
     * nothing. */
    for (uint32_t e = 0; e < n_envs; e++) {
        h->ledger[e] = k26astro_rt_ledger_new();
        if (!h->ledger[e]) {
            kflrl_free_handle_(h);
            return K26RL_E_INTERNAL;
        }
    }
#endif
#endif
    h->baseline = (K26AstroBody *)calloc(
        (size_t)n_envs * (KFLRL_N_BODIES ? KFLRL_N_BODIES : 1),
        sizeof(K26AstroBody));
    h->baseline_t = (K26AstroEpoch *)calloc(n_envs, sizeof(K26AstroEpoch));
    h->episode = (uint32_t *)calloc(n_envs, sizeof(uint32_t));
    h->steps = (uint32_t *)calloc(n_envs, sizeof(uint32_t));
    h->ended = (uint8_t *)calloc(n_envs, 1);
    h->obs = (double *)calloc(
        (size_t)n_envs * (KFLRL_OBS_TOTAL ? KFLRL_OBS_TOTAL : 1),
        sizeof(double));
    h->rew = (double *)calloc((size_t)n_envs * KFLRL_N_AGENTS,
                              sizeof(double));
    h->flags = (uint32_t *)calloc(n_envs, sizeof(uint32_t));
    h->fault = (uint16_t *)calloc(n_envs, sizeof(uint16_t));
    h->dr_vals = (double *)calloc(
        (size_t)n_envs * (KFLRL_N_REC ? KFLRL_N_REC : 1),
        sizeof(double));
    h->wscal = (double *)calloc(
        (size_t)n_envs * (KFLRL_N_WSCAL ? KFLRL_N_WSCAL : 1),
        sizeof(double));
    /* The contact block is allocated for every program, not only
     * for one that declares a vehicle, because the observation
     * function takes it either way and a conditional signature
     * would buy nothing but a second shape to keep agreeing. */
#if KFLRL_N_PLAN > 0
    h->plan_slot = (K26RlRefKnot *)calloc(KFLRL_PLAN_SLOTS,
                                          sizeof(K26RlRefKnot));
    h->plan_scratch = (K26RlRefKnot *)calloc(KFLRL_PLAN_SLOTS,
                                             sizeof(K26RlRefKnot));
    h->plan_buf = (unsigned char *)calloc(KFLRL_PLAN_BYTES, 1);
    if (!h->plan_slot || !h->plan_scratch || !h->plan_buf) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
#endif
    h->join = (KflrlJoin *)calloc(n_envs, sizeof(*h->join));
    h->contact = (KflrlContact *)calloc(
        (size_t)n_envs * KFLRL_N_CONTACT, sizeof(KflrlContact));
#if KFLRL_N_SENSED > 0
    h->sterm = (K26SenseTerm *)calloc(KFLRL_N_STERMS,
                                      sizeof(K26SenseTerm));
    h->sense = (K26SenseState *)calloc(
        (size_t)n_envs * KFLRL_N_SENSED, sizeof(K26SenseState));
    h->sense_ring = (double *)calloc(
        (size_t)n_envs * KFLRL_N_SENSED *
        (KFLRL_SENSE_RING > 0 ? KFLRL_SENSE_RING : 1),
        sizeof(double));
    if (!h->sterm || !h->sense || !h->sense_ring) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
#endif
#if KFLRL_N_PAYLOAD > 0
    h->payloads = (void **)calloc(
        (size_t)n_envs * KFLRL_N_PAYLOAD, sizeof(void *));
    h->payu = (void **)calloc(
        (size_t)n_envs * KFLRL_N_PAYLOAD, sizeof(void *));
    h->payp = (double *)calloc(
        (size_t)n_envs * KFLRL_N_PAYLOAD * KFLRL_PAY_NPARAM,
        sizeof(double));
    h->info_t   = (double *)calloc(n_envs, sizeof(double));
    h->info_day = (int64_t *)calloc(n_envs, sizeof(int64_t));
    if (!h->payloads || !h->payu || !h->payp || !h->info_t ||
        !h->info_day) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
#endif
#if KFLRL_N_EFFECTOR > 0
    h->eng = (KflrlEng *)calloc(n_envs, sizeof(KflrlEng));
    if (!h->eng) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
#endif
    h->scratch = (double *)calloc(
        KFLRL_OBS_TOTAL ? KFLRL_OBS_TOTAL : 1, sizeof(double));
    if (!h->seen_seeds || !h->worlds || !h->baseline || !h->baseline_t ||
        !h->episode || !h->steps || !h->ended || !h->obs || !h->rew ||
        !h->flags || !h->fault || !h->dr_vals || !h->wscal ||
#if KFLRL_N_VEHICLES > 0
        !h->vehicles || !h->act || !h->thrust_ctx ||
#endif
        !h->join ||
        !h->contact || !h->scratch) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
    h->seen_seeds[0] = seed;
    h->n_seen = 1;

    for (uint32_t e = 0; e < n_envs; e++) {
        h->worlds[e] = k26astro_world_create(K26ASTRO_MODE_PORTABLE,
                                             K26ASTRO_COORDS_SECTOR_GRID);
        if (!h->worlds[e]) {
            kflrl_free_handle_(h);
            return K26RL_E_INTERNAL;
        }
        (void)k26astro_world_set_seed(
            h->worlds[e], kflrl_world_seed_(h->key, e, 0u));
#if KFLRL_N_DR > 0
        double dr0[KFLRL_N_DR];
#else
        double *dr0 = NULL;
#endif
        if (kflrl_build_world_(h->worlds[e], h->key, e,
                h->wscal + (size_t)e * KFLRL_N_WSCAL, dr0,
#if KFLRL_N_VEHICLES > 0
                h->vehicles + (size_t)e * KFLRL_N_VEHICLES,
#else
                NULL,
#endif
#if KFLRL_N_PAYLOAD > 0
                h->payloads + (size_t)e * KFLRL_N_PAYLOAD,
                h->payu + (size_t)e * KFLRL_N_PAYLOAD,
                h->payp + (size_t)e * KFLRL_N_PAYLOAD * KFLRL_PAY_NPARAM
#else
                NULL, NULL, NULL
#endif
                ) != 0) {
            kflrl_free_handle_(h);
            return K26RL_E_INTERNAL;
        }
#if KFLRL_N_VEHICLES > 0
        /* The tank as built, and the centre of mass it puts the
         * craft at. The centre of mass is read back off the vehicle
         * rather than recomputed here, so there is one place a
         * constructed value comes from whether the assembly carries
         * a tank or not. */
        for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
            h->prop[(size_t)e * KFLRL_N_VEHICLES + vi] =
                kflrl_veh_prop_cap_[vi];
            KFLRL_THRSC(h, e, vi) = 1.0;
            K26AstroEpoch t0;
            memset(&t0, 0, sizeof t0);
            K26V3 c0 = k26astro_vehicle_com_at(
                h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi], t0);
            double *cs = KFLRL_COM(h, e, vi);
            cs[0] = c0.x; cs[1] = c0.y; cs[2] = c0.z;
        }
#endif
#if KFLRL_N_THRUSTERS > 0
        /* Registered here rather than at world build because the
         * context is the handle and this environment's index, and
         * the handle is what owns them. Registration order is fixed
         * by this loop. */
        h->thrust_ctx[e].h = h;
        h->thrust_ctx[e].e = e;
        if (k26astro_grav_register_perturb(
                k26astro_world_grav(h->worlds[e]),
                kflrl_thrust_perturb_, &h->thrust_ctx[e]) != 0) {
            kflrl_free_handle_(h);
            return K26RL_E_INTERNAL;
        }
#endif
#if KFLRL_N_BODIES > 0
        {
            K26AstroBody *b0 = k26astro_world_body_at(h->worlds[e], 0);
            if (b0) {
                memcpy(h->baseline + (size_t)e * KFLRL_N_BODIES, b0,
                       sizeof(K26AstroBody) * KFLRL_N_BODIES);
            }
        }
#endif
        {
            K26AstroGravState *g = k26astro_world_grav(h->worlds[e]);
            if (g) h->baseline_t[e] = g->t;
        }
#if KFLRL_N_REC > 0
        kflrl_apply_draws_(h->worlds[e], h->key, e, 0,
                           h->dr_vals + (size_t)e * KFLRL_N_REC, dr0,
                           KFLRL_PAYP(h, e));
#else
        kflrl_apply_draws_(h->worlds[e], h->key, e, 0, NULL, dr0,
                           KFLRL_PAYP(h, e));
#endif
#if KFLRL_N_PAYLOAD > 0
        /* The episode epoch, before any stepping: this allocates
         * each target's history ring here in the prefix, where
         * allocation belongs, and gives the interpolator its first
         * sample. One act serves both. */
        h->info_day[e] = 0;
        h->info_t[e]   = 0.0;
        kflrl_info_push_(h, e);
#endif
        kflrl_observe_(h->worlds[e],
                       h->obs + (size_t)e * KFLRL_OBS_TOTAL,
                       &h->contact[(size_t)e * KFLRL_N_CONTACT],
                       &h->join[e], KFLRL_PAYH(h, e), KFLRL_PAYP(h, e),
                       KFLRL_VEHS(h, e), KFLRL_PROP(h, e),
                       KFLRL_COM(h, e, 0), KFLRL_INFODAY(h, e),
                       KFLRL_INFOT(h, e), KFLRL_ENG(h, e), 0.0);
    }

    h->spec_len = kflrl_spec_write_(NULL, h);
    h->spec = (uint8_t *)malloc(h->spec_len);
    if (!h->spec) {
        kflrl_free_handle_(h);
        return K26RL_E_INTERNAL;
    }
    (void)kflrl_spec_write_(h->spec, h);

#if KFLRL_N_SENSED > 0
    /* The declared terms become the library's typed terms
     * here: the artifact is C++11 and cannot name a union
     * member in an initialiser, and the bias walk's decay and
     * driving coefficients follow from the control period,
     * which is read once, here, and never on the step path. */
    for (int t = 0; t < KFLRL_N_STERMS; t++) {
        K26SenseTerm *d = &h->sterm[t];
        d->kind       = (K26SenseKind)kflrl_sterm_[t].kind;
        d->channel    = kflrl_sterm_[t].ch;
        d->channel_ep = kflrl_sterm_[t].ch_ep;
        double p0 = kflrl_sterm_[t].p0;
        double p1 = kflrl_sterm_[t].p1;
        double p2 = kflrl_sterm_[t].p2;
        switch (d->kind) {
        case K26SENSE_ADDITIVE: d->u.additive.sigma = p0; break;
        case K26SENSE_SCALE: d->u.scale.rel_sigma = p0; break;
        case K26SENSE_BIAS_WALK:
            d->u.bias_walk.sigma0 = p0;
            if (k26sense_bias_walk_coeffs(
                    p1, h->control_dt, p2,
                    &d->u.bias_walk.phi,
                    &d->u.bias_walk.q) != K26SENSE_OK) {
                kflrl_free_handle_(h);
                return K26RL_E_INTERNAL;
            }
            break;
        case K26SENSE_LATENCY:
            d->u.latency.depth = (uint32_t)p0; break;
        case K26SENSE_QUANTISE:
            d->u.quantise.lsb = p0;
            d->u.quantise.lo = p1;
            d->u.quantise.hi = p2; break;
        case K26SENSE_DROPOUT: d->u.dropout.p = p0; break;
        default: break;
        }
    }
    for (uint32_t e = 0; e < n_envs; e++) {
        for (int c = 0; c < KFLRL_N_SENSED; c++) {
            size_t k = (size_t)e * KFLRL_N_SENSED + (size_t)c;
            h->sense[k].ring = h->sense_ring + k *
                (KFLRL_SENSE_RING > 0 ? KFLRL_SENSE_RING : 1);
            h->sense[k].ring_cap = KFLRL_SENSE_RING;
        }
        kflrl_sense_reset_(h, e, 0u,
                           h->obs + (size_t)e * KFLRL_OBS_TOTAL);
    }
#endif
    h->at_boundary = 1;
    h->magic = KFLRL_MAGIC;
    *out_env = h;
    return K26RL_OK;
}

extern "C" K26RlStatus k26rl_env_output(K26RlEnv *h, const char *path)
{
    if (!h) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    if (!h->at_boundary) return K26RL_E_OUTPUT_TIMING;
    if (h->writer) {
        K26RlStatus cst = k26rl_episode_writer_close(h->writer);
        h->writer = NULL;
        if (cst != K26RL_OK) return K26RL_E_INTERNAL;
    }
    if (!path) return K26RL_OK;

    K26RlEpisodeGeom geom;
    geom.n_envs = h->n_envs;
    geom.agent_count = KFLRL_N_AGENTS;
    geom.obs_total = KFLRL_OBS_TOTAL;
    geom.act_total = KFLRL_ACT_TOTAL;
    geom.steps_per_chunk = 1024;
    geom.dr_max = KFLRL_N_REC;
    K26RlStatus st = k26rl_episode_writer_open(
        path, &geom, h->seed, h->rekey_ordinal, "3.2",
        K26ASTRO_RT_LIB_VERSION, h->spec, h->spec_len, &h->writer);
    if (st != K26RL_OK) return st;
#if KFLRL_N_REF > 0
    /* Every plan this program flies goes into the record, verbatim
     * and with its own digest beside it, immediately after the file
     * header. A run whose plan is not recoverable from its recording
     * cannot be replayed or explained, and carrying the digest
     * separately is what lets a reader tell a recording flown
     * against the plan it names from one flown against a plan of the
     * same name. A plan is fixed for the whole file, so one frame
     * covers every environment and every episode in it. */
    for (int r = 0; r < KFLRL_N_REF; r++) {
        st = k26rl_episode_writer_plan(
            h->writer, K26RL_PLAN_ROLE_FLOWN, K26RL_PLAN_ALL,
            K26RL_PLAN_ALL, kflrl_ref_digest_[r], kflrl_ref_bytes_[r],
            kflrl_ref_len_[r]);
        if (st != K26RL_OK) {
            (void)k26rl_episode_writer_close(h->writer);
            h->writer = NULL;
            return K26RL_E_INTERNAL;
        }
    }
#endif
    for (uint32_t e = 0; e < h->n_envs; e++) {
        st = k26rl_episode_writer_start(
            h->writer, e, h->episode[e],
            h->obs + (size_t)e * KFLRL_OBS_TOTAL,
#if KFLRL_N_REC > 0
            kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,
#else
            NULL, NULL,
#endif
            KFLRL_N_REC);
        if (st != K26RL_OK) {
            (void)k26rl_episode_writer_close(h->writer);
            h->writer = NULL;
            return K26RL_E_INTERNAL;
        }
    }
    return K26RL_OK;
}

/* The telemetry tap: the same frames the episode writer emits,
 * published on a shared memory ring instead of into a file. It is
 * enabled the same way, at the same moments, with the same refusals,
 * and it is a separate producer sharing no buffer and no counter
 * with the writer, so a run's episode file is identical whether or
 * not this is on. Publication is best-effort: nothing below can fail
 * and nothing below can slow a step, so no status here reaches a
 * stepping caller. */
extern "C" K26RlStatus k26rl_env_tap(K26RlEnv *h, const char *name)
{
    if (!h) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    if (!h->at_boundary) return K26RL_E_OUTPUT_TIMING;
    if (h->tap) {
        k26rl_tap_close(h->tap);
        h->tap = NULL;
    }
    if (!name) return K26RL_OK;

    K26RlEpisodeGeom geom;
    geom.n_envs = h->n_envs;
    geom.agent_count = KFLRL_N_AGENTS;
    geom.obs_total = KFLRL_OBS_TOTAL;
    geom.act_total = KFLRL_ACT_TOTAL;
    geom.steps_per_chunk = 1024;
    geom.dr_max = KFLRL_N_REC;
    K26RlStatus st = k26rl_tap_open(
        name, &geom, h->seed, h->rekey_ordinal, "3.2",
        K26ASTRO_RT_LIB_VERSION, h->spec, h->spec_len, &h->tap);
    if (st != K26RL_OK) return st;
    for (uint32_t e = 0; e < h->n_envs; e++) {
        k26rl_tap_start(
            h->tap, e, h->episode[e],
            h->obs + (size_t)e * KFLRL_OBS_TOTAL,
#if KFLRL_N_REC > 0
            kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,
#else
            NULL, NULL,
#endif
            KFLRL_N_REC);
    }
    return K26RL_OK;
}

#if KFLRL_N_PLAN > 0
/* Write the plan the episode's last actions make.
 *
 * A planner's actions are its knots, and the plan is what its last
 * step asked for. The slots are read straight out of the action
 * vector; which of them are present, what order they go in, and what
 * the file's bytes are, are the plan format's rules and not this
 * artifact's, so a plan this world writes is a plan any reader of
 * that format accepts.
 *
 * It runs only where the episode record is being written, which is
 * the one place this artifact is permitted output at all, and it is
 * called at an episode end and never from a step. It allocates
 * nothing: the buffers came with the handle, the encode writes into
 * them, and the file leaves through the unbuffered calls rather than
 * through a stream that would allocate one.
 *
 * The path carries the episode identity the record uses, so a plan
 * file and the recording that produced it name each other. */
static void kflrl_plan_write_(K26RlEnv *h, uint32_t e,
                              const double *aslice)
{
    for (int p = 0; p < KFLRL_N_PLAN; p++) {
        K26RlRefPlan plan;
        uint64_t len = 0;
        int nslot = kflrl_plan_nslots_[p];
        char path[512];
        int fd;

        for (int k = 0; k < nslot; k++) {
            const double *a = aslice + kflrl_plan_act_[p] + k * 8;
            h->plan_slot[k].t = a[0];
            for (int q = 0; q < 3; q++) {
                h->plan_slot[k].r[q] = a[1 + q];
                h->plan_slot[k].v[q] = a[4 + q];
            }
            h->plan_slot[k].tolerance = a[7];
        }
        plan.frame_kind = kflrl_plan_kind_[p];
        plan.frame_name = kflrl_plan_frame_[p];
        plan.provenance = kflrl_plan_prov_[p];
        plan.epoch      = kflrl_plan_epoch_[p];
        plan.knots      = h->plan_slot;
        plan.knot_count = (uint32_t)nslot;
        if (k26rl_ref_encode_into(&plan, h->plan_scratch, h->plan_buf,
                                  (uint64_t)KFLRL_PLAN_BYTES,
                                  &len) != K26RL_REF_OK) {
            continue;
        }
        snprintf(path, sizeof path, "%s-%lu-%lu-%lu%s",
                 kflrl_plan_file_[p],
                 (unsigned long)h->rekey_ordinal, (unsigned long)e,
                 (unsigned long)h->episode[e], K26RL_REF_SUFFIX);
        fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            const unsigned char *b = h->plan_buf;
            uint64_t left = len;
            while (left > 0) {
                ssize_t w = write(fd, b, (size_t)left);
                if (w <= 0) break;
                b += w;
                left -= (uint64_t)w;
            }
            (void)close(fd);
        }
        (void)k26rl_episode_writer_plan(
            h->writer, K26RL_PLAN_ROLE_EMITTED, e, h->episode[e],
            h->plan_buf + K26RL_REF_DIGEST_OFFSET, h->plan_buf,
            (uint32_t)len);
    }
}
#endif

/* An environment's episode ends by fault: no transition completes,
 * the public observation slice keeps the pre-step values, the fault
 * record and episode-end frame travel the file when enabled. */
static K26RlStatus kflrl_fault_(K26RlEnv *h, uint32_t e,
                                const double *aslice, uint16_t reason)
{
    /* Termination is per environment, so a fault ends the episode
     * for every agent at once and every agent's reward stream
     * carries the zero the surface documents for a faulted step. */
    double zero_reward[KFLRL_N_AGENTS];
    double zero_adj[KFLRL_N_AGENTS];
    for (int a = 0; a < KFLRL_N_AGENTS; a++) {
        h->rew[(size_t)e * KFLRL_N_AGENTS + a] = 0.0;
        zero_reward[a] = 0.0;
        zero_adj[a] = 0.0;
    }
    h->flags[e] = K26RL_FLAG_FAULT;
    h->fault[e] = reason;
    h->ended[e] = 1;
    if (h->writer) {
        K26RlStatus st = k26rl_episode_writer_step(
            h->writer, e, h->obs + (size_t)e * KFLRL_OBS_TOTAL, aslice,
            zero_reward, K26RL_FLAG_FAULT, 0.0);
        if (st != K26RL_OK) return K26RL_E_INTERNAL;
        st = k26rl_episode_writer_end(h->writer, e, K26RL_END_FAULT,
                                      reason, zero_adj);
        if (st != K26RL_OK) return K26RL_E_INTERNAL;
    }
    if (h->tap) {
        k26rl_tap_step(h->tap, e, h->obs + (size_t)e * KFLRL_OBS_TOTAL,
                       aslice, zero_reward, K26RL_FLAG_FAULT, 0.0);
        k26rl_tap_end(h->tap, e, K26RL_END_FAULT, reason, zero_adj);
    }
    return K26RL_OK;
}

extern "C" K26RlStatus k26rl_env_step(K26RlEnv *h, const double *actions)
{
    if (!h) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    if (KFLRL_ACT_TOTAL != 0 && !actions) return K26RL_E_NULL;

    /* A boundary reset that would exhaust its episode coordinates
     * refuses the whole call before anything advances. */
    for (uint32_t e = 0; e < h->n_envs; e++) {
        if (h->ended[e] && h->episode[e] == 0xFFFFFFFFu) {
            return K26RL_E_RNG_EXHAUSTED;
        }
    }
    h->at_boundary = 0;

    for (uint32_t e = 0; e < h->n_envs; e++) {
        const double *aslice = actions
            ? actions + (size_t)e * KFLRL_ACT_TOTAL : NULL;
        const double *wslice = h->wscal + (size_t)e * KFLRL_N_WSCAL;
        (void)wslice;

        if (h->ended[e]) {
            /* Boundary reset: no transition, no time advance, the
             * new episode's initial observation, only the
             * reset-boundary bit. */
            kflrl_reset_env_(h, e, h->episode[e] + 1u);
            h->flags[e] = K26RL_FLAG_RESET_BOUNDARY;
            if (h->writer) {
                K26RlStatus st = k26rl_episode_writer_start(
                    h->writer, e, h->episode[e],
                    h->obs + (size_t)e * KFLRL_OBS_TOTAL,
#if KFLRL_N_REC > 0
                    kflrl_dr_tags_,
                    h->dr_vals + (size_t)e * KFLRL_N_REC,
#else
                    NULL, NULL,
#endif
                    KFLRL_N_REC);
                if (st != K26RL_OK) return K26RL_E_INTERNAL;
            }
            if (h->tap) {
                k26rl_tap_start(
                    h->tap, e, h->episode[e],
                    h->obs + (size_t)e * KFLRL_OBS_TOTAL,
#if KFLRL_N_REC > 0
                    kflrl_dr_tags_,
                    h->dr_vals + (size_t)e * KFLRL_N_REC,
#else
                    NULL, NULL,
#endif
                    KFLRL_N_REC);
            }
            continue;
        }

#if KFLRL_N_EFFECTOR > 0
        /* An engagement is an act of this step alone, so the block
         * is cleared before the body runs: on a step that engages
         * nothing every effector channel reads zero, and the
         * one-engagement-per-payload record starts empty. */
        memset(&h->eng[e], 0, sizeof h->eng[e]);
#endif
        kflrl_on_step_(h->worlds[e], aslice, &h->act[e],
                       KFLRL_PAYH(h, e), KFLRL_PAYU(h, e),
                       KFLRL_PAYP(h, e),
                       h->control_dt, KFLRL_ENG(h, e));
#if KFLRL_N_EFFECTOR > 0
        /* A payload engaged twice inside one step. The compiler
         * refuses the statically visible case; this is the one it
         * cannot see, and the environment faults rather than
         * publishing a result whose value depends on which call ran
         * last. Both engagements have already reached the world, so
         * the transition is abandoned rather than committed. */
        if (h->eng[e].fault) {
            K26RlStatus fst = kflrl_fault_(
                h, e, aslice, (uint16_t)K26RL_E_ENV_INTERNAL);
            if (fst != K26RL_OK) return fst;
            continue;
        }
#endif
        /* A contact is a fact about one transition, so the latch is
         * cleared here and whatever the sub-advances below find is
         * what this step reports. */
        memset(&h->contact[(size_t)e * KFLRL_N_CONTACT], 0,
               sizeof(KflrlContact) * KFLRL_N_CONTACT);
        /* One transition is `substeps` sub-advances. Translation
         * advances first, then attitude by the same interval with
         * the torque held at its start, which is the splitting the
         * attitude library documents and its gates measure.
         *
         * The last sub-advance takes the remainder rather than the
         * quotient, so the simulated time a transition advances sums
         * to control_dt exactly however the division rounded. That
         * subtraction is exact: by the last sub-advance at least
         * half the period has been advanced, so the two operands lie
         * within a factor of two of each other and the difference is
         * representable.
         */
        int rc = 0;
        uint16_t att_reason = 0;
        double advanced = 0.0;
#if KFLRL_N_VEHICLES > 0
        for (int vi_ = 0; vi_ < KFLRL_N_VEHICLES; vi_++)
            h->thr_isum[(size_t)e * KFLRL_N_VEHICLES + vi_] = 0.0;
        h->thr_itime[e] = 0.0;
#endif
        for (uint32_t sub = 0; sub < h->substeps; sub++) {
            double step_dt = (sub + 1u == h->substeps)
                           ? (h->control_dt - advanced)
                           : h->sub_dt;
#if KFLRL_N_COLL > 0
            /* The sweep needs both endpoints of the sub-advance, so
             * the starting configuration is taken before the world
             * moves. Fixed-size buffers over compile-time counts:
             * nothing here allocates. */
            K26AstroPos csnap[KFLRL_N_VEHICLES];
            K26V3       cvel0[KFLRL_N_VEHICLES];
            K26Quat     cquat[KFLRL_N_VEHICLES];
            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
                const K26AstroBody *cb = k26astro_world_body_at(
                    h->worlds[e], kflrl_body_idx_[kflrl_vehicle_body_[vi]]);
                if (!cb) { memset(&csnap[vi], 0, sizeof csnap[vi]);
                           cvel0[vi] = k26m3d_v3(0.0, 0.0, 0.0);
                           cquat[vi] = k26m3d_quat_identity(); continue; }
                csnap[vi] = cb->pos;
                cvel0[vi] = cb->vel;
                cquat[vi] = cb->attitude;
            }
#endif
#if KFLRL_N_PROPELLANT > 0
            /* What this sub-interval's thrust costs, decided before
             * anything moves, because the answer scales the thrust
             * and the torque the advance below applies.
             *
             * Per thruster: exhaust speed is the specific impulse
             * times standard gravity, mass flow is the throttled
             * thrust divided by that speed, and what it spends over
             * the interval is that flow times the interval. The
             * flows are summed over the vehicle's thrusters and the
             * tank is debited once, rather than each thruster
             * drawing on the tank as it is reached: a tank debited
             * in turn would empty part way along the sequence, and
             * which thrusters had already been reached when it did
             * is an accident of declaration order rather than
             * anything the program said.
             *
             * The sub-interval a tank runs out in is apportioned
             * rather than truncated or overrun. Every firing
             * thruster is scaled by the same fraction, which is the
             * part of the interval there was propellant for, so the
             * impulse delivered is exactly the impulse that mass
             * could deliver and the rest of the interval is
             * unpowered. Letting the interval finish would spend
             * propellant the craft does not have, and cutting it
             * short would throw away propellant it does. */
            double burned[KFLRL_N_VEHICLES];
            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
                burned[vi] = 0.0;
                KFLRL_THRSC(h, e, vi) = 1.0;
                if (!(kflrl_veh_prop_cap_[vi] > 0.0)) continue;
                double left = h->prop[(size_t)e * KFLRL_N_VEHICLES + vi];
                double flow = 0.0;
#if KFLRL_N_THRUSTERS > 0
                for (int i = 0; i < KFLRL_N_THRUSTERS; i++) {
                    if (kflrl_thruster_veh_[i] != vi) continue;
                    double u = h->act[e].cmd[
                        KFLRL_N_WHEELS + KFLRL_N_TORQUERS + i];
                    u = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
                    if (u == 0.0) continue;
                    flow += u * kflrl_thruster_desc_[i].max_thrust /
                            (kflrl_thruster_desc_[i].isp_s * KFLRL_G0);
                }
#endif
                double demand = flow * step_dt;
                if (demand <= 0.0) continue;
                if (demand > left) {
                    KFLRL_THRSC(h, e, vi) = left / demand;
                    burned[vi] = left;
                } else {
                    burned[vi] = demand;
                }
            }
#endif
            rc = k26astro_world_step_exact(h->worlds[e], step_dt);
            if (rc != 0) break;
            advanced += step_dt;
#if KFLRL_N_VEHICLES > 0
            for (int vi_ = 0; vi_ < KFLRL_N_VEHICLES; vi_++)
                h->thr_isum[(size_t)e * KFLRL_N_VEHICLES + vi_] +=
                    KFLRL_THRSC(h, e, vi_) * step_dt;
            h->thr_itime[e] += step_dt;
#endif
#if KFLRL_N_VEHICLES > 0
            /* The gravity-gradient torque is part of the torque sum
             * and is computed per vehicle from its own separation
             * from the body it orbits. A vehicle whose body declares
             * no parent has no attractor named and takes no torque
             * from this source. */
            K26V3 gg[KFLRL_N_VEHICLES];
            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
                gg[vi].x = gg[vi].y = gg[vi].z = 0.0;
                K26AstroVehicle *veh =
                    h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi];
                if (!veh) continue;
                const K26AstroBody *vb = k26astro_world_body_at(
                    h->worlds[e], kflrl_body_idx_[kflrl_vehicle_body_[vi]]);
                if (!vb || vb->parent_body_idx < 0) continue;
                const K26AstroBody *pb = k26astro_world_body_at(
                    h->worlds[e], vb->parent_body_idx);
                if (!pb) continue;
                K26V3 rw = k26astro_pos_sub(&pb->pos, &vb->pos);
                (void)k26astro_att_gravity_gradient(veh, rw, pb->gm,
                                                    &gg[vi]);
            }
            /* Each vehicle advances with its own actuators, its own
             * gravity-gradient torque, and the local magnetic field
             * a magnetorquer needs. The actuator view is built over
             * the environment's state and the momenta are stored
             * back, so the state lives where the reset can clear it
             * and nothing here allocates. */
            K26AstroAttStatus ast = K26ASTRO_ATT_OK;
            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
                K26AstroVehicle *veh =
                    h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi];
                if (!veh) continue;
                K26AstroAttWheel wbuf[KFLRL_N_WHEELS > 0 ?
                                      KFLRL_N_WHEELS : 1];
                K26AstroAttTorquer qbuf[KFLRL_N_TORQUERS > 0 ?
                                        KFLRL_N_TORQUERS : 1];
                K26AstroAttThruster tbuf[KFLRL_N_THRUSTERS > 0 ?
                                         KFLRL_N_THRUSTERS : 1];
                int wmap[KFLRL_N_WHEELS > 0 ? KFLRL_N_WHEELS : 1];
                K26AstroAttActuators view;
                kflrl_act_view_(&h->act[e], vi, KFLRL_COM(h, e, vi),
                                KFLRL_THRSC(h, e, vi),
                                wbuf, qbuf, tbuf, &view, wmap);
                K26V3 bfield = kflrl_field_body_(h->worlds[e], veh, vi);
                ast = k26astro_att_step_actuated(veh, &view, gg[vi],
                                                 bfield, step_dt);
                /* The wheel momenta are written back only when the
                 * advance stood. A failed advance leaves the
                 * orientation and rate as they were, so writing the
                 * momenta back would leave the two halves of one
                 * step disagreeing about whether it happened. */
                if (ast != K26ASTRO_ATT_OK) break;
                kflrl_act_store_(&h->act[e], &view, wmap);
            }
            if (ast != K26ASTRO_ATT_OK) {
                att_reason = (ast == K26ASTRO_ATT_E_DIVERGED)
                    ? (uint16_t)K26RL_E_DIVERGED
                    : (uint16_t)K26RL_E_ENV_INTERNAL;
                break;
            }
#if KFLRL_N_PROPELLANT > 0
            /* The propellant is gone, and everything that depends on
             * it follows: the craft's mass, its gravitational
             * parameter, the centre of mass every thruster torque is
             * taken about, and the inertia tensor whose inverse the
             * rotational equation needs. The tank moves the centre of
             * mass unless it sits on it, so a craft's attitude
             * authority changes over a long burn; that is the
             * arrangement the assembly declared and not an artefact
             * of this update.
             *
             * The debit is taken after the advance rather than
             * before it, so the sub-interval is integrated at the
             * mass it began with. An advance that did not stand
             * leaves the tank alone, on the same terms as the wheel
             * momenta above: the interval did not happen, so nothing
             * it would have spent did either.
             *
             * The burn is then registered in the closed-system
             * account, one entry per firing thruster carrying that
             * thruster's own share of the mass, its own exhaust
             * speed and the direction its own exhaust left in. One
             * entry for the whole craft would need a single exhaust
             * speed where thrusters differ in specific impulse, and
             * a single direction, which two opposed thrusters do not
             * have: an attitude couple has no net thrust direction
             * at all while its exhaust still carries real momentum
             * away in two directions. The shares are the same
             * arithmetic the debit above summed, so the account and
             * the tank agree to rounding. */
            for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
                if (!(burned[vi] > 0.0)) continue;
                size_t pix = (size_t)e * KFLRL_N_VEHICLES + vi;
                double left = h->prop[pix] - burned[vi];
                if (left < 0.0) left = 0.0;
                h->prop[pix] = left;
                K26AstroBody *vb = k26astro_world_body_at(h->worlds[e],
                    kflrl_body_idx_[kflrl_vehicle_body_[vi]]);
                kflrl_apply_props_(h->vehicles[pix], vb, vi, left,
                                   KFLRL_COM(h, e, vi));
#if KFLRL_N_THRUSTERS > 0
                if (!vb) continue;
                double sc = KFLRL_THRSC(h, e, vi);
                for (int i = 0; i < KFLRL_N_THRUSTERS; i++) {
                    if (kflrl_thruster_veh_[i] != vi) continue;
                    double u = h->act[e].cmd[
                        KFLRL_N_WHEELS + KFLRL_N_TORQUERS + i];
                    u = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
                    if (u == 0.0) continue;
                    double ve = kflrl_thruster_desc_[i].isp_s * KFLRL_G0;
                    double dm = u * kflrl_thruster_desc_[i].max_thrust /
                                ve * step_dt * sc;
                    if (!(dm > 0.0)) continue;
                    K26V3 dw = k26m3d_quat_rotate_v3(vb->attitude,
                        kflrl_thruster_desc_[i].dir);
                    k26astro_rt_ledger_record_burn(h->ledger[e], dm,
                        vb->vel.x, vb->vel.y, vb->vel.z, ve,
                        -dw.x, -dw.y, -dw.z);
                }
#endif
            }
#endif
#endif
#if KFLRL_N_COLL > 0
            /* The collision pass, between the sub-advances. A body
             * that crosses a target inside one control period is not
             * found by comparing the period's endpoints, and
             * shortening the step to catch it would make the applied
             * duration a function of the geometry; sweeping inside
             * the sub-advance finds it and leaves the duration
             * exactly what was declared.
             *
             * Every position is taken as an exact difference from
             * one reference rather than as a flattened coordinate,
             * because a position here is a sector index and a
             * bounded offset, and flattening two of them before
             * subtracting throws away the precision the sector grid
             * exists to keep. */
            {
                K26AstroCollBody cbody[KFLRL_N_VEHICLES];
                const K26AstroPos *cref = &csnap[0];
                for (int vi = 0; vi < KFLRL_N_VEHICLES; vi++) {
                    memset(&cbody[vi], 0, sizeof cbody[vi]);
                    const K26AstroBody *cb = k26astro_world_body_at(
                        h->worlds[e],
                        kflrl_body_idx_[kflrl_vehicle_body_[vi]]);
                    if (!cb) continue;
                    cbody[vi].pos0 = k26astro_pos_sub(&csnap[vi], cref);
                    cbody[vi].pos1 = k26astro_pos_sub(&cb->pos, cref);
                    cbody[vi].orientation = cquat[vi];
                    cbody[vi].vel0 = cvel0[vi];
                    cbody[vi].vel1 = cb->vel;
                    cbody[vi].shapes = &kflrl_coll_[kflrl_coll_first_[vi]];
                    cbody[vi].n_shapes = kflrl_coll_count_[vi];
                    cbody[vi].bound_radius = kflrl_veh_bound_[vi];
                    cbody[vi].mass = cb->mass;
                    {
                        const double *co_ = KFLRL_COM(h, e, vi);
                        cbody[vi].com_offset =
                            k26m3d_v3(co_[0], co_[1], co_[2]);
                    }
#if KFLRL_N_PORTS > 1
                    /* A joined pair is one body, and the pass tests
                     * pairs of bodies, so the follower's primitives
                     * leave it; left in, the join itself would be
                     * reported as a contact on every sub-advance. */
                    if (h->join[e].active &&
                        h->join[e].follower == vi) {
                        cbody[vi].n_shapes = 0;
                    }
#endif
                    cbody[vi].omega = cb->omega;
                    /* The inverse inertia the angular half of an
                     * impulse turns on, taken from the vehicle's own
                     * attitude state, which is where the assembly's
                     * derived tensor was installed and inverted. Left
                     * at the zero matrix by the clearing above, an
                     * off-centre impact would produce no spin at all
                     * while every other term looked right. */
                    {
                        K26AstroVehicle *cvh =
                            h->vehicles[(size_t)e * KFLRL_N_VEHICLES + vi];
                        const K26AstroAttitudeStateExt *cx = cvh
                            ? k26astro_vehicle_attitude_ext(cvh) : NULL;
                        if (cx) cbody[vi].inv_inertia = cx->inertia_inverse;
                    }
                }
                K26AstroCollContact cc;
                if (k26astro_coll_pass(cbody, KFLRL_N_VEHICLES, step_dt,
                                       &cc) == K26ASTRO_COLL_OK &&
                    cc.hit) {
                    int captured = 0;
#if KFLRL_N_PORTS > 1
                    /* Was this contact between two docking
                     * interfaces, and did it satisfy the envelope?
                     * Both are decided BEFORE any resolution runs,
                     * because a capture takes precedence over the
                     * environment's declared resolution: a
                     * programme cannot be told its craft docked and
                     * then shown it flung away.
                     *
                     * The verdict is the pair's, not one port's.
                     * Either port can be read as the active one and
                     * the two readings differ slightly, so a
                     * capture requires both: the conservative
                     * reading, and the one that leaves the two
                     * ports' channels agreeing about a fact of the
                     * pair. */
                    int pidx[2] = { -1, -1 };
                    K26AstroCollPortState pst[2];
                    int pair0[2] = { cc.body_a, cc.body_b };
                    for (int q = 0; q < KFLRL_N_PORTS; q++) {
                        if (kflrl_ports_[q].veh == cc.body_a &&
                            kflrl_ports_[q].shape == cc.shape_a) pidx[0] = q;
                        if (kflrl_ports_[q].veh == cc.body_b &&
                            kflrl_ports_[q].shape == cc.shape_b) pidx[1] = q;
                    }
                    memset(pst, 0, sizeof pst);
                    if (pidx[0] >= 0 && pidx[1] >= 0) {
                        captured = 1;
                        for (int q = 0; q < 2; q++) {
                            if (k26astro_coll_port_state(
                                    &cbody[pair0[q]],
                                    &kflrl_ports_[pidx[q]].geom,
                                    &cbody[pair0[1 - q]],
                                    &kflrl_ports_[pidx[1 - q]].geom,
                                    cc.time, &pst[q]) != K26ASTRO_COLL_OK) {
                                pidx[0] = -1;
                                captured = 0;
                                break;
                            }
                            if (!k26astro_coll_port_captured(
                                    &pst[q], &kflrl_ports_[pidx[q]].env)) {
                                captured = 0;
                            }
                        }
                    }
#endif
                    /* Arrest, the default resolution: the pair meets
                     * at the sweep's own interpolated configuration,
                     * so the reported contact and the state it
                     * resolves from agree exactly, and the relative
                     * velocity is removed by a momentum-conserving
                     * merge. The remainder of the control period
                     * advances with the pair moving together, which
                     * the correction below is what carries out. */
                    K26V3 apos, avel, bpos, bvel;
                    K26V3 awb = cbody[cc.body_a].omega;
                    K26V3 bwb = cbody[cc.body_b].omega;
#if KFLRL_CONTACT_BOUNCE
                    /* Bounce: one impulse at the contact point,
                     * using the effective mass there, so an
                     * off-centre impact spins the body by the
                     * amount the geometry gives. */
                    K26AstroCollStatus cst = captured
                        ? K26ASTRO_COLL_OK
                        : k26astro_coll_bounce(
                        &cbody[cc.body_a], &cbody[cc.body_b], &cc,
                        h->restitution, h->friction,
                        &apos, &avel, &awb, &bpos, &bvel, &bwb);
#else
                    K26AstroCollStatus cst = captured
                        ? K26ASTRO_COLL_OK
                        : k26astro_coll_arrest(
                        &cbody[cc.body_a], &cbody[cc.body_b], cc.time,
                        &apos, &avel, &bpos, &bvel);
#endif
                    /* A capture is not one of the declared
                     * resolutions and does not run either of them.
                     * It makes the pair one body instead. */
                    if (captured) cst = K26ASTRO_COLL_E_NULL;
#if KFLRL_N_PORTS > 1
                    if (captured) {
                        kflrl_join_form_(h, e, cbody, cc.body_a,
                                         cc.body_b, cc.time,
                                         (1.0 - cc.time) * step_dt,
                                         cref);
                    }
#endif
                    /* What the contact did, carried to the end of
                     * the sub-advance.
                     *
                     * The resolution answers at the contact instant,
                     * part way through the sub-advance; the world
                     * stands at the end of it, and that is where the
                     * observation is taken. Writing the resolved
                     * state in as it stands would leave the pair at a
                     * moment the rest of the world has already left,
                     * and a relative observe taken from a body of the
                     * pair would carry that gap: the chief would
                     * report every other body displaced by its own
                     * unadvanced travel, which at orbital speed is
                     * hundreds of metres for a fraction of a second.
                     *
                     * What the contact changed is each body's
                     * velocity, from the one it arrived with to the
                     * one the resolution gave it. That change is
                     * applied to the state the advance produced,
                     * together with the displacement it makes over
                     * the rest of the sub-advance, so the pair ends
                     * where the resolution puts it and at the instant
                     * every other body is at, with the gravity and
                     * thrust of that sub-advance kept rather than
                     * discarded.
                     *
                     * The arrival velocity is the interpolation
                     * between the two endpoints handed to the pass,
                     * which is the linear model the pass resolves the
                     * contact under; the resolved velocity is on that
                     * same model, so their difference is the whole of
                     * what it did and nothing of what the advance
                     * did.
                     *
                     * A contact with nothing left to remove changes
                     * nothing here, which is what keeps a pair that
                     * has come to rest against each other moving: it
                     * reports a contact at the start of every later
                     * sub-advance, and a resolution written into the
                     * world in place would put the pair back at that
                     * start each time and hold it there for the rest
                     * of the episode. */
                    if (cst == K26ASTRO_COLL_OK) {
                        int pair[2] = { cc.body_a, cc.body_b };
                        K26V3 nv[2] = { avel, bvel };
                        K26V3 nw[2] = { awb, bwb };
                        double crem = (1.0 - cc.time) * step_dt;
                        for (int q = 0; q < 2; q++) {
                            int vi = pair[q];
                            K26AstroBody *cb = k26astro_world_body_at(
                                h->worlds[e],
                                kflrl_body_idx_[kflrl_vehicle_body_[vi]]);
                            if (!cb) continue;
                            K26V3 v0 = cbody[vi].vel0;
                            K26V3 v1 = cbody[vi].vel1;
                            K26V3 dv = k26m3d_v3(
                                nv[q].x - (v0.x + (v1.x - v0.x) * cc.time),
                                nv[q].y - (v0.y + (v1.y - v0.y) * cc.time),
                                nv[q].z - (v0.z + (v1.z - v0.z) * cc.time));
                            /* The correction is applied as a delta so
                             * the sector representation is preserved
                             * rather than rebuilt from a flattened
                             * coordinate. */
                            k26astro_pos_add(&cb->pos, k26m3d_v3(
                                dv.x * crem, dv.y * crem, dv.z * crem));
                            cb->vel = k26m3d_v3(cb->vel.x + dv.x,
                                                cb->vel.y + dv.y,
                                                cb->vel.z + dv.z);
                            cb->omega = nw[q];
                        }
                    }
                    /* The fraction the channels publish is of the
                     * whole control period, not of this sub-advance,
                     * and the first contact of the transition is the
                     * one that is kept. */
                    double cfrac = h->control_dt > 0.0
                        ? ((advanced - step_dt) + cc.time * step_dt)
                          / h->control_dt
                        : 0.0;
                    int pair[2] = { cc.body_a, cc.body_b };
                    for (int q = 0; q < 2; q++) {
                        KflrlContact *ct = &h->contact[
                            (size_t)e * KFLRL_N_CONTACT + pair[q]];
                        if (ct->hit != 0.0) continue;
                        ct->hit      = 1.0;
                        ct->fraction = cfrac;
                        ct->speed    = cc.speed;
#if KFLRL_N_PORTS > 1
                        if (pidx[0] < 0 || pidx[1] < 0) continue;
                        /* The residuals are of this body's own port
                         * as the active one, which is the sense the
                         * form that reads them names, taken at the
                         * impact configuration and computed above,
                         * before any resolution reached the world.
                         * The capture flag is the pair's and is the
                         * same on both ports. */
                        ct->port_hit   = 1.0;
                        ct->captured   = captured ? 1.0 : 0.0;
                        ct->axial      = pst[q].axial;
                        ct->lateral    = pst[q].lateral;
                        ct->pitchyaw   = pst[q].pitchyaw;
                        ct->roll       = pst[q].roll;
                        ct->v_axial    = pst[q].v_axial;
                        ct->v_lateral  = pst[q].v_lateral;
                        ct->v_pitchyaw = pst[q].v_pitchyaw;
                        ct->v_roll     = pst[q].v_roll;
#endif
                    }
                }
#if KFLRL_N_PORTS > 1
                /* A joined pair advances as one body: the follower
                 * is placed from the carrier at the end of every
                 * sub-advance, including the one that formed the
                 * join, rather than integrated on its own. */
                kflrl_join_impose_(h, e, cref);
#endif
            }
#endif
#if KFLRL_N_PAYLOAD > 0
            /* The information state is pushed at the end of each
             * sub-advance, after the collision pass has had its say,
             * so what the history holds is the state the transition
             * actually produced. */
            h->info_t[e] += step_dt;
            kflrl_info_push_(h, e);
#endif
        }
        if (att_reason != 0) {
            K26RlStatus fst = kflrl_fault_(h, e, aslice, att_reason);
            if (fst != K26RL_OK) return fst;
            continue;
        }
        if (rc != 0) {
            int code = rc < 0 ? -rc : rc;
            if (code == K26ASTRO_RT_E_FPU_RACE) return K26RL_E_FPU_RACE;
            if (code == K26ASTRO_RT_E_OOM) return K26RL_E_INTERNAL;
            uint16_t reason = (code == K26ASTRO_RT_E_INTEGRATOR)
                ? (uint16_t)K26RL_E_DIVERGED
                : (uint16_t)K26RL_E_ENV_INTERNAL;
            K26RlStatus fst = kflrl_fault_(h, e, aslice, reason);
            if (fst != K26RL_OK) return fst;
            continue;
        }

        kflrl_observe_(h->worlds[e], h->scratch,
                       &h->contact[(size_t)e * KFLRL_N_CONTACT],
                       &h->join[e], KFLRL_PAYH(h, e), KFLRL_PAYP(h, e),
                       KFLRL_VEHS(h, e), KFLRL_PROP(h, e),
                       KFLRL_COM(h, e, 0), KFLRL_INFODAY(h, e),
                       KFLRL_INFOT(h, e), KFLRL_ENG(h, e),
#@ The transition has been taken but the count is still the one
#@ before it, so the episode's elapsed time is one control period
#@ more than the count says.
                       (double)(h->steps[e] + 1u) * h->control_dt);
        /* The transition index is the draw index every per-step term
         * uses, and h->steps[e] is still the count before this
         * transition, so the first transition of an episode draws at
         * index 0. */
        kflrl_sense_apply_(h, e, h->episode[e], h->steps[e],
                           h->scratch);
        int finite = 1;
        for (uint32_t j = 0; j < KFLRL_OBS_TOTAL; j++) {
            if (!std::isfinite(h->scratch[j])) finite = 0;
        }
        if (!finite) {
            K26RlStatus fst = kflrl_fault_(h, e, aslice,
                                           (uint16_t)K26RL_E_DIVERGED);
            if (fst != K26RL_OK) return fst;
            continue;
        }

        uint32_t ns = h->steps[e] + 1u;
        /* One reward per agent, in agent-index order. A single
         * non-finite value faults the whole environment, because
         * termination is per environment: there is no state in which
         * one agent's episode has ended and another's has not. */
        double r[KFLRL_N_AGENTS];
        int r_finite = 1;
        kflrl_rewards_(h->scratch, aslice, ns, wslice, r);
        for (int a = 0; a < KFLRL_N_AGENTS; a++) {
            if (!std::isfinite(r[a])) r_finite = 0;
        }
        if (!r_finite) {
            K26RlStatus fst = kflrl_fault_(
                h, e, aslice, (uint16_t)K26RL_E_ENV_INTERNAL);
            if (fst != K26RL_OK) return fst;
            continue;
        }

        /* Terminal adjustment on termination only; truncation
         * carries none. Evaluated before the transition commits so
         * a non-finite adjusted reward faults like a non-finite
         * reward, never reaching the recorded stream. */
        int term = kflrl_terminated_(h->scratch, aslice, ns, wslice);
        double tadj[KFLRL_N_AGENTS];
        for (int a = 0; a < KFLRL_N_AGENTS; a++) tadj[a] = 0.0;
        if (term) {
            kflrl_terminals_(h->scratch, aslice, ns, wslice, tadj);
            for (int a = 0; a < KFLRL_N_AGENTS; a++) {
                r[a] += tadj[a];
                if (!std::isfinite(r[a])) r_finite = 0;
            }
            if (!r_finite) {
                K26RlStatus fst = kflrl_fault_(
                    h, e, aslice, (uint16_t)K26RL_E_ENV_INTERNAL);
                if (fst != K26RL_OK) return fst;
                continue;
            }
        }

        /* The transition commits. Termination and truncation are
         * distinct outcomes: termination wins when both land on one
         * step, and the flag word agrees in kind with the episode
         * file's end reason. */
        h->steps[e] = ns;
        memcpy(h->obs + (size_t)e * KFLRL_OBS_TOTAL, h->scratch,
               sizeof(double) * KFLRL_OBS_TOTAL);
        uint32_t f = 0;
        if (term) {
            f |= K26RL_FLAG_TERMINATED;
        } else if (h->horizon != 0 && ns >= h->horizon) {
            f |= K26RL_FLAG_TRUNCATED;
        }
        for (int a = 0; a < KFLRL_N_AGENTS; a++) {
            h->rew[(size_t)e * KFLRL_N_AGENTS + a] = r[a];
        }
        h->flags[e] = f;
        h->fault[e] = 0;
        if (f & (K26RL_FLAG_TERMINATED | K26RL_FLAG_TRUNCATED)) {
            h->ended[e] = 1;
        }
        if (h->writer) {
            K26RlStatus st = k26rl_episode_writer_step(
                h->writer, e, h->obs + (size_t)e * KFLRL_OBS_TOTAL,
                aslice, h->rew + (size_t)e * KFLRL_N_AGENTS, f,
                h->control_dt);
            if (st != K26RL_OK) return K26RL_E_INTERNAL;
            if (h->ended[e]) {
                st = k26rl_episode_writer_end(
                    h->writer, e,
                    term ? K26RL_END_TERMINATED : K26RL_END_TRUNCATED,
                    0, tadj);
                if (st != K26RL_OK) return K26RL_E_INTERNAL;
#if KFLRL_N_PLAN > 0
                kflrl_plan_write_(h, e, aslice);
#endif
            }
        }
        if (h->tap) {
            k26rl_tap_step(h->tap, e,
                           h->obs + (size_t)e * KFLRL_OBS_TOTAL,
                           aslice,
                           h->rew + (size_t)e * KFLRL_N_AGENTS, f,
                           h->control_dt);
            if (h->ended[e]) {
                k26rl_tap_end(
                    h->tap, e,
                    term ? K26RL_END_TERMINATED : K26RL_END_TRUNCATED,
                    0, tadj);
            }
        }
    }
    return K26RL_OK;
}

/* Shared by the two explicit reset calls. With output enabled, an
 * episode cut mid-flight records a truncated end (the recording
 * ended; the old ordinal's frames precede any rekey frame), then
 * every environment restarts and re-emits its episode-start. */
static K26RlStatus kflrl_reset_all_(K26RlEnv *h, int rekey,
                                    uint64_t new_seed)
{
    if (!rekey) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            if (h->episode[e] == 0xFFFFFFFFu) {
                return K26RL_E_RNG_EXHAUSTED;
            }
        }
    }
    double zero_adj[KFLRL_N_AGENTS];
    for (int a = 0; a < KFLRL_N_AGENTS; a++) zero_adj[a] = 0.0;
    if (h->writer) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            if (h->ended[e]) continue;   /* end frame already written */
            K26RlStatus st = k26rl_episode_writer_end(
                h->writer, e, K26RL_END_TRUNCATED, 0, zero_adj);
            if (st != K26RL_OK) return K26RL_E_INTERNAL;
        }
    }
    if (h->tap) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            if (h->ended[e]) continue;   /* end frame already published */
            k26rl_tap_end(h->tap, e, K26RL_END_TRUNCATED, 0, zero_adj);
        }
    }
    if (rekey) {
        h->seed = new_seed;
        h->key = k26rng_key(new_seed);
        h->rekey_ordinal += 1u;
        if (h->writer) {
            K26RlStatus st = k26rl_episode_writer_rekey(
                h->writer, new_seed, NULL);
            if (st != K26RL_OK) return K26RL_E_INTERNAL;
        }
        if (h->tap) k26rl_tap_rekey(h->tap, new_seed);
    }
    for (uint32_t e = 0; e < h->n_envs; e++) {
        uint32_t ep = rekey ? 0u : h->episode[e] + 1u;
        kflrl_reset_env_(h, e, ep);
        h->flags[e] = 0;
    }
    h->at_boundary = 1;
    if (h->writer) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            K26RlStatus st = k26rl_episode_writer_start(
                h->writer, e, h->episode[e],
                h->obs + (size_t)e * KFLRL_OBS_TOTAL,
#if KFLRL_N_REC > 0
                kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,
#else
                NULL, NULL,
#endif
                KFLRL_N_REC);
            if (st != K26RL_OK) return K26RL_E_INTERNAL;
        }
    }
    if (h->tap) {
        for (uint32_t e = 0; e < h->n_envs; e++) {
            k26rl_tap_start(
                h->tap, e, h->episode[e],
                h->obs + (size_t)e * KFLRL_OBS_TOTAL,
#if KFLRL_N_REC > 0
                kflrl_dr_tags_, h->dr_vals + (size_t)e * KFLRL_N_REC,
#else
                NULL, NULL,
#endif
                KFLRL_N_REC);
        }
    }
    return K26RL_OK;
}

extern "C" K26RlStatus k26rl_env_reset(K26RlEnv *h)
{
    if (!h) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    return kflrl_reset_all_(h, 0, 0);
}

extern "C" K26RlStatus k26rl_env_reset_seeded(K26RlEnv *h, uint64_t seed)
{
    if (!h) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    for (uint32_t i = 0; i < h->n_seen; i++) {
        if (h->seen_seeds[i] == seed) return K26RL_E_SEED_REUSE;
    }
    if (h->n_seen == h->cap_seen) {
        uint32_t nc = h->cap_seen * 2u;
        uint64_t *ns = (uint64_t *)realloc(h->seen_seeds,
                                           nc * sizeof(uint64_t));
        if (!ns) return K26RL_E_INTERNAL;
        h->seen_seeds = ns;
        h->cap_seen = nc;
    }
    h->seen_seeds[h->n_seen++] = seed;
    return kflrl_reset_all_(h, 1, seed);
}

extern "C" K26RlStatus k26rl_env_obs(const K26RlEnv *h, double *out)
{
    if (!h || !out) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    memcpy(out, h->obs,
           sizeof(double) * (size_t)h->n_envs * KFLRL_OBS_TOTAL);
    return K26RL_OK;
}

extern "C" K26RlStatus k26rl_env_reward(const K26RlEnv *h, double *out)
{
    if (!h || !out) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    memcpy(out, h->rew,
           sizeof(double) * (size_t)h->n_envs * KFLRL_N_AGENTS);
    return K26RL_OK;
}

extern "C" K26RlStatus k26rl_env_flags(const K26RlEnv *h, uint32_t *out)
{
    if (!h || !out) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    memcpy(out, h->flags, sizeof(uint32_t) * h->n_envs);
    return K26RL_OK;
}

extern "C" K26RlStatus k26rl_env_fault_codes(const K26RlEnv *h,
                                              uint16_t *out)
{
    if (!h || !out) return K26RL_E_NULL;
    if (!kflrl_live_(h)) return K26RL_E_USE_AFTER_DESTROY;
    memcpy(out, h->fault, sizeof(uint16_t) * h->n_envs);
    return K26RL_OK;
}

extern "C" int32_t k26rl_env_spec(const K26RlEnv *h, uint8_t *out,
                                   uint32_t capacity)
{
    if (!h) return -(int32_t)K26RL_E_NULL;
    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;
    if (capacity >= h->spec_len) {
        if (!out) return -(int32_t)K26RL_E_NULL;
        memcpy(out, h->spec, h->spec_len);
    }
    return (int32_t)h->spec_len;
}

/* Body states: positions relative to the reference body, computed
 * with the runtime's exact position subtraction rather than by
 * flattening two absolute coordinates, and the bodies' own
 * velocities. Sized like the spec getter, env-major like every
 * other buffer, and a pure read that allocates nothing. */
extern "C" int32_t k26rl_env_bodies(const K26RlEnv *h, uint32_t reference,
                                     double *out, uint32_t capacity)
{
    if (!h) return -(int32_t)K26RL_E_NULL;
    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;
#if KFLRL_N_BODIES <= 0
    (void)reference; (void)out; (void)capacity;
    return 0;
#else
    if (reference != K26RL_BODY_REF_ORIGIN &&
        reference >= (uint32_t)KFLRL_N_BODIES) {
        return -(int32_t)K26RL_E_GEOMETRY;
    }
    uint64_t need = (uint64_t)h->n_envs * (uint32_t)KFLRL_N_BODIES * 6u;
    if (need > 0x7FFFFFFFu) return -(int32_t)K26RL_E_GEOMETRY;
    if (capacity < need) return (int32_t)need;
    if (!out) return -(int32_t)K26RL_E_NULL;
    for (uint32_t e = 0; e < h->n_envs; e++) {
        K26AstroWorld *w = h->worlds[e];
        const K26AstroBody *rb = NULL;
        if (reference != K26RL_BODY_REF_ORIGIN) {
            rb = k26astro_world_body_at(w, kflrl_body_idx_[reference]);
        }
        for (uint32_t b = 0; b < (uint32_t)KFLRL_N_BODIES; b++) {
            const K26AstroBody *bd =
                k26astro_world_body_at(w, kflrl_body_idx_[b]);
            double *o = out + ((size_t)e * (uint32_t)KFLRL_N_BODIES + b) * 6;
            K26V3 r;
            if (!bd) {
                o[0] = o[1] = o[2] = o[3] = o[4] = o[5] = 0.0;
                continue;
            }
            if (rb) {
                r = k26astro_pos_sub(&bd->pos, &rb->pos);
            } else {
                K26AstroPos origin;
                memset(&origin, 0, sizeof origin);
                r = k26astro_pos_sub(&bd->pos, &origin);
            }
            o[0] = r.x; o[1] = r.y; o[2] = r.z;
            o[3] = bd->vel.x; o[4] = bd->vel.y; o[5] = bd->vel.z;
        }
    }
    return (int32_t)need;
#endif
}

extern "C" int32_t k26rl_env_attitudes(const K26RlEnv *h, double *out,
                                        uint32_t capacity)
{
    if (!h) return -(int32_t)K26RL_E_NULL;
    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;
#if KFLRL_N_BODIES <= 0
    (void)out; (void)capacity;
    return 0;
#else
    uint64_t need = (uint64_t)h->n_envs * (uint32_t)KFLRL_N_BODIES * 7u;
    if (need > 0x7FFFFFFFu) return -(int32_t)K26RL_E_GEOMETRY;
    if (capacity < need) return (int32_t)need;
    if (!out) return -(int32_t)K26RL_E_NULL;
    for (uint32_t e = 0; e < h->n_envs; e++) {
        K26AstroWorld *w = h->worlds[e];
        for (uint32_t b = 0; b < (uint32_t)KFLRL_N_BODIES; b++) {
            const K26AstroBody *bd =
                k26astro_world_body_at(w, kflrl_body_idx_[b]);
            double *o = out + ((size_t)e * (uint32_t)KFLRL_N_BODIES + b) * 7;
            if (!bd) {
                /* No body, no orientation: the identity, which is
                 * what an untracked attitude holds. */
                o[0] = 1.0; o[1] = o[2] = o[3] = 0.0;
                o[4] = o[5] = o[6] = 0.0;
                continue;
            }
            o[0] = bd->attitude.w; o[1] = bd->attitude.x;
            o[2] = bd->attitude.y; o[3] = bd->attitude.z;
            o[4] = bd->omega.x; o[5] = bd->omega.y; o[6] = bd->omega.z;
        }
    }
    return (int32_t)need;
#endif
}

/* The actuator getter reports what the latest step drove: exact
 * imparted force for a thruster, the command clamped to the limit
 * for a wheel or a torquer, per the header's contract. The clamps
 * here mirror the actuator library's own, a non-finite command
 * included, so the figure is the one the dynamics saw and not the
 * one the program assigned. The thruster's propellant fraction is
 * the step mean the step integrated, not the last sub-interval's,
 * so the step a tank runs dry in reports the force it did impart
 * rather than the zero it ended on. */
extern "C" int32_t k26rl_env_actuators(const K26RlEnv *h, double *out,
                                       uint32_t capacity)
{
    if (!h) return -(int32_t)K26RL_E_NULL;
    if (!kflrl_live_(h)) return -(int32_t)K26RL_E_USE_AFTER_DESTROY;
#if KFLRL_N_CMD <= 0
    (void)out; (void)capacity;
    return 0;
#else
    uint64_t need = (uint64_t)h->n_envs * (uint32_t)KFLRL_N_CMD * 10u;
    if (need > 0x7FFFFFFFu) return -(int32_t)K26RL_E_GEOMETRY;
    if (capacity < need) return (int32_t)need;
    if (!out) return -(int32_t)K26RL_E_NULL;
    for (uint32_t e = 0; e < h->n_envs; e++) {
        const KflrlAct *a = &h->act[e];
        double *o = out + (size_t)e * (uint32_t)KFLRL_N_CMD * 10;
#if KFLRL_N_WHEELS > 0
        for (int i = 0; i < KFLRL_N_WHEELS; i++, o += 10) {
            const K26AstroAttWheel *d = &kflrl_wheel_desc_[i];
            double u = a->cmd[i];
            if (!std::isfinite(u)) u = 0.0;
            u = u < -d->max_torque ? -d->max_torque
              : (u > d->max_torque ? d->max_torque : u);
            o[0] = (double)kflrl_vehicle_body_[kflrl_wheel_veh_[i]];
            o[1] = 0.0;
            o[2] = 0.0; o[3] = 0.0; o[4] = 0.0;
            o[5] = d->axis.x; o[6] = d->axis.y; o[7] = d->axis.z;
            o[8] = u; o[9] = d->max_torque;
        }
#endif
#if KFLRL_N_TORQUERS > 0
        for (int i = 0; i < KFLRL_N_TORQUERS; i++, o += 10) {
            const K26AstroAttTorquer *d = &kflrl_torquer_desc_[i];
            double u = a->cmd[KFLRL_N_WHEELS + i];
            if (!std::isfinite(u)) u = 0.0;
            u = u < -d->max_dipole ? -d->max_dipole
              : (u > d->max_dipole ? d->max_dipole : u);
            o[0] = (double)kflrl_vehicle_body_[kflrl_torquer_veh_[i]];
            o[1] = 1.0;
            o[2] = 0.0; o[3] = 0.0; o[4] = 0.0;
            o[5] = d->axis.x; o[6] = d->axis.y; o[7] = d->axis.z;
            o[8] = u; o[9] = d->max_dipole;
        }
#endif
#if KFLRL_N_THRUSTERS > 0
        for (int i = 0; i < KFLRL_N_THRUSTERS; i++, o += 10) {
            const K26AstroAttThruster *d = &kflrl_thruster_desc_[i];
            int vi = kflrl_thruster_veh_[i];
            double u =
                a->cmd[KFLRL_N_WHEELS + KFLRL_N_TORQUERS + i];
            if (!std::isfinite(u)) u = 0.0;
            u = u < 0.0 ? 0.0 : (u > 1.0 ? 1.0 : u);
            o[0] = (double)kflrl_vehicle_body_[vi];
            o[1] = 2.0;
            o[2] = d->at.x; o[3] = d->at.y; o[4] = d->at.z;
            o[5] = d->dir.x; o[6] = d->dir.y; o[7] = d->dir.z;
            o[8] = u * KFLRL_THRMEAN(h, e, vi) * d->max_thrust;
            o[9] = d->max_thrust;
        }
#endif
    }
    return (int32_t)need;
#endif
}

extern "C" void k26rl_env_destroy(K26RlEnv *h)
{
    if (!h || !kflrl_live_(h)) return;
    if (h->writer) {
        (void)k26rl_episode_writer_close(h->writer);
        h->writer = NULL;
    }
    if (h->tap) {
        k26rl_tap_close(h->tap);
        h->tap = NULL;
    }
    h->magic = 0;
    kflrl_free_handle_(h);
}

