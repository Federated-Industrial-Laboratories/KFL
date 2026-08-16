/* test_rl_hotpath.c: the stepping hot loop's no-allocation, no-I/O
 * contract (k26rl_env.h: "The stepping hot loop performs no
 * allocation and no I/O: the per-step path touches preallocated
 * memory only").
 *
 * Mechanism: the gate compiles a small interposer shared object and
 * re-executes itself under LD_PRELOAD of it, so malloc, calloc,
 * realloc, free, write, and fwrite from every module in the child
 * process (the dlopened .rlenv.so included) are counted while a
 * per-region arming flag is set. The counters are proven live by a
 * positive control: a deliberate malloc + free + write inside an
 * armed window must read exactly 1/1/1 (and a deliberate fwrite
 * exactly 1) before the zero assertions mean anything.
 *
 * Gates (serve surface, dlopen):
 *   1. Positive control, as above.
 *   2. Output disabled: create, reset, then 60 steps of 2
 *      environments with the counters armed across the steps only.
 *      The window crosses two horizon-24 truncation boundaries per
 *      environment, so boundary auto-resets are inside the pinned
 *      region. All six counters must read zero.
 *   3. Output enabled: same drive with an episode file attached.
 *      Allocation counters must still read zero; write activity is
 *      bounded, not zero: the writer emits one fwrite per finished
 *      frame, and within this window frames leave only at episode
 *      boundaries (steps_per_chunk is 1024, so no chunk fills
 *      mid-episode; each boundary flushes at most a partial chunk
 *      frame, an end record, and the next start record: 3 frames
 *      per boundary per environment). 2 environments x 2 boundaries
 *      x 3 frames gives 12; the bound asserts (0, 12] and at least
 *      one write-family call.
 *   4. Telemetry tap enabled, no consumer: the same drive again. The
 *      acceptance is equality with gate 2's counts rather than a
 *      bound of its own, because publication into a shared mapping is
 *      a store and not a system call. Both counts must therefore be
 *      exactly what the untapped drive read.
 *   5. Tap enabled with a consumer attached, reading continuously in
 *      its own process for the whole window: the same equality. This
 *      is where an implementation that published over a pipe, or that
 *      remapped or allocated per frame, or that waited on a consumer,
 *      would show up.
 *   7. The defense payloads, over a fixture carrying every payload
 *      kind this surface binds, and a matched control that carries
 *      none. The tier's constructors and destructors are its only
 *      allocating functions and both run outside the window; the one
 *      exception the survey found, the information state's first push
 *      for a target, allocates that target's history ring, so the
 *      binding pushes at the episode epoch before any stepping and
 *      the ring exists before the window opens. The window covers
 *      step 1 and five boundary resets per environment, which is the
 *      point: a window opened at step 2 would measure nothing, and
 *      the control says what the payloads themselves cost.
 *   6. The collision pass and the relative-state observe, over their
 *      own fixture: two collidable bodies that meet inside every
 *      episode, with the second body's state published in the first's
 *      local-vertical local-horizontal frame, driven at 1, 4 and 16
 *      episodes with the counters armed across the steps only. All six
 *      counters must read zero at every length.
 *
 * Batch path: the batch executable shares the stepping machinery
 * with the serve surface by construction, and test_rl_determinism
 * gate 6 pins whole-file batch/serve equivalence; the batch binary
 * offers no hook for arming the counters around its stepping region
 * alone, so this gate scopes the interposition check to the serve
 * surface.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>

#include "rl_gate_util.h"
#include "k26rl_tap.h"
#include <time.h>

#define WORK_DIR "/tmp/kflc_rl_hotpath_test"

/* Same two-body geometry as the determinism gate: the craft sits
 * inside the MERCURIUS transition window around earth, so every
 * step runs the paper-faithful split (WH far pass plus IAS15 near
 * pass), the deepest stepping path. Horizon 24 with no termination
 * predicate: episodes truncate on a fixed cadence inside the armed
 * window. The block writes body state, one velocity key and one
 * position key, so the state accessors and the position fold are
 * measured too.
 *
 * The craft binds a vehicle assembly and declares a subdivision and
 * every attitude key, so the attitude advance, the per-environment
 * vehicles, the gravity-gradient torque and the substep loop are all
 * inside the measured window. Without an assembly the vehicle count
 * is zero and the whole of that path compiles out of the artifact,
 * which would leave the fixed requirement unmeasured for everything
 * this phase added rather than proven for it.
 *
 * The assembly carries all three actuator kinds, and the block below
 * commands each of them, for the same reason one step further on.
 * The counts are compile-time constants in the emitted source, so an
 * assembly declaring no wheel compiles the wheel step, the actuator
 * view and the momentum write-back out of the artifact entirely; an
 * assembly declaring no magnetorquer compiles out the whole field
 * chain including the field model call; and an assembly declaring no
 * thruster compiles out the perturbation callback, which is the one
 * piece of this phase that runs at every stage of the integrator
 * rather than once per sub-advance. A fixture that omits them
 * measures a smaller program than the one this phase produces. Two
 * of each are declared, since a loop over one unit and a loop over
 * many are not the same loop.
 *
 * The collision pass needs two collidable bodies, which is one more
 * vehicle than the fixture above carries, so it has a fixture of its
 * own below rather than a third body bolted onto this one: this
 * fixture randomises its craft's position at every reset, and a second
 * body placed a few metres from a position drawn over two hundred
 * kilometres could not be made to meet it. */
static const char *const HP_ASM =
    "assembly hotpath_box\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    /* Off the coordinate axes, so the frame rotations in the wheel
     * and magnetorquer models are exercised rather than reduced to a
     * single component. */
    "    wheel w_a\n"
    "        axis 0.6 0.8 0.0\n"
    "        spin_inertia 0.05\n"
    "        max_momentum 15.0\n"
    "        max_torque 0.2\n"
    "        viscous 1.0e-4\n"
    "        coulomb 1.0e-4\n"
    "        dead_rate 0.1\n"
    "    end\n"
    "    wheel w_b\n"
    "        axis 0.0 0.6 0.8\n"
    "        spin_inertia 0.04\n"
    "        max_momentum 12.0\n"
    "        max_torque 0.15\n"
    "    end\n"
    "    magnetorquer m_a\n"
    "        axis 0.8 0.0 0.6\n"
    "        max_dipole 30.0\n"
    "    end\n"
    "    magnetorquer m_b\n"
    "        axis 0.0 1.0 0.0\n"
    "        max_dipole 25.0\n"
    "    end\n"
    "    thruster t_a\n"
    "        at 1.05 0.92 0.0\n"
    "        dir 0.0 -1.0 0.0\n"
    "        thrust 400.0\n"
    "    end\n"
    "    thruster t_b\n"
    "        at -1.05 -0.92 0.0\n"
    "        dir 0.0 1.0 0.0\n"
    "        thrust 400.0\n"
    "    end\n"
    "end\n";

static const char *const HP_KFL =
    "form RL_HOTPATH\n"
    "fn world hp_world\n"
    /* The magnetorquers make the parent's rotation model a
     * precondition, and its NAIF id is what names it. */
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " ephem_naif_id=399\n"
    "    astro_body craft assembly=\"hotpath.k26asm\" parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0,10.0) vel_z=0.0"
    " quat_w=1.0 quat_x=0.0 quat_y=0.0 quat_z=0.0"
    " omega_x=0.01 omega_y=0.02 omega_z=0.03\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 24\n"
    "        substeps 8\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 10.0)\n"
    "        reset craft.omega_z uniform(-0.05, 0.05)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    action gear discrete 3 default 1\n"
    "    on_step\n"
    "        let scale: double = 1.0 + push * 0.001\n"
    /* One velocity key and one position key, so the position
     * write's sector-fold renormalisation sits inside the measured
     * window rather than beside it. */
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "        craft.pos_z = craft.pos_z + push\n"
    "        craft.omega_x = craft.omega_x + push * 0.0001\n"
    /* Every actuator commanded, and one wheel read back, so the
     * command store, the wheel step, the field chain, the
     * perturbation callback and the reading accessors are all inside
     * the measured window. */
    "        craft.w_a.torque = push * 0.2\n"
    "        craft.w_b.torque = push * -0.15\n"
    "        craft.m_a.dipole = push * 30.0\n"
    "        craft.m_b.dipole = push * -25.0\n"
    "        craft.t_a.throttle = 0.5 + push * 0.0\n"
    "        craft.t_b.throttle = 0.25 + push * 0.0\n"
    "        craft.pos_y = craft.pos_y + craft.w_a.momentum * 0.0\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    observe attitude of craft as att\n"
    "    objective\n"
    "        reward trk_range + push + att_omega_z\n"
    "    end\n"
    "end\n"
    "end\n";

/* The collision fixture, and the one that carries every recent
 * addition to the stepping path at once: an assembly, an attitude
 * advance with wheels, a collision pass that reaches a contact, a
 * docking port whose capture test runs at that contact, a relative
 * observe and a sensor chain. Measuring them together is the point:
 * each is cheap to measure beside the window and worth nothing there.
 *
 * Two bodies bind the same assembly, so the pass has a pair to test
 * and the broadphase, the sweep and the resolution are all compiled
 * in; a program with one collidable body compiles the pair loop to
 * nothing and would measure an empty pass. The same sharing gives
 * each body a port, which is what the port observe needs: one port on
 * the body it names and exactly one on another.
 *
 * The geometry is arranged so a contact happens inside every episode
 * rather than once at the start of the drive. The two start four
 * metres apart along the docking axis and close at 2.5 m/s. Each
 * carries its mating plane 1.2 m out along that axis, so the planes
 * meet when the centres are 2.4 m apart, after 0.64 s, inside
 * the seventh control period of a 24-step episode. The hulls reach each other only at 2.0 m, so the
 * contact reported is the one at the ports and the capture test runs
 * on it; it refuses, the closing rate being twenty-five times the
 * envelope's limit, which is the arithmetic and not the point. The
 * resolution then removes the closing rate and the pair stays in
 * contact for the rest of the episode, and truncation at the horizon
 * returns them to their declared separation for the next one. Nothing
 * here is randomised, which is what lets a fixed separation survive a
 * reset.
 *
 * The approach is slow relative to the subdivision on purpose: a
 * sub-advance is 12.5 ms, in which the pair closes 31 mm against a
 * contact separation of 2.4 m, so the fixture sits well inside the
 * separation precondition the pass requires rather than at its edge. */
static const char *const HP_COLL_ASM =
    "assembly hotpath_coll\n"
    "    frame x_to_port\n"
    "    provenance mass \"calibration shape, not a craft\" computed\n"
    "    component hull\n"
    "        mass 1000.0\n"
    "        at 0 0 0\n"
    "        collider box 1.0 0.5 0.5\n"
    "    end\n"
    "    port dock\n"
    "        at 1.2 0.0 0.0\n"
    "        axis 1.0 0.0 0.0\n"
    "        roll_ref 0.0 1.0 0.0\n"
    "        capture idss_e\n"
    "    end\n"
    "    wheel spin_a\n"
    "        axis 0.0 1.0 0.0\n"
    "        spin_inertia 0.031\n"
    "        max_momentum 15.0\n"
    "        max_torque 0.10\n"
    "        viscous 1.0e-5\n"
    "        coulomb 2.0e-4\n"
    "        dead_rate 1.0e-3\n"
    "    end\n"
    "    wheel spin_b\n"
    "        axis 0.0 0.0 1.0\n"
    "        spin_inertia 0.027\n"
    "        max_momentum 12.0\n"
    "        max_torque 0.08\n"
    "        viscous 2.0e-5\n"
    "        coulomb 3.0e-4\n"
    "        dead_rate 2.0e-3\n"
    "    end\n"
    "end\n";

/* Every payload kind this surface binds, over two craft that carry
 * vehicles. The information state is what this gate is about: its
 * push is the tier's one function that allocates outside a
 * constructor, and it runs once per sub-advance of every step inside
 * the window. Four sub-advances at a horizon of 12 puts five boundary
 * resets inside a sixty-step window at two environments, so the
 * reset path is measured rather than merely the steady step. */
static const char *const HP_PAY_KFL =
    "form RL_HOTPATH_PAY\n"
    "fn world hpp_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body watcher assembly=\"hpcoll.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0 omega_z=0.01\n"
    "    astro_body mover assembly=\"hpcoll.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=2.0e4 vel_y=7546.0 quat_w=1.0 omega_z=0.05\n"
    "    astro_payload eye body=watcher kind=detect_ir aperture_m=1.0"
    " integration_s=0.5 passband_lo_um=3.0 passband_hi_um=12.0"
    " throughput=0.5 snr_threshold=5.0 target_temp_k=300.0"
    " target_emissivity=0.9\n"
    "    astro_payload rf body=watcher kind=detect_radar p_tx_w=2000.0"
    " g_tx_db=40.0 g_rx_db=40.0 freq_hz=1.0e10 loss_sys_db=3.0"
    " bandwidth_hz=1.0e6 t_sys_k=290.0 noise_figure=2.0"
    " snr_threshold=10.0\n"
    "    astro_payload beam body=watcher kind=detect_lidar"
    " pulse_energy_j=0.1 wavelength_nm=1064.0 aperture_rx_m=0.5"
    " atmospheric_tx=1.0 detector_efficiency=0.3 snr_threshold=5.0"
    " target_albedo=0.2\n"
    "    astro_payload picture body=watcher kind=infostate history=64\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 12\n"
    "        substeps 4\n"
    "    end\n"
    "    agent hunter\n"
    "        action nudge box -1.0 1.0 default 0.0\n"
    "        observe detect eye of mover as ir\n"
    "        observe detect rf of mover as radar\n"
    "        observe detect beam of mover as lidar\n"
    "        observe track picture of mover modality=radar as trk\n"
    "    end\n"
    "    agent quarry\n"
    "        action dodge box -1.0 1.0 default 0.0\n"
    "        observe mover from watcher mode=geometric as los\n"
    "    end\n"
    "    on_step\n"
    "        watcher.vel_x = watcher.vel_x + nudge\n"
    "        mover.vel_x = mover.vel_x + dodge\n"
    "    end\n"
    "end\n"
    "end\n";

/* The matched control: the same world, the same craft, the same
 * period and subdivision and horizon, with the payloads and the
 * channels they publish removed and nothing else changed. Without it
 * a count of zero says only that this program allocates nothing, not
 * what the payloads themselves cost. */
static const char *const HP_NOPAY_KFL =
    "form RL_HOTPATH_NOPAY\n"
    "fn world hpn_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body watcher assembly=\"hpcoll.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0 quat_w=1.0 omega_z=0.01\n"
    "    astro_body mover assembly=\"hpcoll.k26asm\" parent=earth"
    " pos_x=7.0e6 pos_y=2.0e4 vel_y=7546.0 quat_w=1.0 omega_z=0.05\n"
    "    episode\n"
    "        control_dt 0.5\n"
    "        horizon 12\n"
    "        substeps 4\n"
    "    end\n"
    "    agent hunter\n"
    "        action nudge box -1.0 1.0 default 0.0\n"
    "        observe mover from watcher mode=geometric as ir\n"
    "    end\n"
    "    agent quarry\n"
    "        action dodge box -1.0 1.0 default 0.0\n"
    "        observe mover from watcher mode=geometric as los\n"
    "    end\n"
    "    on_step\n"
    "        watcher.vel_x = watcher.vel_x + nudge\n"
    "        mover.vel_x = mover.vel_x + dodge\n"
    "    end\n"
    "end\n"
    "end\n";

/* Two actions, and a horizon of 12 at a period of half a second. */
#define HP_PAY_ACT      2
#define HP_PAY_HORIZON 12
#define HP_PAY_STEPS   60

static const char *const HP_COLL_KFL =
    "form RL_HOTPATH_COLL\n"
    "fn world hpc_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24"
    " ephem_naif_id=399\n"
    "    astro_body alpha assembly=\"hpcoll.k26asm\" parent=earth"
    " pos_x=7.0e6 vel_y=7546.0"
    " quat_w=1.0 omega_x=0.01 omega_y=0.02 omega_z=0.03\n"
    /* Beta faces alpha: the assembly puts its port on the body's
     * first axis, so a half turn about the third points it back down
     * the approach. */
    "    astro_body beta assembly=\"hpcoll.k26asm\" parent=earth"
    " pos_x=7.0000040e6 vel_y=7546.0 vel_x=-2.5"
    " quat_w=0.0 quat_z=1.0 omega_x=-0.02 omega_y=0.01 omega_z=0.02\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 24\n"
    "        substeps 8\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.0\n"
    "    on_step\n"
    "        alpha.omega_x = alpha.omega_x + push * 0.0\n"
    "        alpha.spin_a.torque = push * 0.05\n"
    "        alpha.spin_b.torque = push * 0.03\n"
    "        beta.spin_a.torque = push * 0.02\n"
    "        beta.spin_b.torque = push * 0.04\n"
    "    end\n"
    /* A sensor chain on the tracking observe, so noise applied per
     * channel per step sits inside the armed window rather than beside
     * it. Every drawing kind is declared, because the counts are
     * compile-time constants: a chain without a bias walk compiles the
     * state advance out, and one without a dropout compiles the
     * uniform draw out. `with truth` doubles the observe, so the
     * paired copy is measured too. */
    "    sensor hp_sensor\n"
    "        noise normal 0.0 5.0\n"
    "        scale 0.001\n"
    "        bias_walk 2.0 30.0 0.5\n"
    "        latency 2\n"
    "        quantise 0.5\n"
    "        dropout 0.02\n"
    "    end\n"
    "    observe contact of beta as hit\n"
    "    observe alpha from earth mode=geometric"
    " through hp_sensor with truth as trk\n"
    /* The relative state of the pair, in the chief's own frame. It is
     * declared here rather than in a fixture of its own because it
     * runs inside the same per-step observation function the two above
     * do, and because a pair of collidable craft in proximity is
     * exactly the program that asks for it. Its cost is measured with
     * theirs. */
    "    observe relative beta from alpha as rel\n"
    /* The docking port form, whose residuals are recomputed every
     * step and whose capture test runs inside the contact the pass
     * finds. Both halves are on the stepping path and both are
     * therefore inside the armed window. */
    "    observe port dock of beta as prt\n"
    "    objective\n"
    "        reward hit_hit + trk_range + rel_r_y + prt_axial\n"
    "    end\n"
    "end\n"
    "end\n";

/* Three contact channels, five tracking channels and their five
 * paired truth channels, six relative ones, nine port ones; one
 * action. */
#define HP_COLL_OBS     28
#define HP_COLL_HIT      0
#define HP_COLL_ACT      1
#define HP_COLL_HORIZON 24

/* The interposer, compiled into WORK_DIR/interpose.so at gate time.
 * Allocation calls forward to the glibc-internal entry points;
 * write forwards to the kernel directly; fwrite forwards through a
 * constructor-resolved RTLD_NEXT pointer so no lookup happens while
 * armed. Counter slots: 0 malloc, 1 calloc, 2 realloc, 3 free,
 * 4 write, 5 fwrite. */
static const char *const HP_INTERPOSE_C =
    "#define _GNU_SOURCE\n"
    "#include <dlfcn.h>\n"
    "#include <stddef.h>\n"
    "#include <stdio.h>\n"
    "#include <sys/syscall.h>\n"
    "#include <unistd.h>\n"
    "extern void *__libc_malloc(size_t);\n"
    "extern void *__libc_calloc(size_t, size_t);\n"
    "extern void *__libc_realloc(void *, size_t);\n"
    "extern void __libc_free(void *);\n"
    "volatile unsigned long k26hp_counts[6];\n"
    "volatile int k26hp_armed;\n"
    "static size_t (*real_fwrite_)(const void *, size_t, size_t, FILE *);\n"
    "__attribute__((constructor)) static void k26hp_init_(void)\n"
    "{\n"
    "    real_fwrite_ = (size_t (*)(const void *, size_t, size_t, FILE *))\n"
    "        dlsym(RTLD_NEXT, \"fwrite\");\n"
    "}\n"
    "void *malloc(size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[0]++; return __libc_malloc(n); }\n"
    "void *calloc(size_t m, size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[1]++; return __libc_calloc(m, n); }\n"
    "void *realloc(void *p, size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[2]++; return __libc_realloc(p, n); }\n"
    "void free(void *p)\n"
    "{ if (k26hp_armed) k26hp_counts[3]++; __libc_free(p); }\n"
    "ssize_t write(int fd, const void *buf, size_t n)\n"
    "{ if (k26hp_armed) k26hp_counts[4]++;\n"
    "  return syscall(SYS_write, fd, buf, n); }\n"
    "size_t fwrite(const void *p, size_t sz, size_t n, FILE *f)\n"
    "{ if (k26hp_armed) k26hp_counts[5]++;\n"
    "  return real_fwrite_(p, sz, n, f); }\n";

enum { HP_ENVS = 2, HP_STEPS = 60, HP_ACT = 2 };

/* Write-family bound for the output-enabled window; derivation in
 * the header comment. */
#define HP_WRITE_BOUND 12ul

/* ---- Child: runs under LD_PRELOAD of the interposer ------------- */

static volatile unsigned long *counts_;
static volatile int *armed_;

static void counters_clear_(void)
{
    for (int i = 0; i < 6; i++) counts_[i] = 0;
}

static unsigned long alloc_total_(void)
{
    return counts_[0] + counts_[1] + counts_[2] + counts_[3];
}

static unsigned long write_total_(void)
{
    return counts_[4] + counts_[5];
}

static unsigned long untapped_writes;

static int child_main_(void)
{
    counts_ = (volatile unsigned long *)dlsym(RTLD_DEFAULT, "k26hp_counts");
    armed_  = (volatile int *)dlsym(RTLD_DEFAULT, "k26hp_armed");
    ASSERT(counts_ != NULL && armed_ != NULL);

    /* Gate 1: positive control. Prime stdio buffers and the /dev/null
     * descriptors before arming so only the deliberate calls count. */
    int devnull = open("/dev/null", O_WRONLY);
    ASSERT(devnull >= 0);
    FILE *fnull = fopen("/dev/null", "w");
    ASSERT(fnull != NULL);
    ASSERT(fwrite("p", 1, 1, fnull) == 1);
    printf("test_rl_hotpath: child up\n");
    fflush(stdout);

    counters_clear_();
    *armed_ = 1;
    void *volatile ctrl = malloc(32);
    free((void *)ctrl);
    ASSERT(write(devnull, "x", 1) == 1);
    ASSERT(fwrite("y", 1, 1, fnull) == 1);
    *armed_ = 0;
    ASSERT(counts_[0] == 1);   /* malloc */
    ASSERT(counts_[3] == 1);   /* free */
    ASSERT(counts_[4] == 1);   /* write */
    ASSERT(counts_[5] == 1);   /* fwrite */
    printf("gate 1: positive control malloc/free/write/fwrite"
           " = 1/1/1/1: OK\n");

    void *so = rl_dlopen_(WORK_DIR "/hp.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    static double act[HP_ENVS * HP_ACT];
    for (int e = 0; e < HP_ENVS; e++) {
        act[e * HP_ACT + 0] = 0.25;
        act[e * HP_ACT + 1] = 1.0;
    }

    /* Gate 2: output disabled. The armed window covers the steps
     * alone, first step after reset included (nothing may be
     * lazy-initialised on the step path). */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 2: %d steps x %d envs, output disabled:"
               " alloc-family %lu (malloc %lu calloc %lu realloc %lu"
               " free %lu), write-family %lu\n",
               HP_STEPS, HP_ENVS, a, counts_[0], counts_[1],
               counts_[2], counts_[3], w);
        ASSERT(a == 0);
        ASSERT(w == 0);
        /* Kept as the reference the tap arms below must equal. */
        untapped_writes = w;
        s.destroy(env);
    }
    printf("gate 2: zero allocations, zero writes: OK\n");

    /* Gate 2b: the same drive at three episode lengths. Zero at one
     * length already implies zero per episode, but a count that grew
     * with the number of episodes is the shape a per-episode leak
     * takes, and reading it at 1, 4 and 16 episodes says so directly
     * rather than by inference. The horizon is 24, so each length is
     * a whole number of episodes and crosses every boundary it
     * implies. */
    {
        static const int eps[] = { 1, 4, 16 };
        for (int k = 0; k < 3; k++) {
            K26RlEnv *env = NULL;
            ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
            ASSERT(s.reset(env) == K26RL_OK);
            int steps = eps[k] * 24;
            counters_clear_();
            *armed_ = 1;
            for (int t = 0; t < steps; t++) {
                ASSERT(s.step(env, act) == K26RL_OK);
            }
            *armed_ = 0;
            unsigned long a = alloc_total_(), w = write_total_();
            printf("gate 2b: %2d episode(s), %3d steps x %d envs:"
                   " alloc-family %lu, write-family %lu\n",
                   eps[k], steps, HP_ENVS, a, w);
            ASSERT(a == 0);
            ASSERT(w == 0);
            s.destroy(env);
        }
        printf("gate 2b: the counts do not grow with the number of "
               "episodes: OK\n");
    }

    /* Gate 3: output enabled. Steps still allocate nothing; the
     * writer's flushes stay within the frame bound derived in the
     * header comment. */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.output(env, WORK_DIR "/hp_out.k26epi") == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 3: %d steps x %d envs, output enabled:"
               " alloc-family %lu, write-family %lu (bound %lu)\n",
               HP_STEPS, HP_ENVS, a, w, HP_WRITE_BOUND);
        ASSERT(a == 0);
        ASSERT(w > 0);
        ASSERT(w <= HP_WRITE_BOUND);
        s.destroy(env);
    }
    printf("gate 3: zero allocations, bounded writes: OK\n");

    /* Gates 4 and 5: the telemetry tap costs the step path no
     * allocation and no system call, whether or not anyone is
     * listening. The acceptance is not a bound of its own: it is
     * equality with the untapped drive gate 2 just measured in this
     * same process, over the same fixture and the same window. An
     * implementation that published over a pipe, or remapped per
     * frame, would move the write-family count and fail here. */
    char tap_name[64];
    snprintf(tap_name, sizeof tap_name, "hotpath.%ld", (long)getpid());
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.tap(env, tap_name) == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 4: %d steps x %d envs, tap enabled, no consumer:"
               " alloc-family %lu, write-family %lu"
               " (untapped drive was %lu)\n",
               HP_STEPS, HP_ENVS, a, w, untapped_writes);
        ASSERT(a == 0);
        ASSERT(w == untapped_writes);
        s.destroy(env);
    }
    printf("gate 4: tap enabled costs no allocation and no syscall: OK\n");

    {
        K26RlEnv *env = NULL;
        pid_t consumer;
        int ready[2];
        char ready_byte = 0;

        ASSERT(s.create(42, HP_ENVS, &env) == K26RL_OK);
        ASSERT(s.tap(env, tap_name) == K26RL_OK);
        ASSERT(s.reset(env) == K26RL_OK);

        /* A real consumer, in its own process, reading continuously
         * for as long as the producer runs. Its own allocations are
         * its own process's and cannot reach these counters, which is
         * the point: a consumer is invisible to the producer.
         *
         * The parent waits for the consumer to report itself attached
         * before it arms and steps. Without that handshake the child
         * races the whole measured window and can lose it on a busy
         * machine, and a gate that claims to measure the cost with a
         * consumer attached must know that one is. */
        ASSERT(pipe(ready) == 0);
        consumer = fork();
        ASSERT(consumer >= 0);
        if (consumer == 0) {
            K26RlTapReader *r = NULL;
            uint32_t slot = 0;
            close(ready[0]);
            /* Attach with a wall-clock bound rather than a spin count,
             * so a slow start waits instead of giving up. */
            for (int tries = 0; tries < 10000 && !r; tries++) {
                struct timespec ts;
                if (k26rl_tap_attach(tap_name, 1, &r) == K26RL_OK)
                    break;
                ts.tv_sec = 0;
                ts.tv_nsec = 1000000;   /* 1 ms; 10 s in total */
                nanosleep(&ts, NULL);
            }
            if (!r)
                _exit(2);
            if (k26rl_tap_reader_info(r, &slot, NULL) != K26RL_OK)
                _exit(2);
            uint8_t *buf = (uint8_t *)malloc(slot);
            if (!buf)
                _exit(2);
            if (write(ready[1], "a", 1) != 1)
                _exit(2);
            close(ready[1]);
            for (;;) {
                uint32_t len = 0;
                if (k26rl_tap_read(r, buf, slot, &len, NULL) != K26RL_OK)
                    _exit(2);
                if (!len && k26rl_tap_reader_closed(r))
                    break;
            }
            _exit(0);
        }
        close(ready[1]);
        /* One byte, and only after the consumer holds a mapping. */
        ASSERT(read(ready[0], &ready_byte, 1) == 1);
        ASSERT(ready_byte == 'a');
        close(ready[0]);

        counters_clear_();
        *armed_ = 1;
        for (int t = 0; t < HP_STEPS; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
        }
        *armed_ = 0;

        unsigned long a = alloc_total_(), w = write_total_();
        printf("gate 5: %d steps x %d envs, tap enabled, consumer"
               " attached: alloc-family %lu, write-family %lu"
               " (untapped drive was %lu)\n",
               HP_STEPS, HP_ENVS, a, w, untapped_writes);
        ASSERT(a == 0);
        ASSERT(w == untapped_writes);
        s.destroy(env);

        int cst = 0;
        ASSERT(waitpid(consumer, &cst, 0) == consumer);
        ASSERT(WIFEXITED(cst) && WEXITSTATUS(cst) == 0);
    }
    printf("gate 5: an attached consumer changes neither count: OK\n");

    /* Gate 6: the collision pass inside the measured window.
     *
     * The pass runs between the sub-advances, so it is inside the
     * stepping hot loop and the fixed requirement binds it exactly as
     * it binds the attitude advance beside it. */
    {
        void *cso = rl_dlopen_(WORK_DIR "/hpcoll.rlenv.so");
        RlSurface cs;
        rl_resolve_surface_(cso, &cs);
        ASSERT(cs.abi_version() == K26RL_ABI_VERSION);

        static double cact[HP_ENVS * HP_COLL_ACT];
        for (int e = 0; e < HP_ENVS * HP_COLL_ACT; e++) cact[e] = 0.0;

        /* Contacts are proved before anything is measured over them.
         * A window in which the pair never meets would measure the
         * broadphase rejecting a distant pair and nothing else, and
         * would read zero for an implementation whose resolution
         * allocated on every contact. The proof drive is unarmed, from
         * its own handle, over the same seed and the same actions as
         * the armed drives below. */
        {
            K26RlEnv *env = NULL;
            static double cobs[HP_ENVS * HP_COLL_OBS];
            int contacts = 0, first = -1;

            ASSERT(cs.create(42, HP_ENVS, &env) == K26RL_OK);
            ASSERT(cs.reset(env) == K26RL_OK);
            for (int t = 0; t < HP_STEPS; t++) {
                ASSERT(cs.step(env, cact) == K26RL_OK);
                ASSERT(cs.obs(env, cobs) == K26RL_OK);
                if (cobs[HP_COLL_HIT] > 0.5) {
                    contacts++;
                    if (first < 0) first = t + 1;
                }
            }
            printf("gate 6: fixture check: %d of %d steps carried a"
                   " contact, first at step %d\n",
                   contacts, HP_STEPS, first);
            ASSERT(contacts > 0);
            ASSERT(first > 0);
            cs.destroy(env);
        }

        static const int ceps[] = { 1, 4, 16 };
        for (int k = 0; k < 3; k++) {
            K26RlEnv *env = NULL;
            int steps = ceps[k] * HP_COLL_HORIZON;

            ASSERT(cs.create(42, HP_ENVS, &env) == K26RL_OK);
            ASSERT(cs.reset(env) == K26RL_OK);
            counters_clear_();
            *armed_ = 1;
            for (int t = 0; t < steps; t++) {
                ASSERT(cs.step(env, cact) == K26RL_OK);
            }
            *armed_ = 0;

            unsigned long a = alloc_total_(), w = write_total_();
            printf("gate 6: %2d episode(s), %3d steps x %d envs,"
                   " an assembly, wheels, a contacting pair, a capture"
                   " test, a relative observe and a sensor chain:"
                   " alloc-family %lu"
                   " (malloc %lu calloc %lu realloc %lu free %lu),"
                   " write-family %lu\n",
                   ceps[k], steps, HP_ENVS, a, counts_[0], counts_[1],
                   counts_[2], counts_[3], w);
            ASSERT(a == 0);
            ASSERT(w == 0);
            cs.destroy(env);
        }
        dlclose(cso);
    }
    printf("gate 6: every addition to the stepping path measured here"
           " allocates nothing and writes nothing: OK\n");

    /* Gate 7: the defense payloads, and the matched control.
     *
     * The window opens after create and one reset and then runs sixty
     * steps of two environments at a horizon of twelve, so it holds
     * step 1 and five boundary resets per environment. Both halves
     * are driven identically; the counts are printed as figures
     * rather than only asserted, because what the payloads cost is
     * the reading and not merely whether it is zero. */
    {
        unsigned long pay_a = 0, pay_w = 0, non_a = 0, non_w = 0;
        static const char *const SOS[2] = {
            WORK_DIR "/hppay.rlenv.so", WORK_DIR "/hpnopay.rlenv.so"
        };
        for (int half = 0; half < 2; half++) {
            void *pso = rl_dlopen_(SOS[half]);
            RlSurface ps;
            rl_resolve_surface_(pso, &ps);
            ASSERT(ps.abi_version() == K26RL_ABI_VERSION);

            K26RlEnv *env = NULL;
            static double pact[HP_ENVS * HP_PAY_ACT];
            for (int i = 0; i < HP_ENVS * HP_PAY_ACT; i++) pact[i] = 0.0;
            ASSERT(ps.create(9u, HP_ENVS, &env) == K26RL_OK);
            /* No reset before the window opens. The frozen surface
             * says a handle's observations are valid immediately and
             * the first step steps normally, so stepping straight from
             * create is a route a caller may take; a window that
             * always began with a reset would let a binding that
             * allocated on the first step of a fresh handle pass,
             * because the reset would have done the allocating
             * outside it. */
            counters_clear_();
            *armed_ = 1;
            for (int t = 0; t < HP_PAY_STEPS; t++) {
                ASSERT(ps.step(env, pact) == K26RL_OK);
                /* An explicit reset inside the window, beside the
                 * boundary ones, so the caller-driven reset path is
                 * measured as well as the automatic one. */
                if (t == HP_PAY_STEPS / 2) {
                    ASSERT(ps.reset(env) == K26RL_OK);
                }
            }
            *armed_ = 0;
            unsigned long a = alloc_total_(), w = write_total_();
            if (half == 0) { pay_a = a; pay_w = w; }
            else           { non_a = a; non_w = w; }
            printf("gate 7: %s: %d steps x %d envs, window holds step 1,"
                   " %d boundary reset(s) per environment and 1 driven"
                   " reset, and opens on a fresh handle: alloc-family %lu"
                   " (malloc %lu calloc %lu realloc %lu free %lu),"
                   " write-family %lu\n",
                   half == 0 ? "every payload kind" : "matched control",
                   HP_PAY_STEPS, HP_ENVS,
                   HP_PAY_STEPS / HP_PAY_HORIZON, a,
                   counts_[0], counts_[1], counts_[2], counts_[3], w);
            ASSERT(a == 0);
            ASSERT(w == 0);
            ps.destroy(env);
            dlclose(pso);
        }
        ASSERT(pay_a == 0 && non_a == 0);
        ASSERT(pay_w == 0 && non_w == 0);
        printf("gate 7: the payload-bearing artifact and its matched"
               " control both allocate %lu and write %lu across the"
               " window: OK\n", pay_a, pay_w);
    }

    dlclose(so);
    fclose(fnull);
    close(devnull);
    printf("test_rl_hotpath: 7 gates passed\n");
    return 0;
}

/* ---- Parent: compile fixture + interposer, re-exec under preload */

int main(void)
{
    if (getenv("K26RL_HOTPATH_CHILD")) return child_main_();

    if (!rl_libs_present_("test_rl_hotpath")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/hotpath.k26asm", HP_ASM);
    rl_write_file_(WORK_DIR "/hp.kfl", HP_KFL);
    rl_compile_(WORK_DIR "/hp.kfl", WORK_DIR "/hp", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/hp.rlenv.so"));

    rl_write_file_(WORK_DIR "/hpcoll.k26asm", HP_COLL_ASM);
    rl_write_file_(WORK_DIR "/hpcoll.kfl", HP_COLL_KFL);
    rl_compile_(WORK_DIR "/hpcoll.kfl", WORK_DIR "/hpcoll", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/hpcoll.rlenv.so"));

    rl_write_file_(WORK_DIR "/hppay.kfl", HP_PAY_KFL);
    rl_compile_(WORK_DIR "/hppay.kfl", WORK_DIR "/hppay", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/hppay.rlenv.so"));
    rl_write_file_(WORK_DIR "/hpnopay.kfl", HP_NOPAY_KFL);
    rl_compile_(WORK_DIR "/hpnopay.kfl", WORK_DIR "/hpnopay", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/hpnopay.rlenv.so"));

    rl_write_file_(WORK_DIR "/interpose.c", HP_INTERPOSE_C);
    rl_run_or_die_("cc -O2 -fPIC -shared -o " WORK_DIR "/interpose.so "
                   WORK_DIR "/interpose.c -ldl");

    char self[512];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    ASSERT(n > 0);
    self[n] = '\0';

    char cmd[1024];
    int m = snprintf(cmd, sizeof cmd,
                     "env LD_PRELOAD=" WORK_DIR "/interpose.so"
                     " K26RL_HOTPATH_CHILD=1 %s", self);
    ASSERT((size_t)m < sizeof cmd);
    int rc = system(cmd);
    ASSERT(rc != -1);
    ASSERT(WIFEXITED(rc));
    return WEXITSTATUS(rc);
}
