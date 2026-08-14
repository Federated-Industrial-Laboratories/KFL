/* test_rl_bodies.c: the body-state getter and the two spec tags.
 *
 * The getter exists so a consumer can see the world the observation
 * channels are views of. The gate that matters most is therefore not
 * that the getter returns plausible numbers, but that it returns the
 * same world: the vector it reports between a target and its observer
 * must be the very vector the observation path measured, to the bit.
 * Gate 3 pins that, and it is only reachable because the getter
 * returns the runtime's exact position subtraction rather than a
 * difference of two flattened coordinates.
 *
 * Gates:
 *   1. Sizing and refusals, following the spec getter's convention: a
 *      capacity-0 call returns the requirement and writes nothing, a
 *      short buffer likewise, an exact buffer fills, a reference
 *      naming no body is refused, and the world-origin reference is
 *      accepted.
 *   2. Determinism and vector independence: two runs at one seed and
 *      action stream are bitwise equal, and an environment's bodies
 *      are unmoved by what its neighbours are driven with.
 *   3. The consistency pin, bitwise: with the reference set to a
 *      geometric observe's observer, the target body's reported
 *      vector reproduces that observe's recorded range and direction
 *      channels exactly, reciprocal multiplication included.
 *   4. The tags: every channel carries its observe's mode, the two
 *      observes' modes differ and land on their own channels, and
 *      every body carries its declared name in declaration order.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <math.h>

#include "rl_gate_util.h"

#define WORK_DIR "/tmp/kflc_rl_bodies_test"

/* Two bodies and two observes at different modes. The first observe,
 * `trk`, is geometric, which is the one gate 3 pins: a geometric
 * observe applies no light-time correction, so the vector the
 * observation path measures is exactly the position subtraction the
 * getter reports. */
static const char *const BODIES_KFL =
    "form RL_BODIES\n"
    "fn world bodies_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth"
    " pos_x=uniform(6.9e6,7.1e6) pos_y=0.0 pos_z=0.0"
    " vel_x=0.0 vel_y=normal(7350.0,10.0) vel_z=0.0\n"
    "    let range_scale: double = 7.0e6\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 8\n"
    "        reset craft.pos_x uniform(6.9e6, 7.1e6)\n"
    "        reset craft.vel_y normal(7350.0, 10.0)\n"
    "    end\n"
    "    action push box -1.0 1.0 default 0.25\n"
    "    on_step\n"
    "        craft.vel_x = craft.vel_x + push * 0.01\n"
    "    end\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    observe earth from craft mode=astrometric as rev\n"
    "    objective\n"
    "        reward trk_range / range_scale + push\n"
    "    end\n"
    "end\n"
    "end\n";

/* The same shape, but the action drives the craft's position far out:
 * at 1.5e11 m it sits past two sector edges of the runtime's position
 * grid, whose edge is 2^36 m. Gate 3's near-field fixture keeps every
 * body inside sector 0, where the exact and the flattened forms of a
 * position subtraction are the same arithmetic and a flattened getter
 * would pass; this one separates them. */
static const char *const FAR_KFL =
    "form RL_BODIES_FAR\n"
    "fn world far_world\n"
    "    astro_body star gm=1.32712440018e20 mass=1.989e30"
    " pos_x=8.0e10 pos_y=1.3e9 pos_z=7.0e8"
    " vel_x=11.0 vel_y=1300.0 vel_z=17.0\n"
    "    astro_body craft gm=1.0"
    " pos_x=1.5e11 pos_y=3.1e9 pos_z=1.1e9"
    " vel_x=13.0 vel_y=1100.0 vel_z=19.0\n"
    "    episode\n"
    "        control_dt 1.0\n"
    "        horizon 8\n"
    "    end\n"
    "    action place box 1.0e11 2.0e11 default 1.5e11\n"
    "    on_step\n"
    "        craft.pos_x = place\n"
    "    end\n"
    "    observe craft from star mode=geometric as trk\n"
    "    objective\n"
    "        reward 0.0\n"
    "    end\n"
    "end\n"
    "end\n";

/* A fixture that faults on demand: the reward divides by zero when a
 * caller drives the action to 1.0. It exists to pin what the getter
 * reports after a fault, which is not what the observation getters
 * report. */
static const char *const FAULT_KFL =
    "form RL_BODIES_FAULT\n"
    "fn world bfault_world\n"
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
    "    astro_body craft gm=1.0 parent=earth pos_x=7.0e6 vel_y=7350.0\n"
    "    episode\n"
    "        control_dt 0.1\n"
    "        horizon 6\n"
    "    end\n"
    "    action push box 0.0 4.0 default 0.0\n"
    "    observe craft from earth mode=geometric as trk\n"
    "    objective\n"
    "        reward 1.0 / (1.0 - push)\n"
    "    end\n"
    "end\n"
    "end\n";

/* The runtime's sector edge, 2^36 m. A body beyond it is in a
 * different sector from one near the origin. */
#define SECTOR_EDGE_M 68719476736.0

enum { N_ENVS = 2, N_BODIES = 2, N_ACT = 1, OBS_TOTAL = 10, PER_BODY = 6 };
enum { EARTH = 0, CRAFT = 1 };

static double act_(uint32_t t, uint32_t e)
{
    return ((double)((t * 7u + e * 3u) % 13u)) / 13.0 * 2.0 - 1.0;
}

int main(void)
{
    if (!rl_libs_present_("test_rl_bodies"))
        return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);

    rl_write_file_(WORK_DIR "/bodies.kfl", BODIES_KFL);
    rl_compile_(WORK_DIR "/bodies.kfl", WORK_DIR "/bodies", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/bodies.rlenv.so"));

    void *so = rl_dlopen_(WORK_DIR "/bodies.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);
    /* Major equality and minor at-least: the body getter arrived
     * at minor 2, and a later minor still carries it. */
    ASSERT((s.abi_version() >> 16) == 1 && (s.abi_version() & 0xFFFF) >= 2);

    const int32_t want = (int32_t)(N_ENVS * N_BODIES * PER_BODY);
    double buf[N_ENVS * N_BODIES * PER_BODY];

    /* ---- Gate 1: sizing and refusals ------------------------------ */
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(11, N_ENVS, &env) == K26RL_OK);

        /* Capacity 0 sizes it and writes nothing. */
        for (int i = 0; i < want; i++)
            buf[i] = -1.0;
        ASSERT(s.bodies(env, EARTH, NULL, 0) == want);
        ASSERT(s.bodies(env, EARTH, buf, 0) == want);
        for (int i = 0; i < want; i++)
            ASSERT(buf[i] == -1.0);

        /* One short of the requirement writes nothing and still
         * reports what is needed. */
        ASSERT(s.bodies(env, EARTH, buf, (uint32_t)want - 1) == want);
        for (int i = 0; i < want; i++)
            ASSERT(buf[i] == -1.0);

        /* Exactly the requirement fills it. */
        ASSERT(s.bodies(env, EARTH, buf, (uint32_t)want) == want);
        {
            int moved = 0;
            for (int i = 0; i < want; i++)
                if (buf[i] != -1.0) moved++;
            ASSERT(moved > 0);
        }
        /* The reference body reports itself at the origin of its own
         * frame, which is the cheapest check that the frame is the
         * one claimed. */
        ASSERT(buf[EARTH * PER_BODY + 0] == 0.0);
        ASSERT(buf[EARTH * PER_BODY + 1] == 0.0);
        ASSERT(buf[EARTH * PER_BODY + 2] == 0.0);

        /* A reference naming no body is refused; the world origin is
         * a reference and is accepted. */
        ASSERT(s.bodies(env, 2, buf, (uint32_t)want) ==
               -(int32_t)K26RL_E_GEOMETRY);
        ASSERT(s.bodies(env, K26RL_BODY_REF_ORIGIN, buf, (uint32_t)want) ==
               want);
        s.destroy(env);
        printf("gate 1: sizing, refusal, and the origin reference: OK\n");
    }

    /* ---- Gate 2: determinism and vector independence --------------- */
    {
        double run_a[N_ENVS * N_BODIES * PER_BODY];
        double run_b[N_ENVS * N_BODIES * PER_BODY];
        double alt[N_ENVS * N_BODIES * PER_BODY];

        for (int pass = 0; pass < 3; pass++) {
            K26RlEnv *env = NULL;
            double a[N_ENVS * N_ACT];
            double *dst = pass == 0 ? run_a : (pass == 1 ? run_b : alt);

            ASSERT(s.create(11, N_ENVS, &env) == K26RL_OK);
            for (uint32_t t = 0; t < 8; t++) {
                a[0] = act_(t, 0);
                /* Pass 2 drives the neighbour differently; environment
                 * 0's bodies must not notice. */
                a[1] = (pass == 2) ? -act_(t, 1) : act_(t, 1);
                ASSERT(s.step(env, a) == K26RL_OK);
            }
            ASSERT(s.bodies(env, EARTH, dst, (uint32_t)want) == want);
            s.destroy(env);
        }
        ASSERT(memcmp(run_a, run_b, sizeof run_a) == 0);
        /* Environment 0's slice alone, against the run whose
         * neighbour was driven differently. */
        ASSERT(memcmp(run_a, alt, sizeof(double) * N_BODIES * PER_BODY) == 0);
        /* And the neighbour did move, so the comparison above is not
         * vacuous. */
        ASSERT(memcmp(run_a + N_BODIES * PER_BODY, alt + N_BODIES * PER_BODY,
                      sizeof(double) * N_BODIES * PER_BODY) != 0);
        printf("gate 2: bitwise across runs, and an environment's bodies"
               " unmoved by its neighbour's actions: OK\n");
    }

    /* ---- Gate 3: the consistency pin ------------------------------- */
    {
        K26RlEnv *env = NULL;
        double a[N_ENVS * N_ACT];
        double obs[N_ENVS * OBS_TOTAL];
        uint32_t compared = 0;

        ASSERT(s.create(29, N_ENVS, &env) == K26RL_OK);
        for (uint32_t t = 0; t < 12; t++) {
            a[0] = act_(t, 0);
            a[1] = act_(t, 1);
            ASSERT(s.step(env, a) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            /* The reference is the geometric observe's observer, so
             * the target's reported vector is the very vector the
             * observation path measured. */
            ASSERT(s.bodies(env, EARTH, buf, (uint32_t)want) == want);

            for (uint32_t e = 0; e < N_ENVS; e++) {
                const double *r = buf + ((size_t)e * N_BODIES + CRAFT) * PER_BODY;
                const double *o = obs + (size_t)e * OBS_TOTAL;
                /* The runtime's own expressions, in its own order:
                 * the magnitude as a sum of squares under one square
                 * root, and the direction as the vector times the
                 * reciprocal of that magnitude, never divided. */
                double range = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                double inv = 1.0 / range;
                double dx = r[0] * inv, dy = r[1] * inv, dz = r[2] * inv;

                ASSERT(memcmp(&range, &o[3], sizeof(double)) == 0);
                ASSERT(memcmp(&dx, &o[0], sizeof(double)) == 0);
                ASSERT(memcmp(&dy, &o[1], sizeof(double)) == 0);
                ASSERT(memcmp(&dz, &o[2], sizeof(double)) == 0);
                compared++;
            }
        }
        ASSERT(compared == 12 * N_ENVS);
        s.destroy(env);
        printf("gate 3: %u steps where the getter's vector reproduces the"
               " recorded geometric range and direction bitwise: OK\n",
               compared);
    }

    /* ---- Gate 4: the two spec tags --------------------------------- */
    {
        K26RlEnv *env = NULL;
        RlSpecView v;
        int32_t n;
        uint8_t *blob;

        ASSERT(s.create(11, N_ENVS, &env) == K26RL_OK);
        n = s.spec(env, NULL, 0);
        ASSERT(n > 0);
        blob = malloc((size_t)n);
        ASSERT(blob != NULL);
        ASSERT(s.spec(env, blob, (uint32_t)n) == n);
        rl_parse_spec_(blob, (uint32_t)n, &v);

        ASSERT(v.obs_total == OBS_TOTAL);
        ASSERT(v.n_modes == OBS_TOTAL);
        /* The first observe is geometric, the second astrometric, and
         * each observe's five channels carry its own value. */
        for (int i = 0; i < 5; i++)
            ASSERT(v.modes[i] == K26RL_OBS_MODE_GEOMETRIC);
        for (int i = 5; i < 10; i++)
            ASSERT(v.modes[i] == K26RL_OBS_MODE_ASTROMETRIC);

        ASSERT(v.n_body_names == N_BODIES);
        ASSERT(strcmp(v.body_names[EARTH], "earth") == 0);
        ASSERT(strcmp(v.body_names[CRAFT], "craft") == 0);

        /* An older consumer skips what it does not know and still
         * recovers everything it did: the parse above walked the whole
         * blob to its end, which rl_parse_spec_ asserts, and the
         * totals it recovered are the ones the surface reports. */
        ASSERT(v.abi_version == K26RL_ABI_VERSION);
        ASSERT(v.n_envs == N_ENVS);

        free(blob);
        s.destroy(env);
        printf("gate 4: per-channel modes for both observes, and body"
               " names in declaration order: OK\n");
    }

    /* ---- Gate 5: the pin again, across a sector boundary ----------- */
    {
        void *fso;
        RlSurface fs;
        K26RlEnv *env = NULL;
        double a[1 * 1];
        double obs[1 * 5];
        double fbuf[1 * N_BODIES * PER_BODY];
        const int32_t fwant = (int32_t)(1 * N_BODIES * PER_BODY);
        uint32_t compared = 0;
        int crossed = 0;

        rl_write_file_(WORK_DIR "/far.kfl", FAR_KFL);
        rl_compile_(WORK_DIR "/far.kfl", WORK_DIR "/far", WORK_DIR);
        ASSERT(rl_file_exists_(WORK_DIR "/far.rlenv.so"));
        fso = rl_dlopen_(WORK_DIR "/far.rlenv.so");
        rl_resolve_surface_(fso, &fs);
        ASSERT(fs.create(3, 1, &env) == K26RL_OK);

        for (uint32_t t = 0; t < 4; t++) {
            const double *r;
            double range, inv, dx, dy, dz;

            a[0] = 1.5e11 + (double)t * 7.3e6;   /* stays past the edge */
            ASSERT(fs.step(env, a) == K26RL_OK);
            ASSERT(fs.obs(env, obs) == K26RL_OK);
            ASSERT(fs.bodies(env, EARTH, fbuf, (uint32_t)fwant) == fwant);

            /* The arm is only worth anything if the fixture really
             * does straddle a sector boundary: without this the
             * assertions below would pass on a flattened getter, which
             * is exactly how the near-field gate failed to
             * discriminate. */
            ASSERT(fs.bodies(env, K26RL_BODY_REF_ORIGIN, fbuf,
                             (uint32_t)fwant) == fwant);
            {
                double se = floor(fbuf[EARTH * PER_BODY + 0] / SECTOR_EDGE_M);
                double sc = floor(fbuf[CRAFT * PER_BODY + 0] / SECTOR_EDGE_M);
                /* Different sectors on the axis under test, and both
                 * offsets carrying integration detail, which is the
                 * condition under which a flattened subtraction loses
                 * bits the exact one keeps. */
                if (se != sc)
                    crossed++;
            }
            ASSERT(fs.bodies(env, EARTH, fbuf, (uint32_t)fwant) == fwant);

            r = fbuf + (size_t)CRAFT * PER_BODY;
            range = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            inv = 1.0 / range;
            dx = r[0] * inv; dy = r[1] * inv; dz = r[2] * inv;
            ASSERT(memcmp(&range, &obs[3], sizeof(double)) == 0);
            ASSERT(memcmp(&dx, &obs[0], sizeof(double)) == 0);
            ASSERT(memcmp(&dy, &obs[1], sizeof(double)) == 0);
            ASSERT(memcmp(&dz, &obs[2], sizeof(double)) == 0);
            compared++;
        }
        ASSERT(compared == 4);
        ASSERT(crossed == 4);
        fs.destroy(env);
        dlclose(fso);
        printf("gate 5: %u steps with observer and target in different"
               " sectors, the pin holding bitwise where a flattened"
               " subtraction would not: OK\n", compared);
    }

    /* ---- Gate 6: what the getter reports after a fault ------------- */
    {
        void *xso;
        RlSurface xs;
        K26RlEnv *env = NULL;
        double a[1];
        double obs[5];
        double xbuf[1 * N_BODIES * PER_BODY];
        const int32_t xwant = (int32_t)(1 * N_BODIES * PER_BODY);
        uint32_t fl[1];
        double range;

        rl_write_file_(WORK_DIR "/bfault.kfl", FAULT_KFL);
        rl_compile_(WORK_DIR "/bfault.kfl", WORK_DIR "/bfault", WORK_DIR);
        xso = rl_dlopen_(WORK_DIR "/bfault.rlenv.so");
        rl_resolve_surface_(xso, &xs);
        ASSERT(xs.create(5, 1, &env) == K26RL_OK);

        /* An ordinary step first: the two surfaces agree, which is
         * what makes the disagreement below meaningful. */
        a[0] = 0.0;
        ASSERT(xs.step(env, a) == K26RL_OK);
        ASSERT(xs.obs(env, obs) == K26RL_OK);
        ASSERT(xs.bodies(env, EARTH, xbuf, (uint32_t)xwant) == xwant);
        {
            const double *r = xbuf + (size_t)CRAFT * PER_BODY;
            range = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            ASSERT(memcmp(&range, &obs[3], sizeof(double)) == 0);
        }

        /* Now fault it. The observation getters hold the pre-step
         * values, by design; the body getter reads the live worlds
         * and so reports the state the fault left. The two must
         * therefore disagree, which is the contract the header
         * states. */
        a[0] = 1.0;
        ASSERT(xs.step(env, a) == K26RL_OK);
        ASSERT(xs.flags(env, fl) == K26RL_OK);
        ASSERT(fl[0] & K26RL_FLAG_FAULT);
        ASSERT(xs.obs(env, obs) == K26RL_OK);
        ASSERT(xs.bodies(env, EARTH, xbuf, (uint32_t)xwant) == xwant);
        {
            const double *r = xbuf + (size_t)CRAFT * PER_BODY;
            range = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            ASSERT(memcmp(&range, &obs[3], sizeof(double)) != 0);
        }

        /* And the next boundary reset discards it: the two surfaces
         * agree again on the new episode's initial state. */
        a[0] = 0.0;
        ASSERT(xs.step(env, a) == K26RL_OK);
        ASSERT(xs.flags(env, fl) == K26RL_OK);
        ASSERT(fl[0] & K26RL_FLAG_RESET_BOUNDARY);
        ASSERT(xs.obs(env, obs) == K26RL_OK);
        ASSERT(xs.bodies(env, EARTH, xbuf, (uint32_t)xwant) == xwant);
        {
            const double *r = xbuf + (size_t)CRAFT * PER_BODY;
            range = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
            ASSERT(memcmp(&range, &obs[3], sizeof(double)) == 0);
        }

        xs.destroy(env);
        dlclose(xso);
        printf("gate 6: after a fault the getter reports the state the fault"
               " left while the observation getters hold the last honest"
               " values, and the boundary reset discards it: OK\n");
    }

    dlclose(so);
    printf("test_rl_bodies: 6 gates passed\n");
    return 0;
}
