/* test_rl_sensor.c: the imperfection layer through the stepping
 * surface.
 *
 * The library's own gates measure each model against the behaviour it
 * is defined by, and the draw discipline that says whose a draw is.
 * This one measures what only an artifact can show: that a declared
 * sensor is actually applied on the step path, that it is applied at
 * the coordinates the layer's allocation rule gives, that the truth
 * channels carry the uncorrupted values beside it, that the spec's
 * source tag resolves each channel to its pair, and that two runs of
 * one artifact agree bitwise.
 *
 * What would make these arms vacuous, and how each is ruled out.
 *
 *   An arm that only checked the measured channels differ from the
 *   truth channels would pass for any corruption at all, including
 *   one at the wrong coordinates or in the wrong order. The gate
 *   recomputes the expected measured value with the library, from the
 *   truth channel the same artifact published and the coordinates the
 *   allocation rule gives, and requires the two to agree bitwise.
 *
 *   A fixture whose sensor is a single noise term cannot show that a
 *   chain is applied in the order declared. The fixture carries a
 *   noise term, a delay and a quantiser, so the delay's index shift
 *   and the quantiser's grid are both visible in the published values
 *   and a chain applied in another order lands off the grid.
 *
 *   An arm run at one environment cannot show the draws are addressed
 *   per environment. The determinism arm drives eight environments and
 *   requires environment j's measured stream to match the stream it
 *   produces when it is the only environment in the handle.
 *
 *   A pairing arm that read the tag without reading the channels would
 *   pass for a tag that named the wrong pair. Every measured channel's
 *   pair is followed to the truth channel it names, and that channel
 *   is required to be the uncorrupted value.
 *
 * Requires the sibling stack archives (skips with 77 otherwise).
 */
#define _GNU_SOURCE
#include <signal.h>
#include <unistd.h>
#include <math.h>

#include "rl_gate_util.h"
#include "k26sense.h"

#define WORK_DIR "/tmp/kflc_rl_sensor_test"

static int n_pass = 0;
static double absd_(double x) { return x < 0.0 ? -x : x; }

static const char *rl_stage_name_ = "startup";
static void rl_deadline_fired_(int sig)
{
    (void)sig;
    const char *a = "test_rl_sensor: DEADLINE EXCEEDED at stage: ";
    (void)!write(2, a, strlen(a));
    (void)!write(2, rl_stage_name_, strlen(rl_stage_name_));
    (void)!write(2, "\n", 1);
    _exit(1);
}
static void rl_stage_(const char *name, unsigned secs)
{ rl_stage_name_ = name; alarm(secs); }
static void rl_stage_done_(void) { alarm(0); }

/* The fixture. A noise term, a delay and a quantiser, in that order,
 * so the published value is on the quantiser's grid and lags the truth
 * by exactly two steps. The noise is large against the grid step, so a
 * chain that dropped the noise would sit on the truth's own value and
 * be visible. */
#define SENSOR_KFL \
    "form RL_SENSOR\n" \
    "fn world sn_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    sensor rangefinder\n" \
    "        noise normal 0.0 40.0\n" \
    "        latency 2\n" \
    "        quantise 5.0\n" \
    "    end\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 400\n" \
    "        substeps 1\n" \
    "    end\n" \
    "    action a box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        craft.vel_x = craft.vel_x + a * 0.0\n" \
    "    end\n" \
    "    observe craft from earth mode=geometric" \
    " through rangefinder with truth as look\n" \
    "    objective\n" \
    "        reward look_range + a * 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* Five measured channels then five paired truth channels. */
#define SN_OBS    10
#define SN_MEAS    0
#define SN_TRUTH   5
#define SN_RANGE   3      /* the component this gate follows */
#define SN_LSB     5.0
#define SN_DEPTH   2
#define SN_SIGMA  40.0

int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGALRM, rl_deadline_fired_);

    if (!rl_libs_present_("test_rl_sensor")) return 77;
    rl_run_or_die_("rm -rf " WORK_DIR " && mkdir -p " WORK_DIR);
    rl_write_file_(WORK_DIR "/sn.kfl", SENSOR_KFL);

    rl_stage_("compiling the sensor artifact", 900u);
    rl_compile_(WORK_DIR "/sn.kfl", WORK_DIR "/sn", WORK_DIR);
    ASSERT(rl_file_exists_(WORK_DIR "/sn.rlenv.so"));
    rl_stage_done_();

    void *so = rl_dlopen_(WORK_DIR "/sn.rlenv.so");
    RlSurface s;
    rl_resolve_surface_(so, &s);
    ASSERT(s.abi_version() == K26RL_ABI_VERSION);

    const uint64_t seed = 31u;
    K26RngKey key = k26rng_key(seed);

    /* ---- 1. The spec pairs every channel with its opposite ------- */
    rl_stage_("reading the spec", 60u);
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(seed, 1u, &env) == K26RL_OK);
        int32_t need = s.spec(env, NULL, 0);
        ASSERT(need > 0);
        uint8_t *blob = (uint8_t *)malloc((size_t)need);
        ASSERT(blob);
        ASSERT(s.spec(env, blob, (uint32_t)need) == need);

        uint16_t src[SN_OBS];
        uint32_t pair[SN_OBS];
        int seen = 0;
        for (int i = 0; i < SN_OBS; i++) { src[i] = 0xFFFFu; pair[i] = 0u; }
        uint32_t off = 0;
        while (off + 6 <= (uint32_t)need) {
            uint16_t tag = rl_get_u16_(blob + off);
            uint32_t l   = rl_get_u32_(blob + off + 2);
            if (tag == K26RL_TAG_OBS_CHANNEL_SOURCE && l == 10) {
                uint32_t ch = rl_get_u32_(blob + off + 6);
                if (ch < SN_OBS) {
                    src[ch]  = rl_get_u16_(blob + off + 10);
                    pair[ch] = rl_get_u32_(blob + off + 12);
                    seen++;
                }
            }
            off += 6 + l;
        }
        printf("gate 1: %d channels carry a source tag of %d\n", seen,
               SN_OBS);
        ASSERT(seen == SN_OBS);
        for (int c = 0; c < 5; c++) {
            printf("  channel %d source %u pair %u; channel %d source %u"
                   " pair %u\n", c, src[c], pair[c], c + SN_TRUTH,
                   src[c + SN_TRUTH], pair[c + SN_TRUTH]);
            ASSERT(src[c] == K26RL_OBS_SOURCE_MEASURED);
            ASSERT(pair[c] == (uint32_t)(c + SN_TRUTH));
            ASSERT(src[c + SN_TRUTH] == K26RL_OBS_SOURCE_TRUTH);
            ASSERT(pair[c + SN_TRUTH] == (uint32_t)c);
        }
        free(blob);
        s.destroy(env);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 1: the source tag names every channel's kind and its"
           " pair, both ways round: OK\n");

    /* ---- 2. The chain is applied, at the right coordinates ------- */
    rl_stage_("driving the artifact", 300u);
    {
        K26RlEnv *env = NULL;
        ASSERT(s.create(seed, 1u, &env) == K26RL_OK);
        double obs[SN_OBS], act[1] = { 0.0 };

        /* The reset observation is the uncorrupted value: a per-step
         * term's draw index is the transition index, and at a boundary
         * no transition has been taken. */
        ASSERT(s.obs(env, obs) == K26RL_OK);
        printf("gate 2: at the boundary the measured range is %.6f and"
               " the truth is %.6f\n", obs[SN_MEAS + SN_RANGE],
               obs[SN_TRUTH + SN_RANGE]);
        ASSERT(obs[SN_MEAS + SN_RANGE] == obs[SN_TRUTH + SN_RANGE]);

        /* The chain, rebuilt here. The allocation rule gives the noise
         * term channel 0 of the range component: channels are
         * allocated in source order, one per drawing term, over the
         * observe's components in order, and this sensor's only
         * drawing term is the noise. */
        const uint16_t ch = (uint16_t)SN_RANGE;
        double pending[SN_DEPTH];
        int    n_pending = 0;
        double ring_out = 0.0;
        int    differed = 0, on_grid = 0, checked = 0;
        double reset_value = obs[SN_TRUTH + SN_RANGE];

        for (int t = 0; t < 40; t++) {
            ASSERT(s.step(env, act) == K26RL_OK);
            ASSERT(s.obs(env, obs) == K26RL_OK);
            double truth = obs[SN_TRUTH + SN_RANGE];
            double meas  = obs[SN_MEAS + SN_RANGE];

            /* Noise at this transition's own coordinates, then the
             * delay, then the grid: the chain in the order declared. */
            K26RngCoords c;
            c.stream = K26SENSE_CLASS_SENSOR; c.channel = ch;
            c.environment = 0; c.episode = 0; c.draw = (uint32_t)t;
            double noisy = truth + SN_SIGMA * k26rng_normal(key, c);

            if (n_pending < SN_DEPTH) {
                ring_out = reset_value;
            } else {
                ring_out = pending[t % SN_DEPTH];
            }
            pending[t % SN_DEPTH] = noisy;
            if (n_pending < SN_DEPTH) n_pending++;

            double want = k26sense_quantise(ring_out, SN_LSB, -1.0e15,
                                            1.0e15);
            if (t < 6 || t == 39) {
                printf("  step %2d: truth %.6f, measured %.6f, expected"
                       " %.6f\n", t + 1, truth, meas, want);
            }
            ASSERT(meas == want);
            checked++;
            if (meas != truth) differed++;
            double g = meas / SN_LSB;
            if (absd_(g - (double)(int64_t)(g < 0 ? g - 0.5 : g + 0.5))
                < 1e-9) {
                on_grid++;
            }
        }
        printf("  %d of %d steps agreed with the chain rebuilt here;"
               " %d differed from the truth; %d sat on the grid\n",
               checked, checked, differed, on_grid);
        /* A pass that never ran would leave the measured channel equal
         * to the truth at every step, and off the quantiser's grid. */
        ASSERT(differed > 30);
        ASSERT(on_grid == checked);
        s.destroy(env);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 2: the declared chain is applied on the step path, in"
           " order, at the coordinates the allocation rule gives: OK\n");

    /* ---- 3. Neighbours do not move an environment's stream ------- */
    rl_stage_("driving eight environments", 300u);
    {
        enum { N = 8, T = 30 };
        static double alone[T], wide[T];

        K26RlEnv *one = NULL;
        ASSERT(s.create(seed, 1u, &one) == K26RL_OK);
        double obs1[SN_OBS], act1[1] = { 0.0 };
        for (int t = 0; t < T; t++) {
            ASSERT(s.step(one, act1) == K26RL_OK);
            ASSERT(s.obs(one, obs1) == K26RL_OK);
            alone[t] = obs1[SN_MEAS + SN_RANGE];
        }
        s.destroy(one);

        K26RlEnv *many = NULL;
        ASSERT(s.create(seed, (uint32_t)N, &many) == K26RL_OK);
        static double obsN[N * SN_OBS], actN[N];
        for (int t = 0; t < T; t++) {
            ASSERT(s.step(many, actN) == K26RL_OK);
            ASSERT(s.obs(many, obsN) == K26RL_OK);
            wide[t] = obsN[0 * SN_OBS + SN_MEAS + SN_RANGE];
        }
        int same = memcmp(alone, wide, sizeof alone) == 0;
        /* And a neighbour's own channel is a different stream, so the
         * comparison above is not of two copies of one number. */
        double neighbour = obsN[3 * SN_OBS + SN_MEAS + SN_RANGE];
        printf("gate 3: environment 0's measured stream over %d steps,"
               " alone and beside %d neighbours: %s\n", T, N - 1,
               same ? "bitwise identical" : "DIFFERENT");
        printf("  environment 3's last measured value is %.6f against"
               " environment 0's %.6f\n", neighbour, wide[T - 1]);
        ASSERT(same);
        ASSERT(neighbour != wide[T - 1]);
        s.destroy(many);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 3: an environment's noise is its own, whatever its"
           " neighbours draw: OK\n");

    /* ---- 4. Two runs of one artifact agree bitwise --------------- */
    {
        enum { T = 25 };
        static double first[T], again[T];
        for (int pass = 0; pass < 2; pass++) {
            K26RlEnv *env = NULL;
            ASSERT(s.create(seed, 1u, &env) == K26RL_OK);
            double obs[SN_OBS], act[1] = { 0.0 };
            for (int t = 0; t < T; t++) {
                ASSERT(s.step(env, act) == K26RL_OK);
                ASSERT(s.obs(env, obs) == K26RL_OK);
                (pass ? again : first)[t] = obs[SN_MEAS + SN_RANGE];
            }
            s.destroy(env);
        }
        int same = memcmp(first, again, sizeof first) == 0;
        printf("gate 4: two runs at seed %llu over %d steps: %s\n",
               (unsigned long long)seed, T,
               same ? "bitwise identical" : "DIFFERENT");
        ASSERT(same);
        n_pass++;
    }
    printf("gate 4: the measured stream replays bitwise: OK\n");

    /* ---- 5. The longest name a program can declare survives ----- */
    rl_stage_("compiling the long-name artifact", 900u);
    {
        /* A declarable `as` name is bounded at 53 bytes and a paired
         * channel adds `_truth_range_rate`, 17 more. The spec's name
         * entry holds 96, so the 70-byte derived name must come back
         * whole; at the 64 the entry used to hold it would have come
         * back cut to 64 with no diagnostic, which is the silent
         * truncation this arm exists to keep out. */
        char base[54];
        memset(base, 'a', 53);
        base[53] = '\0';
        char *src = (char *)malloc(4096);
        ASSERT(src);
        snprintf(src, 4096,
            "form RL_LONGNAME\n"
            "fn world ln_world\n"
            "    astro_body earth gm=3.986004418e14 mass=5.972e24\n"
            "    astro_body craft mass=1.0 parent=earth"
            " pos_x=7.0e6 vel_y=7546.0\n"
            "    sensor s\n        noise normal 0.0 1.0\n    end\n"
            "    episode\n        control_dt 1.0\n        horizon 8\n"
            "    end\n"
            "    action a box -1.0 1.0 default 0.0\n"
            "    observe craft from earth mode=geometric"
            " through s with truth as %s\n"
            "    objective\n        reward %s_range + a * 0.0\n"
            "    end\nend\nend\n", base, base);
        rl_write_file_(WORK_DIR "/ln.kfl", src);
        rl_compile_(WORK_DIR "/ln.kfl", WORK_DIR "/ln", WORK_DIR);
        free(src);
        rl_stage_done_();

        void *lso = rl_dlopen_(WORK_DIR "/ln.rlenv.so");
        RlSurface ls;
        rl_resolve_surface_(lso, &ls);
        K26RlEnv *env = NULL;
        ASSERT(ls.create(seed, 1u, &env) == K26RL_OK);
        int32_t need = ls.spec(env, NULL, 0);
        ASSERT(need > 0);
        uint8_t *blob = (uint8_t *)malloc((size_t)need);
        ASSERT(blob);
        ASSERT(ls.spec(env, blob, (uint32_t)need) == need);

        char want[128];
        snprintf(want, sizeof want, "%s_truth_range_rate", base);
        int found = 0;
        size_t longest = 0;
        uint32_t off = 0;
        while (off + 6 <= (uint32_t)need) {
            uint16_t tag = rl_get_u16_(blob + off);
            uint32_t l   = rl_get_u32_(blob + off + 2);
            if (tag == K26RL_TAG_OBS_CHANNEL_NAME && l >= 4) {
                size_t nl = l - 4;
                if (nl > longest) longest = nl;
                if (nl == strlen(want) &&
                    memcmp(blob + off + 10, want, nl) == 0) {
                    found = 1;
                }
            }
            off += 6 + l;
        }
        printf("gate 5: a %zu-byte declared name with a paired truth"
               " channel; the longest published name is %zu bytes and"
               " the %zu-byte derived name came back whole: %s\n",
               strlen(base), longest, strlen(want),
               found ? "yes" : "NO");
        ASSERT(longest == strlen(want));
        ASSERT(found);
        free(blob);
        ls.destroy(env);
        dlclose(lso);
        n_pass++;
    }
    printf("gate 5: the longest name a program can declare survives"
           " the spec entry whole: OK\n");

    dlclose(so);
    printf("test_rl_sensor: %d gates passed\n", n_pass);
    return 0;
}
