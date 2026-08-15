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
#include "k26compute.h"

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


/* A bias walk alone, so the published channel is the truth plus the
 * bias and the bias trajectory can be read straight out of it. A term
 * drawing at two cadences holds two channels; on one, an episode's
 * turn-on bias and its first kick are the same draw. */
#define BIAS_KFL \
    "form RL_BIAS\n" \
    "fn world b_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    sensor imu\n" \
    "        bias_walk 500.0 20.0 100.0\n" \
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
    " through imu with truth as look\n" \
    "    objective\n" \
    "        reward look_range + a * 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* A program that draws from the world's own stateful generator, which
 * the environment layer seeds per environment and per episode. The
 * draw is taken in the world prefix, where an impure builtin is
 * allowed; the per-step body admits only pure calls. */
#define WRNG_KFL \
    "form RL_WRNG\n" \
    "fn world r_world\n" \
    "    let g: rng = astro_world_rng(world)\n" \
    "    let d: double = rng_uniform(g) * 100000.0\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft mass=1.0 parent=earth" \
    " pos_x=7.0e6 pos_z=d vel_y=7546.049108166324\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 400\n" \
    "        substeps 1\n" \
    "    end\n" \
    "    action a box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        craft.vel_x = craft.vel_x + a * 0.0\n" \
    "    end\n" \
    "    observe craft from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward trk_range + a * 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* No sensor at all: every channel must publish as measured and
 * unpaired, which three places in the tree state and nothing gated. */
#define PLAIN_KFL \
    "form RL_PLAIN\n" \
    "fn world p_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    episode\n" \
    "        control_dt 1.0\n" \
    "        horizon 400\n" \
    "        substeps 1\n" \
    "    end\n" \
    "    action a box -1.0 1.0 default 0.0\n" \
    "    on_step\n" \
    "        craft.vel_x = craft.vel_x + a * 0.0\n" \
    "    end\n" \
    "    observe craft from earth mode=geometric as trk\n" \
    "    objective\n" \
    "        reward trk_range + a * 0.0\n" \
    "    end\n" \
    "end\n" \
    "end\n"

/* A step fine enough that an ordinary value is past what a signed
 * 64-bit integer holds. Before this item that published a sign-flipped
 * number with no fault and no diagnostic. */
#define FINE_KFL \
    "form RL_FINE\n" \
    "fn world f_world\n" \
    "    astro_body earth gm=3.986004418e14 mass=5.972e24\n" \
    "    astro_body craft mass=1.0 parent=earth" \
    " pos_x=7.0e6 vel_y=7546.049108166324\n" \
    "    sensor fine\n" \
    "        quantise 1e-14\n" \
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
    " through fine with truth as look\n" \
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

    /* ---- 6. A term drawing at two cadences holds two channels --- */
    rl_stage_("compiling the bias-walk artifact", 900u);
    {
        /* The rule this item exists to have got right, measured where
         * it applies. A bias walk draws once per episode at index 0
         * and once per step, whose first index is also 0. On one
         * channel those are the same number and the turn-on bias is
         * locked to the first kick, in every environment and every
         * episode. The compiler's allocation is what decides it, and
         * nothing here measured it: the published bias trajectory is
         * rebuilt from the two channels the rule gives and required to
         * match bitwise. */
        rl_write_file_(WORK_DIR "/bias.kfl", BIAS_KFL);
        rl_compile_(WORK_DIR "/bias.kfl", WORK_DIR "/bias", WORK_DIR);
        rl_stage_("driving the bias-walk artifact", 300u);
        void *bso = rl_dlopen_(WORK_DIR "/bias.rlenv.so");
        RlSurface bs;
        rl_resolve_surface_(bso, &bs);
        K26RlEnv *env = NULL;
        ASSERT(bs.create(seed, 1u, &env) == K26RL_OK);

        /* The range component is the fourth of five, and the chain has
         * one term drawing at both cadences, so the allocation gives
         * that component the seventh and eighth channels. */
        const uint16_t ch_step = (uint16_t)(2 * SN_RANGE);
        const uint16_t ch_ep   = (uint16_t)(2 * SN_RANGE + 1);
        double phi = 0.0, q = 0.0;
        ASSERT(k26sense_bias_walk_coeffs(20.0, 1.0, 100.0, &phi, &q)
               == K26SENSE_OK);

        K26RngCoords c;
        c.stream = K26SENSE_CLASS_SENSOR; c.environment = 0; c.episode = 0;
        c.channel = ch_ep; c.draw = 0;
        double bias = 500.0 * k26rng_normal(key, c);
        double same_channel = 500.0 * k26rng_normal(key,
            (K26RngCoords){ K26SENSE_CLASS_SENSOR, ch_step, 0, 0, 0 });
        printf("gate 6: the turn-on bias from its own channel is"
               " %+.9f; from the step channel it would be %+.9f\n",
               bias, same_channel);
        ASSERT(absd_(bias - same_channel) > 1.0);

        double obs[SN_OBS], act[1] = { 0.0 };
        int checked = 0;
        for (int t = 0; t < 25; t++) {
            ASSERT(bs.step(env, act) == K26RL_OK);
            ASSERT(bs.obs(env, obs) == K26RL_OK);
            c.channel = ch_step; c.draw = (uint32_t)t;
            bias = phi * bias + q * k26rng_normal(key, c);
            double want = obs[SN_TRUTH + SN_RANGE] + bias;
            if (t < 3) {
                printf("  step %d: published %.9f, rebuilt %.9f\n",
                       t + 1, obs[SN_MEAS + SN_RANGE], want);
            }
            ASSERT(obs[SN_MEAS + SN_RANGE] == want);
            checked++;
        }
        printf("  %d steps of the bias trajectory rebuilt from the two"
               " channels and matched bitwise\n", checked);
        ASSERT(checked == 25);
        bs.destroy(env);
        dlclose(bso);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 6: an episode's turn-on bias and its first kick are"
           " different draws: OK\n");

    /* ---- 7. A program that draws from the world's generator ------ */
    rl_stage_("compiling the world-generator artifact", 900u);
    {
        /* The clause this item could not previously meet. The world's
         * own stateful generator is seeded per environment and per
         * episode from a counter draw, so a program drawing from it
         * gets this environment's stream rather than one shared across
         * the whole vectorised set, and replays with everything else.
         *
         * The arm also checks the seed itself: the published value is
         * predicted here by seeding a generator with what the library
         * says this environment's seed is. The emitter carries its own
         * copy of that function, and this is what stops the two
         * drifting. */
        rl_write_file_(WORK_DIR "/wr.kfl", WRNG_KFL);
        rl_compile_(WORK_DIR "/wr.kfl", WORK_DIR "/wr", WORK_DIR);
        rl_stage_("driving the world-generator artifact", 300u);
        void *wso = rl_dlopen_(WORK_DIR "/wr.rlenv.so");
        RlSurface ws;
        rl_resolve_surface_(wso, &ws);

        enum { NE = 4, NB = 2 };
        static double b0[NE * NB * 6], b1[NE * NB * 6];
        for (int pass = 0; pass < 2; pass++) {
            K26RlEnv *env = NULL;
            ASSERT(ws.create(seed, (uint32_t)NE, &env) == K26RL_OK);
            int32_t need = ws.bodies(env, 0, NULL, 0);
            ASSERT(need == NE * NB * 6);
            ASSERT(ws.bodies(env, 0, pass ? b1 : b0, (uint32_t)need)
                   == need);
            ws.destroy(env);
        }
        printf("gate 7: craft z from the world's own generator:");
        for (int e = 0; e < NE; e++) {
            printf(" %+.6f", b0[e * NB * 6 + 6 + 2]);
        }
        printf("\n");

        /* Every environment differs, which one shared seed could not
         * produce. */
        for (int e = 0; e < NE; e++) {
            for (int f = e + 1; f < NE; f++) {
                ASSERT(b0[e * NB * 6 + 6 + 2] != b0[f * NB * 6 + 6 + 2]);
            }
        }
        /* And two passes are bitwise identical. */
        ASSERT(memcmp(b0, b1, sizeof b0) == 0);

        /* The seed the emitter used is the one the library computes. */
        for (int e = 0; e < NE; e++) {
            K26CRng r;
            k26c_rng_init(&r, k26sense_world_seed(key, (uint32_t)e, 0u));
            double want = k26c_rng_uniform(&r) * 100000.0;
            double got  = b0[e * NB * 6 + 6 + 2];
            if (e == 0) {
                printf("  environment 0 predicted %+.9f from the"
                       " library's seed, published %+.9f\n", want, got);
            }
            ASSERT(got == want);
        }
        dlclose(wso);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 7: a program drawing from the world's generator gets"
           " its own environment's stream, replays, and takes the seed"
           " the library computes: OK\n");

    /* ---- 8. A fine step saturates rather than flipping sign ------ */
    rl_stage_("compiling the fine-step artifact", 900u);
    {
        rl_write_file_(WORK_DIR "/fine.kfl", FINE_KFL);
        rl_compile_(WORK_DIR "/fine.kfl", WORK_DIR "/fine", WORK_DIR);
        rl_stage_("driving the fine-step artifact", 300u);
        void *fso = rl_dlopen_(WORK_DIR "/fine.rlenv.so");
        RlSurface fs;
        rl_resolve_surface_(fso, &fs);
        K26RlEnv *env = NULL;
        ASSERT(fs.create(seed, 1u, &env) == K26RL_OK);
        double obs[SN_OBS], act[1] = { 0.0 };
        for (int t = 0; t < 5; t++) {
            ASSERT(fs.step(env, act) == K26RL_OK);
            ASSERT(fs.obs(env, obs) == K26RL_OK);
            double meas  = obs[SN_MEAS + SN_RANGE];
            double truth = obs[SN_TRUTH + SN_RANGE];
            if (t == 0) {
                printf("gate 8: a step of 1e-14 on a range of %.6e:"
                       " published %+.6e\n", truth, meas);
            }
            /* Signed the same way as the truth, finite, and on the
             * grid: what it must not be is the sign-flipped number an
             * undefined conversion produced. */
            ASSERT(meas > 0.0);
            ASSERT(meas == meas);
            ASSERT(meas <= truth);
        }
        uint32_t fl = 0;
        ASSERT(fs.flags(env, &fl) == K26RL_OK);
        printf("  after five steps the flag word is 0x%08x\n", fl);
        fs.destroy(env);
        dlclose(fso);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 8: a step finer than the integer grid saturates with"
           " its sign intact: OK\n");

    /* ---- 9. A program with no sensor ----------------------------- */
    rl_stage_("compiling the plain artifact", 900u);
    {
        rl_write_file_(WORK_DIR "/plain.kfl", PLAIN_KFL);
        rl_compile_(WORK_DIR "/plain.kfl", WORK_DIR "/plain", WORK_DIR);
        void *pso = rl_dlopen_(WORK_DIR "/plain.rlenv.so");
        RlSurface ps;
        rl_resolve_surface_(pso, &ps);
        K26RlEnv *env = NULL;
        ASSERT(ps.create(seed, 1u, &env) == K26RL_OK);
        int32_t need = ps.spec(env, NULL, 0);
        ASSERT(need > 0);
        uint8_t *blob = (uint8_t *)malloc((size_t)need);
        ASSERT(blob);
        ASSERT(ps.spec(env, blob, (uint32_t)need) == need);
        int seen = 0;
        uint32_t off = 0;
        while (off + 6 <= (uint32_t)need) {
            uint16_t tag = rl_get_u16_(blob + off);
            uint32_t l   = rl_get_u32_(blob + off + 2);
            if (tag == K26RL_TAG_OBS_CHANNEL_SOURCE && l == 10) {
                uint16_t src  = rl_get_u16_(blob + off + 10);
                uint32_t pair = rl_get_u32_(blob + off + 12);
                ASSERT(src == K26RL_OBS_SOURCE_MEASURED);
                ASSERT(pair == K26RL_OBS_PAIR_NONE);
                seen++;
            }
            off += 6 + l;
        }
        printf("gate 9: a program with no sensor: %d channels, every"
               " one measured and unpaired\n", seen);
        ASSERT(seen == 5);
        free(blob);
        ps.destroy(env);
        dlclose(pso);
        n_pass++;
    }
    rl_stage_done_();
    printf("gate 9: no sensor means every channel is measured and"
           " unpaired, which three places state: OK\n");

    dlclose(so);
    printf("test_rl_sensor: %d gates passed\n", n_pass);
    return 0;
}
