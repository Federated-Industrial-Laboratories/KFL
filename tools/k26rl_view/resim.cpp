/* resim.cpp - deterministic re-simulation, reconstruct and compare.
 *
 * The technique is the one the re-simulation gate established and
 * pins: create a handle at the seed the episode's ordinal resolves
 * to, reach the episode index through explicit resets so no simulated
 * time passes on the way, then replay the recorded action row and
 * compare the target environment's slice at every step.
 *
 * The recorded row is driven into every environment of the handle
 * rather than only the target's slot. An environment's streams are
 * unaffected by its neighbours' actions, which the vector
 * independence gate pins, so this reproduces the target exactly while
 * keeping the drive a single memcpy per environment.
 */
#include "resim.h"

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

namespace k26rl_view {

/* The major this build speaks, and the minor it needs. A consumer
 * checks major equality and minor at-least, and this viewer needs
 * nothing the first minor did not have: it reads, it does not tap. */
static const uint32_t RESIM_ABI_MAJOR = 1;
static const uint32_t RESIM_ABI_MINOR_MIN = 0;

namespace {

struct Surface {
    uint32_t (*abi_version)(void);
    K26RlStatus (*create)(uint64_t, uint32_t, K26RlEnv **);
    K26RlStatus (*reset)(K26RlEnv *);
    K26RlStatus (*step)(K26RlEnv *, const double *);
    K26RlStatus (*obs)(const K26RlEnv *, double *);
    K26RlStatus (*reward)(const K26RlEnv *, double *);
    K26RlStatus (*flags)(const K26RlEnv *, uint32_t *);
    const char *(*status_str)(K26RlStatus);
    void (*destroy)(K26RlEnv *);
    /* Added at ABI 1.2 and probed rather than required, so an older
     * artifact still re-simulates; only the world frame goes. */
    int32_t (*bodies)(const K26RlEnv *, uint32_t, double *, uint32_t);
    /* Added at ABI 1.4, probed on the same terms: an artifact
     * without it re-simulates and loses the attitude panel alone. */
    int32_t (*attitudes)(const K26RlEnv *, double *, uint32_t);
    /* Added at ABI 1.6, probed on the same terms: an artifact
     * without it re-simulates and loses the actuator drives alone. */
    int32_t (*actuators)(const K26RlEnv *, double *, uint32_t);
    /* Added at ABI 1.7, probed on the same terms: an artifact
     * without it re-simulates and loses the datalink lines alone. */
    int32_t (*datalinks)(const K26RlEnv *, double *, uint32_t);
};

bool resolve_(void *so, const char *name, void *slot, std::string *err)
{
    void *p = dlsym(so, name);
    if (!p) {
        *err = std::string("artifact is missing required symbol ") + name;
        return false;
    }
    memcpy(slot, &p, sizeof p);
    return true;
}

}  /* namespace */

ResimResult resimulate(const Model &model, const Episode &ep,
                       const std::string &artifact_path, uint32_t reference)
{
    ResimResult r;
    Surface s;
    void *so;
    K26RlEnv *env = 0;
    const uint32_t n = model.info().n_envs ? model.info().n_envs : 1;
    const uint32_t obs_total = model.spec().obs_total;
    const uint32_t act_total = model.spec().act_total;
    const uint32_t agents = model.spec().agent_count ? model.spec().agent_count : 1;

    memset(&s, 0, sizeof s);
    /* A rebuild is the recorded action stream driven again, so a
     * stream with a hole in it cannot be rebuilt: the actions after
     * the hole belong to a state this side never reached. A live
     * source the producer outran holds exactly such a stream, and
     * refusing it here is what keeps the world frame, the attitudes
     * and the scene from being drawn from a reconstruction that
     * diverged at the first missing action. */
    if (!ep.gaps.empty() || !ep.start_seen) {
        char n[64];
        uint32_t missing = 0;
        for (size_t g = 0; g < ep.gaps.size(); g++)
            missing += ep.gaps[g].count;
        snprintf(n, sizeof n, "%u", (unsigned)missing);
        r.message = std::string("the held step stream has holes in it: ") +
                    n + " step records were overwritten before this "
                    "viewer read them, so the action stream cannot be "
                    "driven again";
        return r;
    }

    /* RTLD_LOCAL because an artifact carries its own statically linked
     * physics symbols, and two artifacts loaded together must not see
     * each other's. */
    so = dlopen(artifact_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!so) {
        const char *e = dlerror();
        r.message = std::string("cannot load artifact: ") + (e ? e : "unknown");
        return r;
    }

    if (!resolve_(so, "k26rl_abi_version", &s.abi_version, &r.message) ||
        !resolve_(so, "k26rl_env_create", &s.create, &r.message) ||
        !resolve_(so, "k26rl_env_reset", &s.reset, &r.message) ||
        !resolve_(so, "k26rl_env_step", &s.step, &r.message) ||
        !resolve_(so, "k26rl_env_obs", &s.obs, &r.message) ||
        !resolve_(so, "k26rl_env_reward", &s.reward, &r.message) ||
        !resolve_(so, "k26rl_env_flags", &s.flags, &r.message) ||
        !resolve_(so, "k26rl_status_str", &s.status_str, &r.message) ||
        !resolve_(so, "k26rl_env_destroy", &s.destroy, &r.message)) {
        dlclose(so);
        return r;
    }

    {
        void *p = dlsym(so, "k26rl_env_bodies");
        if (p)
            memcpy(&s.bodies, &p, sizeof p);
        p = dlsym(so, "k26rl_env_attitudes");
        if (p)
            memcpy(&s.attitudes, &p, sizeof p);
        p = dlsym(so, "k26rl_env_actuators");
        if (p)
            memcpy(&s.actuators, &p, sizeof p);
        p = dlsym(so, "k26rl_env_datalinks");
        if (p)
            memcpy(&s.datalinks, &p, sizeof p);
    }
    r.datalink_symbol = s.datalinks != 0;

    r.abi_version = s.abi_version();
    if ((r.abi_version >> 16) != RESIM_ABI_MAJOR) {
        char buf[128];
        snprintf(buf, sizeof buf,
                 "artifact reports ABI major %u; this build speaks major %u",
                 r.abi_version >> 16, RESIM_ABI_MAJOR);
        r.message = buf;
        dlclose(so);
        return r;
    }
    if ((r.abi_version & 0xFFFFu) < RESIM_ABI_MINOR_MIN) {
        char buf[128];
        snprintf(buf, sizeof buf,
                 "artifact reports ABI minor %u; this build needs at least %u",
                 r.abi_version & 0xFFFFu, RESIM_ABI_MINOR_MIN);
        r.message = buf;
        dlclose(so);
        return r;
    }

    {
        K26RlStatus st = s.create(ep.seed, n, &env);
        if (st != K26RL_OK) {
            r.message = std::string("create failed: ") + s.status_str(st);
            dlclose(so);
            return r;
        }
    }

    /* Direct addressing by identity: every explicit reset advances
     * each environment's episode index by one with no step and no
     * time advance, so index `episode` is reached without replaying
     * the episodes before it. */
    for (uint32_t k = 0; k < ep.episode; k++) {
        K26RlStatus st = s.reset(env);
        if (st != K26RL_OK) {
            r.message = std::string("reset failed: ") + s.status_str(st);
            s.destroy(env);
            dlclose(so);
            return r;
        }
    }

    std::vector<double> obs((size_t)n * obs_total);
    std::vector<double> rew((size_t)n * agents);
    std::vector<double> act((size_t)n * (act_total ? act_total : 1));
    std::vector<uint32_t> fl(n);

    if (s.obs(env, obs.empty() ? 0 : &obs[0]) != K26RL_OK) {
        r.message = "initial observation read failed";
        s.destroy(env);
        dlclose(so);
        return r;
    }
    r.initial_obs.assign(obs.begin() + (size_t)ep.env * obs_total,
                         obs.begin() + (size_t)(ep.env + 1) * obs_total);

    std::vector<double> body_buf;
    if (s.bodies) {
        int32_t need = s.bodies(env, reference, 0, 0);
        if (need > 0 && n && (uint32_t)need % (n * 6u) == 0) {
            body_buf.resize((size_t)need);
            r.body_count = (uint32_t)need / (n * 6u);
            r.has_bodies = true;
        }
    }
    /* The attitude getter sizes itself the same way, and its body
     * count is the same body count: a capacity of zero returns the
     * requirement rather than writing anything. */
    std::vector<double> att_buf;
    if (s.attitudes) {
        int32_t need = s.attitudes(env, 0, 0);
        if (need > 0 && n && (uint32_t)need % (n * 7u) == 0) {
            att_buf.resize((size_t)need);
            uint32_t count = (uint32_t)need / (n * 7u);
            if (!r.has_bodies)
                r.body_count = count;
            if (count == r.body_count)
                r.has_attitudes = true;
        }
    }
    /* The actuator getter sizes itself the same way. Its count is
     * its own: actuators, not bodies. */
    std::vector<double> drv_buf;
    if (s.actuators) {
        int32_t need = s.actuators(env, 0, 0);
        if (need > 0 && n && (uint32_t)need % (n * 10u) == 0) {
            drv_buf.resize((size_t)need);
            r.actuator_count = (uint32_t)need / (n * 10u);
            r.has_actuators = true;
        }
    }
    /* The datalink getter sizes itself the same way. Its count is its
     * own again: ordered transmitter and receiver pairs, five doubles
     * each, and zero for a program that declares no datalink. */
    std::vector<double> lnk_buf;
    if (s.datalinks) {
        int32_t need = s.datalinks(env, 0, 0);
        if (need > 0 && n && (uint32_t)need % (n * 5u) == 0) {
            lnk_buf.resize((size_t)need);
            r.datalink_count = (uint32_t)need / (n * 5u);
            r.has_datalinks = true;
        }
    }

    r.ran = true;
    r.equal = true;
    /* The episode-start frame's initial observation is reached with no
     * stepping at all, which is what pins the reset boundary's zero
     * time advance; it is compared before any transition. */
    if (obs_total && ep.initial_obs.size() == obs_total &&
        memcmp(&r.initial_obs[0], &ep.initial_obs[0],
               sizeof(double) * obs_total) != 0) {
        r.equal = false;
        r.first_divergence = 0;
        r.divergence_kind = "initial observation";
    }

    for (uint32_t t = 0; t < ep.step_count; t++) {
        for (uint32_t j = 0; j < n && act_total; j++) {
            memcpy(&act[(size_t)j * act_total],
                   &ep.act[(size_t)t * act_total],
                   sizeof(double) * act_total);
        }
        K26RlStatus st = s.step(env, act.empty() ? 0 : &act[0]);
        if (st != K26RL_OK) {
            r.message = std::string("step failed: ") + s.status_str(st);
            r.equal = false;
            break;
        }
        if (s.obs(env, &obs[0]) != K26RL_OK ||
            s.reward(env, &rew[0]) != K26RL_OK ||
            s.flags(env, &fl[0]) != K26RL_OK) {
            r.message = "getter read failed";
            r.equal = false;
            break;
        }
        r.obs.insert(r.obs.end(), obs.begin() + (size_t)ep.env * obs_total,
                     obs.begin() + (size_t)(ep.env + 1) * obs_total);
        r.rewards.insert(r.rewards.end(),
                         rew.begin() + (size_t)ep.env * agents,
                         rew.begin() + (size_t)(ep.env + 1) * agents);
        r.flags.push_back(fl[ep.env]);
        /* The world frame, taken at the same instant as the streams
         * above so a viewer can draw them together. */
        if (r.has_bodies &&
            s.bodies(env, reference, &body_buf[0],
                     (uint32_t)body_buf.size()) > 0) {
            size_t base = (size_t)ep.env * r.body_count * 6;
            r.bodies.insert(r.bodies.end(), body_buf.begin() + base,
                            body_buf.begin() + base + r.body_count * 6);
        }
        /* The attitudes, at that same instant and for the same
         * bodies, so a panel drawing an orientation beside a position
         * is drawing one moment. */
        if (r.has_attitudes &&
            s.attitudes(env, &att_buf[0], (uint32_t)att_buf.size()) > 0) {
            size_t base = (size_t)ep.env * r.body_count * 7;
            r.attitudes.insert(r.attitudes.end(), att_buf.begin() + base,
                               att_buf.begin() + base + r.body_count * 7);
        }
        /* The actuator drives, at that same instant: what the step
         * just taken applied, which is the figure a force line over
         * this step should draw. */
        if (r.has_actuators &&
            s.actuators(env, &drv_buf[0], (uint32_t)drv_buf.size()) > 0) {
            size_t base = (size_t)ep.env * r.actuator_count * 10;
            r.actuators.insert(r.actuators.end(), drv_buf.begin() + base,
                               drv_buf.begin() + base +
                                   r.actuator_count * 10);
        }
        /* The datalink state, at that same instant: what the step just
         * taken left each pair holding, which is what a link line over
         * this step should draw. */
        if (r.has_datalinks &&
            s.datalinks(env, &lnk_buf[0], (uint32_t)lnk_buf.size()) > 0) {
            size_t base = (size_t)ep.env * r.datalink_count * 5;
            r.datalinks.insert(r.datalinks.end(), lnk_buf.begin() + base,
                               lnk_buf.begin() + base +
                                   r.datalink_count * 5);
        }
        r.steps_compared++;

        if (r.equal) {
            const double *rec_obs = &ep.obs[(size_t)t * obs_total];
            const double *got_obs = &obs[(size_t)ep.env * obs_total];
            if (obs_total &&
                memcmp(got_obs, rec_obs, sizeof(double) * obs_total) != 0) {
                r.equal = false;
                r.first_divergence = t;
                r.divergence_kind = "observation";
            } else if (memcmp(&rew[(size_t)ep.env * agents],
                              &ep.rewards[(size_t)t * agents],
                              sizeof(double) * agents) != 0) {
                r.equal = false;
                r.first_divergence = t;
                r.divergence_kind = "reward";
            } else if (fl[ep.env] != ep.flags[t]) {
                r.equal = false;
                r.first_divergence = t;
                r.divergence_kind = "flags";
            }
        }
    }

    if (r.equal)
        r.message = "every compared value bit-identical";
    s.destroy(env);
    dlclose(so);
    return r;
}

}  /* namespace k26rl_view */
