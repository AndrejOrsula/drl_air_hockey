#!/usr/bin/env -S python3 -O

import os
import time

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent

from drl_air_hockey.agents import SpaceRAgent
from drl_air_hockey.td7 import TD7
from drl_air_hockey.utils.config import (
    ENV,
    EPISODE_MAX_STEPS,
    INTERPOLATION_ORDER,
    RENDER,
    REWARD_FUNCTION,
)

AGENT_SCHEME: int = 1
CONFIG_PRESET: int = 1

OBSERVATION_NOISE_ENABLED: bool = False
NOISE_STD: float = 0.025

# TODO: Fix
AGAINST_PREVIOUS_AGENT: bool = False


def main(argv=None):
    if not os.path.exists("./td7_results"):
        os.makedirs("./td7_results")

    env = make_env()
    eval_env = make_env()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD7.Agent(state_dim, action_dim, max_action)

    train_online(agent, env, eval_env)


def train_online(RL_agent, env, eval_env):
    evals = []
    start_time = time.time()
    allow_train = False

    state, ep_finished = env.reset(), False
    ep_total_reward, ep_timesteps, ep_num = 0, 0, 1

    for t in range(int(5e7 + 1)):
        maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time)

        if allow_train:
            action = RL_agent.select_action(np.array(state))
        else:
            action = env.action_space.sample()

        next_state, reward, ep_finished, _ = env.step(action)

        ep_total_reward += reward
        ep_timesteps += 1

        done = float(ep_finished) if ep_timesteps < EPISODE_MAX_STEPS else 0
        RL_agent.replay_buffer.add(state, action, next_state, reward, done)

        state = next_state

        if ep_finished:
            print(
                f"Total T: {t+1} Episode Num: {ep_num} Episode T: {ep_timesteps} Reward: {ep_total_reward:.3f}"
            )

            if allow_train:
                RL_agent.maybe_train_and_checkpoint(ep_timesteps, ep_total_reward)

            if t >= 1e5:
                allow_train = True

            state, done = env.reset(), False
            ep_total_reward, ep_timesteps = 0, 0
            ep_num += 1


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time):
    if t % 5e5 == 0:
        print("---------------------------------------")
        print(f"Evaluation at {t} time steps")
        print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")

        total_reward = np.zeros(3)
        for ep in range(3):
            state, done = eval_env.reset(), False
            while not done:
                action = RL_agent.select_action(
                    np.array(state), True, use_exploration=False
                )
                state, reward, done, _ = eval_env.step(action)
                total_reward[ep] += reward

        print(f"Average total reward over {3} episodes: {total_reward.mean():.3f}")

        print("---------------------------------------")

        evals.append(total_reward)
        np.save(f"./td7_results/td7_tournament_01", evals)


# Create the environment
def make_env(
    env_str=ENV.to_str(),
    reward_function=REWARD_FUNCTION,
    interpolation_order=INTERPOLATION_ORDER,
):
    env = AirHockeyChallengeWrapper(
        env_str,
        custom_reward_function=reward_function,
        interpolation_order=interpolation_order,
    )

    # Make the agent transparent to the environment
    agent_1 = SpaceRAgent(
        env.env_info,
        train=True,
        scheme=AGENT_SCHEME,
        max_episode_steps=EPISODE_MAX_STEPS,
        agent_id=1,
    )
    if AGAINST_PREVIOUS_AGENT:
        agent_2 = SpaceRAgent(
            env.env_info,
            train=False,
            scheme=AGENT_SCHEME,
            max_episode_steps=EPISODE_MAX_STEPS,
            agent_id=2,
        )
    else:
        agent_2 = BaselineAgent(env.env_info, agent_id=2)

    # Set the agents
    env._agent_1 = agent_1
    env._agent_2 = agent_2

    env.action_idx = (
        np.arange(env.base_env.action_shape[0][0]),
        np.arange(env.base_env.action_shape[1][0]),
    )

    # To make certain functions work (hack)
    env.scheme = env._agent_1.scheme
    env.n_stacked_obs = env._agent_1.n_stacked_obs

    return env


def _apply_monkey_patch_env_step():
    ## Patch action and observation spaces
    AirHockeyChallengeWrapper.action_space = SpaceRAgent.action_space
    AirHockeyChallengeWrapper.observation_space = SpaceRAgent.observation_space

    ## MONKEY PATCH: Make the agent transparent to the environment
    _original_step = AirHockeyChallengeWrapper.step

    def new_step(self, action):
        action_1 = self._agent_1.process_raw_act(action=action)
        action_2 = self._agent_2.draw_action(self._previous_obs2)
        combined_action = (action_1[self.action_idx[0]], action_2[self.action_idx[1]])

        obs, reward, done, info = _original_step(self, combined_action)

        obs1, obs2 = np.split(obs, 2)
        self._previous_obs2 = obs2
        obs = self._agent_1.process_raw_obs(obs=obs1)

        if OBSERVATION_NOISE_ENABLED:
            obs = np.clip(
                obs + np.random.normal(0.0, NOISE_STD, size=obs.shape), -1.0, 1.0
            )

        if RENDER:
            self.render()

        if self.step_counter >= EPISODE_MAX_STEPS:
            done = True
        else:
            self.step_counter += 1

        return obs, reward, done, info

    AirHockeyChallengeWrapper.step = new_step
    ## MONKEY PATCH: Make the agent transparent to the environment

    ## MONKEY PATCH: Make the agent transparent to the environment
    _original_reset = AirHockeyChallengeWrapper.reset

    def new_reset(self, state=None):
        self.step_counter = 0

        self._agent_1.reset()
        self._agent_2.reset()

        obs = _original_reset(self, state)

        obs1, obs2 = np.split(obs, 2)
        self._previous_obs2 = obs2
        obs = self._agent_1.process_raw_obs(obs=obs1)

        return obs

    AirHockeyChallengeWrapper.reset = new_reset
    ## MONKEY PATCH: Make the agent transparent to the environment


if __name__ == "__main__":
    _apply_monkey_patch_env_step()
    main()
