#!/usr/bin/env python3

from typing import Union

from air_hockey_challenge.framework import AirHockeyChallengeWrapper

from drl_air_hockey.agents import SpaceRAgent
from drl_air_hockey.utils.config import INTERPOLATION_ORDER
from drl_air_hockey.utils.task import Task as AirHockeyTask

ENV = AirHockeyTask.R7_TOURNAMENT
RENDER: bool = True


def rand_agent(
    env: Union[AirHockeyTask, str] = ENV,
    interpolation_order: int = INTERPOLATION_ORDER,
    **kwargs,
):
    # Create the environment
    env = AirHockeyChallengeWrapper(
        env.to_str(),
        interpolation_order=interpolation_order,
    )

    # Make the agent transparent to the environment
    env_agent = SpaceRAgent(env.env_info, train=True, **kwargs)
    env._agent = env_agent

    while True:
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print(f"Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
            if RENDER:
                env.render()


def _apply_monkey_patch_env_step():
    ## Patch action and observation spaces
    AirHockeyChallengeWrapper.action_space = SpaceRAgent.action_space
    AirHockeyChallengeWrapper.observation_space = SpaceRAgent.observation_space

    ## MONKEY PATCH: Make the agent transparent to the environment
    _original_step = AirHockeyChallengeWrapper.step

    def new_step(self, action):
        action = self._agent.process_raw_act(action=action)
        obs, reward, done, info = _original_step(self, action)
        obs = self._agent.process_raw_obs(obs=obs)
        if RENDER:
            self.render()
        return obs, reward, done, info

    AirHockeyChallengeWrapper.step = new_step
    ## MONKEY PATCH: Make the agent transparent to the environment

    ## MONKEY PATCH: Make the agent transparent to the environment
    _original_reset = AirHockeyChallengeWrapper.reset

    def new_reset(self, state=None):
        self._agent.reset()
        obs = _original_reset(self, state)
        obs = self._agent.process_raw_obs(obs=obs)
        return obs

    AirHockeyChallengeWrapper.reset = new_reset
    ## MONKEY PATCH: Make the agent transparent to the environment


if __name__ == "__main__":
    _apply_monkey_patch_env_step()
    rand_agent()
