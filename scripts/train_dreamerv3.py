#!/usr/bin/env -S python3 -O

import os
import warnings
from functools import partial

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent
from dreamerv3 import Agent, embodied, wrap_env
from dreamerv3.embodied import wrappers

from drl_air_hockey.agents import SpaceRAgent
from drl_air_hockey.utils.config import (
    ENV,
    EPISODE_MAX_STEPS,
    INTERPOLATION_ORDER,
    RENDER,
    REWARD_FUNCTION,
    config_dreamerv3,
)
from drl_air_hockey.utils.env_wrapper import EmbodiedChallengeWrapper
from drl_air_hockey.utils.train import train_parallel

AGENT_SCHEME: int = 1
CONFIG_PRESET: int = 1

OBSERVATION_NOISE_ENABLED: bool = False
NOISE_STD: float = 0.025

# TODO: Fix
AGAINST_PREVIOUS_AGENT: bool = False


def main(argv=None):
    # Ignore certain warnings
    warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
    warnings.filterwarnings("ignore", ".*using stateful random seeds*")
    warnings.filterwarnings("ignore", ".*is a deprecated alias for.*")
    warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

    # Get config
    config = config_dreamerv3(train=True, preset=CONFIG_PRESET)
    config = embodied.Flags(config).parse(argv=[])

    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    print(config)

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    config.save(logdir / "config.yaml")
    step = embodied.Counter()
    logger = make_logger(logdir, step, config)

    cleanup = []

    command = "parallel" if config.envs.amount > 1 else "train"
    try:
        if command == "train":
            replay = make_replay(config, logdir / "replay")
            env = make_envs(config)
            cleanup.append(env)
            agent = Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train(agent, env, replay, logger, args)

        elif command == "train_save":
            replay = make_replay(config, logdir / "replay")
            env = make_envs(config)
            cleanup.append(env)
            agent = Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_save(agent, env, replay, logger, args)

        elif command == "train_eval":
            replay = make_replay(config, logdir / "replay")
            eval_replay = make_replay(config, logdir / "eval_replay", is_eval=True)
            env = make_envs(config)
            eval_env = make_envs(config)  # mode='eval'
            cleanup += [env, eval_env]
            agent = Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_eval(
                agent, env, eval_env, replay, eval_replay, logger, args
            )

        elif command == "train_holdout":
            replay = make_replay(config, logdir / "replay")
            if config.eval_dir:
                assert not config.train.eval_fill
                eval_replay = make_replay(config, config.eval_dir, is_eval=True)
            else:
                assert 0 < args.eval_fill <= config.replay_size // 10, args.eval_fill
                eval_replay = make_replay(config, logdir / "eval_replay", is_eval=True)
            env = make_envs(config)
            cleanup.append(env)
            agent = Agent(env.obs_space, env.act_space, step, config)
            embodied.run.train_holdout(agent, env, replay, eval_replay, logger, args)

        elif command == "parallel":
            assert config.run.actor_batch <= config.envs.amount, (
                config.run.actor_batch,
                config.envs.amount,
            )
            step = embodied.Counter()
            env = make_env(config)
            agent = Agent(env.obs_space, env.act_space, step, config)
            env.close()
            replay = make_replay(config, logdir / "replay", rate_limit=False)
            train_parallel(
                agent,
                replay,
                logger,
                partial(make_env, config),
                num_envs=config.envs.amount,
                args=args,
            )

        else:
            raise NotImplementedError(command)
    finally:
        for obj in cleanup:
            obj.close()


def make_logger(logdir, step, config):
    multiplier = config.env.get(config.task.split("_")[0], {}).get("repeat", 1)
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TensorBoardOutput(logdir),
        ],
        multiplier,
    )
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False, **kwargs):
    assert config.replay == "uniform" or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == "uniform" or is_eval:
        kw = {"online": config.replay_online}
        if rate_limit and config.run.train_ratio > 0:
            kw["samples_per_insert"] = config.run.train_ratio / config.batch_length
            kw["tolerance"] = 10 * config.batch_size
            kw["min_size"] = config.batch_size
        replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == "reverb":
        replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == "chunks":
        replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
        raise NotImplementedError(config.replay)
    return replay


def make_envs(config, **overrides):
    suite, task = config.task.split("_", 1)
    ctors = []
    for index in range(config.envs.amount):
        ctor = lambda: make_env(config, **overrides)
        if config.envs.parallel != "none":
            ctor = partial(embodied.Parallel, ctor, config.envs.parallel)
        if config.envs.restart:
            ctor = partial(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != "none"))


# Create the environment
def make_env(
    config,
    env_str=ENV.to_str(),
    reward_function=REWARD_FUNCTION,
    interpolation_order=INTERPOLATION_ORDER,
):
    _apply_monkey_patch_env_step()
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

    # Wrap the environment into embodied batch env
    env = EmbodiedChallengeWrapper(env)
    env = wrap_env(env, config)

    env = wrappers.TimeLimit(env, EPISODE_MAX_STEPS)

    return env


def _apply_monkey_patch_dreamerv3():
    ## MONKEY PATCH: Reduce preallocated JAX memory
    __monkey_patch__setup_original = Agent._setup

    def __monkey_patch__setup(self):
        __monkey_patch__setup_original(self)
        # Configuration for a large machine
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.925"

    Agent._setup = __monkey_patch__setup
    ## ~MONKEY PATCH:  Reduce preallocated JAX memory


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
        return obs, reward, done, info

    AirHockeyChallengeWrapper.step = new_step
    ## MONKEY PATCH: Make the agent transparent to the environment

    ## MONKEY PATCH: Make the agent transparent to the environment
    _original_reset = AirHockeyChallengeWrapper.reset

    def new_reset(self, state=None):
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
    _apply_monkey_patch_dreamerv3()
    main()
