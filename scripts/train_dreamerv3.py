#!/usr/bin/env -S python3 -O

import os
import shutil
import warnings
from functools import partial

import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent
from dreamerv3 import Agent, embodied, wrap_env
from dreamerv3.embodied import wrappers

from drl_air_hockey.agents import SpaceRAgent
from drl_air_hockey.utils.config import (
    AGENT_STRATEGY,
    DIR_MODELS,
    ENV,
    EPISODE_MAX_STEPS,
    INTERPOLATION_ORDER,
    RENDER,
    config_dreamerv3,
)
from drl_air_hockey.utils.env_wrapper import EmbodiedChallengeWrapper
from drl_air_hockey.utils.tournament_agent_strategies import (
    strategy_from_str,
    strategy_to_str,
)
from drl_air_hockey.utils.train import train_parallel

AGENT_SCHEME: int = 7
CONFIG_PRESET: int = 1

DELAYED_SELF_PLAY: bool = True
SAVE_NEW_OPPONENT_EVERY_N_EPISODES: int = 1000
MAX_N_MODELS: int = 25

XLA_PYTHON_CLIENT_MEM_FRACTION: float = 0.9


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
    reward_function=AGENT_STRATEGY.get_reward_function(),
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
        agent_id=1,
        interpolation_order=interpolation_order,
        train=True,
        scheme=AGENT_SCHEME,
        **AGENT_STRATEGY.get_env_kwargs(),
    )
    # List of opponents that are randomly selected during the training for each episode
    env._opponent_models = [
        BaselineAgent(env.env_info, agent_id=2),
    ]
    if DELAYED_SELF_PLAY:
        for filename in os.listdir(DIR_MODELS):
            model_path = os.path.join(DIR_MODELS, filename)
            if not (
                os.path.isfile(model_path)
                and filename.startswith(f"scheme")
                and filename.endswith(".ckpt")
                and "_" in filename
            ):
                continue

            scheme = int(filename.split("_")[0][len("scheme") :])

            if "_strategy" in filename:
                strategy = filename.split("_")[1][len("strategy") :]
                strategy = strategy_from_str(strategy)
                strategy_kwargs = strategy.get_env_kwargs()
            else:
                strategy_kwargs = {}

            env._opponent_models.append(
                SpaceRAgent(
                    env.env_info,
                    agent_id=2,
                    interpolation_order=interpolation_order,
                    train=False,
                    scheme=scheme,
                    model_path=model_path,
                    **strategy_kwargs,
                )
            )
        # Counter that determines when to save a new opponent model
        env._save_opponent_model_timeout_counter = 0
        # Counter that determines the name of the model
        env._saved_opponent_model_counter = 0

    # Get path to the models (inefficient hack - get from config)
    config = config_dreamerv3(train=False, preset=CONFIG_PRESET)
    config = embodied.Flags(config).parse(argv=[])
    env._model_logdir = config.logdir

    # Set the agents
    env._agent_1 = agent_1
    env._agent_2 = np.random.choice(env._opponent_models)

    env.action_idx = (
        np.arange(env.base_env.action_shape[0][0]),
        np.arange(env.base_env.action_shape[1][0]),
    )

    # To make certain functions work (hack)
    env.scheme = env._agent_1.scheme
    env.n_stacked_obs_participant_ee_pos = env._agent_1.n_stacked_obs_participant_ee_pos
    env.n_stacked_obs_opponent_ee_pos = env._agent_1.n_stacked_obs_opponent_ee_pos
    env.n_stacked_obs_puck_pos = env._agent_1.n_stacked_obs_puck_pos
    if env.scheme == 7:
        env.n_stacked_obs_puck_rot = env._agent_1.n_stacked_obs_puck_rot

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
        # Configuration for a "large" machine
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            XLA_PYTHON_CLIENT_MEM_FRACTION
        )

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

        if RENDER:
            self.render()
        return obs, reward, done, info

    AirHockeyChallengeWrapper.step = new_step
    ## MONKEY PATCH: Make the agent transparent to the environment

    ## MONKEY PATCH: Make the agent transparent to the environment
    _original_reset = AirHockeyChallengeWrapper.reset

    def new_reset(self, state=None):
        # Regularly add a new opponent from a copy of the current model
        if DELAYED_SELF_PLAY:
            self._save_opponent_model_timeout_counter += 1
            if (
                self._save_opponent_model_timeout_counter
                >= SAVE_NEW_OPPONENT_EVERY_N_EPISODES
            ):
                self._save_opponent_model_timeout_counter = 0

                # If too many opponent models, pop a random model (except the first one)
                while len(self._opponent_models) >= MAX_N_MODELS:
                    self._opponent_models.pop(
                        np.random.randint(1, len(self._opponent_models))
                    )

                self._saved_opponent_model_counter += 1
                checkpoint_path = os.path.join(self._model_logdir, "checkpoint.ckpt")
                if not os.path.exists(checkpoint_path):
                    raise RuntimeError(
                        f"Could not find checkpoint file at {checkpoint_path}"
                    )
                save_model_path = os.path.join(
                    DIR_MODELS,
                    f"scheme{AGENT_SCHEME}_strategy{strategy_to_str(AGENT_STRATEGY)}_mk{self._saved_opponent_model_counter}.ckpt",
                )
                if not os.path.exists(save_model_path):
                    shutil.copyfile(checkpoint_path, save_model_path)
                self._opponent_models.append(
                    SpaceRAgent(
                        self.env_info,
                        agent_id=2,
                        interpolation_order=INTERPOLATION_ORDER,
                        train=False,
                        scheme=AGENT_SCHEME,
                        model_path=save_model_path,
                        **AGENT_STRATEGY.get_env_kwargs(),
                    ),
                )

        # Randomly select a new opponent to train against
        self._agent_2 = np.random.choice(self._opponent_models)

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
