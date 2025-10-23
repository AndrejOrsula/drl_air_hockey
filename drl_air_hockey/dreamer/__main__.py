#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import argparse
import os
import random
import shutil
import sys
from enum import Enum, auto
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from typing_extensions import Self

from drl_air_hockey.agents import agent_from_name
from drl_air_hockey.dreamer.rewards import reward_from_name
from drl_air_hockey.dreamer.utils import LOGS_DIR


def main():
    def impl(subcommand: Literal["agent"], **kwargs):
        match subcommand:
            case "agent":
                run_agent_with_env(**kwargs)

    impl(**vars(parse_cli_args()))


### Agent ###
def run_agent_with_env(
    agent_subcommand: Literal["zero", "rand", "train", "eval"],
    env_id: str,
    sim: str,
    agent: str,
    interpolation_order: str,
    render: bool,
    logdir_path: str,
    **kwargs,
):
    from drl_air_hockey.dreamer.utils import last_logdir, new_logdir

    # Get the log directory based on the workflow
    workflow = kwargs.get("algo") or agent_subcommand
    logdir_root = Path(logdir_path).resolve()
    if model := kwargs.get("model"):
        model = Path(model).resolve()
        assert model.exists(), f"Model path does not exist: {model}"
        logdir = model.parent
        while not (
            logdir.parent.name == workflow and logdir.parent.parent.name == env_id
        ):
            _new_parent = logdir.parent
            if logdir == _new_parent:
                logdir = new_logdir(env_id=env_id, workflow=workflow, root=logdir_root)
                model_symlink_path = logdir.joinpath(model.name)
                model_symlink_path.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(model, model_symlink_path)
                model = model_symlink_path
                break
            logdir = _new_parent
        kwargs["model"] = model
    elif (agent_subcommand == "train" and kwargs["continue_training"]) or (
        agent_subcommand == "eval" and kwargs["algo"]
    ):
        logdir = last_logdir(env_id=env_id, workflow=workflow, root=logdir_root)
    else:
        logdir = new_logdir(env_id=env_id, workflow=workflow, root=logdir_root)

    # Create the environment and initialize it
    env = _make_env(
        env_id=env_id,
        sim=sim,
        agent=agent,
        interpolation_order=None
        if interpolation_order == "none"
        else int(interpolation_order),
        render=render,
        logdir=logdir,
        self_play=agent_subcommand == "train" and kwargs.get("self_play", True),
        self_play_save_model_every_n_episodes=kwargs.get(
            "self_play_save_model_every_n_episodes"
        ),
        self_play_max_opponent_models=kwargs.get("self_play_max_opponent_models"),
        self_play_opponent_models_path=kwargs.get("self_play_opponent_models_path"),
    )

    # Run the implementation
    def agent_impl(**kwargs):
        kwargs.update({"env_id": env_id})

        match agent_subcommand:
            case "zero":
                zero_agent(**kwargs)
            case "rand":
                random_agent(**kwargs)
            case "step":
                step_agent(**kwargs)
            case "limit":
                limit_agent(**kwargs)
            case "train":
                train_agent(**kwargs)
            case "eval":
                eval_agent(**kwargs)

    agent_impl(env=env, logdir=logdir, **kwargs)

    # Close the environment
    env.close()


def random_agent(
    env,
    **kwargs,
):
    env = env()
    env.reset()

    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print("\n---")
        print(f"ACTION\n{action}")
        print(f"OBS\n{observation}")
        print(f"REWARD: {reward}")
        print(f"DONE: {done}")
        print(f"INFO:\n{info}")
        if done:
            env.reset()
            continue


def step_agent(
    env,
    **kwargs,
):
    env = env()
    env.reset()

    t = 0
    while True:
        if t % 50 == 0:
            action = env.action_space.sample()
        t += 1
        observation, reward, done, info = env.step(action)
        print("\n---")
        print(f"ACTION\n{action}")
        print(f"OBS\n{observation}")
        print(f"REWARD: {reward}")
        print(f"DONE: {done}")
        print(f"INFO:\n{info}")
        if done:
            env.reset()
            continue


def limit_agent(
    env,
    **kwargs,
):
    env = env()
    env.reset()

    t = 0
    s = 0
    while True:
        if t % 100 == 0:
            action = env.action_space.sample()
            if s % 8 == 0:
                action[0] = -1
                action[1] = 0
            elif s % 8 == 1:
                action[0] = 1
                action[1] = 0
            elif s % 8 == 2:
                action[0] = 0
                action[1] = 1
            elif s % 8 == 3:
                action[0] = 0
                action[1] = -1
            elif s % 8 == 4:
                action[0] = -1
                action[1] = -1
            elif s % 8 == 5:
                action[0] = 1
                action[1] = -1
            elif s % 8 == 6:
                action[0] = -1
                action[1] = 1
            elif s % 8 == 7:
                action[0] = 1
                action[1] = 1
            s += 1
        t += 1
        observation, reward, done, info = env.step(action)
        print("\n---")
        print(f"ACTION\n{action}")
        print(f"OBS\n{observation}")
        print(f"REWARD: {reward}")
        print(f"DONE: {done}")
        print(f"INFO:\n{info}")
        if done:
            env.reset()
            continue


def zero_agent(
    env,
    **kwargs,
):
    env = env()
    env.reset()

    import numpy

    action = numpy.zeros(env.action_space.shape)

    while True:
        observation, reward, done, info = env.step(action)
        print("\n---")
        print(f"ACTION\n{action}")
        print(f"OBS\n{observation}")
        print(f"REWARD: {reward}")
        print(f"DONE: {done}")
        print(f"INFO:\n{info}")
        if done:
            env.reset()
            continue


def train_agent(algo: str, **kwargs):
    WORKFLOW: str = "train"

    match algo:
        case "dreamer":
            import drl_air_hockey.dreamer.main as dreamer

            dreamer.run(workflow=WORKFLOW, **kwargs)


def eval_agent(algo: str, **kwargs):
    WORKFLOW: str = "eval"

    match algo:
        case "dreamer":
            import drl_air_hockey.dreamer.main as dreamer

            dreamer.run(workflow=WORKFLOW, **kwargs)


def _make_env(
    env_id: str,
    sim: str,
    agent: str,
    interpolation_order: Optional[int],
    render: bool,
    logdir: Path,
    self_play: bool,
    self_play_save_model_every_n_episodes: Optional[int],
    self_play_max_opponent_models: Optional[int],
    self_play_opponent_models_path: Optional[str],
) -> Callable:
    from baseline.baseline_agent.baseline_agent import (
        build_agent as build_baseline_agent,
    )

    if sim == "mujoco":

        def __impl_mujoco():
            env = AirHockeyChallengeWrapper(
                env=env_id,
                custom_reward_function=reward_from_name(env_id),
                interpolation_order=interpolation_order,  # type: ignore
            )

            agent_1 = agent_from_name(agent)(
                env_info=env.env_info,
                agent_id=1,
                interpolation_order=interpolation_order,  # type: ignore
                train=True,
            )
            env._agent_1 = agent_1
            env.action_space = env._agent_1.action_space
            env.observation_space = env._agent_1.observation_space

            env._opponent_models = [
                build_baseline_agent(env_info=env.env_info, agent_id=2),
            ]
            env._num_static_opponents = len(env._opponent_models)
            env._self_play_max_opponent_models = (
                self_play_max_opponent_models or 1
            ) + env._num_static_opponents

            if self_play:
                env._opponents_dir = logdir.joinpath("opponents")

                # Load default opponent models from a specified path
                if (
                    self_play_opponent_models_path
                    and Path(self_play_opponent_models_path).exists()
                    and Path(self_play_opponent_models_path).is_dir()
                ):
                    # Look for directories that are potential models
                    opponent_paths = [
                        p
                        for p in Path(self_play_opponent_models_path).iterdir()
                        if p.is_dir()
                    ]
                    random.shuffle(opponent_paths)
                    for model_dir in opponent_paths:
                        if (
                            len(env._opponent_models)
                            >= env._self_play_max_opponent_models
                        ):
                            print("Reached max number of opponent models.")
                            break
                        print(f"Loading default opponent from: {model_dir}")
                        opponent_agent = agent_from_name(f"{agent}_inference")(
                            env_info=env.env_info,
                            agent_id=2,
                            interpolation_order=interpolation_order,
                            model_path=model_dir,
                            cpu=False,
                        )
                        env._opponent_models.append(opponent_agent)

                elif env._opponents_dir.exists():
                    print(f"Loading default opponent from: {env._opponents_dir}")
                    opponent_paths = [
                        p for p in env._opponents_dir.iterdir() if p.is_dir()
                    ]
                    random.shuffle(opponent_paths)
                    for model_dir in env._opponents_dir.iterdir():
                        if (
                            len(env._opponent_models)
                            >= env._self_play_max_opponent_models
                        ):
                            print("Reached max number of opponent models.")
                            break
                        print(f"Loading default opponent from: {model_dir}")
                        opponent_agent = agent_from_name(f"{agent}_inference")(
                            env_info=env.env_info,
                            agent_id=2,
                            interpolation_order=interpolation_order,
                            model_path=model_dir,
                            cpu=False,
                        )
                        env._opponent_models.append(opponent_agent)
                else:
                    print("No default opponent models path provided.")

                env._opponents_dir.mkdir(parents=True, exist_ok=True)
                env._save_opponent_model_timeout_counter = 0

            env._agent_2 = numpy.random.choice(env._opponent_models)

            env.action_idx = (
                numpy.arange(env.base_env.action_shape[0][0]),
                numpy.arange(env.base_env.action_shape[1][0]),
            )

            _original_step = AirHockeyChallengeWrapper.step

            def _new_step_mujoco(self, action):
                action_1 = self._agent_1.process_raw_act(action=action)
                action_2 = self._agent_2.draw_action(self._previous_obs2)
                combined_action = (
                    action_1[self.action_idx[0]],
                    action_2[self.action_idx[1]],
                )
                obs, reward, done, info = _original_step(self, combined_action)
                obs1, obs2 = numpy.split(obs, 2)
                self._previous_obs2 = obs2
                obs = self._agent_1.process_raw_obs(obs=obs1)
                if render:
                    self.render()
                return obs, reward, done, info

            AirHockeyChallengeWrapper.step = _new_step_mujoco

            _original_reset = AirHockeyChallengeWrapper.reset

            def _new_reset_mujoco(self, state=None):
                if self_play and self_play_save_model_every_n_episodes is not None:
                    self._save_opponent_model_timeout_counter += 1
                    if (
                        self._save_opponent_model_timeout_counter
                        >= self_play_save_model_every_n_episodes
                    ):
                        self._save_opponent_model_timeout_counter = 0
                        checkpoint_dir = logdir.joinpath("ckpt")
                        if (
                            checkpoint_dir.exists()
                            and checkpoint_dir.is_dir()
                            and checkpoint_dir.joinpath("latest").exists()
                        ):
                            latest_model_name = (
                                checkpoint_dir.joinpath("latest").read_text().strip()
                            )
                            latest_model_path = checkpoint_dir.joinpath(
                                latest_model_name
                            )
                            if not latest_model_path.exists():
                                print(
                                    f"Warning: Could not find latest checkpoint file at {latest_model_path}"
                                )
                                return

                            save_model_path = (
                                self._opponents_dir / latest_model_path.name
                            )
                            if not save_model_path.exists():
                                shutil.copytree(latest_model_path, save_model_path)
                                print(
                                    f"Saved opponent model to {save_model_path} from {latest_model_path}"
                                )

                                if (
                                    len(self._opponent_models)
                                    >= env._self_play_max_opponent_models
                                ):
                                    self._opponent_models.pop(
                                        numpy.random.randint(
                                            env._num_static_opponents,
                                            len(self._opponent_models),
                                        )
                                    )

                                opponent_agent = agent_from_name(f"{agent}_inference")(
                                    env_info=self.env_info,
                                    agent_id=2,
                                    interpolation_order=interpolation_order,
                                    model_path=save_model_path,
                                    cpu=False,
                                )
                                self._opponent_models.append(opponent_agent)

                            elif (
                                len(self._opponent_models)
                                < env._self_play_max_opponent_models
                            ):
                                opponent_agent = agent_from_name(f"{agent}_inference")(
                                    env_info=self.env_info,
                                    agent_id=2,
                                    interpolation_order=interpolation_order,
                                    model_path=save_model_path,
                                    cpu=False,
                                )
                                self._opponent_models.append(opponent_agent)

                        else:
                            raise FileNotFoundError(
                                f"Checkpoint directory not found: {checkpoint_dir}"
                            )

                if len(self._opponent_models) > 1:
                    self._agent_2 = numpy.random.choice(self._opponent_models)
                self._agent_1.reset()
                self._agent_2.reset()
                obs = _original_reset(self, state)
                obs1, obs2 = numpy.split(obs, 2)
                self._previous_obs2 = obs2
                obs = self._agent_1.process_raw_obs(obs=obs1)
                return obs

            AirHockeyChallengeWrapper.reset = _new_reset_mujoco

            return env

        return __impl_mujoco

    else:
        raise ValueError(f"Unknown simulator: {sim}")


### CLI ###
def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments for this script.
    """

    parser = argparse.ArgumentParser(
        description="Space Robotics Bench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        title="Subcommands",
        dest="subcommand",
        required=True,
    )

    ## Agent subcommand
    agent_parser = subparsers.add_parser(
        "agent",
        help="Agent subcommands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    agent_subparsers = agent_parser.add_subparsers(
        title="Agent subcommands",
        dest="agent_subcommand",
        required=True,
    )
    zero_agent_parser = agent_subparsers.add_parser(
        "zero",
        help="Agent with zero-valued actions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rand_agent_parser = agent_subparsers.add_parser(
        "rand",
        help="Agent with random actions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    step_agent_parser = agent_subparsers.add_parser(
        "step",
        help="Agent with randomly stepped actions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    limit_agent_parser = agent_subparsers.add_parser(
        "limit",
        help="Agent with actions at limits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_agent_parser = agent_subparsers.add_parser(
        "train",
        help="Train agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    eval_agent_parser = agent_subparsers.add_parser(
        "eval",
        help="Evaluate agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ## Environment args
    for _agent_parser in (
        zero_agent_parser,
        rand_agent_parser,
        step_agent_parser,
        limit_agent_parser,
        train_agent_parser,
        eval_agent_parser,
    ):
        environment_group = _agent_parser.add_argument_group("Environment")
        environment_group.add_argument(
            "-e",
            "--env",
            dest="env_id",
            help="Name of the environment to select",
            type=str,
            # choices=("tournament", "hit", "defend", "prepare"),
            choices=("tournament",),
            default="tournament",
        )
        environment_group.add_argument(
            "--sim",
            help="Type of simulator to use",
            type=str,
            choices=("mujoco", "mjx"),
            default="mujoco",
        )
        environment_group.add_argument(
            "--agent",
            help="Type of agent to use",
            type=str,
            choices=("spacer"),
            default="spacer",
        )
        environment_group.add_argument(
            "--interpolation_order",
            help="Interpolation order for the environment",
            type=str,
            choices=("3", "1", "2", "4", "5", "-1", "none"),
            default="-1",
        )
        environment_group.add_argument(
            "--headless",
            dest="render",
            help="Run in headless mode without rendering",
            action="store_false",
            default=True,
        )

        logging_group = _agent_parser.add_argument_group("Logging")
        logging_group.add_argument(
            "--logdir",
            "--logs",
            dest="logdir_path",
            help="Path to the root directory for storing logs",
            type=str,
            default=LOGS_DIR,
        )

    ## Algorithm args
    _algo_choices = sorted(map(str, SupportedAlgo))
    for _agent_parser in (
        train_agent_parser,
        eval_agent_parser,
    ):
        algorithm_group = _agent_parser.add_argument_group("Algorithm")
        algorithm_group.add_argument(
            "--algo",
            help="Name of the algorithm",
            type=str,
            choices=_algo_choices,
            default=str(SupportedAlgo.DREAMER),
            # required=True,
        )
        algorithm_group.add_argument(
            "--algo_cfg",
            type=str,
            help="Path to the YAML configuration file of the algorithm",
            default=Path(__file__).parent.joinpath("hyperparams.yaml").resolve(),
        )

        if _agent_parser != train_agent_parser:
            algorithm_group.add_argument(
                "--model",
                type=str,
                help="Path to the model checkpoint",
            )

    ## Train args
    train_group = train_agent_parser.add_argument_group("Train")
    mutex_group = train_group.add_mutually_exclusive_group()
    mutex_group.add_argument(
        "--continue_training",
        "--continue",
        help="Continue training the model from the last checkpoint",
        action="store_true",
        default=False,
    )
    mutex_group.add_argument(
        "--model",
        help="Continue training the model from the specified checkpoint",
        type=str,
    )
    train_group.add_argument(
        "--no_self_play",
        dest="self_play",
        help="Disable self-play",
        action="store_false",
        default=True,
    )
    train_group.add_argument(
        "--self_play_save_model_every_n_episodes",
        help="Save opponent model every N episodes",
        type=int,
        default=50,
    )
    train_group.add_argument(
        "--self_play_max_opponent_models",
        help="Maximum number of opponent models to keep",
        type=int,
        default=2,
    )
    train_group.add_argument(
        "--self_play_opponent_models_path",
        help="Path to default opponent models to load at the start",
        type=str,
        default=Path(__file__)
        .parent.parent.joinpath("agents")
        .joinpath("models")
        .resolve()
        .as_posix(),
    )

    # Trigger argcomplete
    if find_spec("argcomplete"):
        import argcomplete

        argcomplete.autocomplete(parser)

    # Allow separation of arguments meant for other purposes
    if "--" in sys.argv:
        forwarded_args = sys.argv[(sys.argv.index("--") + 1) :]
        sys.argv = sys.argv[: sys.argv.index("--")]
    else:
        forwarded_args = []

    # Parse arguments
    args, other_args = parser.parse_known_args()

    # Add forwarded arguments
    args.forwarded_args = forwarded_args

    # Detect any unsupported arguments
    _unsupported_args = [
        arg for arg in other_args if arg.startswith("-") or "=" not in arg
    ]

    # Forward other arguments to hydra
    sys.argv = [sys.argv[0], *other_args]

    return args


class SupportedAlgo(str, Enum):
    # DreamerV3
    DREAMER = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @classmethod
    def from_str(cls, string: str) -> Self | None:
        return next(
            (variant for variant in cls if string.upper() == variant.name), None
        )


if __name__ == "__main__":
    main()
