import datetime
import os
from pathlib import Path

import yaml

LOGS_DIR = Path(__file__).parent.joinpath("logs").resolve()


def stamp_dir(directory: Path, timestamp_format: str = "%Y%m%d_%H%M%S") -> Path:
    return directory.joinpath(datetime.datetime.now().strftime(timestamp_format))


def new_logdir(
    env_id: str,
    workflow: str,
    root: Path = LOGS_DIR,
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> Path:
    return stamp_dir(
        root.joinpath(env_id.removeprefix("srb/")).joinpath(workflow),
        timestamp_format=timestamp_format,
    )


def last_logdir(
    env_id: str,
    workflow: str,
    root: Path = LOGS_DIR,
    modification_time: bool = False,
) -> Path:
    logdir_parent = root.joinpath(env_id.removeprefix("srb/")).joinpath(workflow)
    if not logdir_parent.is_dir():
        raise ValueError(
            f"Path {logdir_parent} is expected to be a directory with logdirs but it "
            + ("is a file" if logdir_parent.is_file() else "does not exist")
        )

    if last_logdir := last_dir(
        directory=logdir_parent, modification_time=modification_time
    ):
        return last_logdir
    else:
        raise FileNotFoundError(f"Path {logdir_parent} does not contain any logdirs")


def last_dir(directory: Path, modification_time: bool = False) -> Path | None:
    assert directory.is_dir()
    if dirs := sorted(
        filter(
            lambda p: p.is_dir(),
            (directory.joinpath(child) for child in os.listdir(directory)),
        ),
        key=os.path.getmtime if modification_time else None,
        reverse=True,
    ):
        return dirs[0]
    else:
        return None


def last_file(directory: Path, modification_time: bool = False) -> Path | None:
    assert directory.is_dir()
    if files := sorted(
        filter(
            lambda p: p.is_file(),
            (directory.joinpath(child) for child in os.listdir(directory)),
        ),
        key=os.path.getmtime if modification_time else None,
        reverse=True,
    ):
        return files[0]
    else:
        return None


def get_config(
    cli_args: dict,
    config_path: Path | None,
    logdir: Path | None,
    model_path: Path | None,
    continue_training: bool,
) -> dict:
    # Load default config
    if config_path is None:
        config_path = Path(__file__).parent.joinpath("hyperparams.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Update config with CLI arguments
    config.update(cli_args)

    # If continuing training, load config from logdir
    if continue_training:
        if logdir is None:
            raise ValueError("logdir must be provided when continue_training is True")
        config_path = logdir.joinpath("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {logdir}")
        with open(config_path, "r") as f:
            config.update(yaml.safe_load(f))

    # If model path is provided, load config from model's logdir
    if model_path is not None:
        model_logdir = model_path.parent
        config_path = model_logdir.joinpath("config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config.update(yaml.safe_load(f))

    return config
