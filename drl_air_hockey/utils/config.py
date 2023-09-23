from os import cpu_count, environ, path
from typing import Any, Dict

from dreamerv3 import configs, embodied

from drl_air_hockey.utils.rewards import (
    DefendReward,
    HitReward,
    PrepareReward,
    TournamentReward,
)
from drl_air_hockey.utils.task import Task as AirHockeyTask

ENV = AirHockeyTask.from_str(environ.get("AIR_HOCKEY_ENV", default="tournament"))

REWARD_FUNCTION = None
if ENV == AirHockeyTask.R7_TOURNAMENT:
    REWARD_FUNCTION = TournamentReward()
elif ENV == AirHockeyTask.R7_HIT:
    REWARD_FUNCTION = HitReward()
elif ENV == AirHockeyTask.R7_DEFEND:
    REWARD_FUNCTION = DefendReward()
elif ENV == AirHockeyTask.R7_PREPARE:
    REWARD_FUNCTION = PrepareReward()
else:
    raise ValueError(f"Unknown environment name: {ENV}")

RENDER: bool = False
EPISODE_MAX_STEPS: int = 500
INTERPOLATION_ORDER: int = -1


def config_dreamerv3(train: bool = False, preset: int = 1) -> Dict[str, Any]:
    ## Create config
    config = embodied.Config(configs["defaults"])
    if preset == 1:
        config = config.update(
            {
                "logdir": path.join(
                    path.dirname(path.dirname(path.abspath(path.dirname(__file__)))),
                    "logdir_p1_" + ENV.to_str().lower().replace("7dof-", ""),
                ),
                "jax.platform": "cpu",
                "jax.precision": "float32",
                "jax.prealloc": True,
                "imag_horizon": 25,
                # encoder/decoder obs keys
                "encoder.mlp_keys": "vector",
                "decoder.mlp_keys": "vector",
                # encoder
                "encoder.mlp_layers": 2,
                "encoder.mlp_units": 256,
                # decoder
                "decoder.mlp_layers": 2,
                "decoder.mlp_units": 256,
                # rssm
                "rssm.deter": 256,
                "rssm.units": 256,
                "rssm.stoch": 32,
                "rssm.classes": 32,
                # actor
                "actor.layers": 2,
                "actor.units": 128,
                # critic
                "critic.layers": 2,
                "critic.units": 128,
                # reward
                "reward_head.layers": 2,
                "reward_head.units": 128,
                # cont
                "cont_head.layers": 2,
                "cont_head.units": 128,
                # disag
                "disag_head.layers": 2,
                "disag_head.units": 128,
            }
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if train:
        # Configuration for a "large" machine
        num_envs = 8
        config = config.update(
            {
                "jax.platform": "gpu",
                "envs.amount": num_envs,
                "run.actor_batch": num_envs,
                "replay_size": 2e6,
                "run.steps": 5e7,
                "run.log_every": 1024,
                "run.train_ratio": 256,
                "batch_size": 96,
                "batch_length": 64,
            }
        )

        # # Configuration for a "small" machine
        # num_envs = 4
        # config = config.update(
        #     {
        #         "jax.platform": "gpu",
        #         "envs.amount": num_envs,
        #         "run.actor_batch": num_envs,
        #         "replay_size": 2e6,
        #         "run.steps": 5e7,
        #         "run.log_every": 1024,
        #         "run.train_ratio": 256,
        #         "batch_size": 64,
        #         "batch_length": 64,
        #     }
        # )

    return config
