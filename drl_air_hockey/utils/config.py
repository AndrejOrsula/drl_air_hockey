from os import environ, path
from typing import Any, Dict

from dreamerv3 import configs, embodied

from drl_air_hockey.utils.task import Task as AirHockeyTask
from drl_air_hockey.utils.tournament_agent_strategies import (
    strategy_from_str,
    strategy_to_str,
)

ENV = AirHockeyTask.from_str(environ.get("AIR_HOCKEY_ENV", default="tournament"))

AGENT_STRATEGY = strategy_from_str(
    environ.get("AIR_HOCKEY_AGENT_STRATEGY", default="aggressive")
)

RENDER: bool = False
INTERPOLATION_ORDER: int = -1

EPISODE_MAX_STEPS: int = 45000
MAX_TIME_UNTIL_PENALTY_S: int = 15.0

DIR_MODELS: str = path.join(
    path.abspath(path.dirname(path.dirname(__file__))),
    "agents",
    "models",
)


def config_dreamerv3(train: bool = False, preset: int = 2) -> Dict[str, Any]:
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
    elif preset == 2:
        config = config.update(
            {
                "logdir": path.join(
                    path.dirname(path.dirname(path.abspath(path.dirname(__file__)))),
                    "logdir_p2_" + ENV.to_str().lower().replace("7dof-", ""),
                ),
                "jax.platform": "cpu",
                "jax.precision": "float32",
                "jax.prealloc": True,
                "imag_horizon": 64,
                # encoder/decoder obs keys
                "encoder.mlp_keys": "vector",
                "decoder.mlp_keys": "vector",
                # encoder
                "encoder.mlp_layers": 2,
                "encoder.mlp_units": 128,
                # decoder
                "decoder.mlp_layers": 2,
                "decoder.mlp_units": 128,
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
                "critic.units": 256,
                # reward
                "reward_head.layers": 2,
                "reward_head.units": 256,
                # cont
                "cont_head.layers": 2,
                "cont_head.units": 256,
                # disag
                "disag_head.layers": 2,
                "disag_head.units": 256,
            }
        )
    elif preset == 3:
        config = config.update(
            {
                "logdir": path.join(
                    path.dirname(path.dirname(path.abspath(path.dirname(__file__)))),
                    "logdir_p3_" + ENV.to_str().lower().replace("7dof-", ""),
                ),
                "jax.platform": "cpu",
                "jax.precision": "float32",
                "jax.prealloc": True,
                "imag_horizon": 50,
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
                "rssm.deter": 512,
                "rssm.units": 512,
                "rssm.stoch": 32,
                "rssm.classes": 32,
                # actor
                "actor.layers": 2,
                "actor.units": 256,
                # critic
                "critic.layers": 2,
                "critic.units": 512,
                # reward
                "reward_head.layers": 2,
                "reward_head.units": 512,
                # cont
                "cont_head.layers": 2,
                "cont_head.units": 512,
                # disag
                "disag_head.layers": 2,
                "disag_head.units": 512,
            }
        )
    elif preset == 4:
        config = config.update(
            {
                "logdir": path.join(
                    path.dirname(path.dirname(path.abspath(path.dirname(__file__)))),
                    "logdir_p4_"
                    + ENV.to_str().lower().replace("7dof-", "")
                    + "_"
                    + strategy_to_str(AGENT_STRATEGY),
                ),
                "jax.platform": "cpu",
                "jax.precision": "float32",
                "jax.prealloc": True,
                "imag_horizon": 50,
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
                "rssm.deter": 512,
                "rssm.units": 512,
                "rssm.stoch": 32,
                "rssm.classes": 32,
                # actor
                "actor.layers": 2,
                "actor.units": 512,
                # critic
                "critic.layers": 2,
                "critic.units": 512,
                # reward
                "reward_head.layers": 2,
                "reward_head.units": 512,
                # cont
                "cont_head.layers": 2,
                "cont_head.units": 512,
                # disag
                "disag_head.layers": 2,
                "disag_head.units": 512,
            }
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if train:
        # Configuration for a "large" machine
        num_envs = 10
        config = config.update(
            {
                "jax.platform": "gpu",
                "envs.amount": num_envs,
                "run.actor_batch": num_envs,
                "replay_size": 1e7,
                "run.steps": 2e8,
                "run.log_every": 1024,
                "run.train_ratio": 384,
                "batch_size": 32,
                "batch_length": 64,
            }
        )

        # # Configuration for a "medium" machine
        # num_envs = 8
        # config = config.update(
        #     {
        #         "jax.platform": "gpu",
        #         "envs.amount": num_envs,
        #         "run.actor_batch": num_envs,
        #         "replay_size": 1e7,
        #         "run.steps": 2e8,
        #         "run.log_every": 1024,
        #         "run.train_ratio": 512,
        #         "batch_size": 16,
        #         "batch_length": 64,
        #     }
        # )

        # # Configuration for a "small" machine
        # num_envs = 5
        # config = config.update(
        #     {
        #         "jax.platform": "gpu",
        #         "envs.amount": num_envs,
        #         "run.actor_batch": num_envs,
        #         "replay_size": 1e7,
        #         "run.steps": 2e8,
        #         "run.log_every": 1024,
        #         "run.train_ratio": 384,
        #         "batch_size": 12,
        #         "batch_length": 64,
        #     }
        # )

    return config
