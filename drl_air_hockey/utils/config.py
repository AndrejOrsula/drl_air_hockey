from os import cpu_count, environ, path
from typing import Any, Dict

from dreamerv3 import configs, embodied

from drl_air_hockey.utils.rewards import DefendReward, HitReward, PrepareReward
from drl_air_hockey.utils.task import Task as AirHockeyTask

ENV = AirHockeyTask.from_str(environ.get("AIR_HOCKEY_ENV", default="7dof-hit"))

REWARD_FUNCTION = None
if ENV == AirHockeyTask.R7_HIT:
    REWARD_FUNCTION = HitReward()
elif ENV == AirHockeyTask.R7_DEFEND:
    REWARD_FUNCTION = DefendReward()
elif ENV == AirHockeyTask.R7_PREPARE:
    REWARD_FUNCTION = PrepareReward()
else:
    raise ValueError(f"Unknown environment name: {ENV}")

RENDER: bool = False
EPISODE_MAX_STEPS: int = 1024
INTERPOLATION_ORDER: int = -1


def config_dreamerv3(train: bool = False, preset: int = 1) -> Dict[str, Any]:
    ## Create config
    config = embodied.Config(configs["defaults"])
    if preset == 1:
        config = config.update(
            {
                "logdir": path.join(
                    path.dirname(path.dirname(path.abspath(path.dirname(__file__)))),
                    "logdir_" + ENV.to_str().lower().replace("7dof-", ""),
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
                "decoder.mlp_layers": 2,
                # decoder
                "encoder.mlp_units": 192,
                "decoder.mlp_units": 192,
                # rssm
                "rssm.deter": 256,
                "rssm.units": 256,
                "rssm.stoch": 16,
                "rssm.classes": 16,
                # actor
                "actor.layers": 2,
                "actor.units": 256,
                # critic
                "critic.layers": 3,
                "critic.units": 512,
                # reward
                "reward_head.layers": 3,
                "reward_head.units": 512,
                # cont
                "cont_head.layers": 3,
                "cont_head.units": 512,
                # disag
                "disag_head.layers": 3,
                "disag_head.units": 512,
            }
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if train:
        num_envs = max(1, min(cpu_count() - 4, cpu_count() // 2))
        config = config.update(
            {
                "jax.platform": "gpu",
                "envs.amount": num_envs,
                "run.actor_batch": num_envs,
                "replay_size": 1e6,
                "run.steps": 5e7,
                "run.log_every": 1024,
                "run.train_ratio": 256,
                "batch_size": 2,
                "batch_length": 64,
            }
        )

    return config
