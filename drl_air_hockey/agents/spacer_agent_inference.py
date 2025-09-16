from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import elements
import numpy as np
import yaml
from dreamerv3 import agent as dreamer_agent

from drl_air_hockey.agents.spacer_agent import SpaceRAgent
from drl_air_hockey.dreamer.inference_driver import InferenceDriver
from drl_air_hockey.dreamer.wrapper import EmbodiedEnvWrapper

MODELS_DIR = Path(__file__).parent.joinpath("models").resolve()
DEFAULT_CONFIG = MODELS_DIR.joinpath("config.yaml").resolve()


class SpaceRAgentInference(SpaceRAgent):
    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        model_path: Optional[str] = MODELS_DIR.joinpath(
            "20250914T215713F016471"
        ).as_posix(),
        cpu: bool = True,
        **kwargs,
    ):
        super().__init__(env_info=env_info, agent_id=agent_id, train=False, **kwargs)

        if model_path is None:
            raise ValueError("model_path must be provided for inference")
        self.model_path = Path(model_path)

        config_path = self.model_path.joinpath("config.yaml")
        if not config_path.exists():
            config_path = self.model_path.parent.joinpath("config.yaml")
            if not config_path.exists():
                config_path = DEFAULT_CONFIG
        config = yaml.safe_load(config_path.read_text())
        config = elements.Flags(config).parse(argv=[])
        config = config.update(
            {
                "logdir": "/dev/null",
                "run.log_every": 100000000,
                "run.report_every": 100000000,
                "run.save_every": 100000000,
            }
        )
        if cpu:
            config = config.update(
                {
                    "jax.platform": "cpu",
                    "jax.compute_dtype": "float32",
                }
            )

        # Setup agent
        self.as_env = EmbodiedEnvWrapper(self)
        self.agent = dreamer_agent.Agent(
            self.as_env.obs_space,
            {"action": elements.Space(np.float32, self.action_space.shape, -1.0, 1.0)},
            elements.Config(
                **config.agent,
                logdir=config.logdir,
                seed=config.seed,
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=config.replica,
                replicas=config.replicas,
            ),
        )

        # Load checkpoint
        checkpoint = elements.Checkpoint()
        checkpoint.agent = self.agent
        checkpoint.load(self.model_path, keys=["agent"])

        # Setup agent driver
        policy = lambda *args: self.agent.policy(*args, mode="eval")
        self.policy_driver = InferenceDriver(
            policy=policy, init_policy=self.agent.init_policy
        )

        self.initialize_inference()
        self.reset()

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        return self.infer_action(self.process_raw_obs(obs))

    def infer_action(self, obs: np.ndarray) -> np.ndarray:
        action = self.policy_driver.infer(obs)["action"].squeeze().clip(-1.0, 1.0)
        return self.process_raw_act(action)

    def initialize_inference(self):
        self.policy_driver.infer(self.observation_space.sample())
        self.policy_driver.reset()

    def reset(self):
        super().reset()
        if hasattr(self, "policy_driver"):
            self.policy_driver.reset()
