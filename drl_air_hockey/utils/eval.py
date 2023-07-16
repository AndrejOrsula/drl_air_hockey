import gym
import numpy as np
from dreamerv3.embodied.core.basics import convert


class PolicyEvalDriver:
    def __init__(self, policy, obs_key="vector", act_key="action"):
        self.policy = policy
        self._obs_key = obs_key
        self._act_key = act_key
        self.reset()

    def reset(self):
        self._state = None

    def infer(self, obs):
        obs = self._obs(obs)

        obs = {k: convert(v) for k, v in obs.items()}

        assert all(len(x) == 1 for x in obs.values()), obs

        act, self._state = self.policy(obs, self._state)
        act = {k: convert(v) for k, v in act.items()}

        return act

    def _obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        # Convert obs from (7,) to 1,7
        obs = np.expand_dims(obs, axis=0)

        obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: np.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=[0.0],
            is_first=[is_first],
            is_last=[is_last],
            is_terminal=[is_terminal],
        )
        return obs

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gym.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result
