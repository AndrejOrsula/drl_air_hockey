import functools
from typing import Generic, TypeVar, Union

import elements
import embodied
import gymnasium
import numpy

U = TypeVar("U")
V = TypeVar("V")


class EmbodiedEnvWrapper(embodied.Env, Generic[U, V]):
    def __init__(
        self,
        env: Union[str, gymnasium.Env[U, V]],
        obs_key="obs",
        act_key="action",
        **kwargs,
    ):
        if isinstance(env, str):
            self._env: gymnasium.Env[U, V] = gymnasium.make(
                env, render_mode="rgb_array", **kwargs
            )
        else:
            assert not kwargs, kwargs
            self._env = env
        self._obs_dict = hasattr(self._env.observation_space, "spaces")
        self._act_dict = hasattr(self._env.action_space, "spaces")
        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._info = None

    @property
    def env(self):
        return self._env

    @property
    def info(self):
        return self._info

    @functools.cached_property
    def obs_space(self):
        if self._obs_dict:
            spaces = self._flatten(self._env.observation_space.spaces)
        else:
            spaces = {self._obs_key: self._env.observation_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        return {
            **spaces,
            "reward": elements.Space(numpy.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }

    @functools.cached_property
    def act_space(self):
        if self._act_dict:
            spaces = self._flatten(self._env.action_space.spaces)
        else:
            spaces = {self._act_key: self._env.action_space}
        spaces = {k: self._convert(v) for k, v in spaces.items()}
        spaces["reset"] = elements.Space(bool)
        return spaces

    def step(self, action):
        if action["reset"] or self._done:
            self._done = False
            # obs, self._info = self._env.reset()
            obs = self._env.reset()
            return self._obs(obs, 0.0, is_first=True)
        if self._act_dict:
            action = self._unflatten(action)
        else:
            action = action[self._act_key]
        # obs, reward, terminated, truncated, self._info = self._env.step(action)
        obs, reward, self._done, self._info = self._env.step(action)
        return self._obs(
            obs,
            reward,
            is_last=bool(self._done),
            is_terminal=bool(self._done),
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        if not self._obs_dict:
            obs = {self._obs_key: obs}
        obs = self._flatten(obs)
        obs = {k: numpy.asarray(v) for k, v in obs.items()}
        obs.update(
            reward=numpy.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )
        return obs

    def render(self):
        image = self._env.render()
        assert image is not None
        return image

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass

    def _flatten(self, nest, prefix=None):
        result = {}
        for key, value in nest.items():
            key = prefix + "/" + key if prefix else key
            if isinstance(value, gymnasium.spaces.Dict):
                value = value.spaces
            if isinstance(value, dict):
                result.update(self._flatten(value, key))
            else:
                result[key] = value
        return result

    def _unflatten(self, flat):
        result = {}
        for key, value in flat.items():
            parts = key.split("/")
            node = result
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        return result

    def _convert(self, space):
        if hasattr(space, "n"):
            return elements.Space(numpy.int32, (), 0, space.n)
        return elements.Space(space.dtype, space.shape, space.low, space.high)
