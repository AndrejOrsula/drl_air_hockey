import functools

import elements
import embodied
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import PipelineEnv
from brax.io import image as brax_image


class EmbodiedBraxEnvWrapper(embodied.Env):
    """
    A wrapper to make Brax/MJX environments compatible with the DreamerV3 agent.
    """

    def __init__(
        self,
        env: PipelineEnv,
        obs_key="obs",
        act_key="action",
        seed=0,
        backend: str = None,
    ):
        """
        Initializes the wrapper.

        Args:
            env: The Brax/MJX environment to wrap.
            obs_key: The key for the observation in the observation dictionary.
            act_key: The key for the action in the action dictionary.
            seed: The random seed for JAX.
            backend: The JAX backend to use for JIT compilation.
        """
        self._env = env
        if not hasattr(self._env, "batch_size"):
            raise ValueError("The underlying Brax environment must be batched.")

        # DreamerV3's embodied.Env is not designed for batched envs.
        # This wrapper will interact with the first environment of the batch.
        if self._env.batch_size > 1:
            print(
                "Warning: Brax environment has batch size > 1. "
                "This wrapper will only interact with the first environment in the batch."
            )

        self._obs_key = obs_key
        self._act_key = act_key
        self._done = True
        self._key = jax.random.PRNGKey(seed)
        self._state = None

        # JIT compile the environment's reset and step functions for performance.
        self._reset = jax.jit(self._env.reset, backend=backend)
        self._step = jax.jit(self._env.step, backend=backend)

    @property
    def env(self):
        return self._env

    @functools.cached_property
    def obs_space(self):
        # Define the observation space based on the underlying Brax environment.
        # This includes the main observation, plus metadata required by DreamerV3.
        obs_space_shape = (self._env.observation_size,)
        space = {
            self._obs_key: elements.Space(np.float32, obs_space_shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
        }
        return space

    @functools.cached_property
    def act_space(self):
        # Define the action space, including the 'reset' action.
        ctrl_range = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        space = {
            self._act_key: elements.Space(
                np.float32, (self._env.action_size,), ctrl_range[:, 0], ctrl_range[:, 1]
            ),
            "reset": elements.Space(bool),
        }
        return space

    def step(self, action):
        if action["reset"] or self._done:
            self._key, reset_key = jax.random.split(self._key)
            self._state = self._reset(reset_key)
            self._done = False
            return self._obs(self._state.obs, reward=0.0, is_first=True)

        # Add a batch dimension to the action for the Brax environment.
        # This assumes the action from the agent is for a single environment.
        batched_action = jnp.expand_dims(action[self._act_key], axis=0)

        # Take a step in the environment.
        self._state = self._step(self._state, batched_action)

        # Remove the batch dimension from the results.
        obs = self._state.obs[0]
        reward = float(self._state.reward[0])
        self._done = bool(self._state.done[0])

        return self._obs(
            obs,
            reward,
            is_last=self._done,
            is_terminal=self._done,
        )

    def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
        # Format the observation dictionary as expected by DreamerV3.
        return {
            self._obs_key: np.asarray(obs),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def render(self):
        """Renders the environment's current state."""
        if self._state is None:
            raise RuntimeError("Must call reset or step before rendering.")
        # Brax rendering expects a batch of pipeline_states.
        return brax_image.render_array(
            self._env.sys, [self._state.pipeline_state], 256, 256
        )[0]

    def close(self):
        # Brax environments do not require explicit closing.
        pass
