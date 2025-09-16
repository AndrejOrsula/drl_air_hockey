from typing import Any, Dict

import numpy as np


class InferenceDriver:
    """
    Drives a policy for single-step inference on an external environment.

    This driver is optimized for inference. It maintains the policy's
    recurrent state between calls but does not manage the environment itself.
    It expects to receive observations from an external source and returns
    the action and other policy outputs for that single step.
    """

    def __init__(self, policy, init_policy, **kwargs: Any):
        """
        Initializes the inference driver.

        Args:
            policy: The policy function to use for action selection.
            **kwargs: Additional keyword arguments to pass to the policy.
        """
        self.policy = policy
        self.init_policy = init_policy
        self.kwargs = kwargs
        self.carry = self.init_policy(1)
        self.reset()  # Initialize the state

    def reset(self):
        """
        Resets the agent's recurrent state.

        This should be called at the beginning of each new episode.
        """
        # The policy expects a batch size, which is 1 for inference.
        # We check for a wrapped policy, which is common with lambdas.
        self.carry = self.init_policy(1)

    def infer(self, obs: Any) -> Dict[str, np.ndarray]:
        """
        Performs a single inference step.

        Args:
            obs: An observation from the environment. Can be a raw np.ndarray
                 or a dictionary of arrays. If it is an array, it will be
                 automatically wrapped in a dictionary with the key 'obs'.

        Returns:
            A dictionary containing the action and other outputs from the policy,
            with the batch dimension removed.
        """
        # If the observation is not a dictionary, wrap it in one.
        # The default key 'obs' is standard for DreamerV3 single-observation envs.
        if not isinstance(obs, dict):
            obs = {"obs": obs}
            obs.update(
                reward=np.float32(0.0),
                is_first=False,
                is_last=False,
                is_terminal=False,
            )

        # The policy expects batched inputs, so we add a leading dimension to each value.
        obs_for_policy = {k: np.array([v]) for k, v in obs.items()}

        # Get the next action and updated recurrent state from the policy.
        new_carry, acts, outs = self.policy(self.carry, obs_for_policy, **self.kwargs)

        # Update the driver's internal state for the next call.
        self.carry = new_carry

        # Combine actions and other policy outputs.
        result = {**acts, **outs}

        # Unpack the batch dimension from the results before returning.
        result_unpacked = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray) and v.shape and v.shape[0] == 1:
                result_unpacked[k] = v[0]
            else:
                # Keep as-is if not batched (e.g., scalars from 'outs')
                result_unpacked[k] = v

        return result_unpacked
