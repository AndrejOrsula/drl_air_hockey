from collections import deque
from os import nice, path
from typing import Any, Dict, Optional

import dreamerv3
import gym
import numpy as np
from air_hockey_challenge.constraints import ConstraintList
from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
)
from dreamerv3 import embodied
from mushroom_rl.core.environment import MDPInfo

from drl_air_hockey.utils.config import (
    DIR_MODELS,
    INTERPOLATION_ORDER,
    MAX_TIME_UNTIL_PENALTY_S,
    config_dreamerv3,
)
from drl_air_hockey.utils.env_wrapper import EmbodiedChallengeWrapper
from drl_air_hockey.utils.eval import PolicyEvalDriver
from drl_air_hockey.utils.task import Task as AirHockeyTask

## Required by async inference
# import os
# import time
# from sys import getswitchinterval as gsi
# from sys import setswitchinterval as ssi
# from threading import Condition, Lock, Thread


class SpaceRAgent(AgentBase):
    # # Whether to use asynchronous inference
    # ASYNC_INFERENCE: bool = False
    # # 20 ms (2 ms reserved for extra processing)
    # STEP_TIME_LIMIT: float = 0.02 - 0.002

    # Default models to use for inference if not specified
    DEFAULT_INFERENCE_MODEL: Dict[AirHockeyTask, str] = {
        AirHockeyTask.R7_HIT: path.join(DIR_MODELS, "hit.ckpt"),
        AirHockeyTask.R7_DEFEND: path.join(DIR_MODELS, "defend.ckpt"),
        AirHockeyTask.R7_PREPARE: path.join(DIR_MODELS, "prepare.ckpt"),
        AirHockeyTask.R7_TOURNAMENT: path.join(DIR_MODELS, "tournament.ckpt"),
    }

    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        interpolation_order: Optional[int] = INTERPOLATION_ORDER,
        # Whether to train or evaluate (inference)
        train: bool = False,
        # Path to the model to load for inference
        load_model_path: Optional[str] = None,
        # Observation scheme used by the agent
        scheme: int = 6,
        # Velocity constraints (0.5 is about safe)
        vel_constraints_scaling_factor: float = 0.65,
        # # Whether to filter actions and by how much
        # filter_actions_enabled: bool = False,
        # filter_actions_coefficient: float = 0.05,
        # Extra offsets for operating area of the agent (in meters)
        operating_area_offset_from_centre: float = 0.17,
        operating_area_offset_from_table: float = 0.02,
        operating_area_offset_from_goal: float = 0.01,
        # Strictness of the Z position (positive only, lower is more strict)
        z_position_control_tolerance: float = 0.35,
        # Noise to apply to the observation of opponent's end-effector position
        noise_obs_opponent_ee_pos_std: float = 0.025,
        **kwargs,
    ):
        try:
            nice(100)
        except Exception:
            pass

        ## Chain up the parent implementation
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)

        ## Get information about the agent
        self.agent_id = agent_id
        self.interpolation_order = interpolation_order
        self.evaluate = not train

        self.scheme = scheme
        self.vel_constraints_scaling_factor = vel_constraints_scaling_factor
        # self.filter_actions_enabled = filter_actions_enabled
        # self.filter_actions_coefficient = filter_actions_coefficient
        self.operating_area_offset_from_centre = operating_area_offset_from_centre
        self.operating_area_offset_from_table = operating_area_offset_from_table
        self.operating_area_offset_from_goal = operating_area_offset_from_goal
        self.z_position_control_tolerance = z_position_control_tolerance
        self.noise_obs_opponent_ee_pos_std = noise_obs_opponent_ee_pos_std

        ## Extract information about the environment and write it to members
        self.extract_env_info()

        ## Determine the scheme (affects mostly just the observation vector)
        if self.scheme == 1:
            raise ValueError("Scheme 1 is no longer supported")
        elif self.scheme == 2:
            self.penalty_timer = 0.0
            self.n_stacked_obs_participant_ee_pos = 1
            self.stacked_obs_participant_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_participant_ee_pos
            )
            self.n_stacked_obs_opponent_ee_pos = 1
            self.stacked_obs_opponent_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_opponent_ee_pos
            )
            self.n_stacked_obs_puck_pos = 1
            self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs_puck_pos)
        elif self.scheme == 3:
            # Differences from 2:
            #  - Longer observation history
            #  - Penalty side is included in the observation
            self.penalty_side = None
            self.penalty_timer = 0.0
            self.n_stacked_obs_participant_ee_pos = 3
            self.stacked_obs_participant_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_participant_ee_pos
            )
            self.n_stacked_obs_opponent_ee_pos = 5
            self.stacked_obs_opponent_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_opponent_ee_pos
            )
            self.n_stacked_obs_puck_pos = 9
            self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs_puck_pos)
        elif self.scheme == 4:
            # Same as 3 but different length of history
            self.penalty_side = None
            self.penalty_timer = 0.0
            self.n_stacked_obs_participant_ee_pos = 2
            self.stacked_obs_participant_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_participant_ee_pos
            )
            self.n_stacked_obs_opponent_ee_pos = 2
            self.stacked_obs_opponent_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_opponent_ee_pos
            )
            self.n_stacked_obs_puck_pos = 2
            self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs_puck_pos)
        elif self.scheme == 5:
            # Same as 3/4 but different length of history
            self.penalty_side = None
            self.penalty_timer = 0.0
            self.n_stacked_obs_participant_ee_pos = 4
            self.stacked_obs_participant_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_participant_ee_pos
            )
            self.n_stacked_obs_opponent_ee_pos = 4
            self.stacked_obs_opponent_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_opponent_ee_pos
            )
            self.n_stacked_obs_puck_pos = 4
            self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs_puck_pos)
        elif self.scheme == 6:
            # Same as 5 but with joint positions as part of the observation
            self.penalty_side = None
            self.penalty_timer = 0.0
            self.n_stacked_obs_participant_ee_pos = 4
            self.stacked_obs_participant_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_participant_ee_pos
            )
            self.n_stacked_obs_opponent_ee_pos = 4
            self.stacked_obs_opponent_ee_pos = deque(
                [], maxlen=self.n_stacked_obs_opponent_ee_pos
            )
            self.n_stacked_obs_puck_pos = 4
            self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs_puck_pos)
        else:
            raise ValueError("Invalid scheme")

        ## For evaluation, the agent is fully internal and loaded from a checkpoint.
        if self.evaluate:
            # Patch DreamerV3
            _apply_monkey_patch_dreamerv3()

            self.load_model_path = (
                load_model_path
                if load_model_path is not None
                else self.DEFAULT_INFERENCE_MODEL[self.task]
            )

            # Setup config
            config = config_dreamerv3()
            config = embodied.Flags(config).parse(argv=[])
            step = embodied.Counter()

            # Setup agent
            self._as_env = EmbodiedChallengeWrapper(self)
            self.agent = dreamerv3.Agent(
                self._as_env.obs_space, self._as_env.act_space, step, config
            )

            # Load checkpoint
            checkpoint = embodied.Checkpoint()
            checkpoint.agent = self.agent
            checkpoint.load(self.load_model_path, keys=["agent"])

            # Setup agent driver
            policy = lambda *args: self.agent.policy(*args, mode="eval")
            self.policy_driver = PolicyEvalDriver(policy=policy)

            self.initialize_inference()

            # # Setup async inference
            # if self.ASYNC_INFERENCE:
            #     self._original_interval = gsi()
            #     self.mutex = Lock()
            #     self.cv_new_obs_avail = Condition(lock=self.mutex)
            #     self.cv_new_act_avail = Condition(lock=self.mutex)
            #     self.inference_in_progress = False
            #     self.new_obs_in_queue = False
            #     self.thread = Thread(target=self.inference_loop)
            #     self.thread.start()

        self.reset()

    @property
    def observation_space(self):
        n_obs = (
            1
            + 2 * self.n_stacked_obs_participant_ee_pos
            + 2 * self.n_stacked_obs_opponent_ee_pos
            + 2 * self.n_stacked_obs_puck_pos
        )
        if self.scheme == 6:
            n_obs += 7

        dtype = np.float32

        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_obs,),
            dtype=dtype,
        )

    @property
    def action_space(self):
        # if self.interpolation_order in [1, 2]:
        # The desired XY position of the mallet
        #  - pos_x
        #  - pos_y

        dtype = np.float32

        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=dtype,
        )

    ##### Evaluation only #####

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        # ## Asynchronous inference
        # if self.ASYNC_INFERENCE:
        #     ssi(self.STEP_TIME_LIMIT)
        #     start_time = time.time()

        #     # Extract and normalize relevant observations
        #     obs = self.process_raw_obs(obs)

        #     # Notify the inference thread that new observation is available
        #     with self.cv_new_obs_avail:
        #         self.obs = obs
        #         if self.inference_in_progress:
        #             self.new_obs_in_queue = True
        #         else:
        #             self.cv_new_obs_avail.notify()

        #     # Wait for new action to be available or until time limit is reached
        #     with self.cv_new_act_avail:
        #         self.cv_new_act_avail.wait(
        #             timeout=self.STEP_TIME_LIMIT - (time.time() - start_time)
        #         )
        #         act = np.copy(self.act)
        #     ssi(self._original_interval)
        # else:
        #     ## Synchronous inference
        #     obs = self.process_raw_obs(obs)
        #     act = self.infer_action(obs)
        # return act

        return self.infer_action(self.process_raw_obs(obs))

    def inference_loop(self):
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)

        while True:
            # Wait for new observation to be available
            with self.cv_new_obs_avail:
                if self.new_obs_in_queue:
                    self.new_obs_in_queue = False
                else:
                    self.cv_new_obs_avail.wait()
                obs = np.copy(self.obs)
                self.inference_in_progress = True

            # Infer a new action and process it
            act = self.infer_action(obs)

            # Notify the main thread that new action is available
            with self.cv_new_act_avail:
                self.inference_in_progress = False
                self.act = act
                self.cv_new_act_avail.notify()

    def infer_action(self, obs):
        return self.process_raw_act(
            self.policy_driver.infer(obs)["action"].squeeze().clip(-1.0, 1.0)
        )

    def initialize_inference(self) -> np.ndarray:
        self.policy_driver.infer(
            np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        )
        self.policy_driver.reset()

    def reset(self):
        self.act = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        self.obs = np.zeros(
            self.observation_space.shape, dtype=self.observation_space.dtype
        )

        # if self.filter_actions_enabled:
        #     self.previous_ee_pos_xy_norm = None

        if self.evaluate:
            self.policy_driver.reset()
            # if self.ASYNC_INFERENCE:
            #     with self.cv_new_obs_avail:
            #         self.cv_new_obs_avail.wait(timeout=self.STEP_TIME_LIMIT)
            #     with self.cv_new_act_avail:
            #         self.cv_new_act_avail.wait(timeout=self.STEP_TIME_LIMIT)
            #     self.inference_in_progress = False
            #     self.new_obs_in_queue = False

        self.penalty_timer = 0.0
        self.stacked_obs_participant_ee_pos.clear()
        self.stacked_obs_opponent_ee_pos.clear()
        self.stacked_obs_puck_pos.clear()

        # Additional reset for scheme 3/4
        if self.scheme != 2:
            self.penalty_side = None

    #### ~Evaluation only~ ####

    ######### Common ##########

    def process_raw_obs(self, obs: np.ndarray) -> np.ndarray:
        # Player's joint states
        # Note: Not used in observation vector, but used for action processing
        self.current_joint_pos = self.get_joint_pos(obs)

        # Note: All obs are normalized with respect to table size in the same manner

        # Player's end-effector position
        self.current_ee_pos = self.get_ee_pose(obs)[0]
        ee_pos_xy_norm = self._normalize_value(
            self.current_ee_pos[:2],
            low_in=self.puck_table_minmax[:, 0],
            high_in=self.puck_table_minmax[:, 1],
        )
        self.current_ee_pos_xy_norm = ee_pos_xy_norm

        # Opponent's end-effector position
        opponent_ee_pos_xy = self.get_opponent_ee_pos(obs)[:2]

        # If training, add noise to the observation of opponent's position
        if not self.evaluate:
            opponent_ee_pos_xy += np.random.normal(
                0.0,
                self.noise_obs_opponent_ee_pos_std,
                size=opponent_ee_pos_xy.shape,
            )

        # Normalize opponent's end-effector position
        opponent_ee_pos_xy_norm = self._normalize_value(
            opponent_ee_pos_xy,
            low_in=self.puck_table_minmax[:, 0],
            high_in=self.puck_table_minmax[:, 1],
        )

        # Puck's position
        puck_pos_xy_norm = self._normalize_value(
            self.get_puck_pos(obs)[:2],
            low_in=self.puck_table_minmax[:, 0],
            high_in=self.puck_table_minmax[:, 1],
        )

        # Concatenate into a single observation vector
        if self.scheme == 2:
            # Append to stacked observation to preserve temporal information
            self.stacked_obs_puck_pos.append(puck_pos_xy_norm)
            while not self.n_stacked_obs_puck_pos == len(self.stacked_obs_puck_pos):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_puck_pos.append(puck_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm)
            while not self.n_stacked_obs_participant_ee_pos == len(
                self.stacked_obs_participant_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm)
            while not self.n_stacked_obs_opponent_ee_pos == len(
                self.stacked_obs_opponent_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm.copy())

            # Compute penalty timer
            if puck_pos_xy_norm[0] < 0.0:
                current_penalty_timer = self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S
                self.penalty_timer += self.sim_dt
            else:
                current_penalty_timer = -1.0
                self.penalty_timer = 0.0

            # Concaternate episode progress with all temporally-stacked observations
            obs = np.clip(
                np.concatenate(
                    (
                        [current_penalty_timer],
                        np.array(self.stacked_obs_puck_pos, dtype=np.float32).flatten(),
                        np.array(
                            self.stacked_obs_participant_ee_pos, dtype=np.float32
                        ).flatten(),
                        np.array(
                            self.stacked_obs_opponent_ee_pos, dtype=np.float32
                        ).flatten(),
                    )
                ),
                -1.0,
                1.0,
            )

        elif self.scheme in [3, 4]:
            # Append to stacked observation to preserve temporal information
            self.stacked_obs_puck_pos.append(puck_pos_xy_norm)
            while not self.n_stacked_obs_puck_pos == len(self.stacked_obs_puck_pos):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_puck_pos.append(puck_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm)
            while not self.n_stacked_obs_participant_ee_pos == len(
                self.stacked_obs_participant_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm)
            while not self.n_stacked_obs_opponent_ee_pos == len(
                self.stacked_obs_opponent_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm.copy())

            # Compute penalty timer (this is the only difference from scheme 2)
            if self.penalty_side is None:
                self.penalty_side = np.sign(puck_pos_xy_norm[0])
            elif np.sign(puck_pos_xy_norm[0]) == self.penalty_side:
                self.penalty_timer += self.sim_dt
            else:
                self.penalty_side *= -1
                self.penalty_timer = 0.0
            current_penalty_timer = (
                self.penalty_side * self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S
            )

            # Concaternate episode progress with all temporally-stacked observations
            obs = np.clip(
                np.concatenate(
                    (
                        [current_penalty_timer],
                        np.array(self.stacked_obs_puck_pos, dtype=np.float32).flatten(),
                        np.array(
                            self.stacked_obs_participant_ee_pos, dtype=np.float32
                        ).flatten(),
                        np.array(
                            self.stacked_obs_opponent_ee_pos, dtype=np.float32
                        ).flatten(),
                    )
                ),
                -1.0,
                1.0,
            )

        elif self.scheme == 5:
            # Append to stacked observation to preserve temporal information
            self.stacked_obs_puck_pos.append(puck_pos_xy_norm)
            while not self.n_stacked_obs_puck_pos == len(self.stacked_obs_puck_pos):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_puck_pos.append(puck_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm)
            while not self.n_stacked_obs_participant_ee_pos == len(
                self.stacked_obs_participant_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm)
            while not self.n_stacked_obs_opponent_ee_pos == len(
                self.stacked_obs_opponent_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm.copy())

            # Compute penalty timer (this is the only difference from scheme 2)
            if self.penalty_side is None:
                self.penalty_side = np.sign(puck_pos_xy_norm[0])
            elif np.sign(puck_pos_xy_norm[0]) == self.penalty_side:
                self.penalty_timer += self.sim_dt
            else:
                self.penalty_side *= -1
                self.penalty_timer = 0.0
            current_penalty_timer = (
                self.penalty_side * self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S
            )

            # Concaternate episode progress with all temporally-stacked observations
            obs = np.clip(
                np.concatenate(
                    (
                        [current_penalty_timer],
                        np.array(self.stacked_obs_puck_pos, dtype=np.float32).flatten(),
                        np.array(
                            self.stacked_obs_participant_ee_pos, dtype=np.float32
                        ).flatten(),
                        np.array(
                            self.stacked_obs_opponent_ee_pos, dtype=np.float32
                        ).flatten(),
                    )
                ),
                -1.0,
                1.0,
            )

        elif self.scheme == 6:
            # Append to stacked observation to preserve temporal information
            self.stacked_obs_puck_pos.append(puck_pos_xy_norm)
            while not self.n_stacked_obs_puck_pos == len(self.stacked_obs_puck_pos):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_puck_pos.append(puck_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm)
            while not self.n_stacked_obs_participant_ee_pos == len(
                self.stacked_obs_participant_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm.copy())

            # Append to stacked observation to preserve temporal information
            self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm)
            while not self.n_stacked_obs_opponent_ee_pos == len(
                self.stacked_obs_opponent_ee_pos
            ):
                # For the first buffer after reset, fill with identical observations until deque is full
                self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm.copy())

            # Compute penalty timer (this is the only difference from scheme 2)
            if self.penalty_side is None:
                self.penalty_side = np.sign(puck_pos_xy_norm[0])
            elif np.sign(puck_pos_xy_norm[0]) == self.penalty_side:
                self.penalty_timer += self.sim_dt
            else:
                self.penalty_side *= -1
                self.penalty_timer = 0.0
            current_penalty_timer = (
                self.penalty_side * self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S
            )

            # Normalize joint positions and concatenate with the rest of the observation
            current_joint_pos_normalized = self._normalize_value(
                self.current_joint_pos,
                low_in=self.robot_joint_pos_limit[0, :],
                high_in=self.robot_joint_pos_limit[1, :],
            )

            # Concaternate episode progress with all temporally-stacked observations
            obs = np.clip(
                np.concatenate(
                    (
                        [current_penalty_timer],
                        current_joint_pos_normalized,
                        np.array(self.stacked_obs_puck_pos, dtype=np.float32).flatten(),
                        np.array(
                            self.stacked_obs_participant_ee_pos, dtype=np.float32
                        ).flatten(),
                        np.array(
                            self.stacked_obs_opponent_ee_pos, dtype=np.float32
                        ).flatten(),
                    )
                ),
                -1.0,
                1.0,
            )

        assert obs.shape == self.observation_space.shape

        return obs

    def process_raw_act(self, action: np.ndarray) -> np.ndarray:
        assert self.interpolation_order in [-1, 1, 2, 3, 4]
        assert action.shape == self.action_space.shape
        assert max(action) <= 1.0 and min(action) >= -1.0

        # if not hasattr(self, "previous_action"):
        #     self.previous_action = np.zeros_like(action)
        # if (
        #     np.linalg.norm(
        #         self.current_ee_pos[:2]
        #         - self._unnormalize_value(
        #             self.previous_action,
        #             low_out=self.ee_table_minmax[:, 0],
        #             high_out=self.ee_table_minmax[:, 1],
        #         )
        #     )
        #     < 0.025
        # ):
        #     self.previous_action = [
        #         np.array([-1.0, -1.0]),
        #         np.array([-1.0, 1.0]),
        #         np.array([1.0, -1.0]),
        #         np.array([1.0, 0.0]),
        #         np.array([-1.0, 0.0]),
        #         np.array([0.0, 0.0]),
        #         np.array([1.0, 1.0]),
        #     ][np.random.randint(0, 7)]
        # action = self.previous_action

        target_ee_pos_xy = action

        # # Filter the target position
        # if self.filter_actions_enabled:
        #     if self.previous_ee_pos_xy_norm is None:
        #         self.previous_ee_pos_xy_norm = self.current_ee_pos_xy_norm
        #     target_ee_pos_xy = (
        #         self.filter_actions_coefficient * self.previous_ee_pos_xy_norm
        #         + (1 - self.filter_actions_coefficient) * target_ee_pos_xy
        #     )
        #     self.previous_ee_pos_xy_norm = target_ee_pos_xy

        # Unnormalize the action and combine with desired height
        target_ee_pos = np.array(
            [
                *self._unnormalize_value(
                    target_ee_pos_xy,
                    low_out=self.ee_table_minmax[:, 0],
                    high_out=self.ee_table_minmax[:, 1],
                ),
                self.robot_ee_desired_height,
            ],
            dtype=action.dtype,
        )

        # Calculate the target joint disp via Inverse Jacobian method
        joint_disp = self.servo(target_ee_pos)

        # Convert to joint velocities based on joint displacements
        joint_vel = joint_disp / self.sim_dt

        # Limit the joint velocities to the maximum allowed
        joints_below_vel_limit = joint_vel < self.robot_joint_vel_limit_scaled[0, :]
        joints_above_vel_limit = joint_vel > self.robot_joint_vel_limit_scaled[1, :]
        joints_outside_vel_limit = np.logical_or(
            joints_below_vel_limit, joints_above_vel_limit
        )
        if np.any(joints_outside_vel_limit):
            downscaling_factor = 1.0
            for joint_i in np.where(joints_outside_vel_limit)[0]:
                downscaling_factor = min(
                    downscaling_factor,
                    1
                    - (
                        (
                            joint_vel[joint_i]
                            - self.robot_joint_vel_limit_scaled[
                                int(joints_above_vel_limit[joint_i]), joint_i
                            ]
                        )
                        / joint_vel[joint_i]
                    ),
                )
            # Scale down the joint velocities to the maximum allowed limits
            joint_vel *= downscaling_factor

        # Update the target joint positions based on joint velocities
        joint_pos = self.current_joint_pos + (self.sim_dt * joint_vel)

        # Assign the action
        if self.interpolation_order in [1, 2]:
            action = np.array(joint_pos)
        else:
            action = np.array([joint_pos, joint_vel])

        return action

    def servo(self, target_ee_pos):
        target_ee_disp = target_ee_pos - self.current_ee_pos

        jac = self.jacobian(self.current_joint_pos)[:3]
        jac_pinv = np.linalg.pinv(jac)

        # Weighted average of joint displacements, such that Z position is closely maintained
        s = np.linalg.svd(jac, compute_uv=False)
        s[:2] = np.mean(s[:2])
        s[2] *= self.z_position_control_tolerance
        s = 1 / s
        s = s / np.sum(s)

        joint_disp = jac_pinv * target_ee_disp
        joint_disp = np.average(joint_disp, axis=1, weights=s)

        return joint_disp

    @classmethod
    def _normalize_value(
        cls, value: np.ndarray, low_in: np.ndarray, high_in: np.ndarray
    ) -> np.ndarray:
        """
        Normalize values to [-1, 1] range.
        """
        return 2 * (value - low_in) / (high_in - low_in) - 1

    @classmethod
    def _unnormalize_value(
        cls, value: np.ndarray, low_out: np.ndarray, high_out: np.ndarray
    ) -> np.ndarray:
        """
        unnormalize values from [-1, 1] range.
        """
        return (high_out - low_out) * (value + 1) / 2 + low_out

    ######## ~Common~ #########

    ######## Utilities ########

    def extract_env_info(self):
        # Number of agents
        __n_agents: int = self.env_info["n_agents"]
        self.is_tournament: bool = __n_agents == 2

        # Information about the table
        __table_width: float = self.env_info["table"]["width"]
        __table_length: float = self.env_info["table"]["length"]
        self.table_size: np.ndarray = np.array([__table_length, __table_width])
        self.table_goal_width: float = self.env_info["table"]["goal_width"]

        # Information about the puck
        self.puck_radius: float = self.env_info["puck"]["radius"]

        # Information about the mallet
        self.mallet_radius: float = self.env_info["mallet"]["radius"]

        # Information about the robot
        self.robot_n_joints: int = self.env_info["robot"]["n_joints"]
        self.robot_control_frequency: float = self.env_info["robot"][
            "control_frequency"
        ]
        self.robot_ee_desired_height: float = self.env_info["robot"][
            "ee_desired_height"
        ]
        self.robot_base_frame: np.ndarray = self.env_info["robot"]["base_frame"]
        self.robot_joint_pos_limit: np.ndarray = self.env_info["robot"][
            "joint_pos_limit"
        ]
        self.robot_joint_vel_limit: np.ndarray = self.env_info["robot"][
            "joint_vel_limit"
        ]
        self.robot_joint_acc_limit: np.ndarray = self.env_info["robot"][
            "joint_acc_limit"
        ]
        self.robot_joint_vel_limit_scaled = (
            self.vel_constraints_scaling_factor * self.robot_joint_vel_limit
        )

        # Information about the simulation
        self.sim_dt: float = self.env_info["dt"]

        # Information about the task
        self.task = AirHockeyTask.from_str(env_name=self.env_info["env_name"])

        # Information about the RL task
        self.rl_info: MDPInfo = self.env_info["rl_info"]

        # Information about the constraints
        self.constraints: ConstraintList = self.env_info["constraints"]

        ## Derived
        self.ee_table_minmax = np.array(
            [
                [
                    np.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.mallet_radius
                    + self.operating_area_offset_from_table
                    + self.operating_area_offset_from_goal,
                    np.abs(self.robot_base_frame[0][0, 3])
                    - self.operating_area_offset_from_centre,
                ],
                [
                    -(self.table_size[1] / 2)
                    + self.mallet_radius
                    + self.operating_area_offset_from_table,
                    (self.table_size[1] / 2)
                    - self.mallet_radius
                    - self.operating_area_offset_from_table,
                ],
            ]
        )

        self.puck_table_minmax = np.array(
            [
                [
                    np.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.puck_radius,
                    np.abs(self.robot_base_frame[0][0, 3])
                    + (self.table_size[0] / 2)
                    - self.puck_radius,
                ],
                [
                    -(self.table_size[1] / 2) + self.puck_radius,
                    (self.table_size[1] / 2) - self.puck_radius,
                ],
            ]
        )

    def forward_kinematics(self, q, link="ee"):
        return forward_kinematics(
            mj_model=self.robot_model, mj_data=self.robot_data, q=q, link=link
        )

    def inverse_kinematics(
        self,
        desired_position,
        desired_rotation=None,
        initial_q=None,
        link="ee",
    ):
        return inverse_kinematics(
            mj_model=self.robot_model,
            mj_data=self.robot_data,
            desired_position=desired_position,
            desired_rotation=desired_rotation,
            initial_q=initial_q,
            link=link,
        )

    def jacobian(self, q, link="ee"):
        return jacobian(
            mj_model=self.robot_model, mj_data=self.robot_data, q=q, link=link
        )

    def get_puck_pos(self, obs):
        return obs[self.env_info["puck_pos_ids"]]

    def get_puck_vel(self, obs):
        return obs[self.env_info["puck_vel_ids"]]

    def get_joint_pos(self, obs):
        return obs[self.env_info["joint_pos_ids"]]

    def get_joint_vel(self, obs):
        return obs[self.env_info["joint_vel_ids"]]

    def get_ee_pose(self, obs):
        return self.forward_kinematics(self.get_joint_pos(obs))

    def get_opponent_ee_pos(self, obs):
        return obs[self.env_info["opponent_ee_ids"]]

    ####### ~Utilities~ #######


def _apply_monkey_patch_dreamerv3():
    ## MONKEY PATCH: Speed up initialization for inference
    def __monkey_patch__init_varibs(self, obs_space, act_space):
        rng = self._next_rngs(self.train_devices, mirror=True)
        obs = self._dummy_batch(obs_space, (1,))
        state, varibs = self._init_policy({}, rng, obs["is_first"])
        varibs = self._policy(varibs, rng, obs, state, mode="eval", init_only=True)
        return varibs

    dreamerv3.jaxagent.JAXAgent._init_varibs = __monkey_patch__init_varibs
    ## ~MONKEY PATCH: Speed up initialization for inference
