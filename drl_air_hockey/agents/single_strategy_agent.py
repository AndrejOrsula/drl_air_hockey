from __future__ import annotations

from collections import deque
from os import path
from typing import Any, Dict, Optional

import dreamerv3
import gym
import numpy as np
from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
)
from dreamerv3 import embodied

from drl_air_hockey.utils.config import (
    DIR_MODELS,
    MAX_TIME_UNTIL_PENALTY_S,
    config_dreamerv3,
)
from drl_air_hockey.utils.env_wrapper import EmbodiedChallengeWrapper
from drl_air_hockey.utils.eval import PolicyEvalDriver
from drl_air_hockey.utils.tournament_agent_strategies import BalancedAgentStrategy


class SingleStrategySpaceRAgent(AgentBase):
    # Filtering of high-level actions
    FILTER_ACTIONS_ENABLED: bool = True
    FILTER_ACTIONS_COEFFICIENT: float = 0.75

    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        model_path: Optional[str] = path.join(DIR_MODELS, "tournament_balanced.ckpt"),
        **kwargs,
    ):
        ## Chain up the parent implementation
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)

        ## Extract information about the environment and write it to members
        self.extract_env_info()
        self.derive_env_info()

        ## Initialize all agent models
        self.init_agent(model_path)

        ## Initialize observation and action schemes
        self.init_observation_scheme()
        self.init_action_scheme()

        ## Trigger the first reset
        self.reset()

    def init_agent(self, model_path: str):
        self.policy = self._init_agent(model_path)

    def _init_agent(self, model_path: str) -> PolicyEvalDriver:
        # Setup config
        config = config_dreamerv3()
        config = embodied.Flags(config).parse(argv=[])
        step = embodied.Counter()

        # Setup agent
        self_as_env = EmbodiedChallengeWrapper(self)
        agent = dreamerv3.Agent(
            self_as_env.obs_space, self_as_env.act_space, step, config
        )

        # Load checkpoint
        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(model_path, keys=["agent"])

        # Setup policy driver
        policy = lambda *args: agent.policy(*args, mode="eval")
        policy_driver = PolicyEvalDriver(policy=policy)

        # Initialize the policy driver
        policy_driver.infer(
            np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        )
        policy_driver.reset()

        # Return the policy driver
        return policy_driver

    def init_observation_scheme(self):
        ## Tracker of the penalty timer in the observation
        self.penalty_side = None
        self.penalty_timer = 0.0

        ## Stacked observations for the position observations (agent, opponent, puck)
        self.n_stacked_obs_participant_ee_pos = 2
        self.stacked_obs_participant_ee_pos = deque(
            [], maxlen=self.n_stacked_obs_participant_ee_pos
        )
        self.n_stacked_obs_opponent_ee_pos = 2
        self.stacked_obs_opponent_ee_pos = deque(
            [], maxlen=self.n_stacked_obs_opponent_ee_pos
        )
        self.n_stacked_obs_puck_pos = 10
        self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs_puck_pos)
        self.n_stacked_obs_puck_rot = 2
        self.stacked_obs_puck_rot = deque([], maxlen=self.n_stacked_obs_puck_rot)

    def init_action_scheme(self):
        self.robot_joint_vel_limit_scaled = {}
        self.ee_table_minmax = {}
        self.z_position_control_tolerance = {}

        strategy_kwargs = BalancedAgentStrategy().get_env_kwargs()

        self.robot_joint_vel_limit_scaled = (
            strategy_kwargs["vel_constraints_scaling_factor"]
            * self.robot_joint_vel_limit
        )
        self.ee_table_minmax = self.compute_ee_table_minmax(
            operating_area_offset_from_table=strategy_kwargs[
                "operating_area_offset_from_table"
            ],
            operating_area_offset_from_centre=strategy_kwargs[
                "operating_area_offset_from_centre"
            ],
            operating_area_offset_from_goal=strategy_kwargs[
                "operating_area_offset_from_goal"
            ],
        )
        self.z_position_control_tolerance = strategy_kwargs[
            "z_position_control_tolerance"
        ]

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(40,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        # The desired XY position of the mallet
        #  - pos_x
        #  - pos_y
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        return self.process_raw_act(
            self.policy.infer(self.process_raw_obs(obs))["action"]
            .squeeze()
            .clip(-1.0, 1.0)
        )

    def reset(self):
        self.policy.reset()

        if self.FILTER_ACTIONS_ENABLED:
            self.previous_ee_pos_xy_norm = None

        self.penalty_side = None
        self.penalty_timer = 0.0

        self.stacked_obs_participant_ee_pos.clear()
        self.stacked_obs_opponent_ee_pos.clear()
        self.stacked_obs_puck_pos.clear()
        self.stacked_obs_puck_rot.clear()

        self._new_episode = True

    def process_raw_obs(self, obs: np.ndarray) -> np.ndarray:
        ## Normalize used observations
        # Player's Joint positions
        self.current_joint_pos = self.get_joint_pos(obs)
        current_joint_pos_normalized = np.clip(
            self._normalize_value(
                self.current_joint_pos,
                low_in=self.robot_joint_pos_limit[0, :],
                high_in=self.robot_joint_pos_limit[1, :],
            ),
            -1.0,
            1.0,
        )

        # Player's end-effector position
        self.current_ee_pos = self.get_ee_pose(obs)[0]
        ee_pos_xy_norm = np.clip(
            self._normalize_value(
                self.current_ee_pos[:2],
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )
        self.current_ee_pos_xy_norm = ee_pos_xy_norm

        # Opponent's end-effector position
        opponent_ee_pos = self.get_opponent_ee_pos(obs)
        opponent_ee_pos_xy_norm = np.clip(
            self._normalize_value(
                opponent_ee_pos[:2],
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )

        # Puck's position
        puck_pos = self.get_puck_pos(obs)
        self.last_puck_pos_xy = puck_pos[:2]
        puck_pos_xy_norm = np.clip(
            self._normalize_value(
                puck_pos[:2],
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )

        # Puck's rotation (sin and cos)
        puck_rot_yaw_norm = np.array([np.sin(puck_pos[2]), np.cos(puck_pos[2])])

        # Puck's x velocity (used for estimating metrics)
        self.last_puck_vel_xy = self.get_puck_vel(obs)[:2]

        ## Compute penalty timer
        if self.penalty_side is None:
            self.penalty_side = np.sign(puck_pos_xy_norm[0])
        elif np.sign(puck_pos_xy_norm[0]) == self.penalty_side:
            self.penalty_timer += self.sim_dt
        else:
            self.penalty_side *= -1
            self.penalty_timer = 0.0
        current_penalty_timer = np.clip(
            self.penalty_side * self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S,
            -1.0,
            1.0,
        )

        ## Append current observation to a stack that preserves temporal information
        if self._new_episode:
            self.stacked_obs_puck_pos.extend(
                np.tile(puck_pos_xy_norm, (self.n_stacked_obs_puck_pos, 1))
            )
            self.stacked_obs_puck_rot.extend(
                np.tile(puck_rot_yaw_norm, (self.n_stacked_obs_puck_rot, 1))
            )
            self.stacked_obs_participant_ee_pos.extend(
                np.tile(ee_pos_xy_norm, (self.n_stacked_obs_participant_ee_pos, 1))
            )
            self.stacked_obs_opponent_ee_pos.extend(
                np.tile(
                    opponent_ee_pos_xy_norm, (self.n_stacked_obs_opponent_ee_pos, 1)
                )
            )
            self._new_episode = False
        else:
            self.stacked_obs_puck_pos.append(puck_pos_xy_norm)
            self.stacked_obs_puck_rot.append(puck_rot_yaw_norm)
            self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm)
            self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm)

        # Concaternate episode progress with all temporally-stacked observations
        obs = np.concatenate(
            (
                np.array((current_penalty_timer,)),
                current_joint_pos_normalized,
                np.array(self.stacked_obs_puck_pos).flatten(),
                np.array(self.stacked_obs_puck_rot).flatten(),
                np.array(self.stacked_obs_participant_ee_pos).flatten(),
                np.array(self.stacked_obs_opponent_ee_pos).flatten(),
            )
        )

        return obs

    def process_raw_act(self, action: np.ndarray) -> np.ndarray:
        target_ee_pos_xy = action

        # Filter the target position
        if self.FILTER_ACTIONS_ENABLED:
            if self.previous_ee_pos_xy_norm is None:
                self.previous_ee_pos_xy_norm = self.current_ee_pos_xy_norm
            target_ee_pos_xy = (
                self.FILTER_ACTIONS_COEFFICIENT * self.previous_ee_pos_xy_norm
                + (1 - self.FILTER_ACTIONS_COEFFICIENT) * target_ee_pos_xy
            )
            self.previous_ee_pos_xy_norm = target_ee_pos_xy

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
            dtype=target_ee_pos_xy.dtype,
        )

        # Calculate the target joint disp via Inverse Jacobian method
        target_ee_disp = target_ee_pos - self.current_ee_pos
        jac = self.jacobian(self.current_joint_pos)[:3]
        jac_pinv = np.linalg.pinv(jac)
        s = np.linalg.svd(jac, compute_uv=False)
        s[:2] = np.mean(s[:2])
        s[2] *= self.z_position_control_tolerance
        s = 1 / s
        s = s / np.sum(s)
        joint_disp = jac_pinv * target_ee_disp
        joint_disp = np.average(joint_disp, axis=1, weights=s)

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
        return np.array([joint_pos, joint_vel])

    @classmethod
    def _normalize_value(
        cls, value: np.ndarray, low_in: np.ndarray, high_in: np.ndarray
    ) -> np.ndarray:
        return 2 * (value - low_in) / (high_in - low_in) - 1

    @classmethod
    def _unnormalize_value(
        cls, value: np.ndarray, low_out: np.ndarray, high_out: np.ndarray
    ) -> np.ndarray:
        return (high_out - low_out) * (value + 1) / 2 + low_out

    def extract_env_info(self):
        # Information about the table
        self.table_size: np.ndarray = np.array(
            [self.env_info["table"]["length"], self.env_info["table"]["width"]]
        )

        # Information about the puck
        self.puck_radius: float = self.env_info["puck"]["radius"]

        # Information about the mallet
        self.mallet_radius: float = self.env_info["mallet"]["radius"]

        # Information about the robot
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

        # Information about the simulation
        self.sim_dt: float = self.env_info["dt"]

    def derive_env_info(self):
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

    def compute_ee_table_minmax(
        self,
        operating_area_offset_from_table: float,
        operating_area_offset_from_centre: float,
        operating_area_offset_from_goal: float,
    ) -> (np.ndarray, np.ndarray, float):
        return np.array(
            [
                [
                    np.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.mallet_radius
                    + operating_area_offset_from_table
                    + operating_area_offset_from_goal,
                    np.abs(self.robot_base_frame[0][0, 3])
                    - operating_area_offset_from_centre,
                ],
                [
                    -(self.table_size[1] / 2)
                    + self.mallet_radius
                    + operating_area_offset_from_table,
                    (self.table_size[1] / 2)
                    - self.mallet_radius
                    - operating_area_offset_from_table,
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

    def get_ee_pose(self, obs):
        return self.forward_kinematics(self.get_joint_pos(obs))

    def get_opponent_ee_pos(self, obs):
        return obs[self.env_info["opponent_ee_ids"]]
