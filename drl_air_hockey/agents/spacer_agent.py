from collections import deque
from os import path
from typing import Any, Dict, Optional

import gymnasium
import numpy
from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    jacobian,
)

# from drl_air_hockey.dreamer.wrapper import EmbodiedEnvWrapper
# from drl_air_hockey.utils._del_eval import PolicyEvalDriver
# from drl_air_hockey.utils._del_task import Task as AirHockeyTask
# from drl_air_hockey.utils._del_config import (
#     DIR_MODELS,
#     INTERPOLATION_ORDER,
#     MAX_TIME_UNTIL_PENALTY_S,
#     config_dreamerv3,
# )

EPISODE_MAX_STEPS: int = 45000
MAX_TIME_UNTIL_PENALTY_S: float = 7.0


class SpaceRAgent(AgentBase):
    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        interpolation_order: Optional[int] = -1,
        # Whether to train or evaluate (inference)
        train: bool = False,
        # Path to the model to load for inference
        model_path: Optional[str] = None,
        # Velocity constraints (0.5 is about safe)
        vel_constraints_scaling_factor: float = 0.65,
        # Whether to filter actions and by how much
        filter_actions_enabled: bool = True,
        filter_actions_coefficient: float = 0.75,
        # Extra offsets for operating area of the agent (in meters)
        operating_area_offset_from_centre: float = 0.17,
        operating_area_offset_from_table: float = 0.02,
        operating_area_offset_from_goal: float = 0.01,
        # Strictness of the Z position (positive only, lower is more strict)
        z_position_control_tolerance: float = 0.35,
        # Noise to apply to the observation of opponent's end-effector position
        noise_obs_opponent_ee_pos_std: float = 0.01,
        noise_obs_ee_pos_std: float = 0.0005,
        noise_obs_puck_pos_std: float = 0.002,
        noise_act_std: float = 0.0005,
        loss_of_tracking_prob_inc_per_step: float = 0.000005,
        loss_of_tracking_min_steps: int = 2,
        loss_of_tracking_max_steps: int = 5,
        **kwargs,
    ):
        ## Chain up the parent implementation
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)

        ## Get information about the agent
        self.agent_id = agent_id
        self.interpolation_order = interpolation_order
        self.evaluate = not train

        self.vel_constraints_scaling_factor = vel_constraints_scaling_factor
        self.filter_actions_enabled = filter_actions_enabled
        self.filter_actions_coefficient = filter_actions_coefficient
        self.operating_area_offset_from_centre = operating_area_offset_from_centre
        self.operating_area_offset_from_table = operating_area_offset_from_table
        self.operating_area_offset_from_goal = operating_area_offset_from_goal
        self.z_position_control_tolerance = z_position_control_tolerance

        self.noise_obs_opponent_ee_pos_std = noise_obs_opponent_ee_pos_std
        self.noise_obs_ee_pos_std = noise_obs_ee_pos_std
        self.noise_obs_puck_pos_std = noise_obs_puck_pos_std
        self.noise_act_std = noise_act_std

        # Emulate loss of tracking during training
        self.loss_of_tracking_prob_inc_per_step = loss_of_tracking_prob_inc_per_step
        self.loss_of_tracking_min_steps = loss_of_tracking_min_steps
        self.loss_of_tracking_max_steps = loss_of_tracking_max_steps
        self.lose_tracking_probability = 0.0
        self.lose_tracking_n_steps_remaining = 0

        ## Extract information about the environment and write it to members
        self.extract_env_info()

        self.penalty_side = None
        self.penalty_timer = 0.0

        # ## For evaluation, the agent is fully internal and loaded from a checkpoint.
        # if self.evaluate:
        #     self.model_path = (
        #         model_path
        #         if model_path is not None
        #         else self.DEFAULT_INFERENCE_MODEL[self.task]
        #     )

        #     # Setup config
        #     config = config_dreamerv3()
        #     config = embodied.Flags(config).parse(argv=[])
        #     step = embodied.Counter()

        #     # Setup agent
        #     self._as_env = EmbodiedEnvWrapper(self)

        #     self.agent = dreamerv3.Agent(
        #         self._as_env.obs_space, self._as_env.act_space, step, config
        #     )

        #     # Load checkpoint
        #     checkpoint = embodied.Checkpoint()
        #     checkpoint.agent = self.agent
        #     checkpoint.load(self.model_path, keys=["agent"])

        #     # Setup agent driver
        #     policy = lambda *args: self.agent.policy(*args, mode="eval")
        #     self.policy_driver = PolicyEvalDriver(policy=policy)

        #     self.initialize_inference()

        self.reset()

    @property
    def observation_space(self):
        n_obs = 0
        n_obs += 1  # penalty_timer
        n_obs += 2 + 2 + 2  # participant_ee_pos  # opponent_ee_pos  # puck_pos
        n_obs += 7  # participant_joint_pos
        n_obs += 7  # participant_joint_vel
        # n_obs += 2  # obs_puck_rot
        n_obs += 2  # obs_puck_vel

        return gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_obs,),
            dtype=numpy.float32,
        )

    @property
    def action_space(self):
        # The desired XY position of the mallet
        #  - pos_x
        #  - pos_y
        return gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=numpy.float32,
        )

    # def draw_action(self, obs: numpy.ndarray) -> numpy.ndarray:
    #     return self.infer_action(self.process_raw_obs(obs))

    # def infer_action(self, obs):
    #     return self.process_raw_act(
    #         self.policy_driver.infer(obs)["action"].squeeze().clip(-1.0, 1.0)
    #     )

    # def initialize_inference(self):
    #     self.policy_driver.infer(
    #         numpy.zeros(
    #             self.observation_space.shape, dtype=self.observation_space.dtype
    #         )
    #     )
    #     self.policy_driver.reset()

    def reset(self):
        # if self.evaluate:
        #     self.policy_driver.reset()

        self.lose_tracking_probability = 0.0
        self.lose_tracking_n_steps_remaining = 0

        if self.filter_actions_enabled:
            self.previous_ee_pos_xy_norm = None

        self.penalty_timer = 0.0
        self.penalty_side = None

    def process_raw_obs(self, obs: numpy.ndarray) -> numpy.ndarray:
        ## Normalize used observations
        # Player's Joint positions
        self.current_joint_pos = self.get_joint_pos(obs)
        current_joint_pos_normalized = numpy.clip(
            self._normalize_value(
                self.current_joint_pos,
                low_in=self.robot_joint_pos_limit[0, :],
                high_in=self.robot_joint_pos_limit[1, :],
            ),
            -1.0,
            1.0,
        )

        current_joint_vel = self.get_joint_vel(obs)

        # Player's end-effector position
        self.current_ee_pos = self.get_ee_pose(obs)[0]
        # if not self.evaluate:
        #     self.current_ee_pos += numpy.random.normal(
        #         0.0,
        #         self.noise_obs_ee_pos_std,
        #         size=self.current_ee_pos.shape,
        #     )
        ee_pos_xy_norm = numpy.clip(
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
        opponent_ee_pos_xy = self.get_opponent_ee_pos(obs)[:2]
        # if not self.evaluate:
        #     opponent_ee_pos_xy += numpy.random.normal(
        #         0.0,
        #         self.noise_obs_opponent_ee_pos_std,
        #         size=opponent_ee_pos_xy.shape,
        #     )
        opponent_ee_pos_xy_norm = numpy.clip(
            self._normalize_value(
                opponent_ee_pos_xy,
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )

        # Puck's position
        puck_pos = self.get_puck_pos(obs)
        # if not self.evaluate:
        #     puck_pos += numpy.random.normal(
        #         0.0,
        #         self.noise_obs_puck_pos_std,
        #         size=puck_pos.shape,
        #     )
        puck_pos_xy_norm = numpy.clip(
            self._normalize_value(
                puck_pos[:2],
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )

        # # Puck's rotation (sin and cos)
        # TODO: check i f pos 2 is yaw
        # puck_rot_yaw_norm = numpy.array([
        #     numpy.sin(puck_pos[2]),
        #     numpy.cos(puck_pos[2]),
        # ])

        puck_vel = self.get_puck_vel(obs)[:2]

        current_joint_pos_normalized = numpy.clip(
            self._normalize_value(
                self.current_joint_pos,
                low_in=self.robot_joint_pos_limit[0, :],
                high_in=self.robot_joint_pos_limit[1, :],
            ),
            -1.0,
            1.0,
        )

        ## Compute penalty timer
        if self.penalty_side is None:
            self.penalty_side = numpy.sign(puck_pos_xy_norm[0])
        elif numpy.sign(puck_pos_xy_norm[0]) == self.penalty_side:
            self.penalty_timer += self.sim_dt
        else:
            self.penalty_side *= -1
            self.penalty_timer = 0.0
        current_penalty_timer = numpy.clip(
            self.penalty_side * self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S,
            -1.0,
            1.0,
        )

        # if numpy.random.rand() < self.lose_tracking_probability:
        #     self.lose_tracking_n_steps_remaining = numpy.random.randint(
        #         self.loss_of_tracking_min_steps,
        #         self.loss_of_tracking_max_steps,
        #     )
        #     self.lose_tracking_probability = 0.0
        # else:
        #     self.lose_tracking_probability += (
        #         self.loss_of_tracking_prob_inc_per_step
        #     )

        # Concaternate episode progress with all observations
        obs = numpy.concatenate(
            (
                numpy.array((current_penalty_timer,)).flatten(),
                current_joint_pos_normalized.flatten(),
                current_joint_vel.flatten(),
                puck_pos_xy_norm.flatten(),
                # puck_rot_yaw_norm.flatten(),
                puck_vel.flatten(),
                ee_pos_xy_norm.flatten(),
                opponent_ee_pos_xy_norm.flatten(),
            )
        )

        # assert obs.shape == self.observation_space.shape, (
        #     f"Expected {self.observation_space.shape}, got {obs.shape}"
        #     + f"puck_vel: {puck_vel.flatten()}"
        #     + f"current_joint_vel: {current_joint_vel.flatten()}"
        # )

        return obs

    def process_raw_act(self, action: numpy.ndarray) -> numpy.ndarray:
        target_ee_pos_xy = action
        # Filter the target position
        if self.filter_actions_enabled:
            if self.previous_ee_pos_xy_norm is None:
                self.previous_ee_pos_xy_norm = self.current_ee_pos_xy_norm
            target_ee_pos_xy = (
                self.filter_actions_coefficient * self.previous_ee_pos_xy_norm
                + (1 - self.filter_actions_coefficient) * target_ee_pos_xy
            )
            self.previous_ee_pos_xy_norm = target_ee_pos_xy

        # Unnormalize the action and combine with desired height
        target_ee_pos = numpy.array(
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

        # if not self.evaluate:
        #     target_ee_pos[:2] += numpy.random.normal(
        #         0.0,
        #         self.noise_act_std,
        #         size=target_ee_pos[:2].shape,
        #     )

        # Calculate the target joint disp via Inverse Jacobian method
        target_ee_disp = target_ee_pos - self.current_ee_pos
        jac = self.jacobian(self.current_joint_pos)[:3]
        jac_pinv = numpy.linalg.pinv(jac)
        s = numpy.linalg.svd(jac, compute_uv=False)
        s[:2] = numpy.mean(s[:2])
        s[2] *= self.z_position_control_tolerance
        s = 1 / s
        s = s / numpy.sum(s)
        joint_disp = jac_pinv * target_ee_disp
        joint_disp = numpy.average(joint_disp, axis=1, weights=s)

        # Convert to joint velocities based on joint displacements
        joint_vel = joint_disp / self.sim_dt

        # Limit the joint velocities to the maximum allowed
        joints_below_vel_limit = joint_vel < self.robot_joint_vel_limit_scaled[0, :]
        joints_above_vel_limit = joint_vel > self.robot_joint_vel_limit_scaled[1, :]
        joints_outside_vel_limit = numpy.logical_or(
            joints_below_vel_limit, joints_above_vel_limit
        )
        if numpy.any(joints_outside_vel_limit):
            downscaling_factor = 1.0
            for joint_i in numpy.where(joints_outside_vel_limit)[0]:
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
        action = numpy.array([joint_pos, joint_vel])

        return action

    @classmethod
    def _normalize_value(
        cls, value: numpy.ndarray, low_in: numpy.ndarray, high_in: numpy.ndarray
    ) -> numpy.ndarray:
        return 2 * (value - low_in) / (high_in - low_in) - 1

    @classmethod
    def _unnormalize_value(
        cls, value: numpy.ndarray, low_out: numpy.ndarray, high_out: numpy.ndarray
    ) -> numpy.ndarray:
        return (high_out - low_out) * (value + 1) / 2 + low_out

    ######## Utilities ########

    def extract_env_info(self):
        # Information about the table
        self.table_size: numpy.ndarray = numpy.array(
            [
                self.env_info["table"]["length"],
                self.env_info["table"]["width"],
            ]
        )

        # Information about the puck
        self.puck_radius: float = self.env_info["puck"]["radius"]

        # Information about the mallet
        self.mallet_radius: float = self.env_info["mallet"]["radius"]

        # Information about the robot
        self.robot_ee_desired_height: float = self.env_info["robot"][
            "ee_desired_height"
        ]
        self.robot_base_frame: numpy.ndarray = self.env_info["robot"]["base_frame"]
        self.robot_joint_pos_limit: numpy.ndarray = self.env_info["robot"][
            "joint_pos_limit"
        ]
        self.robot_joint_vel_limit: numpy.ndarray = self.env_info["robot"][
            "joint_vel_limit"
        ]

        # Information about the simulation
        self.sim_dt: float = self.env_info["dt"]

        ## Derived
        self.robot_joint_vel_limit_scaled = (
            self.vel_constraints_scaling_factor * self.robot_joint_vel_limit
        )
        self.ee_table_minmax = numpy.array(
            [
                [
                    numpy.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.mallet_radius
                    + self.operating_area_offset_from_table
                    + self.operating_area_offset_from_goal,
                    numpy.abs(self.robot_base_frame[0][0, 3])
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
        self.puck_table_minmax = numpy.array(
            [
                [
                    numpy.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.puck_radius,
                    numpy.abs(self.robot_base_frame[0][0, 3])
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
