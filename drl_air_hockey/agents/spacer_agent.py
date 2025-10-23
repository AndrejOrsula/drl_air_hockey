from typing import Any, Dict, Tuple

import gymnasium
import numpy
from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from scipy.signal import butter, lfilter, lfilter_zi

EPISODE_MAX_STEPS: int = 45000
MAX_TIME_UNTIL_PENALTY_S: float = 15.0


class SpaceRAgent(AgentBase):
    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        train: bool = False,
        ## Robot control params
        joint_vel_limit_scaling_factor: float = 0.5,
        joint_acc_limit_scaling_factor: float = 0.5,
        ws_offset_centre: float = 0.2,
        ws_offset_table: float = 0.005,
        ws_offset_goal: float = 0.0075,
        dls_lambda: float = 0.05,
        osc_stiffness_xy_range: Tuple[float, float] = (2.0, 25.0),
        osc_damping_xy_range: Tuple[float, float] = (0.0, 0.5),
        osc_stiffness_z: float = 60.0,
        osc_damping_z: float = 0.3,
        bw_filter_cutoff_ratio: float = 0.025,
        **kwargs,
    ):
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)
        self.extract_env_info(
            joint_vel_limit_scaling_factor=joint_vel_limit_scaling_factor,
            joint_acc_limit_scaling_factor=joint_acc_limit_scaling_factor,
            ws_offset_centre=ws_offset_centre,
            ws_offset_table=ws_offset_table,
            ws_offset_goal=ws_offset_goal,
        )

        self.agent_id = agent_id
        self.train = train

        self.dls_lambda_square_ident = dls_lambda**2 * numpy.eye(3)
        self.osc_stiffness_xy_range = osc_stiffness_xy_range
        self.osc_damping_xy_range = osc_damping_xy_range
        self.osc_stiffness_z = osc_stiffness_z
        self.osc_damping_z = osc_damping_z

        self.is_initialized = False
        self.af_bw_b, self.af_bw_a = butter(  # type: ignore
            4, bw_filter_cutoff_ratio / self.sim_dt, fs=1.0 / self.sim_dt, btype="low"
        )

    @property
    def action_space(self):
        # Action space: [x, y, stiffness_x, stiffness_y, damping_x, damping_y]
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=numpy.float32)

    @property
    def observation_space(self):
        # Observation space: [7 joint pos, 7 joint vel, puck pos (x,y), puck vel (x,y), ee pos (x,y)]
        return gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(20,), dtype=numpy.float32
        )

    def reset(self):
        self.is_initialized = False

    def process_raw_act(self, action: numpy.ndarray) -> numpy.ndarray:
        action = numpy.clip(action, -1.0, 1.0)
        # Filter actions
        if not self.is_initialized:
            self.af_bw_zi_x = lfilter_zi(self.af_bw_b, self.af_bw_a) * action[0]
            self.af_bw_zi_y = lfilter_zi(self.af_bw_b, self.af_bw_a) * action[1]
            self._previous_joint_vel = self.current_joint_vel.copy()
            self.is_initialized = True
        action[0], self.af_bw_zi_x = lfilter(
            self.af_bw_b, self.af_bw_a, (action[0],), zi=self.af_bw_zi_x
        )
        action[1], self.af_bw_zi_y = lfilter(
            self.af_bw_b, self.af_bw_a, (action[1],), zi=self.af_bw_zi_y
        )

        # Action normalization
        target_ee_pos = numpy.array(
            (
                *_unnormalize(
                    action[:2],
                    self.ee_table_minmax[:, 0],
                    self.ee_table_minmax[:, 1],
                ),
                self.robot_ee_desired_height,
            ),
            dtype=action.dtype,
        )
        osc_stiffness = numpy.array(
            (
                *_unnormalize(action[2:4], *self.osc_stiffness_xy_range),
                self.osc_stiffness_z,
            )
        )
        osc_damping = numpy.array(
            (
                *_unnormalize(action[4:6], *self.osc_damping_xy_range),
                self.osc_damping_z,
            )
        )

        # Jacobian
        jacobian_lin = jacobian(
            self.robot_model, self.robot_data, self.current_joint_pos
        )[:3]
        jacobian_inv = jacobian_lin.T @ numpy.linalg.inv(
            jacobian_lin @ jacobian_lin.T + self.dls_lambda_square_ident
        )

        # Control
        error_pos = target_ee_pos - self.current_ee_pos
        error_vel = -(jacobian_lin @ self.current_joint_vel)
        target_ee_vel = osc_stiffness * error_pos + osc_damping * error_vel
        joint_vel = jacobian_inv @ target_ee_vel

        # Velocity limiting
        vel_limit_ratios = numpy.abs(joint_vel) / self.robot_joint_vel_limit_scaled
        max_vel_ratio = numpy.max(vel_limit_ratios)
        if max_vel_ratio > 1.0:
            joint_vel /= max_vel_ratio

        # Acceleration limiting
        joint_acc = (joint_vel - self._previous_joint_vel) / self.sim_dt
        acc_limit_ratios = numpy.abs(joint_acc) / self.robot_joint_acc_limit_scaled
        max_acc_ratio = numpy.max(acc_limit_ratios)
        if max_acc_ratio > 1.0:
            joint_acc /= max_acc_ratio
            joint_vel = self._previous_joint_vel + (joint_acc * self.sim_dt)
        self._previous_joint_vel = joint_vel.copy()

        # Calculate final position
        joint_pos = self.current_joint_pos + (self.sim_dt * joint_vel)

        return numpy.array((joint_pos, joint_vel))

    def process_raw_obs(self, obs: numpy.ndarray) -> numpy.ndarray:
        self.current_joint_pos = obs[self.env_info["joint_pos_ids"]]
        self.current_joint_vel = obs[self.env_info["joint_vel_ids"]]
        puck_pos = obs[self.env_info["puck_pos_ids"]][:2]
        puck_vel = obs[self.env_info["puck_vel_ids"]][:2]

        # Get EE position using Forward kinematics
        self.current_ee_pos, _ = forward_kinematics(
            self.robot_model, self.robot_data, self.current_joint_pos
        )

        current_joint_pos_normalized = _normalize(
            self.current_joint_pos,
            self.robot_joint_pos_limit[0, :],
            self.robot_joint_pos_limit[1, :],
        )
        ee_pos_xy_norm = _normalize(
            self.current_ee_pos[:2],
            self.ee_table_minmax[:, 0],
            self.ee_table_minmax[:, 1],
        )
        puck_pos_xy_norm = _normalize(
            puck_pos, self.puck_table_minmax[:, 0], self.puck_table_minmax[:, 1]
        )

        return numpy.concatenate(
            (
                current_joint_pos_normalized,
                self.current_joint_vel,
                ee_pos_xy_norm,
                puck_pos_xy_norm,
                puck_vel,
            )
        )

    def extract_env_info(
        self,
        joint_vel_limit_scaling_factor: float,
        joint_acc_limit_scaling_factor: float,
        ws_offset_centre: float,
        ws_offset_table: float,
        ws_offset_goal: float,
    ):
        self.sim_dt = self.env_info["dt"]
        self.table_size = numpy.array(
            (
                self.env_info["table"]["length"],
                self.env_info["table"]["width"],
            )
        )
        self.puck_radius = self.env_info["puck"]["radius"]
        self.mallet_radius = self.env_info["mallet"]["radius"]
        self.robot_ee_desired_height = self.env_info["robot"]["ee_desired_height"]
        self.robot_base_frame = self.env_info["robot"]["base_frame"]
        self.robot_joint_pos_limit = self.env_info["robot"]["joint_pos_limit"]
        self.robot_joint_vel_limit = self.env_info["robot"]["joint_vel_limit"]
        self.robot_joint_acc_limit = self.env_info["robot"]["joint_acc_limit"]
        self.robot_joint_vel_limit_scaled = (
            joint_vel_limit_scaling_factor * self.robot_joint_vel_limit
        )
        self.robot_joint_acc_limit_scaled = (
            joint_acc_limit_scaling_factor * self.robot_joint_acc_limit
        )

        self.ee_table_minmax = numpy.array(
            (
                (
                    numpy.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.mallet_radius
                    + ws_offset_table
                    + ws_offset_goal,
                    numpy.abs(self.robot_base_frame[0][0, 3]) - ws_offset_centre,
                ),
                (
                    (-(self.table_size[1] / 2) + self.mallet_radius + ws_offset_table),
                    (self.table_size[1] / 2) - self.mallet_radius - ws_offset_table,
                ),
            )
        )
        self.puck_table_minmax = numpy.array(
            (
                (
                    numpy.abs(self.robot_base_frame[0][0, 3])
                    - (self.table_size[0] / 2)
                    + self.puck_radius,
                    numpy.abs(self.robot_base_frame[0][0, 3])
                    + (self.table_size[0] / 2)
                    - self.puck_radius,
                ),
                (
                    -(self.table_size[1] / 2) + self.puck_radius,
                    (self.table_size[1] / 2) - self.puck_radius,
                ),
            )
        )


def _normalize(
    value: numpy.ndarray,
    low_in: float | numpy.ndarray,
    high_in: float | numpy.ndarray,
) -> numpy.ndarray:
    return 2.0 * (value - low_in) / (high_in - low_in) - 1.0


def _unnormalize(
    value: numpy.ndarray,
    low_out: float | numpy.ndarray,
    high_out: float | numpy.ndarray,
) -> numpy.ndarray:
    return (high_out - low_out) * (value + 1.0) / 2.0 + low_out
