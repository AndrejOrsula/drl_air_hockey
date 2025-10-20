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
        joint_vel_limit_scaling_factor: float = 0.6,
        ws_offset_centre: float = 0.2,
        ws_offset_table: float = 0.005,
        ws_offset_goal: float = 0.0075,
        dls_lambda: float = 0.05,
        osc_stiffness_xy_range: Tuple[float, float] = (2.0, 25.0),
        osc_damping_xy_range: Tuple[float, float] = (0.0, 0.5),
        osc_stiffness_z: float = 40.0,
        osc_damping_z: float = 0.8,
        ## Training noise params
        train_noise_std_action_ee_pos: float = 0.001,
        train_noise_std_action_joint_pos: float = 0.0025 * numpy.pi / 180.0,
        train_noise_std_action_joint_vel: float = 0.0005 * numpy.pi / 180.0,
        train_noise_std_joint_pos_episode: float = 0.15 * numpy.pi / 180.0,
        train_noise_std_joint_pos: float = 0.1 * numpy.pi / 180.0,
        train_noise_std_joint_vel: float = 0.025 * numpy.pi / 180.0,
        train_noise_std_puck_pos_episode: float = 0.009,
        train_noise_std_puck_pos: float = 0.006,
        train_noise_std_puck_vel: float = 0.0015,
        ## Training latency
        train_action_delay_steps: Tuple[int, int] = (0, 1),
        train_proprio_observation_delay_steps: Tuple[int, int] = (0, 1),
        train_extero_observation_delay_steps: Tuple[int, int] = (0, 4),
        **kwargs,
    ):
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)
        self.extract_env_info(
            joint_vel_limit_scaling_factor=joint_vel_limit_scaling_factor,
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

        if self.train:
            self.train_noise_std_action_ee_pos = train_noise_std_action_ee_pos
            self.train_noise_std_action_joint_pos = train_noise_std_action_joint_pos
            self.train_noise_std_action_joint_vel = train_noise_std_action_joint_vel
            self.train_noise_std_joint_pos = train_noise_std_joint_pos
            self.train_noise_std_joint_pos_episode = train_noise_std_joint_pos_episode
            self.train_noise_std_joint_vel = train_noise_std_joint_vel
            self.train_noise_std_puck_pos = train_noise_std_puck_pos
            self.train_noise_std_puck_pos_episode = train_noise_std_puck_pos_episode
            self.train_noise_std_puck_vel = train_noise_std_puck_vel
            (
                self.train_min_action_delay,
                self.train_max_action_delay,
            ) = train_action_delay_steps
            (
                self.train_min_proprio_obs_delay,
                self.train_max_proprio_obs_delay,
            ) = train_proprio_observation_delay_steps
            (
                self.train_min_extero_obs_delay,
                self.train_max_extero_obs_delay,
            ) = train_extero_observation_delay_steps
            self.train_is_latency_initialized = False
            self.train_action_history_buffer = None
            self.train_proprio_obs_history_buffer = None
            self.train_extero_obs_history_buffer = None

        # self.penalty_side = 0
        # self.penalty_timer = 0.0
        self.is_af_initialized = False
        self.af_bw_b, self.af_bw_a = butter(  # type: ignore
            4, 0.08 / self.sim_dt, fs=1.0 / self.sim_dt, btype="low"
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
        # self.penalty_side = 0
        # self.penalty_timer = 0.0
        self.is_af_initialized = False
        if self.train:
            self.train_is_latency_initialized = False
            self.train_joint_pos_episode_noise = numpy.random.normal(
                0.0, self.train_noise_std_joint_pos_episode, size=7
            )
            self.train_puck_pos_episode_noise = numpy.random.normal(
                0.0, self.train_noise_std_puck_pos_episode, size=2
            )

    def process_raw_act(self, action: numpy.ndarray) -> numpy.ndarray:
        action = numpy.clip(action, -1.0, 1.0)
        # Filter actions
        if not self.is_af_initialized:
            self.af_bw_zi_x = lfilter_zi(self.af_bw_b, self.af_bw_a) * action[0]
            self.af_bw_zi_y = lfilter_zi(self.af_bw_b, self.af_bw_a) * action[1]
            self.is_af_initialized = True
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

        # Apply action noise if training
        if self.train:
            target_ee_pos[:2] += numpy.random.normal(
                0.0, self.train_noise_std_action_ee_pos, size=2
            )

        # Control
        error_pos = target_ee_pos - self.current_ee_pos
        error_vel = -(jacobian_lin @ self.current_joint_vel)
        target_ee_vel = osc_stiffness * error_pos + osc_damping * error_vel
        joint_vel = jacobian_inv @ target_ee_vel

        # Velocity limiting
        is_below_limit = joint_vel < self.robot_joint_vel_limit_scaled[0, :]
        is_above_limit = joint_vel > self.robot_joint_vel_limit_scaled[1, :]
        if numpy.any(is_below_limit) or numpy.any(is_above_limit):
            downscaling_factor = 1.0
            for joint_i in numpy.where(is_below_limit | is_above_limit)[0]:
                limit = (
                    self.robot_joint_vel_limit_scaled[1, joint_i]
                    if is_above_limit[joint_i]
                    else self.robot_joint_vel_limit_scaled[0, joint_i]
                )
                if joint_vel[joint_i] != 0.0:
                    downscaling_factor = min(
                        downscaling_factor, abs(limit / joint_vel[joint_i])
                    )
            joint_vel *= downscaling_factor

        joint_pos = self.current_joint_pos + (self.sim_dt * joint_vel)

        if self.train:
            joint_pos += numpy.random.normal(
                0.0, self.train_noise_std_action_joint_pos, size=7
            )
            joint_vel += numpy.random.normal(
                0.0, self.train_noise_std_action_joint_vel, size=7
            )

        if self.train and self.train_action_history_buffer is not None:
            self.train_action_history_buffer[
                self.train_action_history_buffer_ptr
            ] = numpy.array((joint_pos, joint_vel))
            delayed_action = self.train_action_history_buffer[
                (
                    self.train_action_history_buffer_ptr
                    - self.train_action_delay
                    + self.train_max_action_delay
                )
                % self.train_max_action_delay
            ]
            self.train_action_history_buffer_ptr = (
                self.train_action_history_buffer_ptr + 1
            ) % self.train_max_action_delay
            return delayed_action
        else:
            return numpy.array((joint_pos, joint_vel))

    def process_raw_obs(self, obs: numpy.ndarray) -> numpy.ndarray:
        if self.train:
            if not self.train_is_latency_initialized:
                self._initialize_latency_buffers(obs)
            obs = self._get_delayed_observation(obs)

        self.current_joint_pos = obs[self.env_info["joint_pos_ids"]]
        self.current_joint_vel = obs[self.env_info["joint_vel_ids"]]
        puck_pos = obs[self.env_info["puck_pos_ids"]][:2]
        puck_vel = obs[self.env_info["puck_vel_ids"]][:2]

        # Apply observation noise if training
        if self.train:
            self.current_joint_pos += (
                self.train_joint_pos_episode_noise
                + numpy.random.normal(0.0, self.train_noise_std_joint_pos, size=7)
            )
            self.current_joint_vel += numpy.random.normal(
                0.0, self.train_noise_std_joint_vel, size=7
            )
            puck_pos += self.train_puck_pos_episode_noise + numpy.random.normal(
                0.0, self.train_noise_std_puck_pos, size=2
            )
            puck_vel += numpy.random.normal(0.0, self.train_noise_std_puck_vel, size=2)

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

        # # Update penalty timer
        # puck_pos_x_sign = numpy.sign(puck_pos_xy_norm[0])
        # if self.penalty_side == 0:
        #     self.penalty_side = puck_pos_x_sign if puck_pos_x_sign != 0 else 1
        # elif puck_pos_x_sign == self.penalty_side:
        #     self.penalty_timer += self.sim_dt
        # else:
        #     self.penalty_side *= -1
        #     self.penalty_timer = 0.0
        # current_penalty_timer = (
        #     self.penalty_side * self.penalty_timer / MAX_TIME_UNTIL_PENALTY_S
        # )

        return numpy.concatenate(
            (
                # (current_penalty_timer,),
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
        ws_offset_centre: float,
        ws_offset_table: float,
        ws_offset_goal: float,
    ):
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
        self.sim_dt = self.env_info["dt"]
        self.robot_joint_vel_limit_scaled = (
            joint_vel_limit_scaling_factor * self.robot_joint_vel_limit
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

    def _initialize_latency_buffers(self, initial_obs: numpy.ndarray):
        # Randomize delay for this episode
        self.train_action_delay = numpy.random.randint(
            self.train_min_action_delay, self.train_max_action_delay + 1
        )
        self.train_proprio_delay = numpy.random.randint(
            self.train_min_proprio_obs_delay, self.train_max_proprio_obs_delay + 1
        )
        self.train_extero_delay = numpy.random.randint(
            self.train_min_extero_obs_delay, self.train_max_extero_obs_delay + 1
        )

        # Initialize action buffer
        initial_joint_pos = initial_obs[self.env_info["joint_pos_ids"]]
        if self.train_max_action_delay > 0:
            self.train_action_history_buffer_ptr = 0
            self.train_action_history_buffer = numpy.tile(
                numpy.array((initial_joint_pos, numpy.zeros_like(initial_joint_pos))),
                (self.train_max_action_delay, 1, 1),
            )

        # Initialize observation buffers with the first observation
        if self.train_max_proprio_obs_delay > 0:
            self.train_proprio_obs_history_buffer_ptr = 0
            proprio_obs = numpy.concatenate(
                (
                    initial_joint_pos,
                    initial_obs[self.env_info["joint_vel_ids"]],
                )
            )
            self.train_proprio_obs_history_buffer = numpy.tile(
                proprio_obs, (self.train_max_proprio_obs_delay, 1)
            )

        if self.train_max_extero_obs_delay > 0:
            self.train_extero_obs_history_buffer_ptr = 0
            extero_obs = numpy.concatenate(
                (
                    initial_obs[self.env_info["puck_pos_ids"]],
                    initial_obs[self.env_info["puck_vel_ids"]],
                )
            )
            self.train_extero_obs_history_buffer = numpy.tile(
                extero_obs, (self.train_max_extero_obs_delay, 1)
            )

        self.train_is_latency_initialized = True

    def _get_delayed_observation(self, obs: numpy.ndarray) -> numpy.ndarray:
        delayed_obs = obs.copy()

        # Proprioceptive delay
        if self.train_proprio_obs_history_buffer is not None:
            current_proprio = numpy.concatenate(
                (
                    obs[self.env_info["joint_pos_ids"]],
                    obs[self.env_info["joint_vel_ids"]],
                )
            )
            self.train_proprio_obs_history_buffer[
                self.train_proprio_obs_history_buffer_ptr
            ] = current_proprio
            read_idx = (
                self.train_proprio_obs_history_buffer_ptr
                - self.train_proprio_delay
                + self.train_max_proprio_obs_delay
            ) % self.train_max_proprio_obs_delay
            delayed_proprio = self.train_proprio_obs_history_buffer[read_idx]
            delayed_obs[self.env_info["joint_pos_ids"]] = delayed_proprio[:7]
            delayed_obs[self.env_info["joint_vel_ids"]] = delayed_proprio[7:]
            self.train_proprio_obs_history_buffer_ptr = (
                self.train_proprio_obs_history_buffer_ptr + 1
            ) % self.train_max_proprio_obs_delay

        # Exteroceptive delay
        if self.train_extero_obs_history_buffer is not None:
            current_extero = numpy.concatenate(
                (
                    obs[self.env_info["puck_pos_ids"]],
                    obs[self.env_info["puck_vel_ids"]],
                )
            )
            self.train_extero_obs_history_buffer[
                self.train_extero_obs_history_buffer_ptr
            ] = current_extero
            read_idx = (
                self.train_extero_obs_history_buffer_ptr
                - self.train_extero_delay
                + self.train_max_extero_obs_delay
            ) % self.train_max_extero_obs_delay
            delayed_extero = self.train_extero_obs_history_buffer[read_idx]
            delayed_obs[self.env_info["puck_pos_ids"]] = delayed_extero[:3]
            delayed_obs[self.env_info["puck_vel_ids"]] = delayed_extero[3:]
            self.train_extero_obs_history_buffer_ptr = (
                self.train_extero_obs_history_buffer_ptr + 1
            ) % self.train_max_extero_obs_delay

        return delayed_obs


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
