import os
import time
from sys import getswitchinterval as gsi
from sys import setswitchinterval as ssi
from threading import Condition, Lock, Thread
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

from drl_air_hockey.utils.config import INTERPOLATION_ORDER, config_dreamerv3
from drl_air_hockey.utils.env_wrapper import EmbodiedChallengeWrapper
from drl_air_hockey.utils.eval import PolicyEvalDriver
from drl_air_hockey.utils.task import Task as AirHockeyTask


class SpaceRAgent(AgentBase):
    DIR_MODELS: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")
    INFERENCE_MODEL: Dict[AirHockeyTask, str] = {
        AirHockeyTask.R7_HIT: os.path.join(DIR_MODELS, "hit.ckpt"),
        AirHockeyTask.R7_DEFEND: os.path.join(DIR_MODELS, "defend.ckpt"),
        AirHockeyTask.R7_PREPARE: os.path.join(DIR_MODELS, "prepare.ckpt"),
        AirHockeyTask.R7_TOURNAMENT: None,
    }

    ASYNC_INFERENCE: bool = False

    STEP_TIME_LIMIT: float = 0.02 - 0.002  # 20 ms (2 ms reserved for extra processing)

    # MAX_DISPLACEMENT_LIMIT_ENABLED: bool = False
    # MAX_DISPLACEMENT_PER_STEP: float = 0.25
    VEL_CONSTRAINTS_SCALING_FACTOR: float = 0.5
    FILTER_ACTIONS_ENABLED: bool = True
    FILTER_ACTIONS_COEFFICIENT: float = 0.05

    OPERATING_AREA_OFFSET_FROM_CENTRE: float = 0.15
    OPERATING_AREA_OFFSET_FROM_TABLE: float = 0.025

    # Lower is more strict (positive only)
    Z_POSITION_CONTROL_TOLERANCE: float = 0.5

    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        interpolation_order: Optional[int] = INTERPOLATION_ORDER,
        train: bool = False,
        scheme: int = 1,
        max_episode_steps: int = 1024,
        **kwargs,
    ):
        ## Chain up the parent implementation
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)

        ## Extract information about the environment and write it to members
        self.extract_env_info()

        ## Get information about the agent
        self.agent_id = agent_id
        self.interpolation_order = interpolation_order
        self.scheme = scheme
        if self.scheme not in [1, 2]:
            raise ValueError("Invalid scheme")
        elif self.scheme == 2:
            self.max_episode_steps = max_episode_steps

        ## For evaluation, the agent is fully internal and loaded from a checkpoint.
        self.evaluate = not train
        if self.evaluate:
            # Patch DreamerV3
            _apply_monkey_patch_dreamerv3()

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
            checkpoint.load(self.INFERENCE_MODEL[self.task], keys=["agent"])

            # Setup agent driver
            policy = lambda *args: self.agent.policy(*args, mode="eval")
            self.policy_driver = PolicyEvalDriver(policy=policy)

            self.initialize_inference()

            # Setup async inference
            if self.ASYNC_INFERENCE:
                self.mutex = Lock()
                self.cv_new_obs_avail = Condition(lock=self.mutex)
                self.cv_new_act_avail = Condition(lock=self.mutex)
                self.inference_in_progress = False
                self.new_obs_in_queue = False
                self.thread = Thread(target=self.inference_loop)
                self.thread.start()

            self._original_interval = gsi()
            self._inference_interval = self.STEP_TIME_LIMIT

        self.reset()

    @property
    def observation_space(self):
        if self.scheme == 1:
            n_obs = 7
        elif self.scheme == 2:
            n_obs = 10

        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_obs,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        # if self.interpolation_order in [1, 2]:
        # The desired XY position of the mallet
        #  - pos_x
        #  - pos_y
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    ##### Evaluation only #####

    def draw_action(self, obs: np.ndarray) -> np.ndarray:
        ssi(self._inference_interval)
        start_time = time.time()

        # Extract and normalize relevant observations
        obs = self.process_raw_obs(obs)

        ## Asynchronous inference
        if self.ASYNC_INFERENCE:
            # Notify the inference thread that new observation is available
            with self.cv_new_obs_avail:
                self.obs = obs
                if self.inference_in_progress:
                    self.new_obs_in_queue = True
                else:
                    self.cv_new_obs_avail.notify()

            # Wait for new action to be available or until time limit is reached
            with self.cv_new_act_avail:
                self.cv_new_act_avail.wait(
                    timeout=self.STEP_TIME_LIMIT - (time.time() - start_time)
                )
                act = np.copy(self.act)
        else:
            ## Synchronous inference
            act = self.infer_action(obs)

        ssi(self._original_interval)
        return act

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

        if self.FILTER_ACTIONS_ENABLED:
            self.previous_ee_pos_xy_norm = None

        if self.evaluate:
            self.policy_driver.reset()
            if self.ASYNC_INFERENCE:
                with self.cv_new_obs_avail:
                    self.cv_new_obs_avail.wait(timeout=self.STEP_TIME_LIMIT)
                with self.cv_new_act_avail:
                    self.cv_new_act_avail.wait(timeout=self.STEP_TIME_LIMIT)
                self.inference_in_progress = False
                self.new_obs_in_queue = False

        if self.scheme == 2:
            self.episode_step = 0
            self.previous_ee_pos_xy_norm = None
            self.previous_puck_pos_xy_norm = None

    #### ~Evaluation only~ ####

    ######### Common ##########

    def process_raw_obs(self, obs: np.ndarray) -> np.ndarray:
        ## Player's end-effector state
        self.current_ee_pos = self.get_ee_pose(obs)[0]
        ee_pos_xy_norm = self._normalize_value(
            self.current_ee_pos[:2],
            low_in=self.ee_table_minmax[:, 0],
            high_in=self.ee_table_minmax[:, 1],
        )
        self.current_ee_pos_xy_norm = ee_pos_xy_norm

        ## Player's joint states
        # Note: Not used in observation vector, but used for action processing
        self.current_joint_pos = self.get_joint_pos(obs)

        ## Puck's state
        # Position
        puck_pos_xy_norm = self._normalize_value(
            self.get_puck_pos(obs)[:2],
            low_in=self.puck_table_minmax[:, 0],
            high_in=self.puck_table_minmax[:, 1],
        )
        # Velocity (linear and angular)
        puck_vel_xy_theta: np.ndarray = self.get_puck_vel(obs)
        puck_vel_xy_theta = self._normalize_value(
            puck_vel_xy_theta,
            low_in=np.array(
                [-2 * self.table_size[0], -2 * self.table_size[1], -8 * np.pi]
            ),
            high_in=np.array(
                [2 * self.table_size[0], 2 * self.table_size[1], 8 * np.pi]
            ),
        )

        # Form the observation vector
        if self.scheme == 1:
            obs = np.clip(
                np.concatenate(
                    (
                        ee_pos_xy_norm,
                        puck_pos_xy_norm,
                        puck_vel_xy_theta,
                    )
                ),
                -1.0,
                1.0,
            )
        elif self.scheme == 2:
            # Keep track of episode progress to provide the agent with information about time
            episode_progress = self.episode_step / self.max_episode_steps
            self.episode_step += 1

            # For the first step, use previous position of ee and derived position of puck based on its velocity
            if self.previous_ee_pos_xy_norm is None:
                self.previous_ee_pos_xy_norm = ee_pos_xy_norm
                self.previous_puck_pos_xy_norm = (
                    puck_pos_xy_norm - self.sim_dt * puck_vel_xy_theta[:2]
                )

            # Use only the z-rotation of the puck's velocity from observations
            puck_vel_z_rot = puck_vel_xy_theta[2]

            obs = np.clip(
                np.concatenate(
                    (
                        [episode_progress],
                        self.previous_ee_pos_xy_norm,
                        ee_pos_xy_norm,
                        self.previous_puck_pos_xy_norm,
                        puck_pos_xy_norm,
                        [puck_vel_z_rot],
                    )
                ),
                -1.0,
                1.0,
            )

            # Update previous positions
            self.previous_ee_pos_xy_norm = ee_pos_xy_norm.copy()
            self.previous_puck_pos_xy_norm = puck_pos_xy_norm.copy()

        assert obs.shape == self.observation_space.shape
        # assert max(obs) <= 1.0 and min(obs) >= -1.0

        return obs

    def process_raw_act(self, action: np.ndarray) -> np.ndarray:
        assert self.interpolation_order in [-1, 1, 2, 3, 4]
        assert action.shape == self.action_space.shape
        assert max(action) <= 1.0 and min(action) >= -1.0

        # # Limit the displacement to a pre-defined maximum for each step
        # if MAX_DISPLACEMENT_LIMIT_ENABLED:
        #     target_ee_disp_xy = action[:2] - self.current_ee_pos_xy_norm[:2]
        #     disp_xy_norm = np.linalg.norm(target_ee_disp_xy)
        #     if disp_xy_norm > self.MAX_DISPLACEMENT_PER_STEP:
        #         target_ee_disp_xy *= self.MAX_DISPLACEMENT_PER_STEP / disp_xy_norm
        #     target_ee_pos_xy = self.current_ee_pos_xy_norm[:2] + target_ee_disp_xy
        # else:
        #     target_ee_pos_xy = action[:2]
        target_ee_pos_xy = action[:2]

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
            dtype=np.float32,
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
        s[2] *= self.Z_POSITION_CONTROL_TOLERANCE
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
            self.VEL_CONSTRAINTS_SCALING_FACTOR * self.robot_joint_vel_limit
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
                    + self.OPERATING_AREA_OFFSET_FROM_TABLE,
                    np.abs(self.robot_base_frame[0][0, 3])
                    - self.mallet_radius
                    - self.OPERATING_AREA_OFFSET_FROM_CENTRE,
                ],
                [
                    -(self.table_size[1] / 2)
                    + self.mallet_radius
                    + self.OPERATING_AREA_OFFSET_FROM_TABLE,
                    (self.table_size[1] / 2)
                    - self.mallet_radius
                    - self.OPERATING_AREA_OFFSET_FROM_TABLE,
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

    ####### ~Utilities~ #######


def _apply_monkey_patch_dreamerv3():
    ## MONKEY PATCH: Reduce preallocated JAX memory
    __monkey_patch__setup_original = dreamerv3.Agent._setup

    def __monkey_patch__setup(self):
        __monkey_patch__setup_original(self)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    dreamerv3.Agent._setup = __monkey_patch__setup
    ## ~MONKEY PATCH:  Reduce preallocated JAX memory

    ## MONKEY PATCH: Speed up initialization for inference
    def __monkey_patch__init_varibs(self, obs_space, act_space):
        rng = self._next_rngs(self.train_devices, mirror=True)
        obs = self._dummy_batch(obs_space, (1,))
        state, varibs = self._init_policy({}, rng, obs["is_first"])
        varibs = self._policy(varibs, rng, obs, state, mode="eval", init_only=True)
        return varibs

    dreamerv3.jaxagent.JAXAgent._init_varibs = __monkey_patch__init_varibs
    ## ~MONKEY PATCH: Speed up initialization for inference
