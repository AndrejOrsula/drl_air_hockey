from collections import deque
from os import nice, path
from typing import Any, Dict, Optional, Tuple

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
from drl_air_hockey.utils.tournament_agent_strategies import (
    AggressiveAgentStrategy,
    DefensiveAgentStrategy,
    OffensiveAgentStrategy,
    SneakyAgentStrategy,
    strategy_to_str,
)


class MultiStrategySpaceRAgent(AgentBase):
    # Dictionary of paths to inference models for each strategy
    INFERENCE_MODELS: Dict[str, str] = {
        "aggressive": path.join(DIR_MODELS, "tournament_aggressive.ckpt"),
        "offensive": path.join(DIR_MODELS, "tournament_offensive.ckpt"),
        "sneaky": path.join(DIR_MODELS, "tournament_sneaky.ckpt"),
        "defensive": path.join(DIR_MODELS, "tournament_defensive.ckpt"),
    }

    # Initial strategy to use
    INITIAL_STRATEGY = "aggressive"

    # Maximum number of steps in a game
    N_STEPS_GAME: int = 45000
    # Number of steps at the beginning of the game meant to explore the best strategy
    N_STEPS_EARLY_GAME_END: int = 9000
    # Number of steps at the end of the game meant to stick to the best strategy
    N_STEPS_LATE_GAME_START: int = 36000

    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        **kwargs,
    ):
        # Patch DreamerV3
        _apply_monkey_patch_dreamerv3()

        ## Make things nice
        try:
            nice(69)
        except Exception:
            pass

        ## Chain up the parent implementation
        AgentBase.__init__(self, env_info=env_info, agent_id=agent_id, **kwargs)

        ## Extract information about the environment and write it to members
        self.extract_env_info()
        self.derive_env_info()

        ## Initialize all agent models
        self.init_agents()

        ## Initialize observation and action schemes
        self.init_observation_scheme()
        self.init_action_scheme()

        ## Initialize computation of game metrics
        # TODO: Implement
        self.init_game_metrics()

        ## Initialize algorithm for strategy switching
        # TODO: Implement
        self.init_strategy_switching()

        self.reset()

    def init_agents(self):
        # Aggressive (Offensive/Fast)
        self.policy_aggressive = self.init_agent(self.INFERENCE_MODELS["aggressive"])

        # Offensive (Offensive/Normal)
        self.policy_offensive = self.init_agent(self.INFERENCE_MODELS["offensive"])

        # Sneaky (Defensive/Exploit)
        self.policy_sneaky = self.init_agent(self.INFERENCE_MODELS["sneaky"])

        # Defensive (Defensive/Reset)
        self.policy_defensive = self.init_agent(self.INFERENCE_MODELS["defensive"])

        # Dictionary of available policies
        self.policies = {
            "aggressive": self.policy_aggressive,
            "offensive": self.policy_offensive,
            "sneaky": self.policy_sneaky,
            "defensive": self.policy_defensive,
        }

    def init_agent(self, model_path: str) -> PolicyEvalDriver:
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
        self.n_stacked_obs = 4
        self.stacked_obs_participant_ee_pos = deque([], maxlen=self.n_stacked_obs)
        self.stacked_obs_opponent_ee_pos = deque([], maxlen=self.n_stacked_obs)
        self.stacked_obs_puck_pos = deque([], maxlen=self.n_stacked_obs)

        ## Number of joints contained in the position vector of agent's joints
        self.n_joints = 7

    def init_action_scheme(self):
        self.robot_joint_vel_limit_scaled = {}
        self.ee_table_minmax = {}
        self.z_position_control_tolerance = {}

        for strategy in [
            AggressiveAgentStrategy(),
            OffensiveAgentStrategy(),
            SneakyAgentStrategy(),
            DefensiveAgentStrategy(),
        ]:
            strategy_name = strategy_to_str(strategy)
            strategy_kwargs = strategy.get_env_kwargs()

            self.robot_joint_vel_limit_scaled[strategy_name] = (
                strategy_kwargs["vel_constraints_scaling_factor"]
                * self.robot_joint_vel_limit
            )
            self.ee_table_minmax[strategy_name] = self.compute_ee_table_minmax(
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
            self.z_position_control_tolerance[strategy_name] = strategy_kwargs[
                "z_position_control_tolerance"
            ]

    def init_strategy_switching(self):
        self.current_strategy = self.INITIAL_STRATEGY
        self.policy = self.policies[self.current_strategy]

        # Flag to determine if the game is still early and there is time to explore strategy
        self.is_early_game = True
        # Flag to determine if the game is late and we should stick to the best/risky strategy
        self.is_late_game = False

        # Determines if the opponent is known to be susceptible to fauls
        self.is_opponent_susceptible_to_faul: Optional[bool] = None

        # Determines if the next puck is initialized on our side
        # (known due to goal or faul | unknown at beginning and when the puck gets stuck in the middle)
        self.is_next_puck_start_on_our_side: Optional[bool] = None

    def init_game_metrics(self):
        # Counter for the entire game (up to 45000 steps)
        self.step_counter = 0
        # Counter for a single episode used in penalty computation (resets every 500 steps)
        self.episode_counter = 0
        # Number of total exchanges from player's perspective
        self.n_exchanges = 0

        # Score (player, opponent)
        self.score: Tuple[int, int] = (0, 0)
        # Number of scored goals (player, opponent)
        self.goals: Tuple[int, int] = (0, 0)
        # Number of committed fauls (player, opponent)
        self.fauls: Tuple[int, int] = (0, 0)

        # Number of resets due to goal
        self.n_resets_due_to_goal = 0
        # Number of resets due to faul
        self.n_resets_due_to_faul = 0
        # Number of resets due to stuck puck
        self.n_resets_due_to_stuck_puck = 0

        ## Estimators of penalties
        self.n_penalties_ee_pos_player = 0
        self.n_penalties_ee_pos_opponent = 0
        self.n_penalties_joint_vel_player = 0
        self.penalty_occured_this_episode_ee_pos_player = False
        self.penalty_occured_this_episode_ee_pos_opponent = False
        self.penalty_occured_this_episode_joint_vel_player = False

    ### Checked until this point

    @property
    def observation_space(self):
        # n_obs = 1 + self.n_joints + 6 * self.n_stacked_obs
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(32,),
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
        for policy in self.policies.values():
            policy.reset()

        self.penalty_side = None
        self.penalty_timer = 0.0

        self.stacked_obs_participant_ee_pos.clear()
        self.stacked_obs_opponent_ee_pos.clear()
        self.stacked_obs_puck_pos.clear()

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
        opponent_ee_pos_xy_norm = np.clip(
            self._normalize_value(
                self.get_opponent_ee_pos(obs)[:2],
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )

        # Puck's position
        puck_pos_xy_norm = np.clip(
            self._normalize_value(
                self.get_puck_pos(obs)[:2],
                low_in=self.puck_table_minmax[:, 0],
                high_in=self.puck_table_minmax[:, 1],
            ),
            -1.0,
            1.0,
        )

        ## Compute penalty timer (this is the only difference from scheme 2)
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
                np.tile(puck_pos_xy_norm, (self.n_stacked_obs, 1))
            )
            self.stacked_obs_participant_ee_pos.extend(
                np.tile(ee_pos_xy_norm, (self.n_stacked_obs, 1))
            )
            self.stacked_obs_opponent_ee_pos.extend(
                np.tile(opponent_ee_pos_xy_norm, (self.n_stacked_obs, 1))
            )
            self._new_episode = False
        else:
            self.stacked_obs_puck_pos.append(puck_pos_xy_norm)
            self.stacked_obs_participant_ee_pos.append(ee_pos_xy_norm)
            self.stacked_obs_opponent_ee_pos.append(opponent_ee_pos_xy_norm)

        # Concaternate episode progress with all temporally-stacked observations
        obs = np.concatenate(
            (
                np.array((current_penalty_timer,)),
                current_joint_pos_normalized,
                np.array(self.stacked_obs_puck_pos).flatten(),
                np.array(self.stacked_obs_participant_ee_pos).flatten(),
                np.array(self.stacked_obs_opponent_ee_pos).flatten(),
            )
        )

        return obs

    def process_raw_act(self, action: np.ndarray) -> np.ndarray:
        # Unnormalize the action and combine with desired height
        target_ee_pos = np.array(
            [
                *self._unnormalize_value(
                    action,
                    low_out=self.ee_table_minmax[self.current_strategy][:, 0],
                    high_out=self.ee_table_minmax[self.current_strategy][:, 1],
                ),
                self.robot_ee_desired_height,
            ],
            dtype=action.dtype,
        )

        # Calculate the target joint disp via Inverse Jacobian method
        target_ee_disp = target_ee_pos - self.current_ee_pos
        jac = self.jacobian(self.current_joint_pos)[:3]
        jac_pinv = np.linalg.pinv(jac)
        s = np.linalg.svd(jac, compute_uv=False)
        s[:2] = np.mean(s[:2])
        s[2] *= self.z_position_control_tolerance[self.current_strategy]
        s = 1 / s
        s = s / np.sum(s)
        joint_disp = jac_pinv * target_ee_disp
        joint_disp = np.average(joint_disp, axis=1, weights=s)

        # Convert to joint velocities based on joint displacements
        joint_vel = joint_disp / self.sim_dt

        # Limit the joint velocities to the maximum allowed
        joints_below_vel_limit = (
            joint_vel < self.robot_joint_vel_limit_scaled[self.current_strategy][0, :]
        )
        joints_above_vel_limit = (
            joint_vel > self.robot_joint_vel_limit_scaled[self.current_strategy][1, :]
        )
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
                            - self.robot_joint_vel_limit_scaled[self.current_strategy][
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
        action = np.array([joint_pos, joint_vel])

        return action

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

    def get_joint_pos(self, obs):
        return obs[self.env_info["joint_pos_ids"]]

    def get_ee_pose(self, obs):
        return self.forward_kinematics(self.get_joint_pos(obs))

    def get_opponent_ee_pos(self, obs):
        return obs[self.env_info["opponent_ee_ids"]]


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
