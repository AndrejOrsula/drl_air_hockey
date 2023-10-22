from __future__ import annotations

import enum
from collections import deque
from os import nice, path
from sys import getswitchinterval as gsi
from sys import setswitchinterval as ssi
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


@enum.unique
class AgentStrategy(enum.Enum):
    AGGRESSIVE = enum.auto()
    OFFENSIVE = enum.auto()
    SNEAKY = enum.auto()
    DEFENSIVE = enum.auto()

    def to_str(self) -> str:
        if self == AgentStrategy.AGGRESSIVE:
            return "aggressive"
        elif self == AgentStrategy.OFFENSIVE:
            return "offensive"
        elif self == AgentStrategy.SNEAKY:
            return "sneaky"
        elif self == AgentStrategy.DEFENSIVE:
            return "defensive"

    @classmethod
    def from_str(cls, strategy: str) -> AgentStrategy:
        if strategy == "aggressive":
            return AgentStrategy.AGGRESSIVE
        elif strategy == "offensive":
            return AgentStrategy.OFFENSIVE
        elif strategy == "sneaky":
            return AgentStrategy.SNEAKY
        elif strategy == "defensive":
            return AgentStrategy.DEFENSIVE


class MultiStrategySpaceRAgent(AgentBase):
    # Dictionary of paths to inference models for each strategy
    INFERENCE_MODELS: Dict[AgentStrategy, str] = {
        AgentStrategy.AGGRESSIVE: path.join(DIR_MODELS, "tournament_aggressive.ckpt"),
        AgentStrategy.OFFENSIVE: path.join(DIR_MODELS, "tournament_offensive.ckpt"),
        AgentStrategy.SNEAKY: path.join(DIR_MODELS, "tournament_sneaky.ckpt"),
        AgentStrategy.DEFENSIVE: path.join(DIR_MODELS, "tournament_defensive.ckpt"),
    }

    # Initial strategy to use
    INITIAL_STRATEGY: AgentStrategy = AgentStrategy.OFFENSIVE

    # Maximum number of steps in a game
    N_STEPS_GAME: int = 45000
    # Number of steps in a single episode (used for penalty computation)
    N_EPISODE_STEPS: int = 500
    # Number of steps at the beginning of the game meant to explore the best strategy
    N_STEPS_EARLY_GAME_END: int = 1500

    # Maximum number of penalty points per game before getting disqualified
    MAX_PENALTY_POINTS: float = 135.0
    # Number of penalty points before disabling the aggressive strategy
    MAX_PENALTY_SAFETY_THRESHOLD_FIRST: float = 90.0
    # Number of penalty points before switching to safe strategy
    MAX_PENALTY_SAFETY_THRESHOLD_FINAL: float = 120.0

    # Maximum time until faul (in seconds), used for detection of fauls
    PENALTY_THRESHOLD: float = 0.95 * MAX_TIME_UNTIL_PENALTY_S

    # Exponential moving average coefficient for filtering the actions of the agent
    FILTER_ACTIONS_COEFFICIENT: float = 0.1

    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        **kwargs,
    ):
        ## Patch DreamerV3
        _apply_monkey_patch_dreamerv3()

        ## Make things nice
        try:
            nice(69)
        except:
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
        self.init_game_metrics()

        ## Initialize algorithm for strategy switching
        self.init_strategy_switching()

        ## Trigger the first reset
        self.reset()

        ## Get the original switch interval
        self._original_interval = gsi()

    def init_agents(self):
        # Dictionary of available policies
        self.policies = {
            # Aggressive (Offensive/Fast)
            AgentStrategy.AGGRESSIVE: self._init_agent(
                self.INFERENCE_MODELS[AgentStrategy.AGGRESSIVE]
            ),
            # Offensive (Offensive/Normal)
            AgentStrategy.OFFENSIVE: self._init_agent(
                self.INFERENCE_MODELS[AgentStrategy.OFFENSIVE]
            ),
            # Sneaky (Defensive/Exploit)
            AgentStrategy.SNEAKY: self._init_agent(
                self.INFERENCE_MODELS[AgentStrategy.SNEAKY]
            ),
            # Defensive (Defensive/Normal)
            AgentStrategy.DEFENSIVE: self._init_agent(
                self.INFERENCE_MODELS[AgentStrategy.DEFENSIVE]
            ),
        }

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

        for strategy in [
            AggressiveAgentStrategy(),
            OffensiveAgentStrategy(),
            SneakyAgentStrategy(),
            DefensiveAgentStrategy(),
        ]:
            strategy_variant = AgentStrategy.from_str(strategy_to_str(strategy))
            strategy_kwargs = strategy.get_env_kwargs()

            self.robot_joint_vel_limit_scaled[strategy_variant] = (
                strategy_kwargs["vel_constraints_scaling_factor"]
                * self.robot_joint_vel_limit
            )
            self.ee_table_minmax[strategy_variant] = self.compute_ee_table_minmax(
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
            self.z_position_control_tolerance[strategy_variant] = strategy_kwargs[
                "z_position_control_tolerance"
            ]

    def init_strategy_switching(self):
        # The current strategy
        self.current_strategy: AgentStrategy = self.INITIAL_STRATEGY

        # The next strategy that should be used
        self.next_strategy: Optional[AgentStrategy] = None
        # Flag to determine if the policy should be changed as soon as possible
        self.should_change_policy: bool = False

        # Flag to determine if the game is still early and there is time to explore strategy
        self.is_early_game = True

        # Determines if the opponent is known to be susceptible to fauls
        self.is_opponent_susceptible_to_faul: bool = True
        # Determines if the opponent is assumed to be disqualified
        self.is_opponent_dq: bool = False

    def init_game_metrics(self):
        # Counter for the entire game (up to 45000 steps)
        self.step_counter: int = 0
        # Number of total exchanges
        self.n_exchanges: Dict[str, int] = {
            AgentStrategy.AGGRESSIVE: 0,
            AgentStrategy.OFFENSIVE: 0,
            AgentStrategy.SNEAKY: 0,
            AgentStrategy.DEFENSIVE: 0,
        }

        # Score (player, opponent)
        self.score: Tuple[int, int] = (0, 0)
        # Number of scored goals (player, opponent)
        self.goals: Tuple[int, int] = (0, 0)
        # Number of committed fauls (player, opponent)
        self.fauls: Tuple[int, int] = (0, 0)

        # Variables used for penalty computation (initialization)
        self.last_puck_pos_xy = np.zeros(2, dtype=np.float64)

        ## Estimators of penalties
        self.penalty_points_estimate_player: float = 0.0
        self.penalty_points_player_exceeded_threshold_first: bool = False
        self.penalty_points_player_exceeded_threshold_last: bool = False
        self.penalty_points_expectation_per_episode: Dict[str, float] = {
            AgentStrategy.AGGRESSIVE: 4.5,
            AgentStrategy.OFFENSIVE: 0.5,
            AgentStrategy.SNEAKY: 0.25,
            AgentStrategy.DEFENSIVE: 0.2,
        }

        self.penalty_points_estimate_opponent: int = 0
        self.penalty_opponent_ee_position_this_episode: bool = False
        z_tolerance = 0.02
        self.penalty_ee_pos_x_ub = -self.env_info["robot"]["base_frame"][0][0, 3] + (
            self.env_info["table"]["length"] / 2 - self.env_info["mallet"]["radius"]
        )
        self.penalty_ee_pos_y_lb = -(
            self.env_info["table"]["width"] / 2 - self.env_info["mallet"]["radius"]
        )
        self.penalty_ee_pos_y_ub = (
            self.env_info["table"]["width"] / 2 - self.env_info["mallet"]["radius"]
        )
        self.penalty_ee_pos_z_lb = (
            self.env_info["robot"]["ee_desired_height"] - z_tolerance
        )
        self.penalty_ee_pos_z_ub = (
            self.env_info["robot"]["ee_desired_height"] + z_tolerance
        )

    ## Callbacks ##

    def episode_metrics_cb(self):
        self.penalty_points_estimate_player += (
            self.penalty_points_expectation_per_episode[self.current_strategy]
        )

        # Check if the player exceeded the penalty limits
        if self.penalty_points_player_exceeded_threshold_first:
            # Check if the last threshold was exceeded
            if (
                not self.penalty_points_player_exceeded_threshold_last
                and self.penalty_points_estimate_player
                > self.MAX_PENALTY_SAFETY_THRESHOLD_FINAL
            ):
                self.penalty_points_player_exceeded_threshold_last = True
                self.select_next_stragety_based_on_score()
        else:
            # Check if the first threshold was exceeded
            if (
                self.penalty_points_estimate_player
                > self.MAX_PENALTY_SAFETY_THRESHOLD_FIRST
            ):
                self.penalty_points_player_exceeded_threshold_first = True
                self.select_next_stragety_based_on_score()

        if self.penalty_opponent_ee_position_this_episode:
            self.penalty_points_estimate_opponent += 3
            self.penalty_opponent_ee_position_this_episode = False

        if self.is_early_game:
            if self.step_counter >= self.N_STEPS_EARLY_GAME_END:
                self.is_early_game = False
                self.is_opponent_susceptible_to_faul = (
                    self.determine_if_opponent_is_susceptible_to_faul()
                )
                self.select_next_stragety_based_on_score()

    def reset_metrics_cb(self):
        # Check if goal was scored, faul was committed or puck got stuck
        last_puck_pos_x_world_frame = self.last_puck_pos_xy[0] - np.abs(
            self.robot_base_frame[0][0, 3]
        )
        goal_tolerance = 0.075
        fault_tolerance = 0.05
        if (
            np.abs(self.last_puck_pos_xy[1])
            < (self.env_info["table"]["goal_width"] / 2 + goal_tolerance)
            and last_puck_pos_x_world_frame
            > (self.env_info["table"]["length"] / 2 - goal_tolerance)
            and self.last_puck_vel_xy[0] >= 0.0
        ):
            self.goals = (self.goals[0] + 1, self.goals[1])
        elif (
            np.abs(self.last_puck_pos_xy[1])
            < (self.env_info["table"]["goal_width"] / 2 + goal_tolerance)
            and last_puck_pos_x_world_frame
            < (-self.env_info["table"]["length"] / 2 + goal_tolerance)
            and self.last_puck_vel_xy[0] <= 0.0
        ):
            self.goals = (self.goals[0], self.goals[1] + 1)
        elif (
            self.penalty_timer > self.PENALTY_THRESHOLD
            and np.abs(last_puck_pos_x_world_frame) >= (0.15 - fault_tolerance)
            and self.penalty_side == -1
        ):
            self.fauls = (self.fauls[0] + 1, self.fauls[1])
        elif (
            self.penalty_timer > self.PENALTY_THRESHOLD
            and np.abs(last_puck_pos_x_world_frame) >= (0.15 - fault_tolerance)
            and self.penalty_side == 1
        ):
            self.fauls = (self.fauls[0], self.fauls[1] + 1)

        # Update score and check if it changed
        previous_score = self.score
        self.score = (
            self.goals[0] + self.fauls[1] // 3,
            self.goals[1] + self.fauls[0] // 3,
        )
        did_score_change = self.score != previous_score

        # Check again if opponent is susceptible to fauls
        self.is_opponent_susceptible_to_faul = (
            self.determine_if_opponent_is_susceptible_to_faul()
        )
        self.is_opponent_dq = self.determine_if_opponent_is_dq()

        # Update strategy based on updated score if not in early game
        if not self.is_early_game and did_score_change:
            self.select_next_stragety_based_on_score()

    def select_next_stragety_based_on_score(self):
        if (
            self.penalty_points_player_exceeded_threshold_last
            or self.is_opponent_dq
            or (self.score[0] - self.score[1] >= 4)
        ):
            # If the last penalty threshold was exceeded, always use defensive (safe) strategy
            # If the opponent is known to be almost certainly disqualified, always use defensive (safe) strategy
            # If winning by at least 4 points, always use defensive (safe) strategy
            self.select_next_strategy(AgentStrategy.DEFENSIVE)
        elif self.score[0] - self.score[1] >= 2:
            # If winning by at least 2 points
            # TODO: Consider removing sneaky strategy
            if self.is_opponent_susceptible_to_faul:
                # Use sneaky strategy if the opponent is susceptible to fauls
                self.select_next_strategy(AgentStrategy.SNEAKY)
            else:
                # Otherwise, use offensive strategy
                self.select_next_strategy(AgentStrategy.OFFENSIVE)
        else:
            # In all other cases (losing, stalemate, winning by 1 point)
            if self.penalty_points_player_exceeded_threshold_first:
                # If the first threshold was exceeded, use offensive (a bit safer) strategy
                self.select_next_strategy(AgentStrategy.OFFENSIVE)
            else:
                # Otherwise, use aggressive strategy
                self.select_next_strategy(AgentStrategy.AGGRESSIVE)

    def select_next_strategy(self, strategy: AgentStrategy):
        self.next_strategy = strategy
        self.should_change_policy = True

    def use_next_strategy_now(self):
        self.current_strategy = self.next_strategy
        self.should_change_policy = None
        self.next_strategy = None

    def is_safe_to_change_strategy(self) -> bool:
        last_puck_pos_x_world_frame = self.last_puck_pos_xy[0] - np.abs(
            self.robot_base_frame[0][0, 3]
        )
        return (
            np.abs(last_puck_pos_x_world_frame) < 0.15
            and self.last_puck_vel_xy[0] > 0.0
        ) or (
            last_puck_pos_x_world_frame < 0.15
            and np.linalg.norm(self.last_puck_vel_xy) < 0.025
        )

    def determine_if_opponent_is_susceptible_to_faul(self) -> bool:
        return (
            # OR | Has at least 1.5 estimated penalty point per elapsed episode
            self.penalty_points_estimate_opponent
            > 1.5 * (self.step_counter // self.N_EPISODE_STEPS)
            # OR | At least 25% of exchanges during sneaky strategy resulted in opponent fault
            or (self.fauls[1] > 0.25 * self.n_exchanges[AgentStrategy.SNEAKY])
            # OR | At least 5% of total exchanges resulted in opponent fault
            or self.fauls[1]
            > 0.05
            * (
                self.n_exchanges[AgentStrategy.AGGRESSIVE]
                + self.n_exchanges[AgentStrategy.OFFENSIVE]
                + self.n_exchanges[AgentStrategy.SNEAKY]
            )
        )

    def determine_if_opponent_is_dq(self) -> bool:
        return (
            # OR | Has exceeded the maximum penalty points
            self.penalty_points_estimate_opponent >= self.MAX_PENALTY_POINTS
            # OR | Has at least 65% of the maximum penalty points and at least 2 estimated penalty points per elapsed episode
            or (
                self.penalty_points_estimate_opponent >= 0.65 * self.MAX_PENALTY_POINTS
                and self.penalty_points_estimate_opponent
                > 2.0 * (self.step_counter // self.N_EPISODE_STEPS)
            )
        )

    def check_opponent_ee_pos_penalty(self, opponent_ee_pos: np.ndarray) -> bool:
        return (
            opponent_ee_pos[2] < self.penalty_ee_pos_z_lb
            or self.penalty_ee_pos_z_ub < opponent_ee_pos[2]
            or opponent_ee_pos[1] < self.penalty_ee_pos_y_lb
            or self.penalty_ee_pos_y_ub < opponent_ee_pos[1]
            or self.penalty_ee_pos_x_ub < opponent_ee_pos[0]
        )

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
        ssi(0.1)

        self.step_counter += 1
        if self.step_counter % self.N_EPISODE_STEPS == 0:
            self.episode_metrics_cb()

        processed_obs = self.process_raw_obs(obs)

        if self.should_change_policy and self.is_safe_to_change_strategy():
            self.use_next_strategy_now()

        action = self.process_raw_act(
            self.policies[self.current_strategy]
            .infer(processed_obs)["action"]
            .squeeze()
            .clip(-1.0, 1.0)
        )

        ssi(self._original_interval)
        return action

    def reset(self):
        if self.step_counter != 0:
            self.reset_metrics_cb()

        for policy in self.policies.values():
            policy.reset()

        if self.should_change_policy:
            self.use_next_strategy_now()

        self.penalty_side = None
        self.penalty_timer = 0.0

        self.stacked_obs_participant_ee_pos.clear()
        self.stacked_obs_opponent_ee_pos.clear()
        self.stacked_obs_puck_pos.clear()
        self.stacked_obs_puck_rot.clear()

        self.previous_target_ee_pos_xy_norm = None

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

        # Check if the opponent's end-effector position is in the penalty area
        if not self.penalty_opponent_ee_position_this_episode:
            self.penalty_opponent_ee_position_this_episode = (
                self.check_opponent_ee_pos_penalty(opponent_ee_pos)
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

        ## Compute penalty timer (this is the only difference from scheme 2)
        if self.penalty_side is None:
            self.penalty_side = np.sign(puck_pos_xy_norm[0])
        elif np.sign(puck_pos_xy_norm[0]) == self.penalty_side:
            self.penalty_timer += self.sim_dt
        else:
            self.penalty_side *= -1
            self.penalty_timer = 0.0
            self.n_exchanges[self.current_strategy] += 1
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
        if self.previous_target_ee_pos_xy_norm is None:
            self.previous_target_ee_pos_xy_norm = self.current_ee_pos_xy_norm
        target_ee_pos_xy = (
            self.FILTER_ACTIONS_COEFFICIENT * self.previous_target_ee_pos_xy_norm
            + (1 - self.FILTER_ACTIONS_COEFFICIENT) * target_ee_pos_xy
        )
        self.previous_target_ee_pos_xy_norm = target_ee_pos_xy

        # Unnormalize the action and combine with desired height
        target_ee_pos = np.array(
            [
                *self._unnormalize_value(
                    target_ee_pos_xy,
                    low_out=self.ee_table_minmax[self.current_strategy][:, 0],
                    high_out=self.ee_table_minmax[self.current_strategy][:, 1],
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
