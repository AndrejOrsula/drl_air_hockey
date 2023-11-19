from typing import Any, Dict

from drl_air_hockey.utils.rewards import TournamentReward


class AgentStrategy:
    def get_reward_function(self) -> TournamentReward:
        raise NotImplementedError

    def get_env_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError


class BalancedAgentStrategy(AgentStrategy):
    def get_reward_function(self) -> TournamentReward:
        # score goals (+2/3)  |  receive goal (-1.0), cause faul (-1/3)
        return TournamentReward(
            reward_agent_score_goal=+2.0 / 3.0,
            reward_opponent_faul=0.0,
            reward_agent_cause_puck_stuck=0.0,
            reward_agent_receive_goal=-1.0,
            reward_agent_faul=-1.0 / 3.0,
        )

    def get_env_kwargs(self) -> Dict[str, Any]:
        # regular
        return dict(
            vel_constraints_scaling_factor=0.5,
            operating_area_offset_from_centre=0.15,
            operating_area_offset_from_table=0.005,
            operating_area_offset_from_goal=0.01,
            z_position_control_tolerance=0.5,
        )


class AggressiveAgentStrategy(AgentStrategy):
    def get_reward_function(self) -> TournamentReward:
        # score goals (+1.0)  |  receive goal (-1.0), cause faul (-1/3)
        return TournamentReward(
            reward_agent_score_goal=+1.0,
            reward_opponent_faul=0.0,
            reward_agent_cause_puck_stuck=0.0,
            reward_agent_receive_goal=-1.0,
            reward_agent_faul=-1.0 / 3.0,
        )

    def get_env_kwargs(self) -> Dict[str, Any]:
        # high (exceeding limits)
        return dict(
            vel_constraints_scaling_factor=1.0,
            operating_area_offset_from_centre=0.15,
            operating_area_offset_from_table=0.0,
            operating_area_offset_from_goal=0.0,
            z_position_control_tolerance=1.0,
        )


class DefensiveAgentStrategy(AgentStrategy):
    def get_reward_function(self) -> TournamentReward:
        # receive goal (-1.0), cause faul (-1/3)
        return TournamentReward(
            reward_agent_score_goal=0.0,
            reward_opponent_faul=0.0,
            reward_agent_cause_puck_stuck=0.0,
            reward_agent_receive_goal=-1.0,
            reward_agent_faul=-1.0 / 3.0,
        )

    def get_env_kwargs(self) -> Dict[str, Any]:
        # regular-safe
        return dict(
            vel_constraints_scaling_factor=0.45,
            operating_area_offset_from_centre=0.145,
            operating_area_offset_from_table=0.005,
            operating_area_offset_from_goal=0.0075,
            z_position_control_tolerance=0.475,
        )


def strategy_to_str(strategy: AgentStrategy) -> str:
    if isinstance(strategy, BalancedAgentStrategy):
        return "balanced"
    elif isinstance(strategy, AggressiveAgentStrategy):
        return "aggressive"
    elif isinstance(strategy, DefensiveAgentStrategy):
        return "defensive"
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def strategy_from_str(strategy: str) -> AgentStrategy:
    if strategy == "balanced" or strategy == "offensive":
        return BalancedAgentStrategy()
    elif strategy == "aggressive":
        return AggressiveAgentStrategy()
    elif strategy == "defensive":
        return DefensiveAgentStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
