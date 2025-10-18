import numpy as np


class TournamentReward:
    MAX_TIME_UNTIL_PENALTY_S: float = 15.0

    def __init__(
        self,
        reward_agent_score_goal: float = 1.0,
        reward_agent_receive_goal: float = -1.0,
        reward_opponent_fault: float = 0.0,
        reward_agent_fault: float = -1.0 / 3.0,
        reward_stalemate: float = 0.0,
        goal_reward_decay_factor: float = 0.0,
        goal_penalty_scaling_factor: float = 1.0,
        stuck_puck_pos_threshold: float = 0.2,
        stuck_puck_vel_threshold: float = 0.025,
        penalty_fault_threshold_ratio: float = 0.8,
    ):
        self._reward_agent_score_goal = reward_agent_score_goal
        self._reward_agent_receive_goal = reward_agent_receive_goal
        self._reward_opponent_fault = reward_opponent_fault
        self._reward_agent_fault = reward_agent_fault
        self._reward_stalemate = reward_stalemate

        self._goal_reward_decay_factor = goal_reward_decay_factor
        self._goal_penalty_scaling_factor = goal_penalty_scaling_factor

        self._stuck_puck_pos_threshold = stuck_puck_pos_threshold
        self._stuck_puck_vel_threshold = stuck_puck_vel_threshold
        self._penalty_threshold = (
            penalty_fault_threshold_ratio * self.MAX_TIME_UNTIL_PENALTY_S
        )

        self.penalty_timer = 0.0
        self.penalty_side = None
        self.time_penalty_range = [0, self.MAX_TIME_UNTIL_PENALTY_S]

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0.0
        puck_pos, puck_vel = mdp.get_puck(next_state)

        puck_pos_x_sign = np.sign(puck_pos[0])
        if self.penalty_side is None:
            self.penalty_side = puck_pos_x_sign if puck_pos_x_sign != 0 else 1

        if puck_pos_x_sign != 0:
            if puck_pos_x_sign == self.penalty_side:
                self.penalty_timer += mdp.env_info["dt"]
            else:
                self.penalty_side *= -1
                self.penalty_timer = 0.0

        if absorbing or mdp._data.time < mdp.env_info["dt"] * 2:
            table_length = mdp.env_info["table"]["length"]
            is_goal = (
                np.abs(puck_pos[1]) - mdp.env_info["table"]["goal_width"] / 2
            ) <= 0
            if is_goal and puck_pos[0] > table_length / 2:
                reward_range = [
                    self._reward_agent_score_goal,
                    self._reward_agent_score_goal * self._goal_reward_decay_factor,
                ]
                r = np.interp(self.penalty_timer, self.time_penalty_range, reward_range)

            elif is_goal and puck_pos[0] < -table_length / 2:
                extra_penalty_range = [
                    0.0,
                    self._reward_agent_receive_goal * self._goal_penalty_scaling_factor,
                ]
                extra_penalty = np.interp(
                    self.penalty_timer, self.time_penalty_range, extra_penalty_range
                )
                r = self._reward_agent_receive_goal + extra_penalty

            elif (
                self.penalty_timer > self._penalty_threshold
                and np.abs(puck_pos[0]) >= self._stuck_puck_pos_threshold
            ):
                if self.penalty_side == -1:
                    r = self._reward_agent_fault
                elif self.penalty_side == 1:
                    r = self._reward_opponent_fault

            elif (
                np.abs(puck_pos[0]) < self._stuck_puck_pos_threshold
                and np.abs(puck_vel[0]) < self._stuck_puck_vel_threshold
            ):
                r = self._reward_stalemate

            else:
                r = self._reward_stalemate

            self.penalty_timer = 0.0
            self.penalty_side = None

        return r


def reward_from_name(name: str):
    match name:
        case "tournament":
            return TournamentReward()
        case _:
            raise ValueError(f"Unknown reward function name: {name}")
