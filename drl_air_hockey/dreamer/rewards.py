import numpy as np


class TournamentReward:
    MAX_TIME_UNTIL_PENALTY_S: float = 7.0

    def __init__(
        self,
        reward_agent_score_goal: float = 1.0,
        reward_agent_receive_goal: float = -1.0,
        reward_opponent_faul: float = 0.0,
        reward_agent_faul: float = -1.0 / 3.0,
        reward_agent_cause_puck_stuck: float = 0.0,
    ):
        self._reward_agent_score_goal = reward_agent_score_goal
        self._reward_agent_receive_goal = reward_agent_receive_goal
        self._reward_opponent_faul = reward_opponent_faul
        self._reward_agent_faul = reward_agent_faul
        self._reward_agent_cause_puck_stuck = reward_agent_cause_puck_stuck
        self._penalty_threshold = 0.8 * self.MAX_TIME_UNTIL_PENALTY_S

        self.penalty_timer = 0.0
        self.penalty_side = None
        self.time_penalty_range = [0, self.MAX_TIME_UNTIL_PENALTY_S]

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0.0
        puck_pos, puck_vel = mdp.get_puck(next_state)

        ## Penalty checking (timer update)
        # Determine which side the puck is on on the first step
        if self.penalty_side is None:
            self.penalty_side = np.sign(puck_pos[0])

        if np.sign(puck_pos[0]) == self.penalty_side:
            # If the puck is on the same side as the penalty side, increment the penalty timer
            self.penalty_timer += mdp.env_info["dt"]
        else:
            # Otherwise, reset the penalty timer and change the penalty side
            self.penalty_side *= -1
            self.penalty_timer = 0.0
        ## ~ Penalty checking (timer update)

        if absorbing or mdp._data.time < mdp.env_info["dt"] * 2:
            ## Penalty checking
            # If the penalty timer is greater than X seconds and the puck is not in the middle, give reward accordingly
            if (
                self.penalty_timer > self._penalty_threshold
                and np.abs(puck_pos[0]) >= 0.15
            ):
                if self.penalty_side == -1:
                    r = self._reward_agent_faul
                elif self.penalty_side == 1:
                    r = self._reward_opponent_faul
                else:
                    raise ValueError(
                        f"Penalty side should be either -1 or 1, but got {self.penalty_side}"
                    )
            ## ~ Penalty checking

            ## Puck stuck in the middle
            if np.abs(puck_pos[0]) < 0.15 and np.abs(puck_vel[0]) < 0.025:
                r = self._reward_agent_cause_puck_stuck
            ## ~ Puck stuck in the middle

            ## Goal checking
            if (np.abs(puck_pos[1]) - mdp.env_info["table"]["goal_width"] / 2) <= 0:
                if puck_pos[0] > mdp.env_info["table"]["length"] / 2:
                    reward_range = [
                        self._reward_agent_score_goal,
                        self._reward_agent_score_goal * 0.25,
                    ]  # At max time, reward is 25% of original
                    r = np.interp(
                        self.penalty_timer, self.time_penalty_range, reward_range
                    )
                elif puck_pos[0] < -mdp.env_info["table"]["length"] / 2:
                    extra_penalty_range = [
                        0.0,
                        self._reward_agent_receive_goal,
                    ]  # At max time, penalty is doubled
                    extra_penalty = np.interp(
                        self.penalty_timer, self.time_penalty_range, extra_penalty_range
                    )
                    r = self._reward_agent_receive_goal + extra_penalty
            ## ~ Goal checking

            # Reset the penalty timer and side (it is the end of episode)
            self.penalty_timer = 0.0
            self.penalty_side = None

        return r


def reward_from_name(name: str):
    match name:
        case "tournament":
            return TournamentReward()
        case _:
            raise ValueError(f"Unknown reward function name: {name}")
