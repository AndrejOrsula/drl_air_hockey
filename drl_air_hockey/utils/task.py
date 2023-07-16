from __future__ import annotations

import enum


@enum.unique
class Task(enum.Enum):
    """
    Task to be performed by the agent.
    """

    R7_HIT = enum.auto()
    R7_DEFEND = enum.auto()
    R7_PREPARE = enum.auto()
    R7_TOURNAMENT = enum.auto()

    def from_env(env_name: str) -> Task:
        """
        Args:
            env_name: The name of the environment.
                        Possible known options = [
                            "7dof-hit",
                            "7dof-defend",
                            "7dof-prepare",
                            "tournament",
                        ]

        Returns:
            The task to be performed by the agent.
        """

        if "7" in env_name:
            if "hit" in env_name:
                return Task.R7_HIT
            elif "defend" in env_name:
                return Task.R7_DEFEND
            elif "prepare" in env_name:
                return Task.R7_PREPARE
        elif "tournament" in env_name:
            return Task.R7_TOURNAMENT
        else:
            raise ValueError(f"Unknown environment name: {env_name}")

    def to_str(self) -> str:
        """
        Returns:
            The name of the environment corresponding to the task.
        """

        if self.is_hit():
            return "7dof-hit"
        elif self.is_defend():
            return "7dof-defend"
        elif self.is_prepare():
            return "7dof-prepare"
        elif self.is_tournament():
            return "tournament"
        else:
            raise ValueError(f"Unknown task: {self}")

    def is_hit(self) -> bool:
        """
        Returns:
            True if the task is a hit task, False otherwise.
        """

        return self == Task.R7_HIT

    def is_defend(self) -> bool:
        """
        Returns:
            True if the task is a defend task, False otherwise.
        """

        return self == Task.R7_DEFEND

    def is_prepare(self) -> bool:
        """
        Returns:
            True if the task is a prepare task, False otherwise.
        """

        return self == Task.R7_PREPARE

    def is_tournament(self) -> bool:
        """
        Returns:
            True if the task is a tournament task, False otherwise.
        """

        return self == Task.R7_TOURNAMENT

    def n_joint(self) -> int:
        """
        Returns:
            The number of joints in the robot.
        """

        return 7

    def is_3dof(self) -> bool:
        """
        Returns:
            True if the task is a 3 DoF task, False otherwise.
        """

        return False

    def is_7dof(self) -> bool:
        """
        Returns:
            True if the task is a 7 DoF task, False otherwise.
        """

        return True
