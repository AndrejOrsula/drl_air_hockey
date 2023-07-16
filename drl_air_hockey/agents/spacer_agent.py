from typing import Any, Dict, Optional

from air_hockey_challenge.framework import AgentBase


class SpaceRAgent(AgentBase):
    def __init__(
        self,
        env_info: Dict[str, Any],
        agent_id: int = 1,
        interpolation_order: Optional[int] = -1,
        **kwargs,
    ):
        raise NotImplementedError
