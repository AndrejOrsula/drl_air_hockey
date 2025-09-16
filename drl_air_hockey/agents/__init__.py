from typing import Type


def agent_from_name(name: str) -> Type:
    if name == "spacer":
        from .spacer_agent import SpaceRAgent

        return SpaceRAgent
    if name == "spacer_inference":
        from .spacer_agent_inference import SpaceRAgentInference

        return SpaceRAgentInference
    else:
        raise ValueError(f"Unknown agent name: {name}")
