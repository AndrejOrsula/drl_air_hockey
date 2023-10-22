#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import yaml
from air_hockey_agent.agent_builder import build_agent
from air_hockey_challenge.framework.evaluate_tournament import _run_tournament
from air_hockey_challenge.utils.tournament_agent_wrapper import (
    SimpleTournamentAgentWrapper,
)
from baseline.baseline_agent.baseline_agent import BaselineAgent

from drl_air_hockey.utils.tournament_agent_strategies import (
    AggressiveAgentStrategy,
    DefensiveAgentStrategy,
    OffensiveAgentStrategy,
    SneakyAgentStrategy,
)

# Default parameters
RENDER: bool = False
N_ENVIRONMENTS: int = 1
STEPS_PER_GAME: int = 45000


def main(argv=None):
    agent_config_1_path = Path(__file__).parent.joinpath(
        "/src/2023-challenge/air_hockey_agent/agent_config.yml"
    )

    agent_config_2_path = agent_config_1_path

    with open(agent_config_1_path) as stream:
        agent_config_1 = yaml.safe_load(stream)
    with open(agent_config_2_path) as stream:
        agent_config_2 = yaml.safe_load(stream)

    agent_config_1.update(get_args())

    # agent_config_1.update({"initial_strategy": "offensive"})
    # agent_config_2.update({"initial_strategy": "offensive"})

    agent_config_1.update(
        {
            "scheme": 7,
            "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_aggressive.ckpt",
            **AggressiveAgentStrategy().get_env_kwargs(),
        }
        # {
        #     "scheme": 7,
        #     "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_offensive.ckpt",
        #     **OffensiveAgentStrategy().get_env_kwargs(),
        # }
        # {
        #     "scheme": 7,
        #     "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_sneaky.ckpt",
        #     **SneakyAgentStrategy().get_env_kwargs(),
        # }
        # {
        #     "scheme": 7,
        #     "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_defensive.ckpt",
        #     **DefensiveAgentStrategy().get_env_kwargs(),
        # }
    )
    agent_config_2.update(
        {
            "scheme": 7,
            "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_aggressive.ckpt",
            **AggressiveAgentStrategy().get_env_kwargs(),
        }
        # {
        #     "scheme": 7,
        #     "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_offensive.ckpt",
        #     **OffensiveAgentStrategy().get_env_kwargs(),
        # }
        # {
        #     "scheme": 7,
        #     "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_sneaky.ckpt",
        #     **SneakyAgentStrategy().get_env_kwargs(),
        # }
        # {
        #     "scheme": 7,
        #     "load_model_path": "/src/drl_air_hockey/drl_air_hockey/agents/models/tournament_defensive.ckpt",
        #     **DefensiveAgentStrategy().get_env_kwargs(),
        # }
    )

    run_tournament(
        build_agent_1=build_agent,
        build_agent_2=build_agent,
        agent_2_kwargs=agent_config_2,
        **agent_config_1,
    )


def run_tournament(
    build_agent_1,
    log_dir,
    build_agent_2=None,
    agent_2_kwargs={},
    steps_per_game=45000,
    n_episodes=1,
    n_cores=-1,
    seed=None,
    generate_score=None,
    quiet=True,
    render=False,
    save_away_data=False,
    **kwargs,
):
    """
    Run tournament games between two agents which are build by build_agent_1 and build_agent_2. The resulting Dataset
    and constraint stats will be written to folder specified in log_dir. If save_away_data is True the data for the
    second Agent is also saved. The amount of games is specified by n_episodes. The resulting Dataset can be replayed by
    the replay_dataset function in air_hockey_challenge/utils/replay_dataset.py. This function is intended to be called
    by run.py.

    For compatibility with run.py the kwargs for agent_1 are passed via **kwargs and the kwargs for agent_2 are passed
    via agent_2_kwargs.

    Args:
        build_agent_1 ((mdp_info, **kwargs) -> Agent): Function that returns agent_1 given the env_info and **kwargs.
        log_dir (str): The path to the log directory.
        build_agent_2 ((mdp_info, **kwargs) -> Agent, None): Function that returns agent_2 given the env_info and
            **agent_2_kwargs. If None the BaselineAgent is used.
        agent_2_kwargs (dict, {}): The arguments for the second agent.
        steps_per_game (int, 45000): The amount of steps a single game will last
        n_episodes (int, 1): The number of games which are played
        n_cores (int, -1): Number of parallel cores which are used for the computation. -1 Uses all cores.
            When using 1 core the program will not be parallelized (good for debugging)
        seed (int, None): Desired seed to be used. The seed will be set for numpy and torch.
        generate_score(str, None): If set to "phase-3" a score and summary is generated. The achieved score against
            Baseline Agent is what the leaderboard is based on.
        quiet (bool, True): set to True to disable tqdm progress bars
        render (bool, False): set to True to spawn a viewer that renders the simulation
        save_away_data(bool, False): Set True to save the data and generate a score for the second agent.
        kwargs (any): Argument passed to the agent_1 init
    """

    def agent_builder(
        mdp, i, build_agent_1, build_agent_2, agent_1_kwargs, agent_2_kwargs
    ):
        agent_1_kwargs.update({"agent_id": 1})
        agent_2_kwargs.update({"agent_id": 2})

        if build_agent_1 is None:
            agent_1 = BaselineAgent(mdp.env_info, 2)
        else:
            agent_1 = build_agent_1(mdp.env_info, **agent_1_kwargs)
        if build_agent_2 is None:
            agent_2 = BaselineAgent(mdp.env_info, 2)
        else:
            agent_2 = build_agent_2(mdp.env_info, **agent_2_kwargs)

        return SimpleTournamentAgentWrapper(mdp.env_info, agent_1, agent_2)

    interpolation_order = [3, 3]
    if "interpolation_order" in kwargs.keys():
        interpolation_order[0] = kwargs["interpolation_order"]

    if "interpolation_order" in agent_2_kwargs.keys():
        interpolation_order[1] = agent_2_kwargs["interpolation_order"]

    _run_tournament(
        log_dir,
        agent_builder,
        "Home",
        "Away",
        steps_per_game,
        n_episodes,
        n_cores,
        seed,
        generate_score,
        quiet,
        render,
        save_away_data,
        tuple(interpolation_order),
        build_agent_1=build_agent_1,
        build_agent_2=build_agent_2,
        agent_1_kwargs=kwargs,
        agent_2_kwargs=agent_2_kwargs,
    )


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group("override parameters")

    arg_test.add_argument(
        "-n",
        "--n_episodes",
        type=int,
        help="Number of CPU cores (environments) used for evaluation.",
        default=N_ENVIRONMENTS,
    )

    arg_test.add_argument(
        "-s",
        "--steps_per_game",
        type=int,
        help="Number of CPU cores (environments) used for evaluation.",
        default=STEPS_PER_GAME,
    )

    arg_test.add_argument(
        "-r",
        "--render",
        action="store_true",
        help="If set renders the environment",
        default=RENDER,
    )

    args = vars(parser.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    return args


if __name__ == "__main__":
    main()
