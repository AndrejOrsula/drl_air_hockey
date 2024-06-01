# Air Hockey with Deep Reinforcement Learning

<p align="left">
  <a href="https://github.com/AndrejOrsula/drl_air_hockey/actions/workflows/docker.yml"> <img alt="Docker" src="https://github.com/AndrejOrsula/drl_air_hockey/actions/workflows/docker.yml/badge.svg"></a>
</p>

This is a participant repository for the [Robot Air Hockey Challenge 2023](https://air-hockey-challenge.robot-learning.net).

https://github.com/AndrejOrsula/drl_air_hockey/assets/22929099/68b39c9c-2d18-4eae-99e4-58670fb29e2c

## Overview

The implemented approach focuses on applying a model-based deep reinforcement learning algorithm [DreamerV3](https://danijar.com/project/dreamerv3) to acquire a policy capable of playing air hockey with continuous observations and actions.

- Additional details about the approach are presented in the report [here](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/PuzeLiu/AirHockeyChallenge_SpaceR.pdf).
- Models of pre-trained agents can be downloaded from [here](https://drive.google.com/drive/folders/1bfj0VCm7wbGKBh5i_-Oso3XCcJbNZ_8E).

## Instructions

### <img src="https://www.svgrepo.com/show/269868/lab.svg" width="16" height="16"> Quick Test

As a quick test, you can try evaluating a pre-trained agent in a self-play mode by running [`.docker/run.bash`](.docker/run.bash) directly via [`curl`](https://curl.se) as shown below.

```bash
curl -sSfL "https://raw.githubusercontent.com/AndrejOrsula/drl_air_hockey/main/.docker/run.bash" | DOCKER_RUN_OPTS="--rm" bash -s -- drl_air_hockey/scripts/eval_dreamerv3.py -r
```

### <img src="https://www.svgrepo.com/show/354238/python.svg" width="16" height="16"> Python

#### Installation

Install [`air_hockey_challenge`](https://github.com/AndrejOrsula/air_hockey_challenge) and `drl_air_hockey` Python modules with [`pip`](https://pypi.org/project/pip).

```bash
pip3 install git+https://github.com/AndrejOrsula/air_hockey_challenge.git
pip3 install git+https://github.com/AndrejOrsula/drl_air_hockey.git
```

> \[!WARNING\]
> Not all dependencies have their versions pinned, e.g. transitive dependencies of `dreamerv3`. Therefore, the functionality of this repository may be affected in future builds. Consider using the pre-built Docker image for a stable environment.

#### Training and Evaluation

To train a new agent, you can run the included [`train_dreamerv3.py`](scripts/train_dreamerv3.py) script.

```bash
scripts/train_dreamerv3.py
```

To configure the training process, consider modifying any of these files directly:

- [`config.py`](drl_air_hockey/utils/config.py) for algorithm-specific parameters
- [`rewards.py`](drl_air_hockey/utils/rewards.py) and [`tournament_agent_strategies.py`](drl_air_hockey/utils/tournament_agent_strategies.py) for the desired behavior of the agent
- [`agents`](drl_air_hockey/agents/) for additional changes to the observation and action spaces of the agent

Once you are satisfied with the training progress, you can evaluate the agent by adjusting the included [`eval_dreamerv3.py`](scripts/eval_dreamerv3.py) script.

```bash
scripts/eval_dreamerv3.py
```

<details>
<summary><h3><a href="#-docker"><img src="https://www.svgrepo.com/show/448221/docker.svg" width="16" height="16"></a> Docker</h3></summary>

> To install [Docker](https://docs.docker.com/get-docker) on your system, you can run [`.docker/host/install_docker.bash`](.docker/host/install_docker.bash) to configure Docker with NVIDIA GPU support.
>
> ```bash
> .docker/host/install_docker.bash
> ```

#### Build Image

To build a new Docker image from [`Dockerfile`](Dockerfile), you can run [`.docker/build.bash`](.docker/build.bash) as shown below. Note that only the pre-built Docker image already contains [models of pre-trained agents](https://drive.google.com/drive/folders/1bfj0VCm7wbGKBh5i_-Oso3XCcJbNZ_8E).

```bash
.docker/build.bash ${TAG:-latest} ${BUILD_ARGS}
```

#### Run Container

To run the Docker container, you can use [`.docker/run.bash`](.docker/run.bash) as shown below.

```bash
.docker/run.bash ${TAG:-latest} ${CMD:-bash}
```

#### Run Dev Container

To run the Docker container in a development mode (source code mounted as a volume), you can use [`.docker/dev.bash`](.docker/dev.bash) as shown below.

```bash
.docker/dev.bash ${TAG:-latest} ${CMD:-bash}
```

As an alternative, VS Code users familiar with [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) can modify the included [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json) to their needs. For convenience, [`.devcontainer/open.bash`](.devcontainer/open.bash) script is available to open this repository as a Dev Container in VS Code.

```bash
.devcontainer/open.bash
```

#### Join Container

To join a running Docker container from another terminal, you can use [`.docker/join.bash`](.docker/join.bash) as shown below.

```bash
.docker/join.bash ${CMD:-bash}
```

</details>

## Citation

```bibtex
@article{orsula2023learning,
  title   = {{Learning to Play Air Hockey with Model-Based Deep Reinforcement Learning}},
  author  = {Andrej Orsula},
  journal = {NeurIPS 2023 --- The Robot Air Hockey Challenge},
  year    = {2023},
}
```
