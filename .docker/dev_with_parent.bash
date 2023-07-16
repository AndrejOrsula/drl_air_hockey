#!/usr/bin/env bash

## Determine the host directory to be mounted as a development volume
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
DEV_VOLUME_HOST_DIR="${DEV_VOLUME_HOST_DIR:-"${REPOSITORY_DIR}"}"
DEV_VOLUME_HOST_DIR_CHALLENGE="${DEV_VOLUME_HOST_DIR_CHALLENGE:-"$(dirname "${REPOSITORY_DIR}")/_forks/air_hockey_challenge"}"

## Determine the docker directory where the development volume will be mounted
DEV_VOLUME_DOCKER_DIR="${DEV_VOLUME_DOCKER_DIR:-"/src/drl_air_hockey"}"
DEV_VOLUME_DOCKER_DIR_CHALLENGE="${DEV_VOLUME_DOCKER_DIR_CHALLENGE:-"/src/2023-challenge"}"

## Run the docker container with the development volume mounted
echo -e "\033[2;37mDevelopment volume: ${DEV_VOLUME_HOST_DIR} -> ${DEV_VOLUME_DOCKER_DIR}\033[0m" | xargs
echo -e "\033[2;37mDevelopment volume: ${DEV_VOLUME_HOST_DIR_CHALLENGE} -> ${DEV_VOLUME_DOCKER_DIR_CHALLENGE}\033[0m" | xargs
exec "${SCRIPT_DIR}/run.bash" -v "${DEV_VOLUME_HOST_DIR}:${DEV_VOLUME_DOCKER_DIR}" -v "${DEV_VOLUME_HOST_DIR_CHALLENGE}:${DEV_VOLUME_DOCKER_DIR_CHALLENGE}" "${@}"
