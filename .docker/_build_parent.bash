#!/usr/bin/env bash

IMAGE_NAME="airhockeychallenge/challenge"

## Determine the directory of air_hockey_challenge
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
CHALLENGE_DIR="$(dirname "${REPOSITORY_DIR}")/_forks/air_hockey_challenge"

## Parse TAG and forward additional build arguments
if [ "${#}" -gt "0" ]; then
    if [[ "${1}" != "-"* ]]; then
        IMAGE_NAME="${IMAGE_NAME}:${1}"
        BUILD_ARGS=${*:2}
    else
        BUILD_ARGS=${*:1}
    fi
fi

## Build the image
DOCKER_BUILD_CMD=(
    docker build
    "${CHALLENGE_DIR}"
    --tag "${IMAGE_NAME}"
    "${BUILD_ARGS}"
)
echo -e "\033[1;30m${DOCKER_BUILD_CMD[*]}\033[0m" | xargs
# shellcheck disable=SC2048
exec ${DOCKER_BUILD_CMD[*]}
