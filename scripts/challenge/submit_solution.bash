#!/usr/bin/env bash
# shellcheck disable=SC2154
set -e

## Parse arguments
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_parse_team_info.bash"

## Determine the image name
IMAGE_NAME="swr.${swr_server}.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-${team_name}"

## Build the submission image
REPOSITORY_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
DOT_DOCKER_DIR="${REPOSITORY_DIR}/.docker"
# Build parent image
"${DOT_DOCKER_DIR}/build_parent.bash" --target eval
# Build submission image
"${DOT_DOCKER_DIR}/build.bash" --tag "${IMAGE_NAME}"

## Login in to the cloud
docker login -u "${swr_server}@${AK}" -p "${login_key}" "swr.${swr_server}.myhuaweicloud.eu"

## Push to the cloud for evaluation
docker push "${IMAGE_NAME}"
