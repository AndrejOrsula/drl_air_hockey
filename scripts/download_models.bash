#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
REPOSITORY_DIR="$(dirname "${SCRIPT_DIR}")"
MODEL_DIR="${REPOSITORY_DIR}/drl_air_hockey/agents/models"
GDRIVE_FOLDER="XXX"

## Download models
python3 -m pip install -qqq --no-cache-dir gdown
mkdir -p "${MODEL_DIR}"
gdown -q --no-cookies --remaining-ok --folder "${GDRIVE_FOLDER}" -O "${MODEL_DIR}"
