#!/usr/bin/env bash

function parse_yaml {
    local fs s w prefix
    prefix=$2
    s='[[:space:]]*'
    w='[a-zA-Z0-9_]*'
    fs=$(echo @ | tr @ '\034')
    # shellcheck disable=all
    sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" $1 |
    awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
    }'
}

swr_server="${swr_server:-"unknown"}"
team_name="${team_name:-"unknown"}"
login_key="${login_key:-"unknown"}"
AK="${AK:-"unknown"}"
SK="${SK:-"unknown"}"
if [[ "${swr_server}" == "unknown" || "${team_name}" == "unknown" || "${login_key}" == "unknown" || "${AK}" == "unknown" || "${SK}" == "unknown" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" &>/dev/null && pwd)"
    REPOSITORY_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
    AIR_HOCKEY_CHALLENGE_DIR="$(dirname "${REPOSITORY_DIR}")/_forks/air_hockey_challenge"
    TEAM_INFO_PATH="${TEAM_INFO_PATH:-"${AIR_HOCKEY_CHALLENGE_DIR}/air_hockey_agent/team_info.yml"}"
    if [[ -f "${TEAM_INFO_PATH}" ]]; then
        echo "Using team information from '${TEAM_INFO_PATH}'."
        eval "$(parse_yaml "${TEAM_INFO_PATH}")"
    else
        echo >&2 "Cannot parse team information because file '${TEAM_INFO_PATH}' does not exist."
        exit 1
    fi
else
    echo "Using environment variables for team information."
fi

typeset -l swr_server
typeset -l team_name
swr_server=${swr_server// /-}
team_name=${team_name// /-}

echo "Team information:"
echo "  swr_server: ${swr_server}"
echo "  team_name:  ${team_name}"
echo "  login_key:  ${login_key}"
echo "  AK:         ${AK}"
echo "  SK:         ${SK}"
