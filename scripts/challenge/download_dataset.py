#!/usr/bin/env python3

import os
import pathlib
import shlex
import subprocess
import zipfile
from typing import Dict

import obs


def parse_team_info(parse_cript: str) -> Dict[str, str]:
    """
    Parses the team information from the given parse script. The parse script
    should be a bash script that sets the following environment variables:
    - swr_server
    - team_name
    - login_key
    - AK
    - SK
    """

    team_info = {}
    proc = subprocess.Popen(
        shlex.split(
            f"env -i bash -c 'set -a && source {parse_cript} >/dev/null && env'"
        ),
        stdout=subprocess.PIPE,
    )
    for line in proc.stdout:
        kwarg = line.decode().split("=")
        if len(kwarg) != 2:
            continue
        (key, value) = kwarg[0].strip(), kwarg[1].strip()
        if key in ["swr_server", "team_name", "login_key", "AK", "SK"]:
            if value == "":
                raise Exception(f"Could not parse '{key}' from team info.")
            if team_info.get(key, None) is not None:
                raise Exception(f"Duplicate key '{key}' in team info.")
            team_info[key] = value
    return team_info


def main(
    parse_cript: str = os.path.join(os.path.dirname(__file__), "_parse_team_info.bash"),
    download_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets"
    ),
):
    """
    Downloads the latest Dataset from the evaluation if it exists. If the
    latest Dataset already exists in the download folder, it will notify you
    and not download it again.
    """

    # Parse team information
    team_info = parse_team_info(parse_cript=parse_cript)
    swr_server = team_info.get("swr_server", None)
    team_name = team_info.get("team_name", None)
    login_key = team_info.get("login_key", None)
    AK = team_info.get("AK", None)
    SK = team_info.get("SK", None)
    if None in [swr_server, team_name, login_key, AK, SK]:
        raise Exception("Could not parse team info.")
    del team_info
    print("Team information:")
    print(f"  swr_server: {swr_server}")
    print(f"  team_name:  {team_name}")
    print(f"  login_key:  {login_key}")
    print(f"  AK:         {AK}")
    print(f"  SK:         {SK}")

    # Create download directory if it does not exist
    pathlib.Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Create download client
    server = f"https://obs.{swr_server}.myhuaweicloud.eu"
    bucketName = "air-hockey-dataset-eu"
    objectKey = f"data-{team_name}.zip"
    # objectKey = f"friendly_game/data-{team_name}.zip"
    obsClient = obs.ObsClient(access_key_id=AK, secret_access_key=SK, server=server)

    resp = obsClient.getObjectMetadata(bucketName, objectKey)
    if resp["status"] != 200:
        raise Exception("Could not get download object: ", resp["reason"])

    last_modified = resp["header"][3][1][5:19].replace(" ", "-")
    for old_dataset in os.listdir(download_dir):
        if last_modified in old_dataset:
            print("There is no new Dataset available.")
            return

    download_path = os.path.join(download_dir, f"dataset-{last_modified}.zip")
    resp = obsClient.getObject(
        bucketName,
        objectKey,
        downloadPath=download_path,
    )
    if resp["status"] != 200:
        raise Exception("Could not get Object: ", resp["reason"])

    extracted_path = download_path.replace(".zip", "")

    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    print(f"Successfully downloaded dataset to '{extracted_path}'")

    os.remove(download_path)


if __name__ == "__main__":
    main()
