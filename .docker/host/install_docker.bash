#!/usr/bin/env bash

## Install wget if not installed
if [[ ! -x "$(command -v wget)" ]]; then
    echo "Installing wget...."
    sudo apt-get update && sudo apt-get install -y wget
fi

## Install Docker if not installed
if [[ ! -x "$(command -v docker)" ]]; then
    echo "Installing Docker..."
    wget https://get.docker.com -O - -o /dev/null | sh &&
    sudo systemctl --now enable docker
else
    echo "Docker is already installed."
fi

## Install support for NVIDIA if an NVIDIA GPU is detected (install Container Toolkit or Docker depending on Docker version)
check_nvidia_gpu() {
    if ! lshw -C display 2>/dev/null | grep -qi "vendor.*nvidia"; then
        return 1 # NVIDIA GPU is not present
    elif [[ ! -x "$(command -v nvidia-smi)" ]]; then
        return 1 # NVIDIA GPU is present but nvidia-utils not installed
    elif ! nvidia-smi -L &>/dev/null; then
        return 1 # NVIDIA GPU is present but is not working properly
    else
        return 0 # NVIDIA GPU is present and appears to be working
    fi
}
if check_nvidia_gpu; then
    if dpkg --compare-versions "$(sudo docker version --format '{{.Server.Version}}')" gt "19.3"; then
        # With Docker 19.03, nvidia-docker2 is deprecated since NVIDIA GPUs are natively supported as devices in the Docker runtime
        if apt -qq list nvidia-container-toolkit 2>/dev/null | grep -q "installed"; then
            echo "NVIDIA Container Toolkit is already installed."
        else
            echo "Installing NVIDIA Container Toolkit"
            wget https://nvidia.github.io/nvidia-docker/gpgkey -O - -o /dev/null | sudo apt-key add - 2>/dev/null && wget "https://nvidia.github.io/nvidia-docker/$(source /etc/os-release && echo "${ID}${VERSION_ID}")/nvidia-docker.list" -O - -o /dev/null | sed "s#deb https://#deb [arch=$(dpkg --print-architecture)] https://#g" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list >/dev/null
            sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        fi
    else
        if apt -qq list nvidia-docker2 2>/dev/null | grep -q "installed"; then
            echo "NVIDIA Docker (Docker 19.03 or older) is already installed."
        else
            echo "Installing NVIDIA Docker (Docker 19.03 or older)"
            wget https://nvidia.github.io/nvidia-docker/gpgkey -O - -o /dev/null | sudo apt-key add - 2>/dev/null && wget "https://nvidia.github.io/nvidia-docker/$(source /etc/os-release && echo "${ID}${VERSION_ID}")/nvidia-docker.list" -O - -o /dev/null | sed "s#deb https://#deb [arch=$(dpkg --print-architecture)] https://#g" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list >/dev/null
            sudo apt-get update && sudo apt-get install -y nvidia-docker2
        fi
    fi
    sudo systemctl restart docker
fi

## (Optional) Add user to docker group
if [[ $(grep /etc/group -e "docker") != *"${USER}"* ]]; then
    [ -z "${PS1}" ] && read -erp "Do you want to add user ${USER} to the docker group? [Y/n]: " ADD_USER_TO_DOCKER_GROUP
    if [[ "${ADD_USER_TO_DOCKER_GROUP,,}" =~ (y|yes) && ! "${ADD_USER_TO_DOCKER_GROUP,,}" =~ (n|no) ]]; then
        sudo groupadd -f docker && sudo usermod -aG docker "${USER}" && echo -e "User ${USER} was added to the docker group.\nPlease relog or execute the following command for changes to take affect.\n\tnewgrp docker"
    fi
else
    echo "User ${USER} is already in the docker group."
fi
