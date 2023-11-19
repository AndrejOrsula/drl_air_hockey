ARG PARENT_IMAGE=andrejorsula/air_hockey_challenge
ARG PARENT_IMAGE_TAG=latest
FROM ${PARENT_IMAGE}:${PARENT_IMAGE_TAG}

### Use bash as the default shell
SHELL ["/bin/bash", "-c"]

### Install dependencies
RUN python3 -m pip install --no-cache-dir setuptools==65.5.0 pip==21.3.1 && \
    python3 -m pip install --no-cache-dir "dreamerv3 @ git+https://github.com/AndrejOrsula/dreamerv3.git" && \
    python3 -m pip install --no-cache-dir --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 && \
    wget -q https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/configs.yaml -O "$(pip show dreamerv3 | grep Location: | cut -d' ' -f2)/dreamerv3/configs.yaml"

### Enable GUI
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglvnd-dev \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

### Copy the source and install in editable mode
COPY . "/src/drl_air_hockey"
RUN python3 -m pip install --no-cache-dir -e "/src/drl_air_hockey"

### Set the working directory
WORKDIR "/src"

### Define the default command
CMD ["python3", "-O", "/src/2023-challenge/run.py"]
