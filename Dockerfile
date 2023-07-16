ARG PARENT_IMAGE=airhockeychallenge/challenge
ARG PARENT_IMAGE_TAG=latest
FROM ${PARENT_IMAGE}:${PARENT_IMAGE_TAG}

### Use bash as the default shell
SHELL ["/bin/bash", "-c"]

### Configure the workspace
ARG WORKSPACE="/src"
ENV WORKSPACE="${WORKSPACE}"

### Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir "dreamerv3 @ git+https://github.com/AndrejOrsula/dreamerv3.git" && \
    python3 -m pip install --no-cache-dir --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    wget -q https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/configs.yaml -O "$(pip show dreamerv3 | grep Location: | cut -d' ' -f2)/dreamerv3/configs.yaml"

### Copy the source
COPY . "${WORKSPACE}/drl_air_hockey"

### Install the packages in editable mode
RUN pip install --no-cache-dir \
    "${WORKSPACE}/2023-challenge" \
    -e "${WORKSPACE}/drl_air_hockey"

### Set the working directory
WORKDIR "${WORKSPACE}"

### Define the default command
CMD ["python", "-O", "/src/2023-challenge/run.py"]
