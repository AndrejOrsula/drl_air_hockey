ARG PARENT_IMAGE=air_hockey_challenge
ARG PARENT_IMAGE_TAG=latest
FROM ${PARENT_IMAGE}:${PARENT_IMAGE_TAG}

### Use bash as the default shell
SHELL ["/bin/bash", "-c"]

ARG DEV=true
ARG DREAMER_DEV=true
ARG DREAMER_PATH="/src/dreamerv3"
ARG DREAMER_REMOTE="https://github.com/AndrejOrsula/dreamerv3.git"
ARG DREAMER_BRANCH="main"
ARG DREAMER_COMMIT_SHA="4049794d4135e41c691f18da38a9af7541b01553" # 2025-07-16
RUN if [[ "${DEV,,}" = true && "${DREAMER_DEV,,}" = true ]]; then \
    git clone "${DREAMER_REMOTE}" "${DREAMER_PATH}" --branch "${DREAMER_BRANCH}" && \
    git -C "${DREAMER_PATH}" reset --hard "${DREAMER_COMMIT_SHA}" && \
    python3 -m pip install --no-input --no-cache-dir --editable "${DREAMER_PATH}" ; \
    fi

### Copy the source and install in editable mode
COPY . "/src/drl_air_hockey"
RUN python3 -m pip install --no-input --no-cache-dir --editable "/src/drl_air_hockey"

### Set the working directory
WORKDIR "/src"

## Configure argcomplete
RUN echo "source /etc/bash_completion.d/drl_air_hockey" >> "/etc/bash.bashrc" && \
    register-python-argcomplete "drl_air_hockey" > "/etc/bash_completion.d/drl_air_hockey"

### Define the default command
CMD ["python3", "-O", "/src/2025-challenge/run.py"]
