ARG PARENT_IMAGE=airhockeychallenge/challenge
ARG PARENT_IMAGE_TAG=latest
FROM ${PARENT_IMAGE}:${PARENT_IMAGE_TAG}

### Use bash as the default shell
SHELL ["/bin/bash", "-c"]

### Configure the workspace
ARG WORKSPACE="/src"
ENV WORKSPACE="${WORKSPACE}"

### Re-build python with optimizations, if requested
ARG OPTIMIZE_PYTHON=false
ARG PYTHON_VERSION=3.8.17
ENV PYTHON_SRC_DIR="${WORKSPACE}/python${PYTHON_VERSION}"
RUN if [[ "${OPTIMIZE_PYTHON,,}" = true ]]; then \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    libgdm-dev \
    libncurses5-dev \
    libpcap-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libtk8.6 \
    wget && \
    rm -rf /var/lib/apt/lists/* && \
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz" && \
    mkdir -p "${PYTHON_SRC_DIR}" && \
    tar xf "Python-${PYTHON_VERSION}.tar.xz" -C "${PYTHON_SRC_DIR}" --strip-components=1 && \
    rm "Python-${PYTHON_VERSION}.tar.xz" && \
    cd "${PYTHON_SRC_DIR}" && \
    ./configure --enable-optimizations --with-lto --prefix="/usr" && \
    make -j "$(nproc)" && \
    make install ; \
    fi

### Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir "dreamerv3 @ git+https://github.com/AndrejOrsula/dreamerv3.git" && \
    python3 -m pip install --no-cache-dir --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    wget -q https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/configs.yaml -O "$(pip show dreamerv3 | grep Location: | cut -d' ' -f2)/dreamerv3/configs.yaml"

RUN python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

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
