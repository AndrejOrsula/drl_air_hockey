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
    python3 -m pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 && \
    wget -q https://raw.githubusercontent.com/danijar/dreamerv3/main/dreamerv3/configs.yaml -O "$(pip show dreamerv3 | grep Location: | cut -d' ' -f2)/dreamerv3/configs.yaml"

### Fixate versions of Python dependencies
RUN python3 -m pip install --no-cache-dir absl-py==2.0.0 astunparse==1.6.3 cachetools==5.3.1 certifi==2023.7.22 charset-normalizer==3.2.0 chex==0.1.7 cloudpickle==1.6.0 contourpy==1.1.1 crafter==1.8.1 cycler==0.11.0 decorator==5.1.1 dm-tree==0.1.8 dreamerv3==1.5.0 exceptiongroup==1.1.3 filelock==3.12.4 flatbuffers==23.5.26 fonttools==4.42.1 fsspec==2023.4.0 gast==0.4.0 glfw==2.6.2 google-auth==2.23.0 google-auth-oauthlib==1.0.0 google-pasta==0.2.0 grpcio==1.58.0 gym==0.19.0 h5py==3.9.0 idna==3.4 imageio==2.31.3 importlib-metadata==6.8.0 importlib-resources==6.1.0 iniconfig==2.0.0 jax==0.4.13 jaxlib==0.4.13+cuda12.cudnn89 Jinja2==3.1.2 joblib==1.3.2 keras==2.13.1 kiwisolver==1.4.5 libclang==16.0.6 Markdown==3.4.4 markdown-it-py==3.0.0 MarkupSafe==2.1.3 matplotlib==3.7.3 mdurl==0.1.2 ml-dtypes==0.2.0 mpmath==1.3.0 mujoco==2.3.7 mushroom-rl==1.9.2 networkx==3.1 nlopt==2.7.1 numpy==1.24.3 numpy-ml==0.1.2 nvidia-cublas-cu12==12.2.5.6 nvidia-cuda-cupti-cu12==12.2.142 nvidia-cuda-nvcc-cu12==12.2.140 nvidia-cuda-nvrtc-cu12==12.2.140 nvidia-cuda-runtime-cu12==12.2.140 nvidia-cudnn-cu12==8.9.4.25 nvidia-cufft-cu12==11.0.8.103 nvidia-cusolver-cu12==11.5.2.141 nvidia-cusparse-cu12==12.1.2.141 nvidia-nvjitlink-cu12==12.2.140 oauthlib==3.2.2 opencv-python==4.8.0.76 opensimplex==0.4.5 opt-einsum==3.3.0 optax==0.1.7 osqp==0.6.3 packaging==23.1 Pillow==10.0.1 pip==23.2.1 pluggy==1.3.0 protobuf==4.24.3 pyasn1==0.5.0 pyasn1-modules==0.3.0 pygame==2.5.2 Pygments==2.16.1 PyOpenGL==3.1.7 pyparsing==3.1.1 pytest==7.4.2 python-dateutil==2.8.2 pytorch-triton==2.1.0+6e4932cda8 PyYAML==6.0.1 pyzmq==25.1.1 qdldl==0.1.7.post0 requests==2.31.0 requests-oauthlib==1.3.1 rich==13.5.3 rsa==4.9 ruamel.yaml==0.17.32 ruamel.yaml.clib==0.2.7 scikit-learn==1.3.1 scipy==1.10.1 setuptools==45.2.0 six==1.16.0 sympy==1.12 tensorboard==2.13.0 tensorboard-data-server==0.7.1 tensorflow-cpu==2.13.0 tensorflow-estimator==2.13.0 tensorflow-io-gcs-filesystem==0.34.0 tensorflow-probability==0.21.0 termcolor==2.3.0 threadpoolctl==3.2.0 tomli==2.0.1 toolz==0.12.0 tqdm==4.66.1 typing_extensions==4.5.0 urllib3==1.26.16 Werkzeug==2.3.7 wheel==0.34.2 wrapt==1.15.0 zipp==3.17.0 zmq==0.0.0

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
